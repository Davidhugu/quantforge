"""Record live market data as the merged capture the readers already parse.

`readers.read_events` documents a tier it does not yet have a producer for: a
*merged capture*, book updates and prints interleaved in one time-ordered
stream, at native frequency. That is what this module writes, and it is the
tier that supports a real reachability check on fills -- a `ReplayMarket` quote
can be tested against the prints that actually happened, instead of against a
Poisson intensity the harness invented.

Two decisions here are load-bearing, and both are about what a reconnect costs.

**Partial book depth, not the diff stream.** The obvious subscription is
`@depth`, which sends incremental changes and requires the client to maintain a
local book. That book is lost on every disconnect, and no venue offers a
backfill, so the first frame after a reconnect is a book reconstructed from
nothing -- which is exactly the frame the quoting layer is about to act on.
`@depthN` sends absolute top-N snapshots instead, so the book is correct the
instant the stream opens and a reconnect is self-healing. The cost is bandwidth
and duplicate snapshots, which is the right trade for a capture.

**Gaps are recorded, not smoothed over.** Binance sequences trades, so a jump in
trade id is a provable hole; a quiet second is not. The writer refuses to
invent either. The capture stays strictly to the canonical schema, and every
anomaly goes to a sidecar `.meta.jsonl` next to it, so the market data and the
diagnosis of it never get mixed into one file that a reader has to special-case.

The writer is synchronous and has no network dependency, so the whole module is
testable offline; only `BinanceSource` touches the internet, and only
`record()` is required to.
"""
from __future__ import annotations

import asyncio
import importlib
import json
import math
import os
import random
import time

from .schema import ValidationError

#: Binance spot combined-stream endpoint. Depth snapshots at 100ms, plus prints.
SPOT_WS = "wss://stream.binance.com:9443/stream"
#: USD-margined futures, for the correlated-instrument lead the signal layer reads.
FUTURES_WS = "wss://fstream.binance.com/stream"

#: Row of the canonical schema. Book columns and trade columns are mutually
#: exclusive by NaN, exactly as `fixtures.synth_session` builds them, so a
#: recorder and a fixture exercise the same padding path in `resample`.
_BOOK_KEYS = ("ts_ms", "mid", "bid", "ask", "bid_qty", "ask_qty", "futures",
              "px", "qty", "side")
_TRADE_KEYS = ("ts_ms", "mid", "bid", "ask", "bid_qty", "ask_qty", "futures",
               "px", "qty", "side")


def _num(v):
    """JSON has no NaN, and `json.dumps` will happily emit a bare `NaN` token
    that nothing but Python can read back.

    `null` is standard and `_from_jsonl` turns it back into NaN via a float cast,
    so the capture stays readable by `jq`, pandas and anything else while still
    round-tripping through the existing reader. Non-finite *and* absent are the
    same thing in this schema -- both mean "this column does not apply to this
    row" -- so they are encoded identically on purpose.
    """
    if v is None:
        return None
    f = float(v)
    return f if math.isfinite(f) else None


class CaptureWriter:
    """Append-only JSONL merged capture, with rotation and a diagnostic sidecar.

    Flushes every `flush_rows` rows rather than every row: a partial-book
    snapshot stream runs at 10/s per symbol and flushing each one costs more in
    syscalls than the whole rest of the pipeline. The cost of batching is that a
    hard kill loses up to one buffer, which is why the buffer is small and the
    writer is not the place to be clever.
    """

    def __init__(self, path: str, venue: str = "unknown", rotate_bytes: int = 0,
                 flush_rows: int = 512, flush_bytes: int = 1 << 18,
                 futures_stale_ms: int = 5_000, futures_symbol: str = ""):
        if flush_rows < 1:
            raise ValidationError(f"flush_rows must be >= 1, got {flush_rows}")
        if rotate_bytes < 0:
            raise ValidationError(f"rotate_bytes must be >= 0, got {rotate_bytes}")
        if futures_stale_ms < 1:
            raise ValidationError(
                f"futures_stale_ms must be >= 1, got {futures_stale_ms}")
        self.path = os.path.abspath(path)
        self.venue = venue
        self.rotate_bytes = int(rotate_bytes or 0)
        self.flush_rows = int(flush_rows)
        self.flush_bytes = int(flush_bytes)
        self.futures_stale_ms = int(futures_stale_ms)
        self.futures_symbol = futures_symbol
        self.part = 0
        # an id per writer, not per capture: reopening a path to continue a
        # session makes two ids in one file, which is the whole point
        self.session_id = f"{int(time.time() * 1000):x}-{os.getpid():x}"
        self.rows_written = 0
        self.n_book = self.n_trade = 0
        self.anomalies = 0
        self.last_book_ms = None
        self.last_trade_ms = None
        # the highest second written, per stream, for the ordering check. NOT
        # shared: `resample` buckets the merged capture as one timeline, but
        # Binance does not deliver the two streams in one timeline's order. See
        # `_check_order` -- a single shared watermark flagged 52 false positives
        # in a 20s live capture.
        self._max_second = {"book": 0, "trade": 0}
        self._out_of_order = {"book": False, "trade": False}
        self._held_futures = None
        self._held_at_ms = 0
        self._futures_stale = False
        self._closed = False
        self._summary = None
        self._buf: list[str] = []
        self._buf_bytes = 0
        self._fh = None
        self._sidecar = None
        self._open()

    # -- file plumbing --------------------------------------------------------

    def _segment(self) -> str:
        if self.part == 0:
            return self.path
        stem, ext = os.path.splitext(self.path)
        return f"{stem}.part{self.part:04d}{ext}"

    def _open(self) -> None:
        d = os.path.dirname(self.path)
        if d:
            os.makedirs(d, exist_ok=True)
        seg = self._segment()
        # size before opening, because append hides it
        prior = os.path.getsize(seg) if os.path.exists(seg) else 0
        self._fh = open(seg, "a", buffering=1 << 16)
        side = os.path.splitext(seg)[0] + ".meta.jsonl"
        self._sidecar = open(side, "a", buffering=1 << 16)
        if self.part == 0:
            # written directly rather than through `note()`, which counts
            # anomalies: a clean capture reported one anomaly it did not have,
            # and `anomalies` is the number a reader checks to decide whether a
            # file is trustworthy. A session start is not a fault.
            self._sidecar.write(json.dumps({
                "kind": "session_start", "wall": round(time.time(), 3),
                "session": self.session_id, "pid": os.getpid(),
                "resumed": prior > 0, "prior_bytes": prior or None,
                "venue": self.venue, "path": seg,
            }, separators=(",", ":")) + "\n")
            self._sidecar.flush()

    def _live(self, what: str) -> None:
        """Refuse to write to a closed capture.

        A write after `close()` used to append to an in-memory buffer that was
        never flushed again: no exception, `rows_written` incremented, and the
        row vanished. A recorder that quietly drops rows is the one failure this
        module exists to prevent, and the live path is exactly where it would
        happen -- a reconnect handler still draining a queue while the main task
        tears the writer down. So it raises instead, and says which call it was.
        """
        if self._closed:
            raise ValidationError(
                f"capture {self.path} is closed; cannot write a {what} row. "
                f"{self.rows_written} rows were written before close.")

    def _rotate(self) -> None:
        """Start a new segment.

        Only ever called between rows, so a rotation cannot interleave. The
        sidecar moves with it so diagnostics stay next to the data they describe
        even after a rollover.
        """
        self._fh.close()
        self._sidecar.close()
        self.part += 1
        self._open()

    def _push(self, line: str) -> None:
        """Buffer one serialised row, then settle the two size policies.

        Rotation is measured on *committed* bytes, and committing happens first.
        Measuring `tell()` with nothing flushed looks correct and is not: a
        small capture never reaches the disk, so a size-triggered rollover
        silently never fires and one session quietly becomes one unbounded file.
        Flushing whenever the buffer alone passes the threshold also keeps this
        linear, since after a rollover `tell()` is back to zero and the size test
        stays false until another `rotate_bytes` have actually been written.
        """
        self._buf.append(line)
        self._buf_bytes += len(line)
        if self.rotate_bytes and self._buf_bytes >= self.rotate_bytes:
            self.flush()
        if len(self._buf) >= self.flush_rows or self._buf_bytes >= self.flush_bytes:
            self.flush()
        if self.rotate_bytes and self._fh.tell() >= self.rotate_bytes:
            self._rotate()

    def _drop_empty_tail(self) -> str:
        """Delete a rollover segment that never received a row; return its sidecar.

        Rotation happens after a row is committed, so a rotation triggered by
        the *last* row of a session leaves a segment with nothing in it. That is
        not cosmetic: `read_events` rejects a zero-byte file as a missing-column
        error, so the reader hits it before any real data and the entire capture
        becomes unreadable rather than losing only its empty tail.

        Segment 0 is never removed: a capture that wrote nothing still has to
        leave evidence that it was started, and an empty base file is readable
        on its own terms.
        """
        seg, side = self._fh.name, self._sidecar.name
        empty = self._fh.tell() == 0
        self._fh.close()
        self._sidecar.close()
        self._fh = self._sidecar = None
        if empty and self.part > 0:
            for f in (seg, side):
                try:
                    os.remove(f)
                except OSError:
                    pass
            self.part -= 1
            return ""
        return side

    # -- diagnostics ----------------------------------------------------------

    def note(self, kind: str, **fields) -> None:
        """Record an anomaly in the sidecar. Never raises, never writes market data.

        Separate from the capture on purpose. A gap that gets a placeholder row
        in the market file is a gap a reader cannot distinguish from a quiet
        second, which is the failure this whole module is trying to avoid.
        """
        self._live("note")
        self.anomalies += 1
        rec = {"kind": kind, "wall": round(time.time(), 3), **fields}
        self._sidecar.write(json.dumps(rec, separators=(",", ":")) + "\n")
        self._sidecar.flush()

    def note_seq_gap(self, stream: str, expected: int, got: int, **fields) -> None:
        """A provable hole: a venue sequence number skipped values.

        The threshold is the caller's, because only the venue knows what a jump
        means. One skipped trade id is normal reconnect churn; a thousand is a
        capture that missed a busy minute, and the difference decides whether a
        later result is worth believing.
        """
        if got <= expected:
            return
        self.note("seq_gap", stream=stream, expected=expected, got=got,
                  missed=got - expected, **fields)

    def note_stale(self, stream: str, last_ms: int, now_ms: int,
                   after_ms: int, **fields) -> None:
        """No update for longer than the stream's update interval allows.

        Distinct from a sequence gap on purpose: silence is not proof of loss.
        A quiet book is a real market state, so this is reported and the data is
        left alone -- `resample` will mark those seconds `stale` on its own.
        """
        self.note("stale", stream=stream, last_ms=last_ms, now_ms=now_ms,
                  silent_ms=now_ms - last_ms, after_ms=after_ms, **fields)

    # -- capture --------------------------------------------------------------

    def hold_futures(self, mid: float, ts_ms: int | None = None) -> None:
        """Cache the correlated instrument's mid to stamp onto book rows.

        Carried forward rather than written as null on purpose. `resample`
        substitutes the spot mid when the futures column is missing, which is a
        silent lie: the futures-lead feature becomes identically zero, the RLS
        sees "no lead" instead of "no data", and the resulting degradation is
        indistinguishable from a real signal going dead. Holding the last good
        value keeps the feature honest for as long as the feed is connected.

        Once a value has been held for `futures_stale_ms`, the next book row
        writes a `futures_stale` note to the sidecar, once per outage. That is
        what makes the carry-forward safe: the value is a fallback, and its age
        is visible.

        The age is measured on *venue event time*, not the local clock, so a
        machine whose clock is wrong does not manufacture a phantom outage. Pass
        `ts_ms` when the venue gave one; without it the local clock is used and
        the staleness window is only as trustworthy as NTP. The consequence of
        measuring on event time is that the note rides along with the book: a
        perp feed that dies while the spot book is also silent produces no rows,
        and so no note. `resample` still flags those seconds `stale`, and silence
        in *both* streams is a condition the frame already carries, so nothing is
        lost -- the case this guards against is a dead perp next to a live book,
        which is exactly the asymmetric failure the carry-forward would hide.
        """
        self._held_futures = float(mid)
        self._held_at_ms = int(time.time() * 1000) if ts_ms is None else int(ts_ms)
        self._futures_stale = False

    def book(self, ts_ms: int, bid: float, ask: float, bid_qty: float,
             ask_qty: float, futures: float | None = None) -> None:
        """One book snapshot. A level at zero price is dropped, not written.

        A zero price is how some venues report "level removed". Writing it would
        make `resample` see a crossed or single-sided book; the resampler
        already keeps the last real level in that case, so the honest thing here
        is to not manufacture the event.
        """
        self._live("book")
        if not (bid > 0 and ask > 0):
            self.note("bad_book", ts_ms=int(ts_ms), bid=bid, ask=ask)
            return
        fut = self._held_futures if futures is None else futures
        if futures is not None:
            # a live value clears staleness whichever way it arrived. Resetting
            # only in `hold_futures` left the flag latched after a recovery that
            # came through `book`, so a *second* outage went unreported.
            self._held_futures = float(futures)
            self._held_at_ms = int(ts_ms)
            self._futures_stale = False
        elif self._held_futures is not None:
            # The held value is a promise, and this is where it is kept. A perp
            # feed that dies leaves `_held_futures` frozen at its last tick, and
            # a frozen lead is indistinguishable from a genuinely flat basis --
            # which is precisely the "no data" that must never look like "no
            # signal". The docstring claimed this was reported; it was not.
            age = int(ts_ms) - self._held_at_ms
            if age >= self.futures_stale_ms:
                if not self._futures_stale:
                    self._futures_stale = True
                    self.note("futures_stale", age_ms=age,
                              after_ms=self.futures_stale_ms,
                              held_mid=self._held_futures,
                              symbol=self.futures_symbol or None)
        self._push(json.dumps({
            "ts_ms": int(ts_ms), "mid": _num((bid + ask) / 2.0),
            "bid": _num(bid), "ask": _num(ask), "bid_qty": _num(bid_qty),
            "ask_qty": _num(ask_qty), "futures": _num(fut),
            "px": None, "qty": None, "side": None,
        }, separators=(",", ":")) + "\n")
        self._after(int(ts_ms), book=True)

    def trade(self, ts_ms: int, px: float, qty: float, side: int) -> None:
        """One public print. `side` is +1 buyer-initiated, matching the schema.

        `side` is named for the aggressor, not the buyer, and getting it
        backwards inverts the tape and therefore the whole toxicity estimate --
        so it is not defaulted or inferred here. A caller that does not know
        should not be allowed to guess quietly.
        """
        self._live("trade")
        if side not in (1, -1):
            raise ValidationError(f"trade side must be +1 or -1, got {side!r}")
        if not (px > 0 and qty >= 0):
            self.note("bad_trade", ts_ms=int(ts_ms), px=px, qty=qty)
            return
        self._push(json.dumps({
            "ts_ms": int(ts_ms), "mid": None, "bid": None, "ask": None,
            "bid_qty": None, "ask_qty": None, "futures": None,
            "px": _num(px), "qty": _num(qty), "side": int(side),
        }, separators=(",", ":")) + "\n")
        self._after(int(ts_ms))

    def _after(self, ts_ms: int, book: bool = False) -> None:
        """Bookkeeping only. Buffering, flushing and rotation all belong to
        `_push`, which every row goes through; deciding them here as well meant
        the segment was rolled after each row and the unflushed buffer went with
        it."""
        self._check_order(ts_ms, book)
        self.rows_written += 1
        if book:
            self.n_book += 1
            self.last_book_ms = ts_ms
        else:
            self.n_trade += 1
            self.last_trade_ms = ts_ms

    def _check_order(self, ts_ms: int, book: bool) -> None:
        """Note a row that lands in a second *its own stream* has already passed.

        The row is still written, and that is deliberate. Refusing it would lose a
        real print, and reordering at capture time would mean trusting a local sort
        over the venue's own sequence. Instead the event stays where it was filed
        and the sidecar records that the file is no longer monotonic, so
        `resample` relocating the row is a visible fact rather than a silent edit.

        The watermark is per stream, and that is the whole point. This recorder
        subscribes to `@depth10@100ms` and `@trade` on one connection so the file
        is merged by construction, but Binance does not *interleave* those two in
        time order. Measured against the live feed, the depth stream runs about
        1.4s behind the trade stream: over a 20s capture, trades ran 17.8s of
        event time while books ran 18.2s, and every one of the 52 backwards steps
        in the file was a book arriving after a later-dated print. Each stream on
        its own was perfectly monotonic -- zero backwards steps in either.

        A shared watermark therefore flagged a completely healthy capture 52
        times in 20 seconds, which is worse than not checking at all: a note that
        fires on every clean run is a note nobody reads. The earlier reasoning
        here was that two streams interleave freely *within* a second; against
        the real feed they interleave across seconds, by more than a second.

        What survives the narrowing is the case worth catching: one stream moving
        backwards relative to itself -- an NTP step, or a reconnect to a stream
        resuming from a lagging server. That is still flagged, once per
        excursion, and now says which stream it was.

        The test is *the second*, not the millisecond, because that is what
        `resample` buckets on. A 200ms step costs nothing; a 4s step is caught.
        """
        stream = "book" if book else "trade"
        sec = ts_ms // 1000
        if sec >= self._max_second[stream]:
            self._max_second[stream] = sec
            self._out_of_order[stream] = False
            return
        if not self._out_of_order[stream]:
            # once per excursion: a step backwards would otherwise produce one
            # note per row for as long as the feed stayed behind
            self._out_of_order[stream] = True
            self.note("out_of_order", stream=stream, ts_ms=ts_ms, second=sec,
                      behind_ms=self._max_second[stream] * 1000 - ts_ms,
                      passed_second=self._max_second[stream])

    def flush(self) -> None:
        self._live("flush")
        if self._buf:
            self._fh.write("".join(self._buf))
            self._buf.clear()
        self._buf_bytes = 0
        self._fh.flush()

    def close(self) -> dict:
        """Flush, close, and return the summary that goes in the sidecar's last line.

        Idempotent. `__exit__` and an explicit `close()` both fire on the normal
        path, and a `record()` interrupted by Ctrl-C reaches it twice; the second
        call used to raise `AttributeError` on a `None` file handle, turning a
        clean shutdown into a traceback and losing the summary line.
        """
        if self._closed:
            return dict(self._summary)
        self.flush()
        seg = self._fh.name
        # if the tail was an empty rollover it is gone, and the summary belongs
        # in the sidecar of the segment that actually holds the session
        side = self._drop_empty_tail() or \
            os.path.splitext(seg)[0] + ".meta.jsonl"
        summary = {"rows": self.rows_written, "book": self.n_book,
                   "trades": self.n_trade, "anomalies": self.anomalies,
                   "segments": self.part + 1, "session": self.session_id}
        with open(side, "a") as fh:
            fh.write(json.dumps({
                "kind": "closed", "venue": self.venue, **summary,
                "path": self.path,
            }, separators=(",", ":")) + "\n")
        self._fh = self._sidecar = None
        self._closed = True
        self._summary = summary
        return dict(summary)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        if self._fh is not None:
            self.close()
        return False


class BinanceSource:
    """Binance public websocket -> `CaptureWriter`. No auth, no order endpoints.

    Deliberately incapable of placing an order. The moment this class can trade
    is the moment a reconnect bug becomes money, and the sequencing that makes
    that survivable -- capture, replay, paper, then live -- only works while
    capture cannot trade. A separate, reviewed gateway earns that capability.
    """

    def __init__(self, symbol: str, futures_symbol: str | None = None,
                 depth: int = 10, update_ms: int = 100, testnet: bool = False):
        if depth not in (5, 10, 20):
            # the old `1 <= depth <= 20` looked right and was not: Binance serves
            # 5, 10 and 20 only, so depth=7 built a stream URL for a subscription
            # that does not exist and the socket closed on connect
            raise ValidationError(
                f"partial depth must be 5, 10 or 20, got {depth}")
        if update_ms not in (100, 1000):
            raise ValidationError(f"update_ms must be 100 or 1000, got {update_ms}")
        self.symbol = symbol.upper()
        self.futures_symbol = (futures_symbol or symbol).upper()
        self.depth, self.update_ms = depth, update_ms
        self.base = ("wss://testnet.binance.vision/stream" if testnet else SPOT_WS)
        self.fbase = FUTURES_WS

    def _streams(self) -> str:
        # one connection, interleaved, so the file is merged *by construction*
        # rather than by a join on a clock done after the fact
        s = self.symbol.lower()
        return f"{s}@depth{self.depth}@{self.update_ms}ms/{s}@trade"

    async def _run_spot(self, writer: CaptureWriter, stop: asyncio.Event) -> None:
        import websockets

        url = f"{self.base}?streams={self._streams()}"
        backoff = 0.5
        last_trade_id = None
        while not stop.is_set():
            try:
                async with websockets.connect(url, ping_interval=20,
                                              ping_timeout=20) as ws:
                    backoff = 0.5
                    # `while not stop.is_set()`, not `while True` with a check
                    # below: the body returned on stop, so the connect would be
                    # re-established forever and the reconnect loop turned into
                    # a tight spin. More importantly `while not True` is
                    # `while False` -- it is False immediately, so the receive
                    # loop never ran once and the recorder wrote nothing at all
                    # from a live socket while reporting a clean session.
                    while not stop.is_set():
                        raw = await asyncio.wait_for(ws.recv(), timeout=90)
                        msg = json.loads(raw)
                        data = msg.get("data", msg)
                        kind = data.get("e")
                        now = int(time.time() * 1000)
                        if kind == "depthUpdate" or "bids" in data:
                            self._on_depth(writer, data, now)
                        elif kind == "trade":
                            tid = data.get("t")
                            if last_trade_id is not None and tid is not None \
                                    and tid > last_trade_id + 1:
                                writer.note_seq_gap("trade", last_trade_id + 1, tid,
                                                    symbol=self.symbol)
                            if tid is not None:
                                last_trade_id = tid
                            self._on_trade(writer, data)
                        if stop.is_set():
                            return
            except asyncio.CancelledError:
                raise
            except Exception as e:
                # a timeout lands here too, which is right for the spot stream:
                # a silent spot book is a real condition, `resample` flags those
                # seconds `stale`, and a reconnect is better than a dead task
                writer.note("spot_disconnect", error=repr(e),
                            backoff_s=round(backoff, 2))
                await asyncio.sleep(backoff + random.uniform(0, 0.25))
                backoff = min(backoff * 2, 30.0)

    @staticmethod
    def _on_depth(writer: CaptureWriter, d: dict, now_ms: int) -> None:
        bids, asks = d.get("bids") or d.get("b") or [], d.get("asks") or d.get("a") or []
        if not bids or not asks:
            return
        bid, bq = float(bids[0][0]), float(bids[0][1])
        ask, aq = float(asks[0][0]), float(asks[0][1])
        # exchange event time where available: the wall clock a recorder stamps
        # on arrival is contaminated by this machine's own latency, which is
        # exactly the quantity a replay must not bake in
        ts = int(d.get("E") or d.get("T") or now_ms)
        writer.book(ts, bid, ask, bq, aq)

    @staticmethod
    def _on_trade(writer: CaptureWriter, d: dict) -> None:
        px, qty = float(d["p"]), float(d["q"])
        # `m` is "is the buyer the market maker?". True means the buyer was
        # passive, so the aggressor is the seller. Inverted, this silently flips
        # the tape and the toxicity estimate with it.
        side = -1 if d.get("m") else 1
        ts = int(d.get("T") or d.get("E") or 0)
        writer.trade(ts, px, qty, side)

    async def _run_futures(self, writer: CaptureWriter, stop: asyncio.Event) -> None:
        import websockets

        s = self.futures_symbol.lower()
        url = f"{self.fbase}?streams={s}@bookTicker"
        backoff = 0.5
        while not stop.is_set():
            try:
                async with websockets.connect(url, ping_interval=20,
                                              ping_timeout=20) as ws:
                    backoff = 0.5
                    # see `_run_spot`: `while not True` never entered the loop
                    while not stop.is_set():
                        raw = await asyncio.wait_for(ws.recv(), timeout=90)
                        d = json.loads(raw).get("data", {})
                        bid, ask = d.get("b"), d.get("a")
                        if bid and ask:
                            # `E` is the event time; passing it keeps the
                            # staleness window honest if the local clock drifts
                            writer.hold_futures((float(bid) + float(ask)) / 2.0,
                                                ts_ms=d.get("E") or d.get("T"))
                        if stop.is_set():
                            return
            except asyncio.CancelledError:
                raise
            except asyncio.TimeoutError:
                # 90s of a perp bookTicker is silence, not a normal gap, and it
                # used to propagate and kill the task. The held mid is already
                # flagged `futures_stale` by then, so this only has to reconnect.
                writer.note("futures_timeout", silent_s=90, symbol=self.futures_symbol)
                await asyncio.sleep(backoff + random.uniform(0, 0.25))
                backoff = min(backoff * 2, 30.0)
            except Exception as e:
                writer.note("futures_disconnect", error=repr(e),
                            backoff_s=round(backoff, 2))
                await asyncio.sleep(backoff + random.uniform(0, 0.25))
                backoff = min(backoff * 2, 30.0)

    async def _serve(self, writer: CaptureWriter, seconds: float | None) -> None:
        stop = asyncio.Event()
        tasks = [asyncio.create_task(self._run_spot(writer, stop)),
                 asyncio.create_task(self._run_futures(writer, stop))]
        try:
            if seconds:
                await asyncio.sleep(seconds)
            else:
                await asyncio.gather(*tasks)
        except (KeyboardInterrupt, asyncio.CancelledError):
            pass
        finally:
            stop.set()
            for t in tasks:
                t.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)

    def run(self, writer: CaptureWriter, seconds: float | None = None) -> dict:
        """Record until interrupted, or for `seconds` if given.

        `close` is reached from every exit, including an exception out of the
        event loop, because the writer's own `__exit__` would otherwise be the
        only thing standing between a crash and an unclosed capture. It is
        idempotent, so reaching it twice is harmless.
        """
        _require_websocket()
        try:
            asyncio.run(self._serve(writer, seconds))
        except KeyboardInterrupt:
            pass
        finally:
            writer.flush()
        return writer.close()


def record(symbol: str = "btcusdt", out: str = "data/capture.jsonl",
           futures_symbol: str | None = None, seconds: float | None = None,
           depth: int = 10, update_ms: int = 100, venue: str | None = None,
           rotate_mb: int = 0, testnet: bool = False) -> dict:
    """Record `symbol` to a merged JSONL capture. Blocking; Ctrl-C stops cleanly.

    A clean stop is the point of the `finally` in `_serve`: the writer is closed
    on interrupt as well as on normal exit, so a capture always ends with a
    `closed` line and the trailing buffer is never truncated to a half-written
    row. A capture that ends mid-line is discarded by the reader, and losing the
    final second of every session is a bad enough habit to design out.

    Nothing is created until the connection is actually attempted. The obvious
    ordering -- open the writer, then start the source -- leaves a zero-byte
    capture and its sidecar on disk whenever the source refuses to start, which
    is every run without `websockets` installed. The residue then looks like a
    real (empty) capture to anything that lists the directory, and the next
    `capture_to_dataset` on it fails on a file that was never going to have data.
    """
    src = BinanceSource(symbol, futures_symbol, depth, update_ms, testnet)
    venue = venue or f"binance:{src.symbol}"
    if rotate_mb is not None and rotate_mb < 0:
        raise ValidationError(f"rotate_mb must be >= 0, got {rotate_mb}")
    if seconds is not None and seconds <= 0:
        raise ValidationError(f"seconds must be > 0, got {seconds}")
    rot = int(rotate_mb * 1024 * 1024) if rotate_mb else 0
    _require_websocket()
    with CaptureWriter(out, venue=venue, rotate_bytes=rot,
                       futures_symbol=src.futures_symbol) as w:
        return src.run(w, seconds)


def _require_websocket() -> None:
    """Fail before a single file is created, rather than after."""
    try:
        importlib.import_module("websockets")
    except ImportError as e:
        raise ValidationError(
            "recording needs a websocket client; `pip install websockets`. "
            "The writer itself has no such dependency -- this is only the "
            "live source.") from e


def read_sidecar(path: str) -> list[dict]:
    """Every diagnostic line belonging to the capture at `path`.

    Follows the rollover family the same way `capture_to_dataset` does, so the
    result covers the whole session rather than whichever segment happens to hold
    the `closed` line. Unknown or unparseable lines are returned as-is under
    `"raw"`: a sidecar is written by a possibly different build of this module,
    and refusing to read an older capture is worse than showing the line.
    """
    import glob
    import re

    if os.path.isdir(path):
        segs = sorted(glob.glob(os.path.join(path, "*.jsonl")))
    else:
        stem, ext = os.path.splitext(path)
        if re.match(r"^.*\.part\d+$", stem):
            segs = [path]
        else:
            name = os.path.basename(stem)
            fam = re.compile(rf"^{re.escape(name)}\.part\d+{re.escape(ext)}$")
            segs = sorted(
                q for q in glob.glob(f"{glob.escape(stem)}*{ext}")
                if q == path or fam.match(os.path.basename(q)))
    out = []
    for seg in segs:
        side = os.path.splitext(seg)[0] + ".meta.jsonl"
        if not os.path.exists(side):
            continue
        with open(side) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except ValueError:
                    out.append({"raw": line})
                    continue
                out.append(rec if isinstance(rec, dict) else {"raw": rec})
    return out


def sessions(path: str) -> list[dict]:
    """The `session_start` records for a capture, in file order.

    A capture written by two processes -- a crash and restart, or two recorders
    pointed at one path -- has more than one, and that is the fact worth knowing
    before a replay. Length 1 means the file is one unbroken session; anything
    longer means the seam is real and the frames across it were assembled from
    events separated by an unknown gap.
    """
    return [r for r in read_sidecar(path) if r.get("kind") == "session_start"]


def capture_is_continuous(path: str) -> bool:
    """True when the capture is a single unbroken session.

    Convenience over `len(sessions(path)) == 1`, and False rather than raising
    when there is no sidecar at all: a capture with no diagnostics is not
    evidence of continuity, it is absence of evidence.
    """
    return len(sessions(path)) == 1
