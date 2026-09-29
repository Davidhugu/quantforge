"""Columnar store for replayed data, and the event->frame resampler.

The engine's clock is one wall-clock second: `run_day` does `for t in range(T)`
and indexes `Sl[t]`, `Fl[t]`, `tapel[t]` positionally. Real order-book data runs
at hundreds of events per second. This module owns that conversion, in one
place, so the engine never learns what the venue's native frequency was.

Deliberate design choices:

  * **Gaps are filled, not skipped.** A missing second becomes a row with
    `stale=True` and flat prices, so the engine sees a clock with the same
    length on every day and its positional indexing stays honest. Dropping the
    gap instead would compress the day and quietly shorten every horizon.
  * **`npz` is the on-disk format.** numpy is already a dependency; parquet
    would add one for a research dataset this size does not need. `load` will
    read parquet if pyarrow happens to be installed, since the capture tools
    emit that.
  * **Trades are kept at native resolution.** Frames answer "what did the book
    look like"; only the trade table can answer "did anyone print at my price".
"""
from __future__ import annotations

import os

import numpy as np

from .schema import (FRAME_COLUMNS, REQUIRED, TRADE_COLUMNS, ValidationError,
                     imbalance_from_depth, validate)


class Dataset:
    """A validated session: one-second `frames` plus a native-resolution `trades`."""

    def __init__(self, frames: dict, trades: dict, meta: dict | None = None):
        self.frames = {k: np.asarray(v) for k, v in frames.items()}
        # coerce trades to the declared dtypes only after the NaN padding is
        # gone: casting NaN to int is undefined and silently wraps
        self.trades = {k: np.asarray(v).astype(TRADE_COLUMNS[k], copy=False)
                       for k, v in trades.items()}
        self.meta = dict(meta or {})
        validate({"frames": self.frames, "trades": self.trades})
        if "imbalance" not in self.frames:
            self.frames["imbalance"] = imbalance_from_depth(
                self.frames["bid_qty"], self.frames["ask_qty"])
        self._index_trades_into_seconds()

    def _index_trades_into_seconds(self) -> None:
        """Attach each trade to a frame index, for the fill model.

        Built once here so `ReplayMarket.intents` stays O(log n) per call rather
        than scanning the tape. `searchsorted` with `side="right"` puts a trade
        stamped exactly at a second boundary *into* that second, which matches
        how the engine reads `tapel[t]` before deciding fills at t.
        """
        if not len(self.frames["ts_ms"]):
            self._sec_of_trade = np.zeros(0, dtype=np.int64)
            return
        self._sec_of_trade = np.searchsorted(
            self.frames["ts_ms"], self.trades["ts_ms"], side="right") - 1

    @property
    def n_seconds(self) -> int:
        return len(self.frames["mid"])

    def trades_in_second(self, i: int) -> np.ndarray:
        """Indices into `self.trades` belonging to frame `i`."""
        if i < 0 or i >= self.n_seconds or not len(self._sec_of_trade):
            return np.zeros(0, dtype=np.int64)
        hit = self._sec_of_trade == i
        return np.flatnonzero(hit)

    def signed_volume(self) -> np.ndarray:
        """Per-second signed traded volume, +buyer-initiated.

        This is the analogue of the synthetic market's `tape`, and it feeds the
        same VPIN accumulator -- so a toxicity estimate computed on real prints
        is directly comparable to the one the harness already reports.
        """
        out = np.zeros(self.n_seconds, dtype=float)
        if not len(self._sec_of_trade):
            return out
        ok = self._sec_of_trade >= 0
        np.add.at(out, self._sec_of_trade[ok],
                  self.trades["side"][ok] * self.trades["qty"][ok])
        return out

    def __repr__(self) -> str:
        span = ""
        if self.n_seconds:
            span = f" {self.frames['ts_ms'][0]}..{self.frames['ts_ms'][-1]}"
        return (f"<Dataset {self.n_seconds}s{span} "
                f"{len(self.trades['px'])} trades {self.meta}>")


def _ffill(a: np.ndarray) -> np.ndarray:
    """Carry the last non-zero value forward across missing seconds.

    Zero is the sentinel for "no observation" here: a valid price is never 0
    (validate rejects it), and a missing second is left as 0 by the last-wins
    loop above. Leading misses stay missing, so the caller can still tell the
    difference between "the capture began late" and "the capture has a hole".
    """
    seen = np.flatnonzero(a != 0)
    if not len(seen):
        return a
    idx = np.maximum.accumulate(
        np.where(a != 0, np.arange(len(a)), seen[0]))
    idx[:seen[0]] = seen[0]
    out = a[idx]
    out[:seen[0]] = 0.0
    return out


def resample(events: dict, start_ms: int, seconds: int) -> Dataset:
    """Bucket raw depth+trade events into `seconds` one-second frames.

    `events` is a *merged* stream: one row per event, every column the same
    length, with NaN where a column does not apply to that row. A depth event
    carries `bid`/`ask`/`mid` and no `px`; a print carries `px`/`qty`/`side` and
    no book. That padding convention is what lets one time-ordered stream be
    the input instead of two files that have to be joined on a clock, which is
    the usual source of look-ahead in a replay.

    Frames take the *last* book state within the second, not an average: the
    engine acts on the quote it can see now, and a time-averaged book is not a
    book any participant could have quoted against. A second with no book event
    carries the previous state forward and is marked `stale`, so a genuine feed
    gap is visible in the audit trail instead of looking like a quiet market.
    """
    if seconds < 1:
        raise ValidationError(f"seconds must be >= 1, got {seconds}")

    lengths = {len(np.asarray(v)) for k, v in events.items()
               if k in REQUIRED}
    if len(lengths) != 1:
        raise ValidationError(
            f"event columns have differing lengths {sorted(lengths)}; a merged "
            f"stream needs every column padded to the row count (NaN where the "
            f"column does not apply)")

    ts = np.asarray(events["ts_ms"], dtype=np.int64)
    if len(ts) == 0:
        raise ValidationError("no events to resample")

    sec = (ts - start_ms) // 1000
    if sec.min() < 0 or sec.max() >= seconds:
        raise ValidationError(
            f"events span seconds [{int(sec.min())}, {int(sec.max())}] outside "
            f"the requested window [0, {seconds}) from {start_ms}")

    col = lambda k: np.asarray(events[k], dtype=float)  # noqa: E731
    is_book = np.isfinite(col("mid")) & np.isfinite(col("bid")) & np.isfinite(col("ask"))
    is_trade = np.isfinite(col("px"))

    # sort here rather than trusting the caller. "Last event in the second wins"
    # is only meaningful in time order, and a capture that interleaves a second's
    # prints with the next second's depth update is not a rare thing to receive
    # -- it is the normal shape of any stream that is not already merged.
    frames = {k: np.zeros(seconds, dtype=t) for k, t in FRAME_COLUMNS.items()}
    have = np.zeros(seconds, dtype=bool)
    # the frame clock is defined by the window, not by which seconds happened to
    # carry a book event, so it is assigned for every second including the gaps
    frames["ts_ms"] = start_ms + np.arange(seconds, dtype=np.int64) * 1000

    order = np.argsort(ts, kind="stable")
    ts = ts[order]
    sec = sec[order]
    is_book, is_trade = is_book[order], is_trade[order]
    bid, ask = col("bid")[order], col("ask")[order]
    bid_qty = (col("bid_qty") if "bid_qty" in events
               else np.zeros(len(ts)))[order]
    ask_qty = (col("ask_qty") if "ask_qty" in events
               else np.zeros(len(ts)))[order]
    mid = col("mid")[order]
    fut = (col("futures") if "futures" in events else col("mid"))[order]
    fut = np.where(np.isfinite(fut), fut, mid)

    # last-wins per second: walk backwards and let earlier events overwrite.
    # A depth event with qty 0 is a level being removed, not a level of size
    # zero, so a non-positive price keeps whatever the last real print of it was
    # rather than collapsing the book to crossed or single-sided.
    for i in np.flatnonzero(is_book)[::-1]:
        s = int(sec[i])
        frames["bid"][s] = bid[i] if bid[i] > 0 else frames["bid"][s]
        frames["ask"][s] = ask[i] if ask[i] > 0 else frames["ask"][s]
        frames["bid_qty"][s] = max(0.0, bid_qty[i])
        frames["ask_qty"][s] = max(0.0, ask_qty[i])
        frames["futures"][s] = fut[i]
        frames["mid"][s] = mid[i]

        have[s] = True

    if not have.any():
        raise ValidationError(
            f"the window from {start_ms} contains no book events; depth columns "
            f"are all NaN, so there is nothing to quote against")
    if not have.all():
        if not have[0]:
            raise ValidationError(
                f"the window from {start_ms} has no book events in its first "
                f"second; cannot establish an opening book")
        # fill every interior gap from the last real observation, not just the
        # leading prefix. A hole in the middle of a capture is the normal case
        # and it must not silently become a zero-price second, which would show
        # up downstream as a 100% drawdown.
        for k in ("bid", "ask", "bid_qty", "ask_qty", "futures", "mid"):
            frames[k] = _ffill(frames[k])
    frames["stale"] = ~have

    if np.any(frames["ask"] <= 0) or np.any(frames["bid"] <= 0):
        raise ValidationError(
            f"{int(((frames['ask'] <= 0) | (frames['bid'] <= 0)).sum())} second(s) "
            f"have no usable quote after forward-fill; the capture starts inside "
            f"a gap and the window needs widening backwards")

    ti = np.flatnonzero(is_trade)
    trades = {}
    for k in TRADE_COLUMNS:
        if k == "ts_ms":
            trades[k] = ts[ti]
        elif k in events:
            trades[k] = np.asarray(events[k])[order][ti]
        else:
            raise ValidationError(f"event stream has no `{k}` column for trades")
    return Dataset(frames, trades, {"venue": events.get("venue", "unknown")})


def save(ds: Dataset, path: str) -> None:
    """Write a `Dataset` to a single `.npz`."""
    payload = {f"f_{k}": v for k, v in ds.frames.items()}
    payload.update({f"t_{k}": v for k, v in ds.trades.items()})
    payload["meta_keys"] = np.array(list(ds.meta), dtype=object)
    payload["meta_vals"] = np.array([ds.meta[k] for k in ds.meta], dtype=object)
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    np.savez_compressed(path, **payload)


def load(path: str) -> Dataset:
    """Read a `.npz` written by `save`.

    Object arrays are the awkward part of npz: allow_pickle is required to read
    the metadata back, and it is the one place this module loads untrusted
    input, so it is isolated here and documented rather than spread around.
    """
    with np.load(path, allow_pickle=True) as z:
        frames = {k[2:]: z[k] for k in z.files if k.startswith("f_")}
        trades = {k[2:]: z[k] for k in z.files if k.startswith("t_")}
        meta = {}
        if "meta_keys" in z.files:
            for k, v in zip(z["meta_keys"], z["meta_vals"]):
                meta[str(k)] = v
    return Dataset(frames, trades, meta)
