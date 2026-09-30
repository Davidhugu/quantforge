"""Readers: recorded files in, `Dataset` out.

Two tiers of fidelity, kept explicitly separate because confusing them is how a
"real data" backtest quietly becomes a second simulation:

  merged capture   book updates *and* prints, time-ordered in one stream. This
                   is what a websocket recorder writes and the only tier that
                   supports a real reachability check on fills.
  public archives  Binance's bulk `trades`/`aggTrades` CSVs, optionally with
                   5-minute `bookDepth` snapshots. There is no book between
                   snapshots, so depth is either unknown or held flat for
                   minutes at a time. Trade-derived mid and signed tape are
                   genuine; the book is not. Usable for the signal layers, not
                   for anything queue-aware.

Formats read without a new dependency: JSONL (what most recorders emit) and CSV.
Parquet is read if pyarrow is present, since that is what the bigger capture
tools produce.
"""
from __future__ import annotations

import csv
import gzip
import json
import os

import numpy as np

from .schema import REQUIRED, ValidationError
from .store import Dataset, resample


def _finalize(events: dict) -> dict:
    missing = [k for k in REQUIRED if k not in events]
    if missing:
        raise ValidationError(
            f"event stream is missing {missing}; a merged capture needs the "
            f"book columns and the trade columns together")
    ev = {k: np.asarray(v) for k, v in events.items() if k in REQUIRED
          or k in ("futures", "venue")}
    order = np.argsort(ev["ts_ms"], kind="stable")
    return {k: (v[order] if isinstance(v, np.ndarray) else v) for k, v in ev.items()}


def _from_jsonl(path: str) -> dict:
    ev: dict = {}
    with open(path) as fh:
        for i, line in enumerate(fh):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            for k, v in row.items():
                ev.setdefault(k, []).append(v)
    out = {}
    for k, v in ev.items():
        out[k] = np.array(v)
    if "ts_ms" not in out and "ts" in out:
        out["ts_ms"] = (out["ts"] * 1000).astype(np.int64)
    return out


def _from_csv(path: str) -> dict:
    opener = gzip.open if path.endswith(".gz") else open
    cols: dict = {}
    with opener(path, "rt", newline="") as fh:
        for row in csv.DictReader(fh):
            for k, v in row.items():
                cols.setdefault(k, []).append(v)
    out = {}
    for k, v in cols.items():
        try:
            out[k] = np.array(v, dtype=float)
        except ValueError:
            out[k] = np.array(v, dtype=object)
    if "ts_ms" not in out and "ts" in out:
        out["ts_ms"] = (out["ts"] * 1000).astype(np.int64)
    for k in ("bid_qty", "ask_qty"):
        if k in out:
            out[k] = np.nan_to_num(out[k].astype(float))
    return out


def read_events(path: str) -> dict:
    """Read a merged capture into the event dict `resample` expects."""
    ext = os.path.splitext(path)[1].lower()
    if ext in (".jsonl", ".json", ".ndjson"):
        return _finalize(_from_jsonl(path))
    if ext in (".csv", ".gz"):
        return _finalize(_from_csv(path))
    if ext in (".parquet", ".pq"):
        try:
            import pyarrow.parquet as pq
        except ImportError as e:  # pragma: no cover - depends on environment
            raise ValidationError(
                "reading parquet needs pyarrow; install it or convert with "
                "the capture tool's own CSV export") from e
        tbl = pq.read_table(path).to_pydict()
        return _finalize({k: np.array(v) for k, v in tbl.items()})
    raise ValidationError(f"unsupported capture format: {ext}")


def capture_to_dataset(path: str, complete_only: bool = True,
                       venue: str | None = None) -> Dataset:
    """Turn a merged capture into a `Dataset`, deriving the window from the file.

    `resample` needs a `start_ms` and a second count up front. A recorder knows
    those only as it goes, so on the way back in they have to be recovered from
    the data rather than guessed -- and getting them wrong is the quietest
    possible failure, because a window that is too wide silently inserts flat
    seconds the market never had and a window that is too narrow silently drops
    the tail.

    `complete_only` drops the capture's opening second. The first book row in
    any capture is the first thing the recorder ever saw, so prints that landed
    before it were matched against levels nobody was watching, and the signed
    volume of that second -- which feeds the toxicity estimator -- is short by
    an unknown amount. One second of tape is not worth an unquantified bias at
    the start, so it goes.

    The *last* second is kept, and the reason is worth stating because it looks
    inconsistent: nothing is missing after it. It is simply where the recording
    stopped, which is a different thing from a hole, and trimming it would
    discard a whole second of good data on every single run.

    A row whose timestamp lands in a second already passed is written, not
    resorted, and this function does not consult the sidecar's `out_of_order`
    note: the file is read as filed. That is deliberate on the writer's side
    (refusing a real print loses data) but it means a capture with such a note
    replays with one second built from events the venue did not put together.
    Check the sidecar before trusting a replay.

    Reads every rollover segment of the capture, which is what a recorder left
    running over a long session produces. Segments are matched *exactly* -- the
    base name plus `<stem>.partNNNN<ext>` -- and never by a prefix glob:

      * a prefix glob is cross-symbol contamination waiting to happen. `cap*`
        also matches `cap2.jsonl`, so a second instrument's capture is
        concatenated into the first and the result fails validation as a
        "trade px falls outside the book range" error, which points at the feed
        rather than at the reader. `eth*` swallowing `ethereum.jsonl` is the
        same bug with a longer name.
      * a rotation that fires on the final row leaves a zero-byte segment, and
        `read_events` rejects an empty file as a missing-column error. Skipping
        empty segments is therefore mandatory, not defensive: without it a
        capture that happened to stop on a boundary is unreadable *forever*,
        and the whole session goes, not just its tail.

    A directory is read whole, since a directory has no base name to anchor the
    family to -- every capture in it is concatenated. Point at a file unless
    that is what you mean.
    """
    import glob
    import re

    if os.path.isdir(path):
        candidates = sorted(glob.glob(os.path.join(path, "*.jsonl")))
    else:
        stem, ext = os.path.splitext(path)
        m = re.match(r"^(?P<stem>.*)\.part(?P<n>\d+)$", stem)
        if m:
            # a segment named explicitly is one piece of a session, taken alone
            candidates = [path]
        else:
            # the regex is anchored on the *basename*: anchoring it on the full
            # stem would never match, since it is applied to a basename
            name = os.path.basename(stem)
            family = re.compile(rf"^{re.escape(name)}\.part\d+{re.escape(ext)}$")
            candidates = sorted(
                q for q in glob.glob(f"{glob.escape(stem)}*{ext}")
                if q == path or family.match(os.path.basename(q)))
    parts = [q for q in candidates if not q.endswith(".meta.jsonl")
             and os.path.getsize(q) > 0] or [path]
    if len(parts) > 1:
        merged: dict = {}
        for p in parts:
            for k, v in read_events(p).items():
                if k == "venue":
                    continue
                merged.setdefault(k, []).append(np.asarray(v))
        events = {k: np.concatenate(v) for k, v in merged.items()}
    else:
        events = read_events(parts[0])

    ts = np.asarray(events["ts_ms"], dtype=np.int64)
    if not len(ts):
        raise ValidationError(f"capture {parts[0]} has no events")

    # floor to the second *before* subtracting, so the window is aligned to
    # absolute seconds rather than to wherever the first event happened to land
    lo, hi = int(ts.min()) // 1000 * 1000, int(ts.max()) // 1000 * 1000
    if complete_only:
        lo += 1000
    if hi < lo:
        raise ValidationError(
            f"capture {parts[0]} spans less than two complete seconds "
            f"({len(ts)} events); not enough to form a frame interval")
    if complete_only and len(ts):
        # the events have to go, not just the window: leaving them in place puts
        # the dropped second at index -1, which `resample` rejects as out of range
        keep = ts >= lo
        events = {k: (np.asarray(v)[keep] if isinstance(v, np.ndarray)
                      else [x for x, m in zip(v, keep) if m])
                  for k, v in events.items()}
    events["venue"] = venue or _recorded_venue(parts[0]) or \
        os.path.splitext(os.path.basename(parts[0]))[0]
    return resample(events, lo, (hi - lo) // 1000 + 1)


def _recorded_venue(seg: str) -> str | None:
    """The venue the recorder itself recorded, from the sidecar.

    Falling back to the filename stem is a guess -- `btc.jsonl` becomes `btc`,
    which is a symbol and not a venue, and for a multi-venue setup it silently
    merges everything sharing a ticker into one label. The sidecar already
    carries what was actually connected to (`binance:BTCUSDT`), and it lives
    there precisely so the data file does not have to. Returned only for a
    capture this project wrote, so a foreign file still gets the old behaviour.

    Imported lazily: `record` is the heavier module and most readers never
    touch a recorder's output.
    """
    try:
        from .record import read_sidecar
        for rec in read_sidecar(seg):
            if rec.get("kind") == "session_start" and rec.get("venue"):
                return str(rec["venue"])
    except Exception:
        return None          # no sidecar, unreadable, or not a capture at all
    return None


def from_binance_trades(path: str, futures_path: str | None = None) -> Dataset:
    """Build a Dataset from Binance's bulk trade archive. Degraded, on purpose.

    Only trade prints are available, so the mid is the last print in each second
    and the book is carried flat at +/- the observed spread. Depth is genuinely
    unknown and is recorded as equal size either side, which makes the imbalance
    feature identically zero and therefore inert -- the signal layer will see
    no depth imbalance at all, only the futures lead and the tape. That is the
    honest ceiling of this tier, and it is why it is a separate function from
    the merged-capture path rather than a fallback inside it.
    """
    ev = _from_csv(path)
    ts = ev["ts_ms"].astype(np.int64)
    px, qty = ev["px"].astype(float), ev["qty"].astype(float)
    # Binance's maker flag: 1 means the buyer was passive, so a taker BUY is a
    # seller-initiated print. Getting this backwards silently inverts the tape
    # and therefore the toxicity estimate, so it is derived explicitly.
    if "is_maker" in ev:
        side = np.where(ev["is_maker"].astype(float) > 0.5, -1.0, 1.0).astype(np.int8)
    elif "side" in ev:
        side = ev["side"].astype(np.int8)
    else:
        raise ValidationError("trade archive has neither `is_maker` nor `side`; "
                              "the aggressor side is required for the tape")
    sec = np.floor((ts - ts.min()) / 1000.0).astype(np.int64)
    n = int(sec.max()) + 1
    mid = np.full(n, np.nan)
    np.maximum.at(mid, sec, px)          # last print wins, as `resample` does
    last = np.isfinite(mid)
    if not last.all():
        mid[~last] = mid[np.argmax(last)]
    fut = mid
    if futures_path:
        fv = _from_csv(futures_path)
        fmid = np.full(n, np.nan)
        np.maximum.at(fmid, np.floor((fv["ts_ms"].astype(np.int64) - ts.min())
                                     / 1000.0).astype(np.int64), fv["px"].astype(float))
        if np.isfinite(fmid).any():
            fut = np.where(np.isfinite(fmid), fmid, mid)
    half = np.abs(mid) * 1e-4
    return resample({
        "ts_ms": ts, "mid": mid, "bid": mid - half, "ask": mid + half,
        "bid_qty": np.ones(n), "ask_qty": np.ones(n), "futures": fut,
        "px": px, "qty": qty, "side": side, "venue": "binance-trades",
    }, int(ts.min()), n)
