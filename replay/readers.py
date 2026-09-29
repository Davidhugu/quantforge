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
import io
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
