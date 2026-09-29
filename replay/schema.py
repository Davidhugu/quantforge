"""Canonical on-disk/on-wire schema for replayed market data.

Every venue reader normalises into these columns, so the harness never learns a
venue's dialect. A new venue is a missing reader, not a missing schema.

Two resolutions live here, deliberately kept apart:

  * a *frame* table, one row per wall-clock second, which is what `run_day`
    consumes -- the engine's clock is one second and it indexes these directly.
  * a *trade* table, one row per public print, which is what the fill model
    needs. Frames cannot answer "did anyone actually trade at the price we were
    quoting", and pretending otherwise is how a replay quietly becomes a second
    simulation.

The engine is also the reason for one convention: `Il` (imbalance) and `Fl`
(futures lead) are read as *observed* series, whereas the synthetic `Market`
generates them from the latent drift it is trying to predict. That is the whole
point of replaying -- the signal layer finally gets data it did not manufacture.
"""


import numpy as np

# name -> dtype. Kept as a single ordered mapping so the store, the reader and
# the tests all agree on spelling without importing each other.
FRAME_COLUMNS: dict[str, np.dtype] = {
    "ts_ms": np.dtype("int64"),      # exchange event time, ms since epoch
    "mid": np.dtype("float64"),
    "bid": np.dtype("float64"),
    "ask": np.dtype("float64"),
    "bid_qty": np.dtype("float64"),
    "ask_qty": np.dtype("float64"),
    "futures": np.dtype("float64"),  # mid of the correlated instrument
    "stale": np.dtype("bool"),
}

TRADE_COLUMNS: dict[str, np.dtype] = {
    "ts_ms": np.dtype("int64"),
    "px": np.dtype("float64"),
    "qty": np.dtype("float64"),
    "side": np.dtype("int8"),         # +1 buyer-initiated, -1 seller-initiated
}

# What a *merged* event stream must carry: one row per event, every column
# padded to the row count with NaN where it does not apply. A depth event fills
# the book columns and NaN in `px`; a print does the reverse.
REQUIRED = ("ts_ms", "mid", "bid", "ask", "bid_qty", "ask_qty", "px", "qty", "side")


class ValidationError(ValueError):
    """A dataset failed a check that would silently corrupt a backtest.

    Not a dataclass: it is raised with a message like any exception, and
    wrapping it in one gave it a generated `__init__` that rejected its own
    single argument.
    """


def imbalance_from_depth(bid_qty, ask_qty):
    """`bid_qty / (bid_qty + ask_qty) - 0.5`, in [-0.5, 0.5].

    Signed, so positive means bid-heavy. The synthetic market's `I` is also
    signed and roughly in this range, so the RLS and the Avellaneda-Stoikov
    reservation price see the input scale they were built against. Returns 0.0
    where the book is flat rather than nan: a crossed or empty book is a data
    problem, reported by `validate`, not a reason to poison the signal.
    """
    b = np.asarray(bid_qty, dtype=float)
    a = np.asarray(ask_qty, dtype=float)
    total = b + a
    safe = np.where(total > 0, total, 1.0)
    # 0.5 -> 0.0 where the book is flat rather than nan: a crossed or empty book
    # is a data problem, reported by `validate`, not a reason to poison the signal
    return np.where(total > 0, b / safe, 0.5) - 0.5


def validate(tables: dict) -> None:
    """Raise `ValidationError` on anything that would make a replay lie.

    `tables` maps "frames" and "trades" to column dicts, so both are looked up
    in one place and a future table is added by extending the mapping rather
    than the signature.

    Checked here rather than in the reader so that a hand-made or third-party
    dataset gets the same treatment as a downloaded one. The theme is
    look-ahead and internal consistency: a crossed book, a non-positive price, or
    a trade that lands outside the book it was supposedly matched against all
    mean the replay is modelling something other than the market.
    """
    for name, cols in (("frames", FRAME_COLUMNS), ("trades", TRADE_COLUMNS)):
        missing = [c for c in cols if c not in tables.get(name, {})]
        if missing:
            raise ValidationError(f"{name} is missing column(s): {missing}")

    f = tables["frames"]
    n = len(f["mid"])
    if n < 2:
        raise ValidationError(f"need >= 2 frames to have an interval, got {n}")
    for key, col in f.items():
        if len(col) != n:
            raise ValidationError(
                f"frames.{key} has {len(col)} rows, mid has {n}; a ragged "
                f"table means the reader dropped or duplicated a second")

    if np.any(f["ask"] <= 0) or np.any(f["bid"] <= 0) or np.any(f["mid"] <= 0):
        raise ValidationError("non-positive price in frames")
    crossed = f["ask"] < f["bid"]
    if crossed.any():
        i = int(np.argmax(crossed))
        raise ValidationError(
            f"crossed book at frame {i} (ts_ms={int(f['ts_ms'][i])}): "
            f"bid {f['bid'][i]!r} > ask {f['ask'][i]!r}")
    if np.any(f["bid_qty"] < 0) or np.any(f["ask_qty"] < 0):
        raise ValidationError("negative size in frames")

    ts = np.asarray(f["ts_ms"], dtype=np.int64)
    if np.any(np.diff(ts) <= 0):
        i = int(np.argmax(np.diff(ts) <= 0))
        raise ValidationError(
            f"frames are not strictly increasing in ts_ms at index {i} "
            f"({int(ts[i])} -> {int(ts[i + 1])}); the engine indexes these "
            f"positionally, so a gap or a reorder would shift time silently")

    t = tables["trades"]
    if len(t["px"]):
        if np.any(t["px"] <= 0) or np.any(t["qty"] < 0):
            raise ValidationError("non-positive price or negative size in trades")
        if not np.isin(t["side"], (-1, 1)).all():
            raise ValidationError("trade side must be +1 or -1")
        if len(t["side"]) != len(t["px"]):
            raise ValidationError("trades: side/px length mismatch")
        # a print outside the book it was matched in means the two streams came
        # from different sources or the book is being reconstructed wrongly
        lo, hi = f["bid"].min(), f["ask"].max()
        if t["px"].min() < lo - 1e-9 or t["px"].max() > hi + 1e-9:
            raise ValidationError(
                f"trade px [{t['px'].min():.6g}, {t['px'].max():.6g}] falls "
                f"outside the book range [{lo:.6g}, {hi:.6g}] over the same "
                f"window; the book and the tape disagree")
