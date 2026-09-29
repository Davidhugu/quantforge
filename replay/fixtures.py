"""Hermetic test data, so the replay path is verifiable with no network.

These are *not* market data and must never be used to judge a strategy. They
exist for one purpose: to let `ReplayMarket` and the resampler be tested
offline, deterministically, and at any horizon. The generator deliberately
injects the awkward cases the real path has to survive -- a capture gap, a
one-sided book, a second with no prints at all -- because a fixture that is
well-behaved everywhere tests nothing.

To get real data, see `readers`: capture one, or download an archive.
"""
from __future__ import annotations

import numpy as np

from .store import Dataset, resample


def synth_session(seed: int = 0, seconds: int = 900, start_ms: int = 1_757_000_000_000,
                  px0: float = 100.0, gap_at: int | None = None,
                  toxic_share: float = 0.08, vol_scale: float = 1.0,
                  annual_vol: float = 0.25) -> Dataset:
    """A plausible-but-synthetic session as a merged event stream.

    Volatility is specified as an *annualised* figure and converted internally.
    That is not fussiness. The Avellaneda-Stoikov half-spread is
    `_vterm * var + _kterm`, and the variance term is linear in the realised
    per-second variance, so a fixture with per-second noise of 0.02 -- 200 bps a
    second, about 30x a liquid market -- produces a quoting spread an order of
    magnitude wider than the book it is trying to trade against. Nothing fills,
    the ladder comes back all zeros, and the result looks like a finding about
    the strategy when it is a bug in the fixture. `test_replay.py` pins the
    realised volatility so that failure cannot come back silently.

    Sticky toxic flow: an informed "burst" window during which prints are
    disproportionately buyer- or seller-initiated and price drifts with them.
    Aiming it slightly makes a test that trades only informed flow show a
    markout; aiming it randomly makes everything a coin flip.
    """
    rng = np.random.default_rng(seed)
    # per-second log-return sd, from an annualised figure over calendar seconds
    sigma = annual_vol / np.sqrt(365.0 * 86400.0)
    drift = np.zeros(seconds)
    n_bursts = max(1, seconds // 300)
    for _ in range(n_bursts):
        s = int(rng.integers(0, seconds - 30))
        drift[s:s + 30] = rng.normal(0, 6 * sigma)
    step = rng.normal(0, sigma, seconds) + drift
    mid = px0 * np.exp(np.cumsum(step))

    half = np.maximum(1e-4, mid * rng.uniform(2e-4, 6e-4, seconds))
    bid, ask = mid - half, mid + half
    # depth correlated with the move, so imbalance carries real information
    lean = np.tanh(np.cumsum(step) * 20.0)
    bid_qty = np.clip(50 * (1.0 + lean) * rng.uniform(0.5, 1.5, seconds), 1.0, None)
    ask_qty = np.clip(50 * (1.0 - lean) * rng.uniform(0.5, 1.5, seconds), 1.0, None)

    # One merged row per event, NaN where the column does not apply -- the same
    # shape a real capture has, so the fixture exercises the padding path rather
    # than a convenience format only the fixture produces.
    rows = []
    gap = {gap_at, gap_at + 1, gap_at + 2} if gap_at is not None else set()
    for t in range(seconds):
        base = start_ms + t * 1000
        if t not in gap:
            rows.append((base, mid[t], bid[t], ask[t], bid_qty[t], ask_qty[t],
                         mid[t], np.nan, np.nan, np.nan))
        n_tr = int(rng.poisson(6 * vol_scale))
        for _ in range(n_tr):
            toxic = rng.random() < toxic_share
            s = 1.0 if (1.0 if lean[t] > 0 else -1.0) * toxic else \
                rng.choice([-1.0, 1.0])
            q = abs(rng.lognormal(1.0, 0.6)) * (2.5 if toxic else 1.0)
            # prints land strictly inside the book, which `validate` enforces
            price = (rng.uniform(mid[t], ask[t]) if s > 0
                     else rng.uniform(bid[t], mid[t]))
            rows.append((base + int(rng.integers(0, 1000)), np.nan, np.nan, np.nan,
                         np.nan, np.nan, np.nan, price, q, s))

    ev = np.array(rows, dtype=float)
    return resample({
        "ts_ms": ev[:, 0].astype(np.int64),
        "mid": ev[:, 1], "bid": ev[:, 2], "ask": ev[:, 3],
        "bid_qty": ev[:, 4], "ask_qty": ev[:, 5], "futures": ev[:, 6],
        "px": ev[:, 7], "qty": ev[:, 8], "side": ev[:, 9],
        "venue": "synthetic-fixture",
    }, start_ms, seconds)
