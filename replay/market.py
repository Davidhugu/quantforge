"""`ReplayMarket`: the engine's view of a recorded session.

`run_day` is unchanged and does not know it is reading a recording. It wants,
off the market object: `p` (for `T`), the four series `Sl`/`Fl`/`Il`/`tapel`
plus the stale flags, and an `intents(t, ...)` callback returning
`(side, kind, px)` arrivals. That is the entire contract, and a recorded
session satisfies it exactly.

What is now *observed* rather than generated:

  `Il`   real top-of-book depth imbalance, replacing the latent signal the
         synthetic market derived from its own drift
  `Fl`   real mid of the correlated instrument, replacing `S + 4a + noise`
  `tape` real signed traded volume per second, feeding the same VPIN
  `stale` real gaps in the capture, replacing randomly injected blackouts

What is still modelled, and is the honest limit of this stage:

  fills. We can see the book and the prints, but not our counterparty's intent.
  `FillModel` therefore turns *real* traded volume and *real* traded range into
  arrival counts with a tunable intensity. The driving variables are real; the
  constant is a calibration knob, not a measurement. Queue position -- how much
  size stood ahead of us, the single biggest reason the synthetic market reads
  generous -- is deliberately not modelled here.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, replace

import numpy as np

from .schema import ValidationError
from .store import Dataset


@dataclass
class FillModel:
    """Turns observed second-level trading into client arrivals.

    `per_unit_volume` is the intensity: expected arrivals per share traded in a
    second, scaled down by `participation`. It is the one free parameter, and it
    is the honest weak point of replay -- everything it multiplies is measured,
    the multiplier itself is not.

    `informed_share` labels an arrival informed. In the synthetic market this is
    read off the latent state; in a recording it is *unobservable*, so it is
    drawn from a seeded Bernoulli instead. This affects only how a fill is
    attributed between `mo_b` and `mo_i` in the report, never total PnL. An
    impact-based classifier would be a real improvement and is not written yet.
    """
    per_unit_volume: float = 2e-3
    participation: float = 0.25
    informed_share: float = 0.04
    max_arrivals_per_second: int = 6
    seed: int = 0
    _rng: np.random.Generator = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._rng = np.random.default_rng(self.seed)

    def arrivals(self, second, side, px, n_trades, vol, hi, lo):
        """Client arrivals at our quote `px` during frame `second`.

        A fill requires the price to have been *reachable*: something printed at
        or through it. For a client buying from us at our ask, that means the
        second traded up to at least our ask. A quote nobody traded at cannot
        fill, and that check is what keeps replay from being the synthetic model
        wearing a recording's clothes.
        """
        if vol <= 0 or n_trades <= 0:
            return ()
        if side > 0:
            if not (hi >= px):
                return ()
        else:
            if not (lo <= px):
                return ()
        mean = (self.per_unit_volume * self.participation * min(vol, 200.0)
                * (0.5 + n_trades / 2.0))
        mean = min(mean, self.max_arrivals_per_second)
        if mean <= 0:
            return ()
        n = int(self._rng.poisson(mean))
        out = []
        for _ in range(n):
            kind = "informed" if self._rng.random() < self.informed_share else "benign"
            out.append((side, kind, px))
        return tuple(out)


class ReplayMarket:
    """A recorded session presented as a `flow_mm.Market` lookalike."""

    def __init__(self, ds: Dataset, mp=None, fill: FillModel | None = None,
                 normalize: bool = True):
        self.ds = ds
        self.p = mp
        self.fill = fill or FillModel()
        f = ds.frames
        n = ds.n_seconds

        imbalance = np.asarray(f["imbalance"], dtype=float)
        futures = np.asarray(f["futures"], dtype=float)
        if normalize:
            # The RLS and the Avellaneda-Stoikov reservation price were written
            # against `tanh(...)` inputs and contain hardcoded scalings of that
            # magnitude. Depth imbalance is naturally in [-0.5, 0.5] and much
            # quieter, so it is rescaled to unit sd rather than fed in raw --
            # otherwise the signal layer is handed an input it was never sized
            # for and reports a correlation of ~0 as though it were a result.
            s = imbalance.std()
            if s > 1e-12:
                imbalance = imbalance / (2.0 * s)
            # same for the futures lead: centre it, since only the *relative*
            # move carries information and a drifting level would feed the RLS
            # the price, not the basis
            futures = futures - futures[0]
        self._il = imbalance
        self._fl = futures
        self._tape = ds.signed_volume()

        # `run_day` reads Sl[t+1] and Sl[t+H_SIG], so every series must be one
        # longer than the loop. The final element repeats the last observation,
        # which is what a frozen tape looks like at the close.
        self.Sl = np.append(f["mid"], f["mid"][-1]).tolist()
        self.Fl = np.append(futures, futures[-1]).tolist()
        self.Il = np.append(imbalance, imbalance[-1]).tolist()
        self.tapel = np.append(self._tape, 0.0).tolist()
        self.stalel = np.append(f["stale"], f["stale"][-1]).tolist()
        self.al = [0.0] * (n + 1)

        # traded high/low per frame, for the reachability check
        self._hi = np.full(n, -np.inf)
        self._lo = np.full(n, np.inf)
        self._ntr = np.zeros(n, dtype=np.int64)
        if len(ds.trades["px"]):
            sec = np.clip(ds._sec_of_trade, 0, n - 1)
            np.maximum.at(self._hi, sec, ds.trades["px"])
            np.minimum.at(self._lo, sec, ds.trades["px"])
            self._ntr = np.bincount(sec, minlength=n)

    def __len__(self) -> int:
        return self.ds.n_seconds

    def finalize_params(self, steps: int | None = None):
        """Attach a `MarketParams` sized to this recording, if none was given.

        `steps` drives the engine's loop length, so a caller who forgets to set
        it gets a silent no-op rather than an error. Resolving it here and in
        one place means every call site gets it right.
        """
        if self.p is None:
            raise ValidationError(
                "ReplayMarket needs MarketParams; pass mp= or call "
                "finalize_params() to synthesise one sized to the recording")
        if steps is not None and steps != self.p.steps:
            self.p = replace(self.p, steps=steps)
        return self.p

    def intents(self, t: int, bb, ab, bi, ai):
        """Client arrivals at second `t`, given our four quotes.

        Mirrors the synthetic market's signature and its guard that an arrival
        with no quote is simply not reported: the engine treats `px is None` as
        "we had nothing to show" and skips it.
        """
        if t >= self.ds.n_seconds:
            return ()
        vol = abs(self._tape[t])
        if vol <= 0:
            return ()
        n_trades = int(self._ntr[t])
        hi, lo = self._hi[t], self._lo[t]
        out = []
        for side, px in ((+1, ab), (-1, bb), (+1, ai), (-1, bi)):
            if px is None:
                continue
            out.extend(self.fill.arrivals(t, side, float(px), n_trades, vol, hi, lo))
        return tuple(out)
