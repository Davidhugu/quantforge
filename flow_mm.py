#!/usr/bin/env python3
"""
QuantForge -- flow-aware, inventory-managed market making (research harness)

The module is still flow_mm.py; `quantforge` is the project and CLI name.

Layers
  1. Signals   : order-book imbalance + futures/ETF lead (online RLS -> horizon-matched
                  forward return), VPIN-style flow-toxicity on an *exogenous* tape
  2. Quoting   : Avellaneda-Stoikov reservation price / spread, client-tiered quotes
  3. Inventory : net delta hedged with a liquid proxy (futures/ETF), cost-aware trigger
                  + cooldown
  4. Execution : TWAP / VWAP / adaptive IS algos + epsilon-greedy venue router (RL-lite)
  5. Risk      : pre-trade limits, fat-finger, token-bucket message rate, stale-feed
                  pull, throttle, drawdown kill switch, JSONL audit trail

Usage
  ./flow_mm.py backtest [--days 30] [--fee -0.0005] [--audit audit_log.jsonl]
                        [--plot equity.png]
  ./flow_mm.py exec [--exec-n 3000]
  ./flow_mm.py sweep [--days 10] [--out sweep.csv]
  ./flow_mm.py all

  The module keeps its filename, so the entry point is flow_mm.py while the
  project and the --help prog are `quantforge`.

WHAT THIS IS NOT
  The market is SYNTHETIC (latent drift, informed + benign client flow,
  intensity-based fills, no queue position, no exchange latency, no real
  adverse-selection dynamics). See README.md "Known limitations" for the full
  list. It is a mechanics lab, NOT evidence of live profitability.

WHY THE CALIBRATION MATTERS
  `calibration()` prints the derived market statistics. The original defaults put
  latent drift at 0.24% of price variance, which made BOTH the signal layer and
  the adverse-selection layer unmeasurable: with a perfect read on the state you
  could explain <4% of 30-second returns, so no quoting skill and no toxicity
  filter could ever be validated. `alpha_std` is now set so that ~20% of 30-second
  return variance is predictable, which is the only regime in which the ablation
  ladder below measures anything. Every one of those knobs is swept in `sweep`.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from collections import deque
from dataclasses import dataclass, field, replace

import numpy as np

from stats import paired, summarize

TICK, LOT = 0.01, 100
SESSION = 23_400  # seconds in a US equity session
H_SIG = 30  # signal / holding / markout horizon, seconds
TAPE_MAX = 8  # max background arrivals modelled per second

REAL_EXCHANGE_FEE_BPS = 0.3  # ~0.3 bp of notional, for the breakeven comparison


def bps_per_share(fee_ps: float, px: float) -> float:
    """$/share -> basis points of notional, i.e. `fee_ps / px * 1e4`.

    `exec_algos.parent_order` divides by the reference price for exactly this
    reason. Multiplying by 1e4 alone treats a dollar as the unit of account and
    reports a $100 stock's fee 100x too large, which is how `--fee 0.003` -- a
    realistic 0.3 bp -- was printed as 30 bp.
    """
    return fee_ps / px * 1e4


# ============================================================ calibration diagnostics
def calibration(p: "MarketParams", ds=None) -> dict:
    """Derived statistics that decide whether the sim can test anything.

    `predictable_share` is the fraction of H-step return variance explained by
    the best possible state read. If this is ~0, the signal layer is untestable;
    if the informed edge is dwarfed by the spread, there is no adverse
    selection to defend against. Both were broken in the original defaults.

    With `ds` (a replayed recording) the volatility is measured from the data
    instead of read off `sigma_ann`, and the two state-dependent statistics are
    reported as None. A recording has no latent drift and no `alpha_*` to it,
    so those numbers are not "small" -- they are undefined, and printing a
    plausible-looking figure for them next to a real-data PnL is how a result
    ends up quietly meaning nothing.
    """
    h = p.informed_horizon
    if ds is not None:
        mid = np.asarray(ds.frames["mid"], dtype=float)
        r = np.diff(np.log(mid[mid > 0])) if len(mid) > 1 else np.zeros(1)
        sig = float(r.std()) if len(r) else 0.0
        return dict(sig=sig, var=sig ** 2, pred_h=None, noise_h=None,
                    predictable_share=None, edge_sd=None, source=ds.meta.get("venue", "replay"))

    sig = p.sigma_ann * p.s0 / math.sqrt(252 * p.steps)
    gain = sum(p.alpha_phi ** j for j in range(h))  # E[a_{t+h} | a_t], summed
    pred = gain * p.alpha_std  # predictable component of the H-step return
    noise = math.sqrt(h) * sig  # diffusion over the same horizon
    share = pred ** 2 / (pred ** 2 + noise ** 2) if (pred or noise) else 0.0
    # the informed trader's expected edge if fully filled at the touch
    return dict(sig=sig, var=sig ** 2, pred_h=pred, noise_h=noise,
                predictable_share=share, edge_sd=pred, source="synthetic-market")


# ============================================================ synthetic market
@dataclass
class MarketParams:
    s0: float = 100.0
    sigma_ann: float = 0.25
    steps: int = SESSION
    A: float = 0.25              # benign arrivals /side/sec at zero quote distance
    k: float = 100.0             # arrival decay per $ of distance from mid
    alpha_std: float = 1.2e-3    # latent drift, $/sec. RECALIBRATED from 5e-4
    alpha_phi: float = 0.98
    informed_rate: float = 0.04  # informed arrivals /sec. RECALIBRATED from 0.30,
                               # which made informed flow 40-57% of all fills
    informed_horizon: int = H_SIG
    label_acc: float = 0.85      # P(informed client is tagged informed by CRM)
    mislabel: float = 0.05       # share of benign flow wrongly tagged informed
    blackout_prob: float = 2e-4  # stale-feed episodes
    blackout_len: int = 5
    intraday_k: float = 1.5      # U-shape amplitude: ~2.5x at the open/close
    tape_rate: float = 1.2       # exogenous market-wide arrivals /sec (midday)
    tape_beta: float = 0.35      # how strongly imbalance drives the tape's sign


class Market:
    """Synthetic venue. One instance per (params, seed); reused across ladder rungs."""

    def __init__(self, p: MarketParams, seed: int):
        self.p, self.seed = p, seed
        rng = np.random.default_rng(seed)
        T = p.steps

        # --- latent state -----------------------------------------------------
        ea, ev, ez, eb = (rng.standard_normal(T + 1) for _ in range(4))
        sa = p.alpha_std * math.sqrt(1 - p.alpha_phi ** 2)
        a = np.empty(T + 1)
        a[0] = p.alpha_std * ea[0]
        for t in range(1, T + 1):
            a[t] = p.alpha_phi * a[t - 1] + sa * ea[t]
        lv = np.empty(T + 1)
        lv[0] = 0.0
        for t in range(1, T + 1):
            lv[t] = 0.999 * lv[t - 1] + 0.01 * ev[t]
        b = np.empty(T + 1)
        b[0] = 0.0
        for t in range(1, T + 1):
            b[t] = 0.95 * b[t - 1] + 0.003 * eb[t]

        # S is a cumulative sum of drift + stochastic vol -> vectorise it
        S = np.empty(T + 1)
        S[0] = p.s0
        S[1:] = p.s0 + np.cumsum(a[:T] + p.sigma_ann * p.s0 / math.sqrt(252 * T)
                                 * np.exp(lv[:T]) * ez[:T])
        self.S, self.a = S, a
        # normalised imbalance. With alpha_std=0 there is no latent drift to
        # express, so the signal term drops out and only the noise remains --
        # this is the sweep's honest "no exploitable signal" anchor.
        sig_term = (0.6 * a / p.alpha_std if p.alpha_std > 0
                    else np.zeros(T + 1))
        self.I = np.tanh(sig_term + 0.7 * rng.standard_normal(T + 1))
        self.F = S + 4 * a + b  # futures proxy
        self.U = rng.random((T, 6))

        # --- intraday liquidity shape (normalised so the mean multiplier is 1)
        u = np.arange(T) / T
        self.prof = 1.0 + p.intraday_k * (2 * u - 1) ** 2
        self.prof /= self.prof.mean()        # exact on the grid, not just in

        # --- exogenous background tape ---------------------------------------
        # VPIN must read market-wide flow, not our own fills. Feeding Toxicity
        # our fills makes the toxicity estimate a function of our own quoting.
        cnt = np.minimum(rng.poisson(p.tape_rate * self.prof), TAPE_MAX).astype(int)
        qu = rng.random((T, TAPE_MAX))
        qbuy = 0.5 + p.tape_beta * np.tanh(self.I[:T])
        tape = np.zeros(T)
        for j in range(TAPE_MAX):
            tape += np.where(cnt > j, np.where(qu[:, j] < qbuy, float(LOT), -float(LOT)), 0.0)

        # --- stale-feed episodes (clipped so they cannot overrun the session)
        self.stale = np.zeros(T, bool)
        for t in np.flatnonzero(rng.random(T) < p.blackout_prob):
            self.stale[t:min(t + p.blackout_len, T)] = True

        # python lists: the engine loop indexes these ~7x per step, and numpy
        # scalar indexing costs more than the arithmetic around it
        self.al = self.a.tolist()
        self.Sl, self.Fl, self.Il = S.tolist(), self.F.tolist(), self.I.tolist()
        self.tapel, self.profl, self.stalel = tape.tolist(), self.prof.tolist(), self.stale.tolist()
        self.Ul = self.U.tolist()   # indexing a numpy row costs more than the comparison

    def intents(self, t: int, bb, ab, bi, ai) -> list:
        """Every client arrival at second `t`, and what we would charge for it.

        bb/ab and bi/ai are our bid/ask for the benign and informed client
        tiers; either side may be None (not quoted). Returns (side, kind, px).

        An arrival is only reported when we have a quote to show the client:
        every branch is guarded on the price being not None, so a second with no
        quotes returns []. The caller therefore never sees an unquoted arrival
        here, and the toxicity filter is fed from the exogenous tape (see
        `Market.tapel`) rather than from anything here -- feeding it our fills
        would make the estimate a function of our own quoting.
        """
        p, S, u, prof = self.p, self.Sl[t], self.Ul[t], self.profl[t]
        A, k = p.A * prof, p.k
        out = []
        # benign, buy side (our ask): tiers b then i, weighted by the mislabel rate
        if ab is not None and u[1] < 1 - math.exp(-A * (1 - p.mislabel) * math.exp(-k * (ab - S))):
            out.append((+1, "benign", ab))
        if ai is not None and u[3] < 1 - math.exp(-A * p.mislabel * math.exp(-k * (ai - S))):
            out.append((+1, "benign", ai))
        if bb is not None and u[0] < 1 - math.exp(-A * (1 - p.mislabel) * math.exp(-k * (S - bb))):
            out.append((-1, "benign", bb))
        if bi is not None and u[2] < 1 - math.exp(-A * p.mislabel * math.exp(-k * (S - bi))):
            out.append((-1, "benign", bi))
        if u[4] < 1 - math.exp(-p.informed_rate * prof):
            a = self.a[t]
            side = 1 if a > 0 else -1
            px = (ai if u[5] < p.label_acc else ab) if side > 0 else (bi if u[5] < p.label_acc else bb)
            if px is not None and side * (a * p.informed_horizon - (px - S)) > 0:
                out.append((side, "informed", px))
        return out


# ============================================================ signal layer
class RLS:
    """Recursive least squares with exponential forgetting, in pure Python.

    Maps features -> the H-step forward return. Deliberately general in n (the
    production path is n=2) so it can be unit-tested against a known solution:
    this runs 3.5M times per 30-day backtest, where numpy's per-call overhead
    exceeds the arithmetic by roughly 50x.
    """

    def __init__(self, n: int, lam: float = 0.9998, delta: float = 1e3):
        self.n, self.lam = n, lam
        self.w = [0.0] * n
        self.P = [delta if i == j else 0.0 for i in range(n) for j in range(n)]

    def predict(self, x) -> float:
        w = self.w
        if self.n == 2:
            return w[0] * x[0] + w[1] * x[1]
        return sum(w[i] * x[i] for i in range(self.n))

    def update(self, x, y: float) -> None:
        P, w, n = self.P, self.w, self.n
        if n == 2:
            x0, x1 = x[0], x[1]
            Px0 = P[0] * x0 + P[1] * x1
            Px1 = P[2] * x0 + P[3] * x1
            d = self.lam + x0 * Px0 + x1 * Px1
            g0, g1 = Px0 / d, Px1 / d
            r = y - (w[0] * x0 + w[1] * x1)
            w[0] += g0 * r
            w[1] += g1 * r
            il = 1.0 / self.lam
            P[0] = (P[0] - g0 * Px0) * il
            P[1] = (P[1] - g0 * Px1) * il
            P[2] = (P[2] - g1 * Px0) * il
            P[3] = (P[3] - g1 * Px1) * il
            return
        Px = [0.0] * n
        for i in range(n):
            Pi, xi, acc = P[i * n:(i + 1) * n], x[i], 0.0
            for j in range(n):
                acc += Pi[j] * x[j]
            Px[i] = acc
        d = self.lam + sum(x[i] * Px[i] for i in range(n))
        yh = sum(w[i] * x[i] for i in range(n))
        for i in range(n):
            g = Px[i] / d
            w[i] += g * (y - yh)
        # in place: `P[i*n:(i+1)*n]` is a *copy*, so assigning into it is a
        # no-op. (cost is irrelevant here -- n=2 takes the branch above)
        Pn = [0.0] * (n * n)
        for i in range(n):
            base_i = i * n
            for j in range(n):
                Pn[base_i + j] = (P[base_i + j] - (Px[i] / d) * Px[j]) / self.lam
        P[:] = Pn


class Toxicity:
    """VPIN-style: mean |buy-sell|/volume over recent equal-volume buckets.

    Fed the exogenous market tape (`Market.tapel`, consumed in run_day), not our
    fills.
    """

    def __init__(self, bucket: int = 1000, n: int = 20, warm: int = 5):
        self.b, self.buy, self.sell, self.h = bucket, 0, 0, deque(maxlen=n)
        self.tot, self.n = 0.0, 0
        self.warm = warm

    def on_trade(self, signed: float) -> None:
        if signed > 0:
            self.buy += signed
        else:
            self.sell -= signed
        v = self.buy + self.sell
        if v >= self.b:
            x = abs(self.buy - self.sell) / v
            if len(self.h) == self.h.maxlen:
                self.tot -= self.h[0]
            else:
                self.n += 1
            self.h.append(x)
            self.tot += x
            self.buy = self.sell = 0

    def value(self) -> float:
        return self.tot / self.n if self.n >= self.warm else 0.0


# ============================================================ risk & compliance
class Audit:
    """Per-run JSONL trail with a real wall clock plus a sim-time field.

    Opened with "w", so each run starts the file clean rather than appending to
    a previous one. Records are self-identifying via `run`, and a run that
    doubled up on one path would interleave two markets in one file, which is
    worse than losing the old copy.
    """

    def __init__(self, path: str, run_id: str):
        self.f = open(path, "w")
        self.run_id = run_id

    def log(self, t: int, ev: str, **kw) -> None:
        self.f.write(json.dumps({"wall_ns": time.time_ns(), "run": self.run_id,
                                 "sim_s": t, "ev": ev, **kw}) + "\n")

    def close(self) -> None:
        self.f.close()


class TokenBucket:
    """Message-rate governor. The old counter compared a 1-second delta against
    a per-second cap, which a 1-update-per-second strategy can never breach.

    There is deliberately no `take()` here: the hot path in `Engine.quotes`
    inlines the same refill-and-spend so it can also count only *published*
    sends toward `peak` and republish the held book when the bucket is dry. A
    second entry point would have to duplicate that, and the two would drift.
    """

    def __init__(self, rate: float, burst: float):
        self.rate, self.burst, self.tok, self.t = rate, burst, burst, 0.0
        self.peak = 0.0


@dataclass
class Cfg:
    name: str
    skew: bool = True
    signal: bool = True
    tiering: bool = True
    toxic: bool = True
    hedge: bool = True
    naive_half: float = 0.0                      # >0 => fixed half-spread, no A-S maths
    gamma: float = 0.5
    tau: float = 60.0
    k: float = 100.0
    alpha_h: float = 1.0                        # $ of skew per $ of predicted H-step move
    alpha_clip: float = 4.0                     # clip at N x the running sd of the estimate
    benign_mult: float = 0.85
    informed_mult: float = 1.6
    tox_coef: float = 0.03
    tox_floor: float = 0.30
    fee_ps: float = 0.0                         # $/share, positive = cost
    max_pos: int = 1500
    max_notional: float = 250_000
    fat_finger_bps: float = 50
    max_msgs_per_sec: float = 10.0
    hedge_trigger: int = 500
    hedge_target: int = 100
    hedge_cooldown: int = 10
    hedge_cost: float = 0.004
    throttle_dd: float = 3000
    kill_dd: float = 6000
    flatten_cost: float = 0.02


class Risk:
    def __init__(self, c: Cfg, audit: Audit | None):
        self.c, self.audit, self.peak = c, audit, 0.0
        self.halted = self.throttled = False
        self.bucket = TokenBucket(c.max_msgs_per_sec, 2 * c.max_msgs_per_sec)
        self.rej = {"pos_limit": 0, "notional": 0, "fat_finger": 0, "msg_rate": 0, "stale_feed": 0}

    def _reject(self, t: int, why: str, **kw) -> None:
        self.rej[why] += 1
        if self.audit and self.rej[why] in (1, 100, 1000):
            self.audit.log(t, "risk_reject", reason=why, n=self.rej[why], **kw)

    def approve_pair(self, t: int, q, ref: float, inv: int) -> tuple:
        """Approve all four quotes in one pass. side=-1 is our bid (inventory
        rises), +1 is our ask. Every limit blocks only orders that grow exposure,
        so a book at its cap can always reduce."""
        c, ff = self.c, self.c.fat_finger_bps * 1e-4 * ref
        out = []
        for side, px in ((-1, q[0]), (+1, q[1]), (-1, q[2]), (+1, q[3])):
            if px is None:
                out.append(None)
                continue
            new = inv - side * LOT
            if abs(new) > c.max_pos and abs(new) > abs(inv):
                self._reject(t, "pos_limit", inv=inv)
                out.append(None)
            elif abs(new) * ref > c.max_notional and abs(new) > abs(inv):
                self._reject(t, "notional", inv=inv)
                out.append(None)
            elif abs(px - ref) > ff + 1e-9:
                self._reject(t, "fat_finger", px=px)
                out.append(None)
            else:
                out.append(px)
        return tuple(out)

    def approve(self, t: int, side: int, px: float, ref: float, inv: int) -> bool:
        """Single-quote approval, kept for tests and for callers that quote one side."""
        c = self.c
        new = inv - side * LOT
        if abs(new) > c.max_pos and abs(new) > abs(inv):
            self._reject(t, "pos_limit", inv=inv)
            return False
        if abs(new) * ref > c.max_notional and abs(new) > abs(inv):
            self._reject(t, "notional", inv=inv)
            return False
        if abs(px - ref) / ref * 1e4 > c.fat_finger_bps + 1e-9:
            self._reject(t, "fat_finger", px=px)
            return False
        return True

    def on_equity(self, t: int, eq: float):
        self.peak = max(self.peak, eq)
        dd = self.peak - eq
        was = self.throttled
        self.throttled = dd > self.c.throttle_dd
        if self.throttled != was and self.audit:
            self.audit.log(t, "throttle", on=self.throttled, dd=round(dd, 2))
        if dd > self.c.kill_dd and not self.halted:
            self.halted = True
            if self.audit:
                self.audit.log(t, "KILL_SWITCH", dd=round(dd, 2))
            return "kill"
        return None


# ============================================================ quoting + hedging engine
FEAT_N = 2                       # (imbalance, futures lead) -- see run_day


class Engine:
    def __init__(self, c: Cfg, mp: MarketParams, audit: Audit | None = None):
        self.c, self.audit = c, audit
        self.rls, self.tox, self.risk = RLS(FEAT_N), Toxicity(), Risk(c, audit)
        self.var = (mp.sigma_ann * mp.s0) ** 2 / (252 * mp.steps)
        self.prevS, self.last_hedge = None, -10 ** 9
        self.live = None
        self.msgs = 0
        self.alpha_mu, self.alpha_var, self.alpha_last = 0.0, 1e-12, 0.0
        self.buf: deque = deque()  # (I, lead, S) delayed by the holding horizon
        # invariant across the session; recomputing these per step was ~15% of
        # the whole backtest. (the toxicity rail is bounded by tox_floor, so
        # its max is c.tox_coef * (1 - c.tox_floor) -- not worth caching, it is
        # read once per side and the spread maths around it dominates)
        self._kterm = math.log(1 + c.gamma / c.k) / c.gamma
        self._vterm = 0.5 * c.gamma * c.tau
        self._skew = c.gamma * c.tau

    def _alpha(self, I: float, F: float, S: float) -> float:
        """Predicted H-step return, clipped at `alpha_clip` running sigmas.

        The clip is measured against the estimate's own EWMA sd, so the rail
        actually binds on a bad day. The original clipped at a fixed 3e-3 that
        sat 29 sigma out and could never fire.
        """
        a = self.rls.predict((I, (F - S) * 50.0)) if self.c.signal else 0.0
        self.alpha_last = a
        mu = 0.999 * self.alpha_mu + 0.001 * a
        d = a - mu
        self.alpha_var = 0.999 * self.alpha_var + 0.001 * d * d
        self.alpha_mu = mu
        lim = self.c.alpha_clip * (math.sqrt(self.alpha_var) if self.alpha_var > 1e-18 else 1e-9)
        return lim if a > lim else (-lim if a < -lim else a)

    def learn(self, I: float, lead: float, S_now: float) -> None:
        """Learn only once the horizon has elapsed, pairing the features seen at
        t-H with the return actually realised over [t-H, t].

        The original learned on the one-step change, which is diffusion-dominated
        (latent drift is a fraction of a percent of a one-second move) while the
        quote it steers is held for ~H seconds. Features and target have to share
        a horizon or the estimate is not a forecast of anything we can act on.
        """
        if not self.c.signal:
            return
        buf = self.buf
        buf.append((I, lead, S_now))
        if len(buf) > H_SIG:
            I0, lead0, S0 = buf.popleft()
            self.rls.update((I0, lead0), S_now - S0)

    def quotes(self, t: int, S: float, I: float, F: float, inv: int, net: int):
        c, r = self.c, self.risk
        if r.halted:
            return (None, None, None, None)
        if c.naive_half:
            res, hb, hi = S, c.naive_half, c.naive_half
        else:
            var = self.var if self.var > 1e-7 else 1e-7
            res = S - (net if c.skew else 0) / LOT * self._skew * var + self._alpha(I, F, S) * c.alpha_h
            half = self._vterm * var + self._kterm
            if c.tiering:
                hb, hi = half * c.benign_mult, half * c.informed_mult
            else:
                hb = hi = half
            if c.toxic:
                tox = self.tox.value() - c.tox_floor
                if tox > 0.0:
                    tox *= c.tox_coef
                    hi += tox
                    hb += tox * (0.3 if c.tiering else 1.0)
        if r.throttled:
            hb *= 1.5
            hi *= 1.5
        raw = (math.floor((res - hb if res - hb < S - 0.5 * TICK else S - 0.5 * TICK) / TICK) * TICK,
               math.ceil((res + hb if res + hb > S + 0.5 * TICK else S + 0.5 * TICK) / TICK) * TICK,
               math.floor((res - hi if res - hi < S - 0.5 * TICK else S - 0.5 * TICK) / TICK) * TICK,
               math.ceil((res + hi if res + hi > S + 0.5 * TICK else S + 0.5 * TICK) / TICK) * TICK)
        # risk fast path: when the book is inside both caps and the widest quote
        # is inside the fat-finger band, no limit can fire and we skip the
        # per-side audit entirely (that path is ~99% of seconds)
        ff = c.fat_finger_bps * 1e-4 * S
        room = abs(inv) + LOT
        if (room <= c.max_pos and room * S <= c.max_notional
                and raw[2] > S - ff - 1e-9 and raw[3] < S + ff + 1e-9):
            out = raw
        else:
            out = r.approve_pair(t, raw, S, inv)
        live = self.live
        if live is None:
            n = 0
        else:
            n = ((out[0] != live[0]) + (out[1] != live[1])
                 + (out[2] != live[2]) + (out[3] != live[3]))
        if n:
            b = r.bucket
            b.tok += (t - b.t) * b.rate
            b.t = t
            if b.tok > b.burst:
                b.tok = b.burst
            if b.tok < n:
                out = live                       # rate-limited: hold prior quotes
            else:
                b.tok -= n
                self.msgs += n
                if n > b.peak:
                    b.peak = n                  # published, not merely attempted
        self.live = out
        return out

# ============================================================ backtest
def run_day(m: Market, c: Cfg, audit: Audit | None = None) -> dict:
    """One session against a prebuilt market (shared across ladder rungs).

    PnL is reported as an exact additive decomposition rather than one number:

        pnl = spread + inv_drift + hedge_pnl - fees - hedge_cost - flatten

    which follows from summation by parts on the two inventory books and is
    asserted in tests/test_engine.py. `adverse` below is a memo line, not a term
    in that identity: it is the H-step move in the mid after each fill, signed
    so that positive means the market went against us (a client who bought from
    us and then saw the mid rise). It is the NEGATION of the per-fill
    contribution to `inv_drift`, and the sign-flip of the markout -- per fill
    `mo == spread_c - adverse_c`, since (px - S) - (far - S) == (px - far). It
    is a diagnostic on the same events, not a sub-view of `inv_drift` itself, and
    it is not additive. Mind the units: `spread` and `adverse` are dollars, but
    the reported `mo_b`/`mo_i` are basis points per share.
    """
    T = m.p.steps
    e = Engine(c, m.p, audit)
    Sl, Fl, Il, tapel, stalel = m.Sl, m.Fl, m.Il, m.tapel, m.stalel
    cash, inv, hed = 0.0, 0, 0

    spread = inv_drift = hedge_pnl = fees = hedge_cost = flatten = 0.0
    adverse = 0.0
    absinv = [0.0] * T
    curve = [0.0] * (T + 1)
    n_ben = n_inf = 0
    mo_b = mo_i = 0.0
    eod = False
    awin, n_a = deque(maxlen=H_SIG + 1), 0
    sa = sy = saa = say = syy = so = soo = soy = 0.0
    gain = sum(m.p.alpha_phi ** j for j in range(H_SIG))
    tox = e.tox
    tlast = -10 ** 9

    for t in range(T):
        S = Sl[t]
        if e.prevS is not None:
            e.var = 0.995 * e.var + 0.005 * (S - e.prevS) ** 2
        e.prevS = S

        if stalel[t]:
            e.risk._reject(t, "stale_feed")
            e.live = None                        # pull quotes on stale data
            q = (None, None, None, None)
        elif e.risk.halted:
            q = (None, None, None, None)
        else:
            q = e.quotes(t, S, Il[t], Fl[t], inv, inv + hed)
        tv = tapel[t]
        if tv:
            tox.on_trade(tv)                      # exogenous tape -> VPIN

        for side, kind, px in m.intents(t, q[0], q[1], q[2], q[3]):
            if px is None:                        # client arrived; we had no quote
                continue
            inv -= side * LOT
            cash += (side * px - c.fee_ps) * LOT
            spread += side * (px - S) * LOT
            far = Sl[t + H_SIG if t + H_SIG < T else T]
            adverse += side * (far - S) * LOT
            mo = side * (px - far) * LOT
            fees += c.fee_ps * LOT
            if kind == "informed":
                n_inf += 1
                mo_i += mo
            else:
                n_ben += 1
                mo_b += mo
            if audit:
                audit.log(t, "client_fill", side="BUY" if side > 0 else "SELL",
                          kind=kind, px=round(px, 3), qty=LOT, inv=inv,
                          slip_vs_mid=round(side * (px - S), 4))

        # net-delta hedge in the futures proxy, cost- and cooldown-aware
        if c.hedge and not e.risk.halted:
            net = inv + hed
            if abs(net) >= c.hedge_trigger and t - tlast >= c.hedge_cooldown:
                h = round(-(net - math.copysign(c.hedge_target, net)) / LOT) * LOT
                if h:
                    tlast = t
                    hpx = Fl[t] + (c.hedge_cost if h > 0 else -c.hedge_cost)
                    hed += h
                    cash -= h * hpx
                    hedge_cost += abs(h) * c.hedge_cost
                    if audit:
                        audit.log(t, "hedge", qty=h, px=round(hpx, 3), net_delta=inv + hed)

        S1, F1 = Sl[t + 1], Fl[t + 1]
        inv_drift += inv * (S1 - S)
        hedge_pnl += hed * (F1 - Fl[t])
        absinv[t] = abs(inv)
        eq = cash + inv * S1 + hed * F1
        curve[t + 1] = eq
        if e.risk.on_equity(t, eq) == "kill":                # flatten + halt
            flatten = c.flatten_cost * (abs(inv) + abs(hed))
            cash += inv * S1 + hed * F1 - flatten
            inv = hed = 0
            # from t+1, not t+2: the liquidation happens AT t, so curve[t+1] is
            # the post-flatten cash. Backfilling from t+2 left the flatten cost
            # showing up as a one-second crash with no event behind it, and when
            # the kill fired on the final step (range(t+2, T+1) empty) dropped
            # the cost from the curve entirely.
            for u in range(t + 1, T + 1):
                curve[u] = cash
            break
        e.learn(Il[t], (Fl[t] - S) * 50.0, S)
        if c.signal:                        # does the estimate predict anything?
            awin.append((e.alpha_last, S))
            if len(awin) > H_SIG:
                a0, S0 = awin.popleft()
                y = S - S0
                # The state must be read at the START of the window. `a0` was
                # predicted at S0 = Sl[t-H_SIG], so the best forecast available
                # then is a[t-H_SIG]*gain; using a[t] is a look-ahead of the full
                # horizon and reported a ceiling below the one that actually
                # bounds the estimator. gain is the same sum calibration() uses.
                o = m.al[t - H_SIG] * gain
                n_a += 1
                sa += a0; sy += y; saa += a0 * a0; say += a0 * y; syy += y * y
                so += o; soo += o * o; soy += o * y
    else:
        eod = True
        flatten = c.flatten_cost * (abs(inv) + abs(hed))
        cash += inv * Sl[-1] + hed * Fl[-1] - flatten
        curve[-1] = cash

    cv = np.asarray(curve)
    # correlation between the predicted H-step return and the realised one:
    # the single number that says whether the signal layer has any value
    den = math.sqrt(max(0.0, n_a * saa - sa * sa) * max(0.0, n_a * syy - sy * sy))
    alpha_corr = (n_a * say - sa * sy) / den if den > 0 else 0.0
    den_o = math.sqrt(max(0.0, n_a * soo - so * so) * max(0.0, n_a * syy - sy * sy))
    oracle_corr = (n_a * soy - so * sy) / den_o if den_o > 0 else 0.0
    dd = float(np.max(np.maximum.accumulate(cv) - cv))
    n = n_ben + n_inf
    return dict(
        pnl=spread + inv_drift + hedge_pnl - fees - hedge_cost - flatten,
        spread=spread, inv_drift=inv_drift, hedge_pnl=hedge_pnl, adverse=adverse,
        fees=fees, hedge_cost=hedge_cost, flatten=flatten, dd=dd,
        fills=n, n_ben=n_ben, n_inf=n_inf,
        inf_share=n_inf / n if n else 0.0,
        mo_b=mo_b / (n_ben * LOT) * 100 if n_ben else 0.0,
        mo_i=mo_i / (n_inf * LOT) * 100 if n_inf else 0.0,
        avg_inv=sum(absinv) / T, max_inv=int(max(absinv)),
        msgs=e.msgs, peak_rate=e.risk.bucket.peak, tox_mean=e.tox.value(),
        alpha_corr=alpha_corr, oracle_corr=oracle_corr,
        rej=e.risk.rej, killed=e.risk.halted, eod=eod, curve=cv)


LADDER = [
    Cfg("naive 1-tick", naive_half=TICK, skew=False, signal=False, tiering=False,
        toxic=False, hedge=False),
    Cfg("+ A-S skew/spread", signal=False, tiering=False, toxic=False, hedge=False),
    Cfg("+ alpha signal", tiering=False, toxic=False, hedge=False),
    Cfg("+ tiering & toxicity", hedge=False),
    Cfg("+ hedging (full)"),
]

_NUMERIC = ("pnl", "spread", "inv_drift", "hedge_pnl", "adverse", "fees", "hedge_cost",
            "flatten", "dd", "fills", "n_ben", "n_inf", "inf_share", "mo_b", "mo_i",
            "avg_inv", "max_inv", "msgs", "peak_rate", "tox_mean", "alpha_corr",
            "oracle_corr")


def _mean_agg(rows: list) -> dict:
    out = {k: float(np.mean([r[k] for r in rows])) for k in _NUMERIC}
    out["killed"] = sum(r["killed"] for r in rows)
    out["eod"] = sum(r["eod"] for r in rows)
    out["rej"] = {k: float(np.mean([r["rej"][k] for r in rows])) for k in rows[0]["rej"]}
    return out


def _day_rows(mp: MarketParams, ladder, seed: int, fee: float, audit_path=None) -> list:
    """All ladder rungs for one seed. One market build, reused across rungs --
    that alone was a 5x saving over the original, which rebuilt it per rung."""
    m = Market(mp, seed)
    out = []
    for i, c in enumerate(ladder):
        au = (Audit(audit_path, f"{c.name}|seed{seed}")
              if audit_path and seed == 0 and i == len(ladder) - 1 else None)
        out.append(run_day(m, replace(c, fee_ps=fee), au))
        if au:
            au.close()
    return out


def backtest(days: int = 30, audit_path: str | None = None, plot: str | None = None,
             mp: MarketParams | None = None, cfg_fee: float = 0.0,
             ladder=None, quiet: bool = False, markets=None, workers: int = 1) -> dict:
    mp = mp or MarketParams()
    ladder = list(ladder or LADDER)
    # days=0 used to split on `workers`: the serial path raised IndexError, while
    # the parallel path ran seed 0 anyway and reported a full ladder under a
    # "0 paired sessions" header. A session count of zero is a mistake, and
    # returning data for it is worse than refusing.
    if days < 1:
        raise ValueError(f"days must be >= 1, got {days}")

    if markets is not None or workers == 1:
        mkts = markets or [Market(mp, s) for s in range(days)]
        rows_by_cfg = []
        for c in ladder:
            rows = []
            for s, m in enumerate(mkts):
                au = (Audit(audit_path, f"{c.name}|seed{s}")
                      if audit_path and s == 0 and c is ladder[-1] else None)
                rows.append(run_day(m, replace(c, fee_ps=cfg_fee), au))
                if au:
                    au.close()
            rows_by_cfg.append(rows)
    else:
        # Parallelise. Two regimes, because a task is one (seed, rung-set) and
        # the total task count decides how well the pool is used:
        #   * plenty of seeds  -> one task per seed, each reusing one market
        #     build across all rungs (a build is ~5% of a seed's cost)
        #   * few seeds        -> split rungs too, trading ~20% of run time for
        #     up to 5x the task count. A run of 4 seeds on 3 workers would
        #     otherwise leave the pool nearly idle.
        # Seed 0 always runs in the parent so the audit trail is written here.
        nr = len(ladder)
        flat = days < 2 * workers
        head = _day_rows(mp, ladder, 0, cfg_fee, audit_path)   # parent: audit trail
        # (seed, rung_index) pairs; ex.map yields in submission order, so the
        # output index is recoverable without carrying the key back
        if flat:
            tasks = [(mp, [ladder[i]], s, cfg_fee, None)
                     for s in range(1, days) for i in range(nr)]
        else:
            tasks = [(mp, ladder, s, cfg_fee, None) for s in range(1, days)]
        slots = [[head[i]] for i in range(nr)]    # seed 0 already in each rung
        if tasks:
            from concurrent.futures import ProcessPoolExecutor
            with ProcessPoolExecutor(max_workers=workers) as ex:
                for k, res in enumerate(ex.map(_day_rows_args, tasks)):
                    if flat:
                        slots[k % nr].extend(res)          # one rung per task
                    else:
                        for i, r in enumerate(res):
                            slots[i].append(r)       # res[i] is one day's dict
        rows_by_cfg = slots

    # only the serial path builds markets in-process; the parallel path
    # reconstructs them per task, so a replayed run has no single dataset to
    # measure calibration from and falls back to the declared parameters
    cal = calibration(mp, ds=getattr(mkts[0], "ds", None)
                      if (markets is not None and workers == 1 and mkts) else None)
    agg, pnl_by_cfg, curves = {}, {}, {}
    for c, rows in zip(ladder, rows_by_cfg):
        agg[c.name] = _mean_agg(rows)
        pnl_by_cfg[c.name] = [r["pnl"] for r in rows]
        curves[c.name] = [r["curve"] for r in rows]
        agg[c.name]["curve"] = np.mean(np.array(curves[c.name]), axis=0)

    if not quiet:
        _report(cal, mp, ladder, pnl_by_cfg, agg, days, cfg_fee, audit_path,
                {c.name: [r["dd"] for r in rows]
                 for c, rows in zip(ladder, rows_by_cfg)})
    if plot:
        _plot(plot, {c.name: [agg[c.name]["curve"]] for c in ladder}, days)
    return dict(cal=cal, pnl=pnl_by_cfg, agg=agg, curves=curves)


def _day_rows_args(a):
    return _day_rows(*a)


def _report(cal, mp, ladder, pnl_by_cfg, agg, days, cfg_fee, audit_path,
            dd_rows=None) -> None:
    dd_rows = dd_rows or {}
    src = cal.get("source", "synthetic-market")
    replay = src != "synthetic-market"
    print(f"\n{'Replayed-data' if replay else 'Synthetic-market'} ablation: {days} paired "
          f"sessions{f' from {src}' if replay else ''}, 1 lot = {LOT} sh, "
          f"1 tick = ${TICK}, fee = {bps_per_share(cfg_fee, mp.s0):+.2f} bps/share")
    na = "n/a"
    ps = f"{cal['predictable_share'] * 100:.1f}%" if cal["predictable_share"] is not None else na
    ph = f"{cal['pred_h'] * 100:.2f}c" if cal["pred_h"] is not None else na
    print(f"  derived: sigma/s = {cal['sig'] * 1e4:.1f} bps"
          f"{' (measured from the recording)' if replay else ''} | predictable share of "
          f"the {mp.informed_horizon}s return = {ps} | H-step drift sd = {ph}\n")
    if replay:
        print("  predictable share and H-step drift are undefined for a recording: there\n"
              "  is no latent state to hand an oracle. The alpha correlation below is\n"
              "  still measured -- it is the only honest test of the signal layer.\n")
    hdr = (f"{'config':<22}{'PnL/day $':>10}{'sem':>8}{'maxDD $':>9}{'PnL/DD':>8}"
           f"{'avg|inv|':>9}"
           f"{'fills':>7}{'inf%':>6}{'spread$':>9}{'invDrift$':>10}{'hedge$':>8}"
           f"{'fees$':>7}{'pkMsg':>6}{'kill':>5}")
    print(hdr)
    print("-" * len(hdr))
    for c in ladder:
        a, s = agg[c.name], summarize(pnl_by_cfg[c.name])
        print(f"{c.name:<22}{s['mean']:>10,.0f}{s['sem']:>8,.0f}{a['dd']:>9,.0f}"
              f"{s['mean'] / max(1.0, a['dd']):>8.1f}"
              f"{a['avg_inv']:>9,.0f}{a['fills']:>7,.0f}{a['inf_share'] * 100:>6.1f}"
              f"{a['spread']:>9,.0f}{a['inv_drift']:>10,.0f}{a['hedge_pnl']:>8,.0f}"
              f"{a['fees']:>7,.0f}{a['peak_rate']:>6.0f}{int(a['killed']):>5}")

    base = pnl_by_cfg[ladder[0].name]
    print(f"\n  paired deltas $/day, 95% CI and t. 'vs base' isolates a layer from the\n"
          f"  whole stack; 'vs prev' isolates it from the rung directly beneath, which is\n"
          f"  the contrast that actually answers 'did this layer add anything'.\n")
    print(f"  {'rung':<22}{'vs base':>10}{'t':>7}{'':>3}{'vs prev':>10}{'t':>7}{'win%':>7}{'':>3}")
    print("  " + "-" * 70)
    for i, c in enumerate(ladder):
        if i == 0:
            print(f"  {c.name:<22}{'--':>10}{'':>7}{'':>3}{'--':>10}{'':>7}{'':>7}")
            continue
        pb, pp = paired(base, pnl_by_cfg[c.name]), paired(pnl_by_cfg[ladder[i - 1].name],
                                                          pnl_by_cfg[c.name])
        print(f"  {c.name:<22}{pb['delta']:>10,.0f}{pb['t']:>7.1f}"
              f"{'*' if pb['sig'] else ' ':>3}{pp['delta']:>10,.0f}{pp['t']:>7.1f}"
              f"{pp['win'] * 100:>7.0f}{'*' if pp['sig'] else ' ':>3}")
    print("  * = |t| exceeds the 95% critical value for that paired difference.\n")

    # flag any layer that is significant but NEGATIVE against the rung below:
    # a significant loss is a result too, and needs saying out loud
    for i, c in enumerate(ladder[1:], start=1):
        pp = paired(pnl_by_cfg[ladder[i - 1].name], pnl_by_cfg[c.name])
        if not (pp["sig"] and pp["delta"] < 0):
            continue
        a, b = agg[c.name], agg[ladder[i - 1].name]
        dd_prev = list(dd_rows[ladder[i - 1].name])
        dd_cur = list(dd_rows[c.name])
        ddt = paired(dd_prev, dd_cur)["t"] if dd_prev and dd_cur else 0.0
        print(f"  NOTE: '{c.name}' is significantly WORSE than the rung below it on "
              f"PnL\n        ({pp['delta']:,.0f} $/day, t = {pp['t']:.1f}), but that "
              f"is a risk/PnL trade rather than a defect.")
        print(f"        The PnL/DD column is the point: max drawdown moves "
              f"{b['dd']:,.0f} -> {a['dd']:,.0f}")
        if c.hedge:
            # Only hedge the hedging rung. This paragraph used to be emitted
            # unconditionally, so a non-hedging rung that lost to the rung below
            # got told it "books 0 of futures PnL against 0 of cost" and that
            # "the hedge pays to carry MORE gross inventory" -- describing a leg
            # it does not have. Reachable via `ladder=`, which tests and the
            # tox sweep both pass.
            print(f"        (t = {ddt:.1f} on the paired difference) while avg|inv| "
                  f"moves {b['avg_inv']:,.0f} -> {a['avg_inv']:,.0f} and the hedge "
                  f"books")
            print(f"        {a['hedge_pnl']:,.0f} of futures PnL against "
                  f"{a['hedge_cost']:,.0f} of cost. The skew already controls net "
                  f"delta, so the")
            print(f"        hedge mostly pays to carry MORE gross inventory. It is a "
                  f"drawdown control, and")
            print(f"        a poor one unless the drawdown matters more than the "
                  f"PnL.\n")
        else:
            # No futures leg, so no hedge story to tell. Report the risk
            # quantities this rung actually moved, all present in `_NUMERIC`.
            print(f"        (t = {ddt:.1f} on the paired difference) while avg|inv| "
                  f"moves {b['avg_inv']:,.0f} -> {a['avg_inv']:,.0f} and peak "
                  f"inventory moves")
            print(f"        {b['max_inv']:,.0f} -> {a['max_inv']:,.0f}. This rung "
                  f"trades no derivative, so the loss buys nothing in the way of "
                  f"carry: it is paid in")
            print(f"        wider quotes, a throttler that peaked at "
                  f"{a['peak_rate']:,.0f} msgs/s, and "
                  f"{int(a['killed'])} kill switch{'es' if a['killed'] != 1 else ''}"
                  f" over the session.")
            print(f"        Worth keeping only if that quote width and throttle "
                  f"headroom are worth more than the PnL.\n")

    if len(ladder) > 2:
        sig = agg[ladder[2].name]
        print(f"  signal value: corr(predicted {mp.informed_horizon}s return, realised) = "
              f"{sig['alpha_corr']:+.3f}"
              f"{';  oracle corr n/a -- a recording has no latent state to read.' if replay else ';  oracle corr given the true state at prediction = ' + format(sig['oracle_corr'], '+.3f') + '.'}")
        if not replay:
            print("  The gap is what the imbalance and futures-lead features throw away in "
                  "measurement\n  noise. The oracle is the ceiling: no estimator on these "
                  "features can beat it,\n  and it is read at the same instant the prediction "
                  "was made, so it is a fair one.\n")
        else:
            print("  This is the only honest test of the signal layer available on a\n"
                  "  recording: it is measured, not inferred from a ceiling we cannot\n"
                  "  construct. A near-zero correlation means no estimator built on\n"
                  "  these features is worth much, not merely that this one failed.\n")

    best = max(ladder, key=lambda c: summarize(pnl_by_cfg[c.name])["mean"])
    a = agg[best.name]
    gross = (a["spread"] + a["inv_drift"] + a["hedge_pnl"]
             - a["hedge_cost"] - a["flatten"])
    nsh = a["fills"] * LOT
    # The breakeven is a COST, so it carries the sign of the gross: positive
    # when the strategy earns more than it pays in fees. It used to be negated
    # and scaled by 1e4, which printed "-229 bps/share" for a strategy that
    # clears a 2.3 bp fee, and the sentence underneath it then claimed the
    # number was "far above" a positive fee. Both halves had to move together.
    breakeven = bps_per_share(gross / max(1.0, nsh), mp.s0)
    print(f"  best rung: {best.name}")
    print(f"  gross before fees = {gross:,.0f} $/day on {nsh:,.0f} shares "
          f"-> breakeven fee = {breakeven:+.2f} bps/share "
          f"(${breakeven / 1e4 * mp.s0:+.4f}/sh)")
    if breakeven <= 0:
        verdict = ("this rung loses money before fees, so the breakeven is a "
                   "rebate it would have to be paid to break even")
    elif breakeven > REAL_EXCHANGE_FEE_BPS:
        verdict = (f"{breakeven / REAL_EXCHANGE_FEE_BPS:.0f}x the real "
                   f"~{REAL_EXCHANGE_FEE_BPS} bps exchange fee; a breakeven above "
                   f"the real\n  fee means the sim is generous, not that the "
                   f"strategy is good")
    else:
        verdict = (f"below the real ~{REAL_EXCHANGE_FEE_BPS} bps exchange fee, "
                   f"so the edge does\n  not clear a realistic transaction cost")
    cap = verdict[0].upper() + verdict[1:]
    if a["pnl"]:
        print(f"  of which alpha (inv_drift) = {a['inv_drift'] / a['pnl']:+.0%} of net PnL; "
              f"spread capture = {a['spread'] / a['pnl']:+.0%}. {cap}.")
    else:
        print(f"  {cap}.")
    if audit_path:
        # the audited rung is ladder[-1] (see backtest), which is NOT
        # necessarily the best-PnL rung printed above
        print(f"  audit trail ({ladder[-1].name}, seed 0) -> {audit_path}")


def _plot(path: str, curves: dict, days: int) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9, 5))
    for name, paths in curves.items():
        mean = np.asarray(paths[0] if len(paths) == 1 else np.mean(np.array(paths), axis=0))
        ax.plot(np.arange(len(mean)) / 60, mean, label=name)
    ax.set_xlabel("hours into session")
    ax.set_ylabel("mean cumulative PnL ($)")
    ax.legend()
    ax.set_title(f"Ablation ladder (synthetic market, {days} paired sessions)")
    ax.grid(alpha=.3)
    fig.savefig(path, dpi=130, bbox_inches="tight")
    print(f"  plot -> {path}")


# ============================================================ CLI
def _workers(n: int) -> int:
    if n > 0:
        return n
    return max(1, (os.cpu_count() or 1) - 1)


def _at_least(flag: str, minimum: int):
    """argparse `type` that refuses a count too small to mean anything.

    The library entry points raise on these inputs too, but a raw ValueError
    from four frames down is a traceback, not an interface. Validating here
    turns `--days 0` and `--exec-n 1` into a one-line error and exit 2.
    """
    def parse(v):
        try:
            n = int(v)
        except ValueError:
            raise argparse.ArgumentTypeError(f"{flag} must be an integer, got {v!r}")
        if n < minimum:
            raise argparse.ArgumentTypeError(f"{flag} must be >= {minimum}, got {n}")
        return n
    return parse


def _at_floats(flag: str, minimum: float, strict: bool = False):
    """argparse `type` for a float bound, for flags where a bound is the point.

    `--seconds -5` used to be accepted and simply record nothing while
    reporting success; `--rotate-mb -1` was equally meaningless. Neither
    library entry point caught them either, because the live source fails
    first on a missing websocket dependency, so the user sees an unrelated
    error instead of the argument that is actually wrong.
    """
    op = ">" if strict else ">="

    def parse(v):
        try:
            n = float(v)
        except ValueError:
            raise argparse.ArgumentTypeError(f"{flag} must be a number, got {v!r}")
        bad = n <= minimum if strict else n < minimum
        if bad:
            raise argparse.ArgumentTypeError(f"{flag} must be {op} {minimum}, got {n}")
        return n
    return parse


def _positive(flag: str):
    return _at_floats(flag, 0.0, strict=True)


def _nonneg(flag: str):
    return _at_floats(flag, 0.0)


def _depth():
    """argparse `type` for the partial-book depth Binance actually serves.

    The source checks the same set, but only after `websockets` has been
    imported -- so `--depth 7` reported a missing dependency rather than the
    bad flag, and would have reached the stream URL as an invalid depth
    subscription, which Binance answers by closing the socket.
    """
    def parse(v):
        try:
            n = int(v)
        except ValueError:
            raise argparse.ArgumentTypeError(
                f"--depth must be an integer, got {v!r}")
        if n not in (5, 10, 20):
            raise argparse.ArgumentTypeError(
                f"--depth must be 5, 10 or 20, got {n}")
        return n
    return parse


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="quantforge",
                                 description=__doc__.split("\n")[1])
    ap.add_argument("mode", choices=["backtest", "exec", "sweep", "all", "record"],
                    nargs="?", default="all")
    ap.add_argument("--symbol", default="btcusdt",
                    help="record: symbol to capture (default btcusdt)")
    ap.add_argument("--futures-symbol", default=None,
                    help="record: correlated instrument for the futures-lead "
                         "feature (defaults to --symbol on fstream)")
    ap.add_argument("--capture", default="data/capture.jsonl",
                    help="record: output path for the merged JSONL capture")
    ap.add_argument("--seconds", type=_positive("--seconds"), default=None,
                    help="record: stop after this many seconds (default: run "
                         "until interrupted). Must be > 0: a zero-length session "
                         "records nothing and reports success")
    ap.add_argument("--depth", type=_depth(), default=10,
                    help="record: partial book depth, 5/10/20 (default 10)")
    ap.add_argument("--rotate-mb", type=_nonneg("--rotate-mb"), default=0.0,
                    help="record: roll to a new segment past this size, 0 = off")
    ap.add_argument("--days", type=_at_least("--days", 1), default=None,
                    help="paired sessions per config (backtest default 30, "
                         "sweep default 10). Takes precedence over --seeds. "
                         "Must be >= 1: a run of zero sessions reports no "
                         "variance and is refused, not defaulted.")
    ap.add_argument("--fee", type=float, default=0.0,
                    help="per-share fee, positive = cost (default 0)")
    ap.add_argument("--audit", default=None,
                    help="write a JSONL trail for the last ladder rung on seed 0")
    ap.add_argument("--plot", default=None, help="write the mean-PnL path as a PNG")
    ap.add_argument("--out", default="sweep.csv", help="sweep output path")
    ap.add_argument("--seeds", type=_at_least("--seeds", 1), default=10,
                    help="sweep: sessions per grid point. Alias for --days, "
                         "which wins if both are given. Having two flags for "
                         "one quantity is a trap: --days used to be silently "
                         "ignored by the sweep.")
    ap.add_argument("--alpha-std", type=float, default=None,
                    help="override MarketParams.alpha_std ($/sec of latent drift)")
    ap.add_argument("--informed-rate", type=float, default=None,
                    help="override MarketParams.informed_rate")
    ap.add_argument("--exec-n", type=_at_least("--exec-n", 2), default=None,
                    help="exec: common paths (default 3000; the IS-vs-TWAP "
                         "contrast needs ~30k to resolve, so the demo is "
                         "deliberately underpowered and says so). Must be >= 2: "
                         "one path carries no information about a mean.")
    ap.add_argument("--workers", type=_at_least("--workers", 0), default=0,
                    help="parallel worker processes (0 = auto, 1 = serial). "
                         "Negative is refused rather than folded into auto, "
                         "which is what a truthiness check would do with it.")
    a = ap.parse_args(argv)
    if a.mode == "record":
        # returns rather than falling through: recording blocks on a socket, so
        # putting it in the `all` chain would put the three research modes
        # behind a process that never exits on its own
        from replay import ValidationError, record
        try:
            s = record(symbol=a.symbol, out=a.capture,
                       futures_symbol=a.futures_symbol, seconds=a.seconds,
                       depth=a.depth, rotate_mb=a.rotate_mb)
        except ValidationError as e:
            # the common case is a missing websocket client or a bad depth, and
            # a four-frame traceback is not how this CLI reports a mistake
            ap.exit(2, f"quantforge record: {e}\n")
        print(f"captured {s['rows']} rows ({s['book']} book, "
              f"{s['trades']} prints) across {s['segments']} segment(s), "
              f"{s['anomalies']} anomalies")
        # the row count is the only proof a socket produced anything, and a
        # capture that stopped on a rotate boundary or never connected is
        # otherwise indistinguishable from a quiet market
        if s["rows"] == 0:
            print("WARNING: no rows were captured. The stream connected but "
                  "produced nothing, or never connected at all -- do not treat "
                  "this file as a quiet market.", file=sys.stderr)
        from replay import capture_is_continuous
        if not capture_is_continuous(a.capture):
            print(f"WARNING: {a.capture} holds more than one session, so it "
                  f"was resumed or written by two recorders. The frames across "
                  f"the seam span a gap; see sessions() in replay.record.",
                  file=sys.stderr)
        return
    if a.mode in ("backtest", "all"):
        from dataclasses import replace as _replace
        mp = MarketParams()
        if a.alpha_std is not None:
            mp = _replace(mp, alpha_std=a.alpha_std)
        if a.informed_rate is not None:
            mp = _replace(mp, informed_rate=a.informed_rate)
        backtest(a.days if a.days is not None else 30, a.audit, a.plot,
                 cfg_fee=a.fee, mp=mp, workers=_workers(a.workers))
    if a.mode in ("exec", "all"):
        from exec_algos import exec_demo
        if a.exec_n is not None:
            exec_demo(n=a.exec_n)
        else:
            exec_demo()
    if a.mode in ("sweep", "all"):
        from analysis import sweep
        # `is not None`, not `or`: 0 is falsy, so `--days 0` used to fall
        # through to the default and quietly run 10 grid points under a header
        # asking for none.
        sweep(days=a.days if a.days is not None else a.seeds, out=a.out,
              workers=_workers(a.workers))


if __name__ == "__main__":
    main()
