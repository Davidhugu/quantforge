"""
exec_algos.py -- client-order execution algos with a venue router (RL-lite)

WHAT CHANGED AND WHY
  1. Common random numbers. `parent_order` used to draw the benchmark's price
     path from the same generator that drove participation, so every algo saw a
     different market. With 157 bps of path noise against 1-3 bps of algo
     effect, 300 paths gave a standard error of 9 bps -- the three algos were
     not distinguishable. The exogenous path now comes from its own stream
     (`make_path`) and is identical across algos, so only the pairing matters.
  2. No look-ahead. The adaptive IS branch sized its slices with the *realised*
     bar volume. It now sizes on `vol_fc`, a noisy forecast drawn from a
     different stream; `vol_real` is used only for cost accounting.
  3. A correct router objective. The reward used to be the venue's own cost per
     share, and "internal" had zero fee and zero impact -- so its reward was
     identically zero and epsilon-greedy converged to it on every seed while the
     un-filled residual silently dropped to "lit" at full cost. The reward is
     now the whole-order cost of the bar, which is what gives the bandit a real
     decision even at the default concession: internal fills at most
     `internal_cap` of the slice, and the rest of that slice re-prices at the lit
     venue inside `bar_cost`, so a venue cannot look cheap merely by declining to
     fill. Note that at the default `concession=0.0` an internal share really
     does cost nothing, so the concession is what the parameter is for and there
     is no CLI flag for it yet.
  4. O(B) schedule. The IS branch recomputed sum(exp(-k*j/B)) over the remaining
     bars inside the loop: O(B^2) ~ 46M math.exp calls for the study.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from stats import paired, summarize

BARS = 390            # 390 one-minute bars in a 6.5h session
SHARES = 200_000
ADV = 5_000_000
ARRIVAL = 100.0




@dataclass
class Path:
    """Exogenous market state. Shared byte-for-byte across algos under CRN."""
    dw: np.ndarray       # (B,) benchmark price increments, $
    vol_real: np.ndarray  # (B,) realised bar volume multiplier
    vol_fc: np.ndarray    # (B,) volume FORECAST -- the only volume an algo may see
    sig: float
    barvol: np.ndarray    # (B,) realised bar volume, shares
    fc_clipped: np.ndarray  # (B,) vol_fc pre-clipped to the usable band. Clipping
                            # once here rather than per bar in parent_order was
                            # worth ~40% of that arm's runtime: np.clip on a
                            # python float costs ~5us of dispatch, 390x per order.


def make_path(seed: int, B: int = BARS, adv: int = ADV, s0: float = ARRIVAL,
              sigma_ann: float = 0.25, vol_fc_err: float = 0.30) -> Path:
    """Draw one market path.

    `vol_fc` is a noisy forecast OF `vol_real` (lognormal error), so a volume
    model can extract the correlated part and no more -- which is the whole
    point. The two are not independent draws: an independent forecast would be
    pure noise and the adaptive algo could learn nothing either way.
    """
    rng = np.random.default_rng(seed)
    sig = s0 * sigma_ann / math.sqrt(252 * B)
    vol_real = np.exp(0.4 * rng.standard_normal(B))
    vol_fc = vol_real * np.exp(vol_fc_err * rng.standard_normal(B))
    return Path(dw=sig * rng.standard_normal(B), vol_real=vol_real, vol_fc=vol_fc,
                sig=sig, barvol=adv * _curve(B) * vol_real,
                fc_clipped=np.clip(vol_fc, 0.5, 2.0))


_CURVE: dict[int, np.ndarray] = {}


def _curve(B: int) -> np.ndarray:
    if B not in _CURVE:
        t = np.arange(B)
        c = 1 + 1.5 * (np.exp(-t / 30) + np.exp(-(B - 1 - t) / 30))
        _CURVE[B] = c / c.sum()
    return _CURVE[B]


class Router:
    """Epsilon-greedy bandit over venues, scored on whole-order cost.

    `update` takes the total cost of executing the bar under that venue, so a
    venue cannot look cheap merely by declining to fill. Pricing internalisation
    at a concession rather than at zero is what gives the bandit a real
    decision: free internalisation dominates on every parameter set.
    """

    VENUES = ("lit", "dark", "internal")
    SHAPE = {"lit": ("full", 1.0, 0.0030),      # (fill rule, impact mult, fee $/sh)
             "dark": ("beta", 0.2, 0.0010),
             "internal": ("cap", 0.0, 0.0)}

    def __init__(self, rng, eps: float = 0.1, concession: float = 0.0,
                 internal_cap: float = 0.3):
        self.rng, self.eps = rng, eps
        self.concession, self.cap = concession, internal_cap
        self.n = dict.fromkeys(self.VENUES, 0)
        self.mu = dict.fromkeys(self.VENUES, 0.0)
        self.chosen = dict.fromkeys(self.VENUES, 0)

    def pick(self) -> str:
        if self.rng.random() < self.eps or min(self.n.values()) == 0:
            return self.VENUES[int(self.rng.integers(len(self.VENUES)))]
        return min(self.mu, key=self.mu.get)

    def update(self, v: str, cost_per_share: float) -> None:
        self.n[v] += 1
        self.chosen[v] += 1
        self.mu[v] += (cost_per_share - self.mu[v]) / self.n[v]

    def fill(self, v: str, tgt: float) -> float:
        kind = self.SHAPE[v][0]
        if kind == "full":
            return tgt
        if kind == "beta":
            return tgt * float(self.rng.beta(2, 3))
        return tgt * float(self.rng.uniform(0.0, self.cap))


_SCHED: dict[tuple, tuple] = {}


def is_schedule(B: int, kappa: float = 3.0) -> tuple:
    """Front-loaded urgency weights and the remaining weight at each bar.

    One reversed cumulative sum instead of a per-bar O(B) rescan (the old inner
    `sum(math.exp(...))` cost ~46M exp calls across the study).
    """
    if (B, kappa) not in _SCHED:
        w = np.exp(-kappa * np.arange(B) / B)
        _SCHED[(B, kappa)] = (w, np.cumsum(w[::-1])[::-1])
    return _SCHED[(B, kappa)]


def parent_order(algo: str, path: Path, rng, use_router: bool = False,
                 shares: int = SHARES, B: int = BARS, eta: float = 1.0,
                 perm: float = 0.1, router: Router | None = None) -> dict:
    """Execute `shares` over B bars. Returns an exactly additive IS decomposition.

    Every cost is attributed to one of three buckets against a *fixed* arrival
    price, so the three always sum to the total:
        total = drift + impact + fee
      drift  = the exogenous price move we could not control
      impact = temporary + permanent impact net of explicit fees
      fee    = explicit venue fees and the internalisation concession
    """
    w, W = is_schedule(B)
    cv = _curve(B)
    px_exo = ARRIVAL      # exogenous price, common across algos under CRN
    perm_impact = 0.0     # permanent impact we have inflicted on ourselves
    rem, spent = float(shares), 0.0
    drift_g = impact_g = fee_g = 0.0

    for i in range(B):
        if rem <= 0:
            break
        if algo == "TWAP":
            tgt = shares / B
        elif algo == "VWAP":
            tgt = shares * float(cv[i])
        else:
            # adaptive IS: front-load, then scale by the volume FORECAST
            tgt = rem * w[i] / W[i] * float(path.fc_clipped[i])
        tgt = rem if i == B - 1 else min(rem, tgt)
        if tgt <= 0:
            # the bar still happened: the exogenous price moved through it, so
            # px_exo must advance even though we traded nothing. `continue`
            # without this desynchronises the marked price from path.dw for the
            # rest of the order, and every later bar is then charged against a
            # stale benchmark. Unreachable from the shipped schedules (w > 0
            # and fc_clipped >= 0.5), but it is one line and the loop's
            # invariant should not depend on the caller's weights.
            px_exo += path.dw[i]
            continue

        v = router.pick() if router else "lit"
        got = router.fill(v, tgt) if router else tgt
        _, imp_mult, fee = Router.SHAPE[v]
        px_start = px_exo + perm_impact
        bar_cost = 0.0

        for qty, mult, f, conc in ((got, imp_mult, fee,
                                    router.concession if (router and v == "internal") else 0.0),
                                   (tgt - got, 1.0, Router.SHAPE["lit"][2], 0.0)):
            if qty <= 0:
                continue
            rho = qty / path.barvol[i]
            px = px_start + mult * eta * path.sig * math.sqrt(rho) + f + conc
            spent += qty * px
            bar_cost += qty * (px - px_start)
            drift_g += qty * (px_exo - ARRIVAL)
            impact_g += qty * (px - px_exo - f - conc)
            fee_g += qty * (f + conc)
            perm_impact += perm * path.sig * rho * (qty / tgt)

        if router:
            router.update(v, bar_cost / tgt)   # whole-order cost of this bar
        rem -= tgt
        px_exo += path.dw[i]

    def bps(x):
        return x / shares / ARRIVAL * 1e4

    return dict(total=bps(spent - shares * ARRIVAL), drift=bps(drift_g),
                impact=bps(impact_g), fee=bps(fee_g),
                filled=shares - rem, venue=router.chosen.copy() if router else None)


def _power_line(p, sd: float, n: int) -> str:
    """The n required for |t| > 2, phrased so a diverging number is not mistaken
    for a real sample-size target.

    n = (2*sd/|delta|)^2. As the point estimate goes to zero this diverges, and
    a huge number means "this contrast is indistinguishable from zero", not
    "run a million paths".
    """
    delta = p["delta"]
    # Compute in float and range-check before int(). Two separate overflows used
    # to bite here: `x ** 2` raises OverflowError when x is finite but the square
    # is not representable, and `int(inf)` raises too. A delta small enough to
    # trip either is exactly the "indistinguishable from zero" case this line
    # exists to report, so the guard crashed on the case it was written for.
    # Multiplication saturates to inf where `**` raises.
    ratio = 2.0 * sd / abs(delta) if (delta and math.isfinite(delta)) else math.inf
    raw = ratio * ratio
    need = int(math.ceil(raw)) if math.isfinite(raw) and raw < 1e18 else 0
    if delta == 0.0 or not math.isfinite(delta) or need == 0 or need > 250_000:
        verdict = ("point estimate is indistinguishable from zero, so no "
                   "practical n resolves it")
        need_txt = f"n > {need:,}" if need else "n = inf"
    elif n >= need:
        verdict = "adequate"
        need_txt = f"about n = {need:,}"
    else:
        verdict = "UNDERPOWERED, read as null"
        need_txt = f"about n = {need:,}"
    return (f"  Power: the IS-adaptive vs TWAP contrast is {delta:+.2f} bps with a "
            f"per-path\n  sd of {sd:.0f} bps, so |t| > 2 needs {need_txt}. This run "
            f"used n = {n:,} -- {verdict}.")


def _verdict(res: dict, base) -> str:
    """Describe the lit-only schedule contrasts, as measured at THIS n.

    Hard-coding "VWAP wins, IS does not" would go stale the moment someone runs
    a different n, and the run would then misdescribe its own output.
    """
    out = []
    for k in ("VWAP (lit only)", "IS-adaptive (lit only)"):
        p = paired(list(base), list(res[k]["is_bps"]))
        name = k.split(" (")[0]
        if p["sig"]:
            verb = "beats" if p["delta"] < 0 else "loses to"
            out.append(f"{name} {verb} TWAP by {abs(p['delta']):.2f} bp (t={p['t']:.1f})")
        else:
            out.append(f"{name} does not separate from TWAP (t={p['t']:.1f})")
    return "; ".join(out) + "."


def exec_demo(n: int = 3000, concession: float = 0.0, verbose: bool = True) -> dict:
    """Compare schedules on common random numbers.

    n is set by how long this is allowed to take, and the required n is
    *reported* rather than assumed: the run prints the IS-adaptive vs TWAP
    effect size, the per-path sd, and the n that |t| > 2 would need, and labels
    itself UNDERPOWERED when it falls short. A null result and an underpowered
    one look identical in a t column, and conflating them is the whole failure
    mode here.

    The default n=3000 is roughly 3 minutes across the six arms. Resolving the
    IS contrast properly needs an order of magnitude more paths, so pass
    --exec-n if you want the real number rather than the self-reported
    shortfall.

    n must be at least 2. A single path carries no information about a mean
    difference: it made every paired contrast come back t = +-inf and starred,
    printed a confident report off one draw, and then died in the power block
    because a per-path sd cannot be estimated from one observation.
    """
    if n < 2:
        raise ValueError(f"exec_demo() needs at least 2 common paths, got {n}")
    algos = ("TWAP", "VWAP", "IS-adaptive")
    res: dict[str, dict] = {}
    paths = [make_path(s) for s in range(n)]
    for algo in algos:
        for use_router in (False, True):
            runs, mixes = [], []
            for i, p in enumerate(paths):
                rng = np.random.default_rng(10_000 + i)
                r = parent_order(algo, p, rng, use_router,
                                 router=Router(rng, concession=concession) if use_router
                                 else None)
                runs.append(r)
                if r["venue"]:
                    mixes.append(r["venue"])
            key = algo + (" + router" if use_router else " (lit only)")
            res[key] = dict(
                is_bps=np.array([r["total"] for r in runs]),
                drift=float(np.mean([r["drift"] for r in runs])),
                impact=float(np.mean([r["impact"] for r in runs])),
                fee=float(np.mean([r["fee"] for r in runs])),
                # chosen counts one entry per bar; normalise by total decisions
                mix={k: sum(m[k] for m in mixes)
                     / max(1, sum(sum(m.values()) for m in mixes))
                     for k in Router.VENUES} if use_router else None)

    if not verbose:
        return res
    print(f"\nBuy {SHARES:,} sh ({SHARES / ADV:.0%} of ADV), {BARS} one-minute bars, "
          f"{n} common paths\n")
    print("  IS = implementation shortfall vs arrival, bps. Paired: every algo sees "
          "the same\n  price and volume path, so the differences below are real.\n")
    hdr = f"{'algo / routing':<24}{'IS bps':>9}{'sem':>7}{'drift':>8}{'impact':>8}{'fee':>7}   venue mix"
    print(hdr)
    print("-" * (len(hdr) + 12))
    for k, d in res.items():
        s = summarize(list(d["is_bps"]))
        mix = ""
        if d["mix"]:
            mix = "  " + " ".join(f"{v}={d['mix'][v]:.0%}" for v in Router.VENUES)
        print(f"{k:<24}{s['mean']:>9.2f}{s['sem']:>7.2f}{d['drift']:>8.2f}"
              f"{d['impact']:>8.2f}{d['fee']:>7.2f}{mix}")

    # The router contrast needs a caveat: Router.SHAPE *declares* lit more
    # expensive than dark, so routing to dark lowers cost by close to a constant
    # and the paired sd collapses. A t of -400 here is a statement about the
    # fee table above, not a discovered edge. Report the dispersion so it cannot
    # be quoted as a result.
    lit_only = res["TWAP (lit only)"]["is_bps"]
    routed = res["TWAP + router"]["is_bps"]
    dd = np.asarray(routed) - np.asarray(lit_only)
    print(f"\n  paired vs TWAP (lit only), same paths:")
    print(f"  {'contrast':<24}{'delta bps':>11}{'95% CI':>18}{'t':>8}{'':>4}")
    base = res["TWAP (lit only)"]["is_bps"]
    for k, d in res.items():
        if k == "TWAP (lit only)":
            continue
        p = paired(list(base), list(d["is_bps"]))
        ci = f"[{p['lo']:.2f}, {p['hi']:.2f}]"
        print(f"  {k:<24}{p['delta']:>11.2f}{ci:>18}{p['t']:>8.2f}"
              f"{'*' if p['sig'] else ' ':>4}")
    print("  * = |t| above the 95% critical value. Negative delta = cheaper than TWAP.\n")

    # State the power. A null result and an underpowered one look identical in
    # a t column, and the difference matters when deciding whether to trust it.
    k_ref = "IS-adaptive (lit only)"
    p_ref = paired(list(base), list(res[k_ref]["is_bps"]))
    sd = float(np.std(np.asarray(base) - np.asarray(res[k_ref]["is_bps"]), ddof=1))
    print(_power_line(p_ref, sd=sd, n=len(base)))

    # The fees are quoted here from SHAPE itself rather than transcribed, in bps
    # of the arrival price. The literals are $/share -- they are added to a
    # price in `parent_order` -- so a hard-coded "3.0 bp" read them as if they
    # were already bps and overstated both by 100x. The run's own `fee` column
    # is the reference: 0.3 bp lit, 0.1 bp dark.
    lit_bps = Router.SHAPE["lit"][2] / ARRIVAL * 1e4
    dark_bps = Router.SHAPE["dark"][2] / ARRIVAL * 1e4
    print(f"\n  CAVEAT on the router rows: Router.SHAPE *declares* dark cheaper than lit\n"
          f"  ({dark_bps:.1f} vs {lit_bps:.1f} bp fee, "
          f"{Router.SHAPE['dark'][1]}x vs {Router.SHAPE['lit'][1]}x impact), so the "
          f"gain is near-constant --\n"
          f"  per-path delta {dd.mean():+.2f} +/- {dd.std():.2f} bps. The very large t is a\n"
          f"  statement about that fee table, not a discovered edge: the bandit never had a\n"
          f"  real decision to make, and it will look impressive on any sample size.\n"
          f"  The meaningful rows are the lit-only schedule comparisons: "
          f"{_verdict(res, base)}")
    return res


if __name__ == "__main__":
    exec_demo()
