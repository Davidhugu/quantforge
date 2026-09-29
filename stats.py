"""Paired statistics, no scipy.

The ablation ladder runs every config on the *same* seeds, so the informative
quantity is the paired per-day difference, not the level of any one config.
`paired()` is the workhorse; `summarize()` is for standalone series.
"""
from __future__ import annotations

import math
from statistics import NormalDist

# two-sided 95% Student-t critical values, df = 1..30
_T95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365,
        8: 2.306, 9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145,
        15: 2.131, 16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093, 20: 2.086,
        21: 2.080, 22: 2.074, 23: 2.069, 24: 2.064, 25: 2.060, 26: 2.056,
        27: 2.052, 28: 2.048, 29: 2.045, 30: 2.042}


def t_crit95(df: int) -> float:
    """Two-sided 95% critical value; normal limit beyond the table."""
    return _T95[df] if 1 <= df <= 30 else NormalDist().inv_cdf(0.975)


def summarize(x) -> dict:
    """mean, sem and the 95% band of a single series.

    Everything is coerced to native Python types: callers pass numpy arrays and
    otherwise get numpy scalars back, which are not JSON-serialisable and make
    `result["sig"] is True` fail.

    Raises on an empty series rather than returning zeros. The mean of nothing is
    not zero, and a plausible-looking dict is worse than a loud failure.
    """
    n = len(x)
    if not n:
        raise ValueError("summarize() got an empty series")
    m = sum(x) / n
    if n < 2:
        return dict(n=n, mean=float(m), sem=0.0, lo=float(m), hi=float(m), sd=0.0)
    var = sum((v - m) ** 2 for v in x) / (n - 1)
    sem = math.sqrt(var / n)
    tc = t_crit95(n - 1)
    return dict(n=n, mean=float(m), sd=math.sqrt(var), sem=sem,
                lo=float(m - tc * sem), hi=float(m + tc * sem))


def paired(x, y) -> dict:
    """Statistics on the paired difference y - x (same seeds, same market paths).

    Raises when the two series differ in length. `zip` would truncate silently
    and then compute the critical value from the truncated length, so a rung
    measured over a different number of sessions than the baseline would come
    back as a confident answer to the wrong question.

    A single observation is reported as t = 0 and not significant. One draw
    cannot distinguish a real effect from one that happened to land, and
    `sem = 0` used to turn it into t = +-inf, which is how a one-session run
    ended up with every rung starred.
    """
    if len(x) != len(y):
        raise ValueError(f"paired() needs equal-length series, got {len(x)} and {len(y)}")
    d = [float(b) - float(a) for a, b in zip(x, y)]
    s = summarize(d)
    s["delta"] = s["mean"]
    s["win"] = float(sum(v > 0 for v in d) / len(d))
    # t of the mean difference against zero; +/-inf when the difference is
    # deterministic, which -- with n >= 2 -- is the strongest possible result
    if s["n"] < 2:
        s["t"] = 0.0
    elif s["sem"] > 0:
        s["t"] = float(s["mean"] / s["sem"])
    else:
        s["t"] = math.copysign(math.inf, s["mean"]) if s["mean"] else 0.0
    s["t_p95"] = t_crit95(max(1, len(d) - 1))
    s["sig"] = bool(s["n"] >= 2 and abs(s["t"]) > s["t_p95"])
    return s
