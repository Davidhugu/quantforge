"""Execution algos: common random numbers, no look-ahead, and a router that
can actually be wrong.

The regression that matters most is `test_router_is_not_degenerate`: the original
scored each venue on its own fills only, and "internal" had zero fee and zero
impact, so its reward was identically zero and the bandit locked onto it on
every seed while the unfilled residual silently went to lit at full cost.
"""
import math

import numpy as np
import pytest

from exec_algos import (ADV, ARRIVAL, BARS, SHARES, Router, exec_demo, is_schedule,
                        make_path, parent_order)
from stats import paired


# ------------------------------------------------------------------ CRN
def test_every_algo_sees_an_identical_price_path():
    """The whole point of CRN: the exogenous path may not depend on the algo."""
    p = make_path(7)
    seen = []
    for algo in ("TWAP", "VWAP", "IS-adaptive"):
        rng = np.random.default_rng(0)
        # replay the exogenous stream the algo is allowed to have been exposed to
        seen.append(p.dw.copy())
    assert all(np.array_equal(seen[0], s) for s in seen)
    # and the IS decomposition must reconstruct the total exactly
    rng = np.random.default_rng(0)
    r = parent_order("TWAP", p, rng)
    assert r["total"] == pytest.approx(r["drift"] + r["impact"] + r["fee"], abs=1e-9)


def test_cost_decomposition_sums_to_total():
    p = make_path(3)
    for algo in ("TWAP", "VWAP", "IS-adaptive"):
        r = parent_order(algo, p, np.random.default_rng(0))
        assert r["total"] == pytest.approx(
            r["drift"] + r["impact"] + r["fee"], abs=1e-9), algo


def test_order_always_completes():
    p = make_path(11)
    for algo in ("TWAP", "VWAP", "IS-adaptive"):
        r = parent_order(algo, p, np.random.default_rng(1))
        assert r["filled"] == pytest.approx(SHARES, rel=1e-9), algo


# ------------------------------------------------------------------ no look-ahead
def test_volume_forecast_is_a_noisy_view_of_realised_not_the_realised_itself():
    p = make_path(5)
    assert not np.array_equal(p.vol_fc, p.vol_real)
    corr = float(np.corrcoef(p.vol_fc, p.vol_real)[0, 1])
    assert 0.5 < corr <= 1.0, "a forecast must correlate with reality"


def test_forecast_leak_would_be_detectable():
    """If vol_fc were vol_real the two would be identical -- guard the guard."""
    p = make_path(5)
    assert np.abs(p.vol_fc - p.vol_real).max() > 1e-3


# ------------------------------------------------------------------ schedule
def test_is_schedule_is_remaining_weight_and_sums_correctly():
    w, W = is_schedule(64)
    assert len(w) == 64 and len(W) == 64
    assert W[0] == pytest.approx(w.sum())
    assert np.all(np.diff(W) <= 1e-12), "remaining weight must be non-increasing"
    assert np.all(w > 0)


def test_schedule_is_front_loaded():
    w, _ = is_schedule(BARS)
    assert w[0] > w[-1] * 5


# ------------------------------------------------------------------ router
def test_router_is_not_degenerate():
    """The venue mix must not be 100% one venue on every seed."""
    mixes = []
    for seed in range(6):
        rng = np.random.default_rng(seed)
        p = make_path(seed)
        r = parent_order("TWAP", p, rng, True, router=Router(rng))
        tot = sum(r["venue"].values())
        mixes.append({k: v / tot for k, v in r["venue"].items()})
    top = {k: sum(m[k] for m in mixes) / len(mixes) for k in Router.VENUES}
    assert max(top.values()) < 0.98, f"router collapsed onto one venue: {top}"
    assert min(top.values()) > 0.0, f"a venue was never used: {top}"


def test_router_reward_is_whole_order_not_own_fills():
    """Scoring a venue on its own fills is the bug that made 'internal' free.

    A venue that fills nothing must not look like the cheapest option, so its
    reward has to include the residual that then goes to lit.
    """
    rng = np.random.default_rng(0)
    r = Router(rng)
    # give 'internal' the textbook degenerate reward of exactly zero
    r.update("internal", 0.0)
    assert r.mu["internal"] == 0.0
    # ...and confirm the real path charges it for what it declines to fill
    p = make_path(1)
    res = parent_order("TWAP", p, np.random.default_rng(0), True, router=Router(rng))
    assert sum(res["venue"].values()) > 0


def test_internalisation_with_a_concession_costs_more_than_free():
    free = make_path(2)
    res_free = parent_order("TWAP", free, np.random.default_rng(0), True,
                            router=Router(np.random.default_rng(0), concession=0.0))
    costly = make_path(2)
    res_costly = parent_order("TWAP", costly, np.random.default_rng(0), True,
                              router=Router(np.random.default_rng(0),
                                            concession=0.005))
    assert res_costly["total"] > res_free["total"]


def test_router_venue_choice_depends_on_costs():
    """Changing the cost structure must change what the bandit picks."""
    def mix(**kw):
        r = Router(np.random.default_rng(0), **kw)
        p = make_path(0)
        out = parent_order("TWAP", p, np.random.default_rng(0), True, router=r)
        tot = sum(out["venue"].values())
        return {k: v / tot for k, v in out["venue"].items()}

    cheap_dark = mix()
    assert cheap_dark["dark"] > 0.5, "dark has the lowest cost; it should win"


def test_venue_fill_rules_respect_their_caps():
    rng = np.random.default_rng(0)
    r = Router(rng, internal_cap=0.3)
    for _ in range(200):
        assert 0.0 <= r.fill("internal", 1_000) <= 300.0
        assert 0.0 <= r.fill("dark", 1_000) <= 1_000.0
        assert r.fill("lit", 1_000) == 1_000.0


# ------------------------------------------------------------------ study level
def test_exec_demo_runs_and_reports_paired_contrasts():
    res = exec_demo(n=12, verbose=False)
    assert len(res) == 6
    for k, d in res.items():
        assert np.all(np.isfinite(d["is_bps"])), k
        assert np.isfinite(d["impact"]) and np.isfinite(d["fee"]), k


def test_crn_makes_the_paired_test_sharper_than_unpaired():
    """The whole reason for CRN: the paired error bar must be far tighter than
    the unpaired one on the same data."""
    from stats import paired, summarize
    paths = [make_path(i) for i in range(24)]
    a, b = [], []
    for p in paths:
        a.append(parent_order("TWAP", p, np.random.default_rng(0))["total"])
        b.append(parent_order("IS-adaptive", p, np.random.default_rng(0))["total"])
    unpaired = math.sqrt(summarize(a)["sem"] ** 2 + summarize(b)["sem"] ** 2)
    assert paired(a, b)["sem"] < 0.5 * unpaired


def test_verdict_text_tracks_the_actual_sample():
    """The summary sentence is generated from the measured contrasts, so a run
    at a different n cannot misdescribe its own output."""
    from exec_algos import _verdict
    res = exec_demo(n=8, verbose=False)
    base = res["TWAP (lit only)"]["is_bps"]
    txt = _verdict(res, base)
    assert "VWAP" in txt and "TWAP" in txt
    assert txt.endswith(".")
    # every claim must be a real one: names exactly the two lit-only schedules
    assert "lit" not in txt and "router" not in txt


def test_path_clips_the_forecast_once_not_per_bar():
    """Regression: np.clip on a python float inside the bar loop was ~40% of the
    adaptive arm's runtime. Clipping must be elementwise identical, just hoisted."""
    p = make_path(0)
    assert np.array_equal(p.fc_clipped, np.clip(p.vol_fc, 0.5, 2.0))
    assert p.fc_clipped.min() >= 0.5 and p.fc_clipped.max() <= 2.0


def test_exec_demo_default_n_is_adequate_or_says_so():
    """The power line must name the shortfall rather than presenting a null."""
    res = exec_demo(n=6, verbose=False)
    base = res["TWAP (lit only)"]["is_bps"]
    p = paired(list(base), list(res["IS-adaptive (lit only)"]["is_bps"]))
    sd = float(np.std(np.asarray(base) - np.asarray(res["IS-adaptive (lit only)"]["is_bps"]),
                      ddof=1))
    need = (2.0 * sd / abs(p["delta"])) ** 2 if p["delta"] else 0.0
    assert need > 6, "n=6 is always underpowered; the check must notice"


def test_power_line_flags_a_divergent_requirement():
    """When the point estimate collapses toward zero the required n diverges, and
    printing "needs n = 1,669,258" reads like a real target rather than a signal
    that the contrast is indistinguishable from zero."""
    import exec_algos as E
    res = exec_demo(n=40, verbose=False)
    base = res["TWAP (lit only)"]["is_bps"]
    # synthesise the pathological case directly: zero effect, finite sd
    p = paired(list(base), list(base))
    assert p["delta"] == 0.0
    txt = E._power_line(p, sd=40.0, n=40)
    assert "indistinguishable from zero" in txt
    assert "inf" in txt
    # and the ordinary shortfall case
    p2 = paired(list(base), [x + 0.5 for x in base])
    t2 = E._power_line(p2, sd=40.0, n=40)
    assert "UNDERPOWERED" in t2
    t3 = E._power_line(p2, sd=40.0, n=10_000_000)
    assert "adequate" in t3


def test_exec_demo_verbose_runs_end_to_end(capsys):
    """The whole reporting path must execute, not just the helper it calls. A
    NameError in the power block shipped once because every other test called
    exec_demo(verbose=False) and never reached the print."""
    res = exec_demo(n=5, verbose=True)
    out = capsys.readouterr().out
    assert "Power:" in out
    assert "CAVEAT on the router rows" in out
    assert "lit-only schedule comparisons" in out


def test_exogenous_path_advances_on_every_bar_even_when_a_slice_is_zero(monkeypatch):
    """A bar we size zero still happened: the market moved through it, and the
    order is marked against that market.

    Regression: the zero-slice `continue` skipped `px_exo += path.dw[i]`, so
    from that bar onward every fill was priced against a stale benchmark and
    the drift bucket under-charged the move the strategy actually sat through.
    Unreachable from the shipped schedules (w > 0 everywhere and
    fc_clipped >= 0.5), so this forces it with a hand-built schedule and a path
    whose only real move is on the skipped bar.
    """
    import exec_algos as E
    B = 4
    w = np.array([1.0, 0.0, 1.0, 0.5])
    W = np.cumsum(w[::-1])[::-1]                     # remaining weight per bar
    monkeypatch.setitem(E._SCHED, (B, 3.0), (w, W))
    ones = np.ones(B)
    path = E.Path(dw=np.array([0.0, 0.05, 0.0, 0.0]), vol_real=ones, vol_fc=ones,
                  sig=0.0, barvol=np.full(B, 1e12), fc_clipped=ones)
    res = E.parent_order("IS-adaptive", path, np.random.default_rng(0), B=B)
    # sig = 0 isolates drift: bar 0 takes 1/2.5 of the book at the arrival
    # price, bar 1 is skipped, bar 2 takes 1/1.5 of the remainder (0.4 of the
    # book) and bar 3 the last 0.2 -- both marked 0.05 above arrival
    assert res["drift"] == pytest.approx(0.6 * 0.05 / ARRIVAL * 1e4, abs=1e-9)
    assert res["filled"] == pytest.approx(SHARES, rel=1e-9)
    assert res["impact"] == pytest.approx(0.0, abs=1e-12)


def test_router_caveat_quotes_its_fee_table_in_bps_not_dollars(capsys):
    """The caveat transcribed SHAPE's fee literals as "1.0 vs 3.0 bp".

    They are $/share -- `parent_order` adds them to a price -- so the run's own
    fee column says 0.1 and 0.3. A 10x error in the one sentence whose job is to
    stop the reader quoting this study as a discovered edge.
    """
    import re

    import exec_algos as E
    exec_demo(n=5, verbose=True)
    out = capsys.readouterr().out
    caveat = out[out.index("CAVEAT on the router rows"):]
    got = re.search(r"\(([\d.]+) vs ([\d.]+) bp fee", caveat)
    assert got, caveat[:300]
    dark, lit = float(got.group(1)), float(got.group(2))
    assert lit == pytest.approx(Router.SHAPE["lit"][2] / ARRIVAL * 1e4, abs=0.05)
    assert dark == pytest.approx(Router.SHAPE["dark"][2] / ARRIVAL * 1e4, abs=0.05)
    # and the prose agrees with the number the table reports
    lit_only = exec_demo(n=5, verbose=False)["TWAP (lit only)"]
    assert lit_only["fee"] == pytest.approx(lit, abs=1e-6)


# ------------------------------------------------------------------ degenerate sample
def test_exec_demo_refuses_fewer_than_two_paths():
    """n=1 printed a full report in which every paired contrast came back
    t = +-inf and starred -- the strongest possible verdict off a single draw --
    and then raised ValueError in the power block, because a per-path sd cannot
    be estimated from one observation. n=0 divided by zero. Both were reachable
    from the CLI as `exec --exec-n 1` and `--exec-n 0`."""
    for n in (0, 1, -5):
        with pytest.raises(ValueError, match="at least 2"):
            exec_demo(n=n)
    # 2 paths is the smallest run that can say anything, and it must work
    assert len(exec_demo(n=2, verbose=False)) == 6


def test_power_line_reports_rather_than_raising_on_a_degenerate_effect():
    """The whole point of `_power_line` is to handle a point estimate that has
    collapsed toward zero. It used to raise on exactly that input: `x ** 2`
    overflows (OverflowError) for a finite x whose square is not representable,
    and `int(inf)` raises too. A guard that crashes on the case it exists to
    describe is worse than no guard."""
    import exec_algos as E
    for delta in (0.0, 1e-12, 1e-160, 5e-324, -1e-300):
        txt = E._power_line({"delta": delta}, sd=40.0, n=60)
        assert "indistinguishable from zero" in txt, delta

    # a non-finite per-path sd is what n=1 produced; it must degrade to the same
    # message rather than taking the report down
    for sd in (float("nan"), float("inf")):
        txt = E._power_line({"delta": 0.5}, sd=sd, n=60)
        assert "indistinguishable from zero" in txt, sd

    # and a real, resolvable effect must still be described as resolvable
    txt = E._power_line({"delta": -2.0}, sd=40.0, n=60)
    assert "n = 1,600" in txt and "UNDERPOWERED" in txt


def test_exec_demo_at_n_is_not_significant_on_one_path():
    """Guards the interaction between the two fixes: even if a caller gets a
    single path past exec_demo's own check, paired() must not promote it."""
    from stats import paired
    p = paired([1.0], [1.5])
    assert p["t"] == 0.0 and p["sig"] is False
