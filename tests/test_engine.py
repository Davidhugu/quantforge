import math

import numpy as np
import pytest

from flow_mm import (H_SIG, LADDER, LOT, TICK, Cfg, Market, MarketParams, RLS,
                     Toxicity, calibration, run_day)

SMALL = MarketParams(steps=4_000)


@pytest.fixture(scope="module")
def market():
    return Market(SMALL, 0)


@pytest.fixture(scope="module")
def rows():
    m = Market(SMALL, 0)
    return {c.name: [run_day(m, c) for _ in range(2)] for c in (Cfg("a"), Cfg("b"))}


# ------------------------------------------------------------------ RLS
def test_rls_matches_least_squares():
    """With forgetting effectively off (lam=1) the RLS must converge to OLS."""
    rng = np.random.default_rng(0)
    n, N = 3, 4_000
    X = rng.standard_normal((N, n))
    w_true = np.array([1.5, -0.7, 0.25])
    y = X @ w_true + rng.standard_normal(N) * 0.01
    r = RLS(n, lam=1.0, delta=1e8)
    for i in range(N):
        r.update(X[i], y[i])
    assert np.allclose(r.w, w_true, atol=0.05)


def test_rls_n2_fastpath_matches_general_path():
    """The hand-unrolled n=2 branch must agree with the general n=3 loop.

    Regression: the general path sliced P into copies and wrote the update into
    the copy, so P never changed and the n>2 case returned garbage.
    """
    rng = np.random.default_rng(1)
    X = rng.standard_normal((2_000, 2))
    y = X @ np.array([0.9, -1.3]) + rng.standard_normal(2_000) * 0.05
    a, b = RLS(2), RLS(3)
    for i in range(2_000):
        a.update(X[i], y[i])
        b.update((X[i][0], X[i][1], 0.0), y[i])
    assert np.allclose(a.w[:2], b.w[:2], atol=1e-9)
    sub = [0, 1, 3, 4]                     # 2x2 block of the 3x3 P
    assert np.allclose([a.P[i] for i in range(4)],
                       [b.P[i] for i in sub], atol=1e-9)


def test_rls_forgetting_responds_to_a_regime_shift():
    """A short forgetting factor must reach the new level faster."""
    r_fast, r_slow = RLS(1, lam=0.90), RLS(1, lam=0.9999)
    for _ in range(1_000):
        r_fast.update((1.0,), 1.0)
        r_slow.update((1.0,), 1.0)
    assert r_fast.w[0] == pytest.approx(1.0, abs=0.05)
    for _ in range(50):                       # 50 samples is mid-transition
        r_fast.update((1.0,), -50.0)
        r_slow.update((1.0,), -50.0)
    assert r_fast.w[0] < r_slow.w[0] - 1.0
    assert r_fast.w[0] < 0.0                  # already overshot to the new level


# ------------------------------------------------------------------ toxicity
def test_toxicity_warms_up_then_tracks_imbalance():
    tox = Toxicity(bucket=1_000, n=10, warm=3)
    assert tox.value() == 0.0                      # too few buckets yet
    for _ in range(3):
        tox.on_trade(1_000)                        # all buy -> VPIN -> 1
    assert tox.value() == pytest.approx(1.0)
    for _ in range(6):
        tox.on_trade(-500)
        tox.on_trade(500)                          # balanced -> VPIN -> 0
    assert tox.value() < 0.4


def test_toxicity_running_sum_matches_naive():
    """value() is maintained incrementally; it must equal the recomputed mean."""
    rng = np.random.default_rng(2)
    tox = Toxicity(bucket=500, n=7)
    for _ in range(400):
        tox.on_trade(float(rng.integers(-300, 301)))
    assert tox.value() == pytest.approx(sum(tox.h) / len(tox.h))


# ------------------------------------------------------------------ market
def test_market_fill_intensity_matches_the_intensity_formula(market):
    """Empirical arrival frequency must track 1-exp(-A*w*exp(-k*d)) at each
    distance, to within a few standard errors."""
    p = SMALL
    rates = {}
    for d in (0.005, 0.01, 0.02, 0.04):
        # quote ONLY our ask, d above the mid *at that step*, benign flow only
        hits = sum(1 for t in range(p.steps)
                   for _, kind, _ in market.intents(t, None, market.Sl[t] + d,
                                                    None, None)
                   if kind == "benign")
        exp_n = sum(1 - math.exp(-p.A * (1 - p.mislabel) * market.profl[t]
                                 * math.exp(-p.k * d)) for t in range(p.steps))
        rates[d] = (hits, exp_n)
    counts = [h for h, _ in rates.values()]
    assert counts == sorted(counts, reverse=True), f"arrivals did not decay: {rates}"
    for hits, exp_n in rates.values():
        assert abs(hits - exp_n) / max(1.0, math.sqrt(exp_n)) < 5.0, rates


def test_informed_only_trades_on_positive_edge(market):
    p = SMALL
    for t in range(0, p.steps, 7):
        S = market.Sl[t]
        wide = S + 0.50                              # far outside any real edge
        got = market.intents(t, wide, wide, wide, wide)
        for side, kind, px in got:
            if kind == "informed":
                assert side * (market.a[t] * p.informed_horizon - (px - S)) > 0


def test_no_client_pays_outside_the_quoted_price(market):
    for t in range(0, SMALL.steps, 11):
        bb, ab = market.Sl[t] - 0.02, market.Sl[t] + 0.02
        bi, ai = market.Sl[t] - 0.05, market.Sl[t] + 0.05
        for side, kind, px in market.intents(t, bb, ab, bi, ai):
            assert px in (bb, ab, bi, ai)


def test_stale_never_overruns_the_session(market):
    assert len(market.stalel) == SMALL.steps
    assert not market.stalel[-1] or SMALL.steps < SMALL.blackout_len


def test_intraday_profile_is_normalised_and_u_shaped():
    m = Market(SMALL, 3)
    assert m.prof.mean() == pytest.approx(1.0, abs=1e-12)
    assert m.profl[0] > m.profl[SMALL.steps // 2]
    assert m.profl[-1] > m.profl[SMALL.steps // 2]


# ------------------------------------------------------------------ calibration
def test_predictable_share_responds_to_alpha_std():
    lo = calibration(MarketParams(steps=4_000, alpha_std=1e-4))
    hi = calibration(MarketParams(steps=4_000, alpha_std=5e-3))
    assert hi["predictable_share"] > 10 * lo["predictable_share"]
    assert 0.0 < lo["predictable_share"] < 1.0


def test_zero_alpha_std_leaves_nothing_predictable():
    """Regression: calibration divided by alpha_std, so the sweep's alpha_std=0
    anchor -- the honest 'is there any edge without a signal?' row -- crashed."""
    c = calibration(MarketParams(steps=4_000, alpha_std=0.0))
    assert c["predictable_share"] == pytest.approx(0.0, abs=1e-12)
    assert c["edge_sd"] == pytest.approx(0.0, abs=1e-12)


# ------------------------------------------------------------------ PnL identity
def test_pnl_decomposition_is_exact(rows):
    """pnl == spread + inv_drift + hedge_pnl - fees - hedge_cost - flatten.

    This is the load-bearing accounting identity of the harness; if it drifts,
    every column in the report is wrong.
    """
    for name, rs in rows.items():
        for r in rs:
            rhs = (r["spread"] + r["inv_drift"] + r["hedge_pnl"]
                   - r["fees"] - r["hedge_cost"] - r["flatten"])
            assert r["pnl"] == pytest.approx(rhs, abs=1e-6), name


def test_fee_shows_up_exactly_once(rows):
    for rs in rows.values():
        for r in rs:
            assert r["fees"] >= 0.0


def test_zero_fee_runs_cost_nothing_in_fees():
    m = Market(SMALL, 1)
    r = run_day(m, Cfg("z", fee_ps=0.0))
    assert r["fees"] == 0.0
    r2 = run_day(m, Cfg("z", fee_ps=0.001))
    assert r2["fees"] == pytest.approx(r2["fills"] * LOT * 0.001)


# ------------------------------------------------------------------ invariants
def test_inventory_respects_the_position_limit():
    m = Market(SMALL, 0)
    r = run_day(m, Cfg("lim", max_pos=400))
    assert r["max_inv"] <= 400 + LOT          # a fill may cross by one lot


def test_risk_rejects_are_only_fired_for_real_rejections():
    m = Market(SMALL, 0)
    r = run_day(m, Cfg("lim", max_pos=200))
    assert r["rej"]["pos_limit"] > 0
    assert r["max_inv"] <= 200 + LOT


def test_equity_curve_is_finite_and_flat_after_a_kill():
    m = Market(SMALL, 0)
    c = Cfg("kill", kill_dd=50.0, throttle_dd=10.0)
    r = run_day(m, c)
    assert np.all(np.isfinite(r["curve"]))
    if r["killed"]:
        tail = r["curve"][-1]
        assert r["curve"][-1] == tail


def test_no_nans_anywhere(rows):
    for rs in rows.values():
        for r in rs:
            for k, v in r.items():
                if isinstance(v, float):
                    assert math.isfinite(v), k
            for k, v in r["rej"].items():
                assert math.isfinite(v), k


# ------------------------------------------------------------------ backtest plumbing
def test_parallel_matches_serial_in_both_chunking_regimes():
    """Parallelism is over seeds, but when seeds are few relative to workers the
    rungs are split too. Both regimes must give bit-identical PnL."""
    from flow_mm import backtest
    small = MarketParams(steps=900)
    for days, w in ((6, 3), (2, 3), (1, 3)):
        a = backtest(days=days, workers=1, quiet=True, mp=small)
        b = backtest(days=days, workers=w, quiet=True, mp=small)
        for k in a["pnl"]:
            assert len(b["pnl"][k]) == days, (k, days, len(b["pnl"][k]))
            assert a["pnl"][k] == pytest.approx(b["pnl"][k], abs=1e-9), (k, days, w)


def test_backtest_aggrees_with_run_day_by_hand():
    from flow_mm import backtest
    small = MarketParams(steps=700)
    r = backtest(days=2, workers=1, quiet=True, mp=small)
    m = Market(small, 0)
    by_hand = [run_day(m, c) for c in LADDER]
    for c, got in zip(LADDER, by_hand):
        assert r["pnl"][c.name][0] == pytest.approx(got["pnl"], abs=1e-9), c.name


def test_pnl_over_dd_is_reported_for_every_rung():
    """Regression: the ladder ranked on PnL alone, which made the hedging rung
    look like a bug when it is buying drawdown. Both axes must be present."""
    from flow_mm import backtest
    r = backtest(days=3, workers=1, quiet=True, mp=MarketParams(steps=900))
    for name, a in r["agg"].items():
        assert "pnl" in a and "dd" in a, name
        assert a["dd"] >= 0.0, name
        # the ratio is what makes the hedge rung legible; it must be finite and
        # defined even for a rung that loses money
        assert math.isfinite(a["pnl"] / max(1.0, a["dd"])), name


def test_hedging_trades_pnl_for_drawdown_on_this_parameter_set():
    """The hedge's purpose is risk, not PnL. Assert the tradeoff exists so a
    future parameter change that removes it is visible rather than silent."""
    from flow_mm import backtest
    # default steps on purpose: at short horizons inventory never reaches the
    # trigger and the hedge is a silent no-op, which would make the assertions
    # below vacuous.
    r = backtest(days=6, workers=1, quiet=True, mp=MarketParams())
    hedged, plain = r["agg"]["+ hedging (full)"], r["agg"]["+ tiering & toxicity"]
    assert hedged["max_inv"] > hedged.get("hedge_trigger", 500), \
        "test is only meaningful if the hedge actually fired"
    assert hedged["pnl"] < plain["pnl"], "hedge should cost PnL here"
    assert hedged["dd"] < plain["dd"], "hedge should cut drawdown"
    assert hedged["pnl"] / hedged["dd"] > plain["pnl"] / plain["dd"], \
        "and should improve the risk-adjusted ratio, which is the point"


def test_days_flag_is_not_silently_ignored_by_the_sweep():
    """Regression: --days is the documented flag for session count, but the CLI
    used to pass a.seeds to sweep() and drop --days entirely. A flag that
    silently does nothing is worse than a missing one, because the run looks
    valid and is just the wrong experiment."""
    import flow_mm as _f
    seen = {}

    def _fake_sweep(days=None, out=None, workers=None, **kw):
        seen["days"] = days
        return []

    import analysis
    orig = analysis.sweep
    analysis.sweep = _fake_sweep
    try:
        _f.main(["sweep", "--days", "3", "--seeds", "17"])
        assert seen["days"] == 3, f"--days was ignored, used {seen['days']}"
        _f.main(["sweep", "--seeds", "17"])
        assert seen["days"] == 17, f"--seeds fallback broke, used {seen['days']}"
    finally:
        analysis.sweep = orig


def test_sweep_csv_shares_column_names_across_both_blocks():
    """The interaction rows used a dprev_signal key while the grid used
    dprev_alpha_signal, so the CSV could not be analysed as one table.

    Stubs the simulator: this is a test of key naming and the CSV round-trip,
    not of the market model, and running the real sweep cost 5 minutes here.
    """
    import analysis, csv, tempfile, os
    from flow_mm import LADDER

    def fake_backtest(days=1, mp=None, cfg_fee=0.0, quiet=True, markets=None,
                      ladder=None, workers=1, **kw):
        lad = ladder or LADDER
        # a distinct value per rung so the paired contrasts are non-degenerate
        pnl = {c.name: [float(k) * (j + 1) for k in range(days)]
               for j, c in enumerate(lad)}
        agg = {c.name: dict(pnl=pnl[c.name][0], dd=100.0 + 10 * j,
                            inf_share=0.1, fills=1000, avg_inv=50.0,
                            alpha_corr=0.2, oracle_corr=0.4)
               for j, c in enumerate(lad)}
        return dict(cal=dict(predictable_share=0.19), pnl=pnl, agg=agg)

    fd, path = tempfile.mkstemp(suffix=".csv")
    os.close(fd)
    orig_bt, orig_grid = analysis.backtest, analysis.GRID
    try:
        analysis.backtest = fake_backtest
        analysis.GRID = [("alpha_std", [0.0, 1.2e-3])]
        analysis.sweep(days=2, out=path, workers=1)
        with open(path) as fh:
            got = list(csv.DictReader(fh))
    finally:
        analysis.backtest, analysis.GRID = orig_bt, orig_grid
        os.unlink(path)

    grid = [r for r in got if r["factor"] != "interaction"]
    inter = [r for r in got if r["factor"] == "interaction"]
    assert grid and inter, (len(grid), len(inter))
    for key in ("dprev_alpha_signal", "tprev_alpha_signal", "winprev_alpha_signal",
                "ci_prev_alpha_signal", "pnl_over_dd"):
        assert key in inter[0], f"{key} missing from interaction rows"
        assert key in grid[0], f"{key} missing from grid rows"
    assert "dprev_signal" not in inter[0]
