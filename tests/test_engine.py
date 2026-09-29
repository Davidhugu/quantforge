import math
import re

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


def _kill_run(steps=2000, seed=0, **over):
    """Run a day that trips the kill switch, returning (result, kill_index).

    The kill index comes from the audit trail rather than from being inferred
    off the curve, so the assertions below can be stated exactly instead of
    approximately. Two details that cost time to get wrong:

    * the audit is a buffered text file and must be closed before it is read;
    * the shipped market only ever draws down ~$100 over a session, so the
      default kill_dd of 6000 is ~60x the real drawdown and the switch never
      trips. 40 does. `throttle_dd` is raised to 1e9 so throttling cannot stop
      trading before the kill threshold is reached -- this test is about the
      flatten, not the throttle.
    """
    import json, tempfile, os
    from flow_mm import Audit
    fd, path = tempfile.mkstemp(suffix=".jsonl")
    os.close(fd)
    try:
        p = MarketParams(steps=steps)
        cfg = dict(kill_dd=40.0, throttle_dd=1e9, flatten_cost=0.02)
        cfg.update(over)
        a = Audit(path, "t")
        r = run_day(Market(p, seed), Cfg("z", **cfg), audit=a)
        a.close()
        with open(path) as fh:
            kills = [json.loads(l)["sim_s"] for l in fh
                     if json.loads(l)["ev"] == "KILL_SWITCH"]
    finally:
        os.unlink(path)
    assert kills, "the kill switch never fired; the rest of the test is vacuous"
    return r, kills[0]


def test_kill_charges_the_flatten_on_the_step_it_happens():
    """The kill branch flattens the book and stops trading, so from the kill step
    onward the curve is frozen at post-flatten cash: no inventory, no hedge, no
    market risk.

    Regression: the branch backfilled `curve[u] = cash` from t+2 while
    `curve[t+1]` had already been written with pre-flatten equity. So the
    flatten cost showed up one step LATE, leaving curve[t+1] exactly `flatten`
    too high -- the plotted curve steps up into a kill and then falls, which is
    not something the strategy ever did. (The same off-by-one drops the cost
    from the curve altogether if the kill fires on the final step, where
    range(t+2, T+1) is empty; that case is not reachable from the shipped market
    parameters, so it is covered here by the same assertion rather than by a
    test of its own.)
    """
    r, k = _kill_run()
    assert r["flatten"] > 0.0
    assert r["curve"][k + 1] == pytest.approx(r["pnl"], abs=1e-6), (
        f"kill at t={k}: curve[{k}+1]={r['curve'][k + 1]:.4f} but pnl={r['pnl']:.4f} "
        f"-- the curve is {r['curve'][k + 1] - r['pnl']:+.4f} too high, i.e. the "
        f"flatten landed a step late")
    # and the whole post-kill tail is frozen at that same value
    assert all(v == pytest.approx(r["pnl"], abs=1e-6) for v in r["curve"][k + 1:]), \
        "the curve kept moving after the book was flat and trading stopped"
    assert r["curve"][-1] == pytest.approx(r["pnl"], abs=1e-6)


def test_kill_curve_has_no_hump_at_the_kill_step():
    """The same defect from the drawdown side. Flattening is a pure cost, so
    equity must never step UP into the kill and then fall. The original curve
    had curve[t+1] above the frozen tail, i.e. a recovery that never happened.

    Also pins the size of the error: the hump is exactly `flatten`, which is the
    signature of a one-step index shift rather than a modelling difference.
    """
    r, k = _kill_run(steps=3000, seed=0)
    tail = r["curve"][k + 1:]
    assert len(tail) > 2
    assert tail[0] == pytest.approx(min(tail), abs=1e-6), (
        f"curve[{k}+1]={tail[0]:.4f} sits above the frozen tail min={min(tail):.4f} "
        f"by {tail[0] - min(tail):.4f}, which should be exactly flatten="
        f"{r['flatten']:.4f}")


def test_adverse_is_the_negation_of_the_per_fill_drift_contribution():
    """`adverse` is a memo diagnostic, not a PnL term, and its sign is opposite
    to both `inv_drift` and the markouts. It was previously documented as "a
    sub-view of inv_drift", which is false. The identity that does hold is per
    fill:  mo == spread_c - adverse_c,  because all three are side*LOT times a
    difference of prices, and (px - S) - (far - S) == (px - far).

    Note the units: `spread` and `adverse` are dollars, while the reported
    `mo_b` / `mo_i` are basis points per share, so they must be converted before
    they can be added to a dollar figure.
    """
    m = Market(SMALL, 0)
    r = run_day(m, Cfg("a"))
    mo_dollars = (r["mo_b"] * r["n_ben"] * LOT / 100
                  + r["mo_i"] * r["n_inf"] * LOT / 100)
    assert mo_dollars == pytest.approx(r["spread"] - r["adverse"], abs=1e-6)
    # and the sign claim itself: positive adverse == adverse to us == negative
    # markout, so a day that made money on markout must net negative here
    assert (mo_dollars > 0) == (r["adverse"] < r["spread"])


def test_oracle_corr_is_read_at_prediction_time_not_window_end():
    """The report calls this the ceiling no estimator can beat, so it has to be
    the best forecast available WHEN THE PREDICTION WAS MADE.

    Regression: it used a[t] -- the latent state at the far end of the H-step
    window -- which is a look-ahead of the full horizon. It reported 0.119 where
    the honest ceiling is 0.178, i.e. the signal layer looked closer to the
    ceiling than it is.
    """
    p = MarketParams(steps=4000)
    m = Market(p, 0)
    r = run_day(m, Cfg("x"))
    gain = sum(p.alpha_phi ** j for j in range(H_SIG))
    y = [m.Sl[t] - m.Sl[t - H_SIG] for t in range(H_SIG, p.steps)]
    corr = lambda a: float(np.corrcoef(a, y)[0, 1])   # noqa: E731
    honest = corr([m.al[t - H_SIG] * gain for t in range(H_SIG, p.steps)])
    lookahead = corr([m.al[t] * gain for t in range(H_SIG, p.steps)])
    assert r["oracle_corr"] == pytest.approx(honest, abs=1e-6)
    assert abs(lookahead - honest) > 1e-3, "the two must differ or this proves nothing"


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
        # `_interaction` reads its two axes out of GRID, so a narrowed grid has
        # to carry both -- it is a speed knob, not a licence to drop an axis
        analysis.GRID = [("alpha_std", [0.0, 1.2e-3]),
                         ("informed_horizon", [10, 60])]
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


def test_backtest_and_sweep_reject_zero_days():
    """days=0 used to split on `workers`: the serial path raised IndexError
    while the parallel path ran seed 0 anyway and returned a full ladder under
    a "0 paired sessions" header -- with sem = 0 and t = +-inf everywhere. The
    parallel path is the CLI default, so that was the reachable one."""
    from flow_mm import backtest
    small = MarketParams(steps=200)
    for w in (1, 3):
        with pytest.raises(ValueError, match="days must be"):
            backtest(days=0, workers=w, quiet=True, mp=small)
    # and a valid count still runs on both paths
    for w in (1, 3):
        assert len(backtest(days=1, workers=w, quiet=True, mp=small)["pnl"]) == len(LADDER)

    import analysis
    orig_bt, orig_grid = analysis.backtest, analysis.GRID
    try:
        analysis.backtest = lambda **kw: dict(cal=dict(predictable_share=0.19),
                                              pnl={}, agg={})
        analysis.GRID = [("alpha_std", [0.0])]
        with pytest.raises(ValueError, match="days must be"):
            analysis.sweep(days=0, out=None, workers=1)
    finally:
        analysis.backtest, analysis.GRID = orig_bt, orig_grid


# ------------------------------------------------------------------ CLI input
def test_cli_passes_an_explicit_session_count_through(capsys):
    """The wiring from flag to backtest() must be explicit.

    `backtest(a.days or 30, ...)` reads 0 as "unset", so `--days 0` ran the
    full 30-session default. Two things now stop that: argparse rejects an
    impossible count before main() sees it (the test below), and the default is
    applied on `is not None` rather than truthiness. The first is the guard a
    user hits; the second is why no other falsy value can leak through either,
    so this pins the pass-through itself -- an explicit count reaches backtest()
    unchanged, an absent one still defaults, and --days still beats --seeds.
    """
    import flow_mm as _f
    seen = {}
    orig = _f.backtest
    _f.backtest = lambda days=None, *a, **kw: seen.setdefault("days", days)
    try:
        _f.main(["backtest", "--days", "1"])
        assert seen["days"] == 1, f"explicit --days 1 arrived as {seen['days']}"
        seen.clear()
        _f.main(["backtest"])
        assert seen["days"] == 30, f"the no-flag default became {seen['days']}"
        seen.clear()
        # --days must also win over --seeds rather than being dropped
        import analysis
        orig_sweep = analysis.sweep
        analysis.sweep = lambda days=None, **kw: seen.setdefault("days", days)
        try:
            _f.main(["sweep", "--days", "4", "--seeds", "17"])
            assert seen["days"] == 4, f"--days arrived as {seen['days']}"
            seen.clear()
            _f.main(["sweep", "--seeds", "17"])
            assert seen["days"] == 17, f"--seeds fallback became {seen['days']}"
        finally:
            analysis.sweep = orig_sweep
    finally:
        _f.backtest = orig
    capsys.readouterr()


def test_cli_refuses_impossible_counts_instead_of_tracing_back(capsys):
    """A count too small to carry information is user error, and the library
    entry points already say so in a clear sentence. Letting that ValueError
    escape four frames down printed a stack trace instead of a usage line --
    for `--days -1`, `--days 0`, `--exec-n 1` and `--exec-n 0`, all reachable
    from the shell.
    """
    import flow_mm as _f
    cases = [(["backtest", "--days", "0"], "--days", 1),
             (["backtest", "--days", "-1"], "--days", 1),
             (["sweep", "--days", "0"], "--days", 1),
             (["sweep", "--days", "-1"], "--days", 1),
             (["sweep", "--seeds", "0"], "--seeds", 1),
             (["exec", "--exec-n", "0"], "--exec-n", 2),
             (["exec", "--exec-n", "1"], "--exec-n", 2)]
    for argv, flag, bound in cases:
        with pytest.raises(SystemExit) as e:
            _f.main(argv)
        assert e.value.code == 2, argv
        err = capsys.readouterr().err
        # the bound is named explicitly, not left as a bare "invalid value"
        assert f"{flag}: {flag} must be >= {bound}" in err, (argv, err)
    # --workers 0 legitimately means "auto", but a negative count is a typo and
    # must not be quietly resolved to auto the way `n > 0` would resolve it
    with pytest.raises(SystemExit):
        _f.main(["backtest", "--workers", "-1"])
    assert "--workers: --workers must be >= 0" in capsys.readouterr().err


# ------------------------------------------------------------------ fee units
def test_breakeven_fee_is_true_bps_of_notional_with_the_sign_of_the_gross(capsys):
    """The breakeven line printed `-229.05 bps/share` for a rung that clears a
    2.3 bp fee: it negated the number (a fee is a cost, so it takes the sign of
    the gross) and scaled by 1e4 without dividing by the price. exec_algos'
    own `bps()` divides, so the two studies disagreed by 100x.
    """
    from flow_mm import backtest, bps_per_share
    small = MarketParams(steps=2500)
    r = backtest(days=2, workers=1, quiet=False, mp=small)
    out = capsys.readouterr().out
    name = re.search(r"best rung: (.+)", out).group(1).strip()
    got = re.search(r"breakeven fee = ([+-][\d.]+) bps/share", out)
    assert got, out[-900:]
    got = float(got.group(1))
    a = r["agg"][name]
    gross = (a["spread"] + a["inv_drift"] + a["hedge_pnl"]
             - a["hedge_cost"] - a["flatten"])
    want = bps_per_share(gross / max(1.0, a["fills"] * LOT), small.s0)
    assert got == pytest.approx(want, abs=0.01), (got, want)
    # a fee is a cost: positive exactly when the rung earns back its costs
    assert (got > 0) == (gross > 0)
    # and it is emphatically not the old 100x-inflated figure
    assert abs(got - want * 100) > 0.5


def test_fee_header_reports_bps_of_notional_not_of_a_dollar(capsys):
    """`--fee 0.003` is 0.3 bp on a $100 stock -- the real exchange fee. The
    header multiplied by 1e4 alone and printed it as 30.00, a hundred times the
    cost the reader thinks they are charging.
    """
    from flow_mm import backtest
    backtest(days=1, workers=1, quiet=False, mp=MarketParams(steps=400),
             cfg_fee=0.003)
    out = capsys.readouterr().out
    got = re.search(r"fee = ([+-][\d.]+) bps/share", out)
    assert got, out[:400]
    assert float(got.group(1)) == pytest.approx(0.3, abs=0.005)


def test_bps_per_share_agrees_with_the_exec_study_convention():
    """One unit convention across the repo.

    `exec_algos` reports in bps of the arrival price, so the backtest's fee has
    to be scaled the same way or the two reports cannot be read against each
    other. Router.SHAPE stores $/share (they are added to a price in
    `parent_order`), which is the trap: 0.0030 there is 0.3 bp, not 3.0.
    """
    from flow_mm import bps_per_share
    from exec_algos import ARRIVAL, Router, parent_order, make_path
    assert bps_per_share(Router.SHAPE["lit"][2], ARRIVAL) == pytest.approx(0.3)
    assert bps_per_share(Router.SHAPE["dark"][2], ARRIVAL) == pytest.approx(0.1)
    r = parent_order("TWAP", make_path(0), np.random.default_rng(0))
    # lit fills completely, so the whole order pays exactly the lit fee
    assert r["fee"] == pytest.approx(0.3, abs=1e-9)


def test_report_labels_the_audit_with_the_rung_that_was_actually_written(capsys):
    """The audit is written for `ladder[-1]` (see backtest) but the report
    labelled it with `best.name`. Those differ whenever the best-PnL rung is
    not the last -- the shipped case, since hedging normally costs PnL -- and
    then the run points the reader at a rung whose fills are not in the file at
    all. The observed run printed "audit trail (+ tiering & toxicity)" over
    records every one of which read "+ hedging (full)|seed0", including hedge
    events that rung cannot emit because it has hedge=False.

    Driven off synthetic aggregates rather than a real market so the ordering is
    guaranteed: otherwise the test passes or fails with the simulation, and at
    small horizons the hedge never fires and the best rung *is* the last one,
    which is exactly how this slipped through.
    """
    from flow_mm import _report, calibration
    mp = MarketParams(steps=1000)
    # rung 0 clearly best, the rest equal, so the NOTE block does fire (for
    # rung 1, a non-hedging one, well below rung 0) -- irrelevant here, but it is
    # why the aggregates below carry every field the report may read
    pnl = {c.name: ([100.0, 102.0, 98.0, 101.0] if i == 0 else [50.0, 52.0, 48.0, 51.0])
           for i, c in enumerate(LADDER)}
    agg = {name: dict(pnl=sum(v) / len(v), dd=100.0, avg_inv=50.0, max_inv=90.0,
                      fills=1000, inf_share=0.1, spread=100.0, inv_drift=50.0,
                      hedge_pnl=0.0, fees=0.0, hedge_cost=0.0, flatten=0.0,
                      peak_rate=4.0, killed=0, alpha_corr=0.2, oracle_corr=0.4)
           for name, v in pnl.items()}
    _report(calibration(mp), mp, LADDER, pnl, agg, 4, 0.0, "audit.jsonl",
            dd_rows={c.name: [1.0, 2.0, 3.0, 4.0] for c in LADDER})
    out = capsys.readouterr().out
    best = re.search(r"best rung: (.+)", out).group(1).strip()
    label = re.search(r"audit trail \((.+?), seed 0\)", out).group(1)
    # the precondition this test exists for: the two names really do differ
    assert best != LADDER[-1].name, f"ladder ordering is degenerate: {best}"
    assert label == LADDER[-1].name, (label, LADDER[-1].name)


def _find_note(out):
    """Return (rung_name, prose) for the NOTE `_report` emitted, or None.

    The name is split off because it is caller-supplied and may legitimately
    contain the very words the prose must not ("+ A-S skew/spread").
    """
    m = re.search(r"  NOTE: '(.+?)' is significantly WORSE.*?rather than a "
                  r"defect\.\n(.*?)(?=\n  \S|\Z)", out, re.S)
    return None if m is None else (m.group(1), m.group(2))


def _note_prose(capsys, worse_idx):
    """Drive `_report` with synthetic aggregates in which `ladder[worse_idx]` is
    significantly worse than the rung below it, and return its NOTE prose.

    Synthetic rather than simulated on purpose. Which rung actually loses is a
    property of the market draw, so a real run can fail to exercise the branch
    under test and the test still goes green -- the same trap that let the
    audit-label bug through. Here the ordering is guaranteed by construction.
    """
    from flow_mm import _report, calibration
    mp = MarketParams(steps=1000)
    pnl = {c.name: ([100.0, 102.0, 98.0, 101.0] if i != worse_idx
                    else [40.0, 41.0, 39.0, 40.0])
           for i, c in enumerate(LADDER)}
    agg = {}
    for name, v in pnl.items():
        agg[name] = dict(pnl=sum(v) / len(v), dd=100.0, avg_inv=50.0, max_inv=90.0,
                         fills=1000, inf_share=0.1, spread=100.0, inv_drift=50.0,
                         hedge_pnl=0.0, fees=0.0, hedge_cost=0.0, flatten=0.0,
                         peak_rate=4.0, killed=2, alpha_corr=0.2, oracle_corr=0.4)
    _report(calibration(mp), mp, LADDER, pnl, agg, 4, 0.0, None,
            dd_rows={c.name: [1.0, 2.0, 3.0, 4.0] for c in LADDER})
    found = _find_note(capsys.readouterr().out)
    assert found is not None, f"no NOTE emitted for ladder[{worse_idx}]"
    # the precondition: the rung the NOTE names is the one we made worse
    assert found[0] == LADDER[worse_idx].name, found[0]
    return found[1]


def test_worse_non_hedging_rung_is_not_described_as_hedging(capsys):
    """The NOTE block used to end with "the hedge books 0 of futures PnL
    against 0 of cost ... the hedge mostly pays to carry MORE gross inventory"
    for *every* rung that lost to the rung below it. `+ tiering & toxicity` has
    hedge=False and never trades a future, so the paragraph described a leg it
    does not have -- a reader chasing a futures cost that cannot exist. It was
    reachable through the `ladder=` argument, which tests and the tox sweep pass.
    """
    worse = next(i for i, c in enumerate(LADDER) if not c.hedge and i > 0)
    prose = _note_prose(capsys, worse)
    for word in ("hedge", "futures", "skew"):
        assert word not in prose.lower(), f"{word!r} leaked into a non-hedging NOTE:\n{prose}"
    # ...and it still says the useful thing: the risk quantities that moved
    assert "avg|inv|" in prose
    assert "peak inventory" in prose
    assert "kill switch" in prose


def test_worse_hedging_rung_keeps_its_futures_narrative(capsys):
    """The shipped ladder can lose PnL at `+ hedging (full)`, so this is the
    branch a real run may take. The fix must not have cost it its explanation."""
    worse = next(i for i, c in enumerate(LADDER) if c.hedge and i > 0)
    prose = _note_prose(capsys, worse)
    assert "futures PnL against" in prose, prose
    assert "gross inventory" in prose, prose
    assert "kill switch" not in prose, prose


# The NOTE fires for the rung stacked ABOVE a better one, so the loser goes on
# top. These pair a strong rung against a much weaker one to make the loss large
# and consistent by construction.
_WEAK = Cfg("+ A-S skew/spread", signal=False, tiering=False, toxic=False, hedge=False)


def test_real_run_does_not_invent_a_futures_leg_for_a_non_hedging_rung(capsys):
    """Real engine output, not synthetic aggregates. The bug only ever reached
    production through a *custom* ladder, so prove it on a real run: the weak
    rung has hedge=False, sits above the hedging rung, and loses by ~200 $/day.
    Before the fix this printed "the hedge books 0 of futures PnL against 0 of
    cost" for a config that never trades a future.

    `quiet=False` is load-bearing: quiet skips `_report` altogether.
    """
    from flow_mm import backtest
    backtest(days=6, mp=MarketParams(steps=4_000), quiet=False, workers=1,
             ladder=[Cfg("+ hedging (full)"), _WEAK])
    found = _find_note(capsys.readouterr().out)
    assert found is not None, "a 200 $/day loss produced no NOTE at all"
    assert found[0] == _WEAK.name, found[0]
    for word in ("hedge", "futures", "skew"):
        assert word not in found[1].lower(), found[1]


def test_the_hedged_ladder_cannot_reliably_trigger_the_note_on_real_data():
    """Why the futures branch above is tested on synthetic aggregates only.

    The hedge costs PnL against `+ tiering & toxicity` by a mean of ~30 $/day,
    but that is inside the seed noise: the paired t is -1.68 at 10 sessions and
    the mean delta crosses zero by 30 sessions, where it reads +1.4 $/day. A test
    built on the shipped ladder would therefore flip between firing and not
    depending on the draw, and would pass vacuously most of the time. This is
    also the honest reading of the shipped report -- the hedge's cost is not
    established, which is why the note says it is a drawdown-control question
    rather than a defect.
    """
    from flow_mm import backtest
    from stats import paired
    r = backtest(days=10, mp=MarketParams(steps=4_000), quiet=True, workers=1)
    pp = paired(r["pnl"]["+ tiering & toxicity"], r["pnl"]["+ hedging (full)"])
    assert pp["delta"] < 0, "hedging is no longer the worse rung; revisit this note"
    assert not pp["sig"], "the hedge's cost is now significant; say so in the README"


def test_audit_trail_is_rewritten_per_run_not_appended(capsys):
    """`Audit` opens "w". A second run to the same path must not interleave two
    markets in one file -- the docstring and the README both used to call it an
    append-only trail, which described a guarantee nothing provided.
    """
    import json
    import os
    import tempfile
    from flow_mm import backtest
    fd, path = tempfile.mkstemp(suffix=".jsonl")
    os.close(fd)
    try:
        for days in (1, 2):
            backtest(days=days, workers=1, quiet=True,
                     mp=MarketParams(steps=400), audit_path=path)
            with open(path) as fh:
                runs = {json.loads(l)["run"] for l in fh if l.strip()}
            assert runs == {f"{LADDER[-1].name}|seed0"}, (days, runs)
    finally:
        os.unlink(path)


def test_sweep_significance_star_uses_the_df_aware_critical_value():
    """The '*' column must agree with flow_mm._report, which uses
    `paired()["sig"]` == |t| > t_crit95(n-1).

    Regression: the sweep hard-coded |t| > 2.0. At its shipped default
    (days=10 -> df=9 -> 2.262) that starred rungs the backtest table called not
    significant on the same contrast, and it disagreed with `_interaction` in
    the same file, which already used the paired verdict.
    """
    import analysis, io, contextlib
    from stats import paired, t_crit95

    def _series(t_target, n):
        """Paired difference with a realised t of `t_target`.

        t = mean / (sd / sqrt(n)) is invariant to scale but not to dispersion,
        so vary the shape k: k = 0 is a constant shift (t = inf), larger k
        lowers t toward 0.
        """
        x = [((k % (2 * n + 1)) - n) / (n + 1) for k in range(n)]
        t_of = lambda k: paired([0.0] * n, [1.0 + k * v for v in x])["t"]  # noqa: E731
        lo, hi = 0.0, 50.0
        for _ in range(80):
            mid = (lo + hi) / 2
            if t_of(mid) > t_target:
                lo = mid
            else:
                hi = mid
        return [1.0 + hi * v for v in x]

    def _star(days, t_target):
        s = _series(t_target, days)

        def fake_backtest(days=10, mp=None, cfg_fee=0.0, quiet=True, markets=None,
                          ladder=None, workers=1, **kw):
            lad = ladder or LADDER
            idx = {c.name: i for i, c in enumerate(lad)}
            return dict(cal=dict(predictable_share=0.19),
                        pnl={c.name: (s if idx[c.name] == 2 else [0.0] * days)
                             for c in lad},
                        agg={c.name: dict(pnl=0.0, dd=100.0 + 10 * idx[c.name],
                                          inf_share=0.1, fills=1000, avg_inv=50.0,
                                          alpha_corr=0.2, oracle_corr=0.4)
                             for c in lad})

        orig = (analysis.backtest, analysis.GRID, analysis._interaction)
        analysis.backtest, analysis.GRID = fake_backtest, [("alpha_std", [0.0])]
        analysis._interaction = lambda *a, **k: []   # 12 real backtests otherwise
        try:
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                # workers=2 skips the Market cache the serial path builds, which
                # is ~1.5 s per session and pointless when backtest is stubbed
                analysis.sweep(days=days, out=None, workers=2)
            row = [l for l in buf.getvalue().splitlines()
                   if l.strip().startswith("baseline")][0]
            # the star is a 3-wide trailing field, so it must be read off the
            # UNstripped line: stripping an empty star field leaves the win%
            # as the last character
            return paired([0.0] * days, s)["t"], "*" if row.rstrip().endswith("*") else " "
        finally:
            analysis.backtest, analysis.GRID, analysis._interaction = orig

    # 2.262 at days=10: 2.05 and 2.20 sit in the window where the old hard-coded
    # 2.0 said "significant" and the honest answer is "not yet".
    assert t_crit95(9) == pytest.approx(2.262)
    for t_target, expect_star in ((2.05, " "), (2.20, " "), (2.30, "*"), (3.00, "*")):
        got, star = _star(10, t_target)
        assert abs(got - t_target) < 0.05, f"failed to build t={t_target}, got {got}"
        assert star == expect_star, \
            f"t={got:.2f} at days=10 (tcrit {t_crit95(9):.3f}) starred {star!r}"

    # and the star must track df: the same t is significant at days=30 (tcrit
    # 2.045) where it is not at days=3 (tcrit 4.303)
    _, star_lo = _star(3, 3.00)
    _, star_hi = _star(30, 3.00)
    assert star_lo == " " and star_hi == "*", (star_lo, star_hi)
