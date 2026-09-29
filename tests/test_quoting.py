"""Quoting geometry and risk controls.

The Avellaneda-Stoikov skew sign is the single easiest thing to get backwards
and it fails silently: the model still runs, still prints a PnL, it is just
systematically long when it should be short. That gets a dedicated test.
"""
import math

import pytest

from flow_mm import LOT, TICK, Cfg, Engine, Market, MarketParams, Toxicity

MP = MarketParams(steps=2_000)


def eng(**kw):
    c = Cfg("t", **kw)
    return Engine(c, MP), c


def _fresh(tox=None):
    """A quote-independent engine, so the message governor cannot republish."""
    e, _ = eng()
    if tox is not None:
        e.tox = tox
    return e


# ------------------------------------------------------------------ A-S geometry
def test_reservation_price_moves_down_when_long():
    """Long inventory must lower the reservation price: long = want to sell.

    The original sign error here is silent -- the model still runs and still
    prints a PnL, it is just systematically on the wrong side.
    """
    flat, _ = eng()
    longb, _ = eng()
    a = flat.quotes(0, 100.0, 0.0, 100.0, 0, 0)
    b = longb.quotes(0, 100.0, 0.0, 100.0, 0, 8 * LOT)
    assert b[0] < a[0], f"bid must fall when long: {a[0]} -> {b[0]}"
    assert b[1] < a[1], f"ask must fall when long: {a[1]} -> {b[1]}"


def test_reservation_price_moves_up_when_short():
    flat, _ = eng()
    shortb, _ = eng()
    a = flat.quotes(0, 100.0, 0.0, 100.0, 0, 0)
    b = shortb.quotes(0, 100.0, 0.0, 100.0, 0, -8 * LOT)
    assert b[0] > a[0], f"bid must rise when short: {a[0]} -> {b[0]}"
    assert b[1] > a[1], f"ask must rise when short: {a[1]} -> {b[1]}"


def test_skew_is_monotone_in_inventory():
    """More long inventory => lower reservation price, strictly."""
    mids = []
    for n in (-12, -6, 0, 6, 12):
        e, _ = eng()                      # fresh: the governor republishes otherwise
        q = e.quotes(0, 100.0, 0.0, 100.0, 0, n * LOT)
        mids.append((q[0] + q[1]) / 2)
    assert mids == sorted(mids, reverse=True), f"not monotone: {mids}"
    assert mids[0] > mids[-1]


def test_no_skew_when_disabled():
    e, _ = eng(skew=False)
    a = e.quotes(0, 100.0, 0.0, 100.0, 0, 0)
    b = e.quotes(0, 100.0, 0.0, 100.0, 0, 8 * LOT)
    assert (b[0] + b[1]) == pytest.approx(a[0] + a[1], abs=TICK)


def test_quotes_straddle_the_mid_and_sit_on_the_tick_grid():
    for S in (10.0, 55.13, 99.99, 100.0, 250.0):
        e, _ = eng()                      # fresh: the governor republishes otherwise
        q = e.quotes(0, S, 0.0, S, 0, 0)
        assert q[0] < S < q[1], (q, S)
        assert all(px is not None for px in q), f"book silently dropped at S={S}: {q}"
        for px in q:
            assert abs(px / TICK - round(px / TICK)) < 1e-6, "must be a whole tick"


def test_informed_tier_is_wider_than_benign():
    e, _ = eng(tiering=True)
    q = e.quotes(0, 100.0, 0.0, 100.0, 0, 0)
    assert (q[3] - q[2]) > (q[1] - q[0])


def test_tiering_off_makes_both_tiers_equal():
    e, _ = eng(tiering=False)
    q = e.quotes(0, 100.0, 0.0, 100.0, 0, 0)
    assert q[2] == q[0] and q[3] == q[1]


def test_toxicity_widens_only_above_the_floor():
    """tox_coef * (VPIN - tox_floor): quiet tape must not move the spread at all."""
    e, c = eng(toxic=True, tox_floor=0.30)
    quiet, loud = Toxicity(bucket=10, n=10, warm=1), Toxicity(bucket=10, n=10, warm=1)
    for _ in range(20):
        quiet.on_trade(5)
        quiet.on_trade(-5)               # balanced -> VPIN 0
        loud.on_trade(10)                # all buy -> VPIN 1
    assert quiet.value() == 0.0 and loud.value() == pytest.approx(1.0)
    off = _fresh().quotes(1, 100.0, 0.0, 100.0, 0, 0)      # no toxicity at all
    quiet_q = _fresh(quiet).quotes(1, 100.0, 0.0, 100.0, 0, 0)
    assert quiet_q == off, "below the floor the quote must be identical to no-tox"


def test_toxicity_above_the_floor_widens_the_ask():
    e, c = eng(toxic=True, tox_floor=0.30)
    quiet, loud = Toxicity(bucket=10, n=10, warm=1), Toxicity(bucket=10, n=10, warm=1)
    for _ in range(20):
        quiet.on_trade(5); quiet.on_trade(-5)
        loud.on_trade(10)
    e.tox = quiet
    flat = e.quotes(1, 100.0, 0.0, 100.0, 0, 0)
    e.tox = loud
    toxed = e.quotes(2, 100.0, 0.0, 100.0, 0, 0)
    want = c.tox_coef * (1.0 - c.tox_floor) * (1.0 if not c.tiering else 0.3)
    assert toxed[3] - flat[3] == pytest.approx(want, abs=2 * TICK)


def test_alpha_clip_binds_on_an_outlier():
    """The clip is measured against the estimate's own running sd, so an absurd
    estimate must be pulled back to N sd rather than passed through."""
    e, c = eng(signal=True, alpha_clip=3.0)
    for _ in range(200):
        e.learn(0.0, 0.0, 100.0)             # flat history -> tiny variance
    lim = c.alpha_clip * max(1e-9, math.sqrt(e.alpha_var))
    got = e._alpha(0.0, 100.0 + 1e6, 100.0)
    assert abs(got) <= lim + 1e-12, (got, lim)


def test_message_governor_holds_prior_quotes_when_starved():
    """When the bucket is empty we must republish the *previous* quotes, not
    skip publishing (which would show a stale book) and not exceed the rate."""
    e, c = eng(max_msgs_per_sec=1.0)
    e.risk.bucket.rate, e.risk.bucket.burst = 1.0, 2.0
    e.risk.bucket.tok = 0.0
    prev = e.quotes(0, 100.0, 0.0, 100.0, 0, 0)
    for t in range(1, 20):
        assert e.quotes(t, 100.0 + t * 0.01, 0.0, 100.0 + t * 0.01, 0, 0) == prev, \
            "a denied send must republish the held quote, never a stale one"
    assert e.msgs == 0, "nothing beyond the initial publish should have been sent"
    assert e.risk.bucket.peak == 0, "peak must count published sends, not attempts"


# ------------------------------------------------------------------ risk
def test_position_limit_blocks_only_orders_that_grow_exposure():
    """At the cap we must still be able to *reduce*. For a long book that is
    our ask; for a short book it is our bid."""
    q = (99.99, 100.01, 99.98, 100.02)

    long_e, _ = eng(max_pos=500)
    long_e.risk.approve_pair(0, q, 100.0, +500)
    assert long_e.risk.rej["pos_limit"] == 2, "bids that add to a long book are blocked"
    assert long_e.risk.approve(0, +1, 100.01, 100.0, +500) is True, "ask must be allowed"

    short_e, _ = eng(max_pos=500)
    short_e.risk.approve_pair(0, q, 100.0, -500)
    assert short_e.risk.rej["pos_limit"] == 2, "asks that add to a short book are blocked"
    assert short_e.risk.approve(0, -1, 99.99, 100.0, -500) is True, "bid must be allowed"


def test_notional_limit_fires_independently_of_position():
    e, _ = eng(max_pos=10_000, max_notional=5_000)   # 1 lot * $100 = $10k > $5k
    e.risk.approve_pair(0, (99.99, 100.01, 99.98, 100.02), 100.0, 0)
    assert e.risk.rej["notional"] == 4
    assert e.risk.rej["fat_finger"] == 0, "must not be a fat-finger artefact"


def test_fat_finger_threshold_is_exact():
    """50 bps of $100 is $0.50, and the boundary itself must pass."""
    e, _ = eng(fat_finger_bps=50, max_pos=10 ** 9, max_notional=1e12)
    assert e.risk.approve(0, -1, 100.50, 100.0, 0) is True
    assert e.risk.approve(0, -1, 100.51, 100.0, 0) is False
    assert e.risk.rej["fat_finger"] == 1


def test_kill_switch_fires_exactly_above_kill_dd():
    e, c = eng(throttle_dd=100.0, kill_dd=500.0)
    assert e.risk.on_equity(0, 1_000.0) is None
    e.risk.peak = 1_000.0
    assert e.risk.on_equity(1, 899.0) is None       # dd 101 -> throttle only
    assert e.risk.throttled is True
    e.risk.peak = 1_000.0
    assert e.risk.on_equity(2, 499.0) == "kill"     # dd 501 > 500
    assert e.risk.halted is True


def test_halted_engine_returns_no_quotes():
    e, _ = eng()
    e.risk.halted = True
    assert e.quotes(0, 100.0, 0.0, 100.0, 0, 0) == (None, None, None, None)


def test_throttle_widens_the_spread():
    calm, _ = eng()
    a = calm.quotes(0, 100.0, 0.0, 100.0, 0, 0)
    hot, _ = eng()
    hot.risk.throttled = True
    b = hot.quotes(0, 100.0, 0.0, 100.0, 0, 0)
    assert (b[1] - b[0]) > (a[1] - a[0])
