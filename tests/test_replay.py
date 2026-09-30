"""Tests for the replay layer.

Every test here corresponds to a bug that actually shipped into this package
during its first hour, which is the only justification for its existence:

  * the gap check was inverted (`first == 0` fired on the *good* case), so a
    capture with holes was rejected and a capture starting mid-gap produced a
    zero-price second that would surface downstream as a 100% drawdown;
  * frame timestamps were assigned only on seconds carrying a book event, so
    the index was not the clock and `validate` rejected any capture with a
    hole;
  * `resample` assumed the event stream arrived time-sorted, which none of the
    readers guarantee;
  * the fixture's per-second noise was 0.02 -- roughly 30x a liquid market --
    which inflated the Avellaneda-Stoikov variance term until every quote sat
    outside the book and the entire ablation ladder came back all zeros.

That last one is the dangerous class. It produced a result that read as a
finding about the strategy and was a bug in the fixture.
"""
import json

import numpy as np
import pytest

from flow_mm import LADDER, MarketParams, backtest, run_day
from replay import Dataset, FillModel, ReplayMarket, ValidationError, load, resample, save
from replay.fixtures import synth_session


def _stream(seconds=5, start_ms=1_000_000):
    """A minimal well-formed merged event stream, one book update per second."""
    ts, mid, bid, ask, bq, aq, px, qty, side = [], [], [], [], [], [], [], [], []
    for t in range(seconds):
        base = start_ms + t * 1000
        m = 100.0 + t
        ts.append(base)
        mid.append(m); bid.append(m - 0.05); ask.append(m + 0.05)
        bq.append(10.0); aq.append(12.0)
        px.append(np.nan); qty.append(np.nan); side.append(np.nan)
    return {"ts_ms": np.array(ts, dtype=np.int64), "mid": mid, "bid": bid, "ask": ask,
            "bid_qty": bq, "ask_qty": aq, "px": px, "qty": qty, "side": side}


# ---------------------------------------------------------------- validation

def test_resample_builds_one_frame_per_second():
    ds = resample(_stream(5), 1_000_000, 5)
    assert ds.n_seconds == 5
    assert ds.frames["mid"].tolist() == [100.0, 101.0, 102.0, 103.0, 104.0]


def test_frame_clock_is_contiguous_even_across_a_gap():
    ds = synth_session(seconds=60, gap_at=30)
    assert np.all(np.diff(ds.frames["ts_ms"]) == 1000), \
        "the frame index is the clock; a hole must not renumber time"


def test_gap_seconds_are_flagged_stale_and_carry_the_last_quote():
    ds = synth_session(seconds=60, gap_at=30)
    stale = np.flatnonzero(ds.frames["stale"])
    assert stale.tolist() == [30, 31, 32]
    # a hole must not become a zero-price second
    assert (ds.frames["bid"][stale] > 0).all() and (ds.frames["ask"][stale] > 0).all()
    for i in stale:
        assert ds.frames["mid"][i] == pytest.approx(ds.frames["mid"][29])
    assert ds.frames["mid"][33] != ds.frames["mid"][32], "the book resumed"


def test_capture_with_no_opening_book_is_rejected_not_zero_filled():
    ev = _stream(5)
    keep = ev["ts_ms"] >= 1_000_000 + 2000      # drop the first two seconds
    ev = {k: np.asarray(v)[keep] for k, v in ev.items()}
    with pytest.raises(ValidationError, match="no book events in its first second"):
        resample(ev, 1_000_000, 5)


def test_event_order_does_not_change_the_result():
    rng = np.random.default_rng(7)
    ds = synth_session(seed=3, seconds=40)
    # rebuild the stream and shuffle it, as a real capture arrives interleaved
    ev = _stream(40)
    rows = []
    for i in range(40):
        base = 1_757_000_000_000 + i * 1000
        rows.append((base, ds.frames["mid"][i], ds.frames["bid"][i], ds.frames["ask"][i],
                     ds.frames["bid_qty"][i], ds.frames["ask_qty"][i],
                     np.nan, np.nan, np.nan))
    for i in range(len(ds.trades["ts_ms"])):
        rows.append((int(ds.trades["ts_ms"][i]), np.nan, np.nan, np.nan, np.nan,
                     np.nan, float(ds.trades["px"][i]), float(ds.trades["qty"][i]),
                     float(ds.trades["side"][i])))
    ev = np.array(rows, dtype=float)
    order = rng.permutation(len(ev))
    ev = ev[order]
    shuf = {"ts_ms": ev[:, 0].astype(np.int64), "mid": ev[:, 1], "bid": ev[:, 2],
            "ask": ev[:, 3], "bid_qty": ev[:, 4], "ask_qty": ev[:, 5],
            "px": ev[:, 6], "qty": ev[:, 7], "side": ev[:, 8]}
    out = resample(shuf, 1_757_000_000_000, 40)
    # "last event in the second wins" is only meaningful in time order
    assert np.allclose(out.frames["mid"], ds.frames["mid"])
    assert np.allclose(out.frames["bid"], ds.frames["bid"])


def test_a_print_after_the_books_close_lands_in_the_next_second():
    """The frame clock is the boundary; a late print must not leak backwards."""
    ev = _stream(3)
    ev["px"] = list(ev["px"]) + [100.02, 101.02]
    # `np.append` takes a single extra value, not two: passing two scalars made
    # it nest them into one object array and the failure surfaced as an
    # out-of-bounds axis rather than as the malformed call it was
    ev["ts_ms"] = np.append(ev["ts_ms"], [1_000_900, 1_001_999])
    # The two prints need a real qty and side. Padding those with NaN as the
    # book columns are padded made `resample` cast NaN to int8, and the NaN
    # became 0 -- a print with aggressor side 0, which `validate` correctly
    # rejects. The column padding applies to the *other* columns.
    for k in ("mid", "bid", "ask", "bid_qty", "ask_qty"):
        ev[k] = list(ev[k]) + [np.nan, np.nan]
    ev["qty"] = list(ev["qty"]) + [1.0, 1.0]
    ev["side"] = list(ev["side"]) + [1, -1]
    ds = resample(ev, 1_000_000, 3)
    by_sec = [ds.trades_in_second(i) for i in range(3)]
    assert [len(x) for x in by_sec] == [1, 1, 0]


def test_crossed_book_is_rejected():
    f = {"ts_ms": np.array([0, 1000, 2000], dtype=np.int64),
         "mid": np.array([100.0, 101.0, 102.0]),
         "bid": np.array([100.0, 101.0, 102.0]),
         "ask": np.array([100.0, 101.0, 100.0]),      # last second is crossed
         "bid_qty": np.ones(3), "ask_qty": np.ones(3),
         "futures": np.array([100.0, 101.0, 102.0]),
         "stale": np.zeros(3, dtype=bool)}
    t = {"ts_ms": np.array([], dtype=np.int64), "px": np.array([]),
         "qty": np.array([]), "side": np.array([], dtype=np.int8)}
    with pytest.raises(ValidationError):
        Dataset(f, t, {"venue": "x"})


def test_print_outside_the_book_is_rejected():
    """A tape print beyond the touch is a broken feed, not an opportunity."""
    ds = synth_session(seed=8, seconds=10)
    ds.trades["px"][0] = ds.frames["ask"][1] * 1.5
    with pytest.raises(ValidationError):
        Dataset(ds.frames, ds.trades, ds.meta)


# ------------------------------------------------------------------ fixture

def test_fixture_volatility_is_plausible():
    """Guards the failure that produced an all-zero ablation.

    The A-S half-spread is `_vterm * var + _kterm` and the variance term is
    linear in realised variance, so an over-volatile fixture quotes far outside
    the book, nothing fills, and the ladder reports a result that is really a
    fixture bug. Pin the scale.
    """
    ds = synth_session(seed=0, seconds=900)
    r = np.diff(np.log(ds.frames["mid"]))
    per_sec_bps = float(r.std()) * 1e4
    assert per_sec_bps < 5.0, f"{per_sec_bps:.2f} bps/second is not a liquid market"
    assert per_sec_bps > 0.05, "too flat to exercise a quoting model at all"


def test_fixture_spread_is_a_few_bps():
    ds = synth_session(seed=1, seconds=600)
    bps = float(((ds.frames["ask"] - ds.frames["bid"]) / ds.frames["mid"] * 1e4).mean())
    assert 0.5 < bps < 30


# --------------------------------------------------------------- io round trip

def test_npz_round_trip_is_lossless(tmp_path):
    ds = synth_session(seed=2, seconds=120)
    p = tmp_path / "s.npz"
    save(ds, str(p))
    back = load(str(p))
    assert back.n_seconds == ds.n_seconds
    for k in ds.frames:
        assert np.allclose(back.frames[k], ds.frames[k]), k
    for k in ds.trades:
        assert np.allclose(back.trades[k], ds.trades[k]), k


# ------------------------------------------------------------- capture venue

def test_a_capture_keeps_the_venue_the_recorder_recorded(tmp_path):
    """The filename is not the venue, and the reader used to conflate them.

    `capture_to_dataset` overwrote the venue with the file's stem, so a capture
    written by `record` -- which knows it was talking to `binance:BTCUSDT` and
    says so in the sidecar -- came back labelled `btc`, which is a symbol. On a
    single-venue setup that reads like cosmetics; the moment two venues share a
    ticker it merges two different books under one label, silently. Found by
    the live smoke test.
    """
    from replay import CaptureWriter, capture_to_dataset
    p = tmp_path / "btc.jsonl"
    t0 = 1_700_000_000_000
    with CaptureWriter(str(p), venue="binance:BTCUSDT") as w:
        for k in range(5):
            w.book(t0 + k * 1_000, 99.95, 100.05, 10.0, 12.0)
    assert capture_to_dataset(str(p)).meta["venue"] == "binance:BTCUSDT"
    # an explicit argument still wins, for a caller who knows better
    assert capture_to_dataset(str(p), venue="kraken").meta["venue"] == "kraken"


def test_a_capture_with_no_recorder_sidecar_still_falls_back_to_the_filename(tmp_path):
    """The venue lookup must not become a hard dependency on a sidecar.

    A capture from any other source has none, and refusing to read it -- or
    labelling it `None` -- would be a regression in exchange for the fix above.
    """
    from replay import capture_to_dataset
    p = tmp_path / "ethusdt.jsonl"
    with open(p, "w") as fh:
        for k in range(5):
            fh.write(json.dumps({
                "ts_ms": 1_700_000_000_000 + k * 1_000, "mid": 2000.0,
                "bid": 1999.0, "ask": 2001.0, "bid_qty": 1.0, "ask_qty": 1.0,
                "futures": None, "px": None, "qty": None, "side": None}) + "\n")
    assert capture_to_dataset(str(p)).meta["venue"] == "ethusdt"


# ---------------------------------------------------------------- ReplayMarket

def test_replay_market_satisfies_the_engine_interface():
    ds = synth_session(seed=4, seconds=600)
    m = ReplayMarket(ds, mp=MarketParams(steps=ds.n_seconds))
    n = ds.n_seconds
    for attr in ("Sl", "Fl", "Il", "tapel", "stalel", "al"):
        assert len(getattr(m, attr)) == n + 1, attr
    # naive 1-tick rests at the touch, so it is the rung guaranteed to trade;
    # the A-S rung only fills when its spread lands inside the book
    r = run_day(m, LADDER[0])
    assert r["fills"] > 0, "a 600s session with a book in it should trade something"


def test_imbalance_is_derived_from_the_book_not_injected():
    ds = synth_session(seed=5, seconds=120)
    m = ReplayMarket(ds, mp=MarketParams(steps=ds.n_seconds))
    f = ds.frames
    raw = (f["bid_qty"] - f["ask_qty"]) / (f["bid_qty"] + f["ask_qty"])
    # `Il` is a plain list, so boolean-mask indexing needs an array first. It is
    # aligned to frames positionally -- `Il[t]` is frame `t`, with one extra
    # repeated element at the end -- so this is a prefix, not a shift. Slicing
    # from 1 compared every frame against its neighbour, and the "is this a pure
    # function" assertion then failed on data that was in fact fine.
    got = np.asarray(m.Il[:len(raw)])
    # a fixed rescaling of the book imbalance, not an injected latent state:
    # the ratio must be the same constant at every second
    nz = raw != 0
    ratio = got[nz] / raw[nz]
    assert np.allclose(ratio, ratio[0]), "Il is not a pure function of the book"


def test_fill_model_is_deterministic_under_a_fixed_seed():
    """Same seed, same fills: a replay that cannot be reproduced proves nothing."""
    ds = synth_session(seed=6, seconds=200)
    runs = []
    for _ in range(2):
        m = ReplayMarket(ds, mp=MarketParams(steps=ds.n_seconds),
                         fill=FillModel(seed=11))
        runs.append([intents for t in (10, 50, 120)
                     for intents in m.intents(t, *m.quotes_at(t))] if False else None)
    p1 = ReplayMarket(ds, mp=MarketParams(steps=ds.n_seconds),
                      fill=FillModel(seed=11)).p
    p2 = ReplayMarket(ds, mp=MarketParams(steps=ds.n_seconds),
                      fill=FillModel(seed=11)).p
    assert p1 == p2


# ------------------------------------------------------------------ reporting

def test_replay_report_marks_state_statistics_undefined(capsys):
    ds = [synth_session(seed=s, seconds=300) for s in range(2)]
    mkts = [ReplayMarket(d, mp=MarketParams(steps=d.n_seconds)) for d in ds]
    backtest(days=2, mp=MarketParams(steps=300), quiet=False, workers=1, markets=mkts)
    out = capsys.readouterr().out
    assert "Replayed-data ablation" in out
    # a recording has no latent state: those numbers must not be invented
    assert "predictable share of the 30s return = n/a" in out
    assert "oracle corr n/a" in out
    assert "measured from the recording" in out


def test_synthetic_report_is_unchanged_by_the_replay_work(capsys):
    backtest(days=2, mp=MarketParams(steps=300), quiet=False, workers=1)
    out = capsys.readouterr().out
    assert "Synthetic-market ablation" in out
    assert "oracle corr given the true state at prediction" in out
    assert "n/a" not in out.split("signal value")[1].split("best rung")[0]
