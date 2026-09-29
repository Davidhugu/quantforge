"""Tests for the sweep grid's internal consistency.

These are cheap by design: they assert the *wiring* between `GRID` and the
blocks that consume it, not any simulated outcome. Running a real sweep takes
minutes, so nothing here calls `backtest` for real.
"""
import dataclasses

import pytest

import analysis
from flow_mm import LADDER, Cfg, MarketParams


def test_every_swept_field_is_a_real_market_or_cfg_field():
    """A typo in `GRID` is silent: `replace(MarketParams(), alpha_stnd=...)` would
    raise at sweep time, but only after the run has started, and a field that
    happens to be valid on the wrong dataclass would pass a `hasattr` check on
    the wrong one. So check each name against the dataclass the sweep actually
    substitutes into.
    """
    mp_fields = {f.name for f in dataclasses.fields(MarketParams)}
    cfg_fields = {f.name for f in dataclasses.fields(Cfg)}
    for name, _ in analysis.GRID:
        if name is None:
            continue  # the fee sweep, which travels through cfg_fee instead
        assert name in mp_fields or name in cfg_fields, (
            f"GRID sweeps {name!r}, which is neither a MarketParams nor a Cfg "
            f"field; the sweep would fail on the first cell")


def test_interaction_levels_come_from_grid_not_a_private_copy():
    """`_interaction` used to spell out its own alpha_std and informed_horizon
    levels. Editing `GRID` then changed the one-at-a-time sweep and left the
    cross term -- and the README table built from it -- describing a grid that no
    longer existed. Pin the two blocks to the same source."""
    assert analysis._grid_values("alpha_std") == \
        next(v for n, v in analysis.GRID if n == "alpha_std")
    assert analysis._grid_values("informed_horizon") == \
        next(v for n, v in analysis.GRID if n == "informed_horizon")


def test_interaction_loop_actually_walks_the_grid(monkeypatch):
    """The real proof, and the only kind that works here.

    The hardcoded literals `_interaction` used to carry are *currently identical*
    to `GRID`, so a test asserting "12 cells with these 12 values" passes either
    way -- it would not catch the defect at all. What has to be pinned is the
    dependency, so rewrite `GRID` to values the loop has never seen and require
    the cells to follow. Restore the hardcoded `for a_std in (0.0, 5e-4, ...)`
    and this fails.
    """
    seen = []

    def fake_backtest(days, mp, quiet=True, workers=1, ladder=None):
        seen.append((mp.alpha_std, mp.informed_horizon))
        return {"pnl": {c.name: [1.0, 2.0, 3.0, 4.0] for c in LADDER},
                "agg": {c.name: {"inf_share": 0.1, "pnl": 1.0, "dd": 1.0}
                        for c in LADDER},
                "cal": {"predictable_share": 0.5}}

    a_levels, h_levels = [0.007, 0.009], [7, 11]
    real_grid = analysis.GRID
    monkeypatch.setattr(analysis, "backtest", fake_backtest)
    monkeypatch.setattr(analysis, "GRID",
                        [("alpha_std", a_levels), ("informed_horizon", h_levels)])
    rows = analysis._interaction(1, 1)

    assert seen == [(a, h) for a in a_levels for h in h_levels], seen
    assert len(rows) == 4
    assert {r["alpha_std"] for r in rows} == set(a_levels)
    assert {r["informed_horizon"] for r in rows} == set(h_levels)
    # ...and the real grid still yields 12 cells. Restore GRID explicitly rather
    # than monkeypatch.undo(), which would also un-stub `backtest` and run 12
    # real simulations inside a test that is meant to be instant.
    seen.clear()
    monkeypatch.setattr(analysis, "GRID", real_grid)
    analysis._interaction(1, 1)
    assert len(seen) == 12, seen


def test_grid_values_rejects_an_unknown_field_by_name():
    with pytest.raises(KeyError) as e:
        analysis._grid_values("alpha_stnd")
    assert "alpha_stnd" in str(e.value)
    assert "alpha_std" in str(e.value), "the error should list what IS swept"


def test_ref_rung_is_present_in_the_ladder_with_a_rung_below_it():
    """`_REF_IDX` feeds `below_ref`, and `paired` needs a real comparison partner.
    Renaming a rung in `LADDER` used to surface as an IndexError inside the first
    interaction cell; it now fails at import."""
    assert analysis._REF in [c.name for c in LADDER]
    assert analysis._REF_IDX >= 1, "no rung below the reference to compare against"
