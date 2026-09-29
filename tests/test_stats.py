"""Statistics helpers: the t critical values, the paired difference, and the
win rate. The calibration row in the report is only meaningful if these are right.
"""
import math

import pytest

from stats import paired, summarize, t_crit95


def test_t_crit_table_is_positive_and_decreasing_in_df():
    """t_crit95 is indexed by *degrees of freedom*, and shrinks as df grows."""
    for df in range(1, 31):
        assert t_crit95(df) > 1.96
    assert t_crit95(1) > t_crit95(5) > t_crit95(30)


def test_t_crit_matches_known_values():
    assert t_crit95(9) == pytest.approx(2.262, abs=0.01)
    assert t_crit95(19) == pytest.approx(2.093, abs=0.01)
    assert t_crit95(29) == pytest.approx(2.045, abs=0.01)


def test_t_crit_falls_back_for_large_samples():
    assert t_crit95(500) == pytest.approx(1.96, abs=0.01)
    assert t_crit95(10**6) == pytest.approx(1.96, abs=0.01)


def test_summarize_uses_the_sample_sd():
    x = [1.0, 2.0, 3.0, 4.0]
    s = summarize(x)
    assert s["mean"] == pytest.approx(2.5)
    assert s["n"] == 4
    assert s["sd"] == pytest.approx(math.sqrt(5.0 / 3.0))
    assert s["sem"] == pytest.approx(math.sqrt(5.0 / 3.0) / 2, abs=1e-12)


def test_summarize_handles_a_single_sample_without_dividing_by_zero():
    s = summarize([42.0])
    assert s["mean"] == 42.0
    assert s["sem"] == 0.0
    assert math.isfinite(s["lo"]) and math.isfinite(s["hi"])


def test_paired_removes_the_shared_seed_variation():
    """If the only difference between the two arms is a per-seed constant, the
    paired difference must be exactly zero -- the whole reason for CRN."""
    base = [100.0, -50.0, 3.0, 7.0, 220.0]
    a = base
    b = [x + 5.0 for x in base]
    p = paired(a, b)
    assert p["delta"] == pytest.approx(5.0)
    assert p["sem"] == pytest.approx(0.0, abs=1e-12)
    assert p["win"] == 1.0


def test_paired_returns_native_python_types():
    """numpy input must not leak numpy scalars into the report dict."""
    import numpy as np
    p = paired(np.array([1.0, 2.0, 3.0, 4.0]), np.array([2.0, 3.0, 4.0, 5.0]))
    for k in ("delta", "mean", "sem", "t", "win"):
        assert type(p[k]) is float, (k, type(p[k]))
    assert type(p["sig"]) is bool


def test_paired_reports_a_ci_that_contains_the_point_estimate():
    a = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    b = [1.5, 1.0, 4.0, 2.0, 6.0, 3.0]
    p = paired(a, b)
    assert p["lo"] <= p["delta"] <= p["hi"]
    assert p["n"] == 6


def test_paired_flags_a_real_difference_and_not_a_noise_one():
    rng = __import__("numpy").random.default_rng(0)
    base = list(rng.normal(0, 100, 60))
    real = paired(base, [x + 20.0 for x in base])
    noise = paired(base, list(rng.normal(0, 100, 60)))
    assert real["sig"] is True
    assert noise["sig"] is False


def test_paired_carries_delta_for_the_report_table():
    """Regression: the report reads s['delta']; paired() used to omit it."""
    p = paired([1.0, 2.0, 3.0], [2.0, 3.0, 4.0])
    assert "delta" in p and p["delta"] == pytest.approx(1.0)
    assert p["delta"] == p["mean"]
    # a deterministic difference is the strongest possible result, not a crash
    assert math.isinf(p["t"]) and p["sig"] is True
