"""Hand-calculated diagnostic hypotheses; never real subject fixtures."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest


def module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "audit_reference_forensics.py"
    spec = importlib.util.spec_from_file_location("reference_forensics", path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


@pytest.mark.parametrize(
    ("closure", "expected"),
    [("left", 42000 / 490), ("both", (42000 / 490 + 120) / 2), ("neither", None), ("right", 120)],
)
def test_exact_boundaries(closure, expected):
    result = module().window_diagnostics(np.array([0, 210, 700, 1050, 1400]), 210, 1050)
    assert (
        result[closure] == pytest.approx(expected)
        if expected is not None
        else result[closure] is None
    )


def test_zero_rr_is_preserved_and_invalidates_unbounded_mean():
    peaks = np.array([0, 210, 421, 1472, 1472, 1822], dtype=np.int32)
    saved = peaks.copy()
    result = module().window_diagnostics(peaks, 0, 2000)
    assert result["left"] is None
    assert result["bounded_40_200"] == pytest.approx((200 + 42000 / 211 + 120) / 3)
    assert result["zero_rr_count"] == 1
    assert result["low_hr_count"] == 1
    assert result["bounded_rejected_count"] == 2
    np.testing.assert_array_equal(peaks, saved)


def test_bounded_rule_includes_exact_endpoints_and_reports_all_rejections():
    peaks = np.cumsum(np.array([0, 210, 1050, 209, 1051]))
    result = module().window_diagnostics(peaks, 0, 4000)
    assert result["bounded_40_200"] == 120
    assert result["bounded_retained_count"] == 2
    assert result["high_hr_count"] == result["low_hr_count"] == 1


def test_all_rejected_is_explicit_missing():
    result = module().window_diagnostics(np.array([0, 1, 2]), 0, 100)
    assert result["bounded_40_200"] is None
    assert result["left"] == 42000
    assert result["bounded_rejected_count"] == 2


@pytest.mark.parametrize("peaks", [np.array([0, 700, 600]), np.array([0.0, 700.5])])
def test_rejects_out_of_order_or_noninteger_positions(peaks):
    with pytest.raises(ValueError):
        module().window_diagnostics(peaks, 0, 5600)


def test_large_integer_origin_does_not_change_rr_math():
    m = module()
    peaks = np.array([0, 210, 700, 1050], dtype=np.int64)
    before = m.window_diagnostics(peaks, 0, 2000)
    after = m.window_diagnostics(peaks + 10**12, 10**12, 10**12 + 2000)
    assert before == after
