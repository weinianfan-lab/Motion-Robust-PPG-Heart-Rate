import numpy as np
import pytest


def test_ppg_normalization_centers_current_window_and_marks_flat():
    from ppg_hr.preprocessing.normalization import normalize_ppg

    window = np.tile(np.array([-1.0, 1.0]), 256).reshape(1, 512)
    original = window.copy()
    normalized, flat = normalize_ppg(window)
    np.testing.assert_array_equal(window, original)
    np.testing.assert_allclose(normalized, original, rtol=1e-6)
    assert normalized.dtype == np.float32
    assert not flat
    normalized, flat = normalize_ppg(np.full((1, 512), 37.0))
    assert flat
    assert np.isfinite(normalized).all()
    np.testing.assert_array_equal(normalized, 0)


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_nonfinite_ppg_blocks_normalization(value):
    from ppg_hr.errors import DataValidationError
    from ppg_hr.preprocessing.normalization import normalize_ppg

    window = np.zeros((1, 512))
    window[0, 0] = value
    with pytest.raises(DataValidationError, match="finite"):
        normalize_ppg(window)


def test_acc_scale_fits_unique_continuous_training_samples_and_preserves_amplitude():
    from ppg_hr.preprocessing.normalization import fit_acc_scale, normalize_acc

    split = {"train": ["S1", "S2"], "val": ["S3"], "test": ["S4"]}
    records = {"S1": np.array([[0.0, 2.0]] * 3), "S2": np.array([[0.0, 4.0, 0.0, 4.0]] * 3)}
    scale = fit_acc_scale(records, split)
    # Six unique time samples per axis: (2 * 1**2 + 4 * 2**2) / 6 == 3.
    assert scale.value == pytest.approx(np.sqrt(3))
    assert scale.metadata["fit_subjects"] == ["S1", "S2"]
    assert scale.metadata["unique_time_samples"] == 6
    assert len(scale.metadata["data_hash"]) == 64
    assert fit_acc_scale(dict(reversed(list(records.items()))), split).metadata == scale.metadata
    small = np.tile(np.array([-1.0, 1.0]), (3, 128))
    large = small * 5 + np.array([[10], [20], [30]])
    np.testing.assert_allclose(normalize_acc(large, scale), normalize_acc(small, scale) * 5)
    np.testing.assert_allclose(normalize_acc(large, scale).mean(axis=1), 0, atol=1e-7)
    changed = {**records, "S1": records["S1"] + 1}
    assert fit_acc_scale(changed, split).metadata["data_hash"] != scale.metadata["data_hash"]


@pytest.mark.parametrize("unauthorized", ["S3", "S4", "S99"])
def test_acc_fitter_denies_validation_test_and_unknown_subjects(unauthorized):
    from ppg_hr.errors import SplitLeakageError
    from ppg_hr.preprocessing.normalization import fit_acc_scale

    split = {"train": ["S1"], "val": ["S3"], "test": ["S4"]}
    records = {"S1": np.zeros((3, 10)), unauthorized: np.ones((3, 10))}
    with pytest.raises(SplitLeakageError, match="training"):
        fit_acc_scale(records, split)


def test_acc_fitter_denies_missing_training_record_and_window_stack():
    from ppg_hr.errors import DataValidationError, SplitLeakageError
    from ppg_hr.preprocessing.normalization import fit_acc_scale

    split = {"train": ["S1", "S2"], "val": ["S3"], "test": ["S4"]}
    with pytest.raises(SplitLeakageError, match="missing"):
        fit_acc_scale({"S1": np.zeros((3, 10))}, split)
    with pytest.raises(DataValidationError, match="continuous"):
        fit_acc_scale({"S1": np.zeros((2, 3, 256)), "S2": np.zeros((3, 10))}, split)


def test_constant_acc_has_safe_fixed_scale():
    from ppg_hr.preprocessing.normalization import fit_acc_scale, normalize_acc

    split = {"train": ["S1"], "val": ["S2"], "test": ["S3"]}
    scale = fit_acc_scale({"S1": np.full((3, 20), 9.8)}, split)
    assert scale.metadata["flat_training_acc"]
    assert scale.value > 0
    result = normalize_acc(np.full((3, 256), 9.8), scale)
    assert np.isfinite(result).all()
    np.testing.assert_allclose(result, 0, atol=1e-6)
