import numpy as np
import pytest

from ppg_hr.showcase_experiment import fit_training_statistics, summarize_errors


def test_statistics_do_not_read_held_out_labels():
    class Forbidden:
        def __getitem__(self, key):
            raise AssertionError("held-out labels accessed while fitting statistics")

    data = {"S1": {"labels": np.array([60.0, 80.0])}, "S2": Forbidden(), "S3": Forbidden()}
    metadata = {
        "subjects": {"S1": {"continuous_acc_component_sumsq": 12.0, "continuous_acc_samples": 4}}
    }
    split = {"train": ["S1"], "val": ["S2"], "test": ["S3"]}
    result = fit_training_statistics(data, metadata, split)
    assert result["target_mean"] == 70 and result["target_std"] == 10 and result["acc_scale"] == 1
    with pytest.raises(ValueError):
        fit_training_statistics(data, metadata, {"train": ["S1"], "val": ["S1"], "test": ["S3"]})


def test_metrics_keep_all_finite_errors_and_refuse_partial_scoring():
    result = summarize_errors([60, 80], [65, 70])
    assert result["n"] == 2 and result["mae"] == 7.5
    assert result["rmse"] == pytest.approx(np.sqrt(62.5))
    with pytest.raises(ValueError):
        summarize_errors([60, 80], [65, np.nan])
    with pytest.raises(ValueError):
        summarize_errors([60, 80], [65])
