import numpy as np
import pytest

from ppg_hr.showcase_experiment import compatible_protocols, validate_test_predictions


def test_aggregation_rejects_wrong_subject_index_and_replaced_official_label():
    split = {"train": ["S2"], "val": ["S3"], "test": ["S1"]}
    data = {"S1": {"labels": np.array([60.0, 80.0]), "activity": np.array(["1", "mixed"])}}
    rows = [
        {"subject_id": "S1", "label_index": "0", "sample_id": "S1:000000", "official_bpm": "60"},
        {"subject_id": "S1", "label_index": "1", "sample_id": "S1:000001", "official_bpm": "80"},
    ]
    rows[0]["activity_id"] = "1"
    rows[1]["activity_id"] = "mixed"
    validate_test_predictions(rows, split, data)
    for change in [
        {"subject_id": "S2"},
        {"label_index": "5"},
        {"official_bpm": "61"},
        {"sample_id": "S2:000000"},
        {"activity_id": "2"},
    ]:
        modified = [{**rows[0], **change}, rows[1]]
        with pytest.raises(ValueError):
            validate_test_predictions(modified, split, data)


def test_aggregation_requires_identical_data_split_lock_and_config():
    original = {
        "config": {"seed": 42},
        "split_sha256": "a",
        "cache_metadata_sha256": "b",
        "lock_sha256": "c",
    }
    compatible_protocols([original, dict(original)])
    for key in original:
        changed = {**original, key: "different"}
        with pytest.raises(ValueError):
            compatible_protocols([original, changed])
