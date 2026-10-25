import numpy as np
import pytest

from ppg_hr.data.adapter import SOURCE_RULES, SubjectRecord
from ppg_hr.errors import DataAlignmentError
from ppg_hr.showcase_data import engineering_manifest, make_windows


def record():
    return SubjectRecord(
        "S1",
        np.arange(768.0)[:, None],
        np.zeros((384, 3)),
        np.array([61.0, 73.0, 89.0]),
        np.ones(48),
    )


def test_engineering_mapping_preserves_labels_and_does_not_certify_official_generator():
    r = record()
    before = r.labels.copy()
    rows = engineering_manifest(r)
    assert len(rows) == 3 and all(x["eval_eligible"] for x in rows)
    assert rows[1]["bvp_start"] == 128 and rows[1]["acc_end"] == 320
    assert rows[-1]["window_end_s"] == 12
    assert all(not x["source_verified"] for x in rows)
    assert all(x["alignment_basis"] == "engineering_mapping_v1" for x in rows)
    assert not SOURCE_RULES["label_origin_verified"]
    np.testing.assert_array_equal(r.labels, before)
    p, a = make_windows(r.bvp[:, 0], r.acc)
    assert p.shape == (3, 1, 512) and a.shape == (3, 3, 256)
    np.testing.assert_array_equal(p[1, 0], np.arange(128, 640))


def test_engineering_mapping_refuses_missing_labels_and_mismatched_durations():
    r = record()
    r = SubjectRecord(r.subject_id, r.bvp, r.acc, r.labels[:-1], r.activity)
    with pytest.raises(DataAlignmentError):
        engineering_manifest(r)
    r = record()
    r = SubjectRecord(r.subject_id, r.bvp, r.acc[:-1], r.labels, r.activity)
    with pytest.raises(DataAlignmentError):
        engineering_manifest(r)


def test_engineering_mapping_retains_nonfinite_rows_as_explicit_ineligible():
    r = record()
    r.labels[1] = np.nan
    rows = engineering_manifest(r)
    assert len(rows) == 3 and not rows[1]["eval_eligible"]
    assert "nonfinite_reference" in rows[1]["reason"]


def test_engineering_mapping_checks_explicit_timestamps():
    from dataclasses import replace

    r = replace(
        record(),
        metadata={
            "timestamp_audit": {
                "status": "available",
                "channels": {"bvp": {"irregular_step_count": 1}},
            }
        },
    )
    with pytest.raises(DataAlignmentError):
        engineering_manifest(r)
