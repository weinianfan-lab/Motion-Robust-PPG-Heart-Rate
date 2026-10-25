"""Synthetic timestamp and declared-identity audit regressions."""

import json
import pickle

import numpy as np
import pytest

from ppg_hr.data.adapter import validate_payload
from ppg_hr.data.audit import audit_dataset
from ppg_hr.errors import DataValidationError


def payload(origin=1_700_000_000.0):
    return {
        "subject": "S1",
        "signal": {
            "wrist": {"BVP": np.zeros((768, 1)), "ACC": np.zeros((384, 3))},
        },
        "label": np.array([60.0, 75.0, 90.0]),
        "activity": np.ones(48),
        "timestamps": {
            "bvp": origin + np.arange(768) / 64,
            "acc": origin + np.arange(384) / 32,
            "activity": origin + np.arange(48) / 4,
        },
    }


def subject_report(tmp_path, value):
    (tmp_path / "S1.pkl").write_bytes(pickle.dumps(value))
    report = audit_dataset(tmp_path, trust_local_pickle=True)
    json.dumps(report, allow_nan=False)
    return report["subjects"][0]


@pytest.mark.parametrize("origin", [0.0, 1_700_000_000.0])
def test_common_absolute_origin_is_allowed_and_channel_endpoints_are_retained(tmp_path, origin):
    report = subject_report(tmp_path, payload(origin))
    timestamps = report["timestamp_audit"]
    assert timestamps["cross_channel_time_origin"]["status"] == "consistent"
    assert timestamps["cross_channel_time_origin"]["origin_s"] == origin
    for channel, last in (("bvp", 11.984375), ("acc", 11.96875), ("activity", 11.75)):
        assert timestamps["channels"][channel]["first_timestamp_s"] == origin
        assert timestamps["channels"][channel]["last_timestamp_s"] == origin + last
    assert "cross_channel_time_origin_mismatch" not in report["blockers"]
    assert "invalid_timestamps" not in report["blockers"]
    # A common clock origin alone still cannot certify label alignment.
    assert "label_origin_unverified" in report["blockers"]


@pytest.mark.parametrize("channel", ["bvp", "acc", "activity"])
def test_independent_channel_shift_blocks_even_at_epoch_scale(tmp_path, channel):
    value = payload()
    value["timestamps"][channel] += 1 / 64
    report = subject_report(tmp_path, value)
    assert "cross_channel_time_origin_mismatch" in report["blockers"]
    assert report["timestamp_audit"]["cross_channel_time_origin"]["status"] == "mismatch"
    assert report["manifest_summary"]["windows"] == 3
    assert all(not row["eval_eligible"] for row in report["manifest"])


@pytest.mark.parametrize("invalid", [np.nan, np.inf, -np.inf])
def test_nonfinite_first_timestamp_blocks_without_emitting_nonfinite_json(tmp_path, invalid):
    value = payload()
    value["timestamps"]["acc"][0] = invalid
    report = subject_report(tmp_path, value)
    timestamps = report["timestamp_audit"]
    assert "invalid_timestamps" in report["blockers"]
    assert timestamps["channels"]["acc"]["first_timestamp_s"] is None
    assert timestamps["channels"]["acc"]["last_timestamp_s"] == 1_700_000_011.96875
    assert timestamps["cross_channel_time_origin"]["status"] == "not_comparable"


def test_nonfinite_last_timestamp_is_recorded_and_blocked(tmp_path):
    value = payload()
    value["timestamps"]["bvp"][-1] = np.nan
    report = subject_report(tmp_path, value)
    stats = report["timestamp_audit"]["channels"]["bvp"]
    assert stats["first_timestamp_s"] == 1_700_000_000.0
    assert stats["last_timestamp_s"] is None
    assert "invalid_timestamps" in report["blockers"]


def test_single_timestamp_channel_cannot_claim_cross_channel_alignment(tmp_path):
    value = payload()
    value["timestamps"] = {"bvp": value["timestamps"]["bvp"]}
    report = subject_report(tmp_path, value)
    assert report["timestamp_audit"]["cross_channel_time_origin"]["status"] == "not_comparable"
    assert "cross_channel_time_origin_mismatch" not in report["blockers"]


def test_empty_timestamp_channel_has_no_endpoints_and_blocks(tmp_path):
    value = payload()
    value["timestamps"]["activity"] = np.array([])
    report = subject_report(tmp_path, value)
    timestamps = report["timestamp_audit"]
    assert timestamps["channels"]["activity"]["first_timestamp_s"] is None
    assert timestamps["channels"]["activity"]["last_timestamp_s"] is None
    assert timestamps["cross_channel_time_origin"]["status"] == "not_comparable"
    assert "invalid_timestamps" in report["blockers"]


@pytest.mark.parametrize("declared", ["S1", "1", 1, np.int64(1)])
def test_declared_subject_identity_is_recorded_without_new_blocker(tmp_path, declared):
    value = payload()
    value["subject"] = declared
    report = subject_report(tmp_path, value)
    assert report["subject_identity"]["status"] == "declared"
    assert report["subject_identity"]["declared_value"] == str(declared)
    assert report["subject_identity"]["expected_subject_id"] == "S1"
    assert "subject_identity_unverified" not in report["blockers"]


def test_missing_subject_is_supported_for_fixtures_but_blocks_file_audit(tmp_path):
    value = payload()
    del value["subject"]
    record = validate_payload(value, "S1")
    assert record.subject_id == "S1"
    assert record.metadata["subject_identity"]["status"] == "missing"
    report = subject_report(tmp_path, value)
    assert report["subject_identity"]["declared_value"] is None
    assert "subject_identity_unverified" in report["blockers"]
    assert report["manifest_summary"]["windows"] == 3


def test_mismatched_subject_still_raises_and_audit_retains_identity(tmp_path):
    value = payload()
    value["subject"] = "S2"
    with pytest.raises(DataValidationError, match="subject ID disagrees"):
        validate_payload(value, "S1")
    report = subject_report(tmp_path, value)
    assert "invalid_structure" in report["blockers"]
    assert report["subject_identity"]["status"] == "mismatch"
    assert report["subject_identity"]["declared_value"] == "S2"
    assert report["subject_identity"]["expected_subject_id"] == "S1"
