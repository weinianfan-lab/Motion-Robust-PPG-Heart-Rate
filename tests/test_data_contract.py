"""Synthetic fixtures exercise data contracts, never participant records."""

import json
import pickle
from dataclasses import replace

import numpy as np
import pytest

from ppg_hr.data.adapter import load_subject, numeric_stats, validate_payload
from ppg_hr.data.audit import audit_dataset
from ppg_hr.data.prepare import load_numeric_cache, prepare_dataset, write_numeric_cache
from ppg_hr.data.windows import build_manifest, extract_window, reference_rows
from ppg_hr.errors import DataAlignmentError, DataBlockedError, DataValidationError


def payload(subject="S1"):
    """12 seconds: complete starts 0,2,4, with hand-indexable values."""
    return {
        "subject": subject,
        "signal": {
            "wrist": {
                "BVP": np.arange(768, dtype=np.float64)[:, None],
                "ACC": np.arange(1152, dtype=np.float64).reshape(384, 3),
            },
            "chest": {"ECG": np.zeros((8400, 1))},
        },
        "label": np.array([60.0, 75.0, 90.0]),
        "activity": np.ones((48, 1), dtype=np.int64),
        "rpeaks": np.arange(0, 8400, 700),
    }


def record():
    return validate_payload(payload(), "S1")


def test_hand_indexed_windows_use_exact_native_rates_and_reference_indices():
    subject = record()
    rows = build_manifest(subject, alignment_verified=True)
    assert len(rows) == 3
    second = rows[1]
    assert (second["window_start_s"], second["window_end_s"]) == (2.0, 10.0)
    assert (second["bvp_start"], second["bvp_end"]) == (128, 640)
    assert (second["acc_start"], second["acc_end"]) == (64, 320)
    assert second["label_index"] == 1
    assert second["reference_time_s"] == 6.0
    assert second["available_time_s"] == 10.0
    window = extract_window(subject, second)
    assert window.ppg.shape == (1, 512)
    assert window.acc.shape == (3, 256)
    assert window.ppg.dtype == window.acc.dtype == np.float32
    assert (window.ppg[0, 0], window.ppg[0, -1]) == (128, 639)
    assert window.acc[:, 0].tolist() == [192, 193, 194]
    assert reference_rows(subject, rows)[1].hr_bpm == 75.0
    assert not hasattr(window, "hr_bpm")
    assert not hasattr(window, "activity_id")


@pytest.mark.parametrize(
    "field,value",
    [("label_index", 2), ("bvp_start", 129), ("window_start_s", 3.0), ("reference_time_s", 7.0)],
)
def test_tampered_window_metadata_is_blocked(field, value):
    subject = record()
    row = dict(build_manifest(subject, alignment_verified=True)[1])
    row[field] = value
    with pytest.raises(DataAlignmentError):
        extract_window(subject, row)


def test_mismatched_counts_raise_instead_of_truncating():
    subject = replace(record(), labels=np.array([60.0, 75.0]))
    with pytest.raises(DataAlignmentError, match="3.*3.*2"):
        build_manifest(subject, alignment_verified=True)


def test_unverified_semantic_alignment_cannot_become_eligible():
    subject = record()
    rows = build_manifest(subject)
    assert len(rows) == 3
    assert all(not row["eval_eligible"] for row in rows)
    assert all("alignment_unverified" in row["reason"] for row in rows)
    # Equal-length numerical label permutation cannot be detected from lengths.
    changed = replace(subject, labels=np.roll(subject.labels, 1))
    assert len(build_manifest(changed, alignment_verified=True)) == 3


@pytest.mark.parametrize("mutation", ["missing_bvp", "bad_acc", "object_bvp", "rate", "subject"])
def test_invalid_structures_and_declared_rates_raise(mutation):
    value = payload()
    if mutation == "missing_bvp":
        del value["signal"]["wrist"]["BVP"]
    elif mutation == "bad_acc":
        value["signal"]["wrist"]["ACC"] = np.zeros((384, 2))
    elif mutation == "object_bvp":
        value["signal"]["wrist"]["BVP"] = np.array([[object()]], dtype=object)
    elif mutation == "rate":
        value["sample_rates"] = {"bvp": 32}
    else:
        value["subject"] = "S2"
    with pytest.raises(DataValidationError):
        validate_payload(value, "S1")


def test_invalid_inputs_keep_rows_and_flat_and_out_of_band_references_are_retained():
    value = payload()
    value["signal"]["wrist"]["BVP"][:] = 0
    value["label"] = np.array([20.0, 300.0, np.nan])
    subject = validate_payload(value, "S1")
    rows = build_manifest(subject, alignment_verified=True)
    assert [row["eval_eligible"] for row in rows] == [True, True, False]
    assert all(row["flat_signal_flag"] for row in rows)
    assert rows[-1]["reason"] == ["nonfinite_reference"]
    value["signal"]["wrist"]["BVP"][0] = np.nan
    rows = build_manifest(validate_payload(value, "S1"), alignment_verified=True)
    assert len(rows) == 3
    assert not rows[0]["input_valid"]
    assert rows[1]["eval_eligible"]


def test_activity_majority_mixed_and_unknown_do_not_remove_evaluation_windows():
    subject = record()
    mixed = np.tile([1, 2], 24)
    rows = build_manifest(replace(subject, activity=mixed), alignment_verified=True)
    assert all(row["activity_id"] == "mixed" for row in rows)
    assert all(row["activity_purity"] == 0.5 for row in rows)
    assert all(row["eval_eligible"] for row in rows)
    rows = build_manifest(replace(subject, activity=None), alignment_verified=True)
    assert all(row["activity_id"] == "unknown" for row in rows)
    assert all(row["eval_eligible"] for row in rows)


def test_pickle_requires_trust_before_deserialization(tmp_path):
    # Invalid bytes prove the refusal occurs before any unpickling attempt.
    path = tmp_path / "S1.pkl"
    path.write_bytes(b"not a pickle")
    with pytest.raises(DataBlockedError, match="trust"):
        load_subject(path, "S1")
    path.write_bytes(pickle.dumps(payload()))
    assert load_subject(path, "S1", trust_local_pickle=True).bvp.shape == (768, 1)


def test_missing_partial_and_untrusted_audits_are_structured_and_have_real_hashes(tmp_path):
    absent = audit_dataset(None)
    assert absent["status"] == "BLOCKED"
    assert absent["missing_subjects"] == [f"S{i}" for i in range(1, 16)]
    for name in ("S10", "S2"):
        folder = tmp_path / name
        folder.mkdir()
        (folder / f"{name}.pkl").write_bytes(b"not trusted")
    result = audit_dataset(tmp_path)
    assert result["status"] == "BLOCKED"
    assert result["discovered_subjects"] == ["S2", "S10"]
    assert len(result["subjects"][0]["sha256"]) == 64
    assert "pickle_not_trusted" in result["subjects"][0]["blockers"]
    json.dumps(result, allow_nan=False)


def test_trusted_synthetic_subject_audit_records_invalid_values_and_unknown_time_axis(tmp_path):
    folder = tmp_path / "S1"
    folder.mkdir()
    value = payload()
    value["signal"]["wrist"]["BVP"][0] = np.nan
    (folder / "S1.pkl").write_bytes(pickle.dumps(value))
    result = audit_dataset(tmp_path, trust_local_pickle=True)
    report = result["subjects"][0]
    assert report["signals"]["signal.wrist.BVP"]["nan_count"] == 1
    assert report["window_counts"] == {"bvp": 3, "acc": 3, "labels": 3}
    assert report["timestamp_audit"]["status"] == "not_available"
    assert report["manifest_summary"]["windows"] == 3
    assert result["status"] == "BLOCKED"
    json.dumps(result, allow_nan=False)


def test_prepare_blocked_does_not_write_numeric_cache(tmp_path):
    cache = tmp_path / "cache"
    result = prepare_dataset(None, cache)
    assert result["status"] == "BLOCKED"
    assert not cache.exists()


def test_numeric_cache_is_continuous_safe_and_rejects_object_payload(tmp_path):
    subject = record()
    rows = build_manifest(subject, alignment_verified=True)
    cache = write_numeric_cache(subject, rows, tmp_path)
    restored, saved_rows = load_numeric_cache(cache)
    assert restored.bvp.shape == (768, 1)
    assert restored.acc.shape == (384, 3)
    assert len(saved_rows) == 3
    np.testing.assert_array_equal(
        extract_window(restored, saved_rows[2]).ppg, extract_window(subject, rows[2]).ppg
    )
    # Windows keeps memory-mapped files open; corrupt an unopened independent cache.
    cache = write_numeric_cache(subject, rows, tmp_path)
    np.save(cache / "bvp.npy", np.array([{"unsafe": True}], dtype=object))
    with pytest.raises(DataValidationError):
        load_numeric_cache(cache)


def test_cache_rejects_unverified_alignment_before_writing(tmp_path):
    with pytest.raises(DataBlockedError):
        write_numeric_cache(record(), build_manifest(record()), tmp_path / "blocked")
    assert not (tmp_path / "blocked").exists()


def test_cache_rejects_finite_signal_changes_and_wrong_pipeline_identity(tmp_path):
    subject = record()
    rows = build_manifest(subject, alignment_verified=True)
    cache = write_numeric_cache(subject, rows, tmp_path)
    changed = subject.bvp.copy()
    changed[0] += 10
    np.save(cache / "bvp.npy", changed, allow_pickle=False)
    with pytest.raises(DataValidationError, match="identity"):
        load_numeric_cache(cache)
    cache = write_numeric_cache(subject, rows, tmp_path)
    metadata = json.loads((cache / "metadata.json").read_text())
    metadata["identity"]["pipeline"]["id"] = "some_other_pipeline"
    (cache / "metadata.json").write_text(json.dumps(metadata))
    with pytest.raises(DataValidationError, match="identity"):
        load_numeric_cache(cache)


def test_timestamp_audit_detects_gaps_and_length_mismatches(tmp_path):
    value = payload()
    time = np.arange(768, dtype=float) / 64
    time[500:] += 1 / 64
    value["timestamps"] = {"bvp": time, "acc": np.arange(383) / 32}
    folder = tmp_path / "S1"
    folder.mkdir()
    (folder / "S1.pkl").write_bytes(pickle.dumps(value))
    report = audit_dataset(tmp_path, trust_local_pickle=True)["subjects"][0]
    assert "invalid_timestamps" in report["blockers"]
    assert report["timestamp_audit"]["channels"]["bvp"]["gap_count"] == 1
    assert report["timestamp_audit"]["channels"]["acc"]["length_mismatch"]


def test_duplicate_subject_files_are_blocked_and_each_file_is_hashed(tmp_path):
    (tmp_path / "S1.pkl").write_bytes(b"first")
    (tmp_path / "S1").mkdir()
    (tmp_path / "S1" / "S1.pkl").write_bytes(b"second")
    report = audit_dataset(tmp_path)["subjects"][0]
    assert report["blockers"] == ["duplicate_subject_files"]
    assert len(report["files"]) == 2
    assert all(len(item["sha256"]) == 64 for item in report["files"])


def test_count_mismatch_audit_preserves_missing_reference_and_incomplete_input_rows(tmp_path):
    folder = tmp_path / "S1"
    folder.mkdir()
    value = payload()
    value["label"] = np.array([60.0, 75.0])
    value["signal"]["wrist"]["ACC"] = value["signal"]["wrist"]["ACC"][:320]
    (folder / "S1.pkl").write_bytes(pickle.dumps(value))
    report = audit_dataset(tmp_path, trust_local_pickle=True)["subjects"][0]
    assert "window_count_mismatch" in report["blockers"]
    assert len(report["manifest"]) == 3
    assert "missing_reference" in report["manifest"][2]["reason"]
    assert "incomplete_input_window" in report["manifest"][2]["reason"]
    assert all(not row["eval_eligible"] for row in report["manifest"])


def test_large_finite_amplitude_stats_stay_json_safe():
    stats = numeric_stats(np.array([1e308, 1e308]))
    assert stats["mean"] == 1e308
    assert stats["std"] == 0
    json.dumps(stats, allow_nan=False)
