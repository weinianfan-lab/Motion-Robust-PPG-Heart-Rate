"""Hand-calculated synthetic checks for the frozen official-reference diagnostic."""

import csv
import importlib.util
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pytest

from ppg_hr.data.adapter import file_sha256

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "audit_official_reference.py"
SPEC = importlib.util.spec_from_file_location("official_reference_audit", SCRIPT)
reference_audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(reference_audit)


def payload(subject="S1", *, sample_indices=False):
    peaks = np.array([0.5, 1.5, 3.5, 5.5, 7.5, 8.5, 10.5, 12.5])
    return {
        "subject": subject,
        "signal": {
            "wrist": {"BVP": np.zeros((1024, 1)), "ACC": np.zeros((512, 3))},
            "chest": {"ECG": np.zeros((11200, 1))},
        },
        "activity": np.ones(64),
        "label": np.array([37.5, 40.0, 40.0]),
        "rpeaks": (peaks * 700).astype(int) if sample_indices else peaks,
    }


def write_subject(folder, value):
    path = folder / f"{value['subject']}.pkl"
    path.write_bytes(pickle.dumps(value))
    return path


def extraction_receipt(tmp_path):
    extracted = tmp_path / "extracted"
    extracted.mkdir()
    files = []
    for number in range(1, 16):
        path = write_subject(extracted, payload(f"S{number}"))
        files.append({"path": path.name, "bytes": path.stat().st_size, "sha256": file_sha256(path)})
    receipt = {"status": "ready", "crc_verified": True, "sha256": "b" * 64, "files": files}
    return extracted, receipt


def run_audit(monkeypatch, extracted, receipt, output):
    (extracted / ".acquisition-status.json").write_text(json.dumps(receipt), encoding="utf-8")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(SCRIPT),
            "--extracted-root",
            str(extracted),
            "--output-root",
            str(output),
            "--trust-local-pickle",
        ],
    )
    reference_audit.main()


@pytest.mark.parametrize("rate", [4, 32, 64, 700])
@pytest.mark.parametrize(
    "seconds,extra,inclusive,strict,residual_hops,residual_extra",
    [
        (0, 0, 0, 0, 0, 0),
        (8, -1, 0, 0, 4, -1),
        (8, 0, 1, 0, 0, 0),
        (12, -1, 2, 2, 1, -1),
        (12, 0, 3, 2, 0, 0),
        (12, 1, 3, 3, 0, 1),
    ],
)
def test_native_rate_counts_obey_hand_checked_endpoint_cases(
    rate, seconds, extra, inclusive, strict, residual_hops, residual_extra
):
    result = reference_audit.counts(seconds * rate + extra, rate)
    assert result["complete_end_le_N"] == inclusive
    assert result["complete_end_lt_N"] == strict
    assert result["residual_after_last_complete"] == residual_hops * 2 * rate + residual_extra
    assert result["samples"] == seconds * rate + extra
    assert result["duration_s"] == seconds + extra / rate


@pytest.mark.parametrize("sample_indices", [False, True])
def test_four_rr_methods_match_hand_calculation_at_fixed_zero_origin(tmp_path, sample_indices):
    path = write_subject(tmp_path, payload(sample_indices=sample_indices))
    details, rows = reference_audit.inspect(path, "S1", file_sha256(path))
    # RR = [1,2,2,2,1,2,2] s, so instantaneous HR = [60,30,30,30,60,30,30].
    expected = {
        "within_window_rr": [37.5, 40.0, 40.0],
        "right_endpoint": [37.5, 37.5, 37.5],
        "left_endpoint": [42.0, 37.5, 37.5],
        "midpoint": [37.5, 36.0, 36.0],
    }
    assert [row["window_start_s"] for row in rows] == [0, 2, 4]
    assert [row["window_end_s"] for row in rows] == [8, 10, 12]
    assert all(row["reason"] == "" for row in rows)
    for name, values in expected.items():
        np.testing.assert_allclose([row[name] for row in rows], values, rtol=0, atol=1e-12)
    assert details["diagnostic_only"] is True
    assert details["official_rule_confirmed"] is False
    expected_unit = (
        "ecg_sample_indices_from_integer_values_and_record_extent"
        if sample_indices
        else "seconds_from_fractional_values_and_record_duration"
    )
    assert details["rpeaks"]["inferred_unit"] == expected_unit
    assert details["reference_diagnostics"]["within_window_rr"]["matches_atol_1e_6"] == 3


def test_label_values_cannot_select_unit_origin_or_rr_method(tmp_path):
    value = payload()
    value["label"] = np.array([999.0, -1.0, 13.0])
    path = write_subject(tmp_path, value)
    details, rows = reference_audit.inspect(path, "S1", file_sha256(path))
    assert details["proposed_label_window_origin_s"] == 0
    assert set(details["reference_diagnostics"]) == {
        "within_window_rr",
        "right_endpoint",
        "left_endpoint",
        "midpoint",
    }
    assert [row["within_window_rr"] for row in rows] == [37.5, 40.0, 40.0]
    assert [row["midpoint"] for row in rows] == [37.5, 36.0, 36.0]
    assert [row["official_reference_bpm"] for row in rows] == [999.0, -1.0, 13.0]


def test_ambiguous_peak_units_remain_unresolved_even_with_matching_hr_labels(tmp_path):
    value = payload()
    value["rpeaks"] = np.array([1, 2, 3, 4])
    value["label"] = np.array([60.0, 60.0, 60.0])
    path = write_subject(tmp_path, value)
    details, rows = reference_audit.inspect(path, "S1", file_sha256(path))
    assert details["rpeaks"]["inferred_unit"] == "unresolved"
    assert len(rows) == 3
    assert all(row["reason"] == "rpeak_unit_unresolved" for row in rows)
    assert details["rpeaks"]["unit_inference_status"] == "unresolved"
    assert all(stats["n_comparable"] == 0 for stats in details["reference_diagnostics"].values())


@pytest.mark.parametrize(
    "mutation,duplicates,decreasing,reasons",
    [
        ("duplicate", 1, 0, ["duplicate_rpeaks"]),
        ("decreasing", 0, 1, ["decreasing_rpeaks"]),
        ("nonadjacent_duplicate", 1, 1, ["duplicate_rpeaks", "decreasing_rpeaks"]),
    ],
)
def test_invalid_rpeak_order_preserves_every_label_without_repair(
    tmp_path, mutation, duplicates, decreasing, reasons
):
    value = payload(sample_indices=True)
    if mutation == "duplicate":
        value["rpeaks"][2] = value["rpeaks"][1]
    elif mutation == "decreasing":
        value["rpeaks"][[2, 3]] = value["rpeaks"][[3, 2]]
    else:
        value["rpeaks"][3] = value["rpeaks"][1]
    path = write_subject(tmp_path, value)
    before = path.read_bytes()
    details, rows = reference_audit.inspect(path, "S1", file_sha256(path))
    assert path.read_bytes() == before
    assert details["rpeaks"]["duplicate_count"] == duplicates
    assert details["rpeaks"]["decreasing_count"] == decreasing
    assert details["rpeaks"]["unit_inference_status"] == "blocked_by_invalid_rpeaks"
    assert details["rpeaks"]["validation_reasons"] == reasons
    assert len(rows) == details["label_count"] == 3
    assert [row["label_index"] for row in rows] == [0, 1, 2]
    assert [row["window_start_s"] for row in rows] == [0, 2, 4]
    assert [row["official_reference_bpm"] for row in rows] == [37.5, 40.0, 40.0]
    for row in rows:
        assert row["reason"] == ";".join(reasons)
        for method in ("within_window_rr", "right_endpoint", "left_endpoint", "midpoint"):
            assert row[method] is None
    assert all(
        stats["n_labels"] == 3 and stats["n_comparable"] == 0
        for stats in details["reference_diagnostics"].values()
    )


@pytest.mark.parametrize(
    "peaks,reason",
    [(np.array([]), "insufficient_rpeaks"), (np.array([0.5, np.nan]), "nonfinite_rpeaks")],
)
def test_unusable_rpeaks_still_have_json_safe_full_denominators(tmp_path, peaks, reason):
    value = payload()
    value["rpeaks"] = peaks
    path = write_subject(tmp_path, value)
    details, rows = reference_audit.inspect(path, "S1", file_sha256(path))
    assert len(rows) == 3
    assert all(row["reason"] == reason for row in rows)
    json.dumps(details, allow_nan=False)
    json.dumps(rows, allow_nan=False)


def test_hash_gate_rejects_before_unpickling(tmp_path):
    path = tmp_path / "S1.pkl"
    path.write_bytes(b"not a pickle")
    with pytest.raises(RuntimeError, match="hash differs"):
        reference_audit.inspect(path, "S1", "0" * 64)


def test_cli_requires_explicit_pickle_trust(monkeypatch, tmp_path):
    monkeypatch.setattr(sys, "argv", [str(SCRIPT), "--extracted-root", str(tmp_path)])
    with pytest.raises(SystemExit) as error:
        reference_audit.main()
    assert error.value.code == 2


@pytest.mark.parametrize("mutation", ["crc", "status", "missing_subject", "duplicate_subject"])
def test_receipt_gates_cannot_be_skipped(monkeypatch, tmp_path, mutation):
    extracted, receipt = extraction_receipt(tmp_path)
    if mutation == "crc":
        receipt["crc_verified"] = False
    elif mutation == "status":
        receipt["status"] = "incomplete"
    elif mutation == "missing_subject":
        receipt["files"].pop()
    else:
        receipt["files"].append(dict(receipt["files"][0]))
    output = tmp_path / "output"
    with pytest.raises(RuntimeError):
        run_audit(monkeypatch, extracted, receipt, output)
    assert not output.exists()


def test_receipt_paths_cannot_escape_extracted_root(monkeypatch, tmp_path):
    extracted, receipt = extraction_receipt(tmp_path)
    outside = write_subject(tmp_path, payload())
    receipt["files"][0] = {
        "path": "../S1.pkl",
        "bytes": outside.stat().st_size,
        "sha256": file_sha256(outside),
    }
    output = tmp_path / "output"
    with pytest.raises(RuntimeError, match="escapes"):
        run_audit(monkeypatch, extracted, receipt, output)
    assert not output.exists()


@pytest.mark.parametrize("bad_subject", [None, "S6"])
def test_complete_synthetic_cohort_records_exact_provenance_and_csv_identity(
    monkeypatch, tmp_path, bad_subject
):
    extracted, receipt = extraction_receipt(tmp_path)
    if bad_subject:
        value = payload(bad_subject, sample_indices=True)
        value["rpeaks"][[2, 3]] = value["rpeaks"][[3, 2]]
        path = write_subject(extracted, value)
        receipt["files"][5].update(bytes=path.stat().st_size, sha256=file_sha256(path))
    output = tmp_path / "output"
    run_audit(monkeypatch, extracted, receipt, output)
    run = next(output.glob("reference-audit-*"))
    report = json.loads((run / "reference_audit.json").read_text(encoding="utf-8"))
    assert report["status"] == "DIAGNOSTIC_COMPLETE_ALIGNMENT_NOT_CERTIFIED"
    assert report["origin_s"] == 0
    assert report["label_total"] == 45
    assert report["paper_label_total"] == 64697
    assert [subject["subject_id"] for subject in report["subjects"]] == [
        f"S{i}" for i in range(1, 16)
    ]
    provenance = report["provenance"]
    assert provenance["script"]["sha256"] == file_sha256(SCRIPT)
    protocol = SCRIPT.parent.parent / "docs" / "m1_reference_audit_protocol.md"
    assert provenance["protocol"]["sha256"] == file_sha256(protocol)
    receipt_path = extracted / ".acquisition-status.json"
    assert provenance["source_receipt"]["sha256"] == file_sha256(receipt_path)
    assert provenance["source_receipt"]["path"] == str(receipt_path.resolve())
    assert provenance["source_receipt"]["archive_sha256"] == "b" * 64
    artifact = report["diagnostic_csv"]
    assert artifact["path"] == "reference_diagnostics.csv"
    assert artifact["count"] == report["label_total"] == 45
    assert artifact["sha256"] == file_sha256(run / artifact["path"])
    assert artifact["subject_row_counts"] == {f"S{i}": 3 for i in range(1, 16)}
    assert artifact["method_comparable_counts"] == {
        "within_window_rr": 42 if bad_subject else 45,
        "right_endpoint": 42 if bad_subject else 45,
        "left_endpoint": 42 if bad_subject else 45,
        "midpoint": 42 if bad_subject else 45,
    }
    with (run / artifact["path"]).open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 45
    assert rows[0]["subject_id"] == "S1" and rows[-1]["subject_id"] == "S15"
    for row in rows:
        assert row["reason"] == ("decreasing_rpeaks" if row["subject_id"] == bad_subject else "")
        if row["subject_id"] == bad_subject:
            assert all(
                row[method] == ""
                for method in ("within_window_rr", "right_endpoint", "left_endpoint", "midpoint")
            )
