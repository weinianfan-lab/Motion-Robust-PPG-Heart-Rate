"""Audit presentation preserves complete local evidence without repeating window rows."""

import copy
import csv
import hashlib
import json

from ppg_hr.cli import _audit_artifacts
from ppg_hr.data.audit import audit_dataset


def report_with_two_subjects():
    subjects = []
    for name, indices in (("S1", [0, 1]), ("S2", [0])):
        rows = [
            {
                "sample_id": f"{name}:{index:06d}",
                "subject_id": name,
                "window_index": index,
                "label_index": index,
                "window_start_s": float(index * 2),
                "eval_eligible": False,
                "reason": ["alignment_unverified", 'detail: comma, quote " and newline\n尾部'],
            }
            for index in indices
        ]
        if name == "S2":
            rows[0]["tail_samples"] = 17
        subjects.append(
            {
                "subject_id": name,
                "status": "BLOCKED",
                "blockers": ["label_origin_unverified"],
                "sha256": "a" * 64,
                "signals": {
                    "signal.wrist.BVP": {"shape": [768, 1], "nan_count": 0},
                    "signal.wrist.ACC": {"shape": [384, 3], "nan_count": 0},
                },
                "window_counts": {"bvp": 3, "acc": 3, "labels": len(indices)},
                "manifest_summary": {"windows": len(indices), "eval_eligible": 0},
                "manifest": rows,
            }
        )
    return {
        "status": "BLOCKED",
        "expected_subjects": [f"S{i}" for i in range(1, 16)],
        "discovered_subjects": ["S1", "S2"],
        "missing_subjects": [f"S{i}" for i in range(3, 16)],
        "unexpected_subjects": [],
        "blockers": ["missing_subjects", "label_origin_unverified"],
        "source_verification": {
            "label_origin_verified": False,
            "tail_rule_verified": False,
            "blockers": ["label_origin_unverified", "tail_rule_unverified"],
        },
        "local_detail": {"keep_every_field": ["中文", 123]},
        "subjects": subjects,
    }


def test_all_manifest_rows_fields_and_reason_sequences_survive_csv_export(tmp_path):
    report = report_with_two_subjects()
    _audit_artifacts(tmp_path, report)
    manifest_path = tmp_path / "window_manifest.csv"
    assert manifest_path.exists()
    with manifest_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 3
    assert [row["sample_id"] for row in rows] == ["S1:000000", "S1:000001", "S2:000000"]
    originals = [row for subject in report["subjects"] for row in subject["manifest"]]
    for original, exported in zip(originals, rows, strict=True):
        for key, value in original.items():
            if key == "reason":
                assert json.loads(exported[key]) == value
            else:
                assert exported[key] == str(value)
    assert rows[2]["tail_samples"] == "17"


def test_json_references_cohort_csv_with_correct_counts_hash_and_other_fields(tmp_path):
    report = report_with_two_subjects()
    original = copy.deepcopy(report)
    _audit_artifacts(tmp_path, report)
    assert report == original
    saved = json.loads((tmp_path / "data_audit.json").read_text(encoding="utf-8"))
    digest = hashlib.sha256((tmp_path / "window_manifest.csv").read_bytes()).hexdigest()
    assert saved["manifest_artifact"]["count"] == 3
    assert saved["manifest_artifact"]["path"] == "window_manifest.csv"
    assert saved["manifest_artifact"]["sha256"] == digest
    for before, after, count in zip(original["subjects"], saved["subjects"], [2, 1], strict=True):
        assert "manifest" not in after
        assert after["manifest_artifact"]["count"] == count
        assert after["manifest_artifact"]["cohort_count"] == 3
        assert after["manifest_artifact"]["path"] == "window_manifest.csv"
        assert after["manifest_artifact"]["sha256"] == digest
        assert {key: value for key, value in after.items() if key != "manifest_artifact"} == {
            key: value for key, value in before.items() if key != "manifest"
        }
    assert {
        key: value for key, value in saved.items() if key not in ("subjects", "manifest_artifact")
    } == {key: value for key, value in original.items() if key != "subjects"}


def test_markdown_shows_all_expected_subjects_status_counts_and_detail_links(tmp_path):
    _audit_artifacts(tmp_path, report_with_two_subjects())
    markdown = (tmp_path / "data_audit.md").read_text(encoding="utf-8")
    assert "Status: **BLOCKED**" in markdown
    assert "| Subject |" in markdown
    assert "| S1 | BLOCKED |" in markdown
    assert "| S2 | BLOCKED |" in markdown
    assert "| S15 | MISSING |" in markdown
    assert "768" in markdown and "384" in markdown
    assert "label_origin_unverified" in markdown
    assert "tail_rule_unverified" in markdown
    assert "(data_audit.json)" in markdown
    assert "(window_manifest.csv)" in markdown
    assert "```json" not in markdown
    assert "S1:000000" not in markdown
    assert '"manifest"' not in markdown


def test_absent_data_has_explicit_zero_manifest_count_and_no_csv(tmp_path):
    report = audit_dataset(tmp_path / "missing")
    _audit_artifacts(tmp_path, report)
    saved = json.loads((tmp_path / "data_audit.json").read_text(encoding="utf-8"))
    assert not (tmp_path / "window_manifest.csv").exists()
    assert saved["manifest_artifact"] == {"path": None, "count": 0, "sha256": None}
    markdown = (tmp_path / "data_audit.md").read_text(encoding="utf-8")
    assert "Manifest rows: **0**" in markdown
    assert "data_root_missing" in markdown
    assert "| S1 | MISSING |" in markdown
    assert "| S15 | MISSING |" in markdown
    assert "(data_audit.json)" in markdown


def test_empty_subject_manifest_is_replaced_by_an_explicit_zero_row_reference(tmp_path):
    report = report_with_two_subjects()
    report["subjects"] = [report["subjects"][0]]
    report["subjects"][0]["manifest"] = []
    report["subjects"][0]["manifest_summary"] = {"windows": 0, "eval_eligible": 0}
    _audit_artifacts(tmp_path, report)
    saved = json.loads((tmp_path / "data_audit.json").read_text(encoding="utf-8"))
    assert "manifest" not in saved["subjects"][0]
    assert saved["subjects"][0]["manifest_artifact"]["count"] == 0
    assert saved["subjects"][0]["manifest_artifact"]["path"] is None
    assert not (tmp_path / "window_manifest.csv").exists()
