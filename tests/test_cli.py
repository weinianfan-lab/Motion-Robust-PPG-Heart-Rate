import json
import subprocess
import sys
from pathlib import Path

import pytest


def invoke(*args):
    return subprocess.run(
        [sys.executable, "-m", "ppg_hr.cli", *map(str, args)],
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=90,
    )


def test_doctor_reports_actual_core_environment(tmp_path):
    result = invoke("doctor", "--output-root", tmp_path)
    assert result.returncode == 0, result.stderr
    report = json.loads(next(tmp_path.glob("doctor-*/doctor.json")).read_text())
    assert report["python"]["version"].startswith("3.11.")
    assert report["dependencies"]["numpy"] is not None
    assert report["training_performed"] is False


def test_missing_data_emits_blocked_artifact_and_nonzero(tmp_path):
    result = invoke("audit-data", "--data-root", tmp_path / "missing", "--output-root", tmp_path)
    assert result.returncode == 2, result.stderr
    report = json.loads(next(tmp_path.glob("audit-*/data_audit.json")).read_text())
    assert report["status"] == "BLOCKED"
    assert list(tmp_path.glob("audit-*/data_audit.md"))


def test_make_splits_saves_exact_protocol_without_overwrite(tmp_path):
    target = tmp_path / "splits.json"
    result = invoke("make-splits", "--protocol", "loso_v1", "--output", target)
    assert result.returncode == 0, result.stderr
    first = target.read_bytes()
    result = invoke("make-splits", "--protocol", "loso_v1", "--output", target)
    assert result.returncode == 0, result.stderr
    assert target.read_bytes() == first
    assert len(json.loads(first)["folds"]) == 15


def test_synthetic_demo_produces_complete_artifacts(tmp_path):
    import csv

    result = invoke(
        "demo",
        "--synthetic",
        "--scenario",
        "constant_hr",
        "--duration",
        "12",
        "--output-root",
        tmp_path,
    )
    assert result.returncode == 0, result.stderr
    run = next(tmp_path.glob("demo-*"))
    report = json.loads((run / "results.json").read_text())
    assert report["data_kind"] == "SYNTHETIC DATA"
    assert report["real_data_evaluated"] is False
    assert set(report["scenarios"]["constant_hr"]["models"]) == {"A1", "A2"}
    assert (run / "predictions.csv").stat().st_size > 0
    with (run / "predictions.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert all(float(row["available_time_s"]) >= 12 for row in rows)
    assert (run / "manifest.csv").stat().st_size > 0
    assert (run / "resolved_config.yaml").exists()
    assert (run / "provenance.json").exists()
    assert len(list((run / "figures").glob("*.png"))) == 1
    status = json.loads((run / "run_status.json").read_text())
    assert status["status"] == "COMPLETE"


@pytest.mark.parametrize("text", ["typo: 1", "seed: true", "duration_s: .nan"])
def test_invalid_demo_configuration_fails_clearly(tmp_path, text):
    config = tmp_path / "config.yaml"
    config.write_text(text, encoding="utf-8")
    result = invoke("demo", "--synthetic", "--config", config, "--output-root", tmp_path)
    assert result.returncode == 2
    assert "CONFIGURATION_ERROR" in result.stderr


def test_demo_requires_explicit_synthetic_flag(tmp_path):
    result = invoke("demo", "--output-root", tmp_path)
    assert result.returncode == 2
    assert not list(tmp_path.glob("demo-*"))


def test_prepare_missing_data_blocks_without_numeric_cache(tmp_path):
    config = tmp_path / "data.yaml"
    config.write_text(
        f'data_root: "{(tmp_path / "missing").as_posix()}"\n'
        f'cache_root: "{(tmp_path / "cache").as_posix()}"\n',
        encoding="utf-8",
    )
    result = invoke("prepare", "--config", config, "--output-root", tmp_path)
    assert result.returncode == 2, result.stderr
    assert not list(tmp_path.rglob("*.npy"))


def test_conflicting_existing_split_is_preserved(tmp_path):
    target = Path(tmp_path) / "splits.json"
    target.write_text('{"existing": true}', encoding="utf-8")
    result = invoke("make-splits", "--output", target)
    assert result.returncode == 2
    assert target.read_text() == '{"existing": true}'
