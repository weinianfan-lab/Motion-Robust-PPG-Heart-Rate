import json
from dataclasses import FrozenInstanceError

import numpy as np
import pytest


def test_signal_window_has_no_reference_fields():
    from ppg_hr.contracts import SignalWindow

    args = ("S1:0", "S1", 0.0, 8.0, np.zeros((1, 512)), np.zeros((3, 256)))
    window = SignalWindow(*args)
    with pytest.raises(TypeError):
        SignalWindow(*args, hr_bpm=80.0)
    with pytest.raises(FrozenInstanceError):
        window.start_s = 2.0


def test_json_rejects_nan_preserving_previous_artifact(tmp_path):
    from ppg_hr.io import write_json

    path = tmp_path / "nested" / "result.json"
    write_json(path, {"status": "previous"})
    with pytest.raises(ValueError):
        write_json(path, {"bad": float("nan")})
    assert json.loads(path.read_text()) == {"status": "previous"}
    assert len(list(path.parent.iterdir())) == 1


def test_run_directories_preserve_outputs_and_csv_roundtrips(tmp_path):
    import csv

    from ppg_hr.io import new_run_dir, write_csv

    first = new_run_dir(tmp_path, "demo")
    second = new_run_dir(tmp_path, "demo")
    assert first != second
    write_csv(first / "rows.csv", [{"sample_id": "S1:0", "value": None}])
    with (first / "rows.csv").open(encoding="utf-8", newline="") as handle:
        assert list(csv.DictReader(handle)) == [{"sample_id": "S1:0", "value": ""}]


def test_run_prefix_cannot_escape_output_root(tmp_path):
    from ppg_hr.io import new_run_dir

    with pytest.raises(ValueError):
        new_run_dir(tmp_path, "../outside")


def test_provenance_git_identity_uses_source_repository_not_callers_cwd(tmp_path, monkeypatch):
    import subprocess

    from ppg_hr import doctor

    project = tmp_path / "source-project"
    other = tmp_path / "caller-project"
    source = project / "src" / "ppg_hr" / "doctor.py"
    source.parent.mkdir(parents=True)
    source.write_text("# source identity fixture", encoding="utf-8")
    other.mkdir()
    for repository in (project, other):
        subprocess.run(["git", "init", str(repository)], check=True, capture_output=True)
    monkeypatch.setattr(doctor, "__file__", str(source))
    monkeypatch.chdir(other)
    result = doctor.provenance({})
    assert result["git_dirty"] is True  # Source has an untracked file; caller repository is empty.
    assert "src/ppg_hr/doctor.py" in result["source_hashes"]
