"""Registration CLI must reject untrusted or wrong archives without extraction."""

import subprocess
import sys
from pathlib import Path


def test_registration_requires_explicit_trust(tmp_path):
    script = Path(__file__).resolve().parents[1] / "scripts/register_dataset.py"
    destination = tmp_path / "uncreated"
    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--archive",
            str(tmp_path / "data.zip"),
            "--destination",
            str(destination),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2
    assert "--trust-official-archive" in result.stderr
    assert not destination.exists()


def test_registration_rejects_wrong_archive_before_writing(tmp_path):
    script = Path(__file__).resolve().parents[1] / "scripts/register_dataset.py"
    archive = tmp_path / "wrong.zip"
    archive.write_bytes(b"not the official archive")
    destination = tmp_path / "uncreated"
    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--archive",
            str(archive),
            "--destination",
            str(destination),
            "--trust-official-archive",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2
    assert "byte count" in result.stderr
    assert not destination.exists()
