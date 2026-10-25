"""Read-only environment observations and content identities for local runs."""

import hashlib
import importlib.metadata
import json
import platform
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path


def _command(arguments: list[str]) -> dict:
    if shutil.which(arguments[0]) is None:
        return {"available": False, "exit_code": None, "output": "not installed"}
    try:
        result = subprocess.run(arguments, capture_output=True, text=True, timeout=15)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"available": True, "exit_code": None, "output": str(exc)}
    return {
        "available": True,
        "exit_code": result.returncode,
        "output": (result.stdout + result.stderr).strip(),
    }


def environment_report() -> dict:
    dependencies = {}
    for name in ("numpy", "scipy", "matplotlib", "pyyaml", "pytest", "ruff", "torch"):
        try:
            dependencies[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            dependencies[name] = None
    core = all(dependencies[name] for name in ("numpy", "scipy", "matplotlib", "pyyaml"))
    return {
        "status": "READY_CPU" if core and sys.version_info[:2] == (3, 11) else "BLOCKED",
        "observed_at_utc": datetime.now(UTC).isoformat(),
        "python": {"version": platform.python_version(), "executable": sys.executable},
        "system": platform.platform(),
        "dependencies": dependencies,
        "git": _command(["git", "--version"]),
        "uv": _command(["uv", "--version"]),
        "gpu": _command(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"]
        ),
        "torch_cuda_execution": "NOT_VERIFIED; model-stage dependency deferred",
        "training_performed": False,
    }


def hash_json(value: object) -> str:
    data = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(data.encode()).hexdigest()


def provenance(config: dict) -> dict:
    root = Path(__file__).resolve().parents[2]
    sources = sorted((root / "src").rglob("*.py"))
    sources.extend(
        root / name for name in ("pyproject.toml", "uv.lock", "PPG_ACC_ENGINEERING_SPEC.md")
    )
    hashes = {
        path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sources
        if path.is_file()
    }
    head = _command(["git", "-C", str(root), "rev-parse", "HEAD"])
    state = _command(["git", "-C", str(root), "status", "--porcelain"])
    return {
        "config_sha256": hash_json(config),
        "source_hashes": hashes,
        "code_and_lock_sha256": hash_json(hashes),
        "git_commit": head["output"] if head["exit_code"] == 0 else None,
        "git_dirty": bool(state["output"]) if state["exit_code"] == 0 else None,
        "git_identity_note": "Content hashes retain identity even on an unborn or dirty branch",
        "environment": environment_report(),
    }
