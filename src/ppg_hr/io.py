"""Atomic artifacts with explicit finite JSON values and unique run folders."""

import csv
import io
import json
import os
import re
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4


def atomic_text(path: str | Path, text: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", newline="", dir=path.parent, delete=False
        ) as handle:
            temporary = Path(handle.name)
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def write_json(path: str | Path, value: object) -> None:
    serialized = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
    atomic_text(path, serialized + "\n")


def write_csv(path: str | Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError("CSV requires at least one row to define its columns")
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=list(rows[0]), extrasaction="raise")
    writer.writeheader()
    writer.writerows(rows)
    atomic_text(path, buffer.getvalue())


def new_run_dir(output_root: str | Path, prefix: str) -> Path:
    if re.fullmatch(r"[a-zA-Z][a-zA-Z0-9_-]*", prefix) is None:
        raise ValueError("Run prefix must be an alphanumeric name, not a path")
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    path = Path(output_root) / f"{prefix}-{stamp}-{uuid4().hex[:8]}"
    path.mkdir(parents=True, exist_ok=False)
    return path
