"""Continuous numeric caches; real preparation is gated by the full data audit."""

import json
from hashlib import sha256
from pathlib import Path

import numpy as np

from ppg_hr.data.adapter import ADAPTER_VERSION, SubjectRecord, validate_payload
from ppg_hr.data.audit import audit_dataset
from ppg_hr.data.windows import build_manifest, extract_window
from ppg_hr.errors import DataAlignmentError, DataBlockedError, DataValidationError
from ppg_hr.io import new_run_dir, write_json

PIPELINE = {"id": "native_unprocessed_v1", "bvp_hz": 64, "acc_hz": 32}


def _numeric_digest(arrays: dict[str, np.ndarray]) -> str:
    digest = sha256()
    for name, array in sorted(arrays.items()):
        digest.update(name.encode())
        digest.update(str((array.shape, array.dtype.str)).encode())
        # Hash contiguous chunks to avoid duplicating a full subject in memory.
        for start in range(0, len(array), 65536):
            digest.update(array[start : start + 65536].tobytes())
    return digest.hexdigest()


def write_numeric_cache(record: SubjectRecord, manifest: list[dict], cache_root: Path) -> Path:
    """Low-level writer for verified fixtures; real CLI must pass audit first.

    One continuous array per modality is written, without overlapping window
    copies. Cache identity covers source bytes when available, adapter, and
    the unprocessed-native-rates pipeline. Every call creates a unique folder.
    """
    if not manifest or any(not row.get("alignment_verified") for row in manifest):
        raise DataBlockedError("verified alignment is required before writing a numeric cache")
    expected = build_manifest(record, alignment_verified=True)
    if manifest != expected:
        raise DataAlignmentError("manifest differs from the complete validated source record")
    arrays = {"bvp": record.bvp, "acc": record.acc, "labels": record.labels}
    if record.activity is not None:
        arrays["activity"] = record.activity
    identity = {
        "raw_sha256": record.metadata.get("raw_sha256"),
        "numeric_sha256": _numeric_digest(arrays),
        "adapter_version": ADAPTER_VERSION,
        "pipeline": PIPELINE,
    }
    cache_key = sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    folder = new_run_dir(Path(cache_root), f"{record.subject_id}-{cache_key[:12]}")
    write_json(folder / "status.json", {"status": "WRITING"})
    try:
        for name, array in arrays.items():
            if array.dtype.kind not in "iuf":
                raise DataValidationError("numeric cache refuses non-numeric arrays")
            with (folder / f"{name}.npy").open("xb") as handle:
                np.save(handle, array, allow_pickle=False)
        write_json(
            folder / "metadata.json",
            {
                "subject_id": record.subject_id,
                "identity": identity,
                "cache_key": cache_key,
                "array_names": sorted(arrays),
                "manifest": manifest,
            },
        )
        write_json(folder / "status.json", {"status": "READY"})
    except Exception as exc:
        write_json(folder / "status.json", {"status": "FAILED", "error": str(exc)})
        raise
    return folder


def load_numeric_cache(cache_dir: Path) -> tuple[SubjectRecord, list[dict]]:
    """Never enable object unpickling, including for malformed cache files."""
    cache_dir = Path(cache_dir)
    try:
        status = json.loads((cache_dir / "status.json").read_text(encoding="utf-8"))
        if status.get("status") != "READY":
            raise DataValidationError("cache did not finish writing")
        metadata = json.loads((cache_dir / "metadata.json").read_text(encoding="utf-8"))
        names = metadata["array_names"]
        if set(names) not in ({"bvp", "acc", "labels"}, {"bvp", "acc", "labels", "activity"}):
            raise DataValidationError("unexpected numeric cache arrays")
        arrays = {
            name: np.load(cache_dir / f"{name}.npy", allow_pickle=False, mmap_mode="r")
            for name in names
        }
        identity = metadata["identity"]
        actual_key = sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        if (
            identity["numeric_sha256"] != _numeric_digest(arrays)
            or identity["adapter_version"] != ADAPTER_VERSION
            or identity["pipeline"] != PIPELINE
            or actual_key != metadata["cache_key"]
        ):
            raise DataValidationError("cache identity does not match content or adapter/pipeline")
        payload = {
            "subject": metadata["subject_id"],
            "label": arrays["labels"],
            "activity": arrays.get("activity"),
            "signal": {"wrist": {"BVP": arrays["bvp"], "ACC": arrays["acc"]}},
        }
        record = validate_payload(payload, metadata["subject_id"])
        manifest = metadata["manifest"]
        if manifest != build_manifest(record, alignment_verified=True):
            raise DataValidationError("cache manifest was altered")
        for row in manifest:
            extract_window(record, row)
    except (OSError, ValueError, KeyError, TypeError, DataAlignmentError) as exc:
        raise DataValidationError(f"invalid numeric cache: {exc}") from exc
    return record, manifest


def prepare_dataset(
    data_root: Path | None, cache_root: Path, trust_local_pickle: bool = False
) -> dict:
    audit = audit_dataset(data_root, trust_local_pickle=trust_local_pickle)
    # No current documentary evidence establishes exact label origin and tail
    # rules. Never bypass the audit or create misleading real-data caches.
    return {
        "status": "BLOCKED",
        "cache_root": str(Path(cache_root).resolve()),
        "cache_written": False,
        "blockers": audit["blockers"],
        "audit": audit,
    }
