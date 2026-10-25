"""Read-only audit with explicit structural and source-evidence stop gates."""

import os
import re
from pathlib import Path

import numpy as np

from ppg_hr.data.adapter import (
    ACTIVITY_MAP,
    ADAPTER_VERSION,
    SOURCE_RULES,
    SubjectIdentityError,
    file_sha256,
    load_subject,
    numeric_stats,
)
from ppg_hr.data.windows import build_manifest, window_counts
from ppg_hr.errors import DataAlignmentError, DataValidationError

EXPECTED_SUBJECTS = [f"S{number}" for number in range(1, 16)]
# Official README III.1-III.3; nominal rates, not measured hardware clocks.
DOCUMENTED_CHANNEL_RATES_HZ = {
    "chest": dict.fromkeys(("ACC", "ECG", "EDA", "EMG", "RESP", "TEMP"), 700),
    "wrist": {"BVP": 64, "ACC": 32, "EDA": 4, "TEMP": 4},
}


def _channel_metadata(name: str, stats: dict) -> dict:
    parts = name.split(".")
    location, modality = (parts[1], parts[2].upper()) if len(parts) == 3 else (None, None)
    rate = DOCUMENTED_CHANNEL_RATES_HZ.get(location, {}).get(modality)
    shape = stats["shape"]
    duration = shape[0] / rate if shape and rate is not None else None
    rate_basis = (
        "official README III.1-III.3; documented nominal rate, not measured from timestamps"
    )
    duration_basis = "shape[0] / documented sample rate under uniform sampling assumption"
    kind = "sampled_waveform"
    if name == "rpeaks":
        kind = "event_vector"
        rate_basis = "R-peak events are not a uniformly sampled waveform"
        duration_basis = "event count does not determine signal recording duration"
    elif rate is None:
        kind = "unknown"
        rate_basis = "unknown modality or location; no documented sample rate assigned"
        duration_basis = "unknown sample rate; duration cannot be inferred from shape"
    elif not shape:
        duration_basis = "no sample axis in scalar value; duration cannot be inferred"
    raw_unit = (
        {"ACC": "1/64 g", "EDA": "uS", "TEMP": "degrees C"}.get(modality)
        if location == "wrist"
        else None
    )
    return {
        **stats,
        "signal_kind": kind,
        "documented_sample_rate_hz": rate,
        "sample_rate_basis": rate_basis,
        "duration_s": duration,
        "duration_basis": duration_basis,
        "units": "unknown",
        "documented_raw_unit": raw_unit,
        "units_basis": (
            "official README III.2 documents raw E4 CSV units; synchronized-pickle units/scaling "
            "not independently verified; no rescaling applied"
            if raw_unit is not None
            else "units of these stored values are not independently verified; no rescaling applied"
        ),
    }


def _discover(root: Path) -> dict[str, list[Path]]:
    found: dict[str, list[Path]] = {}
    # No recursive drive scan; recognize the official unpacked layout and direct root.
    bases = [root, root / "PPG_FieldStudy"]
    for base in bases:
        if not base.is_dir():
            continue
        candidates = list(base.glob("S*.pkl")) + list(base.glob("S*/S*.pkl"))
        for path in candidates:
            if re.fullmatch(r"S[1-9][0-9]*", path.stem) and path.is_file():
                paths = found.setdefault(path.stem, [])
                if path.resolve() not in paths:
                    paths.append(path.resolve())
    return found


def _subject_report(subject_id: str, path: Path, trust: bool) -> dict:
    report = {
        "subject_id": subject_id,
        "file": str(path),
        "sha256": file_sha256(path),
        "bytes": path.stat().st_size,
        "blockers": [],
        "status": "BLOCKED",
    }
    if not trust:
        report["blockers"].append("pickle_not_trusted")
        return report
    try:
        record = load_subject(path, subject_id, trust_local_pickle=True)
    except DataValidationError as exc:
        report["blockers"].append("invalid_structure")
        report["error"] = str(exc)
        if isinstance(exc, SubjectIdentityError):
            report["subject_identity"] = exc.subject_identity
        return report
    report.update(record.metadata)
    report["signals"] = {
        name: _channel_metadata(name, stats) for name, stats in report["signals"].items()
    }
    report["documented_channel_sample_rates_hz"] = {
        name: stats["documented_sample_rate_hz"] for name, stats in report["signals"].items()
    }
    report["signal_duration_s"] = {
        name: stats["duration_s"] for name, stats in report["signals"].items()
    }
    if record.metadata["subject_identity"]["status"] == "missing":
        report["blockers"].append("subject_identity_unverified")
    counts = window_counts(record)
    report.update(
        {
            "window_counts": counts,
            "duration_s": {
                "bvp": len(record.bvp) / 64,
                "acc": len(record.acc) / 32,
                "activity": len(record.activity) / 4 if record.activity is not None else None,
                "ecg": report["signal_duration_s"].get("signal.chest.ECG"),
            },
            "documented_sample_rates_hz": SOURCE_RULES["documented_rates_hz"],
            "units": {
                "bvp": "unknown; not specified in README",
                "acc": (
                    "unknown; raw E4 CSV documented as 1/64 g; "
                    "synchronized-pickle scaling not locally verified"
                ),
                "label": "BPM",
                "ecg": "unknown; not specified in README",
            },
            "label_stats": numeric_stats(record.labels),
            "activity_codes": sorted(
                np.unique(record.activity[np.isfinite(record.activity)]).astype(int).tolist()
            )
            if record.activity is not None
            else [],
            "activity_map": ACTIVITY_MAP,
            "tail_samples_after_last_complete_window": {
                "bvp": len(record.bvp) - (512 + 128 * (counts["bvp"] - 1))
                if counts["bvp"]
                else len(record.bvp),
                "acc": len(record.acc) - (256 + 64 * (counts["acc"] - 1))
                if counts["acc"]
                else len(record.acc),
            },
            "limitations": [
                "equal label counts do not prove a correct semantic time origin",
                "sample rates use documentary assumptions, not observed hardware clocks",
                "unknown reference sentinel conventions are not guessed or filtered by BPM range",
            ],
        }
    )
    if counts["bvp"] == 0 or counts["acc"] == 0:
        report["blockers"].append("no_complete_windows")
    if record.metadata["timestamp_audit"]["status"] == "available":
        if record.metadata["timestamp_audit"]["cross_channel_time_origin"]["status"] == "mismatch":
            report["blockers"].append("cross_channel_time_origin_mismatch")
        for stats in record.metadata["timestamp_audit"]["channels"].values():
            if (
                stats["duplicate_count"]
                or stats["decreasing_count"]
                or stats["nonfinite_count"]
                or stats["gap_count"]
                or stats["irregular_step_count"]
                or stats["length_mismatch"]
            ):
                report["blockers"].append("invalid_timestamps")
                break
    try:
        manifest = build_manifest(record)
    except DataAlignmentError as exc:
        report["blockers"].append("window_count_mismatch")
        report["alignment_error"] = str(exc)
        manifest = build_manifest(record, retain_mismatched=True)
    report["manifest_summary"] = {
        "windows": len(manifest),
        "input_invalid": sum(not row["input_valid"] for row in manifest),
        "reference_invalid": sum(not row["reference_valid"] for row in manifest),
        "eval_eligible": sum(row["eval_eligible"] for row in manifest),
        "flat_windows": sum(row["flat_signal_flag"] for row in manifest),
        "flat_window_fraction": sum(row["flat_signal_flag"] for row in manifest) / len(manifest)
        if manifest
        else None,
        "reference_outside_search_band": sum(
            row["reference_outside_search_band"] for row in manifest
        ),
        "mixed_activity": sum(row["activity_id"] == "mixed" for row in manifest),
        "unknown_activity": sum(row["activity_id"] == "unknown" for row in manifest),
    }
    report["manifest"] = manifest
    report["blockers"].extend(SOURCE_RULES["blockers"])
    return report


def audit_dataset(data_root: Path | None, trust_local_pickle: bool = False) -> dict:
    """Return a JSON-safe audit even if data is absent, partial, or untrusted.

    Hashing bytes is not deserialization or provenance authentication. Raw data
    is never modified; reports may contain local paths and stay local.
    """
    if data_root is None and os.environ.get("PPG_DALIA_ROOT"):
        data_root = Path(os.environ["PPG_DALIA_ROOT"])
    root = Path(data_root).expanduser().resolve() if data_root is not None else None
    report = {
        "status": "BLOCKED",
        "adapter_version": ADAPTER_VERSION,
        "data_root": str(root) if root is not None else None,
        "expected_subjects": EXPECTED_SUBJECTS,
        "discovered_subjects": [],
        "missing_subjects": EXPECTED_SUBJECTS.copy(),
        "unexpected_subjects": [],
        "subjects": [],
        "blockers": [],
        "source_verification": SOURCE_RULES,
        "pickle_trust_acknowledged": bool(trust_local_pickle),
        "hash_semantics": "file identity only; SHA-256 does not authenticate provenance",
    }
    if root is None or not root.is_dir():
        report["blockers"] = ["data_root_missing", *SOURCE_RULES["blockers"]]
        return report
    discovered = _discover(root)
    subjects = sorted(discovered, key=lambda name: int(name[1:]))
    report["discovered_subjects"] = subjects
    report["missing_subjects"] = [name for name in EXPECTED_SUBJECTS if name not in discovered]
    report["unexpected_subjects"] = [name for name in subjects if name not in EXPECTED_SUBJECTS]
    if report["missing_subjects"]:
        report["blockers"].append("missing_subjects")
    if report["unexpected_subjects"]:
        report["blockers"].append("unexpected_subjects")
    for name in subjects:
        paths = discovered[name]
        if len(paths) != 1:
            files = []
            for path in paths:
                try:
                    files.append({"file": str(path), "sha256": file_sha256(path)})
                except OSError as exc:
                    files.append({"file": str(path), "error": str(exc)})
            report["subjects"].append(
                {
                    "subject_id": name,
                    "files": files,
                    "blockers": ["duplicate_subject_files"],
                    "status": "BLOCKED",
                }
            )
            report["blockers"].append(f"{name}:duplicate_subject_files")
            continue
        try:
            subject = _subject_report(name, paths[0], trust_local_pickle)
        except OSError as exc:
            subject = {
                "subject_id": name,
                "file": str(paths[0]),
                "status": "BLOCKED",
                "blockers": ["file_unreadable"],
                "error": str(exc),
            }
        report["subjects"].append(subject)
        report["blockers"].extend(f"{name}:{reason}" for reason in subject["blockers"])
    report["blockers"].extend(SOURCE_RULES["blockers"])
    report["blockers"] = list(dict.fromkeys(report["blockers"]))
    return report
