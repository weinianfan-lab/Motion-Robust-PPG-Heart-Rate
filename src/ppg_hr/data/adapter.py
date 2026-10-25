"""PPG-DaLiA import boundary; pickle trust is an explicit caller decision.

Documented rates are not measured rates. No local-data alignment is certified
by this adapter merely because the shapes and window counts agree.
"""

import pickle
import re
from dataclasses import dataclass, field
from hashlib import sha256
from pathlib import Path
from typing import Any

import numpy as np

from ppg_hr.errors import DataBlockedError, DataValidationError

ADAPTER_VERSION = "dalia_v1_unverified_alignment"
SAMPLE_RATES = {"bvp": 64, "acc": 32, "activity": 4, "ecg": 700}
ACTIVITY_MAP = {
    0: "transient",
    1: "sitting",
    2: "stairs",
    3: "table_soccer",
    4: "cycling",
    5: "driving",
    6: "lunch_break",
    7: "walking",
    8: "working",
}
SOURCE_RULES = {
    "readme_url": "https://archive.ics.uci.edu/ml/machine-learning-databases/00495/readme.pdf",
    "readme_sha256": "018a3bab7f1d02261ad4a42720f2235ac20e1c3c0a052f02ef91208096d4fc1b",
    "verified_on": "2026-10-25",
    "documented_rates_hz": SAMPLE_RATES,
    "activity_map": ACTIVITY_MAP,
    "label_definition": "mean ECG-based instantaneous HR within 8 s, shifted 2 s",
    "label_origin_verified": False,
    "tail_rule_verified": False,
    "blockers": ["label_origin_unverified", "tail_rule_unverified"],
}


@dataclass(frozen=True)
class SubjectRecord:
    subject_id: str
    bvp: np.ndarray
    acc: np.ndarray
    labels: np.ndarray
    activity: np.ndarray | None
    metadata: dict = field(default_factory=dict)


class SubjectIdentityError(DataValidationError):
    """Keep the declared/file identity discrepancy in a blocked audit."""

    def __init__(self, identity: dict):
        self.subject_identity = identity
        super().__init__(
            f"subject ID disagrees with file identity {identity['expected_subject_id']}"
        )


def file_sha256(path: Path) -> str:
    digest = sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _numeric(value: Any, name: str) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype.kind not in "iuf":
        raise DataValidationError(f"{name} must be real numeric; got {array.dtype}")
    return array


def _vector(value: Any, name: str) -> np.ndarray:
    array = _numeric(value, name)
    if array.ndim == 2 and array.shape[1] == 1:
        array = array[:, 0]
    if array.ndim != 1:
        raise DataValidationError(f"{name} must have shape [N] or [N,1]")
    return array


def numeric_stats(array: np.ndarray) -> dict:
    """JSON-safe statistics: invalid values are counted, never silently hidden."""
    array = np.asarray(array)
    values = array[np.isfinite(array)].astype(np.float64)
    scale = float(np.max(np.abs(values))) if values.size else 0.0
    scaled = values / scale if scale else values
    result = {
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "nan_count": int(np.isnan(array).sum()),
        "positive_inf_count": int(np.isposinf(array).sum()),
        "negative_inf_count": int(np.isneginf(array).sum()),
        "finite_count": int(values.size),
        "minimum": float(values.min()) if values.size else None,
        "maximum": float(values.max()) if values.size else None,
        "mean": float(scaled.mean() * scale) if values.size else None,
        "std": float(scaled.std() * scale) if values.size else None,
    }
    if array.ndim >= 1 and len(array) > 1:
        pairs = np.isfinite(array[:-1]) & np.isfinite(array[1:])
        denominator = int(pairs.sum())
        result["unchanged_adjacent_fraction"] = (
            float(((array[1:] == array[:-1]) & pairs).sum() / denominator) if denominator else None
        )
    else:
        result["unchanged_adjacent_fraction"] = None
    return result


def _structure(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _structure(item) for key, item in value.items()}
    if isinstance(value, np.ndarray):
        return {"shape": list(value.shape), "dtype": str(value.dtype)}
    return {"type": type(value).__name__}


def validate_payload(payload: dict, subject_id: str) -> SubjectRecord:
    """Validate a caller-supplied mapping without loading files or resynchronizing.

    Missing activity is retained as unknown. ECG/R peaks only enter audit
    metadata and are never included in SubjectRecord signal inputs.
    """
    if not re.fullmatch(r"S[1-9][0-9]*", subject_id):
        raise DataValidationError("subject_id must be S followed by a positive integer")
    if not isinstance(payload, dict):
        raise DataValidationError("subject payload must be a dictionary")
    try:
        wrist = payload["signal"]["wrist"]
        bvp = _numeric(wrist["BVP"], "signal.wrist.BVP")
        acc = _numeric(wrist["ACC"], "signal.wrist.ACC")
        labels = _vector(payload["label"], "label")
    except (KeyError, TypeError) as exc:
        raise DataValidationError(f"missing or malformed required field: {exc}") from exc
    if bvp.ndim != 2 or bvp.shape[1] != 1:
        raise DataValidationError("signal.wrist.BVP must have shape [N,1]")
    if acc.ndim != 2 or acc.shape[1] != 3:
        raise DataValidationError("signal.wrist.ACC must have shape [N,3]")
    identity = {
        "status": "declared" if "subject" in payload else "missing",
        "declared_value": str(payload["subject"]) if "subject" in payload else None,
        "expected_subject_id": subject_id,
    }
    if "subject" in payload and identity["declared_value"] not in (subject_id, subject_id[1:]):
        identity["status"] = "mismatch"
        raise SubjectIdentityError(identity)
    rates = payload.get("sample_rates", {})
    if not isinstance(rates, dict):
        raise DataValidationError("sample_rates, when supplied, must be a mapping")
    for name, rate in rates.items():
        if name not in SAMPLE_RATES or rate != SAMPLE_RATES[name]:
            raise DataValidationError(f"unsupported declared sample rate: {name}={rate}")
    activity = None
    if payload.get("activity") is not None:
        activity = _vector(payload["activity"], "activity")
        finite_activity = activity[np.isfinite(activity)]
        if np.any(finite_activity != np.floor(finite_activity)):
            raise DataValidationError("activity codes must be integers")
        unknown = set(finite_activity.astype(int).tolist()) - set(ACTIVITY_MAP)
        if unknown:
            raise DataValidationError(f"undocumented activity codes: {sorted(unknown)}")
    for key, expected in (
        ("label_index", np.arange(len(labels))),
        ("label_start_s", np.arange(len(labels)) * 2.0),
    ):
        if key in payload and not np.array_equal(_vector(payload[key], key), expected):
            raise DataValidationError(f"{key} conflicts with proposed 8/2-second mapping")
    signals = {}
    for location, channels in payload["signal"].items():
        if not isinstance(channels, dict):
            raise DataValidationError(f"signal.{location} must be a channel mapping")
        for name, value in channels.items():
            array = _numeric(value, f"signal.{location}.{name}")
            signals[f"signal.{location}.{name}"] = numeric_stats(array)
    if "rpeaks" in payload:
        signals["rpeaks"] = numeric_stats(_vector(payload["rpeaks"], "rpeaks"))
    timestamps = {
        "status": "not_available",
        "assumption": "time axis constructed under documented uniform sampling assumption; "
        "packet loss cannot be fully detected without acquisition timestamps",
    }
    if "timestamps" in payload:
        timestamps = {"status": "available", "channels": {}}
        if not isinstance(payload["timestamps"], dict):
            raise DataValidationError("timestamps must be a channel mapping")
        for key, raw in payload["timestamps"].items():
            time = _vector(raw, f"timestamps.{key}")
            delta = np.diff(time)
            lengths = {
                "bvp": len(bvp),
                "acc": len(acc),
                "activity": len(activity) if activity is not None else 0,
            }
            if key not in lengths:
                raise DataValidationError(f"unsupported timestamp channel {key}")
            expected_step = 1 / SAMPLE_RATES[key]
            timestamps["channels"][str(key)] = {
                "count": len(time),
                "first_timestamp_s": float(time[0]) if len(time) and np.isfinite(time[0]) else None,
                "last_timestamp_s": float(time[-1])
                if len(time) and np.isfinite(time[-1])
                else None,
                "nonfinite_count": int((~np.isfinite(time)).sum()),
                "duplicate_count": int((delta == 0).sum()),
                "decreasing_count": int((delta < 0).sum()),
                "gap_count": int((delta > expected_step * 1.5).sum()),
                "irregular_step_count": int(
                    (~np.isclose(delta, expected_step, rtol=1e-5, atol=1e-8)).sum()
                ),
                "length_mismatch": len(time) != lengths[key],
                "assumed_timestamp_units": "seconds; optional adapter extension",
            }
        origins = [stats["first_timestamp_s"] for stats in timestamps["channels"].values()]
        origin_audit = {
            "status": "not_comparable",
            "channels": sorted(timestamps["channels"]),
            "origin_s": None,
            "absolute_tolerance_s": 1e-8,
            "relative_tolerance": 0.0,
        }
        if len(origins) >= 2 and all(origin is not None for origin in origins):
            # A relative tolerance would conceal channel shifts at Unix-epoch scale.
            common = max(origins) - min(origins) <= origin_audit["absolute_tolerance_s"]
            origin_audit["status"] = "consistent" if common else "mismatch"
            origin_audit["origin_s"] = min(origins) if common else None
        timestamps["cross_channel_time_origin"] = origin_audit
    return SubjectRecord(
        subject_id,
        bvp,
        acc,
        labels,
        activity,
        {
            "structure": _structure(payload),
            "signals": signals,
            "subject_identity": identity,
            "timestamp_audit": timestamps,
            "declared_sample_rates": rates,
            "sample_rate_basis": "official README; not measured from timestamps",
        },
    )


def load_subject(path: Path, subject_id: str, trust_local_pickle: bool = False) -> SubjectRecord:
    if not trust_local_pickle:
        raise DataBlockedError("explicit trust_local_pickle=True is required before unpickling")
    path = Path(path)
    try:
        with path.open("rb") as handle:
            payload = pickle.load(handle, encoding="latin1")
        record = validate_payload(payload, subject_id)
    except (OSError, ValueError, EOFError, pickle.UnpicklingError, TypeError) as exc:
        raise DataValidationError(f"cannot import {path.name}: {exc}") from exc
    record.metadata["raw_sha256"] = file_sha256(path)
    record.metadata["raw_file"] = str(path.resolve())
    return record
