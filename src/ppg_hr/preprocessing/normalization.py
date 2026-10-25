"""Window PPG normalization and a train-only common ACC amplitude scale."""

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

from ppg_hr.data.splits import validate_subject_split
from ppg_hr.errors import DataValidationError, SplitLeakageError


def _epsilon(value: float) -> float:
    if not np.isfinite(value) or value <= 0:
        raise DataValidationError("epsilon must be finite and positive")
    return float(value)


def _array(value: np.ndarray, shape: tuple[int, ...] | None = None) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype.kind not in "fiu" or not np.isfinite(array).all():
        raise DataValidationError("normalization requires finite numeric samples")
    if shape is not None and array.shape != shape:
        raise DataValidationError(f"expected signal shape {shape}, got {array.shape}")
    return array.astype(np.float64)


def normalize_ppg(ppg: np.ndarray, *, epsilon: float = 1e-8) -> tuple[np.ndarray, bool]:
    """Normalize only this complete [1,512] window; keep flat windows finite.

    Returns ``(normalized_float32, flat_signal_flag)``. A flat flag means the
    population standard deviation is at or below epsilon, not exclusion.
    """
    epsilon = _epsilon(epsilon)
    array = _array(ppg, (1, 512))
    centered = array - array.mean()
    deviation = float(np.std(centered, ddof=0))
    normalized = centered / (deviation + epsilon)
    if not np.isfinite(normalized).all():
        raise DataValidationError("PPG normalization produced nonfinite values")
    return normalized.astype(np.float32), deviation <= epsilon


@dataclass(frozen=True)
class ACCScale:
    """One common scale and auditable training sample identity."""

    value: float
    metadata: dict


def fit_acc_scale(
    records: Mapping[str, np.ndarray], split: Mapping[str, list[str]], *, epsilon: float = 1e-8
) -> ACCScale:
    """Fit one shared RMS scale from unique continuous TRAIN records only.

    Each mapping value is the original continuous ``[3,N]`` array for one
    training subject, with every time sample exactly once, never a window
    stack or concatenated overlapping windows. The adapter owns provenance of
    this continuous input; numeric values alone cannot prove its time identity.
    Remove each record's per-axis mean, pool squared components, and divide by
    ``3 * total_time_samples``. This is a project choice, not a fitted test rule.
    """
    epsilon = _epsilon(epsilon)
    if not all(role in split for role in ("train", "val", "test")):
        raise SplitLeakageError("training scale requires a complete train/val/test split")
    validate_subject_split(split["train"], split["val"], split["test"])
    allowed = set(split["train"])
    supplied = set(records)
    if supplied - allowed:
        raise SplitLeakageError(f"non-training subject access denied: {sorted(supplied - allowed)}")
    if allowed - supplied:
        raise SplitLeakageError(f"missing training subjects: {sorted(allowed - supplied)}")

    subjects = sorted(
        allowed, key=lambda item: (int(item[1:]), item) if item[1:].isdigit() else (0, item)
    )
    fingerprint = hashlib.sha256()
    sample_counts = {}
    sum_squared = 0.0
    for subject in subjects:
        array = _array(records[subject])
        if array.ndim != 2 or array.shape[0] != 3 or array.shape[1] == 0:
            raise DataValidationError("ACC fitting requires a nonempty continuous [3,N] record")
        sample_counts[subject] = int(array.shape[1])
        header = json.dumps({"subject_id": subject, "shape": list(array.shape)}, sort_keys=True)
        fingerprint.update(header.encode("utf-8"))
        fingerprint.update(np.ascontiguousarray(array, dtype="<f8").tobytes())
        centered = array - array.mean(axis=1, keepdims=True)
        sum_squared += float(np.sum(centered * centered))
    count = sum(sample_counts.values())
    raw_scale = float(np.sqrt(sum_squared / (3 * count)))
    if not np.isfinite(raw_scale):
        raise DataValidationError("training ACC scale is nonfinite")
    scale = max(raw_scale, epsilon)
    return ACCScale(
        scale,
        {
            "fit_subjects": subjects,
            "data_hash": fingerprint.hexdigest(),
            "hash_representation": "subject/shape JSON + contiguous little-endian float64 values",
            "unique_time_samples": count,
            "samples_by_subject": sample_counts,
            "method": "continuous_record_axis_demean_then_pooled_component_rms",
            "epsilon": epsilon,
            "flat_training_acc": raw_scale <= epsilon,
            "scale": scale,
        },
    )


def normalize_acc(acc: np.ndarray, scale: ACCScale) -> np.ndarray:
    """Demean each axis of [3,256], retaining relative dynamic amplitude."""
    array = _array(acc, (3, 256))
    if not np.isfinite(scale.value) or scale.value <= 0:
        raise DataValidationError("ACC scale must be finite and positive")
    normalized = (array - array.mean(axis=1, keepdims=True)) / scale.value
    result = normalized.astype(np.float32)
    if not np.isfinite(result).all():
        raise DataValidationError("ACC normalization produced nonfinite values")
    return result
