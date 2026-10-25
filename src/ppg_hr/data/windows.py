"""Auditable left-closed, right-open 8-second windows at 2-second hops."""

import numpy as np

from ppg_hr.contracts import ReferenceRow, SignalWindow
from ppg_hr.data.adapter import SubjectRecord
from ppg_hr.errors import DataAlignmentError


def window_counts(record: SubjectRecord) -> dict:
    return {
        "bvp": max(0, (len(record.bvp) - 512) // 128 + 1),
        "acc": max(0, (len(record.acc) - 256) // 64 + 1),
        "labels": len(record.labels),
    }


def _indices(subject_id: str, index: int) -> dict:
    return {
        "sample_id": f"{subject_id}:{index:06d}",
        "subject_id": subject_id,
        "window_index": index,
        "label_index": index,
        "window_start_s": float(2 * index),
        "window_end_s": float(2 * index + 8),
        "reference_time_s": float(2 * index + 4),
        "available_time_s": float(2 * index + 8),
        "bvp_start": 128 * index,
        "bvp_end": 128 * index + 512,
        "acc_start": 64 * index,
        "acc_end": 64 * index + 256,
        "activity_start": 8 * index,
        "activity_end": 8 * index + 32,
    }


def _activity(record: SubjectRecord, index: int) -> tuple[str, float | None]:
    if record.activity is None:
        return "unknown", None
    start = 8 * index
    labels = record.activity[start : start + 32]
    if len(labels) != 32 or not np.isfinite(labels).all():
        return "unknown", None
    values, counts = np.unique(labels, return_counts=True)
    winner = int(np.argmax(counts))
    purity = float(counts[winner] / 32)
    return (str(int(values[winner])) if purity >= 0.8 else "mixed"), purity


def build_manifest(
    record: SubjectRecord, alignment_verified: bool = False, *, retain_mismatched: bool = False
) -> list[dict]:
    """Keep all complete windows, including excluded rows and hard finite inputs.

    ``alignment_verified=True`` is suitable for explicitly timed synthetic
    fixtures. The real-data audit never sets it: numeric origin and tail rules
    need independent source evidence before a future version may do so.
    Equal-length label value permutations cannot be detected by this function.
    ``retain_mismatched`` is audit-only: union candidate indices are retained
    with missing-reference/incomplete-input reasons; no evaluation is enabled.
    """
    counts = window_counts(record)
    if retain_mismatched and alignment_verified:
        raise DataAlignmentError("audit-only mismatched manifests cannot verify alignment")
    if len(set(counts.values())) != 1 and not retain_mismatched:
        raise DataAlignmentError(
            f"window counts disagree: bvp={counts['bvp']}, acc={counts['acc']}, "
            f"labels={counts['labels']}; no truncation or offset correction applied"
        )
    rows = []
    for index in range(max(counts.values())):
        row = _indices(record.subject_id, index)
        bvp = record.bvp[row["bvp_start"] : row["bvp_end"]]
        acc = record.acc[row["acc_start"] : row["acc_end"]]
        # Float32 is the inference contract; overflow at conversion is invalid too.
        limit = np.finfo(np.float32).max
        complete = bvp.shape == (512, 1) and acc.shape == (256, 3)
        input_valid = bool(
            complete
            and np.isfinite(bvp).all()
            and np.isfinite(acc).all()
            and (np.abs(bvp) <= limit).all()
            and (np.abs(acc) <= limit).all()
        )
        reference_present = index < len(record.labels)
        reference_valid = bool(reference_present and np.isfinite(record.labels[index]))
        flat = bool(len(bvp) and np.isfinite(bvp).all() and np.all(bvp == bvp[0]))
        activity_id, purity = _activity(record, index)
        reasons = []
        if not complete:
            reasons.append("incomplete_input_window")
        elif not input_valid:
            reasons.append("nonfinite_or_unrepresentable_input")
        if not reference_present:
            reasons.append("missing_reference")
        elif not reference_valid:
            reasons.append("nonfinite_reference")
        if not alignment_verified:
            reasons.append("alignment_unverified")
        row.update(
            {
                "input_valid": input_valid,
                "reference_valid": reference_valid,
                "eval_eligible": input_valid and reference_valid and alignment_verified,
                "alignment_verified": alignment_verified,
                "source_verified": alignment_verified,
                "reason": reasons,
                "activity_id": activity_id,
                "activity_purity": purity,
                "flat_signal_flag": flat,
                "reference_outside_search_band": bool(
                    reference_valid and not 30 <= record.labels[index] <= 240
                ),
            }
        )
        rows.append(row)
    return rows


def _validate_row(record: SubjectRecord, row: dict) -> int:
    index = row.get("window_index")
    if type(index) is not int or index < 0 or index >= len(record.labels):
        raise DataAlignmentError("invalid manifest window_index")
    expected = _indices(record.subject_id, index)
    if any(row.get(key) != value for key, value in expected.items()):
        raise DataAlignmentError("manifest indices or timing were altered")
    counts = window_counts(record)
    if len(set(counts.values())) != 1:
        raise DataAlignmentError("source arrays no longer match manifest counts")
    return index


def extract_window(record: SubjectRecord, row: dict) -> SignalWindow:
    _validate_row(record, row)
    return SignalWindow(
        sample_id=row["sample_id"],
        subject_id=record.subject_id,
        start_s=row["window_start_s"],
        end_s=row["window_end_s"],
        ppg=record.bvp[row["bvp_start"] : row["bvp_end"]].T.astype(np.float32),
        acc=record.acc[row["acc_start"] : row["acc_end"]].T.astype(np.float32),
    )


def reference_rows(record: SubjectRecord, manifest: list[dict]) -> list[ReferenceRow]:
    references = []
    for row in manifest:
        index = _validate_row(record, row)
        references.append(
            ReferenceRow(
                sample_id=row["sample_id"],
                hr_bpm=float(record.labels[index]),
                activity_id=row["activity_id"],
                eval_eligible=row["eval_eligible"],
            )
        )
    return references
