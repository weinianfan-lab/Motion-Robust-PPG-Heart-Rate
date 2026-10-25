"""Complete-table evaluation; failed dense runs never get partial scores."""

import math
from collections import Counter
from numbers import Real

import numpy as np

from ppg_hr.contracts import Prediction, ReferenceRow
from ppg_hr.errors import PipelineMismatchError, PredictionIntegrityError

_USABLE_FLAGS = {"ok", "edge_peak", "weak_peak", "harmonic_ambiguity"}


def _finite(value: object) -> bool:
    return isinstance(value, Real) and not isinstance(value, bool) and math.isfinite(value)


def _index(rows: list, name: str, *, dictionaries: bool = False) -> dict:
    indexed = {}
    for row in rows:
        sample_id = row.get("sample_id") if dictionaries else row.sample_id
        if not isinstance(sample_id, str) or not sample_id:
            raise PredictionIntegrityError(f"{name} requires a nonempty sample_id")
        if sample_id in indexed:
            raise PredictionIntegrityError(f"duplicate {name} sample_id: {sample_id}")
        indexed[sample_id] = row
    return indexed


def _check_ids(expected: set[str], actual: set[str], name: str) -> None:
    missing, unknown = expected - actual, actual - expected
    if missing or unknown:
        raise PredictionIntegrityError(
            f"{name} association mismatch: missing={sorted(missing)}, unknown={sorted(unknown)}"
        )


def _window_metrics(errors: list[float]) -> dict:
    if not errors:
        return {
            "mae_bpm": None,
            "rmse_bpm": None,
            "p50_absolute_error_bpm": None,
            "p90_absolute_error_bpm": None,
            "p95_absolute_error_bpm": None,
            "error_gt_10_bpm_fraction": None,
            "signed_bias_bpm": None,
        }
    array = np.asarray(errors, dtype=np.float64)
    absolute = np.abs(array)
    return {
        "mae_bpm": float(absolute.mean()),
        "rmse_bpm": float(np.sqrt(np.mean(array * array))),
        "p50_absolute_error_bpm": float(np.percentile(absolute, 50, method="linear")),
        "p90_absolute_error_bpm": float(np.percentile(absolute, 90, method="linear")),
        "p95_absolute_error_bpm": float(np.percentile(absolute, 95, method="linear")),
        "error_gt_10_bpm_fraction": float(np.mean(absolute > 10)),
        "signed_bias_bpm": float(array.mean()),
    }


def _group_metrics(eligible: list[dict], errors: dict[str, float], key: str, score: bool) -> dict:
    groups = {}
    for row in eligible:
        groups.setdefault(row[key], []).append(row["sample_id"])
    result = {}
    for group, ids in sorted(groups.items()):
        values = [errors[sample_id] for sample_id in ids if sample_id in errors]
        result[group] = {
            "eligible": len(ids),
            "valid": len(values),
            "coverage": len(values) / len(ids),
            **_window_metrics(values if score else []),
        }
    return result


def evaluate_predictions(
    manifest: list[dict],
    references: list[ReferenceRow],
    predictions: list[Prediction],
    allow_rejection: bool = False,
    *,
    base_predictions: list[Prediction] | None = None,
) -> dict:
    """Associate by exact ID and report full-denominator dense/gated results.

    All three tables must contain exactly one row per manifest ID, including
    ineligible rows. Reference eligibility/activity must agree with manifest.
    Missing/duplicate/unknown IDs or mixed identities raise explicit errors.
    Null/nonfinite/failed eligible predictions return BLOCKED without metrics.

    Only explicit ``rejected`` rows may reduce coverage in rejection mode,
    and that mode requires complete, valid ``base_predictions`` of the same
    model/pipeline. Unrejected estimates must equal their base estimates.
    Missing all predictions for a subject makes the gated subject-macro
    undefined, not an average over an easier subset of subjects.
    """
    manifests = _index(manifest, "manifest", dictionaries=True)
    refs = _index(references, "reference")
    preds = _index(predictions, "prediction")
    _check_ids(set(manifests), set(refs), "reference")
    _check_ids(set(manifests), set(preds), "prediction")

    for sample_id, row in manifests.items():
        for key in ("subject_id", "activity_id"):
            if not isinstance(row.get(key), str) or not row[key]:
                raise PredictionIntegrityError(f"manifest {sample_id} requires nonempty {key}")
        if not isinstance(row.get("eval_eligible"), bool):
            raise PredictionIntegrityError(f"manifest {sample_id} requires boolean eval_eligible")
        ref = refs[sample_id]
        if ref.eval_eligible != row["eval_eligible"] or ref.activity_id != row["activity_id"]:
            raise PredictionIntegrityError(f"reference disagrees with manifest: {sample_id}")
        if row["eval_eligible"] and not _finite(ref.hr_bpm):
            raise PredictionIntegrityError(f"eligible reference must be finite: {sample_id}")

    identities = {(pred.model_id, pred.pipeline_id) for pred in predictions}
    if len(identities) > 1:
        raise PipelineMismatchError("cannot aggregate mixed model_id or pipeline_id")
    if identities and any(not isinstance(item, str) or not item for item in next(iter(identities))):
        raise PipelineMismatchError("model_id and pipeline_id must be nonempty")
    model_id, pipeline_id = next(iter(identities)) if identities else (None, None)
    eligible = [row for row in manifest if row["eval_eligible"]]
    errors = {}
    rejected, failed = [], []
    for row in eligible:
        sample_id = row["sample_id"]
        pred = preds[sample_id]
        if pred.status == "rejected":
            rejected.append(sample_id)
            continue
        usable = (
            isinstance(pred.status, str)
            and bool(pred.status)
            and set(pred.status.split("|")) <= _USABLE_FLAGS
            and _finite(pred.hr_bpm)
            and _finite(pred.available_time_s)
            and (pred.quality_score is None or _finite(pred.quality_score))
        )
        if usable:
            error = float(pred.hr_bpm) - float(refs[sample_id].hr_bpm)
            usable = math.isfinite(error)
        if usable:
            errors[sample_id] = error
        else:
            failed.append(sample_id)

    blockers = []
    if not eligible:
        blockers.append("no eval_eligible windows")
    if failed:
        blockers.append("eligible prediction failures; partial metrics suppressed")
    if rejected and not allow_rejection:
        blockers.append("dense prediction mode does not permit rejection")

    base_results = None
    if allow_rejection:
        if base_predictions is None:
            blockers.append("quality-gated results require complete ungated base_predictions")
        else:
            base_results = evaluate_predictions(manifest, references, base_predictions)
            if (base_results["model_id"], base_results["pipeline_id"]) != (model_id, pipeline_id):
                raise PipelineMismatchError(
                    "quality-gated and base predictions must share identity"
                )
            if base_results["status"] != "PASS":
                blockers.append("ungated base predictions did not pass dense evaluation")
            else:
                base_by_id = {prediction.sample_id: prediction for prediction in base_predictions}
                if any(
                    preds[sample_id].hr_bpm != base_by_id[sample_id].hr_bpm for sample_id in errors
                ):
                    raise PredictionIntegrityError(
                        "unrejected predictions differ from ungated base"
                    )
    elif base_predictions is not None:
        raise PredictionIntegrityError("base_predictions is only valid in rejection mode")

    by_subject = _group_metrics(eligible, errors, "subject_id", not blockers)
    by_activity = _group_metrics(eligible, errors, "activity_id", not blockers)
    metrics = None
    if not blockers:
        window = _window_metrics(list(errors.values()))
        subject_mae = [value["mae_bpm"] for value in by_subject.values()]
        complete_subjects = all(value is not None for value in subject_mae)
        scored = {key: value["mae_bpm"] for key, value in by_subject.items() if value["valid"]}
        metrics = {
            "subject_macro_mae_bpm": float(np.mean(subject_mae)) if complete_subjects else None,
            "subject_mae_sd_bpm": float(np.std(subject_mae, ddof=1))
            if complete_subjects and len(subject_mae) > 1
            else None,
            "micro_mae_bpm": window["mae_bpm"],
            "window_rmse_bpm": window["rmse_bpm"],
            "window_p50_absolute_error_bpm": window["p50_absolute_error_bpm"],
            "window_p90_absolute_error_bpm": window["p90_absolute_error_bpm"],
            "window_p95_absolute_error_bpm": window["p95_absolute_error_bpm"],
            "error_gt_10_bpm_fraction": window["error_gt_10_bpm_fraction"],
            "signed_bias_bpm": window["signed_bias_bpm"],
            "worst_subject_id": max(scored, key=scored.get) if scored else None,
            "subjects_without_valid_prediction": [
                key for key, value in by_subject.items() if not value["valid"]
            ],
        }
    return {
        "status": "BLOCKED" if blockers else ("PASS_WITH_REJECTION" if rejected else "PASS"),
        "evaluation_mode": "quality_gated" if allow_rejection else "dense",
        "model_id": model_id,
        "pipeline_id": pipeline_id,
        "counts": {
            "manifest": len(manifest),
            "eligible": len(eligible),
            "ineligible": len(manifest) - len(eligible),
            "valid": len(errors),
            "rejected": len(rejected),
            "failed": len(failed),
        },
        "coverage": len(errors) / len(eligible) if eligible else None,
        "status_counts": dict(Counter(preds[row["sample_id"]].status for row in eligible)),
        "failed_sample_ids": failed,
        "rejected_sample_ids": rejected,
        "blockers": blockers,
        "metrics": metrics,
        "by_subject": by_subject,
        "by_activity": by_activity,
        "base_results": base_results,
        "aggregation": {
            "primary": "arithmetic mean of per-subject MAE across all eligible subjects",
            "subject_sd": "sample SD of subject MAE, ddof=1; null with fewer than two subjects",
            "window_metrics": "pooled valid eligible windows; percentiles use linear interpolation",
            "signed_error": "prediction minus reference in BPM",
            "coverage": "valid predictions / all eval_eligible windows",
            "limitation": "descriptive metrics; overlapping windows are not independent samples",
        },
    }
