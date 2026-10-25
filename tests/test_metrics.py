import math
from dataclasses import replace

import pytest


def fixture():
    from ppg_hr.contracts import Prediction, ReferenceRow

    subjects = ["S1", "S2", "S2", "S2"]
    activities = ["run", "mixed", "run", "unknown"]
    errors = [-2, 0, 4, 12]
    manifest = [
        {
            "sample_id": f"w{i}",
            "subject_id": subject,
            "activity_id": activity,
            "eval_eligible": True,
        }
        for i, (subject, activity) in enumerate(zip(subjects, activities, strict=True))
    ]
    refs = [ReferenceRow(row["sample_id"], 100.0, row["activity_id"], True) for row in manifest]
    predictions = [
        Prediction(f"w{i}", 100.0 + err, None, "ok", 8.0, "A1", "offline_v1")
        for i, err in enumerate(errors)
    ]
    return manifest, refs, predictions


def test_unequal_subject_lengths_macro_micro_and_tail_metrics_are_hand_computed():
    from ppg_hr.evaluation.metrics import evaluate_predictions

    manifest, refs, preds = fixture()
    result = evaluate_predictions(manifest, list(reversed(refs)), list(reversed(preds)))
    assert result["status"] == "PASS"
    assert result["coverage"] == 1
    assert result["counts"] == {
        "manifest": 4,
        "eligible": 4,
        "ineligible": 0,
        "valid": 4,
        "rejected": 0,
        "failed": 0,
    }
    metrics = result["metrics"]
    assert metrics["subject_macro_mae_bpm"] == pytest.approx(11 / 3)
    assert metrics["subject_mae_sd_bpm"] == pytest.approx(5 * math.sqrt(2) / 3)
    assert metrics["micro_mae_bpm"] == pytest.approx(4.5)
    assert metrics["window_rmse_bpm"] == pytest.approx(math.sqrt(41))
    assert metrics["window_p50_absolute_error_bpm"] == 3
    assert metrics["window_p90_absolute_error_bpm"] == pytest.approx(9.6)
    assert metrics["window_p95_absolute_error_bpm"] == pytest.approx(10.8)
    assert metrics["error_gt_10_bpm_fraction"] == 0.25
    assert metrics["signed_bias_bpm"] == 3.5
    assert metrics["worst_subject_id"] == "S2"
    assert result["by_subject"]["S1"]["mae_bpm"] == 2
    assert result["by_subject"]["S2"]["mae_bpm"] == pytest.approx(16 / 3)
    assert result["by_activity"]["run"]["mae_bpm"] == 3
    assert set(result["by_activity"]) == {"run", "mixed", "unknown"}


@pytest.mark.parametrize("table", ["manifest", "references", "predictions"])
def test_duplicate_rows_in_any_association_table_block(table):
    from ppg_hr.errors import PredictionIntegrityError
    from ppg_hr.evaluation.metrics import evaluate_predictions

    tables = dict(zip(["manifest", "references", "predictions"], fixture(), strict=True))
    tables[table].append(tables[table][0])
    with pytest.raises(PredictionIntegrityError, match="duplicate"):
        evaluate_predictions(**tables)


@pytest.mark.parametrize("table", ["references", "predictions"])
def test_missing_or_unknown_associated_id_blocks(table):
    from ppg_hr.errors import PredictionIntegrityError
    from ppg_hr.evaluation.metrics import evaluate_predictions

    tables = dict(zip(["manifest", "references", "predictions"], fixture(), strict=True))
    tables[table].pop()
    with pytest.raises(PredictionIntegrityError, match="missing"):
        evaluate_predictions(**tables)
    tables[table].append(replace(tables[table][0], sample_id="unknown"))
    with pytest.raises(PredictionIntegrityError, match="unknown"):
        evaluate_predictions(**tables)


def test_reference_eligibility_and_activity_cannot_override_manifest():
    from ppg_hr.errors import PredictionIntegrityError
    from ppg_hr.evaluation.metrics import evaluate_predictions

    manifest, refs, preds = fixture()
    for change in ({"eval_eligible": False}, {"activity_id": "other"}):
        altered = [replace(refs[0], **change), *refs[1:]]
        with pytest.raises(PredictionIntegrityError, match="manifest"):
            evaluate_predictions(manifest, altered, preds)


@pytest.mark.parametrize("change", [{"pipeline_id": "causal_v1"}, {"model_id": "A2"}])
def test_mixed_prediction_identity_blocks(change):
    from ppg_hr.errors import PipelineMismatchError
    from ppg_hr.evaluation.metrics import evaluate_predictions

    manifest, refs, preds = fixture()
    with pytest.raises(PipelineMismatchError):
        evaluate_predictions(manifest, refs, [replace(preds[0], **change), *preds[1:]])


@pytest.mark.parametrize("value", [None, math.nan, math.inf])
def test_dense_bad_prediction_blocks_without_silent_row_deletion(value):
    from ppg_hr.evaluation.metrics import evaluate_predictions

    manifest, refs, preds = fixture()
    preds[0] = replace(preds[0], hr_bpm=value)
    result = evaluate_predictions(manifest, refs, preds)
    assert result["status"] == "BLOCKED"
    assert result["metrics"] is None
    assert result["counts"]["eligible"] == 4
    assert result["counts"]["failed"] == 1
    assert result["coverage"] == 0.75
    assert result["failed_sample_ids"] == ["w0"]


def test_rejection_requires_complete_dense_base_and_reports_full_denominators():
    from ppg_hr.evaluation.metrics import evaluate_predictions

    manifest, refs, base = fixture()
    gated = [replace(base[0], hr_bpm=None, status="rejected"), *base[1:]]
    result = evaluate_predictions(manifest, refs, gated, allow_rejection=True)
    assert result["status"] == "BLOCKED"
    assert "base" in " ".join(result["blockers"])
    result = evaluate_predictions(
        manifest, refs, gated, allow_rejection=True, base_predictions=base
    )
    assert result["status"] == "PASS_WITH_REJECTION"
    assert result["counts"]["eligible"] == 4
    assert result["counts"]["rejected"] == 1
    assert result["coverage"] == 0.75
    assert result["base_results"]["coverage"] == 1
    assert result["base_results"]["metrics"]["subject_macro_mae_bpm"] == pytest.approx(11 / 3)
    assert result["by_subject"]["S1"]["coverage"] == 0
    assert result["by_subject"]["S1"]["mae_bpm"] is None
    assert result["metrics"]["subject_macro_mae_bpm"] is None


def test_ineligible_windows_are_counted_but_do_not_inflate_coverage_or_errors():
    from ppg_hr.evaluation.metrics import evaluate_predictions

    manifest, refs, preds = fixture()
    manifest[0]["eval_eligible"] = False
    refs[0] = replace(refs[0], eval_eligible=False)
    preds[0] = replace(preds[0], hr_bpm=None, status="invalid_input")
    result = evaluate_predictions(manifest, refs, preds)
    assert result["status"] == "PASS"
    assert result["counts"]["ineligible"] == 1
    assert result["counts"]["eligible"] == 3
    assert result["coverage"] == 1
    assert result["metrics"]["subject_macro_mae_bpm"] == pytest.approx(16 / 3)
    assert result["metrics"]["subject_mae_sd_bpm"] is None


def test_empty_eligible_denominator_blocks_instead_of_emitting_nan():
    from ppg_hr.evaluation.metrics import evaluate_predictions

    result = evaluate_predictions([], [], [])
    assert result["status"] == "BLOCKED"
    assert result["coverage"] is None
    assert result["metrics"] is None


def test_finite_spectral_quality_flags_remain_in_dense_denominator():
    from ppg_hr.evaluation.metrics import evaluate_predictions

    manifest, refs, preds = fixture()
    preds[0] = replace(preds[0], status="edge_peak|weak_peak|harmonic_ambiguity")
    result = evaluate_predictions(manifest, refs, preds)
    assert result["status"] == "PASS"
    assert result["coverage"] == 1
    assert result["metrics"]["micro_mae_bpm"] == 4.5


def test_rejection_is_not_allowed_in_dense_mode_and_bad_base_cannot_rescue_it():
    from ppg_hr.evaluation.metrics import evaluate_predictions

    manifest, refs, base = fixture()
    gated = [replace(base[0], hr_bpm=None, status="rejected"), *base[1:]]
    assert evaluate_predictions(manifest, refs, gated)["status"] == "BLOCKED"
    result = evaluate_predictions(manifest, refs, gated, True, base_predictions=gated)
    assert result["status"] == "BLOCKED"
    assert result["base_results"]["status"] == "BLOCKED"


def test_gating_cannot_change_unrejected_base_estimates():
    from ppg_hr.errors import PredictionIntegrityError
    from ppg_hr.evaluation.metrics import evaluate_predictions

    manifest, refs, base = fixture()
    gated = [replace(base[0], hr_bpm=90), *base[1:]]
    with pytest.raises(PredictionIntegrityError, match="base"):
        evaluate_predictions(manifest, refs, gated, True, base_predictions=base)


def test_nonfinite_eligible_reference_is_an_integrity_failure():
    from ppg_hr.errors import PredictionIntegrityError
    from ppg_hr.evaluation.metrics import evaluate_predictions

    manifest, refs, preds = fixture()
    refs[0] = replace(refs[0], hr_bpm=math.nan)
    with pytest.raises(PredictionIntegrityError, match="reference"):
        evaluate_predictions(manifest, refs, preds)
