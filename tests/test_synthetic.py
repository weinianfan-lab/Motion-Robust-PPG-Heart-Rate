"""T09: deterministic synthetic fixtures, alignment and honest failure cases."""

import numpy as np
import pytest

from ppg_hr.baselines import ACCSpectralEstimator, WelchEstimator
from ppg_hr.synthetic import (
    SCENARIOS,
    generate_record,
    offline_bandpass_record,
    window_record,
)


def test_generation_is_seeded_and_preserves_native_sample_counts():
    first = generate_record("constant_hr", duration_s=16, seed=42)
    repeated = generate_record("constant_hr", duration_s=16, seed=42)
    independent = generate_record("constant_hr", duration_s=16, seed=43)
    assert first.ppg.shape == (1, 1024)
    assert first.acc.shape == (3, 512)
    np.testing.assert_array_equal(first.ppg, repeated.ppg)
    assert not np.array_equal(first.ppg, independent.ppg)
    assert first.metadata["seed"] == 42
    assert first.metadata["data_kind"] == "SYNTHETIC DATA"
    assert first.metadata["motion_hz"] != first.metadata["hr_start_bpm"] / 60


def test_changing_hr_integrates_phase_and_reference_is_window_average():
    record = generate_record("changing_hr", duration_s=16, seed=42, hr_bpm=60.0)
    windows, references, manifest = window_record(record)
    # Linear rate grows from 60 by 3.75 BPM/s; samples 0..511 average at 3.9921875 s.
    assert references[0].hr_bpm == pytest.approx(74.970703125)
    assert references[0].hr_bpm != pytest.approx(record.hr_bpm[511])
    assert record.phase_rad[1] - record.phase_rad[0] == pytest.approx(2 * np.pi / 64)
    assert record.phase_rad[513] - record.phase_rad[512] == pytest.approx(2 * np.pi * 90 / 60 / 64)
    assert len(windows) == len(references) == len(manifest) == 5
    assert windows[1].start_s == 2.0
    assert windows[1].end_s == 10.0
    assert manifest[1]["ppg_start_index"] == 128
    assert manifest[1]["acc_start_index"] == 64
    assert manifest[1]["label_index"] == 1
    assert manifest[1]["reference_time_s"] == 6.0


@pytest.mark.parametrize("estimator_type", [WelchEstimator, ACCSpectralEstimator])
def test_reference_mutation_cannot_change_signal_windows_or_predictions(estimator_type):
    # A reference-dependent slice, signal transform or prediction would violate T06.
    record = generate_record("changing_hr", duration_s=16, seed=42, hr_bpm=60.0)
    windows_before, references_before, _ = window_record(record)
    estimator = estimator_type()
    predictions_before = [estimator.predict(window) for window in windows_before]
    original_ppg, original_acc = record.ppg.copy(), record.acc.copy()

    record.hr_bpm[:] += 37.0
    windows_after, references_after, _ = window_record(record)
    predictions_after = [estimator.predict(window) for window in windows_after]

    np.testing.assert_array_equal(record.ppg, original_ppg)
    np.testing.assert_array_equal(record.acc, original_acc)
    assert len(references_before) == len(references_after) == 5
    for before, after in zip(references_before, references_after, strict=True):
        assert after.hr_bpm - before.hr_bpm == pytest.approx(37.0)
    assert predictions_after == predictions_before
    for before, after in zip(windows_before, windows_after, strict=True):
        np.testing.assert_array_equal(after.ppg, before.ppg)
        np.testing.assert_array_equal(after.acc, before.acc)


def test_motion_is_independent_unless_deliberate_frequency_collision():
    slow = generate_record("constant_hr", hr_bpm=60.0, seed=3)
    fast = generate_record("constant_hr", hr_bpm=180.0, seed=3)
    np.testing.assert_array_equal(slow.acc, fast.acc)
    assert not np.array_equal(slow.ppg, fast.ppg)
    assert set(SCENARIOS) == {
        "constant_hr",
        "changing_hr",
        "near_frequency",
        "same_frequency",
        "optical_only",
        "short_loss",
    }


def test_all_scenarios_retain_every_window_and_record_generation_parameters():
    for scenario in SCENARIOS:
        record = generate_record(scenario, duration_s=24, seed=31)
        windows, references, manifest = window_record(record)
        assert len(windows) == len(references) == len(manifest) == 9
        assert record.metadata["scenario"] == scenario
        assert record.metadata["phase_method"] == "left_rectangle_integration"
        assert record.metadata["reference_definition"] == (
            "arithmetic_mean_samplewise_hr_over_half_open_window"
        )
        assert all(
            item["sample_id"] == row.sample_id
            for item, row in zip(manifest, references, strict=True)
        )


def test_same_frequency_challenge_retains_a_measured_failure():
    record = generate_record("same_frequency", duration_s=24, seed=42, hr_bpm=90.0)
    windows, references, manifest = window_record(record, ppg=offline_bandpass_record(record.ppg))
    errors = [
        abs(estimator.predict(window).hr_bpm - reference.hr_bpm)
        for estimator in (WelchEstimator(), ACCSpectralEstimator())
        for window, reference in zip(windows, references, strict=True)
    ]
    assert max(errors) > 30.0
    assert all(row["eval_eligible"] for row in manifest)
    assert record.metadata["intentional_hr_motion_collision"] is True


def test_offline_filter_preserves_loss_without_poisoning_entire_record():
    record = generate_record("short_loss", duration_s=32, seed=42)
    filtered = offline_bandpass_record(record.ppg)
    assert np.isfinite(filtered[:, :512]).all()
    assert np.isfinite(filtered[:, -512:]).all()
    assert np.isnan(filtered).any()
    windows, references, manifest = window_record(record, ppg=filtered)
    assert len(windows) == len(references) == len(manifest) == 13
    predictions = [WelchEstimator().predict(window) for window in windows]
    assert any(row.status == "nonfinite_input" for row in predictions)
    assert any(row.hr_bpm is not None for row in predictions)
    assert any(not row["input_valid"] for row in manifest)


def test_window_override_cannot_change_sample_alignment():
    record = generate_record("constant_hr", duration_s=16)
    with pytest.raises(ValueError, match="shape"):
        window_record(record, ppg=np.ones((1, 1023)))


@pytest.mark.parametrize("duration_s", [0, 7.0, 8.01, float("nan")])
def test_generator_rejects_impossible_native_window_duration(duration_s):
    with pytest.raises(ValueError):
        generate_record(duration_s=duration_s)
