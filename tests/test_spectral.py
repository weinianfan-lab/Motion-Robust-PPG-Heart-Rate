"""T07/T08: spectral units and transparent degenerate-input behavior."""

import numpy as np
import pytest

from ppg_hr.baselines import ACCSpectralEstimator, WelchEstimator
from ppg_hr.contracts import SignalWindow


def make_window(ppg, acc=None):
    return SignalWindow(
        sample_id="unit:0000",
        subject_id="unit",
        start_s=0.0,
        end_s=8.0,
        ppg=np.asarray(ppg, dtype=np.float32).reshape(1, -1),
        acc=np.zeros((3, 256), dtype=np.float32) if acc is None else acc,
    )


@pytest.mark.parametrize("estimator_type", [WelchEstimator, ACCSpectralEstimator])
@pytest.mark.parametrize("hr_bpm", [60.0, 90.0, 120.0, 180.0])
@pytest.mark.parametrize("waveform", ["sine", "pulse"])
def test_clean_sine_and_harmonic_pulse_find_fundamental(estimator_type, hr_bpm, waveform):
    # A broken frequency scale, ACC resampling, or harmonic selection fails this test.
    phase = 2 * np.pi * (hr_bpm / 60) * np.arange(512) / 64
    ppg = np.sin(phase)
    if waveform == "pulse":
        ppg += 0.35 * np.sin(2 * phase) + 0.12 * np.sin(3 * phase)
    prediction = estimator_type().predict(make_window(ppg))
    assert prediction.hr_bpm is not None
    assert abs(prediction.hr_bpm - hr_bpm) < 1.0
    assert prediction.available_time_s == 8.0
    assert 0.0 <= prediction.quality_score <= 1.0
    assert prediction.pipeline_id == "offline_v1"


@pytest.mark.parametrize("estimator_type", [WelchEstimator, ACCSpectralEstimator])
@pytest.mark.parametrize(
    ("ppg", "status"),
    [
        (np.zeros(512), "flat_signal"),
        (np.ones(512), "flat_signal"),
        (1e-14 * np.sin(np.arange(512)), "low_amplitude"),
        (np.full(512, np.nan), "nonfinite_input"),
        (np.full(512, np.inf), "nonfinite_input"),
    ],
)
def test_degenerate_signals_produce_explicit_finite_states(estimator_type, ppg, status):
    prediction = estimator_type().predict(make_window(ppg))
    assert prediction.status == status
    assert prediction.hr_bpm is None
    assert prediction.quality_score == 0.0


def test_nonfinite_acc_cannot_silently_pass_a2():
    acc = np.zeros((3, 256), dtype=np.float32)
    acc[1, 17] = np.nan
    prediction = ACCSpectralEstimator().predict(make_window(np.sin(np.arange(512)), acc))
    assert prediction.status == "nonfinite_input"
    assert prediction.hr_bpm is None


def test_native_acc_physical_frequency_grid_penalizes_motion_peak():
    ppg_t, acc_t = np.arange(512) / 64, np.arange(256) / 32
    ppg = np.sin(2 * np.pi * 1.5 * ppg_t) + 1.15 * np.sin(2 * np.pi * 2.5 * ppg_t)
    acc = np.vstack([np.sin(2 * np.pi * 2.5 * acc_t), np.zeros(256), np.zeros(256)])
    window = make_window(ppg, acc)
    assert abs(WelchEstimator().predict(window).hr_bpm - 150.0) < 1.0
    assert abs(ACCSpectralEstimator().predict(window).hr_bpm - 90.0) < 1.0
    spectrum = ACCSpectralEstimator().spectrum(window)
    motion_peak = spectrum["frequencies_hz"][np.argmax(spectrum["acc_psd"])]
    assert abs(motion_peak - 2.5) < 0.02


def test_boundary_weak_and_harmonic_peaks_are_flagged_without_deleting_predictions():
    t = np.arange(512) / 64
    edge = WelchEstimator().predict(make_window(np.sin(2 * np.pi * 4.0 * t)))
    weak = WelchEstimator().predict(make_window(np.random.default_rng(111).normal(size=512)))
    harmonic = WelchEstimator().predict(
        make_window(0.65 * np.sin(2 * np.pi * 1.0 * t) + np.sin(2 * np.pi * 2.0 * t))
    )
    assert "edge_peak" in edge.status
    assert "weak_peak" in weak.status
    assert "harmonic_ambiguity" in harmonic.status
    assert all(p.hr_bpm is not None for p in [edge, weak, harmonic])


def test_only_implemented_pipeline_can_be_claimed():
    with pytest.raises(ValueError, match="offline_v1"):
        WelchEstimator(pipeline_id="causal_v1")


def test_inference_signature_does_not_accept_references():
    with pytest.raises(TypeError):
        WelchEstimator().predict(make_window(np.sin(np.arange(512))), hr_bpm=90.0)


def test_explicit_configuration_changes_the_actual_spectral_grid():
    from ppg_hr.baselines import SpectralConfig

    estimator = WelchEstimator(config=SpectralConfig(nfft=1024))
    spectrum = estimator.spectrum(make_window(np.sin(2 * np.pi * np.arange(512) / 64)))
    assert spectrum["frequencies_hz"][1] - spectrum["frequencies_hz"][0] == 0.0625
    assert estimator.configuration["nfft"] == 1024
    assert estimator.config["nfft"] == 1024


def test_configuration_rejects_invalid_overlap_and_incompatible_native_grid():
    from ppg_hr.baselines import SpectralConfig

    with pytest.raises(ValueError):
        SpectralConfig(noverlap=256)
    with pytest.raises(ValueError):
        SpectralConfig(fs_hz=32)
    with pytest.raises(ValueError):
        SpectralConfig(nperseg=255)


@pytest.mark.parametrize(("f_min_hz", "f_max_hz"), [(20.0, 25.0), (15.0, 17.0)])
def test_a2_rejects_candidate_band_above_acc_nyquist_without_restricting_a1(
    f_min_hz,
    f_max_hz,
):
    from ppg_hr.baselines import SpectralConfig

    config = SpectralConfig(f_min_hz=f_min_hz, f_max_hz=f_max_hz)
    with pytest.raises(ValueError, match="ACC Nyquist"):
        ACCSpectralEstimator(config=config)
    midpoint_hz = (f_min_hz + f_max_hz) / 2
    window = make_window(np.sin(2 * np.pi * midpoint_hz * np.arange(512) / 64))
    assert abs(WelchEstimator(config=config).predict(window).hr_bpm - 60 * midpoint_hz) < 1.0


def test_a2_can_include_the_acc_nyquist_endpoint_without_extrapolation():
    from ppg_hr.baselines import SpectralConfig

    estimator = ACCSpectralEstimator(config=SpectralConfig(f_min_hz=15.0, f_max_hz=16.0))
    window = make_window(np.cos(2 * np.pi * 16 * np.arange(512) / 64))
    spectrum = estimator.spectrum(window)
    assert spectrum["frequencies_hz"][-1] == 16.0
