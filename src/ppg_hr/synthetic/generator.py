"""Phase-integrated synthetic wrist signals and explicit half-open window references."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ppg_hr.contracts import ReferenceRow, SignalWindow

SCENARIOS = (
    "constant_hr",
    "changing_hr",
    "near_frequency",
    "same_frequency",
    "optical_only",
    "short_loss",
)


@dataclass(frozen=True)
class SyntheticRecord:
    ppg: np.ndarray
    acc: np.ndarray
    hr_bpm: np.ndarray
    phase_rad: np.ndarray
    metadata: dict


def generate_record(
    scenario: str = "constant_hr",
    *,
    duration_s: float = 32.0,
    seed: int = 42,
    hr_bpm: float = 90.0,
    waveform: str = "pulse",
    motion_hz: float | None = None,
) -> SyntheticRecord:
    """Generate separate HR and motion sources, with two explicit collision cases.

    The changing-HR scenario increases by 60 BPM over the record. ``same_frequency``
    deliberately phase-cancels the cardiac fundamental and preserves its harmonics,
    exposing a spectral failure; it is not a prevalence or realism claim.
    """
    if scenario not in SCENARIOS:
        raise ValueError(f"Unknown synthetic scenario: {scenario}")
    if not np.isfinite(duration_s) or duration_s < 8 or not float(duration_s * 32).is_integer():
        raise ValueError("duration_s must be >=8 and align exactly to the native 32 Hz grid")
    if not np.isfinite(hr_bpm) or hr_bpm <= 0:
        raise ValueError("hr_bpm must be positive and finite")
    if waveform not in ("sine", "pulse"):
        raise ValueError("waveform must be sine or pulse")
    if not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if motion_hz is None:
        motion_hz = (
            hr_bpm / 60
            if scenario == "same_frequency"
            else (hr_bpm / 60 + 0.12 if scenario == "near_frequency" else 2.2)
        )
    if not np.isfinite(motion_hz) or motion_hz <= 0 or motion_hz >= 16:
        raise ValueError("motion_hz must be finite and between zero and ACC Nyquist")
    if scenario == "same_frequency" and motion_hz != hr_bpm / 60:
        raise ValueError("same_frequency requires motion_hz equal to HR / 60")
    rng = np.random.default_rng(seed)
    ppg_t = np.arange(round(duration_s * 64), dtype=np.float64) / 64
    acc_t = np.arange(round(duration_s * 32), dtype=np.float64) / 32
    hr = np.full_like(ppg_t, hr_bpm)
    if scenario == "changing_hr":
        hr += 60.0 * ppg_t / duration_s
    phase = np.concatenate(([0.0], np.cumsum(2 * np.pi * hr[:-1] / (60.0 * 64))))
    pulse = np.sin(phase)
    harmonics = [0.35, 0.12] if waveform == "pulse" else [0.0, 0.0]
    pulse += harmonics[0] * np.sin(2 * phase) + harmonics[1] * np.sin(3 * phase)
    motion_phase = np.pi if scenario == "same_frequency" else 0.7
    motion_ppg = np.sin(2 * np.pi * motion_hz * ppg_t + motion_phase)
    motion_acc = np.sin(2 * np.pi * motion_hz * acc_t + motion_phase)
    motion_amplitude = {
        "constant_hr": 0.25,
        "changing_hr": 0.3,
        "near_frequency": 1.1,
        "same_frequency": 1.0,
        "optical_only": 1.2,
        "short_loss": 0.3,
    }[scenario]
    # The optical-only fixture has an optical artifact but no corresponding ACC reference.
    optical_only = scenario == "optical_only"
    acc = np.vstack([motion_acc, 0.5 * motion_acc, -0.25 * motion_acc])
    if optical_only:
        acc[:] = 0.0
    acc += rng.normal(0.0, 0.015, size=acc.shape)
    amplitude_step_time = None if scenario == "same_frequency" else duration_s / 2
    envelope = np.ones_like(ppg_t)
    if amplitude_step_time is not None:
        envelope[ppg_t >= amplitude_step_time] = 0.7
    ppg = (
        envelope * pulse
        + motion_amplitude * motion_ppg
        + 0.1 * np.sin(2 * np.pi * 0.15 * ppg_t)
        + rng.normal(0.0, 0.025, size=ppg_t.shape)
    )
    loss_interval = None
    if scenario == "short_loss":
        loss_interval = [duration_s / 2 - 1, duration_s / 2 + 1]
        ppg[(ppg_t >= loss_interval[0]) & (ppg_t < loss_interval[1])] = np.nan
    metadata = {
        "data_kind": "SYNTHETIC DATA",
        "generator_version": "synthetic_v1",
        "scenario": scenario,
        "seed": seed,
        "rng": "numpy.default_rng.PCG64",
        "duration_s": float(duration_s),
        "ppg_fs_hz": 64.0,
        "acc_fs_hz": 32.0,
        "hr_start_bpm": float(hr_bpm),
        "hr_slope_bpm_per_s": 60.0 / duration_s if scenario == "changing_hr" else 0.0,
        "phase_method": "left_rectangle_integration",
        "initial_phase_rad": 0.0,
        "reference_definition": "arithmetic_mean_samplewise_hr_over_half_open_window",
        "waveform": waveform,
        "harmonic_amplitudes": harmonics,
        "motion_hz": float(motion_hz),
        "motion_phase_rad": float(motion_phase),
        "motion_optical_amplitude": motion_amplitude,
        "acc_motion_axis_scales": [0.0, 0.0, 0.0] if optical_only else [1.0, 0.5, -0.25],
        "ppg_noise_std": 0.025,
        "acc_noise_std": 0.015,
        "baseline_drift_hz": 0.15,
        "baseline_drift_amplitude": 0.1,
        "amplitude_step_time_s": amplitude_step_time,
        "post_step_pulse_scale": 0.7 if amplitude_step_time is not None else 1.0,
        "loss_interval_s": loss_interval,
        "intentional_hr_motion_collision": scenario in ("near_frequency", "same_frequency"),
        "collision_design": "fundamental_phase_cancellation_exposes_harmonic"
        if scenario == "same_frequency"
        else None,
        "sequence_id": f"{scenario}:seed{seed}",
        "split_usage": "demonstration_only_no_train_val_test_split",
        "evidence_limit": "algorithm_fixture_not_real_population_or_clinical_validation",
    }
    return SyntheticRecord(
        ppg[None, :].astype(np.float32), acc.astype(np.float32), hr, phase, metadata
    )


def window_record(
    record: SyntheticRecord,
    *,
    ppg: np.ndarray | None = None,
) -> tuple[list[SignalWindow], list[ReferenceRow], list[dict]]:
    """Keep every complete 8 s / 2 s window; isolate reference association here.

    A same-shape preprocessed PPG override is allowed. Nonfinite windows stay in
    the manifest, with input_valid/eval_eligible false; no window is silently lost.
    """
    source = record.ppg if ppg is None else np.asarray(ppg)
    if source.shape != record.ppg.shape:
        raise ValueError("preprocessed PPG shape must match original signal shape")
    n_ppg = max(0, (source.shape[1] - 512) // 128 + 1)
    n_acc = max(0, (record.acc.shape[1] - 256) // 64 + 1)
    if n_ppg != n_acc or source.shape[1] != len(record.hr_bpm):
        raise ValueError("Synthetic PPG, ACC and HR arrays have inconsistent alignment")
    subject_id = f"synthetic_{record.metadata['scenario']}_seed{record.metadata['seed']}"
    windows, references, manifest = [], [], []
    for index in range(n_ppg):
        ppg_start, acc_start, start_s = 128 * index, 64 * index, 2.0 * index
        sample_id = f"{subject_id}:{index:05d}"
        ppg_window = source[:, ppg_start : ppg_start + 512].astype(np.float32)
        acc_window = record.acc[:, acc_start : acc_start + 256].astype(np.float32)
        valid = bool(np.isfinite(ppg_window).all() and np.isfinite(acc_window).all())
        reference = float(np.mean(record.hr_bpm[ppg_start : ppg_start + 512]))
        windows.append(
            SignalWindow(sample_id, subject_id, start_s, start_s + 8, ppg_window, acc_window)
        )
        references.append(ReferenceRow(sample_id, reference, "synthetic", valid))
        manifest.append(
            {
                "sample_id": sample_id,
                "subject_id": subject_id,
                "label_index": index,
                "window_start_s": start_s,
                "window_end_s": start_s + 8,
                "reference_time_s": start_s + 4,
                "available_time_s": start_s + 8,
                "ppg_start_index": ppg_start,
                "ppg_end_index": ppg_start + 512,
                "acc_start_index": acc_start,
                "acc_end_index": acc_start + 256,
                "input_valid": valid,
                "reference_valid": True,
                "eval_eligible": valid,
                "reason": "" if valid else "nonfinite_input",
                "activity_id": "synthetic",
                "activity_purity": 1.0,
                "flat_signal_flag": bool(valid and np.ptp(ppg_window) == 0),
            }
        )
    return windows, references, manifest
