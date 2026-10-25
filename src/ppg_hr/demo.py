"""A complete synthetic-only evaluation and figure loop, without model weights."""

import hashlib
import os
from dataclasses import asdict, fields, replace
from pathlib import Path
from time import perf_counter

import numpy as np
import yaml
from scipy.signal import butter

from ppg_hr.config import DemoConfig, load_mapping
from ppg_hr.doctor import provenance
from ppg_hr.errors import ConfigurationError
from ppg_hr.io import atomic_text, new_run_dir, write_csv, write_json


def _figure(run, scenario, record, windows, references, predictions, estimators):
    os.environ.setdefault("MPLCONFIGDIR", str(run / "matplotlib-cache"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(3, 1, figsize=(11, 10), layout="constrained")
    time = np.arange(record.ppg.shape[1]) / 64.0
    axes[0].plot(time, record.ppg[0], color="#217a8c", linewidth=0.8, label="Raw synthetic PPG")
    axes[0].set(
        title=f"SYNTHETIC DATA | {scenario} | signal", xlabel="Time (s)", ylabel="Arbitrary units"
    )
    axes[0].legend(loc="upper right")
    usable = next((window for window in windows if np.isfinite(window.ppg).all()), None)
    if usable is not None:
        for estimator in estimators:
            spectrum = estimator.spectrum(usable)
            score = spectrum["score"]
            axes[1].plot(
                spectrum["frequencies_hz"] * 60,
                score / max(float(score.max()), 1e-12),
                label=f"{estimator.model_id} normalized spectral score",
            )
        axes[1].legend()
    axes[1].set(
        title="SYNTHETIC DATA | first finite-window spectrum",
        xlabel="Frequency (BPM)",
        ylabel="Normalized score",
    )
    centers = [(window.start_s + window.end_s) / 2 for window in windows]
    axes[2].plot(
        centers,
        [row.hr_bpm for row in references],
        color="#222222",
        label="Window-average synthetic reference",
    )
    for model_id, rows in predictions.items():
        axes[2].plot(
            centers,
            [row.hr_bpm if row.hr_bpm is not None else np.nan for row in rows],
            marker=".",
            label=model_id,
        )
    axes[2].set(
        title="SYNTHETIC DATA | HR estimates and failures",
        xlabel="Window center (s), offline display",
        ylabel="BPM",
    )
    axes[2].legend()
    for axis in axes:
        axis.grid(alpha=0.2)
    figure.suptitle(
        "SYNTHETIC DATA — implementation check; no real-population accuracy claim", fontsize=12
    )
    figure.savefig(run / "figures" / f"{scenario}.png", dpi=130)
    plt.close(figure)


def run_demo(config: DemoConfig, output_root: Path, spectral_path=None) -> tuple[Path, dict]:
    from ppg_hr.baselines import ACCSpectralEstimator, SpectralConfig, WelchEstimator
    from ppg_hr.evaluation.metrics import evaluate_predictions
    from ppg_hr.preprocessing.normalization import normalize_ppg
    from ppg_hr.synthetic import SCENARIOS, generate_record, offline_bandpass_record, window_record

    if config.scenario != "all" and config.scenario not in SCENARIOS:
        raise ConfigurationError(f"Unknown scenario {config.scenario!r}; choices: {SCENARIOS}")
    options = load_mapping(spectral_path, {item.name for item in fields(SpectralConfig)})
    try:
        spectral = SpectralConfig(**options)
        estimators = [
            WelchEstimator(config=spectral),
            ACCSpectralEstimator(config=spectral, motion_weight=config.motion_weight),
        ]
    except (TypeError, ValueError) as exc:
        raise ConfigurationError(f"Invalid spectral configuration: {exc}") from exc
    resolved = {
        "demo": asdict(config),
        "models": [estimator.configuration for estimator in estimators],
        "pipeline_id": "offline_v1",
        "data_kind": "SYNTHETIC DATA",
        "preprocessing": {
            "ppg": "continuous_finite_segment_sosfiltfilt_then_window_center_and_std",
            "butter_design_N": 4,
            "band_hz": [0.5, 4.0],
            "fs_hz": 64,
            "padlen": 27,
            "sos": butter(4, [0.5, 4.0], btype="bandpass", fs=64, output="sos").tolist(),
            "ppg_normalization_epsilon": 1e-8,
            "acc": "native_32hz_center_each_axis; A2 band PSD normalized by its maximum",
            "acc_train_scale": "not needed for scale-invariant A2; no learned model used",
        },
    }
    run = new_run_dir(output_root, "demo")
    (run / "figures").mkdir()
    write_json(run / "run_status.json", {"status": "RUNNING", "data_kind": "SYNTHETIC DATA"})
    atomic_text(run / "resolved_config.yaml", yaml.safe_dump(resolved, sort_keys=True))
    source = provenance(resolved)
    source.update(
        {
            "data_kind": "SYNTHETIC DATA",
            "seed": config.seed,
            "split": "no training; scenario-specific independent generated sequences",
        }
    )
    write_json(run / "provenance.json", source)
    result = {
        "data_kind": "SYNTHETIC DATA",
        "real_data_evaluated": False,
        "pipeline_id": "offline_v1",
        "scenarios": {},
        "status_definition": (
            "PASS means prediction-table integrity and complete eligible-window output; "
            "it does not mean small HR error or clinical accuracy"
        ),
        "limitations": [
            "Zero-phase filtering uses future samples; these results are offline only",
            "Synthetic references are arithmetic means of programmed HR over each half-open window",
            "quality_score is spectral concentration, not calibrated correctness probability",
            "available_time_s is record end plus measured offline processing elapsed time; "
            "this is a simulated availability timestamp, not live causal validation",
            "No trained weights, real participants, causal validation or CPU benchmark involved",
        ],
    }
    all_rows, all_manifest = [], []
    try:
        for scenario in SCENARIOS if config.scenario == "all" else (config.scenario,):
            seed = int(
                np.random.SeedSequence([config.seed, SCENARIOS.index(scenario)]).generate_state(1)[
                    0
                ]
            )
            record = generate_record(
                scenario,
                duration_s=config.duration_s,
                seed=seed,
                hr_bpm=config.hr_bpm,
                waveform=config.waveform,
            )
            processing_start = perf_counter()
            filtered = offline_bandpass_record(record.ppg)
            windows, references, manifest = window_record(record, ppg=filtered)
            normalized = []
            for window in windows:
                if np.isfinite(window.ppg).all():
                    values, _flat = normalize_ppg(window.ppg)
                    normalized.append(replace(window, ppg=values))
                else:
                    normalized.append(window)
            predictions, reports = {}, {}
            for estimator in estimators:
                rows = []
                for window in normalized:
                    prediction = estimator.predict(window)
                    rows.append(
                        replace(
                            prediction,
                            available_time_s=config.duration_s + perf_counter() - processing_start,
                        )
                    )
                predictions[estimator.model_id] = rows
                reports[estimator.model_id] = evaluate_predictions(manifest, references, rows)
                for window, reference, pred, item in zip(
                    windows, references, rows, manifest, strict=True
                ):
                    all_rows.append(
                        {
                            "sample_id": pred.sample_id,
                            "subject_id": window.subject_id,
                            "scenario": scenario,
                            "fold": "synthetic_no_training",
                            "seed": seed,
                            "model_id": pred.model_id,
                            "pipeline_id": pred.pipeline_id,
                            "window_start_s": window.start_s,
                            "window_end_s": window.end_s,
                            "reference_time_s": (window.start_s + window.end_s) / 2,
                            "available_time_s": pred.available_time_s,
                            "reference_hr_bpm": reference.hr_bpm,
                            "raw_prediction_bpm": pred.hr_bpm,
                            "tracked_prediction_bpm": None,
                            "quality_score": pred.quality_score,
                            "status": pred.status,
                            "activity_id": reference.activity_id,
                            "eval_eligible": item["eval_eligible"],
                            "reason": item.get("reason", ""),
                        }
                    )
            for item in manifest:
                all_manifest.append(
                    {
                        **item,
                        "scenario": scenario,
                        "seed": seed,
                        "input_available_time_s": item["available_time_s"],
                        "available_time_s": config.duration_s,
                    }
                )
            digest = hashlib.sha256()
            for array in (record.ppg, record.acc, record.hr_bpm):
                digest.update(np.ascontiguousarray(array).tobytes())
            result["scenarios"][scenario] = {
                "generation": record.metadata,
                "generated_arrays_sha256": digest.hexdigest(),
                "models": reports,
            }
            _figure(run, scenario, record, normalized, references, predictions, estimators)
        write_csv(run / "predictions.csv", all_rows)
        write_csv(run / "manifest.csv", all_manifest)
        write_json(run / "results.json", result)
        blocked = any(
            report["status"] == "BLOCKED"
            for entry in result["scenarios"].values()
            for report in entry["models"].values()
        )
        status = "BLOCKED" if blocked else "COMPLETE"
        write_json(
            run / "run_status.json",
            {"status": status, "data_kind": "SYNTHETIC DATA", "real_data_evaluated": False},
        )
    except Exception as exc:
        write_json(
            run / "run_status.json", {"status": "FAILED", "error": f"{type(exc).__name__}: {exc}"}
        )
        raise
    return run, result
