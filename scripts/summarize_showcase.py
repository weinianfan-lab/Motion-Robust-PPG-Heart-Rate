"""Aggregate complete held-out folds and generate shareable summary charts only."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ppg_hr.data.adapter import file_sha256
from ppg_hr.io import new_run_dir, write_json
from ppg_hr.showcase_data import load_engineering_cache
from ppg_hr.showcase_experiment import (
    compatible_protocols,
    summarize_errors,
    validate_test_predictions,
)

METHODS = ["train_mean", "welch", "acc_spectral", "nlms_welch", "ppg_tcn", "fusion_tcn"]


def summarize(runs, output, cache, split_path):
    rows = []
    seen = set()
    evidence = []
    protocols = []
    _, data = load_engineering_cache(cache)
    frozen_splits = json.loads(split_path.read_text(encoding="utf-8"))
    for run in runs:
        protocol = json.loads((run / "protocol.lock.json").read_text(encoding="utf-8"))
        protocols.append(protocol)
        if protocol["cache_metadata_sha256"] != file_sha256(cache / "metadata.json") or protocol[
            "split_sha256"
        ] != file_sha256(split_path):
            raise ValueError("supplied cache/splits differ from frozen experiment")
        for folder in sorted(run.glob("fold-*")):
            status = json.loads((folder / "status.json").read_text(encoding="utf-8"))
            if status["status"] != "COMPLETE":
                raise ValueError("cannot aggregate incomplete fold")
            info = json.loads((folder / "results.json").read_text(encoding="utf-8"))
            index = int(folder.name.split("-")[-1])
            if info["split"] != frozen_splits["folds"][index]:
                raise ValueError("fold roles differ from frozen split")
            subjects = set(info["split"]["test"])
            if seen & subjects:
                raise ValueError("duplicate held-out subject across runs")
            seen |= subjects
            with (folder / "predictions.csv").open(newline="", encoding="utf-8") as h:
                part = list(csv.DictReader(h))
            validate_test_predictions(part, info["split"], data)
            if len(part) != info["test_windows"]:
                raise ValueError("prediction denominator changed")
            if len({x["sample_id"] for x in part}) != len(part):
                raise ValueError("duplicate window IDs")
            rows.extend(part)
            evidence.append(
                {
                    "source": str(folder),
                    "prediction_sha256": file_sha256(folder / "predictions.csv"),
                    "test_subjects": list(subjects),
                    "n": len(part),
                }
            )
    compatible_protocols(protocols)
    ref = np.array([float(x["official_bpm"]) for x in rows])
    sid = np.array([x["subject_id"] for x in rows])
    activity = np.array([x["activity_id"] for x in rows])
    subjects = sorted(seen, key=lambda x: int(x[1:]))
    complete = seen == {f"S{i}" for i in range(1, 16)}
    if complete and len(rows) != 64697:
        raise ValueError("full cohort must retain 64697 windows")
    result = {
        "status": "COMPLETE_LOSO"
        if complete
        else "SINGLE_FOLD"
        if len(seen) == 1
        else "PARTIAL_LOSO",
        "n_subjects": len(seen),
        "n_windows": len(rows),
        "methods": {},
        "provenance": evidence,
        "config": protocols[0]["config"],
        "aggregation_validation": "all test IDs, frozen roles, original cache labels, "
        "data/split/lock/config identities verified",
        "source_hashes_by_run": [p["source_hashes"] for p in protocols],
        "limitations": [
            "single seed",
            "public development data",
            "no clinical validation",
            "offline processing",
            "fusion has greater parameter count",
        ],
    }
    errors = {}
    for name in METHODS:
        v = np.array([float(x[name]) for x in rows])
        per_subject = {s: summarize_errors(ref[sid == s], v[sid == s]) for s in subjects}
        per_activity = {
            str(s): summarize_errors(ref[activity == s], v[activity == s])
            for s in np.unique(activity)
        }
        macro = float(np.mean([m["mae"] for m in per_subject.values()]))
        result["methods"][name] = {
            "micro": summarize_errors(ref, v),
            "subject_macro_mae": macro,
            "subject_metrics": per_subject,
            "activity_metrics": per_activity,
        }
        errors[name] = v - ref
    out = new_run_dir(output, "summary")
    write_json(out / "results.json", result)
    label = (
        f"PPG-DaLiA | {result['status']} | {len(seen)} held-out subjects / {len(rows):,} windows"
    )
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(figsize=(10, 4.8), layout="constrained")
    ax.bar(
        METHODS,
        [result["methods"][m]["subject_macro_mae"] for m in METHODS],
        color=["#9AA5AE", "#456D8C", "#338A92", "#60A887", "#D6954E", "#9C6499"],
    )
    ax.set(ylabel="Subject-macro MAE (BPM)", title=label)
    ax.tick_params(axis="x", rotation=15)
    fig.savefig(out / "comparison.png", dpi=180)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(11, 4.8), layout="constrained")
    for name in METHODS:
        ax.plot(
            subjects,
            [result["methods"][name]["subject_metrics"][s]["mae"] for s in subjects],
            marker="o",
            label=name,
        )
    ax.set(ylabel="MAE (BPM)", xlabel="Held-out subject", title=label)
    ax.legend(ncol=3, fontsize=8)
    fig.savefig(out / "subject_mae.png", dpi=180)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(10, 4.8), layout="constrained")
    bins = np.arange(0, max(3.0, max(float(np.max(np.abs(e))) for e in errors.values())) + 3.01, 3)
    for name in ["welch", "nlms_welch", "ppg_tcn", "fusion_tcn"]:
        ax.hist(
            np.abs(errors[name]),
            bins=bins,
            histtype="step",
            density=True,
            label=name,
            linewidth=1.8,
        )
    ax.set(xlabel="Absolute error (BPM)", ylabel="Density", title=label)
    ax.legend()
    fig.savefig(out / "error_distribution.png", dpi=180)
    plt.close(fig)
    # Local-only per-window traces are intentionally not copied into the public figures.
    md = (
        f"# Results: {result['status']}\n\n{len(rows):,} held-out windows "
        f"across {len(seen)} subjects. Official labels unchanged.\n\n"
    )
    md += (
        "| Method | Subject-macro MAE | Window-micro MAE | RMSE | Coverage |\n"
        "|---|---:|---:|---:|---:|\n"
    )
    for name, m in result["methods"].items():
        md += (
            f"| {name} | {m['subject_macro_mae']:.3f} | {m['micro']['mae']:.3f} | "
            f"{m['micro']['rmse']:.3f} | 100% |\n"
        )
    md += (
        "\nFixed configuration, one seed, offline processing. Fusion has more parameters. "
        "No test-set tuning or claim of paper-level reproduction/clinical accuracy.\n"
    )
    (out / "results.md").write_text(md, encoding="utf-8")
    print(out)
    return out, result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, nargs="+", required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--splits", type=Path, default=Path("splits/loso_v1.json"))
    parser.add_argument("--output-root", type=Path, default=Path("runs"))
    a = parser.parse_args()
    summarize(a.runs, a.output_root, a.cache, a.splits)
