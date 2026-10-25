"""Frozen zero-origin diagnostics using existing official R peaks, never estimator input."""

import argparse
import json
import pickle
from collections import Counter
from hashlib import sha256
from pathlib import Path

import numpy as np

from ppg_hr.data.adapter import file_sha256
from ppg_hr.io import new_run_dir, write_csv, write_json


def summary(value):
    data = np.asarray(value)
    finite = data[np.isfinite(data)]
    return {
        "shape": list(data.shape),
        "dtype": str(data.dtype),
        "count": int(data.size),
        "nonfinite": int(data.size - finite.size),
        "minimum": float(finite.min()) if finite.size else None,
        "maximum": float(finite.max()) if finite.size else None,
    }


def counts(samples, rate):
    width, hop = 8 * rate, 2 * rate
    inclusive = max(0, (samples - width) // hop + 1)
    strict = max(0, (samples - width - 1) // hop + 1)
    return {
        "samples": samples,
        "fs_hz": rate,
        "duration_s": samples / rate,
        "complete_end_le_N": inclusive,
        "complete_end_lt_N": strict,
        "residual_after_last_complete": samples - (width + (inclusive - 1) * hop)
        if inclusive
        else samples,
        "samples_mod_hop": samples % hop,
    }


def error_summary(estimate, reference):
    valid = np.isfinite(estimate) & np.isfinite(reference)
    difference = np.abs(estimate[valid] - reference[valid])
    return {
        "n_labels": len(reference),
        "n_comparable": int(valid.sum()),
        "mean_abs_difference_bpm": float(difference.mean()) if difference.size else None,
        "max_abs_difference_bpm": float(difference.max()) if difference.size else None,
        "matches_atol_1e_6": int((difference <= 1e-6).sum()),
        "matches_atol_1e_3": int((difference <= 1e-3).sum()),
        "first_reference": float(reference[0]) if len(reference) else None,
        "first_diagnostic": float(estimate[0])
        if len(estimate) and np.isfinite(estimate[0])
        else None,
        "last_reference": float(reference[-1]) if len(reference) else None,
        "last_diagnostic": float(estimate[-1])
        if len(estimate) and np.isfinite(estimate[-1])
        else None,
    }


def inspect(path, subject_id, expected_hash):
    digest = file_sha256(path)
    if digest != expected_hash:
        raise RuntimeError("Extracted raw hash differs from CRC-verified receipt")
    with path.open("rb") as source:
        payload = pickle.load(source, encoding="latin1")
    signals = payload["signal"]
    labels = np.asarray(payload["label"], dtype=float).reshape(-1)
    raw_peaks = np.asarray(payload["rpeaks"])
    peaks = raw_peaks.astype(float).reshape(-1)
    ecg_n = len(signals["chest"]["ECG"])
    ecg_duration = ecg_n / 700
    fractional = int(np.count_nonzero(peaks != np.floor(peaks)))
    valid_peaks = bool(
        len(peaks) > 1
        and np.isfinite(peaks).all()
        and (np.diff(peaks) > 0).all()
        and peaks.min() >= 0
    )
    if valid_peaks and fractional and peaks.max() <= ecg_duration:
        unit = "seconds_from_fractional_values_and_record_duration"
        times = peaks
    elif valid_peaks and not fractional and peaks.max() > ecg_duration and peaks.max() < ecg_n:
        unit = "ecg_sample_indices_from_integer_values_and_record_extent"
        times = peaks / 700
    else:
        unit = "unresolved"
        times = None
    # Count repeated positions without sorting, deduplicating, or changing R peaks.
    repetitions = Counter(peaks[np.isfinite(peaks)].tolist())
    duplicate_count = sum(count - 1 for count in repetitions.values())
    decreasing_count = int(np.count_nonzero(np.diff(peaks) < 0))
    failure_reasons = []
    if len(peaks) <= 1:
        failure_reasons.append("insufficient_rpeaks")
    if not np.isfinite(peaks).all():
        failure_reasons.append("nonfinite_rpeaks")
    if np.any(peaks < 0):
        failure_reasons.append("negative_rpeak_positions")
    if duplicate_count:
        failure_reasons.append("duplicate_rpeaks")
    if decreasing_count:
        failure_reasons.append("decreasing_rpeaks")
    if not valid_peaks:
        unit_status = "blocked_by_invalid_rpeaks"
    elif times is None:
        unit_status = "unresolved"
        failure_reasons.append("rpeak_unit_unresolved")
    else:
        unit_status = "resolved"
    modalities = {
        "bvp": counts(len(signals["wrist"]["BVP"]), 64),
        "acc": counts(len(signals["wrist"]["ACC"]), 32),
        "ecg": counts(ecg_n, 700),
        "activity": counts(len(payload["activity"]), 4),
    }
    details = {
        "subject_id": subject_id,
        "raw_sha256": digest,
        "keys": sorted(payload),
        "subject_field": str(payload.get("subject")),
        "modalities": modalities,
        "labels": summary(labels),
        "label_count": len(labels),
        "rpeaks": {
            **summary(raw_peaks),
            "fractional_count": fractional,
            "duplicate_count": duplicate_count,
            "duplicate_count_basis": "repeated finite positions anywhere in the original sequence",
            "decreasing_count": decreasing_count,
            "strictly_increasing": bool((np.diff(peaks) > 0).all()),
            "first": float(peaks[0]) if len(peaks) and np.isfinite(peaks[0]) else None,
            "last": float(peaks[-1]) if len(peaks) and np.isfinite(peaks[-1]) else None,
            "inferred_unit": unit,
            "unit_inference_status": unit_status,
            "validation_reasons": failure_reasons,
        },
        "proposed_label_window_origin_s": 0,
        "window_s": 8,
        "hop_s": 2,
        "official_rule_confirmed": False,
        "diagnostic_only": True,
    }
    rows = []
    if times is not None:
        hr = 60 / np.diff(times)
        start = 2 * np.arange(len(labels), dtype=float)
        end = start + 8
        cumulative = np.r_[0, np.cumsum(hr)]
        methods = {}
        # Primary: only intervals between two R peaks within [start,end).
        lo = np.searchsorted(times, start, side="left")
        hi = np.searchsorted(times, end, side="left") - 1
        n = hi - lo
        value = np.full(len(labels), np.nan)
        valid = n > 0
        value[valid] = (cumulative[hi[valid]] - cumulative[lo[valid]]) / n[valid]
        methods["within_window_rr"] = value
        for method, anchor in (
            ("right_endpoint", times[1:]),
            ("left_endpoint", times[:-1]),
            ("midpoint", (times[1:] + times[:-1]) / 2),
        ):
            lo = np.searchsorted(anchor, start, side="left")
            hi = np.searchsorted(anchor, end, side="left")
            n = hi - lo
            value = np.full(len(labels), np.nan)
            valid = n > 0
            value[valid] = (cumulative[hi[valid]] - cumulative[lo[valid]]) / n[valid]
            methods[method] = value
    else:
        methods = {
            name: np.full(len(labels), np.nan)
            for name in ("within_window_rr", "right_endpoint", "left_endpoint", "midpoint")
        }
    details["reference_diagnostics"] = {
        name: error_summary(value, labels) for name, value in methods.items()
    }
    for index, reference in enumerate(labels):
        rows.append(
            {
                "subject_id": subject_id,
                "label_index": index,
                "window_start_s": 2 * index,
                "window_end_s": 2 * index + 8,
                "official_reference_bpm": float(reference),
                **{
                    name: float(value[index]) if np.isfinite(value[index]) else None
                    for name, value in methods.items()
                },
                "reason": ";".join(failure_reasons),
            }
        )
    return details, rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--extracted-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=Path("runs"))
    parser.add_argument("--trust-local-pickle", action="store_true", required=True)
    args = parser.parse_args()
    extracted_root = args.extracted_root.resolve()
    receipt_path = extracted_root / ".acquisition-status.json"
    receipt_bytes = receipt_path.read_bytes()
    receipt = json.loads(receipt_bytes)
    if receipt["status"] != "ready" or receipt["crc_verified"] is not True:
        raise RuntimeError("Only completed official extraction is permitted")
    pickle_entries = [item for item in receipt["files"] if item["path"].endswith(".pkl")]
    entries = {Path(item["path"]).stem: item for item in pickle_entries}
    if len(pickle_entries) != 15 or set(entries) != {f"S{i}" for i in range(1, 16)}:
        raise RuntimeError("Official extraction must contain exactly S1..S15 pickles once each")
    for item in pickle_entries:
        if not (extracted_root / item["path"]).resolve().is_relative_to(extracted_root):
            raise RuntimeError("Receipt file path escapes the extracted root")
    script_path = Path(__file__).resolve()
    protocol_path = script_path.parents[1] / "docs" / "m1_reference_audit_protocol.md"
    provenance = {
        "script": {
            "path": str(script_path),
            "bytes": script_path.stat().st_size,
            "sha256": file_sha256(script_path),
        },
        "protocol": {
            "path": str(protocol_path),
            "bytes": protocol_path.stat().st_size,
            "sha256": file_sha256(protocol_path),
        },
        "source_receipt": {
            "path": str(receipt_path),
            "bytes": len(receipt_bytes),
            "sha256": sha256(receipt_bytes).hexdigest(),
            "status": receipt["status"],
            "crc_verified": receipt["crc_verified"],
            "archive_sha256": receipt.get("sha256"),
        },
    }
    run = new_run_dir(args.output_root, "reference-audit")
    report = {
        "status": "RUNNING",
        "diagnostic_only": True,
        "origin_s": 0,
        "subjects": [],
        "label_total": 0,
        "provenance": provenance,
    }
    write_json(run / "status.json", report)
    rows = []
    subject_row_counts = {}
    for number in range(1, 16):
        sid = f"S{number}"
        item = entries[sid]
        details, diagnostics = inspect(extracted_root / item["path"], sid, item["sha256"])
        report["subjects"].append(details)
        report["label_total"] += details["label_count"]
        rows.extend(diagnostics)
        subject_row_counts[sid] = len(diagnostics)
        write_json(run / "reference_audit.json", report)
        print(
            json.dumps(
                {
                    "subject": sid,
                    "labels": details["label_count"],
                    "counts": {k: v["complete_end_le_N"] for k, v in details["modalities"].items()},
                    "rpeak_unit": details["rpeaks"]["inferred_unit"],
                }
            ),
            flush=True,
        )
    if len(rows) != report["label_total"]:
        raise RuntimeError("Diagnostic rows must retain every official label")
    csv_path = run / "reference_diagnostics.csv"
    if rows:
        write_csv(csv_path, rows)
    report["diagnostic_csv"] = {
        "path": csv_path.name if rows else None,
        "count": len(rows),
        "sha256": file_sha256(csv_path) if rows else None,
        "subject_row_counts": subject_row_counts,
        "method_comparable_counts": {
            name: sum(
                subject.get("reference_diagnostics", {}).get(name, {}).get("n_comparable", 0)
                for subject in report["subjects"]
            )
            for name in ("within_window_rr", "right_endpoint", "left_endpoint", "midpoint")
        },
    }
    report.update(status="DIAGNOSTIC_COMPLETE_ALIGNMENT_NOT_CERTIFIED", paper_label_total=64697)
    write_json(run / "reference_audit.json", report)
    write_json(
        run / "status.json", {"status": report["status"], "label_total": report["label_total"]}
    )
    print(
        json.dumps(
            {"report": str(run / "reference_audit.json"), "label_total": report["label_total"]}
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
