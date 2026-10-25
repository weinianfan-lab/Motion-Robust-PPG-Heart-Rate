"""Exploratory reference forensics; fixed hypotheses, no label/data/gate changes."""

import argparse
import csv
import hashlib
import json
import pickle
from pathlib import Path

import numpy as np

from ppg_hr.data.adapter import file_sha256
from ppg_hr.io import new_run_dir, write_csv, write_json

METHODS = ("left", "both", "neither", "right", "bounded_40_200")


def window_diagnostics(peaks, start, end):
    """Compare registered integer-index formulas without mutating peak positions."""
    peaks = np.asarray(peaks)
    if peaks.ndim != 1 or peaks.dtype.kind not in "iu" or np.any(peaks[1:] < peaks[:-1]):
        raise ValueError("Expected nondecreasing integer ECG indices; never sort or round")
    result = {}
    for name, lower, upper in (
        ("left", "left", "left"),
        ("both", "left", "right"),
        ("neither", "right", "left"),
        ("right", "right", "right"),
    ):
        lo, hi = (
            int(np.searchsorted(peaks, start, side=lower)),
            int(np.searchsorted(peaks, end, side=upper)),
        )
        rr = np.diff(peaks[lo:hi].astype(np.int64))
        result[name] = float(np.mean(42000.0 / rr)) if len(rr) and np.all(rr > 0) else None
        if name == "left":
            # Fixed BioSPPy-inspired hypothesis only: do not remove any source peaks.
            retained = (rr >= 210) & (rr <= 1050)
            result.update(
                bounded_40_200=float(np.mean(42000.0 / rr[retained])) if retained.any() else None,
                bounded_retained_count=int(retained.sum()),
                bounded_rejected_count=int((~retained).sum()),
                zero_rr_count=int((rr == 0).sum()),
                low_hr_count=int((rr > 1050).sum()),
                high_hr_count=int(((rr > 0) & (rr < 210)).sum()),
                peak_lo=lo,
                peak_hi=hi,
                peak_at_start=bool(np.any(peaks[lo:hi] == start)),
                peak_at_end=bool(np.any(peaks == end)),
            )
    return result


def matches(value, reference):
    return value is not None and abs(value - reference) <= 1e-6


def summarize(rows):
    result = {"rows": len(rows)}
    for method in METHODS:
        differences = [
            abs(row[method] - row["official_bpm"]) for row in rows if row[method] is not None
        ]
        result[method] = {
            "comparable": len(differences),
            "matches_1e_6": sum(d <= 1e-6 for d in differences),
            "uncomputed": len(rows) - len(differences),
            "max_difference_bpm": max(differences) if differences else None,
        }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--extracted-root", type=Path, required=True)
    parser.add_argument("--previous-reference", type=Path, required=True)
    parser.add_argument("--trust-local-pickle", action="store_true", required=True)
    parser.add_argument("--output-root", type=Path, default=Path("runs"))
    args = parser.parse_args()
    root = args.extracted_root.resolve()
    receipt_path = root / ".acquisition-status.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if receipt["status"] != "ready" or receipt["crc_verified"] is not True:
        raise RuntimeError("Completed CRC-verified extraction required")
    entries = [entry for entry in receipt["files"] if entry["path"].endswith(".pkl")]
    indexed = {Path(entry["path"]).stem: entry for entry in entries}
    if len(entries) != 15 or set(indexed) != {f"S{i}" for i in range(1, 16)}:
        raise RuntimeError("Exactly one official file per subject required")
    previous = json.loads(args.previous_reference.read_text(encoding="utf-8"))
    previous_csv = args.previous_reference.parent / previous["diagnostic_csv"]["path"]
    if file_sha256(previous_csv) != previous["diagnostic_csv"]["sha256"]:
        raise RuntimeError("Previous diagnostic CSV identity changed")
    with previous_csv.open(encoding="utf-8", newline="") as source:
        old_rows = list(csv.DictReader(source))
    old = {(r["subject_id"], int(r["label_index"])): r for r in old_rows}
    if len(old) != len(old_rows) or len(old) != previous["label_total"]:
        raise RuntimeError("Previous full denominator/unique IDs missing")
    old_subjects = {s["subject_id"]: s for s in previous["subjects"]}
    run = new_run_dir(args.output_root, "reference-forensics")
    protocol = Path(__file__).resolve().parents[1] / "docs/m1_forensics_protocol.md"
    report = {
        "status": "RUNNING",
        "diagnostic_only": True,
        "alignment_certified": False,
        "original_labels_changed": False,
        "origin_s": 0,
        "provenance": {
            str(path.resolve()): file_sha256(path)
            for path in (
                Path(__file__),
                protocol,
                receipt_path,
                args.previous_reference,
                previous_csv,
            )
        },
        "subjects": [],
    }
    write_json(run / "status.json", report)
    all_rows, cases, events, links, duplicates = [], [], [], [], []
    for number in range(1, 16):
        sid = f"S{number}"
        entry = indexed[sid]
        path = (root / entry["path"]).resolve()
        if not path.is_relative_to(root) or file_sha256(path) != entry["sha256"]:
            raise RuntimeError("Raw path or identity changed")
        if old_subjects[sid]["raw_sha256"] != entry["sha256"]:
            raise RuntimeError("Previous reference used different original bytes")
        with path.open("rb") as source:
            payload = pickle.load(source, encoding="latin1")
        if str(payload["subject"]) != sid:
            raise RuntimeError("Subject identity mismatch")
        peaks = np.asarray(payload["rpeaks"])
        labels = np.asarray(payload["label"]).reshape(-1)
        ecg = np.asarray(payload["signal"]["chest"]["ECG"]).reshape(-1)
        if peaks.dtype.kind not in "iu" or peaks.ndim != 1 or np.any(peaks[1:] < peaks[:-1]):
            raise RuntimeError("Unexpected peak order/type; cannot silently normalize")
        if np.any(peaks < 0) or np.any(peaks >= len(ecg)) or not np.isfinite(labels).all():
            raise RuntimeError("Out-of-bounds peaks or nonfinite original labels")
        if len(labels) != old_subjects[sid]["label_count"]:
            raise RuntimeError("Label count changed")
        raw_peaks_hash = hashlib.sha256(peaks.tobytes()).hexdigest()
        labels_hash = hashlib.sha256(labels.tobytes()).hexdigest()
        rr = np.diff(peaks.astype(np.int64))
        abnormal = np.flatnonzero((rr < 210) | (rr > 1050))
        for index in abnormal:
            interval = int(rr[index])
            kind = "zero_rr" if interval == 0 else "high_hr" if interval < 210 else "low_hr"
            event = {
                "subject_id": sid,
                "rr_index": int(index),
                "left_peak_sample": int(peaks[index]),
                "right_peak_sample": int(peaks[index + 1]),
                "rr_samples": interval,
                "instantaneous_bpm": 42000.0 / interval if interval else None,
                "kind": kind,
                "left_ecg_value": float(ecg[peaks[index]]),
                "right_ecg_value": float(ecg[peaks[index + 1]]),
                "original_neighbor_peaks": json.dumps(
                    peaks[max(0, index - 2) : index + 4].tolist()
                ),
            }
            events.append(event)
            if interval == 0:
                duplicates.append(event)
        subject_rows = []
        for k, target in enumerate(labels):
            original = old[(sid, k)]
            if float(original["official_reference_bpm"]) != float(target):
                raise RuntimeError("Original labels differ from prior diagnostic")
            start, end = 1400 * k, 1400 * k + 5600
            values = window_diagnostics(peaks, start, end)
            old_value = (
                float(original["within_window_rr"]) if original["within_window_rr"] else None
            )
            old_mismatch = old_value is not None and abs(old_value - target) > 1e-6
            row = {
                "subject_id": sid,
                "label_index": k,
                "start_ecg_sample": start,
                "end_ecg_sample": end,
                "official_bpm": float(target),
                "previous_primary_bpm": old_value,
                "previous_reason": original["reason"],
                "previous_mismatch": old_mismatch,
                **values,
                "arithmetic_difference_bpm": abs(values["left"] - old_value)
                if values["left"] is not None and old_value is not None
                else None,
                "bounded_matches_reference": matches(values["bounded_40_200"], target),
                "matching_closures": ";".join(
                    name for name in METHODS[:4] if matches(values[name], target)
                ),
            }
            subject_rows.append(row)
            for index in abnormal[
                (abnormal >= values["peak_lo"]) & (abnormal < values["peak_hi"] - 1)
            ]:
                links.append(
                    {
                        "subject_id": sid,
                        "label_index": k,
                        "rr_index": int(index),
                        "rr_samples": int(rr[index]),
                        "previous_mismatch": old_mismatch,
                    }
                )
            if old_mismatch:
                lo, hi = values["peak_lo"], values["peak_hi"]
                cases.append(
                    {
                        **row,
                        "original_window_peaks": json.dumps(peaks[lo:hi].tolist()),
                        "original_window_rr_samples": json.dumps(rr[lo : hi - 1].tolist()),
                    }
                )
        if (
            raw_peaks_hash != hashlib.sha256(peaks.tobytes()).hexdigest()
            or labels_hash != hashlib.sha256(labels.tobytes()).hexdigest()
        ):
            raise RuntimeError("Source arrays mutated")
        # Post-read raw identity, in addition to the pre-import identity check.
        if file_sha256(path) != entry["sha256"]:
            raise RuntimeError("Raw file changed during read-only forensics")
        all_rows.extend(subject_rows)
        result = {
            "subject_id": sid,
            "raw_sha256": entry["sha256"],
            "peak_array_sha256": raw_peaks_hash,
            "label_array_sha256": labels_hash,
            "abnormal_rr_events": len(abnormal),
            "zero_rr_events": int((rr == 0).sum()),
            **summarize(subject_rows),
        }
        report["subjects"].append(result)
        write_json(run / "forensics.json", report)
        print(json.dumps(result), flush=True)
        del payload, ecg
    if len(all_rows) != previous["label_total"] or len(cases) != 152:
        raise RuntimeError("Full cohort or the original mismatch inventory was lost")
    report.update(
        status="FORENSICS_COMPLETE_ALIGNMENT_BLOCKED",
        cohort=summarize(all_rows),
        original_152=summarize(cases),
        original_uncomputed=summarize(
            [row for row in all_rows if row["previous_primary_bpm"] is None]
        ),
        arithmetic_max_difference_bpm=max(
            row["arithmetic_difference_bpm"] or 0 for row in all_rows
        ),
        abnormal_rr_event_count=len(events),
        zero_rr_event_count=len(duplicates),
        abnormal_rr_window_links=len(links),
        bounded_rejected_window_count=sum(row["bounded_rejected_count"] > 0 for row in all_rows),
        unexplained_bounded_rows=sum(not row["bounded_matches_reference"] for row in all_rows),
        note=(
            "Exploratory numerical compatibility is not proof of the official generator "
            "or annotation history."
        ),
    )
    report["artifacts"] = {}
    for name, rows in (
        ("all_windows", all_rows),
        ("original_152", cases),
        ("rr_events", events),
        ("duplicate_events", duplicates),
        ("rr_window_links", links),
    ):
        path = run / f"{name}.csv"
        if rows:
            write_csv(path, rows)
        report["artifacts"][name] = {
            "path": path.name if rows else None,
            "rows": len(rows),
            "sha256": file_sha256(path) if rows else None,
        }
    write_json(run / "forensics.json", report)
    write_json(run / "status.json", {"status": report["status"]})
    print(
        json.dumps(
            {
                "report": str(run / "forensics.json"),
                "cohort": report["cohort"],
                "original_152": report["original_152"],
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
