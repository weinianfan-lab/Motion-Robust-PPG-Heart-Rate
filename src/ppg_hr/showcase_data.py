"""Versioned engineering mapping, separate from the preserved strict audit."""

import json
from pathlib import Path

import numpy as np
from scipy.signal import butter, sosfiltfilt, welch

from ppg_hr.baselines.nlms import nlms_cancel
from ppg_hr.data.adapter import SubjectRecord, file_sha256, load_subject
from ppg_hr.data.windows import build_manifest
from ppg_hr.errors import DataAlignmentError, DataBlockedError, DataValidationError
from ppg_hr.io import new_run_dir, write_csv, write_json

ENGINEERING_ADAPTER = "dalia_engineering_mapping_v1"


def engineering_manifest(record: SubjectRecord) -> list[dict]:
    """Accept declared engineering mapping without claiming official certification."""
    if len(record.bvp) / 64 != len(record.acc) / 32:
        raise DataAlignmentError("PPG and ACC durations disagree")
    rows = build_manifest(record, alignment_verified=False)
    timestamp_audit = record.metadata.get("timestamp_audit", {})
    if timestamp_audit.get("status") == "available":
        origin = timestamp_audit.get("cross_channel_time_origin", {})
        if origin.get("status") == "mismatch":
            raise DataAlignmentError("explicit time origins disagree")
        for channel in timestamp_audit.get("channels", {}).values():
            if any(
                channel.get(k, 0)
                for k in [
                    "nonfinite_count",
                    "duplicate_count",
                    "decreasing_count",
                    "gap_count",
                    "irregular_step_count",
                    "length_mismatch",
                ]
            ):
                raise DataAlignmentError("timestamp channel is inconsistent")
    for row in rows:
        row["reason"] = [x for x in row["reason"] if x != "alignment_unverified"]
        row["eval_eligible"] = row["input_valid"] and row["reference_valid"]
        row["alignment_basis"] = "engineering_mapping_v1"
        row["official_generator_verified"] = False
    return rows


def make_windows(ppg, acc):
    """Native-rate complete windows, no implicit truncation of either modality."""
    ppg = np.asarray(ppg)
    acc = np.asarray(acc)
    p = np.lib.stride_tricks.sliding_window_view(ppg, 512)[::128, None, :]
    a = np.lib.stride_tricks.sliding_window_view(acc, 256, axis=0)[::64]
    if len(p) != len(a):
        raise DataAlignmentError("PPG and ACC window counts disagree")
    return p.astype(np.float32), a.astype(np.float32)


def spectral_predictions(ppg, acc):
    """Batched A1/A2 on the existing physical Welch grid; nonusable rows stay NaN."""
    freq, psd = welch(ppg[:, 0], fs=64, nperseg=256, noverlap=128, nfft=4096, axis=-1)
    afreq, apsd = welch(acc, fs=32, nperseg=128, noverlap=64, nfft=2048, axis=-1)
    if not np.array_equal(freq, afreq):
        # ACC has a lower Nyquist, but the shared candidate grid is identical.
        assert np.array_equal(freq[: len(afreq)], afreq)
    band = (freq >= 0.5) & (freq <= 4)
    aband = (afreq >= 0.5) & (afreq <= 4)
    psd = psd[:, band]
    apsd = apsd[:, :, aband].sum(axis=1)
    normalized = psd / np.maximum(psd.sum(axis=1, keepdims=True), 1e-12)
    motion = apsd / np.maximum(apsd.sum(axis=1, keepdims=True), 1e-12)
    first = 60 * freq[band][np.argmax(psd, axis=1)]
    second = 60 * freq[band][np.argmax(normalized / (1 + motion + 1e-12), axis=1)]
    invalid = ~np.isfinite(psd).all(axis=1) | (psd.sum(axis=1) <= 1e-12)
    first[invalid] = np.nan
    second[invalid] = np.nan
    return first, second


def prepare_engineering(root: Path, cache_root: Path, config: dict, *, trust_local_pickle=False):
    """Only trusted, receipt-matching original records; preserve all official targets."""
    if not trust_local_pickle:
        raise DataBlockedError("explicit trust_local_pickle is required")
    root = Path(root).resolve()
    receipt_path = root / ".acquisition-status.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if receipt.get("status") != "ready" or receipt.get("crc_verified") is not True:
        raise DataValidationError("complete CRC-verified acquisition receipt required")
    entries = [x for x in receipt["files"] if x["path"].endswith(".pkl")]
    entries.sort(key=lambda x: int(Path(x["path"]).stem[1:]))
    if [Path(x["path"]).stem for x in entries] != [f"S{i}" for i in range(1, 16)]:
        raise DataValidationError("exactly S1-S15 required")
    run = new_run_dir(cache_root, "engineering")
    write_json(run / "status.json", {"status": "PREPARING", "adapter": ENGINEERING_ADAPTER})
    summary = {
        "adapter": ENGINEERING_ADAPTER,
        "mapping": "t0=0; [2k,2k+8); complete end<=duration",
        "official_generator_verified": False,
        "source_receipt_sha256": file_sha256(receipt_path),
        "labels": "original official HR, no replacement or clipping",
        "subjects": {},
        "strict_provenance": "Deferred; original strict audit unchanged",
        "preprocessing": "offline zero-phase Butterworth N=4 0.5-4 Hz; native PPG64/ACC32",
        "nlms": {k: config[k] for k in ("nlms_taps", "nlms_mu", "nlms_epsilon")},
    }
    all_rows = []
    try:
        for entry in entries:
            sid = Path(entry["path"]).stem
            source = (root / entry["path"]).resolve()
            if not source.is_relative_to(root):
                raise DataValidationError("receipt path escapes source directory")
            before = file_sha256(source)
            if before != entry["sha256"]:
                raise DataValidationError(f"{sid} source hash differs from receipt")
            record = load_subject(source, sid, True)
            if record.metadata["raw_sha256"] != before:
                raise DataValidationError(f"{sid} source changed during import")
            rows = engineering_manifest(record)
            if not all(x["eval_eligible"] for x in rows):
                write_csv(run / f"{sid}-ineligible.csv", rows)
                raise DataValidationError(f"{sid} contains invalid windows; evidence retained")
            # Filter each subject separately; never cross subject or data split boundaries.
            p = sosfiltfilt(
                butter(4, [0.5, 4], fs=64, btype="bandpass", output="sos"), record.bvp[:, 0]
            )
            a = sosfiltfilt(
                butter(4, [0.5, 4], fs=32, btype="bandpass", output="sos"), record.acc, axis=0
            )
            pw, aw = make_windows(p, a)
            residual = nlms_cancel(
                p,
                np.repeat(a, 2, axis=0),
                taps=config["nlms_taps"],
                mu=config["nlms_mu"],
                epsilon=config["nlms_epsilon"],
            )
            rw, _ = make_windows(residual, a)
            # Limit temporary FFT arrays; no data-dependent parameter selection.
            predictions = []
            for start in range(0, len(pw), 256):
                sl = slice(start, start + 256)
                w, aa = spectral_predictions(pw[sl], aw[sl])
                n, _ = spectral_predictions(rw[sl], aw[sl])
                predictions.append(np.column_stack([w, aa, n]))
            baseline = np.concatenate(predictions)
            if not np.isfinite(baseline).all():
                raise DataValidationError(f"{sid} nonfinite baseline output; no partial scoring")
            folder = run / sid
            folder.mkdir()
            arrays = {
                "ppg": pw,
                "acc": aw,
                "labels": record.labels.copy(),
                "activity": np.array([x["activity_id"] for x in rows]),
                "baselines": baseline,
            }
            for name, array in arrays.items():
                np.save(folder / f"{name}.npy", array, allow_pickle=False)
            identity = {
                "raw_sha256": before,
                "labels_numeric_sha256": __import__("hashlib")
                .sha256(record.labels.tobytes())
                .hexdigest(),
                "count": len(rows),
                "duration_s": len(record.bvp) / 64,
                "continuous_acc_component_sumsq": float(np.sum((a - a.mean(axis=0)) ** 2)),
                "continuous_acc_samples": len(a),
                "files": {name: file_sha256(folder / f"{name}.npy") for name in arrays},
            }
            write_json(folder / "identity.json", identity)
            summary["subjects"][sid] = identity
            all_rows.extend(rows)
            print(f"{sid}: {len(rows)} official-label windows prepared", flush=True)
            del record, p, a, pw, aw, residual, rw, arrays
        write_csv(run / "manifest.csv", all_rows)
        summary["manifest_sha256"] = file_sha256(run / "manifest.csv")
        summary["total_windows"] = len(all_rows)
        summary["status"] = "READY_ENGINEERING"
        write_json(run / "metadata.json", summary)
        write_json(
            run / "status.json", {"status": "READY_ENGINEERING", "total_windows": len(all_rows)}
        )
    except Exception as exc:
        write_json(run / "status.json", {"status": "FAILED", "error": str(exc)})
        raise
    return run


def load_engineering_cache(path: Path):
    """Verify every numeric artifact before experiment use; no pickle cache loading."""
    path = Path(path)
    if json.loads((path / "status.json").read_text())["status"] != "READY_ENGINEERING":
        raise DataValidationError("engineering cache is incomplete")
    metadata = json.loads((path / "metadata.json").read_text(encoding="utf-8"))
    if metadata["adapter"] != ENGINEERING_ADAPTER:
        raise DataValidationError("wrong engineering adapter")
    if file_sha256(path / "manifest.csv") != metadata["manifest_sha256"]:
        raise DataValidationError("manifest changed")
    data = {}
    for sid, identity in metadata["subjects"].items():
        data[sid] = {}
        for name, h in identity["files"].items():
            file = path / sid / f"{name}.npy"
            if file_sha256(file) != h:
                raise DataValidationError(f"{sid}/{name} cache changed")
            data[sid][name] = np.load(file, allow_pickle=False, mmap_mode="r")
    return metadata, data
