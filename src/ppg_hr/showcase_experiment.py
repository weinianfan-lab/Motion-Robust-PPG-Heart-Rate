"""Small, fixed-config subject-isolated engineering experiments."""

import json
import time
from pathlib import Path

import numpy as np

from ppg_hr.data.adapter import file_sha256
from ppg_hr.data.splits import validate_subject_split
from ppg_hr.io import new_run_dir, write_csv, write_json
from ppg_hr.showcase_data import load_engineering_cache


def fit_training_statistics(data, metadata, split):
    try:
        validate_subject_split(split["train"], split["val"], split["test"])
    except Exception as exc:
        raise ValueError("subject split is not disjoint") from exc
    train = split["train"]
    target = np.concatenate([data[s]["labels"] for s in train]).astype(np.float64)
    count = sum(metadata["subjects"][s]["continuous_acc_samples"] for s in train)
    sq = sum(metadata["subjects"][s]["continuous_acc_component_sumsq"] for s in train)
    if not np.isfinite(target).all() or count <= 0:
        raise ValueError("invalid training statistics")
    return {
        "target_mean": float(target.mean()),
        "target_std": max(float(target.std()), 1e-6),
        "acc_scale": max(float(np.sqrt(sq / (3 * count))), 1e-8),
        "fit_subjects": list(train),
        "acc_unique_time_samples": count,
        "target_training_windows": len(target),
    }


def summarize_errors(reference, prediction):
    ref = np.asarray(reference, dtype=np.float64)
    pred = np.asarray(prediction, dtype=np.float64)
    if (
        ref.ndim != 1
        or pred.shape != ref.shape
        or not len(ref)
        or not np.isfinite(ref).all()
        or not np.isfinite(pred).all()
    ):
        raise ValueError("complete finite paired predictions required; no partial scoring")
    error = pred - ref
    return {
        "n": len(ref),
        "mae": float(np.abs(error).mean()),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "bias": float(error.mean()),
        "p90_abs_error": float(np.percentile(np.abs(error), 90)),
        "above_10_bpm_fraction": float(np.mean(np.abs(error) > 10)),
        "coverage": 1.0,
    }


def compatible_protocols(protocols):
    """Refuse combining different data, roles, dependencies or model settings."""
    if not protocols:
        raise ValueError("no protocols provided")
    for key in ("config", "split_sha256", "cache_metadata_sha256", "lock_sha256"):
        if any(p[key] != protocols[0][key] for p in protocols):
            raise ValueError(f"incompatible experiment identity: {key}")


def validate_test_predictions(rows, split, data):
    """Check every test ID and original reference against the verified numeric cache."""
    validate_subject_split(split["train"], split["val"], split["test"])
    expected = {
        (sid, i): float(value)
        for sid in split["test"]
        for i, value in enumerate(data[sid]["labels"])
    }
    seen = set()
    for row in rows:
        sid = row["subject_id"]
        index = int(row["label_index"])
        key = (sid, index)
        if key not in expected or key in seen:
            raise ValueError("prediction subject/index is unexpected or duplicated")
        if row["sample_id"] != f"{sid}:{index:06d}":
            raise ValueError("prediction sample_id disagrees with subject/index")
        if float(row["official_bpm"]) != expected[key]:
            raise ValueError("prediction reference differs from original official label")
        if row["activity_id"] != str(data[sid]["activity"][index]):
            raise ValueError("prediction activity differs from cache grouping label")
        seen.add(key)
    if seen != set(expected):
        raise ValueError("test prediction denominator incomplete")


def _inputs(data, subjects, stats):
    p = np.concatenate([data[s]["ppg"] for s in subjects]).astype(np.float32)
    a = np.concatenate([data[s]["acc"] for s in subjects]).astype(np.float32)
    p = (p - p.mean(axis=-1, keepdims=True)) / (p.std(axis=-1, keepdims=True) + 1e-8)
    a = (a - a.mean(axis=-1, keepdims=True)) / stats["acc_scale"]
    return np.ascontiguousarray(p), np.ascontiguousarray(a)


def _predict(model, p, a, batch, device):
    import torch

    values = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(p), batch):
            pp = torch.as_tensor(p[start : start + batch], device=device)
            aa = torch.as_tensor(a[start : start + batch], device=device)
            values.append(model(pp, aa).cpu().numpy())
    return np.concatenate(values).astype(np.float64)


def train_model(kind, data, split, stats, config, folder, device):
    import torch

    from ppg_hr.models.tcn import BPMModel

    torch.manual_seed(config["seed"])
    np.random.seed(config["seed"])
    if device == "cuda":
        torch.cuda.manual_seed_all(config["seed"])
    model = BPMModel(kind, config, stats).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"]
    )
    p, a = _inputs(data, split["train"], stats)
    vp, va = _inputs(data, split["val"], stats)
    y = np.concatenate([data[s]["labels"] for s in split["train"]]).astype(np.float32)
    vy = np.concatenate([data[s]["labels"] for s in split["val"]]).astype(np.float64)
    # Training tensors only; no test targets or features are touched here.
    tp = torch.from_numpy(p).to(device)
    ta = torch.from_numpy(a).to(device)
    ty = torch.from_numpy(y).to(device)
    best = float("inf")
    history = []
    start = time.perf_counter()
    best_epoch = None
    for epoch in range(config["epochs"]):
        model.train()
        total = 0.0
        order = torch.randperm(len(ty), device=device)
        for first in range(0, len(order), config["batch_size"]):
            ix = order[first : first + config["batch_size"]]
            optimizer.zero_grad(set_to_none=True)
            output = model(tp[ix], ta[ix])
            loss = torch.mean(((output - ty[ix]) / stats["target_std"]) ** 2)
            if not torch.isfinite(loss):
                raise ValueError("nonfinite loss; retain failed run")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total += loss.item() * len(ix)
        prediction = _predict(model, vp, va, config["batch_size"], device)
        value = summarize_errors(vy, prediction)["mae"]
        history.append(
            {
                "epoch": epoch + 1,
                "training_normalized_mse": total / len(ty),
                "validation_mae": value,
            }
        )
        if value < best:
            best = value
            best_epoch = epoch + 1
            torch.save(model.state_dict(), folder / f"{kind}-best.pt")
        write_json(folder / f"{kind}-history.json", history)
        print(
            f"{folder.name} {kind} epoch {epoch + 1}/{config['epochs']} val MAE={value:.3f}",
            flush=True,
        )
    model.load_state_dict(
        torch.load(folder / f"{kind}-best.pt", map_location=device, weights_only=True)
    )
    model.eval()
    return model, {
        "best_epoch": best_epoch,
        "validation_mae": best,
        "training_seconds": time.perf_counter() - start,
        "parameters": sum(x.numel() for x in model.parameters()),
        "checkpoint_sha256": file_sha256(folder / f"{kind}-best.pt"),
    }


def export_and_verify(model, kind, folder, config, stats):
    import onnx
    import onnxruntime as ort
    import torch

    from ppg_hr.models.tcn import ONNXFusion, ONNXSingle

    model = model.to("cpu").eval()
    wrapper = ONNXSingle(model) if kind == "ppg_tcn" else ONNXFusion(model)
    generator = torch.Generator().manual_seed(123)
    p = torch.randn(7, 1, 512, generator=generator)
    a = torch.randn(7, 3, 256, generator=generator)
    args = (p,) if kind == "ppg_tcn" else (p, a)
    names = ["ppg"] if kind == "ppg_tcn" else ["ppg", "acc"]
    path = folder / f"{kind}.onnx"
    torch.onnx.export(
        wrapper,
        args,
        str(path),
        input_names=names,
        output_names=["hr_bpm"],
        opset_version=18,
        verbose=False,
        dynamo=True,
        external_data=False,
        dynamic_shapes=tuple({0: torch.export.Dim("batch")} for _ in args),
    )
    graph = onnx.load(str(path))
    onnx.checker.check_model(graph)
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    max_error = 0.0
    for batch in [1, 7]:
        inp = {name: value[:batch].numpy() for name, value in zip(names, args, strict=True)}
        result = session.run(None, inp)[0]
        with torch.inference_mode():
            expected = wrapper(*(x[:batch] for x in args)).numpy()
        difference = float(np.max(np.abs(result - expected)))
        if difference > 1e-4:
            raise ValueError(f"ONNX parity failed: {difference}")
        max_error = max(max_error, difference)
    inp = {name: value[:1].numpy() for name, value in zip(names, args, strict=True)}
    for _ in range(10):
        session.run(None, inp)
    timing = []
    for _ in range(100):
        t = time.perf_counter()
        session.run(None, inp)
        timing.append((time.perf_counter() - t) * 1000)
    report = {
        "status": "PASS",
        "sha256": file_sha256(path),
        "max_absolute_error_bpm": max_error,
        "tested_batches": [1, 7],
        "provider": "CPUExecutionProvider",
        "model_only_latency_ms_median": float(np.median(timing)),
        "model_only_latency_ms_p95": float(np.percentile(timing, 95)),
        "warmup": 10,
        "repeats": 100,
        "config": config,
        "normalization": stats,
        "inputs": (
            "preprocessed native-rate windows, not raw signals; "
            "PPG [B,1,512], optional ACC [B,3,256]"
        ),
        "target_scale": "embedded in exported model",
        "exporter": "torch.export dynamo=True",
    }
    write_json(folder / f"{kind}-onnx.json", report)
    return report


def run_fold(data, metadata, split, config, folder, device, *, export=False):
    import torch

    stats = fit_training_statistics(data, metadata, split)
    write_json(folder / "split.json", split)
    write_json(folder / "normalization.json", stats)
    models = {}
    training = {}
    exports = {}
    for kind in ["ppg_tcn", "fusion_tcn"]:
        model, info = train_model(kind, data, split, stats, config, folder, device)
        models[kind] = model
        training[kind] = info
    # Test features/targets are consumed only after training and checkpoint selection.
    tp, ta = _inputs(data, split["test"], stats)
    reference = np.concatenate([data[s]["labels"] for s in split["test"]]).astype(np.float64)
    activity = np.concatenate([data[s]["activity"] for s in split["test"]])
    subjects = np.concatenate([np.repeat(s, len(data[s]["labels"])) for s in split["test"]])
    baseline = np.concatenate([data[s]["baselines"] for s in split["test"]])
    values = {
        "train_mean": np.full(len(reference), stats["target_mean"]),
        "welch": baseline[:, 0],
        "acc_spectral": baseline[:, 1],
        "nlms_welch": baseline[:, 2],
    }
    for kind, model in models.items():
        values[kind] = _predict(model, tp, ta, config["batch_size"], device)
    metrics = {name: summarize_errors(reference, v) for name, v in values.items()}
    groups = {
        name: {
            str(g): summarize_errors(reference[activity == g], v[activity == g])
            for g in np.unique(activity)
        }
        for name, v in values.items()
    }
    rows = []
    for index in range(len(reference)):
        rows.append(
            {
                "subject_id": str(subjects[index]),
                "label_index": index,
                "sample_id": f"{subjects[index]}:{index:06d}",
                "official_bpm": float(reference[index]),
                "activity_id": str(activity[index]),
                **{name: float(v[index]) for name, v in values.items()},
            }
        )
    write_csv(folder / "predictions.csv", rows)
    report = {
        "status": "COMPLETE",
        "split": split,
        "training": training,
        "metrics": metrics,
        "activity_metrics": groups,
        "test_windows": len(reference),
        "no_rejections": True,
        "label_source": "unchanged official HR",
    }
    write_json(folder / "results.json", report)
    if export:
        for kind, model in models.items():
            exports[kind] = export_and_verify(model, kind, folder, config, stats)
        write_json(folder / "onnx_results.json", exports)
    del models, tp, ta
    if device == "cuda":
        torch.cuda.empty_cache()
    return report


def run_experiment(
    cache: Path,
    config: dict,
    split_path: Path,
    output_root: Path,
    folds: list[int],
    *,
    device="cuda",
):
    import os

    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import torch

    metadata, data = load_engineering_cache(cache)
    splits = json.loads(split_path.read_text(encoding="utf-8"))
    torch.set_num_threads(config["torch_threads"])
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA unavailable; no silent hardware substitution")
    run = new_run_dir(output_root, "showcase")
    write_json(run / "status.json", {"status": "RUNNING", "requested_folds": folds})
    provenance = {
        "config": config,
        "split_sha256": file_sha256(split_path),
        "cache_metadata_sha256": file_sha256(cache / "metadata.json"),
        "folds": folds,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "device": device,
        "device_name": torch.cuda.get_device_name(0) if device == "cuda" else "CPU",
        "source_hashes": {str(p): file_sha256(p) for p in sorted(Path("src/ppg_hr").rglob("*.py"))},
        "lock_sha256": file_sha256(Path("uv.lock")),
        "training_loss": "normalized-target MSE",
        "deterministic_algorithms": True,
        "limitations": "public-data engineering benchmark; no untouched external cohort",
    }
    write_json(run / "protocol.lock.json", provenance)
    try:
        for i in folds:
            folder = run / f"fold-{i:02d}"
            folder.mkdir()
            write_json(folder / "status.json", {"status": "RUNNING"})
            report = run_fold(
                data, metadata, splits["folds"][i], config, folder, device, export=(i == 0)
            )
            write_json(
                folder / "status.json",
                {"status": "COMPLETE", "test_windows": report["test_windows"]},
            )
        write_json(run / "status.json", {"status": "COMPLETE", "completed_folds": folds})
    except Exception as exc:
        write_json(
            run / "status.json", {"status": "FAILED", "error": str(exc), "requested_folds": folds}
        )
        raise
    return run
