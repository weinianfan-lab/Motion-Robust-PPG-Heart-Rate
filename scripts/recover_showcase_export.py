"""Recover export only in a new run; preserve original training failure artifacts."""

import argparse
import json
import shutil
from pathlib import Path

import torch

from ppg_hr.data.adapter import file_sha256
from ppg_hr.io import new_run_dir, write_json
from ppg_hr.models.tcn import BPMModel
from ppg_hr.showcase_experiment import export_and_verify


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=Path("runs"))
    args = parser.parse_args()
    original = args.run
    protocol = json.loads((original / "protocol.lock.json").read_text(encoding="utf-8"))
    if protocol["folds"] != [0]:
        raise ValueError("this recovery is for first-fold export only")
    source = original / "fold-00"
    results = json.loads((source / "results.json").read_text(encoding="utf-8"))
    if results["status"] != "COMPLETE":
        raise ValueError("completed test predictions required")
    stats = json.loads((source / "normalization.json").read_text(encoding="utf-8"))
    run = new_run_dir(args.output_root, "showcase-export-recovery")
    write_json(run / "status.json", {"status": "EXPORT_RECOVERY", "source": str(original)})
    folder = run / "fold-00"
    shutil.copytree(source, folder)
    shutil.copyfile(original / "protocol.lock.json", run / "protocol.lock.json")
    receipt = {
        "source_run": str(original),
        "source_status": json.loads((original / "status.json").read_text()),
        "mode": "export_only; no fitting, training, checkpoint selection or new predictions",
        "prediction_sha256": file_sha256(source / "predictions.csv"),
        "export_helper_sha256": file_sha256(Path("src/ppg_hr/showcase_experiment.py")),
        "checkpoint_hashes": {},
    }
    exports = {}
    try:
        for kind in ["ppg_tcn", "fusion_tcn"]:
            ckpt = source / f"{kind}-best.pt"
            if file_sha256(ckpt) != results["training"][kind]["checkpoint_sha256"]:
                raise ValueError("checkpoint no longer matches training receipt")
            receipt["checkpoint_hashes"][kind] = file_sha256(ckpt)
            model = BPMModel(kind, protocol["config"], stats)
            model.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=True))
            exports[kind] = export_and_verify(model.eval(), kind, folder, protocol["config"], stats)
        if file_sha256(folder / "predictions.csv") != receipt["prediction_sha256"]:
            raise ValueError("copied predictions changed")
        write_json(folder / "onnx_results.json", exports)
        write_json(
            folder / "status.json",
            {
                "status": "COMPLETE",
                "test_windows": results["test_windows"],
                "training_reused_from": str(source),
            },
        )
        write_json(run / "recovery.json", receipt)
        write_json(
            run / "status.json",
            {"status": "COMPLETE", "completed_folds": [0], "recovery": "export_only"},
        )
    except Exception as exc:
        write_json(
            run / "status.json", {"status": "FAILED", "stage": "export_recovery", "error": str(exc)}
        )
        raise
    print(run)


if __name__ == "__main__":
    main()
