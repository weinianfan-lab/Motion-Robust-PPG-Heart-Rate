"""Local model-only inference on normalized signal windows, without reference labels."""

import argparse
import json
from pathlib import Path

import numpy as np


def main():
    import onnxruntime as ort

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument(
        "--inputs", type=Path, help="NPZ containing preprocessed float32 ppg/acc; no labels"
    )
    parser.add_argument("--synthetic-smoke", action="store_true")
    args = parser.parse_args()
    if bool(args.inputs) == bool(args.synthetic_smoke):
        parser.error("select exactly one of --inputs or --synthetic-smoke")
    session = ort.InferenceSession(str(args.model), providers=["CPUExecutionProvider"])
    names = [x.name for x in session.get_inputs()]
    if args.synthetic_smoke:
        rng = np.random.default_rng(42)
        values = {
            "ppg": rng.normal(size=(1, 1, 512)).astype(np.float32),
            "acc": rng.normal(size=(1, 3, 256)).astype(np.float32),
        }
    else:
        with np.load(args.inputs, allow_pickle=False) as archive:
            values = {key: archive[key] for key in names}
    for name in names:
        v = values[name]
        expected = (1, 512) if name == "ppg" else (3, 256)
        if (
            v.ndim != 3
            or v.shape[1:] != expected
            or v.dtype != np.float32
            or not np.isfinite(v).all()
        ):
            raise ValueError(f"{name} must be finite float32 [B,{expected[0]},{expected[1]}]")
    prediction = session.run(None, {k: values[k] for k in names})[0]
    print(
        json.dumps(
            {
                "hr_bpm": prediction.tolist(),
                "input_kind": "synthetic random smoke"
                if args.synthetic_smoke
                else "normalized windows",
                "note": "model-only; synthetic output is not accuracy evidence",
            }
        )
    )


if __name__ == "__main__":
    main()
