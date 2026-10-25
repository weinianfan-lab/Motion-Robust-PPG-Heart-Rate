# Optional extras must be checked before importing the model modules.
# ruff: noqa: E402
import numpy as np
import pytest

torch = pytest.importorskip("torch")
ort = pytest.importorskip("onnxruntime")
pytest.importorskip("onnxscript")

from ppg_hr.models.tcn import BPMModel
from ppg_hr.showcase_experiment import export_and_verify


@pytest.mark.parametrize("kind", ["ppg_tcn", "fusion_tcn"])
def test_export_supports_dynamic_batches_and_signal_only_inputs(kind, tmp_path):
    config = {"channels": [4, 8], "kernel_size": 3, "dilations": [1, 2], "dropout": 0.0}
    stats = {"target_mean": 90.0, "target_std": 20.0, "acc_scale": 1.0}
    model = BPMModel(kind, config, stats).eval()
    result = export_and_verify(model, kind, tmp_path, config, stats)
    assert result["max_absolute_error_bpm"] <= 1e-4
    session = ort.InferenceSession(
        str(tmp_path / f"{kind}.onnx"), providers=["CPUExecutionProvider"]
    )
    assert [x.name for x in session.get_inputs()] == (
        ["ppg"] if kind == "ppg_tcn" else ["ppg", "acc"]
    )
    batch = 3
    inputs = {"ppg": np.zeros((batch, 1, 512), np.float32)}
    if kind == "fusion_tcn":
        inputs["acc"] = np.zeros((batch, 3, 256), np.float32)
    assert session.run(None, inputs)[0].shape == (batch,)


def test_export_does_not_require_unicode_console(tmp_path, monkeypatch):
    import io
    import sys

    output = io.TextIOWrapper(io.BytesIO(), encoding="gbk")
    monkeypatch.setattr(sys, "stdout", output)
    config = {"channels": [4], "kernel_size": 3, "dilations": [1], "dropout": 0.0}
    stats = {"target_mean": 90.0, "target_std": 20.0, "acc_scale": 1.0}
    result = export_and_verify(
        BPMModel("ppg_tcn", config, stats).eval(), "ppg_tcn", tmp_path, config, stats
    )
    assert result["status"] == "PASS"
