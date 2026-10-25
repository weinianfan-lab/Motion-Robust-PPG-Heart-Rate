# Optional extras must be checked before importing the model modules.
# ruff: noqa: E402
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from ppg_hr.models.tcn import PPGTCN, FusionTCN
from ppg_hr.showcase_experiment import fit_training_statistics, summarize_errors


def test_tcn_native_rate_shapes_gradients_and_acc_influence():
    torch.manual_seed(1)
    p = torch.randn(3, 1, 512)
    a = torch.randn(3, 3, 256, requires_grad=True)
    single = PPGTCN()
    fused = FusionTCN()
    assert single(p).shape == (3,)
    fused.eval()
    y = fused(p, a)
    assert y.shape == (3,) and torch.isfinite(y).all()
    y.sum().backward()
    assert a.grad is not None and a.grad.abs().sum() > 0
    assert not torch.equal(y, fused(p, a.detach() + 2))


def test_training_statistics_exclude_held_out_subjects():
    split = {"train": ["S1"], "val": ["S2"], "test": ["S3"]}
    data = {
        "S1": {"labels": np.array([60.0, 80.0])},
        "S2": {"labels": np.array([1e6])},
        "S3": {"labels": np.array([-1e6])},
    }
    meta = {
        "subjects": {
            x: {"continuous_acc_component_sumsq": 12.0, "continuous_acc_samples": 4} for x in data
        }
    }
    stats = fit_training_statistics(data, meta, split)
    assert stats["target_mean"] == 70 and stats["target_std"] == 10
    assert stats["fit_subjects"] == ["S1"] and stats["acc_scale"] == 1
    with pytest.raises(ValueError):
        fit_training_statistics(data, meta, {"train": ["S1"], "val": ["S1"], "test": ["S3"]})


def test_metrics_refuse_missing_predictions_and_keep_full_denominator():
    m = summarize_errors(np.array([60.0, 80.0]), np.array([65.0, 70.0]))
    assert m["n"] == 2 and m["mae"] == 7.5
    assert m["rmse"] == pytest.approx(np.sqrt(62.5))
    with pytest.raises(ValueError):
        summarize_errors(np.array([60.0, 80.0]), np.array([65.0, np.nan]))
