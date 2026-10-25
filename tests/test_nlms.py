import numpy as np
import pytest

from ppg_hr.baselines.nlms import nlms_cancel


def test_nlms_learns_correlated_noise_without_reference_hr():
    rng = np.random.default_rng(7)
    motion = rng.normal(size=(24000, 3))
    pulse = np.sin(np.arange(len(motion)) * 2 * np.pi * 1.3 / 64)
    signal = pulse + 2 * motion[:, 0] - 0.4 * np.roll(motion[:, 1], 2)
    cleaned = nlms_cancel(signal, motion, taps=4, mu=0.1)
    assert cleaned.shape == signal.shape and np.isfinite(cleaned).all()
    assert np.mean((cleaned[2000:] - pulse[2000:]) ** 2) < 0.15
    np.testing.assert_array_equal(nlms_cancel(pulse, np.zeros_like(motion)), pulse)


def test_nlms_rejects_bad_shape_and_unstable_step():
    with pytest.raises(ValueError):
        nlms_cancel(np.ones(30), np.ones((29, 3)))
    with pytest.raises(ValueError):
        nlms_cancel(np.ones(30), np.ones((30, 3)), mu=3)
