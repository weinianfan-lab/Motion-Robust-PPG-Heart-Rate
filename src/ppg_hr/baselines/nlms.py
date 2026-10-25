"""Continuous, label-free multi-reference normalized LMS cancellation."""

import numpy as np


def nlms_cancel(ppg, acc_64hz, *, taps=8, mu=0.05, epsilon=1e-6):
    """Update once per time sample; caller resets by invoking per subject.

    ACC must already be aligned using causal zero-order hold. Each output is
    the a-priori residual before the current update; HR is never an input.
    """
    desired = np.asarray(ppg, dtype=np.float64)
    reference = np.asarray(acc_64hz, dtype=np.float64)
    if (
        desired.ndim != 1
        or reference.shape != (len(desired), 3)
        or not np.isfinite(desired).all()
        or not np.isfinite(reference).all()
    ):
        raise ValueError("NLMS requires finite [N] PPG and aligned [N,3] ACC")
    if type(taps) is not int or taps < 1 or not 0 < mu < 2 or epsilon <= 0:
        raise ValueError("NLMS requires positive taps/epsilon and 0 < mu < 2")
    weights = np.zeros((taps, 3))
    history = np.zeros_like(weights)
    residual = np.empty_like(desired)
    for i in range(len(desired)):
        history[1:] = history[:-1]
        history[0] = reference[i]
        error = desired[i] - np.sum(weights * history)
        residual[i] = error
        weights += mu * error / (epsilon + np.sum(history * history)) * history
    return residual
