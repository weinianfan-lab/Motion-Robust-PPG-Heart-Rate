"""Explicit full-record offline filtering used by the synthetic demo only."""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, sosfiltfilt


def offline_bandpass_record(ppg: np.ndarray) -> np.ndarray:
    """Butterworth N=4 bandpass, SOS forward/backward, finite segments only.

    This uses future samples and must never be represented as causal. Existing
    NaN/Inf samples remain NaN. Finite contiguous runs of 27 or fewer samples
    are marked NaN because they cannot support the explicit padlen=27. Each
    segment longer than that is independently filtered without bridging gaps.
    Both boundaries use odd extension and no startup samples are discarded.
    """
    source = np.asarray(ppg, dtype=np.float64)
    if source.ndim != 2 or source.shape[0] != 1:
        raise ValueError("PPG record must have shape [1, samples]")
    result = np.full(source.shape, np.nan, dtype=np.float64)
    sos = butter(N=4, Wn=[0.5, 4.0], btype="bandpass", fs=64.0, output="sos")
    finite = np.isfinite(source[0])
    transitions = np.diff(np.concatenate(([False], finite, [False])).astype(np.int8))
    starts, ends = np.flatnonzero(transitions == 1), np.flatnonzero(transitions == -1)
    for start, end in zip(starts, ends, strict=True):
        if end - start > 27:
            result[0, start:end] = sosfiltfilt(
                sos,
                source[0, start:end],
                axis=-1,
                padtype="odd",
                padlen=27,
            )
    return result.astype(np.float32)
