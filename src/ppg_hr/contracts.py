"""Signal-only inference contracts; references are associated after prediction."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SignalWindow:
    sample_id: str
    subject_id: str
    start_s: float
    end_s: float
    ppg: np.ndarray
    acc: np.ndarray


@dataclass(frozen=True)
class ReferenceRow:
    sample_id: str
    hr_bpm: float
    activity_id: str
    eval_eligible: bool


@dataclass(frozen=True)
class Prediction:
    sample_id: str
    hr_bpm: float | None
    quality_score: float | None
    status: str
    available_time_s: float
    model_id: str
    pipeline_id: str
