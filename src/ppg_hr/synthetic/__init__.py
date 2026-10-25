"""Deterministic synthetic evidence, never a substitute for real-data validation."""

from .generator import SCENARIOS, SyntheticRecord, generate_record, window_record
from .preprocessing import offline_bandpass_record

__all__ = [
    "SCENARIOS",
    "SyntheticRecord",
    "generate_record",
    "window_record",
    "offline_bandpass_record",
]
