"""A1/A2 project baselines; no reference or activity information enters inference.

Inputs are externally prepared ``offline_v1`` windows: full-record zero-phase
bandpass, followed by per-window PPG normalization. The estimator itself performs
only spectral analysis. Calling it on an unfiltered fixture tests the spectral
stage, not the full preprocessing pipeline. There is no causal implementation.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from scipy.signal import welch

from ppg_hr.contracts import Prediction, SignalWindow

_PPG_FS = 64.0
_ACC_FS = 32.0


@dataclass(frozen=True)
class SpectralConfig:
    """Explicit Welch settings; native signal rates remain fixed by the contract."""

    fs_hz: float = 64.0
    window: str = "hann"
    nperseg: int = 256
    noverlap: int = 128
    nfft: int = 4096
    f_min_hz: float = 0.5
    f_max_hz: float = 4.0
    detrend: str = "constant"
    return_onesided: bool = True
    scaling: str = "density"
    average: str = "mean"
    epsilon: float = 1e-12
    minimum_std: float = 1e-10

    def __post_init__(self) -> None:
        if self.fs_hz != _PPG_FS:
            raise ValueError("SignalWindow requires fs_hz=64")
        if any(
            type(value) is not int or value % 2
            for value in (
                self.nperseg,
                self.noverlap,
                self.nfft,
            )
        ):
            raise ValueError("Welch sample counts must be even integers for native ACC grid")
        if not (2 <= self.nperseg <= 512 and 0 <= self.noverlap < self.nperseg):
            raise ValueError("Welch segment/overlap is incompatible with the 512-sample window")
        if self.nfft < self.nperseg:
            raise ValueError("nfft must be >= nperseg")
        if not (0 < self.f_min_hz < self.f_max_hz < self.fs_hz / 2):
            raise ValueError("Candidate frequencies must lie strictly within Nyquist")
        if not (
            np.isfinite(self.epsilon)
            and self.epsilon > 0
            and np.isfinite(self.minimum_std)
            and self.minimum_std >= 0
        ):
            raise ValueError("epsilon/minimum_std must be finite positive/nonnegative values")
        if (self.window, self.detrend, self.return_onesided, self.scaling, self.average) != (
            "hann",
            "constant",
            True,
            "density",
            "mean",
        ):
            raise ValueError("This baseline requires Hann/constant/one-sided/density/mean Welch")


class WelchEstimator:
    """A1: the largest candidate-band Welch PSD peak, without label-driven choice.

    Finite predictions can carry ``edge_peak``, ``weak_peak``, and
    ``harmonic_ambiguity`` flags joined with ``|``; these are not rejections.
    Degenerate inputs produce a null HR, zero quality and explicit status.
    Quality is local spectral concentration, not calibrated confidence.
    ``available_time_s`` is the window end (input availability lower bound),
    not a measurement of runtime or a claim of online availability.
    """

    model_id = "A1"

    def __init__(
        self,
        *,
        pipeline_id: str = "offline_v1",
        config: SpectralConfig | None = None,
    ) -> None:
        if pipeline_id != "offline_v1":
            raise ValueError("Only externally preprocessed offline_v1 is implemented")
        self.pipeline_id = pipeline_id
        self.spectral_config = SpectralConfig() if config is None else config
        if not isinstance(self.spectral_config, SpectralConfig):
            raise TypeError("config must be a SpectralConfig instance")

    @property
    def config(self) -> dict:
        """Serializable resolved settings alias for artifact builders."""
        return self.configuration

    @property
    def configuration(self) -> dict:
        return {
            **asdict(self.spectral_config),
            "model_id": self.model_id,
            "pipeline_id": self.pipeline_id,
            "preprocessing": "external_full_record_bandpass_then_window_ppg_normalization",
            "candidate_hz": [self.spectral_config.f_min_hz, self.spectral_config.f_max_hz],
            "weak_peak_max_to_median_ratio": 5.0,
            "harmonic_half_frequency_relative_power": 0.25,
            "quality_definition": "score_sum_within_0.15_hz_of_peak_divided_by_band_score_sum",
            "edge_definition": "peak_at_first_or_last_candidate_grid_point",
            "peak_tie_rule": "lowest_frequency",
            "available_time_definition": (
                "window_end_input_availability_lower_bound_no_runtime_added"
            ),
            "zero_padding_note": "dense_grid_does_not_increase_observation_resolution",
        }

    def _invalid_status(self, window: SignalWindow) -> str | None:
        if np.shape(window.ppg) != (1, 512) or np.shape(window.acc) != (3, 256):
            return "invalid_shape"
        ppg = np.asarray(window.ppg, dtype=np.float64)
        acc = np.asarray(window.acc, dtype=np.float64)
        if not np.isfinite(ppg).all() or not np.isfinite(acc).all():
            return "nonfinite_input"
        if np.ptp(ppg) == 0:
            return "flat_signal"
        if np.std(ppg) <= self.spectral_config.minimum_std:
            return "low_amplitude"
        return None

    def _ppg_spectrum(self, window: SignalWindow) -> tuple[np.ndarray, np.ndarray]:
        config = self.spectral_config
        frequencies, psd = welch(
            np.asarray(window.ppg[0], dtype=np.float64),
            fs=config.fs_hz,
            window=config.window,
            nperseg=config.nperseg,
            noverlap=config.noverlap,
            nfft=config.nfft,
            detrend=config.detrend,
            return_onesided=config.return_onesided,
            scaling=config.scaling,
            axis=-1,
            average=config.average,
        )
        band = (frequencies >= config.f_min_hz) & (frequencies <= config.f_max_hz)
        return frequencies[band], psd[band]

    def spectrum(self, window: SignalWindow) -> dict[str, np.ndarray]:
        """Return the public candidate grid and PSD/score used by this estimator."""
        status = self._invalid_status(window)
        if status is not None:
            raise ValueError(f"Cannot form usable spectrum: {status}")
        frequencies, ppg_psd = self._ppg_spectrum(window)
        return {
            "frequencies_hz": frequencies,
            "ppg_psd": ppg_psd,
            "acc_psd": np.zeros_like(ppg_psd),
            "score": ppg_psd,
        }

    def predict(self, window: SignalWindow) -> Prediction:
        status = self._invalid_status(window)
        if status is not None:
            return Prediction(
                window.sample_id,
                None,
                0.0,
                status,
                window.end_s,
                self.model_id,
                self.pipeline_id,
            )
        spectrum = self.spectrum(window)
        frequencies, score = spectrum["frequencies_hz"], spectrum["score"]
        peak_index = int(np.argmax(score))
        peak_hz = float(frequencies[peak_index])
        peak_power = float(score[peak_index])
        total = float(np.sum(score))
        if not np.isfinite(total) or total <= 0:
            return Prediction(
                window.sample_id,
                None,
                0.0,
                "flat_spectrum",
                window.end_s,
                self.model_id,
                self.pipeline_id,
            )
        flags: list[str] = []
        if peak_index in (0, len(frequencies) - 1):
            flags.append("edge_peak")
        if peak_power / max(float(np.median(score)), np.finfo(float).tiny) < 5.0:
            flags.append("weak_peak")
        half_hz = peak_hz / 2
        if half_hz >= frequencies[0]:
            half_power = float(np.interp(half_hz, frequencies, score))
            if half_power >= 0.25 * peak_power:
                flags.append("harmonic_ambiguity")
        neighborhood = np.abs(frequencies - peak_hz) <= 0.15
        quality = float(np.clip(np.sum(score[neighborhood]) / total, 0.0, 1.0))
        return Prediction(
            window.sample_id,
            60.0 * peak_hz,
            quality,
            "|".join(flags) or "ok",
            window.end_s,
            self.model_id,
            self.pipeline_id,
        )


class ACCSpectralEstimator(WelchEstimator):
    """A2: maximum-normalized PPG PSD / (1 + lambda * ACC PSD + epsilon).

    Three centered native 32 Hz ACC axis PSDs are summed and interpolated onto
    the PPG physical frequency grid. Both segment lengths represent 4 seconds.
    The rule can fail when cardiac and motion spectra overlap.
    """

    model_id = "A2"

    def __init__(
        self,
        *,
        pipeline_id: str = "offline_v1",
        motion_weight: float = 1.0,
        config: SpectralConfig | None = None,
    ) -> None:
        super().__init__(pipeline_id=pipeline_id, config=config)
        if self.spectral_config.f_max_hz > _ACC_FS / 2:
            raise ValueError("A2 candidate frequencies must not exceed ACC Nyquist (16 Hz)")
        if not np.isfinite(motion_weight) or motion_weight < 0:
            raise ValueError("motion_weight must be finite and nonnegative")
        self.motion_weight = float(motion_weight)

    @property
    def configuration(self) -> dict:
        return {
            **super().configuration,
            "acc_fs_hz": _ACC_FS,
            "acc_nperseg": self.spectral_config.nperseg // 2,
            "acc_noverlap": self.spectral_config.noverlap // 2,
            "acc_nfft": self.spectral_config.nfft // 2,
            "acc_axes": "per_axis_center_then_sum_psd",
            "acc_grid_mapping": "linear_interpolation_in_hz_onto_ppg_grid",
            "band_normalization": "each_psd_divided_by_its_band_maximum",
            "motion_weight": self.motion_weight,
            "score_formula": "PPG_norm / (1 + motion_weight * ACC_norm + epsilon)",
        }

    def spectrum(self, window: SignalWindow) -> dict[str, np.ndarray]:
        output = super().spectrum(window)
        config = self.spectral_config
        acc = np.asarray(window.acc, dtype=np.float64)
        acc = acc - acc.mean(axis=1, keepdims=True)
        acc_frequencies, acc_psd = welch(
            acc,
            fs=_ACC_FS,
            window=config.window,
            nperseg=config.nperseg // 2,
            noverlap=config.noverlap // 2,
            nfft=config.nfft // 2,
            detrend=config.detrend,
            return_onesided=config.return_onesided,
            scaling=config.scaling,
            axis=-1,
            average=config.average,
        )
        summed = np.sum(acc_psd, axis=0)
        aligned = np.interp(output["frequencies_hz"], acc_frequencies, summed)
        ppg_norm = output["ppg_psd"] / max(float(np.max(output["ppg_psd"])), config.epsilon)
        acc_norm = aligned / max(float(np.max(aligned)), config.epsilon)
        output["acc_psd"] = aligned
        output["score"] = ppg_norm / (1.0 + self.motion_weight * acc_norm + config.epsilon)
        return output
