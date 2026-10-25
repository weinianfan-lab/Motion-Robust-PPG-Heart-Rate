"""Small strict YAML configuration boundary; no ignored keys or duplicate keys."""

import math
from dataclasses import dataclass, fields
from pathlib import Path

import yaml

from ppg_hr.errors import ConfigurationError


class _UniqueLoader(yaml.SafeLoader):
    pass


def _mapping(loader, node):
    result = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node)
        if not isinstance(key, str) or key in result:
            raise ConfigurationError("YAML keys must be unique strings")
        result[key] = loader.construct_object(value_node)
    return result


_UniqueLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _mapping)


def load_mapping(path: str | Path | None, allowed: set[str]) -> dict:
    if path is None:
        return {}
    try:
        value = yaml.load(Path(path).read_text(encoding="utf-8"), Loader=_UniqueLoader)
    except (OSError, yaml.YAMLError) as exc:
        raise ConfigurationError(f"Cannot read YAML configuration: {exc}") from exc
    if not isinstance(value, dict):
        raise ConfigurationError("Configuration must be a YAML mapping")
    unknown = set(value) - allowed
    if unknown:
        raise ConfigurationError(f"Unknown configuration keys: {sorted(unknown)}")
    return value


@dataclass(frozen=True)
class DemoConfig:
    seed: int = 42
    duration_s: float = 32.0
    hr_bpm: float = 90.0
    waveform: str = "pulse"
    scenario: str = "all"
    motion_weight: float = 1.0

    def __post_init__(self):
        if type(self.seed) is not int or not 0 <= self.seed < 2**32:
            raise ConfigurationError("seed must be an integer in [0, 2**32)")
        for name in ("duration_s", "hr_bpm", "motion_weight"):
            value = getattr(self, name)
            if type(value) not in (int, float) or not math.isfinite(value):
                raise ConfigurationError(f"{name} must be finite numeric data")
        if self.duration_s < 8 or self.duration_s > 600:
            raise ConfigurationError("demo duration_s must be between 8 and 600 seconds")
        if not float(self.duration_s * 32).is_integer():
            raise ConfigurationError("duration_s must end on both native sensor sample grids")
        if not 0 < self.hr_bpm <= 240 or self.motion_weight < 0:
            raise ConfigurationError("Synthetic hr_bpm must be in (0,240], motion_weight >= 0")
        if self.waveform not in ("sine", "pulse") or not isinstance(self.scenario, str):
            raise ConfigurationError("waveform must be sine or pulse; scenario must be a string")


def demo_config(path=None, **overrides) -> DemoConfig:
    values = load_mapping(path, {item.name for item in fields(DemoConfig)})
    values.update({key: value for key, value in overrides.items() if value is not None})
    return DemoConfig(**values)


def data_config(path=None) -> dict:
    values = {"data_root": None, "cache_root": "cache", "output_root": "runs"}
    values.update(load_mapping(path, set(values)))
    for name, value in values.items():
        if value is None and name == "data_root":
            continue
        if not isinstance(value, str) or not value.strip():
            raise ConfigurationError(f"{name} must be a nonempty path string")
    return values
