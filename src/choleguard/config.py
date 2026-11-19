"""Decision thresholds, tuned on the validation split only."""
from dataclasses import dataclass, fields
from pathlib import Path

import yaml


@dataclass(frozen=True)
class Thresholds:
    ema_alpha: float
    on_threshold: float
    off_threshold: float
    uncertain_low: float
    uncertain_high: float
    n_consecutive: int
    blur_min: float
    brightness_min: float
    brightness_max: float
    stride: int


def load_thresholds(path) -> Thresholds:
    raw = yaml.safe_load(Path(path).read_text())
    t = Thresholds(**{f.name: raw[f.name] for f in fields(Thresholds)})
    if t.off_threshold > t.on_threshold:
        raise ValueError("off_threshold must be <= on_threshold (hysteresis)")
    if not 0 < t.ema_alpha <= 1 or t.n_consecutive < 1 or t.stride < 1:
        raise ValueError("invalid ema_alpha, n_consecutive or stride")
    return t
