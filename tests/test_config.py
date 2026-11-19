import pytest
import yaml
from choleguard.config import Thresholds, load_thresholds


def test_load_thresholds_roundtrip(tmp_path):
    vals = dict(ema_alpha=0.3, on_threshold=0.6, off_threshold=0.45, uncertain_low=0.4,
                uncertain_high=0.6, n_consecutive=8, blur_min=60.0, brightness_min=25.0,
                brightness_max=235.0, stride=2)
    p = tmp_path / "t.yaml"
    p.write_text(yaml.safe_dump(vals))
    assert load_thresholds(p) == Thresholds(**vals)


def test_repo_config_loads():
    t = load_thresholds("configs/thresholds.yaml")
    assert t.n_consecutive >= 1


def test_load_thresholds_rejects_off_above_on(tmp_path):
    p = tmp_path / "t.yaml"
    p.write_text(yaml.safe_dump(dict(ema_alpha=0.3, on_threshold=0.5, off_threshold=0.7, uncertain_low=0.4,
                                     uncertain_high=0.6, n_consecutive=8, blur_min=60.0, brightness_min=25.0,
                                     brightness_max=235.0, stride=2)))
    with pytest.raises(ValueError):
        load_thresholds(p)
