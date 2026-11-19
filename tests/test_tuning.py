import numpy as np
from choleguard.config import load_thresholds
from choleguard.signal import SignalState
from choleguard.tuning import simulate, tune_thresholds


def _videos():
    rng = np.random.default_rng(0)
    vids, refs = [], []
    for v in range(12):
        n = 30
        ref = np.zeros((n, 3)); ref[20:] = 1 if v % 2 == 0 else 0
        probs = np.clip(ref * 0.8 + rng.normal(0.1, 0.15, (n, 3)), 0, 1)
        vids.append(probs); refs.append(ref)
    return vids, refs


def test_simulate_returns_one_state_per_frame():
    base = load_thresholds("configs/thresholds.yaml")
    states = simulate(np.full((5, 3), 0.9), base)
    assert len(states) == 5 and all(isinstance(s, SignalState) for s in states)


def test_tuned_thresholds_respect_false_green_budget():
    base = load_thresholds("configs/thresholds.yaml")
    vids, refs = _videos()
    t, report = tune_thresholds(vids, refs, base, max_false_green=0.02)
    assert report["false_green_rate"] <= 0.02
    assert report["video_sensitivity"] > 0.5
    assert t.off_threshold <= t.on_threshold
