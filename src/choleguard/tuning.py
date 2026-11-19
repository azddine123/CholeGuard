"""Signal simulation on per-video sequences and threshold tuning (validation split only)."""
from dataclasses import replace
from itertools import product

import numpy as np

from .config import Thresholds
from .metrics import false_green_rate
from .signal import CVSSignal, SignalState


def simulate(probs: np.ndarray, t: Thresholds, quality_ok=None) -> list:
    sig = CVSSignal(t)
    q = quality_ok if quality_ok is not None else [True] * len(probs)
    return [sig.update(p, ok).state for p, ok in zip(probs, q)]


def raw_thresholds(t: Thresholds) -> Thresholds:
    """No smoothing, no hysteresis, no confirmation: the 'before smoothing' reference."""
    return replace(t, ema_alpha=1.0, off_threshold=t.on_threshold, n_consecutive=1,
                   uncertain_low=t.on_threshold, uncertain_high=t.on_threshold)


def summarize(videos, refs, t: Thresholds) -> dict:
    all_states, all_refs, hit, n_pos = [], [], 0, 0
    for probs, ref in zip(videos, refs):
        states = simulate(probs, t)
        all_states += states
        all_refs.append(ref)
        if (np.asarray(ref) >= 0.5).all(1).any():
            n_pos += 1
            hit += SignalState.GREEN in states
    return dict(false_green_rate=false_green_rate(all_states, np.concatenate(all_refs)),
                video_sensitivity=hit / n_pos if n_pos else float("nan"))


def tune_thresholds(videos, refs, base: Thresholds, max_false_green: float = 0.02):
    """Maximise video-level sensitivity subject to a false-green budget."""
    best, best_key = None, None
    for on, n, alpha in product([0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8], [1, 2, 3, 4], [0.5, 0.7, 1.0]):
        t = replace(base, on_threshold=on, off_threshold=round(on - 0.15, 2), n_consecutive=n, ema_alpha=alpha,
                    uncertain_low=round(on - 0.2, 2), uncertain_high=on)
        r = summarize(videos, refs, t)
        fgr = r["false_green_rate"] if not np.isnan(r["false_green_rate"]) else 0.0
        sens = r["video_sensitivity"] if not np.isnan(r["video_sensitivity"]) else 0.0
        key = (fgr <= max_false_green, sens if fgr <= max_false_green else -fgr, -n)
        if best_key is None or key > best_key:
            best, best_key = (t, r), key
    return best
