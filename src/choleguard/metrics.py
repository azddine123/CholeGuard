"""Evaluation metrics (segmentation, CVS, signal)."""
from typing import Callable, Sequence

import numpy as np
from sklearn.metrics import average_precision_score

from .classes import IGNORE_INDEX
from .signal import SignalState

CRITERIA = ("C1", "C2", "C3")


def seg_dice_per_class(pred: np.ndarray, target: np.ndarray, num_classes: int) -> np.ndarray:
    valid = target != IGNORE_INDEX
    out = np.full(num_classes, np.nan)
    for k in range(num_classes):
        p, t = (pred == k) & valid, (target == k) & valid
        denom = p.sum() + t.sum()
        out[k] = 1.0 if denom == 0 else 2 * (p & t).sum() / denom
    return out


def _binary(probs, targets, thr):
    yp, yt = probs >= thr, targets >= 0.5
    tp, tn = (yp & yt).sum(), (~yp & ~yt).sum()
    fp, fn = (yp & ~yt).sum(), (~yp & yt).sum()
    sens = tp / (tp + fn) if tp + fn else np.nan
    spec = tn / (tn + fp) if tn + fp else np.nan
    f1 = 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else np.nan
    return dict(sens=float(sens), spec=float(spec), f1=float(f1), bacc=float(np.nanmean([sens, spec])),
                tp=int(tp), tn=int(tn), fp=int(fp), fn=int(fn))


def cvs_metrics(probs: np.ndarray, targets: np.ndarray, threshold: float = 0.5) -> dict:
    out = {}
    for j, name in enumerate(CRITERIA):
        m = _binary(probs[:, j], targets[:, j], threshold)
        yt = targets[:, j] >= 0.5
        m["ap"] = float(average_precision_score(yt, probs[:, j])) if 0 < yt.sum() < len(yt) else float("nan")
        out[name] = m
    return out


def bootstrap_ci(metric_fn: Callable, probs, targets, video_ids, n: int = 1000, seed: int = 0, alpha: float = 0.05):
    """Percentile CI, resampling whole videos (frames of a video are not independent)."""
    rng = np.random.default_rng(seed)
    vids = np.unique(video_ids)
    idx_by_vid = {v: np.flatnonzero(video_ids == v) for v in vids}
    stats = []
    for _ in range(n):
        idx = np.concatenate([idx_by_vid[v] for v in rng.choice(vids, len(vids), replace=True)])
        val = metric_fn(probs[idx], targets[idx])
        if not np.isnan(val):
            stats.append(val)
    return float(np.quantile(stats, alpha / 2)), float(np.quantile(stats, 1 - alpha / 2))


def fleiss_kappa(votes: np.ndarray) -> float:
    """votes: (N items, R raters) binary."""
    votes = np.asarray(votes, int)
    n = votes.shape[1]
    counts = np.stack([(votes == 0).sum(1), (votes == 1).sum(1)], axis=1)
    p_i = ((counts ** 2).sum(1) - n) / (n * (n - 1))
    p_j = counts.sum(0) / counts.sum()
    pe = (p_j ** 2).sum()
    return float((p_i.mean() - pe) / (1 - pe)) if pe < 1 else 1.0


def false_green_rate(states: Sequence[SignalState], cvs_ref: np.ndarray) -> float:
    """Share of non-CVS frames (reference: not all 3 criteria >= 0.5) shown as GREEN."""
    non_cvs = ~(np.asarray(cvs_ref) >= 0.5).all(axis=1)
    green = np.array([s == SignalState.GREEN for s in states])
    return float((green & non_cvs).sum() / non_cvs.sum()) if non_cvs.any() else float("nan")


def flicker_per_minute(states: Sequence[SignalState], fps: float) -> float:
    changes = sum(a != b for a, b in zip(states[:-1], states[1:]))
    minutes = len(states) / fps / 60
    return changes / minutes if minutes else 0.0
