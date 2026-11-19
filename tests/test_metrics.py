import numpy as np
import pytest
from choleguard.metrics import (bootstrap_ci, cvs_metrics, false_green_rate, fleiss_kappa,
                                flicker_per_minute, seg_dice_per_class)
from choleguard.signal import SignalState


def test_dice_perfect_is_one():
    t = np.random.randint(0, 7, (4, 10, 10)); t[0, 0, :7] = np.arange(7)
    assert np.allclose(seg_dice_per_class(t, t, 7), 1.0)


def test_dice_ignores_255():
    t = np.zeros((1, 4, 4), int); t[0, 0] = 255
    p = np.zeros((1, 4, 4), int); p[0, 0] = 3
    assert seg_dice_per_class(p, t, 7)[0] == pytest.approx(1.0)


def test_cvs_metrics_known_case():
    probs = np.array([[0.9, 0.1, 0.9], [0.8, 0.2, 0.1], [0.2, 0.7, 0.1], [0.1, 0.1, 0.1]])
    targets = np.array([[1, 0, 1], [1, 0, 0], [1, 1, 0], [0, 0, 0]], float)
    m = cvs_metrics(probs, targets)
    assert m["C1"]["sens"] == pytest.approx(2 / 3)
    assert m["C1"]["spec"] == pytest.approx(1.0)
    assert m["C2"]["f1"] == pytest.approx(1.0)
    assert set(m["C3"]) >= {"sens", "spec", "f1", "bacc", "ap"}


def test_fleiss_kappa_perfect_agreement_is_one():
    votes = np.array([[1, 1, 1], [0, 0, 0], [1, 1, 1], [0, 0, 0]])
    assert fleiss_kappa(votes) == pytest.approx(1.0)


def test_fleiss_kappa_known_value():
    # 4 items, 3 raters: P_bar = 2/3, P_e = 0.5 -> kappa = 1/3
    votes = np.array([[1, 1, 0], [0, 0, 1], [1, 1, 1], [0, 0, 0]])
    assert fleiss_kappa(votes) == pytest.approx(1 / 3, abs=1e-6)


def test_bootstrap_ci_contains_point_estimate():
    rng = np.random.default_rng(0)
    probs = rng.random(200); targets = (probs + rng.normal(0, 0.3, 200) > 0.5).astype(float)
    vids = np.repeat(np.arange(20), 10)
    fn = lambda p, t: float(((p > 0.5) == t).mean())
    lo, hi = bootstrap_ci(fn, probs, targets, vids, n=200)
    assert lo <= fn(probs, targets) <= hi


def test_false_green_rate_definition():
    S = SignalState
    states = [S.GREEN, S.GREEN, S.RED, S.ORANGE, S.GREEN]
    cvs_ref = np.array([[1, 1, 1], [1, 1, 0], [0, 0, 0], [1, 0, 0], [1, 1, 1]], float)
    # non-CVS frames: 1,2,3 -> one false green (frame 1)
    assert false_green_rate(states, cvs_ref) == pytest.approx(1 / 3)


def test_flicker_counts_state_changes():
    S = SignalState
    states = [S.RED, S.RED, S.ORANGE, S.RED, S.GREEN, S.GREEN]  # 3 changes
    assert flicker_per_minute(states, fps=1.0) == pytest.approx(3 / (6 / 60))
