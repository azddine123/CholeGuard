import numpy as np
from choleguard.config import Thresholds
from choleguard.signal import CVSSignal, SignalState as S

T = Thresholds(ema_alpha=1.0, on_threshold=0.6, off_threshold=0.45, uncertain_low=0.4, uncertain_high=0.6,
               n_consecutive=3, blur_min=60, brightness_min=25, brightness_max=235, stride=1)
HI, LO = np.array([0.9, 0.9, 0.9]), np.array([0.1, 0.1, 0.1])


def run(sig, seq, ok=True):
    return [sig.update(p, ok).state for p in seq]


def test_red_when_nothing_reached():
    assert run(CVSSignal(T), [LO])[-1] == S.RED


def test_green_requires_n_consecutive():
    assert run(CVSSignal(T), [HI] * 3) == [S.ORANGE, S.ORANGE, S.GREEN]


def test_single_spike_does_not_turn_green():
    assert S.GREEN not in run(CVSSignal(T), [LO, HI, LO, LO])


def test_orange_lists_missing_criterion():
    out = CVSSignal(T).update(np.array([0.9, 0.1, 0.9]), True)
    assert out.state == S.ORANGE and out.missing == ["C2"]


def test_hysteresis_prevents_flicker_near_threshold():
    seq = [np.full(3, v) for v in [0.7, 0.58, 0.62, 0.55, 0.65, 0.5, 0.61] * 3]
    states = run(CVSSignal(T), seq)
    assert sum(a != b for a, b in zip(states, states[1:])) <= 1


def test_smoothing_with_alpha():
    t = Thresholds(**{**T.__dict__, "ema_alpha": 0.5})
    sig = CVSSignal(t)
    sig.update(np.zeros(3), True)
    out = sig.update(np.ones(3), True)
    assert np.allclose(out.smoothed, 0.5)


def test_bad_quality_is_uncertain():
    assert CVSSignal(T).update(HI, False).state == S.UNCERTAIN


def test_nan_probs_are_uncertain():
    assert CVSSignal(T).update(np.array([0.9, np.nan, 0.9]), True).state == S.UNCERTAIN


def test_none_probs_are_uncertain():
    assert CVSSignal(T).update(None, True).state == S.UNCERTAIN


def test_doubtful_last_criterion_is_uncertain():
    sig = CVSSignal(T)
    out = sig.update(np.array([0.9, 0.9, 0.5]), True)
    assert out.state == S.UNCERTAIN


def test_uncertain_frame_resets_green_streak():
    sig = CVSSignal(T)
    states = run(sig, [HI, HI]) + [sig.update(None, True).state] + run(sig, [HI])
    assert S.GREEN not in states


def test_reset_clears_state():
    sig = CVSSignal(T)
    run(sig, [HI] * 3)
    sig.reset()
    assert sig.update(LO, True).state == S.RED
