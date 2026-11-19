"""Decision signal shown to the surgeon."""
from enum import Enum


class SignalState(str, Enum):
    RED = "red"            # CVS not reached
    ORANGE = "orange"      # CVS partially reached
    GREEN = "green"        # CVS detected by the model
    UNCERTAIN = "uncertain"


from dataclasses import dataclass, field

import numpy as np

from .config import Thresholds

CRITERIA = ("C1", "C2", "C3")


@dataclass
class SignalOutput:
    state: SignalState
    smoothed: np.ndarray
    missing: list = field(default_factory=list)


class CVSSignal:
    """EMA smoothing + per-criterion hysteresis + confirmation streak before GREEN."""

    def __init__(self, thresholds: Thresholds):
        self.t = thresholds
        self.reset()

    def reset(self):
        self.smoothed = None
        self.acquired = np.zeros(3, bool)
        self.streak = 0

    def _uncertain(self):
        self.streak = 0
        sm = self.smoothed if self.smoothed is not None else np.full(3, np.nan)
        return SignalOutput(SignalState.UNCERTAIN, sm.copy(), [])

    def update(self, cvs_probs, quality_ok: bool) -> SignalOutput:
        if cvs_probs is None or not quality_ok:
            return self._uncertain()
        p = np.asarray(cvs_probs, dtype=float)
        if p.shape != (3,) or not np.all(np.isfinite(p)):
            return self._uncertain()
        a = self.t.ema_alpha
        self.smoothed = p if self.smoothed is None else a * p + (1 - a) * self.smoothed
        s = self.smoothed
        self.acquired = np.where(self.acquired, s >= self.t.off_threshold, s >= self.t.on_threshold)
        missing = [CRITERIA[j] for j in range(3) if not self.acquired[j]]

        if len(missing) == 1:
            j = CRITERIA.index(missing[0])
            if self.t.uncertain_low <= s[j] < self.t.uncertain_high:
                return self._uncertain()
        if not missing:
            self.streak += 1
            state = SignalState.GREEN if self.streak >= self.t.n_consecutive else SignalState.ORANGE
        else:
            self.streak = 0
            state = SignalState.RED if len(missing) == 3 else SignalState.ORANGE
        return SignalOutput(state, s.copy(), missing)
