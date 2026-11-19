"""Decision signal shown to the surgeon."""
from enum import Enum


class SignalState(str, Enum):
    RED = "red"            # CVS not reached
    ORANGE = "orange"      # CVS partially reached
    GREEN = "green"        # CVS detected by the model
    UNCERTAIN = "uncertain"
