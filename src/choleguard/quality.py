"""Frame quality gate: blur, darkness, smoke/over-exposure."""
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class QualityReport:
    blur: float
    brightness: float
    ok: bool


def image_quality(frame_bgr: np.ndarray, blur_min: float, brightness_min: float, brightness_max: float,
                  size: int = 256) -> QualityReport:
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    scale = size / max(h, w)
    gray = cv2.resize(gray, (max(1, int(w * scale)), max(1, int(h * scale))), interpolation=cv2.INTER_AREA)
    blur = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    bright = float(gray.mean())
    ok = blur >= blur_min and brightness_min <= bright <= brightness_max
    return QualityReport(blur=blur, brightness=bright, ok=ok)
