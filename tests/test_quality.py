import cv2
import numpy as np
from choleguard.quality import image_quality

T = dict(blur_min=60.0, brightness_min=25.0, brightness_max=235.0)


def _textured():
    rng = np.random.default_rng(0)
    img = (rng.random((240, 320, 3)) * 255).astype(np.uint8)
    return cv2.GaussianBlur(img, (3, 3), 0)


def test_sharp_frame_ok():
    assert image_quality(_textured(), **T).ok


def test_blurry_frame_not_ok():
    assert not image_quality(cv2.GaussianBlur(_textured(), (31, 31), 15), **T).ok


def test_black_frame_not_ok():
    assert not image_quality(np.zeros((240, 320, 3), np.uint8), **T).ok


def test_white_smoke_frame_not_ok():
    assert not image_quality(np.full((240, 320, 3), 250, np.uint8), **T).ok
