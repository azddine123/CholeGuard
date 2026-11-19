import numpy as np
from choleguard.classes import CLASSES, NUM_CLASSES, CHOLECSEG_RGB_TO_ID, ENDOSCAPES_TO_ID, remap_cholecseg_mask


def test_class_order():
    assert NUM_CLASSES == len(CLASSES) == 7
    assert CLASSES[0] == "fond"
    assert CLASSES[3] == "canal_cystique"
    assert CLASSES[4] == "artere_cystique"


def test_endoscapes_names_map_to_valid_ids():
    assert ENDOSCAPES_TO_ID["Cystic Duct"] == 3
    assert ENDOSCAPES_TO_ID["Tool"] == 1
    assert set(ENDOSCAPES_TO_ID.values()) <= set(range(NUM_CLASSES))


def test_cholecseg_known_colors():
    assert CHOLECSEG_RGB_TO_ID[(255, 255, 0)] == 3   # cystic duct
    assert CHOLECSEG_RGB_TO_ID[(170, 255, 0)] == 1   # grasper -> tool


def test_cholecseg_unknown_color_maps_to_background():
    rgb = np.full((4, 4, 3), (1, 2, 3), dtype=np.uint8)
    assert remap_cholecseg_mask(rgb).max() == 0


def test_remap_preserves_shape_and_dtype():
    rgb = np.zeros((5, 6, 3), dtype=np.uint8)
    rgb[0, 0] = (255, 160, 165)  # gallbladder
    out = remap_cholecseg_mask(rgb)
    assert out.shape == (5, 6) and out.dtype == np.uint8
    assert out[0, 0] == 2
