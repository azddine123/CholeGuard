"""Segmentation classes shared by training, inference and the UI."""
import numpy as np

CLASSES = [
    "fond",
    "outil",
    "vesicule",
    "canal_cystique",
    "artere_cystique",
    "triangle_hepatocystique",
    "plaque_cystique",
]
NUM_CLASSES = len(CLASSES)

# Endoscapes2023 category names (seg_label_map.txt / COCO categories) -> our ids
ENDOSCAPES_TO_ID = {
    "Gallbladder": 2,
    "Cystic Duct": 3,
    "Cystic Artery": 4,
    "Hepatocystic Triangle Dissection": 5,
    "Cystic Plate": 6,
    "Tool": 1,
}

# CholecSeg8k colour-mask palette -> our ids (classes without equivalent -> 0)
CHOLECSEG_RGB_TO_ID = {
    (127, 127, 127): 0,  # black background
    (210, 140, 140): 0,  # abdominal wall
    (255, 114, 114): 0,  # liver
    (231, 70, 156): 0,   # gastrointestinal tract
    (186, 183, 75): 0,   # fat
    (170, 255, 0): 1,    # grasper
    (255, 85, 0): 0,     # connective tissue
    (255, 0, 0): 0,      # blood
    (255, 255, 0): 3,    # cystic duct
    (169, 255, 184): 1,  # L-hook electrocautery
    (255, 160, 165): 2,  # gallbladder
    (0, 50, 128): 0,     # hepatic vein
    (111, 74, 0): 0,     # liver ligament
}


def remap_cholecseg_mask(rgb: np.ndarray) -> np.ndarray:
    """Convert a CholecSeg8k RGB colour mask (H, W, 3) to a uint8 id map (H, W)."""
    out = np.zeros(rgb.shape[:2], dtype=np.uint8)
    for colour, cid in CHOLECSEG_RGB_TO_ID.items():
        if cid:
            out[np.all(rgb == colour, axis=-1)] = cid
    return out
