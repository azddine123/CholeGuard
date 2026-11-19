"""CholecSeg8k loader (pre-training only)."""
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from ..classes import remap_cholecseg_mask
from .transforms import apply, to_tensor


class CholecSegDataset(Dataset):
    def __init__(self, root, transform=None):
        self.transform = transform
        self.images = sorted(Path(root).rglob("*_endo.png"))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        p = self.images[i]
        img = cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB)
        cm = cv2.cvtColor(cv2.imread(str(p.with_name(p.stem + "_color_mask.png"))), cv2.COLOR_BGR2RGB)
        img, mask = apply(self.transform, img, remap_cholecseg_mask(cm))
        return to_tensor(img), torch.from_numpy(np.ascontiguousarray(mask)).long()
