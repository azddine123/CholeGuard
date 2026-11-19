"""Endoscapes2023 loaders (segmentation: Seg50, CVS: CVS201 keyframes)."""
import ast
import csv
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from ..classes import remap_endoscapes_semseg
from .transforms import apply, to_tensor


def read_split_videos(root, split: str, seg: bool = False) -> list[str]:
    name = f"{split}_seg_vids.txt" if seg else f"{split}_vids.txt"
    return [str(int(float(v))) for v in Path(root, name).read_text().split() if v.strip()]


def _rgb(path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


class EndoscapesSegDataset(Dataset):
    def __init__(self, root, split: str, transform=None):
        self.root = Path(root)
        self.transform = transform
        vids = set(read_split_videos(root, split, seg=True))
        self.items = sorted(p.stem for p in (self.root / "semseg").glob("*.png") if p.stem.split("_")[0] in vids)
        self.img_dirs = [self.root / d for d in ("train", "val", "test", f"{split}_seg")]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        stem = self.items[i]
        img = _rgb(next((d / f"{stem}.jpg" for d in self.img_dirs if (d / f"{stem}.jpg").exists()),
                        self.img_dirs[0] / f"{stem}.jpg"))
        mask = remap_endoscapes_semseg(cv2.imread(str(self.root / "semseg" / f"{stem}.png"), cv2.IMREAD_UNCHANGED))
        img, mask = apply(self.transform, img, mask)
        return to_tensor(img), torch.from_numpy(np.ascontiguousarray(mask)).long()


class EndoscapesCVSDataset(Dataset):
    """CVS keyframes. `cvs` = mean of the 3 expert votes (soft label)."""

    def __init__(self, root, split: str, transform=None, mask_dir=None):
        self.root = Path(root)
        self.split = split
        self.transform = transform
        self.mask_dir = Path(mask_dir) if mask_dir else None
        vids = set(read_split_videos(root, split))
        self.rows = []
        with open(self.root / "all_metadata.csv", newline="") as fh:
            for r in csv.DictReader(fh):
                if r["is_ds_keyframe"] == "True" and r["vid"] in vids:
                    votes = [ast.literal_eval(r[f"cvs_annotator_{k}"]) for k in (1, 2, 3)] \
                        if r["cvs_annotator_1"] else None
                    self.rows.append((r["vid"], int(r["frame"]), votes,
                                      [float(r["C1"]), float(r["C2"]), float(r["C3"])]))

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        vid, frame, votes, avg = self.rows[i]
        img = _rgb(self.root / self.split / f"{vid}_{frame}.jpg")
        img, _ = apply(self.transform, img)
        cvs = torch.tensor(np.mean(votes, axis=0) if votes else avg, dtype=torch.float32)
        item = dict(image=to_tensor(img), cvs=cvs,
                    votes=torch.tensor(votes, dtype=torch.float32) if votes else None,
                    video_id=vid, frame_id=frame, mask_probs=None)
        if self.mask_dir is not None:
            item["mask_probs"] = torch.from_numpy(np.load(self.mask_dir / f"{vid}_{frame}.npy").astype(np.float32))
        return item
