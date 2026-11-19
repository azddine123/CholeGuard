"""Tiny on-disk fixtures that mimic the real dataset layouts."""
import csv

import cv2
import numpy as np
import pytest


def _img(path, h=48, w=64, value=100):
    cv2.imwrite(str(path), np.full((h, w, 3), value, np.uint8))


@pytest.fixture
def endoscapes_root(tmp_path):
    r = tmp_path / "endoscapes"
    for d in ["train", "val", "test", "train_seg", "val_seg", "test_seg", "semseg"]:
        (r / d).mkdir(parents=True)
    vids = {"train": [1, 2], "val": [3], "test": [4]}
    rows = []
    for split, vs in vids.items():
        (r / f"{split}_vids.txt").write_text("\n".join(map(str, vs)))
        (r / f"{split}_seg_vids.txt").write_text("\n".join(map(str, vs)))
        for v in vs:
            for f in (100, 125):
                _img(r / split / f"{v}_{f}.jpg")
                votes = [[1, 0, 0], [1, 1, 0], [0, 0, 0]] if f == 125 else [[0, 0, 0]] * 3
                avg = np.mean(votes, axis=0).tolist()
                rows.append(dict(vid=v, frame=f, avg_cvs=str(avg), C1=avg[0], C2=avg[1], C3=avg[2],
                                 is_ds_keyframe="True", cvs_annotator_1=str(votes[0]),
                                 cvs_annotator_2=str(votes[1]), cvs_annotator_3=str(votes[2]),
                                 mask_path="", label_path=""))
            # one non-keyframe row that must be ignored
            rows.append(dict(vid=v, frame=150, avg_cvs="[0.0,0.0,0.0]", C1=0, C2=0, C3=0, is_ds_keyframe="False",
                             cvs_annotator_1="", cvs_annotator_2="", cvs_annotator_3="", mask_path="", label_path=""))
            _img(r / split / f"{v}_200.jpg")  # seg keyframes live in the CVS split folders
            m = np.zeros((48, 64), np.uint8)
            m[:10] = 4      # cystic duct
            m[10:12] = 255  # ignore
            cv2.imwrite(str(r / "semseg" / f"{v}_200.png"), m)
    with open(r / "all_metadata.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    return r


@pytest.fixture
def cholecseg_root(tmp_path):
    d = tmp_path / "cholecseg8k" / "video01" / "video01_00080"
    d.mkdir(parents=True)
    _img(d / "frame_80_endo.png")
    cm = np.full((48, 64, 3), (127, 127, 127), np.uint8)
    cm[:5] = (255, 255, 0)  # cystic duct (RGB)
    cv2.imwrite(str(d / "frame_80_endo_color_mask.png"), cv2.cvtColor(cm, cv2.COLOR_RGB2BGR))
    return tmp_path / "cholecseg8k"
