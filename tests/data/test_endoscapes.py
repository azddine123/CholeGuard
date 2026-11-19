import torch
from choleguard.data.endoscapes import EndoscapesCVSDataset, EndoscapesSegDataset, read_split_videos


def test_seg_dataset_returns_long_mask_in_range(endoscapes_root):
    ds = EndoscapesSegDataset(endoscapes_root, "train")
    assert len(ds) == 2
    img, mask = ds[0]
    assert img.dtype == torch.float32 and img.shape[0] == 3
    assert mask.dtype == torch.long
    vals = set(mask.unique().tolist())
    assert vals <= set(range(7)) | {255}
    assert 3 in vals  # cystic duct remapped from raw id 4


def test_cvs_dataset_only_keyframes_of_split(endoscapes_root):
    ds = EndoscapesCVSDataset(endoscapes_root, "train")
    assert len(ds) == 4  # 2 videos x 2 keyframes, non-keyframe skipped


def test_cvs_dataset_soft_labels_are_mean_of_votes(endoscapes_root):
    ds = EndoscapesCVSDataset(endoscapes_root, "val")
    item = [ds[i] for i in range(len(ds)) if ds[i]["frame_id"] == 125][0]
    assert torch.allclose(item["cvs"], torch.tensor([2 / 3, 1 / 3, 0.0]))
    assert item["votes"].shape == (3, 3)


def test_cvs_dataset_exposes_video_id(endoscapes_root):
    item = EndoscapesCVSDataset(endoscapes_root, "test")[0]
    assert item["video_id"] == "4"


def test_splits_have_no_shared_video(endoscapes_root):
    s = [set(read_split_videos(endoscapes_root, sp)) for sp in ("train", "val", "test")]
    assert not (s[0] & s[1]) and not (s[0] & s[2]) and not (s[1] & s[2])


def test_split_file_in_scientific_notation(endoscapes_root):
    (endoscapes_root / "val_vids.txt").write_text("3.000000000000000000e+00\n")
    assert read_split_videos(endoscapes_root, "val") == ["3"]
