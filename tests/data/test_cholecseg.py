from choleguard.data.cholecseg import CholecSegDataset


def test_cholecseg_mask_remapped(cholecseg_root):
    ds = CholecSegDataset(cholecseg_root)
    assert len(ds) == 1
    img, mask = ds[0]
    assert img.shape[0] == 3
    assert mask[0, 0].item() == 3 and mask[-1, -1].item() == 0
