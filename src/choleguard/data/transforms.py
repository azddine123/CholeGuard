"""Image/mask transforms shared by all datasets."""
import albumentations as A
import numpy as np
import torch

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def get_transforms(train: bool, size: tuple[int, int] = (256, 448)) -> A.Compose:
    h, w = size
    ops = [A.Resize(h, w)]
    if train:
        ops += [
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=10, border_mode=0, p=0.5),
            A.ColorJitter(0.2, 0.2, 0.15, 0.02, p=0.7),
            A.GaussianBlur(p=0.1),
        ]
    return A.Compose(ops)


def to_tensor(image_rgb: np.ndarray) -> torch.Tensor:
    """HWC uint8 RGB -> normalised CHW float tensor."""
    x = (image_rgb.astype(np.float32) / 255.0 - MEAN) / STD
    return torch.from_numpy(x.transpose(2, 0, 1).copy())


def apply(transform, image, mask=None):
    if transform is None:
        return image, mask
    out = transform(image=image, mask=mask) if mask is not None else transform(image=image)
    return out["image"], out.get("mask")
