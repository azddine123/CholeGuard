"""CholeGuardNet: segmentation + anatomy-guided CVS head."""
from typing import Literal

import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..classes import NUM_CLASSES

CVSMode = Literal["image", "nomask", "masks"]


class CholeGuardNet(nn.Module):
    """
    cvs_mode:
      - "image"  : baseline A, CVS from pooled image features only
      - "nomask" : same head as "masks" but mask inputs replaced by zeros (capacity-matched ablation)
      - "masks"  : anatomy-guided, CVS uses per-structure pooled features + structure areas
    """

    def __init__(self, backbone: str = "mit_b2", num_classes: int = NUM_CLASSES, cvs_mode: CVSMode = "masks",
                 pretrained: bool = True, mask_dropout: float = 0.2, hidden: int = 256):
        super().__init__()
        self.cvs_mode = cvs_mode
        self.num_classes = num_classes
        self.mask_dropout = mask_dropout
        self.seg = smp.FPN(backbone, encoder_weights="imagenet" if pretrained else None, classes=num_classes)
        c = self.seg.encoder.out_channels[-1]
        n_struct = num_classes - 1  # foreground structures
        in_dim = c if cvs_mode == "image" else c + n_struct * c + num_classes
        self.cvs_head = nn.Sequential(nn.Linear(in_dim, hidden), nn.GELU(), nn.Dropout(0.3), nn.Linear(hidden, 3))

    def forward(self, x: torch.Tensor, mask_probs: torch.Tensor | None = None) -> dict:
        feats = self.seg.encoder(x)
        seg_logits = self.seg.segmentation_head(self.seg.decoder(feats))
        if seg_logits.shape[-2:] != x.shape[-2:]:
            seg_logits = F.interpolate(seg_logits, size=x.shape[-2:], mode="bilinear", align_corners=False)
        deep = feats[-1]
        pooled = deep.mean(dim=(2, 3))
        if self.cvs_mode == "image":
            return dict(seg_logits=seg_logits, cvs_logits=self.cvs_head(pooled))

        if mask_probs is None:
            mask_probs = seg_logits.softmax(1)
        p = F.interpolate(mask_probs.detach().float(), size=deep.shape[-2:], mode="bilinear", align_corners=False)
        if self.cvs_mode == "nomask":
            p = torch.zeros_like(p)
        elif self.training and self.mask_dropout > 0:
            keep = (torch.rand(p.shape[0], p.shape[1], 1, 1, device=p.device) > self.mask_dropout).float()
            p = p * keep
        areas = p.mean(dim=(2, 3))                                       # (B, K)
        fg = p[:, 1:]                                                     # (B, K-1, h, w)
        struct = torch.einsum("bkhw,bchw->bkc", fg, deep) / (fg.sum(dim=(2, 3))[..., None] + 1e-6)
        z = torch.cat([pooled, struct.flatten(1), areas], dim=1)
        return dict(seg_logits=seg_logits, cvs_logits=self.cvs_head(z))
