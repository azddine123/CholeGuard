import torch
import torch.nn.functional as F

from .classes import IGNORE_INDEX


def seg_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Cross-entropy + soft Dice (pixels labelled 255 are ignored)."""
    ce = F.cross_entropy(logits, target, ignore_index=IGNORE_INDEX)
    valid = (target != IGNORE_INDEX).unsqueeze(1).float()
    probs = logits.softmax(1) * valid
    onehot = F.one_hot(target.clamp(max=logits.shape[1] - 1), logits.shape[1]).permute(0, 3, 1, 2).float() * valid
    inter = (probs * onehot).sum(dim=(0, 2, 3))
    denom = probs.sum(dim=(0, 2, 3)) + onehot.sum(dim=(0, 2, 3))
    present = onehot.sum(dim=(0, 2, 3)) > 0
    dice = 1 - (2 * inter + 1e-6) / (denom + 1e-6)
    return ce + (dice[present].mean() if present.any() else logits.sum() * 0)


def cvs_loss(logits: torch.Tensor, soft_targets: torch.Tensor, pos_weight: torch.Tensor) -> torch.Tensor:
    return F.binary_cross_entropy_with_logits(logits, soft_targets, pos_weight=pos_weight.to(logits.device))
