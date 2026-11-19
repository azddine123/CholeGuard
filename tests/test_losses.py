import torch
from choleguard.losses import cvs_loss, seg_loss


def test_seg_loss_near_zero_on_perfect_prediction():
    target = torch.randint(0, 7, (2, 16, 16))
    logits = torch.nn.functional.one_hot(target, 7).permute(0, 3, 1, 2).float() * 50
    assert seg_loss(logits, target).item() < 1e-3


def test_seg_loss_ignores_255():
    target = torch.full((1, 8, 8), 255, dtype=torch.long)
    target[0, 0, 0] = 2
    logits = torch.zeros(1, 7, 8, 8); logits[0, 2, 0, 0] = 50
    assert seg_loss(logits, target).item() < 1e-3


def test_cvs_loss_soft_targets_minimal_at_target():
    t = torch.tensor([[2 / 3, 0.0, 1.0]])
    good = torch.logit(t.clamp(1e-4, 1 - 1e-4))
    assert cvs_loss(good, t, torch.ones(3)) < cvs_loss(torch.zeros(1, 3), t, torch.ones(3))
