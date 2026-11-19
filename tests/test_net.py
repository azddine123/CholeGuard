import pytest
import torch
from choleguard.models.net import CholeGuardNet

ENC = "resnet18"  # small encoder, no download, for tests


@pytest.mark.parametrize("mode", ["image", "nomask", "masks"])
def test_forward_shapes_all_modes(mode):
    net = CholeGuardNet(ENC, cvs_mode=mode, pretrained=False).eval()
    out = net(torch.randn(2, 3, 64, 96))
    assert out["seg_logits"].shape == (2, 7, 64, 96)
    assert out["cvs_logits"].shape == (2, 3)


def test_nomask_and_masks_have_same_parameter_count():
    count = lambda m: sum(p.numel() for p in m.parameters())
    assert count(CholeGuardNet(ENC, cvs_mode="nomask", pretrained=False)) == \
        count(CholeGuardNet(ENC, cvs_mode="masks", pretrained=False))


def test_nomask_mode_ignores_mask_branch():
    torch.manual_seed(0)
    net = CholeGuardNet(ENC, cvs_mode="nomask", pretrained=False).eval()
    x = torch.randn(1, 3, 64, 96)
    a = net(x, mask_probs=torch.rand(1, 7, 64, 96))["cvs_logits"]
    b = net(x, mask_probs=torch.rand(1, 7, 64, 96))["cvs_logits"]
    assert torch.allclose(a, b)


def test_masks_mode_uses_external_masks():
    torch.manual_seed(0)
    net = CholeGuardNet(ENC, cvs_mode="masks", pretrained=False).eval()
    x = torch.randn(1, 3, 64, 96)
    m1 = torch.zeros(1, 7, 64, 96); m1[:, 3] = 1
    m2 = torch.zeros(1, 7, 64, 96); m2[:, 0] = 1
    assert not torch.allclose(net(x, mask_probs=m1)["cvs_logits"], net(x, mask_probs=m2)["cvs_logits"])


def test_mask_dropout_off_in_eval():
    net = CholeGuardNet(ENC, cvs_mode="masks", pretrained=False, mask_dropout=0.9).eval()
    x = torch.randn(1, 3, 64, 96)
    m = torch.rand(1, 7, 64, 96)
    assert torch.allclose(net(x, mask_probs=m)["cvs_logits"], net(x, mask_probs=m)["cvs_logits"])
