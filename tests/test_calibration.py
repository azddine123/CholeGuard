import torch
from choleguard.calibration import fit_temperature


def test_temperature_finite_on_degenerate_targets():
    logits = torch.tensor([[-8.0, -9.0, -7.0]] * 4)
    targets = torch.zeros(4, 3)  # perfectly separable -> unconstrained optimum would diverge
    t = fit_temperature(logits, targets)
    assert torch.isfinite(t).all() and (t >= 0.05).all() and (t <= 20).all()


def test_temperature_recovers_scale():
    torch.manual_seed(0)
    true = torch.randn(4000, 3) * 2
    targets = torch.bernoulli(torch.sigmoid(true))
    t = fit_temperature(true * 3, targets)  # over-confident by 3x
    assert torch.allclose(t, torch.full((3,), 3.0), atol=0.4)
