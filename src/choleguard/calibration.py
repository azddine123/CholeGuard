"""Per-criterion temperature scaling (fitted on the validation split)."""
import torch

T_MIN, T_MAX = 0.05, 20.0


def fit_temperature(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    log_t = torch.zeros(logits.shape[1], requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=100)
    lo, hi = torch.log(torch.tensor(T_MIN)), torch.log(torch.tensor(T_MAX))

    def closure():
        opt.zero_grad()
        t = log_t.clamp(lo, hi).exp()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits / t, targets)
        loss.backward()
        return loss
    opt.step(closure)
    t = log_t.detach().clamp(lo, hi).exp()
    return torch.where(torch.isfinite(t), t, torch.ones_like(t))
