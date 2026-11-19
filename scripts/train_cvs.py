"""CVS head training on a frozen segmentation network (modes: image | nomask | masks) + temperature scaling."""
import argparse
import json

import numpy as np
import torch

from choleguard.data.endoscapes import EndoscapesCVSDataset
from choleguard.data.transforms import get_transforms
from choleguard.losses import cvs_loss
from choleguard.metrics import cvs_metrics
from choleguard.training import add_common_args, cosine_lr, cvs_collate, load_ckpt, loader, pick_device, save_ckpt, seed_all


@torch.no_grad()
def predict_logits(model, dl, device, max_steps=0):
    model.eval()
    logits, targets, vids, frames = [], [], [], []
    for i, b in enumerate(dl):
        if max_steps and i >= max_steps:
            break
        logits.append(model(b["image"].to(device))["cvs_logits"].float().cpu())
        targets.append(b["cvs"]); vids += b["video_id"]; frames += b["frame_id"]
    return torch.cat(logits), torch.cat(targets), np.array(vids), np.array(frames)


def fit_temperature(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    log_t = torch.zeros(3, requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=100)

    def closure():
        opt.zero_grad()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits / log_t.exp(), targets)
        loss.backward()
        return loss
    opt.step(closure)
    return log_t.exp().detach()


def main():
    ap = add_common_args(argparse.ArgumentParser())
    ap.add_argument("--endoscapes", required=True)
    ap.add_argument("--seg-ckpt", required=True)
    ap.add_argument("--mode", choices=["image", "nomask", "masks"], required=True)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    seed_all(a.seed)
    device = pick_device(a.device)
    size = tuple(a.size)
    model, ck = load_ckpt(a.seg_ckpt, device, cvs_mode=a.mode)
    for n, p in model.named_parameters():
        p.requires_grad = n.startswith("cvs_head.")

    train_ds = EndoscapesCVSDataset(a.endoscapes, "train", get_transforms(True, size))
    pos = np.array([r[3] for r in train_ds.rows]) >= 0.5
    pos_weight = torch.tensor(np.clip((1 - pos.mean(0)) / np.maximum(pos.mean(0), 1e-3), 1, 10), dtype=torch.float32)
    train = loader(train_ds, a.bs, True, a.workers, cvs_collate)
    val = loader(EndoscapesCVSDataset(a.endoscapes, "val", get_transforms(False, size)), a.bs, False, a.workers, cvs_collate)

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=a.lr, weight_decay=1e-3)
    total, step, best = a.epochs * (a.max_steps or len(train)), 0, -1.0
    for ep in range(a.epochs):
        model.train()
        model.seg.eval()  # frozen BN statistics
        for i, b in enumerate(train):
            if a.max_steps and i >= a.max_steps:
                break
            cosine_lr(opt, step, total, a.lr)
            with torch.autocast(device.type, enabled=device.type == "cuda"):
                out = model(b["image"].to(device))
            loss = cvs_loss(out["cvs_logits"].float(), b["cvs"].to(device), pos_weight)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            step += 1
        logits, targets, _, _ = predict_logits(model, val, device, a.max_steps)
        m = cvs_metrics(torch.sigmoid(logits).numpy(), targets.numpy())
        score = float(np.nanmean([m[c]["ap"] for c in m])) if not np.isnan([m[c]["ap"] for c in m]).all() else 0.0
        print(f"epoch {ep} val mAP {score:.4f}", flush=True)
        if score > best or ep == 0:
            best = score
            temperature = fit_temperature(logits, targets)
            save_ckpt(a.out, model, ck["backbone"], size, temperature=temperature.tolist(), val_map=score)
    print(json.dumps(dict(best_val_map=best)))


if __name__ == "__main__":
    main()
