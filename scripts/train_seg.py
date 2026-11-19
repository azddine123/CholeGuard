"""Segmentation training: optional CholecSeg8k pre-training, then Endoscapes Seg50 fine-tuning."""
import argparse
import json

import numpy as np
import torch
from torch.utils.data import ConcatDataset

from choleguard.classes import NUM_CLASSES
from choleguard.data.cholecseg import CholecSegDataset
from choleguard.data.endoscapes import EndoscapesSegDataset
from choleguard.data.transforms import get_transforms
from choleguard.losses import seg_loss
from choleguard.metrics import seg_dice_per_class
from choleguard.models.net import CholeGuardNet
from choleguard.training import add_common_args, cosine_lr, loader, pick_device, save_ckpt, seed_all


def run_epochs(model, dl, epochs, lr, device, max_steps, scaler):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    total, step = epochs * (max_steps or len(dl)), 0
    for _ in range(epochs):
        model.train()
        for i, (x, y) in enumerate(dl):
            if max_steps and i >= max_steps:
                break
            cosine_lr(opt, step, total, lr)
            x, y = x.to(device), y.to(device)
            with torch.autocast(device.type, enabled=device.type == "cuda"):
                loss = seg_loss(model(x)["seg_logits"].float(), y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            step += 1
        yield


@torch.no_grad()
def evaluate(model, dl, device, max_steps=0):
    model.eval()
    dices = []
    for i, (x, y) in enumerate(dl):
        if max_steps and i >= max_steps:
            break
        pred = model(x.to(device))["seg_logits"].argmax(1).cpu().numpy()
        dices.append(seg_dice_per_class(pred, y.numpy(), NUM_CLASSES))
    d = np.nanmean(np.stack(dices), axis=0)
    return d, float(np.nanmean(d[1:]))


def main():
    ap = add_common_args(argparse.ArgumentParser())
    ap.add_argument("--endoscapes", required=True)
    ap.add_argument("--cholecseg")
    ap.add_argument("--epochs-pre", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--out", default="out/seg.pt")
    a = ap.parse_args()
    seed_all(a.seed)
    device = pick_device(a.device)
    size = tuple(a.size)
    model = CholeGuardNet(a.backbone, cvs_mode="image", pretrained=not a.no_pretrained).to(device)
    scaler = torch.amp.GradScaler(enabled=device.type == "cuda")

    if a.cholecseg and a.epochs_pre:
        dl = loader(CholecSegDataset(a.cholecseg, get_transforms(True, size)), a.bs, True, a.workers)
        for _ in run_epochs(model, dl, a.epochs_pre, a.lr, device, a.max_steps, scaler):
            pass

    train = loader(EndoscapesSegDataset(a.endoscapes, "train", get_transforms(True, size)), a.bs, True, a.workers)
    val = loader(EndoscapesSegDataset(a.endoscapes, "val", get_transforms(False, size)), a.bs, False, a.workers)
    best, history = -1.0, []
    for ep, _ in enumerate(run_epochs(model, train, a.epochs, a.lr, device, a.max_steps, scaler)):
        per_class, mean = evaluate(model, val, device, a.max_steps)
        history.append(dict(epoch=ep, val_dice_fg=mean))
        print(f"epoch {ep} val dice(fg) {mean:.4f}", flush=True)
        if mean > best:
            best = mean
            save_ckpt(a.out, model, a.backbone, size, val_dice=per_class.tolist(), val_dice_fg=mean)
    with open(str(a.out) + ".history.json", "w") as fh:
        json.dump(history, fh)


if __name__ == "__main__":
    main()
