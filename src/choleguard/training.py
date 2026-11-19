"""Helpers shared by the training / evaluation scripts."""
import math
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .models.net import CholeGuardNet


def pick_device(name: str = "auto") -> torch.device:
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(name)


def add_common_args(ap):
    ap.add_argument("--backbone", default="mit_b2")
    ap.add_argument("--no-pretrained", action="store_true")
    ap.add_argument("--size", nargs=2, type=int, default=[256, 448], metavar=("H", "W"))
    ap.add_argument("--device", default="auto")
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--max-steps", type=int, default=0, help="stop each epoch after N steps (smoke tests)")
    ap.add_argument("--seed", type=int, default=0)
    return ap


def seed_all(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)


def loader(ds, bs, shuffle, workers, collate=None):
    return DataLoader(ds, batch_size=bs, shuffle=shuffle, num_workers=workers, drop_last=False,
                      collate_fn=collate, pin_memory=torch.cuda.is_available())


def cvs_collate(batch):
    return dict(image=torch.stack([b["image"] for b in batch]), cvs=torch.stack([b["cvs"] for b in batch]),
                video_id=[b["video_id"] for b in batch], frame_id=[b["frame_id"] for b in batch])


def save_ckpt(path, model: CholeGuardNet, backbone: str, size, **extra):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(backbone=backbone, cvs_mode=model.cvs_mode, size=list(size),
                    state_dict=model.state_dict(), **extra), path)


def load_ckpt(path, device="cpu", cvs_mode=None) -> tuple[CholeGuardNet, dict]:
    ck = torch.load(path, map_location=device, weights_only=False)
    model = CholeGuardNet(ck["backbone"], cvs_mode=cvs_mode or ck["cvs_mode"], pretrained=False)
    sd = ck["state_dict"]
    if cvs_mode and cvs_mode != ck["cvs_mode"]:
        sd = {k: v for k, v in sd.items() if not k.startswith("cvs_head.")}
    model.load_state_dict(sd, strict=not (cvs_mode and cvs_mode != ck["cvs_mode"]))
    return model.to(device), ck


def cosine_lr(opt, step, total, base_lr, warmup=50):
    lr = base_lr * min(1.0, (step + 1) / warmup) * 0.5 * (1 + math.cos(math.pi * min(step, total) / max(total, 1)))
    for g in opt.param_groups:
        g["lr"] = lr * g.get("lr_scale", 1.0)
