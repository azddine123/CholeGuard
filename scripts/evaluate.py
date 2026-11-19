"""Full evaluation: segmentation Dice, CVS ablation (image / nomask / masks) with video-bootstrap CIs,
threshold tuning on validation, signal metrics on test, figures."""
import argparse
import json
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

from choleguard.classes import CLASSES, NUM_CLASSES
from choleguard.config import load_thresholds
from choleguard.data.endoscapes import EndoscapesCVSDataset, EndoscapesSegDataset
from choleguard.data.transforms import get_transforms
from choleguard.metrics import CRITERIA, bootstrap_ci, cvs_metrics, flicker_per_minute, seg_dice_per_class
from choleguard.training import cvs_collate, load_ckpt, loader, pick_device
from choleguard.tuning import raw_thresholds, simulate, summarize, tune_thresholds

MODES = ("image", "nomask", "masks")
KEYFRAME_PERIOD_S = 5.0


@torch.no_grad()
def predict(model, temperature, dl, device):
    model.eval()
    T = torch.tensor(temperature)
    P, Y, V, F = [], [], [], []
    for b in dl:
        P.append(torch.sigmoid(model(b["image"].to(device))["cvs_logits"].float().cpu() / T))
        Y.append(b["cvs"]); V += b["video_id"]; F += b["frame_id"]
    return torch.cat(P).numpy(), torch.cat(Y).numpy(), np.array(V), np.array(F)


def by_video(P, Y, V, F):
    vids, refs = [], []
    for v in sorted(set(V), key=lambda s: int(s)):
        idx = np.flatnonzero(V == v)
        idx = idx[np.argsort(F[idx])]
        vids.append(P[idx]); refs.append(Y[idx])
    return vids, refs


def ci_metrics(P, Y, V, n_boot):
    out = cvs_metrics(P, Y)
    for j, c in enumerate(CRITERIA):
        def bacc(p, y, j=j):
            return cvs_metrics(p, y)[CRITERIA[j]]["bacc"]

        def ap(p, y, j=j):
            return cvs_metrics(p, y)[CRITERIA[j]]["ap"]
        out[c]["bacc_ci"] = bootstrap_ci(bacc, P, Y, V, n=n_boot)
        out[c]["ap_ci"] = bootstrap_ci(ap, P, Y, V, n=n_boot)
    out["mean_ap"] = float(np.nanmean([out[c]["ap"] for c in CRITERIA]))
    out["mean_bacc"] = float(np.nanmean([out[c]["bacc"] for c in CRITERIA]))
    return out


def video_level(vids, refs, thr=0.5):
    ref = np.array([(r >= 0.5).all(1).any() for r in refs])
    pred = np.array([(p >= thr).all(1).any() for p in vids])
    tp, tn = (pred & ref).sum(), (~pred & ~ref).sum()
    return dict(videos=int(len(ref)), cvs_videos=int(ref.sum()),
                sens=float(tp / ref.sum()) if ref.sum() else None,
                spec=float(tn / (~ref).sum()) if (~ref).sum() else None)


def time_to_cvs(vids, refs, t):
    deltas = []
    for p, r in zip(vids, refs):
        ref_idx = np.flatnonzero((r >= 0.5).all(1))
        states = simulate(p, t)
        green = [i for i, s in enumerate(states) if s.value == "green"]
        if len(ref_idx) and green:
            deltas.append((green[0] - ref_idx[0]) * KEYFRAME_PERIOD_S)
    return dict(n=len(deltas), median_delay_s=float(np.median(deltas)) if deltas else None,
                note="approximate: expert labels are sampled every 5 s")


def figures(outdir, results, test_preds):
    outdir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 3.5))
    for k, m in enumerate(MODES):
        r = results[m]
        vals = [r[c]["bacc"] for c in CRITERIA]
        err = np.array([[v - r[c]["bacc_ci"][0], r[c]["bacc_ci"][1] - v] for c, v in zip(CRITERIA, vals)]).T
        ax.bar(np.arange(3) + (k - 1) * 0.27, vals, 0.27, yerr=err, capsize=3, label=m)
    ax.set_xticks(range(3), CRITERIA); ax.set_ylabel("balanced accuracy (test)"); ax.set_ylim(0.4, 1); ax.legend()
    fig.tight_layout(); fig.savefig(outdir / "ablation.png", dpi=150); plt.close(fig)

    P, Y = test_preds["masks"][:2]
    fig, ax = plt.subplots(figsize=(4, 4))
    for j, c in enumerate(CRITERIA):
        bins = np.linspace(0, 1, 11)
        idx = np.clip(np.digitize(P[:, j], bins) - 1, 0, 9)
        xs = [P[idx == b, j].mean() for b in range(10) if (idx == b).sum() >= 5]
        ys = [(Y[idx == b, j] >= 0.5).mean() for b in range(10) if (idx == b).sum() >= 5]
        ax.plot(xs, ys, "o-", label=c)
    ax.plot([0, 1], [0, 1], "k--", lw=0.8); ax.set_xlabel("predicted probability"); ax.set_ylabel("observed frequency")
    ax.legend(); fig.tight_layout(); fig.savefig(outdir / "calibration.png", dpi=150); plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(9, 3))
    for j, c in enumerate(CRITERIA):
        m = results["masks"][c]
        cm = np.array([[m["tn"], m["fp"]], [m["fn"], m["tp"]]])
        axes[j].imshow(cm, cmap="Blues")
        for (y, x), v in np.ndenumerate(cm):
            axes[j].text(x, y, str(v), ha="center", va="center")
        axes[j].set_title(c); axes[j].set_xticks([0, 1], ["pred 0", "pred 1"]); axes[j].set_yticks([0, 1], ["ref 0", "ref 1"])
    fig.tight_layout(); fig.savefig(outdir / "confusion.png", dpi=150); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--endoscapes", required=True)
    ap.add_argument("--ckpt-dir", required=True)
    ap.add_argument("--metrics-out", default="docs/results/metrics.json")
    ap.add_argument("--figures", default="docs/results/figures")
    ap.add_argument("--thresholds-in", default="configs/thresholds.yaml")
    ap.add_argument("--thresholds-out", default="configs/thresholds.yaml")
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--size", nargs=2, type=int, default=[256, 448])
    ap.add_argument("--device", default="auto")
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--max-false-green", type=float, default=0.02)
    a = ap.parse_args()
    device = pick_device(a.device)
    size, ckd = tuple(a.size), Path(a.ckpt_dir)
    metrics = {}

    seg_model, _ = load_ckpt(ckd / "seg.pt", device)
    seg_model.eval()
    dices = []
    with torch.no_grad():
        for x, y in loader(EndoscapesSegDataset(a.endoscapes, "test", get_transforms(False, size)), a.bs, False, 0):
            dices.append(seg_dice_per_class(seg_model(x.to(device))["seg_logits"].argmax(1).cpu().numpy(), y.numpy(), NUM_CLASSES))
    d = np.nanmean(np.stack(dices), 0)
    metrics["segmentation"] = dict(test_dice={c: float(v) for c, v in zip(CLASSES, d)}, test_dice_fg=float(np.nanmean(d[1:])))

    dls = {s: loader(EndoscapesCVSDataset(a.endoscapes, s, get_transforms(False, size)), a.bs, False, 0, cvs_collate)
           for s in ("val", "test")}
    results, test_preds, val_preds = {}, {}, {}
    for m in MODES:
        model, ck = load_ckpt(ckd / f"cvs_{m}.pt", device)
        val_preds[m] = predict(model, ck["temperature"], dls["val"], device)
        test_preds[m] = predict(model, ck["temperature"], dls["test"], device)
        P, Y, V, F = test_preds[m]
        results[m] = ci_metrics(P, Y, V, a.n_boot)
        results[m]["video_level"] = video_level(*by_video(P, Y, V, F))
    metrics["cvs"] = results

    base = load_thresholds(a.thresholds_in)
    tuned, val_report = tune_thresholds(*by_video(*val_preds["masks"]), base, a.max_false_green)
    test_v, test_r = by_video(*test_preds["masks"])
    raw = raw_thresholds(tuned)
    states_s = [simulate(p, tuned) for p in test_v]
    states_r = [simulate(p, raw) for p in test_v]
    fpm = lambda S: float(np.mean([flicker_per_minute(s, 1 / KEYFRAME_PERIOD_S) for s in S if len(s) > 1]))
    metrics["signal"] = dict(
        tuned_on="val", val=val_report, test=summarize(test_v, test_r, tuned),
        test_raw_no_smoothing=summarize(test_v, test_r, raw),
        flicker_per_min=dict(raw=fpm(states_r), smoothed=fpm(states_s)),
        time_to_cvs=time_to_cvs(test_v, test_r, tuned), thresholds=asdict(tuned))
    Path(a.thresholds_out).write_text(
        "# Tuned on the validation split by scripts/evaluate.py — do not edit by hand.\n" + yaml.safe_dump(asdict(tuned), sort_keys=False))
    Path(a.metrics_out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.metrics_out).write_text(json.dumps(metrics, indent=2, default=float))
    figures(Path(a.figures), results, test_preds)
    print(json.dumps({m: dict(mean_ap=results[m]["mean_ap"], mean_bacc=results[m]["mean_bacc"]) for m in MODES}, indent=1))
    print(json.dumps(metrics["signal"]["test"]))


if __name__ == "__main__":
    main()
