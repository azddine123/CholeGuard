"""Each pipeline script must run end-to-end on the tiny fixture (CPU, 1 step)."""
import json
import subprocess
import sys

import yaml

COMMON = ["--backbone", "resnet18", "--no-pretrained", "--size", "64", "96", "--device", "cpu", "--max-steps", "1"]


def run(args, cwd):
    r = subprocess.run([sys.executable, *args], cwd=cwd, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-2000:]


def test_full_pipeline_smoke(endoscapes_root, cholecseg_root, tmp_path):
    repo = __import__("pathlib").Path(__file__).resolve().parents[1]
    out = tmp_path / "out"
    run(["scripts/train_seg.py", "--endoscapes", str(endoscapes_root), "--cholecseg", str(cholecseg_root),
         "--epochs-pre", "1", "--epochs", "1", "--out", str(out / "seg.pt"), *COMMON], repo)
    assert (out / "seg.pt").exists()
    for mode in ("image", "nomask", "masks"):
        run(["scripts/train_cvs.py", "--endoscapes", str(endoscapes_root), "--seg-ckpt", str(out / "seg.pt"),
             "--mode", mode, "--epochs", "1", "--out", str(out / f"cvs_{mode}.pt"), *COMMON], repo)
    thr = tmp_path / "thresholds.yaml"
    run(["scripts/evaluate.py", "--endoscapes", str(endoscapes_root), "--ckpt-dir", str(out),
         "--metrics-out", str(tmp_path / "metrics.json"), "--figures", str(tmp_path / "fig"),
         "--thresholds-in", "configs/thresholds.yaml", "--thresholds-out", str(thr), "--n-boot", "10",
         "--size", "64", "96", "--device", "cpu"], repo)
    m = json.loads((tmp_path / "metrics.json").read_text())
    assert {"segmentation", "cvs", "signal"} <= set(m)
    assert set(m["cvs"]) == {"image", "nomask", "masks"}
    assert yaml.safe_load(thr.read_text())["n_consecutive"] >= 1
    run(["scripts/export_onnx.py", "--ckpt", str(out / "cvs_masks.pt"), "--out", str(tmp_path / "m.onnx"),
         "--size", "64", "96"], repo)
    assert (tmp_path / "m.onnx").exists()
