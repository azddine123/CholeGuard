"""Build a demo video (H.264, browser-playable) from Endoscapes 1-fps test frames and pre-analyse it."""
import argparse
import json
import subprocess
import tempfile
from pathlib import Path

from choleguard.config import load_thresholds
from choleguard.inference import OnnxModel, analyze_video


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames-dir", required=True, help="Endoscapes test/ folder")
    ap.add_argument("--video", type=int, required=True)
    ap.add_argument("--start", type=int, default=0, help="first frame number (25 per second)")
    ap.add_argument("--end", type=int, default=10 ** 9)
    ap.add_argument("--model", default="models/choleguard.onnx")
    ap.add_argument("--thresholds", default="configs/thresholds.yaml")
    ap.add_argument("--out-dir", default="backend/samples")
    ap.add_argument("--name")
    a = ap.parse_args()
    frames = sorted((p for p in Path(a.frames_dir).glob(f"{a.video}_*.jpg")
                     if a.start <= int(p.stem.split("_")[1]) <= a.end), key=lambda p: int(p.stem.split("_")[1]))
    if not frames:
        raise SystemExit("no frames")
    name = a.name or f"video{a.video}"
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        for i, f in enumerate(frames):
            (Path(tmp) / f"{i:06d}.jpg").symlink_to(f.resolve())
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", "1", "-i", f"{tmp}/%06d.jpg",
                        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-r", "1", "-preset", "veryfast", "-crf", "23",
                        str(out / f"{name}.mp4")], check=True)
    res = analyze_video(out / f"{name}.mp4", OnnxModel(a.model), load_thresholds(a.thresholds), stride=1)
    (out / f"{name}.json").write_text(json.dumps(res.to_dict()))
    print(f"{name}: {len(frames)} frames, {res.processing_fps:.1f} img/s")


if __name__ == "__main__":
    main()
