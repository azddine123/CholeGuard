"""Streaming video analysis with the exported ONNX model (CPU, no external service)."""
import base64
import math
import time
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
import onnxruntime as ort

from .config import Thresholds
from .data.transforms import MEAN, STD
from .quality import image_quality
from .signal import CVSSignal, SignalState

DEFAULT_FPS = 25.0
TUNED_PERIOD_S = 5.0  # thresholds are tuned on expert keyframes sampled every 5 s
MIN_TIME_CONSTANT_S = 3.0  # on continuous video, never display unsmoothed probabilities (anti-flicker)


class OnnxModel:
    def __init__(self, path, threads: int = 0):
        so = ort.SessionOptions()
        if threads:
            so.intra_op_num_threads = threads
        self.session = ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])
        self.input_size = tuple(self.session.get_inputs()[0].shape[2:4])  # (H, W)

    def predict(self, frame_bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        h, w = self.input_size
        rgb = cv2.cvtColor(cv2.resize(frame_bgr, (w, h), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
        x = ((rgb.astype(np.float32) / 255.0 - MEAN) / STD).transpose(2, 0, 1)[None]
        seg, cvs = self.session.run(None, {"image": np.ascontiguousarray(x)})
        return seg[0].argmax(0).astype(np.uint8), cvs[0].astype(float)


@dataclass
class FrameResult:
    index: int
    t: float
    state: SignalState
    smoothed: list
    raw: list
    missing: list
    quality_ok: bool
    mask_png_b64: str


@dataclass
class AnalysisResult:
    fps: float
    n_frames: int
    width: int
    height: int
    stride: int
    processing_fps: float
    frames: list = field(default_factory=list)

    def to_dict(self) -> dict:
        d = asdict(self)
        for f in d["frames"]:
            f["state"] = f["state"].value if isinstance(f["state"], SignalState) else f["state"]
        return d


def rescale_for_period(t: Thresholds, tuned_period_s: float, period_s: float,
                       min_time_constant_s: float = MIN_TIME_CONSTANT_S) -> Thresholds:
    """Keep the tuned smoothing time-constant and confirmation duration at another update rate,
    with a minimum smoothing time-constant so the video signal does not flicker."""
    k = tuned_period_s / period_s
    alpha = 1 - (1 - t.ema_alpha) ** (1 / k) if t.ema_alpha < 1 else 1.0
    if min_time_constant_s > 0:
        alpha = min(alpha, 1 - math.exp(-period_s / min_time_constant_s))
    return replace(t, ema_alpha=alpha, n_consecutive=max(1, math.ceil(t.n_consecutive * k)))


def _read_fps(cap) -> float:
    return float(cap.get(cv2.CAP_PROP_FPS) or 0.0)


def _encode_mask(label: np.ndarray) -> str:
    ok, buf = cv2.imencode(".png", label)
    return base64.b64encode(buf.tobytes()).decode() if ok else ""


def analyze_video(path, model, thresholds: Thresholds, stride: int | None = None,
                  progress: Callable[[float], None] | None = None) -> AnalysisResult:
    cap = cv2.VideoCapture(str(path))
    ok, first = cap.read() if cap.isOpened() else (False, None)
    if not ok:
        cap.release()
        raise ValueError(f"Unreadable or unsupported video: {Path(path).name}")
    fps = _read_fps(cap)
    if not (0 < fps < 1000):
        fps = DEFAULT_FPS
    stride = stride or thresholds.stride
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    sig = CVSSignal(rescale_for_period(thresholds, TUNED_PERIOD_S, stride / fps))
    res = AnalysisResult(fps=fps, n_frames=total, width=first.shape[1], height=first.shape[0], stride=stride,
                         processing_fps=0.0)
    t0, idx, frame = time.perf_counter(), 0, first
    while frame is not None:
        if idx % stride == 0:
            q = image_quality(frame, thresholds.blur_min, thresholds.brightness_min, thresholds.brightness_max)
            try:
                label, probs = model.predict(frame)
            except Exception:
                label, probs = None, None
            out = sig.update(probs, q.ok)
            res.frames.append(FrameResult(
                index=idx, t=idx / fps, state=out.state, smoothed=[float(v) for v in out.smoothed],
                raw=[float(v) for v in probs] if probs is not None else [], missing=out.missing,
                quality_ok=q.ok, mask_png_b64=_encode_mask(label) if label is not None else ""))
            if progress and total:
                progress(min(idx / total, 0.999))
        ok, frame = cap.read()
        frame = frame if ok else None
        idx += 1
    cap.release()
    res.n_frames = idx
    res.processing_fps = len(res.frames) / max(time.perf_counter() - t0, 1e-6)
    if progress:
        progress(1.0)
    return res
