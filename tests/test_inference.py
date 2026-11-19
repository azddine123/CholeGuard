import base64
import tracemalloc

import cv2
import numpy as np
import pytest
import torch

from choleguard.config import load_thresholds
from choleguard.inference import OnnxModel, analyze_video, rescale_for_period
from choleguard.signal import SignalState

H, W = 32, 48


class Dummy(torch.nn.Module):
    def forward(self, image):
        b = image.shape[0]
        seg = torch.zeros(b, 7, H, W)
        seg[:, 3] = 1.0
        cvs = torch.sigmoid(image.mean(dim=(1, 2, 3)))[:, None].repeat(1, 3) * 0 + 0.9
        return seg + image[:, :1, :H, :W] * 0, cvs + image.mean() * 0


@pytest.fixture(scope="module")
def onnx_path(tmp_path_factory):
    p = tmp_path_factory.mktemp("m") / "dummy.onnx"
    torch.onnx.export(Dummy(), torch.zeros(1, 3, H, W), p, input_names=["image"],
                      output_names=["seg_probs", "cvs_probs"], opset_version=17, dynamo=False)
    return p


def make_video(path, n, fps=25.0, size=(96, 64)):
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), fps, size)
    rng = np.random.default_rng(0)
    for _ in range(n):
        vw.write((rng.random((size[1], size[0], 3)) * 255).astype(np.uint8))
    vw.release()
    return path


def test_predict_shapes(onnx_path):
    m = OnnxModel(onnx_path)
    label, probs = m.predict(np.zeros((64, 96, 3), np.uint8))
    assert label.shape == (H, W) and label.dtype == np.uint8 and (label == 3).all()
    assert probs.shape == (3,)


def test_analyze_video_frame_count_respects_stride(onnx_path, tmp_path):
    v = make_video(tmp_path / "v.avi", 20)
    t = load_thresholds("configs/thresholds.yaml")
    res = analyze_video(v, OnnxModel(onnx_path), t, stride=2)
    assert len(res.frames) == 10
    assert [f.index for f in res.frames[:3]] == [0, 2, 4]


def test_timestamps_follow_fps(onnx_path, tmp_path):
    v = make_video(tmp_path / "v.avi", 10, fps=10.0)
    res = analyze_video(v, OnnxModel(onnx_path), load_thresholds("configs/thresholds.yaml"), stride=1)
    assert res.fps == pytest.approx(10.0)
    assert res.frames[5].t == pytest.approx(0.5)


def test_timestamps_from_index_when_fps_missing(monkeypatch, onnx_path, tmp_path):
    import choleguard.inference as inf
    monkeypatch.setattr(inf, "_read_fps", lambda cap: 0.0)
    v = make_video(tmp_path / "v.avi", 6)
    res = analyze_video(v, OnnxModel(onnx_path), load_thresholds("configs/thresholds.yaml"), stride=1)
    assert res.fps == 25.0 and res.frames[5].t == pytest.approx(0.2)


def test_progress_reaches_one(onnx_path, tmp_path):
    seen = []
    analyze_video(make_video(tmp_path / "v.avi", 8), OnnxModel(onnx_path),
                  load_thresholds("configs/thresholds.yaml"), stride=1, progress=seen.append)
    assert seen[-1] == pytest.approx(1.0) and seen == sorted(seen)


def test_mask_png_decodes_to_label_map(onnx_path, tmp_path):
    res = analyze_video(make_video(tmp_path / "v.avi", 2), OnnxModel(onnx_path),
                        load_thresholds("configs/thresholds.yaml"), stride=1)
    png = cv2.imdecode(np.frombuffer(base64.b64decode(res.frames[0].mask_png_b64), np.uint8), cv2.IMREAD_UNCHANGED)
    assert png.shape == (H, W) and (png == 3).all()


def test_streaming_memory_bounded(onnx_path, tmp_path):
    v = make_video(tmp_path / "long.avi", 3000)
    tracemalloc.start()
    analyze_video(v, OnnxModel(onnx_path), load_thresholds("configs/thresholds.yaml"), stride=10)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    assert peak < 200 * 1024 ** 2


def test_model_error_yields_uncertain(onnx_path, tmp_path):
    class Broken:
        input_size = (H, W)

        def predict(self, frame):
            raise RuntimeError("boom")
    res = analyze_video(make_video(tmp_path / "v.avi", 3), Broken(), load_thresholds("configs/thresholds.yaml"), stride=1)
    assert all(f.state == SignalState.UNCERTAIN for f in res.frames)


def test_unreadable_video_raises(onnx_path, tmp_path):
    bad = tmp_path / "bad.mp4"
    bad.write_bytes(b"not a video")
    with pytest.raises(ValueError):
        analyze_video(bad, OnnxModel(onnx_path), load_thresholds("configs/thresholds.yaml"))


def test_rescale_for_period_keeps_confirmation_time():
    t = load_thresholds("configs/thresholds.yaml")
    r = rescale_for_period(t, tuned_period_s=5.0, period_s=0.5)
    assert r.n_consecutive == t.n_consecutive * 10
    assert 0 < r.ema_alpha < t.ema_alpha
