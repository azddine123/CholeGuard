import json
import time

import pytest
from fastapi.testclient import TestClient

from backend.app import create_app
from tests.conftest import write_video


@pytest.fixture
def client(dummy_onnx, tmp_path):
    samples = tmp_path / "samples"
    samples.mkdir()
    (samples / "demo.json").write_text(json.dumps({"fps": 25.0, "frames": []}))
    write_video(samples / "demo.avi")
    app = create_app(model_path=dummy_onnx, thresholds_path="configs/thresholds.yaml",
                     samples_dir=samples, max_upload_mb=1)
    return TestClient(app)


def wait_done(client, job_id, timeout=20):
    t0 = time.time()
    while time.time() - t0 < timeout:
        s = client.get(f"/api/jobs/{job_id}").json()
        if s["status"] in ("done", "error"):
            return s
        time.sleep(0.1)
    raise TimeoutError


def test_health(client):
    assert client.get("/api/health").json() == {"status": "ok"}


def test_analyze_returns_job_and_completes(client, tmp_path):
    v = write_video(tmp_path / "v.avi", n=10)
    r = client.post("/api/analyze", files={"file": ("v.avi", v.read_bytes(), "video/x-msvideo")})
    assert r.status_code == 200
    job = r.json()["job_id"]
    assert wait_done(client, job)["status"] == "done"
    res = client.get(f"/api/jobs/{job}/result").json()
    assert len(res["frames"]) > 0 and res["frames"][0]["state"] in {"red", "orange", "green", "uncertain"}


def test_rejects_non_video_extension_400(client):
    r = client.post("/api/analyze", files={"file": ("notes.txt", b"hello", "text/plain")})
    assert r.status_code == 400


def test_corrupted_video_400_and_no_stuck_job(client):
    r = client.post("/api/analyze", files={"file": ("broken.mp4", b"\x00" * 2048, "video/mp4")})
    assert r.status_code == 400
    assert client.get("/api/jobs").json() == []


def test_too_large_413(client):
    r = client.post("/api/analyze", files={"file": ("big.mp4", b"\x00" * (2 * 1024 * 1024), "video/mp4")})
    assert r.status_code == 413


def test_samples_available_offline(client):
    names = [s["name"] for s in client.get("/api/samples").json()]
    assert names == ["demo"]
    assert client.get("/api/samples/demo").json()["fps"] == 25.0
    assert client.get("/api/samples/demo/video").status_code == 200


def test_unknown_job_404(client):
    assert client.get("/api/jobs/nope").status_code == 404
    assert client.get("/api/samples/nope").status_code == 404


def test_result_before_done_409(client, tmp_path, monkeypatch):
    import backend.app as appmod
    monkeypatch.setattr(appmod, "_run_job", lambda *a, **k: time.sleep(2))
    v = write_video(tmp_path / "v.avi")
    job = client.post("/api/analyze", files={"file": ("v.avi", v.read_bytes(), "video/x-msvideo")}).json()["job_id"]
    assert client.get(f"/api/jobs/{job}/result").status_code == 409
