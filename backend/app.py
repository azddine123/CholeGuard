"""CholeGuard API — analyses surgical videos locally with the exported ONNX model."""
import json
from contextlib import asynccontextmanager
import os
import shutil
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from choleguard.config import load_thresholds
from choleguard.inference import OnnxModel, analyze_video

from .jobs import Job, JobStore

ROOT = Path(__file__).resolve().parents[1]
VIDEO_EXT = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
CHUNK = 1024 * 1024


def _run_job(job: Job, path: Path, model, thresholds):
    job.status = "running"
    try:
        res = analyze_video(path, model, thresholds, progress=lambda p: setattr(job, "progress", round(p, 3)))
        job.result = res.to_dict()
        job.status, job.progress = "done", 1.0
    except Exception as e:  # surfaced to the UI, never crashes the server
        job.status, job.error = "error", str(e)
    finally:
        path.unlink(missing_ok=True)


def create_app(model_path=None, thresholds_path=None, samples_dir=None, max_upload_mb: int = 500,
               workers: int = 1) -> FastAPI:
    model_path = Path(model_path or os.environ.get("CHOLEGUARD_MODEL", ROOT / "models" / "choleguard.onnx"))
    thresholds = load_thresholds(thresholds_path or ROOT / "configs" / "thresholds.yaml")
    samples_dir = Path(samples_dir or ROOT / "backend" / "samples")
    model = OnnxModel(model_path) if model_path.exists() else None
    store, pool = JobStore(), ThreadPoolExecutor(max_workers=workers)
    upload_dir = Path(tempfile.mkdtemp(prefix="choleguard_"))

    @asynccontextmanager
    async def lifespan(_app):
        yield
        pool.shutdown(wait=False)
        shutil.rmtree(upload_dir, ignore_errors=True)

    app = FastAPI(title="CholeGuard API", lifespan=lifespan)
    app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

    @app.get("/api/health")
    def health():
        return {"status": "ok"}

    @app.post("/api/analyze")
    async def analyze(file: UploadFile):
        ext = Path(file.filename or "").suffix.lower()
        if ext not in VIDEO_EXT:
            raise HTTPException(400, f"Format non supporté ({ext or 'inconnu'}). Formats acceptés : {', '.join(sorted(VIDEO_EXT))}")
        if model is None:
            raise HTTPException(503, "Modèle introuvable sur le serveur")
        dest, size = upload_dir / f"{os.urandom(6).hex()}{ext}", 0
        with open(dest, "wb") as fh:
            while chunk := await file.read(CHUNK):
                size += len(chunk)
                if size > max_upload_mb * CHUNK:
                    fh.close(); dest.unlink(missing_ok=True)
                    raise HTTPException(413, f"Fichier trop volumineux (max {max_upload_mb} Mo)")
                fh.write(chunk)
        cap = cv2.VideoCapture(str(dest))
        readable = cap.isOpened() and cap.read()[0]
        cap.release()
        if not readable:
            dest.unlink(missing_ok=True)
            raise HTTPException(400, "Vidéo illisible ou corrompue")
        job = store.create(file.filename)
        pool.submit(lambda: _run_job(job, dest, model, thresholds))
        return {"job_id": job.id}

    @app.get("/api/jobs")
    def jobs():
        return [dict(id=j.id, filename=j.filename, status=j.status, progress=j.progress) for j in store.all()]

    def _job(job_id):
        job = store.get(job_id)
        if job is None:
            raise HTTPException(404, "Analyse inconnue")
        return job

    @app.get("/api/jobs/{job_id}")
    def job_status(job_id: str):
        j = _job(job_id)
        return dict(id=j.id, status=j.status, progress=j.progress, error=j.error)

    @app.get("/api/jobs/{job_id}/result")
    def job_result(job_id: str):
        j = _job(job_id)
        if j.status != "done":
            raise HTTPException(409, "Analyse pas encore terminée")
        return j.result

    @app.get("/api/samples")
    def samples():
        return [dict(name=p.stem) for p in sorted(samples_dir.glob("*.json"))]

    def _sample(name):
        p = samples_dir / f"{Path(name).name}.json"
        if not p.exists():
            raise HTTPException(404, "Exemple inconnu")
        return p

    @app.get("/api/samples/{name}")
    def sample(name: str):
        return json.loads(_sample(name).read_text())

    @app.get("/api/samples/{name}/video")
    def sample_video(name: str):
        p = _sample(name)
        for ext in VIDEO_EXT:
            if (v := p.with_suffix(ext)).exists():
                return FileResponse(v)
        raise HTTPException(404, "Vidéo d'exemple introuvable")

    dist = ROOT / "frontend" / "dist"
    if dist.exists():
        app.mount("/", StaticFiles(directory=dist, html=True), name="ui")

    return app


app = create_app()
