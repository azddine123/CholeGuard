"""In-memory analysis job registry."""
import threading
import uuid
from dataclasses import dataclass, field


@dataclass
class Job:
    id: str
    filename: str
    status: str = "queued"      # queued | running | done | error
    progress: float = 0.0
    error: str | None = None
    result: dict | None = field(default=None, repr=False)


class JobStore:
    def __init__(self):
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()

    def create(self, filename: str) -> Job:
        job = Job(id=uuid.uuid4().hex[:12], filename=filename)
        with self._lock:
            self._jobs[job.id] = job
        return job

    def get(self, job_id: str) -> Job | None:
        return self._jobs.get(job_id)

    def all(self) -> list[Job]:
        return list(self._jobs.values())
