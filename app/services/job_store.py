import uuid
from typing import Any, Optional
from app.schemas.jobs import AnalysisResult, ExecutionTrace


class JobStore:
    """In-memory + File job storage service for tracking asynchronous job execution."""

    def __init__(self):
        self._jobs: dict[str, dict[str, Any]] = {}

    def create_job(self, query: str, image_paths: list[str]) -> str:
        job_id = f"job_{uuid.uuid4().hex[:12]}"
        self._jobs[job_id] = {
            "job_id": job_id,
            "status": "queued",
            "query": query,
            "image_paths": image_paths,
            "result": None,
            "trace": None,
            "error": None
        }
        return job_id

    def update_status(self, job_id: str, status: str, error: Optional[str] = None):
        if job_id in self._jobs:
            self._jobs[job_id]["status"] = status
            if error:
                self._jobs[job_id]["error"] = error

    def set_result(self, job_id: str, result: AnalysisResult, trace: ExecutionTrace):
        if job_id in self._jobs:
            self._jobs[job_id]["result"] = result
            self._jobs[job_id]["trace"] = trace
            self._jobs[job_id]["status"] = "completed"

    def get_job(self, job_id: str) -> Optional[dict[str, Any]]:
        return self._jobs.get(job_id)


job_store = JobStore()
