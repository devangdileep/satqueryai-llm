from fastapi import APIRouter, HTTPException
from app.services.job_store import job_store

router = APIRouter(prefix="/api/v1/jobs", tags=["Jobs"])


@router.get("/{job_id}")
async def get_job_status(job_id: str):
    """GET /api/v1/jobs/{job_id}: Returns job status and final analysis result when complete."""
    job = job_store.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")

    res = job.get("result")
    return {
        "job_id": job["job_id"],
        "status": job["status"],
        "error": job.get("error"),
        "result": res.model_dump() if res else None
    }


@router.get("/{job_id}/trace")
async def get_job_trace(job_id: str):
    """GET /api/v1/jobs/{job_id}/trace: Returns observable execution trace."""
    job = job_store.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")

    trace = job.get("trace")
    if not trace:
        return {"job_id": job_id, "status": job["status"], "trace": []}

    return trace.model_dump()


@router.get("/{job_id}/evidence")
async def get_job_evidence(job_id: str):
    """GET /api/v1/jobs/{job_id}/evidence: Returns extracted visual/spatial evidence items."""
    job = job_store.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")

    res = job.get("result")
    if not res:
        return {"job_id": job_id, "evidence": []}

    return {"job_id": job_id, "evidence": [e.model_dump() for e in res.evidence]}


@router.get("/{job_id}/artifacts")
async def get_job_artifacts(job_id: str):
    """GET /api/v1/jobs/{job_id}/artifacts: Returns addressable artifacts produced by tools."""
    job = job_store.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")

    res = job.get("result")
    if not res:
        return {"job_id": job_id, "artifacts": []}

    return {"job_id": job_id, "artifacts": res.artifacts}
