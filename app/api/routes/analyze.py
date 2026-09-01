import json
from pathlib import Path
from typing import Optional
from fastapi import APIRouter, BackgroundTasks, Form, HTTPException, UploadFile, File
from app.agent.agent import satquery_agent
from app.core.config import settings
from app.core.security import get_safe_file_path, validate_file_extension
from app.services.job_store import job_store

router = APIRouter(prefix="/api/v1", tags=["Analysis"])


async def process_analysis_job(job_id: str, query: str, image_paths: list[str], metadata: dict | None = None):
    try:
        job_store.update_status(job_id, "processing")
        result, trace = await satquery_agent.run_analysis(job_id, query, image_paths, metadata)
        job_store.set_result(job_id, result, trace)
    except Exception as e:
        job_store.update_status(job_id, "failed", error=str(e))


@router.post("/analyze")
async def analyze_query(
    background_tasks: BackgroundTasks,
    query: str = Form(...),
    metadata: Optional[str] = Form(None),
    images: list[UploadFile] = File(...)
):
    """POST /api/v1/analyze: Accepts natural language question + satellite imagery.
    Returns a job_id for asynchronous tracking.
    """
    if not images:
        raise HTTPException(status_code=400, detail="At least one satellite image must be uploaded.")

    upload_dir = Path(settings.STORAGE_PATH) / "uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)

    saved_paths = []
    for img in images:
        if not validate_file_extension(img.filename or ""):
            raise HTTPException(status_code=400, detail=f"Unsupported file format for '{img.filename}'.")

        safe_path = get_safe_file_path(str(upload_dir), img.filename or "uploaded_image.png")
        content = await img.read()
        with open(safe_path, "wb") as f:
            f.write(content)
        saved_paths.append(str(safe_path))

    parsed_metadata = json.loads(metadata) if metadata else None

    # Create Job
    job_id = job_store.create_job(query, saved_paths)

    # Queue background task
    background_tasks.add_task(process_analysis_job, job_id, query, saved_paths, parsed_metadata)

    return {
        "job_id": job_id,
        "status": "queued",
        "message": "Analysis job submitted successfully.",
        "image_count": len(saved_paths)
    }
