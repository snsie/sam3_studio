from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status

from app.api.routes_utils import get_job_queue
from app.models.schemas import JobStatusResponse
from app.services.job_queue import JobQueue


router = APIRouter(prefix="/jobs", tags=["jobs"])


@router.get("/{job_id}", response_model=JobStatusResponse)
def get_job_status(
    job_id: str,
    queue: JobQueue = Depends(get_job_queue),
) -> JobStatusResponse:
    job = queue.get(job_id)
    if job is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Job not found")
    return job
