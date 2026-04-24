from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Request, status

from app.api.routes_utils import get_job_queue, get_sam3_service
from app.models.schemas import JobSubmissionResponse, SegmentationRequest
from app.services.job_queue import JobQueue
from app.services.sam3_service import SAM3Service


router = APIRouter(prefix="/segment", tags=["segment"])


@router.post("/jobs", response_model=JobSubmissionResponse, status_code=status.HTTP_202_ACCEPTED)
def create_segmentation_job(
    payload: SegmentationRequest,
    request: Request,
    queue: JobQueue = Depends(get_job_queue),
    service: SAM3Service = Depends(get_sam3_service),
) -> JobSubmissionResponse:
    try:
        service.ensure_session_exists(payload.session_id)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc

    job = queue.submit(payload)
    return JobSubmissionResponse(
        job_id=job.job_id,
        status=job.status,
        status_url=f"{request.app.state.settings.api_prefix}/jobs/{job.job_id}",
    )
