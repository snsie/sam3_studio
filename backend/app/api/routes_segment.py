from __future__ import annotations

import asyncio

from fastapi import APIRouter, Depends, HTTPException, Request, status

from app.api.routes_utils import get_job_queue, get_sam3_service
from app.models.schemas import JobStatusResponse, SegmentationRequest
from app.services.job_queue import InferenceQueue, QueueFullError
from app.services.sam3_service import SAM3Service


router = APIRouter(prefix="/segment", tags=["segment"])


@router.post("/jobs", response_model=JobStatusResponse)
async def create_segmentation_job(
    payload: SegmentationRequest,
    request: Request,
    queue: InferenceQueue = Depends(get_job_queue),
    service: SAM3Service = Depends(get_sam3_service),
) -> JobStatusResponse:
    try:
        service.ensure_session_exists(payload.session_id)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc

    try:
        return await queue.submit_and_wait(
            payload,
            timeout_seconds=request.app.state.settings.sam3_request_timeout_seconds,
        )
    except QueueFullError as exc:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)) from exc
    except asyncio.TimeoutError as exc:
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail=(
                "Segmentation request timed out while waiting for an available model. "
                "Try again or increase SAM3_REQUEST_TIMEOUT_SECONDS."
            ),
        ) from exc
