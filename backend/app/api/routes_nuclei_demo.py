from __future__ import annotations

import asyncio
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Response, status
from fastapi.responses import FileResponse

from app.api.routes_utils import get_nuclei_demo_service
from app.models.schemas import (
    NucleiImageListResponse,
    NucleiPredictionResponse,
    NucleiPredictRequest,
)
from app.services.job_queue import QueueFullError
from app.services.nuclei_demo_service import (
    NucleiDemoError,
    NucleiDemoService,
    NucleiModelUnavailableError,
)

router = APIRouter(prefix="/nuclei-demo", tags=["nuclei-demo"])


@router.get("/images", response_model=NucleiImageListResponse)
def list_nuclei_images(
    service: NucleiDemoService = Depends(get_nuclei_demo_service),
) -> NucleiImageListResponse:
    return service.list_images()


@router.get("/images/{image_id}/image")
def get_nuclei_image(
    image_id: str,
    service: NucleiDemoService = Depends(get_nuclei_demo_service),
) -> FileResponse:
    try:
        return FileResponse(service.get_image_path(image_id), media_type="image/png")
    except FileNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc


@router.get("/images/{image_id}/thumbnail")
def get_nuclei_thumbnail(
    image_id: str,
    service: NucleiDemoService = Depends(get_nuclei_demo_service),
) -> Response:
    try:
        content = service.get_thumbnail(image_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    return Response(
        content=content,
        media_type="image/png",
        headers={"Cache-Control": "public, max-age=86400"},
    )


@router.get("/metrics")
def get_nuclei_metrics(
    service: NucleiDemoService = Depends(get_nuclei_demo_service),
) -> dict[str, Any]:
    try:
        return service.get_metrics_summary()
    except NucleiDemoError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc


@router.get("/metrics/{image_id}")
def get_nuclei_image_metrics(
    image_id: str,
    service: NucleiDemoService = Depends(get_nuclei_demo_service),
) -> dict[str, Any]:
    try:
        return service.get_image_metrics(image_id)
    except (FileNotFoundError, NucleiDemoError) as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc


@router.post("/images/{image_id}/predict", response_model=NucleiPredictionResponse)
async def predict_nuclei(
    image_id: str,
    payload: NucleiPredictRequest,
    service: NucleiDemoService = Depends(get_nuclei_demo_service),
) -> NucleiPredictionResponse:
    try:
        return await service.predict(image_id, payload.model_key)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    except NucleiModelUnavailableError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc
    except QueueFullError as exc:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)) from exc
    except asyncio.TimeoutError as exc:
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail="Nuclei inference timed out while waiting for an available model.",
        ) from exc
    except NucleiDemoError as exc:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc)) from exc
