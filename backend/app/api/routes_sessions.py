from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import FileResponse

from app.api.routes_utils import get_sam3_service
from app.models.schemas import SessionInfoResponse, SessionResetResponse
from app.services.sam3_service import SAM3Service


router = APIRouter(prefix="/sessions", tags=["sessions"])


@router.get("/{session_id}", response_model=SessionInfoResponse)
def get_session_info(
    session_id: str,
    service: SAM3Service = Depends(get_sam3_service),
) -> SessionInfoResponse:
    try:
        return service.get_session_info(session_id)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc


@router.get("/{session_id}/image")
def get_session_image(
    session_id: str,
    service: SAM3Service = Depends(get_sam3_service),
) -> FileResponse:
    try:
        image_path = service.get_image_path(session_id)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    return FileResponse(image_path, media_type="image/png")


@router.post("/{session_id}/reset-cache", response_model=SessionResetResponse)
def reset_session_cache(
    session_id: str,
    service: SAM3Service = Depends(get_sam3_service),
) -> SessionResetResponse:
    try:
        reset = service.reset_session_cache(session_id)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    return SessionResetResponse(session_id=session_id, reset=reset)
