from __future__ import annotations

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status

from app.api.routes_utils import get_sam3_service
from app.models.schemas import UploadResponse
from app.services.sam3_service import SAM3Service


router = APIRouter(prefix="/upload", tags=["upload"])


@router.post("", response_model=UploadResponse, status_code=status.HTTP_201_CREATED)
async def upload_image(
    file: UploadFile = File(...),
    service: SAM3Service = Depends(get_sam3_service),
) -> UploadResponse:
    try:
        content = await file.read()
        return service.create_session(
            filename=file.filename or "upload.png",
            content_type=file.content_type,
            content=content,
        )
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
