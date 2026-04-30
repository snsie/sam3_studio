from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, status

from app.api.routes_utils import get_sam3_service
from app.core.config import get_settings
from app.models.schemas import UploadResponse
from app.services.sam3_service import SAM3Service


router = APIRouter(prefix="/samples", tags=["samples"])

SAMPLE_CONTENT_TYPES = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
}


def _get_samples_dir() -> Path:
    return get_settings().data_dir / "samples"


def _list_sample_paths() -> list[Path]:
    samples_dir = _get_samples_dir()
    if not samples_dir.exists():
        return []

    return sorted(
        (
            sample_path
            for sample_path in samples_dir.iterdir()
            if sample_path.is_file() and sample_path.suffix.lower() in SAMPLE_CONTENT_TYPES
        ),
        key=lambda sample_path: sample_path.name.lower(),
    )


def _resolve_sample_path(filename: str) -> Path:
    if Path(filename).name != filename:
        raise FileNotFoundError("Sample image not found.")

    for sample_path in _list_sample_paths():
        if sample_path.name == filename:
            return sample_path

    raise FileNotFoundError("Sample image not found.")


@router.get("", response_model=list[str])
def list_samples() -> list[str]:
    return [sample_path.name for sample_path in _list_sample_paths()]


@router.post("/{filename}/load", response_model=UploadResponse, status_code=status.HTTP_201_CREATED)
def load_sample(
    filename: str,
    service: SAM3Service = Depends(get_sam3_service),
) -> UploadResponse:
    try:
        sample_path = _resolve_sample_path(filename)
        return service.create_session(
            filename=sample_path.name,
            content_type=SAMPLE_CONTENT_TYPES[sample_path.suffix.lower()],
            content=sample_path.read_bytes(),
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc