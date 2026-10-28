from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from threading import Lock
from uuid import uuid4

from PIL import Image, UnidentifiedImageError

from app.core.config import Settings
from app.models.schemas import ChallengeType, SessionInfoResponse, UploadResponse

ALLOWED_CONTENT_TYPES = {
    "image/png",
    "image/jpeg",
    "image/jpg",
    "image/webp",
}


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


@dataclass(slots=True)
class SessionAsset:
    session_id: str
    filename: str
    content_type: str
    image_path: Path
    width: int
    height: int
    challenge_id: ChallengeType | None = None
    created_at: datetime = field(default_factory=utcnow)
    cache_version: int = 0
    cache_ready: bool = False
    source_key: str | None = None


class SAM3Service:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._sessions: dict[str, SessionAsset] = {}
        self._sessions_by_source: dict[str, str] = {}
        self._sessions_lock = Lock()

    def create_session(
        self,
        filename: str,
        content_type: str | None,
        content: bytes,
        challenge_id: ChallengeType | None = None,
        source_key: str | None = None,
    ) -> UploadResponse:
        # Callers that re-open a fixed asset (challenge images) pass a stable source_key so
        # revisiting an image reuses its session instead of writing another copy to disk and
        # discarding the worker's cached embeddings.
        if source_key is not None:
            existing = self._find_session_by_source(source_key)
            if existing is not None:
                return self._build_upload_response(existing)

        if len(content) > self._settings.max_upload_size_mb * 1024 * 1024:
            raise ValueError(
                f"Uploaded image exceeds the {self._settings.max_upload_size_mb} MB limit."
            )

        normalized_content_type = (content_type or "").lower()
        if (
            normalized_content_type
            and normalized_content_type not in ALLOWED_CONTENT_TYPES
        ):
            raise ValueError("Unsupported image type. Use PNG, JPG, JPEG, or WEBP.")

        try:
            image = Image.open(BytesIO(content)).convert("RGB")
        except UnidentifiedImageError as exc:
            raise ValueError("The uploaded file is not a supported image.") from exc

        session_id = str(uuid4())
        image_path = self._settings.uploads_dir / f"{session_id}.png"
        image.save(image_path, format="PNG")

        asset = SessionAsset(
            session_id=session_id,
            filename=filename or "upload.png",
            content_type="image/png",
            image_path=image_path,
            width=image.width,
            height=image.height,
            challenge_id=challenge_id,
            source_key=source_key,
        )
        with self._sessions_lock:
            self._sessions[session_id] = asset
            if source_key is not None:
                self._sessions_by_source[source_key] = session_id

        return self._build_upload_response(asset)

    def _find_session_by_source(self, source_key: str) -> SessionAsset | None:
        with self._sessions_lock:
            session_id = self._sessions_by_source.get(source_key)
            if session_id is None:
                return None
            asset = self._sessions.get(session_id)
            if asset is None:
                self._sessions_by_source.pop(source_key, None)
                return None
            if not asset.image_path.exists():
                self._sessions.pop(session_id, None)
                self._sessions_by_source.pop(source_key, None)
                return None
            return asset

    def _build_upload_response(self, asset: SessionAsset) -> UploadResponse:
        return UploadResponse(
            session_id=asset.session_id,
            filename=asset.filename,
            content_type=asset.content_type,
            width=asset.width,
            height=asset.height,
            image_url=f"{self._settings.api_prefix}/sessions/{asset.session_id}/image",
            created_at=asset.created_at,
        )

    def get_session_info(self, session_id: str) -> SessionInfoResponse:
        asset = self._get_session(session_id)
        return SessionInfoResponse(
            session_id=asset.session_id,
            filename=asset.filename,
            content_type=asset.content_type,
            width=asset.width,
            height=asset.height,
            image_url=f"{self._settings.api_prefix}/sessions/{session_id}/image",
            created_at=asset.created_at,
            cache_ready=asset.cache_ready,
        )

    def get_image_path(self, session_id: str) -> Path:
        return self._get_session(session_id).image_path

    def reset_session_cache(self, session_id: str) -> bool:
        with self._sessions_lock:
            asset = self._require_session_locked(session_id)
            asset.cache_version += 1
            asset.cache_ready = False
        return True

    def ensure_session_exists(self, session_id: str) -> None:
        self._get_session(session_id)

    def get_session_asset(self, session_id: str) -> SessionAsset:
        return self._get_session(session_id)

    def mark_cache_ready(self, session_id: str, cache_version: int) -> None:
        with self._sessions_lock:
            asset = self._require_session_locked(session_id)
            if asset.cache_version == cache_version:
                asset.cache_ready = True

    def _get_session(self, session_id: str) -> SessionAsset:
        with self._sessions_lock:
            return self._require_session_locked(session_id)

    def _require_session_locked(self, session_id: str) -> SessionAsset:
        asset = self._sessions.get(session_id)
        if asset is None:
            raise ValueError(f"Unknown session_id: {session_id}")
        return asset
