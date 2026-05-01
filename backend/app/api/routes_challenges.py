from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import FileResponse

from app.api.routes_utils import (
    get_challenge_service,
    get_leaderboard_service,
    get_sam3_service,
    get_scoring_service,
)
from app.models.schemas import (
    ChallengeExamplesResponse,
    ChallengeImageListResponse,
    ChallengeScoreRequest,
    ChallengeScoreSummary,
    ChallengeSummary,
    ChallengeType,
    UploadResponse,
)
from app.services.challenge_service import ChallengeService
from app.services.leaderboard_service import (
    DuplicateSubmissionError,
    LeaderboardService,
)
from app.services.sam3_service import SAM3Service
from app.services.scoring_service import ScoringService

router = APIRouter(prefix="/challenges", tags=["challenges"])


def _file_response(path: Path) -> FileResponse:
    suffix = path.suffix.lower()
    media_type = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".webp": "image/webp",
    }.get(suffix, "application/octet-stream")
    return FileResponse(path, media_type=media_type, filename=path.name)


@router.get("", response_model=list[ChallengeSummary])
def list_challenges(
    challenge_service: ChallengeService = Depends(get_challenge_service),
) -> list[ChallengeSummary]:
    return challenge_service.list_challenges()


@router.get("/{challenge_id}/images", response_model=ChallengeImageListResponse)
def list_challenge_images(
    challenge_id: ChallengeType,
    challenge_service: ChallengeService = Depends(get_challenge_service),
) -> ChallengeImageListResponse:
    return challenge_service.list_images(challenge_id)


@router.get("/{challenge_id}/images/{image_id}")
def get_challenge_image(
    challenge_id: ChallengeType,
    image_id: str,
    challenge_service: ChallengeService = Depends(get_challenge_service),
) -> FileResponse:
    try:
        return _file_response(challenge_service.get_image_path(challenge_id, image_id))
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)
        ) from exc


@router.post(
    "/{challenge_id}/images/{image_id}/load",
    response_model=UploadResponse,
    status_code=status.HTTP_201_CREATED,
)
def load_challenge_image(
    challenge_id: ChallengeType,
    image_id: str,
    challenge_service: ChallengeService = Depends(get_challenge_service),
    sam3_service: SAM3Service = Depends(get_sam3_service),
) -> UploadResponse:
    try:
        return challenge_service.create_session_for_image(
            challenge_id, image_id, sam3_service
        )
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)
        ) from exc
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)
        ) from exc


@router.get("/{challenge_id}/examples", response_model=ChallengeExamplesResponse)
def get_challenge_examples(
    challenge_id: ChallengeType,
    challenge_service: ChallengeService = Depends(get_challenge_service),
) -> ChallengeExamplesResponse:
    return challenge_service.get_examples(challenge_id)


@router.get("/{challenge_id}/ground-truth/{image_id}/{mask_id}")
def get_ground_truth_mask(
    challenge_id: ChallengeType,
    image_id: str,
    mask_id: str,
    challenge_service: ChallengeService = Depends(get_challenge_service),
) -> FileResponse:
    try:
        return _file_response(
            challenge_service.get_mask_path(challenge_id, image_id, mask_id)
        )
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)
        ) from exc


@router.post("/{challenge_id}/score", response_model=ChallengeScoreSummary)
def score_challenge_dataset(
    challenge_id: ChallengeType,
    request: ChallengeScoreRequest,
    scoring_service: ScoringService = Depends(get_scoring_service),
    leaderboard_service: LeaderboardService = Depends(get_leaderboard_service),
) -> ChallengeScoreSummary:
    try:
        summary = scoring_service.score_submission(challenge_id, request)
        stored_summary = leaderboard_service.add_entry(summary)
        return stored_summary
    except DuplicateSubmissionError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT, detail=str(exc)
        ) from exc
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)
        ) from exc
