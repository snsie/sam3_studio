from __future__ import annotations

from fastapi import APIRouter, Depends

from app.api.routes_utils import get_leaderboard_service
from app.models.schemas import ChallengeType, LeaderboardResponse
from app.services.leaderboard_service import LeaderboardService

router = APIRouter(prefix="/leaderboard", tags=["leaderboard"])


@router.get("", response_model=LeaderboardResponse)
def list_leaderboard_entries(
    leaderboard_service: LeaderboardService = Depends(get_leaderboard_service),
) -> LeaderboardResponse:
    return LeaderboardResponse(entries=leaderboard_service.list_entries())


@router.get("/{challenge_id}", response_model=LeaderboardResponse)
def list_challenge_leaderboard_entries(
    challenge_id: ChallengeType,
    leaderboard_service: LeaderboardService = Depends(get_leaderboard_service),
) -> LeaderboardResponse:
    return LeaderboardResponse(
        challenge_id=challenge_id,
        entries=leaderboard_service.list_entries(challenge_id),
    )
