from __future__ import annotations

from fastapi import Request

from app.services.job_queue import InferenceQueue
from app.services.challenge_service import ChallengeService
from app.services.leaderboard_service import LeaderboardService
from app.services.nuclei_demo_service import NucleiDemoService
from app.services.sam3_service import SAM3Service
from app.services.scoring_service import ScoringService


def get_sam3_service(request: Request) -> SAM3Service:
    return request.app.state.sam3_service


def get_job_queue(request: Request) -> InferenceQueue:
    return request.app.state.job_queue


def get_challenge_service(request: Request) -> ChallengeService:
    return request.app.state.challenge_service


def get_scoring_service(request: Request) -> ScoringService:
    return request.app.state.scoring_service


def get_leaderboard_service(request: Request) -> LeaderboardService:
    return request.app.state.leaderboard_service


def get_nuclei_demo_service(request: Request) -> NucleiDemoService:
    return request.app.state.nuclei_demo_service
