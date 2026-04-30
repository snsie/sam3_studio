from __future__ import annotations

from fastapi import Request

from app.services.job_queue import InferenceQueue
from app.services.sam3_service import SAM3Service


def get_sam3_service(request: Request) -> SAM3Service:
    return request.app.state.sam3_service


def get_job_queue(request: Request) -> InferenceQueue:
    return request.app.state.job_queue
