from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, Field, model_validator


class JobStatus(StrEnum):
    QUEUED = "queued"
    LOADING_MODEL = "loading_model"
    LOADING_IMAGE_EMBEDDINGS = "loading_image_embeddings"
    RUNNING_SEGMENTATION = "running_segmentation"
    PACKAGING_MASKS = "packaging_masks"
    COMPLETE = "complete"
    FAILED = "failed"


class PointLabel(StrEnum):
    POSITIVE = "positive"
    NEGATIVE = "negative"


class InteractionMode(StrEnum):
    CLICK = "click"
    POSITIVE_POINT = "positive_point"
    NEGATIVE_POINT = "negative_point"
    BOX = "box"
    TEXT = "text"


class PromptPoint(BaseModel):
    x: float = Field(ge=0)
    y: float = Field(ge=0)
    label: PointLabel


class PromptBox(BaseModel):
    x0: float = Field(ge=0)
    y0: float = Field(ge=0)
    x1: float = Field(ge=0)
    y1: float = Field(ge=0)
    label: PointLabel = PointLabel.POSITIVE

    @model_validator(mode="after")
    def validate_box(self) -> "PromptBox":
        if self.x1 <= self.x0 or self.y1 <= self.y0:
            raise ValueError("Bounding boxes must have positive width and height.")
        return self


class SegmentationRequest(BaseModel):
    session_id: str
    interaction_mode: InteractionMode
    points: list[PromptPoint] = Field(default_factory=list)
    boxes: list[PromptBox] = Field(default_factory=list)
    text_prompt: str | None = None
    multimask_output: bool = True

    @model_validator(mode="after")
    def validate_request(self) -> "SegmentationRequest":
        prompt_count = len(self.points) + len(self.boxes)
        has_text = bool(self.text_prompt and self.text_prompt.strip())
        if self.interaction_mode == InteractionMode.TEXT and not has_text:
            raise ValueError("Text mode requires a non-empty text prompt.")
        if prompt_count == 0 and not has_text:
            raise ValueError("At least one point, box, or text prompt is required.")
        return self



class MaskCandidate(BaseModel):
    candidate_id: str
    score: float | None = None
    width: int
    height: int
    area: int
    image_data_url: str
    color_rgb: list[int]
    centroid: list[float] | None = None
    bounding_box: list[float] | None = None


class SegmentationResult(BaseModel):
    session_id: str
    mode: InteractionMode
    source: Literal["interactive", "text_grounding"]
    text_prompt_supported: bool
    supports_point_refinement: bool
    supports_box_refinement: bool
    warnings: list[str] = Field(default_factory=list)
    candidates: list[MaskCandidate] = Field(default_factory=list)


class UploadResponse(BaseModel):
    session_id: str
    filename: str
    content_type: str
    width: int
    height: int
    image_url: str
    created_at: datetime


class SessionInfoResponse(BaseModel):
    session_id: str
    filename: str
    content_type: str
    width: int
    height: int
    image_url: str
    created_at: datetime
    cache_ready: bool


class SessionResetResponse(BaseModel):
    session_id: str
    reset: bool


class JobSubmissionResponse(BaseModel):
    job_id: str
    status: JobStatus
    status_url: str


class JobStatusResponse(BaseModel):
    job_id: str
    session_id: str
    status: JobStatus
    progress: float = Field(ge=0, le=1)
    message: str
    created_at: datetime
    updated_at: datetime
    result: SegmentationResult | None = None
    error: str | None = None
