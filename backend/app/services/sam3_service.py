from __future__ import annotations

import base64
from dataclasses import dataclass, field
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from threading import Lock
from contextlib import nullcontext
from typing import Any
from uuid import uuid4

import numpy as np
from PIL import Image, UnidentifiedImageError

from app.core.config import Settings
from app.models.schemas import (
    InteractionMode,
    JobStatus,
    MaskCandidate,
    PointLabel,
    SegmentationRequest,
    SegmentationResult,
    SessionInfoResponse,
    UploadResponse,
)
from app.services.job_queue import ProgressCallback


ALLOWED_CONTENT_TYPES = {
    "image/png",
    "image/jpeg",
    "image/jpg",
    "image/webp",
}

MASK_PALETTE: tuple[tuple[int, int, int], ...] = (
    (33, 211, 197),
    (244, 114, 182),
    (251, 191, 36),
    (96, 165, 250),
)


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
    created_at: datetime = field(default_factory=utcnow)
    base_state: dict[str, Any] | None = None


class SAM3Service:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._sessions: dict[str, SessionAsset] = {}
        self._sessions_lock = Lock()
        self._model_lock = Lock()
        self._model: Any | None = None
        self._processor: Any | None = None
        self._device: str | None = None

    def preload_model(self) -> None:
        self._ensure_model_loaded()

    def create_session(self, filename: str, content_type: str | None, content: bytes) -> UploadResponse:
        if len(content) > self._settings.max_upload_size_mb * 1024 * 1024:
            raise ValueError(
                f"Uploaded image exceeds the {self._settings.max_upload_size_mb} MB limit."
            )

        normalized_content_type = (content_type or "").lower()
        if normalized_content_type and normalized_content_type not in ALLOWED_CONTENT_TYPES:
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
        )
        with self._sessions_lock:
            self._sessions[session_id] = asset

        return UploadResponse(
            session_id=session_id,
            filename=asset.filename,
            content_type=asset.content_type,
            width=asset.width,
            height=asset.height,
            image_url=f"{self._settings.api_prefix}/sessions/{session_id}/image",
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
            cache_ready=asset.base_state is not None,
        )

    def get_image_path(self, session_id: str) -> Path:
        return self._get_session(session_id).image_path

    def reset_session_cache(self, session_id: str) -> bool:
        asset = self._get_session(session_id)
        asset.base_state = None
        return True

    def ensure_session_exists(self, session_id: str) -> None:
        self._get_session(session_id)

    def run_segmentation(
        self,
        payload: SegmentationRequest,
        progress: ProgressCallback,
    ) -> SegmentationResult:
        has_text = bool(payload.text_prompt and payload.text_prompt.strip())
        has_points = bool(payload.points)

        progress(JobStatus.LOADING_MODEL, "Loading SAM3 model", 0.05)
        self._ensure_model_loaded()

        with self._inference_context():
            progress(JobStatus.LOADING_IMAGE_EMBEDDINGS, "Loading image embeddings", 0.25)
            session = self._get_session(payload.session_id)
            base_state = self._get_or_build_base_state(session)

            progress(JobStatus.RUNNING_SEGMENTATION, "Running SAM3 segmentation", 0.65)
            if has_text and has_points:
                result = self._run_text_guided_interactive_segmentation(
                    session,
                    payload,
                    base_state,
                )
            elif has_text:
                result = self._run_text_grounding(session, payload, base_state)
            else:
                result = self._run_interactive_segmentation(session, payload, base_state)

        progress(JobStatus.PACKAGING_MASKS, "Packaging masks", 0.9)
        return result

    def _ensure_model_loaded(self) -> None:
        if self._model is not None and self._processor is not None:
            return

        with self._model_lock:
            if self._model is not None and self._processor is not None:
                return

            import torch
            from sam3 import build_sam3_image_model
            from sam3.model.sam3_image_processor import Sam3Processor

            requested_device = self._settings.sam3_device.lower()
            if requested_device == "auto":
                self._device = "cuda" if torch.cuda.is_available() else "cpu"
            else:
                self._device = requested_device

            self._model = build_sam3_image_model(
                checkpoint_path=self._settings.sam3_checkpoint_path,
                bpe_path=self._settings.sam3_bpe_path,
                device=self._device,
                eval_mode=True,
                load_from_HF=self._settings.sam3_load_from_hf,
                enable_inst_interactivity=True,
                compile=self._settings.sam3_compile,
            )
            self._processor = Sam3Processor(
                self._model,
                device=self._device,
                confidence_threshold=self._settings.sam3_confidence_threshold,
            )

    def _get_or_build_base_state(self, session: SessionAsset) -> dict[str, Any]:
        if session.base_state is not None:
            return session.base_state

        if self._processor is None:
            raise RuntimeError("SAM3 processor is not initialized.")

        with Image.open(session.image_path) as image:
            session.base_state = self._processor.set_image(image.convert("RGB"))
        return session.base_state

    def _run_interactive_segmentation(
        self,
        session: SessionAsset,
        payload: SegmentationRequest,
        base_state: dict[str, Any],
    ) -> SegmentationResult:
        masks, scores, warnings = self._predict_interactive_masks(payload, base_state)

        candidates = self._build_candidates(masks=masks, scores=scores)
        if not candidates:
            warnings.append("SAM3 returned no masks for the current prompts.")

        return SegmentationResult(
            session_id=session.session_id,
            mode=payload.interaction_mode,
            source="interactive",
            text_prompt_supported=False,
            supports_point_refinement=True,
            supports_box_refinement=True,
            warnings=warnings,
            candidates=candidates,
        )

    def _predict_interactive_masks(
        self,
        payload: SegmentationRequest,
        base_state: dict[str, Any],
    ) -> tuple[np.ndarray, np.ndarray, list[str]]:
        if self._model is None:
            raise RuntimeError("SAM3 model is not initialized.")

        point_coords = None
        point_labels = None
        if payload.points:
            point_coords = np.array([[point.x, point.y] for point in payload.points], dtype=np.float32)
            point_labels = np.array(
                [1 if point.label == PointLabel.POSITIVE else 0 for point in payload.points],
                dtype=np.int32,
            )

        selected_box = payload.boxes[-1] if payload.boxes else None
        box = None
        warnings: list[str] = []
        if len(payload.boxes) > 1:
            warnings.append("Interactive SAM3 currently uses the most recent box prompt only.")
        if selected_box is not None:
            box = np.array([selected_box.x0, selected_box.y0, selected_box.x1, selected_box.y1], dtype=np.float32)

        if point_coords is None and box is None:
            raise ValueError("Provide at least one point or a box for interactive segmentation.")

        masks, scores, _ = self._model.predict_inst(
            self._clone_state(base_state),
            point_coords=point_coords,
            point_labels=point_labels,
            box=box,
            multimask_output=payload.multimask_output,
        )

        return masks, scores, warnings

    def _run_text_guided_interactive_segmentation(
        self,
        session: SessionAsset,
        payload: SegmentationRequest,
        base_state: dict[str, Any],
    ) -> SegmentationResult:
        interactive_masks, interactive_scores, warnings = self._predict_interactive_masks(
            payload,
            base_state,
        )
        text_masks, _, _, text_warnings = self._predict_text_grounding_outputs(
            session,
            payload,
            base_state,
        )
        warnings.extend(text_warnings)

        filtered_masks, filtered_scores, text_guidance_warnings = self._apply_text_guidance(
            interactive_masks,
            interactive_scores,
            text_masks,
        )
        warnings.extend(text_guidance_warnings)

        candidates = self._build_candidates(masks=filtered_masks, scores=filtered_scores)
        if not candidates:
            warnings.append("SAM3 returned no masks for the current combined prompts.")

        return SegmentationResult(
            session_id=session.session_id,
            mode=payload.interaction_mode,
            source="interactive",
            text_prompt_supported=True,
            supports_point_refinement=True,
            supports_box_refinement=True,
            warnings=warnings,
            candidates=candidates,
        )

    def _run_text_grounding(
        self,
        session: SessionAsset,
        payload: SegmentationRequest,
        base_state: dict[str, Any],
    ) -> SegmentationResult:
        masks, scores, boxes, warnings = self._predict_text_grounding_outputs(
            session,
            payload,
            base_state,
        )

        candidates = self._build_candidates(masks=masks, scores=scores, boxes=boxes)
        if not candidates:
            warnings.append("SAM3 returned no text-grounded masks for the current prompt.")

        return SegmentationResult(
            session_id=session.session_id,
            mode=payload.interaction_mode,
            source="text_grounding",
            text_prompt_supported=True,
            supports_point_refinement=False,
            supports_box_refinement=True,
            warnings=warnings,
            candidates=candidates,
        )

    def _predict_text_grounding_outputs(
        self,
        session: SessionAsset,
        payload: SegmentationRequest,
        base_state: dict[str, Any],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
        if self._processor is None:
            raise RuntimeError("SAM3 processor is not initialized.")
        text_prompt = (payload.text_prompt or "").strip()
        if not text_prompt:
            raise ValueError("Text prompting requires a non-empty text prompt.")

        working_state = self._clone_state(base_state)
        working_state = self._run_text_grounding_prompt(text_prompt, working_state)

        for box in payload.boxes:
            normalized = self._to_normalized_cxcywh(box, session.width, session.height)
            working_state = self._processor.add_geometric_prompt(
                normalized,
                label=box.label == PointLabel.POSITIVE,
                state=working_state,
            )

        masks_tensor = working_state.get("masks")
        scores_tensor = working_state.get("scores")
        boxes_tensor = working_state.get("boxes")

        masks = self._to_numpy(masks_tensor)
        scores = self._to_numpy(scores_tensor)
        boxes = self._to_numpy(boxes_tensor)

        return masks, scores, boxes, []

    def _apply_text_guidance(
        self,
        interactive_masks: Any,
        interactive_scores: Any,
        text_masks: Any,
    ) -> tuple[np.ndarray, np.ndarray, list[str]]:
        interactive_mask_array = self._normalize_masks(interactive_masks)
        interactive_score_array = np.atleast_1d(self._to_numpy(interactive_scores).astype(np.float32))
        text_mask_array = self._normalize_masks(text_masks)
        warnings: list[str] = []

        if interactive_mask_array.size == 0:
            return interactive_mask_array, interactive_score_array, warnings
        if text_mask_array.size == 0:
            warnings.append(
                "Text grounding produced no masks, so interactive prompts were returned unchanged."
            )
            return interactive_mask_array, interactive_score_array, warnings

        text_union = np.any(text_mask_array > 0.5, axis=0)
        overlap_ratios = np.zeros(interactive_mask_array.shape[0], dtype=np.float32)
        for index, mask in enumerate(interactive_mask_array):
            mask_bool = mask > 0.5
            mask_area = float(mask_bool.sum())
            if mask_area == 0:
                continue
            overlap = float(np.logical_and(mask_bool, text_union).sum())
            overlap_ratios[index] = overlap / mask_area

        overlapping_indices = np.flatnonzero(overlap_ratios > 0.0)
        if len(overlapping_indices) == 0:
            warnings.append(
                "Text prompt did not overlap the interactive masks, so interactive results were returned unchanged."
            )
            return interactive_mask_array, interactive_score_array, warnings

        ranking = overlap_ratios[overlapping_indices] * 1000.0 + interactive_score_array[overlapping_indices]
        sorted_indices = overlapping_indices[np.argsort(-ranking, kind="stable")]
        warnings.append(
            "Text prompt was used as an additional grounding prior over the interactive prompts."
        )
        return (
            interactive_mask_array[sorted_indices],
            interactive_score_array[sorted_indices],
            warnings,
        )

    def _run_text_grounding_prompt(
        self,
        prompt: str,
        state: dict[str, Any],
    ) -> dict[str, Any]:
        if self._processor is None or self._model is None or self._device is None:
            raise RuntimeError("SAM3 text grounding is not initialized.")
        if "backbone_out" not in state:
            raise ValueError("You must call set_image before text grounding.")

        text_outputs = self._model.backbone.forward_text([prompt], device=self._device)
        state["backbone_out"].update(self._align_model_dtype(text_outputs))

        if "geometric_prompt" not in state:
            state["geometric_prompt"] = self._model._get_dummy_prompt()

        return self._processor._forward_grounding(state)

    def _build_candidates(
        self,
        masks: Any,
        scores: Any,
        boxes: Any | None = None,
    ) -> list[MaskCandidate]:
        mask_array = self._normalize_masks(masks)
        if mask_array.size == 0:
            return []

        raw_score_array = np.atleast_1d(self._to_numpy(scores).astype(np.float32))
        box_array = None if boxes is None else np.atleast_2d(self._to_numpy(boxes).astype(np.float32))

        ranking_scores = np.full(mask_array.shape[0], -np.inf, dtype=np.float32)
        score_count = min(len(raw_score_array), len(ranking_scores))
        if score_count > 0:
            ranking_scores[:score_count] = raw_score_array[:score_count]

        sorted_indices = np.argsort(-ranking_scores, kind="stable")
        kept_indices: list[int] = []
        kept_masks: list[np.ndarray] = []

        for candidate_index in sorted_indices:
            candidate_mask = mask_array[candidate_index] > 0.5
            if any(self._compute_mask_iou(candidate_mask, existing_mask) > 0.8 for existing_mask in kept_masks):
                continue
            kept_indices.append(int(candidate_index))
            kept_masks.append(candidate_mask)

        candidates: list[MaskCandidate] = []
        for index, source_index in enumerate(kept_indices):
            mask_bool = kept_masks[index]
            color = MASK_PALETTE[index % len(MASK_PALETTE)]
            image_data_url = self._mask_to_data_url(mask_bool, color)
            score = float(raw_score_array[source_index]) if source_index < len(raw_score_array) else None
            centroid = self._compute_mask_centroid(mask_bool)
            bounding_box = (
                box_array[source_index].tolist()
                if box_array is not None and source_index < len(box_array)
                else self._compute_mask_bounding_box(mask_bool)
            )

            candidates.append(
                MaskCandidate(
                    candidate_id=f"mask-{index + 1}",
                    score=score,
                    width=int(mask_bool.shape[1]),
                    height=int(mask_bool.shape[0]),
                    area=int(mask_bool.sum()),
                    image_data_url=image_data_url,
                    color_rgb=list(color),
                    centroid=centroid,
                    bounding_box=bounding_box,
                )
            )

        return candidates

    def _compute_mask_iou(self, left_mask: np.ndarray, right_mask: np.ndarray) -> float:
        intersection = float(np.logical_and(left_mask, right_mask).sum())
        if intersection == 0.0:
            return 0.0
        union = float(np.logical_or(left_mask, right_mask).sum())
        if union == 0.0:
            return 0.0
        return intersection / union

    def _compute_mask_bounding_box(self, mask: np.ndarray) -> list[float] | None:
        y_coords, x_coords = np.nonzero(mask)
        if len(x_coords) == 0 or len(y_coords) == 0:
            return None
        return [
            float(x_coords.min()),
            float(y_coords.min()),
            float(x_coords.max() + 1),
            float(y_coords.max() + 1),
        ]

    def _compute_mask_centroid(self, mask: np.ndarray) -> list[float] | None:
        y_coords, x_coords = np.nonzero(mask)
        if len(x_coords) == 0 or len(y_coords) == 0:
            return None
        return [float(x_coords.mean()), float(y_coords.mean())]

    def _normalize_masks(self, masks: Any) -> np.ndarray:
        array = self._to_numpy(masks).astype(np.float32)
        if array.size == 0:
            return np.empty((0, 0, 0), dtype=np.float32)
        if array.ndim == 2:
            return array[np.newaxis, ...]
        if array.ndim == 4 and array.shape[1] == 1:
            return array[:, 0, ...]
        return array

    def _clone_state(self, base_state: dict[str, Any]) -> dict[str, Any]:
        cloned = dict(base_state)
        if "backbone_out" in base_state:
            cloned["backbone_out"] = dict(base_state["backbone_out"])
        return cloned

    def _align_model_dtype(self, values: dict[str, Any]) -> dict[str, Any]:
        if self._model is None:
            return values

        model_dtype = next(self._model.parameters()).dtype
        aligned: dict[str, Any] = {}
        for key, value in values.items():
            if hasattr(value, "is_floating_point") and value.is_floating_point():
                aligned[key] = value.to(dtype=model_dtype)
            else:
                aligned[key] = value
        return aligned

    def _inference_context(self):
        if self._device and self._device.startswith("cuda"):
            import torch

            return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        return nullcontext()

    def _to_numpy(self, value: Any) -> np.ndarray:
        if value is None:
            return np.array([])
        if isinstance(value, np.ndarray):
            return value

        if hasattr(value, "detach"):
            tensor = value.detach().cpu()
            if str(getattr(tensor, "dtype", "")) == "torch.bfloat16":
                tensor = tensor.float()
            return tensor.numpy()
        return np.asarray(value)

    def _mask_to_data_url(self, mask: np.ndarray, color: tuple[int, int, int]) -> str:
        rgba = np.zeros((mask.shape[0], mask.shape[1], 4), dtype=np.uint8)
        rgba[..., 0] = color[0]
        rgba[..., 1] = color[1]
        rgba[..., 2] = color[2]
        rgba[..., 3] = mask.astype(np.uint8) * 255

        buffer = BytesIO()
        Image.fromarray(rgba, mode="RGBA").save(buffer, format="PNG")
        encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
        return f"data:image/png;base64,{encoded}"

    def _to_normalized_cxcywh(self, box: Any, width: int, height: int) -> list[float]:
        center_x = ((box.x0 + box.x1) / 2.0) / width
        center_y = ((box.y0 + box.y1) / 2.0) / height
        box_width = (box.x1 - box.x0) / width
        box_height = (box.y1 - box.y0) / height
        return [center_x, center_y, box_width, box_height]

    def _get_session(self, session_id: str) -> SessionAsset:
        with self._sessions_lock:
            asset = self._sessions.get(session_id)
        if asset is None:
            raise ValueError(f"Unknown session_id: {session_id}")
        return asset
