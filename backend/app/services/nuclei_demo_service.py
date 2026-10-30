from __future__ import annotations

import asyncio
import base64
import colorsys
import json
import time
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from threading import Lock
from typing import Any

import numpy as np
from PIL import Image

from app.core.config import Settings
from app.models.schemas import (
    InteractionMode,
    JobStatus,
    MaskCandidate,
    NucleiImageItem,
    NucleiImageListResponse,
    NucleiInstance,
    NucleiModelKey,
    NucleiModelOption,
    NucleiOverlays,
    NucleiPredictionResponse,
    SegmentationRequest,
)
from app.services.bbbc038_data import (
    MODELS_DIR,
    TEST_DIR,
    ImageType,
    image_path,
    load_ground_truth_labels,
    load_split,
    masks_to_label_map,
)
from app.services.job_queue import InferenceQueue, ModelWorker
from app.services.nuclei_metrics import classify_matches, compute_image_metrics, iou_matrix
from app.services.sam3_runtime import SAM3ModelRuntime, release_cuda_cache
from app.services.sam3_service import SAM3Service

DEFAULT_PROMPT = "blobs"
THUMBNAIL_SIZE = 192

TRUE_POSITIVE_RGB = (34, 197, 94)
FALSE_POSITIVE_RGB = (239, 68, 68)
FALSE_NEGATIVE_RGB = (250, 204, 21)
GROUND_TRUTH_OUTLINE_RGB = (255, 255, 255)
PREDICTION_OUTLINE_RGB = (33, 211, 197)

MODEL_LABELS: dict[NucleiModelKey, tuple[str, str]] = {
    NucleiModelKey.TRAINED: (
        "Trained on pseudo-labels",
        "SAM3 + LoRA fine-tuned only on SAM3 'blobs' pseudo-labels.",
    ),
    NucleiModelKey.ZERO_SHOT: (
        "Zero-shot SAM3",
        "The base SAM3 checkpoint with the same text prompt.",
    ),
    NucleiModelKey.GROUND_TRUTH_TRAINED: (
        "Trained on ground truth",
        "Same SAM3 + LoRA setup trained on stage1_train ground truth (upper bound).",
    ),
}

LORA_RUN_DIRS: dict[NucleiModelKey, str] = {
    NucleiModelKey.TRAINED: "sam3_lora_pseudo",
    NucleiModelKey.GROUND_TRUTH_TRAINED: "sam3_lora_gt",
}


class NucleiDemoError(RuntimeError):
    pass


class NucleiModelUnavailableError(NucleiDemoError):
    pass


class NucleiLoRARuntime(SAM3ModelRuntime):
    """SAM3 runtime that serves a LoRA fine-tuned checkpoint.

    Only model construction differs from the sandbox runtime; embedding caching, text
    grounding and mask packaging are inherited. The model loads on the first job.
    """

    def __init__(
        self, settings: Settings, sessions: SAM3Service, lora_checkpoint: Path
    ) -> None:
        super().__init__(settings, sessions)
        self._lora_checkpoint = lora_checkpoint

    def _ensure_model_loaded(self) -> None:
        if self._model is not None and self._processor is not None:
            return

        with self._model_lock:
            if self._model is not None and self._processor is not None:
                return

            import torch
            from sam3.model.sam3_image_processor import Sam3Processor

            from app.services.sam3_lora import build_lora_sam3_model

            requested_device = self._settings.sam3_device.lower()
            if requested_device == "auto":
                self._device = "cuda" if torch.cuda.is_available() else "cpu"
            else:
                self._device = requested_device

            self._model, _ = build_lora_sam3_model(
                self._lora_checkpoint,
                checkpoint_path=self._settings.sam3_checkpoint_path,
                bpe_path=self._settings.sam3_bpe_path,
                load_from_hf=self._settings.sam3_load_from_hf,
                device=self._device,
            )
            self._processor = Sam3Processor(
                self._model,
                device=self._device,
                confidence_threshold=self._settings.sam3_confidence_threshold,
            )


@dataclass(slots=True)
class _MetricsCache:
    mtime: float
    payload: dict[str, Any]


class NucleiDemoService:
    """BBBC038 stage1_test browser, precomputed metrics and live inference."""

    def __init__(
        self,
        settings: Settings,
        sam3_service: SAM3Service,
        zero_shot_queue: InferenceQueue,
        *,
        models_dir: Path = MODELS_DIR,
        test_dir: Path = TEST_DIR,
    ) -> None:
        self._settings = settings
        self._sam3_service = sam3_service
        self._zero_shot_queue = zero_shot_queue
        self._models_dir = models_dir
        self._test_dir = test_dir
        # Each fine-tuned model's queue remembers the checkpoint mtime it was built from.
        self._lora_queues: dict[NucleiModelKey, tuple[float, InferenceQueue]] = {}
        self._queue_lock = asyncio.Lock()
        self._cache_lock = Lock()
        self._metrics_cache: _MetricsCache | None = None
        self._image_items: list[NucleiImageItem] | None = None
        self._thumbnails: dict[str, bytes] = {}
        self._ground_truth: dict[str, np.ndarray] = {}

    async def stop(self) -> None:
        for _, queue in self._lora_queues.values():
            await queue.stop()
        self._lora_queues.clear()

    # Dataset browsing -------------------------------------------------------------

    def list_images(self) -> NucleiImageListResponse:
        items = self._get_image_items()
        counts = {image_type.value: 0 for image_type in ImageType}
        for item in items:
            counts[item.image_type] = counts.get(item.image_type, 0) + 1
        return NucleiImageListResponse(
            images=items,
            image_type_counts=counts,
            models=[
                NucleiModelOption(
                    model_key=model_key,
                    label=label,
                    description=description,
                    available=self._is_model_available(model_key),
                )
                for model_key, (label, description) in MODEL_LABELS.items()
            ],
            prompt=self._prompt(),
        )

    def get_image_path(self, image_id: str) -> Path:
        self._require_image(image_id)
        return image_path(self._test_dir, image_id)

    def get_thumbnail(self, image_id: str) -> bytes:
        self._require_image(image_id)
        with self._cache_lock:
            cached = self._thumbnails.get(image_id)
        if cached is not None:
            return cached

        with Image.open(image_path(self._test_dir, image_id)) as image:
            thumbnail = image.convert("RGB")
            thumbnail.thumbnail((THUMBNAIL_SIZE, THUMBNAIL_SIZE))
            buffer = BytesIO()
            thumbnail.save(buffer, format="PNG")
        data = buffer.getvalue()
        with self._cache_lock:
            self._thumbnails[image_id] = data
        return data

    # Metrics ----------------------------------------------------------------------

    def get_metrics_summary(self) -> dict[str, Any]:
        payload = self._load_metrics()
        return {key: value for key, value in payload.items() if key != "per_image"}

    def get_image_metrics(self, image_id: str) -> dict[str, Any]:
        self._require_image(image_id)
        record = self._load_metrics().get("per_image", {}).get(image_id)
        if record is None:
            raise NucleiDemoError(f"No precomputed metrics for image {image_id}.")
        return {"image_id": image_id, **record}

    # Inference --------------------------------------------------------------------

    async def predict(
        self, image_id: str, model_key: NucleiModelKey
    ) -> NucleiPredictionResponse:
        self._require_image(image_id)
        if not self._is_model_available(model_key):
            raise NucleiModelUnavailableError(
                f"{MODEL_LABELS[model_key][0]} has no trained checkpoint yet."
            )

        started = time.perf_counter()
        source_path = image_path(self._test_dir, image_id)
        session = await asyncio.to_thread(
            self._sam3_service.create_session,
            f"{image_id}.png",
            "image/png",
            source_path.read_bytes(),
            None,
            f"bbbc038:{image_id}",
        )
        queue = (
            self._zero_shot_queue
            if model_key == NucleiModelKey.ZERO_SHOT
            else await self._get_lora_queue(model_key)
        )
        job = await queue.submit_and_wait(
            SegmentationRequest(
                session_id=session.session_id,
                interaction_mode=InteractionMode.TEXT,
                text_prompt=self._prompt(),
            ),
            timeout_seconds=self._settings.sam3_request_timeout_seconds,
        )
        if job.status != JobStatus.COMPLETE or job.result is None:
            raise NucleiDemoError(job.error or "Nuclei inference failed.")

        return await asyncio.to_thread(
            self._build_prediction_response,
            image_id,
            model_key,
            job.job_id,
            job.result.candidates,
            job.result.warnings,
            started,
        )

    async def _get_lora_queue(self, model_key: NucleiModelKey) -> InferenceQueue:
        checkpoint = self._lora_checkpoint(model_key)
        mtime = checkpoint.stat().st_mtime
        async with self._queue_lock:
            existing = self._lora_queues.get(model_key)
            if existing is not None and existing[0] == mtime:
                return existing[1]
            if existing is not None:
                # A retrained checkpoint replaced the one this worker loaded.
                await existing[1].stop()
                del self._lora_queues[model_key]
                existing = None
                release_cuda_cache()

            runtime = NucleiLoRARuntime(self._settings, self._sam3_service, checkpoint)
            queue = InferenceQueue(
                workers=[ModelWorker(worker_id=1, runtime=runtime)],
                max_queue_size=self._settings.sam3_queue_max_size,
            )
            await queue.start()
            self._lora_queues[model_key] = (mtime, queue)
            return queue

    def _build_prediction_response(
        self,
        image_id: str,
        model_key: NucleiModelKey,
        job_id: str,
        candidates: list[MaskCandidate],
        warnings: list[str],
        started: float,
    ) -> NucleiPredictionResponse:
        gt_labels = self._get_ground_truth(image_id)
        masks = [decode_mask_data_url(candidate.image_data_url) for candidate in candidates]
        scores = [
            candidate.score if candidate.score is not None else 0.0
            for candidate in candidates
        ]
        pred_labels = masks_to_label_map(masks, gt_labels.shape, scores)
        # Painting can hide instances under higher-scoring ones; keep scores aligned to
        # the surviving labels.
        label_scores = label_scores_after_painting(masks, scores, pred_labels)

        metrics = compute_image_metrics(gt_labels, pred_labels)
        matches = classify_matches(gt_labels, pred_labels)
        iou = iou_matrix(gt_labels, pred_labels)

        predictions = [
            NucleiInstance(
                instance_id=pred_id,
                status="tp" if pred_id in matches.matched_gt_for_pred else "fp",
                area=area,
                bounding_box=box,
                score=label_scores.get(pred_id),
                matched_id=matches.matched_gt_for_pred.get(pred_id),
                iou=(
                    float(iou[matches.matched_gt_for_pred[pred_id] - 1, pred_id - 1])
                    if pred_id in matches.matched_gt_for_pred
                    else None
                ),
            )
            for pred_id, area, box in instance_geometry(pred_labels)
        ]
        pred_for_gt = {gt_id: pred_id for pred_id, gt_id in matches.matched_gt_for_pred.items()}
        missed = set(matches.false_negative_gt)
        ground_truth = [
            NucleiInstance(
                instance_id=gt_id,
                status="fn" if gt_id in missed else "matched",
                area=area,
                bounding_box=box,
                matched_id=pred_for_gt.get(gt_id),
            )
            for gt_id, area, box in instance_geometry(gt_labels)
        ]

        height, width = gt_labels.shape
        return NucleiPredictionResponse(
            image_id=image_id,
            image_type=self._image_types().get(image_id, "unknown"),
            width=width,
            height=height,
            model_key=model_key,
            model_label=MODEL_LABELS[model_key][0],
            prompt=self._prompt(),
            job_id=job_id,
            elapsed_seconds=time.perf_counter() - started,
            metrics=metrics.to_dict(),
            counts={
                "tp": len(matches.true_positive_pred),
                "fp": len(matches.false_positive_pred),
                "fn": len(matches.false_negative_gt),
                "pred": metrics.pred_count,
                "gt": metrics.gt_count,
            },
            overlays=NucleiOverlays(
                prediction=rgba_to_data_url(render_instances(pred_labels)),
                ground_truth=rgba_to_data_url(render_instances(gt_labels)),
                ground_truth_outline=rgba_to_data_url(
                    render_outline(gt_labels, GROUND_TRUTH_OUTLINE_RGB)
                ),
                prediction_outline=rgba_to_data_url(
                    render_outline(pred_labels, PREDICTION_OUTLINE_RGB)
                ),
                comparison=rgba_to_data_url(
                    render_comparison(
                        gt_labels,
                        pred_labels,
                        set(matches.true_positive_pred),
                        missed,
                    )
                ),
            ),
            predictions=predictions,
            ground_truth=ground_truth,
            warnings=warnings,
        )

    # Helpers ----------------------------------------------------------------------

    def _lora_checkpoint(self, model_key: NucleiModelKey) -> Path:
        return self._models_dir / LORA_RUN_DIRS[model_key] / "checkpoints" / "best_lora.pt"

    def _is_model_available(self, model_key: NucleiModelKey) -> bool:
        if model_key == NucleiModelKey.ZERO_SHOT:
            return True
        return self._lora_checkpoint(model_key).is_file()

    def _load_metrics(self) -> dict[str, Any]:
        metrics_path = self._models_dir / "metrics.json"
        if not metrics_path.is_file():
            raise NucleiDemoError(
                "metrics.json has not been generated yet. Run scripts/evaluate_bbbc038.py."
            )
        mtime = metrics_path.stat().st_mtime
        with self._cache_lock:
            if self._metrics_cache is not None and self._metrics_cache.mtime == mtime:
                return self._metrics_cache.payload
        payload = json.loads(metrics_path.read_text())
        with self._cache_lock:
            self._metrics_cache = _MetricsCache(mtime=mtime, payload=payload)
        return payload

    def _prompt(self) -> str:
        try:
            return self._load_metrics().get("prompt", DEFAULT_PROMPT)
        except NucleiDemoError:
            return DEFAULT_PROMPT

    def _image_types(self) -> dict[str, str]:
        return {item.image_id: item.image_type for item in self._get_image_items()}

    def _get_image_items(self) -> list[NucleiImageItem]:
        with self._cache_lock:
            if self._image_items is not None:
                return self._image_items

        split = load_split(self._test_dir.parents[1] / "split.json")
        prefix = f"{self._settings.api_prefix}/nuclei-demo/images"
        items = []
        for image_id in split["test"]:
            with Image.open(image_path(self._test_dir, image_id)) as image:
                width, height = image.size
            items.append(
                NucleiImageItem(
                    image_id=image_id,
                    image_type=split["image_types"][image_id],
                    width=width,
                    height=height,
                    gt_count=len(list((self._test_dir / image_id / "masks").glob("*.png"))),
                    thumbnail_url=f"{prefix}/{image_id}/thumbnail",
                    image_url=f"{prefix}/{image_id}/image",
                )
            )
        with self._cache_lock:
            self._image_items = items
        return items

    def _require_image(self, image_id: str) -> None:
        if image_id not in self._image_types():
            raise FileNotFoundError(f"Unknown stage1_test image: {image_id}")

    def _get_ground_truth(self, image_id: str) -> np.ndarray:
        with self._cache_lock:
            cached = self._ground_truth.get(image_id)
        if cached is not None:
            return cached
        labels = load_ground_truth_labels(self._test_dir, image_id)
        with self._cache_lock:
            self._ground_truth[image_id] = labels
        return labels


def decode_mask_data_url(data_url: str) -> np.ndarray:
    _, encoded = data_url.split(",", 1)
    with Image.open(BytesIO(base64.b64decode(encoded))) as image:
        return np.asarray(image.convert("RGBA"))[..., 3] > 0


def label_scores_after_painting(
    masks: list[np.ndarray], scores: list[float], labels: np.ndarray
) -> dict[int, float]:
    result: dict[int, float] = {}
    for mask, score in zip(masks, scores):
        values = labels[mask]
        values = values[values > 0]
        if values.size == 0:
            continue
        label = int(np.bincount(values).argmax())
        result[label] = max(score, result.get(label, score))
    return result


def instance_geometry(labels: np.ndarray) -> list[tuple[int, int, list[float]]]:
    from scipy import ndimage

    count = int(labels.max(initial=0))
    if count == 0:
        return []
    areas = np.bincount(labels.ravel(), minlength=count + 1)
    geometry = []
    for index, slices in enumerate(ndimage.find_objects(labels), start=1):
        if slices is None:
            continue
        rows, cols = slices
        geometry.append(
            (
                index,
                int(areas[index]),
                [float(cols.start), float(rows.start), float(cols.stop), float(rows.stop)],
            )
        )
    return geometry


def instance_palette(count: int) -> np.ndarray:
    palette = np.zeros((count + 1, 3), dtype=np.uint8)
    for index in range(1, count + 1):
        hue = (index * 0.618033988749895) % 1.0
        red, green, blue = colorsys.hsv_to_rgb(hue, 0.65, 0.95)
        palette[index] = (int(red * 255), int(green * 255), int(blue * 255))
    return palette


def instance_boundaries(labels: np.ndarray) -> np.ndarray:
    padded = np.pad(labels, 1, mode="edge")
    center = padded[1:-1, 1:-1]
    differs = (
        (center != padded[:-2, 1:-1])
        | (center != padded[2:, 1:-1])
        | (center != padded[1:-1, :-2])
        | (center != padded[1:-1, 2:])
    )
    return differs & (labels > 0)


def render_instances(labels: np.ndarray) -> np.ndarray:
    palette = instance_palette(int(labels.max(initial=0)))
    rgba = np.zeros((*labels.shape, 4), dtype=np.uint8)
    foreground = labels > 0
    rgba[..., :3] = palette[labels]
    rgba[..., 3] = np.where(foreground, 170, 0)
    rgba[instance_boundaries(labels), 3] = 255
    return rgba


def render_outline(labels: np.ndarray, color: tuple[int, int, int]) -> np.ndarray:
    rgba = np.zeros((*labels.shape, 4), dtype=np.uint8)
    boundary = instance_boundaries(labels)
    rgba[boundary] = (*color, 255)
    return rgba


def render_comparison(
    gt_labels: np.ndarray,
    pred_labels: np.ndarray,
    true_positive_pred: set[int],
    false_negative_gt: set[int],
) -> np.ndarray:
    rgba = np.zeros((*gt_labels.shape, 4), dtype=np.uint8)
    missed = np.isin(gt_labels, list(false_negative_gt)) if false_negative_gt else None
    if missed is not None:
        rgba[missed] = (*FALSE_NEGATIVE_RGB, 170)
    predicted = pred_labels > 0
    true_positive = np.isin(pred_labels, list(true_positive_pred)) & predicted
    rgba[true_positive] = (*TRUE_POSITIVE_RGB, 170)
    rgba[predicted & ~true_positive] = (*FALSE_POSITIVE_RGB, 170)
    rgba[instance_boundaries(pred_labels), 3] = 255
    if missed is not None:
        rgba[instance_boundaries(np.where(missed, gt_labels, 0)), 3] = 255
    return rgba


def rgba_to_data_url(rgba: np.ndarray) -> str:
    buffer = BytesIO()
    Image.fromarray(rgba).save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")
