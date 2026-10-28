from __future__ import annotations

import base64
import itertools
from dataclasses import dataclass
from datetime import datetime, timezone
from io import BytesIO

import numpy as np
from PIL import Image

from app.models.schemas import (
    ChallengeScoreRequest,
    ChallengeScoreSummary,
    ChallengeType,
)
from app.services.challenge_service import ChallengeService


@dataclass(frozen=True, slots=True)
class BinaryMetrics:
    iou: float
    dice: float
    pixel_accuracy: float


class ScoringService:
    def __init__(self, challenge_service: ChallengeService) -> None:
        self._challenge_service = challenge_service

    def score_submission(
        self,
        challenge_id: ChallengeType,
        request: ChallengeScoreRequest,
    ) -> ChallengeScoreSummary:
        submissions_by_image = self._index_submissions(challenge_id, request)
        image_records = self._challenge_service.get_image_records(challenge_id)
        dataset_size = len(image_records)

        full_iou: list[float] = []
        full_dice: list[float] = []
        full_pixel_accuracy: list[float] = []
        labeled_iou: list[float] = []
        labeled_dice: list[float] = []
        labeled_pixel_accuracy: list[float] = []
        warnings: list[str] = []
        rescaled_image_ids: set[str] = set()
        images_labeled = 0

        for record in image_records:
            submission = submissions_by_image.get(record.image_id)
            if submission is None:
                full_iou.append(0.0)
                full_dice.append(0.0)
                full_pixel_accuracy.append(0.0)
                continue

            # Score on the image's own pixel grid rather than on the first ground-truth
            # mask: an image can legitimately have no labelled classes at all.
            reference_shape = self._challenge_service.get_image_shape(
                challenge_id, record.image_id
            )
            ground_truth_masks = [
                self._resize_mask(mask, reference_shape)
                for mask in self._challenge_service.load_ground_truth_masks(
                    challenge_id,
                    record.image_id,
                )
            ]
            predicted_masks = []
            for layer in submission.masks:
                decoded = self._decode_mask(layer.image_data_url)
                if decoded.shape != reference_shape:
                    rescaled_image_ids.add(record.image_id)
                predicted_masks.append(self._resize_mask(decoded, reference_shape))

            if challenge_id == ChallengeType.SINGLE_CLASS:
                ground_truth = (
                    ground_truth_masks[0]
                    if ground_truth_masks
                    else np.zeros(reference_shape, dtype=bool)
                )
                prediction = np.zeros(reference_shape, dtype=bool)
                for mask in predicted_masks:
                    prediction |= mask
                metrics = self._compute_binary_metrics(prediction, ground_truth)
            else:
                metrics = self._score_multi_class(
                    predicted_masks, ground_truth_masks, reference_shape
                )

            images_labeled += 1
            full_iou.append(metrics.iou)
            full_dice.append(metrics.dice)
            full_pixel_accuracy.append(metrics.pixel_accuracy)
            labeled_iou.append(metrics.iou)
            labeled_dice.append(metrics.dice)
            labeled_pixel_accuracy.append(metrics.pixel_accuracy)

        missing_count = dataset_size - images_labeled
        if missing_count > 0:
            warnings.append(
                f"{missing_count} image(s) were not stored before scoring and were counted as zero-score images."
            )
        if rescaled_image_ids:
            # Silently rescaling would make a client-side sizing bug look like sloppy
            # annotation, so say so instead.
            warnings.append(
                f"{len(rescaled_image_ids)} image(s) were submitted at a different resolution "
                "than the source image and were rescaled before scoring: "
                f"{', '.join(sorted(rescaled_image_ids))}."
            )

        mean_iou = float(np.mean(full_iou)) if full_iou else 0.0
        mean_dice = float(np.mean(full_dice)) if full_dice else 0.0
        pixel_accuracy = (
            float(np.mean(full_pixel_accuracy)) if full_pixel_accuracy else 0.0
        )
        labeled_mean_iou = float(np.mean(labeled_iou)) if labeled_iou else 0.0
        labeled_mean_dice = float(np.mean(labeled_dice)) if labeled_dice else 0.0
        labeled_pixel = (
            float(np.mean(labeled_pixel_accuracy)) if labeled_pixel_accuracy else 0.0
        )
        percent_completed = images_labeled / dataset_size if dataset_size else 0.0
        overall_score = self._compute_overall_score(
            mean_iou=mean_iou,
            mean_dice=mean_dice,
            pixel_accuracy=pixel_accuracy,
            percent_completed=percent_completed,
        )

        return ChallengeScoreSummary(
            challenge_id=challenge_id,
            user_name=request.user_name.strip(),
            dataset_size=dataset_size,
            images_labeled=images_labeled,
            percent_completed=percent_completed,
            mean_iou=mean_iou,
            mean_dice=mean_dice,
            pixel_accuracy=pixel_accuracy,
            labeled_mean_iou=labeled_mean_iou,
            labeled_mean_dice=labeled_mean_dice,
            labeled_pixel_accuracy=labeled_pixel,
            overall_score=overall_score,
            submitted_at=datetime.now(timezone.utc),
            warnings=warnings,
        )

    def _index_submissions(
        self,
        challenge_id: ChallengeType,
        request: ChallengeScoreRequest,
    ) -> dict[str, object]:
        valid_image_ids = {
            record.image_id
            for record in self._challenge_service.get_image_records(challenge_id)
        }
        submissions_by_image: dict[str, object] = {}
        for submission in request.submissions:
            if submission.image_id not in valid_image_ids:
                raise ValueError(
                    f"Unknown image_id in score submission: {submission.image_id}"
                )
            if submission.image_id in submissions_by_image:
                raise ValueError(
                    f"Duplicate submission for image_id: {submission.image_id}"
                )
            submissions_by_image[submission.image_id] = submission
        return submissions_by_image

    def _score_multi_class(
        self,
        predicted_masks: list[np.ndarray],
        ground_truth_masks: list[np.ndarray],
        reference_shape: tuple[int, int],
    ) -> BinaryMetrics:
        if not ground_truth_masks:
            # Some label maps are genuinely empty. Score them like the single-class path:
            # an empty submission is correct, anything drawn is a false positive.
            empty = np.zeros(reference_shape, dtype=bool)
            prediction = np.zeros(reference_shape, dtype=bool)
            for mask in predicted_masks:
                prediction |= mask
            return self._compute_binary_metrics(prediction, empty)

        matched_predictions, unmatched_predictions = self._match_predictions(
            predicted_masks, ground_truth_masks
        )
        per_class_metrics = [
            self._compute_binary_metrics(prediction, ground_truth)
            for prediction, ground_truth in zip(
                matched_predictions, ground_truth_masks, strict=False
            )
        ]

        gt_label_map = self._build_label_map(ground_truth_masks)
        matched_prediction_map = self._build_label_map(matched_predictions)
        if unmatched_predictions:
            extra_mask = np.zeros_like(gt_label_map, dtype=bool)
            for prediction in unmatched_predictions:
                extra_mask |= prediction
            matched_prediction_map[extra_mask] = len(ground_truth_masks) + 1

        return BinaryMetrics(
            iou=(
                float(np.mean([metrics.iou for metrics in per_class_metrics]))
                if per_class_metrics
                else 0.0
            ),
            dice=(
                float(np.mean([metrics.dice for metrics in per_class_metrics]))
                if per_class_metrics
                else 0.0
            ),
            pixel_accuracy=float(np.mean(gt_label_map == matched_prediction_map)),
        )

    def _match_predictions(
        self,
        predicted_masks: list[np.ndarray],
        ground_truth_masks: list[np.ndarray],
    ) -> tuple[list[np.ndarray], list[np.ndarray]]:
        if not predicted_masks:
            return [np.zeros_like(mask, dtype=bool) for mask in ground_truth_masks], []

        # The multi-class workshop data behaves like instance masks, so we greedily
        # match submitted layers to GT objects by Dice score and treat unmatched GT
        # objects as misses while leaving extra predicted layers as false positives.
        candidate_pairs: list[tuple[float, int, int]] = []
        for prediction_index, prediction in enumerate(predicted_masks):
            for gt_index, ground_truth in enumerate(ground_truth_masks):
                dice = self._compute_binary_metrics(prediction, ground_truth).dice
                candidate_pairs.append((dice, prediction_index, gt_index))

        candidate_pairs.sort(reverse=True)
        matched_predictions: list[np.ndarray] = [
            np.zeros_like(mask, dtype=bool) for mask in ground_truth_masks
        ]
        used_predictions: set[int] = set()
        used_ground_truth: set[int] = set()

        for score, prediction_index, gt_index in candidate_pairs:
            if score <= 0:
                break
            if prediction_index in used_predictions or gt_index in used_ground_truth:
                continue
            matched_predictions[gt_index] = predicted_masks[prediction_index]
            used_predictions.add(prediction_index)
            used_ground_truth.add(gt_index)

        unmatched_predictions = [
            predicted_masks[index]
            for index in range(len(predicted_masks))
            if index not in used_predictions
        ]
        return matched_predictions, unmatched_predictions

    def _compute_binary_metrics(
        self, predicted: np.ndarray, ground_truth: np.ndarray
    ) -> BinaryMetrics:
        predicted = predicted.astype(bool)
        ground_truth = ground_truth.astype(bool)
        intersection = np.logical_and(predicted, ground_truth).sum(dtype=np.float64)
        union = np.logical_or(predicted, ground_truth).sum(dtype=np.float64)
        predicted_sum = predicted.sum(dtype=np.float64)
        ground_truth_sum = ground_truth.sum(dtype=np.float64)
        denominator = predicted_sum + ground_truth_sum

        return BinaryMetrics(
            iou=float(intersection / union) if union else 1.0,
            dice=float((2.0 * intersection) / denominator) if denominator else 1.0,
            pixel_accuracy=float(np.mean(predicted == ground_truth)),
        )

    def _build_label_map(self, masks: list[np.ndarray]) -> np.ndarray:
        label_map = np.zeros_like(masks[0], dtype=np.int16)
        for index, mask in enumerate(masks, start=1):
            label_map[mask] = index
        return label_map

    def _decode_mask(self, image_data_url: str) -> np.ndarray:
        _, encoded = image_data_url.split(",", 1)
        image_bytes = base64.b64decode(encoded)
        image = Image.open(BytesIO(image_bytes))
        # convert("RGBA") synthesizes a fully opaque alpha channel for formats that have
        # none, which would decode a JPEG or opaque-PNG mask as the whole frame. Decide
        # which channel carries the mask from the source mode, before converting.
        source_has_alpha = image.mode in {"RGBA", "LA", "PA"} or (
            image.mode == "P" and "transparency" in image.info
        )
        image_array = np.asarray(image.convert("RGBA"))
        if source_has_alpha:
            return image_array[..., 3] > 0
        return np.any(image_array[..., :3] > 0, axis=-1)

    def _resize_mask(self, mask: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
        """Remaps a mask onto `shape`, given as numpy's (height, width)."""
        if mask.shape == shape:
            return mask
        height, width = shape
        resized = Image.fromarray((mask.astype(np.uint8) * 255), mode="L").resize(
            (width, height),
            Image.Resampling.NEAREST,
        )
        return np.asarray(resized) > 0

    def _compute_overall_score(
        self,
        *,
        mean_iou: float,
        mean_dice: float,
        pixel_accuracy: float,
        percent_completed: float,
    ) -> float:
        return 100.0 * (
            0.45 * mean_iou
            + 0.35 * mean_dice
            + 0.10 * pixel_accuracy
            + 0.10 * percent_completed
        )
