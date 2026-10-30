"""Instance segmentation metrics for the 2018 Data Science Bowl.

All functions take non-overlapping int label maps (0 = background, 1..N = instances).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

# Official DSB 2018 thresholds: 0.50, 0.55, ..., 0.95.
DSB_IOU_THRESHOLDS: tuple[float, ...] = tuple(
    round(0.5 + 0.05 * step, 2) for step in range(10)
)


@dataclass(frozen=True, slots=True)
class ImageMetrics:
    dsb_score: float
    f1_at_50: float
    precision_at_50: float
    recall_at_50: float
    pixel_dice: float
    pixel_iou: float
    instance_mean_iou: float
    gt_count: int
    pred_count: int
    count_error: int
    abs_count_error: int
    ap_per_threshold: list[float]

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class MatchClassification:
    true_positive_pred: list[int]
    false_positive_pred: list[int]
    false_negative_gt: list[int]
    matched_gt_for_pred: dict[int, int]


def iou_matrix(gt_labels: np.ndarray, pred_labels: np.ndarray) -> np.ndarray:
    """Returns IoU[gt_index, pred_index] for instances 1..N (row/col 0 = label 1)."""
    if gt_labels.shape != pred_labels.shape:
        raise ValueError(
            f"Label maps differ in shape: {gt_labels.shape} vs {pred_labels.shape}"
        )
    gt = gt_labels.astype(np.int64).ravel()
    pred = pred_labels.astype(np.int64).ravel()
    num_gt = int(gt.max(initial=0))
    num_pred = int(pred.max(initial=0))
    if num_gt == 0 or num_pred == 0:
        return np.zeros((num_gt, num_pred), dtype=np.float64)

    joint = np.bincount(
        gt * (num_pred + 1) + pred, minlength=(num_gt + 1) * (num_pred + 1)
    ).reshape(num_gt + 1, num_pred + 1)
    gt_areas = joint.sum(axis=1)
    pred_areas = joint.sum(axis=0)
    intersection = joint[1:, 1:].astype(np.float64)
    union = gt_areas[1:, None] + pred_areas[None, 1:] - intersection
    return np.divide(
        intersection, union, out=np.zeros_like(intersection), where=union > 0
    )


def precision_at(iou: np.ndarray, threshold: float) -> tuple[int, int, int]:
    """Counts TP/FP/FN at an IoU threshold.

    For thresholds >= 0.5 each object can match at most one other, so counting entries
    above the threshold is an exact one-to-one matching.
    """
    matches = iou > threshold
    true_positives = int(matches.any(axis=1).sum())
    false_negatives = int((~matches.any(axis=1)).sum())
    false_positives = int((~matches.any(axis=0)).sum())
    return true_positives, false_positives, false_negatives


def compute_image_metrics(gt_labels: np.ndarray, pred_labels: np.ndarray) -> ImageMetrics:
    iou = iou_matrix(gt_labels, pred_labels)
    num_gt, num_pred = iou.shape

    ap_per_threshold: list[float] = []
    for threshold in DSB_IOU_THRESHOLDS:
        tp, fp, fn = precision_at(iou, threshold)
        denominator = tp + fp + fn
        # An image with no objects on either side is a perfect prediction.
        ap_per_threshold.append(1.0 if denominator == 0 else tp / denominator)

    tp, fp, fn = precision_at(iou, 0.5)
    f1 = 1.0 if (2 * tp + fp + fn) == 0 else (2 * tp) / (2 * tp + fp + fn)
    precision = 1.0 if (tp + fp) == 0 else tp / (tp + fp)
    recall = 1.0 if (tp + fn) == 0 else tp / (tp + fn)

    gt_foreground = gt_labels > 0
    pred_foreground = pred_labels > 0
    intersection = int(np.logical_and(gt_foreground, pred_foreground).sum())
    gt_area = int(gt_foreground.sum())
    pred_area = int(pred_foreground.sum())
    union = gt_area + pred_area - intersection
    pixel_dice = 1.0 if gt_area + pred_area == 0 else 2 * intersection / (gt_area + pred_area)
    pixel_iou = 1.0 if union == 0 else intersection / union

    # Best IoU for every ground-truth object, with missed objects counting as 0.
    if num_gt == 0:
        instance_mean_iou = 1.0 if num_pred == 0 else 0.0
    elif num_pred == 0:
        instance_mean_iou = 0.0
    else:
        instance_mean_iou = float(iou.max(axis=1).mean())

    return ImageMetrics(
        dsb_score=float(np.mean(ap_per_threshold)),
        f1_at_50=float(f1),
        precision_at_50=float(precision),
        recall_at_50=float(recall),
        pixel_dice=float(pixel_dice),
        pixel_iou=float(pixel_iou),
        instance_mean_iou=instance_mean_iou,
        gt_count=num_gt,
        pred_count=num_pred,
        count_error=num_pred - num_gt,
        abs_count_error=abs(num_pred - num_gt),
        ap_per_threshold=[float(value) for value in ap_per_threshold],
    )


def classify_matches(
    gt_labels: np.ndarray,
    pred_labels: np.ndarray,
    threshold: float = 0.5,
) -> MatchClassification:
    """Labels each prediction TP/FP and each ground-truth object matched/FN (1-based ids)."""
    iou = iou_matrix(gt_labels, pred_labels)
    matches = iou > threshold
    matched_gt_for_pred: dict[int, int] = {}
    for gt_index, pred_index in zip(*np.nonzero(matches)):
        matched_gt_for_pred[int(pred_index) + 1] = int(gt_index) + 1

    num_gt, num_pred = iou.shape
    true_positive = sorted(matched_gt_for_pred)
    false_positive = [
        pred_id for pred_id in range(1, num_pred + 1) if pred_id not in matched_gt_for_pred
    ]
    matched_gt = set(matched_gt_for_pred.values())
    false_negative = [gt_id for gt_id in range(1, num_gt + 1) if gt_id not in matched_gt]
    return MatchClassification(
        true_positive_pred=true_positive,
        false_positive_pred=false_positive,
        false_negative_gt=false_negative,
        matched_gt_for_pred=matched_gt_for_pred,
    )


AGGREGATE_KEYS = (
    "dsb_score",
    "f1_at_50",
    "precision_at_50",
    "recall_at_50",
    "pixel_dice",
    "pixel_iou",
    "instance_mean_iou",
    "abs_count_error",
)


def aggregate_metrics(per_image: list[dict]) -> dict:
    """Means over images, plus count error summaries and per-threshold AP."""
    if not per_image:
        return {"image_count": 0}

    summary: dict = {"image_count": len(per_image)}
    for key in AGGREGATE_KEYS:
        summary[key] = float(np.mean([record[key] for record in per_image]))
    summary["mean_count_error"] = float(np.mean([record["count_error"] for record in per_image]))
    gt_total = sum(record["gt_count"] for record in per_image)
    pred_total = sum(record["pred_count"] for record in per_image)
    summary["gt_count_total"] = gt_total
    summary["pred_count_total"] = pred_total
    summary["relative_abs_count_error"] = float(
        np.mean(
            [
                record["abs_count_error"] / max(record["gt_count"], 1)
                for record in per_image
            ]
        )
    )
    summary["ap_per_threshold"] = [
        float(value)
        for value in np.mean([record["ap_per_threshold"] for record in per_image], axis=0)
    ]
    return summary
