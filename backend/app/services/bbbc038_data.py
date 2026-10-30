"""Dataset helpers for BBBC038 (2018 Data Science Bowl nuclei).

Shared by the offline scripts in ``scripts/`` and by the backend's nuclei demo, so it
only depends on NumPy and Pillow.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

import numpy as np
from PIL import Image

BACKEND_DIR = Path(__file__).resolve().parents[2]
DATASET_DIR = BACKEND_DIR / "data" / "datasets" / "bbbc038"
RAW_DIR = DATASET_DIR / "raw"
TRAIN_DIR = RAW_DIR / "stage1_train"
TEST_DIR = RAW_DIR / "stage1_test"
SPLIT_PATH = DATASET_DIR / "split.json"
SAM3_ANNOTATIONS_DIR = DATASET_DIR / "sam3_annotations"
MODELS_DIR = BACKEND_DIR / "data" / "models" / "bbbc038"

# Images whose channels differ by less than this (mean absolute difference, 0-1 scale)
# are treated as grayscale. Stage1 grayscale images are exactly 0; stained histology
# sits above 0.05, so the threshold has a wide margin on both sides.
GRAYSCALE_CHANNEL_DIFF = 0.01
# Grayscale images with a median intensity above this have a bright background.
BRIGHT_BACKGROUND_MEDIAN = 0.5


class ImageType(StrEnum):
    FLUORESCENT = "fluorescent"
    BRIGHTFIELD = "brightfield"
    HISTOLOGY = "histology"


@dataclass(frozen=True, slots=True)
class ColorStats:
    channel_diff: float
    median_intensity: float


def image_path(split_dir: Path, image_id: str) -> Path:
    return split_dir / image_id / "images" / f"{image_id}.png"


def masks_dir(split_dir: Path, image_id: str) -> Path:
    return split_dir / image_id / "masks"


def list_image_ids(split_dir: Path) -> list[str]:
    return sorted(path.name for path in split_dir.iterdir() if path.is_dir())


def load_rgb(path: Path) -> np.ndarray:
    """Loads any stage1 image (RGBA, RGB, grayscale, 16-bit) as uint8 RGB."""
    with Image.open(path) as image:
        if image.mode in ("I;16", "I;16B", "I"):
            array = np.asarray(image, dtype=np.float32)
            peak = float(array.max()) or 1.0
            image = Image.fromarray((array / peak * 255.0).astype(np.uint8))
        return np.asarray(image.convert("RGB"), dtype=np.uint8)


def compute_color_stats(rgb: np.ndarray) -> ColorStats:
    pixels = rgb.astype(np.float32) / 255.0
    channel_diff = float(
        np.abs(pixels[..., 0] - pixels[..., 1]).mean()
        + np.abs(pixels[..., 1] - pixels[..., 2]).mean()
    )
    median_intensity = float(np.median(pixels.mean(axis=-1)))
    return ColorStats(channel_diff=channel_diff, median_intensity=median_intensity)


def classify_image_type(rgb: np.ndarray) -> ImageType:
    stats = compute_color_stats(rgb)
    if stats.channel_diff >= GRAYSCALE_CHANNEL_DIFF:
        return ImageType.HISTOLOGY
    if stats.median_intensity >= BRIGHT_BACKGROUND_MEDIAN:
        return ImageType.BRIGHTFIELD
    return ImageType.FLUORESCENT


def rle_decode(encoded: str, height: int, width: int) -> np.ndarray:
    """Decodes a Kaggle DSB run-length string.

    Runs are ``start length`` pairs with 1-indexed starts over pixels numbered top to
    bottom, then left to right (column-major / Fortran order).
    """
    flat = np.zeros(height * width, dtype=bool)
    values = np.asarray(encoded.split(), dtype=np.int64)
    if values.size % 2:
        raise ValueError("Run-length string must contain start/length pairs.")
    for start, length in zip(values[0::2] - 1, values[1::2]):
        flat[start : start + length] = True
    return flat.reshape((width, height)).T


def rle_encode(mask: np.ndarray) -> str:
    flat = np.concatenate([[0], mask.T.reshape(-1).astype(np.uint8), [0]])
    changes = np.flatnonzero(flat[1:] != flat[:-1]) + 1
    changes[1::2] -= changes[0::2]
    return " ".join(str(value) for value in changes)


def load_instance_masks(split_dir: Path, image_id: str) -> list[np.ndarray]:
    mask_paths = sorted(masks_dir(split_dir, image_id).glob("*.png"))
    masks: list[np.ndarray] = []
    for mask_path in mask_paths:
        with Image.open(mask_path) as mask_image:
            masks.append(np.asarray(mask_image.convert("L")) > 0)
    return masks


def suppress_duplicate_masks(
    masks: list[np.ndarray],
    scores: list[float],
    iou_threshold: float = 0.8,
) -> tuple[list[np.ndarray], list[float]]:
    """Greedy mask NMS, highest score first.

    Mirrors the de-duplication SAM3ModelRuntime applies to its candidates, so offline
    metrics match what the app returns for the same image.
    """
    order = np.argsort(-np.asarray(scores, dtype=np.float32), kind="stable")
    kept: list[int] = []
    boxes: dict[int, tuple[int, int, int, int]] = {}
    for index in order:
        ys, xs = np.nonzero(masks[index])
        if len(ys) == 0:
            continue
        box = (int(ys.min()), int(ys.max()) + 1, int(xs.min()), int(xs.max()) + 1)
        duplicate = False
        for kept_index in kept:
            other = boxes[kept_index]
            y0, y1 = max(box[0], other[0]), min(box[1], other[1])
            x0, x1 = max(box[2], other[2]), min(box[3], other[3])
            if y0 >= y1 or x0 >= x1:
                continue
            y0, y1 = min(box[0], other[0]), max(box[1], other[1])
            x0, x1 = min(box[2], other[2]), max(box[3], other[3])
            left = masks[index][y0:y1, x0:x1]
            right = masks[kept_index][y0:y1, x0:x1]
            union = np.logical_or(left, right).sum()
            if union and np.logical_and(left, right).sum() / union > iou_threshold:
                duplicate = True
                break
        if not duplicate:
            kept.append(int(index))
            boxes[int(index)] = box
    return [masks[index] for index in kept], [float(scores[index]) for index in kept]


def masks_to_label_map(
    masks: list[np.ndarray],
    shape: tuple[int, int],
    scores: list[float] | None = None,
) -> np.ndarray:
    """Paints instance masks into a non-overlapping int32 label map (1..N).

    When scores are given, higher-scoring masks win contested pixels; otherwise later
    masks win. Instances that end up with no pixels are dropped and labels compacted.
    """
    labels = np.zeros(shape, dtype=np.int32)
    order = (
        np.argsort(np.asarray(scores, dtype=np.float32), kind="stable")
        if scores is not None
        else np.arange(len(masks))
    )
    for paint_index, mask_index in enumerate(order, start=1):
        labels[masks[mask_index]] = paint_index
    return relabel_sequential(labels)


def relabel_sequential(labels: np.ndarray) -> np.ndarray:
    present = np.unique(labels)
    present = present[present > 0]
    lookup = np.zeros(int(labels.max()) + 1, dtype=np.int32)
    lookup[present] = np.arange(1, len(present) + 1, dtype=np.int32)
    return lookup[labels]


def load_ground_truth_labels(split_dir: Path, image_id: str) -> np.ndarray:
    masks = load_instance_masks(split_dir, image_id)
    with Image.open(image_path(split_dir, image_id)) as image:
        shape = (image.height, image.width)
    return masks_to_label_map(masks, shape)


def load_split(path: Path = SPLIT_PATH) -> dict:
    return json.loads(path.read_text())
