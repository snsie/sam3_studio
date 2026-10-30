#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pycocotools.mask as mask_utils
import torch
import torch.distributed
import torch.nn.functional as F
from scipy import ndimage
from torch import nn
from torch.utils.data import DataLoader

from sam3 import build_sam3_image_model
from sam3.model.model_misc import SAM3Output
from sam3.model.utils.misc import copy_data_to_device
from sam3.train.data.collator import collate_fn_api
from sam3.train.data.coco_json_loaders import (
    convert_boxlist_to_normalized_tensor,
    load_coco_and_group_by_image,
)
from sam3.train.data.sam3_image_dataset import Sam3ImageDataset
from sam3.train.loss.loss_fns import (
    Boxes,
    IABCEMdetr,
    Masks,
    SemanticSegCriterion,
    instance_masks_to_semantic_masks,
)
from sam3.train.loss.sam3_loss import Sam3LossWrapper
from sam3.train.matcher import BinaryHungarianMatcherV2, BinaryOneToManyMatcher
from sam3.train.transforms.basic_for_api import (
    ComposeAPI,
    NormalizeAPI,
    PadToSizeAPI,
    RandomHorizontalFlip,
    RandomResizeAPI,
    ToTensorAPI,
)
from sam3.train.transforms.point_sampling import RandomizeInputBbox
from sam3.train.transforms.segmentation import DecodeRle

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "backend"))

from app.services.bbbc038_data import (  # noqa: E402
    masks_to_label_map,
    suppress_duplicate_masks,
)
from app.services.nuclei_metrics import aggregate_metrics, compute_image_metrics  # noqa: E402
from app.services.sam3_lora import inject_lora, lora_state_dict  # noqa: E402

DEFAULT_DATASET = Path(
    "/home/neph-workshop-user/workshop-demo/sam3_studio/datasets/single_class_org"
)
DEFAULT_OUTPUT_DIR = Path(
    "/home/neph-workshop-user/workshop-demo/sam3_studio/tmp_sam3_lora"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fine-tune SAM3 with LoRA adapters on a single-class image/mask dataset. "
            "By default the script trains on images 1-100 and evaluates on 101-200."
        )
    )
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--class-name", default="foreground object")
    parser.add_argument("--train-count", type=int, default=100)
    parser.add_argument("--test-count", type=int, default=100)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--resolution", type=int, default=1008)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--checkpoint-path", type=Path, default=None)
    parser.add_argument(
        "--load-from-hf",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Load the base SAM3 checkpoint from Hugging Face when --checkpoint-path is not provided.",
    )
    parser.add_argument(
        "--train-with-box-prompts",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include positive instance boxes as training prompts.",
    )
    parser.add_argument(
        "--eval-with-box-prompts",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Include positive instance boxes at evaluation time. Disabled by default to avoid GT-box leakage.",
    )
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-alpha", type=float, default=16.0)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument(
        "--lora-prefix",
        action="append",
        default=None,
        help="Module-name prefix to consider for LoRA injection. May be supplied multiple times.",
    )
    parser.add_argument(
        "--lora-suffix",
        action="append",
        default=None,
        help="Linear layer suffix to wrap with LoRA. May be supplied multiple times.",
    )
    parser.add_argument("--bpe-path", type=Path, default=None)
    parser.add_argument(
        "--coco-annotation",
        type=Path,
        default=None,
        help=(
            "Train from an existing COCO (RLE) file instead of --dataset-dir masks. Requires "
            "--image-dir and --split-file; file_name entries are relative to --image-dir."
        ),
    )
    parser.add_argument("--image-dir", type=Path, default=None)
    parser.add_argument(
        "--split-file",
        type=Path,
        default=None,
        help="JSON file with image-id lists for --train-split-key and --val-split-key.",
    )
    parser.add_argument("--train-split-key", default="train")
    parser.add_argument("--val-split-key", default="val")
    parser.add_argument(
        "--min-score",
        type=float,
        default=None,
        help="Drop COCO annotations whose 'score' is below this (pseudo-labels only).",
    )
    parser.add_argument(
        "--skip-empty-images",
        action="store_true",
        help=(
            "Drop images left with no annotations (after --min-score) from the training "
            "and selection sets. Useful for pseudo-labels, where an empty image usually "
            "means the labeller missed every object rather than that there are none."
        ),
    )
    parser.add_argument(
        "--hflip",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Randomly flip training images horizontally.",
    )
    parser.add_argument(
        "--selection-annotation",
        type=Path,
        default=None,
        help=(
            "COCO file whose val-split images score checkpoints (instance DSB score). "
            "Defaults to --coco-annotation."
        ),
    )
    parser.add_argument(
        "--diagnostic-annotation",
        type=Path,
        default=None,
        help="Optional COCO file scored on the val split for reporting only, never for selection.",
    )
    parser.add_argument(
        "--eval-confidence-threshold",
        type=float,
        default=0.5,
        help="Sam3Processor confidence threshold used for instance-level validation.",
    )
    parser.add_argument("--eval-every-steps", type=int, default=None)
    parser.add_argument("--max-train-steps", type=int, default=None)
    parser.add_argument("--max-eval-steps", type=int, default=None)
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Only generate split manifests and exit without building the model.",
    )
    return parser.parse_args()


@dataclass(slots=True)
class SplitArtifacts:
    train_annotation: Path
    test_annotation: Path
    train_images: list[Path]
    test_images: list[Path]


class SingleClassMaskCocoLoader:
    def __init__(
        self,
        annotation_file: str | Path,
        prompt_text: str,
        include_input_boxes: bool,
    ) -> None:
        self._raw_data, self._cat_idx_to_text = load_coco_and_group_by_image(
            str(annotation_file)
        )
        self._prompt_text = prompt_text
        self._include_input_boxes = include_input_boxes

    def getDatapointIds(self) -> list[int]:
        return list(range(len(self._raw_data)))

    def loadQueriesAndAnnotationsFromDatapoint(self, idx: int):
        raw_annotations = self._raw_data[idx]["annotations"]
        image_info = self._raw_data[idx]["image"]
        width = image_info["width"]
        height = image_info["height"]

        annotations = []
        annotation_ids = []
        input_boxes: list[list[float]] = []

        for ann in raw_annotations:
            normalized_box = convert_boxlist_to_normalized_tensor(
                [ann["bbox"]], width, height
            )[0]
            annotation = {
                "image_id": 0,
                "bbox": normalized_box,
                "area": float(normalized_box[2] * normalized_box[3]),
                "segmentation": ann.get("segmentation"),
                "object_id": len(annotations),
                "is_crowd": ann.get("iscrowd", 0),
                "id": len(annotations),
            }
            annotations.append(annotation)
            annotation_ids.append(annotation["id"])
            if self._include_input_boxes:
                input_boxes.append(normalized_box.tolist())

        query = {
            "id": 0,
            "original_cat_id": 1,
            "object_ids_output": annotation_ids,
            "query_text": self._prompt_text,
            "query_processing_order": 0,
            "ptr_x_query_id": None,
            "ptr_y_query_id": None,
            "image_id": 0,
            "input_box": input_boxes or None,
            "input_box_label": ([1] * len(input_boxes)) if input_boxes else None,
            "input_points": None,
            "is_exhaustive": True,
        }
        return [query], annotations

    def loadImagesFromDatapoint(self, idx: int):
        img_data = self._raw_data[idx]["image"]
        return [
            {
                "id": 0,
                "file_name": img_data["file_name"],
                "original_img_id": img_data["id"],
                "coco_img_id": img_data["id"],
            }
        ]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device_name: str) -> torch.device:
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_name)


def sorted_pairs(dataset_dir: Path) -> list[tuple[Path, Path]]:
    image_dir = dataset_dir / "images"
    mask_dir = dataset_dir / "masks"
    if not image_dir.is_dir() or not mask_dir.is_dir():
        raise FileNotFoundError(
            f"Expected {image_dir} and {mask_dir} to exist inside the dataset directory."
        )

    image_paths = sorted(image_dir.glob("*.jpg")) + sorted(image_dir.glob("*.png"))
    if not image_paths:
        raise FileNotFoundError(f"No images were found in {image_dir}.")

    pairs: list[tuple[Path, Path]] = []
    for image_path in sorted(image_paths):
        suffix = image_path.stem.split("_")[-1]
        mask_path = mask_dir / f"label_{suffix}.png"
        if not mask_path.is_file():
            raise FileNotFoundError(
                f"Expected mask {mask_path.name} for image {image_path.name}."
            )
        pairs.append((image_path, mask_path))
    return pairs


def build_split_artifacts(
    dataset_dir: Path,
    output_dir: Path,
    train_count: int,
    test_count: int,
    class_name: str,
) -> SplitArtifacts:
    pairs = sorted_pairs(dataset_dir)
    required = train_count + test_count
    if len(pairs) < required:
        raise ValueError(
            f"Need at least {required} images for the requested split, found {len(pairs)}."
        )

    train_pairs = pairs[:train_count]
    test_pairs = pairs[train_count : train_count + test_count]

    manifest_dir = output_dir / "manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)

    train_annotation = manifest_dir / "train_annotations.coco.json"
    test_annotation = manifest_dir / "test_annotations.coco.json"

    write_coco_manifest(train_pairs, train_annotation, class_name)
    write_coco_manifest(test_pairs, test_annotation, class_name)

    return SplitArtifacts(
        train_annotation=train_annotation,
        test_annotation=test_annotation,
        train_images=[image for image, _ in train_pairs],
        test_images=[image for image, _ in test_pairs],
    )


def coco_image_key(image: dict[str, Any]) -> str:
    return image.get("image_key") or Path(image["file_name"]).parts[0]


def filter_coco(
    coco: dict[str, Any],
    image_keys: set[str],
    min_score: float | None,
    skip_empty_images: bool = False,
) -> dict[str, Any]:
    images = [image for image in coco["images"] if coco_image_key(image) in image_keys]
    kept_ids = {image["id"] for image in images}
    annotations = [
        ann
        for ann in coco["annotations"]
        if ann["image_id"] in kept_ids
        and (min_score is None or ann.get("score", 1.0) >= min_score)
    ]
    if skip_empty_images:
        annotated_ids = {ann["image_id"] for ann in annotations}
        images = [image for image in images if image["id"] in annotated_ids]
    return {
        "images": images,
        "annotations": annotations,
        "categories": coco["categories"],
    }


def build_coco_split_artifacts(
    coco_path: Path,
    image_dir: Path,
    split_file: Path,
    output_dir: Path,
    train_key: str,
    val_key: str,
    min_score: float | None,
    skip_empty_images: bool,
) -> SplitArtifacts:
    coco = json.loads(coco_path.read_text())
    split = json.loads(split_file.read_text())
    manifest_dir = output_dir / "manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)

    paths = {}
    image_lists = {}
    for name, key in (("train", train_key), ("val", val_key)):
        subset = filter_coco(coco, set(split[key]), min_score, skip_empty_images)
        if not subset["images"]:
            raise ValueError(f"No images in {coco_path} belong to split '{key}'.")
        paths[name] = manifest_dir / f"{name}_annotations.coco.json"
        paths[name].write_text(json.dumps(subset))
        image_lists[name] = [image_dir / image["file_name"] for image in subset["images"]]

    return SplitArtifacts(
        train_annotation=paths["train"],
        test_annotation=paths["val"],
        train_images=image_lists["train"],
        test_images=image_lists["val"],
    )


def write_coco_manifest(
    pairs: Sequence[tuple[Path, Path]],
    annotation_path: Path,
    class_name: str,
) -> None:
    images = []
    annotations = []
    annotation_id = 1

    for image_id, (image_path, mask_path) in enumerate(pairs, start=1):
        mask = read_mask(mask_path)
        height, width = mask.shape

        images.append(
            {
                "id": image_id,
                "file_name": image_path.name,
                "width": width,
                "height": height,
            }
        )

        components, component_count = ndimage.label(mask)
        for component_index in range(1, component_count + 1):
            component_mask = components == component_index
            if not component_mask.any():
                continue

            y_coords, x_coords = np.nonzero(component_mask)
            x0 = int(x_coords.min())
            y0 = int(y_coords.min())
            x1 = int(x_coords.max()) + 1
            y1 = int(y_coords.max()) + 1
            bbox = [x0, y0, x1 - x0, y1 - y0]

            rle = mask_utils.encode(np.asfortranarray(component_mask.astype(np.uint8)))
            rle["counts"] = rle["counts"].decode("ascii")

            annotations.append(
                {
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": 1,
                    "bbox": bbox,
                    "segmentation": rle,
                    "area": int(component_mask.sum()),
                    "iscrowd": 0,
                }
            )
            annotation_id += 1

    payload = {
        "images": images,
        "annotations": annotations,
        "categories": [{"id": 1, "name": class_name}],
    }
    annotation_path.write_text(json.dumps(payload, indent=2))


def read_mask(mask_path: Path) -> np.ndarray:
    mask = torch.from_numpy(np.array(ImageLoader.load(mask_path), dtype=np.uint8))
    if mask.ndim != 2:
        raise ValueError(
            f"Expected a 2D mask in {mask_path}, found shape {tuple(mask.shape)}."
        )
    return (mask.numpy() > 0).astype(np.uint8)


class ImageLoader:
    @staticmethod
    def load(path: Path):
        from PIL import Image

        with Image.open(path) as image:
            return image.copy()


def build_transforms(
    resolution: int,
    with_box_prompts: bool,
    training: bool,
    hflip: bool = False,
) -> list[Any]:
    transforms: list[Any] = []
    if training and with_box_prompts:
        transforms.append(RandomizeInputBbox(box_noise_std=0.1, box_noise_max=20))
    transforms.append(DecodeRle())
    if training and hflip:
        transforms.append(RandomHorizontalFlip(consistent_transform=False))
    transforms.append(
        RandomResizeAPI(
            sizes=[resolution],
            max_size=resolution,
            square=True,
            consistent_transform=False,
        )
    )
    if training:
        transforms.append(PadToSizeAPI(size=resolution, consistent_transform=False))
    transforms.append(ToTensorAPI())
    transforms.append(NormalizeAPI(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]))
    return transforms


def build_dataset(
    image_dir: Path,
    annotation_path: Path,
    class_name: str,
    with_box_prompts: bool,
    resolution: int,
    training: bool,
    hflip: bool = False,
) -> Sam3ImageDataset:
    transforms = ComposeAPI(
        build_transforms(
            resolution=resolution,
            with_box_prompts=with_box_prompts,
            training=training,
            hflip=hflip,
        )
    )
    return Sam3ImageDataset(
        img_folder=str(image_dir),
        ann_file=str(annotation_path),
        transforms=[transforms],
        max_ann_per_img=500000,
        multiplier=1,
        training=training,
        load_segmentation=True,
        max_train_queries=50000,
        max_val_queries=50000,
        use_caching=False,
        coco_json_loader=lambda ann_file: SingleClassMaskCocoLoader(
            annotation_file=ann_file,
            prompt_text=class_name,
            include_input_boxes=with_box_prompts,
        ),
    )


def build_loader(
    dataset: Sam3ImageDataset,
    batch_size: int,
    num_workers: int,
    shuffle: bool,
    pin_memory: bool,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
        collate_fn=lambda batch: collate_fn_api(
            batch,
            dict_key="single_class",
            with_seg_masks=True,
            repeats=1,
        ),
    )


def default_lora_prefixes() -> list[str]:
    return [
        "backbone.vision_backbone.trunk.blocks",
        "transformer",
        "input_geometry_encoder",
        "dot_prod_scoring",
        "segmentation_head.cross_attend_prompt",
    ]


def default_lora_suffixes() -> list[str]:
    return [
        "qkv",
        "proj",
        "fc1",
        "fc2",
        "out_proj",
        "linear1",
        "linear2",
        "c_fc",
        "c_proj",
    ]


def build_loss() -> Sam3LossWrapper:
    matcher = BinaryHungarianMatcherV2(
        focal=True,
        cost_class=2.0,
        cost_bbox=5.0,
        cost_giou=2.0,
        alpha=0.25,
        gamma=2.0,
        stable=False,
    )
    o2m_matcher = BinaryOneToManyMatcher(alpha=0.3, threshold=0.4, topk=4)
    return Sam3LossWrapper(
        normalization="local",
        matcher=matcher,
        o2m_matcher=o2m_matcher,
        o2m_weight=2.0,
        use_o2m_matcher_on_o2m_aux=False,
        loss_fns_find=[
            Boxes(weight_dict={"loss_bbox": 5.0, "loss_giou": 2.0}),
            IABCEMdetr(
                pos_weight=10.0,
                alpha=0.25,
                gamma=2.0,
                weak_loss=False,
                use_presence=False,
                pad_n_queries=200,
                pad_scale_pos=1.0,
                weight_dict={"loss_ce": 20.0},
            ),
            Masks(
                focal_alpha=0.25,
                focal_gamma=2.0,
                compute_aux=False,
                weight_dict={"loss_mask": 200.0, "loss_dice": 10.0},
            ),
        ],
        loss_fn_semantic_seg=SemanticSegCriterion(
            presence_head=False,
            presence_loss=False,
            focal=True,
            focal_alpha=0.6,
            focal_gamma=2.0,
            downsample=False,
            weight_dict={"loss_semantic_seg": 20.0, "loss_semantic_dice": 30.0},
        ),
        scale_by_find_batch_size=True,
    )


def named_single_batch(batch: dict[str, Any]) -> tuple[str, Any]:
    if len(batch) != 1:
        raise ValueError(f"Expected a single keyed batch, found keys: {list(batch)}")
    return next(iter(batch.items()))


def unwrap_last_stage(model_output: SAM3Output) -> dict[str, torch.Tensor]:
    with SAM3Output.iteration_mode(
        model_output, SAM3Output.IterMode.LAST_STEP_PER_STAGE
    ):
        return model_output[0]


def union_semantic_masks(targets: Any) -> torch.Tensor:
    if targets.semantic_segments is not None and targets.semantic_segments.numel() > 0:
        return targets.semantic_segments.bool()
    return instance_masks_to_semantic_masks(
        targets.segments.bool(),
        targets.num_boxes,
    )


def train_one_epoch(
    model: nn.Module,
    criterion: Sam3LossWrapper,
    optimizer: torch.optim.Optimizer,
    loader: DataLoader,
    device: torch.device,
    scaler: torch.cuda.amp.GradScaler | None,
    max_grad_norm: float,
    max_steps: int | None,
    epoch: int,
) -> float:
    model.train()
    if hasattr(loader.dataset, "set_curr_epoch"):
        loader.dataset.set_curr_epoch(epoch)

    total_loss = 0.0
    step_count = 0
    for step, batch in enumerate(loader, start=1):
        _, batch_data = named_single_batch(batch)
        batch_data = copy_data_to_device(
            batch_data, device, non_blocking=device.type == "cuda"
        )

        optimizer.zero_grad(set_to_none=True)
        autocast_enabled = scaler is not None
        with torch.amp.autocast(
            device_type=device.type,
            enabled=autocast_enabled,
            dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
        ):
            outputs = model(batch_data)
            targets = [
                model.back_convert(find_target)
                for find_target in batch_data.find_targets
            ]
            loss_dict = criterion(outputs, targets)
            loss = loss_dict["core_loss"]

        if scaler is None:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_parameters(model), max_grad_norm)
            optimizer.step()
        else:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(trainable_parameters(model), max_grad_norm)
            scaler.step(optimizer)
            scaler.update()

        total_loss += float(loss.item())
        step_count += 1
        if max_steps is not None and step >= max_steps:
            break

    return total_loss / max(step_count, 1)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    max_steps: int | None,
) -> dict[str, float]:
    model.eval()
    dice_scores: list[float] = []
    iou_scores: list[float] = []

    for step, batch in enumerate(loader, start=1):
        _, batch_data = named_single_batch(batch)
        batch_data = copy_data_to_device(
            batch_data, device, non_blocking=device.type == "cuda"
        )
        outputs = unwrap_last_stage(model(batch_data))

        prediction = outputs["semantic_seg"].sigmoid()
        targets = union_semantic_masks(batch_data.find_targets[0]).float()
        prediction = F.interpolate(
            prediction,
            size=targets.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        prediction_mask = prediction.squeeze(1) > 0.5
        target_mask = targets.bool()

        for predicted, target in zip(prediction_mask, target_mask):
            dice, iou = binary_metrics(predicted, target)
            dice_scores.append(dice)
            iou_scores.append(iou)

        if max_steps is not None and step >= max_steps:
            break

    if not dice_scores:
        return {"mean_dice": 0.0, "mean_iou": 0.0}
    return {
        "mean_dice": float(np.mean(dice_scores)),
        "mean_iou": float(np.mean(iou_scores)),
    }


def coco_label_maps(annotation_path: Path) -> dict[str, tuple[str, np.ndarray]]:
    """Maps image key -> (file_name, non-overlapping instance label map)."""
    coco = json.loads(annotation_path.read_text())
    annotations_by_image: dict[int, list[dict[str, Any]]] = {}
    for ann in coco["annotations"]:
        annotations_by_image.setdefault(ann["image_id"], []).append(ann)

    label_maps = {}
    for image in coco["images"]:
        anns = annotations_by_image.get(image["id"], [])
        masks = [mask_utils.decode(ann["segmentation"]).astype(bool) for ann in anns]
        scores = [ann.get("score", 1.0) for ann in anns] if anns else None
        label_maps[coco_image_key(image)] = (
            image["file_name"],
            masks_to_label_map(masks, (image["height"], image["width"]), scores),
        )
    return label_maps


def predict_label_map(processor: Any, image: Any, prompt: str) -> np.ndarray:
    state = processor.set_image(image)
    state = processor.set_text_prompt(prompt, state)
    masks = state["masks"].detach().cpu().numpy()
    if masks.ndim == 4:
        masks = masks[:, 0]
    scores = state["scores"].detach().float().cpu().numpy().tolist()
    instance_masks, scores = suppress_duplicate_masks(
        [mask.astype(bool) for mask in masks], scores
    )
    return masks_to_label_map(instance_masks, (image.height, image.width), scores)


@torch.no_grad()
def evaluate_instances(
    model: nn.Module,
    image_dir: Path,
    label_maps: dict[str, tuple[str, np.ndarray]],
    prompt: str,
    device: torch.device,
    confidence_threshold: float,
    max_images: int | None,
) -> dict[str, float]:
    """DSB-style instance metrics through Sam3Processor, the same path used at inference."""
    from PIL import Image
    from sam3.model.sam3_image_processor import Sam3Processor

    model.eval()
    processor = Sam3Processor(
        model, device=str(device), confidence_threshold=confidence_threshold
    )
    per_image = []
    for index, (file_name, gt_labels) in enumerate(label_maps.values()):
        if max_images is not None and index >= max_images:
            break
        with Image.open(image_dir / file_name) as raw_image:
            image = raw_image.convert("RGB")
        with torch.autocast(
            device_type=device.type,
            enabled=device.type == "cuda",
            dtype=torch.bfloat16,
        ):
            pred_labels = predict_label_map(processor, image, prompt)
        per_image.append(compute_image_metrics(gt_labels, pred_labels).to_dict())

    summary = aggregate_metrics(per_image)
    return {
        key: summary[key]
        for key in ("dsb_score", "f1_at_50", "pixel_dice", "pixel_iou", "abs_count_error")
        if key in summary
    }


def binary_metrics(
    prediction: torch.Tensor, target: torch.Tensor
) -> tuple[float, float]:
    prediction = prediction.bool()
    target = target.bool()
    intersection = int(torch.logical_and(prediction, target).sum().item())
    union = int(torch.logical_or(prediction, target).sum().item())
    pred_sum = int(prediction.sum().item())
    target_sum = int(target.sum().item())

    if pred_sum == 0 and target_sum == 0:
        return 1.0, 1.0

    dice = (2.0 * intersection) / max(pred_sum + target_sum, 1)
    iou = intersection / max(union, 1)
    return dice, iou


def trainable_parameters(model: nn.Module) -> Iterable[nn.Parameter]:
    return (parameter for parameter in model.parameters() if parameter.requires_grad)


def count_parameters(parameters: Iterable[nn.Parameter]) -> int:
    return sum(parameter.numel() for parameter in parameters)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    coco_mode = args.coco_annotation is not None

    if coco_mode:
        if args.image_dir is None or args.split_file is None:
            raise ValueError("--coco-annotation requires --image-dir and --split-file.")
        image_dir = args.image_dir.resolve()
        split = build_coco_split_artifacts(
            coco_path=args.coco_annotation.resolve(),
            image_dir=image_dir,
            split_file=args.split_file.resolve(),
            output_dir=output_dir,
            train_key=args.train_split_key,
            val_key=args.val_split_key,
            min_score=args.min_score,
            skip_empty_images=args.skip_empty_images,
        )
        dataset_source = str(args.coco_annotation.resolve())
    else:
        dataset_dir = args.dataset_dir.resolve()
        image_dir = dataset_dir / "images"
        split = build_split_artifacts(
            dataset_dir=dataset_dir,
            output_dir=output_dir,
            train_count=args.train_count,
            test_count=args.test_count,
            class_name=args.class_name,
        )
        dataset_source = str(dataset_dir)

    split_summary = {
        "dataset": dataset_source,
        "train_images": [path.name for path in split.train_images],
        "test_images": [path.name for path in split.test_images],
        "train_annotation": str(split.train_annotation),
        "test_annotation": str(split.test_annotation),
    }
    write_json(output_dir / "split_summary.json", split_summary)

    if args.prepare_only:
        print(json.dumps(split_summary, indent=2))
        return

    device = resolve_device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")

    train_dataset = build_dataset(
        image_dir=image_dir,
        annotation_path=split.train_annotation,
        class_name=args.class_name,
        with_box_prompts=args.train_with_box_prompts,
        resolution=args.resolution,
        training=True,
        hflip=args.hflip,
    )
    train_loader = build_loader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
        pin_memory=device.type == "cuda",
    )

    test_loader = None
    eval_sets: dict[str, dict[str, tuple[str, np.ndarray]]] = {}
    if coco_mode:
        # Score checkpoints on instance metrics for the val split; the diagnostic set is
        # reported alongside but never used to pick a checkpoint.
        val_keys = set(json.loads(args.split_file.read_text())[args.val_split_key])
        selection_source = args.selection_annotation or args.coco_annotation
        for name, source in (("val", selection_source), ("val_diagnostic", args.diagnostic_annotation)):
            if source is None:
                continue
            subset_path = output_dir / "manifests" / f"{name}_selection.coco.json"
            is_training_source = source == args.coco_annotation
            subset = filter_coco(
                json.loads(source.read_text()),
                val_keys,
                args.min_score if is_training_source else None,
                args.skip_empty_images and is_training_source,
            )
            subset_path.write_text(json.dumps(subset))
            eval_sets[name] = coco_label_maps(subset_path)
    else:
        test_dataset = build_dataset(
            image_dir=image_dir,
            annotation_path=split.test_annotation,
            class_name=args.class_name,
            with_box_prompts=args.eval_with_box_prompts,
            resolution=args.resolution,
            training=False,
        )
        test_loader = build_loader(
            dataset=test_dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            shuffle=False,
            pin_memory=device.type == "cuda",
        )

    checkpoint_path = (
        None if args.checkpoint_path is None else str(args.checkpoint_path.resolve())
    )
    model = build_sam3_image_model(
        checkpoint_path=checkpoint_path,
        bpe_path=None if args.bpe_path is None else str(args.bpe_path.resolve()),
        device=str(device),
        eval_mode=False,
        load_from_HF=args.load_from_hf,
        enable_segmentation=True,
        enable_inst_interactivity=False,
        compile=False,
    )

    prefixes = args.lora_prefix or default_lora_prefixes()
    suffixes = args.lora_suffix or default_lora_suffixes()
    replaced_modules = inject_lora(
        model,
        prefixes=prefixes,
        suffixes=suffixes,
        rank=args.lora_rank,
        alpha=args.lora_alpha,
        dropout=args.lora_dropout,
    )
    model.to(device)

    criterion = build_loss().to(device)
    optimizer = torch.optim.AdamW(
        trainable_parameters(model),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None

    total_params = count_parameters(model.parameters())
    trainable_params = count_parameters(trainable_parameters(model))

    run_summary = {
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "base_checkpoint": checkpoint_path,
        "bpe_path": None if args.bpe_path is None else str(args.bpe_path.resolve()),
        "dataset": dataset_source,
        "split_file": None if args.split_file is None else str(args.split_file.resolve()),
        "selection_annotation": None
        if not coco_mode
        else str((args.selection_annotation or args.coco_annotation).resolve()),
        "diagnostic_annotation": None
        if args.diagnostic_annotation is None
        else str(args.diagnostic_annotation.resolve()),
        "prompt": args.class_name,
        "train_image_count": len(split.train_images),
        "val_image_count": len(split.test_images),
        "min_score": args.min_score,
        "skip_empty_images": args.skip_empty_images,
        "hflip": args.hflip,
        "resolution": args.resolution,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "lora_prefixes": list(prefixes),
        "lora_suffixes": list(suffixes),
        "lora_module_count": len(replaced_modules),
        "lora_modules": replaced_modules,
        "lora_rank": args.lora_rank,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": args.lora_dropout,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "epochs": args.epochs,
        "train_with_box_prompts": args.train_with_box_prompts,
        "eval_with_box_prompts": args.eval_with_box_prompts,
        "eval_confidence_threshold": args.eval_confidence_threshold,
        "selection_metric": "val.dsb_score" if coco_mode else "mean_dice",
    }
    write_json(output_dir / "run_summary.json", run_summary)

    history: list[dict[str, Any]] = []
    best_score = -1.0
    best_checkpoint_path = output_dir / "checkpoints" / "best_lora.pt"
    last_checkpoint_path = output_dir / "checkpoints" / "last_lora.pt"
    best_metrics: dict[str, Any] | None = None
    best_epoch: int | None = None
    training_started = time.perf_counter()

    for epoch in range(args.epochs):
        epoch_started = time.perf_counter()
        train_loss = train_one_epoch(
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            loader=train_loader,
            device=device,
            scaler=scaler,
            max_grad_norm=args.max_grad_norm,
            max_steps=args.max_train_steps,
            epoch=epoch,
        )
        train_seconds = time.perf_counter() - epoch_started

        if coco_mode:
            metrics = {
                name: evaluate_instances(
                    model=model,
                    image_dir=image_dir,
                    label_maps=label_maps,
                    prompt=args.class_name,
                    device=device,
                    confidence_threshold=args.eval_confidence_threshold,
                    max_images=args.max_eval_steps,
                )
                for name, label_maps in eval_sets.items()
            }
            selection_score = metrics["val"]["dsb_score"]
        else:
            metrics = evaluate(
                model=model,
                loader=test_loader,
                device=device,
                max_steps=args.max_eval_steps,
            )
            selection_score = metrics["mean_dice"]

        record = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_seconds": train_seconds,
            "epoch_seconds": time.perf_counter() - epoch_started,
            **metrics,
        }
        history.append(record)
        print(json.dumps(record), flush=True)
        write_json(output_dir / "history.json", history)

        checkpoint_payload = {
            "config": run_summary,
            "epoch": epoch + 1,
            "history": history,
            "metrics": metrics,
            "lora_state_dict": lora_state_dict(model),
        }
        last_checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(checkpoint_payload, last_checkpoint_path)

        if selection_score > best_score:
            best_score = selection_score
            best_metrics = metrics
            best_epoch = epoch + 1
            torch.save(checkpoint_payload, best_checkpoint_path)

    final_report = {
        "best_epoch": best_epoch,
        "best_metrics": best_metrics or {},
        "selection_metric": run_summary["selection_metric"],
        "training_seconds": time.perf_counter() - training_started,
        "history": history,
        "best_checkpoint": str(best_checkpoint_path),
        "last_checkpoint": str(last_checkpoint_path),
    }
    write_json(output_dir / "metrics.json", final_report)
    print(json.dumps({key: value for key, value in final_report.items() if key != "history"}, indent=2))


if __name__ == "__main__":
    main()
