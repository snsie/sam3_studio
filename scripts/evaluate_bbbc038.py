#!/usr/bin/env python3
"""Score SAM3 variants on BBBC038 stage1_test against stage1_solution.csv.

Models (each predicts with Sam3Processor.set_text_prompt and the same prompt/threshold):
  sam3_lora_pseudo  LoRA fine-tuned on SAM3 "blobs" pseudo-labels (the trained model)
  sam3_zero_shot    the base SAM3 checkpoint, no fine-tuning
  sam3_lora_gt      LoRA fine-tuned on stage1_train ground truth (upper-bound reference)

Test predictions are cached as COCO files under backend/data/models/bbbc038/predictions/.
The test set is only scored here; nothing is tuned on it.

Writes backend/data/models/bbbc038/metrics.json with per-image, aggregate and per-image-type
metrics, plus the training curves and pseudo-label quality the demo page shows.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "backend"))

from annotate_bbbc038_sam3 import (  # noqa: E402
    annotate_image,
    build_coco,
    record_label_map,
)
from app.core.config import get_settings  # noqa: E402
from app.services.bbbc038_data import (  # noqa: E402
    MODELS_DIR,
    SAM3_ANNOTATIONS_DIR,
    TEST_DIR,
    ImageType,
    image_path,
    load_ground_truth_labels,
    load_rgb,
    load_split,
)
from app.services.nuclei_metrics import (  # noqa: E402
    DSB_IOU_THRESHOLDS,
    aggregate_metrics,
    compute_image_metrics,
)
from app.services.sam3_lora import build_lora_sam3_model  # noqa: E402

MODEL_SPECS: dict[str, dict[str, Any]] = {
    "sam3_lora_pseudo": {
        "label": "Trained on pseudo-labels",
        "description": "SAM3 + LoRA fine-tuned on SAM3 'blobs' pseudo-labels of the train split.",
        "lora_dir": "sam3_lora_pseudo",
    },
    "sam3_zero_shot": {
        "label": "Zero-shot SAM3",
        "description": "Base SAM3 checkpoint with the text prompt, no fine-tuning.",
        "lora_dir": None,
    },
    "sam3_lora_gt": {
        "label": "Trained on ground truth",
        "description": "Same SAM3 + LoRA setup trained on stage1_train ground truth (upper-bound reference).",
        "lora_dir": "sam3_lora_gt",
    },
}


def parse_args() -> argparse.Namespace:
    settings = get_settings()
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--prompt", default="blobs")
    parser.add_argument("--threshold", type=float, default=settings.sam3_confidence_threshold)
    parser.add_argument("--models", nargs="+", default=list(MODEL_SPECS), choices=list(MODEL_SPECS))
    parser.add_argument(
        "--force", action="store_true", help="Recompute cached test predictions."
    )
    parser.add_argument("--checkpoint-path", default=settings.sam3_checkpoint_path)
    parser.add_argument("--bpe-path", default=settings.sam3_bpe_path)
    parser.add_argument(
        "--load-from-hf",
        action=argparse.BooleanOptionalAction,
        default=settings.sam3_load_from_hf,
    )
    parser.add_argument("--output", type=Path, default=MODELS_DIR / "metrics.json")
    return parser.parse_args()


def lora_checkpoint(spec: dict[str, Any]) -> Path | None:
    if spec["lora_dir"] is None:
        return None
    return MODELS_DIR / spec["lora_dir"] / "checkpoints" / "best_lora.pt"


def build_model(args: argparse.Namespace, spec: dict[str, Any], device: str):
    from sam3 import build_sam3_image_model

    checkpoint = lora_checkpoint(spec)
    if checkpoint is None:
        return build_sam3_image_model(
            checkpoint_path=args.checkpoint_path,
            bpe_path=args.bpe_path,
            device=device,
            eval_mode=True,
            load_from_HF=args.load_from_hf,
            enable_inst_interactivity=False,
            compile=False,
        )
    model, _ = build_lora_sam3_model(
        checkpoint,
        checkpoint_path=args.checkpoint_path,
        bpe_path=args.bpe_path,
        load_from_hf=args.load_from_hf,
        device=device,
    )
    return model


def predict_test_set(
    args: argparse.Namespace, model_key: str, test_ids: list[str], device: str
) -> list[dict[str, Any]]:
    from sam3.model.sam3_image_processor import Sam3Processor

    predictions_dir = MODELS_DIR / "predictions"
    predictions_dir.mkdir(parents=True, exist_ok=True)
    cache_path = predictions_dir / f"{model_key}.records.json"
    if cache_path.exists() and not args.force:
        cached = json.loads(cache_path.read_text())
        if cached["prompt"] == args.prompt and cached["threshold"] == args.threshold:
            print(f"[{model_key}] using cached predictions {cache_path}")
            return cached["records"]

    spec = MODEL_SPECS[model_key]
    model = build_model(args, spec, device)
    processor = Sam3Processor(model, device=device, confidence_threshold=args.threshold)
    records = []
    for image_id in test_ids:
        rgb = load_rgb(image_path(TEST_DIR, image_id))
        result = annotate_image(processor, rgb, args.prompt, device)
        records.append(
            {"image_id": image_id, "height": int(rgb.shape[0]), "width": int(rgb.shape[1]), **result}
        )
    del processor, model
    if device.startswith("cuda"):
        torch.cuda.empty_cache()

    cache_path.write_text(
        json.dumps({"prompt": args.prompt, "threshold": args.threshold, "records": records})
    )
    (predictions_dir / f"{model_key}.coco.json").write_text(
        json.dumps(build_coco(records, args.prompt))
    )
    print(f"[{model_key}] predicted {len(records)} test images")
    return records


def group_by_type(per_image: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = {image_type.value: [] for image_type in ImageType}
    for record in per_image:
        groups[record["image_type"]].append(record)
    return {name: aggregate_metrics(records) for name, records in groups.items()}


def training_summary(model_dir: Path) -> dict[str, Any] | None:
    metrics_path = model_dir / "metrics.json"
    history_path = model_dir / "history.json"
    summary_path = model_dir / "run_summary.json"
    if not history_path.exists():
        return None
    run_summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    final = json.loads(metrics_path.read_text()) if metrics_path.exists() else {}
    return {
        "history": json.loads(history_path.read_text()),
        "best_epoch": final.get("best_epoch"),
        "selection_metric": run_summary.get("selection_metric"),
        "training_seconds": final.get("training_seconds"),
        "config": {
            key: run_summary.get(key)
            for key in (
                "dataset",
                "train_image_count",
                "val_image_count",
                "epochs",
                "learning_rate",
                "lora_rank",
                "lora_alpha",
                "lora_module_count",
                "trainable_parameters",
                "total_parameters",
                "resolution",
                "hflip",
                "skip_empty_images",
                "gpu",
            )
        },
    }


def pseudo_label_summary() -> dict[str, Any] | None:
    quality_path = SAM3_ANNOTATIONS_DIR / "stage1_train_blobs.quality.json"
    manifest_path = SAM3_ANNOTATIONS_DIR / "stage1_train_blobs.manifest.json"
    if not quality_path.exists() or not manifest_path.exists():
        return None
    quality = json.loads(quality_path.read_text())
    manifest = json.loads(manifest_path.read_text())
    return {
        "quality": {key: quality[key] for key in ("overall", "by_image_type", "by_split")},
        "annotation": {
            key: manifest[key]
            for key in (
                "prompt",
                "confidence_threshold",
                "checkpoint_path",
                "device",
                "gpu",
                "images",
                "annotations",
                "images_without_detections",
                "images_at_query_cap",
                "timing",
            )
        },
    }


def main() -> None:
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    split = load_split()
    test_ids = split["test"]
    image_types = split["image_types"]
    gt_labels = {image_id: load_ground_truth_labels(TEST_DIR, image_id) for image_id in test_ids}

    models: dict[str, Any] = {}
    per_image: dict[str, dict[str, Any]] = {
        image_id: {
            "image_type": image_types[image_id],
            "gt_count": int(gt_labels[image_id].max()),
            "models": {},
        }
        for image_id in test_ids
    }

    for model_key in args.models:
        spec = MODEL_SPECS[model_key]
        checkpoint = lora_checkpoint(spec)
        if checkpoint is not None and not checkpoint.exists():
            print(f"[{model_key}] skipped: {checkpoint} not found")
            continue

        started = time.perf_counter()
        records = predict_test_set(args, model_key, test_ids, device)
        model_per_image = []
        for record in records:
            metrics = compute_image_metrics(gt_labels[record["image_id"]], record_label_map(record))
            entry = {**metrics.to_dict(), "inference_seconds": record["seconds"]}
            per_image[record["image_id"]]["models"][model_key] = entry
            model_per_image.append({**entry, "image_type": image_types[record["image_id"]]})

        models[model_key] = {
            "label": spec["label"],
            "description": spec["description"],
            "lora_checkpoint": None if checkpoint is None else str(checkpoint),
            "aggregate": aggregate_metrics(model_per_image),
            "by_image_type": group_by_type(model_per_image),
            "inference_seconds_mean": float(np.mean([r["seconds"] for r in records])),
        }
        print(
            f"[{model_key}] dsb={models[model_key]['aggregate']['dsb_score']:.4f} "
            f"f1@0.5={models[model_key]['aggregate']['f1_at_50']:.4f} "
            f"({time.perf_counter() - started:.1f}s)"
        )

    type_counts = {
        image_type.value: sum(image_types[image_id] == image_type.value for image_id in test_ids)
        for image_type in ImageType
    }
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "prompt": args.prompt,
        "confidence_threshold": args.threshold,
        "test_set": {
            "name": "stage1_test",
            "ground_truth": "stage1_solution.csv",
            "images": len(test_ids),
            "nuclei": int(sum(per_image[image_id]["gt_count"] for image_id in test_ids)),
            "image_types": type_counts,
        },
        "split_counts": split["counts"],
        "metric_definitions": {
            "dsb_score": "Official DSB 2018 score: mean over IoU thresholds 0.50-0.95 (step 0.05) of TP/(TP+FP+FN), averaged over images.",
            "f1_at_50": "Instance F1 with one-to-one matching at IoU > 0.5.",
            "pixel_dice": "Foreground Dice of the union of instances.",
            "pixel_iou": "Foreground IoU of the union of instances.",
            "instance_mean_iou": "Mean over ground-truth nuclei of the best-matching predicted IoU (misses count as 0).",
            "abs_count_error": "Mean absolute difference between predicted and true nucleus counts per image.",
            "mean_count_error": "Mean signed count difference (negative = undercounting).",
        },
        "iou_thresholds": list(DSB_IOU_THRESHOLDS),
        "models": models,
        "training": {
            key: training_summary(MODELS_DIR / spec["lora_dir"])
            for key, spec in MODEL_SPECS.items()
            if spec["lora_dir"] is not None
        },
        "pseudo_labels": pseudo_label_summary(),
        "per_image": per_image,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2))
    print(f"[evaluate] wrote {args.output}")


if __name__ == "__main__":
    main()
