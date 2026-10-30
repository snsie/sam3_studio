#!/usr/bin/env python3
"""Auto-annotate BBBC038 with SAM3 text-prompt grounding.

Runs ``Sam3Processor.set_image`` + ``Sam3Processor.set_text_prompt`` (the backend's
text-grounding path) with one prompt on every image of a stage1 folder and writes:

  <run>.coco.json      instance masks as COCO RLE, with a score per annotation and
                       per-image detection statistics on each image entry
  <run>.manifest.json  prompt, threshold, checkpoint, device, timing
  <run>.quality.json   pseudo-label quality vs. ground truth (stage1_train only)

Progress is appended to <run>.partial.jsonl after every image, so an interrupted run
resumes where it stopped.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pycocotools.mask as mask_utils
import torch
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "backend"))

from app.core.config import get_settings  # noqa: E402
from app.services.bbbc038_data import (  # noqa: E402
    RAW_DIR,
    SAM3_ANNOTATIONS_DIR,
    image_path,
    list_image_ids,
    load_ground_truth_labels,
    load_rgb,
    load_split,
    masks_to_label_map,
    suppress_duplicate_masks,
)
from app.services.nuclei_metrics import aggregate_metrics, compute_image_metrics  # noqa: E402

# SAM3's detector has a fixed number of object queries, so an image can never yield
# more instances than this. Images that hit the cap are flagged in the manifest.
SAM3_NUM_QUERIES = 200


def parse_args() -> argparse.Namespace:
    settings = get_settings()
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--source", choices=["stage1_train", "stage1_test"], default="stage1_train"
    )
    parser.add_argument("--prompt", default="blobs")
    parser.add_argument(
        "--threshold",
        type=float,
        default=settings.sam3_confidence_threshold,
        help="Sam3Processor confidence threshold (defaults to SAM3_CONFIDENCE_THRESHOLD).",
    )
    parser.add_argument("--output-dir", type=Path, default=SAM3_ANNOTATIONS_DIR)
    parser.add_argument(
        "--run-name", default=None, help="Defaults to <source>_<prompt>."
    )
    parser.add_argument("--checkpoint-path", default=settings.sam3_checkpoint_path)
    parser.add_argument("--bpe-path", default=settings.sam3_bpe_path)
    parser.add_argument(
        "--load-from-hf",
        action=argparse.BooleanOptionalAction,
        default=settings.sam3_load_from_hf,
    )
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--limit", type=int, default=None, help="Only annotate N images.")
    return parser.parse_args()


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def build_processor(args: argparse.Namespace, device: str):
    from sam3 import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor

    model = build_sam3_image_model(
        checkpoint_path=args.checkpoint_path,
        bpe_path=args.bpe_path,
        device=device,
        eval_mode=True,
        load_from_HF=args.load_from_hf,
        enable_inst_interactivity=False,
        compile=False,
    )
    return Sam3Processor(model, device=device, confidence_threshold=args.threshold)


def inference_context(device: str):
    if device.startswith("cuda"):
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return torch.autocast(device_type="cpu", enabled=False)


def encode_instances(state: dict[str, Any]) -> list[dict[str, Any]]:
    masks = state["masks"].detach().cpu().numpy()
    scores = state["scores"].detach().float().cpu().numpy()
    if masks.ndim == 4:
        masks = masks[:, 0]

    instances: list[dict[str, Any]] = []
    for mask, score in zip(masks, scores):
        mask = mask.astype(bool)
        area = int(mask.sum())
        if area == 0:
            continue
        rle = mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))
        bbox = [float(value) for value in mask_utils.toBbox(rle)]
        rle["counts"] = rle["counts"].decode("ascii")
        instances.append(
            {"segmentation": rle, "score": float(score), "bbox": bbox, "area": area}
        )
    return instances


def annotate_image(processor, rgb: np.ndarray, prompt: str, device: str) -> dict[str, Any]:
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    started = time.perf_counter()
    with torch.inference_mode(), inference_context(device):
        state = processor.set_image(Image.fromarray(rgb))
        state = processor.set_text_prompt(prompt, state)
    instances = encode_instances(state)
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    return {"instances": instances, "seconds": time.perf_counter() - started}


def load_partial(partial_path: Path) -> dict[str, dict[str, Any]]:
    done: dict[str, dict[str, Any]] = {}
    if not partial_path.exists():
        return done
    with partial_path.open() as handle:
        for line in handle:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                # A run killed mid-write leaves a truncated last line; redo that image.
                continue
            done[record["image_id"]] = record
    return done


def build_coco(records: list[dict[str, Any]], prompt: str) -> dict[str, Any]:
    images = []
    annotations = []
    annotation_id = 1
    for image_index, record in enumerate(records, start=1):
        scores = [instance["score"] for instance in record["instances"]]
        images.append(
            {
                "id": image_index,
                "image_key": record["image_id"],
                "file_name": f"{record['image_id']}/images/{record['image_id']}.png",
                "width": record["width"],
                "height": record["height"],
                "num_detections": len(scores),
                "mean_score": float(np.mean(scores)) if scores else None,
                "max_score": float(np.max(scores)) if scores else None,
                "inference_seconds": record["seconds"],
            }
        )
        for instance in record["instances"]:
            annotations.append(
                {
                    "id": annotation_id,
                    "image_id": image_index,
                    "category_id": 1,
                    "segmentation": instance["segmentation"],
                    "bbox": instance["bbox"],
                    "area": instance["area"],
                    "score": instance["score"],
                    "iscrowd": 0,
                }
            )
            annotation_id += 1
    return {
        "info": {"description": f"SAM3 pseudo-labels for prompt '{prompt}'"},
        "images": images,
        "annotations": annotations,
        "categories": [{"id": 1, "name": "nucleus"}],
    }


def record_label_map(record: dict[str, Any]) -> np.ndarray:
    """Instance label map with the backend's duplicate suppression applied."""
    masks = [
        mask_utils.decode(instance["segmentation"]).astype(bool)
        for instance in record["instances"]
    ]
    scores = [instance["score"] for instance in record["instances"]]
    masks, scores = suppress_duplicate_masks(masks, scores)
    return masks_to_label_map(masks, (record["height"], record["width"]), scores)


def evaluate_against_ground_truth(
    records: list[dict[str, Any]], source_dir: Path
) -> dict[str, Any]:
    split = load_split()
    image_types = split["image_types"]
    split_of = {image_id: "train" for image_id in split["train"]}
    split_of.update({image_id: "val" for image_id in split["val"]})

    per_image: list[dict[str, Any]] = []
    for record in records:
        gt_labels = load_ground_truth_labels(source_dir, record["image_id"])
        metrics = compute_image_metrics(gt_labels, record_label_map(record)).to_dict()
        metrics["image_id"] = record["image_id"]
        metrics["image_type"] = image_types[record["image_id"]]
        metrics["split"] = split_of.get(record["image_id"], "unknown")
        per_image.append(metrics)

    def group(key: str) -> dict[str, Any]:
        groups: dict[str, list[dict[str, Any]]] = {}
        for metrics in per_image:
            groups.setdefault(metrics[key], []).append(metrics)
        return {name: aggregate_metrics(items) for name, items in sorted(groups.items())}

    return {
        "overall": aggregate_metrics(per_image),
        "by_image_type": group("image_type"),
        "by_split": group("split"),
        "per_image": per_image,
    }


def main() -> None:
    args = parse_args()
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    source_dir = RAW_DIR / args.source
    run_name = args.run_name or f"{args.source}_{args.prompt.replace(' ', '_')}"
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    partial_path = output_dir / f"{run_name}.partial.jsonl"
    manifest_path = output_dir / f"{run_name}.manifest.json"

    image_ids = list_image_ids(source_dir)
    if args.limit is not None:
        image_ids = image_ids[: args.limit]

    done = load_partial(partial_path)
    pending = [image_id for image_id in image_ids if image_id not in done]
    previous_manifest = (
        json.loads(manifest_path.read_text()) if manifest_path.exists() and done else {}
    )
    sessions: list[dict[str, Any]] = previous_manifest.get("sessions", [])
    print(f"[annotate] {len(done)} done, {len(pending)} pending in {source_dir}")

    session: dict[str, Any] = {"started_at": utcnow(), "images": 0}
    if pending:
        load_started = time.perf_counter()
        processor = build_processor(args, device)
        session["model_load_seconds"] = time.perf_counter() - load_started
        loop_started = time.perf_counter()

        with partial_path.open("a+") as handle:
            # Terminate a line truncated by an interrupted run before appending.
            if handle.tell() > 0:
                handle.seek(handle.tell() - 1)
                if handle.read(1) != "\n":
                    handle.write("\n")
            for index, image_id in enumerate(pending, start=1):
                rgb = load_rgb(image_path(source_dir, image_id))
                result = annotate_image(processor, rgb, args.prompt, device)
                record = {
                    "image_id": image_id,
                    "height": int(rgb.shape[0]),
                    "width": int(rgb.shape[1]),
                    **result,
                }
                handle.write(json.dumps(record) + "\n")
                handle.flush()
                done[image_id] = record
                session["images"] += 1
                if index % 25 == 0 or index == len(pending):
                    print(
                        f"[annotate] {index}/{len(pending)} {image_id[:12]} "
                        f"n={len(result['instances'])} {result['seconds']:.2f}s",
                        flush=True,
                    )
        session["loop_seconds"] = time.perf_counter() - loop_started
        session["finished_at"] = utcnow()
        sessions.append(session)
        del processor
        if device.startswith("cuda"):
            torch.cuda.empty_cache()

    records = [done[image_id] for image_id in image_ids]
    coco = build_coco(records, args.prompt)
    coco_path = output_dir / f"{run_name}.coco.json"
    coco_path.write_text(json.dumps(coco))

    per_image_seconds = [record["seconds"] for record in records]
    detections = [len(record["instances"]) for record in records]
    manifest = {
        "run_name": run_name,
        "source": args.source,
        "image_root": str(source_dir),
        "prompt": args.prompt,
        "confidence_threshold": args.threshold,
        "inference_path": "Sam3Processor.set_image + Sam3Processor.set_text_prompt (bf16 autocast on CUDA)",
        "checkpoint_path": args.checkpoint_path,
        "bpe_path": args.bpe_path,
        "load_from_hf": args.load_from_hf,
        "device": device,
        "gpu": torch.cuda.get_device_name(0) if device.startswith("cuda") else None,
        "torch_version": torch.__version__,
        "python_version": platform.python_version(),
        "images": len(records),
        "annotations": len(coco["annotations"]),
        "images_without_detections": int(sum(count == 0 for count in detections)),
        "images_at_query_cap": int(sum(count >= SAM3_NUM_QUERIES for count in detections)),
        "timing": {
            "inference_seconds_total": float(np.sum(per_image_seconds)),
            "inference_seconds_mean": float(np.mean(per_image_seconds)),
            "inference_seconds_median": float(np.median(per_image_seconds)),
            "wall_seconds_total": float(
                sum(
                    item.get("model_load_seconds", 0.0) + item.get("loop_seconds", 0.0)
                    for item in sessions
                )
            ),
        },
        "sessions": sessions,
        "coco_path": str(coco_path),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"[annotate] wrote {coco_path}")
    print(json.dumps({key: manifest[key] for key in ("images", "annotations", "timing")}, indent=2))

    if args.source == "stage1_train":
        quality = evaluate_against_ground_truth(records, source_dir)
        quality_path = output_dir / f"{run_name}.quality.json"
        quality_path.write_text(json.dumps(quality, indent=2))
        summary = {
            name: {
                key: round(value, 4)
                for key, value in metrics.items()
                if key in ("dsb_score", "f1_at_50", "instance_mean_iou", "pixel_iou")
            }
            for name, metrics in {"overall": quality["overall"], **quality["by_image_type"]}.items()
        }
        print(f"[quality] wrote {quality_path}\n{json.dumps(summary, indent=2)}")


if __name__ == "__main__":
    main()
