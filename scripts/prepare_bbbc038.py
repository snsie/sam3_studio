#!/usr/bin/env python3
"""Stage BBBC038 (DSB 2018 stage1) for the nuclei demo.

1. Extracts stage1_train.zip / stage1_test.zip into backend/data/datasets/bbbc038/raw/.
2. Checks the RLE decoder against every stage1_train mask PNG.
3. Decodes stage1_solution.csv into per-nucleus PNGs under raw/stage1_test/<id>/masks/.
4. Classifies every image (fluorescent / brightfield / histology) from color statistics
   and writes a seeded, type-stratified train/val split of stage1_train to split.json.
5. Writes the stage1_train ground truth as COCO RLE (ground_truth/stage1_train.coco.json)
   for the ground-truth-trained upper-bound reference model.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import re
import sys
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pycocotools.mask as mask_utils
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "backend"))

from app.services.bbbc038_data import (  # noqa: E402
    DATASET_DIR,
    RAW_DIR,
    SPLIT_PATH,
    TEST_DIR,
    TRAIN_DIR,
    ImageType,
    classify_image_type,
    compute_color_stats,
    image_path,
    list_image_ids,
    load_instance_masks,
    load_rgb,
    masks_dir,
    rle_decode,
)

ZIP_MEMBER_PATTERN = re.compile(r"^[0-9a-f]{64}/((images|masks)/([0-9a-f]{64}\.png)?)?$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--train-zip", type=Path, default=REPO_ROOT / "stage1_train.zip")
    parser.add_argument("--test-zip", type=Path, default=REPO_ROOT / "stage1_test.zip")
    parser.add_argument(
        "--train-labels", type=Path, default=REPO_ROOT / "stage1_train_labels.csv"
    )
    parser.add_argument("--solution", type=Path, default=REPO_ROOT / "stage1_solution.csv")
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=2018)
    parser.add_argument(
        "--overwrite-test-masks",
        action="store_true",
        help="Rewrite decoded stage1_test masks even if they already exist.",
    )
    return parser.parse_args()


def extract_zip(zip_path: Path, destination: Path) -> None:
    if destination.is_dir() and any(destination.iterdir()):
        print(f"[extract] {destination} already populated, skipping")
        return
    with zipfile.ZipFile(zip_path) as archive:
        unexpected = [
            name for name in archive.namelist() if not ZIP_MEMBER_PATTERN.match(name)
        ]
        if unexpected:
            raise ValueError(f"{zip_path} has unexpected members: {unexpected[:5]}")
        destination.mkdir(parents=True, exist_ok=True)
        archive.extractall(destination)
    print(f"[extract] {zip_path.name} -> {destination}")


def read_rle_rows(csv_path: Path) -> dict[str, list[dict[str, str]]]:
    rows: dict[str, list[dict[str, str]]] = defaultdict(list)
    with csv_path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            rows[row["ImageId"]].append(row)
    return rows


def mask_digest(mask: np.ndarray) -> str:
    return hashlib.sha1(np.packbits(mask).tobytes() + str(mask.shape).encode()).hexdigest()


def verify_train_decoder(labels_csv: Path) -> dict:
    rows_by_image = read_rle_rows(labels_csv)
    image_ids = list_image_ids(TRAIN_DIR)
    exact = 0
    total_csv = 0
    total_png = 0
    mismatched_images: list[str] = []

    for image_id in image_ids:
        with Image.open(image_path(TRAIN_DIR, image_id)) as image:
            height, width = image.height, image.width
        png_digests = Counter(
            mask_digest(mask) for mask in load_instance_masks(TRAIN_DIR, image_id)
        )
        csv_digests = Counter(
            mask_digest(rle_decode(row["EncodedPixels"], height, width))
            for row in rows_by_image.get(image_id, [])
        )
        total_png += sum(png_digests.values())
        total_csv += sum(csv_digests.values())
        exact += sum((png_digests & csv_digests).values())
        if png_digests != csv_digests:
            mismatched_images.append(image_id)

    report = {
        "images": len(image_ids),
        "png_masks": total_png,
        "csv_masks": total_csv,
        "exact_matches": exact,
        "mismatched_images": mismatched_images,
    }
    print(f"[verify] {json.dumps(report)}")
    if exact == 0 or len(mismatched_images) > len(image_ids) * 0.01:
        raise RuntimeError("RLE decoder does not reproduce the stage1_train mask PNGs.")
    return report


def decode_test_solution(solution_csv: Path, overwrite: bool) -> dict:
    rows_by_image = read_rle_rows(solution_csv)
    test_ids = list_image_ids(TEST_DIR)
    missing = sorted(set(test_ids) - set(rows_by_image))
    if missing:
        raise RuntimeError(f"stage1_solution.csv has no rows for: {missing}")

    nuclei = 0
    overlapping_pixels = 0
    for image_id in test_ids:
        output_dir = masks_dir(TEST_DIR, image_id)
        if output_dir.is_dir() and any(output_dir.glob("*.png")) and not overwrite:
            nuclei += len(list(output_dir.glob("*.png")))
            continue
        with Image.open(image_path(TEST_DIR, image_id)) as image:
            height, width = image.height, image.width
        output_dir.mkdir(parents=True, exist_ok=True)
        for stale in output_dir.glob("*.png"):
            stale.unlink()

        coverage = np.zeros((height, width), dtype=np.int32)
        for index, row in enumerate(rows_by_image[image_id]):
            if (int(row["Height"]), int(row["Width"])) != (height, width):
                raise RuntimeError(
                    f"{image_id}: solution size {row['Height']}x{row['Width']} "
                    f"differs from image {height}x{width}"
                )
            mask = rle_decode(row["EncodedPixels"], height, width)
            coverage += mask
            Image.fromarray(mask.astype(np.uint8) * 255).save(
                output_dir / f"{index:04d}.png"
            )
            nuclei += 1
        overlapping_pixels += int((coverage > 1).sum())

    report = {
        "images": len(test_ids),
        "nuclei": nuclei,
        "overlapping_pixels": overlapping_pixels,
    }
    print(f"[solution] {json.dumps(report)}")
    return report


def write_ground_truth_coco(output_path: Path) -> dict:
    images = []
    annotations = []
    for image_index, image_id in enumerate(list_image_ids(TRAIN_DIR), start=1):
        masks = load_instance_masks(TRAIN_DIR, image_id)
        height, width = masks[0].shape
        images.append(
            {
                "id": image_index,
                "image_key": image_id,
                "file_name": f"{image_id}/images/{image_id}.png",
                "width": width,
                "height": height,
            }
        )
        for mask in masks:
            rle = mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))
            bbox = [float(value) for value in mask_utils.toBbox(rle)]
            rle["counts"] = rle["counts"].decode("ascii")
            annotations.append(
                {
                    "id": len(annotations) + 1,
                    "image_id": image_index,
                    "category_id": 1,
                    "segmentation": rle,
                    "bbox": bbox,
                    "area": int(mask.sum()),
                    "iscrowd": 0,
                }
            )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(
            {
                "info": {"description": "BBBC038 stage1_train ground truth"},
                "images": images,
                "annotations": annotations,
                "categories": [{"id": 1, "name": "nucleus"}],
            }
        )
    )
    report = {"path": str(output_path), "images": len(images), "annotations": len(annotations)}
    print(f"[gt-coco] {json.dumps(report)}")
    return report


def stratified_split(
    image_types: dict[str, str], val_fraction: float, seed: int
) -> tuple[list[str], list[str]]:
    rng = random.Random(seed)
    train_ids: list[str] = []
    val_ids: list[str] = []
    by_type: dict[str, list[str]] = defaultdict(list)
    for image_id, image_type in sorted(image_types.items()):
        by_type[image_type].append(image_id)
    for image_type in sorted(by_type):
        ids = by_type[image_type]
        rng.shuffle(ids)
        val_count = max(1, round(len(ids) * val_fraction))
        val_ids.extend(ids[:val_count])
        train_ids.extend(ids[val_count:])
    return sorted(train_ids), sorted(val_ids)


def main() -> None:
    args = parse_args()
    extract_zip(args.train_zip, TRAIN_DIR)
    extract_zip(args.test_zip, TEST_DIR)

    decoder_report = verify_train_decoder(args.train_labels)
    solution_report = decode_test_solution(args.solution, args.overwrite_test_masks)

    image_types: dict[str, dict[str, str]] = {}
    color_stats: dict[str, dict[str, float]] = {}
    for split_name, split_dir in (("stage1_train", TRAIN_DIR), ("stage1_test", TEST_DIR)):
        image_types[split_name] = {}
        for image_id in list_image_ids(split_dir):
            rgb = load_rgb(image_path(split_dir, image_id))
            stats = compute_color_stats(rgb)
            image_types[split_name][image_id] = classify_image_type(rgb).value
            color_stats[image_id] = {
                "channel_diff": round(stats.channel_diff, 5),
                "median_intensity": round(stats.median_intensity, 5),
            }

    train_ids, val_ids = stratified_split(
        image_types["stage1_train"], args.val_fraction, args.seed
    )
    test_ids = sorted(image_types["stage1_test"])

    def type_counts(ids: list[str], split_name: str) -> dict[str, int]:
        counts = Counter(image_types[split_name][image_id] for image_id in ids)
        return {image_type.value: counts.get(image_type.value, 0) for image_type in ImageType}

    payload = {
        "seed": args.seed,
        "val_fraction": args.val_fraction,
        "stratify_by": "image_type (color statistics: channel difference, median intensity)",
        "raw_dir": str(RAW_DIR),
        "counts": {
            "train": type_counts(train_ids, "stage1_train"),
            "val": type_counts(val_ids, "stage1_train"),
            "test": type_counts(test_ids, "stage1_test"),
        },
        "train": train_ids,
        "val": val_ids,
        "test": test_ids,
        "image_types": {**image_types["stage1_train"], **image_types["stage1_test"]},
        "color_stats": color_stats,
        "decoder_check": decoder_report,
        "solution_decode": solution_report,
    }
    SPLIT_PATH.parent.mkdir(parents=True, exist_ok=True)
    SPLIT_PATH.write_text(json.dumps(payload, indent=2))
    print(f"[split] wrote {SPLIT_PATH}: {json.dumps(payload['counts'])}")

    write_ground_truth_coco(DATASET_DIR / "ground_truth" / "stage1_train.coco.json")


if __name__ == "__main__":
    main()
