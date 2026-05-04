#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import shutil
from io import BytesIO
from pathlib import Path
import sys
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = REPO_ROOT / "datasets" / "PanNuke" / "data"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "datasets" / "PanNuke"
CATEGORY_NAMES = (
    "Neoplastic",
    "Inflammatory",
    "Connective",
    "Dead",
    "Epithelial",
)
TISSUE_NAMES = (
    "Adrenal Gland",
    "Bile Duct",
    "Bladder",
    "Breast",
    "Cervix",
    "Colon",
    "Esophagus",
    "Head & Neck",
    "Kidney",
    "Liver",
    "Lung",
    "Ovarian",
    "Pancreatic",
    "Prostate",
    "Skin",
    "Stomach",
    "Testis",
    "Thyroid",
    "Uterus",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Unpack the official PanNuke parquet folds into JPEG images and one single-channel "
            "label map per image."
        )
    )
    parser.add_argument(
        "--data-dir",
        default=str(DEFAULT_DATA_DIR),
        help=f"Directory containing PanNuke parquet fold files. Defaults to: {DEFAULT_DATA_DIR}",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help=(
            "Directory where images/, labels/, and metadata will be created. "
            f"Defaults to: {DEFAULT_OUTPUT_DIR}"
        ),
    )
    parser.add_argument(
        "--label-format",
        choices=("png", "jpg"),
        default="png",
        help=(
            "Label image format. Defaults to png because JPEG is lossy and can alter label IDs."
        ),
    )
    parser.add_argument(
        "--image-quality",
        type=int,
        default=95,
        help="JPEG quality for exported images. Defaults to 95.",
    )
    parser.add_argument(
        "--label-quality",
        type=int,
        default=100,
        help="JPEG quality for exported labels when --label-format=jpg. Defaults to 100.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Rows to decode from parquet at a time. Defaults to 32.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optionally export only the first N images across all folds.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace any existing images/, labels/, and metadata files in the output directory.",
    )
    return parser.parse_args()


def load_pillow():
    try:
        from PIL import Image
    except ImportError as error:
        raise SystemExit(
            "Pillow is required for this script. Install it with 'python -m pip install Pillow'."
        ) from error

    return Image


def load_pyarrow_parquet_module():
    try:
        import pyarrow.parquet as pq
    except ImportError as error:
        raise SystemExit(
            "pyarrow is required for this script. Install it with 'python -m pip install pyarrow'."
        ) from error

    return pq


def validate_args(args: argparse.Namespace) -> tuple[Path, Path]:
    data_dir = Path(args.data_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()

    if not data_dir.is_dir():
        raise SystemExit(f"PanNuke data directory does not exist: {data_dir}")

    parquet_files = sorted(data_dir.glob("fold*-*.parquet"))
    if not parquet_files:
        raise SystemExit(f"No PanNuke parquet fold files were found in: {data_dir}")

    if not 1 <= args.image_quality <= 95:
        raise SystemExit("--image-quality must be between 1 and 95.")

    if not 1 <= args.label_quality <= 100:
        raise SystemExit("--label-quality must be between 1 and 100.")

    if args.batch_size <= 0:
        raise SystemExit("--batch-size must be greater than 0.")

    if args.limit is not None and args.limit <= 0:
        raise SystemExit("--limit must be greater than 0 when provided.")

    if args.label_format == "jpg":
        print(
            "Warning: JPEG label maps are lossy and may change instance IDs. "
            "Use --label-format png if the labels will be used for training or scoring.",
            file=sys.stderr,
        )

    return data_dir, output_dir


def remove_path(path: Path) -> None:
    if path.is_dir():
        shutil.rmtree(path)
    elif path.exists():
        path.unlink()


def prepare_output_dir(
    output_dir: Path, overwrite: bool
) -> tuple[Path, Path, Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)

    images_dir = output_dir / "images"
    labels_dir = output_dir / "labels"
    manifest_path = output_dir / "label_metadata.csv"
    summary_path = output_dir / "unpack_summary.json"

    existing_paths = [
        path
        for path in (images_dir, labels_dir, manifest_path, summary_path)
        if path.exists()
    ]
    if existing_paths and not overwrite:
        joined = ", ".join(str(path) for path in existing_paths)
        raise SystemExit(
            f"Output paths already exist: {joined}. Use --overwrite to replace them."
        )

    for path in existing_paths:
        remove_path(path)

    images_dir.mkdir(parents=True, exist_ok=False)
    labels_dir.mkdir(parents=True, exist_ok=False)
    return images_dir, labels_dir, manifest_path, summary_path


def build_output_stem(fold_name: str, row_index: int) -> str:
    return f"{fold_name}_{row_index:05d}"


def normalize_named_label(value: Any, names: tuple[str, ...]) -> tuple[int | None, str]:
    if value is None:
        return None, ""

    if isinstance(value, str):
        try:
            return names.index(value), value
        except ValueError:
            return None, value

    numeric_value = int(value)
    if 0 <= numeric_value < len(names):
        return numeric_value, names[numeric_value]
    return numeric_value, str(numeric_value)


def extract_record_bytes(record: Any) -> bytes:
    if isinstance(record, (bytes, bytearray)):
        return bytes(record)

    if not isinstance(record, dict):
        raise ValueError(f"Unsupported image record type: {type(record).__name__}")

    payload = record.get("bytes")
    if payload is not None:
        return bytes(payload)

    path = record.get("path")
    if path:
        file_path = Path(path)
        if file_path.is_file():
            return file_path.read_bytes()

    raise ValueError("Image record did not contain readable bytes or a local path.")


def decode_image(record: Any, image_module, mode: str) -> Any:
    with image_module.open(BytesIO(extract_record_bytes(record))) as image:
        return image.convert(mode)


def iter_pannuke_rows(data_dir: Path, batch_size: int):
    pq = load_pyarrow_parquet_module()

    for parquet_path in sorted(data_dir.glob("fold*-*.parquet")):
        fold_name = parquet_path.stem.split("-", 1)[0]
        row_index = 0
        parquet_file = pq.ParquetFile(parquet_path)
        for batch in parquet_file.iter_batches(batch_size=batch_size):
            for row in batch.to_pylist():
                yield fold_name, row_index, row
                row_index += 1


def build_label_map(
    *,
    instance_records: list[Any],
    categories: list[Any],
    height: int,
    width: int,
    label_format: str,
    image_module,
) -> tuple[np.ndarray, list[dict[str, Any]], int]:
    if label_format == "jpg" and len(instance_records) > 255:
        raise ValueError(
            f"Found {len(instance_records)} instances in one image. "
            "A single-channel JPEG label map cannot encode more than 255 distinct instance IDs."
        )

    if len(instance_records) > 65535:
        raise ValueError(
            f"Found {len(instance_records)} instances in one image. "
            "This exporter supports at most 65535 distinct instance IDs in a single label map."
        )

    if categories and len(categories) != len(instance_records):
        raise ValueError(
            f"Instance/category length mismatch: {len(instance_records)} instances vs {len(categories)} categories."
        )

    label_dtype = np.uint16 if len(instance_records) > 255 else np.uint8
    label_map = np.zeros((height, width), dtype=label_dtype)
    metadata_rows: list[dict[str, Any]] = []
    overlapping_pixels = 0

    for pixel_value, instance_record in enumerate(instance_records, start=1):
        mask_image = decode_image(instance_record, image_module, "1")
        mask_array = np.asarray(mask_image, dtype=np.uint8) > 0
        if mask_array.shape != (height, width):
            raise ValueError(
                f"Mask shape {mask_array.shape} does not match image shape {(height, width)}."
            )

        overlap = mask_array & (label_map > 0)
        overlapping_pixels += int(overlap.sum())
        label_map[mask_array] = pixel_value

        category_id, category_name = normalize_named_label(
            categories[pixel_value - 1] if categories else None,
            CATEGORY_NAMES,
        )
        metadata_rows.append(
            {
                "pixel_value": pixel_value,
                "category_id": "" if category_id is None else category_id,
                "category_name": category_name,
                "area": int(mask_array.sum()),
            }
        )

    return label_map, metadata_rows, overlapping_pixels


def save_label_map(
    label_map: np.ndarray,
    output_path: Path,
    label_format: str,
    label_quality: int,
    image_module,
) -> None:
    if label_format == "jpg" and label_map.dtype != np.uint8:
        raise ValueError("JPEG label export only supports 8-bit label maps.")

    label_image = image_module.fromarray(label_map)
    if label_format == "jpg":
        label_image.save(output_path, format="JPEG", quality=label_quality)
        return

    label_image.save(output_path, format="PNG")


def export_dataset(args: argparse.Namespace) -> dict[str, Any]:
    image_module = load_pillow()
    data_dir, output_dir = validate_args(args)
    images_dir, labels_dir, manifest_path, summary_path = prepare_output_dir(
        output_dir, args.overwrite
    )

    manifest_fields = [
        "image_filename",
        "label_filename",
        "fold",
        "row_index",
        "tissue_id",
        "tissue_name",
        "instance_count",
        "overlapping_pixels",
        "pixel_value",
        "category_id",
        "category_name",
        "area",
    ]

    total_images = 0
    total_instances = 0
    total_overlapping_pixels = 0
    fold_counts: dict[str, int] = {}

    with manifest_path.open("w", newline="", encoding="utf-8") as manifest_file:
        writer = csv.DictWriter(manifest_file, fieldnames=manifest_fields)
        writer.writeheader()

        for fold_name, row_index, row in iter_pannuke_rows(data_dir, args.batch_size):
            if args.limit is not None and total_images >= args.limit:
                break

            image = decode_image(row["image"], image_module, "RGB")
            width, height = image.size
            categories = list(row.get("categories") or [])
            instances = list(row.get("instances") or [])
            label_map, metadata_rows, overlapping_pixels = build_label_map(
                instance_records=instances,
                categories=categories,
                height=height,
                width=width,
                label_format=args.label_format,
                image_module=image_module,
            )

            stem = build_output_stem(fold_name, row_index)
            image_filename = f"{stem}.jpg"
            label_filename = f"{stem}.{args.label_format}"
            image_output_path = images_dir / image_filename
            label_output_path = labels_dir / label_filename

            image.save(image_output_path, format="JPEG", quality=args.image_quality)
            save_label_map(
                label_map,
                label_output_path,
                args.label_format,
                args.label_quality,
                image_module,
            )

            tissue_id, tissue_name = normalize_named_label(
                row.get("tissue"), TISSUE_NAMES
            )
            for metadata_row in metadata_rows:
                writer.writerow(
                    {
                        "image_filename": image_filename,
                        "label_filename": label_filename,
                        "fold": fold_name,
                        "row_index": row_index,
                        "tissue_id": "" if tissue_id is None else tissue_id,
                        "tissue_name": tissue_name,
                        "instance_count": len(metadata_rows),
                        "overlapping_pixels": overlapping_pixels,
                        **metadata_row,
                    }
                )

            total_images += 1
            total_instances += len(metadata_rows)
            total_overlapping_pixels += overlapping_pixels
            fold_counts[fold_name] = fold_counts.get(fold_name, 0) + 1

            if total_images % 200 == 0:
                print(f"Exported {total_images} images...", file=sys.stderr)

    summary = {
        "data_dir": str(data_dir),
        "output_dir": str(output_dir),
        "images_dir": str(images_dir),
        "labels_dir": str(labels_dir),
        "label_format": args.label_format,
        "image_quality": args.image_quality,
        "label_quality": args.label_quality,
        "images_exported": total_images,
        "instances_exported": total_instances,
        "overlapping_pixels": total_overlapping_pixels,
        "fold_counts": fold_counts,
        "manifest_path": str(manifest_path),
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> int:
    args = parse_args()
    summary = export_dataset(args)
    print(
        "Exported "
        f"{summary['images_exported']} image(s) and {summary['instances_exported']} instance label(s) "
        f"to {summary['output_dir']}"
    )
    print(f"Images: {summary['images_dir']}")
    print(f"Labels: {summary['labels_dir']}")
    print(f"Metadata: {summary['manifest_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
