#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
from pathlib import Path
import sys

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_DIR = REPO_ROOT / "datasets" / "single_class_dir"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "datasets" / "single_class_org"
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Reorganize a single-class dataset by merging each sample directory's masks into "
            "one label image and writing numbered image/label pairs."
        )
    )
    parser.add_argument(
        "input_dir",
        nargs="?",
        default=str(DEFAULT_INPUT_DIR),
        help=f"Directory containing sample folders. Defaults to: {DEFAULT_INPUT_DIR}",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help=f"Output directory to create. Defaults to: {DEFAULT_OUTPUT_DIR}",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=1,
        help="Starting index for output filenames. Defaults to 1.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optionally process only the first N sample directories.",
    )
    parser.add_argument(
        "--foreground-value",
        type=int,
        default=None,
        help=(
            "Override the positive label value for merged masks. By default the script preserves "
            "the shared non-zero value found in the source masks, or falls back to 255."
        ),
    )
    parser.add_argument(
        "--image-quality",
        type=int,
        default=95,
        help="JPEG quality for exported images. Defaults to 95.",
    )
    parser.add_argument(
        "--mask-quality",
        type=int,
        default=100,
        help="JPEG quality for exported masks when using .jpg output. Defaults to 100.",
    )
    parser.add_argument(
        "--mask-format",
        choices=("png", "jpg"),
        default="png",
        help=(
            "Output format for merged masks. Defaults to png because JPEG introduces extra label "
            "values and is not lossless."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Remove an existing output directory before writing new files.",
    )
    return parser.parse_args()


def iter_image_files(directory: Path) -> list[Path]:
    return sorted(
        path for path in directory.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )


def prepare_output_dir(output_dir: Path, overwrite: bool) -> tuple[Path, Path]:
    if output_dir.exists():
        if not overwrite:
            raise SystemExit(
                f"Output directory already exists: {output_dir}. Use --overwrite to replace it."
            )
        shutil.rmtree(output_dir)

    images_dir = output_dir / "images"
    masks_dir = output_dir / "masks"
    images_dir.mkdir(parents=True, exist_ok=False)
    masks_dir.mkdir(parents=True, exist_ok=False)
    return images_dir, masks_dir


def load_pillow():
    try:
        from PIL import Image
    except ImportError as error:
        raise SystemExit(
            "Pillow is required for this script. Install it with 'python -m pip install Pillow'."
        ) from error

    return Image


def open_rgb_image(image_path: Path):
    image_module = load_pillow()
    with image_module.open(image_path) as image:
        if "A" in image.getbands():
            rgba_image = image.convert("RGBA")
            background = image_module.new("RGB", rgba_image.size, (255, 255, 255))
            background.paste(rgba_image, mask=rgba_image.getchannel("A"))
            return background
        return image.convert("RGB")


def load_mask(mask_path: Path) -> np.ndarray:
    image_module = load_pillow()
    with image_module.open(mask_path) as mask_image:
        return np.asarray(mask_image.convert("L"), dtype=np.uint8)


def merge_masks(mask_paths: list[Path], foreground_override: int | None) -> np.ndarray:
    if not mask_paths:
        raise ValueError("Expected at least one mask file.")

    combined_foreground: np.ndarray | None = None
    positive_values: set[int] = set()
    expected_shape: tuple[int, int] | None = None

    for mask_path in mask_paths:
        mask = load_mask(mask_path)
        if mask.ndim != 2:
            raise ValueError(f"Mask must be a single-channel image: {mask_path}")

        if expected_shape is None:
            expected_shape = mask.shape
            combined_foreground = np.zeros(mask.shape, dtype=bool)
        elif mask.shape != expected_shape:
            raise ValueError(
                f"Mask shape {mask.shape} does not match expected shape {expected_shape}: {mask_path}"
            )

        positive_pixels = mask[mask > 0]
        if positive_pixels.size:
            positive_values.update(int(value) for value in np.unique(positive_pixels))

        combined_foreground |= mask > 0

    if combined_foreground is None:
        raise ValueError("Unable to initialize merged mask.")

    if foreground_override is not None:
        foreground_value = foreground_override
    elif not positive_values:
        foreground_value = 255
    elif len(positive_values) == 1:
        foreground_value = next(iter(positive_values))
    else:
        raise ValueError(
            "Source masks contain multiple non-zero label values. Use --foreground-value to choose one."
        )

    merged_mask = np.zeros(combined_foreground.shape, dtype=np.uint8)
    merged_mask[combined_foreground] = foreground_value
    return merged_mask


def build_output_name(prefix: str, index: int, extension: str) -> str:
    return f"{prefix}_{index:05d}.{extension}"


def export_sample(
    sample_dir: Path,
    output_images_dir: Path,
    output_masks_dir: Path,
    output_index: int,
    image_quality: int,
    mask_quality: int,
    mask_format: str,
    foreground_override: int | None,
) -> None:
    image_module = load_pillow()

    images_dir = sample_dir / "images"
    masks_dir = sample_dir / "masks"
    if not images_dir.is_dir():
        raise ValueError(f"Missing images directory: {images_dir}")
    if not masks_dir.is_dir():
        raise ValueError(f"Missing masks directory: {masks_dir}")

    source_images = iter_image_files(images_dir)
    if len(source_images) != 1:
        raise ValueError(
            f"Expected exactly one source image in {images_dir}, found {len(source_images)}."
        )

    source_masks = iter_image_files(masks_dir)
    if not source_masks:
        raise ValueError(f"No mask files found in {masks_dir}.")

    image_output_path = output_images_dir / build_output_name("image", output_index, "jpg")
    mask_output_path = output_masks_dir / build_output_name("label", output_index, mask_format)

    rgb_image = open_rgb_image(source_images[0])
    merged_mask = merge_masks(source_masks, foreground_override)

    rgb_image.save(image_output_path, format="JPEG", quality=image_quality)
    merged_mask_image = image_module.fromarray(merged_mask)
    if mask_format == "jpg":
        merged_mask_image.save(
            mask_output_path,
            format="JPEG",
            quality=mask_quality,
            subsampling=0,
        )
    else:
        merged_mask_image.save(mask_output_path, format="PNG")


def main() -> int:
    args = parse_args()

    input_dir = Path(args.input_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()

    if not input_dir.is_dir():
        print(f"Input directory does not exist: {input_dir}", file=sys.stderr)
        return 1

    if args.start_index <= 0:
        print("--start-index must be greater than 0.", file=sys.stderr)
        return 1

    if args.limit is not None and args.limit <= 0:
        print("--limit must be greater than 0 when provided.", file=sys.stderr)
        return 1

    if not 1 <= args.image_quality <= 95:
        print("--image-quality must be between 1 and 95.", file=sys.stderr)
        return 1

    if not 1 <= args.mask_quality <= 100:
        print("--mask-quality must be between 1 and 100.", file=sys.stderr)
        return 1

    if args.foreground_value is not None and not 1 <= args.foreground_value <= 255:
        print("--foreground-value must be between 1 and 255.", file=sys.stderr)
        return 1

    sample_dirs = sorted(path for path in input_dir.iterdir() if path.is_dir())
    if not sample_dirs:
        print(f"No sample directories found in: {input_dir}", file=sys.stderr)
        return 1

    if args.mask_format == "jpg":
        print(
            "Warning: JPEG masks are lossy and may not preserve exact label values. "
            "Use --mask-format png for training-safe masks.",
            file=sys.stderr,
        )

    if args.limit is not None:
        sample_dirs = sample_dirs[: args.limit]

    output_images_dir, output_masks_dir = prepare_output_dir(output_dir, args.overwrite)

    try:
        for offset, sample_dir in enumerate(sample_dirs):
            export_sample(
                sample_dir=sample_dir,
                output_images_dir=output_images_dir,
                output_masks_dir=output_masks_dir,
                output_index=args.start_index + offset,
                image_quality=args.image_quality,
                mask_quality=args.mask_quality,
                mask_format=args.mask_format,
                foreground_override=args.foreground_value,
            )
    except ValueError as error:
        print(str(error), file=sys.stderr)
        return 1

    print(
        f"Exported {len(sample_dirs)} sample(s) from {input_dir} to {output_dir} "
        f"with numbered image/label pairs."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())