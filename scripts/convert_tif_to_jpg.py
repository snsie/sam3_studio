#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

DEFAULT_INPUT_DIR = (
    Path(__file__).resolve().parents[1]
    / "datasets"
    / "MoNuSeg 2018 Training Data"
    / "Tissue Images"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert .tif images in a directory to .jpg files in a sibling output directory."
    )
    parser.add_argument(
        "--input_dir",
        nargs="?",
        default=str(DEFAULT_INPUT_DIR),
        help=("Directory containing .tif images. " f"Defaults to: {DEFAULT_INPUT_DIR}"),
    )
    parser.add_argument(
        "--output-dir",
        help="Optional output directory. Defaults to a sibling directory with '_jpg' appended.",
    )
    parser.add_argument(
        "--quality",
        type=int,
        default=95,
        help="JPEG quality from 1 to 95. Defaults to 95.",
    )
    return parser.parse_args()


def resolve_output_dir(input_dir: Path, output_dir: str | None) -> Path:
    if output_dir:
        return Path(output_dir).expanduser().resolve()
    return input_dir.parent / f"{input_dir.name}_jpg"


def convert_images(input_dir: Path, output_dir: Path, quality: int) -> int:
    try:
        from PIL import Image
    except ImportError as error:
        raise SystemExit(
            "Pillow is required for this script. Install it with 'python -m pip install Pillow'."
        ) from error

    tif_files = sorted(input_dir.glob("*.tif")) + sorted(input_dir.glob("*.tiff"))
    if not tif_files:
        raise SystemExit(f"No .tif or .tiff files found in: {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    converted_count = 0
    for tif_path in tif_files:
        jpg_path = output_dir / f"{tif_path.stem}.jpg"
        with Image.open(tif_path) as image:
            rgb_image = image.convert("RGB")
            rgb_image.save(jpg_path, format="JPEG", quality=quality)
        converted_count += 1

    return converted_count


def main() -> int:
    args = parse_args()

    input_dir = Path(args.input_dir).expanduser().resolve()
    if not input_dir.is_dir():
        print(f"Input directory does not exist: {input_dir}", file=sys.stderr)
        return 1

    if not 1 <= args.quality <= 95:
        print("JPEG quality must be between 1 and 95.", file=sys.stderr)
        return 1

    output_dir = resolve_output_dir(input_dir, args.output_dir)
    if output_dir == input_dir:
        print(
            "Output directory must be different from the input directory.",
            file=sys.stderr,
        )
        return 1

    converted_count = convert_images(input_dir, output_dir, args.quality)
    print(f"Converted {converted_count} image(s) from {input_dir} to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
