#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np


DEFAULT_INPUT_FILE = (
    Path(__file__).resolve().parents[1] / "datasets" / "not_sure" / "labels.npy"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert an .npy array into one or more JPEG images."
    )
    parser.add_argument(
        "input_file",
        nargs="?",
        default=str(DEFAULT_INPUT_FILE),
        help=f"Path to the .npy file to convert. Defaults to: {DEFAULT_INPUT_FILE}",
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
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optionally convert only the first N frames for large arrays.",
    )
    return parser.parse_args()


def resolve_output_dir(input_file: Path, output_dir: str | None) -> Path:
    if output_dir:
        return Path(output_dir).expanduser().resolve()
    return input_file.parent / f"{input_file.stem}_jpg"


def to_uint8(image: np.ndarray) -> np.ndarray:
    array = np.asarray(image)
    if array.dtype == np.uint8:
        return array

    finite_mask = np.isfinite(array)
    if not finite_mask.any():
        return np.zeros(array.shape, dtype=np.uint8)

    valid_values = array[finite_mask]
    min_value = float(valid_values.min())
    max_value = float(valid_values.max())

    if min_value == max_value:
        return np.zeros(array.shape, dtype=np.uint8)

    scaled = (array.astype(np.float32) - min_value) / (max_value - min_value)
    scaled = np.clip(scaled, 0.0, 1.0)
    scaled[~finite_mask] = 0.0
    return (scaled * 255.0).round().astype(np.uint8)


def save_image(image: np.ndarray, output_path: Path, quality: int) -> None:
    try:
        from PIL import Image
    except ImportError as error:
        raise SystemExit(
            "Pillow is required for this script. Install it with 'python -m pip install Pillow'."
        ) from error

    array = np.asarray(image)
    if array.ndim == 2:
        pil_image = Image.fromarray(to_uint8(array))
    elif array.ndim == 3 and array.shape[-1] in (3, 4):
        uint8_array = to_uint8(array)
        pil_image = Image.fromarray(uint8_array)
        if uint8_array.shape[-1] == 4:
            pil_image = pil_image.convert("RGB")
    else:
        raise ValueError(f"Unsupported image shape for JPEG export: {array.shape}")

    pil_image.save(output_path, format="JPEG", quality=quality)


def convert_array(data: np.ndarray, output_dir: Path, stem: str, quality: int, limit: int | None) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)

    if data.ndim == 2:
        save_image(data, output_dir / f"{stem}.jpg", quality)
        return 1

    if data.ndim == 3 and data.shape[-1] in (3, 4):
        save_image(data, output_dir / f"{stem}.jpg", quality)
        return 1

    written = 0

    if data.ndim == 3:
        frame_count = data.shape[0] if limit is None else min(limit, data.shape[0])
        for index in range(frame_count):
            save_image(data[index], output_dir / f"{stem}_{index:05d}.jpg", quality)
            written += 1
        return written

    if data.ndim == 4:
        frame_count = data.shape[0] if limit is None else min(limit, data.shape[0])

        if data.shape[-1] in (1, 3, 4):
            for index in range(frame_count):
                frame = data[index]
                if frame.shape[-1] == 1:
                    frame = frame[..., 0]
                save_image(frame, output_dir / f"{stem}_{index:05d}.jpg", quality)
                written += 1
            return written

        channel_count = data.shape[-1]
        for index in range(frame_count):
            for channel in range(channel_count):
                channel_image = data[index, ..., channel]
                save_image(
                    channel_image,
                    output_dir / f"{stem}_{index:05d}_channel{channel}.jpg",
                    quality,
                )
                written += 1
        return written

    raise SystemExit(
        "Unsupported array shape. Expected 2D, 3D, or 4D arrays that can be interpreted as images."
    )


def main() -> int:
    args = parse_args()

    input_file = Path(args.input_file).expanduser().resolve()
    if not input_file.is_file():
        print(f"Input file does not exist: {input_file}", file=sys.stderr)
        return 1

    if input_file.suffix.lower() != ".npy":
        print("Input file must have a .npy extension.", file=sys.stderr)
        return 1

    if not 1 <= args.quality <= 95:
        print("JPEG quality must be between 1 and 95.", file=sys.stderr)
        return 1

    if args.limit is not None and args.limit <= 0:
        print("--limit must be greater than 0 when provided.", file=sys.stderr)
        return 1

    try:
        data = np.load(input_file, mmap_mode="r")
    except ValueError as error:
        print(
            "Unable to memory-map the .npy file. Only standard numeric arrays are supported.",
            file=sys.stderr,
        )
        print(str(error), file=sys.stderr)
        return 1

    if data.dtype == object:
        print("Object arrays are not supported for JPEG export.", file=sys.stderr)
        return 1

    output_dir = resolve_output_dir(input_file, args.output_dir)
    if output_dir == input_file.parent:
        print("Output directory must not be the input file's parent directory.", file=sys.stderr)
        return 1

    written = convert_array(data, output_dir, input_file.stem, args.quality, args.limit)
    print(f"Wrote {written} JPEG image(s) to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())