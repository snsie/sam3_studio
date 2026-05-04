#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEFAULT_ANNOTATION_DIR="${ROOT_DIR}/datasets/MoNuSeg 2018 Training Data/Annotations"
DEFAULT_IMAGE_DIR="${ROOT_DIR}/datasets/MoNuSeg 2018 Training Data/Tissue Images"

ANNOTATION_DIR="$DEFAULT_ANNOTATION_DIR"
IMAGE_DIR="$DEFAULT_IMAGE_DIR"
OUTPUT_DIR=""
FORMAT="jpg"
QUALITY=100
OVERWRITE=0

usage() {
  cat <<'EOF'
Usage: convert_monusg_xml_to_jpg.sh [options]

Rasterize MoNuSeg XML polygon annotations into binary mask images.
Each XML file is matched to a same-stem .tif or .tiff image to determine mask size.

Options:
  --annotation-dir PATH  Directory containing .xml annotations.
  --image-dir PATH       Directory containing matching .tif/.tiff images.
  --output-dir PATH      Directory to write mask files into.
  --format TYPE          Output format: jpg or png. Default: jpg.
  --quality N            JPEG quality from 1 to 100. Default: 100.
  --overwrite            Replace existing output files.
  -h, --help             Show this help message.

Notes:
  JPEG is lossy, so mask edges may gain intermediate grayscale values.
  PNG preserves exact binary mask values.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --annotation-dir)
      ANNOTATION_DIR="$2"
      shift 2
      ;;
    --image-dir)
      IMAGE_DIR="$2"
      shift 2
      ;;
    --output-dir)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --format)
      FORMAT="$2"
      shift 2
      ;;
    --quality)
      QUALITY="$2"
      shift 2
      ;;
    --overwrite)
      OVERWRITE=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Error: unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

if [[ ! -d "$ANNOTATION_DIR" ]]; then
  echo "Error: annotation directory not found: $ANNOTATION_DIR" >&2
  exit 1
fi

if [[ ! -d "$IMAGE_DIR" ]]; then
  echo "Error: image directory not found: $IMAGE_DIR" >&2
  exit 1
fi

if [[ "$FORMAT" != "jpg" && "$FORMAT" != "png" ]]; then
  echo "Error: --format must be either 'jpg' or 'png'." >&2
  exit 1
fi

if ! [[ "$QUALITY" =~ ^[0-9]+$ ]] || (( QUALITY < 1 || QUALITY > 100 )); then
  echo "Error: --quality must be an integer between 1 and 100." >&2
  exit 1
fi

if [[ -z "$OUTPUT_DIR" ]]; then
  OUTPUT_DIR="${ROOT_DIR}/datasets/MoNuSeg 2018 Training Data/Annotations_${FORMAT}"
fi

mkdir -p "$OUTPUT_DIR"

python_bin="${PYTHON_BIN:-}"
if [[ -z "$python_bin" ]]; then
  if command -v /home/neph-workshop-user/miniconda3/envs/sam3/bin/python >/dev/null 2>&1; then
    python_bin="/home/neph-workshop-user/miniconda3/envs/sam3/bin/python"
  elif command -v python3 >/dev/null 2>&1; then
    python_bin="$(command -v python3)"
  elif command -v python >/dev/null 2>&1; then
    python_bin="$(command -v python)"
  else
    echo "Error: could not find a Python interpreter." >&2
    exit 1
  fi
fi

"$python_bin" - "$ANNOTATION_DIR" "$IMAGE_DIR" "$OUTPUT_DIR" "$FORMAT" "$QUALITY" "$OVERWRITE" <<'PY'
from __future__ import annotations

from pathlib import Path
import sys
import xml.etree.ElementTree as ET

annotation_dir = Path(sys.argv[1]).expanduser().resolve()
image_dir = Path(sys.argv[2]).expanduser().resolve()
output_dir = Path(sys.argv[3]).expanduser().resolve()
output_format = sys.argv[4]
quality = int(sys.argv[5])
overwrite = sys.argv[6] == "1"

try:
    from PIL import Image, ImageDraw
except ImportError as error:
    raise SystemExit(
        "Pillow is required for this script. Install it with 'python -m pip install Pillow'."
    ) from error


def find_matching_image(stem: str) -> Path:
    for suffix in (".tif", ".tiff", ".TIF", ".TIFF"):
        candidate = image_dir / f"{stem}{suffix}"
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"No matching .tif/.tiff image found for annotation: {stem}")


def load_regions(xml_path: Path) -> list[list[tuple[float, float]]]:
    root = ET.parse(xml_path).getroot()
    polygons: list[list[tuple[float, float]]] = []

    for region in root.findall(".//Region"):
        vertices = []
        for vertex in region.findall("./Vertices/Vertex"):
            x = vertex.get("X")
            y = vertex.get("Y")
            if x is None or y is None:
                continue
            vertices.append((float(x), float(y)))

        if len(vertices) >= 3:
            polygons.append(vertices)

    return polygons


def rasterize_annotation(xml_path: Path) -> None:
    image_path = find_matching_image(xml_path.stem)
  output_path = output_dir / f"{xml_path.stem}.{output_format}"

    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Output file already exists: {output_path}")

    polygons = load_regions(xml_path)
    with Image.open(image_path) as image:
        width, height = image.size

    mask = Image.new("L", (width, height), 0)
    drawer = ImageDraw.Draw(mask)
    for polygon in polygons:
        drawer.polygon(polygon, fill=255, outline=255)

    if output_format == "png":
      mask.save(output_path, format="PNG")
    else:
      mask.save(output_path, format="JPEG", quality=quality, subsampling=0)
    print(f"Wrote {output_path.name} ({len(polygons)} regions, {width}x{height})")


xml_files = sorted(annotation_dir.glob("*.xml"))
if not xml_files:
    raise SystemExit(f"No .xml files found in: {annotation_dir}")

failures = 0
for xml_file in xml_files:
    try:
        rasterize_annotation(xml_file)
    except Exception as error:  # pragma: no cover - shell wrapper reports failures directly
        failures += 1
        print(f"Failed {xml_file.name}: {error}", file=sys.stderr)

if failures:
    raise SystemExit(f"Finished with {failures} failure(s).")

print(f"Converted {len(xml_files)} annotation file(s) into {output_format.upper()} masks in {output_dir}")
PY