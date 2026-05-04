#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TARGET_DIR="${1:-${ROOT_DIR}/multi_class/labels}"

usage() {
  echo "Usage: $(basename "$0") [label_directory]"
  echo "Removes labels_*_channel0.* files and renames labels_*_channel1.* to labels_*.*"
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

if [[ ! -d "$TARGET_DIR" ]]; then
  echo "Error: label directory not found: $TARGET_DIR" >&2
  exit 1
fi

shopt -s nullglob

channel0_files=("$TARGET_DIR"/labels_*_channel0.*)
channel1_files=("$TARGET_DIR"/labels_*_channel1.*)

removed_count=0
renamed_count=0

for channel0_path in "${channel0_files[@]}"; do
  rm -- "$channel0_path"
  ((removed_count += 1))
done

for channel1_path in "${channel1_files[@]}"; do
  filename="$(basename "$channel1_path")"
  stem="${filename%.*}"
  extension="${filename##*.}"
  target_name="${stem%_channel1}.${extension}"
  target_path="${TARGET_DIR}/${target_name}"

  if [[ -e "$target_path" ]]; then
    echo "Error: target file already exists: $target_path" >&2
    exit 1
  fi

  mv -- "$channel1_path" "$target_path"
  ((renamed_count += 1))
done

echo "Removed ${removed_count} channel0 label images from $TARGET_DIR"
echo "Renamed ${renamed_count} channel1 label images in $TARGET_DIR"