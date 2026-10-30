#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="$ROOT_DIR/.venv"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu128}"

if ! command -v uv >/dev/null 2>&1; then
  echo "uv is required: https://docs.astral.sh/uv/getting-started/installation/" >&2
  exit 1
fi

cd "$ROOT_DIR"

if [[ ! -x "$VENV_DIR/bin/python" ]]; then
  echo "Creating uv virtual environment at $VENV_DIR"
  uv venv "$VENV_DIR" --python 3.12
fi

echo "Installing PyTorch from $TORCH_INDEX_URL"
uv pip install --python "$VENV_DIR/bin/python" torch torchvision --index-url "$TORCH_INDEX_URL"

echo "Installing backend dependencies"
uv pip install --python "$VENV_DIR/bin/python" -r backend/requirements.txt

echo "Installing frontend dependencies"
cd frontend
npm install

echo "Setup complete"
