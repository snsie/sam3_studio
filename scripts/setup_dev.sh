#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ "${CONDA_DEFAULT_ENV:-}" != "sam3" ]]; then
  echo "Activate the existing Conda environment first: conda activate sam3" >&2
  exit 1
fi

cd "$ROOT_DIR"

echo "Installing backend dependencies into the active Conda environment: ${CONDA_DEFAULT_ENV}"
python -m pip install --upgrade pip
python -m pip install -r backend/requirements.txt

echo "Installing frontend dependencies"
cd frontend
npm install

echo "Setup complete"
