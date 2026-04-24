#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ "${CONDA_DEFAULT_ENV:-}" != "sam3" ]]; then
  echo "Activate the existing Conda environment first: conda activate sam3" >&2
  exit 1
fi

cd "$ROOT_DIR/backend"
exec python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
