#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKEND_DIR="${ROOT_DIR}/backend"
FRONTEND_DIR="${ROOT_DIR}/frontend"

MODE="${1:-dev}"

case "${MODE}" in
  dev)
    DEFAULT_FRONTEND_PORT=5173
    DEFAULT_BACKEND_PORT=8000
    BACKEND_EXTRA_ARGS=(--reload)
    FRONTEND_SCRIPT=dev
    START_FRONTEND=true
    ;;
  prod)
    DEFAULT_FRONTEND_PORT=4173
    DEFAULT_BACKEND_PORT=8001
    BACKEND_EXTRA_ARGS=()
    FRONTEND_SCRIPT=""
    START_FRONTEND=false
    ;;
  *)
    echo "Usage: ./run_studio.sh [dev|prod]"
    exit 1
    ;;
esac

FRONTEND_PORT="${FRONTEND_PORT:-${DEFAULT_FRONTEND_PORT}}"
BACKEND_PORT="${BACKEND_PORT:-${DEFAULT_BACKEND_PORT}}"
TARGET_PORTS=("${BACKEND_PORT}" "${FRONTEND_PORT}")
FRONTEND_ENV=(
  "VITE_BACKEND_PORT=${VITE_BACKEND_PORT:-${BACKEND_PORT}}"
  "VITE_FRONTEND_PORT=${VITE_FRONTEND_PORT:-${FRONTEND_PORT}}"
)

if [[ -n "${VITE_API_BASE_URL:-}" ]]; then
  FRONTEND_ENV+=("VITE_API_BASE_URL=${VITE_API_BASE_URL}")
fi

BACKEND_ENV=()

if [[ -n "${FRONTEND_ORIGINS:-}" ]]; then
  BACKEND_ENV+=("FRONTEND_ORIGINS=${FRONTEND_ORIGINS}")
fi

VENV_DIR="${ROOT_DIR}/.venv"

if [[ ! -x "${VENV_DIR}/bin/uvicorn" ]]; then
  echo "Error: uvicorn not found in ${VENV_DIR}. Run ./scripts/setup_dev.sh first."
  exit 1
fi

if ! command -v npm >/dev/null 2>&1; then
  echo "Error: npm is required to start the frontend."
  exit 1
fi

find_port_pids() {
  local port

  if command -v lsof >/dev/null 2>&1; then
    for port in "$@"; do
      lsof -tiTCP:"${port}" -sTCP:LISTEN 2>/dev/null || true
    done
    return 0
  fi

  if command -v fuser >/dev/null 2>&1; then
    for port in "$@"; do
      fuser "${port}/tcp" 2>/dev/null || true
    done | tr ' ' '\n'
    return 0
  fi

  echo "Error: lsof or fuser is required to clear occupied ports."
  exit 1
}

kill_port_processes() {
  local ports=("$@")
  local pids=()

  mapfile -t pids < <(find_port_pids "${ports[@]}")
  if [[ ${#pids[@]} -eq 0 ]]; then
    return 0
  fi

  echo "Killing processes on ports: ${ports[*]}"
  kill "${pids[@]}" >/dev/null 2>&1 || true

  mapfile -t pids < <(find_port_pids "${ports[@]}")
  if [[ ${#pids[@]} -gt 0 ]]; then
    kill -9 "${pids[@]}" >/dev/null 2>&1 || true
  fi
}

export PATH="${VENV_DIR}/bin:${PATH}"

if [[ "${MODE}" == "prod" ]]; then
  echo "Building frontend for production..."
  (
    cd "${FRONTEND_DIR}"
    env "${FRONTEND_ENV[@]}" npm run build
  )
fi

PIDS=()

cleanup() {
  for pid in "${PIDS[@]:-}"; do
    if kill -0 "${pid}" >/dev/null 2>&1; then
      kill "${pid}" >/dev/null 2>&1 || true
    fi
  done

  kill_port_processes "${TARGET_PORTS[@]}"
  wait >/dev/null 2>&1 || true
}

kill_port_processes "${TARGET_PORTS[@]}"

trap cleanup EXIT INT TERM

echo "Starting backend on port ${BACKEND_PORT} (${MODE})..."
(
  cd "${BACKEND_DIR}"
  env "${BACKEND_ENV[@]}" uvicorn app.main:app --host 0.0.0.0 --port "${BACKEND_PORT}" "${BACKEND_EXTRA_ARGS[@]}"
) &
PIDS+=("$!")

if [[ "${START_FRONTEND}" == "true" ]]; then
  echo "Starting frontend on port ${FRONTEND_PORT} (${MODE})..."
  (
    cd "${FRONTEND_DIR}"
    env "${FRONTEND_ENV[@]}" npm run "${FRONTEND_SCRIPT}" -- --host 0.0.0.0 --port "${FRONTEND_PORT}"
  ) &
  PIDS+=("$!")
fi

if [[ "${MODE}" == "prod" ]]; then
  echo "App:      http://localhost:${BACKEND_PORT}"
else
  echo "Frontend: http://localhost:${FRONTEND_PORT}"
  echo "Backend:  http://localhost:${BACKEND_PORT}"
fi

set +e
wait -n "${PIDS[@]}"
EXIT_CODE=$?
set -e

exit "${EXIT_CODE}"