# SAM3 Studio

SAM3 Studio is a local FastAPI + React research/demo app for interactive exploration of the installed `sam3` package. It supports image upload, click-to-segment, positive/negative point prompts, box prompts, experimental text prompting, candidate-mask switching, overlay opacity control, and synchronous segmentation requests backed by a local in-process queue and worker pool.

## Stack

- Backend: FastAPI, Pydantic, Pillow, NumPy
- Frontend: React, Vite, TypeScript
- Model integration: direct import from the existing `sam3` Conda environment
- Runtime model flow: local-only, single-machine, no auth, no database, no external inference service

## Project Layout

- `backend/`: FastAPI app, typed schemas, SAM3 service, local queue, upload cache
- `frontend/`: React + Vite TypeScript app and interactive viewer
- `scripts/`: Conda-aware local setup and run helpers

## SAM3 Integration Notes

The backend uses the installed `sam3` package directly and chooses the best supported SAM3 inference path for the currently active prompts:

- Interactive point/box segmentation uses `build_sam3_image_model(..., enable_inst_interactivity=True)` plus `model.predict_inst(...)`.
- Text prompting uses `Sam3Processor.set_text_prompt(...)` and optional `add_geometric_prompt(...)` box refinement.

In product terms, this means:

- Any active prompts are applied together whenever the installed SAM3 APIs support that combination.
- Text-only prompting is exposed as experimental grounding.
- Point + text requests keep the interactive masks and use the text prompt as an additional grounding prior.
- Text + box requests are supported automatically without exposing the lower-level SAM3 path details in the UI.

## Prerequisites

1. Conda environment `sam3` already exists.
2. The `sam3` Python package is already installed in that environment.
3. Node.js and npm are available locally.
4. Activate the existing environment yourself before setup or running anything:

```bash
conda activate sam3
```

## Setup

From the repository root:

```bash
./scripts/setup_dev.sh
```

That script only installs dependencies into the currently active `sam3` environment. It does not create a new environment, virtualenv, Poetry env, pipenv env, or uv env.

## Optional Backend Configuration

Copy `backend/.env.example` to `backend/.env` if you want to override defaults.

Important variables:

- `SAM3_DEVICE=auto|cuda|cpu`
- `SAM3_CHECKPOINT_PATH=/absolute/path/to/checkpoint.pt`
- `SAM3_BPE_PATH=/absolute/path/to/bpe.model`
- `SAM3_LOAD_FROM_HF=true|false`
- `SAM3_WORKER_COUNT=3`
- `SAM3_QUEUE_MAX_SIZE=0`
- `SAM3_REQUEST_TIMEOUT_SECONDS=300`
- `SAM3_CONFIDENCE_THRESHOLD=0.5`
- `FRONTEND_ORIGINS=https://faiht.ai,https://www.faiht.ai`

Optional frontend variable:

- `VITE_SEGMENTATION_TIMEOUT_MS=300000`

If you want the app to stay strictly offline after startup, provide local checkpoint assets and set `SAM3_LOAD_FROM_HF=false`.

### HiPerGator configuration

The tracked `backend/.env.hipergator.example` records the private-use cluster settings. On HiPerGator, copy it to `backend/.env` and replace the placeholder paths with your group's storage and actual model assets:

```bash
cp backend/.env.hipergator.example backend/.env
```

Stage challenge datasets into `CHALLENGE_DATASETS_DIR` if you use them. The template uses one model worker and requires CUDA; start the backend inside a Slurm GPU allocation and access it through an SSH tunnel. Dotenv settings do not request cluster resources or prevent job expiration or idle-GPU termination. The active `backend/.env` remains ignored by Git; keep credentials out of the tracked template.

## Run Locally

Start the backend in one terminal:

```bash
conda activate sam3
./scripts/run_backend.sh
```

The backend preloads a pool of SAM3 workers during startup so the first prompt does not pay the model-load penalty. By default it tries to load three independent model instances in the FastAPI process. If the GPU cannot fit the requested worker count, startup now either reduces the pool with an explicit warning or fails fast with a clear error.

Start the frontend in another terminal:

```bash
./scripts/run_frontend.sh
```

Open `http://localhost:5173`.

## Run In Production

From the repository root:

```bash
./run_studio.sh prod
```

Production mode now builds the Vite frontend and serves both the UI and `/api` routes from the FastAPI process on `http://localhost:8001`. This is the mode to place behind `faiht.ai`.

For the `faiht.ai` deployment:

- Point `faiht.ai` and `www.faiht.ai` at the VM.
- Use Caddy to reverse-proxy all traffic to `http://127.0.0.1:8001`.
- Add the `faiht.ai` and `www.faiht.ai` site block to the Caddyfile you already run on the server.
- Once DNS is pointed at the VM, Caddy can provision TLS automatically for both hostnames.
- If you terminate TLS or serve the frontend from a different origin, set `FRONTEND_ORIGINS` explicitly in `backend/.env`.

## Workflow

1. Upload an image.
2. Choose an interaction mode.
3. Place points or drag a box directly on the canvas, or enter a text prompt.
4. Run segmentation, or use `Click` mode for immediate single-click mask generation.
5. If all model workers are busy, the request waits in the in-memory FIFO queue until a worker becomes available.
6. Inspect returned masks, switch between candidates, and tune overlay opacity.

## Backend API Summary

- `POST /api/upload`: upload a local image and create a session
- `GET /api/sessions/{session_id}`: fetch session metadata
- `GET /api/sessions/{session_id}/image`: fetch the normalized uploaded image
- `POST /api/sessions/{session_id}/reset-cache`: clear cached embeddings for the session
- `POST /api/segment/jobs`: enqueue a segmentation job and hold the request open until the queued job completes
- `GET /api/jobs/{job_id}`: inspect queued/running/completed job status for compatibility or debugging

## Development Notes

- The inference queue is intentionally simple: one in-process FIFO dispatcher plus a fixed pool of independently loaded SAM3 workers.
- By default, only three segmentation jobs run at the same time on the single VM GPU while additional HTTP requests wait in memory.
- Uploaded images are normalized to PNG and stored temporarily in `backend/data/uploads/`.
- The frontend uses Vite dev-server proxying to forward `/api` calls to the local FastAPI backend.
- Multiple mask candidates are returned when SAM3 provides them; the UI lets you switch the active overlay.
