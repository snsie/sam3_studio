# SAM3 Studio

SAM3 Studio is a local FastAPI + React research/demo app for interactive exploration of the installed `sam3` package. It supports image upload, click-to-segment, positive/negative point prompts, box prompts, experimental text prompting, candidate-mask switching, overlay opacity control, and progressive job status updates backed by a local in-process queue.

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
- `SAM3_CONFIDENCE_THRESHOLD=0.5`

If you want the app to stay strictly offline after startup, provide local checkpoint assets and set `SAM3_LOAD_FROM_HF=false`.

## Run Locally

Start the backend in one terminal:

```bash
conda activate sam3
./scripts/run_backend.sh
```

The backend preloads the SAM3 model during startup so the first prompt does not pay the model-load penalty. If checkpoints or tokenizer assets are misconfigured, the backend now fails fast during startup instead of on the first segmentation request.

Start the frontend in another terminal:

```bash
./scripts/run_frontend.sh
```

Open `http://localhost:5173`.

## Workflow

1. Upload an image.
2. Choose an interaction mode.
3. Place points or drag a box directly on the canvas, or enter a text prompt.
4. Run segmentation, or use `Click` mode for immediate single-click mask generation.
5. Polling updates show job state while the backend processes the request.
6. Inspect returned masks, switch between candidates, and tune overlay opacity.

## Backend API Summary

- `POST /api/upload`: upload a local image and create a session
- `GET /api/sessions/{session_id}`: fetch session metadata
- `GET /api/sessions/{session_id}/image`: fetch the normalized uploaded image
- `POST /api/sessions/{session_id}/reset-cache`: clear cached embeddings for the session
- `POST /api/segment/jobs`: enqueue a segmentation job
- `GET /api/jobs/{job_id}`: poll job status and retrieve results

## Development Notes

- The job queue is intentionally simple: one in-process worker thread for maintainable local research/demo use.
- Uploaded images are normalized to PNG and stored temporarily in `backend/data/uploads/`.
- The frontend uses Vite dev-server proxying to forward `/api` calls to the local FastAPI backend.
- Multiple mask candidates are returned when SAM3 provides them; the UI lets you switch the active overlay.
