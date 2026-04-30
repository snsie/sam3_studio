from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.api.routes_jobs import router as jobs_router
from app.api.routes_samples import router as samples_router
from app.api.routes_segment import router as segment_router
from app.api.routes_sessions import router as sessions_router
from app.api.routes_upload import router as upload_router
from app.core.config import get_settings
from app.services.job_queue import InferenceQueue, ModelWorker
from app.services.sam3_runtime import (
    SAM3ModelRuntime,
    is_gpu_capacity_error,
    release_cuda_cache,
)
from app.services.sam3_service import SAM3Service

logger = logging.getLogger(__name__)
FRONTEND_DIST_DIR = Path(__file__).resolve().parents[2] / "frontend" / "dist"


def build_model_workers(settings, sam3_service: SAM3Service) -> list[ModelWorker]:
    workers: list[ModelWorker] = []

    for worker_id in range(1, settings.sam3_worker_count + 1):
        worker = ModelWorker(
            worker_id=worker_id,
            runtime=SAM3ModelRuntime(settings, sam3_service),
        )
        try:
            worker.preload()
        except Exception as exc:
            release_cuda_cache()
            if workers and is_gpu_capacity_error(exc):
                logger.warning(
                    "Started %s of %s requested SAM3 workers because the GPU could not fit another model instance: %s",
                    len(workers),
                    settings.sam3_worker_count,
                    exc,
                )
                break

            raise RuntimeError(
                f"Failed to initialize SAM3 worker {worker_id}/{settings.sam3_worker_count}: {exc}"
            ) from exc

        workers.append(worker)

    if not workers:
        raise RuntimeError(
            "Unable to initialize any SAM3 workers. Reduce SAM3_WORKER_COUNT or check the GPU/model configuration."
        )

    logger.info(
        "Initialized %s SAM3 worker(s) (requested=%s)",
        len(workers),
        settings.sam3_worker_count,
    )
    return workers


def create_app() -> FastAPI:
    settings = get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        sam3_service = SAM3Service(settings)
        workers = await asyncio.to_thread(build_model_workers, settings, sam3_service)
        job_queue = InferenceQueue(
            workers=workers,
            max_queue_size=settings.sam3_queue_max_size,
        )
        await job_queue.start()

        app.state.settings = settings
        app.state.sam3_service = sam3_service
        app.state.job_queue = job_queue
        try:
            yield
        finally:
            await job_queue.stop()

    app = FastAPI(
        title=settings.app_name,
        version="0.1.0",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.frontend_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/health")
    def healthcheck() -> dict[str, str]:
        return {"status": "ok"}

    app.include_router(upload_router, prefix=settings.api_prefix)
    app.include_router(segment_router, prefix=settings.api_prefix)
    app.include_router(jobs_router, prefix=settings.api_prefix)
    app.include_router(samples_router, prefix=settings.api_prefix)
    app.include_router(sessions_router, prefix=settings.api_prefix)

    if FRONTEND_DIST_DIR.is_dir():
        logger.info("Serving frontend from %s", FRONTEND_DIST_DIR)
        app.mount("/", StaticFiles(directory=FRONTEND_DIST_DIR, html=True), name="frontend")

    return app


app = create_app()
