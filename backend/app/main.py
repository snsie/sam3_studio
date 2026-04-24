from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes_jobs import router as jobs_router
from app.api.routes_segment import router as segment_router
from app.api.routes_sessions import router as sessions_router
from app.api.routes_upload import router as upload_router
from app.core.config import get_settings
from app.services.job_queue import JobQueue
from app.services.sam3_service import SAM3Service


def create_app() -> FastAPI:
    settings = get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        sam3_service = SAM3Service(settings)
        sam3_service.preload_model()
        job_queue = JobQueue(worker=sam3_service.run_segmentation)
        job_queue.start()

        app.state.settings = settings
        app.state.sam3_service = sam3_service
        app.state.job_queue = job_queue
        try:
            yield
        finally:
            job_queue.stop()

    app = FastAPI(
        title=settings.app_name,
        version="0.1.0",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.frontend_origins,
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/health")
    def healthcheck() -> dict[str, str]:
        return {"status": "ok"}

    app.include_router(upload_router, prefix=settings.api_prefix)
    app.include_router(segment_router, prefix=settings.api_prefix)
    app.include_router(jobs_router, prefix=settings.api_prefix)
    app.include_router(sessions_router, prefix=settings.api_prefix)
    return app


app = create_app()