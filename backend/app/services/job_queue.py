from __future__ import annotations

import asyncio
import logging
from contextlib import suppress
from dataclasses import dataclass, field
from datetime import datetime, timezone
from uuid import uuid4

from app.models.schemas import JobStatus, JobStatusResponse, SegmentationRequest, SegmentationResult
from app.services.sam3_runtime import ProgressCallback, SAM3ModelRuntime


logger = logging.getLogger(__name__)


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class QueueFullError(RuntimeError):
    pass


# Finished jobs stay queryable via GET /jobs/{id}, but each one retains a full
# SegmentationResult with base64 mask PNGs, so only keep a recent window of them.
MAX_RETAINED_TERMINAL_JOBS = 64


@dataclass(slots=True)
class SegmentationJob:
    job_id: str
    request: SegmentationRequest
    future: asyncio.Future[JobStatusResponse]
    status: JobStatus = JobStatus.QUEUED
    progress: float = 0.05
    message: str = "Waiting for available model..."
    created_at: datetime = field(default_factory=utcnow)
    updated_at: datetime = field(default_factory=utcnow)
    result: SegmentationResult | None = None
    error: str | None = None
    abandoned: bool = False

    def to_response(self) -> JobStatusResponse:
        return JobStatusResponse(
            job_id=self.job_id,
            session_id=self.request.session_id,
            status=self.status,
            progress=self.progress,
            message=self.message,
            created_at=self.created_at,
            updated_at=self.updated_at,
            result=self.result,
            error=self.error,
        )


class ModelWorker:
    def __init__(self, worker_id: int, runtime: SAM3ModelRuntime) -> None:
        self.worker_id = worker_id
        self._runtime = runtime
        self.busy = False

    def preload(self) -> None:
        self._runtime.preload_model()

    async def run(
        self,
        job: SegmentationJob,
        progress: ProgressCallback,
    ) -> SegmentationResult:
        self.busy = True
        try:
            return await asyncio.to_thread(self._runtime.run_segmentation, job.request, progress)
        finally:
            self.busy = False


class InferenceQueue:
    """Simple in-process FIFO dispatcher for a fixed pool of SAM3 model workers."""

    def __init__(self, workers: list[ModelWorker], *, max_queue_size: int = 0) -> None:
        if not workers:
            raise ValueError("At least one model worker is required.")

        self._jobs: dict[str, SegmentationJob] = {}
        self._pending_jobs: asyncio.Queue[SegmentationJob] = asyncio.Queue(maxsize=max_queue_size)
        self._available_workers: asyncio.Queue[ModelWorker] = asyncio.Queue()
        self._dispatcher_task: asyncio.Task[None] | None = None
        self._job_tasks: set[asyncio.Task[None]] = set()
        self._stopping = False

        for worker in workers:
            self._available_workers.put_nowait(worker)

    async def start(self) -> None:
        if self._dispatcher_task is None:
            self._dispatcher_task = asyncio.create_task(
                self._dispatch_loop(),
                name="sam3-inference-dispatcher",
            )

    async def stop(self) -> None:
        self._stopping = True
        if self._dispatcher_task is not None:
            self._dispatcher_task.cancel()
            with suppress(asyncio.CancelledError):
                await self._dispatcher_task
            self._dispatcher_task = None

        if self._job_tasks:
            for task in tuple(self._job_tasks):
                task.cancel()
            await asyncio.gather(*tuple(self._job_tasks), return_exceptions=True)
            self._job_tasks.clear()

        # Cancelling a job task raises CancelledError, which _process does not catch, so
        # its future would stay pending and every waiting client would block until its
        # request timeout instead of learning that the server is going away.
        while True:
            try:
                self._pending_jobs.get_nowait()
            except asyncio.QueueEmpty:
                break
            self._pending_jobs.task_done()

        for job in tuple(self._jobs.values()):
            if not job.future.done():
                self._set_failed(job, "The segmentation service is shutting down.")

    async def submit(self, payload: SegmentationRequest) -> SegmentationJob:
        if self._stopping:
            raise RuntimeError("The inference queue is shutting down.")

        loop = asyncio.get_running_loop()
        job = SegmentationJob(
            job_id=str(uuid4()),
            request=payload,
            future=loop.create_future(),
        )
        self._jobs[job.job_id] = job
        self._prune_terminal_jobs()

        try:
            self._pending_jobs.put_nowait(job)
        except asyncio.QueueFull as exc:
            self._jobs.pop(job.job_id, None)
            raise QueueFullError(
                "All SAM3 workers are busy and the in-memory queue is full. Try again shortly."
            ) from exc

        logger.info(
            "Enqueued segmentation job %s (queue_length=%s)",
            job.job_id,
            self._pending_jobs.qsize(),
        )
        return job

    async def submit_and_wait(
        self,
        payload: SegmentationRequest,
        *,
        timeout_seconds: float,
    ) -> JobStatusResponse:
        job = await self.submit(payload)
        try:
            return await asyncio.wait_for(asyncio.shield(job.future), timeout=timeout_seconds)
        except asyncio.TimeoutError:
            # Nobody is left to receive this result. Flag it so a worker does not spend GPU
            # time on it: the frontend aborts its request on every image switch, so queued
            # work outlives the interest in it and would delay the user's live request.
            job.abandoned = True
            logger.warning(
                "Segmentation job %s exceeded the request timeout after %.1f seconds",
                job.job_id,
                timeout_seconds,
            )
            raise
        except asyncio.CancelledError:
            job.abandoned = True
            logger.info(
                "Segmentation job %s was abandoned by its caller before completing",
                job.job_id,
            )
            raise

    def _prune_terminal_jobs(self) -> None:
        # dict preserves insertion order, so these are oldest-first.
        terminal_job_ids = [
            job_id
            for job_id, job in self._jobs.items()
            if job.status in (JobStatus.COMPLETE, JobStatus.FAILED)
        ]
        overflow = len(terminal_job_ids) - MAX_RETAINED_TERMINAL_JOBS
        for job_id in terminal_job_ids[:overflow]:
            self._jobs.pop(job_id, None)

    async def get(self, job_id: str) -> JobStatusResponse | None:
        job = self._jobs.get(job_id)
        if job is None:
            return None
        return job.to_response()

    async def _dispatch_loop(self) -> None:
        while True:
            job = await self._pending_jobs.get()
            worker = await self._available_workers.get()
            logger.info(
                "Assigned segmentation job %s to worker-%s (queue_length=%s)",
                job.job_id,
                worker.worker_id,
                self._pending_jobs.qsize(),
            )
            task = asyncio.create_task(
                self._process(worker, job),
                name=f"sam3-worker-{worker.worker_id}-job-{job.job_id}",
            )
            self._job_tasks.add(task)
            task.add_done_callback(self._job_tasks.discard)

    async def _process(self, worker: ModelWorker, job: SegmentationJob) -> None:
        progress_callback = self._make_progress_callback(job)

        try:
            if job.abandoned:
                logger.info(
                    "Skipping abandoned segmentation job %s before it reached worker-%s",
                    job.job_id,
                    worker.worker_id,
                )
                self._set_failed(job, "The request was abandoned before it started.")
                return

            logger.info("Starting segmentation job %s on worker-%s", job.job_id, worker.worker_id)
            result = await worker.run(job, progress_callback)
            self._set_complete(job, result)
            logger.info("Completed segmentation job %s on worker-%s", job.job_id, worker.worker_id)
        except Exception as exc:
            logger.exception("Segmentation job %s failed on worker-%s", job.job_id, worker.worker_id)
            self._set_failed(job, str(exc))
        finally:
            self._available_workers.put_nowait(worker)
            self._pending_jobs.task_done()

    def _make_progress_callback(self, job: SegmentationJob) -> ProgressCallback:
        loop = asyncio.get_running_loop()

        def callback(status: JobStatus, message: str, progress: float) -> None:
            loop.call_soon_threadsafe(self._set_progress, job.job_id, status, message, progress)

        return callback

    def _set_progress(self, job_id: str, status: JobStatus, message: str, progress: float) -> None:
        job = self._jobs.get(job_id)
        if job is None:
            return

        job.status = status
        job.message = message
        job.progress = max(0.0, min(progress, 1.0))
        job.updated_at = utcnow()

    def _set_complete(self, job: SegmentationJob, result: SegmentationResult) -> None:
        job.status = JobStatus.COMPLETE
        job.message = "Segmentation complete"
        job.progress = 1.0
        job.result = result
        job.updated_at = utcnow()
        if not job.future.done():
            job.future.set_result(job.to_response())

    def _set_failed(self, job: SegmentationJob, error: str) -> None:
        message = error or "Segmentation failed."
        job.status = JobStatus.FAILED
        job.message = "Segmentation failed"
        job.progress = 1.0
        job.error = message
        job.updated_at = utcnow()
        if not job.future.done():
            job.future.set_result(job.to_response())


JobQueue = InferenceQueue