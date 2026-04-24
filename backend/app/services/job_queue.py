from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from queue import Queue
from threading import Event, Lock, Thread
from typing import Callable
from uuid import uuid4

from app.models.schemas import JobStatus, JobStatusResponse, SegmentationRequest, SegmentationResult


ProgressCallback = Callable[[JobStatus, str, float], None]
JobWorker = Callable[[SegmentationRequest, ProgressCallback], SegmentationResult]


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


@dataclass(slots=True)
class JobRecord:
    job_id: str
    request: SegmentationRequest
    status: JobStatus = JobStatus.QUEUED
    progress: float = 0.0
    message: str = "Queued"
    created_at: datetime = field(default_factory=utcnow)
    updated_at: datetime = field(default_factory=utcnow)
    result: SegmentationResult | None = None
    error: str | None = None


class JobQueue:
    def __init__(self, worker: JobWorker) -> None:
        self._worker = worker
        self._jobs: dict[str, JobRecord] = {}
        self._queue: Queue[str | None] = Queue()
        self._lock = Lock()
        self._stop_event = Event()
        self._thread = Thread(target=self._run, name="sam3-job-worker", daemon=True)

    def start(self) -> None:
        if not self._thread.is_alive():
            self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        self._queue.put(None)
        self._thread.join(timeout=2)

    def submit(self, payload: SegmentationRequest) -> JobRecord:
        job = JobRecord(job_id=str(uuid4()), request=payload)
        with self._lock:
            self._jobs[job.job_id] = job
        self._queue.put(job.job_id)
        return job

    def get(self, job_id: str) -> JobStatusResponse | None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return None
            return JobStatusResponse(
                job_id=job.job_id,
                session_id=job.request.session_id,
                status=job.status,
                progress=job.progress,
                message=job.message,
                created_at=job.created_at,
                updated_at=job.updated_at,
                result=job.result,
                error=job.error,
            )

    def _run(self) -> None:
        while not self._stop_event.is_set():
            job_id = self._queue.get()
            if job_id is None:
                self._queue.task_done()
                break

            try:
                self._process(job_id)
            finally:
                self._queue.task_done()

    def _process(self, job_id: str) -> None:
        record = self._get_record(job_id)
        if record is None:
            return

        try:
            result = self._worker(record.request, self._make_progress_callback(job_id))
            self._set_complete(job_id, result)
        except Exception as exc:
            self._set_failed(job_id, str(exc))

    def _make_progress_callback(self, job_id: str) -> ProgressCallback:
        def callback(status: JobStatus, message: str, progress: float) -> None:
            with self._lock:
                job = self._jobs.get(job_id)
                if job is None:
                    return
                job.status = status
                job.message = message
                job.progress = max(0.0, min(progress, 1.0))
                job.updated_at = utcnow()

        return callback

    def _set_complete(self, job_id: str, result: SegmentationResult) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return
            job.status = JobStatus.COMPLETE
            job.message = "Segmentation complete"
            job.progress = 1.0
            job.result = result
            job.updated_at = utcnow()

    def _set_failed(self, job_id: str, error: str) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return
            job.status = JobStatus.FAILED
            job.message = "Segmentation failed"
            job.progress = 1.0
            job.error = error
            job.updated_at = utcnow()

    def _get_record(self, job_id: str) -> JobRecord | None:
        with self._lock:
            return self._jobs.get(job_id)
