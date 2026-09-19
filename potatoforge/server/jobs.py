"""Shared in-memory lifecycle and SSE delivery for local jobs."""

from __future__ import annotations

import asyncio
import json
import threading
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

SseEvent = tuple[str, dict[str, Any] | None]


class ActiveJobError(RuntimeError):
    """Raised when a second local job is requested while one is active."""


class UnknownJobError(LookupError):
    """Raised when a client requests a non-current job."""


class InactiveJobError(RuntimeError):
    """Raised when cancellation is requested for a completed job."""


@dataclass(eq=False)
class Subscriber:
    loop: asyncio.AbstractEventLoop
    queue: asyncio.Queue[SseEvent] = field(default_factory=asyncio.Queue)
    active: bool = True
    job: Job | None = None


@dataclass
class Job:
    id: str
    status: str
    started_at: str
    completed_at: str | None = None
    progress: dict[str, Any] | None = None
    result: dict[str, Any] | None = None
    error: str | None = None
    cancel_requested: bool = False
    output_events: list[dict[str, Any]] = field(default_factory=list)
    subscribers: set[Subscriber] = field(default_factory=set)


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


class LocalJobManager:
    """Own one local job, its replayable output, and SSE subscribers."""

    _ACTIVE_STATUSES = frozenset({"starting", "running", "stopping"})
    _TERMINAL_STATUSES = frozenset({"succeeded", "failed", "cancelled"})

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._job: Job | None = None

    def start(self, runner: Callable[[Job], Awaitable[None]]) -> dict[str, Any]:
        with self._lock:
            if self._job is not None and self._job.status in self._ACTIVE_STATUSES:
                raise ActiveJobError
            job = Job(id=str(uuid4()), status="starting", started_at=_timestamp())
            self._job = job
            asyncio.create_task(runner(job))
            return self._snapshot_locked(job)

    def current(self) -> dict[str, Any] | None:
        with self._lock:
            return None if self._job is None else self._snapshot_locked(self._job)

    def cancel(self, job_id: str) -> dict[str, Any]:
        with self._lock:
            job = self._job
            if job is None or job.id != job_id:
                raise UnknownJobError(job_id)
            if job.status not in self._ACTIVE_STATUSES:
                raise InactiveJobError(job_id)
            job.cancel_requested = True
            job.status = "stopping"
            snapshot = self._snapshot_locked(job)
            self._publish_locked(job, "state", snapshot)
            return snapshot

    def subscribe(self, job_id: str) -> Subscriber:
        subscriber = Subscriber(asyncio.get_running_loop())
        with self._lock:
            job = self._job
            if job is None or job.id != job_id:
                raise UnknownJobError(job_id)
            subscriber.job = job
            job.subscribers.add(subscriber)
            snapshot = self._snapshot_locked(job)
            output_events = [dict(event) for event in job.output_events]
            completed = job.status in self._TERMINAL_STATUSES
        subscriber.queue.put_nowait(("state", snapshot))
        for event in output_events:
            subscriber.queue.put_nowait(("output", event))
        if completed:
            subscriber.queue.put_nowait(("complete", snapshot))
        return subscriber

    def remove_subscriber(self, subscriber: Subscriber) -> None:
        with self._lock:
            subscriber.active = False
            if subscriber.job is not None:
                subscriber.job.subscribers.discard(subscriber)

    async def stream(self, subscriber: Subscriber):
        try:
            while True:
                try:
                    name, payload = await asyncio.wait_for(subscriber.queue.get(), timeout=15)
                except asyncio.TimeoutError:
                    yield ": keepalive\n\n"
                    continue
                yield self._encode_event(name, payload)
                if name == "complete":
                    return
        finally:
            self.remove_subscriber(subscriber)

    def set_running(self, job: Job) -> None:
        with self._lock:
            if self._job is not job:
                return
            job.status = "running"
            self._publish_locked(job, "state", self._snapshot_locked(job))

    def is_cancelled(self, job: Job) -> bool:
        with self._lock:
            return self._job is job and job.cancel_requested

    def record_progress(self, job: Job, progress: dict[str, Any], line: str) -> None:
        with self._lock:
            if self._job is not job or job.status not in {"starting", "running"}:
                return
            event = {
                "sequence": len(job.output_events) + 1,
                "line": line,
                "progress": progress,
            }
            job.progress = progress
            job.output_events.append(event)
            self._publish_locked(job, "output", dict(event))

    def finish(
        self,
        job: Job,
        status: str,
        *,
        result: dict[str, Any] | None = None,
        error: str | None = None,
    ) -> None:
        with self._lock:
            if self._job is not job or job.status in self._TERMINAL_STATUSES:
                return
            job.status = status
            job.completed_at = _timestamp()
            job.result = result
            job.error = error
            snapshot = self._snapshot_locked(job)
            self._publish_locked(job, "state", snapshot)
            self._publish_locked(job, "complete", snapshot)

    def _publish_locked(self, job: Job, name: str, payload: dict[str, Any] | None) -> None:
        for subscriber in tuple(job.subscribers):
            try:
                subscriber.loop.call_soon_threadsafe(
                    self._put_event, subscriber, (name, payload)
                )
            except RuntimeError:
                subscriber.active = False
                job.subscribers.discard(subscriber)

    @staticmethod
    def _put_event(subscriber: Subscriber, event: SseEvent) -> None:
        if subscriber.active:
            subscriber.queue.put_nowait(event)

    @staticmethod
    def _snapshot_locked(job: Job) -> dict[str, Any]:
        return {
            "id": job.id,
            "status": job.status,
            "started_at": job.started_at,
            "completed_at": job.completed_at,
            "progress": None if job.progress is None else dict(job.progress),
            "result": None if job.result is None else dict(job.result),
            "error": job.error,
        }

    @staticmethod
    def _encode_event(name: str, payload: dict[str, Any] | None) -> str:
        return f"event: {name}\ndata: {json.dumps(payload, separators=(',', ':'))}\n\n"
