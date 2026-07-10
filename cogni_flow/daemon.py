from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from queue import Empty, Full, Queue
from threading import Event, RLock, Thread
from time import monotonic, time
from typing import Iterator

from .harness import FailureTrace
from .logdb import LogDB


class FailureQueueOverflow(RuntimeError):
    """Raised when a failure cannot be durably admitted to the bounded queue."""


class FailureWriterError(RuntimeError):
    """Raised after the asynchronous database writer has failed."""


@dataclass(frozen=True)
class CaptureStats:
    submitted: int
    recorded: int
    rejected: int


@dataclass(frozen=True)
class _QueuedFailure:
    trace: FailureTrace
    timestamp: float


class FailureCaptureDaemon:
    """Bounded, fail-closed bridge from workflow failures to local ``LogDB``.

    Producers never write SQLite on their normal path. Queue admission is
    non-blocking so an overloaded logger cannot stall an inference workflow.
    A full queue is *not* treated as success: an audit write is attempted and
    ``FailureQueueOverflow`` is raised to make the loss visible to the caller.
    """

    def __init__(
        self,
        logdb: LogDB,
        *,
        capacity: int = 256,
        excerpt_limit: int = 4_000,
        stop_timeout: float = 5.0,
    ) -> None:
        if capacity < 1:
            raise ValueError("capacity must be positive")
        if excerpt_limit < 1:
            raise ValueError("excerpt_limit must be positive")
        if stop_timeout <= 0:
            raise ValueError("stop_timeout must be positive")
        self.logdb = logdb
        self.capacity = capacity
        self.excerpt_limit = excerpt_limit
        self.stop_timeout = stop_timeout
        self._queue: Queue[_QueuedFailure] = Queue(maxsize=capacity)
        self._stop_requested = Event()
        self._thread: Thread | None = None
        self._lock = RLock()
        self._accepting = False
        self._writer_error: BaseException | None = None
        self._submitted = 0
        self._recorded = 0
        self._rejected = 0

    @property
    def running(self) -> bool:
        with self._lock:
            return bool(
                self._accepting and self._thread is not None and self._thread.is_alive()
            )

    @property
    def stats(self) -> CaptureStats:
        with self._lock:
            return CaptureStats(self._submitted, self._recorded, self._rejected)

    def start(self) -> FailureCaptureDaemon:
        with self._lock:
            if self._accepting:
                return self
            if self._thread is not None and self._thread.is_alive():
                raise RuntimeError("failure capture daemon is still stopping")
            self._queue = Queue(maxsize=self.capacity)
            self._stop_requested.clear()
            self._writer_error = None
            self._accepting = True
            self._thread = Thread(
                target=self._writer_loop,
                name="cogni-failure-writer",
                daemon=True,
            )
            self._thread.start()
        self._audit("failure_daemon", "capture", "started")
        return self

    def stop(self, timeout: float | None = None) -> None:
        wait_seconds = self.stop_timeout if timeout is None else timeout
        if wait_seconds <= 0:
            raise ValueError("timeout must be positive")
        with self._lock:
            thread = self._thread
            if thread is None:
                self._accepting = False
                return
            self._accepting = False
            self._stop_requested.set()

        thread.join(wait_seconds)
        if thread.is_alive():
            self._audit(
                "failure_daemon_stop_timeout",
                "capture",
                f"pending={self._queue.qsize()}",
            )
            raise TimeoutError("failure capture daemon did not stop cleanly")

        with self._lock:
            self._thread = None
            writer_error = self._writer_error
        self._audit("failure_daemon", "capture", "stopped")
        if writer_error is not None:
            raise FailureWriterError(
                "failure writer stopped after a persistence error"
            ) from writer_error

    def submit(self, trace: FailureTrace, *, timestamp: float | None = None) -> None:
        item = _QueuedFailure(trace, time() if timestamp is None else timestamp)
        with self._lock:
            if not self._accepting:
                raise RuntimeError("failure capture daemon is not running")
            if self._writer_error is not None:
                self._rejected += 1
                raise FailureWriterError(
                    "failure writer is unhealthy; refusing new events"
                ) from self._writer_error
            try:
                self._queue.put_nowait(item)
            except Full as exc:
                self._rejected += 1
                detail = f"capacity={self.capacity};exception={trace.exception_type}"
                self._audit("failure_queue_overflow", trace.test_id, detail)
                raise FailureQueueOverflow(
                    "failure queue is full; event was not accepted"
                ) from exc
            self._submitted += 1

    def capture_exception(
        self,
        workflow_id: str,
        error: BaseException,
        *,
        verifier_code: str = "workflow_runtime",
        mechanism: str = "workflow",
        excerpt: str | None = None,
    ) -> None:
        exception_type = (
            "Timeout" if isinstance(error, TimeoutError) else type(error).__name__
        )
        message = str(error) if excerpt is None else excerpt
        self.submit(
            FailureTrace(
                workflow_id,
                exception_type,
                verifier_code,
                mechanism,
                self._bounded_excerpt(message),
            )
        )

    def capture_timeout(
        self,
        workflow_id: str,
        timeout_seconds: float,
        *,
        verifier_code: str = "workflow_deadline",
        mechanism: str = "workflow",
        excerpt: str = "",
    ) -> None:
        detail = excerpt or f"deadline exceeded after {timeout_seconds:g}s"
        self.submit(
            FailureTrace(
                workflow_id,
                "Timeout",
                verifier_code,
                mechanism,
                self._bounded_excerpt(detail),
            )
        )

    @contextmanager
    def observe(
        self,
        workflow_id: str,
        *,
        verifier_code: str = "workflow_runtime",
        mechanism: str = "workflow",
    ) -> Iterator[None]:
        """Capture an exception asynchronously and preserve normal propagation."""

        try:
            yield
        except Exception as exc:
            self.capture_exception(
                workflow_id,
                exc,
                verifier_code=verifier_code,
                mechanism=mechanism,
            )
            raise

    def flush(self, timeout: float | None = None) -> None:
        """Wait for accepted events without admitting new failure semantics."""

        wait_seconds = self.stop_timeout if timeout is None else timeout
        if wait_seconds <= 0:
            raise ValueError("timeout must be positive")
        deadline = monotonic() + wait_seconds
        while self._queue.unfinished_tasks:
            if monotonic() >= deadline:
                raise TimeoutError("timed out waiting for failure records to flush")
            self._stop_requested.wait(0.01)
        with self._lock:
            error = self._writer_error
        if error is not None:
            raise FailureWriterError(
                "failure writer could not persist an event"
            ) from error

    def __enter__(self) -> FailureCaptureDaemon:
        return self.start()

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.stop()

    def _writer_loop(self) -> None:
        while True:
            try:
                item = self._queue.get(timeout=0.05)
            except Empty:
                if self._stop_requested.is_set() and self._queue.empty():
                    return
                continue
            try:
                self.logdb.record_failure(item.trace, timestamp=item.timestamp)
                with self._lock:
                    self._recorded += 1
            except BaseException as exc:
                with self._lock:
                    if self._writer_error is None:
                        self._writer_error = exc
                self._audit(
                    "failure_writer_error",
                    item.trace.test_id,
                    type(exc).__name__,
                )
            finally:
                self._queue.task_done()

    def _bounded_excerpt(self, value: str) -> str:
        normalized = value.replace("\x00", "").strip()
        return normalized[-self.excerpt_limit :]

    def _audit(self, kind: str, subject: str, detail: str) -> None:
        try:
            self.logdb.audit(kind, subject, detail)
        except Exception:
            # Audit failure must not disguise the primary queue/writer error.
            # The caller still receives the fail-closed exception.
            return
