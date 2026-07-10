import tempfile
import unittest
from pathlib import Path
from threading import Event

from cogni_flow.daemon import (
    FailureCaptureDaemon,
    FailureQueueOverflow,
    FailureWriterError,
)
from cogni_flow.harness import FailureTrace
from cogni_flow.logdb import LogDB


class BlockingLogDB(LogDB):
    def __init__(self, path: Path):
        super().__init__(path)
        self.entered = Event()
        self.release = Event()

    def record_failure(self, trace, timestamp=None):
        self.entered.set()
        if not self.release.wait(timeout=2):
            raise TimeoutError("test writer was not released")
        return super().record_failure(trace, timestamp)


class FailingLogDB(LogDB):
    def record_failure(self, trace, timestamp=None):
        raise OSError("disk unavailable")


class TestFailureCaptureDaemon(unittest.TestCase):
    def test_context_captures_exception_and_timeout_asynchronously(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = LogDB(Path(tmp) / "events.sqlite3")
            daemon = FailureCaptureDaemon(db, capacity=4)
            with daemon:
                daemon.capture_exception("wf-value", ValueError("bad value"))
                daemon.capture_timeout("wf-slow", 1.5)
                with self.assertRaises(KeyError):
                    with daemon.observe("wf-context", mechanism="node"):
                        raise KeyError("missing")
                daemon.flush()
                self.assertTrue(daemon.running)

            self.assertFalse(daemon.running)
            traces = db.failures_since(0)
            self.assertEqual(
                [(trace.test_id, trace.exception_type) for trace in traces],
                [
                    ("wf-value", "ValueError"),
                    ("wf-slow", "Timeout"),
                    ("wf-context", "KeyError"),
                ],
            )
            self.assertEqual(daemon.stats.submitted, 3)
            self.assertEqual(daemon.stats.recorded, 3)
            self.assertEqual(
                [
                    event.detail
                    for event in db.audit_events()
                    if event.kind == "failure_daemon"
                ],
                ["started", "stopped"],
            )

    def test_full_queue_is_audited_and_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = BlockingLogDB(Path(tmp) / "events.sqlite3")
            daemon = FailureCaptureDaemon(db, capacity=1)
            daemon.start()
            try:
                daemon.submit(FailureTrace("wf-1", "Error", "V", "node"))
                self.assertTrue(db.entered.wait(timeout=1))
                daemon.submit(FailureTrace("wf-2", "Error", "V", "node"))
                with self.assertRaises(FailureQueueOverflow):
                    daemon.submit(FailureTrace("wf-3", "Error", "V", "node"))
            finally:
                db.release.set()
                daemon.stop()

            self.assertEqual(
                [trace.test_id for trace in db.failures_since(0)], ["wf-1", "wf-2"]
            )
            overflow = [
                event
                for event in db.audit_events()
                if event.kind == "failure_queue_overflow"
            ]
            self.assertEqual(len(overflow), 1)
            self.assertEqual(overflow[0].subject, "wf-3")
            self.assertEqual(daemon.stats.rejected, 1)

    def test_submit_requires_a_live_daemon(self):
        with tempfile.TemporaryDirectory() as tmp:
            daemon = FailureCaptureDaemon(LogDB(Path(tmp) / "events.sqlite3"))
            with self.assertRaisesRegex(RuntimeError, "not running"):
                daemon.submit(FailureTrace("wf", "Error", "V", "node"))

    def test_writer_failure_is_not_hidden_by_clean_thread_exit(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = FailingLogDB(Path(tmp) / "events.sqlite3")
            daemon = FailureCaptureDaemon(db)
            daemon.start()
            daemon.submit(FailureTrace("wf", "Error", "V", "node"))
            with self.assertRaisesRegex(FailureWriterError, "persistence"):
                daemon.stop()
            self.assertFalse(daemon.running)
            self.assertEqual(
                [event.kind for event in db.audit_events()],
                ["failure_daemon", "failure_writer_error", "failure_daemon"],
            )


if __name__ == "__main__":
    unittest.main()
