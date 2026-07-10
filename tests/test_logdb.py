import tempfile
import unittest
from pathlib import Path

from cogni_flow.harness import FailureTrace
from cogni_flow.logdb import LogDB


class TestLogDB(unittest.TestCase):
    def test_local_failure_and_audit_round_trip(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = LogDB(Path(tmp) / "logs.sqlite3")
            db.record_failure(
                FailureTrace("t1", "Timeout", "V1", "tool_loop", "deadline"),
                timestamp=10,
            )
            db.record_failure(
                FailureTrace("t2", "Assertion", "V2", "answer", "wrong"), timestamp=20
            )
            self.assertEqual([trace.test_id for trace in db.failures_since(15)], ["t2"])
            sequence = db.audit("promotion", "pkg/a.py", "tests passed")
            self.assertEqual(sequence, 1)
            self.assertEqual(db.audit_events()[0].kind, "promotion")


if __name__ == "__main__":
    unittest.main()
