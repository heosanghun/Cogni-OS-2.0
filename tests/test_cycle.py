import sys
import tempfile
import unittest
from pathlib import Path

from cogni_flow.cycle import SelfHarness
from cogni_flow.harness import (
    FailureTrace,
    PatchPolicy,
    PatchProposal,
    SafeHarnessPatcher,
    SubprocessSandbox,
    file_digest,
)
from cogni_flow.logdb import LogDB
from cogni_flow.rhythm import RhythmController, SystemMode


class TestSelfHarnessCycle(unittest.TestCase):
    def test_cycle_promotes_first_regression_safe_candidate(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "pkg").mkdir()
            (root / "tests").mkdir()
            target = root / "pkg" / "value.py"
            target.write_text("VALUE=0\n", encoding="utf-8")
            (root / "tests" / "test_value.py").write_text(
                "import unittest\nfrom pkg.value import VALUE\n"
                "class T(unittest.TestCase):\n def test_v(self): self.assertEqual(VALUE,1)\n",
                encoding="utf-8",
            )
            rhythm = RhythmController()
            db = LogDB(root / "events.sqlite3")
            db.record_failure(FailureTrace("t", "Assertion", "V", "answer"))

            class KernelIsolatedTestRunner(SubprocessSandbox):
                # Test double only: production supplies an actual VM/container
                # implementation with the same explicit trust marker.
                kernel_isolated = True

            patcher = SafeHarnessPatcher(
                root,
                rhythm,
                policy=PatchPolicy(allowed_roots=("pkg",)),
                sandbox=KernelIsolatedTestRunner(),
                test_command=(
                    sys.executable,
                    "-m",
                    "unittest",
                    "discover",
                    "-s",
                    "tests",
                ),
            )

            def proposer(cluster):
                return [
                    PatchProposal(
                        "pkg/value.py", file_digest(target), "VALUE=1\n", "repair"
                    )
                ]

            report = SelfHarness(
                rhythm, db, patcher, proposer, lambda: None
            ).run_night_cycle()
            self.assertTrue(report.promoted)
            self.assertEqual(rhythm.mode, SystemMode.INFERENCE)
            self.assertEqual(len(db.audit_events()), 1)


if __name__ == "__main__":
    unittest.main()
