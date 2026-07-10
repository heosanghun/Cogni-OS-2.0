import tempfile
from threading import Event, Thread
import unittest
from unittest.mock import patch
from pathlib import Path

from cogni_flow.harness import (
    FailureTrace,
    PatchPolicy,
    PatchProposal,
    SafeHarnessPatcher,
    SandboxResult,
    file_digest,
    mine_weaknesses,
)
from cogni_flow.rhythm import RhythmController, SystemMode


class PassingKernelSandbox:
    kernel_isolated = True

    def __init__(self) -> None:
        self.calls = 0

    def run(self, project, command, timeout_seconds):
        self.calls += 1
        return SandboxResult(True, 0, "isolated regression suite passed")


class ProcessOnlySpySandbox:
    kernel_isolated = False

    def __init__(self) -> None:
        self.called = False

    def run(self, project, command, timeout_seconds):
        self.called = True
        return SandboxResult(True, 0, "must never execute")


class TestRhythm(unittest.TestCase):
    def test_day_night_exclusion(self):
        rhythm = RhythmController()
        with rhythm.inference_slot():
            with self.assertRaises(RuntimeError):
                rhythm.enter_evolution(lambda: None)
        rhythm.enter_evolution(lambda: None)
        self.assertEqual(rhythm.mode, SystemMode.EVOLUTION)
        with self.assertRaises(RuntimeError):
            with rhythm.inference_slot():
                pass

    def test_active_evolution_task_blocks_concurrent_resume(self):
        rhythm = RhythmController()
        rhythm.enter_evolution(lambda: None)
        entered = Event()
        release = Event()
        errors = []

        def evolution_worker():
            try:
                with rhythm.evolution_slot():
                    entered.set()
                    release.wait(2)
            except Exception as exc:  # pragma: no cover - asserted below
                errors.append(exc)

        worker = Thread(target=evolution_worker)
        worker.start()
        self.assertTrue(entered.wait(1))
        self.assertEqual(rhythm.active_evolution_tasks, 1)
        with self.assertRaisesRegex(RuntimeError, "evolution tasks are active"):
            rhythm.resume_inference("racing resume")
        with self.assertRaisesRegex(RuntimeError, "evolution tasks are active"):
            rhythm.transition(SystemMode.INFERENCE, "direct racing transition")
        self.assertEqual(rhythm.mode, SystemMode.EVOLUTION)
        release.set()
        worker.join(2)
        self.assertFalse(worker.is_alive())
        self.assertEqual(errors, [])
        self.assertEqual(rhythm.active_evolution_tasks, 0)
        rhythm.resume_inference()
        self.assertEqual(rhythm.mode, SystemMode.INFERENCE)


class TestHarness(unittest.TestCase):
    def test_failure_mining_is_deterministic(self):
        traces = [
            FailureTrace("a", "Timeout", "V1", "tool_loop"),
            FailureTrace("b", "Timeout", "V1", "tool_loop"),
            FailureTrace("c", "Assertion", "V2", "answer"),
        ]
        clusters = mine_weaknesses(traces)
        self.assertEqual(len(clusters[0].traces), 2)

    def test_policy_rejects_network_and_path_escape(self):
        policy = PatchPolicy(allowed_roots=("pkg",))
        with self.assertRaises(ValueError):
            policy.validate(PatchProposal("../bad.py", "0", "x=1", "bad"))
        with self.assertRaises(ValueError):
            policy.validate(PatchProposal("pkg/bad.py", "0", "import requests", "bad"))

    def test_policy_rejects_aliased_process_file_and_dynamic_code_bypasses(self):
        policy = PatchPolicy(allowed_roots=("pkg",))
        dangerous_sources = (
            "import os as safe\nsafe.system('whoami')\n",
            "from os import system as harmless\nharmless('whoami')\n",
            "import subprocess as safe\nsafe.run(['python'])\n",
            "import smtplib\nsmtplib.SMTP('host')\n",
            "import shutil as safe\nsafe.rmtree('target')\n",
            "from pathlib import Path\nPath('target').unlink()\n",
            "from pathlib import Path\np = Path('a')\np.replace('b')\n",
            "from pathlib import Path\nPath('a').open('w')\n",
            "class Box: pass\nBox().eval('payload')\n",
            "getattr(object(), 'eval')('payload')\n",
            "__builtins__['eval']('payload')\n",
        )
        for source in dangerous_sources:
            with self.subTest(source=source):
                with self.assertRaises(ValueError):
                    policy.validate(PatchProposal("pkg/bad.py", "0", source, "bad"))

    def test_policy_allows_read_only_os_and_path_operations(self):
        policy = PatchPolicy(allowed_roots=("pkg",))
        source = (
            "import os\n"
            "from pathlib import Path\n"
            "NAME = os.path.basename(Path('a/b').as_posix())\n"
        )
        self.assertEqual(
            policy.validate(PatchProposal("pkg/good.py", "0", source, "safe")),
            Path("pkg/good.py"),
        )

    def test_regression_gated_atomic_promotion(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "pkg").mkdir()
            (root / "tests").mkdir()
            target = root / "pkg" / "value.py"
            target.write_text("VALUE = 1\n", encoding="utf-8")
            (root / "tests" / "test_value.py").write_text(
                "import unittest\nfrom pkg.value import VALUE\n"
                "class T(unittest.TestCase):\n    def test_value(self): self.assertEqual(VALUE, 2)\n",
                encoding="utf-8",
            )
            rhythm = RhythmController()
            rhythm.enter_evolution(lambda: None)
            patcher = SafeHarnessPatcher(
                root,
                rhythm,
                policy=PatchPolicy(allowed_roots=("pkg",)),
                sandbox=PassingKernelSandbox(),
                test_command=("isolated-test",),
            )
            result = patcher.validate_and_promote(
                PatchProposal("pkg/value.py", file_digest(target), "VALUE = 2\n", "fix")
            )
            self.assertTrue(result.promoted)
            self.assertEqual(target.read_text(encoding="utf-8"), "VALUE = 2\n")
            rhythm.resume_inference()
            self.assertEqual(rhythm.mode, SystemMode.INFERENCE)

    def test_production_gate_rejects_process_only_sandbox(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "pkg").mkdir()
            target = root / "pkg" / "a.py"
            target.write_text("x=1\n", encoding="utf-8")
            rhythm = RhythmController()
            rhythm.enter_evolution(lambda: None)
            sandbox = ProcessOnlySpySandbox()
            patcher = SafeHarnessPatcher(
                root,
                rhythm,
                policy=PatchPolicy(allowed_roots=("pkg",)),
                sandbox=sandbox,
            )
            with self.assertRaises(RuntimeError):
                patcher.validate_and_promote(
                    PatchProposal("pkg/a.py", file_digest(target), "x=2\n", "fix")
                )
            self.assertFalse(sandbox.called)
            self.assertEqual(rhythm.active_evolution_tasks, 0)
            self.assertEqual(rhythm.mode, SystemMode.EVOLUTION)

    def test_kernel_requirement_cannot_be_disabled(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "pkg").mkdir()
            target = root / "pkg" / "a.py"
            target.write_text("x=1\n", encoding="utf-8")
            rhythm = RhythmController()
            rhythm.enter_evolution(lambda: None)
            sandbox = ProcessOnlySpySandbox()
            patcher = SafeHarnessPatcher(
                root,
                rhythm,
                policy=PatchPolicy(allowed_roots=("pkg",)),
                sandbox=sandbox,
                require_kernel_isolation=False,
            )
            with self.assertRaisesRegex(RuntimeError, "cannot be disabled"):
                patcher.validate_and_promote(
                    PatchProposal("pkg/a.py", file_digest(target), "x=2\n", "fix")
                )
            self.assertFalse(sandbox.called)

    def test_resume_is_blocked_through_atomic_promotion(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "pkg").mkdir()
            target = root / "pkg" / "a.py"
            target.write_text("x=1\n", encoding="utf-8")
            rhythm = RhythmController()
            rhythm.enter_evolution(lambda: None)
            patcher = SafeHarnessPatcher(
                root,
                rhythm,
                policy=PatchPolicy(allowed_roots=("pkg",)),
                sandbox=PassingKernelSandbox(),
                test_command=("isolated-test",),
            )
            promotion_entered = Event()
            release_promotion = Event()
            result = []
            errors = []
            real_replace = __import__("os").replace

            def blocking_replace(source, destination):
                promotion_entered.set()
                release_promotion.wait(2)
                real_replace(source, destination)

            def promote_worker():
                try:
                    result.append(
                        patcher.validate_and_promote(
                            PatchProposal(
                                "pkg/a.py", file_digest(target), "x=2\n", "fix"
                            )
                        )
                    )
                except Exception as exc:  # pragma: no cover - asserted below
                    errors.append(exc)

            with patch("cogni_flow.harness.os.replace", side_effect=blocking_replace):
                worker = Thread(target=promote_worker)
                worker.start()
                self.assertTrue(promotion_entered.wait(1))
                self.assertEqual(rhythm.mode, SystemMode.PROMOTING)
                self.assertEqual(rhythm.active_evolution_tasks, 1)
                with self.assertRaisesRegex(RuntimeError, "evolution tasks are active"):
                    rhythm.resume_inference("promotion race")
                release_promotion.set()
                worker.join(2)

            self.assertFalse(worker.is_alive())
            self.assertEqual(errors, [])
            self.assertTrue(result[0].promoted)
            self.assertEqual(target.read_text(encoding="utf-8"), "x=2\n")
            self.assertEqual(rhythm.active_evolution_tasks, 0)
            rhythm.resume_inference()
            self.assertEqual(rhythm.mode, SystemMode.INFERENCE)


if __name__ == "__main__":
    unittest.main()
