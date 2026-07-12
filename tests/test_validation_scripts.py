from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


class TestValidationScriptBootstrap(unittest.TestCase):
    def test_readme_direct_commands_import_checkout_from_any_working_directory(self):
        root = Path(__file__).resolve().parents[1]
        scripts = (
            "validate_agent_casual_korean.py",
            "validate_agent_completion.py",
            "validate_gemma4.py",
            "validate_gemma4_deq.py",
            "validate_gemma4_runtime.py",
        )
        environment = os.environ.copy()
        environment.pop("PYTHONPATH", None)
        with tempfile.TemporaryDirectory() as other_directory:
            for name in scripts:
                with self.subTest(script=name):
                    completed = subprocess.run(
                        [
                            sys.executable,
                            "-I",
                            str(root / "scripts" / name),
                            "--help",
                        ],
                        cwd=other_directory,
                        env=environment,
                        capture_output=True,
                        text=True,
                        timeout=30,
                        check=False,
                    )
                    self.assertEqual(
                        completed.returncode,
                        0,
                        msg=completed.stdout + completed.stderr,
                    )
                    self.assertIn("usage:", completed.stdout.lower())

    def test_agent_completion_checks_reject_repeated_sentences(self):
        from scripts.validate_agent_completion import _answer_checks

        state = {"status": "succeeded", "stage": "complete"}
        answer = {
            "content": "한 번만 자연스럽게 설명하고 끝냅니다.",
            "finish_reason": "stop",
            "truncated": False,
        }
        self.assertTrue(all(_answer_checks(answer, state).values()))
        answer["content"] = (
            "같은 문장을 반복해서 출력하면 안 됩니다. "
            "같은 문장을 반복해서 출력하면 안 됩니다"
        )
        self.assertFalse(_answer_checks(answer, state)["no_repeated_sentence"])

    def test_agent_completion_checks_reject_empty_list_and_underfilled_request(self):
        from scripts.validate_agent_completion import _answer_checks

        state = {"status": "succeeded", "stage": "complete"}
        answer = {
            "content": "원칙은 다음과 같습니다:\n\n1.",
            "finish_reason": "stop",
            "truncated": False,
        }
        checks = _answer_checks(answer, state, "원칙을 두 문장으로 설명하세요.")
        self.assertFalse(checks["korean_complete"])
        self.assertFalse(checks["request_contract_fulfilled"])


if __name__ == "__main__":
    unittest.main()
