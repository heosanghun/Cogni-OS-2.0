from __future__ import annotations

from argparse import Namespace
from io import StringIO
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest

import torch

from scripts.validate_agent_runtime import (
    JSON_PREFIX,
    build_parser,
    emit_metrics,
    execute_validation,
)


class FakeService:
    def __init__(self, calls, *, fail_generation=False):
        self.calls = calls
        self.fail_generation = fail_generation
        self.worker_pid = None
        self.is_running = False

    def start(self):
        self.calls.append("start")
        self.worker_pid = 4321
        self.is_running = True
        return self

    def generate(self, prompt, *, max_new_tokens):
        self.calls.append(("generate", prompt, max_new_tokens))
        if self.fail_generation:
            raise RuntimeError("bounded fake failure")
        return SimpleNamespace(
            text="로컬 응답",
            token_ids=torch.tensor([10, 11, 12], dtype=torch.int64),
        )

    def stop(self):
        self.calls.append("stop")
        self.is_running = False


def arguments() -> Namespace:
    return Namespace(
        model="C:/local/gemma",
        manifest="C:/local/manifest.toml",
        prompt="테스트 요청",
        max_new_tokens=7,
    )


class TestAgentRuntimeValidationScript(unittest.TestCase):
    def test_manifest_precedes_worker_and_success_reports_cleanup(self):
        calls = []

        def verify(model, manifest):
            calls.append(("verify", model, manifest))
            return SimpleNamespace(files=(Path("a"), Path("b")))

        def factory(model, **kwargs):
            calls.append(("factory", model, kwargs))
            return FakeService(calls)

        metrics, exit_code = execute_validation(
            arguments(),
            verifier=verify,
            service_factory=factory,
            cuda_available=lambda: True,
            cuda_device_name=lambda _index: "Fake CUDA",
        )

        self.assertEqual(exit_code, 0)
        self.assertEqual(metrics["status"], "ok")
        self.assertEqual(metrics["verified_files"], 2)
        self.assertEqual(metrics["response"], "로컬 응답")
        self.assertEqual(metrics["response_tokens"], 3)
        self.assertTrue(metrics["cuda_worker_cleaned"])
        self.assertEqual(calls[0][0], "verify")
        self.assertEqual(calls[1][0], "factory")
        self.assertEqual(calls[-1], "stop")
        self.assertEqual(calls[1][2]["max_new_tokens"], 7)

    def test_generation_failure_still_reaps_worker_and_returns_failure(self):
        calls = []

        def factory(_model, **_kwargs):
            calls.append("factory")
            return FakeService(calls, fail_generation=True)

        metrics, exit_code = execute_validation(
            arguments(),
            verifier=lambda *_args: SimpleNamespace(files=(Path("a"),)),
            service_factory=factory,
            cuda_available=lambda: True,
            cuda_device_name=lambda _index: "Fake CUDA",
        )
        self.assertEqual(exit_code, 1)
        self.assertEqual(metrics["status"], "failed")
        self.assertIn("RuntimeError", metrics["error"])
        self.assertTrue(metrics["cuda_worker_cleaned"])
        self.assertEqual(calls[-1], "stop")

    def test_manifest_and_cuda_failures_never_construct_a_worker(self):
        calls = []

        def factory(*_args, **_kwargs):
            calls.append("factory")
            raise AssertionError("worker factory must not run")

        failed_manifest, manifest_code = execute_validation(
            arguments(),
            verifier=lambda *_args: (_ for _ in ()).throw(RuntimeError("bad manifest")),
            service_factory=factory,
            cuda_available=lambda: True,
        )
        self.assertEqual(manifest_code, 1)
        self.assertFalse(failed_manifest["worker_started"])

        failed_cuda, cuda_code = execute_validation(
            arguments(),
            verifier=lambda *_args: SimpleNamespace(files=(Path("a"),)),
            service_factory=factory,
            cuda_available=lambda: False,
        )
        self.assertEqual(cuda_code, 1)
        self.assertFalse(failed_cuda["worker_started"])
        self.assertEqual(calls, [])

    def test_json_and_human_metrics_share_the_same_result(self):
        metrics = {
            "status": "ok",
            "verified_files": 2,
            "response": "한 줄\n응답",
            "response_tokens": 3,
            "use_cache": False,
            "fast_weight": "gated_off",
            "fp_ewc": "excluded",
            "cuda_worker_cleaned": True,
            "elapsed_seconds": 1.25,
        }
        stream = StringIO()
        emit_metrics(metrics, stream)
        lines = stream.getvalue().splitlines()
        payload = json.loads(
            next(line for line in lines if line.startswith(JSON_PREFIX))[
                len(JSON_PREFIX) :
            ]
        )
        self.assertEqual(payload, metrics)
        self.assertIn('response="한 줄\\n응답"', lines)

    def test_argument_bounds_and_isolated_help_bootstrap(self):
        parser = build_parser()
        for value in ("0", "513"):
            with self.subTest(value=value), self.assertRaises(SystemExit):
                parser.parse_args(
                    ["--model", "m", "--manifest", "x", "--max-new-tokens", value]
                )
        with self.assertRaises(SystemExit):
            parser.parse_args(
                ["--model", "m", "--manifest", "x", "--prompt", "x" * 8_193]
            )

        root = Path(__file__).resolve().parents[1]
        environment = os.environ.copy()
        environment.pop("PYTHONPATH", None)
        with tempfile.TemporaryDirectory() as other_directory:
            completed = subprocess.run(
                [
                    sys.executable,
                    "-I",
                    str(root / "scripts" / "validate_agent_runtime.py"),
                    "--help",
                ],
                cwd=other_directory,
                env=environment,
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
            )
        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        self.assertIn("usage:", completed.stdout.lower())


if __name__ == "__main__":
    unittest.main()
