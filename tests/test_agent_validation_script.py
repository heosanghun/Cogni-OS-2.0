from __future__ import annotations

from argparse import Namespace
from hashlib import sha256
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
    def __init__(
        self,
        calls,
        *,
        fail_generation=False,
        response_text="로컬 응답입니다.",
        finish_reason="stop",
    ):
        self.calls = calls
        self.fail_generation = fail_generation
        self.response_text = response_text
        self.finish_reason = finish_reason
        self.worker_pid = None
        self.is_running = False
        self.tokenizer = SimpleNamespace(
            bos_token=None,
            eos_token_id=2,
            apply_chat_template=lambda payload, **_kwargs: (
                "CHAT:" + payload[-1]["content"]
            ),
            decode=lambda _tokens, **_kwargs: self.response_text,
        )

    def start(self):
        self.calls.append("start")
        self.worker_pid = 4321
        self.is_running = True
        return self

    def generate(self, prompt, *, max_new_tokens, stop_token_ids):
        self.calls.append(
            (
                "generate",
                prompt,
                max_new_tokens,
                tuple(map(int, stop_token_ids.tolist())),
            )
        )
        if self.fail_generation:
            raise RuntimeError("bounded fake failure")
        return SimpleNamespace(
            text=self.response_text,
            token_ids=torch.tensor([10, 11, 12], dtype=torch.int64),
            finish_reason=self.finish_reason,
        )

    def stop(self):
        self.calls.append("stop")
        self.is_running = False


def arguments(manifest: str = "C:/local/manifest.toml") -> Namespace:
    return Namespace(
        model="C:/local/gemma",
        manifest=manifest,
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

        with tempfile.TemporaryDirectory() as temporary:
            manifest = Path(temporary) / "gemma4-e4b-it.manifest.toml"
            manifest.write_bytes(b"verified instruction-tuned manifest\n")
            expected_digest = sha256(manifest.read_bytes()).hexdigest()
            metrics, exit_code = execute_validation(
                arguments(str(manifest)),
                verifier=verify,
                service_factory=factory,
                cuda_available=lambda: True,
                cuda_device_name=lambda _index: "Fake CUDA",
            )

        self.assertEqual(exit_code, 0)
        self.assertEqual(metrics["status"], "ok")
        self.assertEqual(metrics["verified_files"], 2)
        self.assertEqual(metrics["response"], "로컬 응답입니다.")
        self.assertEqual(metrics["response_tokens"], 3)
        self.assertEqual(metrics["finish_reason"], "stop")
        self.assertEqual(metrics["quality_gate"], "passed")
        self.assertTrue(metrics["cuda_worker_cleaned"])
        self.assertEqual(calls[0][0], "verify")
        self.assertEqual(calls[1][0], "factory")
        self.assertEqual(calls[-1], "stop")
        self.assertEqual(calls[1][2]["max_new_tokens"], 7)
        self.assertEqual(calls[1][2]["manifest_path"], str(manifest))
        self.assertEqual(calls[1][2]["artifact_digest"], expected_digest)
        self.assertEqual(calls[3], ("generate", "CHAT:테스트 요청", 7, (2,)))

    def test_repetitive_response_fails_the_quality_gate_and_reaps_worker(self):
        calls = []

        def factory(_model, **_kwargs):
            return FakeService(
                calls,
                response_text="요청에 " + "응에 " * 12,
            )

        with tempfile.TemporaryDirectory() as temporary:
            manifest = Path(temporary) / "gemma4-e4b-it.manifest.toml"
            manifest.write_bytes(b"verified instruction-tuned manifest\n")
            metrics, exit_code = execute_validation(
                arguments(str(manifest)),
                verifier=lambda *_args: SimpleNamespace(files=(Path("a"),)),
                service_factory=factory,
                cuda_available=lambda: True,
                cuda_device_name=lambda _index: "Fake CUDA",
            )

        self.assertEqual(exit_code, 1)
        self.assertEqual(metrics["status"], "failed")
        self.assertEqual(metrics["quality_gate"], "failed")
        self.assertIn("quality gate rejected", metrics["error"])
        self.assertEqual(calls[-1], "stop")

    def test_length_finish_fails_even_when_visible_text_is_complete(self):
        calls = []

        def factory(_model, **_kwargs):
            return FakeService(calls, finish_reason="length")

        with tempfile.TemporaryDirectory() as temporary:
            manifest = Path(temporary) / "gemma4-e4b-it.manifest.toml"
            manifest.write_bytes(b"verified instruction-tuned manifest\n")
            metrics, exit_code = execute_validation(
                arguments(str(manifest)),
                verifier=lambda *_args: SimpleNamespace(files=(Path("a"),)),
                service_factory=factory,
                cuda_available=lambda: True,
                cuda_device_name=lambda _index: "Fake CUDA",
            )

        self.assertEqual(exit_code, 1)
        self.assertEqual(metrics["finish_reason"], "length")
        self.assertEqual(metrics["quality_gate"], "failed")
        self.assertIn("terminal stop", metrics["error"])
        self.assertEqual(calls[-1], "stop")

    def test_generation_failure_still_reaps_worker_and_returns_failure(self):
        calls = []

        def factory(_model, **_kwargs):
            calls.append("factory")
            return FakeService(calls, fail_generation=True)

        with tempfile.TemporaryDirectory() as temporary:
            manifest = Path(temporary) / "gemma4-e4b-it.manifest.toml"
            manifest.write_bytes(b"verified instruction-tuned manifest\n")
            metrics, exit_code = execute_validation(
                arguments(str(manifest)),
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
