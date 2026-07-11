"""One-turn offline GPU validation for the resident Cogni-Agent runtime."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from time import perf_counter
from typing import Any, Callable, Sequence, TextIO

import torch

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from cogni_agent.model_service import ModelService  # noqa: E402
from cogni_os.artifacts import verify_artifact_manifest  # noqa: E402


MAX_PROMPT_CHARS = 8_192
MAX_NEW_TOKENS = 512
JSON_PREFIX = "AGENT_RUNTIME_JSON="


def _bounded_prompt(value: str) -> str:
    if not value or len(value) > MAX_PROMPT_CHARS:
        raise argparse.ArgumentTypeError(
            f"prompt must contain 1-{MAX_PROMPT_CHARS} characters"
        )
    if any(ord(character) < 32 and character not in "\t\r\n" for character in value):
        raise argparse.ArgumentTypeError(
            "prompt contains unsupported control characters"
        )
    return value


def _bounded_new_tokens(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("max-new-tokens must be an integer") from exc
    if not 1 <= parsed <= MAX_NEW_TOKENS:
        raise argparse.ArgumentTypeError(
            f"max-new-tokens must be in [1, {MAX_NEW_TOKENS}]"
        )
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Verify a local Gemma manifest, run BIO-HAMA -> DEQ/CTS -> System4 "
            "-> System3 -> cache-free Gemma decode once, and reap the worker."
        )
    )
    parser.add_argument("--model", required=True, help="verified local Gemma directory")
    parser.add_argument("--manifest", required=True, help="local SHA-256 manifest")
    parser.add_argument(
        "--prompt",
        type=_bounded_prompt,
        default="Cogni-OS의 현재 안전 상태를 한 문장으로 설명하세요.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=_bounded_new_tokens,
        default=128,
    )
    return parser


def _offline_environment() -> None:
    os.environ.update(
        {
            "HF_HUB_OFFLINE": "1",
            "HF_HUB_DISABLE_TELEMETRY": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "HF_DATASETS_OFFLINE": "1",
            "WANDB_MODE": "offline",
            "TOKENIZERS_PARALLELISM": "false",
            "PYTHONUTF8": "1",
        }
    )


def _bounded_error(error: BaseException) -> str:
    text = f"{type(error).__name__}: {error}".replace("\x00", "")
    text = " ".join(text.split())
    return text[:256]


def execute_validation(
    args: argparse.Namespace,
    *,
    verifier: Callable[..., Any] = verify_artifact_manifest,
    service_factory: Callable[..., ModelService] = ModelService.for_local_gemma,
    cuda_available: Callable[[], bool] = torch.cuda.is_available,
    cuda_device_name: Callable[[int], str] = torch.cuda.get_device_name,
) -> tuple[dict[str, Any], int]:
    """Execute one bounded turn and always return cleanup-aware metrics."""

    _offline_environment()
    started = perf_counter()
    service: ModelService | None = None
    exit_code = 1
    metrics: dict[str, Any] = {
        "status": "failed",
        "verified_files": 0,
        "worker_started": False,
        "worker_pid": None,
        "cuda_worker_cleaned": True,
        "response": "",
        "response_tokens": 0,
        "core_path": "BIO-HAMA>Gemma-feature>DEQ/CTS>System4>System3>Gemma-decode",
        "use_cache": False,
        "fast_weight": "gated_off",
        "fp_ewc": "excluded",
    }
    try:
        # Integrity verification deliberately precedes CUDA/model-process work.
        verified = verifier(args.model, args.manifest)
        metrics["verified_files"] = len(verified.files)
        if not cuda_available():
            raise RuntimeError("agent runtime validation requires a CUDA device")
        metrics["cuda_device"] = cuda_device_name(0)

        load_started = perf_counter()
        service = service_factory(
            args.model,
            max_new_tokens=args.max_new_tokens,
        )
        service.start()
        metrics["worker_started"] = True
        metrics["worker_pid"] = service.worker_pid
        metrics["worker_start_seconds"] = perf_counter() - load_started

        turn_started = perf_counter()
        result = service.generate(
            args.prompt,
            max_new_tokens=args.max_new_tokens,
        )
        metrics["turn_seconds"] = perf_counter() - turn_started
        metrics["response"] = result.text
        metrics["response_tokens"] = int(result.token_ids.numel())
        metrics["status"] = "ok"
        exit_code = 0
    except (Exception, KeyboardInterrupt) as error:
        metrics["error"] = _bounded_error(error)
    finally:
        if service is not None:
            try:
                service.stop()
                metrics["cuda_worker_cleaned"] = not service.is_running
            except Exception as cleanup_error:
                metrics["cuda_worker_cleaned"] = False
                metrics["cleanup_error"] = _bounded_error(cleanup_error)
                metrics["status"] = "failed"
                exit_code = 1
        metrics["elapsed_seconds"] = perf_counter() - started
    return metrics, exit_code


def emit_metrics(metrics: dict[str, Any], stream: TextIO = sys.stdout) -> None:
    ordered = (
        "status",
        "verified_files",
        "cuda_device",
        "worker_started",
        "worker_pid",
        "worker_start_seconds",
        "turn_seconds",
        "response_tokens",
        "use_cache",
        "fast_weight",
        "fp_ewc",
        "core_path",
        "cuda_worker_cleaned",
        "elapsed_seconds",
        "error",
        "cleanup_error",
    )
    for name in ordered:
        if name in metrics:
            value = metrics[name]
            if isinstance(value, float):
                value = f"{value:.6f}"
            print(f"{name}={value}", file=stream)
    print(
        "response=" + json.dumps(metrics.get("response", ""), ensure_ascii=False),
        file=stream,
    )
    print(
        JSON_PREFIX
        + json.dumps(
            metrics, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ),
        file=stream,
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    metrics, exit_code = execute_validation(args)
    emit_metrics(metrics, sys.stdout if exit_code == 0 else sys.stderr)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
