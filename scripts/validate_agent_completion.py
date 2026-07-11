"""Offline, real-model regression for complete multi-turn Cogni-Agent answers."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import sys
from time import monotonic, sleep
from typing import Any, Sequence

import torch

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from cogni_agent.manager import ACTIVE_AGENT_STATUSES, AgentManager  # noqa: E402
from cogni_agent.model_service import ModelService  # noqa: E402
from cogni_agent.tools import WorkspaceToolExecutor  # noqa: E402
from cogni_os.artifacts import verify_artifact_manifest  # noqa: E402


DEFAULT_PROMPTS = (
    "'로컬 실행 검증을 완료했습니다.'라는 문장을 그대로 한 번만 출력하세요.",
    "Cogni-OS의 CTS, System 1.5, System 2.5, System 3, System 4를 각각 한 문장씩 "
    "설명하세요. 검증된 기능과 설계 목표를 구분하고 마지막은 반드시 '이상입니다.'로 끝내세요.",
    "방금 답변에서 실제 검증과 향후 목표를 구분하는 원칙만 두 문장으로 요약하고 마침표로 끝내세요.",
)
_ROLE_LEAK = re.compile(
    r"(?im)^\s*(?:USER|ASSISTANT|SYSTEM|MODEL|TOOL|사용자|어시스턴트|시스템)\s*:"
)
_CONTROL_MARKERS = (
    "<|turn>",
    "<turn|>",
    "<|channel>",
    "<channel|>",
    "<|tool_response>",
    "<unused",
    "[multimodal]",
    "<|endoftext|>",
    "<|startoftext|>",
)
_COMPLETE_ENDINGS = (".", "!", "?", "。", "！", "？", ".”", ".'", '."')


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


def _wait_for_turn(manager: AgentManager, timeout: float) -> dict[str, Any]:
    deadline = monotonic() + timeout
    while monotonic() < deadline:
        state = manager.snapshot()
        if state["status"] not in ACTIVE_AGENT_STATUSES:
            return state
        sleep(0.05)
    manager.cancel()
    raise TimeoutError("real-model agent turn exceeded its bounded deadline")


def _answer_checks(answer: dict[str, Any], state: dict[str, Any]) -> dict[str, bool]:
    text = str(answer.get("content", "")).strip()
    return {
        "succeeded": state.get("status") == "succeeded",
        "complete_stage": state.get("stage") == "complete",
        "finish_stop": answer.get("finish_reason") == "stop",
        "not_truncated": answer.get("truncated") is False,
        "non_empty": bool(text),
        # Short noun/name answers can be complete without punctuation. Longer
        # prose must expose an actual sentence boundary as well as model stop.
        "natural_boundary": len(text) < 80 or text.endswith(_COMPLETE_ENDINGS),
        "no_role_leak": _ROLE_LEAK.search(text) is None,
        "no_control_marker": not any(marker in text for marker in _CONTROL_MARKERS),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run three offline Gemma 4 chat turns and verify clean completion."
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--timeout", type=float, default=600.0)
    return parser


def execute(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    _offline_environment()
    report: dict[str, Any] = {
        "status": "failed",
        "turns": [],
        "all_checks_passed": False,
    }
    service: ModelService | None = None
    manager: AgentManager | None = None
    try:
        verified = verify_artifact_manifest(args.model, args.manifest)
        report["verified_files"] = len(verified.files)
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for the real-model completion test")
        report["cuda_device"] = torch.cuda.get_device_name(0)
        service = ModelService.for_local_gemma(
            args.model,
            vram_limit_gib=16.7,
            max_input_tokens=4_096,
            max_new_tokens=512,
            max_prompt_chars=32_000,
            max_response_chars=32_000,
            request_timeout=240.0,
        )
        manager = AgentManager(
            service,
            WorkspaceToolExecutor(_PROJECT_ROOT),
        )
        for prompt in DEFAULT_PROMPTS:
            manager.start_turn(prompt, "chat")
            state = _wait_for_turn(manager, args.timeout)
            assistants = [
                message
                for message in state["conversation"]
                if message.get("role") == "assistant"
            ]
            if not assistants:
                raise RuntimeError("agent state contains no assistant response")
            answer = assistants[-1]
            checks = _answer_checks(answer, state)
            report["turns"].append(
                {
                    "prompt": prompt,
                    "answer": answer.get("content", ""),
                    "finish_reason": answer.get("finish_reason"),
                    "continuations": answer.get("continuations"),
                    "generated_tokens": answer.get("generated_tokens"),
                    "checks": checks,
                }
            )
        report["all_checks_passed"] = (
            len(report["turns"]) == len(DEFAULT_PROMPTS)
            and all(
                all(turn["checks"].values())
                for turn in report["turns"]
            )
        )
        report["status"] = "passed" if report["all_checks_passed"] else "failed"
    except BaseException as error:
        report["error"] = f"{type(error).__name__}: {error}"[:512]
    finally:
        try:
            if manager is not None:
                manager.shutdown()
            elif service is not None:
                service.stop()
        except BaseException as cleanup_error:
            report["cleanup_error"] = (
                f"{type(cleanup_error).__name__}: {cleanup_error}"[:512]
            )
            report["status"] = "failed"
            report["all_checks_passed"] = False
        report["worker_cleaned"] = service is None or not service.is_running
    return report, 0 if report["all_checks_passed"] else 1


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report, code = execute(args)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
