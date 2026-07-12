"""Release gate for natural, bounded Korean conversation with the local model.

This suite intentionally complements ``validate_agent_completion.py``.  The
completion stress suite checks formal contracts; this suite reproduces the
short, open-ended conversation that previously fell through to the fixed
quality fallback and then exercises nearby real-world variations.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
import sys
from time import monotonic
from typing import Any, Sequence

import torch

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from cogni_agent.conversation_fastpath import ConversationFastPath  # noqa: E402
from cogni_agent.fact_grounding import RuntimeFactGrounder  # noqa: E402
from cogni_agent.manager import SYSTEM_PROMPT, AgentManager  # noqa: E402
from cogni_agent.model_service import ModelService  # noqa: E402
from cogni_agent.response_quality import (  # noqa: E402
    QualityAction,
    inspect_response,
    requested_exact_item_count,
    requested_maximum_items,
)
from cogni_agent.tools import WorkspaceToolExecutor  # noqa: E402
from cogni_flow.rhythm import RhythmController  # noqa: E402
from cogni_os.artifacts import verify_artifact_manifest  # noqa: E402
from cogni_os.factbook import build_runtime_factbook_from_verified  # noqa: E402
from cogni_os.gpu_lease import GPULeaseManager  # noqa: E402
from cogni_os.version import __version__  # noqa: E402
from scripts.validate_agent_completion import (  # noqa: E402
    _answer_checks,
    _atomic_external_report,
    _has_conservative_near_duplicate,
    _offline_environment,
    _sentence_repetition_metrics,
    _wait_for_turn,
)


MAX_CASUAL_TURN_SECONDS = 120.0
MAX_FAST_PATH_RELEASE_TURNS = 2
_TOKEN_RE = re.compile(r"[0-9A-Za-z가-힣]+")
_CANNED_RECOVERY_FRAGMENTS = (
    "로컬 모델의 답변 후보가 품질 검증을 통과하지 못했습니다",
    "이번에는 추측해서 답하지 않았습니다",
    "표현을 바꿔 다시 요청해 주세요",
)


@dataclass(frozen=True)
class CasualCase:
    """One natural-conversation turn and its observable release contract."""

    label: str
    session: str
    prompt: str
    categories: tuple[str, ...]
    expected_mode: str
    required_groups: tuple[tuple[str, ...], ...]


CASUAL_CASES = (
    CasualCase(
        "reported-collaboration",
        "reported-sequence",
        "그럼, 나와함께 재미있는 프로젝트를 합시다.",
        ("exact_reproduction", "collaboration_proposal"),
        "conversation_fastpath",
        (
            ("함께", "같이", "좋", "재미"),
            ("프로젝트", "만들", "시작", "아이디어", "목표"),
        ),
    ),
    CasualCase(
        "reported-capabilities-followup",
        "reported-sequence",
        "나와 어떤 일을 함께 할 수 있나요?",
        ("exact_reproduction", "follow_up"),
        "conversation_fastpath",
        (("함께", "도와", "할 수"), ("작업", "프로젝트", "코드", "문서", "아이디어")),
    ),
    CasualCase(
        "independent-greeting",
        "independent-casual",
        "안녕하세요! 정해진 인사말 말고, 오늘 함께 이야기해 볼 주제를 하나 자연스럽게 제안해 주세요.",
        ("independent_casual",),
        "cogni_core",
        (("안녕", "반가", "좋"), ("이야기", "주제", "오늘", "제안")),
    ),
    CasualCase(
        "open-project-idea",
        "typo",
        "프로잭트 아이디어가 막혔어요. 생각을 풀 수 있도록 질문 하나를 자연스럽게 해 주세요.",
        ("typo", "collaboration_proposal"),
        "cogni_core",
        (("프로젝트", "아이디어", "주제"), ("무엇", "어떤", "누구", "까요")),
    ),
    CasualCase(
        "capability-paraphrase",
        "paraphrase",
        "제가 도움을 부탁할 수 있는 범위를 예시와 함께 편한 말로 설명해 주세요.",
        ("paraphrase",),
        "cogni_core",
        (
            ("도와", "할 수", "함께", "지원"),
            ("작업", "코드", "문서", "아이디어", "설명", "검토"),
        ),
    ),
    CasualCase(
        "followup-context-seed",
        "context-chain",
        "오프라인 AI 데모에서 개인정보 보호를 가장 먼저 보여주고 싶어요. 어떤 흐름이 자연스러울까요?",
        ("follow_up", "collaboration_proposal"),
        "cogni_core",
        (("데모", "오프라인", "AI"), ("개인정보", "보호", "흐름", "기능")),
    ),
    CasualCase(
        "followup-first-step",
        "context-chain",
        "좋아요. 방금 제안에서 제가 먼저 정해야 할 것 하나만 질문해 주세요.",
        ("follow_up",),
        "cogni_core",
        (("무엇", "어떤", "누구", "까요"), ("데모", "목표", "사용", "기능", "대상")),
    ),
    CasualCase(
        "context-transition",
        "context-chain",
        "그 이야기는 잠시 접어두고, 파이썬 리스트와 튜플의 차이를 한 문장으로 알려 주세요.",
        ("context_transition",),
        "cogni_core",
        (("리스트",), ("튜플",), ("변경", "불변", "수정", "mutable", "immutable")),
    ),
    CasualCase(
        "formal-quality-regression",
        "formal",
        "모델 응답 품질을 배포 전에 검증하는 절차를 세 문장으로 설명해 주세요.",
        ("formal_regression",),
        "cogni_core",
        (("검증", "테스트", "평가"), ("품질", "반복", "완결", "배포", "기준")),
    ),
    CasualCase(
        "formal-factbook-regression",
        "factbook",
        "당신은 정확히 어떤 모델이며 현재 검증된 기능과 제한은 무엇인가요?",
        ("formal_regression",),
        "factbook",
        (("Gemma", "Cogni"), ("검증", "제한", "Fact-book", "상태")),
    ),
)

REQUIRED_CATEGORIES = frozenset(
    {
        "exact_reproduction",
        "independent_casual",
        "collaboration_proposal",
        "paraphrase",
        "typo",
        "follow_up",
        "context_transition",
        "formal_regression",
    }
)

_GENERIC_ONLY_ANSWERS = frozenset(
    {
        "네.",
        "좋아요.",
        "좋은 생각입니다.",
        "알겠습니다.",
        "도와드릴 수 있습니다.",
    }
)
_COMPLETE_SENTENCE_TEXT_RE = re.compile(r".+?[.!?。！？]+(?=\s|$)", re.DOTALL)
_COLLABORATION_NEGATION_RE = re.compile(
    r"(?:하지\s*않|못\s*(?:하|만들|시작)|싫|거절|중단|아닙니다|아니에요)"
)


def _lexical_diversity(text: str) -> float:
    tokens = [token.casefold() for token in _TOKEN_RE.findall(text)]
    if not tokens:
        return 0.0
    return len(set(tokens)) / len(tokens)


def _sentence_keys(text: str, *, minimum_chars: int = 1) -> frozenset[str]:
    keys: set[str] = set()
    for match in _COMPLETE_SENTENCE_TEXT_RE.finditer(text[:2_000]):
        key = re.sub(r"\s+", " ", match.group(0)).strip().casefold()
        key = key.rstrip(" .!?。！？")
        if len(key) >= minimum_chars:
            keys.add(key)
    return frozenset(keys)


def _has_short_sentence_loop(text: str) -> bool:
    observed: set[str] = set()
    for match in _COMPLETE_SENTENCE_TEXT_RE.finditer(text[:2_000]):
        key = re.sub(r"\s+", " ", match.group(0)).strip().casefold()
        key = key.rstrip(" .!?。！？")
        if not key:
            continue
        if key in observed:
            return True
        observed.add(key)
    return False


def _casual_checks(
    case: CasualCase,
    answer: dict[str, Any],
    state: dict[str, Any],
    *,
    new_assistant_count: int,
    elapsed_seconds: float,
    latency_limit_seconds: float,
    prior_sentence_keys: frozenset[str] = frozenset(),
    expected_factbook: dict[str, Any] | None = None,
) -> dict[str, bool]:
    """Return deterministic observables; this is not a subjective LLM judge."""

    text = str(answer.get("content", "")).strip()
    folded = text.casefold()
    prompt_folded = re.sub(r"\s+", " ", case.prompt).strip().casefold()
    structured_items = (
        requested_exact_item_count(case.prompt) is not None
        or requested_maximum_items(case.prompt) is not None
    )
    checks = _answer_checks(
        answer,
        state,
        case.prompt,
        required_groups=case.required_groups,
        expected_factbook=expected_factbook,
    )
    sentence_keys = _sentence_keys(text, minimum_chars=16)
    checks.update(
        {
            "exactly_one_assistant": new_assistant_count == 1,
            "expected_generation_mode": answer.get("generation_mode")
            == case.expected_mode,
            "zero_quality_fallback": answer.get("generation_mode")
            != "quality_fallback",
            "no_canned_recovery_copy": not any(
                fragment in text for fragment in _CANNED_RECOVERY_FRAGMENTS
            ),
            "not_prompt_echo": bool(text)
            and prompt_folded not in re.sub(r"\s+", " ", text).strip().casefold(),
            "bounded_response_size": 8 <= len(text) <= 2_000,
            "lexically_non_degenerate": _lexical_diversity(text) >= 0.20,
            "quality_report_accepts": inspect_response(
                text,
                final=True,
            ).recommended_action
            is QualityAction.ACCEPT,
            "no_short_sentence_loop": not _has_short_sentence_loop(text),
            "no_near_duplicate_sentence": not _has_conservative_near_duplicate(
                text,
                structured_items=structured_items,
            ),
            "not_generic_only": text not in _GENERIC_ONLY_ANSWERS,
            "case_relevance": all(
                any(term.casefold() in folded for term in group)
                for group in case.required_groups
            ),
            "intent_alignment": "collaboration_proposal" not in case.categories
            or _COLLABORATION_NEGATION_RE.search(text) is None,
            "no_cross_turn_sentence_reuse": not bool(
                sentence_keys & prior_sentence_keys
            ),
            "bounded_latency": 0.0
            <= float(elapsed_seconds)
            <= float(latency_limit_seconds),
        }
    )
    return checks


def _casual_summary(
    turns: list[dict[str, Any]],
    *,
    latency_limit_seconds: float,
) -> dict[str, Any]:
    categories_seen = {
        category for turn in turns for category in turn.get("categories", [])
    }
    fallback_turns = sum(
        turn.get("generation_mode") == "quality_fallback" for turn in turns
    )
    fastpath_turns = sum(
        turn.get("generation_mode") == "conversation_fastpath" for turn in turns
    )
    failed_checks: dict[str, int] = {}
    for turn in turns:
        for name, passed in dict(turn.get("checks", {})).items():
            if passed is not True:
                failed_checks[name] = failed_checks.get(name, 0) + 1
    exact_turns = [
        turn for turn in turns if "exact_reproduction" in turn.get("categories", [])
    ]
    expected_count = len(CASUAL_CASES)
    passed_turns = sum(turn.get("passed") is True for turn in turns)
    strict = (
        len(turns) == expected_count
        and passed_turns == expected_count
        and fallback_turns == 0
        and len(exact_turns) == 2
        and all(turn.get("passed") is True for turn in exact_turns)
        and fastpath_turns <= MAX_FAST_PATH_RELEASE_TURNS
        and REQUIRED_CATEGORIES.issubset(categories_seen)
    )
    return {
        "expected_turns": expected_count,
        "completed_turns": len(turns),
        "passed_turns": passed_turns,
        "failed_turns": expected_count - passed_turns,
        "quality_fallback_turns": fallback_turns,
        "quality_fallback_gate_passed": fallback_turns == 0,
        "fastpath_turns": fastpath_turns,
        "fastpath_ratio": 0.0 if not turns else fastpath_turns / len(turns),
        "fastpath_ratio_gate_passed": len(turns) == expected_count
        and fastpath_turns <= MAX_FAST_PATH_RELEASE_TURNS,
        "single_assistant_gate_passed": all(
            turn.get("checks", {}).get("exactly_one_assistant") is True
            for turn in turns
        )
        and len(turns) == expected_count,
        "natural_completion_gate_passed": all(
            turn.get("checks", {}).get("korean_complete") is True
            and turn.get("checks", {}).get("lexically_non_degenerate") is True
            and turn.get("checks", {}).get("no_repeated_sentence") is True
            for turn in turns
        )
        and len(turns) == expected_count,
        "bounded_latency_gate_passed": all(
            float(turn.get("elapsed_seconds", latency_limit_seconds + 1))
            <= latency_limit_seconds
            for turn in turns
        )
        and len(turns) == expected_count,
        "latency_limit_seconds": float(latency_limit_seconds),
        "peak_turn_latency_seconds": max(
            (float(turn.get("elapsed_seconds", 0.0)) for turn in turns),
            default=None,
        ),
        "exact_reproduction_gate_passed": len(exact_turns) == 2
        and all(turn.get("passed") is True for turn in exact_turns),
        "required_categories": sorted(REQUIRED_CATEGORIES),
        "observed_categories": sorted(categories_seen),
        "category_coverage_gate_passed": REQUIRED_CATEGORIES.issubset(categories_seen),
        "failed_check_counts": failed_checks,
        "strict_casual_gate_passed": strict,
    }


def _bounded_timeout(value: str) -> float:
    try:
        timeout = float(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("timeout must be a number") from error
    if not 1.0 <= timeout <= MAX_CASUAL_TURN_SECONDS:
        raise argparse.ArgumentTypeError(
            f"timeout must be in [1, {MAX_CASUAL_TURN_SECONDS:.0f}] seconds"
        )
    return timeout


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the exact reported Korean conversation and nearby natural-language "
            "regressions against the offline local Gemma worker."
        )
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument(
        "--timeout",
        type=_bounded_timeout,
        default=MAX_CASUAL_TURN_SECONDS,
        help="per-turn hard deadline and release latency ceiling in seconds",
    )
    parser.add_argument(
        "--output",
        help="optional atomic JSON evidence path outside the source tree",
    )
    return parser


def execute(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    _offline_environment()
    timeout = float(args.timeout)
    if not 1.0 <= timeout <= MAX_CASUAL_TURN_SECONDS:
        raise ValueError(
            f"timeout must be in [1, {MAX_CASUAL_TURN_SECONDS:.0f}] seconds"
        )
    report: dict[str, Any] = {
        "schema": "cogni.agent.casual-korean.v1",
        "status": "running",
        "build_version": __version__,
        "criteria": {
            "exact_reproduction": [case.prompt for case in CASUAL_CASES[:2]],
            "exact_sequence_routes": [
                "conversation_fastpath",
                "conversation_fastpath",
            ],
            "evidence_boundary": (
                "the exact reported social turns exercise the same bounded "
                "ConversationFastPath that the product injects before Fact-book/model routing"
            ),
            "quality_fallback": "zero permitted",
            "fastpath": (
                f"at most {MAX_FAST_PATH_RELEASE_TURNS} of {len(CASUAL_CASES)} turns"
            ),
            "ownership": "exactly one newly owned assistant message per turn",
            "natural_completion": (
                "complete Korean boundary, no sentence loop, no role/control leak, "
                "non-degenerate lexical diversity"
            ),
            "latency": f"every turn completes within {timeout:.3f} seconds",
            "coverage": sorted(REQUIRED_CATEGORIES),
        },
        "turns": [],
        "all_checks_passed": False,
    }
    leases = GPULeaseManager()
    rhythm = RhythmController()
    service: ModelService | None = None
    managers: dict[str, AgentManager] = {}
    prior_answer_digests: dict[str, set[str]] = {}
    prior_sentence_keys: dict[str, set[str]] = {}
    expected_factbook: dict[str, Any] | None = None
    try:
        verified = verify_artifact_manifest(args.model, args.manifest)
        report["verified_files"] = len(verified.files)
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for the casual Korean release gate")
        report["cuda_device"] = torch.cuda.get_device_name(0)
        factbook = build_runtime_factbook_from_verified(
            verified,
            args.manifest,
            build_version=__version__,
            device=report["cuda_device"],
        )
        report["factbook"] = factbook.as_payload()
        expected_factbook = {
            "build_version": factbook.build_version,
            "model_label": factbook.model.label,
            "stored_parameters": factbook.model.stored_parameters,
            "effective_parameters": factbook.model.effective_parameters,
        }
        service = ModelService.for_local_gemma(
            args.model,
            manifest_path=args.manifest,
            artifact_digest=factbook.model.manifest_sha256,
            vram_limit_gib=16.7,
            max_input_tokens=4_096,
            max_new_tokens=512,
            max_prompt_chars=32_000,
            max_response_chars=32_000,
            request_timeout=timeout,
            gpu_lease_manager=leases,
            gpu_lease_owner="casual-korean-validator-model",
            gpu_lease_purpose="inference",
            gpu_lease_vram_bytes=leases.max_vram_bytes,
        )
        # The public latency ceiling includes first-use model loading. Reserve
        # a bounded startup/cleanup margin so the manager's generation fence
        # cannot expire at the exact same instant as the outer release gate.
        manager_decode_seconds = max(
            0.01,
            timeout - min(30.0, timeout * 0.25),
        )
        for case in CASUAL_CASES:
            if case.session not in managers:
                managers[case.session] = AgentManager(
                    service,
                    WorkspaceToolExecutor(_PROJECT_ROOT),
                    session_id=f"casual-{case.session}",
                    conversation_fast_path=ConversationFastPath(),
                    fact_grounder=RuntimeFactGrounder(factbook),
                    # Match the product path: deterministic Fact-book answers are
                    # routed outside the generative prompt and must not inflate
                    # every casual turn or remain in model-visible history.
                    system_prompt=SYSTEM_PROMPT,
                    rhythm=rhythm,
                    max_decode_seconds=manager_decode_seconds,
                )
            manager = managers[case.session]
            session_answer_digests = prior_answer_digests.setdefault(
                case.session,
                set(),
            )
            session_sentence_keys = prior_sentence_keys.setdefault(
                case.session,
                set(),
            )
            before = manager.snapshot()
            known_assistant_ids = {
                str(message.get("id"))
                for message in before.get("conversation", [])
                if message.get("role") == "assistant"
            }
            started = monotonic()
            error: str | None = None
            try:
                manager.start_turn(case.prompt, "chat")
                state = _wait_for_turn(manager, timeout)
            except BaseException as caught:
                error = f"{type(caught).__name__}: {caught}"[:512]
                try:
                    manager.cancel()
                    state = _wait_for_turn(
                        manager,
                        min(30.0, max(2.0, timeout * 0.25)),
                    )
                except BaseException:
                    state = manager.snapshot()
            elapsed = monotonic() - started
            assistants = [
                message
                for message in state.get("conversation", [])
                if message.get("role") == "assistant"
                and str(message.get("id")) not in known_assistant_ids
            ]
            answer = assistants[-1] if assistants else {}
            checks = _casual_checks(
                case,
                answer,
                state,
                new_assistant_count=len(assistants),
                elapsed_seconds=elapsed,
                latency_limit_seconds=timeout,
                prior_sentence_keys=frozenset(session_sentence_keys),
                expected_factbook=expected_factbook,
            )
            text = str(answer.get("content", "")).strip()
            answer_digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
            checks["no_cross_turn_exact_duplicate"] = bool(text) and (
                answer_digest not in session_answer_digests
            )
            record: dict[str, Any] = {
                "turn": len(report["turns"]) + 1,
                "case": case.label,
                "session_id": manager.session_id,
                "categories": list(case.categories),
                "prompt": case.prompt,
                "answer": text,
                "answer_sha256": answer_digest,
                "generation_mode": answer.get("generation_mode"),
                "finish_reason": answer.get("finish_reason"),
                "generated_tokens": int(answer.get("generated_tokens", 0) or 0),
                "continuations": int(answer.get("continuations", 0) or 0),
                "new_assistant_count": len(assistants),
                "elapsed_seconds": round(elapsed, 6),
                "lexical_diversity": round(_lexical_diversity(text), 6),
                "repetition": _sentence_repetition_metrics(text),
                "checks": checks,
                "passed": all(checks.values()) and error is None,
            }
            if error is not None:
                record["error"] = error
            report["turns"].append(record)
            if text:
                session_answer_digests.add(answer_digest)
                session_sentence_keys.update(_sentence_keys(text, minimum_chars=16))
            report["summary"] = _casual_summary(
                report["turns"], latency_limit_seconds=timeout
            )
            if args.output:
                _atomic_external_report(args.output, report)
            print(
                f"casual turn {record['turn']}/{len(CASUAL_CASES)} "
                f"status={'passed' if record['passed'] else 'failed'} "
                f"elapsed={elapsed:.3f}s",
                file=sys.stderr,
                flush=True,
            )
    except BaseException as error:
        report["error"] = f"{type(error).__name__}: {error}"[:512]
    finally:
        for manager in managers.values():
            try:
                manager.shutdown()
            except BaseException as cleanup_error:
                report.setdefault("cleanup_errors", []).append(
                    f"{type(cleanup_error).__name__}: {cleanup_error}"[:512]
                )
        if not managers and service is not None:
            try:
                service.stop()
            except BaseException as cleanup_error:
                report.setdefault("cleanup_errors", []).append(
                    f"{type(cleanup_error).__name__}: {cleanup_error}"[:512]
                )
        report["summary"] = _casual_summary(
            report["turns"], latency_limit_seconds=timeout
        )
        report["worker_cleaned"] = service is None or not service.is_running
        report["gpu_lease_released"] = leases.active is None
        report["all_checks_passed"] = bool(
            report["summary"]["strict_casual_gate_passed"]
            and report["worker_cleaned"]
            and report["gpu_lease_released"]
            and "error" not in report
            and not report.get("cleanup_errors")
        )
        report["status"] = "passed" if report["all_checks_passed"] else "failed"
    return report, 0 if report["all_checks_passed"] else 1


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report, code = execute(args)
    if args.output:
        target = _atomic_external_report(args.output, report)
        payload = {
            "status": report["status"],
            "all_checks_passed": report["all_checks_passed"],
            "output": str(target),
            "summary": report["summary"],
        }
    else:
        payload = report
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
