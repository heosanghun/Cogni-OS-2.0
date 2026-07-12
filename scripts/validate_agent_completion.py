"""Offline, real-model regression for complete multi-turn Cogni-Agent answers."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from difflib import SequenceMatcher
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from time import monotonic, sleep
from typing import Any, Sequence
import unicodedata

import torch

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from cogni_agent.fact_grounding import RuntimeFactGrounder  # noqa: E402
from cogni_agent.manager import (  # noqa: E402
    ACTIVE_AGENT_STATUSES,
    SYSTEM_PROMPT,
    AgentManager,
)
from cogni_agent.model_service import ModelService  # noqa: E402
from cogni_agent.response_quality import (  # noqa: E402
    QualityAction,
    has_near_duplicate_sentences,
    inspect_response,
    requested_exact_item_count,
    requested_maximum_items,
    response_contract_satisfied,
)
from cogni_agent.tools import WorkspaceToolExecutor  # noqa: E402
from cogni_flow.rhythm import RhythmController  # noqa: E402
from cogni_os.artifacts import verify_artifact_manifest  # noqa: E402
from cogni_os.factbook import build_runtime_factbook_from_verified  # noqa: E402
from cogni_os.gpu_lease import GPULeaseManager  # noqa: E402
from cogni_os.version import __version__  # noqa: E402


DEFAULT_PROMPTS = (
    "당신은 정확히 어떤 모델이며 저장 파라미터와 effective 파라미터는 각각 몇 개인가요? "
    "검증된 Runtime Fact-book 수치만 한 번씩 답하세요.",
    "자가 거울치료가 무엇인가요? 핵심만 세 문장 이내로 설명하고 같은 문장이나 "
    "문단을 반복하지 마세요.",
    "Cogni-OS의 CTS, System 1.5, System 2.5, System 3, System 4를 각각 한 문장씩 "
    "설명하세요. 검증된 기능과 설계 목표를 구분하고 마지막은 반드시 '이상입니다.'로 끝내세요.",
    "방금 답변에서 실제 검증과 향후 목표를 구분하는 원칙만 두 문장으로 요약하고 마침표로 끝내세요.",
)
GROUNDED_TURNS = 4
GROUNDED_STRESS_INDICES = frozenset({9})
DEFAULT_TURNS = 20
RECOMMENDED_STRESS_TURNS = 20
MAX_STRESS_TURNS = 100
MAX_INTERACTIVE_TURN_SECONDS = 120.0
STRESS_PROMPTS = (
    "온디바이스 AI의 장점 두 가지와 한계 한 가지를 세 문장으로 설명하세요. 같은 문장을 반복하지 마세요.",
    "사용자가 잘못된 사실을 정정했을 때 대화형 AI가 취해야 할 절차를 세 단계로 간결하게 답하세요.",
    "확인된 사실과 추론을 섞지 않기 위한 답변 원칙을 두 문장으로 설명하고 마침표로 끝내세요.",
    "긴 답변이 중간에 끊기지 않도록 생성 시스템이 확인해야 할 항목을 네 가지 이내로 정리하세요.",
    "로컬 파일을 수정하기 전에 백업, 검증, 롤백을 어떻게 준비해야 하는지 세 문장으로 설명하세요.",
    "예외가 발생한 작업을 무한 재시도하지 않고 안전하게 종료하는 기준을 두 문장으로 답하세요.",
    "반복 없는 좋은 요약문의 조건을 세 가지 제시하고 마지막 문장은 자연스럽게 끝내세요.",
    "개인정보가 포함된 요청을 오프라인 환경에서 처리할 때 지켜야 할 원칙을 세 문장으로 답하세요.",
    "제한된 GPU 메모리에서 추론할 때 측정값과 설계 목표를 구분해야 하는 이유를 설명하세요.",
    "도구 실행 결과를 확인하지 못했을 때 AI가 성공했다고 말하면 안 되는 이유를 두 문장으로 답하세요.",
    "오류 복구 과정에서 원인, 수정, 회귀 테스트를 어떤 순서로 기록해야 하는지 설명하세요.",
    "긴 대화에서 오래된 문맥을 줄이면서 사용자 의도를 보존하는 방법을 세 문장으로 답하세요.",
    "불확실한 답변을 사실처럼 단정하지 않기 위한 표현 원칙을 두 문장으로 설명하세요.",
    "사용자 권한과 시스템 안전 경계를 함께 지키는 작업 실행 원칙을 세 문장으로 답하세요.",
    "소프트웨어 수정 완료를 선언하기 전에 필요한 자체 검증을 네 항목 이내로 정리하세요.",
    "자연스러운 한국어 답변의 완결성을 판정할 때 확인할 사항을 세 문장으로 설명하세요.",
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
_SENTENCE_RE = re.compile(r".+?(?:[.!?。！？]+(?=\s|$)|$)", re.DOTALL)
_HANGUL_RE = re.compile(r"[가-힣]")
_DANGLING_KOREAN_RE = re.compile(
    r"(?:이는\s+내가|그\s+이유는|예를\s+들어|따라서|그리고|하지만|반면에|"
    r"하기\s+위해|수\s+있으며|것은|경우에는)\s*[.!?。！？]*\s*$"
)
_TRAILING_LIST_MARKER_RE = re.compile(r"(?:^|\s)(?P<marker>[-*+]|\d{1,4}[.)])\s*$")
_NUMBERED_ITEM_WITH_CONTENT_RE = re.compile(r"(?:^|\s)1[.)]\s+\S")
_TRUNCATED_FINISH_REASONS = {"length", "max_tokens", "truncated"}
_REQUIRED_LITERAL_END_RE = re.compile(
    r"(?:반드시\s*)?['\"“”‘’](?P<literal>[^'\"“”‘’\r\n]{1,80})['\"“”‘’]"
    r"(?:로|으로)\s*(?:끝|마무리|종료)"
)
_PERIOD_END_REQUEST_RE = re.compile(r"마침표(?:로|를 사용해)?\s*(?:끝|마무리)")


DEFAULT_ANCHOR_GROUPS = (
    (("gemma", "cogni"), ("파라미터", "effective", "저장")),
    (("자가 거울치료", "self-harness"), ("패치", "검증", "제안", "코드")),
    (
        ("cts",),
        ("system 1.5",),
        ("system 2.5",),
        ("system 3",),
        ("system 4",),
    ),
    (("검증", "실측", "사실"), ("목표", "향후", "설계")),
)
STRESS_ANCHOR_GROUPS = (
    (
        ("온디바이스", "기기", "로컬"),
        ("장점", "보안", "응답", "오프라인"),
        ("한계", "제약", "메모리", "성능", "전력"),
    ),
    (("정정", "사실", "수정"), ("확인", "검증", "반영")),
    (("사실", "확인"), ("추론", "판단")),
    (("문장", "답변", "생성"), ("끝", "완결", "길이", "토큰", "중단")),
    (("백업",), ("검증", "테스트"), ("롤백", "복구")),
    (("재시도", "예외", "오류"), ("종료", "중단", "한도", "횟수")),
    (("요약", "핵심 내용"), ("반복", "핵심", "간결")),
    (("개인정보", "데이터"), ("오프라인", "로컬", "장치")),
    (("측정값", "측정"), ("설계 목표", "목표"), ("메모리", "gpu")),
    (("실행", "도구", "결과"), ("확인", "검증", "성공")),
    (("원인",), ("수정",), ("회귀", "테스트")),
    (("문맥", "대화"), ("의도", "사용자"), ("오래된", "요약", "줄")),
    (("불확실", "추측"), ("사실", "단정", "근거")),
    (("권한",), ("안전", "경계"), ("작업", "실행")),
    (
        ("검증", "테스트", "확인"),
        ("수정", "기능"),
        ("보안", "성능", "회귀", "작동"),
    ),
    (("한국어", "답변"), ("완결", "문장", "종결"), ("반복", "자연", "문법")),
)


@dataclass(frozen=True)
class PromptCase:
    """One deterministic stress turn and its expected answer route."""

    label: str
    prompt: str
    expected_route: str
    required_groups: tuple[tuple[str, ...], ...] = ()


def _bounded_turn_count(value: str) -> int:
    try:
        count = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("turns must be an integer") from error
    if not 1 <= count <= MAX_STRESS_TURNS:
        raise argparse.ArgumentTypeError(f"turns must be in [1, {MAX_STRESS_TURNS}]")
    return count


def _bounded_timeout(value: str) -> float:
    try:
        timeout = float(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("timeout must be a number") from error
    if not 1.0 <= timeout <= MAX_INTERACTIVE_TURN_SECONDS:
        raise argparse.ArgumentTypeError(
            f"timeout must be in [1, {MAX_INTERACTIVE_TURN_SECONDS:.0f}] seconds"
        )
    return timeout


def _prompt_cases(turns: int) -> tuple[PromptCase, ...]:
    if not 1 <= int(turns) <= MAX_STRESS_TURNS:
        raise ValueError(f"turns must be in [1, {MAX_STRESS_TURNS}]")
    base = tuple(
        PromptCase(
            label=f"baseline-{index + 1:02d}",
            prompt=prompt,
            expected_route="grounded" if index < GROUNDED_TURNS else "generated",
            required_groups=DEFAULT_ANCHOR_GROUPS[index],
        )
        for index, prompt in enumerate(DEFAULT_PROMPTS)
    )
    cases = list(base[:turns])
    while len(cases) < turns:
        stress_index = len(cases) - len(base)
        prompt_index = stress_index % len(STRESS_PROMPTS)
        cycle = stress_index // len(STRESS_PROMPTS)
        prompt = STRESS_PROMPTS[prompt_index]
        if cycle:
            prompt = f"{prompt} 검증 반복 {cycle + 1}에서는 앞선 표현을 그대로 복사하지 마세요."
        cases.append(
            PromptCase(
                label=f"stress-{len(cases) + 1:02d}",
                prompt=prompt,
                expected_route=(
                    "grounded"
                    if prompt_index in GROUNDED_STRESS_INDICES
                    else "generated"
                ),
                required_groups=STRESS_ANCHOR_GROUPS[prompt_index],
            )
        )
    return tuple(cases)


def _has_repeated_sentence(text: str) -> bool:
    seen: set[str] = set()
    for match in _SENTENCE_RE.finditer(text):
        sentence = re.sub(r"\s+", " ", match.group(0)).strip().casefold()
        sentence = sentence.rstrip(" .!?。！？")
        if len(sentence) < 12:
            continue
        if sentence in seen:
            return True
        seen.add(sentence)
    return False


def _complete_sentence_keys(text: str) -> tuple[str, ...]:
    keys: list[str] = []
    for match in _SENTENCE_RE.finditer(text[:32_000]):
        key = re.sub(r"\s+", " ", match.group(0)).strip().casefold()
        key = re.sub(r"^(?:[-*+]\s+|\d{1,4}[.)]\s+)", "", key)
        key = key.rstrip(" .!?。！？")
        if key:
            keys.append(key)
    return tuple(keys)


def _has_short_sentence_loop(text: str) -> bool:
    observed: set[str] = set()
    for key in _complete_sentence_keys(text):
        if key in observed:
            return True
        observed.add(key)
    return False


def _has_repeated_paragraph(text: str) -> bool:
    observed: set[str] = set()
    for block in re.split(r"(?:\r?\n\s*)+", text[:32_000]):
        key = re.sub(r"\s+", " ", block).strip().casefold()
        if len(key) < 8:
            continue
        if key in observed:
            return True
        observed.add(key)
    return False


def _has_conservative_near_duplicate(
    text: str,
    *,
    structured_items: bool,
) -> bool:
    """Detect near copies without rejecting ordinary parallel list scaffolds."""

    if not structured_items:
        return has_near_duplicate_sentences(text)
    sentences = [key[:256] for key in _complete_sentence_keys(text) if len(key) >= 20]
    for index, first in enumerate(sentences):
        for second in sentences[index + 1 :]:
            if first == second:
                return True
            if SequenceMatcher(None, first, second, autojunk=False).ratio() >= 0.985:
                return True
    return False


def _topic_anchors_satisfied(
    text: str,
    groups: tuple[tuple[str, ...], ...],
) -> bool:
    if not groups:
        return True
    folded = unicodedata.normalize("NFKC", text).casefold()
    return all(any(term.casefold() in folded for term in group) for group in groups)


def _required_literal_ending(request: str) -> str | None:
    match = _REQUIRED_LITERAL_END_RE.search(request[:2_000])
    return None if match is None else match.group("literal")


def _literal_and_period_contract(request: str, response: str) -> dict[str, bool]:
    literal = _required_literal_ending(request)
    period_required = _PERIOD_END_REQUEST_RE.search(request[:2_000]) is not None
    stripped = response.rstrip()
    return {
        "required_literal_ending": literal is None or stripped.endswith(literal),
        "required_period_ending": not period_required or stripped.endswith("."),
    }


def _has_empty_trailing_list_item(text: str) -> bool:
    stripped = text.rstrip()
    match = _TRAILING_LIST_MARKER_RE.search(stripped)
    if match is None:
        return False
    marker = match.group("marker")
    if (
        stripped == marker
        or marker in {"-", "*", "+"}
        or "\n" in match.group(0)
        or "\r" in match.group(0)
    ):
        return True
    return _NUMBERED_ITEM_WITH_CONTENT_RE.search(stripped[: match.start()]) is not None


def _factbook_identity_checks(
    request: str,
    response: str,
    expected: dict[str, Any] | None,
) -> dict[str, bool]:
    identity_request = "어떤 모델" in request or "파라미터" in request
    if not identity_request:
        return {
            "factbook_model_exact": True,
            "factbook_version_exact": True,
            "factbook_parameters_exact": True,
        }
    if not expected:
        return {
            "factbook_model_exact": False,
            "factbook_version_exact": False,
            "factbook_parameters_exact": False,
        }
    folded = unicodedata.normalize("NFKC", response).casefold()
    digits = re.sub(r"(?<=\d),(?=\d)", "", folded)
    return {
        "factbook_model_exact": str(expected["model_label"]).casefold() in folded,
        "factbook_version_exact": str(expected["build_version"]).casefold() in folded,
        "factbook_parameters_exact": str(expected["stored_parameters"]) in digits
        and str(expected["effective_parameters"]) in digits,
    }


def _expected_factbook_identity(factbook: Any) -> dict[str, Any]:
    """Extract the release identity from the typed Runtime Fact-book."""

    inventory = factbook.model.inventory
    return {
        "build_version": factbook.build_version,
        "model_label": factbook.model.label,
        "stored_parameters": inventory.stored_parameters,
        "effective_parameters": inventory.effective_parameters,
    }


def _sentence_repetition_metrics(text: str) -> dict[str, Any]:
    normalized: list[str] = []
    for match in _SENTENCE_RE.finditer(text):
        sentence = re.sub(r"\s+", " ", match.group(0)).strip().casefold()
        sentence = sentence.rstrip(" .!?。！？")
        if sentence:
            normalized.append(sentence)
    counts: dict[str, int] = {}
    for sentence in normalized:
        counts[sentence] = counts.get(sentence, 0) + 1
    duplicate_count = sum(max(0, count - 1) for count in counts.values())
    total = len(normalized)
    repeated = [sentence for sentence, count in counts.items() if count > 1]
    return {
        "sentence_count": total,
        "unique_sentence_count": len(counts),
        "duplicate_sentence_count": duplicate_count,
        "duplicate_sentence_rate": 0.0 if total == 0 else duplicate_count / total,
        "repeated_sentences": repeated[:8],
    }


def _substantive_sentence_keys(text: str) -> frozenset[str]:
    keys: set[str] = set()
    for match in _SENTENCE_RE.finditer(text):
        key = re.sub(r"\s+", " ", match.group(0)).strip()
        key = re.sub(r"^(?:[-*+]\s+|\d{1,4}[.)]\s+)", "", key)
        key = key.rstrip(" .!?。！？").casefold()
        if len(key) >= 24:
            keys.add(key)
    return frozenset(keys)


def _korean_completion_metrics(text: str) -> dict[str, Any]:
    stripped = text.strip()
    contains_korean = _HANGUL_RE.search(stripped) is not None
    reasons: list[str] = []
    if contains_korean and not stripped.endswith(_COMPLETE_ENDINGS):
        reasons.append("missing_terminal_punctuation")
    if contains_korean and _DANGLING_KOREAN_RE.search(stripped) is not None:
        reasons.append("dangling_korean_clause")
    if contains_korean and _has_empty_trailing_list_item(stripped):
        reasons.append("empty_trailing_list_item")
    return {
        "contains_korean": contains_korean,
        "complete": contains_korean and not reasons,
        "reasons": reasons,
    }


def _role_token_leaks(text: str) -> list[str]:
    return [match.group(0).strip() for match in _ROLE_LEAK.finditer(text)][:16]


def _control_marker_leaks(text: str) -> list[str]:
    lowered = text.casefold()
    return [marker for marker in _CONTROL_MARKERS if marker.casefold() in lowered]


def _explicitly_truncated(answer: dict[str, Any], state: dict[str, Any]) -> bool:
    finish_reason = str(answer.get("finish_reason", "")).casefold()
    completion = state.get("completion")
    completion_truncated = bool(
        isinstance(completion, dict) and completion.get("truncated") is True
    )
    return (
        answer.get("truncated") is True
        or finish_reason in _TRUNCATED_FINISH_REASONS
        or completion_truncated
    )


def _snapshot_conversation_digest(state: dict[str, Any]) -> str:
    conversation = state.get("conversation")
    payload = conversation if isinstance(conversation, list) else []
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _read_process_rss_bytes(pid: int) -> int:
    if not isinstance(pid, int) or isinstance(pid, bool) or pid <= 0:
        raise ValueError("pid must be a positive integer")
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes

        class ProcessMemoryCountersEx(ctypes.Structure):
            _fields_ = [
                ("cb", wintypes.DWORD),
                ("PageFaultCount", wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
                ("PrivateUsage", ctypes.c_size_t),
            ]

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        psapi = ctypes.WinDLL("psapi", use_last_error=True)
        kernel32.OpenProcess.argtypes = [
            wintypes.DWORD,
            wintypes.BOOL,
            wintypes.DWORD,
        ]
        kernel32.OpenProcess.restype = wintypes.HANDLE
        kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
        kernel32.CloseHandle.restype = wintypes.BOOL
        psapi.GetProcessMemoryInfo.argtypes = [
            wintypes.HANDLE,
            ctypes.POINTER(ProcessMemoryCountersEx),
            wintypes.DWORD,
        ]
        psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
        handle = kernel32.OpenProcess(0x1000 | 0x0010, False, pid)
        if not handle:
            raise OSError(ctypes.get_last_error(), "OpenProcess failed")
        try:
            counters = ProcessMemoryCountersEx()
            counters.cb = ctypes.sizeof(counters)
            if not psapi.GetProcessMemoryInfo(
                handle, ctypes.byref(counters), counters.cb
            ):
                raise OSError(ctypes.get_last_error(), "GetProcessMemoryInfo failed")
            return int(counters.WorkingSetSize)
        finally:
            kernel32.CloseHandle(handle)
    status_path = Path(f"/proc/{pid}/status")
    if status_path.is_file():
        for line in status_path.read_text(encoding="utf-8").splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) * 1024
        raise RuntimeError("VmRSS is absent from process status")
    completed = subprocess.run(
        ["ps", "-o", "rss=", "-p", str(pid)],
        capture_output=True,
        text=True,
        timeout=3,
        check=False,
    )
    if completed.returncode != 0 or not completed.stdout.strip():
        raise RuntimeError("process RSS is unavailable")
    return int(completed.stdout.strip().split()[0]) * 1024


def _query_nvidia_smi_gpu_memory_bytes(pid: int) -> tuple[int | None, str]:
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,used_gpu_memory",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        return None, f"unavailable:{type(error).__name__}"
    if completed.returncode != 0:
        return None, f"command_failed:{completed.returncode}"
    total_mib = 0
    matched = False
    unreported = False
    for line in completed.stdout.splitlines():
        fields = [field.strip() for field in line.split(",", 1)]
        if len(fields) != 2 or fields[0] != str(pid):
            continue
        matched = True
        value = fields[1]
        if value.casefold() in {"n/a", "[n/a]", "not supported"}:
            unreported = True
            continue
        number = re.search(r"\d+(?:\.\d+)?", value)
        if number is None:
            unreported = True
            continue
        total_mib += int(float(number.group(0)))
    if total_mib:
        return total_mib * 1024**2, "measured"
    if matched and unreported:
        return None, "driver_unreported"
    return None, "process_not_listed"


def _sample_worker_memory(
    pid: int | None,
    *,
    vram_limit_bytes: int,
    rss_reader: Any = _read_process_rss_bytes,
    gpu_reader: Any = _query_nvidia_smi_gpu_memory_bytes,
) -> dict[str, Any]:
    if pid is None:
        return {
            "worker_rss_bytes": None,
            "worker_rss_status": "worker_not_started",
            "worker_gpu_memory_bytes": None,
            "worker_gpu_memory_status": "worker_not_started",
            "gpu_memory_within_limit": None,
            "vram_limit_bytes": int(vram_limit_bytes),
            "memory_observed": False,
        }
    try:
        rss_bytes = int(rss_reader(pid))
        rss_status = "measured"
    except BaseException as error:
        rss_bytes = None
        rss_status = f"unavailable:{type(error).__name__}"
    try:
        gpu_bytes, gpu_status = gpu_reader(pid)
    except BaseException as error:
        gpu_bytes = None
        gpu_status = f"unavailable:{type(error).__name__}"
    gpu_within = None if gpu_bytes is None else int(gpu_bytes) <= int(vram_limit_bytes)
    return {
        "worker_rss_bytes": rss_bytes,
        "worker_rss_status": rss_status,
        "worker_gpu_memory_bytes": gpu_bytes,
        "worker_gpu_memory_status": str(gpu_status),
        "gpu_memory_within_limit": gpu_within,
        "vram_limit_bytes": int(vram_limit_bytes),
        "memory_observed": rss_bytes is not None or gpu_bytes is not None,
    }


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


def _answer_checks(
    answer: dict[str, Any],
    state: dict[str, Any],
    request: str = "",
    *,
    required_groups: tuple[tuple[str, ...], ...] = (),
    expected_factbook: dict[str, Any] | None = None,
) -> dict[str, bool]:
    text = str(answer.get("content", "")).strip()
    korean = _korean_completion_metrics(text)
    parallel_item_shape = (
        requested_exact_item_count(request) is not None
        or requested_maximum_items(request) is not None
    )
    quality = inspect_response(text, final=True)
    checks = {
        "succeeded": state.get("status") == "succeeded",
        "complete_stage": state.get("stage") == "complete",
        "finish_stop": answer.get("finish_reason") == "stop",
        "not_truncated": answer.get("truncated") is False,
        "not_explicitly_truncated": not _explicitly_truncated(answer, state),
        "non_empty": bool(text),
        # Short noun/name answers can be complete without punctuation. Longer
        # prose must expose an actual sentence boundary as well as model stop.
        "natural_boundary": len(text) < 80 or text.endswith(_COMPLETE_ENDINGS),
        "no_role_leak": _ROLE_LEAK.search(text) is None,
        "no_control_marker": not _control_marker_leaks(text),
        "no_repeated_sentence": not _has_repeated_sentence(text),
        "no_short_sentence_loop": not _has_short_sentence_loop(text),
        "no_repeated_paragraph": not _has_repeated_paragraph(text),
        "no_near_duplicate_sentence": not _has_conservative_near_duplicate(
            text,
            structured_items=parallel_item_shape,
        ),
        "quality_report_accepts": quality.recommended_action is QualityAction.ACCEPT,
        "contains_korean": bool(korean["contains_korean"]),
        "korean_complete": bool(korean["complete"]),
        "topic_anchors_satisfied": _topic_anchors_satisfied(text, required_groups),
        "request_contract_fulfilled": answer.get("generation_mode")
        != "quality_fallback"
        and response_contract_satisfied(request, text),
        "no_false_7b_identity": "70억" not in text
        and "7 billion" not in text.casefold(),
    }
    checks.update(_literal_and_period_contract(request, text))
    checks.update(_factbook_identity_checks(request, text, expected_factbook))
    return checks


def _worker_snapshot(
    service: ModelService,
    *,
    expected_running: bool,
    stable_pid: int | None,
    vram_limit_bytes: int,
    memory_sampler: Any = _sample_worker_memory,
) -> dict[str, Any]:
    running = bool(service.is_running)
    pid = service.worker_pid
    active_request_id = service.active_request_id
    pid_stable = stable_pid is None or pid == stable_pid
    memory = memory_sampler(pid, vram_limit_bytes=vram_limit_bytes)
    worker_healthy = (
        active_request_id is None
        and pid_stable
        and (
            (running and isinstance(pid, int) and pid > 0)
            if expected_running
            else (pid is None or running)
        )
    )
    return {
        "expected_running": bool(expected_running),
        "running": running,
        "pid": pid,
        "stable_pid_before_turn": stable_pid,
        "pid_stable": pid_stable,
        "active_request_id": active_request_id,
        "healthy": worker_healthy,
        "memory": memory,
    }


def _turn_record(
    *,
    turn_number: int,
    case: PromptCase,
    session_id: str,
    peer_session_id: str,
    state: dict[str, Any],
    answer: dict[str, Any],
    elapsed_seconds: float,
    worker: dict[str, Any],
    peer_before_digest: str,
    peer_after_digest: str,
    prior_answer_digests: set[str],
    prior_sentence_keys: set[str] | None = None,
    expected_factbook: dict[str, Any] | None = None,
    error: str | None = None,
) -> dict[str, Any]:
    text = str(answer.get("content", "")).strip()
    repetition = _sentence_repetition_metrics(text)
    korean = _korean_completion_metrics(text)
    role_leaks = _role_token_leaks(text)
    control_leaks = _control_marker_leaks(text)
    answer_digest = hashlib.sha256(
        re.sub(r"\s+", " ", text).strip().casefold().encode("utf-8")
    ).hexdigest()
    cross_turn_duplicate = bool(text) and answer_digest in prior_answer_digests
    current_sentence_keys = _substantive_sentence_keys(text)
    prior_keys = prior_sentence_keys or set()
    reused_sentence_keys = current_sentence_keys & prior_keys
    cross_turn_sentence_echo = len(reused_sentence_keys) >= 2 or (
        len(current_sentence_keys) == 1
        and bool(reused_sentence_keys)
        and len(next(iter(current_sentence_keys))) >= 96
    )
    generated_tokens = max(0, int(answer.get("generated_tokens", 0) or 0))
    route_ok = (
        generated_tokens == 0
        if case.expected_route == "grounded"
        else generated_tokens > 0
    )
    peer_unchanged = peer_before_digest == peer_after_digest
    memory = worker["memory"]
    checks = _answer_checks(
        answer,
        state,
        case.prompt,
        required_groups=case.required_groups,
        expected_factbook=expected_factbook,
    )
    checks.update(
        {
            "grounding_route": route_ok,
            "no_cross_turn_exact_duplicate": not cross_turn_duplicate,
            "no_cross_turn_sentence_echo": not cross_turn_sentence_echo,
            "worker_healthy": bool(worker["healthy"]),
            "memory_observed_when_required": not worker["expected_running"]
            or bool(memory["memory_observed"]),
            "gpu_memory_within_limit_when_observed": memory["gpu_memory_within_limit"]
            is not False,
            "session_isolated": peer_session_id != session_id and peer_unchanged,
            "interactive_latency_within_limit": float(elapsed_seconds)
            <= MAX_INTERACTIVE_TURN_SECONDS,
        }
    )
    passed = all(checks.values())
    record: dict[str, Any] = {
        "turn": int(turn_number),
        "case": case.label,
        "session_id": session_id,
        "peer_session_id": peer_session_id,
        "expected_route": case.expected_route,
        "prompt": case.prompt,
        "answer": text,
        "answer_sha256": answer_digest,
        "finish_reason": answer.get("finish_reason"),
        "generation_mode": answer.get("generation_mode"),
        "continuations": answer.get("continuations"),
        "generated_tokens": generated_tokens,
        "explicit_truncation": _explicitly_truncated(answer, state),
        "elapsed_seconds": round(max(0.0, float(elapsed_seconds)), 6),
        "repetition": repetition,
        "role_token_leaks": role_leaks,
        "control_marker_leaks": control_leaks,
        "empty_answer": not bool(text),
        "korean_completion": korean,
        "cross_turn_exact_duplicate": cross_turn_duplicate,
        "cross_turn_sentence_reuse": sorted(reused_sentence_keys)[:8],
        "worker": worker,
        "session_isolation": {
            "peer_conversation_before_sha256": peer_before_digest,
            "peer_conversation_after_sha256": peer_after_digest,
            "peer_unchanged": peer_unchanged,
        },
        "checks": checks,
        "passed": passed,
    }
    if error is not None:
        record["error"] = error[:512]
    return record


def _summarize_turns(
    turns: list[dict[str, Any]], requested_turns: int
) -> dict[str, Any]:
    passed_turns = sum(turn.get("passed") is True for turn in turns)
    failures: dict[str, int] = {}
    total_sentences = 0
    duplicate_sentences = 0
    gpu_samples = 0
    gpu_over_limit = 0
    peak_rss = 0
    peak_gpu = 0
    worker_expected_turns = 0
    worker_memory_observed_turns = 0
    quality_fallback_turns = 0
    for turn in turns:
        if turn.get("generation_mode") == "quality_fallback":
            quality_fallback_turns += 1
        for name, passed in dict(turn.get("checks", {})).items():
            if passed is not True:
                failures[name] = failures.get(name, 0) + 1
        repetition = dict(turn.get("repetition", {}))
        total_sentences += int(repetition.get("sentence_count", 0) or 0)
        duplicate_sentences += int(repetition.get("duplicate_sentence_count", 0) or 0)
        worker = dict(turn.get("worker", {}))
        memory = dict(worker.get("memory", {}))
        if worker.get("expected_running") is True:
            worker_expected_turns += 1
            if memory.get("memory_observed") is True:
                worker_memory_observed_turns += 1
        rss = memory.get("worker_rss_bytes")
        gpu = memory.get("worker_gpu_memory_bytes")
        if isinstance(rss, int):
            peak_rss = max(peak_rss, rss)
        if isinstance(gpu, int):
            gpu_samples += 1
            peak_gpu = max(peak_gpu, gpu)
            if memory.get("gpu_memory_within_limit") is False:
                gpu_over_limit += 1
    success_rate = 0.0 if requested_turns <= 0 else passed_turns / int(requested_turns)
    # A bounded safety answer is valid UI behavior, but it is not a completed
    # model answer. Release evidence therefore permits no quality fallback.
    allowed_quality_fallback_turns = 0
    quality_fallback_gate_passed = (
        quality_fallback_turns <= allowed_quality_fallback_turns
    )
    return {
        "requested_turns": int(requested_turns),
        "completed_turns": len(turns),
        "passed_turns": passed_turns,
        "failed_turns": max(0, int(requested_turns) - passed_turns),
        "turn_success_rate": success_rate,
        "quality_fallback_turns": quality_fallback_turns,
        "quality_fallback_rate": (
            0.0
            if requested_turns <= 0
            else quality_fallback_turns / int(requested_turns)
        ),
        "allowed_quality_fallback_turns": allowed_quality_fallback_turns,
        "quality_fallback_gate_passed": quality_fallback_gate_passed,
        "content_answer_rate": (
            0.0
            if requested_turns <= 0
            else (int(requested_turns) - quality_fallback_turns) / int(requested_turns)
        ),
        "failed_check_counts": failures,
        "sentence_repetition_rate": (
            0.0 if total_sentences == 0 else duplicate_sentences / total_sentences
        ),
        "worker_expected_turns": worker_expected_turns,
        "worker_memory_observed_turns": worker_memory_observed_turns,
        "worker_memory_coverage_rate": (
            1.0
            if worker_expected_turns == 0
            else worker_memory_observed_turns / worker_expected_turns
        ),
        "peak_worker_rss_bytes": peak_rss or None,
        "gpu_memory_observed_turns": gpu_samples,
        "gpu_memory_coverage_rate": (
            0.0 if worker_expected_turns == 0 else gpu_samples / worker_expected_turns
        ),
        "peak_worker_gpu_memory_bytes": peak_gpu or None,
        "gpu_memory_over_limit_turns": gpu_over_limit,
        "gpu_memory_verdict": (
            "unverified"
            if gpu_samples == 0
            else ("failed" if gpu_over_limit else "passed")
        ),
        "release_schedule_gate_passed": requested_turns == RECOMMENDED_STRESS_TURNS
        and len(turns) == RECOMMENDED_STRESS_TURNS,
        "strict_turn_gate_passed": requested_turns == RECOMMENDED_STRESS_TURNS
        and len(turns) == requested_turns
        and passed_turns == requested_turns
        and quality_fallback_gate_passed,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run offline Gemma 4 chat turns and emit raw per-turn integrity evidence "
            f"(default {DEFAULT_TURNS}; recommended stress run "
            f"{RECOMMENDED_STRESS_TURNS})."
        )
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument(
        "--timeout",
        type=_bounded_timeout,
        default=MAX_INTERACTIVE_TURN_SECONDS,
    )
    parser.add_argument(
        "--output",
        help=(
            "optional JSON evidence path outside the source tree; written atomically"
        ),
    )
    parser.add_argument(
        "--turns",
        type=_bounded_turn_count,
        default=DEFAULT_TURNS,
        help=(
            f"number of deterministic turns in [1, {MAX_STRESS_TURNS}]; "
            f"use {RECOMMENDED_STRESS_TURNS} for the release stress gate"
        ),
    )
    return parser


def _atomic_external_report(path: str | Path, report: dict[str, Any]) -> Path:
    """Persist a crash-readable checkpoint outside the source tree."""

    target = Path(path).expanduser().resolve(strict=False)
    source = _PROJECT_ROOT.resolve(strict=True)
    if target.is_relative_to(source):
        raise ValueError("completion evidence must be stored outside the source tree")
    target.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
    return target


def execute(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    _offline_environment()
    requested_turns = int(getattr(args, "turns", DEFAULT_TURNS))
    timeout = float(args.timeout)
    if not 1.0 <= timeout <= MAX_INTERACTIVE_TURN_SECONDS:
        raise ValueError(
            f"timeout must be in [1, {MAX_INTERACTIVE_TURN_SECONDS:.0f}] seconds"
        )
    cases = _prompt_cases(requested_turns)
    vram_limit_bytes = int(16.7 * 1024**3)
    report: dict[str, Any] = {
        "schema": "cogni.agent.completion.stress.v1",
        "status": "running",
        "requested_turns": requested_turns,
        "recommended_stress_turns": RECOMMENDED_STRESS_TURNS,
        "criteria": {
            "strict_turn_gate": "every requested turn must pass every boolean check",
            "repetition": (
                "zero repeated sentences, exact cross-turn answers, and substantive "
                "cross-turn sentence-block reuse"
            ),
            "completion": "stop finish, no explicit truncation, complete Korean boundary",
            "content": (
                "all 20 release turns must publish a grounded or Cogni-Core answer "
                "rather than a quality fallback"
            ),
            "session": "the inactive A/B peer conversation digest must remain unchanged",
            "worker": "resident worker healthy, stable, idle after each generated turn",
            "latency": (
                f"every turn must finish within {MAX_INTERACTIVE_TURN_SECONDS:.0f} seconds"
            ),
            "memory": (
                "worker RSS is required when resident; GPU memory is reported separately "
                "as unverified when the driver does not expose a per-process value"
            ),
        },
        "turns": [],
        "all_checks_passed": False,
    }
    service: ModelService | None = None
    managers: dict[str, AgentManager] = {}
    leases = GPULeaseManager()
    rhythm = RhythmController()
    prior_answer_digests: set[str] = set()
    prior_sentence_keys: set[str] = set()
    stable_worker_pid: int | None = None
    expected_factbook: dict[str, Any] | None = None
    try:
        verified = verify_artifact_manifest(args.model, args.manifest)
        report["verified_files"] = len(verified.files)
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for the real-model completion test")
        report["cuda_device"] = torch.cuda.get_device_name(0)
        factbook = build_runtime_factbook_from_verified(
            verified,
            args.manifest,
            build_version=__version__,
            device=report["cuda_device"],
        )
        report["factbook"] = factbook.as_payload()
        expected_factbook = _expected_factbook_identity(factbook)
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
            gpu_lease_owner="completion-validator-model",
            gpu_lease_purpose="inference",
            gpu_lease_vram_bytes=leases.max_vram_bytes,
        )
        for key in ("a", "b"):
            managers[key] = AgentManager(
                service,
                WorkspaceToolExecutor(_PROJECT_ROOT),
                session_id=f"completion-{key}",
                fact_grounder=RuntimeFactGrounder(factbook),
                # Runtime facts are routed through the deterministic grounder.
                # Repeating the entire Fact-book in every model prompt caused
                # later open turns to copy product metadata and safety prose.
                system_prompt=SYSTEM_PROMPT,
                rhythm=rhythm,
            )
        for index, case in enumerate(cases):
            selected_key = (
                "a"
                if index < len(DEFAULT_PROMPTS)
                else ("b" if (index - len(DEFAULT_PROMPTS)) % 2 == 0 else "a")
            )
            peer_key = "b" if selected_key == "a" else "a"
            manager = managers[selected_key]
            peer = managers[peer_key]
            before = manager.snapshot()
            peer_before_digest = _snapshot_conversation_digest(peer.snapshot())
            prior_assistant_ids = {
                str(message.get("id"))
                for message in before["conversation"]
                if message.get("role") == "assistant"
            }
            started = monotonic()
            turn_error: str | None = None
            try:
                manager.start_turn(case.prompt, "chat")
                state = _wait_for_turn(manager, timeout)
            except BaseException as error:
                turn_error = f"{type(error).__name__}: {error}"
                state = manager.snapshot()
            if state.get("status") != "succeeded" and turn_error is None:
                failure = state.get("error") or {}
                turn_error = (
                    f"agent turn failed: {failure.get('code', 'unknown')}: "
                    f"{failure.get('message', state.get('stage', 'failed'))}"
                )
            assistants = [
                message
                for message in state["conversation"]
                if message.get("role") == "assistant"
                and str(message.get("id")) not in prior_assistant_ids
            ]
            answer = assistants[-1] if assistants else {}
            if not assistants and turn_error is None:
                turn_error = (
                    "successful agent turn contains no newly owned assistant response"
                )
            expected_worker = (
                case.expected_route == "generated" or stable_worker_pid is not None
            )
            worker = _worker_snapshot(
                service,
                expected_running=expected_worker,
                stable_pid=stable_worker_pid,
                vram_limit_bytes=vram_limit_bytes,
            )
            if stable_worker_pid is None and worker["running"]:
                stable_worker_pid = int(worker["pid"])
            peer_after_digest = _snapshot_conversation_digest(peer.snapshot())
            record = _turn_record(
                turn_number=index + 1,
                case=case,
                session_id=manager.session_id,
                peer_session_id=peer.session_id,
                state=state,
                answer=answer,
                elapsed_seconds=monotonic() - started,
                worker=worker,
                peer_before_digest=peer_before_digest,
                peer_after_digest=peer_after_digest,
                prior_answer_digests=prior_answer_digests,
                prior_sentence_keys=prior_sentence_keys,
                expected_factbook=expected_factbook,
                error=turn_error,
            )
            report["turns"].append(record)
            if record["answer"]:
                prior_answer_digests.add(record["answer_sha256"])
                prior_sentence_keys.update(_substantive_sentence_keys(record["answer"]))
            report["completed_turns"] = len(report["turns"])
            report["summary"] = _summarize_turns(report["turns"], requested_turns)
            if getattr(args, "output", None):
                _atomic_external_report(args.output, report)
            print(
                (
                    f"turn {index + 1}/{requested_turns} "
                    f"status={'passed' if record['passed'] else 'failed'} "
                    f"elapsed={record['elapsed_seconds']:.3f}s"
                ),
                file=sys.stderr,
                flush=True,
            )
            if manager.is_active:
                report["aborted_after_turn"] = index + 1
                break
    except BaseException as error:
        report["error"] = f"{type(error).__name__}: {error}"[:512]
    finally:
        report["summary"] = _summarize_turns(report["turns"], requested_turns)
        try:
            for manager in managers.values():
                manager.shutdown()
            if not managers and service is not None:
                service.stop()
        except BaseException as cleanup_error:
            report["cleanup_error"] = (
                f"{type(cleanup_error).__name__}: {cleanup_error}"[:512]
            )
            report["status"] = "failed"
            report["all_checks_passed"] = False
        report["worker_cleaned"] = service is None or not service.is_running
        report["gpu_lease_released"] = leases.active is None
        report["gpu_lease_history"] = [
            {
                "epoch": event.lease.epoch,
                "purpose": event.lease.purpose,
                "reason": event.reason,
            }
            for event in leases.history
        ]
        report["cleanup_checks"] = {
            "worker_cleaned": report["worker_cleaned"],
            "gpu_lease_released": report["gpu_lease_released"],
        }
        report["all_checks_passed"] = bool(
            report["summary"]["strict_turn_gate_passed"]
            and report["worker_cleaned"]
            and report["gpu_lease_released"]
            and "cleanup_error" not in report
            and "error" not in report
        )
        report["status"] = "passed" if report["all_checks_passed"] else "failed"
    return report, 0 if report["all_checks_passed"] else 1


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report, code = execute(args)
    reconfigure = getattr(sys.stdout, "reconfigure", None)
    if callable(reconfigure):
        reconfigure(encoding="utf-8", errors="replace")
    if args.output:
        target = _atomic_external_report(args.output, report)
        print(
            json.dumps(
                {
                    "status": report["status"],
                    "all_checks_passed": report["all_checks_passed"],
                    "output": str(target),
                    "summary": report["summary"],
                },
                ensure_ascii=False,
                indent=2,
            )
        )
    else:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
