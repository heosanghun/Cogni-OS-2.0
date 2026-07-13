"""Thread-safe product manager for chat, bounded tools, and UI telemetry."""

from __future__ import annotations

from collections import deque
from collections.abc import Callable, Mapping
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
from inspect import Parameter, signature
import re
import secrets
from threading import Condition, Event, RLock, Thread
from time import monotonic
from typing import Any, Protocol
import unicodedata

import torch

from cogni_flow.rhythm import RhythmController

from .conversation import BoundedConversationStore
from .conversation_fastpath import ConversationFastPath
from .fact_grounding import RuntimeFactGrounder
from .model_service import (
    GenerationCancelled,
    ModelServiceError,
    truncate_repeated_tokens,
)
from .prompting import decode_response, render_chat_prompt, stop_token_ids
from .response_quality import (
    QualityAction,
    QualityCode,
    ResponseQualityError,
    compose_observed_contract_response,
    has_near_duplicate_sentences,
    inspect_response,
    normalize_exact_item_response,
    normalize_exact_sentence_response,
    normalize_maximum_item_response,
    request_topic_terms,
    requested_category_counts,
    requested_exact_item_count,
    requested_exact_sentence_count,
    requested_maximum_items,
    response_avoids_dangling_sentence_start,
    response_avoids_generic_outline,
    response_avoids_prompt_echo,
    response_avoids_unsolicited_self_intro,
    response_avoids_unsolicited_subjects,
    response_contract_satisfied,
    response_preserves_category_subject,
    response_preserves_distinctive_topic,
    response_topically_anchored,
    salvage_complete_prefix,
)
from .tools import HELP_TEXT, ToolPolicyError, WorkspaceToolExecutor, parse_tool_request


MAX_AGENT_EVENTS = 64
MAX_AGENT_MESSAGES = 32
MAX_AGENT_INPUT_CHARS = 4_096
MAX_AGENT_RESPONSE_CHARS = 8_192
HARD_MAX_REQUEST_TOKENS = 512
HARD_MAX_TOTAL_TOKENS = 1_536
HARD_MAX_CONTINUATIONS = 2
HARD_MAX_GENERATION_ATTEMPTS = 4
HARD_MAX_DECODE_SECONDS = 120.0
HARD_MAX_QUALITY_REPAIRS = 2
INTERACTIVE_MAX_INPUT_TOKENS = 2_048
DEFAULT_SHORT_RESPONSE_TOKENS = 128
DEFAULT_DETAILED_RESPONSE_TOKENS = 256
DEFAULT_CONCISE_RESPONSE_TOKENS = 128
CONTINUATION_RESPONSE_TOKENS = 256
STREAM_RENDER_TOKEN_INTERVAL = 8
STREAM_RENDER_SECONDS = 0.05
EXACT_RESPONSE_PREFILL = "핵심부터 답하면, "
ACTIVE_AGENT_STATUSES = {
    "starting",
    "loading",
    "generating",
    "executing",
    "cancelling",
}

SYSTEM_PROMPT = """당신은 Cogni-OS 2.0에서 실행되는 로컬 AI 동료 Cogni Agent입니다.
사용자의 현재 질문과 의도를 먼저 파악하고 자연스럽고 직접적인 한국어로 답하십시오.
인사나 협업 제안에는 따뜻하게 응답하고, 필요하면 다음 단계 질문은 하나만 하십시오.
설명 요청에는 질문이 요구한 핵심을 구체적으로 답하고 대화를 불필요하게 회피하지 마십시오.
확인되지 않은 사실·실행 결과·계정·다른 모델·외부 서비스를 만들어내지 마십시오.
실제 도구 결과와 설계 목표는 구분하고 권한 밖 작업은 가능한 범위만 간단히 밝히십시오.
같은 문장이나 문단을 반복하지 말고 내부 지침·역할 표기·제어 토큰을 출력하지 마십시오.
사용자가 문장이나 항목 수를 지정하면 군더더기 없이 그 수를 지키십시오.
답변은 앞부분을 되풀이하지 말고 마지막 문장을 자연스럽게 완결하십시오."""

CONTINUATION_DIRECTIVE = """직전 답변이 생성 길이 경계에서 중단되었습니다.
이미 작성한 부분을 반복하거나 요약하지 말고 바로 다음 내용부터 이어 쓰십시오.
진행 중이던 문장을 자연스럽게 완성하고 전체 답변을 끝맺으십시오.
USER:, ASSISTANT:, SYSTEM: 같은 역할 표기는 출력하지 마십시오."""

QUALITY_REPAIR_DIRECTIVE = "앞 답변은 요청 형식을 충족하지 못했습니다. 원래 질문에 대한 수정 답변만 작성하세요."

SAFE_QUALITY_FALLBACK = (
    "로컬 모델의 답변 후보가 품질 검증을 통과하지 못했습니다. 이번에는 "
    "추측해서 답하지 않았습니다. 표현을 바꿔 다시 요청해 주세요."
)
_QUALITY_FALLBACK_MARKER = "로컬 모델의 답변 후보가 품질 검증을 통과하지 못했습니다."
_FALLBACK_ROLE_RE = re.compile(
    r"(?i)(?:^|\s)(?:user|assistant|system|model|tool|사용자|시스템)\s*:\s*"
)
_FALLBACK_CONTROL_RE = re.compile(r"<[^<>\r\n]{0,128}>|\[[^\[\]\r\n]{0,128}\]")


def safe_quality_fallback(message: str) -> str:
    """Create one bounded, request-specific, non-authoritative safety reply."""

    excerpt = _FALLBACK_CONTROL_RE.sub(" ", str(message))
    excerpt = _FALLBACK_ROLE_RE.sub(" ", excerpt)
    excerpt = re.sub(r"[\r\n\t]+", " ", excerpt)
    excerpt = re.sub(r"\s+", " ", excerpt).strip(" \"'“”‘’")
    if not excerpt:
        return SAFE_QUALITY_FALLBACK
    if len(excerpt) > 42:
        excerpt = excerpt[:42].rstrip() + "…"
    return (
        f"요청하신 ‘{excerpt}’에 대해 {_QUALITY_FALLBACK_MARKER} "
        "이번에는 추측해서 답하지 않았습니다. 표현을 바꿔 다시 요청해 주세요."
    )


def _is_quality_fallback(text: str) -> bool:
    return _QUALITY_FALLBACK_MARKER in text and text.endswith(
        "표현을 바꿔 다시 요청해 주세요."
    )


_DETAILED_INTENT_TERMS = (
    "모두",
    "자세히",
    "상세",
    "심층",
    "단계",
    "계획",
    "분석",
    "비교",
    "설명",
    "구현",
    "코드",
    "complete",
    "detailed",
    "explain",
    "step by step",
)
_CONTINUE_REQUEST_RE = re.compile(
    r"^\s*(?:계속(?:해서|해|해주세요|해줘)?|계속\s*이어서(?:\s*답(?:변)?(?:해|해주세요|해줘)?)?|"
    r"이어서(?:\s*답(?:변)?(?:해|해주세요|해줘)?)?|이어(?:\s*)?줘|continue|go\s+on)\s*[.!?~]*\s*$",
    re.IGNORECASE,
)
_CONCISE_INTENT_RE = re.compile(
    r"(?:[한두세네다섯여섯일곱여덟아홉열1-9]\s*문장|"
    r"핵심만|간결(?:하게)?|짧게|\d+\s*개\s*이내|이내로)",
    re.IGNORECASE,
)
_EXACT_SENTENCE_PHRASE_RE = re.compile(
    r"(?:한|두|세|네|다섯|여섯|일곱|여덟|아홉|열|[1-9])\s*문장(?:으로)?",
    re.IGNORECASE,
)
_EXACT_ITEM_PHRASE_RE = re.compile(
    r"(?:한|두|세|네|다섯|여섯|일곱|여덟|아홉|열|[1-9])\s*"
    r"(?:단계|항목|가지|개)"
)
_FORMAL_INSTRUCTION_RE = re.compile(
    r"(?P<action>설명|정리|답|제시|알려)(?:해|하)?(?:\s*주세요|세요|줘|주십시오)?"
    r"\s*[.!?。！？]*\s*$",
    re.IGNORECASE,
)
_CONTEXT_REFERENCE_RE = re.compile(
    r"(?:방금|앞서|앞선|이전\s*(?:답변|말|내용)|위\s*(?:답변|내용)|"
    r"그중|그것|그걸|그\s*이야기|그럼|이제|이어서|계속\s*(?:이어|설명|답)|"
    r"(?:첫|두|세)\s*번째|마지막\s*(?:답변|항목))",
    re.IGNORECASE,
)


def _requires_prior_context(message: str) -> bool:
    """Keep model-visible history only for an explicit contextual reference."""

    return _CONTEXT_REFERENCE_RE.search(message[:512]) is not None


def _exact_response_prefill(message: str) -> str:
    """Build a bounded, request-grounded Korean continuation anchor."""

    match = _EXACT_SENTENCE_PHRASE_RE.search(message[:512])
    if match is None:
        return EXACT_RESPONSE_PREFILL
    subject = re.sub(r"\s+", " ", message[: match.start()]).strip(" ,.:;!?。！？")
    if not 4 <= len(subject) <= 180:
        return EXACT_RESPONSE_PREFILL
    if subject.endswith(("을", "를")):
        return subject + " 설명하면, "
    return subject + "에 답하면, "


def _exact_item_response_prefill(message: str) -> str:
    """Build a request-grounded anchor for an exact step/item count."""

    match = _EXACT_ITEM_PHRASE_RE.search(message[:512])
    if match is None:
        return EXACT_RESPONSE_PREFILL
    subject = re.sub(r"\s+", " ", message[: match.start()]).strip(" ,.:;!?。！？")
    if not 4 <= len(subject) <= 180:
        return EXACT_RESPONSE_PREFILL
    return subject + " 정리하면, "


def _formal_response_prefill(message: str) -> str | None:
    """Anchor an explicit explanation/summary request without exposing it."""

    match = _FORMAL_INSTRUCTION_RE.search(message[:512])
    if match is None:
        return None
    subject = re.sub(r"\s+", " ", message[: match.start()]).strip(" ,.:;!?。！？")
    if not 4 <= len(subject) <= 220:
        return None
    action = match.group("action")
    continuation = {
        "설명": "설명하면",
        "정리": "정리하면",
        "답": "답하면",
        "제시": "제시하면",
        "알려": "답하면",
    }[action]
    return f"{subject} {continuation}, "


_LEADING_ASSISTANT_RE = re.compile(
    r"\A\s*(?:(?:(?:<start_of_turn>|<\|start_of_turn\|>|<\|turn>)\s*)"
    r"(?:assistant|model)\s*:?\s*|(?:assistant|model|어시스턴트)\s*:\s*)",
    re.IGNORECASE,
)
_ROLE_BOUNDARY_RE = re.compile(
    r"(?im)^[ \t]*(?:(?:<start_of_turn>|<\|start_of_turn\|>)[ \t]*)?"
    r"(?:user|assistant|model|system|tool|사용자|어시스턴트|시스템)[ \t]*:[ \t]*"
)
_TURN_TOKEN_RE = re.compile(
    r"(?:<end_of_turn>|<\|end_of_turn\|>|<\|eot_id\|>|<turn\|>|\[턴\s*종료\])",
    re.IGNORECASE,
)
_TURN_START_BOUNDARY_RE = re.compile(
    r"(?im)^[ \t]*(?:<start_of_turn>|<\|start_of_turn\|>|<\|turn>)[ \t]*"
    r"(?:user|assistant|model|system|tool)[ \t]*"
)
_RESERVED_OUTPUT_RE = re.compile(
    r"(?:<unused\d+>|\[multimodal\]|<\|(?:image|audio|endoftext|startoftext)\|>)",
    re.IGNORECASE,
)
_FACTBOOK_ECHO_RE = re.compile(r"(?im)^[ \t]*\[Runtime Fact-book:")
_QUALITY_REPAIR_ECHO_RE = re.compile(
    r"앞 답변은 요청 형식을 충족하지 못했습니다|"
    r"원래 질문에 대한 수정 답변만 작성하세요|"
    r"사용자 질문에 처음부터 다시 답하되|"
    r"사용자가 문장 수를 지정했다면|"
    r"사용자가 문장이나 항목 수를 지정하면|"
    r"반복된 질문에는 최신 답변만 제공하세요|"
    r"최신 답변을 제공하고, 이전 답변은|"
    r"\d+개의 완결된 문장으로 작성해야 합니다|"
    r"물음의 핵심과 요청 형식에 맞게|"
    r"아래 내용을\s*\d+\s*문장으로 작성하세요|"
    r"현재 답변이 사용자 요청의 전체 범위|"
    r"앞 답변의\s*\d+\s*[~～-]\s*\d+번 핵심|"
    r"수정 답변을 작성해서 사용자와 시스템|"
    r"서로 다른 핵심 내용을 반복하지 않고 한 번씩만|"
    r"각 문장을 자연스러운 서술어로 끝까지 완결|"
    r"질문의 핵심 용어인|"
    r"같은 문장이나 표현을 반복하지 말고|"
    r"요청한 범위를 빠뜨리지 말고|"
    r"질문의 핵심 용어를 직접 유지하세요|"
    r"제목·목록·HTML 표기 없이"
)
_SENTENCE_UNIT_RE = re.compile(
    r".+?(?:[.!?。！？]+(?=\s|$)|\n{2,}|$)",
    re.DOTALL,
)
_STRUCTURED_REPEAT_RE = re.compile(
    r"(?m)^\s*(?:```|\||[-*+]\s+|\d+[.)]\s+)",
)


@dataclass(frozen=True)
class ResponseBudget:
    """A deterministic, bounded decode budget for one user turn."""

    first_request: int
    total: int
    max_continuations: int


@dataclass(frozen=True, slots=True)
class _DecodeDeadlineChunk:
    """Internal terminal marker for a backend-enforced total decode timeout."""

    token_ids: torch.Tensor
    request_id: int = 0
    generated_total: int = 0
    final: bool = True
    cancelled: bool = False
    finish_reason: str = "stop"
    generation_mode: str = "cogni_core"
    deadline_exceeded: bool = True


class AgentBusyError(RuntimeError):
    pass


class NoActiveAgentTurnError(RuntimeError):
    pass


class GenerationBackend(Protocol):
    tokenizer: Any
    active_request_id: int | None

    def start(self) -> Any: ...

    def iter_generate_tokens(
        self,
        prompt: str,
        *,
        max_new_tokens: int,
        decode_mode: str = "conversation",
        sampling_seed: int | None = None,
        total_timeout: float | None = None,
    ) -> Any: ...

    def cancel(self, request_id: int | None = None) -> bool: ...

    def stop(self, timeout: float = 10.0) -> None: ...


FailureSink = Callable[[str, str], None]
EvolutionSnapshot = Callable[[], Mapping[str, Any]]
AvailabilityCheck = Callable[[], bool]
TurnFinish = tuple[str, str, int]


class AgentManager:
    """Own one bounded conversational turn and expose immutable UI snapshots."""

    def __init__(
        self,
        model_service: GenerationBackend,
        tool_executor: WorkspaceToolExecutor,
        *,
        session_id: str | None = None,
        max_new_tokens: int = HARD_MAX_REQUEST_TOKENS,
        max_total_new_tokens: int | None = None,
        max_continuations: int = HARD_MAX_CONTINUATIONS,
        max_generation_attempts: int = HARD_MAX_GENERATION_ATTEMPTS,
        max_decode_seconds: float = HARD_MAX_DECODE_SECONDS,
        system_prompt: str = SYSTEM_PROMPT,
        failure_sink: FailureSink | None = None,
        evolution_snapshot: EvolutionSnapshot | None = None,
        availability_check: AvailabilityCheck | None = None,
        conversation_fast_path: ConversationFastPath | None = None,
        fact_grounder: RuntimeFactGrounder | None = None,
        rhythm: RhythmController | None = None,
    ) -> None:
        if not callable(getattr(model_service, "iter_generate_tokens", None)):
            raise TypeError("model_service must provide tensor generation")
        if not isinstance(tool_executor, WorkspaceToolExecutor):
            raise TypeError("tool_executor must be WorkspaceToolExecutor")
        if not 1 <= max_new_tokens <= HARD_MAX_REQUEST_TOKENS:
            raise ValueError("max_new_tokens must be in [1, 512]")
        if max_total_new_tokens is None:
            max_total_new_tokens = min(
                HARD_MAX_TOTAL_TOKENS, max_new_tokens * (HARD_MAX_CONTINUATIONS + 1)
            )
        if not max_new_tokens <= max_total_new_tokens <= HARD_MAX_TOTAL_TOKENS:
            raise ValueError("max_total_new_tokens must fit [max_new_tokens, 1536]")
        if not 0 <= max_continuations <= HARD_MAX_CONTINUATIONS:
            raise ValueError("max_continuations must be in [0, 2]")
        if (
            not isinstance(max_generation_attempts, int)
            or isinstance(max_generation_attempts, bool)
            or not 1 <= max_generation_attempts <= HARD_MAX_GENERATION_ATTEMPTS
        ):
            raise ValueError("max_generation_attempts must be in [1, 4]")
        if (
            not isinstance(max_decode_seconds, (int, float))
            or isinstance(max_decode_seconds, bool)
            or not 0.01 <= float(max_decode_seconds) <= HARD_MAX_DECODE_SECONDS
        ):
            raise ValueError("max_decode_seconds must be in [0.01, 120.0]")
        if not isinstance(system_prompt, str) or not 1 <= len(system_prompt) <= 8_192:
            raise ValueError("system_prompt must be bounded text")
        if fact_grounder is not None and not isinstance(
            fact_grounder, RuntimeFactGrounder
        ):
            raise TypeError("fact_grounder must be a RuntimeFactGrounder or None")
        if conversation_fast_path is not None and not isinstance(
            conversation_fast_path, ConversationFastPath
        ):
            raise TypeError(
                "conversation_fast_path must be a ConversationFastPath or None"
            )
        if rhythm is not None and not isinstance(rhythm, RhythmController):
            raise TypeError("rhythm must be a RhythmController or None")
        selected_session = (
            f"conversation-{secrets.token_hex(12)}"
            if session_id is None
            else session_id
        )
        if (
            not isinstance(selected_session, str)
            or not 1 <= len(selected_session) <= 64
            or any(
                character
                not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_"
                for character in selected_session
            )
        ):
            raise ValueError("session_id must be a bounded ASCII identifier")
        self.model_service = model_service
        self.tool_executor = tool_executor
        self.session_id = selected_session
        self.max_new_tokens = int(max_new_tokens)
        self.max_total_new_tokens = int(max_total_new_tokens)
        self.max_continuations = int(max_continuations)
        self.max_generation_attempts = int(max_generation_attempts)
        self.max_decode_seconds = float(max_decode_seconds)
        self.system_prompt = system_prompt
        self.failure_sink = failure_sink
        self.evolution_snapshot = evolution_snapshot
        self.availability_check = availability_check
        self.conversation_fast_path = conversation_fast_path
        self.fact_grounder = fact_grounder
        self.rhythm = rhythm if rhythm is not None else RhythmController()
        self.conversations = BoundedConversationStore(
            max_sessions=1,
            max_turns=24,
            max_chars=24_000,
            max_message_chars=MAX_AGENT_RESPONSE_CHARS,
        )
        self._condition = Condition(RLock())
        self._status = "ready"
        self._stage = "ready"
        self._sequence = 0
        self._progress = 100
        self._events: deque[dict[str, Any]] = deque(maxlen=MAX_AGENT_EVENTS)
        self._messages: deque[dict[str, Any]] = deque(maxlen=MAX_AGENT_MESSAGES)
        self._model_excluded_exchanges: deque[tuple[str, str]] = deque(maxlen=24)
        self._active_turn: str | None = None
        self._active_user_sequence: int | None = None
        self._cancel_event = Event()
        self._thread: Thread | None = None
        self._error: dict[str, str] | None = None
        self._core: dict[str, Any] = self._core_state()
        self._completion: dict[str, Any] = self._completion_state()
        self._response_stop_ids = stop_token_ids(self.model_service.tokenizer)

    @property
    def is_active(self) -> bool:
        with self._condition:
            return self._status in ACTIVE_AGENT_STATUSES

    def snapshot(self, *, after: int | None = None) -> dict[str, Any]:
        with self._condition:
            events = list(self._events)
            if after is not None:
                events = [event for event in events if event["seq"] > after]
            evolution = {
                "failures": 0,
                "last_run": None,
                "sandbox": "미구성",
                "status": "대기",
                "running": False,
            }
            if self.evolution_snapshot is not None:
                try:
                    candidate = dict(self.evolution_snapshot())
                except Exception:
                    candidate = {"status": "상태 오류"}
                evolution.update(candidate)
            return {
                "status": self._status,
                "stage": self._stage,
                "seq": self._sequence,
                "progress": self._progress,
                "events": deepcopy(events),
                "conversation": deepcopy(list(self._messages)),
                "active_turn": self._active_turn,
                "error": deepcopy(self._error),
                "core": deepcopy(self._core),
                "completion": deepcopy(self._completion),
                "evolution": evolution,
            }

    def wait_snapshot(self, after: int, timeout: float = 10.0) -> dict[str, Any]:
        if after < 0 or not 0 <= timeout <= 15:
            raise ValueError("invalid agent state wait")
        with self._condition:
            self._condition.wait_for(lambda: self._sequence > after, timeout=timeout)
            return self.snapshot(after=after)

    def start_turn(self, message: str, mode: str = "chat") -> str:
        text = self._validate_message(message)
        if mode not in {"chat", "task"}:
            raise ValueError("mode must be chat or task")
        with self._condition:
            if self._status in ACTIVE_AGENT_STATUSES:
                raise AgentBusyError("another agent turn is active")
            availability = self.availability_check
        if availability is not None and not availability():
            raise AgentBusyError("the GPU is reserved by another system mode")
        with self._condition:
            if self._status in ACTIVE_AGENT_STATUSES:
                raise AgentBusyError("another agent turn is active")
            turn_id = secrets.token_hex(12)
            resume_truncated = mode == "chat" and self._resume_requested_locked(text)
            user_sequence = self.conversations.begin_user_turn(self.session_id, text)
            self._active_turn = turn_id
            self._active_user_sequence = user_sequence
            self._cancel_event = Event()
            self._error = None
            self._completion = self._completion_state(state="pending")
            self._append_message("user", text)
            self._transition_locked("starting", "accepted", 0)
            thread = Thread(
                target=self._run_turn,
                args=(turn_id, user_sequence, text, mode, resume_truncated),
                name=f"cogni-agent-{turn_id[:8]}",
                daemon=True,
            )
            self._thread = thread
            thread.start()
            return turn_id

    def cancel(self) -> None:
        with self._condition:
            if self._status not in ACTIVE_AGENT_STATUSES:
                raise NoActiveAgentTurnError("there is no active agent turn")
            self._cancel_event.set()
            self.model_service.cancel(None)
            self._transition_locked("cancelling", "cancelling", self._progress)

    def reset(self) -> None:
        with self._condition:
            if self._status in ACTIVE_AGENT_STATUSES:
                raise AgentBusyError("cannot reset an active agent turn")
            self.conversations.clear(self.session_id)
            self.session_id = f"conversation-{secrets.token_hex(12)}"
            self._messages.clear()
            self._model_excluded_exchanges.clear()
            self._error = None
            self._completion = self._completion_state()
            self._transition_locked("ready", "ready", 100)

    def stop_model(self) -> None:
        if self.is_active:
            raise AgentBusyError("cannot stop the model during an active turn")
        self.model_service.stop()

    def shutdown(self) -> None:
        if self.is_active:
            try:
                self.cancel()
            except NoActiveAgentTurnError:
                pass
        thread = self._thread
        if thread is not None:
            thread.join(timeout=5.0)
        self.model_service.stop()

    def _run_turn(
        self,
        turn_id: str,
        user_sequence: int,
        message: str,
        mode: str,
        resume_truncated: bool,
    ) -> None:
        try:
            with self.rhythm.inference_slot():
                if mode == "task":
                    finish = self._run_task(turn_id, user_sequence, message)
                else:
                    finish = self._run_chat(
                        turn_id,
                        user_sequence,
                        message,
                        resume_truncated=resume_truncated,
                    )
            if finish is not None:
                with self._condition:
                    self._finish_locked(turn_id, *finish)
        except GenerationCancelled:
            self._finish_cancelled(turn_id, user_sequence)
        except BaseException as exc:
            self._finish_failed(turn_id, user_sequence, exc)

    def _run_task(
        self, turn_id: str, user_sequence: int, message: str
    ) -> TurnFinish | None:
        with self._condition:
            if self._active_turn != turn_id:
                return
            self._transition_locked("executing", "tool", 35)
        request = parse_tool_request(message)
        if request is None:
            output = "작업 모드는 안전한 명시적 명령만 실행합니다.\n\n" + HELP_TEXT
            role = "system"
            artifact = None
        else:
            result = self.tool_executor.execute(request)
            output = result.output
            role = "tool"
            artifact = result.artifact
            if not result.ok:
                raise ToolPolicyError(result.output)
        if self._cancel_event.is_set():
            raise GenerationCancelled("task cancelled")
        self.conversations.commit_assistant_turn(
            self.session_id, user_sequence, output[:MAX_AGENT_RESPONSE_CHARS]
        )
        with self._condition:
            self._append_message(role, output, artifact=artifact)
            self._completion = self._completion_state(
                state="complete",
                finish_reason="tool",
            )
        return "succeeded", "complete", 100

    def _run_chat(
        self,
        turn_id: str,
        user_sequence: int,
        message: str,
        *,
        resume_truncated: bool,
    ) -> TurnFinish | None:
        if self.conversation_fast_path is not None and not resume_truncated:
            previous_user, previous_assistant = self._previous_fast_path_exchange()
            fast_answer = self.conversation_fast_path.answer(
                message,
                previous_user=previous_user,
                previous_assistant=previous_assistant,
            )
            if fast_answer is not None:
                return self._run_conversation_fast_path_answer(
                    turn_id,
                    user_sequence,
                    fast_answer,
                )
        if self.fact_grounder is not None and not resume_truncated:
            grounded = self.fact_grounder.answer(message)
            if grounded is None:
                previous = self._previous_assistant_content()
                grounded = self.fact_grounder.answer_followup(message, previous)
            if grounded is not None:
                return self._run_grounded_answer(
                    turn_id,
                    user_sequence,
                    message,
                    grounded,
                )
        with self._condition:
            if self._active_turn != turn_id:
                return
            self._core = self._core_state(active=("router", "swarm", "cts"))
            self._transition_locked("loading", "model", 10)
        self.model_service.start()
        if self._cancel_event.is_set():
            raise GenerationCancelled("cancelled before generation")
        budget = self._response_budget(message, resume_truncated=resume_truncated)
        exact_sentence_count = (
            None if resume_truncated else requested_exact_sentence_count(message)
        )
        exact_item_count = (
            None if resume_truncated else requested_exact_item_count(message)
        )
        maximum_item_count = (
            None if resume_truncated else requested_maximum_items(message)
        )
        isolate_turn_history = not resume_truncated and not _requires_prior_context(
            message
        )
        response_prefill = None
        if exact_sentence_count is not None:
            response_prefill = _exact_response_prefill(message)
        elif exact_item_count is not None:
            response_prefill = _exact_item_response_prefill(message)
        elif not resume_truncated:
            response_prefill = _formal_response_prefill(message)
        prompt = self._build_prompt(
            resume_truncated=resume_truncated,
            partial_assistant=response_prefill,
            isolate_history=isolate_turn_history,
        )
        message_id = self._append_message(
            "assistant",
            "",
            streaming=True,
            finish_reason=None,
            continuations=0,
            truncated=False,
            generated_tokens=0,
        )
        response = ""
        total_generated = 0
        continuations = 0
        request_budget = budget.first_request
        finish_reason = "stop"
        char_truncated = False
        repetition_boundary = False
        quality_boundary = False
        generation_mode = "cogni_core"
        quality_repair_attempts = 0
        generation_attempts = 0
        decode_started_at = monotonic()
        decode_deadline = decode_started_at + self.max_decode_seconds
        # Open conversation follows the model-card sampling profile. Explicit
        # explanation/shape requests use a bounded request-grounded prefill and
        # strict decode: the local E4B otherwise tends to echo the prompt or
        # drift topics while attempting to satisfy the requested shape.
        decode_mode = "strict" if response_prefill is not None else "conversation"
        best_safe_prefix = ""
        observed_sentences: list[str] = []
        last_candidate = ""
        diagnostic_codes: set[str] = set()
        decode_bound_reached = False
        with self._condition:
            self._core = self._core_state(active=("gemma", "router", "swarm", "cts"))
            self._transition_locked("generating", "decode", 55)
        while True:
            if generation_attempts >= self.max_generation_attempts:
                diagnostic_codes.add("attempt_limit")
                decode_bound_reached = True
                break
            if monotonic() >= decode_deadline:
                diagnostic_codes.add("decode_deadline")
                decode_bound_reached = True
                break
            generation_attempts += 1
            request_tokens: list[torch.Tensor] = []
            request_generated = 0
            request_reason = "stop"
            role_boundary = False
            response_before_request = response
            made_progress = False
            received_tokens = 0
            rendered_tokens = 0
            rendered_at = monotonic()
            for chunk in self._generation_stream(
                prompt,
                request_budget,
                decode_mode=decode_mode,
                timeout_seconds=max(0.01, decode_deadline - monotonic()),
                sampling_seed=self._sampling_seed(
                    user_sequence,
                    generation_attempts,
                    repair_message=(message if quality_repair_attempts else None),
                ),
            ):
                if getattr(chunk, "deadline_exceeded", False):
                    diagnostic_codes.add("decode_deadline")
                    decode_bound_reached = True
                    break
                if monotonic() >= decode_deadline:
                    diagnostic_codes.add("decode_deadline")
                    decode_bound_reached = True
                    try:
                        self.model_service.cancel(
                            getattr(self.model_service, "active_request_id", None)
                        )
                    except Exception:
                        pass
                    break
                if chunk.cancelled or self._cancel_event.is_set():
                    raise GenerationCancelled("generation cancelled")
                request_generated = max(request_generated, int(chunk.generated_total))
                if chunk.token_ids.numel():
                    request_tokens.append(chunk.token_ids)
                    received_tokens += int(chunk.token_ids.numel())
                render_due = bool(request_tokens) and (
                    chunk.final
                    or received_tokens - rendered_tokens >= STREAM_RENDER_TOKEN_INTERVAL
                    or monotonic() - rendered_at >= STREAM_RENDER_SECONDS
                )
                if render_due:
                    decoded, token_repetition = self._decode(request_tokens)
                    if continuations == 0:
                        decoded, _user_echo = self._strip_leading_user_echo(
                            decoded, message
                        )
                    segment, segment_role_boundary = self._clean_model_text(decoded)
                    role_boundary = role_boundary or segment_role_boundary
                    repetition_boundary = repetition_boundary or token_repetition
                    merged = self._merge_response(response_before_request, segment)
                    merged, segment_repetition = self._trim_repeated_text(merged)
                    repetition_boundary = repetition_boundary or segment_repetition
                    response, clipped = self._clip_response(merged)
                    char_truncated = char_truncated or clipped
                    quality = inspect_response(response, final=False)
                    if quality.should_stop_generation:
                        cut = quality.recommended_cut_index
                        if cut is not None:
                            response = response[:cut].rstrip()
                        quality_boundary = True
                    made_progress = made_progress or response != response_before_request
                    with self._condition:
                        self._update_message(
                            message_id,
                            response,
                            streaming=True,
                            finish_reason=None,
                            continuations=continuations,
                            truncated=False,
                            generated_tokens=total_generated + request_generated,
                        )
                        self._completion = self._completion_state(
                            state="streaming",
                            continuations=continuations,
                            generated_tokens=total_generated + request_generated,
                        )
                        self._transition_locked(
                            "generating",
                            "continuing" if continuations else "decode",
                            min(
                                95,
                                55
                                + int(
                                    (total_generated + request_generated)
                                    * 40
                                    / budget.total
                                ),
                            ),
                        )
                    rendered_tokens = received_tokens
                    rendered_at = monotonic()
                if chunk.final:
                    request_reason = self._chunk_finish_reason(chunk)
                    chunk_mode = getattr(chunk, "generation_mode", "cogni_core")
                    if chunk_mode != "cogni_core":
                        raise ModelServiceError(
                            "worker returned an unattested generation mode"
                        )

            total_generated += request_generated
            finish_reason = (
                "stop"
                if role_boundary or repetition_boundary or quality_boundary
                else request_reason
            )
            remaining = budget.total - total_generated
            terminal_quality = inspect_response(response, final=True)
            if response.strip():
                last_candidate = response
            if repetition_boundary:
                diagnostic_codes.add("token_repetition")
            if quality_boundary:
                diagnostic_codes.add("quality_boundary")
            diagnostic_codes.update(
                finding.code.value for finding in terminal_quality.findings
            )
            response_contract_incomplete = not response_contract_satisfied(
                message, response
            )
            if response_contract_incomplete:
                diagnostic_codes.add("response_contract")
            normalized = (
                normalize_exact_sentence_response(
                    message,
                    response,
                )
                or normalize_exact_item_response(
                    message,
                    response,
                )
                or normalize_maximum_item_response(message, response)
            )
            if normalized is not None:
                response = normalized
                terminal_quality = inspect_response(response, final=True)
                response_contract_incomplete = False
                finish_reason = "stop"
            observed_sentences = self._merge_distinct_sentences(
                observed_sentences,
                response,
            )
            observed_contract = compose_observed_contract_response(
                message,
                observed_sentences,
            )
            if (
                observed_contract
                and not self._response_adequate_for_request(message, response)
                and self._response_adequate_for_request(message, observed_contract)
                and not self._has_cross_turn_sentence_echo(observed_contract)
            ):
                response = observed_contract
                terminal_quality = inspect_response(response, final=True)
                response_contract_incomplete = False
                finish_reason = "stop"
            if (
                exact_sentence_count is None
                and exact_item_count is None
                and maximum_item_count is None
                and has_near_duplicate_sentences(response)
            ):
                deduplicated = " ".join(observed_sentences).strip()
                if deduplicated:
                    response = deduplicated
                    terminal_quality = inspect_response(response, final=True)
                    finish_reason = "stop"
                    diagnostic_codes.add("near_duplicate_trimmed")
            cross_turn_echo = self._has_cross_turn_sentence_echo(response)
            if cross_turn_echo:
                diagnostic_codes.add("cross_turn_echo")
            if not response.strip():
                diagnostic_codes.add("empty_candidate")
            near_duplicate_candidate = (
                exact_sentence_count is not None
                and has_near_duplicate_sentences(response)
            )
            if near_duplicate_candidate:
                diagnostic_codes.add("near_duplicate")
            response_adequate = self._response_adequate_for_request(
                message,
                response,
            )
            inadequacy_is_terminal = (
                bool(response.strip())
                and not response_adequate
                and finish_reason != "length"
                and terminal_quality.recommended_action is QualityAction.ACCEPT
            )
            if inadequacy_is_terminal:
                diagnostic_codes.add("insufficient_detail")

            safe_prefix = salvage_complete_prefix(response)
            if (
                safe_prefix
                and self._response_adequate_for_request(message, safe_prefix)
                and not self._has_cross_turn_sentence_echo(safe_prefix)
                and len(safe_prefix) > len(best_safe_prefix)
            ):
                best_safe_prefix = safe_prefix
            bounded_repair_needed = (
                not response.strip()
                or inadequacy_is_terminal
                or response_contract_incomplete
                or cross_turn_echo
                or (
                    budget.max_continuations == 0
                    and terminal_quality.recommended_action is QualityAction.CONTINUE
                )
            )
            if (
                bounded_repair_needed
                and quality_repair_attempts < HARD_MAX_QUALITY_REPAIRS
                and generation_attempts < self.max_generation_attempts
                and monotonic() < decode_deadline
                and remaining > 0
            ):
                quality_repair_attempts += 1
                last_candidate = response
                isolate_repair = (
                    isolate_turn_history
                    or cross_turn_echo
                    or (
                        response_contract_incomplete
                        and requested_exact_sentence_count(message) is not None
                    )
                )
                adaptive_sampling_repair = decode_mode == "strict" and (
                    bool(
                        diagnostic_codes
                        & {
                            QualityCode.TEMPLATE_REPETITION.value,
                            QualityCode.LOW_INFORMATION_REPETITION.value,
                            "token_repetition",
                            "quality_boundary",
                            "empty_candidate",
                            "near_duplicate",
                        }
                    )
                    or (
                        "response_contract" in diagnostic_codes
                        and requested_category_counts(message) is None
                    )
                    or (
                        quality_repair_attempts >= 2
                        and "insufficient_detail" in diagnostic_codes
                        and requested_category_counts(message) is None
                    )
                )
                repair_prefill = response_prefill
                if adaptive_sampling_repair:
                    # A deterministic retry reproduces the same local optimum.
                    # Switch only degeneration repairs to the bounded Gemma 4
                    # conversation profile; shape/topic-only repairs stay strict.
                    decode_mode = "conversation"
                    repair_prefill = None
                response = ""
                request_budget = min(
                    DEFAULT_CONCISE_RESPONSE_TOKENS,
                    self.max_new_tokens,
                    remaining,
                )
                prompt = self._build_quality_repair_prompt(
                    message,
                    issue_codes=tuple(sorted(diagnostic_codes)),
                    isolate_history=isolate_repair,
                    partial_assistant=repair_prefill,
                )
                repetition_boundary = False
                quality_boundary = False
                with self._condition:
                    self._transition_locked("generating", "repairing", self._progress)
                continue
            quality_needs_continuation = (
                finish_reason == "stop"
                and terminal_quality.recommended_action is QualityAction.CONTINUE
            )
            emergency_continuation = (
                (quality_needs_continuation or response_contract_incomplete)
                and quality_repair_attempts >= HARD_MAX_QUALITY_REPAIRS
                and continuations < 1
            )
            can_continue = (
                (
                    finish_reason == "length"
                    or quality_needs_continuation
                    or emergency_continuation
                )
                and not role_boundary
                and not repetition_boundary
                and (not quality_boundary or emergency_continuation)
                and not char_truncated
                and made_progress
                and (continuations < budget.max_continuations or emergency_continuation)
                and generation_attempts < self.max_generation_attempts
                and monotonic() < decode_deadline
                and remaining > 0
            )
            if not can_continue:
                break
            continuations += 1
            request_budget = min(
                self.max_new_tokens,
                CONTINUATION_RESPONSE_TOKENS,
                remaining,
            )
            if quality_needs_continuation:
                incomplete_start = min(
                    finding.start
                    for finding in terminal_quality.findings
                    if finding.code is QualityCode.INCOMPLETE_KOREAN_CLAUSE
                )
                response = self._complete_prefix(
                    response,
                    cutoff=incomplete_start if incomplete_start > 0 else None,
                )
                prompt = (
                    self._build_continuation_prompt(
                        response,
                        isolate_history=isolate_turn_history,
                    )
                    if response
                    else self._build_quality_repair_prompt(
                        message,
                        issue_codes=(QualityCode.INCOMPLETE_KOREAN_CLAUSE.value,),
                        isolate_history=isolate_turn_history,
                    )
                )
                with self._condition:
                    self._update_message(
                        message_id,
                        response,
                        streaming=True,
                        finish_reason=None,
                        continuations=continuations,
                        truncated=False,
                        generated_tokens=total_generated,
                    )
            else:
                prompt = self._build_continuation_prompt(
                    response,
                    isolate_history=isolate_turn_history,
                )
            with self._condition:
                self._transition_locked("generating", "continuing", self._progress)

        response = self._ensure_terminal_punctuation(response.strip())
        truncated = char_truncated or finish_reason == "length"
        if repetition_boundary:
            diagnostic_codes.add("token_repetition")
        if quality_boundary:
            diagnostic_codes.add("quality_boundary")
        if decode_bound_reached:
            diagnostic_codes.add("decode_bound")
        if truncated:
            # A length boundary is not permission to publish an unfinished
            # clause. A complete, adequate observed prefix is itself a safe
            # final answer; an incomplete public tail is never exposed.
            raw_response = response
            complete = salvage_complete_prefix(response)
            if (
                complete
                and self._response_adequate_for_request(message, complete)
                and not self._has_cross_turn_sentence_echo(complete)
            ):
                response = complete
                completed_sentences = len(re.findall(r"[.!?。！？]+(?=\s|$)", complete))
                if complete == raw_response.rstrip() or (
                    len(complete) >= 80 and completed_sentences >= 2
                ):
                    diagnostic_codes.add("length_salvaged")
                    finish_reason = "stop"
                    truncated = False
            else:
                diagnostic_codes.add("unsafe_truncated_tail")
                response = safe_quality_fallback(message)
                finish_reason = "stop"
                truncated = False
                generation_mode = "quality_fallback"
        if not truncated:
            if repetition_boundary or quality_boundary:
                response = self._complete_prefix(response)
            final_quality = inspect_response(response, final=True)
            diagnostic_codes.update(
                finding.code.value for finding in final_quality.findings
            )
            if final_quality.should_stop_generation:
                cut = final_quality.recommended_cut_index
                response = self._complete_prefix(response, cutoff=cut)
                final_quality = inspect_response(response, final=True)

            acceptable = (
                bool(response)
                and self._response_adequate_for_request(message, response)
                and not self._has_cross_turn_sentence_echo(response)
            )
            if not acceptable:
                if final_quality.recommended_action is not QualityAction.ACCEPT:
                    diagnostic_codes.add(final_quality.recommended_action.value)
                if not response_contract_satisfied(message, response):
                    diagnostic_codes.add("response_contract")
                if self._has_cross_turn_sentence_echo(response):
                    diagnostic_codes.add("cross_turn_echo")

                salvage_candidates = (
                    salvage_complete_prefix(response),
                    best_safe_prefix,
                    compose_observed_contract_response(
                        message,
                        observed_sentences,
                    ),
                )
                safe_response = max(
                    (
                        candidate
                        for candidate in salvage_candidates
                        if candidate
                        and self._response_adequate_for_request(message, candidate)
                        and not self._has_cross_turn_sentence_echo(candidate)
                    ),
                    key=len,
                    default="",
                )
                if safe_response:
                    response = safe_response
                    finish_reason = "stop"
                    truncated = False
                else:
                    response = safe_quality_fallback(message)
                    finish_reason = "stop"
                    truncated = False
                    generation_mode = "quality_fallback"
        if not response:
            diagnostic_codes.add("empty_candidate")
            if best_safe_prefix and self._response_adequate_for_request(
                message,
                best_safe_prefix,
            ):
                response = best_safe_prefix
                finish_reason = "stop"
                truncated = False
            else:
                response = safe_quality_fallback(message)
                finish_reason = "stop"
                truncated = False
                generation_mode = "quality_fallback"
        if generation_mode == "quality_fallback" and self.failure_sink is not None:
            try:
                self.failure_sink(
                    "ResponseQualityError",
                    self._quality_failure_diagnostic(
                        last_candidate,
                        diagnostic_codes,
                        attempts=generation_attempts,
                        generated_tokens=total_generated,
                    ),
                )
            except Exception:
                pass
        self.conversations.commit_assistant_turn(
            self.session_id, user_sequence, response
        )
        final_stage = "truncated" if truncated else "complete"
        with self._condition:
            self._update_message(
                message_id,
                response,
                streaming=False,
                finish_reason=finish_reason,
                continuations=continuations,
                truncated=truncated,
                generated_tokens=total_generated,
                generation_mode=generation_mode,
            )
            self._completion = self._completion_state(
                state=final_stage,
                finish_reason=finish_reason,
                continuations=continuations,
                truncated=truncated,
                generated_tokens=total_generated,
                generation_mode=generation_mode,
            )
            self._core = self._core_state()
        return "succeeded", final_stage, 100

    def _run_conversation_fast_path_answer(
        self,
        turn_id: str,
        user_sequence: int,
        answer: str,
    ) -> TurnFinish | None:
        """Publish one bounded social reply without loading the local model."""

        with self._condition:
            if self._active_turn != turn_id:
                return None
            self._core = self._core_state(active=("conversation_fastpath",))
            self._transition_locked("executing", "conversation_fastpath", 50)
        if self._cancel_event.is_set():
            raise GenerationCancelled("conversation fast path cancelled")
        response, clipped = self._clip_response(answer.strip())
        quality = inspect_response(response, final=True)
        if clipped or quality.recommended_action is not QualityAction.ACCEPT:
            raise ResponseQualityError(
                "conversation fast path produced an unsafe answer"
            )
        self.conversations.commit_assistant_turn(
            self.session_id,
            user_sequence,
            response,
        )
        with self._condition:
            self._append_message(
                "assistant",
                response,
                streaming=False,
                finish_reason="stop",
                continuations=0,
                truncated=False,
                generated_tokens=0,
                generation_mode="conversation_fastpath",
            )
            self._completion = self._completion_state(
                state="complete",
                finish_reason="stop",
                generation_mode="conversation_fastpath",
            )
            self._core = self._core_state()
        return "succeeded", "complete", 100

    def _run_grounded_answer(
        self,
        turn_id: str,
        user_sequence: int,
        question: str,
        answer: str,
    ) -> TurnFinish | None:
        """Publish a verified Fact-book answer without acquiring the model."""

        with self._condition:
            if self._active_turn != turn_id:
                return
            self._core = self._core_state(active=("factbook",))
            self._transition_locked("executing", "factbook", 50)
        if self._cancel_event.is_set():
            raise GenerationCancelled("fact lookup cancelled")
        response, clipped = self._clip_response(answer.strip())
        quality = inspect_response(response, final=True)
        if clipped or quality.recommended_action is not QualityAction.ACCEPT:
            raise ResponseQualityError("Runtime Fact-book produced an unsafe answer")
        self.conversations.commit_assistant_turn(
            self.session_id, user_sequence, response
        )
        with self._condition:
            # Long status/identity prose remains visible and auditable in the
            # UI but must not steer later open conversation. A short grounded
            # collaboration answer is useful dialogue context, however, so it
            # stays available for references such as "첫 번째부터 해보자".
            # Fact-book status markers provide an explicit, bounded boundary.
            if "Fact-book:" in response or len(response) > 512:
                self._model_excluded_exchanges.append((question, response))
            self._append_message(
                "assistant",
                response,
                streaming=False,
                finish_reason="stop",
                continuations=0,
                truncated=False,
                generated_tokens=0,
                generation_mode="factbook",
            )
            self._completion = self._completion_state(
                state="complete",
                finish_reason="stop",
                generation_mode="factbook",
            )
            self._core = self._core_state()
        return "succeeded", "complete", 100

    def _previous_assistant_content(self) -> str:
        messages = self.conversations.snapshot(self.session_id).as_messages()
        for item in reversed(messages):
            if item["role"] == "assistant":
                return item["content"]
        return ""

    def _previous_fast_path_exchange(self) -> tuple[str | None, str | None]:
        """Return only the immediately preceding completed fast-path exchange."""

        with self._condition:
            messages = list(self._messages)
        if messages and messages[-1].get("role") == "user":
            messages.pop()
        if len(messages) < 2:
            return None, None
        user, assistant = messages[-2:]
        if (
            user.get("role") != "user"
            or assistant.get("role") != "assistant"
            or assistant.get("generation_mode") != "conversation_fastpath"
        ):
            return None, None
        return str(user.get("content", "")), str(assistant.get("content", ""))

    @staticmethod
    def _substantive_sentence_keys(text: str) -> frozenset[str]:
        keys: set[str] = set()
        for match in _SENTENCE_UNIT_RE.finditer(text[:MAX_AGENT_RESPONSE_CHARS]):
            key = re.sub(r"\s+", " ", match.group(0)).strip()
            key = re.sub(r"^(?:[-*+]\s+|\d{1,4}[.)]\s+)", "", key)
            key = key.rstrip(" .!?。！？").casefold()
            if len(key) >= 24:
                keys.add(key)
        return frozenset(keys)

    def _has_cross_turn_sentence_echo(self, response: str) -> bool:
        current = self._substantive_sentence_keys(response)
        if not current:
            return False
        messages = self.conversations.snapshot(self.session_id).as_messages()
        excluded_answers = {
            answer for _question, answer in self._model_excluded_exchanges
        }
        for item in messages:
            if (
                item["role"] != "assistant"
                or _is_quality_fallback(item["content"])
                or item["content"] in excluded_answers
            ):
                continue
            overlap = current & self._substantive_sentence_keys(item["content"])
            if len(overlap) >= 2:
                return True
            if len(current) == 1 and overlap and len(next(iter(current))) >= 96:
                return True
        return False

    def _build_prompt(
        self,
        *,
        resume_truncated: bool = False,
        partial_assistant: str | None = None,
        isolate_history: bool = False,
    ) -> str:
        messages = self._model_messages()
        if isolate_history and messages and messages[-1]["role"] == "user":
            messages = [messages[-1]]
        if resume_truncated and messages and messages[-1]["role"] == "user":
            messages[-1] = {"role": "user", "content": CONTINUATION_DIRECTIVE}
        return self._render_bounded_prompt(
            messages,
            partial_assistant=partial_assistant,
        )

    def _build_continuation_prompt(
        self,
        response: str,
        *,
        isolate_history: bool = False,
    ) -> str:
        """Continue the same open model turn without adding transcript roles."""

        messages = self._model_messages()
        if isolate_history and messages and messages[-1]["role"] == "user":
            messages = [messages[-1]]
        return self._render_bounded_prompt(
            messages,
            partial_assistant=response,
        )

    def _build_quality_repair_prompt(
        self,
        message: str,
        *,
        issue_codes: tuple[str, ...] = (),
        isolate_history: bool = False,
        partial_assistant: str | None = None,
    ) -> str:
        """Retry from the user intent without re-feeding the failed candidate."""

        messages = self._model_messages()
        current = {"role": "user", "content": message}
        if messages and messages[-1]["role"] == "user":
            current = dict(messages[-1])
        if isolate_history:
            messages = [current]
        elif messages and messages[-1]["role"] == "user":
            messages[-1] = current
        else:
            messages.append(current)

        code_set = frozenset(issue_codes)
        directions = [QUALITY_REPAIR_DIRECTIVE]
        if "response_contract" in code_set:
            exact = requested_exact_sentence_count(message)
            categories = requested_category_counts(message)
            if categories is not None:
                positive_count, negative_count = categories
                directions.append(
                    f"정확히 장점 {positive_count}개와 한계 {negative_count}개를 "
                    "각각 별도 문장으로 쓰고, 각 문장에 '장점' 또는 '한계'를 표시하세요."
                )
                directions.append(
                    "제목·목록·HTML 표기 없이 짧은 평문 문장만 작성하세요."
                )
            elif exact is not None:
                directions.append(
                    f"서론이나 맺음말 없이 정확히 {exact}개의 완결된 문장만 작성하세요."
                )
            else:
                directions.append("사용자가 지정한 개수와 형식을 정확히 지키세요.")
        if "cross_turn_echo" in code_set:
            directions.append(
                "이전 답변의 문장을 재사용하지 말고 새 표현으로 답하세요."
            )
        if code_set & {
            QualityCode.TEMPLATE_REPETITION.value,
            QualityCode.LOW_INFORMATION_REPETITION.value,
            "token_repetition",
            "near_duplicate",
        }:
            directions.append(
                "같은 문장이나 표현을 반복하지 말고 서로 다른 핵심을 한 번씩만 답하세요."
            )
        if code_set & {
            QualityCode.INCOMPLETE_KOREAN_CLAUSE.value,
            "empty_candidate",
            "decode_deadline",
        }:
            directions.append("각 문장을 자연스러운 서술어로 끝까지 완결하세요.")
        if "insufficient_detail" in code_set:
            directions.append(
                "요청한 범위를 빠뜨리지 말고 서로 다른 핵심 내용을 충분히 설명하세요."
            )
            topics = request_topic_terms(message)
            if topics:
                directions.append(
                    "질문의 핵심 용어를 직접 유지하세요: " + ", ".join(topics) + "."
                )

        messages[-1] = {
            "role": "user",
            "content": current["content"] + "\n\n" + " ".join(directions),
        }
        return self._render_bounded_prompt(
            messages,
            partial_assistant=partial_assistant,
            system_prompt=self.system_prompt,
        )

    def _model_messages(self) -> list[dict[str, str]]:
        """Exclude deterministic and failed exchanges from Gemma conditioning.

        Fact-book and quality-fallback answers remain visible in the UI and
        audit history.  Feeding either prose block back to Gemma made later
        conversational turns copy product boilerplate or safety text.  Removing
        each complete pair preserves role alternation and keeps only genuine
        model conversation in the prompt.
        """

        messages = list(self.conversations.snapshot(self.session_id).as_messages())
        excluded = set(self._model_excluded_exchanges)
        retained: list[dict[str, str]] = []
        index = 0
        while index < len(messages):
            item = messages[index]
            if (
                item["role"] == "user"
                and index + 1 < len(messages)
                and messages[index + 1]["role"] == "assistant"
                and (item["content"], messages[index + 1]["content"]) in excluded
            ):
                index += 2
                continue
            if (
                item["role"] == "assistant"
                and _is_quality_fallback(item["content"])
                and retained
                and retained[-1]["role"] == "user"
            ):
                retained.pop()
                index += 1
                continue
            retained.append(item)
            index += 1
        return retained

    def _render_bounded_prompt(
        self,
        messages: list[dict[str, str]],
        *,
        partial_assistant: str | None = None,
        system_prompt: str | None = None,
    ) -> str:
        """Drop only oldest complete exchanges until the model input fits."""

        retained = list(messages)
        tokenizer = self.model_service.tokenizer
        token_limit = min(
            int(getattr(self.model_service, "max_input_tokens", 4_096)),
            INTERACTIVE_MAX_INPUT_TOKENS,
        )
        prompt_context = self.system_prompt if system_prompt is None else system_prompt
        while True:
            rendered = render_chat_prompt(
                tokenizer,
                prompt_context,
                retained,
                partial_assistant=partial_assistant,
            )
            if not callable(tokenizer):
                return rendered
            try:
                encoded = tokenizer(rendered, return_tensors="pt", truncation=False)
                input_ids = torch.as_tensor(encoded["input_ids"])
            except (KeyError, TypeError, ValueError):
                return rendered
            if input_ids.ndim == 2 and input_ids.shape[0] == 1:
                if input_ids.shape[1] <= token_limit:
                    return rendered
            if (
                len(retained) >= 3
                and retained[0].get("role") == "user"
                and retained[1].get("role") == "assistant"
            ):
                retained = retained[2:]
                continue
            raise ModelServiceError(
                f"the current chat turn exceeds the {token_limit:,}-token "
                "interactive context bound"
            )

    def _generation_stream(
        self,
        prompt: str,
        max_new_tokens: int,
        *,
        decode_mode: str,
        timeout_seconds: float,
        sampling_seed: int,
    ) -> Any:
        """Call one backend generation API without masking backend TypeErrors."""

        generator = self.model_service.iter_generate_tokens
        try:
            parameters = signature(generator).parameters
        except (TypeError, ValueError):
            parameters = {}
        variadic = any(
            parameter.kind is Parameter.VAR_KEYWORD for parameter in parameters.values()
        )

        def supported(name: str) -> bool:
            return variadic or name in parameters

        kwargs: dict[str, Any] = {"max_new_tokens": max_new_tokens}
        if supported("stop_token_ids"):
            kwargs["stop_token_ids"] = self._response_stop_ids
        if supported("conversation_id"):
            kwargs["conversation_id"] = self.session_id
        if supported("decode_mode"):
            kwargs["decode_mode"] = decode_mode
        if supported("sampling_seed"):
            kwargs["sampling_seed"] = sampling_seed
        if supported("timeout"):
            kwargs["timeout"] = min(
                self.max_decode_seconds,
                max(0.01, float(timeout_seconds)),
            )
        if supported("total_timeout"):
            kwargs["total_timeout"] = min(
                self.max_decode_seconds,
                max(0.01, float(timeout_seconds)),
            )

        deadline_chunk = _DecodeDeadlineChunk(
            token_ids=torch.empty(0, dtype=torch.int64)
        )
        try:
            stream = generator(prompt, **kwargs)
        except TimeoutError:
            return iter((deadline_chunk,))

        def guarded_stream() -> Any:
            try:
                yield from stream
            except TimeoutError:
                yield deadline_chunk

        return guarded_stream()

    def _sampling_seed(
        self,
        user_sequence: int,
        generation_attempt: int,
        *,
        repair_message: str | None = None,
    ) -> int:
        """Derive one deterministic but attempt-distinct signed-63-bit seed."""

        if repair_message is None:
            material = (
                f"{self.session_id}:{int(user_sequence)}:{int(generation_attempt)}"
            ).encode("ascii")
        else:
            material = (
                f"repair:{int(user_sequence)}:{int(generation_attempt)}:"
                + repair_message[:MAX_AGENT_INPUT_CHARS]
            ).encode("utf-8")
        return int.from_bytes(hashlib.sha256(material).digest()[:8], "big") & (
            (1 << 63) - 1
        )

    @staticmethod
    def _candidate_sentences(text: str) -> list[str]:
        """Extract bounded, complete public sentences from one model attempt."""

        result: list[str] = []
        for match in _SENTENCE_UNIT_RE.finditer(text[:MAX_AGENT_RESPONSE_CHARS]):
            sentence = " ".join(match.group(0).split()).strip()
            sentence = re.sub(
                r"^(?:[-*+]\s+|\d{1,4}(?:[.)]|단계(?:는|로|:)?)\s*)",
                "",
                sentence,
            )
            if not sentence.endswith((".", "!", "?", "。", "！", "？")):
                continue
            if sentence.endswith(
                (
                    "설명하겠습니다.",
                    "답변하겠습니다.",
                    "다음과 같습니다.",
                    "완결된 문장으로 작성해야 합니다.",
                )
            ):
                continue
            if (
                inspect_response(sentence, final=True).recommended_action
                is not QualityAction.ACCEPT
            ):
                continue
            result.append(sentence)
            if len(result) >= 32:
                break
        return result

    @classmethod
    def _merge_distinct_sentences(
        cls,
        existing: list[str],
        candidate: str,
    ) -> list[str]:
        """Accumulate distinct observed sentences without inventing content."""

        merged = list(existing[:32])
        keys = {re.sub(r"\s+", " ", item).casefold() for item in merged}
        for sentence in cls._candidate_sentences(candidate):
            key = re.sub(r"\s+", " ", sentence).casefold()
            if key in keys:
                continue
            if any(
                has_near_duplicate_sentences(previous + " " + sentence)
                for previous in merged
            ):
                continue
            merged.append(sentence)
            keys.add(key)
            if len(merged) >= 32:
                break
        return merged

    @staticmethod
    def _response_adequate_for_request(message: str, response: str) -> bool:
        """Reject generic one-line salvage for an explicitly broad request."""

        candidate = response.strip()
        if not candidate or not response_contract_satisfied(message, candidate):
            return False
        if not response_avoids_unsolicited_subjects(message, candidate):
            return False
        if not response_avoids_unsolicited_self_intro(message, candidate):
            return False
        if not response_avoids_dangling_sentence_start(candidate):
            return False
        if not response_avoids_generic_outline(message, candidate):
            return False
        if not response_avoids_prompt_echo(message, candidate):
            return False
        if (
            inspect_response(candidate, final=True).recommended_action
            is not QualityAction.ACCEPT
        ):
            return False
        exact = requested_exact_sentence_count(message)
        exact_items = requested_exact_item_count(message)
        categories = requested_category_counts(message)
        explicit_formal_request = _FORMAL_INSTRUCTION_RE.search(message) is not None
        if (
            exact_items is None
            and (exact is not None or explicit_formal_request)
            and not response_topically_anchored(message, candidate)
            and not (
                categories is not None
                and response_preserves_category_subject(message, candidate)
            )
        ):
            return False
        if (
            categories is None
            and (exact or 0) >= 3
            and not response_preserves_distinctive_topic(
                message,
                candidate,
            )
        ):
            return False
        if exact_items is not None and not response_preserves_distinctive_topic(
            message,
            candidate,
        ):
            return False
        if (exact or 0) >= 2 and has_near_duplicate_sentences(candidate):
            return False
        if exact is not None or exact_items is not None:
            return True

        lowered = message.casefold()
        detail_score = sum(term in lowered for term in _DETAILED_INTENT_TERMS)
        broad_request = detail_score >= 2 or len(message) >= 180
        if not broad_request:
            return True

        completed_sentences = len(
            re.findall(r"[.!?。！？]+(?=\s|$)", candidate[:MAX_AGENT_RESPONSE_CHARS])
        )
        structured_items = len(
            re.findall(
                r"(?m)^\s*(?:[-*+]\s+|\d{1,4}[.)]\s+).{8,}$",
                candidate[:MAX_AGENT_RESPONSE_CHARS],
            )
        )
        return (
            len(candidate) >= 32
            and max(
                completed_sentences,
                structured_items,
            )
            >= 2
        )

    def _decode(self, chunks: list[torch.Tensor]) -> tuple[str, bool]:
        if not chunks:
            return "", False
        tokens = torch.cat(chunks)
        public_tokens, repeated = truncate_repeated_tokens(tokens)
        return (
            decode_response(
                self.model_service.tokenizer,
                public_tokens,
                self._response_stop_ids,
            ),
            repeated,
        )

    def _response_budget(
        self, message: str, *, resume_truncated: bool
    ) -> ResponseBudget:
        lowered = message.casefold()
        if not resume_truncated and _CONCISE_INTENT_RE.search(message) is not None:
            concise = min(self.max_new_tokens, DEFAULT_CONCISE_RESPONSE_TOKENS)
            return ResponseBudget(
                first_request=max(1, concise),
                total=min(self.max_total_new_tokens, max(1, concise * 3)),
                max_continuations=0,
            )
        detail_score = sum(term in lowered for term in _DETAILED_INTENT_TERMS)
        if resume_truncated or detail_score >= 2 or len(message) >= 600:
            first = self.max_new_tokens
        elif detail_score == 1 or len(message) >= 180:
            first = min(self.max_new_tokens, DEFAULT_DETAILED_RESPONSE_TOKENS)
        else:
            first = min(self.max_new_tokens, DEFAULT_SHORT_RESPONSE_TOKENS)
        return ResponseBudget(
            first_request=max(1, first),
            total=self.max_total_new_tokens,
            max_continuations=self.max_continuations,
        )

    @staticmethod
    def _chunk_finish_reason(chunk: Any) -> str:
        """Read v2 reasons while preserving injected legacy backend behavior."""

        value = getattr(chunk, "finish_reason", None)
        if value in {"stop", "length", "cancelled"}:
            return value
        return "stop"

    @staticmethod
    def _strip_leading_user_echo(text: str, message: str) -> tuple[str, bool]:
        """Remove one exact whitespace-normalized restatement of the user turn."""

        words = message.split()
        if len(words) < 4 or len(message.strip()) < 12:
            return text, False
        pattern = re.compile(
            r"\A\s*" + r"\s+".join(re.escape(word) for word in words) + r"\s*"
        )
        matched = pattern.match(text)
        if matched is None:
            return text, False
        return text[matched.end() :].lstrip(), True

    @staticmethod
    def _clean_model_text(text: str) -> tuple[str, bool]:
        """Remove model headers and stop at the next generated chat role."""

        normalized = text.replace("\r\n", "\n").replace("\r", "\n")
        leading_quote = re.match(r"\A\s*([“‘])\s*", normalized)
        if leading_quote is not None:
            closing_quote = "”" if leading_quote.group(1) == "“" else "’"
            if closing_quote not in normalized[leading_quote.end() :]:
                normalized = normalized[leading_quote.end() :].lstrip()
        normalized = _LEADING_ASSISTANT_RE.sub("", normalized, count=1)
        turn_token = _TURN_TOKEN_RE.search(normalized)
        turn_start = _TURN_START_BOUNDARY_RE.search(normalized)
        role_marker = _ROLE_BOUNDARY_RE.search(normalized)
        reserved = _RESERVED_OUTPUT_RE.search(normalized)
        factbook_echo = _FACTBOOK_ECHO_RE.search(normalized)
        repair_echo = _QUALITY_REPAIR_ECHO_RE.search(normalized)
        folded = unicodedata.normalize("NFKC", normalized)
        folded_role = _ROLE_BOUNDARY_RE.search(folded)
        folded_leading = _LEADING_ASSISTANT_RE.search(folded)
        boundaries = [
            match.start()
            for match in (
                turn_token,
                turn_start,
                role_marker,
                reserved,
                factbook_echo,
                repair_echo,
                folded_role,
                folded_leading,
            )
            if match is not None
        ]
        if boundaries:
            return normalized[: min(boundaries)].rstrip(), True
        return normalized.rstrip(), False

    @staticmethod
    def _merge_response(existing: str, segment: str) -> str:
        base = existing.rstrip()
        leading_space = bool(segment[:1].isspace())
        addition = segment.lstrip()
        if not base:
            return addition
        if not addition or base.endswith(addition):
            return base
        if addition.startswith(base):
            addition = addition[len(base) :].lstrip()
            if not addition:
                return base
        overlap_limit = min(len(base), len(addition), 2_048)
        overlap = 0
        for size in range(overlap_limit, 7, -1):
            if base.endswith(addition[:size]):
                overlap = size
                break
        addition = addition[overlap:].lstrip()
        if not addition:
            return base
        if base.endswith((".", "!", "?", "。", "다.")):
            separator = "\n\n"
        elif leading_space:
            separator = " "
        else:
            separator = ""
        return base + separator + addition

    @staticmethod
    def _trim_repeated_text(text: str) -> tuple[str, bool]:
        """Remove only exact, adjacent sentence-block cycles.

        Long blocks are cut on their second copy; short blocks need three
        copies. Code fences, tables, and explicit list items are excluded so
        meaningful structured repetition is preserved.
        """

        if len(text) < 6:
            return text, False
        units: list[tuple[str, int, int]] = []
        for match in _SENTENCE_UNIT_RE.finditer(text):
            normalized = re.sub(r"\s+", " ", match.group(0)).strip()
            normalized = normalized.rstrip(" .!?。！？")
            if normalized:
                units.append((normalized, match.start(), match.end()))
        if len(units) < 2:
            return text, False
        maximum = min(16, len(units) // 2)
        for block_size in range(maximum, 0, -1):
            for first in range(0, len(units) - block_size * 2 + 1):
                pattern = tuple(unit[0] for unit in units[first : first + block_size])
                pattern_chars = len(" ".join(pattern))
                copies = 2 if pattern_chars >= 24 else 3
                if first + block_size * copies > len(units):
                    continue
                if any(
                    tuple(
                        unit[0]
                        for unit in units[
                            first + copy * block_size : first + (copy + 1) * block_size
                        ]
                    )
                    != pattern
                    for copy in range(1, copies)
                ):
                    continue
                raw_block = text[units[first][1] : units[first + block_size - 1][2]]
                if _STRUCTURED_REPEAT_RE.search(raw_block):
                    continue
                second_start = units[first + block_size][1]
                return text[:second_start].rstrip(), True
        return text, False

    @staticmethod
    def _ensure_terminal_punctuation(text: str) -> str:
        """Add one period only to a complete Korean predicate missing punctuation."""

        stripped = text.rstrip()
        if (
            stripped
            and not stripped.endswith((".", "!", "?", "。", "！", "？"))
            and re.search(r"[가-힣](?:다|요)$", stripped) is not None
        ):
            return stripped + "."
        return stripped

    @staticmethod
    def _complete_prefix(text: str, cutoff: int | None = None) -> str:
        """Return only a quality-clean prefix ending at an observed boundary."""
        return salvage_complete_prefix(text, cutoff=cutoff)

    @staticmethod
    def _quality_failure_diagnostic(
        candidate: str,
        codes: set[str],
        *,
        attempts: int,
        generated_tokens: int,
    ) -> str:
        """Build a bounded diagnostic without retaining generated user content."""

        digest = hashlib.sha256(candidate.encode("utf-8")).hexdigest()[:16]
        bounded_codes = sorted(
            {
                code
                for code in codes
                if isinstance(code, str)
                and 1 <= len(code) <= 40
                and re.fullmatch(r"[a-z0-9_]+", code) is not None
            }
        )[:12]
        code_text = ",".join(bounded_codes) or "unspecified"
        return (
            "quality_gate_v2;"
            f"codes={code_text};candidate_sha256={digest};"
            f"candidate_chars={min(len(candidate), MAX_AGENT_RESPONSE_CHARS)};"
            f"attempts={max(0, min(attempts, HARD_MAX_GENERATION_ATTEMPTS))};"
            f"generated_tokens={max(0, min(generated_tokens, HARD_MAX_TOTAL_TOKENS))}"
        )[:512]

    @staticmethod
    def _close_repetition_boundary(text: str) -> str:
        """Finish a token-cycle cut without inventing new answer content."""

        value = text.rstrip()
        if not value or value.endswith((".", "!", "?", "。", "！", "？")):
            return value
        if value.endswith((",", ";", ":", "，", "；", "：", "-", "—")):
            boundary = max(
                value.rfind(mark) for mark in (".", "!", "?", "。", "！", "？")
            )
            if boundary >= 0 and len(value) - boundary <= 256:
                return value[: boundary + 1].rstrip()
        return value + "."

    @staticmethod
    def _clip_response(text: str) -> tuple[str, bool]:
        if len(text) <= MAX_AGENT_RESPONSE_CHARS:
            return text, False
        return text[: MAX_AGENT_RESPONSE_CHARS - 1].rstrip() + "…", True

    def _finish_cancelled(self, turn_id: str, user_sequence: int) -> None:
        self.conversations.abort_user_turn(self.session_id, user_sequence)
        with self._condition:
            if self._active_turn != turn_id:
                return
            self._remove_streaming_message()
            self._append_message("system", "요청이 안전하게 중단되었습니다.")
            self._completion = self._completion_state(
                state="cancelled", finish_reason="cancelled"
            )
            self._core = self._core_state()
            self._finish_locked(turn_id, "cancelled", "cancelled", self._progress)

    def _finish_failed(
        self, turn_id: str, user_sequence: int, exc: BaseException
    ) -> None:
        self.conversations.abort_user_turn(self.session_id, user_sequence)
        code = type(exc).__name__
        message = str(exc)[:512] or "agent turn failed"
        if self.failure_sink is not None:
            try:
                self.failure_sink(code, message)
            except Exception:
                pass
        with self._condition:
            if self._active_turn != turn_id:
                return
            self._remove_streaming_message()
            self._append_message("system", f"요청 실패: {message}")
            self._error = {"code": code, "message": message}
            self._completion = self._completion_state(
                state="failed", finish_reason="error"
            )
            self._core = self._core_state()
            self._finish_locked(turn_id, "failed", "failed", self._progress)

    def _finish_locked(
        self, turn_id: str, status: str, stage: str, progress: int
    ) -> None:
        if self._active_turn != turn_id:
            return
        self._active_turn = None
        self._active_user_sequence = None
        self._transition_locked(status, stage, progress)

    def _transition_locked(self, status: str, stage: str, progress: int) -> None:
        self._status = status
        self._stage = stage
        self._progress = max(0, min(100, int(progress)))
        self._sequence += 1
        self._events.append(
            {
                "seq": self._sequence,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "status": status,
                "stage": stage,
                "progress": self._progress,
            }
        )
        self._condition.notify_all()

    def _append_message(
        self,
        role: str,
        content: str,
        *,
        streaming: bool = False,
        artifact: str | None = None,
        finish_reason: str | None = None,
        continuations: int = 0,
        truncated: bool = False,
        generated_tokens: int = 0,
        generation_mode: str | None = None,
    ) -> str:
        with self._condition:
            message_id = secrets.token_hex(8)
            payload: dict[str, Any] = {
                "id": message_id,
                "role": role,
                "content": content[:MAX_AGENT_RESPONSE_CHARS],
                "created_at": datetime.now(timezone.utc).isoformat(),
                "streaming": bool(streaming),
            }
            if role == "assistant":
                if generation_mode not in {
                    None,
                    "cogni_core",
                    "conversation_fastpath",
                    "factbook",
                    "quality_fallback",
                }:
                    raise ValueError("generation_mode is invalid")
                payload.update(
                    {
                        "finish_reason": finish_reason,
                        "continuations": max(0, int(continuations)),
                        "truncated": bool(truncated),
                        "generated_tokens": max(0, int(generated_tokens)),
                        "generation_mode": generation_mode,
                    }
                )
            if artifact is not None:
                payload["artifact"] = artifact
            self._messages.append(payload)
            return message_id

    def _update_message(
        self,
        message_id: str,
        content: str,
        *,
        streaming: bool,
        finish_reason: str | None = None,
        continuations: int = 0,
        truncated: bool = False,
        generated_tokens: int = 0,
        generation_mode: str | None = None,
    ) -> None:
        with self._condition:
            for message in self._messages:
                if message["id"] == message_id:
                    message["content"] = content[:MAX_AGENT_RESPONSE_CHARS]
                    message["streaming"] = bool(streaming)
                    message["finish_reason"] = finish_reason
                    message["continuations"] = max(0, int(continuations))
                    message["truncated"] = bool(truncated)
                    message["generated_tokens"] = max(0, int(generated_tokens))
                    if generation_mode is not None:
                        if generation_mode not in {
                            "cogni_core",
                            "conversation_fastpath",
                            "factbook",
                            "quality_fallback",
                        }:
                            raise ValueError("generation_mode is invalid")
                        message["generation_mode"] = generation_mode
                    return
            raise RuntimeError("streaming message ownership was lost")

    def _remove_streaming_message(self) -> None:
        retained = [message for message in self._messages if not message["streaming"]]
        self._messages.clear()
        self._messages.extend(retained)

    def _resume_requested_locked(self, text: str) -> bool:
        if _CONTINUE_REQUEST_RE.fullmatch(text) is None:
            return False
        for message in reversed(self._messages):
            if message.get("role") == "assistant":
                return message.get("truncated") is True
        return False

    @staticmethod
    def _completion_state(
        *,
        state: str = "idle",
        finish_reason: str | None = None,
        continuations: int = 0,
        truncated: bool = False,
        generated_tokens: int = 0,
        generation_mode: str | None = None,
    ) -> dict[str, Any]:
        if generation_mode not in {
            None,
            "cogni_core",
            "conversation_fastpath",
            "factbook",
            "quality_fallback",
        }:
            raise ValueError("generation_mode is invalid")
        return {
            "state": state,
            "finish_reason": finish_reason,
            "continuations": max(0, int(continuations)),
            "truncated": bool(truncated),
            "generated_tokens": max(0, int(generated_tokens)),
            "generation_mode": generation_mode,
        }

    def _core_state(self, active: tuple[str, ...] = ()) -> dict[str, Any]:
        model_loaded = bool(getattr(self.model_service, "is_running", False))
        if active == ("conversation_fastpath",):
            verdict = "대화 응답"
        elif active == ("factbook",):
            verdict = "Fact-book 사실 응답"
        elif active:
            verdict = "실행 중"
        else:
            verdict = "모델 준비" if model_loaded else "모델 대기"
        return {
            "verdict": verdict,
            "active_modules": list(active),
            "model_loaded": model_loaded,
            "modules": {
                "gemma": "local" if model_loaded else "not_loaded",
                "router": "ready",
                "swarm": "advisory",
                "cts": "ready" if model_loaded else "not_loaded",
                "fast": "gated",
            },
        }

    @staticmethod
    def _validate_message(message: str) -> str:
        if not isinstance(message, str):
            raise TypeError("message must be text")
        text = message.strip()
        if not text or len(text) > MAX_AGENT_INPUT_CHARS:
            raise ValueError("message is empty or exceeds 4,096 characters")
        if any(ord(character) < 32 and character not in "\t\r\n" for character in text):
            raise ValueError("message contains unsupported control characters")
        return text


__all__ = [
    "ACTIVE_AGENT_STATUSES",
    "AgentBusyError",
    "AgentManager",
    "NoActiveAgentTurnError",
    "SYSTEM_PROMPT",
]
