"""Thread-safe product manager for chat, bounded tools, and UI telemetry."""

from __future__ import annotations

from collections import deque
from collections.abc import Callable, Mapping
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
import re
import secrets
from threading import Condition, Event, RLock, Thread
from time import monotonic
from typing import Any, Protocol

import torch

from .conversation import BoundedConversationStore
from .model_service import GenerationCancelled, ModelServiceError
from .prompting import decode_response, render_chat_prompt, stop_token_ids
from .tools import HELP_TEXT, ToolPolicyError, WorkspaceToolExecutor, parse_tool_request


MAX_AGENT_EVENTS = 64
MAX_AGENT_MESSAGES = 32
MAX_AGENT_INPUT_CHARS = 4_096
MAX_AGENT_RESPONSE_CHARS = 8_192
HARD_MAX_REQUEST_TOKENS = 512
HARD_MAX_TOTAL_TOKENS = 1_536
HARD_MAX_CONTINUATIONS = 2
DEFAULT_SHORT_RESPONSE_TOKENS = 256
DEFAULT_DETAILED_RESPONSE_TOKENS = 384
CONTINUATION_RESPONSE_TOKENS = 512
STREAM_RENDER_TOKEN_INTERVAL = 8
STREAM_RENDER_SECONDS = 0.05
ACTIVE_AGENT_STATUSES = {
    "starting",
    "loading",
    "generating",
    "executing",
    "cancelling",
}

SYSTEM_PROMPT = """당신은 Cogni-OS 2.0의 로컬 AI 동료입니다.
모든 처리는 폐쇄망 PC 안에서 수행됩니다. 확인되지 않은 성능·사실·실행 결과를
만들어내지 말고, 실제 도구 실행 결과와 설계 목표를 구분하십시오. 간결하고
정확한 한국어로 답하되 사용자가 다른 언어로 말하면 그 언어를 따르십시오.
임의 셸·네트워크·무검증 소스 수정 권한이 없으며, 작업 모드는 별도 허용 목록을
통해서만 실행된다는 점을 숨기지 마십시오.
답변 본문에 USER:, ASSISTANT:, SYSTEM: 같은 역할 표기를 출력하지 마십시오.
요청 범위가 넓으면 핵심부터 구조화하고, 문장을 끝맺은 뒤 턴 종료 토큰으로
종료하십시오. 길이 경계에서 이어 쓸 때는 앞부분을 반복하지 마십시오."""

CONTINUATION_DIRECTIVE = """직전 답변이 생성 길이 경계에서 중단되었습니다.
이미 작성한 부분을 반복하거나 요약하지 말고 바로 다음 내용부터 이어 쓰십시오.
진행 중이던 문장을 자연스럽게 완성하고 전체 답변을 끝맺으십시오.
USER:, ASSISTANT:, SYSTEM: 같은 역할 표기는 출력하지 마십시오."""

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
    r"(?:<end_of_turn>|<\|end_of_turn\|>|<\|eot_id\|>|<turn\|>)",
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


@dataclass(frozen=True)
class ResponseBudget:
    """A deterministic, bounded decode budget for one user turn."""

    first_request: int
    total: int
    max_continuations: int


class AgentBusyError(RuntimeError):
    pass


class NoActiveAgentTurnError(RuntimeError):
    pass


class GenerationBackend(Protocol):
    tokenizer: Any
    active_request_id: int | None

    def start(self) -> Any: ...

    def iter_generate_tokens(self, prompt: str, *, max_new_tokens: int) -> Any: ...

    def cancel(self, request_id: int | None = None) -> bool: ...

    def stop(self, timeout: float = 10.0) -> None: ...


FailureSink = Callable[[str, str], None]
EvolutionSnapshot = Callable[[], Mapping[str, Any]]
AvailabilityCheck = Callable[[], bool]


class AgentManager:
    """Own one bounded conversational turn and expose immutable UI snapshots."""

    def __init__(
        self,
        model_service: GenerationBackend,
        tool_executor: WorkspaceToolExecutor,
        *,
        session_id: str = "primary",
        max_new_tokens: int = HARD_MAX_REQUEST_TOKENS,
        max_total_new_tokens: int | None = None,
        max_continuations: int = HARD_MAX_CONTINUATIONS,
        system_prompt: str = SYSTEM_PROMPT,
        failure_sink: FailureSink | None = None,
        evolution_snapshot: EvolutionSnapshot | None = None,
        availability_check: AvailabilityCheck | None = None,
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
        if not isinstance(system_prompt, str) or not 1 <= len(system_prompt) <= 8_192:
            raise ValueError("system_prompt must be bounded text")
        self.model_service = model_service
        self.tool_executor = tool_executor
        self.session_id = session_id
        self.max_new_tokens = int(max_new_tokens)
        self.max_total_new_tokens = int(max_total_new_tokens)
        self.max_continuations = int(max_continuations)
        self.system_prompt = system_prompt
        self.failure_sink = failure_sink
        self.evolution_snapshot = evolution_snapshot
        self.availability_check = availability_check
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
            if self.availability_check is not None and not self.availability_check():
                raise AgentBusyError("the GPU is reserved by another system mode")
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
            self._messages.clear()
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
            if mode == "task":
                self._run_task(turn_id, user_sequence, message)
            else:
                self._run_chat(
                    turn_id,
                    user_sequence,
                    message,
                    resume_truncated=resume_truncated,
                )
        except GenerationCancelled:
            self._finish_cancelled(turn_id, user_sequence)
        except BaseException as exc:
            self._finish_failed(turn_id, user_sequence, exc)

    def _run_task(self, turn_id: str, user_sequence: int, message: str) -> None:
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
            self._finish_locked(turn_id, "succeeded", "complete", 100)

    def _run_chat(
        self,
        turn_id: str,
        user_sequence: int,
        message: str,
        *,
        resume_truncated: bool,
    ) -> None:
        with self._condition:
            if self._active_turn != turn_id:
                return
            self._core = self._core_state(active=("router", "swarm", "cts"))
            self._transition_locked("loading", "model", 10)
        self.model_service.start()
        if self._cancel_event.is_set():
            raise GenerationCancelled("cancelled before generation")
        budget = self._response_budget(message, resume_truncated=resume_truncated)
        prompt = self._build_prompt(resume_truncated=resume_truncated)
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
        with self._condition:
            self._core = self._core_state(active=("gemma", "router", "swarm", "cts"))
            self._transition_locked("generating", "decode", 55)
        while True:
            request_tokens: list[torch.Tensor] = []
            request_generated = 0
            request_reason = "stop"
            role_boundary = False
            response_before_request = response
            made_progress = False
            received_tokens = 0
            rendered_tokens = 0
            rendered_at = monotonic()
            for chunk in self._generation_stream(prompt, request_budget):
                if chunk.cancelled or self._cancel_event.is_set():
                    raise GenerationCancelled("generation cancelled")
                request_generated = max(request_generated, int(chunk.generated_total))
                if chunk.token_ids.numel():
                    request_tokens.append(chunk.token_ids)
                    received_tokens += int(chunk.token_ids.numel())
                render_due = bool(request_tokens) and (
                    chunk.final
                    or received_tokens - rendered_tokens
                    >= STREAM_RENDER_TOKEN_INTERVAL
                    or monotonic() - rendered_at >= STREAM_RENDER_SECONDS
                )
                if render_due:
                    segment, segment_role_boundary = self._clean_model_text(
                        self._decode(request_tokens)
                    )
                    role_boundary = role_boundary or segment_role_boundary
                    merged = self._merge_response(response_before_request, segment)
                    response, clipped = self._clip_response(merged)
                    char_truncated = char_truncated or clipped
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

            total_generated += request_generated
            finish_reason = "stop" if role_boundary else request_reason
            remaining = budget.total - total_generated
            can_continue = (
                finish_reason == "length"
                and not role_boundary
                and not char_truncated
                and made_progress
                and continuations < budget.max_continuations
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
            prompt = self._build_continuation_prompt(response)
            with self._condition:
                self._transition_locked("generating", "continuing", self._progress)

        response = response.strip()
        if not response:
            raise ModelServiceError("local model produced an empty response")
        self.conversations.commit_assistant_turn(
            self.session_id, user_sequence, response
        )
        truncated = char_truncated or finish_reason == "length"
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
            )
            self._completion = self._completion_state(
                state=final_stage,
                finish_reason=finish_reason,
                continuations=continuations,
                truncated=truncated,
                generated_tokens=total_generated,
            )
            self._core = self._core_state()
            self._finish_locked(turn_id, "succeeded", final_stage, 100)

    def _build_prompt(self, *, resume_truncated: bool = False) -> str:
        snapshot = self.conversations.snapshot(self.session_id)
        messages = list(snapshot.as_messages())
        if resume_truncated and messages and messages[-1]["role"] == "user":
            messages[-1] = {"role": "user", "content": CONTINUATION_DIRECTIVE}
        return self._render_bounded_prompt(messages)

    def _build_continuation_prompt(self, response: str) -> str:
        """Continue the same open model turn without adding transcript roles."""

        messages = list(self.conversations.snapshot(self.session_id).as_messages())
        return self._render_bounded_prompt(
            messages,
            partial_assistant=response,
        )

    def _render_bounded_prompt(
        self,
        messages: list[dict[str, str]],
        *,
        partial_assistant: str | None = None,
    ) -> str:
        """Drop only oldest complete exchanges until the model input fits."""

        retained = list(messages)
        tokenizer = self.model_service.tokenizer
        token_limit = int(getattr(self.model_service, "max_input_tokens", 4_096))
        while True:
            rendered = render_chat_prompt(
                tokenizer,
                self.system_prompt,
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
                "the current chat turn exceeds the 4,096-token local context bound"
            )

    def _generation_stream(self, prompt: str, max_new_tokens: int) -> Any:
        try:
            return self.model_service.iter_generate_tokens(
                prompt,
                max_new_tokens=max_new_tokens,
                stop_token_ids=self._response_stop_ids,
            )
        except TypeError:
            # Only lightweight injected legacy backends omit the v2 keyword.
            return self.model_service.iter_generate_tokens(
                prompt,
                max_new_tokens=max_new_tokens,
            )

    def _decode(self, chunks: list[torch.Tensor]) -> str:
        if not chunks:
            return ""
        tokens = torch.cat(chunks)
        return decode_response(
            self.model_service.tokenizer,
            tokens,
            self._response_stop_ids,
        )

    def _response_budget(
        self, message: str, *, resume_truncated: bool
    ) -> ResponseBudget:
        lowered = message.casefold()
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
    def _clean_model_text(text: str) -> tuple[str, bool]:
        """Remove model headers and stop at the next generated chat role."""

        normalized = text.replace("\r\n", "\n").replace("\r", "\n")
        normalized = _LEADING_ASSISTANT_RE.sub("", normalized, count=1)
        turn_token = _TURN_TOKEN_RE.search(normalized)
        turn_start = _TURN_START_BOUNDARY_RE.search(normalized)
        role_marker = _ROLE_BOUNDARY_RE.search(normalized)
        reserved = _RESERVED_OUTPUT_RE.search(normalized)
        boundaries = [
            match.start()
            for match in (turn_token, turn_start, role_marker, reserved)
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
        overlap_limit = min(len(base), len(addition), 512)
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
                payload.update(
                    {
                        "finish_reason": finish_reason,
                        "continuations": max(0, int(continuations)),
                        "truncated": bool(truncated),
                        "generated_tokens": max(0, int(generated_tokens)),
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
    ) -> dict[str, Any]:
        return {
            "state": state,
            "finish_reason": finish_reason,
            "continuations": max(0, int(continuations)),
            "truncated": bool(truncated),
            "generated_tokens": max(0, int(generated_tokens)),
        }

    @staticmethod
    def _core_state(active: tuple[str, ...] = ()) -> dict[str, Any]:
        return {
            "verdict": "실행 중" if active else "대기",
            "active_modules": list(active),
            "modules": {
                "gemma": "local",
                "router": "ready",
                "swarm": "advisory",
                "cts": "ready",
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
