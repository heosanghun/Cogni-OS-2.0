"""Thread-safe product manager for chat, bounded tools, and UI telemetry."""

from __future__ import annotations

from collections import deque
from collections.abc import Callable, Mapping
from copy import deepcopy
from datetime import datetime, timezone
import secrets
from threading import Condition, Event, RLock, Thread
from typing import Any, Protocol

import torch

from .conversation import BoundedConversationStore
from .model_service import GenerationCancelled, ModelServiceError
from .tools import HELP_TEXT, ToolPolicyError, WorkspaceToolExecutor, parse_tool_request


MAX_AGENT_EVENTS = 64
MAX_AGENT_MESSAGES = 32
MAX_AGENT_MESSAGE_CHARS = 4_096
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
통해서만 실행된다는 점을 숨기지 마십시오."""


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
        max_new_tokens: int = 192,
        system_prompt: str = SYSTEM_PROMPT,
        failure_sink: FailureSink | None = None,
        evolution_snapshot: EvolutionSnapshot | None = None,
        availability_check: AvailabilityCheck | None = None,
    ) -> None:
        if not callable(getattr(model_service, "iter_generate_tokens", None)):
            raise TypeError("model_service must provide tensor generation")
        if not isinstance(tool_executor, WorkspaceToolExecutor):
            raise TypeError("tool_executor must be WorkspaceToolExecutor")
        if not 1 <= max_new_tokens <= 512:
            raise ValueError("max_new_tokens must be in [1, 512]")
        if not isinstance(system_prompt, str) or not 1 <= len(system_prompt) <= 8_192:
            raise ValueError("system_prompt must be bounded text")
        self.model_service = model_service
        self.tool_executor = tool_executor
        self.session_id = session_id
        self.max_new_tokens = int(max_new_tokens)
        self.system_prompt = system_prompt
        self.failure_sink = failure_sink
        self.evolution_snapshot = evolution_snapshot
        self.availability_check = availability_check
        self.conversations = BoundedConversationStore(
            max_sessions=1,
            max_turns=24,
            max_chars=24_000,
            max_message_chars=MAX_AGENT_MESSAGE_CHARS,
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
            user_sequence = self.conversations.begin_user_turn(self.session_id, text)
            self._active_turn = turn_id
            self._active_user_sequence = user_sequence
            self._cancel_event = Event()
            self._error = None
            self._append_message("user", text)
            self._transition_locked("starting", "accepted", 0)
            thread = Thread(
                target=self._run_turn,
                args=(turn_id, user_sequence, text, mode),
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
        self, turn_id: str, user_sequence: int, message: str, mode: str
    ) -> None:
        try:
            if mode == "task":
                self._run_task(turn_id, user_sequence, message)
            else:
                self._run_chat(turn_id, user_sequence)
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
            self.session_id, user_sequence, output[:MAX_AGENT_MESSAGE_CHARS]
        )
        with self._condition:
            self._append_message(role, output, artifact=artifact)
            self._finish_locked(turn_id, "succeeded", "complete", 100)

    def _run_chat(self, turn_id: str, user_sequence: int) -> None:
        with self._condition:
            if self._active_turn != turn_id:
                return
            self._core = self._core_state(active=("router", "swarm", "cts"))
            self._transition_locked("loading", "model", 10)
        self.model_service.start()
        if self._cancel_event.is_set():
            raise GenerationCancelled("cancelled before generation")
        prompt = self._build_prompt()
        message_id = self._append_message("assistant", "", streaming=True)
        tokens: list[torch.Tensor] = []
        with self._condition:
            self._core = self._core_state(active=("gemma", "router", "swarm", "cts"))
            self._transition_locked("generating", "decode", 55)
        for chunk in self.model_service.iter_generate_tokens(
            prompt, max_new_tokens=self.max_new_tokens
        ):
            if chunk.cancelled or self._cancel_event.is_set():
                raise GenerationCancelled("generation cancelled")
            if chunk.token_ids.numel():
                tokens.append(chunk.token_ids)
                decoded = self._decode(tokens)
                with self._condition:
                    self._update_message(message_id, decoded, streaming=not chunk.final)
                    self._transition_locked(
                        "generating",
                        "decode",
                        min(
                            95,
                            55 + int(chunk.generated_total * 40 / self.max_new_tokens),
                        ),
                    )
        response = self._decode(tokens).strip()
        if not response:
            raise ModelServiceError("local model produced an empty response")
        self.conversations.commit_assistant_turn(
            self.session_id, user_sequence, response[:MAX_AGENT_MESSAGE_CHARS]
        )
        with self._condition:
            self._update_message(message_id, response, streaming=False)
            self._core = self._core_state()
            self._finish_locked(turn_id, "succeeded", "complete", 100)

    def _build_prompt(self) -> str:
        snapshot = self.conversations.snapshot(self.session_id)
        messages = list(snapshot.as_messages())
        tokenizer = self.model_service.tokenizer
        template = getattr(tokenizer, "apply_chat_template", None)
        if callable(template):
            try:
                rendered = template(
                    [{"role": "system", "content": self.system_prompt}, *messages],
                    tokenize=False,
                    add_generation_prompt=True,
                )
                if isinstance(rendered, str) and rendered:
                    return rendered
            except (TypeError, ValueError, KeyError):
                pass
        parts = [f"SYSTEM:\n{self.system_prompt}"]
        for message in messages:
            role = "USER" if message["role"] == "user" else "ASSISTANT"
            parts.append(f"{role}:\n{message['content']}")
        parts.append("ASSISTANT:\n")
        return "\n\n".join(parts)

    def _decode(self, chunks: list[torch.Tensor]) -> str:
        if not chunks:
            return ""
        tokens = torch.cat(chunks)
        text = self.model_service.tokenizer.decode(
            tokens.tolist(),
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        if not isinstance(text, str):
            raise ModelServiceError("tokenizer decode did not return text")
        return text[:MAX_AGENT_MESSAGE_CHARS]

    def _finish_cancelled(self, turn_id: str, user_sequence: int) -> None:
        self.conversations.abort_user_turn(self.session_id, user_sequence)
        with self._condition:
            if self._active_turn != turn_id:
                return
            self._remove_streaming_message()
            self._append_message("system", "요청이 안전하게 중단되었습니다.")
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
    ) -> str:
        message_id = secrets.token_hex(8)
        payload: dict[str, Any] = {
            "id": message_id,
            "role": role,
            "content": content[:MAX_AGENT_MESSAGE_CHARS],
            "created_at": datetime.now(timezone.utc).isoformat(),
            "streaming": bool(streaming),
        }
        if artifact is not None:
            payload["artifact"] = artifact
        self._messages.append(payload)
        return message_id

    def _update_message(
        self, message_id: str, content: str, *, streaming: bool
    ) -> None:
        for message in self._messages:
            if message["id"] == message_id:
                message["content"] = content[:MAX_AGENT_MESSAGE_CHARS]
                message["streaming"] = bool(streaming)
                return
        raise RuntimeError("streaming message ownership was lost")

    def _remove_streaming_message(self) -> None:
        retained = [message for message in self._messages if not message["streaming"]]
        self._messages.clear()
        self._messages.extend(retained)

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
        if not text or len(text) > MAX_AGENT_MESSAGE_CHARS:
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
