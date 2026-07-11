"""Bounded, transactional multi-turn conversation storage."""

from __future__ import annotations

from collections import OrderedDict, deque
from dataclasses import dataclass
from threading import RLock


HARD_MAX_SESSIONS = 128
HARD_MAX_TURNS = 128
HARD_MAX_MESSAGE_CHARS = 32_000
HARD_MAX_CONVERSATION_CHARS = 256_000
MAX_SESSION_ID_CHARS = 64
_ROLES = frozenset({"user", "assistant"})


class ConversationError(RuntimeError):
    """Raised when a turn would violate conversation ordering or bounds."""


@dataclass(frozen=True)
class ConversationTurn:
    sequence: int
    role: str
    text: str


@dataclass(frozen=True)
class ConversationSnapshot:
    session_id: str
    turns: tuple[ConversationTurn, ...]
    total_chars: int
    next_sequence: int

    def as_messages(self) -> tuple[dict[str, str], ...]:
        """Return a fresh web/tokenizer-boundary representation."""

        return tuple({"role": turn.role, "content": turn.text} for turn in self.turns)


@dataclass
class _Conversation:
    turns: deque[ConversationTurn]
    total_chars: int = 0
    next_sequence: int = 1


class BoundedConversationStore:
    """LRU session store with bounded turns, characters, and transactional replies.

    A user turn is opened first and its exact sequence must be supplied when an
    assistant reply is committed.  Cancellation can therefore remove only the
    still-pending user turn and cannot corrupt another request's history.
    """

    def __init__(
        self,
        *,
        max_sessions: int = 16,
        max_turns: int = 32,
        max_chars: int = 32_768,
        max_message_chars: int = 8_192,
    ) -> None:
        if not 1 <= max_sessions <= HARD_MAX_SESSIONS:
            raise ValueError("max_sessions exceeds its hard bound")
        if not 2 <= max_turns <= HARD_MAX_TURNS:
            raise ValueError("max_turns must be in the bounded multi-turn range")
        if not 1 <= max_message_chars <= HARD_MAX_MESSAGE_CHARS:
            raise ValueError("max_message_chars exceeds its hard bound")
        if not 2 * max_message_chars <= max_chars <= HARD_MAX_CONVERSATION_CHARS:
            raise ValueError("max_chars must fit one complete maximum-sized exchange")
        self.max_sessions = int(max_sessions)
        self.max_turns = int(max_turns)
        self.max_chars = int(max_chars)
        self.max_message_chars = int(max_message_chars)
        self._sessions: OrderedDict[str, _Conversation] = OrderedDict()
        self._lock = RLock()

    @property
    def session_ids(self) -> tuple[str, ...]:
        """Return session ids from least to most recently used."""

        with self._lock:
            return tuple(self._sessions)

    def begin_user_turn(self, session_id: str, text: str) -> int:
        session = self._validate_session_id(session_id)
        message = self._validate_text(text)
        with self._lock:
            conversation = self._get_or_create(session)
            if conversation.turns and conversation.turns[-1].role == "user":
                raise ConversationError("the previous user turn is still pending")
            turn = self._append(conversation, "user", message)
            self._trim(conversation)
            return turn.sequence

    def commit_assistant_turn(
        self,
        session_id: str,
        expected_user_sequence: int,
        text: str,
    ) -> ConversationTurn:
        session = self._validate_session_id(session_id)
        message = self._validate_text(text)
        if (
            not isinstance(expected_user_sequence, int)
            or isinstance(expected_user_sequence, bool)
            or expected_user_sequence < 1
        ):
            raise ConversationError("expected user sequence is invalid")
        with self._lock:
            conversation = self._require(session)
            if not conversation.turns:
                raise ConversationError("there is no pending user turn")
            pending = conversation.turns[-1]
            if pending.role != "user" or pending.sequence != expected_user_sequence:
                raise ConversationError("assistant reply does not own the pending turn")
            turn = self._append(conversation, "assistant", message)
            self._trim(conversation)
            return turn

    def abort_user_turn(self, session_id: str, expected_user_sequence: int) -> bool:
        session = self._validate_session_id(session_id)
        with self._lock:
            conversation = self._sessions.get(session)
            if conversation is None or not conversation.turns:
                return False
            pending = conversation.turns[-1]
            if pending.role != "user" or pending.sequence != expected_user_sequence:
                return False
            conversation.turns.pop()
            conversation.total_chars -= len(pending.text)
            self._sessions.move_to_end(session)
            return True

    def snapshot(self, session_id: str) -> ConversationSnapshot:
        session = self._validate_session_id(session_id)
        with self._lock:
            conversation = self._require(session)
            self._sessions.move_to_end(session)
            return ConversationSnapshot(
                session,
                tuple(conversation.turns),
                conversation.total_chars,
                conversation.next_sequence,
            )

    def clear(self, session_id: str) -> bool:
        session = self._validate_session_id(session_id)
        with self._lock:
            return self._sessions.pop(session, None) is not None

    def _get_or_create(self, session_id: str) -> _Conversation:
        conversation = self._sessions.get(session_id)
        if conversation is None:
            if len(self._sessions) >= self.max_sessions:
                self._sessions.popitem(last=False)
            conversation = _Conversation(deque())
            self._sessions[session_id] = conversation
        else:
            self._sessions.move_to_end(session_id)
        return conversation

    def _require(self, session_id: str) -> _Conversation:
        conversation = self._sessions.get(session_id)
        if conversation is None:
            raise ConversationError("conversation session does not exist")
        return conversation

    @staticmethod
    def _append(conversation: _Conversation, role: str, text: str) -> ConversationTurn:
        turn = ConversationTurn(conversation.next_sequence, role, text)
        conversation.next_sequence += 1
        conversation.turns.append(turn)
        conversation.total_chars += len(text)
        return turn

    def _trim(self, conversation: _Conversation) -> None:
        while (
            len(conversation.turns) > self.max_turns
            or conversation.total_chars > self.max_chars
        ):
            removed = conversation.turns.popleft()
            conversation.total_chars -= len(removed.text)
        # A retained conversation must never begin with an orphaned reply.
        if conversation.turns and conversation.turns[0].role == "assistant":
            removed = conversation.turns.popleft()
            conversation.total_chars -= len(removed.text)

    def _validate_text(self, text: str) -> str:
        if not isinstance(text, str):
            raise TypeError("conversation text must be a string")
        if not text or len(text) > self.max_message_chars:
            raise ConversationError("conversation message exceeds its character bound")
        if any(ord(character) < 32 and character not in "\t\r\n" for character in text):
            raise ConversationError("conversation message contains control characters")
        return text

    @staticmethod
    def _validate_session_id(session_id: str) -> str:
        if (
            not isinstance(session_id, str)
            or not 1 <= len(session_id) <= MAX_SESSION_ID_CHARS
            or any(
                character
                not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-"
                for character in session_id
            )
        ):
            raise ConversationError("session id is invalid")
        return session_id


__all__ = [
    "BoundedConversationStore",
    "ConversationError",
    "ConversationSnapshot",
    "ConversationTurn",
    "HARD_MAX_CONVERSATION_CHARS",
    "HARD_MAX_MESSAGE_CHARS",
    "HARD_MAX_SESSIONS",
    "HARD_MAX_TURNS",
]
