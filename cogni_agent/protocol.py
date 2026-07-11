"""Tensor-only IPC protocol for the resident local generation worker.

Natural language is permitted at the web/control boundary.  It must be
tokenized before this protocol is used; no Python string, JSON document, or
arbitrary object crosses the model-process boundary.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias

import torch
from torch import Tensor


PROTOCOL_VERSION = 2

OP_GENERATE = 1
OP_STOP = 2

STATUS_OK = 0
STATUS_CANCELLED = 1
STATUS_INVALID_REQUEST = 2
STATUS_MODEL_ERROR = 3
STATUS_BASE_MUTATED = 4

# Terminal generation cause.  This travels in the fourth tensor so the
# controller never confuses a token-budget cut with a real model stop.
FINISH_NONE = 0
FINISH_STOP = 1
FINISH_LENGTH = 2
FINISH_CANCELLED = 3
FINISH_ERROR = 4

HARD_MAX_INPUT_TOKENS = 8_192
HARD_MAX_NEW_TOKENS = 2_048
HARD_MAX_STOP_TOKENS = 128

TensorMessage: TypeAlias = tuple[Tensor, Tensor, Tensor, Tensor]


class TensorProtocolError(ValueError):
    """Raised when an IPC frame violates the closed tensor schema."""


@dataclass(frozen=True)
class GenerationRequest:
    request_id: int
    input_ids: Tensor
    attention_mask: Tensor
    stop_token_ids: Tensor
    max_new_tokens: int


@dataclass(frozen=True)
class ResponseFrame:
    request_id: int
    status: int
    token_ids: Tensor
    generated_total: int
    final: bool
    finish_reason: int


def _cpu_i64(value: Tensor, name: str, *, ndim: int) -> Tensor:
    if not isinstance(value, Tensor):
        raise TensorProtocolError(f"{name} must be a tensor")
    if value.device.type != "cpu":
        raise TensorProtocolError(f"{name} must remain on CPU across IPC")
    if value.dtype != torch.int64 or value.ndim != ndim:
        raise TensorProtocolError(f"{name} must be a rank-{ndim} int64 tensor")
    if not value.is_contiguous():
        raise TensorProtocolError(f"{name} must be contiguous")
    return value


def _empty() -> Tensor:
    return torch.empty(0, dtype=torch.int64)


def make_generate_request(
    request_id: int,
    input_ids: Tensor,
    attention_mask: Tensor | None,
    *,
    max_new_tokens: int,
    stop_token_ids: Tensor | None = None,
) -> TensorMessage:
    if (
        not isinstance(request_id, int)
        or isinstance(request_id, bool)
        or request_id < 1
    ):
        raise TensorProtocolError("request_id must be a positive integer")
    if (
        not isinstance(max_new_tokens, int)
        or isinstance(max_new_tokens, bool)
        or not 1 <= max_new_tokens <= HARD_MAX_NEW_TOKENS
    ):
        raise TensorProtocolError("max_new_tokens exceeds the hard protocol bound")
    ids = _cpu_i64(input_ids, "input_ids", ndim=2)
    if ids.shape[0] != 1 or not 1 <= ids.shape[1] <= HARD_MAX_INPUT_TOKENS:
        raise TensorProtocolError("input_ids must have bounded batch-one shape")
    if bool((ids < 0).any()):
        raise TensorProtocolError("input_ids cannot contain negative token ids")
    mask = (
        torch.ones_like(ids)
        if attention_mask is None
        else _cpu_i64(attention_mask, "attention_mask", ndim=2)
    )
    if mask.shape != ids.shape or bool(((mask != 0) & (mask != 1)).any()):
        raise TensorProtocolError("attention_mask must be binary and match input_ids")
    stop = (
        _empty()
        if stop_token_ids is None
        else _cpu_i64(stop_token_ids, "stop_token_ids", ndim=1)
    )
    if stop.numel() > HARD_MAX_STOP_TOKENS or bool((stop < 0).any()):
        raise TensorProtocolError("stop token ids exceed their hard bound")
    header = torch.tensor(
        [PROTOCOL_VERSION, OP_GENERATE, request_id, max_new_tokens],
        dtype=torch.int64,
    )
    return header, ids.clone(), mask.clone(), stop.clone()


def make_stop_request() -> TensorMessage:
    header = torch.tensor(
        [PROTOCOL_VERSION, OP_STOP, 0, 0],
        dtype=torch.int64,
    )
    return header, _empty(), _empty(), _empty()


def parse_request(message: object) -> GenerationRequest | None:
    if not isinstance(message, tuple) or len(message) != 4:
        raise TensorProtocolError("request must contain exactly four tensors")
    header = _cpu_i64(message[0], "request header", ndim=1)
    if header.shape != (4,):
        raise TensorProtocolError("request header shape is invalid")
    version, opcode, request_id, max_new_tokens = map(int, header.tolist())
    if version != PROTOCOL_VERSION:
        raise TensorProtocolError("request protocol version is unsupported")
    if opcode == OP_STOP:
        for index, value in enumerate(message[1:], start=1):
            tensor = _cpu_i64(value, f"stop field {index}", ndim=1)
            if tensor.numel():
                raise TensorProtocolError("stop request fields must be empty")
        if request_id != 0 or max_new_tokens != 0:
            raise TensorProtocolError("stop request header is invalid")
        return None
    if opcode != OP_GENERATE:
        raise TensorProtocolError("request opcode is unsupported")
    # Reuse the constructor as the single source of all request bounds.
    rebuilt = make_generate_request(
        request_id,
        message[1],
        message[2],
        max_new_tokens=max_new_tokens,
        stop_token_ids=message[3],
    )
    return GenerationRequest(
        request_id=request_id,
        input_ids=rebuilt[1],
        attention_mask=rebuilt[2],
        stop_token_ids=rebuilt[3],
        max_new_tokens=max_new_tokens,
    )


def make_response(
    request_id: int,
    status: int,
    token_ids: Tensor | None = None,
    *,
    generated_total: int = 0,
    final: bool,
    finish_reason: int | None = None,
) -> TensorMessage:
    if (
        not isinstance(request_id, int)
        or isinstance(request_id, bool)
        or request_id < 0
    ):
        raise TensorProtocolError("response request_id is invalid")
    if status not in {
        STATUS_OK,
        STATUS_CANCELLED,
        STATUS_INVALID_REQUEST,
        STATUS_MODEL_ERROR,
        STATUS_BASE_MUTATED,
    }:
        raise TensorProtocolError("response status is unsupported")
    if (
        not isinstance(generated_total, int)
        or not 0 <= generated_total <= HARD_MAX_NEW_TOKENS
    ):
        raise TensorProtocolError("response token count exceeds its hard bound")
    tokens = _empty() if token_ids is None else _cpu_i64(token_ids, "token_ids", ndim=1)
    if tokens.numel() > HARD_MAX_NEW_TOKENS or bool((tokens < 0).any()):
        raise TensorProtocolError("response tokens exceed their hard bound")
    if not final and status != STATUS_OK:
        raise TensorProtocolError("only successful response chunks may be non-terminal")
    if finish_reason is None:
        if not final:
            finish_reason = FINISH_NONE
        elif status == STATUS_OK:
            finish_reason = FINISH_STOP
        elif status == STATUS_CANCELLED:
            finish_reason = FINISH_CANCELLED
        else:
            finish_reason = FINISH_ERROR
    if finish_reason not in {
        FINISH_NONE,
        FINISH_STOP,
        FINISH_LENGTH,
        FINISH_CANCELLED,
        FINISH_ERROR,
    }:
        raise TensorProtocolError("response finish reason is unsupported")
    if not final and finish_reason != FINISH_NONE:
        raise TensorProtocolError("non-terminal chunks cannot have a finish reason")
    if final and finish_reason == FINISH_NONE:
        raise TensorProtocolError("terminal responses require a finish reason")
    if status == STATUS_OK and final and finish_reason not in {
        FINISH_STOP,
        FINISH_LENGTH,
    }:
        raise TensorProtocolError("successful terminal response has invalid finish reason")
    if status == STATUS_CANCELLED and finish_reason != FINISH_CANCELLED:
        raise TensorProtocolError("cancelled response has invalid finish reason")
    if status not in {STATUS_OK, STATUS_CANCELLED} and finish_reason != FINISH_ERROR:
        raise TensorProtocolError("failed response has invalid finish reason")
    header = torch.tensor(
        [PROTOCOL_VERSION, status, request_id, int(bool(final))],
        dtype=torch.int64,
    )
    counters = torch.tensor([generated_total], dtype=torch.int64)
    reason = torch.tensor([finish_reason], dtype=torch.int64)
    return header, tokens.clone(), counters, reason


def parse_response(message: object) -> ResponseFrame:
    if not isinstance(message, tuple) or len(message) != 4:
        raise TensorProtocolError("response must contain exactly four tensors")
    header = _cpu_i64(message[0], "response header", ndim=1)
    tokens = _cpu_i64(message[1], "response token_ids", ndim=1)
    counters = _cpu_i64(message[2], "response counters", ndim=1)
    reason = _cpu_i64(message[3], "response finish reason", ndim=1)
    if header.shape != (4,) or counters.shape != (1,) or reason.shape != (1,):
        raise TensorProtocolError("response tensor shapes are invalid")
    version, status, request_id, final = map(int, header.tolist())
    if version != PROTOCOL_VERSION or final not in {0, 1}:
        raise TensorProtocolError("response header is invalid")
    # Re-encode to apply status, token, and aggregate bounds consistently.
    checked = make_response(
        request_id,
        status,
        tokens,
        generated_total=int(counters.item()),
        final=bool(final),
        finish_reason=int(reason.item()),
    )
    return ResponseFrame(
        request_id=request_id,
        status=status,
        token_ids=checked[1],
        generated_total=int(checked[2].item()),
        final=bool(final),
        finish_reason=int(checked[3].item()),
    )


__all__ = [
    "GenerationRequest",
    "FINISH_CANCELLED",
    "FINISH_ERROR",
    "FINISH_LENGTH",
    "FINISH_NONE",
    "FINISH_STOP",
    "HARD_MAX_INPUT_TOKENS",
    "HARD_MAX_NEW_TOKENS",
    "HARD_MAX_STOP_TOKENS",
    "OP_GENERATE",
    "OP_STOP",
    "PROTOCOL_VERSION",
    "ResponseFrame",
    "STATUS_BASE_MUTATED",
    "STATUS_CANCELLED",
    "STATUS_INVALID_REQUEST",
    "STATUS_MODEL_ERROR",
    "STATUS_OK",
    "TensorMessage",
    "TensorProtocolError",
    "make_generate_request",
    "make_response",
    "make_stop_request",
    "parse_request",
    "parse_response",
]
