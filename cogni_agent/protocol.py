"""Tensor-only IPC protocol for the resident local generation worker.

Natural language is permitted at the web/control boundary.  It must be
tokenized before this protocol is used; no Python string, JSON document, or
arbitrary object crosses the model-process boundary.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from typing import TypeAlias

import torch
from torch import Tensor


PROTOCOL_VERSION = 3

OP_GENERATE = 1
OP_STOP = 2

STATUS_OK = 0
STATUS_CANCELLED = 1
STATUS_INVALID_REQUEST = 2
STATUS_MODEL_ERROR = 3
STATUS_BASE_MUTATED = 4
STATUS_AUTHORITY_REJECTED = 5

# Terminal generation cause.  This travels in the fourth tensor so the
# controller never confuses a token-budget cut with a real model stop.
FINISH_NONE = 0
FINISH_STOP = 1
FINISH_LENGTH = 2
FINISH_CANCELLED = 3
FINISH_ERROR = 4
# Reserved legacy terminal causes. The product ModelService rejects these
# success frames because a sampled runtime signature cannot attest every base-
# model value after an arbitrary Cogni-Core failure.
FINISH_LKG_STOP = 5
FINISH_LKG_LENGTH = 6

HARD_MAX_INPUT_TOKENS = 8_192
HARD_MAX_NEW_TOKENS = 2_048
HARD_MAX_STOP_TOKENS = 128
DIGEST_BYTES = 32
NO_DEADLINE_NS = 2**63 - 1

# A deterministic compatibility value for callers that do not explicitly
# select a conversation. Production callers hash their bounded conversation
# identifier before it crosses this tensor-only boundary.
DEFAULT_SESSION_DIGEST = torch.tensor(
    list(sha256(b"primary").digest()), dtype=torch.int64
)
ZERO_ARTIFACT_DIGEST = torch.zeros(DIGEST_BYTES, dtype=torch.int64)

TensorMessage: TypeAlias = tuple[Tensor, Tensor, Tensor, Tensor]


class TensorProtocolError(ValueError):
    """Raised when an IPC frame violates the closed tensor schema."""


@dataclass(frozen=True)
class GenerationRequest:
    request_id: int
    job_id: int
    lease_epoch: int
    request_deadline_ns: int
    lease_deadline_ns: int
    artifact_digest: Tensor
    session_digest: Tensor
    input_ids: Tensor
    attention_mask: Tensor
    stop_token_ids: Tensor
    max_new_tokens: int


@dataclass(frozen=True)
class ResponseFrame:
    request_id: int
    job_id: int
    lease_epoch: int
    request_deadline_ns: int
    lease_deadline_ns: int
    artifact_digest: Tensor
    session_digest: Tensor
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


def _digest(value: Tensor | None, name: str, *, default: Tensor) -> Tensor:
    tensor = default if value is None else _cpu_i64(value, name, ndim=1)
    if tensor.shape != (DIGEST_BYTES,) or bool(((tensor < 0) | (tensor > 255)).any()):
        raise TensorProtocolError(f"{name} must contain exactly 32 byte values")
    return tensor.contiguous()


def _positive_i63(value: object, name: str) -> int:
    if (
        not isinstance(value, int)
        or isinstance(value, bool)
        or not 1 <= value <= NO_DEADLINE_NS
    ):
        raise TensorProtocolError(f"{name} must be a positive signed-63-bit integer")
    return int(value)


def _authority_fields(
    request_id: int,
    *,
    job_id: int | None,
    lease_epoch: int,
    request_deadline_ns: int,
    lease_deadline_ns: int,
    artifact_digest: Tensor | None,
    session_digest: Tensor | None,
) -> tuple[int, int, int, int, Tensor, Tensor]:
    job = request_id if job_id is None else _positive_i63(job_id, "job_id")
    if (
        not isinstance(lease_epoch, int)
        or isinstance(lease_epoch, bool)
        or not 0 <= lease_epoch <= NO_DEADLINE_NS
    ):
        raise TensorProtocolError(
            "lease_epoch must be a non-negative signed-63-bit integer"
        )
    request_deadline = _positive_i63(request_deadline_ns, "request_deadline_ns")
    if (
        not isinstance(lease_deadline_ns, int)
        or isinstance(lease_deadline_ns, bool)
        or not 0 <= lease_deadline_ns <= NO_DEADLINE_NS
    ):
        raise TensorProtocolError("lease_deadline_ns is invalid")
    if (lease_epoch == 0) != (lease_deadline_ns == 0):
        raise TensorProtocolError(
            "lease_epoch and lease_deadline_ns must both be zero or both be set"
        )
    artifact = _digest(
        artifact_digest,
        "artifact_digest",
        default=ZERO_ARTIFACT_DIGEST,
    )
    session = _digest(
        session_digest,
        "session_digest",
        default=DEFAULT_SESSION_DIGEST,
    )
    return job, lease_epoch, request_deadline, lease_deadline_ns, artifact, session


def make_generate_request(
    request_id: int,
    input_ids: Tensor,
    attention_mask: Tensor | None,
    *,
    max_new_tokens: int,
    stop_token_ids: Tensor | None = None,
    job_id: int | None = None,
    lease_epoch: int = 0,
    request_deadline_ns: int = NO_DEADLINE_NS,
    lease_deadline_ns: int = 0,
    artifact_digest: Tensor | None = None,
    session_digest: Tensor | None = None,
) -> TensorMessage:
    if (
        not isinstance(request_id, int)
        or isinstance(request_id, bool)
        or not 1 <= request_id <= NO_DEADLINE_NS
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
    job, epoch, request_deadline, lease_deadline, artifact, session = _authority_fields(
        request_id,
        job_id=job_id,
        lease_epoch=lease_epoch,
        request_deadline_ns=request_deadline_ns,
        lease_deadline_ns=lease_deadline_ns,
        artifact_digest=artifact_digest,
        session_digest=session_digest,
    )
    header = torch.tensor(
        [
            PROTOCOL_VERSION,
            OP_GENERATE,
            request_id,
            max_new_tokens,
            job,
            epoch,
            request_deadline,
            lease_deadline,
        ],
        dtype=torch.int64,
    )
    control = torch.cat(
        (
            torch.tensor([stop.numel()], dtype=torch.int64),
            artifact,
            session,
            stop,
        )
    ).contiguous()
    return header, ids.clone(), mask.clone(), control


def make_stop_request() -> TensorMessage:
    header = torch.tensor(
        [PROTOCOL_VERSION, OP_STOP, 0, 0, 0, 0, 0, 0],
        dtype=torch.int64,
    )
    return header, _empty(), _empty(), _empty()


def parse_request(message: object) -> GenerationRequest | None:
    if not isinstance(message, tuple) or len(message) != 4:
        raise TensorProtocolError("request must contain exactly four tensors")
    header = _cpu_i64(message[0], "request header", ndim=1)
    if header.shape != (8,):
        raise TensorProtocolError("request header shape is invalid")
    (
        version,
        opcode,
        request_id,
        max_new_tokens,
        job_id,
        lease_epoch,
        request_deadline_ns,
        lease_deadline_ns,
    ) = map(int, header.tolist())
    if version != PROTOCOL_VERSION:
        raise TensorProtocolError("request protocol version is unsupported")
    if opcode == OP_STOP:
        for index, value in enumerate(message[1:], start=1):
            tensor = _cpu_i64(value, f"stop field {index}", ndim=1)
            if tensor.numel():
                raise TensorProtocolError("stop request fields must be empty")
        if any(
            value != 0
            for value in (
                request_id,
                max_new_tokens,
                job_id,
                lease_epoch,
                request_deadline_ns,
                lease_deadline_ns,
            )
        ):
            raise TensorProtocolError("stop request header is invalid")
        return None
    if opcode != OP_GENERATE:
        raise TensorProtocolError("request opcode is unsupported")
    # Reuse the constructor as the single source of all request bounds.
    control = _cpu_i64(message[3], "request control", ndim=1)
    minimum = 1 + 2 * DIGEST_BYTES
    if control.numel() < minimum:
        raise TensorProtocolError("request control tensor is truncated")
    stop_count = int(control[0].item())
    if (
        not 0 <= stop_count <= HARD_MAX_STOP_TOKENS
        or control.numel() != minimum + stop_count
    ):
        raise TensorProtocolError("request control tensor has an invalid stop count")
    artifact = control[1 : 1 + DIGEST_BYTES]
    session = control[1 + DIGEST_BYTES : minimum]
    stop = control[minimum:]
    rebuilt = make_generate_request(
        request_id,
        message[1],
        message[2],
        max_new_tokens=max_new_tokens,
        stop_token_ids=stop,
        job_id=job_id,
        lease_epoch=lease_epoch,
        request_deadline_ns=request_deadline_ns,
        lease_deadline_ns=lease_deadline_ns,
        artifact_digest=artifact,
        session_digest=session,
    )
    return GenerationRequest(
        request_id=request_id,
        job_id=job_id,
        lease_epoch=lease_epoch,
        request_deadline_ns=request_deadline_ns,
        lease_deadline_ns=lease_deadline_ns,
        artifact_digest=artifact.clone(),
        session_digest=session.clone(),
        input_ids=rebuilt[1],
        attention_mask=rebuilt[2],
        stop_token_ids=stop.clone(),
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
    job_id: int | None = None,
    lease_epoch: int = 0,
    request_deadline_ns: int = NO_DEADLINE_NS,
    lease_deadline_ns: int = 0,
    artifact_digest: Tensor | None = None,
    session_digest: Tensor | None = None,
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
        STATUS_AUTHORITY_REJECTED,
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
        FINISH_LKG_STOP,
        FINISH_LKG_LENGTH,
    }:
        raise TensorProtocolError("response finish reason is unsupported")
    if not final and finish_reason != FINISH_NONE:
        raise TensorProtocolError("non-terminal chunks cannot have a finish reason")
    if final and finish_reason == FINISH_NONE:
        raise TensorProtocolError("terminal responses require a finish reason")
    if (
        status == STATUS_OK
        and final
        and finish_reason
        not in {
            FINISH_STOP,
            FINISH_LENGTH,
            FINISH_LKG_STOP,
            FINISH_LKG_LENGTH,
        }
    ):
        raise TensorProtocolError(
            "successful terminal response has invalid finish reason"
        )
    if status == STATUS_CANCELLED and finish_reason != FINISH_CANCELLED:
        raise TensorProtocolError("cancelled response has invalid finish reason")
    if status not in {STATUS_OK, STATUS_CANCELLED} and finish_reason != FINISH_ERROR:
        raise TensorProtocolError("failed response has invalid finish reason")
    if request_id == 0:
        if job_id not in {None, 0} or lease_epoch != 0 or lease_deadline_ns != 0:
            raise TensorProtocolError("request-zero response cannot carry authority")
        if request_deadline_ns not in {0, NO_DEADLINE_NS}:
            raise TensorProtocolError("request-zero response deadline is invalid")
        job = 0
        epoch = 0
        request_deadline = 0
        lease_deadline = 0
        artifact = _digest(
            artifact_digest, "artifact_digest", default=ZERO_ARTIFACT_DIGEST
        )
        session = _digest(
            session_digest, "session_digest", default=DEFAULT_SESSION_DIGEST
        )
        if not torch.equal(artifact, ZERO_ARTIFACT_DIGEST) or not torch.equal(
            session, DEFAULT_SESSION_DIGEST
        ):
            raise TensorProtocolError(
                "request-zero response digest authority is invalid"
            )
    else:
        job, epoch, request_deadline, lease_deadline, artifact, session = (
            _authority_fields(
                request_id,
                job_id=job_id,
                lease_epoch=lease_epoch,
                request_deadline_ns=request_deadline_ns,
                lease_deadline_ns=lease_deadline_ns,
                artifact_digest=artifact_digest,
                session_digest=session_digest,
            )
        )
    header = torch.tensor(
        [PROTOCOL_VERSION, status, request_id, job, epoch, int(bool(final))],
        dtype=torch.int64,
    )
    counters = torch.cat(
        (
            torch.tensor(
                [generated_total, request_deadline, lease_deadline],
                dtype=torch.int64,
            ),
            artifact,
            session,
        )
    ).contiguous()
    reason = torch.tensor([finish_reason], dtype=torch.int64)
    return header, tokens.clone(), counters, reason


def parse_response(message: object) -> ResponseFrame:
    if not isinstance(message, tuple) or len(message) != 4:
        raise TensorProtocolError("response must contain exactly four tensors")
    header = _cpu_i64(message[0], "response header", ndim=1)
    tokens = _cpu_i64(message[1], "response token_ids", ndim=1)
    counters = _cpu_i64(message[2], "response counters", ndim=1)
    reason = _cpu_i64(message[3], "response finish reason", ndim=1)
    if (
        header.shape != (6,)
        or counters.shape != (3 + 2 * DIGEST_BYTES,)
        or reason.shape != (1,)
    ):
        raise TensorProtocolError("response tensor shapes are invalid")
    version, status, request_id, job_id, lease_epoch, final = map(int, header.tolist())
    if version != PROTOCOL_VERSION or final not in {0, 1}:
        raise TensorProtocolError("response header is invalid")
    # Re-encode to apply status, token, and aggregate bounds consistently.
    checked = make_response(
        request_id,
        status,
        tokens,
        generated_total=int(counters[0].item()),
        final=bool(final),
        finish_reason=int(reason.item()),
        job_id=job_id,
        lease_epoch=lease_epoch,
        request_deadline_ns=int(counters[1].item()),
        lease_deadline_ns=int(counters[2].item()),
        artifact_digest=counters[3 : 3 + DIGEST_BYTES],
        session_digest=counters[3 + DIGEST_BYTES :],
    )
    return ResponseFrame(
        request_id=request_id,
        job_id=job_id,
        lease_epoch=lease_epoch,
        request_deadline_ns=int(checked[2][1].item()),
        lease_deadline_ns=int(checked[2][2].item()),
        artifact_digest=checked[2][3 : 3 + DIGEST_BYTES].clone(),
        session_digest=checked[2][3 + DIGEST_BYTES :].clone(),
        status=status,
        token_ids=checked[1],
        generated_total=int(checked[2][0].item()),
        final=bool(final),
        finish_reason=int(checked[3].item()),
    )


__all__ = [
    "GenerationRequest",
    "FINISH_CANCELLED",
    "FINISH_ERROR",
    "FINISH_LENGTH",
    "FINISH_LKG_LENGTH",
    "FINISH_LKG_STOP",
    "FINISH_NONE",
    "FINISH_STOP",
    "HARD_MAX_INPUT_TOKENS",
    "HARD_MAX_NEW_TOKENS",
    "HARD_MAX_STOP_TOKENS",
    "DEFAULT_SESSION_DIGEST",
    "DIGEST_BYTES",
    "NO_DEADLINE_NS",
    "OP_GENERATE",
    "OP_STOP",
    "PROTOCOL_VERSION",
    "ResponseFrame",
    "STATUS_BASE_MUTATED",
    "STATUS_AUTHORITY_REJECTED",
    "STATUS_CANCELLED",
    "STATUS_INVALID_REQUEST",
    "STATUS_MODEL_ERROR",
    "STATUS_OK",
    "TensorMessage",
    "TensorProtocolError",
    "ZERO_ARTIFACT_DIGEST",
    "make_generate_request",
    "make_response",
    "make_stop_request",
    "parse_request",
    "parse_response",
]
