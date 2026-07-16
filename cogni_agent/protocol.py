"""Tensor-only IPC protocol for the resident local generation worker.

Natural language is permitted at the web/control boundary.  It must be
tokenized before this protocol is used; no Python string, JSON document, or
arbitrary object crosses the model-process boundary.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import math
from typing import TypeAlias

import torch
from torch import Tensor


PROTOCOL_VERSION = 5

OP_GENERATE = 1
OP_STOP = 2

# Decode policy is an explicit, bounded scalar in the tensor protocol.  Normal
# conversation uses the model-recommended sampling profile; callers that need
# exact-copy/replay semantics must opt into strict greedy decoding.
DECODE_STRICT = 0
DECODE_CONVERSATION = 1

STATUS_OK = 0
STATUS_CANCELLED = 1
STATUS_INVALID_REQUEST = 2
STATUS_MODEL_ERROR = 3
STATUS_BASE_MUTATED = 4
STATUS_AUTHORITY_REJECTED = 5
STATUS_DEADLINE_EXCEEDED = 6

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
# A processed image is deliberately smaller than the generic preprocessing
# bundle ceiling. Packing float32 bit patterns into the existing int64 control
# tensor doubles their byte footprint, so this limit keeps one IPC frame below
# a bounded 32 MiB payload plus metadata while admitting Gemma 4's observed
# 7.8 MiB image feature tensor.
HARD_MAX_IMAGE_TENSOR_BYTES = 16 * 1024 * 1024
HARD_MAX_IMAGE_PACKED_VALUES = HARD_MAX_IMAGE_TENSOR_BYTES // 4
HARD_MAX_AUDIO_TENSOR_BYTES = 8 * 1024 * 1024
HARD_MAX_AUDIO_PACKED_VALUES = HARD_MAX_AUDIO_TENSOR_BYTES // 4
HARD_MAX_IMAGE_DIMENSION = 16_384
HARD_MAX_MULTIMODAL_DIMENSION = HARD_MAX_IMAGE_DIMENSION
MAX_MULTIMODAL_FIELDS = 3
MAX_MULTIMODAL_FIELD_RANK = 4

MODALITY_TEXT = 0
MODALITY_IMAGE = 1
MODALITY_AUDIO = 2

# Fixed integer field identifiers are the only image schema crossing IPC. No
# filename, media type string, JSON key, or arbitrary Python object is sent to
# the resident model process.
IMAGE_FIELD_MM_TOKEN_TYPE_IDS = 1
IMAGE_FIELD_PIXEL_VALUES = 2
IMAGE_FIELD_POSITION_IDS = 3
AudioTensorField: TypeAlias = tuple[int, Tensor]
AUDIO_FIELD_MM_TOKEN_TYPE_IDS = 4
AUDIO_FIELD_INPUT_FEATURES = 5
AUDIO_FIELD_INPUT_FEATURES_MASK = 6
_IMAGE_FIELD_CODES = frozenset(
    {
        IMAGE_FIELD_MM_TOKEN_TYPE_IDS,
        IMAGE_FIELD_PIXEL_VALUES,
        IMAGE_FIELD_POSITION_IDS,
    }
)
_IMAGE_REQUIRED_FIELDS = _IMAGE_FIELD_CODES
_AUDIO_FIELD_CODES = frozenset(
    {
        AUDIO_FIELD_MM_TOKEN_TYPE_IDS,
        AUDIO_FIELD_INPUT_FEATURES,
        AUDIO_FIELD_INPUT_FEATURES_MASK,
    }
)
_AUDIO_REQUIRED_FIELDS = _AUDIO_FIELD_CODES
_IMAGE_DTYPE_I64 = 1
_IMAGE_DTYPE_F32 = 2
_MULTIMODAL_DTYPE_BOOL = 3
_MULTIMODAL_DESCRIPTOR_VALUES = 4 + MAX_MULTIMODAL_FIELD_RANK
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
    decode_mode: int
    sampling_seed: int
    modality: int
    image_tensors: tuple[tuple[int, Tensor], ...]
    audio_tensors: tuple[AudioTensorField, ...]


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


def _multimodal_field_dtype(field_code: int) -> torch.dtype:
    if field_code in {IMAGE_FIELD_PIXEL_VALUES, AUDIO_FIELD_INPUT_FEATURES}:
        return torch.float32
    if field_code in {
        IMAGE_FIELD_MM_TOKEN_TYPE_IDS,
        IMAGE_FIELD_POSITION_IDS,
        AUDIO_FIELD_MM_TOKEN_TYPE_IDS,
    }:
        return torch.int64
    if field_code == AUDIO_FIELD_INPUT_FEATURES_MASK:
        return torch.bool
    raise TensorProtocolError("multimodal field code is unsupported")


def _validate_multimodal_tensors(
    value: object,
    *,
    input_length: int,
    modality: int,
) -> tuple[tuple[int, Tensor], ...]:
    if value is None or (isinstance(value, tuple) and not value):
        return ()
    if modality == MODALITY_IMAGE:
        allowed_fields = _IMAGE_FIELD_CODES
        required_fields = _IMAGE_REQUIRED_FIELDS
        max_tensor_bytes = HARD_MAX_IMAGE_TENSOR_BYTES
        max_packed_values = HARD_MAX_IMAGE_PACKED_VALUES
        label = "image"
    elif modality == MODALITY_AUDIO:
        allowed_fields = _AUDIO_FIELD_CODES
        required_fields = _AUDIO_REQUIRED_FIELDS
        max_tensor_bytes = HARD_MAX_AUDIO_TENSOR_BYTES
        max_packed_values = HARD_MAX_AUDIO_PACKED_VALUES
        label = "audio"
    else:
        raise TensorProtocolError("multimodal tensor modality is unsupported")
    if not isinstance(value, tuple) or not 1 <= len(value) <= MAX_MULTIMODAL_FIELDS:
        raise TensorProtocolError(f"{label} tensors must use the fixed tuple schema")
    rows: list[tuple[int, Tensor]] = []
    seen: set[int] = set()
    raw_bytes = 0
    packed_values = 0
    for item in value:
        if not isinstance(item, tuple) or len(item) != 2:
            raise TensorProtocolError("image tensor entry is invalid")
        field_code, tensor = item
        if (
            not isinstance(field_code, int)
            or isinstance(field_code, bool)
            or field_code not in allowed_fields
            or field_code in seen
        ):
            raise TensorProtocolError(f"{label} tensor field is unknown or duplicated")
        if not isinstance(tensor, Tensor):
            raise TensorProtocolError(f"{label} field must be a tensor")
        if tensor.device.type != "cpu" or not tensor.is_contiguous():
            raise TensorProtocolError(
                f"{label} tensors must be contiguous CPU tensors across IPC"
            )
        expected_dtype = _multimodal_field_dtype(field_code)
        if tensor.dtype != expected_dtype:
            raise TensorProtocolError(f"{label} tensor dtype is invalid")
        if not 1 <= tensor.ndim <= MAX_MULTIMODAL_FIELD_RANK or tensor.numel() < 1:
            raise TensorProtocolError(f"{label} tensor rank or size is invalid")
        if any(
            not 1 <= int(dimension) <= HARD_MAX_MULTIMODAL_DIMENSION
            for dimension in tensor.shape
        ):
            raise TensorProtocolError(
                f"{label} tensor dimension exceeds its hard bound"
            )
        if int(tensor.shape[0]) != 1:
            raise TensorProtocolError(f"{label} tensor batch size must be one")
        if field_code in {
            IMAGE_FIELD_MM_TOKEN_TYPE_IDS,
            AUDIO_FIELD_MM_TOKEN_TYPE_IDS,
        }:
            if tensor.ndim != 2 or tuple(tensor.shape) != (1, input_length):
                raise TensorProtocolError("multimodal token types must match input_ids")
        elif field_code == IMAGE_FIELD_PIXEL_VALUES:
            if tensor.ndim != 3 or not bool(torch.isfinite(tensor).all()):
                raise TensorProtocolError("pixel values must be finite rank-three data")
        elif field_code == IMAGE_FIELD_POSITION_IDS and tensor.ndim not in {2, 3, 4}:
            raise TensorProtocolError("image position ids have an invalid rank")
        if tensor.dtype == torch.int64 and (
            bool((tensor < -(2**31)).any()) or bool((tensor > 2**31 - 1).any())
        ):
            raise TensorProtocolError(f"{label} integer values exceed their hard bound")
        raw_bytes += tensor.numel() * tensor.element_size()
        packed_values += tensor.numel()
        if raw_bytes > max_tensor_bytes or packed_values > max_packed_values:
            raise TensorProtocolError(f"{label} tensors exceed their IPC byte bound")
        seen.add(field_code)
        rows.append((field_code, tensor))
    if seen != required_fields:
        raise TensorProtocolError(f"{label} request omitted a required Gemma 4 tensor")
    mapping = dict(rows)
    if modality == MODALITY_AUDIO:
        features = mapping[AUDIO_FIELD_INPUT_FEATURES]
        feature_mask = mapping[AUDIO_FIELD_INPUT_FEATURES_MASK]
        if features.ndim != 3 or not bool(torch.isfinite(features).all()):
            raise TensorProtocolError(
                "audio input features must be finite rank-three data"
            )
        if feature_mask.ndim != 2 or tuple(feature_mask.shape) != tuple(
            features.shape[:2]
        ):
            raise TensorProtocolError(
                "audio feature mask must match the input feature frames"
            )
    return tuple(sorted(rows, key=lambda row: row[0]))


def _pack_multimodal_tensors(
    tensors: tuple[tuple[int, Tensor], ...],
) -> tuple[Tensor, Tensor]:
    descriptors: list[Tensor] = []
    payloads: list[Tensor] = []
    for field_code, tensor in tensors:
        dimensions = list(map(int, tensor.shape))
        padded_dimensions = dimensions + [0] * (
            MAX_MULTIMODAL_FIELD_RANK - len(dimensions)
        )
        dtype_code = {
            torch.int64: _IMAGE_DTYPE_I64,
            torch.float32: _IMAGE_DTYPE_F32,
            torch.bool: _MULTIMODAL_DTYPE_BOOL,
        }[tensor.dtype]
        descriptors.append(
            torch.tensor(
                [
                    field_code,
                    dtype_code,
                    tensor.ndim,
                    tensor.numel(),
                    *padded_dimensions,
                ],
                dtype=torch.int64,
            )
        )
        flat = tensor.reshape(-1)
        if tensor.dtype == torch.float32:
            # Preserve IEEE-754 bits exactly. Signed int32 values are promoted
            # to int64 only for transport and reconstructed by the parser.
            payload = flat.view(torch.int32).to(torch.int64)
        elif tensor.dtype == torch.bool:
            payload = flat.to(torch.int64)
        else:
            payload = flat.clone()
        payloads.append(payload.contiguous())
    descriptor_tensor = torch.cat(descriptors).contiguous() if descriptors else _empty()
    payload_tensor = torch.cat(payloads).contiguous() if payloads else _empty()
    return descriptor_tensor, payload_tensor


def _unpack_multimodal_tensors(
    descriptors: Tensor,
    payload: Tensor,
    *,
    field_count: int,
    input_length: int,
    modality: int,
) -> tuple[tuple[int, Tensor], ...]:
    if descriptors.numel() != field_count * _MULTIMODAL_DESCRIPTOR_VALUES:
        raise TensorProtocolError("multimodal descriptors are truncated")
    if field_count == 0:
        if payload.numel():
            raise TensorProtocolError("text request cannot carry multimodal payload")
        return ()
    allowed_fields = {
        MODALITY_IMAGE: _IMAGE_FIELD_CODES,
        MODALITY_AUDIO: _AUDIO_FIELD_CODES,
    }.get(modality)
    if allowed_fields is None:
        raise TensorProtocolError("multimodal descriptor modality is unsupported")
    cursor = 0
    rows: list[tuple[int, Tensor]] = []
    for index in range(field_count):
        start = index * _MULTIMODAL_DESCRIPTOR_VALUES
        values = list(
            map(
                int,
                descriptors[start : start + _MULTIMODAL_DESCRIPTOR_VALUES].tolist(),
            )
        )
        field_code, dtype_code, rank, numel = values[:4]
        dimensions = values[4:]
        if (
            field_code not in allowed_fields
            or dtype_code
            not in {
                _IMAGE_DTYPE_I64,
                _IMAGE_DTYPE_F32,
                _MULTIMODAL_DTYPE_BOOL,
            }
            or not 1 <= rank <= MAX_MULTIMODAL_FIELD_RANK
            or any(dimension != 0 for dimension in dimensions[rank:])
            or any(
                not 1 <= dimension <= HARD_MAX_MULTIMODAL_DIMENSION
                for dimension in dimensions[:rank]
            )
        ):
            raise TensorProtocolError("multimodal descriptor is invalid")
        shape = tuple(dimensions[:rank])
        expected_numel = math.prod(shape)
        if numel != expected_numel or cursor + numel > payload.numel():
            raise TensorProtocolError("multimodal descriptor size is inconsistent")
        expected_dtype = _multimodal_field_dtype(field_code)
        expected_dtype_code = {
            torch.int64: _IMAGE_DTYPE_I64,
            torch.float32: _IMAGE_DTYPE_F32,
            torch.bool: _MULTIMODAL_DTYPE_BOOL,
        }[expected_dtype]
        if dtype_code != expected_dtype_code:
            raise TensorProtocolError("multimodal descriptor dtype is inconsistent")
        packed = payload[cursor : cursor + numel]
        cursor += numel
        if dtype_code == _IMAGE_DTYPE_F32:
            if bool((packed < -(2**31)).any()) or bool((packed > 2**31 - 1).any()):
                raise TensorProtocolError("packed pixel bits exceed int32")
            tensor = packed.to(torch.int32).view(torch.float32).reshape(shape)
        elif dtype_code == _MULTIMODAL_DTYPE_BOOL:
            if bool(((packed != 0) & (packed != 1)).any()):
                raise TensorProtocolError("packed boolean mask is not binary")
            tensor = packed.to(torch.bool).reshape(shape)
        else:
            tensor = packed.reshape(shape).clone()
        rows.append((field_code, tensor.contiguous()))
    if cursor != payload.numel():
        raise TensorProtocolError("multimodal payload has trailing values")
    return _validate_multimodal_tensors(
        tuple(rows),
        input_length=input_length,
        modality=modality,
    )


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
    decode_mode: int = DECODE_CONVERSATION,
    sampling_seed: int = 0,
    image_tensors: tuple[tuple[int, Tensor], ...] | None = None,
    audio_tensors: tuple[AudioTensorField, ...] | None = None,
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
    if decode_mode not in {DECODE_STRICT, DECODE_CONVERSATION}:
        raise TensorProtocolError("decode_mode is unsupported")
    if (
        not isinstance(sampling_seed, int)
        or isinstance(sampling_seed, bool)
        or not 0 <= sampling_seed <= NO_DEADLINE_NS
    ):
        raise TensorProtocolError(
            "sampling_seed must be a non-negative signed-63-bit integer"
        )
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
    image = _validate_multimodal_tensors(
        image_tensors,
        input_length=int(ids.shape[1]),
        modality=MODALITY_IMAGE,
    )
    audio = _validate_multimodal_tensors(
        audio_tensors,
        input_length=int(ids.shape[1]),
        modality=MODALITY_AUDIO,
    )
    if image and audio:
        raise TensorProtocolError("one request cannot contain image and audio tensors")
    modality = MODALITY_IMAGE if image else MODALITY_AUDIO if audio else MODALITY_TEXT
    multimodal_tensors = image or audio
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
            decode_mode,
            sampling_seed,
        ],
        dtype=torch.int64,
    )
    descriptors, multimodal_payload = _pack_multimodal_tensors(multimodal_tensors)
    control = torch.cat(
        (
            torch.tensor(
                [stop.numel(), modality, len(multimodal_tensors)],
                dtype=torch.int64,
            ),
            artifact,
            session,
            stop,
            descriptors,
            multimodal_payload,
        )
    ).contiguous()
    return header, ids.clone(), mask.clone(), control


def make_stop_request() -> TensorMessage:
    header = torch.tensor(
        [PROTOCOL_VERSION, OP_STOP, 0, 0, 0, 0, 0, 0, 0, 0],
        dtype=torch.int64,
    )
    return header, _empty(), _empty(), _empty()


def parse_request(message: object) -> GenerationRequest | None:
    if not isinstance(message, tuple) or len(message) != 4:
        raise TensorProtocolError("request must contain exactly four tensors")
    header = _cpu_i64(message[0], "request header", ndim=1)
    if header.shape != (10,):
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
        decode_mode,
        sampling_seed,
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
                decode_mode,
                sampling_seed,
            )
        ):
            raise TensorProtocolError("stop request header is invalid")
        return None
    if opcode != OP_GENERATE:
        raise TensorProtocolError("request opcode is unsupported")
    # Reuse the constructor as the single source of all request bounds.
    raw_ids = _cpu_i64(message[1], "input_ids", ndim=2)
    if raw_ids.shape[0] != 1 or not 1 <= raw_ids.shape[1] <= HARD_MAX_INPUT_TOKENS:
        raise TensorProtocolError("input_ids must have bounded batch-one shape")
    control = _cpu_i64(message[3], "request control", ndim=1)
    minimum = 3 + 2 * DIGEST_BYTES
    if control.numel() < minimum:
        raise TensorProtocolError("request control tensor is truncated")
    maximum = (
        minimum
        + HARD_MAX_STOP_TOKENS
        + MAX_MULTIMODAL_FIELDS * _MULTIMODAL_DESCRIPTOR_VALUES
        + max(HARD_MAX_IMAGE_PACKED_VALUES, HARD_MAX_AUDIO_PACKED_VALUES)
    )
    if control.numel() > maximum:
        raise TensorProtocolError("request control tensor exceeds its hard bound")
    stop_count = int(control[0].item())
    modality = int(control[1].item())
    field_count = int(control[2].item())
    if (
        not 0 <= stop_count <= HARD_MAX_STOP_TOKENS
        or modality not in {MODALITY_TEXT, MODALITY_IMAGE, MODALITY_AUDIO}
        or not 0 <= field_count <= MAX_MULTIMODAL_FIELDS
        or (modality == MODALITY_TEXT) != (field_count == 0)
    ):
        raise TensorProtocolError(
            "request control tensor has invalid modality metadata"
        )
    descriptor_values = field_count * _MULTIMODAL_DESCRIPTOR_VALUES
    payload_start = minimum + stop_count + descriptor_values
    if payload_start > control.numel():
        raise TensorProtocolError("request control tensor is truncated")
    artifact = control[3 : 3 + DIGEST_BYTES]
    session = control[3 + DIGEST_BYTES : minimum]
    stop = control[minimum : minimum + stop_count]
    descriptors = control[
        minimum + stop_count : minimum + stop_count + descriptor_values
    ]
    payload = control[payload_start:]
    multimodal_tensors = _unpack_multimodal_tensors(
        descriptors,
        payload,
        field_count=field_count,
        input_length=int(raw_ids.shape[1]),
        modality=modality,
    )
    image_tensors = multimodal_tensors if modality == MODALITY_IMAGE else ()
    audio_tensors = multimodal_tensors if modality == MODALITY_AUDIO else ()
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
        decode_mode=decode_mode,
        sampling_seed=sampling_seed,
        image_tensors=image_tensors,
        audio_tensors=audio_tensors,
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
        decode_mode=decode_mode,
        sampling_seed=sampling_seed,
        modality=modality,
        image_tensors=tuple((code, tensor.clone()) for code, tensor in image_tensors),
        audio_tensors=tuple((code, tensor.clone()) for code, tensor in audio_tensors),
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
        STATUS_DEADLINE_EXCEEDED,
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
    "AUDIO_FIELD_INPUT_FEATURES",
    "AUDIO_FIELD_INPUT_FEATURES_MASK",
    "AUDIO_FIELD_MM_TOKEN_TYPE_IDS",
    "DECODE_CONVERSATION",
    "DECODE_STRICT",
    "FINISH_CANCELLED",
    "FINISH_ERROR",
    "FINISH_LENGTH",
    "FINISH_LKG_LENGTH",
    "FINISH_LKG_STOP",
    "FINISH_NONE",
    "FINISH_STOP",
    "HARD_MAX_INPUT_TOKENS",
    "HARD_MAX_AUDIO_PACKED_VALUES",
    "HARD_MAX_AUDIO_TENSOR_BYTES",
    "HARD_MAX_IMAGE_PACKED_VALUES",
    "HARD_MAX_IMAGE_TENSOR_BYTES",
    "HARD_MAX_NEW_TOKENS",
    "HARD_MAX_STOP_TOKENS",
    "DEFAULT_SESSION_DIGEST",
    "DIGEST_BYTES",
    "NO_DEADLINE_NS",
    "MODALITY_IMAGE",
    "MODALITY_AUDIO",
    "MODALITY_TEXT",
    "IMAGE_FIELD_MM_TOKEN_TYPE_IDS",
    "IMAGE_FIELD_PIXEL_VALUES",
    "IMAGE_FIELD_POSITION_IDS",
    "OP_GENERATE",
    "OP_STOP",
    "PROTOCOL_VERSION",
    "ResponseFrame",
    "STATUS_BASE_MUTATED",
    "STATUS_AUTHORITY_REJECTED",
    "STATUS_CANCELLED",
    "STATUS_DEADLINE_EXCEEDED",
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
