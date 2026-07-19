"""Resident, single-owner local model process with tensor-only IPC."""

from __future__ import annotations

from collections import deque
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from hashlib import sha256
import math
import multiprocessing
import os
from pathlib import Path
from queue import Empty, Full
import secrets
import sys
from threading import Lock, RLock
from time import monotonic, monotonic_ns, sleep
from typing import Any, Callable, Iterator, Mapping

import torch
from torch import Tensor, nn

from cogni_agent.core_pipeline import (
    CoreTurnAuthorityError,
    CoreTurnPipeline,
    CoreTurnRequest,
    CoreTurnResult,
)
from cogni_core.backbone import (
    LocalGemmaFeatureBackbone,
    load_local_gemma,
    verify_local_gemma_path,
)
from cogni_core.cts_policy import (
    ACTION_WIDTH,
    ACT_BOUNDS,
    DEFAULT_CHECKPOINT_SHA256,
    EXPLORATION_BOUNDS,
    META_CONTROL_DIM,
    SUMMARY_DIM,
    TEMPERATURE_BOUNDS,
    TOLERANCE_BOUNDS,
    CTSCheckpointError,
    LearnedCTSController,
    load_default_bounded_cts_controller,
)
from cogni_core.meta_router import cognitive_state_tensor
from cogni_core.search import (
    BoundedPUCTSearchV2,
    CertifiedBroydenTransitionV2,
    CertifiedPUCTConfigV2,
    SearchControlsV2,
)
from cogni_os.config import load_config
from cogni_os.artifacts import (
    VerifiedArtifactSet,
    verify_artifact_manifest,
)
from cogni_os.factory import build_genesis_runtime
from cogni_os.gpu_lease import (
    DEFAULT_MAX_VRAM_BYTES,
    GPULease,
    GPULeaseBudgetError,
    GPULeaseManager,
    StaleGPULeaseError,
)
from cogni_os.runtime import SearchCollaboratorsV2

from .conditioning import build_latent_logits_processor
from .multimodal import (
    MultimodalPreprocessError,
    MultimodalTensorBundle,
    VerifiedGemma4MultimodalProcessor,
)
from . import model_trust as _model_trust
from .protocol import (
    AUDIO_FIELD_INPUT_FEATURES,
    AUDIO_FIELD_INPUT_FEATURES_MASK,
    AUDIO_FIELD_MM_TOKEN_TYPE_IDS,
    DECODE_CONVERSATION,
    DECODE_STRICT,
    DIGEST_BYTES,
    FINISH_CANCELLED,
    FINISH_ERROR,
    FINISH_LENGTH,
    FINISH_STOP,
    HARD_MAX_INPUT_TOKENS,
    HARD_MAX_NEW_TOKENS,
    IMAGE_FIELD_MM_TOKEN_TYPE_IDS,
    IMAGE_FIELD_PIXEL_VALUES,
    IMAGE_FIELD_POSITION_IDS,
    NO_DEADLINE_NS,
    STATUS_BASE_MUTATED,
    STATUS_AUTHORITY_REJECTED,
    STATUS_CANCELLED,
    STATUS_DEADLINE_EXCEEDED,
    STATUS_INVALID_REQUEST,
    STATUS_MODEL_ERROR,
    STATUS_OK,
    TensorMessage,
    TensorProtocolError,
    ZERO_ARTIFACT_DIGEST,
    make_generate_request,
    make_response,
    make_stop_request,
    parse_request,
    parse_response,
)
from .prompting import reserved_stop_sequences


HARD_MAX_PROMPT_CHARS = 64_000
HARD_MAX_RESPONSE_CHARS = 64_000
DEFAULT_RESPONSE_QUEUE_SIZE = 64
WORKER_RETIRE_ACK_TIMEOUT_SECONDS = 2.0
MAX_SIGNATURE_TENSORS = 16_384
MAX_SIGNATURE_MODULES = 32_768
CONTENT_FINGERPRINT_TENSOR_BUDGET = 128
CONTENT_FINGERPRINT_VALUES_PER_TENSOR = 16
PRODUCTION_CTS_DEPTH = 100
PRODUCTION_CTS_NODE_CAPACITY = 301
# Depth-100 validation must not be shortened by the learned ACT head. The
# certified arena admits at most 100 full width-3 expansions; all 301 bounded
# simulations remain available for selection/backup evidence.
PRODUCTION_CTS_ACT_HARD_FLOOR = 301

# Re-export the legacy model-service symbols while keeping one authoritative
# trust-root implementation shared with the multimodal boundary.
TRUSTED_GEMMA4_E4B_IT_REVISION = _model_trust.TRUSTED_GEMMA4_E4B_IT_REVISION
TRUSTED_GEMMA4_E4B_IT_DIGESTS = _model_trust.TRUSTED_GEMMA4_E4B_IT_DIGESTS


def _require_instruction_tuned_e4b(verified: VerifiedArtifactSet) -> None:
    _model_trust.require_instruction_tuned_e4b(verified)


def _verify_instruction_tuned_e4b_snapshot(
    verified: VerifiedArtifactSet,
) -> None:
    _model_trust.verify_instruction_tuned_e4b_snapshot(verified)


# Gemma 4 e4b's local generation profile. These values are deliberately
# constants rather than user-controlled floats so the worker's search space
# remains bounded and the tensor protocol only needs a policy enum and seed.
CONVERSATION_TEMPERATURE = 1.0
CONVERSATION_TOP_P = 0.95
CONVERSATION_TOP_K = 64

_FINISH_REASON_NAMES = {
    FINISH_STOP: "stop",
    FINISH_LENGTH: "length",
    FINISH_CANCELLED: "cancelled",
    FINISH_ERROR: "error",
}

# Startup diagnostics cross the process boundary as one packed int64 value,
# never as exception text. The upper bits hold the last startup stage and the
# low byte holds a coarse error class. One shared scalar avoids a torn
# stage/error observation across processes.
_STARTUP_STAGE_CODES = {
    "worker_startup": 1,
    "worker_authority": 2,
    "model_factory": 3,
    "prepare_base_model": 4,
    "core_pipeline_factory": 5,
    "base_signature": 6,
    "model_device": 7,
    "product_factory_check": 8,
    "ready": 100,
}
_STARTUP_STAGE_LABELS = {
    1: "worker bootstrap",
    2: "GPU lease authority verification",
    3: "verified local model loading",
    4: "base-model preparation",
    5: "Cogni-Core pipeline initialization",
    6: "base-model integrity signature",
    7: "model device verification",
    8: "product factory verification",
    100: "ready publication",
}
_STARTUP_ERROR_NONE = 0
_STARTUP_ERROR_GENERAL = 1
_STARTUP_ERROR_CTS_CHECKPOINT = 2
_STARTUP_ERROR_GPU_MEMORY = 3


class ModelServiceError(RuntimeError):
    """Base error for the local generation service."""


class RequestLimitError(ModelServiceError):
    pass


class ServiceBusyError(ModelServiceError):
    pass


class WorkerStartupError(ModelServiceError):
    pass


class WorkerExecutionError(ModelServiceError):
    pass


class GenerationCancelled(ModelServiceError):
    pass


class BaseModelMutationError(ModelServiceError):
    pass


class WorkerAuthorityError(ModelServiceError):
    """A stale/replayed worker capability was rejected before publication."""


def _validated_digest_hex(value: object, field: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field} must be a SHA-256 hex string")
    normalized = value.strip().lower()
    if len(normalized) != 64 or any(
        character not in "0123456789abcdef" for character in normalized
    ):
        raise ValueError(f"{field} must be a SHA-256 hex string")
    return normalized


def _digest_tensor(value: str) -> Tensor:
    return torch.tensor(list(bytes.fromhex(value)), dtype=torch.int64)


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class GenerationChunk:
    request_id: int
    token_ids: Tensor
    generated_total: int
    final: bool
    cancelled: bool = False
    finish_reason: str | None = None
    generation_mode: str = "cogni_core"
    media_sha256: str | None = None
    runtime_identity: ModelRuntimeIdentity | None = None


@dataclass(frozen=True)
class GenerationResult:
    request_id: int
    token_ids: Tensor
    text: str
    finish_reason: str
    generation_mode: str = "cogni_core"
    media_sha256: str | None = None
    runtime_identity: ModelRuntimeIdentity | None = None


@dataclass(frozen=True, slots=True)
class ModelRuntimeIdentity:
    """Process-local identity of one live, manifest-bound resident worker."""

    service_nonce: str
    worker_incarnation: int
    worker_pid: int
    lease_epoch: int
    lease_deadline_ns: int
    artifact_digest: str
    model_root: str
    manifest_path: str
    processor_root: str
    processor_manifest_path: str


@dataclass(frozen=True, slots=True)
class _RequestAuthority:
    request_id: int
    job_id: int
    lease_epoch: int
    request_deadline_ns: int
    lease_deadline_ns: int
    artifact_digest: Tensor
    session_digest: Tensor

    @classmethod
    def from_request(cls, request: object) -> _RequestAuthority:
        return cls(
            request_id=int(getattr(request, "request_id")),
            job_id=int(getattr(request, "job_id")),
            lease_epoch=int(getattr(request, "lease_epoch")),
            request_deadline_ns=int(getattr(request, "request_deadline_ns")),
            lease_deadline_ns=int(getattr(request, "lease_deadline_ns")),
            artifact_digest=getattr(request, "artifact_digest").clone(),
            session_digest=getattr(request, "session_digest").clone(),
        )

    def response_kwargs(self) -> dict[str, object]:
        return {
            "job_id": self.job_id,
            "lease_epoch": self.lease_epoch,
            "request_deadline_ns": self.request_deadline_ns,
            "lease_deadline_ns": self.lease_deadline_ns,
            "artifact_digest": self.artifact_digest,
            "session_digest": self.session_digest,
        }

    def validate_frame(self, frame: object) -> None:
        scalar_fields = (
            "request_id",
            "job_id",
            "lease_epoch",
            "request_deadline_ns",
            "lease_deadline_ns",
        )
        if any(
            int(getattr(frame, name)) != int(getattr(self, name))
            for name in scalar_fields
        ):
            raise WorkerAuthorityError("worker response authority changed")
        if not torch.equal(
            getattr(frame, "artifact_digest"), self.artifact_digest
        ) or not torch.equal(getattr(frame, "session_digest"), self.session_digest):
            raise WorkerAuthorityError("worker response digest authority changed")


def _session_digest(session_id: str) -> Tensor:
    if not isinstance(session_id, str):
        raise TypeError("conversation_id must be a string")
    normalized = session_id.strip()
    if not 1 <= len(normalized) <= 128:
        raise ValueError("conversation_id must contain 1-128 characters")
    if any(ord(character) < 32 or ord(character) == 127 for character in normalized):
        raise ValueError("conversation_id contains a control character")
    return torch.tensor(
        list(sha256(normalized.encode("utf-8")).digest()), dtype=torch.int64
    )


def _bind_media_session_digest(session_digest: Tensor, content_sha256: str) -> Tensor:
    """Bind every worker response to the exact locally decoded media bytes."""

    digest_hex = _validated_digest_hex(content_sha256, "media content digest")
    if session_digest.shape != (DIGEST_BYTES,) or session_digest.dtype != torch.int64:
        raise TensorProtocolError("conversation digest is invalid")
    bound = sha256(
        bytes(map(int, session_digest.tolist())) + bytes.fromhex(digest_hex)
    ).digest()
    return torch.tensor(list(bound), dtype=torch.int64)


def _image_bundle_inputs(
    bundle: MultimodalTensorBundle,
    *,
    max_input_tokens: int,
) -> tuple[Tensor, Tensor, tuple[tuple[int, Tensor], ...], str]:
    if (
        not isinstance(bundle, MultimodalTensorBundle)
        or bundle.modality != "image"
        or bundle.processor_verified is not True
    ):
        raise RequestLimitError("image was not produced by the verified processor")
    digest = _validated_digest_hex(bundle.content_sha256, "image content digest")
    mapping = bundle.as_mapping()
    if set(mapping) != {
        "input_ids",
        "attention_mask",
        "mm_token_type_ids",
        "pixel_values",
        "image_position_ids",
    }:
        raise RequestLimitError("verified image bundle has an unsupported schema")
    input_ids = mapping["input_ids"]
    attention_mask = mapping["attention_mask"]
    if (
        input_ids.device.type != "cpu"
        or input_ids.dtype != torch.int64
        or not input_ids.is_contiguous()
        or input_ids.ndim != 2
        or input_ids.shape[0] != 1
        or not 1 <= input_ids.shape[1] <= max_input_tokens
    ):
        raise RequestLimitError("image prompt exceeds the service token bound")
    if (
        attention_mask.device.type != "cpu"
        or attention_mask.dtype != torch.int64
        or not attention_mask.is_contiguous()
        or attention_mask.shape != input_ids.shape
    ):
        raise RequestLimitError("image attention mask is invalid")
    image_tensors = (
        (IMAGE_FIELD_MM_TOKEN_TYPE_IDS, mapping["mm_token_type_ids"]),
        (IMAGE_FIELD_PIXEL_VALUES, mapping["pixel_values"]),
        (IMAGE_FIELD_POSITION_IDS, mapping["image_position_ids"]),
    )
    return input_ids, attention_mask, image_tensors, digest


def _image_generation_kwargs(
    image_tensors: tuple[tuple[int, Tensor], ...],
    device: torch.device,
) -> dict[str, Tensor]:
    names = {
        IMAGE_FIELD_MM_TOKEN_TYPE_IDS: "mm_token_type_ids",
        IMAGE_FIELD_PIXEL_VALUES: "pixel_values",
        IMAGE_FIELD_POSITION_IDS: "image_position_ids",
    }
    if {code for code, _tensor in image_tensors} != set(names):
        raise TensorProtocolError("worker image tensor schema is incomplete")
    return {
        names[code]: tensor.to(device, non_blocking=False)
        for code, tensor in image_tensors
    }


def _audio_bundle_inputs(
    bundle: MultimodalTensorBundle,
    *,
    max_input_tokens: int,
) -> tuple[Tensor, Tensor, tuple[tuple[int, Tensor], ...], str]:
    if (
        not isinstance(bundle, MultimodalTensorBundle)
        or bundle.modality != "audio"
        or bundle.processor_verified is not True
    ):
        raise RequestLimitError("audio was not produced by the verified processor")
    digest = _validated_digest_hex(bundle.content_sha256, "audio content digest")
    mapping = bundle.as_mapping()
    if set(mapping) != {
        "input_ids",
        "attention_mask",
        "mm_token_type_ids",
        "input_features",
        "input_features_mask",
    }:
        raise RequestLimitError("verified audio bundle has an unsupported schema")
    input_ids = mapping["input_ids"]
    attention_mask = mapping["attention_mask"]
    if (
        input_ids.device.type != "cpu"
        or input_ids.dtype != torch.int64
        or not input_ids.is_contiguous()
        or input_ids.ndim != 2
        or input_ids.shape[0] != 1
        or not 1 <= input_ids.shape[1] <= max_input_tokens
    ):
        raise RequestLimitError("audio prompt exceeds the service token bound")
    if (
        attention_mask.device.type != "cpu"
        or attention_mask.dtype != torch.int64
        or not attention_mask.is_contiguous()
        or attention_mask.shape != input_ids.shape
    ):
        raise RequestLimitError("audio attention mask is invalid")
    audio_tensors = (
        (AUDIO_FIELD_MM_TOKEN_TYPE_IDS, mapping["mm_token_type_ids"]),
        (AUDIO_FIELD_INPUT_FEATURES, mapping["input_features"]),
        (AUDIO_FIELD_INPUT_FEATURES_MASK, mapping["input_features_mask"]),
    )
    return input_ids, attention_mask, audio_tensors, digest


def _audio_generation_kwargs(
    audio_tensors: tuple[tuple[int, Tensor], ...],
    device: torch.device,
) -> dict[str, Tensor]:
    names = {
        AUDIO_FIELD_MM_TOKEN_TYPE_IDS: "mm_token_type_ids",
        AUDIO_FIELD_INPUT_FEATURES: "input_features",
        AUDIO_FIELD_INPUT_FEATURES_MASK: "input_features_mask",
    }
    if {code for code, _tensor in audio_tensors} != set(names):
        raise TensorProtocolError("worker audio tensor schema is incomplete")
    return {
        names[code]: tensor.to(device, non_blocking=False)
        for code, tensor in audio_tensors
    }


def _decode_mode_code(value: object) -> int:
    if value == "conversation":
        return DECODE_CONVERSATION
    if value == "strict":
        return DECODE_STRICT
    raise ValueError("decode_mode must be 'conversation' or 'strict'")


def _validated_sampling_seed(value: object) -> int | None:
    if value is None:
        return None
    if (
        not isinstance(value, int)
        or isinstance(value, bool)
        or not 0 <= value <= NO_DEADLINE_NS
    ):
        raise ValueError("sampling_seed must be a non-negative signed-63-bit integer")
    return int(value)


def _derived_sampling_seed(input_ids: Tensor, session_digest: Tensor) -> int:
    """Derive a stable seed from the bounded request, independent of process RNG."""

    ids = input_ids.detach().to("cpu", dtype=torch.int64).flatten()
    if not 1 <= ids.numel() <= HARD_MAX_INPUT_TOKENS:
        raise TensorProtocolError("sampling seed input exceeds its token bound")
    if bool((ids < 0).any()):
        raise TensorProtocolError("sampling seed input contains a negative token id")
    if session_digest.shape != (DIGEST_BYTES,):
        raise TensorProtocolError("sampling seed session digest is malformed")
    digest = sha256(b"CogniBoard/conversation-sampling/v1\0")
    digest.update(bytes(map(int, session_digest.tolist())))
    for token in ids.tolist():
        digest.update(int(token).to_bytes(8, "big", signed=False))
    return int.from_bytes(digest.digest()[:8], "big") & NO_DEADLINE_NS


@contextmanager
def _request_sampling_rng(seed: int, device: torch.device) -> Iterator[None]:
    """Seed and restore only the RNGs used by this single-owner worker request."""

    cuda_devices: list[int] = []
    if device.type == "cuda":
        cuda_devices = [
            torch.cuda.current_device() if device.index is None else int(device.index)
        ]
    with torch.random.fork_rng(devices=cuda_devices):
        torch.random.default_generator.manual_seed(seed)
        if cuda_devices:
            with torch.cuda.device(cuda_devices[0]):
                torch.cuda.manual_seed(seed)
        yield


def _worker_authority_tensor(artifact_digest: Tensor) -> Tensor:
    if artifact_digest.shape != (DIGEST_BYTES,):
        raise ValueError("artifact digest tensor has an invalid shape")
    return torch.cat(
        (torch.zeros(2, dtype=torch.int64), artifact_digest.clone())
    ).contiguous()


def _worker_startup_status_tensor() -> Tensor:
    return torch.zeros(1, dtype=torch.int64).share_memory_()


def _record_worker_startup_status(
    status: Tensor | None,
    stage: str,
    *,
    error_code: int = _STARTUP_ERROR_NONE,
) -> None:
    if status is None:
        return
    if (
        not isinstance(status, Tensor)
        or status.device.type != "cpu"
        or status.dtype != torch.int64
        or status.shape != (1,)
        or not status.is_contiguous()
    ):
        raise WorkerStartupError("worker startup status tensor is invalid")
    stage_code = _STARTUP_STAGE_CODES.get(stage)
    if stage_code is None:
        raise WorkerStartupError("worker startup stage is unknown")
    if error_code not in {
        _STARTUP_ERROR_NONE,
        _STARTUP_ERROR_GENERAL,
        _STARTUP_ERROR_CTS_CHECKPOINT,
        _STARTUP_ERROR_GPU_MEMORY,
    }:
        raise WorkerStartupError("worker startup error code is unknown")
    status[0] = (int(stage_code) << 8) | int(error_code)


def _worker_startup_failure_message(status: Tensor | None) -> str:
    if (
        not isinstance(status, Tensor)
        or status.device.type != "cpu"
        or status.dtype != torch.int64
        or status.shape != (1,)
        or not status.is_contiguous()
    ):
        return "local model worker failed to initialize"
    try:
        packed = int(status[0].item())
    except (RuntimeError, TypeError, ValueError):
        return "local model worker failed to initialize"
    if packed < 0:
        return "local model worker failed to initialize"
    stage_code = packed >> 8
    error_code = packed & 0xFF
    if error_code == _STARTUP_ERROR_CTS_CHECKPOINT:
        return "CTS policy checkpoint integrity verification failed"
    if error_code == _STARTUP_ERROR_GPU_MEMORY:
        return "local model worker could not reserve bounded GPU memory"
    stage = _STARTUP_STAGE_LABELS.get(stage_code)
    if stage:
        return f"local model worker failed during {stage}"
    return "local model worker failed to initialize"


def _parse_worker_authority(value: Tensor | None) -> tuple[int, int, Tensor]:
    authority = (
        _worker_authority_tensor(ZERO_ARTIFACT_DIGEST) if value is None else value
    )
    if (
        not isinstance(authority, Tensor)
        or authority.device.type != "cpu"
        or authority.dtype != torch.int64
        or authority.shape != (2 + DIGEST_BYTES,)
        or not authority.is_contiguous()
    ):
        raise WorkerAuthorityError("worker launch authority tensor is invalid")
    epoch = int(authority[0].item())
    deadline = int(authority[1].item())
    artifact = authority[2:].clone()
    if epoch < 0 or deadline < 0:
        raise WorkerAuthorityError("worker launch lease authority is negative")
    if (epoch == 0) != (deadline == 0):
        raise WorkerAuthorityError("worker launch lease authority is inconsistent")
    if bool(((artifact < 0) | (artifact > 255)).any()):
        raise WorkerAuthorityError("worker launch artifact digest is invalid")
    if epoch and monotonic_ns() >= deadline:
        raise WorkerAuthorityError("worker launch GPU lease is already expired")
    return epoch, deadline, artifact


def _response_for(
    authority: _RequestAuthority,
    status: int,
    token_ids: Tensor | None = None,
    *,
    generated_total: int = 0,
    final: bool,
    finish_reason: int | None = None,
) -> TensorMessage:
    return make_response(
        authority.request_id,
        status,
        token_ids,
        generated_total=generated_total,
        final=final,
        finish_reason=finish_reason,
        **authority.response_kwargs(),
    )


@dataclass(frozen=True)
class LocalGemmaModelFactory:
    """Picklable worker-side loader for one verified local Gemma directory."""

    model_path: str
    vram_limit_gib: float = 16.7
    manifest_path: str | None = None
    artifact_digest: str | None = None

    def __post_init__(self) -> None:
        root = verify_local_gemma_path(self.model_path)
        object.__setattr__(self, "model_path", str(root))
        if not 0.0 < float(self.vram_limit_gib) <= 16.7:
            raise ValueError("vram_limit_gib must lie in (0, 16.7]")
        manifest = self.manifest_path
        if manifest is not None:
            manifest_file = Path(manifest).expanduser().resolve(strict=True)
            verified = verify_artifact_manifest(root, manifest_file)
            if verified.root != root:
                raise ValueError("manifest artifact root does not match model_path")
            _verify_instruction_tuned_e4b_snapshot(verified)
            actual_digest = _sha256_file(manifest_file)
            object.__setattr__(self, "manifest_path", str(manifest_file))
        else:
            # Legacy/local test construction has no external manifest. Bind the
            # worker capability to the verified Gemma config identity; product
            # construction supplies a full hash manifest below.
            actual_digest = _sha256_file(root / "config.json")
        supplied = self.artifact_digest
        if (
            supplied is not None
            and _validated_digest_hex(supplied, "artifact_digest") != actual_digest
        ):
            raise ValueError("artifact_digest does not match the verified manifest")
        object.__setattr__(self, "artifact_digest", actual_digest)

    def __call__(self) -> nn.Module:
        _force_offline_environment()
        if self.manifest_path is not None:
            # Re-verify and inventory in the spawned worker immediately before
            # loading. This narrows the parent/child interval but does not replace
            # OS-level read-only controls for adversarial swap-and-restore attacks.
            verified = verify_artifact_manifest(self.model_path, self.manifest_path)
            if verified.root != Path(self.model_path):
                raise RuntimeError("worker artifact root changed after spawn")
            _verify_instruction_tuned_e4b_snapshot(verified)
            if _sha256_file(Path(self.manifest_path)) != self.artifact_digest:
                raise RuntimeError("worker manifest digest changed after spawn")
        model, _tokenizer = load_local_gemma(
            self.model_path,
            vram_limit_gib=self.vram_limit_gib,
        )
        if self.manifest_path is not None:
            # Repeat verification after Transformers returns so changes left in
            # place during loading are detected. A swap restored before this read
            # is outside what pre/post user-space checks can prove.
            verified_after_load = verify_artifact_manifest(
                self.model_path, self.manifest_path
            )
            _verify_instruction_tuned_e4b_snapshot(verified_after_load)
            if _sha256_file(Path(self.manifest_path)) != self.artifact_digest:
                raise RuntimeError("worker manifest changed during model load")
        return model


@dataclass(frozen=True, slots=True)
class _VerifiedMetaControllerV2:
    controller: LearnedCTSController
    tolerance_ceiling: float
    act_hard_floor: int
    max_simulations: int

    @staticmethod
    def _bounded(value: Tensor, bounds: tuple[float, float], label: str) -> float:
        scalar = float(value.detach())
        if not math.isfinite(scalar):
            raise RuntimeError(f"learned CTS {label} became non-finite")
        return min(max(scalar, bounds[0]), bounds[1])

    def __call__(self, root: Tensor) -> SearchControlsV2:
        learned = self.controller.meta_controls(root)
        exploration = self._bounded(
            learned.exploration, EXPLORATION_BOUNDS, "exploration"
        )
        learned_tolerance = self._bounded(
            learned.tolerance, TOLERANCE_BOUNDS, "tolerance"
        )
        # A learned tolerance below the arithmetic resolution can make a
        # mathematically contractive BF16 solve reject every root edge.  Clamp
        # it to a dtype-aware floor while preserving the certified ceiling.
        numeric_floor = min(
            self.tolerance_ceiling,
            max(TOLERANCE_BOUNDS[0], 4.0 * float(torch.finfo(root.dtype).eps)),
        )
        tolerance = min(max(learned_tolerance, numeric_floor), self.tolerance_ceiling)
        temperature = self._bounded(
            learned.temperature, TEMPERATURE_BOUNDS, "temperature"
        )
        learned_act = self._bounded(learned.act, ACT_BOUNDS, "ACT")
        act_simulations = max(
            self.act_hard_floor,
            min(self.max_simulations, int(math.ceil(learned_act))),
        )
        return SearchControlsV2(
            exploration=exploration,
            tolerance=tolerance,
            policy_temperature=temperature,
            act_simulations=act_simulations,
        )


def _production_cts_plan(
    hidden_size: int,
    *,
    max_iter: int,
    history: int,
) -> tuple[CertifiedPUCTConfigV2, int]:
    """Derive non-zero callback MAC certificates from the actual latent width."""

    if hidden_size <= 0 or max_iter < 17 or history != 16:
        raise ValueError("production CTS requires a positive latent and rank-16 solve")
    summary_macs = hidden_size * SUMMARY_DIM
    meta_policy_macs = summary_macs + SUMMARY_DIM * META_CONTROL_DIM
    action_policy_macs = summary_macs + SUMMARY_DIM * ACTION_WIDTH
    critic_macs = summary_macs + SUMMARY_DIM
    retrieval_macs = PRODUCTION_CTS_NODE_CAPACITY * hidden_size
    # One transition callback solves all three actions. The multiplier covers
    # residual evaluation plus the rank-16 multisecant dot/update work.
    transition_macs = ACTION_WIDTH * max_iter * hidden_size * (2 * history + 8)
    config = CertifiedPUCTConfigV2(
        width=ACTION_WIDTH,
        max_nodes=PRODUCTION_CTS_NODE_CAPACITY,
        max_depth=PRODUCTION_CTS_DEPTH,
        simulations=PRODUCTION_CTS_ACT_HARD_FLOOR,
        meta_policy_macs=meta_policy_macs,
        action_policy_macs=action_policy_macs,
        critic_macs=critic_macs,
        retrieval_macs=retrieval_macs,
        transition_macs=transition_macs,
    )
    per_simulation = action_policy_macs + critic_macs + retrieval_macs + transition_macs
    request_budget = meta_policy_macs + PRODUCTION_CTS_ACT_HARD_FLOOR * per_simulation
    return config, request_budget


def _verified_production_controller(model: nn.Module) -> LearnedCTSController:
    controller = load_default_bounded_cts_controller(device=_model_device(model))
    if not isinstance(controller, LearnedCTSController):
        raise CTSCheckpointError("production CTS loader returned an unknown controller")
    if controller.checkpoint_sha256 != DEFAULT_CHECKPOINT_SHA256:
        raise CTSCheckpointError(
            "production CTS checkpoint is not the bundled generation"
        )
    if controller.training or any(
        parameter.requires_grad for parameter in controller.parameters()
    ):
        raise CTSCheckpointError("production CTS controller is not frozen")
    if controller.device != _model_device(model):
        raise CTSCheckpointError("production CTS controller crossed the model device")
    return controller


@dataclass(frozen=True)
class LocalGemmaCorePipelineFactory:
    """Build the production answer-conditioned Cogni-Core worker path."""

    transition_contraction: float = 0.4
    spectral_margin: float = 0.95
    tolerance: float = 5.0e-3
    max_iter: int = 32
    history: int = 16
    fallback_steps: int = 32
    contextual_tokens: bool = True
    max_abs_logit_bias: float = 0.05
    # A bounded sampled signature is a useful runtime tripwire, but it cannot
    # prove that every unsampled model value remained immutable. Product
    # generation must therefore fail closed on every Cogni-Core failure.
    gemma_lkg_fallback: bool = False

    def __post_init__(self) -> None:
        if not 0.0 < self.transition_contraction < self.spectral_margin < 1.0:
            raise ValueError("DEQ contraction must remain below the spectral margin")
        for name in ("max_iter", "history", "fallback_steps"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.max_iter < 17:
            raise ValueError("production rank-16 Broyden requires max_iter >= 17")
        if self.history != 16:
            raise ValueError("production Broyden history must remain rank 16")
        if not TOLERANCE_BOUNDS[0] <= self.tolerance <= TOLERANCE_BOUNDS[1]:
            raise ValueError("DEQ tolerance lies outside the learned safety bounds")
        if not isinstance(self.contextual_tokens, bool):
            raise TypeError("contextual_tokens must be bool")
        if not self.contextual_tokens:
            raise ValueError(
                "the production Cogni-Core path requires contextual_tokens=True"
            )
        if (
            not isinstance(self.max_abs_logit_bias, (int, float))
            or isinstance(self.max_abs_logit_bias, bool)
            or not math.isfinite(float(self.max_abs_logit_bias))
            or not 0.0 < float(self.max_abs_logit_bias) <= 0.1
        ):
            raise ValueError("max_abs_logit_bias must be finite and lie in (0, 0.1]")
        if not isinstance(self.gemma_lkg_fallback, bool):
            raise TypeError("gemma_lkg_fallback must be bool")
        if self.gemma_lkg_fallback:
            raise ValueError(
                "Gemma LKG fallback is disabled until full base-model "
                "immutability can be attested"
            )

    def __call__(self, model: nn.Module) -> CoreTurnPipeline:
        hidden_size = _hidden_size(model)
        controller = _verified_production_controller(model)
        runtime = build_genesis_runtime(
            LocalGemmaFeatureBackbone(
                model,
                contextual_tokens=self.contextual_tokens,
            ),
            load_config(),
            input_dim=hidden_size,
            state_dim=64,
        )
        search_config, request_mac_budget = _production_cts_plan(
            hidden_size,
            max_iter=self.max_iter,
            history=self.history,
        )
        runtime.install_certified_search_v2(
            BoundedPUCTSearchV2(search_config),
            request_mac_budget=request_mac_budget,
        )
        transition = CertifiedBroydenTransitionV2(
            contraction=self.transition_contraction,
            spectral_margin=self.spectral_margin,
            max_iter=self.max_iter,
            operator_id="local-gemma-certified-broyden-v2-rank16",
        )
        collaborators = SearchCollaboratorsV2(
            action_policy=controller.policy_logits,
            critic=controller.critic,
            meta_controller=_VerifiedMetaControllerV2(
                controller=controller,
                tolerance_ceiling=self.tolerance,
                act_hard_floor=PRODUCTION_CTS_ACT_HARD_FLOOR,
                max_simulations=search_config.simulations,
            ),
        )
        return CoreTurnPipeline(runtime, transition, collaborators)


def _strict_local_options(options: Mapping[str, Any] | None) -> dict[str, Any]:
    result = dict(options or {})
    if result.get("local_files_only") is False:
        raise ValueError("local_files_only=False violates the offline policy")
    if result.get("trust_remote_code") is True:
        raise ValueError("trust_remote_code=True violates the offline policy")
    if result.get("force_download") is True:
        raise ValueError("force_download=True violates the offline policy")
    result.update(
        local_files_only=True,
        trust_remote_code=False,
        force_download=False,
    )
    return result


def load_local_tokenizer(
    model_path: str | Path,
    *,
    tokenizer_class: Any | None = None,
    tokenizer_kwargs: Mapping[str, Any] | None = None,
) -> Any:
    """Load only a verified local tokenizer; ``transformers`` stays lazy."""

    root = verify_local_gemma_path(model_path)
    _force_offline_environment()
    if tokenizer_class is None:
        try:
            from transformers import AutoTokenizer
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise RuntimeError("transformers is required for a real tokenizer") from exc
        tokenizer_class = AutoTokenizer
    tokenizer = tokenizer_class.from_pretrained(
        str(root), **_strict_local_options(tokenizer_kwargs)
    )
    if not callable(tokenizer) or not callable(getattr(tokenizer, "decode", None)):
        raise TypeError("local tokenizer must be callable and provide decode()")
    return tokenizer


def _force_offline_environment() -> None:
    os.environ.update(
        {
            "HF_HUB_OFFLINE": "1",
            "HF_HUB_DISABLE_TELEMETRY": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "HF_DATASETS_OFFLINE": "1",
            "WANDB_MODE": "offline",
            "TOKENIZERS_PARALLELISM": "false",
        }
    )


def _hidden_size(model: nn.Module) -> int:
    config = getattr(model, "config", None)
    text_config = getattr(config, "text_config", None)
    value = getattr(text_config, "hidden_size", None)
    if value is None:
        value = getattr(config, "hidden_size", None)
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise RuntimeError("could not determine the local Gemma hidden size")
    return value


def _evenly_spaced_indices(length: int, limit: int) -> tuple[int, ...]:
    if length <= limit:
        return tuple(range(length))
    if limit == 1:
        return (0,)
    return tuple(index * (length - 1) // (limit - 1) for index in range(limit))


def _sampled_content_fingerprint(value: Tensor) -> str:
    """Hash a fixed number of logical tensor values without a full clone."""

    if value.layout != torch.strided or value.device.type == "meta":
        raise BaseModelMutationError(
            "base-model fingerprint requires materialized strided tensors"
        )
    digest = sha256()
    digest.update(str(value.dtype).encode("ascii"))
    digest.update(repr(tuple(value.shape)).encode("ascii"))
    if value.numel() == 0:
        return digest.hexdigest()
    positions = _evenly_spaced_indices(
        int(value.numel()),
        CONTENT_FINGERPRINT_VALUES_PER_TENSOR,
    )
    linear = torch.tensor(positions, device=value.device, dtype=torch.int64)
    coordinates: list[Tensor] = [linear] * value.ndim
    remainder = linear
    for dimension in range(value.ndim - 1, -1, -1):
        size = int(value.shape[dimension])
        coordinates[dimension] = remainder.remainder(size)
        remainder = torch.div(remainder, size, rounding_mode="floor")
    try:
        sampled = value.detach()[tuple(coordinates)]
        # A sampled scalar remains zero-dimensional.  PyTorch cannot reinterpret
        # a zero-dimensional multi-byte tensor (notably bfloat16 model buffers)
        # as bytes, so give the storage an explicit logical dimension first.
        encoded = sampled.to("cpu").contiguous().reshape(-1).view(torch.uint8).flatten()
    except (RuntimeError, TypeError, ValueError) as exc:
        raise BaseModelMutationError(
            "base-model content fingerprint could not sample a tensor"
        ) from exc
    digest.update(bytes(encoded.tolist()))
    return digest.hexdigest()


def _parameter_signature(model: nn.Module) -> tuple[tuple[Any, ...], ...]:
    """Return a bounded structural and sampled-value base-model signature.

    Every parameter, buffer, and module contributes identity/metadata, version,
    train/eval, and requires-grad state. Content hashing is deliberately capped
    at 128 evenly distributed tensors and 16 values per selected tensor, so an
    8B model is never cloned or fully transferred to CPU. This catches broad
    ``.data`` writes and all mutations touching sampled values, but it is not a
    cryptographic proof for unsampled elements. Startup manifest verification
    remains the full artifact-integrity authority; this is a bounded runtime
    tamper tripwire for the trusted resident model.
    """

    modules = tuple(model.named_modules())
    tensors = tuple(
        ("parameter", name, value) for name, value in model.named_parameters()
    ) + tuple(("buffer", name, value) for name, value in model.named_buffers())
    if len(modules) > MAX_SIGNATURE_MODULES:
        raise BaseModelMutationError("base model exceeds the module signature bound")
    if len(tensors) > MAX_SIGNATURE_TENSORS:
        raise BaseModelMutationError("base model exceeds the tensor signature bound")
    sampled = set(
        _evenly_spaced_indices(len(tensors), CONTENT_FINGERPRINT_TENSOR_BUDGET)
    )
    signature: list[tuple[Any, ...]] = [
        ("model", id(model), bool(model.training), len(modules), len(tensors))
    ]
    signature.extend(
        ("module", name, id(module), bool(module.training)) for name, module in modules
    )
    for index, (kind, name, value) in enumerate(tensors):
        signature.append(
            (
                kind,
                name,
                id(value),
                value.data_ptr(),
                int(value._version),
                tuple(value.shape),
                value.dtype,
                value.device.type,
                value.device.index,
                bool(value.requires_grad) if kind == "parameter" else None,
                _sampled_content_fingerprint(value) if index in sampled else None,
            )
        )
    return tuple(signature)


def _prepare_base_model(model: nn.Module) -> tuple[tuple[Any, ...], ...]:
    if not isinstance(model, nn.Module):
        raise TypeError("model factory must return torch.nn.Module")
    if not callable(getattr(model, "generate", None)):
        raise TypeError("model must expose generate()")
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    config = getattr(model, "config", None)
    if config is not None and hasattr(config, "use_cache"):
        config.use_cache = False
    return _parameter_signature(model)


def _model_device(model: nn.Module) -> torch.device:
    device = getattr(model, "device", None)
    if device is not None:
        try:
            return torch.device(device)
        except (TypeError, RuntimeError):
            pass
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


class _TokenRepetitionGuard:
    """Detect exact generated-token cycles in bounded CPU memory.

    Semantic similarity is deliberately out of scope: only an immediately
    repeated token period can stop generation. Longer periods need two copies;
    shorter periods need enough copies to cover at least 48 tokens. This keeps
    ordinary emphasis and short list patterns from being treated as loops.
    """

    window_tokens = 384
    min_period_tokens = 8
    long_period_tokens = 24
    max_period_tokens = 192
    min_repeat_span_tokens = 48
    min_unique_tokens = 4
    max_identical_run_tokens = 24
    prompt_match_tokens = 24
    leading_prompt_echo_allowance = 64

    def __init__(self, prompt_ids: Tensor | None = None) -> None:
        self._tokens: deque[int] = deque(maxlen=self.window_tokens)
        self.triggered = False
        self.repeat_cut_index: int | None = None
        self.repeat_period: int | None = None
        self.trigger_reason: str | None = None
        self._observed = 0
        prompt = (
            torch.empty(0, dtype=torch.int64)
            if prompt_ids is None
            else torch.as_tensor(prompt_ids)
            .detach()
            .to("cpu", dtype=torch.int64)
            .flatten()
        )
        values = tuple(map(int, prompt.tolist()))
        width = self.prompt_match_tokens
        self._prompt_ngrams = frozenset(
            window
            for index in range(max(0, len(values) - width + 1))
            if len(set(window := values[index : index + width]))
            >= self.min_unique_tokens
        )

    def observe(self, token_ids: Tensor) -> bool:
        if self.triggered:
            return True
        values = torch.as_tensor(token_ids).detach().to("cpu", dtype=torch.int64)
        for value in values.flatten().tolist():
            self._tokens.append(int(value))
            self._observed += 1
            identical_run_start = self._identical_run_start()
            if identical_run_start is not None:
                self.triggered = True
                self.trigger_reason = "identical_token_run"
                self.repeat_cut_index = identical_run_start
                break
            prompt_echo_start = self._prompt_echo_start()
            if (
                prompt_echo_start is not None
                and prompt_echo_start >= self.leading_prompt_echo_allowance
            ):
                self.triggered = True
                self.trigger_reason = "prompt_echo"
                self.repeat_cut_index = prompt_echo_start
                break
            match = self._repeated_suffix()
            if match is not None:
                period, copies = match
                self.triggered = True
                self.repeat_period = period
                self.trigger_reason = "token_cycle"
                self.repeat_cut_index = self._observed - period * (copies - 1)
                break
        return self.triggered

    def _identical_run_start(self) -> int | None:
        if len(self._tokens) < self.max_identical_run_tokens:
            return None
        values = tuple(self._tokens)
        last = values[-1]
        run = 1
        for previous in reversed(values[:-1]):
            if previous != last:
                break
            run += 1
        if run < self.max_identical_run_tokens:
            return None
        return self._observed - run

    def _prompt_echo_start(self) -> int | None:
        if not self._prompt_ngrams or len(self._tokens) < self.prompt_match_tokens:
            return None
        suffix = tuple(self._tokens)[-self.prompt_match_tokens :]
        if suffix not in self._prompt_ngrams:
            return None
        return self._observed - self.prompt_match_tokens

    def _repeated_suffix(self) -> tuple[int, int] | None:
        values = tuple(self._tokens)
        maximum = min(self.max_period_tokens, len(values) // 2)
        for period in range(maximum, self.min_period_tokens - 1, -1):
            copies = (
                2
                if period >= self.long_period_tokens
                else max(3, math.ceil(self.min_repeat_span_tokens / period))
            )
            span = period * copies
            if span > len(values):
                continue
            pattern = values[-period:]
            if len(set(pattern)) < self.min_unique_tokens:
                continue
            start = len(values) - span
            if all(
                values[start + index * period : start + (index + 1) * period] == pattern
                for index in range(copies)
            ):
                return period, copies
        return None


def truncate_repeated_tokens(token_ids: Tensor) -> tuple[Tensor, bool]:
    """Keep the first copy of a bounded exact token cycle."""

    tokens = torch.as_tensor(token_ids).detach().to("cpu", dtype=torch.int64).flatten()
    guard = _TokenRepetitionGuard()
    if not guard.observe(tokens):
        return tokens.contiguous(), False
    if (
        guard.repeat_cut_index is None
        or not 0 <= guard.repeat_cut_index <= tokens.numel()
    ):
        raise TensorProtocolError("repetition guard produced an invalid cut point")
    return tokens[: guard.repeat_cut_index].contiguous(), True


class _GenerationStoppingCriteria:
    def __init__(
        self,
        cancel_event: Any,
        stop_token_ids: Tensor,
        stop_sequences: Tensor | None = None,
        repetition_guard: _TokenRepetitionGuard | None = None,
        request_deadline_ns: int = 2**63 - 1,
        lease_deadline_ns: int = 0,
    ) -> None:
        self.cancel_event = cancel_event
        self.stop_token_ids = tuple(map(int, stop_token_ids.tolist()))
        sequences = (
            torch.empty((0, 0), dtype=torch.int64)
            if stop_sequences is None
            else torch.as_tensor(stop_sequences, dtype=torch.int64)
        )
        self.stop_sequences = tuple(
            tuple(int(value) for value in row.tolist() if int(value) >= 0)
            for row in sequences
        )
        self.repetition_guard = repetition_guard
        self.request_deadline_ns = int(request_deadline_ns)
        self.lease_deadline_ns = int(lease_deadline_ns)
        self.cancel_observed = False
        self.deadline_observed = False
        self.stop_observed = False
        self.repetition_observed = False

    def __call__(self, input_ids: Tensor, _scores: Any, **_kwargs: Any) -> Tensor:
        batch = int(input_ids.shape[0]) if input_ids.ndim else 1
        now = monotonic_ns()
        deadline_now = now >= self.request_deadline_ns or bool(
            self.lease_deadline_ns and now >= self.lease_deadline_ns
        )
        self.deadline_observed = self.deadline_observed or deadline_now
        cancelled_now = bool(self.cancel_event.is_set()) or deadline_now
        self.cancel_observed = self.cancel_observed or cancelled_now
        cancelled = torch.full(
            (batch,),
            cancelled_now,
            dtype=torch.bool,
            device=input_ids.device,
        )
        if input_ids.ndim != 2 or input_ids.shape[1] == 0:
            return cancelled
        reached_stop = torch.zeros((batch,), dtype=torch.bool, device=input_ids.device)
        if self.stop_token_ids:
            stop_ids = torch.tensor(
                self.stop_token_ids, dtype=input_ids.dtype, device=input_ids.device
            )
            reached_stop |= (input_ids[:, -1, None] == stop_ids[None, :]).any(dim=1)
        for sequence in self.stop_sequences:
            if not sequence or input_ids.shape[1] < len(sequence):
                continue
            expected = torch.tensor(
                sequence, dtype=input_ids.dtype, device=input_ids.device
            )
            reached_stop |= (input_ids[:, -len(sequence) :] == expected).all(dim=1)
        repeated_now = bool(
            self.repetition_guard is not None and self.repetition_guard.triggered
        )
        self.repetition_observed = self.repetition_observed or repeated_now
        if repeated_now:
            reached_stop |= torch.ones_like(reached_stop)
        self.stop_observed = self.stop_observed or bool(reached_stop.any())
        return cancelled | reached_stop


def _stopping_criteria(criteria: _GenerationStoppingCriteria) -> Any:
    # Transformers accepts a list-like custom criteria collection and merges
    # it into its own StoppingCriteriaList. Keeping this object dependency-free
    # avoids importing the optional runtime for injected/fake model workers.
    return [criteria]


class _TensorResponseStreamer:
    """Convert model streamer callbacks directly into bounded token tensors."""

    def __init__(
        self,
        response_queue: Any,
        authority: _RequestAuthority,
        prompt_ids: Tensor,
        max_new_tokens: int,
        cancel_event: Any,
        repetition_guard: _TokenRepetitionGuard | None = None,
    ) -> None:
        self.response_queue = response_queue
        self.authority = authority
        self.request_id = authority.request_id
        self.prompt_ids = prompt_ids.detach().to("cpu", dtype=torch.int64).flatten()
        self.max_new_tokens = max_new_tokens
        self.cancel_event = cancel_event
        self.repetition_guard = repetition_guard or _TokenRepetitionGuard()
        self.emitted: list[int] = []
        self._prompt_checked = False

    def put(self, value: Any) -> None:
        if self.cancel_event.is_set():
            return
        now = monotonic_ns()
        if now >= self.authority.request_deadline_ns or bool(
            self.authority.lease_deadline_ns and now >= self.authority.lease_deadline_ns
        ):
            raise CoreTurnAuthorityError("generation authority expired while streaming")
        tensor = torch.as_tensor(value).detach()
        if tensor.ndim > 2 or (tensor.ndim == 2 and tensor.shape[0] != 1):
            raise TensorProtocolError("streamer only accepts batch-one token ids")
        tokens = tensor.to("cpu", dtype=torch.int64).flatten().contiguous()
        if not self._prompt_checked:
            self._prompt_checked = True
            if tokens.shape == self.prompt_ids.shape and torch.equal(
                tokens, self.prompt_ids
            ):
                return
        if not tokens.numel():
            return
        if (
            bool((tokens < 0).any())
            or len(self.emitted) + tokens.numel() > self.max_new_tokens
        ):
            raise TensorProtocolError(
                "model streamer exceeded the response token bound"
            )
        self.emitted.extend(map(int, tokens.tolist()))
        self.repetition_guard.observe(tokens)
        _queue_put(
            self.response_queue,
            _response_for(
                self.authority,
                STATUS_OK,
                tokens,
                generated_total=len(self.emitted),
                final=False,
            ),
        )

    def end(self) -> None:
        return


def _generated_suffix(output: Any, prompt_ids: Tensor) -> Tensor:
    sequences = getattr(output, "sequences", output)
    if (
        not isinstance(sequences, Tensor)
        or sequences.ndim != 2
        or sequences.shape[0] != 1
    ):
        raise TensorProtocolError("model returned an invalid batch-one token sequence")
    sequence = sequences.detach().to("cpu", dtype=torch.int64).flatten().contiguous()
    prompt = prompt_ids.detach().to("cpu", dtype=torch.int64).flatten()
    if sequence.numel() >= prompt.numel() and torch.equal(
        sequence[: prompt.numel()], prompt
    ):
        return sequence[prompt.numel() :].contiguous()
    return sequence


def _queue_put(queue: Any, message: TensorMessage) -> None:
    try:
        queue.put(message, timeout=2.0)
    except Full as exc:
        raise RuntimeError("bounded response queue is saturated") from exc


def _debug_worker_error(stage: str, error: BaseException) -> None:
    if os.environ.get("COGNI_AGENT_DEBUG") != "1":
        return
    detail = " ".join(f"{type(error).__name__}: {error}".replace("\x00", "").split())
    print(f"[cogni-worker:{stage}] {detail[:256]}", file=sys.stderr, flush=True)


def _cognitive_state(
    input_ids: Tensor, attention_mask: Tensor, sequence_budget: int
) -> Tensor:
    """Derive bounded BIO-HAMA telemetry from token tensors only."""

    mask = attention_mask.to(dtype=torch.float32)
    active = mask.sum(dim=1)
    load = (active / float(sequence_budget)).clamp(0.0, 1.0)
    attention = mask.mean(dim=1).clamp(0.0, 1.0)
    if input_ids.shape[1] > 1:
        pair_mask = mask[:, 1:] * mask[:, :-1]
        changes = (input_ids[:, 1:] != input_ids[:, :-1]).to(mask.dtype) * pair_mask
        uncertainty = (changes.sum(dim=1) / pair_mask.sum(dim=1).clamp_min(1.0)).clamp(
            0.0, 1.0
        )
    else:
        uncertainty = torch.zeros_like(load)
    memory = load
    affect = torch.zeros_like(load)
    return cognitive_state_tensor(memory, affect, attention, uncertainty, load)


def _run_core_turn(
    pipeline: CoreTurnPipeline,
    input_ids: Tensor,
    attention_mask: Tensor,
    *,
    sequence_budget: int,
    estimated_workspace_bytes: int,
    authority: _RequestAuthority,
) -> CoreTurnResult:
    result = pipeline.run(
        CoreTurnRequest(
            inputs=input_ids,
            cognitive_state=_cognitive_state(
                input_ids, attention_mask, sequence_budget
            ),
            backbone_kwargs={"attention_mask": attention_mask},
            # Untrained Fast Weight remains inactive. Admission, external
            # quality, and OOD calibration are mandatory before a later turn
            # can supply either FastWeight field.
            fast_weight=None,
            compile_fast_weight=None,
            # Only the digest crosses IPC. Its stable hexadecimal encoding is
            # a valid bounded runtime key and keeps System-4 warm state isolated
            # per conversation without exposing natural-language metadata.
            swarm_session_id=bytes(
                int(value) for value in authority.session_digest.tolist()
            ).hex(),
            estimated_workspace_bytes=estimated_workspace_bytes,
            request_id=authority.request_id,
            job_id=authority.job_id,
            lease_epoch=authority.lease_epoch,
            request_deadline_ns=authority.request_deadline_ns,
            lease_deadline_ns=authority.lease_deadline_ns,
            artifact_digest=authority.artifact_digest,
        )
    )
    telemetry = getattr(result, "telemetry", None)
    if getattr(telemetry, "advisory_only", None) is not True:
        raise RuntimeError("Cogni-Core auxiliaries crossed the advisory boundary")
    return result


def _required_product_logits_processor(
    model: nn.Module,
    result: object,
    *,
    max_abs_logit_bias: float,
) -> object:
    """Build the mandatory production processor from the certified CTS result."""

    if not isinstance(result, CoreTurnResult):
        raise RuntimeError("production Cogni-Core returned no CoreTurnResult")
    inference = getattr(result, "inference", None)
    search = getattr(inference, "search", None)
    telemetry = getattr(search, "telemetry", None)
    v2_safety = (
        getattr(telemetry, "safe_for_decode", None),
        getattr(telemetry, "linear_solve_fallbacks", None),
        getattr(telemetry, "unsafe_silent_fallbacks", None),
    )
    if any(value is not None for value in v2_safety):
        safe_for_decode, linear_fallbacks, silent_fallbacks = v2_safety
        if (
            safe_for_decode is not True
            or not isinstance(linear_fallbacks, int)
            or isinstance(linear_fallbacks, bool)
            or linear_fallbacks != 0
            or not isinstance(silent_fallbacks, int)
            or isinstance(silent_fallbacks, bool)
            or silent_fallbacks != 0
        ):
            raise RuntimeError(
                "production conditioning rejected unsafe CTS V2 telemetry"
            )
    best_state = getattr(search, "best_state", None)
    if not isinstance(best_state, Tensor):
        raise RuntimeError("production Cogni-Core returned no search.best_state")
    if best_state.ndim != 2 or best_state.shape[0] != 1:
        raise RuntimeError("production search.best_state must have shape [1, H]")
    if (
        not torch.is_floating_point(best_state)
        or best_state.shape[1] == 0
        or not bool(torch.isfinite(best_state).all())
    ):
        raise RuntimeError("production search.best_state must be finite floating data")
    return build_latent_logits_processor(
        model,
        best_state.detach(),
        max_abs_bias=float(max_abs_logit_bias),
    )


def _validate_worker_request_authority(
    authority: _RequestAuthority,
    *,
    expected_lease_epoch: int,
    expected_lease_deadline_ns: int,
    expected_artifact_digest: Tensor,
    last_request_id: int,
    seen_job_ids: set[int],
) -> None:
    now = monotonic_ns()
    if authority.request_id <= last_request_id or authority.job_id in seen_job_ids:
        raise WorkerAuthorityError("worker rejected a replayed request/job id")
    if authority.lease_epoch != expected_lease_epoch:
        raise WorkerAuthorityError("worker rejected a stale GPU lease epoch")
    if authority.lease_deadline_ns != expected_lease_deadline_ns:
        raise WorkerAuthorityError("worker rejected a changed GPU lease deadline")
    if not torch.equal(authority.artifact_digest, expected_artifact_digest):
        raise WorkerAuthorityError("worker rejected a stale artifact digest")
    if not bool(authority.session_digest.count_nonzero()):
        raise WorkerAuthorityError("worker rejected an empty conversation digest")
    if now >= authority.request_deadline_ns:
        raise WorkerAuthorityError("worker rejected an expired request deadline")
    if expected_lease_epoch:
        if now >= expected_lease_deadline_ns:
            raise WorkerAuthorityError("worker rejected an expired GPU lease")
        if authority.request_deadline_ns > expected_lease_deadline_ns:
            raise WorkerAuthorityError("request authority exceeds its GPU lease")


def _await_retirement_ack(retire_ack_event: Any) -> None:
    """Keep terminal tensor storage alive until its parent reconstructs it."""

    if retire_ack_event is None:
        return
    try:
        retire_ack_event.wait(timeout=WORKER_RETIRE_ACK_TIMEOUT_SECONDS)
    except BaseException:
        # Retirement is still bounded and unconditional when IPC is broken.
        pass


def _worker_main(
    model_factory: Callable[[], nn.Module],
    core_pipeline_factory: Callable[[nn.Module], CoreTurnPipeline] | None,
    request_queue: Any,
    response_queue: Any,
    cancel_event: Any,
    ready_event: Any,
    failed_event: Any,
    reserved_sequences: Tensor,
    max_input_tokens: int,
    core_workspace_bytes: int,
    worker_authority: Tensor | None = None,
    startup_status: Tensor | None = None,
    retire_ack_event: Any = None,
) -> None:
    stage = "worker_startup"
    try:
        _force_offline_environment()
        _record_worker_startup_status(startup_status, stage)
        stage = "worker_authority"
        _record_worker_startup_status(startup_status, stage)
        expected_epoch, expected_lease_deadline, expected_artifact = (
            _parse_worker_authority(worker_authority)
        )
        stage = "model_factory"
        _record_worker_startup_status(startup_status, stage)
        model = model_factory()
        stage = "prepare_base_model"
        _record_worker_startup_status(startup_status, stage)
        _prepare_base_model(model)
        stage = "core_pipeline_factory"
        _record_worker_startup_status(startup_status, stage)
        pipeline = (
            None if core_pipeline_factory is None else core_pipeline_factory(model)
        )
        if pipeline is not None and not callable(getattr(pipeline, "run", None)):
            raise TypeError("core pipeline factory must return an object with run()")
        stage = "base_signature"
        _record_worker_startup_status(startup_status, stage)
        base_signature = _parameter_signature(model)
        stage = "model_device"
        _record_worker_startup_status(startup_status, stage)
        device = _model_device(model)
        stage = "product_factory_check"
        _record_worker_startup_status(startup_status, stage)
        product_factory = (
            core_pipeline_factory
            if isinstance(core_pipeline_factory, LocalGemmaCorePipelineFactory)
            else None
        )
        # Model loading can consume a material part of the lease. Recheck the
        # immutable launch fence before advertising readiness.
        if expected_epoch and monotonic_ns() >= expected_lease_deadline:
            raise WorkerAuthorityError("GPU lease expired during model loading")
        stage = "ready"
        _record_worker_startup_status(startup_status, stage)
        ready_event.set()
    except BaseException as error:
        error_code = _STARTUP_ERROR_GENERAL
        if isinstance(error, CTSCheckpointError):
            error_code = _STARTUP_ERROR_CTS_CHECKPOINT
        elif isinstance(error, getattr(torch, "OutOfMemoryError", ())):
            error_code = _STARTUP_ERROR_GPU_MEMORY
        try:
            _record_worker_startup_status(
                startup_status,
                stage,
                error_code=error_code,
            )
        except BaseException:
            pass
        try:
            _debug_worker_error(stage, error)
        except BaseException:
            pass
        try:
            failed_event.set()
        except BaseException:
            pass
        return
    last_request_id = 0
    seen_job_ids: set[int] = set()

    while True:
        try:
            raw = request_queue.get()
            request = parse_request(raw)
        except BaseException:
            try:
                _queue_put(
                    response_queue,
                    make_response(0, STATUS_INVALID_REQUEST, final=True),
                )
            except BaseException:
                return
            continue
        if request is None:
            return

        authority = _RequestAuthority.from_request(request)
        try:
            _validate_worker_request_authority(
                authority,
                expected_lease_epoch=expected_epoch,
                expected_lease_deadline_ns=expected_lease_deadline,
                expected_artifact_digest=expected_artifact,
                last_request_id=last_request_id,
                seen_job_ids=seen_job_ids,
            )
        except WorkerAuthorityError as error:
            _debug_worker_error("request_authority", error)
            try:
                _queue_put(
                    response_queue,
                    _response_for(
                        authority,
                        STATUS_AUTHORITY_REJECTED,
                        final=True,
                        finish_reason=FINISH_ERROR,
                    ),
                )
            finally:
                # A stale/expired launch fence means this resident must not
                # continue owning CUDA, even if the request itself was forged.
                _await_retirement_ack(retire_ack_event)
                return
        last_request_id = authority.request_id
        seen_job_ids.add(authority.job_id)
        if len(seen_job_ids) > 4_096:
            # Request ids are strictly monotonic; old job ids cannot replay
            # without also failing the request-id fence.
            seen_job_ids.clear()
            seen_job_ids.add(authority.job_id)

        repetition_guard = _TokenRepetitionGuard(request.input_ids)
        streamer = _TensorResponseStreamer(
            response_queue,
            authority,
            request.input_ids,
            request.max_new_tokens,
            cancel_event,
            repetition_guard,
        )
        status = STATUS_OK
        finish_reason = FINISH_ERROR
        remaining = torch.empty(0, dtype=torch.int64)
        stop_criteria = _GenerationStoppingCriteria(
            cancel_event,
            request.stop_token_ids,
            reserved_sequences,
            repetition_guard,
            request_deadline_ns=authority.request_deadline_ns,
            lease_deadline_ns=authority.lease_deadline_ns,
        )
        stage = "request"
        try:
            input_ids = request.input_ids.to(device)
            attention_mask = request.attention_mask.to(device)
            multimodal_generation_kwargs: dict[str, Tensor] = {}
            if request.image_tensors:
                multimodal_generation_kwargs = _image_generation_kwargs(
                    request.image_tensors,
                    device,
                )
            elif request.audio_tensors:
                multimodal_generation_kwargs = _audio_generation_kwargs(
                    request.audio_tensors,
                    device,
                )
            logits_processor = None
            if pipeline is not None:
                stage = "core_pipeline"
                if cancel_event.is_set():
                    status = STATUS_CANCELLED
                else:
                    try:
                        # CTS guards input mutation through Tensor._version; unlike
                        # inference_mode, no_grad preserves that safety counter.
                        with torch.no_grad():
                            core_result = _run_core_turn(
                                pipeline,
                                input_ids,
                                attention_mask,
                                sequence_budget=max_input_tokens,
                                estimated_workspace_bytes=core_workspace_bytes,
                                authority=authority,
                            )
                        if product_factory is not None:
                            stage = "answer_conditioning"
                            logits_processor = _required_product_logits_processor(
                                model,
                                core_result,
                                max_abs_logit_bias=product_factory.max_abs_logit_bias,
                            )
                    except CoreTurnAuthorityError:
                        raise
                    except Exception:
                        # Startup manifest verification proves the on-disk
                        # artifact. The bounded runtime signature below is only
                        # a tamper tripwire, so it cannot authorize a base answer
                        # after an arbitrary Cogni-Core failure.
                        raise
                    if cancel_event.is_set():
                        status = STATUS_CANCELLED
            if status == STATUS_CANCELLED:
                output = input_ids
            else:
                stage = "base_generate"
                with torch.inference_mode():
                    generation_options = dict(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        max_new_tokens=request.max_new_tokens,
                        num_beams=1,
                        num_return_sequences=1,
                        use_cache=False,
                        streamer=streamer,
                        stopping_criteria=_stopping_criteria(stop_criteria),
                    )
                    # Only the fixed tensor schema reconstructed by protocol.py
                    # reaches generate(); raw image bytes, paths, strings and
                    # JSON never enter the resident model process.
                    generation_options.update(multimodal_generation_kwargs)
                    if request.decode_mode == DECODE_CONVERSATION:
                        generation_options.update(
                            do_sample=True,
                            temperature=CONVERSATION_TEMPERATURE,
                            top_p=CONVERSATION_TOP_P,
                            top_k=CONVERSATION_TOP_K,
                        )
                        rng_context = _request_sampling_rng(
                            request.sampling_seed, device
                        )
                    elif request.decode_mode == DECODE_STRICT:
                        generation_options["do_sample"] = False
                        rng_context = nullcontext()
                    else:  # Protocol parsing should already make this unreachable.
                        raise TensorProtocolError("worker decode mode is unsupported")
                    if logits_processor is not None:
                        generation_options["logits_processor"] = [logits_processor]
                    with rng_context:
                        output = model.generate(**generation_options)
            stage = "token_postcheck"
            generated = _generated_suffix(output, request.input_ids)
            if generated.numel() > request.max_new_tokens:
                raise TensorProtocolError("model exceeded max_new_tokens")
            emitted = torch.tensor(streamer.emitted, dtype=torch.int64)
            if emitted.numel():
                if generated.numel() < emitted.numel() or not torch.equal(
                    generated[: emitted.numel()], emitted
                ):
                    raise TensorProtocolError("streamed and returned tokens disagree")
                remaining = generated[emitted.numel() :].contiguous()
            else:
                remaining = generated
            if stop_criteria.deadline_observed:
                status = STATUS_DEADLINE_EXCEEDED
                finish_reason = FINISH_ERROR
                remaining = torch.empty(0, dtype=torch.int64)
            elif status == STATUS_CANCELLED or stop_criteria.cancel_observed:
                status = STATUS_CANCELLED
                finish_reason = FINISH_CANCELLED
            else:
                status = STATUS_OK
                stopped_by_token = bool(
                    generated.numel()
                    and request.stop_token_ids.numel()
                    and bool((generated[-1] == request.stop_token_ids).any())
                )
                finish_reason = (
                    FINISH_STOP
                    if stopped_by_token
                    or stop_criteria.stop_observed
                    or repetition_guard.triggered
                    or generated.numel() < request.max_new_tokens
                    else FINISH_LENGTH
                )
        except CoreTurnAuthorityError as error:
            _debug_worker_error(stage, error)
            status = (
                STATUS_DEADLINE_EXCEEDED
                if monotonic_ns() >= authority.request_deadline_ns
                else STATUS_AUTHORITY_REJECTED
            )
            finish_reason = FINISH_ERROR
            remaining = torch.empty(0, dtype=torch.int64)
        except BaseException as error:
            _debug_worker_error(stage, error)
            status = STATUS_MODEL_ERROR
            finish_reason = FINISH_ERROR
            remaining = torch.empty(0, dtype=torch.int64)

        if _parameter_signature(model) != base_signature:
            status = STATUS_BASE_MUTATED
            finish_reason = FINISH_ERROR
            remaining = torch.empty(0, dtype=torch.int64)
        total = len(streamer.emitted) + int(remaining.numel())
        try:
            _queue_put(
                response_queue,
                _response_for(
                    authority,
                    status,
                    remaining,
                    generated_total=total,
                    final=True,
                    finish_reason=finish_reason,
                ),
            )
        except BaseException:
            return
        if status in {STATUS_BASE_MUTATED, STATUS_AUTHORITY_REJECTED}:
            # Never continue serving from a model whose invariant or immutable
            # launch/request capability was broken.
            _await_retirement_ack(retire_ack_event)
            return


class ModelService:
    """Own one resident model process and at most one outstanding generation."""

    def __init__(
        self,
        tokenizer: Any,
        model_factory: Callable[[], nn.Module],
        *,
        core_pipeline_factory: Callable[[nn.Module], CoreTurnPipeline] | None = None,
        core_workspace_mib: int = 512,
        max_input_tokens: int = 4_096,
        max_new_tokens: int = 512,
        max_prompt_chars: int = 32_000,
        max_response_chars: int = 32_000,
        startup_timeout: float = 180.0,
        request_timeout: float = 180.0,
        cancellation_timeout: float = 10.0,
        start_method: str = "spawn",
        gpu_lease_manager: GPULeaseManager | None = None,
        gpu_lease_owner: str = "resident-model",
        gpu_lease_purpose: str | Callable[[], str] = "inference",
        gpu_lease_vram_bytes: int = DEFAULT_MAX_VRAM_BYTES,
        worker_lifetime_seconds: float = 3_600.0,
        artifact_digest: str | None = None,
        multimodal_processor_config: tuple[str | Path, str | Path] | None = None,
    ) -> None:
        if not callable(tokenizer) or not callable(getattr(tokenizer, "decode", None)):
            raise TypeError("tokenizer must be callable and provide decode()")
        if isinstance(model_factory, (str, bytes, os.PathLike)) or not callable(
            model_factory
        ):
            raise TypeError(
                "model_factory must be a callable, never a model id or path"
            )
        if core_pipeline_factory is not None and not callable(core_pipeline_factory):
            raise TypeError("core_pipeline_factory must be callable")
        if (
            not isinstance(core_workspace_mib, int)
            or isinstance(core_workspace_mib, bool)
            or not 1 <= core_workspace_mib <= 4_096
        ):
            raise ValueError("core_workspace_mib must be in [1, 4096]")
        if not 1 <= max_input_tokens <= HARD_MAX_INPUT_TOKENS:
            raise ValueError("max_input_tokens exceeds the protocol bound")
        if (
            isinstance(model_factory, LocalGemmaModelFactory)
            and max_input_tokens > 4_096
        ):
            raise ValueError(
                "the production CoreTurnPipeline caps local Gemma chat at 4096 tokens"
            )
        if not 1 <= max_new_tokens <= HARD_MAX_NEW_TOKENS:
            raise ValueError("max_new_tokens exceeds the protocol bound")
        if not 1 <= max_prompt_chars <= HARD_MAX_PROMPT_CHARS:
            raise ValueError("max_prompt_chars exceeds the hard bound")
        if not 1 <= max_response_chars <= HARD_MAX_RESPONSE_CHARS:
            raise ValueError("max_response_chars exceeds the hard bound")
        if startup_timeout <= 0 or request_timeout <= 0 or cancellation_timeout <= 0:
            raise ValueError("service timeouts must be positive")
        if gpu_lease_manager is not None and not isinstance(
            gpu_lease_manager, GPULeaseManager
        ):
            raise TypeError("gpu_lease_manager must be a GPULeaseManager")
        if not isinstance(gpu_lease_purpose, str) and not callable(gpu_lease_purpose):
            raise TypeError("gpu_lease_purpose must be text or a callable")
        if (
            not isinstance(gpu_lease_vram_bytes, int)
            or isinstance(gpu_lease_vram_bytes, bool)
            or gpu_lease_vram_bytes <= 0
        ):
            raise ValueError("gpu_lease_vram_bytes must be a positive integer")
        if (
            not isinstance(worker_lifetime_seconds, (int, float))
            or isinstance(worker_lifetime_seconds, bool)
            or not math.isfinite(float(worker_lifetime_seconds))
            or float(worker_lifetime_seconds) <= 0.0
        ):
            raise ValueError("worker_lifetime_seconds must be finite and positive")
        self.tokenizer = tokenizer
        self.model_factory = model_factory
        self.core_pipeline_factory = (
            LocalGemmaCorePipelineFactory()
            if core_pipeline_factory is None
            and isinstance(model_factory, LocalGemmaModelFactory)
            else core_pipeline_factory
        )
        self.core_workspace_bytes = int(core_workspace_mib) * 1024**2
        self.max_input_tokens = int(max_input_tokens)
        self.max_new_tokens = int(max_new_tokens)
        self.max_prompt_chars = int(max_prompt_chars)
        self.max_response_chars = int(max_response_chars)
        self.startup_timeout = float(startup_timeout)
        self.request_timeout = float(request_timeout)
        self.cancellation_timeout = float(cancellation_timeout)
        self.gpu_lease_manager = gpu_lease_manager
        self.gpu_lease_owner = self._bounded_gpu_label(
            gpu_lease_owner, "gpu_lease_owner"
        )
        self.gpu_lease_purpose = gpu_lease_purpose
        if isinstance(gpu_lease_purpose, str):
            self._bounded_gpu_label(gpu_lease_purpose, "gpu_lease_purpose")
        self.gpu_lease_vram_bytes = int(gpu_lease_vram_bytes)
        self.worker_lifetime_seconds = float(worker_lifetime_seconds)
        factory_digest = (
            model_factory.artifact_digest
            if isinstance(model_factory, LocalGemmaModelFactory)
            else None
        )
        if artifact_digest is not None and factory_digest is not None:
            if (
                _validated_digest_hex(artifact_digest, "artifact_digest")
                != factory_digest
            ):
                raise ValueError("service and model-factory artifact digests disagree")
        selected_digest = artifact_digest or factory_digest
        self.artifact_digest = (
            None
            if selected_digest is None
            else _validated_digest_hex(selected_digest, "artifact_digest")
        )
        self._artifact_digest_tensor = (
            ZERO_ARTIFACT_DIGEST.clone()
            if self.artifact_digest is None
            else _digest_tensor(self.artifact_digest)
        )
        self._context = multiprocessing.get_context(start_method)
        self._request_lock = Lock()
        self._state_lock = RLock()
        self._next_request_id = 1
        self._active_request_id: int | None = None
        self._service_nonce = secrets.token_hex(16)
        self._worker_incarnation = 0
        self._process: Any = None
        self._request_queue: Any = None
        self._response_queue: Any = None
        self._cancel_event: Any = None
        self._retire_ack_event: Any = None
        self._ready_event: Any = None
        self._failed_event: Any = None
        self._gpu_lease: GPULease | None = None
        self._worker_authority: Tensor | None = None
        self._startup_status: Tensor | None = None
        # Set before retiring a worker whose cancelled request never produced
        # a trusted terminal frame. No successor request may reuse that IPC or
        # CUDA resident until ``stop()`` positively confirms worker death.
        self._retire_required = False
        self._reserved_stop_sequences = reserved_stop_sequences(tokenizer)
        if multimodal_processor_config is None:
            self._multimodal_processor_config: tuple[Path, Path] | None = None
        else:
            if (
                not isinstance(multimodal_processor_config, tuple)
                or len(multimodal_processor_config) != 2
            ):
                raise TypeError("multimodal processor config must contain two paths")
            processor_root = Path(multimodal_processor_config[0]).resolve(strict=True)
            processor_manifest = Path(multimodal_processor_config[1]).resolve(
                strict=True
            )
            self._multimodal_processor_config = (
                processor_root,
                processor_manifest,
            )
        self._multimodal_processor: VerifiedGemma4MultimodalProcessor | None = None
        self._multimodal_processor_lock = Lock()

    @classmethod
    def for_local_gemma(
        cls,
        model_path: str | Path,
        *,
        vram_limit_gib: float = 16.7,
        tokenizer_kwargs: Mapping[str, Any] | None = None,
        manifest_path: str | Path | None = None,
        artifact_digest: str | None = None,
        **service_kwargs: Any,
    ) -> ModelService:
        if manifest_path is None:
            raise ValueError(
                "interactive Gemma startup requires a trusted artifact manifest"
            )
        root = verify_local_gemma_path(model_path)
        factory = LocalGemmaModelFactory(
            str(root),
            vram_limit_gib,
            None if manifest_path is None else str(manifest_path),
            artifact_digest,
        )
        verified_before_tokenizer = verify_artifact_manifest(root, manifest_path)
        _verify_instruction_tuned_e4b_snapshot(verified_before_tokenizer)
        tokenizer = load_local_tokenizer(root, tokenizer_kwargs=tokenizer_kwargs)
        verified_after_tokenizer = verify_artifact_manifest(root, manifest_path)
        _verify_instruction_tuned_e4b_snapshot(verified_after_tokenizer)
        if _sha256_file(Path(manifest_path).expanduser().resolve(strict=True)) != (
            factory.artifact_digest
        ):
            raise RuntimeError("manifest changed during tokenizer load")
        return cls(
            tokenizer,
            factory,
            artifact_digest=factory.artifact_digest,
            multimodal_processor_config=(root, manifest_path),
            **service_kwargs,
        )

    @property
    def is_running(self) -> bool:
        process = self._process
        try:
            return bool(process is not None and process.is_alive())
        except (AssertionError, OSError, ValueError):
            return False

    @property
    def worker_pid(self) -> int | None:
        process = self._process
        return None if process is None else process.pid

    @property
    def active_request_id(self) -> int | None:
        with self._state_lock:
            return self._active_request_id

    @property
    def gpu_lease(self) -> GPULease | None:
        with self._state_lock:
            return self._gpu_lease

    def runtime_identity(self) -> ModelRuntimeIdentity | None:
        """Return identity only while the exact configured worker is live.

        The incarnation counter changes for every spawn attempt.  A prior
        image attestation therefore cannot survive worker death/restart even
        if the operating system later reuses the same PID.
        """

        with self._state_lock:
            return self._runtime_identity_locked()

    def _runtime_identity_locked(self) -> ModelRuntimeIdentity | None:
        """Return exact worker identity while ``_state_lock`` is held."""

        process = self._process
        ready = self._ready_event
        factory = self.model_factory
        processor = self._multimodal_processor_config
        if (
            type(factory) is not LocalGemmaModelFactory
            or factory.manifest_path is None
            or self.artifact_digest is None
            or factory.artifact_digest != self.artifact_digest
            or not isinstance(processor, tuple)
            or len(processor) != 2
            or process is None
            or ready is None
            or not ready.is_set()
            or self._worker_incarnation < 1
        ):
            return None
        try:
            factory_root = Path(factory.model_path)
            factory_manifest = Path(factory.manifest_path)
            processor_root, processor_manifest = processor
            if (
                processor_root != factory_root
                or processor_manifest != factory_manifest
                or _sha256_file(factory_manifest) != self.artifact_digest
            ):
                return None
            if not process.is_alive() or process.pid is None:
                return None
            self._validate_running_gpu_lease()
        except BaseException:
            return None
        lease = self._gpu_lease
        lease_epoch = 0 if lease is None else int(lease.epoch)
        lease_deadline_ns = 0 if lease is None else int(lease.deadline * 1_000_000_000)
        return ModelRuntimeIdentity(
            service_nonce=self._service_nonce,
            worker_incarnation=self._worker_incarnation,
            worker_pid=int(process.pid),
            lease_epoch=lease_epoch,
            lease_deadline_ns=lease_deadline_ns,
            artifact_digest=self.artifact_digest,
            model_root=str(factory_root),
            manifest_path=str(factory_manifest),
            processor_root=str(processor_root),
            processor_manifest_path=str(processor_manifest),
        )

    @staticmethod
    def _bounded_gpu_label(value: object, field: str) -> str:
        if not isinstance(value, str):
            raise TypeError(f"{field} must be a string")
        normalized = value.strip()
        if not 1 <= len(normalized) <= 64:
            raise ValueError(f"{field} must contain 1-64 characters")
        if any(
            ord(character) < 32 or ord(character) == 127 for character in normalized
        ):
            raise ValueError(f"{field} contains a control character")
        return normalized

    def _current_gpu_purpose(self) -> str:
        provider = self.gpu_lease_purpose
        value = provider() if callable(provider) else provider
        return self._bounded_gpu_label(value, "gpu_lease_purpose")

    def _gpu_deadline(self) -> float:
        manager = self.gpu_lease_manager
        if manager is None:
            raise RuntimeError("GPU lease deadline requested without a manager")
        return manager.deadline_after(self.worker_lifetime_seconds)

    def _validate_running_gpu_lease(self) -> None:
        manager = self.gpu_lease_manager
        lease = self._gpu_lease
        if manager is None:
            if lease is not None:
                raise WorkerExecutionError(
                    "GPU lease capability exists without its manager"
                )
            return
        if lease is None:
            raise WorkerExecutionError("running model worker has no GPU lease")
        purpose = self._current_gpu_purpose()
        if lease.owner != self.gpu_lease_owner:
            raise StaleGPULeaseError("GPU lease owner does not match the worker")
        if lease.purpose != purpose:
            raise StaleGPULeaseError("GPU lease purpose does not match the worker")
        if lease.vram_budget_bytes != self.gpu_lease_vram_bytes:
            raise GPULeaseBudgetError(
                "GPU lease budget does not exactly match the worker budget"
            )
        manager.validate(
            lease,
            purpose=purpose,
            required_vram_bytes=self.gpu_lease_vram_bytes,
        )

    def _process_image(
        self,
        content: bytes,
        prompt: str,
    ) -> MultimodalTensorBundle:
        config = self._multimodal_processor_config
        if config is None:
            raise RequestLimitError(
                "image inference requires a verified local Gemma 4 processor"
            )
        try:
            with self._multimodal_processor_lock:
                processor = self._multimodal_processor
                if processor is None:
                    processor = VerifiedGemma4MultimodalProcessor(*config)
                    self._multimodal_processor = processor
                return processor.process_image(content, prompt)
        except MultimodalPreprocessError as exc:
            raise RequestLimitError("local image preprocessing was rejected") from exc

    def _process_audio_wav(
        self,
        content: bytes,
        prompt: str,
    ) -> MultimodalTensorBundle:
        config = self._multimodal_processor_config
        if config is None:
            raise RequestLimitError(
                "audio inference requires a verified local Gemma 4 processor"
            )
        try:
            with self._multimodal_processor_lock:
                processor = self._multimodal_processor
                if processor is None:
                    processor = VerifiedGemma4MultimodalProcessor(*config)
                    self._multimodal_processor = processor
                return processor.process_audio_wav(content, prompt)
        except MultimodalPreprocessError as exc:
            raise RequestLimitError("local audio preprocessing was rejected") from exc

    def start(self) -> ModelService:
        def process_alive(candidate: Any) -> bool:
            try:
                return bool(candidate is not None and candidate.is_alive())
            except BaseException:
                return False

        with self._state_lock:
            created_attempt = False
            if self._retire_required:
                raise WorkerExecutionError(
                    "local model worker retirement is required before restart"
                )
            if self._process is None and self._gpu_lease is not None:
                raise WorkerExecutionError(
                    "GPU lease exists without a worker handle; death is unproven"
                )
            if self._process is not None and not process_alive(self._process):
                # A crashed worker must be reaped, its exact lease retired,
                # and all old IPC closed before a successor gets a new epoch.
                self.stop()
            if self._process is None:
                created_attempt = True
                # Increment before any fallible setup.  Failed attempts consume
                # an incarnation so no later worker can ever compare equal.
                self._worker_incarnation += 1
                try:
                    self._request_queue = self._context.Queue(maxsize=2)
                    self._response_queue = self._context.Queue(
                        maxsize=DEFAULT_RESPONSE_QUEUE_SIZE
                    )
                    self._cancel_event = self._context.Event()
                    self._retire_ack_event = self._context.Event()
                    self._ready_event = self._context.Event()
                    self._failed_event = self._context.Event()
                    self._worker_authority = _worker_authority_tensor(
                        self._artifact_digest_tensor
                    )
                    self._startup_status = _worker_startup_status_tensor()
                    self._process = self._context.Process(
                        target=_worker_main,
                        args=(
                            self.model_factory,
                            self.core_pipeline_factory,
                            self._request_queue,
                            self._response_queue,
                            self._cancel_event,
                            self._ready_event,
                            self._failed_event,
                            self._reserved_stop_sequences,
                            self.max_input_tokens,
                            self.core_workspace_bytes,
                            self._worker_authority,
                            self._startup_status,
                            self._retire_ack_event,
                        ),
                        name="cogni-local-model",
                        daemon=True,
                    )
                except BaseException as error:
                    try:
                        self.stop()
                    except BaseException as cleanup_error:
                        raise WorkerStartupError(
                            "local model worker IPC setup failed and cleanup "
                            "could not close the partial attempt"
                        ) from cleanup_error
                    raise WorkerStartupError(
                        "local model worker IPC setup failed"
                    ) from error

                process = self._process
                spawn_state = {"completed": False}
                manager = self.gpu_lease_manager
                if manager is not None:

                    def owner_alive() -> bool:
                        # Between acquire() and Process.start(), is_alive() is
                        # false even though the lease must remain fenced. Health
                        # becomes process-backed only after spawn completes.
                        if not spawn_state["completed"]:
                            return True
                        try:
                            return bool(process.is_alive())
                        except BaseException:
                            return True

                    try:
                        self._gpu_lease = manager.acquire(
                            self.gpu_lease_owner,
                            self._current_gpu_purpose(),
                            self.gpu_lease_vram_bytes,
                            deadline=self._gpu_deadline(),
                            owner_alive=owner_alive,
                        )
                        self._worker_authority[0] = self._gpu_lease.epoch
                        self._worker_authority[1] = int(
                            self._gpu_lease.deadline * 1_000_000_000
                        )
                    except BaseException:
                        # The process object exists but was never spawned. Reuse
                        # the death-confirming cleanup path before propagating the
                        # admission error.
                        self.stop()
                        raise
                try:
                    process.start()
                except BaseException as error:
                    try:
                        self.stop()
                    except BaseException as cleanup_error:
                        raise WorkerStartupError(
                            "local model process spawn failed and cleanup could not "
                            "confirm worker death"
                        ) from cleanup_error
                    raise WorkerStartupError(
                        "local model process spawn failed"
                    ) from error
                spawn_state["completed"] = True
            else:
                process = self._process

            ready_event = self._ready_event
            failed_event = self._failed_event
            startup_status = self._startup_status
            if ready_event is None or failed_event is None or startup_status is None:
                self.stop()
                raise WorkerStartupError(
                    "local model worker startup controls are incomplete"
                )
            if not created_attempt and ready_event.is_set():
                if failed_event.is_set() or not process_alive(process):
                    self.stop()
                    raise WorkerStartupError(
                        _worker_startup_failure_message(startup_status)
                    )
                # Re-validating an already resident worker must not destroy it
                # when the caller presents a changed purpose/budget. The
                # existing lease remains authoritative for its original scope.
                self._validate_running_gpu_lease()
                if failed_event.is_set() or not process_alive(process):
                    self.stop()
                    raise WorkerStartupError(
                        _worker_startup_failure_message(startup_status)
                    )
                return self

        def cleanup_attempt() -> None:
            with self._state_lock:
                if self._process is process:
                    self.stop()

        deadline = monotonic() + self.startup_timeout
        while monotonic() < deadline:
            failed = failed_event.is_set()
            alive = process_alive(process)
            if failed or not alive:
                cleanup_attempt()
                raise WorkerStartupError(
                    _worker_startup_failure_message(startup_status)
                )
            if ready_event.is_set():
                liveness_failure = False
                try:
                    with self._state_lock:
                        if self._process is not process:
                            raise WorkerStartupError(
                                "local model worker startup was superseded"
                            )
                        if failed_event.is_set() or not process_alive(process):
                            liveness_failure = True
                        else:
                            self._validate_running_gpu_lease()
                            liveness_failure = failed_event.is_set() or not (
                                process_alive(process)
                            )
                except BaseException:
                    # A purpose/budget/deadline change during model loading is
                    # a startup failure. Retire the process before surfacing
                    # the lease error so no unauthorized resident remains.
                    cleanup_attempt()
                    raise
                if liveness_failure:
                    cleanup_attempt()
                    raise WorkerStartupError(
                        _worker_startup_failure_message(startup_status)
                    )
                return self
            sleep(0.01)
        cleanup_attempt()
        stage = _worker_startup_failure_message(startup_status)
        raise WorkerStartupError(f"local model worker startup timed out ({stage})")

    def iter_generate_tokens(
        self,
        prompt: str,
        *,
        image_content: bytes | None = None,
        expected_runtime_identity: ModelRuntimeIdentity | None = None,
        audio_wav_content: bytes | None = None,
        max_new_tokens: int | None = None,
        stop_token_ids: Tensor | None = None,
        timeout: float | None = None,
        total_timeout: float | None = None,
        conversation_id: str = "primary",
        decode_mode: str = "conversation",
        sampling_seed: int | None = None,
    ) -> Iterator[GenerationChunk]:
        if expected_runtime_identity is not None and not isinstance(
            expected_runtime_identity, ModelRuntimeIdentity
        ):
            raise TypeError(
                "expected_runtime_identity must be a ModelRuntimeIdentity or None"
            )
        if expected_runtime_identity is not None and image_content is None:
            raise RequestLimitError(
                "expected runtime identity is valid only for an image request"
            )
        decode_mode_code = _decode_mode_code(decode_mode)
        requested_seed = _validated_sampling_seed(sampling_seed)
        wait_seconds = self.request_timeout if timeout is None else float(timeout)
        if not math.isfinite(wait_seconds) or wait_seconds <= 0:
            raise ValueError("request timeout must be finite and positive")
        explicit_total_timeout = total_timeout is not None
        total_wait_seconds = (
            self.request_timeout if total_timeout is None else float(total_timeout)
        )
        if not math.isfinite(total_wait_seconds) or total_wait_seconds <= 0:
            raise ValueError("total request timeout must be finite and positive")
        # A single absolute fence starts when the iterator itself begins work.
        # Model startup, tokenization, queue admission, and every healthy
        # intermediate frame all consume the same wall-clock budget.
        total_deadline = monotonic() + total_wait_seconds
        total_deadline_ns = monotonic_ns() + int(total_wait_seconds * 1_000_000_000)
        image_tensors: tuple[tuple[int, Tensor], ...] = ()
        audio_tensors: tuple[tuple[int, Tensor], ...] = ()
        media_digest: str | None = None
        if image_content is not None and audio_wav_content is not None:
            raise RequestLimitError("one turn cannot contain image and audio media")
        if image_content is None and audio_wav_content is None:
            input_ids, attention_mask = self._tokenize(prompt)
        elif image_content is not None:
            bundle = self._process_image(image_content, prompt)
            input_ids, attention_mask, image_tensors, media_digest = (
                _image_bundle_inputs(
                    bundle,
                    max_input_tokens=self.max_input_tokens,
                )
            )
        else:
            bundle = self._process_audio_wav(audio_wav_content, prompt)
            input_ids, attention_mask, audio_tensors, media_digest = (
                _audio_bundle_inputs(
                    bundle,
                    max_input_tokens=self.max_input_tokens,
                )
            )
        if monotonic() >= total_deadline:
            raise TimeoutError("local generation exceeded its total deadline")
        self.start()
        if monotonic() >= total_deadline:
            try:
                self.stop(timeout=0.01)
            except BaseException as error:
                raise WorkerExecutionError(
                    "timed-out model startup could not safely retire its worker"
                ) from error
            raise TimeoutError("local generation exceeded its total deadline")
        if not self._request_lock.acquire(blocking=False):
            raise ServiceBusyError("one generation request is already active")
        completed = False
        deadline_exceeded = False
        request_id = 0
        observed_total = 0
        buffered_tokens: list[Tensor] = []
        authority: _RequestAuthority | None = None
        request_runtime_identity: ModelRuntimeIdentity | None = None
        try:
            if monotonic() >= total_deadline:
                deadline_exceeded = True
                raise TimeoutError("local generation exceeded its total deadline")
            requested = (
                self.max_new_tokens if max_new_tokens is None else max_new_tokens
            )
            if (
                not isinstance(requested, int)
                or isinstance(requested, bool)
                or not 1 <= requested <= self.max_new_tokens
            ):
                raise RequestLimitError("max_new_tokens exceeds the service budget")
            # ``timeout`` remains an idle-worker timeout for API compatibility.
            # ``total_timeout`` is the independent wall-clock fence used by the
            # interactive manager. Healthy token frames may refresh the former,
            # but can never extend the latter.
            session_digest = _session_digest(conversation_id)
            if media_digest is not None:
                session_digest = _bind_media_session_digest(
                    session_digest,
                    media_digest,
                )
            effective_seed = (
                requested_seed
                if requested_seed is not None
                else (
                    _derived_sampling_seed(input_ids, session_digest)
                    if decode_mode_code == DECODE_CONVERSATION
                    else 0
                )
            )
            with self._state_lock:
                self._validate_running_gpu_lease()
                if image_content is not None:
                    request_runtime_identity = self._runtime_identity_locked()
                    if request_runtime_identity is None:
                        raise WorkerAuthorityError(
                            "image request lacks a live runtime identity"
                        )
                    if (
                        expected_runtime_identity is not None
                        and request_runtime_identity != expected_runtime_identity
                    ):
                        raise WorkerAuthorityError(
                            "image worker identity did not match the admitted runtime"
                        )
                request_id = self._next_request_id
                self._next_request_id += 1
                self._active_request_id = request_id
                self._cancel_event.clear()
                lease = self._gpu_lease
                lease_epoch = 0 if lease is None else lease.epoch
                lease_deadline_ns = (
                    0 if lease is None else int(lease.deadline * 1_000_000_000)
                )
                if lease is not None and (lease_epoch >= 2**31 or request_id >= 2**32):
                    raise WorkerAuthorityError(
                        "lease/request counters exceeded the bounded job-id layout"
                    )
                job_id = (
                    request_id if lease is None else (lease_epoch << 32) | request_id
                )
                request_deadline_ns = (
                    NO_DEADLINE_NS
                    if lease is None and not explicit_total_timeout
                    else min(total_deadline_ns, NO_DEADLINE_NS)
                )
                if lease is not None:
                    request_deadline_ns = min(
                        request_deadline_ns,
                        lease_deadline_ns,
                    )
                authority = _RequestAuthority(
                    request_id=request_id,
                    job_id=job_id,
                    lease_epoch=lease_epoch,
                    request_deadline_ns=request_deadline_ns,
                    lease_deadline_ns=lease_deadline_ns,
                    artifact_digest=self._artifact_digest_tensor.clone(),
                    session_digest=session_digest,
                )
            message = make_generate_request(
                request_id,
                input_ids,
                attention_mask,
                max_new_tokens=requested,
                stop_token_ids=stop_token_ids,
                decode_mode=decode_mode_code,
                sampling_seed=effective_seed,
                image_tensors=image_tensors,
                audio_tensors=audio_tensors,
                **authority.response_kwargs(),
            )
            admission_remaining = total_deadline - monotonic()
            if admission_remaining <= 0:
                deadline_exceeded = True
                self._cancel_event.set()
                raise TimeoutError("local generation exceeded its total deadline")
            try:
                self._request_queue.put(
                    message,
                    timeout=min(1.0, admission_remaining),
                )
            except Full as exc:
                if monotonic() >= total_deadline:
                    deadline_exceeded = True
                    self._cancel_event.set()
                    raise TimeoutError(
                        "local generation exceeded its total deadline"
                    ) from exc
                raise ServiceBusyError("bounded request queue is full") from exc
            deadline = monotonic() + wait_seconds
            while True:
                now = monotonic()
                remaining = deadline - now
                total_remaining = total_deadline - now
                if total_remaining <= 0:
                    deadline_exceeded = True
                    self._cancel_event.set()
                    raise TimeoutError("local generation exceeded its total deadline")
                if remaining <= 0:
                    deadline_exceeded = True
                    self._cancel_event.set()
                    raise TimeoutError("local generation exceeded its deadline")
                if not self.is_running:
                    raise WorkerExecutionError(
                        "local model worker stopped unexpectedly"
                    )
                try:
                    raw = self._response_queue.get(
                        timeout=min(0.1, remaining, total_remaining)
                    )
                except Empty:
                    continue
                if monotonic() >= total_deadline:
                    deadline_exceeded = True
                    self._cancel_event.set()
                    raise TimeoutError("local generation exceeded its total deadline")
                try:
                    frame = parse_response(raw)
                except TensorProtocolError as exc:
                    raise WorkerExecutionError(
                        "worker response protocol failed"
                    ) from exc
                authority.validate_frame(frame)
                if frame.final and frame.status in {
                    STATUS_BASE_MUTATED,
                    STATUS_AUTHORITY_REJECTED,
                }:
                    retire_ack_event = self._retire_ack_event
                    if retire_ack_event is not None:
                        retire_ack_event.set()
                expected_total = observed_total + int(frame.token_ids.numel())
                if frame.generated_total != expected_total:
                    raise WorkerExecutionError(
                        "worker response token count is not monotonic"
                    )
                if expected_total > requested:
                    raise WorkerExecutionError(
                        "worker response exceeded the atomic token buffer"
                    )
                observed_total = expected_total
                if frame.status == STATUS_OK:
                    if frame.token_ids.numel():
                        buffered_tokens.append(
                            frame.token_ids.detach().to("cpu").clone().contiguous()
                        )
                    if frame.final:
                        finish_reason = _FINISH_REASON_NAMES.get(frame.finish_reason)
                        if finish_reason not in {"stop", "length"}:
                            raise WorkerExecutionError(
                                "worker success lacked a valid terminal reason"
                            )
                        tokens = (
                            torch.cat(buffered_tokens).contiguous()
                            if buffered_tokens
                            else torch.empty(0, dtype=torch.int64)
                        )
                        if int(tokens.numel()) != observed_total:
                            raise WorkerExecutionError(
                                "atomic token buffer disagrees with worker total"
                            )
                        terminal_media_sha256: str | None = None
                        terminal_runtime_identity: ModelRuntimeIdentity | None = None
                        if image_content is not None:
                            if media_digest is None:
                                raise WorkerAuthorityError(
                                    "image request lacks a media digest"
                                )
                            terminal_runtime_identity = self.runtime_identity()
                            if (
                                terminal_runtime_identity is None
                                or terminal_runtime_identity != request_runtime_identity
                                or (
                                    expected_runtime_identity is not None
                                    and terminal_runtime_identity
                                    != expected_runtime_identity
                                )
                            ):
                                raise WorkerAuthorityError(
                                    "image worker identity changed before terminal publication"
                                )
                            # ``authority.validate_frame`` above proves that the
                            # terminal frame belongs to the session digest that
                            # was cryptographically bound to this exact image.
                            terminal_media_sha256 = media_digest
                        completed = True
                        # Atomic publication: no STATUS_OK token crosses this
                        # boundary until the worker's terminal frame confirms
                        # generation success and base-model integrity.
                        yield GenerationChunk(
                            request_id=request_id,
                            token_ids=tokens,
                            generated_total=observed_total,
                            final=True,
                            cancelled=False,
                            finish_reason=finish_reason,
                            generation_mode="cogni_core",
                            media_sha256=terminal_media_sha256,
                            runtime_identity=terminal_runtime_identity,
                        )
                        return
                    # The timeout measures worker silence. Intermediate frames
                    # prove liveness even though their tokens remain private in
                    # the bounded atomic buffer until terminal success.
                    deadline = monotonic() + wait_seconds
                    continue
                if frame.status == STATUS_CANCELLED:
                    completed = True
                    yield GenerationChunk(
                        request_id=request_id,
                        token_ids=torch.empty(0, dtype=torch.int64),
                        generated_total=0,
                        final=True,
                        cancelled=True,
                        finish_reason="cancelled",
                        generation_mode="none",
                    )
                    return
                if frame.status == STATUS_BASE_MUTATED:
                    completed = True
                    raise BaseModelMutationError("base-model immutability check failed")
                if frame.status == STATUS_AUTHORITY_REJECTED:
                    completed = True
                    raise WorkerAuthorityError(
                        "local model worker rejected stale request authority"
                    )
                if frame.status == STATUS_DEADLINE_EXCEEDED:
                    deadline_exceeded = True
                    completed = True
                    raise TimeoutError("local generation exceeded its total deadline")
                if frame.status in {STATUS_INVALID_REQUEST, STATUS_MODEL_ERROR}:
                    completed = True
                    raise WorkerExecutionError("local model worker rejected generation")
                completed = True
                raise WorkerExecutionError("local model worker returned unknown status")
        finally:
            retirement_error: BaseException | None = None
            retire_worker = deadline_exceeded
            if request_id and not completed:
                self._cancel_event.set()
                drained = self._drain_cancelled_request(
                    request_id,
                    authority=authority,
                )
                if not drained:
                    retire_worker = True
            if retire_worker:
                with self._state_lock:
                    self._retire_required = True
                try:
                    # Cancellation was already signalled and drained above.
                    # stop() confirms process death before releasing the exact
                    # GPU capability, escalating to terminate/kill if needed.
                    self.stop(timeout=min(1.0, self.cancellation_timeout))
                except BaseException as error:
                    retirement_error = error
            with self._state_lock:
                if self._active_request_id == request_id:
                    self._active_request_id = None
            self._request_lock.release()
            if retirement_error is not None:
                raise WorkerExecutionError(
                    "cancelled model request could not safely retire its worker"
                ) from retirement_error

    def generate(
        self,
        prompt: str,
        *,
        image_content: bytes | None = None,
        expected_runtime_identity: ModelRuntimeIdentity | None = None,
        audio_wav_content: bytes | None = None,
        max_new_tokens: int | None = None,
        stop_token_ids: Tensor | None = None,
        timeout: float | None = None,
        total_timeout: float | None = None,
        conversation_id: str = "primary",
        decode_mode: str = "conversation",
        sampling_seed: int | None = None,
    ) -> GenerationResult:
        chunks: list[Tensor] = []
        request_id = 0
        cancelled = False
        finish_reason: str | None = None
        generation_mode = "cogni_core"
        media_sha256: str | None = None
        runtime_identity: ModelRuntimeIdentity | None = None
        for chunk in self.iter_generate_tokens(
            prompt,
            image_content=image_content,
            expected_runtime_identity=expected_runtime_identity,
            audio_wav_content=audio_wav_content,
            max_new_tokens=max_new_tokens,
            stop_token_ids=stop_token_ids,
            timeout=timeout,
            total_timeout=total_timeout,
            conversation_id=conversation_id,
            decode_mode=decode_mode,
            sampling_seed=sampling_seed,
        ):
            request_id = chunk.request_id
            if chunk.token_ids.numel():
                chunks.append(chunk.token_ids)
            cancelled = cancelled or chunk.cancelled
            if chunk.final:
                finish_reason = chunk.finish_reason
                generation_mode = chunk.generation_mode
                media_sha256 = chunk.media_sha256
                runtime_identity = chunk.runtime_identity
        tokens = torch.cat(chunks) if chunks else torch.empty(0, dtype=torch.int64)
        if tokens.numel() > self.max_new_tokens:
            raise WorkerExecutionError("worker exceeded the aggregate token budget")
        if cancelled:
            raise GenerationCancelled("local generation was cancelled")
        text = self.tokenizer.decode(
            tokens.tolist(),
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        if not isinstance(text, str) or len(text) > self.max_response_chars:
            raise RequestLimitError("decoded response exceeds its character bound")
        if finish_reason not in {"stop", "length"}:
            raise WorkerExecutionError("generation completed without a finish reason")
        return GenerationResult(
            request_id,
            tokens,
            text,
            finish_reason,
            generation_mode,
            media_sha256,
            runtime_identity,
        )

    def cancel(self, request_id: int | None = None) -> bool:
        with self._state_lock:
            active = self._active_request_id
            if active is None or (request_id is not None and request_id != active):
                return False
            self._cancel_event.set()
            return True

    def _release_gpu_lease_after_worker_death(self) -> None:
        lease = self._gpu_lease
        manager = self.gpu_lease_manager
        if lease is None:
            return
        if manager is None:
            raise WorkerExecutionError(
                "cannot release GPU lease without its manager; capability retained"
            )
        try:
            manager.release(lease)
        except StaleGPULeaseError:
            # A watchdog may have reaped this exact capability after observing
            # the same process death. No other stale/revoked state is safe to
            # ignore, because it could conceal an ownership protocol breach.
            reaped = any(
                event.lease == lease and event.reason == "owner_confirmed_dead"
                for event in manager.history
            )
            if not reaped:
                raise
        self._gpu_lease = None

    def stop(self, timeout: float = 10.0) -> None:
        if timeout <= 0:
            raise ValueError("stop timeout must be positive")
        with self._state_lock:
            process = self._process
            if process is None:
                if self._gpu_lease is not None:
                    raise WorkerExecutionError(
                        "GPU lease exists without a worker handle; capability retained"
                    )
                # A setup failure can occur after one or both queues were
                # allocated but before Process construction completed.
                for queue in (self._request_queue, self._response_queue):
                    if queue is not None:
                        queue.close()
                        queue.join_thread()
                self._request_queue = None
                self._response_queue = None
                self._cancel_event = None
                self._ready_event = None
                self._retire_ack_event = None
                self._failed_event = None
                self._worker_authority = None
                self._startup_status = None
                self._active_request_id = None
                self._retire_required = False
                return
            shutdown_errors: list[str] = []

            def worker_alive() -> bool:
                try:
                    return bool(process.is_alive())
                except (AssertionError, OSError, ValueError) as error:
                    shutdown_errors.append(f"liveness check: {type(error).__name__}")
                    return False

            def join_worker(stage: str, wait_seconds: float) -> None:
                try:
                    process.join(wait_seconds)
                except Exception as error:  # final liveness check remains authoritative
                    shutdown_errors.append(f"{stage} join: {type(error).__name__}")

            if self._cancel_event is not None:
                self._cancel_event.set()
            if worker_alive() and self._request_queue is not None:
                try:
                    self._request_queue.put(make_stop_request(), timeout=0.2)
                except Full:
                    pass
                except Exception as error:
                    # Broken IPC must not prevent OS-level termination of a
                    # worker that may still own model memory or CUDA state.
                    shutdown_errors.append(f"stop request: {type(error).__name__}")

            # Cooperative shutdown is always attempted first. A worker may be
            # loading a multi-gigabyte model, so the caller controls this wait.
            join_worker("graceful", float(timeout))
            if worker_alive():
                try:
                    process.terminate()
                except Exception as error:
                    shutdown_errors.append(f"terminate: {type(error).__name__}")
                join_worker("terminate", 2.0)

            if worker_alive():
                kill = getattr(process, "kill", None)
                if callable(kill):
                    try:
                        kill()
                    except Exception as error:
                        shutdown_errors.append(f"kill: {type(error).__name__}")
                    join_worker("kill", 2.0)
                else:
                    shutdown_errors.append("kill: unavailable")

            # Never destroy IPC or forget a process that may still own CUDA.
            # Keeping every reference makes the failure observable and allows
            # a supervisor/operator to retry or inspect the live worker.
            if worker_alive():
                detail = (
                    "; ".join(shutdown_errors)
                    if shutdown_errors
                    else "all shutdown signals returned without worker exit"
                )
                raise WorkerExecutionError(
                    "local model worker survived graceful, terminate, and kill "
                    f"shutdown stages; IPC retained ({detail})"
                )

            # Queue feeder threads are process IPC and are closed only after
            # worker death has been positively confirmed above.
            for queue in (self._request_queue, self._response_queue):
                if queue is not None:
                    queue.close()
                    queue.join_thread()
            self._request_queue = None
            self._response_queue = None
            # Exact capability release is intentionally last: the worker is
            # dead and all IPC feeder state has been cleaned, so no process can
            # issue more work under this epoch. Unexpected stale cleanup is
            # surfaced while retaining the dead process and lease references.
            self._release_gpu_lease_after_worker_death()
            close_error: Exception | None = None
            close_process = getattr(process, "close", None)
            if callable(close_process):
                try:
                    close_process()
                except Exception as error:
                    close_error = error
            self._cancel_event = None
            self._ready_event = None
            self._retire_ack_event = None
            self._failed_event = None
            self._worker_authority = None
            self._startup_status = None
            self._active_request_id = None
            if close_error is not None:
                # Retain the dead Process handle so an operator/supervisor can
                # retry cleanup. Silently forgetting it would hide an OS
                # handle leak on repeated Windows restarts.
                raise WorkerExecutionError(
                    "dead local model process handle could not be closed"
                ) from close_error
            self._process = None
            self._retire_required = False

    def _tokenize(self, prompt: str) -> tuple[Tensor, Tensor]:
        if not isinstance(prompt, str):
            raise TypeError("prompt must be text at the control boundary")
        if not prompt or len(prompt) > self.max_prompt_chars:
            raise RequestLimitError("prompt exceeds its character bound")
        if any(
            ord(character) < 32 and character not in "\t\r\n" for character in prompt
        ):
            raise RequestLimitError("prompt contains unsupported control characters")
        encoded = self.tokenizer(prompt, return_tensors="pt", truncation=False)
        if not isinstance(encoded, Mapping) or "input_ids" not in encoded:
            raise RequestLimitError("tokenizer did not return input_ids")
        input_ids = torch.as_tensor(encoded["input_ids"]).detach()
        if input_ids.ndim != 2 or input_ids.shape[0] != 1:
            raise RequestLimitError("tokenizer output must be batch one")
        if not 1 <= input_ids.shape[1] <= self.max_input_tokens:
            raise RequestLimitError("tokenized prompt exceeds its token bound")
        input_ids = input_ids.to("cpu", dtype=torch.int64).contiguous()
        raw_mask = encoded.get("attention_mask")
        attention_mask = (
            torch.ones_like(input_ids)
            if raw_mask is None
            else torch.as_tensor(raw_mask)
            .detach()
            .to("cpu", dtype=torch.int64)
            .contiguous()
        )
        return input_ids, attention_mask

    def _drain_cancelled_request(
        self, request_id: int, *, authority: _RequestAuthority | None = None
    ) -> bool:
        """Return true only after a trusted terminal frame or proven worker death."""

        deadline = monotonic() + self.cancellation_timeout
        while self.is_running and monotonic() < deadline:
            try:
                frame = parse_response(self._response_queue.get(timeout=0.05))
            except Empty:
                continue
            except TensorProtocolError:
                return False
            if authority is not None:
                try:
                    authority.validate_frame(frame)
                except WorkerAuthorityError:
                    return False
            if frame.request_id == request_id and frame.final:
                if frame.status in {
                    STATUS_BASE_MUTATED,
                    STATUS_AUTHORITY_REJECTED,
                }:
                    retire_ack_event = self._retire_ack_event
                    if retire_ack_event is not None:
                        retire_ack_event.set()
                return True
        return not self.is_running

    def __enter__(self) -> ModelService:
        return self.start()

    def __exit__(self, _exc_type: Any, _exc: Any, _traceback: Any) -> None:
        self.stop()


__all__ = [
    "BaseModelMutationError",
    "CONVERSATION_TEMPERATURE",
    "CONVERSATION_TOP_K",
    "CONVERSATION_TOP_P",
    "GenerationCancelled",
    "GenerationChunk",
    "GenerationResult",
    "HARD_MAX_PROMPT_CHARS",
    "HARD_MAX_RESPONSE_CHARS",
    "LocalGemmaCorePipelineFactory",
    "LocalGemmaModelFactory",
    "ModelRuntimeIdentity",
    "ModelService",
    "ModelServiceError",
    "RequestLimitError",
    "ServiceBusyError",
    "WorkerExecutionError",
    "WorkerAuthorityError",
    "WorkerStartupError",
    "load_local_tokenizer",
    "truncate_repeated_tokens",
]
