"""Fail-closed physical-GPU boundary for the Cogni-OS lab server.

The laboratory permits physical GPU indices 0 through 5.  This checkout is
stricter: it may use physical GPU 5 only.  The guard never enumerates devices;
every ``nvidia-smi`` command names exact index 5 or the pinned GPU5 UUID, so
GPUs 0 through 4, 6, and 7 are never queried, enumerated, exposed, reserved,
allocated, or used.

Docker remaps physical GPU 5 to the container's logical ``cuda:0``.  Therefore
the generated container environment deliberately omits
``CUDA_VISIBLE_DEVICES`` instead of setting it to the host index.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import csv
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re
import secrets
import shlex
import stat
import subprocess
import sys
import tarfile
from time import time_ns
from types import MappingProxyType
from typing import Any, Iterator, Mapping, Sequence

try:
    import fcntl
    import resource
except ImportError:  # pragma: no cover - the server gate is Linux-only
    fcntl = None
    resource = None


LAB_ALLOWED_PHYSICAL_GPU_INDICES = frozenset(range(6))
PROJECT_PHYSICAL_GPU_INDEX = 5
PROJECT_GPU_UUID = "GPU-84d7eeb0-65e0-a5b1-d7db-d09ef59fe03a"
NVIDIA_SMI_TIMEOUT_SECONDS = 5.0
MAX_NVIDIA_SMI_OUTPUT_CHARS = 65_536
MAX_IDLE_DRIVER_MEMORY_MIB = 64
MAX_EVIDENCE_BYTES = 4 * 1024 * 1024
MAX_SOURCE_ARCHIVE_BYTES = 128 * 1024 * 1024
GUARD_STATE_PARENT = Path("/home/shoon")
GUARD_STATE_ROOT = GUARD_STATE_PARENT / ".cognios-gpu5-guard"
DOCKER_CONFIG_ROOT = GUARD_STATE_ROOT / "docker-empty-config"
GPU5_LEASE_PATH = GUARD_STATE_ROOT / "gpu5-project.lock"
GPU5_SCHEDULER_RESERVATION_PATH = Path(
    "/run/cognios-lab-scheduler/gpu5-reservation.json"
)
GPU5_SCHEDULER_RESERVATION_OWNER_UID = 0
GPU5_SCHEDULER_RESERVATION_SCHEMA = "cogni.lab.gpu5.reservation.v1"
MAX_GPU5_SCHEDULER_RESERVATION_BYTES = 4_096
MAX_GPU5_SCHEDULER_RESERVATION_WINDOW_NS = 24 * 60 * 60 * 1_000_000_000
GPU5_SCHEDULER_CLEANUP_GRACE_NS = 5 * 60 * 1_000_000_000
EVIDENCE_HOST_ROOT = GUARD_STATE_ROOT / "evidence"
SOURCE_SNAPSHOT_ROOT = GUARD_STATE_ROOT / "source-snapshots"
MODEL_SNAPSHOT_ROOT = GUARD_STATE_ROOT / "model-snapshots"
NVIDIA_SMI_EXECUTABLE = Path("/usr/bin/nvidia-smi")
DOCKER_EXECUTABLE = Path("/usr/bin/docker")
GIT_EXECUTABLE = Path("/usr/bin/git")
DOCKER_SOCKET_PATH = Path("/run/docker.sock")
DOCKER_HOST_URI = "unix:///run/docker.sock"


PINNED_DOCKER_IMAGE = "cogni-os-dev@sha256:20aaf1d7cde8d6a504ba08f158a34a1907eac9413f3578acc4637f0a1b2ec8ba"
_GPU_QUERY_FIELDS = (
    "index",
    "uuid",
    "utilization.gpu",
    "memory.used",
    "memory.total",
)
_COMPUTE_QUERY_FIELDS = ("gpu_uuid", "pid", "used_gpu_memory")
_ENV_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")
_RESERVED_CONTAINER_GPU_ENV = frozenset({"CUDA_VISIBLE_DEVICES"})
_CONTAINER_NAME = re.compile(r"cognios-gpu5-[a-z0-9]{12}\Z")
_LAUNCH_NONCE = re.compile(r"[0-9a-f]{32}\Z")
_CONTAINER_ID = re.compile(r"[0-9a-f]{64}\Z")
_GUARD_LABEL = "io.cognios.guard"
_SOURCE_COMMIT_LABEL = "io.cognios.source-commit"
_LAUNCH_NONCE_LABEL = "io.cognios.launch-nonce"
_EXECUTION_PROFILE_LABEL = "io.cognios.execution-profile"
_VALIDATION_ARTIFACT_PROFILE_LABEL = "io.cognios.validation-artifact-profile"
BASE_CANARY_ARTIFACT_PROFILE = "base-canary"
PRODUCT_E4B_IT_ARTIFACT_PROFILE = "product-e4b-it"
_PRODUCT_ACCEPTANCE_CASES = (
    "product-identity",
    "product-self-harness",
    "product-systems",
    "product-grounding-followup",
    "casual-greeting",
    "typo-tolerance",
    "follow-up-context",
    "context-switch",
    "continuation",
    "repetition",
    "safe-correction",
    "safe-patching",
    "bounded-retry",
    "privacy",
    "measurement",
    "tool-truth",
    "error-record",
    "long-context",
    "uncertainty",
    "natural-finish",
)
_PRODUCT_ACCEPTANCE_PROMPTS = (
    "안녕하세여! 당신은 어떤 모델이며 저장 파라미터와 effective 파라미터는 각각 몇 개인가요? "
    "검증된 Runtime Fact-book 수치만 한 번씩 답하세요.",
    "자가 거울치료가 무엇인가요? 핵심만 세 문장 이내로 자연스럽게 설명해 주세요.",
    "Cogni-OS의 CTS, System 1.5, 2.5, 3, 4를 검증 상태와 설계 목표를 구분해 설명하고 "
    "마지막은 반드시 '이상입니다.'로 끝내세요.",
    "방금 답변에서 실제 검증과 향후 목표를 구분하는 원칙만 두 문장으로 요약하고 마침표로 끝내세요.",
    "안녕하세요! 오늘 저와 어떤 일을 함께 할 수 있나요? 두 문장으로 편안하게 답해주세요.",
    "온디바이스 AI 장점 두개랑 한계 한개를 세문장으로 알려주세여. 같은 말은 반복하지 마세요.",
    "방금 말한 일들 가운데 코드 POC를 만들 때 가장 먼저 할 일을 한 문장으로 골라주세요.",
    "이제 주제를 바꿀게요. 확인된 사실과 추론을 섞지 않는 원칙을 두 문장으로 설명해 주세요.",
    "긴 답변이 중간에 끊기지 않도록 자동 이어쓰기를 실제로 검증합니다. 생성 중단 감지, "
    "앞부분 중복 방지, "
    "잘린 문장 복구, 반복 루프 차단, 최종 종료 판정을 각각 원인과 검증 기준까지 "
    "자세히 분석해 다섯 가지 항목으로 모두 완결하세요.",
    "반복 없는 좋은 요약문의 조건을 세 가지 제시하고 각 조건은 서로 다른 내용으로 끝내세요.",
    "사용자가 잘못된 사실을 정정했을 때 대화형 AI가 취해야 할 절차를 세 단계로 간결하게 답하세요.",
    "로컬 파일을 수정하기 전에 백업, 검증, 롤백을 어떻게 준비해야 하는지 세 문장으로 설명하세요.",
    "예외가 발생한 작업을 무한 재시도하지 않고 안전하게 종료하는 기준을 두 문장으로 답하세요.",
    "개인정보가 포함된 요청을 오프라인 환경에서 처리할 때 지켜야 할 원칙을 세 문장으로 답하세요.",
    "제한된 GPU 메모리에서 추론할 때 측정값과 설계 목표를 구분해야 하는 이유를 설명하세요.",
    "도구 실행 결과를 확인하지 못했을 때 AI가 성공했다고 말하면 안 되는 이유를 두 문장으로 답하세요.",
    "오류 복구 과정에서 원인, 수정, 회귀 테스트를 어떤 순서로 기록해야 하는지 설명하세요.",
    "긴 대화에서 오래된 문맥을 줄이면서 사용자 의도를 보존하는 방법을 세 문장으로 답하세요.",
    "불확실한 답변을 사실처럼 단정하지 않기 위한 표현 원칙을 두 문장으로 설명하세요.",
    "자연스러운 한국어 답변의 완결성을 판정할 때 확인할 사항을 세 문장으로 설명하세요.",
)
_PRODUCT_REQUIRED_TERM_GROUPS = (
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
    (),
    (
        ("온디바이스", "기기", "장치", "로컬"),
        ("장점", "보안", "보호", "응답", "오프라인", "인터넷"),
        ("한계", "제약", "제한", "메모리", "성능", "전력"),
    ),
    (),
    (
        ("사실", "확인", "검증된 정보"),
        ("추론", "판단", "정보만", "구분", "근거", "원칙"),
    ),
    (("문장", "답변", "생성"), ("끝", "완결", "길이", "토큰", "중단", "끊")),
    (
        ("요약", "요약문", "군더더기", "원문", "세부 사항", "수식어"),
        ("반복", "핵심", "간결"),
    ),
    (("정정", "사실", "수정"), ("확인", "검증", "검토", "반영", "수용")),
    (("백업",), ("검증", "테스트"), ("롤백", "복구")),
    (("재시도", "예외", "오류"), ("종료", "중단", "한도", "횟수")),
    (("개인정보", "데이터"), ("오프라인", "로컬", "장치")),
    (("측정값", "측정"), ("설계 목표", "목표"), ("메모리", "gpu")),
    (("실행", "도구", "결과"), ("확인", "검증", "성공")),
    (("원인",), ("수정",), ("회귀", "테스트")),
    (("문맥", "대화"), ("의도", "사용자"), ("오래된", "요약", "줄")),
    (("불확실", "추측"), ("사실", "단정", "근거", "정확")),
    (("한국어", "답변", "문장"), ("완결", "문장", "종결"), ("반복", "자연", "문법")),
)
_PRODUCT_CONTINUATION_TURN = 9
_PRODUCT_CONTINUATION_COUNT = 1
_PRODUCT_FACTBOOK_TURNS = frozenset({1, 2, 3, 4, 16})
_PRODUCT_WORKER_EXPECTED_TURNS = frozenset(range(5, 21))
_PRODUCT_REQUIRED_TURN_CHECKS = frozenset(
    {
        "airgap_scope_respected",
        "balanced_smart_quotes",
        "canonical_user_prompt",
        "complete_stage",
        "contains_korean",
        "continuation_contract",
        "exactly_one_assistant",
        "factbook_model_exact",
        "factbook_parameters_exact",
        "factbook_version_exact",
        "finish_stop",
        "grounding_route",
        "intent_contract_satisfied",
        "interactive_latency_within_limit",
        "korean_complete",
        "natural_boundary",
        "no_control_marker",
        "no_cross_turn_exact_duplicate",
        "no_cross_turn_sentence_echo",
        "no_dangling_sentence_start",
        "no_false_7b_identity",
        "no_full_prompt_echo",
        "no_generic_outline",
        "no_instruction_echo",
        "no_meta_format_discussion",
        "no_near_duplicate_sentence",
        "no_placeholder_scaffolding",
        "no_repeated_paragraph",
        "no_repeated_sentence",
        "no_role_leak",
        "no_semantic_redundancy",
        "no_short_sentence_loop",
        "no_unsolicited_self_intro",
        "non_empty",
        "not_explicitly_truncated",
        "not_truncated",
        "post_turn_gpu_memory_spot_sample_observed_when_required",
        "post_turn_gpu_memory_spot_sample_within_limit_when_required",
        "post_turn_memory_spot_sample_observed_when_required",
        "post_turn_memory_spot_sample_scope_valid",
        "quality_report_accepts",
        "request_contract_fulfilled",
        "request_facets_covered",
        "requested_examples_present",
        "required_literal_ending",
        "required_period_ending",
        "session_isolated",
        "succeeded",
        "topic_anchors_satisfied",
        "worker_healthy",
    }
)
_PRODUCT_GPU_SPOT_SAMPLE_LIMIT_BYTES = int(16.7 * 1024**3)
_PRODUCT_MEMORY_SAMPLE_SCOPE = "post_turn_spot_sample"
_PRODUCT_MAX_ANSWER_CHARS = 8_192
_PRODUCT_MAX_TURN_SECONDS = 120.0
_PRODUCT_BUILD_VERSION = "0.4.1"
_PRODUCT_MODEL_LABEL = "gemma4-e4b-it"
_PRODUCT_STORED_PARAMETERS = 7_996_157_418
_PRODUCT_EFFECTIVE_PARAMETERS = 4_506_496_490
_PRODUCT_HANGUL = re.compile(r"[가-힣]")
_PRODUCT_SENTENCE = re.compile(r".+?(?:[.!?。！？]+(?=\s|$)|$)", re.DOTALL)
_PRODUCT_COMPLETE_ENDINGS = (".", "!", "?", "。", "！", "？", ".”", ".'", '."')
_PRODUCT_ROLE_LEAK = re.compile(
    r"(?im)^\s*(?:USER|ASSISTANT|SYSTEM|MODEL|TOOL|사용자|어시스턴트|시스템)\s*:"
)
_PRODUCT_CONTROL_MARKERS = (
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
_PRODUCT_DANGLING_KOREAN_RE = re.compile(
    r"(?:이는\s+내가|그\s+이유는|예를\s+들어|따라서|그리고|하지만|반면에|"
    r"하기\s+위해|수\s+있으며|것은|경우에는)\s*[.!?。！？]*\s*$"
)
_PRODUCT_TRAILING_LIST_MARKER_RE = re.compile(
    r"(?:^|\s)(?P<marker>[-*+]|\d{1,4}[.)])\s*$"
)
_PRODUCT_NUMBERED_ITEM_WITH_CONTENT_RE = re.compile(r"(?:^|\s)1[.)]\s+\S")
_GPU5_SCHEDULER_RESERVATION_ID = re.compile(r"[A-Za-z0-9._-]{16,128}\Z")
_SOURCE_CHECKOUT_ROOT = Path("/home/shoon/workspace/Cogni-OS-2.0-v041")
_BASE_CANARY_MODEL_ROOT = Path("/home/shoon/models/gemma4-e4b")
_PRODUCT_E4B_IT_MODEL_ROOT = Path("/home/shoon/models/gemma4-e4b-it")
_ALLOWED_MOUNT_ROOTS = (
    _SOURCE_CHECKOUT_ROOT,
    _BASE_CANARY_MODEL_ROOT,
    _PRODUCT_E4B_IT_MODEL_ROOT,
)
# Kept as the base-canary compatibility mapping for native helpers and existing
# inspection callers. Docker release paths select their raw model capability
# exclusively through the immutable artifact profile below.
_EXPECTED_READ_ONLY_MOUNTS = {
    _SOURCE_CHECKOUT_ROOT: Path("/workspace"),
    _BASE_CANARY_MODEL_ROOT: Path("/models/gemma4-e4b"),
}
_REQUIRED_CONTAINER_ENVIRONMENT = {
    "HF_HOME": "/tmp/cognios-hf",
    "HF_HUB_OFFLINE": "1",
    "HOME": "/tmp/cognios-home",
    "LD_PRELOAD": "",
    "NVIDIA_DRIVER_CAPABILITIES": "compute,utility",
    "NVIDIA_VISIBLE_DEVICES": PROJECT_GPU_UUID,
    "PATH": "/usr/local/bin:/usr/bin:/bin",
    "PYTHONDONTWRITEBYTECODE": "1",
    "PYTHONNOUSERSITE": "1",
    "PYTHONPATH": "/nonexistent-cognios-pythonpath",
    "PYTHONSAFEPATH": "1",
    "TRANSFORMERS_OFFLINE": "1",
}
_PINNED_CONTAINER_PYTHON = "/usr/local/bin/python"
_PINNED_CONTAINER_USER = "8001:8001"
_PINNED_CONTAINER_TMPFS = "/tmp:rw,nosuid,nodev,noexec,size=268435456,mode=1777"
_PINNED_CONTAINER_PIDS_LIMIT = "512"
_MINIMAL_HOST_ENVIRONMENT = {
    "GIT_CONFIG_GLOBAL": "/dev/null",
    "GIT_CONFIG_NOSYSTEM": "1",
    "GIT_OPTIONAL_LOCKS": "0",
    "HOME": "/nonexistent",
    "LANG": "C",
    "LC_ALL": "C",
    "PATH": "/usr/bin:/bin",
    "PYTHONNOUSERSITE": "1",
}
_PINNED_WORKDIR = "/workspace"
_SENSITIVE_MOUNT_SOURCES = (
    Path("/"),
    Path("/dev"),
    Path("/proc"),
    Path("/sys"),
    Path("/run"),
)
_EVIDENCE_FILENAME = re.compile(r"gpu5-[a-z0-9][a-z0-9._-]{0,79}\.(?:json|jsonl|log)\Z")
_ALLOWED_VALIDATORS = frozenset(
    {
        "/workspace/scripts/validate_agent_completion.py",
        "/workspace/scripts/validate_agent_casual_korean.py",
        "/workspace/scripts/validate_gemma4_deq.py",
        "/workspace/scripts/validate_gemma4_runtime.py",
    }
)
_VALIDATOR_VALUE_OPTIONS = {
    "/workspace/scripts/validate_agent_completion.py": frozenset(
        {
            "--model",
            "--manifest",
            "--timeout",
            "--physical-gpu-index",
            "--gpu-query-context",
            "--suite",
            "--turns",
        }
    ),
    "/workspace/scripts/validate_agent_casual_korean.py": frozenset(
        {
            "--model",
            "--manifest",
            "--timeout",
            "--physical-gpu-index",
            "--gpu-query-context",
        }
    ),
    "/workspace/scripts/validate_gemma4_runtime.py": frozenset(
        {
            "--model",
            "--manifest",
            "--prompt",
            "--workspace-mib",
            "--vram-limit-gib",
            "--physical-gpu-index",
            "--gpu-query-context",
        }
    ),
    "/workspace/scripts/validate_gemma4_deq.py": frozenset(
        {
            "--model",
            "--manifest",
            "--prompt",
            "--layer-index",
            "--tolerance",
            "--max-iter",
            "--history",
            "--fallback-steps",
            "--fallback-damping",
            "--contractive-delta-scale",
            "--certified-delta-lipschitz-bound",
            "--contractivity-provenance",
            "--vram-limit-gib",
            "--physical-gpu-index",
            "--gpu-query-context",
        }
    ),
}
_VALIDATOR_FLAG_OPTIONS = {
    "/workspace/scripts/validate_agent_completion.py": frozenset({"--strict-json"}),
    "/workspace/scripts/validate_agent_casual_korean.py": frozenset(),
    "/workspace/scripts/validate_gemma4_runtime.py": frozenset({"--event-stream"}),
    "/workspace/scripts/validate_gemma4_deq.py": frozenset(
        {"--allow-uncertified-experimental"}
    ),
}


@dataclass(frozen=True)
class ValidationArtifactProfile:
    """One immutable raw-model, manifest, mount and validator capability set."""

    name: str
    raw_model_root: Path
    container_model_root: Path
    manifest_relative_path: str
    validators: frozenset[str]


_VALIDATION_ARTIFACT_PROFILES: Mapping[str, ValidationArtifactProfile] = (
    MappingProxyType(
        {
            BASE_CANARY_ARTIFACT_PROFILE: ValidationArtifactProfile(
                name=BASE_CANARY_ARTIFACT_PROFILE,
                raw_model_root=_BASE_CANARY_MODEL_ROOT,
                container_model_root=Path("/models/gemma4-e4b"),
                manifest_relative_path="config/gemma4-e4b.manifest.toml",
                validators=frozenset(
                    {
                        "/workspace/scripts/validate_agent_completion.py",
                        "/workspace/scripts/validate_gemma4_deq.py",
                        "/workspace/scripts/validate_gemma4_runtime.py",
                    }
                ),
            ),
            PRODUCT_E4B_IT_ARTIFACT_PROFILE: ValidationArtifactProfile(
                name=PRODUCT_E4B_IT_ARTIFACT_PROFILE,
                raw_model_root=_PRODUCT_E4B_IT_MODEL_ROOT,
                container_model_root=Path("/models/gemma4-e4b-it"),
                manifest_relative_path="config/gemma4-e4b-it.manifest.toml",
                validators=frozenset(
                    {
                        "/workspace/scripts/validate_agent_completion.py",
                        "/workspace/scripts/validate_agent_casual_korean.py",
                    }
                ),
            ),
        }
    )
)
MAX_VALIDATOR_ARGV_TOKENS = 32
MAX_VALIDATOR_TOKEN_CHARS = 2_048
MAX_VALIDATOR_PROMPT_CHARS = 512
MAX_DEQ_RAW_DELTA_LIPSCHITZ_BOUND = 1.0e6
MAX_DEQ_EFFECTIVE_LIPSCHITZ_BOUND = 0.95
DEFAULT_DEQ_CONTRACTIVE_DELTA_SCALE = 0.05
MAX_CONTAINER_ENVIRONMENT_ITEMS = 16
MAX_CONTAINER_ENVIRONMENT_VALUE_CHARS = 4_096
MAX_CONTAINER_ENVIRONMENT_TOTAL_CHARS = 16_384
MAX_GIT_OUTPUT_BYTES = 32 * 1024 * 1024
MAX_MODEL_MANIFEST_BYTES = 64 * 1024
MAX_MODEL_MANIFEST_LINES = 256
MAX_MODEL_MANIFEST_ENTRIES = 128
MAX_MODEL_MANIFEST_PATH_CHARS = 512
MAX_SOURCE_SNAPSHOTS = 64
MAX_SOURCE_SNAPSHOT_FILES = 32_768
MAX_SOURCE_SNAPSHOT_BYTES = MAX_SOURCE_ARCHIVE_BYTES
MAX_SOURCE_SNAPSHOT_STORE_BYTES = 8 * 1024 * 1024 * 1024
MAX_MODEL_SNAPSHOTS = 3
MAX_MODEL_SNAPSHOT_BYTES = 32 * 1024 * 1024 * 1024
MAX_MODEL_SNAPSHOT_STORE_BYTES = 96 * 1024 * 1024 * 1024
MODEL_SNAPSHOT_COPY_CHUNK_BYTES = 1024 * 1024
MAX_FILESYSTEM_IDENTITY = (1 << 64) - 1
_SHA256_DIGEST = re.compile(r"[0-9a-f]{64}\Z")
_MODEL_SNAPSHOT_DIRECTORY = re.compile(
    r"model-(?P<manifest>[0-9a-f]{64})-(?P<nonce>[0-9a-f]{32})\Z"
)
EXPECTED_SOURCE_COMMIT_PATTERN = re.compile(r"[0-9a-f]{40}\Z")
_STRICT_MANIFEST_ENTRY = re.compile(
    r'"(?P<path>[A-Za-z0-9][A-Za-z0-9._/-]{0,511})" = '
    r'"(?P<digest>[0-9a-f]{64})"\Z'
)
_STRICT_MODEL_IDENTITY_ENTRY = re.compile(
    r"(?P<key>family|variant|role|source|revision) = "
    r'"(?P<value>[^"\\]{1,128})"\Z'
)
_MODEL_IDENTITY_KEYS = frozenset({"family", "variant", "role", "source", "revision"})


class GPU5BoundaryError(RuntimeError):
    """Raised before execution whenever the physical-GPU contract is uncertain."""


class ArtifactVerificationError(GPU5BoundaryError):
    """The pinned model manifest or closed-world layout failed verification."""


@dataclass(frozen=True)
class ArtifactIdentity:
    family: str
    variant: str
    role: str
    source: str
    revision: str


@dataclass(frozen=True)
class VerifiedArtifactSet:
    root: Path
    files: tuple[Path, ...]
    identity: ArtifactIdentity | None = None
    digests: tuple[tuple[str, str], ...] = ()


class GPU5DockerExecutionError(GPU5BoundaryError):
    """A fail-closed launch error with bounded machine-readable evidence."""

    def __init__(self, message: str, evidence: Mapping[str, Any]) -> None:
        super().__init__(message)
        self.evidence = dict(evidence)


class GPU5CleanupError(GPU5BoundaryError):
    """Cleanup failed after bounded best-effort actions were recorded."""

    def __init__(self, message: str, evidence: Mapping[str, Any]) -> None:
        super().__init__(message)
        self.evidence = dict(evidence)


class GPU5AggregateError(GPU5BoundaryError):
    """Python 3.10-compatible bounded secondary-failure record."""

    def __init__(
        self,
        message: str,
        failures: Sequence[BaseException],
        evidence_error: GPU5BoundaryError,
    ) -> None:
        if not 1 <= len(failures) <= 16:
            raise GPU5BoundaryError(
                "GPU5 aggregate failure count must remain in [1, 16]"
            )
        super().__init__(message)
        self.failures = tuple(failures)
        self.evidence_error = evidence_error
        self.evidence = dict(getattr(evidence_error, "evidence", {}))


def _flatten_failure_objects(error: BaseException) -> list[BaseException]:
    """Return original phase failures linked through our bounded aggregate cause."""

    cause = error.__cause__
    if isinstance(cause, GPU5AggregateError):
        return [error, *cause.failures]
    return [error]


def _attach_failure_evidence(
    error: BaseException,
    attribute: str,
    evidence_error: GPU5BoundaryError,
) -> bool:
    """Attach structured evidence while preserving a single fatal object exactly."""

    try:
        setattr(error, attribute, evidence_error)
    except (AttributeError, TypeError):
        return False
    return True


def _cleanup_evidence_from_error(error: BaseException) -> dict[str, Any] | None:
    """Recover cleanup evidence from a direct, attached, or grouped failure."""

    if isinstance(error, GPU5CleanupError):
        return dict(error.evidence)
    attached = getattr(error, "gpu5_cleanup_error", None)
    if isinstance(attached, GPU5CleanupError):
        return dict(attached.evidence)
    cause = error.__cause__
    if isinstance(cause, GPU5AggregateError):
        if isinstance(cause.evidence_error, GPU5CleanupError):
            return dict(cause.evidence_error.evidence)
        for child in cause.failures:
            evidence = _cleanup_evidence_from_error(child)
            if evidence is not None:
                return evidence
    return None


def _secondary_failures(
    failures: Sequence[BaseException], primary: BaseException
) -> list[BaseException]:
    """Remove one identity-equal primary while retaining deterministic order."""

    secondary: list[BaseException] = []
    removed = False
    for error in failures:
        if not removed and error is primary:
            removed = True
            continue
        secondary.append(error)
    return secondary


def _raise_cleanup_failures(
    message: str,
    payload: Mapping[str, Any],
    caught_errors: Sequence[BaseException],
) -> None:
    """Preserve fatal controls and keep the structured cleanup record reachable."""

    cleanup_failure = GPU5CleanupError(message, payload)
    failures: list[BaseException] = []
    for error in caught_errors:
        failures.extend(_flatten_failure_objects(error))
    fatal_controls = [error for error in failures if not isinstance(error, Exception)]
    structured_errors = payload.get("errors")
    one_structured_error = (
        isinstance(structured_errors, list) and len(structured_errors) == 1
    )
    if (
        len(failures) == 1
        and len(fatal_controls) == 1
        and one_structured_error
        and _attach_failure_evidence(
            fatal_controls[0], "gpu5_cleanup_error", cleanup_failure
        )
    ):
        raise fatal_controls[0]
    if fatal_controls:
        primary = fatal_controls[0]
        aggregate = GPU5AggregateError(
            "GPU5 cleanup control and safety failures",
            [*_secondary_failures(failures, primary), cleanup_failure],
            cleanup_failure,
        )
        _attach_failure_evidence(primary, "gpu5_cleanup_error", cleanup_failure)
        _attach_failure_evidence(primary, "gpu5_aggregate_error", aggregate)
        raise primary from aggregate
    if failures:
        raise cleanup_failure from failures[0]
    raise cleanup_failure


@dataclass(frozen=True)
class GPUComputeProcess:
    gpu_uuid: str
    pid: int
    used_memory_mib: int | None


@dataclass(frozen=True)
class GPU5Snapshot:
    physical_index: int
    uuid: str
    utilization_percent: int
    memory_used_mib: int
    memory_total_mib: int
    compute_processes: tuple[GPUComputeProcess, ...]

    def as_payload(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class GuardedDockerResult:
    argv: tuple[str, ...]
    preflight: GPU5Snapshot
    postflight: GPU5Snapshot
    image_digest: str
    returncode: int
    evidence_path: str
    evidence_bytes: int
    evidence_sha256: str
    source_commit: str
    source_tree_digest: str
    source_identity_digest: str
    model_manifest_sha256: str
    model_tree_digest: str
    model_identity_digest: str
    container_name: str
    launch_nonce: str
    snapshot_path: str
    snapshot_mode: int
    cleanup: dict[str, Any]
    validation_artifact_profile: str = BASE_CANARY_ARTIFACT_PROFILE
    evidence_component_schema: str | None = None
    output_policy: str = "bounded_file_capture"


@dataclass(frozen=True)
class ExecutionScope:
    source_commit: str
    source_tree_digest: str
    source_identity_digest: str
    source_file_count: int
    source_root_device: int
    source_root_inode: int
    model_manifest_sha256: str
    model_tree_digest: str
    model_identity_digest: str
    model_file_count: int
    model_root_device: int
    model_root_inode: int
    validation_artifact_profile: str = BASE_CANARY_ARTIFACT_PROFILE
    snapshot_path: str = ""
    snapshot_nonce: str = ""
    snapshot_mode: int = 0
    working_tree_digest: str = ""
    working_identity_digest: str = ""

    def as_payload(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class GPU5Identity:
    physical_index: int
    uuid: str
    query_context: str
    logical_device_count: int
    logical_device_index: int

    def as_payload(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class _EvidenceHandle:
    root_path: Path
    filename: str
    root_fd: int
    file_fd: int
    root_device: int
    root_inode: int
    file_device: int
    file_inode: int


@dataclass(frozen=True)
class _OwnedContainer:
    container_id: str
    name: str
    labels: dict[str, str]


@dataclass
class _GPU5ProjectLease:
    _launch_may_have_occurred: bool = False
    _safe_to_release: bool = False

    def mark_launch_attempted(self) -> None:
        """Poison by default from the instant Docker may have accepted a run."""

        self._launch_may_have_occurred = True

    def mark_safe_to_release(self) -> None:
        """Commit release only after every required safety proof succeeded."""

        self._safe_to_release = True


@dataclass(frozen=True)
class WorkingCheckoutIdentity:
    source_commit: str
    content_digest: str
    identity_digest: str
    file_count: int


@dataclass(frozen=True)
class SourceSnapshot:
    source_commit: str
    launch_nonce: str
    root_path: str
    root_device: int
    root_inode: int
    root_mode: int
    file_count: int
    content_digest: str
    identity_digest: str

    def as_payload(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ModelSnapshot:
    root_path: str
    root_device: int
    root_inode: int
    root_mode: int
    file_count: int
    total_bytes: int
    manifest_sha256: str
    content_digest: str
    identity_digest: str

    def as_payload(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class NativeExecutionSnapshot:
    source: SourceSnapshot
    model: ModelSnapshot
    manifest_path: str
    workspace_root: str

    def as_payload(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class NativeGPU5ServerAuthority:
    """One-use authority held by the stdlib-only server bootstrap."""

    expected_source_commit: str
    physical_gpu_index: int
    gpu_query_context: str
    gpu_uuid: str
    preflight: GPU5Snapshot
    checkout: WorkingCheckoutIdentity
    execution_snapshot: NativeExecutionSnapshot
    _lease: _GPU5ProjectLease
    _consumed: bool = False
    _launch_marked: bool = False
    _safe_to_release: bool = False

    def consume(
        self,
        *,
        expected_source_commit: str,
        physical_gpu_index: int,
        gpu_query_context: str,
        gpu_uuid: str,
        source_snapshot_root: str,
        source_snapshot_nonce: str,
        model_snapshot_root: str,
        model_manifest_path: str,
        model_manifest_sha256: str,
        workspace_root: str,
        source_content_digest: str,
        source_identity_digest: str,
        source_file_count: int,
        source_root_device: int,
        source_root_inode: int,
        model_content_digest: str,
        model_identity_digest: str,
        model_file_count: int,
        model_root_device: int,
        model_root_inode: int,
        model_total_bytes: int,
    ) -> None:
        if self._consumed:
            raise GPU5BoundaryError("native GPU5 server authority is already consumed")
        handoff = _validated_native_snapshot_handoff(
            source_content_digest=source_content_digest,
            source_identity_digest=source_identity_digest,
            source_file_count=source_file_count,
            source_root_device=source_root_device,
            source_root_inode=source_root_inode,
            model_content_digest=model_content_digest,
            model_identity_digest=model_identity_digest,
            model_file_count=model_file_count,
            model_root_device=model_root_device,
            model_root_inode=model_root_inode,
            model_total_bytes=model_total_bytes,
        )
        if (
            expected_source_commit != self.expected_source_commit
            or physical_gpu_index != self.physical_gpu_index
            or gpu_query_context != self.gpu_query_context
            or gpu_uuid != self.gpu_uuid
            or source_snapshot_root != self.execution_snapshot.source.root_path
            or source_snapshot_nonce != self.execution_snapshot.source.launch_nonce
            or model_snapshot_root != self.execution_snapshot.model.root_path
            or model_manifest_path != self.execution_snapshot.manifest_path
            or model_manifest_sha256 != self.execution_snapshot.model.manifest_sha256
            or workspace_root != self.execution_snapshot.workspace_root
            or handoff != _native_snapshot_handoff(self.execution_snapshot)
        ):
            raise GPU5BoundaryError(
                "native GPU5 server arguments do not match early authority"
            )
        self._consumed = True

    def mark_launch_attempted(self) -> None:
        if not self._consumed or self._launch_marked:
            raise GPU5BoundaryError("native GPU5 launch authority is out of order")
        self._lease.mark_launch_attempted()
        self._launch_marked = True

    def mark_safe_to_release(self) -> None:
        if not self._consumed or not self._launch_marked or self._safe_to_release:
            raise GPU5BoundaryError("native GPU5 release authority is out of order")
        self._lease.mark_safe_to_release()
        self._safe_to_release = True


def require_project_gpu_index(value: object) -> int:
    """Accept only this project's physical GPU, never a lab-neighbour device."""

    if isinstance(value, bool) or not isinstance(value, int):
        raise GPU5BoundaryError("an explicit integer physical GPU index is required")
    if value not in LAB_ALLOWED_PHYSICAL_GPU_INDICES:
        raise GPU5BoundaryError("physical GPU index is outside the laboratory boundary")
    if value != PROJECT_PHYSICAL_GPU_INDEX:
        raise GPU5BoundaryError("this project is pinned to physical GPU 5")
    return value


def _validated_expected_source_commit(value: object) -> str:
    if (
        not isinstance(value, str)
        or EXPECTED_SOURCE_COMMIT_PATTERN.fullmatch(value) is None
    ):
        raise GPU5BoundaryError(
            "an exact lowercase 40-character source commit is required"
        )
    return value


def _validated_native_snapshot_digest(value: object, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256_DIGEST.fullmatch(value) is None:
        raise GPU5BoundaryError(f"{label} must be an exact lowercase SHA-256 digest")
    return value


def _validated_native_snapshot_integer(
    value: object,
    *,
    label: str,
    maximum: int,
    minimum: int = 0,
) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < minimum
        or value > maximum
    ):
        raise GPU5BoundaryError(f"{label} must be an integer in [{minimum}, {maximum}]")
    return value


def _validated_native_snapshot_handoff(
    *,
    source_content_digest: object,
    source_identity_digest: object,
    source_file_count: object,
    source_root_device: object,
    source_root_inode: object,
    model_content_digest: object,
    model_identity_digest: object,
    model_file_count: object,
    model_root_device: object,
    model_root_inode: object,
    model_total_bytes: object,
) -> tuple[str, str, int, int, int, str, str, int, int, int, int]:
    """Canonicalize the prepare-to-sealed identity capability.

    Paths alone are not capabilities: an attacker able to replace a path
    between the two bootstrap processes must also be unable to make the
    sealed process silently accept the replacement's freshly computed
    identity.  These values are therefore supplied by the prepare process and
    compared with a complete re-inventory before GPU preflight.
    """

    return (
        _validated_native_snapshot_digest(
            source_content_digest,
            label="native source content digest",
        ),
        _validated_native_snapshot_digest(
            source_identity_digest,
            label="native source identity digest",
        ),
        _validated_native_snapshot_integer(
            source_file_count,
            label="native source file count",
            maximum=MAX_SOURCE_SNAPSHOT_FILES,
        ),
        _validated_native_snapshot_integer(
            source_root_device,
            label="native source root device",
            maximum=MAX_FILESYSTEM_IDENTITY,
        ),
        _validated_native_snapshot_integer(
            source_root_inode,
            label="native source root inode",
            maximum=MAX_FILESYSTEM_IDENTITY,
            minimum=1,
        ),
        _validated_native_snapshot_digest(
            model_content_digest,
            label="native model content digest",
        ),
        _validated_native_snapshot_digest(
            model_identity_digest,
            label="native model identity digest",
        ),
        _validated_native_snapshot_integer(
            model_file_count,
            label="native model file count",
            maximum=MAX_MODEL_MANIFEST_ENTRIES,
            minimum=1,
        ),
        _validated_native_snapshot_integer(
            model_root_device,
            label="native model root device",
            maximum=MAX_FILESYSTEM_IDENTITY,
        ),
        _validated_native_snapshot_integer(
            model_root_inode,
            label="native model root inode",
            maximum=MAX_FILESYSTEM_IDENTITY,
            minimum=1,
        ),
        _validated_native_snapshot_integer(
            model_total_bytes,
            label="native model total bytes",
            maximum=MAX_MODEL_SNAPSHOT_BYTES,
        ),
    )


def _native_snapshot_handoff(
    snapshot: NativeExecutionSnapshot,
) -> tuple[str, str, int, int, int, str, str, int, int, int, int]:
    if not isinstance(snapshot, NativeExecutionSnapshot):
        raise GPU5BoundaryError("native execution snapshot type is invalid")
    return (
        snapshot.source.content_digest,
        snapshot.source.identity_digest,
        snapshot.source.file_count,
        snapshot.source.root_device,
        snapshot.source.root_inode,
        snapshot.model.content_digest,
        snapshot.model.identity_digest,
        snapshot.model.file_count,
        snapshot.model.root_device,
        snapshot.model.root_inode,
        snapshot.model.total_bytes,
    )


def _validated_executable(path: Path) -> str:
    if not isinstance(path, Path) or not path.is_absolute():
        raise GPU5BoundaryError("host executable path must be absolute")
    try:
        unresolved = path.lstat()
        resolved = path.resolve(strict=True)
        metadata = resolved.stat()
    except OSError as error:
        raise GPU5BoundaryError(
            f"required host executable is unavailable: {path}"
        ) from error
    if stat.S_ISLNK(unresolved.st_mode) or resolved != path:
        raise GPU5BoundaryError(f"host executable must already be its realpath: {path}")
    if not stat.S_ISREG(metadata.st_mode) or not os.access(resolved, os.X_OK):
        raise GPU5BoundaryError(f"host executable is not a regular executable: {path}")
    if metadata.st_mode & (stat.S_IWGRP | stat.S_IWOTH):
        raise GPU5BoundaryError(f"host executable is group/world writable: {path}")
    return str(resolved)


def _effective_uid() -> int:
    """Return the POSIX effective UID required by the Linux host trust gate."""

    getter = getattr(os, "geteuid", None)
    if not callable(getter):
        raise GPU5BoundaryError("trusted path ownership requires a POSIX effective UID")
    value = getter()
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise GPU5BoundaryError("the POSIX effective UID is invalid")
    return value


def _trusted_owner_uid(uid: object) -> bool:
    """Accept immutable operator assets owned only by root or this service UID."""

    return (
        isinstance(uid, int)
        and not isinstance(uid, bool)
        and uid
        in {
            0,
            _effective_uid(),
        }
    )


def _group_or_world_writable(mode: int) -> bool:
    return bool(mode & (stat.S_IWGRP | stat.S_IWOTH))


def _trusted_directory_chain(path: Path, *, label: str) -> Path:
    """Reject symlinked or peer-writable roots and every lexical parent.

    A sticky writable ancestor such as ``/tmp`` is permitted only above the
    requested root.  Its sticky bit prevents another UID from renaming the
    service-owned child; the requested root itself must never be group/world
    writable.  Production paths therefore require a root/euid-owned, sealed
    chain from ``/`` through the exact source/model directory.
    """

    if not isinstance(path, Path) or not path.is_absolute():
        raise GPU5BoundaryError(f"{label} must be an absolute path")
    normalized = Path(os.path.normpath(os.fspath(path)))
    if normalized != path:
        raise GPU5BoundaryError(f"{label} must be a canonical lexical path")
    current = Path(path.anchor)
    components = path.parts[1:]
    candidates = (
        current,
        *(
            current.joinpath(*components[:index])
            for index in range(1, len(components) + 1)
        ),
    )
    for candidate in candidates:
        try:
            metadata = candidate.lstat()
        except OSError as error:
            raise GPU5BoundaryError(
                f"{label} path component is unavailable: {candidate}"
            ) from error
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
            raise GPU5BoundaryError(
                f"{label} path component is not a real directory: {candidate}"
            )
        if not _trusted_owner_uid(metadata.st_uid):
            raise GPU5BoundaryError(
                f"{label} path component has an untrusted owner: {candidate}"
            )
        if _group_or_world_writable(metadata.st_mode):
            sticky_ancestor = candidate != path and bool(
                metadata.st_mode & stat.S_ISVTX
            )
            if not sticky_ancestor:
                raise GPU5BoundaryError(
                    f"{label} path component is group/world writable: {candidate}"
                )
    try:
        resolved = path.resolve(strict=True)
    except OSError as error:
        raise GPU5BoundaryError(f"{label} could not be resolved") from error
    if resolved != path:
        raise GPU5BoundaryError(f"{label} must already be its exact realpath")
    return resolved


def validate_trusted_import_directory(path: str | Path) -> Path:
    """Validate one existing host import directory before bootstrap uses it.

    The public wrapper deliberately exposes only the strict owner, mode,
    realpath, and ancestor-chain proof.  It does not add the directory to
    ``sys.path`` or otherwise grant import authority itself.
    """

    if not isinstance(path, (str, Path)):
        raise GPU5BoundaryError("trusted import directory must be a path")
    return _trusted_directory_chain(
        Path(path),
        label="trusted import directory",
    )


def _minimal_host_environment() -> dict[str, str]:
    """Return a non-inheriting environment for all host control binaries."""

    return dict(_MINIMAL_HOST_ENVIRONMENT)


def _validated_docker_socket() -> str:
    try:
        unresolved = DOCKER_SOCKET_PATH.lstat()
        resolved = DOCKER_SOCKET_PATH.resolve(strict=True)
        metadata = resolved.stat()
    except OSError as error:
        raise GPU5BoundaryError(
            "the pinned local Docker socket is unavailable"
        ) from error
    if stat.S_ISLNK(unresolved.st_mode) or resolved != DOCKER_SOCKET_PATH:
        raise GPU5BoundaryError("Docker socket must already be its exact realpath")
    if not stat.S_ISSOCK(metadata.st_mode):
        raise GPU5BoundaryError("the pinned Docker endpoint is not a Unix socket")
    return DOCKER_HOST_URI


def _ensure_private_directory(path: Path, *, create: bool) -> Path:
    if not path.is_absolute():
        raise GPU5BoundaryError("guard state paths must be absolute")
    try:
        if create:
            path.mkdir(mode=0o700, parents=False, exist_ok=True)
        unresolved = path.lstat()
        resolved = path.resolve(strict=True)
        metadata = resolved.stat()
    except OSError as error:
        raise GPU5BoundaryError(f"guard directory is unavailable: {path}") from error
    if stat.S_ISLNK(unresolved.st_mode) or resolved != path:
        raise GPU5BoundaryError(f"guard directory must already be its realpath: {path}")
    if not stat.S_ISDIR(metadata.st_mode) or metadata.st_uid != os.geteuid():
        raise GPU5BoundaryError(f"guard directory ownership is invalid: {path}")
    if metadata.st_mode & 0o077:
        raise GPU5BoundaryError(f"guard directory must have mode 0700: {path}")
    return resolved


def _prepare_guard_state() -> None:
    parent = GUARD_STATE_ROOT.parent.resolve(strict=True)
    if parent != GUARD_STATE_PARENT.resolve(strict=True):
        raise GPU5BoundaryError(
            "guard state parent is not the pinned project owner home"
        )
    if not GUARD_STATE_ROOT.exists():
        try:
            os.mkdir(GUARD_STATE_ROOT, 0o700)
        except OSError as error:
            raise GPU5BoundaryError(
                "could not create the private GPU5 guard root"
            ) from error
    _ensure_private_directory(GUARD_STATE_ROOT, create=False)
    if not DOCKER_CONFIG_ROOT.exists():
        try:
            os.mkdir(DOCKER_CONFIG_ROOT, 0o700)
        except OSError as error:
            raise GPU5BoundaryError(
                "could not create the empty Docker config root"
            ) from error
    config = _ensure_private_directory(DOCKER_CONFIG_ROOT, create=False)
    try:
        entries = tuple(config.iterdir())
    except OSError as error:
        raise GPU5BoundaryError(
            "Docker config root could not be inventoried"
        ) from error
    if entries:
        raise GPU5BoundaryError("Docker config root must remain exactly empty")
    for private_root in (
        EVIDENCE_HOST_ROOT,
        SOURCE_SNAPSHOT_ROOT,
        MODEL_SNAPSHOT_ROOT,
    ):
        if not private_root.exists():
            try:
                os.mkdir(private_root, 0o700)
            except OSError as error:
                raise GPU5BoundaryError(
                    f"could not create private guard directory: {private_root}"
                ) from error
        _ensure_private_directory(private_root, create=False)

    repo_root = Path(
        next(
            source
            for source, destination in _EXPECTED_READ_ONLY_MOUNTS.items()
            if destination == Path("/workspace")
        )
    ).resolve(strict=True)
    model_source_root = Path(
        next(
            source
            for source, destination in _EXPECTED_READ_ONLY_MOUNTS.items()
            if destination == Path("/models/gemma4-e4b")
        )
    ).resolve(strict=True)
    private_roots = (
        EVIDENCE_HOST_ROOT.resolve(strict=True),
        SOURCE_SNAPSHOT_ROOT.resolve(strict=True),
        MODEL_SNAPSHOT_ROOT.resolve(strict=True),
    )
    protected_roots = (repo_root, model_source_root)
    for private_root in private_roots:
        for protected_root in protected_roots:
            if (
                private_root == protected_root
                or private_root.is_relative_to(protected_root)
                or protected_root.is_relative_to(private_root)
            ):
                raise GPU5BoundaryError(
                    "guard snapshot/evidence roots must remain outside mutable inputs"
                )
    for index, private_root in enumerate(private_roots):
        for other in private_roots[index + 1 :]:
            if (
                private_root == other
                or private_root.is_relative_to(other)
                or other.is_relative_to(private_root)
            ):
                raise GPU5BoundaryError(
                    "guard evidence/source/model snapshot roots must not overlap"
                )


@contextmanager
def _gpu5_project_lease(expected_source_commit: str) -> Iterator[_GPU5ProjectLease]:
    """Hold a crash-detecting, project-specific host lease for the entire run."""

    _validated_expected_source_commit(expected_source_commit)
    if fcntl is None:
        raise GPU5BoundaryError("GPU5 project lease requires Linux flock")
    _prepare_guard_state()
    flags = os.O_RDWR | os.O_CREAT
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(GPU5_LEASE_PATH, flags, 0o600)
    except OSError as error:
        raise GPU5BoundaryError(
            "GPU5 project lease file could not be opened"
        ) from error
    acquired = False
    try:
        metadata = os.fstat(descriptor)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_uid != os.geteuid()
            or metadata.st_nlink != 1
            or metadata.st_mode & 0o077
        ):
            raise GPU5BoundaryError("GPU5 project lease file has unsafe metadata")
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            raise GPU5BoundaryError("GPU5 project lease is already held") from error
        acquired = True
        os.lseek(descriptor, 0, os.SEEK_SET)
        stale = os.read(descriptor, 4_096)
        if stale:
            raise GPU5BoundaryError(
                "stale GPU5 project lease requires explicit operator review"
            )
        payload = json.dumps(
            {
                "pid": os.getpid(),
                "source_commit": expected_source_commit,
                "started_unix_ns": time_ns(),
            },
            sort_keys=True,
        ).encode("ascii")
        os.lseek(descriptor, 0, os.SEEK_SET)
        os.write(descriptor, payload)
        os.ftruncate(descriptor, len(payload))
        os.fsync(descriptor)
        lease = _GPU5ProjectLease()
        try:
            yield lease
        except BaseException:
            # Once exact cleanup/postflight proofs have committed safety, a
            # later application failure must not manufacture a stale lease.
            clear_payload = (
                not lease._launch_may_have_occurred or lease._safe_to_release
            )
            if clear_payload:
                os.lseek(descriptor, 0, os.SEEK_SET)
                os.ftruncate(descriptor, 0)
                os.fsync(descriptor)
            raise
        if not lease._launch_may_have_occurred or lease._safe_to_release:
            os.lseek(descriptor, 0, os.SEEK_SET)
            os.ftruncate(descriptor, 0)
            os.fsync(descriptor)
    finally:
        if acquired:
            try:
                fcntl.flock(descriptor, fcntl.LOCK_UN)
            except OSError:
                pass
        os.close(descriptor)


@contextmanager
def gpu5_project_lease(
    expected_source_commit: str,
) -> Iterator[_GPU5ProjectLease]:
    """Expose the audited host flock contract to the native product server."""

    with _gpu5_project_lease(expected_source_commit) as lease:
        yield lease


def native_gpu5_environment(
    base: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Return a native-host environment exposing physical GPU 5 only."""

    environment = dict(os.environ if base is None else base)
    configured = environment.get("CUDA_VISIBLE_DEVICES")
    if configured is not None and configured.strip() != PROJECT_GPU_UUID:
        raise GPU5BoundaryError("native CUDA visibility conflicts with physical GPU 5")
    device_order = environment.get("CUDA_DEVICE_ORDER")
    if device_order is not None and device_order.strip() != "PCI_BUS_ID":
        raise GPU5BoundaryError("native CUDA device order must be exactly PCI_BUS_ID")
    nvidia_configured = environment.get("NVIDIA_VISIBLE_DEVICES")
    if nvidia_configured is not None and nvidia_configured.strip() != PROJECT_GPU_UUID:
        raise GPU5BoundaryError(
            "native NVIDIA visibility conflicts with physical GPU 5"
        )
    environment["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    environment["CUDA_VISIBLE_DEVICES"] = PROJECT_GPU_UUID
    environment["NVIDIA_VISIBLE_DEVICES"] = PROJECT_GPU_UUID
    return environment


def _nvidia_smi_gpu_argv() -> tuple[str, ...]:
    return (
        _validated_executable(NVIDIA_SMI_EXECUTABLE),
        "-i",
        str(PROJECT_PHYSICAL_GPU_INDEX),
        f"--query-gpu={','.join(_GPU_QUERY_FIELDS)}",
        "--format=csv,noheader,nounits",
    )


def _nvidia_smi_compute_argv() -> tuple[str, ...]:
    return (
        _validated_executable(NVIDIA_SMI_EXECUTABLE),
        "-i",
        str(PROJECT_PHYSICAL_GPU_INDEX),
        f"--query-compute-apps={','.join(_COMPUTE_QUERY_FIELDS)}",
        "--format=csv,noheader,nounits",
    )


def _run_nvidia_smi(
    argv: tuple[str, ...],
    *,
    runner: Any = subprocess.run,
) -> str:
    executable = _validated_executable(NVIDIA_SMI_EXECUTABLE)
    if argv[:3] != (executable, "-i", "5"):
        raise GPU5BoundaryError("unsafe nvidia-smi command rejected")
    try:
        completed = runner(
            list(argv),
            capture_output=True,
            text=True,
            timeout=NVIDIA_SMI_TIMEOUT_SECONDS,
            check=False,
            env=_minimal_host_environment(),
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise GPU5BoundaryError(
            f"bounded GPU5 inspection failed: {type(error).__name__}"
        ) from error
    if not isinstance(completed.stdout, str) or not isinstance(completed.stderr, str):
        raise GPU5BoundaryError("nvidia-smi did not return bounded text output")
    if (
        len(completed.stdout) > MAX_NVIDIA_SMI_OUTPUT_CHARS
        or len(completed.stderr) > MAX_NVIDIA_SMI_OUTPUT_CHARS
    ):
        raise GPU5BoundaryError("nvidia-smi output exceeded the safety bound")
    if completed.returncode != 0:
        raise GPU5BoundaryError(
            f"nvidia-smi GPU5 query failed with exit {completed.returncode}"
        )
    return completed.stdout


def validate_guarded_gpu5_identity(
    *,
    physical_gpu_index: object,
    gpu_query_context: object,
    torch_module: Any,
    runner: Any = subprocess.run,
) -> GPU5Identity:
    """Bind a guarded validator to physical GPU5 before model allocation."""

    selected_index = require_project_gpu_index(physical_gpu_index)
    context = str(gpu_query_context)
    if context == "native-host":
        if os.environ.get("CUDA_DEVICE_ORDER") != "PCI_BUS_ID":
            raise GPU5BoundaryError(
                "native GPU5 validation requires CUDA_DEVICE_ORDER=PCI_BUS_ID "
                "before Python starts"
            )
        if os.environ.get("CUDA_VISIBLE_DEVICES") != PROJECT_GPU_UUID:
            raise GPU5BoundaryError(
                "native GPU5 validation requires the pinned UUID in "
                "CUDA_VISIBLE_DEVICES before Python starts"
            )
        if os.environ.get("NVIDIA_VISIBLE_DEVICES") != PROJECT_GPU_UUID:
            raise GPU5BoundaryError(
                "native GPU5 validation requires the pinned NVIDIA_VISIBLE_DEVICES UUID"
            )
        selector = str(selected_index)
    elif context == "gpu5-container":
        cuda_visibility = os.environ.get("CUDA_VISIBLE_DEVICES")
        nvidia_visibility = os.environ.get("NVIDIA_VISIBLE_DEVICES")
        if cuda_visibility not in {None, ""}:
            raise GPU5BoundaryError(
                "GPU5 container must not override its single remapped CUDA device"
            )
        if nvidia_visibility not in {None, "", PROJECT_GPU_UUID}:
            raise GPU5BoundaryError(
                "GPU5 container NVIDIA visibility is not the pinned UUID"
            )
        selector = PROJECT_GPU_UUID
    else:
        raise GPU5BoundaryError("an explicit guarded GPU5 query context is required")

    executable = _validated_executable(NVIDIA_SMI_EXECUTABLE)
    argv = (
        executable,
        "-i",
        selector,
        "--query-gpu=index,uuid",
        "--format=csv,noheader,nounits",
    )
    try:
        completed = runner(
            list(argv),
            capture_output=True,
            text=True,
            timeout=NVIDIA_SMI_TIMEOUT_SECONDS,
            check=False,
            env=_minimal_host_environment(),
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise GPU5BoundaryError(
            f"guarded GPU5 identity query failed: {type(error).__name__}"
        ) from error
    if (
        completed.returncode != 0
        or not isinstance(completed.stdout, str)
        or not isinstance(completed.stderr, str)
        or len(completed.stdout) > MAX_NVIDIA_SMI_OUTPUT_CHARS
        or len(completed.stderr) > MAX_NVIDIA_SMI_OUTPUT_CHARS
        or completed.stderr.strip()
    ):
        raise GPU5BoundaryError("guarded GPU5 identity query returned invalid output")
    rows = _csv_rows(completed.stdout)
    if len(rows) != 1 or len(rows[0]) != 2:
        raise GPU5BoundaryError("guarded GPU5 identity must return exactly one row")
    index_raw, uuid = rows[0]
    if context == "native-host":
        require_project_gpu_index(_nonnegative_integer(index_raw, "physical GPU index"))
    elif re.fullmatch(r"[0-9]+", index_raw) is None:
        raise GPU5BoundaryError("container GPU identity returned an invalid index")
    if uuid != PROJECT_GPU_UUID:
        raise GPU5BoundaryError("guarded validator GPU UUID mismatch")

    if not torch_module.cuda.is_available():
        raise GPU5BoundaryError("guarded GPU5 validator requires CUDA")
    logical_count = int(torch_module.cuda.device_count())
    logical_index = int(torch_module.cuda.current_device())
    if logical_count != 1 or logical_index != 0:
        raise GPU5BoundaryError(
            "guarded GPU5 validator must expose exactly one logical cuda:0 device"
        )
    return GPU5Identity(
        physical_index=selected_index,
        uuid=uuid,
        query_context=context,
        logical_device_count=logical_count,
        logical_device_index=logical_index,
    )


def _csv_rows(text: str) -> list[list[str]]:
    try:
        return [
            [field.strip() for field in row]
            for row in csv.reader(text.splitlines())
            if row and any(field.strip() for field in row)
        ]
    except csv.Error as error:
        raise GPU5BoundaryError("malformed nvidia-smi CSV") from error


def _nonnegative_integer(value: str, field: str) -> int:
    if re.fullmatch(r"[0-9]+", value) is None:
        raise GPU5BoundaryError(f"invalid {field} from nvidia-smi")
    parsed = int(value)
    if parsed < 0:
        raise GPU5BoundaryError(f"negative {field} from nvidia-smi")
    return parsed


def _optional_memory(value: str) -> int | None:
    if value.casefold() in {"n/a", "[n/a]", "not supported"}:
        return None
    return _nonnegative_integer(value, "compute process memory")


def query_gpu5_snapshot(*, runner: Any = subprocess.run) -> GPU5Snapshot:
    """Query only GPU5 and validate its immutable identity and bounded telemetry."""

    gpu_rows = _csv_rows(_run_nvidia_smi(_nvidia_smi_gpu_argv(), runner=runner))
    if len(gpu_rows) != 1 or len(gpu_rows[0]) != len(_GPU_QUERY_FIELDS):
        raise GPU5BoundaryError("GPU5 query must return exactly one complete row")
    index_raw, uuid, utilization_raw, used_raw, total_raw = gpu_rows[0]
    index = _nonnegative_integer(index_raw, "physical GPU index")
    require_project_gpu_index(index)
    if uuid != PROJECT_GPU_UUID:
        raise GPU5BoundaryError("physical GPU5 UUID mismatch")
    utilization = _nonnegative_integer(utilization_raw, "GPU utilization")
    memory_used = _nonnegative_integer(used_raw, "GPU memory used")
    memory_total = _nonnegative_integer(total_raw, "GPU memory total")
    if utilization > 100 or memory_total <= 0 or memory_used > memory_total:
        raise GPU5BoundaryError("invalid GPU5 utilization or memory telemetry")

    process_rows = _csv_rows(_run_nvidia_smi(_nvidia_smi_compute_argv(), runner=runner))
    processes: list[GPUComputeProcess] = []
    seen_pids: set[int] = set()
    for row in process_rows:
        if len(row) != len(_COMPUTE_QUERY_FIELDS):
            raise GPU5BoundaryError("malformed GPU5 compute-process row")
        process_uuid, pid_raw, process_memory_raw = row
        if process_uuid != PROJECT_GPU_UUID:
            raise GPU5BoundaryError("compute-process GPU UUID mismatch")
        pid = _nonnegative_integer(pid_raw, "compute process PID")
        if pid <= 0 or pid in seen_pids:
            raise GPU5BoundaryError("invalid or duplicate GPU5 compute process PID")
        seen_pids.add(pid)
        processes.append(
            GPUComputeProcess(
                gpu_uuid=process_uuid,
                pid=pid,
                used_memory_mib=_optional_memory(process_memory_raw),
            )
        )
    return GPU5Snapshot(
        physical_index=index,
        uuid=uuid,
        utilization_percent=utilization,
        memory_used_mib=memory_used,
        memory_total_mib=memory_total,
        compute_processes=tuple(processes),
    )


def assert_gpu5_idle(
    snapshot: GPU5Snapshot,
    *,
    max_driver_memory_mib: int = MAX_IDLE_DRIVER_MEMORY_MIB,
) -> None:
    """Fail closed on utilization, non-driver memory, or any compute PID."""

    require_project_gpu_index(snapshot.physical_index)
    if snapshot.uuid != PROJECT_GPU_UUID:
        raise GPU5BoundaryError("physical GPU5 UUID mismatch")
    if isinstance(max_driver_memory_mib, bool) or not 0 <= max_driver_memory_mib <= 256:
        raise GPU5BoundaryError("idle driver-memory bound must be in [0, 256] MiB")
    if snapshot.compute_processes:
        raise GPU5BoundaryError("GPU5 has a foreign compute PID")
    if snapshot.utilization_percent != 0:
        raise GPU5BoundaryError("GPU5 is not idle: utilization is non-zero")
    if snapshot.memory_used_mib > max_driver_memory_mib:
        raise GPU5BoundaryError(
            "GPU5 is not idle: memory exceeds driver-only allowance"
        )


def preflight_gpu5(*, runner: Any = subprocess.run) -> GPU5Snapshot:
    snapshot = query_gpu5_snapshot(runner=runner)
    assert_gpu5_idle(snapshot)
    return snapshot


def _run_git(
    root: Path,
    arguments: Sequence[str],
    *,
    runner: Any = subprocess.run,
    binary: bool = False,
) -> bytes | str:
    executable = _validated_executable(GIT_EXECUTABLE)
    argv = (
        executable,
        "-c",
        "core.fsmonitor=false",
        "-c",
        "core.untrackedCache=false",
        "-C",
        str(root),
        *arguments,
    )
    try:
        completed = runner(
            list(argv),
            capture_output=True,
            text=not binary,
            timeout=30.0,
            check=False,
            env=_minimal_host_environment(),
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise GPU5BoundaryError(
            f"bounded source-scope query failed: {type(error).__name__}"
        ) from error
    stdout = completed.stdout
    stderr = completed.stderr
    expected_type = bytes if binary else str
    if not isinstance(stdout, expected_type) or not isinstance(stderr, expected_type):
        raise GPU5BoundaryError("source-scope query returned an invalid stream type")
    if len(stdout) > MAX_GIT_OUTPUT_BYTES or len(stderr) > MAX_GIT_OUTPUT_BYTES:
        raise GPU5BoundaryError("source-scope query exceeded its output bound")
    if completed.returncode != 0 or stderr.strip():
        raise GPU5BoundaryError("source-scope query failed closed")
    return stdout


def _hash_regular_file(
    path: Path, *, expected_root: Path
) -> tuple[str, tuple[int, ...]]:
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise GPU5BoundaryError(f"scoped file could not be opened: {path}") from error
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or not _trusted_owner_uid(before.st_uid)
            or _group_or_world_writable(before.st_mode)
            or before.st_nlink != 1
        ):
            raise GPU5BoundaryError(f"scoped file metadata is unsafe: {path}")
        resolved = path.resolve(strict=True)
        if not resolved.is_relative_to(expected_root):
            raise GPU5BoundaryError(f"scoped file escaped its exact root: {path}")
        digest = hashlib.sha256()
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
        after = os.fstat(descriptor)
        identity = (
            int(after.st_dev),
            int(after.st_ino),
            int(after.st_uid),
            int(after.st_nlink),
            int(after.st_size),
            int(after.st_mtime_ns),
            int(after.st_mode),
        )
        before_identity = (
            int(before.st_dev),
            int(before.st_ino),
            int(before.st_uid),
            int(before.st_nlink),
            int(before.st_size),
            int(before.st_mtime_ns),
            int(before.st_mode),
        )
        if identity != before_identity:
            raise GPU5BoundaryError(f"scoped file changed while hashing: {path}")
        return digest.hexdigest(), identity
    finally:
        os.close(descriptor)


def _strict_model_manifest_entries(
    manifest_path: Path,
) -> tuple[tuple[tuple[str, str], ...], ArtifactIdentity | None]:
    """Parse the pinned minimal ``[files]`` grammar without importing TOML code."""

    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(manifest_path, flags)
    except OSError as error:
        raise ArtifactVerificationError("model manifest could not be opened") from error
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or not _trusted_owner_uid(before.st_uid)
            or _group_or_world_writable(before.st_mode)
            or before.st_nlink != 1
            or before.st_size <= 0
            or before.st_size > MAX_MODEL_MANIFEST_BYTES
        ):
            raise ArtifactVerificationError("model manifest metadata is unsafe")
        chunks: list[bytes] = []
        observed_bytes = 0
        while True:
            chunk = os.read(descriptor, min(16_384, MAX_MODEL_MANIFEST_BYTES + 1))
            if not chunk:
                break
            observed_bytes += len(chunk)
            if observed_bytes > MAX_MODEL_MANIFEST_BYTES:
                raise ArtifactVerificationError("model manifest exceeds its byte bound")
            chunks.append(chunk)
        after = os.fstat(descriptor)
        before_identity = (
            int(before.st_dev),
            int(before.st_ino),
            int(before.st_uid),
            int(before.st_nlink),
            int(before.st_size),
            int(before.st_mtime_ns),
            int(before.st_mode),
        )
        after_identity = (
            int(after.st_dev),
            int(after.st_ino),
            int(after.st_uid),
            int(after.st_nlink),
            int(after.st_size),
            int(after.st_mtime_ns),
            int(after.st_mode),
        )
        if before_identity != after_identity or observed_bytes != before.st_size:
            raise ArtifactVerificationError("model manifest changed while reading")
    finally:
        os.close(descriptor)

    try:
        text = b"".join(chunks).decode("utf-8")
    except UnicodeDecodeError as error:
        raise ArtifactVerificationError(
            "model manifest must be strict UTF-8"
        ) from error
    if any(ord(character) < 32 and character != "\n" for character in text):
        raise ArtifactVerificationError("model manifest contains control characters")
    lines = text.split("\n")
    if len(lines) > MAX_MODEL_MANIFEST_LINES:
        raise ArtifactVerificationError("model manifest exceeds its line bound")

    active_table: str | None = None
    model_table_seen = False
    files_table_seen = False
    model_values: dict[str, str] = {}
    entries: dict[str, str] = {}
    for line_number, line in enumerate(lines, start=1):
        if not line:
            continue
        if line == "[model]":
            if model_table_seen:
                raise ArtifactVerificationError(
                    "model manifest [model] table is duplicated"
                )
            active_table = "model"
            model_table_seen = True
            continue
        if line == "[files]":
            if files_table_seen or active_table == "files":
                raise ArtifactVerificationError(
                    "model manifest [files] table is duplicated"
                )
            if model_table_seen and set(model_values) != _MODEL_IDENTITY_KEYS:
                raise ArtifactVerificationError(
                    "model manifest [model] requires exactly five identity keys"
                )
            active_table = "files"
            files_table_seen = True
            continue
        if line.startswith("[") or active_table is None:
            raise ArtifactVerificationError(
                f"model manifest line {line_number} contains an unsupported table or key"
            )
        if active_table == "model":
            identity_match = _STRICT_MODEL_IDENTITY_ENTRY.fullmatch(line)
            if identity_match is None:
                raise ArtifactVerificationError(
                    f"model identity line {line_number} is outside the pinned grammar"
                )
            key = identity_match.group("key")
            value = identity_match.group("value")
            if key in model_values:
                raise ArtifactVerificationError(
                    f"model manifest identity key is duplicated: {key}"
                )
            if any(ord(character) < 32 or ord(character) == 127 for character in value):
                raise ArtifactVerificationError(
                    f"model manifest identity value is unsafe: {key}"
                )
            model_values[key] = value
            continue
        match = _STRICT_MANIFEST_ENTRY.fullmatch(line)
        if match is None:
            raise ArtifactVerificationError(
                f"model manifest line {line_number} is outside the pinned grammar"
            )
        relative_name = match.group("path")
        if len(relative_name) > MAX_MODEL_MANIFEST_PATH_CHARS:
            raise ArtifactVerificationError("model manifest path exceeds its bound")
        parts = relative_name.split("/")
        if any(not part or part in {".", ".."} for part in parts):
            raise ArtifactVerificationError(
                f"model manifest contains an unsafe path: {relative_name}"
            )
        if relative_name in entries:
            raise ArtifactVerificationError(
                f"model manifest contains a duplicate path: {relative_name}"
            )
        entries[relative_name] = match.group("digest")
        if len(entries) > MAX_MODEL_MANIFEST_ENTRIES:
            raise ArtifactVerificationError("model manifest exceeds its entry bound")
    if not files_table_seen or not entries:
        raise ArtifactVerificationError(
            "model manifest must contain one non-empty [files] table"
        )
    if model_table_seen and set(model_values) != _MODEL_IDENTITY_KEYS:
        raise ArtifactVerificationError(
            "model manifest [model] requires exactly five identity keys"
        )

    names = set(entries)
    for relative_name in names:
        parts = relative_name.split("/")
        for boundary in range(1, len(parts)):
            if "/".join(parts[:boundary]) in names:
                raise ArtifactVerificationError(
                    "model manifest path is both a file and a directory"
                )
    identity = ArtifactIdentity(**model_values) if model_table_seen else None
    if identity is not None and identity.role not in {"base", "instruction_tuned"}:
        raise ArtifactVerificationError("model manifest role is unsupported")
    return tuple(sorted(entries.items())), identity


def verify_artifact_manifest(
    root: str | Path, manifest: str | Path
) -> VerifiedArtifactSet:
    """Verify the pinned model with a Python-3.10-compatible strict manifest."""

    root_input = Path(root).expanduser()
    manifest_input = Path(manifest).expanduser()
    try:
        root_path = _trusted_directory_chain(root_input, label="model root")
        manifest_parent = _trusted_directory_chain(
            manifest_input.parent, label="model manifest parent"
        )
        root_lstat = root_input.lstat()
        manifest_lstat = manifest_input.lstat()
        manifest_path = manifest_input.resolve(strict=True)
    except (GPU5BoundaryError, OSError) as error:
        raise ArtifactVerificationError(
            "model root or manifest path hierarchy is unsafe"
        ) from error
    if manifest_parent != manifest_path.parent:
        raise ArtifactVerificationError("model manifest parent identity is unsafe")
    if (
        stat.S_ISLNK(root_lstat.st_mode)
        or not stat.S_ISDIR(root_lstat.st_mode)
        or not _trusted_owner_uid(root_lstat.st_uid)
        or _group_or_world_writable(root_lstat.st_mode)
    ):
        raise ArtifactVerificationError("model root metadata is unsafe")
    if (
        stat.S_ISLNK(manifest_lstat.st_mode)
        or not stat.S_ISREG(manifest_lstat.st_mode)
        or not _trusted_owner_uid(manifest_lstat.st_uid)
        or _group_or_world_writable(manifest_lstat.st_mode)
        or manifest_lstat.st_nlink != 1
        or manifest_path != manifest_input
    ):
        raise ArtifactVerificationError("model manifest must be a regular file")
    root_identity = (
        int(root_lstat.st_dev),
        int(root_lstat.st_ino),
        int(root_lstat.st_uid),
        int(root_lstat.st_mode),
    )
    entries, identity = _strict_model_manifest_entries(manifest_path)
    verified_files: list[Path] = []
    verified_digests: list[tuple[str, str]] = []
    for relative_name, expected_digest in entries:
        parts = relative_name.split("/")
        parent = root_path
        for part in parts[:-1]:
            parent /= part
            try:
                parent_metadata = parent.lstat()
            except OSError as error:
                raise ArtifactVerificationError(
                    f"model artifact parent is missing: {relative_name}"
                ) from error
            if (
                stat.S_ISLNK(parent_metadata.st_mode)
                or not stat.S_ISDIR(parent_metadata.st_mode)
                or not _trusted_owner_uid(parent_metadata.st_uid)
                or _group_or_world_writable(parent_metadata.st_mode)
            ):
                raise ArtifactVerificationError(
                    f"model artifact parent metadata is unsafe: {relative_name}"
                )
        candidate = root_path.joinpath(*parts)
        try:
            actual_digest, _identity = _hash_regular_file(
                candidate, expected_root=root_path
            )
        except GPU5BoundaryError as error:
            raise ArtifactVerificationError(
                f"model artifact metadata is unsafe: {relative_name}"
            ) from error
        if actual_digest != expected_digest:
            raise ArtifactVerificationError(
                f"model artifact digest mismatch: {relative_name}"
            )
        verified_files.append(candidate)
        verified_digests.append((relative_name, actual_digest))
    root_after = root_path.lstat()
    after_identity = (
        int(root_after.st_dev),
        int(root_after.st_ino),
        int(root_after.st_uid),
        int(root_after.st_mode),
    )
    if root_identity != after_identity:
        raise ArtifactVerificationError("model root changed during verification")
    try:
        _trusted_directory_chain(root_path, label="model root postcheck")
    except GPU5BoundaryError as error:
        raise ArtifactVerificationError(
            "model root hierarchy changed during verification"
        ) from error
    return VerifiedArtifactSet(
        root=root_path,
        files=tuple(verified_files),
        identity=identity,
        digests=tuple(verified_digests),
    )


def verify_closed_world_artifact_layout(
    verified: VerifiedArtifactSet,
    *,
    allowed_unmanifested_files: tuple[str, ...] = (),
    allowed_unmanifested_directories: tuple[str, ...] = (),
) -> VerifiedArtifactSet:
    """Require the model root to equal the manifest recursively, with no extras."""

    if not isinstance(verified, VerifiedArtifactSet):
        raise TypeError("verified must be a VerifiedArtifactSet")
    if allowed_unmanifested_files or allowed_unmanifested_directories:
        raise ValueError("the GPU5 host guard does not permit unmanifested entries")
    if not verified.digests:
        raise ArtifactVerificationError("closed-world verification needs digests")
    expected_files = {relative for relative, _digest in verified.digests}
    if len(expected_files) != len(verified.digests):
        raise ArtifactVerificationError("closed-world digest paths are duplicated")
    expected_directories = {"."}
    for relative_name in expected_files:
        parts = relative_name.split("/")
        for boundary in range(1, len(parts)):
            expected_directories.add("/".join(parts[:boundary]))

    root = verified.root
    try:
        trusted_root = _trusted_directory_chain(root, label="closed-world model root")
        root_before = root.lstat()
        verified_names = {path.relative_to(root).as_posix() for path in verified.files}
    except (GPU5BoundaryError, OSError, ValueError) as error:
        raise ArtifactVerificationError(
            "closed-world model root or verified files are invalid"
        ) from error
    if (
        trusted_root != root
        or stat.S_ISLNK(root_before.st_mode)
        or not stat.S_ISDIR(root_before.st_mode)
        or not _trusted_owner_uid(root_before.st_uid)
        or _group_or_world_writable(root_before.st_mode)
        or verified_names != expected_files
        or len(verified.files) != len(expected_files)
    ):
        raise ArtifactVerificationError("closed-world verified set is inconsistent")
    root_identity = (
        int(root_before.st_dev),
        int(root_before.st_ino),
        int(root_before.st_uid),
        int(root_before.st_mode),
    )

    observed_files: set[str] = set()
    observed_directories: set[str] = set()

    def _walk_error(error: OSError) -> None:
        raise ArtifactVerificationError(
            "closed-world model inventory failed"
        ) from error

    for directory, child_directories, filenames in os.walk(
        root, followlinks=False, onerror=_walk_error
    ):
        directory_path = Path(directory)
        relative_directory = directory_path.relative_to(root).as_posix()
        if relative_directory == ".":
            relative_directory = "."
        directory_metadata = directory_path.lstat()
        if (
            relative_directory not in expected_directories
            or stat.S_ISLNK(directory_metadata.st_mode)
            or not stat.S_ISDIR(directory_metadata.st_mode)
            or not _trusted_owner_uid(directory_metadata.st_uid)
            or _group_or_world_writable(directory_metadata.st_mode)
        ):
            raise ArtifactVerificationError(
                f"closed-world directory is forbidden: {relative_directory}"
            )
        observed_directories.add(relative_directory)
        for child in child_directories:
            child_path = directory_path / child
            child_metadata = child_path.lstat()
            relative_child = child_path.relative_to(root).as_posix()
            if (
                relative_child not in expected_directories
                or stat.S_ISLNK(child_metadata.st_mode)
                or not stat.S_ISDIR(child_metadata.st_mode)
                or not _trusted_owner_uid(child_metadata.st_uid)
                or _group_or_world_writable(child_metadata.st_mode)
            ):
                raise ArtifactVerificationError(
                    f"closed-world directory is forbidden: {relative_child}"
                )
        for filename in filenames:
            candidate = directory_path / filename
            metadata = candidate.lstat()
            relative_name = candidate.relative_to(root).as_posix()
            if (
                relative_name not in expected_files
                or stat.S_ISLNK(metadata.st_mode)
                or not stat.S_ISREG(metadata.st_mode)
                or not _trusted_owner_uid(metadata.st_uid)
                or _group_or_world_writable(metadata.st_mode)
                or metadata.st_nlink != 1
            ):
                raise ArtifactVerificationError(
                    f"closed-world file is forbidden: {relative_name}"
                )
            observed_files.add(relative_name)
    root_after = root.lstat()
    after_identity = (
        int(root_after.st_dev),
        int(root_after.st_ino),
        int(root_after.st_uid),
        int(root_after.st_mode),
    )
    if (
        root_identity != after_identity
        or observed_files != expected_files
        or observed_directories != expected_directories
    ):
        raise ArtifactVerificationError("closed-world model inventory changed")
    try:
        _trusted_directory_chain(root, label="closed-world model root postcheck")
    except GPU5BoundaryError as error:
        raise ArtifactVerificationError(
            "closed-world model hierarchy changed during inventory"
        ) from error
    return verified


def _canonical_scope_digest(records: Sequence[Sequence[object]]) -> str:
    encoded = json.dumps(
        [list(record) for record in records],
        ensure_ascii=True,
        separators=(",", ":"),
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def _snapshot_store_usage(
    root: Path,
    *,
    max_snapshots: int,
    max_bytes: int,
) -> tuple[int, int]:
    """Inventory every final or partial snapshot without deleting anything."""

    store = _ensure_private_directory(root, create=False)
    snapshot_count = 0
    total_bytes = 0
    try:
        entries = tuple(store.iterdir())
    except OSError as error:
        raise GPU5BoundaryError("snapshot store could not be inventoried") from error
    for entry in entries:
        try:
            metadata = entry.lstat()
        except OSError as error:
            raise GPU5BoundaryError("snapshot store entry is unavailable") from error
        if (
            stat.S_ISLNK(metadata.st_mode)
            or not stat.S_ISDIR(metadata.st_mode)
            or not _trusted_owner_uid(metadata.st_uid)
            or _group_or_world_writable(metadata.st_mode)
        ):
            raise GPU5BoundaryError("snapshot store contains an unsafe entry")
        snapshot_count += 1
        if snapshot_count > max_snapshots:
            raise GPU5BoundaryError("snapshot store count quota is exhausted")
        for directory, child_dirs, filenames in os.walk(entry, followlinks=False):
            directory_path = Path(directory)
            directory_metadata = directory_path.lstat()
            if (
                stat.S_ISLNK(directory_metadata.st_mode)
                or not stat.S_ISDIR(directory_metadata.st_mode)
                or not _trusted_owner_uid(directory_metadata.st_uid)
                or _group_or_world_writable(directory_metadata.st_mode)
            ):
                raise GPU5BoundaryError("snapshot store contains an unsafe directory")
            for child in child_dirs:
                child_metadata = (directory_path / child).lstat()
                if (
                    stat.S_ISLNK(child_metadata.st_mode)
                    or not stat.S_ISDIR(child_metadata.st_mode)
                    or not _trusted_owner_uid(child_metadata.st_uid)
                    or _group_or_world_writable(child_metadata.st_mode)
                ):
                    raise GPU5BoundaryError(
                        "snapshot store contains an unsafe child directory"
                    )
            for filename in filenames:
                file_metadata = (directory_path / filename).lstat()
                if (
                    stat.S_ISLNK(file_metadata.st_mode)
                    or not stat.S_ISREG(file_metadata.st_mode)
                    or not _trusted_owner_uid(file_metadata.st_uid)
                    or _group_or_world_writable(file_metadata.st_mode)
                    or file_metadata.st_nlink != 1
                    or file_metadata.st_size < 0
                ):
                    raise GPU5BoundaryError(
                        "snapshot store contains an unsafe regular file"
                    )
                total_bytes += int(file_metadata.st_size)
                if total_bytes > max_bytes:
                    raise GPU5BoundaryError("snapshot store byte quota is exhausted")
    return snapshot_count, total_bytes


def _enforce_snapshot_store_quota(
    root: Path,
    *,
    max_snapshots: int,
    max_bytes: int,
    reserve_snapshots: int,
    reserve_bytes: int,
) -> None:
    if (
        isinstance(reserve_snapshots, bool)
        or not isinstance(reserve_snapshots, int)
        or reserve_snapshots < 0
        or isinstance(reserve_bytes, bool)
        or not isinstance(reserve_bytes, int)
        or reserve_bytes < 0
    ):
        raise GPU5BoundaryError("snapshot quota reservation is invalid")
    observed_count, observed_bytes = _snapshot_store_usage(
        root,
        max_snapshots=max_snapshots,
        max_bytes=max_bytes,
    )
    if observed_count + reserve_snapshots > max_snapshots:
        raise GPU5BoundaryError("snapshot store count quota is exhausted")
    if observed_bytes + reserve_bytes > max_bytes:
        raise GPU5BoundaryError("snapshot store byte quota is exhausted")


def _snapshot_path(expected_source_commit: str, launch_nonce: str) -> Path:
    commit = _validated_expected_source_commit(expected_source_commit)
    nonce = _launch_nonce(launch_nonce)
    return SOURCE_SNAPSHOT_ROOT / f"source-{commit}-{nonce}"


def _git_object_hasher(object_format: str, size: int) -> Any:
    if object_format not in {"sha1", "sha256"}:
        raise GPU5BoundaryError("unsupported Git object format")
    digest = hashlib.new(object_format)
    digest.update(f"blob {size}\0".encode("ascii"))
    return digest


def _safe_source_name(encoded_name: bytes) -> str:
    try:
        name = encoded_name.decode("utf-8", errors="strict")
    except UnicodeDecodeError as error:
        raise GPU5BoundaryError("source path is not canonical UTF-8") from error
    relative = Path(name)
    if (
        not name
        or "\\" in name
        or relative.is_absolute()
        or ".." in relative.parts
        or ".git" in relative.parts
        or "__pycache__" in relative.parts
        or relative.suffix.casefold() in {".pyc", ".pyo"}
        or relative.as_posix() != name
    ):
        raise GPU5BoundaryError("source path is unsafe for the closed-world snapshot")
    return name


def _commit_tree(
    repo_root: Path,
    expected_source_commit: str,
    *,
    git_runner: Any,
) -> tuple[str, dict[str, tuple[str, str]]]:
    object_format = str(
        _run_git(
            repo_root,
            ("rev-parse", "--show-object-format"),
            runner=git_runner,
        )
    ).strip()
    raw = bytes(
        _run_git(
            repo_root,
            (
                "ls-tree",
                "-r",
                "-z",
                "--full-tree",
                _validated_expected_source_commit(expected_source_commit),
            ),
            runner=git_runner,
            binary=True,
        )
    )
    entries: dict[str, tuple[str, str]] = {}
    oid_pattern = re.compile(
        rb"[0-9a-f]{40}" if object_format == "sha1" else rb"[0-9a-f]{64}"
    )
    for record in (item for item in raw.split(b"\0") if item):
        try:
            metadata, encoded_name = record.split(b"\t", 1)
            mode, kind, oid = metadata.split(b" ", 2)
        except ValueError as error:
            raise GPU5BoundaryError("Git commit tree row is malformed") from error
        if mode not in {b"100644", b"100755"} or kind != b"blob":
            raise GPU5BoundaryError(
                "source snapshot permits regular 100644/100755 blobs only"
            )
        if oid_pattern.fullmatch(oid) is None:
            raise GPU5BoundaryError("Git commit tree returned an invalid blob id")
        name = _safe_source_name(encoded_name)
        if name in entries:
            raise GPU5BoundaryError("Git commit tree contains a duplicate path")
        entries[name] = (mode.decode("ascii"), oid.decode("ascii"))
    if not entries:
        raise GPU5BoundaryError("expected source commit contains no regular files")
    return object_format, entries


def _hash_git_worktree_file(
    path: Path,
    *,
    expected_root: Path,
    object_format: str,
) -> tuple[str, str, tuple[int, ...], str]:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise GPU5BoundaryError(
            f"tracked source file could not be opened: {path}"
        ) from error
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or not _trusted_owner_uid(before.st_uid)
            or _group_or_world_writable(before.st_mode)
            or before.st_nlink != 1
        ):
            raise GPU5BoundaryError(f"tracked source metadata is unsafe: {path}")
        if not path.resolve(strict=True).is_relative_to(expected_root):
            raise GPU5BoundaryError("tracked source file escaped the checkout")
        content = hashlib.sha256()
        blob = _git_object_hasher(object_format, int(before.st_size))
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            content.update(chunk)
            blob.update(chunk)
        after = os.fstat(descriptor)
        identity = (
            int(after.st_dev),
            int(after.st_ino),
            int(after.st_uid),
            int(after.st_nlink),
            int(after.st_size),
            int(after.st_mtime_ns),
            int(after.st_mode),
        )
        before_identity = (
            int(before.st_dev),
            int(before.st_ino),
            int(before.st_uid),
            int(before.st_nlink),
            int(before.st_size),
            int(before.st_mtime_ns),
            int(before.st_mode),
        )
        if identity != before_identity:
            raise GPU5BoundaryError("tracked source changed while hashing")
        git_mode = "100755" if after.st_mode & 0o111 else "100644"
        return content.hexdigest(), blob.hexdigest(), identity, git_mode
    finally:
        os.close(descriptor)


def _verify_working_checkout(
    repo_root: Path,
    expected_source_commit: str,
    *,
    git_runner: Any,
) -> tuple[str, str, int, str, dict[str, tuple[str, str]]]:
    expected_commit = _validated_expected_source_commit(expected_source_commit)
    repo_root = _trusted_directory_chain(repo_root, label="source checkout root")
    top_level = str(
        _run_git(repo_root, ("rev-parse", "--show-toplevel"), runner=git_runner)
    ).strip()
    if Path(top_level).resolve(strict=True) != repo_root:
        raise GPU5BoundaryError("Git top-level does not match the pinned repo root")
    head = str(_run_git(repo_root, ("rev-parse", "HEAD"), runner=git_runner)).strip()
    if head != expected_commit:
        raise GPU5BoundaryError("source HEAD does not match the expected commit")
    dirty = bytes(
        _run_git(
            repo_root,
            ("status", "--porcelain=v1", "-z", "--untracked-files=normal"),
            runner=git_runner,
            binary=True,
        )
    )
    if dirty:
        raise GPU5BoundaryError("GPU evidence requires a clean source checkout")
    object_format, entries = _commit_tree(
        repo_root, expected_commit, git_runner=git_runner
    )
    tagged = bytes(
        _run_git(
            repo_root,
            ("ls-files", "-v", "-z", "--cached"),
            runner=git_runner,
            binary=True,
        )
    )
    observed_names: list[str] = []
    for record in (item for item in tagged.split(b"\0") if item):
        if len(record) < 3 or record[:2] != b"H ":
            raise GPU5BoundaryError(
                "assume-unchanged, skip-worktree, or non-normal index state is forbidden"
            )
        observed_names.append(_safe_source_name(record[2:]))
    if set(observed_names) != set(entries) or len(observed_names) != len(entries):
        raise GPU5BoundaryError("working index does not equal the expected commit tree")
    content_records: list[tuple[object, ...]] = []
    identity_records: list[tuple[object, ...]] = []
    trusted_parents: set[Path] = {repo_root}
    for name in sorted(entries):
        expected_mode, expected_oid = entries[name]
        tracked_parent = (repo_root / name).parent
        if tracked_parent not in trusted_parents:
            _trusted_directory_chain(
                tracked_parent,
                label=f"tracked source parent for {name}",
            )
            trusted_parents.add(tracked_parent)
        sha256, oid, identity, git_mode = _hash_git_worktree_file(
            repo_root / name,
            expected_root=repo_root,
            object_format=object_format,
        )
        if oid != expected_oid or git_mode != expected_mode:
            raise GPU5BoundaryError(
                f"working source does not match expected commit blob: {name}"
            )
        content_records.append((name, sha256, expected_mode, expected_oid))
        identity_records.append((name, *identity, sha256))
    for parent in sorted(trusted_parents):
        _trusted_directory_chain(
            parent,
            label="tracked source parent postcheck",
        )
    return (
        _canonical_scope_digest(content_records),
        _canonical_scope_digest(identity_records),
        len(entries),
        object_format,
        entries,
    )


def verify_exact_source_checkout(
    expected_source_commit: str,
    *,
    git_runner: Any = subprocess.run,
) -> WorkingCheckoutIdentity:
    """Bind the pinned worktree to one clean, normal-index source commit."""

    commit = _validated_expected_source_commit(expected_source_commit)
    repo_root = next(
        source
        for source, destination in _EXPECTED_READ_ONLY_MOUNTS.items()
        if destination == Path("/workspace")
    )
    content, identity, count, _object_format, _entries = _verify_working_checkout(
        repo_root,
        commit,
        git_runner=git_runner,
    )
    return WorkingCheckoutIdentity(
        source_commit=commit,
        content_digest=content,
        identity_digest=identity,
        file_count=count,
    )


@contextmanager
def native_gpu5_server_authority(
    expected_source_commit: str,
    *,
    physical_gpu_index: object,
    gpu_query_context: object,
    gpu_uuid: object,
    source_snapshot_root: str,
    source_snapshot_nonce: str,
    model_snapshot_root: str,
    model_manifest_path: str,
    model_manifest_sha256: str,
    workspace_root: str,
    source_content_digest: str,
    source_identity_digest: str,
    source_file_count: int,
    source_root_device: int,
    source_root_inode: int,
    model_content_digest: str,
    model_identity_digest: str,
    model_file_count: int,
    model_root_device: int,
    model_root_inode: int,
    model_total_bytes: int,
    smi_runner: Any = subprocess.run,
) -> Iterator[NativeGPU5ServerAuthority]:
    """Acquire the shared flock and prove idle GPU5 before product imports."""

    commit = _validated_expected_source_commit(expected_source_commit)
    selected_index = require_project_gpu_index(physical_gpu_index)
    if gpu_query_context != "native-host":
        raise GPU5BoundaryError("native server requires native-host query context")
    if gpu_uuid != PROJECT_GPU_UUID:
        raise GPU5BoundaryError("native server GPU UUID is not the pinned GPU5")
    if os.environ.get("CUDA_DEVICE_ORDER") != "PCI_BUS_ID":
        raise GPU5BoundaryError(
            "native server requires CUDA_DEVICE_ORDER=PCI_BUS_ID before startup"
        )
    if os.environ.get("CUDA_VISIBLE_DEVICES") != PROJECT_GPU_UUID:
        raise GPU5BoundaryError(
            "native server requires the pinned CUDA_VISIBLE_DEVICES UUID"
        )
    if os.environ.get("NVIDIA_VISIBLE_DEVICES") != PROJECT_GPU_UUID:
        raise GPU5BoundaryError(
            "native server requires the pinned NVIDIA_VISIBLE_DEVICES UUID"
        )
    expected_handoff = _validated_native_snapshot_handoff(
        source_content_digest=source_content_digest,
        source_identity_digest=source_identity_digest,
        source_file_count=source_file_count,
        source_root_device=source_root_device,
        source_root_inode=source_root_inode,
        model_content_digest=model_content_digest,
        model_identity_digest=model_identity_digest,
        model_file_count=model_file_count,
        model_root_device=model_root_device,
        model_root_inode=model_root_inode,
        model_total_bytes=model_total_bytes,
    )

    with gpu5_project_lease(commit) as lease:
        execution_snapshot = _capture_native_execution_snapshot(
            commit,
            source_snapshot_root=source_snapshot_root,
            source_snapshot_nonce=source_snapshot_nonce,
            model_snapshot_root=model_snapshot_root,
            model_manifest_path=model_manifest_path,
            model_manifest_sha256=model_manifest_sha256,
        )
        if _native_snapshot_handoff(execution_snapshot) != expected_handoff:
            raise GPU5BoundaryError(
                "native snapshot handoff does not match sealed execution snapshot"
            )
        if workspace_root != execution_snapshot.workspace_root:
            raise GPU5BoundaryError(
                "native workspace handoff does not match sealed execution snapshot"
            )
        checkout = WorkingCheckoutIdentity(
            source_commit=commit,
            content_digest=execution_snapshot.source.content_digest,
            identity_digest=execution_snapshot.source.identity_digest,
            file_count=execution_snapshot.source.file_count,
        )
        snapshot = preflight_gpu5(runner=smi_runner)
        authority = NativeGPU5ServerAuthority(
            expected_source_commit=commit,
            physical_gpu_index=selected_index,
            gpu_query_context="native-host",
            gpu_uuid=PROJECT_GPU_UUID,
            preflight=snapshot,
            checkout=checkout,
            execution_snapshot=execution_snapshot,
            _lease=lease,
        )
        yield authority


def _bound_source_archive_size() -> None:
    if resource is None:
        raise RuntimeError("RLIMIT_FSIZE is unavailable")
    resource.setrlimit(
        resource.RLIMIT_FSIZE,
        (MAX_SOURCE_ARCHIVE_BYTES, MAX_SOURCE_ARCHIVE_BYTES),
    )


def _write_all(descriptor: int, data: bytes) -> None:
    offset = 0
    while offset < len(data):
        written = os.write(descriptor, data[offset:])
        if written <= 0:
            raise GPU5BoundaryError("snapshot file write made no progress")
        offset += written


def _snapshot_inventory(
    snapshot_root: Path,
    *,
    source_commit: str,
    launch_nonce: str,
    object_format: str,
    expected_entries: Mapping[str, tuple[str, str]],
) -> SourceSnapshot:
    root_metadata = snapshot_root.stat()
    if (
        not stat.S_ISDIR(root_metadata.st_mode)
        or root_metadata.st_uid != os.geteuid()
        or stat.S_IMODE(root_metadata.st_mode) != 0o555
        or snapshot_root.resolve(strict=True) != snapshot_root
    ):
        raise GPU5BoundaryError("source snapshot root metadata is unsafe")
    observed: set[str] = set()
    content_records: list[tuple[object, ...]] = []
    identity_records: list[tuple[object, ...]] = [
        (
            "directory",
            ".",
            int(root_metadata.st_dev),
            int(root_metadata.st_ino),
            int(root_metadata.st_uid),
            stat.S_IMODE(root_metadata.st_mode),
        )
    ]
    for directory, child_dirs, filenames in os.walk(snapshot_root, followlinks=False):
        directory_path = Path(directory)
        if directory_path.is_symlink():
            raise GPU5BoundaryError("source snapshot contains a symlink directory")
        for child in child_dirs:
            child_path = directory_path / child
            metadata = child_path.lstat()
            if (
                not stat.S_ISDIR(metadata.st_mode)
                or metadata.st_uid != os.geteuid()
                or stat.S_IMODE(metadata.st_mode) != 0o555
            ):
                raise GPU5BoundaryError(
                    "source snapshot contains a non-directory parent"
                )
            identity_records.append(
                (
                    "directory",
                    child_path.relative_to(snapshot_root).as_posix(),
                    int(metadata.st_dev),
                    int(metadata.st_ino),
                    int(metadata.st_uid),
                    stat.S_IMODE(metadata.st_mode),
                )
            )
        for filename in filenames:
            path = directory_path / filename
            relative = path.relative_to(snapshot_root).as_posix()
            _safe_source_name(relative.encode("utf-8"))
            if relative not in expected_entries or relative in observed:
                raise GPU5BoundaryError(
                    "source snapshot closed-world inventory mismatch"
                )
            observed.add(relative)
            expected_mode, expected_oid = expected_entries[relative]
            sha256, oid, identity, git_mode = _hash_git_worktree_file(
                path,
                expected_root=snapshot_root,
                object_format=object_format,
            )
            if oid != expected_oid or git_mode != expected_mode:
                raise GPU5BoundaryError("source snapshot blob or mode mismatch")
            sealed_mode = 0o555 if expected_mode == "100755" else 0o444
            if stat.S_IMODE(identity[-1]) != sealed_mode:
                raise GPU5BoundaryError("source snapshot file is not sealed read-only")
            content_records.append((relative, sha256, expected_mode, expected_oid))
            identity_records.append(("file", relative, *identity, sha256))
    if observed != set(expected_entries):
        raise GPU5BoundaryError("source snapshot is incomplete")
    root_after = snapshot_root.stat()
    if (
        int(root_after.st_dev),
        int(root_after.st_ino),
        int(root_after.st_uid),
        stat.S_IMODE(root_after.st_mode),
    ) != (
        int(root_metadata.st_dev),
        int(root_metadata.st_ino),
        int(root_metadata.st_uid),
        stat.S_IMODE(root_metadata.st_mode),
    ):
        raise GPU5BoundaryError("source snapshot root changed during inventory")
    return SourceSnapshot(
        source_commit=source_commit,
        launch_nonce=launch_nonce,
        root_path=str(snapshot_root),
        root_device=int(root_metadata.st_dev),
        root_inode=int(root_metadata.st_ino),
        root_mode=stat.S_IMODE(root_metadata.st_mode),
        file_count=len(observed),
        content_digest=_canonical_scope_digest(sorted(content_records)),
        identity_digest=_canonical_scope_digest(sorted(identity_records)),
    )


def _seal_snapshot_directories(snapshot_root: Path) -> None:
    """Fsync and seal every snapshot directory bottom-up as mode 0555."""

    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    for directory, child_dirs, _filenames in os.walk(
        snapshot_root,
        topdown=False,
        followlinks=False,
    ):
        directory_path = Path(directory)
        for child in child_dirs:
            metadata = (directory_path / child).lstat()
            if not stat.S_ISDIR(metadata.st_mode) or metadata.st_uid != os.geteuid():
                raise GPU5BoundaryError(
                    "snapshot directory sealing found unsafe metadata"
                )
        descriptor = os.open(directory_path, flags)
        try:
            metadata = os.fstat(descriptor)
            if not stat.S_ISDIR(metadata.st_mode) or metadata.st_uid != os.geteuid():
                raise GPU5BoundaryError("snapshot directory sealing lost ownership")
            os.fsync(descriptor)
            os.fchmod(descriptor, 0o555)
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


def prepare_source_snapshot(
    expected_source_commit: str,
    launch_nonce: str,
    *,
    git_runner: Any = subprocess.run,
) -> SourceSnapshot:
    """Create a unique, closed-world snapshot of one exact Git commit."""

    _prepare_guard_state()
    _enforce_snapshot_store_quota(
        SOURCE_SNAPSHOT_ROOT,
        max_snapshots=MAX_SOURCE_SNAPSHOTS,
        max_bytes=MAX_SOURCE_SNAPSHOT_STORE_BYTES,
        reserve_snapshots=1,
        reserve_bytes=2 * MAX_SOURCE_ARCHIVE_BYTES,
    )
    commit = _validated_expected_source_commit(expected_source_commit)
    nonce = _launch_nonce(launch_nonce)
    repo_root = next(
        source
        for source, destination in _EXPECTED_READ_ONLY_MOUNTS.items()
        if destination == Path("/workspace")
    ).resolve(strict=True)
    _, _, _, object_format, entries = _verify_working_checkout(
        repo_root, commit, git_runner=git_runner
    )
    snapshot_root = _snapshot_path(commit, nonce)
    try:
        os.mkdir(snapshot_root, 0o700)
    except OSError as error:
        raise GPU5BoundaryError(
            "unique source snapshot already exists or could not be created"
        ) from error
    archive_path = snapshot_root / ".git-archive.partial"
    archive_flags = os.O_RDWR | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    try:
        archive_fd = os.open(archive_path, archive_flags, 0o600)
    except OSError as error:
        raise GPU5BoundaryError(
            "bounded source archive could not be created"
        ) from error
    try:
        executable = _validated_executable(GIT_EXECUTABLE)
        argv = (
            executable,
            "-c",
            "core.fsmonitor=false",
            "-c",
            "core.untrackedCache=false",
            "-C",
            str(repo_root),
            "archive",
            "--format=tar",
            commit,
        )
        with os.fdopen(os.dup(archive_fd), "wb", buffering=0) as archive_stream:
            try:
                completed = git_runner(
                    list(argv),
                    stdout=archive_stream,
                    stderr=subprocess.PIPE,
                    text=False,
                    timeout=60.0,
                    check=False,
                    preexec_fn=_bound_source_archive_size,
                    env=_minimal_host_environment(),
                )
            except (OSError, subprocess.TimeoutExpired) as error:
                raise GPU5BoundaryError(
                    f"bounded Git archive failed: {type(error).__name__}"
                ) from error
            stderr = completed.stderr
            if (
                completed.returncode != 0
                or not isinstance(stderr, bytes)
                or stderr.strip()
                or len(stderr) > 65_536
            ):
                raise GPU5BoundaryError("bounded Git archive command failed closed")
            archive_stream.flush()
            os.fsync(archive_stream.fileno())
        metadata = os.fstat(archive_fd)
        if not 0 < metadata.st_size <= MAX_SOURCE_ARCHIVE_BYTES:
            raise GPU5BoundaryError("source archive size is outside the fixed bound")
        os.lseek(archive_fd, 0, os.SEEK_SET)
        allowed_directories = {
            parent.as_posix()
            for name in entries
            for parent in Path(name).parents
            if parent != Path(".")
        }
        seen_files: set[str] = set()
        seen_directories: set[str] = set()
        with os.fdopen(os.dup(archive_fd), "rb") as archive_file:
            try:
                archive = tarfile.open(fileobj=archive_file, mode="r:")
            except tarfile.TarError as error:
                raise GPU5BoundaryError(
                    "Git archive is not a valid bounded tar"
                ) from error
            with archive:
                for member in archive:
                    try:
                        encoded_name = member.name.encode("utf-8", errors="strict")
                    except UnicodeEncodeError as error:
                        raise GPU5BoundaryError(
                            "tar source path is not canonical UTF-8"
                        ) from error
                    name = _safe_source_name(encoded_name)
                    if member.isdir():
                        if (
                            name not in allowed_directories
                            or name in seen_directories
                            or member.mode & ~0o777
                            or member.mode & 0o111 != 0o111
                        ):
                            raise GPU5BoundaryError(
                                "tar contains an unexpected directory"
                            )
                        seen_directories.add(name)
                        (snapshot_root / name).mkdir(
                            parents=True, exist_ok=True, mode=0o700
                        )
                        continue
                    if not member.isreg() or member.linkname:
                        raise GPU5BoundaryError(
                            "tar symlink, hardlink, device, FIFO, or special member rejected"
                        )
                    if name not in entries or name in seen_files:
                        raise GPU5BoundaryError(
                            "tar contains an unexpected or duplicate file"
                        )
                    seen_files.add(name)
                    expected_mode, expected_oid = entries[name]
                    expected_executable = expected_mode == "100755"
                    if (
                        member.mode & ~0o777
                        or bool(member.mode & 0o111) != expected_executable
                    ):
                        raise GPU5BoundaryError(
                            "tar member mode differs from commit tree"
                        )
                    source = archive.extractfile(member)
                    if source is None:
                        raise GPU5BoundaryError("tar regular member could not be read")
                    data = source.read(MAX_SOURCE_ARCHIVE_BYTES + 1)
                    if len(data) != member.size or len(data) > MAX_SOURCE_ARCHIVE_BYTES:
                        raise GPU5BoundaryError("tar member size is invalid")
                    blob = _git_object_hasher(object_format, len(data))
                    blob.update(data)
                    if blob.hexdigest() != expected_oid:
                        raise GPU5BoundaryError(
                            "tar member blob differs from commit tree"
                        )
                    destination = snapshot_root / name
                    destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
                    flags = (
                        os.O_WRONLY
                        | os.O_CREAT
                        | os.O_EXCL
                        | getattr(os, "O_NOFOLLOW", 0)
                    )
                    descriptor = os.open(destination, flags, 0o600)
                    try:
                        _write_all(descriptor, data)
                        os.fsync(descriptor)
                        os.fchmod(descriptor, 0o555 if expected_executable else 0o444)
                        os.fsync(descriptor)
                    finally:
                        os.close(descriptor)
        if seen_files != set(entries):
            raise GPU5BoundaryError("tar archive is incomplete")
        os.unlink(archive_path)
        _seal_snapshot_directories(snapshot_root)
        return _snapshot_inventory(
            snapshot_root,
            source_commit=commit,
            launch_nonce=nonce,
            object_format=object_format,
            expected_entries=entries,
        )
    except BaseException:
        # Never reuse a partial snapshot. Its unique nonce-named directory is
        # retained for operator review and cannot satisfy a future launch.
        raise
    finally:
        os.close(archive_fd)


def _source_snapshot_inventory_from_sealed_root(
    root: Path,
    *,
    source_commit: str,
    launch_nonce: str,
) -> SourceSnapshot:
    """Re-inventory a sealed source snapshot without consulting mutable Git."""

    commit = _validated_expected_source_commit(source_commit)
    nonce = _launch_nonce(launch_nonce)
    expected_root = _snapshot_path(commit, nonce)
    trusted_root = _trusted_directory_chain(root, label="source snapshot root")
    if trusted_root != expected_root:
        raise GPU5BoundaryError("source snapshot root escaped its nonce capability")
    root_metadata = trusted_root.lstat()
    if (
        not stat.S_ISDIR(root_metadata.st_mode)
        or not _trusted_owner_uid(root_metadata.st_uid)
        or stat.S_IMODE(root_metadata.st_mode) != 0o555
    ):
        raise GPU5BoundaryError("source snapshot root is not sealed")
    root_identity = (
        int(root_metadata.st_dev),
        int(root_metadata.st_ino),
        int(root_metadata.st_uid),
        stat.S_IMODE(root_metadata.st_mode),
    )
    content_records: list[tuple[object, ...]] = []
    identity_records: list[tuple[object, ...]] = [("directory", ".", *root_identity)]
    observed: set[str] = set()
    total_bytes = 0

    def _walk_error(error: OSError) -> None:
        raise GPU5BoundaryError("source snapshot inventory failed") from error

    for directory, child_dirs, filenames in os.walk(
        trusted_root, followlinks=False, onerror=_walk_error
    ):
        directory_path = Path(directory)
        directory_metadata = directory_path.lstat()
        if (
            stat.S_ISLNK(directory_metadata.st_mode)
            or not stat.S_ISDIR(directory_metadata.st_mode)
            or not _trusted_owner_uid(directory_metadata.st_uid)
            or stat.S_IMODE(directory_metadata.st_mode) != 0o555
        ):
            raise GPU5BoundaryError("source snapshot directory is not sealed")
        for child in child_dirs:
            child_path = directory_path / child
            child_metadata = child_path.lstat()
            if (
                stat.S_ISLNK(child_metadata.st_mode)
                or not stat.S_ISDIR(child_metadata.st_mode)
                or not _trusted_owner_uid(child_metadata.st_uid)
                or stat.S_IMODE(child_metadata.st_mode) != 0o555
            ):
                raise GPU5BoundaryError("source snapshot contains an unsafe directory")
            identity_records.append(
                (
                    "directory",
                    child_path.relative_to(trusted_root).as_posix(),
                    int(child_metadata.st_dev),
                    int(child_metadata.st_ino),
                    int(child_metadata.st_uid),
                    stat.S_IMODE(child_metadata.st_mode),
                )
            )
        for filename in filenames:
            path = directory_path / filename
            relative = path.relative_to(trusted_root).as_posix()
            _safe_source_name(relative.encode("utf-8"))
            if relative in observed:
                raise GPU5BoundaryError("source snapshot contains a duplicate file")
            observed.add(relative)
            if len(observed) > MAX_SOURCE_SNAPSHOT_FILES:
                raise GPU5BoundaryError("source snapshot exceeds its file bound")
            metadata = path.lstat()
            sealed_mode = stat.S_IMODE(metadata.st_mode)
            if (
                stat.S_ISLNK(metadata.st_mode)
                or not stat.S_ISREG(metadata.st_mode)
                or not _trusted_owner_uid(metadata.st_uid)
                or metadata.st_nlink != 1
                or sealed_mode not in {0o444, 0o555}
                or metadata.st_size < 0
            ):
                raise GPU5BoundaryError("source snapshot contains an unsafe file")
            total_bytes += int(metadata.st_size)
            if total_bytes > MAX_SOURCE_SNAPSHOT_BYTES:
                raise GPU5BoundaryError("source snapshot exceeds its byte bound")
            sha256, oid, identity, git_mode = _hash_git_worktree_file(
                path,
                expected_root=trusted_root,
                object_format="sha1",
            )
            expected_git_mode = "100755" if sealed_mode == 0o555 else "100644"
            if git_mode != expected_git_mode:
                raise GPU5BoundaryError("source snapshot executable mode changed")
            content_records.append((relative, sha256, git_mode, oid))
            identity_records.append(("file", relative, *identity, sha256))
    root_after = trusted_root.lstat()
    if (
        int(root_after.st_dev),
        int(root_after.st_ino),
        int(root_after.st_uid),
        stat.S_IMODE(root_after.st_mode),
    ) != root_identity:
        raise GPU5BoundaryError("source snapshot root changed during inventory")
    return SourceSnapshot(
        source_commit=commit,
        launch_nonce=nonce,
        root_path=str(trusted_root),
        root_device=int(root_metadata.st_dev),
        root_inode=int(root_metadata.st_ino),
        root_mode=stat.S_IMODE(root_metadata.st_mode),
        file_count=len(observed),
        content_digest=_canonical_scope_digest(sorted(content_records)),
        identity_digest=_canonical_scope_digest(sorted(identity_records)),
    )


def _model_snapshot_path(manifest_sha256: str, launch_nonce: str) -> Path:
    if (
        not isinstance(manifest_sha256, str)
        or _SHA256_DIGEST.fullmatch(manifest_sha256) is None
    ):
        raise GPU5BoundaryError("model snapshot requires a manifest SHA-256")
    nonce = _launch_nonce(launch_nonce)
    return MODEL_SNAPSHOT_ROOT / f"model-{manifest_sha256}-{nonce}"


def _descriptor_identity(metadata: os.stat_result) -> tuple[int, ...]:
    return (
        int(metadata.st_dev),
        int(metadata.st_ino),
        int(metadata.st_uid),
        int(metadata.st_nlink),
        int(metadata.st_size),
        int(metadata.st_mtime_ns),
        int(metadata.st_ctime_ns),
        int(metadata.st_mode),
    )


def _copy_model_snapshot_artifact(
    source: Path,
    destination: Path,
    *,
    source_root: Path,
    expected_digest: str,
) -> int:
    no_follow = getattr(os, "O_NOFOLLOW", None)
    if not isinstance(no_follow, int):
        raise GPU5BoundaryError("model snapshot copy requires O_NOFOLLOW")
    source_fd = -1
    destination_fd = -1
    try:
        source_fd = os.open(source, os.O_RDONLY | no_follow)
        before = os.fstat(source_fd)
        before_identity = _descriptor_identity(before)
        if (
            not stat.S_ISREG(before.st_mode)
            or not _trusted_owner_uid(before.st_uid)
            or _group_or_world_writable(before.st_mode)
            or before.st_nlink != 1
            or before.st_size < 0
            or before.st_size > MAX_MODEL_SNAPSHOT_BYTES
            or not source.resolve(strict=True).is_relative_to(source_root)
        ):
            raise GPU5BoundaryError("model snapshot source metadata is unsafe")
        destination_fd = os.open(
            destination,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | no_follow,
            0o600,
        )
        destination_before = os.fstat(destination_fd)
        if (
            not stat.S_ISREG(destination_before.st_mode)
            or destination_before.st_uid != _effective_uid()
            or destination_before.st_nlink != 1
            or (destination_before.st_dev, destination_before.st_ino)
            == (before.st_dev, before.st_ino)
        ):
            raise GPU5BoundaryError("model snapshot destination metadata is unsafe")
        digest = hashlib.sha256()
        copied = 0
        while True:
            chunk = os.read(source_fd, MODEL_SNAPSHOT_COPY_CHUNK_BYTES)
            if not chunk:
                break
            copied += len(chunk)
            if copied > before.st_size or copied > MAX_MODEL_SNAPSHOT_BYTES:
                raise GPU5BoundaryError("model snapshot copy exceeded its byte bound")
            digest.update(chunk)
            _write_all(destination_fd, chunk)
        source_after = os.fstat(source_fd)
        if (
            _descriptor_identity(source_after) != before_identity
            or copied != before.st_size
            or digest.hexdigest() != expected_digest
        ):
            raise GPU5BoundaryError("model artifact changed during snapshot copy")
        os.fsync(destination_fd)
        os.fchmod(destination_fd, 0o444)
        os.fsync(destination_fd)
        destination_after = os.fstat(destination_fd)
        if (
            not stat.S_ISREG(destination_after.st_mode)
            or destination_after.st_uid != _effective_uid()
            or destination_after.st_nlink != 1
            or destination_after.st_size != copied
            or stat.S_IMODE(destination_after.st_mode) != 0o444
            or (destination_after.st_dev, destination_after.st_ino)
            == (source_after.st_dev, source_after.st_ino)
        ):
            raise GPU5BoundaryError("model snapshot destination was not sealed")
        return copied
    except OSError as error:
        raise GPU5BoundaryError("model snapshot streaming copy failed") from error
    finally:
        if destination_fd >= 0:
            os.close(destination_fd)
        if source_fd >= 0:
            os.close(source_fd)


def _model_snapshot_inventory(
    root: Path,
    *,
    manifest_entries: Sequence[tuple[str, str]],
    manifest_sha256: str,
) -> ModelSnapshot:
    if (
        not manifest_entries
        or len(manifest_entries) > MAX_MODEL_MANIFEST_ENTRIES
        or _SHA256_DIGEST.fullmatch(manifest_sha256) is None
    ):
        raise GPU5BoundaryError("model snapshot manifest authority is invalid")
    trusted_store = _ensure_private_directory(MODEL_SNAPSHOT_ROOT, create=False)
    trusted_root = _trusted_directory_chain(root, label="model snapshot root")
    name_match = _MODEL_SNAPSHOT_DIRECTORY.fullmatch(trusted_root.name)
    if (
        trusted_root.parent != trusted_store
        or name_match is None
        or name_match.group("manifest") != manifest_sha256
    ):
        raise GPU5BoundaryError("model snapshot escaped its manifest capability")
    expected_files = dict(manifest_entries)
    if len(expected_files) != len(manifest_entries):
        raise GPU5BoundaryError("model snapshot manifest contains duplicate files")
    expected_directories = {"."}
    for relative_name in expected_files:
        parts = relative_name.split("/")
        for boundary in range(1, len(parts)):
            expected_directories.add("/".join(parts[:boundary]))

    root_metadata = trusted_root.lstat()
    if (
        not stat.S_ISDIR(root_metadata.st_mode)
        or root_metadata.st_uid != _effective_uid()
        or stat.S_IMODE(root_metadata.st_mode) != 0o555
    ):
        raise GPU5BoundaryError("model snapshot root is not sealed")
    root_identity = (
        int(root_metadata.st_dev),
        int(root_metadata.st_ino),
        int(root_metadata.st_uid),
        stat.S_IMODE(root_metadata.st_mode),
    )
    observed_files: set[str] = set()
    observed_directories: set[str] = set()
    content_records: list[tuple[object, ...]] = []
    identity_records: list[tuple[object, ...]] = [("directory", ".", *root_identity)]
    total_bytes = 0

    def _walk_error(error: OSError) -> None:
        raise GPU5BoundaryError("model snapshot inventory failed") from error

    for directory, child_dirs, filenames in os.walk(
        trusted_root, followlinks=False, onerror=_walk_error
    ):
        directory_path = Path(directory)
        relative_directory = directory_path.relative_to(trusted_root).as_posix()
        if relative_directory == ".":
            relative_directory = "."
        directory_metadata = directory_path.lstat()
        if (
            relative_directory not in expected_directories
            or stat.S_ISLNK(directory_metadata.st_mode)
            or not stat.S_ISDIR(directory_metadata.st_mode)
            or directory_metadata.st_uid != _effective_uid()
            or stat.S_IMODE(directory_metadata.st_mode) != 0o555
        ):
            raise GPU5BoundaryError("model snapshot contains an unsafe directory")
        observed_directories.add(relative_directory)
        for child in child_dirs:
            child_path = directory_path / child
            relative_child = child_path.relative_to(trusted_root).as_posix()
            child_metadata = child_path.lstat()
            if (
                relative_child not in expected_directories
                or stat.S_ISLNK(child_metadata.st_mode)
                or not stat.S_ISDIR(child_metadata.st_mode)
                or child_metadata.st_uid != _effective_uid()
                or stat.S_IMODE(child_metadata.st_mode) != 0o555
            ):
                raise GPU5BoundaryError(
                    "model snapshot contains an unsafe child directory"
                )
            identity_records.append(
                (
                    "directory",
                    relative_child,
                    int(child_metadata.st_dev),
                    int(child_metadata.st_ino),
                    int(child_metadata.st_uid),
                    stat.S_IMODE(child_metadata.st_mode),
                )
            )
        for filename in filenames:
            path = directory_path / filename
            relative_name = path.relative_to(trusted_root).as_posix()
            metadata = path.lstat()
            if (
                relative_name not in expected_files
                or relative_name in observed_files
                or stat.S_ISLNK(metadata.st_mode)
                or not stat.S_ISREG(metadata.st_mode)
                or metadata.st_uid != _effective_uid()
                or metadata.st_nlink != 1
                or stat.S_IMODE(metadata.st_mode) != 0o444
                or metadata.st_size < 0
            ):
                raise GPU5BoundaryError("model snapshot contains an unsafe file")
            observed_files.add(relative_name)
            total_bytes += int(metadata.st_size)
            if total_bytes > MAX_MODEL_SNAPSHOT_BYTES:
                raise GPU5BoundaryError("model snapshot exceeds its byte bound")
            digest, identity = _hash_regular_file(
                path,
                expected_root=trusted_root,
            )
            if digest != expected_files[relative_name]:
                raise GPU5BoundaryError(
                    f"model snapshot digest mismatch: {relative_name}"
                )
            content_records.append((relative_name, digest, int(metadata.st_size)))
            identity_records.append(("file", relative_name, *identity, digest))
    root_after = trusted_root.lstat()
    if (
        observed_files != set(expected_files)
        or observed_directories != expected_directories
        or (
            int(root_after.st_dev),
            int(root_after.st_ino),
            int(root_after.st_uid),
            stat.S_IMODE(root_after.st_mode),
        )
        != root_identity
    ):
        raise GPU5BoundaryError("model snapshot closed-world inventory changed")
    return ModelSnapshot(
        root_path=str(trusted_root),
        root_device=int(root_metadata.st_dev),
        root_inode=int(root_metadata.st_ino),
        root_mode=stat.S_IMODE(root_metadata.st_mode),
        file_count=len(observed_files),
        total_bytes=total_bytes,
        manifest_sha256=manifest_sha256,
        content_digest=_canonical_scope_digest(sorted(content_records)),
        identity_digest=_canonical_scope_digest(sorted(identity_records)),
    )


def _prepare_model_snapshot(
    verified: VerifiedArtifactSet,
    *,
    manifest_sha256: str,
    launch_nonce: str,
) -> ModelSnapshot:
    if not isinstance(verified, VerifiedArtifactSet) or not verified.digests:
        raise GPU5BoundaryError("verified model artifacts are required")
    if len(verified.digests) > MAX_MODEL_MANIFEST_ENTRIES:
        raise GPU5BoundaryError("model snapshot exceeds its file bound")
    nonce = _launch_nonce(launch_nonce)
    expected_files = dict(verified.digests)
    source_files = {
        path.relative_to(verified.root).as_posix(): path for path in verified.files
    }
    if set(source_files) != set(expected_files):
        raise GPU5BoundaryError("verified model file set is inconsistent")
    expected_total_bytes = 0
    for relative_name, source in source_files.items():
        metadata = source.lstat()
        if (
            stat.S_ISLNK(metadata.st_mode)
            or not stat.S_ISREG(metadata.st_mode)
            or not _trusted_owner_uid(metadata.st_uid)
            or _group_or_world_writable(metadata.st_mode)
            or metadata.st_nlink != 1
            or metadata.st_size < 0
        ):
            raise GPU5BoundaryError(f"model snapshot source is unsafe: {relative_name}")
        expected_total_bytes += int(metadata.st_size)
        if expected_total_bytes > MAX_MODEL_SNAPSHOT_BYTES:
            raise GPU5BoundaryError("model snapshot exceeds its byte bound")
    _enforce_snapshot_store_quota(
        MODEL_SNAPSHOT_ROOT,
        max_snapshots=MAX_MODEL_SNAPSHOTS,
        max_bytes=MAX_MODEL_SNAPSHOT_STORE_BYTES,
        reserve_snapshots=1,
        reserve_bytes=expected_total_bytes,
    )
    target = _model_snapshot_path(manifest_sha256, nonce)
    staging = MODEL_SNAPSHOT_ROOT / f".model-{nonce}.partial"
    for path, label in ((target, "target"), (staging, "partial")):
        try:
            path.lstat()
        except FileNotFoundError:
            pass
        except OSError as error:
            raise GPU5BoundaryError(
                f"model snapshot {label} could not be inspected"
            ) from error
        else:
            raise GPU5BoundaryError(f"model snapshot {label} already exists")
    try:
        os.mkdir(staging, 0o700)
    except OSError as error:
        raise GPU5BoundaryError(
            "model snapshot partial root could not be created"
        ) from error

    copied_total = 0
    try:
        for relative_name, expected_digest in sorted(expected_files.items()):
            parts = relative_name.split("/")
            parent = staging
            for part in parts[:-1]:
                parent /= part
                try:
                    os.mkdir(parent, 0o700)
                except FileExistsError:
                    metadata = parent.lstat()
                    if (
                        stat.S_ISLNK(metadata.st_mode)
                        or not stat.S_ISDIR(metadata.st_mode)
                        or metadata.st_uid != _effective_uid()
                        or stat.S_IMODE(metadata.st_mode) != 0o700
                    ):
                        raise GPU5BoundaryError(
                            "model snapshot partial parent is unsafe"
                        )
            copied_total += _copy_model_snapshot_artifact(
                source_files[relative_name],
                staging.joinpath(*parts),
                source_root=verified.root,
                expected_digest=expected_digest,
            )
            if copied_total > expected_total_bytes:
                raise GPU5BoundaryError("model snapshot copy exceeded reserved bytes")
        if copied_total != expected_total_bytes:
            raise GPU5BoundaryError("model snapshot copy size is incomplete")
        _seal_snapshot_directories(staging)
        no_follow = getattr(os, "O_NOFOLLOW", None)
        directory_flag = getattr(os, "O_DIRECTORY", None)
        if not isinstance(no_follow, int) or not isinstance(directory_flag, int):
            raise GPU5BoundaryError(
                "model snapshot atomic publish requires Linux directory flags"
            )
        store_fd = os.open(
            MODEL_SNAPSHOT_ROOT,
            os.O_RDONLY | no_follow | directory_flag,
        )
        try:
            store_metadata = os.fstat(store_fd)
            if (
                not stat.S_ISDIR(store_metadata.st_mode)
                or store_metadata.st_uid != _effective_uid()
                or stat.S_IMODE(store_metadata.st_mode) != 0o700
            ):
                raise GPU5BoundaryError("model snapshot store lost private ownership")
            os.fsync(store_fd)
            os.rename(staging, target)
            os.fsync(store_fd)
        finally:
            os.close(store_fd)
    except BaseException:
        # Partial snapshots are retained and count against quota. Automatic
        # deletion could remove evidence or race an active capability.
        raise
    return _model_snapshot_inventory(
        target,
        manifest_entries=verified.digests,
        manifest_sha256=manifest_sha256,
    )


def _capture_native_execution_snapshot(
    expected_source_commit: str,
    *,
    source_snapshot_root: str,
    source_snapshot_nonce: str,
    model_snapshot_root: str,
    model_manifest_path: str,
    model_manifest_sha256: str,
) -> NativeExecutionSnapshot:
    commit = _validated_expected_source_commit(expected_source_commit)
    if not all(
        isinstance(value, str) and value
        for value in (
            source_snapshot_root,
            source_snapshot_nonce,
            model_snapshot_root,
            model_manifest_path,
            model_manifest_sha256,
        )
    ):
        raise GPU5BoundaryError("native snapshot capability is incomplete")
    source = _source_snapshot_inventory_from_sealed_root(
        Path(source_snapshot_root),
        source_commit=commit,
        launch_nonce=source_snapshot_nonce,
    )
    manifest = Path(model_manifest_path)
    try:
        manifest_digest, _manifest_identity = _hash_regular_file(
            manifest,
            expected_root=Path(source.root_path),
        )
    except GPU5BoundaryError as error:
        raise GPU5BoundaryError("native snapshot manifest is unsafe") from error
    if manifest_digest != model_manifest_sha256:
        raise GPU5BoundaryError("native snapshot manifest digest changed")
    manifest_entries, _identity = _strict_model_manifest_entries(manifest)
    model = _model_snapshot_inventory(
        Path(model_snapshot_root),
        manifest_entries=manifest_entries,
        manifest_sha256=manifest_digest,
    )
    workspace_source = next(
        source_path
        for source_path, destination in _EXPECTED_READ_ONLY_MOUNTS.items()
        if destination == Path("/workspace")
    )
    workspace = _trusted_directory_chain(
        workspace_source,
        label="native workspace capability",
    )
    return NativeExecutionSnapshot(
        source=source,
        model=model,
        manifest_path=str(manifest.resolve(strict=True)),
        workspace_root=str(workspace),
    )


def prepare_native_execution_snapshot(
    expected_source_commit: str,
    model_source: str | Path,
    manifest_relative_name: str,
    *,
    git_runner: Any = subprocess.run,
) -> NativeExecutionSnapshot:
    """Atomically prepare sealed source/model capabilities for native execution."""

    commit = _validated_expected_source_commit(expected_source_commit)
    if not isinstance(manifest_relative_name, str):
        raise GPU5BoundaryError("native snapshot manifest name must be text")
    manifest_name = _safe_source_name(manifest_relative_name.encode("utf-8"))
    if manifest_name != manifest_relative_name:
        raise GPU5BoundaryError("native snapshot manifest name is not canonical")
    workspace_source = next(
        source_path
        for source_path, destination in _EXPECTED_READ_ONLY_MOUNTS.items()
        if destination == Path("/workspace")
    )
    workspace = _trusted_directory_chain(
        workspace_source,
        label="native workspace capability",
    )
    nonce = _launch_nonce()
    with _gpu5_project_lease(commit) as lease:
        source = prepare_source_snapshot(commit, nonce, git_runner=git_runner)
        manifest = Path(source.root_path) / manifest_name
        manifest_sha256, _manifest_identity = _hash_regular_file(
            manifest,
            expected_root=Path(source.root_path),
        )
        verified = verify_closed_world_artifact_layout(
            verify_artifact_manifest(model_source, manifest)
        )
        model = _prepare_model_snapshot(
            verified,
            manifest_sha256=manifest_sha256,
            launch_nonce=nonce,
        )
        snapshot = NativeExecutionSnapshot(
            source=source,
            model=model,
            manifest_path=str(manifest),
            workspace_root=str(workspace),
        )
        observed = verify_native_execution_snapshot(snapshot)
        if observed != snapshot:
            raise GPU5BoundaryError("native execution snapshot changed before publish")
        lease.mark_safe_to_release()
        return snapshot


def verify_native_execution_snapshot(
    snapshot: NativeExecutionSnapshot,
) -> NativeExecutionSnapshot:
    """Fully re-inventory an already published native execution capability."""

    if not isinstance(snapshot, NativeExecutionSnapshot):
        raise GPU5BoundaryError("native execution snapshot type is invalid")
    observed = _capture_native_execution_snapshot(
        snapshot.source.source_commit,
        source_snapshot_root=snapshot.source.root_path,
        source_snapshot_nonce=snapshot.source.launch_nonce,
        model_snapshot_root=snapshot.model.root_path,
        model_manifest_path=snapshot.manifest_path,
        model_manifest_sha256=snapshot.model.manifest_sha256,
    )
    if observed.workspace_root != snapshot.workspace_root or observed != snapshot:
        raise GPU5BoundaryError("native execution snapshot identity changed")
    return observed


def _validation_artifact_profile(value: str) -> ValidationArtifactProfile:
    if not isinstance(value, str):
        raise GPU5BoundaryError("validation artifact profile must be text")
    try:
        return _VALIDATION_ARTIFACT_PROFILES[value]
    except KeyError as error:
        raise GPU5BoundaryError("unknown validation artifact profile") from error


def _raw_model_source_root(
    validation_artifact_profile: str = BASE_CANARY_ARTIFACT_PROFILE,
) -> Path:
    profile = _validation_artifact_profile(validation_artifact_profile)
    source = profile.raw_model_root
    root = _trusted_directory_chain(
        source, label=f"{profile.name} raw model source root"
    )
    if root != source.resolve(strict=True):
        raise GPU5BoundaryError("raw model capability changed identity")
    return root


def _prepare_docker_model_snapshot(
    source_snapshot: SourceSnapshot,
    *,
    launch_nonce: str,
    validation_artifact_profile: str = BASE_CANARY_ARTIFACT_PROFILE,
) -> ModelSnapshot:
    """Copy the manifest-authorized raw model into a sealed Docker capability."""

    if not isinstance(source_snapshot, SourceSnapshot):
        raise GPU5BoundaryError("Docker model snapshot requires a source snapshot")
    nonce = _launch_nonce(launch_nonce)
    if source_snapshot.launch_nonce != nonce:
        raise GPU5BoundaryError("Docker source/model snapshot nonce mismatch")
    observed_source = _source_snapshot_inventory_from_sealed_root(
        Path(source_snapshot.root_path),
        source_commit=source_snapshot.source_commit,
        launch_nonce=nonce,
    )
    if observed_source != source_snapshot:
        raise GPU5BoundaryError("source snapshot changed before model snapshot copy")
    profile = _validation_artifact_profile(validation_artifact_profile)
    manifest_path = Path(observed_source.root_path) / profile.manifest_relative_path
    manifest_sha256, _manifest_identity = _hash_regular_file(
        manifest_path,
        expected_root=Path(observed_source.root_path),
    )
    verified = verify_closed_world_artifact_layout(
        verify_artifact_manifest(
            _raw_model_source_root(profile.name),
            manifest_path,
        )
    )
    return _prepare_model_snapshot(
        verified,
        manifest_sha256=manifest_sha256,
        launch_nonce=nonce,
    )


def _validated_docker_snapshot_capabilities(
    expected_source_commit: str,
    launch_nonce: str,
    *,
    source_snapshot: SourceSnapshot,
    model_snapshot: ModelSnapshot,
) -> tuple[Path, Path]:
    """Validate exact sealed root identities without consulting a raw model path."""

    commit = _validated_expected_source_commit(expected_source_commit)
    nonce = _launch_nonce(launch_nonce)
    if (
        not isinstance(source_snapshot, SourceSnapshot)
        or source_snapshot.source_commit != commit
        or source_snapshot.launch_nonce != nonce
        or source_snapshot.root_mode != 0o555
        or source_snapshot.file_count < 0
        or _SHA256_DIGEST.fullmatch(source_snapshot.content_digest) is None
        or _SHA256_DIGEST.fullmatch(source_snapshot.identity_digest) is None
    ):
        raise GPU5BoundaryError("Docker source snapshot capability is invalid")
    if (
        not isinstance(model_snapshot, ModelSnapshot)
        or model_snapshot.root_mode != 0o555
        or model_snapshot.file_count <= 0
        or model_snapshot.total_bytes <= 0
        or _SHA256_DIGEST.fullmatch(model_snapshot.manifest_sha256) is None
        or _SHA256_DIGEST.fullmatch(model_snapshot.content_digest) is None
        or _SHA256_DIGEST.fullmatch(model_snapshot.identity_digest) is None
    ):
        raise GPU5BoundaryError("Docker model snapshot capability is invalid")
    try:
        source_root = _trusted_directory_chain(
            Path(source_snapshot.root_path),
            label="Docker source snapshot capability",
        )
        model_root = _trusted_directory_chain(
            Path(model_snapshot.root_path),
            label="Docker model snapshot capability",
        )
        expected_source_root = _snapshot_path(commit, nonce).resolve(strict=True)
        expected_model_root = _model_snapshot_path(
            model_snapshot.manifest_sha256, nonce
        ).resolve(strict=True)
        source_metadata = source_root.lstat()
        model_metadata = model_root.lstat()
    except OSError as error:
        raise GPU5BoundaryError(
            "Docker snapshot capability root is unavailable"
        ) from error
    if source_root != expected_source_root or model_root != expected_model_root:
        raise GPU5BoundaryError("Docker snapshot capability escaped its nonce scope")
    for metadata, snapshot, label in (
        (source_metadata, source_snapshot, "source"),
        (model_metadata, model_snapshot, "model"),
    ):
        if (
            stat.S_ISLNK(metadata.st_mode)
            or not stat.S_ISDIR(metadata.st_mode)
            or metadata.st_uid != _effective_uid()
            or stat.S_IMODE(metadata.st_mode) != 0o555
            or int(metadata.st_dev) != snapshot.root_device
            or int(metadata.st_ino) != snapshot.root_inode
        ):
            raise GPU5BoundaryError(f"Docker {label} snapshot root identity changed")
    if (
        source_root == model_root
        or source_root.is_relative_to(model_root)
        or model_root.is_relative_to(source_root)
        or (source_snapshot.root_device, source_snapshot.root_inode)
        == (model_snapshot.root_device, model_snapshot.root_inode)
    ):
        raise GPU5BoundaryError("Docker source/model snapshots are not independent")
    return source_root, model_root


def capture_execution_scope(
    expected_source_commit: str,
    *,
    source_snapshot: SourceSnapshot,
    model_snapshot: ModelSnapshot,
    validation_artifact_profile: str = BASE_CANARY_ARTIFACT_PROFILE,
    git_runner: Any = subprocess.run,
) -> ExecutionScope:
    """Bind a GPU execution to a clean checkout and two sealed snapshots."""

    expected_commit = _validated_expected_source_commit(expected_source_commit)
    repo_root = next(
        source
        for source, destination in _EXPECTED_READ_ONLY_MOUNTS.items()
        if destination == Path("/workspace")
    )
    repo_root = _trusted_directory_chain(repo_root, label="source checkout root")
    if not repo_root.is_dir():
        raise GPU5BoundaryError("source checkout root must be a real directory")

    working_tree_digest, working_identity_digest, _, object_format, entries = (
        _verify_working_checkout(repo_root, expected_commit, git_runner=git_runner)
    )
    if not isinstance(source_snapshot, SourceSnapshot):
        raise GPU5BoundaryError("an exact source snapshot is required")
    expected_snapshot_path = _snapshot_path(
        expected_commit, source_snapshot.launch_nonce
    ).resolve(strict=True)
    if Path(source_snapshot.root_path).resolve(strict=True) != expected_snapshot_path:
        raise GPU5BoundaryError("source snapshot path is outside its nonce scope")
    observed_snapshot = _snapshot_inventory(
        expected_snapshot_path,
        source_commit=expected_commit,
        launch_nonce=source_snapshot.launch_nonce,
        object_format=object_format,
        expected_entries=entries,
    )
    if observed_snapshot != source_snapshot:
        raise GPU5BoundaryError("source snapshot changed identity or content")

    _source_root, model_root = _validated_docker_snapshot_capabilities(
        expected_commit,
        source_snapshot.launch_nonce,
        source_snapshot=source_snapshot,
        model_snapshot=model_snapshot,
    )
    profile = _validation_artifact_profile(validation_artifact_profile)
    manifest_path = Path(source_snapshot.root_path) / profile.manifest_relative_path
    manifest_sha256, _manifest_identity = _hash_regular_file(
        manifest_path,
        expected_root=Path(source_snapshot.root_path),
    )
    manifest_entries, _manifest_file_identity = _strict_model_manifest_entries(
        manifest_path
    )
    if manifest_sha256 != model_snapshot.manifest_sha256:
        raise GPU5BoundaryError("model snapshot manifest capability changed")
    observed_model = _model_snapshot_inventory(
        model_root,
        manifest_entries=manifest_entries,
        manifest_sha256=manifest_sha256,
    )
    if observed_model != model_snapshot:
        raise GPU5BoundaryError("model snapshot changed identity or content")
    return ExecutionScope(
        source_commit=expected_commit,
        source_tree_digest=source_snapshot.content_digest,
        source_identity_digest=source_snapshot.identity_digest,
        source_file_count=source_snapshot.file_count,
        source_root_device=source_snapshot.root_device,
        source_root_inode=source_snapshot.root_inode,
        model_manifest_sha256=manifest_sha256,
        model_tree_digest=model_snapshot.content_digest,
        model_identity_digest=model_snapshot.identity_digest,
        model_file_count=model_snapshot.file_count,
        model_root_device=model_snapshot.root_device,
        model_root_inode=model_snapshot.root_inode,
        validation_artifact_profile=profile.name,
        snapshot_path=source_snapshot.root_path,
        snapshot_nonce=source_snapshot.launch_nonce,
        snapshot_mode=source_snapshot.root_mode,
        working_tree_digest=working_tree_digest,
        working_identity_digest=working_identity_digest,
    )


def _validated_environment(environment: Mapping[str, str] | None) -> tuple[str, ...]:
    supplied = dict(environment or {})
    if supplied != _REQUIRED_CONTAINER_ENVIRONMENT:
        raise GPU5BoundaryError(
            "container environment must equal the fixed offline allowlist"
        )
    values: list[str] = []
    total_chars = 0
    for name, value in sorted(supplied.items()):
        if _ENV_NAME.fullmatch(name) is None:
            raise GPU5BoundaryError("invalid Docker environment name")
        if name in _RESERVED_CONTAINER_GPU_ENV:
            raise GPU5BoundaryError(
                f"container {name} override is forbidden; physical GPU5 maps to cuda:0"
            )
        if (
            not isinstance(value, str)
            or "\x00" in value
            or len(value) > MAX_CONTAINER_ENVIRONMENT_VALUE_CHARS
        ):
            raise GPU5BoundaryError("invalid Docker environment value")
        total_chars += len(name) + len(value)
        values.extend(("--env", f"{name}={value}"))
    if (
        len(supplied) > MAX_CONTAINER_ENVIRONMENT_ITEMS
        or total_chars > MAX_CONTAINER_ENVIRONMENT_TOTAL_CHARS
    ):
        raise GPU5BoundaryError("container environment exceeds the fixed bound")
    return tuple(values)


def _container_name(value: str | None = None) -> str:
    name = value or f"cognios-gpu5-{secrets.token_hex(6)}"
    if _CONTAINER_NAME.fullmatch(name) is None:
        raise GPU5BoundaryError("invalid or non-project Docker container name")
    return name


def _launch_nonce(value: str | None = None) -> str:
    nonce = secrets.token_hex(16) if value is None else value
    if not isinstance(nonce, str) or _LAUNCH_NONCE.fullmatch(nonce) is None:
        raise GPU5BoundaryError("invalid GPU5 launch nonce")
    return nonce


def _expected_launch_labels(
    *,
    expected_source_commit: str,
    launch_nonce: str,
    execution_profile: str = "release",
    validation_artifact_profile: str = BASE_CANARY_ARTIFACT_PROFILE,
) -> dict[str, str]:
    if execution_profile not in {"release", "inspection"}:
        raise GPU5BoundaryError("invalid GPU5 execution profile")
    artifact_profile = _validation_artifact_profile(validation_artifact_profile)
    return {
        _GUARD_LABEL: "gpu5",
        _SOURCE_COMMIT_LABEL: _validated_expected_source_commit(expected_source_commit),
        _LAUNCH_NONCE_LABEL: _launch_nonce(launch_nonce),
        _EXECUTION_PROFILE_LABEL: execution_profile,
        _VALIDATION_ARTIFACT_PROFILE_LABEL: artifact_profile.name,
    }


def _source_within_allowed_roots(
    source: Path,
    roots: Sequence[Path] = _ALLOWED_MOUNT_ROOTS,
) -> bool:
    return any(source == root or source.is_relative_to(root) for root in roots)


def _runtime_mount_map(
    *,
    expected_source_commit: str,
    launch_nonce: str,
    source_snapshot: SourceSnapshot,
    model_snapshot: ModelSnapshot,
    validation_artifact_profile: str = BASE_CANARY_ARTIFACT_PROFILE,
) -> dict[Path, Path]:
    profile = _validation_artifact_profile(validation_artifact_profile)
    source_root, model_root = _validated_docker_snapshot_capabilities(
        expected_source_commit,
        launch_nonce,
        source_snapshot=source_snapshot,
        model_snapshot=model_snapshot,
    )
    return {
        source_root: Path("/workspace"),
        model_root: profile.container_model_root,
    }


def _validated_read_only_mount(
    value: str, *, expected_mounts: Mapping[Path, Path]
) -> str:
    if not isinstance(value, str) or not value or "\x00" in value:
        raise GPU5BoundaryError("invalid Docker mount")
    try:
        source_raw, destination_raw, mode = value.rsplit(":", 2)
    except ValueError as error:
        raise GPU5BoundaryError(
            "Docker mounts must use source:destination:ro"
        ) from error
    if mode != "ro":
        raise GPU5BoundaryError("Docker mounts must be read-only")
    source = Path(source_raw)
    destination = Path(destination_raw)
    if not source.is_absolute() or not destination.is_absolute():
        raise GPU5BoundaryError("Docker mount paths must be absolute")
    try:
        resolved_source = source.resolve(strict=True)
        expected = {
            root.resolve(strict=True): container
            for root, container in expected_mounts.items()
        }
    except OSError as error:
        raise GPU5BoundaryError("Docker mount source must exist") from error
    normalized_destination = Path(os.path.normpath(destination.as_posix()))
    if resolved_source not in expected:
        raise GPU5BoundaryError("Docker mount source must be an exact pinned root")
    if normalized_destination != expected[resolved_source]:
        raise GPU5BoundaryError(
            "Docker mount destination does not match the pinned root"
        )
    return f"{resolved_source}:{normalized_destination}:ro"


def _argument_value(command: Sequence[str], option: str) -> str:
    if command.count(option) != 1:
        raise GPU5BoundaryError(f"validator requires exactly one {option}")
    position = command.index(option)
    if position + 1 >= len(command):
        raise GPU5BoundaryError(f"validator {option} value is missing")
    return command[position + 1]


def _number_option(
    options: Mapping[str, str | None],
    name: str,
    *,
    minimum: float,
    maximum: float,
    integer: bool = False,
) -> float:
    if name not in options or options[name] is None:
        raise GPU5BoundaryError(f"validator requires {name}")
    raw = str(options[name])
    try:
        value = float(raw)
    except ValueError as error:
        raise GPU5BoundaryError(f"validator {name} must be numeric") from error
    if integer and (not value.is_integer() or re.fullmatch(r"-?[0-9]+", raw) is None):
        raise GPU5BoundaryError(f"validator {name} must be an integer")
    if not minimum <= value <= maximum:
        raise GPU5BoundaryError(f"validator {name} is outside its bounded range")
    return value


def _validated_validator_options(values: tuple[str, ...]) -> dict[str, str | None]:
    if not 7 <= len(values) <= MAX_VALIDATOR_ARGV_TOKENS:
        raise GPU5BoundaryError("validator argv length is outside the fixed bound")
    if values[:2] != ("-I", "-B") or values[2] not in _ALLOWED_VALIDATORS:
        raise GPU5BoundaryError("container command is outside the validator allowlist")
    for token in values:
        if (
            len(token) > MAX_VALIDATOR_TOKEN_CHARS
            or any(character in token for character in ("\x00", "\n", "\r", "`"))
            or token in {"--", "-c", "-m", ";", "|", "||", "&&", ">", ">>", "<", "&"}
        ):
            raise GPU5BoundaryError(
                "validator argv contains a shell or oversized token"
            )
    script = values[2]
    value_options = _VALIDATOR_VALUE_OPTIONS[script]
    flag_options = _VALIDATOR_FLAG_OPTIONS[script]
    parsed: dict[str, str | None] = {}
    position = 3
    while position < len(values):
        option = values[position]
        if option in parsed:
            raise GPU5BoundaryError(f"duplicate validator option: {option}")
        if option in flag_options:
            parsed[option] = None
            position += 1
            continue
        if option not in value_options:
            raise GPU5BoundaryError(f"unknown validator option: {option}")
        if position + 1 >= len(values) or values[position + 1].startswith("--"):
            raise GPU5BoundaryError(f"validator value is missing for {option}")
        parsed[option] = values[position + 1]
        position += 2
    for required in ("--model", "--manifest"):
        if required not in parsed:
            raise GPU5BoundaryError(f"validator requires {required}")
    for required in ("--physical-gpu-index", "--gpu-query-context"):
        if required not in parsed:
            raise GPU5BoundaryError(f"guarded GPU5 validator requires {required}")
    _number_option(parsed, "--physical-gpu-index", minimum=5, maximum=5, integer=True)
    if parsed.get("--gpu-query-context") != "gpu5-container":
        raise GPU5BoundaryError(
            "guarded Docker validators require the GPU5 container query context"
        )
    prompt = parsed.get("--prompt")
    if prompt is not None and len(prompt) > MAX_VALIDATOR_PROMPT_CHARS:
        raise GPU5BoundaryError("validator prompt exceeds the bounded length")
    if script.endswith("validate_agent_completion.py"):
        if "--timeout" in parsed:
            _number_option(parsed, "--timeout", minimum=1, maximum=120)
        if "--turns" in parsed:
            _number_option(parsed, "--turns", minimum=1, maximum=100, integer=True)
        suite = parsed.get("--suite")
        if suite is not None and suite not in {
            "completion-stress",
            "product-e4b-it-20",
        }:
            raise GPU5BoundaryError("validator suite is outside the fixed allowlist")
    elif script.endswith("validate_agent_casual_korean.py"):
        if "--timeout" in parsed:
            _number_option(parsed, "--timeout", minimum=1, maximum=120)
    elif script.endswith("validate_gemma4_runtime.py"):
        if "--workspace-mib" in parsed:
            _number_option(
                parsed, "--workspace-mib", minimum=1, maximum=4_096, integer=True
            )
        if "--vram-limit-gib" in parsed:
            _number_option(parsed, "--vram-limit-gib", minimum=1, maximum=16.7)
    else:
        ranges = {
            "--layer-index": (-1, 128, True),
            "--tolerance": (1.0e-9, 5.0e-3, False),
            "--max-iter": (1, 128, True),
            "--history": (1, 64, True),
            "--fallback-steps": (1, 256, True),
            "--fallback-damping": (0, 1, False),
            "--contractive-delta-scale": (0, 1, False),
            "--certified-delta-lipschitz-bound": (
                0,
                MAX_DEQ_RAW_DELTA_LIPSCHITZ_BOUND,
                False,
            ),
            "--vram-limit-gib": (1, 16.7, False),
        }
        numeric_values: dict[str, float] = {}
        for option, (minimum, maximum, integer) in ranges.items():
            if option in parsed:
                numeric_values[option] = _number_option(
                    parsed,
                    option,
                    minimum=minimum,
                    maximum=maximum,
                    integer=integer,
                )
        for positive_option in (
            "--fallback-damping",
            "--contractive-delta-scale",
        ):
            if numeric_values.get(positive_option, 1.0) <= 0.0:
                raise GPU5BoundaryError(
                    f"validator {positive_option} must be strictly positive"
                )
        raw_bound = numeric_values.get("--certified-delta-lipschitz-bound")
        scale = numeric_values.get(
            "--contractive-delta-scale", DEFAULT_DEQ_CONTRACTIVE_DELTA_SCALE
        )
        if (
            raw_bound is not None
            and scale * raw_bound > MAX_DEQ_EFFECTIVE_LIPSCHITZ_BOUND
        ):
            raise GPU5BoundaryError(
                "DEQ scale*raw-bound exceeds the 0.95 experimental safety margin"
            )
        if "--allow-uncertified-experimental" not in parsed:
            raise GPU5BoundaryError(
                "DEQ inspection requires explicitly labelled uncertified experimental mode"
            )
        provenance = parsed.get("--contractivity-provenance")
        if provenance is not None:
            relative = Path(provenance)
            if (
                relative.is_absolute()
                or not relative.parts
                or ".." in relative.parts
                or relative.as_posix() != provenance
            ):
                raise GPU5BoundaryError(
                    "DEQ contractivity provenance must be a safe relative POSIX path"
                )
    return parsed


def _validated_validator_command(
    command: Sequence[str],
    validation_artifact_profile: str = BASE_CANARY_ARTIFACT_PROFILE,
) -> tuple[str, ...]:
    values = tuple(command)
    options = _validated_validator_options(values)
    profile = _validation_artifact_profile(validation_artifact_profile)
    if values[2] not in profile.validators:
        raise GPU5BoundaryError(
            "validator is outside the selected artifact profile allowlist"
        )
    if options["--model"] != profile.container_model_root.as_posix():
        raise GPU5BoundaryError(
            "validator model path is outside the pinned read-only mount"
        )
    expected_manifest = f"/workspace/{profile.manifest_relative_path}"
    if options["--manifest"] != expected_manifest:
        raise GPU5BoundaryError(
            "validator manifest path is outside the pinned checkout"
        )
    return values


def _validated_release_validator_command(
    command: Sequence[str],
    validation_artifact_profile: str = BASE_CANARY_ARTIFACT_PROFILE,
) -> tuple[str, ...]:
    values = _validated_validator_command(command, validation_artifact_profile)
    profile = _validation_artifact_profile(validation_artifact_profile)
    options = _validated_validator_options(values)
    if profile.name == PRODUCT_E4B_IT_ARTIFACT_PROFILE:
        if values[2] != "/workspace/scripts/validate_agent_completion.py":
            raise GPU5BoundaryError(
                "product Stage G release requires the single 20-turn acceptance validator"
            )
        if options.get("--suite") != "product-e4b-it-20":
            raise GPU5BoundaryError(
                "product Stage G release requires the product-e4b-it-20 suite"
            )
        if options.get("--turns") != "20":
            raise GPU5BoundaryError(
                "product Stage G release requires exactly 20 conversation turns"
            )
        if "--strict-json" not in options:
            raise GPU5BoundaryError(
                "product Stage G release requires strict JSON-only evidence"
            )
        return values
    if values[2].endswith("validate_gemma4_deq.py"):
        raise GPU5BoundaryError(
            "Stage G DEQ release is disabled until a cryptographically pinned "
            "trust contract is implemented"
        )
    return values


@contextmanager
def _open_evidence_target(filename: str) -> Iterator[_EvidenceHandle]:
    if not isinstance(filename, str) or _EVIDENCE_FILENAME.fullmatch(filename) is None:
        raise GPU5BoundaryError("invalid bounded evidence filename")
    configured_root = EVIDENCE_HOST_ROOT
    root_flags = os.O_RDONLY
    if hasattr(os, "O_DIRECTORY"):
        root_flags |= os.O_DIRECTORY
    if hasattr(os, "O_NOFOLLOW"):
        root_flags |= os.O_NOFOLLOW
    try:
        root_fd = os.open(configured_root, root_flags)
    except OSError as error:
        raise GPU5BoundaryError(
            "exact server evidence directory must already exist"
        ) from error
    file_fd: int | None = None
    try:
        root_metadata = os.fstat(root_fd)
        if (
            not stat.S_ISDIR(root_metadata.st_mode)
            or root_metadata.st_uid != os.geteuid()
            or stat.S_IMODE(root_metadata.st_mode) != 0o700
        ):
            raise GPU5BoundaryError(
                "server evidence root must remain owner-controlled mode 0700"
            )
        file_flags = os.O_RDWR | os.O_CREAT | os.O_EXCL
        if hasattr(os, "O_NOFOLLOW"):
            file_flags |= os.O_NOFOLLOW
        try:
            file_fd = os.open(filename, file_flags, 0o600, dir_fd=root_fd)
        except OSError as error:
            raise GPU5BoundaryError(
                "evidence target must be a new non-symlink file"
            ) from error
        os.fchmod(file_fd, 0o600)
        file_metadata = os.fstat(file_fd)
        if (
            not stat.S_ISREG(file_metadata.st_mode)
            or file_metadata.st_uid != os.geteuid()
            or file_metadata.st_nlink != 1
            or stat.S_IMODE(file_metadata.st_mode) != 0o600
        ):
            raise GPU5BoundaryError("new evidence file metadata is unsafe")
        os.fsync(root_fd)
        handle = _EvidenceHandle(
            root_path=configured_root.resolve(strict=True),
            filename=filename,
            root_fd=root_fd,
            file_fd=file_fd,
            root_device=int(root_metadata.st_dev),
            root_inode=int(root_metadata.st_ino),
            file_device=int(file_metadata.st_dev),
            file_inode=int(file_metadata.st_ino),
        )
        yield handle
    finally:
        if file_fd is not None:
            os.close(file_fd)
        os.close(root_fd)


def _bound_evidence_file_size() -> None:
    if resource is None:  # pragma: no cover - guarded before Linux server execution
        raise RuntimeError("RLIMIT_FSIZE is unavailable")
    resource.setrlimit(resource.RLIMIT_FSIZE, (MAX_EVIDENCE_BYTES, MAX_EVIDENCE_BYTES))


def _evidence_snapshot(handle: _EvidenceHandle) -> tuple[bytes, int, str]:
    """Read, bind, and hash validator evidence in one identity transaction."""

    try:
        root_now = os.stat(EVIDENCE_HOST_ROOT, follow_symlinks=False)
        named_now = os.stat(
            handle.filename,
            dir_fd=handle.root_fd,
            follow_symlinks=False,
        )
        file_now = os.fstat(handle.file_fd)
    except OSError as error:
        raise GPU5BoundaryError("bounded evidence identity is unreadable") from error
    if (
        not stat.S_ISDIR(root_now.st_mode)
        or (int(root_now.st_dev), int(root_now.st_ino))
        != (handle.root_device, handle.root_inode)
        or not stat.S_ISREG(named_now.st_mode)
        or (int(named_now.st_dev), int(named_now.st_ino))
        != (handle.file_device, handle.file_inode)
        or (int(file_now.st_dev), int(file_now.st_ino))
        != (handle.file_device, handle.file_inode)
        or root_now.st_uid != os.geteuid()
        or stat.S_IMODE(root_now.st_mode) != 0o700
        or named_now.st_uid != os.geteuid()
        or stat.S_IMODE(named_now.st_mode) != 0o600
        or file_now.st_uid != os.geteuid()
        or stat.S_IMODE(file_now.st_mode) != 0o600
        or file_now.st_nlink != 1
    ):
        raise GPU5BoundaryError("evidence root or target changed identity")
    if file_now.st_size <= 0:
        raise GPU5BoundaryError("validator produced no evidence")
    if file_now.st_size > MAX_EVIDENCE_BYTES:
        raise GPU5BoundaryError("validator evidence exceeded the file-size bound")
    digest = hashlib.sha256()
    chunks: list[bytes] = []
    observed = 0
    os.lseek(handle.file_fd, 0, os.SEEK_SET)
    while True:
        chunk = os.read(
            handle.file_fd, min(1024 * 1024, MAX_EVIDENCE_BYTES + 1 - observed)
        )
        if not chunk:
            break
        observed += len(chunk)
        if observed > MAX_EVIDENCE_BYTES:
            raise GPU5BoundaryError("validator evidence exceeded the file-size bound")
        chunks.append(chunk)
        digest.update(chunk)
    after_root = os.stat(EVIDENCE_HOST_ROOT, follow_symlinks=False)
    after_named = os.stat(
        handle.filename,
        dir_fd=handle.root_fd,
        follow_symlinks=False,
    )
    after = os.fstat(handle.file_fd)
    if observed != file_now.st_size or (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    ) != (file_now.st_dev, file_now.st_ino, file_now.st_size, file_now.st_mtime_ns):
        raise GPU5BoundaryError("validator evidence changed while hashing")
    if (
        (int(after_root.st_dev), int(after_root.st_ino))
        != (handle.root_device, handle.root_inode)
        or after_root.st_uid != os.geteuid()
        or stat.S_IMODE(after_root.st_mode) != 0o700
        or (int(after_named.st_dev), int(after_named.st_ino))
        != (handle.file_device, handle.file_inode)
        or after_named.st_uid != os.geteuid()
        or stat.S_IMODE(after_named.st_mode) != 0o600
        or after_named.st_nlink != 1
        or after.st_uid != os.geteuid()
        or stat.S_IMODE(after.st_mode) != 0o600
        or after.st_nlink != 1
    ):
        raise GPU5BoundaryError("evidence ownership or mode changed after execution")
    os.fsync(handle.file_fd)
    os.fsync(handle.root_fd)
    encoded = b"".join(chunks)
    if len(encoded) != observed:
        raise GPU5BoundaryError("validator evidence snapshot length is inconsistent")
    return encoded, observed, digest.hexdigest()


def _evidence_digest(handle: _EvidenceHandle) -> tuple[int, str]:
    """Compatibility wrapper for identity/digest-only guard tests."""

    _encoded, observed, digest = _evidence_snapshot(handle)
    return observed, digest


def _product_substantive_sentence_keys(text: str) -> frozenset[str]:
    keys: set[str] = set()
    for match in _PRODUCT_SENTENCE.finditer(text):
        key = re.sub(r"\s+", " ", match.group(0)).strip()
        key = re.sub(r"^(?:[-*+]\s+|\d{1,4}[.)]\s+)", "", key)
        key = key.rstrip(" .!?。！？").casefold()
        if len(key) >= 24:
            keys.add(key)
    return frozenset(keys)


def _product_complete_sentence_keys(text: str) -> tuple[str, ...]:
    keys: list[str] = []
    for match in _PRODUCT_SENTENCE.finditer(text[:32_000]):
        key = re.sub(r"\s+", " ", match.group(0)).strip().casefold()
        key = re.sub(r"^(?:[-*+]\s+|\d{1,4}[.)]\s+)", "", key)
        key = key.rstrip(" .!?。！？")
        if key:
            keys.append(key)
    return tuple(keys)


def _product_repetition_metrics(text: str) -> dict[str, Any]:
    sentences = list(_product_complete_sentence_keys(text))
    counts: dict[str, int] = {}
    for sentence in sentences:
        counts[sentence] = counts.get(sentence, 0) + 1
    duplicate_count = sum(max(0, count - 1) for count in counts.values())
    repeated = [sentence for sentence, count in counts.items() if count > 1]
    return {
        "sentence_count": len(sentences),
        "unique_sentence_count": len(counts),
        "duplicate_sentence_count": duplicate_count,
        "duplicate_sentence_rate": (
            0.0 if not sentences else duplicate_count / len(sentences)
        ),
        "repeated_sentences": repeated[:8],
    }


def _product_has_repeated_paragraph(text: str) -> bool:
    observed: set[str] = set()
    for block in re.split(r"(?:\r?\n\s*)+", text[:32_000]):
        key = re.sub(r"\s+", " ", block).strip().casefold()
        if len(key) < 8:
            continue
        if key in observed:
            return True
        observed.add(key)
    return False


def _product_has_empty_trailing_list_item(text: str) -> bool:
    stripped = text.rstrip()
    match = _PRODUCT_TRAILING_LIST_MARKER_RE.search(stripped)
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
    return (
        _PRODUCT_NUMBERED_ITEM_WITH_CONTENT_RE.search(stripped[: match.start()])
        is not None
    )


def _product_korean_completion_metrics(text: str) -> dict[str, Any]:
    stripped = text.strip()
    contains_korean = _PRODUCT_HANGUL.search(stripped) is not None
    reasons: list[str] = []
    if contains_korean and not stripped.endswith(_PRODUCT_COMPLETE_ENDINGS):
        reasons.append("missing_terminal_punctuation")
    if contains_korean and _PRODUCT_DANGLING_KOREAN_RE.search(stripped) is not None:
        reasons.append("dangling_korean_clause")
    if contains_korean and _product_has_empty_trailing_list_item(stripped):
        reasons.append("empty_trailing_list_item")
    return {
        "contains_korean": contains_korean,
        "complete": contains_korean and not reasons,
        "reasons": reasons,
    }


def _product_role_token_leaks(text: str) -> list[str]:
    return [match.group(0).strip() for match in _PRODUCT_ROLE_LEAK.finditer(text)][:16]


def _product_control_marker_leaks(text: str) -> list[str]:
    lowered = text.casefold()
    return [
        marker for marker in _PRODUCT_CONTROL_MARKERS if marker.casefold() in lowered
    ]


def _product_topic_anchors_satisfied(
    text: str,
    groups: tuple[tuple[str, ...], ...],
) -> bool:
    folded = text.casefold()
    return all(any(term.casefold() in folded for term in group) for group in groups)


def _validate_product_acceptance_payload(
    payload: Any,
    *,
    expected_model_manifest_sha256: str | None = None,
) -> str:
    """Validate the exact machine-readable product acceptance component."""

    if not isinstance(payload, dict):
        raise GPU5BoundaryError("product acceptance evidence must be a JSON object")
    summary = payload.get("summary")
    turns = payload.get("turns")
    if (
        payload.get("schema") != "cogni.agent.completion.stress.v2"
        or payload.get("suite") != "product-e4b-it-20"
        or payload.get("status") != "passed"
        or payload.get("all_checks_passed") is not True
        or payload.get("requested_turns") != 20
        or not isinstance(turns, list)
        or len(turns) != 20
        or not isinstance(summary, dict)
        or summary.get("strict_completion_stress_gate_passed") is not True
    ):
        raise GPU5BoundaryError(
            "product acceptance JSON does not satisfy the exact 20-turn release schema"
        )

    forbidden_failure_fields = {
        "error",
        "cleanup_error",
        "post_manifest_error",
        "post_gpu_identity_error",
        "aborted_after_turn",
    }
    cleanup = payload.get("cleanup_checks")
    identity_before = payload.get("gpu_identity_before")
    identity_after = payload.get("gpu_identity_after")
    memory_scope = payload.get("memory_evidence_scope")
    factbook = payload.get("factbook")
    factbook_model = factbook.get("model") if isinstance(factbook, dict) else None
    if expected_model_manifest_sha256 is not None and (
        not isinstance(expected_model_manifest_sha256, str)
        or not _SHA256_DIGEST.fullmatch(expected_model_manifest_sha256)
    ):
        raise GPU5BoundaryError("expected product manifest digest is invalid")
    expected_identity = {
        "physical_index": PROJECT_PHYSICAL_GPU_INDEX,
        "uuid": PROJECT_GPU_UUID,
        "query_context": "gpu5-container",
        "logical_device_count": 1,
        "logical_device_index": 0,
    }
    if (
        payload.get("completed_turns") != 20
        or payload.get("recommended_stress_turns") != 20
        or payload.get("worker_cleaned") is not True
        or payload.get("gpu_lease_released") is not True
        or cleanup != {"worker_cleaned": True, "gpu_lease_released": True}
        or payload.get("verified_files") != 7
        or payload.get("verified_files_after") != 7
        or payload.get("physical_gpu_index") != PROJECT_PHYSICAL_GPU_INDEX
        or payload.get("gpu_query_context") != "gpu5-container"
        or payload.get("logical_cuda_device_count") != 1
        or payload.get("logical_cuda_device_index") != 0
        or identity_before != expected_identity
        or identity_after != expected_identity
        or payload.get("gpu_lease_history")
        != [{"epoch": 1, "purpose": "inference", "reason": "released"}]
        or not isinstance(memory_scope, dict)
        or memory_scope.get("kind") != _PRODUCT_MEMORY_SAMPLE_SCOPE
        or memory_scope.get("one_sample_per_expected_resident_turn") is not True
        or memory_scope.get("captures_peak") is not False
        or memory_scope.get("captures_sustained_usage") is not False
        or memory_scope.get("gpu_memory_spot_sample_threshold_bytes")
        != _PRODUCT_GPU_SPOT_SAMPLE_LIMIT_BYTES
        or memory_scope.get("full_runtime_peak_validator")
        != "scripts/validate_gemma4_runtime.py"
        or memory_scope.get("full_runtime_peak_metric")
        != "torch.cuda.max_memory_allocated"
        or not isinstance(factbook, dict)
        or factbook.get("schema_version") != 1
        or factbook.get("build_version") != _PRODUCT_BUILD_VERSION
        or not isinstance(factbook.get("device"), str)
        or not factbook["device"].strip()
        or factbook.get("target_device") != "RTX 4090 24GB"
        or not isinstance(factbook_model, dict)
        or factbook_model.get("label") != _PRODUCT_MODEL_LABEL
        or factbook_model.get("dense") is not True
        or factbook_model.get("stored_parameters") != _PRODUCT_STORED_PARAMETERS
        or factbook_model.get("effective_parameters") != _PRODUCT_EFFECTIVE_PARAMETERS
        or not isinstance(factbook_model.get("manifest_sha256"), str)
        or not _SHA256_DIGEST.fullmatch(factbook_model["manifest_sha256"])
        or (
            expected_model_manifest_sha256 is not None
            and factbook_model["manifest_sha256"] != expected_model_manifest_sha256
        )
        or not isinstance(factbook_model.get("config_sha256"), str)
        or not _SHA256_DIGEST.fullmatch(factbook_model["config_sha256"])
        or any(field in payload for field in forbidden_failure_fields)
    ):
        raise GPU5BoundaryError(
            "product acceptance JSON lacks exact cleanup, manifest, or GPU5 identity proof"
        )

    resident_pids: set[int] = set()
    answer_digests: set[str] = set()
    prior_sentence_keys: set[str] = set()
    expected_worker_turns = 0
    observed_gpu_samples = 0
    maximum_gpu_sample = 0
    for expected_turn, (turn, expected_case) in enumerate(
        zip(turns, _PRODUCT_ACCEPTANCE_CASES, strict=True), 1
    ):
        if not isinstance(turn, dict):
            raise GPU5BoundaryError("product acceptance turn must be a JSON object")
        checks = turn.get("checks")
        worker = turn.get("worker")
        expected_generation_mode = (
            "factbook" if expected_turn in _PRODUCT_FACTBOOK_TURNS else "cogni_core"
        )
        expected_route = (
            "grounded" if expected_turn in _PRODUCT_FACTBOOK_TURNS else "generated"
        )
        expected_worker_running = expected_turn in _PRODUCT_WORKER_EXPECTED_TURNS
        expected_session_id = (
            "completion-a"
            if expected_turn <= 4 or expected_turn % 2 == 0
            else "completion-b"
        )
        expected_peer_session_id = (
            "completion-b" if expected_session_id == "completion-a" else "completion-a"
        )
        answer = turn.get("answer")
        elapsed = turn.get("elapsed_seconds")
        isolation = turn.get("session_isolation")
        repetition = turn.get("repetition")
        korean = turn.get("korean_completion")
        if not isinstance(answer, str):
            raise GPU5BoundaryError("product acceptance turn answer is not text")
        normalized_answer = re.sub(r"\s+", " ", answer).strip().casefold()
        digits_answer = re.sub(r"(?<=\d),(?=\d)", "", normalized_answer)
        answer_digest = hashlib.sha256(normalized_answer.encode("utf-8")).hexdigest()
        current_sentence_keys = _product_substantive_sentence_keys(answer)
        reused_sentence_keys = current_sentence_keys & prior_sentence_keys
        cross_turn_sentence_echo = len(reused_sentence_keys) >= 2 or (
            len(current_sentence_keys) == 1
            and bool(reused_sentence_keys)
            and len(next(iter(current_sentence_keys))) >= 96
        )
        recomputed_repetition = _product_repetition_metrics(answer)
        recomputed_korean = _product_korean_completion_metrics(answer)
        recomputed_role_leaks = _product_role_token_leaks(answer)
        recomputed_control_leaks = _product_control_marker_leaks(answer)
        expected_continuations = (
            _PRODUCT_CONTINUATION_COUNT
            if expected_turn == _PRODUCT_CONTINUATION_TURN
            else 0
        )
        identity_model_exact = normalized_answer.count(_PRODUCT_MODEL_LABEL) == 1
        identity_version_exact = normalized_answer.count(_PRODUCT_BUILD_VERSION) == 1
        identity_parameters_exact = (
            digits_answer.count(str(_PRODUCT_STORED_PARAMETERS)) == 1
            and digits_answer.count(str(_PRODUCT_EFFECTIVE_PARAMETERS)) == 1
        )
        complete_sentence_keys = _product_complete_sentence_keys(answer)
        independently_recomputed_checks = {
            "balanced_smart_quotes": answer.count("“") == answer.count("”")
            and answer.count("‘") == answer.count("’"),
            "canonical_user_prompt": turn.get("new_user_count") == 1
            and turn.get("observed_user_prompt")
            == _PRODUCT_ACCEPTANCE_PROMPTS[expected_turn - 1],
            "complete_stage": turn.get("state_stage") == "complete",
            "contains_korean": recomputed_korean["contains_korean"] is True,
            "continuation_contract": turn.get("continuations")
            == expected_continuations,
            "exactly_one_assistant": turn.get("new_assistant_count") == 1,
            "factbook_model_exact": expected_turn != 1 or identity_model_exact,
            "factbook_parameters_exact": expected_turn != 1
            or identity_parameters_exact,
            "factbook_version_exact": expected_turn != 1 or identity_version_exact,
            "finish_stop": turn.get("finish_reason") == "stop",
            "grounding_route": (
                turn.get("generated_tokens") == 0
                if expected_route == "grounded"
                else type(turn.get("generated_tokens")) is int
                and turn["generated_tokens"] > 0
            ),
            "interactive_latency_within_limit": type(elapsed) in {int, float}
            and math.isfinite(float(elapsed))
            and 0 <= float(elapsed) <= _PRODUCT_MAX_TURN_SECONDS,
            "korean_complete": recomputed_korean["complete"] is True,
            "natural_boundary": len(answer.strip()) < 80
            or answer.rstrip().endswith(_PRODUCT_COMPLETE_ENDINGS),
            "no_control_marker": not recomputed_control_leaks,
            "no_cross_turn_exact_duplicate": answer_digest not in answer_digests,
            "no_cross_turn_sentence_echo": not cross_turn_sentence_echo,
            "no_false_7b_identity": re.search(
                r"(?:70\s*억|\b7\s*b(?:illion)?\b)",
                normalized_answer,
            )
            is None,
            "no_repeated_paragraph": not _product_has_repeated_paragraph(answer),
            "no_repeated_sentence": recomputed_repetition["duplicate_sentence_count"]
            == 0,
            "no_role_leak": not recomputed_role_leaks,
            "no_short_sentence_loop": len(complete_sentence_keys)
            == len(set(complete_sentence_keys)),
            "non_empty": bool(normalized_answer),
            "not_explicitly_truncated": turn.get("explicit_truncation") is False,
            "not_truncated": turn.get("answer_truncated") is False,
            "required_literal_ending": expected_turn != 3
            or answer.rstrip().endswith("이상입니다."),
            "required_period_ending": expected_turn != 4
            or answer.rstrip().endswith("."),
            "succeeded": turn.get("state_status") == "succeeded",
            "topic_anchors_satisfied": _product_topic_anchors_satisfied(
                answer,
                _PRODUCT_REQUIRED_TERM_GROUPS[expected_turn - 1],
            ),
        }
        if (
            turn.get("turn") != expected_turn
            or turn.get("case") != expected_case
            or turn.get("prompt") != _PRODUCT_ACCEPTANCE_PROMPTS[expected_turn - 1]
            or turn.get("observed_user_prompt")
            != _PRODUCT_ACCEPTANCE_PROMPTS[expected_turn - 1]
            or type(turn.get("new_user_count")) is not int
            or turn["new_user_count"] != 1
            or turn.get("passed") is not True
            or "error" in turn
            or turn.get("generation_mode") != expected_generation_mode
            or turn.get("expected_route") != expected_route
            or turn.get("session_id") != expected_session_id
            or turn.get("peer_session_id") != expected_peer_session_id
            or not normalized_answer
            or len(answer) > _PRODUCT_MAX_ANSWER_CHARS
            or _PRODUCT_HANGUL.search(answer) is None
            or turn.get("answer_sha256") != answer_digest
            or answer_digest in answer_digests
            or (
                expected_turn == 1
                and (
                    _PRODUCT_MODEL_LABEL not in normalized_answer
                    or str(_PRODUCT_STORED_PARAMETERS) not in digits_answer
                    or str(_PRODUCT_EFFECTIVE_PARAMETERS) not in digits_answer
                    or "70억" in normalized_answer
                    or "7 billion" in normalized_answer
                )
            )
            or type(turn.get("generated_tokens")) is not int
            or (
                turn["generated_tokens"] != 0
                if expected_route == "grounded"
                else turn["generated_tokens"] <= 0
            )
            or type(turn.get("continuations")) is not int
            or turn["continuations"] != expected_continuations
            or type(turn.get("new_assistant_count")) is not int
            or turn["new_assistant_count"] != 1
            or type(elapsed) not in {int, float}
            or not math.isfinite(float(elapsed))
            or not 0 <= float(elapsed) <= _PRODUCT_MAX_TURN_SECONDS
            or turn.get("explicit_truncation") is not False
            or turn.get("empty_answer") is not False
            or turn.get("finish_reason") != "stop"
            or not isinstance(checks, dict)
            or set(checks) != _PRODUCT_REQUIRED_TURN_CHECKS
            or any(
                type(value) is not bool or value is not True
                for value in checks.values()
            )
            or any(
                checks.get(name) is not expected
                for name, expected in independently_recomputed_checks.items()
            )
            or not isinstance(repetition, dict)
            or repetition != recomputed_repetition
            or recomputed_repetition["sentence_count"] <= 0
            or recomputed_repetition["duplicate_sentence_count"] != 0
            or turn.get("role_token_leaks") != recomputed_role_leaks
            or recomputed_role_leaks
            or turn.get("control_marker_leaks") != recomputed_control_leaks
            or recomputed_control_leaks
            or turn.get("cross_turn_exact_duplicate") is not False
            or turn.get("cross_turn_sentence_reuse") != sorted(reused_sentence_keys)[:8]
            or cross_turn_sentence_echo
            or not isinstance(korean, dict)
            or korean != recomputed_korean
            or recomputed_korean["contains_korean"] is not True
            or recomputed_korean["complete"] is not True
            or not isinstance(worker, dict)
            or worker.get("healthy") is not True
            or worker.get("pid_stable") is not True
            or worker.get("active_request_id") is not None
            or worker.get("expected_running") is not expected_worker_running
        ):
            raise GPU5BoundaryError(
                "product acceptance turn does not satisfy its exact passed contract"
            )
        if (
            not isinstance(isolation, dict)
            or isolation.get("peer_unchanged") is not True
            or not isinstance(isolation.get("peer_conversation_before_sha256"), str)
            or not _SHA256_DIGEST.fullmatch(
                isolation["peer_conversation_before_sha256"]
            )
            or isolation.get("peer_conversation_after_sha256")
            != isolation["peer_conversation_before_sha256"]
        ):
            raise GPU5BoundaryError("product acceptance session isolation is invalid")
        answer_digests.add(answer_digest)
        prior_sentence_keys.update(current_sentence_keys)
        if not expected_worker_running:
            if (
                worker.get("running") is not False
                or worker.get("pid") is not None
                or not isinstance(worker.get("memory"), dict)
                or worker["memory"].get("sample_scope") != _PRODUCT_MEMORY_SAMPLE_SCOPE
                or worker["memory"].get("captures_peak") is not False
                or worker["memory"].get("spot_sample_observed") is not False
                or worker["memory"].get("gpu_memory_spot_sample_bytes") is not None
                or worker["memory"].get("gpu_memory_spot_sample_status")
                != "worker_not_started"
                or worker["memory"].get("gpu_memory_spot_sample_threshold_bytes")
                != _PRODUCT_GPU_SPOT_SAMPLE_LIMIT_BYTES
                or worker["memory"].get("gpu_memory_spot_sample_within_threshold")
                is not None
            ):
                raise GPU5BoundaryError(
                    "product acceptance pre-worker turn state is invalid"
                )
            continue
        expected_worker_turns += 1
        pid = worker.get("pid")
        memory = worker.get("memory")
        if (
            worker.get("running") is not True
            or type(pid) is not int
            or pid <= 0
            or not isinstance(memory, dict)
            or memory.get("sample_scope") != _PRODUCT_MEMORY_SAMPLE_SCOPE
            or memory.get("captures_peak") is not False
            or memory.get("spot_sample_observed") is not True
            or memory.get("gpu_memory_spot_sample_status") != "measured_aggregate"
            or type(memory.get("gpu_memory_spot_sample_bytes")) is not int
            or memory["gpu_memory_spot_sample_bytes"] < 0
            or memory["gpu_memory_spot_sample_bytes"]
            > _PRODUCT_GPU_SPOT_SAMPLE_LIMIT_BYTES
            or memory.get("gpu_memory_spot_sample_threshold_bytes")
            != _PRODUCT_GPU_SPOT_SAMPLE_LIMIT_BYTES
            or memory.get("gpu_memory_spot_sample_within_threshold") is not True
        ):
            raise GPU5BoundaryError(
                "product acceptance resident worker or GPU spot sample is invalid"
            )
        resident_pids.add(pid)
        observed_gpu_samples += 1
        maximum_gpu_sample = max(
            maximum_gpu_sample,
            memory["gpu_memory_spot_sample_bytes"],
        )

    expected_summary = {
        "requested_turns": 20,
        "completed_turns": 20,
        "passed_turns": 20,
        "failed_turns": 0,
        "turn_success_rate": 1.0,
        "quality_fallback_turns": 0,
        "allowed_quality_fallback_turns": 0,
        "quality_fallback_gate_passed": True,
        "content_answer_rate": 1.0,
        "failed_check_counts": {},
        "worker_expected_turns": expected_worker_turns,
        "resident_worker_pids": sorted(resident_pids),
        "single_resident_worker_scope": expected_worker_turns > 0
        and len(resident_pids) == 1,
        "post_turn_gpu_memory_spot_sample_observed_turns": observed_gpu_samples,
        "post_turn_gpu_memory_spot_sample_coverage_rate": 1.0,
        "maximum_observed_post_turn_gpu_memory_spot_sample_bytes": (
            maximum_gpu_sample or None
        ),
        "post_turn_gpu_memory_spot_samples_over_threshold": 0,
        "post_turn_gpu_memory_spot_sample_coverage_verdict": "complete",
        "post_turn_gpu_memory_spot_sample_threshold_observation": (
            "observed_at_or_below_threshold"
        ),
        "post_turn_gpu_memory_spot_sample_coverage_complete": True,
        "post_turn_gpu_memory_spot_sample_threshold_gate_passed": True,
        "recommended_stress_schedule_completed": True,
        "strict_completion_stress_gate_passed": True,
    }
    if expected_worker_turns <= 0 or len(resident_pids) != 1:
        raise GPU5BoundaryError("product acceptance did not prove one resident worker")
    if any(summary.get(key) != value for key, value in expected_summary.items()):
        raise GPU5BoundaryError(
            "product acceptance summary does not match independently recomputed turns"
        )
    return str(payload["schema"])


def _reject_non_finite_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant: {value}")


def _reject_duplicate_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key: {key}")
        result[key] = value
    return result


def _decode_product_acceptance_json(encoded: bytes) -> dict[str, Any]:
    if not isinstance(encoded, bytes) or not encoded:
        raise GPU5BoundaryError("product acceptance evidence bytes are invalid")
    try:
        payload = json.loads(
            encoded.decode("utf-8", errors="strict"),
            parse_constant=_reject_non_finite_json_constant,
            object_pairs_hook=_reject_duplicate_json_object,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise GPU5BoundaryError(
            "product acceptance evidence must be one strict UTF-8 JSON value"
        ) from error
    if not isinstance(payload, dict):
        raise GPU5BoundaryError("product acceptance evidence must be a JSON object")
    return payload


def _validate_gpu5_scheduler_reservation_payload(
    payload: Any,
    *,
    expected_source_commit: str,
    effective_uid: int,
    now_ns: int,
    minimum_remaining_ns: int = 0,
) -> dict[str, Any]:
    """Validate a scheduler-issued, exact-physical-GPU5 reservation claim."""

    required_keys = {
        "schema",
        "status",
        "physical_gpu_index",
        "gpu_uuid",
        "source_commit",
        "subject_uid",
        "reservation_id",
        "issued_unix_ns",
        "expires_unix_ns",
    }
    if not isinstance(payload, dict) or set(payload) != required_keys:
        raise GPU5BoundaryError(
            "GPU5 scheduler reservation must use the exact closed schema"
        )
    source_commit = _validated_expected_source_commit(expected_source_commit)
    if (
        isinstance(effective_uid, bool)
        or not isinstance(effective_uid, int)
        or effective_uid < 0
        or isinstance(now_ns, bool)
        or not isinstance(now_ns, int)
        or now_ns <= 0
        or isinstance(minimum_remaining_ns, bool)
        or not isinstance(minimum_remaining_ns, int)
        or minimum_remaining_ns < 0
    ):
        raise GPU5BoundaryError("GPU5 scheduler reservation verifier state is invalid")
    issued = payload.get("issued_unix_ns")
    expires = payload.get("expires_unix_ns")
    if (
        payload.get("schema") != GPU5_SCHEDULER_RESERVATION_SCHEMA
        or payload.get("status") != "reserved"
        or payload.get("physical_gpu_index") != PROJECT_PHYSICAL_GPU_INDEX
        or payload.get("gpu_uuid") != PROJECT_GPU_UUID
        or payload.get("source_commit") != source_commit
        or payload.get("subject_uid") != effective_uid
        or not isinstance(payload.get("reservation_id"), str)
        or _GPU5_SCHEDULER_RESERVATION_ID.fullmatch(payload["reservation_id"]) is None
        or isinstance(issued, bool)
        or not isinstance(issued, int)
        or isinstance(expires, bool)
        or not isinstance(expires, int)
        or issued <= 0
        or issued > now_ns
        or expires <= now_ns
        or expires - now_ns < minimum_remaining_ns
        or expires <= issued
        or expires - issued > MAX_GPU5_SCHEDULER_RESERVATION_WINDOW_NS
    ):
        raise GPU5BoundaryError(
            "GPU5 scheduler reservation is missing, stale, mismatched, or overbroad"
        )
    return dict(payload)


def _require_external_gpu5_scheduler_reservation(
    expected_source_commit: str,
    *,
    required_run_seconds: float,
    path: Path | None = None,
) -> dict[str, Any]:
    """Fail closed unless the lab scheduler has reserved exact GPU5 for this run.

    A private process-local lock cannot exclude unrelated laboratory jobs. The
    scheduler artifact is therefore required before Docker or any GPU query.
    """

    if os.name != "posix":
        raise GPU5BoundaryError(
            "GPU5 Stage G requires the Linux lab scheduler reservation gate"
        )
    selected = GPU5_SCHEDULER_RESERVATION_PATH if path is None else path
    if (
        isinstance(required_run_seconds, bool)
        or not isinstance(required_run_seconds, (int, float))
        or not math.isfinite(float(required_run_seconds))
        or float(required_run_seconds) <= 0
    ):
        raise GPU5BoundaryError("GPU5 scheduler reservation run duration is invalid")
    if not isinstance(selected, Path) or not selected.is_absolute():
        raise GPU5BoundaryError("GPU5 scheduler reservation path must be absolute")
    selected = Path(os.path.normpath(os.fspath(selected)))
    _trusted_directory_chain(
        selected.parent,
        label="GPU5 scheduler reservation directory",
    )
    scheduler_directory = selected.parent.lstat()
    if (
        scheduler_directory.st_uid != GPU5_SCHEDULER_RESERVATION_OWNER_UID
        or _group_or_world_writable(scheduler_directory.st_mode)
    ):
        raise GPU5BoundaryError(
            "GPU5 scheduler reservation directory must remain root-owned and sealed"
        )
    try:
        unresolved = selected.lstat()
        resolved = selected.resolve(strict=True)
    except OSError as error:
        raise GPU5BoundaryError(
            "GPU5 Stage G is an external blocker until the lab scheduler reserves GPU5"
        ) from error
    if stat.S_ISLNK(unresolved.st_mode) or resolved != selected:
        raise GPU5BoundaryError(
            "GPU5 scheduler reservation must be a non-symlink canonical file"
        )
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    no_follow = getattr(os, "O_NOFOLLOW", None)
    if no_follow is None:
        raise GPU5BoundaryError("GPU5 scheduler reservation requires O_NOFOLLOW")
    flags |= no_follow
    descriptor = -1
    try:
        descriptor = os.open(selected, flags)
        before = os.fstat(descriptor)
        mode = stat.S_IMODE(before.st_mode)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_uid != GPU5_SCHEDULER_RESERVATION_OWNER_UID
            or before.st_nlink != 1
            or mode not in {0o400, 0o440, 0o444}
            or not 0 < before.st_size <= MAX_GPU5_SCHEDULER_RESERVATION_BYTES
        ):
            raise GPU5BoundaryError(
                "GPU5 scheduler reservation ownership or mode is unsafe"
            )
        encoded = b""
        while len(encoded) <= MAX_GPU5_SCHEDULER_RESERVATION_BYTES:
            chunk = os.read(
                descriptor,
                min(
                    4_096,
                    MAX_GPU5_SCHEDULER_RESERVATION_BYTES + 1 - len(encoded),
                ),
            )
            if not chunk:
                break
            encoded += chunk
        after = os.fstat(descriptor)
    except OSError as error:
        raise GPU5BoundaryError(
            "GPU5 scheduler reservation could not be read safely"
        ) from error
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    if (
        len(encoded) != before.st_size
        or len(encoded) > MAX_GPU5_SCHEDULER_RESERVATION_BYTES
        or (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
        )
        != (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
        )
    ):
        raise GPU5BoundaryError(
            "GPU5 scheduler reservation changed while it was being verified"
        )
    try:
        payload = json.loads(
            encoded.decode("utf-8", errors="strict"),
            parse_constant=_reject_non_finite_json_constant,
            object_pairs_hook=_reject_duplicate_json_object,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise GPU5BoundaryError(
            "GPU5 scheduler reservation must be one strict UTF-8 JSON value"
        ) from error
    validated = _validate_gpu5_scheduler_reservation_payload(
        payload,
        expected_source_commit=expected_source_commit,
        effective_uid=_effective_uid(),
        now_ns=time_ns(),
        minimum_remaining_ns=(
            math.ceil(float(required_run_seconds) * 1_000_000_000)
            + GPU5_SCHEDULER_CLEANUP_GRACE_NS
        ),
    )
    return {
        "schema": validated["schema"],
        "reservation_id": validated["reservation_id"],
        "physical_gpu_index": validated["physical_gpu_index"],
        "gpu_uuid": validated["gpu_uuid"],
        "source_commit": validated["source_commit"],
        "subject_uid": validated["subject_uid"],
        "issued_unix_ns": validated["issued_unix_ns"],
        "expires_unix_ns": validated["expires_unix_ns"],
        "artifact_sha256": hashlib.sha256(encoded).hexdigest(),
    }


def _validate_product_acceptance_evidence(
    encoded: bytes,
    *,
    expected_model_manifest_sha256: str | None = None,
) -> str:
    """Require one standalone passed JSON component for product Stage G."""

    if not encoded or len(encoded) > MAX_EVIDENCE_BYTES:
        raise GPU5BoundaryError("product acceptance evidence size is invalid")
    payload = _decode_product_acceptance_json(encoded)
    return _validate_product_acceptance_payload(
        payload,
        expected_model_manifest_sha256=expected_model_manifest_sha256,
    )


def _docker_cli_prefix() -> tuple[str, ...]:
    config_root = _ensure_private_directory(DOCKER_CONFIG_ROOT, create=False)
    if tuple(config_root.iterdir()):
        raise GPU5BoundaryError("Docker config root must remain exactly empty")
    return (
        _validated_executable(DOCKER_EXECUTABLE),
        "--host",
        _validated_docker_socket(),
        "--config",
        str(config_root),
    )


def build_gpu5_docker_argv(
    image: str,
    command: Sequence[str],
    *,
    expected_source_commit: str,
    source_snapshot: SourceSnapshot,
    model_snapshot: ModelSnapshot,
    environment: Mapping[str, str] | None = None,
    mounts: Sequence[str] = (),
    workdir: str | None = None,
    container_name: str | None = None,
    launch_nonce: str | None = None,
    release_profile: bool = False,
    validation_artifact_profile: str = BASE_CANARY_ARTIFACT_PROFILE,
) -> tuple[str, ...]:
    """Build the only accepted offline Docker launch shape for this project."""

    if image != PINNED_DOCKER_IMAGE:
        raise GPU5BoundaryError("Docker image must match the pinned immutable digest")
    source_commit = _validated_expected_source_commit(expected_source_commit)
    nonce = _launch_nonce(launch_nonce)
    artifact_profile = _validation_artifact_profile(validation_artifact_profile)
    _validated_docker_snapshot_capabilities(
        source_commit,
        nonce,
        source_snapshot=source_snapshot,
        model_snapshot=model_snapshot,
    )
    docker_prefix = _docker_cli_prefix()
    if not command or any(
        not isinstance(token, str) or not token or "\x00" in token for token in command
    ):
        raise GPU5BoundaryError("a bounded non-empty container command is required")
    validated_command = (
        _validated_release_validator_command(command, artifact_profile.name)
        if release_profile
        else _validated_validator_command(command, artifact_profile.name)
    )
    if len(mounts) != 2:
        raise GPU5BoundaryError(
            "Docker requires exactly the sealed source/model snapshot mounts"
        )
    runtime_mounts = _runtime_mount_map(
        expected_source_commit=source_commit,
        launch_nonce=nonce,
        source_snapshot=source_snapshot,
        model_snapshot=model_snapshot,
        validation_artifact_profile=artifact_profile.name,
    )
    validated_mounts = tuple(
        _validated_read_only_mount(mount, expected_mounts=runtime_mounts)
        for mount in mounts
    )
    expected_mounts = tuple(
        f"{source.resolve(strict=True)}:{destination}:ro"
        for source, destination in runtime_mounts.items()
    )
    if len(set(validated_mounts)) != 2 or set(validated_mounts) != set(expected_mounts):
        raise GPU5BoundaryError(
            "Docker source/model snapshot mounts must each appear exactly once"
        )
    if workdir != _PINNED_WORKDIR:
        raise GPU5BoundaryError("Docker workdir must be exactly /workspace")
    name = _container_name(container_name)
    execution_profile = "release" if release_profile else "inspection"
    argv: list[str] = [
        *docker_prefix,
        "run",
        "--rm",
        "--name",
        name,
        "--gpus",
        f"device={PROJECT_GPU_UUID}",
        "--network",
        "none",
        "--log-driver",
        "none",
        "--pull=never",
        "--user",
        _PINNED_CONTAINER_USER,
        "--cap-drop",
        "ALL",
        "--security-opt",
        "no-new-privileges:true",
        "--read-only",
        "--tmpfs",
        _PINNED_CONTAINER_TMPFS,
        "--pids-limit",
        _PINNED_CONTAINER_PIDS_LIMIT,
        "--entrypoint",
        _PINNED_CONTAINER_PYTHON,
        "--label",
        f"{_GUARD_LABEL}=gpu5",
        "--label",
        f"{_SOURCE_COMMIT_LABEL}={source_commit}",
        "--label",
        f"{_LAUNCH_NONCE_LABEL}={nonce}",
        "--label",
        f"{_EXECUTION_PROFILE_LABEL}={execution_profile}",
        "--label",
        f"{_VALIDATION_ARTIFACT_PROFILE_LABEL}={artifact_profile.name}",
    ]
    argv.extend(_validated_environment(environment))
    for mount in expected_mounts:
        argv.extend(("--volume", mount))
    argv.extend(("--workdir", _PINNED_WORKDIR))
    argv.extend(("--", image, *validated_command))
    built = tuple(argv)
    validate_gpu5_docker_argv(
        built,
        source_snapshot=source_snapshot,
        model_snapshot=model_snapshot,
    )
    return built


def validate_gpu5_docker_argv(
    argv: Sequence[str],
    *,
    source_snapshot: SourceSnapshot,
    model_snapshot: ModelSnapshot,
) -> None:
    """Reject every Docker launch except the exact production GPU5 contract."""

    values = tuple(argv)
    source_commit_values = [
        value.removeprefix(f"{_SOURCE_COMMIT_LABEL}=")
        for value in values
        if value.startswith(f"{_SOURCE_COMMIT_LABEL}=")
    ]
    if len(source_commit_values) != 1:
        raise GPU5BoundaryError("Docker source commit label is missing or duplicated")
    source_commit = _validated_expected_source_commit(source_commit_values[0])
    launch_nonce_values = [
        value.removeprefix(f"{_LAUNCH_NONCE_LABEL}=")
        for value in values
        if value.startswith(f"{_LAUNCH_NONCE_LABEL}=")
    ]
    if len(launch_nonce_values) != 1:
        raise GPU5BoundaryError("Docker launch nonce label is missing or duplicated")
    nonce = _launch_nonce(launch_nonce_values[0])
    execution_profile_values = [
        value.removeprefix(f"{_EXECUTION_PROFILE_LABEL}=")
        for value in values
        if value.startswith(f"{_EXECUTION_PROFILE_LABEL}=")
    ]
    if len(execution_profile_values) != 1 or execution_profile_values[0] not in {
        "release",
        "inspection",
    }:
        raise GPU5BoundaryError("Docker execution profile is missing or invalid")
    execution_profile = execution_profile_values[0]
    artifact_profile_values = [
        value.removeprefix(f"{_VALIDATION_ARTIFACT_PROFILE_LABEL}=")
        for value in values
        if value.startswith(f"{_VALIDATION_ARTIFACT_PROFILE_LABEL}=")
    ]
    if len(artifact_profile_values) != 1:
        raise GPU5BoundaryError(
            "Docker validation artifact profile is missing or duplicated"
        )
    artifact_profile = _validation_artifact_profile(artifact_profile_values[0])
    expected_prefix = (*_docker_cli_prefix(), "run")
    if values[: len(expected_prefix)] != expected_prefix or values.count("--") != 1:
        raise GPU5BoundaryError("unrecognized Docker launch contract")
    boundary = values.index("--")
    if boundary + 2 >= len(values):
        raise GPU5BoundaryError("Docker image and command are required")
    options = values[len(expected_prefix) : boundary]
    if options.count("--name") != 1:
        raise GPU5BoundaryError("Docker must have one unique project container name")
    name_position = options.index("--name")
    if name_position + 1 >= len(options):
        raise GPU5BoundaryError("Docker container name is missing")
    name = _container_name(options[name_position + 1])
    runtime_mounts = _runtime_mount_map(
        expected_source_commit=source_commit,
        launch_nonce=nonce,
        source_snapshot=source_snapshot,
        model_snapshot=model_snapshot,
        validation_artifact_profile=artifact_profile.name,
    )
    expected_mounts = tuple(
        f"{source.resolve(strict=True)}:{destination}:ro"
        for source, destination in runtime_mounts.items()
    )
    expected_options = (
        "--rm",
        "--name",
        name,
        "--gpus",
        f"device={PROJECT_GPU_UUID}",
        "--network",
        "none",
        "--log-driver",
        "none",
        "--pull=never",
        "--user",
        _PINNED_CONTAINER_USER,
        "--cap-drop",
        "ALL",
        "--security-opt",
        "no-new-privileges:true",
        "--read-only",
        "--tmpfs",
        _PINNED_CONTAINER_TMPFS,
        "--pids-limit",
        _PINNED_CONTAINER_PIDS_LIMIT,
        "--entrypoint",
        _PINNED_CONTAINER_PYTHON,
        "--label",
        f"{_GUARD_LABEL}=gpu5",
        "--label",
        f"{_SOURCE_COMMIT_LABEL}={source_commit}",
        "--label",
        f"{_LAUNCH_NONCE_LABEL}={nonce}",
        "--label",
        f"{_EXECUTION_PROFILE_LABEL}={execution_profile}",
        "--label",
        f"{_VALIDATION_ARTIFACT_PROFILE_LABEL}={artifact_profile.name}",
        *_validated_environment(_REQUIRED_CONTAINER_ENVIRONMENT),
        "--volume",
        expected_mounts[0],
        "--volume",
        expected_mounts[1],
        "--workdir",
        _PINNED_WORKDIR,
    )
    if options != expected_options:
        raise GPU5BoundaryError(
            "Docker options differ from the exact offline GPU5 production contract"
        )
    if values[boundary + 1] != PINNED_DOCKER_IMAGE:
        raise GPU5BoundaryError(
            "Docker image digest is not the pinned production image"
        )
    if execution_profile == "release":
        _validated_release_validator_command(
            values[boundary + 2 :], artifact_profile.name
        )
    else:
        _validated_validator_command(values[boundary + 2 :], artifact_profile.name)


def _docker_control(
    runner: Any,
    argv: Sequence[str],
    *,
    timeout: float,
) -> int:
    completed = runner(
        list(argv),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        timeout=timeout,
        check=False,
        env=_minimal_host_environment(),
    )
    return int(completed.returncode)


def _prove_exact_container_name_absent(name: str, *, runner: Any) -> dict[str, Any]:
    """Prove that the exact generated name is absent without mutating Docker."""

    validated_name = _container_name(name)
    argv = (
        *_docker_cli_prefix(),
        "ps",
        "--all",
        "--filter",
        f"name=^/{validated_name}$",
        "--format",
        "{{.Names}}",
    )
    try:
        completed = runner(
            list(argv),
            capture_output=True,
            text=True,
            timeout=5.0,
            check=False,
            env=_minimal_host_environment(),
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise GPU5BoundaryError(
            f"exact container-name absence proof failed: {type(error).__name__}"
        ) from error
    if (
        not isinstance(completed.stdout, str)
        or not isinstance(completed.stderr, str)
        or len(completed.stdout) > 4_096
        or len(completed.stderr) > 4_096
    ):
        raise GPU5BoundaryError("container-name absence output is invalid")
    if completed.returncode != 0 or completed.stderr.strip():
        raise GPU5BoundaryError("Docker daemon could not prove container-name absence")
    names = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
    if names:
        raise GPU5BoundaryError(
            "generated GPU5 container name is already occupied; no cleanup attempted"
        )
    return {
        "action": "exact_name_ps_absence",
        "returncode": int(completed.returncode),
        "container_absent": True,
    }


def _inspect_owned_container(
    name: str,
    *,
    selector: str,
    expected_source_commit: str,
    launch_nonce: str,
    validation_artifact_profile: str = BASE_CANARY_ARTIFACT_PROFILE,
    runner: Any,
) -> tuple[_OwnedContainer | None, list[dict[str, Any]]]:
    """Return only a container bearing this launch's immutable exact labels."""

    validated_name = _container_name(name)
    if selector != validated_name and _CONTAINER_ID.fullmatch(selector) is None:
        raise GPU5BoundaryError("invalid Docker cleanup selector")
    expected_labels = _expected_launch_labels(
        expected_source_commit=expected_source_commit,
        launch_nonce=launch_nonce,
        validation_artifact_profile=validation_artifact_profile,
    )
    argv = (
        *_docker_cli_prefix(),
        "inspect",
        "--type",
        "container",
        "--format",
        "{{.Id}}\t{{.Name}}\t{{json .Config.Labels}}",
        selector,
    )
    try:
        completed = runner(
            list(argv),
            capture_output=True,
            text=True,
            timeout=5.0,
            check=False,
            env=_minimal_host_environment(),
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise GPU5BoundaryError(
            f"bounded Docker label inspection failed: {type(error).__name__}"
        ) from error
    if (
        not isinstance(completed.stdout, str)
        or not isinstance(completed.stderr, str)
        or len(completed.stdout) > 8_192
        or len(completed.stderr) > 8_192
    ):
        raise GPU5BoundaryError("Docker label inspection output is invalid")
    actions: list[dict[str, Any]] = [
        {
            "action": "inspect_immutable_labels",
            "selector": selector,
            "returncode": int(completed.returncode),
        }
    ]
    if completed.returncode != 0:
        # Docker inspect uses a non-zero exit both for absence and daemon errors.
        # The exact-name ps proof distinguishes the safe absence case.
        proof = _prove_exact_container_name_absent(validated_name, runner=runner)
        actions.append(proof)
        return None, actions
    if completed.stderr.strip():
        raise GPU5BoundaryError("Docker label inspection emitted unexpected stderr")
    lines = [line for line in completed.stdout.splitlines() if line.strip()]
    if len(lines) != 1:
        raise GPU5BoundaryError("Docker label inspection must return exactly one row")
    fields = lines[0].split("\t", 2)
    if len(fields) != 3:
        raise GPU5BoundaryError("Docker label inspection row is malformed")
    container_id, observed_name, labels_raw = fields
    if _CONTAINER_ID.fullmatch(container_id) is None:
        raise GPU5BoundaryError("Docker label inspection returned an invalid ID")
    if selector != validated_name and container_id != selector:
        raise GPU5BoundaryError("Docker cleanup container identity changed")
    if observed_name != f"/{validated_name}":
        raise GPU5BoundaryError("Docker cleanup container name changed")
    try:
        labels = json.loads(labels_raw)
    except (json.JSONDecodeError, TypeError) as error:
        raise GPU5BoundaryError("Docker cleanup labels are malformed") from error
    if not isinstance(labels, dict) or any(
        not isinstance(key, str) or not isinstance(value, str)
        for key, value in labels.items()
    ):
        raise GPU5BoundaryError("Docker cleanup labels are not a string map")
    if any(labels.get(key) != value for key, value in expected_labels.items()):
        raise GPU5BoundaryError(
            "container has foreign or missing immutable launch labels; no cleanup attempted"
        )
    actions[0]["container_id"] = container_id
    return (
        _OwnedContainer(
            container_id=container_id,
            name=validated_name,
            labels=dict(labels),
        ),
        actions,
    )


def _ensure_container_absent(
    name: str,
    *,
    expected_source_commit: str,
    launch_nonce: str,
    validation_artifact_profile: str = BASE_CANARY_ARTIFACT_PROFILE,
    runner: Any,
) -> dict[str, Any]:
    """Remove only this exact labelled launch, then prove exact-name absence."""

    actions: list[dict[str, Any]] = []
    caught_errors: list[BaseException] = []
    try:
        owned, inspection = _inspect_owned_container(
            name,
            selector=name,
            expected_source_commit=expected_source_commit,
            launch_nonce=launch_nonce,
            validation_artifact_profile=validation_artifact_profile,
            runner=runner,
        )
        actions.extend(inspection)
    except BaseException as error:
        caught_errors.append(error)
        cleanup_errors = [
            {"action": "inspect_owned_name", "error": type(error).__name__}
        ]
        actions.append({"action": "inspect_owned_name", "error": type(error).__name__})
        try:
            proof = _prove_exact_container_name_absent(name, runner=runner)
            actions.append(proof)
            absent = True
        except BaseException as proof_error:
            caught_errors.append(proof_error)
            actions.append(
                {"action": "final_absence_proof", "error": type(proof_error).__name__}
            )
            cleanup_errors.append(
                {
                    "action": "final_absence_proof",
                    "error": type(proof_error).__name__,
                }
            )
            absent = False
        _raise_cleanup_failures(
            "container ownership could not be established; no mutation attempted",
            {
                "container_name": name,
                "container_absent": absent,
                "actions": actions,
                "errors": cleanup_errors,
            },
            caught_errors,
        )
    if owned is None:
        return {"container_name": name, "container_absent": True, "actions": actions}

    docker_prefix = _docker_cli_prefix()
    cleanup_errors: list[dict[str, str]] = []
    try:
        returncode = _docker_control(
            runner,
            (*docker_prefix, "stop", "--time", "5", owned.container_id),
            timeout=10.0,
        )
        actions.append({"action": "stop_owned_id", "returncode": returncode})
        if returncode != 0:
            cleanup_errors.append(
                {"action": "stop_owned_id", "error": f"returncode:{returncode}"}
            )
    except BaseException as error:
        caught_errors.append(error)
        actions.append({"action": "stop_owned_id", "error": type(error).__name__})
        cleanup_errors.append(
            {"action": "stop_owned_id", "error": type(error).__name__}
        )

    # Re-inspect the immutable ID and labels immediately before rm. A name can
    # be recycled after stop; this guard will never remove that replacement.
    owned_after_stop = None
    try:
        owned_after_stop, inspection = _inspect_owned_container(
            name,
            selector=owned.container_id,
            expected_source_commit=expected_source_commit,
            launch_nonce=launch_nonce,
            validation_artifact_profile=validation_artifact_profile,
            runner=runner,
        )
        actions.extend(inspection)
    except BaseException as error:
        caught_errors.append(error)
        actions.append({"action": "reinspect_owned_id", "error": type(error).__name__})
        cleanup_errors.append(
            {"action": "reinspect_owned_id", "error": type(error).__name__}
        )
    if owned_after_stop is not None:
        try:
            returncode = _docker_control(
                runner,
                (*docker_prefix, "rm", "--force", owned_after_stop.container_id),
                timeout=10.0,
            )
            actions.append({"action": "rm_owned_id", "returncode": returncode})
            if returncode != 0:
                cleanup_errors.append(
                    {"action": "rm_owned_id", "error": f"returncode:{returncode}"}
                )
        except BaseException as error:
            caught_errors.append(error)
            actions.append({"action": "rm_owned_id", "error": type(error).__name__})
            cleanup_errors.append(
                {"action": "rm_owned_id", "error": type(error).__name__}
            )
    try:
        proof = _prove_exact_container_name_absent(name, runner=runner)
        actions.append(proof)
    except BaseException as error:
        caught_errors.append(error)
        actions.append({"action": "final_absence_proof", "error": type(error).__name__})
        cleanup_errors.append(
            {"action": "final_absence_proof", "error": type(error).__name__}
        )
    payload = {
        "container_name": name,
        "container_absent": not any(
            item["action"] == "final_absence_proof" for item in cleanup_errors
        ),
        "actions": actions,
        "errors": cleanup_errors,
    }
    if cleanup_errors:
        _raise_cleanup_failures(
            "GPU5 cleanup completed with errors", payload, caught_errors
        )
    return payload


def run_guarded_gpu5_container(
    image: str,
    command: Sequence[str],
    *,
    expected_source_commit: str,
    environment: Mapping[str, str] | None = None,
    mounts: Sequence[str] = (),
    workdir: str | None = None,
    run_timeout_seconds: float,
    evidence_filename: str,
    validation_artifact_profile: str = BASE_CANARY_ARTIFACT_PROFILE,
    smi_runner: Any = subprocess.run,
    docker_runner: Any = subprocess.run,
    cleanup_runner: Any = subprocess.run,
    scope_reader: Any = capture_execution_scope,
    snapshot_factory: Any = prepare_source_snapshot,
    model_snapshot_factory: Any = _prepare_docker_model_snapshot,
) -> GuardedDockerResult:
    """Preflight, bounded run, forced cleanup, then require idle postflight."""

    if (
        isinstance(run_timeout_seconds, bool)
        or not 1.0 <= float(run_timeout_seconds) <= 86_400.0
    ):
        raise GPU5BoundaryError("Docker timeout must be in [1, 86400] seconds")
    if resource is None:
        raise GPU5BoundaryError("bounded evidence capture requires Linux RLIMIT_FSIZE")
    source_commit = _validated_expected_source_commit(expected_source_commit)
    completed: Any | None = None
    execution_error: BaseException | None = None
    cleanup_evidence: dict[str, Any] | None = None
    cleanup_error: BaseException | None = None
    postflight: GPU5Snapshot | None = None
    postflight_error: BaseException | None = None
    evidence_bytes: int | None = None
    evidence_sha256: str | None = None
    evidence_error: BaseException | None = None
    scope_after: ExecutionScope | None = None
    scope_error: BaseException | None = None
    evidence_path = ""
    evidence_component_schema: str | None = None
    nonce = _launch_nonce()
    name = _container_name(f"cognios-gpu5-{nonce[:12]}")
    artifact_profile = _validation_artifact_profile(validation_artifact_profile)

    with _gpu5_project_lease(source_commit) as lease:
        source_snapshot = snapshot_factory(source_commit, nonce)
        if not isinstance(source_snapshot, SourceSnapshot):
            raise GPU5BoundaryError(
                "source snapshot factory returned an invalid snapshot"
            )
        if artifact_profile.name == BASE_CANARY_ARTIFACT_PROFILE:
            model_snapshot = model_snapshot_factory(
                source_snapshot,
                launch_nonce=nonce,
            )
        else:
            model_snapshot = model_snapshot_factory(
                source_snapshot,
                launch_nonce=nonce,
                validation_artifact_profile=artifact_profile.name,
            )
        if not isinstance(model_snapshot, ModelSnapshot):
            raise GPU5BoundaryError(
                "model snapshot factory returned an invalid snapshot"
            )
        runtime_mounts = _runtime_mount_map(
            expected_source_commit=source_commit,
            launch_nonce=nonce,
            source_snapshot=source_snapshot,
            model_snapshot=model_snapshot,
            validation_artifact_profile=artifact_profile.name,
        )
        exact_mounts = tuple(
            f"{source.resolve(strict=True)}:{destination}:ro"
            for source, destination in runtime_mounts.items()
        )
        if mounts and set(mounts) != set(exact_mounts):
            raise GPU5BoundaryError(
                "guarded execution mounts must equal the internally generated snapshot/model mounts"
            )
        argv = build_gpu5_docker_argv(
            image,
            command,
            expected_source_commit=source_commit,
            source_snapshot=source_snapshot,
            model_snapshot=model_snapshot,
            environment=environment,
            mounts=exact_mounts,
            workdir=workdir,
            container_name=name,
            launch_nonce=nonce,
            release_profile=True,
            validation_artifact_profile=artifact_profile.name,
        )
        with _open_evidence_target(evidence_filename) as evidence_handle:
            evidence_path = str(evidence_handle.root_path / evidence_filename)
            _prove_exact_container_name_absent(name, runner=cleanup_runner)
            scope_arguments: dict[str, Any] = {
                "source_snapshot": source_snapshot,
                "model_snapshot": model_snapshot,
            }
            if artifact_profile.name != BASE_CANARY_ARTIFACT_PROFILE:
                scope_arguments["validation_artifact_profile"] = artifact_profile.name
            scope_before = scope_reader(source_commit, **scope_arguments)
            if not isinstance(scope_before, ExecutionScope):
                raise GPU5BoundaryError(
                    "execution scope reader returned an invalid snapshot"
                )
            preflight = preflight_gpu5(runner=smi_runner)
            try:
                with os.fdopen(
                    os.dup(evidence_handle.file_fd), "wb", buffering=0
                ) as evidence_stream:
                    lease.mark_launch_attempted()
                    completed = docker_runner(
                        list(argv),
                        stdout=evidence_stream,
                        stderr=subprocess.STDOUT,
                        timeout=float(run_timeout_seconds),
                        check=False,
                        preexec_fn=_bound_evidence_file_size,
                        start_new_session=True,
                        env=_minimal_host_environment(),
                    )
                    evidence_stream.flush()
                    os.fsync(evidence_stream.fileno())
            except BaseException as error:
                execution_error = error

            try:
                cleanup_evidence = _ensure_container_absent(
                    name,
                    expected_source_commit=source_commit,
                    launch_nonce=nonce,
                    validation_artifact_profile=artifact_profile.name,
                    runner=cleanup_runner,
                )
            except BaseException as error:
                recovered_cleanup_evidence = _cleanup_evidence_from_error(error)
                if recovered_cleanup_evidence is not None:
                    cleanup_evidence = recovered_cleanup_evidence
                cleanup_error = error

            try:
                postflight = preflight_gpu5(runner=smi_runner)
            except BaseException as error:
                postflight_error = error

            try:
                scope_after = scope_reader(source_commit, **scope_arguments)
                if scope_after != scope_before:
                    raise GPU5BoundaryError(
                        "source/model execution scope changed during GPU validation"
                    )
            except BaseException as error:
                scope_error = error

            try:
                evidence_payload, evidence_bytes, evidence_sha256 = _evidence_snapshot(
                    evidence_handle
                )
                if artifact_profile.name == PRODUCT_E4B_IT_ARTIFACT_PROFILE:
                    evidence_component_schema = _validate_product_acceptance_evidence(
                        evidence_payload,
                        expected_model_manifest_sha256=(
                            scope_before.model_manifest_sha256
                        ),
                    )
            except BaseException as error:
                evidence_error = error

        if (
            cleanup_error is None
            and cleanup_evidence is not None
            and cleanup_evidence.get("container_absent") is True
            and postflight_error is None
            and postflight is not None
            and scope_error is None
            and scope_after == scope_before
            and evidence_error is None
            and evidence_bytes is not None
            and evidence_sha256 is not None
        ):
            lease.mark_safe_to_release()

    returncode = None if completed is None else int(completed.returncode)
    if (
        execution_error is not None
        or returncode != 0
        or cleanup_error is not None
        or postflight_error is not None
        or postflight is None
        or evidence_error is not None
        or evidence_bytes is None
        or evidence_sha256 is None
        or scope_error is not None
        or scope_after is None
    ):
        evidence = {
            "argv": argv,
            "image_digest": PINNED_DOCKER_IMAGE,
            "container_name": name,
            "launch_nonce": nonce,
            "validation_artifact_profile": artifact_profile.name,
            "source_snapshot": source_snapshot.as_payload(),
            "model_snapshot": model_snapshot.as_payload(),
            "returncode": returncode,
            "execution_error": (
                None if execution_error is None else type(execution_error).__name__
            ),
            "cleanup": cleanup_evidence,
            "cleanup_error": (
                None if cleanup_error is None else type(cleanup_error).__name__
            ),
            "postflight_error": (
                None if postflight_error is None else type(postflight_error).__name__
            ),
            "source_scope_before": scope_before.as_payload(),
            "source_scope_after": (
                None if scope_after is None else scope_after.as_payload()
            ),
            "scope_error": None if scope_error is None else type(scope_error).__name__,
            "evidence_path": evidence_path,
            "evidence_bytes": evidence_bytes,
            "evidence_sha256": evidence_sha256,
            "evidence_error": (
                None if evidence_error is None else type(evidence_error).__name__
            ),
            "output_policy": (
                "strict_json_component"
                if artifact_profile.name == PRODUCT_E4B_IT_ARTIFACT_PROFILE
                else "bounded_file_capture"
            ),
            "evidence_component_schema": evidence_component_schema,
        }
        docker_failure = GPU5DockerExecutionError(
            "GPU5 Docker run failed closed", evidence
        )
        phase_failures: list[BaseException] = []
        for error in (
            execution_error,
            cleanup_error,
            postflight_error,
            scope_error,
            evidence_error,
        ):
            if error is not None:
                phase_failures.extend(_flatten_failure_objects(error))
        fatal_controls = [
            error for error in phase_failures if not isinstance(error, Exception)
        ]
        if fatal_controls:
            fatal = fatal_controls[0]
            fatal_is_only_failure = (
                len(phase_failures) == 1
                and len(fatal_controls) == 1
                and returncode in (None, 0)
                and (completed is not None or execution_error is fatal)
                and (
                    cleanup_evidence is not None
                    and cleanup_evidence.get("container_absent") is True
                )
                and (postflight is not None or postflight_error is fatal)
                and (
                    (scope_after is not None and scope_after == scope_before)
                    or scope_error is fatal
                )
                and (
                    (evidence_bytes is not None and evidence_sha256 is not None)
                    or evidence_error is fatal
                )
            )
            if fatal_is_only_failure and _attach_failure_evidence(
                fatal, "gpu5_docker_execution_error", docker_failure
            ):
                raise fatal
            aggregate = GPU5AggregateError(
                "GPU5 Docker control and safety failures",
                [*_secondary_failures(phase_failures, fatal), docker_failure],
                docker_failure,
            )
            _attach_failure_evidence(
                fatal, "gpu5_docker_execution_error", docker_failure
            )
            _attach_failure_evidence(fatal, "gpu5_aggregate_error", aggregate)
            raise fatal from aggregate
        raise docker_failure
    assert postflight is not None
    assert evidence_bytes is not None
    assert evidence_sha256 is not None
    assert cleanup_evidence is not None
    return GuardedDockerResult(
        argv=argv,
        preflight=preflight,
        postflight=postflight,
        image_digest=PINNED_DOCKER_IMAGE,
        returncode=returncode,
        evidence_path=evidence_path,
        evidence_bytes=evidence_bytes,
        evidence_sha256=evidence_sha256,
        source_commit=scope_before.source_commit,
        source_tree_digest=scope_before.source_tree_digest,
        source_identity_digest=scope_before.source_identity_digest,
        model_manifest_sha256=scope_before.model_manifest_sha256,
        model_tree_digest=scope_before.model_tree_digest,
        model_identity_digest=scope_before.model_identity_digest,
        container_name=name,
        launch_nonce=nonce,
        snapshot_path=source_snapshot.root_path,
        snapshot_mode=source_snapshot.root_mode,
        cleanup=cleanup_evidence,
        validation_artifact_profile=artifact_profile.name,
        evidence_component_schema=evidence_component_schema,
        output_policy=(
            "strict_json_component"
            if artifact_profile.name == PRODUCT_E4B_IT_ARTIFACT_PROFILE
            else "bounded_file_capture"
        ),
    )


def _host_run_pycache_prefix() -> Path:
    return GUARD_STATE_ROOT / "host-never-pycache"


def _validate_run_bootstrap() -> None:
    """Require the exact isolated, bytecode-free host Python launch contract."""

    expected_prefix = str(_host_run_pycache_prefix())
    if not sys.flags.isolated:
        raise GPU5BoundaryError("run requires host Python isolated mode (-I)")
    if not sys.dont_write_bytecode:
        raise GPU5BoundaryError("run requires host Python bytecode suppression (-B)")
    if sys.pycache_prefix != expected_prefix:
        raise GPU5BoundaryError(
            "run requires the fixed host-never-pycache -X pycache_prefix"
        )
    if os.path.lexists(_host_run_pycache_prefix()):
        raise GPU5BoundaryError("host-never-pycache path must remain nonexistent")


def _add_common_launch_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--image", required=True)
    parser.add_argument("--expected-source-commit", required=True)
    parser.add_argument("--workdir", default=_PINNED_WORKDIR)
    parser.add_argument(
        "--validation-artifact-profile",
        choices=tuple(_VALIDATION_ARTIFACT_PROFILES),
        default=BASE_CANARY_ARTIFACT_PROFILE,
        help=(
            "immutable raw-model/manifest/validator capability: historical base "
            "canary or product E4B-it"
        ),
    )
    parser.add_argument("container_command", nargs=argparse.REMAINDER)


def _container_command_from_args(args: argparse.Namespace) -> list[str]:
    command = list(args.container_command)
    if command[:1] == ["--"]:
        command = command[1:]
    return command


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fail-closed physical GPU5 inspection and guarded execution."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("inspect", help="fail unless the pinned GPU5 is idle")
    docker = subparsers.add_parser(
        "docker-argv", help="create a snapshot and print a non-release Docker argv"
    )
    _add_common_launch_arguments(docker)
    run = subparsers.add_parser(
        "run", help="snapshot, execute, clean up, and emit bounded JSON evidence"
    )
    _add_common_launch_arguments(run)
    run.add_argument("--timeout", required=True, type=float)
    run.add_argument("--evidence-filename", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "inspect":
        print(json.dumps(preflight_gpu5().as_payload(), indent=2, sort_keys=True))
        return 0
    container_command = _container_command_from_args(args)
    if args.command == "run":
        try:
            _validate_run_bootstrap()
            source_commit = _validated_expected_source_commit(
                args.expected_source_commit
            )
            scheduler_reservation = _require_external_gpu5_scheduler_reservation(
                source_commit,
                required_run_seconds=args.timeout,
            )
            result = run_guarded_gpu5_container(
                args.image,
                container_command,
                expected_source_commit=source_commit,
                environment=_REQUIRED_CONTAINER_ENVIRONMENT,
                workdir=args.workdir,
                run_timeout_seconds=args.timeout,
                evidence_filename=args.evidence_filename,
                validation_artifact_profile=args.validation_artifact_profile,
            )
        except GPU5BoundaryError as error:
            payload: dict[str, Any] = {
                "status": "failed_closed",
                "error_type": type(error).__name__,
                "error": str(error),
            }
            if isinstance(error, (GPU5DockerExecutionError, GPU5CleanupError)):
                payload["evidence"] = error.evidence
            print(json.dumps(payload, indent=2, sort_keys=True))
            return 2
        print(
            json.dumps(
                {
                    "status": "passed",
                    "scheduler_reservation": scheduler_reservation,
                    "result": asdict(result),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    source_commit = _validated_expected_source_commit(args.expected_source_commit)
    _prepare_guard_state()
    nonce = _launch_nonce()
    snapshot = prepare_source_snapshot(source_commit, nonce)
    model_snapshot = _prepare_docker_model_snapshot(
        snapshot,
        launch_nonce=nonce,
        validation_artifact_profile=args.validation_artifact_profile,
    )
    capture_execution_scope(
        source_commit,
        source_snapshot=snapshot,
        model_snapshot=model_snapshot,
        validation_artifact_profile=args.validation_artifact_profile,
    )
    runtime_mounts = _runtime_mount_map(
        expected_source_commit=source_commit,
        launch_nonce=nonce,
        source_snapshot=snapshot,
        model_snapshot=model_snapshot,
        validation_artifact_profile=args.validation_artifact_profile,
    )
    mounts = tuple(
        f"{source.resolve(strict=True)}:{destination}:ro"
        for source, destination in runtime_mounts.items()
    )
    docker_argv = build_gpu5_docker_argv(
        args.image,
        container_command,
        expected_source_commit=source_commit,
        source_snapshot=snapshot,
        model_snapshot=model_snapshot,
        environment=_REQUIRED_CONTAINER_ENVIRONMENT,
        mounts=mounts,
        workdir=args.workdir,
        launch_nonce=nonce,
        validation_artifact_profile=args.validation_artifact_profile,
    )
    print(json.dumps({"argv": docker_argv, "shell": shlex.join(docker_argv)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
