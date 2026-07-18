"""Standard-library, loopback-only control plane for the graphical demo."""

from __future__ import annotations

from collections import deque
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from hashlib import sha256
from http import HTTPStatus
from http.cookies import SimpleCookie
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from http.client import HTTPConnection
from inspect import Parameter, signature
import argparse
import hmac
import json
from math import isfinite
import os
from pathlib import Path
from queue import Empty, Full, Queue
import re
import secrets
import shutil
import signal
import stat
import subprocess
import sys
from threading import Condition, Event, RLock, Thread, get_ident
from time import monotonic, sleep
from typing import Any, BinaryIO
from urllib.parse import parse_qs, urlsplit
import webbrowser

from cogni_demo.lens_api import LensApiClient
from cogni_demo.protocol import (
    EVENT_SENTINEL,
    MAX_EVENT_LINE_BYTES,
    PHASE_STAGES,
    ProtocolError,
    WorkerEvent,
    parse_event_line,
    validate_terminal_metrics,
)
from cogni_demo.workspace_capabilities import (
    AKASICDB_AUDITED_REVISION,
    AKASICDB_REPOSITORY,
    MAX_ATTACHMENT_BYTES,
    MAX_ATTACHMENT_BASE64_CHARS,
    MAX_INDEXED_TEXT_CHARS,
    RAG_ANSWER_INTEGRATION_SCHEMA,
    RAG_EMBEDDING_PROFILE,
    RAG_QUERY_SCHEMA_VERSION,
    RAG_RETRIEVAL_MODE,
    WorkspaceCapabilityError,
    WorkspaceCapabilityService,
    web_policy_from_environment,
)
from cogni_agent.tools import (
    MAX_PROJECT_COMMAND_BYTES,
    ToolPolicyError,
    parse_tool_request,
)
from cogni_agent.local_voice import (
    Gemma4ModelSpeechTranscriber,
    LocalVoiceError,
    LocalVoiceService,
    MAX_TTS_TEXT_CHARS,
    MAX_VOICE_BASE64_CHARS,
    WindowsSpeechSynthesizer,
)
from cogni_flow.rhythm import RhythmController, SystemMode
from cogni_flow.proposal_review import (
    ProposalReviewError,
    build_proposal_review,
)
from cogni_os.artifacts import verify_artifact_manifest
from cogni_os.gpu_lease import (
    GPULease,
    GPULeaseManager,
    StaleGPULeaseError,
)


MAX_REQUEST_BODY_BYTES = 8 * 1024
MAX_AGENT_CHAT_REQUEST_BODY_BYTES = 64 * 1024
# ``/api/agent/chat`` carries both ordinary dialogue and the explicitly typed
# ``/project`` command.  The latter has a 1 MiB decoded-command ceiling, while
# JSON string escaping may make its wire envelope larger.  Keep the HTTP read
# bounded, then apply the stricter mode-aware decoded limits below.
MAX_AGENT_PROJECT_REQUEST_BODY_BYTES = 2 * MAX_PROJECT_COMMAND_BYTES + 4 * 1024
MAX_ATTACHMENT_REQUEST_BODY_BYTES = MAX_ATTACHMENT_BASE64_CHARS + 4 * 1024
MAX_VOICE_REQUEST_BODY_BYTES = MAX_VOICE_BASE64_CHARS + 4 * 1024
MAX_TTS_REQUEST_BODY_BYTES = MAX_TTS_TEXT_CHARS * 4 + 1024
MAX_AGENT_CHAT_MESSAGE_CHARS = 4_096
MAX_RAG_QUERY_CHARS = 1_024
MAX_RAG_EVIDENCE_CHARS = 6_000
MAX_RAG_EVIDENCE_CHUNKS = 5
MAX_RAG_EVIDENCE_CHUNK_CHARS = 1_600
MAX_PROMPT_LENGTH = 256
MAX_STATE_EVENTS = 64
MAX_DIAGNOSTIC_LINES = 200
MAX_EVENT_QUEUE = 64
READ_CHUNK_BYTES = 4096
MAX_ASSET_BYTES = 2 * 1024 * 1024
DEFAULT_PORT = 8765
DEFAULT_WATCHDOG_SECONDS = 60.0
COMPONENT_SHUTDOWN_WAIT_SECONDS = 45.0
SERVER_GPU_IDENTITY_PROBE_TIMEOUT_SECONDS = 45.0
MAX_SESSION_BYTES = 4096
SESSION_VERSION = 1
SERVICE_MARKER = "cogniboard"
DESKTOP_VALIDATION_PROFILE = "desktop-ui-only"
SERVER_GPU5_VALIDATION_PROFILE = "server-gpu5-native"
SERVER_VALIDATION_PHYSICAL_GPU_INDEX = 5
SERVER_VALIDATION_GPU_QUERY_CONTEXT = "native-host"
SERVER_VALIDATION_GPU_UUID = "GPU-84d7eeb0-65e0-a5b1-d7db-d09ef59fe03a"
_EXPECTED_SOURCE_COMMIT = re.compile(r"[0-9a-f]{40}\Z")
_NATIVE_GPU5_LIFECYCLE_ERROR_MESSAGE = (
    "native GPU5 server lifecycle reported multiple failures"
)
_NATIVE_GPU5_LIFECYCLE_TOKEN = object()

_SERVER_VALIDATION_ENVIRONMENT = {
    "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
    "CUDA_VISIBLE_DEVICES": SERVER_VALIDATION_GPU_UUID,
    "NVIDIA_VISIBLE_DEVICES": SERVER_VALIDATION_GPU_UUID,
}
_SERVER_REJECTED_PARENT_ENVIRONMENT = frozenset(
    {
        "LD_PRELOAD",
        "PYTHONHOME",
        "PYTHONINSPECT",
        "PYTHONPATH",
        "PYTHONSTARTUP",
        "PYTHONUSERBASE",
    }
)
_SERVER_CHILD_REMOVED_ENVIRONMENT = frozenset(
    {
        "PYTHONHOME",
        "PYTHONINSPECT",
        "PYTHONSTARTUP",
        "PYTHONUSERBASE",
    }
)
_SERVER_CHILD_FIXED_ENVIRONMENT = {
    "LD_PRELOAD": "",
    "PATH": "/usr/local/bin:/usr/bin:/bin",
    "PYTHONDONTWRITEBYTECODE": "1",
    "PYTHONHASHSEED": "0",
    "PYTHONNOUSERSITE": "1",
    "PYTHONPATH": "/nonexistent-cogniboard-pythonpath",
    "PYTHONSAFEPATH": "1",
}

_SERVER_IDENTITY_PROBE_FIXED_ENVIRONMENT = {
    "PATH": "/usr/local/bin:/usr/bin:/bin",
    "LANG": "C.UTF-8",
    "LC_ALL": "C.UTF-8",
    "HF_HUB_OFFLINE": "1",
    "HF_HUB_DISABLE_TELEMETRY": "1",
    "TRANSFORMERS_OFFLINE": "1",
    "HF_DATASETS_OFFLINE": "1",
    "WANDB_MODE": "offline",
    "TOKENIZERS_PARALLELISM": "false",
    "PYTHONDONTWRITEBYTECODE": "1",
    "PYTHONHASHSEED": "0",
    "PYTHONNOUSERSITE": "1",
    "PYTHONSAFEPATH": "1",
    "COGNI_OS_GPU_UUID": SERVER_VALIDATION_GPU_UUID,
    **_SERVER_VALIDATION_ENVIRONMENT,
}

_RAG_SOURCE_RESPONSE_KEYS = frozenset(
    {
        "schema_version",
        "attachment_id",
        "chunk_index",
        "name",
        "media_type",
        "text",
        "representation",
        "page_number",
        "char_start",
        "char_end",
        "offset_basis",
        "excerpt_sha256",
    }
)
_RAG_SOURCE_OFFSET_BASES = frozenset(
    {"normalized_document_text_v1", "normalized_pdf_page_text_v1"}
)
_RAG_SOURCE_REPRESENTATION = "normalized_extracted_excerpt_v1"
_RAG_SOURCE_MEDIA_TYPES = frozenset(
    {
        "application/json",
        "application/pdf",
        "text/csv",
        "text/markdown",
        "text/plain",
    }
)
_RAG_QUERY_RESPONSE_KEYS = frozenset(
    {
        "schema_version",
        "engine",
        "repository",
        "revision",
        "retrieval_mode",
        "embedding",
        "semantic_embedding",
        "answer_integration",
        "answer_integration_schema",
        "query",
        "results",
        "count",
    }
)
_RAG_QUERY_RESULT_KEYS = frozenset(
    (_RAG_SOURCE_RESPONSE_KEYS - {"schema_version"}) | {"source_sha256", "score"}
)

_ACTIVE_STATUSES = {"starting", "running", "cancelling"}
_STATIC_ASSETS = {
    "/": ("index.html", "text/html; charset=utf-8"),
    "/index.html": ("index.html", "text/html; charset=utf-8"),
    "/assets/app.css": ("app.css", "text/css; charset=utf-8"),
    "/assets/app.js": ("app.js", "text/javascript; charset=utf-8"),
    "/assets/favicon.svg": ("favicon.svg", "image/svg+xml"),
}
_CSP = (
    "default-src 'self'; script-src 'self'; style-src 'self'; "
    "connect-src 'self'; img-src 'self' blob:; media-src 'self' blob:; "
    "object-src 'none'; "
    "base-uri 'none'; frame-ancestors 'none'; form-action 'none'"
)


INITIAL_METRICS: dict[str, Any] = {
    # No historical observation is valid evidence for a newly started server.
    # These fields are populated only after this JobManager accepts one complete,
    # ordered, zero-exit validation result from the current process.
    "evidence_kind": "unverified",
    "measured_at": None,
    "source": None,
    "peak_allocated_vram_gib": None,
    "peak_reserved_vram_gib": None,
    "peak_vram_gib": None,
    "vram_limit_gib": None,
    "requested_depth": None,
    "reached_depth": None,
    "nodes_used": None,
    "node_capacity": None,
    "transition_residual": None,
    "transition_converged": None,
    "transition_used_fallback": None,
    "cts_protocol_version": None,
    "safe_for_decode": None,
    "unsafe_silent_fallbacks": None,
    "linear_solve_fallbacks": None,
    "solver_rank": None,
    "solver_history_peak": None,
    "solver_failures": None,
    "failed_edges": None,
    "q_zero_backups": None,
    "mac_budget": None,
    "mac_reserved": None,
    "act_applied": None,
    "trace_digest": None,
    "causal_bridge_answer_bearing": None,
    "causal_bridge_bias_nonzero": None,
    "causal_bridge_bias_max": None,
    "conditioned_generated_tokens": None,
    "finite": None,
    "verified_files": None,
    "model_class": None,
    "hidden_size": None,
    "search_allocated_bytes": None,
    "load_seconds": None,
    "inference_seconds": None,
    # Test counts belong to a signed release-evidence snapshot.  Keeping them
    # out of live telemetry prevents a stale build-time number being presented
    # as if the currently running process had just executed the suite.
    "tests": None,
    "subtests": None,
    "device": None,
    "target": "RTX 4090 24GB",
}

_AGENT_FAILURE_ROUTES: dict[str, tuple[str, str, str]] = {
    "ResponseQualityError": (
        "agent_response_quality",
        "agent_manager",
        "cogni_agent/manager.py",
    ),
    "WorkerExecutionError": (
        "agent_worker_execution",
        "resident_model_worker",
        "cogni_agent/model_service.py",
    ),
    "ModelServiceError": (
        "agent_model_service",
        "resident_model_worker",
        "cogni_agent/model_service.py",
    ),
    "TimeoutError": (
        "agent_timeout",
        "agent_manager",
        "cogni_agent/manager.py",
    ),
}
_UNCLASSIFIED_AGENT_FAILURE = (
    "agent_unclassified",
    "agent_manager",
    "cogni_agent/manager.py",
)


def _agent_failure_route(code: str) -> tuple[str, str, str]:
    """Keep terminal causes in distinct, bounded Self-Harness clusters."""

    if not isinstance(code, str) or not code or len(code) > 128:
        return _UNCLASSIFIED_AGENT_FAILURE
    return _AGENT_FAILURE_ROUTES.get(code, _UNCLASSIFIED_AGENT_FAILURE)


def _rag_source_payload_is_valid(
    payload: object, *, attachment_id: str, chunk_index: int
) -> bool:
    """Validate the exact browser-visible source schema at the HTTP boundary."""

    if not isinstance(payload, dict) or set(payload) != _RAG_SOURCE_RESPONSE_KEYS:
        return False
    schema_version = payload.get("schema_version")
    payload_chunk_index = payload.get("chunk_index")
    name = payload.get("name")
    media_type = payload.get("media_type")
    text = payload.get("text")
    representation = payload.get("representation")
    page_number = payload.get("page_number")
    char_start = payload.get("char_start")
    char_end = payload.get("char_end")
    offset_basis = payload.get("offset_basis")
    excerpt_sha256 = payload.get("excerpt_sha256")
    if (
        not isinstance(schema_version, int)
        or isinstance(schema_version, bool)
        or schema_version != 2
        or payload.get("attachment_id") != attachment_id
        or not isinstance(payload_chunk_index, int)
        or isinstance(payload_chunk_index, bool)
        or payload_chunk_index != chunk_index
        or not isinstance(name, str)
        or not 1 <= len(name) <= 128
        or name in {".", ".."}
        or "/" in name
        or "\\" in name
        or any(
            ord(character) < 32 or 127 <= ord(character) <= 159 for character in name
        )
        or not isinstance(media_type, str)
        or media_type not in _RAG_SOURCE_MEDIA_TYPES
        or not isinstance(text, str)
        or not 1 <= len(text) <= MAX_RAG_EVIDENCE_CHUNK_CHARS
        or not text.strip()
        or representation != _RAG_SOURCE_REPRESENTATION
        or any(
            (ord(character) < 32 and character not in "\t\r\n")
            or 127 <= ord(character) <= 159
            for character in text
        )
        or not isinstance(char_start, int)
        or isinstance(char_start, bool)
        or char_start < 0
        or char_start >= MAX_INDEXED_TEXT_CHARS
        or not isinstance(char_end, int)
        or isinstance(char_end, bool)
        or char_end <= char_start
        or char_end > MAX_INDEXED_TEXT_CHARS
        or char_end - char_start != len(text)
        or offset_basis not in _RAG_SOURCE_OFFSET_BASES
        or not isinstance(excerpt_sha256, str)
        or re.fullmatch(r"[0-9a-f]{64}", excerpt_sha256) is None
        or excerpt_sha256 != sha256(text.encode("utf-8")).hexdigest()
    ):
        return False
    if offset_basis == "normalized_document_text_v1":
        return page_number is None
    return (
        isinstance(page_number, int)
        and not isinstance(page_number, bool)
        and 1 <= page_number <= 128
    )


def _admitted_product_retrieval_identity(item: object) -> tuple[str, int] | None:
    """Return the canonical source identity only for exact product authority."""

    from cogni_agent.manager import RetrievalEvidence, RetrievalProvenance

    if type(item) is not RetrievalEvidence:
        return None
    source_id = item.source_id
    match = re.fullmatch(r"([0-9a-f]{24})\.(0|[1-9][0-9]{0,2})", source_id)
    if match is None:
        return None
    attachment_id = match.group(1)
    chunk_index = int(match.group(2))
    if chunk_index >= 128:
        return None
    provenance = item.provenance
    selected_text = item.text
    if (
        type(provenance) is not RetrievalProvenance
        or provenance.repository != AKASICDB_REPOSITORY
        or provenance.revision != AKASICDB_AUDITED_REVISION
        or provenance.retrieval_mode != RAG_RETRIEVAL_MODE
        or provenance.embedding != RAG_EMBEDDING_PROFILE
        or provenance.semantic_embedding is not False
        or provenance.answer_integration_schema != RAG_ANSWER_INTEGRATION_SCHEMA
        or re.fullmatch(r"[0-9a-f]{64}", provenance.source_sha256) is None
        or not provenance.source_sha256.startswith(attachment_id)
        or re.fullmatch(r"[0-9a-f]{64}", provenance.indexed_excerpt_sha256) is None
        or type(provenance.indexed_excerpt_chars) is not int
        or not 1 <= provenance.indexed_excerpt_chars <= MAX_RAG_EVIDENCE_CHUNK_CHARS
        or not isinstance(selected_text, str)
        or not 1 <= len(selected_text) <= provenance.indexed_excerpt_chars
        or (
            len(selected_text) == provenance.indexed_excerpt_chars
            and sha256(selected_text.encode("utf-8")).hexdigest()
            != provenance.indexed_excerpt_sha256
        )
    ):
        return None
    return attachment_id, chunk_index


def _product_retrieval_evidence_matches_current_source(
    item: object, workspace_service: object | None
) -> bool:
    """Bind an answer citation to the workspace's current exact source snapshot."""

    identity = _admitted_product_retrieval_identity(item)
    if identity is None or workspace_service is None:
        return False
    authority_reader = getattr(workspace_service, "current_rag_source_authority", None)
    if not callable(authority_reader):
        return False
    attachment_id, chunk_index = identity
    try:
        authority = authority_reader(attachment_id, chunk_index)
    except (OSError, TypeError, ValueError, WorkspaceCapabilityError):
        return False
    if not isinstance(authority, dict) or set(authority) != {
        "source",
        "source_sha256",
    }:
        return False
    current = authority["source"]
    current_source_sha256 = authority["source_sha256"]
    if not _rag_source_payload_is_valid(
        current,
        attachment_id=attachment_id,
        chunk_index=chunk_index,
    ):
        return False
    provenance = item.provenance
    current_text = current["text"]
    return bool(
        isinstance(current_source_sha256, str)
        and re.fullmatch(r"[0-9a-f]{64}", current_source_sha256) is not None
        and current_source_sha256 == provenance.source_sha256
        and current["name"] == item.title
        and current["excerpt_sha256"] == provenance.indexed_excerpt_sha256
        and len(current_text) == provenance.indexed_excerpt_chars
        and current_text.startswith(item.text)
    )


class DemoServerError(RuntimeError):
    """Base class for bounded control-plane failures."""


class JobAlreadyRunningError(DemoServerError):
    pass


class NoActiveJobError(DemoServerError):
    pass


class ComputeBusyError(DemoServerError):
    """Raised before a second local workload can acquire the GPU owner slot."""


class WorkerTerminationError(DemoServerError):
    """Raised when a validation worker cannot be proven dead within bounds."""


class _ComponentShutdownState(Enum):
    IDLE = "idle"
    RUNNING = "running"
    DONE = "done"


class GPUExecutionBoundaryError(DemoServerError):
    """Raised when live model validation lacks an exact server GPU boundary."""


_CLEANUP_ERROR_MESSAGE = "CogniBoard cleanup reported multiple failures"


def _append_cleanup_failure(
    failures: list[BaseException], error: BaseException
) -> None:
    """Flatten only groups produced by this helper while preserving order."""

    if (
        isinstance(error, BaseExceptionGroup)
        and error.message == _CLEANUP_ERROR_MESSAGE
    ):
        for nested in error.exceptions:
            _append_cleanup_failure(failures, nested)
        return
    failures.append(error)


def _run_best_effort_cleanup(
    callbacks: Sequence[Callable[[], object]],
    *,
    primary: BaseException | None = None,
) -> None:
    """Run every cleanup step and retain every failure in execution order."""

    failures: list[BaseException] = []
    if primary is not None:
        failures.append(primary)
    for callback in callbacks:
        try:
            callback()
        except BaseException as error:
            _append_cleanup_failure(failures, error)
    if len(failures) == 1:
        raise failures[0]
    if failures:
        raise BaseExceptionGroup(_CLEANUP_ERROR_MESSAGE, failures)


def _shutdown_product_components(
    manager: object,
    evolution_manager: object | None,
    agent_manager: object | None,
    *,
    primary: BaseException | None = None,
) -> None:
    """Shut down validation, evolution, and resident-model owners independently."""

    callbacks: list[Callable[[], object]] = []
    for component in (manager, evolution_manager, agent_manager):
        shutdown = getattr(component, "shutdown", None)
        if callable(shutdown):
            callbacks.append(shutdown)
    _run_best_effort_cleanup(callbacks, primary=primary)


class EvolutionAlreadyRunningError(DemoServerError):
    pass


@dataclass(frozen=True)
class SessionMetadata:
    pid: int
    port: int
    token: str
    started_at: str

    @property
    def bootstrap_url(self) -> str:
        return f"http://127.0.0.1:{self.port}/?token={self.token}"

    def as_payload(self) -> dict[str, Any]:
        return {
            "service": SERVICE_MARKER,
            "v": SESSION_VERSION,
            "pid": self.pid,
            "port": self.port,
            "token": self.token,
            "started_at": self.started_at,
        }


@dataclass(frozen=True)
class WorkerLaunch:
    command: tuple[str, ...]
    cwd: Path
    environment: Mapping[str, str]


@dataclass(frozen=True)
class GPUExecutionBoundary:
    """Explicit authority for a native-host live validation subprocess.

    The Windows appliance may start in ``desktop-ui-only`` mode without this
    authority.  In that mode conversation and other local UI features remain
    available, while a live hardware-validation request fails before Popen.
    """

    physical_gpu_index: int
    gpu_query_context: str
    gpu_uuid: str

    def __post_init__(self) -> None:
        if (
            type(self.physical_gpu_index) is not int
            or self.physical_gpu_index != SERVER_VALIDATION_PHYSICAL_GPU_INDEX
        ):
            raise GPUExecutionBoundaryError(
                "server validation physical GPU index is not the pinned project index"
            )
        if self.gpu_query_context != SERVER_VALIDATION_GPU_QUERY_CONTEXT:
            raise GPUExecutionBoundaryError(
                "the direct product worker requires the native-host query context"
            )
        if self.gpu_uuid != SERVER_VALIDATION_GPU_UUID:
            raise GPUExecutionBoundaryError(
                "server validation GPU UUID does not match the pinned project device"
            )

    def require_native_environment(self, environment: Mapping[str, str]) -> None:
        """Reject a server process that was not isolated before Python started."""

        for name, expected in _SERVER_VALIDATION_ENVIRONMENT.items():
            if environment.get(name) != expected:
                raise GPUExecutionBoundaryError(
                    f"server validation requires exact pre-start environment: {name}"
                )
        for name in _SERVER_REJECTED_PARENT_ENVIRONMENT:
            if environment.get(name):
                raise GPUExecutionBoundaryError(
                    f"server validation rejects pre-start environment: {name}"
                )

    def child_environment(self, environment: Mapping[str, str]) -> dict[str, str]:
        """Return the exact inherited child environment, rejecting conflicts."""

        child = dict(environment)
        for name, expected in _SERVER_VALIDATION_ENVIRONMENT.items():
            configured = child.get(name)
            if configured is not None and configured != expected:
                raise GPUExecutionBoundaryError(
                    f"conflicting server validation environment: {name}"
                )
            child[name] = expected
        for name in _SERVER_CHILD_REMOVED_ENVIRONMENT:
            child.pop(name, None)
        child.update(_SERVER_CHILD_FIXED_ENVIRONMENT)
        return child

    @property
    def validator_arguments(self) -> tuple[str, ...]:
        return (
            "--physical-gpu-index",
            str(self.physical_gpu_index),
            "--gpu-query-context",
            self.gpu_query_context,
        )


LaunchFactory = Callable[[str], WorkerLaunch]


def default_session_path() -> Path:
    """Return the same-user session file beneath the Windows profile."""

    local_app_data = os.environ.get("LOCALAPPDATA")
    if local_app_data:
        root = Path(local_app_data)
    else:
        root = Path.home() / "AppData" / "Local"
    return root / "CogniOS" / "cogniboard-session.json"


def _valid_session_payload(payload: object) -> SessionMetadata | None:
    if not isinstance(payload, dict) or set(payload) != {
        "service",
        "v",
        "pid",
        "port",
        "token",
        "started_at",
    }:
        return None
    if (
        payload["service"] != SERVICE_MARKER
        or not isinstance(payload["v"], int)
        or isinstance(payload["v"], bool)
        or payload["v"] != SESSION_VERSION
    ):
        return None
    pid, port, token, started_at = (
        payload["pid"],
        payload["port"],
        payload["token"],
        payload["started_at"],
    )
    if (
        not isinstance(pid, int)
        or isinstance(pid, bool)
        or pid <= 0
        or not isinstance(port, int)
        or isinstance(port, bool)
        or not 1 <= port <= 65535
        or not isinstance(token, str)
        or not 32 <= len(token) <= 128
        or any(
            character
            not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_"
            for character in token
        )
        or not isinstance(started_at, str)
        or not started_at
        or len(started_at) > 64
    ):
        return None
    return SessionMetadata(pid, port, token, started_at)


def read_session_metadata(path: str | Path | None = None) -> SessionMetadata | None:
    """Read a bounded, non-symlink session document; malformed data is stale."""

    candidate = Path(path) if path is not None else default_session_path()
    try:
        if candidate.is_symlink():
            return None
        info = candidate.stat()
        if not stat.S_ISREG(info.st_mode) or not 0 < info.st_size <= MAX_SESSION_BYTES:
            return None
        payload = json.loads(candidate.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    return _valid_session_payload(payload)


def _restrict_path_to_owner(path: Path, *, directory: bool) -> None:
    """Apply an owner-only DACL on Windows or owner mode bits on POSIX."""

    if os.name != "nt":
        path.chmod(stat.S_IRWXU if directory else stat.S_IRUSR | stat.S_IWUSR)
        return

    import ctypes
    from ctypes import wintypes

    advapi32 = ctypes.WinDLL("advapi32", use_last_error=True)
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    convert = advapi32.ConvertStringSecurityDescriptorToSecurityDescriptorW
    convert.argtypes = [
        wintypes.LPCWSTR,
        wintypes.DWORD,
        ctypes.POINTER(wintypes.LPVOID),
        ctypes.POINTER(wintypes.DWORD),
    ]
    convert.restype = wintypes.BOOL
    set_security = advapi32.SetFileSecurityW
    set_security.argtypes = [
        wintypes.LPCWSTR,
        wintypes.DWORD,
        wintypes.LPVOID,
    ]
    set_security.restype = wintypes.BOOL
    kernel32.LocalFree.argtypes = [wintypes.HLOCAL]
    kernel32.LocalFree.restype = wintypes.HLOCAL

    descriptor = wintypes.LPVOID()
    # Protected DACL, full access only through the object's Owner Rights SID.
    if not convert("D:P(A;;GA;;;OW)", 1, ctypes.byref(descriptor), None):
        raise DemoServerError("could not build the owner-only session ACL")
    try:
        if not set_security(str(path), 0x00000004, descriptor):
            raise DemoServerError("could not apply the owner-only session ACL")
    finally:
        kernel32.LocalFree(descriptor)


def write_session_metadata(
    metadata: SessionMetadata, path: str | Path | None = None
) -> Path:
    """Atomically persist only the loopback rendezvous secret with owner intent."""

    if _valid_session_payload(metadata.as_payload()) != metadata:
        raise ValueError("session metadata is invalid")
    target = Path(path) if path is not None else default_session_path()
    directory = target.parent
    directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    if directory.is_symlink() or (target.exists() and target.is_symlink()):
        raise DemoServerError("session metadata path cannot be a symlink")
    _restrict_path_to_owner(directory, directory=True)
    encoded = json.dumps(
        metadata.as_payload(), ensure_ascii=True, separators=(",", ":")
    ).encode("utf-8")
    if len(encoded) > MAX_SESSION_BYTES:
        raise DemoServerError("session metadata exceeded its byte budget")
    temporary = target.with_name(
        f".{target.name}.{os.getpid()}.{secrets.token_hex(4)}.tmp"
    )
    descriptor = os.open(
        temporary,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL,
        stat.S_IRUSR | stat.S_IWUSR,
    )
    replaced = False
    try:
        _restrict_path_to_owner(temporary, directory=False)
        stream = os.fdopen(descriptor, "wb")
        descriptor = -1
        with stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, target)
        replaced = True
        _restrict_path_to_owner(target, directory=False)
    except BaseException:
        if replaced:
            try:
                target.unlink()
            except (FileNotFoundError, PermissionError):
                pass
        raise
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
    return target


def remove_session_metadata(
    path: str | Path | None = None, *, expected: SessionMetadata | None = None
) -> None:
    """Remove only this instance's record; never delete a replacement session."""

    target = Path(path) if path is not None else default_session_path()
    if expected is not None:
        current = read_session_metadata(target)
        if current != expected:
            return
    try:
        target.unlink()
    except (FileNotFoundError, IsADirectoryError, PermissionError):
        pass


def ping_session(metadata: SessionMetadata, timeout: float = 0.5) -> bool:
    """Check the public marker without disclosing or transmitting the token."""

    if timeout <= 0:
        raise ValueError("ping timeout must be positive")
    connection = HTTPConnection("127.0.0.1", metadata.port, timeout=timeout)
    try:
        connection.request("GET", "/api/ping")
        response = connection.getresponse()
        body = response.read(1025)
        if response.status != HTTPStatus.OK or len(body) > 1024:
            return False
        payload = json.loads(body.decode("utf-8"))
        return payload == {"service": SERVICE_MARKER, "protocol": SESSION_VERSION}
    except (OSError, TimeoutError, UnicodeDecodeError, json.JSONDecodeError):
        return False
    finally:
        connection.close()


def find_live_session(path: str | Path | None = None) -> SessionMetadata | None:
    """Return a responsive prior instance and safely clear stale metadata."""

    target = Path(path) if path is not None else default_session_path()
    metadata = read_session_metadata(target)
    if metadata is not None and ping_session(metadata):
        return metadata
    remove_session_metadata(target)
    return None


def _find_edge() -> Path | None:
    if os.name != "nt":
        return None
    candidates: list[Path] = []
    for variable in ("PROGRAMFILES(X86)", "PROGRAMFILES", "LOCALAPPDATA"):
        value = os.environ.get(variable)
        if value:
            candidates.append(
                Path(value) / "Microsoft" / "Edge" / "Application" / "msedge.exe"
            )
    located = shutil.which("msedge.exe")
    if located:
        candidates.append(Path(located))
    return next(
        (candidate.resolve() for candidate in candidates if candidate.is_file()), None
    )


def open_graphical_app(url: str) -> str:
    """Prefer an Edge app window and fall back to the registered browser."""

    edge = _find_edge()
    if edge is not None:
        try:
            subprocess.Popen(
                [
                    str(edge),
                    f"--app={url}",
                    "--start-maximized",
                    "--no-first-run",
                ],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                shell=False,
                close_fds=True,
            )
            return "edge"
        except OSError:
            pass
    webbrowser.open(url, new=1)
    return "browser"


def _validated_python_invocation_path(value: str | Path) -> str:
    """Validate a lexical interpreter path without dereferencing a venv link."""

    raw = os.fspath(value)
    if not isinstance(raw, str):
        raise TypeError("python_executable must be a text path")
    if (
        not raw
        or len(raw) > 4096
        or any(ord(character) < 32 or ord(character) == 127 for character in raw)
    ):
        raise ValueError("python_executable contains an invalid path")
    if not Path(raw).is_absolute():
        raise ValueError("python_executable must be an absolute path")
    if os.path.normpath(raw) != raw:
        raise ValueError("python_executable must be lexically normalized")
    invocation = Path(raw)
    if not invocation.is_file():
        raise ValueError("python_executable must name an existing file")
    if os.name != "nt" and not os.access(raw, os.X_OK):
        raise ValueError("python_executable is not executable")
    # Returning the trusted lexical name is intentional. Resolving a venv
    # ``bin/python`` symlink would silently select its base interpreter.
    return raw


def _identity_probe_environment(boundary: GPUExecutionBoundary) -> dict[str, str]:
    """Build the exact, closed-world environment for the identity child."""

    if not isinstance(boundary, GPUExecutionBoundary):
        raise TypeError("boundary must be GPUExecutionBoundary")
    home = os.environ.get("HOME", "")
    if (
        not home.startswith("/")
        or len(home) > 4096
        or any(ord(character) < 32 or ord(character) == 127 for character in home)
    ):
        raise GPUExecutionBoundaryError("identity probe HOME is invalid")
    environment = dict(_SERVER_IDENTITY_PROBE_FIXED_ENVIRONMENT)
    environment["HOME"] = home
    boundary.require_native_environment(environment)
    return environment


def production_launch_factory(
    project_root: str | Path,
    model_directory: str | Path,
    manifest: str | Path,
    *,
    python_executable: str | Path = sys.executable,
    gpu_boundary: GPUExecutionBoundary | None = None,
) -> LaunchFactory:
    """Create the fixed, shell-free command for the sole CUDA owner.

    A desktop appliance may construct the factory without a server boundary so
    the UI and conversation runtime remain available.  The returned callable
    then rejects live hardware validation before a child process is spawned.
    """

    root = Path(project_root).resolve(strict=True)
    worker = (root / "scripts" / "validate_gemma4_runtime.py").resolve(strict=True)
    model = Path(model_directory).resolve(strict=True)
    manifest_path = Path(manifest).resolve(strict=True)
    if not root.is_dir() or not model.is_dir() or not manifest_path.is_file():
        raise ValueError("production demo paths are incomplete")
    executable = _validated_python_invocation_path(python_executable)
    if gpu_boundary is not None:
        if not isinstance(gpu_boundary, GPUExecutionBoundary):
            raise TypeError("gpu_boundary must be GPUExecutionBoundary or None")
        gpu_boundary.require_native_environment(os.environ)

    def build(prompt: str) -> WorkerLaunch:
        if gpu_boundary is None:
            raise GPUExecutionBoundaryError(
                "live model validation is disabled in the desktop-ui-only profile"
            )
        command = [
            executable,
            "-I",
            "-B",
            "-u",
            str(worker),
            "--model",
            str(model),
            "--manifest",
            str(manifest_path),
            "--vram-limit-gib",
            "16.7",
            "--workspace-mib",
            "512",
            "--event-stream",
        ]
        command.extend(gpu_boundary.validator_arguments)
        if prompt:
            command.extend(("--prompt", prompt))
        environment = gpu_boundary.child_environment(os.environ)
        environment.update(
            {
                "HF_HUB_OFFLINE": "1",
                "HF_HUB_DISABLE_TELEMETRY": "1",
                "TRANSFORMERS_OFFLINE": "1",
                "HF_DATASETS_OFFLINE": "1",
                "WANDB_MODE": "offline",
                "TOKENIZERS_PARALLELISM": "false",
                "PYTHONUTF8": "1",
                "PYTHONUNBUFFERED": "1",
            }
        )
        return WorkerLaunch(tuple(command), root, environment)

    return build


def _bounded_lines(stream: BinaryIO) -> Iterator[tuple[bytes, bool]]:
    """Drain a pipe without ever retaining an unbounded unterminated line."""

    buffer = bytearray()
    discarding = False
    while True:
        chunk = stream.read(READ_CHUNK_BYTES)
        if not chunk:
            break
        for byte in chunk:
            if discarding:
                if byte == 10:
                    discarding = False
                continue
            buffer.append(byte)
            if byte == 10:
                yield bytes(buffer), False
                buffer.clear()
            elif len(buffer) > MAX_EVENT_LINE_BYTES:
                yield bytes(buffer[:MAX_EVENT_LINE_BYTES]), True
                buffer.clear()
                discarding = True
    if buffer and not discarding:
        yield bytes(buffer), False


class JobManager:
    """One-worker state machine; the child process exclusively owns CUDA."""

    def __init__(
        self,
        launch_factory: LaunchFactory,
        *,
        max_runtime_seconds: float = 20 * 60,
        availability_check: Callable[[], bool] | None = None,
        gpu_lease_manager: GPULeaseManager | None = None,
    ) -> None:
        if max_runtime_seconds <= 0:
            raise ValueError("max_runtime_seconds must be positive")
        if availability_check is not None and not callable(availability_check):
            raise TypeError("availability_check must be callable")
        if gpu_lease_manager is not None and not isinstance(
            gpu_lease_manager, GPULeaseManager
        ):
            raise TypeError("gpu_lease_manager must be GPULeaseManager or None")
        self._launch_factory = launch_factory
        self.availability_check = availability_check
        self.gpu_lease_manager = gpu_lease_manager
        self.max_runtime_seconds = float(max_runtime_seconds)
        self._condition = Condition(RLock())
        self._status = "ready"
        self._stage = "ready"
        self._sequence = 0
        self._progress = 100
        self._events: deque[dict[str, Any]] = deque(maxlen=MAX_STATE_EVENTS)
        self._metrics: dict[str, Any] = deepcopy(INITIAL_METRICS)
        self._error: dict[str, str] | None = None
        self._active_job: str | None = None
        self._process: subprocess.Popen[bytes] | None = None
        self._gpu_lease: GPULease | None = None
        self._cancel_event = Event()
        self._worker_thread: Thread | None = None
        self._diagnostics: deque[dict[str, str]] = deque(maxlen=MAX_DIAGNOSTIC_LINES)

    @property
    def is_active(self) -> bool:
        with self._condition:
            return self._status in _ACTIVE_STATUSES

    def snapshot(self, *, after: int | None = None) -> dict[str, Any]:
        with self._condition:
            events = list(self._events)
            if after is not None:
                events = [event for event in events if event["seq"] > after]
            return {
                "status": self._status,
                "stage": self._stage,
                "seq": self._sequence,
                "progress": self._progress,
                "events": deepcopy(events),
                "metrics": deepcopy(self._metrics),
                "error": deepcopy(self._error),
                "active_job": self._active_job,
            }

    def wait_snapshot(self, after: int, timeout: float = 10.0) -> dict[str, Any]:
        if after < 0 or not 0 <= timeout <= 15:
            raise ValueError("invalid state wait")
        with self._condition:
            self._condition.wait_for(
                lambda: self._sequence > after, timeout=float(timeout)
            )
            return self.snapshot(after=after)

    def start(self, prompt: str = "") -> str:
        if not isinstance(prompt, str) or len(prompt) > MAX_PROMPT_LENGTH:
            raise ValueError("prompt must be text no longer than 256 characters")
        if any(
            ord(character) < 32 and character not in "\t\r\n" for character in prompt
        ):
            raise ValueError("prompt contains unsupported control characters")
        with self._condition:
            if self._status in _ACTIVE_STATUSES:
                raise JobAlreadyRunningError("a validation job is already active")
            availability = self.availability_check
        if availability is not None and not availability():
            raise JobAlreadyRunningError("local compute is owned by another mode")
        with self._condition:
            if self._status in _ACTIVE_STATUSES:
                raise JobAlreadyRunningError("a validation job is already active")
            job_id = secrets.token_hex(16)
            self._cancel_event = Event()
            self._active_job = job_id
            self._error = None
            self._transition_locked("starting", "starting", 0)
            thread = Thread(
                target=self._run_job,
                args=(job_id, prompt),
                name=f"cogni-demo-{job_id[:8]}",
                daemon=True,
            )
            self._worker_thread = thread
            thread.start()
            return job_id

    def cancel(self) -> None:
        with self._condition:
            if self._status not in _ACTIVE_STATUSES:
                raise NoActiveJobError("there is no active validation job")
            if self._status != "cancelling":
                self._transition_locked("cancelling", "cancelling", self._progress)
            self._cancel_event.set()

    def shutdown(self, timeout: float = 5.0) -> None:
        if timeout <= 0:
            raise ValueError("shutdown timeout must be positive")
        deadline = monotonic() + float(timeout)
        with self._condition:
            active = self._status in _ACTIVE_STATUSES
        if active:
            try:
                self.cancel()
            except NoActiveJobError:
                pass
        thread = self._worker_thread
        if thread is not None:
            thread.join(timeout=max(0.0, deadline - monotonic()))
        process = self._process
        if process is not None and process.poll() is None:
            self._terminate(process)
        if thread is not None and thread.is_alive():
            thread.join(timeout=max(0.0, deadline - monotonic()))
        if thread is not None and thread.is_alive():
            raise WorkerTerminationError(
                "validation controller thread did not stop within its deadline"
            )

        process = self._process
        lease = self._gpu_lease
        if process is not None and process.poll() is None:
            raise WorkerTerminationError(
                "validation worker remained alive after shutdown escalation"
            )
        if process is not None or lease is not None:
            self._release_job_lease(process, lease)
        with self._condition:
            job_id = self._active_job
            progress = self._progress
        if job_id is not None:
            self._finish_job(
                job_id,
                "cancelled" if self._cancel_event.is_set() else "failed",
                "cancelled" if self._cancel_event.is_set() else "shutdown_failed",
                progress,
                error=(
                    None
                    if self._cancel_event.is_set()
                    else {
                        "code": "WORKER_SHUTDOWN",
                        "message": "validation worker stopped during shutdown",
                    }
                ),
            )

    def _transition_locked(
        self,
        status: str,
        stage: str,
        progress: int,
        *,
        error: dict[str, str] | None = None,
        clear_job: bool = False,
    ) -> None:
        self._status = status
        self._stage = stage
        self._progress = max(0, min(100, int(progress)))
        self._error = error
        if clear_job:
            self._active_job = None
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

    def _append_diagnostic(self, source: str, line: bytes, truncated: bool) -> None:
        text = line.decode("utf-8", errors="replace").strip()
        if len(text) > 512:
            text = text[:512]
            truncated = True
        with self._condition:
            self._diagnostics.append(
                {"source": source, "text": text, "truncated": str(truncated).lower()}
            )

    def _stream_reader(
        self,
        stream: BinaryIO,
        source: str,
        messages: Queue[tuple[str, object]],
    ) -> None:
        try:
            for line, truncated in _bounded_lines(stream):
                if source == "stdout":
                    if truncated:
                        if line.startswith(EVENT_SENTINEL.encode("ascii")):
                            self._put_protocol_message(
                                messages,
                                ProtocolError("worker event line exceeded its bound"),
                            )
                        else:
                            self._append_diagnostic(source, line, True)
                        continue
                    try:
                        event = parse_event_line(line)
                    except ProtocolError as exc:
                        self._put_protocol_message(messages, exc)
                        continue
                    if event is not None:
                        self._put_protocol_message(messages, event)
                        continue
                self._append_diagnostic(source, line, truncated)
        finally:
            stream.close()

    @staticmethod
    def _put_protocol_message(
        messages: Queue[tuple[str, object]], value: WorkerEvent | ProtocolError
    ) -> None:
        item = ("event" if isinstance(value, WorkerEvent) else "error", value)
        try:
            messages.put(item, timeout=1.0)
        except Full:
            # A well-formed worker emits only six events. Saturation is itself
            # a bounded protocol failure; the runner will also reject a missing
            # terminal event after the child exits.
            return

    def _run_job(self, job_id: str, prompt: str) -> None:
        process: subprocess.Popen[bytes] | None = None
        lease: GPULease | None = None
        # The health callback must fence the pre-spawn window.  It closes over
        # this exact worker, never the mutable ``self._process`` replacement.
        process_holder: list[subprocess.Popen[bytes] | None] = [None]
        messages: Queue[tuple[str, object]] = Queue(maxsize=MAX_EVENT_QUEUE)
        protocol_failure: str | None = None
        terminal_metrics: dict[str, Any] | None = None
        terminal_count = 0
        expected_worker_sequence = 1
        expected_phase_index = 0
        last_progress = -1
        readers: list[Thread] = []
        started = monotonic()
        terminal_status = "failed"
        terminal_stage = "server_failed"
        terminal_progress = 0
        terminal_error: dict[str, str] | None = None
        live_metrics: dict[str, Any] | None = None
        try:
            launch = self._launch_factory(prompt)
            if not launch.command or any(
                not isinstance(item, str) for item in launch.command
            ):
                raise ValueError("worker command must be a non-empty string sequence")
            if self._cancel_event.is_set():
                terminal_status = "cancelled"
                terminal_stage = "cancelled"
                return
            lease = self._acquire_job_lease(job_id, process_holder)
            if self._cancel_event.is_set():
                terminal_status = "cancelled"
                terminal_stage = "cancelled"
                return
            creationflags = 0
            if os.name == "nt":
                creationflags = subprocess.CREATE_NEW_PROCESS_GROUP
            process = subprocess.Popen(
                list(launch.command),
                cwd=launch.cwd,
                env=dict(launch.environment),
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                shell=False,
                bufsize=0,
                creationflags=creationflags,
                close_fds=True,
            )
            process_holder[0] = process
            with self._condition:
                if self._active_job != job_id:
                    raise RuntimeError("job ownership changed during worker start")
                self._process = process
                self._transition_locked("running", "worker_started", 1)
            assert process.stdout is not None and process.stderr is not None
            for source, stream in (
                ("stdout", process.stdout),
                ("stderr", process.stderr),
            ):
                reader = Thread(
                    target=self._stream_reader,
                    args=(stream, source, messages),
                    name=f"cogni-{source}-{job_id[:8]}",
                    daemon=True,
                )
                readers.append(reader)
                reader.start()

            terminated = False
            while (
                process.poll() is None
                or any(reader.is_alive() for reader in readers)
                or not messages.empty()
            ):
                if self._cancel_event.is_set() and not terminated:
                    self._terminate(process)
                    terminated = True
                if monotonic() - started > self.max_runtime_seconds and not terminated:
                    protocol_failure = "worker exceeded the runtime deadline"
                    self._terminate(process)
                    terminated = True
                try:
                    message_kind, value = messages.get(timeout=0.05)
                except Empty:
                    continue
                if message_kind == "error":
                    protocol_failure = str(value)[:256]
                    continue
                event = value
                assert isinstance(event, WorkerEvent)
                if event.sequence != expected_worker_sequence:
                    protocol_failure = "worker event sequence is not contiguous"
                    continue
                expected_worker_sequence += 1
                if event.progress < last_progress:
                    protocol_failure = "worker progress moved backwards"
                    continue
                last_progress = event.progress
                if event.kind == "phase":
                    if (
                        expected_phase_index >= len(PHASE_STAGES)
                        or event.stage != PHASE_STAGES[expected_phase_index]
                    ):
                        protocol_failure = "worker phase order is invalid"
                        continue
                    expected_phase_index += 1
                    with self._condition:
                        if self._active_job == job_id:
                            self._transition_locked(
                                "running", event.stage, event.progress
                            )
                else:
                    terminal_count += 1
                    if expected_phase_index != len(PHASE_STAGES):
                        protocol_failure = "terminal result arrived before all phases"
                        continue
                    terminal_metrics = validate_terminal_metrics(event.metrics or {})

            for reader in readers:
                reader.join(timeout=1.0)
            return_code = process.wait(timeout=1.0)
            if self._cancel_event.is_set():
                terminal_status = "cancelled"
                terminal_stage = "cancelled"
                terminal_progress = self._progress
            elif return_code != 0:
                terminal_status = "failed"
                terminal_stage = "worker_failed"
                terminal_progress = self._progress
                terminal_error = {
                    "code": "WORKER_EXIT",
                    "message": f"validation worker exited with code {return_code}",
                }
            elif (
                protocol_failure is not None
                or terminal_count != 1
                or terminal_metrics is None
            ):
                detail = protocol_failure or "worker emitted no unique terminal result"
                terminal_status = "failed"
                terminal_stage = "protocol_failed"
                terminal_progress = self._progress
                terminal_error = {
                    "code": "WORKER_PROTOCOL",
                    "message": detail[:256],
                }
            else:
                # Keep the dashboard evidence schema stable while replacing
                # every metric produced by the current hardware run.
                live_metrics = deepcopy(INITIAL_METRICS)
                live_metrics.update(terminal_metrics)
                live_metrics.update(
                    {
                        "evidence_kind": "live_runtime_validation",
                        "measured_at": datetime.now(timezone.utc).isoformat(),
                        "source": "scripts/validate_gemma4_runtime.py --event-stream",
                        "target": "RTX 4090 24GB",
                    }
                )
                terminal_status = "succeeded"
                terminal_stage = "complete"
                terminal_progress = 100
        except BaseException as exc:
            if self._cancel_event.is_set():
                terminal_status = "cancelled"
                terminal_stage = "cancelled"
                terminal_progress = self._progress
            else:
                terminal_status = "failed"
                terminal_stage = "server_failed"
                terminal_progress = self._progress
                terminal_error = {
                    "code": "SERVER_WORKER_FAILURE",
                    "message": f"{type(exc).__name__}: worker could not be managed"[
                        :256
                    ],
                }
        finally:
            cleanup_error: BaseException | None = None
            try:
                if process is not None and process.poll() is None:
                    self._terminate(process)
                self._release_job_lease(process, lease)
            except BaseException as exc:
                cleanup_error = exc
            if cleanup_error is not None:
                self._record_cleanup_failure(
                    job_id,
                    cleanup_error,
                    process=process,
                    lease=lease,
                )
            else:
                if live_metrics is not None:
                    with self._condition:
                        if self._active_job == job_id:
                            self._metrics = live_metrics
                self._finish_job(
                    job_id,
                    terminal_status,
                    terminal_stage,
                    terminal_progress,
                    error=terminal_error,
                )

    def _acquire_job_lease(
        self,
        job_id: str,
        process_holder: list[subprocess.Popen[bytes] | None],
    ) -> GPULease | None:
        authority = self.gpu_lease_manager
        if authority is None:
            return None

        def owner_alive() -> bool:
            candidate = process_holder[0]
            # Another contender may call reap between acquire and Popen.  The
            # exact pre-spawn owner must remain fenced during that window.
            return candidate is None or candidate.poll() is None

        lease = authority.acquire(
            f"validation-{job_id}",
            "validation",
            authority.max_vram_bytes,
            deadline=authority.deadline_after(self.max_runtime_seconds + 5.0),
            owner_alive=owner_alive,
        )
        with self._condition:
            if self._active_job != job_id:
                authority.release(lease)
                raise RuntimeError("job ownership changed during lease acquisition")
            self._gpu_lease = lease
        return lease

    def _release_job_lease(
        self,
        process: subprocess.Popen[bytes] | None,
        lease: GPULease | None,
    ) -> None:
        if process is not None and process.poll() is None:
            raise WorkerTerminationError(
                "validation worker must exit before its GPU lease is released"
            )
        authority = self.gpu_lease_manager
        if authority is not None and lease is not None:
            try:
                authority.release(lease)
            except StaleGPULeaseError:
                # A supervisor may already have reaped this exact dead owner.
                # Stale cleanup must never touch a replacement epoch.
                pass
        with self._condition:
            if self._process is process:
                self._process = None
            if self._gpu_lease is lease:
                self._gpu_lease = None

    def _record_cleanup_failure(
        self,
        job_id: str,
        error: BaseException,
        *,
        process: subprocess.Popen[bytes] | None,
        lease: GPULease | None,
    ) -> None:
        with self._condition:
            if self._active_job != job_id:
                return
            if process is not None:
                self._process = process
            if lease is not None:
                self._gpu_lease = lease
            code = (
                "WORKER_TERMINATION"
                if isinstance(error, WorkerTerminationError)
                else "GPU_LEASE_CLEANUP"
            )
            self._transition_locked(
                "cancelling",
                "termination_failed",
                self._progress,
                error={
                    "code": code,
                    "message": f"{type(error).__name__}: cleanup did not complete"[
                        :256
                    ],
                },
            )

    def _finish_job(
        self,
        job_id: str,
        status: str,
        stage: str,
        progress: int,
        *,
        error: dict[str, str] | None = None,
    ) -> None:
        with self._condition:
            if self._active_job == job_id:
                self._transition_locked(
                    status,
                    stage,
                    progress,
                    error=error,
                    clear_job=True,
                )

    @staticmethod
    def _terminate(process: subprocess.Popen[bytes]) -> None:
        if process.poll() is not None:
            return
        try:
            if os.name == "nt":
                process.send_signal(signal.CTRL_BREAK_EVENT)
            else:
                process.terminate()
            process.wait(timeout=0.75)
        except (OSError, subprocess.TimeoutExpired):
            pass
        if process.poll() is not None:
            return
        try:
            process.terminate()
            process.wait(timeout=1.0)
        except (OSError, subprocess.TimeoutExpired):
            pass
        if process.poll() is not None:
            return
        try:
            process.kill()
            process.wait(timeout=1.0)
        except (OSError, subprocess.TimeoutExpired):
            pass
        if process.poll() is None:
            raise WorkerTerminationError(
                "validation worker survived graceful, terminate, and kill escalation"
            )


class EvolutionController:
    """Run one cooperative Self-Harness cycle without blocking an HTTP worker."""

    def __init__(
        self,
        harness: Any,
        *,
        availability_check: Callable[[], bool] | None = None,
        worker_cleanup: Callable[[], None] | None = None,
    ) -> None:
        if not callable(getattr(harness, "tick", None)):
            raise TypeError("harness must provide tick()")
        if availability_check is not None and not callable(availability_check):
            raise TypeError("availability_check must be callable")
        if worker_cleanup is not None and not callable(worker_cleanup):
            raise TypeError("worker_cleanup must be callable")
        self.harness = harness
        self.availability_check = availability_check
        self.worker_cleanup = worker_cleanup
        self._condition = Condition(RLock())
        self._active = False
        self._sequence = 0
        self._job_id: str | None = None
        self._status = "ready"
        self._last_run: str | None = None
        self._last_result: dict[str, Any] | None = None
        self._error: dict[str, str] | None = None
        self._thread: Thread | None = None
        self._stopped = False

    @property
    def is_active(self) -> bool:
        with self._condition:
            return self._active

    def snapshot(self) -> dict[str, Any]:
        status = getattr(self.harness, "status", None)
        promotion_mode = getattr(status, "promotion_mode", "unknown")
        if isinstance(promotion_mode, Enum):
            promotion_mode = promotion_mode.value
        raw_integrity_errors = getattr(status, "proposal_integrity_errors", ())
        integrity_errors: list[dict[str, str]] = []
        if isinstance(raw_integrity_errors, (tuple, list)):
            for item in raw_integrity_errors[:64]:
                if (
                    isinstance(item, (tuple, list))
                    and len(item) == 2
                    and isinstance(item[0], str)
                    and isinstance(item[1], str)
                ):
                    integrity_errors.append(
                        {"proposal_id": item[0][:64], "reason": item[1][:512]}
                    )
        unreviewable = max(0, int(getattr(status, "unreviewable_proposals", 0) or 0))
        with self._condition:
            return {
                "running": self._active,
                "status": self._status,
                "seq": self._sequence,
                "job_id": self._job_id,
                "last_run": self._last_run,
                "last_result": deepcopy(self._last_result),
                "error": deepcopy(self._error),
                "sandbox": str(promotion_mode),
                "promotion_enabled": bool(getattr(status, "promotion_enabled", False)),
                "blocked_reason": getattr(status, "blocked_reason", None),
                "pending_proposals": int(getattr(status, "pending_proposals", 0) or 0),
                "rich_pending_proposals": int(
                    getattr(status, "rich_pending_proposals", 0) or 0
                ),
                "negative_proposals": int(
                    getattr(status, "negative_proposals", 0) or 0
                ),
                "unreviewable_proposals": unreviewable,
                "proposal_integrity_errors": integrity_errors,
                "integrity_degraded": unreviewable > 0,
                "evidence_failures": int(getattr(status, "evidence_failures", 0) or 0),
                "evidence_successes": int(
                    getattr(status, "evidence_successes", 0) or 0
                ),
                "evidence_capture_ratio": float(
                    getattr(status, "evidence_capture_ratio", 1.0) or 0.0
                ),
                "failures": self._failure_count(),
                "daemon_running": bool(getattr(status, "running", False)),
            }

    def proposal_review(self) -> dict[str, object]:
        """Return inert proposal diffs without exposing any mutation operation."""

        ledger = getattr(self.harness, "proposal_ledger", None)
        if ledger is None:
            raise ProposalReviewError("proposal ledger is unavailable")
        project_root = getattr(ledger, "project_root", None)
        proposals = getattr(self.harness, "evidence_proposals", None)
        reviewable = getattr(ledger, "reviewable_patches", None)
        if project_root is None or proposals is None or reviewable is None:
            raise ProposalReviewError("proposal review evidence is unavailable")
        try:
            reviewable_patches = dict(reviewable)
        except (TypeError, ValueError) as exc:
            raise ProposalReviewError("proposal review evidence is invalid") from exc
        return build_proposal_review(
            project_root,
            tuple(proposals),
            reviewable_patches,
        )

    def start(self) -> str:
        with self._condition:
            if self._stopped:
                raise RuntimeError("evolution controller is stopped")
            if self._active:
                raise EvolutionAlreadyRunningError("an evolution cycle is active")
            availability = self.availability_check
        if availability is not None and not availability():
            raise ComputeBusyError("local compute is owned by another mode")
        with self._condition:
            if self._stopped:
                raise RuntimeError("evolution controller is stopped")
            if self._active:
                raise EvolutionAlreadyRunningError("an evolution cycle is active")
            job_id = secrets.token_hex(12)
            self._active = True
            self._job_id = job_id
            self._status = "running"
            self._error = None
            self._sequence += 1
            thread = Thread(
                target=self._run,
                args=(job_id,),
                name=f"cogni-evolution-{job_id[:8]}",
                daemon=True,
            )
            self._thread = thread
            thread.start()
            return job_id

    def shutdown(self, timeout: float = 30.0) -> None:
        if timeout <= 0:
            raise ValueError("shutdown timeout must be positive")
        with self._condition:
            self._stopped = True
            thread = self._thread
        stop = getattr(self.harness, "stop", None)
        if callable(stop):
            stop()
        if thread is not None:
            thread.join(timeout=timeout)
        if thread is not None and thread.is_alive():
            raise WorkerTerminationError(
                "evolution thread did not stop within its shutdown deadline"
            )
        if self.worker_cleanup is not None:
            self.worker_cleanup()

    def _run(self, job_id: str) -> None:
        try:
            tick = self.harness.tick()
            result = self._tick_payload(tick)
            status = "succeeded" if result.get("decision") == "ran" else "skipped"
            error = None
        except BaseException as exc:
            result = None
            status = "failed"
            error = {
                "code": type(exc).__name__,
                "message": (str(exc) or "evolution cycle failed")[:512],
            }
        try:
            if self.worker_cleanup is not None:
                self.worker_cleanup()
        except BaseException as exc:
            result = None
            status = "failed"
            error = {
                "code": type(exc).__name__,
                "message": (str(exc) or "evolution worker cleanup failed")[:512],
            }
        with self._condition:
            if self._job_id != job_id:
                return
            self._active = False
            self._status = status
            self._last_result = result
            self._last_run = datetime.now(timezone.utc).isoformat()
            self._error = error
            self._sequence += 1
            self._condition.notify_all()

    def _failure_count(self) -> int:
        """Read only the bounded SQLite count; never materialize failure text."""

        logdb = getattr(self.harness, "logdb", None)
        connect = getattr(logdb, "_connect", None)
        if not callable(connect):
            return 0
        try:
            with connect() as database:
                row = database.execute("SELECT COUNT(*) FROM failures").fetchone()
            count = 0 if row is None else int(row[0])
            maximum = int(getattr(logdb, "max_failure_records", 100_000))
            return max(0, min(count, maximum, 100_000))
        except Exception:
            return 0

    @staticmethod
    def _tick_payload(tick: Any) -> dict[str, Any]:
        decision = getattr(tick, "decision", "unknown")
        if isinstance(decision, Enum):
            decision = decision.value
        payload: dict[str, Any] = {
            "decision": str(decision),
            "idle_for": float(getattr(tick, "idle_for", 0.0)),
        }
        report = getattr(tick, "result", None)
        if report is not None:
            payload["report"] = {
                "clusters": int(getattr(report, "clusters", 0)),
                "proposals": int(getattr(report, "proposals", 0)),
                "promoted": bool(getattr(report, "promoted", False)),
                "target": getattr(report, "target", None),
                "proposal_only": bool(getattr(report, "proposal_only", False)),
                "blocked_reason": getattr(report, "blocked_reason", None),
            }
        return payload


class _ServiceBackedPatchModel:
    """HF-shaped adapter that reuses the sole resident ModelService worker."""

    device = "cpu"

    def __init__(self, service: Any, tokenizer: Any) -> None:
        self.service = service
        self.tokenizer = tokenizer

    def eval(self) -> _ServiceBackedPatchModel:
        return self

    def generate(self, **kwargs: Any) -> Any:
        import torch

        input_ids = kwargs.get("input_ids")
        requested = kwargs.get("max_new_tokens")
        if (
            not isinstance(input_ids, torch.Tensor)
            or input_ids.ndim != 2
            or input_ids.shape[0] != 1
        ):
            raise ValueError("patch generation requires one bounded token sequence")
        if kwargs.get("use_cache") is not False or kwargs.get("do_sample") is not False:
            raise ValueError("patch generation must be deterministic and cache-free")
        if (
            not isinstance(requested, int)
            or isinstance(requested, bool)
            or requested <= 0
        ):
            raise ValueError("max_new_tokens must be a positive integer")
        for name, value in kwargs.items():
            if name in {"use_cache", "do_sample", "max_new_tokens"}:
                continue
            if not isinstance(value, torch.Tensor):
                raise TypeError(f"patch model argument {name} must be a tensor")
        prompt = self.tokenizer.decode(
            input_ids[0].detach().cpu().tolist(),
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        if not isinstance(prompt, str) or not prompt:
            raise ValueError("patch prompt could not be decoded")
        generated = self.service.generate(
            prompt,
            max_new_tokens=requested,
            decode_mode="strict",
        ).token_ids
        generated = generated.detach().to(
            device=input_ids.device, dtype=input_ids.dtype
        )
        return torch.cat((input_ids, generated.unsqueeze(0)), dim=-1)


def _write_product_checkpoint(project_root: Path) -> None:
    """Persist a bounded source digest checkpoint before an evolution cycle."""

    relative_paths = (
        "cogni_agent/manager.py",
        "cogni_core/search.py",
        "cogni_flow/production.py",
    )
    files: dict[str, str] = {}
    for relative in relative_paths:
        path = (project_root / relative).resolve(strict=True)
        if not path.is_file() or not path.is_relative_to(project_root):
            raise RuntimeError("checkpoint source escaped the project root")
        files[relative] = sha256(path.read_bytes()).hexdigest()
    state = project_root / ".cogni_state" / "self_harness"
    state.mkdir(parents=True, exist_ok=True)
    target = state / "pre-evolution-checkpoint.json"
    temporary = state / "pre-evolution-checkpoint.json.tmp"
    temporary.write_text(
        json.dumps(
            {
                "created_at": datetime.now(timezone.utc).isoformat(),
                "files": files,
            },
            ensure_ascii=False,
            separators=(",", ":"),
        ),
        encoding="utf-8",
    )
    os.replace(temporary, target)


class DemoHTTPServer(ThreadingHTTPServer):
    """HTTP server that refuses non-loopback binding at construction."""

    daemon_threads = True
    allow_reuse_address = False

    def __init__(
        self,
        manager: JobManager,
        asset_directory: str | Path,
        *,
        agent_manager: Any | None = None,
        evolution_manager: EvolutionController | Any | None = None,
        workspace_service: WorkspaceCapabilityService | Any | None = None,
        voice_service: LocalVoiceService | Any | None = None,
        port: int = DEFAULT_PORT,
        token: str | None = None,
        watchdog_timeout: float | None = DEFAULT_WATCHDOG_SECONDS,
    ) -> None:
        if not 0 <= int(port) <= 65535:
            raise ValueError("port must be in [0, 65535]")
        self.manager = manager
        self.agent_manager = agent_manager
        self.evolution_manager = evolution_manager
        self.workspace_service = workspace_service
        self.voice_service = voice_service or LocalVoiceService()
        self.asset_directory = Path(asset_directory).resolve(strict=True)
        if not self.asset_directory.is_dir():
            raise ValueError("asset_directory must be a directory")
        self.token = token or secrets.token_urlsafe(32)
        if len(self.token) < 32:
            raise ValueError("server token is too short")
        if watchdog_timeout is not None and watchdog_timeout <= 0:
            raise ValueError("watchdog_timeout must be positive or None")
        self.watchdog_timeout = (
            None if watchdog_timeout is None else float(watchdog_timeout)
        )
        self._lifecycle_lock = RLock()
        self._component_shutdown_condition = Condition(self._lifecycle_lock)
        self._compute_lock = RLock()
        self._watchdog_stop = Event()
        self._watchdog_thread: Thread | None = None
        self._last_state_poll = monotonic()
        self._shutdown_requested = False
        self._transport_shutdown_thread: Thread | None = None
        self._component_shutdown_state = _ComponentShutdownState.IDLE
        self._components_shutdown_owner: int | None = None
        self._components_shutdown_error: BaseException | None = None
        self.manager.availability_check = self._validation_compute_available
        if self.agent_manager is not None and hasattr(
            self.agent_manager, "availability_check"
        ):
            self.agent_manager.availability_check = self._agent_compute_available
        if self.evolution_manager is not None and hasattr(
            self.evolution_manager, "availability_check"
        ):
            self.evolution_manager.availability_check = (
                self._evolution_compute_available
            )
        super().__init__(("127.0.0.1", int(port)), DemoRequestHandler)

    @property
    def origin(self) -> str:
        return f"http://127.0.0.1:{self.server_port}"

    @property
    def bootstrap_url(self) -> str:
        return f"{self.origin}/?token={self.token}"

    @staticmethod
    def _has_explicit_image_parameter(target: object) -> bool:
        if not callable(target):
            return False
        try:
            parameter = signature(target).parameters.get("image_content")
        except (TypeError, ValueError):
            return False
        return parameter is not None and parameter.kind in {
            Parameter.POSITIONAL_OR_KEYWORD,
            Parameter.KEYWORD_ONLY,
        }

    def image_to_model_integration_ready(self) -> bool:
        """Return only answer-bearing, currently attested image readiness."""

        return bool(self.image_to_model_integration_status()["runtime_ready"])

    def image_to_model_integration_status(self) -> dict[str, object]:
        """Separate image configuration from processor and inference proof.

        A similarly shaped test double or arbitrary generation backend is not a
        product capability.  The exact production manager, ModelService, local
        Gemma factory, artifact digest and processor binding must all agree.
        Merely satisfying those structural checks is configuration evidence,
        not proof that an image processor or answer-bearing model inference ran
        successfully in this process.
        """

        configured = False
        processor_probed = False
        try:
            from cogni_agent.manager import AgentManager
            from cogni_agent.model_service import LocalGemmaModelFactory, ModelService

            agent = self.agent_manager
            if not isinstance(
                agent, AgentManager
            ) or not self._has_explicit_image_parameter(agent.start_turn):
                raise TypeError("production agent image boundary is not configured")
            service = getattr(agent, "model_service", None)
            if not isinstance(
                service, ModelService
            ) or not self._has_explicit_image_parameter(service.iter_generate_tokens):
                raise TypeError("production model image boundary is not configured")
            factory = getattr(service, "model_factory", None)
            if (
                not isinstance(factory, LocalGemmaModelFactory)
                or factory.manifest_path is None
                or service.artifact_digest != factory.artifact_digest
            ):
                raise TypeError("model artifact authority is not configured")
            processor = getattr(service, "_multimodal_processor_config", None)
            if not isinstance(processor, tuple) or len(processor) != 2:
                raise TypeError("multimodal processor is not configured")
            processor_root, processor_manifest = map(Path, processor)
            configured = processor_root == Path(
                factory.model_path
            ) and processor_manifest == Path(factory.manifest_path)
            # A lazily constructed processor object is not probe evidence: it
            # can exist even when preprocessing later fails.  A future guarded
            # validation path must supply an explicit successful probe and
            # answer-bearing inference attestation before runtime enablement.
            processor_probed = False
        except (AttributeError, ImportError, OSError, TypeError, ValueError):
            configured = False
            processor_probed = False
        return {
            "state": (
                "processor_probed_inference_unattested"
                if processor_probed
                else "configured_unverified"
                if configured
                else "not_configured"
            ),
            "selected_model_only": True,
            "configured": configured,
            "processor_probed": processor_probed,
            "model_inference_attested": False,
            "runtime_ready": False,
            "attestation_path": "guarded_current_process_model_inference_required",
            "disabled_reason": (
                "IMAGE_MODEL_INFERENCE_NOT_ATTESTED"
                if configured
                else "IMAGE_MODEL_NOT_CONFIGURED"
            ),
        }

    def touch_authenticated_state_poll(self) -> None:
        with self._lifecycle_lock:
            self._last_state_poll = monotonic()

    def start_watchdog(self) -> None:
        with self._lifecycle_lock:
            if self.watchdog_timeout is None or self._watchdog_thread is not None:
                return
            self._last_state_poll = monotonic()
            thread = Thread(
                target=self._watchdog_loop,
                name="cogni-demo-watchdog",
                daemon=True,
            )
            self._watchdog_thread = thread
            thread.start()

    def _watchdog_loop(self) -> None:
        assert self.watchdog_timeout is not None
        interval = min(1.0, max(0.02, self.watchdog_timeout / 4.0))
        while not self._watchdog_stop.wait(interval):
            with self._lifecycle_lock:
                elapsed = monotonic() - self._last_state_poll
            if elapsed >= self.watchdog_timeout:
                self.request_shutdown()
                return

    def serve_forever(self, poll_interval: float = 0.5) -> None:
        self.start_watchdog()
        super().serve_forever(poll_interval=poll_interval)

    def start_validation(self, prompt: str) -> str:
        """Acquire the sole CUDA slot and start the existing validation worker."""

        with self._compute_lock:
            self._require_admission_open()
            if self._agent_active() or self._evolution_active():
                raise ComputeBusyError("agent or evolution owns local compute")
            if self.agent_manager is not None:
                # A resident but idle chat model still owns the model VRAM.
                # Stop it before the validation child is allowed to load Gemma.
                self.agent_manager.stop_model()
            return self.manager.start(prompt)

    def start_agent_turn(
        self,
        message: str,
        mode: str,
        *,
        evidence: tuple[Any, ...] = (),
        image_content: bytes | None = None,
        retrieval_requested: bool = False,
    ) -> str:
        with self._compute_lock:
            self._require_admission_open()
            if self.agent_manager is None:
                raise RuntimeError("agent manager is unavailable")
            if self.manager.is_active or self._evolution_active():
                raise ComputeBusyError("validation or evolution owns local compute")
            if type(retrieval_requested) is not bool:
                raise TypeError("retrieval_requested must be a bool")
            if type(evidence) is not tuple:
                raise TypeError("evidence must be an immutable tuple")
            effective_retrieval = retrieval_requested or bool(evidence)
            if effective_retrieval and image_content is not None:
                raise ValueError("retrieval and image content are mutually exclusive")
            if evidence:
                # Product callers may bypass the HTTP adapter.  Do not let
                # unprovenanced generic AgentManager evidence acquire the
                # answer-bearing RAG authority at this boundary, including
                # when the caller omits ``retrieval_requested``.
                for item in evidence:
                    if not _product_retrieval_evidence_matches_current_source(
                        item, self.workspace_service
                    ):
                        raise ValueError("retrieval evidence lacks admitted provenance")
            if effective_retrieval:
                return self.agent_manager.start_turn(
                    message,
                    mode,
                    evidence=evidence,
                    retrieval_requested=True,
                )
            if image_content is not None:
                return self.agent_manager.start_turn(
                    message,
                    mode,
                    evidence=evidence,
                    image_content=image_content,
                )
            return self.agent_manager.start_turn(message, mode)

    def transcribe_voice(
        self, audio_wav_base64: object, *, language: object = "auto"
    ) -> dict[str, object]:
        """Run resident-model STT only while it exclusively owns compute."""

        with self._compute_lock:
            self._require_admission_open()
            if (
                self.manager.is_active
                or self._agent_active()
                or self._evolution_active()
            ):
                raise ComputeBusyError("local compute is already in use")
            return self.voice_service.transcribe_base64(
                audio_wav_base64,
                language=language,
            )

    def synthesize_voice(
        self, text: object, *, language: object = "auto"
    ) -> dict[str, object]:
        """Create bounded local speech without overlapping product work."""

        with self._compute_lock:
            self._require_admission_open()
            if (
                self.manager.is_active
                or self._agent_active()
                or self._evolution_active()
            ):
                raise ComputeBusyError("local compute is already in use")
            return self.voice_service.synthesize(text, language=language)

    def start_evolution(self) -> str:
        with self._compute_lock:
            self._require_admission_open()
            if self.evolution_manager is None:
                raise RuntimeError("evolution manager is unavailable")
            if self.manager.is_active or self._agent_active():
                raise ComputeBusyError("validation or agent owns local compute")
            if self.agent_manager is not None:
                self.agent_manager.stop_model()
            return self.evolution_manager.start()

    def _require_admission_open(self) -> None:
        with self._lifecycle_lock:
            if self._shutdown_requested:
                raise ComputeBusyError("local compute is shutting down")

    def _agent_active(self) -> bool:
        return bool(
            self.agent_manager is not None
            and getattr(self.agent_manager, "is_active", False)
        )

    def _evolution_active(self) -> bool:
        return bool(
            self.evolution_manager is not None
            and getattr(self.evolution_manager, "is_active", False)
        )

    def _validation_compute_available(self) -> bool:
        return not self._agent_active() and not self._evolution_active()

    def _agent_compute_available(self) -> bool:
        return not self.manager.is_active and not self._evolution_active()

    def _evolution_compute_available(self) -> bool:
        return not self.manager.is_active and not self._agent_active()

    def request_shutdown(self) -> None:
        """Close admission immediately and trigger transport shutdown once."""

        with self._component_shutdown_condition:
            self._shutdown_requested = True
            self._watchdog_stop.set()
            if self._transport_shutdown_thread is not None:
                return

            def stop() -> None:
                _run_best_effort_cleanup((self.shutdown_components, self.shutdown))

            thread = Thread(
                target=stop,
                name="cogni-demo-shutdown",
                daemon=True,
            )
            self._transport_shutdown_thread = thread
            try:
                # Publish and start under one short lifecycle critical section.
                # The child only waits for this lock; it never owns compute yet.
                thread.start()
            except BaseException:
                # Admission stays closed, but a later request may retry transport.
                self._transport_shutdown_thread = None
                raise

    def server_close(self) -> None:
        self._watchdog_stop.set()
        close_transport = super().server_close
        _run_best_effort_cleanup((self.shutdown_components, close_transport))

    def shutdown_components(
        self,
        *,
        wait_timeout: float = COMPONENT_SHUTDOWN_WAIT_SECONDS,
    ) -> None:
        """Run component cleanup once; concurrent callers await the same result."""

        if (
            isinstance(wait_timeout, bool)
            or not 0.0 < float(wait_timeout) <= COMPONENT_SHUTDOWN_WAIT_SECONDS
        ):
            raise ValueError("component shutdown wait_timeout is outside its bound")
        caller = get_ident()
        deadline = monotonic() + float(wait_timeout)
        owns_cleanup = False
        must_wait = False
        # Admission closes before the compute fence. Never hold lifecycle while
        # acquiring compute: admitted starts briefly take lifecycle while they
        # already own compute, so reversing that order would deadlock.
        with self._component_shutdown_condition:
            self._shutdown_requested = True
            self._watchdog_stop.set()
            if self._component_shutdown_state is _ComponentShutdownState.DONE:
                failure = self._components_shutdown_error
            elif self._component_shutdown_state is _ComponentShutdownState.RUNNING:
                if self._components_shutdown_owner == caller:
                    return
                failure = None
                must_wait = True
            else:
                self._component_shutdown_state = _ComponentShutdownState.RUNNING
                self._components_shutdown_owner = caller
                failure = None
                owns_cleanup = True

        if must_wait:
            with self._component_shutdown_condition:
                while (
                    self._component_shutdown_state is not _ComponentShutdownState.DONE
                ):
                    remaining = deadline - monotonic()
                    if remaining <= 0.0:
                        raise WorkerTerminationError(
                            "component shutdown did not complete within its wait bound"
                        )
                    self._component_shutdown_condition.wait(timeout=remaining)
                failure = self._components_shutdown_error

        if not owns_cleanup:
            if failure is not None:
                raise failure
            return

        acquired_compute = False
        try:
            remaining = deadline - monotonic()
            if remaining <= 0.0 or not self._compute_lock.acquire(timeout=remaining):
                raise WorkerTerminationError(
                    "component shutdown could not acquire compute within its wait bound"
                )
            acquired_compute = True
            # This acquire/release is a quiescence fence. Admission is closed,
            # so callbacks may safely re-enter shutdown after it is released.
            self._compute_lock.release()
            acquired_compute = False
            _shutdown_product_components(
                self.manager,
                self.evolution_manager,
                self.agent_manager,
            )
        except BaseException as error:
            failure = error
        finally:
            if acquired_compute:
                self._compute_lock.release()
            with self._component_shutdown_condition:
                self._components_shutdown_error = failure
                self._component_shutdown_state = _ComponentShutdownState.DONE
                self._components_shutdown_owner = None
                self._component_shutdown_condition.notify_all()
        if failure is not None:
            raise failure


class DemoRequestHandler(BaseHTTPRequestHandler):
    """Exact-route handler; the repository is never exposed as a file tree."""

    server: DemoHTTPServer
    protocol_version = "HTTP/1.1"

    def log_message(self, _format: str, *_args: object) -> None:
        return

    def do_GET(self) -> None:  # noqa: N802
        if not self._valid_host():
            self._json_error(HTTPStatus.BAD_REQUEST, "INVALID_HOST")
            return
        parsed = urlsplit(self.path)
        if parsed.path == "/api/ping" and not parsed.query:
            self._json(
                HTTPStatus.OK,
                {"service": SERVICE_MARKER, "protocol": SESSION_VERSION},
            )
            return
        if parsed.path == "/" and self._bootstrap(parsed.query):
            return
        if not self._authenticated():
            self._json_error(HTTPStatus.FORBIDDEN, "AUTH_REQUIRED")
            return
        if parsed.path == "/api/state":
            self._state(parsed.query)
            return
        if parsed.path == "/api/agent/state":
            self._agent_state(parsed.query)
            return
        if parsed.path == "/api/evolution/proposals" and not parsed.query:
            manager = self.server.evolution_manager
            review = getattr(manager, "proposal_review", None)
            if manager is None or not callable(review):
                self._json_error(
                    HTTPStatus.SERVICE_UNAVAILABLE, "EVOLUTION_UNAVAILABLE"
                )
                return
            self.server.touch_authenticated_state_poll()
            try:
                payload = review()
            except ProposalReviewError:
                self._json_error(
                    HTTPStatus.CONFLICT, "PROPOSAL_REVIEW_INTEGRITY_FAILED"
                )
                return
            except (OSError, RuntimeError, TypeError, ValueError):
                self._json_error(
                    HTTPStatus.SERVICE_UNAVAILABLE, "PROPOSAL_REVIEW_UNAVAILABLE"
                )
                return
            items = payload.get("items") if isinstance(payload, dict) else None
            if (
                not isinstance(payload, dict)
                or payload.get("mode") != "proposal_only_read_only"
                or payload.get("mutation_endpoint") is not False
                or payload.get("execution_endpoint") is not False
                or not isinstance(items, list)
                or len(items) > 8
            ):
                self._json_error(
                    HTTPStatus.SERVICE_UNAVAILABLE, "PROPOSAL_REVIEW_UNAVAILABLE"
                )
                return
            self._json(HTTPStatus.OK, payload)
            return
        if parsed.path == "/api/workspace/capabilities" and not parsed.query:
            if self.server.workspace_service is None:
                self._json_error(
                    HTTPStatus.SERVICE_UNAVAILABLE, "WORKSPACE_UNAVAILABLE"
                )
                return
            self.server.touch_authenticated_state_poll()
            payload = self.server.workspace_service.capability_payload()
            if not isinstance(payload, dict):
                self._json_error(
                    HTTPStatus.SERVICE_UNAVAILABLE, "WORKSPACE_UNAVAILABLE"
                )
                return
            # Browser capture, loopback transport and optional local STT are a
            # separate boundary from attachment/RAG capabilities.  Publish the
            # live voice service state so an audio-advertising checkpoint is
            # never mistaken for a working transcription artifact.
            payload = deepcopy(payload)
            models = payload.get("models")
            model_items = models.get("items") if isinstance(models, dict) else None
            selected_items = (
                [
                    item
                    for item in model_items
                    if isinstance(item, dict) and item.get("selected") is True
                ]
                if isinstance(model_items, list)
                else []
            )
            selected_model = selected_items[0] if len(selected_items) == 1 else None
            checkpoint_image = bool(
                isinstance(selected_model, dict)
                and isinstance(selected_model.get("checkpoint_modalities"), list)
                and "image" in selected_model["checkpoint_modalities"]
            )
            image_capability = self.server.image_to_model_integration_status()
            if not checkpoint_image:
                image_capability = {
                    **image_capability,
                    "state": "selected_checkpoint_not_supported",
                    "configured": False,
                    "processor_probed": False,
                    "model_inference_attested": False,
                    "runtime_ready": False,
                    "disabled_reason": "SELECTED_MODEL_IMAGE_NOT_ADVERTISED",
                }
            image_integration = bool(image_capability.get("runtime_ready") is True)
            attachment_capability = payload.get("attachments")
            if isinstance(attachment_capability, dict):
                attachment_capability["image_to_model_integration"] = image_integration
                attachment_capability["image_capability"] = image_capability
                attachment_capability["image_selection"] = (
                    "explicit_single_next_turn" if image_integration else "disabled"
                )
            if isinstance(models, dict):
                models["image_to_model_integration"] = image_integration
                models["image_capability"] = image_capability
            if image_integration and isinstance(selected_model, dict):
                runtime_modalities = selected_model.get("runtime_input_modalities")
                if (
                    isinstance(runtime_modalities, list)
                    and "image" not in runtime_modalities
                ):
                    runtime_modalities.append("image")
                for key in (
                    "advertised_but_not_wired",
                    "unwired_checkpoint_modalities",
                ):
                    unwired = selected_model.get(key)
                    if isinstance(unwired, list):
                        selected_model[key] = [
                            modality for modality in unwired if modality != "image"
                        ]
            checkpoint_microphone = payload.get("microphone")
            voice = self.server.voice_service.capability_payload()
            if isinstance(checkpoint_microphone, dict):
                voice["checkpoint_advertises_audio"] = bool(
                    checkpoint_microphone.get("checkpoint_advertises_audio", False)
                )
            payload["microphone"] = voice
            self._json(HTTPStatus.OK, payload)
            return
        if parsed.path == "/api/workspace/attachments" and not parsed.query:
            if self.server.workspace_service is None:
                self._json_error(
                    HTTPStatus.SERVICE_UNAVAILABLE, "WORKSPACE_UNAVAILABLE"
                )
                return
            self.server.touch_authenticated_state_poll()
            self._json(HTTPStatus.OK, self.server.workspace_service.list_attachments())
            return
        if parsed.path in {
            "/api/workspace/attachments/preview",
            "/api/workspace/attachments/content",
        }:
            workspace = self.server.workspace_service
            if workspace is None:
                self._json_error(
                    HTTPStatus.SERVICE_UNAVAILABLE, "WORKSPACE_UNAVAILABLE"
                )
                return
            values = parse_qs(parsed.query, keep_blank_values=True)
            if set(values) != {"attachment_id"} or len(values["attachment_id"]) != 1:
                self._json_error(HTTPStatus.BAD_REQUEST, "INVALID_QUERY")
                return
            attachment_id = values["attachment_id"][0]
            try:
                if parsed.path.endswith("/preview"):
                    payload = workspace.preview_attachment(attachment_id)
                    self._json(HTTPStatus.OK, payload)
                else:
                    content, media_type = workspace.image_attachment_content(
                        attachment_id
                    )
                    self._send(HTTPStatus.OK, content, media_type)
            except WorkspaceCapabilityError as exc:
                self._workspace_error(exc)
            return
        if parsed.path == "/api/workspace/rag/source":
            workspace = self.server.workspace_service
            if workspace is None:
                self._json_error(
                    HTTPStatus.SERVICE_UNAVAILABLE, "WORKSPACE_UNAVAILABLE"
                )
                return
            # This evidence URL has one canonical representation: attachment_id
            # first, then chunk_index.  Matching the raw query (rather than a
            # percent-decoded mapping) rejects encoded unreserved characters,
            # reordered/duplicate keys, empty separators and trailing data.
            source_query = re.fullmatch(
                r"attachment_id=([0-9a-f]{24})&chunk_index=(0|[1-9][0-9]{0,2})",
                parsed.query,
            )
            if source_query is None:
                self._json_error(HTTPStatus.BAD_REQUEST, "INVALID_QUERY")
                return
            attachment_id, raw_chunk_index = source_query.groups()
            chunk_index = int(raw_chunk_index)
            if chunk_index >= 128:
                self._json_error(HTTPStatus.BAD_REQUEST, "INVALID_QUERY")
                return
            self.server.touch_authenticated_state_poll()
            try:
                payload = workspace.preview_rag_source(attachment_id, chunk_index)
            except WorkspaceCapabilityError as exc:
                self._workspace_error(exc)
                return
            except (OSError, RuntimeError, TypeError, ValueError):
                self._json_error(
                    HTTPStatus.SERVICE_UNAVAILABLE, "WORKSPACE_RESPONSE_INVALID"
                )
                return
            if not _rag_source_payload_is_valid(
                payload, attachment_id=attachment_id, chunk_index=chunk_index
            ):
                self._json_error(
                    HTTPStatus.SERVICE_UNAVAILABLE, "WORKSPACE_RESPONSE_INVALID"
                )
                return
            self._json(HTTPStatus.OK, payload)
            return
        asset = _STATIC_ASSETS.get(parsed.path)
        if asset is None or parsed.query:
            self._json_error(HTTPStatus.NOT_FOUND, "NOT_FOUND")
            return
        filename, content_type = asset
        candidate = self.server.asset_directory / filename
        try:
            resolved = candidate.resolve(strict=True)
        except FileNotFoundError:
            self._json_error(HTTPStatus.NOT_FOUND, "ASSET_MISSING")
            return
        if (
            candidate.is_symlink()
            or not resolved.is_relative_to(self.server.asset_directory)
            or not resolved.is_file()
            or resolved.stat().st_size > MAX_ASSET_BYTES
        ):
            self._json_error(HTTPStatus.NOT_FOUND, "ASSET_MISSING")
            return
        self._send(HTTPStatus.OK, resolved.read_bytes(), content_type)

    def do_POST(self) -> None:  # noqa: N802
        from cogni_agent.manager import AgentBusyError, NoActiveAgentTurnError

        if not self._valid_host():
            self._json_error(HTTPStatus.BAD_REQUEST, "INVALID_HOST")
            return
        if not self._authenticated() or not self._valid_origin():
            self._json_error(HTTPStatus.FORBIDDEN, "AUTH_REQUIRED")
            return
        parsed = urlsplit(self.path)
        if parsed.query or parsed.path not in {
            "/api/run",
            "/api/cancel",
            "/api/shutdown",
            "/api/agent/chat",
            "/api/agent/cancel",
            "/api/agent/reset",
            "/api/evolution/run",
            "/api/workspace/attachments/add",
            "/api/workspace/attachments/delete",
            "/api/workspace/rag/index",
            "/api/workspace/rag/reindex",
            "/api/workspace/rag/query",
            "/api/workspace/lens/search",
            "/api/workspace/lens/search-and-index",
            "/api/workspace/models/select",
            "/api/workspace/voice/transcribe",
            "/api/workspace/voice/synthesize",
        }:
            self._json_error(HTTPStatus.NOT_FOUND, "NOT_FOUND")
            return
        body_limit = {
            "/api/agent/chat": MAX_AGENT_PROJECT_REQUEST_BODY_BYTES,
            "/api/workspace/attachments/add": MAX_ATTACHMENT_REQUEST_BODY_BYTES,
            "/api/workspace/voice/transcribe": MAX_VOICE_REQUEST_BODY_BYTES,
            "/api/workspace/voice/synthesize": MAX_TTS_REQUEST_BODY_BYTES,
        }.get(parsed.path, MAX_REQUEST_BODY_BYTES)
        body = self._read_json_body(maximum_bytes=body_limit)
        if body is None:
            return
        if parsed.path.startswith("/api/workspace/"):
            self._workspace_post(parsed.path, body)
            return
        if parsed.path == "/api/run":
            if not set(body) <= {"prompt"}:
                self._json_error(HTTPStatus.BAD_REQUEST, "INVALID_BODY")
                return
            try:
                job_id = self.server.start_validation(body.get("prompt", ""))
            except (JobAlreadyRunningError, AgentBusyError, ComputeBusyError):
                self._json_error(HTTPStatus.CONFLICT, "JOB_ALREADY_RUNNING")
                return
            except (TypeError, ValueError):
                self._json_error(HTTPStatus.BAD_REQUEST, "INVALID_BODY")
                return
            self._json(
                HTTPStatus.ACCEPTED,
                {"job_id": job_id, **self.server.manager.snapshot()},
            )
            return
        if parsed.path == "/api/agent/chat":
            if self.server.agent_manager is None:
                self._json_error(HTTPStatus.SERVICE_UNAVAILABLE, "AGENT_UNAVAILABLE")
                return
            envelope_message = body.get("message")
            envelope_mode = body.get("mode", "chat")
            envelope_project = False
            if isinstance(envelope_message, str) and envelope_message.strip():
                envelope_first_token = (
                    envelope_message.strip().partition("\n")[0].split(maxsplit=1)[0]
                )
                envelope_project = (
                    envelope_mode == "task"
                    and envelope_first_token.casefold() == "/project"
                )
            request_body_bytes = int(self.headers["Content-Length"])
            if (
                request_body_bytes > MAX_AGENT_CHAT_REQUEST_BODY_BYTES
                and not envelope_project
            ):
                self._json_error(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, "BODY_TOO_LARGE")
                return
            if (
                not set(body) <= {"message", "mode", "rag", "image_attachment_id"}
                or "message" not in body
            ):
                self._json_error(HTTPStatus.BAD_REQUEST, "INVALID_BODY")
                return
            rag_requested = body.get("rag", False)
            if not isinstance(rag_requested, bool):
                self._json_error(HTTPStatus.BAD_REQUEST, "INVALID_BODY")
                return
            mode = body.get("mode", "chat")
            image_attachment_id = body.get("image_attachment_id")
            image_requested = "image_attachment_id" in body
            if image_requested and (
                not isinstance(image_attachment_id, str)
                or re.fullmatch(r"[0-9a-f]{24}", image_attachment_id) is None
                or rag_requested
                or mode != "chat"
            ):
                self._json_error(HTTPStatus.BAD_REQUEST, "INVALID_BODY")
                return
            message = body.get("message")
            if (
                not isinstance(message, str)
                or not message.strip()
                or any(
                    ord(character) < 32 and character not in "\t\r\n"
                    for character in message
                )
            ):
                self._json_error(HTTPStatus.BAD_REQUEST, "INVALID_BODY")
                return
            stripped_message = message.strip()
            first_token = stripped_message.partition("\n")[0].split(maxsplit=1)[0]
            project_candidate = first_token.casefold() == "/project"
            if project_candidate:
                if mode != "task" or rag_requested or image_requested:
                    self._json_error(HTTPStatus.BAD_REQUEST, "INVALID_BODY")
                    return
                if len(stripped_message.encode("utf-8")) > MAX_PROJECT_COMMAND_BYTES:
                    self._json_error(
                        HTTPStatus.REQUEST_ENTITY_TOO_LARGE, "BODY_TOO_LARGE"
                    )
                    return
                try:
                    project_request = parse_tool_request(stripped_message)
                except (ToolPolicyError, TypeError, UnicodeError):
                    self._json_error(HTTPStatus.BAD_REQUEST, "INVALID_BODY")
                    return
                if project_request is None or project_request.operation != "project":
                    self._json_error(HTTPStatus.BAD_REQUEST, "INVALID_BODY")
                    return
            elif len(stripped_message) > MAX_AGENT_CHAT_MESSAGE_CHARS:
                self._json_error(HTTPStatus.BAD_REQUEST, "INVALID_BODY")
                return
            evidence: tuple[Any, ...] = ()
            image_content: bytes | None = None
            image_media_type: str | None = None
            if rag_requested:
                if self.server.workspace_service is None:
                    self._json_error(
                        HTTPStatus.SERVICE_UNAVAILABLE, "WORKSPACE_UNAVAILABLE"
                    )
                    return
                try:
                    rag_query = message.strip()[:MAX_RAG_QUERY_CHARS]
                    retrieved = self.server.workspace_service.query_rag(
                        rag_query, limit=5
                    )
                    evidence = self._retrieval_evidence(
                        retrieved, expected_query=rag_query
                    )
                except WorkspaceCapabilityError as exc:
                    self._workspace_error(exc)
                    return
            if image_requested:
                workspace = self.server.workspace_service
                if workspace is None:
                    self._json_error(
                        HTTPStatus.SERVICE_UNAVAILABLE, "WORKSPACE_UNAVAILABLE"
                    )
                    return
                if not self.server.image_to_model_integration_ready():
                    self._json_error(
                        HTTPStatus.SERVICE_UNAVAILABLE, "IMAGE_MODEL_UNAVAILABLE"
                    )
                    return
                try:
                    image_content, image_media_type = (
                        workspace.image_attachment_content(image_attachment_id)
                    )
                except WorkspaceCapabilityError as exc:
                    self._workspace_error(exc)
                    return
                except (OSError, TypeError, ValueError):
                    self._json_error(
                        HTTPStatus.BAD_REQUEST, "WORKSPACE_RESPONSE_INVALID"
                    )
                    return
                if (
                    type(image_content) is not bytes
                    or not 1 <= len(image_content) <= MAX_ATTACHMENT_BYTES
                    or image_media_type not in {"image/png", "image/jpeg", "image/webp"}
                ):
                    self._json_error(
                        HTTPStatus.BAD_REQUEST, "WORKSPACE_RESPONSE_INVALID"
                    )
                    return
            try:
                turn_id = self.server.start_agent_turn(
                    message,
                    mode,
                    evidence=evidence,
                    image_content=image_content,
                    retrieval_requested=rag_requested,
                )
            except (AgentBusyError, ComputeBusyError):
                self._json_error(HTTPStatus.CONFLICT, "COMPUTE_BUSY")
                return
            except (TypeError, ValueError):
                self._json_error(HTTPStatus.BAD_REQUEST, "INVALID_BODY")
                return
            self._json(
                HTTPStatus.ACCEPTED,
                {
                    "turn_id": turn_id,
                    "rag_requested": rag_requested,
                    "rag_evidence_count": len(evidence),
                    "image_requested": image_requested,
                    "image_input_admitted": image_content is not None,
                    "image_media_type": image_media_type,
                    **self.server.agent_manager.snapshot(),
                },
            )
            return
        if body:
            self._json_error(HTTPStatus.BAD_REQUEST, "INVALID_BODY")
            return
        if parsed.path == "/api/cancel":
            try:
                self.server.manager.cancel()
            except NoActiveJobError:
                self._json_error(HTTPStatus.CONFLICT, "NO_ACTIVE_JOB")
                return
            self._json(HTTPStatus.ACCEPTED, self.server.manager.snapshot())
            return
        if parsed.path == "/api/agent/cancel":
            if self.server.agent_manager is None:
                self._json_error(HTTPStatus.SERVICE_UNAVAILABLE, "AGENT_UNAVAILABLE")
                return
            try:
                self.server.agent_manager.cancel()
            except NoActiveAgentTurnError:
                self._json_error(HTTPStatus.CONFLICT, "NO_ACTIVE_AGENT_TURN")
                return
            self._json(HTTPStatus.ACCEPTED, self.server.agent_manager.snapshot())
            return
        if parsed.path == "/api/agent/reset":
            if self.server.agent_manager is None:
                self._json_error(HTTPStatus.SERVICE_UNAVAILABLE, "AGENT_UNAVAILABLE")
                return
            try:
                self.server.agent_manager.reset()
            except AgentBusyError:
                self._json_error(HTTPStatus.CONFLICT, "AGENT_BUSY")
                return
            self._json(HTTPStatus.OK, self.server.agent_manager.snapshot())
            return
        if parsed.path == "/api/evolution/run":
            if self.server.evolution_manager is None:
                self._json_error(
                    HTTPStatus.SERVICE_UNAVAILABLE, "EVOLUTION_UNAVAILABLE"
                )
                return
            try:
                job_id = self.server.start_evolution()
            except (EvolutionAlreadyRunningError, ComputeBusyError):
                self._json_error(HTTPStatus.CONFLICT, "COMPUTE_BUSY")
                return
            self._json(
                HTTPStatus.ACCEPTED,
                {"job_id": job_id, **self.server.evolution_manager.snapshot()},
            )
            return
        self._json(HTTPStatus.ACCEPTED, {"status": "shutting_down"})
        self.server.request_shutdown()

    def _bootstrap(self, query: str) -> bool:
        supplied = parse_qs(query, keep_blank_values=True)
        values = supplied.get("token", [])
        if set(supplied) != {"token"} or len(values) != 1:
            return False
        if not hmac.compare_digest(values[0], self.server.token):
            self._json_error(HTTPStatus.FORBIDDEN, "AUTH_REQUIRED")
            return True
        cookie = (
            f"CogniDemo={self.server.token}; Path=/; HttpOnly; "
            "SameSite=Strict; Max-Age=3600"
        )
        self.send_response(HTTPStatus.SEE_OTHER)
        self._security_headers()
        self.send_header("Set-Cookie", cookie)
        self.send_header("Location", "/")
        self.send_header("Content-Length", "0")
        self.end_headers()
        return True

    def _valid_host(self) -> bool:
        expected = f"127.0.0.1:{self.server.server_port}"
        return hmac.compare_digest(self.headers.get("Host", ""), expected)

    def _valid_origin(self) -> bool:
        return hmac.compare_digest(self.headers.get("Origin", ""), self.server.origin)

    def _authenticated(self) -> bool:
        raw = self.headers.get("Cookie", "")
        try:
            cookie = SimpleCookie(raw)
            morsel = cookie.get("CogniDemo")
        except Exception:
            return False
        return morsel is not None and hmac.compare_digest(
            morsel.value, self.server.token
        )

    def _state(self, query: str) -> None:
        values = parse_qs(query, keep_blank_values=True)
        if not values:
            self.server.touch_authenticated_state_poll()
            self._json(HTTPStatus.OK, self.server.manager.snapshot())
            return
        if set(values) != {"after"} or len(values["after"]) != 1:
            self._json_error(HTTPStatus.BAD_REQUEST, "INVALID_QUERY")
            return
        try:
            after = int(values["after"][0])
            if after < 0:
                raise ValueError
            self.server.touch_authenticated_state_poll()
            state = self.server.manager.wait_snapshot(after)
        except (TypeError, ValueError):
            self._json_error(HTTPStatus.BAD_REQUEST, "INVALID_QUERY")
            return
        self._json(HTTPStatus.OK, state)

    def _agent_state(self, query: str) -> None:
        manager = self.server.agent_manager
        if manager is None:
            self._json_error(HTTPStatus.SERVICE_UNAVAILABLE, "AGENT_UNAVAILABLE")
            return
        values = parse_qs(query, keep_blank_values=True)
        if not values:
            self.server.touch_authenticated_state_poll()
            self._json(HTTPStatus.OK, manager.snapshot())
            return
        if set(values) != {"after"} or len(values["after"]) != 1:
            self._json_error(HTTPStatus.BAD_REQUEST, "INVALID_QUERY")
            return
        try:
            after = int(values["after"][0])
            if after < 0:
                raise ValueError
            self.server.touch_authenticated_state_poll()
            state = manager.wait_snapshot(after)
        except (TypeError, ValueError):
            self._json_error(HTTPStatus.BAD_REQUEST, "INVALID_QUERY")
            return
        self._json(HTTPStatus.OK, state)

    def _workspace_post(self, path: str, body: dict[str, Any]) -> None:
        try:
            if path == "/api/workspace/voice/transcribe":
                if set(body) not in (
                    {"audio_wav_base64"},
                    {"audio_wav_base64", "language"},
                ):
                    raise WorkspaceCapabilityError(
                        "INVALID_BODY", "voice transcription body fields are invalid"
                    )
                self._json(
                    HTTPStatus.OK,
                    self.server.transcribe_voice(
                        body["audio_wav_base64"],
                        language=body.get("language", "auto"),
                    ),
                )
                return
            if path == "/api/workspace/voice/synthesize":
                if set(body) not in ({"text"}, {"text", "language"}):
                    raise WorkspaceCapabilityError(
                        "INVALID_BODY", "voice synthesis body fields are invalid"
                    )
                self._json(
                    HTTPStatus.OK,
                    self.server.synthesize_voice(
                        body["text"],
                        language=body.get("language", "auto"),
                    ),
                )
                return
            workspace = self.server.workspace_service
            if workspace is None:
                self._json_error(
                    HTTPStatus.SERVICE_UNAVAILABLE, "WORKSPACE_UNAVAILABLE"
                )
                return
            if path == "/api/workspace/attachments/add":
                if set(body) != {"name", "media_type", "content_base64"}:
                    raise WorkspaceCapabilityError(
                        "INVALID_BODY", "attachment body fields are invalid"
                    )
                payload = workspace.add_attachment(
                    name=body["name"],
                    media_type=body["media_type"],
                    content_base64=body["content_base64"],
                )
                self._json(HTTPStatus.CREATED, payload)
                return
            if path == "/api/workspace/attachments/delete":
                if set(body) != {"attachment_id"}:
                    raise WorkspaceCapabilityError(
                        "INVALID_BODY", "attachment deletion body fields are invalid"
                    )
                payload = workspace.delete_attachment(body["attachment_id"])
                self._json(HTTPStatus.OK, payload)
                return
            if path == "/api/workspace/rag/index":
                if set(body) != {"attachment_ids"}:
                    raise WorkspaceCapabilityError(
                        "INVALID_BODY", "RAG index body fields are invalid"
                    )
                payload = workspace.index_attachments(body["attachment_ids"])
                self._json(HTTPStatus.OK, payload)
                return
            if path == "/api/workspace/rag/reindex":
                if set(body) != {"attachment_ids"}:
                    raise WorkspaceCapabilityError(
                        "INVALID_BODY", "RAG reindex body fields are invalid"
                    )
                payload = workspace.reindex_attachments(body["attachment_ids"])
                self._json(HTTPStatus.OK, payload)
                return
            if path == "/api/workspace/rag/query":
                if set(body) not in ({"query"}, {"query", "limit"}):
                    raise WorkspaceCapabilityError(
                        "INVALID_BODY", "RAG query body fields are invalid"
                    )
                payload = workspace.query_rag(body["query"], limit=body.get("limit", 5))
                self._json(HTTPStatus.OK, payload)
                return
            if path in {
                "/api/workspace/lens/search",
                "/api/workspace/lens/search-and-index",
            }:
                if set(body) not in (
                    {"kind", "query"},
                    {"kind", "query", "limit"},
                ):
                    raise WorkspaceCapabilityError(
                        "INVALID_BODY", "Lens search body fields are invalid"
                    )
                payload = workspace.search_lens(
                    body["kind"],
                    body["query"],
                    limit=body.get("limit", 5),
                    index_in_akasicdb=(path == "/api/workspace/lens/search-and-index"),
                )
                self._json(HTTPStatus.OK, payload)
                return
            if path == "/api/workspace/models/select":
                if set(body) != {"model_id"}:
                    raise WorkspaceCapabilityError(
                        "INVALID_BODY", "model selection body fields are invalid"
                    )
                self._json(HTTPStatus.OK, workspace.select_model(body["model_id"]))
                return
        except (KeyError, TypeError):
            self._json_error(HTTPStatus.BAD_REQUEST, "INVALID_BODY")
            return
        except ComputeBusyError:
            self._json_error(HTTPStatus.CONFLICT, "COMPUTE_BUSY")
            return
        except LocalVoiceError as exc:
            self._voice_error(exc)
            return
        except WorkspaceCapabilityError as exc:
            self._workspace_error(exc)
            return
        self._json_error(HTTPStatus.NOT_FOUND, "NOT_FOUND")

    def _workspace_error(self, error: WorkspaceCapabilityError) -> None:
        # Error messages can contain local context.  The browser receives only
        # the bounded stable code; no filesystem path or credential is echoed.
        code = error.code
        if re.fullmatch(r"[A-Z][A-Z0-9_]{0,63}", code) is None:
            code = "WORKSPACE_REQUEST_REJECTED"
        self._json_error(HTTPStatus.BAD_REQUEST, code)

    def _voice_error(self, error: LocalVoiceError) -> None:
        # Never echo decoder, artifact or local filesystem diagnostics.
        code = error.code
        if re.fullmatch(r"[A-Z][A-Z0-9_]{0,63}", code) is None:
            code = "VOICE_REQUEST_REJECTED"
        self._json_error(HTTPStatus.BAD_REQUEST, code)

    @staticmethod
    def _retrieval_evidence(payload: object, *, expected_query: str) -> tuple[Any, ...]:
        """Validate the local adapter response before it can enter a prompt."""

        from cogni_agent.manager import RetrievalEvidence, RetrievalProvenance

        if (
            not isinstance(payload, dict)
            or set(payload) != _RAG_QUERY_RESPONSE_KEYS
            or type(payload.get("schema_version")) is not int
            or payload.get("schema_version") != RAG_QUERY_SCHEMA_VERSION
            or payload.get("engine") != "AkasicDB"
            or payload.get("repository") != AKASICDB_REPOSITORY
            or payload.get("revision") != AKASICDB_AUDITED_REVISION
            or payload.get("retrieval_mode") != RAG_RETRIEVAL_MODE
            or payload.get("embedding") != RAG_EMBEDDING_PROFILE
            or payload.get("semantic_embedding") is not False
            or payload.get("answer_integration") is not True
            or payload.get("answer_integration_schema") != RAG_ANSWER_INTEGRATION_SCHEMA
            or payload.get("query") != expected_query
        ):
            raise WorkspaceCapabilityError(
                "WORKSPACE_RESPONSE_INVALID",
                "RAG response provenance is invalid",
            )
        results = payload.get("results")
        count = payload.get("count")
        if (
            not isinstance(results, list)
            or len(results) > 12
            or not isinstance(count, int)
            or isinstance(count, bool)
            or count != len(results)
        ):
            raise WorkspaceCapabilityError(
                "WORKSPACE_RESPONSE_INVALID", "RAG results are invalid"
            )
        candidates: list[tuple[str, int, str, str, float, str, str]] = []
        for item in results:
            if not isinstance(item, dict) or set(item) != _RAG_QUERY_RESULT_KEYS:
                raise WorkspaceCapabilityError(
                    "WORKSPACE_RESPONSE_INVALID", "RAG result is invalid"
                )
            attachment_id = item.get("attachment_id")
            chunk_index = item.get("chunk_index")
            title = item.get("name")
            text = item.get("text")
            score = item.get("score")
            source_sha256 = item.get("source_sha256")
            excerpt_sha256 = item.get("excerpt_sha256")
            if (
                not isinstance(attachment_id, str)
                or re.fullmatch(r"[0-9a-f]{24}", attachment_id) is None
                or not isinstance(chunk_index, int)
                or isinstance(chunk_index, bool)
                or not 0 <= chunk_index < 128
                or not isinstance(title, str)
                or not 1 <= len(title) <= 128
                or any(ord(character) < 32 for character in title)
                or not isinstance(text, str)
                or not 1 <= len(text) <= 1_600
                or not text.strip()
                or any(
                    (ord(character) < 32 and character not in "\t\r\n")
                    or 127 <= ord(character) <= 159
                    for character in text
                )
                or not isinstance(score, (int, float))
                or isinstance(score, bool)
                or not isfinite(float(score))
                or not isinstance(source_sha256, str)
                or re.fullmatch(r"[0-9a-f]{64}", source_sha256) is None
                or not isinstance(excerpt_sha256, str)
                or re.fullmatch(r"[0-9a-f]{64}", excerpt_sha256) is None
                or excerpt_sha256 != sha256(text.encode("utf-8")).hexdigest()
                or not source_sha256.startswith(attachment_id)
            ):
                raise WorkspaceCapabilityError(
                    "WORKSPACE_RESPONSE_INVALID", "RAG result fields are invalid"
                )
            source_payload = {
                "schema_version": 2,
                **{
                    key: item[key]
                    for key in _RAG_SOURCE_RESPONSE_KEYS
                    if key != "schema_version"
                },
            }
            if not _rag_source_payload_is_valid(
                source_payload,
                attachment_id=attachment_id,
                chunk_index=chunk_index,
            ):
                raise WorkspaceCapabilityError(
                    "WORKSPACE_RESPONSE_INVALID",
                    "RAG result provenance is invalid",
                )
            numeric_score = float(score)
            if numeric_score <= 0.0:
                continue
            if numeric_score > 1.0:
                raise WorkspaceCapabilityError(
                    "WORKSPACE_RESPONSE_INVALID", "RAG score is invalid"
                )
            candidates.append(
                (
                    attachment_id,
                    chunk_index,
                    title,
                    text,
                    numeric_score,
                    source_sha256,
                    excerpt_sha256,
                )
            )
        evidence: list[Any] = []
        remaining_chars = MAX_RAG_EVIDENCE_CHARS
        selected_candidates = candidates[:MAX_RAG_EVIDENCE_CHUNKS]
        for index, (
            attachment_id,
            chunk_index,
            title,
            text,
            score,
            source_sha256,
            excerpt_sha256,
        ) in enumerate(selected_candidates):
            remaining_slots = len(selected_candidates) - index
            text_limit = min(
                MAX_RAG_EVIDENCE_CHUNK_CHARS,
                remaining_chars // remaining_slots,
            )
            bounded_text = text[:text_limit].rstrip()
            if not bounded_text:
                continue
            selected = RetrievalEvidence(
                source_id=f"{attachment_id}.{chunk_index}",
                title=title,
                text=bounded_text,
                score=score,
                provenance=RetrievalProvenance(
                    repository=AKASICDB_REPOSITORY,
                    revision=AKASICDB_AUDITED_REVISION,
                    retrieval_mode=RAG_RETRIEVAL_MODE,
                    embedding=RAG_EMBEDDING_PROFILE,
                    semantic_embedding=False,
                    answer_integration_schema=RAG_ANSWER_INTEGRATION_SCHEMA,
                    source_sha256=source_sha256,
                    indexed_excerpt_sha256=excerpt_sha256,
                    indexed_excerpt_chars=len(text),
                ),
            )
            if _admitted_product_retrieval_identity(selected) is None:
                raise WorkspaceCapabilityError(
                    "WORKSPACE_RESPONSE_INVALID",
                    "RAG result product authority is invalid",
                )
            evidence.append(selected)
            remaining_chars -= len(bounded_text)
        return tuple(evidence)

    def _read_json_body(
        self, *, maximum_bytes: int = MAX_REQUEST_BODY_BYTES
    ) -> dict[str, Any] | None:
        if self.headers.get_content_type() != "application/json":
            self._json_error(HTTPStatus.UNSUPPORTED_MEDIA_TYPE, "JSON_REQUIRED")
            return None
        raw_length = self.headers.get("Content-Length")
        try:
            length = int(raw_length or "")
        except ValueError:
            self._json_error(HTTPStatus.LENGTH_REQUIRED, "CONTENT_LENGTH_REQUIRED")
            return None
        if not 0 <= length <= maximum_bytes:
            self._json_error(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, "BODY_TOO_LARGE")
            return None
        try:
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            self._json_error(HTTPStatus.BAD_REQUEST, "INVALID_JSON")
            return None
        if not isinstance(payload, dict):
            self._json_error(HTTPStatus.BAD_REQUEST, "INVALID_BODY")
            return None
        return payload

    def _json_error(self, status: HTTPStatus, code: str) -> None:
        self._json(status, {"error": {"code": code}})

    def _json(self, status: HTTPStatus, payload: Mapping[str, Any]) -> None:
        body = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode(
            "utf-8"
        )
        self._send(status, body, "application/json; charset=utf-8")

    def _send(self, status: HTTPStatus, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self._security_headers()
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _security_headers(self) -> None:
        self.send_header("Content-Security-Policy", _CSP)
        self.send_header("Permissions-Policy", "microphone=(self)")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("X-Frame-Options", "DENY")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("Cache-Control", "no-store")


def _build_product_controls(
    project_root: Path,
    model_path: str | Path,
    manifest_path: str | Path,
    validator: JobManager,
    *,
    gpu_lease_manager: GPULeaseManager | None = None,
    rhythm: RhythmController | None = None,
) -> tuple[Any, EvolutionController]:
    """Build lazy local chat, bounded tools, and proposal-only Self-Harness."""

    # Chat can load the resident model without launching the validation worker.
    # Verify the exact local artifact set before constructing any product
    # component so this path cannot bypass the signed manifest boundary.
    verified = verify_artifact_manifest(model_path, manifest_path)

    from cogni_agent.conversation_fastpath import ConversationFastPath
    from cogni_agent.fact_grounding import RuntimeFactGrounder
    from cogni_agent.manager import (
        RETRIEVAL_EVIDENCE_SCHEMA,
        SYSTEM_PROMPT,
        AgentManager,
    )
    from cogni_agent.model_service import (
        ModelService,
        _require_instruction_tuned_e4b,
    )
    from cogni_agent.tools import WorkspaceToolExecutor
    from cogni_flow.production import (
        ProductionHarnessConfig,
        PromotionMode,
        build_production_self_harness,
    )
    from cogni_os.factbook import build_runtime_factbook_from_verified
    from cogni_os.version import __version__

    # A syntactically valid manifest is not a model trust root.  Bind the
    # complete verified digest set to the pinned official E4B-it checkpoint
    # before publishing any Fact-book claim or constructing product services.
    _require_instruction_tuned_e4b(verified)

    # The running source tree is the authority.  An older installed wheel may
    # coexist during an in-place upgrade and must never overwrite the product
    # identity exposed by this exact code.
    build_version = __version__
    factbook = build_runtime_factbook_from_verified(
        verified,
        manifest_path,
        build_version=build_version,
        device="현재 프로세스 라이브 검증 전 미측정",
    )
    workspace_capabilities: WorkspaceCapabilityService | None
    try:
        workspace_capabilities = WorkspaceCapabilityService.from_runtime_factbook(
            project_root,
            model_path,
            manifest_path,
            factbook,
            model_registry_root=(os.environ.get("COGNI_OS_MODEL_REGISTRY_DIR") or None),
            akasicdb_path=os.environ.get("COGNI_OS_AKASICDB_DIR") or None,
            web_policy=web_policy_from_environment(os.environ),
            lens_client=LensApiClient.from_environment(os.environ),
            answer_integration_schema=(
                RETRIEVAL_EVIDENCE_SCHEMA
                if AgentManager.RETRIEVAL_EVIDENCE_SCHEMA
                == RAG_ANSWER_INTEGRATION_SCHEMA
                else None
            ),
        )
    except (OSError, ValueError, WorkspaceCapabilityError):
        # Chat remains available if this optional non-GPU service cannot be
        # constructed.  The HTTP control plane then reports 503 instead of
        # advertising an attachment or RAG capability that is not running.
        workspace_capabilities = None
    lease_authority = gpu_lease_manager or GPULeaseManager()
    active_rhythm = rhythm or RhythmController()

    def model_lease_purpose() -> str:
        return (
            "inference" if active_rhythm.mode is SystemMode.INFERENCE else "evolution"
        )

    service = ModelService.for_local_gemma(
        model_path,
        manifest_path=manifest_path,
        artifact_digest=factbook.model.manifest_sha256,
        vram_limit_gib=16.7,
        max_input_tokens=4_096,
        max_new_tokens=512,
        max_prompt_chars=32_000,
        max_response_chars=32_000,
        # Interactive requests fail closed instead of appearing frozen for many
        # minutes when a no-KV-cache decode or driver call stalls.
        request_timeout=180.0,
        gpu_lease_manager=lease_authority,
        gpu_lease_owner="cogni-resident-model",
        gpu_lease_purpose=model_lease_purpose,
        gpu_lease_vram_bytes=lease_authority.max_vram_bytes,
    )
    patch_model = _ServiceBackedPatchModel(service, service.tokenizer)
    config = ProductionHarnessConfig(
        allowed_roots=("cogni_agent", "cogni_core", "cogni_flow"),
        idle_seconds=0.0,
        promotion_mode=PromotionMode.PROPOSAL_ONLY,
    )
    target_allowlist = {
        ("RuntimeError", verifier, mechanism): target
        for verifier, mechanism, target in {
            *_AGENT_FAILURE_ROUTES.values(),
            _UNCLASSIFIED_AGENT_FAILURE,
        }
    }
    harness = build_production_self_harness(
        project_root,
        patch_model,
        service.tokenizer,
        target_allowlist,
        lambda: _write_product_checkpoint(project_root),
        config=config,
        rhythm=active_rhythm,
    )
    try:
        harness.start()
        evolution = EvolutionController(harness, worker_cleanup=service.stop)

        def capture_failure(code: str, message: str) -> None:
            verifier_code, mechanism, _target = _agent_failure_route(code)
            harness.capture_exception(
                f"agent-{code}"[:128],
                RuntimeError(f"{code}: {message}"[:512]),
                verifier_code=verifier_code,
                mechanism=mechanism,
            )

        agent = AgentManager(
            service,
            WorkspaceToolExecutor(project_root),
            failure_sink=capture_failure,
            evolution_snapshot=evolution.snapshot,
            availability_check=lambda: not validator.is_active,
            conversation_fast_path=ConversationFastPath(),
            fact_grounder=RuntimeFactGrounder(factbook),
            # Product identity and capability questions are answered by the
            # deterministic RuntimeFactGrounder before generation.  Repeating
            # the complete Fact-book in every ordinary Gemma prompt polluted
            # short social turns and made the small E4B imitate status prose.
            system_prompt=SYSTEM_PROMPT,
            rhythm=active_rhythm,
        )
        agent.workspace_capability_service = workspace_capabilities
        return agent, evolution
    except BaseException as error:
        _run_best_effort_cleanup(
            (harness.stop, service.stop),
            primary=error,
        )
        raise AssertionError("unreachable cleanup path")


def _build_local_voice_service(
    model_path: str | Path,
    manifest_path: str | Path,
    agent_manager: object,
) -> LocalVoiceService:
    """Bind voice features to verified, already-running local authorities.

    STT receives the exact ``ModelService`` owned by ``AgentManager`` so it
    cannot create a second Gemma model.  Windows TTS is advertised only after
    a real installed voice has produced and validated a bounded WAV probe.
    Optional construction failures intentionally leave capture transport
    available while the corresponding STT/TTS capability stays disabled.
    """

    transcriber = None
    try:
        transcriber = Gemma4ModelSpeechTranscriber(agent_manager.model_service)
    except (AttributeError, OSError, TypeError, ValueError):
        pass

    synthesizer = None
    tts_host_probe_passed = False
    try:
        candidate = WindowsSpeechSynthesizer()
        probe = candidate.synthesize(text="Cogni", language="auto")
        if (
            not isinstance(probe, dict)
            or probe.get("source") != "verified_windows_system_speech"
            or probe.get("external_calls") != 0
            or not isinstance(probe.get("audio_wav_base64"), str)
        ):
            raise ValueError("Windows TTS probe returned invalid evidence")
        synthesizer = candidate
        tts_host_probe_passed = True
    except (LocalVoiceError, OSError, TypeError, ValueError):
        pass

    return LocalVoiceService.for_verified_gemma4(
        model_path,
        manifest_path,
        transcriber=transcriber,
        synthesizer=synthesizer,
        tts_host_probe_passed=tts_host_probe_passed,
    )


def _validation_boundary_from_arguments(
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
) -> GPUExecutionBoundary | None:
    """Translate an explicit launch profile into live-validation authority."""

    boundary_values = (
        args.validation_physical_gpu_index,
        args.validation_gpu_query_context,
        args.validation_gpu_uuid,
        args.expected_source_commit,
        args.native_snapshot_stage,
        args.native_source_snapshot_root,
        args.native_source_snapshot_nonce,
        args.native_workspace_root,
        args.native_source_content_digest,
        args.native_source_identity_digest,
        args.native_source_file_count,
        args.native_source_root_device,
        args.native_source_root_inode,
        args.native_model_snapshot_root,
        args.native_model_manifest_sha256,
        args.native_model_content_digest,
        args.native_model_identity_digest,
        args.native_model_file_count,
        args.native_model_root_device,
        args.native_model_root_inode,
        args.native_model_total_bytes,
    )
    if args.validation_profile == DESKTOP_VALIDATION_PROFILE:
        if any(value is not None for value in boundary_values):
            parser.error(
                "desktop-ui-only does not accept server GPU boundary arguments"
            )
        if not _is_windows_desktop_platform():
            parser.error(
                "desktop-ui-only is restricted to the Windows appliance; "
                "the Linux server requires an explicit GPU5 validation profile"
            )
        return None

    if any(value is None for value in boundary_values[:3]):
        parser.error(
            "server-gpu5-native requires physical index, query context, and UUID"
        )
    if (
        not isinstance(args.expected_source_commit, str)
        or _EXPECTED_SOURCE_COMMIT.fullmatch(args.expected_source_commit) is None
    ):
        parser.error("server-gpu5-native requires an exact lowercase source commit")
    if (
        args.native_snapshot_stage != "sealed"
        or not isinstance(args.native_source_snapshot_root, str)
        or not isinstance(args.native_source_snapshot_nonce, str)
        or re.fullmatch(r"[0-9a-f]{32}", args.native_source_snapshot_nonce) is None
        or not isinstance(args.native_workspace_root, str)
        or not isinstance(args.native_source_content_digest, str)
        or re.fullmatch(r"[0-9a-f]{64}", args.native_source_content_digest) is None
        or not isinstance(args.native_source_identity_digest, str)
        or re.fullmatch(r"[0-9a-f]{64}", args.native_source_identity_digest) is None
        or not isinstance(args.native_source_file_count, int)
        or not 1 <= args.native_source_file_count <= 1_000_000
        or not isinstance(args.native_source_root_device, int)
        or not 0 <= args.native_source_root_device <= (2**64) - 1
        or not isinstance(args.native_source_root_inode, int)
        or not 1 <= args.native_source_root_inode <= (2**64) - 1
        or not isinstance(args.native_model_snapshot_root, str)
        or not isinstance(args.native_model_manifest_sha256, str)
        or re.fullmatch(r"[0-9a-f]{64}", args.native_model_manifest_sha256) is None
        or not isinstance(args.native_model_content_digest, str)
        or re.fullmatch(r"[0-9a-f]{64}", args.native_model_content_digest) is None
        or not isinstance(args.native_model_identity_digest, str)
        or re.fullmatch(r"[0-9a-f]{64}", args.native_model_identity_digest) is None
        or not isinstance(args.native_model_file_count, int)
        or not 1 <= args.native_model_file_count <= 1_000_000
        or not isinstance(args.native_model_root_device, int)
        or not 0 <= args.native_model_root_device <= (2**64) - 1
        or not isinstance(args.native_model_root_inode, int)
        or not 1 <= args.native_model_root_inode <= (2**64) - 1
        or not isinstance(args.native_model_total_bytes, int)
        or not 1 <= args.native_model_total_bytes <= 96 * 1024 * 1024 * 1024
    ):
        parser.error("server-gpu5-native requires sealed source/model capabilities")
    try:
        boundary = GPUExecutionBoundary(
            physical_gpu_index=args.validation_physical_gpu_index,
            gpu_query_context=args.validation_gpu_query_context,
            gpu_uuid=args.validation_gpu_uuid,
        )
        # The product process also owns a resident model service.  Its native
        # visibility must therefore be fixed before this Python process starts,
        # not only on the later validation child.
        boundary.require_native_environment(os.environ)
        _require_isolated_server_python()
    except GPUExecutionBoundaryError as error:
        parser.error(str(error))
    return boundary


def _is_windows_desktop_platform() -> bool:
    return os.name == "nt"


def _require_isolated_server_python() -> None:
    """Require startup flags that cannot be made effective after import time."""

    if (
        sys.flags.isolated != 1
        or sys.flags.dont_write_bytecode != 1
        or sys.flags.no_user_site != 1
        or sys.flags.safe_path is not True
    ):
        raise GPUExecutionBoundaryError(
            "server-gpu5-native must start Python with -I -B"
        )


def _preflight_server_gpu_idle(boundary: GPUExecutionBoundary) -> None:
    """Reprove physical GPU5 idle/no-PID before any CUDA-capable probe."""

    if not isinstance(boundary, GPUExecutionBoundary):
        raise TypeError("boundary must be GPUExecutionBoundary")
    from scripts.gpu5_boundary_guard import (
        GPU5BoundaryError,
        assert_gpu5_idle,
        preflight_gpu5,
    )

    try:
        snapshot = preflight_gpu5()
        assert_gpu5_idle(snapshot)
        if (
            snapshot.physical_index != boundary.physical_gpu_index
            or snapshot.uuid != boundary.gpu_uuid
        ):
            raise GPU5BoundaryError(
                "native server idle snapshot returned an invalid GPU5 scope"
            )
    except (AttributeError, GPU5BoundaryError, TypeError, ValueError) as error:
        raise GPUExecutionBoundaryError(
            "native server requires an idle, process-free physical GPU5"
        ) from error


def _preflight_server_gpu_identity(boundary: GPUExecutionBoundary) -> None:
    """Run logical identity in a bounded child; the parent never imports torch."""

    if not isinstance(boundary, GPUExecutionBoundary):
        raise TypeError("boundary must be GPUExecutionBoundary")
    project_root = Path(__file__).resolve().parents[1]
    probe = (project_root / "scripts" / "probe_native_gpu5_identity.py").resolve(
        strict=True
    )
    if not probe.is_file() or not probe.is_relative_to(project_root):
        raise GPUExecutionBoundaryError(
            "resident model GPU identity probe escaped the source checkout"
        )
    executable = _validated_python_invocation_path(sys.executable)
    command = (
        executable,
        "-I",
        "-B",
        os.fspath(probe),
        "--physical-gpu-index",
        str(boundary.physical_gpu_index),
        "--gpu-query-context",
        boundary.gpu_query_context,
        "--gpu-uuid",
        boundary.gpu_uuid,
    )
    try:
        completed = subprocess.run(
            command,
            cwd=os.fspath(project_root),
            env=_identity_probe_environment(boundary),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            shell=False,
            close_fds=True,
            check=False,
            timeout=SERVER_GPU_IDENTITY_PROBE_TIMEOUT_SECONDS,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise GPUExecutionBoundaryError(
            "resident model GPU identity child did not terminate cleanly"
        ) from error
    if completed.returncode != 0:
        raise GPUExecutionBoundaryError(
            "resident model GPU identity/logical-device child rejected startup"
        )


def _postflight_server_gpu_absence(boundary: GPUExecutionBoundary) -> None:
    """Prove the exact physical GPU5 is idle after every product cleanup."""

    from scripts.gpu5_boundary_guard import (
        GPU5BoundaryError,
        assert_gpu5_idle,
        preflight_gpu5,
    )

    try:
        snapshot = preflight_gpu5()
        assert_gpu5_idle(snapshot)
        if (
            snapshot.physical_index != boundary.physical_gpu_index
            or snapshot.uuid != boundary.gpu_uuid
        ):
            raise GPU5BoundaryError(
                "native server postflight returned an invalid GPU5 scope"
            )
    except (AttributeError, GPU5BoundaryError, TypeError, ValueError) as error:
        raise GPUExecutionBoundaryError(
            "native server could not prove post-cleanup GPU5 absence"
        ) from error


@contextmanager
def _native_gpu5_server_lifecycle(
    boundary: GPUExecutionBoundary,
    expected_source_commit: str,
    authority: object,
    *,
    source_snapshot_root: str,
    source_snapshot_nonce: str,
    workspace_root: str,
    source_content_digest: str,
    source_identity_digest: str,
    source_file_count: int,
    source_root_device: int,
    source_root_inode: int,
    model_snapshot_root: str,
    model_manifest_path: str,
    model_manifest_sha256: str,
    model_content_digest: str,
    model_identity_digest: str,
    model_file_count: int,
    model_root_device: int,
    model_root_inode: int,
    model_total_bytes: int,
) -> Iterator[None]:
    """Consume the pre-import authority and retain poison on uncertainty."""

    if not isinstance(boundary, GPUExecutionBoundary):
        raise TypeError("boundary must be GPUExecutionBoundary")
    if (
        not isinstance(expected_source_commit, str)
        or _EXPECTED_SOURCE_COMMIT.fullmatch(expected_source_commit) is None
    ):
        raise GPUExecutionBoundaryError(
            "server-gpu5-native requires an exact lowercase source commit"
        )

    from scripts.gpu5_boundary_guard import (
        GPU5BoundaryError,
        NativeGPU5ServerAuthority,
        verify_native_execution_snapshot,
    )

    if not isinstance(authority, NativeGPU5ServerAuthority):
        raise GPUExecutionBoundaryError(
            "server-gpu5-native requires bootstrap-held GPU5 authority"
        )
    try:
        authority.consume(
            expected_source_commit=expected_source_commit,
            physical_gpu_index=boundary.physical_gpu_index,
            gpu_query_context=boundary.gpu_query_context,
            gpu_uuid=boundary.gpu_uuid,
            source_snapshot_root=source_snapshot_root,
            source_snapshot_nonce=source_snapshot_nonce,
            workspace_root=workspace_root,
            source_content_digest=source_content_digest,
            source_identity_digest=source_identity_digest,
            source_file_count=source_file_count,
            source_root_device=source_root_device,
            source_root_inode=source_root_inode,
            model_snapshot_root=model_snapshot_root,
            model_manifest_path=model_manifest_path,
            model_manifest_sha256=model_manifest_sha256,
            model_content_digest=model_content_digest,
            model_identity_digest=model_identity_digest,
            model_file_count=model_file_count,
            model_root_device=model_root_device,
            model_root_inode=model_root_inode,
            model_total_bytes=model_total_bytes,
        )
        execution_snapshot_before = authority.execution_snapshot
        if (
            verify_native_execution_snapshot(execution_snapshot_before)
            != execution_snapshot_before
        ):
            raise GPUExecutionBoundaryError(
                "native server execution snapshot changed after early gate"
            )
        _preflight_server_gpu_idle(boundary)
        authority.mark_launch_attempted()
        primary: BaseException | None = None
        try:
            _preflight_server_gpu_identity(boundary)
            # The short-lived probe may create a context. Prove it exited and
            # released the physical device before parent product residency.
            _preflight_server_gpu_idle(boundary)
            yield
        except BaseException as error:
            primary = error

        postflight_failures: list[BaseException] = []
        try:
            _postflight_server_gpu_absence(boundary)
        except BaseException as error:
            postflight_failures.append(error)
        try:
            execution_snapshot_after = verify_native_execution_snapshot(
                execution_snapshot_before
            )
            if execution_snapshot_after != execution_snapshot_before:
                raise GPUExecutionBoundaryError(
                    "native server execution snapshot changed during residency"
                )
        except BaseException as error:
            postflight_failures.append(error)

        # The lifecycle cannot yet distinguish a harmless application error
        # from a component/worker termination failure.  Releasing on either
        # would create a late-child GPU acquisition race after an idle sample.
        # Therefore only a fully normal product return plus both independent
        # postflight proofs may clear the crash marker. Any primary error stays
        # fail-closed for explicit operator review.
        if primary is None and not postflight_failures:
            authority.mark_safe_to_release()
        failures = ([primary] if primary is not None else []) + postflight_failures
        if len(failures) == 1:
            raise failures[0]
        if failures:
            raise BaseExceptionGroup(
                _NATIVE_GPU5_LIFECYCLE_ERROR_MESSAGE,
                failures,
            )
    except GPU5BoundaryError as error:
        raise GPUExecutionBoundaryError(
            "native GPU5 host lease or source boundary rejected startup"
        ) from error


def main(
    argv: Sequence[str] | None = None,
    *,
    native_gpu5_authority: object | None = None,
    _native_lifecycle_token: object | None = None,
) -> int:
    parser = argparse.ArgumentParser(
        prog="python scripts/run_cogniboard_server.py",
        allow_abbrev=False,
    )
    arguments = tuple(sys.argv[1:] if argv is None else argv)
    project_root = Path(__file__).resolve().parents[1]
    parser.add_argument(
        "--model",
        default=os.environ.get(
            "COGNI_OS_MODEL_DIR", r"C:\Project\cognios\gemma4-e4b-it"
        ),
    )
    parser.add_argument(
        "--manifest",
        default=str(project_root / "config" / "gemma4-e4b-it.manifest.toml"),
    )
    parser.add_argument("--assets", default=str(project_root / "cogni_demo" / "static"))
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument(
        "--validation-profile",
        choices=(DESKTOP_VALIDATION_PROFILE, SERVER_GPU5_VALIDATION_PROFILE),
        default=DESKTOP_VALIDATION_PROFILE,
    )
    parser.add_argument("--validation-physical-gpu-index", type=int)
    parser.add_argument(
        "--validation-gpu-query-context",
        choices=(SERVER_VALIDATION_GPU_QUERY_CONTEXT,),
    )
    parser.add_argument("--validation-gpu-uuid")
    parser.add_argument("--expected-source-commit")
    parser.add_argument("--native-snapshot-stage", choices=("sealed",))
    parser.add_argument("--native-source-snapshot-root")
    parser.add_argument("--native-source-snapshot-nonce")
    parser.add_argument("--native-workspace-root")
    parser.add_argument("--native-source-content-digest")
    parser.add_argument("--native-source-identity-digest")
    parser.add_argument("--native-source-file-count", type=int)
    parser.add_argument("--native-source-root-device", type=int)
    parser.add_argument("--native-source-root-inode", type=int)
    parser.add_argument("--native-model-snapshot-root")
    parser.add_argument("--native-model-manifest-sha256")
    parser.add_argument("--native-model-content-digest")
    parser.add_argument("--native-model-identity-digest")
    parser.add_argument("--native-model-file-count", type=int)
    parser.add_argument("--native-model-root-device", type=int)
    parser.add_argument("--native-model-root-inode", type=int)
    parser.add_argument("--native-model-total-bytes", type=int)
    args = parser.parse_args(arguments)
    validation_boundary = _validation_boundary_from_arguments(parser, args)

    session_path = default_session_path()
    existing = find_live_session(session_path)
    if existing is not None:
        if validation_boundary is not None:
            parser.error(
                "server-gpu5-native refuses to reuse an existing CogniBoard "
                "session; stop the existing process before guarded startup"
            )
        print(f"existing_demo_url=http://127.0.0.1:{existing.port}/", flush=True)
        if not args.no_browser:
            open_graphical_app(existing.bootstrap_url)
        return 0

    if validation_boundary is None:
        if native_gpu5_authority is not None or _native_lifecycle_token is not None:
            raise GPUExecutionBoundaryError(
                "desktop-ui-only rejects native GPU5 server authority"
            )
    elif native_gpu5_authority is None:
        raise GPUExecutionBoundaryError(
            "server-gpu5-native must start through run_cogniboard_server.py"
        )
    elif _native_lifecycle_token is not _NATIVE_GPU5_LIFECYCLE_TOKEN:
        with _native_gpu5_server_lifecycle(
            validation_boundary,
            args.expected_source_commit,
            native_gpu5_authority,
            source_snapshot_root=args.native_source_snapshot_root,
            source_snapshot_nonce=args.native_source_snapshot_nonce,
            workspace_root=args.native_workspace_root,
            source_content_digest=args.native_source_content_digest,
            source_identity_digest=args.native_source_identity_digest,
            source_file_count=args.native_source_file_count,
            source_root_device=args.native_source_root_device,
            source_root_inode=args.native_source_root_inode,
            model_snapshot_root=args.native_model_snapshot_root,
            model_manifest_path=args.manifest,
            model_manifest_sha256=args.native_model_manifest_sha256,
            model_content_digest=args.native_model_content_digest,
            model_identity_digest=args.native_model_identity_digest,
            model_file_count=args.native_model_file_count,
            model_root_device=args.native_model_root_device,
            model_root_inode=args.native_model_root_inode,
            model_total_bytes=args.native_model_total_bytes,
        ):
            return main(
                arguments,
                native_gpu5_authority=native_gpu5_authority,
                _native_lifecycle_token=_NATIVE_GPU5_LIFECYCLE_TOKEN,
            )

    if validation_boundary is not None:
        from scripts.gpu5_boundary_guard import NativeGPU5ServerAuthority

        if not isinstance(native_gpu5_authority, NativeGPU5ServerAuthority):
            raise GPUExecutionBoundaryError(
                "native server lost its execution snapshot authority"
            )
        execution_snapshot = native_gpu5_authority.execution_snapshot
        try:
            source_root = Path(execution_snapshot.source.root_path).resolve(strict=True)
            model_root = Path(execution_snapshot.model.root_path).resolve(strict=True)
            manifest_path = Path(execution_snapshot.manifest_path).resolve(strict=True)
            workspace_root = Path(execution_snapshot.workspace_root).resolve(
                strict=True
            )
            requested_model = Path(args.model).resolve(strict=True)
            requested_manifest = Path(args.manifest).resolve(strict=True)
            requested_assets = Path(args.assets).resolve(strict=True)
        except OSError as error:
            raise GPUExecutionBoundaryError(
                "native execution snapshot paths are unavailable"
            ) from error
        if (
            project_root != source_root
            or requested_model != model_root
            or requested_manifest != manifest_path
            or manifest_path.parent.parent != source_root
            or requested_assets != source_root / "cogni_demo" / "static"
            or workspace_root == source_root
            or source_root.is_relative_to(workspace_root)
            or workspace_root.is_relative_to(source_root)
        ):
            raise GPUExecutionBoundaryError(
                "native product paths differ from execution snapshot authority"
            )

    # Validate the content-addressed CTS policy in the actual backend process
    # before binding HTTP or publishing session metadata. The native launcher
    # preflight is diagnostic only and cannot authorize this later process.
    from cogni_core.cts_policy import load_default_bounded_cts_controller

    load_default_bounded_cts_controller(device="cpu")

    gpu_lease_manager = GPULeaseManager()
    rhythm = RhythmController()
    manager = JobManager(
        production_launch_factory(
            project_root,
            args.model,
            args.manifest,
            gpu_boundary=validation_boundary,
        ),
        gpu_lease_manager=gpu_lease_manager,
    )
    agent_manager, evolution_manager = _build_product_controls(
        (
            project_root
            if validation_boundary is None
            else Path(native_gpu5_authority.execution_snapshot.workspace_root)
        ),
        args.model,
        args.manifest,
        manager,
        gpu_lease_manager=gpu_lease_manager,
        rhythm=rhythm,
    )
    workspace_service = getattr(agent_manager, "workspace_capability_service", None)
    try:
        voice_service = _build_local_voice_service(
            args.model,
            args.manifest,
            agent_manager,
        )
    except BaseException as error:
        _shutdown_product_components(
            manager,
            evolution_manager,
            agent_manager,
            primary=error,
        )
        raise AssertionError("unreachable cleanup path")
    try:
        server = DemoHTTPServer(
            manager,
            args.assets,
            agent_manager=agent_manager,
            evolution_manager=evolution_manager,
            workspace_service=workspace_service,
            voice_service=voice_service,
            port=args.port,
        )
    except OSError as error:
        if validation_boundary is not None:
            boundary_error = GPUExecutionBoundaryError(
                "server-gpu5-native refuses session reuse after a bind conflict"
            )
            boundary_error.__cause__ = error
            _shutdown_product_components(
                manager,
                evolution_manager,
                agent_manager,
                primary=boundary_error,
            )
            raise AssertionError("unreachable cleanup path")
        # A concurrent launcher may bind the fixed port just before publishing
        # its atomic session document. Wait briefly and reuse it; never start a
        # second control plane or CUDA worker.
        for _ in range(10):
            sleep(0.1)
            existing = read_session_metadata(session_path)
            if existing is not None and ping_session(existing):
                print(
                    f"existing_demo_url=http://127.0.0.1:{existing.port}/",
                    flush=True,
                )
                _shutdown_product_components(
                    manager,
                    evolution_manager,
                    agent_manager,
                )
                if not args.no_browser:
                    open_graphical_app(existing.bootstrap_url)
                return 0
        _shutdown_product_components(
            manager,
            evolution_manager,
            agent_manager,
            primary=error,
        )
        raise AssertionError("unreachable cleanup path")
    except BaseException as error:
        _shutdown_product_components(
            manager,
            evolution_manager,
            agent_manager,
            primary=error,
        )
        raise AssertionError("unreachable cleanup path")

    metadata = SessionMetadata(
        os.getpid(),
        server.server_port,
        server.token,
        datetime.now(timezone.utc).isoformat(),
    )
    try:
        write_session_metadata(metadata, session_path)
    except BaseException as error:
        _run_best_effort_cleanup((server.server_close,), primary=error)
        raise AssertionError("unreachable cleanup path")
    cleanup_server = (
        server.server_close,
        lambda: remove_session_metadata(session_path, expected=metadata),
    )
    try:
        print(f"demo_url={server.origin}/", flush=True)
        if not args.no_browser:
            open_graphical_app(server.bootstrap_url)
        server.serve_forever(poll_interval=0.2)
    except KeyboardInterrupt:
        _run_best_effort_cleanup(cleanup_server)
    except BaseException as error:
        _run_best_effort_cleanup(cleanup_server, primary=error)
        raise AssertionError("unreachable cleanup path")
    else:
        _run_best_effort_cleanup(cleanup_server)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ComputeBusyError",
    "GPUExecutionBoundary",
    "GPUExecutionBoundaryError",
    "DemoHTTPServer",
    "DemoServerError",
    "EvolutionController",
    "INITIAL_METRICS",
    "JobAlreadyRunningError",
    "JobManager",
    "NoActiveJobError",
    "WorkerLaunch",
    "WorkerTerminationError",
    "main",
    "production_launch_factory",
]
