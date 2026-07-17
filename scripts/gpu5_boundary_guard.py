"""Fail-closed physical-GPU boundary for the Cogni-OS lab server.

The laboratory permits physical GPU indices 0 through 5.  This checkout is
stricter: it may use physical GPU 5 only.  The guard never enumerates devices;
every ``nvidia-smi`` command names index 5 explicitly, so GPUs 0 through 4,
6, and 7 are never queried, enumerated, exposed, reserved, allocated, or used.

Docker remaps physical GPU 5 to the container's logical ``cuda:0``.  Therefore
the generated container environment deliberately omits
``CUDA_VISIBLE_DEVICES`` instead of setting it to the host index.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import secrets
import shlex
import subprocess
from typing import Any, Mapping, Sequence

try:
    import resource
except ImportError:  # pragma: no cover - the server gate is Linux-only
    resource = None


LAB_ALLOWED_PHYSICAL_GPU_INDICES = frozenset(range(6))
PROJECT_PHYSICAL_GPU_INDEX = 5
PROJECT_GPU_UUID = "GPU-84d7eeb0-65e0-a5b1-d7db-d09ef59fe03a"
NVIDIA_SMI_TIMEOUT_SECONDS = 5.0
MAX_NVIDIA_SMI_OUTPUT_CHARS = 65_536
MAX_IDLE_DRIVER_MEMORY_MIB = 64
MAX_EVIDENCE_BYTES = 4 * 1024 * 1024
EVIDENCE_HOST_ROOT = Path(
    "/home/shoon/workspace/Cogni-OS-2.0-v041/outputs/server-evidence"
)


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
_RESERVED_CONTAINER_GPU_ENV = frozenset(
    {"CUDA_VISIBLE_DEVICES", "NVIDIA_VISIBLE_DEVICES"}
)
_CONTAINER_NAME = re.compile(r"cognios-gpu5-[a-z0-9]{12}\Z")
_ALLOWED_MOUNT_ROOTS = (
    Path("/home/shoon/workspace/Cogni-OS-2.0-v041"),
    Path("/home/shoon/models/gemma4-e4b"),
)
_EXPECTED_READ_ONLY_MOUNTS = {
    _ALLOWED_MOUNT_ROOTS[0]: Path("/workspace"),
    _ALLOWED_MOUNT_ROOTS[1]: Path("/models/gemma4-e4b"),
}
_REQUIRED_CONTAINER_ENVIRONMENT = {
    "HF_HUB_OFFLINE": "1",
    "PYTHONDONTWRITEBYTECODE": "1",
    "TRANSFORMERS_OFFLINE": "1",
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
            "--turns",
        }
    ),
    "/workspace/scripts/validate_gemma4_runtime.py": frozenset(
        {"--model", "--manifest", "--prompt", "--workspace-mib", "--vram-limit-gib"}
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
            "--vram-limit-gib",
        }
    ),
}
_VALIDATOR_FLAG_OPTIONS = {
    "/workspace/scripts/validate_agent_completion.py": frozenset(),
    "/workspace/scripts/validate_gemma4_runtime.py": frozenset({"--event-stream"}),
    "/workspace/scripts/validate_gemma4_deq.py": frozenset(
        {"--allow-uncertified-experimental"}
    ),
}
MAX_VALIDATOR_ARGV_TOKENS = 32
MAX_VALIDATOR_TOKEN_CHARS = 2_048
MAX_VALIDATOR_PROMPT_CHARS = 512
MAX_CONTAINER_ENVIRONMENT_ITEMS = 16
MAX_CONTAINER_ENVIRONMENT_VALUE_CHARS = 4_096
MAX_CONTAINER_ENVIRONMENT_TOTAL_CHARS = 16_384


class GPU5BoundaryError(RuntimeError):
    """Raised before execution whenever the physical-GPU contract is uncertain."""


class GPU5DockerExecutionError(GPU5BoundaryError):
    """A fail-closed launch error with bounded machine-readable evidence."""

    def __init__(self, message: str, evidence: Mapping[str, Any]) -> None:
        super().__init__(message)
        self.evidence = dict(evidence)


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
    output_policy: str = "bounded_file_capture"


def require_project_gpu_index(value: object) -> int:
    """Accept only this project's physical GPU, never a lab-neighbour device."""

    if isinstance(value, bool) or not isinstance(value, int):
        raise GPU5BoundaryError("an explicit integer physical GPU index is required")
    if value not in LAB_ALLOWED_PHYSICAL_GPU_INDICES:
        raise GPU5BoundaryError("physical GPU index is outside the laboratory boundary")
    if value != PROJECT_PHYSICAL_GPU_INDEX:
        raise GPU5BoundaryError("this project is pinned to physical GPU 5")
    return value


def native_gpu5_environment(
    base: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Return a native-host environment exposing physical GPU 5 only."""

    environment = dict(os.environ if base is None else base)
    configured = environment.get("CUDA_VISIBLE_DEVICES")
    if configured is not None and configured.strip() != str(PROJECT_PHYSICAL_GPU_INDEX):
        raise GPU5BoundaryError("native CUDA visibility conflicts with physical GPU 5")
    nvidia_configured = environment.get("NVIDIA_VISIBLE_DEVICES")
    if nvidia_configured is not None and nvidia_configured.strip() not in {
        str(PROJECT_PHYSICAL_GPU_INDEX),
        PROJECT_GPU_UUID,
    }:
        raise GPU5BoundaryError(
            "native NVIDIA visibility conflicts with physical GPU 5"
        )
    environment["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    environment["CUDA_VISIBLE_DEVICES"] = str(PROJECT_PHYSICAL_GPU_INDEX)
    environment["NVIDIA_VISIBLE_DEVICES"] = PROJECT_GPU_UUID
    return environment


def _nvidia_smi_gpu_argv() -> tuple[str, ...]:
    return (
        "nvidia-smi",
        "-i",
        str(PROJECT_PHYSICAL_GPU_INDEX),
        f"--query-gpu={','.join(_GPU_QUERY_FIELDS)}",
        "--format=csv,noheader,nounits",
    )


def _nvidia_smi_compute_argv() -> tuple[str, ...]:
    return (
        "nvidia-smi",
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
    if argv[:3] != ("nvidia-smi", "-i", "5"):
        raise GPU5BoundaryError("unsafe nvidia-smi command rejected")
    try:
        completed = runner(
            list(argv),
            capture_output=True,
            text=True,
            timeout=NVIDIA_SMI_TIMEOUT_SECONDS,
            check=False,
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


def _source_within_allowed_roots(
    source: Path,
    roots: Sequence[Path] = _ALLOWED_MOUNT_ROOTS,
) -> bool:
    return any(source == root or source.is_relative_to(root) for root in roots)


def _validated_read_only_mount(value: str) -> str:
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
            for root, container in _EXPECTED_READ_ONLY_MOUNTS.items()
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
    if not 6 <= len(values) <= MAX_VALIDATOR_ARGV_TOKENS:
        raise GPU5BoundaryError("validator argv length is outside the fixed bound")
    if values[0] != "python" or values[1] not in _ALLOWED_VALIDATORS:
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
    script = values[1]
    value_options = _VALIDATOR_VALUE_OPTIONS[script]
    flag_options = _VALIDATOR_FLAG_OPTIONS[script]
    parsed: dict[str, str | None] = {}
    position = 2
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
    prompt = parsed.get("--prompt")
    if prompt is not None and len(prompt) > MAX_VALIDATOR_PROMPT_CHARS:
        raise GPU5BoundaryError("validator prompt exceeds the bounded length")
    if script.endswith("validate_agent_completion.py"):
        _number_option(
            parsed, "--physical-gpu-index", minimum=5, maximum=5, integer=True
        )
        if parsed.get("--gpu-query-context") != "gpu5-container":
            raise GPU5BoundaryError(
                "completion validator must use the GPU5 container query context"
            )
        if "--timeout" in parsed:
            _number_option(parsed, "--timeout", minimum=1, maximum=120)
        if "--turns" in parsed:
            _number_option(parsed, "--turns", minimum=1, maximum=100, integer=True)
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
            "--tolerance": (1.0e-9, 1.0, False),
            "--max-iter": (1, 128, True),
            "--history": (1, 64, True),
            "--fallback-steps": (1, 256, True),
            "--fallback-damping": (0, 1, False),
            "--contractive-delta-scale": (0, 1, False),
            "--certified-delta-lipschitz-bound": (0, 0.95, False),
            "--vram-limit-gib": (1, 16.7, False),
        }
        for option, (minimum, maximum, integer) in ranges.items():
            if option in parsed:
                _number_option(
                    parsed,
                    option,
                    minimum=minimum,
                    maximum=maximum,
                    integer=integer,
                )
        certified = "--certified-delta-lipschitz-bound" in parsed
        experimental = "--allow-uncertified-experimental" in parsed
        if certified == experimental:
            raise GPU5BoundaryError(
                "DEQ validation needs exactly one certified or labelled experimental mode"
            )
    return parsed


def _validated_validator_command(command: Sequence[str]) -> tuple[str, ...]:
    values = tuple(command)
    options = _validated_validator_options(values)
    if options["--model"] != "/models/gemma4-e4b":
        raise GPU5BoundaryError(
            "validator model path is outside the pinned read-only mount"
        )
    if options["--manifest"] != "/workspace/config/gemma4-e4b.manifest.toml":
        raise GPU5BoundaryError(
            "validator manifest path is outside the pinned checkout"
        )
    return values


def _prepare_evidence_target(filename: str) -> Path:
    if not isinstance(filename, str) or _EVIDENCE_FILENAME.fullmatch(filename) is None:
        raise GPU5BoundaryError("invalid bounded evidence filename")
    configured_root = EVIDENCE_HOST_ROOT
    if configured_root.is_symlink():
        raise GPU5BoundaryError("server evidence root must not be a symlink")
    try:
        root = configured_root.resolve(strict=True)
    except OSError as error:
        raise GPU5BoundaryError(
            "exact server evidence directory must already exist"
        ) from error
    if not root.is_dir():
        raise GPU5BoundaryError("server evidence root must be a real directory")
    target = root / filename
    if target.exists() or target.is_symlink() or target.parent != root:
        raise GPU5BoundaryError("evidence target must be new and inside the exact root")
    return target


def _bound_evidence_file_size() -> None:
    if resource is None:  # pragma: no cover - guarded before Linux server execution
        raise RuntimeError("RLIMIT_FSIZE is unavailable")
    resource.setrlimit(resource.RLIMIT_FSIZE, (MAX_EVIDENCE_BYTES, MAX_EVIDENCE_BYTES))


def _evidence_digest(target: Path) -> tuple[int, str]:
    try:
        with target.open("rb") as stream:
            payload = stream.read(MAX_EVIDENCE_BYTES + 1)
    except OSError as error:
        raise GPU5BoundaryError("bounded evidence is unreadable") from error
    if not payload:
        raise GPU5BoundaryError("validator produced no evidence")
    if len(payload) > MAX_EVIDENCE_BYTES:
        raise GPU5BoundaryError("validator evidence exceeded the file-size bound")
    return len(payload), hashlib.sha256(payload).hexdigest()


def build_gpu5_docker_argv(
    image: str,
    command: Sequence[str],
    *,
    environment: Mapping[str, str] | None = None,
    mounts: Sequence[str] = (),
    workdir: str | None = None,
    container_name: str | None = None,
) -> tuple[str, ...]:
    """Build the only accepted offline Docker launch shape for this project."""

    if image != PINNED_DOCKER_IMAGE:
        raise GPU5BoundaryError("Docker image must match the pinned immutable digest")
    if not command or any(
        not isinstance(token, str) or not token or "\x00" in token for token in command
    ):
        raise GPU5BoundaryError("a bounded non-empty container command is required")
    validated_command = _validated_validator_command(command)
    if len(mounts) != 2:
        raise GPU5BoundaryError("Docker requires exactly the pinned repo/model mounts")
    validated_mounts = tuple(_validated_read_only_mount(mount) for mount in mounts)
    expected_mounts = tuple(
        f"{source.resolve(strict=True)}:{destination}:ro"
        for source, destination in _EXPECTED_READ_ONLY_MOUNTS.items()
    )
    if len(set(validated_mounts)) != 2 or set(validated_mounts) != set(expected_mounts):
        raise GPU5BoundaryError(
            "Docker repo/model mounts must each appear exactly once"
        )
    if workdir != _PINNED_WORKDIR:
        raise GPU5BoundaryError("Docker workdir must be exactly /workspace")
    name = _container_name(container_name)
    argv: list[str] = [
        "docker",
        "run",
        "--rm",
        "--name",
        name,
        "--gpus",
        "device=5",
        "--network",
        "none",
        "--log-driver",
        "none",
    ]
    argv.extend(_validated_environment(environment))
    for mount in expected_mounts:
        argv.extend(("--volume", mount))
    argv.extend(("--workdir", _PINNED_WORKDIR))
    argv.extend(("--", image, *validated_command))
    built = tuple(argv)
    validate_gpu5_docker_argv(built)
    return built


def validate_gpu5_docker_argv(argv: Sequence[str]) -> None:
    """Reject every Docker launch except the exact production GPU5 contract."""

    values = tuple(argv)
    if values[:2] != ("docker", "run") or values.count("--") != 1:
        raise GPU5BoundaryError("unrecognized Docker launch contract")
    boundary = values.index("--")
    if boundary + 2 >= len(values):
        raise GPU5BoundaryError("Docker image and command are required")
    options = values[2:boundary]
    if options.count("--name") != 1:
        raise GPU5BoundaryError("Docker must have one unique project container name")
    name_position = options.index("--name")
    if name_position + 1 >= len(options):
        raise GPU5BoundaryError("Docker container name is missing")
    name = _container_name(options[name_position + 1])
    expected_mounts = tuple(
        f"{source.resolve(strict=True)}:{destination}:ro"
        for source, destination in _EXPECTED_READ_ONLY_MOUNTS.items()
    )
    expected_options = (
        "--rm",
        "--name",
        name,
        "--gpus",
        "device=5",
        "--network",
        "none",
        "--log-driver",
        "none",
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
    _validated_validator_command(values[boundary + 2 :])


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
    )
    return int(completed.returncode)


def _ensure_container_absent(name: str, *, runner: Any) -> dict[str, Any]:
    """Best-effort stop/rm followed by a fail-closed bounded absence proof."""

    actions: list[dict[str, Any]] = []
    for action, argv, timeout in (
        ("stop", ("docker", "stop", "--time", "5", name), 10.0),
        ("rm", ("docker", "rm", "--force", name), 10.0),
    ):
        try:
            returncode = _docker_control(runner, argv, timeout=timeout)
            actions.append({"action": action, "returncode": returncode})
        except (OSError, subprocess.TimeoutExpired) as error:
            actions.append({"action": action, "error": type(error).__name__})
    verification_argv = (
        "docker",
        "ps",
        "--all",
        "--filter",
        f"name=^/{name}$",
        "--format",
        "{{.Names}}",
    )
    try:
        completed = runner(
            list(verification_argv),
            capture_output=True,
            text=True,
            timeout=5.0,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise GPU5BoundaryError(
            f"container cleanup verification failed: {type(error).__name__}"
        ) from error
    if (
        not isinstance(completed.stdout, str)
        or not isinstance(completed.stderr, str)
        or len(completed.stdout) > 4_096
        or len(completed.stderr) > 4_096
    ):
        raise GPU5BoundaryError("container cleanup verification output is invalid")
    verification = {
        "action": "exact_name_ps_absence",
        "returncode": int(completed.returncode),
    }
    actions.append(verification)
    if completed.returncode != 0:
        raise GPU5BoundaryError("Docker daemon could not prove container absence")
    if completed.stderr.strip():
        raise GPU5BoundaryError("Docker absence proof emitted unexpected stderr")
    names = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
    if names:
        raise GPU5BoundaryError("named GPU5 container still exists after cleanup")
    return {"container_name": name, "container_absent": True, "actions": actions}


def run_guarded_gpu5_container(
    image: str,
    command: Sequence[str],
    *,
    environment: Mapping[str, str] | None = None,
    mounts: Sequence[str] = (),
    workdir: str | None = None,
    run_timeout_seconds: float,
    evidence_filename: str,
    smi_runner: Any = subprocess.run,
    docker_runner: Any = subprocess.run,
    cleanup_runner: Any = subprocess.run,
) -> GuardedDockerResult:
    """Preflight, bounded run, forced cleanup, then require idle postflight."""

    if (
        isinstance(run_timeout_seconds, bool)
        or not 1.0 <= float(run_timeout_seconds) <= 86_400.0
    ):
        raise GPU5BoundaryError("Docker timeout must be in [1, 86400] seconds")
    if resource is None:
        raise GPU5BoundaryError("bounded evidence capture requires Linux RLIMIT_FSIZE")
    evidence_target = _prepare_evidence_target(evidence_filename)
    name = _container_name()
    argv = build_gpu5_docker_argv(
        image,
        command,
        environment=environment,
        mounts=mounts,
        workdir=workdir,
        container_name=name,
    )
    preflight = preflight_gpu5(runner=smi_runner)
    completed: Any | None = None
    execution_error: BaseException | None = None
    try:
        with evidence_target.open("xb", buffering=0) as evidence_stream:
            completed = docker_runner(
                list(argv),
                stdout=evidence_stream,
                stderr=subprocess.STDOUT,
                timeout=float(run_timeout_seconds),
                check=False,
                preexec_fn=_bound_evidence_file_size,
                start_new_session=True,
            )
            evidence_stream.flush()
            os.fsync(evidence_stream.fileno())
    except (OSError, subprocess.SubprocessError) as error:
        execution_error = error

    cleanup_evidence: dict[str, Any] | None = None
    cleanup_error: BaseException | None = None
    try:
        cleanup_evidence = _ensure_container_absent(name, runner=cleanup_runner)
    except BaseException as error:
        cleanup_error = error

    postflight: GPU5Snapshot | None = None
    postflight_error: BaseException | None = None
    try:
        postflight = preflight_gpu5(runner=smi_runner)
    except BaseException as error:
        postflight_error = error

    evidence_bytes: int | None = None
    evidence_sha256: str | None = None
    evidence_error: BaseException | None = None
    try:
        evidence_bytes, evidence_sha256 = _evidence_digest(evidence_target)
    except BaseException as error:
        evidence_error = error

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
    ):
        evidence = {
            "argv": argv,
            "image_digest": PINNED_DOCKER_IMAGE,
            "container_name": name,
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
            "evidence_path": str(evidence_target),
            "evidence_bytes": evidence_bytes,
            "evidence_sha256": evidence_sha256,
            "evidence_error": (
                None if evidence_error is None else type(evidence_error).__name__
            ),
            "output_policy": "bounded_file_capture",
        }
        raise GPU5DockerExecutionError("GPU5 Docker run failed closed", evidence)
    assert postflight is not None
    assert evidence_bytes is not None
    assert evidence_sha256 is not None
    return GuardedDockerResult(
        argv=argv,
        preflight=preflight,
        postflight=postflight,
        image_digest=PINNED_DOCKER_IMAGE,
        returncode=returncode,
        evidence_path=str(evidence_target),
        evidence_bytes=evidence_bytes,
        evidence_sha256=evidence_sha256,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Inspect only physical GPU5 or print its offline Docker argv."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("inspect", help="fail unless the pinned GPU5 is idle")
    docker = subparsers.add_parser(
        "docker-argv", help="print, but do not run, Docker argv"
    )
    docker.add_argument("--image", required=True)
    docker.add_argument("--workdir")
    docker.add_argument("--mount", action="append", default=[])
    docker.add_argument("container_command", nargs=argparse.REMAINDER)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "inspect":
        print(json.dumps(preflight_gpu5().as_payload(), indent=2, sort_keys=True))
        return 0
    container_command = list(args.container_command)
    if container_command[:1] == ["--"]:
        container_command = container_command[1:]
    docker_argv = build_gpu5_docker_argv(
        args.image,
        container_command,
        mounts=args.mount,
        workdir=args.workdir,
    )
    print(json.dumps({"argv": docker_argv, "shell": shlex.join(docker_argv)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
