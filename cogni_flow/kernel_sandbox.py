"""Fail-closed Linux OCI sandbox for Self-Harness candidate evaluation.

The runner deliberately supports only a locally pinned image digest and a
locally pinned engine/socket.  It never mounts the container engine socket or
any GPU device into the candidate container.  Merely constructing this class
does not authorize source promotion: :mod:`cogni_flow.production` still
requires the evidence digest to be explicitly trusted by the operator.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import os
from pathlib import Path, PurePosixPath
import re
import signal
import stat
import subprocess
import sys
import tempfile
from typing import Any, Mapping

from .harness import SandboxResult
from .production import RunnerAttestation, command_sha256


_EVIDENCE_SCHEMA = "cogni.kernel-sandbox-evidence.v1"
_MAX_EVIDENCE_BYTES = 64 * 1024
_MAX_OUTPUT_CHARS = 40_000
_MAX_COMMANDS = 16
_IMAGE_REFERENCE = re.compile(r"^[a-z0-9][a-z0-9._/:-]{0,254}@sha256:[0-9a-f]{64}$")
_DIGEST = re.compile(r"^[0-9a-f]{64}$")
_RUNNER_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_EVIDENCE_FIELDS = {
    "schema",
    "runner_id",
    "engine_path",
    "engine_sha256",
    "daemon_socket",
    "image_reference",
    "allowed_command_sha256",
    "runtime",
    "network_mode",
    "read_only_rootfs",
    "read_only_project",
    "cap_drop_all",
    "no_new_privileges",
    "ephemeral_workspace",
    "non_root_uid",
    "max_memory_bytes",
    "max_pids",
    "max_cpus",
    "tmpfs_bytes",
}


class KernelSandboxError(RuntimeError):
    """Raised when a required isolation property cannot be proven."""


@dataclass(frozen=True)
class KernelSandboxEvidence:
    """Strict, operator-reviewed OCI runner evidence."""

    runner_id: str
    engine_path: str
    engine_sha256: str
    daemon_socket: str
    image_reference: str
    allowed_command_sha256: tuple[str, ...]
    runtime: str
    non_root_uid: int
    max_memory_bytes: int
    max_pids: int
    max_cpus: float
    tmpfs_bytes: int


def parse_kernel_sandbox_evidence(payload: bytes) -> KernelSandboxEvidence:
    """Parse strict canonical runner evidence without touching the host."""

    if not payload or len(payload) > _MAX_EVIDENCE_BYTES:
        raise KernelSandboxError("sandbox evidence size is outside its bound")
    try:
        data = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise KernelSandboxError("sandbox evidence is not strict UTF-8 JSON") from exc
    if not isinstance(data, dict) or set(data) != _EVIDENCE_FIELDS:
        raise KernelSandboxError("sandbox evidence fields are not the v1 schema")
    if data["schema"] != _EVIDENCE_SCHEMA:
        raise KernelSandboxError("unsupported sandbox evidence schema")
    for flag in (
        "read_only_rootfs",
        "read_only_project",
        "cap_drop_all",
        "no_new_privileges",
        "ephemeral_workspace",
    ):
        if data[flag] is not True:
            raise KernelSandboxError(f"sandbox evidence does not prove {flag}")
    if data["network_mode"] != "none":
        raise KernelSandboxError("sandbox evidence must require network_mode=none")
    if data["runtime"] != "runc":
        raise KernelSandboxError("only the audited runc runtime is accepted")
    runner_id = _bounded_string(data["runner_id"], "runner_id", 128)
    if _RUNNER_ID.fullmatch(runner_id) is None:
        raise KernelSandboxError("sandbox runner id is invalid")
    engine_path = _absolute_posix_path(data["engine_path"], "engine_path")
    daemon_socket = _absolute_posix_path(data["daemon_socket"], "daemon_socket")
    engine_digest = _digest_value(data["engine_sha256"], "engine_sha256")
    image_reference = _bounded_string(data["image_reference"], "image_reference", 326)
    if _IMAGE_REFERENCE.fullmatch(image_reference) is None:
        raise KernelSandboxError("sandbox image must use an exact sha256 digest")
    commands = data["allowed_command_sha256"]
    if not isinstance(commands, list) or not 1 <= len(commands) <= _MAX_COMMANDS:
        raise KernelSandboxError("sandbox command allowlist is outside its bound")
    allowed_commands = tuple(
        _digest_value(item, "allowed command digest") for item in commands
    )
    if len(set(allowed_commands)) != len(allowed_commands):
        raise KernelSandboxError("sandbox command allowlist contains duplicates")
    non_root_uid = _bounded_int(data["non_root_uid"], "non_root_uid", 1, 65534)
    memory = _bounded_int(
        data["max_memory_bytes"],
        "max_memory_bytes",
        128 * 1024 * 1024,
        64 * 1024 * 1024 * 1024,
    )
    pids = _bounded_int(data["max_pids"], "max_pids", 8, 4096)
    cpus = data["max_cpus"]
    if isinstance(cpus, bool) or not isinstance(cpus, (int, float)):
        raise KernelSandboxError("max_cpus must be numeric")
    cpus = float(cpus)
    if not 0.1 <= cpus <= 64.0:
        raise KernelSandboxError("max_cpus is outside its bound")
    tmpfs = _bounded_int(
        data["tmpfs_bytes"],
        "tmpfs_bytes",
        16 * 1024 * 1024,
        16 * 1024 * 1024 * 1024,
    )
    return KernelSandboxEvidence(
        runner_id=runner_id,
        engine_path=engine_path,
        engine_sha256=engine_digest,
        daemon_socket=daemon_socket,
        image_reference=image_reference,
        allowed_command_sha256=allowed_commands,
        runtime="runc",
        non_root_uid=non_root_uid,
        max_memory_bytes=memory,
        max_pids=pids,
        max_cpus=cpus,
        tmpfs_bytes=tmpfs,
    )


class LinuxOciSandboxRunner:
    """Execute one allowlisted argv inside an evidence-bound OCI container."""

    kernel_isolated = True

    def __init__(self, evidence_path: str | Path) -> None:
        if not sys.platform.startswith("linux") or os.name != "posix":
            raise KernelSandboxError(
                "kernel-isolated candidate execution is supported only on Linux"
            )
        evidence_file = Path(evidence_path)
        _require_regular_nofollow(evidence_file, "sandbox evidence")
        raw = evidence_file.read_bytes()
        self.evidence = parse_kernel_sandbox_evidence(raw)
        self._evidence_sha256 = sha256(raw).hexdigest()
        self._engine = Path(self.evidence.engine_path)
        _require_regular_nofollow(self._engine, "container engine")
        if sha256(self._engine.read_bytes()).hexdigest() != self.evidence.engine_sha256:
            raise KernelSandboxError("container engine digest does not match evidence")
        socket_path = Path(self.evidence.daemon_socket)
        try:
            socket_mode = socket_path.stat(follow_symlinks=False).st_mode
        except OSError as exc:
            raise KernelSandboxError("container daemon socket is unavailable") from exc
        if socket_path.is_symlink() or not stat.S_ISSOCK(socket_mode):
            raise KernelSandboxError("container daemon endpoint is not a local socket")

    def isolation_attestation(self) -> RunnerAttestation:
        """Return evidence for the existing production attestation gate."""

        return RunnerAttestation(
            version=1,
            runner_id=self.evidence.runner_id,
            evidence_sha256=self._evidence_sha256,
            kernel_boundary=True,
            network_isolated=True,
            host_filesystem_isolated=True,
            ephemeral_workspace=True,
            allowed_command_sha256=self.evidence.allowed_command_sha256,
        )

    def run(
        self, project: Path, command: tuple[str, ...], timeout_seconds: int
    ) -> SandboxResult:
        """Run an exact command and reap the entire container on every exit."""

        digest = command_sha256(command)
        if digest not in self.evidence.allowed_command_sha256:
            raise KernelSandboxError("candidate command is not evidence-allowlisted")
        _require_regular_nofollow(self._engine, "container engine")
        if sha256(self._engine.read_bytes()).hexdigest() != self.evidence.engine_sha256:
            raise KernelSandboxError("container engine changed after attestation")
        if not isinstance(timeout_seconds, int) or not 1 <= timeout_seconds <= 3600:
            raise KernelSandboxError("candidate timeout is outside its bound")
        root = Path(project)
        _require_directory_nofollow(root, "candidate project")
        resolved = root.resolve(strict=True)
        if resolved != root.absolute():
            raise KernelSandboxError("candidate project traverses a symbolic link")
        mount_source = str(resolved)
        if any(item in mount_source for item in (",", "\n", "\r", "\x00")):
            raise KernelSandboxError("candidate project path is unsafe for OCI mount")

        with tempfile.TemporaryDirectory(prefix="cogni-oci-host-") as tmp:
            host_state = Path(tmp)
            cidfile = host_state / "container.cid"
            docker_config = host_state / "docker-config"
            docker_config.mkdir(mode=0o700)
            argv = self._build_argv(resolved, command, cidfile)
            env = {
                "PATH": f"{self._engine.parent}:/usr/bin:/bin",
                "HOME": str(host_state),
                "DOCKER_CONFIG": str(docker_config),
                "DOCKER_HOST": f"unix://{self.evidence.daemon_socket}",
            }
            process = subprocess.Popen(
                argv,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
                env=env,
                start_new_session=True,
            )
            timed_out = False
            try:
                stdout, stderr = process.communicate(timeout=timeout_seconds)
            except subprocess.TimeoutExpired:
                timed_out = True
                _terminate_process_group(process)
                stdout, stderr = process.communicate()
            finally:
                cleanup_error = self._force_remove(cidfile, env)
            output = (stdout + "\n" + stderr)[-_MAX_OUTPUT_CHARS:]
            if cleanup_error:
                output = (output + "\n" + cleanup_error)[-_MAX_OUTPUT_CHARS:]
            if timed_out:
                return SandboxResult(False, 124, output)
            passed = process.returncode == 0 and cleanup_error is None
            return SandboxResult(passed, int(process.returncode or 0), output)

    def _build_argv(
        self, project: Path, command: tuple[str, ...], cidfile: Path
    ) -> tuple[str, ...]:
        uid = str(self.evidence.non_root_uid)
        return (
            str(self._engine),
            "--host",
            f"unix://{self.evidence.daemon_socket}",
            "run",
            "--rm",
            "--pull",
            "never",
            "--runtime",
            self.evidence.runtime,
            "--cidfile",
            str(cidfile),
            "--network",
            "none",
            "--read-only",
            "--cap-drop",
            "ALL",
            "--security-opt",
            "no-new-privileges=true",
            "--pids-limit",
            str(self.evidence.max_pids),
            "--memory",
            str(self.evidence.max_memory_bytes),
            "--cpus",
            format(self.evidence.max_cpus, "g"),
            "--user",
            f"{uid}:{uid}",
            "--workdir",
            "/project",
            "--mount",
            f"type=bind,src={project},dst=/project,readonly",
            "--tmpfs",
            f"/tmp:rw,noexec,nosuid,nodev,size={self.evidence.tmpfs_bytes}",
            "--tmpfs",
            "/run:rw,noexec,nosuid,nodev,size=16777216",
            "--env",
            "PYTHONPATH=/project",
            "--env",
            "PYTHONDONTWRITEBYTECODE=1",
            "--env",
            "PYTHONNOUSERSITE=1",
            "--env",
            "HF_HUB_OFFLINE=1",
            "--env",
            "TRANSFORMERS_OFFLINE=1",
            "--env",
            "NVIDIA_VISIBLE_DEVICES=void",
            "--env",
            "CUDA_VISIBLE_DEVICES=",
            "--env",
            "HIP_VISIBLE_DEVICES=",
            "--env",
            "ROCR_VISIBLE_DEVICES=",
            self.evidence.image_reference,
            *command,
        )

    def _force_remove(self, cidfile: Path, env: Mapping[str, str]) -> str | None:
        if not cidfile.is_file():
            return None
        try:
            container_id = cidfile.read_text(encoding="ascii").strip()
        except (OSError, UnicodeDecodeError):
            return "container cleanup failed: unreadable cidfile"
        if re.fullmatch(r"[0-9a-f]{12,64}", container_id) is None:
            return "container cleanup failed: invalid container id"
        try:
            completed = subprocess.run(
                (
                    str(self._engine),
                    "--host",
                    f"unix://{self.evidence.daemon_socket}",
                    "rm",
                    "--force",
                    container_id,
                ),
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
                env=dict(env),
                timeout=30,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            return f"container cleanup failed: {type(exc).__name__}"
        # Docker returns 1 when --rm already removed the container.  Confirm
        # absence rather than trusting that ambiguous return code.
        try:
            inspect = subprocess.run(
                (
                    str(self._engine),
                    "--host",
                    f"unix://{self.evidence.daemon_socket}",
                    "inspect",
                    container_id,
                ),
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                env=dict(env),
                timeout=30,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            return f"container survivor check failed: {type(exc).__name__}"
        if inspect.returncode == 0:
            detail = (completed.stderr or completed.stdout).strip()[-512:]
            return f"container cleanup failed: survivor remains {detail}".strip()
        return None


def build_kernel_sandbox_evidence_payload(
    *,
    runner_id: str,
    engine_path: str,
    engine_sha256: str,
    daemon_socket: str,
    image_reference: str,
    commands: tuple[tuple[str, ...], ...],
    non_root_uid: int = 65534,
    max_memory_bytes: int = 4 * 1024 * 1024 * 1024,
    max_pids: int = 256,
    max_cpus: float = 4.0,
    tmpfs_bytes: int = 512 * 1024 * 1024,
) -> bytes:
    """Create canonical evidence bytes for independent operator review."""

    payload: dict[str, Any] = {
        "schema": _EVIDENCE_SCHEMA,
        "runner_id": runner_id,
        "engine_path": engine_path,
        "engine_sha256": engine_sha256,
        "daemon_socket": daemon_socket,
        "image_reference": image_reference,
        "allowed_command_sha256": [command_sha256(item) for item in commands],
        "runtime": "runc",
        "network_mode": "none",
        "read_only_rootfs": True,
        "read_only_project": True,
        "cap_drop_all": True,
        "no_new_privileges": True,
        "ephemeral_workspace": True,
        "non_root_uid": non_root_uid,
        "max_memory_bytes": max_memory_bytes,
        "max_pids": max_pids,
        "max_cpus": max_cpus,
        "tmpfs_bytes": tmpfs_bytes,
    }
    encoded = json.dumps(
        payload, ensure_ascii=True, sort_keys=True, separators=(",", ":")
    ).encode("ascii")
    # Reuse the strict parser so a draft can never omit a required boundary.
    parse_kernel_sandbox_evidence(encoded)
    return encoded


def _terminate_process_group(process: subprocess.Popen[str]) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=3)
    except (OSError, subprocess.TimeoutExpired):
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except OSError:
            process.kill()


def _require_regular_nofollow(path: Path, label: str) -> None:
    try:
        mode = path.stat(follow_symlinks=False).st_mode
    except OSError as exc:
        raise KernelSandboxError(f"{label} is unavailable") from exc
    if path.is_symlink() or not stat.S_ISREG(mode):
        raise KernelSandboxError(f"{label} must be a regular non-symlink file")
    if path.resolve(strict=True) != path.absolute():
        raise KernelSandboxError(f"{label} traverses a symbolic link")


def _require_directory_nofollow(path: Path, label: str) -> None:
    try:
        mode = path.stat(follow_symlinks=False).st_mode
    except OSError as exc:
        raise KernelSandboxError(f"{label} is unavailable") from exc
    if path.is_symlink() or not stat.S_ISDIR(mode):
        raise KernelSandboxError(f"{label} must be a non-symlink directory")


def _bounded_string(value: object, label: str, maximum: int) -> str:
    if not isinstance(value, str) or not value or len(value) > maximum:
        raise KernelSandboxError(f"{label} must be a bounded non-empty string")
    if any(ord(char) < 32 for char in value):
        raise KernelSandboxError(f"{label} contains a control character")
    return value


def _absolute_posix_path(value: object, label: str) -> str:
    text = _bounded_string(value, label, 512)
    path = PurePosixPath(text)
    if not path.is_absolute() or ".." in path.parts:
        raise KernelSandboxError(f"{label} must be an absolute POSIX path")
    return text


def _digest_value(value: object, label: str) -> str:
    text = _bounded_string(value, label, 64).lower()
    if _DIGEST.fullmatch(text) is None:
        raise KernelSandboxError(f"{label} must be a SHA-256 digest")
    return text


def _bounded_int(value: object, label: str, minimum: int, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise KernelSandboxError(f"{label} must be an integer")
    if not minimum <= value <= maximum:
        raise KernelSandboxError(f"{label} is outside its bound")
    return value


__all__ = [
    "KernelSandboxError",
    "KernelSandboxEvidence",
    "LinuxOciSandboxRunner",
    "build_kernel_sandbox_evidence_payload",
    "parse_kernel_sandbox_evidence",
]
