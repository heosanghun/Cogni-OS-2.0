"""Fail-closed Linux OCI sandbox for Self-Harness candidate evaluation.

The runner deliberately supports only a locally pinned image digest and a
locally pinned engine/socket.  It never mounts the container engine socket or
any GPU device into the candidate container.  This module only supplies an
integration-smoke boundary; its configuration evidence deliberately cannot
authorize production source promotion.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import os
from pathlib import Path, PurePosixPath
import re
from secrets import token_hex
import selectors
import signal
import stat
import subprocess
import sys
import tempfile
import time
from typing import TYPE_CHECKING, Any, Mapping

from .harness import SandboxResult

if TYPE_CHECKING:
    from .production import RunnerAttestation


_EVIDENCE_SCHEMA = "cogni.kernel-sandbox-evidence.v1"
_MAX_EVIDENCE_BYTES = 64 * 1024
_MAX_OUTPUT_BYTES = 40_000
_MAX_CONTROL_OUTPUT_BYTES = 4_096
_MAX_COMMANDS = 16
_MAX_PROJECT_ENTRIES = 100_000
_MAX_MOUNTINFO_BYTES = 2 * 1024 * 1024
_IMAGE_REFERENCE = re.compile(r"^[a-z0-9][a-z0-9._/:-]{0,254}@sha256:[0-9a-f]{64}$")
_DIGEST = re.compile(r"^[0-9a-f]{64}$")
_RUNNER_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_CONTAINER_NAME = re.compile(r"^cogni-candidate-[0-9a-f]{32}$")
_NO_SUCH_CONTAINER = re.compile(
    r"Error(?: response from daemon)?: No such (?:object|container): [^\r\n]+",
    re.IGNORECASE,
)
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


@dataclass(frozen=True)
class KernelSandboxRunResult(SandboxResult):
    """Sandbox result carrying an explicit, host-verified cleanup outcome."""

    cleanup_verified: bool
    container_name: str


def parse_kernel_sandbox_evidence(payload: bytes) -> KernelSandboxEvidence:
    """Parse strict-schema runner configuration without touching the host."""

    if not payload or len(payload) > _MAX_EVIDENCE_BYTES:
        raise KernelSandboxError("sandbox evidence size is outside its bound")
    try:
        data = json.loads(
            payload.decode("utf-8"), object_pairs_hook=_reject_duplicate_json_keys
        )
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

    # This runner is an implementation-level integration boundary.  It is not
    # a production attestation of the host kernel, daemon, runtime, or socket.
    kernel_isolated = False
    integration_smoke_only = True

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
        self._engine_identity = _path_identity(self._engine)
        self._socket = Path(self.evidence.daemon_socket)
        _require_socket_nofollow(self._socket, "container daemon socket")
        self._socket_identity = _path_identity(self._socket)

    def isolation_attestation(self) -> RunnerAttestation:
        """Return a deliberately non-production configuration statement.

        The evidence file binds requested Docker flags.  It does not attest the
        daemon, runtime, kernel, or security profiles, so configuration alone
        must never unlock production source promotion.
        """

        # Keep the host runner lightweight: importing production eagerly would
        # import the local-model stack (including torch) even for a standalone
        # kernel-boundary validation.
        from .production import RunnerAttestation

        return RunnerAttestation(
            version=1,
            runner_id=self.evidence.runner_id,
            evidence_sha256=self._evidence_sha256,
            kernel_boundary=False,
            network_isolated=False,
            host_filesystem_isolated=False,
            ephemeral_workspace=False,
            allowed_command_sha256=self.evidence.allowed_command_sha256,
        )

    def run(
        self, project: Path, command: tuple[str, ...], timeout_seconds: int
    ) -> SandboxResult:
        """Run an exact command and fail if cleanup cannot be proven."""

        digest = _command_sha256(command)
        if digest not in self.evidence.allowed_command_sha256:
            raise KernelSandboxError("candidate command is not evidence-allowlisted")
        self._assert_host_dependencies()
        if (
            isinstance(timeout_seconds, bool)
            or not isinstance(timeout_seconds, int)
            or not 1 <= timeout_seconds <= 3600
        ):
            raise KernelSandboxError("candidate timeout is outside its bound")
        root = Path(project)
        _require_directory_nofollow(root, "candidate project")
        resolved = root.resolve(strict=True)
        if resolved != root.absolute():
            raise KernelSandboxError("candidate project traverses a symbolic link")
        if _path_is_within(self._socket, resolved):
            raise KernelSandboxError(
                "container daemon socket is inside the project mount"
            )
        _validate_project_tree(resolved)
        mount_source = str(resolved)
        if any(item in mount_source for item in (",", "\n", "\r", "\x00")):
            raise KernelSandboxError("candidate project path is unsafe for OCI mount")

        with tempfile.TemporaryDirectory(prefix="cogni-oci-host-") as tmp:
            host_state = Path(tmp)
            cidfile = host_state / "container.cid"
            docker_config = host_state / "docker-config"
            docker_config.mkdir(mode=0o700)
            container_name = f"cogni-candidate-{token_hex(16)}"
            argv = self._build_argv(
                resolved, command, cidfile, container_name=container_name
            )
            env = self._engine_environment(host_state)
            process = subprocess.Popen(
                argv,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                bufsize=0,
                env=env,
                start_new_session=True,
            )
            raw_output = b""
            timed_out = False
            output_limited = False
            collection_error: BaseException | None = None
            try:
                raw_output, timed_out, output_limited = _collect_bounded_output(
                    process,
                    timeout_seconds=timeout_seconds,
                    maximum_bytes=_MAX_OUTPUT_BYTES,
                )
            except BaseException as exc:
                collection_error = exc
                _terminate_process_group(process)
            finally:
                cleanup_error = self._force_remove(
                    cidfile, container_name=container_name, env=env
                )
            if cleanup_error:
                raise KernelSandboxError(cleanup_error) from collection_error
            if collection_error is not None:
                raise collection_error
            output = raw_output.decode("utf-8", errors="replace")
            if timed_out:
                return KernelSandboxRunResult(False, 124, output, True, container_name)
            if output_limited:
                return KernelSandboxRunResult(
                    False,
                    125,
                    f"{output}\ncandidate output exceeded {_MAX_OUTPUT_BYTES} bytes",
                    True,
                    container_name,
                )
            passed = process.returncode == 0 and cleanup_error is None
            return KernelSandboxRunResult(
                passed,
                int(process.returncode or 0),
                output,
                True,
                container_name,
            )

    def _build_argv(
        self,
        project: Path,
        command: tuple[str, ...],
        cidfile: Path,
        *,
        container_name: str,
    ) -> tuple[str, ...]:
        if _CONTAINER_NAME.fullmatch(container_name) is None:
            raise KernelSandboxError("internal container name is invalid")
        uid = str(self.evidence.non_root_uid)
        return (
            str(self._engine),
            "--host",
            f"unix://{self.evidence.daemon_socket}",
            "run",
            "--rm",
            "--name",
            container_name,
            "--label",
            f"cogni.sandbox.run={container_name}",
            "--pull",
            "never",
            "--no-healthcheck",
            "--log-driver",
            "none",
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
            "--memory-swap",
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
            "--entrypoint",
            command[0],
            self.evidence.image_reference,
            *command[1:],
        )

    def _force_remove(
        self,
        cidfile: Path,
        *,
        container_name: str,
        env: Mapping[str, str],
    ) -> str | None:
        """Remove a tracked container and prove both name and CID are absent."""

        if _CONTAINER_NAME.fullmatch(container_name) is None:
            return "container cleanup failed: invalid tracked container name"
        container_id: str | None = None
        cid_error: str | None = None
        if not cidfile.exists():
            cid_error = "container cleanup unverified: cidfile is missing"
        else:
            try:
                _require_regular_nofollow(cidfile, "container cidfile")
                if cidfile.stat().st_size > 128:
                    raise KernelSandboxError("container cidfile is oversized")
                container_id = cidfile.read_text(encoding="ascii").strip()
            except (OSError, UnicodeDecodeError, KernelSandboxError):
                cid_error = "container cleanup unverified: unreadable cidfile"
            if (
                container_id is not None
                and re.fullmatch(r"[0-9a-f]{12,64}", container_id) is None
            ):
                cid_error = "container cleanup unverified: invalid container id"
                container_id = None
        try:
            returncode, detail = self._docker_control(
                ("rm", "--force", container_name), env
            )
        except KernelSandboxError as exc:
            return f"container cleanup failed: {exc}"
        if returncode != 0 and not _is_exact_not_found(detail):
            return f"container cleanup failed: docker rm was ambiguous {detail[-512:]}".strip()
        identifiers = [container_name]
        if container_id is not None:
            identifiers.append(container_id)
        for attempt in range(2):
            for identifier in identifiers:
                error = self._container_absence_error(identifier, env)
                if error is not None:
                    return error
            if attempt == 0:
                time.sleep(0.1)
        return cid_error

    def verify_container_absent(self, container_name: str) -> None:
        """Independently fail unless a tracked integration container is absent."""

        if _CONTAINER_NAME.fullmatch(container_name) is None:
            raise KernelSandboxError("tracked container name is invalid")
        self._assert_host_dependencies()
        with tempfile.TemporaryDirectory(prefix="cogni-oci-inspect-") as tmp:
            host_state = Path(tmp)
            (host_state / "docker-config").mkdir(mode=0o700)
            error = self._container_absence_error(
                container_name, self._engine_environment(host_state)
            )
        if error is not None:
            raise KernelSandboxError(error)

    def _container_absence_error(
        self, identifier: str, env: Mapping[str, str]
    ) -> str | None:
        try:
            returncode, detail = self._docker_control(
                ("inspect", "--type", "container", identifier), env
            )
        except KernelSandboxError as exc:
            return f"container survivor check failed: {exc}"
        if returncode == 0:
            return f"container cleanup failed: survivor remains {identifier}"
        if not _is_exact_not_found(detail):
            return f"container survivor check ambiguous: {detail[-512:]}".strip()
        return None

    def _docker_control(
        self, arguments: tuple[str, ...], env: Mapping[str, str]
    ) -> tuple[int, str]:
        self._assert_host_dependencies()
        process = subprocess.Popen(
            (
                str(self._engine),
                "--host",
                f"unix://{self.evidence.daemon_socket}",
                *arguments,
            ),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=0,
            env=dict(env),
            start_new_session=True,
        )
        output, timed_out, output_limited = _collect_bounded_output(
            process, timeout_seconds=30, maximum_bytes=_MAX_CONTROL_OUTPUT_BYTES
        )
        detail = output.decode("utf-8", errors="replace").strip()
        if timed_out:
            raise KernelSandboxError("Docker control command timed out")
        if output_limited:
            raise KernelSandboxError("Docker control output exceeded its bound")
        return int(process.returncode or 0), detail

    def _assert_host_dependencies(self) -> None:
        _require_regular_nofollow(self._engine, "container engine")
        if _path_identity(self._engine) != self._engine_identity:
            raise KernelSandboxError("container engine identity changed")
        if sha256(self._engine.read_bytes()).hexdigest() != self.evidence.engine_sha256:
            raise KernelSandboxError(
                "container engine changed after configuration review"
            )
        _require_socket_nofollow(self._socket, "container daemon socket")
        if _path_identity(self._socket) != self._socket_identity:
            raise KernelSandboxError("container daemon socket identity changed")

    def _engine_environment(self, host_state: Path) -> dict[str, str]:
        return {
            "PATH": f"{self._engine.parent}:/usr/bin:/bin",
            "HOME": str(host_state),
            "DOCKER_CONFIG": str(host_state / "docker-config"),
            "DOCKER_HOST": f"unix://{self.evidence.daemon_socket}",
        }


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
        "allowed_command_sha256": [_command_sha256(item) for item in commands],
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


def _terminate_process_group(process: subprocess.Popen[Any]) -> None:
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except OSError:
        if process.poll() is None:
            process.terminate()
    try:
        if process.poll() is None:
            process.wait(timeout=1)
    except subprocess.TimeoutExpired:
        pass
    # The session may still contain a descendant holding a pipe after its
    # leader exits.  Always make one final group-wide kill attempt.
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except OSError:
        if process.poll() is None:
            process.kill()
    if process.poll() is None:
        try:
            process.wait(timeout=3)
        except subprocess.TimeoutExpired as exc:
            raise KernelSandboxError("host process group could not be reaped") from exc


def _collect_bounded_output(
    process: subprocess.Popen[bytes],
    *,
    timeout_seconds: int,
    maximum_bytes: int,
) -> tuple[bytes, bool, bool]:
    """Collect one combined pipe without ever buffering beyond its hard cap."""

    if process.stdout is None:
        raise KernelSandboxError("candidate process exposes no output pipe")
    if maximum_bytes < 1:
        raise KernelSandboxError("output bound must be positive")
    deadline = time.monotonic() + timeout_seconds
    output = bytearray()
    timed_out = False
    output_limited = False
    termination_deadline: float | None = None
    pipe_open = True
    selector = selectors.DefaultSelector()
    selector.register(process.stdout, selectors.EVENT_READ)
    try:
        while pipe_open or process.poll() is None:
            now = time.monotonic()
            if not timed_out and not output_limited and now >= deadline:
                timed_out = True
                _terminate_process_group(process)
                termination_deadline = time.monotonic() + 3
            if termination_deadline is not None and now >= termination_deadline:
                raise KernelSandboxError(
                    "host output pipe did not close after termination"
                )
            if not pipe_open:
                if process.poll() is None:
                    time.sleep(0.01)
                continue
            remaining = max(0.0, deadline - time.monotonic())
            events = selector.select(timeout=min(0.1, remaining))
            if not events:
                continue
            for key, _ in events:
                room = maximum_bytes - len(output)
                read_size = 8192 if room == 0 else min(8192, room + 1)
                chunk = os.read(key.fd, read_size)
                if not chunk:
                    selector.unregister(process.stdout)
                    pipe_open = False
                    break
                output.extend(chunk[:room])
                if len(chunk) > room and not output_limited:
                    output_limited = True
                    _terminate_process_group(process)
                    termination_deadline = time.monotonic() + 3
        if process.poll() is None:
            process.wait(timeout=3)
    finally:
        selector.close()
        process.stdout.close()
    return bytes(output), timed_out, output_limited


def _reject_duplicate_json_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise KernelSandboxError(f"sandbox evidence contains duplicate key: {key}")
        result[key] = value
    return result


def _is_exact_not_found(detail: str) -> bool:
    lines = detail.strip().splitlines()
    if lines[:1] == ["[]"]:
        lines = lines[1:]
    if len(lines) != 1:
        return False
    return _NO_SUCH_CONTAINER.fullmatch(lines[0]) is not None


def _path_identity(path: Path) -> tuple[int, int, int, int, int]:
    try:
        value = path.stat(follow_symlinks=False)
    except OSError as exc:
        raise KernelSandboxError(f"host dependency disappeared: {path}") from exc
    return (
        int(value.st_dev),
        int(value.st_ino),
        int(value.st_uid),
        int(value.st_gid),
        int(value.st_mode),
    )


def _require_socket_nofollow(path: Path, label: str) -> None:
    try:
        mode = path.stat(follow_symlinks=False).st_mode
    except OSError as exc:
        raise KernelSandboxError(f"{label} is unavailable") from exc
    if path.is_symlink() or not stat.S_ISSOCK(mode):
        raise KernelSandboxError(f"{label} must be a local socket without symlinks")
    if path.resolve(strict=True) != path.absolute():
        raise KernelSandboxError(f"{label} traverses a symbolic link")


def _path_is_within(path: Path, root: Path) -> bool:
    try:
        path.resolve(strict=True).relative_to(root)
    except ValueError:
        return False
    return True


def _validate_project_tree(root: Path) -> None:
    """Reject symlinks, special files, and already-mounted descendants."""

    _reject_nested_mounts(root)
    root_device = root.stat(follow_symlinks=False).st_dev
    stack = [root]
    entries = 0
    while stack:
        directory = stack.pop()
        try:
            with os.scandir(directory) as iterator:
                children = tuple(iterator)
        except OSError as exc:
            raise KernelSandboxError(
                "candidate project tree changed during review"
            ) from exc
        for child in children:
            entries += 1
            if entries > _MAX_PROJECT_ENTRIES:
                raise KernelSandboxError(
                    "candidate project tree exceeds its entry bound"
                )
            try:
                value = child.stat(follow_symlinks=False)
            except OSError as exc:
                raise KernelSandboxError(
                    "candidate project entry changed during review"
                ) from exc
            mode = value.st_mode
            if stat.S_ISLNK(mode):
                raise KernelSandboxError(
                    f"candidate project contains a symbolic link: {child.path}"
                )
            if value.st_dev != root_device:
                raise KernelSandboxError(
                    f"candidate project crosses a filesystem boundary: {child.path}"
                )
            if stat.S_ISDIR(mode):
                stack.append(Path(child.path))
            elif not stat.S_ISREG(mode):
                raise KernelSandboxError(
                    f"candidate project contains a special file: {child.path}"
                )


def _reject_nested_mounts(root: Path) -> None:
    mountinfo = Path("/proc/self/mountinfo")
    try:
        raw = mountinfo.read_bytes()
    except OSError as exc:
        raise KernelSandboxError("host mount topology is unavailable") from exc
    if len(raw) > _MAX_MOUNTINFO_BYTES:
        raise KernelSandboxError("host mount topology exceeds its bound")
    try:
        lines = raw.decode("utf-8").splitlines()
    except UnicodeDecodeError as exc:
        raise KernelSandboxError("host mount topology is not UTF-8") from exc
    for line in lines:
        fields = line.split(" ")
        if len(fields) < 6:
            raise KernelSandboxError("host mount topology is malformed")
        mount_point = Path(_decode_mountinfo_path(fields[4]))
        try:
            mount_point.relative_to(root)
        except ValueError:
            continue
        if mount_point != root:
            raise KernelSandboxError(
                f"candidate project contains a nested mount: {mount_point}"
            )


def _decode_mountinfo_path(value: str) -> str:
    def replace(match: re.Match[str]) -> str:
        return chr(int(match.group(1), 8))

    return re.sub(r"\\([0-7]{3})", replace, value)


def _command_sha256(command: tuple[str, ...]) -> str:
    if (
        not isinstance(command, tuple)
        or not command
        or len(command) > 64
        or any(
            not isinstance(item, str) or not item or len(item) > 4096 or "\x00" in item
            for item in command
        )
    ):
        raise KernelSandboxError("candidate command is not a bounded argv tuple")
    payload = b"".join(
        len(item.encode("utf-8")).to_bytes(4, "big") + item.encode("utf-8")
        for item in command
    )
    return sha256(payload).hexdigest()


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
    "KernelSandboxRunResult",
    "LinuxOciSandboxRunner",
    "build_kernel_sandbox_evidence_payload",
    "parse_kernel_sandbox_evidence",
]
