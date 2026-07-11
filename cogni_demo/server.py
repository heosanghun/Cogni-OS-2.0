"""Standard-library, loopback-only control plane for the graphical demo."""

from __future__ import annotations

from collections import deque
from collections.abc import Callable, Iterator, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
from http import HTTPStatus
from http.cookies import SimpleCookie
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from http.client import HTTPConnection
import argparse
import hmac
import json
import os
from pathlib import Path
from queue import Empty, Full, Queue
import secrets
import shutil
import signal
import stat
import subprocess
import sys
from threading import Condition, Event, RLock, Thread
from time import monotonic, sleep
from typing import Any, BinaryIO
from urllib.parse import parse_qs, urlsplit
import webbrowser

from cogni_demo.protocol import (
    EVENT_SENTINEL,
    MAX_EVENT_LINE_BYTES,
    PHASE_STAGES,
    ProtocolError,
    WorkerEvent,
    parse_event_line,
    validate_terminal_metrics,
)


MAX_REQUEST_BODY_BYTES = 8 * 1024
MAX_PROMPT_LENGTH = 256
MAX_STATE_EVENTS = 64
MAX_DIAGNOSTIC_LINES = 200
MAX_EVENT_QUEUE = 64
READ_CHUNK_BYTES = 4096
MAX_ASSET_BYTES = 2 * 1024 * 1024
DEFAULT_PORT = 8765
DEFAULT_WATCHDOG_SECONDS = 60.0
MAX_SESSION_BYTES = 4096
SESSION_VERSION = 1
SERVICE_MARKER = "cogniboard"

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
    "connect-src 'self'; img-src 'self' data:; object-src 'none'; "
    "base-uri 'none'; frame-ancestors 'none'; form-action 'none'"
)


INITIAL_METRICS: dict[str, Any] = {
    "evidence_kind": "measured_internal",
    "measured_at": "2026-07-11T00:00:00+09:00",
    "source": "scripts/validate_gemma4_runtime.py internal hardware validation",
    "peak_vram_gib": 14.8560,
    "vram_limit_gib": 16.7,
    "requested_depth": 100,
    "reached_depth": 100,
    "nodes_used": 301,
    "node_capacity": 301,
    "transition_residual": 0.00390625,
    "transition_converged": True,
    "transition_used_fallback": False,
    "finite": True,
    "verified_files": 6,
    "model_class": "Gemma4ForConditionalGeneration",
    "hidden_size": 2560,
    "search_allocated_bytes": 14_994_009,
    "load_seconds": 19.789,
    "inference_seconds": 3.729,
    "tests": 178,
    "subtests": 32,
    "device": "RTX 5090 Laptop GPU",
    "target": "RTX 4090 24GB",
}


class DemoServerError(RuntimeError):
    """Base class for bounded control-plane failures."""


class JobAlreadyRunningError(DemoServerError):
    pass


class NoActiveJobError(DemoServerError):
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


def production_launch_factory(
    project_root: str | Path,
    model_directory: str | Path,
    manifest: str | Path,
    *,
    python_executable: str | Path = sys.executable,
) -> LaunchFactory:
    """Create the fixed, shell-free command for the sole CUDA owner."""

    root = Path(project_root).resolve(strict=True)
    worker = (root / "scripts" / "validate_gemma4_runtime.py").resolve(strict=True)
    model = Path(model_directory).resolve(strict=True)
    manifest_path = Path(manifest).resolve(strict=True)
    if not root.is_dir() or not model.is_dir() or not manifest_path.is_file():
        raise ValueError("production demo paths are incomplete")
    executable = str(Path(python_executable).resolve(strict=True))

    def build(prompt: str) -> WorkerLaunch:
        command = [
            executable,
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
        if prompt:
            command.extend(("--prompt", prompt))
        environment = os.environ.copy()
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
    ) -> None:
        if max_runtime_seconds <= 0:
            raise ValueError("max_runtime_seconds must be positive")
        self._launch_factory = launch_factory
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
        with self._condition:
            active = self._status in _ACTIVE_STATUSES
        if active:
            try:
                self.cancel()
            except NoActiveJobError:
                pass
        thread = self._worker_thread
        if thread is not None:
            thread.join(timeout=timeout)
        process = self._process
        if process is not None and process.poll() is None:
            self._terminate(process)

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
        messages: Queue[tuple[str, object]] = Queue(maxsize=MAX_EVENT_QUEUE)
        protocol_failure: str | None = None
        terminal_metrics: dict[str, Any] | None = None
        terminal_count = 0
        expected_worker_sequence = 1
        expected_phase_index = 0
        last_progress = -1
        readers: list[Thread] = []
        started = monotonic()
        try:
            launch = self._launch_factory(prompt)
            if not launch.command or any(
                not isinstance(item, str) for item in launch.command
            ):
                raise ValueError("worker command must be a non-empty string sequence")
            if self._cancel_event.is_set():
                self._finish_job(job_id, "cancelled", "cancelled", 0)
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
                self._finish_job(job_id, "cancelled", "cancelled", self._progress)
            elif return_code != 0:
                self._finish_job(
                    job_id,
                    "failed",
                    "worker_failed",
                    self._progress,
                    error={
                        "code": "WORKER_EXIT",
                        "message": f"validation worker exited with code {return_code}",
                    },
                )
            elif (
                protocol_failure is not None
                or terminal_count != 1
                or terminal_metrics is None
            ):
                detail = protocol_failure or "worker emitted no unique terminal result"
                self._finish_job(
                    job_id,
                    "failed",
                    "protocol_failed",
                    self._progress,
                    error={"code": "WORKER_PROTOCOL", "message": detail[:256]},
                )
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
                with self._condition:
                    self._metrics = live_metrics
                self._finish_job(job_id, "succeeded", "complete", 100)
        except BaseException as exc:
            if process is not None and process.poll() is None:
                self._terminate(process)
            if self._cancel_event.is_set():
                self._finish_job(job_id, "cancelled", "cancelled", self._progress)
                return
            self._finish_job(
                job_id,
                "failed",
                "server_failed",
                self._progress,
                error={
                    "code": "SERVER_WORKER_FAILURE",
                    "message": f"{type(exc).__name__}: worker could not be managed"[
                        :256
                    ],
                },
            )
        finally:
            with self._condition:
                if self._process is process:
                    self._process = None

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
            return
        except (OSError, subprocess.TimeoutExpired):
            pass
        try:
            process.terminate()
            process.wait(timeout=1.0)
            return
        except (OSError, subprocess.TimeoutExpired):
            pass
        try:
            process.kill()
            process.wait(timeout=1.0)
        except (OSError, subprocess.TimeoutExpired):
            pass


class DemoHTTPServer(ThreadingHTTPServer):
    """HTTP server that refuses non-loopback binding at construction."""

    daemon_threads = True
    allow_reuse_address = False

    def __init__(
        self,
        manager: JobManager,
        asset_directory: str | Path,
        *,
        port: int = DEFAULT_PORT,
        token: str | None = None,
        watchdog_timeout: float | None = DEFAULT_WATCHDOG_SECONDS,
    ) -> None:
        if not 0 <= int(port) <= 65535:
            raise ValueError("port must be in [0, 65535]")
        self.manager = manager
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
        self._watchdog_stop = Event()
        self._watchdog_thread: Thread | None = None
        self._last_state_poll = monotonic()
        self._shutdown_requested = False
        super().__init__(("127.0.0.1", int(port)), DemoRequestHandler)

    @property
    def origin(self) -> str:
        return f"http://127.0.0.1:{self.server_port}"

    @property
    def bootstrap_url(self) -> str:
        return f"{self.origin}/?token={self.token}"

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

    def request_shutdown(self) -> None:
        with self._lifecycle_lock:
            if self._shutdown_requested:
                return
            self._shutdown_requested = True
            self._watchdog_stop.set()

        def stop() -> None:
            self.manager.shutdown()
            self.shutdown()

        Thread(target=stop, name="cogni-demo-shutdown", daemon=True).start()

    def server_close(self) -> None:
        self._watchdog_stop.set()
        super().server_close()


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
        }:
            self._json_error(HTTPStatus.NOT_FOUND, "NOT_FOUND")
            return
        body = self._read_json_body()
        if body is None:
            return
        if parsed.path == "/api/run":
            if not set(body) <= {"prompt"}:
                self._json_error(HTTPStatus.BAD_REQUEST, "INVALID_BODY")
                return
            try:
                job_id = self.server.manager.start(body.get("prompt", ""))
            except JobAlreadyRunningError:
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

    def _read_json_body(self) -> dict[str, Any] | None:
        if self.headers.get_content_type() != "application/json":
            self._json_error(HTTPStatus.UNSUPPORTED_MEDIA_TYPE, "JSON_REQUIRED")
            return None
        raw_length = self.headers.get("Content-Length")
        try:
            length = int(raw_length or "")
        except ValueError:
            self._json_error(HTTPStatus.LENGTH_REQUIRED, "CONTENT_LENGTH_REQUIRED")
            return None
        if not 0 <= length <= MAX_REQUEST_BODY_BYTES:
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
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("X-Frame-Options", "DENY")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("Cache-Control", "no-store")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m cogni_demo.server")
    project_root = Path(__file__).resolve().parents[1]
    parser.add_argument(
        "--model",
        default=os.environ.get("COGNI_OS_MODEL_DIR", r"C:\Project\cognios\gemma4-e4b"),
    )
    parser.add_argument(
        "--manifest", default=str(project_root / "config" / "gemma4-e4b.manifest.toml")
    )
    parser.add_argument("--assets", default=str(project_root / "cogni_demo" / "static"))
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args(argv)

    session_path = default_session_path()
    existing = find_live_session(session_path)
    if existing is not None:
        print(f"existing_demo_url=http://127.0.0.1:{existing.port}/", flush=True)
        if not args.no_browser:
            open_graphical_app(existing.bootstrap_url)
        return 0

    manager = JobManager(
        production_launch_factory(project_root, args.model, args.manifest)
    )
    try:
        server = DemoHTTPServer(manager, args.assets, port=args.port)
    except OSError:
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
                if not args.no_browser:
                    open_graphical_app(existing.bootstrap_url)
                return 0
        raise

    metadata = SessionMetadata(
        os.getpid(),
        server.server_port,
        server.token,
        datetime.now(timezone.utc).isoformat(),
    )
    try:
        write_session_metadata(metadata, session_path)
    except BaseException:
        manager.shutdown()
        server.server_close()
        raise
    print(f"demo_url={server.origin}/", flush=True)
    if not args.no_browser:
        open_graphical_app(server.bootstrap_url)
    try:
        server.serve_forever(poll_interval=0.2)
    except KeyboardInterrupt:
        pass
    finally:
        manager.shutdown()
        server.server_close()
        remove_session_metadata(session_path, expected=metadata)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "DemoHTTPServer",
    "DemoServerError",
    "INITIAL_METRICS",
    "JobAlreadyRunningError",
    "JobManager",
    "NoActiveJobError",
    "WorkerLaunch",
    "main",
    "production_launch_factory",
]
