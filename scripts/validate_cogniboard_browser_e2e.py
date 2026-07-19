"""Bounded, dependency-free CogniBoard Firefox E2E validator.

The default invocation never starts a browser.  A real run requires ``--run``
and fails closed unless both of these host-enforced boundaries are already
true:

* neither ``/dev/dri`` nor any ``/dev/nvidia*`` path is visible; and
* the network namespace exposes only the loopback interface.

The validator serves the *current production static assets* from an
authenticated loopback-only fixture.  Its fake agent/workspace/voice APIs are
deliberately deterministic: this is a browser rendering and interaction gate,
not model or GPU evidence.  Firefox is controlled through geckodriver's W3C
HTTP protocol using only the Python standard library.
"""

from __future__ import annotations

import argparse
from base64 import b64decode
import binascii
from dataclasses import dataclass, field
from datetime import UTC, datetime
from hashlib import sha256
from http import HTTPStatus
from http.cookies import SimpleCookie
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import re
import secrets
import signal
import socket
import stat
import subprocess
import sys
from tempfile import TemporaryDirectory
from threading import Thread
from time import monotonic, sleep
from typing import Any, Callable, Mapping
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qs, urlsplit
from urllib.request import HTTPRedirectHandler, ProxyHandler, Request, build_opener


RESULT_SCHEMA = "cogni.browser-e2e.v1"
FIXTURE_SCHEMA = "cogni.browser-e2e.fixture.v1"
VIEWPORTS = ((1366, 768), (1920, 1080))
MAX_HTTP_BYTES = 2 * 1024 * 1024
MAX_DRIVER_RESPONSE_BYTES = 4 * 1024 * 1024
MAX_DRIVER_LOG_CHARS = 64 * 1024
SESSION_COOKIE = "cogni_browser_e2e"
ASSISTANT_COMPLETION = (
    "브라우저 E2E 검증 응답입니다. 요청을 한 번만 처리했고 문장을 완전하게 마쳤습니다."
)
VOICE_TRANSCRIPT = "음성 첫 사용 검증 문장"
W3C_ELEMENT_KEY = "element-6066-11e4-a52e-4f735466cecf"
STARTUP_ERROR_SCRIPT = b"""
window.__cogniE2EErrors = [];
window.__cogniE2EHookInstalled = true;
window.addEventListener('error', event => {
  window.__cogniE2EErrors.push(String(event.message || 'window error').slice(0, 512));
});
window.addEventListener('unhandledrejection', event => {
  window.__cogniE2EErrors.push(String(event.reason || 'unhandled rejection').slice(0, 512));
});
"""
_COMMIT_RE = re.compile(r"[0-9a-f]{40}")
_WEBDRIVER_ID_RE = re.compile(r"[A-Za-z0-9._-]{1,128}")
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_PASS_CHECK_NAMES = {
    "executable_identity",
    "browser_capability_identity",
    "viewport_fail_closed_1366x768",
    "viewport_fail_closed_1920x1080",
    "capabilities_disabled",
    "viewport_enabled_1366x768",
    "viewport_enabled_1920x1080",
    "capabilities_enabled",
    "optional_ui_interactions",
    "single_complete_conversation",
    "assistant_not_repeated",
    "javascript_errors",
    "webdriver_loopback_only",
    "fixture_requests_same_origin",
    "boundary_rechecked",
}


class _RejectRedirects(HTTPRedirectHandler):
    """Never let an initially-loopback WebDriver request follow a redirect."""

    def redirect_request(self, *_args: object, **_kwargs: object):
        return None


class BrowserE2EError(RuntimeError):
    """Stable, user-safe validator failure."""


class WebDriverProtocolError(BrowserE2EError):
    """A bounded W3C WebDriver request failed."""


def _utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _is_loopback_url(value: str) -> bool:
    parsed = urlsplit(value)
    return (
        parsed.scheme == "http"
        and parsed.hostname == "127.0.0.1"
        and isinstance(parsed.port, int)
        and parsed.username is None
        and parsed.password is None
        and parsed.fragment == ""
    )


@dataclass(frozen=True)
class ExecutionBoundary:
    """Observed process boundary required before a browser can start."""

    gpu_device_paths: tuple[str, ...]
    network_interfaces: tuple[str, ...]
    environment_violations: tuple[str, ...]

    @property
    def gpu_device_free(self) -> bool:
        return not self.gpu_device_paths

    @property
    def loopback_only(self) -> bool:
        return self.network_interfaces == ("lo",)

    @property
    def ready(self) -> bool:
        return (
            self.gpu_device_free
            and self.loopback_only
            and not self.environment_violations
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "gpu_device_free": self.gpu_device_free,
            "gpu_device_paths": list(self.gpu_device_paths),
            "loopback_only": self.loopback_only,
            "network_interfaces": list(self.network_interfaces),
            "environment_violations": list(self.environment_violations),
            "ready": self.ready,
        }


def inspect_execution_boundary(
    *,
    dev_root: str | Path = "/dev",
    network_root: str | Path = "/sys/class/net",
    environment: Mapping[str, str] | None = None,
) -> ExecutionBoundary:
    """Inspect device/network visibility without probing any GPU."""

    selected_dev = Path(dev_root)
    selected_network = Path(network_root)
    gpu_paths: list[str] = []
    dri = selected_dev / "dri"
    if dri.exists() or dri.is_symlink():
        gpu_paths.append(str(dri))
    try:
        device_entries = tuple(selected_dev.iterdir())
    except OSError:
        device_entries = ()
    for entry in device_entries:
        if entry.name.startswith("nvidia"):
            gpu_paths.append(str(entry))

    try:
        interfaces = tuple(sorted(item.name for item in selected_network.iterdir()))
    except OSError:
        interfaces = ()

    env = os.environ if environment is None else environment
    violations: list[str] = []
    nvidia_visible = env.get("NVIDIA_VISIBLE_DEVICES", "").strip().lower()
    if nvidia_visible not in {"", "none", "void"}:
        violations.append("NVIDIA_VISIBLE_DEVICES")
    cuda_visible = env.get("CUDA_VISIBLE_DEVICES", "").strip().lower()
    if cuda_visible not in {"", "-1", "none", "void"}:
        violations.append("CUDA_VISIBLE_DEVICES")
    return ExecutionBoundary(
        tuple(sorted(set(gpu_paths))),
        interfaces,
        tuple(sorted(violations)),
    )


def require_execution_boundary(boundary: ExecutionBoundary) -> None:
    if boundary.ready:
        return
    reasons: list[str] = []
    if not boundary.gpu_device_free:
        reasons.append("GPU_DEVICE_VISIBLE")
    if not boundary.loopback_only:
        reasons.append("NETWORK_NOT_LOOPBACK_ONLY")
    reasons.extend(boundary.environment_violations)
    raise BrowserE2EError("unsafe browser boundary: " + ",".join(reasons))


Transport = Callable[[Request, float], Any]


@dataclass(frozen=True)
class ExecutableIdentity:
    path: str
    sha256: str
    size_bytes: int
    device: int
    inode: int

    def to_dict(self) -> dict[str, object]:
        return {
            "path": self.path,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
            "device": self.device,
            "inode": self.inode,
        }


def inspect_executable(path: Path, *, label: str) -> ExecutableIdentity:
    """Hash one canonical, non-link regular executable and detect read races."""

    if not path.is_absolute():
        raise BrowserE2EError(f"{label} path must be absolute")
    try:
        before = os.lstat(path)
        resolved = path.resolve(strict=True)
    except OSError as error:
        raise BrowserE2EError(f"{label} path is invalid") from error
    if (
        resolved != path
        or stat.S_ISLNK(before.st_mode)
        or not stat.S_ISREG(before.st_mode)
    ):
        raise BrowserE2EError(f"{label} must be one canonical non-symlink file")
    if os.name == "posix" and not os.access(path, os.X_OK):
        raise BrowserE2EError(f"{label} is not executable")
    digest = sha256()
    try:
        with path.open("rb") as stream:
            opened = os.fstat(stream.fileno())
            while chunk := stream.read(1024 * 1024):
                digest.update(chunk)
        after = os.lstat(path)
    except OSError as error:
        raise BrowserE2EError(f"{label} could not be attested") from error
    identity = (before.st_dev, before.st_ino, before.st_size)
    if identity != (opened.st_dev, opened.st_ino, opened.st_size) or identity != (
        after.st_dev,
        after.st_ino,
        after.st_size,
    ):
        raise BrowserE2EError(f"{label} changed while it was attested")
    return ExecutableIdentity(
        path=str(resolved),
        sha256=digest.hexdigest(),
        size_bytes=before.st_size,
        device=before.st_dev,
        inode=before.st_ino,
    )


def attest_browser_process(
    client: "W3CWebDriverClient", firefox: ExecutableIdentity
) -> dict[str, object]:
    """Bind returned Firefox capabilities to the exact executable on Linux."""

    pid = client.browser_process_id
    if os.name != "posix" or not isinstance(pid, int) or pid <= 1:
        raise BrowserE2EError("Firefox process identity is unavailable")
    proc_exe = Path(f"/proc/{pid}/exe")
    try:
        resolved = proc_exe.resolve(strict=True)
        observed = os.stat(proc_exe)
    except OSError as error:
        raise BrowserE2EError("Firefox process executable is unavailable") from error
    if (
        str(resolved) != firefox.path
        or observed.st_dev != firefox.device
        or observed.st_ino != firefox.inode
    ):
        raise BrowserE2EError("Firefox process executable identity did not match")
    return {
        "browser_name": client.capabilities["browserName"],
        "browser_version": str(client.capabilities["browserVersion"])[:128],
        "process_id": pid,
        "binary_sha256": firefox.sha256,
    }


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def cleanup_process_tree(
    process: subprocess.Popen[Any], *, browser_pid: int | None
) -> None:
    """Best-effort TERM/KILL of the isolated driver group and browser PID."""

    if os.name != "posix":
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=3)
        return

    targets = {pid for pid in (browser_pid,) if isinstance(pid, int) and pid > 1}
    for selected_signal in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(process.pid, selected_signal)
        except ProcessLookupError:
            pass
        for pid in targets:
            if _pid_alive(pid):
                try:
                    os.kill(pid, selected_signal)
                except ProcessLookupError:
                    pass
        deadline = monotonic() + (3.0 if selected_signal == signal.SIGTERM else 2.0)
        while monotonic() < deadline:
            driver_dead = process.poll() is not None
            browser_dead = all(not _pid_alive(pid) for pid in targets)
            if driver_dead and browser_dead:
                return
            sleep(0.05)
    if process.poll() is None or any(_pid_alive(pid) for pid in targets):
        raise BrowserE2EError("browser process tree did not terminate")


class W3CWebDriverClient:
    """Small bounded W3C client; all traffic is restricted to 127.0.0.1."""

    def __init__(
        self,
        endpoint: str,
        *,
        timeout: float = 10.0,
        transport: Transport | None = None,
    ) -> None:
        endpoint = endpoint.rstrip("/")
        parsed_endpoint = urlsplit(endpoint)
        if (
            not _is_loopback_url(endpoint)
            or parsed_endpoint.path not in {"", "/"}
            or parsed_endpoint.query
        ):
            raise ValueError("WebDriver endpoint must be http://127.0.0.1:<port>")
        if timeout <= 0 or timeout > 60:
            raise ValueError("timeout must be in (0, 60]")
        self.endpoint = f"http://127.0.0.1:{parsed_endpoint.port}"
        self.timeout = float(timeout)
        self._transport = transport or self._default_transport
        self.session_id: str | None = None
        self.capabilities: dict[str, object] = {}
        self.browser_process_id: int | None = None
        self.request_urls: list[str] = []

    @staticmethod
    def _default_transport(request: Request, timeout: float):
        return build_opener(ProxyHandler({}), _RejectRedirects()).open(
            request, timeout=timeout
        )

    def request(
        self,
        method: str,
        path: str,
        payload: object | None = None,
    ) -> object:
        if not path.startswith("/") or path.startswith("//"):
            raise ValueError("WebDriver path must be origin-relative")
        url = self.endpoint + path
        if not _is_loopback_url(url):
            raise ValueError("WebDriver request escaped loopback")
        body = None
        headers = {"Accept": "application/json"}
        if payload is not None:
            body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
            if len(body) > MAX_HTTP_BYTES:
                raise ValueError("WebDriver request is too large")
            headers["Content-Type"] = "application/json"
        request = Request(url, data=body, headers=headers, method=method)
        self.request_urls.append(url)
        try:
            response = self._transport(request, self.timeout)
            with response:
                raw = response.read(MAX_DRIVER_RESPONSE_BYTES + 1)
                status = int(getattr(response, "status", 200))
                final_url = str(response.geturl())
                headers_object = getattr(response, "headers", None)
                content_type = (
                    headers_object.get_content_type()
                    if headers_object is not None
                    and callable(getattr(headers_object, "get_content_type", None))
                    else ""
                )
        except HTTPError as error:
            raw = error.read(MAX_DRIVER_RESPONSE_BYTES + 1)
            status = error.code
            final_url = str(error.geturl())
            content_type = error.headers.get_content_type() if error.headers else ""
        except (OSError, URLError) as error:
            raise WebDriverProtocolError("WebDriver connection failed") from error
        if final_url != url:
            raise WebDriverProtocolError("WebDriver response escaped its exact origin")
        if not 200 <= status < 300:
            raise WebDriverProtocolError(f"unexpected WebDriver HTTP status {status}")
        if content_type != "application/json":
            raise WebDriverProtocolError("WebDriver response is not application/json")
        if len(raw) > MAX_DRIVER_RESPONSE_BYTES:
            raise WebDriverProtocolError("WebDriver response exceeded its limit")
        try:
            decoded = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise WebDriverProtocolError("WebDriver returned invalid JSON") from error
        if not isinstance(decoded, dict) or "value" not in decoded:
            raise WebDriverProtocolError("WebDriver response schema is invalid")
        value = decoded["value"]
        if isinstance(value, dict) and isinstance(value.get("error"), str):
            code = (
                value.get("error", "webdriver_error")
                if isinstance(value, dict)
                else "webdriver_error"
            )
            raise WebDriverProtocolError(str(code)[:128])
        return value

    def wait_ready(self, timeout: float = 10.0) -> None:
        deadline = monotonic() + timeout
        while monotonic() < deadline:
            try:
                value = self.request("GET", "/status")
                if isinstance(value, dict) and value.get("ready") is True:
                    return
            except WebDriverProtocolError:
                pass
            sleep(0.05)
        raise WebDriverProtocolError("geckodriver did not become ready")

    def create_session(self, firefox_binary: str) -> str:
        options: dict[str, object] = {
            "args": ["-headless"],
            "prefs": {
                "browser.shell.checkDefaultBrowser": False,
                "browser.startup.page": 0,
                "browser.safebrowsing.downloads.enabled": False,
                "browser.safebrowsing.malware.enabled": False,
                "browser.safebrowsing.phishing.enabled": False,
                "datareporting.healthreport.uploadEnabled": False,
                "gfx.webrender.all": False,
                "layers.acceleration.disabled": True,
                "media.navigator.permission.disabled": True,
                "media.navigator.streams.fake": True,
                "network.dns.disablePrefetch": True,
                "network.prefetch-next": False,
                "network.proxy.no_proxies_on": "127.0.0.1,localhost",
                "toolkit.telemetry.enabled": False,
            },
            "log": {"level": "warn"},
        }
        selected = Path(firefox_binary)
        if not selected.is_absolute():
            raise ValueError("Firefox binary must be absolute")
        options["binary"] = str(selected)
        value = self.request(
            "POST",
            "/session",
            {
                "capabilities": {
                    "alwaysMatch": {
                        "browserName": "firefox",
                        "acceptInsecureCerts": False,
                        "moz:firefoxOptions": options,
                    }
                }
            },
        )
        if not isinstance(value, dict) or not isinstance(value.get("sessionId"), str):
            raise WebDriverProtocolError("session id is missing")
        session_id = value["sessionId"]
        capabilities = value.get("capabilities")
        if _WEBDRIVER_ID_RE.fullmatch(session_id) is None:
            raise WebDriverProtocolError("session id is invalid")
        if (
            not isinstance(capabilities, dict)
            or capabilities.get("browserName") != "firefox"
            or not isinstance(capabilities.get("browserVersion"), str)
            or not capabilities["browserVersion"][:128]
            or not isinstance(capabilities.get("moz:processID"), int)
            or isinstance(capabilities.get("moz:processID"), bool)
            or capabilities["moz:processID"] <= 1
        ):
            raise WebDriverProtocolError("Firefox capability identity is invalid")
        self.session_id = session_id
        self.capabilities = dict(capabilities)
        self.browser_process_id = capabilities["moz:processID"]
        return self.session_id

    def _session_path(self, suffix: str) -> str:
        if not self.session_id:
            raise WebDriverProtocolError("WebDriver session is not open")
        return f"/session/{self.session_id}{suffix}"

    def navigate(self, url: str) -> None:
        if not _is_loopback_url(url):
            raise ValueError("browser navigation must remain on 127.0.0.1")
        self.request("POST", self._session_path("/url"), {"url": url})

    def set_window_rect(self, width: int, height: int) -> None:
        if (width, height) not in VIEWPORTS:
            raise ValueError("viewport is outside the approved matrix")
        self.request(
            "POST",
            self._session_path("/window/rect"),
            {"x": 0, "y": 0, "width": width, "height": height},
        )

    def execute(self, script: str, args: list[object] | None = None) -> object:
        if not isinstance(script, str) or not script or len(script) > 64 * 1024:
            raise ValueError("script is invalid or too large")
        return self.request(
            "POST",
            self._session_path("/execute/sync"),
            {"script": script, "args": list(args or [])},
        )

    def find(self, css_selector: str) -> str:
        value = self.request(
            "POST",
            self._session_path("/element"),
            {"using": "css selector", "value": css_selector},
        )
        if not isinstance(value, dict) or not isinstance(
            value.get(W3C_ELEMENT_KEY), str
        ):
            raise WebDriverProtocolError("element id is missing")
        element_id = value[W3C_ELEMENT_KEY]
        if _WEBDRIVER_ID_RE.fullmatch(element_id) is None:
            raise WebDriverProtocolError("element id is invalid")
        return element_id

    def element_enabled(self, element_id: str) -> bool:
        return bool(
            self.request("GET", self._session_path(f"/element/{element_id}/enabled"))
        )

    def click(self, element_id: str) -> None:
        self.request("POST", self._session_path(f"/element/{element_id}/click"), {})

    def send_keys(self, element_id: str, text: str) -> None:
        self.request(
            "POST",
            self._session_path(f"/element/{element_id}/value"),
            {"text": text, "value": list(text)},
        )

    def clear(self, element_id: str) -> None:
        self.request("POST", self._session_path(f"/element/{element_id}/clear"), {})

    def close(self) -> None:
        session = self.session_id
        self.session_id = None
        if session:
            self.request("DELETE", f"/session/{session}")


def _model_item(
    model_id: str, label: str, *, selected: bool, selectable: bool
) -> dict[str, object]:
    return {
        "model_id": model_id,
        "label": label,
        "selected": selected,
        "selectable": selectable,
        "verification": "fixture_verified",
        "checkpoint_modalities": ["text"],
        "runtime_input_modalities": ["text"],
        "unwired_checkpoint_modalities": [],
    }


@dataclass
class FixtureState:
    """Deterministic fake capability/API authority used only by this E2E."""

    enabled: bool
    seq: int = 1
    voice_attested: bool = False
    conversation: list[dict[str, object]] = field(default_factory=list)
    requests: list[dict[str, str]] = field(default_factory=list)
    attachments: list[dict[str, object]] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.enabled and not self.attachments:
            self.attachments.append(
                {
                    "attachment_id": "a" * 24,
                    "name": "fixture.md",
                    "media_type": "text/markdown",
                    "size_bytes": 32,
                    "text_indexable": True,
                    "indexed": True,
                    "preview_kind": "text",
                    "preview_available": True,
                }
            )

    def validation_state(self) -> dict[str, object]:
        return {
            "seq": 1,
            "status": "ready",
            "stage": "ready",
            "events": [],
            "metrics": None,
        }

    def agent_state(self) -> dict[str, object]:
        return {
            "seq": self.seq,
            "status": "succeeded" if self.conversation else "ready",
            "conversation": list(self.conversation),
            "completion": {"generation_mode": "cogni_core"}
            if self.conversation
            else None,
            "core": {"model_loaded": False, "capabilities": {}, "modules": {}},
            "evolution": {
                "status": "ready",
                "running": False,
                "promotion_enabled": False,
            },
        }

    def voice_payload(self) -> dict[str, object]:
        configured = self.enabled
        ready = configured and self.voice_attested
        return {
            "state": "ready"
            if ready
            else "configured_unverified"
            if configured
            else "capture_transport_configured",
            "capture_state": "browser_get_user_media",
            "permission_state": "requested_only_on_user_action",
            "transport_state": "authenticated_loopback_ready",
            "capture_transport": {"state": "configured"},
            "external_calls": 0,
            "processor": {"configured": configured, "probe_passed": ready},
            "transcriber": {"configured": configured, "artifact_verified": configured},
            "model_inference_attested": ready,
            "transcription_state": "ready"
            if ready
            else "configured_unverified"
            if configured
            else "local_artifact_required",
            "runtime_audio_input": ready,
            "stt": {
                "mode": "local_only",
                "artifact_verified": configured,
                "runtime_ready": ready,
                "disabled_reason": None
                if ready
                else "LOCAL_STT_INFERENCE_UNVERIFIED"
                if configured
                else "LOCAL_STT_ARTIFACT_REQUIRED",
            },
            "tts": {
                "state": "disabled",
                "mode": "local_only",
                "source": None,
                "host_probe_passed": False,
                "browser_playback_verified": False,
                "disabled_reason": "LOCAL_TTS_ARTIFACT_REQUIRED",
            },
        }

    def capability_payload(self) -> dict[str, object]:
        models = [
            _model_item(
                "fixture-current",
                "Fixture current",
                selected=True,
                selectable=True,
            )
        ]
        if self.enabled:
            models.append(
                _model_item(
                    "fixture-secondary",
                    "Fixture secondary",
                    selected=False,
                    selectable=False,
                )
            )
        return {
            "schema_version": 1,
            "fixture_schema": FIXTURE_SCHEMA,
            "attachments": {
                "state": "enabled" if self.enabled else "disabled",
                "max_bytes_each": 1024 * 1024,
                "accepted_media_types": ["text/markdown", "text/plain"],
                "image_to_model_integration": False,
                "image_capability": {
                    "runtime_ready": False,
                    "model_inference_attested": False,
                },
            },
            "rag": {
                "state": "local_index_ready" if self.enabled else "unavailable",
                "documents": 1 if self.enabled else 0,
                "answer_integration": self.enabled,
                "answer_integration_schema": "cogni.agent.retrieval-evidence.v1"
                if self.enabled
                else None,
            },
            "models": {
                "state": "verified_local_registry",
                "switching": "idempotent_current_model_only",
                "items": models,
            },
            "microphone": self.voice_payload(),
            "web_search": {
                "mode": "air_gapped",
                "official_lens_connector": {
                    "state": "disabled",
                    "executor_implemented": True,
                    "external_calls": 0,
                },
            },
        }

    def chat(self, body: Mapping[str, object]) -> dict[str, object]:
        message = body.get("message")
        if not isinstance(message, str) or not message.strip():
            raise BrowserE2EError("INVALID_BODY")
        now = _utc_now()
        self.seq += 1
        self.conversation.extend(
            [
                {
                    "id": f"user-{self.seq}",
                    "role": "user",
                    "content": message.strip(),
                    "created_at": now,
                    "streaming": False,
                },
                {
                    "id": f"assistant-{self.seq}",
                    "role": "assistant",
                    "content": ASSISTANT_COMPLETION,
                    "created_at": now,
                    "streaming": False,
                    "finish_reason": "stop",
                    "truncated": False,
                    "generated_tokens": 24,
                    "generation_mode": "cogni_core_rag"
                    if body.get("rag") is True
                    else "cogni_core",
                    "sources": [],
                },
            ]
        )
        return self.agent_state()


class BrowserFixtureServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = False

    def __init__(
        self, assets: str | Path, *, enabled: bool, token: str | None = None
    ) -> None:
        self.assets = Path(assets).resolve(strict=True)
        for name in ("index.html", "app.js", "app.css", "favicon.svg"):
            if not (self.assets / name).is_file():
                raise ValueError(f"production asset is missing: {name}")
        self.state = FixtureState(enabled=enabled)
        self.token = token or secrets.token_urlsafe(32)
        if len(self.token) < 32:
            raise ValueError("fixture token is too short")
        super().__init__(("127.0.0.1", 0), BrowserFixtureHandler)

    @property
    def origin(self) -> str:
        return f"http://127.0.0.1:{self.server_port}"

    @property
    def bootstrap_url(self) -> str:
        return f"{self.origin}/?token={self.token}"

    def instrumented_index(self) -> bytes:
        raw = (self.assets / "index.html").read_bytes()
        marker = b"<head>"
        if raw.count(marker) != 1:
            raise BrowserE2EError("production index cannot be instrumented safely")
        hook = b'<script src="/__e2e__/startup-errors.js"></script>'
        return raw.replace(marker, marker + hook, 1)


class BrowserFixtureHandler(BaseHTTPRequestHandler):
    server: BrowserFixtureServer

    def log_message(self, _format: str, *_args: object) -> None:
        return

    def _headers(self, status: HTTPStatus, content_type: str, length: int) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(length))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Permissions-Policy", "microphone=(self)")
        self.send_header(
            "Content-Security-Policy",
            "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' blob:; connect-src 'self'; media-src 'self' blob:; object-src 'none'; frame-ancestors 'none'",
        )

    def _bytes(self, status: HTTPStatus, content_type: str, content: bytes) -> None:
        self._headers(status, content_type, len(content))
        self.end_headers()
        self.wfile.write(content)

    def _json(self, status: HTTPStatus, payload: object) -> None:
        content = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode(
            "utf-8"
        )
        self._bytes(status, "application/json; charset=utf-8", content)

    def _error(self, status: HTTPStatus, code: str) -> None:
        self._json(status, {"error": {"code": code}})

    def _host_valid(self) -> bool:
        return self.headers.get("Host") == f"127.0.0.1:{self.server.server_port}"

    def _authenticated(self) -> bool:
        cookie = SimpleCookie()
        try:
            cookie.load(self.headers.get("Cookie", ""))
        except Exception:
            return False
        morsel = cookie.get(SESSION_COOKIE)
        return morsel is not None and secrets.compare_digest(
            morsel.value, self.server.token
        )

    def _record(self) -> None:
        parsed = urlsplit(self.path)
        path = parsed.path
        if parsed.query:
            path += (
                "?<redacted>" if "token=" in parsed.query else "?" + parsed.query[:256]
            )
        self.server.state.requests.append(
            {"method": self.command, "origin": self.server.origin, "path": path[:512]}
        )

    def _read_json(self) -> dict[str, object]:
        raw_length = self.headers.get("Content-Length")
        if raw_length is None or not raw_length.isdigit():
            raise BrowserE2EError("INVALID_BODY")
        length = int(raw_length)
        if length < 2 or length > MAX_HTTP_BYTES:
            raise BrowserE2EError("BODY_TOO_LARGE")
        raw = self.rfile.read(length)
        try:
            payload = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise BrowserE2EError("INVALID_BODY") from error
        if not isinstance(payload, dict):
            raise BrowserE2EError("INVALID_BODY")
        return payload

    def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler contract
        self._record()
        if not self._host_valid():
            self._error(HTTPStatus.MISDIRECTED_REQUEST, "LOOPBACK_HOST_REQUIRED")
            return
        parsed = urlsplit(self.path)
        if parsed.path == "/" and not self._authenticated():
            token = parse_qs(parsed.query, strict_parsing=False).get("token", [])
            if len(token) == 1 and secrets.compare_digest(token[0], self.server.token):
                self.send_response(HTTPStatus.SEE_OTHER)
                self.send_header("Location", "/")
                self.send_header("Cache-Control", "no-store")
                self.send_header(
                    "Set-Cookie",
                    f"{SESSION_COOKIE}={self.server.token}; HttpOnly; SameSite=Strict; Path=/",
                )
                self.send_header("Content-Length", "0")
                self.end_headers()
            else:
                self._error(HTTPStatus.UNAUTHORIZED, "AUTH_REQUIRED")
            return
        if not self._authenticated():
            self._error(HTTPStatus.UNAUTHORIZED, "AUTH_REQUIRED")
            return

        static = {
            "/assets/app.js": ("app.js", "text/javascript; charset=utf-8"),
            "/assets/app.css": ("app.css", "text/css; charset=utf-8"),
            "/assets/favicon.svg": ("favicon.svg", "image/svg+xml"),
        }
        if parsed.path == "/" and not parsed.query:
            self._bytes(
                HTTPStatus.OK,
                "text/html; charset=utf-8",
                self.server.instrumented_index(),
            )
        elif parsed.path == "/__e2e__/startup-errors.js" and not parsed.query:
            self._bytes(
                HTTPStatus.OK,
                "text/javascript; charset=utf-8",
                STARTUP_ERROR_SCRIPT,
            )
        elif parsed.path in static and not parsed.query:
            name, content_type = static[parsed.path]
            self._bytes(
                HTTPStatus.OK, content_type, (self.server.assets / name).read_bytes()
            )
        elif parsed.path == "/api/state":
            self._json(HTTPStatus.OK, self.server.state.validation_state())
        elif parsed.path == "/api/agent/state":
            self._json(HTTPStatus.OK, self.server.state.agent_state())
        elif parsed.path == "/api/workspace/capabilities" and not parsed.query:
            self._json(HTTPStatus.OK, self.server.state.capability_payload())
        elif parsed.path == "/api/workspace/attachments" and not parsed.query:
            items = list(self.server.state.attachments)
            self._json(HTTPStatus.OK, {"items": items, "count": len(items)})
        else:
            self._error(HTTPStatus.NOT_FOUND, "NOT_FOUND")

    def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler contract
        self._record()
        if not self._host_valid() or not self._authenticated():
            self._error(HTTPStatus.UNAUTHORIZED, "AUTH_REQUIRED")
            return
        path = urlsplit(self.path).path
        try:
            body = self._read_json()
            if path == "/api/agent/chat":
                self._json(HTTPStatus.ACCEPTED, self.server.state.chat(body))
                return
            if path == "/api/agent/reset":
                self.server.state.conversation.clear()
                self.server.state.seq += 1
                self._json(HTTPStatus.OK, self.server.state.agent_state())
                return
            if path == "/api/workspace/attachments/add":
                if not self.server.state.enabled:
                    raise BrowserE2EError("ATTACHMENTS_DISABLED")
                if set(body) != {"content_base64", "name", "media_type"}:
                    raise BrowserE2EError("INVALID_BODY")
                raw = body.get("content_base64")
                name = body.get("name")
                media_type = body.get("media_type")
                if (
                    not isinstance(raw, str)
                    or not isinstance(name, str)
                    or not isinstance(media_type, str)
                ):
                    raise BrowserE2EError("INVALID_BODY")
                try:
                    content = b64decode(raw, validate=True)
                except (ValueError, binascii.Error) as error:
                    raise BrowserE2EError("INVALID_ATTACHMENT") from error
                attachment_id = sha256(content).hexdigest()[:24]
                item = {
                    "attachment_id": attachment_id,
                    "name": name[:128],
                    "media_type": media_type[:128],
                    "size_bytes": len(content),
                    "text_indexable": media_type.startswith("text/"),
                    "indexed": media_type.startswith("text/"),
                    "preview_kind": "text" if media_type.startswith("text/") else "",
                    "preview_available": media_type.startswith("text/"),
                }
                self.server.state.attachments = [
                    prior
                    for prior in self.server.state.attachments
                    if prior["attachment_id"] != attachment_id
                ] + [item]
                self._json(HTTPStatus.CREATED, item)
                return
            if path in {"/api/workspace/rag/index", "/api/workspace/rag/reindex"}:
                if not self.server.state.enabled or set(body) != {"attachment_ids"}:
                    raise BrowserE2EError("RAG_UNAVAILABLE")
                ids = body.get("attachment_ids")
                known = {
                    str(item["attachment_id"]) for item in self.server.state.attachments
                }
                if (
                    not isinstance(ids, list)
                    or not ids
                    or any(
                        not isinstance(value, str) or value not in known
                        for value in ids
                    )
                ):
                    raise BrowserE2EError("INVALID_BODY")
                self._json(
                    HTTPStatus.OK,
                    {
                        "results": [
                            {"attachment_id": value, "indexed": True} for value in ids
                        ],
                        "documents": len(ids),
                    },
                )
                return
            if path == "/api/workspace/voice/transcribe":
                if not self.server.state.enabled:
                    raise BrowserE2EError("LOCAL_STT_ARTIFACT_REQUIRED")
                if set(body) not in (
                    {"audio_wav_base64"},
                    {"audio_wav_base64", "language"},
                ):
                    raise BrowserE2EError("INVALID_BODY")
                encoded = body.get("audio_wav_base64")
                if not isinstance(encoded, str) or not encoded:
                    raise BrowserE2EError("INVALID_BODY")
                try:
                    wav = b64decode(encoded, validate=True)
                except (ValueError, binascii.Error) as error:
                    raise BrowserE2EError("VOICE_WAV_FORMAT_INVALID") from error
                if len(wav) < 44 or wav[:4] != b"RIFF" or wav[8:12] != b"WAVE":
                    raise BrowserE2EError("VOICE_WAV_FORMAT_INVALID")
                self.server.state.voice_attested = True
                self._json(
                    HTTPStatus.OK,
                    {
                        "transcript": VOICE_TRANSCRIPT,
                        "language": "ko",
                        "external_calls": 0,
                    },
                )
                return
            if path == "/api/workspace/models/select":
                model_id = body.get("model_id")
                if set(body) != {"model_id"}:
                    raise BrowserE2EError("INVALID_BODY")
                if model_id == "fixture-secondary":
                    raise BrowserE2EError("MODEL_SWITCH_UNAVAILABLE")
                if model_id != "fixture-current":
                    raise BrowserE2EError("MODEL_NOT_VERIFIED")
                self._json(
                    HTTPStatus.OK,
                    {
                        **_model_item(
                            "fixture-current",
                            "Fixture current",
                            selected=True,
                            selectable=True,
                        )
                    },
                )
                return
        except BrowserE2EError as error:
            self._error(HTTPStatus.BAD_REQUEST, str(error))
            return
        self._error(HTTPStatus.NOT_FOUND, "NOT_FOUND")


def _asset_digests(assets: Path) -> dict[str, str]:
    return {
        name: sha256((assets / name).read_bytes()).hexdigest()
        for name in ("index.html", "app.js", "app.css")
    }


def _git_commit(project_root: Path) -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=project_root,
        check=True,
        capture_output=True,
        text=True,
        timeout=5,
    )
    commit = completed.stdout.strip()
    if _COMMIT_RE.fullmatch(commit) is None:
        raise BrowserE2EError("source commit is invalid")
    return commit


def _check(name: str, passed: bool, detail: object) -> dict[str, object]:
    return {"name": name, "passed": bool(passed), "detail": detail}


def validate_result_schema(payload: Mapping[str, object]) -> None:
    required = {
        "schema_version",
        "status",
        "executed",
        "source_commit",
        "started_at",
        "finished_at",
        "policy",
        "assets",
        "viewports",
        "profiles",
        "checks",
        "js_errors",
        "network_requests",
        "driver_log",
    }
    if set(payload) != required or payload.get("schema_version") != RESULT_SCHEMA:
        raise BrowserE2EError("result schema is invalid")
    if payload.get("status") not in {"PASS", "FAIL", "NOT_RUN"}:
        raise BrowserE2EError("result status is invalid")
    if not isinstance(payload.get("executed"), bool):
        raise BrowserE2EError("executed flag is invalid")
    if _COMMIT_RE.fullmatch(str(payload.get("source_commit", ""))) is None:
        raise BrowserE2EError("result source commit is invalid")
    checks = payload.get("checks")
    if not isinstance(checks, list) or any(
        not isinstance(item, dict)
        or set(item) != {"name", "passed", "detail"}
        or not isinstance(item["name"], str)
        or not isinstance(item["passed"], bool)
        for item in checks
    ):
        raise BrowserE2EError("result checks are invalid")
    check_names = [str(item["name"]) for item in checks]
    if len(check_names) != len(set(check_names)):
        raise BrowserE2EError("result check names are not unique")
    errors = payload.get("js_errors")
    if not isinstance(errors, list) or any(
        not isinstance(item, str) for item in errors
    ):
        raise BrowserE2EError("JavaScript error list is invalid")
    driver_log = payload.get("driver_log")
    if not isinstance(driver_log, str) or len(driver_log) > MAX_DRIVER_LOG_CHARS:
        raise BrowserE2EError("driver log is invalid")
    assets = payload.get("assets")
    if (
        not isinstance(assets, dict)
        or set(assets) != {"index.html", "app.js", "app.css"}
        or any(_SHA256_RE.fullmatch(str(value)) is None for value in assets.values())
    ):
        raise BrowserE2EError("asset evidence is invalid")
    network_requests = payload.get("network_requests")
    if not isinstance(network_requests, list) or any(
        not isinstance(item, dict)
        or set(item) != {"method", "origin", "path"}
        or item["method"] not in {"GET", "POST"}
        or not isinstance(item["origin"], str)
        or not isinstance(item["path"], str)
        or "token=" in item["path"]
        for item in network_requests
    ):
        raise BrowserE2EError("network request evidence is invalid")

    status = payload.get("status")
    if status == "PASS":
        policy = payload.get("policy")
        viewports = payload.get("viewports")
        profiles = payload.get("profiles")
        expected_viewports = {
            (profile, width, height)
            for profile in ("fail_closed", "enabled")
            for width, height in VIEWPORTS
        }
        observed_viewports: set[tuple[str, int, int]] = set()
        if not isinstance(viewports, list):
            raise BrowserE2EError("PASS viewport evidence is invalid")
        for item in viewports:
            if (
                not isinstance(item, dict)
                or set(item) != {"profile", "requested", "observed", "passed"}
                or item.get("passed") is not True
                or not isinstance(item.get("requested"), list)
                or len(item["requested"]) != 2
                or any(not isinstance(value, int) for value in item["requested"])
            ):
                raise BrowserE2EError("PASS viewport evidence is invalid")
            observed_viewports.add(
                (str(item["profile"]), item["requested"][0], item["requested"][1])
            )
        profile_map = (
            {
                item.get("name"): item.get("checks")
                for item in profiles
                if isinstance(item, dict) and set(item) == {"name", "checks"}
            }
            if isinstance(profiles, list)
            else {}
        )
        if (
            payload.get("executed") is not True
            or not isinstance(policy, dict)
            or policy.get("ready") is not True
            or errors
            or set(check_names) != _PASS_CHECK_NAMES
            or not all(item["passed"] for item in checks)
            or observed_viewports != expected_viewports
            or profile_map != {"fail_closed": 3, "enabled": 6}
            or not network_requests
        ):
            raise BrowserE2EError("PASS result contains unverified evidence")
    elif status == "NOT_RUN" and (
        payload.get("executed") is not False
        or payload.get("viewports") != []
        or payload.get("profiles") != []
        or payload.get("network_requests") != []
    ):
        raise BrowserE2EError("NOT_RUN result contains execution evidence")


def _verify_startup_js_error_collector(client: W3CWebDriverClient) -> None:
    installed = client.execute(
        "return window.__cogniE2EHookInstalled === true && Array.isArray(window.__cogniE2EErrors);"
    )
    if installed is not True:
        raise BrowserE2EError("startup JavaScript error collector was not installed")


def _wait_script(
    client: W3CWebDriverClient,
    script: str,
    *,
    timeout: float = 8.0,
) -> object:
    deadline = monotonic() + timeout
    last: object = None
    while monotonic() < deadline:
        last = client.execute(script)
        if last:
            return last
        sleep(0.05)
    raise BrowserE2EError(f"browser condition timed out: {last!r}")


def _profile_checks(
    client: W3CWebDriverClient,
    server: BrowserFixtureServer,
    *,
    enabled: bool,
    upload_path: Path,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    checks: list[dict[str, object]] = []
    viewports: list[dict[str, object]] = []
    profile = "enabled" if enabled else "fail_closed"
    client.navigate(server.bootstrap_url)
    _wait_script(client, "return document.readyState === 'complete';")
    _verify_startup_js_error_collector(client)
    _wait_script(client, "return document.querySelector('#agent-input') !== null;")
    _wait_script(
        client,
        "return document.querySelector('#agent-model-selector')?.dataset.selectableCount === '1';",
    )
    for width, height in VIEWPORTS:
        client.set_window_rect(width, height)
        geometry = client.execute(
            """
            const composer = document.querySelector('.chat-composer');
            const input = document.querySelector('#agent-input');
            const rect = composer?.getBoundingClientRect();
            return {
              width: window.innerWidth,
              height: window.innerHeight,
              composerVisible: Boolean(rect && rect.width > 0 && rect.height > 0 && rect.top < window.innerHeight && rect.bottom > 0),
              inputVisible: Boolean(input && input.getClientRects().length),
              horizontalOverflow: document.documentElement.scrollWidth > window.innerWidth + 1,
            };
            """
        )
        passed = (
            isinstance(geometry, dict)
            and geometry.get("composerVisible") is True
            and geometry.get("inputVisible") is True
            and geometry.get("horizontalOverflow") is False
        )
        viewports.append(
            {
                "profile": profile,
                "requested": [width, height],
                "observed": geometry,
                "passed": passed,
            }
        )
        checks.append(_check(f"viewport_{profile}_{width}x{height}", passed, geometry))

    states = client.execute(
        """
        const disabled = selector => document.querySelector(selector)?.disabled === true;
        return {
          attachmentDisabled: disabled('[data-action="workspace-attach"]'),
          ragDisabled: disabled('[data-action="workspace-rag-toggle"]'),
          microphoneDisabled: disabled('[data-action="workspace-microphone"]'),
          modelDisabled: disabled('#agent-model-selector'),
          modelSelectableCount: Number(document.querySelector('#agent-model-selector')?.dataset.selectableCount || 0),
        };
        """
    )
    expected = {
        "attachmentDisabled": not enabled,
        "ragDisabled": not enabled,
        "microphoneDisabled": not enabled,
        "modelDisabled": True,
        "modelSelectableCount": 1,
    }
    checks.append(
        _check(
            f"capabilities_{'enabled' if enabled else 'disabled'}",
            states == expected,
            {"observed": states, "expected": expected},
        )
    )

    if enabled:
        attachment_input_id = client.find("#agent-attachment-input")
        client.send_keys(attachment_input_id, str(upload_path))
        attachment_ready = _wait_script(
            client,
            """
            return [...document.querySelectorAll('#agent-attachment-list .attachment-chip')]
              .some(item => item.textContent.includes('browser-e2e.txt'));
            """,
        )
        rag_id = client.find('[data-action="workspace-rag-toggle"]')
        client.click(rag_id)
        rag_ready = _wait_script(
            client,
            """return document.querySelector('[data-action="workspace-rag-toggle"]')
              ?.getAttribute('aria-pressed') === 'true';""",
        )
        voice_posts_before_cancel = sum(
            item.get("method") == "POST"
            and item.get("path") == "/api/workspace/voice/transcribe"
            for item in server.state.requests
        )
        microphone_id = client.find('[data-action="workspace-microphone"]')
        client.click(microphone_id)
        _wait_script(
            client,
            """return document.querySelector('[data-action="workspace-microphone"]')
              ?.getAttribute('aria-pressed') === 'true';""",
        )
        cancel_id = client.find('[data-action="workspace-voice-cancel"]')
        client.click(cancel_id)
        microphone_cancelled = _wait_script(
            client,
            """
            const microphone = document.querySelector('[data-action="workspace-microphone"]');
            const panel = document.querySelector('#agent-voice-capture');
            return microphone?.getAttribute('aria-pressed') === 'false'
              && panel?.dataset.state === 'idle';
            """,
        )
        voice_posts_after_cancel = sum(
            item.get("method") == "POST"
            and item.get("path") == "/api/workspace/voice/transcribe"
            for item in server.state.requests
        )
        client.click(microphone_id)
        _wait_script(
            client,
            """return document.querySelector('[data-action="workspace-microphone"]')
              ?.getAttribute('aria-pressed') === 'true';""",
        )
        sleep(0.25)
        client.click(microphone_id)
        voice_ready = _wait_script(
            client,
            "return document.querySelector('#agent-input')?.value.includes("
            + json.dumps(VOICE_TRANSCRIPT)
            + ");",
        )
        routes = {
            (item.get("method"), item.get("path")) for item in server.state.requests
        }
        expected_routes = {
            ("POST", "/api/workspace/attachments/add"),
            ("POST", "/api/workspace/rag/index"),
            ("POST", "/api/workspace/voice/transcribe"),
        }
        checks.append(
            _check(
                "optional_ui_interactions",
                attachment_ready is True
                and rag_ready is True
                and microphone_cancelled is True
                and voice_posts_after_cancel == voice_posts_before_cancel
                and voice_ready is True
                and expected_routes <= routes,
                {
                    "attachment": attachment_ready,
                    "rag": rag_ready,
                    "microphone_cancelled": microphone_cancelled,
                    "voice_posts_during_cancel": (
                        voice_posts_after_cancel - voice_posts_before_cancel
                    ),
                    "microphone": voice_ready,
                    "api_routes": sorted(f"{method} {path}" for method, path in routes),
                },
            )
        )
        input_id = client.find("#agent-input")
        send_id = client.find('[data-action="agent-send"]')
        prompt = "한 번만 자연스럽게 답해 주세요."
        client.clear(input_id)
        client.send_keys(input_id, prompt)
        client.click(send_id)
        conversation = _wait_script(
            client,
            """
            const rows = [...document.querySelectorAll('#chat-transcript .chat-message')];
            if (rows.length !== 2) return null;
            return rows.map(row => ({role: row.dataset.role, text: row.querySelector('.chat-bubble p')?.textContent || '', streaming: row.classList.contains('is-streaming'), truncated: row.classList.contains('is-truncated')}));
            """,
        )
        expected_conversation = [
            {"role": "user", "text": prompt, "streaming": False, "truncated": False},
            {
                "role": "assistant",
                "text": ASSISTANT_COMPLETION,
                "streaming": False,
                "truncated": False,
            },
        ]
        checks.append(
            _check(
                "single_complete_conversation",
                conversation == expected_conversation,
                conversation,
            )
        )
        checks.append(
            _check(
                "assistant_not_repeated",
                isinstance(conversation, list)
                and sum(
                    str(item.get("text", "")).count(ASSISTANT_COMPLETION)
                    for item in conversation
                    if isinstance(item, dict)
                )
                == 1,
                conversation,
            )
        )

    return checks, viewports


def run_browser_e2e(
    *,
    project_root: Path,
    geckodriver: Path,
    firefox_binary: Path | None,
    boundary: ExecutionBoundary,
) -> dict[str, object]:
    require_execution_boundary(boundary)
    if firefox_binary is None:
        raise BrowserE2EError("an explicit Firefox binary is required")
    driver_identity = inspect_executable(geckodriver, label="geckodriver")
    firefox_identity = inspect_executable(firefox_binary, label="Firefox")
    assets = (project_root / "cogni_demo" / "static").resolve(strict=True)
    source_commit = _git_commit(project_root)
    started_at = _utc_now()
    checks: list[dict[str, object]] = []
    viewports: list[dict[str, object]] = []
    profiles: list[dict[str, object]] = []
    js_errors: list[str] = []
    network_requests: list[dict[str, str]] = []
    driver_log = ""
    final_boundary: ExecutionBoundary | None = None

    with TemporaryDirectory(prefix="cogniboard-browser-e2e-") as temporary:
        temp = Path(temporary)
        probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        probe.bind(("127.0.0.1", 0))
        driver_port = int(probe.getsockname()[1])
        probe.close()
        log_path = temp / "geckodriver.log"
        environment = {
            "HOME": str(temp / "home"),
            "LANG": "C.UTF-8",
            "LIBGL_ALWAYS_SOFTWARE": "1",
            "MOZ_HEADLESS": "1",
            "MOZ_WEBRENDER": "0",
            "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
            "TMPDIR": str(temp),
            "XDG_RUNTIME_DIR": str(temp / "runtime"),
        }
        (temp / "home").mkdir()
        (temp / "runtime").mkdir(mode=0o700)
        upload_path = temp / "browser-e2e.txt"
        upload_path.write_text("bounded browser E2E attachment\n", encoding="utf-8")
        with log_path.open("w", encoding="utf-8") as log:
            process = subprocess.Popen(  # noqa: S603 - absolute validated binary
                [
                    str(geckodriver),
                    "--host",
                    "127.0.0.1",
                    "--port",
                    str(driver_port),
                    "--profile-root",
                    str(temp),
                ],
                cwd=project_root,
                env=environment,
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                shell=False,
                start_new_session=True,
            )
            if inspect_executable(geckodriver, label="geckodriver") != driver_identity:
                cleanup_process_tree(process, browser_pid=None)
                raise BrowserE2EError("geckodriver changed before process start")
            client = W3CWebDriverClient(f"http://127.0.0.1:{driver_port}")
            servers: list[tuple[BrowserFixtureServer, Thread]] = []
            cleanup_errors: list[str] = []
            browser_identity: dict[str, object] | None = None
            try:
                client.wait_ready()
                if (
                    inspect_executable(firefox_binary, label="Firefox")
                    != firefox_identity
                ):
                    raise BrowserE2EError("Firefox changed before session start")
                client.create_session(firefox_identity.path)
                browser_identity = attest_browser_process(client, firefox_identity)
                if (
                    inspect_executable(firefox_binary, label="Firefox")
                    != firefox_identity
                ):
                    raise BrowserE2EError("Firefox changed during session start")
                checks.append(
                    _check(
                        "executable_identity",
                        True,
                        {
                            "geckodriver": driver_identity.to_dict(),
                            "firefox": firefox_identity.to_dict(),
                        },
                    )
                )
                checks.append(
                    _check("browser_capability_identity", True, browser_identity)
                )
                for enabled in (False, True):
                    server = BrowserFixtureServer(assets, enabled=enabled)
                    thread = Thread(target=server.serve_forever, daemon=True)
                    thread.start()
                    servers.append((server, thread))
                    selected_checks, selected_viewports = _profile_checks(
                        client,
                        server,
                        enabled=enabled,
                        upload_path=upload_path,
                    )
                    checks.extend(selected_checks)
                    viewports.extend(selected_viewports)
                    errors = client.execute("return window.__cogniE2EErrors || [];")
                    if isinstance(errors, list):
                        js_errors.extend(str(item)[:512] for item in errors)
                    network_requests.extend(server.state.requests)
                    profiles.append(
                        {
                            "name": "enabled" if enabled else "fail_closed",
                            "checks": len(selected_checks),
                        }
                    )
            finally:
                primary_error = sys.exc_info()[0] is not None
                try:
                    client.close()
                except Exception as error:  # cleanup must continue independently
                    cleanup_errors.append(f"webdriver_close:{type(error).__name__}")
                for server, thread in servers:
                    try:
                        server.shutdown()
                    except Exception as error:
                        cleanup_errors.append(f"server_shutdown:{type(error).__name__}")
                    try:
                        server.server_close()
                    except Exception as error:
                        cleanup_errors.append(f"server_close:{type(error).__name__}")
                    thread.join(timeout=2)
                    if thread.is_alive():
                        cleanup_errors.append("server_thread:alive")
                try:
                    cleanup_process_tree(process, browser_pid=client.browser_process_id)
                except Exception as error:
                    cleanup_errors.append(f"process_tree:{type(error).__name__}")
                if cleanup_errors and not primary_error:
                    raise BrowserE2EError(
                        "browser cleanup failed: " + ",".join(cleanup_errors)
                    )
        try:
            driver_log = log_path.read_text(encoding="utf-8", errors="replace")[
                -MAX_DRIVER_LOG_CHARS:
            ]
        except OSError:
            driver_log = ""

    final_boundary = inspect_execution_boundary()
    checks.append(_check("javascript_errors", not js_errors, js_errors))
    checks.append(
        _check(
            "webdriver_loopback_only",
            bool(client.request_urls)
            and all(_is_loopback_url(url) for url in client.request_urls),
            {"request_count": len(client.request_urls)},
        )
    )
    allowed_fixture_origins = {str(item.get("origin", "")) for item in network_requests}
    checks.append(
        _check(
            "fixture_requests_same_origin",
            bool(network_requests)
            and all(
                _is_loopback_url(str(item.get("origin", "")))
                and str(item.get("path", "")).startswith("/")
                and "token=" not in str(item.get("path", ""))
                for item in network_requests
            ),
            {
                "request_count": len(network_requests),
                "origins": sorted(allowed_fixture_origins),
            },
        )
    )
    checks.append(
        _check(
            "boundary_rechecked",
            final_boundary.ready and final_boundary == boundary,
            {"initial": boundary.to_dict(), "final": final_boundary.to_dict()},
        )
    )
    status = "PASS" if checks and all(item["passed"] for item in checks) else "FAIL"
    result: dict[str, object] = {
        "schema_version": RESULT_SCHEMA,
        "status": status,
        "executed": True,
        "source_commit": source_commit,
        "started_at": started_at,
        "finished_at": _utc_now(),
        "policy": boundary.to_dict(),
        "assets": _asset_digests(assets),
        "viewports": viewports,
        "profiles": profiles,
        "checks": checks,
        "js_errors": js_errors,
        "network_requests": network_requests,
        "driver_log": driver_log,
    }
    validate_result_schema(result)
    return result


def _not_run_result(
    project_root: Path, boundary: ExecutionBoundary
) -> dict[str, object]:
    now = _utc_now()
    assets = (project_root / "cogni_demo" / "static").resolve(strict=True)
    result: dict[str, object] = {
        "schema_version": RESULT_SCHEMA,
        "status": "NOT_RUN",
        "executed": False,
        "source_commit": _git_commit(project_root),
        "started_at": now,
        "finished_at": now,
        "policy": boundary.to_dict(),
        "assets": _asset_digests(assets),
        "viewports": [],
        "profiles": [],
        "checks": [
            _check(
                "explicit_run_required",
                False,
                "pass --run inside an isolated loopback-only, GPU-device-free namespace",
            )
        ],
        "js_errors": [],
        "network_requests": [],
        "driver_log": "",
    }
    validate_result_schema(result)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run",
        action="store_true",
        help="start the explicitly isolated real browser gate",
    )
    parser.add_argument(
        "--project-root", type=Path, default=Path(__file__).resolve().parents[1]
    )
    parser.add_argument(
        "--geckodriver", type=Path, default=Path("/snap/bin/geckodriver")
    )
    parser.add_argument("--firefox-binary", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    project_root = args.project_root.resolve(strict=True)
    boundary = inspect_execution_boundary()
    try:
        result = (
            run_browser_e2e(
                project_root=project_root,
                geckodriver=args.geckodriver,
                firefox_binary=args.firefox_binary,
                boundary=boundary,
            )
            if args.run
            else _not_run_result(project_root, boundary)
        )
    except (BrowserE2EError, OSError, subprocess.SubprocessError) as error:
        print(
            json.dumps(
                {
                    "schema_version": RESULT_SCHEMA,
                    "status": "FAIL",
                    "error": str(error),
                },
                ensure_ascii=False,
            )
        )
        return 1
    encoded = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output is not None:
        target = args.output.resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(encoded, encoding="utf-8")
    print(encoded, end="")
    return 0 if result["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
