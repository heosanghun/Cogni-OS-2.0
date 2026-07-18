from __future__ import annotations

from base64 import b64encode
from contextlib import contextmanager, redirect_stdout
from http.client import HTTPConnection
from io import StringIO
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Thread
import unittest
from unittest.mock import patch

from scripts.validate_cogniboard_browser_e2e import (
    ASSISTANT_COMPLETION,
    BrowserE2EError,
    BrowserFixtureServer,
    ExecutionBoundary,
    RESULT_SCHEMA,
    SESSION_COOKIE,
    VOICE_TRANSCRIPT,
    W3CWebDriverClient,
    WebDriverProtocolError,
    inspect_execution_boundary,
    main,
    require_execution_boundary,
    validate_result_schema,
)


ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "cogni_demo" / "static"
TOKEN = "fixture-token-" + ("a" * 32)


class _Response:
    def __init__(self, payload: object, *, status: int = 200) -> None:
        self.status = status
        self._raw = json.dumps(payload).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def read(self, limit: int = -1) -> bytes:
        return self._raw if limit < 0 else self._raw[:limit]


class _Transport:
    def __init__(self, *responses: _Response) -> None:
        self.responses = list(responses)
        self.requests: list[tuple[str, str, object | None, float]] = []

    def __call__(self, request, timeout: float) -> _Response:
        raw = request.data
        payload = json.loads(raw.decode("utf-8")) if raw is not None else None
        self.requests.append((request.method, request.full_url, payload, timeout))
        return self.responses.pop(0)


@contextmanager
def _fixture(*, enabled: bool):
    server = BrowserFixtureServer(ASSETS, enabled=enabled, token=TOKEN)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def _request(
    server: BrowserFixtureServer,
    method: str,
    path: str,
    *,
    cookie: str | None = None,
    payload: object | None = None,
    host: str | None = None,
) -> tuple[int, dict[str, str], bytes]:
    connection = HTTPConnection("127.0.0.1", server.server_port, timeout=3)
    headers: dict[str, str] = {}
    if cookie:
        headers["Cookie"] = cookie
    if host:
        headers["Host"] = host
    body = None
    if payload is not None:
        body = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    connection.request(method, path, body=body, headers=headers)
    response = connection.getresponse()
    raw = response.read()
    selected_headers = {key.lower(): value for key, value in response.getheaders()}
    connection.close()
    return response.status, selected_headers, raw


def _bootstrap(server: BrowserFixtureServer) -> str:
    status, headers, raw = _request(server, "GET", f"/?token={TOKEN}")
    if status != 303 or raw:
        raise AssertionError((status, raw))
    cookie = headers["set-cookie"].split(";", 1)[0]
    if not cookie.startswith(f"{SESSION_COOKIE}="):
        raise AssertionError(cookie)
    return cookie


def _json(raw: bytes) -> dict[str, object]:
    value = json.loads(raw.decode("utf-8"))
    if not isinstance(value, dict):
        raise AssertionError(value)
    return value


class TestExecutionBoundary(unittest.TestCase):
    def test_accepts_only_loopback_with_no_visible_gpu_devices(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            dev = root / "dev"
            network = root / "net"
            dev.mkdir()
            (network / "lo").mkdir(parents=True)
            boundary = inspect_execution_boundary(
                dev_root=dev,
                network_root=network,
                environment={
                    "NVIDIA_VISIBLE_DEVICES": "void",
                    "CUDA_VISIBLE_DEVICES": "-1",
                },
            )
        self.assertTrue(boundary.ready)
        self.assertEqual(boundary.network_interfaces, ("lo",))
        require_execution_boundary(boundary)

    def test_rejects_gpu_device_network_and_environment_visibility(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            dev = root / "dev"
            network = root / "net"
            (dev / "dri").mkdir(parents=True)
            (dev / "nvidia0").write_text("", encoding="utf-8")
            (network / "lo").mkdir(parents=True)
            (network / "eth0").mkdir()
            boundary = inspect_execution_boundary(
                dev_root=dev,
                network_root=network,
                environment={
                    "NVIDIA_VISIBLE_DEVICES": "all",
                    "CUDA_VISIBLE_DEVICES": "0",
                },
            )
        self.assertFalse(boundary.ready)
        self.assertEqual(len(boundary.gpu_device_paths), 2)
        self.assertEqual(boundary.network_interfaces, ("eth0", "lo"))
        self.assertEqual(
            boundary.environment_violations,
            ("CUDA_VISIBLE_DEVICES", "NVIDIA_VISIBLE_DEVICES"),
        )
        with self.assertRaisesRegex(BrowserE2EError, "GPU_DEVICE_VISIBLE"):
            require_execution_boundary(boundary)


class TestW3CWebDriverClient(unittest.TestCase):
    def test_posts_bounded_w3c_session_and_uses_session_id(self) -> None:
        transport = _Transport(
            _Response({"value": {"sessionId": "session-1", "capabilities": {}}}),
            _Response({"value": None}),
            _Response({"value": None}),
        )
        client = W3CWebDriverClient(
            "http://127.0.0.1:4444", timeout=3, transport=transport
        )
        self.assertEqual(client.create_session(), "session-1")
        client.navigate("http://127.0.0.1:8080/?token=fixture")
        client.close()
        first = transport.requests[0]
        self.assertEqual(first[:2], ("POST", "http://127.0.0.1:4444/session"))
        self.assertEqual(
            first[2]["capabilities"]["alwaysMatch"]["browserName"], "firefox"
        )
        self.assertIn(
            "-headless",
            first[2]["capabilities"]["alwaysMatch"]["moz:firefoxOptions"]["args"],
        )
        self.assertEqual(
            transport.requests[1][1],
            "http://127.0.0.1:4444/session/session-1/url",
        )
        self.assertEqual(
            transport.requests[2][1], "http://127.0.0.1:4444/session/session-1"
        )

    def test_rejects_non_loopback_driver_and_navigation(self) -> None:
        with self.assertRaisesRegex(ValueError, "127.0.0.1"):
            W3CWebDriverClient("http://example.com:4444")
        client = W3CWebDriverClient("http://127.0.0.1:4444", transport=_Transport())
        client.session_id = "session-1"
        with self.assertRaisesRegex(ValueError, "127.0.0.1"):
            client.navigate("http://localhost:8080/")
        with self.assertRaisesRegex(ValueError, "origin-relative"):
            client.request("GET", "//outside.example/status")

    def test_rejects_protocol_errors_and_oversized_responses(self) -> None:
        error_client = W3CWebDriverClient(
            "http://127.0.0.1:4444",
            transport=_Transport(
                _Response({"value": {"error": "invalid session id"}}, status=404)
            ),
        )
        with self.assertRaisesRegex(WebDriverProtocolError, "invalid session id"):
            error_client.request("GET", "/status")

        oversized = _Response({"value": None})
        oversized._raw = b"{" + (b"x" * (4 * 1024 * 1024 + 1))
        size_client = W3CWebDriverClient(
            "http://127.0.0.1:4444", transport=_Transport(oversized)
        )
        with self.assertRaisesRegex(WebDriverProtocolError, "exceeded"):
            size_client.request("GET", "/status")


class TestBrowserFixtureServer(unittest.TestCase):
    def test_authenticates_and_serves_exact_production_static_assets(self) -> None:
        with _fixture(enabled=False) as server:
            self.assertEqual(server.server_address[0], "127.0.0.1")
            status, _headers, _raw = _request(server, "GET", "/")
            self.assertEqual(status, 401)
            cookie = _bootstrap(server)
            status, headers, raw = _request(server, "GET", "/app.js", cookie=cookie)
            self.assertEqual(status, 200)
            self.assertEqual(raw, (ASSETS / "app.js").read_bytes())
            self.assertIn("connect-src 'self'", headers["content-security-policy"])
            self.assertEqual(headers["cache-control"], "no-store")
            status, _headers, _raw = _request(
                server,
                "GET",
                "/api/state",
                cookie=cookie,
                host="outside.example",
            )
            self.assertEqual(status, 421)

    def test_fail_closed_capabilities_disable_optional_controls(self) -> None:
        with _fixture(enabled=False) as server:
            cookie = _bootstrap(server)
            status, _headers, raw = _request(
                server, "GET", "/api/workspace/capabilities", cookie=cookie
            )
            self.assertEqual(status, 200)
            capabilities = _json(raw)
            self.assertEqual(capabilities["attachments"]["state"], "disabled")
            self.assertEqual(capabilities["rag"]["state"], "unavailable")
            self.assertFalse(capabilities["rag"]["answer_integration"])
            self.assertEqual(len(capabilities["models"]["items"]), 1)
            self.assertFalse(capabilities["microphone"]["stt"]["runtime_ready"])

    def test_enabled_profile_exercises_attachment_rag_stt_and_model_apis(self) -> None:
        with _fixture(enabled=True) as server:
            cookie = _bootstrap(server)
            status, _headers, raw = _request(
                server, "GET", "/api/workspace/capabilities", cookie=cookie
            )
            before = _json(raw)
            self.assertEqual(before["attachments"]["state"], "enabled")
            self.assertEqual(before["rag"]["state"], "local_index_ready")
            self.assertTrue(before["rag"]["answer_integration"])
            self.assertEqual(len(before["models"]["items"]), 2)
            self.assertEqual(before["microphone"]["state"], "configured_unverified")

            status, _headers, raw = _request(
                server,
                "POST",
                "/api/workspace/attachments/add",
                cookie=cookie,
                payload={
                    "name": "new.txt",
                    "media_type": "text/plain",
                    "content_base64": b64encode(b"bounded fixture").decode("ascii"),
                },
            )
            self.assertEqual(status, 201)
            attachment = _json(raw)
            self.assertTrue(attachment["indexed"])

            status, _headers, raw = _request(
                server,
                "POST",
                "/api/workspace/rag/index",
                cookie=cookie,
                payload={"attachment_ids": [attachment["attachment_id"]]},
            )
            self.assertEqual(status, 200)
            self.assertEqual(_json(raw)["documents"], 1)

            status, _headers, raw = _request(
                server,
                "POST",
                "/api/workspace/voice/transcribe",
                cookie=cookie,
                payload={"audio_base64": "fixture"},
            )
            self.assertEqual(status, 200)
            transcript = _json(raw)
            self.assertEqual(transcript["transcript"], VOICE_TRANSCRIPT)
            self.assertEqual(transcript["external_calls"], 0)

            status, _headers, raw = _request(
                server, "GET", "/api/workspace/capabilities", cookie=cookie
            )
            after = _json(raw)
            self.assertEqual(after["microphone"]["state"], "ready")
            self.assertTrue(after["microphone"]["stt"]["runtime_ready"])

            status, _headers, raw = _request(
                server,
                "POST",
                "/api/workspace/models/select",
                cookie=cookie,
                payload={"model_id": "fixture-secondary"},
            )
            self.assertEqual(status, 200)
            self.assertEqual(_json(raw)["model_id"], "fixture-secondary")

    def test_chat_is_single_complete_non_repeating_and_stable_across_polls(
        self,
    ) -> None:
        with _fixture(enabled=True) as server:
            cookie = _bootstrap(server)
            prompt = "한 번만 자연스럽게 답해 주세요."
            status, _headers, raw = _request(
                server,
                "POST",
                "/api/agent/chat",
                cookie=cookie,
                payload={"message": prompt, "rag": True},
            )
            self.assertEqual(status, 202)
            first = _json(raw)
            status, _headers, raw = _request(
                server, "GET", "/api/agent/state?after=0", cookie=cookie
            )
            self.assertEqual(status, 200)
            second = _json(raw)
            self.assertEqual(first["conversation"], second["conversation"])
            self.assertEqual(len(second["conversation"]), 2)
            assistant = second["conversation"][1]
            self.assertEqual(assistant["content"], ASSISTANT_COMPLETION)
            self.assertEqual(assistant["finish_reason"], "stop")
            self.assertFalse(assistant["truncated"])
            self.assertEqual(
                sum(
                    item["content"].count(ASSISTANT_COMPLETION)
                    for item in second["conversation"]
                ),
                1,
            )


class TestResultSchemaAndEntryPoint(unittest.TestCase):
    @staticmethod
    def _valid_result() -> dict[str, object]:
        return {
            "schema_version": RESULT_SCHEMA,
            "status": "PASS",
            "executed": True,
            "source_commit": "a" * 40,
            "started_at": "2026-07-19T00:00:00Z",
            "finished_at": "2026-07-19T00:00:01Z",
            "policy": {"ready": True},
            "assets": {"app.js": "b" * 64},
            "viewports": [],
            "profiles": [],
            "checks": [{"name": "gate", "passed": True, "detail": {}}],
            "js_errors": [],
            "network_requests": [],
            "driver_log": "",
        }

    def test_result_schema_accepts_complete_pass_and_rejects_false_evidence(
        self,
    ) -> None:
        result = self._valid_result()
        validate_result_schema(result)
        result["checks"][0]["passed"] = False
        with self.assertRaisesRegex(BrowserE2EError, "unverified"):
            validate_result_schema(result)
        result = self._valid_result()
        result["js_errors"] = ["boom"]
        with self.assertRaisesRegex(BrowserE2EError, "unverified"):
            validate_result_schema(result)

    def test_default_entry_point_never_starts_geckodriver(self) -> None:
        output = StringIO()
        with (
            patch(
                "scripts.validate_cogniboard_browser_e2e._git_commit",
                return_value="a" * 40,
            ),
            patch("scripts.validate_cogniboard_browser_e2e.subprocess.Popen") as popen,
            redirect_stdout(output),
        ):
            exit_code = main(["--project-root", str(ROOT)])
        self.assertEqual(exit_code, 2)
        popen.assert_not_called()
        payload = json.loads(output.getvalue())
        self.assertEqual(payload["status"], "NOT_RUN")
        self.assertFalse(payload["executed"])

    def test_unsafe_explicit_run_fails_before_geckodriver_start(self) -> None:
        unsafe = ExecutionBoundary(("/dev/nvidia0",), ("lo",), ())
        output = StringIO()
        with (
            patch(
                "scripts.validate_cogniboard_browser_e2e.inspect_execution_boundary",
                return_value=unsafe,
            ),
            patch("scripts.validate_cogniboard_browser_e2e.subprocess.Popen") as popen,
            redirect_stdout(output),
        ):
            exit_code = main(
                [
                    "--run",
                    "--project-root",
                    str(ROOT),
                    "--geckodriver",
                    str(ROOT / "not-used-geckodriver"),
                ]
            )
        self.assertEqual(exit_code, 1)
        popen.assert_not_called()
        payload = json.loads(output.getvalue())
        self.assertEqual(payload["status"], "FAIL")
        self.assertIn("GPU_DEVICE_VISIBLE", payload["error"])


if __name__ == "__main__":
    unittest.main()
