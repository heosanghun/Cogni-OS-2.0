from __future__ import annotations

from http.client import HTTPConnection
import json
import os
from pathlib import Path
import sys
import tempfile
from threading import Thread
from time import monotonic, sleep
import unittest
from unittest.mock import patch

from cogni_demo.server import (
    DemoHTTPServer,
    JobAlreadyRunningError,
    JobManager,
    SessionMetadata,
    WorkerLaunch,
    find_live_session,
    open_graphical_app,
    ping_session,
    production_launch_factory,
    read_session_metadata,
    remove_session_metadata,
    write_session_metadata,
    main,
)


ROOT = Path(__file__).resolve().parents[1]
FAKE_WORKER = ROOT / "tests" / "fixtures" / "fake_demo_worker.py"


def launch_factory_for(mode: str):
    def launch(_prompt: str) -> WorkerLaunch:
        environment = os.environ.copy()
        environment["PYTHONUNBUFFERED"] = "1"
        return WorkerLaunch(
            (sys.executable, "-u", str(FAKE_WORKER), mode), ROOT, environment
        )

    return launch


def manager_for(mode: str, *, timeout: float = 10.0) -> JobManager:
    return JobManager(launch_factory_for(mode), max_runtime_seconds=timeout)


def wait_for_terminal(manager: JobManager, timeout: float = 10.0) -> dict:
    deadline = monotonic() + timeout
    while monotonic() < deadline:
        state = manager.snapshot()
        if state["status"] not in {"starting", "running", "cancelling"}:
            return state
        sleep(0.02)
    raise AssertionError("demo job did not become terminal")


class TestDemoJobManager(unittest.TestCase):
    def test_production_command_is_absolute_shell_free_and_offline(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            model = Path(temporary) / "model"
            model.mkdir()
            manifest = Path(temporary) / "manifest.toml"
            manifest.write_text("[files]", encoding="utf-8")
            launch = production_launch_factory(ROOT, model, manifest)("x & calc")
        self.assertEqual(Path(launch.command[0]), Path(sys.executable).resolve())
        self.assertIn("--event-stream", launch.command)
        prompt_index = launch.command.index("--prompt") + 1
        self.assertEqual(launch.command[prompt_index], "x & calc")
        self.assertEqual(launch.environment["TRANSFORMERS_OFFLINE"], "1")
        self.assertEqual(launch.environment["HF_HUB_OFFLINE"], "1")

    def test_initial_snapshot_has_stable_verified_evidence(self) -> None:
        state = manager_for("success").snapshot()
        self.assertEqual(
            set(state),
            {
                "status",
                "stage",
                "seq",
                "progress",
                "events",
                "metrics",
                "error",
                "active_job",
            },
        )
        self.assertEqual(state["status"], "ready")
        self.assertEqual(state["metrics"]["evidence_kind"], "measured_internal")
        self.assertEqual(state["metrics"]["requested_depth"], 100)

    def test_success_requires_ordered_phases_unique_typed_terminal_and_exit_zero(
        self,
    ) -> None:
        manager = manager_for("success")
        manager.start()
        state = wait_for_terminal(manager)
        self.assertEqual(state["status"], "succeeded")
        self.assertEqual(state["progress"], 100)
        self.assertEqual(state["metrics"]["evidence_kind"], "live_runtime_validation")
        self.assertEqual(state["metrics"]["reached_depth"], 100)
        self.assertEqual(state["metrics"]["tests"], 178)
        self.assertEqual(state["metrics"]["target"], "RTX 4090 24GB")
        self.assertIsNone(state["active_job"])

    def test_nonzero_malformed_duplicate_and_unsafe_results_fail(self) -> None:
        for mode in ("fail", "malformed", "duplicate_result", "over_vram"):
            with self.subTest(mode=mode):
                manager = manager_for(mode)
                manager.start()
                state = wait_for_terminal(manager)
                self.assertEqual(state["status"], "failed")
                self.assertIsNotNone(state["error"])

    def test_stderr_flood_is_bounded_and_does_not_deadlock_success(self) -> None:
        for mode in ("stderr_flood", "stdout_flood"):
            with self.subTest(mode=mode):
                manager = manager_for(mode)
                manager.start()
                state = wait_for_terminal(manager)
                self.assertEqual(state["status"], "succeeded")
                self.assertLessEqual(len(manager._diagnostics), 200)
                self.assertTrue(
                    any(item["truncated"] == "true" for item in manager._diagnostics)
                )

    def test_duplicate_run_is_rejected_and_cancel_reaps_worker(self) -> None:
        manager = manager_for("hang")
        manager.start()
        with self.assertRaises(JobAlreadyRunningError):
            manager.start()
        manager.cancel()
        state = wait_for_terminal(manager)
        self.assertEqual(state["status"], "cancelled")
        self.assertIsNone(state["active_job"])
        manager.shutdown()


class TestDemoHTTPControlPlane(unittest.TestCase):
    def setUp(self) -> None:
        self.assets_context = tempfile.TemporaryDirectory()
        assets = Path(self.assets_context.name)
        (assets / "index.html").write_text("<main>Cogni</main>", encoding="utf-8")
        (assets / "app.css").write_text("body{}", encoding="utf-8")
        (assets / "app.js").write_text("void 0", encoding="utf-8")
        self.manager = manager_for("success")
        self.server = DemoHTTPServer(
            self.manager,
            assets,
            port=0,
            token="t" * 32,
            watchdog_timeout=None,
        )
        self.thread = Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.cookie = self._bootstrap()

    def tearDown(self) -> None:
        self.manager.shutdown()
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)
        self.assets_context.cleanup()

    def _connection(self) -> HTTPConnection:
        return HTTPConnection("127.0.0.1", self.server.server_port, timeout=5)

    def _bootstrap(self) -> str:
        connection = self._connection()
        connection.request("GET", "/?token=" + self.server.token)
        response = connection.getresponse()
        response.read()
        self.assertEqual(response.status, 303)
        cookie = response.getheader("Set-Cookie").split(";", 1)[0]
        connection.close()
        return cookie

    def _post(self, path: str, body: dict) -> tuple[int, dict, dict]:
        connection = self._connection()
        encoded = json.dumps(body).encode("utf-8")
        connection.request(
            "POST",
            path,
            body=encoded,
            headers={
                "Cookie": self.cookie,
                "Origin": self.server.origin,
                "Content-Type": "application/json",
            },
        )
        response = connection.getresponse()
        payload = json.loads(response.read())
        headers = dict(response.getheaders())
        status = response.status
        connection.close()
        return status, payload, headers

    def test_loopback_static_state_security_headers_and_exact_routes(self) -> None:
        self.assertEqual(self.server.server_address[0], "127.0.0.1")
        connection = self._connection()
        connection.request("GET", "/api/state", headers={"Cookie": self.cookie})
        response = connection.getresponse()
        state = json.loads(response.read())
        self.assertEqual(response.status, 200)
        self.assertEqual(state["status"], "ready")
        self.assertIn(
            "default-src 'self'", response.getheader("Content-Security-Policy")
        )
        self.assertIsNone(response.getheader("Access-Control-Allow-Origin"))
        connection.close()

        connection = self._connection()
        connection.request("GET", "/api/ping")
        response = connection.getresponse()
        marker = json.loads(response.read())
        self.assertEqual(response.status, 200)
        self.assertEqual(marker, {"service": "cogniboard", "protocol": 1})
        self.assertNotIn("token", marker)
        connection.close()

        connection = self._connection()
        connection.request("GET", "/assets/app.js", headers={"Cookie": self.cookie})
        response = connection.getresponse()
        self.assertEqual(response.read(), b"void 0")
        self.assertEqual(response.status, 200)
        connection.close()

        connection = self._connection()
        connection.request("GET", "/secret", headers={"Cookie": self.cookie})
        response = connection.getresponse()
        response.read()
        self.assertEqual(response.status, 404)
        connection.close()

    def test_run_endpoint_and_origin_cookie_guards(self) -> None:
        status, payload, _headers = self._post("/api/run", {})
        self.assertEqual(status, 202)
        self.assertIn("job_id", payload)
        state = wait_for_terminal(self.manager)
        self.assertEqual(state["status"], "succeeded")

        connection = self._connection()
        body = b"{}"
        connection.request(
            "POST",
            "/api/run",
            body=body,
            headers={"Content-Type": "application/json", "Cookie": self.cookie},
        )
        response = connection.getresponse()
        response.read()
        self.assertEqual(response.status, 403)
        connection.close()

    def test_cancel_and_shutdown_endpoints_control_worker_lifetime(self) -> None:
        self.manager._launch_factory = launch_factory_for("hang")
        status, _payload, _headers = self._post("/api/run", {})
        self.assertEqual(status, 202)
        status, _payload, _headers = self._post("/api/cancel", {})
        self.assertEqual(status, 202)
        self.assertEqual(wait_for_terminal(self.manager)["status"], "cancelled")

        status, payload, _headers = self._post("/api/shutdown", {})
        self.assertEqual(status, 202)
        self.assertEqual(payload["status"], "shutting_down")
        self.thread.join(timeout=2)
        self.assertFalse(self.thread.is_alive())

    def test_invalid_host_and_oversized_body_are_rejected(self) -> None:
        connection = self._connection()
        connection.putrequest("GET", "/api/state", skip_host=True)
        connection.putheader("Host", "evil.example")
        connection.putheader("Cookie", self.cookie)
        connection.endheaders()
        response = connection.getresponse()
        response.read()
        self.assertEqual(response.status, 400)
        connection.close()

        connection = self._connection()
        connection.putrequest("POST", "/api/run")
        connection.putheader("Host", f"127.0.0.1:{self.server.server_port}")
        connection.putheader("Cookie", self.cookie)
        connection.putheader("Origin", self.server.origin)
        connection.putheader("Content-Type", "application/json")
        connection.putheader("Content-Length", "9000")
        connection.endheaders()
        response = connection.getresponse()
        response.read()
        self.assertEqual(response.status, 413)
        connection.close()


class TestDemoApplicationLifecycle(unittest.TestCase):
    @patch("cogni_demo.server.production_launch_factory")
    @patch("cogni_demo.server.find_live_session")
    def test_second_main_reuses_existing_server_without_building_worker(
        self, find_session, launch_factory
    ) -> None:
        find_session.return_value = SessionMetadata(
            os.getpid(), 8765, "e" * 32, "2026-07-11T00:00:00Z"
        )
        with (
            patch("cogni_demo.server.open_graphical_app") as open_app,
            patch("builtins.print") as output,
        ):
            self.assertEqual(main(["--no-browser"]), 0)
            open_app.assert_not_called()
            self.assertEqual(main([]), 0)
            open_app.assert_called_once_with(find_session.return_value.bootstrap_url)
            self.assertNotIn("e" * 32, " ".join(map(str, output.call_args_list)))
        launch_factory.assert_not_called()

    def test_session_metadata_is_bounded_atomic_and_reuses_live_server(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            assets = Path(temporary) / "assets"
            assets.mkdir()
            for name in ("index.html", "app.css", "app.js"):
                (assets / name).write_text(name, encoding="utf-8")
            manager = manager_for("success")
            server = DemoHTTPServer(
                manager, assets, port=0, token="s" * 32, watchdog_timeout=None
            )
            thread = Thread(target=server.serve_forever, daemon=True)
            thread.start()
            session_path = Path(temporary) / "CogniOS" / "cogniboard-session.json"
            metadata = SessionMetadata(
                os.getpid(), server.server_port, server.token, "2026-07-11T00:00:00Z"
            )
            written = write_session_metadata(metadata, session_path)
            self.assertEqual(written, session_path)
            self.assertEqual(read_session_metadata(session_path), metadata)
            self.assertTrue(ping_session(metadata))
            self.assertEqual(find_live_session(session_path), metadata)

            remove_session_metadata(session_path, expected=metadata)
            self.assertFalse(session_path.exists())
            server.request_shutdown()
            thread.join(timeout=2)
            server.server_close()

    def test_malformed_or_symlink_session_is_stale_and_never_followed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "session.json"
            path.write_text("not-json", encoding="utf-8")
            self.assertIsNone(find_live_session(path))
            self.assertFalse(path.exists())

            target = Path(temporary) / "target.json"
            target.write_text("protected", encoding="utf-8")
            try:
                path.symlink_to(target)
            except OSError:
                # Standard non-elevated Windows commonly disables symlink
                # creation; the malformed-file branch remains exercised.
                self.assertEqual(target.read_text(encoding="utf-8"), "protected")
                return
            self.assertIsNone(find_live_session(path))
            self.assertFalse(path.exists())
            self.assertEqual(target.read_text(encoding="utf-8"), "protected")

    @patch("cogni_demo.server.subprocess.Popen")
    @patch("cogni_demo.server._find_edge")
    def test_edge_app_mode_and_browser_fallback(self, find_edge, popen) -> None:
        find_edge.return_value = Path(r"C:\Program Files\Microsoft\Edge\msedge.exe")
        self.assertEqual(open_graphical_app("http://127.0.0.1:8765/x"), "edge")
        command = popen.call_args.args[0]
        self.assertEqual(
            command,
            [
                str(find_edge.return_value),
                "--app=http://127.0.0.1:8765/x",
                "--start-maximized",
                "--no-first-run",
            ],
        )
        self.assertFalse(popen.call_args.kwargs["shell"])

        find_edge.return_value = None
        with patch("cogni_demo.server.webbrowser.open") as browser:
            self.assertEqual(open_graphical_app("http://127.0.0.1:8765/y"), "browser")
            browser.assert_called_once_with("http://127.0.0.1:8765/y", new=1)

    def test_watchdog_cancels_worker_and_stops_when_polling_disappears(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            assets = Path(temporary)
            for name in ("index.html", "app.css", "app.js"):
                (assets / name).write_text(name, encoding="utf-8")
            manager = manager_for("hang")
            manager.start()
            server = DemoHTTPServer(
                manager,
                assets,
                port=0,
                token="w" * 32,
                watchdog_timeout=0.15,
            )
            thread = Thread(target=server.serve_forever, daemon=True)
            thread.start()
            thread.join(timeout=3)
            self.assertFalse(thread.is_alive())
            self.assertEqual(wait_for_terminal(manager)["status"], "cancelled")
            server.server_close()

    def test_authenticated_state_poll_keeps_watchdog_alive(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            assets = Path(temporary)
            for name in ("index.html", "app.css", "app.js"):
                (assets / name).write_text(name, encoding="utf-8")
            manager = manager_for("success")
            server = DemoHTTPServer(
                manager,
                assets,
                port=0,
                token="p" * 32,
                watchdog_timeout=0.2,
            )
            thread = Thread(target=server.serve_forever, daemon=True)
            thread.start()
            connection = HTTPConnection("127.0.0.1", server.server_port, timeout=2)
            connection.request("GET", "/?token=" + server.token)
            response = connection.getresponse()
            response.read()
            cookie = response.getheader("Set-Cookie").split(";", 1)[0]
            connection.close()
            deadline = monotonic() + 0.5
            while monotonic() < deadline:
                connection = HTTPConnection("127.0.0.1", server.server_port, timeout=2)
                connection.request("GET", "/api/state", headers={"Cookie": cookie})
                response = connection.getresponse()
                response.read()
                self.assertEqual(response.status, 200)
                connection.close()
                sleep(0.05)
            self.assertTrue(thread.is_alive())
            server.request_shutdown()
            thread.join(timeout=2)
            server.server_close()


if __name__ == "__main__":
    unittest.main()
