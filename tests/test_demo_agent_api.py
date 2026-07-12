from __future__ import annotations

from http.client import HTTPConnection
import json
import tempfile
from pathlib import Path
from threading import Thread
from time import monotonic, sleep
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from cogni_agent.manager import AgentBusyError, NoActiveAgentTurnError
from cogni_demo.server import (
    ComputeBusyError,
    DemoHTTPServer,
    EvolutionController,
    JobAlreadyRunningError,
    _ServiceBackedPatchModel,
    _build_product_controls,
)
from cogni_flow.cycle import EvolutionReport
from cogni_flow.harness import FailureTrace
from cogni_flow.production import BoundedLogDB, PromotionMode
from cogni_flow.rhythm import RhythmController
from cogni_flow.scheduler import ScheduleDecision, ScheduleTick
from cogni_os.gpu_lease import GPULeaseManager
from tests.test_demo_server import manager_for, wait_for_terminal


class _FakeAgentManager:
    def __init__(self) -> None:
        self.active = False
        self.sequence = 0
        self.messages: list[dict[str, object]] = []
        self.model_stops = 0
        self.shutdown_called = False
        self.waited_after: int | None = None
        self.availability_check = None

    @property
    def is_active(self) -> bool:
        return self.active

    def snapshot(self) -> dict[str, object]:
        return {
            "status": "generating" if self.active else "ready",
            "stage": "decode" if self.active else "ready",
            "seq": self.sequence,
            "progress": 50 if self.active else 100,
            "events": [],
            "conversation": list(self.messages),
            "active_turn": "turn-1" if self.active else None,
            "error": None,
            "core": {"active_modules": []},
            "evolution": {"running": False},
        }

    def wait_snapshot(self, after: int) -> dict[str, object]:
        self.waited_after = after
        return self.snapshot()

    def start_turn(self, message: str, mode: str = "chat") -> str:
        if self.active:
            raise AgentBusyError("active")
        if self.availability_check is not None and not self.availability_check():
            raise AgentBusyError("compute busy")
        if not isinstance(message, str) or not message:
            raise ValueError("message")
        if mode not in {"chat", "task"}:
            raise ValueError("mode")
        self.active = True
        self.sequence += 1
        self.messages.append({"role": "user", "content": message})
        return "turn-1"

    def cancel(self) -> None:
        if not self.active:
            raise NoActiveAgentTurnError("inactive")
        self.active = False
        self.sequence += 1

    def reset(self) -> None:
        if self.active:
            raise AgentBusyError("active")
        self.messages.clear()
        self.sequence += 1

    def stop_model(self) -> None:
        if self.active:
            raise AgentBusyError("active")
        self.model_stops += 1

    def shutdown(self) -> None:
        self.active = False
        self.shutdown_called = True


class _FakeEvolutionManager:
    def __init__(self) -> None:
        self.active = False
        self.sequence = 0
        self.shutdown_called = False
        self.availability_check = None

    @property
    def is_active(self) -> bool:
        return self.active

    def snapshot(self) -> dict[str, object]:
        return {
            "running": self.active,
            "status": "running" if self.active else "ready",
            "seq": self.sequence,
            "job_id": "evolution-1" if self.active else None,
        }

    def start(self) -> str:
        if self.active:
            raise RuntimeError("active")
        if self.availability_check is not None and not self.availability_check():
            raise RuntimeError("compute busy")
        self.active = True
        self.sequence += 1
        return "evolution-1"

    def shutdown(self) -> None:
        self.active = False
        self.shutdown_called = True


class _PatchTokenizer:
    def decode(self, tokens, **_kwargs) -> str:
        return "bounded repair prompt" if list(tokens) == [1, 2] else "replacement"


class _PatchService:
    def __init__(self) -> None:
        self.calls: list[tuple[str, int]] = []

    def generate(self, prompt: str, *, max_new_tokens: int):
        self.calls.append((prompt, max_new_tokens))
        return SimpleNamespace(token_ids=torch.tensor([3, 4], dtype=torch.int64))


class _ProductTokenizer:
    eos_token_id = 3

    def __call__(self, text, **_kwargs):
        return {"input_ids": torch.tensor([[len(text)]], dtype=torch.int64)}

    def decode(self, _tokens, **_kwargs):
        return "local"

    def apply_chat_template(self, messages, **_kwargs):
        return "|".join(item["content"] for item in messages)


class _ProductService:
    def __init__(self) -> None:
        self.tokenizer = _ProductTokenizer()
        self.active_request_id = None
        self.stop_calls = 0

    def start(self):
        return self

    def iter_generate_tokens(self, _prompt, *, max_new_tokens):
        del max_new_tokens
        if False:
            yield None

    def generate(self, _prompt, *, max_new_tokens):
        del max_new_tokens
        return SimpleNamespace(token_ids=torch.tensor([1], dtype=torch.int64))

    def cancel(self, _request_id=None):
        return False

    def stop(self, timeout=10.0):
        del timeout
        self.stop_calls += 1


class _ProductHarness:
    def __init__(self) -> None:
        self.started = False
        self.stopped = False
        self.status = SimpleNamespace(
            promotion_mode=PromotionMode.PROPOSAL_ONLY,
            promotion_enabled=False,
            blocked_reason="proposal only",
            pending_proposals=0,
            running=True,
        )

    def start(self):
        self.started = True
        return self

    def stop(self):
        self.stopped = True

    def tick(self):
        return ScheduleTick(ScheduleDecision.NOT_IDLE, 0.0)

    def capture_exception(self, *_args, **_kwargs):
        return None


class _HarnessFixture:
    def __init__(self, logdb: BoundedLogDB) -> None:
        self.logdb = logdb
        self.status = SimpleNamespace(
            promotion_mode=PromotionMode.PROPOSAL_ONLY,
            promotion_enabled=False,
            blocked_reason="proposal only",
            pending_proposals=0,
            unreviewable_proposals=2,
            proposal_integrity_errors=(("a" * 64, "replacement blob is missing"),),
            running=True,
        )
        self.stopped = False

    def tick(self) -> ScheduleTick:
        return ScheduleTick(
            ScheduleDecision.RAN,
            0.0,
            EvolutionReport(2, 1, False, None),
        )

    def stop(self) -> None:
        self.stopped = True


class TestAgentHTTPControlPlane(unittest.TestCase):
    def setUp(self) -> None:
        self.assets_context = tempfile.TemporaryDirectory()
        assets = Path(self.assets_context.name)
        (assets / "index.html").write_text("<main>Cogni</main>", encoding="utf-8")
        (assets / "app.css").write_text("body{}", encoding="utf-8")
        (assets / "app.js").write_text("void 0", encoding="utf-8")
        (assets / "favicon.svg").write_text("<svg/>", encoding="utf-8")
        self.validator = manager_for("success")
        self.agent = _FakeAgentManager()
        self.evolution = _FakeEvolutionManager()
        self.server = DemoHTTPServer(
            self.validator,
            assets,
            agent_manager=self.agent,
            evolution_manager=self.evolution,
            port=0,
            token="a" * 32,
            watchdog_timeout=None,
        )
        self.thread = Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.cookie = self._bootstrap()

    def tearDown(self) -> None:
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

    def _get(self, path: str) -> tuple[int, dict[str, object]]:
        connection = self._connection()
        connection.request("GET", path, headers={"Cookie": self.cookie})
        response = connection.getresponse()
        payload = json.loads(response.read())
        status = response.status
        connection.close()
        return status, payload

    def _post(
        self, path: str, body: dict[str, object]
    ) -> tuple[int, dict[str, object]]:
        connection = self._connection()
        connection.request(
            "POST",
            path,
            body=json.dumps(body).encode("utf-8"),
            headers={
                "Cookie": self.cookie,
                "Origin": self.server.origin,
                "Content-Type": "application/json",
            },
        )
        response = connection.getresponse()
        payload = json.loads(response.read())
        status = response.status
        connection.close()
        return status, payload

    def test_agent_state_chat_cancel_reset_and_long_poll(self) -> None:
        status, state = self._get("/api/agent/state")
        self.assertEqual(status, 200)
        self.assertEqual(state["status"], "ready")

        status, state = self._get("/api/agent/state?after=7")
        self.assertEqual(status, 200)
        self.assertEqual(self.agent.waited_after, 7)

        status, state = self._post(
            "/api/agent/chat", {"message": "로컬 상태를 설명해줘", "mode": "chat"}
        )
        self.assertEqual(status, 202)
        self.assertEqual(state["turn_id"], "turn-1")
        self.assertTrue(self.agent.is_active)

        status, payload = self._post("/api/run", {})
        self.assertEqual(status, 409)
        self.assertEqual(payload["error"]["code"], "JOB_ALREADY_RUNNING")
        with self.assertRaises(JobAlreadyRunningError):
            self.validator.start()

        status, _state = self._post("/api/agent/cancel", {})
        self.assertEqual(status, 202)
        self.assertFalse(self.agent.is_active)

        status, state = self._post("/api/agent/reset", {})
        self.assertEqual(status, 200)
        self.assertEqual(state["conversation"], [])

    def test_validator_agent_and_evolution_have_one_compute_owner(self) -> None:
        status, _payload = self._post("/api/run", {})
        self.assertEqual(status, 202)
        self.assertEqual(self.agent.model_stops, 1)
        with self.assertRaises(AgentBusyError):
            self.agent.start_turn("direct start must also be blocked")
        with self.assertRaisesRegex(RuntimeError, "compute busy"):
            self.evolution.start()

        status, payload = self._post(
            "/api/agent/chat", {"message": "동시에 실행하면 안 됨"}
        )
        self.assertEqual(status, 409)
        self.assertEqual(payload["error"]["code"], "COMPUTE_BUSY")
        status, payload = self._post("/api/evolution/run", {})
        self.assertEqual(status, 409)
        self.assertEqual(payload["error"]["code"], "COMPUTE_BUSY")
        self.assertEqual(wait_for_terminal(self.validator)["status"], "succeeded")

        status, state = self._post("/api/evolution/run", {})
        self.assertEqual(status, 202)
        self.assertEqual(state["job_id"], "evolution-1")
        self.assertTrue(self.evolution.is_active)
        self.assertEqual(self.agent.model_stops, 2)

        status, _payload = self._post(
            "/api/agent/chat", {"message": "진화 중에는 차단"}
        )
        self.assertEqual(status, 409)
        status, _payload = self._post("/api/run", {})
        self.assertEqual(status, 409)

    def test_invalid_agent_payload_and_component_cleanup_fail_closed(self) -> None:
        status, payload = self._post("/api/agent/chat", {"mode": "chat"})
        self.assertEqual(status, 400)
        self.assertEqual(payload["error"]["code"], "INVALID_BODY")
        status, _payload = self._post(
            "/api/agent/chat", {"message": "x", "extra": True}
        )
        self.assertEqual(status, 400)

        self.server.shutdown_components()
        self.assertTrue(self.agent.shutdown_called)
        self.assertTrue(self.evolution.shutdown_called)
        with self.assertRaisesRegex(ComputeBusyError, "shutting down"):
            self.server.start_agent_turn("closed", "chat")


class TestEvolutionAndPatchServiceIntegration(unittest.TestCase):
    def test_evolution_cleanup_failure_is_terminal_failed_not_complete(self) -> None:
        harness = _ProductHarness()

        def fail_cleanup() -> None:
            raise RuntimeError("worker still owns GPU")

        controller = EvolutionController(harness, worker_cleanup=fail_cleanup)
        controller.start()
        deadline = monotonic() + 2.0
        while controller.is_active and monotonic() < deadline:
            sleep(0.01)

        state = controller.snapshot()
        self.assertEqual(state["status"], "failed")
        self.assertEqual(state["error"]["code"], "RuntimeError")
        self.assertIn("worker still owns GPU", state["error"]["message"])
        controller.worker_cleanup = None
        controller.shutdown()

    def test_product_controls_share_one_gpu_authority_and_rhythm(self) -> None:
        from tests.test_agent_quality_integration import _factbook

        validator = manager_for("success")
        leases = GPULeaseManager()
        rhythm = RhythmController()
        service = _ProductService()
        harness = _ProductHarness()
        facts = _factbook()

        with (
            patch(
                "cogni_demo.server.verify_artifact_manifest",
                return_value=object(),
            ),
            patch(
                "cogni_os.factbook.build_runtime_factbook_from_verified",
                return_value=facts,
            ),
            patch(
                "cogni_agent.model_service.ModelService.for_local_gemma",
                return_value=service,
            ) as service_factory,
            patch(
                "cogni_flow.production.build_production_self_harness",
                return_value=harness,
            ) as harness_factory,
        ):
            agent, evolution = _build_product_controls(
                Path.cwd(),
                "local-model",
                "manifest.toml",
                validator,
                gpu_lease_manager=leases,
                rhythm=rhythm,
            )

        service_options = service_factory.call_args.kwargs
        self.assertIs(service_options["gpu_lease_manager"], leases)
        self.assertEqual(service_options["gpu_lease_vram_bytes"], leases.max_vram_bytes)
        purpose = service_options["gpu_lease_purpose"]
        self.assertEqual(purpose(), "inference")
        rhythm.enter_evolution(lambda: None)
        self.assertEqual(purpose(), "evolution")
        rhythm.resume_inference("test complete")
        self.assertIs(agent.rhythm, rhythm)
        self.assertIs(harness_factory.call_args.kwargs["rhythm"], rhythm)
        self.assertIs(evolution.worker_cleanup.__self__, service)
        self.assertTrue(harness.started)

        evolution.shutdown()
        agent.shutdown()
        self.assertGreaterEqual(service.stop_calls, 2)

    def test_product_controls_fail_before_model_creation_on_manifest_error(
        self,
    ) -> None:
        validator = manager_for("success")
        with patch(
            "cogni_demo.server.verify_artifact_manifest",
            side_effect=RuntimeError("digest mismatch"),
        ) as verify:
            with self.assertRaisesRegex(RuntimeError, "digest mismatch"):
                _build_product_controls(
                    Path.cwd(), "local-model", "manifest.toml", validator
                )
        verify.assert_called_once_with("local-model", "manifest.toml")

    def test_patch_adapter_reuses_the_single_injected_model_service(self) -> None:
        service = _PatchService()
        model = _ServiceBackedPatchModel(service, _PatchTokenizer())

        output = model.generate(
            input_ids=torch.tensor([[1, 2]], dtype=torch.int64),
            attention_mask=torch.ones(1, 2, dtype=torch.int64),
            use_cache=False,
            do_sample=False,
            max_new_tokens=2,
        )

        self.assertTrue(torch.equal(output, torch.tensor([[1, 2, 3, 4]])))
        self.assertEqual(service.calls, [("bounded repair prompt", 2)])
        with self.assertRaises(ValueError):
            model.generate(
                input_ids=torch.tensor([[1, 2]]),
                use_cache=True,
                do_sample=False,
                max_new_tokens=2,
            )

    def test_evolution_snapshot_reads_bounded_failure_count_and_async_result(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            logdb = BoundedLogDB(
                Path(temporary) / "events.sqlite3",
                max_failure_records=4,
                max_audit_records=4,
            )
            for index in range(2):
                logdb.record_failure(
                    FailureTrace(
                        f"turn-{index}",
                        "RuntimeError",
                        "agent_runtime",
                        "agent_manager",
                        "bounded",
                    )
                )
            harness = _HarnessFixture(logdb)
            controller = EvolutionController(harness)

            self.assertEqual(controller.snapshot()["failures"], 2)
            self.assertTrue(controller.snapshot()["integrity_degraded"])
            self.assertEqual(controller.snapshot()["unreviewable_proposals"], 2)
            self.assertEqual(
                controller.snapshot()["proposal_integrity_errors"][0]["proposal_id"],
                "a" * 64,
            )
            controller.start()
            deadline = monotonic() + 2.0
            while controller.is_active and monotonic() < deadline:
                sleep(0.01)
            state = controller.snapshot()
            self.assertFalse(state["running"])
            self.assertEqual(state["status"], "succeeded")
            self.assertEqual(state["last_result"]["decision"], "ran")
            self.assertEqual(state["last_result"]["report"]["proposals"], 1)
            controller.shutdown()
            self.assertTrue(harness.stopped)


if __name__ == "__main__":
    unittest.main()
