from __future__ import annotations

from pathlib import Path
from time import monotonic, sleep
import tempfile
import unittest

import torch

from cogni_agent.manager import ACTIVE_AGENT_STATUSES, AgentBusyError, AgentManager
from cogni_agent.model_service import GenerationChunk
from cogni_agent.tools import WorkspaceToolExecutor


class _Tokenizer:
    def decode(self, tokens, **_kwargs):
        return "".join(chr(value) for value in tokens)

    def apply_chat_template(self, messages, **_kwargs):
        return "|".join(f"{item['role']}:{item['content']}" for item in messages)


class _Service:
    def __init__(self, *, delay: float = 0.0) -> None:
        self.tokenizer = _Tokenizer()
        self.active_request_id = None
        self.delay = delay
        self.cancelled = False
        self.started = False
        self.prompts = []

    def start(self):
        self.started = True
        return self

    def iter_generate_tokens(self, prompt, *, max_new_tokens):
        self.prompts.append(prompt)
        self.active_request_id = 1
        for index, value in enumerate((79, 75), 1):
            if self.delay:
                sleep(self.delay)
            if self.cancelled:
                yield GenerationChunk(
                    1, torch.empty(0, dtype=torch.int64), index - 1, True, True
                )
                self.active_request_id = None
                return
            yield GenerationChunk(1, torch.tensor([value]), index, index == 2)
        self.active_request_id = None

    def cancel(self, _request_id=None):
        self.cancelled = True
        return True

    def stop(self, timeout=10.0):
        self.started = False


def _wait(manager: AgentManager, timeout: float = 5.0):
    deadline = monotonic() + timeout
    while monotonic() < deadline:
        state = manager.snapshot()
        if state["status"] not in ACTIVE_AGENT_STATUSES:
            return state
        sleep(0.01)
    raise AssertionError("agent turn did not finish")


class TestAgentManager(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        (self.root / "README.md").write_text("local project", encoding="utf-8")

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def manager(self, service=None, **kwargs):
        return AgentManager(
            service or _Service(),
            WorkspaceToolExecutor(self.root, timeout_seconds=5),
            **kwargs,
        )

    def test_streaming_chat_commits_bounded_multi_turn_history(self) -> None:
        service = _Service()
        manager = self.manager(service)
        manager.start_turn("안녕하세요", "chat")
        state = _wait(manager)

        self.assertEqual(state["status"], "succeeded")
        self.assertEqual(
            [item["role"] for item in state["conversation"]], ["user", "assistant"]
        )
        self.assertEqual(state["conversation"][-1]["content"], "OK")
        self.assertFalse(state["conversation"][-1]["streaming"])
        self.assertIn("system:", service.prompts[0])
        self.assertEqual(manager.conversations.snapshot("primary").turns[-1].text, "OK")

    def test_task_mode_runs_only_typed_allowlist(self) -> None:
        manager = self.manager()
        manager.start_turn("/read README.md", "task")
        state = _wait(manager)
        self.assertEqual(state["status"], "succeeded")
        self.assertEqual(state["conversation"][-1]["role"], "tool")
        self.assertIn("local project", state["conversation"][-1]["content"])

        manager.start_turn("임의 명령을 실행해줘", "task")
        state = _wait(manager)
        self.assertEqual(state["status"], "succeeded")
        self.assertEqual(state["conversation"][-1]["role"], "system")
        self.assertIn("/list", state["conversation"][-1]["content"])

    def test_single_turn_gate_and_cooperative_cancel(self) -> None:
        manager = self.manager(_Service(delay=0.15))
        manager.start_turn("긴 요청", "chat")
        with self.assertRaises(AgentBusyError):
            manager.start_turn("겹친 요청", "chat")
        deadline = monotonic() + 3
        while not manager.model_service.active_request_id and monotonic() < deadline:
            sleep(0.01)
        manager.cancel()
        state = _wait(manager)
        self.assertEqual(state["status"], "cancelled")
        self.assertFalse(any(item["streaming"] for item in state["conversation"]))

    def test_failure_sink_and_gpu_availability_are_enforced(self) -> None:
        failures = []
        manager = self.manager(
            failure_sink=lambda code, message: failures.append((code, message)),
            availability_check=lambda: False,
        )
        with self.assertRaises(AgentBusyError):
            manager.start_turn("blocked", "chat")
        self.assertEqual(failures, [])

    def test_reset_clears_ui_and_model_stop_is_explicit(self) -> None:
        service = _Service()
        manager = self.manager(service)
        manager.start_turn("hello", "chat")
        _wait(manager)
        manager.reset()
        self.assertEqual(manager.snapshot()["conversation"], [])
        manager.stop_model()
        self.assertFalse(service.started)


if __name__ == "__main__":
    unittest.main()
