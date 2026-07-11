from __future__ import annotations

from pathlib import Path
from time import monotonic, sleep
import tempfile
from types import SimpleNamespace
import unittest

import torch

from cogni_agent.manager import ACTIVE_AGENT_STATUSES, AgentBusyError, AgentManager
from cogni_agent.model_service import GenerationChunk
from cogni_agent.tools import WorkspaceToolExecutor


class _Tokenizer:
    eos_token_id = 3

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


class _ScriptedService(_Service):
    def __init__(self, plans):
        super().__init__()
        self.plans = list(plans)
        self.budgets = []
        self.stop_ids = []

    def iter_generate_tokens(self, prompt, *, max_new_tokens, stop_token_ids=None):
        self.prompts.append(prompt)
        self.budgets.append(max_new_tokens)
        self.stop_ids.append(None if stop_token_ids is None else stop_token_ids.clone())
        self.active_request_id = len(self.prompts)
        text, finish_reason = self.plans.pop(0)
        tokens = torch.tensor([ord(character) for character in text], dtype=torch.int64)
        yield SimpleNamespace(
            request_id=self.active_request_id,
            token_ids=tokens,
            generated_total=int(tokens.numel()),
            final=True,
            cancelled=False,
            finish_reason=finish_reason,
        )
        self.active_request_id = None


class _TokenStreamService(_Service):
    def __init__(self, text):
        super().__init__()
        self.text = text

    def iter_generate_tokens(self, prompt, *, max_new_tokens, stop_token_ids=None):
        self.prompts.append(prompt)
        self.active_request_id = 1
        for index, character in enumerate(self.text, 1):
            yield SimpleNamespace(
                request_id=1,
                token_ids=torch.tensor([ord(character)], dtype=torch.int64),
                generated_total=index,
                final=False,
                cancelled=False,
                finish_reason=None,
            )
        yield SimpleNamespace(
            request_id=1,
            token_ids=torch.empty(0, dtype=torch.int64),
            generated_total=len(self.text),
            final=True,
            cancelled=False,
            finish_reason="stop",
        )
        self.active_request_id = None


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
        self.assertEqual(len(service.prompts), 1)

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

    def test_length_finish_auto_continues_and_exposes_completion_metadata(self) -> None:
        service = _ScriptedService(
            [
                ("첫 문장은 중간에서", "length"),
                (" 이어지고 자연스럽게 끝납니다.", "stop"),
            ]
        )
        manager = self.manager(service)
        manager.start_turn("기능을 모두 심층 분석하고 단계별로 설명해 주세요.", "chat")
        state = _wait(manager)

        answer = state["conversation"][-1]
        self.assertEqual(service.budgets, [512, 512])
        self.assertEqual(
            answer["content"], "첫 문장은 중간에서 이어지고 자연스럽게 끝납니다."
        )
        self.assertEqual(answer["finish_reason"], "stop")
        self.assertEqual(answer["continuations"], 1)
        self.assertFalse(answer["truncated"])
        self.assertEqual(state["completion"]["state"], "complete")
        self.assertTrue(torch.equal(service.stop_ids[0], torch.tensor([3])))

    def test_role_marker_is_a_turn_boundary_not_visible_answer_text(self) -> None:
        service = _ScriptedService(
            [("정상 답변입니다.\n\nUSER:\n프롬프트 반복\nASSISTANT:\n오염", "stop")]
        )
        manager = self.manager(service)
        manager.start_turn("간단히 답해 주세요.", "chat")
        state = _wait(manager)

        answer = state["conversation"][-1]
        self.assertEqual(answer["content"], "정상 답변입니다.")
        self.assertNotIn("USER:", answer["content"])
        self.assertFalse(answer["truncated"])

    def test_reserved_pseudo_eos_is_not_visible_or_continued(self) -> None:
        service = _ScriptedService(
            [("자연스럽게 끝납니다.<|endoftext|><|startoftext|>", "length")]
        )
        manager = self.manager(service)
        manager.start_turn("답변", "chat")
        state = _wait(manager)

        answer = state["conversation"][-1]
        self.assertEqual(answer["content"], "자연스럽게 끝납니다.")
        self.assertEqual(answer["finish_reason"], "stop")
        self.assertEqual(answer["continuations"], 0)
        self.assertFalse(answer["truncated"])

    def test_hard_total_budget_reports_truncation_and_explicit_resume(self) -> None:
        service = _ScriptedService(
            [
                ("중단 답변", "length"),
                ("완성합니다.", "stop"),
            ]
        )
        manager = self.manager(
            service,
            max_new_tokens=8,
            max_total_new_tokens=8,
            max_continuations=0,
        )
        manager.start_turn("긴 답변", "chat")
        first = _wait(manager)
        self.assertEqual(first["stage"], "truncated")
        self.assertTrue(first["conversation"][-1]["truncated"])

        manager.start_turn("계속 이어서 답해주세요!", "chat")
        second = _wait(manager)
        self.assertIn("직전 답변이 생성 길이 경계", service.prompts[-1])
        self.assertFalse(second["conversation"][-1]["truncated"])

    def test_dynamic_budget_is_deterministic_and_bounded(self) -> None:
        service = _ScriptedService([("짧은 답.", "stop"), ("상세 답.", "stop")])
        manager = self.manager(service)
        manager.start_turn("안녕하세요?", "chat")
        _wait(manager)
        manager.start_turn(
            "모든 기능을 심층 분석하고 단계별 계획을 설명해 주세요.", "chat"
        )
        _wait(manager)

        self.assertEqual(service.budgets, [256, 512])
        self.assertTrue(all(1 <= value <= 512 for value in service.budgets))

    def test_response_character_bound_is_explicit(self) -> None:
        text, truncated = AgentManager._clip_response("가" * 9_000)
        self.assertEqual(len(text), 8_192)
        self.assertTrue(text.endswith("…"))
        self.assertTrue(truncated)

    def test_empty_terminal_frame_flushes_batched_stream_once(self) -> None:
        expected = "스트리밍 최종 프레임이 비어도 답변은 끝까지 보입니다."
        manager = self.manager(_TokenStreamService(expected), system_prompt="local")
        manager.start_turn("테스트", "chat")
        state = _wait(manager)

        answer = state["conversation"][-1]
        self.assertEqual(answer["content"], expected)
        self.assertFalse(answer["streaming"])
        self.assertEqual(answer["finish_reason"], "stop")
        # UI/state rendering is batched instead of re-decoding every token.
        self.assertLess(len(state["events"]), len(expected))


if __name__ == "__main__":
    unittest.main()
