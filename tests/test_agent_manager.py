from __future__ import annotations

from pathlib import Path
from threading import Event
from time import monotonic, sleep
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from cogni_agent.manager import (
    ACTIVE_AGENT_STATUSES,
    SAFE_QUALITY_FALLBACK,
    AgentBusyError,
    AgentManager,
    safe_quality_fallback,
)
from cogni_agent.fact_grounding import RuntimeFactGrounder
from cogni_agent.model_service import GenerationChunk
from cogni_agent.tools import ToolResult, WorkspaceToolExecutor
from cogni_flow.rhythm import RhythmController


class _Tokenizer:
    eos_token_id = 3

    def decode(self, tokens, **_kwargs):
        return "".join(chr(value) for value in tokens)

    def apply_chat_template(self, messages, **_kwargs):
        return "|".join(f"{item['role']}:{item['content']}" for item in messages)


class _CountingTokenizer(_Tokenizer):
    def __call__(self, text, **_kwargs):
        return {"input_ids": torch.zeros((1, len(text)), dtype=torch.int64)}


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
        selected = self.plans.pop(0)
        text, finish_reason = selected[:2]
        generation_mode = selected[2] if len(selected) > 2 else "cogni_core"
        tokens = torch.tensor([ord(character) for character in text], dtype=torch.int64)
        yield SimpleNamespace(
            request_id=self.active_request_id,
            token_ids=tokens,
            generated_total=int(tokens.numel()),
            final=True,
            cancelled=False,
            finish_reason=finish_reason,
            generation_mode=generation_mode,
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


class _BlockingFactGrounder(RuntimeFactGrounder):
    def __init__(
        self,
        entered: Event,
        release: Event,
        *,
        error: BaseException | None = None,
    ) -> None:
        self.entered = entered
        self.release = release
        self.error = error

    def answer(self, _question: str) -> str:
        self.entered.set()
        if not self.release.wait(2):
            raise TimeoutError("fact grounder test release timed out")
        if self.error is not None:
            raise self.error
        return "검증된 Runtime Fact-book 응답입니다."


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
        self.assertEqual(
            manager.conversations.snapshot(manager.session_id).turns[-1].text,
            "OK",
        )
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
        rhythm = RhythmController()
        manager = self.manager(_Service(delay=0.15), rhythm=rhythm)
        manager.start_turn("긴 요청", "chat")
        with self.assertRaises(AgentBusyError):
            manager.start_turn("겹친 요청", "chat")
        deadline = monotonic() + 3
        while not manager.model_service.active_request_id and monotonic() < deadline:
            sleep(0.01)
        self.assertEqual(rhythm.active_requests, 1)
        manager.cancel()
        state = _wait(manager)
        self.assertEqual(state["status"], "cancelled")
        self.assertEqual(rhythm.active_requests, 0)
        self.assertFalse(any(item["streaming"] for item in state["conversation"]))

    def test_shared_rhythm_covers_factbook_bypass_without_loading_model(self) -> None:
        rhythm = RhythmController()
        entered = Event()
        release = Event()
        service = _Service()
        manager = self.manager(
            service,
            rhythm=rhythm,
            fact_grounder=_BlockingFactGrounder(entered, release),
        )

        manager.start_turn("Cogni-OS 정체성을 알려줘", "chat")
        self.assertTrue(entered.wait(1))
        self.assertEqual(rhythm.active_requests, 1)
        self.assertFalse(service.started)
        with self.assertRaisesRegex(RuntimeError, "active inference requests"):
            rhythm.enter_evolution(lambda: None)
        release.set()
        state = _wait(manager)

        self.assertEqual(state["status"], "succeeded")
        self.assertEqual(state["stage"], "complete")
        self.assertIn("Runtime Fact-book", state["conversation"][-1]["content"])
        self.assertEqual(
            state["conversation"][-1]["generation_mode"],
            "factbook",
        )
        self.assertEqual(state["completion"]["generation_mode"], "factbook")
        self.assertFalse(state["core"]["model_loaded"])
        self.assertEqual(state["core"]["modules"]["gemma"], "not_loaded")
        self.assertEqual(rhythm.active_requests, 0)

    def test_general_question_starts_model_and_uses_cogni_core_generation(self) -> None:
        from tests.test_agent_quality_integration import _factbook

        service = _ScriptedService(
            [
                (
                    "대한민국의 수도는 서울입니다. 서울은 대한민국의 행정 중심지입니다.",
                    "stop",
                )
            ]
        )
        manager = self.manager(
            service,
            fact_grounder=RuntimeFactGrounder(_factbook()),
        )

        manager.start_turn("대한민국의 수도를 한 문장으로 알려줘.", "chat")
        state = _wait(manager)

        self.assertEqual(state["status"], "succeeded")
        self.assertTrue(service.started)
        self.assertEqual(len(service.prompts), 1)
        self.assertEqual(
            state["conversation"][-1]["generation_mode"],
            "cogni_core",
        )
        self.assertEqual(state["completion"]["generation_mode"], "cogni_core")

    def test_shared_rhythm_covers_task_mode_until_tool_completion(self) -> None:
        rhythm = RhythmController()
        entered = Event()
        release = Event()
        manager = self.manager(rhythm=rhythm)

        def blocking_execute(_request):
            entered.set()
            if not release.wait(2):
                raise TimeoutError("tool test release timed out")
            return ToolResult("read", True, "bounded tool complete", 0.0)

        with patch.object(
            manager.tool_executor,
            "execute",
            side_effect=blocking_execute,
        ):
            manager.start_turn("/read README.md", "task")
            self.assertTrue(entered.wait(1))
            self.assertEqual(rhythm.active_requests, 1)
            with self.assertRaisesRegex(RuntimeError, "active inference requests"):
                rhythm.enter_evolution(lambda: None)
            release.set()
            state = _wait(manager)

        self.assertEqual(state["status"], "succeeded")
        self.assertEqual(state["conversation"][-1]["content"], "bounded tool complete")
        self.assertEqual(rhythm.active_requests, 0)

    def test_rhythm_slot_releases_before_exception_becomes_terminal(self) -> None:
        rhythm = RhythmController()
        entered = Event()
        release = Event()
        release.set()
        manager = self.manager(
            rhythm=rhythm,
            fact_grounder=_BlockingFactGrounder(
                entered,
                release,
                error=RuntimeError("grounded failure"),
            ),
        )

        manager.start_turn("Cogni-OS 정체성", "chat")
        state = _wait(manager)

        self.assertTrue(entered.is_set())
        self.assertEqual(state["status"], "failed")
        self.assertEqual(state["error"]["code"], "RuntimeError")
        self.assertEqual(rhythm.active_requests, 0)

    def test_default_rhythm_is_private_and_evolution_mode_fails_closed(self) -> None:
        first = self.manager()
        second = self.manager()
        self.assertIsInstance(first.rhythm, RhythmController)
        self.assertIsNot(first.rhythm, second.rhythm)

        first.rhythm.enter_evolution(lambda: None)
        first.start_turn("실행하면 안 되는 대화", "chat")
        state = _wait(first)
        self.assertEqual(state["status"], "failed")
        self.assertIn("inference unavailable", state["error"]["message"])
        self.assertEqual(first.rhythm.active_requests, 0)

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
        first_session = manager.session_id
        manager.start_turn("hello", "chat")
        _wait(manager)
        manager.reset()
        self.assertEqual(manager.snapshot()["conversation"], [])
        self.assertNotEqual(manager.session_id, first_session)
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

    def test_unattested_gemma_lkg_mode_is_rejected(self) -> None:
        service = _ScriptedService(
            [("검증된 기본 모델로 안전하게 복귀했습니다.", "stop", "gemma_lkg")]
        )
        manager = self.manager(service)
        manager.start_turn("복귀 시험", "chat")
        state = _wait(manager)
        self.assertEqual(state["status"], "failed")
        self.assertEqual(state["error"]["code"], "ModelServiceError")
        self.assertFalse(
            any(item["role"] == "assistant" for item in state["conversation"])
        )

    def test_repeated_sentence_block_is_exposed_once_and_never_continued(self) -> None:
        block = (
            "백본은 확인된 기능과 설계 목표를 구분하여 정확하게 답합니다. "
            "같은 설명은 불필요하게 다시 출력하지 않습니다."
        )
        service = _ScriptedService(
            [
                (block + "\n\n" + block, "length"),
                ("이 청크는 실행되면 안 됩니다.", "stop"),
            ]
        )
        manager = self.manager(service)
        manager.start_turn("백본을 설명해 주세요.", "chat")
        state = _wait(manager)

        answer = state["conversation"][-1]
        self.assertEqual(answer["content"], block)
        self.assertEqual(answer["finish_reason"], "stop")
        self.assertEqual(answer["continuations"], 0)
        self.assertFalse(answer["truncated"])
        self.assertEqual(len(service.budgets), 1)

    def test_short_emphasis_and_structured_text_are_not_over_deduplicated(self) -> None:
        self.assertEqual(
            AgentManager._trim_repeated_text("네. 네."),
            ("네. 네.", False),
        )
        self.assertEqual(
            AgentManager._trim_repeated_text("네. 네. 네."),
            ("네.", True),
        )
        listed = "1. 확인합니다.\n2. 확인합니다.\n1. 확인합니다.\n2. 확인합니다."
        self.assertEqual(AgentManager._trim_repeated_text(listed), (listed, False))

    def test_whitespace_variation_in_adjacent_blocks_is_trimmed(self) -> None:
        first = "첫 문장은 충분히 길고 완전한 설명을 제공합니다. 둘째 문장도 자연스럽게 끝납니다."
        repeated = (
            first
            + "\n\n첫 문장은   충분히 길고 완전한 설명을 제공합니다.\n"
            + "둘째 문장도 자연스럽게 끝납니다."
        )
        trimmed, detected = AgentManager._trim_repeated_text(repeated)
        self.assertTrue(detected)
        self.assertEqual(trimmed, first)
        sentence = "충분히 긴 동일 문장은 마지막 마침표가 없어도 한 번만 남습니다."
        trimmed, detected = AgentManager._trim_repeated_text(
            sentence + " " + sentence.rstrip(".")
        )
        self.assertTrue(detected)
        self.assertEqual(trimmed, sentence)
        self.assertEqual(
            AgentManager._close_repetition_boundary(sentence.rstrip(".")),
            sentence,
        )

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

    def test_explicit_concise_request_disables_unbounded_continuation(self) -> None:
        service = _ScriptedService([("세 문장으로 끝납니다.", "stop")])
        manager = self.manager(service)
        selected = manager._response_budget(
            "핵심만 세 문장으로 설명하세요.", resume_truncated=False
        )
        self.assertEqual(selected.first_request, 192)
        self.assertEqual(selected.total, 576)
        self.assertEqual(selected.max_continuations, 0)

    def test_raw_factbook_prompt_echo_is_cut_at_its_first_marker(self) -> None:
        cleaned, boundary = AgentManager._clean_model_text(
            "자연스럽게 끝난 답변입니다.\n"
            "[Runtime Fact-book: 내부 원문은 공개하지 않습니다.]"
        )
        self.assertTrue(boundary)
        self.assertEqual(cleaned, "자연스럽게 끝난 답변입니다.")

        cleaned, boundary = AgentManager._clean_model_text(
            "두 번째 정상 답변입니다.\n[턴 종료]\n["
        )
        self.assertTrue(boundary)
        self.assertEqual(cleaned, "두 번째 정상 답변입니다.")

    def test_one_leading_user_question_echo_is_removed(self) -> None:
        question = "온디바이스 AI의 장점 두 가지를 설명해 주세요."
        cleaned, removed = AgentManager._strip_leading_user_echo(
            question + "\n실제 답변은 이 문장입니다.", question
        )
        self.assertTrue(removed)
        self.assertEqual(cleaned, "실제 답변은 이 문장입니다.")

    def test_interactive_prompt_drops_only_old_pairs_before_2048_token_boundary(
        self,
    ) -> None:
        service = _Service()
        service.tokenizer = _CountingTokenizer()
        service.max_input_tokens = 4_096
        manager = self.manager(service)
        messages = [
            {"role": "user", "content": "old-user-1 " + "가" * 600},
            {"role": "assistant", "content": "old-answer-1 " + "나" * 600},
            {"role": "user", "content": "old-user-2 " + "다" * 600},
            {"role": "assistant", "content": "old-answer-2 " + "라" * 600},
            {"role": "user", "content": "current-request"},
        ]

        rendered = manager._render_bounded_prompt(messages)

        self.assertLessEqual(len(rendered), 2_048)
        self.assertNotIn("old-user-1", rendered)
        self.assertNotIn("old-answer-1", rendered)
        self.assertIn("current-request", rendered)

    def test_echo_only_empty_candidate_gets_exactly_one_bounded_repair(self) -> None:
        question = "온디바이스 AI의 장점을 세 문장으로 설명하세요."
        service = _ScriptedService(
            [
                (question + "\n[턴 종료]", "stop"),
                ("첫째 장점입니다. 둘째 장점입니다. 마지막 한계입니다.", "stop"),
            ]
        )
        manager = self.manager(service)
        manager.start_turn(question, "chat")
        state = _wait(manager)
        self.assertEqual(state["status"], "succeeded")
        self.assertEqual(len(service.budgets), 2)
        self.assertLessEqual(service.budgets[1], 192)
        self.assertEqual(
            state["conversation"][-1]["content"],
            "첫째 장점입니다. 둘째 장점입니다. 마지막 한계입니다.",
        )

    def test_two_failed_repairs_publish_one_honest_quality_fallback(self) -> None:
        question = "온디바이스 AI의 장점을 세 문장으로 설명하세요."
        echo = question + "\n[턴 종료]"
        service = _ScriptedService([(echo, "stop"), (echo, "stop"), (echo, "stop")])
        manager = self.manager(service)
        manager.start_turn(question, "chat")
        state = _wait(manager)
        answer = state["conversation"][-1]
        self.assertEqual(state["status"], "succeeded")
        self.assertEqual(len(service.budgets), 3)
        self.assertEqual(answer["content"], safe_quality_fallback(question))
        self.assertEqual(answer["generation_mode"], "quality_fallback")

    def test_quality_fallback_exchange_is_not_fed_back_to_the_model(self) -> None:
        question = "온디바이스 AI의 장점을 한 문장으로 설명하세요."
        echo = question + "\n[턴 종료]"
        service = _ScriptedService(
            [
                (echo, "stop"),
                (echo, "stop"),
                (echo, "stop"),
                ("다음 요청에는 검증 가능한 사실만 답합니다.", "stop"),
            ]
        )
        manager = self.manager(service)
        manager.start_turn(question, "chat")
        first = _wait(manager)
        self.assertEqual(
            first["conversation"][-1]["content"], safe_quality_fallback(question)
        )

        manager.start_turn("다음 답변 원칙을 알려주세요.", "chat")
        second = _wait(manager)
        self.assertEqual(second["status"], "succeeded")
        self.assertNotIn(SAFE_QUALITY_FALLBACK, service.prompts[-1])

    def test_two_sentence_cross_turn_echo_is_repaired_without_prior_history(
        self,
    ) -> None:
        echoed = (
            "개인정보는 장치 안에서만 처리되어 외부 노출을 줄입니다. "
            "네트워크 왕복이 없어 응답 지연도 줄어듭니다."
        )
        corrected = (
            "확인된 사실은 근거와 함께 명시합니다. 추론은 별도 표현으로 구분합니다."
        )
        service = _ScriptedService(
            [
                (echoed, "stop"),
                (echoed, "stop"),
                (corrected, "stop"),
            ]
        )
        manager = self.manager(service)
        manager.start_turn("온디바이스 처리의 특징을 두 문장으로 답하세요.", "chat")
        first = _wait(manager)
        self.assertEqual(first["conversation"][-1]["content"], echoed)

        manager.start_turn("사실과 추론의 구분 원칙을 두 문장으로 답하세요.", "chat")
        second = _wait(manager)
        self.assertEqual(second["conversation"][-1]["content"], corrected)
        self.assertEqual(service.prompts[-1].count(echoed), 1)
        self.assertIn("사실과 추론의 구분 원칙", service.prompts[-1])

    def test_explicit_sentence_minimum_triggers_bounded_repair(self) -> None:
        question = "안전 원칙을 세 문장으로 설명하세요."
        service = _ScriptedService(
            [
                ("한 문장만 답했습니다.", "stop"),
                ("첫 문장입니다. 둘째 문장입니다. 셋째 문장입니다.", "stop"),
            ]
        )
        manager = self.manager(service)
        manager.start_turn(question, "chat")
        state = _wait(manager)
        self.assertEqual(state["status"], "succeeded")
        self.assertEqual(len(service.budgets), 2)
        self.assertEqual(
            state["conversation"][-1]["content"],
            "첫 문장입니다. 둘째 문장입니다. 셋째 문장입니다.",
        )

    def test_exact_sentence_request_injects_shape_contract_and_preserves_repair_budget(
        self,
    ) -> None:
        question = "장점 두 가지와 한계 한 가지를 세 문장으로 설명하세요."
        service = _ScriptedService(
            [
                (
                    "서론입니다. 첫 장점입니다. 둘째 장점입니다. 한계입니다.",
                    "stop",
                ),
                ("첫 장점입니다. 둘째 장점입니다. 한계입니다.", "stop"),
            ]
        )
        manager = self.manager(service)
        manager.start_turn(question, "chat")
        state = _wait(manager)

        self.assertEqual(state["status"], "succeeded")
        self.assertEqual(service.budgets, [192, 192])
        self.assertIn("서론입니다. 첫 장점입니다.", service.prompts[1])
        self.assertIn("수정 답변만 작성하세요", service.prompts[1])
        self.assertEqual(
            state["conversation"][-1]["content"],
            "첫 장점입니다. 둘째 장점입니다. 한계입니다.",
        )

    def test_quality_repair_instruction_echo_is_never_published(self) -> None:
        cleaned, boundary = AgentManager._clean_model_text(
            "앞 답변은 요청 형식을 충족하지 못했습니다. 원래 질문에 대한 수정 답변만 작성하세요."
        )
        self.assertTrue(boundary)
        self.assertEqual(cleaned, "")

    def test_concise_mixed_subject_fragment_is_repaired_from_scratch(self) -> None:
        service = _ScriptedService(
            [
                ("확인하지 못한 결과를 두 문장으로 설명하면 AI가", "stop"),
                (
                    "확인되지 않은 결과는 성공으로 단정할 수 없습니다. "
                    "증거를 확인한 뒤 상태를 보고해야 합니다.",
                    "stop",
                ),
            ]
        )
        manager = self.manager(service)
        manager.start_turn("두 문장으로 설명하세요.", "chat")
        state = _wait(manager)
        self.assertEqual(state["status"], "succeeded")
        self.assertEqual(len(service.budgets), 2)
        self.assertNotIn("AI가", state["conversation"][-1]["content"])

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
