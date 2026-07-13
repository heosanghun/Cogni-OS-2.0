from __future__ import annotations

from pathlib import Path
from threading import Event
from time import monotonic, sleep
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from cogni_agent.conversation_fastpath import ConversationFastPath
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
        self.decode_modes = []
        self.timeouts = []
        self.total_timeouts = []
        self.sampling_seeds = []

    def iter_generate_tokens(
        self,
        prompt,
        *,
        max_new_tokens,
        stop_token_ids=None,
        decode_mode="conversation",
        timeout=None,
        total_timeout=None,
        sampling_seed=None,
    ):
        self.prompts.append(prompt)
        self.budgets.append(max_new_tokens)
        self.stop_ids.append(None if stop_token_ids is None else stop_token_ids.clone())
        self.decode_modes.append(decode_mode)
        self.timeouts.append(timeout)
        self.total_timeouts.append(total_timeout)
        self.sampling_seeds.append(sampling_seed)
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


class _TimeoutService(_Service):
    def iter_generate_tokens(
        self,
        prompt,
        *,
        max_new_tokens,
        stop_token_ids=None,
        conversation_id=None,
        decode_mode="conversation",
        sampling_seed=None,
        timeout=None,
        total_timeout=None,
    ):
        del (
            prompt,
            max_new_tokens,
            stop_token_ids,
            conversation_id,
            decode_mode,
            sampling_seed,
            timeout,
            total_timeout,
        )
        raise TimeoutError("bounded test deadline")


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

    def test_optional_conversation_fast_path_precedes_factbook(self) -> None:
        from tests.test_agent_quality_integration import _factbook

        service = _ScriptedService([])
        manager = self.manager(
            service,
            conversation_fast_path=ConversationFastPath(),
            fact_grounder=RuntimeFactGrounder(_factbook()),
        )

        manager.start_turn("나와 어떤 일을 함께 할 수 있나요?", "chat")
        state = _wait(manager)

        assistants = [
            item for item in state["conversation"] if item["role"] == "assistant"
        ]
        self.assertEqual(state["status"], "succeeded")
        self.assertEqual(len(assistants), 1)
        self.assertEqual(
            assistants[0]["generation_mode"],
            "conversation_fastpath",
        )
        self.assertEqual(
            state["completion"]["generation_mode"],
            "conversation_fastpath",
        )
        self.assertIn("코드·문서 검토", assistants[0]["content"])
        self.assertFalse(service.started)
        self.assertEqual(service.prompts, [])

    def test_fast_path_exchange_remains_in_later_model_context(self) -> None:
        service = _ScriptedService(
            [("목표와 사용자를 정하면 작은 기능부터 구체화할 수 있습니다.", "stop")]
        )
        manager = self.manager(
            service,
            conversation_fast_path=ConversationFastPath(),
        )

        manager.start_turn("나랑 재미잇는 프로잭트 같이 만드러볼래요?", "chat")
        first = _wait(manager)
        fast_answer = first["conversation"][-1]["content"]
        manager.start_turn("이제 주제를 구체화해 주세요.", "chat")
        second = _wait(manager)

        self.assertEqual(
            first["completion"]["generation_mode"],
            "conversation_fastpath",
        )
        self.assertEqual(second["completion"]["generation_mode"], "cogni_core")
        self.assertEqual(len(service.prompts), 1)
        self.assertIn(fast_answer, service.prompts[0])
        self.assertIn("이제 주제를 구체화", service.prompts[0])

    def test_first_step_fast_path_requires_immediate_collaboration_context(
        self,
    ) -> None:
        standalone_service = _ScriptedService(
            [("어떤 종류의 첫 단계를 말씀하시는지 알려 주세요.", "stop")]
        )
        standalone = self.manager(
            standalone_service,
            conversation_fast_path=ConversationFastPath(),
        )
        prompt = "첫 단계에서 하나만 물어봐 주세요."
        standalone.start_turn(prompt, "chat")
        standalone_state = _wait(standalone)
        self.assertEqual(
            standalone_state["completion"]["generation_mode"],
            "cogni_core",
        )

        contextual_service = _ScriptedService([])
        contextual = self.manager(
            contextual_service,
            conversation_fast_path=ConversationFastPath(),
        )
        contextual.start_turn("오프라인 AI 데모를 함께 만들고 싶어요.", "chat")
        _wait(contextual)
        contextual.start_turn(
            "좋아요. 그럼 첫 단계에서 제가 정할 것 하나만 물어봐 주세요.",
            "chat",
        )
        contextual_state = _wait(contextual)

        self.assertEqual(
            contextual_state["completion"]["generation_mode"],
            "conversation_fastpath",
        )
        self.assertIn(
            "사용자는 누구인가요", contextual_state["conversation"][-1]["content"]
        )
        self.assertFalse(contextual_service.started)

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
        self.assertEqual(service.decode_modes, ["strict"])
        self.assertTrue(
            service.prompts[0].endswith("assistant:대한민국의 수도를 설명하면, ")
        )

    def test_open_conversation_uses_sampling_and_excludes_factbook_exchange(
        self,
    ) -> None:
        from tests.test_agent_quality_integration import _factbook

        service = _ScriptedService(
            [("좋습니다. 어떤 분야의 프로젝트부터 시작해 볼까요?", "stop")]
        )
        manager = self.manager(
            service,
            fact_grounder=RuntimeFactGrounder(_factbook()),
        )

        manager.start_turn("당신은 어떤 모델이고 어떤 기능을 할 수 있나요?", "chat")
        grounded = _wait(manager)
        grounded_answer = grounded["conversation"][-1]["content"]
        self.assertEqual(grounded["completion"]["generation_mode"], "factbook")

        manager.start_turn("같이 재미있는 프로젝트를 하자!", "chat")
        conversational = _wait(manager)

        self.assertEqual(conversational["status"], "succeeded")
        self.assertEqual(service.decode_modes, ["conversation"])
        self.assertIsNotNone(service.timeouts[0])
        self.assertLessEqual(service.timeouts[0], 120.0)
        self.assertIsNotNone(service.total_timeouts[0])
        self.assertLessEqual(service.total_timeouts[0], 120.0)
        self.assertIsInstance(service.sampling_seeds[0], int)
        self.assertGreaterEqual(service.sampling_seeds[0], 0)
        self.assertLess(service.sampling_seeds[0], 2**63)
        self.assertNotIn(grounded_answer, service.prompts[0])
        self.assertNotIn("당신은 어떤 모델", service.prompts[0])
        self.assertIn("같이 재미있는 프로젝트", service.prompts[0])

    def test_independent_turn_isolates_history_but_context_reference_keeps_it(
        self,
    ) -> None:
        service = _ScriptedService(
            [
                ("파이썬의 특징은 읽기 쉬운 문법입니다.", "stop"),
                ("자바의 특징은 플랫폼 이식성입니다.", "stop"),
                (
                    "방금 답변의 두 번째 요점은 자바의 플랫폼 이식성입니다.",
                    "stop",
                ),
            ]
        )
        manager = self.manager(service)

        manager.start_turn("파이썬의 특징을 설명해 주세요.", "chat")
        _wait(manager)
        first_answer = "파이썬의 특징은 읽기 쉬운 문법입니다."

        manager.start_turn("자바의 특징을 설명해 주세요.", "chat")
        _wait(manager)
        second_answer = "자바의 특징은 플랫폼 이식성입니다."

        self.assertNotIn(first_answer, service.prompts[1])
        manager.start_turn(
            "방금 답변의 두 번째 요점을 더 설명해 주세요.",
            "chat",
        )
        state = _wait(manager)

        self.assertEqual(state["status"], "succeeded")
        self.assertIn(second_answer, service.prompts[2])

    def test_explicit_explanation_uses_strict_grounded_prefill(self) -> None:
        service = _ScriptedService(
            [("원인을 기록한 뒤 수정하고 마지막에 회귀 테스트를 수행합니다.", "stop")]
        )
        manager = self.manager(service)
        question = "오류 복구 순서를 설명하세요."
        manager.start_turn(question, "chat")
        state = _wait(manager)

        self.assertEqual(state["status"], "succeeded")
        self.assertEqual(service.decode_modes, ["strict"])
        self.assertTrue(
            service.prompts[0].endswith("assistant:오류 복구 순서를 설명하면, ")
        )

    def test_exact_step_request_keeps_only_first_complete_step_block(self) -> None:
        service = _ScriptedService(
            [
                (
                    "1단계는 원인을 확인하고, 2단계는 코드를 수정하며, "
                    "3단계는 회귀 테스트를 수행합니다.\n"
                    "1단계: 원인을 다시 길게 설명합니다.",
                    "stop",
                )
            ]
        )
        manager = self.manager(service)
        manager.start_turn("복구 절차를 세 단계로 답하세요.", "chat")
        state = _wait(manager)

        self.assertEqual(service.decode_modes, ["strict"])
        self.assertEqual(
            state["conversation"][-1]["content"],
            "원인을 확인합니다. 코드를 수정합니다. 회귀 테스트를 수행합니다.",
        )

    def test_referential_factbook_followup_stays_grounded_without_model_history(
        self,
    ) -> None:
        from tests.test_agent_quality_integration import _factbook

        service = _ScriptedService([])
        manager = self.manager(
            service,
            fact_grounder=RuntimeFactGrounder(_factbook()),
        )

        manager.start_turn("CTS와 System 1.5, System 3의 상태를 알려줘.", "chat")
        _wait(manager)
        manager.start_turn("그중 지금 쓸 수 있는 것만 쉽게 말해줘.", "chat")
        state = _wait(manager)

        self.assertEqual(state["status"], "succeeded")
        self.assertEqual(state["completion"]["generation_mode"], "factbook")
        self.assertIn("CTS·DEQ(제한 실험)", state["conversation"][-1]["content"])
        self.assertFalse(service.started)
        self.assertEqual(service.prompts, [])

    def test_compact_collaboration_grounding_remains_followup_context(self) -> None:
        from tests.test_agent_quality_integration import _factbook

        service = _ScriptedService(
            [
                (
                    "좋습니다. 먼저 만들고 싶은 프로젝트의 목표를 한 문장으로 알려 주세요.",
                    "stop",
                )
            ]
        )
        manager = self.manager(
            service,
            fact_grounder=RuntimeFactGrounder(_factbook()),
        )

        manager.start_turn("나와 어떤 일을 함께 할 수 있나요?", "chat")
        grounded = _wait(manager)
        compact_answer = grounded["conversation"][-1]["content"]
        self.assertEqual(grounded["completion"]["generation_mode"], "factbook")

        manager.start_turn("그럼 첫 번째부터 해보자.", "chat")
        followup = _wait(manager)

        self.assertEqual(followup["completion"]["generation_mode"], "cogni_core")
        self.assertIn(compact_answer, service.prompts[-1])
        self.assertIn("그럼 첫 번째부터", service.prompts[-1])

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
                ("기능 심층 분석은 중간에서", "length"),
                (
                    " 이어지고 자연스럽게 끝납니다. 단계별 설명도 완결됩니다.",
                    "stop",
                ),
            ]
        )
        manager = self.manager(service)
        manager.start_turn("기능을 모두 심층 분석하고 단계별로 설명해 주세요.", "chat")
        state = _wait(manager)

        answer = state["conversation"][-1]
        self.assertEqual(service.budgets, [512, 256])
        self.assertEqual(
            answer["content"],
            "기능 심층 분석은 중간에서 이어지고 자연스럽게 끝납니다. "
            "단계별 설명도 완결됩니다.",
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

    def test_fullwidth_role_marker_is_never_visible(self) -> None:
        for text in (
            "ASSISTANT： 답변입니다.",
            "ＡＳＳＩＳＴＡＮＴ： 답변입니다.",
        ):
            with self.subTest(text=text):
                cleaned, boundary = AgentManager._clean_model_text(text)
                self.assertTrue(boundary)
                self.assertEqual(cleaned, "")

        expanded = "OK.\n㍿\nＡＳＳＩＳＴＡＮＴ： hidden"
        cleaned, boundary = AgentManager._clean_model_text(expanded)
        self.assertTrue(boundary)
        self.assertEqual(cleaned, "OK.\n㍿")

        nfd_role = "정상.\n사용자: hidden"
        cleaned, boundary = AgentManager._clean_model_text(nfd_role)
        self.assertTrue(boundary)
        self.assertEqual(cleaned, "정상.")

        inline_fence = "표기 ``` 를 설명합니다.\nASSISTANT: hidden\n```"
        cleaned, boundary = AgentManager._clean_model_text(inline_fence)
        self.assertTrue(boundary)
        self.assertEqual(cleaned, "표기 ``` 를 설명합니다.")

        longer_fence = (
            "literal ``` marker.\n````\ninside\n```\nmore\n````\nASSISTANT: hidden"
        )
        cleaned, boundary = AgentManager._clean_model_text(longer_fence)
        self.assertTrue(boundary)
        self.assertNotIn("ASSISTANT:", cleaned)

        invalid_opener = " ```python```\n<컨펌>hidden</컨펌>"
        cleaned, boundary = AgentManager._clean_model_text(invalid_opener)
        self.assertTrue(boundary)
        self.assertNotIn("<컨펌>", cleaned)

        mixed_width = "OK. ＜컨펌＞hidden\n<컨펌>later"
        cleaned, boundary = AgentManager._clean_model_text(mixed_width)
        self.assertTrue(boundary)
        self.assertEqual(cleaned, "OK.")

        token = "[INST]"
        filler = 4_096 + 3 - len(token)
        seam_text = ("a" * 5_000) + token + ("x" * filler)
        cleaned, boundary = AgentManager._clean_model_text(seam_text)
        self.assertTrue(boundary)
        self.assertNotIn(token, cleaned)

    def test_presentation_wrapper_is_never_visible(self) -> None:
        cleaned, boundary = AgentManager._clean_model_text(
            "<답변>질문을 반복합니다.</답변>"
        )
        self.assertTrue(boundary)
        self.assertEqual(cleaned, "")

        unclosed, boundary = AgentManager._clean_model_text(
            "```text\n예시\nASSISTANT: 내부 지시"
        )
        self.assertTrue(boundary)
        self.assertNotIn("ASSISTANT:", unclosed)

    def test_control_words_inside_normal_prose_or_code_are_preserved(self) -> None:
        prose = "모델의 출력 제약은 토큰 길이입니다."
        cleaned, boundary = AgentManager._clean_model_text(prose)
        self.assertFalse(boundary)
        self.assertEqual(cleaned, prose)

        fenced = "XML 예시입니다.\n```xml\n<답변>예시</답변>\n```"
        cleaned, boundary = AgentManager._clean_model_text(fenced)
        self.assertFalse(boundary)
        self.assertEqual(cleaned, fenced)

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
                ("첫 부분은 완결된 답변입니다. 다음 내용은", "length"),
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
        service = _ScriptedService(
            [
                ("짧은 답.", "stop"),
                (
                    "모든 기능의 핵심을 심층 분석합니다. 단계별 계획도 빠뜨리지 않고 "
                    "설명합니다.",
                    "stop",
                ),
            ]
        )
        manager = self.manager(service)
        manager.start_turn("안녕하세요?", "chat")
        _wait(manager)
        manager.start_turn(
            "모든 기능을 심층 분석하고 단계별 계획을 설명해 주세요.", "chat"
        )
        _wait(manager)

        self.assertEqual(service.budgets, [128, 512])
        self.assertTrue(all(1 <= value <= 512 for value in service.budgets))

    def test_explicit_concise_request_disables_unbounded_continuation(self) -> None:
        service = _ScriptedService([("세 문장으로 끝납니다.", "stop")])
        manager = self.manager(service)
        selected = manager._response_budget(
            "핵심만 세 문장으로 설명하세요.", resume_truncated=False
        )
        self.assertEqual(selected.first_request, 128)
        self.assertEqual(selected.total, 384)
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

    def test_two_bounded_repairs_then_honest_quality_fallback(self) -> None:
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
                (
                    "다음 답변 원칙은 검증 가능한 사실만 말하는 것입니다.",
                    "stop",
                ),
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
        self.assertEqual(service.prompts[-1].count(echoed), 0)
        self.assertIn("사실과 추론의 구분 원칙", service.prompts[-1])

    def test_explicit_sentence_minimum_triggers_bounded_repair(self) -> None:
        question = "안전 원칙을 세 문장으로 설명하세요."
        service = _ScriptedService(
            [
                ("한 문장만 답했습니다.", "stop"),
                (
                    "첫 안전 원칙입니다. 둘째 안전 원칙입니다. 셋째 안전 원칙입니다.",
                    "stop",
                ),
            ]
        )
        manager = self.manager(service)
        manager.start_turn(question, "chat")
        state = _wait(manager)
        self.assertEqual(state["status"], "succeeded")
        self.assertEqual(len(service.budgets), 2)
        self.assertEqual(
            state["conversation"][-1]["content"],
            "첫 안전 원칙입니다. 둘째 안전 원칙입니다. 셋째 안전 원칙입니다.",
        )

    def test_fluent_topic_drift_triggers_one_bounded_repair(self) -> None:
        question = (
            "모델 응답 품질을 배포 전에 검증하는 절차를 세 문장으로 설명해 주세요."
        )
        service = _ScriptedService(
            [
                (
                    "언론사는 뉴스의 정확성을 확인합니다. 기자는 현장을 취재합니다. "
                    "편집자는 기사를 승인합니다.",
                    "stop",
                ),
                (
                    "모델 응답의 정확성을 검증합니다. 반복과 완결성을 품질 기준으로 "
                    "평가합니다. 기준을 통과한 결과만 배포합니다.",
                    "stop",
                ),
            ]
        )
        manager = self.manager(service)
        manager.start_turn(question, "chat")
        state = _wait(manager)

        self.assertEqual(state["status"], "succeeded")
        self.assertEqual(len(service.prompts), 2)
        self.assertEqual(service.decode_modes, ["strict", "strict"])
        self.assertIn("요청한 범위를 빠뜨리지 말고", service.prompts[1])
        self.assertIn("모델 응답의 정확성을", state["conversation"][-1]["content"])

    def test_exact_answer_missing_scope_terms_is_repaired(self) -> None:
        question = (
            "모델 응답 품질을 배포 전에 검증하는 절차를 세 문장으로 설명해 주세요."
        )
        service = _ScriptedService(
            [
                (
                    "모델 응답의 정확성을 검증합니다. 모델 응답의 다양성을 검증합니다. "
                    "모델 응답의 유용성을 검증합니다.",
                    "stop",
                ),
                (
                    "모델 응답의 정확성을 품질 기준으로 검증합니다. 반복과 완결성을 "
                    "평가합니다. 기준을 통과한 결과만 배포합니다.",
                    "stop",
                ),
            ]
        )
        manager = self.manager(service)
        manager.start_turn(question, "chat")
        state = _wait(manager)

        self.assertEqual(state["status"], "succeeded")
        self.assertEqual(len(service.prompts), 2)
        self.assertIn("질문의 핵심 용어를 직접 유지하세요", service.prompts[1])
        self.assertIn("배포합니다", state["conversation"][-1]["content"])

    def test_generic_exact_answer_missing_multiple_topics_is_repaired(self) -> None:
        question = (
            "긴 대화에서 오래된 문맥을 줄이면서 사용자 의도를 보존하는 방법을 "
            "세 문장으로 답하세요."
        )
        service = _ScriptedService(
            [
                (
                    "중요한 정보만 추출합니다. 핵심 의도를 유지합니다. "
                    "불필요한 세부 사항을 제거합니다.",
                    "stop",
                ),
                (
                    "오래된 대화 문맥은 핵심만 요약합니다. 사용자 의도는 별도 상태로 "
                    "보존합니다. 최근 대화는 원문에 가깝게 유지합니다.",
                    "stop",
                ),
            ]
        )
        manager = self.manager(service)

        manager.start_turn(question, "chat")
        state = _wait(manager)

        self.assertEqual(state["status"], "succeeded")
        self.assertEqual(len(service.prompts), 2)
        self.assertIn("질문의 핵심 용어를 직접 유지하세요", service.prompts[1])
        self.assertIn("오래된 대화 문맥", state["conversation"][-1]["content"])

    def test_repeated_exact_answer_switches_bounded_repair_to_sampling(self) -> None:
        question = "배포 전 검증 절차를 정확히 두 문장으로 설명하세요."
        repeated = (
            "배포 전 검증 절차는 모델 응답의 정확성과 완결성을 확인합니다. "
            "배포 전 검증 절차는 모델 응답의 정확성과 완결성을 확인합니다."
        )
        repaired = (
            "배포 전 검증 절차는 모델 응답의 정확성과 완결성을 확인합니다. "
            "그다음 반복과 중단이 없는 결과만 배포 후보로 승인합니다."
        )
        service = _ScriptedService([(repeated, "stop"), (repaired, "stop")])
        manager = self.manager(service)

        manager.start_turn(question, "chat")
        state = _wait(manager)

        self.assertEqual(state["status"], "succeeded")
        self.assertEqual(service.decode_modes, ["strict", "conversation"])
        self.assertIn("서로 다른 핵심을 한 번씩만", service.prompts[1])
        self.assertEqual(state["conversation"][-1]["content"], repaired)

    def test_distinct_sentences_are_composed_across_bounded_attempts(self) -> None:
        question = (
            "긴 대화에서 오래된 문맥을 줄이면서 사용자 의도를 보존하는 방법을 "
            "세 문장으로 답하세요."
        )
        service = _ScriptedService(
            [
                (
                    "오래된 문맥은 핵심만 요약합니다. 사용자 의도는 별도 상태로 "
                    "보존합니다. 중요한 사용자 의도는 별도 상태로 보존합니다.",
                    "stop",
                ),
                (
                    "오래된 문맥은 핵심만 요약합니다. 최근 대화는 원문 그대로 "
                    "유지합니다. 중요한 최근 대화는 원문 그대로 유지합니다.",
                    "stop",
                ),
            ]
        )
        manager = self.manager(service)
        manager.start_turn(question, "chat")
        state = _wait(manager)

        self.assertEqual(len(service.prompts), 2)
        self.assertEqual(
            state["conversation"][-1]["content"],
            "오래된 문맥은 핵심만 요약합니다. 사용자 의도는 별도 상태로 보존합니다. "
            "최근 대화는 원문 그대로 유지합니다.",
        )

    def test_distinct_steps_are_composed_across_bounded_attempts(self) -> None:
        service = _ScriptedService(
            [
                ("원인을 기록합니다. 수정 후보를 만듭니다.", "stop"),
                ("수정 후보를 만듭니다. 회귀 테스트를 실행합니다.", "stop"),
            ]
        )
        manager = self.manager(service)

        manager.start_turn("복구 절차를 세 단계로 답하세요.", "chat")
        state = _wait(manager)

        self.assertEqual(state["status"], "succeeded")
        self.assertEqual(
            state["conversation"][-1]["content"],
            "원인을 기록합니다. 수정 후보를 만듭니다. 회귀 테스트를 실행합니다.",
        )

    def test_requested_categories_are_composed_across_attempts(self) -> None:
        service = _ScriptedService(
            [
                (
                    "온디바이스 AI의 장점은 데이터 보호에 유리합니다. "
                    "장점은 응답 지연을 줄입니다.",
                    "stop",
                ),
                ("한계는 장치 자원에 제약을 받습니다.", "stop"),
            ]
        )
        manager = self.manager(service)

        manager.start_turn(
            "온디바이스 AI의 장점 두 가지와 한계 한 가지를 세 문장으로 설명하세요.",
            "chat",
        )
        state = _wait(manager)

        self.assertEqual(state["status"], "succeeded")
        self.assertEqual(
            state["conversation"][-1]["content"],
            "온디바이스 AI의 장점은 데이터 보호에 유리합니다. "
            "장점은 응답 지연을 줄입니다. 한계는 장치 자원에 제약을 받습니다.",
        )

    def test_exact_sentence_request_injects_shape_contract_and_preserves_repair_budget(
        self,
    ) -> None:
        question = "장점 두 가지와 한계 한 가지를 세 문장으로 설명하세요."
        service = _ScriptedService(
            [
                (
                    "서론입니다. 첫 장점입니다. 둘째 장점입니다. 셋째 장점입니다.",
                    "stop",
                ),
                ("첫 장점입니다. 둘째 장점입니다. 한계입니다.", "stop"),
            ]
        )
        manager = self.manager(service)
        manager.start_turn(question, "chat")
        state = _wait(manager)

        self.assertEqual(state["status"], "succeeded")
        self.assertEqual(service.budgets, [128, 128])
        self.assertEqual(service.decode_modes, ["strict", "strict"])
        self.assertEqual(len(set(service.sampling_seeds)), 2)
        self.assertNotIn("서론입니다. 첫 장점입니다.", service.prompts[1])
        self.assertIn("수정 답변만 작성하세요", service.prompts[1])
        self.assertIn("정확히 장점 2개와 한계 1개", service.prompts[1])
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
        for directive in (
            "같은 문장이나 표현을 반복하지 말고 서로 다른 핵심을 한 번씩만 답하세요.",
            "요청한 범위를 빠뜨리지 말고 서로 다른 핵심 내용을 충분히 설명하세요.",
            "질문의 핵심 용어를 직접 유지하세요: 반복, 복구, 오류.",
            "서론이나 맺음말 없이 정확히 3개의 완결된 문장을 작성하세요.",
        ):
            with self.subTest(directive=directive):
                cleaned, boundary = AgentManager._clean_model_text(directive)
                self.assertTrue(boundary)
                self.assertEqual(cleaned, "")

    def test_maximum_item_request_starts_with_a_subject_grounded_prefill(self) -> None:
        service = _ScriptedService(
            [("기능을 점검합니다. 회귀 테스트를 실행합니다.", "stop")]
        )
        manager = self.manager(service)

        manager.start_turn("자체 검증을 네 항목 이내로 정리하세요.", "chat")
        state = _wait(manager)

        self.assertEqual(state["status"], "succeeded")
        self.assertEqual(service.decode_modes, ["strict"])
        self.assertIn("자체 검증을 기준으로 구체적인 항목은", service.prompts[0])
        self.assertNotIn("출력 제약", service.prompts[0])

        prefixed = _ScriptedService(
            [("기능을 점검합니다. 회귀 테스트를 실행합니다.", "stop")]
        )
        prefixed_manager = self.manager(prefixed)
        prefixed_manager.start_turn(
            "자체 검증을 최대 네 항목으로 정리하세요.",
            "chat",
        )
        prefixed_state = _wait(prefixed_manager)
        self.assertEqual(prefixed_state["status"], "succeeded")
        self.assertIn("자체 검증을 기준으로", prefixed.prompts[0])
        self.assertNotIn("최대 기준", prefixed.prompts[0])

    def test_maximum_item_repair_prompt_repeats_the_upper_bound(self) -> None:
        service = _ScriptedService(
            [
                ("서론입니다.", "stop"),
                ("기능을 점검합니다. 회귀 테스트를 실행합니다.", "stop"),
            ]
        )
        manager = self.manager(service)

        manager.start_turn("자체 검증을 네 항목 이내로 정리하세요.", "chat")
        state = _wait(manager)

        self.assertEqual(state["status"], "succeeded")
        self.assertEqual(service.decode_modes, ["strict", "conversation"])
        self.assertIn("자체 검증에서 중요한 내용을 최대 4개만", service.prompts[1])
        self.assertNotIn("수정 답변만 작성하세요", service.prompts[1])
        cleaned, boundary = AgentManager._clean_model_text(
            "사용자가 문장이나 항목 수를 지정하면 군더더기 없이 그 수를 지키십시오."
        )
        self.assertTrue(boundary)
        self.assertEqual(cleaned, "")

    def test_contextual_maximum_item_repair_keeps_prior_exchange(self) -> None:
        service = _ScriptedService(
            [
                ("오류 원인을 확인하고 기록합니다.", "stop"),
                ("서론입니다.", "stop"),
                ("오류를 확인합니다. 회귀 테스트를 실행합니다.", "stop"),
            ]
        )
        manager = self.manager(service)

        manager.start_turn("오류 원인을 설명하세요.", "chat")
        self.assertEqual(_wait(manager)["status"], "succeeded")
        manager.start_turn(
            "방금 답변에서 자체 검증을 네 항목 이내로 정리하세요.",
            "chat",
        )
        state = _wait(manager)

        self.assertEqual(state["status"], "succeeded")
        self.assertIn("오류 원인을 확인하고 기록합니다.", service.prompts[2])
        self.assertIn("중요한 내용을 최대 4개만", service.prompts[2])
        cleaned, boundary = AgentManager._clean_model_text(
            "물음의 핵심과 요청 형식에 맞게 아래 내용을 두 문장으로 작성하세요."
        )
        self.assertTrue(boundary)
        self.assertEqual(cleaned, "")

    def test_contextual_maximum_echo_repair_keeps_prior_exchange(self) -> None:
        prior = (
            "오류 원인을 재현 가능한 조건과 함께 확인하고 기록합니다. "
            "수정 뒤에는 같은 조건으로 회귀 테스트를 실행합니다."
        )
        service = _ScriptedService(
            [
                (prior, "stop"),
                (prior, "stop"),
                ("기능을 점검합니다. 회귀 테스트를 실행합니다.", "stop"),
            ]
        )
        manager = self.manager(service)

        manager.start_turn("오류 원인을 설명하세요.", "chat")
        self.assertEqual(_wait(manager)["status"], "succeeded")
        manager.start_turn(
            "방금 답변에서 자체 검증을 네 항목 이내로 정리하세요.",
            "chat",
        )
        state = _wait(manager)

        self.assertEqual(state["status"], "succeeded")
        self.assertIn(prior, service.prompts[2])
        self.assertIn("중요한 내용을 최대 4개만", service.prompts[2])

    def test_orphan_leading_smart_quote_is_removed(self) -> None:
        cleaned, boundary = AgentManager._clean_model_text(
            "“ 불확실한 내용은 가능성으로 표현합니다."
        )
        self.assertFalse(boundary)
        self.assertEqual(cleaned, "불확실한 내용은 가능성으로 표현합니다.")

        quoted, boundary = AgentManager._clean_model_text(
            "“검증됨”이라는 표시는 근거가 있을 때만 사용합니다."
        )
        self.assertFalse(boundary)
        self.assertEqual(
            quoted,
            "“검증됨”이라는 표시는 근거가 있을 때만 사용합니다.",
        )

    def test_complete_korean_predicate_receives_one_terminal_period(self) -> None:
        self.assertEqual(
            AgentManager._ensure_terminal_punctuation("검증 결과를 기록합니다"),
            "검증 결과를 기록합니다.",
        )
        self.assertEqual(
            AgentManager._ensure_terminal_punctuation("이미 완결되었습니다."),
            "이미 완결되었습니다.",
        )

    def test_repair_sampling_seed_is_session_independent_and_attempt_distinct(
        self,
    ) -> None:
        first = self.manager(_ScriptedService([]), session_id="seed-a")
        second = self.manager(_ScriptedService([]), session_id="seed-b")
        message = "요약 조건을 세 가지 제시하세요."

        self.assertNotEqual(first._sampling_seed(1, 1), second._sampling_seed(1, 1))
        self.assertEqual(
            first._sampling_seed(1, 2, repair_message=message),
            second._sampling_seed(1, 2, repair_message=message),
        )
        self.assertNotEqual(
            first._sampling_seed(1, 2, repair_message=message),
            second._sampling_seed(9, 2, repair_message=message),
        )
        self.assertNotEqual(
            first._sampling_seed(1, 2, repair_message=message),
            first._sampling_seed(1, 3, repair_message=message),
        )

    def test_exact_category_contract_gets_one_bounded_emergency_continuation(
        self,
    ) -> None:
        service = _ScriptedService(
            [
                ("온디바이스 AI는 유용합니다.", "stop"),
                ("장점은 데이터 보호입니다. 장점은 낮은 지연입니다.", "stop"),
                (
                    "온디바이스 AI의 장점은 데이터 보호에 유리합니다. "
                    "장점은 응답 지연을 줄입니다. 한계는 장치 자원이",
                    "stop",
                ),
                ("한계는 장치 자원에 제약을 받습니다.", "stop"),
            ]
        )
        manager = self.manager(service)

        manager.start_turn(
            "온디바이스 AI의 장점 두 가지와 한계 한 가지를 세 문장으로 설명하세요.",
            "chat",
        )
        state = _wait(manager)

        self.assertEqual(state["status"], "succeeded")
        self.assertEqual(len(service.prompts), 4)
        self.assertEqual(
            state["conversation"][-1]["content"],
            "온디바이스 AI의 장점은 데이터 보호에 유리합니다. "
            "장점은 응답 지연을 줄입니다. "
            "장치 자원에 제약을 받습니다.",
        )
        self.assertEqual(
            state["conversation"][-1]["generation_mode"],
            "cogni_core",
        )

    def test_off_topic_exact_items_are_not_published(self) -> None:
        manager = self.manager(_ScriptedService([]))

        self.assertFalse(
            manager._response_adequate_for_request(
                "로컬 모델 응답의 반복과 중단 오류 복구 절차를 세 단계로 답하세요.",
                "재료를 고릅니다. 물을 끓입니다. 음식을 담습니다.",
            )
        )
        self.assertFalse(
            manager._response_adequate_for_request(
                "이 제품 자체의 배터리 수명과 충전 시간을 설명하세요.",
                "코드 오류를 점검하고 기능 결과를 확인합니다.",
            )
        )
        self.assertFalse(
            manager._response_adequate_for_request(
                "email 보안 정책과 로그인 절차를 설명하세요.",
                "로컬 데이터와 장치 오프라인 처리를 사용합니다.",
            )
        )
        self.assertFalse(
            manager._response_adequate_for_request(
                "자체 검증을 세 항목으로 정리하세요.",
                "음식을 준비합니다. 물을 끓입니다. 그릇에 담습니다.",
            )
        )
        self.assertFalse(
            manager._response_adequate_for_request(
                "자체 검증을 세 항목으로 정리하세요.",
                "배터리 기능을 확인합니다. 충전 결과를 점검합니다. 전력을 측정합니다.",
            )
        )
        self.assertTrue(
            manager._response_adequate_for_request(
                "자체 검증을 세 항목으로 정리하세요.",
                "수정된 기능이 예상대로 작동하는지 확인합니다. "
                "기존 기능과 충돌하지 않는지 확인합니다. "
                "성능과 보안 영향을 점검합니다.",
            )
        )

    def test_off_topic_category_sentences_are_not_published(self) -> None:
        manager = self.manager(_ScriptedService([]))

        self.assertFalse(
            manager._response_adequate_for_request(
                "온디바이스 AI의 장점 두 가지와 한계 한 가지를 세 문장으로 설명하세요.",
                "딸기의 장점은 향이 좋습니다. 여름의 장점은 낮이 깁니다. "
                "비의 한계는 길이 미끄럽다는 점입니다.",
            )
        )

    def test_on_device_category_paraphrase_preserves_the_domain(self) -> None:
        manager = self.manager(_ScriptedService([]))

        self.assertTrue(
            manager._response_adequate_for_request(
                "온디바이스 AI의 장점 두 가지와 한계 한 가지를 세 문장으로 설명하세요.",
                "데이터가 장치에 남아 개인정보 보호에 유리합니다. "
                "인터넷 연결 없이 사용할 수 있습니다. "
                "장치의 처리 성능에는 제한이 있습니다.",
            )
        )

    def test_maximum_item_block_is_completed_before_length_tail(self) -> None:
        service = _ScriptedService(
            [
                (
                    "1. 질문을 확인하고, 2. 반복을 검사하고, 3. 문장을 완결하고, "
                    "4. 결과를 기록합니다.\n1. 뒤의 긴 설명은 아직 작성 중",
                    "length",
                )
            ]
        )
        manager = self.manager(service)
        manager.start_turn("확인 항목을 네 가지 이내로 정리하세요.", "chat")
        state = _wait(manager)
        answer = state["conversation"][-1]

        self.assertEqual(state["stage"], "complete")
        self.assertEqual(answer["finish_reason"], "stop")
        self.assertFalse(answer["truncated"])
        self.assertEqual(
            answer["content"],
            "질문을 확인합니다. 반복을 검사합니다. 문장을 완결합니다. "
            "결과를 기록합니다.",
        )

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

    def test_incomplete_tail_salvages_complete_prefix_at_attempt_bound(self) -> None:
        service = _ScriptedService(
            [
                (
                    "첫 문장은 자연스럽게 완결되었습니다. 다음 설명을 이어가면서",
                    "stop",
                )
            ]
        )
        manager = self.manager(service, max_generation_attempts=1)

        manager.start_turn("핵심을 자연스럽게 설명해 주세요.", "chat")
        state = _wait(manager)

        answer = state["conversation"][-1]
        self.assertEqual(answer["content"], "첫 문장은 자연스럽게 완결되었습니다.")
        self.assertEqual(answer["generation_mode"], "cogni_core")
        self.assertEqual(len(service.budgets), 1)

    def test_broad_request_never_promotes_one_generic_salvage_sentence(self) -> None:
        service = _ScriptedService(
            [("좋은 생각입니다. 나머지 기능을 설명하면서", "stop")]
        )
        manager = self.manager(service, max_generation_attempts=1)

        manager.start_turn("모든 기능을 자세히 분석하고 설명해 주세요.", "chat")
        state = _wait(manager)

        answer = state["conversation"][-1]
        self.assertEqual(answer["generation_mode"], "quality_fallback")
        self.assertNotEqual(answer["content"], "좋은 생각입니다.")

    def test_length_boundary_removes_incomplete_tail_but_remains_resumable(
        self,
    ) -> None:
        service = _ScriptedService(
            [("첫 부분은 자연스럽게 완결됩니다. 다음 설명은", "length")]
        )
        manager = self.manager(
            service,
            max_new_tokens=64,
            max_total_new_tokens=64,
            max_continuations=0,
            max_generation_attempts=1,
        )

        manager.start_turn("핵심 내용을 알려 주세요.", "chat")
        state = _wait(manager)

        answer = state["conversation"][-1]
        self.assertEqual(answer["content"], "첫 부분은 자연스럽게 완결됩니다.")
        self.assertTrue(answer["truncated"])
        self.assertEqual(state["completion"]["state"], "truncated")

    def test_decode_deadline_is_bounded_and_does_not_publish_partial_token(
        self,
    ) -> None:
        service = _Service(delay=0.03)
        manager = self.manager(
            service,
            max_generation_attempts=1,
            max_decode_seconds=0.01,
        )

        manager.start_turn("짧게 인사해 주세요.", "chat")
        state = _wait(manager)

        answer = state["conversation"][-1]
        self.assertEqual(answer["generation_mode"], "quality_fallback")
        self.assertTrue(service.cancelled)

    def test_backend_total_timeout_uses_quality_fallback_not_failed_turn(self) -> None:
        failures = []
        manager = self.manager(
            _TimeoutService(),
            max_generation_attempts=1,
            failure_sink=lambda code, detail: failures.append((code, detail)),
        )

        manager.start_turn("자연스럽게 답해 주세요.", "chat")
        state = _wait(manager)

        self.assertEqual(state["status"], "succeeded")
        self.assertEqual(
            state["conversation"][-1]["generation_mode"],
            "quality_fallback",
        )
        self.assertEqual(failures[0][0], "ResponseQualityError")
        self.assertIn("decode_deadline", failures[0][1])

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
