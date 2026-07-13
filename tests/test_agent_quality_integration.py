from __future__ import annotations

from pathlib import Path
from time import monotonic, sleep
import tempfile
from types import SimpleNamespace
import unittest

import torch

from cogni_agent.fact_grounding import RuntimeFactGrounder
from cogni_agent.manager import (
    ACTIVE_AGENT_STATUSES,
    AgentManager,
    safe_quality_fallback,
)
from cogni_agent.tools import WorkspaceToolExecutor
from cogni_os.capabilities import baseline_capability_registry
from cogni_os.factbook import ModelArtifactFacts, RuntimeFactBook, TensorInventory


class _Tokenizer:
    eos_token_id = 3

    def decode(self, tokens, **_kwargs):
        return "".join(chr(value) for value in tokens)

    def apply_chat_template(self, messages, **_kwargs):
        return "|".join(f"{item['role']}:{item['content']}" for item in messages)


class _ScriptedService:
    def __init__(self, plans=()):
        self.tokenizer = _Tokenizer()
        self.active_request_id = None
        self.plans = list(plans)
        self.start_count = 0
        self.prompts: list[str] = []

    def start(self):
        self.start_count += 1
        return self

    def iter_generate_tokens(self, prompt, *, max_new_tokens, stop_token_ids=None):
        del max_new_tokens, stop_token_ids
        self.prompts.append(prompt)
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

    def cancel(self, _request_id=None):
        return False

    def stop(self, timeout=10.0):
        del timeout


def _factbook() -> RuntimeFactBook:
    return RuntimeFactBook(
        schema_version=1,
        generated_at="2026-07-12T00:00:00+00:00",
        build_version="0.2.2-test",
        device="test RTX",
        target_device="RTX 4090 24GB",
        model=ModelArtifactFacts(
            label="gemma4-e4b-test",
            architecture="Gemma4ForConditionalGeneration",
            hidden_size=2_560,
            layers=42,
            dense=True,
            inventory=TensorInventory(
                tensor_count=3,
                stored_parameters=4_500_000_000,
                effective_parameters=4_000_000_000,
                embedding_parameters=500_000_000,
                dtype_parameters=(("BF16", 4_500_000_000),),
            ),
            manifest_sha256="a" * 64,
            config_sha256="b" * 64,
        ),
        capabilities=baseline_capability_registry(),
    )


def _wait(manager: AgentManager, timeout: float = 5.0):
    deadline = monotonic() + timeout
    while monotonic() < deadline:
        state = manager.snapshot()
        if state["status"] not in ACTIVE_AGENT_STATUSES:
            return state
        sleep(0.01)
    raise AssertionError("agent turn did not finish")


class TestAgentQualityIntegration(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def manager(self, service, **kwargs) -> AgentManager:
        return AgentManager(
            service,
            WorkspaceToolExecutor(self.root, timeout_seconds=5),
            **kwargs,
        )

    def test_fact_question_bypasses_model_and_uses_verified_counts(self) -> None:
        service = _ScriptedService()
        manager = self.manager(
            service,
            fact_grounder=RuntimeFactGrounder(_factbook()),
        )

        manager.start_turn("Gemma 백본의 파라미터와 모델 크기는?", "chat")
        state = _wait(manager)

        answer = state["conversation"][-1]
        self.assertEqual(state["status"], "succeeded")
        self.assertEqual(service.start_count, 0)
        self.assertEqual(service.prompts, [])
        self.assertEqual(answer["generated_tokens"], 0)
        self.assertIn("effective 파라미터 4,000,000,000개", answer["content"])
        self.assertNotIn("70억", answer["content"])

    def test_incomplete_punctuated_fragment_is_discarded_and_repaired(self) -> None:
        service = _ScriptedService(
            [("이는 내가.", "stop"), ("정상적으로 완결된 답변입니다.", "stop")]
        )
        manager = self.manager(service)

        manager.start_turn("정확히 설명해 주세요.", "chat")
        state = _wait(manager)

        self.assertEqual(state["status"], "succeeded")
        self.assertEqual(len(service.prompts), 2)
        self.assertIn("앞 답변은 요청 형식을 충족하지 못했습니다", service.prompts[1])
        self.assertEqual(
            state["conversation"][-1]["content"],
            "정상적으로 완결된 답변입니다.",
        )

    def test_semantic_template_cycle_is_published_once(self) -> None:
        text = (
            "첫 번째 결과는 매우 빠릅니다. "
            "두 번째 결과는 정말 안전합니다. "
            "세 번째 결과는 아주 정확합니다."
        )
        service = _ScriptedService([(text, "stop")])
        manager = self.manager(service)

        manager.start_turn("결과를 설명해 주세요.", "chat")
        state = _wait(manager)

        self.assertEqual(state["status"], "succeeded")
        self.assertEqual(
            state["conversation"][-1]["content"],
            "첫 번째 결과는 매우 빠릅니다.",
        )

    def test_persistent_fragment_publishes_one_honest_quality_fallback(self) -> None:
        failures: list[tuple[str, str]] = []
        service = _ScriptedService(
            [
                ("이는 내가.", "stop"),
                ("나는.", "stop"),
                ("제가.", "stop"),
                ("그것은.", "stop"),
            ]
        )
        manager = self.manager(
            service,
            failure_sink=lambda code, message: failures.append((code, message)),
        )

        manager.start_turn("완전한 문장으로 설명해 주세요.", "chat")
        state = _wait(manager)

        self.assertEqual(state["status"], "succeeded")
        self.assertEqual(state["completion"]["state"], "complete")
        answer = state["conversation"][-1]
        self.assertEqual(answer["role"], "assistant")
        self.assertEqual(
            answer["content"],
            safe_quality_fallback("완전한 문장으로 설명해 주세요."),
        )
        self.assertEqual(answer["generation_mode"], "quality_fallback")
        self.assertEqual(len(service.prompts), 4)
        self.assertEqual(failures[0][0], "ResponseQualityError")
        diagnostic = failures[0][1]
        self.assertIn("quality_gate_v2;", diagnostic)
        self.assertIn("candidate_sha256=", diagnostic)
        self.assertIn("incomplete_korean_clause", diagnostic)
        self.assertNotIn("이는 내가", diagnostic)
        self.assertNotIn("나는.", diagnostic)
        self.assertLessEqual(len(diagnostic), 512)


if __name__ == "__main__":
    unittest.main()
