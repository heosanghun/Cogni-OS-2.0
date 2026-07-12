import unittest

from cogni_agent.fact_grounding import (
    FactGroundingError,
    FactGroundingLimits,
    RuntimeFactGrounder,
    ground_runtime_fact,
)
from cogni_os.capabilities import baseline_capability_registry
from cogni_os.factbook import ModelArtifactFacts, RuntimeFactBook, TensorInventory


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


class TestRuntimeFactGrounder(unittest.TestCase):
    def setUp(self) -> None:
        self.facts = _factbook()
        self.grounder = RuntimeFactGrounder(self.facts)

    def test_identity_and_parameters_come_from_runtime_factbook(self):
        identity = self.grounder.answer("당신은 누구이며 어떤 모델인가요?")
        self.assertIsNotNone(identity)
        self.assertIn("gemma4-e4b-test", identity)
        self.assertIn("0.2.2-test", identity)
        self.assertIn("test RTX", identity)
        self.assertIn("RTX 4090 24GB", identity)

        parameters = self.grounder.answer("Gemma 백본의 파라미터와 모델 크기는?")
        self.assertIsNotNone(parameters)
        self.assertIn("저장 파라미터 4,500,000,000개", parameters)
        self.assertIn("effective 파라미터 4,000,000,000개", parameters)
        self.assertIn("hidden size는 2,560", parameters)
        self.assertIn("레이어는 42개", parameters)

    def test_each_core_system_reports_honest_capability_state(self):
        cases = (
            ("CTS와 DEQ는 답변에 어떻게 쓰이나요?", "CTS · DEQ", "canary"),
            ("System 1.5 Fast Weight를 설명해줘", "System 1.5", "gated"),
            ("FP-EWC와 C-FIRE가 포함된 System 2.5는?", "System 2.5", "night_only"),
            ("System 3 sparse expert가 뭐야?", "System 3", "advisory"),
            ("System 4 tensor swarm을 알려줘", "System 4", "advisory"),
        )
        for question, title, state in cases:
            with self.subTest(question=question):
                answer = self.grounder.answer(question)
                self.assertIsNotNone(answer)
                self.assertIn(title, answer)
                self.assertIn(state, answer)

    def test_mirror_therapy_is_product_self_harness_not_psychotherapy(self):
        answer = self.grounder.answer("자가 거울치료가 무엇인가요?")
        self.assertIsNotNone(answer)
        self.assertIn("Self-Harness", answer)
        self.assertIn("의학적 심리치료가 아니라", answer)
        self.assertIn("proposal_only", answer)
        self.assertIn("활성 소스를 스스로 덮어쓰는 권한은 없습니다", answer)

    def test_overview_is_one_stably_ordered_answer(self):
        answer = self.grounder.answer("당신의 모든 기능을 설명해 주세요")
        self.assertIsNotNone(answer)
        labels = (
            "Cogni Agent",
            "모델 파라미터",
            "CTS · DEQ",
            "System 1.5",
            "System 2.5",
            "System 3",
            "System 4",
            "Self-Harness",
        )
        offsets = [answer.index(label) for label in labels]
        self.assertEqual(offsets, sorted(offsets))

    def test_exact_operator_greeting_returns_identity_and_capability_overview(self):
        answer = self.grounder.answer(
            "안녕하세여! 당신은 어떤 모델이고 어떤 기능을 할 수 있나요?"
        )

        self.assertIsNotNone(answer)
        assert answer is not None
        for label in (
            "gemma4-e4b-test",
            "CTS · DEQ",
            "System 1.5",
            "System 2.5",
            "System 3",
            "System 4",
            "Self-Harness",
        ):
            with self.subTest(label=label):
                self.assertIn(label, answer)

    def test_general_conversation_falls_through_to_local_model(self):
        for question in (
            "대한민국의 수도를 한 문장으로 알려줘.",
            "오늘 날씨가 어때?",
            "파이썬에서 리스트를 정렬하는 방법을 알려줘.",
            "심리학에서 거울 노출 기법은 무엇인가요?",
            "안녕하세요",
            "",
        ):
            with self.subTest(question=question):
                self.assertIsNone(self.grounder.answer(question))

    def test_factbook_followup_separates_verified_state_from_targets(self):
        previous = self.grounder.answer(
            "CTS와 System 1.5의 검증 상태를 각각 설명하세요."
        )
        self.assertIsNotNone(previous)
        answer = self.grounder.answer_followup(
            "방금 답변에서 실제 검증과 향후 목표를 두 문장으로 구분하세요.",
            previous,
        )
        self.assertIsNotNone(answer)
        self.assertEqual(answer.count("."), 2)
        self.assertIn("measured 또는 verified", answer)
        self.assertIn("target·plan·research·canary", answer)
        self.assertIsNone(
            self.grounder.answer_followup(
                "방금 답변을 요약하세요.", "일반적인 모델 답변입니다."
            )
        )

    def test_unverified_tool_success_is_grounded_in_operational_policy(self):
        answer = self.grounder.answer(
            "도구 실행 결과를 확인하지 못했을 때 성공했다고 말하면 안 되는 이유를 두 문장으로 답하세요."
        )
        self.assertIsNotNone(answer)
        self.assertEqual(answer.count("."), 2)
        self.assertIn("성공을 뒷받침할 증거가 없으므로", answer)
        self.assertIn("검증 상태를 그대로 밝혀야", answer)
        self.assertIsNone(self.grounder.answer("일반적인 도구 실행 방법을 설명하세요."))

    def test_multi_topic_question_returns_each_topic_once(self):
        answer = self.grounder.answer("System 1.5와 System 2.5의 차이는?")
        self.assertIsNotNone(answer)
        self.assertEqual(answer.count("System 1.5 · Fast Weight"), 1)
        self.assertEqual(answer.count("System 2.5 · FP-EWC/C-FIRE"), 1)

    def test_brand_prefix_does_not_add_identity_or_duplicate_parameter_counts(self):
        modules = self.grounder.answer(
            "Cogni-OS의 CTS, System 1.5, System 3, System 4를 설명하고 이상입니다로 끝내세요"
        )
        self.assertIsNotNone(modules)
        self.assertNotIn("저는 Cogni-OS", modules)
        self.assertTrue(modules.endswith("이상입니다."))

        identity = self.grounder.answer(
            "당신은 어떤 모델이며 저장 파라미터와 effective 파라미터는 몇 개인가요?"
        )
        self.assertIsNotNone(identity)
        self.assertEqual(identity.count("저장 파라미터 4,500,000,000개"), 1)
        self.assertEqual(identity.count("effective 파라미터 4,000,000,000개"), 1)

    def test_input_and_output_are_hard_bounded(self):
        limited_input = RuntimeFactGrounder(
            self.facts,
            FactGroundingLimits(question_chars=8, answer_chars=8_192),
        )
        with self.assertRaises(FactGroundingError):
            limited_input.answer("x" * 9)

        limited_output = RuntimeFactGrounder(
            self.facts,
            FactGroundingLimits(question_chars=4_096, answer_chars=16),
        )
        with self.assertRaises(FactGroundingError):
            limited_output.answer("CTS가 무엇인가요?")

        with self.assertRaises(TypeError):
            self.grounder.answer(123)  # type: ignore[arg-type]

    def test_convenience_function_has_same_semantics(self):
        answer = ground_runtime_fact("Self-Harness 상태는?", self.facts)
        self.assertEqual(answer, self.grounder.answer("Self-Harness 상태는?"))


if __name__ == "__main__":
    unittest.main()
