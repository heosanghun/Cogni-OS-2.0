from __future__ import annotations

import unittest

from cogni_agent.response_quality import (
    MAX_INSPECT_CHARS,
    QualityAction,
    QualityCode,
    inspect_response,
)


class ResponseQualityTests(unittest.TestCase):
    def test_number_ordinal_and_adjective_template_is_detected(self) -> None:
        text = (
            "1번째 응답은 매우 빠릅니다.\n"
            "2번째 응답은 정말 안전합니다.\n"
            "3번째 응답은 아주 정확합니다."
        )

        report = inspect_response(text)

        self.assertTrue(report.has(QualityCode.TEMPLATE_REPETITION))
        self.assertEqual(report.recommended_action, QualityAction.TRIM_AND_STOP)
        self.assertEqual(report.recommended_cut_index, text.index("2번째"))

    def test_substantive_numbered_steps_are_not_template_repetition(self) -> None:
        text = (
            "1. 원인을 분석합니다.\n"
            "2. 코드를 수정합니다.\n"
            "3. 전체 회귀 테스트를 실행합니다."
        )

        report = inspect_response(text, final=True)

        self.assertTrue(report.clean)
        self.assertEqual(report.recommended_action, QualityAction.ACCEPT)

    def test_duplicate_numbered_item_bodies_are_not_hidden_by_list_markers(
        self,
    ) -> None:
        text = (
            "1. 개인정보 시스템은 분리되어야 합니다.\n"
            "2. 개인정보 시스템은 분리되어야 합니다.\n"
            "3. 접근 기록은 보존되어야 합니다."
        )
        report = inspect_response(text)
        self.assertTrue(report.has(QualityCode.TEMPLATE_REPETITION))
        self.assertEqual(
            report.recommended_cut_index,
            text.index("개인정보 시스템", text.index("2.")),
        )

    def test_long_exact_sentence_block_is_trimmed_at_second_copy(self) -> None:
        sentence = "검증된 결과만 표시하고 목표 수치는 명확하게 분리하여 설명합니다."
        text = sentence + " " + sentence

        report = inspect_response(text)

        self.assertTrue(report.has(QualityCode.TEMPLATE_REPETITION))
        self.assertEqual(report.recommended_cut_index, len(sentence) + 1)

    def test_short_low_information_sentence_cycle_is_detected(self) -> None:
        text = "네, 확인했습니다. 네, 확인했습니다. 네, 확인했습니다."

        report = inspect_response(text)

        self.assertTrue(report.has(QualityCode.LOW_INFORMATION_REPETITION))
        self.assertEqual(report.recommended_cut_index, text.index("네", 1))

    def test_low_information_token_cycle_without_punctuation_is_detected(self) -> None:
        text = "네 알겠습니다 네 알겠습니다 네 알겠습니다"

        report = inspect_response(text)

        self.assertTrue(report.has(QualityCode.LOW_INFORMATION_REPETITION))
        self.assertEqual(report.recommended_action, QualityAction.TRIM_AND_STOP)

    def test_normal_reuse_of_domain_words_is_not_low_information(self) -> None:
        text = (
            "모델을 먼저 검증합니다. 모델이 통과하면 CTS를 실행합니다. "
            "마지막으로 모델 응답과 CTS 지표를 함께 기록합니다."
        )

        report = inspect_response(text, final=True)

        self.assertFalse(report.has(QualityCode.LOW_INFORMATION_REPETITION))
        self.assertEqual(report.recommended_action, QualityAction.ACCEPT)

    def test_role_marker_is_a_hard_generation_boundary(self) -> None:
        text = "정상 답변입니다.\nASSISTANT: 반복된 새 역할"

        report = inspect_response(text)

        self.assertTrue(report.has(QualityCode.ROLE_MARKER))
        self.assertEqual(report.recommended_action, QualityAction.TRIM_AND_STOP)
        self.assertEqual(report.recommended_cut_index, text.index("ASSISTANT:"))

    def test_reserved_control_tokens_are_detected(self) -> None:
        samples = (
            "정상 답변<end_of_turn>",
            "정상 답변<|eot_id|>",
            "정상 답변[INST]오염[/INST]",
            "정상 답변<unused123>",
            "정상 답변[턴 종료]",
        )

        for text in samples:
            with self.subTest(text=text):
                report = inspect_response(text)
                self.assertTrue(report.has(QualityCode.CONTROL_TOKEN))
                self.assertEqual(
                    report.recommended_action,
                    QualityAction.TRIM_AND_STOP,
                )

    def test_runtime_factbook_prompt_echo_is_a_control_boundary(self) -> None:
        text = "완결된 답변입니다.\n[Runtime Fact-book: 내부 프롬프트 원문]"
        report = inspect_response(text)
        self.assertTrue(report.has(QualityCode.CONTROL_TOKEN))
        self.assertEqual(report.recommended_cut_index, text.index("[Runtime"))

    def test_mixed_latin_subject_with_korean_particle_is_incomplete(self) -> None:
        report = inspect_response("도구 결과를 확인하지 못했을 때 AI가", final=True)
        self.assertTrue(report.has(QualityCode.INCOMPLETE_KOREAN_CLAUSE))
        self.assertEqual(report.recommended_action, QualityAction.CONTINUE)

    def test_korean_connective_fragments_request_continuation_only_when_final(
        self,
    ) -> None:
        samples = (
            "회귀 테스트를 모두 실행하고",
            "안전한 자동 승격을 위해",
            "현재 시스템은",
            "다음 검증 단계는:",
        )

        for text in samples:
            with self.subTest(text=text):
                streaming = inspect_response(text, final=False)
                final = inspect_response(text, final=True)
                self.assertFalse(streaming.has(QualityCode.INCOMPLETE_KOREAN_CLAUSE))
                self.assertTrue(final.has(QualityCode.INCOMPLETE_KOREAN_CLAUSE))
                self.assertEqual(final.recommended_action, QualityAction.CONTINUE)
                self.assertIsNone(final.recommended_cut_index)

    def test_complete_korean_and_terminal_code_fence_are_accepted(self) -> None:
        samples = (
            "회귀 테스트를 모두 실행했습니다",
            "모든 작업이 안전하게 완료되었습니다.",
            "예시는 다음과 같습니다:\n```python\nvalue = 1\n```",
        )

        for text in samples:
            with self.subTest(text=text):
                report = inspect_response(text, final=True)
                self.assertFalse(report.has(QualityCode.INCOMPLETE_KOREAN_CLAUSE))
                self.assertEqual(report.recommended_action, QualityAction.ACCEPT)

    def test_terminal_punctuation_does_not_hide_a_pronoun_fragment(self) -> None:
        for text in ("이는 내가.", "나는.", "현재 책임자는 제가."):
            with self.subTest(text=text):
                report = inspect_response(text, final=True)
                self.assertTrue(report.has(QualityCode.INCOMPLETE_KOREAN_CLAUSE))
                self.assertEqual(report.recommended_action, QualityAction.CONTINUE)

        self.assertEqual(
            inspect_response("제가 담당자입니다.", final=True).recommended_action,
            QualityAction.ACCEPT,
        )

    def test_empty_trailing_list_marker_is_incomplete_despite_period(self) -> None:
        for text in (
            "원칙은 다음과 같습니다:\n\n1.",
            "첫 항목은 완전합니다.\n2.",
            "절차는 다음과 같습니다:\n-",
        ):
            with self.subTest(text=text):
                report = inspect_response(text, final=True)
                self.assertTrue(report.has(QualityCode.INCOMPLETE_KOREAN_CLAUSE))
                self.assertEqual(report.recommended_action, QualityAction.CONTINUE)

    def test_long_unpunctuated_noun_fragment_requires_repair(self) -> None:
        incomplete = (
            "반복 없는 좋은 요약문은 핵심을 간결하게 전달하고 같은 표현을 "
            "되풀이하지 않아야 합니다. 이 답변은 반복 없는 요약 조건을 충족"
        )
        complete = (
            "반복 없는 좋은 요약문은 핵심을 간결하게 전달하고 같은 표현을 "
            "되풀이하지 않아야 합니다. 이 답변은 반복 없는 요약 조건을 충족합니다"
        )
        self.assertTrue(
            inspect_response(incomplete, final=True).has(
                QualityCode.INCOMPLETE_KOREAN_CLAUSE
            )
        )
        self.assertEqual(
            inspect_response(complete, final=True).recommended_action,
            QualityAction.ACCEPT,
        )

    def test_explicit_request_minimum_counts_completed_sentences(self) -> None:
        from cogni_agent.response_quality import (
            requested_exact_sentence_count,
            response_contract_satisfied,
        )

        request = "안전 원칙을 세 문장으로 설명하세요."
        self.assertFalse(response_contract_satisfied(request, "한 문장입니다."))
        self.assertTrue(
            response_contract_satisfied(
                request,
                "첫 문장입니다. 둘째 문장입니다. 셋째 문장입니다.",
            )
        )
        self.assertFalse(
            response_contract_satisfied(
                request,
                "첫 문장입니다. 둘째 문장입니다. 셋째 문장입니다. 초과 문장입니다.",
            )
        )
        self.assertEqual(requested_exact_sentence_count(request), 3)
        self.assertIsNone(
            requested_exact_sentence_count("안전 원칙을 세 문장 이상 설명하세요.")
        )
        self.assertIsNone(
            requested_exact_sentence_count("각 모듈을 각각 한 문장씩 설명하세요.")
        )
        self.assertTrue(
            response_contract_satisfied(
                "안전 원칙을 세 문장 이상 설명하세요.",
                "첫 문장입니다. 둘째 문장입니다. 셋째 문장입니다. 추가 문장입니다.",
            )
        )
        self.assertTrue(
            response_contract_satisfied(
                "네 항목 이내로 정리하세요.",
                "한 항목만 답합니다.",
            )
        )

    def test_exact_sentence_normalizer_preserves_requested_categories(self) -> None:
        from cogni_agent.response_quality import normalize_exact_sentence_response

        request = (
            "온디바이스 AI의 장점 두 가지와 한계 한 가지를 세 문장으로 설명하세요."
        )
        candidate = (
            "온디바이스 AI의 장점과 한계를 설명하겠습니다.\n"
            "1. 개인정보 보호: 데이터가 장치 밖으로 나가지 않아 노출을 줄입니다.\n"
            "2. 빠른 응답: 네트워크 왕복 없이 결과를 제공합니다.\n"
            "3. 오프라인 동작: 연결이 없어도 사용할 수 있습니다.\n"
            "한계:\n"
            "1. 제한된 자원: 장치 성능에 따라 실행 가능한 모델이 제한됩니다.\n"
            "2. 유지 관리: 업데이트 비용이 커질 수 있습니다.\n"
            "3."
        )
        normalized = normalize_exact_sentence_response(request, candidate)
        self.assertEqual(
            normalized,
            "개인정보 보호: 데이터가 장치 밖으로 나가지 않아 노출을 줄입니다. "
            "빠른 응답: 네트워크 왕복 없이 결과를 제공합니다. "
            "제한된 자원: 장치 성능에 따라 실행 가능한 모델이 제한됩니다.",
        )

    def test_exact_sentence_normalizer_does_not_invent_missing_category(self) -> None:
        from cogni_agent.response_quality import normalize_exact_sentence_response

        request = "장점 두 가지와 한계 한 가지를 세 문장으로 설명하세요."
        candidate = "첫 장점입니다. 둘째 장점입니다. 셋째 장점입니다."
        self.assertIsNone(normalize_exact_sentence_response(request, candidate))

    def test_exact_sentence_normalizer_splits_one_complete_korean_coordinate(
        self,
    ) -> None:
        from cogni_agent.response_quality import normalize_exact_sentence_response

        normalized = normalize_exact_sentence_response(
            "확인된 사실과 추론을 두 문장으로 구분하세요.",
            "사실과 추론을 구분하고, 추론은 명확한 근거를 제시해야 합니다.",
        )
        self.assertEqual(
            normalized,
            "사실과 추론을 구분합니다. 추론은 명확한 근거를 제시해야 합니다.",
        )

    def test_code_content_is_not_classified_as_prose_repetition(self) -> None:
        text = "```text\n네 네 네 네 네 네\n```"

        report = inspect_response(text, final=True)

        self.assertFalse(report.has(QualityCode.LOW_INFORMATION_REPETITION))
        self.assertFalse(report.has(QualityCode.TEMPLATE_REPETITION))

    def test_oversized_input_inspects_both_ends_with_fixed_work_bound(self) -> None:
        text = "x" * 9_000 + "<|eot_id|>"

        report = inspect_response(text)

        self.assertTrue(report.input_truncated)
        self.assertEqual(report.inspected_characters, MAX_INSPECT_CHARS)
        self.assertEqual(report.input_characters, len(text))
        self.assertTrue(report.has(QualityCode.CONTROL_TOKEN))
        self.assertEqual(report.recommended_cut_index, text.index("<|eot_id|>"))

    def test_reports_are_deterministic_and_input_type_is_checked(self) -> None:
        text = "첫 번째 결과는 매우 빠릅니다. 두 번째 결과는 정말 안전합니다."

        self.assertEqual(inspect_response(text), inspect_response(text))
        with self.assertRaises(TypeError):
            inspect_response(123)  # type: ignore[arg-type]


if __name__ == "__main__":
    unittest.main()
