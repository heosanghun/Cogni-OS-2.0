from __future__ import annotations

import unittest

from cogni_agent.response_quality import (
    MAX_INSPECT_CHARS,
    QualityAction,
    QualityCode,
    inspect_response,
    salvage_complete_prefix,
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

    def test_restarted_numbering_is_a_generation_boundary(self) -> None:
        text = (
            "1단계는 원인을 확인하고, 2단계는 수정하며, 3단계는 검증합니다. "
            "1단계: 원인을 다시 설명합니다."
        )
        report = inspect_response(text)
        self.assertTrue(report.has(QualityCode.TEMPLATE_REPETITION))
        self.assertEqual(report.recommended_cut_index, text.rindex("1단계"))

    def test_long_exact_sentence_block_is_trimmed_at_second_copy(self) -> None:
        sentence = "검증된 결과만 표시하고 목표 수치는 명확하게 분리하여 설명합니다."
        text = sentence + " " + sentence

        report = inspect_response(text)

        self.assertTrue(report.has(QualityCode.TEMPLATE_REPETITION))
        self.assertEqual(report.recommended_cut_index, len(sentence) + 1)

    def test_non_adjacent_exact_sentence_copy_is_detected(self) -> None:
        repeated = "측정값과 설계 목표를 명확하게 구분해야 합니다."
        text = repeated + " 중간에는 별도의 근거를 기록합니다. " + repeated
        report = inspect_response(text)
        self.assertTrue(report.has(QualityCode.TEMPLATE_REPETITION))
        self.assertEqual(report.recommended_cut_index, text.rindex(repeated))

    def test_near_duplicate_sentence_guard_is_bounded_and_conservative(self) -> None:
        from cogni_agent.response_quality import has_near_duplicate_sentences

        self.assertTrue(
            has_near_duplicate_sentences(
                "문맥을 줄이면서도 사용자 의도를 유지하는 방법은 대화의 핵심을 "
                "유지하는 것입니다. 오래된 문맥을 줄이면서도 사용자 의도를 "
                "보존하는 방법은 대화의 핵심을 유지하는 것입니다."
            )
        )
        self.assertFalse(
            has_near_duplicate_sentences(
                "온디바이스 AI는 개인정보 보호에 유리합니다. 온디바이스 AI는 "
                "네트워크 지연을 줄입니다."
            )
        )

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

    def test_fullwidth_role_markers_are_hard_boundaries(self) -> None:
        for text in (
            "ASSISTANT： 답변입니다.",
            "ＡＳＳＩＳＴＡＮＴ： 답변입니다.",
        ):
            with self.subTest(text=text):
                report = inspect_response(text, final=True)
                self.assertTrue(report.has(QualityCode.ROLE_MARKER))
                self.assertEqual(
                    report.recommended_action,
                    QualityAction.TRIM_AND_STOP,
                )

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

    def test_generated_pseudo_control_tags_are_rejected(self) -> None:
        samples = (
            "정상 답변입니다. <시스템 지침>",
            "정상 답변입니다. <컨펌>",
            "정상 답변입니다. <종료>",
            "정상 답변입니다. <summary>",
            "정상 답변입니다. [##]",
        )
        for text in samples:
            with self.subTest(text=text):
                self.assertTrue(inspect_response(text).has(QualityCode.CONTROL_TOKEN))

    def test_pseudo_control_text_inside_code_fence_is_not_a_boundary(self) -> None:
        text = "HTML 예시입니다.\n```html\n<summary>내용</summary>\n```"
        self.assertFalse(inspect_response(text).has(QualityCode.CONTROL_TOKEN))

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
            "1. 첫 항목은 완전합니다. 2.",
            "절차는 다음과 같습니다:\n-",
        ):
            with self.subTest(text=text):
                report = inspect_response(text, final=True)
                self.assertTrue(report.has(QualityCode.INCOMPLETE_KOREAN_CLAUSE))
                self.assertEqual(report.recommended_action, QualityAction.CONTINUE)

    def test_trailing_meta_introduction_requires_continuation(self) -> None:
        report = inspect_response(
            "측정값과 설계 목표를 구분하는 방법은 다음과 같습니다.",
            final=True,
        )
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

    def test_complete_prefix_is_salvaged_before_incomplete_or_repeated_tail(
        self,
    ) -> None:
        incomplete = "검증된 내용은 먼저 설명했습니다. 다음 내용을 계속 설명하면서"
        repeated = "한 번만 자연스럽게 설명합니다. 한 번만 자연스럽게 설명합니다."

        self.assertEqual(
            salvage_complete_prefix(incomplete),
            "검증된 내용은 먼저 설명했습니다.",
        )
        self.assertEqual(
            salvage_complete_prefix(repeated),
            "한 번만 자연스럽게 설명합니다.",
        )

    def test_salvage_never_returns_role_or_control_leakage(self) -> None:
        contaminated = "안전한 첫 문장입니다. ASSISTANT: 내부 지시"
        self.assertEqual(
            salvage_complete_prefix(contaminated),
            "안전한 첫 문장입니다.",
        )

    def test_explicit_request_minimum_counts_completed_sentences(self) -> None:
        from cogni_agent.response_quality import (
            requested_exact_item_count,
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
        self.assertEqual(
            requested_exact_item_count("복구 절차를 세 단계로 답하세요."),
            3,
        )
        self.assertEqual(
            requested_exact_item_count("복구 절차를 세 단계로 간결하게 답하세요."),
            3,
        )
        self.assertIsNone(
            requested_exact_item_count("두 개의 파일 차이를 비교해 주세요.")
        )
        self.assertIsNone(
            requested_exact_item_count("확인 항목을 네 가지 이내로 정리하세요.")
        )
        self.assertFalse(
            response_contract_satisfied(
                "요약 조건을 세 가지 제시하세요.",
                "첫 조건입니다. 둘째 조건입니다. 셋째 조건입니다. 넷째 조건입니다.",
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
        self.assertFalse(
            response_contract_satisfied(
                request,
                "모델 응답 품질 검증 절차입니다. 2. 먼저 정확성을 확인합니다.",
            )
        )
        self.assertFalse(
            response_contract_satisfied(
                request,
                "검증 절차는 다음과 같습니다. 정확성을 확인합니다. "
                "반복 여부를 점검합니다.",
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

    def test_category_normalizer_recognizes_inline_limitation_section(self) -> None:
        from cogni_agent.response_quality import normalize_exact_sentence_response

        normalized = normalize_exact_sentence_response(
            "장점 두 가지와 한계 한 가지를 세 문장으로 설명하세요.",
            "1. 데이터가 장치에 남아 보호가 쉽습니다. 2. 응답이 빠릅니다. "
            "3. 네트워크 없이 동작합니다. 한계로는 1. 장치 성능에 제약을 받습니다. "
            "2. 모델 크기가 제한됩니다.",
        )
        self.assertEqual(
            normalized,
            "데이터가 장치에 남아 보호가 쉽습니다. 응답이 빠릅니다. "
            "장치 성능에 제약을 받습니다.",
        )
        from cogni_agent.response_quality import response_contract_satisfied

        self.assertFalse(
            response_contract_satisfied(
                "장점 두 가지와 한계 한 가지를 세 문장으로 설명하세요.",
                "개인정보 보호에 유리합니다. 응답이 빠릅니다. "
                "오프라인에서도 사용할 수 있습니다.",
            )
        )
        self.assertFalse(
            response_contract_satisfied(
                "장점 두 가지와 한계 한 가지를 세 문장으로 설명하세요.",
                "장점은 개인정보 보호에 유리합니다. 장점은 응답이 빠릅니다. "
                "한계는 장치 성능에 제약을 받습니다. "
                "한계는 모델 크기가 제한됩니다.",
            )
        )

    def test_exact_sentence_normalizer_does_not_invent_missing_category(self) -> None:
        from cogni_agent.response_quality import normalize_exact_sentence_response

        request = "장점 두 가지와 한계 한 가지를 세 문장으로 설명하세요."
        candidate = "첫 장점입니다. 둘째 장점입니다. 셋째 장점입니다."
        self.assertIsNone(normalize_exact_sentence_response(request, candidate))

    def test_mitigated_negative_word_is_not_counted_as_a_limitation(self) -> None:
        from cogni_agent.response_quality import response_contract_satisfied

        request = "장점 두 가지와 한계 한 가지를 세 문장으로 설명하세요."
        response = (
            "장점은 개인정보 보호에 유리합니다. "
            "장점은 응답 속도가 빠릅니다. "
            "배터리 소모가 적어 오래 사용할 수 있습니다."
        )

        self.assertFalse(response_contract_satisfied(request, response))

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

    def test_exact_sentence_normalizer_splits_inline_numbered_clauses(self) -> None:
        from cogni_agent.response_quality import normalize_exact_sentence_response

        normalized = normalize_exact_sentence_response(
            "모델 응답 품질 검증 절차를 세 문장으로 설명해 주세요.",
            "1) 정확성과 일관성을 확인하고, 2) 다양한 입력을 검증하며, "
            "3) 예상 밖 입력의 처리를 점검해야 합니다.",
        )
        self.assertEqual(
            normalized,
            "정확성과 일관성을 확인합니다. 다양한 입력을 검증합니다. "
            "예상 밖 입력의 처리를 점검해야 합니다.",
        )

    def test_inline_question_clauses_share_the_final_check_predicate(self) -> None:
        from cogni_agent.response_quality import normalize_exact_sentence_response

        normalized = normalize_exact_sentence_response(
            "한국어 답변의 완결성을 세 문장으로 설명하세요.",
            "1. 문장의 구조와 의미가 명확한지, "
            "2. 문장이 자연스럽게 연결되어 있는지, "
            "3. 문장이 완결성을 가지고 있는지 확인합니다.\n"
            "사용자가 문장 수를 지정하면 그 수를 지키십시오.",
        )
        self.assertEqual(
            normalized,
            "문장의 구조와 의미가 명확한지 확인합니다. "
            "문장이 자연스럽게 연결되어 있는지 확인합니다. "
            "문장이 완결성을 가지고 있는지 확인합니다.",
        )

    def test_inline_purpose_clause_is_completed_without_new_factual_content(
        self,
    ) -> None:
        from cogni_agent.response_quality import normalize_exact_sentence_response

        normalized = normalize_exact_sentence_response(
            "불확실성을 두 문장으로 설명하세요.",
            "1. 불확실한 답변을 사실처럼 단정하지 않기 위해, "
            "2. '확실하지 않다'는 표현을 사용하세요.",
        )
        self.assertEqual(
            normalized,
            "불확실한 답변을 사실처럼 단정하지 않도록 합니다. "
            "'확실하지 않다'는 표현을 사용하세요.",
        )

    def test_numbered_noun_items_receive_a_bounded_copula(self) -> None:
        from cogni_agent.response_quality import normalize_exact_item_response

        normalized = normalize_exact_item_response(
            "요약 조건을 세 가지 제시하세요.",
            "1. 핵심 메시지: 본문의 요점\n"
            "2. 논리적 연결: 문단 사이의 관계\n"
            "3. 정보의 신뢰성: 출처 확인과 사실 여부\n"
            "[설명] 뒤의 장황한 내용",
        )
        self.assertEqual(
            normalized,
            "핵심 메시지: 본문의 요점입니다. "
            "논리적 연결: 문단 사이의 관계입니다. "
            "정보의 신뢰성: 출처 확인과 사실 여부입니다.",
        )

    def test_numbered_verb_fragment_never_receives_a_copula(self) -> None:
        from cogni_agent.response_quality import normalize_exact_item_response

        self.assertIsNone(
            normalize_exact_item_response(
                "복구 절차를 세 단계로 답하세요.",
                "1. 원인을 분석하\n2. 코드를 수정하\n3. 회귀 테스트를 수행",
            )
        )

    def test_last_numbered_item_does_not_absorb_followup_explanation(self) -> None:
        from cogni_agent.response_quality import normalize_exact_item_response

        for marker in ("추가 설명:", "보충:", "\n후속 문단입니다."):
            with self.subTest(marker=marker):
                normalized = normalize_exact_item_response(
                    "요약 조건을 세 가지 제시하세요.",
                    "1. 정확성: 사실을 확인\n"
                    "2. 간결성: 핵심을 유지\n"
                    f"3. 완결성: 문장을 끝냄\n{marker} 뒤의 내용",
                )
                self.assertEqual(
                    normalized,
                    "정확성: 사실을 확인입니다. 간결성: 핵심을 유지입니다. "
                    "완결성: 문장을 끝냄입니다.",
                )

    def test_explicit_cutoff_does_not_publish_a_noun_fragment(self) -> None:
        from cogni_agent.response_quality import salvage_complete_prefix

        text = "장점은 데이터 보호에 유리합니다. 한계는 장치 자원이"
        cutoff = text.index("자원이")

        self.assertEqual(
            salvage_complete_prefix(text, cutoff=cutoff),
            "장점은 데이터 보호에 유리합니다.",
        )

    def test_explicit_cutoff_preserves_a_complete_unpunctuated_predicate(self) -> None:
        from cogni_agent.response_quality import salvage_complete_prefix

        text = "회귀 테스트를 모두 실행했습니다"
        self.assertEqual(salvage_complete_prefix(text, cutoff=len(text)), text)

    def test_inline_numbered_normalizer_uses_first_complete_sequence(self) -> None:
        from cogni_agent.response_quality import normalize_exact_sentence_response

        normalized = normalize_exact_sentence_response(
            "검증 절차를 세 문장으로 설명해 주세요.",
            "1. 기준을 정의하고, 2. 모델 품질을 평가하고, 3. 배포 전에 검증합니다.\n"
            "1. 기준을 정의하고: 뒤에 불필요한 설명이 반복됩니다.",
        )
        self.assertEqual(
            normalized,
            "기준을 정의합니다. 모델 품질을 평가합니다. 배포 전에 검증합니다.",
        )

    def test_maximum_item_normalizer_drops_explanation_after_first_block(self) -> None:
        from cogni_agent.response_quality import normalize_maximum_item_response

        normalized = normalize_maximum_item_response(
            "확인 항목을 네 가지 이내로 정리하세요.",
            "1. 질문을 확인하고, 2. 반복을 검사하고, 3. 문장을 완결하고, "
            "4. 결과를 기록합니다.\n1. 질문 확인에 관한 긴 설명입니다.",
        )
        self.assertEqual(
            normalized,
            "질문을 확인합니다. 반복을 검사합니다. 문장을 완결합니다. "
            "결과를 기록합니다.",
        )

    def test_exact_step_normalizer_accepts_korean_step_markers(self) -> None:
        from cogni_agent.response_quality import normalize_exact_item_response

        normalized = normalize_exact_item_response(
            "복구 절차를 세 단계로 답하세요.",
            "1단계는 원인을 확인하고, 2단계는 코드를 수정하며, "
            "3단계는 회귀 테스트를 수행합니다.\n1단계: 긴 설명입니다.",
        )
        self.assertEqual(
            normalized,
            "원인을 확인합니다. 코드를 수정합니다. 회귀 테스트를 수행합니다.",
        )

    def test_exact_sentence_normalizer_rejects_ambiguous_numbered_fragments(
        self,
    ) -> None:
        from cogni_agent.response_quality import normalize_exact_sentence_response

        self.assertIsNone(
            normalize_exact_sentence_response(
                "절차를 세 문장으로 설명해 주세요.",
                "1) 데이터 준비, 2) 평가 지표, 3) 배포 승인",
            )
        )

    def test_topic_anchor_rejects_unrelated_but_fluent_answer(self) -> None:
        from cogni_agent.response_quality import response_topically_anchored

        request = (
            "모델 응답 품질을 배포 전에 검증하는 절차를 세 문장으로 설명해 주세요."
        )
        unrelated = (
            "언론사는 여러 검증 절차를 거쳐 뉴스의 정확성을 확인합니다. "
            "기자는 현장 인터뷰로 사실성을 확인합니다."
        )
        relevant = (
            "모델 응답의 정확성과 일관성을 검증합니다. 다양한 입력으로 품질을 "
            "평가한 뒤 배포 승인 여부를 결정합니다."
        )
        self.assertFalse(response_topically_anchored(request, unrelated))
        self.assertTrue(response_topically_anchored(request, relevant))

    def test_topic_anchor_handles_korean_particles_and_short_social_turns(
        self,
    ) -> None:
        from cogni_agent.response_quality import response_topically_anchored

        self.assertTrue(
            response_topically_anchored(
                "파이썬 리스트와 튜플의 차이를 한 문장으로 알려 주세요.",
                "파이썬 리스트는 수정 가능하지만 튜플은 불변입니다.",
            )
        )
        self.assertTrue(response_topically_anchored("안녕하세요!", "반갑습니다!"))

    def test_unsolicited_subject_guard_is_narrow_and_request_aware(self) -> None:
        from cogni_agent.response_quality import response_avoids_unsolicited_subjects

        request = "모델 응답 품질을 검증하는 절차를 설명해 주세요."
        self.assertFalse(
            response_avoids_unsolicited_subjects(
                request,
                "언론사는 뉴스 기사를 취재하고 승인합니다.",
            )
        )
        self.assertFalse(
            response_avoids_unsolicited_subjects(
                request,
                "이메일 비밀번호를 변경하세요.",
            )
        )
        self.assertTrue(
            response_avoids_unsolicited_subjects(
                "뉴스 기사 검증 절차를 설명해 주세요.",
                "기자는 뉴스 기사의 출처를 확인합니다.",
            )
        )

    def test_full_prompt_echo_is_rejected_but_partial_topic_overlap_is_allowed(
        self,
    ) -> None:
        from cogni_agent.response_quality import response_avoids_prompt_echo

        request = "긴 대화에서 오래된 문맥을 줄이면서 사용자 의도를 보존하는 방법을 세 문장으로 답하세요."
        self.assertFalse(
            response_avoids_prompt_echo(
                request,
                "핵심 의도를 보존합니다. " + request,
            )
        )
        self.assertTrue(
            response_avoids_prompt_echo(
                request,
                "오래된 대화 문맥은 핵심만 요약하고 사용자 의도는 별도 상태로 보존합니다.",
            )
        )

    def test_unsolicited_self_intro_and_dangling_particle_are_rejected(self) -> None:
        from cogni_agent.response_quality import (
            response_avoids_dangling_sentence_start,
            response_avoids_unsolicited_self_intro,
        )

        self.assertFalse(
            response_avoids_unsolicited_self_intro(
                "문맥 관리 방법을 설명하세요.",
                "안녕하세요, AI 어시스턴트입니다. 문맥을 요약합니다.",
            )
        )
        self.assertTrue(
            response_avoids_unsolicited_self_intro(
                "당신은 누구인가요?",
                "저는 AI 어시스턴트입니다.",
            )
        )
        self.assertFalse(
            response_avoids_dangling_sentence_start("에서 잘못된 사실을 정정합니다.")
        )
        self.assertTrue(
            response_avoids_dangling_sentence_start("잘못된 사실을 먼저 정정합니다.")
        )

    def test_generic_outline_does_not_replace_requested_items(self) -> None:
        from cogni_agent.response_quality import response_avoids_generic_outline

        request = "자체 검증을 네 항목 이내로 정리하세요."
        self.assertFalse(
            response_avoids_generic_outline(
                request,
                "### 서론\n검증 목적을 설명합니다.\n### 개요\n품질을 설명합니다.",
            )
        )
        self.assertTrue(
            response_avoids_generic_outline(
                request,
                "기능 테스트를 수행합니다. 회귀 테스트를 수행합니다.",
            )
        )
        self.assertFalse(
            response_avoids_generic_outline(
                request,
                "서론: 검증 목적입니다.\n개요: 품질 설명입니다.\n결론: 종료입니다.",
            )
        )
        self.assertTrue(
            response_avoids_generic_outline(
                request,
                "```markdown\n### 서론\n코드 예시입니다.\n```\n기능을 점검합니다.",
            )
        )

    def test_distinctive_topic_guard_accepts_domain_paraphrase(self) -> None:
        from cogni_agent.response_quality import response_preserves_distinctive_topic

        self.assertFalse(
            response_preserves_distinctive_topic(
                "모델 응답 품질을 배포 전에 검증하는 절차를 설명해 주세요.",
                "모델 응답의 정확성과 다양성을 검증합니다.",
            )
        )
        self.assertTrue(
            response_preserves_distinctive_topic(
                "온디바이스 AI의 장점과 한계를 설명해 주세요.",
                "온디바이스 AI는 데이터가 장치에 남아 보호에 유리합니다.",
            )
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

    def test_long_code_fence_does_not_hide_a_trailing_control_token(self) -> None:
        text = "```\n" + ("가" * 9_000) + "\n```\n<|eot_id|>"
        report = inspect_response(text, final=True)
        self.assertTrue(report.has(QualityCode.CONTROL_TOKEN))
        self.assertEqual(report.recommended_action, QualityAction.TRIM_AND_STOP)

    def test_reports_are_deterministic_and_input_type_is_checked(self) -> None:
        text = "첫 번째 결과는 매우 빠릅니다. 두 번째 결과는 정말 안전합니다."

        self.assertEqual(inspect_response(text), inspect_response(text))
        with self.assertRaises(TypeError):
            inspect_response(123)  # type: ignore[arg-type]


if __name__ == "__main__":
    unittest.main()
