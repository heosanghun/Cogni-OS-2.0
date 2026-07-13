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

        expanded_prefix = "OK.\n㍿\nＡＳＳＩＳＴＡＮＴ： hidden"
        expanded_report = inspect_response(expanded_prefix, final=True)
        self.assertTrue(expanded_report.has(QualityCode.ROLE_MARKER))
        self.assertEqual(
            expanded_report.recommended_cut_index,
            expanded_prefix.index("Ａ"),
        )
        nfd_role = "정상.\n사용자: hidden"
        nfd_report = inspect_response(nfd_role, final=True)
        self.assertTrue(nfd_report.has(QualityCode.ROLE_MARKER))
        self.assertEqual(nfd_report.recommended_cut_index, nfd_role.index("ᄉ"))

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

        mixed_width = "OK. ＜컨펌＞hidden\n<컨펌>later"
        mixed_report = inspect_response(mixed_width, final=True)
        self.assertTrue(mixed_report.has(QualityCode.CONTROL_TOKEN))
        self.assertEqual(
            mixed_report.recommended_cut_index,
            mixed_width.index("＜"),
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
            "정상 답변입니다. <답변>오염</답변>",
        )
        for text in samples:
            with self.subTest(text=text):
                self.assertTrue(inspect_response(text).has(QualityCode.CONTROL_TOKEN))

    def test_maximum_item_contract_enforces_only_the_requested_upper_bound(
        self,
    ) -> None:
        from cogni_agent.response_quality import (
            requested_maximum_items,
            requested_minimum_units,
            response_contract_satisfied,
        )

        request = "자체 검증을 네 항목 이내로 정리하세요."
        self.assertTrue(response_contract_satisfied(request, "한 문장입니다."))
        self.assertTrue(
            response_contract_satisfied(
                request,
                "기능을 점검합니다. 회귀 테스트를 실행합니다.",
            )
        )
        self.assertFalse(
            response_contract_satisfied(
                request,
                "첫째입니다. 둘째입니다. 셋째입니다. 넷째입니다. 다섯째입니다.",
            )
        )
        for wording in (
            "최대 네 항목으로 답하세요.",
            "확인 결과를 4개 이하로 정리하세요.",
            "핵심을 네 문장 이내로 설명하세요.",
            "핵심을 최대 10개로 제시하세요.",
            "핵심을 10문장 이내로 작성하세요.",
        ):
            with self.subTest(wording=wording):
                self.assertIsNotNone(requested_maximum_items(wording))
                self.assertIsNone(requested_minimum_units(wording))
        self.assertIsNone(requested_maximum_items("오류가 네 개 이내인지 확인하세요."))

    def test_pseudo_control_text_inside_code_fence_is_not_a_boundary(self) -> None:
        text = "HTML 예시입니다.\n```html\n<summary>내용</summary>\n```"
        self.assertFalse(inspect_response(text).has(QualityCode.CONTROL_TOKEN))

    def test_unclosed_code_fence_is_never_a_final_answer(self) -> None:
        text = "코드 예시입니다.\n```text\nASSISTANT: 내부 지시"
        streaming = inspect_response(text, final=False)
        final = inspect_response(text, final=True)

        self.assertEqual(streaming.recommended_action, QualityAction.ACCEPT)
        self.assertTrue(final.has(QualityCode.INCOMPLETE_KOREAN_CLAUSE))
        self.assertEqual(final.recommended_action, QualityAction.CONTINUE)

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

    def test_incomplete_colon_item_does_not_receive_a_copula(self) -> None:
        from cogni_agent.response_quality import normalize_maximum_item_response

        normalized = normalize_maximum_item_response(
            "자체 검증을 네 항목 이내로 정리하세요.",
            "1. 목적 정의: 변경 범위를 명확히 정의합니다.\n"
            "2. 코드 검증: 회귀 테스트를 실행합니다.\n"
            "3. 문서화: 코드가 왜 변경",
        )
        self.assertEqual(
            normalized,
            "목적 정의: 변경 범위를 명확히 정의합니다. "
            "코드 검증: 회귀 테스트를 실행합니다.",
        )

        for fragment in (
            "결과 기록: 로그에 저장",
            "문서화: 변경된 이유를 설명",
            "배포 준비: 결과를 제공",
        ):
            with self.subTest(fragment=fragment):
                self.assertIsNone(
                    normalize_maximum_item_response(
                        "자체 검증을 한 항목 이내로 정리하세요.",
                        f"1. {fragment}",
                    )
                )

    def test_last_numbered_item_does_not_absorb_followup_explanation(self) -> None:
        from cogni_agent.response_quality import normalize_exact_item_response

        for marker in ("추가 설명:", "보충:", "\n후속 문단입니다."):
            with self.subTest(marker=marker):
                normalized = normalize_exact_item_response(
                    "요약 조건을 세 가지 제시하세요.",
                    "1. 정확성: 확인된 사실\n"
                    "2. 간결성: 핵심 정보\n"
                    f"3. 완결성: 자연스러운 마무리\n{marker} 뒤의 내용",
                )
                self.assertEqual(
                    normalized,
                    "정확성: 확인된 사실입니다. 간결성: 핵심 정보입니다. "
                    "완결성: 자연스러운 마무리입니다.",
                )

        normalized = normalize_exact_item_response(
            "요약 조건을 세 가지 제시하세요.",
            "1. 정확성: 확인된 사실\n"
            "2. 간결성: 핵심 정보\n"
            "3. 완결성: 자연스러운 마무리\n"
            "후속 문단입니다.",
        )
        self.assertEqual(
            normalized,
            "정확성: 확인된 사실입니다. 간결성: 핵심 정보입니다. "
            "완결성: 자연스러운 마무리입니다.",
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
        for status in (
            "회귀 테스트 완료",
            "자체 검증 PASS",
            "검증 성공",
            "Regression tests passed",
            "**회귀 테스트를 모두 실행했습니다**",
        ):
            with self.subTest(status=status):
                self.assertEqual(
                    salvage_complete_prefix(status, cutoff=len(status)),
                    status,
                )

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

        missing_space = normalize_maximum_item_response(
            "자체 검증을 네 항목 이내로 정리하세요.",
            "1. 예상 결과와 일치하는지 확인합니다. "
            "2. 테스트 환경을 점검합니다. "
            "3. 사용자 테스트를 수행합니다. "
            "4. 오류가 발생하는지 확인합니다.이러한 검증으로 마무리합니다.\n"
            "파란색 지침은 시스템 메시지입니다.",
        )
        self.assertEqual(
            missing_space,
            "예상 결과와 일치하는지 확인합니다. 테스트 환경을 점검합니다. "
            "사용자 테스트를 수행합니다. 오류가 발생하는지 확인합니다.",
        )

        live_candidate = normalize_maximum_item_response(
            "소프트웨어 수정 완료를 선언하기 전에 필요한 자체 검증을 네 항목 이내로 정리하세요.",
            "1. 예상 문제를 고려하고 실제 결과와 일치하는지 확인하십시오. "
            "2. 테스트 환경에서 가능한 한 최선의 환경을 선택하십시오. "
            "3. 사용자 테스트를 수행하여 기능을 확인하고 보완할 수 있는 피드백을 얻으십시오. "
            "4. 코드에서 오류나 오버플로우가 발생하는지 확인하십시오.이러한 검증을 완료하면 "
            "완료할 수 있다고 선언할 수 있습니다.\n"
            "파란색으로 표시된 지침은 시스템에 대한 메시지입니다.",
        )
        self.assertEqual(
            live_candidate,
            "예상 문제를 고려하고 실제 결과와 일치하는지 확인하십시오. "
            "테스트 환경에서 가능한 한 최선의 환경을 선택하십시오. "
            "사용자 테스트를 수행하여 기능을 확인하고 보완할 수 있는 피드백을 얻으십시오. "
            "코드에서 오류나 오버플로우가 발생하는지 확인하십시오.",
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
                "보고서의 서론, 개요, 결론을 세 문장으로 설명하세요.",
                "서론: 배경을 설명합니다.\n개요: 핵심을 정리합니다.\n"
                "결론: 결과를 요약합니다.",
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
                "자체 검증을 세 문장으로 정리하세요.",
                "서론: 자체 검증의 배경을 정리합니다. "
                "개요: 자체 검증의 기능 단계입니다. "
                "결론: 자체 검증 결과를 확인합니다.",
            )
        )
        self.assertFalse(
            response_avoids_generic_outline(
                "자체 검증을 세 항목으로 정리하세요.",
                "표기 ``` 를 설명합니다.\n### 개요\n"
                "자체 검증 기능을 점검합니다. 회귀 테스트 결과를 확인합니다.\n```",
            )
        )

    def test_meta_format_discussion_is_not_an_answer(self) -> None:
        from cogni_agent.response_quality import (
            response_avoids_generic_outline,
            response_avoids_meta_format_discussion,
        )

        request = "확인 항목을 네 가지 이내로 정리하세요."
        response = (
            "질문의 형식을 다시 정리합니다. 사용자는 항목 수를 네 개로 제한하라고 "
            "요청합니다. 이는 형식으로 해석됩니다."
        )
        self.assertFalse(response_avoids_meta_format_discussion(request, response))
        self.assertTrue(
            response_avoids_meta_format_discussion(
                "질문의 형식을 설명하세요.",
                response,
            )
        )
        self.assertTrue(
            response_avoids_meta_format_discussion(
                request,
                "사용자는 잘못된 사실의 수정을 요청합니다.",
            )
        )
        self.assertTrue(
            response_avoids_meta_format_discussion(
                "출력 형식을 정리하세요.",
                "질문의 형식을 정리합니다.",
            )
        )
        self.assertFalse(
            response_avoids_meta_format_discussion(
                "답변 형식을 지키며 확인 항목을 네 가지 이내로 정리하세요.",
                "질문의 형식을 정리합니다.",
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
        from cogni_agent.response_quality import (
            response_preserves_distinctive_topic,
            response_topically_anchored,
        )

        self.assertFalse(
            response_preserves_distinctive_topic(
                "모델 응답 품질을 배포 전에 검증하는 절차를 설명해 주세요.",
                "모델 응답의 정확성과 다양성을 검증합니다.",
            )
        )
        self.assertFalse(
            response_preserves_distinctive_topic(
                "투자가 검증할 재무 지표와 수익 조건을 설명하세요.",
                "코드 오류를 점검하고 기능 결과를 확인합니다.",
            )
        )
        self.assertTrue(
            response_preserves_distinctive_topic(
                "온디바이스 AI의 장점과 한계를 설명해 주세요.",
                "온디바이스 AI는 데이터가 장치에 남아 보호에 유리합니다.",
            )
        )
        self.assertTrue(
            response_topically_anchored(
                "소프트웨어 수정 완료 전에 필요한 자체 검증을 정리하세요.",
                "실제 결과를 확인하고 테스트 환경과 코드 오류를 점검합니다.",
            )
        )
        self.assertFalse(
            response_topically_anchored(
                "이 제품 자체의 배터리 수명과 충전 시간을 설명하세요.",
                "코드 오류를 점검하고 기능 결과를 확인합니다.",
            )
        )
        self.assertFalse(
            response_topically_anchored(
                "email 보안 정책과 로그인 절차를 설명하세요.",
                "로컬 데이터와 장치 오프라인 처리를 사용합니다.",
            )
        )
        self.assertFalse(
            response_preserves_distinctive_topic(
                "자체 검증을 세 항목으로 정리하세요.",
                "음식을 준비합니다. 물을 끓입니다. 그릇에 담습니다.",
            )
        )
        self.assertFalse(
            response_preserves_distinctive_topic(
                "자체 검증을 세 항목으로 정리하세요.",
                "음식 조리 기능을 준비합니다. 조리 결과를 확인합니다. "
                "그릇에 음식을 담습니다.",
            )
        )
        self.assertFalse(
            response_preserves_distinctive_topic(
                "자체 검증을 세 항목으로 정리하세요.",
                "배터리 기능을 확인합니다. 충전 결과를 점검합니다. 전력을 측정합니다.",
            )
        )
        self.assertTrue(
            response_preserves_distinctive_topic(
                "자체 검증을 세 항목으로 정리하세요.",
                "수정된 기능이 예상대로 작동하는지 확인합니다. "
                "기존 기능과 충돌하지 않는지 확인합니다. "
                "성능과 보안 영향을 점검합니다.",
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

        for token in ("<|eot_id|>", "ASSISTANT:", "[INST]"):
            with self.subTest(token=token):
                split = 3
                filler = 4_096 + split - len(token)
                prefix = ("a" * 4_999) + "\n" if token == "ASSISTANT:" else "a" * 5_000
                seam_text = prefix + token + ("x" * filler)
                seam_report = inspect_response(seam_text, final=True)
                expected = (
                    QualityCode.ROLE_MARKER
                    if token == "ASSISTANT:"
                    else QualityCode.CONTROL_TOKEN
                )
                self.assertTrue(seam_report.has(expected))

    def test_long_code_fence_does_not_hide_a_trailing_control_token(self) -> None:
        text = "```\n" + ("가" * 9_000) + "\n```\n<|eot_id|>"
        report = inspect_response(text, final=True)
        self.assertTrue(report.has(QualityCode.CONTROL_TOKEN))
        self.assertEqual(report.recommended_action, QualityAction.TRIM_AND_STOP)

        split_closing_fence = (
            "```\n" + ("a" * 4_999) + "\n```\n" + ("x" * 4_083) + "<|eot_id|>"
        )
        split_report = inspect_response(split_closing_fence, final=True)
        self.assertTrue(split_report.has(QualityCode.CONTROL_TOKEN))
        self.assertEqual(
            split_report.recommended_action,
            QualityAction.TRIM_AND_STOP,
        )

        longer_fence = (
            "literal ``` marker.\n````\ninside\n```\nmore\n````\n[INST]hidden[/INST]"
        )
        longer_report = inspect_response(longer_fence, final=True)
        self.assertTrue(longer_report.has(QualityCode.CONTROL_TOKEN))
        self.assertEqual(
            longer_report.recommended_action,
            QualityAction.TRIM_AND_STOP,
        )
        invalid_long_line = (
            ("a" * 5_000)
            + "\n```bad`"
            + ("x" * 2_000)
            + "<|eot_id|>"
            + ("y" * 3_000)
            + "\nend"
        )
        invalid_report = inspect_response(invalid_long_line, final=True)
        self.assertTrue(invalid_report.has(QualityCode.CONTROL_TOKEN))
        invalid_after_tail_start = (
            ("a" * 5_000)
            + "\n```bad"
            + ("x" * 1_000)
            + "`"
            + ("x" * 1_000)
            + "<|eot_id|>"
            + ("y" * 3_000)
            + "\nend"
        )
        after_tail_report = inspect_response(invalid_after_tail_start, final=True)
        self.assertTrue(after_tail_report.has(QualityCode.CONTROL_TOKEN))

        remote_closing_fence = "```\n" + ("a" * 20_000) + "\n```\n<|eot_id|>"
        remote_report = inspect_response(remote_closing_fence, final=True)
        self.assertTrue(remote_report.has(QualityCode.CONTROL_TOKEN))
        self.assertEqual(
            remote_report.recommended_action,
            QualityAction.TRIM_AND_STOP,
        )
        long_opener_info = (
            ("a" * 5_000)
            + "\n```"
            + ("x" * 2_000)
            + "\n"
            + ("c" * 1_000)
            + "\n```\n<|eot_id|>"
            + ("y" * 2_000)
        )
        long_info_report = inspect_response(long_opener_info, final=True)
        self.assertTrue(long_info_report.has(QualityCode.CONTROL_TOKEN))

    def test_multiline_last_item_preserves_its_continuation(self) -> None:
        from cogni_agent.response_quality import normalize_exact_item_response

        continued = normalize_exact_item_response(
            "검증 조건을 세 가지 제시하세요.",
            "1. 기능을 확인합니다.\n"
            "2. 회귀 테스트를 실행합니다.\n"
            "3. 결과를 확인하고\n로그에 기록합니다.",
        )
        self.assertEqual(
            continued,
            "기능을 확인합니다. 회귀 테스트를 실행합니다. "
            "결과를 확인하고 로그에 기록합니다.",
        )
        noun_continuation = normalize_exact_item_response(
            "검증 조건을 세 가지 제시하세요.",
            "1. 기능을 확인합니다.\n"
            "2. 회귀 테스트를 실행합니다.\n"
            "3. 완결성: 자연스러운\n마무리를 보장합니다.",
        )
        self.assertEqual(
            noun_continuation,
            "기능을 확인합니다. 회귀 테스트를 실행합니다. "
            "완결성: 자연스러운 마무리를 보장합니다.",
        )

    def test_reports_are_deterministic_and_input_type_is_checked(self) -> None:
        text = "첫 번째 결과는 매우 빠릅니다. 두 번째 결과는 정말 안전합니다."

        self.assertEqual(inspect_response(text), inspect_response(text))
        with self.assertRaises(TypeError):
            inspect_response(123)  # type: ignore[arg-type]

    def test_single_question_contract_rejects_explanation_and_multiple_questions(
        self,
    ) -> None:
        from cogni_agent.response_quality import (
            requested_exact_question_count,
            response_contract_satisfied,
        )

        request = "생각을 풀 수 있도록 질문 하나를 자연스럽게 해 주세요."
        self.assertEqual(requested_exact_question_count(request), 1)
        self.assertTrue(
            response_contract_satisfied(
                request,
                "어떤 사용자를 위한 프로젝트를 만들고 싶으신가요?",
            )
        )
        self.assertFalse(
            response_contract_satisfied(
                request,
                "먼저 목표를 정해 보세요. 어떤 프로젝트를 만들고 싶으신가요?",
            )
        )
        self.assertFalse(
            response_contract_satisfied(
                request,
                "어떤 분야인가요? 누구를 위한 것인가요?",
            )
        )

    def test_meta_count_announcements_are_not_counted_as_content(self) -> None:
        from cogni_agent.response_quality import (
            response_avoids_meta_format_discussion,
            response_contract_satisfied,
        )

        two_sentences = (
            "3가지로 요약할 수 있습니다. 첫째, 오류 횟수를 확인해 종료합니다."
        )
        self.assertFalse(
            response_avoids_meta_format_discussion(
                "안전 종료 기준을 두 문장으로 답하세요.",
                two_sentences,
            )
        )
        self.assertFalse(
            response_contract_satisfied(
                "안전 종료 기준을 두 문장으로 답하세요.",
                two_sentences,
            )
        )
        trailing_promise = (
            "회귀 테스트를 실행합니다. 사용자께서 요청하신 검증 사항을 "
            "최대 4개까지 작성하겠습니다."
        )
        self.assertFalse(
            response_avoids_meta_format_discussion(
                "자체 검증을 네 항목 이내로 정리하세요.",
                trailing_promise,
            )
        )
        self.assertFalse(
            response_avoids_meta_format_discussion(
                "모델 품질을 세 문장으로 설명하세요.",
                "정확성을 평가합니다. 완결성을 확인합니다. 사용자의 요청이 완벽히 "
                "충족되는 답변을 제출하세요: 서론과 맺음말 없이 정확히 3개의 문장.",
            )
        )

    def test_semantic_redundancy_detects_reordered_fact_but_not_distinct_checks(
        self,
    ) -> None:
        from cogni_agent.response_quality import has_semantic_redundancy

        self.assertTrue(
            has_semantic_redundancy(
                "개인정보를 안전하게 처리해야 합니다. "
                "요청을 오프라인 환경에서 처리할 때 개인정보를 보호해야 합니다. "
                "개인정보를 처리할 때 요청을 안전하게 처리해야 합니다."
            )
        )
        self.assertFalse(
            has_semantic_redundancy(
                "수정된 기능이 예상대로 작동하는지 확인합니다. "
                "기존 기능과 충돌하지 않는지 회귀 테스트를 실행합니다. "
                "성능과 메모리 상한을 측정합니다."
            )
        )

    def test_placeholder_scaffolding_and_example_intent_are_bounded(self) -> None:
        from cogni_agent.response_quality import (
            response_avoids_placeholder_scaffolding,
        )

        self.assertFalse(
            response_avoids_placeholder_scaffolding(
                "[사용자 상황 파악]\n어떤 기능이 필요한가요?"
            )
        )
        self.assertFalse(response_avoids_placeholder_scaffolding("...\n질문입니다?"))

    def test_shared_conversation_intents_reject_the_reported_bad_outputs(self) -> None:
        from cogni_agent.response_quality import (
            ResponseIntent,
            compile_response_intent,
            normalize_single_question_response,
            response_respects_airgap_scope,
            response_satisfies_intent,
        )

        proposal_request = (
            "안녕하세요! 정해진 인사말 말고, 오늘 함께 이야기해 볼 주제를 하나 "
            "자연스럽게 제안해 주세요."
        )
        proposal = (
            '오늘 함께 이야기해 볼 주제로는 "AI와 인간의 협업"을 제안해볼까요? '
            "어떤 분야에서 협력이 효과적일지 함께 생각해봐요."
        )
        self.assertIs(
            compile_response_intent(proposal_request),
            ResponseIntent.ONE_TOPIC_PROPOSAL,
        )
        self.assertTrue(response_satisfies_intent(proposal_request, proposal))

        question_request = (
            "프로잭트 아이디어가 막혔어요. 생각을 풀 수 있도록 질문 하나를 "
            "자연스럽게 해 주세요."
        )
        self.assertIs(
            compile_response_intent(question_request),
            ResponseIntent.SINGLE_QUESTION,
        )
        self.assertTrue(
            response_satisfies_intent(
                question_request,
                "이 아이디어에서 가장 먼저 해결하고 싶은 문제는 무엇인가요?",
            )
        )
        self.assertFalse(
            response_satisfies_intent(
                question_request,
                "요청을 먼저 파악하고 자연스럽게 답하십시오.",
            )
        )
        self.assertEqual(
            normalize_single_question_response(
                question_request,
                "먼저 상황을 살펴볼게요. 이 아이디어에서 해결하고 싶은 문제는 "
                "무엇인가요? 다른 설명입니다.",
            ),
            "이 아이디어에서 해결하고 싶은 문제는 무엇인가요?",
        )

        capability_request = (
            "제가 도움을 부탁할 수 있는 범위를 예시와 함께 편한 말로 설명해 주세요."
        )
        reported_bad = (
            "제 이름은 Cogni Agent이고 질문의 의도를 파악해 답하고 싶어요. "
            "예를 들어 파이썬 함수 정의 문법을 설명할 수 있어요."
        )
        useful = (
            "도움을 부탁할 수 있는 일은 코드 검토, 문서 정리, 아이디어 구체화처럼 "
            "다양해요. 예를 들어 오류가 난 함수를 함께 고치거나 보고서를 짧게 "
            "정리할 수 있어요."
        )
        self.assertIs(
            compile_response_intent(capability_request),
            ResponseIntent.CAPABILITY_SCOPE_EXAMPLES,
        )
        self.assertFalse(response_satisfies_intent(capability_request, reported_bad))
        self.assertTrue(response_satisfies_intent(capability_request, useful))

        flow_request = (
            "오프라인 AI 데모에서 개인정보 보호를 가장 먼저 보여주고 싶어요. "
            "어떤 흐름이 자연스러울까요?"
        )
        useful_flow = (
            "먼저 네트워크 차단 상태와 로컬 처리를 보여줍니다. "
            "그 뒤 외부 전송이 없다는 실행 기록을 확인합니다."
        )
        self.assertIs(
            compile_response_intent(flow_request),
            ResponseIntent.DEMONSTRATION_FLOW,
        )
        self.assertTrue(response_satisfies_intent(flow_request, useful_flow))
        self.assertTrue(
            response_satisfies_intent(
                flow_request,
                "첫 화면에서 오프라인 상태와 개인정보의 로컬 처리를 보여줍니다. "
                "이어서 외부 전송 기록이 0건인지 확인합니다.",
            )
        )
        self.assertFalse(
            response_satisfies_intent(
                flow_request,
                "온라인 날씨를 조회할 수 있고 여러 기능이 있습니다.",
            )
        )
        contradictory_flow = (
            "먼저 네트워크 차단 상태를 보여줍니다. 다음으로 네트워크 연결을 "
            "설정하고 외부 정보를 자동으로 불러옵니다."
        )
        self.assertFalse(
            response_respects_airgap_scope(flow_request, contradictory_flow)
        )
        self.assertFalse(response_satisfies_intent(flow_request, contradictory_flow))
        self.assertTrue(
            response_respects_airgap_scope(
                flow_request,
                "먼저 네트워크 연결을 차단합니다. 다음으로 외부 전송 0건을 확인합니다.",
            )
        )
        for unsafe in (
            "먼저 웹에서 최신 날씨를 검색하고 다음으로 API에서 정보를 가져옵니다.",
            "네트워크는 차단되어 있지만 클라우드 서비스를 호출합니다.",
            "먼저 원격 서버에 접속하고 다음으로 결과를 보여줍니다.",
        ):
            with self.subTest(unsafe=unsafe):
                self.assertFalse(response_respects_airgap_scope(flow_request, unsafe))
        self.assertTrue(
            response_respects_airgap_scope(
                flow_request,
                "먼저 인터넷 연결 상태가 오프라인인지 점검해 보여주고, "
                "다음으로 외부 전송 0건을 확인합니다.",
            )
        )
        self.assertFalse(
            response_satisfies_intent(
                flow_request,
                "가장 자연스러운 데모 흐름은 먼저 네트워크 차단 상태와 데이터가 "
                "PC 밖으로 나가지 않는 모습을 보여주고, 끝입니다.",
            )
        )
        self.assertIs(
            compile_response_intent("오프라인 AI를 어떻게 시연하면 자연스러울까요?"),
            ResponseIntent.DEMONSTRATION_FLOW,
        )
        self.assertIs(
            compile_response_intent("데모 실행 화면을 보여 주세요."),
            ResponseIntent.GENERAL,
        )
        self.assertFalse(
            response_satisfies_intent(
                capability_request,
                "도움을 부탁할 수 있는 일은 코드 검토, 문서 정리, "
                "아이디어 구체화처럼 여러 가지입니다.",
            )
        )

    def test_negated_structural_requests_do_not_activate_opposite_intent(self) -> None:
        from cogni_agent.response_quality import (
            ResponseIntent,
            compile_response_intent,
            requested_exact_question_count,
        )

        general_requests = (
            "질문 하나만 하지 말고 바로 답을 설명해 주세요.",
            "질문 하나만으로는 부족하니 예시도 설명해 주세요.",
            "주제를 하나만 제안하지 말고 여러 개를 추천해 주세요.",
            "오프라인 데모 흐름은 설명하지 말고 코드를 작성해 주세요.",
            "도움 범위를 예시 없이 설명해 주세요.",
        )
        for request in general_requests:
            with self.subTest(request=request):
                self.assertIs(compile_response_intent(request), ResponseIntent.GENERAL)
        self.assertIsNone(
            requested_exact_question_count(
                "질문 하나만 하지 말고 바로 답을 설명해 주세요."
            )
        )

        positive_requests = (
            ("질문을 하나만 해 주세요.", ResponseIntent.SINGLE_QUESTION),
            (
                "오늘 이야기할 주제를 하나 제안해 주세요.",
                ResponseIntent.ONE_TOPIC_PROPOSAL,
            ),
            (
                "오프라인 AI 데모 흐름을 단계별로 설명해 주세요.",
                ResponseIntent.DEMONSTRATION_FLOW,
            ),
            (
                "도움을 부탁할 수 있는 범위를 예시와 함께 설명해 주세요.",
                ResponseIntent.CAPABILITY_SCOPE_EXAMPLES,
            ),
        )
        for request, expected in positive_requests:
            with self.subTest(request=request):
                self.assertIs(compile_response_intent(request), expected)

    def test_instruction_echo_and_extended_self_intro_are_rejected(self) -> None:
        from cogni_agent.response_quality import (
            response_avoids_instruction_echo,
            response_avoids_unsolicited_self_intro,
            response_fulfills_examples_request,
        )

        instructions = (
            "당신은 로컬 AI 동료입니다.\n"
            "사용자의 현재 질문과 의도를 먼저 파악하고 자연스럽고 직접적인 "
            "한국어로 답하십시오."
        )
        copied = (
            "사용자의 현재 질문과 의도를 먼저 파악하고 자연스럽고 직접적인 "
            "한국어로 답하십시오."
        )
        self.assertFalse(response_avoids_instruction_echo(instructions, copied))
        self.assertTrue(
            response_avoids_instruction_echo(
                instructions,
                "이 아이디어에서 해결하고 싶은 핵심 문제는 무엇인가요?",
            )
        )
        self.assertFalse(
            response_avoids_unsolicited_self_intro(
                "도움 범위를 알려 주세요.",
                "안녕하세요, 사용자님! 제 이름은 Cogni Agent이고 도움을 드려요.",
            )
        )
        self.assertTrue(
            response_fulfills_examples_request(
                "예시와 함께 설명해 주세요.",
                "코드 검토나 문서 정리처럼 구체적인 작업을 도울 수 있습니다.",
            )
        )
        self.assertFalse(
            response_fulfills_examples_request(
                "예시와 함께 설명해 주세요.",
                "여러 작업을 도울 수 있습니다.",
            )
        )
        self.assertTrue(
            response_fulfills_examples_request(
                "예시 없이 원칙만 설명하세요.",
                "핵심 원칙만 설명합니다.",
            )
        )

    def test_topic_anchor_tolerates_one_general_hangul_typo(self) -> None:
        from cogni_agent.response_quality import response_topically_anchored

        self.assertTrue(
            response_topically_anchored(
                "프로잭트 아이디어가 막혀 생각을 풀고 싶습니다.",
                "어떤 프로젝트를 만들고 싶으신가요?",
            )
        )
        self.assertFalse(
            response_topically_anchored(
                "프로잭트 아이디어가 막혀 생각을 풀고 싶습니다.",
                "오늘 날씨는 어떤가요?",
            )
        )
        self.assertFalse(
            response_topically_anchored(
                "개인정보와 데이터보안, 접근통제의 차이를 한 문장으로 설명해 주세요.",
                "개인정보는 중요합니다.",
            )
        )

    def test_explicit_request_facets_must_survive_generation(self) -> None:
        from cogni_agent.response_quality import (
            missing_request_facets,
            request_required_facets,
            response_covers_request_facets,
            response_satisfies_intent,
        )

        cases = (
            (
                "개인정보가 포함된 요청을 오프라인 환경에서 처리할 때 지켜야 할 "
                "원칙을 세 문장으로 답하세요.",
                "개인정보는 암호화하고 최소 범위에서만 접근해야 합니다. "
                "처리 뒤에는 안전하게 삭제해야 합니다. 보관 기간을 기록해야 합니다.",
                "개인정보는 로컬 장치 안에서 암호화해 처리해야 합니다. "
                "필요한 최소 범위에서만 접근해야 합니다. 처리 뒤에는 안전하게 "
                "삭제해야 합니다.",
                "오프라인·로컬 처리",
            ),
            (
                "제한된 GPU 메모리에서 추론할 때 측정값과 설계 목표를 구분해야 "
                "하는 이유를 설명하세요.",
                "품질 검증을 통과하지 못해 이번에는 추측해서 답하지 않았습니다.",
                "GPU 메모리 실측값은 현재 실행의 관측 결과이고 설계 목표는 "
                "달성해야 할 상한이므로 둘을 구분해야 합니다.",
                "측정값",
            ),
            (
                "불확실한 답변을 사실처럼 단정하지 않기 위한 표현 원칙을 두 "
                "문장으로 설명하세요.",
                "사실 여부를 검증하고 단정하지 않아야 합니다. 독자에게 확인을 "
                "권장해야 합니다.",
                "불확실한 내용은 추정이나 가능성이라고 명시해야 합니다. 확인된 "
                "근거가 없으면 사실로 단정하지 않아야 합니다.",
                "불확실성",
            ),
            (
                "사용자 권한과 시스템 안전 경계를 함께 지키는 작업 실행 원칙을 "
                "세 문장으로 답하세요.",
                "요청을 수행할 때 안전 경계를 넘지 않아야 합니다. 위험하면 "
                "거절해야 합니다. 모호하면 질문해야 합니다.",
                "작업은 사용자가 허용한 권한 안에서만 실행해야 합니다. 시스템 "
                "안전 경계를 넘는 명령은 차단해야 합니다. 실행 결과를 기록해야 "
                "합니다.",
                "사용자 권한",
            ),
            (
                "소프트웨어 수정 완료를 선언하기 전에 필요한 자체 검증을 네 "
                "항목 이내로 정리하세요.",
                "계정 입력을 검사합니다. 예외 처리 코드를 확인합니다. 테스트 "
                "문서를 작성합니다. 성공 메시지를 표시합니다.",
                "수정된 기능을 직접 실행합니다. 전체 회귀 테스트를 수행합니다. "
                "오류 로그와 성능을 확인합니다. 검증이 통과했을 때만 완료를 "
                "선언합니다.",
                "소프트웨어 수정",
            ),
            (
                "자연스러운 한국어 답변의 완결성을 판정할 때 확인할 사항을 세 "
                "문장으로 설명하세요.",
                "100자 이내로 요약하세요. 답변의 완결성을 판정할 사항을 "
                "설명하세요. 준비가 완료되었습니다.",
                "한국어 문장은 서술어와 마침표로 완결되어야 합니다. 같은 뜻의 "
                "반복이 없어야 합니다. 문법과 표현이 자연스러운지 확인해야 "
                "합니다.",
                "자연스러움·반복 방지",
            ),
        )
        for request, bad, good, expected_missing in cases:
            with self.subTest(request=request):
                self.assertIn(expected_missing, request_required_facets(request))
                self.assertIn(expected_missing, missing_request_facets(request, bad))
                self.assertFalse(response_covers_request_facets(request, bad))
                self.assertFalse(response_satisfies_intent(request, bad))
                self.assertEqual(missing_request_facets(request, good), ())
                self.assertTrue(response_covers_request_facets(request, good))
                self.assertTrue(response_satisfies_intent(request, good))

    def test_facet_gate_is_inactive_for_ordinary_conversation(self) -> None:
        from cogni_agent.response_quality import (
            request_required_facets,
            response_covers_request_facets,
        )

        request = "오늘 기분이 어때요? 자연스럽게 이야기해 주세요."
        self.assertEqual(request_required_facets(request), ())
        self.assertTrue(
            response_covers_request_facets(request, "좋아요, 함께 이야기해요.")
        )

    def test_paraphrased_prompt_echo_and_readiness_meta_are_rejected(self) -> None:
        from cogni_agent.response_quality import (
            response_avoids_meta_format_discussion,
            response_avoids_prompt_echo,
        )

        request = (
            "자연스러운 한국어 답변의 완결성을 판정할 때 확인할 사항을 세 "
            "문장으로 설명하세요."
        )
        response = (
            "100자 이내로 요약하세요. 사용자가 요청한 답변의 완결성을 판정할 "
            "때 확인할 사항을 세 문장으로 설명하세요. 사용자의 요청을 이해하고 "
            "답변을 작성할 준비가 완료되었습니다."
        )
        self.assertFalse(response_avoids_prompt_echo(request, response))
        self.assertFalse(response_avoids_meta_format_discussion(request, response))


if __name__ == "__main__":
    unittest.main()
