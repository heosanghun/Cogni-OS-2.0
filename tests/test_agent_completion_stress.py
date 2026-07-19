from __future__ import annotations

import os
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from scripts.gpu5_boundary_guard import (
    NVIDIA_SMI_EXECUTABLE,
    PROJECT_GPU_UUID,
    _MINIMAL_HOST_ENVIRONMENT,
    _PRODUCT_BUILD_VERSION,
    _PRODUCT_EFFECTIVE_PARAMETERS,
    _PRODUCT_ACCEPTANCE_PROMPTS as GUARDED_PRODUCT_ACCEPTANCE_PROMPTS,
    _PRODUCT_REQUIRED_TURN_CHECKS,
    _PRODUCT_STORED_PARAMETERS,
)
from cogni_os.version import __version__
from scripts.validate_agent_completion import (
    DEFAULT_PROMPTS,
    POST_TURN_MEMORY_SAMPLE_SCOPE,
    PRODUCT_ACCEPTANCE_COVERAGE,
    PRODUCT_ACCEPTANCE_PROMPTS,
    PRODUCT_ACCEPTANCE_SUITE,
    PRODUCT_CONTINUATION_COUNT,
    PRODUCT_CONTINUATION_FIRST_TOKENS,
    PRODUCT_CONTINUATION_TOTAL_TOKENS,
    PRODUCT_CONTINUATION_TURN,
    PromptCase,
    _answer_checks,
    _expected_factbook_identity,
    _korean_completion_metrics,
    _prompt_cases,
    _product_acceptance_cases,
    _query_nvidia_smi_gpu_memory_bytes,
    _read_process_rss_bytes,
    _sample_worker_memory,
    _sentence_repetition_metrics,
    _substantive_sentence_keys,
    _summarize_turns,
    _turn_record,
    _worker_snapshot,
    build_parser,
    execute,
)


def _healthy_worker(*, gpu_bytes: int | None = 800) -> dict[str, object]:
    return {
        "expected_running": True,
        "running": True,
        "pid": 1234,
        "stable_pid_before_turn": 1234,
        "pid_stable": True,
        "active_request_id": None,
        "healthy": True,
        "memory": {
            "sample_scope": POST_TURN_MEMORY_SAMPLE_SCOPE,
            "captures_peak": False,
            "worker_rss_spot_sample_bytes": 500,
            "worker_rss_spot_sample_status": "measured",
            "gpu_memory_spot_sample_bytes": gpu_bytes,
            "gpu_memory_spot_sample_status": (
                "measured" if gpu_bytes is not None else "driver_unreported"
            ),
            "gpu_memory_spot_sample_within_threshold": (
                None if gpu_bytes is None else gpu_bytes <= 1_000
            ),
            "gpu_memory_spot_sample_threshold_bytes": 1_000,
            "spot_sample_observed": True,
        },
    }


def _complete_state() -> dict[str, object]:
    return {
        "status": "succeeded",
        "stage": "complete",
        "completion": {"truncated": False},
    }


def _complete_answer(
    text: str = "검증된 답변을 한 번만 제공합니다.",
) -> dict[str, object]:
    return {
        "content": text,
        "finish_reason": "stop",
        "truncated": False,
        "continuations": 0,
        "generated_tokens": 12,
    }


class TestAgentCompletionStressValidation(unittest.TestCase):
    def test_product_acceptance_suite_is_exactly_20_and_covers_public_failures(self):
        cases = _product_acceptance_cases()
        self.assertEqual(len(cases), 20)
        self.assertEqual(len({case.label for case in cases}), 20)
        self.assertEqual(
            PRODUCT_ACCEPTANCE_COVERAGE,
            {
                "casual_korean",
                "typo_tolerance",
                "follow_up_context",
                "continuation_completion",
                "repetition_resistance",
                "truthful_identity",
                "false_7b_rejection",
                "role_control_leak_rejection",
                "cutoff_rejection",
                "zero_quality_fallback",
            },
        )
        prompts = "\n".join(case.prompt for case in cases)
        self.assertIn("안녕하세여", prompts)
        self.assertIn("방금", prompts)
        self.assertIn("끊기지", prompts)
        self.assertIn("반복", prompts)
        self.assertIn("파라미터", prompts)
        self.assertEqual(
            PRODUCT_ACCEPTANCE_PROMPTS, tuple(case.prompt for case in cases)
        )
        self.assertEqual(PRODUCT_ACCEPTANCE_PROMPTS, GUARDED_PRODUCT_ACCEPTANCE_PROMPTS)
        self.assertEqual(PRODUCT_CONTINUATION_TURN, 9)
        self.assertEqual(PRODUCT_CONTINUATION_COUNT, 1)
        self.assertEqual(PRODUCT_CONTINUATION_FIRST_TOKENS, 96)
        self.assertEqual(PRODUCT_CONTINUATION_TOTAL_TOKENS, 192)

    def test_product_guard_check_contract_matches_the_turn_record_schema(self):
        record = _turn_record(
            turn_number=1,
            case=PromptCase("schema", "검증을 설명하세요.", "generated"),
            session_id="completion-a",
            peer_session_id="completion-b",
            state=_complete_state(),
            answer={**_complete_answer(), "generation_mode": "cogni_core"},
            elapsed_seconds=1.0,
            worker=_healthy_worker(),
            peer_before_digest="a" * 64,
            peer_after_digest="a" * 64,
            prior_answer_digests=set(),
            new_assistant_count=1,
            new_user_count=1,
            observed_user_prompt="검증을 설명하세요.",
        )

        self.assertEqual(set(record["checks"]), _PRODUCT_REQUIRED_TURN_CHECKS)

    def test_turn_record_requires_one_new_assistant_and_exact_continuation_probe(self):
        ordinary = _turn_record(
            turn_number=1,
            case=PromptCase("ordinary", "검증을 설명하세요.", "generated"),
            session_id="completion-a",
            peer_session_id="completion-b",
            state=_complete_state(),
            answer={**_complete_answer(), "generation_mode": "cogni_core"},
            elapsed_seconds=1.0,
            worker=_healthy_worker(),
            peer_before_digest="a" * 64,
            peer_after_digest="a" * 64,
            prior_answer_digests=set(),
            new_assistant_count=2,
            new_user_count=1,
            observed_user_prompt="검증을 설명하세요.",
        )
        self.assertFalse(ordinary["checks"]["exactly_one_assistant"])
        self.assertFalse(ordinary["passed"])

        probe = _turn_record(
            turn_number=PRODUCT_CONTINUATION_TURN,
            case=PromptCase("continuation", "자세히 분석하세요.", "generated"),
            session_id="completion-b",
            peer_session_id="completion-a",
            state=_complete_state(),
            answer={
                **_complete_answer(),
                "generation_mode": "cogni_core",
                "continuations": PRODUCT_CONTINUATION_COUNT,
            },
            elapsed_seconds=1.0,
            worker=_healthy_worker(),
            peer_before_digest="a" * 64,
            peer_after_digest="a" * 64,
            prior_answer_digests=set(),
            new_assistant_count=1,
            new_user_count=1,
            observed_user_prompt="자세히 분석하세요.",
        )
        self.assertTrue(probe["checks"]["exactly_one_assistant"])
        self.assertTrue(probe["checks"]["continuation_contract"])

    def test_product_guard_identity_constants_match_the_release_factbook(self):
        self.assertEqual(_PRODUCT_BUILD_VERSION, __version__)
        self.assertEqual(_PRODUCT_EFFECTIVE_PARAMETERS, 4_506_496_490)
        self.assertEqual(_PRODUCT_STORED_PARAMETERS, 7_996_157_418)

    def test_product_suite_parser_requires_exact_turn_count_at_execution(self):
        parser = build_parser()
        required = [
            "--model",
            "m",
            "--manifest",
            "x",
            "--physical-gpu-index",
            "5",
            "--gpu-query-context",
            "gpu5-container",
            "--suite",
            PRODUCT_ACCEPTANCE_SUITE,
            "--strict-json",
        ]
        accepted = parser.parse_args(required)
        self.assertEqual(accepted.turns, 20)
        self.assertTrue(accepted.strict_json)
        rejected = parser.parse_args([*required, "--turns", "19"])
        with self.assertRaisesRegex(ValueError, "exactly 20"):
            execute(rejected)

    def test_release_defaults_to_twenty_turns_and_120_second_ceiling(self):
        parser = build_parser()
        required = [
            "--model",
            "m",
            "--manifest",
            "x",
            "--physical-gpu-index",
            "5",
            "--gpu-query-context",
            "gpu5-container",
        ]
        default = parser.parse_args(required)
        stress = parser.parse_args([*required, "--turns", "20"])
        self.assertEqual(default.turns, 20)
        self.assertEqual(default.physical_gpu_index, 5)
        self.assertEqual(default.gpu_query_context, "gpu5-container")
        self.assertEqual(default.timeout, 120.0)
        self.assertEqual(stress.turns, 20)
        for missing in (
            [
                "--model",
                "m",
                "--manifest",
                "x",
                "--gpu-query-context",
                "gpu5-container",
            ],
            ["--model", "m", "--manifest", "x", "--physical-gpu-index", "5"],
        ):
            with self.subTest(missing=missing), self.assertRaises(SystemExit):
                parser.parse_args(missing)
        for value in ("0", "101", "not-a-number"):
            with self.subTest(value=value), self.assertRaises(SystemExit):
                parser.parse_args([*required, "--turns", value])
        for value in ("120.001", "nan", "inf", "-inf"):
            with self.subTest(timeout=value), self.assertRaises(SystemExit):
                parser.parse_args([*required, "--timeout", value])
        for value in ("0", "4", "6", "7", "not-a-number"):
            with self.subTest(gpu=value), self.assertRaises(SystemExit):
                parser.parse_args(
                    [
                        "--model",
                        "m",
                        "--manifest",
                        "x",
                        "--physical-gpu-index",
                        value,
                        "--gpu-query-context",
                        "gpu5-container",
                    ]
                )

        diagnostic = [
            {
                "passed": True,
                "checks": {},
                "repetition": {},
                "worker": {},
                "generation_mode": "cogni_core",
            }
            for _ in range(4)
        ]
        summary = _summarize_turns(diagnostic, 4)
        self.assertFalse(summary["recommended_stress_schedule_completed"])
        self.assertFalse(summary["strict_completion_stress_gate_passed"])

    def test_twenty_turn_schedule_keeps_original_four_and_is_deterministic(self):
        first = _prompt_cases(20)
        second = _prompt_cases(20)
        self.assertEqual(first, second)
        self.assertEqual(tuple(case.prompt for case in first[:4]), DEFAULT_PROMPTS)
        self.assertEqual([case.expected_route for case in first[:4]], ["grounded"] * 4)
        self.assertEqual(first[13].expected_route, "grounded")
        self.assertTrue(
            all(
                case.expected_route == "generated"
                for index, case in enumerate(first[4:], start=4)
                if index != 13
            )
        )
        self.assertEqual(len({case.label for case in first}), 20)

    def test_repetition_role_truncation_and_korean_completion_are_explicit(self):
        metrics = _sentence_repetition_metrics(
            "같은 문장을 반복하면 실패합니다. 같은 문장을 반복하면 실패합니다."
        )
        self.assertEqual(metrics["sentence_count"], 2)
        self.assertEqual(metrics["duplicate_sentence_count"], 1)
        self.assertEqual(metrics["duplicate_sentence_rate"], 0.5)

    def test_adversarial_repetition_topic_and_trailing_marker_are_rejected(self):
        state = _complete_state()
        short_loop = _complete_answer("좋아요. " * 30)
        checks = _answer_checks(short_loop, state)
        self.assertFalse(checks["no_short_sentence_loop"])
        self.assertFalse(checks["quality_report_accepts"])

        off_topic = _complete_answer(
            "사과는 빨갛습니다. 바다는 넓습니다. 하늘은 맑습니다."
        )
        checks = _answer_checks(
            off_topic,
            state,
            "백업과 검증과 롤백을 세 문장으로 설명하세요.",
            required_groups=(("백업",), ("검증",), ("롤백",)),
        )
        self.assertFalse(checks["topic_anchors_satisfied"])

        truncated_list = _complete_answer(
            "1. 백업을 만듭니다. 2. 회귀 테스트를 실행합니다. 3."
        )
        checks = _answer_checks(truncated_list, state)
        self.assertFalse(checks["korean_complete"])

        unbalanced_quote = _complete_answer("“불확실한 내용은 가능성으로 표현합니다.")
        checks = _answer_checks(unbalanced_quote, state)
        self.assertFalse(checks["balanced_smart_quotes"])

        echoed_request = (
            "오래된 대화 문맥을 줄이면서 사용자 의도를 보존하는 방법을 설명하세요."
        )
        checks = _answer_checks(
            _complete_answer("핵심을 요약합니다. " + echoed_request),
            state,
            echoed_request,
        )
        self.assertFalse(checks["no_full_prompt_echo"])

        self_intro = _complete_answer(
            "안녕하세요, AI 어시스턴트입니다. 문맥을 요약합니다."
        )
        checks = _answer_checks(
            self_intro,
            state,
            "문맥을 요약하는 방법을 설명하세요.",
        )
        self.assertFalse(checks["no_unsolicited_self_intro"])

        dangling = _complete_answer("에서 잘못된 사실을 정정합니다.")
        checks = _answer_checks(dangling, state)
        self.assertFalse(checks["no_dangling_sentence_start"])

        outline = _complete_answer(
            "### 서론\n검증 목적을 설명합니다.\n### 개요\n품질을 설명합니다."
        )
        checks = _answer_checks(
            outline,
            state,
            "자체 검증을 네 항목 이내로 정리하세요.",
        )
        self.assertFalse(checks["no_generic_outline"])

        meta = _answer_checks(
            _complete_answer(
                "질문의 형식을 다시 정리합니다. 사용자는 네 개로 제한하라고 요청합니다."
            ),
            state,
            "확인 항목을 네 가지 이내로 정리하세요.",
        )
        self.assertFalse(meta["no_meta_format_discussion"])

    def test_semantic_on_device_and_summary_answers_keep_topic_anchors(self) -> None:
        state = _complete_state()
        cases = _prompt_cases(20)

        on_device = _answer_checks(
            _complete_answer(
                "데이터가 인터넷에 노출되지 않아 개인정보가 보호됩니다. "
                "장치에서 처리하므로 인터넷 연결 없이 사용할 수 있습니다. "
                "처리 가능한 데이터 크기에는 제한이 있습니다."
            ),
            state,
            cases[4].prompt,
            required_groups=cases[4].required_groups,
        )
        self.assertTrue(on_device["topic_anchors_satisfied"])

        summary = _answer_checks(
            _complete_answer(
                "군더더기 없이 핵심만 담아야 합니다. "
                "문장마다 다른 내용을 간결하게 표현해야 합니다. "
                "마지막 문장을 자연스럽게 완결해야 합니다."
            ),
            state,
            cases[10].prompt,
            required_groups=cases[10].required_groups,
        )
        self.assertTrue(summary["topic_anchors_satisfied"])

        observed_paraphrases = (
            (
                5,
                "사용자의 정정 내용을 경청하고 수용합니다. 내부 지식 기반에서 "
                "정보의 정확성을 검토하고 업데이트합니다. 정정된 사실을 다시 "
                "전달합니다.",
            ),
            (
                6,
                "검증된 정보만 제시해야 합니다. 이 원칙은 답변의 신뢰성을 "
                "유지하기 위한 핵심 기준입니다.",
            ),
            (
                10,
                "원문의 핵심 내용을 빠짐없이 담아야 합니다. 불필요한 세부 "
                "사항과 수식어를 제거해 간결해야 합니다. 자신의 언어로 "
                "재구성해 반복을 피해야 합니다.",
            ),
        )
        for case_index, text in observed_paraphrases:
            with self.subTest(case=cases[case_index].label):
                checks = _answer_checks(
                    _complete_answer(text),
                    state,
                    cases[case_index].prompt,
                    required_groups=cases[case_index].required_groups,
                )
                self.assertTrue(checks["topic_anchors_satisfied"])

    def test_required_literal_period_and_factbook_values_are_exact(self):
        state = _complete_state()
        literal = _answer_checks(
            _complete_answer("검증을 마쳤습니다."),
            state,
            "답변은 반드시 '이상입니다.'로 끝내세요.",
        )
        self.assertFalse(literal["required_literal_ending"])
        period = _answer_checks(
            _complete_answer("검증을 완료했습니다!"),
            state,
            "한 문장으로 설명하고 마침표로 끝내세요.",
        )
        self.assertFalse(period["required_period_ending"])

        expected = {
            "build_version": "0.3.2",
            "model_label": "gemma4-e4b",
            "stored_parameters": 7_996_157_418,
            "effective_parameters": 4_506_496_490,
        }
        correct = _complete_answer(
            "gemma4-e4b 모델이며 저장 파라미터는 7,996,157,418개, "
            "effective 파라미터는 4,506,496,490개입니다. 현재 빌드는 0.3.2입니다."
        )
        correct_checks = _answer_checks(
            correct,
            state,
            "정확히 어떤 모델이며 파라미터는 몇 개인가요?",
            expected_factbook=expected,
        )
        self.assertTrue(correct_checks["factbook_model_exact"])
        self.assertTrue(correct_checks["factbook_version_exact"])
        self.assertTrue(correct_checks["factbook_parameters_exact"])

        wrong = dict(correct)
        wrong["content"] = str(correct["content"]).replace("0.3.2", "0.3.1")
        wrong_checks = _answer_checks(
            wrong,
            state,
            "정확히 어떤 모델이며 파라미터는 몇 개인가요?",
            expected_factbook=expected,
        )
        self.assertFalse(wrong_checks["factbook_version_exact"])

    def test_expected_factbook_identity_reads_typed_inventory(self):
        factbook = SimpleNamespace(
            build_version="0.3.2",
            model=SimpleNamespace(
                label="gemma4-e4b",
                inventory=SimpleNamespace(
                    stored_parameters=7_996_157_418,
                    effective_parameters=4_506_496_490,
                ),
            ),
        )

        self.assertEqual(
            _expected_factbook_identity(factbook),
            {
                "build_version": "0.3.2",
                "model_label": "gemma4-e4b",
                "stored_parameters": 7_996_157_418,
                "effective_parameters": 4_506_496_490,
            },
        )

    def test_parallel_structured_checks_are_not_near_duplicate_false_positives(self):
        answer = _complete_answer(
            "수정된 기능이 예상대로 작동하는지 확인합니다. "
            "수정된 기능이 다른 기능에 영향을 미치지 않는지 확인합니다. "
            "수정된 기능이 성능에 영향을 미치지 않는지 확인합니다. "
            "수정된 기능이 보안에 영향을 미치지 않는지 확인합니다."
        )
        checks = _answer_checks(
            answer,
            _complete_state(),
            "자체 검증을 네 항목 이내로 정리하세요.",
        )
        self.assertTrue(checks["no_near_duplicate_sentence"])

    def test_release_gate_rejects_any_quality_fallback(self):
        base = {
            "passed": True,
            "checks": {},
            "repetition": {},
            "worker": {},
            "generation_mode": "cogni_core",
        }
        one_fallback = [dict(base) for _ in range(20)]
        one_fallback[0]["generation_mode"] = "quality_fallback"
        rejected_once = _summarize_turns(one_fallback, 20)
        self.assertFalse(rejected_once["quality_fallback_gate_passed"])
        self.assertFalse(rejected_once["strict_completion_stress_gate_passed"])
        self.assertEqual(rejected_once["content_answer_rate"], 0.95)

        two_fallbacks = [dict(item) for item in one_fallback]
        two_fallbacks[1]["generation_mode"] = "quality_fallback"
        rejected_twice = _summarize_turns(two_fallbacks, 20)
        self.assertFalse(rejected_twice["quality_fallback_gate_passed"])
        self.assertFalse(rejected_twice["strict_completion_stress_gate_passed"])

        incomplete = _korean_completion_metrics("이는 내가.")
        self.assertFalse(incomplete["complete"])
        self.assertIn("dangling_korean_clause", incomplete["reasons"])
        self.assertTrue(_korean_completion_metrics("검증을 완료했습니다.")["complete"])

        answer = _complete_answer("ASSISTANT: 검증 중인 답변")
        answer["truncated"] = True
        answer["finish_reason"] = "length"
        checks = _answer_checks(answer, _complete_state())
        self.assertFalse(checks["no_role_leak"])
        self.assertFalse(checks["not_truncated"])
        self.assertFalse(checks["not_explicitly_truncated"])
        self.assertFalse(checks["korean_complete"])

    def test_gpu_memory_query_fails_closed_without_project_selector(self):
        calls = []

        def forbidden_runner(*args, **kwargs):
            calls.append((args, kwargs))
            raise AssertionError("nvidia-smi must not run without the selector")

        self.assertEqual(
            _query_nvidia_smi_gpu_memory_bytes(77, runner=forbidden_runner),
            (None, "gpu_selector_required"),
        )
        self.assertEqual(
            _query_nvidia_smi_gpu_memory_bytes(
                77,
                physical_gpu_index=6,
                runner=forbidden_runner,
            ),
            (None, "gpu_selector_rejected"),
        )
        self.assertEqual(calls, [])

    def test_gpu_memory_query_names_only_index5_and_checks_uuid(self):
        calls = []

        def runner(argv, **kwargs):
            calls.append((tuple(argv), kwargs))
            return SimpleNamespace(
                returncode=0,
                stdout=f"{PROJECT_GPU_UUID}, 77, 900\n",
                stderr="",
            )

        with patch(
            "scripts.validate_agent_completion._validated_executable",
            return_value=str(NVIDIA_SMI_EXECUTABLE),
        ):
            measured = _query_nvidia_smi_gpu_memory_bytes(
                77,
                physical_gpu_index=5,
                gpu_query_context="native-host",
                runner=runner,
            )
        self.assertEqual(measured, (900 * 1024**2, "measured"))
        self.assertEqual(calls[0][0][:3], (str(NVIDIA_SMI_EXECUTABLE), "-i", "5"))
        self.assertNotIn("6", calls[0][0][:3])
        self.assertNotIn("7", calls[0][0][:3])
        self.assertEqual(calls[0][1]["timeout"], 5)
        self.assertEqual(calls[0][1]["env"], _MINIMAL_HOST_ENVIRONMENT)

        def wrong_uuid(argv, **kwargs):
            return SimpleNamespace(
                returncode=0,
                stdout="GPU-wrong, 77, 900\n",
                stderr="",
            )

        with patch(
            "scripts.validate_agent_completion._validated_executable",
            return_value=str(NVIDIA_SMI_EXECUTABLE),
        ):
            self.assertEqual(
                _query_nvidia_smi_gpu_memory_bytes(
                    77,
                    physical_gpu_index=5,
                    gpu_query_context="native-host",
                    runner=wrong_uuid,
                ),
                (None, "gpu_uuid_mismatch"),
            )

    def test_container_gpu_memory_query_uses_uuid_aggregate_not_worker_pid(self):
        calls = []

        def runner(argv, **kwargs):
            calls.append((tuple(argv), kwargs))
            return SimpleNamespace(
                returncode=0,
                stdout=f"{PROJECT_GPU_UUID}, 900\n",
                stderr="",
            )

        with patch(
            "scripts.validate_agent_completion._validated_executable",
            return_value=str(NVIDIA_SMI_EXECUTABLE),
        ):
            measured = _query_nvidia_smi_gpu_memory_bytes(
                999999,
                physical_gpu_index=5,
                gpu_query_context="gpu5-container",
                runner=runner,
            )
        self.assertEqual(measured, (900 * 1024**2, "measured_aggregate"))
        self.assertEqual(
            calls[0][0][:3],
            (str(NVIDIA_SMI_EXECUTABLE), "-i", PROJECT_GPU_UUID),
        )
        self.assertIn("--query-gpu=uuid,memory.used", calls[0][0])
        self.assertFalse(any("query-compute-apps" in token for token in calls[0][0]))
        self.assertEqual(calls[0][1]["env"], _MINIMAL_HOST_ENVIRONMENT)

        def forbidden_runner(*args, **kwargs):
            raise AssertionError("invalid context must not run nvidia-smi")

        self.assertEqual(
            _query_nvidia_smi_gpu_memory_bytes(
                77, physical_gpu_index=5, runner=forbidden_runner
            ),
            (None, "gpu_query_context_required"),
        )

    def test_execute_without_physical_selector_fails_before_model_or_cuda(self):
        report, code = execute(
            SimpleNamespace(
                model="does-not-exist",
                manifest="does-not-exist",
                timeout=120.0,
                turns=1,
                output=None,
            )
        )
        self.assertEqual(code, 1)
        self.assertIn("GPU5BoundaryError", report["error"])
        self.assertNotIn("verified_files", report)

    def test_execute_without_gpu_query_context_fails_before_model_or_cuda(self):
        report, code = execute(
            SimpleNamespace(
                model="does-not-exist",
                manifest="does-not-exist",
                timeout=120.0,
                turns=1,
                output=None,
                physical_gpu_index=5,
                gpu_query_context=None,
            )
        )
        self.assertEqual(code, 1)
        self.assertIn("GPU5BoundaryError", report["error"])
        self.assertNotIn("verified_files", report)

    def test_execute_preserves_primary_error_and_post_identity_error(self):
        identity_before = SimpleNamespace(as_payload=lambda: {"uuid": "GPU5-before"})
        identity_after = SimpleNamespace(as_payload=lambda: {"uuid": "GPU5-after"})
        verified = SimpleNamespace(files=())
        args = SimpleNamespace(
            model="model",
            manifest="manifest",
            timeout=120.0,
            turns=1,
            output=None,
            physical_gpu_index=5,
            gpu_query_context="gpu5-container",
        )
        with (
            patch(
                "scripts.validate_agent_completion.validate_guarded_gpu5_identity",
                side_effect=(identity_before, identity_after),
            ) as identity_check,
            patch(
                "scripts.validate_agent_completion.verify_artifact_manifest",
                side_effect=(verified, verified),
            ) as manifest_check,
            patch(
                "scripts.validate_agent_completion.torch.cuda.is_available",
                return_value=False,
            ),
        ):
            report, code = execute(args)

        self.assertEqual(code, 1)
        self.assertEqual(report["schema"], "cogni.agent.completion.stress.v2")
        self.assertEqual(
            report["memory_evidence_scope"]["kind"],
            POST_TURN_MEMORY_SAMPLE_SCOPE,
        )
        self.assertIs(report["memory_evidence_scope"]["captures_peak"], False)
        self.assertIn("CUDA is required", report["error"])
        self.assertIn("identity changed", report["post_gpu_identity_error"])
        self.assertEqual(identity_check.call_count, 2)
        self.assertEqual(manifest_check.call_count, 2)

    def test_memory_sampler_keeps_gpu_unverified_separate_from_rss(self):
        observed = _sample_worker_memory(
            77,
            gpu_spot_sample_threshold_bytes=1_000,
            rss_reader=lambda _pid: 500,
            gpu_reader=lambda _pid: (900, "measured"),
        )
        self.assertEqual(observed["sample_scope"], POST_TURN_MEMORY_SAMPLE_SCOPE)
        self.assertIs(observed["captures_peak"], False)
        self.assertTrue(observed["spot_sample_observed"])
        self.assertTrue(observed["gpu_memory_spot_sample_within_threshold"])

        driver_hidden = _sample_worker_memory(
            77,
            gpu_spot_sample_threshold_bytes=1_000,
            rss_reader=lambda _pid: 500,
            gpu_reader=lambda _pid: (None, "driver_unreported"),
        )
        self.assertTrue(driver_hidden["spot_sample_observed"])
        self.assertIsNone(driver_hidden["gpu_memory_spot_sample_within_threshold"])
        self.assertEqual(
            driver_hidden["gpu_memory_spot_sample_status"], "driver_unreported"
        )

    def test_current_process_rss_is_observable_without_optional_packages(self):
        self.assertGreater(_read_process_rss_bytes(os.getpid()), 0)

    def test_worker_snapshot_requires_stable_idle_resident_worker(self):
        service = SimpleNamespace(
            is_running=True,
            worker_pid=81,
            active_request_id=None,
        )

        def sampler(pid, *, gpu_spot_sample_threshold_bytes):
            self.assertEqual(pid, 81)
            self.assertEqual(gpu_spot_sample_threshold_bytes, 1_000)
            return _healthy_worker()["memory"]

        healthy = _worker_snapshot(
            service,
            expected_running=True,
            stable_pid=81,
            gpu_spot_sample_threshold_bytes=1_000,
            memory_sampler=sampler,
        )
        self.assertTrue(healthy["healthy"])

        service.active_request_id = 9
        busy = _worker_snapshot(
            service,
            expected_running=True,
            stable_pid=81,
            gpu_spot_sample_threshold_bytes=1_000,
            memory_sampler=sampler,
        )
        self.assertFalse(busy["healthy"])

    def test_turn_record_and_summary_fail_closed_on_duplicate_or_session_leak(self):
        case = PromptCase("stress-05", "간결하게 답하세요.", "generated")
        answer = _complete_answer()
        first = _turn_record(
            turn_number=1,
            case=case,
            session_id="completion-a",
            peer_session_id="completion-b",
            state=_complete_state(),
            answer=answer,
            elapsed_seconds=1.0,
            worker=_healthy_worker(),
            peer_before_digest="a" * 64,
            peer_after_digest="a" * 64,
            prior_answer_digests=set(),
            new_assistant_count=1,
            new_user_count=1,
            observed_user_prompt=case.prompt,
        )
        self.assertTrue(first["passed"])

        second = _turn_record(
            turn_number=2,
            case=case,
            session_id="completion-a",
            peer_session_id="completion-b",
            state=_complete_state(),
            answer=answer,
            elapsed_seconds=1.0,
            worker=_healthy_worker(gpu_bytes=None),
            peer_before_digest="a" * 64,
            peer_after_digest="b" * 64,
            prior_answer_digests={first["answer_sha256"]},
            new_assistant_count=1,
            new_user_count=1,
            observed_user_prompt=case.prompt,
        )
        self.assertFalse(second["passed"])
        self.assertFalse(second["checks"]["no_cross_turn_exact_duplicate"])
        self.assertFalse(second["checks"]["session_isolated"])

        summary = _summarize_turns([first, second], 2)
        self.assertEqual(summary["turn_success_rate"], 0.5)
        self.assertFalse(summary["strict_completion_stress_gate_passed"])
        self.assertFalse(summary["post_turn_gpu_memory_spot_sample_coverage_complete"])
        self.assertEqual(
            summary["post_turn_gpu_memory_spot_sample_coverage_verdict"],
            "incomplete",
        )

    def test_completion_gate_requires_full_post_turn_gpu_spot_sample_coverage(self):
        base = {
            "passed": True,
            "checks": {},
            "repetition": {},
            "worker": _healthy_worker(),
            "generation_mode": "cogni_core",
        }
        complete = _summarize_turns([dict(base) for _ in range(20)], 20)
        self.assertTrue(complete["post_turn_gpu_memory_spot_sample_coverage_complete"])
        self.assertEqual(
            complete["post_turn_gpu_memory_spot_sample_coverage_rate"], 1.0
        )
        self.assertEqual(
            complete["post_turn_gpu_memory_spot_sample_coverage_verdict"],
            "complete",
        )
        self.assertTrue(complete["strict_completion_stress_gate_passed"])
        self.assertTrue(
            complete["post_turn_gpu_memory_spot_sample_threshold_gate_passed"]
        )
        self.assertNotIn("peak_worker_gpu_memory_bytes", complete)
        self.assertEqual(
            complete["maximum_observed_post_turn_gpu_memory_spot_sample_bytes"],
            800,
        )
        self.assertEqual(complete["resident_worker_pids"], [1234])
        self.assertTrue(complete["single_resident_worker_scope"])

        second_worker = _healthy_worker()
        second_worker["pid"] = 4321
        second_worker["stable_pid_before_turn"] = 4321
        mixed_workers = [dict(base) for _ in range(20)]
        mixed_workers[-1] = {**base, "worker": second_worker}
        mixed_summary = _summarize_turns(mixed_workers, 20)
        self.assertEqual(mixed_summary["resident_worker_pids"], [1234, 4321])
        self.assertFalse(mixed_summary["single_resident_worker_scope"])
        self.assertFalse(mixed_summary["strict_completion_stress_gate_passed"])

        legacy_worker = _healthy_worker()
        legacy_worker["memory"].pop("sample_scope")
        legacy = _summarize_turns(
            [{**base, "worker": legacy_worker} for _ in range(20)], 20
        )
        self.assertFalse(legacy["post_turn_gpu_memory_spot_sample_coverage_complete"])

        over_threshold = _summarize_turns(
            [{**base, "worker": _healthy_worker(gpu_bytes=1_200)} for _ in range(20)],
            20,
        )
        self.assertTrue(
            over_threshold["post_turn_gpu_memory_spot_sample_coverage_complete"]
        )
        self.assertFalse(over_threshold["strict_completion_stress_gate_passed"])
        self.assertFalse(
            over_threshold["post_turn_gpu_memory_spot_sample_threshold_gate_passed"]
        )
        self.assertEqual(
            over_threshold["post_turn_gpu_memory_spot_sample_threshold_observation"],
            "observed_above_threshold",
        )

        missing = [dict(base) for _ in range(20)]
        missing[-1] = {**base, "worker": _healthy_worker(gpu_bytes=None)}
        incomplete = _summarize_turns(missing, 20)
        self.assertFalse(
            incomplete["post_turn_gpu_memory_spot_sample_coverage_complete"]
        )
        self.assertEqual(
            incomplete["post_turn_gpu_memory_spot_sample_coverage_verdict"],
            "incomplete",
        )
        self.assertFalse(incomplete["strict_completion_stress_gate_passed"])

        missing_resident = {**base, "worker": _healthy_worker(gpu_bytes=None)}
        nonresident = {**base, "worker": _healthy_worker()}
        nonresident["worker"]["expected_running"] = False
        uncompensated = _summarize_turns([missing_resident, nonresident], 2)
        self.assertEqual(uncompensated["worker_expected_turns"], 1)
        self.assertEqual(
            uncompensated["post_turn_gpu_memory_spot_sample_observed_turns"], 0
        )
        self.assertEqual(
            uncompensated["post_turn_gpu_memory_spot_sample_coverage_rate"], 0.0
        )
        self.assertFalse(
            uncompensated["post_turn_gpu_memory_spot_sample_coverage_complete"]
        )
        self.assertEqual(
            uncompensated["post_turn_gpu_memory_spot_sample_coverage_verdict"],
            "unverified",
        )

    def test_turn_record_rejects_observed_gpu_memory_above_limit(self):
        record = _turn_record(
            turn_number=1,
            case=PromptCase("stress-vram", "간결하게 답하세요.", "generated"),
            session_id="completion-a",
            peer_session_id="completion-b",
            state=_complete_state(),
            answer=_complete_answer(),
            elapsed_seconds=1.0,
            worker=_healthy_worker(gpu_bytes=1_200),
            peer_before_digest="a" * 64,
            peer_after_digest="a" * 64,
            prior_answer_digests=set(),
            new_assistant_count=1,
            new_user_count=1,
            observed_user_prompt="간결하게 답하세요.",
        )

        self.assertFalse(record["passed"])
        self.assertFalse(
            record["checks"][
                "post_turn_gpu_memory_spot_sample_within_limit_when_required"
            ]
        )

    def test_turn_record_rejects_substantive_cross_turn_sentence_echo(self):
        prior = (
            "개인정보는 장치 안에서 처리되어 외부 노출을 줄입니다. "
            "네트워크 왕복이 없어 응답 지연도 줄어듭니다."
        )
        echoed = _complete_answer(prior)
        record = _turn_record(
            turn_number=2,
            case=PromptCase("stress-07", "다른 원칙을 답하세요.", "generated"),
            session_id="completion-b",
            peer_session_id="completion-a",
            state=_complete_state(),
            answer=echoed,
            elapsed_seconds=1.0,
            worker=_healthy_worker(),
            peer_before_digest="a" * 64,
            peer_after_digest="a" * 64,
            prior_answer_digests=set(),
            new_assistant_count=1,
            new_user_count=1,
            observed_user_prompt="다른 원칙을 답하세요.",
            prior_sentence_keys=set(_substantive_sentence_keys(prior)),
        )
        self.assertFalse(record["passed"])
        self.assertFalse(record["checks"]["no_cross_turn_sentence_echo"])
        self.assertEqual(len(record["cross_turn_sentence_reuse"]), 2)

    def test_turn_record_rejects_interactive_latency_over_two_minutes(self):
        record = _turn_record(
            turn_number=1,
            case=PromptCase("slow", "간결하게 답하세요.", "generated"),
            session_id="completion-a",
            peer_session_id="completion-b",
            state=_complete_state(),
            answer=_complete_answer(),
            elapsed_seconds=120.001,
            worker=_healthy_worker(),
            peer_before_digest="a" * 64,
            peer_after_digest="a" * 64,
            prior_answer_digests=set(),
            new_assistant_count=1,
            new_user_count=1,
            observed_user_prompt="간결하게 답하세요.",
        )
        self.assertFalse(record["passed"])
        self.assertFalse(record["checks"]["interactive_latency_within_limit"])


if __name__ == "__main__":
    unittest.main()
