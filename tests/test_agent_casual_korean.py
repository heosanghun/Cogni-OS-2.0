from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import scripts.validate_agent_casual_korean as casual
from scripts.gpu5_boundary_guard import GPU5BoundaryError, require_project_gpu_index
from scripts.validate_agent_casual_korean import (
    CASUAL_CASES,
    REQUIRED_CASUAL_CHECKS,
    REQUIRED_CATEGORIES,
    _casual_checks,
    _casual_summary,
    _lexical_diversity,
    build_parser,
)


def _complete_state() -> dict[str, object]:
    return {
        "status": "succeeded",
        "stage": "complete",
        "completion": {"truncated": False},
    }


class TestCasualKoreanReleaseGate(unittest.TestCase):
    def test_exact_reported_sequence_is_first_and_context_bound(self) -> None:
        self.assertEqual(
            [case.prompt for case in CASUAL_CASES[:2]],
            [
                "그럼, 나와함께 재미있는 프로젝트를 합시다.",
                "나와 어떤 일을 함께 할 수 있나요?",
            ],
        )
        self.assertEqual(CASUAL_CASES[0].session, CASUAL_CASES[1].session)
        self.assertEqual(
            [case.expected_mode for case in CASUAL_CASES[:2]],
            ["conversation_fastpath"] * 2,
        )
        self.assertEqual(
            [case.expected_mode for case in CASUAL_CASES[2:]],
            ["cogni_core"] * 7 + ["factbook"],
        )
        covered = {category for case in CASUAL_CASES for category in case.categories}
        self.assertTrue(REQUIRED_CATEGORIES.issubset(covered))

    def test_natural_checks_reject_fallback_and_multiple_assistants(self) -> None:
        case = CASUAL_CASES[0]
        answer = {
            "content": (
                "좋아요. 관심 있는 분야와 만들고 싶은 결과를 정한 뒤, "
                "재미있는 프로젝트를 함께 시작해 봅시다."
            ),
            "finish_reason": "stop",
            "truncated": False,
            "generation_mode": case.expected_mode,
        }
        checks = _casual_checks(
            case,
            answer,
            _complete_state(),
            new_assistant_count=1,
            elapsed_seconds=12.0,
            latency_limit_seconds=120.0,
        )
        self.assertTrue(all(checks.values()), checks)

        answer["generation_mode"] = "quality_fallback"
        fallback = _casual_checks(
            case,
            answer,
            _complete_state(),
            new_assistant_count=2,
            elapsed_seconds=121.0,
            latency_limit_seconds=120.0,
        )
        self.assertFalse(fallback["expected_generation_mode"])
        self.assertFalse(fallback["zero_quality_fallback"])
        self.assertFalse(fallback["exactly_one_assistant"])
        self.assertFalse(fallback["bounded_latency"])

    def test_canned_recovery_and_degenerate_text_are_detected(self) -> None:
        case = CASUAL_CASES[1]
        answer = {
            "content": (
                "로컬 모델의 답변 후보가 품질 검증을 통과하지 못했습니다. "
                "표현을 바꿔 다시 요청해 주세요."
            ),
            "finish_reason": "stop",
            "truncated": False,
            "generation_mode": case.expected_mode,
        }
        checks = _casual_checks(
            case,
            answer,
            _complete_state(),
            new_assistant_count=1,
            elapsed_seconds=1.0,
            latency_limit_seconds=120.0,
        )
        self.assertFalse(checks["no_canned_recovery_copy"])
        self.assertLess(_lexical_diversity("반복 반복 반복 반복 반복 반복"), 0.20)

    def test_keyword_traps_generic_answers_and_short_loops_are_rejected(self) -> None:
        generic = {
            "content": "좋은 생각입니다.",
            "finish_reason": "stop",
            "truncated": False,
            "generation_mode": CASUAL_CASES[0].expected_mode,
        }
        generic_checks = _casual_checks(
            CASUAL_CASES[0],
            generic,
            _complete_state(),
            new_assistant_count=1,
            elapsed_seconds=1.0,
            latency_limit_seconds=120.0,
        )
        self.assertFalse(generic_checks["not_generic_only"])
        self.assertFalse(generic_checks["case_relevance"])

        keyword_trap = dict(generic, content="리스트가 좋습니다.")
        transition_checks = _casual_checks(
            CASUAL_CASES[7],
            keyword_trap,
            _complete_state(),
            new_assistant_count=1,
            elapsed_seconds=1.0,
            latency_limit_seconds=120.0,
        )
        self.assertFalse(transition_checks["case_relevance"])

        negated = dict(
            generic,
            content="함께 재미있는 프로젝트를 만들면 좋지만 시작하지 않겠습니다.",
        )
        negated_checks = _casual_checks(
            CASUAL_CASES[0],
            negated,
            _complete_state(),
            new_assistant_count=1,
            elapsed_seconds=1.0,
            latency_limit_seconds=120.0,
        )
        self.assertFalse(negated_checks["intent_alignment"])

        echoed = dict(generic, content=CASUAL_CASES[0].prompt + " 좋아요.")
        echoed_checks = _casual_checks(
            CASUAL_CASES[0],
            echoed,
            _complete_state(),
            new_assistant_count=1,
            elapsed_seconds=1.0,
            latency_limit_seconds=120.0,
        )
        self.assertFalse(echoed_checks["not_prompt_echo"])

        loop = dict(
            generic,
            content=("프로젝트를 함께 시작합시다. 프로젝트를 함께 시작합시다."),
        )
        loop_checks = _casual_checks(
            CASUAL_CASES[0],
            loop,
            _complete_state(),
            new_assistant_count=1,
            elapsed_seconds=1.0,
            latency_limit_seconds=120.0,
        )
        self.assertFalse(loop_checks["no_short_sentence_loop"])

    def test_partial_sentence_reuse_across_turns_is_rejected(self) -> None:
        repeated_sentence = "프로젝트 목표를 함께 정한 뒤 작은 기능부터 시작하겠습니다."
        answer = {
            "content": repeated_sentence + " 결과를 검증하며 다음 단계로 넘어갑니다.",
            "finish_reason": "stop",
            "truncated": False,
            "generation_mode": CASUAL_CASES[0].expected_mode,
        }
        checks = _casual_checks(
            CASUAL_CASES[0],
            answer,
            _complete_state(),
            new_assistant_count=1,
            elapsed_seconds=1.0,
            latency_limit_seconds=120.0,
            prior_sentence_keys=frozenset({repeated_sentence.rstrip(".").casefold()}),
        )
        self.assertFalse(checks["no_cross_turn_sentence_reuse"])

    def test_factbook_turn_requires_exact_build_model_and_parameters(self) -> None:
        expected = {
            "build_version": "0.3.2",
            "model_label": "gemma4-e4b",
            "stored_parameters": 7_996_157_418,
            "effective_parameters": 4_506_496_490,
        }
        answer = {
            "content": (
                "Cogni Agent의 로컬 백본은 gemma4-e4b이며 저장 파라미터는 "
                "7,996,157,418개, effective 파라미터는 4,506,496,490개입니다. "
                "현재 빌드는 0.3.2이고 검증 상태와 제한은 Runtime Fact-book 기준입니다."
            ),
            "finish_reason": "stop",
            "truncated": False,
            "generation_mode": "factbook",
        }
        checks = _casual_checks(
            CASUAL_CASES[-1],
            answer,
            _complete_state(),
            new_assistant_count=1,
            elapsed_seconds=1.0,
            latency_limit_seconds=120.0,
            expected_factbook=expected,
        )
        self.assertTrue(checks["factbook_model_exact"])
        self.assertTrue(checks["factbook_version_exact"])
        self.assertTrue(checks["factbook_parameters_exact"])

        answer["content"] = str(answer["content"]).replace("0.3.2", "0.3.1")
        wrong = _casual_checks(
            CASUAL_CASES[-1],
            answer,
            _complete_state(),
            new_assistant_count=1,
            elapsed_seconds=1.0,
            latency_limit_seconds=120.0,
            expected_factbook=expected,
        )
        self.assertFalse(wrong["factbook_version_exact"])

    def test_summary_requires_every_turn_and_zero_fallback(self) -> None:
        turns = []
        for index, case in enumerate(CASUAL_CASES):
            turns.append(
                {
                    "case": case.label,
                    "categories": list(case.categories),
                    "generation_mode": case.expected_mode,
                    "elapsed_seconds": 10.0 + index,
                    "checks": {
                        **{name: True for name in REQUIRED_CASUAL_CHECKS},
                        "exactly_one_assistant": True,
                        "korean_complete": True,
                        "lexically_non_degenerate": True,
                        "no_repeated_sentence": True,
                    },
                    "passed": True,
                }
            )
        passed = _casual_summary(turns, latency_limit_seconds=120.0)
        self.assertTrue(passed["strict_casual_gate_passed"])
        self.assertTrue(passed["exact_reproduction_gate_passed"])
        self.assertEqual(passed["quality_fallback_turns"], 0)
        self.assertEqual(passed["fastpath_turns"], 2)
        self.assertTrue(passed["fastpath_ratio_gate_passed"])
        self.assertTrue(passed["checks_schema_gate_passed"])

        missing = turns[3]["checks"].pop("request_contract_fulfilled")
        incomplete_schema = _casual_summary(turns, latency_limit_seconds=120.0)
        self.assertFalse(incomplete_schema["checks_schema_gate_passed"])
        self.assertFalse(incomplete_schema["strict_casual_gate_passed"])
        turns[3]["checks"]["request_contract_fulfilled"] = missing

        turns[2]["generation_mode"] = "conversation_fastpath"
        over_routed = _casual_summary(turns, latency_limit_seconds=120.0)
        self.assertFalse(over_routed["fastpath_ratio_gate_passed"])
        self.assertFalse(over_routed["strict_casual_gate_passed"])
        turns[2]["generation_mode"] = CASUAL_CASES[2].expected_mode

        turns[0]["generation_mode"] = "quality_fallback"
        failed = _casual_summary(turns, latency_limit_seconds=120.0)
        self.assertFalse(failed["quality_fallback_gate_passed"])
        self.assertFalse(failed["strict_casual_gate_passed"])

    def test_cli_uses_bounded_release_latency_default(self) -> None:
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
        args = build_parser().parse_args(required)
        self.assertEqual(args.timeout, 120.0)
        with self.assertRaises(SystemExit):
            build_parser().parse_args([*required, "--timeout", "120.001"])

    def test_cli_and_execution_contract_reject_non5_or_missing_selector(self) -> None:
        with self.assertRaises(SystemExit):
            build_parser().parse_args(["--model", "m", "--manifest", "x"])
        for rejected in (0, 1, 2, 3, 4, 6, 7):
            with self.subTest(rejected=rejected), self.assertRaises(GPU5BoundaryError):
                require_project_gpu_index(rejected)

    def test_identity_precheck_runs_before_manifest_or_model_access(self) -> None:
        args = SimpleNamespace(
            model="m",
            manifest="x",
            physical_gpu_index=5,
            gpu_query_context="gpu5-container",
            timeout=120.0,
            output=None,
        )
        identity = Mock(side_effect=GPU5BoundaryError("identity rejected"))
        manifest = Mock()
        model = Mock()
        with (
            patch.object(casual, "validate_guarded_gpu5_identity", identity),
            patch.object(casual, "verify_artifact_manifest", manifest),
            patch.object(casual.ModelService, "for_local_gemma", model),
        ):
            report, code = casual.execute(args)
        self.assertEqual(code, 1)
        self.assertEqual(report["status"], "failed")
        identity.assert_called_once()
        manifest.assert_not_called()
        model.assert_not_called()

    def test_primary_failure_still_runs_manifest_and_gpu_postchecks(self) -> None:
        args = SimpleNamespace(
            model="m",
            manifest="x",
            physical_gpu_index=5,
            gpu_query_context="gpu5-container",
            timeout=120.0,
            output=None,
        )
        identity_value = SimpleNamespace(
            as_payload=lambda: {
                "physical_index": 5,
                "uuid": "gpu5",
                "logical_device_count": 1,
                "logical_device_index": 0,
            }
        )
        verified = SimpleNamespace(files=())
        identity = Mock(side_effect=[identity_value, identity_value])
        manifest = Mock(side_effect=[verified, verified])
        torch_cuda = SimpleNamespace(
            is_available=lambda: True,
            device_count=lambda: 1,
            current_device=lambda: 0,
            get_device_name=lambda _index: "GPU5",
        )
        with (
            patch.object(casual, "validate_guarded_gpu5_identity", identity),
            patch.object(casual, "verify_artifact_manifest", manifest),
            patch.object(casual.torch, "cuda", torch_cuda),
            patch.object(
                casual,
                "build_runtime_factbook_from_verified",
                side_effect=RuntimeError("primary"),
            ),
        ):
            report, code = casual.execute(args)
        self.assertEqual(code, 1)
        self.assertEqual(identity.call_count, 2)
        self.assertEqual(manifest.call_count, 2)
        self.assertEqual(report["verified_files_after"], 0)
        self.assertIn("gpu_identity_after", report)

    def test_turn_keyboard_interrupt_is_re_raised_after_cleanup(self) -> None:
        args = SimpleNamespace(
            model="m",
            manifest="x",
            physical_gpu_index=5,
            gpu_query_context="gpu5-container",
            timeout=120.0,
            output=None,
        )
        identity_value = SimpleNamespace(
            as_payload=lambda: {
                "physical_index": 5,
                "uuid": "gpu5",
                "logical_device_count": 1,
                "logical_device_index": 0,
            }
        )
        verified = SimpleNamespace(files=())
        factbook = SimpleNamespace(
            model=SimpleNamespace(manifest_sha256="f" * 64),
            as_payload=lambda: {},
        )
        service = SimpleNamespace(is_running=False, stop=Mock())
        shutdown = Mock()

        class InterruptingManager:
            session_id = "casual-interrupt"
            is_active = False

            def __init__(self, *_args, **_kwargs) -> None:
                pass

            def snapshot(self) -> dict[str, object]:
                return {"status": "idle", "conversation": []}

            def start_turn(self, *_args, **_kwargs) -> None:
                raise KeyboardInterrupt("operator stop")

            def shutdown(self) -> None:
                shutdown()

        torch_cuda = SimpleNamespace(
            is_available=lambda: True,
            device_count=lambda: 1,
            current_device=lambda: 0,
            get_device_name=lambda _index: "GPU5",
        )
        with (
            patch.object(
                casual,
                "validate_guarded_gpu5_identity",
                side_effect=[identity_value, identity_value],
            ),
            patch.object(
                casual,
                "verify_artifact_manifest",
                side_effect=[verified, verified],
            ),
            patch.object(casual.torch, "cuda", torch_cuda),
            patch.object(
                casual, "build_runtime_factbook_from_verified", return_value=factbook
            ),
            patch.object(casual, "_expected_factbook_identity", return_value={}),
            patch.object(casual, "RuntimeFactGrounder", return_value=Mock()),
            patch.object(casual.ModelService, "for_local_gemma", return_value=service),
            patch.object(casual, "AgentManager", InterruptingManager),
            self.assertRaisesRegex(KeyboardInterrupt, "operator stop"),
        ):
            casual.execute(args)
        shutdown.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
