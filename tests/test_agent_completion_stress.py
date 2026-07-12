from __future__ import annotations

import os
from types import SimpleNamespace
import unittest

from scripts.validate_agent_completion import (
    DEFAULT_PROMPTS,
    PromptCase,
    _answer_checks,
    _korean_completion_metrics,
    _prompt_cases,
    _read_process_rss_bytes,
    _sample_worker_memory,
    _sentence_repetition_metrics,
    _substantive_sentence_keys,
    _summarize_turns,
    _turn_record,
    _worker_snapshot,
    build_parser,
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
            "worker_rss_bytes": 500,
            "worker_rss_status": "measured",
            "worker_gpu_memory_bytes": gpu_bytes,
            "worker_gpu_memory_status": (
                "measured" if gpu_bytes is not None else "driver_unreported"
            ),
            "gpu_memory_within_limit": (
                None if gpu_bytes is None else gpu_bytes <= 1_000
            ),
            "vram_limit_bytes": 1_000,
            "memory_observed": True,
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
    def test_turn_argument_preserves_four_turn_default_and_accepts_twenty(self):
        parser = build_parser()
        default = parser.parse_args(["--model", "m", "--manifest", "x"])
        stress = parser.parse_args(["--model", "m", "--manifest", "x", "--turns", "20"])
        self.assertEqual(default.turns, 4)
        self.assertEqual(default.timeout, 180.0)
        self.assertEqual(stress.turns, 20)
        for value in ("0", "101", "not-a-number"):
            with self.subTest(value=value), self.assertRaises(SystemExit):
                parser.parse_args(["--model", "m", "--manifest", "x", "--turns", value])

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
        self.assertFalse(rejected_once["strict_turn_gate_passed"])
        self.assertEqual(rejected_once["content_answer_rate"], 0.95)

        two_fallbacks = [dict(item) for item in one_fallback]
        two_fallbacks[1]["generation_mode"] = "quality_fallback"
        rejected_twice = _summarize_turns(two_fallbacks, 20)
        self.assertFalse(rejected_twice["quality_fallback_gate_passed"])
        self.assertFalse(rejected_twice["strict_turn_gate_passed"])

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

    def test_memory_sampler_keeps_gpu_unverified_separate_from_rss(self):
        observed = _sample_worker_memory(
            77,
            vram_limit_bytes=1_000,
            rss_reader=lambda _pid: 500,
            gpu_reader=lambda _pid: (900, "measured"),
        )
        self.assertTrue(observed["memory_observed"])
        self.assertTrue(observed["gpu_memory_within_limit"])

        driver_hidden = _sample_worker_memory(
            77,
            vram_limit_bytes=1_000,
            rss_reader=lambda _pid: 500,
            gpu_reader=lambda _pid: (None, "driver_unreported"),
        )
        self.assertTrue(driver_hidden["memory_observed"])
        self.assertIsNone(driver_hidden["gpu_memory_within_limit"])
        self.assertEqual(driver_hidden["worker_gpu_memory_status"], "driver_unreported")

    def test_current_process_rss_is_observable_without_optional_packages(self):
        self.assertGreater(_read_process_rss_bytes(os.getpid()), 0)

    def test_worker_snapshot_requires_stable_idle_resident_worker(self):
        service = SimpleNamespace(
            is_running=True,
            worker_pid=81,
            active_request_id=None,
        )

        def sampler(pid, *, vram_limit_bytes):
            self.assertEqual(pid, 81)
            self.assertEqual(vram_limit_bytes, 1_000)
            return _healthy_worker()["memory"]

        healthy = _worker_snapshot(
            service,
            expected_running=True,
            stable_pid=81,
            vram_limit_bytes=1_000,
            memory_sampler=sampler,
        )
        self.assertTrue(healthy["healthy"])

        service.active_request_id = 9
        busy = _worker_snapshot(
            service,
            expected_running=True,
            stable_pid=81,
            vram_limit_bytes=1_000,
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
        )
        self.assertFalse(second["passed"])
        self.assertFalse(second["checks"]["no_cross_turn_exact_duplicate"])
        self.assertFalse(second["checks"]["session_isolated"])

        summary = _summarize_turns([first, second], 2)
        self.assertEqual(summary["turn_success_rate"], 0.5)
        self.assertFalse(summary["strict_turn_gate_passed"])
        self.assertEqual(summary["gpu_memory_verdict"], "passed")

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
            prior_sentence_keys=set(_substantive_sentence_keys(prior)),
        )
        self.assertFalse(record["passed"])
        self.assertFalse(record["checks"]["no_cross_turn_sentence_echo"])
        self.assertEqual(len(record["cross_turn_sentence_reuse"]), 2)

    def test_turn_record_rejects_interactive_latency_over_three_minutes(self):
        record = _turn_record(
            turn_number=1,
            case=PromptCase("slow", "간결하게 답하세요.", "generated"),
            session_id="completion-a",
            peer_session_id="completion-b",
            state=_complete_state(),
            answer=_complete_answer(),
            elapsed_seconds=180.001,
            worker=_healthy_worker(),
            peer_before_digest="a" * 64,
            peer_after_digest="a" * 64,
            prior_answer_digests=set(),
        )
        self.assertFalse(record["passed"])
        self.assertFalse(record["checks"]["interactive_latency_within_limit"])


if __name__ == "__main__":
    unittest.main()
