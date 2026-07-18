from __future__ import annotations

from contextlib import nullcontext
import hashlib
from io import StringIO
import json
import os
from pathlib import Path
import socket
import shutil
import stat
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import scripts.gpu5_boundary_guard as guard
from scripts.gpu5_boundary_guard import (
    GPU5AggregateError,
    GPU5BoundaryError,
    GPU5CleanupError,
    GPU5DockerExecutionError,
    ExecutionScope,
    PINNED_DOCKER_IMAGE,
    PROJECT_GPU_UUID,
    SourceSnapshot,
    _evidence_digest,
    _gpu5_project_lease,
    _open_evidence_target,
    _run_nvidia_smi,
    _ensure_container_absent,
    _source_within_allowed_roots,
    _validated_validator_command,
    build_gpu5_docker_argv,
    capture_execution_scope,
    native_gpu5_environment,
    preflight_gpu5,
    query_gpu5_snapshot,
    require_project_gpu_index,
    run_guarded_gpu5_container,
    validate_guarded_gpu5_identity,
    validate_gpu5_docker_argv,
)


EXPECTED_COMMIT = "a" * 40
TEST_NONCE = "0123456789abcdef0123456789abcdef"
TEST_CONTAINER_NAME = "cognios-gpu5-0123456789ab"
TEST_CONTAINER_ID = "b" * 64


def _valid_product_acceptance_payload() -> dict[str, object]:
    identity = {
        "physical_index": 5,
        "uuid": PROJECT_GPU_UUID,
        "query_context": "gpu5-container",
        "logical_device_count": 1,
        "logical_device_index": 0,
    }
    threshold = int(16.7 * 1024**3)
    answers = (
        "Cogni-OS 0.4.0에서 실행되는 gemma4-e4b-it 모델이며, effective 파라미터는 "
        "4,506,496,490개이고 저장 파라미터는 7,996,157,418개입니다.",
        "자가 거울치료는 실패 로그에서 코드 패치 후보를 만들고, 격리 검증을 통과한 제안만 "
        "승격하는 절차입니다.",
        "CTS는 고정 용량 탐색, System 1.5는 Fast Weight, System 2.5는 안정성 방어, "
        "System 3은 희소 전문가, System 4는 텐서 협업을 설계 목표로 두며 각각의 검증 상태를 "
        "별도로 공개합니다. 이상입니다.",
        "실제 검증으로 확인된 사실은 그대로 표시하고, 아직 실측되지 않은 설계 목표는 향후 "
        "검증 대상으로 분리합니다.",
        "오늘은 요구사항을 정리하고 작은 구현부터 함께 확인할 수 있습니다. 결과가 나오면 "
        "근거와 다음 단계를 편안하게 설명하겠습니다.",
        "온디바이스 AI는 로컬 처리로 보안을 높이고 네트워크 없이 빠르게 응답한다는 장점이 "
        "있습니다. 반면 장치의 GPU 메모리와 성능 제약을 직접 관리해야 합니다.",
        "코드 POC에서는 먼저 성공 조건과 실패 조건을 한 문장으로 고정하겠습니다.",
        "검증된 사실은 근거와 함께 적고, 추론이나 판단은 별도 표현으로 구분하는 원칙을 "
        "지킵니다.",
        "1. 생성 길이 경계를 만나면 중단 신호를 기록합니다. 2. 이어진 답변은 앞 문장과 "
        "대조해 중복을 제거합니다. 3. 잘린 문장은 마지막 완결 경계부터 복구합니다. 4. 반복 "
        "루프가 감지되면 즉시 생성을 끝냅니다. 5. 최종 문장과 종료 사유가 모두 완결됐는지 "
        "검증합니다.",
        "좋은 요약문은 원문의 핵심을 보존합니다. 불필요한 수식어를 줄여 간결하게 씁니다. "
        "같은 의미의 반복은 제거합니다.",
        "먼저 사용자의 사실 정정을 수용하고 원자료를 확인합니다. 다음으로 잘못된 내용을 "
        "수정한 뒤 검증 결과를 반영합니다.",
        "수정 전에는 원본을 백업합니다. 변경안은 격리된 테스트로 검증합니다. 문제가 생기면 "
        "준비한 롤백 절차로 복구합니다.",
        "예외나 오류의 재시도 횟수와 시간 한도를 먼저 정하고, 한도를 넘으면 안전하게 "
        "종료합니다.",
        "개인정보는 필요한 범위만 로컬 장치에서 처리합니다. 오프라인 저장 자료에는 접근 "
        "통제와 삭제 기준을 적용합니다.",
        "GPU 메모리 측정값은 현재 실행에서 관찰한 사실이고, 설계 목표는 별도 시험으로 "
        "입증해야 하는 기준입니다.",
        "도구 실행 결과를 직접 확인하지 않았다면 성공을 검증할 근거가 없으므로 완료라고 "
        "말해서는 안 됩니다.",
        "오류의 원인을 먼저 기록하고 수정 내용을 연결한 뒤, 같은 실패를 막는 회귀 테스트 "
        "결과를 남깁니다.",
        "오래된 대화 문맥은 핵심 결정만 요약하고, 현재 사용자 의도와 제약은 원문에 가깝게 "
        "보존합니다.",
        "근거가 부족해 불확실한 부분은 추측이라고 밝히고, 사실처럼 단정하지 않으며 확인 "
        "방법을 제시합니다.",
        "자연스러운 한국어 답변은 각 문장이 완결된 종결 표현을 가져야 합니다. 같은 문장이나 "
        "문단의 반복이 없어야 합니다. 마지막 문법과 문장 부호까지 확인합니다.",
    )
    turns: list[dict[str, object]] = []
    for turn_number, (case, prompt, answer) in enumerate(
        zip(
            guard._PRODUCT_ACCEPTANCE_CASES,
            guard._PRODUCT_ACCEPTANCE_PROMPTS,
            answers,
            strict=True,
        ),
        1,
    ):
        worker_expected = turn_number >= 5
        factbook_turn = turn_number in guard._PRODUCT_FACTBOOK_TURNS
        session_id = (
            "completion-a"
            if turn_number <= 4 or turn_number % 2 == 0
            else "completion-b"
        )
        peer_session_id = (
            "completion-b" if session_id == "completion-a" else "completion-a"
        )
        answer_sha256 = hashlib.sha256(answer.casefold().encode("utf-8")).hexdigest()
        memory = {
            "sample_scope": "post_turn_spot_sample",
            "captures_peak": False,
            "gpu_memory_spot_sample_bytes": 1024 if worker_expected else None,
            "gpu_memory_spot_sample_status": (
                "measured_aggregate" if worker_expected else "worker_not_started"
            ),
            "gpu_memory_spot_sample_threshold_bytes": threshold,
            "gpu_memory_spot_sample_within_threshold": (
                True if worker_expected else None
            ),
            "spot_sample_observed": worker_expected,
        }
        turns.append(
            {
                "turn": turn_number,
                "case": case,
                "prompt": prompt,
                "observed_user_prompt": prompt,
                "new_user_count": 1,
                "passed": True,
                "generation_mode": "factbook" if factbook_turn else "cogni_core",
                "expected_route": "grounded" if factbook_turn else "generated",
                "session_id": session_id,
                "peer_session_id": peer_session_id,
                "answer": answer,
                "answer_sha256": answer_sha256,
                "state_status": "succeeded",
                "state_stage": "complete",
                "answer_truncated": False,
                "generated_tokens": 0 if factbook_turn else 10,
                "new_assistant_count": 1,
                "continuations": 1 if turn_number == 9 else 0,
                "elapsed_seconds": 1.0,
                "explicit_truncation": False,
                "empty_answer": False,
                "finish_reason": "stop",
                "repetition": guard._product_repetition_metrics(answer),
                "role_token_leaks": [],
                "control_marker_leaks": [],
                "cross_turn_exact_duplicate": False,
                "cross_turn_sentence_reuse": [],
                "korean_completion": guard._product_korean_completion_metrics(answer),
                "checks": {key: True for key in guard._PRODUCT_REQUIRED_TURN_CHECKS},
                "worker": {
                    "healthy": True,
                    "pid_stable": True,
                    "active_request_id": None,
                    "expected_running": worker_expected,
                    "running": worker_expected,
                    "pid": 1234 if worker_expected else None,
                    "memory": memory,
                },
                "session_isolation": {
                    "peer_conversation_before_sha256": "a" * 64,
                    "peer_conversation_after_sha256": "a" * 64,
                    "peer_unchanged": True,
                },
            }
        )
    summary = {
        "requested_turns": 20,
        "completed_turns": 20,
        "passed_turns": 20,
        "failed_turns": 0,
        "turn_success_rate": 1.0,
        "quality_fallback_turns": 0,
        "allowed_quality_fallback_turns": 0,
        "quality_fallback_gate_passed": True,
        "content_answer_rate": 1.0,
        "failed_check_counts": {},
        "worker_expected_turns": 16,
        "resident_worker_pids": [1234],
        "single_resident_worker_scope": True,
        "post_turn_gpu_memory_spot_sample_observed_turns": 16,
        "post_turn_gpu_memory_spot_sample_coverage_rate": 1.0,
        "maximum_observed_post_turn_gpu_memory_spot_sample_bytes": 1024,
        "post_turn_gpu_memory_spot_samples_over_threshold": 0,
        "post_turn_gpu_memory_spot_sample_coverage_verdict": "complete",
        "post_turn_gpu_memory_spot_sample_threshold_observation": (
            "observed_at_or_below_threshold"
        ),
        "post_turn_gpu_memory_spot_sample_coverage_complete": True,
        "post_turn_gpu_memory_spot_sample_threshold_gate_passed": True,
        "recommended_stress_schedule_completed": True,
        "strict_completion_stress_gate_passed": True,
    }
    return {
        "schema": "cogni.agent.completion.stress.v2",
        "suite": "product-e4b-it-20",
        "status": "passed",
        "all_checks_passed": True,
        "requested_turns": 20,
        "recommended_stress_turns": 20,
        "completed_turns": 20,
        "worker_cleaned": True,
        "gpu_lease_released": True,
        "cleanup_checks": {
            "worker_cleaned": True,
            "gpu_lease_released": True,
        },
        "verified_files": 7,
        "verified_files_after": 7,
        "physical_gpu_index": 5,
        "gpu_query_context": "gpu5-container",
        "logical_cuda_device_count": 1,
        "logical_cuda_device_index": 0,
        "gpu_identity_before": dict(identity),
        "gpu_identity_after": dict(identity),
        "factbook": {
            "schema_version": 1,
            "build_version": "0.4.0",
            "device": "Test GPU5",
            "target_device": "RTX 4090 24GB",
            "model": {
                "label": "gemma4-e4b-it",
                "dense": True,
                "stored_parameters": 7_996_157_418,
                "effective_parameters": 4_506_496_490,
                "manifest_sha256": "b" * 64,
                "config_sha256": "c" * 64,
            },
        },
        "gpu_lease_history": [
            {"epoch": 1, "purpose": "inference", "reason": "released"}
        ],
        "memory_evidence_scope": {
            "kind": "post_turn_spot_sample",
            "one_sample_per_expected_resident_turn": True,
            "captures_peak": False,
            "captures_sustained_usage": False,
            "gpu_memory_spot_sample_threshold_bytes": threshold,
            "full_runtime_peak_validator": "scripts/validate_gemma4_runtime.py",
            "full_runtime_peak_metric": "torch.cuda.max_memory_allocated",
        },
        "turns": turns,
        "summary": summary,
    }


def _completed(
    *, stdout: str = "", stderr: str = "", returncode: int = 0
) -> SimpleNamespace:
    return SimpleNamespace(stdout=stdout, stderr=stderr, returncode=returncode)


def _idle_smi_responses() -> list[SimpleNamespace]:
    return [
        _completed(stdout=f"5, {PROJECT_GPU_UUID}, 0, 19, 49140\n"),
        _completed(stdout=""),
    ]


def _inspect_row(
    *,
    name: str = TEST_CONTAINER_NAME,
    container_id: str = TEST_CONTAINER_ID,
    labels: dict[str, str] | None = None,
) -> SimpleNamespace:
    label_map = (
        {
            "io.cognios.guard": "gpu5",
            "io.cognios.source-commit": EXPECTED_COMMIT,
            "io.cognios.launch-nonce": TEST_NONCE,
            "io.cognios.execution-profile": "release",
            "io.cognios.validation-artifact-profile": "base-canary",
        }
        if labels is None
        else labels
    )
    return _completed(
        stdout=f"{container_id}\t/{name}\t{json.dumps(label_map, sort_keys=True)}\n"
    )


class _RecordedRunner:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls: list[tuple[tuple[str, ...], dict[str, object]]] = []

    def __call__(self, argv, **kwargs):
        self.calls.append((tuple(argv), dict(kwargs)))
        stream = kwargs.get("stdout")
        if hasattr(stream, "write"):
            stream.write(b'{"status":"passed"}\n')
        if not self.responses:
            raise AssertionError("unexpected subprocess invocation")
        response = self.responses.pop(0)
        if isinstance(response, BaseException):
            raise response
        return response


class TestTrustedMetadataPrimitives(unittest.TestCase):
    def test_validation_artifact_profiles_are_immutable_and_exact(self) -> None:
        base = guard._validation_artifact_profile("base-canary")
        product = guard._validation_artifact_profile("product-e4b-it")
        self.assertEqual(base.raw_model_root, Path("/home/shoon/models/gemma4-e4b"))
        self.assertEqual(base.container_model_root, Path("/models/gemma4-e4b"))
        self.assertEqual(base.manifest_relative_path, "config/gemma4-e4b.manifest.toml")
        self.assertEqual(
            product.raw_model_root,
            Path("/home/shoon/models/gemma4-e4b-it"),
        )
        self.assertEqual(
            product.container_model_root,
            Path("/models/gemma4-e4b-it"),
        )
        self.assertEqual(
            product.manifest_relative_path,
            "config/gemma4-e4b-it.manifest.toml",
        )
        with self.assertRaises(TypeError):
            guard._VALIDATION_ARTIFACT_PROFILES["foreign"] = product
        with self.assertRaises(GPU5BoundaryError):
            guard._validation_artifact_profile("foreign")

    def test_product_manifest_is_exactly_seven_it_files_not_base_six(self) -> None:
        repository = Path(guard.__file__).resolve().parents[1]
        with (
            patch.object(guard, "_effective_uid", return_value=0),
            patch.object(guard, "_group_or_world_writable", return_value=False),
        ):
            product_entries, product_identity = guard._strict_model_manifest_entries(
                repository / "config/gemma4-e4b-it.manifest.toml"
            )
            base_entries, base_identity = guard._strict_model_manifest_entries(
                repository / "config/gemma4-e4b.manifest.toml"
            )

        self.assertEqual(len(product_entries), 7)
        self.assertEqual(len(base_entries), 6)
        self.assertIn("chat_template.jinja", dict(product_entries))
        self.assertNotIn("chat_template.jinja", dict(base_entries))
        self.assertIsNotNone(product_identity)
        self.assertEqual(product_identity.role, "instruction_tuned")
        self.assertIsNone(base_identity)

    def test_product_release_command_is_exact_20_turn_json_component(self) -> None:
        command = (
            "-I",
            "-B",
            "/workspace/scripts/validate_agent_completion.py",
            "--model",
            "/models/gemma4-e4b-it",
            "--manifest",
            "/workspace/config/gemma4-e4b-it.manifest.toml",
            "--physical-gpu-index",
            "5",
            "--gpu-query-context",
            "gpu5-container",
            "--turns",
            "20",
            "--suite",
            "product-e4b-it-20",
            "--strict-json",
        )
        self.assertEqual(
            guard._validated_release_validator_command(command, "product-e4b-it"),
            command,
        )
        for replacement in ("19", "21"):
            mutated = list(command)
            mutated[mutated.index("--turns") + 1] = replacement
            with self.subTest(turns=replacement), self.assertRaises(GPU5BoundaryError):
                guard._validated_release_validator_command(
                    mutated,
                    "product-e4b-it",
                )
        without_strict = tuple(token for token in command if token != "--strict-json")
        with self.assertRaises(GPU5BoundaryError):
            guard._validated_release_validator_command(
                without_strict,
                "product-e4b-it",
            )
        with self.assertRaises(GPU5BoundaryError):
            guard._validated_release_validator_command(command, "base-canary")

    def test_product_casual_diagnostic_is_profile_bound_but_not_a_release_gate(
        self,
    ) -> None:
        command = (
            "-I",
            "-B",
            "/workspace/scripts/validate_agent_casual_korean.py",
            "--model",
            "/models/gemma4-e4b-it",
            "--manifest",
            "/workspace/config/gemma4-e4b-it.manifest.toml",
            "--physical-gpu-index",
            "5",
            "--gpu-query-context",
            "gpu5-container",
        )
        self.assertEqual(
            guard._validated_validator_command(command, "product-e4b-it"),
            command,
        )
        with self.assertRaisesRegex(GPU5BoundaryError, "single 20-turn"):
            guard._validated_release_validator_command(command, "product-e4b-it")
        with self.assertRaises(GPU5BoundaryError):
            guard._validated_validator_command(command, "base-canary")

    def test_product_acceptance_payload_requires_one_passed_20_turn_schema(
        self,
    ) -> None:
        payload = _valid_product_acceptance_payload()
        self.assertEqual(
            guard._validate_product_acceptance_payload(payload),
            "cogni.agent.completion.stress.v2",
        )
        for key, value in (
            ("requested_turns", 19),
            ("status", "failed"),
            ("all_checks_passed", False),
        ):
            rejected = dict(payload)
            rejected[key] = value
            with self.subTest(key=key), self.assertRaises(GPU5BoundaryError):
                guard._validate_product_acceptance_payload(rejected)

        encoded = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.assertEqual(guard._decode_product_acceptance_json(encoded), payload)
        self.assertEqual(
            guard._validate_product_acceptance_evidence(encoded),
            "cogni.agent.completion.stress.v2",
        )
        self.assertEqual(
            guard._validate_product_acceptance_evidence(
                encoded,
                expected_model_manifest_sha256="b" * 64,
            ),
            "cogni.agent.completion.stress.v2",
        )
        with self.assertRaises(GPU5BoundaryError):
            guard._validate_product_acceptance_evidence(
                encoded,
                expected_model_manifest_sha256="d" * 64,
            )
        with self.assertRaisesRegex(GPU5BoundaryError, "one strict UTF-8 JSON"):
            guard._decode_product_acceptance_json(b"turn 1/20 passed\n" + encoded)

        mutations = (
            ("empty turns", lambda candidate: candidate.__setitem__("turns", [])),
            (
                "worker cleanup",
                lambda candidate: candidate.__setitem__("worker_cleaned", False),
            ),
            (
                "GPU identity",
                lambda candidate: candidate["gpu_identity_after"].__setitem__(
                    "uuid", "GPU-wrong"
                ),
            ),
            (
                "turn check",
                lambda candidate: candidate["turns"][8]["checks"].__setitem__(
                    "complete", False
                ),
            ),
            (
                "canonical prompt substitution",
                lambda candidate: candidate["turns"][8].__setitem__(
                    "prompt", "다른 질문입니다."
                ),
            ),
            (
                "raw user prompt substitution",
                lambda candidate: candidate["turns"][8].__setitem__(
                    "observed_user_prompt", "다른 질문입니다."
                ),
            ),
            (
                "multiple assistant messages",
                lambda candidate: candidate["turns"][8].__setitem__(
                    "new_assistant_count", 2
                ),
            ),
            (
                "continuation probe not exercised",
                lambda candidate: candidate["turns"][8].__setitem__("continuations", 0),
            ),
            (
                "sparse self-asserted checks",
                lambda candidate: candidate["turns"][8].__setitem__(
                    "checks", {"succeeded": True}
                ),
            ),
            (
                "duplicate answer",
                lambda candidate: (
                    candidate["turns"][8].__setitem__(
                        "answer", candidate["turns"][7]["answer"]
                    ),
                    candidate["turns"][8].__setitem__(
                        "answer_sha256", candidate["turns"][7]["answer_sha256"]
                    ),
                ),
            ),
            (
                "self-asserted repetition metadata",
                lambda candidate: (
                    candidate["turns"][8].__setitem__(
                        "answer",
                        f"{candidate['turns'][8]['answer']} "
                        f"{candidate['turns'][8]['answer']}",
                    ),
                    candidate["turns"][8].__setitem__(
                        "answer_sha256",
                        hashlib.sha256(
                            candidate["turns"][8]["answer"].casefold().encode("utf-8")
                        ).hexdigest(),
                    ),
                ),
            ),
            (
                "resident PID",
                lambda candidate: candidate["turns"][8]["worker"].__setitem__(
                    "pid", 5678
                ),
            ),
            (
                "GPU threshold",
                lambda candidate: candidate["turns"][8]["worker"]["memory"].__setitem__(
                    "gpu_memory_spot_sample_bytes", threshold + 1
                ),
            ),
            (
                "summary mismatch",
                lambda candidate: candidate["summary"].__setitem__("passed_turns", 19),
            ),
        )
        for label, mutate in mutations:
            candidate = json.loads(json.dumps(payload))
            threshold = int(16.7 * 1024**3)
            mutate(candidate)
            with self.subTest(mutation=label), self.assertRaises(GPU5BoundaryError):
                guard._validate_product_acceptance_payload(candidate)

        with self.assertRaisesRegex(GPU5BoundaryError, "one strict UTF-8 JSON"):
            guard._decode_product_acceptance_json(b'{"elapsed_seconds":NaN}')
        with self.assertRaisesRegex(GPU5BoundaryError, "one strict UTF-8 JSON"):
            guard._decode_product_acceptance_json(b'{"status":"passed","status":"x"}')

    def test_scheduler_reservation_payload_is_exact_gpu5_commit_and_time_bound(self):
        now_ns = 2_000_000_000_000
        payload = {
            "schema": guard.GPU5_SCHEDULER_RESERVATION_SCHEMA,
            "status": "reserved",
            "physical_gpu_index": 5,
            "gpu_uuid": PROJECT_GPU_UUID,
            "source_commit": EXPECTED_COMMIT,
            "subject_uid": 1234,
            "reservation_id": "cognios-gpu5-0123456789abcdef",
            "issued_unix_ns": now_ns - 1_000_000,
            "expires_unix_ns": now_ns + 1_000_000,
        }
        self.assertEqual(
            guard._validate_gpu5_scheduler_reservation_payload(
                payload,
                expected_source_commit=EXPECTED_COMMIT,
                effective_uid=1234,
                now_ns=now_ns,
                minimum_remaining_ns=500_000,
            ),
            payload,
        )
        mutations = (
            ("wrong GPU", {"physical_gpu_index": 4}),
            ("wrong UUID", {"gpu_uuid": "GPU-wrong"}),
            ("wrong commit", {"source_commit": "b" * 40}),
            ("wrong subject", {"subject_uid": 1235}),
            ("expired", {"expires_unix_ns": now_ns}),
            ("future issue", {"issued_unix_ns": now_ns + 1}),
            (
                "overbroad window",
                {
                    "issued_unix_ns": now_ns - 1,
                    "expires_unix_ns": now_ns
                    + guard.MAX_GPU5_SCHEDULER_RESERVATION_WINDOW_NS
                    + 1,
                },
            ),
            ("not reserved", {"status": "released"}),
        )
        for label, replacement in mutations:
            candidate = {**payload, **replacement}
            with self.subTest(label=label), self.assertRaises(GPU5BoundaryError):
                guard._validate_gpu5_scheduler_reservation_payload(
                    candidate,
                    expected_source_commit=EXPECTED_COMMIT,
                    effective_uid=1234,
                    now_ns=now_ns,
                    minimum_remaining_ns=500_000,
                )
        with self.assertRaises(GPU5BoundaryError):
            guard._validate_gpu5_scheduler_reservation_payload(
                {**payload, "extra": True},
                expected_source_commit=EXPECTED_COMMIT,
                effective_uid=1234,
                now_ns=now_ns,
                minimum_remaining_ns=500_000,
            )
        with self.assertRaises(GPU5BoundaryError):
            guard._validate_gpu5_scheduler_reservation_payload(
                payload,
                expected_source_commit=EXPECTED_COMMIT,
                effective_uid=1234,
                now_ns=now_ns,
                minimum_remaining_ns=2_000_000,
            )

    def test_run_cli_blocks_before_gpu_when_scheduler_reservation_is_missing(self):
        output = StringIO()
        arguments = [
            "run",
            "--image",
            PINNED_DOCKER_IMAGE,
            "--expected-source-commit",
            EXPECTED_COMMIT,
            "--timeout",
            "60",
            "--evidence-filename",
            "gpu5-scheduler-blocked.jsonl",
            "--",
            "/usr/local/bin/python",
            "-I",
        ]
        with (
            patch.object(guard, "_validate_run_bootstrap"),
            patch.object(
                guard,
                "_require_external_gpu5_scheduler_reservation",
                side_effect=GPU5BoundaryError("external scheduler reservation missing"),
            ) as reservation,
            patch.object(guard, "run_guarded_gpu5_container") as guarded_run,
            patch.object(guard, "preflight_gpu5") as preflight,
            patch("sys.stdout", output),
        ):
            self.assertEqual(guard.main(arguments), 2)
        reservation.assert_called_once_with(
            EXPECTED_COMMIT,
            required_run_seconds=60.0,
        )
        guarded_run.assert_not_called()
        preflight.assert_not_called()
        payload = json.loads(output.getvalue())
        self.assertEqual(payload["status"], "failed_closed")

    def test_trusted_owner_policy_accepts_only_root_or_effective_uid(self) -> None:
        with patch.object(guard, "_effective_uid", return_value=1234):
            self.assertTrue(guard._trusted_owner_uid(0))
            self.assertTrue(guard._trusted_owner_uid(1234))
            self.assertFalse(guard._trusted_owner_uid(1235))
            self.assertFalse(guard._trusted_owner_uid(True))

    def test_native_snapshot_dataclass_contract_is_exact(self) -> None:
        self.assertEqual(
            tuple(guard.ModelSnapshot.__dataclass_fields__),
            (
                "root_path",
                "root_device",
                "root_inode",
                "root_mode",
                "file_count",
                "total_bytes",
                "manifest_sha256",
                "content_digest",
                "identity_digest",
            ),
        )
        self.assertEqual(
            tuple(guard.NativeExecutionSnapshot.__dataclass_fields__),
            ("source", "model", "manifest_path", "workspace_root"),
        )
        self.assertEqual(guard.MAX_SOURCE_SNAPSHOTS, 64)
        self.assertEqual(guard.MAX_SOURCE_SNAPSHOT_STORE_BYTES, 8 * 1024**3)
        self.assertEqual(guard.MAX_MODEL_SNAPSHOTS, 3)
        self.assertEqual(guard.MAX_MODEL_SNAPSHOT_STORE_BYTES, 96 * 1024**3)

    @unittest.skipUnless(os.name == "posix", "Linux import trust contract")
    def test_public_import_directory_gate_rejects_writable_target(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary) / "imports"
            target.mkdir(mode=0o700)
            target.chmod(0o770)
            with self.assertRaisesRegex(GPU5BoundaryError, "writable"):
                guard.validate_trusted_import_directory(target)

    @unittest.skipUnless(os.name == "posix", "Linux import trust contract")
    def test_public_import_directory_gate_rejects_symlink(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            target = root / "real-imports"
            target.mkdir(mode=0o700)
            alias = root / "alias-imports"
            alias.symlink_to(target, target_is_directory=True)
            with self.assertRaisesRegex(GPU5BoundaryError, "real directory"):
                guard.validate_trusted_import_directory(alias)

    def test_tampered_native_handoff_is_rejected_before_preflight(self) -> None:
        source = SourceSnapshot(
            source_commit=EXPECTED_COMMIT,
            launch_nonce=TEST_NONCE,
            root_path="/sealed/source",
            root_device=10,
            root_inode=11,
            root_mode=0o555,
            file_count=1,
            content_digest="1" * 64,
            identity_digest="2" * 64,
        )
        model = guard.ModelSnapshot(
            root_path="/sealed/model",
            root_device=20,
            root_inode=21,
            root_mode=0o555,
            file_count=1,
            total_bytes=1024,
            manifest_sha256="3" * 64,
            content_digest="4" * 64,
            identity_digest="5" * 64,
        )
        execution_snapshot = guard.NativeExecutionSnapshot(
            source=source,
            model=model,
            manifest_path="/sealed/source/config/model.toml",
            workspace_root="/trusted/workspace",
        )
        environment = {
            "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
            "CUDA_VISIBLE_DEVICES": PROJECT_GPU_UUID,
            "NVIDIA_VISIBLE_DEVICES": PROJECT_GPU_UUID,
        }
        with (
            patch.dict(os.environ, environment, clear=True),
            patch.object(
                guard,
                "gpu5_project_lease",
                return_value=nullcontext(SimpleNamespace()),
            ),
            patch.object(
                guard,
                "_capture_native_execution_snapshot",
                return_value=execution_snapshot,
            ) as capture,
            patch.object(guard, "preflight_gpu5") as preflight,
            self.assertRaisesRegex(GPU5BoundaryError, "handoff"),
        ):
            with guard.native_gpu5_server_authority(
                EXPECTED_COMMIT,
                physical_gpu_index=5,
                gpu_query_context="native-host",
                gpu_uuid=PROJECT_GPU_UUID,
                source_snapshot_root=source.root_path,
                source_snapshot_nonce=source.launch_nonce,
                model_snapshot_root=model.root_path,
                model_manifest_path=execution_snapshot.manifest_path,
                model_manifest_sha256=model.manifest_sha256,
                workspace_root=execution_snapshot.workspace_root,
                source_content_digest=source.content_digest,
                source_identity_digest="f" * 64,
                source_file_count=source.file_count,
                source_root_device=source.root_device,
                source_root_inode=source.root_inode,
                model_content_digest=model.content_digest,
                model_identity_digest=model.identity_digest,
                model_file_count=model.file_count,
                model_root_device=model.root_device,
                model_root_inode=model.root_inode,
                model_total_bytes=model.total_bytes,
            ):
                self.fail("tampered native handoff was accepted")
        capture.assert_called_once()
        preflight.assert_not_called()


@unittest.skipUnless(
    hasattr(socket, "AF_UNIX"),
    "Linux/Unix-domain socket contract; pure guard tests remain cross-platform",
)
class TestGPU5BoundaryGuard(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        base = Path(temporary.name)
        self.repo_root = base / "Cogni-OS-2.0-v041"
        self.model_root = base / "gemma4-e4b"
        for directory in (self.repo_root, self.model_root):
            directory.mkdir()
        self.state_parent = base / "home"
        self.state_parent.mkdir(mode=0o700)
        self.state_root = self.state_parent / ".cognios-gpu5-guard"
        self.docker_config = self.state_root / "docker-empty-config"
        self.lease_path = self.state_root / "gpu5-project.lock"
        self.evidence_root = self.state_root / "evidence"
        self.snapshot_root = self.state_root / "source-snapshots"
        self.model_snapshot_root = self.state_root / "model-snapshots"
        self.state_root.mkdir(mode=0o700)
        self.docker_config.mkdir(mode=0o700)
        self.evidence_root.mkdir(mode=0o700)
        self.snapshot_root.mkdir(mode=0o700)
        self.model_snapshot_root.mkdir(mode=0o700)
        self.snapshot_path = (
            self.snapshot_root / f"source-{EXPECTED_COMMIT}-{TEST_NONCE}"
        )
        self.snapshot_path.mkdir(mode=0o700)
        self.model_snapshot_manifest_sha256 = "5" * 64
        self.model_snapshot_path = self.model_snapshot_root / (
            f"model-{self.model_snapshot_manifest_sha256}-{TEST_NONCE}"
        )
        self.model_snapshot_path.mkdir(mode=0o700)
        (self.model_snapshot_path / "model.bin").write_bytes(b"x")

        executable_root = base / "bin"
        executable_root.mkdir()
        self.nvidia_smi = executable_root / "nvidia-smi"
        self.docker = executable_root / "docker"
        self.git = executable_root / "git"
        for executable in (self.nvidia_smi, self.docker, self.git):
            executable.write_text("#!/bin/sh\nexit 127\n", encoding="ascii")
            executable.chmod(0o755)

        self.docker_socket_path = base / "docker.sock"
        self.docker_socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.docker_socket.bind(str(self.docker_socket_path))
        self.addCleanup(self.docker_socket.close)
        expected_mounts = {
            self.repo_root: Path("/workspace"),
            self.model_root: Path("/models/gemma4-e4b"),
        }
        patchers = (
            patch.object(guard, "EVIDENCE_HOST_ROOT", self.evidence_root),
            patch.object(guard, "SOURCE_SNAPSHOT_ROOT", self.snapshot_root),
            patch.object(guard, "MODEL_SNAPSHOT_ROOT", self.model_snapshot_root),
            patch.object(
                guard,
                "_ALLOWED_MOUNT_ROOTS",
                (self.repo_root, self.model_root),
            ),
            patch.object(guard, "_EXPECTED_READ_ONLY_MOUNTS", expected_mounts),
            patch.object(guard, "GUARD_STATE_PARENT", self.state_parent),
            patch.object(guard, "GUARD_STATE_ROOT", self.state_root),
            patch.object(guard, "DOCKER_CONFIG_ROOT", self.docker_config),
            patch.object(guard, "GPU5_LEASE_PATH", self.lease_path),
            patch.object(guard, "NVIDIA_SMI_EXECUTABLE", self.nvidia_smi),
            patch.object(guard, "DOCKER_EXECUTABLE", self.docker),
            patch.object(guard, "GIT_EXECUTABLE", self.git),
            patch.object(guard, "DOCKER_SOCKET_PATH", self.docker_socket_path),
            patch.object(
                guard,
                "DOCKER_HOST_URI",
                f"unix://{self.docker_socket_path}",
            ),
        )
        for patcher in patchers:
            patcher.start()
            self.addCleanup(patcher.stop)

    @property
    def environment(self) -> dict[str, str]:
        return dict(guard._REQUIRED_CONTAINER_ENVIRONMENT)

    @property
    def mounts(self) -> tuple[str, str]:
        return (
            f"{self.snapshot_path}:/workspace:ro",
            f"{self.model_snapshot_path}:/models/gemma4-e4b:ro",
        )

    def source_snapshot(self, nonce: str = TEST_NONCE) -> SourceSnapshot:
        snapshot_path = self.snapshot_root / f"source-{EXPECTED_COMMIT}-{nonce}"
        snapshot_path.mkdir(mode=0o700, exist_ok=True)
        snapshot_path.chmod(0o555)
        metadata = snapshot_path.stat()
        return SourceSnapshot(
            source_commit=EXPECTED_COMMIT,
            launch_nonce=nonce,
            root_path=str(snapshot_path),
            root_device=int(metadata.st_dev),
            root_inode=int(metadata.st_ino),
            root_mode=stat.S_IMODE(metadata.st_mode),
            file_count=0,
            content_digest="1" * 64,
            identity_digest="2" * 64,
        )

    def model_snapshot(self, nonce: str = TEST_NONCE) -> guard.ModelSnapshot:
        snapshot_path = self.model_snapshot_root / (
            f"model-{self.model_snapshot_manifest_sha256}-{nonce}"
        )
        snapshot_path.mkdir(mode=0o700, exist_ok=True)
        artifact = snapshot_path / "model.bin"
        if not artifact.exists():
            artifact.write_bytes(b"x")
        artifact.chmod(0o444)
        snapshot_path.chmod(0o555)
        metadata = snapshot_path.stat()
        return guard.ModelSnapshot(
            root_path=str(snapshot_path),
            root_device=int(metadata.st_dev),
            root_inode=int(metadata.st_ino),
            root_mode=stat.S_IMODE(metadata.st_mode),
            file_count=1,
            total_bytes=1,
            manifest_sha256=self.model_snapshot_manifest_sha256,
            content_digest="6" * 64,
            identity_digest="7" * 64,
        )

    def native_execution_snapshot(self) -> guard.NativeExecutionSnapshot:
        source = self.source_snapshot()
        digest = "5" * 64
        model_metadata = self.model_snapshot_root.stat()
        model = guard.ModelSnapshot(
            root_path=str(self.model_snapshot_root),
            root_device=int(model_metadata.st_dev),
            root_inode=int(model_metadata.st_ino),
            root_mode=stat.S_IMODE(model_metadata.st_mode),
            file_count=1,
            total_bytes=1,
            manifest_sha256=digest,
            content_digest="6" * 64,
            identity_digest="7" * 64,
        )
        return guard.NativeExecutionSnapshot(
            source=source,
            model=model,
            manifest_path=str(Path(source.root_path) / "config" / "model.toml"),
            workspace_root=str(self.repo_root),
        )

    @staticmethod
    def native_handoff_arguments(
        snapshot: guard.NativeExecutionSnapshot,
    ) -> dict[str, object]:
        return {
            "source_content_digest": snapshot.source.content_digest,
            "source_identity_digest": snapshot.source.identity_digest,
            "source_file_count": snapshot.source.file_count,
            "source_root_device": snapshot.source.root_device,
            "source_root_inode": snapshot.source.root_inode,
            "model_content_digest": snapshot.model.content_digest,
            "model_identity_digest": snapshot.model.identity_digest,
            "model_file_count": snapshot.model.file_count,
            "model_root_device": snapshot.model.root_device,
            "model_root_inode": snapshot.model.root_inode,
            "model_total_bytes": snapshot.model.total_bytes,
        }

    def runtime_command(self, *extra: str) -> tuple[str, ...]:
        return (
            "-I",
            "-B",
            "/workspace/scripts/validate_gemma4_runtime.py",
            "--model",
            "/models/gemma4-e4b",
            "--manifest",
            "/workspace/config/gemma4-e4b.manifest.toml",
            "--physical-gpu-index",
            "5",
            "--gpu-query-context",
            "gpu5-container",
            *extra,
        )

    def completion_command(self, *extra: str) -> tuple[str, ...]:
        return (
            "-I",
            "-B",
            "/workspace/scripts/validate_agent_completion.py",
            "--model",
            "/models/gemma4-e4b",
            "--manifest",
            "/workspace/config/gemma4-e4b.manifest.toml",
            "--physical-gpu-index",
            "5",
            "--gpu-query-context",
            "gpu5-container",
            *extra,
        )

    def docker_argv(self, command: tuple[str, ...] | None = None) -> tuple[str, ...]:
        return build_gpu5_docker_argv(
            PINNED_DOCKER_IMAGE,
            command or self.runtime_command(),
            expected_source_commit=EXPECTED_COMMIT,
            source_snapshot=self.source_snapshot(),
            model_snapshot=self.model_snapshot(),
            environment=self.environment,
            mounts=self.mounts,
            workdir="/workspace",
            container_name=TEST_CONTAINER_NAME,
            launch_nonce=TEST_NONCE,
        )

    def validate_docker_argv(self, argv: tuple[str, ...] | list[str]) -> None:
        validate_gpu5_docker_argv(
            argv,
            source_snapshot=self.source_snapshot(),
            model_snapshot=self.model_snapshot(),
        )

    def scope(self, *, suffix: str = "0") -> ExecutionScope:
        digest = suffix * 64
        return ExecutionScope(
            source_commit=EXPECTED_COMMIT,
            source_tree_digest=digest,
            source_identity_digest=digest,
            source_file_count=12,
            source_root_device=1,
            source_root_inode=2,
            model_manifest_sha256=digest,
            model_tree_digest=digest,
            model_identity_digest=digest,
            model_file_count=6,
            model_root_device=3,
            model_root_inode=4,
            snapshot_path=str(self.snapshot_path),
            snapshot_nonce=TEST_NONCE,
            snapshot_mode=0o555,
            working_tree_digest="3" * 64,
            working_identity_digest="4" * 64,
        )

    def initialize_real_git_checkout(self) -> tuple[Path, str]:
        executable = shutil.which("git")
        if executable is None:
            self.skipTest("Git is required for source-snapshot contract tests")
        git = Path(executable).resolve(strict=True)

        def run(*arguments: str) -> str:
            completed = subprocess.run(
                [str(git), "-C", str(self.repo_root), *arguments],
                capture_output=True,
                text=True,
                check=True,
            )
            return completed.stdout.strip()

        run("init", "--quiet")
        (self.repo_root / ".gitignore").write_text("*.pyc\n", encoding="utf-8")
        scripts = self.repo_root / "scripts"
        scripts.mkdir()
        tracked = scripts / "tool.py"
        tracked.write_text("VALUE = 1\n", encoding="utf-8")
        tracked.chmod(0o644)
        (scripts / "ignored.pyc").write_bytes(b"ignored-bytecode")
        run("add", ".gitignore", "scripts/tool.py")
        run(
            "-c",
            "user.name=Cogni Test",
            "-c",
            "user.email=cogni@example.invalid",
            "commit",
            "--quiet",
            "-m",
            "snapshot fixture",
        )
        return git, run("rev-parse", "HEAD")

    def _write_strict_manifest(self, text: str) -> Path:
        manifest = self.repo_root / "model.manifest.toml"
        manifest.write_text(text, encoding="utf-8", newline="\n")
        return manifest

    @staticmethod
    def _manifest_entry(name: str, payload: bytes) -> str:
        digest = hashlib.sha256(payload).hexdigest()
        return f'"{name}" = "{digest}"'

    def test_stdlib_manifest_verifier_hashes_and_closes_the_exact_layout(self) -> None:
        config_payload = b'{"model_type":"gemma4"}\n'
        weights_payload = b"bounded-test-weights"
        config = self.model_root / "config.json"
        weights_directory = self.model_root / "weights"
        weights_directory.mkdir()
        weights = weights_directory / "model.safetensors"
        config.write_bytes(config_payload)
        weights.write_bytes(weights_payload)
        manifest = self._write_strict_manifest(
            "\n".join(
                (
                    "[model]",
                    'family = "gemma4"',
                    'variant = "E4B"',
                    'role = "instruction_tuned"',
                    'source = "google/gemma-4-E4B-it"',
                    'revision = "pinned-test-revision"',
                    "",
                    "[files]",
                    self._manifest_entry("config.json", config_payload),
                    self._manifest_entry("weights/model.safetensors", weights_payload),
                    "",
                )
            )
        )

        verified = guard.verify_artifact_manifest(self.model_root, manifest)
        self.assertEqual(
            verified.identity,
            guard.ArtifactIdentity(
                family="gemma4",
                variant="E4B",
                role="instruction_tuned",
                source="google/gemma-4-E4B-it",
                revision="pinned-test-revision",
            ),
        )
        self.assertEqual(
            verified.digests,
            (
                ("config.json", hashlib.sha256(config_payload).hexdigest()),
                (
                    "weights/model.safetensors",
                    hashlib.sha256(weights_payload).hexdigest(),
                ),
            ),
        )
        self.assertIs(guard.verify_closed_world_artifact_layout(verified), verified)

        manifest = self._write_strict_manifest(
            "\n".join(
                (
                    "[files]",
                    self._manifest_entry("config.json", config_payload),
                    self._manifest_entry("weights/model.safetensors", weights_payload),
                    "",
                    "[model]",
                    'family = "gemma4"',
                    'variant = "E4B"',
                    'role = "instruction_tuned"',
                    'source = "google/gemma-4-E4B-it"',
                    'revision = "pinned-test-revision"',
                    "",
                )
            )
        )
        files_first_verified = guard.verify_artifact_manifest(self.model_root, manifest)
        self.assertEqual(files_first_verified.identity, verified.identity)
        self.assertEqual(files_first_verified.digests, verified.digests)

        rogue = self.model_root / "unmanifested.bin"
        rogue.write_bytes(b"forbidden")
        with self.assertRaisesRegex(
            guard.ArtifactVerificationError, "closed-world file is forbidden"
        ):
            guard.verify_closed_world_artifact_layout(verified)
        rogue.unlink()
        config.write_bytes(b"changed")
        with self.assertRaisesRegex(guard.ArtifactVerificationError, "digest mismatch"):
            guard.verify_artifact_manifest(self.model_root, manifest)

    def test_stdlib_manifest_parser_rejects_extra_duplicate_malformed_and_unsafe(
        self,
    ) -> None:
        payload = b"artifact"
        (self.model_root / "config.json").write_bytes(payload)
        digest = hashlib.sha256(payload).hexdigest()
        valid_entry = f'"config.json" = "{digest}"'
        malformed_manifests = {
            "extra root key": f'owner = "operator"\n[files]\n{valid_entry}\n',
            "extra table": f"[files]\n{valid_entry}\n[model]\n",
            "duplicate": f"[files]\n{valid_entry}\n{valid_entry}\n",
            "unquoted key": f'[files]\nconfig.json = "{digest}"\n',
            "inline comment": f"[files]\n{valid_entry} # forbidden\n",
            "invalid digest": '[files]\n"config.json" = "abcd"\n',
            "uppercase digest": f'[files]\n"config.json" = "{digest.upper()}"\n',
            "unsafe parent": f'[files]\n"../config.json" = "{digest}"\n',
            "unsafe absolute": f'[files]\n"/config.json" = "{digest}"\n',
            "unsafe empty segment": (f'[files]\n"weights//config.json" = "{digest}"\n'),
            "unsafe dot segment": (f'[files]\n"weights/./config.json" = "{digest}"\n'),
            "control character": f"[files]\n{valid_entry}\x00\n",
            "duplicate identity": (
                "[model]\n"
                'family = "gemma4"\n'
                'family = "gemma4"\n'
                'variant = "E4B"\n'
                'role = "base"\n'
                'source = "google/gemma-4-E4B"\n'
                'revision = "pinned"\n'
                f"[files]\n{valid_entry}\n"
            ),
            "extra identity key": (
                "[model]\n"
                'family = "gemma4"\n'
                'variant = "E4B"\n'
                'role = "base"\n'
                'source = "google/gemma-4-E4B"\n'
                'revision = "pinned"\n'
                'owner = "operator"\n'
                f"[files]\n{valid_entry}\n"
            ),
            "missing identity key": (
                "[model]\n"
                'family = "gemma4"\n'
                'variant = "E4B"\n'
                'role = "base"\n'
                'source = "google/gemma-4-E4B"\n'
                f"[files]\n{valid_entry}\n"
            ),
            "unsupported identity role": (
                "[model]\n"
                'family = "gemma4"\n'
                'variant = "E4B"\n'
                'role = "chat-ish"\n'
                'source = "google/gemma-4-E4B"\n'
                'revision = "pinned"\n'
                f"[files]\n{valid_entry}\n"
            ),
        }
        for label, text in malformed_manifests.items():
            with self.subTest(label=label):
                manifest = self._write_strict_manifest(text)
                with self.assertRaises(guard.ArtifactVerificationError):
                    guard.verify_artifact_manifest(self.model_root, manifest)

    @unittest.skipUnless(os.name == "posix", "Linux ownership and mode contract")
    def test_model_trust_rejects_writable_symlinked_and_hardlinked_inputs(
        self,
    ) -> None:
        payload = b"trusted-model-artifact"
        artifact_parent = self.model_root / "weights"
        artifact_parent.mkdir(mode=0o755)
        artifact = artifact_parent / "model.safetensors"
        artifact.write_bytes(payload)
        artifact.chmod(0o644)
        manifest = self._write_strict_manifest(
            f"[files]\n{self._manifest_entry('weights/model.safetensors', payload)}\n"
        )
        manifest.chmod(0o644)

        for target in (self.model_root, artifact_parent, manifest, artifact):
            original_mode = stat.S_IMODE(target.stat().st_mode)
            target.chmod(original_mode | 0o020)
            try:
                with (
                    self.subTest(target=target),
                    self.assertRaises(guard.ArtifactVerificationError),
                ):
                    guard.verify_artifact_manifest(self.model_root, manifest)
            finally:
                target.chmod(original_mode)

        manifest_link = manifest.with_name("model.manifest.hardlink.toml")
        os.link(manifest, manifest_link)
        try:
            with self.assertRaises(guard.ArtifactVerificationError):
                guard.verify_artifact_manifest(self.model_root, manifest)
        finally:
            manifest_link.unlink()

        model_link = self.model_root.with_name("gemma4-e4b-link")
        model_link.symlink_to(self.model_root, target_is_directory=True)
        try:
            with self.assertRaises(guard.ArtifactVerificationError):
                guard.verify_artifact_manifest(model_link, manifest)
        finally:
            model_link.unlink()

    @unittest.skipUnless(os.name == "posix", "Linux ownership and mode contract")
    def test_source_trust_rejects_writable_parent_root_directory_and_file(
        self,
    ) -> None:
        git, commit = self.initialize_real_git_checkout()
        scripts = self.repo_root / "scripts"
        tracked = scripts / "tool.py"
        with patch.object(guard, "GIT_EXECUTABLE", git):
            for target in (self.repo_root.parent, self.repo_root, scripts, tracked):
                original_mode = stat.S_IMODE(target.stat().st_mode)
                target.chmod(original_mode | 0o020)
                try:
                    with (
                        self.subTest(target=target),
                        self.assertRaisesRegex(
                            GPU5BoundaryError, "group/world writable|metadata is unsafe"
                        ),
                    ):
                        guard._verify_working_checkout(
                            self.repo_root,
                            commit,
                            git_runner=subprocess.run,
                        )
                finally:
                    target.chmod(original_mode)

            hardlink = scripts / "tool-hardlink.py"
            os.link(tracked, hardlink)
            try:
                with self.assertRaisesRegex(GPU5BoundaryError, "metadata is unsafe"):
                    guard._hash_git_worktree_file(
                        tracked,
                        expected_root=self.repo_root,
                        object_format="sha1",
                    )
            finally:
                hardlink.unlink()

    def test_pinned_repository_manifest_bytes_match_the_strict_host_grammar(
        self,
    ) -> None:
        pinned = (
            Path(guard.__file__).resolve().parents[1]
            / "config"
            / "gemma4-e4b.manifest.toml"
        )
        manifest_copy = self.repo_root / "pinned.manifest.toml"
        manifest_copy.write_bytes(pinned.read_bytes())
        entries, identity = guard._strict_model_manifest_entries(manifest_copy)
        self.assertGreaterEqual(len(entries), 1)
        self.assertIn("config.json", dict(entries))
        if identity is not None:
            self.assertIn(identity.role, {"base", "instruction_tuned"})

    def test_project_accepts_only_physical_gpu5(self) -> None:
        self.assertEqual(require_project_gpu_index(5), 5)
        for rejected in (None, True, False, 0, 1, 2, 3, 4, 6, 7, -1, "5"):
            with self.subTest(rejected=rejected), self.assertRaises(GPU5BoundaryError):
                require_project_gpu_index(rejected)

    def test_native_environment_exposes_only_physical_gpu5(self) -> None:
        environment = native_gpu5_environment({"PATH": "/bin"})
        self.assertEqual(environment["CUDA_VISIBLE_DEVICES"], PROJECT_GPU_UUID)
        self.assertEqual(environment["CUDA_DEVICE_ORDER"], "PCI_BUS_ID")
        self.assertEqual(environment["NVIDIA_VISIBLE_DEVICES"], PROJECT_GPU_UUID)
        inherited = native_gpu5_environment(
            {
                "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
                "CUDA_VISIBLE_DEVICES": PROJECT_GPU_UUID,
                "NVIDIA_VISIBLE_DEVICES": PROJECT_GPU_UUID,
            }
        )
        self.assertEqual(inherited["NVIDIA_VISIBLE_DEVICES"], PROJECT_GPU_UUID)
        for value in ("0", "0,5", "4", "5", "6", "7", "all"):
            with self.subTest(value=value), self.assertRaises(GPU5BoundaryError):
                native_gpu5_environment({"CUDA_VISIBLE_DEVICES": value})
            with self.subTest(nvidia_value=value), self.assertRaises(GPU5BoundaryError):
                native_gpu5_environment({"NVIDIA_VISIBLE_DEVICES": value})
        with self.assertRaisesRegex(GPU5BoundaryError, "PCI_BUS_ID"):
            native_gpu5_environment({"CUDA_DEVICE_ORDER": "FASTEST_FIRST"})

    def test_guarded_identity_requires_exact_visibility_uuid_and_logical_zero(
        self,
    ) -> None:
        cuda = SimpleNamespace(
            is_available=lambda: True,
            device_count=lambda: 1,
            current_device=lambda: 0,
        )
        torch_module = SimpleNamespace(cuda=cuda)
        native_runner = _RecordedRunner([_completed(stdout=f"5, {PROJECT_GPU_UUID}\n")])
        with patch.dict(
            os.environ,
            {
                "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
                "CUDA_VISIBLE_DEVICES": PROJECT_GPU_UUID,
                "NVIDIA_VISIBLE_DEVICES": PROJECT_GPU_UUID,
            },
            clear=True,
        ):
            identity = validate_guarded_gpu5_identity(
                physical_gpu_index=5,
                gpu_query_context="native-host",
                torch_module=torch_module,
                runner=native_runner,
            )
        self.assertEqual(identity.uuid, PROJECT_GPU_UUID)
        self.assertEqual(identity.logical_device_index, 0)
        self.assertEqual(native_runner.calls[0][0][1:3], ("-i", "5"))
        self.assertEqual(
            native_runner.calls[0][1]["env"], guard._MINIMAL_HOST_ENVIRONMENT
        )

        for wrong_order in (None, "", "FASTEST_FIRST"):
            native_environment = {
                "CUDA_VISIBLE_DEVICES": PROJECT_GPU_UUID,
                "NVIDIA_VISIBLE_DEVICES": PROJECT_GPU_UUID,
            }
            if wrong_order is not None:
                native_environment["CUDA_DEVICE_ORDER"] = wrong_order
            rejected_order = _RecordedRunner([])
            with (
                self.subTest(wrong_order=wrong_order),
                patch.dict(os.environ, native_environment, clear=True),
                self.assertRaisesRegex(GPU5BoundaryError, "CUDA_DEVICE_ORDER"),
            ):
                validate_guarded_gpu5_identity(
                    physical_gpu_index=5,
                    gpu_query_context="native-host",
                    torch_module=torch_module,
                    runner=rejected_order,
                )
            self.assertEqual(rejected_order.calls, [])

        container_runner = _RecordedRunner(
            [_completed(stdout=f"0, {PROJECT_GPU_UUID}\n")]
        )
        with patch.dict(os.environ, {}, clear=True):
            container_identity = validate_guarded_gpu5_identity(
                physical_gpu_index=5,
                gpu_query_context="gpu5-container",
                torch_module=torch_module,
                runner=container_runner,
            )
        self.assertEqual(container_identity.logical_device_count, 1)
        self.assertEqual(container_runner.calls[0][0][1:3], ("-i", PROJECT_GPU_UUID))

        rejected = _RecordedRunner([])
        with (
            patch.dict(
                os.environ,
                {
                    "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
                    "CUDA_VISIBLE_DEVICES": "0",
                    "NVIDIA_VISIBLE_DEVICES": PROJECT_GPU_UUID,
                },
                clear=True,
            ),
            self.assertRaises(GPU5BoundaryError),
        ):
            validate_guarded_gpu5_identity(
                physical_gpu_index=5,
                gpu_query_context="native-host",
                torch_module=torch_module,
                runner=rejected,
            )
        self.assertEqual(rejected.calls, [])

        wrong_uuid = _RecordedRunner([_completed(stdout="0, GPU-wrong\n")])
        with (
            patch.dict(os.environ, {}, clear=True),
            self.assertRaisesRegex(GPU5BoundaryError, "UUID mismatch"),
        ):
            validate_guarded_gpu5_identity(
                physical_gpu_index=5,
                gpu_query_context="gpu5-container",
                torch_module=torch_module,
                runner=wrong_uuid,
            )
        bad_logical = SimpleNamespace(
            cuda=SimpleNamespace(
                is_available=lambda: True,
                device_count=lambda: 2,
                current_device=lambda: 0,
            )
        )
        with (
            patch.dict(os.environ, {}, clear=True),
            self.assertRaisesRegex(GPU5BoundaryError, "exactly one logical"),
        ):
            validate_guarded_gpu5_identity(
                physical_gpu_index=5,
                gpu_query_context="gpu5-container",
                torch_module=bad_logical,
                runner=_RecordedRunner([_completed(stdout=f"0, {PROJECT_GPU_UUID}\n")]),
            )

    def test_host_control_environment_never_inherits_docker_or_loader_state(
        self,
    ) -> None:
        hostile = {
            "PATH": "/tmp/shadow-bin",
            "DOCKER_HOST": "tcp://attacker:2375",
            "DOCKER_CONTEXT": "remote",
            "DOCKER_CONFIG": "/tmp/hostile-config",
            "DOCKER_TLS_VERIFY": "1",
            "LD_PRELOAD": "/tmp/inject.so",
        }
        with patch.dict(os.environ, hostile, clear=True):
            argv = self.docker_argv()
            environment = guard._minimal_host_environment()
        self.assertEqual(argv[0], str(self.docker))
        self.assertEqual(argv[1:3], ("--host", guard.DOCKER_HOST_URI))
        self.assertEqual(environment, guard._MINIMAL_HOST_ENVIRONMENT)
        for forbidden in hostile:
            if forbidden != "PATH":
                self.assertNotIn(forbidden, environment)
        self.assertEqual(environment["PATH"], "/usr/bin:/bin")

    def test_host_snapshot_never_uses_selectorless_or_other_gpu_queries(self) -> None:
        runner = _RecordedRunner(_idle_smi_responses())
        snapshot = query_gpu5_snapshot(runner=runner)
        self.assertEqual(snapshot.physical_index, 5)
        self.assertEqual(snapshot.uuid, PROJECT_GPU_UUID)
        self.assertEqual(len(runner.calls), 2)
        for argv, kwargs in runner.calls:
            self.assertEqual(argv[:3], (str(self.nvidia_smi), "-i", "5"))
            self.assertEqual(kwargs["timeout"], 5.0)
            self.assertEqual(kwargs["env"], guard._MINIMAL_HOST_ENVIRONMENT)
        forbidden = _RecordedRunner([])
        with self.assertRaises(GPU5BoundaryError):
            _run_nvidia_smi(
                (
                    str(self.nvidia_smi),
                    "--query-gpu=index",
                    "--format=csv,noheader",
                ),
                runner=forbidden,
            )
        self.assertEqual(forbidden.calls, [])

    def test_snapshot_and_idle_checks_fail_closed(self) -> None:
        cases = (
            ([_completed(stdout="5, GPU-wrong, 0, 19, 49140\n")], "UUID"),
            (
                [
                    _completed(stdout=f"5, {PROJECT_GPU_UUID}, 1, 19, 49140\n"),
                    _completed(),
                ],
                "utilization",
            ),
            (
                [
                    _completed(stdout=f"5, {PROJECT_GPU_UUID}, 0, 65, 49140\n"),
                    _completed(),
                ],
                "memory",
            ),
            (
                [
                    _completed(stdout=f"5, {PROJECT_GPU_UUID}, 0, 19, 49140\n"),
                    _completed(stdout=f"{PROJECT_GPU_UUID}, 4242, 1024\n"),
                ],
                "foreign compute PID",
            ),
        )
        for responses, message in cases:
            with (
                self.subTest(message=message),
                self.assertRaisesRegex(GPU5BoundaryError, message),
            ):
                preflight_gpu5(runner=_RecordedRunner(responses))
        with self.assertRaisesRegex(GPU5BoundaryError, "TimeoutExpired"):
            preflight_gpu5(
                runner=_RecordedRunner(
                    [subprocess.TimeoutExpired(cmd="nvidia-smi", timeout=5.0)]
                )
            )

    def test_execution_scope_rejects_dirty_source_before_model_access(self) -> None:
        runner = _RecordedRunner(
            [
                _completed(stdout=f"{self.repo_root}\n"),
                _completed(stdout=f"{EXPECTED_COMMIT}\n"),
                _completed(stdout=b"?? rogue.py\0", stderr=b""),
            ]
        )
        with self.assertRaisesRegex(GPU5BoundaryError, "clean"):
            capture_execution_scope(
                EXPECTED_COMMIT,
                source_snapshot=self.source_snapshot(),
                model_snapshot=self.model_snapshot(),
                git_runner=runner,
            )
        self.assertEqual(len(runner.calls), 3)
        for argv, kwargs in runner.calls:
            self.assertEqual(argv[0], str(self.git))
            self.assertEqual(kwargs["env"], guard._MINIMAL_HOST_ENVIRONMENT)

    def test_exact_commit_snapshot_excludes_git_ignored_and_bytecode_state(
        self,
    ) -> None:
        if os.name != "posix":
            self.skipTest("the production snapshot gate is Linux-only")
        git, commit = self.initialize_real_git_checkout()
        nonce = "1" * 32
        with patch.object(guard, "GIT_EXECUTABLE", git):
            snapshot = guard.prepare_source_snapshot(commit, nonce)
            self.assertEqual(snapshot.source_commit, commit)
            self.assertEqual(snapshot.launch_nonce, nonce)
            self.assertEqual(snapshot.root_mode, 0o555)
            root = Path(snapshot.root_path)
            self.assertEqual(
                sorted(
                    path.relative_to(root).as_posix()
                    for path in root.rglob("*")
                    if path.is_file()
                ),
                [".gitignore", "scripts/tool.py"],
            )
            self.assertFalse((root / ".git").exists())
            self.assertFalse((root / "scripts" / "ignored.pyc").exists())
            self.assertEqual(stat.S_IMODE((root / "scripts").stat().st_mode), 0o555)
            self.assertEqual(
                stat.S_IMODE((root / "scripts" / "tool.py").stat().st_mode),
                0o444,
            )
            with self.assertRaisesRegex(GPU5BoundaryError, "already exists"):
                guard.prepare_source_snapshot(commit, nonce)

    def test_snapshot_rejects_index_hiding_and_changed_snapshot_blob(self) -> None:
        if os.name != "posix":
            self.skipTest("the production snapshot gate is Linux-only")
        git, commit = self.initialize_real_git_checkout()
        nonce = "2" * 32
        with patch.object(guard, "GIT_EXECUTABLE", git):
            snapshot = guard.prepare_source_snapshot(commit, nonce)
            root = Path(snapshot.root_path)
            tracked_snapshot = root / "scripts" / "tool.py"
            tracked_snapshot.chmod(0o644)
            tracked_snapshot.write_text("VALUE = 2\n", encoding="utf-8")
            object_format, entries = guard._commit_tree(
                self.repo_root,
                commit,
                git_runner=subprocess.run,
            )
            with self.assertRaisesRegex(GPU5BoundaryError, "blob or mode"):
                guard._snapshot_inventory(
                    root,
                    source_commit=commit,
                    launch_nonce=nonce,
                    object_format=object_format,
                    expected_entries=entries,
                )
            subprocess.run(
                [
                    str(git),
                    "-C",
                    str(self.repo_root),
                    "update-index",
                    "--assume-unchanged",
                    "scripts/tool.py",
                ],
                check=True,
            )
            with self.assertRaisesRegex(GPU5BoundaryError, "assume-unchanged"):
                guard.prepare_source_snapshot(commit, "3" * 32)

    def test_snapshot_path_policy_rejects_traversal_metadata_and_bytecode(self) -> None:
        for encoded in (
            b"../escape.py",
            b".git/config",
            b"pkg/__pycache__/module.py",
            b"pkg/module.pyc",
            b"pkg\\module.py",
        ):
            with self.subTest(encoded=encoded), self.assertRaises(GPU5BoundaryError):
                guard._safe_source_name(encoded)

    @unittest.skipUnless(os.name == "posix", "Linux native snapshot contract")
    def test_native_snapshot_copies_model_to_separate_sealed_inodes(self) -> None:
        git, _first_commit = self.initialize_real_git_checkout()
        payload = b"small-model-snapshot-fixture"
        weights = self.model_root / "weights"
        weights.mkdir(mode=0o755)
        source_artifact = weights / "model.safetensors"
        source_artifact.write_bytes(payload)
        source_artifact.chmod(0o644)
        config = self.repo_root / "config"
        config.mkdir(mode=0o755)
        manifest = config / "gemma4-e4b-it.manifest.toml"
        manifest.write_text(
            f"[files]\n{self._manifest_entry('weights/model.safetensors', payload)}\n",
            encoding="utf-8",
            newline="\n",
        )
        subprocess.run(
            [str(git), "-C", str(self.repo_root), "add", str(manifest)],
            check=True,
        )
        subprocess.run(
            [
                str(git),
                "-C",
                str(self.repo_root),
                "-c",
                "user.name=Cogni Test",
                "-c",
                "user.email=cogni@example.invalid",
                "commit",
                "--quiet",
                "-m",
                "native snapshot fixture",
            ],
            check=True,
        )
        commit = subprocess.run(
            [str(git), "-C", str(self.repo_root), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()

        with patch.object(guard, "GIT_EXECUTABLE", git):
            snapshot = guard.prepare_native_execution_snapshot(
                commit,
                self.model_root,
                "config/gemma4-e4b-it.manifest.toml",
                git_runner=subprocess.run,
            )
        self.assertEqual(guard.verify_native_execution_snapshot(snapshot), snapshot)
        self.assertEqual(snapshot.source.root_mode, 0o555)
        self.assertEqual(snapshot.model.root_mode, 0o555)
        self.assertEqual(snapshot.model.file_count, 1)
        self.assertEqual(snapshot.model.total_bytes, len(payload))
        self.assertEqual(Path(snapshot.workspace_root), self.repo_root)
        copied = Path(snapshot.model.root_path) / "weights" / "model.safetensors"
        self.assertEqual(stat.S_IMODE(copied.stat().st_mode), 0o444)
        self.assertEqual(copied.read_bytes(), payload)
        self.assertNotEqual(
            (copied.stat().st_dev, copied.stat().st_ino),
            (source_artifact.stat().st_dev, source_artifact.stat().st_ino),
        )
        self.assertEqual(copied.stat().st_nlink, 1)

        copied.chmod(0o644)
        with self.assertRaises(GPU5BoundaryError) as caught:
            guard.verify_native_execution_snapshot(snapshot)
        self.assertEqual(
            str(caught.exception),
            "model snapshot contains an unsafe file",
        )

    @unittest.skipUnless(os.name == "posix", "Linux Docker snapshot contract")
    def test_docker_snapshot_mounts_only_separate_sealed_model_copy(self) -> None:
        git, _first_commit = self.initialize_real_git_checkout()
        payload = b"docker-model-snapshot-fixture"
        source_artifact = self.model_root / "model.safetensors"
        source_artifact.write_bytes(payload)
        source_artifact.chmod(0o644)
        config = self.repo_root / "config"
        config.mkdir(mode=0o755)
        manifest = config / "gemma4-e4b.manifest.toml"
        manifest.write_text(
            f"[files]\n{self._manifest_entry('model.safetensors', payload)}\n",
            encoding="utf-8",
            newline="\n",
        )
        subprocess.run(
            [str(git), "-C", str(self.repo_root), "add", str(manifest)],
            check=True,
        )
        subprocess.run(
            [
                str(git),
                "-C",
                str(self.repo_root),
                "-c",
                "user.name=Cogni Test",
                "-c",
                "user.email=cogni@example.invalid",
                "commit",
                "--quiet",
                "-m",
                "Docker snapshot fixture",
            ],
            check=True,
        )
        commit = subprocess.run(
            [str(git), "-C", str(self.repo_root), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        nonce = "8" * 32

        # This unit test must exercise only its bounded temporary fixture.  The
        # production profile intentionally points at the real sealed model
        # directory, which would make a Linux CPU test copy the host model and
        # turn a unit test into an unbounded external dependency.
        with (
            patch.object(guard, "GIT_EXECUTABLE", git),
            patch.object(
                guard,
                "_raw_model_source_root",
                return_value=self.model_root,
            ) as raw_model_root,
        ):
            source_snapshot = guard.prepare_source_snapshot(
                commit,
                nonce,
                git_runner=subprocess.run,
            )
            model_snapshot = guard._prepare_docker_model_snapshot(
                source_snapshot,
                launch_nonce=nonce,
            )
            scope = guard.capture_execution_scope(
                commit,
                source_snapshot=source_snapshot,
                model_snapshot=model_snapshot,
                git_runner=subprocess.run,
            )
        raw_model_root.assert_called_once_with(guard.BASE_CANARY_ARTIFACT_PROFILE)
        copied = Path(model_snapshot.root_path) / "model.safetensors"
        self.assertEqual(copied.read_bytes(), payload)
        self.assertEqual(stat.S_IMODE(copied.stat().st_mode), 0o444)
        self.assertNotEqual(
            (copied.stat().st_dev, copied.stat().st_ino),
            (source_artifact.stat().st_dev, source_artifact.stat().st_ino),
        )
        mounts = guard._runtime_mount_map(
            expected_source_commit=commit,
            launch_nonce=nonce,
            source_snapshot=source_snapshot,
            model_snapshot=model_snapshot,
        )
        self.assertEqual(
            mounts[Path(model_snapshot.root_path)], Path("/models/gemma4-e4b")
        )
        self.assertNotIn(self.model_root.resolve(), mounts)
        self.assertEqual(scope.model_tree_digest, model_snapshot.content_digest)

    @unittest.skipUnless(os.name == "posix", "Linux snapshot quota contract")
    def test_snapshot_quota_counts_partial_directories_and_bytes(self) -> None:
        partial = self.model_snapshot_root / ".model-leftover.partial"
        partial.mkdir(mode=0o700)
        with self.assertRaisesRegex(GPU5BoundaryError, "count quota"):
            guard._enforce_snapshot_store_quota(
                self.model_snapshot_root,
                max_snapshots=1,
                max_bytes=1024,
                reserve_snapshots=1,
                reserve_bytes=0,
            )
        retained = partial / "retained.bin"
        retained.write_bytes(b"retained evidence")
        retained.chmod(0o600)
        with self.assertRaisesRegex(GPU5BoundaryError, "byte quota"):
            guard._enforce_snapshot_store_quota(
                self.model_snapshot_root,
                max_snapshots=2,
                max_bytes=1,
                reserve_snapshots=0,
                reserve_bytes=0,
            )
        self.assertTrue(partial.is_dir())
        self.assertTrue(retained.is_file())

    def test_docker_contract_is_exact_digest_offline_gpu5_and_read_only(self) -> None:
        argv = self.docker_argv()
        self.validate_docker_argv(argv)
        pairs = tuple(zip(argv, argv[1:]))
        self.assertIn(("--gpus", f"device={PROJECT_GPU_UUID}"), pairs)
        self.assertIn(("--network", "none"), pairs)
        self.assertIn(("--workdir", "/workspace"), pairs)
        self.assertIn(("--entrypoint", "/usr/local/bin/python"), pairs)
        self.assertIn("--pull=never", argv)
        hardening = (
            "--user",
            "8001:8001",
            "--cap-drop",
            "ALL",
            "--security-opt",
            "no-new-privileges:true",
            "--read-only",
            "--tmpfs",
            "/tmp:rw,nosuid,nodev,noexec,size=268435456,mode=1777",
            "--pids-limit",
            "512",
        )
        pull_position = argv.index("--pull=never")
        entrypoint_position = argv.index("--entrypoint")
        self.assertEqual(argv[pull_position + 1 : entrypoint_position], hardening)
        for option in (
            "--user",
            "--cap-drop",
            "--security-opt",
            "--read-only",
            "--tmpfs",
            "--pids-limit",
        ):
            with self.subTest(option=option):
                self.assertEqual(argv.count(option), 1)
        self.assertEqual(argv[0], str(self.docker))
        self.assertEqual(
            argv[1:5],
            ("--host", guard.DOCKER_HOST_URI, "--config", str(self.docker_config)),
        )
        self.assertIn("io.cognios.guard=gpu5", argv)
        self.assertIn(f"io.cognios.source-commit={EXPECTED_COMMIT}", argv)
        self.assertIn(f"io.cognios.launch-nonce={TEST_NONCE}", argv)
        self.assertIn("io.cognios.execution-profile=inspection", argv)
        self.assertIn(PINNED_DOCKER_IMAGE, argv)
        volumes = [
            argv[index + 1] for index, token in enumerate(argv) if token == "--volume"
        ]
        self.assertEqual(len(volumes), 2)
        self.assertTrue(all(volume.endswith(":ro") for volume in volumes))
        self.assertTrue(all("/evidence" not in volume for volume in volumes))
        self.assertIn(
            f"{self.snapshot_path.resolve()}:/workspace:ro",
            volumes,
        )
        self.assertIn(
            f"{self.model_snapshot_path.resolve()}:/models/gemma4-e4b:ro",
            volumes,
        )
        self.assertNotIn(str(self.model_root.resolve()), "\n".join(argv))
        for name, value in guard._REQUIRED_CONTAINER_ENVIRONMENT.items():
            self.assertIn(f"{name}={value}", argv)

    def test_production_argv_rejects_every_non5_or_missing_gpu_selector(self) -> None:
        base = list(self.docker_argv())
        for selector in (
            "all",
            *(f"device={index}" for index in range(8)),
            "device=GPU-wrong",
        ):
            changed = list(base)
            changed[changed.index(f"device={PROJECT_GPU_UUID}")] = selector
            with self.subTest(selector=selector), self.assertRaises(GPU5BoundaryError):
                self.validate_docker_argv(changed)
        missing = list(base)
        position = missing.index("--gpus")
        del missing[position : position + 2]
        with self.assertRaises(GPU5BoundaryError):
            self.validate_docker_argv(missing)

    def test_container_hardening_rejects_bounded_contract_mutations(self) -> None:
        base = list(self.docker_argv())
        option_widths = (
            ("--user", 2),
            ("--cap-drop", 2),
            ("--security-opt", 2),
            ("--read-only", 1),
            ("--tmpfs", 2),
            ("--pids-limit", 2),
        )
        for option, width in option_widths:
            position = base.index(option)
            missing = list(base)
            del missing[position : position + width]
            with (
                self.subTest(kind="missing", option=option),
                self.assertRaises(GPU5BoundaryError),
            ):
                self.validate_docker_argv(missing)

            duplicated = list(base)
            duplicated[position:position] = base[position : position + width]
            with (
                self.subTest(kind="duplicated", option=option),
                self.assertRaises(GPU5BoundaryError),
            ):
                self.validate_docker_argv(duplicated)

        reordered = list(base)
        user_position = reordered.index("--user")
        reordered[user_position : user_position + 4] = (
            "--cap-drop",
            "ALL",
            "--user",
            "8001:8001",
        )
        with (
            self.subTest(kind="reordered"),
            self.assertRaises(GPU5BoundaryError),
        ):
            self.validate_docker_argv(reordered)

        wrong_values = (
            ("--user", "0:0"),
            ("--user", "8001:0"),
            ("--cap-drop", "NET_RAW"),
            ("--security-opt", "no-new-privileges:false"),
            ("--security-opt", "seccomp=unconfined"),
            (
                "--tmpfs",
                "/tmp:rw,nosuid,nodev,size=268435456,mode=1777",
            ),
            (
                "--tmpfs",
                "/tmp:rw,nosuid,nodev,exec,size=268435456,mode=1777",
            ),
            (
                "--tmpfs",
                "/workspace:rw,nosuid,nodev,noexec,size=268435456,mode=1777",
            ),
            ("--pids-limit", "0"),
            ("--pids-limit", "-1"),
            ("--pids-limit", "1024"),
        )
        for option, wrong_value in wrong_values:
            changed = list(base)
            changed[changed.index(option) + 1] = wrong_value
            with (
                self.subTest(kind="value", option=option, wrong_value=wrong_value),
                self.assertRaises(GPU5BoundaryError),
            ):
                self.validate_docker_argv(changed)

        writable_root = list(base)
        writable_root[writable_root.index("--read-only")] = "--read-only=false"
        with (
            self.subTest(kind="value", option="--read-only"),
            self.assertRaises(GPU5BoundaryError),
        ):
            self.validate_docker_argv(writable_root)

    def test_build_rejects_image_env_mount_workdir_and_privilege_mutations(
        self,
    ) -> None:
        mutations = (
            {"image": "cogni-os-dev:v0.4.1-cu128"},
            {"environment": {}},
            {"environment": {**self.environment, "PYTHONPATH": "/tmp"}},
            {"mounts": self.mounts[:1]},
            {"mounts": (*self.mounts, f"{self.evidence_root}:/evidence:rw")},
            {"workdir": "/tmp"},
        )
        defaults = {
            "image": PINNED_DOCKER_IMAGE,
            "environment": self.environment,
            "mounts": self.mounts,
            "workdir": "/workspace",
        }
        for mutation in mutations:
            arguments = {**defaults, **mutation}
            with self.subTest(mutation=mutation), self.assertRaises(GPU5BoundaryError):
                build_gpu5_docker_argv(
                    arguments["image"],
                    self.runtime_command(),
                    expected_source_commit=EXPECTED_COMMIT,
                    source_snapshot=self.source_snapshot(),
                    model_snapshot=self.model_snapshot(),
                    environment=arguments["environment"],
                    mounts=arguments["mounts"],
                    workdir=arguments["workdir"],
                    launch_nonce=TEST_NONCE,
                )
        for extra in (
            "--privileged",
            "--device=/dev/nvidia6",
            "--network=host",
            "--user=0:0",
            "--cap-add=ALL",
            "--security-opt=seccomp=unconfined",
            "--read-only=false",
            "--tmpfs=/workspace:rw",
            "--pids-limit=-1",
        ):
            changed = list(self.docker_argv())
            changed.insert(changed.index("--"), extra)
            with self.subTest(extra=extra), self.assertRaises(GPU5BoundaryError):
                self.validate_docker_argv(changed)

    def test_mounts_require_exact_roots_and_destinations(self) -> None:
        child = self.repo_root / "child"
        sibling = self.repo_root.parent / "other"
        child.mkdir()
        sibling.mkdir()
        self.assertTrue(
            _source_within_allowed_roots(
                self.repo_root, (self.repo_root, self.model_root)
            )
        )
        self.assertTrue(
            _source_within_allowed_roots(child, (self.repo_root, self.model_root))
        )
        self.assertFalse(
            _source_within_allowed_roots(sibling, (self.repo_root, self.model_root))
        )
        for bad_mounts in (
            (f"{child}:/workspace:ro", self.mounts[1]),
            (f"{sibling}:/workspace:ro", self.mounts[1]),
            (f"{self.repo_root}:/other:ro", self.mounts[1]),
            (f"{self.repo_root}:/workspace:rw", self.mounts[1]),
            (self.mounts[0], f"{self.model_root}:/models/gemma4-e4b:ro"),
        ):
            with self.subTest(mounts=bad_mounts), self.assertRaises(GPU5BoundaryError):
                build_gpu5_docker_argv(
                    PINNED_DOCKER_IMAGE,
                    self.runtime_command(),
                    expected_source_commit=EXPECTED_COMMIT,
                    source_snapshot=self.source_snapshot(),
                    model_snapshot=self.model_snapshot(),
                    environment=self.environment,
                    mounts=bad_mounts,
                    workdir="/workspace",
                    launch_nonce=TEST_NONCE,
                )

    def test_validator_allowlist_rejects_unknown_duplicate_oversized_and_shell_args(
        self,
    ) -> None:
        accepted = (
            self.runtime_command("--event-stream", "--workspace-mib", "512"),
            self.completion_command("--turns", "20", "--timeout", "120"),
            (
                "-I",
                "-B",
                "/workspace/scripts/validate_gemma4_deq.py",
                "--model",
                "/models/gemma4-e4b",
                "--manifest",
                "/workspace/config/gemma4-e4b.manifest.toml",
                "--physical-gpu-index",
                "5",
                "--gpu-query-context",
                "gpu5-container",
                "--allow-uncertified-experimental",
            ),
        )
        for command in accepted:
            self.assertEqual(_validated_validator_command(command), command)
        rejected = (
            self.runtime_command("--unknown", "1"),
            self.runtime_command("--model", "/models/gemma4-e4b"),
            self.runtime_command("--prompt", "x" * 513),
            self.runtime_command("--", "echo"),
            self.runtime_command(";", "echo"),
            self.completion_command("--turns", "101"),
            self.completion_command("--timeout", "121"),
            self.completion_command("--output", "/evidence/gpu5.json"),
            self.completion_command("--gpu-query-context", "native-host"),
        )
        for command in rejected:
            with self.subTest(command=command), self.assertRaises(GPU5BoundaryError):
                _validated_validator_command(command)
        experimental_deq = accepted[2]
        bounded_experimental_deq = (
            *experimental_deq[:-1],
            "--contractive-delta-scale",
            "0.001",
            "--certified-delta-lipschitz-bound",
            "900",
            experimental_deq[-1],
        )
        self.assertEqual(
            _validated_validator_command(bounded_experimental_deq),
            bounded_experimental_deq,
        )
        provenance_experimental_deq = (
            *experimental_deq,
            "--contractivity-provenance",
            "evidence/deq-contractivity.json",
        )
        self.assertEqual(
            _validated_validator_command(provenance_experimental_deq),
            provenance_experimental_deq,
        )
        for non_experimental_deq in (
            bounded_experimental_deq[:-1],
            (
                *experimental_deq[:-1],
                "--contractivity-provenance",
                "../escaped.json",
                experimental_deq[-1],
            ),
            (
                *experimental_deq[:-1],
                "--contractive-delta-scale",
                "0",
                experimental_deq[-1],
            ),
            (
                *experimental_deq[:-1],
                "--fallback-damping",
                "0",
                experimental_deq[-1],
            ),
            (
                *experimental_deq[:-1],
                "--contractive-delta-scale",
                "0.001",
                "--certified-delta-lipschitz-bound",
                "951",
                experimental_deq[-1],
            ),
        ):
            with self.assertRaises(GPU5BoundaryError):
                _validated_validator_command(non_experimental_deq)
        with self.assertRaisesRegex(GPU5BoundaryError, "DEQ release is disabled"):
            build_gpu5_docker_argv(
                PINNED_DOCKER_IMAGE,
                experimental_deq,
                expected_source_commit=EXPECTED_COMMIT,
                source_snapshot=self.source_snapshot(),
                model_snapshot=self.model_snapshot(),
                environment=self.environment,
                mounts=self.mounts,
                workdir="/workspace",
                launch_nonce=TEST_NONCE,
                release_profile=True,
            )
        inspection_argv = list(self.docker_argv(experimental_deq))
        profile = inspection_argv.index("io.cognios.execution-profile=inspection")
        inspection_argv[profile] = "io.cognios.execution-profile=release"
        with self.assertRaisesRegex(GPU5BoundaryError, "DEQ release is disabled"):
            self.validate_docker_argv(inspection_argv)
        with self.assertRaisesRegex(GPU5BoundaryError, "DEQ release is disabled"):
            build_gpu5_docker_argv(
                PINNED_DOCKER_IMAGE,
                bounded_experimental_deq,
                expected_source_commit=EXPECTED_COMMIT,
                source_snapshot=self.source_snapshot(),
                model_snapshot=self.model_snapshot(),
                environment=self.environment,
                mounts=self.mounts,
                workdir="/workspace",
                launch_nonce=TEST_NONCE,
                release_profile=True,
            )
        with self.assertRaises(GPU5BoundaryError):
            _validated_validator_command(
                (
                    *experimental_deq[:-1],
                    "--certified-delta-lipschitz-bound",
                    "0.95",
                    "--tolerance",
                    "0.0051",
                    experimental_deq[-1],
                )
            )

    def test_every_guarded_validator_checks_identity_before_artifacts_and_model(
        self,
    ) -> None:
        script_root = Path(guard.__file__).resolve().parent
        for filename, model_marker in (
            ("validate_agent_completion.py", "verify_artifact_manifest("),
            ("validate_agent_casual_korean.py", "ModelService.for_local_gemma("),
            ("validate_gemma4_runtime.py", "load_local_gemma("),
            ("validate_gemma4_deq.py", "load_local_gemma("),
        ):
            source = (script_root / filename).read_text(encoding="utf-8")
            first_identity = source.index("validate_guarded_gpu5_identity(")
            first_manifest = source.index("verify_artifact_manifest(")
            first_model = source.index(model_marker)
            with self.subTest(filename=filename):
                self.assertLess(first_identity, first_manifest)
                self.assertLess(first_identity, first_model)
                self.assertGreaterEqual(
                    source.count("validate_guarded_gpu5_identity("), 2
                )

    def test_evidence_parent_and_target_identity_swaps_fail_closed(self) -> None:
        with _open_evidence_target("gpu5-parent-swap.jsonl") as handle:
            self.assertEqual(stat.S_IMODE(os.fstat(handle.file_fd).st_mode), 0o600)
            os.write(handle.file_fd, b'{"ok":true}\n')
            os.fsync(handle.file_fd)
            moved_root = self.evidence_root.with_name("server-evidence-old")
            self.evidence_root.rename(moved_root)
            self.evidence_root.mkdir(mode=0o700)
            with self.assertRaisesRegex(GPU5BoundaryError, "identity"):
                _evidence_digest(handle)

        target = self.evidence_root / "gpu5-target-swap.jsonl"
        with _open_evidence_target(target.name) as handle:
            os.write(handle.file_fd, b'{"ok":true}\n')
            os.fsync(handle.file_fd)
            target.rename(self.evidence_root / "gpu5-original-moved.jsonl")
            target.write_bytes(b'{"replacement":true}\n')
            target.chmod(0o600)
            with self.assertRaisesRegex(GPU5BoundaryError, "identity"):
                _evidence_digest(handle)

    def test_evidence_root_rejects_group_or_world_writable_modes(self) -> None:
        for mode in (0o720, 0o702, 0o777):
            self.evidence_root.chmod(mode)
            with (
                self.subTest(mode=oct(mode)),
                self.assertRaisesRegex(GPU5BoundaryError, "mode 0700"),
            ):
                with _open_evidence_target(f"gpu5-unsafe-{mode:o}.jsonl"):
                    self.fail("unsafe evidence root unexpectedly accepted")
        self.evidence_root.chmod(0o700)

    def test_evidence_postcheck_rejects_target_mode_change(self) -> None:
        with _open_evidence_target("gpu5-mode-change.jsonl") as handle:
            os.write(handle.file_fd, b'{"ok":true}\n')
            os.fsync(handle.file_fd)
            os.fchmod(handle.file_fd, 0o640)
            with self.assertRaises(GPU5BoundaryError):
                _evidence_digest(handle)

    def test_product_schema_is_checked_against_the_same_hashed_snapshot(self) -> None:
        initial = b'{"not":"product-acceptance"}'
        forged = json.dumps(
            {
                "schema": "cogni.agent.completion.stress.v2",
                "suite": "product-e4b-it-20",
                "status": "passed",
                "all_checks_passed": True,
                "requested_turns": 20,
                "turns": [{"turn": index + 1} for index in range(20)],
                "summary": {"strict_completion_stress_gate_passed": True},
            },
            separators=(",", ":"),
        ).encode("utf-8")
        with _open_evidence_target("gpu5-product-snapshot.json") as handle:
            os.write(handle.file_fd, initial)
            os.fsync(handle.file_fd)
            encoded, size, digest = guard._evidence_snapshot(handle)
            self.assertEqual(encoded, initial)
            self.assertEqual(size, len(initial))
            self.assertEqual(digest, hashlib.sha256(initial).hexdigest())

            os.lseek(handle.file_fd, 0, os.SEEK_SET)
            os.ftruncate(handle.file_fd, 0)
            os.write(handle.file_fd, forged)
            os.fsync(handle.file_fd)

            with self.assertRaises(GPU5BoundaryError):
                guard._validate_product_acceptance_evidence(encoded)

    def test_project_lease_blocks_concurrency_and_stale_crash_payload(self) -> None:
        with _gpu5_project_lease(EXPECTED_COMMIT) as lease:
            with self.assertRaisesRegex(GPU5BoundaryError, "already held"):
                with _gpu5_project_lease(EXPECTED_COMMIT):
                    self.fail("nested lease unexpectedly acquired")
            lease.mark_safe_to_release()
        self.assertEqual(self.lease_path.read_bytes(), b"")
        self.lease_path.write_text("stale-crash-payload", encoding="ascii")
        self.lease_path.chmod(0o600)
        with self.assertRaisesRegex(GPU5BoundaryError, "stale"):
            with _gpu5_project_lease(EXPECTED_COMMIT):
                self.fail("stale lease unexpectedly acquired")

    def test_safe_release_clears_payload_even_when_application_raises(self) -> None:
        application_failure = RuntimeError("application failed after safe postflight")
        with self.assertRaises(RuntimeError) as captured:
            with _gpu5_project_lease(EXPECTED_COMMIT) as lease:
                lease.mark_launch_attempted()
                lease.mark_safe_to_release()
                raise application_failure
        self.assertIs(captured.exception, application_failure)
        self.assertEqual(self.lease_path.read_bytes(), b"")
        with _gpu5_project_lease(EXPECTED_COMMIT) as lease:
            lease.mark_safe_to_release()
        self.assertEqual(self.lease_path.read_bytes(), b"")

    def test_native_server_authority_shares_guarded_stage_lease(self) -> None:
        environment = {
            "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
            "CUDA_VISIBLE_DEVICES": PROJECT_GPU_UUID,
            "NVIDIA_VISIBLE_DEVICES": PROJECT_GPU_UUID,
        }
        execution_snapshot = self.native_execution_snapshot()
        with (
            patch.dict(os.environ, environment, clear=True),
            patch.object(guard, "_capture_native_execution_snapshot") as capture,
            patch.object(guard, "preflight_gpu5") as preflight,
            _gpu5_project_lease(EXPECTED_COMMIT) as outer_lease,
        ):
            with self.assertRaisesRegex(GPU5BoundaryError, "already held"):
                with guard.native_gpu5_server_authority(
                    EXPECTED_COMMIT,
                    physical_gpu_index=5,
                    gpu_query_context="native-host",
                    gpu_uuid=PROJECT_GPU_UUID,
                    source_snapshot_root="/sealed/source",
                    source_snapshot_nonce=TEST_NONCE,
                    model_snapshot_root="/sealed/model",
                    model_manifest_path="/sealed/source/config/model.toml",
                    model_manifest_sha256="5" * 64,
                    workspace_root=execution_snapshot.workspace_root,
                    **self.native_handoff_arguments(execution_snapshot),
                ):
                    self.fail("native server unexpectedly acquired the stage lease")
            capture.assert_not_called()
            preflight.assert_not_called()
            outer_lease.mark_safe_to_release()
        self.assertEqual(self.lease_path.read_bytes(), b"")

    def test_native_server_authority_orders_poison_and_safe_release(self) -> None:
        environment = {
            "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
            "CUDA_VISIBLE_DEVICES": PROJECT_GPU_UUID,
            "NVIDIA_VISIBLE_DEVICES": PROJECT_GPU_UUID,
        }
        execution_snapshot = self.native_execution_snapshot()
        snapshot = SimpleNamespace(
            index=5,
            uuid=PROJECT_GPU_UUID,
            compute_processes=(),
            memory_used_mib=0,
        )
        with (
            patch.dict(os.environ, environment, clear=True),
            patch.object(
                guard,
                "_capture_native_execution_snapshot",
                return_value=execution_snapshot,
            ),
            patch.object(guard, "preflight_gpu5", return_value=snapshot),
            guard.native_gpu5_server_authority(
                EXPECTED_COMMIT,
                physical_gpu_index=5,
                gpu_query_context="native-host",
                gpu_uuid=PROJECT_GPU_UUID,
                source_snapshot_root=execution_snapshot.source.root_path,
                source_snapshot_nonce=execution_snapshot.source.launch_nonce,
                model_snapshot_root=execution_snapshot.model.root_path,
                model_manifest_path=execution_snapshot.manifest_path,
                model_manifest_sha256=execution_snapshot.model.manifest_sha256,
                workspace_root=execution_snapshot.workspace_root,
                **self.native_handoff_arguments(execution_snapshot),
            ) as authority,
        ):
            consume_arguments = {
                "expected_source_commit": EXPECTED_COMMIT,
                "physical_gpu_index": 5,
                "gpu_query_context": "native-host",
                "gpu_uuid": PROJECT_GPU_UUID,
                "source_snapshot_root": execution_snapshot.source.root_path,
                "source_snapshot_nonce": execution_snapshot.source.launch_nonce,
                "model_snapshot_root": execution_snapshot.model.root_path,
                "model_manifest_path": execution_snapshot.manifest_path,
                "model_manifest_sha256": execution_snapshot.model.manifest_sha256,
                "workspace_root": execution_snapshot.workspace_root,
                **self.native_handoff_arguments(execution_snapshot),
            }
            with self.assertRaisesRegex(GPU5BoundaryError, "early authority"):
                authority.consume(
                    **{**consume_arguments, "workspace_root": "/wrong/workspace"}
                )
            authority.consume(**consume_arguments)
            with self.assertRaisesRegex(GPU5BoundaryError, "out of order"):
                authority.mark_safe_to_release()
            authority.mark_launch_attempted()
            authority.mark_safe_to_release()
        self.assertEqual(self.lease_path.read_bytes(), b"")

    def test_native_server_rejects_tampered_handoff_before_gpu_preflight(self) -> None:
        environment = {
            "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
            "CUDA_VISIBLE_DEVICES": PROJECT_GPU_UUID,
            "NVIDIA_VISIBLE_DEVICES": PROJECT_GPU_UUID,
        }
        execution_snapshot = self.native_execution_snapshot()
        handoff = self.native_handoff_arguments(execution_snapshot)
        handoff["source_identity_digest"] = "a" * 64
        with (
            patch.dict(os.environ, environment, clear=True),
            patch.object(
                guard,
                "_capture_native_execution_snapshot",
                return_value=execution_snapshot,
            ) as capture,
            patch.object(guard, "preflight_gpu5") as preflight,
            self.assertRaisesRegex(GPU5BoundaryError, "handoff"),
        ):
            with guard.native_gpu5_server_authority(
                EXPECTED_COMMIT,
                physical_gpu_index=5,
                gpu_query_context="native-host",
                gpu_uuid=PROJECT_GPU_UUID,
                source_snapshot_root=execution_snapshot.source.root_path,
                source_snapshot_nonce=execution_snapshot.source.launch_nonce,
                model_snapshot_root=execution_snapshot.model.root_path,
                model_manifest_path=execution_snapshot.manifest_path,
                model_manifest_sha256=execution_snapshot.model.manifest_sha256,
                workspace_root=execution_snapshot.workspace_root,
                **handoff,
            ):
                self.fail("tampered native snapshot handoff was accepted")
        capture.assert_called_once()
        preflight.assert_not_called()

    def test_native_server_rejects_workspace_mismatch_before_gpu_preflight(
        self,
    ) -> None:
        environment = {
            "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
            "CUDA_VISIBLE_DEVICES": PROJECT_GPU_UUID,
            "NVIDIA_VISIBLE_DEVICES": PROJECT_GPU_UUID,
        }
        execution_snapshot = self.native_execution_snapshot()
        with (
            patch.dict(os.environ, environment, clear=True),
            patch.object(
                guard,
                "_capture_native_execution_snapshot",
                return_value=execution_snapshot,
            ) as capture,
            patch.object(guard, "preflight_gpu5") as preflight,
            self.assertRaisesRegex(GPU5BoundaryError, "workspace handoff"),
        ):
            with guard.native_gpu5_server_authority(
                EXPECTED_COMMIT,
                physical_gpu_index=5,
                gpu_query_context="native-host",
                gpu_uuid=PROJECT_GPU_UUID,
                source_snapshot_root=execution_snapshot.source.root_path,
                source_snapshot_nonce=execution_snapshot.source.launch_nonce,
                model_snapshot_root=execution_snapshot.model.root_path,
                model_manifest_path=execution_snapshot.manifest_path,
                model_manifest_sha256=execution_snapshot.model.manifest_sha256,
                workspace_root="/different/trusted/workspace",
                **self.native_handoff_arguments(execution_snapshot),
            ):
                self.fail("mismatched native workspace handoff was accepted")
        capture.assert_called_once()
        preflight.assert_not_called()

    def test_cli_prints_the_same_fixed_environment_contract(self) -> None:
        output = StringIO()
        arguments = [
            "docker-argv",
            "--image",
            PINNED_DOCKER_IMAGE,
            "--expected-source-commit",
            EXPECTED_COMMIT,
            "--workdir",
            "/workspace",
            "--",
            *self.runtime_command(),
        ]
        with (
            patch.object(
                guard.secrets,
                "token_hex",
                side_effect=lambda size: TEST_NONCE if size == 16 else TEST_NONCE[:12],
            ),
            patch.object(
                guard,
                "prepare_source_snapshot",
                return_value=self.source_snapshot(),
            ),
            patch.object(
                guard,
                "_prepare_docker_model_snapshot",
                return_value=self.model_snapshot(),
            ),
            patch.object(guard, "capture_execution_scope", return_value=self.scope()),
            patch("sys.stdout", output),
        ):
            self.assertEqual(guard.main(arguments), 0)
        payload = output.getvalue()
        self.assertIn(str(self.docker), payload)
        self.assertIn("--pull=never", payload)
        self.assertIn(guard.DOCKER_HOST_URI, payload)
        self.assertIn(str(self.model_snapshot_path.resolve()), payload)
        self.assertNotIn(str(self.model_root.resolve()), payload)

    def test_run_cli_rejects_ordinary_host_python_before_docker_or_gpu(self) -> None:
        output = StringIO()
        arguments = [
            "run",
            "--image",
            PINNED_DOCKER_IMAGE,
            "--expected-source-commit",
            EXPECTED_COMMIT,
            "--timeout",
            "60",
            "--evidence-filename",
            "gpu5-bootstrap-rejected.jsonl",
            "--",
            *self.runtime_command(),
        ]
        with (
            patch.object(
                guard,
                "sys",
                SimpleNamespace(
                    flags=SimpleNamespace(isolated=0),
                    dont_write_bytecode=False,
                    pycache_prefix=None,
                ),
            ),
            patch.object(guard, "run_guarded_gpu5_container") as guarded_run,
            patch("sys.stdout", output),
        ):
            self.assertEqual(guard.main(arguments), 2)
        guarded_run.assert_not_called()
        payload = json.loads(output.getvalue())
        self.assertEqual(payload["status"], "failed_closed")
        self.assertIn("isolated", payload["error"].casefold())

    def test_run_cli_requires_external_scheduler_reservation_before_execution(
        self,
    ) -> None:
        output = StringIO()
        arguments = [
            "run",
            "--image",
            PINNED_DOCKER_IMAGE,
            "--expected-source-commit",
            EXPECTED_COMMIT,
            "--timeout",
            "60",
            "--evidence-filename",
            "gpu5-scheduler-blocked.jsonl",
            "--",
            *self.runtime_command(),
        ]
        with (
            patch.object(guard, "_validate_run_bootstrap"),
            patch.object(
                guard,
                "_require_external_gpu5_scheduler_reservation",
                side_effect=GPU5BoundaryError("external scheduler reservation missing"),
            ) as reservation,
            patch.object(guard, "run_guarded_gpu5_container") as guarded_run,
            patch.object(guard, "preflight_gpu5") as preflight,
            patch("sys.stdout", output),
        ):
            self.assertEqual(guard.main(arguments), 2)
        reservation.assert_called_once_with(
            EXPECTED_COMMIT,
            required_run_seconds=60.0,
        )
        guarded_run.assert_not_called()
        preflight.assert_not_called()
        payload = json.loads(output.getvalue())
        self.assertEqual(payload["status"], "failed_closed")
        self.assertIn("scheduler", payload["error"].casefold())

    def _run_runners(
        self,
        docker_response,
        *,
        postflight=None,
        absence_returncode=0,
        absence_stdout="",
        absence_stderr="",
    ):
        smi = _RecordedRunner(
            _idle_smi_responses()
            + (_idle_smi_responses() if postflight is None else list(postflight))
        )
        docker = _RecordedRunner([docker_response])
        cleanup = _RecordedRunner(
            [
                _completed(),
                _completed(returncode=1),
                _completed(
                    stdout=absence_stdout,
                    stderr=absence_stderr,
                    returncode=absence_returncode,
                ),
                _completed(
                    stdout=absence_stdout,
                    stderr=absence_stderr,
                    returncode=absence_returncode,
                ),
            ]
        )
        return smi, docker, cleanup

    def test_cleanup_absence_proof_distinguishes_absent_daemon_error_and_exists(
        self,
    ) -> None:
        absent = _RecordedRunner([_completed(returncode=1), _completed()])
        proof = _ensure_container_absent(
            TEST_CONTAINER_NAME,
            expected_source_commit=EXPECTED_COMMIT,
            launch_nonce=TEST_NONCE,
            runner=absent,
        )
        self.assertTrue(proof["container_absent"])
        self.assertEqual(
            absent.calls[1][0],
            (
                str(self.docker),
                "--host",
                guard.DOCKER_HOST_URI,
                "--config",
                str(self.docker_config),
                "ps",
                "--all",
                "--filter",
                f"name=^/{TEST_CONTAINER_NAME}$",
                "--format",
                "{{.Names}}",
            ),
        )
        for verification in (
            _completed(returncode=1, stderr="Cannot connect to the Docker daemon"),
            _completed(stdout=f"{TEST_CONTAINER_NAME}\n"),
        ):
            runner = _RecordedRunner(
                [_completed(returncode=1), verification, verification]
            )
            with self.assertRaises(GPU5BoundaryError):
                _ensure_container_absent(
                    TEST_CONTAINER_NAME,
                    expected_source_commit=EXPECTED_COMMIT,
                    launch_nonce=TEST_NONCE,
                    runner=runner,
                )

    def test_owned_cleanup_inspects_exact_labels_before_stop_and_rm(self) -> None:
        runner = _RecordedRunner(
            [
                _inspect_row(),
                _completed(),
                _inspect_row(),
                _completed(),
                _completed(),
            ]
        )
        proof = _ensure_container_absent(
            TEST_CONTAINER_NAME,
            expected_source_commit=EXPECTED_COMMIT,
            launch_nonce=TEST_NONCE,
            runner=runner,
        )
        self.assertTrue(proof["container_absent"])
        commands = [call[0][5] for call in runner.calls]
        self.assertEqual(commands, ["inspect", "stop", "inspect", "rm", "ps"])
        self.assertEqual(runner.calls[1][0][-1], TEST_CONTAINER_ID)
        self.assertEqual(runner.calls[3][0][-1], TEST_CONTAINER_ID)

    def test_cleanup_groups_multiple_fatal_controls_after_best_effort(self) -> None:
        stop_control = KeyboardInterrupt("stop-control")
        rm_control = SystemExit(9)
        runner = _RecordedRunner(
            [
                _inspect_row(),
                stop_control,
                _inspect_row(),
                rm_control,
                _completed(),
            ]
        )
        with self.assertRaises(KeyboardInterrupt) as captured:
            _ensure_container_absent(
                TEST_CONTAINER_NAME,
                expected_source_commit=EXPECTED_COMMIT,
                launch_nonce=TEST_NONCE,
                runner=runner,
            )
        self.assertIs(captured.exception, stop_control)
        aggregate = captured.exception.__cause__
        self.assertIsInstance(aggregate, GPU5AggregateError)
        self.assertEqual(len(aggregate.failures), 2)
        self.assertIs(aggregate.failures[0], rm_control)
        self.assertIsInstance(aggregate.failures[1], GPU5CleanupError)
        self.assertIs(aggregate.evidence_error, aggregate.failures[1])
        self.assertTrue(aggregate.failures[1].evidence["container_absent"])
        self.assertEqual(
            [call[0][5] for call in runner.calls],
            ["inspect", "stop", "inspect", "rm", "ps"],
        )
        self.assertEqual(
            [item["action"] for item in aggregate.failures[1].evidence["errors"]],
            ["stop_owned_id", "rm_owned_id"],
        )

    def test_cleanup_single_fatal_rethrows_same_object_with_evidence(self) -> None:
        stop_control = KeyboardInterrupt("stop-control")
        runner = _RecordedRunner(
            [
                _inspect_row(),
                stop_control,
                _inspect_row(),
                _completed(),
                _completed(),
            ]
        )
        with self.assertRaises(KeyboardInterrupt) as captured:
            _ensure_container_absent(
                TEST_CONTAINER_NAME,
                expected_source_commit=EXPECTED_COMMIT,
                launch_nonce=TEST_NONCE,
                runner=runner,
            )
        self.assertIs(captured.exception, stop_control)
        cleanup_failure = captured.exception.gpu5_cleanup_error
        self.assertIsInstance(cleanup_failure, GPU5CleanupError)
        self.assertTrue(cleanup_failure.evidence["container_absent"])
        self.assertEqual(
            [call[0][5] for call in runner.calls],
            ["inspect", "stop", "inspect", "rm", "ps"],
        )

    def test_preflight_name_collision_never_runs_or_queries_gpu(self) -> None:
        cleanup = _RecordedRunner([_completed(stdout=f"{TEST_CONTAINER_NAME}\n")])
        docker = _RecordedRunner([])
        smi = _RecordedRunner([])
        with (
            patch.object(guard.secrets, "token_hex", return_value=TEST_NONCE),
            self.assertRaisesRegex(GPU5BoundaryError, "already occupied"),
        ):
            run_guarded_gpu5_container(
                PINNED_DOCKER_IMAGE,
                self.runtime_command(),
                expected_source_commit=EXPECTED_COMMIT,
                environment=self.environment,
                mounts=(),
                workdir="/workspace",
                run_timeout_seconds=60,
                evidence_filename="gpu5-preflight-collision.jsonl",
                smi_runner=smi,
                docker_runner=docker,
                cleanup_runner=cleanup,
                scope_reader=lambda _commit, **_kwargs: self.scope(),
                snapshot_factory=lambda _commit, _nonce: self.source_snapshot(_nonce),
                model_snapshot_factory=lambda _source, *, launch_nonce: (
                    self.model_snapshot(launch_nonce)
                ),
            )
        self.assertEqual(len(cleanup.calls), 1)
        self.assertEqual(docker.calls, [])
        self.assertEqual(smi.calls, [])

    def test_model_snapshot_tamper_is_rejected_before_gpu_preflight(self) -> None:
        cleanup = _RecordedRunner([_completed()])
        docker = _RecordedRunner([])
        smi = _RecordedRunner([])

        def tampering_scope(_commit, *, source_snapshot, model_snapshot):
            del source_snapshot
            artifact = Path(model_snapshot.root_path) / "model.bin"
            artifact.chmod(0o644)
            artifact.write_bytes(b"tampered")
            artifact.chmod(0o444)
            guard._model_snapshot_inventory(
                Path(model_snapshot.root_path),
                manifest_entries=(("model.bin", hashlib.sha256(b"x").hexdigest()),),
                manifest_sha256=model_snapshot.manifest_sha256,
            )
            self.fail("tampered model snapshot was accepted")

        with (
            patch.object(guard.secrets, "token_hex", return_value=TEST_NONCE),
            self.assertRaisesRegex(GPU5BoundaryError, "digest mismatch"),
        ):
            run_guarded_gpu5_container(
                PINNED_DOCKER_IMAGE,
                self.runtime_command(),
                expected_source_commit=EXPECTED_COMMIT,
                environment=self.environment,
                mounts=(),
                workdir="/workspace",
                run_timeout_seconds=60,
                evidence_filename="gpu5-model-tamper.jsonl",
                smi_runner=smi,
                docker_runner=docker,
                cleanup_runner=cleanup,
                scope_reader=tampering_scope,
                snapshot_factory=lambda _commit, _nonce: self.source_snapshot(_nonce),
                model_snapshot_factory=lambda _source, *, launch_nonce: (
                    self.model_snapshot(launch_nonce)
                ),
            )
        self.assertEqual(len(cleanup.calls), 1)
        self.assertEqual(docker.calls, [])
        self.assertEqual(smi.calls, [])

    @unittest.skipUnless(os.name == "posix", "Linux sealed mode contract")
    def test_model_snapshot_mode_tamper_is_rejected_before_gpu_preflight(
        self,
    ) -> None:
        cleanup = _RecordedRunner([_completed()])
        docker = _RecordedRunner([])
        smi = _RecordedRunner([])

        def tampering_scope(_commit, *, source_snapshot, model_snapshot):
            del source_snapshot
            artifact = Path(model_snapshot.root_path) / "model.bin"
            artifact.chmod(0o644)
            guard._model_snapshot_inventory(
                Path(model_snapshot.root_path),
                manifest_entries=(("model.bin", hashlib.sha256(b"x").hexdigest()),),
                manifest_sha256=model_snapshot.manifest_sha256,
            )
            self.fail("model snapshot mode tamper was accepted")

        with (
            patch.object(guard.secrets, "token_hex", return_value=TEST_NONCE),
            self.assertRaises(GPU5BoundaryError) as caught,
        ):
            run_guarded_gpu5_container(
                PINNED_DOCKER_IMAGE,
                self.runtime_command(),
                expected_source_commit=EXPECTED_COMMIT,
                environment=self.environment,
                mounts=(),
                workdir="/workspace",
                run_timeout_seconds=60,
                evidence_filename="gpu5-model-mode-tamper.jsonl",
                smi_runner=smi,
                docker_runner=docker,
                cleanup_runner=cleanup,
                scope_reader=tampering_scope,
                snapshot_factory=lambda _commit, _nonce: self.source_snapshot(_nonce),
                model_snapshot_factory=lambda _source, *, launch_nonce: (
                    self.model_snapshot(launch_nonce)
                ),
            )
        self.assertEqual(
            str(caught.exception),
            "model snapshot contains an unsafe file",
        )
        self.assertEqual(len(cleanup.calls), 1)
        self.assertEqual(docker.calls, [])
        self.assertEqual(smi.calls, [])

    def test_post_preflight_foreign_collision_never_stops_or_removes(self) -> None:
        smi = _RecordedRunner(_idle_smi_responses() + _idle_smi_responses())
        docker = _RecordedRunner([_completed(returncode=125)])
        cleanup = _RecordedRunner(
            [
                _completed(),
                _inspect_row(labels={"io.cognios.guard": "gpu5"}),
                _completed(stdout=f"{TEST_CONTAINER_NAME}\n"),
            ]
        )
        with (
            patch.object(guard.secrets, "token_hex", return_value=TEST_NONCE),
            self.assertRaises(GPU5DockerExecutionError) as captured,
        ):
            run_guarded_gpu5_container(
                PINNED_DOCKER_IMAGE,
                self.runtime_command(),
                expected_source_commit=EXPECTED_COMMIT,
                environment=self.environment,
                mounts=(),
                workdir="/workspace",
                run_timeout_seconds=60,
                evidence_filename="gpu5-post-preflight-collision.jsonl",
                smi_runner=smi,
                docker_runner=docker,
                cleanup_runner=cleanup,
                scope_reader=lambda _commit, **_kwargs: self.scope(),
                snapshot_factory=lambda _commit, _nonce: self.source_snapshot(_nonce),
                model_snapshot_factory=lambda _source, *, launch_nonce: (
                    self.model_snapshot(launch_nonce)
                ),
            )
        self.assertEqual(
            captured.exception.evidence["cleanup_error"], "GPU5CleanupError"
        )
        commands = [call[0][5] for call in cleanup.calls]
        self.assertEqual(commands, ["ps", "inspect", "ps"])
        self.assertNotIn("stop", commands)
        self.assertNotIn("rm", commands)

    def test_name_reuse_race_after_stop_never_removes_replacement(self) -> None:
        smi = _RecordedRunner(_idle_smi_responses() + _idle_smi_responses())
        docker = _RecordedRunner([subprocess.TimeoutExpired(cmd="docker", timeout=60)])
        cleanup = _RecordedRunner(
            [
                _completed(),
                _inspect_row(),
                _completed(),
                _completed(returncode=1),
                _completed(stdout=f"{TEST_CONTAINER_NAME}\n"),
                _completed(stdout=f"{TEST_CONTAINER_NAME}\n"),
            ]
        )
        with (
            patch.object(guard.secrets, "token_hex", return_value=TEST_NONCE),
            self.assertRaises(GPU5DockerExecutionError) as captured,
        ):
            run_guarded_gpu5_container(
                PINNED_DOCKER_IMAGE,
                self.runtime_command(),
                expected_source_commit=EXPECTED_COMMIT,
                environment=self.environment,
                mounts=(),
                workdir="/workspace",
                run_timeout_seconds=60,
                evidence_filename="gpu5-name-reuse-race.jsonl",
                smi_runner=smi,
                docker_runner=docker,
                cleanup_runner=cleanup,
                scope_reader=lambda _commit, **_kwargs: self.scope(),
                snapshot_factory=lambda _commit, _nonce: self.source_snapshot(_nonce),
                model_snapshot_factory=lambda _source, *, launch_nonce: (
                    self.model_snapshot(launch_nonce)
                ),
            )
        self.assertEqual(
            captured.exception.evidence["cleanup_error"], "GPU5CleanupError"
        )
        commands = [call[0][5] for call in cleanup.calls]
        self.assertEqual(commands, ["ps", "inspect", "stop", "inspect", "ps", "ps"])
        self.assertNotIn("rm", commands)

    def run_guarded(self, docker_response, filename: str, **runner_options):
        smi, docker, cleanup = self._run_runners(docker_response, **runner_options)
        with patch.object(guard.secrets, "token_hex", return_value=TEST_NONCE):
            result = run_guarded_gpu5_container(
                PINNED_DOCKER_IMAGE,
                self.runtime_command(),
                expected_source_commit=EXPECTED_COMMIT,
                environment=self.environment,
                mounts=(),
                workdir="/workspace",
                run_timeout_seconds=60,
                evidence_filename=filename,
                smi_runner=smi,
                docker_runner=docker,
                cleanup_runner=cleanup,
                scope_reader=lambda _commit, **_kwargs: self.scope(),
                snapshot_factory=lambda _commit, _nonce: self.source_snapshot(_nonce),
                model_snapshot_factory=lambda _source, *, launch_nonce: (
                    self.model_snapshot(launch_nonce)
                ),
            )
        return result, smi, docker, cleanup

    def test_guarded_success_captures_bounded_evidence_cleans_and_postflights(
        self,
    ) -> None:
        result, smi, docker, cleanup = self.run_guarded(
            _completed(returncode=0), "gpu5-success.jsonl"
        )
        self.assertEqual(result.returncode, 0)
        self.assertEqual(result.image_digest, PINNED_DOCKER_IMAGE)
        self.assertIn("io.cognios.execution-profile=release", result.argv)
        self.assertEqual(result.output_policy, "bounded_file_capture")
        self.assertGreater(result.evidence_bytes, 0)
        self.assertEqual(len(result.evidence_sha256), 64)
        self.assertEqual(len(smi.calls), 4)
        self.assertEqual(
            [call[0][5] for call in cleanup.calls], ["ps", "inspect", "ps"]
        )
        kwargs = docker.calls[0][1]
        self.assertTrue(hasattr(kwargs["stdout"], "write"))
        self.assertIs(kwargs["stderr"], subprocess.STDOUT)
        self.assertIn("preexec_fn", kwargs)
        self.assertNotIn("capture_output", kwargs)
        self.assertEqual(kwargs["env"], guard._MINIMAL_HOST_ENVIRONMENT)
        for _argv, call_kwargs in (*smi.calls, *cleanup.calls):
            self.assertEqual(call_kwargs["env"], guard._MINIMAL_HOST_ENVIRONMENT)

    def test_single_fatal_control_rethrows_same_object_after_all_proofs(self) -> None:
        for index, fatal in enumerate(
            (
                KeyboardInterrupt("cancel-run"),
                SystemExit("terminate-run"),
                GeneratorExit("close-run"),
            )
        ):
            smi, docker, cleanup = self._run_runners(fatal)
            with (
                self.subTest(fatal=type(fatal).__name__),
                self.assertRaises(type(fatal)) as captured,
            ):
                run_guarded_gpu5_container(
                    PINNED_DOCKER_IMAGE,
                    self.runtime_command(),
                    expected_source_commit=EXPECTED_COMMIT,
                    environment=self.environment,
                    mounts=(),
                    workdir="/workspace",
                    run_timeout_seconds=60,
                    evidence_filename=f"gpu5-fatal-control-{index}.jsonl",
                    smi_runner=smi,
                    docker_runner=docker,
                    cleanup_runner=cleanup,
                    scope_reader=lambda _commit, **_kwargs: self.scope(),
                    snapshot_factory=lambda _commit, _nonce: self.source_snapshot(
                        _nonce
                    ),
                    model_snapshot_factory=lambda _source, *, launch_nonce: (
                        self.model_snapshot(launch_nonce)
                    ),
                )
            self.assertIs(captured.exception, fatal)
            docker_failure = captured.exception.gpu5_docker_execution_error
            self.assertIsInstance(docker_failure, GPU5DockerExecutionError)
            self.assertEqual(
                docker_failure.evidence["execution_error"], type(fatal).__name__
            )
            self.assertEqual(len(cleanup.calls), 3)
            self.assertEqual(len(smi.calls), 4)
            self.assertEqual(self.lease_path.read_bytes(), b"")

    def test_fatal_control_and_postflight_control_are_grouped_with_evidence(
        self,
    ) -> None:
        execution_control = KeyboardInterrupt("cancel-run")
        postflight_control = SystemExit("postflight-control")
        smi = _RecordedRunner([*_idle_smi_responses(), postflight_control])
        docker = _RecordedRunner([execution_control])
        cleanup = _RecordedRunner(
            [_completed(), _completed(returncode=1), _completed()]
        )
        with (
            patch.object(guard.secrets, "token_hex", return_value=TEST_NONCE),
            self.assertRaises(KeyboardInterrupt) as captured,
        ):
            run_guarded_gpu5_container(
                PINNED_DOCKER_IMAGE,
                self.runtime_command(),
                expected_source_commit=EXPECTED_COMMIT,
                environment=self.environment,
                mounts=(),
                workdir="/workspace",
                run_timeout_seconds=60,
                evidence_filename="gpu5-control-group.jsonl",
                smi_runner=smi,
                docker_runner=docker,
                cleanup_runner=cleanup,
                scope_reader=lambda _commit, **_kwargs: self.scope(),
                snapshot_factory=lambda _commit, _nonce: self.source_snapshot(_nonce),
                model_snapshot_factory=lambda _source, *, launch_nonce: (
                    self.model_snapshot(launch_nonce)
                ),
            )
        self.assertIs(captured.exception, execution_control)
        aggregate = captured.exception.__cause__
        self.assertIsInstance(aggregate, GPU5AggregateError)
        self.assertEqual(len(aggregate.failures), 2)
        self.assertIs(aggregate.failures[0], postflight_control)
        self.assertIsInstance(aggregate.failures[1], GPU5DockerExecutionError)
        self.assertIs(aggregate.evidence_error, aggregate.failures[1])
        self.assertEqual(
            aggregate.failures[1].evidence["execution_error"], "KeyboardInterrupt"
        )
        self.assertEqual(
            aggregate.failures[1].evidence["postflight_error"], "SystemExit"
        )
        self.assertEqual(len(cleanup.calls), 3)
        self.assertEqual(len(smi.calls), 3)
        self.assertTrue(self.lease_path.read_bytes())

    def test_fatal_control_and_cleanup_failure_keep_fatal_top_level(self) -> None:
        execution_control = KeyboardInterrupt("cancel-run")
        smi, docker, cleanup = self._run_runners(
            execution_control,
            absence_returncode=1,
            absence_stderr="Cannot connect to the Docker daemon",
        )
        with (
            patch.object(guard.secrets, "token_hex", return_value=TEST_NONCE),
            self.assertRaises(KeyboardInterrupt) as captured,
        ):
            run_guarded_gpu5_container(
                PINNED_DOCKER_IMAGE,
                self.runtime_command(),
                expected_source_commit=EXPECTED_COMMIT,
                environment=self.environment,
                mounts=(),
                workdir="/workspace",
                run_timeout_seconds=60,
                evidence_filename="gpu5-control-cleanup-group.jsonl",
                smi_runner=smi,
                docker_runner=docker,
                cleanup_runner=cleanup,
                scope_reader=lambda _commit, **_kwargs: self.scope(),
                snapshot_factory=lambda _commit, _nonce: self.source_snapshot(_nonce),
                model_snapshot_factory=lambda _source, *, launch_nonce: (
                    self.model_snapshot(launch_nonce)
                ),
            )
        self.assertIs(captured.exception, execution_control)
        aggregate = captured.exception.__cause__
        self.assertIsInstance(aggregate, GPU5AggregateError)
        self.assertEqual(len(aggregate.failures), 2)
        self.assertIsInstance(aggregate.failures[0], GPU5CleanupError)
        self.assertIsInstance(aggregate.failures[1], GPU5DockerExecutionError)
        self.assertEqual(
            aggregate.failures[1].evidence["cleanup_error"], "GPU5CleanupError"
        )
        self.assertTrue(aggregate.failures[1].evidence["cleanup"])
        self.assertEqual(len(cleanup.calls), 4)
        self.assertEqual(len(smi.calls), 4)
        self.assertTrue(self.lease_path.read_bytes())

    def test_scope_identity_change_after_run_fails_closed(self) -> None:
        snapshots = [self.scope(suffix="0"), self.scope(suffix="1")]

        def scope_reader(_commit, **_kwargs):
            return snapshots.pop(0)

        smi, docker, cleanup = self._run_runners(_completed(returncode=0))
        with self.assertRaises(GPU5DockerExecutionError) as captured:
            run_guarded_gpu5_container(
                PINNED_DOCKER_IMAGE,
                self.runtime_command(),
                expected_source_commit=EXPECTED_COMMIT,
                environment=self.environment,
                mounts=(),
                workdir="/workspace",
                run_timeout_seconds=60,
                evidence_filename="gpu5-scope-swap.jsonl",
                smi_runner=smi,
                docker_runner=docker,
                cleanup_runner=cleanup,
                scope_reader=scope_reader,
                snapshot_factory=lambda _commit, _nonce: self.source_snapshot(_nonce),
                model_snapshot_factory=lambda _source, *, launch_nonce: (
                    self.model_snapshot(launch_nonce)
                ),
            )
        self.assertEqual(
            captured.exception.evidence["scope_error"], "GPU5BoundaryError"
        )
        self.assertEqual(len(cleanup.calls), 3)
        self.assertEqual(len(smi.calls), 4)
        self.assertTrue(self.lease_path.read_bytes())
        with self.assertRaisesRegex(GPU5BoundaryError, "stale"):
            with _gpu5_project_lease(EXPECTED_COMMIT):
                self.fail("scope-uncertain lease unexpectedly reacquired")

    def test_timeout_nonzero_cleanup_and_postflight_fail_closed(self) -> None:
        scenarios = (
            (
                subprocess.TimeoutExpired(cmd="docker", timeout=60),
                {},
                "execution_error",
            ),
            (_completed(returncode=17), {}, "returncode"),
            (
                _completed(returncode=0),
                {
                    "absence_returncode": 1,
                    "absence_stderr": "Cannot connect to the Docker daemon",
                },
                "cleanup_error",
            ),
            (
                _completed(returncode=0),
                {
                    "postflight": [
                        _completed(stdout=f"5, {PROJECT_GPU_UUID}, 99, 1024, 49140\n"),
                        _completed(),
                    ]
                },
                "postflight_error",
            ),
        )
        for index, (response, runner_options, expected_key) in enumerate(scenarios):
            smi, docker, cleanup = self._run_runners(response, **runner_options)
            with (
                self.subTest(expected_key=expected_key),
                self.assertRaises(GPU5DockerExecutionError) as captured,
            ):
                run_guarded_gpu5_container(
                    PINNED_DOCKER_IMAGE,
                    self.runtime_command(),
                    expected_source_commit=EXPECTED_COMMIT,
                    environment=self.environment,
                    mounts=(),
                    workdir="/workspace",
                    run_timeout_seconds=60,
                    evidence_filename=f"gpu5-failure-{index}.jsonl",
                    smi_runner=smi,
                    docker_runner=docker,
                    cleanup_runner=cleanup,
                    scope_reader=lambda _commit, **_kwargs: self.scope(),
                    snapshot_factory=lambda _commit, _nonce: self.source_snapshot(
                        _nonce
                    ),
                    model_snapshot_factory=lambda _source, *, launch_nonce: (
                        self.model_snapshot(launch_nonce)
                    ),
                )
            self.assertEqual(
                len(cleanup.calls), 4 if expected_key == "cleanup_error" else 3
            )
            self.assertEqual(len(smi.calls), 4)
            self.assertIsNotNone(captured.exception.evidence[expected_key])
            self.assertEqual(
                captured.exception.evidence["image_digest"], PINNED_DOCKER_IMAGE
            )
            poison_expected = expected_key in {"cleanup_error", "postflight_error"}
            if poison_expected:
                self.assertTrue(self.lease_path.read_bytes())
                with self.assertRaisesRegex(GPU5BoundaryError, "stale"):
                    with _gpu5_project_lease(EXPECTED_COMMIT):
                        self.fail("uncertain cleanup lease unexpectedly reacquired")
                self.lease_path.write_bytes(b"")
                self.lease_path.chmod(0o600)
            else:
                self.assertEqual(self.lease_path.read_bytes(), b"")


if __name__ == "__main__":
    unittest.main()
