from __future__ import annotations

from builtins import BaseExceptionGroup
from hashlib import sha256
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from scripts import validate_agent_completion as completion
from scripts import validate_gemma4_deq as deq
from scripts import validate_gemma4_runtime as runtime


RUNTIME_REQUIRED = (
    "--model",
    "model",
    "--manifest",
    "manifest",
    "--physical-gpu-index",
    "5",
    "--gpu-query-context",
    "gpu5-container",
)
DEQ_REQUIRED = (*RUNTIME_REQUIRED, "--allow-uncertified-experimental")


def _telemetry(**changes):
    values = {
        "solver_residual_max": 0.004,
        "max_depth_reached": 100,
        "safe_for_decode": True,
        "unsafe_silent_fallbacks": 0,
        "linear_solve_fallbacks": 0,
        "solver_calls": 3,
        "solver_successes": 3,
        "solver_failures": 0,
        "failed_edges": 0,
        "q_zero_backups": 0,
        "nodes_used": 301,
        "node_capacity": 301,
        "allocated_bytes": 1024,
    }
    values.update(changes)
    return SimpleNamespace(**values)


class TestGPU5ValidatorParsers(unittest.TestCase):
    def test_runtime_and_deq_require_both_guard_selectors(self):
        for parser in (runtime.build_parser(), deq.build_parser()):
            for argv in (
                ("--model", "model", "--manifest", "manifest"),
                (
                    "--model",
                    "model",
                    "--manifest",
                    "manifest",
                    "--physical-gpu-index",
                    "5",
                ),
                (
                    "--model",
                    "model",
                    "--manifest",
                    "manifest",
                    "--gpu-query-context",
                    "gpu5-container",
                ),
            ):
                with self.subTest(parser=parser.prog, argv=argv):
                    with self.assertRaises(SystemExit):
                        parser.parse_args(argv)

    def test_runtime_rejects_nonfinite_and_out_of_range_values(self):
        parser = runtime.build_parser()
        cases = (
            ("--workspace-mib", "0"),
            ("--workspace-mib", "4097"),
            ("--workspace-mib", "1.5"),
            ("--vram-limit-gib", "0.99"),
            ("--vram-limit-gib", "16.7001"),
            ("--vram-limit-gib", "nan"),
            ("--vram-limit-gib", "inf"),
            ("--prompt", ""),
            ("--prompt", "   "),
            ("--prompt", "x" * 513),
            ("--physical-gpu-index", "4"),
            ("--physical-gpu-index", "6"),
            ("--physical-gpu-index", "7"),
        )
        for option, value in cases:
            with (
                self.subTest(option=option, value=value),
                self.assertRaises(SystemExit),
            ):
                parser.parse_args([*RUNTIME_REQUIRED, option, value])

    def test_deq_rejects_nonfinite_and_out_of_range_values(self):
        parser = deq.build_parser()
        cases = (
            ("--layer-index", "129"),
            ("--tolerance", "0.005001"),
            ("--tolerance", "nan"),
            ("--max-iter", "0"),
            ("--history", "65"),
            ("--fallback-steps", "257"),
            ("--fallback-damping", "inf"),
            ("--fallback-damping", "0"),
            ("--contractive-delta-scale", "0"),
            ("--contractive-delta-scale", "1.0001"),
            ("--certified-delta-lipschitz-bound", "1000000.1"),
            ("--certified-delta-lipschitz-bound", "nan"),
            ("--vram-limit-gib", "16.7001"),
            ("--prompt", ""),
            ("--prompt", "   "),
            ("--prompt", "x" * 513),
            ("--physical-gpu-index", "6"),
            ("--physical-gpu-index", "7"),
        )
        for option, value in cases:
            with (
                self.subTest(option=option, value=value),
                self.assertRaises(SystemExit),
            ):
                parser.parse_args([*DEQ_REQUIRED, option, value])

    def test_runtime_and_deq_reject_all_prompt_control_characters(self):
        values = (
            "line\nfeed",
            "carriage\rreturn",
            "horizontal\ttab",
            "nul\x00byte",
            "unit\x1fseparator",
            "delete\x7fcharacter",
        )
        for parser, required in (
            (runtime.build_parser(), RUNTIME_REQUIRED),
            (deq.build_parser(), DEQ_REQUIRED),
        ):
            for value in values:
                with (
                    self.subTest(parser=parser.prog, value=repr(value)),
                    self.assertRaises(SystemExit),
                ):
                    parser.parse_args([*required, "--prompt", value])


class TestGPU5ValidatorScopes(unittest.TestCase):
    def _args(self):
        return SimpleNamespace(
            model="model",
            manifest="manifest",
            physical_gpu_index=5,
            gpu_query_context="gpu5-container",
        )

    def _assert_primary_and_post_error_are_preserved(self, module):
        identity_before = SimpleNamespace(marker="before")
        identity_after = SimpleNamespace(marker="after")
        verified = SimpleNamespace(files=())
        with (
            patch.object(
                module,
                "validate_guarded_gpu5_identity",
                side_effect=(identity_before, identity_after),
            ) as identity_check,
            patch.object(
                module,
                "verify_artifact_manifest",
                side_effect=(verified, verified),
            ) as manifest_check,
        ):
            with self.assertRaises(BaseExceptionGroup) as raised:
                with module._guarded_validation_scope(
                    self._args(), torch_module=SimpleNamespace()
                ):
                    raise RuntimeError("model-load failed")

        failures = raised.exception.exceptions
        self.assertEqual(len(failures), 2)
        self.assertEqual(str(failures[0]), "model-load failed")
        self.assertIn("identity changed", str(failures[1]))
        self.assertEqual(identity_check.call_count, 2)
        self.assertEqual(manifest_check.call_count, 2)

    def test_runtime_model_failure_still_runs_both_postchecks(self):
        self._assert_primary_and_post_error_are_preserved(runtime)

    def test_deq_forward_failure_still_runs_both_postchecks(self):
        self._assert_primary_and_post_error_are_preserved(deq)

    def test_fatal_primary_and_post_failure_preserve_original_objects(self):
        fatal_types = (KeyboardInterrupt, SystemExit, GeneratorExit)
        for module in (runtime, deq):
            for fatal_type in fatal_types:
                primary = fatal_type("primary-control")
                identity_before = SimpleNamespace(marker="before")
                identity_after = SimpleNamespace(marker="after")
                verified = SimpleNamespace(files=())
                with (
                    self.subTest(module=module.__name__, fatal=fatal_type.__name__),
                    patch.object(
                        module,
                        "validate_guarded_gpu5_identity",
                        side_effect=(identity_before, identity_after),
                    ),
                    patch.object(
                        module,
                        "verify_artifact_manifest",
                        side_effect=(verified, verified),
                    ),
                    self.assertRaises(BaseExceptionGroup) as raised,
                ):
                    with module._guarded_validation_scope(
                        self._args(), torch_module=SimpleNamespace()
                    ):
                        raise primary

                failures = raised.exception.exceptions
                self.assertEqual(len(failures), 2)
                self.assertIs(failures[0], primary)
                self.assertIn("identity changed", str(failures[1]))

    def test_post_only_fatal_failure_rethrows_same_object(self):
        for module in (runtime, deq):
            post_failure = KeyboardInterrupt("post-manifest-control")
            identity = SimpleNamespace(marker="stable")
            verified = SimpleNamespace(files=())
            with (
                self.subTest(module=module.__name__),
                patch.object(
                    module,
                    "validate_guarded_gpu5_identity",
                    side_effect=(identity, identity),
                ),
                patch.object(
                    module,
                    "verify_artifact_manifest",
                    side_effect=(verified, post_failure),
                ),
                self.assertRaises(KeyboardInterrupt) as raised,
            ):
                with module._guarded_validation_scope(
                    self._args(), torch_module=SimpleNamespace()
                ):
                    pass
            self.assertIs(raised.exception, post_failure)

    def test_two_simultaneous_fatal_post_failures_are_grouped(self):
        for module in (runtime, deq):
            manifest_failure = KeyboardInterrupt("post-manifest-control")
            identity_failure = GeneratorExit("post-identity-control")
            identity = SimpleNamespace(marker="stable")
            verified = SimpleNamespace(files=())
            with (
                self.subTest(module=module.__name__),
                patch.object(
                    module,
                    "validate_guarded_gpu5_identity",
                    side_effect=(identity, identity_failure),
                ),
                patch.object(
                    module,
                    "verify_artifact_manifest",
                    side_effect=(verified, manifest_failure),
                ),
                self.assertRaises(BaseExceptionGroup) as raised,
            ):
                with module._guarded_validation_scope(
                    self._args(), torch_module=SimpleNamespace()
                ):
                    pass
            self.assertEqual(
                raised.exception.exceptions, (manifest_failure, identity_failure)
            )

    def _assert_main_wraps_execution_failure(self, module, execute_name, args):
        identity = SimpleNamespace(marker="stable")
        verified = SimpleNamespace(files=())
        parser = SimpleNamespace(parse_args=lambda: args)
        patches = [
            patch.object(module, "build_parser", return_value=parser),
            patch.object(
                module,
                "validate_guarded_gpu5_identity",
                side_effect=(identity, identity),
            ),
            patch.object(
                module,
                "verify_artifact_manifest",
                side_effect=(verified, verified),
            ),
            patch.object(
                module, execute_name, side_effect=RuntimeError("forward failed")
            ),
        ]
        if module is runtime:
            patches.append(patch.object(module, "EventEmitter", return_value=object()))
        entered = [patcher.start() for patcher in patches]
        self.addCleanup(lambda: [patcher.stop() for patcher in reversed(patches)])
        with self.assertRaisesRegex(RuntimeError, "forward failed"):
            module.main()
        self.assertEqual(entered[1].call_count, 2)
        self.assertEqual(entered[2].call_count, 2)

    def test_runtime_main_wraps_model_execution_failure(self):
        args = SimpleNamespace(
            model="model",
            manifest="manifest",
            physical_gpu_index=5,
            gpu_query_context="gpu5-container",
            event_stream=False,
        )
        self._assert_main_wraps_execution_failure(runtime, "_execute_runtime", args)

    def test_deq_main_wraps_forward_failure(self):
        args = SimpleNamespace(
            model="model",
            manifest="manifest",
            physical_gpu_index=5,
            gpu_query_context="gpu5-container",
            certified_delta_lipschitz_bound=0.9,
            contractivity_provenance=None,
            allow_uncertified_experimental=True,
        )
        self._assert_main_wraps_execution_failure(deq, "_execute_deq", args)


class TestGPU5ValidatorPostconditions(unittest.TestCase):
    def test_runtime_rejects_residual_and_fallback(self):
        with self.assertRaisesRegex(RuntimeError, "residual"):
            runtime._validate_runtime_postconditions(
                telemetry=_telemetry(solver_residual_max=0.005001),
                finite=True,
                peak_allocated_gib=9.0,
                peak_reserved_gib=10.0,
                vram_limit_gib=16.7,
                requested_depth=100,
            )

    def test_runtime_rejects_each_nonzero_solver_failure_counter(self):
        for counter in ("solver_failures", "failed_edges", "q_zero_backups"):
            with (
                self.subTest(counter=counter),
                self.assertRaisesRegex(RuntimeError, "decode-safe"),
            ):
                runtime._validate_runtime_postconditions(
                    telemetry=_telemetry(**{counter: 1}),
                    finite=True,
                    peak_allocated_gib=9.0,
                    peak_reserved_gib=10.0,
                    vram_limit_gib=16.7,
                    requested_depth=100,
                )

    def test_runtime_transition_converged_is_computed_from_real_conditions(self):
        residual, transition_converged = runtime._validate_runtime_postconditions(
            telemetry=_telemetry(safe_for_decode=True),
            finite=True,
            peak_allocated_gib=9.0,
            peak_reserved_gib=10.0,
            vram_limit_gib=16.7,
            requested_depth=100,
        )
        self.assertEqual(residual, 0.004)
        self.assertIs(transition_converged, True)

    def test_runtime_rejects_invalid_allocated_and_reserved_vram_peaks(self):
        cases = (
            (float("nan"), 10.0),
            (9.0, float("inf")),
            (-0.1, 10.0),
            (10.1, 10.0),
            (9.0, 16.7001),
            (True, 10.0),
            (9.0, False),
        )
        for allocated, reserved in cases:
            with self.subTest(allocated=allocated, reserved=reserved):
                with self.assertRaises(RuntimeError):
                    runtime._validate_runtime_postconditions(
                        telemetry=_telemetry(),
                        finite=True,
                        peak_allocated_gib=allocated,
                        peak_reserved_gib=reserved,
                        vram_limit_gib=16.7,
                        requested_depth=100,
                    )

    def test_runtime_rejects_inconsistent_solver_totals_and_arena_bounds(self):
        cases = (
            {"solver_calls": 2, "solver_successes": 1},
            {"nodes_used": 302, "node_capacity": 301},
            {"nodes_used": 0},
            {"node_capacity": 0},
            {"allocated_bytes": -1},
        )
        for changes in cases:
            with self.subTest(changes=changes), self.assertRaises(RuntimeError):
                runtime._validate_runtime_postconditions(
                    telemetry=_telemetry(**changes),
                    finite=True,
                    peak_allocated_gib=9.0,
                    peak_reserved_gib=10.0,
                    vram_limit_gib=16.7,
                    requested_depth=100,
                )

    def test_runtime_rejects_malformed_boolean_nan_and_counter_types(self):
        cases = (
            {"solver_residual_max": True},
            {"solver_residual_max": float("nan")},
            {"solver_calls": True},
            {"solver_successes": 3.0},
            {"safe_for_decode": 1},
        )
        for changes in cases:
            with self.subTest(changes=changes), self.assertRaises(RuntimeError):
                runtime._validate_runtime_postconditions(
                    telemetry=_telemetry(**changes),
                    finite=True,
                    peak_allocated_gib=9.0,
                    peak_reserved_gib=10.0,
                    vram_limit_gib=16.7,
                    requested_depth=100,
                )
        with self.assertRaisesRegex(RuntimeError, "decode-safe"):
            runtime._validate_runtime_postconditions(
                telemetry=_telemetry(linear_solve_fallbacks=1),
                finite=True,
                peak_allocated_gib=9.0,
                peak_reserved_gib=10.0,
                vram_limit_gib=16.7,
                requested_depth=100,
            )

    def test_deq_rejects_residual_fallback_and_bad_certified_spectral_norm(self):
        base = {
            "converged": True,
            "residual": 0.004,
            "used_fallback": False,
            "spectral_norm": 0.9,
        }
        for change, message in (
            ({"residual": 0.005001}, "residual"),
            ({"used_fallback": True}, "fallback"),
            ({"spectral_norm": 0.9501}, "spectral"),
            ({"spectral_norm": float("nan")}, "spectral"),
        ):
            values = {**base, **change}
            with (
                self.subTest(change=change),
                self.assertRaisesRegex(RuntimeError, message),
            ):
                deq._validate_deq_postconditions(
                    info=SimpleNamespace(**values),
                    finite=True,
                    peak_gib=10.0,
                    vram_limit_gib=16.7,
                    tolerance=0.005,
                    certified=True,
                )


class TestDEQEvidenceClassification(unittest.TestCase):
    def _args(self, **changes):
        values = {
            "allow_uncertified_experimental": False,
            "contractivity_provenance": None,
            "certified_delta_lipschitz_bound": 10.0,
            "contractive_delta_scale": 0.05,
            "layer_index": -1,
        }
        values.update(changes)
        return SimpleNamespace(**values)

    def test_cli_numeric_bound_alone_never_becomes_release_evidence(self):
        verified = SimpleNamespace()
        with self.assertRaisesRegex(
            RuntimeError, "release DEQ certification is unsupported"
        ):
            deq._resolve_contractivity_evidence(self._args(), verified)

        evidence = deq._resolve_contractivity_evidence(
            self._args(allow_uncertified_experimental=True), verified
        )
        self.assertIs(evidence["certified"], False)
        self.assertIs(evidence["release_evidence_eligible"], False)
        self.assertEqual(evidence["evidence_class"], "experimental-non-release")
        self.assertEqual(evidence["effective_bound"], 0.5)

    def test_effective_bound_uses_scale_times_raw_bound(self):
        self.assertEqual(deq._effective_lipschitz_bound(0.05, 10.0), 0.5)
        with self.assertRaisesRegex(RuntimeError, r"scale\*raw_bound"):
            deq._effective_lipschitz_bound(0.1, 10.0)

        parsed = deq.build_parser().parse_args(
            [
                *DEQ_REQUIRED,
                "--contractive-delta-scale",
                "0.05",
                "--certified-delta-lipschitz-bound",
                "10",
            ]
        )
        self.assertEqual(parsed.certified_delta_lipschitz_bound, 10.0)

    def test_manifest_bound_self_attestation_remains_non_release(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            name = "deq-contractivity.json"
            payload = {
                "schema": "cogni.deq.contractivity.provenance.v1",
                "independent_from_runtime_cli": True,
                "method": "formal-global-lipschitz-bound",
                "verifier": "independent-reviewer",
                "tool": "bounded-analysis-v1",
                "analysis_artifact_sha256": "a" * 64,
                "model_revision": "revision-1",
                "layer_index": -1,
                "contractive_delta_scale": 0.05,
                "delta_lipschitz_upper_bound": 10.0,
                "effective_lipschitz_upper_bound": 0.5,
            }
            encoded = json.dumps(payload, sort_keys=True).encode("utf-8")
            (root / name).write_bytes(encoded)
            verified = SimpleNamespace(
                root=root,
                digests=((name, sha256(encoded).hexdigest()),),
                identity=SimpleNamespace(revision="revision-1"),
            )

            evidence = deq._resolve_contractivity_evidence(
                self._args(
                    contractivity_provenance=name,
                    allow_uncertified_experimental=True,
                ),
                verified,
            )

        self.assertIs(evidence["certified"], False)
        self.assertIs(evidence["release_evidence_eligible"], False)
        self.assertEqual(
            evidence["evidence_class"], "experimental-provenance-non-release"
        )
        self.assertEqual(evidence["raw_bound"], 10.0)
        self.assertEqual(evidence["effective_bound"], 0.5)
        self.assertEqual(evidence["provenance_method"], payload["method"])


class _FakeCompletionManager:
    def __init__(
        self,
        session_id: str,
        *,
        start_error: BaseException | None = None,
        shutdown_error: BaseException | None = None,
    ) -> None:
        self.session_id = session_id
        self.start_error = start_error
        self.shutdown_error = shutdown_error
        self.shutdown_calls = 0

    def snapshot(self):
        return {"conversation": [], "status": "idle", "stage": "idle"}

    def start_turn(self, _prompt, _mode):
        if self.start_error is not None:
            raise self.start_error

    def shutdown(self):
        self.shutdown_calls += 1
        if self.shutdown_error is not None:
            raise self.shutdown_error


class TestCompletionMemorySamplingControlFlow(unittest.TestCase):
    def test_ordinary_reader_errors_remain_bounded_unavailable(self):
        def unavailable_rss(_pid):
            raise OSError("rss unavailable")

        def unavailable_gpu(_pid):
            raise RuntimeError("driver unavailable")

        observed = completion._sample_worker_memory(
            123,
            gpu_spot_sample_threshold_bytes=1_000,
            rss_reader=unavailable_rss,
            gpu_reader=unavailable_gpu,
        )

        self.assertEqual(
            observed["worker_rss_spot_sample_status"], "unavailable:OSError"
        )
        self.assertEqual(
            observed["gpu_memory_spot_sample_status"],
            "unavailable:RuntimeError",
        )
        self.assertIsNone(observed["worker_rss_spot_sample_bytes"])
        self.assertIsNone(observed["gpu_memory_spot_sample_bytes"])
        self.assertIsNone(observed["gpu_memory_spot_sample_within_threshold"])
        self.assertIs(observed["spot_sample_observed"], False)
        self.assertEqual(
            observed["sample_scope"], completion.POST_TURN_MEMORY_SAMPLE_SCOPE
        )
        self.assertIs(observed["captures_peak"], False)

    def test_rss_fatal_controls_rethrow_same_object_before_gpu_reader(self):
        for fatal_type in (KeyboardInterrupt, SystemExit, GeneratorExit):
            fatal = fatal_type("rss control")
            gpu_calls = []

            def fatal_rss(_pid, *, error=fatal):
                raise error

            def observed_gpu(_pid):
                gpu_calls.append(_pid)
                return 100, "measured"

            with (
                self.subTest(fatal=fatal_type.__name__),
                self.assertRaises(fatal_type) as raised,
            ):
                completion._sample_worker_memory(
                    123,
                    gpu_spot_sample_threshold_bytes=1_000,
                    rss_reader=fatal_rss,
                    gpu_reader=observed_gpu,
                )
            self.assertIs(raised.exception, fatal)
            self.assertEqual(gpu_calls, [])

    def test_gpu_fatal_controls_rethrow_same_object(self):
        for fatal_type in (KeyboardInterrupt, SystemExit, GeneratorExit):
            fatal = fatal_type("gpu control")

            def fatal_gpu(_pid, *, error=fatal):
                raise error

            with (
                self.subTest(fatal=fatal_type.__name__),
                self.assertRaises(fatal_type) as raised,
            ):
                completion._sample_worker_memory(
                    123,
                    gpu_spot_sample_threshold_bytes=1_000,
                    rss_reader=lambda _pid: 500,
                    gpu_reader=fatal_gpu,
                )
            self.assertIs(raised.exception, fatal)


class TestCompletionFatalControlFlow(unittest.TestCase):
    def _args(self):
        return SimpleNamespace(
            model="model",
            manifest="manifest",
            timeout=120.0,
            turns=1,
            output=None,
            physical_gpu_index=5,
            gpu_query_context="gpu5-container",
        )

    def _execute_with_fatal(
        self,
        fatal: BaseException,
        *,
        post_identity_error: BaseException | None = None,
        shutdown_error: BaseException | None = None,
    ):
        manager_a = _FakeCompletionManager(
            "completion-a",
            start_error=fatal,
            shutdown_error=shutdown_error,
        )
        manager_b = _FakeCompletionManager("completion-b")
        self._last_managers = (manager_a, manager_b)
        identity = SimpleNamespace(as_payload=lambda: {"uuid": "GPU5-stable"})
        identity_effects = (
            (identity, identity)
            if post_identity_error is None
            else (identity, post_identity_error)
        )
        verified = SimpleNamespace(files=())
        factbook = SimpleNamespace(
            model=SimpleNamespace(manifest_sha256="manifest-digest"),
            as_payload=lambda: {},
        )
        service = SimpleNamespace(is_running=False, stop=lambda: None)
        with (
            patch.object(
                completion,
                "validate_guarded_gpu5_identity",
                side_effect=identity_effects,
            ) as identity_check,
            patch.object(
                completion,
                "verify_artifact_manifest",
                side_effect=(verified, verified),
            ) as manifest_check,
            patch.object(
                completion,
                "build_runtime_factbook_from_verified",
                return_value=factbook,
            ),
            patch.object(completion, "_expected_factbook_identity", return_value={}),
            patch.object(
                completion.ModelService,
                "for_local_gemma",
                return_value=service,
            ),
            patch.object(
                completion, "AgentManager", side_effect=(manager_a, manager_b)
            ),
            patch.object(completion, "WorkspaceToolExecutor", return_value=object()),
            patch.object(completion, "RuntimeFactGrounder", return_value=object()),
            patch.object(completion.torch.cuda, "is_available", return_value=True),
            patch.object(completion.torch.cuda, "device_count", return_value=1),
            patch.object(completion.torch.cuda, "current_device", return_value=0),
            patch.object(
                completion.torch.cuda,
                "get_device_name",
                return_value="mock-project-gpu5",
            ),
        ):
            try:
                completion.execute(self._args())
            finally:
                self._last_call_counts = (
                    identity_check.call_count,
                    manifest_check.call_count,
                )

    def test_cancellation_runs_cleanup_and_rethrows_original_object(self):
        fatal = KeyboardInterrupt("cancel-turn")
        with self.assertRaises(KeyboardInterrupt) as raised:
            self._execute_with_fatal(fatal)
        manager_a, manager_b = self._last_managers
        self.assertIs(raised.exception, fatal)
        self.assertEqual(manager_a.shutdown_calls, 1)
        self.assertEqual(manager_b.shutdown_calls, 1)
        self.assertEqual(self._last_call_counts, (2, 2))

    def test_primary_post_and_cleanup_fatals_preserve_all_objects(self):
        primary = KeyboardInterrupt("cancel-turn")
        cleanup = GeneratorExit("shutdown-control")
        post = SystemExit("post-identity-control")
        with self.assertRaises(BaseExceptionGroup) as raised:
            self._execute_with_fatal(
                primary,
                post_identity_error=post,
                shutdown_error=cleanup,
            )
        manager_a, manager_b = self._last_managers
        self.assertEqual(raised.exception.exceptions, (primary, cleanup, post))
        self.assertEqual(manager_a.shutdown_calls, 1)
        self.assertEqual(manager_b.shutdown_calls, 1)
        self.assertEqual(self._last_call_counts, (2, 2))


if __name__ == "__main__":
    unittest.main()
