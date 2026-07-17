from __future__ import annotations

from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import scripts.gpu5_boundary_guard as guard
from scripts.gpu5_boundary_guard import (
    GPU5BoundaryError,
    GPU5DockerExecutionError,
    PINNED_DOCKER_IMAGE,
    PROJECT_GPU_UUID,
    _run_nvidia_smi,
    _ensure_container_absent,
    _source_within_allowed_roots,
    _validated_validator_command,
    build_gpu5_docker_argv,
    native_gpu5_environment,
    preflight_gpu5,
    query_gpu5_snapshot,
    require_project_gpu_index,
    run_guarded_gpu5_container,
    validate_gpu5_docker_argv,
)


def _completed(
    *, stdout: str = "", stderr: str = "", returncode: int = 0
) -> SimpleNamespace:
    return SimpleNamespace(stdout=stdout, stderr=stderr, returncode=returncode)


def _idle_smi_responses() -> list[SimpleNamespace]:
    return [
        _completed(stdout=f"5, {PROJECT_GPU_UUID}, 0, 19, 49140\n"),
        _completed(stdout=""),
    ]


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


class TestGPU5BoundaryGuard(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        base = Path(temporary.name)
        self.repo_root = base / "Cogni-OS-2.0-v041"
        self.model_root = base / "gemma4-e4b"
        self.evidence_root = base / "server-evidence"
        for directory in (self.repo_root, self.model_root, self.evidence_root):
            directory.mkdir()
        expected_mounts = {
            self.repo_root: Path("/workspace"),
            self.model_root: Path("/models/gemma4-e4b"),
        }
        patchers = (
            patch.object(guard, "EVIDENCE_HOST_ROOT", self.evidence_root),
            patch.object(
                guard,
                "_ALLOWED_MOUNT_ROOTS",
                (self.repo_root, self.model_root),
            ),
            patch.object(guard, "_EXPECTED_READ_ONLY_MOUNTS", expected_mounts),
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
            f"{self.repo_root}:/workspace:ro",
            f"{self.model_root}:/models/gemma4-e4b:ro",
        )

    def runtime_command(self, *extra: str) -> tuple[str, ...]:
        return (
            "python",
            "/workspace/scripts/validate_gemma4_runtime.py",
            "--model",
            "/models/gemma4-e4b",
            "--manifest",
            "/workspace/config/gemma4-e4b.manifest.toml",
            *extra,
        )

    def completion_command(self, *extra: str) -> tuple[str, ...]:
        return (
            "python",
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
            environment=self.environment,
            mounts=self.mounts,
            workdir="/workspace",
            container_name="cognios-gpu5-0123456789ab",
        )

    def test_project_accepts_only_physical_gpu5(self) -> None:
        self.assertEqual(require_project_gpu_index(5), 5)
        for rejected in (None, True, False, 0, 1, 2, 3, 4, 6, 7, -1, "5"):
            with self.subTest(rejected=rejected), self.assertRaises(GPU5BoundaryError):
                require_project_gpu_index(rejected)

    def test_native_environment_exposes_only_physical_gpu5(self) -> None:
        environment = native_gpu5_environment({"PATH": "/bin"})
        self.assertEqual(environment["CUDA_VISIBLE_DEVICES"], "5")
        self.assertEqual(environment["CUDA_DEVICE_ORDER"], "PCI_BUS_ID")
        self.assertEqual(environment["NVIDIA_VISIBLE_DEVICES"], PROJECT_GPU_UUID)
        inherited = native_gpu5_environment(
            {"CUDA_VISIBLE_DEVICES": "5", "NVIDIA_VISIBLE_DEVICES": PROJECT_GPU_UUID}
        )
        self.assertEqual(inherited["NVIDIA_VISIBLE_DEVICES"], PROJECT_GPU_UUID)
        for value in ("0", "0,5", "4", "6", "7", "all"):
            with self.subTest(value=value), self.assertRaises(GPU5BoundaryError):
                native_gpu5_environment({"CUDA_VISIBLE_DEVICES": value})
            with self.subTest(nvidia_value=value), self.assertRaises(GPU5BoundaryError):
                native_gpu5_environment({"NVIDIA_VISIBLE_DEVICES": value})

    def test_host_snapshot_never_uses_selectorless_or_other_gpu_queries(self) -> None:
        runner = _RecordedRunner(_idle_smi_responses())
        snapshot = query_gpu5_snapshot(runner=runner)
        self.assertEqual(snapshot.physical_index, 5)
        self.assertEqual(snapshot.uuid, PROJECT_GPU_UUID)
        self.assertEqual(len(runner.calls), 2)
        for argv, kwargs in runner.calls:
            self.assertEqual(argv[:3], ("nvidia-smi", "-i", "5"))
            self.assertEqual(kwargs["timeout"], 5.0)
        forbidden = _RecordedRunner([])
        with self.assertRaises(GPU5BoundaryError):
            _run_nvidia_smi(
                ("nvidia-smi", "--query-gpu=index", "--format=csv,noheader"),
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

    def test_docker_contract_is_exact_digest_offline_gpu5_and_read_only(self) -> None:
        argv = self.docker_argv()
        validate_gpu5_docker_argv(argv)
        pairs = tuple(zip(argv, argv[1:]))
        self.assertIn(("--gpus", "device=5"), pairs)
        self.assertIn(("--network", "none"), pairs)
        self.assertIn(("--workdir", "/workspace"), pairs)
        self.assertIn(PINNED_DOCKER_IMAGE, argv)
        volumes = [
            argv[index + 1] for index, token in enumerate(argv) if token == "--volume"
        ]
        self.assertEqual(len(volumes), 2)
        self.assertTrue(all(volume.endswith(":ro") for volume in volumes))
        self.assertTrue(all("/evidence" not in volume for volume in volumes))

    def test_production_argv_rejects_every_non5_or_missing_gpu_selector(self) -> None:
        base = list(self.docker_argv())
        for selector in (
            "all",
            *(f"device={index}" for index in range(8) if index != 5),
        ):
            changed = list(base)
            changed[changed.index("device=5")] = selector
            with self.subTest(selector=selector), self.assertRaises(GPU5BoundaryError):
                validate_gpu5_docker_argv(changed)
        missing = list(base)
        position = missing.index("--gpus")
        del missing[position : position + 2]
        with self.assertRaises(GPU5BoundaryError):
            validate_gpu5_docker_argv(missing)

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
                    environment=arguments["environment"],
                    mounts=arguments["mounts"],
                    workdir=arguments["workdir"],
                )
        for extra in ("--privileged", "--device=/dev/nvidia6", "--network=host"):
            changed = list(self.docker_argv())
            changed.insert(changed.index("--"), extra)
            with self.subTest(extra=extra), self.assertRaises(GPU5BoundaryError):
                validate_gpu5_docker_argv(changed)

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
        ):
            with self.subTest(mounts=bad_mounts), self.assertRaises(GPU5BoundaryError):
                build_gpu5_docker_argv(
                    PINNED_DOCKER_IMAGE,
                    self.runtime_command(),
                    environment=self.environment,
                    mounts=bad_mounts,
                    workdir="/workspace",
                )

    def test_validator_allowlist_rejects_unknown_duplicate_oversized_and_shell_args(
        self,
    ) -> None:
        accepted = (
            self.runtime_command("--event-stream", "--workspace-mib", "512"),
            self.completion_command("--turns", "20", "--timeout", "120"),
            (
                "python",
                "/workspace/scripts/validate_gemma4_deq.py",
                "--model",
                "/models/gemma4-e4b",
                "--manifest",
                "/workspace/config/gemma4-e4b.manifest.toml",
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
                _completed(returncode=1),
                _completed(returncode=1),
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
        name = "cognios-gpu5-0123456789ab"
        absent = _RecordedRunner(
            [_completed(returncode=1), _completed(returncode=1), _completed()]
        )
        proof = _ensure_container_absent(name, runner=absent)
        self.assertTrue(proof["container_absent"])
        self.assertEqual(
            absent.calls[2][0],
            (
                "docker",
                "ps",
                "--all",
                "--filter",
                f"name=^/{name}$",
                "--format",
                "{{.Names}}",
            ),
        )
        for verification in (
            _completed(returncode=1, stderr="Cannot connect to the Docker daemon"),
            _completed(stdout=f"{name}\n"),
        ):
            runner = _RecordedRunner(
                [_completed(returncode=1), _completed(returncode=1), verification]
            )
            with self.assertRaises(GPU5BoundaryError):
                _ensure_container_absent(name, runner=runner)

    def run_guarded(self, docker_response, filename: str, **runner_options):
        smi, docker, cleanup = self._run_runners(docker_response, **runner_options)
        result = run_guarded_gpu5_container(
            PINNED_DOCKER_IMAGE,
            self.runtime_command(),
            environment=self.environment,
            mounts=self.mounts,
            workdir="/workspace",
            run_timeout_seconds=60,
            evidence_filename=filename,
            smi_runner=smi,
            docker_runner=docker,
            cleanup_runner=cleanup,
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
        self.assertEqual(result.output_policy, "bounded_file_capture")
        self.assertGreater(result.evidence_bytes, 0)
        self.assertEqual(len(result.evidence_sha256), 64)
        self.assertEqual(len(smi.calls), 4)
        self.assertEqual([call[0][1] for call in cleanup.calls], ["stop", "rm", "ps"])
        kwargs = docker.calls[0][1]
        self.assertTrue(hasattr(kwargs["stdout"], "write"))
        self.assertIs(kwargs["stderr"], subprocess.STDOUT)
        self.assertIn("preexec_fn", kwargs)
        self.assertNotIn("capture_output", kwargs)

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
                    environment=self.environment,
                    mounts=self.mounts,
                    workdir="/workspace",
                    run_timeout_seconds=60,
                    evidence_filename=f"gpu5-failure-{index}.jsonl",
                    smi_runner=smi,
                    docker_runner=docker,
                    cleanup_runner=cleanup,
                )
            self.assertEqual(len(cleanup.calls), 3)
            self.assertEqual(len(smi.calls), 4)
            self.assertIsNotNone(captured.exception.evidence[expected_key])
            self.assertEqual(
                captured.exception.evidence["image_digest"], PINNED_DOCKER_IMAGE
            )


if __name__ == "__main__":
    unittest.main()
