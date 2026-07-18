from __future__ import annotations

from contextlib import contextmanager
from http.client import HTTPConnection
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from threading import Event, Thread
from types import SimpleNamespace
from time import monotonic, sleep
import unittest
from unittest.mock import Mock, patch

import cogni_demo.server as demo_server
import scripts.probe_native_gpu5_identity as identity_probe
import scripts.run_cogniboard_server as server_bootstrap
from cogni_demo.server import (
    _agent_failure_route,
    DemoHTTPServer,
    JobAlreadyRunningError,
    JobManager,
    SessionMetadata,
    WorkerLaunch,
    WorkerTerminationError,
    find_live_session,
    open_graphical_app,
    ping_session,
    production_launch_factory,
    read_session_metadata,
    remove_session_metadata,
    write_session_metadata,
    main,
)
from cogni_os.gpu_lease import (
    GPULeaseBusyError,
    GPULeaseManager,
)


ROOT = Path(__file__).resolve().parents[1]
FAKE_WORKER = ROOT / "tests" / "fixtures" / "fake_demo_worker.py"
SERVER_SOURCE_COMMIT = "a" * 40
SERVER_GPU_UUID = "GPU-84d7eeb0-65e0-a5b1-d7db-d09ef59fe03a"
SERVER_SNAPSHOT_NONCE = "b" * 32
SERVER_MODEL_MANIFEST_SHA256 = "c" * 64
SERVER_SOURCE_CONTENT_DIGEST = "d" * 64
SERVER_SOURCE_IDENTITY_DIGEST = "e" * 64
SERVER_MODEL_CONTENT_DIGEST = "f" * 64
SERVER_MODEL_IDENTITY_DIGEST = "0" * 64
SERVER_SNAPSHOT_FILE_COUNT = 1
SERVER_MODEL_TOTAL_BYTES = 1
SERVER_WORKSPACE_ROOT = (
    Path(tempfile.gettempdir()).resolve() / "cogniboard-native-test-workspace"
)
SERVER_WORKSPACE_ROOT.mkdir(parents=True, exist_ok=True)


def server_gpu5_argv(
    *,
    model: str | Path | None = None,
    manifest: str | Path | None = None,
    assets: str | Path | None = None,
    workspace_root: str | Path | None = None,
    source_snapshot_root: str | Path | None = None,
    model_snapshot_root: str | Path | None = None,
) -> list[str]:
    selected_model = ROOT if model is None else Path(model)
    selected_manifest = (
        ROOT / "config" / "gemma4-e4b-it.manifest.toml"
        if manifest is None
        else Path(manifest)
    )
    selected_assets = ROOT / "cogni_demo" / "static" if assets is None else Path(assets)
    selected_workspace = (
        SERVER_WORKSPACE_ROOT if workspace_root is None else Path(workspace_root)
    )
    selected_source_snapshot = (
        ROOT if source_snapshot_root is None else Path(source_snapshot_root)
    )
    selected_model_snapshot = (
        selected_model if model_snapshot_root is None else Path(model_snapshot_root)
    )
    source_metadata = selected_source_snapshot.stat()
    model_metadata = selected_model_snapshot.stat()
    return [
        "--no-browser",
        "--model",
        str(selected_model),
        "--manifest",
        str(selected_manifest),
        "--assets",
        str(selected_assets),
        "--validation-profile",
        "server-gpu5-native",
        "--validation-physical-gpu-index",
        "5",
        "--validation-gpu-query-context",
        "native-host",
        "--validation-gpu-uuid",
        SERVER_GPU_UUID,
        "--expected-source-commit",
        SERVER_SOURCE_COMMIT,
        "--native-snapshot-stage",
        "sealed",
        "--native-source-snapshot-root",
        str(selected_source_snapshot),
        "--native-source-snapshot-nonce",
        SERVER_SNAPSHOT_NONCE,
        "--native-workspace-root",
        str(selected_workspace),
        "--native-source-content-digest",
        SERVER_SOURCE_CONTENT_DIGEST,
        "--native-source-identity-digest",
        SERVER_SOURCE_IDENTITY_DIGEST,
        "--native-source-file-count",
        str(SERVER_SNAPSHOT_FILE_COUNT),
        "--native-source-root-device",
        str(source_metadata.st_dev),
        "--native-source-root-inode",
        str(source_metadata.st_ino),
        "--native-model-snapshot-root",
        str(selected_model_snapshot),
        "--native-model-manifest-sha256",
        SERVER_MODEL_MANIFEST_SHA256,
        "--native-model-content-digest",
        SERVER_MODEL_CONTENT_DIGEST,
        "--native-model-identity-digest",
        SERVER_MODEL_IDENTITY_DIGEST,
        "--native-model-file-count",
        str(SERVER_SNAPSHOT_FILE_COUNT),
        "--native-model-root-device",
        str(model_metadata.st_dev),
        "--native-model-root-inode",
        str(model_metadata.st_ino),
        "--native-model-total-bytes",
        str(SERVER_MODEL_TOTAL_BYTES),
    ]


def native_execution_snapshot(
    *,
    source_root: str | Path = ROOT,
    model_root: str | Path = ROOT,
    manifest_path: str | Path | None = None,
    workspace_root: str | Path | None = None,
):
    import scripts.gpu5_boundary_guard as gpu_guard

    source = Path(source_root).resolve(strict=True)
    model = Path(model_root).resolve(strict=True)
    manifest = (
        ROOT / "config" / "gemma4-e4b-it.manifest.toml"
        if manifest_path is None
        else Path(manifest_path).resolve(strict=True)
    )
    workspace = (
        SERVER_WORKSPACE_ROOT
        if workspace_root is None
        else Path(workspace_root).resolve(strict=True)
    )
    source_metadata = source.stat()
    model_metadata = model.stat()
    source_snapshot = gpu_guard.SourceSnapshot(
        source_commit=SERVER_SOURCE_COMMIT,
        launch_nonce=SERVER_SNAPSHOT_NONCE,
        root_path=str(source),
        root_device=int(source_metadata.st_dev),
        root_inode=int(source_metadata.st_ino),
        root_mode=int(source_metadata.st_mode & 0o7777),
        file_count=SERVER_SNAPSHOT_FILE_COUNT,
        content_digest=SERVER_SOURCE_CONTENT_DIGEST,
        identity_digest=SERVER_SOURCE_IDENTITY_DIGEST,
    )
    model_snapshot = gpu_guard.ModelSnapshot(
        root_path=str(model),
        root_device=int(model_metadata.st_dev),
        root_inode=int(model_metadata.st_ino),
        root_mode=int(model_metadata.st_mode & 0o7777),
        file_count=SERVER_SNAPSHOT_FILE_COUNT,
        total_bytes=SERVER_MODEL_TOTAL_BYTES,
        manifest_sha256=SERVER_MODEL_MANIFEST_SHA256,
        content_digest=SERVER_MODEL_CONTENT_DIGEST,
        identity_digest=SERVER_MODEL_IDENTITY_DIGEST,
    )
    return gpu_guard.NativeExecutionSnapshot(
        source=source_snapshot,
        model=model_snapshot,
        manifest_path=str(manifest),
        workspace_root=str(workspace),
    )


def native_snapshot_handoff(execution_snapshot) -> dict[str, object]:
    return {
        "source_snapshot_root": execution_snapshot.source.root_path,
        "source_snapshot_nonce": execution_snapshot.source.launch_nonce,
        "workspace_root": execution_snapshot.workspace_root,
        "source_content_digest": execution_snapshot.source.content_digest,
        "source_identity_digest": execution_snapshot.source.identity_digest,
        "source_file_count": execution_snapshot.source.file_count,
        "source_root_device": execution_snapshot.source.root_device,
        "source_root_inode": execution_snapshot.source.root_inode,
        "model_snapshot_root": execution_snapshot.model.root_path,
        "model_manifest_path": execution_snapshot.manifest_path,
        "model_manifest_sha256": execution_snapshot.model.manifest_sha256,
        "model_content_digest": execution_snapshot.model.content_digest,
        "model_identity_digest": execution_snapshot.model.identity_digest,
        "model_file_count": execution_snapshot.model.file_count,
        "model_root_device": execution_snapshot.model.root_device,
        "model_root_inode": execution_snapshot.model.root_inode,
        "model_total_bytes": execution_snapshot.model.total_bytes,
    }


def native_test_authority(*, execution_snapshot=None):
    import scripts.gpu5_boundary_guard as gpu_guard

    snapshot = execution_snapshot or native_execution_snapshot()
    return gpu_guard.NativeGPU5ServerAuthority(
        expected_source_commit=SERVER_SOURCE_COMMIT,
        physical_gpu_index=5,
        gpu_query_context="native-host",
        gpu_uuid=SERVER_GPU_UUID,
        preflight=SimpleNamespace(),
        checkout=gpu_guard.WorkingCheckoutIdentity(
            source_commit=SERVER_SOURCE_COMMIT,
            content_digest="1" * 64,
            identity_digest="2" * 64,
            file_count=1,
        ),
        execution_snapshot=snapshot,
        _lease=Mock(),
    )


@contextmanager
def admitted_native_lifecycle(*_args, **_kwargs):
    yield


def launch_factory_for(mode: str):
    def launch(_prompt: str) -> WorkerLaunch:
        environment = os.environ.copy()
        environment["PYTHONUNBUFFERED"] = "1"
        return WorkerLaunch(
            (sys.executable, "-u", str(FAKE_WORKER), mode), ROOT, environment
        )

    return launch


def manager_for(mode: str, *, timeout: float = 10.0) -> JobManager:
    return JobManager(launch_factory_for(mode), max_runtime_seconds=timeout)


class _RecordingLeaseManager(GPULeaseManager):
    def __init__(self, events: list[str]) -> None:
        super().__init__()
        self.events = events
        self.status_at_release: str | None = None
        self.process_code_at_release: int | None = None
        self.job_manager: JobManager | None = None

    def acquire(self, *args, **kwargs):
        self.events.append("acquire")
        return super().acquire(*args, **kwargs)

    def release(self, lease):
        self.events.append("release")
        manager = self.job_manager
        if manager is not None:
            self.status_at_release = manager.snapshot()["status"]
            process = manager._process
            self.process_code_at_release = None if process is None else process.poll()
        return super().release(lease)


class _PreSpawnPausingLeaseManager(GPULeaseManager):
    def __init__(self) -> None:
        super().__init__()
        self.acquired = Event()
        self.continue_spawn = Event()

    def acquire(self, *args, **kwargs):
        lease = super().acquire(*args, **kwargs)
        self.acquired.set()
        if not self.continue_spawn.wait(timeout=5.0):
            raise RuntimeError("test did not release the pre-spawn lease gate")
        return lease


class _UnkillableProcess:
    def __init__(self) -> None:
        self.signals = 0
        self.terminations = 0
        self.kills = 0

    def poll(self):
        return None

    def send_signal(self, _signal) -> None:
        self.signals += 1

    def terminate(self) -> None:
        self.terminations += 1

    def kill(self) -> None:
        self.kills += 1

    def wait(self, timeout=None):
        raise subprocess.TimeoutExpired("unkillable", timeout)


def wait_for_terminal(manager: JobManager, timeout: float = 10.0) -> dict:
    deadline = monotonic() + timeout
    while monotonic() < deadline:
        state = manager.snapshot()
        if state["status"] not in {"starting", "running", "cancelling"}:
            return state
        sleep(0.02)
    raise AssertionError("demo job did not become terminal")


class TestDemoJobManager(unittest.TestCase):
    def test_agent_failure_causes_keep_distinct_self_harness_signatures(self) -> None:
        quality = _agent_failure_route("ResponseQualityError")
        worker = _agent_failure_route("WorkerExecutionError")
        unknown = _agent_failure_route("UnexpectedFailure")
        self.assertNotEqual(quality[:2], worker[:2])
        self.assertEqual(quality[2], "cogni_agent/manager.py")
        self.assertEqual(worker[2], "cogni_agent/model_service.py")
        self.assertEqual(unknown[0], "agent_unclassified")

    def test_production_command_is_absolute_shell_free_and_offline(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            model = Path(temporary) / "model"
            model.mkdir()
            manifest = Path(temporary) / "manifest.toml"
            manifest.write_text("[files]", encoding="utf-8")
            with patch.dict(
                os.environ,
                {
                    "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
                    "CUDA_VISIBLE_DEVICES": (
                        "GPU-84d7eeb0-65e0-a5b1-d7db-d09ef59fe03a"
                    ),
                    "NVIDIA_VISIBLE_DEVICES": (
                        "GPU-84d7eeb0-65e0-a5b1-d7db-d09ef59fe03a"
                    ),
                },
                clear=True,
            ):
                boundary = demo_server.GPUExecutionBoundary(
                    physical_gpu_index=5,
                    gpu_query_context="native-host",
                    gpu_uuid="GPU-84d7eeb0-65e0-a5b1-d7db-d09ef59fe03a",
                )
                launch = production_launch_factory(
                    ROOT, model, manifest, gpu_boundary=boundary
                )("x & calc")
        self.assertEqual(Path(launch.command[0]), Path(sys.executable).absolute())
        self.assertEqual(launch.command[1:4], ("-I", "-B", "-u"))
        self.assertIn("--event-stream", launch.command)
        index = launch.command.index("--physical-gpu-index") + 1
        self.assertEqual(launch.command[index], "5")
        context = launch.command.index("--gpu-query-context") + 1
        self.assertEqual(launch.command[context], "native-host")
        prompt_index = launch.command.index("--prompt") + 1
        self.assertEqual(launch.command[prompt_index], "x & calc")
        self.assertEqual(launch.environment["TRANSFORMERS_OFFLINE"], "1")
        self.assertEqual(launch.environment["HF_HUB_OFFLINE"], "1")
        self.assertEqual(launch.environment["CUDA_DEVICE_ORDER"], "PCI_BUS_ID")
        self.assertEqual(
            launch.environment["CUDA_VISIBLE_DEVICES"],
            "GPU-84d7eeb0-65e0-a5b1-d7db-d09ef59fe03a",
        )
        self.assertEqual(
            launch.environment["NVIDIA_VISIBLE_DEVICES"],
            "GPU-84d7eeb0-65e0-a5b1-d7db-d09ef59fe03a",
        )
        self.assertEqual(launch.environment["LD_PRELOAD"], "")
        self.assertEqual(
            launch.environment["PYTHONPATH"],
            "/nonexistent-cogniboard-pythonpath",
        )
        self.assertEqual(launch.environment["PYTHONNOUSERSITE"], "1")
        self.assertEqual(launch.environment["PYTHONDONTWRITEBYTECODE"], "1")

    def test_production_command_preserves_venv_symlink_invocation(self) -> None:
        if os.name == "nt":
            self.skipTest("POSIX venv symlink semantics are validated on Linux")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            executable = root / "venv" / "bin" / "python"
            executable.parent.mkdir(parents=True)
            try:
                executable.symlink_to(Path(sys.executable).resolve())
            except OSError as error:
                self.skipTest(f"symlink unavailable: {error}")
            model = root / "model"
            model.mkdir()
            manifest = root / "manifest.toml"
            manifest.write_text("[files]", encoding="utf-8")
            environment = {
                "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
                "CUDA_VISIBLE_DEVICES": SERVER_GPU_UUID,
                "NVIDIA_VISIBLE_DEVICES": SERVER_GPU_UUID,
            }
            with patch.dict(os.environ, environment, clear=True):
                worker = production_launch_factory(
                    ROOT,
                    model,
                    manifest,
                    python_executable=executable,
                    gpu_boundary=demo_server.GPUExecutionBoundary(
                        physical_gpu_index=5,
                        gpu_query_context="native-host",
                        gpu_uuid=SERVER_GPU_UUID,
                    ),
                )("")
            self.assertEqual(worker.command[0], os.fspath(executable.absolute()))
            self.assertTrue(Path(worker.command[0]).is_symlink())

    def test_production_command_rejects_nonlexical_python_invocation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            model = Path(temporary) / "model"
            model.mkdir()
            manifest = Path(temporary) / "manifest.toml"
            manifest.write_text("[files]", encoding="utf-8")
            nonlexical = (
                Path(sys.executable).parent
                / ".."
                / Path(sys.executable).parent.name
                / Path(sys.executable).name
            )
            for executable, message in (
                (Path("venv/bin/python"), "absolute path"),
                (nonlexical, "lexically normalized"),
            ):
                with (
                    self.subTest(executable=executable),
                    self.assertRaisesRegex(ValueError, message),
                ):
                    production_launch_factory(
                        ROOT,
                        model,
                        manifest,
                        python_executable=executable,
                    )

    def test_production_command_requires_explicit_server_gpu_boundary(self) -> None:
        boundary_type = getattr(demo_server, "GPUExecutionBoundary", None)
        self.assertIsNotNone(
            boundary_type,
            "live validation has no explicit server GPU execution boundary",
        )
        with tempfile.TemporaryDirectory() as temporary:
            model = Path(temporary) / "model"
            model.mkdir()
            manifest = Path(temporary) / "manifest.toml"
            manifest.write_text("[files]", encoding="utf-8")
            launch = production_launch_factory(ROOT, model, manifest)
            manager = JobManager(launch)
            with patch("cogni_demo.server.subprocess.Popen") as spawn:
                manager.start("live validation")
                state = wait_for_terminal(manager)
            spawn.assert_not_called()
            self.assertEqual(state["status"], "failed")
            self.assertEqual(state["error"]["code"], "SERVER_WORKER_FAILURE")

    def test_server_gpu_boundary_rejects_wrong_identity_fields(self) -> None:
        cases = (
            {
                "physical_gpu_index": 4,
                "gpu_query_context": "native-host",
                "gpu_uuid": "GPU-84d7eeb0-65e0-a5b1-d7db-d09ef59fe03a",
            },
            {
                "physical_gpu_index": 5,
                "gpu_query_context": "gpu5-container",
                "gpu_uuid": "GPU-84d7eeb0-65e0-a5b1-d7db-d09ef59fe03a",
            },
            {
                "physical_gpu_index": 5,
                "gpu_query_context": "native-host",
                "gpu_uuid": "GPU-invalid",
            },
        )
        for fields in cases:
            with self.subTest(fields=fields):
                with self.assertRaises(demo_server.GPUExecutionBoundaryError):
                    demo_server.GPUExecutionBoundary(**fields)

    def test_server_gpu_boundary_rejects_conflicting_parent_environment(self) -> None:
        boundary = demo_server.GPUExecutionBoundary(
            physical_gpu_index=5,
            gpu_query_context="native-host",
            gpu_uuid="GPU-84d7eeb0-65e0-a5b1-d7db-d09ef59fe03a",
        )
        with tempfile.TemporaryDirectory() as temporary:
            model = Path(temporary) / "model"
            model.mkdir()
            manifest = Path(temporary) / "manifest.toml"
            manifest.write_text("[files]", encoding="utf-8")
            base = {
                "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
                "CUDA_VISIBLE_DEVICES": ("GPU-84d7eeb0-65e0-a5b1-d7db-d09ef59fe03a"),
                "NVIDIA_VISIBLE_DEVICES": ("GPU-84d7eeb0-65e0-a5b1-d7db-d09ef59fe03a"),
            }
            for name, invalid in (
                ("CUDA_DEVICE_ORDER", "FASTEST_FIRST"),
                ("CUDA_VISIBLE_DEVICES", "5"),
                ("NVIDIA_VISIBLE_DEVICES", "GPU-invalid"),
                ("PYTHONPATH", "/tmp/injected"),
                ("LD_PRELOAD", "/tmp/injected.so"),
            ):
                environment = {**base, name: invalid}
                with (
                    self.subTest(name=name),
                    patch.dict(os.environ, environment, clear=True),
                    self.assertRaises(demo_server.GPUExecutionBoundaryError),
                ):
                    production_launch_factory(
                        ROOT, model, manifest, gpu_boundary=boundary
                    )

    def test_server_gpu_identity_preflight_uses_exact_bounded_isolated_child(
        self,
    ) -> None:
        boundary = demo_server.GPUExecutionBoundary(
            physical_gpu_index=5,
            gpu_query_context="native-host",
            gpu_uuid=SERVER_GPU_UUID,
        )
        completed = SimpleNamespace(returncode=0)
        with (
            patch.dict(os.environ, {"HOME": "/home/test"}, clear=True),
            patch("cogni_demo.server.subprocess.run", return_value=completed) as run,
        ):
            demo_server._preflight_server_gpu_identity(boundary)
        command = run.call_args.args[0]
        self.assertEqual(command[0], os.fspath(Path(sys.executable).absolute()))
        self.assertEqual(command[1:3], ("-I", "-B"))
        self.assertEqual(Path(command[3]).name, "probe_native_gpu5_identity.py")
        self.assertEqual(
            command[4:],
            (
                "--physical-gpu-index",
                "5",
                "--gpu-query-context",
                "native-host",
                "--gpu-uuid",
                SERVER_GPU_UUID,
            ),
        )
        kwargs = run.call_args.kwargs
        self.assertIs(kwargs["stdin"], subprocess.DEVNULL)
        self.assertIs(kwargs["stdout"], subprocess.DEVNULL)
        self.assertIs(kwargs["stderr"], subprocess.DEVNULL)
        self.assertFalse(kwargs["shell"])
        self.assertTrue(kwargs["close_fds"])
        self.assertFalse(kwargs["check"])
        self.assertEqual(
            kwargs["timeout"], demo_server.SERVER_GPU_IDENTITY_PROBE_TIMEOUT_SECONDS
        )
        self.assertEqual(
            kwargs["env"],
            {
                **demo_server._SERVER_IDENTITY_PROBE_FIXED_ENVIRONMENT,
                "HOME": "/home/test",
            },
        )
        import inspect

        self.assertNotIn(
            "import torch",
            inspect.getsource(demo_server._preflight_server_gpu_identity),
        )

    def test_server_gpu_identity_preflight_rejects_child_failure_and_timeout(
        self,
    ) -> None:
        boundary = demo_server.GPUExecutionBoundary(
            physical_gpu_index=5,
            gpu_query_context="native-host",
            gpu_uuid=SERVER_GPU_UUID,
        )
        for result in (
            SimpleNamespace(returncode=9),
            subprocess.TimeoutExpired("identity-probe", 1.0),
            OSError("spawn failed"),
        ):
            configured = (
                {"side_effect": result}
                if isinstance(result, BaseException)
                else {"return_value": result}
            )
            with (
                self.subTest(result=type(result).__name__),
                patch.dict(os.environ, {"HOME": "/home/test"}, clear=True),
                patch("cogni_demo.server.subprocess.run", **configured),
                self.assertRaises(demo_server.GPUExecutionBoundaryError),
            ):
                demo_server._preflight_server_gpu_identity(boundary)

    def test_gpu_identity_child_accepts_exact_scope_and_rejects_mismatch(self) -> None:
        expected = SimpleNamespace(
            physical_index=5,
            uuid=SERVER_GPU_UUID,
            query_context="native-host",
            logical_device_count=1,
            logical_device_index=0,
        )
        argv = (
            "--physical-gpu-index",
            "5",
            "--gpu-query-context",
            "native-host",
            "--gpu-uuid",
            SERVER_GPU_UUID,
        )
        environment = {**identity_probe._FIXED_ENVIRONMENT, "HOME": "/home/test"}
        flags = SimpleNamespace(
            isolated=1,
            dont_write_bytecode=1,
            no_user_site=1,
            safe_path=True,
        )
        validator = Mock(return_value=expected)
        with (
            patch.dict(identity_probe.os.environ, environment, clear=True),
            patch.object(identity_probe.sys, "flags", flags),
        ):
            self.assertEqual(
                identity_probe.main(
                    argv,
                    torch_module=object(),
                    identity_validator=validator,
                ),
                0,
            )
        validator.assert_called_once()

        mismatches = {
            "physical_index": 4,
            "uuid": "GPU-invalid",
            "query_context": "gpu5-container",
            "logical_device_count": 2,
            "logical_device_index": 1,
        }
        for field, value in mismatches.items():
            identity = SimpleNamespace(**{**vars(expected), field: value})
            with (
                self.subTest(field=field),
                patch.dict(identity_probe.os.environ, environment, clear=True),
                patch.object(identity_probe.sys, "flags", flags),
                self.assertRaisesRegex(RuntimeError, "invalid logical scope"),
            ):
                identity_probe.main(
                    argv,
                    torch_module=object(),
                    identity_validator=Mock(return_value=identity),
                )

    def test_server_python_isolation_requires_every_startup_flag(self) -> None:
        exact = {
            "isolated": 1,
            "dont_write_bytecode": 1,
            "no_user_site": 1,
            "safe_path": True,
        }
        with patch(
            "cogni_demo.server.sys.flags",
            SimpleNamespace(**exact),
        ):
            demo_server._require_isolated_server_python()

        for field, value in {
            "isolated": 0,
            "dont_write_bytecode": 0,
            "no_user_site": 0,
            "safe_path": False,
        }.items():
            flags = SimpleNamespace(**{**exact, field: value})
            with (
                self.subTest(field=field),
                patch("cogni_demo.server.sys.flags", flags),
                self.assertRaises(demo_server.GPUExecutionBoundaryError),
            ):
                demo_server._require_isolated_server_python()

    def test_initial_snapshot_has_no_stale_measured_evidence(self) -> None:
        state = manager_for("success").snapshot()
        self.assertEqual(
            set(state),
            {
                "status",
                "stage",
                "seq",
                "progress",
                "events",
                "metrics",
                "error",
                "active_job",
            },
        )
        self.assertEqual(state["status"], "ready")
        self.assertEqual(state["metrics"]["evidence_kind"], "unverified")
        for field in (
            "measured_at",
            "source",
            "peak_allocated_vram_gib",
            "peak_reserved_vram_gib",
            "peak_vram_gib",
            "requested_depth",
            "reached_depth",
            "transition_residual",
            "transition_converged",
            "verified_files",
            "device",
        ):
            with self.subTest(field=field):
                self.assertIsNone(state["metrics"][field])

    def test_success_requires_ordered_phases_unique_typed_terminal_and_exit_zero(
        self,
    ) -> None:
        manager = manager_for("success")
        manager.start()
        state = wait_for_terminal(manager)
        self.assertEqual(state["status"], "succeeded")
        self.assertEqual(state["progress"], 100)
        self.assertEqual(state["metrics"]["evidence_kind"], "live_runtime_validation")
        self.assertEqual(state["metrics"]["reached_depth"], 100)
        self.assertEqual(state["metrics"]["peak_allocated_vram_gib"], 14.5)
        self.assertEqual(state["metrics"]["peak_reserved_vram_gib"], 14.856)
        self.assertEqual(
            state["metrics"]["peak_vram_gib"],
            max(
                state["metrics"]["peak_allocated_vram_gib"],
                state["metrics"]["peak_reserved_vram_gib"],
            ),
        )
        self.assertIsNone(state["metrics"]["tests"])
        self.assertEqual(state["metrics"]["target"], "RTX 4090 24GB")
        self.assertIsNone(state["active_job"])

    def test_nonzero_malformed_duplicate_and_unsafe_results_fail(self) -> None:
        for mode in ("fail", "malformed", "duplicate_result", "over_vram"):
            with self.subTest(mode=mode):
                manager = manager_for(mode)
                manager.start()
                state = wait_for_terminal(manager)
                self.assertEqual(state["status"], "failed")
                self.assertIsNotNone(state["error"])
                self.assertEqual(state["metrics"]["evidence_kind"], "unverified")
                for field in (
                    "peak_allocated_vram_gib",
                    "peak_reserved_vram_gib",
                    "peak_vram_gib",
                ):
                    self.assertIsNone(state["metrics"][field])
                self.assertIsNone(state["metrics"]["reached_depth"])

    def test_high_residual_worker_fails_closed_and_clears_evidence(self) -> None:
        manager = manager_for("high_residual")
        manager.start()
        state = wait_for_terminal(manager)

        self.assertEqual(state["status"], "failed")
        self.assertIsNotNone(state["error"])
        self.assertEqual(state["metrics"]["evidence_kind"], "unverified")
        for field in (
            "peak_allocated_vram_gib",
            "peak_reserved_vram_gib",
            "peak_vram_gib",
            "reached_depth",
            "transition_residual",
            "transition_converged",
        ):
            with self.subTest(field=field):
                self.assertIsNone(state["metrics"][field])

    def test_stderr_flood_is_bounded_and_does_not_deadlock_success(self) -> None:
        for mode in ("stderr_flood", "stdout_flood"):
            with self.subTest(mode=mode):
                manager = manager_for(mode)
                manager.start()
                state = wait_for_terminal(manager)
                self.assertEqual(state["status"], "succeeded")
                self.assertLessEqual(len(manager._diagnostics), 200)
                self.assertTrue(
                    any(item["truncated"] == "true" for item in manager._diagnostics)
                )

    def test_duplicate_run_is_rejected_and_cancel_reaps_worker(self) -> None:
        manager = manager_for("hang")
        manager.start()
        with self.assertRaises(JobAlreadyRunningError):
            manager.start()
        manager.cancel()
        state = wait_for_terminal(manager)
        self.assertEqual(state["status"], "cancelled")
        self.assertIsNone(state["active_job"])
        manager.shutdown()

    def test_gpu_lease_precedes_popen_and_release_precedes_terminal(self) -> None:
        events: list[str] = []
        authority = _RecordingLeaseManager(events)
        manager = JobManager(
            launch_factory_for("success"),
            max_runtime_seconds=10.0,
            gpu_lease_manager=authority,
        )
        authority.job_manager = manager
        real_popen = subprocess.Popen

        def recording_popen(*args, **kwargs):
            events.append("popen")
            return real_popen(*args, **kwargs)

        with patch("cogni_demo.server.subprocess.Popen", side_effect=recording_popen):
            manager.start()
            state = wait_for_terminal(manager)

        self.assertEqual(state["status"], "succeeded")
        self.assertLess(events.index("acquire"), events.index("popen"))
        self.assertLess(events.index("popen"), events.index("release"))
        self.assertIn(authority.status_at_release, {"starting", "running"})
        self.assertIsNotNone(authority.process_code_at_release)
        self.assertIsNone(authority.active)
        self.assertIsNone(manager._process)
        self.assertIsNone(manager._gpu_lease)

    def test_pre_spawn_health_probe_keeps_exact_validation_fence(self) -> None:
        authority = _PreSpawnPausingLeaseManager()
        manager = JobManager(
            launch_factory_for("success"),
            max_runtime_seconds=10.0,
            gpu_lease_manager=authority,
        )
        try:
            manager.start()
            self.assertTrue(authority.acquired.wait(timeout=2.0))
            lease = authority.active
            self.assertIsNotNone(lease)
            self.assertIsNone(authority.reap())
            with self.assertRaises(GPULeaseBusyError):
                authority.acquire(
                    "contender",
                    "inference",
                    authority.max_vram_bytes,
                    deadline=monotonic() + 10.0,
                )
            self.assertEqual(authority.active, lease)
        finally:
            authority.continue_spawn.set()
        self.assertEqual(wait_for_terminal(manager)["status"], "succeeded")
        self.assertIsNone(authority.active)

    def test_shutdown_reaps_thread_process_and_validation_lease(self) -> None:
        authority = GPULeaseManager()
        manager = JobManager(
            launch_factory_for("hang"),
            max_runtime_seconds=10.0,
            gpu_lease_manager=authority,
        )
        manager.start()
        deadline = monotonic() + 2.0
        while manager._process is None and monotonic() < deadline:
            sleep(0.01)
        self.assertIsNotNone(manager._process)
        self.assertIsNotNone(authority.active)

        manager.shutdown(timeout=5.0)

        thread = manager._worker_thread
        self.assertTrue(thread is None or not thread.is_alive())
        self.assertIsNone(manager._process)
        self.assertIsNone(manager._gpu_lease)
        self.assertIsNone(authority.active)
        self.assertEqual(manager.snapshot()["status"], "cancelled")

    def test_unreapable_process_surfaces_failure_and_preserves_fence(self) -> None:
        authority = GPULeaseManager()
        process = _UnkillableProcess()
        lease = authority.acquire(
            "validation-test",
            "validation",
            authority.max_vram_bytes,
            deadline=monotonic() + 10.0,
            owner_alive=lambda: True,
        )
        manager = JobManager(
            launch_factory_for("success"),
            gpu_lease_manager=authority,
        )
        manager._status = "cancelling"
        manager._active_job = "job-test"
        manager._process = process
        manager._gpu_lease = lease

        with self.assertRaises(WorkerTerminationError):
            manager.shutdown(timeout=0.1)

        self.assertIs(manager._process, process)
        self.assertEqual(manager._gpu_lease, lease)
        self.assertEqual(authority.active, lease)
        self.assertEqual(manager.snapshot()["active_job"], "job-test")
        self.assertGreaterEqual(process.signals + process.terminations, 1)
        self.assertEqual(process.kills, 1)
        authority.release(lease)


class TestDemoHTTPControlPlane(unittest.TestCase):
    def setUp(self) -> None:
        self.assets_context = tempfile.TemporaryDirectory()
        assets = Path(self.assets_context.name)
        (assets / "index.html").write_text("<main>Cogni</main>", encoding="utf-8")
        (assets / "app.css").write_text("body{}", encoding="utf-8")
        (assets / "app.js").write_text("void 0", encoding="utf-8")
        (assets / "favicon.svg").write_text("<svg/>", encoding="utf-8")
        self.manager = manager_for("success")
        self.server = DemoHTTPServer(
            self.manager,
            assets,
            port=0,
            token="t" * 32,
            watchdog_timeout=None,
        )
        self.thread = Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.cookie = self._bootstrap()

    def tearDown(self) -> None:
        self.manager.shutdown()
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)
        self.assets_context.cleanup()

    def _connection(self) -> HTTPConnection:
        return HTTPConnection("127.0.0.1", self.server.server_port, timeout=5)

    def _bootstrap(self) -> str:
        connection = self._connection()
        connection.request("GET", "/?token=" + self.server.token)
        response = connection.getresponse()
        response.read()
        self.assertEqual(response.status, 303)
        cookie = response.getheader("Set-Cookie").split(";", 1)[0]
        connection.close()
        return cookie

    def _post(self, path: str, body: dict) -> tuple[int, dict, dict]:
        connection = self._connection()
        encoded = json.dumps(body).encode("utf-8")
        connection.request(
            "POST",
            path,
            body=encoded,
            headers={
                "Cookie": self.cookie,
                "Origin": self.server.origin,
                "Content-Type": "application/json",
            },
        )
        response = connection.getresponse()
        payload = json.loads(response.read())
        headers = dict(response.getheaders())
        status = response.status
        connection.close()
        return status, payload, headers

    def test_loopback_static_state_security_headers_and_exact_routes(self) -> None:
        self.assertEqual(self.server.server_address[0], "127.0.0.1")
        connection = self._connection()
        connection.request("GET", "/api/state", headers={"Cookie": self.cookie})
        response = connection.getresponse()
        state = json.loads(response.read())
        self.assertEqual(response.status, 200)
        self.assertEqual(state["status"], "ready")
        self.assertIn(
            "default-src 'self'", response.getheader("Content-Security-Policy")
        )
        self.assertIsNone(response.getheader("Access-Control-Allow-Origin"))
        connection.close()

        connection = self._connection()
        connection.request("GET", "/api/ping")
        response = connection.getresponse()
        marker = json.loads(response.read())
        self.assertEqual(response.status, 200)
        self.assertEqual(marker, {"service": "cogniboard", "protocol": 1})
        self.assertNotIn("token", marker)
        connection.close()

        connection = self._connection()
        connection.request("GET", "/assets/app.js", headers={"Cookie": self.cookie})
        response = connection.getresponse()
        self.assertEqual(response.read(), b"void 0")
        self.assertEqual(response.status, 200)
        connection.close()

        connection = self._connection()
        connection.request(
            "GET", "/assets/favicon.svg", headers={"Cookie": self.cookie}
        )
        response = connection.getresponse()
        self.assertEqual(response.read(), b"<svg/>")
        self.assertEqual(response.status, 200)
        self.assertEqual(response.getheader("Content-Type"), "image/svg+xml")
        connection.close()

        connection = self._connection()
        connection.request("GET", "/secret", headers={"Cookie": self.cookie})
        response = connection.getresponse()
        response.read()
        self.assertEqual(response.status, 404)
        connection.close()

    def test_run_endpoint_and_origin_cookie_guards(self) -> None:
        status, payload, _headers = self._post("/api/run", {})
        self.assertEqual(status, 202)
        self.assertIn("job_id", payload)
        state = wait_for_terminal(self.manager)
        self.assertEqual(state["status"], "succeeded")

        connection = self._connection()
        body = b"{}"
        connection.request(
            "POST",
            "/api/run",
            body=body,
            headers={"Content-Type": "application/json", "Cookie": self.cookie},
        )
        response = connection.getresponse()
        response.read()
        self.assertEqual(response.status, 403)
        connection.close()

    def test_cancel_and_shutdown_endpoints_control_worker_lifetime(self) -> None:
        self.manager._launch_factory = launch_factory_for("hang")
        status, _payload, _headers = self._post("/api/run", {})
        self.assertEqual(status, 202)
        status, _payload, _headers = self._post("/api/cancel", {})
        self.assertEqual(status, 202)
        self.assertEqual(wait_for_terminal(self.manager)["status"], "cancelled")

        status, payload, _headers = self._post("/api/shutdown", {})
        self.assertEqual(status, 202)
        self.assertEqual(payload["status"], "shutting_down")
        self.thread.join(timeout=2)
        self.assertFalse(self.thread.is_alive())

    def test_invalid_host_and_oversized_body_are_rejected(self) -> None:
        connection = self._connection()
        connection.putrequest("GET", "/api/state", skip_host=True)
        connection.putheader("Host", "evil.example")
        connection.putheader("Cookie", self.cookie)
        connection.endheaders()
        response = connection.getresponse()
        response.read()
        self.assertEqual(response.status, 400)
        connection.close()

        connection = self._connection()
        connection.putrequest("POST", "/api/run")
        connection.putheader("Host", f"127.0.0.1:{self.server.server_port}")
        connection.putheader("Cookie", self.cookie)
        connection.putheader("Origin", self.server.origin)
        connection.putheader("Content-Type", "application/json")
        connection.putheader("Content-Length", "9000")
        connection.endheaders()
        response = connection.getresponse()
        response.read()
        self.assertEqual(response.status, 413)
        connection.close()


class TestNativeGPU5ServerLifecycle(unittest.TestCase):
    def test_early_authority_failure_never_imports_product_server(self) -> None:
        import scripts.gpu5_boundary_guard as gpu_guard

        execution_snapshot = native_execution_snapshot()
        native_arguments = server_bootstrap._NativeAuthorityArguments(
            source_commit=SERVER_SOURCE_COMMIT,
            physical_index=5,
            query_context="native-host",
            gpu_uuid=SERVER_GPU_UUID,
            snapshot_stage="sealed",
            source_snapshot_root=execution_snapshot.source.root_path,
            source_snapshot_nonce=execution_snapshot.source.launch_nonce,
            workspace_root=execution_snapshot.workspace_root,
            source_content_digest=execution_snapshot.source.content_digest,
            source_identity_digest=execution_snapshot.source.identity_digest,
            source_file_count=execution_snapshot.source.file_count,
            source_root_device=execution_snapshot.source.root_device,
            source_root_inode=execution_snapshot.source.root_inode,
            model_snapshot_root=execution_snapshot.model.root_path,
            model_manifest_sha256=execution_snapshot.model.manifest_sha256,
            model_content_digest=execution_snapshot.model.content_digest,
            model_identity_digest=execution_snapshot.model.identity_digest,
            model_file_count=execution_snapshot.model.file_count,
            model_root_device=execution_snapshot.model.root_device,
            model_root_inode=execution_snapshot.model.root_inode,
            model_total_bytes=execution_snapshot.model.total_bytes,
        )
        original_import = __import__
        for message in ("GPU5 project lease is already held", "GPU5 is not idle"):
            with self.subTest(message=message):
                failure = gpu_guard.GPU5BoundaryError(message)
                product_imports: list[str] = []

                @contextmanager
                def rejected_authority(*_args, **_kwargs):
                    raise failure
                    yield

                def watched_import(name, *args, **kwargs):
                    if name == "cogni_demo.server":
                        product_imports.append(name)
                    return original_import(name, *args, **kwargs)

                with (
                    patch.object(
                        server_bootstrap,
                        "_admitted_profile",
                        return_value="server-gpu5-native",
                    ),
                    patch.object(
                        server_bootstrap,
                        "_native_authority_arguments",
                        return_value=native_arguments,
                    ),
                    patch.object(
                        gpu_guard,
                        "validate_trusted_import_directory",
                    ),
                    patch.object(server_bootstrap, "_validate_sealed_sys_path"),
                    patch.object(
                        gpu_guard,
                        "native_gpu5_server_authority",
                        side_effect=rejected_authority,
                    ),
                    patch("builtins.__import__", side_effect=watched_import),
                    self.assertRaises(gpu_guard.GPU5BoundaryError) as captured,
                ):
                    server_bootstrap._run(tuple(server_gpu5_argv()))
                self.assertIs(captured.exception, failure)
                self.assertEqual(product_imports, [])

    @staticmethod
    def _authority(events: list[str]):
        import scripts.gpu5_boundary_guard as gpu_guard

        class LeaseProbe:
            def mark_launch_attempted(self) -> None:
                events.append("mark_launch_attempted")

            def mark_safe_to_release(self) -> None:
                events.append("mark_safe_to_release")

        checkout = gpu_guard.WorkingCheckoutIdentity(
            source_commit=SERVER_SOURCE_COMMIT,
            content_digest="1" * 64,
            identity_digest="2" * 64,
            file_count=7,
        )
        execution_snapshot = native_execution_snapshot()
        authority = gpu_guard.NativeGPU5ServerAuthority(
            expected_source_commit=SERVER_SOURCE_COMMIT,
            physical_gpu_index=5,
            gpu_query_context="native-host",
            gpu_uuid=SERVER_GPU_UUID,
            preflight=SimpleNamespace(),
            checkout=checkout,
            execution_snapshot=execution_snapshot,
            _lease=LeaseProbe(),
        )
        return authority, execution_snapshot

    @staticmethod
    def _boundary():
        return demo_server.GPUExecutionBoundary(
            physical_gpu_index=5,
            gpu_query_context="native-host",
            gpu_uuid=SERVER_GPU_UUID,
        )

    def test_logical_cuda_probe_occurs_only_after_lease_poison(self) -> None:
        events: list[str] = []
        authority, execution_snapshot = self._authority(events)
        with (
            patch(
                "scripts.gpu5_boundary_guard.verify_native_execution_snapshot",
                side_effect=(execution_snapshot, execution_snapshot),
                create=True,
            ),
            patch(
                "cogni_demo.server._preflight_server_gpu_idle",
                side_effect=lambda _boundary: events.append("idle_recheck"),
            ),
            patch(
                "cogni_demo.server._preflight_server_gpu_identity",
                side_effect=lambda _boundary: events.append("logical_cuda_probe"),
            ),
            patch(
                "cogni_demo.server._postflight_server_gpu_absence",
                side_effect=lambda _boundary: events.append("gpu5_postflight"),
            ),
        ):
            with demo_server._native_gpu5_server_lifecycle(
                self._boundary(),
                SERVER_SOURCE_COMMIT,
                authority,
                **native_snapshot_handoff(execution_snapshot),
            ):
                events.append("product_lifetime")
        self.assertEqual(
            events,
            [
                "idle_recheck",
                "mark_launch_attempted",
                "logical_cuda_probe",
                "idle_recheck",
                "product_lifetime",
                "gpu5_postflight",
                "mark_safe_to_release",
            ],
        )

    def test_idle_recheck_failure_prevents_poison_probe_and_product(self) -> None:
        events: list[str] = []
        authority, execution_snapshot = self._authority(events)
        idle_failure = demo_server.GPUExecutionBoundaryError("GPU5 busy")
        with (
            patch(
                "scripts.gpu5_boundary_guard.verify_native_execution_snapshot",
                return_value=execution_snapshot,
                create=True,
            ),
            patch(
                "cogni_demo.server._preflight_server_gpu_idle",
                side_effect=idle_failure,
            ),
            patch(
                "cogni_demo.server._preflight_server_gpu_identity",
            ) as logical_probe,
            self.assertRaises(demo_server.GPUExecutionBoundaryError) as captured,
        ):
            with demo_server._native_gpu5_server_lifecycle(
                self._boundary(),
                SERVER_SOURCE_COMMIT,
                authority,
                **native_snapshot_handoff(execution_snapshot),
            ):
                self.fail("busy GPU5 unexpectedly entered product lifetime")
        self.assertIs(captured.exception, idle_failure)
        self.assertEqual(events, [])
        logical_probe.assert_not_called()

    def test_fatal_logical_probe_and_failed_postflight_retain_poison(self) -> None:
        events: list[str] = []
        authority, execution_snapshot = self._authority(events)
        logical_failure = KeyboardInterrupt("logical probe failed")
        postflight_failure = demo_server.GPUExecutionBoundaryError(
            "postflight uncertain"
        )
        with (
            patch(
                "scripts.gpu5_boundary_guard.verify_native_execution_snapshot",
                side_effect=(execution_snapshot, execution_snapshot),
                create=True,
            ),
            patch("cogni_demo.server._preflight_server_gpu_idle"),
            patch(
                "cogni_demo.server._preflight_server_gpu_identity",
                side_effect=logical_failure,
            ),
            patch(
                "cogni_demo.server._postflight_server_gpu_absence",
                side_effect=postflight_failure,
            ),
            self.assertRaises(BaseExceptionGroup) as captured,
        ):
            with demo_server._native_gpu5_server_lifecycle(
                self._boundary(),
                SERVER_SOURCE_COMMIT,
                authority,
                **native_snapshot_handoff(execution_snapshot),
            ):
                self.fail("fatal logical probe unexpectedly entered product lifetime")
        self.assertEqual(
            captured.exception.exceptions,
            (logical_failure, postflight_failure),
        )
        self.assertEqual(events, ["mark_launch_attempted"])

    def test_every_primary_failure_retains_poison_after_clean_postflight(self) -> None:
        primary_failures = (
            WorkerTerminationError("component cleanup failed"),
            RuntimeError("application failed"),
            KeyboardInterrupt("fatal application interruption"),
        )
        for primary in primary_failures:
            events: list[str] = []
            authority, execution_snapshot = self._authority(events)
            with (
                self.subTest(primary=type(primary).__name__),
                patch(
                    "scripts.gpu5_boundary_guard.verify_native_execution_snapshot",
                    side_effect=(execution_snapshot, execution_snapshot),
                    create=True,
                ),
                patch("cogni_demo.server._preflight_server_gpu_idle"),
                patch("cogni_demo.server._preflight_server_gpu_identity"),
                patch("cogni_demo.server._postflight_server_gpu_absence"),
                self.assertRaises(type(primary)) as captured,
            ):
                with demo_server._native_gpu5_server_lifecycle(
                    self._boundary(),
                    SERVER_SOURCE_COMMIT,
                    authority,
                    **native_snapshot_handoff(execution_snapshot),
                ):
                    raise primary
            self.assertIs(captured.exception, primary)
            self.assertEqual(events, ["mark_launch_attempted"])

    def test_native_lifecycle_rejects_each_snapshot_argument_mismatch_before_probe(
        self,
    ) -> None:
        cases = {
            "source_snapshot_root": str(ROOT / "mutable-source"),
            "source_snapshot_nonce": "9" * 32,
            "workspace_root": str(SERVER_WORKSPACE_ROOT.parent),
            "source_content_digest": "9" * 64,
            "source_identity_digest": "8" * 64,
            "source_file_count": SERVER_SNAPSHOT_FILE_COUNT + 1,
            "source_root_device": int(ROOT.stat().st_dev) + 1,
            "source_root_inode": int(ROOT.stat().st_ino) + 1,
            "model_snapshot_root": str(ROOT / "mutable-model"),
            "model_manifest_path": str(ROOT / "mutable-manifest.toml"),
            "model_manifest_sha256": "7" * 64,
            "model_content_digest": "6" * 64,
            "model_identity_digest": "5" * 64,
            "model_file_count": SERVER_SNAPSHOT_FILE_COUNT + 1,
            "model_root_device": int(ROOT.stat().st_dev) + 1,
            "model_root_inode": int(ROOT.stat().st_ino) + 1,
            "model_total_bytes": SERVER_MODEL_TOTAL_BYTES + 1,
        }
        for field, mismatch in cases.items():
            events: list[str] = []
            authority, execution_snapshot = self._authority(events)
            values = native_snapshot_handoff(execution_snapshot)
            values[field] = mismatch
            import scripts.gpu5_boundary_guard as gpu_guard

            with (
                self.subTest(field=field),
                patch.object(
                    gpu_guard,
                    "verify_native_execution_snapshot",
                    create=True,
                ) as verify_snapshot,
                patch("cogni_demo.server._preflight_server_gpu_idle") as idle,
                patch("cogni_demo.server._preflight_server_gpu_identity") as identity,
                self.assertRaises(demo_server.GPUExecutionBoundaryError),
            ):
                with demo_server._native_gpu5_server_lifecycle(
                    self._boundary(),
                    SERVER_SOURCE_COMMIT,
                    authority,
                    **values,
                ):
                    self.fail("mismatched snapshot unexpectedly reached residency")
            verify_snapshot.assert_not_called()
            idle.assert_not_called()
            identity.assert_not_called()
            self.assertEqual(events, [])

    def test_product_parser_rejects_prepare_stage_before_lifecycle_or_bind(
        self,
    ) -> None:
        arguments = server_gpu5_argv()
        stage = arguments.index("--native-snapshot-stage") + 1
        arguments[stage] = "prepare"
        with (
            patch("cogni_demo.server._native_gpu5_server_lifecycle") as lifecycle,
            patch("cogni_demo.server.production_launch_factory") as worker_factory,
            patch("cogni_demo.server.DemoHTTPServer") as http_server,
            self.assertRaises(SystemExit),
        ):
            main(arguments, native_gpu5_authority=native_test_authority())
        lifecycle.assert_not_called()
        worker_factory.assert_not_called()
        http_server.assert_not_called()

    def test_each_native_handoff_field_is_required_before_idle_worker_or_bind(
        self,
    ) -> None:
        required_options = (
            "--native-workspace-root",
            "--native-source-content-digest",
            "--native-source-identity-digest",
            "--native-source-file-count",
            "--native-source-root-device",
            "--native-source-root-inode",
            "--native-model-manifest-sha256",
            "--native-model-content-digest",
            "--native-model-identity-digest",
            "--native-model-file-count",
            "--native-model-root-device",
            "--native-model-root-inode",
            "--native-model-total-bytes",
        )
        for option in required_options:
            arguments = server_gpu5_argv()
            index = arguments.index(option)
            del arguments[index : index + 2]
            with (
                self.subTest(option=option),
                patch("cogni_demo.server._native_gpu5_server_lifecycle") as lifecycle,
                patch("cogni_demo.server._preflight_server_gpu_idle") as idle,
                patch("cogni_demo.server.production_launch_factory") as worker_factory,
                patch("cogni_demo.server.DemoHTTPServer") as http_server,
                self.assertRaises(SystemExit),
            ):
                main(arguments, native_gpu5_authority=native_test_authority())
            lifecycle.assert_not_called()
            idle.assert_not_called()
            worker_factory.assert_not_called()
            http_server.assert_not_called()

    def test_each_native_handoff_mismatch_fails_before_idle_worker_or_bind(
        self,
    ) -> None:
        import scripts.gpu5_boundary_guard as gpu_guard

        root_metadata = ROOT.stat()
        mismatches = {
            "--native-workspace-root": str(SERVER_WORKSPACE_ROOT.parent),
            "--native-source-content-digest": "9" * 64,
            "--native-source-identity-digest": "9" * 64,
            "--native-source-file-count": "2",
            "--native-source-root-device": str(int(root_metadata.st_dev) + 1),
            "--native-source-root-inode": str(int(root_metadata.st_ino) + 1),
            "--native-model-manifest-sha256": "9" * 64,
            "--native-model-content-digest": "9" * 64,
            "--native-model-identity-digest": "9" * 64,
            "--native-model-file-count": "2",
            "--native-model-root-device": str(int(root_metadata.st_dev) + 1),
            "--native-model-root-inode": str(int(root_metadata.st_ino) + 1),
            "--native-model-total-bytes": "2",
        }
        environment = {
            "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
            "CUDA_VISIBLE_DEVICES": SERVER_GPU_UUID,
            "HOME": str(ROOT),
            "NVIDIA_VISIBLE_DEVICES": SERVER_GPU_UUID,
        }
        for option, mismatch in mismatches.items():
            arguments = server_gpu5_argv()
            arguments[arguments.index(option) + 1] = mismatch
            with (
                self.subTest(option=option),
                patch.dict(os.environ, environment, clear=True),
                patch("cogni_demo.server._require_isolated_server_python"),
                patch(
                    "cogni_demo.server.default_session_path",
                    return_value=Path(tempfile.gettempdir())
                    / "cogniboard-session.json",
                ),
                patch("cogni_demo.server.find_live_session", return_value=None),
                patch.object(
                    gpu_guard,
                    "verify_native_execution_snapshot",
                    create=True,
                ) as verify_snapshot,
                patch("cogni_demo.server._preflight_server_gpu_idle") as idle,
                patch("cogni_demo.server._preflight_server_gpu_identity") as identity,
                patch("cogni_demo.server.production_launch_factory") as worker_factory,
                patch("cogni_demo.server._build_product_controls") as controls,
                patch("cogni_demo.server.DemoHTTPServer") as http_server,
                self.assertRaises(demo_server.GPUExecutionBoundaryError),
            ):
                main(arguments, native_gpu5_authority=native_test_authority())
            verify_snapshot.assert_not_called()
            idle.assert_not_called()
            identity.assert_not_called()
            worker_factory.assert_not_called()
            controls.assert_not_called()
            http_server.assert_not_called()

    def test_raw_source_model_manifest_or_assets_never_reaches_worker_or_http_bind(
        self,
    ) -> None:
        environment = {
            "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
            "CUDA_VISIBLE_DEVICES": SERVER_GPU_UUID,
            "HOME": str(ROOT),
            "NVIDIA_VISIBLE_DEVICES": SERVER_GPU_UUID,
        }
        with tempfile.TemporaryDirectory() as temporary:
            temporary_root = Path(temporary)
            sealed_model = temporary_root / "sealed-model"
            raw_model = temporary_root / "raw-model"
            raw_source = temporary_root / "raw-source"
            raw_assets = temporary_root / "raw-assets"
            sealed_model.mkdir()
            raw_model.mkdir()
            raw_source.mkdir()
            raw_assets.mkdir()
            raw_manifest = temporary_root / "raw-manifest.toml"
            raw_manifest.write_text("[files]\n", encoding="utf-8")
            sealed_manifest = ROOT / "config" / "gemma4-e4b-it.manifest.toml"

            valid_snapshot = native_execution_snapshot(
                source_root=ROOT,
                model_root=sealed_model,
                manifest_path=sealed_manifest,
            )
            cases = (
                (
                    "source",
                    native_execution_snapshot(
                        source_root=raw_source,
                        model_root=sealed_model,
                        manifest_path=sealed_manifest,
                    ),
                    server_gpu5_argv(
                        model=sealed_model,
                        manifest=sealed_manifest,
                        source_snapshot_root=raw_source,
                        model_snapshot_root=sealed_model,
                    ),
                ),
                (
                    "model",
                    valid_snapshot,
                    server_gpu5_argv(
                        model=raw_model,
                        manifest=sealed_manifest,
                        source_snapshot_root=ROOT,
                        model_snapshot_root=sealed_model,
                    ),
                ),
                (
                    "manifest",
                    valid_snapshot,
                    server_gpu5_argv(
                        model=sealed_model,
                        manifest=raw_manifest,
                        source_snapshot_root=ROOT,
                        model_snapshot_root=sealed_model,
                    ),
                ),
                (
                    "assets",
                    valid_snapshot,
                    server_gpu5_argv(
                        model=sealed_model,
                        manifest=sealed_manifest,
                        assets=raw_assets,
                        source_snapshot_root=ROOT,
                        model_snapshot_root=sealed_model,
                    ),
                ),
            )
            for label, execution_snapshot, arguments in cases:
                authority = native_test_authority(execution_snapshot=execution_snapshot)
                with (
                    self.subTest(label=label),
                    patch.dict(os.environ, environment, clear=True),
                    patch("cogni_demo.server._require_isolated_server_python"),
                    patch(
                        "cogni_demo.server.default_session_path",
                        return_value=temporary_root / "session.json",
                    ),
                    patch("cogni_demo.server.find_live_session", return_value=None),
                    patch(
                        "cogni_demo.server.production_launch_factory"
                    ) as worker_factory,
                    patch(
                        "cogni_demo.server._build_product_controls"
                    ) as product_controls,
                    patch("cogni_demo.server.DemoHTTPServer") as http_server,
                    self.assertRaises(demo_server.GPUExecutionBoundaryError),
                ):
                    main(
                        arguments,
                        native_gpu5_authority=authority,
                        _native_lifecycle_token=(
                            demo_server._NATIVE_GPU5_LIFECYCLE_TOKEN
                        ),
                    )
                worker_factory.assert_not_called()
                product_controls.assert_not_called()
                http_server.assert_not_called()


class TestDemoApplicationLifecycle(unittest.TestCase):
    @patch("cogni_demo.server.production_launch_factory")
    @patch("cogni_demo.server.find_live_session")
    def test_second_main_reuses_existing_server_without_building_worker(
        self, find_session, launch_factory
    ) -> None:
        find_session.return_value = SessionMetadata(
            os.getpid(), 8765, "e" * 32, "2026-07-11T00:00:00Z"
        )
        with (
            patch(
                "cogni_demo.server._is_windows_desktop_platform",
                return_value=True,
            ),
            patch("cogni_demo.server.open_graphical_app") as open_app,
            patch("builtins.print") as output,
        ):
            self.assertEqual(main(["--no-browser"]), 0)
            open_app.assert_not_called()
            self.assertEqual(main([]), 0)
            open_app.assert_called_once_with(find_session.return_value.bootstrap_url)
            self.assertNotIn("e" * 32, " ".join(map(str, output.call_args_list)))
        launch_factory.assert_not_called()

    def test_component_shutdown_continues_after_first_failure(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            assets = Path(temporary)
            for name in ("index.html", "app.css", "app.js"):
                (assets / name).write_text(name, encoding="utf-8")
            failure = RuntimeError("validation cleanup failed")
            manager = Mock()
            manager.shutdown.side_effect = failure
            evolution_manager = Mock()
            agent_manager = Mock()
            server = DemoHTTPServer(
                manager,
                assets,
                agent_manager=agent_manager,
                evolution_manager=evolution_manager,
                port=0,
                watchdog_timeout=None,
            )
            try:
                with (
                    patch(
                        "cogni_demo.server.ThreadingHTTPServer.server_close",
                        autospec=True,
                    ) as close_transport,
                    self.assertRaises(RuntimeError) as caught,
                ):
                    server.server_close()
                self.assertIs(caught.exception, failure)
                manager.shutdown.assert_called_once_with()
                evolution_manager.shutdown.assert_called_once_with()
                agent_manager.shutdown.assert_called_once_with()
                close_transport.assert_called_once_with(server)
            finally:
                super(DemoHTTPServer, server).server_close()

    def test_concurrent_shutdown_callers_observe_same_ordered_failure(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            assets = Path(temporary)
            for name in ("index.html", "app.css", "app.js"):
                (assets / name).write_text(name, encoding="utf-8")
            entered = Event()
            release = Event()
            manager_failure = RuntimeError("validation cleanup failed")
            evolution_failure = RuntimeError("evolution cleanup failed")
            agent_failure = RuntimeError("agent cleanup failed")
            manager = Mock()

            def blocked_manager_shutdown() -> None:
                entered.set()
                self.assertTrue(release.wait(timeout=2.0))
                raise manager_failure

            manager.shutdown.side_effect = blocked_manager_shutdown
            evolution_manager = Mock()
            evolution_manager.shutdown.side_effect = evolution_failure
            agent_manager = Mock()
            agent_manager.shutdown.side_effect = agent_failure
            server = DemoHTTPServer(
                manager,
                assets,
                agent_manager=agent_manager,
                evolution_manager=evolution_manager,
                port=0,
                watchdog_timeout=None,
            )
            failures: list[BaseException] = []

            def stop_components() -> None:
                try:
                    server.shutdown_components()
                except BaseException as error:
                    failures.append(error)

            owner = Thread(target=stop_components)
            waiter = Thread(target=stop_components)
            try:
                owner.start()
                self.assertTrue(entered.wait(timeout=1.0))
                waiter.start()
                waiter.join(timeout=0.05)
                self.assertTrue(waiter.is_alive())
                release.set()
                owner.join(timeout=2.0)
                waiter.join(timeout=2.0)
                self.assertFalse(owner.is_alive())
                self.assertFalse(waiter.is_alive())
                self.assertEqual(len(failures), 2)
                self.assertIs(failures[0], failures[1])
                self.assertIsInstance(failures[0], BaseExceptionGroup)
                self.assertEqual(
                    failures[0].exceptions,
                    (manager_failure, evolution_failure, agent_failure),
                )
                manager.shutdown.assert_called_once_with()
                evolution_manager.shutdown.assert_called_once_with()
                agent_manager.shutdown.assert_called_once_with()
                with self.assertRaises(BaseExceptionGroup) as repeated:
                    server.shutdown_components()
                self.assertIs(repeated.exception, failures[0])
            finally:
                release.set()
                owner.join(timeout=2.0)
                waiter.join(timeout=2.0)
                super(DemoHTTPServer, server).server_close()

    def test_server_close_waits_for_in_progress_component_shutdown(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            assets = Path(temporary)
            for name in ("index.html", "app.css", "app.js"):
                (assets / name).write_text(name, encoding="utf-8")
            entered = Event()
            release = Event()
            manager = Mock()

            def blocked_manager_shutdown() -> None:
                entered.set()
                self.assertTrue(release.wait(timeout=2.0))

            manager.shutdown.side_effect = blocked_manager_shutdown
            evolution_manager = Mock()
            agent_manager = Mock()
            server = DemoHTTPServer(
                manager,
                assets,
                agent_manager=agent_manager,
                evolution_manager=evolution_manager,
                port=0,
                watchdog_timeout=None,
            )
            failures: list[BaseException] = []

            def capture(callback) -> None:
                try:
                    callback()
                except BaseException as error:
                    failures.append(error)

            owner = Thread(target=lambda: capture(server.shutdown_components))
            closer = Thread(target=lambda: capture(server.server_close))
            try:
                with patch(
                    "cogni_demo.server.ThreadingHTTPServer.server_close",
                    autospec=True,
                ) as close_transport:
                    owner.start()
                    self.assertTrue(entered.wait(timeout=1.0))
                    closer.start()
                    closer.join(timeout=0.05)
                    self.assertTrue(closer.is_alive())
                    close_transport.assert_not_called()
                    release.set()
                    owner.join(timeout=2.0)
                    closer.join(timeout=2.0)
                    self.assertFalse(owner.is_alive())
                    self.assertFalse(closer.is_alive())
                    self.assertEqual(failures, [])
                    close_transport.assert_called_once_with(server)
                manager.shutdown.assert_called_once_with()
                evolution_manager.shutdown.assert_called_once_with()
                agent_manager.shutdown.assert_called_once_with()
            finally:
                release.set()
                owner.join(timeout=2.0)
                closer.join(timeout=2.0)
                super(DemoHTTPServer, server).server_close()

    def test_component_shutdown_owner_reentry_does_not_deadlock(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            assets = Path(temporary)
            for name in ("index.html", "app.css", "app.js"):
                (assets / name).write_text(name, encoding="utf-8")
            manager = Mock()
            evolution_manager = Mock()
            agent_manager = Mock()
            server = DemoHTTPServer(
                manager,
                assets,
                agent_manager=agent_manager,
                evolution_manager=evolution_manager,
                port=0,
                watchdog_timeout=None,
            )
            manager.shutdown.side_effect = server.shutdown_components
            try:
                server.shutdown_components()
                manager.shutdown.assert_called_once_with()
                evolution_manager.shutdown.assert_called_once_with()
                agent_manager.shutdown.assert_called_once_with()
                server.shutdown_components()
            finally:
                super(DemoHTTPServer, server).server_close()

    def test_component_shutdown_waiter_timeout_is_bounded(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            assets = Path(temporary)
            for name in ("index.html", "app.css", "app.js"):
                (assets / name).write_text(name, encoding="utf-8")
            entered = Event()
            release = Event()
            manager = Mock()

            def blocked_manager_shutdown() -> None:
                entered.set()
                self.assertTrue(release.wait(timeout=2.0))

            manager.shutdown.side_effect = blocked_manager_shutdown
            server = DemoHTTPServer(
                manager,
                assets,
                port=0,
                watchdog_timeout=None,
            )
            owner_failures: list[BaseException] = []

            def owner_cleanup() -> None:
                try:
                    server.shutdown_components()
                except BaseException as error:
                    owner_failures.append(error)

            owner = Thread(target=owner_cleanup)
            try:
                owner.start()
                self.assertTrue(entered.wait(timeout=1.0))
                with self.assertRaisesRegex(
                    WorkerTerminationError,
                    "wait bound",
                ):
                    server.shutdown_components(wait_timeout=0.01)
                release.set()
                owner.join(timeout=2.0)
                self.assertFalse(owner.is_alive())
                self.assertEqual(owner_failures, [])
                manager.shutdown.assert_called_once_with()
            finally:
                release.set()
                owner.join(timeout=2.0)
                super(DemoHTTPServer, server).server_close()

    def test_component_shutdown_compute_fence_timeout_reuses_exact_error(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            assets = Path(temporary)
            for name in ("index.html", "app.css", "app.js"):
                (assets / name).write_text(name, encoding="utf-8")
            manager = Mock()
            server = DemoHTTPServer(
                manager,
                assets,
                port=0,
                watchdog_timeout=None,
            )
            entered = Event()
            release = Event()

            def hold_compute() -> None:
                with server._compute_lock:
                    entered.set()
                    self.assertTrue(release.wait(timeout=2.0))

            holder = Thread(target=hold_compute)
            try:
                holder.start()
                self.assertTrue(entered.wait(timeout=1.0))
                started = monotonic()
                with self.assertRaisesRegex(
                    WorkerTerminationError, "compute.*wait bound"
                ) as first:
                    server.shutdown_components(wait_timeout=0.01)
                self.assertLess(monotonic() - started, 0.5)
                self.assertTrue(server._shutdown_requested)
                manager.shutdown.assert_not_called()
                release.set()
                holder.join(timeout=2.0)
                with self.assertRaises(WorkerTerminationError) as repeated:
                    server.shutdown_components(wait_timeout=1.0)
                self.assertIs(repeated.exception, first.exception)
                manager.shutdown.assert_not_called()
            finally:
                release.set()
                holder.join(timeout=2.0)
                super(DemoHTTPServer, server).server_close()

    def test_component_first_then_transport_stops_real_server_exactly_once(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            assets = Path(temporary)
            for name in ("index.html", "app.css", "app.js"):
                (assets / name).write_text(name, encoding="utf-8")
            manager = Mock()
            server = DemoHTTPServer(
                manager,
                assets,
                port=0,
                watchdog_timeout=None,
            )
            serving = Thread(target=server.serve_forever, daemon=True)
            connection = HTTPConnection("127.0.0.1", server.server_port, timeout=1.0)
            try:
                serving.start()
                connection.request("GET", "/api/ping")
                response = connection.getresponse()
                response.read()
                self.assertEqual(response.status, 200)
                server.shutdown_components()
                self.assertIsNone(server._transport_shutdown_thread)
                with patch.object(
                    server, "shutdown", wraps=server.shutdown
                ) as shutdown:
                    server.request_shutdown()
                    transport = server._transport_shutdown_thread
                    self.assertIsNotNone(transport)
                    server.request_shutdown()
                    assert transport is not None
                    transport.join(timeout=2.0)
                    serving.join(timeout=2.0)
                    self.assertFalse(transport.is_alive())
                    self.assertFalse(serving.is_alive())
                    server.request_shutdown()
                    shutdown.assert_called_once_with()
                manager.shutdown.assert_called_once_with()
            finally:
                connection.close()
                if serving.is_alive():
                    Thread(target=server.shutdown, daemon=True).start()
                    serving.join(timeout=2.0)
                super(DemoHTTPServer, server).server_close()

    def test_transport_thread_start_failure_retries_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            assets = Path(temporary)
            for name in ("index.html", "app.css", "app.js"):
                (assets / name).write_text(name, encoding="utf-8")
            manager = Mock()
            server = DemoHTTPServer(
                manager,
                assets,
                port=0,
                watchdog_timeout=None,
            )
            real_thread = demo_server.Thread
            failed_thread = Mock()
            failed_thread.start.side_effect = RuntimeError("thread start failed")
            constructed = 0

            def thread_factory(*args, **kwargs):
                nonlocal constructed
                constructed += 1
                if constructed == 1:
                    return failed_thread
                return real_thread(*args, **kwargs)

            try:
                with (
                    patch("cogni_demo.server.Thread", side_effect=thread_factory),
                    patch.object(server, "shutdown") as shutdown,
                ):
                    with self.assertRaisesRegex(RuntimeError, "thread start failed"):
                        server.request_shutdown()
                    self.assertTrue(server._shutdown_requested)
                    self.assertIsNone(server._transport_shutdown_thread)
                    server.request_shutdown()
                    transport = server._transport_shutdown_thread
                    self.assertIsNotNone(transport)
                    assert transport is not None
                    transport.join(timeout=2.0)
                    self.assertFalse(transport.is_alive())
                    server.request_shutdown()
                    self.assertEqual(constructed, 2)
                    shutdown.assert_called_once_with()
                manager.shutdown.assert_called_once_with()
            finally:
                super(DemoHTTPServer, server).server_close()

    def test_shutdown_transition_serializes_after_each_admitted_start(self) -> None:
        for route in ("validation", "agent", "evolution"):
            with self.subTest(route=route), tempfile.TemporaryDirectory() as temporary:
                assets = Path(temporary)
                for name in ("index.html", "app.css", "app.js"):
                    (assets / name).write_text(name, encoding="utf-8")
                entered = Event()
                release = Event()
                cleanup_seen = Event()
                events: list[str] = []
                failures: list[BaseException] = []
                results: list[str] = []
                manager = Mock()
                manager.is_active = False
                agent_manager = Mock()
                agent_manager.is_active = False
                evolution_manager = Mock()
                evolution_manager.is_active = False
                server = DemoHTTPServer(
                    manager,
                    assets,
                    agent_manager=agent_manager,
                    evolution_manager=evolution_manager,
                    port=0,
                    watchdog_timeout=None,
                )

                def blocked_start(*_args, **_kwargs) -> str:
                    events.append("start_enter")
                    entered.set()
                    self.assertTrue(release.wait(timeout=2.0))
                    events.append("start_return")
                    return "job"

                if route == "validation":
                    manager.start.side_effect = blocked_start

                    def start() -> str:
                        return server.start_validation("prompt")

                elif route == "agent":
                    agent_manager.start_turn.side_effect = blocked_start

                    def start() -> str:
                        return server.start_agent_turn("hello", "conversation")

                else:
                    evolution_manager.start.side_effect = blocked_start
                    start = server.start_evolution

                def manager_shutdown() -> None:
                    events.append("shutdown_cleanup")
                    cleanup_seen.set()

                manager.shutdown.side_effect = manager_shutdown

                def capture_start() -> None:
                    try:
                        results.append(start())
                    except BaseException as error:
                        failures.append(error)

                def capture_shutdown() -> None:
                    try:
                        server.request_shutdown()
                    except BaseException as error:
                        failures.append(error)

                starter = Thread(target=capture_start)
                shutdown = Thread(target=capture_shutdown)
                try:
                    with patch.object(server, "shutdown"):
                        starter.start()
                        self.assertTrue(entered.wait(timeout=1.0))
                        shutdown.start()
                        shutdown.join(timeout=0.05)
                        self.assertFalse(shutdown.is_alive())
                        self.assertTrue(server._shutdown_requested)
                        self.assertFalse(cleanup_seen.is_set())
                        release.set()
                        starter.join(timeout=2.0)
                        shutdown.join(timeout=2.0)
                        self.assertTrue(cleanup_seen.wait(timeout=2.0))
                        transport = server._transport_shutdown_thread
                        self.assertIsNotNone(transport)
                        assert transport is not None
                        transport.join(timeout=2.0)
                        self.assertFalse(transport.is_alive())
                        server.shutdown_components()
                    self.assertFalse(starter.is_alive())
                    self.assertFalse(shutdown.is_alive())
                    self.assertEqual(failures, [])
                    self.assertEqual(results, ["job"])
                    self.assertLess(
                        events.index("start_return"),
                        events.index("shutdown_cleanup"),
                    )
                finally:
                    release.set()
                    starter.join(timeout=2.0)
                    shutdown.join(timeout=2.0)
                    super(DemoHTTPServer, server).server_close()

    def test_main_general_failure_groups_ordered_cleanup_faults(self) -> None:
        environment = {
            "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
            "CUDA_VISIBLE_DEVICES": ("GPU-84d7eeb0-65e0-a5b1-d7db-d09ef59fe03a"),
            "HOME": "/tmp/cogniboard-test-home",
            "LOCALAPPDATA": tempfile.gettempdir(),
            "NVIDIA_VISIBLE_DEVICES": ("GPU-84d7eeb0-65e0-a5b1-d7db-d09ef59fe03a"),
        }
        argv = server_gpu5_argv()
        original = RuntimeError("voice construction failed")
        manager_failure = RuntimeError("validation cleanup failed")
        evolution_failure = RuntimeError("evolution cleanup failed")
        manager = Mock()
        manager.shutdown.side_effect = manager_failure
        evolution_manager = Mock()
        evolution_manager.shutdown.side_effect = evolution_failure
        agent_manager = Mock()
        with (
            patch.dict(os.environ, environment, clear=True),
            patch("cogni_demo.server._require_isolated_server_python"),
            patch("cogni_demo.server.find_live_session", return_value=None),
            patch(
                "cogni_demo.server._native_gpu5_server_lifecycle",
                admitted_native_lifecycle,
            ),
            patch("cogni_core.cts_policy.load_default_bounded_cts_controller"),
            patch("cogni_demo.server.production_launch_factory", return_value=object()),
            patch("cogni_demo.server.JobManager", return_value=manager),
            patch(
                "cogni_demo.server._build_product_controls",
                return_value=(agent_manager, evolution_manager),
            ),
            patch(
                "cogni_demo.server._build_local_voice_service",
                side_effect=original,
            ),
            patch("cogni_demo.server.DemoHTTPServer") as server_factory,
            self.assertRaises(BaseExceptionGroup) as caught,
        ):
            main(argv, native_gpu5_authority=native_test_authority())
        self.assertEqual(
            caught.exception.exceptions,
            (original, manager_failure, evolution_failure),
        )
        manager.shutdown.assert_called_once_with()
        evolution_manager.shutdown.assert_called_once_with()
        agent_manager.shutdown.assert_called_once_with()
        server_factory.assert_not_called()

    def test_server_profile_refuses_any_existing_live_session(self) -> None:
        existing = SessionMetadata(os.getpid(), 8765, "e" * 32, "2026-07-11T00:00:00Z")
        environment = {
            "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
            "CUDA_VISIBLE_DEVICES": ("GPU-84d7eeb0-65e0-a5b1-d7db-d09ef59fe03a"),
            "HOME": "/tmp/cogniboard-test-home",
            "LOCALAPPDATA": tempfile.gettempdir(),
            "NVIDIA_VISIBLE_DEVICES": ("GPU-84d7eeb0-65e0-a5b1-d7db-d09ef59fe03a"),
        }
        argv = server_gpu5_argv()
        with (
            patch.dict(os.environ, environment, clear=True),
            patch("cogni_demo.server._require_isolated_server_python"),
            patch("cogni_demo.server.find_live_session", return_value=existing),
            patch("cogni_demo.server._preflight_server_gpu_identity") as preflight,
            patch(
                "cogni_core.cts_policy.load_default_bounded_cts_controller"
            ) as load_cts,
            patch("cogni_demo.server.open_graphical_app") as open_app,
            self.assertRaises(SystemExit),
        ):
            main(argv)
        preflight.assert_not_called()
        load_cts.assert_not_called()
        open_app.assert_not_called()

    def test_server_profile_refuses_bind_race_session_reuse(self) -> None:
        environment = {
            "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
            "CUDA_VISIBLE_DEVICES": ("GPU-84d7eeb0-65e0-a5b1-d7db-d09ef59fe03a"),
            "HOME": "/tmp/cogniboard-test-home",
            "LOCALAPPDATA": tempfile.gettempdir(),
            "NVIDIA_VISIBLE_DEVICES": ("GPU-84d7eeb0-65e0-a5b1-d7db-d09ef59fe03a"),
        }
        argv = server_gpu5_argv()
        manager = Mock()
        agent_manager = Mock()
        evolution_manager = Mock()
        with (
            patch.dict(os.environ, environment, clear=True),
            patch("cogni_demo.server._require_isolated_server_python"),
            patch("cogni_demo.server.find_live_session", return_value=None),
            patch(
                "cogni_demo.server._native_gpu5_server_lifecycle",
                admitted_native_lifecycle,
            ),
            patch("cogni_core.cts_policy.load_default_bounded_cts_controller"),
            patch("cogni_demo.server.production_launch_factory", return_value=object()),
            patch("cogni_demo.server.JobManager", return_value=manager),
            patch(
                "cogni_demo.server._build_product_controls",
                return_value=(agent_manager, evolution_manager),
            ),
            patch("cogni_demo.server._build_local_voice_service"),
            patch("cogni_demo.server.DemoHTTPServer", side_effect=OSError("bind")),
            patch("cogni_demo.server.read_session_metadata") as read_session,
            self.assertRaises(demo_server.GPUExecutionBoundaryError),
        ):
            main(argv, native_gpu5_authority=native_test_authority())
        read_session.assert_not_called()
        evolution_manager.shutdown.assert_called_once_with()
        agent_manager.shutdown.assert_called_once_with()
        manager.shutdown.assert_called_once_with()

    def test_server_profile_wires_exact_boundary_into_product_factory(self) -> None:
        lifecycle: list[str] = []

        def stop_after_boundary(*_args, **_kwargs):
            lifecycle.append("resident_model_controls")
            raise RuntimeError("stop after boundary wiring")

        @contextmanager
        def observe_native_lifecycle(*_args, **_kwargs):
            lifecycle.append("native_lifecycle")
            yield

        environment = {
            "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
            "CUDA_VISIBLE_DEVICES": ("GPU-84d7eeb0-65e0-a5b1-d7db-d09ef59fe03a"),
            "HOME": "/tmp",
            "LOCALAPPDATA": tempfile.gettempdir(),
            "NVIDIA_VISIBLE_DEVICES": ("GPU-84d7eeb0-65e0-a5b1-d7db-d09ef59fe03a"),
        }
        argv = server_gpu5_argv()
        with (
            patch.dict(os.environ, environment, clear=True),
            patch("cogni_demo.server.find_live_session", return_value=None),
            patch("cogni_demo.server._require_isolated_server_python"),
            patch(
                "cogni_demo.server._native_gpu5_server_lifecycle",
                observe_native_lifecycle,
            ),
            patch("cogni_core.cts_policy.load_default_bounded_cts_controller"),
            patch("cogni_demo.server.production_launch_factory") as launch_factory,
            patch("cogni_demo.server.JobManager") as job_manager,
            patch(
                "cogni_demo.server._build_product_controls",
                side_effect=stop_after_boundary,
            ),
        ):
            launch_factory.return_value = object()
            job_manager.return_value = object()
            with self.assertRaisesRegex(RuntimeError, "stop after boundary wiring"):
                main(argv, native_gpu5_authority=native_test_authority())
        boundary = launch_factory.call_args.kwargs["gpu_boundary"]
        self.assertEqual(lifecycle, ["native_lifecycle", "resident_model_controls"])
        self.assertEqual(boundary.physical_gpu_index, 5)
        self.assertEqual(boundary.gpu_query_context, "native-host")
        self.assertEqual(
            boundary.gpu_uuid,
            "GPU-84d7eeb0-65e0-a5b1-d7db-d09ef59fe03a",
        )

    def test_server_profile_rejects_missing_native_environment(self) -> None:
        with patch.dict(
            os.environ,
            {
                "CUDA_DEVICE_ORDER": "",
                "CUDA_VISIBLE_DEVICES": "",
                "NVIDIA_VISIBLE_DEVICES": "",
            },
            clear=True,
        ):
            with self.assertRaises(SystemExit):
                main(server_gpu5_argv())

    def test_linux_server_rejects_desktop_ui_only_profile(self) -> None:
        with (
            patch(
                "cogni_demo.server._is_windows_desktop_platform",
                return_value=False,
            ),
            self.assertRaises(SystemExit),
        ):
            main(["--no-browser", "--validation-profile", "desktop-ui-only"])

    def test_server_parser_rejects_abbreviated_validation_profile(self) -> None:
        with (
            patch(
                "cogni_demo.server._is_windows_desktop_platform",
                return_value=True,
            ),
            patch(
                "cogni_demo.server.find_live_session",
                side_effect=AssertionError(
                    "abbreviated option reached product startup"
                ),
            ),
            self.assertRaises(SystemExit),
        ):
            main(["--no-browser", "--validation-profi", "desktop-ui-only"])

    def test_session_metadata_is_bounded_atomic_and_reuses_live_server(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            assets = Path(temporary) / "assets"
            assets.mkdir()
            for name in ("index.html", "app.css", "app.js"):
                (assets / name).write_text(name, encoding="utf-8")
            manager = manager_for("success")
            server = DemoHTTPServer(
                manager, assets, port=0, token="s" * 32, watchdog_timeout=None
            )
            thread = Thread(target=server.serve_forever, daemon=True)
            thread.start()
            session_path = Path(temporary) / "CogniOS" / "cogniboard-session.json"
            metadata = SessionMetadata(
                os.getpid(), server.server_port, server.token, "2026-07-11T00:00:00Z"
            )
            written = write_session_metadata(metadata, session_path)
            self.assertEqual(written, session_path)
            self.assertEqual(read_session_metadata(session_path), metadata)
            self.assertTrue(ping_session(metadata))
            self.assertEqual(find_live_session(session_path), metadata)

            remove_session_metadata(session_path, expected=metadata)
            self.assertFalse(session_path.exists())
            server.request_shutdown()
            thread.join(timeout=2)
            server.server_close()

    def test_malformed_or_symlink_session_is_stale_and_never_followed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "session.json"
            path.write_text("not-json", encoding="utf-8")
            self.assertIsNone(find_live_session(path))
            self.assertFalse(path.exists())

            target = Path(temporary) / "target.json"
            target.write_text("protected", encoding="utf-8")
            try:
                path.symlink_to(target)
            except OSError:
                # Standard non-elevated Windows commonly disables symlink
                # creation; the malformed-file branch remains exercised.
                self.assertEqual(target.read_text(encoding="utf-8"), "protected")
                return
            self.assertIsNone(find_live_session(path))
            self.assertFalse(path.exists())
            self.assertEqual(target.read_text(encoding="utf-8"), "protected")

    @patch("cogni_demo.server.subprocess.Popen")
    @patch("cogni_demo.server._find_edge")
    def test_edge_app_mode_and_browser_fallback(self, find_edge, popen) -> None:
        find_edge.return_value = Path(r"C:\Program Files\Microsoft\Edge\msedge.exe")
        self.assertEqual(open_graphical_app("http://127.0.0.1:8765/x"), "edge")
        command = popen.call_args.args[0]
        self.assertEqual(
            command,
            [
                str(find_edge.return_value),
                "--app=http://127.0.0.1:8765/x",
                "--start-maximized",
                "--no-first-run",
            ],
        )
        self.assertFalse(popen.call_args.kwargs["shell"])

        find_edge.return_value = None
        with patch("cogni_demo.server.webbrowser.open") as browser:
            self.assertEqual(open_graphical_app("http://127.0.0.1:8765/y"), "browser")
            browser.assert_called_once_with("http://127.0.0.1:8765/y", new=1)

    def test_watchdog_cancels_worker_and_stops_when_polling_disappears(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            assets = Path(temporary)
            for name in ("index.html", "app.css", "app.js"):
                (assets / name).write_text(name, encoding="utf-8")
            manager = manager_for("hang")
            manager.start()
            server = DemoHTTPServer(
                manager,
                assets,
                port=0,
                token="w" * 32,
                watchdog_timeout=0.15,
            )
            thread = Thread(target=server.serve_forever, daemon=True)
            thread.start()
            thread.join(timeout=3)
            self.assertFalse(thread.is_alive())
            self.assertEqual(wait_for_terminal(manager)["status"], "cancelled")
            server.server_close()

    def test_authenticated_state_poll_keeps_watchdog_alive(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            assets = Path(temporary)
            for name in ("index.html", "app.css", "app.js"):
                (assets / name).write_text(name, encoding="utf-8")
            manager = manager_for("success")
            server = DemoHTTPServer(
                manager,
                assets,
                port=0,
                token="p" * 32,
                watchdog_timeout=0.2,
            )
            thread = Thread(target=server.serve_forever, daemon=True)
            thread.start()
            connection = HTTPConnection("127.0.0.1", server.server_port, timeout=2)
            connection.request("GET", "/?token=" + server.token)
            response = connection.getresponse()
            response.read()
            cookie = response.getheader("Set-Cookie").split(";", 1)[0]
            connection.close()
            deadline = monotonic() + 0.5
            while monotonic() < deadline:
                connection = HTTPConnection("127.0.0.1", server.server_port, timeout=2)
                connection.request("GET", "/api/state", headers={"Cookie": cookie})
                response = connection.getresponse()
                response.read()
                self.assertEqual(response.status, 200)
                connection.close()
                sleep(0.05)
            self.assertTrue(thread.is_alive())
            server.request_shutdown()
            thread.join(timeout=2)
            server.server_close()


if __name__ == "__main__":
    unittest.main()
