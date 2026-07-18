from __future__ import annotations

from contextlib import contextmanager
from hashlib import sha256
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import tomllib
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
import venv
from zipfile import ZipFile

from cogni_core.cts_policy import DEFAULT_CHECKPOINT_SHA256


ROOT = Path(__file__).resolve().parents[1]


class TestReleaseBundleIntegrity(unittest.TestCase):
    @staticmethod
    def _run_bootstrap_with_import_tripwire(
        arguments: tuple[str, ...],
    ) -> subprocess.CompletedProcess[str]:
        bootstrap = ROOT / "scripts" / "run_cogniboard_server.py"
        source = f"""
import runpy
import sys

def audit(event, args):
    if event == "import" and args and args[0] == "cogni_demo.server":
        raise RuntimeError("PRODUCT_IMPORT_BEFORE_ADMISSION")

sys.addaudithook(audit)
sys.argv = [{str(bootstrap)!r}, *{list(arguments)!r}]
runpy.run_path({str(bootstrap)!r}, run_name="__main__")
"""
        return subprocess.run(
            [sys.executable, "-I", "-B", "-c", source],
            cwd=tempfile.gettempdir(),
            env={
                "PATH": os.environ.get("PATH", ""),
                "PYTHONDONTWRITEBYTECODE": "1",
            },
            capture_output=True,
            check=False,
            text=True,
            timeout=30,
        )

    def test_isolated_server_bootstrap_anchors_the_repository(self) -> None:
        bootstrap = ROOT / "scripts" / "run_cogniboard_server.py"
        source = bootstrap.read_text(encoding="utf-8")
        self.assertLess(
            source.index("sys.path.insert(0, str(_PROJECT_ROOT))"),
            source.index("from cogni_demo.server import main"),
        )
        self.assertIn(
            "allow_abbrev=False",
            (ROOT / "cogni_demo/server.py").read_text(encoding="utf-8"),
        )

        for help_option in ("-h", "--help"):
            with self.subTest(help_option=help_option):
                completed = self._run_bootstrap_with_import_tripwire((help_option,))
                self.assertEqual(completed.returncode, 0, completed.stderr)
                self.assertIn("--validation-profile", completed.stdout)
                self.assertNotIn(
                    "PRODUCT_IMPORT_BEFORE_ADMISSION",
                    completed.stdout + completed.stderr,
                )

    def test_product_server_and_lazy_public_exports_keep_heavy_modules_unloaded(
        self,
    ) -> None:
        source = f"""
import sys

sys.path.insert(0, {str(ROOT)!r})
import cogni_demo.server

heavy_modules = (
    "torch",
    "cogni_agent.model_service",
    "cogni_flow.evolution",
)
assert all(name not in sys.modules for name in heavy_modules), tuple(
    name for name in heavy_modules if name in sys.modules
)

from cogni_agent import WorkspaceToolExecutor
from cogni_flow import RhythmController

assert WorkspaceToolExecutor.__name__ == "WorkspaceToolExecutor"
assert WorkspaceToolExecutor.__module__ == "cogni_agent.tools"
assert RhythmController.__name__ == "RhythmController"
assert RhythmController.__module__ == "cogni_flow.rhythm"
assert all(name not in sys.modules for name in heavy_modules), tuple(
    name for name in heavy_modules if name in sys.modules
)
print("EARLY_IMPORT_GRAPH_OK")
"""
        completed = subprocess.run(
            [sys.executable, "-I", "-B", "-c", source],
            cwd=tempfile.gettempdir(),
            env={
                "PATH": os.environ.get("PATH", ""),
                "PYTHONDONTWRITEBYTECODE": "1",
            },
            capture_output=True,
            check=False,
            text=True,
            timeout=30,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(completed.stdout.strip(), "EARLY_IMPORT_GRAPH_OK")

    @unittest.skipUnless(sys.platform.startswith("linux"), "Linux bootstrap contract")
    def test_linux_bootstrap_rejects_profiles_before_product_import(self) -> None:
        invalid_arguments = (
            (),
            ("--no-browser",),
            ("--validation-profile", "desktop-ui-only"),
            ("--validation-profile=",),
            ("--validation-profi", "server-gpu5-native"),
            ("--validation-profile", "server-gpu5-native"),
            ("--help", "--no-browser"),
        )
        for arguments in invalid_arguments:
            with self.subTest(arguments=arguments):
                completed = self._run_bootstrap_with_import_tripwire(arguments)
                combined = completed.stdout + completed.stderr
                self.assertNotEqual(completed.returncode, 0, combined)
                self.assertNotIn("PRODUCT_IMPORT_BEFORE_ADMISSION", combined)

    def test_windows_bootstrap_admits_desktop_only(self) -> None:
        from scripts import run_cogniboard_server as bootstrap

        self.assertEqual(
            bootstrap._admitted_profile(
                ("--validation-profile", "desktop-ui-only"),
                platform="win32",
            ),
            "desktop-ui-only",
        )
        for arguments in (
            (),
            ("--validation-profile=",),
            ("--validation-profi", "desktop-ui-only"),
            ("--validation-profile", "server-gpu5-native"),
        ):
            with self.subTest(arguments=arguments), self.assertRaises(RuntimeError):
                bootstrap._admitted_profile(arguments, platform="win32")

    def test_sealed_sys_path_requires_trusted_wrapper_and_no_workspace_overlap(
        self,
    ) -> None:
        from scripts import run_cogniboard_server as bootstrap

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            workspace = root / "workspace"
            workspace_child = workspace / "import-child"
            trusted = root / "trusted-sibling"
            workspace_child.mkdir(parents=True)
            trusted.mkdir()

            with (
                patch.object(bootstrap.sys, "path", [str(trusted)]),
                self.assertRaisesRegex(
                    RuntimeError,
                    "sealed import-path validator is unavailable",
                ),
            ):
                bootstrap._validate_sealed_sys_path(SimpleNamespace(), workspace)

            validator = Mock(side_effect=lambda path: Path(path).resolve(strict=True))
            boundary_guard = SimpleNamespace(
                validate_trusted_import_directory=validator
            )
            with patch.object(bootstrap.sys, "path", [str(trusted)]):
                bootstrap._validate_sealed_sys_path(boundary_guard, workspace)
            validator.assert_called_once_with(trusted)

            for imported_root in (workspace_child, root):
                validator.reset_mock()
                with (
                    self.subTest(imported_root=imported_root),
                    patch.object(bootstrap.sys, "path", [str(imported_root)]),
                    self.assertRaisesRegex(
                        RuntimeError,
                        "mutable workspace remained on the sealed sys.path",
                    ),
                ):
                    bootstrap._validate_sealed_sys_path(boundary_guard, workspace)
                validator.assert_not_called()

    def test_native_prepare_stage_snapshots_before_reexec_without_product_import(
        self,
    ) -> None:
        """Stage 0 must hand only sealed paths to a fresh isolated process."""

        from scripts import run_cogniboard_server as bootstrap

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            raw_model = root / "mutable-model"
            raw_manifest = ROOT / "config" / "gemma4-e4b-it.manifest.toml"
            source_snapshot = root / "sealed-source"
            model_snapshot = root / "sealed-model"
            workspace_snapshot = root / "sealed-workspace"
            snapshot_manifest = (
                source_snapshot / "config" / "gemma4-e4b-it.manifest.toml"
            )
            snapshot_bootstrap = (
                source_snapshot / "scripts" / "run_cogniboard_server.py"
            )
            snapshot_assets = source_snapshot / "cogni_demo" / "static"
            raw_model.mkdir()
            model_snapshot.mkdir()
            workspace_snapshot.mkdir()
            snapshot_manifest.parent.mkdir(parents=True)
            snapshot_bootstrap.parent.mkdir(parents=True)
            snapshot_assets.mkdir(parents=True)
            snapshot_manifest.write_text("[files]\n", encoding="utf-8")
            snapshot_bootstrap.write_text("# sealed bootstrap\n", encoding="utf-8")

            nonce = "1" * 32
            manifest_digest = "2" * 64
            source_content_digest = "3" * 64
            source_identity_digest = "4" * 64
            model_content_digest = "5" * 64
            model_identity_digest = "6" * 64
            source_metadata = source_snapshot.stat()
            model_metadata = model_snapshot.stat()
            capability = SimpleNamespace(
                source=SimpleNamespace(
                    root_path=str(source_snapshot),
                    launch_nonce=nonce,
                    content_digest=source_content_digest,
                    identity_digest=source_identity_digest,
                    file_count=17,
                    root_device=int(source_metadata.st_dev),
                    root_inode=int(source_metadata.st_ino),
                ),
                model=SimpleNamespace(
                    root_path=str(model_snapshot),
                    manifest_sha256=manifest_digest,
                    content_digest=model_content_digest,
                    identity_digest=model_identity_digest,
                    file_count=7,
                    root_device=int(model_metadata.st_dev),
                    root_inode=int(model_metadata.st_ino),
                    total_bytes=4096,
                ),
                manifest_path=str(snapshot_manifest),
                workspace_root=str(workspace_snapshot),
            )
            arguments = (
                "--no-browser",
                "--model",
                str(raw_model),
                "--manifest",
                str(raw_manifest),
                "--validation-profile",
                "server-gpu5-native",
                "--validation-physical-gpu-index",
                "5",
                "--validation-gpu-query-context",
                "native-host",
                "--validation-gpu-uuid",
                bootstrap._GPU5_UUID,
                "--expected-source-commit",
                "a" * 40,
                "--native-snapshot-stage",
                "prepare",
            )
            events: list[str] = []
            observed_exec: dict[str, object] = {}

            def prepare(*factory_arguments):
                events.append("prepare")
                self.assertEqual(
                    factory_arguments,
                    (
                        "a" * 40,
                        str(raw_model),
                        "config/gemma4-e4b-it.manifest.toml",
                    ),
                )
                return capability

            def execve(executable, argv, environment):
                events.append("exec")
                observed_exec.update(
                    executable=executable,
                    argv=tuple(argv),
                    environment=dict(environment),
                )
                raise RuntimeError("EXECVE_CAPTURED")

            product_imports: list[str] = []
            original_import = __import__

            def watched_import(name, *args, **kwargs):
                if name == "cogni_demo.server":
                    product_imports.append(name)
                return original_import(name, *args, **kwargs)

            with (
                patch(
                    "scripts.gpu5_boundary_guard.prepare_native_execution_snapshot",
                    side_effect=prepare,
                    create=True,
                ),
                patch("builtins.__import__", side_effect=watched_import),
                patch.object(bootstrap.os, "chdir") as chdir,
                patch.object(bootstrap.os, "execve", side_effect=execve),
                self.assertRaisesRegex(RuntimeError, "EXECVE_CAPTURED"),
            ):
                bootstrap._prepare_and_reexec_native_snapshots(
                    arguments,
                    "a" * 40,
                )

            self.assertEqual(events, ["prepare", "exec"])
            self.assertEqual(product_imports, [])
            chdir.assert_called_once_with(source_snapshot.resolve())
            exec_argv = observed_exec["argv"]
            self.assertIsInstance(exec_argv, tuple)
            assert isinstance(exec_argv, tuple)
            self.assertEqual(Path(exec_argv[3]), snapshot_bootstrap.resolve())
            self.assertEqual(
                exec_argv[exec_argv.index("--native-snapshot-stage") + 1],
                "sealed",
            )
            self.assertEqual(
                Path(exec_argv[exec_argv.index("--model") + 1]),
                model_snapshot.resolve(),
            )
            self.assertEqual(
                Path(exec_argv[exec_argv.index("--manifest") + 1]),
                snapshot_manifest.resolve(),
            )
            self.assertEqual(
                Path(exec_argv[exec_argv.index("--assets") + 1]),
                snapshot_assets.resolve(),
            )
            self.assertEqual(
                Path(exec_argv[exec_argv.index("--native-source-snapshot-root") + 1]),
                source_snapshot.resolve(),
            )
            self.assertEqual(
                Path(exec_argv[exec_argv.index("--native-workspace-root") + 1]),
                workspace_snapshot.resolve(),
            )
            self.assertEqual(
                exec_argv[exec_argv.index("--native-model-manifest-sha256") + 1],
                manifest_digest,
            )
            expected_handoff = {
                "--native-source-content-digest": source_content_digest,
                "--native-source-identity-digest": source_identity_digest,
                "--native-source-file-count": "17",
                "--native-source-root-device": str(source_metadata.st_dev),
                "--native-source-root-inode": str(source_metadata.st_ino),
                "--native-model-content-digest": model_content_digest,
                "--native-model-identity-digest": model_identity_digest,
                "--native-model-file-count": "7",
                "--native-model-root-device": str(model_metadata.st_dev),
                "--native-model-root-inode": str(model_metadata.st_ino),
                "--native-model-total-bytes": "4096",
            }
            for option, expected in expected_handoff.items():
                with self.subTest(option=option):
                    self.assertEqual(exec_argv[exec_argv.index(option) + 1], expected)
            self.assertNotIn(str(raw_model), exec_argv)
            self.assertNotIn(str(raw_manifest), exec_argv)
            environment = observed_exec["environment"]
            self.assertIsInstance(environment, dict)
            assert isinstance(environment, dict)
            self.assertEqual(
                Path(environment["COGNI_OS_MODEL_DIR"]),
                model_snapshot.resolve(),
            )

    def test_native_snapshot_stages_reject_injected_or_mismatched_capabilities(
        self,
    ) -> None:
        from scripts import run_cogniboard_server as bootstrap

        base = (
            "--model",
            str(ROOT),
            "--manifest",
            str(ROOT / "config" / "gemma4-e4b-it.manifest.toml"),
            "--validation-profile",
            "server-gpu5-native",
            "--validation-physical-gpu-index",
            "5",
            "--validation-gpu-query-context",
            "native-host",
            "--validation-gpu-uuid",
            bootstrap._GPU5_UUID,
            "--expected-source-commit",
            "a" * 40,
        )
        isolated_flags = SimpleNamespace(
            isolated=1,
            dont_write_bytecode=1,
            no_user_site=1,
            safe_path=True,
        )
        sealed_workspace = (
            Path(tempfile.gettempdir()).resolve() / "cogniboard-release-test-workspace"
        )
        sealed_workspace.mkdir(parents=True, exist_ok=True)
        sealed_handoff = (
            "--native-source-content-digest",
            "3" * 64,
            "--native-source-identity-digest",
            "4" * 64,
            "--native-source-file-count",
            "1",
            "--native-source-root-device",
            "0",
            "--native-source-root-inode",
            "1",
            "--native-model-content-digest",
            "5" * 64,
            "--native-model-identity-digest",
            "6" * 64,
            "--native-model-file-count",
            "1",
            "--native-model-root-device",
            "0",
            "--native-model-root-inode",
            "1",
            "--native-model-total-bytes",
            "1",
        )
        with (
            patch.object(bootstrap, "_validate_exact_server_environment"),
            patch.object(bootstrap.sys, "flags", isolated_flags),
        ):
            with self.assertRaisesRegex(
                RuntimeError,
                "prepare stage rejects caller-supplied snapshot",
            ):
                bootstrap._native_authority_arguments(
                    (
                        *base,
                        "--native-snapshot-stage",
                        "prepare",
                        "--native-source-snapshot-root",
                        str(ROOT),
                    )
                )

            with self.assertRaisesRegex(
                RuntimeError,
                "prepare stage rejects caller-supplied snapshot",
            ):
                bootstrap._native_authority_arguments(
                    (
                        *base,
                        "--assets",
                        str(ROOT / "cogni_demo" / "static"),
                        "--native-snapshot-stage",
                        "prepare",
                    )
                )

            with (
                patch(
                    "scripts.gpu5_boundary_guard.prepare_native_execution_snapshot"
                ) as snapshot_factory,
                patch.object(bootstrap.os, "execve") as execve,
                self.assertRaisesRegex(
                    RuntimeError,
                    "native snapshot preparation rejects caller assets",
                ),
            ):
                bootstrap._prepare_and_reexec_native_snapshots(
                    (
                        *base,
                        "--assets",
                        str(ROOT / "cogni_demo" / "static"),
                        "--native-snapshot-stage",
                        "prepare",
                    ),
                    "a" * 40,
                )
            snapshot_factory.assert_not_called()
            execve.assert_not_called()

            sealed_root_arguments = (
                *base,
                "--assets",
                str(ROOT / "cogni_demo" / "static"),
                "--native-snapshot-stage",
                "sealed",
                "--native-source-snapshot-root",
                str(ROOT),
                "--native-source-snapshot-nonce",
                "1" * 32,
                "--native-workspace-root",
                str(sealed_workspace),
                "--native-model-snapshot-root",
                str(ROOT),
                "--native-model-manifest-sha256",
                "2" * 64,
                *sealed_handoff,
            )
            required_handoff_options = (
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
            for option in required_handoff_options:
                index = sealed_root_arguments.index(option)
                missing = (
                    *sealed_root_arguments[:index],
                    *sealed_root_arguments[index + 2 :],
                )
                with (
                    self.subTest(missing=option),
                    self.assertRaisesRegex(
                        RuntimeError,
                        "sealed stage requires exact source/model snapshots",
                    ),
                ):
                    bootstrap._native_authority_arguments(missing)

            workspace_index = sealed_root_arguments.index("--native-workspace-root") + 1
            overlapping_workspace = list(sealed_root_arguments)
            overlapping_workspace[workspace_index] = str(ROOT)
            with self.assertRaisesRegex(
                RuntimeError,
                "paths escaped their snapshot capability",
            ):
                bootstrap._native_authority_arguments(tuple(overlapping_workspace))

            with tempfile.TemporaryDirectory() as temporary:
                escaped_source = Path(temporary) / "source"
                escaped_model = Path(temporary) / "model"
                escaped_assets = escaped_source / "cogni_demo" / "static"
                raw_assets = Path(temporary) / "raw-assets"
                escaped_assets.mkdir(parents=True)
                escaped_model.mkdir()
                raw_assets.mkdir()
                sealed_base = bootstrap._replace_exact_option(
                    base,
                    "--model",
                    str(escaped_model),
                )
                with self.assertRaisesRegex(
                    RuntimeError,
                    "paths escaped their snapshot capability",
                ):
                    bootstrap._native_authority_arguments(
                        (
                            *sealed_base,
                            "--native-snapshot-stage",
                            "sealed",
                            "--native-source-snapshot-root",
                            str(escaped_source),
                            "--native-source-snapshot-nonce",
                            "1" * 32,
                            "--native-workspace-root",
                            str(sealed_workspace),
                            "--native-model-snapshot-root",
                            str(escaped_model),
                            "--native-model-manifest-sha256",
                            "2" * 64,
                            *sealed_handoff,
                            "--assets",
                            str(escaped_assets),
                        )
                    )

                with self.assertRaisesRegex(
                    RuntimeError,
                    "paths escaped their snapshot capability",
                ):
                    bootstrap._native_authority_arguments(
                        (
                            *base,
                            "--native-snapshot-stage",
                            "sealed",
                            "--native-source-snapshot-root",
                            str(ROOT),
                            "--native-source-snapshot-nonce",
                            "1" * 32,
                            "--native-workspace-root",
                            str(sealed_workspace),
                            "--native-model-snapshot-root",
                            str(ROOT),
                            "--native-model-manifest-sha256",
                            "2" * 64,
                            *sealed_handoff,
                            "--assets",
                            str(raw_assets),
                        )
                    )

    def test_sealed_sys_path_validation_precedes_native_authority_context(self) -> None:
        from scripts import gpu5_boundary_guard as boundary_guard
        from scripts import run_cogniboard_server as bootstrap

        workspace = (
            Path(tempfile.gettempdir()).resolve() / "cogniboard-order-test-workspace"
        )
        workspace.mkdir(parents=True, exist_ok=True)
        admitted = bootstrap._NativeAuthorityArguments(
            source_commit="a" * 40,
            physical_index=5,
            query_context="native-host",
            gpu_uuid=bootstrap._GPU5_UUID,
            snapshot_stage="sealed",
            source_snapshot_root=str(ROOT),
            source_snapshot_nonce="1" * 32,
            workspace_root=str(workspace),
            source_content_digest="2" * 64,
            source_identity_digest="3" * 64,
            source_file_count=1,
            source_root_device=0,
            source_root_inode=1,
            model_snapshot_root=str(ROOT),
            model_manifest_sha256="4" * 64,
            model_content_digest="5" * 64,
            model_identity_digest="6" * 64,
            model_file_count=1,
            model_root_device=0,
            model_root_inode=1,
            model_total_bytes=1,
        )
        events: list[str] = []

        @contextmanager
        def rejected_authority(*_args, **_kwargs):
            events.append("authority_context")
            raise RuntimeError("AUTHORITY_CONTEXT_REACHED")
            yield

        with (
            patch.object(
                bootstrap,
                "_admitted_profile",
                return_value="server-gpu5-native",
            ),
            patch.object(
                bootstrap,
                "_native_authority_arguments",
                return_value=admitted,
            ),
            patch.object(
                boundary_guard,
                "validate_trusted_import_directory",
                side_effect=lambda _path: events.append("workspace_trust"),
            ),
            patch.object(
                bootstrap,
                "_validate_sealed_sys_path",
                side_effect=lambda _guard, _workspace: events.append("sys_path"),
            ),
            patch.object(
                boundary_guard,
                "native_gpu5_server_authority",
                rejected_authority,
            ),
            self.assertRaisesRegex(RuntimeError, "AUTHORITY_CONTEXT_REACHED"),
        ):
            bootstrap._run(
                (
                    "--manifest",
                    str(ROOT / "config" / "gemma4-e4b-it.manifest.toml"),
                )
            )
        self.assertEqual(events, ["workspace_trust", "sys_path", "authority_context"])

    def test_noncanonical_sealed_paths_fail_before_authority_context(self) -> None:
        from scripts import gpu5_boundary_guard as boundary_guard
        from scripts import run_cogniboard_server as bootstrap

        workspace = (
            Path(tempfile.gettempdir()).resolve()
            / "cogniboard-canonical-test-workspace"
        )
        workspace_child = workspace / "child"
        workspace_child.mkdir(parents=True, exist_ok=True)
        manifest = ROOT / "config" / "gemma4-e4b-it.manifest.toml"
        assets = ROOT / "cogni_demo" / "static"
        arguments = (
            "--model",
            str(ROOT),
            "--manifest",
            str(manifest),
            "--assets",
            str(assets),
            "--validation-profile",
            "server-gpu5-native",
            "--validation-physical-gpu-index",
            "5",
            "--validation-gpu-query-context",
            "native-host",
            "--validation-gpu-uuid",
            bootstrap._GPU5_UUID,
            "--expected-source-commit",
            "a" * 40,
            "--native-snapshot-stage",
            "sealed",
            "--native-source-snapshot-root",
            str(ROOT),
            "--native-source-snapshot-nonce",
            "1" * 32,
            "--native-workspace-root",
            str(workspace),
            "--native-source-content-digest",
            "2" * 64,
            "--native-source-identity-digest",
            "3" * 64,
            "--native-source-file-count",
            "1",
            "--native-source-root-device",
            "0",
            "--native-source-root-inode",
            "1",
            "--native-model-snapshot-root",
            str(ROOT),
            "--native-model-manifest-sha256",
            "4" * 64,
            "--native-model-content-digest",
            "5" * 64,
            "--native-model-identity-digest",
            "6" * 64,
            "--native-model-file-count",
            "1",
            "--native-model-root-device",
            "0",
            "--native-model-root-inode",
            "1",
            "--native-model-total-bytes",
            "1",
        )
        canonical_paths = {
            "--native-source-snapshot-root": ROOT,
            "--native-workspace-root": workspace,
            "--native-model-snapshot-root": ROOT,
            "--model": ROOT,
            "--manifest": manifest,
            "--assets": assets,
        }
        dotdot_aliases = {
            "--native-source-snapshot-root": ROOT / "cogni_demo" / "..",
            "--native-workspace-root": workspace_child / "..",
            "--native-model-snapshot-root": ROOT / "config" / "..",
            "--model": ROOT / "cogni_demo" / "..",
            "--manifest": manifest.parent / ".." / "config" / manifest.name,
            "--assets": assets / ".." / "static",
        }
        cases = tuple(
            (option, "dotdot", os.fspath(dotdot_aliases[option]))
            for option in canonical_paths
        ) + tuple(
            (
                option,
                "relative",
                os.path.relpath(path, start=Path.cwd()),
            )
            for option, path in canonical_paths.items()
        )
        isolated_flags = SimpleNamespace(
            isolated=1,
            dont_write_bytecode=1,
            no_user_site=1,
            safe_path=True,
        )
        for option, form, replacement in cases:
            mutated = list(arguments)
            mutated[mutated.index(option) + 1] = replacement
            with (
                self.subTest(option=option, form=form),
                patch.object(
                    bootstrap,
                    "_admitted_profile",
                    return_value="server-gpu5-native",
                ),
                patch.object(bootstrap, "_validate_exact_server_environment"),
                patch.object(bootstrap.sys, "flags", isolated_flags),
                patch.object(
                    bootstrap,
                    "_validate_sealed_sys_path",
                ) as validate_sys_path,
                patch.object(
                    boundary_guard,
                    "native_gpu5_server_authority",
                ) as authority_context,
                self.assertRaisesRegex(
                    RuntimeError,
                    "paths escaped their snapshot capability",
                ),
            ):
                bootstrap._run(tuple(mutated))
            validate_sys_path.assert_not_called()
            authority_context.assert_not_called()

    def test_gpu5_launcher_seals_source_model_and_bootstrap_before_exec(self) -> None:
        source = (ROOT / "Run-CogniOS-Server-GPU5.sh").read_text(encoding="utf-8")
        trust_gate = source.index(
            'validate_trusted_path_chain "${PROJECT_ROOT}" "project root"'
        )
        bootstrap_gate = source.index(
            'validate_trusted_regular_file "${SERVER_BOOTSTRAP}"'
        )
        manifest_gate = source.index('validate_trusted_regular_file "${MANIFEST_PATH}"')
        model_gate = source.index(
            'validate_trusted_path_chain "${model_path}" "COGNI_OS_MODEL_DIR"'
        )
        launch = source.index('"${PYTHON_INVOCATION}" -I -B "${SERVER_BOOTSTRAP}"')
        self.assertLess(trust_gate, bootstrap_gate)
        self.assertLess(bootstrap_gate, manifest_gate)
        self.assertLess(manifest_gate, model_gate)
        self.assertLess(model_gate, launch)
        self.assertIn("must have exactly one hard link", source)
        self.assertIn("must not be group/world writable", source)

    @staticmethod
    def _make_linux_launcher_fixture(
        temporary: str,
    ) -> tuple[Path, Path, Path, Path]:
        root = Path(temporary)
        repository = root / "repository"
        scripts = repository / "scripts"
        config = repository / "config"
        model = root / "model"
        home = root / "home"
        scripts.mkdir(parents=True)
        config.mkdir()
        model.mkdir()
        home.mkdir()

        launcher = repository / "Run-CogniOS-Server-GPU5.sh"
        shutil.copy2(ROOT / launcher.name, launcher)
        launcher.chmod(0o755)
        (config / "gemma4-e4b-it.manifest.toml").write_text(
            "[model]\nfamily = 'fixture'\n",
            encoding="utf-8",
        )
        (scripts / "run_cogniboard_server.py").write_text(
            """from __future__ import annotations
import json
import os
from pathlib import Path
import sys

payload = {
    "prefix": sys.prefix,
    "base_prefix": sys.base_prefix,
    "executable": sys.executable,
    "real_executable": str(Path(sys.executable).resolve()),
    "isolated": sys.flags.isolated,
    "dont_write_bytecode": sys.flags.dont_write_bytecode,
    "no_user_site": sys.flags.no_user_site,
    "safe_path": sys.flags.safe_path,
    "environment": dict(os.environ),
}
print("BOOTSTRAP_OBS=" + json.dumps(payload, sort_keys=True))
""",
            encoding="utf-8",
        )

        git_environment = {
            "PATH": "/usr/bin:/bin",
            "HOME": str(home),
            "LANG": "C",
            "LC_ALL": "C",
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_CONFIG_GLOBAL": "/dev/null",
        }
        commands = (
            ["/usr/bin/git", "init", "-q", str(repository)],
            ["/usr/bin/git", "-C", str(repository), "add", "--all"],
            [
                "/usr/bin/git",
                "-C",
                str(repository),
                "-c",
                "user.name=Cogni Test",
                "-c",
                "user.email=cogni@example.invalid",
                "commit",
                "-q",
                "-m",
                "fixture",
            ],
        )
        for command in commands:
            completed = subprocess.run(
                command,
                env=git_environment,
                capture_output=True,
                check=False,
                text=True,
                timeout=30,
            )
            if completed.returncode != 0:
                raise AssertionError(completed.stderr)
        return launcher, repository, model, home

    @staticmethod
    def _hostile_launcher_environment(
        *, python: Path, model: Path, home: Path
    ) -> dict[str, str]:
        return {
            "HOME": str(home),
            "COGNI_OS_PYTHON": str(python),
            "COGNI_OS_MODEL_DIR": str(model),
            "COGNI_OS_SANITIZED_LAUNCH": "cogni-server-gpu5-sanitized-v1",
            "PATH": "/definitely/untrusted",
            "LD_LIBRARY_PATH": "/definitely/untrusted",
            "BASH_ENV": "/dev/null",
            "ENV": "/dev/null",
            "BASH_FUNC_exec%%": "() { printf 'HOSTILE_EXEC_RAN\\n'; return 0; }",
            "BASH_FUNC_exit%%": "() { printf 'HOSTILE_EXIT_RAN\\n'; return 0; }",
            "BASH_FUNC_printf%%": "() { command printf 'HOSTILE_PRINTF_RAN\\n'; }",
            "GIT_DIR": "/definitely/missing",
            "GIT_WORK_TREE": "/definitely/missing",
            "GIT_INDEX_FILE": "/definitely/missing",
            "GIT_CONFIG_COUNT": "1",
            "PYTHONWARNINGS": "error",
            "HTTP_PROXY": "http://credential.invalid",
            "HTTPS_PROXY": "http://credential.invalid",
            "AWS_ACCESS_KEY_ID": "do-not-forward",
            "GITHUB_TOKEN": "do-not-forward",
            "CUSTOM_SECRET": "do-not-forward",
        }

    @unittest.skipUnless(
        sys.platform.startswith("linux")
        and Path("/usr/bin/bash").is_file()
        and Path("/usr/bin/git").is_file(),
        "Linux launcher contract",
    )
    def test_linux_launcher_preserves_venv_and_exact_child_environment(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            launcher, _repository, model, home = self._make_linux_launcher_fixture(
                temporary
            )
            virtual_environment = Path(temporary) / "venv"
            venv.EnvBuilder(with_pip=False, symlinks=True).create(virtual_environment)
            python = virtual_environment / "bin" / "python"
            completed = subprocess.run(
                [str(launcher)],
                env=self._hostile_launcher_environment(
                    python=python,
                    model=model,
                    home=home,
                ),
                capture_output=True,
                check=False,
                text=True,
                timeout=30,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertNotIn("HOSTILE_", completed.stdout + completed.stderr)
            observation_line = next(
                line
                for line in completed.stdout.splitlines()
                if line.startswith("BOOTSTRAP_OBS=")
            )
            observation = json.loads(observation_line.partition("=")[2])

            self.assertEqual(Path(observation["prefix"]), virtual_environment)
            self.assertNotEqual(observation["prefix"], observation["base_prefix"])
            self.assertEqual(Path(observation["executable"]), python)
            self.assertEqual(
                Path(observation["real_executable"]),
                python.resolve(),
            )
            self.assertEqual(observation["isolated"], 1)
            self.assertEqual(observation["dont_write_bytecode"], 1)
            self.assertEqual(observation["no_user_site"], 1)
            self.assertIs(observation["safe_path"], True)

            expected_environment = {
                "PATH": "/usr/local/bin:/usr/bin:/bin",
                "HOME": str(home),
                "LANG": "C.UTF-8",
                "LC_ALL": "C.UTF-8",
                "HF_HUB_OFFLINE": "1",
                "HF_HUB_DISABLE_TELEMETRY": "1",
                "TRANSFORMERS_OFFLINE": "1",
                "HF_DATASETS_OFFLINE": "1",
                "WANDB_MODE": "offline",
                "TOKENIZERS_PARALLELISM": "false",
                "PYTHONDONTWRITEBYTECODE": "1",
                "PYTHONHASHSEED": "0",
                "PYTHONNOUSERSITE": "1",
                "PYTHONSAFEPATH": "1",
                "COGNI_OS_MODEL_DIR": str(model.resolve()),
                "COGNI_OS_GPU_UUID": "GPU-84d7eeb0-65e0-a5b1-d7db-d09ef59fe03a",
                "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
                "CUDA_VISIBLE_DEVICES": "GPU-84d7eeb0-65e0-a5b1-d7db-d09ef59fe03a",
                "NVIDIA_VISIBLE_DEVICES": "GPU-84d7eeb0-65e0-a5b1-d7db-d09ef59fe03a",
            }
            self.assertEqual(observation["environment"], expected_environment)

    @unittest.skipUnless(
        sys.platform.startswith("linux")
        and Path("/usr/bin/true").is_file()
        and Path("/usr/bin/git").is_file(),
        "Linux launcher contract",
    )
    def test_linux_launcher_rejects_exit_zero_non_python(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            launcher, _repository, model, home = self._make_linux_launcher_fixture(
                temporary
            )
            completed = subprocess.run(
                [str(launcher)],
                env=self._hostile_launcher_environment(
                    python=Path("/usr/bin/true"),
                    model=model,
                    home=home,
                ),
                capture_output=True,
                check=False,
                text=True,
                timeout=30,
            )
            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("Python runtime sentinel mismatch", completed.stderr)
            self.assertNotIn("BOOTSTRAP_OBS=", completed.stdout)

    @unittest.skipUnless(
        sys.platform.startswith("linux")
        and Path("/usr/bin/bash").is_file()
        and Path("/usr/bin/git").is_file(),
        "Linux launcher contract",
    )
    def test_linux_launcher_rejects_shell_sentinel_wrapper(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            launcher, _repository, model, home = self._make_linux_launcher_fixture(
                temporary
            )
            wrapper = Path(temporary) / "fake-python"
            wrapper.write_text(
                """#!/usr/bin/bash
printf 'WRAPPER_RAN\\n'
printf 'cogni-python-runtime-v1|implementation=cpython|version=3.11+|isolated=1|dont_write_bytecode=1|no_user_site=1|safe_path=1|realpath=%s|proc_exe=%s\\n' "$0" "$0"
exit 0
""",
                encoding="utf-8",
            )
            wrapper.chmod(0o755)
            completed = subprocess.run(
                [str(launcher)],
                env=self._hostile_launcher_environment(
                    python=wrapper,
                    model=model,
                    home=home,
                ),
                capture_output=True,
                check=False,
                text=True,
                timeout=30,
            )
            self.assertNotEqual(completed.returncode, 0)
            self.assertIn(
                "must be an operator-trusted ELF executable", completed.stderr
            )
            self.assertNotIn("WRAPPER_RAN", completed.stdout + completed.stderr)
            self.assertNotIn("BOOTSTRAP_OBS=", completed.stdout)

    def test_release_publication_is_bound_to_reviewed_cpu_gpu5_evidence(self) -> None:
        script = (ROOT / "scripts" / "build_release_bundle.ps1").read_text(
            encoding="utf-8"
        )
        verifier = (ROOT / "scripts" / "validate_release_evidence.py").read_text(
            encoding="utf-8"
        )
        for contract in (
            "[switch]$PublishRelease",
            "[string]$ReleaseEvidenceSummaryPath",
            "[string]$ReleaseEvidenceSummarySha256",
            "[string]$CpuGateEvidencePath",
            "[string]$Gpu5GateEvidencePath",
            "[string]$ReleaseAttestationPath",
            "[string]$ReleaseAttestationSignaturePath",
            "[string]$VerifierPublicKeyPath",
            "[string]$RuntimeEvidencePath",
            "[string]$CompletionEvidencePath",
            "[string]$IdentityPreEvidencePath",
            "[string]$IdentityPostEvidencePath",
            "[string]$ConfigEvidencePath",
            "[string]$DeviceEvidencePath",
            "[string]$ModelInventoryPath",
            "[string]$ManualPdfSha256",
            "validate_release_evidence.py",
            "CPU_GATE_EVIDENCE.json",
            "GPU5_GATE_EVIDENCE.json",
            "GPU5_RUNTIME_EVIDENCE.json",
            "GPU5_COMPLETION_EVIDENCE.json",
            "GPU5_IDENTITY_PRE.json",
            "GPU5_IDENTITY_POST.json",
            "GPU5_CONFIG_EVIDENCE.json",
            "GPU5_DEVICE_EVIDENCE.json",
            "GPU5_MODEL_INVENTORY.json",
            "RELEASE_ATTESTATION.sig",
            "VERIFIER_PUBLIC_KEY.json",
            "RELEASE_VERIFIER_POLICY.json",
            "config\\gemma4-e4b-it.manifest.toml",
            "config\\release-verifier-policy.json",
            "config\\release-toolchain-policy.json",
            "$PinnedReleaseToolchainPolicySha256",
            "$powerShellExecutable",
            "protected-no-profile-isolated-runner",
            "EXTERNAL_BLOCKER: protected whole-build closure",
            "before tool discovery",
            "Executing release builder differs from the exact archived commit",
            "Archived toolchain policy differs from the source-pinned policy",
            "validate_master_acceptance_checklist.py",
            "render_outstanding_checklist.py",
            "Archived master acceptance checklist validation failed",
            "Archived outstanding checklist is not derived from the master ledger",
            "Publication requires a completely clean current HEAD worktree",
            "& $python -I -S -B -",
            "--git-executable $gitExecutable",
            "--git-executable-sha256 $gitExecutableSha",
            "New-VerifiedEvidenceSnapshot",
            "Copy-VerifiedSnapshotToBundle",
            "release_evidence_status=$releaseEvidenceStatus",
            "release_evidence_summary_sha256=$releaseEvidenceSummarySha",
        ):
            with self.subTest(contract=contract):
                self.assertIn(contract, script)
        for contract in (
            "cogni.release.gates.v2",
            "cogni.cpu.gates.v1",
            "cogni.gpu5.gates.v2",
            "cogni.release.attestation.v2",
            "cogni.release.verifier-policy.v1",
            "cogni.gpu5.runtime.v1",
            "cogni.gpu5.completion.v1",
            "cogni.gpu5.model-inventory.v1",
            "rsa-pkcs1v15-sha256",
            "duplicate JSON key",
            "type(item) is not bool",
            "GPU-84d7eeb0-65e0-a5b1-d7db-d09ef59fe03a",
            "CPU and GPU5 evidence source trees differ",
            "signed attestation scope does not match raw evidence",
            "source-approved immutable key",
            "compute_guard_source_tree_digest",
            "[path, sha256, git_mode, git_blob_oid]",
        ):
            with self.subTest(verifier_contract=contract):
                self.assertIn(contract, verifier)
        self.assertNotIn("RunModelSmoke", script)
        self.assertNotIn("validate_agent_runtime.py", script)
        self.assertNotIn("release_bundle=PASS", script)
        self.assertNotIn(
            "& $python (Join-Path $source 'scripts\\validate_release_evidence.py')",
            script,
        )

        policy = json.loads(
            (ROOT / "config" / "release-verifier-policy.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(
            policy,
            {
                "schema": "cogni.release.verifier-policy.v1",
                "status": "unconfigured",
                "verifier_id": None,
                "public_key_sha256": None,
            },
        )

        toolchain_policy_path = ROOT / "config" / "release-toolchain-policy.json"
        toolchain_policy_bytes = toolchain_policy_path.read_bytes()
        self.assertEqual(
            json.loads(toolchain_policy_bytes),
            {
                "schema": "cogni.release.toolchain-policy.v2",
                "status": "unconfigured",
                "runner_mode": None,
                "powershell_path": None,
                "powershell_sha256": None,
                "python_path": None,
                "python_sha256": None,
                "git_path": None,
                "git_sha256": None,
                "build_closure_manifest_path": None,
                "build_closure_manifest_sha256": None,
                "offline_wheelhouse_manifest_path": None,
                "offline_wheelhouse_manifest_sha256": None,
            },
        )
        self.assertIn(sha256(toolchain_policy_bytes).hexdigest(), script)

        manifest_path = ROOT / "config" / "gemma4-e4b-it.manifest.toml"
        with manifest_path.open("rb") as stream:
            manifest = tomllib.load(stream)
        self.assertEqual(
            manifest["model"],
            {
                "family": "gemma4",
                "variant": "E4B",
                "role": "instruction_tuned",
                "source": "google/gemma-4-E4B-it",
                "revision": "a4c2d58be94dda072b918d9db64ee85c8ed34e3f",
            },
        )
        self.assertEqual(
            set(manifest["files"]),
            {
                "chat_template.jinja",
                "config.json",
                "generation_config.json",
                "model.safetensors",
                "processor_config.json",
                "tokenizer.json",
                "tokenizer_config.json",
            },
        )

    def test_checkpoint_is_exported_as_an_opaque_git_artifact(self) -> None:
        attributes = (ROOT / ".gitattributes").read_text(encoding="utf-8")
        self.assertIn("/cogni_core/cts_policy_checkpoint.json -text", attributes)

    def test_guard_scoped_source_archive_preserves_cmd_bytes_and_omits_generated_artifacts(
        self,
    ) -> None:
        attributes = (ROOT / ".gitattributes").read_text(encoding="utf-8")
        ignores = (ROOT / ".gitignore").read_text(encoding="utf-8")
        self.assertIn("*.cmd -text", attributes)
        self.assertIn("/config/release-toolchain-policy.json -text", attributes)
        self.assertIn("/config/release-verifier-policy.json -text", attributes)
        self.assertNotIn("*.cmd text eol=crlf", attributes)
        for pattern in (
            "/CogniBoard.exe",
            "release/*.exe",
            "release/*.whl",
            "release/*.zip",
        ):
            with self.subTest(pattern=pattern):
                self.assertIn(pattern, ignores)

    def test_archive_with_autocrlf_disabled_preserves_checkpoint_bytes(self) -> None:
        if not (ROOT / ".git").is_dir():
            self.skipTest("exact git archive reproduction requires a Git checkout")
        with tempfile.TemporaryDirectory() as temporary:
            archive = Path(temporary) / "source.zip"
            subprocess.run(
                [
                    "git",
                    "-C",
                    str(ROOT),
                    "-c",
                    "core.autocrlf=false",
                    "archive",
                    "--format=zip",
                    "--prefix=source/",
                    f"--output={archive}",
                    "HEAD",
                ],
                check=True,
                capture_output=True,
            )
            with ZipFile(archive) as bundle:
                payload = bundle.read("source/cogni_core/cts_policy_checkpoint.json")

        self.assertEqual(sha256(payload).hexdigest(), DEFAULT_CHECKPOINT_SHA256)

    def test_release_script_pins_commit_and_publishes_atomically(self) -> None:
        script = (ROOT / "scripts" / "build_release_bundle.ps1").read_text(
            encoding="utf-8"
        )
        for contract in (
            "rev-parse --verify --end-of-options",
            "core.autocrlf=false",
            "Source archive changed the CTS policy checkpoint bytes",
            "Release output already exists; refusing to merge",
            ".cogni-release-staging-",
            "Move-Item -LiteralPath $publishStage -Destination $publishedOutput",
            "SOURCE_DATE_EPOCH",
            "commit_oid=$commitOid",
            "SBOM.cdx.json",
            "THIRD_PARTY_NOTICES.md",
            "unsigned-no-code-signing-certificate-provided",
            "artifact_build_status=PASS",
            "artifact_build=PASS",
        ):
            with self.subTest(contract=contract):
                self.assertIn(contract, script)

        release_scope = script.index(
            "$sourceTreeDigest = $releaseValidation.source_tree_digest"
        )
        acceptance_gate = script.index(
            "& $python -I -S -B $archivedAcceptanceValidator"
        )
        renderer_gate = script.index("& $python -I -S -B $archivedOutstandingRenderer")
        effective_gate = script.index("--require-complete")
        staging = script.index(
            "$publishStage = New-ReleasePublishStage $publishedOutput"
        )
        self.assertLess(release_scope, acceptance_gate)
        self.assertLess(acceptance_gate, renderer_gate)
        self.assertLess(renderer_gate, effective_gate)
        self.assertLess(effective_gate, staging)
        for context_option in (
            "--expected-source-commit",
            "--expected-source-tree-digest",
            "--expected-model-sha256",
            "--expected-config-sha256",
            "--expected-device-sha256",
            "--expected-policy-sha256",
        ):
            with self.subTest(context_option=context_option):
                self.assertNotIn(context_option, script[acceptance_gate:staging])
        for detached_contract in (
            "AcceptanceBundleRoot",
            "COGNIBOARD_EFFECTIVE_ACCEPTANCE_CHECKLIST_KO.md",
            "--release-attestation",
            "--release-attestation-signature",
            "--verifier-public-key",
            "--require-complete",
        ):
            with self.subTest(detached_contract=detached_contract):
                self.assertIn(detached_contract, script)

    @staticmethod
    def _powershell_51_path() -> Path:
        return (
            Path(os.environ.get("SystemRoot", r"C:\Windows"))
            / "System32"
            / "WindowsPowerShell"
            / "v1.0"
            / "powershell.exe"
        )

    @staticmethod
    def _write_release_toolchain_gate_fixture(
        fixture_root: Path,
        *,
        policy_bytes: bytes | None,
        pinned_policy_sha256: str | None = None,
    ) -> tuple[Path, Path, Path]:
        scripts = fixture_root / "scripts"
        config = fixture_root / "config"
        fake_path = fixture_root / "fake-path"
        scripts.mkdir(parents=True)
        config.mkdir(parents=True)
        fake_path.mkdir(parents=True)

        source_script = ROOT / "scripts" / "build_release_bundle.ps1"
        script_text = source_script.read_text(encoding="utf-8")
        if pinned_policy_sha256 is not None:
            original_policy_sha256 = sha256(
                (ROOT / "config" / "release-toolchain-policy.json").read_bytes()
            ).hexdigest()
            if original_policy_sha256 not in script_text:
                raise AssertionError("source policy pin was not found in builder")
            script_text = script_text.replace(
                original_policy_sha256,
                pinned_policy_sha256,
                1,
            )
        script_path = scripts / "build_release_bundle.ps1"
        script_path.write_text(script_text, encoding="utf-8")
        if policy_bytes is not None:
            (config / "release-toolchain-policy.json").write_bytes(policy_bytes)

        marker = fixture_root / "path-tool-executed.txt"
        marker_literal = str(marker).replace("%", "%%")
        for tool in ("python.cmd", "git.cmd"):
            (fake_path / tool).write_bytes(
                (
                    f'@echo off\r\n>>"{marker_literal}" echo {tool}\r\nexit /b 97\r\n'
                ).encode("ascii")
            )
        return script_path, fake_path, marker

    def _run_release_toolchain_gate(
        self,
        script_path: Path,
        fake_path: Path,
    ) -> subprocess.CompletedProcess[str]:
        powershell = self._powershell_51_path()
        if not powershell.is_file():
            self.skipTest("Windows PowerShell 5.1 is unavailable")
        environment = os.environ.copy()
        environment["PATH"] = str(fake_path)
        return subprocess.run(
            [
                str(powershell),
                "-NoLogo",
                "-NoProfile",
                "-NonInteractive",
                "-ExecutionPolicy",
                "Bypass",
                "-File",
                str(script_path),
                "-PublishRelease",
            ],
            cwd=script_path.parent.parent,
            env=environment,
            capture_output=True,
            check=False,
            text=True,
            timeout=30,
        )

    @unittest.skipUnless(sys.platform == "win32", "Windows release bootstrap")
    def test_publish_unconfigured_policy_blocks_before_path_tool_discovery(
        self,
    ) -> None:
        policy_bytes = (ROOT / "config" / "release-toolchain-policy.json").read_bytes()
        with tempfile.TemporaryDirectory() as temporary:
            fixture = Path(temporary)
            script, fake_path, marker = self._write_release_toolchain_gate_fixture(
                fixture,
                policy_bytes=policy_bytes,
            )
            completed = self._run_release_toolchain_gate(script, fake_path)

            self.assertNotEqual(completed.returncode, 0)
            self.assertIn(
                "publication is blocked before tool discovery",
                completed.stdout + completed.stderr,
            )
            self.assertFalse(
                marker.exists(), marker.read_text() if marker.exists() else ""
            )

    @unittest.skipUnless(sys.platform == "win32", "Windows release bootstrap")
    def test_publish_rejects_missing_or_mutated_toolchain_policy_before_path(
        self,
    ) -> None:
        source_policy = (ROOT / "config" / "release-toolchain-policy.json").read_bytes()
        cases = (
            ("missing", None, "must be an existing canonical absolute regular file"),
            (
                "digest-mismatch",
                source_policy + b" ",
                "bytes differ from the source-pinned digest",
            ),
            (
                "duplicate-key-mutation",
                source_policy.replace(
                    b'"status": "unconfigured"',
                    b'"status": "unconfigured",\n  "status": "approved"',
                ),
                "bytes differ from the source-pinned digest",
            ),
        )
        for name, policy_bytes, expected_error in cases:
            with self.subTest(name=name), tempfile.TemporaryDirectory() as temporary:
                script, fake_path, marker = self._write_release_toolchain_gate_fixture(
                    Path(temporary),
                    policy_bytes=policy_bytes,
                )
                completed = self._run_release_toolchain_gate(script, fake_path)
                self.assertNotEqual(completed.returncode, 0)
                self.assertIn(expected_error, completed.stdout + completed.stderr)
                self.assertFalse(marker.exists())

    @unittest.skipUnless(sys.platform == "win32", "Windows release bootstrap")
    def test_publish_correct_host_still_blocks_without_enforced_build_closure(
        self,
    ) -> None:
        powershell = self._powershell_51_path()
        if not powershell.is_file():
            self.skipTest("Windows PowerShell 5.1 is unavailable")
        with tempfile.TemporaryDirectory() as temporary:
            fixture = Path(temporary)
            tools = fixture / "approved-tools"
            tools.mkdir()
            files = {
                name: tools / name
                for name in (
                    "python.exe",
                    "git.exe",
                    "closure.json",
                    "wheelhouse.json",
                )
            }
            for path in files.values():
                path.write_bytes(b"must never execute\n")
            policy = {
                "schema": "cogni.release.toolchain-policy.v2",
                "status": "approved",
                "runner_mode": "protected-no-profile-isolated-runner",
                "powershell_path": str(powershell.resolve()),
                "powershell_sha256": sha256(powershell.read_bytes()).hexdigest(),
                "python_path": str(files["python.exe"].resolve()),
                "python_sha256": sha256(files["python.exe"].read_bytes()).hexdigest(),
                "git_path": str(files["git.exe"].resolve()),
                "git_sha256": sha256(files["git.exe"].read_bytes()).hexdigest(),
                "build_closure_manifest_path": str(files["closure.json"].resolve()),
                "build_closure_manifest_sha256": sha256(
                    files["closure.json"].read_bytes()
                ).hexdigest(),
                "offline_wheelhouse_manifest_path": str(
                    files["wheelhouse.json"].resolve()
                ),
                "offline_wheelhouse_manifest_sha256": sha256(
                    files["wheelhouse.json"].read_bytes()
                ).hexdigest(),
            }
            policy_bytes = (json.dumps(policy, indent=2) + "\n").encode("utf-8")
            script, fake_path, marker = self._write_release_toolchain_gate_fixture(
                fixture,
                policy_bytes=policy_bytes,
                pinned_policy_sha256=sha256(policy_bytes).hexdigest(),
            )
            completed = self._run_release_toolchain_gate(script, fake_path)
            self.assertNotEqual(completed.returncode, 0)
            self.assertIn(
                "EXTERNAL_BLOCKER: protected whole-build closure",
                completed.stdout + completed.stderr,
            )
            self.assertFalse(marker.exists())

    @unittest.skipUnless(sys.platform == "win32", "Windows release bootstrap")
    def test_publish_rejects_wrong_powershell_path_or_digest_before_other_tools(
        self,
    ) -> None:
        powershell = self._powershell_51_path()
        if not powershell.is_file():
            self.skipTest("Windows PowerShell 5.1 is unavailable")
        cases = ("wrong-host-path", "wrong-host-digest")
        for case in cases:
            with self.subTest(case=case), tempfile.TemporaryDirectory() as temporary:
                fixture = Path(temporary)
                tools = fixture / "approved-tools"
                tools.mkdir()
                python_tool = tools / "python.exe"
                git_tool = tools / "git.exe"
                python_tool.write_bytes(b"not an executable; must never run\n")
                git_tool.write_bytes(b"not an executable; must never run\n")
                closure_manifest = tools / "closure.json"
                wheelhouse_manifest = tools / "wheelhouse.json"
                closure_manifest.write_bytes(b"{}\n")
                wheelhouse_manifest.write_bytes(b"{}\n")
                powershell_path = str(powershell.resolve())
                powershell_digest = sha256(powershell.read_bytes()).hexdigest()
                if case == "wrong-host-path":
                    powershell_path = str(python_tool.resolve())
                    expected_error = "Running PowerShell host path differs"
                else:
                    powershell_digest = "0" * 64
                    expected_error = "digest differs from the source-pinned policy"
                policy = {
                    "schema": "cogni.release.toolchain-policy.v2",
                    "status": "approved",
                    "runner_mode": "protected-no-profile-isolated-runner",
                    "powershell_path": powershell_path,
                    "powershell_sha256": powershell_digest,
                    "python_path": str(python_tool.resolve()),
                    "python_sha256": sha256(python_tool.read_bytes()).hexdigest(),
                    "git_path": str(git_tool.resolve()),
                    "git_sha256": sha256(git_tool.read_bytes()).hexdigest(),
                    "build_closure_manifest_path": str(closure_manifest.resolve()),
                    "build_closure_manifest_sha256": sha256(
                        closure_manifest.read_bytes()
                    ).hexdigest(),
                    "offline_wheelhouse_manifest_path": str(
                        wheelhouse_manifest.resolve()
                    ),
                    "offline_wheelhouse_manifest_sha256": sha256(
                        wheelhouse_manifest.read_bytes()
                    ).hexdigest(),
                }
                policy_bytes = (json.dumps(policy, indent=2) + "\n").encode("utf-8")
                policy_digest = sha256(policy_bytes).hexdigest()
                script, fake_path, marker = self._write_release_toolchain_gate_fixture(
                    fixture,
                    policy_bytes=policy_bytes,
                    pinned_policy_sha256=policy_digest,
                )
                completed = self._run_release_toolchain_gate(script, fake_path)
                self.assertNotEqual(completed.returncode, 0)
                self.assertIn(expected_error, completed.stdout + completed.stderr)
                self.assertFalse(marker.exists())

    @unittest.skipUnless(sys.platform == "win32", "Windows release bootstrap")
    def test_publish_rejects_source_pinned_policy_with_nonexact_keys(self) -> None:
        base_policy = {
            "schema": "cogni.release.toolchain-policy.v2",
            "status": "unconfigured",
            "runner_mode": None,
            "powershell_path": None,
            "powershell_sha256": None,
            "python_path": None,
            "python_sha256": None,
            "git_path": None,
            "git_sha256": None,
            "build_closure_manifest_path": None,
            "build_closure_manifest_sha256": None,
            "offline_wheelhouse_manifest_path": None,
            "offline_wheelhouse_manifest_sha256": None,
        }
        cases = {
            "extra": {**base_policy, "unexpected": None},
            "missing": {
                key: value for key, value in base_policy.items() if key != "git_sha256"
            },
        }
        for name, policy in cases.items():
            with self.subTest(name=name), tempfile.TemporaryDirectory() as temporary:
                policy_bytes = (json.dumps(policy, indent=2) + "\n").encode("utf-8")
                script, fake_path, marker = self._write_release_toolchain_gate_fixture(
                    Path(temporary),
                    policy_bytes=policy_bytes,
                    pinned_policy_sha256=sha256(policy_bytes).hexdigest(),
                )
                completed = self._run_release_toolchain_gate(script, fake_path)
                self.assertNotEqual(completed.returncode, 0)
                self.assertIn(
                    "keys do not match the closed schema",
                    completed.stdout + completed.stderr,
                )
                self.assertFalse(marker.exists())

    def test_publish_toolchain_gate_precedes_all_path_discovery_and_git_use(
        self,
    ) -> None:
        script = (ROOT / "scripts" / "build_release_bundle.ps1").read_text(
            encoding="utf-8"
        )
        gate = script.index("if ($PublishRelease) {")
        path_discovery = script.index(
            "$python = (Microsoft.PowerShell.Core\\Get-Command python"
        )
        first_git_execution = script.index("$commitOid = (& $gitExecutable")
        self.assertLess(gate, path_discovery)
        self.assertLess(path_discovery, first_git_execution)
        self.assertLess(script.index("function Open-EarlyPinnedFile"), gate)
        self.assertIn("[IO.FileShare]::Read", script[:path_discovery])
        self.assertNotIn("& powershell ", script.casefold())

    @unittest.skipUnless(sys.platform == "win32", "Windows release bootstrap")
    def test_publish_bootstrap_ignores_malicious_shadow_commands(self) -> None:
        policy_bytes = (ROOT / "config" / "release-toolchain-policy.json").read_bytes()
        powershell = self._powershell_51_path()
        if not powershell.is_file():
            self.skipTest("Windows PowerShell 5.1 is unavailable")
        with tempfile.TemporaryDirectory() as temporary:
            fixture = Path(temporary)
            script, fake_path, path_marker = self._write_release_toolchain_gate_fixture(
                fixture,
                policy_bytes=policy_bytes,
            )
            shadow_marker = fixture / "shadow-command-executed.txt"
            runner = fixture / "hostile-runner.ps1"
            command_names = (
                "Resolve-Path",
                "Get-FileHash",
                "Expand-Archive",
                "Copy-Item",
                "Move-Item",
                "Get-ChildItem",
                "Test-Path",
                "Remove-Item",
                "New-Item",
                "Get-Item",
                "Join-Path",
                "Split-Path",
                "Compare-Object",
                "New-Object",
                "Add-Type",
            )
            marker_literal = str(shadow_marker).replace("'", "''")
            definitions = "\n".join(
                f"function global:{name} {{ "
                f"[IO.File]::AppendAllText('{marker_literal}', '{name}\\n'); "
                "throw 'MALICIOUS SHADOW RAN' }"
                for name in command_names
            )
            script_literal = str(script).replace("'", "''")
            runner.write_text(
                definitions + f"\n& '{script_literal}' -PublishRelease\n",
                encoding="utf-8-sig",
            )
            environment = os.environ.copy()
            environment["PATH"] = str(fake_path)
            completed = subprocess.run(
                [
                    str(powershell),
                    "-NoLogo",
                    "-NoProfile",
                    "-NonInteractive",
                    "-ExecutionPolicy",
                    "Bypass",
                    "-File",
                    str(runner),
                ],
                cwd=fixture,
                env=environment,
                capture_output=True,
                check=False,
                text=True,
                timeout=30,
            )
            self.assertNotEqual(completed.returncode, 0)
            self.assertIn(
                "publication is blocked before tool discovery",
                completed.stdout + completed.stderr,
            )
            self.assertFalse(shadow_marker.exists())
            self.assertFalse(path_marker.exists())

    @unittest.skipUnless(sys.platform == "win32", "Windows artifact-only build")
    def test_artifact_only_release_build_still_passes_on_powershell_51(self) -> None:
        powershell = self._powershell_51_path()
        git = shutil.which("git")
        if not powershell.is_file() or git is None:
            self.skipTest("Windows PowerShell 5.1 and Git are required")

        with tempfile.TemporaryDirectory() as temporary:
            temporary_root = Path(temporary)
            repository = temporary_root / "repository"
            output = temporary_root / "artifact"
            hostile_path = temporary_root / "hostile-path"
            hostile_path.mkdir()
            powershell_marker = temporary_root / "path-powershell-executed.txt"
            (hostile_path / "powershell.cmd").write_text(
                "@echo off\n"
                f'>>"{powershell_marker}" echo PATH_POWERSHELL_RAN\n'
                "exit /b 97\n",
                encoding="utf-8",
            )
            shutil.copytree(
                ROOT,
                repository,
                ignore=shutil.ignore_patterns(
                    ".git",
                    ".pytest_cache",
                    ".ruff_cache",
                    "__pycache__",
                    "work",
                    "*.pyc",
                ),
            )
            for arguments in (
                ("init", "-q"),
                ("config", "user.email", "release-regression@example.invalid"),
                ("config", "user.name", "Release Regression"),
                ("config", "core.autocrlf", "false"),
                ("add", "-A"),
                ("commit", "-q", "-m", "artifact-only regression"),
            ):
                subprocess.run(
                    [git, "-C", str(repository), *arguments],
                    check=True,
                    capture_output=True,
                    timeout=30,
                )

            environment = os.environ.copy()
            environment["PATH"] = str(hostile_path) + os.pathsep + environment["PATH"]
            completed = subprocess.run(
                [
                    str(powershell),
                    "-NoLogo",
                    "-NoProfile",
                    "-NonInteractive",
                    "-ExecutionPolicy",
                    "Bypass",
                    "-File",
                    str(repository / "scripts" / "build_release_bundle.ps1"),
                    "-OutputDirectory",
                    str(output),
                ],
                cwd=repository,
                env=environment,
                capture_output=True,
                check=False,
                text=True,
                timeout=180,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertIn("artifact_build=PASS", completed.stdout)
            self.assertIn("release_evidence_status=UNVERIFIED", completed.stdout)
            manifest = (output / "BUILD_MANIFEST.txt").read_text(encoding="utf-8")
            self.assertIn("artifact_build_status=PASS", manifest)
            self.assertIn("release_evidence_status=UNVERIFIED", manifest)
            checksum_records = {}
            for line in (
                (output / "SHA256SUMS.txt").read_text(encoding="utf-8").splitlines()
            ):
                digest, relative = line.split("  ", 1)
                checksum_records[relative] = digest
            expected_paths = {
                path.relative_to(output).as_posix()
                for path in output.rglob("*")
                if path.is_file() and path.name != "SHA256SUMS.txt"
            }
            self.assertEqual(set(checksum_records), expected_paths)
            for relative, digest in checksum_records.items():
                self.assertEqual(
                    digest,
                    sha256((output / relative).read_bytes()).hexdigest(),
                )
            self.assertFalse(powershell_marker.exists())

    @unittest.skipUnless(sys.platform == "win32", "Windows publication gate")
    @unittest.skip(
        "Checklist publication integration requires the externally protected "
        "whole-closure runner; the local builder is intentionally fail-closed"
    )
    def test_publish_acceptance_checklist_failures_create_no_release_output(
        self,
    ) -> None:
        powershell = self._powershell_51_path()
        git_command = shutil.which("git")
        if not powershell.is_file() or git_command is None:
            self.skipTest("Windows PowerShell 5.1 and Git are required")
        git = Path(git_command).resolve()
        python = Path(sys.executable).resolve()

        for corruption in ("master", "outstanding"):
            with (
                self.subTest(corruption=corruption),
                tempfile.TemporaryDirectory() as temporary,
            ):
                temporary_root = Path(temporary)
                repository = temporary_root / "repository"
                output = temporary_root / "published-release"
                shutil.copytree(
                    ROOT,
                    repository,
                    ignore=shutil.ignore_patterns(
                        ".git",
                        ".pytest_cache",
                        ".ruff_cache",
                        "__pycache__",
                        "work",
                        "*.pyc",
                    ),
                )

                policy = {
                    "schema": "cogni.release.toolchain-policy.v2",
                    "status": "approved",
                    "runner_mode": "protected-no-profile-isolated-runner",
                    "powershell_path": str(powershell.resolve()),
                    "powershell_sha256": sha256(powershell.read_bytes()).hexdigest(),
                    "python_path": str(python),
                    "python_sha256": sha256(python.read_bytes()).hexdigest(),
                    "git_path": str(git),
                    "git_sha256": sha256(git.read_bytes()).hexdigest(),
                    "build_closure_manifest_path": str(python),
                    "build_closure_manifest_sha256": sha256(
                        python.read_bytes()
                    ).hexdigest(),
                    "offline_wheelhouse_manifest_path": str(git),
                    "offline_wheelhouse_manifest_sha256": sha256(
                        git.read_bytes()
                    ).hexdigest(),
                }
                policy_bytes = (json.dumps(policy, indent=2) + "\n").encode("utf-8")
                policy_path = repository / "config" / "release-toolchain-policy.json"
                policy_path.write_bytes(policy_bytes)
                builder = repository / "scripts" / "build_release_bundle.ps1"
                builder_source = builder.read_text(encoding="utf-8")
                unconfigured_policy_sha = sha256(
                    (ROOT / "config" / "release-toolchain-policy.json").read_bytes()
                ).hexdigest()
                builder.write_text(
                    builder_source.replace(
                        unconfigured_policy_sha,
                        sha256(policy_bytes).hexdigest(),
                        1,
                    ),
                    encoding="utf-8",
                )

                # The orchestration test substitutes an exact archived/current
                # validator that emits a bounded passing scope.  Cryptographic
                # release-evidence semantics are covered independently by
                # test_release_evidence_validation.py; this fixture reaches the
                # two checklist gates without fabricating lab GPU evidence.
                release_validator = (
                    repository / "scripts" / "validate_release_evidence.py"
                )
                atomic_marker = temporary_root / "atomic-replace-result.txt"
                release_validator.write_text(
                    f"""from __future__ import annotations
import json
import os
from pathlib import Path
import sys

def value(name: str) -> str:
    return sys.argv[sys.argv.index(name) + 1]

payload = {{
    "schema": "cogni.release.validation.v2",
    "status": "passed",
    "source_commit": value("--expected-source-commit"),
    "source_tree_digest": "1" * 64,
    "model_tree_digest": "2" * 64,
    "config_digest": "3" * 64,
    "device_digest": "4" * 64,
}}
target = (
    Path(value("--expanded-source"))
    / "docs"
    / "COGNIBOARD_MASTER_ACCEPTANCE_CHECKLIST_KO.md"
)
replacement = target.with_name(target.name + ".atomic-replacement")
replacement.write_bytes(target.read_bytes() + b"\\natomic replacement\\n")
try:
    os.replace(replacement, target)
except PermissionError:
    Path({str(atomic_marker)!r}).write_text("BLOCKED\\n", encoding="utf-8")
    replacement.unlink()
else:
    Path({str(atomic_marker)!r}).write_text("REPLACED\\n", encoding="utf-8")
print(json.dumps(payload, sort_keys=True))
""",
                    encoding="utf-8",
                )
                dummy_evidence = temporary_root / "dummy-evidence.bin"
                dummy_evidence.write_bytes(b"orchestration-only evidence\n")
                dummy_sha256 = sha256(dummy_evidence.read_bytes()).hexdigest()
                evidence_arguments: list[str] = []
                for parameter in (
                    "ReleaseEvidenceSummary",
                    "CpuGateEvidence",
                    "Gpu5GateEvidence",
                    "ReleaseAttestation",
                    "ReleaseAttestationSignature",
                    "RuntimeEvidence",
                    "CompletionEvidence",
                    "IdentityPreEvidence",
                    "IdentityPostEvidence",
                    "ConfigEvidence",
                    "DeviceEvidence",
                    "ModelInventory",
                ):
                    evidence_arguments.extend(
                        (
                            f"-{parameter}Path",
                            str(dummy_evidence),
                            f"-{parameter}Sha256",
                            dummy_sha256,
                        )
                    )
                evidence_arguments.extend(
                    ("-VerifierPublicKeyPath", str(dummy_evidence))
                )
                acceptance_bundle = temporary_root / "acceptance-bundle"
                (acceptance_bundle / "docs").mkdir(parents=True)
                (acceptance_bundle / "release" / "evidence").mkdir(parents=True)
                shutil.copy2(
                    repository
                    / "docs"
                    / "COGNIBOARD_MASTER_ACCEPTANCE_CHECKLIST_KO.md",
                    acceptance_bundle
                    / "docs"
                    / "COGNIBOARD_EFFECTIVE_ACCEPTANCE_CHECKLIST_KO.md",
                )
                shutil.copy2(
                    dummy_evidence,
                    acceptance_bundle / "release" / "evidence" / "dummy.bin",
                )
                evidence_arguments.extend(
                    ("-AcceptanceBundleRoot", str(acceptance_bundle))
                )

                if corruption == "master":
                    master = (
                        repository
                        / "docs"
                        / "COGNIBOARD_MASTER_ACCEPTANCE_CHECKLIST_KO.md"
                    )
                    with master.open("a", encoding="utf-8") as stream:
                        stream.write(
                            "\n| 1 | [ ] | duplicate | `PARTIAL` | evidence | gate |\n"
                        )
                    expected_error = "requirement ID coverage failed"
                else:
                    outstanding = (
                        repository
                        / "docs"
                        / "COGNIBOARD_OUTSTANDING_IMPLEMENTATION_CHECKLIST_KO.md"
                    )
                    with outstanding.open("a", encoding="utf-8") as stream:
                        stream.write("\nintentional renderer drift\n")
                    expected_error = "not derived from the master ledger"

                for arguments in (
                    ("init", "-q"),
                    ("config", "user.email", "release-regression@example.invalid"),
                    ("config", "user.name", "Release Regression"),
                    ("config", "core.autocrlf", "false"),
                    ("add", "-A"),
                    ("commit", "-q", "-m", f"acceptance {corruption} failure"),
                ):
                    subprocess.run(
                        [str(git), "-C", str(repository), *arguments],
                        check=True,
                        capture_output=True,
                        timeout=30,
                    )

                completed = subprocess.run(
                    [
                        str(powershell),
                        "-NoLogo",
                        "-NoProfile",
                        "-NonInteractive",
                        "-ExecutionPolicy",
                        "Bypass",
                        "-File",
                        str(builder),
                        "-PublishRelease",
                        "-OutputDirectory",
                        str(output),
                        *evidence_arguments,
                    ],
                    cwd=repository,
                    env=os.environ.copy(),
                    capture_output=True,
                    check=False,
                    text=True,
                    timeout=60,
                )
                self.assertNotEqual(completed.returncode, 0)
                self.assertIn(expected_error, completed.stdout + completed.stderr)
                self.assertEqual(
                    atomic_marker.read_text(encoding="utf-8"),
                    "BLOCKED\n",
                )
                self.assertFalse(output.exists())
                self.assertEqual(
                    list(temporary_root.glob(".cogni-release-staging-*")),
                    [],
                )

    def test_expanded_source_tree_is_closed_world_checksummed(self) -> None:
        script = (ROOT / "scripts" / "build_release_bundle.ps1").read_text(
            encoding="utf-8"
        )

        for contract in (
            "Get-ClosedTreeChecksumLines",
            "Open-VerifiedSourceReadLocks",
            "[IO.FileShare]::Read",
            "Expanded-source inventory changed while acquiring read locks",
            "foreach ($sourceReadLock in $sourceReadLocks)",
            "Bundled expanded source inventory or bytes differ from the archived commit",
            "SOURCE_TREE_SHA256SUMS.txt",
            "Get-LockedTreeChecksumLines",
            "$payloadStageInventory",
            "$finalStageInventory",
            "Locked published release output",
        ):
            with self.subTest(contract=contract):
                self.assertIn(contract, script)

        inventory = script.index(
            "$archivedTreeChecksums = @(Get-ClosedTreeChecksumLines $source)"
        )
        locks = script.index(
            "Open-VerifiedSourceReadLocks $source $archivedTreeChecksums"
        )
        release_gate = script.index(
            "& $python -I -S -B -",
        )
        acceptance_gate = script.index(
            "& $python -I -S -B $archivedAcceptanceValidator",
        )
        publication = script.index(
            "$publishStage = New-ReleasePublishStage $publishedOutput",
        )
        unlock = script.index(
            "foreach ($sourceReadLock in $sourceReadLocks)",
        )
        self.assertLess(inventory, locks)
        self.assertLess(locks, release_gate)
        self.assertLess(release_gate, acceptance_gate)
        self.assertLess(acceptance_gate, publication)
        self.assertLess(publication, unlock)

    def test_manual_pdf_and_final_publication_are_digest_bound(self) -> None:
        script = (ROOT / "scripts" / "build_release_bundle.ps1").read_text(
            encoding="utf-8"
        )
        for contract in (
            "[string]$ManualPdfSha256",
            "Assert-ExactSha256 $ManualPdfSha256 'Manual PDF digest'",
            "New-VerifiedEvidenceSnapshot",
            "$ManualPdfPath $ManualPdfSha256 'Manual PDF'",
            "Copy-VerifiedSnapshotToBundle",
            "$manualPdfSnapshot $ManualPdfSha256",
            "Get-LockedTreeChecksumLines $payloadStageReadLocks",
            "Open-VerifiedSourceReadLocks $publishStage $finalStageInventory",
            "Assert-ClosedTreeInventory",
            "Open-VerifiedSourceReadLocks $publishedOutput $finalStageInventory",
        ):
            with self.subTest(contract=contract):
                self.assertIn(contract, script)

    def test_release_script_supports_windows_powershell_51_path_apis(self) -> None:
        script = (ROOT / "scripts" / "build_release_bundle.ps1").read_text(
            encoding="utf-8"
        )

        self.assertNotIn("[IO.Path]::GetRelativePath", script)
        self.assertNotIn("[IO.Path]::IsPathFullyQualified", script)
        self.assertIn("function Get-SafeRelativePath", script)
        self.assertIn("function Test-FullyQualifiedCanonicalPath", script)
        self.assertIn("[StringComparison]::OrdinalIgnoreCase", script)

    def test_validation_docs_publish_no_direct_lab_cuda_bypass(self) -> None:
        validation = (ROOT / "docs" / "VALIDATION.md").read_text(encoding="utf-8")
        gemma = (ROOT / "docs" / "GEMMA4_VALIDATION.md").read_text(encoding="utf-8")
        plan = (
            ROOT / "docs" / "COGNIBOARD_V041_SERVER_IMPLEMENTATION_PLAN_KO.md"
        ).read_text(encoding="utf-8")

        self.assertNotIn("  --device cuda `", validation)
        self.assertNotIn("python -m scripts.validate_gemma4_local_voice", validation)
        self.assertNotIn("python -m scripts.validate_gemma4_local_image", validation)
        self.assertNotIn("python scripts\\validate_agent_completion.py", gemma)
        self.assertNotIn("python scripts\\validate_agent_casual_korean.py", gemma)
        self.assertIn("sole GPU5", validation)
        self.assertIn("guard allowlist", validation)
        self.assertIn(
            "Stage G product-completion gate is `EXTERNAL_BLOCKER / NOT RUN`",
            gemma,
        )
        self.assertIn("증거 완료율 0%", plan)

    def test_archive_entries_are_validated_before_expansion(self) -> None:
        script = (ROOT / "scripts" / "build_release_bundle.ps1").read_text(
            encoding="utf-8"
        )
        validation = "$archiveCheckpoint = Get-ArchiveEntrySha256"
        expansion = "[IO.Compression.ZipFile]::ExtractToDirectory"

        self.assertEqual(script.count(validation), 1)
        self.assertLess(script.index(validation), script.index(expansion))
        self.assertIn("[StringComparer]::OrdinalIgnoreCase", script)
        self.assertIn("$name.Contains('\\')", script)
        self.assertIn("$expandedBytes -gt 2147483648", script)


if __name__ == "__main__":
    unittest.main()
