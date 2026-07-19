from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
from types import SimpleNamespace
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import cogni_flow.kernel_sandbox as kernel_sandbox
from cogni_flow.kernel_sandbox import (
    KernelSandboxError,
    LinuxOciSandboxRunner,
    build_kernel_sandbox_evidence_payload,
    parse_kernel_sandbox_evidence,
)
from cogni_flow.production import command_sha256
from scripts.validate_kernel_sandbox import _canonical_daemon_socket


ENGINE_DIGEST = "a" * 64
IMAGE = "cogni-os-dev@sha256:" + "b" * 64
COMMANDS = (("python", "-I", "/project/check.py"),)


def evidence_payload(**changes) -> bytes:
    values = {
        "runner_id": "lab-gpu5-cpu-sandbox",
        "engine_path": "/usr/bin/docker",
        "engine_sha256": ENGINE_DIGEST,
        "daemon_socket": "/var/run/docker.sock",
        "image_reference": IMAGE,
        "commands": COMMANDS,
        "max_memory_bytes": 512 * 1024 * 1024,
        "max_pids": 64,
        "max_cpus": 2.0,
        "tmpfs_bytes": 64 * 1024 * 1024,
    }
    values.update(changes)
    return build_kernel_sandbox_evidence_payload(**values)


class TestKernelSandboxEvidence(unittest.TestCase):
    def test_canonical_builder_round_trip_binds_exact_commands(self):
        raw = evidence_payload()
        parsed = parse_kernel_sandbox_evidence(raw)

        self.assertEqual(parsed.runner_id, "lab-gpu5-cpu-sandbox")
        self.assertEqual(parsed.image_reference, IMAGE)
        self.assertEqual(
            parsed.allowed_command_sha256,
            (command_sha256(COMMANDS[0]),),
        )
        self.assertEqual(
            raw,
            json.dumps(
                json.loads(raw),
                ensure_ascii=True,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("ascii"),
        )

    def test_parser_rejects_missing_or_false_boundary(self):
        data = json.loads(evidence_payload())
        data.pop("cap_drop_all")
        with self.assertRaisesRegex(KernelSandboxError, "fields"):
            parse_kernel_sandbox_evidence(json.dumps(data).encode())

        data = json.loads(evidence_payload())
        data["read_only_project"] = False
        with self.assertRaisesRegex(KernelSandboxError, "read_only_project"):
            parse_kernel_sandbox_evidence(json.dumps(data).encode())

    def test_parser_rejects_duplicate_json_keys(self):
        raw = evidence_payload().decode("ascii")
        duplicate = raw.replace(
            '"schema":', '"schema":"attacker-selected","schema":', 1
        ).encode("ascii")

        with self.assertRaisesRegex(KernelSandboxError, "duplicate key: schema"):
            parse_kernel_sandbox_evidence(duplicate)

    def test_parser_rejects_unpinned_image_and_network(self):
        data = json.loads(evidence_payload())
        data["image_reference"] = "cogni-os-dev:latest"
        with self.assertRaisesRegex(KernelSandboxError, "exact sha256"):
            parse_kernel_sandbox_evidence(json.dumps(data).encode())

        data = json.loads(evidence_payload())
        data["network_mode"] = "bridge"
        with self.assertRaisesRegex(KernelSandboxError, "network_mode=none"):
            parse_kernel_sandbox_evidence(json.dumps(data).encode())

    def test_parser_rejects_duplicate_commands_and_unsafe_bounds(self):
        data = json.loads(evidence_payload())
        data["allowed_command_sha256"].append(data["allowed_command_sha256"][0])
        with self.assertRaisesRegex(KernelSandboxError, "duplicates"):
            parse_kernel_sandbox_evidence(json.dumps(data).encode())

        data = json.loads(evidence_payload())
        data["max_pids"] = 100_000
        with self.assertRaisesRegex(KernelSandboxError, "max_pids"):
            parse_kernel_sandbox_evidence(json.dumps(data).encode())

    def test_parser_is_cross_platform_for_linux_absolute_paths(self):
        parsed = parse_kernel_sandbox_evidence(evidence_payload())
        self.assertEqual(parsed.engine_path, "/usr/bin/docker")
        self.assertEqual(parsed.daemon_socket, "/var/run/docker.sock")

    @unittest.skipIf(sys.platform.startswith("linux"), "Windows fail-closed test")
    def test_runner_fails_closed_outside_linux_before_reading_evidence(self):
        with self.assertRaisesRegex(KernelSandboxError, "only on Linux"):
            LinuxOciSandboxRunner(Path("missing-evidence.json"))

    def test_evidence_digest_changes_for_any_reviewed_constraint(self):
        original = evidence_payload()
        changed = evidence_payload(max_pids=65)
        self.assertNotEqual(sha256(original).digest(), sha256(changed).digest())

    def test_configuration_never_self_attests_production_isolation(self):
        runner = object.__new__(LinuxOciSandboxRunner)
        runner.evidence = parse_kernel_sandbox_evidence(evidence_payload())
        runner._evidence_sha256 = sha256(evidence_payload()).hexdigest()

        attestation = runner.isolation_attestation()

        self.assertFalse(runner.kernel_isolated)
        self.assertTrue(runner.integration_smoke_only)
        self.assertFalse(attestation.kernel_boundary)
        self.assertFalse(attestation.network_isolated)
        self.assertFalse(attestation.host_filesystem_isolated)
        self.assertFalse(attestation.ephemeral_workspace)

    def test_argv_hardens_image_execution_and_resource_bounds(self):
        runner = object.__new__(LinuxOciSandboxRunner)
        runner.evidence = parse_kernel_sandbox_evidence(evidence_payload())
        runner._engine = Path("/usr/bin/docker")
        name = "cogni-candidate-" + "c" * 32

        argv = runner._build_argv(
            Path("/tmp/project"),
            COMMANDS[0],
            Path("/tmp/container.cid"),
            container_name=name,
        )

        self.assertIn("--no-healthcheck", argv)
        self.assertEqual(argv[argv.index("--log-driver") + 1], "none")
        self.assertEqual(argv[argv.index("--memory-swap") + 1], str(512 * 1024 * 1024))
        self.assertEqual(argv[argv.index("--name") + 1], name)
        self.assertEqual(argv[argv.index("--entrypoint") + 1], "python")
        image_index = argv.index(IMAGE)
        self.assertEqual(argv[image_index + 1 :], ("-I", "/project/check.py"))


class TestKernelSandboxCleanup(unittest.TestCase):
    def setUp(self):
        self.runner = object.__new__(LinuxOciSandboxRunner)
        self.runner.evidence = SimpleNamespace(daemon_socket="/var/run/docker.sock")
        self.name = "cogni-candidate-" + "d" * 32
        self.not_found = f"Error response from daemon: No such container: {self.name}"

    def test_missing_cidfile_never_verifies_cleanup(self):
        with tempfile.TemporaryDirectory() as tmp:
            cidfile = Path(tmp) / "missing.cid"
            responses = [
                (1, self.not_found),
                (1, f"Error: No such object: {self.name}"),
                (1, f"Error: No such object: {self.name}"),
            ]
            with (
                patch.object(self.runner, "_docker_control", side_effect=responses),
                patch.object(kernel_sandbox.time, "sleep"),
            ):
                error = self.runner._force_remove(
                    cidfile, container_name=self.name, env={}
                )

        self.assertEqual(error, "container cleanup unverified: cidfile is missing")

    def test_ambiguous_inspect_failure_is_not_absence(self):
        with tempfile.TemporaryDirectory() as tmp:
            cidfile = Path(tmp) / "container.cid"
            cidfile.write_text("a" * 64, encoding="ascii")
            with patch.object(
                self.runner,
                "_docker_control",
                side_effect=[
                    (1, self.not_found),
                    (1, "Cannot connect to the Docker daemon"),
                ],
            ):
                error = self.runner._force_remove(
                    cidfile, container_name=self.name, env={}
                )

        self.assertIn("survivor check ambiguous", error)

    def test_exact_not_found_for_name_and_cid_verifies_cleanup(self):
        container_id = "a" * 64
        absent_name = f"Error: No such object: {self.name}"
        absent_id = f"Error: No such object: {container_id}"
        with tempfile.TemporaryDirectory() as tmp:
            cidfile = Path(tmp) / "container.cid"
            cidfile.write_text(container_id, encoding="ascii")
            responses = [
                (1, self.not_found),
                (1, f"[]\n{absent_name}"),
                (1, f"[]\n{absent_id}"),
                (1, f"[]\n{absent_name}"),
                (1, f"[]\n{absent_id}"),
            ]
            with (
                patch.object(self.runner, "_docker_control", side_effect=responses),
                patch.object(kernel_sandbox.time, "sleep"),
            ):
                error = self.runner._force_remove(
                    cidfile, container_name=self.name, env={}
                )

        self.assertIsNone(error)


class TestDockerNotFoundParsing(unittest.TestCase):
    def test_accepts_optional_empty_inspect_json_line(self):
        name = "cogni-candidate-" + "e" * 32
        error = f"Error response from daemon: No such container: {name}"

        self.assertTrue(kernel_sandbox._is_exact_not_found(error))
        self.assertTrue(kernel_sandbox._is_exact_not_found(f"[]\n{error}"))
        self.assertTrue(kernel_sandbox._is_exact_not_found(f"[]\r\n{error}"))

    def test_rejects_extra_or_other_daemon_output(self):
        name = "cogni-candidate-" + "e" * 32
        error = f"Error response from daemon: No such container: {name}"
        rejected = (
            f"{{}}\n{error}",
            f"[]\n[]\n{error}",
            f"[]\n{error}\nwarning: daemon restarted",
            "[]\nCannot connect to the Docker daemon",
            f"container list follows\n{error}",
        )

        for detail in rejected:
            with self.subTest(detail=detail):
                self.assertFalse(kernel_sandbox._is_exact_not_found(detail))


class TestValidatorSocketCanonicalization(unittest.TestCase):
    def test_operator_socket_is_resolved_strictly_before_evidence(self):
        canonical = Path("/run/docker.sock")
        with patch.object(Path, "resolve", return_value=canonical) as resolve:
            result = _canonical_daemon_socket("/var/run/docker.sock")

        self.assertEqual(result, canonical)
        resolve.assert_called_once_with(strict=True)


class TestBoundedOutput(unittest.TestCase):
    @unittest.skipUnless(sys.platform.startswith("linux"), "Linux pipe selector only")
    def test_output_is_hard_bounded_and_process_is_stopped(self):
        process = subprocess.Popen(
            (sys.executable, "-c", "print('x' * 100000)"),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )

        output, timed_out, output_limited = kernel_sandbox._collect_bounded_output(
            process, timeout_seconds=10, maximum_bytes=4_096
        )

        self.assertEqual(len(output), 4_096)
        self.assertFalse(timed_out)
        self.assertTrue(output_limited)
        self.assertIsNotNone(process.returncode)


class TestKernelSandboxLinuxHostValidation(unittest.TestCase):
    @unittest.skipUnless(sys.platform.startswith("linux"), "Linux host only")
    def test_runner_rejects_engine_digest_mismatch_before_socket_use(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            engine = root / "docker"
            engine.write_bytes(b"not-docker")
            engine.chmod(0o755)
            evidence = root / "evidence.json"
            evidence.write_bytes(
                evidence_payload(
                    engine_path=str(engine),
                    daemon_socket="/var/run/docker.sock",
                )
            )
            with self.assertRaisesRegex(KernelSandboxError, "digest"):
                LinuxOciSandboxRunner(evidence)

    @unittest.skipUnless(sys.platform.startswith("linux"), "Linux host only")
    def test_runner_rejects_non_socket_daemon_endpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            engine = root / "docker"
            engine.write_bytes(b"engine")
            engine.chmod(0o755)
            endpoint = root / "docker.sock"
            endpoint.write_text("not a socket", encoding="ascii")
            evidence = root / "evidence.json"
            evidence.write_bytes(
                evidence_payload(
                    engine_path=str(engine),
                    engine_sha256=sha256(engine.read_bytes()).hexdigest(),
                    daemon_socket=str(endpoint),
                )
            )
            with self.assertRaisesRegex(KernelSandboxError, "local socket"):
                LinuxOciSandboxRunner(evidence)


if __name__ == "__main__":
    unittest.main()
