from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
import sys
import tempfile
import unittest

from cogni_flow.kernel_sandbox import (
    KernelSandboxError,
    LinuxOciSandboxRunner,
    build_kernel_sandbox_evidence_payload,
    parse_kernel_sandbox_evidence,
)
from cogni_flow.production import command_sha256


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
