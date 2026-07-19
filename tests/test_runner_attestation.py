from dataclasses import asdict
from hashlib import sha256
import inspect
import os
from pathlib import Path
import tempfile
from time import time_ns
import unittest

from cogni_flow.approval import canonical_json_bytes, ed25519_backend_available
from cogni_flow.kernel_sandbox import KernelSandboxEvidence, LinuxOciSandboxRunner
from cogni_flow.production import (
    ProductionHarnessConfig,
    PromotionMode,
    command_sha256,
    verify_runner_attestation,
)
from cogni_flow.runner_attestation import (
    RUNNER_ATTESTATION_ASSURANCE,
    RUNNER_ATTESTATION_SCHEMA,
    RunnerAttestationImportError,
    RunnerAttestationImportLedger,
    RunnerAttestationReplayError,
    SignedRunnerAttestationV1,
    load_externally_attested_runner,
)


@unittest.skipUnless(ed25519_backend_available(), "Ed25519 backend unavailable")
class TestDetachedRunnerAttestation(unittest.TestCase):
    def setUp(self) -> None:
        from cryptography.hazmat.primitives import serialization
        from cryptography.hazmat.primitives.asymmetric.ed25519 import (
            Ed25519PrivateKey,
        )

        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.private = Ed25519PrivateKey.from_private_bytes(b"\x2a" * 32)
        self.public = self.private.public_key().public_bytes(
            serialization.Encoding.Raw,
            serialization.PublicFormat.Raw,
        )
        self.key = self.root / "attestor-public.key"
        self.key.write_bytes(self.public)
        self.key_sha256 = sha256(self.public).hexdigest()
        self.regression_command = ("python", "-m", "pytest", "-q")
        self.health_command = ("python", "-m", "pytest", "-q", "tests/smoke")
        self.commands = tuple(
            sorted(
                (
                    command_sha256(self.regression_command),
                    command_sha256(self.health_command),
                )
            )
        )
        runner = object.__new__(LinuxOciSandboxRunner)
        runner.evidence = KernelSandboxEvidence(
            runner_id="production-runner-01",
            engine_path="/usr/bin/docker",
            engine_sha256="3" * 64,
            daemon_socket="/run/docker.sock",
            image_reference=f"cogni-os@sha256:{'4' * 64}",
            allowed_command_sha256=self.commands,
            runtime="runc",
            non_root_uid=65534,
            max_memory_bytes=1024**3,
            max_pids=128,
            max_cpus=2.0,
            tmpfs_bytes=128 * 1024**2,
        )
        runner._evidence_sha256 = "5" * 64
        self.runner = runner
        source = Path(inspect.getsourcefile(LinuxOciSandboxRunner) or "")
        self.source_sha256 = sha256(source.read_bytes()).hexdigest()
        self.now = time_ns()

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _payload(self, **overrides):
        payload = {
            "schema": RUNNER_ATTESTATION_SCHEMA,
            "runner_id": self.runner.evidence.runner_id,
            "runner_evidence_sha256": self.runner._evidence_sha256,
            "runner_source_sha256": self.source_sha256,
            "engine_path": self.runner.evidence.engine_path,
            "engine_sha256": self.runner.evidence.engine_sha256,
            "daemon_socket": self.runner.evidence.daemon_socket,
            "image_reference": self.runner.evidence.image_reference,
            "runtime": "runc",
            "allowed_command_sha256": list(self.commands),
            "kernel_boundary": True,
            "network_isolated": True,
            "host_filesystem_isolated": True,
            "ephemeral_workspace": True,
            "production_attestation": True,
            "assurance": RUNNER_ATTESTATION_ASSURANCE,
            "attestor_id": "assessor.test",
            "public_key_sha256": self.key_sha256,
            "nonce": "runner_attestation_nonce_0123456789",
            "issued_ns": self.now - 1_000,
            "expires_ns": self.now + 10_000_000_000,
        }
        payload.update(overrides)
        payload["statement_id"] = sha256(canonical_json_bytes(payload)).hexdigest()
        return payload

    def _write(self, payload):
        raw = canonical_json_bytes(payload)
        statement = self.root / "runner-attestation.json"
        signature = self.root / "runner-attestation.sig"
        statement.write_bytes(raw)
        signature.write_bytes(self.private.sign(raw))
        return statement, signature

    def _load(self, payload, *, ledger=None, **kwargs):
        statement, signature = self._write(payload)
        return load_externally_attested_runner(
            self.runner,
            statement,
            signature,
            self.key,
            expected_public_key_sha256=self.key_sha256,
            attestor_ids=("assessor.test",),
            import_ledger=ledger
            or RunnerAttestationImportLedger(self.root / "imports"),
            now_ns=self.now,
            **kwargs,
        )

    def test_valid_detached_attestation_seals_exact_runner(self):
        adapter = self._load(self._payload())
        attestation = adapter.isolation_attestation()
        self.assertTrue(adapter.kernel_isolated)
        self.assertTrue(adapter.production_attestation)
        self.assertFalse(adapter.integration_smoke_only)
        self.assertEqual(attestation.runner_id, self.runner.evidence.runner_id)
        self.assertEqual(attestation.allowed_command_sha256, self.commands)
        self.assertEqual(
            attestation.evidence_sha256,
            sha256((self.root / "runner-attestation.json").read_bytes()).hexdigest(),
        )
        verified = verify_runner_attestation(
            adapter,
            ProductionHarnessConfig(
                promotion_mode=PromotionMode.ATTESTED,
                regression_command=self.regression_command,
                health_check_command=self.health_command,
                trusted_runner_evidence_sha256=(adapter.attestation_sha256,),
                trusted_runner_ids=(attestation.runner_id,),
            ),
        )
        self.assertEqual(verified.runner_id, attestation.runner_id)

    def test_signature_and_statement_tamper_fail_closed(self):
        payload = self._payload()
        statement, signature = self._write(payload)
        signature.write_bytes(b"x" * 64)
        with self.assertRaisesRegex(
            RunnerAttestationImportError, "signature is invalid"
        ):
            load_externally_attested_runner(
                self.runner,
                statement,
                signature,
                self.key,
                expected_public_key_sha256=self.key_sha256,
                attestor_ids=("assessor.test",),
                import_ledger=RunnerAttestationImportLedger(self.root / "imports"),
                now_ns=self.now,
            )
        payload["runner_id"] = "tampered-runner"
        statement.write_bytes(canonical_json_bytes(payload))
        with self.assertRaises(RunnerAttestationImportError):
            load_externally_attested_runner(
                self.runner,
                statement,
                signature,
                self.key,
                expected_public_key_sha256=self.key_sha256,
                attestor_ids=("assessor.test",),
                import_ledger=RunnerAttestationImportLedger(self.root / "imports-2"),
                now_ns=self.now,
            )

    def test_expiry_and_excessive_validity_fail_closed(self):
        with self.assertRaisesRegex(RunnerAttestationImportError, "currently valid"):
            self._load(
                self._payload(
                    issued_ns=self.now - 20_000,
                    expires_ns=self.now - 1,
                )
            )
        with self.assertRaisesRegex(RunnerAttestationImportError, "validity exceeds"):
            self._load(
                self._payload(
                    issued_ns=self.now - 1,
                    expires_ns=self.now + 20_000_000_000,
                ),
                max_validity_seconds=1,
            )

    def test_source_image_and_command_mismatch_fail_closed(self):
        cases = (
            ({"runner_source_sha256": "6" * 64}, "implementation source"),
            ({"engine_path": "/opt/untrusted/docker"}, "engine_path"),
            ({"image_reference": f"other@sha256:{'7' * 64}"}, "image_reference"),
            ({"allowed_command_sha256": ["8" * 64]}, "allowed_command_sha256"),
        )
        for index, (overrides, message) in enumerate(cases):
            with (
                self.subTest(field=message),
                self.assertRaisesRegex(RunnerAttestationImportError, message),
            ):
                self._load(
                    self._payload(**overrides),
                    ledger=RunnerAttestationImportLedger(
                        self.root / f"imports-{index}"
                    ),
                )

    def test_statement_and_nonce_replay_are_rejected(self):
        ledger = RunnerAttestationImportLedger(self.root / "imports")
        payload = self._payload()
        self._load(payload, ledger=ledger)
        with self.assertRaises(RunnerAttestationReplayError):
            self._load(payload, ledger=ledger)
        second = self._payload(expires_ns=self.now + 9_000_000_000)
        with self.assertRaisesRegex(RunnerAttestationReplayError, "nonce"):
            self._load(second, ledger=ledger)

    def test_symlink_and_oversize_inputs_are_rejected(self):
        payload = self._payload()
        statement, signature = self._write(payload)
        signature.write_bytes(b"x" * 65)
        with self.assertRaises(RunnerAttestationImportError):
            load_externally_attested_runner(
                self.runner,
                statement,
                signature,
                self.key,
                expected_public_key_sha256=self.key_sha256,
                attestor_ids=("assessor.test",),
                import_ledger=RunnerAttestationImportLedger(self.root / "imports"),
                now_ns=self.now,
            )
        if os.name != "nt":
            link = self.root / "statement-link.json"
            link.symlink_to(statement)
            signature.write_bytes(self.private.sign(statement.read_bytes()))
            with self.assertRaises(RunnerAttestationImportError):
                load_externally_attested_runner(
                    self.runner,
                    link,
                    signature,
                    self.key,
                    expected_public_key_sha256=self.key_sha256,
                    attestor_ids=("assessor.test",),
                    import_ledger=RunnerAttestationImportLedger(
                        self.root / "imports-link"
                    ),
                    now_ns=self.now,
                )

    def test_import_record_survives_restart_and_detects_tamper(self):
        ledger_path = self.root / "imports"
        payload = self._payload()
        self._load(payload, ledger=RunnerAttestationImportLedger(ledger_path))
        restarted = RunnerAttestationImportLedger(ledger_path)
        with self.assertRaises(RunnerAttestationReplayError):
            self._load(payload, ledger=restarted)
        record = next(ledger_path.glob("*.json"))
        parsed = SignedRunnerAttestationV1.from_mapping(payload)
        self.assertTrue(record.name.startswith("consumed-nonce-"))
        data = asdict(parsed)
        self.assertEqual(data["statement_id"], parsed.statement_id)
        record.write_text("{}", encoding="utf-8")
        with self.assertRaises(RunnerAttestationImportError):
            restarted.consume(
                parsed,
                attestation_sha256="9" * 64,
                imported_ns=self.now,
            )


if __name__ == "__main__":
    unittest.main()
