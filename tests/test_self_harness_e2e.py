import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from cogni_flow.approval import ed25519_backend_available
from cogni_flow.production import ProductionHarnessConfig, PromotionMode
from cogni_flow.self_harness_e2e import (
    OperatorSelfHarnessE2E,
    SelfHarnessE2EError,
    SelfHarnessE2EEventV1,
    SelfHarnessE2ELedger,
    SelfHarnessE2EReplayError,
    validate_self_harness_e2e,
)
from tests.test_production_harness import (
    FakeAttestedRunner,
    FakeClock,
    ProductionHarnessFixture,
    _signed_approval,
    _signed_rollback,
    _signing_authority,
)
from scripts.validate_self_harness_e2e import main as validate_cli_main


@unittest.skipUnless(ed25519_backend_available(), "Ed25519 backend unavailable")
class TestOperatorSelfHarnessE2E(unittest.TestCase):
    @staticmethod
    def _build(root: Path, outcomes=(True, True, True)):
        fixture = ProductionHarnessFixture(root)
        clock = FakeClock()
        private, verifier = _signing_authority(root)
        config = ProductionHarnessConfig(
            idle_seconds=1,
            promotion_mode=PromotionMode.ATTESTED,
            regression_command=("python", "-m", "pytest", "-q"),
            health_check_command=("python", "-m", "pytest", "-q"),
            trusted_runner_evidence_sha256=("a" * 64,),
            trusted_runner_ids=("audited-test-runner",),
        )
        runner = FakeAttestedRunner(
            (config.regression_command, config.health_check_command),
            outcomes=outcomes,
        )
        service = fixture.build(
            config,
            clock=clock,
            runner=runner,
            approval_verifier=verifier,
        )
        return fixture, clock, private, verifier, service

    @staticmethod
    def _evaluation(fixture, service, clock):
        tick = fixture.submit_evidence_and_run(service, clock)
        if not tick.ran or not service.candidate_evaluations:
            raise AssertionError("fixture did not produce a candidate evaluation")
        return service.candidate_evaluations[-1]

    def test_full_signed_promotion_and_byte_identical_rollback_chain(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixture, clock, private, verifier, service = self._build(root)
            original = fixture.target.read_bytes()
            evidence_dir = root / ".operator-e2e"
            with service:
                evaluation = self._evaluation(fixture, service, clock)
                ledger = SelfHarnessE2ELedger(evidence_dir)
                operator = OperatorSelfHarnessE2E(service, ledger)
                prepared = operator.prepare(
                    evaluation.evaluation_id,
                    run_nonce="operator_e2e_nonce_0123456789abcd",
                )
                promoted = operator.promote(
                    prepared.run_id,
                    _signed_approval(private, verifier, evaluation),
                )
                record = service.journal.records()[-1]
                self.assertEqual(promoted.stage, "promotion_committed")
                self.assertNotEqual(fixture.target.read_bytes(), original)
                rolled_back = operator.rollback(
                    prepared.run_id,
                    _signed_rollback(private, verifier, service, record),
                )
                self.assertEqual(rolled_back.stage, "rollback_completed")
                self.assertEqual(fixture.target.read_bytes(), original)

            restarted = SelfHarnessE2ELedger(evidence_dir)
            result = validate_self_harness_e2e(restarted, prepared.run_id, verifier)
            self.assertTrue(result.full_e2e_complete)
            self.assertTrue(result.runner_attestation_digest_bound)
            self.assertFalse(result.production_attestation_reverified)
            self.assertEqual(result.event_count, 3)
            self.assertEqual(result.terminal_stage, "rollback_completed")
            with patch("builtins.print") as emit:
                exit_code = validate_cli_main(
                    [
                        "--evidence-dir",
                        str(evidence_dir),
                        "--run-id",
                        prepared.run_id,
                        "--approval-public-key",
                        str(root / "operator-ed25519-public.key"),
                        "--approval-public-key-sha256",
                        verifier.public_key_sha256,
                        "--approver-id",
                        "operator.test",
                    ]
                )
            self.assertEqual(exit_code, 0)
            self.assertIn('"full_e2e_complete": true', emit.call_args.args[0])

    def test_health_failure_records_verified_original_byte_restore(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixture, clock, private, verifier, service = self._build(
                root, outcomes=(True, False)
            )
            original = fixture.target.read_bytes()
            with service:
                evaluation = self._evaluation(fixture, service, clock)
                ledger = SelfHarnessE2ELedger(root / ".operator-e2e")
                operator = OperatorSelfHarnessE2E(service, ledger)
                prepared = operator.prepare(
                    evaluation.evaluation_id,
                    run_nonce="operator_e2e_health_nonce_012345678",
                )
                outcome = operator.promote(
                    prepared.run_id,
                    _signed_approval(private, verifier, evaluation),
                )
                self.assertEqual(outcome.stage, "promotion_health_restore")
                self.assertEqual(fixture.target.read_bytes(), original)
                result = validate_self_harness_e2e(ledger, prepared.run_id, verifier)
                self.assertFalse(result.full_e2e_complete)
                self.assertEqual(result.terminal_stage, "promotion_health_restore")

    def test_rollback_health_failure_reapplies_committed_bytes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixture, clock, private, verifier, service = self._build(
                root, outcomes=(True, True, False)
            )
            with service:
                evaluation = self._evaluation(fixture, service, clock)
                ledger = SelfHarnessE2ELedger(root / ".operator-e2e")
                operator = OperatorSelfHarnessE2E(service, ledger)
                prepared = operator.prepare(
                    evaluation.evaluation_id,
                    run_nonce="operator_e2e_rollback_nonce_01234567",
                )
                operator.promote(
                    prepared.run_id,
                    _signed_approval(private, verifier, evaluation),
                )
                record = service.journal.records()[-1]
                committed = fixture.target.read_bytes()
                outcome = operator.rollback(
                    prepared.run_id,
                    _signed_rollback(private, verifier, service, record),
                )
                self.assertEqual(outcome.stage, "rollback_health_restore")
                self.assertEqual(fixture.target.read_bytes(), committed)
                result = validate_self_harness_e2e(ledger, prepared.run_id, verifier)
                self.assertFalse(result.full_e2e_complete)
                self.assertEqual(result.terminal_stage, "rollback_health_restore")

    def test_nonce_replay_is_rejected_across_ledger_restart(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixture, clock, _, _, service = self._build(root)
            with service:
                evaluation = self._evaluation(fixture, service, clock)
                directory = root / ".operator-e2e"
                nonce = "operator_e2e_replay_nonce_012345678"
                OperatorSelfHarnessE2E(
                    service, SelfHarnessE2ELedger(directory)
                ).prepare(evaluation.evaluation_id, run_nonce=nonce)
                restarted = OperatorSelfHarnessE2E(
                    service, SelfHarnessE2ELedger(directory)
                )
                with self.assertRaises(SelfHarnessE2EReplayError):
                    restarted.prepare(evaluation.evaluation_id, run_nonce=nonce)

    def test_event_tamper_and_health_command_mismatch_are_detected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixture, clock, private, verifier, service = self._build(root)
            with service:
                evaluation = self._evaluation(fixture, service, clock)
                ledger = SelfHarnessE2ELedger(root / ".operator-e2e")
                operator = OperatorSelfHarnessE2E(service, ledger)
                prepared = operator.prepare(
                    evaluation.evaluation_id,
                    run_nonce="operator_e2e_tamper_nonce_012345678",
                )
                operator.promote(
                    prepared.run_id,
                    _signed_approval(private, verifier, evaluation),
                )
                record = service.journal.records()[-1]
                operator.rollback(
                    prepared.run_id,
                    _signed_rollback(private, verifier, service, record),
                )
                events = ledger.events(prepared.run_id)

            tampered_dir = root / ".tampered-e2e"
            tampered = SelfHarnessE2ELedger(tampered_dir)
            tampered.append(events[0])
            tampered.append(events[1])
            payload = dict(events[2].payload)
            payload["health_command_sha256"] = "f" * 64
            altered = SelfHarnessE2EEventV1.create(
                run_id=events[2].run_id,
                sequence=3,
                stage=events[2].stage,
                previous_event_sha256=events[2].previous_event_sha256,
                created_ns=events[2].created_ns,
                payload=payload,
            )
            tampered.append(altered)
            with self.assertRaises(Exception):
                validate_self_harness_e2e(tampered, prepared.run_id, verifier)

            first_path = next(
                path
                for path in tampered_dir.glob("*.json")
                if path.name.endswith("-01.json")
            )
            data = json.loads(first_path.read_text(encoding="utf-8"))
            data["payload"]["candidate_count"] = 99
            first_path.write_text(
                json.dumps(data, sort_keys=True, separators=(",", ":")),
                encoding="utf-8",
            )
            with self.assertRaises(SelfHarnessE2EError):
                SelfHarnessE2ELedger(tampered_dir).events(prepared.run_id)

    def test_proposal_only_mode_cannot_construct_operator_orchestrator(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixture = ProductionHarnessFixture(root)
            service = fixture.build(
                ProductionHarnessConfig(),
                clock=FakeClock(),
            )
            with self.assertRaisesRegex(SelfHarnessE2EError, "proposal-only"):
                OperatorSelfHarnessE2E(
                    service, SelfHarnessE2ELedger(root / ".operator-e2e")
                )


if __name__ == "__main__":
    unittest.main()
