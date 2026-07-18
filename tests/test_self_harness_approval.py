from dataclasses import asdict
from hashlib import sha256
import tempfile
from pathlib import Path
from types import SimpleNamespace
from time import time_ns
import unittest
from unittest.mock import patch

from cogni_flow.approval import (
    APPROVAL_SCHEMA,
    ROLLBACK_AUTHORIZATION_SCHEMA,
    ApprovalError,
    ApprovalReplayError,
    CandidateEvaluationLedger,
    CandidateEvaluationV1,
    ConsumedApprovalLedger,
    Ed25519ApprovalVerifier,
    HumanApprovalV1,
    RollbackAuthorizationV1,
    ed25519_backend_available,
)


def _evaluation(now: int) -> CandidateEvaluationV1:
    return CandidateEvaluationV1.create(
        proposal_id="1" * 64,
        relative_path="cogni_flow/target.py",
        base_sha256="2" * 64,
        replacement_sha256="3" * 64,
        source_surface_sha256="4" * 64,
        snapshot_tree_sha256="8" * 64,
        runner_id="audited-runner",
        runner_evidence_sha256="5" * 64,
        regression_command_sha256="6" * 64,
        result_sha256="7" * 64,
        returncode=0,
        completed_ns=now,
        expires_ns=now + 10_000_000_000,
    )


def _approval(
    evaluation: CandidateEvaluationV1,
    public_key_sha256: str,
) -> HumanApprovalV1:
    return HumanApprovalV1.from_mapping(
        {
            "schema": APPROVAL_SCHEMA,
            "evaluation_id": evaluation.evaluation_id,
            "proposal_id": evaluation.proposal_id,
            "relative_path": evaluation.relative_path,
            "base_sha256": evaluation.base_sha256,
            "replacement_sha256": evaluation.replacement_sha256,
            "source_surface_sha256": evaluation.source_surface_sha256,
            "snapshot_tree_sha256": evaluation.snapshot_tree_sha256,
            "runner_id": evaluation.runner_id,
            "runner_evidence_sha256": evaluation.runner_evidence_sha256,
            "regression_command_sha256": evaluation.regression_command_sha256,
            "nonce": "approval_nonce_0123456789abcdef0",
            "approver_id": "operator.test",
            "issued_ns": evaluation.completed_ns,
            "expires_ns": evaluation.expires_ns,
            "decision": "approve_once",
            "public_key_sha256": public_key_sha256,
            "signature": "00" * 64,
        }
    )


def _rollback_authorization(
    *,
    nonce: str = "rollback_nonce_0123456789abcdef0",
) -> RollbackAuthorizationV1:
    now = time_ns()
    return RollbackAuthorizationV1.from_mapping(
        {
            "schema": ROLLBACK_AUTHORIZATION_SCHEMA,
            "journal_record_id": "a" * 32,
            "relative_path": "cogni_flow/target.py",
            "before_sha256": "1" * 64,
            "after_sha256": "2" * 64,
            "source_surface_sha256": "3" * 64,
            "runner_id": "audited-runner",
            "runner_evidence_sha256": "4" * 64,
            "health_command_sha256": "5" * 64,
            "nonce": nonce,
            "approver_id": "operator.test",
            "issued_ns": now,
            "expires_ns": now + 10_000_000_000,
            "decision": "rollback_committed_once",
            "public_key_sha256": "6" * 64,
            "signature": "00" * 64,
        }
    )


class TestSelfHarnessApproval(unittest.TestCase):
    def test_evaluation_ledger_rejects_tampered_immutable_record(self):
        with tempfile.TemporaryDirectory() as tmp:
            now = time_ns()
            evaluation = _evaluation(now)
            ledger = CandidateEvaluationLedger(Path(tmp) / "evaluations")
            path = ledger.record(evaluation)
            payload = asdict(evaluation)
            payload["base_sha256"] = "8" * 64
            import json

            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ApprovalError, "digest"):
                ledger.load(evaluation.evaluation_id)

    def test_approval_consumption_is_atomic_and_replay_safe(self):
        with tempfile.TemporaryDirectory() as tmp:
            evaluation = _evaluation(time_ns())
            key_digest = sha256(bytes(range(32))).hexdigest()
            approval = _approval(evaluation, key_digest)
            ledger = ConsumedApprovalLedger(Path(tmp) / "consumed")
            first = ledger.consume_once(approval, evaluation)
            self.assertEqual(first, approval.approval_id)
            with self.assertRaises(ApprovalReplayError):
                ledger.consume_once(approval, evaluation)

            resigned_payload = asdict(approval)
            resigned_payload["expires_ns"] -= 1
            resigned_payload["signature"] = "11" * 64
            resigned = HumanApprovalV1.from_mapping(resigned_payload)
            self.assertNotEqual(resigned.approval_id, approval.approval_id)
            with self.assertRaises(ApprovalReplayError):
                ledger.consume_once(resigned, evaluation)

    def test_rollback_nonce_is_consumed_in_global_mutation_namespace(self):
        with tempfile.TemporaryDirectory() as tmp:
            evaluation = _evaluation(time_ns())
            key_digest = sha256(bytes(range(32))).hexdigest()
            approval = _approval(evaluation, key_digest)
            authorization = _rollback_authorization(nonce=approval.nonce)
            ledger = ConsumedApprovalLedger(Path(tmp) / "consumed")

            ledger.consume_once(approval, evaluation)
            with self.assertRaises(ApprovalReplayError):
                ledger.consume_rollback_once(authorization)

    def test_rollback_nonce_cannot_be_resigned_for_a_second_authority(self):
        with tempfile.TemporaryDirectory() as tmp:
            authorization = _rollback_authorization()
            ledger = ConsumedApprovalLedger(Path(tmp) / "consumed")
            first = ledger.consume_rollback_once(authorization)
            self.assertEqual(first, authorization.authorization_id)

            resigned_payload = asdict(authorization)
            resigned_payload["expires_ns"] -= 1
            resigned_payload["signature"] = "11" * 64
            resigned = RollbackAuthorizationV1.from_mapping(resigned_payload)
            self.assertNotEqual(
                resigned.authorization_id,
                authorization.authorization_id,
            )
            with self.assertRaises(ApprovalReplayError):
                ledger.consume_rollback_once(resigned)

    def test_rollback_authorization_rejects_unversioned_or_unbound_shapes(self):
        authorization = _rollback_authorization()
        malformed = asdict(authorization)
        malformed["schema"] = "rollback"
        with self.assertRaisesRegex(ApprovalError, "schema"):
            RollbackAuthorizationV1.from_mapping(malformed)
        malformed = asdict(authorization)
        malformed.pop("source_surface_sha256")
        with self.assertRaisesRegex(ApprovalError, "fields"):
            RollbackAuthorizationV1.from_mapping(malformed)

    def test_missing_optional_cryptography_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            public = bytes(range(32))
            key = root / "operator.key"
            key.write_bytes(public)
            verifier = Ed25519ApprovalVerifier(
                key,
                expected_sha256=sha256(public).hexdigest(),
                approver_ids=("operator.test",),
            )
            evaluation = _evaluation(time_ns())
            approval = _approval(evaluation, verifier.public_key_sha256)
            real_import = __import__

            def guarded_import(name, *args, **kwargs):
                if name.startswith("cryptography"):
                    raise ImportError("injected optional dependency absence")
                return real_import(name, *args, **kwargs)

            with patch("builtins.__import__", side_effect=guarded_import):
                self.assertFalse(ed25519_backend_available())
                with self.assertRaisesRegex(ApprovalError, "unavailable"):
                    verifier.verify(
                        approval,
                        evaluation,
                        now_ns=evaluation.completed_ns + 1,
                    )

    def test_broken_binary_backend_fails_closed_during_key_construction(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            public = bytes(range(32))
            key = root / "operator.key"
            key.write_bytes(public)
            verifier = Ed25519ApprovalVerifier(
                key,
                expected_sha256=sha256(public).hexdigest(),
                approver_ids=("operator.test",),
            )
            evaluation = _evaluation(time_ns())
            approval = _approval(evaluation, verifier.public_key_sha256)
            real_import = __import__

            class FakeInvalidSignature(Exception):
                pass

            class BrokenPublicKey:
                @classmethod
                def from_public_bytes(cls, payload):
                    raise ModuleNotFoundError("No module named '_cffi_backend'")

            class BrokenPrivateKey:
                @classmethod
                def from_private_bytes(cls, payload):
                    raise ModuleNotFoundError("No module named '_cffi_backend'")

            def broken_backend_import(name, *args, **kwargs):
                if name == "cryptography.exceptions":
                    return SimpleNamespace(InvalidSignature=FakeInvalidSignature)
                if name == "cryptography.hazmat.primitives.asymmetric.ed25519":
                    return SimpleNamespace(
                        Ed25519PrivateKey=BrokenPrivateKey,
                        Ed25519PublicKey=BrokenPublicKey,
                    )
                return real_import(name, *args, **kwargs)

            with patch("builtins.__import__", side_effect=broken_backend_import):
                self.assertFalse(ed25519_backend_available())
                with self.assertRaisesRegex(ApprovalError, "unavailable") as caught:
                    verifier.verify(
                        approval,
                        evaluation,
                        now_ns=evaluation.completed_ns + 1,
                    )
            self.assertIsInstance(caught.exception.__cause__, ModuleNotFoundError)


if __name__ == "__main__":
    unittest.main()
