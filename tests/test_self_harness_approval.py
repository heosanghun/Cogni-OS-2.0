from dataclasses import asdict
from hashlib import sha256
import tempfile
from pathlib import Path
from time import time_ns
import unittest
from unittest.mock import patch

from cogni_flow.approval import (
    APPROVAL_SCHEMA,
    ApprovalError,
    ApprovalReplayError,
    CandidateEvaluationLedger,
    CandidateEvaluationV1,
    ConsumedApprovalLedger,
    Ed25519ApprovalVerifier,
    HumanApprovalV1,
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
                with self.assertRaisesRegex(ApprovalError, "unavailable"):
                    verifier.verify(
                        approval,
                        evaluation,
                        now_ns=evaluation.completed_ns + 1,
                    )


if __name__ == "__main__":
    unittest.main()
