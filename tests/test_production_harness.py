from dataclasses import asdict
from hashlib import sha256
import tempfile
from pathlib import Path
from time import time_ns
import unittest
from unittest.mock import patch

import torch

from cogni_flow.approval import (
    APPROVAL_SCHEMA,
    ROLLBACK_AUTHORIZATION_SCHEMA,
    ApprovalError,
    ApprovalReplayError,
    Ed25519ApprovalVerifier,
    HumanApprovalV1,
    RollbackAuthorizationV1,
    canonical_json_bytes,
    ed25519_backend_available,
)
from cogni_flow.harness import FailureTrace, SandboxResult
import cogni_flow.production as production_module
from cogni_flow.production import (
    BackupJournal,
    BoundedLogDB,
    IsolationAttestationError,
    JournalIntegrityError,
    ProductionHarnessConfig,
    PromotionMode,
    RunnerAttestation,
    build_production_self_harness,
    command_sha256,
)
from cogni_flow.proposals import ProposalOnlyError


class FakeClock:
    def __init__(self) -> None:
        self.value = 0.0

    def __call__(self) -> float:
        return self.value

    def advance(self, seconds: float) -> None:
        self.value += seconds


class FakeTokenizer:
    def __init__(self, decoded: str = "VALUE = 2") -> None:
        self.decoded = decoded

    def __call__(self, prompt, **kwargs):
        return {
            "input_ids": torch.tensor([[10, 11]]),
            "attention_mask": torch.tensor([[1, 1]]),
        }

    def decode(self, tokens, **kwargs):
        return self.decoded


class FakeModel:
    device = torch.device("cpu")

    def eval(self):
        return self

    def generate(self, **kwargs):
        suffix = torch.tensor([[99]], device=kwargs["input_ids"].device)
        return torch.cat((kwargs["input_ids"], suffix), dim=1)


class FakeAttestedRunner:
    kernel_isolated = True

    def __init__(self, commands, outcomes=(True, True)) -> None:
        self.commands = tuple(commands)
        self.outcomes = list(outcomes)
        self.calls = []

    def isolation_attestation(self):
        return RunnerAttestation(
            version=1,
            runner_id="audited-test-runner",
            evidence_sha256="a" * 64,
            kernel_boundary=True,
            network_isolated=True,
            host_filesystem_isolated=True,
            ephemeral_workspace=True,
            allowed_command_sha256=tuple(
                command_sha256(item) for item in self.commands
            ),
        )

    def run(self, project, command, timeout_seconds):
        source = (project / "cogni_flow" / "target.py").read_text(encoding="utf-8")
        self.calls.append((command, source, timeout_seconds))
        passed = self.outcomes.pop(0)
        return SandboxResult(passed, 0 if passed else 9, "ok" if passed else "bad")


class MarkerOnlyRunner:
    kernel_isolated = True

    def run(self, project, command, timeout_seconds):
        return SandboxResult(True, 0, "not attested")


class ProductionHarnessFixture:
    def __init__(self, root: Path) -> None:
        self.root = root
        (root / "cogni_flow").mkdir()
        (root / "tests").mkdir()
        self.target = root / "cogni_flow" / "target.py"
        self.target.write_text("VALUE = 1\n", encoding="utf-8")
        (root / "tests" / "test_target.py").write_text(
            "import unittest\n"
            "from cogni_flow.target import VALUE\n"
            "class T(unittest.TestCase):\n"
            "    def test_value(self): self.assertEqual(VALUE, 2)\n",
            encoding="utf-8",
        )
        self.signature = ("ValueError", "V1", "workflow")
        self.targets = {self.signature: "cogni_flow/target.py"}

    def build(
        self,
        config,
        *,
        clock,
        runner=None,
        checkpoint=None,
        approval_verifier=None,
    ):
        if (
            config.promotion_mode == PromotionMode.ATTESTED
            and approval_verifier is None
        ):
            key = self.root / "test-approval-public.key"
            key.write_bytes(bytes(range(32)))
            approval_verifier = Ed25519ApprovalVerifier(
                key,
                expected_sha256=sha256(key.read_bytes()).hexdigest(),
                approver_ids=("operator.test",),
            )
        # Native Windows remains fail-closed in product code.  This narrowly
        # scoped test seam exercises platform-independent transaction logic.
        with patch.object(production_module, "_require_kernel_promotion_platform"):
            return build_production_self_harness(
                self.root,
                FakeModel(),
                FakeTokenizer(),
                self.targets,
                checkpoint or (lambda: None),
                config=config,
                runner=runner,
                approval_verifier=approval_verifier,
                clock=clock,
            )

    @staticmethod
    def submit_and_run(service, clock):
        service.capture_exception(
            "wf-1", ValueError("wrong result"), verifier_code="V1"
        )
        service.failure_daemon.flush()
        clock.advance(service.config.idle_seconds)
        return service.tick()

    def submit_evidence_and_run(self, service, clock):
        service.capture_exception(
            "wf-1", ValueError("wrong result"), verifier_code="V1"
        )
        service.failure_daemon.flush()
        cluster = next(
            item
            for item in service.proposal_ledger.clusters()
            if item.signature == self.signature
        )
        for replacement in ("VALUE = 3", "VALUE = 4", "VALUE = 2"):
            service.proposer.proposer.tokenizer.decoded = replacement
            service.proposer(cluster)
        clock.advance(service.config.idle_seconds)
        return service.tick()


def _signing_authority(root: Path):
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

    private = Ed25519PrivateKey.from_private_bytes(b"\x19" * 32)
    public = private.public_key().public_bytes(
        serialization.Encoding.Raw,
        serialization.PublicFormat.Raw,
    )
    key = root / "operator-ed25519-public.key"
    key.write_bytes(public)
    verifier = Ed25519ApprovalVerifier(
        key,
        expected_sha256=sha256(public).hexdigest(),
        approver_ids=("operator.test",),
    )
    return private, verifier


def _signed_approval(private, verifier, evaluation, **overrides):
    issued = max(evaluation.completed_ns, time_ns() - 1)
    payload = {
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
        "issued_ns": issued,
        "expires_ns": evaluation.expires_ns,
        "decision": "approve_once",
        "public_key_sha256": verifier.public_key_sha256,
    }
    payload.update(overrides)
    payload["signature"] = private.sign(canonical_json_bytes(payload)).hex()
    return HumanApprovalV1.from_mapping(payload)


def _signed_rollback(private, verifier, service, record, **overrides):
    patcher = service.harness.patcher
    issued = max(record.created_ns, time_ns() - 1)
    payload = {
        "schema": ROLLBACK_AUTHORIZATION_SCHEMA,
        "journal_record_id": record.record_id,
        "relative_path": record.relative_path,
        "before_sha256": record.before_sha256,
        "after_sha256": record.after_sha256,
        "source_surface_sha256": service._source_surface_digest(),
        "runner_id": patcher.sandbox.runner_id,
        "runner_evidence_sha256": patcher.sandbox.evidence_sha256,
        "health_command_sha256": command_sha256(service.config.health_check_command),
        "nonce": "rollback_nonce_0123456789abcdef0",
        "approver_id": "operator.test",
        "issued_ns": issued,
        "expires_ns": issued + 10_000_000_000,
        "decision": "rollback_committed_once",
        "public_key_sha256": verifier.public_key_sha256,
    }
    payload.update(overrides)
    payload["signature"] = private.sign(canonical_json_bytes(payload)).hex()
    return RollbackAuthorizationV1.from_mapping(payload)


def _promote_for_rollback(fixture, service, clock, private, verifier):
    tick = fixture.submit_evidence_and_run(service, clock)
    evaluation = service.candidate_evaluations[0]
    approval = _signed_approval(private, verifier, evaluation)
    result = service.promote_approved_once(tick.result.evaluation_id, approval)
    if not result.promoted:
        raise AssertionError("rollback fixture promotion did not commit")
    return service.journal.records()[0]


class TestProductionSelfHarness(unittest.TestCase):
    def test_default_mode_generates_bounded_proposal_without_mutating_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            fixture = ProductionHarnessFixture(Path(tmp))
            clock = FakeClock()
            config = ProductionHarnessConfig(
                idle_seconds=5,
                max_pending_proposals=2,
            )
            service = fixture.build(config, clock=clock)

            with service:
                tick = fixture.submit_and_run(service, clock)
                self.assertTrue(tick.ran)
                self.assertTrue(tick.result.proposal_only)
                self.assertIn("attestation", tick.result.blocked_reason)
                self.assertFalse(service.status.promotion_enabled)
                self.assertEqual(len(service.pending_proposals), 0)

            self.assertEqual(fixture.target.read_text(encoding="utf-8"), "VALUE = 1\n")
            self.assertEqual(len(service.proposal_ledger.failures), 1)
            self.assertEqual(len(service.proposal_ledger.successes), 1)
            self.assertEqual(service.status.evidence_capture_ratio, 1.0)
            self.assertEqual(service.status.rich_pending_proposals, 0)
            self.assertEqual(
                len(
                    tuple(
                        service.proposal_ledger.state_directory.glob("failure-*.json")
                    )
                ),
                1,
            )
            self.assertEqual(
                len(
                    tuple(
                        service.proposal_ledger.state_directory.glob("success-*.json")
                    )
                ),
                1,
            )
            candidates = [
                event
                for event in service.logdb.audit_events()
                if event.kind == "candidate"
            ]
            self.assertEqual(candidates[-1].detail, "proposal_only")

    def test_three_distinct_model_drafts_become_evidence_linked_review_records(self):
        with tempfile.TemporaryDirectory() as tmp:
            fixture = ProductionHarnessFixture(Path(tmp))
            clock = FakeClock()
            service = fixture.build(
                ProductionHarnessConfig(idle_seconds=1),
                clock=clock,
            )
            with service:
                service.capture_exception(
                    "wf-1", ValueError("wrong result"), verifier_code="V1"
                )
                service.failure_daemon.flush()
                cluster = next(
                    item
                    for item in service.proposal_ledger.clusters()
                    if item.signature == fixture.signature
                )
                for replacement in ("VALUE = 2", "VALUE = 3", "VALUE = 4"):
                    service.proposer.proposer.tokenizer.decoded = replacement
                    service.proposer(cluster)

            self.assertEqual(len(service.evidence_proposals), 3)
            self.assertEqual(service.status.rich_pending_proposals, 3)
            self.assertEqual(fixture.target.read_text(encoding="utf-8"), "VALUE = 1\n")
            for proposal in service.evidence_proposals:
                self.assertFalse(proposal.source_mutation_allowed)
                self.assertTrue(proposal.primary_evidence_sha256)
            rejected = service.evidence_proposals[0]
            negative = service.reject_evidence_proposal(
                rejected.proposal_id,
                reason_code="held_out_regression",
                evidence_sha256="d" * 64,
            )
            self.assertEqual(negative.proposal_id, rejected.proposal_id)
            self.assertEqual(len(service.evidence_proposals), 2)
            self.assertEqual(len(service.pending_proposals), 2)
            self.assertEqual(service.status.negative_proposals, 1)
            service.proposer.proposer.tokenizer.decoded = "VALUE = 2"
            service.proposer(cluster)
            self.assertEqual(len(service.evidence_proposals), 2)
            self.assertEqual(len(service.pending_proposals), 2)

    def test_restart_restores_review_queue_and_negative_suppression(self):
        with tempfile.TemporaryDirectory() as tmp:
            fixture = ProductionHarnessFixture(Path(tmp))
            config = ProductionHarnessConfig(idle_seconds=1)
            first = fixture.build(config, clock=FakeClock())
            with first:
                first.capture_exception(
                    "wf-1", ValueError("wrong result"), verifier_code="V1"
                )
                first.failure_daemon.flush()
                cluster = next(
                    item
                    for item in first.proposal_ledger.clusters()
                    if item.signature == fixture.signature
                )
                for replacement in ("VALUE = 2", "VALUE = 3", "VALUE = 4"):
                    first.proposer.proposer.tokenizer.decoded = replacement
                    first.proposer(cluster)
                rejected = first.evidence_proposals[0]
                first.reject_evidence_proposal(
                    rejected.proposal_id,
                    reason_code="held_out_regression",
                    evidence_sha256="d" * 64,
                )

            restarted = fixture.build(config, clock=FakeClock())
            self.assertEqual(len(restarted.proposal_ledger.failures), 1)
            self.assertEqual(restarted.status.evidence_capture_ratio, 1.0)
            self.assertEqual(restarted.status.negative_proposals, 1)
            self.assertEqual(len(restarted.evidence_proposals), 2)
            self.assertEqual(restarted.status.pending_proposals, 2)
            self.assertEqual(
                {item.replacement for item in restarted.pending_proposals},
                {"VALUE = 3\n", "VALUE = 4\n"},
            )
            self.assertNotIn(
                rejected.proposal_id,
                {item.proposal_id for item in restarted.evidence_proposals},
            )
            cluster = next(
                item
                for item in restarted.proposal_ledger.clusters()
                if item.signature == fixture.signature
            )
            # Rejected and already-pending replacements must not count toward
            # the next K=3 bank or recreate an existing evidence file.
            for replacement in (
                "VALUE = 2",
                "VALUE = 3",
                "VALUE = 4",
                "VALUE = 5",
                "VALUE = 6",
            ):
                restarted.proposer.proposer.tokenizer.decoded = replacement
                restarted.proposer(cluster)
            self.assertEqual(len(restarted.evidence_proposals), 2)
            self.assertEqual(len(restarted.pending_proposals), 2)
            restarted.proposer.proposer.tokenizer.decoded = "VALUE = 7"
            restarted.proposer(cluster)
            self.assertEqual(len(restarted.evidence_proposals), 5)
            self.assertEqual(len(restarted.pending_proposals), 5)

    def test_restart_excludes_tampered_blob_from_reviewable_pending(self):
        with tempfile.TemporaryDirectory() as tmp:
            fixture = ProductionHarnessFixture(Path(tmp))
            config = ProductionHarnessConfig(idle_seconds=1)
            first = fixture.build(config, clock=FakeClock())
            first.proposal_ledger.record_failure(
                terminal_verifier_cause="ValueError",
                causal_status="V1",
                agent_mechanism="workflow",
                primary_evidence_sha256="a" * 64,
                source_sha256=sha256(fixture.target.read_bytes()).hexdigest(),
                reproduction="pytest target",
                observed_ns=1,
            )
            cluster = first.proposal_ledger.clusters()[0]
            for replacement in ("VALUE = 2", "VALUE = 3", "VALUE = 4"):
                first.proposer.proposer.tokenizer.decoded = replacement
                first.proposer(cluster)
            proposal = first.evidence_proposals[0]
            blob = (
                first.proposal_ledger.replacement_blob_directory
                / f"replacement-{proposal.replacement_sha256}.utf8"
            )
            blob.write_bytes(b"VALUE = 999")

            restarted = fixture.build(config, clock=FakeClock())
            self.assertEqual(len(restarted.proposal_ledger.failures), 1)
            self.assertEqual(len(restarted.proposal_ledger.proposals), 3)
            self.assertEqual(restarted.status.pending_proposals, 2)
            self.assertEqual(restarted.status.rich_pending_proposals, 2)
            self.assertEqual(restarted.status.unreviewable_proposals, 1)
            self.assertEqual(
                restarted.status.proposal_integrity_errors[0][0],
                proposal.proposal_id,
            )
            self.assertNotIn(
                proposal.proposal_id,
                {item.proposal_id for item in restarted.evidence_proposals},
            )

    def test_restart_fails_closed_when_persisted_proposal_is_tampered(self):
        with tempfile.TemporaryDirectory() as tmp:
            fixture = ProductionHarnessFixture(Path(tmp))
            config = ProductionHarnessConfig(idle_seconds=1)
            first = fixture.build(config, clock=FakeClock())
            first.proposal_ledger.record_failure(
                terminal_verifier_cause="ValueError",
                causal_status="V1",
                agent_mechanism="workflow",
                primary_evidence_sha256="a" * 64,
                source_sha256=sha256(fixture.target.read_bytes()).hexdigest(),
                reproduction="pytest target",
                observed_ns=1,
            )
            cluster = first.proposal_ledger.clusters()[0]
            for replacement in ("VALUE = 2", "VALUE = 3", "VALUE = 4"):
                first.proposer.proposer.tokenizer.decoded = replacement
                first.proposer(cluster)
            proposal = first.evidence_proposals[0]
            proposal_path = (
                first.proposal_ledger.state_directory
                / f"proposal-{proposal.proposal_id}.json"
            )
            proposal_path.write_text("{}", encoding="ascii")

            with self.assertRaises(ProposalOnlyError):
                fixture.build(config, clock=FakeClock())

    def test_restart_fails_closed_instead_of_truncating_pending_queue(self):
        with tempfile.TemporaryDirectory() as tmp:
            fixture = ProductionHarnessFixture(Path(tmp))
            first = fixture.build(
                ProductionHarnessConfig(max_pending_proposals=8),
                clock=FakeClock(),
            )
            first.proposal_ledger.record_failure(
                terminal_verifier_cause="ValueError",
                causal_status="V1",
                agent_mechanism="workflow",
                primary_evidence_sha256="a" * 64,
                source_sha256=sha256(fixture.target.read_bytes()).hexdigest(),
                reproduction="pytest target",
                observed_ns=1,
            )
            cluster = first.proposal_ledger.clusters()[0]
            for replacement in ("VALUE = 2", "VALUE = 3", "VALUE = 4"):
                first.proposer.proposer.tokenizer.decoded = replacement
                first.proposer(cluster)
            self.assertEqual(len(first.evidence_proposals), 3)

            with self.assertRaisesRegex(ProposalOnlyError, "configured capacity"):
                fixture.build(
                    ProductionHarnessConfig(max_pending_proposals=2),
                    clock=FakeClock(),
                )

    def test_proposal_only_cycle_enters_safe_mode_if_checkpoint_mutates_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            fixture = ProductionHarnessFixture(Path(tmp))
            clock = FakeClock()

            def corrupt_checkpoint() -> None:
                fixture.target.write_text("VALUE = 99\n", encoding="utf-8")

            service = fixture.build(
                ProductionHarnessConfig(idle_seconds=1),
                clock=clock,
                checkpoint=corrupt_checkpoint,
            )
            with service:
                service.capture_exception(
                    "wf-1", ValueError("wrong result"), verifier_code="V1"
                )
                service.failure_daemon.flush()
                clock.advance(1)
                with self.assertRaisesRegex(
                    JournalIntegrityError, "proposal-only cycle changed"
                ):
                    service.tick()
            self.assertEqual(service.rhythm.mode.value, "safe_mode")

    def test_attested_mode_rejects_marker_only_or_untrusted_runners(self):
        with tempfile.TemporaryDirectory() as tmp:
            fixture = ProductionHarnessFixture(Path(tmp))
            config = ProductionHarnessConfig(
                promotion_mode=PromotionMode.ATTESTED,
                regression_command=("regression",),
                health_check_command=("health",),
                trusted_runner_evidence_sha256=("a" * 64,),
            )
            with self.assertRaisesRegex(IsolationAttestationError, "attestation"):
                fixture.build(config, clock=FakeClock(), runner=MarkerOnlyRunner())

            runner = FakeAttestedRunner(
                (config.regression_command, config.health_check_command)
            )
            untrusted = ProductionHarnessConfig(
                promotion_mode=PromotionMode.ATTESTED,
                regression_command=("regression",),
                health_check_command=("health",),
            )
            with self.assertRaisesRegex(
                IsolationAttestationError, "explicitly trusted"
            ):
                fixture.build(untrusted, clock=FakeClock(), runner=runner)

    def test_attested_mode_requires_a_pinned_external_approval_verifier(self):
        with tempfile.TemporaryDirectory() as tmp:
            fixture = ProductionHarnessFixture(Path(tmp))
            config = ProductionHarnessConfig(
                promotion_mode=PromotionMode.ATTESTED,
                regression_command=("regression",),
                health_check_command=("health",),
                trusted_runner_evidence_sha256=("a" * 64,),
            )
            runner = FakeAttestedRunner(
                (config.regression_command, config.health_check_command)
            )
            with patch.object(production_module, "_require_kernel_promotion_platform"):
                with self.assertRaisesRegex(ApprovalError, "approval verifier"):
                    build_production_self_harness(
                        fixture.root,
                        FakeModel(),
                        FakeTokenizer(),
                        fixture.targets,
                        lambda: None,
                        config=config,
                        runner=runner,
                        clock=FakeClock(),
                    )

    @unittest.skipUnless(production_module.os.name == "nt", "native Windows only")
    def test_native_windows_attested_mode_fails_closed_before_state_creation(self):
        with tempfile.TemporaryDirectory() as tmp:
            fixture = ProductionHarnessFixture(Path(tmp))
            config = ProductionHarnessConfig(
                promotion_mode=PromotionMode.ATTESTED,
                regression_command=("regression",),
                health_check_command=("health",),
                trusted_runner_evidence_sha256=("a" * 64,),
            )
            runner = FakeAttestedRunner(
                (config.regression_command, config.health_check_command)
            )
            with self.assertRaisesRegex(IsolationAttestationError, "native Windows"):
                build_production_self_harness(
                    fixture.root,
                    FakeModel(),
                    FakeTokenizer(),
                    fixture.targets,
                    lambda: None,
                    config=config,
                    runner=runner,
                    clock=FakeClock(),
                )
            self.assertFalse((fixture.root / ".cogni_state").exists())

    def test_attested_regression_only_creates_awaiting_approval_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            fixture = ProductionHarnessFixture(Path(tmp))
            clock = FakeClock()
            config = ProductionHarnessConfig(
                idle_seconds=1,
                promotion_mode=PromotionMode.ATTESTED,
                regression_command=("regression",),
                health_check_command=("health",),
                trusted_runner_evidence_sha256=("a" * 64,),
                trusted_runner_ids=("audited-test-runner",),
            )
            runner = FakeAttestedRunner(
                (config.regression_command, config.health_check_command)
            )
            service = fixture.build(config, clock=clock, runner=runner)

            with service:
                tick = fixture.submit_evidence_and_run(service, clock)

            self.assertFalse(tick.result.promoted)
            self.assertTrue(tick.result.awaiting_approval)
            self.assertFalse(service.status.promotion_enabled)
            self.assertIn("one-time approval", service.status.blocked_reason)
            self.assertEqual(len(service.candidate_evaluations), 1)
            self.assertEqual(
                tick.result.evaluation_id,
                service.candidate_evaluations[0].evaluation_id,
            )
            self.assertEqual(fixture.target.read_text(encoding="utf-8"), "VALUE = 1\n")
            self.assertEqual(
                [item[0] for item in runner.calls],
                [config.regression_command],
            )
            self.assertIn("VALUE = 2", runner.calls[0][1])
            self.assertFalse(service.journal.records())

    @unittest.skipUnless(
        ed25519_backend_available(),
        "optional Ed25519 backend is not functional",
    )
    def test_valid_external_approval_promotes_exactly_once(self):
        with tempfile.TemporaryDirectory() as tmp:
            fixture = ProductionHarnessFixture(Path(tmp))
            private, verifier = _signing_authority(fixture.root)
            clock = FakeClock()
            config = ProductionHarnessConfig(
                idle_seconds=1,
                promotion_mode=PromotionMode.ATTESTED,
                regression_command=("regression",),
                health_check_command=("health",),
                trusted_runner_evidence_sha256=("a" * 64,),
            )
            runner = FakeAttestedRunner(
                (config.regression_command, config.health_check_command)
            )
            service = fixture.build(
                config,
                clock=clock,
                runner=runner,
                approval_verifier=verifier,
            )

            with service:
                tick = fixture.submit_evidence_and_run(service, clock)
                evaluation = service.candidate_evaluations[0]
                approval = _signed_approval(private, verifier, evaluation)
                result = service.promote_approved_once(
                    tick.result.evaluation_id, approval
                )

                self.assertTrue(result.promoted)
                self.assertEqual(
                    fixture.target.read_text(encoding="utf-8"), "VALUE = 2\n"
                )
                self.assertEqual(service.journal.records()[0].status, "committed")
                with self.assertRaises(ApprovalReplayError):
                    service.harness.patcher.consumed_approvals.consume_once(
                        approval, evaluation
                    )

            self.assertEqual(
                [item[0] for item in runner.calls],
                [config.regression_command, config.health_check_command],
            )

    @unittest.skipUnless(
        ed25519_backend_available(),
        "optional Ed25519 backend is not functional",
    )
    def test_failed_post_promotion_health_restores_verified_backup(self):
        with tempfile.TemporaryDirectory() as tmp:
            fixture = ProductionHarnessFixture(Path(tmp))
            private, verifier = _signing_authority(fixture.root)
            clock = FakeClock()
            config = ProductionHarnessConfig(
                idle_seconds=1,
                promotion_mode=PromotionMode.ATTESTED,
                regression_command=("regression",),
                health_check_command=("health",),
                trusted_runner_evidence_sha256=("a" * 64,),
            )
            runner = FakeAttestedRunner(
                (config.regression_command, config.health_check_command),
                outcomes=(True, False),
            )
            service = fixture.build(
                config,
                clock=clock,
                runner=runner,
                approval_verifier=verifier,
            )

            with service:
                tick = fixture.submit_evidence_and_run(service, clock)
                evaluation = service.candidate_evaluations[0]
                approval = _signed_approval(private, verifier, evaluation)
                result = service.promote_approved_once(
                    tick.result.evaluation_id, approval
                )

            self.assertFalse(result.promoted)
            self.assertEqual(fixture.target.read_text(encoding="utf-8"), "VALUE = 1\n")
            self.assertEqual(service.journal.records()[0].status, "rolled_back")
            self.assertEqual(service.rhythm.mode.value, "inference")

    @unittest.skipUnless(
        ed25519_backend_available(),
        "optional Ed25519 backend is not functional",
    )
    def test_forged_and_expired_approvals_never_mutate_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            fixture = ProductionHarnessFixture(Path(tmp))
            private, verifier = _signing_authority(fixture.root)
            clock = FakeClock()
            config = ProductionHarnessConfig(
                idle_seconds=1,
                promotion_mode=PromotionMode.ATTESTED,
                regression_command=("regression",),
                health_check_command=("health",),
                trusted_runner_evidence_sha256=("a" * 64,),
            )
            runner = FakeAttestedRunner(
                (config.regression_command, config.health_check_command)
            )
            service = fixture.build(
                config,
                clock=clock,
                runner=runner,
                approval_verifier=verifier,
            )
            with service:
                tick = fixture.submit_evidence_and_run(service, clock)
                evaluation = service.candidate_evaluations[0]
                valid = _signed_approval(private, verifier, evaluation)
                forged_payload = asdict(valid)
                forged_payload["signature"] = "00" * 64
                forged = HumanApprovalV1.from_mapping(forged_payload)
                with self.assertRaisesRegex(ApprovalError, "signature"):
                    service.promote_approved_once(tick.result.evaluation_id, forged)
                expired = _signed_approval(
                    private,
                    verifier,
                    evaluation,
                    issued_ns=evaluation.completed_ns,
                    expires_ns=evaluation.completed_ns + 1,
                )
                with self.assertRaisesRegex(ApprovalError, "currently valid"):
                    service.promote_approved_once(tick.result.evaluation_id, expired)
            self.assertEqual(fixture.target.read_text(encoding="utf-8"), "VALUE = 1\n")
            self.assertFalse(service.journal.records())

    @unittest.skipUnless(
        ed25519_backend_available(),
        "optional Ed25519 backend is not functional",
    )
    def test_source_change_after_evaluation_invalidates_approval(self):
        with tempfile.TemporaryDirectory() as tmp:
            fixture = ProductionHarnessFixture(Path(tmp))
            private, verifier = _signing_authority(fixture.root)
            clock = FakeClock()
            config = ProductionHarnessConfig(
                idle_seconds=1,
                promotion_mode=PromotionMode.ATTESTED,
                regression_command=("regression",),
                health_check_command=("health",),
                trusted_runner_evidence_sha256=("a" * 64,),
            )
            runner = FakeAttestedRunner(
                (config.regression_command, config.health_check_command)
            )
            service = fixture.build(
                config,
                clock=clock,
                runner=runner,
                approval_verifier=verifier,
            )
            with service:
                tick = fixture.submit_evidence_and_run(service, clock)
                evaluation = service.candidate_evaluations[0]
                approval = _signed_approval(private, verifier, evaluation)
                fixture.target.write_text("VALUE = 99\n", encoding="utf-8")
                with self.assertRaisesRegex(ApprovalError, "source surface changed"):
                    service.promote_approved_once(tick.result.evaluation_id, approval)
            self.assertEqual(fixture.target.read_text(encoding="utf-8"), "VALUE = 99\n")
            self.assertFalse(service.journal.records())

    @unittest.skipUnless(
        ed25519_backend_available(),
        "optional Ed25519 backend is not functional",
    )
    def test_signed_operator_rollback_restores_committed_backup_once(self):
        with tempfile.TemporaryDirectory() as tmp:
            fixture = ProductionHarnessFixture(Path(tmp))
            private, verifier = _signing_authority(fixture.root)
            checkpoints = []
            clock = FakeClock()
            config = ProductionHarnessConfig(
                idle_seconds=1,
                promotion_mode=PromotionMode.ATTESTED,
                regression_command=("regression",),
                health_check_command=("health",),
                trusted_runner_evidence_sha256=("a" * 64,),
            )
            runner = FakeAttestedRunner(
                (config.regression_command, config.health_check_command),
                outcomes=(True, True, True),
            )
            service = fixture.build(
                config,
                clock=clock,
                runner=runner,
                checkpoint=lambda: checkpoints.append("checkpoint"),
                approval_verifier=verifier,
            )

            with service:
                record = _promote_for_rollback(
                    fixture, service, clock, private, verifier
                )
                authorization = _signed_rollback(private, verifier, service, record)
                result = service.rollback_committed_once(
                    record.record_id,
                    authorization,
                )

                self.assertTrue(result.rolled_back)
                self.assertEqual(
                    fixture.target.read_text(encoding="utf-8"), "VALUE = 1\n"
                )
                self.assertEqual(
                    service.journal.records()[0].status,
                    "operator_rolled_back",
                )
                self.assertEqual(service.rhythm.mode.value, "inference")
                self.assertEqual(len(checkpoints), 3)
                rollback_audit = [
                    event
                    for event in service.logdb.audit_events()
                    if event.kind == "operator_rollback"
                ]
                self.assertEqual(len(rollback_audit), 1)
                self.assertEqual(rollback_audit[0].detail, "operator_rolled_back")

            self.assertEqual(
                [item[0] for item in runner.calls],
                [
                    config.regression_command,
                    config.health_check_command,
                    config.health_check_command,
                ],
            )

    @unittest.skipUnless(
        ed25519_backend_available(),
        "optional Ed25519 backend is not functional",
    )
    def test_operator_rollback_health_failure_reapplies_exact_commit(self):
        with tempfile.TemporaryDirectory() as tmp:
            fixture = ProductionHarnessFixture(Path(tmp))
            private, verifier = _signing_authority(fixture.root)
            clock = FakeClock()
            config = ProductionHarnessConfig(
                idle_seconds=1,
                promotion_mode=PromotionMode.ATTESTED,
                regression_command=("regression",),
                health_check_command=("health",),
                trusted_runner_evidence_sha256=("a" * 64,),
            )
            runner = FakeAttestedRunner(
                (config.regression_command, config.health_check_command),
                outcomes=(True, True, False),
            )
            service = fixture.build(
                config,
                clock=clock,
                runner=runner,
                approval_verifier=verifier,
            )

            with service:
                record = _promote_for_rollback(
                    fixture, service, clock, private, verifier
                )
                committed_bytes = fixture.target.read_bytes()
                authorization = _signed_rollback(private, verifier, service, record)
                result = service.rollback_committed_once(
                    record.record_id,
                    authorization,
                )

                self.assertFalse(result.rolled_back)
                self.assertEqual(fixture.target.read_bytes(), committed_bytes)
                self.assertEqual(service.journal.records()[0].status, "committed")
                self.assertEqual(service.rhythm.mode.value, "inference")

    @unittest.skipUnless(
        ed25519_backend_available(),
        "optional Ed25519 backend is not functional",
    )
    def test_operator_rollback_rejects_forged_expired_and_stale_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            fixture = ProductionHarnessFixture(Path(tmp))
            private, verifier = _signing_authority(fixture.root)
            clock = FakeClock()
            config = ProductionHarnessConfig(
                idle_seconds=1,
                promotion_mode=PromotionMode.ATTESTED,
                regression_command=("regression",),
                health_check_command=("health",),
                trusted_runner_evidence_sha256=("a" * 64,),
            )
            runner = FakeAttestedRunner(
                (config.regression_command, config.health_check_command),
                outcomes=(True, True),
            )
            service = fixture.build(
                config,
                clock=clock,
                runner=runner,
                approval_verifier=verifier,
            )

            with service:
                record = _promote_for_rollback(
                    fixture, service, clock, private, verifier
                )
                valid = _signed_rollback(private, verifier, service, record)
                forged_payload = asdict(valid)
                forged_payload["signature"] = "00" * 64
                forged = RollbackAuthorizationV1.from_mapping(forged_payload)
                with self.assertRaisesRegex(ApprovalError, "signature"):
                    service.rollback_committed_once(record.record_id, forged)
                expired = _signed_rollback(
                    private,
                    verifier,
                    service,
                    record,
                    issued_ns=record.created_ns,
                    expires_ns=record.created_ns + 1,
                    nonce="rollback_expired_0123456789abcdef0",
                )
                with self.assertRaisesRegex(ApprovalError, "currently valid"):
                    service.rollback_committed_once(record.record_id, expired)
                stale = _signed_rollback(
                    private,
                    verifier,
                    service,
                    record,
                    nonce="rollback_stale_0123456789abcdef000",
                )
                (fixture.root / "cogni_flow" / "unrelated.py").write_text(
                    "UNCHANGED = False\n", encoding="utf-8"
                )
                with self.assertRaisesRegex(ApprovalError, "source_surface"):
                    service.rollback_committed_once(record.record_id, stale)

            self.assertEqual(fixture.target.read_text(encoding="utf-8"), "VALUE = 2\n")
            self.assertEqual(service.journal.records()[0].status, "committed")

    @unittest.skipUnless(
        ed25519_backend_available(),
        "optional Ed25519 backend is not functional",
    )
    def test_operator_rollback_refuses_live_bytes_outside_commit(self):
        with tempfile.TemporaryDirectory() as tmp:
            fixture = ProductionHarnessFixture(Path(tmp))
            private, verifier = _signing_authority(fixture.root)
            clock = FakeClock()
            config = ProductionHarnessConfig(
                idle_seconds=1,
                promotion_mode=PromotionMode.ATTESTED,
                regression_command=("regression",),
                health_check_command=("health",),
                trusted_runner_evidence_sha256=("a" * 64,),
            )
            runner = FakeAttestedRunner(
                (config.regression_command, config.health_check_command),
                outcomes=(True, True),
            )
            service = fixture.build(
                config,
                clock=clock,
                runner=runner,
                approval_verifier=verifier,
            )

            with service:
                record = _promote_for_rollback(
                    fixture, service, clock, private, verifier
                )
                authorization = _signed_rollback(private, verifier, service, record)
                fixture.target.write_text("VALUE = 99\n", encoding="utf-8")
                with self.assertRaisesRegex(
                    JournalIntegrityError, "committed after digest"
                ):
                    service.rollback_committed_once(
                        record.record_id,
                        authorization,
                    )

            self.assertEqual(fixture.target.read_text(encoding="utf-8"), "VALUE = 99\n")
            self.assertEqual(service.journal.records()[0].status, "committed")

    @unittest.skipUnless(
        ed25519_backend_available(),
        "optional Ed25519 backend is not functional",
    )
    def test_operator_rollback_write_fault_preserves_commit_and_burns_nonce(self):
        with tempfile.TemporaryDirectory() as tmp:
            fixture = ProductionHarnessFixture(Path(tmp))
            private, verifier = _signing_authority(fixture.root)
            clock = FakeClock()
            config = ProductionHarnessConfig(
                idle_seconds=1,
                promotion_mode=PromotionMode.ATTESTED,
                regression_command=("regression",),
                health_check_command=("health",),
                trusted_runner_evidence_sha256=("a" * 64,),
            )
            runner = FakeAttestedRunner(
                (config.regression_command, config.health_check_command),
                outcomes=(True, True),
            )
            service = fixture.build(
                config,
                clock=clock,
                runner=runner,
                approval_verifier=verifier,
            )

            with service:
                record = _promote_for_rollback(
                    fixture, service, clock, private, verifier
                )
                authorization = _signed_rollback(private, verifier, service, record)
                real_atomic_write = production_module._atomic_write_bytes
                injected = False

                def fail_operator_restore(target, payload, **kwargs):
                    nonlocal injected
                    if not injected:
                        injected = True
                        raise OSError("injected atomic restore failure")
                    return real_atomic_write(target, payload, **kwargs)

                with patch.object(
                    production_module,
                    "_atomic_write_bytes",
                    side_effect=fail_operator_restore,
                ):
                    with self.assertRaisesRegex(OSError, "injected"):
                        service.rollback_committed_once(
                            record.record_id,
                            authorization,
                        )
                self.assertEqual(
                    fixture.target.read_text(encoding="utf-8"), "VALUE = 2\n"
                )
                self.assertEqual(service.journal.records()[0].status, "committed")
                self.assertEqual(service.rhythm.mode.value, "inference")
                with self.assertRaises(ApprovalReplayError):
                    service.rollback_committed_once(
                        record.record_id,
                        authorization,
                    )

    def test_startup_recovers_exact_crash_interrupted_candidate(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "cogni_flow").mkdir()
            target = root / "cogni_flow" / "target.py"
            before = b"VALUE = 1\n"
            after = b"VALUE = 2\n"
            target.write_bytes(before)
            journal = BackupJournal(
                root,
                root / ".state" / "journal",
                allowed_roots=("cogni_flow",),
                max_records=4,
                max_backup_bytes=1_024,
            )
            journal.prepare(Path("cogni_flow/target.py"), before, after)
            target.write_bytes(after)

            self.assertEqual(journal.recover_incomplete(), 1)
            self.assertEqual(target.read_bytes(), before)
            self.assertEqual(journal.records()[0].status, "recovered_rollback")

    def test_service_recovers_journal_only_inside_checkpointed_evolution(self):
        with tempfile.TemporaryDirectory() as tmp:
            fixture = ProductionHarnessFixture(Path(tmp))
            checkpoints = []
            service = fixture.build(
                ProductionHarnessConfig(),
                clock=FakeClock(),
                checkpoint=lambda: checkpoints.append("checkpoint"),
            )
            before = fixture.target.read_bytes()
            after = b"VALUE = 2\n"
            service.journal.prepare(Path("cogni_flow/target.py"), before, after)
            fixture.target.write_bytes(after)

            service.start()
            try:
                self.assertEqual(fixture.target.read_bytes(), before)
                self.assertEqual(checkpoints, ["checkpoint"])
                self.assertEqual(service.rhythm.mode.value, "inference")
                self.assertEqual(
                    service.journal.records()[0].status, "recovered_rollback"
                )
            finally:
                service.stop()

    def test_journal_refuses_to_overwrite_an_unknown_live_digest(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "cogni_flow").mkdir()
            target = root / "cogni_flow" / "target.py"
            before = b"VALUE = 1\n"
            after = b"VALUE = 2\n"
            unknown = b"VALUE = 3\n"
            target.write_bytes(before)
            journal = BackupJournal(
                root,
                root / ".state" / "journal",
                allowed_roots=("cogni_flow",),
                max_records=4,
                max_backup_bytes=1_024,
            )
            record = journal.prepare(Path("cogni_flow/target.py"), before, after)
            target.write_bytes(unknown)

            with self.assertRaisesRegex(JournalIntegrityError, "unsafe rollback"):
                journal.rollback(record)
            self.assertEqual(target.read_bytes(), unknown)

    def test_target_mapping_cannot_escape_the_mutable_allowlist(self):
        with tempfile.TemporaryDirectory() as tmp:
            fixture = ProductionHarnessFixture(Path(tmp))
            with self.assertRaisesRegex(ValueError, "allowlist"):
                build_production_self_harness(
                    fixture.root,
                    FakeModel(),
                    FakeTokenizer(),
                    {fixture.signature: "../outside.py"},
                    lambda: None,
                )

    def test_local_sqlite_ledgers_prune_to_hard_record_bounds(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = BoundedLogDB(
                Path(tmp) / "events.sqlite3",
                max_failure_records=2,
                max_audit_records=2,
            )
            for index in range(4):
                db.record_failure(FailureTrace(str(index), "Error", "V", "workflow"))
                db.audit("event", str(index), "detail")

            self.assertEqual(
                [item.test_id for item in db.failures_since(0)], ["2", "3"]
            )
            self.assertEqual([item.subject for item in db.audit_events()], ["2", "3"])


if __name__ == "__main__":
    unittest.main()
