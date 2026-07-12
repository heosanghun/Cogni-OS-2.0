from hashlib import sha256
import tempfile
from pathlib import Path
import unittest

import torch

from cogni_flow.harness import FailureTrace, SandboxResult
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

    def build(self, config, *, clock, runner=None, checkpoint=None):
        return build_production_self_harness(
            self.root,
            FakeModel(),
            FakeTokenizer(),
            self.targets,
            checkpoint or (lambda: None),
            config=config,
            runner=runner,
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

    def test_attested_promotion_runs_staging_then_post_atomic_health(self):
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
                tick = fixture.submit_and_run(service, clock)

            self.assertTrue(tick.result.promoted)
            self.assertEqual(fixture.target.read_text(encoding="utf-8"), "VALUE = 2\n")
            self.assertEqual(
                [item[0] for item in runner.calls],
                [config.regression_command, config.health_check_command],
            )
            self.assertTrue(all("VALUE = 2" in item[1] for item in runner.calls))
            records = service.journal.records()
            self.assertEqual(len(records), 1)
            self.assertEqual(records[0].status, "committed")

    def test_failed_post_promotion_health_restores_verified_backup(self):
        with tempfile.TemporaryDirectory() as tmp:
            fixture = ProductionHarnessFixture(Path(tmp))
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
            service = fixture.build(config, clock=clock, runner=runner)

            with service:
                tick = fixture.submit_and_run(service, clock)

            self.assertFalse(tick.result.promoted)
            self.assertEqual(fixture.target.read_text(encoding="utf-8"), "VALUE = 1\n")
            self.assertEqual(service.journal.records()[0].status, "rolled_back")
            self.assertEqual(service.rhythm.mode.value, "inference")

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
