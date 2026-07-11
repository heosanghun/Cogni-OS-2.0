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
                self.assertEqual(len(service.pending_proposals), 1)

            self.assertEqual(fixture.target.read_text(encoding="utf-8"), "VALUE = 1\n")
            candidates = [
                event
                for event in service.logdb.audit_events()
                if event.kind == "candidate"
            ]
            self.assertEqual(candidates[-1].detail, "proposal_only")

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
