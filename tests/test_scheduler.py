import tempfile
import unittest
from pathlib import Path
from threading import Event, Thread

from cogni_flow.aflow import (
    AFlowBudget,
    AFlowOptimizer,
    ActionSpec,
    NodeSpec,
    WorkflowSpec,
)
from cogni_flow.cycle import SelfHarness
from cogni_flow.logdb import LogDB
from cogni_flow.orchestrator import WorkflowEvolutionCoordinator
from cogni_flow.rhythm import RhythmController, SystemMode
from cogni_flow.scheduler import IdleNightScheduler, ScheduleDecision


class FakeClock:
    def __init__(self, value=0.0):
        self.value = value

    def __call__(self):
        return self.value

    def advance(self, seconds):
        self.value += seconds


class TestIdleNightScheduler(unittest.TestCase):
    def test_self_harness_runs_once_after_monotonic_idle_window(self):
        with tempfile.TemporaryDirectory() as tmp:
            clock = FakeClock(100.0)
            rhythm = RhythmController()
            checkpoints = []
            harness = SelfHarness(
                rhythm,
                LogDB(Path(tmp) / "events.sqlite3"),
                patcher=object(),
                proposer=lambda cluster: (),
                checkpoint=lambda: checkpoints.append("checkpoint"),
            )
            scheduler = IdleNightScheduler.for_self_harness(
                harness, idle_seconds=10, clock=clock
            )

            clock.advance(9)
            self.assertEqual(scheduler.tick().decision, ScheduleDecision.NOT_IDLE)
            clock.advance(1)
            result = scheduler.run_once()
            self.assertTrue(result.ran)
            self.assertFalse(result.result.promoted)
            self.assertEqual(checkpoints, ["checkpoint"])
            self.assertEqual(rhythm.mode, SystemMode.INFERENCE)
            self.assertEqual(scheduler.tick().decision, ScheduleDecision.NOT_IDLE)

    def test_active_inference_and_non_day_mode_block_cycle(self):
        clock = FakeClock()
        rhythm = RhythmController()
        calls = []
        scheduler = IdleNightScheduler(
            rhythm, lambda: calls.append("ran"), idle_seconds=5, clock=clock
        )
        clock.advance(5)
        with rhythm.inference_slot():
            result = scheduler.tick()
        self.assertEqual(result.decision, ScheduleDecision.INFERENCE_ACTIVE)
        self.assertFalse(calls)

        clock.advance(5)
        rhythm.enter_evolution(lambda: None)
        result = scheduler.tick()
        self.assertEqual(result.decision, ScheduleDecision.MODE_BLOCKED)
        self.assertFalse(calls)
        rhythm.resume_inference()

    def test_concurrent_ticks_cannot_duplicate_a_cycle(self):
        clock = FakeClock(10)
        rhythm = RhythmController()
        started = Event()
        release = Event()
        results = []

        def cycle():
            started.set()
            release.wait(timeout=2)
            return "done"

        scheduler = IdleNightScheduler(rhythm, cycle, idle_seconds=0, clock=clock)
        worker = Thread(target=lambda: results.append(scheduler.tick()))
        worker.start()
        self.assertTrue(started.wait(timeout=1))
        duplicate = scheduler.tick()
        self.assertEqual(duplicate.decision, ScheduleDecision.CYCLE_ACTIVE)
        release.set()
        worker.join(timeout=2)
        self.assertFalse(worker.is_alive())
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].result, "done")

    def test_workflow_coordinator_adapter_runs_in_evolution_mode(self):
        clock = FakeClock()
        rhythm = RhythmController()
        initial = WorkflowSpec("w0", (NodeSpec("n", (ActionSpec("solve"),)),), ())
        optimizer = AFlowOptimizer(
            lambda parent, context: parent,
            lambda workflow: 1.0,
            budget=AFlowBudget(max_iterations=1, max_archive_entries=2),
        )
        coordinator = WorkflowEvolutionCoordinator(rhythm, optimizer, lambda: None)
        scheduler = IdleNightScheduler.for_workflow(
            coordinator, initial, idle_seconds=1, clock=clock
        )
        clock.advance(1)
        result = scheduler.tick()
        self.assertTrue(result.ran)
        self.assertTrue(result.result.resumed_inference)
        self.assertEqual(rhythm.mode, SystemMode.INFERENCE)


if __name__ == "__main__":
    unittest.main()
