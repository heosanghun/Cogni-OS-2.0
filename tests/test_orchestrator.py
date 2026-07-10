import unittest
from threading import Event, Thread

from cogni_flow.aflow import (
    AFlowBudget,
    AFlowOptimizer,
    ActionSpec,
    NodeSpec,
    WorkflowSpec,
)
from cogni_flow.orchestrator import WorkflowEvolutionCoordinator
from cogni_flow.rhythm import RhythmController, SystemMode


class TestWorkflowEvolutionCoordinator(unittest.TestCase):
    def test_aflow_runs_only_inside_night_window_and_resumes(self):
        initial = WorkflowSpec("w0", (NodeSpec("n", (ActionSpec("solve"),)),), ())

        def proposer(parent, context):
            return WorkflowSpec(
                f"w{context.iteration}",
                (
                    NodeSpec(
                        "n", (ActionSpec("solve", (("v", str(context.iteration)),)),)
                    ),
                ),
                (),
            )

        def evaluator(workflow):
            return float(workflow.workflow_id[1:])

        optimizer = AFlowOptimizer(
            proposer,
            evaluator,
            budget=AFlowBudget(max_iterations=2, max_archive_entries=3),
        )
        rhythm = RhythmController()
        report = WorkflowEvolutionCoordinator(rhythm, optimizer, lambda: None).run(
            initial
        )
        self.assertEqual(report.search.best.workflow.workflow_id, "w2")
        self.assertEqual(rhythm.mode, SystemMode.INFERENCE)

    def test_workflow_search_lifetime_blocks_concurrent_resume(self):
        initial = WorkflowSpec("w0", (NodeSpec("n", (ActionSpec("solve"),)),), ())
        started = Event()
        release = Event()

        def proposer(parent, context):
            started.set()
            release.wait(timeout=2)
            return parent

        optimizer = AFlowOptimizer(
            proposer,
            lambda workflow: 0.0,
            budget=AFlowBudget(max_iterations=1, max_archive_entries=2),
        )
        rhythm = RhythmController()
        coordinator = WorkflowEvolutionCoordinator(rhythm, optimizer, lambda: None)
        errors = []

        def run():
            try:
                coordinator.run(initial)
            except Exception as exc:  # pragma: no cover - surfaced by assertion
                errors.append(exc)

        worker = Thread(target=run)
        worker.start()
        self.assertTrue(started.wait(timeout=2))
        with self.assertRaisesRegex(RuntimeError, "evolution tasks"):
            rhythm.resume_inference("unsafe concurrent resume")
        release.set()
        worker.join(timeout=3)
        self.assertFalse(worker.is_alive())
        self.assertFalse(errors)
        self.assertEqual(rhythm.mode, SystemMode.INFERENCE)


if __name__ == "__main__":
    unittest.main()
