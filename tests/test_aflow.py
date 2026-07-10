import dataclasses
import math
import unittest

from cogni_flow.aflow import (
    AFlowBudget,
    AFlowOptimizer,
    ActionSpec,
    EdgeSpec,
    NodeSpec,
    WorkflowSpec,
    WorkflowValidationError,
    WorkflowValidator,
    soft_mixed_probabilities,
)


def action(name="generate", **arguments):
    return ActionSpec(name, tuple(sorted(arguments.items())))


def single_node(workflow_id, operator="generate"):
    return WorkflowSpec(
        workflow_id,
        (NodeSpec("node", (action(operator),)),),
    )


class TestWorkflowValidation(unittest.TestCase):
    def test_specs_are_deeply_immutable(self):
        workflow = single_node("immutable")
        with self.assertRaises(dataclasses.FrozenInstanceError):
            workflow.workflow_id = "changed"
        self.assertIsInstance(workflow.nodes, tuple)
        self.assertIsInstance(workflow.nodes[0].actions, tuple)

    def test_dag_and_each_static_budget_are_enforced(self):
        cyclic = WorkflowSpec(
            "cycle",
            (
                NodeSpec("a", (action(),)),
                NodeSpec("b", (action(),)),
            ),
            (EdgeSpec("a", "b"), EdgeSpec("b", "a")),
        )
        with self.assertRaisesRegex(WorkflowValidationError, "acyclic"):
            WorkflowValidator(AFlowBudget()).validate(cyclic)

        two_nodes = WorkflowSpec(
            "two",
            (
                NodeSpec("a", (action(),)),
                NodeSpec("b", (action(),)),
            ),
            (EdgeSpec("a", "b"),),
        )
        with self.assertRaisesRegex(WorkflowValidationError, "node budget"):
            WorkflowValidator(AFlowBudget(max_nodes=1)).validate(two_nodes)
        with self.assertRaisesRegex(WorkflowValidationError, "edge budget"):
            WorkflowValidator(AFlowBudget(max_edges=0)).validate(two_nodes)

        two_actions = WorkflowSpec(
            "actions", (NodeSpec("a", (action("one"), action("two"))),)
        )
        with self.assertRaisesRegex(WorkflowValidationError, "action budget"):
            WorkflowValidator(AFlowBudget(max_actions=1)).validate(two_actions)

    def test_valid_dag_summary_is_deterministic(self):
        workflow = WorkflowSpec(
            "dag",
            (
                NodeSpec("b", (action(),)),
                NodeSpec("a", (action(),)),
                NodeSpec("c", (action(),)),
            ),
            (EdgeSpec("a", "c"), EdgeSpec("b", "c")),
        )
        summary = WorkflowValidator(AFlowBudget()).validate(workflow)
        self.assertEqual(summary.roots, ("a", "b"))
        self.assertEqual(summary.leaves, ("c",))
        self.assertEqual((summary.nodes, summary.edges, summary.actions), (3, 2, 3))


class TestSoftMixedSelection(unittest.TestCase):
    def test_equation_matches_uniform_plus_stable_softmax(self):
        scores = (1.0, 3.0)
        probabilities = soft_mixed_probabilities(scores, lambda_mix=0.2, alpha=0.4)
        weights = (math.exp(-0.8), 1.0)
        expected = tuple(0.2 / 2 + 0.8 * weight / sum(weights) for weight in weights)
        self.assertAlmostEqual(sum(probabilities), 1.0)
        self.assertAlmostEqual(probabilities[0], expected[0])
        self.assertAlmostEqual(probabilities[1], expected[1])

    def test_equal_scores_are_uniform(self):
        self.assertEqual(
            soft_mixed_probabilities((7.0, 7.0, 7.0)),
            (1 / 3, 1 / 3, 1 / 3),
        )


class TestAFlowSearch(unittest.TestCase):
    def test_seeded_search_is_repeatable(self):
        def proposer(parent, context):
            return single_node(f"candidate-{context.iteration}")

        def evaluator(workflow):
            if workflow.workflow_id == "root":
                return 0.0
            return float(workflow.workflow_id.rsplit("-", 1)[1])

        budget = AFlowBudget(max_iterations=8, max_archive_entries=16)
        first = AFlowOptimizer(
            proposer, evaluator, budget=budget, seed=19, top_k=3
        ).search(WorkflowSpec("root"))
        second = AFlowOptimizer(
            proposer, evaluator, budget=budget, seed=19, top_k=3
        ).search(WorkflowSpec("root"))
        first_shape = tuple(
            (record.workflow.workflow_id, record.parent_id, record.visits)
            for record in first.archive
        )
        second_shape = tuple(
            (record.workflow.workflow_id, record.parent_id, record.visits)
            for record in second.archive
        )
        self.assertEqual(first_shape, second_shape)
        self.assertEqual(first.best_workflow.workflow_id, "candidate-8")

    def test_invalid_candidates_consume_iterations_but_are_not_evaluated(self):
        evaluator_calls = []

        def evaluator(workflow):
            evaluator_calls.append(workflow.workflow_id)
            return 0.0

        invalid = WorkflowSpec(
            "invalid",
            (NodeSpec("a", (action(),)), NodeSpec("b", (action(),))),
            (EdgeSpec("a", "b"), EdgeSpec("b", "a")),
        )
        result = AFlowOptimizer(
            lambda parent, context: invalid,
            evaluator,
            budget=AFlowBudget(max_iterations=3),
        ).search(WorkflowSpec("root"))
        self.assertEqual(result.iterations_used, 3)
        self.assertEqual(result.stop_reason, "iteration_budget_exhausted")
        self.assertEqual(evaluator_calls, ["root"])
        self.assertEqual(len(result.archive), 1)
        self.assertEqual(len(result.rejected), 3)
        self.assertTrue(
            all(item.reason.startswith("invalid_workflow:") for item in result.rejected)
        )

    def test_iteration_and_archive_budgets_stop_search(self):
        calls = []

        def proposer(parent, context):
            calls.append(context.iteration)
            return single_node(f"candidate-{context.iteration}")

        no_iterations = AFlowOptimizer(
            proposer,
            lambda workflow: 0.0,
            budget=AFlowBudget(max_iterations=0),
        ).search(WorkflowSpec("root"))
        self.assertEqual(calls, [])
        self.assertEqual(no_iterations.evaluations, 1)
        self.assertEqual(no_iterations.stop_reason, "iteration_budget_exhausted")

        archive_limited = AFlowOptimizer(
            proposer,
            lambda workflow: float(len(workflow.nodes)),
            budget=AFlowBudget(max_iterations=10, max_archive_entries=2),
        ).search(WorkflowSpec("root"))
        self.assertEqual(len(archive_limited.archive), 2)
        self.assertEqual(archive_limited.iterations_used, 1)
        self.assertEqual(archive_limited.stop_reason, "archive_budget_exhausted")

    def test_best_archive_entry_is_promoted_only_on_strict_improvement(self):
        scores = {"candidate-1": 1.0, "candidate-2": 5.0, "candidate-3": 3.0}

        def proposer(parent, context):
            return single_node(f"candidate-{context.iteration}")

        result = AFlowOptimizer(
            proposer,
            lambda workflow: scores.get(workflow.workflow_id, 0.0),
            budget=AFlowBudget(max_iterations=3),
            seed=4,
        ).search(WorkflowSpec("root"))
        self.assertEqual(result.best_workflow.workflow_id, "candidate-2")
        self.assertEqual(result.best_score, 5.0)
        self.assertEqual(result.promotions, (0, 1, 2))

    def test_action_payload_is_inert_and_evaluation_is_delegated(self):
        observed = []
        candidate = WorkflowSpec(
            "declarative",
            (
                NodeSpec(
                    "node",
                    (
                        action(
                            "python_source",
                            source="__import__('os').system('not executed')",
                        ),
                    ),
                ),
            ),
        )

        def evaluator(workflow):
            observed.append(workflow)
            return 1.0 if workflow.workflow_id == "declarative" else 0.0

        result = AFlowOptimizer(
            lambda parent, context: candidate,
            evaluator,
            budget=AFlowBudget(max_iterations=1),
        ).search(WorkflowSpec("root"))
        self.assertEqual(observed, [WorkflowSpec("root"), candidate])
        self.assertEqual(result.best_workflow, candidate)


if __name__ == "__main__":
    unittest.main()
