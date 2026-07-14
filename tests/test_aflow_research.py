import dataclasses
import inspect
import json
import unittest

from cogni_flow.aflow import AFlowBudget, ActionSpec, EdgeSpec, NodeSpec, WorkflowSpec
from cogni_flow.aflow_research import (
    AttestedLocalGemmaWorkflowProposer,
    BenchmarkCase,
    BenchmarkSplit,
    BenchmarkSuite,
    EvaluationObservation,
    EvaluationPolicy,
    EvaluatorAttestation,
    EvaluatorSurfaceTampered,
    LocalGemmaProposerAttestation,
    MutationKind,
    ProposalRequest,
    PromotionTarget,
    ResearchAFlowExecutor,
    ResearchSearchPolicy,
    ResearchWorkflowCoordinator,
    ResearchWorkflowValidator,
    ResearchValidationError,
    SealedCandidateEvaluator,
    WorkflowOperator,
    WorkflowProposal,
    digest_workflow,
)
from cogni_flow.rhythm import RhythmController, SystemMode
from cogni_flow.scheduler import IdleNightScheduler


def workflow(workflow_id, operator=WorkflowOperator.GENERATE, **arguments):
    return WorkflowSpec(
        workflow_id,
        (
            NodeSpec(
                "root",
                (ActionSpec(operator, tuple(sorted(arguments.items()))),),
            ),
        ),
        (),
    )


def suite():
    return BenchmarkSuite(
        "phase10",
        "v1",
        (
            BenchmarkCase("held-in-1", BenchmarkSplit.HELD_IN, "input/a", "score/a"),
            BenchmarkCase("held-out-1", BenchmarkSplit.HELD_OUT, "input/b", "score/b"),
        ),
    )


def attestation():
    return EvaluatorAttestation("deterministic-evaluator", "v1", "a" * 64)


class ConstantEvaluator:
    def __init__(self, quality=0.6):
        self.quality = quality
        self.calls = []

    def __call__(self, candidate, case, repeat_seed):
        self.calls.append((candidate.workflow_id, case.case_id, repeat_seed))
        return EvaluationObservation(
            case.case_id,
            self.quality,
            latency_ms=2.0,
            resource_units=3.0,
            tool_calls=1,
        )


class TestStrictResearchWorkflow(unittest.TestCase):
    def test_typed_operator_workflow_is_deeply_immutable(self):
        candidate = workflow("typed", prompt_id="prompt/v1")
        ResearchWorkflowValidator(AFlowBudget()).validate(candidate)
        with self.assertRaises(dataclasses.FrozenInstanceError):
            candidate.workflow_id = "changed"
        with self.assertRaises(dataclasses.FrozenInstanceError):
            candidate.nodes[0].node_id = "changed"

    def test_cycle_and_each_budget_fail_closed(self):
        cyclic = WorkflowSpec(
            "cyclic",
            (
                NodeSpec("a", (ActionSpec(WorkflowOperator.GENERATE),)),
                NodeSpec("b", (ActionSpec(WorkflowOperator.REVIEW),)),
            ),
            (EdgeSpec("a", "b"), EdgeSpec("b", "a")),
        )
        with self.assertRaisesRegex(ValueError, "acyclic"):
            ResearchWorkflowValidator(AFlowBudget()).validate(cyclic)

        two_nodes = WorkflowSpec(
            "two",
            (
                NodeSpec("a", (ActionSpec(WorkflowOperator.GENERATE),)),
                NodeSpec("b", (ActionSpec(WorkflowOperator.REVIEW),)),
            ),
            (EdgeSpec("a", "b"),),
        )
        with self.assertRaisesRegex(ValueError, "node budget"):
            ResearchWorkflowValidator(AFlowBudget(max_nodes=1)).validate(two_nodes)
        with self.assertRaisesRegex(ValueError, "edge budget"):
            ResearchWorkflowValidator(AFlowBudget(max_edges=0)).validate(two_nodes)
        with self.assertRaisesRegex(ValueError, "action budget"):
            ResearchWorkflowValidator(AFlowBudget(max_actions=0)).validate(
                workflow("one")
            )

    def test_untyped_or_arbitrary_code_payload_is_rejected(self):
        untyped = WorkflowSpec(
            "untyped", (NodeSpec("n", (ActionSpec("generate"),)),), ()
        )
        with self.assertRaisesRegex(ValueError, "typed WorkflowOperator"):
            ResearchWorkflowValidator(AFlowBudget()).validate(untyped)

        source_payload = workflow(
            "source",
            WorkflowOperator.PROGRAMMER,
            source="__import__/os/system",
        )
        with self.assertRaisesRegex(ValueError, "not allowed"):
            ResearchWorkflowValidator(AFlowBudget()).validate(source_payload)

    @staticmethod
    def _proposer_attestation(**overrides):
        values = {
            "proposer_id": "local-gemma-e4b-v1",
            "model_manifest_sha256": "a" * 64,
            "proposer_artifact_sha256": "b" * 64,
            "held_out_evidence_sha256": "c" * 64,
            "passed": True,
            "independent": True,
        }
        values.update(overrides)
        return LocalGemmaProposerAttestation(**values)

    def test_attested_local_gemma_adapter_parses_only_inert_typed_json(self):
        payload = {
            "workflow_id": "gemma-candidate-1",
            "mutation_kind": "minor",
            "nodes": [
                {
                    "node_id": "root",
                    "actions": [
                        {
                            "operator": "review",
                            "arguments": {
                                "rubric_id": "rubric/v1",
                                "schema_id": "schema/v1",
                            },
                        }
                    ],
                }
            ],
            "edges": [],
        }
        prompts = []

        def generate(prompt, seed):
            prompts.append((prompt, seed))
            return json.dumps(payload)

        parent = workflow("parent")
        request = ProposalRequest(1, 91, 0, digest_workflow(parent), ())
        proposer = AttestedLocalGemmaWorkflowProposer(
            generate, self._proposer_attestation()
        )
        proposal = proposer(parent, request)

        ResearchWorkflowValidator(AFlowBudget()).validate(proposal.workflow)
        self.assertEqual(proposal.parent_workflow_ids, ("parent",))
        self.assertEqual(proposal.lineage_seed, 91)
        self.assertIs(proposal.mutation_kind, MutationKind.MINOR)
        self.assertEqual(
            proposal.workflow.nodes[0].actions[0].operator,
            WorkflowOperator.REVIEW,
        )
        self.assertEqual(len(prompts), 1)
        self.assertNotIn("production", prompts[0][0])

    def test_local_gemma_adapter_requires_independent_evidence_and_exact_schema(self):
        with self.assertRaisesRegex(Exception, "independent"):
            AttestedLocalGemmaWorkflowProposer(
                lambda _prompt, _seed: "{}",
                self._proposer_attestation(independent=False),
            )
        proposer = AttestedLocalGemmaWorkflowProposer(
            lambda _prompt, _seed: json.dumps(
                {
                    "workflow_id": "unsafe",
                    "mutation_kind": "minor",
                    "nodes": [],
                    "edges": [],
                    "source": "import os",
                }
            ),
            self._proposer_attestation(),
        )
        parent = workflow("parent")
        request = ProposalRequest(1, 1, 0, digest_workflow(parent), ())
        with self.assertRaisesRegex(ResearchValidationError, "schema"):
            proposer(parent, request)


class TestSealedEvaluation(unittest.TestCase):
    def test_repeats_are_at_least_five_and_both_splits_are_aggregated(self):
        with self.assertRaisesRegex(ValueError, "at least five"):
            EvaluationPolicy(repeats=4)

        callback = ConstantEvaluator()
        sealed = SealedCandidateEvaluator(
            suite(), EvaluationPolicy(repeats=5), attestation(), callback
        )
        result = sealed.evaluate(workflow("root"), seed=7)
        self.assertEqual(len(callback.calls), 10)
        self.assertEqual(result.samples, 10)
        self.assertEqual(result.held_in.samples, 5)
        self.assertEqual(result.held_out.samples, 5)
        self.assertEqual(result.quality_mean, 0.6)
        self.assertEqual(result.latency_max_ms, 2.0)
        self.assertEqual(result.latency_std_ms, 0.0)
        self.assertEqual(result.resource_max, 3.0)
        self.assertEqual(result.resource_std, 0.0)
        self.assertEqual(result.tool_calls_total, 10)

    def test_policy_callable_and_result_tampering_are_rejected(self):
        sealed = SealedCandidateEvaluator(
            suite(), EvaluationPolicy(), attestation(), ConstantEvaluator()
        )
        result = sealed.evaluate(workflow("root"), seed=4)
        forged = dataclasses.replace(result, policy_digest="0" * 64)
        with self.assertRaisesRegex(EvaluatorSurfaceTampered, "policy"):
            sealed.validate_result(forged)

        object.__setattr__(sealed.policy, "repeats", 6)
        with self.assertRaisesRegex(EvaluatorSurfaceTampered, "surface"):
            sealed.verify()

        sealed = SealedCandidateEvaluator(
            suite(), EvaluationPolicy(), attestation(), ConstantEvaluator()
        )
        object.__setattr__(sealed, "evaluator", ConstantEvaluator(0.9))
        with self.assertRaisesRegex(EvaluatorSurfaceTampered, "callable"):
            sealed.verify()

        budget = AFlowBudget(max_iterations=1, max_archive_entries=8)
        executor = ResearchAFlowExecutor(
            SealedCandidateEvaluator(
                suite(), EvaluationPolicy(), attestation(), ConstantEvaluator()
            ),
            budget=budget,
            policy=ResearchSearchPolicy(max_successful_traces=4, max_failed_traces=4),
        )
        object.__setattr__(budget, "max_iterations", 2)
        with self.assertRaisesRegex(EvaluatorSurfaceTampered, "budget"):
            executor.search(workflow("root"))

        executor = ResearchAFlowExecutor(
            SealedCandidateEvaluator(
                suite(), EvaluationPolicy(), attestation(), ConstantEvaluator()
            ),
            budget=AFlowBudget(max_iterations=1, max_archive_entries=8),
            policy=ResearchSearchPolicy(max_successful_traces=4, max_failed_traces=4),
        )
        executor.evaluator = SealedCandidateEvaluator(
            suite(), EvaluationPolicy(), attestation(), ConstantEvaluator()
        )
        with self.assertRaisesRegex(EvaluatorSurfaceTampered, "replaced"):
            executor.search(workflow("root"))


class TestResearchSearch(unittest.TestCase):
    def _sealed(self, callback, **policy_overrides):
        return SealedCandidateEvaluator(
            suite(), EvaluationPolicy(**policy_overrides), attestation(), callback
        )

    def _search_policy(self, **overrides):
        values = {
            "max_successful_traces": 4,
            "max_failed_traces": 4,
            "early_stop_patience": 20,
            "min_iterations_before_early_stop": 0,
        }
        values.update(overrides)
        return ResearchSearchPolicy(**values)

    def test_seeded_lineage_and_evaluation_replay_are_deterministic(self):
        callback = ConstantEvaluator(0.7)

        def proposer(parent, request):
            return WorkflowProposal(
                workflow(f"candidate-{request.iteration}"),
                (parent.workflow_id,),
                MutationKind.MINOR,
                request.lineage_seed,
            )

        budget = AFlowBudget(max_iterations=4, max_archive_entries=8)
        first = ResearchAFlowExecutor(
            self._sealed(callback),
            proposer,
            budget=budget,
            policy=self._search_policy(),
            seed=91,
        ).search(workflow("baseline"))
        second = ResearchAFlowExecutor(
            self._sealed(callback),
            proposer,
            budget=budget,
            policy=self._search_policy(),
            seed=91,
        ).search(workflow("baseline"))
        self.assertEqual(first.replay_digest, second.replay_digest)
        self.assertEqual(first.search_policy_digest, second.search_policy_digest)
        first_lineage = tuple(
            (
                trace.iteration,
                trace.parent_trace_id,
                trace.lineage_seed,
                trace.workflow_digest,
            )
            for trace in first.archive.successful
        )
        second_lineage = tuple(
            (
                trace.iteration,
                trace.parent_trace_id,
                trace.lineage_seed,
                trace.workflow_digest,
            )
            for trace in second.archive.successful
        )
        self.assertEqual(first_lineage, second_lineage)

    def test_major_change_must_name_exactly_one_selected_parent(self):
        callback = ConstantEvaluator()

        def invalid_proposer(parent, request):
            return WorkflowProposal(
                workflow("bad"),
                (parent.workflow_id, "second-parent"),
                MutationKind.MAJOR,
                request.lineage_seed,
            )

        result = ResearchAFlowExecutor(
            self._sealed(callback),
            invalid_proposer,
            budget=AFlowBudget(max_iterations=1, max_archive_entries=8),
            policy=self._search_policy(),
        ).search(workflow("baseline"))
        self.assertEqual(len(result.archive.successful), 1)
        self.assertEqual(len(result.archive.failed), 1)
        self.assertIn("exactly one parent", result.archive.failed[0].failure_reason)
        self.assertEqual(len(callback.calls), 10)

    def test_non_regression_variance_and_improvement_gates(self):
        def evaluator(candidate, case, repeat_seed):
            if candidate.workflow_id == "candidate-1":
                quality = 0.4
            elif candidate.workflow_id == "candidate-2":
                quality = (repeat_seed % 101) / 100.0
            elif candidate.workflow_id == "candidate-3":
                quality = 0.8
            else:
                quality = 0.6
            return EvaluationObservation(case.case_id, quality, 2.0, 2.0, 1)

        def proposer(parent, request):
            return WorkflowProposal(
                workflow(f"candidate-{request.iteration}"),
                (parent.workflow_id,),
                MutationKind.MAJOR,
                request.lineage_seed,
            )

        result = ResearchAFlowExecutor(
            self._sealed(
                evaluator,
                max_held_in_std=0.01,
                max_held_out_std=0.01,
                min_held_out_improvement=0.05,
            ),
            proposer,
            budget=AFlowBudget(max_iterations=3, max_archive_entries=8),
            policy=self._search_policy(),
            seed=3,
        ).search(workflow("baseline"))
        reasons = {trace.failure_reason for trace in result.archive.failed}
        self.assertIn("held_out_non_regression_gate", reasons)
        self.assertTrue({"held_in_variance_gate", "held_out_variance_gate"} & reasons)
        self.assertEqual(result.best.workflow.workflow_id, "candidate-3")
        self.assertTrue(result.best.promoted)

    def test_top_k_archives_are_bounded_and_patience_stops_early(self):
        callback = ConstantEvaluator(0.6)

        def duplicate(parent, request):
            return WorkflowProposal(
                workflow("same"),
                (parent.workflow_id,),
                MutationKind.MINOR,
                request.lineage_seed,
            )

        policy = self._search_policy(
            max_successful_traces=1,
            max_failed_traces=2,
            early_stop_patience=3,
            min_iterations_before_early_stop=3,
        )
        result = ResearchAFlowExecutor(
            self._sealed(callback),
            duplicate,
            budget=AFlowBudget(max_iterations=20, max_archive_entries=3),
            policy=policy,
        ).search(workflow("baseline"))
        self.assertEqual(result.stop_reason, "early_stop_patience")
        self.assertEqual(result.iterations_used, 3)
        self.assertLessEqual(len(result.archive.successful), 1)
        self.assertLessEqual(len(result.archive.failed), 2)

    def test_default_proposer_is_gated_and_result_has_no_production_path(self):
        result = ResearchAFlowExecutor(
            self._sealed(ConstantEvaluator()),
            budget=AFlowBudget(max_iterations=2, max_archive_entries=8),
            policy=self._search_policy(
                early_stop_patience=2, min_iterations_before_early_stop=2
            ),
        ).search(workflow("baseline"))
        self.assertEqual(result.iterations_used, 2)
        self.assertTrue(
            all(
                "local_gemma_quality_evidence_required" in trace.failure_reason
                for trace in result.archive.failed
            )
        )
        self.assertIs(
            result.archive.promotion_target, PromotionTarget.RESEARCH_ARCHIVE_ONLY
        )
        self.assertFalse(hasattr(result, "install"))
        source = inspect.getsource(
            __import__("cogni_flow.aflow_research").aflow_research
        )
        self.assertNotIn("from .production import", source)

    def test_scheduler_runs_archive_only_search_in_night_window(self):
        now = [10.0]
        rhythm = RhythmController()
        checkpoints = []
        executor = ResearchAFlowExecutor(
            self._sealed(ConstantEvaluator()),
            budget=AFlowBudget(max_iterations=1, max_archive_entries=2),
            policy=self._search_policy(
                max_successful_traces=1,
                max_failed_traces=1,
                early_stop_patience=1,
                min_iterations_before_early_stop=1,
            ),
        )
        coordinator = ResearchWorkflowCoordinator(
            rhythm, executor, lambda: checkpoints.append("checkpoint")
        )
        scheduler = IdleNightScheduler.for_research_workflow(
            coordinator,
            workflow("baseline"),
            idle_seconds=5.0,
            clock=lambda: now[0],
        )
        now[0] += 5.0
        tick = scheduler.tick()
        self.assertTrue(tick.ran)
        self.assertTrue(tick.result.resumed_inference)
        self.assertIs(
            tick.result.search.archive.promotion_target,
            PromotionTarget.RESEARCH_ARCHIVE_ONLY,
        )
        self.assertEqual(checkpoints, ["checkpoint"])
        self.assertIs(rhythm.mode, SystemMode.INFERENCE)


if __name__ == "__main__":
    unittest.main()
