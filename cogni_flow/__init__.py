"""Auditable control-plane primitives for Cogni-OS day/night evolution.

Exports are resolved lazily so importing the rhythm or proposal-review control
plane cannot preload evolution/model code or ``torch`` before GPU5 authority.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any


_EXPORTS = {
    "ApprovalError": ("approval", "ApprovalError"),
    "ApprovalReplayError": ("approval", "ApprovalReplayError"),
    "CandidateEvaluationLedger": ("approval", "CandidateEvaluationLedger"),
    "CandidateEvaluationV1": ("approval", "CandidateEvaluationV1"),
    "ConsumedApprovalLedger": ("approval", "ConsumedApprovalLedger"),
    "Ed25519ApprovalVerifier": ("approval", "Ed25519ApprovalVerifier"),
    "HumanApprovalV1": ("approval", "HumanApprovalV1"),
    "RollbackAuthorizationV1": ("approval", "RollbackAuthorizationV1"),
    "SafeProjectSnapshotBuilder": ("snapshot", "SafeProjectSnapshotBuilder"),
    "SnapshotBoundaryError": ("snapshot", "SnapshotBoundaryError"),
    "SnapshotEvidence": ("snapshot", "SnapshotEvidence"),
    "LogDB": ("logdb", "LogDB"),
    "AFlowOptimizer": ("aflow", "AFlowOptimizer"),
    "AttestedLocalGemmaWorkflowProposer": (
        "aflow_research",
        "AttestedLocalGemmaWorkflowProposer",
    ),
    "BenchmarkCase": ("aflow_research", "BenchmarkCase"),
    "BenchmarkSplit": ("aflow_research", "BenchmarkSplit"),
    "BenchmarkSuite": ("aflow_research", "BenchmarkSuite"),
    "CandidateDraft": ("proposals", "CandidateDraft"),
    "CaptureCoverage": ("proposals", "CaptureCoverage"),
    "FailureCaptureDaemon": ("daemon", "FailureCaptureDaemon"),
    "CFireTarget": ("evolution", "CFireTarget"),
    "EvolutionTransaction": ("evolution", "EvolutionTransaction"),
    "EvaluationObservation": ("aflow_research", "EvaluationObservation"),
    "EvaluationPolicy": ("aflow_research", "EvaluationPolicy"),
    "EvaluatorAttestation": ("aflow_research", "EvaluatorAttestation"),
    "FailureEventV1": ("proposals", "FailureEventV1"),
    "GatedLocalGemmaProposer": ("aflow_research", "GatedLocalGemmaProposer"),
    "GenerationCheckpointStore": ("evolution", "GenerationCheckpointStore"),
    "IdleNightScheduler": ("scheduler", "IdleNightScheduler"),
    "LocalGemmaPatchProposer": ("local_proposer", "LocalGemmaPatchProposer"),
    "LocalGemmaProposerAttestation": (
        "aflow_research",
        "LocalGemmaProposerAttestation",
    ),
    "MutationKind": ("aflow_research", "MutationKind"),
    "NegativeProposalV1": ("proposals", "NegativeProposalV1"),
    "PatchProposalV1": ("proposals", "PatchProposalV1"),
    "ProposalOnlyError": ("proposals", "ProposalOnlyError"),
    "ProposalOnlySelfHarness": ("proposals", "ProposalOnlySelfHarness"),
    "PromotionTarget": ("aflow_research", "PromotionTarget"),
    "ProductionHarnessConfig": ("production", "ProductionHarnessConfig"),
    "OperatorRollbackResult": ("production", "OperatorRollbackResult"),
    "ProductionSelfHarness": ("production", "ProductionSelfHarness"),
    "PromotionMode": ("production", "PromotionMode"),
    "ResolvedPatchTarget": ("local_proposer", "ResolvedPatchTarget"),
    "RhythmController": ("rhythm", "RhythmController"),
    "ResearchAFlowExecutor": ("aflow_research", "ResearchAFlowExecutor"),
    "ResearchArchive": ("aflow_research", "ResearchArchive"),
    "ResearchNightReport": ("aflow_research", "ResearchNightReport"),
    "ResearchSearchPolicy": ("aflow_research", "ResearchSearchPolicy"),
    "ResearchSearchResult": ("aflow_research", "ResearchSearchResult"),
    "ResearchWorkflowCoordinator": (
        "aflow_research",
        "ResearchWorkflowCoordinator",
    ),
    "ResearchWorkflowValidator": (
        "aflow_research",
        "ResearchWorkflowValidator",
    ),
    "SafeHarnessPatcher": ("harness", "SafeHarnessPatcher"),
    "SelfHarness": ("cycle", "SelfHarness"),
    "SealedCandidateEvaluator": ("aflow_research", "SealedCandidateEvaluator"),
    "SuccessInvariantV1": ("proposals", "SuccessInvariantV1"),
    "SystemMode": ("rhythm", "SystemMode"),
    "RunnerAttestation": ("production", "RunnerAttestation"),
    "WorkflowSpec": ("aflow", "WorkflowSpec"),
    "build_production_self_harness": (
        "production",
        "build_production_self_harness",
    ),
    "CapabilityError": ("task_plan", "CapabilityError"),
    "CapabilityToken": ("task_plan", "CapabilityToken"),
    "continual_transfer_metrics": ("evolution", "continual_transfer_metrics"),
    "ExpectedArtifact": ("task_plan", "ExpectedArtifact"),
    "InMemoryProposalStager": ("task_plan", "InMemoryProposalStager"),
    "RequiredInput": ("task_plan", "RequiredInput"),
    "RiskTier": ("task_plan", "RiskTier"),
    "summarize_seeded_transfer": ("evolution", "summarize_seeded_transfer"),
    "TaskAction": ("task_plan", "TaskAction"),
    "TaskActionKind": ("task_plan", "TaskActionKind"),
    "TaskBudget": ("task_plan", "TaskBudget"),
    "TaskExecutionError": ("task_plan", "TaskExecutionError"),
    "TaskPlanExecutor": ("task_plan", "TaskPlanExecutor"),
    "TaskPlanPolicy": ("task_plan", "TaskPlanPolicy"),
    "TaskPlanResult": ("task_plan", "TaskPlanResult"),
    "TaskPolicyError": ("task_plan", "TaskPolicyError"),
    "TaskVerifier": ("task_plan", "TaskVerifier"),
    "TypedTaskPlan": ("task_plan", "TypedTaskPlan"),
    "UnverifiedPlannerError": ("task_plan", "UnverifiedPlannerError"),
    "UnverifiedPlannerGate": ("task_plan", "UnverifiedPlannerGate"),
    "VerifierKind": ("task_plan", "VerifierKind"),
    "WorkflowOperator": ("aflow_research", "WorkflowOperator"),
    "WorkflowProposal": ("aflow_research", "WorkflowProposal"),
}

__all__ = list(_EXPORTS)


def __getattr__(name: str) -> Any:
    try:
        module_name, attribute_name = _EXPORTS[name]
    except KeyError as error:
        raise AttributeError(
            f"module {__name__!r} has no attribute {name!r}"
        ) from error
    value = getattr(import_module(f"{__name__}.{module_name}"), attribute_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
