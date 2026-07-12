"""Bounded AFlow/ADAS research search with sealed evaluation surfaces.

This module intentionally has no workflow interpreter, Python execution hook,
filesystem installer, or production promotion callback.  It searches immutable,
declarative ``WorkflowSpec`` values and can promote a candidate only into the
bounded in-memory research archive returned to its caller.

The original :mod:`cogni_flow.aflow` optimizer remains a small compatibility
surface.  Phase 10 uses the stricter types and gates in this module.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import hashlib
import inspect
import json
import math
import random
import statistics
from types import MappingProxyType
from typing import Callable, Protocol, Sequence

from .aflow import (
    AFlowBudget,
    ActionSpec,
    EdgeSpec,
    NodeSpec,
    WorkflowSpec,
    WorkflowValidationError,
    WorkflowValidator,
    soft_mixed_probabilities,
)
from .rhythm import RhythmController, SystemMode


class ResearchValidationError(ValueError):
    """Raised when a Phase-10 research surface fails closed."""


class EvaluatorSurfaceTampered(ResearchValidationError):
    """Raised when a sealed evaluator, suite, or policy changed after creation."""


class ResearchEvaluationError(RuntimeError):
    """Raised when the immutable baseline cannot be evaluated."""


class ProposerGatedError(RuntimeError):
    """Raised by the default proposer until real local quality evidence exists."""


class WorkflowOperator(str, Enum):
    """Closed set of declarative operators accepted by the research executor."""

    GENERATE = "generate"
    REVIEW = "review"
    REVISE = "revise"
    ENSEMBLE = "ensemble"
    TEST = "test"
    PROGRAMMER = "programmer"


_SAFE_ARGUMENTS = MappingProxyType(
    {
        WorkflowOperator.GENERATE: frozenset(
            {"prompt_id", "schema_id", "temperature_id"}
        ),
        WorkflowOperator.REVIEW: frozenset(
            {"rubric_id", "schema_id", "review_depth_id"}
        ),
        WorkflowOperator.REVISE: frozenset(
            {"rubric_id", "revision_policy_id", "schema_id"}
        ),
        WorkflowOperator.ENSEMBLE: frozenset(
            {"aggregation_id", "member_limit_id", "schema_id"}
        ),
        WorkflowOperator.TEST: frozenset(
            {"suite_id", "assertion_profile_id", "schema_id"}
        ),
        WorkflowOperator.PROGRAMMER: frozenset(
            {"template_id", "task_id", "language_id", "schema_id"}
        ),
    }
)


class ResearchWorkflowValidator:
    """Strictly validate a bounded DAG made only from typed inert operators."""

    def __init__(self, budget: AFlowBudget) -> None:
        self.budget = budget
        self._base = WorkflowValidator(budget)

    def validate(self, workflow: WorkflowSpec):
        summary = self._base.validate(workflow)
        for node in workflow.nodes:
            for action in node.actions:
                if type(action.operator) is not WorkflowOperator:
                    raise WorkflowValidationError(
                        "research actions require a typed WorkflowOperator"
                    )
                allowed = _SAFE_ARGUMENTS[action.operator]
                for key, value in action.arguments:
                    if key not in allowed:
                        raise WorkflowValidationError(
                            f"argument {key!r} is not allowed for "
                            f"{action.operator.value}"
                        )
                    if not _is_safe_identifier(value):
                        raise WorkflowValidationError(
                            "research arguments must be bounded identifiers"
                        )
        return summary


def _is_safe_identifier(value: str) -> bool:
    if not value or len(value) > 128:
        return False
    return all(character.isalnum() or character in "._:-/" for character in value)


class MutationKind(str, Enum):
    MINOR = "minor"
    MAJOR = "major"


@dataclass(frozen=True, slots=True)
class ProposalRequest:
    """Deterministic, read-only proposal context."""

    iteration: int
    lineage_seed: int
    parent_trace_id: int
    parent_workflow_digest: str
    visible_archive: tuple["ResearchTraceSummary", ...]


@dataclass(frozen=True, slots=True)
class WorkflowProposal:
    """A single-parent mutation proposed from one selected workflow."""

    workflow: WorkflowSpec
    parent_workflow_ids: tuple[str, ...]
    mutation_kind: MutationKind
    lineage_seed: int


class ResearchWorkflowProposer(Protocol):
    def __call__(
        self, parent: WorkflowSpec, request: ProposalRequest
    ) -> WorkflowProposal: ...


class GatedLocalGemmaProposer:
    """Fail-closed default: no local Gemma quality evidence is fabricated."""

    def __call__(
        self, parent: WorkflowSpec, request: ProposalRequest
    ) -> WorkflowProposal:
        del parent, request
        raise ProposerGatedError("local_gemma_quality_evidence_required")


@dataclass(frozen=True, slots=True)
class LocalGemmaProposerAttestation:
    """Independent evidence required before local Gemma may propose a DAG."""

    proposer_id: str
    model_manifest_sha256: str
    proposer_artifact_sha256: str
    held_out_evidence_sha256: str
    passed: bool
    independent: bool

    def __post_init__(self) -> None:
        if not _is_safe_identifier(self.proposer_id):
            raise ValueError("proposer_id must be a bounded identifier")
        for name in (
            "model_manifest_sha256",
            "proposer_artifact_sha256",
            "held_out_evidence_sha256",
        ):
            value = getattr(self, name)
            if (
                not isinstance(value, str)
                or len(value) != 64
                or any(character not in "0123456789abcdefABCDEF" for character in value)
            ):
                raise ValueError(f"{name} must be SHA-256 hex")
        if not isinstance(self.passed, bool) or not isinstance(self.independent, bool):
            raise TypeError("proposer attestation verdicts must be bool")

    @property
    def digest(self) -> str:
        return _sha256_json(
            {
                "proposer_id": self.proposer_id,
                "model_manifest_sha256": self.model_manifest_sha256.lower(),
                "proposer_artifact_sha256": self.proposer_artifact_sha256.lower(),
                "held_out_evidence_sha256": self.held_out_evidence_sha256.lower(),
                "passed": self.passed,
                "independent": self.independent,
            }
        )


class AttestedLocalGemmaWorkflowProposer:
    """Parse only a sealed, inert workflow schema from a local generator.

    ``generate`` is injected by the single resident local-model owner.  The
    adapter has no model loader, network client, workflow interpreter, source
    writer, or production promotion callback.  An independent held-out
    attestation is mandatory; without it callers must keep using the default
    :class:`GatedLocalGemmaProposer`.
    """

    def __init__(
        self,
        generate: Callable[[str, int], str],
        attestation: LocalGemmaProposerAttestation,
        *,
        max_output_chars: int = 8_192,
    ) -> None:
        if not callable(generate):
            raise TypeError("local Gemma generator must be callable")
        if not isinstance(attestation, LocalGemmaProposerAttestation):
            raise TypeError("local Gemma proposer requires a typed attestation")
        if not attestation.passed or not attestation.independent:
            raise ProposerGatedError("independent_local_gemma_evidence_required")
        if not 256 <= max_output_chars <= 65_536:
            raise ValueError("max_output_chars is outside the bounded range")
        self._generate = generate
        self.attestation = attestation
        self.max_output_chars = max_output_chars

    @property
    def seal(self) -> str:
        return _sha256_json(
            {
                "attestation": self.attestation.digest,
                "generator": _callable_fingerprint(self._generate),
                "max_output_chars": self.max_output_chars,
            }
        )

    def __call__(
        self, parent: WorkflowSpec, request: ProposalRequest
    ) -> WorkflowProposal:
        prompt = self._prompt(parent, request)
        raw = self._generate(prompt, request.lineage_seed)
        if (
            type(raw) is not str
            or not raw
            or len(raw) > self.max_output_chars
            or "\x00" in raw
        ):
            raise ResearchValidationError(
                "local Gemma returned an invalid bounded proposal payload"
            )
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ResearchValidationError(
                "local Gemma proposal is not strict JSON"
            ) from exc
        workflow, mutation = self._parse_payload(payload)
        return WorkflowProposal(
            workflow=workflow,
            parent_workflow_ids=(parent.workflow_id,),
            mutation_kind=mutation,
            lineage_seed=request.lineage_seed,
        )

    @staticmethod
    def _prompt(parent: WorkflowSpec, request: ProposalRequest) -> str:
        payload = {
            "contract": "cogni-aflow-workflow-proposal-v1",
            "parent": {
                "workflow_id": parent.workflow_id,
                "digest": request.parent_workflow_digest,
            },
            "iteration": request.iteration,
            "lineage_seed": request.lineage_seed,
            "allowed_operators": [item.value for item in WorkflowOperator],
            "rule": (
                "Return JSON only. Use bounded identifiers and one single-parent "
                "workflow mutation. No code, paths, commands, or prose."
            ),
        }
        return json.dumps(
            payload, ensure_ascii=True, sort_keys=True, separators=(",", ":")
        )

    @staticmethod
    def _parse_payload(payload: object) -> tuple[WorkflowSpec, MutationKind]:
        if not isinstance(payload, dict) or set(payload) != {
            "workflow_id",
            "mutation_kind",
            "nodes",
            "edges",
        }:
            raise ResearchValidationError("local Gemma proposal schema is invalid")
        workflow_id = payload["workflow_id"]
        if not isinstance(workflow_id, str) or not _is_safe_identifier(workflow_id):
            raise ResearchValidationError("workflow_id must be a bounded identifier")
        try:
            mutation = MutationKind(payload["mutation_kind"])
        except (TypeError, ValueError) as exc:
            raise ResearchValidationError("mutation_kind is invalid") from exc
        raw_nodes = payload["nodes"]
        raw_edges = payload["edges"]
        if not isinstance(raw_nodes, list) or not isinstance(raw_edges, list):
            raise ResearchValidationError("nodes and edges must be JSON arrays")
        nodes: list[NodeSpec] = []
        for raw_node in raw_nodes:
            if not isinstance(raw_node, dict) or set(raw_node) != {
                "node_id",
                "actions",
            }:
                raise ResearchValidationError("node schema is invalid")
            node_id = raw_node["node_id"]
            actions_payload = raw_node["actions"]
            if not isinstance(node_id, str) or not _is_safe_identifier(node_id):
                raise ResearchValidationError("node_id must be a bounded identifier")
            if not isinstance(actions_payload, list):
                raise ResearchValidationError("node actions must be a JSON array")
            actions: list[ActionSpec] = []
            for raw_action in actions_payload:
                if not isinstance(raw_action, dict) or set(raw_action) != {
                    "operator",
                    "arguments",
                }:
                    raise ResearchValidationError("action schema is invalid")
                try:
                    operator = WorkflowOperator(raw_action["operator"])
                except (TypeError, ValueError) as exc:
                    raise ResearchValidationError("action operator is invalid") from exc
                arguments = raw_action["arguments"]
                if not isinstance(arguments, dict) or any(
                    not isinstance(key, str) or not isinstance(value, str)
                    for key, value in arguments.items()
                ):
                    raise ResearchValidationError(
                        "action arguments must be a string JSON object"
                    )
                actions.append(ActionSpec(operator, tuple(sorted(arguments.items()))))
            nodes.append(NodeSpec(node_id, tuple(actions)))
        edges: list[EdgeSpec] = []
        for raw_edge in raw_edges:
            if not isinstance(raw_edge, dict) or set(raw_edge) != {
                "source",
                "target",
                "channel",
            }:
                raise ResearchValidationError("edge schema is invalid")
            values = (
                raw_edge["source"],
                raw_edge["target"],
                raw_edge["channel"],
            )
            if any(not isinstance(value, str) for value in values):
                raise ResearchValidationError("edge values must be strings")
            edges.append(EdgeSpec(*values))
        return WorkflowSpec(workflow_id, tuple(nodes), tuple(edges)), mutation


class BenchmarkSplit(str, Enum):
    HELD_IN = "held_in"
    HELD_OUT = "held_out"


@dataclass(frozen=True, slots=True)
class BenchmarkCase:
    """An immutable benchmark reference, not executable source text."""

    case_id: str
    split: BenchmarkSplit
    input_ref: str
    scoring_ref: str

    def __post_init__(self) -> None:
        for name in ("case_id", "input_ref", "scoring_ref"):
            if not _is_safe_identifier(getattr(self, name)):
                raise ValueError(f"{name} must be a bounded identifier")
        if type(self.split) is not BenchmarkSplit:
            raise ValueError("split must be a BenchmarkSplit")


@dataclass(frozen=True, slots=True)
class BenchmarkSuite:
    suite_id: str
    version: str
    cases: tuple[BenchmarkCase, ...]

    def __post_init__(self) -> None:
        if not _is_safe_identifier(self.suite_id) or not _is_safe_identifier(
            self.version
        ):
            raise ValueError("suite id and version must be bounded identifiers")
        if type(self.cases) is not tuple or not self.cases:
            raise ValueError("benchmark cases must be a non-empty tuple")
        ids = [case.case_id for case in self.cases]
        if len(set(ids)) != len(ids):
            raise ValueError("benchmark case ids must be unique")
        splits = {case.split for case in self.cases}
        if splits != {BenchmarkSplit.HELD_IN, BenchmarkSplit.HELD_OUT}:
            raise ValueError("suite requires held-in and held-out cases")

    @property
    def digest(self) -> str:
        return _sha256_json(
            {
                "suite_id": self.suite_id,
                "version": self.version,
                "cases": [
                    [
                        case.case_id,
                        case.split.value,
                        case.input_ref,
                        case.scoring_ref,
                    ]
                    for case in self.cases
                ],
            }
        )


@dataclass(frozen=True, slots=True)
class EvaluationPolicy:
    """Sealed repeat, resource, safety, and statistical promotion policy."""

    repeats: int = 5
    max_latency_ms: float = 10_000.0
    max_resource_units: float = 1_000.0
    max_tool_calls_per_sample: int = 32
    max_safety_violations: int = 0
    max_held_in_std: float = 0.20
    max_held_out_std: float = 0.20
    non_regression_tolerance: float = 0.0
    min_held_out_improvement: float = 0.0

    def __post_init__(self) -> None:
        if type(self.repeats) is not int or self.repeats < 5:
            raise ValueError("evaluation repeats must be at least five")
        for name in (
            "max_latency_ms",
            "max_resource_units",
            "max_held_in_std",
            "max_held_out_std",
            "non_regression_tolerance",
            "min_held_out_improvement",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
        for name in ("max_tool_calls_per_sample", "max_safety_violations"):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")

    @property
    def digest(self) -> str:
        return _sha256_json(
            {
                "repeats": self.repeats,
                "max_latency_ms": self.max_latency_ms,
                "max_resource_units": self.max_resource_units,
                "max_tool_calls_per_sample": self.max_tool_calls_per_sample,
                "max_safety_violations": self.max_safety_violations,
                "max_held_in_std": self.max_held_in_std,
                "max_held_out_std": self.max_held_out_std,
                "non_regression_tolerance": self.non_regression_tolerance,
                "min_held_out_improvement": self.min_held_out_improvement,
            }
        )


@dataclass(frozen=True, slots=True)
class EvaluatorAttestation:
    evaluator_id: str
    version: str
    artifact_sha256: str

    def __post_init__(self) -> None:
        if not _is_safe_identifier(self.evaluator_id) or not _is_safe_identifier(
            self.version
        ):
            raise ValueError("evaluator id and version must be bounded identifiers")
        if len(self.artifact_sha256) != 64 or any(
            character not in "0123456789abcdef" for character in self.artifact_sha256
        ):
            raise ValueError("artifact_sha256 must be a lowercase SHA-256 digest")

    @property
    def digest(self) -> str:
        return _sha256_json([self.evaluator_id, self.version, self.artifact_sha256])


@dataclass(frozen=True, slots=True)
class EvaluationObservation:
    """One evaluator observation for exactly one case and repeat."""

    case_id: str
    quality: float
    latency_ms: float
    resource_units: float
    tool_calls: int
    safety_violations: int = 0


class ObservationEvaluator(Protocol):
    def __call__(
        self, workflow: WorkflowSpec, case: BenchmarkCase, repeat_seed: int
    ) -> EvaluationObservation: ...


@dataclass(frozen=True, slots=True)
class SplitMetrics:
    samples: int
    quality_mean: float
    quality_std: float


@dataclass(frozen=True, slots=True)
class CandidateEvaluation:
    workflow_digest: str
    suite_digest: str
    policy_digest: str
    evaluator_digest: str
    samples: int
    repeats: int
    held_in: SplitMetrics
    held_out: SplitMetrics
    quality_mean: float
    quality_std: float
    latency_mean_ms: float
    latency_std_ms: float
    latency_max_ms: float
    resource_mean: float
    resource_std: float
    resource_max: float
    tool_calls_total: int
    tool_calls_max: int
    safety_violations: int
    trace_digest: str


@dataclass(frozen=True, slots=True)
class SealedCandidateEvaluator:
    """Own repeated evaluation and detect surface replacement or mutation."""

    suite: BenchmarkSuite
    policy: EvaluationPolicy
    attestation: EvaluatorAttestation
    evaluator: ObservationEvaluator = field(repr=False, compare=False)
    _seal: str = field(init=False, repr=False, compare=False)
    _callable_seal: str = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not callable(self.evaluator):
            raise ValueError("evaluator must be callable")
        object.__setattr__(
            self, "_callable_seal", _callable_fingerprint(self.evaluator)
        )
        object.__setattr__(self, "_seal", self._surface_digest())

    @property
    def seal(self) -> str:
        return self._seal

    def verify(self) -> None:
        if self._surface_digest() != self._seal:
            raise EvaluatorSurfaceTampered("evaluation surface digest changed")
        if _callable_fingerprint(self.evaluator) != self._callable_seal:
            raise EvaluatorSurfaceTampered("evaluator callable changed")

    def evaluate(self, workflow: WorkflowSpec, *, seed: int) -> CandidateEvaluation:
        self.verify()
        workflow_digest = digest_workflow(workflow)
        observations: list[tuple[BenchmarkSplit, EvaluationObservation]] = []
        raw_trace: list[list[object]] = []
        for case in self.suite.cases:
            for repeat in range(self.policy.repeats):
                repeat_seed = _derived_seed(seed, workflow_digest, case.case_id, repeat)
                observation = self.evaluator(workflow, case, repeat_seed)
                self.verify()
                _validate_observation(observation, case)
                observations.append((case.split, observation))
                raw_trace.append(
                    [
                        case.case_id,
                        case.split.value,
                        repeat,
                        repeat_seed,
                        observation.quality,
                        observation.latency_ms,
                        observation.resource_units,
                        observation.tool_calls,
                        observation.safety_violations,
                    ]
                )

        held_in_quality = [
            observation.quality
            for split, observation in observations
            if split is BenchmarkSplit.HELD_IN
        ]
        held_out_quality = [
            observation.quality
            for split, observation in observations
            if split is BenchmarkSplit.HELD_OUT
        ]
        all_quality = [observation.quality for _, observation in observations]
        latencies = [observation.latency_ms for _, observation in observations]
        resources = [observation.resource_units for _, observation in observations]
        tool_calls = [observation.tool_calls for _, observation in observations]
        safety = sum(observation.safety_violations for _, observation in observations)
        return CandidateEvaluation(
            workflow_digest=workflow_digest,
            suite_digest=self.suite.digest,
            policy_digest=self.policy.digest,
            evaluator_digest=self.attestation.digest,
            samples=len(observations),
            repeats=self.policy.repeats,
            held_in=_split_metrics(held_in_quality),
            held_out=_split_metrics(held_out_quality),
            quality_mean=statistics.fmean(all_quality),
            quality_std=statistics.pstdev(all_quality),
            latency_mean_ms=statistics.fmean(latencies),
            latency_std_ms=statistics.pstdev(latencies),
            latency_max_ms=max(latencies),
            resource_mean=statistics.fmean(resources),
            resource_std=statistics.pstdev(resources),
            resource_max=max(resources),
            tool_calls_total=sum(tool_calls),
            tool_calls_max=max(tool_calls),
            safety_violations=safety,
            trace_digest=_sha256_json(raw_trace),
        )

    def validate_result(self, result: CandidateEvaluation) -> None:
        self.verify()
        if result.suite_digest != self.suite.digest:
            raise EvaluatorSurfaceTampered("evaluation suite digest mismatch")
        if result.policy_digest != self.policy.digest:
            raise EvaluatorSurfaceTampered("evaluation policy digest mismatch")
        if result.evaluator_digest != self.attestation.digest:
            raise EvaluatorSurfaceTampered("evaluator attestation mismatch")
        expected_samples = len(self.suite.cases) * self.policy.repeats
        if result.repeats != self.policy.repeats or result.samples != expected_samples:
            raise EvaluatorSurfaceTampered("evaluation sample count mismatch")

    def _surface_digest(self) -> str:
        return _sha256_json(
            [self.suite.digest, self.policy.digest, self.attestation.digest]
        )


def _validate_observation(
    observation: EvaluationObservation, case: BenchmarkCase
) -> None:
    if not isinstance(observation, EvaluationObservation):
        raise ResearchValidationError("evaluator returned an invalid observation")
    if observation.case_id != case.case_id:
        raise ResearchValidationError("evaluator observation case mismatch")
    for name in ("quality", "latency_ms", "resource_units"):
        value = getattr(observation, name)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ResearchValidationError(f"{name} must be numeric")
        if not math.isfinite(float(value)):
            raise ResearchValidationError(f"{name} must be finite")
    if not 0.0 <= observation.quality <= 1.0:
        raise ResearchValidationError("quality must be in [0, 1]")
    if observation.latency_ms < 0.0 or observation.resource_units < 0.0:
        raise ResearchValidationError("latency and resources cannot be negative")
    for name in ("tool_calls", "safety_violations"):
        value = getattr(observation, name)
        if type(value) is not int or value < 0:
            raise ResearchValidationError(f"{name} must be a non-negative integer")


def _split_metrics(quality: Sequence[float]) -> SplitMetrics:
    return SplitMetrics(
        samples=len(quality),
        quality_mean=statistics.fmean(quality),
        quality_std=statistics.pstdev(quality),
    )


class PromotionTarget(str, Enum):
    RESEARCH_ARCHIVE_ONLY = "research_archive_only"


@dataclass(frozen=True, slots=True)
class ResearchSearchPolicy:
    top_k_parents: int = 5
    max_successful_traces: int = 8
    max_failed_traces: int = 8
    early_stop_patience: int = 8
    min_iterations_before_early_stop: int = 5
    lambda_mix: float = 0.2
    alpha: float = 0.4

    def __post_init__(self) -> None:
        for name in (
            "top_k_parents",
            "max_successful_traces",
            "max_failed_traces",
            "early_stop_patience",
        ):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if (
            type(self.min_iterations_before_early_stop) is not int
            or self.min_iterations_before_early_stop < 0
        ):
            raise ValueError(
                "min_iterations_before_early_stop must be a non-negative integer"
            )
        soft_mixed_probabilities((0.0,), lambda_mix=self.lambda_mix, alpha=self.alpha)

    @property
    def digest(self) -> str:
        return _sha256_json(
            {
                "top_k_parents": self.top_k_parents,
                "max_successful_traces": self.max_successful_traces,
                "max_failed_traces": self.max_failed_traces,
                "early_stop_patience": self.early_stop_patience,
                "min_iterations_before_early_stop": (
                    self.min_iterations_before_early_stop
                ),
                "lambda_mix": self.lambda_mix,
                "alpha": self.alpha,
            }
        )


@dataclass(frozen=True, slots=True)
class ResearchTraceSummary:
    trace_id: int
    workflow_id: str
    workflow_digest: str
    held_out_mean: float
    eligible: bool


@dataclass(frozen=True, slots=True)
class ResearchTrace:
    trace_id: int
    iteration: int
    workflow: WorkflowSpec | None
    workflow_digest: str | None
    parent_trace_id: int | None
    parent_workflow_digest: str | None
    lineage_seed: int
    mutation_kind: MutationKind | None
    evaluation: CandidateEvaluation | None
    eligible: bool
    promoted: bool
    failure_reason: str | None


@dataclass(frozen=True, slots=True)
class ResearchArchive:
    successful: tuple[ResearchTrace, ...]
    failed: tuple[ResearchTrace, ...]
    promoted_trace_ids: tuple[int, ...]
    promotion_target: PromotionTarget = PromotionTarget.RESEARCH_ARCHIVE_ONLY


@dataclass(frozen=True, slots=True)
class ResearchSearchResult:
    best: ResearchTrace
    baseline: ResearchTrace
    archive: ResearchArchive
    iterations_used: int
    evaluations: int
    stop_reason: str
    search_seed: int
    budget_digest: str
    search_policy_digest: str
    evaluator_seal: str
    replay_digest: str


@dataclass(frozen=True, slots=True)
class ResearchNightReport:
    """Night-cycle result; it contains no production promotion capability."""

    search: ResearchSearchResult
    resumed_inference: bool


@dataclass(slots=True)
class _CandidateState:
    trace: ResearchTrace
    visits: int = 1
    value_sum: float = 0.0

    @property
    def mean_value(self) -> float:
        return self.value_sum / self.visits


class ResearchAFlowExecutor:
    """Seeded, bounded search whose only promotion target is research archive."""

    def __init__(
        self,
        evaluator: SealedCandidateEvaluator,
        proposer: ResearchWorkflowProposer | None = None,
        *,
        budget: AFlowBudget | None = None,
        policy: ResearchSearchPolicy | None = None,
        seed: int = 0,
    ) -> None:
        if type(seed) is not int:
            raise ValueError("seed must be an integer")
        self.evaluator = evaluator
        self.proposer = proposer or GatedLocalGemmaProposer()
        self.budget = budget or AFlowBudget()
        self.policy = policy or ResearchSearchPolicy()
        self.seed = seed
        self.validator = ResearchWorkflowValidator(self.budget)
        self._evaluator_identity = id(self.evaluator)
        self._evaluator_seal = self.evaluator.seal
        self._proposer_seal = _callable_fingerprint(self.proposer)
        self._budget_seal = _digest_budget(self.budget)
        self._policy_seal = self.policy.digest
        if (
            self.policy.max_successful_traces + self.policy.max_failed_traces
            > self.budget.max_archive_entries
        ):
            raise ValueError(
                "successful and failed trace caps exceed archive entry budget"
            )

    def search(self, initial: WorkflowSpec) -> ResearchSearchResult:
        self._verify_surfaces()
        self.validator.validate(initial)
        try:
            baseline_evaluation = self.evaluator.evaluate(
                initial, seed=_derived_seed(self.seed, "baseline")
            )
            self.evaluator.validate_result(baseline_evaluation)
        except Exception as exc:
            raise ResearchEvaluationError(
                f"baseline evaluation failed: {type(exc).__name__}"
            ) from exc

        baseline = ResearchTrace(
            trace_id=0,
            iteration=0,
            workflow=initial,
            workflow_digest=baseline_evaluation.workflow_digest,
            parent_trace_id=None,
            parent_workflow_digest=None,
            lineage_seed=_derived_seed(self.seed, "baseline_lineage"),
            mutation_kind=None,
            evaluation=baseline_evaluation,
            eligible=True,
            promoted=True,
            failure_reason=None,
        )
        states = [
            _CandidateState(
                trace=baseline,
                value_sum=baseline_evaluation.held_out.quality_mean,
            )
        ]
        successful = [baseline]
        failed: list[ResearchTrace] = []
        best = baseline
        next_trace_id = 1
        evaluations = 1
        iterations_used = 0
        no_improvement = 0
        stop_reason = "iteration_budget_exhausted"
        selection_rng = random.Random(self.seed)
        replay_events: list[list[object]] = [
            [0, baseline.workflow_digest, baseline_evaluation.trace_digest]
        ]

        for iteration in range(1, self.budget.max_iterations + 1):
            self._verify_surfaces()
            if len(states) >= self.budget.max_archive_entries:
                stop_reason = "archive_budget_exhausted"
                break
            iterations_used = iteration
            parent_index = self._select_parent(states, selection_rng)
            parent = states[parent_index].trace
            assert parent.workflow is not None
            assert parent.workflow_digest is not None
            lineage_seed = _derived_seed(
                self.seed, "proposal", iteration, parent.workflow_digest
            )
            request = ProposalRequest(
                iteration=iteration,
                lineage_seed=lineage_seed,
                parent_trace_id=parent.trace_id,
                parent_workflow_digest=parent.workflow_digest,
                visible_archive=self._visible_archive(states),
            )
            proposal: WorkflowProposal | None = None
            try:
                proposal = self.proposer(parent.workflow, request)
                self._validate_proposal(proposal, parent, lineage_seed)
                self.validator.validate(proposal.workflow)
                if any(state.trace.workflow == proposal.workflow for state in states):
                    raise ResearchValidationError("duplicate_workflow")
                if any(
                    state.trace.workflow is not None
                    and state.trace.workflow.workflow_id
                    == proposal.workflow.workflow_id
                    for state in states
                ):
                    raise ResearchValidationError("duplicate_workflow_id")
            except Exception as exc:
                failure = ResearchTrace(
                    trace_id=next_trace_id,
                    iteration=iteration,
                    workflow=(
                        proposal.workflow
                        if isinstance(proposal, WorkflowProposal)
                        else None
                    ),
                    workflow_digest=None,
                    parent_trace_id=parent.trace_id,
                    parent_workflow_digest=parent.workflow_digest,
                    lineage_seed=lineage_seed,
                    mutation_kind=(
                        proposal.mutation_kind
                        if isinstance(proposal, WorkflowProposal)
                        else None
                    ),
                    evaluation=None,
                    eligible=False,
                    promoted=False,
                    failure_reason=f"proposal_rejected:{type(exc).__name__}:{exc}",
                )
                failed.append(failure)
                replay_events.append(
                    [iteration, parent.trace_id, lineage_seed, failure.failure_reason]
                )
                next_trace_id += 1
                no_improvement += 1
                if self._should_early_stop(iteration, no_improvement):
                    stop_reason = "early_stop_patience"
                    break
                continue

            try:
                evaluation = self.evaluator.evaluate(
                    proposal.workflow,
                    seed=_derived_seed(self.seed, "evaluation", iteration),
                )
                self.evaluator.validate_result(evaluation)
                evaluations += 1
                gate_failure = self._gate_failure(evaluation, baseline_evaluation)
            except Exception as exc:
                evaluation = None
                gate_failure = f"evaluation_error:{type(exc).__name__}:{exc}"

            workflow_digest = digest_workflow(proposal.workflow)
            if evaluation is None or gate_failure is not None:
                failure = ResearchTrace(
                    trace_id=next_trace_id,
                    iteration=iteration,
                    workflow=proposal.workflow,
                    workflow_digest=workflow_digest,
                    parent_trace_id=parent.trace_id,
                    parent_workflow_digest=parent.workflow_digest,
                    lineage_seed=lineage_seed,
                    mutation_kind=proposal.mutation_kind,
                    evaluation=evaluation,
                    eligible=False,
                    promoted=False,
                    failure_reason=gate_failure,
                )
                failed.append(failure)
                replay_events.append(
                    [
                        iteration,
                        parent.trace_id,
                        lineage_seed,
                        workflow_digest,
                        gate_failure,
                    ]
                )
                next_trace_id += 1
                no_improvement += 1
                if self._should_early_stop(iteration, no_improvement):
                    stop_reason = "early_stop_patience"
                    break
                continue

            promotes = (
                evaluation.held_out.quality_mean
                > best.evaluation.held_out.quality_mean
                + self.evaluator.policy.min_held_out_improvement
            )
            trace = ResearchTrace(
                trace_id=next_trace_id,
                iteration=iteration,
                workflow=proposal.workflow,
                workflow_digest=workflow_digest,
                parent_trace_id=parent.trace_id,
                parent_workflow_digest=parent.workflow_digest,
                lineage_seed=lineage_seed,
                mutation_kind=proposal.mutation_kind,
                evaluation=evaluation,
                eligible=True,
                promoted=promotes,
                failure_reason=None,
            )
            next_trace_id += 1
            state = _CandidateState(
                trace=trace,
                value_sum=evaluation.held_out.quality_mean,
            )
            states.append(state)
            successful.append(trace)
            self._backpropagate(states, parent.trace_id, evaluation)
            if promotes:
                best = trace
                no_improvement = 0
            else:
                no_improvement += 1
            replay_events.append(
                [
                    iteration,
                    parent.trace_id,
                    lineage_seed,
                    workflow_digest,
                    evaluation.trace_digest,
                    promotes,
                ]
            )
            if self._should_early_stop(iteration, no_improvement):
                stop_reason = "early_stop_patience"
                break

        successful_archive = self._top_successful(successful)
        archive = ResearchArchive(
            successful=successful_archive,
            failed=self._top_failed(failed),
            promoted_trace_ids=tuple(
                trace.trace_id for trace in successful_archive if trace.promoted
            ),
        )
        return ResearchSearchResult(
            best=best,
            baseline=baseline,
            archive=archive,
            iterations_used=iterations_used,
            evaluations=evaluations,
            stop_reason=stop_reason,
            search_seed=self.seed,
            budget_digest=self._budget_seal,
            search_policy_digest=self.policy.digest,
            evaluator_seal=self.evaluator.seal,
            replay_digest=_sha256_json(replay_events),
        )

    def _verify_surfaces(self) -> None:
        if (
            id(self.evaluator) != self._evaluator_identity
            or self.evaluator.seal != self._evaluator_seal
        ):
            raise EvaluatorSurfaceTampered("sealed evaluator was replaced")
        self.evaluator.verify()
        if _callable_fingerprint(self.proposer) != self._proposer_seal:
            raise EvaluatorSurfaceTampered("workflow proposer was replaced")
        if _digest_budget(self.budget) != self._budget_seal:
            raise EvaluatorSurfaceTampered("search budget digest changed")
        if self.policy.digest != self._policy_seal:
            raise EvaluatorSurfaceTampered("search policy digest changed")

    @staticmethod
    def _validate_proposal(
        proposal: WorkflowProposal,
        parent: ResearchTrace,
        lineage_seed: int,
    ) -> None:
        if not isinstance(proposal, WorkflowProposal):
            raise ResearchValidationError("proposer must return WorkflowProposal")
        if type(proposal.parent_workflow_ids) is not tuple:
            raise ResearchValidationError("proposal parents must be a tuple")
        if parent.workflow is None:
            raise ResearchValidationError("selected parent has no workflow")
        # A stricter single-parent rule is used for both mutation classes.  It
        # necessarily enforces the required single-parent major-change rule.
        if proposal.parent_workflow_ids != (parent.workflow.workflow_id,):
            raise ResearchValidationError("proposal must have exactly one parent")
        if proposal.lineage_seed != lineage_seed:
            raise ResearchValidationError("proposal lineage seed mismatch")
        if type(proposal.mutation_kind) is not MutationKind:
            raise ResearchValidationError("mutation kind must be typed")

    def _gate_failure(
        self,
        evaluation: CandidateEvaluation,
        baseline: CandidateEvaluation,
    ) -> str | None:
        policy = self.evaluator.policy
        if evaluation.safety_violations > policy.max_safety_violations:
            return "safety_gate"
        if evaluation.latency_max_ms > policy.max_latency_ms:
            return "latency_gate"
        if evaluation.resource_max > policy.max_resource_units:
            return "resource_gate"
        if evaluation.tool_calls_max > policy.max_tool_calls_per_sample:
            return "tool_gate"
        if evaluation.held_in.quality_std > policy.max_held_in_std:
            return "held_in_variance_gate"
        if evaluation.held_out.quality_std > policy.max_held_out_std:
            return "held_out_variance_gate"
        if (
            evaluation.held_out.quality_mean + policy.non_regression_tolerance
            < baseline.held_out.quality_mean
        ):
            return "held_out_non_regression_gate"
        return None

    def _select_parent(self, states: list[_CandidateState], rng: random.Random) -> int:
        ranked = sorted(
            range(len(states)),
            key=lambda index: (-states[index].mean_value, states[index].trace.trace_id),
        )
        candidates = ranked[: self.policy.top_k_parents]
        if 0 not in candidates:
            candidates.append(0)
        probabilities = soft_mixed_probabilities(
            [states[index].mean_value for index in candidates],
            lambda_mix=self.policy.lambda_mix,
            alpha=self.policy.alpha,
        )
        draw = rng.random()
        cumulative = 0.0
        for index, probability in zip(candidates, probabilities):
            cumulative += probability
            if draw < cumulative:
                return index
        return candidates[-1]

    @staticmethod
    def _backpropagate(
        states: list[_CandidateState],
        parent_trace_id: int,
        evaluation: CandidateEvaluation,
    ) -> None:
        by_trace_id = {state.trace.trace_id: state for state in states}
        current: int | None = parent_trace_id
        while current is not None:
            state = by_trace_id[current]
            state.visits += 1
            state.value_sum += evaluation.held_out.quality_mean
            current = state.trace.parent_trace_id

    @staticmethod
    def _visible_archive(
        states: list[_CandidateState],
    ) -> tuple[ResearchTraceSummary, ...]:
        return tuple(
            ResearchTraceSummary(
                trace_id=state.trace.trace_id,
                workflow_id=state.trace.workflow.workflow_id,
                workflow_digest=state.trace.workflow_digest,
                held_out_mean=state.trace.evaluation.held_out.quality_mean,
                eligible=state.trace.eligible,
            )
            for state in states
            if state.trace.workflow is not None
            and state.trace.workflow_digest is not None
            and state.trace.evaluation is not None
        )

    def _should_early_stop(self, iteration: int, no_improvement: int) -> bool:
        return (
            iteration >= self.policy.min_iterations_before_early_stop
            and no_improvement >= self.policy.early_stop_patience
        )

    def _top_successful(self, traces: list[ResearchTrace]) -> tuple[ResearchTrace, ...]:
        ranked = sorted(
            traces,
            key=lambda trace: (
                -trace.evaluation.held_out.quality_mean,
                trace.evaluation.held_out.quality_std,
                trace.trace_id,
            ),
        )
        return tuple(ranked[: self.policy.max_successful_traces])

    def _top_failed(self, traces: list[ResearchTrace]) -> tuple[ResearchTrace, ...]:
        ranked = sorted(traces, key=lambda trace: (-trace.iteration, trace.trace_id))
        return tuple(ranked[: self.policy.max_failed_traces])


class ResearchWorkflowCoordinator:
    """Run one bounded research search inside the legal night rhythm window."""

    def __init__(
        self,
        rhythm: RhythmController,
        executor: ResearchAFlowExecutor,
        checkpoint: Callable[[], None],
    ) -> None:
        self.rhythm = rhythm
        self.executor = executor
        self.checkpoint = checkpoint

    def run(self, initial: WorkflowSpec) -> ResearchNightReport:
        self.rhythm.enter_evolution(self.checkpoint)
        try:
            with self.rhythm.evolution_slot():
                result = self.executor.search(initial)
            self.rhythm.resume_inference("research workflow search completed")
            return ResearchNightReport(result, True)
        except Exception:
            if self.rhythm.mode == SystemMode.EVOLUTION:
                self.rhythm.resume_inference("research workflow search failed")
            raise


def digest_workflow(workflow: WorkflowSpec) -> str:
    """Canonical digest of a strict declarative workflow."""

    return _sha256_json(
        {
            "workflow_id": workflow.workflow_id,
            "nodes": [
                [
                    node.node_id,
                    [
                        [
                            (
                                action.operator.value
                                if isinstance(action.operator, WorkflowOperator)
                                else str(action.operator)
                            ),
                            list(action.arguments),
                        ]
                        for action in node.actions
                    ],
                ]
                for node in workflow.nodes
            ],
            "edges": [
                [edge.source, edge.target, edge.channel] for edge in workflow.edges
            ],
        }
    )


def _derived_seed(seed: int, *parts: object) -> int:
    digest = hashlib.sha256(
        json.dumps(
            [seed, *parts],
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:8], "big")


def _digest_budget(budget: AFlowBudget) -> str:
    return _sha256_json(
        {
            "max_nodes": budget.max_nodes,
            "max_edges": budget.max_edges,
            "max_actions": budget.max_actions,
            "max_iterations": budget.max_iterations,
            "max_archive_entries": budget.max_archive_entries,
            "max_control_chars": budget.max_control_chars,
        }
    )


def _sha256_json(value: object) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _callable_fingerprint(callback: Callable[..., object]) -> str:
    """Best-effort local seal; artifact attestation remains the trust root."""

    target = callback
    if not inspect.isfunction(target) and hasattr(target, "__call__"):
        target = target.__call__
    code = getattr(target, "__code__", None)
    payload = {
        "identity": id(callback),
        "module": getattr(callback, "__module__", type(callback).__module__),
        "qualname": getattr(callback, "__qualname__", type(callback).__qualname__),
        "code": code.co_code.hex() if code is not None else None,
        "consts": repr(code.co_consts) if code is not None else None,
        "defaults": repr(getattr(target, "__defaults__", None)),
        "kwdefaults": repr(getattr(target, "__kwdefaults__", None)),
        "declared_seal": (
            getattr(callback, "seal")
            if isinstance(getattr(callback, "seal", None), str)
            else None
        ),
    }
    return _sha256_json(payload)


__all__ = [
    "AttestedLocalGemmaWorkflowProposer",
    "BenchmarkCase",
    "BenchmarkSplit",
    "BenchmarkSuite",
    "CandidateEvaluation",
    "EvaluationObservation",
    "EvaluationPolicy",
    "EvaluatorAttestation",
    "EvaluatorSurfaceTampered",
    "GatedLocalGemmaProposer",
    "LocalGemmaProposerAttestation",
    "MutationKind",
    "PromotionTarget",
    "ProposalRequest",
    "ProposerGatedError",
    "ResearchAFlowExecutor",
    "ResearchArchive",
    "ResearchEvaluationError",
    "ResearchNightReport",
    "ResearchSearchPolicy",
    "ResearchSearchResult",
    "ResearchTrace",
    "ResearchValidationError",
    "ResearchWorkflowValidator",
    "ResearchWorkflowCoordinator",
    "SealedCandidateEvaluator",
    "SplitMetrics",
    "WorkflowOperator",
    "WorkflowProposal",
    "digest_workflow",
    # Re-export the immutable declarative schema for a single Phase-10 import.
    "ActionSpec",
    "EdgeSpec",
    "NodeSpec",
    "WorkflowSpec",
]
