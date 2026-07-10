"""Bounded, offline workflow search inspired by AFlow and ADAS.

The optimizer deliberately operates on declarative workflow specifications.  It
does not interpret action names or parameter values and it has no workflow
executor.  A trusted, injected evaluator owns sandboxed candidate execution.
This keeps workflow search in the control plane and makes the air-gap boundary
explicit.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import random
from typing import Protocol, Sequence


class WorkflowValidationError(ValueError):
    """Raised when a declarative workflow violates its schema or budget."""


class EvaluationError(RuntimeError):
    """Raised when the initial workflow cannot be scored."""


@dataclass(frozen=True, slots=True)
class ActionSpec:
    """An inert operator invocation understood only by an external evaluator.

    ``arguments`` is a tuple instead of a mapping so the entire specification
    remains deeply immutable and hashable.  Values are control-plane strings;
    the optimizer never parses or executes them.
    """

    operator: str
    arguments: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True, slots=True)
class NodeSpec:
    """A named workflow node containing one or more declarative actions."""

    node_id: str
    actions: tuple[ActionSpec, ...]


@dataclass(frozen=True, slots=True)
class EdgeSpec:
    """A directed dependency between two nodes.

    ``channel`` is metadata for the external runtime.  It is intentionally not
    treated as an executable condition by the optimizer.
    """

    source: str
    target: str
    channel: str = ""


@dataclass(frozen=True, slots=True)
class WorkflowSpec:
    """A complete, immutable candidate workflow."""

    workflow_id: str
    nodes: tuple[NodeSpec, ...] = ()
    edges: tuple[EdgeSpec, ...] = ()


@dataclass(frozen=True, slots=True)
class AFlowBudget:
    """Hard limits for a workflow and one optimizer run."""

    max_nodes: int = 16
    max_edges: int = 32
    max_actions: int = 32
    max_iterations: int = 32
    max_archive_entries: int = 64
    max_control_chars: int = 65_536

    def __post_init__(self) -> None:
        nonnegative = (
            "max_nodes",
            "max_edges",
            "max_actions",
            "max_iterations",
        )
        for name in nonnegative:
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        for name in ("max_archive_entries", "max_control_chars"):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")


@dataclass(frozen=True, slots=True)
class ValidationSummary:
    nodes: int
    edges: int
    actions: int
    roots: tuple[str, ...]
    leaves: tuple[str, ...]


class WorkflowValidator:
    """Validates immutable schema, DAG structure, and static resource limits."""

    def __init__(self, budget: AFlowBudget) -> None:
        self.budget = budget

    def validate(self, workflow: WorkflowSpec) -> ValidationSummary:
        if not isinstance(workflow, WorkflowSpec):
            raise WorkflowValidationError("candidate must be a WorkflowSpec")
        if type(workflow.nodes) is not tuple or type(workflow.edges) is not tuple:
            raise WorkflowValidationError("workflow nodes and edges must be tuples")

        control_chars = _text_size(workflow.workflow_id, "workflow_id")
        if len(workflow.nodes) > self.budget.max_nodes:
            raise WorkflowValidationError("node budget exceeded")
        if len(workflow.edges) > self.budget.max_edges:
            raise WorkflowValidationError("edge budget exceeded")

        node_ids: set[str] = set()
        action_count = 0
        for node in workflow.nodes:
            if not isinstance(node, NodeSpec):
                raise WorkflowValidationError("all nodes must be NodeSpec values")
            control_chars += _text_size(node.node_id, "node_id")
            if node.node_id in node_ids:
                raise WorkflowValidationError(f"duplicate node id: {node.node_id}")
            node_ids.add(node.node_id)
            if type(node.actions) is not tuple or not node.actions:
                raise WorkflowValidationError("each node must contain an action tuple")
            action_count += len(node.actions)
            if action_count > self.budget.max_actions:
                raise WorkflowValidationError("action budget exceeded")
            for action in node.actions:
                if not isinstance(action, ActionSpec):
                    raise WorkflowValidationError(
                        "all node actions must be ActionSpec values"
                    )
                control_chars += _text_size(action.operator, "action operator")
                if type(action.arguments) is not tuple:
                    raise WorkflowValidationError("action arguments must be a tuple")
                argument_names: set[str] = set()
                for argument in action.arguments:
                    if type(argument) is not tuple or len(argument) != 2:
                        raise WorkflowValidationError(
                            "action arguments must be (name, value) string pairs"
                        )
                    key, value = argument
                    control_chars += _text_size(key, "argument name")
                    control_chars += _text_size(
                        value, "argument value", allow_empty=True
                    )
                    if key in argument_names:
                        raise WorkflowValidationError(
                            f"duplicate action argument: {key}"
                        )
                    argument_names.add(key)

        indegree = {node_id: 0 for node_id in node_ids}
        outgoing: dict[str, list[str]] = {node_id: [] for node_id in node_ids}
        edge_keys: set[tuple[str, str, str]] = set()
        for edge in workflow.edges:
            if not isinstance(edge, EdgeSpec):
                raise WorkflowValidationError("all edges must be EdgeSpec values")
            control_chars += _text_size(edge.source, "edge source")
            control_chars += _text_size(edge.target, "edge target")
            control_chars += _text_size(edge.channel, "edge channel", allow_empty=True)
            if edge.source not in node_ids or edge.target not in node_ids:
                raise WorkflowValidationError("edge references an unknown node")
            edge_key = (edge.source, edge.target, edge.channel)
            if edge_key in edge_keys:
                raise WorkflowValidationError("duplicate edge")
            edge_keys.add(edge_key)
            outgoing[edge.source].append(edge.target)
            indegree[edge.target] += 1

        if control_chars > self.budget.max_control_chars:
            raise WorkflowValidationError("control-plane text budget exceeded")

        roots = tuple(
            sorted(node_id for node_id, degree in indegree.items() if degree == 0)
        )
        leaves = tuple(
            sorted(node_id for node_id, targets in outgoing.items() if not targets)
        )
        queue = list(roots)
        visited = 0
        cursor = 0
        while cursor < len(queue):
            node_id = queue[cursor]
            cursor += 1
            visited += 1
            for target in outgoing[node_id]:
                indegree[target] -= 1
                if indegree[target] == 0:
                    queue.append(target)
        if visited != len(node_ids):
            raise WorkflowValidationError("workflow graph must be acyclic")

        return ValidationSummary(
            nodes=len(workflow.nodes),
            edges=len(workflow.edges),
            actions=action_count,
            roots=roots,
            leaves=leaves,
        )


def _text_size(value: object, field: str, *, allow_empty: bool = False) -> int:
    if not isinstance(value, str):
        raise WorkflowValidationError(f"{field} must be a string")
    if not value and not allow_empty:
        raise WorkflowValidationError(f"{field} must not be empty")
    if "\x00" in value:
        raise WorkflowValidationError(f"{field} contains a null character")
    return len(value)


def soft_mixed_probabilities(
    scores: Sequence[float], *, lambda_mix: float = 0.2, alpha: float = 0.4
) -> tuple[float, ...]:
    """Return AFlow's stable soft mixed-probability distribution.

    ``P(i) = lambda_mix / n + (1 - lambda_mix) * softmax(alpha * score_i)``
    uses a max subtraction for numerical stability.  The defaults follow the
    values stated with Equation (3) in the AFlow paper.
    """

    if not scores:
        raise ValueError("scores must not be empty")
    if not math.isfinite(lambda_mix) or not 0.0 <= lambda_mix <= 1.0:
        raise ValueError("lambda_mix must be finite and in [0, 1]")
    if not math.isfinite(alpha) or alpha < 0.0:
        raise ValueError("alpha must be finite and non-negative")
    numeric_scores = tuple(float(score) for score in scores)
    if not all(math.isfinite(score) for score in numeric_scores):
        raise ValueError("scores must be finite")

    maximum = max(numeric_scores)
    weights = tuple(math.exp(alpha * (score - maximum)) for score in numeric_scores)
    weight_sum = math.fsum(weights)
    uniform = 1.0 / len(weights)
    probabilities = tuple(
        lambda_mix * uniform + (1.0 - lambda_mix) * weight / weight_sum
        for weight in weights
    )
    normalization = math.fsum(probabilities)
    return tuple(probability / normalization for probability in probabilities)


@dataclass(frozen=True, slots=True)
class Evaluation:
    """Score and inert audit metadata returned by an external evaluator."""

    score: float
    metadata: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True, slots=True)
class ArchiveRecord:
    archive_id: int
    workflow: WorkflowSpec
    score: float
    parent_id: int | None
    iteration: int
    visits: int
    value_sum: float

    @property
    def mean_value(self) -> float:
        return self.value_sum / self.visits


@dataclass(frozen=True, slots=True)
class ProposalContext:
    """Read-only tree experience supplied to a local proposer."""

    iteration: int
    seed: int
    parent: ArchiveRecord
    archive: tuple[ArchiveRecord, ...]


@dataclass(frozen=True, slots=True)
class RejectedCandidate:
    iteration: int
    parent_id: int
    workflow_id: str | None
    reason: str


@dataclass(frozen=True, slots=True)
class SearchResult:
    best: ArchiveRecord
    archive: tuple[ArchiveRecord, ...]
    promotions: tuple[int, ...]
    rejected: tuple[RejectedCandidate, ...]
    iterations_used: int
    evaluations: int
    exhausted: bool
    stop_reason: str

    @property
    def best_workflow(self) -> WorkflowSpec:
        return self.best.workflow

    @property
    def best_score(self) -> float:
        return self.best.score


class WorkflowProposer(Protocol):
    def __call__(
        self, parent: WorkflowSpec, context: ProposalContext
    ) -> WorkflowSpec: ...


class WorkflowEvaluator(Protocol):
    def __call__(self, workflow: WorkflowSpec) -> float | Evaluation: ...


@dataclass(slots=True)
class _MutableRecord:
    workflow: WorkflowSpec
    score: float
    parent_id: int | None
    iteration: int
    visits: int = 1
    value_sum: float = 0.0


class AFlowOptimizer:
    """Seeded, archive-backed search over complete declarative workflows.

    The proposer may construct specifications and the evaluator may execute a
    candidate in its own sandbox.  This class only validates, selects, archives,
    and backpropagates numeric scores.
    """

    def __init__(
        self,
        proposer: WorkflowProposer,
        evaluator: WorkflowEvaluator,
        *,
        budget: AFlowBudget | None = None,
        seed: int = 0,
        top_k: int = 5,
        lambda_mix: float = 0.2,
        alpha: float = 0.4,
    ) -> None:
        if type(seed) is not int:
            raise ValueError("seed must be an integer")
        if type(top_k) is not int or top_k < 1:
            raise ValueError("top_k must be a positive integer")
        # Validate probability parameters once without retaining a special case.
        soft_mixed_probabilities((0.0,), lambda_mix=lambda_mix, alpha=alpha)
        self.proposer = proposer
        self.evaluator = evaluator
        self.budget = budget or AFlowBudget()
        self.validator = WorkflowValidator(self.budget)
        self.seed = seed
        self.top_k = top_k
        self.lambda_mix = lambda_mix
        self.alpha = alpha

    def search(self, initial: WorkflowSpec) -> SearchResult:
        self.validator.validate(initial)
        initial_score = self._initial_score(initial)
        records = [
            _MutableRecord(
                workflow=initial,
                score=initial_score,
                parent_id=None,
                iteration=0,
                value_sum=initial_score,
            )
        ]
        seen = {initial}
        best_id = 0
        promotions = [0]
        rejected: list[RejectedCandidate] = []
        evaluations = 1
        iterations_used = 0
        stop_reason = "iteration_budget_exhausted"
        selection_rng = random.Random(self.seed)

        for iteration in range(1, self.budget.max_iterations + 1):
            if len(records) >= self.budget.max_archive_entries:
                stop_reason = "archive_budget_exhausted"
                break
            iterations_used = iteration
            parent_id = self._select_parent(records, selection_rng)
            archive_view = self._snapshot(records)
            context = ProposalContext(
                iteration=iteration,
                seed=self._proposal_seed(iteration),
                parent=archive_view[parent_id],
                archive=archive_view,
            )
            try:
                candidate = self.proposer(records[parent_id].workflow, context)
            except Exception as exc:
                rejected.append(
                    RejectedCandidate(
                        iteration,
                        parent_id,
                        None,
                        f"proposer_error:{type(exc).__name__}",
                    )
                )
                continue

            candidate_id = (
                candidate.workflow_id if isinstance(candidate, WorkflowSpec) else None
            )
            try:
                self.validator.validate(candidate)
            except WorkflowValidationError as exc:
                rejected.append(
                    RejectedCandidate(
                        iteration,
                        parent_id,
                        candidate_id,
                        f"invalid_workflow:{exc}",
                    )
                )
                continue
            if candidate in seen:
                rejected.append(
                    RejectedCandidate(
                        iteration, parent_id, candidate_id, "duplicate_workflow"
                    )
                )
                continue

            evaluations += 1
            try:
                candidate_score = self._score(self.evaluator(candidate))
            except Exception as exc:
                rejected.append(
                    RejectedCandidate(
                        iteration,
                        parent_id,
                        candidate_id,
                        f"evaluation_error:{type(exc).__name__}",
                    )
                )
                continue

            archive_id = len(records)
            records.append(
                _MutableRecord(
                    workflow=candidate,
                    score=candidate_score,
                    parent_id=parent_id,
                    iteration=iteration,
                    value_sum=candidate_score,
                )
            )
            seen.add(candidate)
            self._backpropagate(records, parent_id, candidate_score)
            if candidate_score > records[best_id].score:
                best_id = archive_id
                promotions.append(archive_id)

        archive = self._snapshot(records)
        return SearchResult(
            best=archive[best_id],
            archive=archive,
            promotions=tuple(promotions),
            rejected=tuple(rejected),
            iterations_used=iterations_used,
            evaluations=evaluations,
            exhausted=True,
            stop_reason=stop_reason,
        )

    def _initial_score(self, workflow: WorkflowSpec) -> float:
        try:
            return self._score(self.evaluator(workflow))
        except Exception as exc:
            raise EvaluationError(
                f"initial workflow evaluation failed: {type(exc).__name__}"
            ) from exc

    @staticmethod
    def _score(result: float | Evaluation) -> float:
        raw_score = result.score if isinstance(result, Evaluation) else result
        if isinstance(raw_score, bool):
            raise ValueError("evaluation score must be a finite real number")
        try:
            score = float(raw_score)
        except (TypeError, ValueError) as exc:
            raise ValueError("evaluation score must be a finite real number") from exc
        if not math.isfinite(score):
            raise ValueError("evaluation score must be finite")
        return score

    def _select_parent(self, records: list[_MutableRecord], rng: random.Random) -> int:
        ranked = sorted(
            range(len(records)),
            key=lambda index: (
                -(records[index].value_sum / records[index].visits),
                -records[index].score,
                index,
            ),
        )
        candidates = ranked[: self.top_k]
        if 0 not in candidates:
            candidates.append(0)
        scores = [
            records[index].value_sum / records[index].visits for index in candidates
        ]
        probabilities = soft_mixed_probabilities(
            scores, lambda_mix=self.lambda_mix, alpha=self.alpha
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
        records: list[_MutableRecord], parent_id: int, score: float
    ) -> None:
        current: int | None = parent_id
        while current is not None:
            record = records[current]
            record.visits += 1
            record.value_sum += score
            current = record.parent_id

    @staticmethod
    def _snapshot(records: list[_MutableRecord]) -> tuple[ArchiveRecord, ...]:
        return tuple(
            ArchiveRecord(
                archive_id=index,
                workflow=record.workflow,
                score=record.score,
                parent_id=record.parent_id,
                iteration=record.iteration,
                visits=record.visits,
                value_sum=record.value_sum,
            )
            for index, record in enumerate(records)
        )

    def _proposal_seed(self, iteration: int) -> int:
        # SplitMix64-style integer mixing keeps proposer randomness isolated from
        # the selection RNG and reproducible across optimizer runs.
        value = (self.seed + iteration * 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
        value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
        value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & 0xFFFFFFFFFFFFFFFF
        return value ^ (value >> 31)
