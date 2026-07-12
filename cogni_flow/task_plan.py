"""Fail-closed typed task planning and local Cogni-Flow execution.

The language model is never an authority at this boundary.  A caller must
submit an immutable :class:`TypedTaskPlan`, pass policy admission, obtain a
single-use capability, and only then may the local executor perform a bounded
operation.  T2 source changes are inert Self-Harness proposals; T3 actions are
not executable through this module.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from enum import Enum
from hashlib import sha256
import json
import os
from pathlib import Path, PurePosixPath
import re
import secrets
import shutil
import stat
import subprocess
import sys
from threading import Event, Lock, Thread
from time import monotonic, monotonic_ns
from typing import Callable, Protocol
import unicodedata

from .harness import PatchPolicy, PatchProposal


MAX_PLAN_ACTIONS = 32
MAX_PLAN_PATHS = 64
MAX_PLAN_INPUTS = 64
MAX_PLAN_ARTIFACTS = 32
MAX_OBJECTIVE_CHARS = 2_048
MAX_ACTION_TEXT_BYTES = 256 * 1024
MAX_ACTION_OUTPUT_BYTES = 1 * 1024 * 1024
MAX_READ_BYTES = 128 * 1024
MAX_SAVE_BYTES = 64 * 1024
MAX_SEARCH_FILES = 2_000
MAX_SEARCH_MATCHES = 100
MAX_LIST_ENTRIES = 200
MAX_CAPABILITIES = 128
CAPABILITY_TTL_SECONDS = 30.0
REPARSE_POINT_ATTRIBUTE = 0x400
ALLOWED_ARTIFACT_SUFFIXES = frozenset({".txt", ".md", ".json", ".csv"})
IGNORED_DIRECTORIES = frozenset(
    {
        ".git",
        ".pytest_cache",
        ".ruff_cache",
        "__pycache__",
        "build",
        "dist",
        "work",
    }
)
_PLAN_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{7,63}\Z")
_PERCENT_ESCAPE = re.compile(r"%[0-9A-Fa-f]{2}")
_WINDOWS_DEVICE = re.compile(
    r"(?i)(?:con|prn|aux|nul|clock\$|com[1-9]|lpt[1-9])(?:\..*)?\Z"
)


class TaskPlanError(ValueError):
    """Base class for a rejected task plan."""


class TaskPolicyError(TaskPlanError):
    """Raised when a plan requests authority that policy cannot grant."""


class CapabilityError(TaskPlanError):
    """Raised for missing, expired, mismatched, or replayed capabilities."""


class TaskExecutionError(RuntimeError):
    """Raised when an admitted local task fails its execution contract."""


class UnverifiedPlannerError(TaskPolicyError):
    """Raised when unverified natural-language output reaches admission."""


class RiskTier(str, Enum):
    T0 = "T0"
    T1 = "T1"
    T2 = "T2"
    T3 = "T3"


class TaskActionKind(str, Enum):
    HELP = "help"
    LIST = "list"
    READ = "read"
    SEARCH = "search"
    STATUS = "status"
    RUN_TEST = "run_test"
    WRITE_ARTIFACT = "write_artifact"
    STAGE_SOURCE_CHANGE = "stage_source_change"
    NETWORK = "network"
    ARBITRARY_SHELL = "arbitrary_shell"
    EVALUATOR_MUTATION = "evaluator_mutation"
    SECURITY_MUTATION = "security_mutation"
    UPDATER_MUTATION = "updater_mutation"
    ROLLBACK_MUTATION = "rollback_mutation"


class VerifierKind(str, Enum):
    ALL_ACTIONS = "all_actions"
    ARTIFACT_SHA256 = "artifact_sha256"
    PYTEST_PASS = "pytest_pass"
    PROPOSAL_STAGED = "proposal_staged"


ACTION_RISK: Mapping[TaskActionKind, RiskTier] = {
    TaskActionKind.HELP: RiskTier.T0,
    TaskActionKind.LIST: RiskTier.T0,
    TaskActionKind.READ: RiskTier.T0,
    TaskActionKind.SEARCH: RiskTier.T0,
    TaskActionKind.STATUS: RiskTier.T0,
    TaskActionKind.RUN_TEST: RiskTier.T1,
    TaskActionKind.WRITE_ARTIFACT: RiskTier.T1,
    TaskActionKind.STAGE_SOURCE_CHANGE: RiskTier.T2,
    TaskActionKind.NETWORK: RiskTier.T3,
    TaskActionKind.ARBITRARY_SHELL: RiskTier.T3,
    TaskActionKind.EVALUATOR_MUTATION: RiskTier.T3,
    TaskActionKind.SECURITY_MUTATION: RiskTier.T3,
    TaskActionKind.UPDATER_MUTATION: RiskTier.T3,
    TaskActionKind.ROLLBACK_MUTATION: RiskTier.T3,
}
_RISK_ORDER = {tier: index for index, tier in enumerate(RiskTier)}


@dataclass(frozen=True, slots=True)
class TaskBudget:
    time_seconds: float
    cpu_seconds: float
    ram_bytes: int
    vram_bytes: int
    max_output_bytes: int = 40_000


@dataclass(frozen=True, slots=True)
class RequiredInput:
    path: str
    sha256: str
    max_bytes: int = MAX_READ_BYTES


@dataclass(frozen=True, slots=True)
class ExpectedArtifact:
    path: str
    sha256: str
    max_bytes: int = MAX_SAVE_BYTES


@dataclass(frozen=True, slots=True)
class TaskVerifier:
    kind: VerifierKind
    minimum_passed_actions: int = 1


@dataclass(frozen=True, slots=True)
class TaskAction:
    action_id: str
    kind: TaskActionKind
    path: str = ""
    query: str = ""
    content: str = ""
    argv: tuple[str, ...] = ()
    expected_sha256: str = ""
    rationale: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.argv, tuple):
            raise TypeError("action argv must be an immutable tuple")


@dataclass(frozen=True, slots=True)
class TypedTaskPlan:
    plan_id: str
    objective: str
    actions: tuple[TaskAction, ...]
    allowed_paths: tuple[str, ...]
    required_inputs: tuple[RequiredInput, ...]
    expected_artifacts: tuple[ExpectedArtifact, ...]
    verifier: TaskVerifier
    budget: TaskBudget
    risk_tier: RiskTier
    schema_version: int = 1

    def __post_init__(self) -> None:
        for name in (
            "actions",
            "allowed_paths",
            "required_inputs",
            "expected_artifacts",
        ):
            if not isinstance(getattr(self, name), tuple):
                raise TypeError(f"{name} must be an immutable tuple")

    @property
    def digest(self) -> str:
        payload = {
            "schema_version": self.schema_version,
            "plan_id": self.plan_id,
            "objective": self.objective,
            "actions": [
                {
                    **asdict(action),
                    "kind": action.kind.value,
                    "argv": list(action.argv),
                }
                for action in self.actions
            ],
            "allowed_paths": list(self.allowed_paths),
            "required_inputs": [asdict(item) for item in self.required_inputs],
            "expected_artifacts": [asdict(item) for item in self.expected_artifacts],
            "verifier": {
                "kind": self.verifier.kind.value,
                "minimum_passed_actions": self.verifier.minimum_passed_actions,
            },
            "budget": asdict(self.budget),
            "risk_tier": self.risk_tier.value,
        }
        canonical = json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return sha256(canonical).hexdigest()


@dataclass(frozen=True, slots=True)
class CapabilityToken:
    value: str
    plan_sha256: str
    expires_at_ns: int


@dataclass(frozen=True, slots=True)
class VerifiedArtifact:
    path: str
    sha256: str
    size_bytes: int


@dataclass(frozen=True, slots=True)
class StagedPatchProposal:
    proposal_id: str
    plan_sha256: str
    proposal: PatchProposal


@dataclass(frozen=True, slots=True)
class TaskActionResult:
    action_id: str
    kind: TaskActionKind
    output: str
    duration_seconds: float
    artifact: VerifiedArtifact | None = None
    proposal_id: str | None = None


@dataclass(frozen=True, slots=True)
class TaskPlanResult:
    plan_id: str
    plan_sha256: str
    action_results: tuple[TaskActionResult, ...]
    artifacts: tuple[VerifiedArtifact, ...]
    proposal_ids: tuple[str, ...]
    verifier_passed: bool
    duration_seconds: float


class ProposalStager(Protocol):
    def stage(self, proposal: PatchProposal, plan_sha256: str) -> str: ...


class InMemoryProposalStager:
    """Bounded, inert hand-off queue for Self-Harness proposal-only mode."""

    def __init__(self, *, max_pending: int = 32) -> None:
        if not 1 <= max_pending <= 128:
            raise ValueError("max_pending must be in [1, 128]")
        self.max_pending = max_pending
        self._lock = Lock()
        self._pending: list[StagedPatchProposal] = []

    def stage(self, proposal: PatchProposal, plan_sha256: str) -> str:
        if not isinstance(proposal, PatchProposal):
            raise TypeError("proposal must be PatchProposal")
        if not _is_sha256(plan_sha256):
            raise ValueError("invalid plan digest")
        with self._lock:
            if len(self._pending) >= self.max_pending:
                raise TaskExecutionError("Self-Harness proposal queue is full")
            proposal_id = "proposal-" + secrets.token_hex(12)
            self._pending.append(
                StagedPatchProposal(proposal_id, plan_sha256, proposal)
            )
            return proposal_id

    @property
    def pending(self) -> tuple[StagedPatchProposal, ...]:
        with self._lock:
            return tuple(self._pending)


class TaskPlanPolicy:
    """Statically validate a plan before any capability can be minted."""

    def __init__(
        self,
        *,
        max_time_seconds: float = 900.0,
        max_cpu_seconds: float = 900.0,
        max_ram_bytes: int = 8 * 1024**3,
        max_output_bytes: int = MAX_ACTION_OUTPUT_BYTES,
        patch_policy: PatchPolicy | None = None,
    ) -> None:
        self.max_time_seconds = float(max_time_seconds)
        self.max_cpu_seconds = float(max_cpu_seconds)
        self.max_ram_bytes = int(max_ram_bytes)
        self.max_output_bytes = int(max_output_bytes)
        self.patch_policy = patch_policy or PatchPolicy()

    def validate(self, plan: TypedTaskPlan) -> None:
        if not isinstance(plan, TypedTaskPlan):
            raise TypeError("plan must be TypedTaskPlan")
        if type(plan.schema_version) is not int or plan.schema_version != 1:
            raise TaskPolicyError("unsupported task-plan schema")
        if not isinstance(plan.risk_tier, RiskTier):
            raise TaskPolicyError("risk tier must be typed")
        if not isinstance(plan.plan_id, str) or not _PLAN_ID.fullmatch(plan.plan_id):
            raise TaskPolicyError("invalid plan id")
        if (
            not isinstance(plan.objective, str)
            or not plan.objective.strip()
            or len(plan.objective) > MAX_OBJECTIVE_CHARS
        ):
            raise TaskPolicyError("objective is empty or exceeds its bound")
        if not 1 <= len(plan.actions) <= MAX_PLAN_ACTIONS:
            raise TaskPolicyError("action count is outside the fixed bound")
        if not 1 <= len(plan.allowed_paths) <= MAX_PLAN_PATHS:
            raise TaskPolicyError("allowed-path count is outside the fixed bound")
        if len(plan.required_inputs) > MAX_PLAN_INPUTS:
            raise TaskPolicyError("required-input count exceeds the fixed bound")
        if len(plan.expected_artifacts) > MAX_PLAN_ARTIFACTS:
            raise TaskPolicyError("artifact count exceeds the fixed bound")
        allowed = tuple(
            _normalize_relative(path, allow_root=True) for path in plan.allowed_paths
        )
        if len(set(allowed)) != len(allowed):
            raise TaskPolicyError("allowed paths must be unique")
        self._validate_budget(plan.budget)

        ids: set[str] = set()
        observed_risk = RiskTier.T0
        for action in plan.actions:
            if not isinstance(action, TaskAction):
                raise TaskPolicyError("actions must be typed TaskAction values")
            if not all(
                isinstance(value, str)
                for value in (
                    action.action_id,
                    action.path,
                    action.query,
                    action.content,
                    action.expected_sha256,
                    action.rationale,
                )
            ):
                raise TaskPolicyError("action text fields must be strings")
            if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", action.action_id):
                raise TaskPolicyError("invalid action id")
            if action.action_id in ids:
                raise TaskPolicyError("action ids must be unique")
            ids.add(action.action_id)
            if not isinstance(action.kind, TaskActionKind):
                raise TaskPolicyError("unknown action kind")
            risk = ACTION_RISK[action.kind]
            if _RISK_ORDER[risk] > _RISK_ORDER[observed_risk]:
                observed_risk = risk
            self._validate_action(action, allowed)

        if observed_risk is not plan.risk_tier:
            raise TaskPolicyError("risk tier does not exactly match the action set")
        if observed_risk is RiskTier.T3:
            raise TaskPolicyError("T3 actions are permanently forbidden")

        for required in plan.required_inputs:
            if not isinstance(required, RequiredInput):
                raise TaskPolicyError("required inputs must be typed")
            if (
                not isinstance(required.path, str)
                or not isinstance(required.sha256, str)
                or isinstance(required.max_bytes, bool)
                or not isinstance(required.max_bytes, int)
            ):
                raise TaskPolicyError("required input fields have invalid types")
            path = _normalize_relative(required.path)
            if not _within_allowed(path, allowed):
                raise TaskPolicyError("required input is outside allowed paths")
            if not _is_sha256(required.sha256):
                raise TaskPolicyError("required input must carry a SHA-256 digest")
            if not 1 <= required.max_bytes <= MAX_ACTION_TEXT_BYTES:
                raise TaskPolicyError("required-input bound is invalid")

        artifact_paths: set[str] = set()
        for artifact in plan.expected_artifacts:
            if not isinstance(artifact, ExpectedArtifact):
                raise TaskPolicyError("expected artifacts must be typed")
            if (
                not isinstance(artifact.path, str)
                or not isinstance(artifact.sha256, str)
                or isinstance(artifact.max_bytes, bool)
                or not isinstance(artifact.max_bytes, int)
            ):
                raise TaskPolicyError("artifact fields have invalid types")
            path = _normalize_relative(artifact.path)
            self._validate_artifact_path(path)
            if path in artifact_paths:
                raise TaskPolicyError("artifact paths must be unique")
            artifact_paths.add(path)
            if not _within_allowed(path, allowed):
                raise TaskPolicyError("artifact is outside allowed paths")
            if not _is_sha256(artifact.sha256):
                raise TaskPolicyError("artifact must carry a SHA-256 digest")
            if not 1 <= artifact.max_bytes <= MAX_SAVE_BYTES:
                raise TaskPolicyError("artifact bound is invalid")

        write_paths = {
            _normalize_relative(action.path)
            for action in plan.actions
            if action.kind is TaskActionKind.WRITE_ARTIFACT
        }
        if write_paths != artifact_paths:
            raise TaskPolicyError(
                "write actions and expected artifacts must have an exact mapping"
            )
        write_actions = [
            action
            for action in plan.actions
            if action.kind is TaskActionKind.WRITE_ARTIFACT
        ]
        if len(write_actions) != len(write_paths):
            raise TaskPolicyError("artifact paths cannot be written more than once")
        artifact_specs = {
            _normalize_relative(item.path): item for item in plan.expected_artifacts
        }
        for action in write_actions:
            action_path = _normalize_relative(action.path)
            spec = artifact_specs[action_path]
            encoded = action.content.encode("utf-8")
            if (
                len(encoded) > spec.max_bytes
                or sha256(encoded).hexdigest() != spec.sha256
            ):
                raise TaskPolicyError(
                    "artifact content does not match its declared size/SHA-256"
                )
        self._validate_verifier(plan)

    def _validate_budget(self, budget: TaskBudget) -> None:
        if not isinstance(budget, TaskBudget):
            raise TaskPolicyError("budget must be TaskBudget")
        if (
            isinstance(budget.time_seconds, bool)
            or not isinstance(budget.time_seconds, (int, float))
            or isinstance(budget.cpu_seconds, bool)
            or not isinstance(budget.cpu_seconds, (int, float))
            or isinstance(budget.ram_bytes, bool)
            or not isinstance(budget.ram_bytes, int)
            or isinstance(budget.vram_bytes, bool)
            or not isinstance(budget.vram_bytes, int)
            or isinstance(budget.max_output_bytes, bool)
            or not isinstance(budget.max_output_bytes, int)
        ):
            raise TaskPolicyError("budget fields have invalid types")
        if not 0.05 <= budget.time_seconds <= self.max_time_seconds:
            raise TaskPolicyError("wall-clock budget is invalid")
        if not 0.01 <= budget.cpu_seconds <= self.max_cpu_seconds:
            raise TaskPolicyError("CPU budget is invalid")
        if not 16 * 1024**2 <= budget.ram_bytes <= self.max_ram_bytes:
            raise TaskPolicyError("RAM budget is invalid")
        if budget.vram_bytes != 0:
            raise TaskPolicyError("local task plans cannot own VRAM")
        if not 1 <= budget.max_output_bytes <= self.max_output_bytes:
            raise TaskPolicyError("output budget is invalid")

    def _validate_action(
        self, action: TaskAction, allowed_paths: tuple[str, ...]
    ) -> None:
        encoded = (action.query + action.content + action.rationale).encode("utf-8")
        if len(encoded) > MAX_ACTION_TEXT_BYTES or b"\x00" in encoded:
            raise TaskPolicyError("action text is binary or exceeds its bound")
        if action.argv:
            if action.kind is not TaskActionKind.RUN_TEST:
                raise TaskPolicyError("argv is accepted only by fixed test actions")
            if any(
                not isinstance(item, str) or not item or "\x00" in item
                for item in action.argv
            ):
                raise TaskPolicyError("invalid test argv")
        if action.kind in {TaskActionKind.HELP, TaskActionKind.STATUS}:
            if any((action.path, action.query, action.content, action.argv)):
                raise TaskPolicyError("parameterless action carries extra fields")
            return

        path = _normalize_relative(
            action.path,
            allow_root=action.kind in {TaskActionKind.LIST, TaskActionKind.SEARCH},
        )
        if not _within_allowed(path, allowed_paths):
            raise TaskPolicyError("action path is outside the declared capability")
        if action.kind in {TaskActionKind.LIST, TaskActionKind.READ}:
            if any((action.query, action.content, action.argv)):
                raise TaskPolicyError("read action carries unrelated fields")
        elif action.kind is TaskActionKind.SEARCH:
            if not action.query or len(action.query) > 256:
                raise TaskPolicyError("search query is empty or too long")
            if any(ord(char) < 32 for char in action.query):
                raise TaskPolicyError("search query contains control characters")
            if action.content or action.argv:
                raise TaskPolicyError("search action carries unrelated fields")
        elif action.kind is TaskActionKind.RUN_TEST:
            if action.content or action.query:
                raise TaskPolicyError("test action carries unrelated text")
            if path != "tests" and not (
                path.startswith("tests/") and path.endswith(".py")
            ):
                raise TaskPolicyError("tests are restricted to tests/*.py")
            expected_argv = (sys.executable, "-m", "pytest", path, "-q")
            if action.argv != expected_argv:
                raise TaskPolicyError("test argv is not the fixed allowlisted argv")
        elif action.kind is TaskActionKind.WRITE_ARTIFACT:
            self._validate_artifact_path(path)
            data = action.content.encode("utf-8")
            if not data or len(data) > MAX_SAVE_BYTES or b"\x00" in data:
                raise TaskPolicyError("artifact payload is empty, binary, or too large")
            if (
                action.argv
                or action.query
                or action.expected_sha256
                or action.rationale
            ):
                raise TaskPolicyError("artifact action carries unrelated fields")
        elif action.kind is TaskActionKind.STAGE_SOURCE_CHANGE:
            if action.argv or action.query:
                raise TaskPolicyError("source proposal cannot carry argv or query")
            if not action.content or not action.rationale.strip():
                raise TaskPolicyError(
                    "source proposal requires replacement and rationale"
                )
            if not _is_sha256(action.expected_sha256):
                raise TaskPolicyError("source proposal requires a base SHA-256")
            proposal = PatchProposal(
                path,
                action.expected_sha256,
                action.content,
                action.rationale,
            )
            self.patch_policy.validate(proposal)
        elif ACTION_RISK[action.kind] is RiskTier.T3:
            raise TaskPolicyError("T3 action is permanently forbidden")

    @staticmethod
    def _validate_artifact_path(path: str) -> None:
        prefix = "outputs/agent-workspace/"
        if not path.startswith(prefix):
            raise TaskPolicyError("artifacts are confined to outputs/agent-workspace")
        name = path.removeprefix(prefix)
        if (
            "/" in name
            or not name
            or Path(name).suffix.lower() not in ALLOWED_ARTIFACT_SUFFIXES
        ):
            raise TaskPolicyError("artifact must be one safe filename")

    @staticmethod
    def _validate_verifier(plan: TypedTaskPlan) -> None:
        verifier = plan.verifier
        if not isinstance(verifier, TaskVerifier):
            raise TaskPolicyError("verifier must be typed")
        if not isinstance(verifier.kind, VerifierKind):
            raise TaskPolicyError("verifier kind is not recognized")
        if isinstance(verifier.minimum_passed_actions, bool) or not isinstance(
            verifier.minimum_passed_actions, int
        ):
            raise TaskPolicyError("verifier action threshold must be an integer")
        if not 1 <= verifier.minimum_passed_actions <= len(plan.actions):
            raise TaskPolicyError("verifier action threshold is invalid")
        kinds = {action.kind for action in plan.actions}
        if (
            verifier.kind is VerifierKind.ARTIFACT_SHA256
            and not plan.expected_artifacts
        ):
            raise TaskPolicyError("artifact verifier requires expected artifacts")
        if (
            verifier.kind is VerifierKind.PYTEST_PASS
            and TaskActionKind.RUN_TEST not in kinds
        ):
            raise TaskPolicyError("pytest verifier requires a fixed test action")
        if (
            verifier.kind is VerifierKind.PROPOSAL_STAGED
            and TaskActionKind.STAGE_SOURCE_CHANGE not in kinds
        ):
            raise TaskPolicyError("proposal verifier requires a staged source action")


class UnverifiedPlannerGate:
    """Explicitly refuse model-produced/NL plans until a planner is certified."""

    def __init__(self, policy: TaskPlanPolicy | None = None) -> None:
        self.policy = policy or TaskPlanPolicy()

    def admit_typed(self, candidate: TypedTaskPlan) -> TypedTaskPlan:
        if not isinstance(candidate, TypedTaskPlan):
            raise UnverifiedPlannerError(
                "only an immutable TypedTaskPlan is admissible"
            )
        self.policy.validate(candidate)
        return candidate

    def plan_natural_language(self, _request: str) -> TypedTaskPlan:
        raise UnverifiedPlannerError(
            "natural-language planner is gated pending deterministic certification"
        )


@dataclass(frozen=True, slots=True)
class _FileIdentity:
    device: int
    inode: int
    mode: int
    size: int
    mtime_ns: int
    attributes: int


@dataclass(slots=True)
class _CapabilityRecord:
    digest: str
    expires_at_ns: int


class TaskPlanExecutor:
    """Execute admitted T0/T1 plans and stage T2 proposals, never T3."""

    def __init__(
        self,
        project_root: str | Path,
        *,
        policy: TaskPlanPolicy | None = None,
        proposal_stager: ProposalStager | None = None,
        help_text: str = "Bounded local task executor",
        clock_ns: Callable[[], int] = monotonic_ns,
    ) -> None:
        supplied_root = Path(project_root)
        root = supplied_root.resolve(strict=True)
        if not root.is_dir():
            raise ValueError("project_root must be a directory")
        self.project_root = root
        self.policy = policy or TaskPlanPolicy()
        self.proposal_stager = proposal_stager
        self.help_text = help_text
        if not callable(clock_ns):
            raise TypeError("clock_ns must be callable")
        self._clock_ns = clock_ns
        self._lock = Lock()
        self._capability_lock = Lock()
        self._capabilities: dict[str, _CapabilityRecord] = {}
        self._python_executable = Path(sys.executable).resolve(strict=True)
        self._python_identity = self._identity(self._python_executable)
        if not stat.S_ISREG(self._python_identity.mode):
            raise ValueError("Python runtime is not a regular executable")
        located_git = shutil.which("git")
        self._git_executable = (
            Path(located_git).resolve(strict=True) if located_git is not None else None
        )
        self._git_identity = (
            self._identity(self._git_executable)
            if self._git_executable is not None
            else None
        )

    def authorize(
        self,
        plan: TypedTaskPlan,
        *,
        ttl_seconds: float = CAPABILITY_TTL_SECONDS,
    ) -> CapabilityToken:
        self.policy.validate(plan)
        if plan.risk_tier is RiskTier.T2 and self.proposal_stager is None:
            raise TaskPolicyError(
                "T2 requires an attached Self-Harness proposal stager"
            )
        if not 0.05 <= ttl_seconds <= 300.0:
            raise ValueError("capability TTL must be in [0.05, 300] seconds")
        now = self._clock_ns()
        if not isinstance(now, int) or isinstance(now, bool) or now < 0:
            raise RuntimeError("capability clock returned an invalid timestamp")
        expires = now + int(ttl_seconds * 1_000_000_000)
        with self._capability_lock:
            self._purge_capabilities(now)
            if len(self._capabilities) >= MAX_CAPABILITIES:
                raise CapabilityError("capability table is full")
            value = secrets.token_urlsafe(32)
            self._capabilities[value] = _CapabilityRecord(plan.digest, expires)
        return CapabilityToken(value, plan.digest, expires)

    def revoke(self, token: CapabilityToken) -> None:
        if not isinstance(token, CapabilityToken):
            raise TypeError("token must be CapabilityToken")
        with self._capability_lock:
            self._capabilities.pop(token.value, None)

    def execute(
        self, plan: TypedTaskPlan, capability: CapabilityToken
    ) -> TaskPlanResult:
        self.policy.validate(plan)
        self._consume_capability(plan, capability)
        started = monotonic()
        deadline = started + plan.budget.time_seconds
        results: list[TaskActionResult] = []
        artifacts: list[VerifiedArtifact] = []
        proposal_ids: list[str] = []
        initial_inputs: dict[str, str] = {}
        with self._lock:
            for item in plan.required_inputs:
                digest = self._hash_required_input(item)
                if digest != item.sha256:
                    raise TaskExecutionError(
                        f"required input digest mismatch: {item.path}"
                    )
                initial_inputs[item.path] = digest

            for action in plan.actions:
                if monotonic() >= deadline:
                    raise TaskExecutionError("task wall-clock budget exhausted")
                action_started = monotonic()
                output, artifact, proposal_id = self._execute_action(
                    action,
                    plan,
                    deadline,
                )
                if len(output.encode("utf-8")) > plan.budget.max_output_bytes:
                    raise TaskExecutionError("action output exceeded the plan bound")
                result = TaskActionResult(
                    action.action_id,
                    action.kind,
                    output,
                    monotonic() - action_started,
                    artifact,
                    proposal_id,
                )
                results.append(result)
                if artifact is not None:
                    artifacts.append(artifact)
                if proposal_id is not None:
                    proposal_ids.append(proposal_id)

            for item in plan.required_inputs:
                if self._hash_required_input(item) != initial_inputs[item.path]:
                    raise TaskExecutionError(
                        f"required input changed during task: {item.path}"
                    )

            self._verify(plan, results, artifacts, proposal_ids)

        return TaskPlanResult(
            plan.plan_id,
            plan.digest,
            tuple(results),
            tuple(artifacts),
            tuple(proposal_ids),
            True,
            monotonic() - started,
        )

    def _execute_action(
        self,
        action: TaskAction,
        plan: TypedTaskPlan,
        deadline: float,
    ) -> tuple[str, VerifiedArtifact | None, str | None]:
        if action.kind is TaskActionKind.HELP:
            return self.help_text, None, None
        if action.kind is TaskActionKind.LIST:
            return self._list(action.path), None, None
        if action.kind is TaskActionKind.READ:
            data, _digest = self._secure_read(action.path, MAX_READ_BYTES)
            return data.decode("utf-8", errors="strict"), None, None
        if action.kind is TaskActionKind.SEARCH:
            return self._search(action.query, action.path), None, None
        if action.kind is TaskActionKind.STATUS:
            argv = self._git_status_argv()
            timeout = min(30.0, max(0.01, deadline - monotonic()))
            output, returncode = self._run_bounded_process(
                argv,
                timeout,
                plan.budget.max_output_bytes,
                plan.budget.cpu_seconds,
                plan.budget.ram_bytes,
                status_environment=True,
            )
            if returncode != 0:
                raise TaskExecutionError("fixed git status command failed")
            return output.strip() or "working tree clean", None, None
        if action.kind is TaskActionKind.RUN_TEST:
            test_path, test_identity = self._secure_resolve(
                action.path, must_exist=True
            )
            assert test_identity is not None
            timeout = min(plan.budget.time_seconds, max(0.01, deadline - monotonic()))
            output, returncode = self._run_bounded_process(
                action.argv,
                timeout,
                plan.budget.max_output_bytes,
                plan.budget.cpu_seconds,
                plan.budget.ram_bytes,
                status_environment=False,
            )
            if returncode != 0:
                raise TaskExecutionError(
                    f"fixed pytest action failed ({returncode})\n{output[-20_000:]}"
                )
            if not self._same_identity(test_identity, self._identity(test_path)):
                raise TaskExecutionError("test target changed during execution")
            return output.strip(), None, None
        if action.kind is TaskActionKind.WRITE_ARTIFACT:
            artifact = self._write_artifact(action.path, action.content)
            return (
                f"saved: {artifact.path} ({artifact.size_bytes} bytes, sha256={artifact.sha256})",
                artifact,
                None,
            )
        if action.kind is TaskActionKind.STAGE_SOURCE_CHANGE:
            if self.proposal_stager is None:
                raise TaskExecutionError("Self-Harness proposal stager is unavailable")
            data, digest = self._secure_read(action.path, MAX_ACTION_TEXT_BYTES)
            if digest != action.expected_sha256:
                raise TaskExecutionError("source changed before proposal staging")
            data.decode("utf-8", errors="strict")
            proposal = PatchProposal(
                action.path,
                digest,
                action.content,
                action.rationale,
            )
            self.policy.patch_policy.validate(proposal)
            proposal_id = self.proposal_stager.stage(proposal, plan.digest)
            # Source is re-read after hand-off: staging must not mutate it.
            _after, after_digest = self._secure_read(action.path, MAX_ACTION_TEXT_BYTES)
            if after_digest != digest:
                raise TaskExecutionError("proposal staging mutated the source tree")
            return f"Self-Harness proposal staged: {proposal_id}", None, proposal_id
        raise TaskPolicyError("action cannot be executed by the local task boundary")

    def _consume_capability(
        self, plan: TypedTaskPlan, capability: CapabilityToken
    ) -> None:
        if not isinstance(capability, CapabilityToken):
            raise CapabilityError("a typed capability token is required")
        now = self._clock_ns()
        if not isinstance(now, int) or isinstance(now, bool) or now < 0:
            raise RuntimeError("capability clock returned an invalid timestamp")
        with self._capability_lock:
            self._purge_capabilities(now)
            record = self._capabilities.pop(capability.value, None)
        if record is None:
            raise CapabilityError(
                "capability is unknown, expired, revoked, or replayed"
            )
        if now >= record.expires_at_ns or now >= capability.expires_at_ns:
            raise CapabilityError("capability expired")
        if (
            record.digest != plan.digest
            or capability.plan_sha256 != plan.digest
            or capability.expires_at_ns != record.expires_at_ns
        ):
            raise CapabilityError("capability does not authorize this exact plan")

    def _purge_capabilities(self, now_ns: int) -> None:
        expired = [
            value
            for value, record in self._capabilities.items()
            if now_ns >= record.expires_at_ns
        ]
        for value in expired:
            self._capabilities.pop(value, None)

    def _hash_required_input(self, item: RequiredInput) -> str:
        _data, digest = self._secure_read(item.path, item.max_bytes)
        return digest

    def _secure_resolve(
        self,
        raw: str,
        *,
        must_exist: bool,
        require_file: bool = False,
        require_directory: bool = False,
    ) -> tuple[Path, _FileIdentity | None]:
        relative = _normalize_relative(raw, allow_root=True)
        candidate = (
            self.project_root
            if relative == "."
            else self.project_root.joinpath(*relative.split("/"))
        )
        if not _lexically_within(candidate, self.project_root):
            raise TaskPolicyError("path escaped the project root")
        current = self.project_root
        identity: _FileIdentity | None = self._identity(current)
        ancestors: list[tuple[Path, _FileIdentity]] = [(current, identity)]
        for part in () if relative == "." else relative.split("/"):
            current = current / part
            try:
                identity = self._identity(current)
            except FileNotFoundError:
                identity = None
                if must_exist or current != candidate:
                    raise TaskPolicyError("path does not exist") from None
                break
            if self._is_linklike(current, identity):
                raise TaskPolicyError(
                    "symbolic-link, junction, or reparse path rejected"
                )
            ancestors.append((current, identity))
        if identity is None:
            return candidate, None
        resolved = current.resolve(strict=True)
        if not _same_path(resolved, current.absolute()) or not _lexically_within(
            resolved, self.project_root
        ):
            raise TaskPolicyError("canonical path escaped or changed identity")
        for ancestor, expected in ancestors:
            if not self._same_identity(expected, self._identity(ancestor)):
                raise TaskExecutionError(
                    "path ancestry changed during canonicalization"
                )
        if require_file and not stat.S_ISREG(identity.mode):
            raise TaskPolicyError("path is not a regular file")
        if require_directory and not stat.S_ISDIR(identity.mode):
            raise TaskPolicyError("path is not a directory")
        return resolved, identity

    def _secure_read(self, raw: str, max_bytes: int) -> tuple[bytes, str]:
        path, before = self._secure_resolve(raw, must_exist=True, require_file=True)
        assert before is not None
        flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(path, flags)
        try:
            opened = self._identity_from_stat(os.fstat(descriptor))
            if not self._same_identity(before, opened):
                raise TaskExecutionError("path changed before secure open")
            chunks: list[bytes] = []
            remaining = max_bytes + 1
            while remaining > 0:
                chunk = os.read(descriptor, min(64 * 1024, remaining))
                if not chunk:
                    break
                chunks.append(chunk)
                remaining -= len(chunk)
            data = b"".join(chunks)
            if len(data) > max_bytes:
                raise TaskExecutionError("file exceeds its bounded read limit")
            after_fd = self._identity_from_stat(os.fstat(descriptor))
            if not self._same_identity(opened, after_fd):
                raise TaskExecutionError("file changed during secure read")
        finally:
            os.close(descriptor)
        after_path = self._identity(path)
        if not self._same_identity(before, after_path):
            raise TaskExecutionError("path changed during secure read")
        if b"\x00" in data:
            raise TaskExecutionError("binary files are not exposed to the task plane")
        return data, sha256(data).hexdigest()

    def _list(self, raw: str) -> str:
        path, before = self._secure_resolve(
            raw, must_exist=True, require_directory=True
        )
        assert before is not None
        entries: list[str] = []
        with os.scandir(path) as iterator:
            for entry in iterator:
                if entry.name in IGNORED_DIRECTORIES:
                    continue
                child = path / entry.name
                child_identity = self._identity(child)
                if self._is_linklike(child, child_identity):
                    entries.append(f"[blocked-link] {entry.name}")
                else:
                    kind = "dir" if stat.S_ISDIR(child_identity.mode) else "file"
                    entries.append(f"[{kind}] {entry.name}")
                if len(entries) >= MAX_LIST_ENTRIES:
                    entries.append("[entry limit reached]")
                    break
        after = self._identity(path)
        if not self._same_identity(before, after):
            raise TaskExecutionError("directory changed during bounded listing")
        entries.sort(key=str.casefold)
        return "\n".join(entries) or "(empty directory)"

    def _search(self, query: str, raw_scope: str) -> str:
        scope, identity = self._secure_resolve(raw_scope, must_exist=True)
        assert identity is not None
        files = [scope] if stat.S_ISREG(identity.mode) else self._walk_files(scope)
        matches: list[str] = []
        visited = 0
        needle = query.casefold()
        for path in files:
            if visited >= MAX_SEARCH_FILES or len(matches) >= MAX_SEARCH_MATCHES:
                break
            relative = path.relative_to(self.project_root).as_posix()
            try:
                data, _digest = self._secure_read(relative, MAX_READ_BYTES)
                text = data.decode("utf-8", errors="strict")
            except (TaskExecutionError, UnicodeError):
                continue
            visited += 1
            for line_number, line in enumerate(text.splitlines(), 1):
                if needle in line.casefold():
                    matches.append(f"{relative}:{line_number}: {line[:320]}")
                    if len(matches) >= MAX_SEARCH_MATCHES:
                        break
        if visited >= MAX_SEARCH_FILES or len(matches) >= MAX_SEARCH_MATCHES:
            matches.append("[search bound reached]")
        return "\n".join(matches) or "no matches"

    def _walk_files(self, root: Path) -> list[Path]:
        pending = [root]
        files: list[Path] = []
        while pending and len(files) < MAX_SEARCH_FILES:
            directory = pending.pop()
            relative = directory.relative_to(self.project_root).as_posix() or "."
            checked, before = self._secure_resolve(
                relative, must_exist=True, require_directory=True
            )
            assert before is not None
            with os.scandir(checked) as iterator:
                entries = sorted(iterator, key=lambda item: item.name.casefold())
            for entry in entries:
                if entry.name in IGNORED_DIRECTORIES:
                    continue
                child = checked / entry.name
                child_identity = self._identity(child)
                if self._is_linklike(child, child_identity):
                    continue
                if stat.S_ISDIR(child_identity.mode):
                    pending.append(child)
                elif stat.S_ISREG(child_identity.mode):
                    files.append(child)
                    if len(files) >= MAX_SEARCH_FILES:
                        break
            if not self._same_identity(before, self._identity(checked)):
                raise TaskExecutionError("search directory changed during traversal")
        files.sort(key=lambda item: item.as_posix().casefold())
        return files

    def _write_artifact(self, raw: str, content: str) -> VerifiedArtifact:
        relative = _normalize_relative(raw)
        self.policy._validate_artifact_path(relative)
        encoded = content.encode("utf-8")
        expected = sha256(encoded).hexdigest()
        output_root = self._ensure_output_root()
        parent_before = self._identity(output_root)
        target = output_root / Path(relative).name
        target_before: _FileIdentity | None
        try:
            target_before = self._identity(target)
        except FileNotFoundError:
            target_before = None
        if target_before is not None and (
            self._is_linklike(target, target_before)
            or not stat.S_ISREG(target_before.mode)
        ):
            raise TaskPolicyError("artifact target is not a trusted regular file")
        temporary = output_root / f".{target.name}.{secrets.token_hex(12)}.tmp"
        descriptor = os.open(
            temporary,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0),
            0o600,
        )
        try:
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(encoded)
                stream.flush()
                os.fsync(stream.fileno())
            if not self._same_object(parent_before, self._identity(output_root)):
                raise TaskExecutionError("artifact directory changed before commit")
            if target_before is not None and not self._same_identity(
                target_before, self._identity(target)
            ):
                raise TaskExecutionError("artifact target changed before commit")
            os.replace(temporary, target)
        finally:
            temporary.unlink(missing_ok=True)
        if not self._same_object(parent_before, self._identity(output_root)):
            raise TaskExecutionError("artifact directory changed during commit")
        data, readback = self._secure_read(relative, MAX_SAVE_BYTES)
        if data != encoded or readback != expected:
            raise TaskExecutionError("artifact read-back SHA-256 verification failed")
        return VerifiedArtifact(relative, readback, len(data))

    def _ensure_output_root(self) -> Path:
        current = self.project_root
        for name in ("outputs", "agent-workspace"):
            candidate = current / name
            try:
                os.mkdir(candidate, 0o700)
            except FileExistsError:
                pass
            identity = self._identity(candidate)
            if self._is_linklike(candidate, identity) or not stat.S_ISDIR(
                identity.mode
            ):
                raise TaskPolicyError("artifact output path is a link or non-directory")
            resolved = candidate.resolve(strict=True)
            if not _same_path(resolved, candidate.absolute()) or not _lexically_within(
                resolved, self.project_root
            ):
                raise TaskPolicyError("artifact output directory escaped the project")
            current = resolved
        return current

    def _run_bounded_process(
        self,
        argv: tuple[str, ...],
        timeout_seconds: float,
        max_output_bytes: int,
        cpu_seconds: float,
        ram_bytes: int,
        *,
        status_environment: bool,
    ) -> tuple[str, int]:
        if not isinstance(argv, tuple) or not argv:
            raise TaskPolicyError("subprocess argv must be a non-empty tuple")
        if status_environment:
            if argv != self._git_status_argv():
                raise TaskPolicyError("subprocess argv is not fixed git status")
        else:
            if len(argv) != 5:
                raise TaskPolicyError("subprocess argv is not fixed pytest")
            target = _normalize_relative(argv[3])
            expected = (sys.executable, "-m", "pytest", target, "-q")
            if argv != expected or (
                target != "tests"
                and not (target.startswith("tests/") and target.endswith(".py"))
            ):
                raise TaskPolicyError("subprocess argv is not fixed pytest")
            if Path(argv[0]).resolve(
                strict=True
            ) != self._python_executable or not self._same_identity(
                self._python_identity, self._identity(self._python_executable)
            ):
                raise TaskExecutionError("trusted Python executable changed identity")
        if timeout_seconds <= 0:
            raise TaskExecutionError("subprocess deadline already expired")
        env = self._bounded_environment(status_environment=status_environment)
        creationflags = (
            getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0
        )
        process = subprocess.Popen(
            list(argv),
            cwd=self.project_root,
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            shell=False,
            close_fds=True,
            creationflags=creationflags,
        )
        assert process.stdout is not None and process.stderr is not None
        chunks: list[bytes] = []
        output_lock = Lock()
        overflow = Event()
        total = 0

        def drain(stream: object) -> None:
            nonlocal total
            read = getattr(stream, "read")
            while not overflow.is_set():
                block = read(4096)
                if not block:
                    return
                with output_lock:
                    remaining = max_output_bytes - total
                    if remaining > 0:
                        chunks.append(block[:remaining])
                        total += min(len(block), remaining)
                    if len(block) > remaining:
                        overflow.set()
                        return

        threads = [
            Thread(target=drain, args=(process.stdout,), daemon=True),
            Thread(target=drain, args=(process.stderr,), daemon=True),
        ]
        for thread in threads:
            thread.start()
        deadline = monotonic() + timeout_seconds
        timed_out = False
        resource_violation: str | None = None
        while process.poll() is None:
            if overflow.is_set() or monotonic() >= deadline:
                timed_out = monotonic() >= deadline
                process.kill()
                break
            used_cpu, used_ram = self._process_usage(process)
            if used_cpu > cpu_seconds:
                resource_violation = "subprocess exceeded its CPU budget"
                process.kill()
                break
            if used_ram > ram_bytes:
                resource_violation = "subprocess exceeded its RAM budget"
                process.kill()
                break
            Event().wait(0.01)
        try:
            returncode = process.wait(timeout=2.0)
        except subprocess.TimeoutExpired:
            process.kill()
            returncode = process.wait(timeout=2.0)
        for thread in threads:
            thread.join(timeout=1.0)
        process.stdout.close()
        process.stderr.close()
        output = b"".join(chunks).decode("utf-8", errors="replace")
        if overflow.is_set():
            raise TaskExecutionError("subprocess output exceeded the fixed byte bound")
        if timed_out:
            raise TaskExecutionError("subprocess exceeded its wall-clock timeout")
        if resource_violation is not None:
            raise TaskExecutionError(resource_violation)
        return output, returncode

    @staticmethod
    def _process_usage(process: subprocess.Popen[bytes]) -> tuple[float, int]:
        """Return child CPU seconds and resident bytes or fail closed.

        Phase 9 targets Windows, while the Linux branch keeps CI deterministic.
        Fixed T1 tests are trusted project tests; no general process-tree or
        arbitrary command authority is exposed.
        """

        if os.name == "nt":
            import ctypes
            from ctypes import wintypes

            class FileTime(ctypes.Structure):
                _fields_ = (("low", wintypes.DWORD), ("high", wintypes.DWORD))

            class ProcessMemoryCounters(ctypes.Structure):
                _fields_ = (
                    ("cb", wintypes.DWORD),
                    ("PageFaultCount", wintypes.DWORD),
                    ("PeakWorkingSetSize", ctypes.c_size_t),
                    ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t),
                    ("PeakPagefileUsage", ctypes.c_size_t),
                )

            creation = FileTime()
            exit_time = FileTime()
            kernel = FileTime()
            user = FileTime()
            handle = wintypes.HANDLE(int(process._handle))  # type: ignore[attr-defined]
            if not ctypes.windll.kernel32.GetProcessTimes(  # type: ignore[attr-defined]
                handle,
                ctypes.byref(creation),
                ctypes.byref(exit_time),
                ctypes.byref(kernel),
                ctypes.byref(user),
            ):
                raise TaskExecutionError("cannot enforce child CPU budget")
            counters = ProcessMemoryCounters()
            counters.cb = ctypes.sizeof(counters)
            if not ctypes.windll.psapi.GetProcessMemoryInfo(  # type: ignore[attr-defined]
                handle,
                ctypes.byref(counters),
                counters.cb,
            ):
                raise TaskExecutionError("cannot enforce child RAM budget")

            def ticks(value: FileTime) -> int:
                return (int(value.high) << 32) | int(value.low)

            cpu = (ticks(kernel) + ticks(user)) / 10_000_000.0
            return cpu, int(counters.WorkingSetSize)

        proc = Path("/proc") / str(process.pid)
        try:
            fields = (proc / "stat").read_text(encoding="ascii").split()
            ticks_per_second = os.sysconf("SC_CLK_TCK")
            cpu = (int(fields[13]) + int(fields[14])) / float(ticks_per_second)
            status = (proc / "status").read_text(encoding="ascii")
            match = re.search(r"^VmRSS:\s+(\d+)\s+kB$", status, re.MULTILINE)
            if match is None:
                raise ValueError("VmRSS missing")
            return cpu, int(match.group(1)) * 1024
        except (OSError, ValueError, IndexError) as exc:
            raise TaskExecutionError("cannot enforce child CPU/RAM budget") from exc

    def _bounded_environment(self, *, status_environment: bool) -> dict[str, str]:
        env = {
            "PATH": os.environ.get("PATH", ""),
            "PYTHONPATH": str(self.project_root),
            "PYTHONNOUSERSITE": "1",
            "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "HF_DATASETS_OFFLINE": "1",
            "WANDB_MODE": "offline",
            "CUDA_VISIBLE_DEVICES": "",
            "NO_PROXY": "",
            "no_proxy": "",
            "HTTP_PROXY": "http://127.0.0.1:9",
            "HTTPS_PROXY": "http://127.0.0.1:9",
        }
        if status_environment:
            env.update(
                {
                    "GIT_OPTIONAL_LOCKS": "0",
                    "GIT_CONFIG_NOSYSTEM": "1",
                    "GIT_TERMINAL_PROMPT": "0",
                }
            )
        return env

    def _git_status_argv(self) -> tuple[str, ...]:
        if self._git_executable is None or self._git_identity is None:
            raise TaskExecutionError("git executable is unavailable")
        if not self._same_identity(
            self._git_identity, self._identity(self._git_executable)
        ):
            raise TaskExecutionError("trusted git executable changed identity")
        null_device = "NUL" if os.name == "nt" else "/dev/null"
        return (
            str(self._git_executable),
            "-c",
            "core.fsmonitor=false",
            "-c",
            f"core.hooksPath={null_device}",
            "-c",
            "diff.external=",
            "status",
            "--short",
            "--branch",
        )

    def _verify(
        self,
        plan: TypedTaskPlan,
        results: list[TaskActionResult],
        artifacts: list[VerifiedArtifact],
        proposal_ids: list[str],
    ) -> None:
        if len(results) < plan.verifier.minimum_passed_actions:
            raise TaskExecutionError("task verifier action threshold was not met")
        expected = {item.path: item.sha256 for item in plan.expected_artifacts}
        actual = {item.path: item.sha256 for item in artifacts}
        if expected != actual:
            raise TaskExecutionError(
                "artifact verifier did not match exact SHA-256 set"
            )
        if plan.verifier.kind is VerifierKind.PYTEST_PASS and not any(
            item.kind is TaskActionKind.RUN_TEST for item in results
        ):
            raise TaskExecutionError("pytest verifier received no successful test")
        if plan.verifier.kind is VerifierKind.PROPOSAL_STAGED and not proposal_ids:
            raise TaskExecutionError("proposal verifier received no staged proposal")

    @staticmethod
    def _identity(path: Path) -> _FileIdentity:
        return TaskPlanExecutor._identity_from_stat(os.lstat(path))

    @staticmethod
    def _identity_from_stat(value: os.stat_result) -> _FileIdentity:
        return _FileIdentity(
            int(value.st_dev),
            int(value.st_ino),
            int(value.st_mode),
            int(value.st_size),
            int(value.st_mtime_ns),
            int(getattr(value, "st_file_attributes", 0)),
        )

    @staticmethod
    def _same_identity(left: _FileIdentity, right: _FileIdentity) -> bool:
        return left == right

    @staticmethod
    def _same_object(left: _FileIdentity, right: _FileIdentity) -> bool:
        return (
            left.device,
            left.inode,
            left.mode,
            left.attributes,
        ) == (
            right.device,
            right.inode,
            right.mode,
            right.attributes,
        )

    @staticmethod
    def _is_linklike(path: Path, identity: _FileIdentity) -> bool:
        if stat.S_ISLNK(identity.mode) or identity.attributes & REPARSE_POINT_ATTRIBUTE:
            return True
        isjunction = getattr(os.path, "isjunction", None)
        return bool(isjunction is not None and isjunction(path))


def _normalize_relative(raw: str, *, allow_root: bool = False) -> str:
    if not isinstance(raw, str):
        raise TaskPolicyError("path must be text")
    if not raw or raw != raw.strip() or len(raw) > 512 or "\x00" in raw:
        raise TaskPolicyError("path is empty, padded, binary, or too long")
    value = unicodedata.normalize("NFC", raw).replace("\\", "/")
    if value == ".":
        if allow_root:
            return value
        raise TaskPolicyError("project root is not valid for this action")
    if (
        value.startswith("/")
        or value.startswith("//")
        or re.match(r"^[A-Za-z]:", value)
    ):
        raise TaskPolicyError("absolute, UNC, and drive paths are forbidden")
    if _PERCENT_ESCAPE.search(value):
        raise TaskPolicyError("URL-escaped path syntax is forbidden")
    raw_parts = value.split("/")
    if any(part in {"", ".", ".."} for part in raw_parts):
        raise TaskPolicyError("empty, current, or parent path segments are forbidden")
    for part in raw_parts:
        if any(ord(char) < 32 for char in part):
            raise TaskPolicyError("control characters are forbidden in paths")
        if ":" in part or part.rstrip(" .") != part or _WINDOWS_DEVICE.fullmatch(part):
            raise TaskPolicyError("device, ADS, or ambiguous path segment rejected")
    normalized = PurePosixPath(*raw_parts).as_posix()
    if normalized != value:
        raise TaskPolicyError("path is not in canonical relative form")
    return normalized


def _within_allowed(path: str, allowed_paths: tuple[str, ...]) -> bool:
    return any(
        allowed == "." or path == allowed or path.startswith(allowed + "/")
        for allowed in allowed_paths
    )


def _same_path(left: Path, right: Path) -> bool:
    return os.path.normcase(os.path.abspath(left)) == os.path.normcase(
        os.path.abspath(right)
    )


def _lexically_within(candidate: Path, root: Path) -> bool:
    try:
        return os.path.commonpath(
            (os.path.abspath(candidate), os.path.abspath(root))
        ) == os.path.abspath(root)
    except ValueError:
        return False


def _is_sha256(value: str) -> bool:
    return bool(re.fullmatch(r"[0-9a-f]{64}", value))


__all__ = [
    "CapabilityError",
    "CapabilityToken",
    "ExpectedArtifact",
    "InMemoryProposalStager",
    "RequiredInput",
    "RiskTier",
    "StagedPatchProposal",
    "TaskAction",
    "TaskActionKind",
    "TaskActionResult",
    "TaskBudget",
    "TaskExecutionError",
    "TaskPlanError",
    "TaskPlanExecutor",
    "TaskPlanPolicy",
    "TaskPlanResult",
    "TaskPolicyError",
    "TaskVerifier",
    "TypedTaskPlan",
    "UnverifiedPlannerError",
    "UnverifiedPlannerGate",
    "VerifiedArtifact",
    "VerifierKind",
]
