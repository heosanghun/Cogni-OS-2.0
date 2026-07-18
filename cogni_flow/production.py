from __future__ import annotations

from collections import deque
from contextlib import contextmanager
from dataclasses import asdict, dataclass, replace
from enum import Enum
from hashlib import sha256
import json
import os
from pathlib import Path
from secrets import token_hex
import sys
import tempfile
from threading import RLock
from time import monotonic, time, time_ns
from typing import Any, Callable, Iterator, Mapping, Protocol

from .approval import (
    ApprovalError,
    CandidateEvaluationLedger,
    CandidateEvaluationV1,
    ConsumedApprovalLedger,
    Ed25519ApprovalVerifier,
    HumanApprovalV1,
    RollbackAuthorizationV1,
    canonical_json_bytes,
)
from .cycle import EvolutionReport, SelfHarness
from .daemon import FailureCaptureDaemon
from .harness import (
    FailureTrace,
    PatchPolicy,
    PatchProposal,
    PromotionResult,
    SafeHarnessPatcher,
    SandboxResult,
    SandboxRunner,
    WeaknessCluster,
)
from .local_proposer import LocalGemmaPatchProposer, ResolvedPatchTarget
from .logdb import LogDB
from .rhythm import RhythmController, SystemMode
from .scheduler import IdleNightScheduler, ScheduleTick
from .snapshot import SafeProjectSnapshotBuilder, SnapshotEvidence
from .proposals import (
    CandidateDraft,
    NegativeProposalV1,
    PatchProposalV1,
    ProposalOnlyError,
    ProposalOnlySelfHarness,
)


_DIGEST_LENGTH = 64
_MAX_RUNNER_OUTPUT_CHARS = 40_000
_MAX_JOURNAL_METADATA_BYTES = 16_384
_TERMINAL_JOURNAL_STATES = {
    "committed",
    "rolled_back",
    "recovered_rollback",
    "aborted",
    "operator_rolled_back",
}
_DEFAULT_TEST_COMMAND = (
    sys.executable,
    "-m",
    "unittest",
    "discover",
    "-s",
    "tests",
    "-v",
)


class IsolationAttestationError(RuntimeError):
    """Raised when a runner cannot prove an explicitly trusted boundary."""


class JournalIntegrityError(RuntimeError):
    """Raised when a backup or live file no longer matches its journal digest."""


class PromotionMode(str, Enum):
    PROPOSAL_ONLY = "proposal_only"
    ATTESTED = "attested"


def command_sha256(command: tuple[str, ...]) -> str:
    """Return an unambiguous digest for one shell-free argv sequence."""

    _validate_command(command, "command")
    payload = b"".join(
        len(item.encode("utf-8")).to_bytes(4, "big") + item.encode("utf-8")
        for item in command
    )
    return sha256(payload).hexdigest()


@dataclass(frozen=True)
class RunnerAttestation:
    """Out-of-band evidence describing one candidate-execution boundary.

    The evidence digest must be placed in ``ProductionHarnessConfig`` by a
    trusted operator after independent review. Merely naming WSL, Windows
    Sandbox, Docker, or an external command never creates this attestation.
    """

    version: int
    runner_id: str
    evidence_sha256: str
    kernel_boundary: bool
    network_isolated: bool
    host_filesystem_isolated: bool
    ephemeral_workspace: bool
    allowed_command_sha256: tuple[str, ...]


class AttestedSandboxRunner(SandboxRunner, Protocol):
    def isolation_attestation(self) -> RunnerAttestation: ...


@dataclass(frozen=True)
class ProductionHarnessConfig:
    state_directory: str = ".cogni_state/self_harness"
    allowed_roots: tuple[str, ...] = ("cogni_core", "cogni_flow")
    max_patch_bytes: int = 256_000
    queue_capacity: int = 256
    excerpt_limit: int = 4_000
    idle_seconds: float = 900.0
    promotion_mode: PromotionMode = PromotionMode.PROPOSAL_ONLY
    regression_command: tuple[str, ...] = _DEFAULT_TEST_COMMAND
    health_check_command: tuple[str, ...] = _DEFAULT_TEST_COMMAND
    regression_timeout_seconds: int = 180
    health_timeout_seconds: int = 180
    trusted_runner_evidence_sha256: tuple[str, ...] = ()
    trusted_runner_ids: tuple[str, ...] = ()
    max_failure_records: int = 2_048
    max_audit_records: int = 4_096
    max_journal_records: int = 32
    max_pending_proposals: int = 8
    max_target_mappings: int = 128
    evaluation_ttl_seconds: int = 86_400
    max_evaluation_records: int = 64
    max_consumed_approvals: int = 256

    def __post_init__(self) -> None:
        state = Path(self.state_directory)
        if state.is_absolute() or not state.parts or ".." in state.parts:
            raise ValueError("state_directory must be a safe relative path")
        if state.parts[0] in self.allowed_roots:
            raise ValueError("state_directory must be outside mutable roots")
        if not self.allowed_roots or len(set(self.allowed_roots)) != len(
            self.allowed_roots
        ):
            raise ValueError("allowed_roots must be a non-empty unique tuple")
        for root in self.allowed_roots:
            relative = Path(root)
            if (
                relative.is_absolute()
                or len(relative.parts) != 1
                or root
                in {
                    "",
                    ".",
                    "..",
                }
            ):
                raise ValueError("each mutable root must be one safe path component")
        if not 1 <= self.max_patch_bytes <= 1_000_000:
            raise ValueError("max_patch_bytes is outside the bounded range")
        if not 1 <= self.queue_capacity <= 4_096:
            raise ValueError("queue_capacity is outside the bounded range")
        if not 1 <= self.excerpt_limit <= 16_000:
            raise ValueError("excerpt_limit is outside the bounded range")
        if not 0 <= self.idle_seconds <= 604_800:
            raise ValueError("idle_seconds is outside the bounded range")
        if not isinstance(self.promotion_mode, PromotionMode):
            raise TypeError("promotion_mode must be a PromotionMode")
        _validate_command(self.regression_command, "regression_command")
        _validate_command(self.health_check_command, "health_check_command")
        if not 1 <= self.regression_timeout_seconds <= 3_600:
            raise ValueError("regression timeout is outside the bounded range")
        if not 1 <= self.health_timeout_seconds <= 3_600:
            raise ValueError("health timeout is outside the bounded range")
        for digest in self.trusted_runner_evidence_sha256:
            _require_digest(digest, "trusted runner evidence")
        if any(not item or len(item) > 128 for item in self.trusted_runner_ids):
            raise ValueError("trusted runner ids must be bounded non-empty strings")
        for name, value, maximum in (
            ("max_failure_records", self.max_failure_records, 100_000),
            ("max_audit_records", self.max_audit_records, 100_000),
            ("max_journal_records", self.max_journal_records, 256),
            ("max_pending_proposals", self.max_pending_proposals, 64),
            ("max_target_mappings", self.max_target_mappings, 256),
            ("evaluation_ttl_seconds", self.evaluation_ttl_seconds, 604_800),
            ("max_evaluation_records", self.max_evaluation_records, 1_024),
            ("max_consumed_approvals", self.max_consumed_approvals, 4_096),
        ):
            if not 1 <= value <= maximum:
                raise ValueError(f"{name} is outside the bounded range")


@dataclass(frozen=True)
class ProductionHarnessStatus:
    running: bool
    promotion_mode: PromotionMode
    promotion_enabled: bool
    blocked_reason: str | None
    pending_proposals: int
    failure_cursor: float
    evidence_failures: int
    evidence_successes: int
    evidence_capture_ratio: float
    rich_pending_proposals: int
    negative_proposals: int
    unreviewable_proposals: int
    proposal_integrity_errors: tuple[tuple[str, str], ...]
    awaiting_approval_evaluations: int
    latest_evaluation_id: str | None


class _VerifiedRunner:
    """Private adapter created only after explicit attestation verification."""

    kernel_isolated = True

    def __init__(
        self,
        delegate: AttestedSandboxRunner,
        allowed_commands: tuple[str, ...],
        attestation: RunnerAttestation,
    ) -> None:
        self._delegate = delegate
        self._allowed_commands = frozenset(allowed_commands)
        self.runner_id = attestation.runner_id
        self.evidence_sha256 = attestation.evidence_sha256.lower()

    def run(
        self, project: Path, command: tuple[str, ...], timeout_seconds: int
    ) -> SandboxResult:
        digest = command_sha256(command)
        if digest not in self._allowed_commands:
            raise IsolationAttestationError(
                "runner attestation does not cover the requested command"
            )
        result = self._delegate.run(project, command, timeout_seconds)
        if not isinstance(result, SandboxResult):
            raise TypeError("attested runner returned an invalid SandboxResult")
        output = result.output
        if not isinstance(output, str):
            raise TypeError("attested runner output must be text")
        if not isinstance(result.returncode, int) or not isinstance(
            result.passed, bool
        ):
            raise TypeError("attested runner returned invalid result fields")
        return SandboxResult(
            result.passed,
            result.returncode,
            output[-_MAX_RUNNER_OUTPUT_CHARS:],
        )


def verify_runner_attestation(
    runner: AttestedSandboxRunner,
    config: ProductionHarnessConfig,
) -> SandboxRunner:
    """Verify an operator-allowlisted attestation and return a sealed adapter."""

    if getattr(runner, "kernel_isolated", None) is not True:
        raise IsolationAttestationError("runner does not claim kernel isolation")
    getter = getattr(runner, "isolation_attestation", None)
    if not callable(getter):
        raise IsolationAttestationError("runner exposes no isolation attestation")
    attestation = getter()
    if not isinstance(attestation, RunnerAttestation):
        raise IsolationAttestationError("runner returned an invalid attestation")
    if attestation.version != 1:
        raise IsolationAttestationError("unsupported runner attestation version")
    if not attestation.runner_id or len(attestation.runner_id) > 128:
        raise IsolationAttestationError("runner id is invalid")
    try:
        evidence = _require_digest(attestation.evidence_sha256, "runner evidence")
    except ValueError as exc:
        raise IsolationAttestationError("runner evidence digest is invalid") from exc
    trusted_evidence = {item.lower() for item in config.trusted_runner_evidence_sha256}
    if not trusted_evidence or evidence not in trusted_evidence:
        raise IsolationAttestationError(
            "runner evidence digest is not explicitly trusted"
        )
    if config.trusted_runner_ids and attestation.runner_id not in set(
        config.trusted_runner_ids
    ):
        raise IsolationAttestationError("runner id is not explicitly trusted")
    boundaries = (
        attestation.kernel_boundary,
        attestation.network_isolated,
        attestation.host_filesystem_isolated,
        attestation.ephemeral_workspace,
    )
    if not all(item is True for item in boundaries):
        raise IsolationAttestationError(
            "runner attestation does not prove every required isolation boundary"
        )
    try:
        attested_commands = {
            _require_digest(item, "attested command")
            for item in attestation.allowed_command_sha256
        }
    except (TypeError, ValueError) as exc:
        raise IsolationAttestationError(
            "runner command attestation is invalid"
        ) from exc
    required_commands = {
        command_sha256(config.regression_command),
        command_sha256(config.health_check_command),
    }
    if not required_commands <= attested_commands:
        raise IsolationAttestationError(
            "runner attestation does not cover regression and health commands"
        )
    return _VerifiedRunner(runner, tuple(required_commands), attestation)


class BoundedLogDB(LogDB):
    """SQLite log store with hard row-count retention on both local ledgers."""

    def __init__(
        self,
        path: str | Path,
        *,
        max_failure_records: int,
        max_audit_records: int,
    ) -> None:
        self.max_failure_records = max_failure_records
        self.max_audit_records = max_audit_records
        super().__init__(path)

    def record_failure(
        self, trace: FailureTrace, timestamp: float | None = None
    ) -> int:
        with self._connect() as db:
            cursor = db.execute(
                "INSERT INTO failures(timestamp,test_id,exception_type,verifier_code,mechanism,excerpt) "
                "VALUES(?,?,?,?,?,?)",
                (
                    time() if timestamp is None else timestamp,
                    trace.test_id[-512:],
                    trace.exception_type[-128:],
                    trace.verifier_code[-128:],
                    trace.mechanism[-128:],
                    trace.excerpt[-16_000:],
                ),
            )
            db.execute(
                "DELETE FROM failures WHERE id NOT IN "
                "(SELECT id FROM failures ORDER BY id DESC LIMIT ?)",
                (self.max_failure_records,),
            )
            return int(cursor.lastrowid)

    def audit(self, kind: str, subject: str, detail: str) -> int:
        with self._connect() as db:
            cursor = db.execute(
                "INSERT INTO audit(timestamp,kind,subject,detail) VALUES(?,?,?,?)",
                (time(), kind[-128:], subject[-512:], detail[-2_048:]),
            )
            db.execute(
                "DELETE FROM audit WHERE sequence NOT IN "
                "(SELECT sequence FROM audit ORDER BY sequence DESC LIMIT ?)",
                (self.max_audit_records,),
            )
            return int(cursor.lastrowid)


class AllowlistedTargetResolver:
    """Resolve a failure signature to an operator-declared existing source file."""

    def __init__(
        self,
        project_root: Path,
        targets: Mapping[tuple[str, str, str], str],
        policy: PatchPolicy,
        *,
        max_targets: int,
    ) -> None:
        self.project_root = project_root.resolve(strict=True)
        self.policy = policy
        if len(targets) > max_targets:
            raise ValueError("target allowlist exceeds its configured bound")
        normalized: dict[tuple[str, str, str], Path] = {}
        for signature, value in targets.items():
            if (
                not isinstance(signature, tuple)
                or len(signature) != 3
                or any(
                    not isinstance(item, str) or len(item) > 256 for item in signature
                )
            ):
                raise ValueError("target signature must contain three bounded strings")
            normalized[signature] = self._validate_relative(value)
        self._targets = normalized

    def __call__(self, cluster) -> ResolvedPatchTarget | None:
        relative = self._targets.get(cluster.signature)
        if relative is None:
            return None
        unresolved = self.project_root / relative
        if unresolved.is_symlink() or not unresolved.is_file():
            raise RuntimeError("allowlisted patch target is no longer a regular file")
        resolved = unresolved.resolve(strict=True)
        if resolved != unresolved.absolute():
            raise RuntimeError("allowlisted patch target traverses a symbolic link")
        payload = resolved.read_bytes()
        if len(payload) > self.policy.max_bytes:
            raise ValueError("allowlisted source exceeds the patch size limit")
        source = payload.decode("utf-8")
        return ResolvedPatchTarget(
            relative.as_posix(), sha256(payload).hexdigest(), source
        )

    def _validate_relative(self, value: str) -> Path:
        if not isinstance(value, str) or not value or len(value) > 512:
            raise ValueError("target path must be a bounded string")
        relative = Path(value)
        if (
            relative.is_absolute()
            or ".." in relative.parts
            or not relative.parts
            or relative.parts[0] not in self.policy.allowed_roots
            or relative.suffix != ".py"
        ):
            raise ValueError("target path is outside the Python mutable allowlist")
        unresolved = self.project_root / relative
        resolved = unresolved.resolve(strict=True)
        if self.project_root not in resolved.parents:
            raise ValueError("target path escaped the project root")
        if unresolved.is_symlink() or resolved != unresolved.absolute():
            raise ValueError("symbolic-link patch targets are not allowed")
        if not resolved.is_file():
            raise ValueError("patch target must be an existing regular file")
        return relative


class RecordingProposer:
    """Keep a bounded review queue backed by inert content-addressed blobs."""

    def __init__(
        self,
        proposer: LocalGemmaPatchProposer,
        capacity: int,
        *,
        evidence_ledger: ProposalOnlySelfHarness | None = None,
    ) -> None:
        self.proposer = proposer
        self.capacity = capacity
        self.evidence_ledger = evidence_ledger
        self._pending: deque[PatchProposal] = deque(maxlen=capacity)
        self._rich_pending: deque[PatchProposalV1] = deque(
            maxlen=0 if evidence_ledger is None else capacity
        )
        self._candidate_bank: dict[tuple[str, str, str], dict[str, PatchProposal]] = {}
        self._negative_replacements: set[tuple[tuple[str, str, str], str]] = set()
        self._persisted_replacements: set[tuple[tuple[str, str, str], str]] = set()
        self._lock = RLock()
        if evidence_ledger is not None:
            proposal_by_id = {
                item.proposal_id: item for item in evidence_ledger.proposals
            }
            if len(proposal_by_id) != len(evidence_ledger.proposals):
                raise ProposalOnlyError(
                    "hydrated proposal queue contains duplicate IDs"
                )
            self._persisted_replacements.update(
                (item.signature, item.replacement_sha256)
                for item in evidence_ledger.proposals
            )
            rejected_ids: set[str] = set()
            for negative in evidence_ledger.negative_archive:
                proposal = proposal_by_id.get(negative.proposal_id)
                if proposal is None:
                    raise ProposalOnlyError(
                        "hydrated suppression references an unknown proposal"
                    )
                rejected_ids.add(proposal.proposal_id)
                self._negative_replacements.add(
                    (proposal.signature, proposal.replacement_sha256)
                )
            reviewable_patches = dict(evidence_ledger.reviewable_patches)
            if not set(reviewable_patches).issubset(proposal_by_id):
                raise ProposalOnlyError(
                    "hydrated review queue references an unknown proposal"
                )
            active = tuple(
                proposal
                for proposal in evidence_ledger.proposals
                if proposal.proposal_id not in rejected_ids
                and proposal.proposal_id in reviewable_patches
            )
            if len(active) > capacity:
                raise ProposalOnlyError(
                    "hydrated pending proposal queue crossed its configured capacity"
                )
            active_replacements: set[tuple[tuple[str, str, str], str]] = set()
            for proposal in active:
                key = (proposal.signature, proposal.replacement_sha256)
                if key in self._negative_replacements:
                    raise ProposalOnlyError(
                        "hydrated pending proposal conflicts with negative suppression"
                    )
                if key in active_replacements:
                    raise ProposalOnlyError(
                        "hydrated pending queue repeats a replacement"
                    )
                active_replacements.add(key)
                self._pending.append(reviewable_patches[proposal.proposal_id])
                self._rich_pending.append(proposal)

    def __call__(self, cluster) -> tuple[PatchProposal, ...]:
        proposals = tuple(self.proposer(cluster))
        with self._lock:
            ledger = self.evidence_ledger
            if ledger is None:
                self._pending.extend(proposals)
            else:
                signature = cluster.signature
                if (
                    signature not in self._candidate_bank
                    and len(self._candidate_bank) >= self.capacity
                ):
                    self._candidate_bank.pop(next(iter(self._candidate_bank)))
                bank = self._candidate_bank.setdefault(signature, {})
                for proposal in proposals:
                    replacement_digest = sha256(
                        proposal.replacement.encode("utf-8")
                    ).hexdigest()
                    if (
                        len(bank) < 8
                        and (signature, replacement_digest)
                        not in self._negative_replacements
                        and (signature, replacement_digest)
                        not in self._persisted_replacements
                    ):
                        bank.setdefault(replacement_digest, proposal)
                available = self.capacity - len(self._rich_pending)
                if (
                    len(bank) >= ledger.minimum_candidates
                    and available >= ledger.minimum_candidates
                ):
                    reproduction = (
                        cluster.traces[0].excerpt
                        if cluster.traces and cluster.traces[0].excerpt
                        else f"reproduce verifier {signature[1]}"
                    )
                    drafts = tuple(
                        CandidateDraft(
                            patch,
                            expected_behavior=(
                                f"terminal verifier {signature[0]} no longer reproduces"
                            ),
                            risk="unexecuted local-model source proposal",
                            reproduction_test=reproduction[:4_096],
                            rollback_trigger=(
                                "reject on stale digest, policy failure, or held-out regression"
                            ),
                        )
                        for patch in tuple(bank.values())[: min(8, available)]
                    )
                    rich = ledger.submit_cluster_candidates(cluster, drafts)
                    reviewable = dict(ledger.reviewable_patches)
                    self._pending.extend(reviewable[item.proposal_id] for item in rich)
                    self._rich_pending.extend(rich)
                    self._persisted_replacements.update(
                        (item.signature, item.replacement_sha256) for item in rich
                    )
                    self._candidate_bank.pop(signature, None)
        return proposals

    @property
    def pending(self) -> tuple[PatchProposal, ...]:
        with self._lock:
            return tuple(self._pending)

    @property
    def rich_pending(self) -> tuple[PatchProposalV1, ...]:
        with self._lock:
            return tuple(self._rich_pending)

    def proposal_id_for_patch(self, patch: PatchProposal) -> str:
        """Resolve one plain patch back to its immutable evidence record."""

        if not isinstance(patch, PatchProposal):
            raise TypeError("patch must be a PatchProposal")
        replacement_sha256 = sha256(patch.replacement.encode("utf-8")).hexdigest()
        with self._lock:
            matches = tuple(
                item
                for item in self._rich_pending
                if item.relative_path == patch.relative_path
                and item.base_sha256 == patch.base_sha256.lower()
                and item.replacement_sha256 == replacement_sha256
            )
        if len(matches) != 1:
            raise ProposalOnlyError(
                "candidate execution requires one evidence-linked proposal"
            )
        return matches[0].proposal_id

    def reject_rich(
        self,
        proposal_id: str,
        *,
        reason_code: str,
        evidence_sha256: str,
    ) -> NegativeProposalV1:
        with self._lock:
            ledger = self.evidence_ledger
            proposal = next(
                (
                    item
                    for item in self._rich_pending
                    if item.proposal_id == proposal_id
                ),
                None,
            )
            if ledger is None or proposal is None:
                raise ValueError("review proposal is not pending")
            negative = ledger.archive_negative(
                proposal_id,
                reason_code=reason_code,
                evidence_sha256=evidence_sha256,
            )
            self._negative_replacements.add(
                (proposal.signature, proposal.replacement_sha256)
            )
            retained = tuple(
                item for item in self._rich_pending if item.proposal_id != proposal_id
            )
            self._rich_pending.clear()
            self._rich_pending.extend(retained)
            retained_patches = tuple(
                patch
                for patch in self._pending
                if not (
                    patch.relative_path == proposal.relative_path
                    and patch.base_sha256.lower() == proposal.base_sha256
                    and sha256(patch.replacement.encode("utf-8")).hexdigest()
                    == proposal.replacement_sha256
                )
            )
            self._pending.clear()
            self._pending.extend(retained_patches)
            return negative

    def clear(self) -> None:
        with self._lock:
            self._pending.clear()
            self._rich_pending.clear()
            self._candidate_bank.clear()


@dataclass(frozen=True)
class BackupRecord:
    record_id: str
    relative_path: str
    before_sha256: str
    after_sha256: str
    backup_file: str
    file_mode: int
    status: str
    created_ns: int


@dataclass(frozen=True)
class _CommittedRollbackMaterial:
    record: BackupRecord
    target: Path
    before: bytes
    after: bytes


class BackupJournal:
    """Bounded, digest-verified source backup journal for atomic promotion."""

    def __init__(
        self,
        project_root: Path,
        directory: Path,
        *,
        allowed_roots: tuple[str, ...],
        max_records: int,
        max_backup_bytes: int,
    ) -> None:
        self.project_root = project_root.resolve(strict=True)
        self.directory = directory.resolve()
        if (
            self.directory == self.project_root
            or self.project_root not in self.directory.parents
        ):
            raise ValueError("journal directory must remain inside the project root")
        relative_directory = self.directory.relative_to(self.project_root)
        if relative_directory.parts[0] in allowed_roots:
            raise ValueError("journal directory must remain outside mutable roots")
        self.allowed_roots = allowed_roots
        self.max_records = max_records
        self.max_backup_bytes = max_backup_bytes
        self._lock = RLock()
        self.directory.mkdir(parents=True, exist_ok=True)

    def prepare(
        self,
        relative: Path,
        before: bytes,
        after: bytes,
    ) -> BackupRecord:
        target = self._safe_target(relative, require_existing=True)
        if len(before) > self.max_backup_bytes or len(after) > self.max_backup_bytes:
            raise ValueError("journal payload exceeds the configured size limit")
        with self._lock:
            self._ensure_capacity()
            before_digest = sha256(before).hexdigest()
            after_digest = sha256(after).hexdigest()
            entropy = (
                f"{time_ns()}:{token_hex(8)}:{relative.as_posix()}:"
                f"{before_digest}:{after_digest}"
            ).encode("utf-8")
            record_id = sha256(entropy).hexdigest()[:32]
            backup_file = f"{record_id}.bak"
            record = BackupRecord(
                record_id,
                relative.as_posix(),
                before_digest,
                after_digest,
                backup_file,
                target.stat().st_mode & 0o777,
                "prepared",
                time_ns(),
            )
            _atomic_write_bytes(self.directory / backup_file, before, exclusive=True)
            if (
                sha256((self.directory / backup_file).read_bytes()).hexdigest()
                != before_digest
            ):
                raise JournalIntegrityError("backup digest verification failed")
            self._write_record(record)
            return record

    def mark(self, record: BackupRecord, status: str) -> BackupRecord:
        if status not in _TERMINAL_JOURNAL_STATES - {"operator_rolled_back"}:
            raise ValueError("invalid terminal journal status")
        with self._lock:
            current = self._load_record(record.record_id)
            if current.status != "prepared":
                if current.status == status:
                    return current
                raise JournalIntegrityError("journal record is already terminal")
            updated = replace(current, status=status)
            self._write_record(updated)
            return updated

    def rollback(
        self, record: BackupRecord, *, recovered: bool = False
    ) -> BackupRecord:
        with self._lock:
            current = self._load_record(record.record_id)
            if current.status != "prepared":
                raise JournalIntegrityError("only a prepared record can be rolled back")
            relative = Path(current.relative_path)
            target = self._safe_target(relative, require_existing=True)
            live = target.read_bytes()
            if sha256(live).hexdigest() != current.after_sha256:
                raise JournalIntegrityError(
                    "live file changed after promotion; refusing unsafe rollback"
                )
            backup_path = self.directory / current.backup_file
            backup = backup_path.read_bytes()
            if len(backup) > self.max_backup_bytes:
                raise JournalIntegrityError("backup exceeds the journal size limit")
            if sha256(backup).hexdigest() != current.before_sha256:
                raise JournalIntegrityError("backup digest verification failed")
            _atomic_write_bytes(target, backup, mode=current.file_mode)
            if sha256(target.read_bytes()).hexdigest() != current.before_sha256:
                raise JournalIntegrityError("restored file digest verification failed")
            return self.mark(
                current, "recovered_rollback" if recovered else "rolled_back"
            )

    def committed_rollback_material(self, record_id: str) -> _CommittedRollbackMaterial:
        """Load exact before/after bytes for one still-committed promotion."""

        with self._lock:
            current = self._load_record(record_id)
            if current.status != "committed":
                raise JournalIntegrityError(
                    "operator rollback requires a committed journal record"
                )
            target = self._safe_target(
                Path(current.relative_path), require_existing=True
            )
            after = target.read_bytes()
            if len(after) > self.max_backup_bytes:
                raise JournalIntegrityError(
                    "committed source exceeds the journal size limit"
                )
            if sha256(after).hexdigest() != current.after_sha256:
                raise JournalIntegrityError(
                    "live source differs from the committed after digest"
                )
            if target.stat().st_mode & 0o777 != current.file_mode:
                raise JournalIntegrityError(
                    "live source mode differs from the committed file mode"
                )
            backup_path = self.directory / current.backup_file
            if backup_path.is_symlink() or backup_path.parent != self.directory:
                raise JournalIntegrityError("journal backup cannot be linked")
            before = backup_path.read_bytes()
            if len(before) > self.max_backup_bytes:
                raise JournalIntegrityError("backup exceeds the journal size limit")
            if sha256(before).hexdigest() != current.before_sha256:
                raise JournalIntegrityError("backup digest verification failed")
            return _CommittedRollbackMaterial(current, target, before, after)

    def mark_operator_rolled_back(self, record: BackupRecord) -> BackupRecord:
        """Finalize one committed record after an operator rollback passes health."""

        with self._lock:
            current = self._load_record(record.record_id)
            if current.status == "operator_rolled_back":
                return current
            if current.status != "committed":
                raise JournalIntegrityError(
                    "only a committed record can be operator rolled back"
                )
            updated = replace(current, status="operator_rolled_back")
            self._write_record(updated)
            return updated

    def recover_incomplete(self) -> int:
        """Fail closed or restore any crash-interrupted exact candidate digest."""

        recovered = 0
        with self._lock:
            self._remove_orphan_temporaries()
            records = self.records()
            for record in records:
                if record.status != "prepared":
                    continue
                target = self._safe_target(
                    Path(record.relative_path), require_existing=True
                )
                digest = sha256(target.read_bytes()).hexdigest()
                if digest == record.before_sha256:
                    self.mark(record, "aborted")
                elif digest == record.after_sha256:
                    self.rollback(record, recovered=True)
                    recovered += 1
                else:
                    raise JournalIntegrityError(
                        "incomplete journal record does not match before or after digest"
                    )
            return recovered

    def records(self) -> tuple[BackupRecord, ...]:
        with self._lock:
            paths = sorted(self.directory.glob("*.json"))
            if len(paths) > self.max_records:
                raise JournalIntegrityError("journal record count exceeds its bound")
            records = tuple(self._read_record_path(path) for path in paths)
            return tuple(sorted(records, key=lambda item: item.created_ns))

    def _ensure_capacity(self) -> None:
        self._remove_orphan_temporaries()
        records = list(self.records())
        while len(records) >= self.max_records:
            terminal = next(
                (item for item in records if item.status in _TERMINAL_JOURNAL_STATES),
                None,
            )
            if terminal is None:
                raise RuntimeError("journal is full of unresolved promotion records")
            self._delete_record(terminal)
            records.remove(terminal)

    def _delete_record(self, record: BackupRecord) -> None:
        metadata = self.directory / f"{record.record_id}.json"
        backup = self.directory / record.backup_file
        if metadata.parent != self.directory or backup.parent != self.directory:
            raise JournalIntegrityError("journal deletion escaped its directory")
        metadata.unlink(missing_ok=True)
        backup.unlink(missing_ok=True)

    def _safe_target(self, relative: Path, *, require_existing: bool) -> Path:
        if (
            relative.is_absolute()
            or ".." in relative.parts
            or not relative.parts
            or relative.parts[0] not in self.allowed_roots
            or relative.suffix != ".py"
        ):
            raise JournalIntegrityError("journal target escaped the mutable allowlist")
        unresolved = self.project_root / relative
        if unresolved.is_symlink():
            raise JournalIntegrityError("journal target cannot be a symbolic link")
        resolved = unresolved.resolve(strict=require_existing)
        if (
            self.project_root not in resolved.parents
            or resolved != unresolved.absolute()
        ):
            raise JournalIntegrityError("journal target escaped the project root")
        if require_existing and not resolved.is_file():
            raise JournalIntegrityError("journal target must be an existing file")
        return resolved

    def _load_record(self, record_id: str) -> BackupRecord:
        if len(record_id) != 32 or any(
            ch not in "0123456789abcdef" for ch in record_id
        ):
            raise JournalIntegrityError("invalid journal record id")
        return self._read_record_path(self.directory / f"{record_id}.json")

    def _read_record_path(self, path: Path) -> BackupRecord:
        if path.parent != self.directory or not path.is_file():
            raise JournalIntegrityError("journal metadata is missing")
        if path.stat().st_size > _MAX_JOURNAL_METADATA_BYTES:
            raise JournalIntegrityError("journal metadata exceeds its size bound")
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if set(payload) != set(BackupRecord.__dataclass_fields__):
                raise ValueError("unexpected metadata fields")
            record = BackupRecord(**payload)
        except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise JournalIntegrityError("journal metadata is malformed") from exc
        if path.name != f"{record.record_id}.json":
            raise JournalIntegrityError(
                "journal metadata filename does not match its id"
            )
        if record.backup_file != f"{record.record_id}.bak":
            raise JournalIntegrityError("journal backup filename is invalid")
        if not isinstance(record.file_mode, int) or not 0 <= record.file_mode <= 0o777:
            raise JournalIntegrityError("journal file mode is invalid")
        _require_digest(record.before_sha256, "journal before digest")
        _require_digest(record.after_sha256, "journal after digest")
        if record.status not in {"prepared", *_TERMINAL_JOURNAL_STATES}:
            raise JournalIntegrityError("journal status is invalid")
        self._safe_target(Path(record.relative_path), require_existing=True)
        backup = self.directory / record.backup_file
        if not backup.is_file() or backup.stat().st_size > self.max_backup_bytes:
            raise JournalIntegrityError("journal backup is missing or oversized")
        return record

    def _write_record(self, record: BackupRecord) -> None:
        data = json.dumps(
            asdict(record), ensure_ascii=True, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        if len(data) > _MAX_JOURNAL_METADATA_BYTES:
            raise ValueError("journal metadata exceeds its size bound")
        _atomic_write_bytes(self.directory / f"{record.record_id}.json", data)

    def _remove_orphan_temporaries(self) -> None:
        for path in self.directory.iterdir():
            if path.is_file() and ".cogni-tmp-" in path.name:
                path.unlink(missing_ok=True)
        metadata_ids = {path.stem for path in self.directory.glob("*.json")}
        for backup in self.directory.glob("*.bak"):
            if backup.stem not in metadata_ids:
                backup.unlink(missing_ok=True)


@dataclass(frozen=True)
class CandidateEvaluationResult:
    passed: bool
    sandbox: SandboxResult
    target: Path
    evaluation: CandidateEvaluationV1 | None


@dataclass(frozen=True)
class OperatorRollbackResult:
    rolled_back: bool
    health: SandboxResult
    target: Path
    journal_record: BackupRecord


class JournaledHarnessPatcher(SafeHarnessPatcher):
    """Attested evaluation plus externally approved one-time promotion."""

    def __init__(
        self,
        project_root: Path,
        rhythm: RhythmController,
        *,
        policy: PatchPolicy,
        sandbox: SandboxRunner,
        journal: BackupJournal,
        test_command: tuple[str, ...],
        health_check_command: tuple[str, ...],
        timeout_seconds: int,
        health_timeout_seconds: int,
        ignored_stage_root: str,
        evaluation_ledger: CandidateEvaluationLedger,
        approval_verifier: Ed25519ApprovalVerifier,
        consumed_approvals: ConsumedApprovalLedger,
        source_surface_digest: Callable[[], str],
        evaluation_ttl_seconds: int,
    ) -> None:
        if not isinstance(sandbox, _VerifiedRunner):
            raise IsolationAttestationError(
                "journaled promotion requires a verified attested runner"
            )
        super().__init__(
            project_root,
            rhythm,
            policy=policy,
            sandbox=sandbox,
            test_command=test_command,
            timeout_seconds=timeout_seconds,
        )
        self.journal = journal
        self.health_check_command = health_check_command
        self.health_timeout_seconds = health_timeout_seconds
        self.ignored_stage_root = ignored_stage_root
        self.evaluation_ledger = evaluation_ledger
        self.approval_verifier = approval_verifier
        self.consumed_approvals = consumed_approvals
        self.source_surface_digest = source_surface_digest
        self.evaluation_ttl_seconds = evaluation_ttl_seconds
        self.snapshot_builder = SafeProjectSnapshotBuilder(
            project_root,
            excluded_roots=(ignored_stage_root,),
        )
        self._operator_rollback_lock = RLock()

    @property
    def requires_manual_approval(self) -> bool:
        return True

    def validate_and_promote(self, proposal: PatchProposal) -> PromotionResult:
        """Never turn a scheduled regression result into mutation authority."""

        raise ApprovalError(
            "automatic promotion is disabled; evaluate and import one signed approval"
        )

    def evaluate_candidate(
        self, proposal: PatchProposal, *, proposal_id: str
    ) -> CandidateEvaluationResult:
        """Run regression in the attested boundary and persist immutable evidence."""

        if self.rhythm.mode != SystemMode.EVOLUTION:
            raise RuntimeError("candidate evaluation requires evolution mode")
        with self.rhythm.evolution_slot():
            target = self.validate_proposal(proposal)
            if not target.is_file():
                raise ValueError("production evaluation only patches existing files")
            relative = target.relative_to(self.project_root)
            before = target.read_bytes()
            if sha256(before).hexdigest() != proposal.base_sha256:
                raise RuntimeError("base file changed during proposal validation")
            after = proposal.replacement.encode("utf-8")
            source_surface = self.source_surface_digest()
            self.rhythm.transition(
                SystemMode.VALIDATING, f"validating {relative.as_posix()}"
            )
            with tempfile.TemporaryDirectory(prefix="cogni-regression-") as tmp:
                stage = Path(tmp) / "project"
                snapshot = self._copy_project(stage)
                staged_target = stage / relative
                staged_target.parent.mkdir(parents=True, exist_ok=True)
                staged_target.write_bytes(after)
                regression = self.sandbox.run(
                    stage, self.test_command, self.timeout_seconds
                )
            if self.source_surface_digest() != source_surface:
                self.rhythm.transition(
                    SystemMode.ROLLING_BACK,
                    "candidate runner changed the active source surface",
                )
                self.rhythm.transition(
                    SystemMode.SAFE_MODE,
                    "candidate isolation integrity failure",
                )
                raise JournalIntegrityError(
                    "candidate runner changed the active source surface"
                )
            if not regression.passed:
                self.rhythm.transition(
                    SystemMode.EVOLUTION, "candidate failed regression tests"
                )
                return CandidateEvaluationResult(False, regression, target, None)

            completed = time_ns()
            result_sha256 = sha256(
                canonical_json_bytes(
                    {
                        "passed": regression.passed,
                        "returncode": regression.returncode,
                        "output_sha256": sha256(
                            regression.output.encode("utf-8")
                        ).hexdigest(),
                    }
                )
            ).hexdigest()
            evaluation = CandidateEvaluationV1.create(
                proposal_id=proposal_id,
                relative_path=relative.as_posix(),
                base_sha256=proposal.base_sha256.lower(),
                replacement_sha256=sha256(after).hexdigest(),
                source_surface_sha256=source_surface,
                snapshot_tree_sha256=snapshot.tree_sha256,
                runner_id=self.sandbox.runner_id,
                runner_evidence_sha256=self.sandbox.evidence_sha256,
                regression_command_sha256=command_sha256(self.test_command),
                result_sha256=result_sha256,
                returncode=regression.returncode,
                completed_ns=completed,
                expires_ns=completed + self.evaluation_ttl_seconds * 1_000_000_000,
            )
            self.evaluation_ledger.record(evaluation)
            self.rhythm.transition(
                SystemMode.EVOLUTION, "candidate awaits external human approval"
            )
            return CandidateEvaluationResult(True, regression, target, evaluation)

    def promote_approved_once(
        self,
        proposal: PatchProposal,
        *,
        evaluation_id: str,
        approval: HumanApprovalV1,
    ) -> PromotionResult:
        """Install one exact candidate after signed, one-time approval."""

        if self.rhythm.mode != SystemMode.EVOLUTION:
            raise RuntimeError("approved promotion requires a fresh evolution mode")
        record: BackupRecord | None = None
        target: Path | None = None
        with self.rhythm.evolution_slot():
            evaluation = self.evaluation_ledger.load(evaluation_id)
            if self.source_surface_digest() != evaluation.source_surface_sha256:
                raise ApprovalError("source surface changed after candidate evaluation")
            target = self.validate_proposal(proposal)
            if not target.is_file():
                raise ValueError("production promotion only patches existing files")
            relative = target.relative_to(self.project_root)
            before = target.read_bytes()
            after = proposal.replacement.encode("utf-8")
            self._match_evaluation(evaluation, proposal, relative)
            self.approval_verifier.verify(approval, evaluation)
            if sha256(target.read_bytes()).hexdigest() != evaluation.base_sha256:
                raise ApprovalError("candidate base changed after human approval")
            # Consumption is the last pre-mutation operation.  Once claimed,
            # an approval is never reusable even when journal preparation or
            # health checking later fails; the operator must issue a new nonce.
            self.consumed_approvals.consume_once(approval, evaluation)
            self.rhythm.transition(
                SystemMode.VALIDATING, "signed approval and candidate digests verified"
            )
            self.rhythm.transition(
                SystemMode.PROMOTING, f"promoting {relative.as_posix()}"
            )
            try:
                record = self.journal.prepare(relative, before, after)
                if sha256(target.read_bytes()).hexdigest() != record.before_sha256:
                    raise RuntimeError("base file changed before atomic promotion")
                _atomic_write_bytes(target, after, mode=record.file_mode)
                if sha256(target.read_bytes()).hexdigest() != record.after_sha256:
                    raise JournalIntegrityError(
                        "promoted file digest verification failed"
                    )
                with tempfile.TemporaryDirectory(prefix="cogni-health-") as tmp:
                    health_stage = Path(tmp) / "project"
                    self._copy_project(health_stage)
                    health = self.sandbox.run(
                        health_stage,
                        self.health_check_command,
                        self.health_timeout_seconds,
                    )
                if not health.passed:
                    self.rhythm.transition(
                        SystemMode.ROLLING_BACK,
                        "post-promotion health check failed",
                    )
                    self.journal.rollback(record)
                    self.rhythm.transition(
                        SystemMode.EVOLUTION, "digest-verified rollback complete"
                    )
                    return PromotionResult(False, health, target)
                self.journal.mark(record, "committed")
                return PromotionResult(True, health, target)
            except BaseException:
                self._rollback_after_exception(record, target)
                raise

    def rollback_committed_once(
        self,
        record_id: str,
        authorization: RollbackAuthorizationV1,
    ) -> OperatorRollbackResult:
        """Restore one committed backup only under exact signed authority."""

        if self.rhythm.mode != SystemMode.EVOLUTION:
            raise RuntimeError("operator rollback requires a fresh evolution mode")
        material: _CommittedRollbackMaterial | None = None
        terminal = False
        with self._operator_rollback_lock, self.rhythm.evolution_slot():
            try:
                material = self.journal.committed_rollback_material(record_id)
                source_surface = self.source_surface_digest()
                self.approval_verifier.verify_rollback(
                    authorization,
                    journal_record_id=material.record.record_id,
                    relative_path=material.record.relative_path,
                    before_sha256=material.record.before_sha256,
                    after_sha256=material.record.after_sha256,
                    source_surface_sha256=source_surface,
                    runner_id=self.sandbox.runner_id,
                    runner_evidence_sha256=self.sandbox.evidence_sha256,
                    health_command_sha256=command_sha256(self.health_check_command),
                    record_created_ns=material.record.created_ns,
                )
                # Close the check/use window before burning the one-time nonce.
                current = self.journal.committed_rollback_material(record_id)
                if current != material:
                    raise JournalIntegrityError(
                        "committed rollback material changed during authorization"
                    )
                if self.source_surface_digest() != source_surface:
                    raise ApprovalError(
                        "source surface changed after rollback authorization"
                    )
                self.consumed_approvals.consume_rollback_once(authorization)
                self.rhythm.transition(
                    SystemMode.VALIDATING,
                    "signed operator rollback authorization verified",
                )
                self.rhythm.transition(
                    SystemMode.ROLLING_BACK,
                    f"operator rollback {material.record.relative_path}",
                )
                _atomic_write_bytes(
                    material.target,
                    material.before,
                    mode=material.record.file_mode,
                )
                self._verify_material(
                    material.target,
                    material.record.before_sha256,
                    material.record.file_mode,
                    "operator-restored source",
                )
                with tempfile.TemporaryDirectory(
                    prefix="cogni-operator-rollback-health-"
                ) as tmp:
                    health_stage = Path(tmp) / "project"
                    self._copy_project(health_stage)
                    health = self.sandbox.run(
                        health_stage,
                        self.health_check_command,
                        self.health_timeout_seconds,
                    )
                if not health.passed:
                    self._ensure_committed_after(material)
                    self.rhythm.transition(
                        SystemMode.EVOLUTION,
                        "operator rollback health failed; committed source restored",
                    )
                    return OperatorRollbackResult(
                        False,
                        health,
                        material.target,
                        material.record,
                    )
                finalized = self.journal.mark_operator_rolled_back(material.record)
                terminal = True
                self.rhythm.transition(
                    SystemMode.EVOLUTION,
                    "operator rollback health verified",
                )
                return OperatorRollbackResult(
                    True,
                    health,
                    material.target,
                    finalized,
                )
            except BaseException:
                if material is not None and not terminal:
                    try:
                        self._ensure_committed_after(material)
                    except BaseException:
                        if self.rhythm.mode == SystemMode.EVOLUTION:
                            self.rhythm.transition(
                                SystemMode.SAFE_MODE,
                                "operator rollback could not preserve committed source",
                            )
                        elif self.rhythm.mode in {
                            SystemMode.VALIDATING,
                            SystemMode.ROLLING_BACK,
                        }:
                            if self.rhythm.mode == SystemMode.VALIDATING:
                                self.rhythm.transition(
                                    SystemMode.ROLLING_BACK,
                                    "operator rollback recovery failed",
                                )
                            self.rhythm.transition(
                                SystemMode.SAFE_MODE,
                                "operator rollback could not restore committed source",
                            )
                        raise
                if self.rhythm.mode == SystemMode.VALIDATING:
                    self.rhythm.transition(
                        SystemMode.ROLLING_BACK,
                        "operator rollback validation failed",
                    )
                if self.rhythm.mode == SystemMode.ROLLING_BACK:
                    self.rhythm.transition(
                        SystemMode.EVOLUTION,
                        "operator rollback rejection restored committed source",
                    )
                raise

    @staticmethod
    def _verify_material(
        target: Path,
        digest: str,
        file_mode: int,
        label: str,
    ) -> None:
        if sha256(target.read_bytes()).hexdigest() != digest:
            raise JournalIntegrityError(f"{label} digest verification failed")
        if target.stat().st_mode & 0o777 != file_mode:
            raise JournalIntegrityError(f"{label} mode verification failed")

    def _ensure_committed_after(self, material: _CommittedRollbackMaterial) -> None:
        digest = sha256(material.target.read_bytes()).hexdigest()
        if digest != material.record.after_sha256:
            if digest != material.record.before_sha256:
                raise JournalIntegrityError(
                    "operator rollback left an unknown live source digest"
                )
            _atomic_write_bytes(
                material.target,
                material.after,
                mode=material.record.file_mode,
            )
        self._verify_material(
            material.target,
            material.record.after_sha256,
            material.record.file_mode,
            "reapplied committed source",
        )

    @staticmethod
    def _match_evaluation(
        evaluation: CandidateEvaluationV1,
        proposal: PatchProposal,
        relative: Path,
    ) -> None:
        expected = {
            "relative_path": relative.as_posix(),
            "base_sha256": proposal.base_sha256.lower(),
            "replacement_sha256": sha256(
                proposal.replacement.encode("utf-8")
            ).hexdigest(),
        }
        for field, value in expected.items():
            if getattr(evaluation, field) != value:
                raise ApprovalError(f"evaluation does not match candidate {field}")

    def _copy_project(self, destination: Path) -> SnapshotEvidence:
        return self.snapshot_builder.copy_to(destination)

    def _rollback_after_exception(
        self, record: BackupRecord | None, target: Path
    ) -> None:
        if self.rhythm.mode in {SystemMode.PROMOTING, SystemMode.VALIDATING}:
            self.rhythm.transition(SystemMode.ROLLING_BACK, "promotion exception")
        try:
            if record is not None and target.is_file():
                digest = sha256(target.read_bytes()).hexdigest()
                if digest == record.after_sha256:
                    self.journal.rollback(record)
                elif digest == record.before_sha256:
                    self.journal.mark(record, "aborted")
                else:
                    raise JournalIntegrityError(
                        "live file has an unknown digest during rollback"
                    )
            if self.rhythm.mode == SystemMode.ROLLING_BACK:
                self.rhythm.transition(
                    SystemMode.EVOLUTION, "promotion exception rollback complete"
                )
        except BaseException:
            if self.rhythm.mode == SystemMode.ROLLING_BACK:
                self.rhythm.transition(
                    SystemMode.SAFE_MODE, "promotion rollback integrity failure"
                )
            raise


class ProductionSelfHarness:
    """Lifecycle boundary joining capture, proposal, scheduling, and promotion."""

    def __init__(
        self,
        *,
        config: ProductionHarnessConfig,
        rhythm: RhythmController,
        logdb: BoundedLogDB,
        daemon: FailureCaptureDaemon,
        proposer: RecordingProposer,
        proposal_ledger: ProposalOnlySelfHarness,
        harness: SelfHarness,
        journal: BackupJournal,
        evaluation_ledger: CandidateEvaluationLedger | None,
        source_digest_resolver: Callable[[tuple[str, str, str]], str],
        source_surface_digest: Callable[[], str],
        clock: Callable[[], float],
    ) -> None:
        self.config = config
        self.rhythm = rhythm
        self.logdb = logdb
        self.failure_daemon = daemon
        self.proposer = proposer
        self.proposal_ledger = proposal_ledger
        self.harness = harness
        self.journal = journal
        self.evaluation_ledger = evaluation_ledger
        self._source_digest_resolver = source_digest_resolver
        self._source_surface_digest = source_surface_digest
        self._failure_cursor = 0.0
        self._lock = RLock()
        self.scheduler = IdleNightScheduler(
            rhythm,
            self._run_cycle,
            idle_seconds=config.idle_seconds,
            clock=clock,
        )

    @property
    def status(self) -> ProductionHarnessStatus:
        proposal_only = self.config.promotion_mode == PromotionMode.PROPOSAL_ONLY
        evaluations = (
            () if self.evaluation_ledger is None else self.evaluation_ledger.records()
        )
        return ProductionHarnessStatus(
            self.failure_daemon.running,
            self.config.promotion_mode,
            False,
            self.harness.proposal_only_reason
            if proposal_only
            else "automatic promotion disabled: external one-time approval required",
            len(self.proposer.pending),
            self._failure_cursor,
            len(self.proposal_ledger.failures),
            len(self.proposal_ledger.successes),
            self.proposal_ledger.capture_coverage.ratio,
            len(self.proposer.rich_pending),
            len(self.proposal_ledger.negative_archive),
            len(self.proposal_ledger.unreviewable_proposals),
            self.proposal_ledger.unreviewable_proposals,
            len(evaluations),
            None if not evaluations else evaluations[-1].evaluation_id,
        )

    @property
    def pending_proposals(self) -> tuple[PatchProposal, ...]:
        return self.proposer.pending

    @property
    def evidence_proposals(self) -> tuple[PatchProposalV1, ...]:
        return self.proposer.rich_pending

    def reject_evidence_proposal(
        self,
        proposal_id: str,
        *,
        reason_code: str,
        evidence_sha256: str,
    ) -> NegativeProposalV1:
        """Archive review/held-out rejection and suppress the same replacement."""

        return self.proposer.reject_rich(
            proposal_id,
            reason_code=reason_code,
            evidence_sha256=evidence_sha256,
        )

    @property
    def candidate_evaluations(self) -> tuple[CandidateEvaluationV1, ...]:
        if self.evaluation_ledger is None:
            return ()
        return self.evaluation_ledger.records()

    def promote_approved_once(
        self,
        evaluation_id: str,
        approval: HumanApprovalV1,
    ) -> PromotionResult:
        """Run one fresh drain/checkpoint transaction for signed promotion."""

        if self.config.promotion_mode != PromotionMode.ATTESTED:
            raise ApprovalError("source promotion is disabled in proposal-only mode")
        if not self.failure_daemon.running:
            raise RuntimeError("production self-harness is not running")
        if self.evaluation_ledger is None:
            raise ApprovalError("candidate evaluation ledger is unavailable")
        evaluation = self.evaluation_ledger.load(evaluation_id)
        patch = dict(self.proposal_ledger.reviewable_patches).get(
            evaluation.proposal_id
        )
        if not isinstance(patch, PatchProposal):
            raise ApprovalError("approved proposal is no longer reviewable")
        patcher = self.harness.patcher
        if not isinstance(patcher, JournaledHarnessPatcher):
            raise ApprovalError("approved promotion patcher is unavailable")
        with self._lock:
            self.rhythm.enter_evolution(self.harness.checkpoint)
            try:
                result = patcher.promote_approved_once(
                    patch,
                    evaluation_id=evaluation_id,
                    approval=approval,
                )
                self.rhythm.resume_inference(
                    "approved patch promoted"
                    if result.promoted
                    else "approved patch rolled back after health failure"
                )
                self.logdb.audit(
                    "approved_candidate",
                    evaluation.proposal_id,
                    "promoted" if result.promoted else "rolled_back",
                )
                return result
            except BaseException:
                if self.rhythm.mode == SystemMode.EVOLUTION:
                    self.rhythm.resume_inference("approved promotion rejected")
                elif self.rhythm.mode == SystemMode.VALIDATING:
                    self.rhythm.transition(
                        SystemMode.ROLLING_BACK,
                        "approved promotion validation failed",
                    )
                    self.rhythm.transition(
                        SystemMode.INFERENCE,
                        "approved promotion validation rollback complete",
                    )
                raise

    def rollback_committed_once(
        self,
        record_id: str,
        authorization: RollbackAuthorizationV1,
    ) -> OperatorRollbackResult:
        """Run one fresh drain/checkpoint transaction for signed rollback."""

        if self.config.promotion_mode != PromotionMode.ATTESTED:
            raise ApprovalError("operator rollback is disabled in proposal-only mode")
        if not self.failure_daemon.running:
            raise RuntimeError("production self-harness is not running")
        patcher = self.harness.patcher
        if not isinstance(patcher, JournaledHarnessPatcher):
            raise ApprovalError("operator rollback patcher is unavailable")
        with self._lock:
            self.rhythm.enter_evolution(self.harness.checkpoint)
            try:
                result = patcher.rollback_committed_once(
                    record_id,
                    authorization,
                )
                self.rhythm.resume_inference(
                    "signed operator rollback complete"
                    if result.rolled_back
                    else "operator rollback rejected by health check",
                )
                self.logdb.audit(
                    "operator_rollback",
                    result.journal_record.record_id,
                    "operator_rolled_back"
                    if result.rolled_back
                    else "committed_source_reapplied",
                )
                return result
            except BaseException:
                if self.rhythm.mode == SystemMode.EVOLUTION:
                    self.rhythm.resume_inference("operator rollback rejected")
                elif self.rhythm.mode == SystemMode.VALIDATING:
                    self.rhythm.transition(
                        SystemMode.ROLLING_BACK,
                        "operator rollback validation failed",
                    )
                    self.rhythm.transition(
                        SystemMode.INFERENCE,
                        "operator rollback validation rollback complete",
                    )
                raise

    def _record_failure_evidence(
        self,
        workflow_id: str,
        error: BaseException,
        *,
        verifier_code: str,
        mechanism: str,
    ) -> None:
        signature = (type(error).__name__, verifier_code, mechanism)
        message = str(error) or type(error).__name__
        evidence = sha256(
            "\0".join((workflow_id, *signature, message)).encode("utf-8")
        ).hexdigest()
        reproduction = (
            f"workflow={workflow_id}; verifier={verifier_code}; "
            f"mechanism={mechanism}; failure={type(error).__name__}: {message}"
        )[:4_096]
        with self._lock:
            self.proposal_ledger.record_failure(
                terminal_verifier_cause=signature[0],
                causal_status=signature[1],
                agent_mechanism=signature[2],
                primary_evidence_sha256=evidence,
                source_sha256=self._source_digest_resolver(signature),
                reproduction=reproduction,
            )

    def _record_success_evidence(
        self,
        workflow_id: str,
        *,
        verifier_code: str,
        mechanism: str,
        evidence_payload: str,
    ) -> None:
        evidence = sha256(
            "\0".join((workflow_id, verifier_code, mechanism, evidence_payload)).encode(
                "utf-8"
            )
        ).hexdigest()
        with self._lock:
            self.proposal_ledger.record_success(
                verifier_code=verifier_code,
                causal_status="verified_success",
                agent_mechanism=mechanism,
                primary_evidence_sha256=evidence,
            )

    def start(self) -> ProductionSelfHarness:
        with self._lock:
            if self.failure_daemon.running:
                return self
            prepared = any(
                record.status == "prepared" for record in self.journal.records()
            )
            if prepared:
                self.rhythm.enter_evolution(self.harness.checkpoint)
                try:
                    with self.rhythm.evolution_slot():
                        recovered = self.journal.recover_incomplete()
                    self.rhythm.resume_inference("startup journal recovery complete")
                    self.logdb.audit(
                        "journal_recovery",
                        "self_harness",
                        f"restored={recovered}",
                    )
                except BaseException:
                    if self.rhythm.mode == SystemMode.EVOLUTION:
                        self.rhythm.transition(
                            SystemMode.SAFE_MODE,
                            "startup journal recovery integrity failure",
                        )
                    raise
            self.failure_daemon.start()
            self.scheduler.note_activity()
        return self

    def stop(self) -> None:
        self.failure_daemon.stop()

    def note_activity(self) -> None:
        self.scheduler.note_activity()

    def capture_exception(
        self,
        workflow_id: str,
        error: BaseException,
        *,
        verifier_code: str = "workflow_runtime",
        mechanism: str = "workflow",
    ) -> None:
        self._record_failure_evidence(
            workflow_id,
            error,
            verifier_code=verifier_code,
            mechanism=mechanism,
        )
        self.failure_daemon.capture_exception(
            workflow_id,
            error,
            verifier_code=verifier_code,
            mechanism=mechanism,
        )
        self.note_activity()

    def capture_timeout(
        self,
        workflow_id: str,
        timeout_seconds: float,
        *,
        verifier_code: str = "workflow_deadline",
        mechanism: str = "workflow",
    ) -> None:
        timeout_error = TimeoutError(f"deadline exceeded after {timeout_seconds:g}s")
        self._record_failure_evidence(
            workflow_id,
            timeout_error,
            verifier_code=verifier_code,
            mechanism=mechanism,
        )
        self.failure_daemon.capture_timeout(
            workflow_id,
            timeout_seconds,
            verifier_code=verifier_code,
            mechanism=mechanism,
        )
        self.note_activity()

    @contextmanager
    def observe_workflow(
        self,
        workflow_id: str,
        *,
        verifier_code: str = "workflow_runtime",
        mechanism: str = "workflow",
    ) -> Iterator[None]:
        self.note_activity()
        try:
            try:
                with self.failure_daemon.observe(
                    workflow_id,
                    verifier_code=verifier_code,
                    mechanism=mechanism,
                ):
                    yield
            except BaseException as error:
                self._record_failure_evidence(
                    workflow_id,
                    error,
                    verifier_code=verifier_code,
                    mechanism=mechanism,
                )
                raise
            else:
                self._record_success_evidence(
                    workflow_id,
                    verifier_code=verifier_code,
                    mechanism=mechanism,
                    evidence_payload=self._source_surface_digest(),
                )
        finally:
            self.note_activity()

    def tick(self) -> ScheduleTick:
        if not self.failure_daemon.running:
            raise RuntimeError("production self-harness is not running")
        return self.scheduler.tick()

    def __enter__(self) -> ProductionSelfHarness:
        return self.start()

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.stop()

    def _run_cycle(self) -> EvolutionReport:
        cutoff = time()
        source_before = self._source_surface_digest()
        report = self.harness.run_night_cycle(since=self._failure_cursor)
        source_after = self._source_surface_digest()
        if source_before != source_after:
            proposal_only = self.config.promotion_mode == PromotionMode.PROPOSAL_ONLY
            self.rhythm.transition(
                SystemMode.SAFE_MODE,
                "proposal-only source surface changed"
                if proposal_only
                else "scheduled Self-Harness cycle changed the source surface",
            )
            raise JournalIntegrityError(
                "proposal-only cycle changed the active source surface"
                if proposal_only
                else "scheduled Self-Harness cycle changed the active source surface"
            )
        if self.config.promotion_mode == PromotionMode.PROPOSAL_ONLY:
            self._record_success_evidence(
                "proposal-only-night-cycle",
                verifier_code="source_surface_immutable",
                mechanism="self_harness",
                evidence_payload=source_before + source_after,
            )
        self._failure_cursor = cutoff
        return report


def _source_surface_sha256(
    project_root: Path,
    allowed_roots: tuple[str, ...],
    *,
    max_files: int = 4_096,
    max_total_bytes: int = 128 * 1024**2,
) -> str:
    """Hash the bounded mutable Python surface without following links."""

    digest = sha256()
    file_count = 0
    total_bytes = 0
    for root_name in sorted(allowed_roots):
        source_root = project_root / root_name
        if source_root.is_symlink():
            raise JournalIntegrityError("mutable source root is linked")
        if not source_root.exists():
            continue
        if not source_root.is_dir():
            raise JournalIntegrityError("mutable source root is not a directory")
        for unresolved in sorted(source_root.rglob("*.py")):
            if unresolved.is_symlink():
                raise JournalIntegrityError("mutable source surface contains a link")
            stat = os.lstat(unresolved)
            if getattr(stat, "st_file_attributes", 0) & 0x400:
                raise JournalIntegrityError(
                    "mutable source surface contains a reparse point"
                )
            path = unresolved.resolve(strict=True)
            if project_root not in path.parents or not path.is_file():
                raise JournalIntegrityError(
                    "mutable source surface escaped the project"
                )
            payload = path.read_bytes()
            file_count += 1
            total_bytes += len(payload)
            if file_count > max_files or total_bytes > max_total_bytes:
                raise JournalIntegrityError("mutable source surface exceeded its bound")
            relative = path.relative_to(project_root).as_posix().encode("utf-8")
            digest.update(len(relative).to_bytes(4, "big"))
            digest.update(relative)
            digest.update(len(payload).to_bytes(8, "big"))
            digest.update(sha256(payload).digest())
    return digest.hexdigest()


def build_production_self_harness(
    project_root: str | Path,
    model: Any,
    tokenizer: Any,
    target_allowlist: Mapping[tuple[str, str, str], str],
    checkpoint: Callable[[], None],
    *,
    config: ProductionHarnessConfig | None = None,
    rhythm: RhythmController | None = None,
    runner: AttestedSandboxRunner | None = None,
    approval_verifier: Ed25519ApprovalVerifier | None = None,
    clock: Callable[[], float] | None = None,
) -> ProductionSelfHarness:
    """Build a local Self-Harness without implicitly enabling code mutation."""

    selected = config or ProductionHarnessConfig()
    root = Path(project_root).resolve(strict=True)
    if not root.is_dir():
        raise ValueError("project_root must be an existing directory")
    if not callable(checkpoint):
        raise TypeError("checkpoint must be callable")
    verified_runner: SandboxRunner | None = None
    if selected.promotion_mode == PromotionMode.ATTESTED:
        _require_kernel_promotion_platform()
        if runner is None:
            raise IsolationAttestationError(
                "attested promotion mode requires an injected runner"
            )
        # Verify the external trust boundary before creating state files.
        verified_runner = verify_runner_attestation(runner, selected)
        if not isinstance(approval_verifier, Ed25519ApprovalVerifier):
            raise ApprovalError(
                "attested promotion requires a pinned Ed25519 approval verifier"
            )
    state_root = (root / selected.state_directory).resolve()
    if root not in state_root.parents:
        raise ValueError("state directory escaped project root")
    state_root.mkdir(parents=True, exist_ok=True)
    policy = PatchPolicy(
        allowed_roots=selected.allowed_roots,
        max_bytes=selected.max_patch_bytes,
    )
    resolver = AllowlistedTargetResolver(
        root,
        target_allowlist,
        policy,
        max_targets=selected.max_target_mappings,
    )

    def source_surface_digest() -> str:
        return _source_surface_sha256(root, selected.allowed_roots)

    def source_digest_resolver(signature: tuple[str, str, str]) -> str:
        target = resolver(WeaknessCluster(signature, ()))
        return source_surface_digest() if target is None else target.base_sha256

    local_proposer = LocalGemmaPatchProposer(
        model,
        tokenizer,
        resolver,
        policy=policy,
    )
    proposal_ledger = ProposalOnlySelfHarness(
        root,
        state_root / "proposal-ledger-v1",
        policy=policy,
        minimum_candidates=3,
        max_events=min(4_096, selected.max_failure_records),
        max_proposals=max(3, min(256, selected.max_pending_proposals * 8)),
        max_negatives=min(512, selected.max_audit_records),
    )
    proposer = RecordingProposer(
        local_proposer,
        selected.max_pending_proposals,
        evidence_ledger=proposal_ledger,
    )
    active_rhythm = rhythm or RhythmController()
    logdb = BoundedLogDB(
        state_root / "events.sqlite3",
        max_failure_records=selected.max_failure_records,
        max_audit_records=selected.max_audit_records,
    )
    daemon = FailureCaptureDaemon(
        logdb,
        capacity=selected.queue_capacity,
        excerpt_limit=selected.excerpt_limit,
    )
    journal = BackupJournal(
        root,
        state_root / "journal",
        allowed_roots=selected.allowed_roots,
        max_records=selected.max_journal_records,
        max_backup_bytes=selected.max_patch_bytes,
    )
    evaluation_ledger: CandidateEvaluationLedger | None = None
    proposal_only_reason: str | None
    if selected.promotion_mode == PromotionMode.PROPOSAL_ONLY:
        patcher: SafeHarnessPatcher = SafeHarnessPatcher(
            root,
            active_rhythm,
            policy=policy,
        )
        proposal_only_reason = (
            "promotion disabled: proposal-only mode requires an explicitly "
            "trusted kernel-isolation attestation"
        )
    else:
        assert verified_runner is not None
        assert approval_verifier is not None
        evaluation_ledger = CandidateEvaluationLedger(
            state_root / "candidate-evaluations-v1",
            max_records=selected.max_evaluation_records,
        )
        consumed_approvals = ConsumedApprovalLedger(
            state_root / "consumed-approvals-v1",
            max_records=selected.max_consumed_approvals,
        )
        patcher = JournaledHarnessPatcher(
            root,
            active_rhythm,
            policy=policy,
            sandbox=verified_runner,
            journal=journal,
            test_command=selected.regression_command,
            health_check_command=selected.health_check_command,
            timeout_seconds=selected.regression_timeout_seconds,
            health_timeout_seconds=selected.health_timeout_seconds,
            ignored_stage_root=Path(selected.state_directory).parts[0],
            evaluation_ledger=evaluation_ledger,
            approval_verifier=approval_verifier,
            consumed_approvals=consumed_approvals,
            source_surface_digest=source_surface_digest,
            evaluation_ttl_seconds=selected.evaluation_ttl_seconds,
        )
        proposal_only_reason = None
    harness = SelfHarness(
        active_rhythm,
        logdb,
        patcher,
        proposer,
        checkpoint,
        proposal_only_reason=proposal_only_reason,
    )
    return ProductionSelfHarness(
        config=selected,
        rhythm=active_rhythm,
        logdb=logdb,
        daemon=daemon,
        proposer=proposer,
        proposal_ledger=proposal_ledger,
        harness=harness,
        journal=journal,
        evaluation_ledger=evaluation_ledger,
        source_digest_resolver=source_digest_resolver,
        source_surface_digest=source_surface_digest,
        clock=clock or monotonic,
    )


def _validate_command(command: tuple[str, ...], label: str) -> None:
    if (
        not isinstance(command, tuple)
        or not command
        or len(command) > 32
        or any(
            not isinstance(item, str) or not item or len(item) > 4_096 or "\x00" in item
            for item in command
        )
    ):
        raise ValueError(f"{label} must be a bounded, non-empty argv tuple")


def _require_kernel_promotion_platform() -> None:
    """Keep native Windows fail-closed until its full isolation is attested."""

    if os.name == "nt":
        raise IsolationAttestationError(
            "native Windows source promotion has no approved kernel/process/"
            "network/filesystem isolation boundary"
        )


def _require_digest(value: str, label: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a SHA-256 string")
    normalized = value.lower()
    if len(normalized) != _DIGEST_LENGTH or any(
        character not in "0123456789abcdef" for character in normalized
    ):
        raise ValueError(f"{label} must be a SHA-256 string")
    return normalized


def _atomic_write_bytes(
    target: Path,
    payload: bytes,
    *,
    exclusive: bool = False,
    mode: int = 0o600,
) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if exclusive:
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        descriptor = os.open(target, flags, mode)
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(target, mode)
        _fsync_directory(target.parent)
        return
    temporary = target.with_name(f"{target.name}.cogni-tmp-{token_hex(8)}")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(temporary, mode)
        os.replace(temporary, target)
        _fsync_directory(target.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _fsync_directory(directory: Path) -> None:
    """Persist directory entries where the host exposes POSIX directory fsync."""

    if os.name == "nt":
        return
    descriptor = os.open(directory, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
