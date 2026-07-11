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
import shutil
import sys
import tempfile
from threading import RLock
from time import monotonic, time, time_ns
from typing import Any, Callable, Iterator, Mapping, Protocol

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
)
from .local_proposer import LocalGemmaPatchProposer, ResolvedPatchTarget
from .logdb import LogDB
from .rhythm import RhythmController, SystemMode
from .scheduler import IdleNightScheduler, ScheduleTick


_DIGEST_LENGTH = 64
_MAX_RUNNER_OUTPUT_CHARS = 40_000
_MAX_JOURNAL_METADATA_BYTES = 16_384
_TERMINAL_JOURNAL_STATES = {
    "committed",
    "rolled_back",
    "recovered_rollback",
    "aborted",
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


class _VerifiedRunner:
    """Private adapter created only after explicit attestation verification."""

    kernel_isolated = True

    def __init__(
        self,
        delegate: AttestedSandboxRunner,
        allowed_commands: tuple[str, ...],
    ) -> None:
        self._delegate = delegate
        self._allowed_commands = frozenset(allowed_commands)

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
    return _VerifiedRunner(runner, tuple(required_commands))


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
    """Keep a bounded in-memory review queue without persisting source text."""

    def __init__(self, proposer: LocalGemmaPatchProposer, capacity: int) -> None:
        self.proposer = proposer
        self._pending: deque[PatchProposal] = deque(maxlen=capacity)
        self._lock = RLock()

    def __call__(self, cluster) -> tuple[PatchProposal, ...]:
        proposals = tuple(self.proposer(cluster))
        with self._lock:
            self._pending.extend(proposals)
        return proposals

    @property
    def pending(self) -> tuple[PatchProposal, ...]:
        with self._lock:
            return tuple(self._pending)

    def clear(self) -> None:
        with self._lock:
            self._pending.clear()


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
        if status not in _TERMINAL_JOURNAL_STATES:
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


class JournaledHarnessPatcher(SafeHarnessPatcher):
    """Attested staging, journaled promotion, live health gate, and rollback."""

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

    def validate_and_promote(self, proposal: PatchProposal) -> PromotionResult:
        if self.rhythm.mode != SystemMode.EVOLUTION:
            raise RuntimeError("patching is allowed only during evolution mode")
        record: BackupRecord | None = None
        target: Path | None = None
        with self.rhythm.evolution_slot():
            target = self.validate_proposal(proposal)
            if not target.is_file():
                raise ValueError("production promotion only patches existing files")
            relative = target.relative_to(self.project_root)
            before = target.read_bytes()
            if sha256(before).hexdigest() != proposal.base_sha256:
                raise RuntimeError("base file changed during proposal validation")
            after = proposal.replacement.encode("utf-8")
            self.rhythm.transition(
                SystemMode.VALIDATING, f"validating {relative.as_posix()}"
            )
            with tempfile.TemporaryDirectory(prefix="cogni-regression-") as tmp:
                stage = Path(tmp) / "project"
                self._copy_project(stage)
                staged_target = stage / relative
                staged_target.parent.mkdir(parents=True, exist_ok=True)
                staged_target.write_bytes(after)
                regression = self.sandbox.run(
                    stage, self.test_command, self.timeout_seconds
                )
            if not regression.passed:
                self.rhythm.transition(
                    SystemMode.EVOLUTION, "candidate failed regression tests"
                )
                return PromotionResult(False, regression, target)

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
                        SystemMode.ROLLING_BACK, "post-promotion health check failed"
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

    def _copy_project(self, destination: Path) -> None:
        shutil.copytree(
            self.project_root,
            destination,
            ignore=shutil.ignore_patterns(
                "work",
                ".git",
                "__pycache__",
                "*.pyc",
                self.ignored_stage_root,
            ),
        )

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
        harness: SelfHarness,
        journal: BackupJournal,
        clock: Callable[[], float],
    ) -> None:
        self.config = config
        self.rhythm = rhythm
        self.logdb = logdb
        self.failure_daemon = daemon
        self.proposer = proposer
        self.harness = harness
        self.journal = journal
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
        return ProductionHarnessStatus(
            self.failure_daemon.running,
            self.config.promotion_mode,
            not proposal_only,
            self.harness.proposal_only_reason,
            len(self.proposer.pending),
            self._failure_cursor,
        )

    @property
    def pending_proposals(self) -> tuple[PatchProposal, ...]:
        return self.proposer.pending

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
            with self.failure_daemon.observe(
                workflow_id,
                verifier_code=verifier_code,
                mechanism=mechanism,
            ):
                yield
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
        report = self.harness.run_night_cycle(since=self._failure_cursor)
        self._failure_cursor = cutoff
        return report


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
        if runner is None:
            raise IsolationAttestationError(
                "attested promotion mode requires an injected runner"
            )
        # Verify the external trust boundary before creating state files.
        verified_runner = verify_runner_attestation(runner, selected)
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
    local_proposer = LocalGemmaPatchProposer(
        model,
        tokenizer,
        resolver,
        policy=policy,
    )
    proposer = RecordingProposer(local_proposer, selected.max_pending_proposals)
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
        harness=harness,
        journal=journal,
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
        _fsync_directory(target.parent)
        return
    temporary = target.with_name(f"{target.name}.cogni-tmp-{token_hex(8)}")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
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
