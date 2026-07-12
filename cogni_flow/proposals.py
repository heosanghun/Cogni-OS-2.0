"""Evidence-linked, proposal-only Self-Harness for Phase 11.

The engine in this module never executes candidate code and never writes the
active source tree.  It captures successful invariants and terminal failures,
clusters the three causal fields, validates a bounded set of minimal patch
drafts, and archives rejected drafts as negative experience.  Promotion is a
separate Phase-12 concern and is intentionally absent from this API.
"""

from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass
from hashlib import sha256
import json
import os
from pathlib import Path
import stat
from time import time_ns
from typing import Any, Mapping, Sequence

from .harness import PatchPolicy, PatchProposal, WeaknessCluster


PROPOSAL_SCHEMA = "cogni-self-harness-proposal-v1"
NEGATIVE_SCHEMA = "cogni-self-harness-negative-v1"
MAX_EVENT_TEXT = 4_096
MAX_EVENTS = 4_096
MAX_PROPOSALS = 256
MAX_NEGATIVES = 512
FILE_ATTRIBUTE_REPARSE_POINT = 0x400
MAX_RECORD_BYTES = 64 * 1024
REPLACEMENT_BLOB_DIRECTORY = "replacement-blobs-v1"
REPLACEMENT_BLOB_PREFIX = "replacement-"
REPLACEMENT_BLOB_SUFFIX = ".utf8"

_FAILURE_KEYS = frozenset(
    {
        "event_id",
        "terminal_verifier_cause",
        "causal_status",
        "agent_mechanism",
        "primary_evidence_sha256",
        "source_sha256",
        "reproduction",
        "observed_ns",
    }
)
_SUCCESS_KEYS = frozenset(
    {
        "invariant_id",
        "verifier_code",
        "causal_status",
        "agent_mechanism",
        "primary_evidence_sha256",
        "observed_ns",
    }
)
_PROPOSAL_KEYS = frozenset(
    {
        "schema",
        "proposal_id",
        "signature",
        "event_ids",
        "relative_path",
        "base_sha256",
        "replacement_sha256",
        "rationale",
        "expected_behavior",
        "risk",
        "reproduction_test",
        "rollback_trigger",
        "primary_evidence_sha256",
        "source_mutation_allowed",
        "status",
    }
)
_NEGATIVE_KEYS = frozenset(
    {
        "schema",
        "proposal_id",
        "proposal_sha256",
        "reason_code",
        "evidence_sha256",
        "archived_ns",
    }
)


class ProposalOnlyError(RuntimeError):
    """Raised when proposal-only evidence or policy fails closed."""


@dataclass(frozen=True, slots=True)
class FailureEventV1:
    event_id: str
    terminal_verifier_cause: str
    causal_status: str
    agent_mechanism: str
    primary_evidence_sha256: str
    source_sha256: str
    reproduction: str
    observed_ns: int

    @property
    def signature(self) -> tuple[str, str, str]:
        return (
            self.terminal_verifier_cause,
            self.causal_status,
            self.agent_mechanism,
        )


@dataclass(frozen=True, slots=True)
class SuccessInvariantV1:
    invariant_id: str
    verifier_code: str
    causal_status: str
    agent_mechanism: str
    primary_evidence_sha256: str
    observed_ns: int


@dataclass(frozen=True, slots=True)
class CandidateDraft:
    patch: PatchProposal
    expected_behavior: str
    risk: str
    reproduction_test: str
    rollback_trigger: str


@dataclass(frozen=True, slots=True)
class PatchProposalV1:
    schema: str
    proposal_id: str
    signature: tuple[str, str, str]
    event_ids: tuple[str, ...]
    relative_path: str
    base_sha256: str
    replacement_sha256: str
    rationale: str
    expected_behavior: str
    risk: str
    reproduction_test: str
    rollback_trigger: str
    primary_evidence_sha256: tuple[str, ...]
    source_mutation_allowed: bool = False
    status: str = "pending_review"


@dataclass(frozen=True, slots=True)
class NegativeProposalV1:
    schema: str
    proposal_id: str
    proposal_sha256: str
    reason_code: str
    evidence_sha256: str
    archived_ns: int


@dataclass(frozen=True, slots=True)
class CaptureCoverage:
    attempted: int
    persisted: int

    @property
    def ratio(self) -> float:
        return 1.0 if self.attempted == 0 else self.persisted / self.attempted


def _digest(value: str, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdefABCDEF" for character in value)
    ):
        raise ValueError(f"{label} must be SHA-256 hex")
    return value.lower()


def _text(value: str, label: str, *, maximum: int = MAX_EVENT_TEXT) -> str:
    if not isinstance(value, str) or not value or len(value) > maximum:
        raise ValueError(f"{label} must be bounded non-empty text")
    if "\x00" in value:
        raise ValueError(f"{label} contains a NUL byte")
    return value


def _json_digest(payload: Mapping[str, object]) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")
    return sha256(encoded).hexdigest()


def _integer(value: object, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ProposalOnlyError(f"{label} must be a non-negative integer")
    return value


def _strict_json_object(raw: bytes, label: str) -> dict[str, object]:
    if not raw or len(raw) > MAX_RECORD_BYTES:
        raise ProposalOnlyError(f"{label} crossed its JSON size bound")

    def object_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ProposalOnlyError(f"{label} contains a duplicate JSON key")
            result[key] = value
        return result

    try:
        decoded = json.loads(raw.decode("ascii"), object_pairs_hook=object_pairs)
    except ProposalOnlyError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ProposalOnlyError(f"{label} is not canonical JSON") from error
    if not isinstance(decoded, dict):
        raise ProposalOnlyError(f"{label} must contain one JSON object")
    return decoded


def _exact_keys(
    payload: Mapping[str, object], expected: frozenset[str], label: str
) -> None:
    if frozenset(payload) != expected:
        raise ProposalOnlyError(f"{label} does not match its V1 schema")


class ProposalOnlySelfHarness:
    """Bounded proposal ledger with no source mutation or execution method."""

    IMMUTABLE_SURFACES = frozenset(
        {
            "cogni_flow/harness.py",
            "cogni_flow/production.py",
            "cogni_flow/proposals.py",
            "cogni_flow/rhythm.py",
            "cogni_flow/evolution.py",
            "cogni_os/runtime.py",
            "cogni_os/factory.py",
        }
    )

    def __init__(
        self,
        project_root: str | Path,
        state_directory: str | Path,
        *,
        policy: PatchPolicy | None = None,
        minimum_candidates: int = 3,
        max_events: int = MAX_EVENTS,
        max_proposals: int = MAX_PROPOSALS,
        max_negatives: int = MAX_NEGATIVES,
    ) -> None:
        root = Path(project_root).resolve(strict=True)
        state = Path(state_directory)
        if not state.is_absolute():
            state = root / state
        state = Path(os.path.abspath(state))
        if state == root or root not in state.parents:
            raise ValueError("proposal state must remain inside the project root")
        cursor = root
        for component in state.relative_to(root).parts:
            cursor = cursor / component
            if cursor.exists() or cursor.is_symlink():
                item_stat = os.lstat(cursor)
                attributes = getattr(item_stat, "st_file_attributes", 0)
                if cursor.is_symlink() or attributes & FILE_ATTRIBUTE_REPARSE_POINT:
                    raise ProposalOnlyError(
                        "proposal state traverses a link/reparse point"
                    )
                if not cursor.is_dir():
                    raise ProposalOnlyError(
                        "proposal state path contains a non-directory"
                    )
            else:
                cursor.mkdir()
        state = state.resolve(strict=True)
        self.policy = policy or PatchPolicy()
        if state == root or root not in state.parents:
            raise ValueError("proposal state must remain inside the project root")
        relative_state = state.relative_to(root)
        if relative_state.parts[0] in self.policy.allowed_roots:
            raise ValueError("proposal state must remain outside mutable source roots")
        if not 1 <= minimum_candidates <= 8:
            raise ValueError("minimum_candidates must be in [1, 8]")
        for value, limit, label in (
            (max_events, MAX_EVENTS, "max_events"),
            (max_proposals, MAX_PROPOSALS, "max_proposals"),
            (max_negatives, MAX_NEGATIVES, "max_negatives"),
        ):
            if (
                not isinstance(value, int)
                or isinstance(value, bool)
                or not 1 <= value <= limit
            ):
                raise ValueError(f"{label} crossed its hard bound")
        self.project_root = root
        self.state_directory = state
        self.replacement_blob_directory = state / REPLACEMENT_BLOB_DIRECTORY
        try:
            self.replacement_blob_directory.mkdir(mode=0o700)
        except FileExistsError:
            pass
        self._require_blob_directory()
        self.minimum_candidates = minimum_candidates
        self.max_events = max_events
        self.max_proposals = max_proposals
        self.max_negatives = max_negatives
        self._events: deque[FailureEventV1] = deque(maxlen=max_events)
        self._successes: deque[SuccessInvariantV1] = deque(maxlen=max_events)
        self._proposals: deque[PatchProposalV1] = deque(maxlen=max_proposals)
        self._negatives: deque[NegativeProposalV1] = deque(maxlen=max_negatives)
        self._reviewable_patches: dict[str, PatchProposal] = {}
        self._unreviewable_proposals: dict[str, str] = {}
        self._capture_attempted = 0
        self._capture_persisted = 0
        self._hydrate()

    @property
    def capture_coverage(self) -> CaptureCoverage:
        return CaptureCoverage(self._capture_attempted, self._capture_persisted)

    @property
    def failures(self) -> tuple[FailureEventV1, ...]:
        return tuple(self._events)

    @property
    def successes(self) -> tuple[SuccessInvariantV1, ...]:
        return tuple(self._successes)

    @property
    def proposals(self) -> tuple[PatchProposalV1, ...]:
        return tuple(self._proposals)

    @property
    def negative_archive(self) -> tuple[NegativeProposalV1, ...]:
        return tuple(self._negatives)

    @property
    def reviewable_patches(self) -> tuple[tuple[str, PatchProposal], ...]:
        """Return source-bearing candidates that passed current hydration gates."""

        return tuple(
            (proposal.proposal_id, self._reviewable_patches[proposal.proposal_id])
            for proposal in self._proposals
            if proposal.proposal_id in self._reviewable_patches
        )

    @property
    def unreviewable_proposals(self) -> tuple[tuple[str, str], ...]:
        """Expose bounded integrity failures without deleting their evidence."""

        return tuple(sorted(self._unreviewable_proposals.items()))

    def _require_blob_directory(self) -> None:
        try:
            item_stat = os.lstat(self.replacement_blob_directory)
        except FileNotFoundError as error:
            raise ProposalOnlyError("replacement blob directory is missing") from error
        attributes = getattr(item_stat, "st_file_attributes", 0)
        if (
            self.replacement_blob_directory.is_symlink()
            or attributes & FILE_ATTRIBUTE_REPARSE_POINT
            or not stat.S_ISDIR(item_stat.st_mode)
            or self.replacement_blob_directory.resolve(strict=True)
            != self.replacement_blob_directory.absolute()
            or self.replacement_blob_directory.parent != self.state_directory
        ):
            raise ProposalOnlyError(
                "replacement blob directory is not a regular non-reparse directory"
            )

    def _hydrate(self) -> None:
        """Load strict evidence and expose only candidates with valid source blobs."""

        try:
            records = self._state_records()
            failures = tuple(
                self._parse_failure(path, payload)
                for path, payload in records["failure"]
            )
            successes = tuple(
                self._parse_success(path, payload)
                for path, payload in records["success"]
            )
            failure_by_id = {item.event_id: item for item in failures}
            if len(failure_by_id) != len(failures):
                raise ProposalOnlyError("failure ledger contains a duplicate ID")
            proposals = tuple(
                self._parse_proposal(path, payload, failure_by_id)
                for path, payload in records["proposal"]
            )
            proposal_by_id = {item.proposal_id: item for item in proposals}
            if len(proposal_by_id) != len(proposals):
                raise ProposalOnlyError("proposal ledger contains a duplicate ID")
            negatives = tuple(
                self._parse_negative(path, payload, proposal_by_id)
                for path, payload in records["negative"]
            )
            rejected_ids = {item.proposal_id for item in negatives}
            if len(rejected_ids) != len(negatives):
                raise ProposalOnlyError(
                    "negative ledger contains a duplicate proposal reference"
                )
            reviewable: dict[str, PatchProposal] = {}
            unreviewable: dict[str, str] = {}
            signature_counts: dict[tuple[str, str, str], int] = {}
            for proposal in proposals:
                signature_counts[proposal.signature] = (
                    signature_counts.get(proposal.signature, 0) + 1
                )
            for proposal in proposals:
                if proposal.proposal_id in rejected_ids:
                    continue
                if signature_counts[proposal.signature] < self.minimum_candidates:
                    unreviewable[proposal.proposal_id] = (
                        "proposal batch does not satisfy its candidate minimum"
                    )
                    continue
                try:
                    reviewable[proposal.proposal_id] = self._restore_patch(proposal)
                except (
                    OSError,
                    UnicodeError,
                    SyntaxError,
                    ValueError,
                    ProposalOnlyError,
                ) as error:
                    unreviewable[proposal.proposal_id] = str(error)[:512]
        except ProposalOnlyError:
            raise
        except (KeyError, TypeError, ValueError) as error:
            raise ProposalOnlyError(
                "proposal state failed strict V1 hydration"
            ) from error

        self._events.extend(
            sorted(failures, key=lambda item: (item.observed_ns, item.event_id))
        )
        self._successes.extend(
            sorted(successes, key=lambda item: (item.observed_ns, item.invariant_id))
        )
        self._proposals.extend(sorted(proposals, key=lambda item: item.proposal_id))
        self._negatives.extend(
            sorted(negatives, key=lambda item: (item.archived_ns, item.proposal_id))
        )
        self._reviewable_patches.update(reviewable)
        self._unreviewable_proposals.update(unreviewable)
        # An event file is the durable acknowledgement of one capture attempt.
        # Failed writes have no durable claim, so both recovered counters resume
        # from the exact acknowledged set instead of reporting zero coverage.
        self._capture_attempted = len(failures)
        self._capture_persisted = len(failures)

    def _state_records(
        self,
    ) -> dict[str, list[tuple[Path, dict[str, object]]]]:
        directory_stat = os.lstat(self.state_directory)
        attributes = getattr(directory_stat, "st_file_attributes", 0)
        if (
            self.state_directory.is_symlink()
            or attributes & FILE_ATTRIBUTE_REPARSE_POINT
            or not self.state_directory.is_dir()
        ):
            raise ProposalOnlyError(
                "proposal state directory is not a regular non-reparse directory"
            )
        records: dict[str, list[tuple[Path, dict[str, object]]]] = {
            "failure": [],
            "success": [],
            "proposal": [],
            "negative": [],
        }
        limits = {
            "failure": self.max_events,
            "success": self.max_events,
            "proposal": self.max_proposals,
            "negative": self.max_negatives,
        }
        prefixes = tuple(f"{name}-" for name in records)
        for entry in sorted(
            os.scandir(self.state_directory), key=lambda item: item.name
        ):
            entry_stat = entry.stat(follow_symlinks=False)
            entry_attributes = getattr(entry_stat, "st_file_attributes", 0)
            if entry.name == REPLACEMENT_BLOB_DIRECTORY:
                if (
                    entry.is_symlink()
                    or entry_attributes & FILE_ATTRIBUTE_REPARSE_POINT
                    or not entry.is_dir(follow_symlinks=False)
                    or Path(entry.path).resolve(strict=True)
                    != Path(entry.path).absolute()
                ):
                    raise ProposalOnlyError(
                        "replacement blob directory is not regular/non-reparse"
                    )
                continue
            if (
                entry.is_symlink()
                or entry_attributes & FILE_ATTRIBUTE_REPARSE_POINT
                or not entry.is_file(follow_symlinks=False)
            ):
                raise ProposalOnlyError(
                    "proposal state contains a non-regular/reparse entry"
                )
            kind = next(
                (
                    name
                    for name, prefix in zip(records, prefixes)
                    if entry.name.startswith(prefix)
                ),
                None,
            )
            if kind is None or not entry.name.endswith(".json"):
                raise ProposalOnlyError("proposal state contains an unknown file")
            identity = entry.name[len(kind) + 1 : -5]
            if _digest(identity, f"{kind} filename ID") != identity:
                raise ProposalOnlyError(
                    "proposal state filename is not canonical lowercase SHA-256"
                )
            if len(records[kind]) >= limits[kind]:
                raise ProposalOnlyError(f"{kind} ledger crossed its hard bound")
            path = Path(entry.path)
            if path.resolve(strict=True) != path.absolute():
                raise ProposalOnlyError(
                    "proposal state record traverses a link/reparse point"
                )
            records[kind].append(
                (
                    path,
                    _strict_json_object(
                        self._read_bounded_regular(path, entry_stat), entry.name
                    ),
                )
            )
        return records

    @staticmethod
    def _read_bounded_regular(path: Path, expected_stat: os.stat_result) -> bytes:
        if expected_stat.st_size > MAX_RECORD_BYTES:
            raise ProposalOnlyError(f"{path.name} crossed its JSON size bound")
        with path.open("rb") as stream:
            opened_stat = os.fstat(stream.fileno())
            attributes = getattr(opened_stat, "st_file_attributes", 0)
            identity_changed = bool(expected_stat.st_ino and opened_stat.st_ino) and (
                (expected_stat.st_dev, expected_stat.st_ino)
                != (opened_stat.st_dev, opened_stat.st_ino)
            )
            if (
                not stat.S_ISREG(opened_stat.st_mode)
                or attributes & FILE_ATTRIBUTE_REPARSE_POINT
                or identity_changed
            ):
                raise ProposalOnlyError(
                    "proposal state record changed during hydration"
                )
            raw = stream.read(MAX_RECORD_BYTES + 1)
            final_stat = os.fstat(stream.fileno())
        if (
            len(raw) > MAX_RECORD_BYTES
            or opened_stat.st_size != len(raw)
            or (opened_stat.st_size, opened_stat.st_mtime_ns)
            != (final_stat.st_size, final_stat.st_mtime_ns)
        ):
            raise ProposalOnlyError(
                "proposal state record changed or crossed its bound during hydration"
            )
        return raw

    @staticmethod
    def _parse_failure(path: Path, payload: Mapping[str, object]) -> FailureEventV1:
        _exact_keys(payload, _FAILURE_KEYS, path.name)
        event_id = _digest(payload["event_id"], "failure event ID")
        event = FailureEventV1(
            event_id,
            _text(
                payload["terminal_verifier_cause"],
                "terminal verifier cause",
                maximum=256,
            ),
            _text(payload["causal_status"], "causal status", maximum=256),
            _text(payload["agent_mechanism"], "agent mechanism", maximum=256),
            _digest(payload["primary_evidence_sha256"], "primary evidence"),
            _digest(payload["source_sha256"], "source"),
            _text(payload["reproduction"], "reproduction"),
            _integer(payload["observed_ns"], "failure observed_ns"),
        )
        expected = _json_digest(
            {
                "cause": event.terminal_verifier_cause,
                "causal": event.causal_status,
                "mechanism": event.agent_mechanism,
                "evidence": event.primary_evidence_sha256,
                "source": event.source_sha256,
                "reproduction": event.reproduction,
                "observed_ns": event.observed_ns,
            }
        )
        if event_id != expected or path.name != f"failure-{event_id}.json":
            raise ProposalOnlyError("failure event identity hash does not verify")
        return event

    @staticmethod
    def _parse_success(path: Path, payload: Mapping[str, object]) -> SuccessInvariantV1:
        _exact_keys(payload, _SUCCESS_KEYS, path.name)
        invariant_id = _digest(payload["invariant_id"], "success invariant ID")
        item = SuccessInvariantV1(
            invariant_id,
            _text(payload["verifier_code"], "verifier code", maximum=256),
            _text(payload["causal_status"], "causal status", maximum=256),
            _text(payload["agent_mechanism"], "agent mechanism", maximum=256),
            _digest(payload["primary_evidence_sha256"], "primary evidence"),
            _integer(payload["observed_ns"], "success observed_ns"),
        )
        expected = _json_digest(
            {
                "code": item.verifier_code,
                "causal": item.causal_status,
                "mechanism": item.agent_mechanism,
                "evidence": item.primary_evidence_sha256,
                "observed_ns": item.observed_ns,
            }
        )
        if invariant_id != expected or path.name != f"success-{invariant_id}.json":
            raise ProposalOnlyError("success invariant identity hash does not verify")
        return item

    def _parse_proposal(
        self,
        path: Path,
        payload: Mapping[str, object],
        failure_by_id: Mapping[str, FailureEventV1],
    ) -> PatchProposalV1:
        _exact_keys(payload, _PROPOSAL_KEYS, path.name)
        if payload["schema"] != PROPOSAL_SCHEMA:
            raise ProposalOnlyError("proposal schema is not supported")
        if payload["source_mutation_allowed"] is not False:
            raise ProposalOnlyError("hydrated proposal attempted to enable mutation")
        if payload["status"] != "pending_review":
            raise ProposalOnlyError("proposal has an unsupported lifecycle status")
        signature_value = payload["signature"]
        event_value = payload["event_ids"]
        evidence_value = payload["primary_evidence_sha256"]
        if not isinstance(signature_value, list) or len(signature_value) != 3:
            raise ProposalOnlyError("proposal signature must contain three fields")
        signature = tuple(
            _text(item, "proposal signature", maximum=256) for item in signature_value
        )
        if (
            not isinstance(event_value, list)
            or not 1 <= len(event_value) <= self.max_events
        ):
            raise ProposalOnlyError("proposal event references crossed their bound")
        event_ids = tuple(_digest(item, "proposal event ID") for item in event_value)
        if len(set(event_ids)) != len(event_ids):
            raise ProposalOnlyError("proposal repeats a failure event reference")
        if (
            not isinstance(evidence_value, list)
            or not 1 <= len(evidence_value) <= self.max_events
        ):
            raise ProposalOnlyError("proposal evidence references crossed their bound")
        primary = tuple(
            _digest(item, "proposal primary evidence") for item in evidence_value
        )
        if primary != tuple(sorted(set(primary))):
            raise ProposalOnlyError(
                "proposal primary evidence must be sorted and unique"
            )
        proposal_id = _digest(payload["proposal_id"], "proposal ID")
        proposal = PatchProposalV1(
            PROPOSAL_SCHEMA,
            proposal_id,
            signature,
            event_ids,
            _text(payload["relative_path"], "proposal path", maximum=512),
            _digest(payload["base_sha256"], "proposal base"),
            _digest(payload["replacement_sha256"], "proposal replacement"),
            _text(payload["rationale"], "proposal rationale"),
            _text(payload["expected_behavior"], "expected behavior"),
            _text(payload["risk"], "risk", maximum=1_024),
            _text(payload["reproduction_test"], "reproduction test"),
            _text(payload["rollback_trigger"], "rollback trigger"),
            primary,
        )
        identity_payload = asdict(proposal)
        identity_payload.pop("proposal_id")
        expected_id = _json_digest(identity_payload)
        if proposal_id != expected_id or path.name != f"proposal-{proposal_id}.json":
            raise ProposalOnlyError("proposal identity hash does not verify")
        try:
            events = tuple(failure_by_id[event_id] for event_id in event_ids)
        except KeyError as error:
            raise ProposalOnlyError(
                "proposal references an unknown failure event"
            ) from error
        if any(event.signature != signature for event in events):
            raise ProposalOnlyError("proposal/failure signatures do not match")
        if proposal.base_sha256 not in {event.source_sha256 for event in events}:
            raise ProposalOnlyError("proposal base is not linked to failure evidence")
        expected_primary = tuple(
            sorted({event.primary_evidence_sha256 for event in events})
        )
        if proposal.primary_evidence_sha256 != expected_primary:
            raise ProposalOnlyError(
                "proposal primary evidence cross-reference does not verify"
            )
        return proposal

    @staticmethod
    def _parse_negative(
        path: Path,
        payload: Mapping[str, object],
        proposal_by_id: Mapping[str, PatchProposalV1],
    ) -> NegativeProposalV1:
        _exact_keys(payload, _NEGATIVE_KEYS, path.name)
        if payload["schema"] != NEGATIVE_SCHEMA:
            raise ProposalOnlyError("negative proposal schema is not supported")
        proposal_id = _digest(payload["proposal_id"], "negative proposal ID")
        try:
            proposal = proposal_by_id[proposal_id]
        except KeyError as error:
            raise ProposalOnlyError(
                "negative record references an unknown proposal"
            ) from error
        negative = NegativeProposalV1(
            NEGATIVE_SCHEMA,
            proposal_id,
            _digest(payload["proposal_sha256"], "negative proposal digest"),
            _text(payload["reason_code"], "negative reason", maximum=256),
            _digest(payload["evidence_sha256"], "negative evidence"),
            _integer(payload["archived_ns"], "negative archived_ns"),
        )
        if path.name != f"negative-{proposal_id}.json":
            raise ProposalOnlyError("negative record filename does not match proposal")
        if negative.proposal_sha256 != _json_digest(asdict(proposal)):
            raise ProposalOnlyError("negative proposal digest does not verify")
        return negative

    def _validate_active_target(self, proposal: PatchProposalV1) -> None:
        relative = Path(proposal.relative_path)
        if (
            relative.is_absolute()
            or not relative.parts
            or ".." in relative.parts
            or relative.as_posix() != proposal.relative_path
            or relative.parts[0] not in self.policy.allowed_roots
            or relative.suffix != ".py"
            or proposal.relative_path in self.IMMUTABLE_SURFACES
        ):
            raise ProposalOnlyError("pending proposal target is outside policy")
        cursor = self.project_root
        for component in relative.parts:
            cursor = cursor / component
            if not (cursor.exists() or cursor.is_symlink()):
                raise ProposalOnlyError("pending proposal target no longer exists")
            item_stat = os.lstat(cursor)
            attributes = getattr(item_stat, "st_file_attributes", 0)
            if cursor.is_symlink() or attributes & FILE_ATTRIBUTE_REPARSE_POINT:
                raise ProposalOnlyError(
                    "pending proposal target traverses a link/reparse point"
                )
        target = (self.project_root / relative).resolve(strict=True)
        if self.project_root not in target.parents or not target.is_file():
            raise ProposalOnlyError("pending proposal target is not a regular file")
        if sha256(target.read_bytes()).hexdigest() != proposal.base_sha256:
            raise ProposalOnlyError("pending proposal base digest is stale")

    def _replacement_blob_path(self, replacement_sha256: str) -> Path:
        digest = _digest(replacement_sha256, "replacement blob")
        return self.replacement_blob_directory / (
            f"{REPLACEMENT_BLOB_PREFIX}{digest}{REPLACEMENT_BLOB_SUFFIX}"
        )

    def _read_replacement_blob(self, replacement_sha256: str) -> str:
        self._require_blob_directory()
        path = self._replacement_blob_path(replacement_sha256)
        try:
            expected_stat = os.lstat(path)
        except FileNotFoundError as error:
            raise ProposalOnlyError("replacement blob is missing") from error
        attributes = getattr(expected_stat, "st_file_attributes", 0)
        if (
            path.is_symlink()
            or attributes & FILE_ATTRIBUTE_REPARSE_POINT
            or not stat.S_ISREG(expected_stat.st_mode)
            or path.resolve(strict=True) != path.absolute()
            or path.parent != self.replacement_blob_directory
        ):
            raise ProposalOnlyError("replacement blob is non-regular or reparse-backed")
        if expected_stat.st_size > self.policy.max_bytes:
            raise ProposalOnlyError("replacement blob crossed its byte bound")
        flags = os.O_RDONLY | getattr(os, "O_BINARY", 0)
        flags |= getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(path, flags)
        try:
            opened_stat = os.fstat(descriptor)
            opened_attributes = getattr(opened_stat, "st_file_attributes", 0)
            identity_changed = bool(expected_stat.st_ino and opened_stat.st_ino) and (
                (expected_stat.st_dev, expected_stat.st_ino)
                != (opened_stat.st_dev, opened_stat.st_ino)
            )
            if (
                not stat.S_ISREG(opened_stat.st_mode)
                or opened_attributes & FILE_ATTRIBUTE_REPARSE_POINT
                or identity_changed
                or opened_stat.st_size > self.policy.max_bytes
            ):
                raise ProposalOnlyError("replacement blob changed before open")
            with os.fdopen(descriptor, "rb", closefd=False) as stream:
                raw = stream.read(self.policy.max_bytes + 1)
            final_stat = os.fstat(descriptor)
        finally:
            os.close(descriptor)
        if (
            len(raw) > self.policy.max_bytes
            or len(raw) != opened_stat.st_size
            or (opened_stat.st_size, opened_stat.st_mtime_ns)
            != (final_stat.st_size, final_stat.st_mtime_ns)
        ):
            raise ProposalOnlyError(
                "replacement blob changed or crossed its bound during hydration"
            )
        try:
            replacement = raw.decode("utf-8")
        except UnicodeDecodeError as error:
            raise ProposalOnlyError("replacement blob is not strict UTF-8") from error
        if sha256(raw).hexdigest() != replacement_sha256:
            raise ProposalOnlyError("replacement blob digest does not verify")
        return replacement

    def _persist_replacement_blob(self, replacement: str) -> str:
        if not isinstance(replacement, str):
            raise TypeError("replacement must be text")
        raw = replacement.encode("utf-8")
        if len(raw) > self.policy.max_bytes:
            raise ProposalOnlyError("replacement blob crossed its byte bound")
        digest = sha256(raw).hexdigest()
        path = self._replacement_blob_path(digest)
        if path.exists() or path.is_symlink():
            if self._read_replacement_blob(digest) != replacement:
                raise ProposalOnlyError("existing replacement blob content changed")
            return digest
        self._require_blob_directory()
        temporary = self.replacement_blob_directory / f".{digest}.tmp"
        if temporary.exists() or temporary.is_symlink():
            raise ProposalOnlyError("replacement temporary blob already exists")
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        flags |= getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(temporary, flags, 0o600)
        try:
            with os.fdopen(descriptor, "wb", closefd=False) as stream:
                stream.write(raw)
                stream.flush()
                os.fsync(stream.fileno())
        finally:
            os.close(descriptor)
        try:
            if path.exists() or path.is_symlink():
                raise ProposalOnlyError("replacement blob path appeared during commit")
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)
        if self._read_replacement_blob(digest) != replacement:
            raise ProposalOnlyError("replacement blob read-back failed")
        return digest

    def _restore_patch(self, proposal: PatchProposalV1) -> PatchProposal:
        self._validate_active_target(proposal)
        replacement = self._read_replacement_blob(proposal.replacement_sha256)
        patch = PatchProposal(
            proposal.relative_path,
            proposal.base_sha256,
            replacement,
            proposal.rationale,
        )
        relative = self.policy.validate(patch)
        if relative.as_posix() != proposal.relative_path:
            raise ProposalOnlyError("replacement blob target does not match proposal")
        return patch

    def record_failure(
        self,
        *,
        terminal_verifier_cause: str,
        causal_status: str,
        agent_mechanism: str,
        primary_evidence_sha256: str,
        source_sha256: str,
        reproduction: str,
        observed_ns: int | None = None,
    ) -> FailureEventV1:
        self._capture_attempted += 1
        if len(self._events) >= self.max_events:
            raise ProposalOnlyError("failure evidence retention bound was reached")
        cause = _text(terminal_verifier_cause, "terminal verifier cause", maximum=256)
        causal = _text(causal_status, "causal status", maximum=256)
        mechanism = _text(agent_mechanism, "agent mechanism", maximum=256)
        evidence = _digest(primary_evidence_sha256, "primary evidence")
        source = _digest(source_sha256, "source")
        reproduce = _text(reproduction, "reproduction")
        timestamp = time_ns() if observed_ns is None else observed_ns
        if (
            not isinstance(timestamp, int)
            or isinstance(timestamp, bool)
            or timestamp < 0
        ):
            raise ValueError("observed_ns must be a non-negative integer")
        identity = _json_digest(
            {
                "cause": cause,
                "causal": causal,
                "mechanism": mechanism,
                "evidence": evidence,
                "source": source,
                "reproduction": reproduce,
                "observed_ns": timestamp,
            }
        )
        event = FailureEventV1(
            identity,
            cause,
            causal,
            mechanism,
            evidence,
            source,
            reproduce,
            timestamp,
        )
        self._persist_json(
            self.state_directory / f"failure-{event.event_id}.json",
            asdict(event),
        )
        self._events.append(event)
        self._capture_persisted += 1
        return event

    def record_success(
        self,
        *,
        verifier_code: str,
        causal_status: str,
        agent_mechanism: str,
        primary_evidence_sha256: str,
        observed_ns: int | None = None,
    ) -> SuccessInvariantV1:
        if len(self._successes) >= self.max_events:
            raise ProposalOnlyError("success evidence retention bound was reached")
        code = _text(verifier_code, "verifier code", maximum=256)
        causal = _text(causal_status, "causal status", maximum=256)
        mechanism = _text(agent_mechanism, "agent mechanism", maximum=256)
        evidence = _digest(primary_evidence_sha256, "primary evidence")
        timestamp = time_ns() if observed_ns is None else observed_ns
        identity = _json_digest(
            {
                "code": code,
                "causal": causal,
                "mechanism": mechanism,
                "evidence": evidence,
                "observed_ns": timestamp,
            }
        )
        item = SuccessInvariantV1(
            identity, code, causal, mechanism, evidence, timestamp
        )
        self._persist_json(
            self.state_directory / f"success-{item.invariant_id}.json",
            asdict(item),
        )
        self._successes.append(item)
        return item

    def clusters(self) -> tuple[WeaknessCluster, ...]:
        grouped: dict[tuple[str, str, str], list[FailureEventV1]] = {}
        for event in self._events:
            grouped.setdefault(event.signature, []).append(event)
        # WeaknessCluster retains the legacy FailureTrace type, so proposal V1
        # uses a lightweight compatible object only for its public signature
        # and length. The primary event ledger remains authoritative.
        from .harness import FailureTrace

        result = []
        for signature, events in grouped.items():
            traces = tuple(
                FailureTrace(
                    event.event_id,
                    event.terminal_verifier_cause,
                    event.causal_status,
                    event.agent_mechanism,
                    event.reproduction,
                )
                for event in events
            )
            result.append(WeaknessCluster(signature, traces))
        return tuple(
            sorted(result, key=lambda item: (-len(item.traces), item.signature))
        )

    def _events_for_cluster(
        self, cluster: WeaknessCluster
    ) -> tuple[FailureEventV1, ...]:
        events = tuple(
            event for event in self._events if event.signature == cluster.signature
        )
        if not events:
            raise ProposalOnlyError("cluster has no primary failure evidence")
        return events

    def _safe_target(self, patch: PatchProposal) -> Path:
        relative = self.policy.validate(patch)
        rendered = relative.as_posix()
        if patch.relative_path != rendered:
            raise ProposalOnlyError(
                "candidate target path is not canonical POSIX relative"
            )
        if rendered in self.IMMUTABLE_SURFACES:
            raise ProposalOnlyError("candidate targets an immutable control surface")
        cursor = self.project_root
        for component in relative.parts:
            cursor = cursor / component
            if cursor.exists() or cursor.is_symlink():
                stat = os.lstat(cursor)
                attributes = getattr(stat, "st_file_attributes", 0)
                if cursor.is_symlink() or attributes & FILE_ATTRIBUTE_REPARSE_POINT:
                    raise ProposalOnlyError(
                        "candidate target traverses a link/reparse point"
                    )
        target = (self.project_root / relative).resolve(strict=True)
        if self.project_root not in target.parents or not target.is_file():
            raise ProposalOnlyError("candidate target escaped the regular source tree")
        payload = target.read_bytes()
        if sha256(payload).hexdigest() != patch.base_sha256.lower():
            raise ProposalOnlyError("candidate base digest is stale")
        return target

    def submit_cluster_candidates(
        self,
        cluster: WeaknessCluster,
        drafts: Sequence[CandidateDraft],
    ) -> tuple[PatchProposalV1, ...]:
        if not self.minimum_candidates <= len(drafts) <= 8:
            raise ProposalOnlyError(
                "each failure cluster requires the configured minimum distinct candidates"
            )
        events = self._events_for_cluster(cluster)
        replacements: set[str] = set()
        prepared: list[tuple[PatchProposalV1, PatchProposal]] = []
        if len(self._proposals) + len(drafts) > self.max_proposals:
            raise ProposalOnlyError("proposal evidence retention bound was reached")
        existing_ids = {proposal.proposal_id for proposal in self._proposals}
        for draft in drafts:
            if not isinstance(draft, CandidateDraft):
                raise TypeError("candidate drafts must be CandidateDraft values")
            self._safe_target(draft.patch)
            base_sha = _digest(draft.patch.base_sha256, "candidate base")
            if base_sha not in {event.source_sha256 for event in events}:
                raise ProposalOnlyError(
                    "candidate base digest is not linked to primary failure evidence"
                )
            replacement_sha = sha256(
                draft.patch.replacement.encode("utf-8")
            ).hexdigest()
            if replacement_sha in replacements:
                raise ProposalOnlyError(
                    "cluster candidates must be structurally distinct"
                )
            replacements.add(replacement_sha)
            expected = _text(draft.expected_behavior, "expected behavior")
            risk = _text(draft.risk, "risk", maximum=1_024)
            reproduction_test = _text(draft.reproduction_test, "reproduction test")
            rollback = _text(draft.rollback_trigger, "rollback trigger")
            rationale = _text(draft.patch.rationale, "rationale")
            primary = tuple(sorted({event.primary_evidence_sha256 for event in events}))
            event_ids = tuple(event.event_id for event in events)
            payload = {
                "schema": PROPOSAL_SCHEMA,
                "signature": cluster.signature,
                "event_ids": event_ids,
                "relative_path": draft.patch.relative_path,
                "base_sha256": base_sha,
                "replacement_sha256": replacement_sha,
                "rationale": rationale,
                "expected_behavior": expected,
                "risk": risk,
                "reproduction_test": reproduction_test,
                "rollback_trigger": rollback,
                "primary_evidence_sha256": primary,
                "source_mutation_allowed": False,
                "status": "pending_review",
            }
            proposal_id = _json_digest(payload)
            proposal = PatchProposalV1(
                PROPOSAL_SCHEMA,
                proposal_id,
                cluster.signature,
                event_ids,
                draft.patch.relative_path,
                base_sha,
                replacement_sha,
                rationale,
                expected,
                risk,
                reproduction_test,
                rollback,
                primary,
            )
            if proposal_id in existing_ids or any(
                item.proposal_id == proposal_id for item, _patch in prepared
            ):
                raise ProposalOnlyError("proposal evidence identity already exists")
            proposal_path = self.state_directory / f"proposal-{proposal_id}.json"
            if proposal_path.exists() or proposal_path.is_symlink():
                raise ProposalOnlyError("proposal evidence path already exists")
            prepared.append(
                (
                    proposal,
                    PatchProposal(
                        proposal.relative_path,
                        proposal.base_sha256,
                        draft.patch.replacement,
                        proposal.rationale,
                    ),
                )
            )
        accepted = [proposal for proposal, _patch in prepared]
        accepted_patches = {proposal.proposal_id: patch for proposal, patch in prepared}
        created_proposals: list[Path] = []
        created_blobs: list[Path] = []
        prior_blob_digests = {
            proposal.replacement_sha256 for proposal in self._proposals
        }
        try:
            for proposal, patch in prepared:
                blob_path = self._replacement_blob_path(proposal.replacement_sha256)
                blob_existed = blob_path.exists() or blob_path.is_symlink()
                persisted_digest = self._persist_replacement_blob(patch.replacement)
                if persisted_digest != proposal.replacement_sha256:
                    raise ProposalOnlyError(
                        "replacement blob digest changed during commit"
                    )
                if not blob_existed:
                    created_blobs.append(blob_path)
                proposal_path = (
                    self.state_directory / f"proposal-{proposal.proposal_id}.json"
                )
                self._persist_json(proposal_path, asdict(proposal))
                created_proposals.append(proposal_path)
        except BaseException:
            rollback_failed = False
            for path in reversed(created_proposals):
                try:
                    path.unlink(missing_ok=True)
                except OSError:
                    rollback_failed = True
            for path in reversed(created_blobs):
                digest = path.name[
                    len(REPLACEMENT_BLOB_PREFIX) : -len(REPLACEMENT_BLOB_SUFFIX)
                ]
                if digest in prior_blob_digests:
                    continue
                try:
                    path.unlink(missing_ok=True)
                except OSError:
                    rollback_failed = True
            if rollback_failed:
                raise ProposalOnlyError(
                    "proposal batch commit failed and rollback was incomplete"
                )
            raise
        self._proposals.extend(accepted)
        self._reviewable_patches.update(accepted_patches)
        return tuple(accepted)

    def archive_negative(
        self,
        proposal_id: str,
        *,
        reason_code: str,
        evidence_sha256: str,
    ) -> NegativeProposalV1:
        proposal = next(
            (item for item in self._proposals if item.proposal_id == proposal_id),
            None,
        )
        if proposal is None:
            raise ProposalOnlyError("negative archive proposal is unknown")
        if len(self._negatives) >= self.max_negatives:
            raise ProposalOnlyError("negative evidence retention bound was reached")
        reason = _text(reason_code, "negative reason", maximum=256)
        evidence = _digest(evidence_sha256, "negative evidence")
        proposal_digest = _json_digest(asdict(proposal))
        negative = NegativeProposalV1(
            NEGATIVE_SCHEMA,
            proposal_id,
            proposal_digest,
            reason,
            evidence,
            time_ns(),
        )
        self._persist_json(
            self.state_directory / f"negative-{proposal_id}.json",
            asdict(negative),
        )
        self._negatives.append(negative)
        self._reviewable_patches.pop(proposal_id, None)
        self._unreviewable_proposals.pop(proposal_id, None)
        return negative

    def _persist_json(self, path: Path, payload: Mapping[str, object]) -> None:
        if path.parent != self.state_directory:
            raise ProposalOnlyError("proposal evidence escaped its state directory")
        directory_stat = os.lstat(self.state_directory)
        attributes = getattr(directory_stat, "st_file_attributes", 0)
        if (
            self.state_directory.is_symlink()
            or attributes & FILE_ATTRIBUTE_REPARSE_POINT
            or not self.state_directory.is_dir()
        ):
            raise ProposalOnlyError("proposal state directory integrity changed")
        if path.exists() or path.is_symlink():
            raise ProposalOnlyError("proposal evidence path already exists")
        encoded = json.dumps(
            payload,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("ascii")
        if len(encoded) > MAX_RECORD_BYTES:
            raise ProposalOnlyError("proposal evidence crossed its JSON size bound")
        temporary = path.with_suffix(path.suffix + ".tmp")
        if temporary.exists() or temporary.is_symlink():
            raise ProposalOnlyError("proposal temporary evidence path already exists")
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        no_follow = getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(temporary, flags | no_follow, 0o600)
        try:
            with os.fdopen(descriptor, "wb", closefd=False) as stream:
                stream.write(encoded)
                stream.flush()
                os.fsync(stream.fileno())
        finally:
            os.close(descriptor)
        os.replace(temporary, path)


__all__ = [
    "CandidateDraft",
    "CaptureCoverage",
    "FailureEventV1",
    "NegativeProposalV1",
    "PatchProposalV1",
    "ProposalOnlyError",
    "ProposalOnlySelfHarness",
    "SuccessInvariantV1",
]
