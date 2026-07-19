"""Content-addressed runtime evidence and last-known-good FactBook storage.

This module deliberately separates a public capability *claim* from the raw
evidence that can authorize it.  A verified claim is usable only while every
cited record is present, content-addressed, bound to the claim, and measured
under the exact current model/code/config/device scope.

The journal and FactBook snapshots must live outside the source tree.  Source
packages therefore never become a mutable evidence database, and a successful
test cannot silently rewrite the code it is supposed to attest.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import stat
from threading import RLock
from typing import Mapping, Sequence
from uuid import uuid4

from .capabilities import (
    CapabilityRecord,
    CapabilityRegistry,
    CapabilityState,
    EvidenceClass,
)
from .factbook import ModelArtifactFacts, RuntimeFactBook, TensorInventory


_DIGEST_RE = re.compile(r"^[0-9a-f]{64}$")
_IDENTIFIER_RE = re.compile(r"^[a-z][a-z0-9_.:-]{0,127}$")
_EVIDENCE_ID_RE = re.compile(r"^ev1-[0-9a-f]{64}$")
_SNAPSHOT_ID_RE = re.compile(r"^fb1-[0-9a-f]{64}$")
_MAX_JOURNAL_BYTES = 256 * 1024 * 1024
_MAX_JOURNAL_RECORDS = 1_000_000
_MAX_SNAPSHOT_BYTES = 16 * 1024 * 1024
MODEL_ARTIFACT_CLAIM_ID = "model.artifact"


class EvidenceError(RuntimeError):
    """Raised when evidence is missing, stale, malformed, or unsafe."""


class ClaimAssessment(str, Enum):
    """Fail-closed result of evaluating one claim against the current scope."""

    VERIFIED = "verified"
    NON_VERIFIED = "non_verified"
    MISSING_EVIDENCE = "missing_evidence"
    INVALID_EVIDENCE = "invalid_evidence"
    STALE = "stale"


def _canonical_bytes(payload: object) -> bytes:
    try:
        encoded = json.dumps(
            payload,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError) as exc:
        raise EvidenceError("evidence payload is not canonical JSON") from exc
    return encoded.encode("utf-8")


def _digest(value: object, label: str) -> str:
    if not isinstance(value, str) or _DIGEST_RE.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _identifier(value: object, label: str) -> str:
    if not isinstance(value, str) or _IDENTIFIER_RE.fullmatch(value) is None:
        raise ValueError(f"{label} must be bounded lowercase ASCII")
    return value


def _timestamp(value: object, label: str) -> str:
    if not isinstance(value, str) or not 1 <= len(value) <= 64:
        raise ValueError(f"{label} must be a bounded ISO-8601 timestamp")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"{label} must be ISO-8601") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"{label} must include a timezone")
    return value


def sha256_json(payload: object) -> str:
    """Return the deterministic SHA-256 of a strict JSON value."""

    return sha256(_canonical_bytes(payload)).hexdigest()


def sha256_text(value: str) -> str:
    """Hash an exact UTF-8 descriptor, suitable for a device fingerprint."""

    if not isinstance(value, str) or not value:
        raise ValueError("digest input must be a non-empty string")
    return sha256(value.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class EvidenceScopeV1:
    """Exact artifact boundary under which an evidence record is valid."""

    model_sha256: str
    code_sha256: str
    config_sha256: str
    device_sha256: str

    def __post_init__(self) -> None:
        _digest(self.model_sha256, "model_sha256")
        _digest(self.code_sha256, "code_sha256")
        _digest(self.config_sha256, "config_sha256")
        _digest(self.device_sha256, "device_sha256")

    @property
    def digest(self) -> str:
        return sha256_json(self.as_payload())

    def as_payload(self) -> dict[str, str]:
        return {
            "model_sha256": self.model_sha256,
            "code_sha256": self.code_sha256,
            "config_sha256": self.config_sha256,
            "device_sha256": self.device_sha256,
        }

    @classmethod
    def from_payload(cls, value: object) -> EvidenceScopeV1:
        if not isinstance(value, dict) or set(value) != {
            "model_sha256",
            "code_sha256",
            "config_sha256",
            "device_sha256",
        }:
            raise EvidenceError("evidence scope schema is invalid")
        try:
            return cls(**value)
        except (TypeError, ValueError) as exc:
            raise EvidenceError("evidence scope is invalid") from exc

    @classmethod
    def for_factbook(
        cls,
        factbook: RuntimeFactBook,
        *,
        code_sha256: str,
        runtime_config_sha256: str,
        device_descriptor: str,
    ) -> EvidenceScopeV1:
        """Bind a FactBook to exact code, runtime config, and physical device."""

        if not isinstance(factbook, RuntimeFactBook):
            raise TypeError("factbook must be a RuntimeFactBook")
        return cls(
            model_sha256=factbook.model.manifest_sha256,
            code_sha256=code_sha256,
            config_sha256=runtime_config_sha256,
            device_sha256=sha256_text(device_descriptor),
        )


@dataclass(frozen=True, slots=True)
class EvidenceRecordV1:
    """One immutable, content-addressed raw evidence record."""

    evidence_id: str
    recorded_at: str
    kind: str
    producer: str
    run_id: str
    evidence_class: EvidenceClass
    claim_ids: tuple[str, ...]
    artifact_sha256: str
    payload_sha256: str
    scope: EvidenceScopeV1
    metadata: tuple[tuple[str, str], ...] = ()
    schema_version: int = 1

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError("EvidenceRecordV1 requires schema_version=1")
        _timestamp(self.recorded_at, "recorded_at")
        _identifier(self.kind, "kind")
        _identifier(self.producer, "producer")
        _identifier(self.run_id, "run_id")
        if not isinstance(self.evidence_class, EvidenceClass):
            raise TypeError("evidence_class must be EvidenceClass")
        if not isinstance(self.claim_ids, tuple) or not self.claim_ids:
            raise ValueError("evidence must bind at least one claim id")
        for claim_id in self.claim_ids:
            _identifier(claim_id, "claim_id")
        if tuple(sorted(set(self.claim_ids))) != self.claim_ids:
            raise ValueError("claim_ids must be unique and sorted")
        _digest(self.artifact_sha256, "artifact_sha256")
        _digest(self.payload_sha256, "payload_sha256")
        if not isinstance(self.scope, EvidenceScopeV1):
            raise TypeError("scope must be EvidenceScopeV1")
        if not isinstance(self.metadata, tuple) or len(self.metadata) > 64:
            raise ValueError("metadata must be a bounded tuple")
        previous = ""
        for key, value in self.metadata:
            _identifier(key, "metadata key")
            if key <= previous:
                raise ValueError("metadata keys must be unique and sorted")
            if not isinstance(value, str) or not 1 <= len(value) <= 1_024:
                raise ValueError("metadata values must contain 1-1024 characters")
            previous = key
        expected = "ev1-" + sha256_json(self._identity_payload())
        if self.evidence_id != expected:
            raise ValueError("evidence_id does not match record content")

    def _identity_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "recorded_at": self.recorded_at,
            "kind": self.kind,
            "producer": self.producer,
            "run_id": self.run_id,
            "evidence_class": self.evidence_class.value,
            "claim_ids": list(self.claim_ids),
            "artifact_sha256": self.artifact_sha256,
            "payload_sha256": self.payload_sha256,
            "scope": self.scope.as_payload(),
            "metadata": dict(self.metadata),
        }

    def as_payload(self) -> dict[str, object]:
        return {
            "record_type": "evidence",
            "evidence_id": self.evidence_id,
            **self._identity_payload(),
        }

    def is_stale(self, current_scope: EvidenceScopeV1) -> bool:
        if not isinstance(current_scope, EvidenceScopeV1):
            raise TypeError("current_scope must be EvidenceScopeV1")
        return self.scope != current_scope

    @classmethod
    def create(
        cls,
        *,
        recorded_at: str,
        kind: str,
        producer: str,
        run_id: str,
        evidence_class: EvidenceClass,
        claim_ids: Sequence[str],
        artifact_sha256: str,
        payload_sha256: str,
        scope: EvidenceScopeV1,
        metadata: Mapping[str, str] | None = None,
    ) -> EvidenceRecordV1:
        if not isinstance(evidence_class, EvidenceClass):
            raise TypeError("evidence_class must be EvidenceClass")
        ordered_claims = tuple(sorted(claim_ids))
        ordered_metadata = tuple(sorted((metadata or {}).items()))
        identity = {
            "schema_version": 1,
            "recorded_at": recorded_at,
            "kind": kind,
            "producer": producer,
            "run_id": run_id,
            "evidence_class": evidence_class.value,
            "claim_ids": list(ordered_claims),
            "artifact_sha256": artifact_sha256,
            "payload_sha256": payload_sha256,
            "scope": scope.as_payload(),
            "metadata": dict(ordered_metadata),
        }
        return cls(
            evidence_id="ev1-" + sha256_json(identity),
            recorded_at=recorded_at,
            kind=kind,
            producer=producer,
            run_id=run_id,
            evidence_class=evidence_class,
            claim_ids=ordered_claims,
            artifact_sha256=artifact_sha256,
            payload_sha256=payload_sha256,
            scope=scope,
            metadata=ordered_metadata,
        )

    @classmethod
    def from_payload(cls, value: object) -> EvidenceRecordV1:
        if not isinstance(value, dict):
            raise EvidenceError("evidence record must be an object")
        expected = {
            "record_type",
            "schema_version",
            "evidence_id",
            "recorded_at",
            "kind",
            "producer",
            "run_id",
            "evidence_class",
            "claim_ids",
            "artifact_sha256",
            "payload_sha256",
            "scope",
            "metadata",
        }
        if set(value) != expected or value.get("record_type") != "evidence":
            raise EvidenceError("evidence record schema is invalid")
        metadata = value.get("metadata")
        claim_ids = value.get("claim_ids")
        if not isinstance(metadata, dict) or not isinstance(claim_ids, list):
            raise EvidenceError("evidence record collections are invalid")
        try:
            return cls(
                evidence_id=value["evidence_id"],
                recorded_at=value["recorded_at"],
                kind=value["kind"],
                producer=value["producer"],
                run_id=value["run_id"],
                evidence_class=EvidenceClass(value["evidence_class"]),
                claim_ids=tuple(claim_ids),
                artifact_sha256=value["artifact_sha256"],
                payload_sha256=value["payload_sha256"],
                scope=EvidenceScopeV1.from_payload(value["scope"]),
                metadata=tuple(metadata.items()),
                schema_version=value["schema_version"],
            )
        except (TypeError, ValueError) as exc:
            raise EvidenceError("evidence record validation failed") from exc


@dataclass(frozen=True, slots=True)
class ClaimRecord:
    """Public claim whose verified state always cites concrete evidence IDs."""

    claim_id: str
    statement: str
    evidence_class: EvidenceClass
    evidence_ids: tuple[str, ...] = ()
    schema_version: int = 1

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError("ClaimRecord requires schema_version=1")
        _identifier(self.claim_id, "claim_id")
        if not isinstance(self.statement, str) or not 1 <= len(self.statement) <= 2_048:
            raise ValueError("claim statement must contain 1-2048 characters")
        if not isinstance(self.evidence_class, EvidenceClass):
            raise TypeError("evidence_class must be EvidenceClass")
        if not isinstance(self.evidence_ids, tuple):
            raise TypeError("evidence_ids must be a tuple")
        if tuple(sorted(set(self.evidence_ids))) != self.evidence_ids:
            raise ValueError("evidence_ids must be unique and sorted")
        for evidence_id in self.evidence_ids:
            if (
                not isinstance(evidence_id, str)
                or _EVIDENCE_ID_RE.fullmatch(evidence_id) is None
            ):
                raise ValueError("claim evidence_id is invalid")
        if self.evidence_class is EvidenceClass.VERIFIED and not self.evidence_ids:
            raise ValueError("verified claim requires at least one evidence_id")

    def as_payload(self) -> dict[str, object]:
        return {
            "record_type": "claim",
            "schema_version": self.schema_version,
            "claim_id": self.claim_id,
            "statement": self.statement,
            "evidence_class": self.evidence_class.value,
            "evidence_ids": list(self.evidence_ids),
        }

    def assess(
        self,
        evidence: Mapping[str, EvidenceRecordV1],
        current_scope: EvidenceScopeV1,
    ) -> ClaimAssessment:
        if not isinstance(current_scope, EvidenceScopeV1):
            raise TypeError("current_scope must be EvidenceScopeV1")
        if self.evidence_class is not EvidenceClass.VERIFIED:
            return ClaimAssessment.NON_VERIFIED
        for evidence_id in self.evidence_ids:
            record = evidence.get(evidence_id)
            if record is None:
                return ClaimAssessment.MISSING_EVIDENCE
            if not isinstance(record, EvidenceRecordV1):
                return ClaimAssessment.INVALID_EVIDENCE
            if self.claim_id not in record.claim_ids:
                return ClaimAssessment.INVALID_EVIDENCE
            if record.evidence_class is not EvidenceClass.VERIFIED:
                return ClaimAssessment.INVALID_EVIDENCE
            if record.is_stale(current_scope):
                return ClaimAssessment.STALE
        return ClaimAssessment.VERIFIED

    @classmethod
    def from_payload(cls, value: object) -> ClaimRecord:
        if not isinstance(value, dict) or set(value) != {
            "record_type",
            "schema_version",
            "claim_id",
            "statement",
            "evidence_class",
            "evidence_ids",
        }:
            raise EvidenceError("claim record schema is invalid")
        if value.get("record_type") != "claim" or not isinstance(
            value.get("evidence_ids"), list
        ):
            raise EvidenceError("claim record collections are invalid")
        try:
            return cls(
                claim_id=value["claim_id"],
                statement=value["statement"],
                evidence_class=EvidenceClass(value["evidence_class"]),
                evidence_ids=tuple(value["evidence_ids"]),
                schema_version=value["schema_version"],
            )
        except (TypeError, ValueError) as exc:
            raise EvidenceError("claim record validation failed") from exc


def assess_claims(
    claims: Sequence[ClaimRecord],
    records: Sequence[EvidenceRecordV1],
    current_scope: EvidenceScopeV1,
) -> dict[str, ClaimAssessment]:
    """Evaluate a bounded set of claims without silently accepting duplicates."""

    if len(claims) > 10_000 or len(records) > 100_000:
        raise EvidenceError("claim assessment bound exceeded")
    by_evidence: dict[str, EvidenceRecordV1] = {}
    for record in records:
        if record.evidence_id in by_evidence:
            raise EvidenceError("duplicate evidence_id")
        by_evidence[record.evidence_id] = record
    result: dict[str, ClaimAssessment] = {}
    for claim in claims:
        if claim.claim_id in result:
            raise EvidenceError("duplicate claim_id")
        result[claim.claim_id] = claim.assess(by_evidence, current_scope)
    return result


def capability_claim_id(name: str) -> str:
    """Return the stable claim ID used to bind a capability disclosure."""

    _identifier(name, "capability name")
    return f"capability.{name}"


def build_capability_claims(
    registry: CapabilityRegistry,
    evidence_ids: Mapping[str, Sequence[str]],
) -> tuple[ClaimRecord, ...]:
    """Convert the existing capability registry into evidence-bearing claims.

    This is intentionally strict: every registry entry labelled ``verified``
    must receive at least one content-addressed evidence ID.  Other evidence
    classes may remain uncited because they do not assert verification.
    """

    if not isinstance(registry, CapabilityRegistry):
        raise TypeError("registry must be CapabilityRegistry")
    unknown = set(evidence_ids) - {record.name for record in registry.records}
    if unknown:
        raise EvidenceError(
            f"evidence supplied for unknown capabilities: {sorted(unknown)}"
        )
    claims = []
    for record in registry.records:
        claims.append(
            ClaimRecord(
                claim_id=capability_claim_id(record.name),
                statement=record.detail,
                evidence_class=record.evidence,
                evidence_ids=tuple(sorted(evidence_ids.get(record.name, ()))),
            )
        )
    return tuple(claims)


def build_factbook_claims(
    factbook: RuntimeFactBook,
    *,
    model_evidence_ids: Sequence[str],
    capability_evidence_ids: Mapping[str, Sequence[str]],
) -> tuple[ClaimRecord, ...]:
    """Build complete model + capability claims for an LKG FactBook snapshot."""

    if not isinstance(factbook, RuntimeFactBook):
        raise TypeError("factbook must be RuntimeFactBook")
    model_claim = ClaimRecord(
        claim_id=MODEL_ARTIFACT_CLAIM_ID,
        statement=(
            f"Local model artifact {factbook.model.label} matches the recorded "
            "manifest and bounded configuration facts."
        ),
        evidence_class=EvidenceClass.VERIFIED,
        evidence_ids=tuple(sorted(model_evidence_ids)),
    )
    return (model_claim,) + build_capability_claims(
        factbook.capabilities, capability_evidence_ids
    )


def _is_reparse(path: Path) -> bool:
    try:
        info = path.lstat()
    except FileNotFoundError:
        return False
    attributes = int(getattr(info, "st_file_attributes", 0))
    return path.is_symlink() or bool(attributes & 0x400)


def _require_regular_file(path: Path, label: str) -> None:
    if _is_reparse(path):
        raise EvidenceError(f"{label} cannot be a symlink or reparse point")
    try:
        mode = path.stat().st_mode
    except FileNotFoundError as exc:
        raise EvidenceError(f"{label} is missing") from exc
    if not stat.S_ISREG(mode):
        raise EvidenceError(f"{label} must be a regular file")


class _ExternalEvidenceRoot:
    def __init__(self, root: str | Path, *, source_root: str | Path) -> None:
        source = Path(source_root).expanduser().resolve(strict=True)
        if not source.is_dir():
            raise EvidenceError("source_root must be a directory")
        requested = Path(root).expanduser().absolute()
        prospective = requested.resolve(strict=False)
        if prospective.is_relative_to(source) or source.is_relative_to(prospective):
            raise EvidenceError("evidence storage must be outside the source tree")
        if requested.exists() and _is_reparse(requested):
            raise EvidenceError("evidence root cannot be a symlink or reparse point")
        requested.mkdir(parents=True, exist_ok=True)
        resolved = requested.resolve(strict=True)
        if resolved.is_relative_to(source) or source.is_relative_to(resolved):
            raise EvidenceError("evidence storage must be outside the source tree")
        if not resolved.is_dir() or _is_reparse(resolved):
            raise EvidenceError("evidence root must be a regular directory")
        self.root = resolved
        self.source_root = source


class AppendOnlyEvidenceJournal(_ExternalEvidenceRoot):
    """Single-writer JSONL evidence journal with a SHA-256 hash chain.

    Each append is one ``O_APPEND`` write followed by ``fsync``.  The complete
    existing chain is revalidated before every append; a truncation, rewrite,
    duplicate ID, malformed line, or symlink substitution fails closed.
    """

    def __init__(
        self,
        root: str | Path,
        *,
        source_root: str | Path,
        filename: str = "evidence.v1.jsonl",
    ) -> None:
        super().__init__(root, source_root=source_root)
        if filename != "evidence.v1.jsonl":
            raise EvidenceError("the evidence journal filename is fixed by schema")
        self.path = self.root / filename
        self._lock = RLock()
        self._trusted_head: tuple[int, str] | None = None
        if self.path.exists():
            self.read_records()

    def _read_entries(self) -> tuple[list[EvidenceRecordV1], tuple[int, str]]:
        if not self.path.exists():
            return [], (0, "0" * 64)
        _require_regular_file(self.path, "evidence journal")
        size = self.path.stat().st_size
        if size > _MAX_JOURNAL_BYTES:
            raise EvidenceError("evidence journal exceeds its byte bound")
        raw = self.path.read_bytes()
        if raw and not raw.endswith(b"\n"):
            raise EvidenceError("evidence journal has a torn final record")
        lines = raw.splitlines(keepends=True)
        if len(lines) > _MAX_JOURNAL_RECORDS:
            raise EvidenceError("evidence journal exceeds its record bound")
        previous = "0" * 64
        seen: set[str] = set()
        records: list[EvidenceRecordV1] = []
        for expected_sequence, encoded in enumerate(lines, start=1):
            try:
                value = json.loads(encoded.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise EvidenceError("evidence journal contains invalid JSONL") from exc
            if not isinstance(value, dict) or set(value) != {
                "sequence",
                "previous_sha256",
                "record",
                "entry_sha256",
            }:
                raise EvidenceError("evidence journal entry schema is invalid")
            body = {
                "sequence": expected_sequence,
                "previous_sha256": previous,
                "record": value["record"],
            }
            entry_digest = sha256_json(body)
            if (
                value["sequence"] != expected_sequence
                or value["previous_sha256"] != previous
                or value["entry_sha256"] != entry_digest
                or encoded
                != _canonical_bytes({**body, "entry_sha256": entry_digest}) + b"\n"
            ):
                raise EvidenceError("evidence journal hash chain is invalid")
            record = EvidenceRecordV1.from_payload(value["record"])
            if record.evidence_id in seen:
                raise EvidenceError("evidence journal contains a duplicate evidence_id")
            seen.add(record.evidence_id)
            records.append(record)
            previous = entry_digest
        return records, (len(lines), previous)

    def read_records(self) -> tuple[EvidenceRecordV1, ...]:
        with self._lock:
            records, head = self._read_entries()
            if self._trusted_head is not None and head != self._trusted_head:
                raise EvidenceError("evidence journal changed outside this writer")
            self._trusted_head = head
            return tuple(records)

    def append(self, record: EvidenceRecordV1) -> int:
        if not isinstance(record, EvidenceRecordV1):
            raise TypeError("record must be EvidenceRecordV1")
        with self._lock:
            records, head = self._read_entries()
            if self._trusted_head is not None and head != self._trusted_head:
                raise EvidenceError("evidence journal changed outside this writer")
            if any(item.evidence_id == record.evidence_id for item in records):
                raise EvidenceError("evidence_id is already present")
            sequence = head[0] + 1
            body = {
                "sequence": sequence,
                "previous_sha256": head[1],
                "record": record.as_payload(),
            }
            entry_digest = sha256_json(body)
            encoded = _canonical_bytes({**body, "entry_sha256": entry_digest}) + b"\n"
            if self.path.exists():
                _require_regular_file(self.path, "evidence journal")
            flags = os.O_APPEND | os.O_CREAT | os.O_WRONLY
            if hasattr(os, "O_BINARY"):
                flags |= os.O_BINARY
            descriptor = os.open(self.path, flags, 0o600)
            try:
                written = os.write(descriptor, encoded)
                if written != len(encoded):
                    raise EvidenceError("evidence journal append was incomplete")
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
            self._trusted_head = (sequence, entry_digest)
            return sequence


@dataclass(frozen=True, slots=True)
class FactBookSnapshotV1:
    sequence: int
    snapshot_id: str
    factbook: RuntimeFactBook
    scope: EvidenceScopeV1
    claims: tuple[ClaimRecord, ...]
    evidence: tuple[EvidenceRecordV1, ...]


def _factbook_from_payload(value: object) -> RuntimeFactBook:
    if not isinstance(value, dict) or set(value) != {
        "schema_version",
        "generated_at",
        "build_version",
        "device",
        "target_device",
        "model",
        "capabilities",
    }:
        raise EvidenceError("FactBook snapshot schema is invalid")
    model = value["model"]
    if not isinstance(model, dict) or set(model) != {
        "label",
        "architecture",
        "hidden_size",
        "layers",
        "dense",
        "stored_parameters",
        "effective_parameters",
        "embedding_parameters",
        "tensor_count",
        "dtype_parameters",
        "manifest_sha256",
        "config_sha256",
    }:
        raise EvidenceError("FactBook model snapshot schema is invalid")
    dtypes = model["dtype_parameters"]
    capabilities = value["capabilities"]
    if not isinstance(dtypes, dict) or not isinstance(capabilities, list):
        raise EvidenceError("FactBook snapshot collections are invalid")
    try:
        inventory = TensorInventory(
            tensor_count=model["tensor_count"],
            stored_parameters=model["stored_parameters"],
            effective_parameters=model["effective_parameters"],
            embedding_parameters=model["embedding_parameters"],
            dtype_parameters=tuple(sorted(dtypes.items())),
        )
        model_facts = ModelArtifactFacts(
            label=model["label"],
            architecture=model["architecture"],
            hidden_size=model["hidden_size"],
            layers=model["layers"],
            dense=model["dense"],
            inventory=inventory,
            manifest_sha256=model["manifest_sha256"],
            config_sha256=model["config_sha256"],
        )
        records = []
        for item in capabilities:
            if not isinstance(item, dict) or set(item) != {
                "name",
                "state",
                "evidence",
                "answer_bearing",
                "runtime_mutation_allowed",
                "detail",
            }:
                raise EvidenceError("FactBook capability snapshot schema is invalid")
            records.append(
                CapabilityRecord(
                    name=item["name"],
                    state=CapabilityState(item["state"]),
                    evidence=EvidenceClass(item["evidence"]),
                    answer_bearing=item["answer_bearing"],
                    runtime_mutation_allowed=item["runtime_mutation_allowed"],
                    detail=item["detail"],
                )
            )
        return RuntimeFactBook(
            schema_version=value["schema_version"],
            generated_at=value["generated_at"],
            build_version=value["build_version"],
            device=value["device"],
            target_device=value["target_device"],
            model=model_facts,
            capabilities=CapabilityRegistry(tuple(records)),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise EvidenceError("FactBook snapshot validation failed") from exc


class FactBookSnapshotStore(_ExternalEvidenceRoot):
    """Immutable FactBook snapshots selected through one atomic LKG pointer."""

    def __init__(self, root: str | Path, *, source_root: str | Path) -> None:
        super().__init__(root, source_root=source_root)
        self.journal = AppendOnlyEvidenceJournal(
            self.root,
            source_root=self.source_root,
        )
        self.directory = self.root / "factbook-v1"
        self.snapshots = self.directory / "snapshots"
        self.directory.mkdir(exist_ok=True)
        self.snapshots.mkdir(exist_ok=True)
        if _is_reparse(self.directory) or _is_reparse(self.snapshots):
            raise EvidenceError("FactBook store cannot traverse a reparse point")
        self.pointer = self.directory / "CURRENT.json"
        self._lock = RLock()

    def record_evidence(self, record: EvidenceRecordV1) -> int:
        """Persist raw evidence before any snapshot is allowed to cite it."""

        return self.journal.append(record)

    def _snapshot_files(self) -> tuple[Path, ...]:
        files = []
        for path in self.snapshots.glob("*.json"):
            _require_regular_file(path, "FactBook snapshot")
            files.append(path)
        return tuple(sorted(files, reverse=True))

    def _next_sequence(self) -> int:
        highest = 0
        for path in self._snapshot_files():
            prefix = path.name.split("-", 1)[0]
            if prefix.isdigit():
                highest = max(highest, int(prefix))
        return highest + 1

    def _atomic_pointer(self, payload: dict[str, object]) -> None:
        encoded = _canonical_bytes(payload) + b"\n"
        temporary = self.directory / f".CURRENT.{uuid4().hex}.tmp"
        try:
            with temporary.open("xb") as stream:
                stream.write(encoded)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, self.pointer)
        finally:
            temporary.unlink(missing_ok=True)

    def publish(
        self,
        factbook: RuntimeFactBook,
        *,
        scope: EvidenceScopeV1,
        claims: Sequence[ClaimRecord],
        evidence: Sequence[EvidenceRecordV1],
    ) -> FactBookSnapshotV1:
        """Publish only if every ``verified`` claim is current and evidenced."""

        if not isinstance(factbook, RuntimeFactBook):
            raise TypeError("factbook must be RuntimeFactBook")
        if not isinstance(scope, EvidenceScopeV1):
            raise TypeError("scope must be EvidenceScopeV1")
        if scope.model_sha256 != factbook.model.manifest_sha256:
            raise EvidenceError(
                "FactBook model manifest digest does not match the evidence scope"
            )
        journal_records = {
            record.evidence_id: record for record in self.journal.read_records()
        }
        missing_raw = sorted(
            record.evidence_id
            for record in evidence
            if journal_records.get(record.evidence_id) != record
        )
        if missing_raw:
            raise EvidenceError(
                "FactBook evidence is not present in the append-only raw journal: "
                f"{missing_raw}"
            )
        by_claim_id = {claim.claim_id: claim for claim in claims}
        if len(by_claim_id) != len(claims):
            raise EvidenceError("FactBook claim IDs must be unique")
        required_factbook_claims = {
            MODEL_ARTIFACT_CLAIM_ID,
            *(
                capability_claim_id(record.name)
                for record in factbook.capabilities.records
                if record.evidence is EvidenceClass.VERIFIED
            ),
        }
        missing_factbook_claims = sorted(required_factbook_claims - set(by_claim_id))
        if missing_factbook_claims:
            raise EvidenceError(
                "FactBook verified facts lack evidence-bearing claims: "
                f"{missing_factbook_claims}"
            )
        for claim_id in required_factbook_claims:
            if by_claim_id[claim_id].evidence_class is not EvidenceClass.VERIFIED:
                raise EvidenceError(
                    f"FactBook verified capability claim was downgraded: {claim_id}"
                )
        assessments = assess_claims(claims, evidence, scope)
        failed = {
            claim_id: state.value
            for claim_id, state in assessments.items()
            if state not in {ClaimAssessment.VERIFIED, ClaimAssessment.NON_VERIFIED}
        }
        if failed:
            raise EvidenceError(f"FactBook has unpublishable claims: {failed}")
        ordered_claims = tuple(sorted(claims, key=lambda item: item.claim_id))
        ordered_evidence = tuple(sorted(evidence, key=lambda item: item.evidence_id))
        with self._lock:
            sequence = self._next_sequence()
            identity = {
                "schema_version": 1,
                "sequence": sequence,
                "scope": scope.as_payload(),
                "factbook": factbook.as_payload(),
                "claims": [item.as_payload() for item in ordered_claims],
                "evidence": [item.as_payload() for item in ordered_evidence],
            }
            snapshot_id = "fb1-" + sha256_json(identity)
            payload = {**identity, "snapshot_id": snapshot_id}
            encoded = _canonical_bytes(payload) + b"\n"
            file_sha256 = sha256(encoded).hexdigest()
            filename = f"{sequence:020d}-{snapshot_id}.json"
            target = self.snapshots / filename
            with target.open("xb") as stream:
                stream.write(encoded)
                stream.flush()
                os.fsync(stream.fileno())
            self._atomic_pointer(
                {
                    "schema_version": 1,
                    "sequence": sequence,
                    "snapshot_id": snapshot_id,
                    "filename": filename,
                    "file_sha256": file_sha256,
                }
            )
            return FactBookSnapshotV1(
                sequence=sequence,
                snapshot_id=snapshot_id,
                factbook=factbook,
                scope=scope,
                claims=ordered_claims,
                evidence=ordered_evidence,
            )

    def _load_snapshot(
        self,
        path: Path,
        *,
        current_scope: EvidenceScopeV1,
        expected_file_sha256: str | None = None,
        expected_snapshot_id: str | None = None,
        expected_sequence: int | None = None,
    ) -> FactBookSnapshotV1:
        _require_regular_file(path, "FactBook snapshot")
        raw = path.read_bytes()
        if not raw or len(raw) > _MAX_SNAPSHOT_BYTES or not raw.endswith(b"\n"):
            raise EvidenceError("FactBook snapshot is empty, torn, or oversized")
        file_digest = sha256(raw).hexdigest()
        if expected_file_sha256 is not None and file_digest != expected_file_sha256:
            raise EvidenceError("FactBook snapshot file digest changed")
        try:
            value = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise EvidenceError("FactBook snapshot JSON is invalid") from exc
        if not isinstance(value, dict) or set(value) != {
            "schema_version",
            "sequence",
            "snapshot_id",
            "scope",
            "factbook",
            "claims",
            "evidence",
        }:
            raise EvidenceError("FactBook snapshot schema is invalid")
        if (
            value["schema_version"] != 1
            or not isinstance(value["sequence"], int)
            or isinstance(value["sequence"], bool)
        ):
            raise EvidenceError("FactBook snapshot version or sequence is invalid")
        if value["sequence"] < 1 or (
            expected_sequence is not None and value["sequence"] != expected_sequence
        ):
            raise EvidenceError("FactBook snapshot sequence is invalid")
        if not isinstance(value["claims"], list) or not isinstance(
            value["evidence"], list
        ):
            raise EvidenceError("FactBook snapshot collections are invalid")
        identity = dict(value)
        snapshot_id = identity.pop("snapshot_id")
        if (
            not isinstance(snapshot_id, str)
            or _SNAPSHOT_ID_RE.fullmatch(snapshot_id) is None
            or snapshot_id != "fb1-" + sha256_json(identity)
            or (
                expected_snapshot_id is not None and snapshot_id != expected_snapshot_id
            )
        ):
            raise EvidenceError("FactBook snapshot identity is invalid")
        expected_filename = f"{value['sequence']:020d}-{snapshot_id}.json"
        if path.name != expected_filename:
            raise EvidenceError("FactBook snapshot filename is not content-addressed")
        scope = EvidenceScopeV1.from_payload(value["scope"])
        if scope != current_scope:
            raise EvidenceError("FactBook snapshot is stale for the current scope")
        factbook = _factbook_from_payload(value["factbook"])
        if scope.model_sha256 != factbook.model.manifest_sha256:
            raise EvidenceError(
                "FactBook model manifest digest does not match the evidence scope"
            )
        claims = tuple(ClaimRecord.from_payload(item) for item in value["claims"])
        evidence = tuple(
            EvidenceRecordV1.from_payload(item) for item in value["evidence"]
        )
        journal_records = {
            record.evidence_id: record for record in self.journal.read_records()
        }
        if any(
            journal_records.get(record.evidence_id) != record for record in evidence
        ):
            raise EvidenceError(
                "FactBook snapshot cites evidence absent from raw journal"
            )
        assessments = assess_claims(claims, evidence, current_scope)
        if any(
            state not in {ClaimAssessment.VERIFIED, ClaimAssessment.NON_VERIFIED}
            for state in assessments.values()
        ):
            raise EvidenceError("FactBook snapshot contains stale or missing evidence")
        return FactBookSnapshotV1(
            sequence=value["sequence"],
            snapshot_id=snapshot_id,
            factbook=factbook,
            scope=scope,
            claims=claims,
            evidence=evidence,
        )

    def load_current(self, current_scope: EvidenceScopeV1) -> FactBookSnapshotV1:
        """Load the atomic pointer target without silently falling back."""

        with self._lock:
            _require_regular_file(self.pointer, "FactBook CURRENT pointer")
            try:
                pointer = json.loads(self.pointer.read_text(encoding="utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise EvidenceError("FactBook CURRENT pointer is invalid") from exc
            if not isinstance(pointer, dict) or set(pointer) != {
                "schema_version",
                "sequence",
                "snapshot_id",
                "filename",
                "file_sha256",
            }:
                raise EvidenceError("FactBook CURRENT pointer schema is invalid")
            filename = pointer["filename"]
            if (
                pointer["schema_version"] != 1
                or not isinstance(pointer["sequence"], int)
                or isinstance(pointer["sequence"], bool)
                or pointer["sequence"] < 1
                or not isinstance(filename, str)
                or Path(filename).name != filename
            ):
                raise EvidenceError("FactBook CURRENT pointer values are invalid")
            _digest(pointer["file_sha256"], "file_sha256")
            return self._load_snapshot(
                self.snapshots / filename,
                current_scope=current_scope,
                expected_file_sha256=pointer["file_sha256"],
                expected_snapshot_id=pointer["snapshot_id"],
                expected_sequence=pointer["sequence"],
            )

    def recover_last_known_good(
        self, current_scope: EvidenceScopeV1
    ) -> FactBookSnapshotV1:
        """Recover the newest valid same-scope snapshot and repair ``CURRENT``."""

        with self._lock:
            try:
                return self.load_current(current_scope)
            except (EvidenceError, ValueError):
                pass
            for path in self._snapshot_files():
                try:
                    snapshot = self._load_snapshot(
                        path,
                        current_scope=current_scope,
                    )
                except (EvidenceError, ValueError):
                    continue
                raw = path.read_bytes()
                self._atomic_pointer(
                    {
                        "schema_version": 1,
                        "sequence": snapshot.sequence,
                        "snapshot_id": snapshot.snapshot_id,
                        "filename": path.name,
                        "file_sha256": sha256(raw).hexdigest(),
                    }
                )
                return snapshot
            raise EvidenceError("no valid same-scope FactBook snapshot is available")


__all__ = [
    "AppendOnlyEvidenceJournal",
    "ClaimAssessment",
    "ClaimRecord",
    "EvidenceError",
    "EvidenceRecordV1",
    "EvidenceScopeV1",
    "FactBookSnapshotStore",
    "FactBookSnapshotV1",
    "MODEL_ARTIFACT_CLAIM_ID",
    "assess_claims",
    "build_capability_claims",
    "build_factbook_claims",
    "capability_claim_id",
    "sha256_json",
    "sha256_text",
]
