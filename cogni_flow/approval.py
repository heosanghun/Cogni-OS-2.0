"""Immutable evaluation evidence and externally signed Self-Harness approvals.

This module deliberately contains no signing key and no approval shortcut.  A
trusted operator signs the canonical approval payload outside the CogniBoard
process.  The runtime only verifies an Ed25519 signature against an explicitly
pinned raw public key and consumes each approval exactly once.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
import os
from pathlib import Path
import re
from time import time_ns
from typing import Any, Mapping


EVALUATION_SCHEMA = "cogni.self_harness.candidate_evaluation.v1"
APPROVAL_SCHEMA = "cogni.self_harness.human_approval.v1"
CONSUMPTION_SCHEMA = "cogni.self_harness.approval_consumption.v1"
ROLLBACK_AUTHORIZATION_SCHEMA = "cogni.self_harness.operator_rollback_authorization.v1"
ROLLBACK_CONSUMPTION_SCHEMA = "cogni.self_harness.operator_rollback_consumption.v1"
_DIGEST = re.compile(r"[0-9a-f]{64}\Z")
_RECORD_ID = re.compile(r"[0-9a-f]{32}\Z")
_SIGNATURE = re.compile(r"[0-9a-f]{128}\Z")
_NONCE = re.compile(r"[A-Za-z0-9_-]{32,128}\Z")
_MAX_EVIDENCE_BYTES = 32_768
_FILE_ATTRIBUTE_REPARSE_POINT = 0x400


class ApprovalError(RuntimeError):
    """Base error for malformed, untrusted, stale, or unavailable approval."""


class ApprovalReplayError(ApprovalError):
    """Raised when a one-time approval has already been consumed."""


def ed25519_backend_available() -> bool:
    """Return whether the optional Ed25519 backend works end to end.

    Package discovery or even module import is insufficient: binary wheels can
    be present while their CFFI/OpenSSL backend is missing.  This probe stays
    lazy and exercises key construction, signing, and verification without
    making ``cryptography`` a base dependency.
    """

    try:
        from cryptography.hazmat.primitives.asymmetric.ed25519 import (
            Ed25519PrivateKey,
        )

        private_key = Ed25519PrivateKey.from_private_bytes(b"\x17" * 32)
        message = b"cogni-ed25519-backend-probe-v1"
        signature = private_key.sign(message)
        private_key.public_key().verify(signature, message)
    except Exception:
        return False
    return True


def canonical_json_bytes(payload: Mapping[str, Any]) -> bytes:
    """Encode one bounded contract without implementation-dependent whitespace."""

    try:
        data = json.dumps(
            dict(payload),
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ApprovalError("approval payload is not canonical JSON") from exc
    if len(data) > _MAX_EVIDENCE_BYTES:
        raise ApprovalError("approval payload exceeds its bounded size")
    return data


def _digest(value: object, label: str) -> str:
    if not isinstance(value, str) or _DIGEST.fullmatch(value) is None:
        raise ApprovalError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _record_id(value: object) -> str:
    if not isinstance(value, str) or _RECORD_ID.fullmatch(value) is None:
        raise ApprovalError("journal record id must be lowercase hexadecimal")
    return value


def _text(value: object, label: str, *, maximum: int = 256) -> str:
    if (
        not isinstance(value, str)
        or not value
        or len(value) > maximum
        or any(ord(character) < 32 for character in value)
    ):
        raise ApprovalError(f"{label} must be bounded printable text")
    return value


def _integer(value: object, label: str, *, minimum: int = 0) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        raise ApprovalError(f"{label} must be a bounded integer")
    return value


def _safe_relative_python(value: object) -> str:
    text = _text(value, "relative path", maximum=512)
    relative = Path(text)
    if (
        relative.is_absolute()
        or not relative.parts
        or ".." in relative.parts
        or relative.suffix != ".py"
        or relative.as_posix() != text
    ):
        raise ApprovalError("approval target is not a safe relative Python path")
    return text


@dataclass(frozen=True, slots=True)
class CandidateEvaluationV1:
    schema: str
    evaluation_id: str
    proposal_id: str
    relative_path: str
    base_sha256: str
    replacement_sha256: str
    source_surface_sha256: str
    snapshot_tree_sha256: str
    runner_id: str
    runner_evidence_sha256: str
    regression_command_sha256: str
    result_sha256: str
    returncode: int
    completed_ns: int
    expires_ns: int
    status: str

    @classmethod
    def create(
        cls,
        *,
        proposal_id: str,
        relative_path: str,
        base_sha256: str,
        replacement_sha256: str,
        source_surface_sha256: str,
        snapshot_tree_sha256: str,
        runner_id: str,
        runner_evidence_sha256: str,
        regression_command_sha256: str,
        result_sha256: str,
        returncode: int,
        completed_ns: int,
        expires_ns: int,
    ) -> CandidateEvaluationV1:
        payload = {
            "schema": EVALUATION_SCHEMA,
            "proposal_id": _digest(proposal_id, "proposal id"),
            "relative_path": _safe_relative_python(relative_path),
            "base_sha256": _digest(base_sha256, "base digest"),
            "replacement_sha256": _digest(replacement_sha256, "replacement digest"),
            "source_surface_sha256": _digest(
                source_surface_sha256, "source surface digest"
            ),
            "snapshot_tree_sha256": _digest(
                snapshot_tree_sha256, "snapshot tree digest"
            ),
            "runner_id": _text(runner_id, "runner id", maximum=128),
            "runner_evidence_sha256": _digest(
                runner_evidence_sha256, "runner evidence digest"
            ),
            "regression_command_sha256": _digest(
                regression_command_sha256, "regression command digest"
            ),
            "result_sha256": _digest(result_sha256, "result digest"),
            "returncode": _integer(returncode, "return code"),
            "completed_ns": _integer(completed_ns, "completion time", minimum=1),
            "expires_ns": _integer(expires_ns, "expiry time", minimum=1),
            "status": "awaiting_approval",
        }
        if payload["expires_ns"] <= payload["completed_ns"]:
            raise ApprovalError("evaluation expiry must follow completion")
        evaluation_id = sha256(canonical_json_bytes(payload)).hexdigest()
        return cls(evaluation_id=evaluation_id, **payload)

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> CandidateEvaluationV1:
        if set(payload) != set(cls.__dataclass_fields__):
            raise ApprovalError("evaluation evidence fields are invalid")
        try:
            candidate = cls(**dict(payload))
        except TypeError as exc:
            raise ApprovalError("evaluation evidence is malformed") from exc
        expected = cls.create(
            proposal_id=candidate.proposal_id,
            relative_path=candidate.relative_path,
            base_sha256=candidate.base_sha256,
            replacement_sha256=candidate.replacement_sha256,
            source_surface_sha256=candidate.source_surface_sha256,
            snapshot_tree_sha256=candidate.snapshot_tree_sha256,
            runner_id=candidate.runner_id,
            runner_evidence_sha256=candidate.runner_evidence_sha256,
            regression_command_sha256=candidate.regression_command_sha256,
            result_sha256=candidate.result_sha256,
            returncode=candidate.returncode,
            completed_ns=candidate.completed_ns,
            expires_ns=candidate.expires_ns,
        )
        if (
            candidate.schema != EVALUATION_SCHEMA
            or candidate.status != "awaiting_approval"
        ):
            raise ApprovalError("evaluation evidence state is invalid")
        if candidate.evaluation_id != expected.evaluation_id:
            raise ApprovalError("evaluation evidence digest is invalid")
        return candidate


@dataclass(frozen=True, slots=True)
class HumanApprovalV1:
    schema: str
    evaluation_id: str
    proposal_id: str
    relative_path: str
    base_sha256: str
    replacement_sha256: str
    source_surface_sha256: str
    snapshot_tree_sha256: str
    runner_id: str
    runner_evidence_sha256: str
    regression_command_sha256: str
    nonce: str
    approver_id: str
    issued_ns: int
    expires_ns: int
    decision: str
    public_key_sha256: str
    signature: str

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> HumanApprovalV1:
        if set(payload) != set(cls.__dataclass_fields__):
            raise ApprovalError("human approval fields are invalid")
        try:
            approval = cls(**dict(payload))
        except TypeError as exc:
            raise ApprovalError("human approval is malformed") from exc
        approval._validate_shape()
        return approval

    def _validate_shape(self) -> None:
        if self.schema != APPROVAL_SCHEMA:
            raise ApprovalError("human approval schema is unsupported")
        for value, label in (
            (self.evaluation_id, "evaluation id"),
            (self.proposal_id, "proposal id"),
            (self.base_sha256, "base digest"),
            (self.replacement_sha256, "replacement digest"),
            (self.source_surface_sha256, "source surface digest"),
            (self.snapshot_tree_sha256, "snapshot tree digest"),
            (self.runner_evidence_sha256, "runner evidence digest"),
            (self.regression_command_sha256, "regression command digest"),
            (self.public_key_sha256, "public key digest"),
        ):
            _digest(value, label)
        _safe_relative_python(self.relative_path)
        _text(self.runner_id, "runner id", maximum=128)
        _text(self.approver_id, "approver id", maximum=128)
        if not isinstance(self.nonce, str) or _NONCE.fullmatch(self.nonce) is None:
            raise ApprovalError("approval nonce is invalid")
        _integer(self.issued_ns, "approval issue time", minimum=1)
        _integer(self.expires_ns, "approval expiry time", minimum=1)
        if self.expires_ns <= self.issued_ns:
            raise ApprovalError("approval expiry must follow issue time")
        if self.decision != "approve_once":
            raise ApprovalError("approval decision must be approve_once")
        if (
            not isinstance(self.signature, str)
            or _SIGNATURE.fullmatch(self.signature) is None
        ):
            raise ApprovalError("approval signature must be lowercase Ed25519 hex")

    def signed_payload(self) -> dict[str, Any]:
        payload = asdict(self)
        payload.pop("signature")
        return payload

    @property
    def approval_id(self) -> str:
        self._validate_shape()
        return sha256(canonical_json_bytes(asdict(self))).hexdigest()


@dataclass(frozen=True, slots=True)
class RollbackAuthorizationV1:
    """External one-time authority for one exact committed journal rollback."""

    schema: str
    journal_record_id: str
    relative_path: str
    before_sha256: str
    after_sha256: str
    source_surface_sha256: str
    runner_id: str
    runner_evidence_sha256: str
    health_command_sha256: str
    nonce: str
    approver_id: str
    issued_ns: int
    expires_ns: int
    decision: str
    public_key_sha256: str
    signature: str

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> RollbackAuthorizationV1:
        if set(payload) != set(cls.__dataclass_fields__):
            raise ApprovalError("rollback authorization fields are invalid")
        try:
            authorization = cls(**dict(payload))
        except TypeError as exc:
            raise ApprovalError("rollback authorization is malformed") from exc
        authorization._validate_shape()
        return authorization

    def _validate_shape(self) -> None:
        if self.schema != ROLLBACK_AUTHORIZATION_SCHEMA:
            raise ApprovalError("rollback authorization schema is unsupported")
        _record_id(self.journal_record_id)
        _safe_relative_python(self.relative_path)
        for value, label in (
            (self.before_sha256, "rollback before digest"),
            (self.after_sha256, "rollback after digest"),
            (self.source_surface_sha256, "rollback source surface digest"),
            (self.runner_evidence_sha256, "rollback runner evidence digest"),
            (self.health_command_sha256, "rollback health command digest"),
            (self.public_key_sha256, "rollback public key digest"),
        ):
            _digest(value, label)
        _text(self.runner_id, "rollback runner id", maximum=128)
        _text(self.approver_id, "rollback approver id", maximum=128)
        if not isinstance(self.nonce, str) or _NONCE.fullmatch(self.nonce) is None:
            raise ApprovalError("rollback authorization nonce is invalid")
        _integer(self.issued_ns, "rollback issue time", minimum=1)
        _integer(self.expires_ns, "rollback expiry time", minimum=1)
        if self.expires_ns <= self.issued_ns:
            raise ApprovalError("rollback expiry must follow issue time")
        if self.decision != "rollback_committed_once":
            raise ApprovalError("rollback decision must be rollback_committed_once")
        if (
            not isinstance(self.signature, str)
            or _SIGNATURE.fullmatch(self.signature) is None
        ):
            raise ApprovalError("rollback signature must be lowercase Ed25519 hex")

    def signed_payload(self) -> dict[str, Any]:
        payload = asdict(self)
        payload.pop("signature")
        return payload

    @property
    def authorization_id(self) -> str:
        self._validate_shape()
        return sha256(canonical_json_bytes(asdict(self))).hexdigest()


class Ed25519ApprovalVerifier:
    """Verify externally generated approvals without importing crypto eagerly."""

    def __init__(
        self,
        public_key_path: str | Path,
        *,
        expected_sha256: str,
        approver_ids: tuple[str, ...],
    ) -> None:
        expected = _digest(expected_sha256, "pinned public key digest")
        if not approver_ids or len(set(approver_ids)) != len(approver_ids):
            raise ApprovalError("approver allowlist must be non-empty and unique")
        for approver in approver_ids:
            _text(approver, "approver id", maximum=128)
        path = Path(public_key_path).expanduser()
        try:
            stat = os.lstat(path)
            resolved = path.resolve(strict=True)
            payload = resolved.read_bytes()
        except OSError as exc:
            raise ApprovalError("approval public key is unavailable") from exc
        if (
            path.is_symlink()
            or getattr(stat, "st_file_attributes", 0) & _FILE_ATTRIBUTE_REPARSE_POINT
        ):
            raise ApprovalError("approval public key cannot be a link/reparse point")
        if not resolved.is_file() or len(payload) != 32:
            raise ApprovalError("approval public key must be 32 raw Ed25519 bytes")
        if sha256(payload).hexdigest() != expected:
            raise ApprovalError("approval public key digest is not pinned")
        self.public_key_sha256 = expected
        self.approver_ids = frozenset(approver_ids)
        self._public_key = payload

    def verify(
        self,
        approval: HumanApprovalV1,
        evaluation: CandidateEvaluationV1,
        *,
        now_ns: int | None = None,
    ) -> str:
        if not isinstance(approval, HumanApprovalV1):
            raise ApprovalError("approval must be a HumanApprovalV1 record")
        if not isinstance(evaluation, CandidateEvaluationV1):
            raise ApprovalError("evaluation must be a CandidateEvaluationV1 record")
        approval._validate_shape()
        expected = {
            "evaluation_id": evaluation.evaluation_id,
            "proposal_id": evaluation.proposal_id,
            "relative_path": evaluation.relative_path,
            "base_sha256": evaluation.base_sha256,
            "replacement_sha256": evaluation.replacement_sha256,
            "source_surface_sha256": evaluation.source_surface_sha256,
            "snapshot_tree_sha256": evaluation.snapshot_tree_sha256,
            "runner_id": evaluation.runner_id,
            "runner_evidence_sha256": evaluation.runner_evidence_sha256,
            "regression_command_sha256": evaluation.regression_command_sha256,
        }
        for field, value in expected.items():
            if getattr(approval, field) != value:
                raise ApprovalError(f"approval does not bind the evaluation {field}")
        if approval.public_key_sha256 != self.public_key_sha256:
            raise ApprovalError("approval names an unpinned public key")
        if approval.approver_id not in self.approver_ids:
            raise ApprovalError("approver identity is not allowlisted")
        current = time_ns() if now_ns is None else _integer(now_ns, "current time")
        if approval.issued_ns < evaluation.completed_ns:
            raise ApprovalError("approval predates its candidate evaluation")
        if not approval.issued_ns <= current < approval.expires_ns:
            raise ApprovalError("human approval is not currently valid")
        if (
            current >= evaluation.expires_ns
            or approval.expires_ns > evaluation.expires_ns
        ):
            raise ApprovalError("human approval exceeds evaluation validity")
        self._verify_signature(
            approval.signature,
            approval.signed_payload(),
            invalid_message="human approval signature is invalid",
        )
        return approval.approval_id

    def verify_rollback(
        self,
        authorization: RollbackAuthorizationV1,
        *,
        journal_record_id: str,
        relative_path: str,
        before_sha256: str,
        after_sha256: str,
        source_surface_sha256: str,
        runner_id: str,
        runner_evidence_sha256: str,
        health_command_sha256: str,
        record_created_ns: int,
        now_ns: int | None = None,
    ) -> str:
        """Verify one rollback authorization against live committed facts."""

        if not isinstance(authorization, RollbackAuthorizationV1):
            raise ApprovalError(
                "rollback authorization must be a RollbackAuthorizationV1 record"
            )
        authorization._validate_shape()
        expected = {
            "journal_record_id": _record_id(journal_record_id),
            "relative_path": _safe_relative_python(relative_path),
            "before_sha256": _digest(before_sha256, "rollback before digest"),
            "after_sha256": _digest(after_sha256, "rollback after digest"),
            "source_surface_sha256": _digest(
                source_surface_sha256, "rollback source surface digest"
            ),
            "runner_id": _text(runner_id, "rollback runner id", maximum=128),
            "runner_evidence_sha256": _digest(
                runner_evidence_sha256, "rollback runner evidence digest"
            ),
            "health_command_sha256": _digest(
                health_command_sha256, "rollback health command digest"
            ),
        }
        for field, value in expected.items():
            if getattr(authorization, field) != value:
                raise ApprovalError(f"rollback authorization does not bind {field}")
        if authorization.public_key_sha256 != self.public_key_sha256:
            raise ApprovalError("rollback authorization names an unpinned public key")
        if authorization.approver_id not in self.approver_ids:
            raise ApprovalError("rollback approver identity is not allowlisted")
        created = _integer(record_created_ns, "journal creation time", minimum=1)
        current = time_ns() if now_ns is None else _integer(now_ns, "current time")
        if authorization.issued_ns < created:
            raise ApprovalError("rollback authorization predates the journal record")
        if not authorization.issued_ns <= current < authorization.expires_ns:
            raise ApprovalError("rollback authorization is not currently valid")
        self._verify_signature(
            authorization.signature,
            authorization.signed_payload(),
            invalid_message="rollback authorization signature is invalid",
        )
        return authorization.authorization_id

    def _verify_signature(
        self,
        signature_hex: str,
        payload: Mapping[str, Any],
        *,
        invalid_message: str,
    ) -> None:
        signature = bytes.fromhex(signature_hex)
        message = canonical_json_bytes(payload)
        try:
            from cryptography.exceptions import InvalidSignature
            from cryptography.hazmat.primitives.asymmetric.ed25519 import (
                Ed25519PublicKey,
            )
        except Exception as exc:
            raise ApprovalError(
                "Ed25519 verification is unavailable; signed source mutation remains disabled"
            ) from exc
        try:
            key = Ed25519PublicKey.from_public_bytes(self._public_key)
        except Exception as exc:
            raise ApprovalError(
                "Ed25519 verification is unavailable; signed source mutation remains disabled"
            ) from exc
        try:
            key.verify(signature, message)
        except InvalidSignature as exc:
            raise ApprovalError(invalid_message) from exc
        except Exception as exc:
            raise ApprovalError(
                "Ed25519 verification is unavailable; signed source mutation remains disabled"
            ) from exc


class CandidateEvaluationLedger:
    """Append-only bounded store for immutable passing regression evidence."""

    def __init__(self, directory: str | Path, *, max_records: int = 64) -> None:
        if not 1 <= max_records <= 1_024:
            raise ValueError("evaluation ledger bound is invalid")
        self.directory = _prepare_directory(Path(directory))
        self.max_records = max_records

    def record(self, evaluation: CandidateEvaluationV1) -> Path:
        if not isinstance(evaluation, CandidateEvaluationV1):
            raise TypeError("evaluation must be CandidateEvaluationV1")
        CandidateEvaluationV1.from_mapping(asdict(evaluation))
        if len(tuple(self.directory.glob("evaluation-*.json"))) >= self.max_records:
            raise ApprovalError("evaluation ledger reached its hard bound")
        target = self.directory / f"evaluation-{evaluation.evaluation_id}.json"
        _atomic_write_exclusive(target, canonical_json_bytes(asdict(evaluation)))
        return target

    def load(self, evaluation_id: str) -> CandidateEvaluationV1:
        digest = _digest(evaluation_id, "evaluation id")
        target = self.directory / f"evaluation-{digest}.json"
        payload = _read_json_record(target)
        return CandidateEvaluationV1.from_mapping(payload)

    def records(self) -> tuple[CandidateEvaluationV1, ...]:
        paths = sorted(self.directory.glob("evaluation-*.json"))
        if len(paths) > self.max_records:
            raise ApprovalError("evaluation ledger crossed its hard bound")
        records = tuple(
            CandidateEvaluationV1.from_mapping(_read_json_record(path))
            for path in paths
        )
        return tuple(sorted(records, key=lambda item: item.completed_ns))


class ConsumedApprovalLedger:
    """Atomically consume each valid external approval before source mutation."""

    def __init__(self, directory: str | Path, *, max_records: int = 256) -> None:
        if not 1 <= max_records <= 4_096:
            raise ValueError("approval consumption bound is invalid")
        self.directory = _prepare_directory(Path(directory))
        self.max_records = max_records

    def consume_once(
        self,
        approval: HumanApprovalV1,
        evaluation: CandidateEvaluationV1,
        *,
        consumed_ns: int | None = None,
    ) -> str:
        approval_id = approval.approval_id
        if len(tuple(self.directory.glob("consumed-nonce-*.json"))) >= self.max_records:
            raise ApprovalError("approval consumption ledger reached its hard bound")
        payload = {
            "schema": CONSUMPTION_SCHEMA,
            "approval_id": approval_id,
            "evaluation_id": evaluation.evaluation_id,
            "proposal_id": evaluation.proposal_id,
            "nonce": approval.nonce,
            "approver_id": approval.approver_id,
            "consumed_ns": time_ns() if consumed_ns is None else consumed_ns,
        }
        _integer(payload["consumed_ns"], "consumption time", minimum=1)
        # Key the exclusive record by nonce rather than approval signature.
        # Re-signing the same nonce with a changed expiry must never create a
        # second mutation authority.
        nonce_id = sha256(approval.nonce.encode("ascii")).hexdigest()
        target = self.directory / f"consumed-nonce-{nonce_id}.json"
        try:
            _atomic_write_exclusive(target, canonical_json_bytes(payload))
        except FileExistsError as exc:
            raise ApprovalReplayError("human approval was already consumed") from exc
        return approval_id

    def consume_rollback_once(
        self,
        authorization: RollbackAuthorizationV1,
        *,
        consumed_ns: int | None = None,
    ) -> str:
        """Consume a rollback nonce in the same global namespace as promotion."""

        if not isinstance(authorization, RollbackAuthorizationV1):
            raise TypeError("authorization must be RollbackAuthorizationV1")
        authorization._validate_shape()
        authorization_id = authorization.authorization_id
        if len(tuple(self.directory.glob("consumed-nonce-*.json"))) >= self.max_records:
            raise ApprovalError("approval consumption ledger reached its hard bound")
        payload = {
            "schema": ROLLBACK_CONSUMPTION_SCHEMA,
            "authorization_id": authorization_id,
            "journal_record_id": authorization.journal_record_id,
            "relative_path": authorization.relative_path,
            "before_sha256": authorization.before_sha256,
            "after_sha256": authorization.after_sha256,
            "source_surface_sha256": authorization.source_surface_sha256,
            "nonce": authorization.nonce,
            "approver_id": authorization.approver_id,
            "consumed_ns": time_ns() if consumed_ns is None else consumed_ns,
        }
        _integer(payload["consumed_ns"], "rollback consumption time", minimum=1)
        nonce_id = sha256(authorization.nonce.encode("ascii")).hexdigest()
        target = self.directory / f"consumed-nonce-{nonce_id}.json"
        try:
            _atomic_write_exclusive(target, canonical_json_bytes(payload))
        except FileExistsError as exc:
            raise ApprovalReplayError(
                "rollback authorization was already consumed"
            ) from exc
        return authorization_id


def _prepare_directory(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    try:
        stat = os.lstat(path)
        resolved = path.resolve(strict=True)
    except OSError as exc:
        raise ApprovalError("approval ledger directory is unavailable") from exc
    if (
        path.is_symlink()
        or getattr(stat, "st_file_attributes", 0) & _FILE_ATTRIBUTE_REPARSE_POINT
    ):
        raise ApprovalError("approval ledger directory cannot be linked")
    if not resolved.is_dir():
        raise ApprovalError("approval ledger path is not a directory")
    return resolved


def _atomic_write_exclusive(target: Path, payload: bytes) -> None:
    if (
        target.parent.resolve(strict=True) != target.parent
        or target.exists()
        or target.is_symlink()
    ):
        if target.exists():
            raise FileExistsError(target)
        raise ApprovalError("approval evidence path crossed its directory")
    descriptor = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        if os.name != "nt":
            directory_fd = os.open(
                target.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
            )
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
    except BaseException:
        target.unlink(missing_ok=True)
        raise


def _read_json_record(path: Path) -> Mapping[str, Any]:
    try:
        stat = os.lstat(path)
        if (
            path.is_symlink()
            or getattr(stat, "st_file_attributes", 0) & _FILE_ATTRIBUTE_REPARSE_POINT
        ):
            raise ApprovalError("approval evidence cannot be linked")
        if not path.is_file() or stat.st_size > _MAX_EVIDENCE_BYTES:
            raise ApprovalError("approval evidence is missing or oversized")
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ApprovalError("approval evidence is unreadable") from exc
    if not isinstance(payload, dict):
        raise ApprovalError("approval evidence must be a JSON object")
    return payload


__all__ = [
    "APPROVAL_SCHEMA",
    "ApprovalError",
    "ApprovalReplayError",
    "CandidateEvaluationLedger",
    "CandidateEvaluationV1",
    "ConsumedApprovalLedger",
    "Ed25519ApprovalVerifier",
    "HumanApprovalV1",
    "ROLLBACK_AUTHORIZATION_SCHEMA",
    "RollbackAuthorizationV1",
    "canonical_json_bytes",
    "ed25519_backend_available",
]
