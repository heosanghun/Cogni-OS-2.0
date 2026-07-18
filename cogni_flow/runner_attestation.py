"""Externally signed production-runner attestation import.

This module contains no signing key and never promotes an integration smoke
into a production claim.  A separate operator/assessor signs a canonical JSON
statement that binds the exact OCI runner configuration, engine, image,
commands, and implementation source.  Import is one-shot and fail-closed.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import inspect
import json
import os
from pathlib import Path
import re
import stat
from time import time_ns
from typing import Any, Mapping

from .approval import ApprovalError, canonical_json_bytes, ed25519_backend_available
from .harness import SandboxResult
from .kernel_sandbox import LinuxOciSandboxRunner
from .production import RunnerAttestation


RUNNER_ATTESTATION_SCHEMA = "cogni.self_harness.runner_attestation.v1"
RUNNER_ATTESTATION_ASSURANCE = "independently_attested_production_boundary"
RUNNER_ATTESTATION_IMPORT_SCHEMA = "cogni.self_harness.runner_attestation_import.v1"
_DIGEST = re.compile(r"[0-9a-f]{64}\Z")
_NONCE = re.compile(r"[A-Za-z0-9_-]{32,128}\Z")
_RUNNER_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")
_MAX_STATEMENT_BYTES = 32_768
_MAX_IMPORT_RECORDS = 256
_FILE_ATTRIBUTE_REPARSE_POINT = 0x400


class RunnerAttestationImportError(ApprovalError):
    """Raised when detached runner evidence is malformed or untrusted."""


class RunnerAttestationReplayError(RunnerAttestationImportError):
    """Raised when an attestation statement or nonce was already imported."""


@dataclass(frozen=True, slots=True)
class SignedRunnerAttestationV1:
    schema: str
    statement_id: str
    runner_id: str
    runner_evidence_sha256: str
    runner_source_sha256: str
    engine_path: str
    engine_sha256: str
    daemon_socket: str
    image_reference: str
    runtime: str
    allowed_command_sha256: tuple[str, ...]
    kernel_boundary: bool
    network_isolated: bool
    host_filesystem_isolated: bool
    ephemeral_workspace: bool
    production_attestation: bool
    assurance: str
    attestor_id: str
    public_key_sha256: str
    nonce: str
    issued_ns: int
    expires_ns: int

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> SignedRunnerAttestationV1:
        if set(payload) != set(cls.__dataclass_fields__):
            raise RunnerAttestationImportError(
                "runner attestation fields are not the V1 schema"
            )
        normalized = dict(payload)
        commands = normalized.get("allowed_command_sha256")
        if not isinstance(commands, list):
            raise RunnerAttestationImportError(
                "runner attestation command allowlist must be a JSON array"
            )
        normalized["allowed_command_sha256"] = tuple(commands)
        try:
            statement = cls(**normalized)
        except TypeError as exc:
            raise RunnerAttestationImportError(
                "runner attestation is malformed"
            ) from exc
        statement._validate_shape()
        expected = statement.identity_sha256()
        if statement.statement_id != expected:
            raise RunnerAttestationImportError(
                "runner attestation statement identity is invalid"
            )
        return statement

    def _validate_shape(self) -> None:
        if self.schema != RUNNER_ATTESTATION_SCHEMA:
            raise RunnerAttestationImportError(
                "runner attestation schema is unsupported"
            )
        for value, label in (
            (self.statement_id, "statement id"),
            (self.runner_evidence_sha256, "runner evidence"),
            (self.runner_source_sha256, "runner source"),
            (self.engine_sha256, "engine"),
            (self.public_key_sha256, "public key"),
        ):
            _require_digest(value, label)
        if (
            not isinstance(self.runner_id, str)
            or _RUNNER_ID.fullmatch(self.runner_id) is None
        ):
            raise RunnerAttestationImportError("runner id is invalid")
        _bounded_text(self.daemon_socket, "daemon socket", 1_024)
        if not self.daemon_socket.startswith("/"):
            raise RunnerAttestationImportError("daemon socket must be absolute")
        _bounded_text(self.engine_path, "engine path", 1_024)
        if not self.engine_path.startswith("/"):
            raise RunnerAttestationImportError("engine path must be absolute")
        _bounded_text(self.image_reference, "image reference", 512)
        if "@sha256:" not in self.image_reference:
            raise RunnerAttestationImportError(
                "runner image must be pinned by sha256 digest"
            )
        if self.runtime != "runc":
            raise RunnerAttestationImportError("runner runtime must be runc")
        commands = self.allowed_command_sha256
        if not 1 <= len(commands) <= 16:
            raise RunnerAttestationImportError(
                "runner command allowlist is outside its bound"
            )
        for command in commands:
            _require_digest(command, "allowed command")
        if tuple(sorted(set(commands))) != commands:
            raise RunnerAttestationImportError(
                "runner commands must be sorted and unique"
            )
        boundaries = (
            self.kernel_boundary,
            self.network_isolated,
            self.host_filesystem_isolated,
            self.ephemeral_workspace,
            self.production_attestation,
        )
        if not all(value is True for value in boundaries):
            raise RunnerAttestationImportError(
                "runner attestation does not assert every production boundary"
            )
        if self.assurance != RUNNER_ATTESTATION_ASSURANCE:
            raise RunnerAttestationImportError(
                "runner assurance is not an independent production attestation"
            )
        _bounded_text(self.attestor_id, "attestor id", 128)
        if not isinstance(self.nonce, str) or _NONCE.fullmatch(self.nonce) is None:
            raise RunnerAttestationImportError("runner attestation nonce is invalid")
        if (
            not isinstance(self.issued_ns, int)
            or isinstance(self.issued_ns, bool)
            or self.issued_ns < 1
            or not isinstance(self.expires_ns, int)
            or isinstance(self.expires_ns, bool)
            or self.expires_ns <= self.issued_ns
        ):
            raise RunnerAttestationImportError(
                "runner attestation validity interval is invalid"
            )

    def identity_payload(self) -> dict[str, Any]:
        payload = asdict(self)
        payload.pop("statement_id")
        payload["allowed_command_sha256"] = list(self.allowed_command_sha256)
        return payload

    def identity_sha256(self) -> str:
        return sha256(canonical_json_bytes(self.identity_payload())).hexdigest()


class RunnerAttestationImportLedger:
    """Bounded one-shot ledger for statement IDs and nonces."""

    def __init__(self, directory: str | Path, *, max_records: int = 256) -> None:
        if not 1 <= max_records <= _MAX_IMPORT_RECORDS:
            raise ValueError("runner attestation import bound is invalid")
        self.directory = Path(directory)
        self.max_records = max_records
        self.directory.mkdir(parents=True, exist_ok=True)
        _require_directory_nofollow(self.directory, "attestation import ledger")

    def consume(
        self,
        statement: SignedRunnerAttestationV1,
        *,
        attestation_sha256: str,
        imported_ns: int,
    ) -> None:
        records = self._records()
        if len(records) >= self.max_records:
            raise RunnerAttestationImportError(
                "runner attestation import ledger reached its hard bound"
            )
        if statement.statement_id in {item["statement_id"] for item in records}:
            raise RunnerAttestationReplayError(
                "runner attestation statement was already imported"
            )
        if statement.nonce in {item["nonce"] for item in records}:
            raise RunnerAttestationReplayError(
                "runner attestation nonce was already imported"
            )
        payload = {
            "schema": RUNNER_ATTESTATION_IMPORT_SCHEMA,
            "statement_id": statement.statement_id,
            "nonce": statement.nonce,
            "attestation_sha256": _require_digest(attestation_sha256, "attestation"),
            "imported_ns": imported_ns,
        }
        raw = canonical_json_bytes(payload)
        nonce_id = sha256(statement.nonce.encode("ascii")).hexdigest()
        target = self.directory / f"consumed-nonce-{nonce_id}.json"
        _write_exclusive(target, raw)

    def _records(self) -> tuple[dict[str, Any], ...]:
        _require_directory_nofollow(self.directory, "attestation import ledger")
        records: list[dict[str, Any]] = []
        entries = sorted(os.scandir(self.directory), key=lambda item: item.name)
        if len(entries) > self.max_records:
            raise RunnerAttestationImportError(
                "runner attestation import ledger crossed its hard bound"
            )
        for entry in entries:
            if len(records) >= self.max_records:
                raise RunnerAttestationImportError(
                    "runner attestation import ledger crossed its hard bound"
                )
            path = Path(entry.path)
            item_stat = entry.stat(follow_symlinks=False)
            attributes = getattr(item_stat, "st_file_attributes", 0)
            if (
                entry.is_symlink()
                or attributes & _FILE_ATTRIBUTE_REPARSE_POINT
                or not stat.S_ISREG(item_stat.st_mode)
                or not entry.name.startswith("consumed-nonce-")
                or not entry.name.endswith(".json")
                or _DIGEST.fullmatch(entry.name[15:-5]) is None
            ):
                raise RunnerAttestationImportError(
                    "runner attestation import ledger contains an invalid entry"
                )
            raw = _read_regular_bounded(path, _MAX_STATEMENT_BYTES)
            data = _strict_json(raw, "runner attestation import record")
            if set(data) != {
                "schema",
                "statement_id",
                "nonce",
                "attestation_sha256",
                "imported_ns",
            }:
                raise RunnerAttestationImportError(
                    "runner attestation import record fields are invalid"
                )
            if data["schema"] != RUNNER_ATTESTATION_IMPORT_SCHEMA:
                raise RunnerAttestationImportError(
                    "runner attestation import record schema is invalid"
                )
            _require_digest(data["statement_id"], "statement id")
            if (
                not isinstance(data["nonce"], str)
                or _NONCE.fullmatch(data["nonce"]) is None
            ):
                raise RunnerAttestationImportError(
                    "runner attestation import nonce is invalid"
                )
            nonce_id = sha256(data["nonce"].encode("ascii")).hexdigest()
            if entry.name != f"consumed-nonce-{nonce_id}.json":
                raise RunnerAttestationImportError(
                    "runner attestation import filename is invalid"
                )
            _require_digest(data["attestation_sha256"], "attestation")
            if (
                not isinstance(data["imported_ns"], int)
                or isinstance(data["imported_ns"], bool)
                or data["imported_ns"] < 1
            ):
                raise RunnerAttestationImportError(
                    "runner attestation import time is invalid"
                )
            if canonical_json_bytes(data) != raw:
                raise RunnerAttestationImportError(
                    "runner attestation import record is not canonical"
                )
            records.append(data)
        if len({item["nonce"] for item in records}) != len(records):
            raise RunnerAttestationImportError(
                "runner attestation import ledger contains a repeated nonce"
            )
        return tuple(records)


class ExternallyAttestedLinuxOciRunner:
    """Sealed adapter created only by detached attestation verification."""

    kernel_isolated = True
    production_attestation = True
    integration_smoke_only = False

    def __init__(
        self,
        delegate: LinuxOciSandboxRunner,
        statement: SignedRunnerAttestationV1,
        attestation_sha256: str,
    ) -> None:
        self.__delegate = delegate
        self.__run = delegate.run
        self.statement = statement
        self.attestation_sha256 = attestation_sha256

    def isolation_attestation(self) -> RunnerAttestation:
        return RunnerAttestation(
            version=1,
            runner_id=self.statement.runner_id,
            evidence_sha256=self.attestation_sha256,
            kernel_boundary=True,
            network_isolated=True,
            host_filesystem_isolated=True,
            ephemeral_workspace=True,
            allowed_command_sha256=self.statement.allowed_command_sha256,
        )

    def run(
        self, project: Path, command: tuple[str, ...], timeout_seconds: int
    ) -> SandboxResult:
        if not self.statement.issued_ns <= time_ns() < self.statement.expires_ns:
            raise RunnerAttestationImportError(
                "runner attestation expired before command execution"
            )
        _verify_delegate_binding(self.__delegate, self.statement)
        return self.__run(project, command, timeout_seconds)


def load_externally_attested_runner(
    delegate: LinuxOciSandboxRunner,
    attestation_path: str | Path,
    signature_path: str | Path,
    public_key_path: str | Path,
    *,
    expected_public_key_sha256: str,
    attestor_ids: tuple[str, ...],
    import_ledger: RunnerAttestationImportLedger,
    now_ns: int | None = None,
    max_validity_seconds: int = 86_400,
) -> ExternallyAttestedLinuxOciRunner:
    """Verify and one-shot import a detached external runner attestation."""

    if type(delegate) is not LinuxOciSandboxRunner:
        raise RunnerAttestationImportError(
            "runner delegate must be the exact audited LinuxOciSandboxRunner"
        )
    if not isinstance(import_ledger, RunnerAttestationImportLedger):
        raise RunnerAttestationImportError(
            "runner attestation requires a one-shot import ledger"
        )
    if not 1 <= max_validity_seconds <= 604_800:
        raise ValueError("runner attestation maximum validity is invalid")
    if not attestor_ids or len(set(attestor_ids)) != len(attestor_ids):
        raise RunnerAttestationImportError(
            "runner attestor allowlist must be non-empty and unique"
        )
    for attestor in attestor_ids:
        _bounded_text(attestor, "attestor id", 128)

    raw = _read_regular_bounded(Path(attestation_path), _MAX_STATEMENT_BYTES)
    signature = _read_regular_bounded(Path(signature_path), 64)
    if len(signature) != 64:
        raise RunnerAttestationImportError(
            "runner attestation signature must be 64 raw Ed25519 bytes"
        )
    key = _read_regular_bounded(Path(public_key_path), 32)
    if len(key) != 32:
        raise RunnerAttestationImportError(
            "runner attestation public key must be 32 raw Ed25519 bytes"
        )
    expected_key = _require_digest(expected_public_key_sha256, "public key")
    if sha256(key).hexdigest() != expected_key:
        raise RunnerAttestationImportError(
            "runner attestation public key digest is not pinned"
        )
    data = _strict_json(raw, "runner attestation")
    statement = SignedRunnerAttestationV1.from_mapping(data)
    if (
        canonical_json_bytes(
            {
                **asdict(statement),
                "allowed_command_sha256": list(statement.allowed_command_sha256),
            }
        )
        != raw
    ):
        raise RunnerAttestationImportError("runner attestation JSON is not canonical")
    current = time_ns() if now_ns is None else now_ns
    if not isinstance(current, int) or isinstance(current, bool) or current < 1:
        raise RunnerAttestationImportError("current time is invalid")
    if statement.attestor_id not in frozenset(attestor_ids):
        raise RunnerAttestationImportError("runner attestor is not allowlisted")
    if statement.public_key_sha256 != expected_key:
        raise RunnerAttestationImportError(
            "runner attestation names an unpinned public key"
        )
    if not statement.issued_ns <= current < statement.expires_ns:
        raise RunnerAttestationImportError("runner attestation is not currently valid")
    if (
        statement.expires_ns - statement.issued_ns
        > max_validity_seconds * 1_000_000_000
    ):
        raise RunnerAttestationImportError(
            "runner attestation validity exceeds the operator bound"
        )
    _verify_ed25519(key, signature, raw)
    _verify_delegate_binding(delegate, statement)
    attestation_sha256 = sha256(raw).hexdigest()
    import_ledger.consume(
        statement,
        attestation_sha256=attestation_sha256,
        imported_ns=current,
    )
    return ExternallyAttestedLinuxOciRunner(delegate, statement, attestation_sha256)


def _verify_delegate_binding(
    delegate: LinuxOciSandboxRunner,
    statement: SignedRunnerAttestationV1,
) -> None:
    evidence = delegate.evidence
    evidence_bindings = {
        "runner_id": evidence.runner_id,
        "engine_path": evidence.engine_path,
        "engine_sha256": evidence.engine_sha256,
        "daemon_socket": evidence.daemon_socket,
        "image_reference": evidence.image_reference,
        "runtime": evidence.runtime,
        "allowed_command_sha256": tuple(sorted(evidence.allowed_command_sha256)),
    }
    for field, expected in evidence_bindings.items():
        if getattr(statement, field) != expected:
            raise RunnerAttestationImportError(
                f"runner attestation does not bind delegate {field}"
            )
    delegate_evidence = getattr(delegate, "_evidence_sha256", None)
    if statement.runner_evidence_sha256 != delegate_evidence:
        raise RunnerAttestationImportError(
            "runner attestation does not bind delegate evidence bytes"
        )
    source_path_value = inspect.getsourcefile(LinuxOciSandboxRunner)
    if not source_path_value:
        raise RunnerAttestationImportError(
            "runner implementation source is unavailable"
        )
    source_path = Path(source_path_value)
    source_sha256 = sha256(
        _read_regular_bounded(source_path, 2 * 1024 * 1024)
    ).hexdigest()
    if statement.runner_source_sha256 != source_sha256:
        raise RunnerAttestationImportError(
            "runner attestation does not bind the loaded implementation source"
        )


def _verify_ed25519(public_key: bytes, signature: bytes, payload: bytes) -> None:
    if not ed25519_backend_available():
        raise RunnerAttestationImportError("Ed25519 backend is unavailable")
    try:
        from cryptography.hazmat.primitives.asymmetric.ed25519 import (
            Ed25519PublicKey,
        )

        Ed25519PublicKey.from_public_bytes(public_key).verify(signature, payload)
    except RunnerAttestationImportError:
        raise
    except Exception as exc:
        raise RunnerAttestationImportError(
            "runner attestation signature is invalid"
        ) from exc


def _strict_json(raw: bytes, label: str) -> dict[str, Any]:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise RunnerAttestationImportError(f"{label} repeats a JSON key")
            result[key] = value
        return result

    try:
        data = json.loads(raw.decode("utf-8"), object_pairs_hook=reject_duplicates)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RunnerAttestationImportError(f"{label} is not strict UTF-8 JSON") from exc
    if not isinstance(data, dict):
        raise RunnerAttestationImportError(f"{label} must be a JSON object")
    return data


def _read_regular_bounded(path: Path, maximum: int) -> bytes:
    try:
        item_stat = os.lstat(path)
        attributes = getattr(item_stat, "st_file_attributes", 0)
        if (
            path.is_symlink()
            or attributes & _FILE_ATTRIBUTE_REPARSE_POINT
            or not stat.S_ISREG(item_stat.st_mode)
            or item_stat.st_size > maximum
            or path.resolve(strict=True) != path.absolute()
        ):
            raise RunnerAttestationImportError(
                f"{path.name} must be a bounded regular non-link file"
            )
        with path.open("rb") as stream:
            opened = os.fstat(stream.fileno())
            raw = stream.read(maximum + 1)
            final = os.fstat(stream.fileno())
    except RunnerAttestationImportError:
        raise
    except OSError as exc:
        raise RunnerAttestationImportError(f"{path.name} is unavailable") from exc
    identity_changed = bool(item_stat.st_ino and opened.st_ino) and (
        (item_stat.st_dev, item_stat.st_ino) != (opened.st_dev, opened.st_ino)
    )
    if (
        identity_changed
        or not stat.S_ISREG(opened.st_mode)
        or len(raw) > maximum
        or len(raw) != opened.st_size
        or (opened.st_size, opened.st_mtime_ns) != (final.st_size, final.st_mtime_ns)
    ):
        raise RunnerAttestationImportError(f"{path.name} changed during bounded read")
    return raw


def _require_directory_nofollow(path: Path, label: str) -> None:
    try:
        item_stat = os.lstat(path)
    except OSError as exc:
        raise RunnerAttestationImportError(f"{label} is unavailable") from exc
    attributes = getattr(item_stat, "st_file_attributes", 0)
    if (
        path.is_symlink()
        or attributes & _FILE_ATTRIBUTE_REPARSE_POINT
        or not stat.S_ISDIR(item_stat.st_mode)
        or path.resolve(strict=True) != path.absolute()
    ):
        raise RunnerAttestationImportError(
            f"{label} must be a regular non-link directory"
        )


def _write_exclusive(path: Path, payload: bytes) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    try:
        descriptor = os.open(path, flags, 0o600)
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(path, 0o600)
        if os.name != "nt":
            descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
    except FileExistsError as exc:
        raise RunnerAttestationReplayError(
            "runner attestation statement was already imported"
        ) from exc
    except BaseException:
        path.unlink(missing_ok=True)
        raise


def _require_digest(value: object, label: str) -> str:
    if not isinstance(value, str) or _DIGEST.fullmatch(value) is None:
        raise RunnerAttestationImportError(
            f"{label} must be a lowercase SHA-256 digest"
        )
    return value


def _bounded_text(value: object, label: str, maximum: int) -> str:
    if (
        not isinstance(value, str)
        or not value
        or len(value) > maximum
        or any(ord(character) < 32 for character in value)
    ):
        raise RunnerAttestationImportError(f"{label} must be bounded text")
    return value


__all__ = [
    "ExternallyAttestedLinuxOciRunner",
    "RUNNER_ATTESTATION_ASSURANCE",
    "RUNNER_ATTESTATION_SCHEMA",
    "RunnerAttestationImportError",
    "RunnerAttestationImportLedger",
    "RunnerAttestationReplayError",
    "SignedRunnerAttestationV1",
    "load_externally_attested_runner",
]
