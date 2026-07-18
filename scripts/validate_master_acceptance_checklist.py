"""Validate the 1--170 CogniBoard acceptance ledger.

The ledger is a release gate, not prose.  This validator rejects missing or
duplicated requirement IDs, unknown states, empty evidence/exit conditions,
and a checked box that does not exactly match ``COMPLETED``.  It also verifies
the summary totals printed near the top of the Markdown document.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import hmac
import os
from pathlib import Path, PurePosixPath
import argparse
import json
import re
import stat
from typing import Any


EXPECTED_IDS = frozenset(range(1, 171))
VALID_STATES = frozenset(
    {
        "COMPLETED",
        "IMPLEMENTED_UNVERIFIED",
        "PARTIAL",
        "NOT_IMPLEMENTED",
        "EXTERNAL_BLOCKER",
    }
)
_ROW = re.compile(
    r"^\|\s*(?P<id>\d{1,3})\s*\|\s*\[(?P<check>[ xX])\]\s*\|"
    r"(?P<requirement>.*?)\|\s*`(?P<state>[A-Z_]+)`\s*\|"
    r"(?P<evidence>.*?)\|(?P<condition>.*?)\|\s*$"
)
_SUMMARY = re.compile(
    r"현재 스냅샷 집계는\s*`COMPLETED\s+(?P<COMPLETED>\d+)\s*/\s*"
    r"IMPLEMENTED_UNVERIFIED\s+(?P<IMPLEMENTED_UNVERIFIED>\d+)\s*/\s*"
    r"PARTIAL\s+(?P<PARTIAL>\d+)\s*/\s*NOT_IMPLEMENTED\s+"
    r"(?P<NOT_IMPLEMENTED>\d+)\s*/\s*EXTERNAL_BLOCKER\s+"
    r"(?P<EXTERNAL_BLOCKER>\d+)`",
    re.MULTILINE,
)
_EVIDENCE_PATH = re.compile(
    r"(?P<path>(?:validation|release)/evidence/[A-Za-z0-9._/-]+\.json)"
)
_EVIDENCE_CITATION = re.compile(
    r"(?P<path>(?:validation|release)/evidence/[A-Za-z0-9._/-]+\.json)"
    r"#sha256=(?P<sha256>[0-9a-f]{64})"
)
_BASIS_TOKEN = re.compile(
    r"`basis=(?P<basis>STATIC_ARTIFACT|CPU_VERIFIED|MODEL_MEASURED|GPU_MEASURED|EXTERNAL_VERIFIED)`"
)
_VALID_BASES = frozenset(
    {
        "STATIC_ARTIFACT",
        "CPU_VERIFIED",
        "MODEL_MEASURED",
        "GPU_MEASURED",
        "EXTERNAL_VERIFIED",
    }
)
_BASIS_KIND = {
    "STATIC_ARTIFACT": "acceptance.static",
    "CPU_VERIFIED": "acceptance.cpu",
    "MODEL_MEASURED": "acceptance.model",
    "GPU_MEASURED": "acceptance.gpu",
    "EXTERNAL_VERIFIED": "acceptance.external",
}
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_IDENTIFIER = re.compile(r"^[a-z][a-z0-9_.:-]{0,127}$")
_COMMIT = re.compile(r"^[0-9a-f]{40}$")
_HEX = re.compile(r"^[0-9a-f]+$")
_RSA_SHA256_DIGEST_INFO = bytes.fromhex("3031300d060960864801650304020105000420")
_MAX_EVIDENCE_BYTES = 4 * 1024 * 1024
_MAX_RAW_BYTES = 256 * 1024 * 1024
_CLAIM_POLICY_RELATIVE = "config/acceptance-evidence-policy.json"
_VERIFIER_POLICY_RELATIVE = "config/release-verifier-policy.json"
_SOURCE_ROOT = Path(__file__).resolve().parents[1]
# Updated only after reviewing the complete closed 1..170 policy.  A modified
# policy cannot silently relax a claim because the validator pins its bytes.
_CLAIM_POLICY_SHA256 = (
    "fd3fe466de82199df67e52fdc9515c127c0d4df7e440111bc503cefdb99f1080"
)
# This source pin is deliberately fail-closed (`status=unconfigured`) until an
# independently governed release verifier key is approved.  Updating it is a
# reviewed source change; a detached checklist cannot substitute its own key.
_VERIFIER_POLICY_SHA256 = (
    "793ce682c30c60447db0ee28d219949f758c03fb9f447dcff6e02ad657e48315"
)


class ChecklistValidationError(ValueError):
    """The acceptance ledger violates its machine-checkable contract."""


@dataclass(frozen=True, slots=True)
class ChecklistRecord:
    requirement_id: int
    checked: bool
    requirement: str
    state: str
    evidence: str
    completion_condition: str


@dataclass(frozen=True, slots=True)
class ChecklistReport:
    records: tuple[ChecklistRecord, ...]
    counts: dict[str, int]

    def as_payload(self) -> dict[str, object]:
        incomplete = [
            record.requirement_id
            for record in self.records
            if record.state != "COMPLETED"
        ]
        return {
            "schema_version": 1,
            "requirements": len(self.records),
            "counts": dict(self.counts),
            "incomplete_count": len(incomplete),
            "incomplete_ids": incomplete,
            "valid": True,
        }


@dataclass(frozen=True, slots=True)
class ClaimEvidencePolicy:
    claim_id: str
    allowed_bases: tuple[str, ...]
    allowed_kinds: tuple[str, ...]
    required_components: tuple[str, ...]
    raw_artifact_schema: str


@dataclass(frozen=True, slots=True)
class TrustedValidationContext:
    """Subject scope derived from a source-pinned signed release attestation."""

    source_commit: str
    source_tree_digest: str
    model_sha256: str
    config_sha256: str
    device_sha256: str
    policy_sha256: str
    verifier_id: str
    public_key_sha256: str
    verifier_policy_sha256: str

    def validate(self) -> None:
        if _COMMIT.fullmatch(self.source_commit) is None:
            raise ChecklistValidationError(
                "trusted validation context source commit is invalid"
            )
        for field in (
            "source_tree_digest",
            "model_sha256",
            "config_sha256",
            "device_sha256",
            "policy_sha256",
            "public_key_sha256",
            "verifier_policy_sha256",
        ):
            if _SHA256.fullmatch(getattr(self, field)) is None:
                raise ChecklistValidationError(
                    f"trusted validation context {field} is invalid"
                )
        if _IDENTIFIER.fullmatch(self.verifier_id) is None:
            raise ChecklistValidationError(
                "trusted validation context verifier id is invalid"
            )

    def scope(self) -> dict[str, str]:
        return {
            "model_sha256": self.model_sha256,
            "code_sha256": self.source_tree_digest,
            "config_sha256": self.config_sha256,
            "device_sha256": self.device_sha256,
            "policy_sha256": self.policy_sha256,
        }


def _reject_duplicate_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
    value: dict[str, object] = {}
    for key, item in pairs:
        if key in value:
            raise ChecklistValidationError(f"duplicate evidence JSON key: {key}")
        value[key] = item
    return value


def _canonical_bytes(value: object) -> bytes:
    try:
        encoded = json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError) as exc:
        raise ChecklistValidationError("evidence is not canonical JSON") from exc
    return encoded.encode("utf-8")


def _load_strict_json(data: bytes, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(
            data.decode("utf-8", errors="strict"),
            object_pairs_hook=_reject_duplicate_pairs,
            parse_constant=lambda value: (_ for _ in ()).throw(
                ChecklistValidationError(f"non-finite evidence number: {value}")
            ),
        )
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ChecklistValidationError(f"{label} is not strict JSON") from exc
    if type(payload) is not dict:
        raise ChecklistValidationError(f"{label} must be a JSON object")
    return payload


def _is_reparse(stat_result: object) -> bool:
    attributes = getattr(stat_result, "st_file_attributes", 0)
    flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    return bool(attributes & flag)


def _normalized_repository_path(
    *, root: Path, relative: str, label: str, allowed_prefixes: tuple[str, ...]
) -> tuple[Path, Path, os.stat_result]:
    if (
        type(relative) is not str
        or not relative
        or "\\" in relative
        or ":" in relative
        or relative.startswith("/")
        or re.fullmatch(r"[A-Za-z0-9._/-]+", relative) is None
    ):
        raise ChecklistValidationError(f"{label} path is not normalized repo-relative")
    parts = relative.split("/")
    if any(part in {"", ".", ".."} for part in parts):
        raise ChecklistValidationError(f"{label} path contains an unsafe segment")
    pure = PurePosixPath(relative)
    if pure.is_absolute() or pure.as_posix() != relative:
        raise ChecklistValidationError(f"{label} path is not canonical")
    if not any(relative.startswith(prefix) for prefix in allowed_prefixes):
        raise ChecklistValidationError(f"{label} path is outside its approved subtree")

    try:
        resolved_root = root.resolve(strict=True)
    except OSError as exc:
        raise ChecklistValidationError("repository root cannot be resolved") from exc
    candidate = resolved_root
    try:
        for part in parts:
            candidate = candidate / part
            item_stat = candidate.lstat()
            if stat.S_ISLNK(item_stat.st_mode) or _is_reparse(item_stat):
                raise ChecklistValidationError(
                    f"{label} path crosses a link/reparse point"
                )
        resolved = candidate.resolve(strict=True)
    except OSError as exc:
        raise ChecklistValidationError(f"cannot resolve {label} path") from exc
    if not resolved.is_relative_to(resolved_root):
        raise ChecklistValidationError(f"{label} path escapes the repository")
    final_stat = resolved.lstat()
    if stat.S_ISLNK(final_stat.st_mode) or _is_reparse(final_stat):
        raise ChecklistValidationError(f"{label} is a link/reparse point")
    if not stat.S_ISREG(final_stat.st_mode):
        raise ChecklistValidationError(f"{label} is not a regular file")
    return resolved_root, resolved, final_stat


def _file_object_identity(value: os.stat_result) -> tuple[int, ...]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_nlink,
        value.st_size,
    )


def _file_content_identity(value: os.stat_result) -> tuple[int, ...]:
    return (
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _open_repository_descriptor(
    *, resolved_root: Path, relative: str, path: Path, label: str
) -> int:
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0)
    if os.name == "nt":
        # FILE_SHARE_READ permits nested validators to read an outer release
        # transaction's locked snapshot, while still denying write/delete (and
        # therefore rename replacement) until post-read identity checks finish.
        # OPEN_REPARSE_POINT ensures the leaf is not followed.
        import ctypes
        import msvcrt

        create_file = ctypes.windll.kernel32.CreateFileW
        create_file.argtypes = (
            ctypes.c_wchar_p,
            ctypes.c_uint32,
            ctypes.c_uint32,
            ctypes.c_void_p,
            ctypes.c_uint32,
            ctypes.c_uint32,
            ctypes.c_void_p,
        )
        create_file.restype = ctypes.c_void_p
        handle = create_file(
            str(path),
            0x80000000,  # GENERIC_READ
            1,  # FILE_SHARE_READ only; no write/delete sharing
            None,
            3,  # OPEN_EXISTING
            0x00200000 | 0x08000000,  # OPEN_REPARSE_POINT | SEQUENTIAL_SCAN
            None,
        )
        invalid = ctypes.c_void_p(-1).value
        if handle in {None, invalid}:
            raise ChecklistValidationError(
                f"cannot exclusively open {label} without following reparse points"
            )
        try:
            return msvcrt.open_osfhandle(handle, flags)
        except OSError:
            ctypes.windll.kernel32.CloseHandle(handle)
            raise

    directory_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    no_follow = getattr(os, "O_NOFOLLOW", 0)
    if not no_follow or not hasattr(os, "supports_dir_fd"):
        raise ChecklistValidationError(
            f"{label} cannot be opened with a no-follow directory chain"
        )
    directory_flags |= no_follow
    directory_descriptor = os.open(resolved_root, directory_flags)
    try:
        parts = relative.split("/")
        for part in parts[:-1]:
            next_descriptor = os.open(
                part,
                directory_flags,
                dir_fd=directory_descriptor,
            )
            os.close(directory_descriptor)
            directory_descriptor = next_descriptor
        return os.open(
            parts[-1],
            flags | no_follow,
            dir_fd=directory_descriptor,
        )
    finally:
        os.close(directory_descriptor)


def _read_repository_file(
    *,
    root: Path,
    relative: str,
    maximum: int,
    label: str,
    allowed_prefixes: tuple[str, ...],
) -> bytes:
    """Read one admitted repository file from a single locked identity.

    Path validation, link/reparse rejection, bounded reading and both identity
    checks happen around one OS file descriptor.  No second path-based
    ``read_bytes`` call is made after admission.
    """

    resolved_root, path, before = _normalized_repository_path(
        root=root,
        relative=relative,
        label=label,
        allowed_prefixes=allowed_prefixes,
    )
    try:
        descriptor = _open_repository_descriptor(
            resolved_root=resolved_root,
            relative=relative,
            path=path,
            label=label,
        )
    except OSError as exc:
        raise ChecklistValidationError(
            f"cannot open {label} without following links"
        ) from exc
    try:
        opened = os.fstat(descriptor)
        if not stat.S_ISREG(opened.st_mode):
            raise ChecklistValidationError(f"{label} descriptor is not a regular file")
        if opened.st_nlink != 1:
            raise ChecklistValidationError(f"{label} must not be hard-linked")
        if _file_object_identity(before) != _file_object_identity(opened):
            raise ChecklistValidationError(f"{label} identity changed before open")
        if not 0 < opened.st_size <= maximum:
            raise ChecklistValidationError(f"{label} size is outside policy")
        chunks: list[bytes] = []
        remaining = opened.st_size
        while remaining:
            chunk = os.read(descriptor, min(remaining, 1024 * 1024))
            if not chunk:
                raise ChecklistValidationError(
                    f"{label} ended before its admitted size"
                )
            chunks.append(chunk)
            remaining -= len(chunk)
        if os.read(descriptor, 1):
            raise ChecklistValidationError(f"{label} grew while being read")
        after_handle = os.fstat(descriptor)
        try:
            after_path = path.lstat()
            after_resolved = path.resolve(strict=True)
        except OSError as exc:
            raise ChecklistValidationError(f"cannot revalidate {label}") from exc
        if (
            stat.S_ISLNK(after_path.st_mode)
            or _is_reparse(after_path)
            or after_resolved != path
            or not after_resolved.is_relative_to(resolved_root)
            or _file_object_identity(opened) != _file_object_identity(after_handle)
            or _file_content_identity(opened) != _file_content_identity(after_handle)
            or _file_object_identity(opened) != _file_object_identity(after_path)
        ):
            raise ChecklistValidationError(f"{label} identity changed while being read")
        data = b"".join(chunks)
        if len(data) != opened.st_size:
            raise ChecklistValidationError(f"{label} changed while being read")
        return data
    except OSError as exc:
        raise ChecklistValidationError(f"cannot read {label}") from exc
    finally:
        os.close(descriptor)


def _require_digest(value: object, label: str) -> str:
    if type(value) is not str or _SHA256.fullmatch(value) is None:
        raise ChecklistValidationError(f"{label} is not a lowercase SHA-256 digest")
    return value


def _load_claim_policy(
    source_root: Path = _SOURCE_ROOT,
) -> tuple[dict[str, ClaimEvidencePolicy], str]:
    root = source_root
    policy_data = _read_repository_file(
        root=root,
        relative=_CLAIM_POLICY_RELATIVE,
        maximum=1024 * 1024,
        label="acceptance claim policy",
        allowed_prefixes=("config/",),
    )
    policy_sha256 = hashlib.sha256(policy_data).hexdigest()
    if not hmac.compare_digest(policy_sha256, _CLAIM_POLICY_SHA256):
        raise ChecklistValidationError(
            "acceptance claim policy digest differs from the source-pinned digest"
        )
    payload = _load_strict_json(policy_data, "acceptance claim policy")
    if set(payload) != {"schema", "claims"}:
        raise ChecklistValidationError("acceptance claim policy schema is invalid")
    if payload["schema"] != "cogni.acceptance.claim-policy.v1":
        raise ChecklistValidationError("acceptance claim policy identity is invalid")
    entries = payload["claims"]
    if type(entries) is not list or len(entries) != len(EXPECTED_IDS):
        raise ChecklistValidationError("acceptance claim policy must define 170 claims")
    policies: dict[str, ClaimEvidencePolicy] = {}
    required_keys = {
        "claim_id",
        "allowed_bases",
        "allowed_kinds",
        "required_components",
        "raw_artifact_schema",
    }
    for entry in entries:
        if type(entry) is not dict or set(entry) != required_keys:
            raise ChecklistValidationError("acceptance claim policy entry is invalid")
        claim_id = entry["claim_id"]
        if (
            type(claim_id) is not str
            or re.fullmatch(r"acceptance\.id(?:[1-9]|[1-9]\d|1[0-6]\d|170)", claim_id)
            is None
            or claim_id in policies
        ):
            raise ChecklistValidationError("acceptance claim policy ID is invalid")
        bases = entry["allowed_bases"]
        kinds = entry["allowed_kinds"]
        components = entry["required_components"]
        if (
            type(bases) is not list
            or not bases
            or bases != sorted(set(bases))
            or any(type(item) is not str or item not in _VALID_BASES for item in bases)
        ):
            raise ChecklistValidationError(f"{claim_id} allowed_bases are invalid")
        expected_kinds = sorted(_BASIS_KIND[item] for item in bases)
        if (
            type(kinds) is not list
            or kinds != expected_kinds
            or kinds != sorted(set(kinds))
        ):
            raise ChecklistValidationError(f"{claim_id} allowed_kinds are invalid")
        if (
            type(components) is not list
            or not components
            or components != sorted(set(components))
            or any(
                type(item) is not str or _IDENTIFIER.fullmatch(item) is None
                for item in components
            )
        ):
            raise ChecklistValidationError(
                f"{claim_id} required_components are invalid"
            )
        if entry["raw_artifact_schema"] != "cogni.acceptance.artifact.v1":
            raise ChecklistValidationError(
                f"{claim_id} raw artifact schema is not approved"
            )
        policies[claim_id] = ClaimEvidencePolicy(
            claim_id=claim_id,
            allowed_bases=tuple(bases),
            allowed_kinds=tuple(kinds),
            required_components=tuple(components),
            raw_artifact_schema=entry["raw_artifact_schema"],
        )
    expected_claims = {f"acceptance.id{item}" for item in EXPECTED_IDS}
    if set(policies) != expected_claims:
        raise ChecklistValidationError("acceptance claim policy coverage is incomplete")
    return policies, policy_sha256


def _load_verifier_policy(
    source_root: Path = _SOURCE_ROOT,
) -> tuple[str, str, str]:
    policy_data = _read_repository_file(
        root=source_root,
        relative=_VERIFIER_POLICY_RELATIVE,
        maximum=64 * 1024,
        label="release verifier policy",
        allowed_prefixes=("config/",),
    )
    verifier_policy_sha256 = hashlib.sha256(policy_data).hexdigest()
    if not hmac.compare_digest(verifier_policy_sha256, _VERIFIER_POLICY_SHA256):
        raise ChecklistValidationError(
            "release verifier policy digest differs from the source-pinned digest"
        )
    policy = _load_strict_json(
        policy_data,
        "release verifier policy",
    )
    if set(policy) != {"schema", "status", "verifier_id", "public_key_sha256"}:
        raise ChecklistValidationError("release verifier policy schema is invalid")
    if policy["schema"] != "cogni.release.verifier-policy.v1":
        raise ChecklistValidationError("release verifier policy identity is invalid")
    if policy["status"] != "approved":
        raise ChecklistValidationError(
            "release verifier policy is not independently approved; completion is blocked"
        )
    verifier_id = policy["verifier_id"]
    if type(verifier_id) is not str or _IDENTIFIER.fullmatch(verifier_id) is None:
        raise ChecklistValidationError("approved verifier id is invalid")
    return (
        verifier_id,
        _require_digest(policy["public_key_sha256"], "approved verifier public key"),
        verifier_policy_sha256,
    )


def _load_public_key(data: bytes) -> tuple[str, int, int]:
    key = _load_strict_json(data, "acceptance verifier public key")
    if set(key) != {"schema", "key_id", "algorithm", "modulus_hex", "exponent"}:
        raise ChecklistValidationError(
            "acceptance verifier public key schema is invalid"
        )
    if (
        key["schema"] != "cogni.rsa.public_key.v1"
        or key["algorithm"] != "rsa-pkcs1v15-sha256"
        or type(key["key_id"]) is not str
        or _IDENTIFIER.fullmatch(key["key_id"]) is None
        or type(key["modulus_hex"]) is not str
        or not 512 <= len(key["modulus_hex"]) <= 2048
        or len(key["modulus_hex"]) % 2
        or _HEX.fullmatch(key["modulus_hex"]) is None
        or type(key["exponent"]) is not int
        or key["exponent"] != 65537
    ):
        raise ChecklistValidationError("acceptance verifier public key is invalid")
    modulus = int(key["modulus_hex"], 16)
    if not 2048 <= modulus.bit_length() <= 8192:
        raise ChecklistValidationError("acceptance verifier public key size is invalid")
    return key["key_id"], modulus, key["exponent"]


def _verify_signature(
    data: bytes, signature_data: bytes, modulus: int, exponent: int
) -> None:
    try:
        signature_text = signature_data.decode("ascii", errors="strict")
    except UnicodeError as exc:
        raise ChecklistValidationError("acceptance signature is not ASCII") from exc
    if _HEX.fullmatch(signature_text) is None or len(signature_text) % 2:
        raise ChecklistValidationError(
            "acceptance signature must be exact lowercase hex"
        )
    signature = bytes.fromhex(signature_text)
    width = (modulus.bit_length() + 7) // 8
    if len(signature) != width:
        raise ChecklistValidationError("acceptance signature width is invalid")
    encoded = pow(int.from_bytes(signature, "big"), exponent, modulus).to_bytes(
        width, "big"
    )
    digest_info = _RSA_SHA256_DIGEST_INFO + hashlib.sha256(data).digest()
    padding_length = width - len(digest_info) - 3
    if padding_length < 8:
        raise ChecklistValidationError("acceptance verifier key is too small")
    expected = b"\x00\x01" + b"\xff" * padding_length + b"\x00" + digest_info
    if not hmac.compare_digest(encoded, expected):
        raise ChecklistValidationError("acceptance evidence signature is invalid")


def _read_detached_file(path: str | Path, maximum: int, label: str) -> bytes:
    """Read one canonical absolute detached input without following links."""

    candidate = Path(path)
    if not candidate.is_absolute():
        raise ChecklistValidationError(f"{label} path must be canonical absolute")
    try:
        resolved = candidate.resolve(strict=True)
    except OSError as exc:
        raise ChecklistValidationError(f"cannot resolve {label} path") from exc
    if candidate != resolved or resolved.name in {"", ".", ".."}:
        raise ChecklistValidationError(f"{label} path must be canonical absolute")
    return _read_repository_file(
        root=resolved.parent,
        relative=resolved.name,
        maximum=maximum,
        label=label,
        allowed_prefixes=(resolved.name,),
    )


def _load_trusted_validation_context(
    *,
    attestation_path: str | Path,
    signature_path: str | Path,
    public_key_path: str | Path,
    claim_policy_sha256: str,
    source_root: Path = _SOURCE_ROOT,
) -> TrustedValidationContext:
    """Derive SUBJECT scope from the existing independently signed release attestation.

    The caller supplies files, never trusted digest strings.  The verifier key
    must match the policy whose exact bytes are pinned in this source file.
    This keeps the effective completion ledger and its evidence detached from
    the immutable release-candidate SUBJECT tree and removes the former hash
    fixed-point/self-reference problem.
    """

    verifier_id, approved_key_sha256, verifier_policy_sha256 = _load_verifier_policy(
        source_root
    )
    key_data = _read_detached_file(
        public_key_path, 64 * 1024, "release verifier public key"
    )
    if not hmac.compare_digest(
        hashlib.sha256(key_data).hexdigest(), approved_key_sha256
    ):
        raise ChecklistValidationError(
            "release verifier public key differs from source-pinned policy"
        )
    key_id, modulus, exponent = _load_public_key(key_data)
    if key_id != verifier_id:
        raise ChecklistValidationError(
            "release verifier public key id differs from source-pinned policy"
        )
    attestation_data = _read_detached_file(
        attestation_path, _MAX_EVIDENCE_BYTES, "release subject attestation"
    )
    signature_data = _read_detached_file(
        signature_path, 64 * 1024, "release subject attestation signature"
    )
    _verify_signature(attestation_data, signature_data, modulus, exponent)
    attestation = _load_strict_json(attestation_data, "release subject attestation")
    required = {
        "schema",
        "status",
        "verifier_id",
        "source_commit",
        "summary_sha256",
        "cpu_evidence_sha256",
        "gpu5_evidence_sha256",
        "source_tree_digest",
        "model_manifest_sha256",
        "model_tree_digest",
        "config_digest",
        "device_digest",
        "runtime_evidence_sha256",
        "completion_evidence_sha256",
        "identity_pre_sha256",
        "identity_post_sha256",
        "config_evidence_sha256",
        "device_evidence_sha256",
        "model_inventory_sha256",
        "issued_at_utc",
    }
    if set(attestation) != required:
        raise ChecklistValidationError("release subject attestation schema is invalid")
    if (
        attestation["schema"] != "cogni.release.attestation.v2"
        or attestation["status"] != "passed"
        or attestation["verifier_id"] != verifier_id
        or type(attestation["source_commit"]) is not str
        or _COMMIT.fullmatch(attestation["source_commit"]) is None
    ):
        raise ChecklistValidationError(
            "release subject attestation identity is invalid"
        )
    digest_fields = required.difference(
        {"schema", "status", "verifier_id", "source_commit", "issued_at_utc"}
    )
    for field in digest_fields:
        _require_digest(attestation[field], f"release attestation {field}")
    try:
        issued = datetime.fromisoformat(
            str(attestation["issued_at_utc"]).replace("Z", "+00:00")
        )
    except ValueError as exc:
        raise ChecklistValidationError(
            "release subject attestation timestamp is invalid"
        ) from exc
    if issued.tzinfo is None or issued.utcoffset() != timezone.utc.utcoffset(issued):
        raise ChecklistValidationError(
            "release subject attestation timestamp must identify UTC"
        )
    context = TrustedValidationContext(
        source_commit=attestation["source_commit"],
        source_tree_digest=attestation["source_tree_digest"],
        model_sha256=attestation["model_tree_digest"],
        config_sha256=attestation["config_digest"],
        device_sha256=attestation["device_digest"],
        policy_sha256=claim_policy_sha256,
        verifier_id=verifier_id,
        public_key_sha256=approved_key_sha256,
        verifier_policy_sha256=verifier_policy_sha256,
    )
    context.validate()
    return context


def _validate_acceptance_payload(
    *,
    data: bytes,
    record: dict[str, Any],
    basis: str,
    verifier_id: str,
    components: list[str],
    policy_sha256: str,
    validation_context: TrustedValidationContext,
) -> None:
    payload = _load_strict_json(data, "acceptance raw payload")
    required = {
        "schema",
        "status",
        "basis",
        "verifier_id",
        "source_commit",
        "source_tree_digest",
        "model_sha256",
        "config_sha256",
        "device_sha256",
        "claim_ids",
        "artifact_sha256",
        "components",
        "policy_sha256",
    }
    if set(payload) != required:
        raise ChecklistValidationError("acceptance raw payload schema is invalid")
    scope = record["scope"]
    if (
        payload["schema"] != "cogni.acceptance.attestation.v1"
        or payload["status"] != "passed"
        or payload["basis"] != basis
        or payload["verifier_id"] != verifier_id
        or type(payload["source_commit"]) is not str
        or _COMMIT.fullmatch(payload["source_commit"]) is None
        or payload["source_commit"] != validation_context.source_commit
        or payload["source_tree_digest"] != scope["code_sha256"]
        or payload["model_sha256"] != scope["model_sha256"]
        or payload["config_sha256"] != scope["config_sha256"]
        or payload["device_sha256"] != scope["device_sha256"]
        or payload["claim_ids"] != record["claim_ids"]
        or payload["artifact_sha256"] != record["artifact_sha256"]
        or payload["components"] != components
        or payload["policy_sha256"] != policy_sha256
        or scope != validation_context.scope()
    ):
        raise ChecklistValidationError(
            "acceptance raw payload does not bind exact scope"
        )


def _validate_raw_artifact(
    *,
    data: bytes,
    record: dict[str, Any],
    basis: str,
    components: list[str],
    claim_policies: dict[str, ClaimEvidencePolicy],
    policy_sha256: str,
) -> None:
    artifact = _load_strict_json(data, "acceptance raw artifact")
    required = {
        "schema",
        "status",
        "basis",
        "components",
        "component_results",
        "claim_ids",
        "claim_results",
        "policy_sha256",
        "scope",
    }
    if set(artifact) != required:
        raise ChecklistValidationError("acceptance raw artifact schema is invalid")
    if (
        artifact["schema"] != "cogni.acceptance.artifact.v1"
        or artifact["status"] != "passed"
        or artifact["basis"] != basis
        or artifact["components"] != components
        or artifact["claim_ids"] != record["claim_ids"]
        or artifact["policy_sha256"] != policy_sha256
        or artifact["scope"] != record["scope"]
    ):
        raise ChecklistValidationError("acceptance raw artifact is not exact-scope")
    component_results = artifact["component_results"]
    if type(component_results) is not list or len(component_results) != len(components):
        raise ChecklistValidationError(
            "acceptance raw artifact component results are invalid"
        )
    result_components: list[str] = []
    for result in component_results:
        if (
            type(result) is not dict
            or set(result) != {"component", "status", "evidence_sha256"}
            or type(result["component"]) is not str
            or result["component"] not in components
            or result["status"] != "passed"
            or type(result["evidence_sha256"]) is not str
            or _SHA256.fullmatch(result["evidence_sha256"]) is None
        ):
            raise ChecklistValidationError(
                "acceptance raw artifact component result is invalid"
            )
        result_components.append(result["component"])
    if result_components != components:
        raise ChecklistValidationError(
            "acceptance raw artifact component results are not canonical"
        )
    results = artifact["claim_results"]
    if type(results) is not list or len(results) != len(record["claim_ids"]):
        raise ChecklistValidationError(
            "acceptance raw artifact claim results are invalid"
        )
    result_claims: list[str] = []
    for result in results:
        if (
            type(result) is not dict
            or set(result) != {"claim_id", "status", "result_sha256"}
            or type(result["claim_id"]) is not str
            or result["claim_id"] not in claim_policies
            or result["status"] != "passed"
            or type(result["result_sha256"]) is not str
            or _SHA256.fullmatch(result["result_sha256"]) is None
        ):
            raise ChecklistValidationError(
                "acceptance raw artifact claim result is invalid"
            )
        result_claims.append(result["claim_id"])
    if result_claims != record["claim_ids"]:
        raise ChecklistValidationError(
            "acceptance raw artifact claim results are not canonical"
        )


def _load_approved_evidence(
    *,
    root: Path,
    relative: str,
    pinned_sha256: str,
    basis: str,
    requirement_id: int,
    claim_policies: dict[str, ClaimEvidencePolicy],
    policy_sha256: str,
    validation_context: TrustedValidationContext,
) -> None:
    validation_context.validate()
    if validation_context.policy_sha256 != policy_sha256:
        raise ChecklistValidationError(
            "trusted validation context does not bind the source-pinned policy"
        )
    evidence_data = _read_repository_file(
        root=root,
        relative=relative,
        maximum=_MAX_EVIDENCE_BYTES,
        label="acceptance evidence",
        allowed_prefixes=("release/evidence/", "validation/evidence/"),
    )
    if hashlib.sha256(evidence_data).hexdigest() != pinned_sha256:
        raise ChecklistValidationError("acceptance evidence citation digest mismatch")
    payload = _load_strict_json(evidence_data, "acceptance evidence")
    required = {
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
    if set(payload) != required:
        raise ChecklistValidationError(
            "acceptance evidence does not match EvidenceRecordV1"
        )
    if (
        payload["record_type"] != "evidence"
        or type(payload["schema_version"]) is not int
        or payload["schema_version"] != 1
        or type(payload["evidence_id"]) is not str
        or re.fullmatch(r"ev1-[0-9a-f]{64}", payload["evidence_id"]) is None
        or type(payload["recorded_at"]) is not str
        or payload["evidence_class"] != "verified"
    ):
        raise ChecklistValidationError("acceptance evidence identity is invalid")
    try:
        recorded = datetime.fromisoformat(payload["recorded_at"].replace("Z", "+00:00"))
    except ValueError as exc:
        raise ChecklistValidationError(
            "acceptance evidence timestamp is invalid"
        ) from exc
    if recorded.tzinfo is None or recorded.utcoffset() is None:
        raise ChecklistValidationError("acceptance evidence timestamp lacks a timezone")
    for field in ("kind", "producer", "run_id"):
        if (
            type(payload[field]) is not str
            or _IDENTIFIER.fullmatch(payload[field]) is None
        ):
            raise ChecklistValidationError(f"acceptance evidence {field} is invalid")
    for field in ("artifact_sha256", "payload_sha256"):
        if type(payload[field]) is not str or _SHA256.fullmatch(payload[field]) is None:
            raise ChecklistValidationError(f"acceptance evidence {field} is invalid")
    claims = payload["claim_ids"]
    if (
        type(claims) is not list
        or not claims
        or len(claims) > 10000
        or any(
            type(item) is not str or _IDENTIFIER.fullmatch(item) is None
            for item in claims
        )
        or claims != sorted(set(claims))
    ):
        raise ChecklistValidationError("acceptance evidence claim_ids are invalid")
    claim_id = f"acceptance.id{requirement_id}"
    if claim_id not in claims:
        raise ChecklistValidationError(
            f"acceptance evidence does not authorize {claim_id}"
        )
    if any(item not in claim_policies for item in claims):
        raise ChecklistValidationError(
            "acceptance evidence contains a claim outside the closed policy"
        )
    metadata = payload["metadata"]
    if (
        type(metadata) is not dict
        or len(metadata) > 64
        or any(
            type(key) is not str
            or _IDENTIFIER.fullmatch(key) is None
            or type(value) is not str
            or not 1 <= len(value) <= 1024
            for key, value in metadata.items()
        )
    ):
        raise ChecklistValidationError("acceptance evidence metadata is invalid")
    required_metadata = {
        "acceptance_basis",
        "artifact_path",
        "payload_path",
        "public_key_path",
        "signature_path",
        "verifier_id",
        "components",
        "verifier_policy_sha256",
    }
    if not required_metadata.issubset(metadata):
        raise ChecklistValidationError("acceptance evidence metadata is incomplete")
    if metadata["acceptance_basis"] != basis or basis not in _VALID_BASES:
        raise ChecklistValidationError("acceptance evidence basis is mismatched")
    expected_kind = _BASIS_KIND[basis]
    if payload["kind"] != expected_kind:
        raise ChecklistValidationError(
            "acceptance evidence kind does not match its basis"
        )
    scope = payload["scope"]
    if (
        type(scope) is not dict
        or set(scope)
        != {
            "model_sha256",
            "code_sha256",
            "config_sha256",
            "device_sha256",
            "policy_sha256",
        }
        or any(
            type(value) is not str or _SHA256.fullmatch(value) is None
            for value in scope.values()
        )
    ):
        raise ChecklistValidationError("acceptance evidence scope is invalid")
    if scope["policy_sha256"] != policy_sha256:
        raise ChecklistValidationError(
            "acceptance evidence does not bind the source-pinned claim policy"
        )
    if scope != validation_context.scope():
        raise ChecklistValidationError(
            "acceptance evidence scope differs from trusted current validation context"
        )
    component_text = metadata["components"]
    if type(component_text) is not str:
        raise ChecklistValidationError("acceptance evidence components are invalid")
    components = component_text.split(",")
    if (
        not components
        or components != sorted(set(components))
        or any(_IDENTIFIER.fullmatch(item) is None for item in components)
    ):
        raise ChecklistValidationError("acceptance evidence components are invalid")
    for asserted_claim in claims:
        claim_policy = claim_policies[asserted_claim]
        if basis not in claim_policy.allowed_bases:
            raise ChecklistValidationError(
                f"{asserted_claim} cannot be completed with basis {basis}"
            )
        if payload["kind"] not in claim_policy.allowed_kinds:
            raise ChecklistValidationError(
                f"{asserted_claim} cannot be completed with kind {payload['kind']}"
            )
        missing_components = sorted(
            set(claim_policy.required_components).difference(components)
        )
        if missing_components:
            raise ChecklistValidationError(
                f"{asserted_claim} is missing required evidence components: "
                + ", ".join(missing_components)
            )

    identity = {
        "schema_version": payload["schema_version"],
        "recorded_at": payload["recorded_at"],
        "kind": payload["kind"],
        "producer": payload["producer"],
        "run_id": payload["run_id"],
        "evidence_class": payload["evidence_class"],
        "claim_ids": payload["claim_ids"],
        "artifact_sha256": payload["artifact_sha256"],
        "payload_sha256": payload["payload_sha256"],
        "scope": payload["scope"],
        "metadata": payload["metadata"],
    }
    expected_id = "ev1-" + hashlib.sha256(_canonical_bytes(identity)).hexdigest()
    if payload["evidence_id"] != expected_id:
        raise ChecklistValidationError(
            "acceptance evidence_id does not match canonical EvidenceRecordV1 content"
        )

    raw_files: dict[str, bytes] = {}
    for field in ("artifact_path", "payload_path", "public_key_path", "signature_path"):
        raw_files[field] = _read_repository_file(
            root=root,
            relative=metadata[field],
            maximum=_MAX_RAW_BYTES
            if field in {"artifact_path", "payload_path"}
            else _MAX_EVIDENCE_BYTES,
            label=f"acceptance {field}",
            allowed_prefixes=("release/evidence/", "validation/evidence/"),
        )
    if (
        hashlib.sha256(raw_files["artifact_path"]).hexdigest()
        != payload["artifact_sha256"]
    ):
        raise ChecklistValidationError("acceptance raw artifact digest mismatch")
    if (
        hashlib.sha256(raw_files["payload_path"]).hexdigest()
        != payload["payload_sha256"]
    ):
        raise ChecklistValidationError("acceptance raw payload digest mismatch")

    verifier_id = validation_context.verifier_id
    approved_key_sha256 = validation_context.public_key_sha256
    verifier_policy_sha256 = validation_context.verifier_policy_sha256
    if metadata["verifier_id"] != verifier_id or payload["producer"] != verifier_id:
        raise ChecklistValidationError("acceptance verifier identity is not approved")
    if metadata["verifier_policy_sha256"] != verifier_policy_sha256:
        raise ChecklistValidationError(
            "acceptance record does not bind the exact verifier policy"
        )
    if hashlib.sha256(raw_files["public_key_path"]).hexdigest() != approved_key_sha256:
        raise ChecklistValidationError("acceptance public key is not source-approved")
    key_id, modulus, exponent = _load_public_key(raw_files["public_key_path"])
    if key_id != verifier_id:
        raise ChecklistValidationError("acceptance public key id is not approved")
    _verify_signature(evidence_data, raw_files["signature_path"], modulus, exponent)
    _validate_raw_artifact(
        data=raw_files["artifact_path"],
        record=payload,
        basis=basis,
        components=components,
        claim_policies=claim_policies,
        policy_sha256=policy_sha256,
    )
    _validate_acceptance_payload(
        data=raw_files["payload_path"],
        record=payload,
        basis=basis,
        verifier_id=verifier_id,
        components=components,
        policy_sha256=policy_sha256,
        validation_context=validation_context,
    )


def validate_checklist(
    path: str | Path,
    *,
    release_attestation: str | Path | None = None,
    release_attestation_signature: str | Path | None = None,
    verifier_public_key: str | Path | None = None,
    require_complete: bool = False,
    _source_root: Path = _SOURCE_ROOT,
) -> ChecklistReport:
    claim_policies, policy_sha256 = _load_claim_policy(_source_root)
    context_inputs = (
        release_attestation,
        release_attestation_signature,
        verifier_public_key,
    )
    if any(item is not None for item in context_inputs) and not all(
        item is not None for item in context_inputs
    ):
        raise ChecklistValidationError(
            "signed detached release context requires attestation, signature, and key"
        )
    validation_context = (
        _load_trusted_validation_context(
            attestation_path=release_attestation,
            signature_path=release_attestation_signature,
            public_key_path=verifier_public_key,
            claim_policy_sha256=policy_sha256,
            source_root=_source_root,
        )
        if all(item is not None for item in context_inputs)
        else None
    )
    try:
        source = Path(path).resolve(strict=True)
    except OSError as exc:
        raise ChecklistValidationError("cannot resolve acceptance checklist") from exc
    checklist_data = _read_repository_file(
        root=source.parent,
        relative=source.name,
        maximum=_MAX_EVIDENCE_BYTES,
        label="acceptance checklist",
        allowed_prefixes=(source.name,),
    )
    try:
        text = checklist_data.decode("utf-8", errors="strict")
    except UnicodeError as exc:
        raise ChecklistValidationError(
            "acceptance checklist is not strict UTF-8"
        ) from exc
    records: list[ChecklistRecord] = []
    for line_number, line in enumerate(text.splitlines(), 1):
        match = _ROW.fullmatch(line)
        if match is None:
            continue
        requirement_id = int(match.group("id"))
        state = match.group("state")
        if state not in VALID_STATES:
            raise ChecklistValidationError(
                f"line {line_number}: unknown state {state!r}"
            )
        record = ChecklistRecord(
            requirement_id=requirement_id,
            checked=match.group("check").casefold() == "x",
            requirement=match.group("requirement").strip(),
            state=state,
            evidence=match.group("evidence").strip(),
            completion_condition=match.group("condition").strip(),
        )
        if not all((record.requirement, record.evidence, record.completion_condition)):
            raise ChecklistValidationError(
                f"line {line_number}: requirement, evidence, and completion "
                "condition must be non-empty"
            )
        if record.checked != (record.state == "COMPLETED"):
            raise ChecklistValidationError(
                f"line {line_number}: checkbox must be checked exactly for COMPLETED"
            )
        if record.state == "COMPLETED":
            bases = list(_BASIS_TOKEN.finditer(record.evidence))
            if len(bases) != 1:
                raise ChecklistValidationError(
                    f"line {line_number}: COMPLETED requires exactly one approved basis field"
                )
            loose_paths = {
                match.group("path")
                for match in _EVIDENCE_PATH.finditer(record.evidence)
            }
            citations = list(_EVIDENCE_CITATION.finditer(record.evidence))
            bound_paths = {match.group("path") for match in citations}
            if len(citations) != 1 or loose_paths != bound_paths:
                raise ChecklistValidationError(
                    f"line {line_number}: COMPLETED requires one content-addressed evidence citation"
                )
            if validation_context is None:
                raise ChecklistValidationError(
                    f"line {line_number}: COMPLETED requires a source-pinned signed "
                    "detached release attestation"
                )
            citation = citations[0]
            _load_approved_evidence(
                root=source.parent.parent,
                relative=citation.group("path"),
                pinned_sha256=citation.group("sha256"),
                basis=bases[0].group("basis"),
                requirement_id=requirement_id,
                claim_policies=claim_policies,
                policy_sha256=policy_sha256,
                validation_context=validation_context,
            )
        elif _BASIS_TOKEN.search(record.evidence) is not None:
            raise ChecklistValidationError(
                f"line {line_number}: only COMPLETED may declare an approved basis"
            )
        records.append(record)

    identifiers = [record.requirement_id for record in records]
    duplicates = sorted(
        requirement_id
        for requirement_id, count in Counter(identifiers).items()
        if count > 1
    )
    missing = sorted(EXPECTED_IDS.difference(identifiers))
    unexpected = sorted(set(identifiers).difference(EXPECTED_IDS))
    if duplicates or missing or unexpected:
        raise ChecklistValidationError(
            "requirement ID coverage failed: "
            f"duplicates={duplicates}, missing={missing}, unexpected={unexpected}"
        )
    ordered = tuple(sorted(records, key=lambda record: record.requirement_id))
    counts = {state: 0 for state in sorted(VALID_STATES)}
    counts.update(Counter(record.state for record in ordered))
    summaries = list(_SUMMARY.finditer(text))
    if len(summaries) != 1:
        raise ChecklistValidationError(
            "exactly one machine-readable status summary is required"
        )
    summary = summaries[0]
    declared = {state: int(summary.group(state)) for state in VALID_STATES}
    if declared != counts:
        raise ChecklistValidationError(
            f"declared status counts {declared} do not match ledger {counts}"
        )
    if sum(counts.values()) != len(EXPECTED_IDS):
        raise ChecklistValidationError("status counts do not total 170")
    if require_complete and counts["COMPLETED"] != len(EXPECTED_IDS):
        raise ChecklistValidationError(
            "release acceptance requires all 170 detached effective rows COMPLETED"
        )
    return ChecklistReport(ordered, counts)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "checklist",
        nargs="?",
        default="docs/COGNIBOARD_MASTER_ACCEPTANCE_CHECKLIST_KO.md",
    )
    parser.add_argument("--json", action="store_true", dest="as_json")
    parser.add_argument("--release-attestation")
    parser.add_argument("--release-attestation-signature")
    parser.add_argument("--verifier-public-key")
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()
    report = validate_checklist(
        args.checklist,
        release_attestation=args.release_attestation,
        release_attestation_signature=args.release_attestation_signature,
        verifier_public_key=args.verifier_public_key,
        require_complete=args.require_complete,
    )
    payload = report.as_payload()
    if args.as_json:
        print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    else:
        counts = payload["counts"]
        print(
            "PASS: 170 requirements; "
            + ", ".join(f"{state}={counts[state]}" for state in sorted(counts))
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
