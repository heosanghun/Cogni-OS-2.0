"""Manifest-bound, CPU-only semantic embedding boundary.

The production RAG path remains lexical unless an operator supplies a local
embedding artifact and separately attests retrieval quality.  This module
implements the missing *runtime boundary* without silently downloading a model
or turning an untested embedding into answer-bearing authority.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from math import isfinite, sqrt
import os
from pathlib import Path, PurePosixPath
import re
import stat
from threading import RLock
from typing import Callable, Mapping, Protocol, Sequence


SEMANTIC_EMBEDDER_SCHEMA = "cogni.semantic-embedder.v1"
SEMANTIC_EMBEDDER_MANIFEST = "semantic-embedder.manifest.json"
TRANSFORMERS_MEAN_POOL_BACKEND = "transformers_mean_pool_v1"
MAX_MANIFEST_BYTES = 128 * 1024
MAX_ARTIFACT_FILES = 128
MAX_ARTIFACT_DIRECTORIES = 64
MAX_ARTIFACT_BYTES = 2 * 1024 * 1024 * 1024
MAX_MODEL_CONFIG_BYTES = 1024 * 1024
MAX_EMBEDDING_DIMENSIONS = 8_192
MAX_EMBEDDING_BATCH = 16
MAX_EMBEDDING_INPUT_CHARS = 16_384
MAX_SEMANTIC_MODEL_PARAMETERS = 250_000_000
MAX_SEMANTIC_ESTIMATED_PEAK_RAM_BYTES = 4 * 1024 * 1024 * 1024
_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_MODEL_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._/-]{0,127}\Z")
_SPDX_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9.+-]{0,63}\Z")
_REVISION_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{6,127}\Z")
_WINDOWS_REPARSE_POINT = 0x0400
_SAFE_MODEL_FIELDS = {
    "bert": (
        "hidden_size",
        "intermediate_size",
        "num_hidden_layers",
        "num_attention_heads",
    ),
    "deberta-v2": (
        "hidden_size",
        "intermediate_size",
        "num_hidden_layers",
        "num_attention_heads",
    ),
    "modernbert": (
        "hidden_size",
        "intermediate_size",
        "num_hidden_layers",
        "num_attention_heads",
    ),
    "mpnet": (
        "hidden_size",
        "intermediate_size",
        "num_hidden_layers",
        "num_attention_heads",
    ),
    "roberta": (
        "hidden_size",
        "intermediate_size",
        "num_hidden_layers",
        "num_attention_heads",
    ),
    "xlm-roberta": (
        "hidden_size",
        "intermediate_size",
        "num_hidden_layers",
        "num_attention_heads",
    ),
    "distilbert": ("dim", "hidden_dim", "n_layers", "n_heads"),
}


class SemanticEmbedderError(RuntimeError):
    """A stable fail-closed semantic embedder error."""

    def __init__(self, code: str, message: str) -> None:
        if re.fullmatch(r"[A-Z][A-Z0-9_]{0,63}", code) is None:
            raise ValueError("semantic embedder error code is invalid")
        super().__init__(message)
        self.code = code


class SemanticEncoderBackend(Protocol):
    """Small injectable boundary used by the verified session."""

    def encode(self, texts: tuple[str, ...]) -> Sequence[Sequence[float]]: ...

    def close(self) -> None: ...


@dataclass(frozen=True, slots=True)
class SemanticEmbedderManifest:
    root: Path
    path: Path
    manifest_sha256: str
    model_id: str
    revision: str
    backend: str
    dimensions: int
    max_input_chars: int
    max_batch_size: int
    license_spdx: str
    license_file: str
    files: tuple[tuple[str, str], ...]
    total_artifact_bytes: int
    model_type: str
    model_max_tokens: int
    estimated_parameter_upper_bound: int
    estimated_peak_ram_bytes: int

    @property
    def profile(self) -> str:
        return f"local_semantic_{self.manifest_sha256[:16]}_v1"

    def status_payload(self, *, loaded: bool) -> dict[str, object]:
        return {
            "schema": SEMANTIC_EMBEDDER_SCHEMA,
            "model_id": self.model_id,
            "revision": self.revision,
            "backend": self.backend,
            "profile": self.profile,
            "dimensions": self.dimensions,
            "max_input_chars": self.max_input_chars,
            "max_batch_size": self.max_batch_size,
            "manifest_sha256": self.manifest_sha256,
            "artifact_files": len(self.files),
            "artifact_bytes": self.total_artifact_bytes,
            "model_type": self.model_type,
            "model_max_tokens": self.model_max_tokens,
            "estimated_parameter_upper_bound": (self.estimated_parameter_upper_bound),
            "estimated_peak_ram_bytes": self.estimated_peak_ram_bytes,
            "resource_policy": "bounded_transformer_encoder_v1",
            "artifact_verified": True,
            "device": "cpu",
            "vram_bytes": 0,
            "network_access": False,
            "loaded": loaded,
            "semantic_embedding": True,
            "license_spdx_declared": self.license_spdx,
            "license_status": "manifest_declared_unreviewed",
            "quality_attested": False,
            "answer_bearing": False,
            "production_ready": False,
        }


@dataclass(frozen=True, slots=True)
class _FileIdentity:
    device: int
    inode: int
    mode: int
    links: int
    size: int
    modified_ns: int
    changed_ns: int


@dataclass(frozen=True, slots=True)
class _ObservedFile:
    path: Path
    identity: _FileIdentity


@dataclass(frozen=True, slots=True)
class _SafeModelConfig:
    model_type: str
    hidden_size: int
    intermediate_size: int
    layers: int
    attention_heads: int
    vocabulary_size: int
    max_tokens: int
    estimated_parameter_upper_bound: int


def _is_reparse_or_link(path: Path) -> bool:
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise SemanticEmbedderError(
            "SEMANTIC_ARTIFACT_UNAVAILABLE", "semantic artifact could not be inspected"
        ) from exc
    attributes = int(getattr(metadata, "st_file_attributes", 0))
    return stat.S_ISLNK(metadata.st_mode) or bool(attributes & _WINDOWS_REPARSE_POINT)


def _identity(metadata: os.stat_result) -> _FileIdentity:
    return _FileIdentity(
        device=int(metadata.st_dev),
        inode=int(metadata.st_ino),
        mode=int(metadata.st_mode),
        links=int(metadata.st_nlink),
        size=int(metadata.st_size),
        modified_ns=int(metadata.st_mtime_ns),
        changed_ns=int(metadata.st_ctime_ns),
    )


def _object_identity(value: _FileIdentity) -> tuple[int, int, int, int]:
    return (value.device, value.inode, value.mode, value.links)


def _reject_network_root(value: str | Path) -> None:
    raw = os.fspath(Path(value).expanduser())
    normalized = raw.replace("/", "\\")
    drive = Path(raw).drive.replace("/", "\\")
    if normalized.startswith("\\\\") or drive.startswith("\\\\"):
        raise SemanticEmbedderError(
            "SEMANTIC_ARTIFACT_UNSAFE",
            "semantic artifact root cannot be a UNC or network path",
        )


def _regular_path_identity(
    path: Path,
    *,
    code: str,
    label: str,
) -> _FileIdentity:
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise SemanticEmbedderError(code, f"{label} is unavailable") from exc
    attributes = int(getattr(metadata, "st_file_attributes", 0))
    if (
        stat.S_ISLNK(metadata.st_mode)
        or bool(attributes & _WINDOWS_REPARSE_POINT)
        or not stat.S_ISREG(metadata.st_mode)
    ):
        raise SemanticEmbedderError(code, f"{label} must be a local regular file")
    return _identity(metadata)


def _assert_canonical_artifact_path(root: Path, path: Path, *, label: str) -> None:
    try:
        resolved = path.resolve(strict=True)
    except OSError as exc:
        raise SemanticEmbedderError(
            "SEMANTIC_ARTIFACT_UNAVAILABLE", f"{label} is unavailable"
        ) from exc
    if resolved != path or not resolved.is_relative_to(root):
        raise SemanticEmbedderError(
            "SEMANTIC_ARTIFACT_UNSAFE", f"{label} escaped the semantic root"
        )


def _read_bounded_regular_file(
    path: Path,
    *,
    maximum_bytes: int,
    code: str,
    label: str,
    expected_identity: _FileIdentity | None = None,
) -> tuple[bytes, _FileIdentity]:
    before = _regular_path_identity(path, code=code, label=label)
    if expected_identity is not None and before != expected_identity:
        raise SemanticEmbedderError(code, f"{label} changed before it was opened")
    if not 1 <= before.size <= maximum_bytes:
        raise SemanticEmbedderError(code, f"{label} exceeds its byte limit")
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = -1
    try:
        descriptor = os.open(path, flags)
        opened = _identity(os.fstat(descriptor))
        if (
            not stat.S_ISREG(opened.mode)
            or _object_identity(opened) != _object_identity(before)
            or opened.size != before.size
        ):
            raise SemanticEmbedderError(code, f"{label} changed while being opened")
        output = bytearray()
        while len(output) <= maximum_bytes:
            remaining = maximum_bytes + 1 - len(output)
            chunk = os.read(descriptor, min(1024 * 1024, remaining))
            if not chunk:
                break
            output.extend(chunk)
        if len(output) > maximum_bytes or len(output) != before.size:
            raise SemanticEmbedderError(code, f"{label} changed while being read")
        after = _identity(os.fstat(descriptor))
    except SemanticEmbedderError:
        raise
    except OSError as exc:
        raise SemanticEmbedderError(code, f"{label} could not be read") from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    path_after = _regular_path_identity(path, code=code, label=label)
    if after != opened or path_after != before:
        raise SemanticEmbedderError(code, f"{label} changed while being read")
    return bytes(output), before


def _hash_bounded_regular_file(
    root: Path,
    observed: _ObservedFile,
    *,
    maximum_bytes: int,
) -> tuple[str, int]:
    path = observed.path
    _assert_canonical_artifact_path(root, path, label="semantic artifact file")
    before = _regular_path_identity(
        path,
        code="SEMANTIC_ARTIFACT_UNAVAILABLE",
        label="semantic artifact file",
    )
    if before != observed.identity:
        raise SemanticEmbedderError(
            "SEMANTIC_ARTIFACT_CHANGED",
            "semantic artifact changed after inventory",
        )
    if not 1 <= before.size <= maximum_bytes:
        raise SemanticEmbedderError(
            "SEMANTIC_ARTIFACT_TOO_LARGE",
            "semantic artifact exceeds the remaining byte limit",
        )
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = -1
    try:
        descriptor = os.open(path, flags)
        opened = _identity(os.fstat(descriptor))
        if (
            not stat.S_ISREG(opened.mode)
            or _object_identity(opened) != _object_identity(before)
            or opened.size != before.size
        ):
            raise SemanticEmbedderError(
                "SEMANTIC_ARTIFACT_CHANGED",
                "semantic artifact changed while being opened",
            )
        digest = sha256()
        consumed = 0
        while consumed <= maximum_bytes:
            remaining = maximum_bytes + 1 - consumed
            chunk = os.read(descriptor, min(1024 * 1024, remaining))
            if not chunk:
                break
            consumed += len(chunk)
            digest.update(chunk)
        if consumed > maximum_bytes or consumed != before.size:
            raise SemanticEmbedderError(
                "SEMANTIC_ARTIFACT_CHANGED",
                "semantic artifact grew or shrank while being hashed",
            )
        after = _identity(os.fstat(descriptor))
    except SemanticEmbedderError:
        raise
    except OSError as exc:
        raise SemanticEmbedderError(
            "SEMANTIC_ARTIFACT_UNAVAILABLE",
            "semantic artifact could not be hashed",
        ) from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    path_after = _regular_path_identity(
        path,
        code="SEMANTIC_ARTIFACT_UNAVAILABLE",
        label="semantic artifact file",
    )
    if after != opened or path_after != before:
        raise SemanticEmbedderError(
            "SEMANTIC_ARTIFACT_CHANGED",
            "semantic artifact changed while being hashed",
        )
    return digest.hexdigest(), consumed


def _safe_relative_file(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value or len(value) > 256:
        raise SemanticEmbedderError(
            "SEMANTIC_MANIFEST_INVALID", f"{label} must be a bounded relative path"
        )
    candidate = PurePosixPath(value)
    if (
        candidate.is_absolute()
        or value.startswith(("/", "\\"))
        or "\\" in value
        or any(part in {"", ".", ".."} for part in candidate.parts)
        or candidate.as_posix() != value
    ):
        raise SemanticEmbedderError(
            "SEMANTIC_MANIFEST_INVALID", f"{label} must be a safe POSIX path"
        )
    return value


def _bounded_int(value: object, *, minimum: int, maximum: int, label: str) -> int:
    if (
        not isinstance(value, int)
        or isinstance(value, bool)
        or not minimum <= value <= maximum
    ):
        raise SemanticEmbedderError(
            "SEMANTIC_MANIFEST_INVALID", f"{label} is outside the admitted bound"
        )
    return value


def _strict_json_object(
    raw: bytes,
    *,
    code: str,
    label: str,
) -> dict[str, object]:
    def reject_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
        output: dict[str, object] = {}
        for key, value in pairs:
            if key in output:
                raise SemanticEmbedderError(code, f"{label} contains duplicate keys")
            output[key] = value
        return output

    def reject_constant(_value: str) -> object:
        raise SemanticEmbedderError(code, f"{label} contains a non-finite number")

    try:
        payload = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=reject_duplicates,
            parse_constant=reject_constant,
        )
    except SemanticEmbedderError:
        raise
    except (UnicodeError, json.JSONDecodeError, RecursionError) as exc:
        raise SemanticEmbedderError(code, f"{label} could not be decoded") from exc
    if not isinstance(payload, dict):
        raise SemanticEmbedderError(code, f"{label} must be an object")
    return payload


def _read_manifest(
    path: Path,
) -> tuple[dict[str, object], bytes, _FileIdentity]:
    try:
        raw, identity = _read_bounded_regular_file(
            path,
            maximum_bytes=MAX_MANIFEST_BYTES,
            code="SEMANTIC_MANIFEST_INVALID",
            label="semantic manifest",
        )
    except SemanticEmbedderError:
        raise
    except OSError as exc:
        raise SemanticEmbedderError(
            "SEMANTIC_MANIFEST_INVALID", "semantic manifest could not be read"
        ) from exc
    payload = _strict_json_object(
        raw,
        code="SEMANTIC_MANIFEST_INVALID",
        label="semantic manifest",
    )
    return payload, raw, identity


def _directory_identity(path: Path) -> _FileIdentity:
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise SemanticEmbedderError(
            "SEMANTIC_ARTIFACT_UNAVAILABLE",
            "semantic artifact directory could not be inspected",
        ) from exc
    attributes = int(getattr(metadata, "st_file_attributes", 0))
    if (
        stat.S_ISLNK(metadata.st_mode)
        or bool(attributes & _WINDOWS_REPARSE_POINT)
        or not stat.S_ISDIR(metadata.st_mode)
    ):
        raise SemanticEmbedderError(
            "SEMANTIC_ARTIFACT_UNSAFE",
            "semantic artifact directories must be local regular directories",
        )
    return _identity(metadata)


def _inventory_regular_files(root: Path) -> dict[str, _ObservedFile]:
    observed: dict[str, _ObservedFile] = {}
    root_identity = _directory_identity(root)
    pending = [(root, root_identity)]
    directory_count = 1
    while pending:
        directory, expected_directory_identity = pending.pop()
        before_directory = _directory_identity(directory)
        if before_directory != expected_directory_identity:
            raise SemanticEmbedderError(
                "SEMANTIC_ARTIFACT_CHANGED",
                "semantic artifact directory changed during inventory",
            )
        try:
            entries = os.scandir(directory)
        except OSError as exc:
            raise SemanticEmbedderError(
                "SEMANTIC_ARTIFACT_UNAVAILABLE",
                "semantic artifact directory could not be inventoried",
            ) from exc
        try:
            with entries:
                for entry in entries:
                    candidate = Path(entry.path)
                    try:
                        # Windows DirEntry.stat() can report zeroed inode/device/link
                        # values. Path.lstat() preserves the identity used by the
                        # subsequent no-follow open/fstat checks.
                        metadata = candidate.lstat()
                    except OSError as exc:
                        raise SemanticEmbedderError(
                            "SEMANTIC_ARTIFACT_UNAVAILABLE",
                            "semantic artifact changed during inventory",
                        ) from exc
                    attributes = int(getattr(metadata, "st_file_attributes", 0))
                    if stat.S_ISLNK(metadata.st_mode) or bool(
                        attributes & _WINDOWS_REPARSE_POINT
                    ):
                        raise SemanticEmbedderError(
                            "SEMANTIC_ARTIFACT_UNSAFE",
                            "semantic artifact cannot contain links or reparse points",
                        )
                    identity = _identity(metadata)
                    relative = candidate.relative_to(root).as_posix()
                    if stat.S_ISDIR(metadata.st_mode):
                        directory_count += 1
                        if directory_count > MAX_ARTIFACT_DIRECTORIES:
                            raise SemanticEmbedderError(
                                "SEMANTIC_ARTIFACT_TOO_LARGE",
                                "semantic artifact exceeds its directory limit",
                            )
                        pending.append((candidate, identity))
                    elif stat.S_ISREG(metadata.st_mode):
                        if relative in observed:
                            raise SemanticEmbedderError(
                                "SEMANTIC_ARTIFACT_UNSAFE",
                                "semantic artifact contains a duplicate path",
                            )
                        if len(observed) >= MAX_ARTIFACT_FILES + 1:
                            raise SemanticEmbedderError(
                                "SEMANTIC_ARTIFACT_TOO_LARGE",
                                "semantic artifact exceeds its file inventory limit",
                            )
                        observed[relative] = _ObservedFile(candidate, identity)
                    else:
                        raise SemanticEmbedderError(
                            "SEMANTIC_ARTIFACT_UNSAFE",
                            "semantic artifact contains an unsupported filesystem entry",
                        )
        except SemanticEmbedderError:
            raise
        except OSError as exc:
            raise SemanticEmbedderError(
                "SEMANTIC_ARTIFACT_UNAVAILABLE",
                "semantic artifact changed during inventory",
            ) from exc
        after_directory = _directory_identity(directory)
        if after_directory != before_directory:
            raise SemanticEmbedderError(
                "SEMANTIC_ARTIFACT_CHANGED",
                "semantic artifact directory changed during inventory",
            )
    return observed


def _config_int(
    payload: Mapping[str, object],
    key: str,
    *,
    minimum: int,
    maximum: int,
) -> int:
    value = payload.get(key)
    if (
        not isinstance(value, int)
        or isinstance(value, bool)
        or not minimum <= value <= maximum
    ):
        raise SemanticEmbedderError(
            "SEMANTIC_MODEL_CONFIG_UNSAFE",
            f"semantic model config field {key} is outside its bound",
        )
    return value


def _safe_model_config(raw: bytes, *, dimensions: int) -> _SafeModelConfig:
    payload = _strict_json_object(
        raw,
        code="SEMANTIC_MODEL_CONFIG_UNSAFE",
        label="semantic model config",
    )
    model_type = payload.get("model_type")
    if not isinstance(model_type, str) or model_type not in _SAFE_MODEL_FIELDS:
        raise SemanticEmbedderError(
            "SEMANTIC_MODEL_CONFIG_UNSAFE",
            "semantic model type is not in the bounded encoder allowlist",
        )
    if any(
        key in payload
        for key in ("auto_map", "custom_pipelines", "quantization_config")
    ):
        raise SemanticEmbedderError(
            "SEMANTIC_MODEL_CONFIG_UNSAFE",
            "semantic model config requests an unsupported dynamic runtime",
        )
    hidden_key, intermediate_key, layers_key, heads_key = _SAFE_MODEL_FIELDS[model_type]
    hidden_size = _config_int(
        payload, hidden_key, minimum=2, maximum=MAX_EMBEDDING_DIMENSIONS
    )
    intermediate_size = _config_int(
        payload, intermediate_key, minimum=hidden_size, maximum=32_768
    )
    layers = _config_int(payload, layers_key, minimum=1, maximum=48)
    attention_heads = _config_int(payload, heads_key, minimum=1, maximum=128)
    vocabulary_size = _config_int(payload, "vocab_size", minimum=128, maximum=500_000)
    max_tokens = _config_int(
        payload, "max_position_embeddings", minimum=32, maximum=8_192
    )
    type_vocabulary_size = payload.get("type_vocab_size", 1)
    if (
        not isinstance(type_vocabulary_size, int)
        or isinstance(type_vocabulary_size, bool)
        or not 1 <= type_vocabulary_size <= 32
    ):
        raise SemanticEmbedderError(
            "SEMANTIC_MODEL_CONFIG_UNSAFE",
            "semantic model type vocabulary is outside its bound",
        )
    if hidden_size != dimensions or hidden_size % attention_heads:
        raise SemanticEmbedderError(
            "SEMANTIC_MODEL_CONFIG_UNSAFE",
            "semantic model dimensions or attention geometry do not match",
        )

    embedding_parameters = (
        vocabulary_size * hidden_size
        + max_tokens * hidden_size
        + type_vocabulary_size * hidden_size
        + hidden_size * hidden_size
    )
    per_layer_parameters = (
        4 * hidden_size * hidden_size
        + 2 * hidden_size * intermediate_size
        + 16 * hidden_size
        + intermediate_size
    )
    raw_upper_bound = embedding_parameters + layers * per_layer_parameters
    estimated_parameter_upper_bound = (raw_upper_bound * 5 + 3) // 4
    if estimated_parameter_upper_bound > MAX_SEMANTIC_MODEL_PARAMETERS:
        raise SemanticEmbedderError(
            "SEMANTIC_MODEL_CONFIG_UNSAFE",
            "semantic model estimated parameter count exceeds its limit",
        )
    return _SafeModelConfig(
        model_type=model_type,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        layers=layers,
        attention_heads=attention_heads,
        vocabulary_size=vocabulary_size,
        max_tokens=max_tokens,
        estimated_parameter_upper_bound=estimated_parameter_upper_bound,
    )


def verify_semantic_embedder_manifest(
    root: str | Path,
    manifest_path: str | Path | None = None,
) -> SemanticEmbedderManifest:
    """Verify a closed-world local embedding model without loading executable code."""

    _reject_network_root(root)
    lexical_root = Path(os.path.abspath(Path(root).expanduser()))
    try:
        if _is_reparse_or_link(lexical_root):
            raise SemanticEmbedderError(
                "SEMANTIC_ARTIFACT_UNSAFE",
                "semantic artifact root cannot be a link or reparse point",
            )
        selected_root = lexical_root.resolve(strict=True)
    except OSError as exc:
        raise SemanticEmbedderError(
            "SEMANTIC_ARTIFACT_UNAVAILABLE", "semantic artifact root is unavailable"
        ) from exc
    if selected_root != lexical_root or not selected_root.is_dir():
        raise SemanticEmbedderError(
            "SEMANTIC_ARTIFACT_UNSAFE",
            "semantic artifact root must be a local directory",
        )
    lexical_manifest = Path(
        selected_root / SEMANTIC_EMBEDDER_MANIFEST
        if manifest_path is None
        else os.path.abspath(Path(manifest_path).expanduser())
    )
    if manifest_path is not None:
        _reject_network_root(manifest_path)
    try:
        if _is_reparse_or_link(lexical_manifest):
            raise SemanticEmbedderError(
                "SEMANTIC_MANIFEST_INVALID", "semantic manifest cannot be a link"
            )
        selected_manifest = lexical_manifest.resolve(strict=True)
    except OSError as exc:
        raise SemanticEmbedderError(
            "SEMANTIC_MANIFEST_INVALID", "semantic manifest is unavailable"
        ) from exc
    if (
        selected_manifest != lexical_manifest
        or selected_manifest.parent != selected_root
        or selected_manifest.name != SEMANTIC_EMBEDDER_MANIFEST
    ):
        raise SemanticEmbedderError(
            "SEMANTIC_MANIFEST_INVALID",
            "semantic manifest must be the fixed root manifest",
        )

    payload, raw_manifest, manifest_identity = _read_manifest(selected_manifest)
    expected_keys = {
        "schema",
        "model_id",
        "revision",
        "backend",
        "dimensions",
        "max_input_chars",
        "max_batch_size",
        "license_spdx",
        "license_file",
        "files",
    }
    if (
        set(payload) != expected_keys
        or payload.get("schema") != SEMANTIC_EMBEDDER_SCHEMA
    ):
        raise SemanticEmbedderError(
            "SEMANTIC_MANIFEST_INVALID", "semantic manifest fields are invalid"
        )
    model_id = payload["model_id"]
    revision = payload["revision"]
    backend = payload["backend"]
    license_spdx = payload["license_spdx"]
    if not isinstance(model_id, str) or _MODEL_ID_RE.fullmatch(model_id) is None:
        raise SemanticEmbedderError(
            "SEMANTIC_MANIFEST_INVALID", "semantic model identity is invalid"
        )
    if not isinstance(revision, str) or _REVISION_RE.fullmatch(revision) is None:
        raise SemanticEmbedderError(
            "SEMANTIC_MANIFEST_INVALID", "semantic model revision is invalid"
        )
    if backend != TRANSFORMERS_MEAN_POOL_BACKEND:
        raise SemanticEmbedderError(
            "SEMANTIC_BACKEND_UNSUPPORTED", "semantic backend is not admitted"
        )
    if not isinstance(license_spdx, str) or _SPDX_RE.fullmatch(license_spdx) is None:
        raise SemanticEmbedderError(
            "SEMANTIC_MANIFEST_INVALID", "semantic model license declaration is invalid"
        )
    dimensions = _bounded_int(
        payload["dimensions"],
        minimum=2,
        maximum=MAX_EMBEDDING_DIMENSIONS,
        label="dimensions",
    )
    max_input_chars = _bounded_int(
        payload["max_input_chars"],
        minimum=32,
        maximum=MAX_EMBEDDING_INPUT_CHARS,
        label="max_input_chars",
    )
    max_batch_size = _bounded_int(
        payload["max_batch_size"],
        minimum=1,
        maximum=MAX_EMBEDDING_BATCH,
        label="max_batch_size",
    )
    license_file = _safe_relative_file(payload["license_file"], label="license_file")
    files_payload = payload["files"]
    if (
        not isinstance(files_payload, dict)
        or not 1 <= len(files_payload) <= MAX_ARTIFACT_FILES
    ):
        raise SemanticEmbedderError(
            "SEMANTIC_MANIFEST_INVALID", "semantic artifact file table is invalid"
        )
    files: list[tuple[str, str]] = []
    for raw_name, raw_digest in files_payload.items():
        name = _safe_relative_file(raw_name, label="artifact file")
        if not isinstance(raw_digest, str) or _SHA256_RE.fullmatch(raw_digest) is None:
            raise SemanticEmbedderError(
                "SEMANTIC_MANIFEST_INVALID", "semantic artifact digest is invalid"
            )
        files.append((name, raw_digest))
    if license_file not in files_payload:
        raise SemanticEmbedderError(
            "SEMANTIC_LICENSE_UNVERIFIED", "declared license file is not manifested"
        )
    manifested_names = set(files_payload)
    if not {"config.json", "tokenizer.json"}.issubset(manifested_names) or not any(
        name.endswith(".safetensors") for name in manifested_names
    ):
        raise SemanticEmbedderError(
            "SEMANTIC_MANIFEST_INVALID",
            "semantic artifact requires config, tokenizer, and safetensors weights",
        )
    if any(name.endswith((".bin", ".pt", ".pth", ".pkl")) for name in manifested_names):
        raise SemanticEmbedderError(
            "SEMANTIC_ARTIFACT_UNSAFE",
            "pickle-compatible semantic weights are not admitted",
        )

    observed = _inventory_regular_files(selected_root)
    observed_manifest = observed.get(SEMANTIC_EMBEDDER_MANIFEST)
    if observed_manifest is None or observed_manifest.identity != manifest_identity:
        raise SemanticEmbedderError(
            "SEMANTIC_ARTIFACT_CHANGED",
            "semantic manifest changed before artifact inventory completed",
        )
    observed_without_manifest = {
        name: observed_file
        for name, observed_file in observed.items()
        if name != SEMANTIC_EMBEDDER_MANIFEST
    }
    if set(observed_without_manifest) != set(files_payload):
        raise SemanticEmbedderError(
            "SEMANTIC_ARTIFACT_CLOSED_WORLD_FAILED",
            "semantic artifact inventory does not match its manifest",
        )
    total_bytes = 0
    for name, expected_digest in files:
        remaining = MAX_ARTIFACT_BYTES - total_bytes
        actual_digest, size = _hash_bounded_regular_file(
            selected_root,
            observed_without_manifest[name],
            maximum_bytes=remaining,
        )
        total_bytes += size
        if actual_digest != expected_digest:
            raise SemanticEmbedderError(
                "SEMANTIC_ARTIFACT_DIGEST_MISMATCH",
                "semantic artifact digest does not match its manifest",
            )

    config_observed = observed_without_manifest["config.json"]
    config_raw, _config_identity = _read_bounded_regular_file(
        config_observed.path,
        maximum_bytes=MAX_MODEL_CONFIG_BYTES,
        code="SEMANTIC_MODEL_CONFIG_UNSAFE",
        label="semantic model config",
        expected_identity=config_observed.identity,
    )
    if sha256(config_raw).hexdigest() != files_payload["config.json"]:
        raise SemanticEmbedderError(
            "SEMANTIC_ARTIFACT_CHANGED",
            "semantic model config changed after artifact hashing",
        )
    safe_config = _safe_model_config(config_raw, dimensions=dimensions)
    activation_upper_bound = max_batch_size * safe_config.max_tokens * dimensions * 16
    estimated_peak_ram_bytes = (
        total_bytes
        + safe_config.estimated_parameter_upper_bound * 8
        + activation_upper_bound
    )
    if estimated_peak_ram_bytes > MAX_SEMANTIC_ESTIMATED_PEAK_RAM_BYTES:
        raise SemanticEmbedderError(
            "SEMANTIC_MODEL_CONFIG_UNSAFE",
            "semantic model estimated peak CPU RAM exceeds its limit",
        )

    final_payload, final_raw, final_manifest_identity = _read_manifest(
        selected_manifest
    )
    if (
        final_raw != raw_manifest
        or final_payload != payload
        or final_manifest_identity != manifest_identity
    ):
        raise SemanticEmbedderError(
            "SEMANTIC_ARTIFACT_CHANGED",
            "semantic manifest changed during verification",
        )
    final_observed = _inventory_regular_files(selected_root)
    if {name: value.identity for name, value in final_observed.items()} != {
        name: value.identity for name, value in observed.items()
    }:
        raise SemanticEmbedderError(
            "SEMANTIC_ARTIFACT_CHANGED",
            "semantic artifact inventory changed during verification",
        )
    return SemanticEmbedderManifest(
        root=selected_root,
        path=selected_manifest,
        manifest_sha256=sha256(raw_manifest).hexdigest(),
        model_id=model_id,
        revision=revision,
        backend=backend,
        dimensions=dimensions,
        max_input_chars=max_input_chars,
        max_batch_size=max_batch_size,
        license_spdx=license_spdx,
        license_file=license_file,
        files=tuple(sorted(files)),
        total_artifact_bytes=total_bytes,
        model_type=safe_config.model_type,
        model_max_tokens=safe_config.max_tokens,
        estimated_parameter_upper_bound=(safe_config.estimated_parameter_upper_bound),
        estimated_peak_ram_bytes=estimated_peak_ram_bytes,
    )


class TransformersMeanPoolBackend:
    """CPU-only Hugging Face encoder with network and remote code disabled."""

    def __init__(self, manifest: SemanticEmbedderManifest) -> None:
        try:
            import torch
            from transformers import AutoModel, AutoTokenizer
        except ImportError as exc:
            raise SemanticEmbedderError(
                "SEMANTIC_RUNTIME_UNAVAILABLE",
                "local transformers embedding runtime is not installed",
            ) from exc
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                manifest.root,
                local_files_only=True,
                trust_remote_code=False,
            )
            model = AutoModel.from_pretrained(
                manifest.root,
                local_files_only=True,
                trust_remote_code=False,
            )
            model.to("cpu")
            model.eval()
        except Exception as exc:  # noqa: BLE001 - third-party local loader boundary
            raise SemanticEmbedderError(
                "SEMANTIC_MODEL_LOAD_FAILED",
                "verified semantic model could not be loaded locally",
            ) from exc
        if any(parameter.device.type != "cpu" for parameter in model.parameters()):
            raise SemanticEmbedderError(
                "SEMANTIC_DEVICE_POLICY_FAILED", "semantic model must remain on CPU"
            )
        self._torch = torch
        self._tokenizer = tokenizer
        self._model = model
        self._manifest = manifest

    def encode(self, texts: tuple[str, ...]) -> Sequence[Sequence[float]]:
        torch = self._torch
        try:
            encoded = self._tokenizer(
                list(texts),
                padding=True,
                truncation=True,
                max_length=self._manifest.model_max_tokens,
                return_tensors="pt",
            )
            cpu_encoded = {
                key: value.to("cpu")
                for key, value in encoded.items()
                if hasattr(value, "to")
            }
            with torch.inference_mode():
                output = self._model(**cpu_encoded)
            hidden = getattr(output, "last_hidden_state", None)
            mask = cpu_encoded.get("attention_mask")
            if hidden is None or mask is None or hidden.ndim != 3 or mask.ndim != 2:
                raise ValueError("model did not return a token embedding tensor")
            weights = mask.unsqueeze(-1).to(dtype=hidden.dtype)
            denominator = weights.sum(dim=1).clamp_min(1.0)
            pooled = (hidden * weights).sum(dim=1) / denominator
            pooled = torch.nn.functional.normalize(pooled.float(), p=2, dim=1)
            return pooled.cpu().tolist()
        except SemanticEmbedderError:
            raise
        except Exception as exc:  # noqa: BLE001 - third-party inference boundary
            raise SemanticEmbedderError(
                "SEMANTIC_INFERENCE_FAILED", "semantic embedding inference failed"
            ) from exc

    def close(self) -> None:
        self._model = None
        self._tokenizer = None


BackendFactory = Callable[[SemanticEmbedderManifest], SemanticEncoderBackend]


class LocalSemanticEmbedder:
    """Thread-safe verified local encoder that never grants answer authority."""

    def __init__(
        self,
        root: str | Path,
        *,
        manifest_path: str | Path | None = None,
        backend_factory: BackendFactory = TransformersMeanPoolBackend,
    ) -> None:
        self.manifest = verify_semantic_embedder_manifest(root, manifest_path)
        if not callable(backend_factory):
            raise TypeError("backend_factory must be callable")
        self._backend_factory = backend_factory
        self._backend: SemanticEncoderBackend | None = None
        self._lock = RLock()

    @property
    def loaded(self) -> bool:
        with self._lock:
            return self._backend is not None

    def status_payload(self) -> dict[str, object]:
        return self.manifest.status_payload(loaded=self.loaded)

    def load(self) -> None:
        with self._lock:
            if self._backend is not None:
                return
            # The constructor verification is not sufficient: the operator can
            # leave a session idle before its first use.  Re-hash the exact
            # closed-world snapshot immediately before and after the local
            # loader so a stale or load-time-mutated artifact is never promoted.
            before = verify_semantic_embedder_manifest(
                self.manifest.root, self.manifest.path
            )
            if before != self.manifest:
                raise SemanticEmbedderError(
                    "SEMANTIC_ARTIFACT_CHANGED",
                    "semantic artifact changed after admission",
                )
            try:
                backend = self._backend_factory(self.manifest)
            except SemanticEmbedderError:
                raise
            except Exception as exc:  # noqa: BLE001 - injected runtime boundary
                raise SemanticEmbedderError(
                    "SEMANTIC_MODEL_LOAD_FAILED",
                    "semantic embedding backend could not be initialized",
                ) from exc
            if not callable(getattr(backend, "encode", None)) or not callable(
                getattr(backend, "close", None)
            ):
                try:
                    close = getattr(backend, "close", None)
                    if callable(close):
                        close()
                except Exception:
                    pass
                raise SemanticEmbedderError(
                    "SEMANTIC_MODEL_LOAD_FAILED",
                    "semantic embedding backend contract is invalid",
                )
            try:
                after = verify_semantic_embedder_manifest(
                    self.manifest.root, self.manifest.path
                )
            except SemanticEmbedderError:
                try:
                    backend.close()
                except Exception:
                    pass
                raise
            if after != self.manifest:
                try:
                    backend.close()
                except Exception:
                    pass
                raise SemanticEmbedderError(
                    "SEMANTIC_ARTIFACT_CHANGED",
                    "semantic artifact changed during local model load",
                )
            self._backend = backend

    def unload(self) -> None:
        with self._lock:
            backend = self._backend
            self._backend = None
        if backend is not None:
            try:
                backend.close()
            except Exception as exc:  # noqa: BLE001 - backend cleanup boundary
                raise SemanticEmbedderError(
                    "SEMANTIC_MODEL_UNLOAD_FAILED",
                    "semantic embedding backend did not close cleanly",
                ) from exc

    def encode(self, texts: Sequence[str]) -> tuple[tuple[float, ...], ...]:
        if isinstance(texts, (str, bytes)) or not isinstance(texts, Sequence):
            raise SemanticEmbedderError(
                "SEMANTIC_INPUT_INVALID", "semantic input must be a text sequence"
            )
        try:
            batch_size = len(texts)
        except Exception as exc:  # noqa: BLE001 - caller-owned sequence boundary
            raise SemanticEmbedderError(
                "SEMANTIC_INPUT_INVALID", "semantic input length is unavailable"
            ) from exc
        if not 1 <= batch_size <= self.manifest.max_batch_size:
            raise SemanticEmbedderError(
                "SEMANTIC_BATCH_LIMIT", "semantic batch exceeds the manifest bound"
            )
        try:
            bounded = tuple(texts[index] for index in range(batch_size))
            stable_size = len(texts)
        except Exception as exc:  # noqa: BLE001 - caller-owned sequence boundary
            raise SemanticEmbedderError(
                "SEMANTIC_INPUT_INVALID", "semantic input changed while being copied"
            ) from exc
        if stable_size != batch_size or len(bounded) != batch_size:
            raise SemanticEmbedderError(
                "SEMANTIC_INPUT_INVALID", "semantic input changed while being copied"
            )
        for text in bounded:
            if (
                not isinstance(text, str)
                or not text.strip()
                or len(text) > self.manifest.max_input_chars
                or any(
                    ord(character) < 32 and character not in "\t\r\n"
                    for character in text
                )
            ):
                raise SemanticEmbedderError(
                    "SEMANTIC_INPUT_INVALID", "semantic input text is invalid"
                )
        self.load()
        with self._lock:
            backend = self._backend
            if backend is None:
                raise SemanticEmbedderError(
                    "SEMANTIC_RUNTIME_UNAVAILABLE", "semantic backend is not loaded"
                )
            try:
                raw = backend.encode(bounded)
            except SemanticEmbedderError:
                raise
            except Exception as exc:  # noqa: BLE001 - injected runtime boundary
                raise SemanticEmbedderError(
                    "SEMANTIC_INFERENCE_FAILED", "semantic embedding inference failed"
                ) from exc
        if not isinstance(raw, Sequence) or len(raw) != len(bounded):
            raise SemanticEmbedderError(
                "SEMANTIC_OUTPUT_INVALID", "semantic output batch shape is invalid"
            )
        normalized: list[tuple[float, ...]] = []
        for vector in raw:
            if (
                not isinstance(vector, Sequence)
                or len(vector) != self.manifest.dimensions
            ):
                raise SemanticEmbedderError(
                    "SEMANTIC_OUTPUT_INVALID", "semantic vector dimensions are invalid"
                )
            values: list[float] = []
            for value in vector:
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    raise SemanticEmbedderError(
                        "SEMANTIC_OUTPUT_INVALID", "semantic vector value is invalid"
                    )
                numeric = float(value)
                if not isfinite(numeric):
                    raise SemanticEmbedderError(
                        "SEMANTIC_OUTPUT_INVALID", "semantic vector must be finite"
                    )
                values.append(numeric)
            magnitude = sqrt(sum(value * value for value in values))
            if not isfinite(magnitude) or magnitude <= 1e-12:
                raise SemanticEmbedderError(
                    "SEMANTIC_OUTPUT_INVALID", "semantic vector norm is invalid"
                )
            normalized.append(tuple(value / magnitude for value in values))
        return tuple(normalized)


__all__ = [
    "LocalSemanticEmbedder",
    "MAX_ARTIFACT_BYTES",
    "MAX_ARTIFACT_DIRECTORIES",
    "MAX_ARTIFACT_FILES",
    "MAX_EMBEDDING_BATCH",
    "MAX_EMBEDDING_DIMENSIONS",
    "MAX_EMBEDDING_INPUT_CHARS",
    "MAX_MODEL_CONFIG_BYTES",
    "MAX_SEMANTIC_ESTIMATED_PEAK_RAM_BYTES",
    "MAX_SEMANTIC_MODEL_PARAMETERS",
    "SEMANTIC_EMBEDDER_MANIFEST",
    "SEMANTIC_EMBEDDER_SCHEMA",
    "SemanticEmbedderError",
    "SemanticEmbedderManifest",
    "TransformersMeanPoolBackend",
    "verify_semantic_embedder_manifest",
]
