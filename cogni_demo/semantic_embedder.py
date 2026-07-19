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
from typing import Callable, Protocol, Sequence


SEMANTIC_EMBEDDER_SCHEMA = "cogni.semantic-embedder.v1"
SEMANTIC_EMBEDDER_MANIFEST = "semantic-embedder.manifest.json"
TRANSFORMERS_MEAN_POOL_BACKEND = "transformers_mean_pool_v1"
MAX_MANIFEST_BYTES = 128 * 1024
MAX_ARTIFACT_FILES = 128
MAX_ARTIFACT_BYTES = 2 * 1024 * 1024 * 1024
MAX_EMBEDDING_DIMENSIONS = 8_192
MAX_EMBEDDING_BATCH = 16
MAX_EMBEDDING_INPUT_CHARS = 16_384
_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_MODEL_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._/-]{0,127}\Z")
_SPDX_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9.+-]{0,63}\Z")
_REVISION_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{6,127}\Z")
_WINDOWS_REPARSE_POINT = 0x0400


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


def _is_reparse_or_link(path: Path) -> bool:
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise SemanticEmbedderError(
            "SEMANTIC_ARTIFACT_UNAVAILABLE", "semantic artifact could not be inspected"
        ) from exc
    attributes = int(getattr(metadata, "st_file_attributes", 0))
    return stat.S_ISLNK(metadata.st_mode) or bool(attributes & _WINDOWS_REPARSE_POINT)


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


def _read_manifest(path: Path) -> tuple[dict[str, object], bytes]:
    try:
        if _is_reparse_or_link(path) or not path.is_file():
            raise SemanticEmbedderError(
                "SEMANTIC_MANIFEST_INVALID", "semantic manifest must be a regular file"
            )
        size = path.stat().st_size
        if not 1 <= size <= MAX_MANIFEST_BYTES:
            raise SemanticEmbedderError(
                "SEMANTIC_MANIFEST_INVALID", "semantic manifest size is invalid"
            )
        raw = path.read_bytes()
        payload = json.loads(raw.decode("utf-8"))
    except SemanticEmbedderError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise SemanticEmbedderError(
            "SEMANTIC_MANIFEST_INVALID", "semantic manifest could not be decoded"
        ) from exc
    if not isinstance(payload, dict):
        raise SemanticEmbedderError(
            "SEMANTIC_MANIFEST_INVALID", "semantic manifest must be an object"
        )
    return payload, raw


def _inventory_regular_files(root: Path) -> dict[str, Path]:
    observed: dict[str, Path] = {}
    pending = [root]
    while pending:
        directory = pending.pop()
        try:
            entries = tuple(os.scandir(directory))
        except OSError as exc:
            raise SemanticEmbedderError(
                "SEMANTIC_ARTIFACT_UNAVAILABLE",
                "semantic artifact directory could not be inventoried",
            ) from exc
        for entry in entries:
            candidate = Path(entry.path)
            if _is_reparse_or_link(candidate):
                raise SemanticEmbedderError(
                    "SEMANTIC_ARTIFACT_UNSAFE",
                    "semantic artifact cannot contain links or reparse points",
                )
            relative = candidate.relative_to(root).as_posix()
            try:
                if entry.is_dir(follow_symlinks=False):
                    pending.append(candidate)
                elif entry.is_file(follow_symlinks=False):
                    observed[relative] = candidate
                else:
                    raise SemanticEmbedderError(
                        "SEMANTIC_ARTIFACT_UNSAFE",
                        "semantic artifact contains an unsupported filesystem entry",
                    )
            except OSError as exc:
                raise SemanticEmbedderError(
                    "SEMANTIC_ARTIFACT_UNAVAILABLE",
                    "semantic artifact changed during inventory",
                ) from exc
    return observed


def verify_semantic_embedder_manifest(
    root: str | Path,
    manifest_path: str | Path | None = None,
) -> SemanticEmbedderManifest:
    """Verify a closed-world local embedding model without loading executable code."""

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

    payload, raw_manifest = _read_manifest(selected_manifest)
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
    observed_without_manifest = {
        name: path
        for name, path in observed.items()
        if name != SEMANTIC_EMBEDDER_MANIFEST
    }
    if set(observed_without_manifest) != set(files_payload):
        raise SemanticEmbedderError(
            "SEMANTIC_ARTIFACT_CLOSED_WORLD_FAILED",
            "semantic artifact inventory does not match its manifest",
        )
    total_bytes = 0
    for name, expected_digest in files:
        candidate = observed_without_manifest[name]
        try:
            size = candidate.stat().st_size
        except OSError as exc:
            raise SemanticEmbedderError(
                "SEMANTIC_ARTIFACT_UNAVAILABLE",
                "semantic artifact changed during hashing",
            ) from exc
        if size <= 0:
            raise SemanticEmbedderError(
                "SEMANTIC_ARTIFACT_INVALID", "semantic artifact file is empty"
            )
        total_bytes += size
        if total_bytes > MAX_ARTIFACT_BYTES:
            raise SemanticEmbedderError(
                "SEMANTIC_ARTIFACT_TOO_LARGE",
                "semantic artifact exceeds the byte limit",
            )
        digest = sha256()
        try:
            with candidate.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(chunk)
        except OSError as exc:
            raise SemanticEmbedderError(
                "SEMANTIC_ARTIFACT_UNAVAILABLE", "semantic artifact could not be hashed"
            ) from exc
        if digest.hexdigest() != expected_digest:
            raise SemanticEmbedderError(
                "SEMANTIC_ARTIFACT_DIGEST_MISMATCH",
                "semantic artifact digest does not match its manifest",
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
                max_length=min(4096, self._manifest.max_input_chars),
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
        bounded = tuple(texts)
        if not 1 <= len(bounded) <= self.manifest.max_batch_size:
            raise SemanticEmbedderError(
                "SEMANTIC_BATCH_LIMIT", "semantic batch exceeds the manifest bound"
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
    "MAX_EMBEDDING_BATCH",
    "MAX_EMBEDDING_DIMENSIONS",
    "MAX_EMBEDDING_INPUT_CHARS",
    "SEMANTIC_EMBEDDER_MANIFEST",
    "SEMANTIC_EMBEDDER_SCHEMA",
    "SemanticEmbedderError",
    "SemanticEmbedderManifest",
    "TransformersMeanPoolBackend",
    "verify_semantic_embedder_manifest",
]
