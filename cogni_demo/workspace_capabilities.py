"""Fail-closed local workspace capabilities for CogniBoard.

This module deliberately separates *artifact storage* from *model input
authority*.  Accepting an image attachment does not imply that the current
text-only product pipeline can pass it to Gemma.  Likewise, indexing local
text in AkasicDB does not make retrieved chunks answer-bearing until a later
prompt-injection-safe evidence bridge is implemented and verified.

AkasicDB retrieval can become answer-bearing only when the server injects
validated ``RetrievalEvidence`` into the agent prompt.  The service therefore
publishes that state through an explicit construction flag rather than
inferring it from the mere presence of an index.

AkasicDB remains an external component.  Its public repository currently has
an MIT badge in README.md but no LICENSE file, so no source is copied or
vendored here.  The adapter loads only a pinned, hash-verified local clone and
uses its actual GraphStore, RelationalStore, and VectorStore interfaces.  It
never imports or starts AkasicDB's network-facing demo server.
"""

from __future__ import annotations

from base64 import b64decode
from binascii import Error as Base64Error
from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
from importlib.util import module_from_spec, spec_from_file_location
import json
from math import sqrt
import os
from pathlib import Path
import re
import secrets
import stat
from threading import RLock
from types import ModuleType
from typing import Any, Mapping
from urllib.parse import urlsplit

from cogni_os.factbook import RuntimeFactBook


MAX_ATTACHMENT_BYTES = 8 * 1024 * 1024
MAX_ATTACHMENT_TOTAL_BYTES = 64 * 1024 * 1024
MAX_ATTACHMENT_BASE64_CHARS = 4 * ((MAX_ATTACHMENT_BYTES + 2) // 3)
MAX_ATTACHMENT_COUNT = 32
MAX_ATTACHMENT_NAME_CHARS = 128
MAX_JSON_ATTACHMENT_BYTES = 1024 * 1024
MAX_JSON_NESTING = 64
MAX_INDEXED_TEXT_CHARS = 256_000
MAX_RAG_CHUNKS_PER_DOCUMENT = 128
MAX_RAG_QUERY_CHARS = 1_024
MAX_RAG_RESULTS = 12
RAG_VECTOR_DIMENSIONS = 256
RAG_CHUNK_CHARS = 1_600
RAG_CHUNK_OVERLAP_CHARS = 200
MAX_WEB_ALLOWLIST_HOSTS = 32

AKASICDB_REPOSITORY = "https://github.com/heosanghun/AkasicDB.git"
AKASICDB_AUDITED_REVISION = "a6c8e8ebd487e7cb86079f9804a66aaf0914d1dc"
AKASICDB_AUDITED_DIGESTS: dict[str, str] = {
    "akasic/storage/graph_store.py": (
        "fad8977c1be2269f78670e8a9d9b41e0ae6751cc3b874d5f82f95ff639f21ce4"
    ),
    "akasic/storage/relational_store.py": (
        "1a66bb519244cbfc759848fbcac7d4584dce60c746ae0971372f4929f832daf3"
    ),
    "akasic/storage/vector_store.py": (
        "2fbba721966fc0ad90b3a9b315e16b04d5f798e12ab66f5c1b60fcf2934ef939"
    ),
}

_TEXT_MEDIA_TYPES = {
    ".txt": "text/plain",
    ".md": "text/markdown",
    ".csv": "text/csv",
    ".json": "application/json",
}
_BINARY_MEDIA_TYPES = {
    ".pdf": "application/pdf",
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
}
_ALLOWED_MEDIA_TYPES = {**_TEXT_MEDIA_TYPES, **_BINARY_MEDIA_TYPES}
_TOKEN_RE = re.compile(r"[0-9a-zA-Z_\u3131-\u318e\uac00-\ud7a3]{2,}")
_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_BLOB_NAME_RE = re.compile(r"(?P<digest>[0-9a-f]{24})(?P<suffix>\.[a-z0-9]+)\Z")


class WorkspaceCapabilityError(ValueError):
    """One bounded workspace request failed before crossing its authority."""

    def __init__(self, code: str, message: str) -> None:
        if (
            not isinstance(code, str)
            or re.fullmatch(r"[A-Z][A-Z0-9_]{0,63}", code) is None
        ):
            raise ValueError("workspace error code is invalid")
        if not isinstance(message, str) or not 1 <= len(message) <= 512:
            raise ValueError("workspace error message is invalid")
        super().__init__(message)
        self.code = code


@dataclass(frozen=True, slots=True)
class AttachmentRecord:
    attachment_id: str
    name: str
    media_type: str
    size_bytes: int
    sha256: str
    stored_path: Path
    created_at: str
    text_indexable: bool

    def as_payload(self) -> dict[str, object]:
        return {
            "attachment_id": self.attachment_id,
            "name": self.name,
            "media_type": self.media_type,
            "size_bytes": self.size_bytes,
            "sha256": self.sha256,
            "created_at": self.created_at,
            "text_indexable": self.text_indexable,
            # Storage is local, but no absolute host path is exposed to the UI.
            "storage": "local_content_addressed",
        }


@dataclass(frozen=True, slots=True)
class VerifiedModelMetadata:
    model_id: str
    label: str
    architecture: str
    manifest_sha256: str
    config_sha256: str
    checkpoint_modalities: tuple[str, ...]
    runtime_input_modalities: tuple[str, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.model_id, str) or not self.model_id:
            raise ValueError("model_id must be non-empty text")
        if not isinstance(self.label, str) or not self.label:
            raise ValueError("label must be non-empty text")
        if not isinstance(self.architecture, str) or not self.architecture:
            raise ValueError("architecture must be non-empty text")
        _require_sha256(self.manifest_sha256, "manifest_sha256")
        _require_sha256(self.config_sha256, "config_sha256")
        allowed = {"text", "image", "audio", "video"}
        if (
            not self.checkpoint_modalities
            or any(item not in allowed for item in self.checkpoint_modalities)
            or any(item not in allowed for item in self.runtime_input_modalities)
            or not set(self.runtime_input_modalities).issubset(
                self.checkpoint_modalities
            )
        ):
            raise ValueError("model modalities are invalid")

    def as_payload(self) -> dict[str, object]:
        unavailable = sorted(
            set(self.checkpoint_modalities) - set(self.runtime_input_modalities)
        )
        return {
            "model_id": self.model_id,
            "label": self.label,
            "architecture": self.architecture,
            "verification": "runtime_factbook_and_config_digest",
            "manifest_sha256": self.manifest_sha256,
            "config_sha256": self.config_sha256,
            "checkpoint_modalities": list(self.checkpoint_modalities),
            "runtime_input_modalities": list(self.runtime_input_modalities),
            "advertised_but_not_wired": unavailable,
            "selected": True,
        }


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_sha256(value: object, field: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{field} must be a lowercase SHA-256 digest")
    return value


def _prepare_storage_root(project_root: Path) -> Path:
    """Create the bounded attachment root without following directory links."""

    current = project_root
    for name in ("outputs", "agent-workspace", "attachments"):
        candidate = current / name
        if candidate.exists() or candidate.is_symlink():
            if candidate.is_symlink() or not candidate.is_dir():
                raise WorkspaceCapabilityError(
                    "ATTACHMENT_STORAGE_UNSAFE",
                    "attachment storage contains an unsafe path component",
                )
        else:
            candidate.mkdir(mode=0o700)
        try:
            resolved = candidate.resolve(strict=True)
        except OSError as exc:
            raise WorkspaceCapabilityError(
                "ATTACHMENT_STORAGE_UNSAFE",
                "attachment storage could not be verified",
            ) from exc
        if not resolved.is_relative_to(project_root):
            raise WorkspaceCapabilityError(
                "ATTACHMENT_STORAGE_UNSAFE",
                "attachment storage escaped the project root",
            )
        current = resolved
    return current


def _inventory_storage_root(storage_root: Path) -> tuple[frozenset[Path], int, int]:
    """Inventory persisted blobs so process restarts cannot reset disk quotas.

    The attachment catalog intentionally remains process-local because the
    original filename and admission metadata are not persisted.  Disk quota
    accounting is different: every pre-existing entry consumes both count and
    byte budget, while only content-addressed regular files whose digest prefix
    still matches are eligible for same-blob reuse.
    """

    valid: set[Path] = set()
    entry_count = 0
    total_bytes = 0
    for candidate in storage_root.iterdir():
        entry_count += 1
        if candidate.is_symlink() or not candidate.is_file():
            raise WorkspaceCapabilityError(
                "ATTACHMENT_STORAGE_UNSAFE",
                "attachment storage contains a non-regular entry",
            )
        try:
            size = candidate.stat().st_size
        except OSError as exc:
            raise WorkspaceCapabilityError(
                "ATTACHMENT_STORAGE_UNSAFE",
                "attachment storage inventory could not be read",
            ) from exc
        total_bytes += max(0, size)
        matched = _BLOB_NAME_RE.fullmatch(candidate.name)
        if (
            matched is None
            or matched.group("suffix") not in _ALLOWED_MEDIA_TYPES
            or not 1 <= size <= MAX_ATTACHMENT_BYTES
        ):
            continue
        try:
            digest = _sha256_file(candidate)
        except OSError:
            continue
        if digest.startswith(matched.group("digest")):
            valid.add(candidate.resolve(strict=True))
    return frozenset(valid), entry_count, total_bytes


def _json_nesting_is_bounded(text: str) -> bool:
    """Reject deeply nested JSON before handing it to the recursive decoder."""

    depth = 0
    in_string = False
    escaped = False
    for character in text:
        if in_string:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == '"':
                in_string = False
            continue
        if character == '"':
            in_string = True
        elif character in "[{":
            depth += 1
            if depth > MAX_JSON_NESTING:
                return False
        elif character in "]}":
            depth -= 1
            if depth < 0:
                return False
    return depth == 0 and not in_string


def _atomic_content_write(path: Path, content: bytes, expected_digest: str) -> None:
    """Publish one same-directory blob atomically and verify the final inode."""

    _require_sha256(expected_digest, "expected_digest")
    parent = path.parent.resolve(strict=True)
    if path.parent.is_symlink() or not path.resolve(strict=False).is_relative_to(
        parent
    ):
        raise WorkspaceCapabilityError(
            "ATTACHMENT_STORAGE_CONFLICT", "attachment destination is unsafe"
        )
    temporary = parent / (f".{path.name}.{os.getpid()}.{secrets.token_hex(8)}.tmp")
    descriptor = -1
    published = False
    try:
        descriptor = os.open(
            temporary,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            stat.S_IRUSR | stat.S_IWUSR,
        )
        with os.fdopen(descriptor, "wb") as stream:
            descriptor = -1
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        if path.exists() or path.is_symlink():
            if (
                path.is_symlink()
                or not path.is_file()
                or _sha256_file(path) != expected_digest
            ):
                raise WorkspaceCapabilityError(
                    "ATTACHMENT_STORAGE_CONFLICT",
                    "content-addressed attachment path is occupied",
                )
        else:
            os.replace(temporary, path)
            published = True
        if (
            path.is_symlink()
            or not path.resolve(strict=True).is_relative_to(parent)
            or _sha256_file(path) != expected_digest
        ):
            raise WorkspaceCapabilityError(
                "ATTACHMENT_STORAGE_CONFLICT",
                "published attachment failed its integrity check",
            )
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        if published and not path.exists():
            raise WorkspaceCapabilityError(
                "ATTACHMENT_STORAGE_CONFLICT", "attachment publication was lost"
            )


def _safe_clone_head(root: Path) -> str | None:
    """Read a non-worktree Git HEAD without invoking a shell or Git hooks."""

    git = root / ".git"
    if not git.is_dir() or git.is_symlink():
        return None
    head_path = git / "HEAD"
    try:
        raw = head_path.read_text(encoding="ascii").strip()
    except (OSError, UnicodeError):
        return None
    if re.fullmatch(r"[0-9a-f]{40}", raw):
        return raw
    if not raw.startswith("ref: "):
        return None
    reference = raw[5:]
    if (
        not reference.startswith("refs/")
        or ".." in Path(reference).parts
        or Path(reference).is_absolute()
    ):
        return None
    ref_path = git / Path(reference)
    try:
        value = ref_path.read_text(encoding="ascii").strip()
    except (OSError, UnicodeError):
        value = ""
    if re.fullmatch(r"[0-9a-f]{40}", value):
        return value
    try:
        packed = (git / "packed-refs").read_text(encoding="ascii")
    except (OSError, UnicodeError):
        return None
    for line in packed.splitlines():
        if line.startswith(("#", "^")):
            continue
        commit, separator, name = line.partition(" ")
        if separator and name == reference and re.fullmatch(r"[0-9a-f]{40}", commit):
            return commit
    return None


def _load_verified_class(path: Path, class_name: str, digest: str) -> type[Any]:
    _require_sha256(digest, "audited module digest")
    module_name = f"_cogni_akasic_{path.stem}_{digest[:12]}"
    spec = spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise WorkspaceCapabilityError(
            "AKASICDB_IMPORT_FAILED", f"cannot load audited module: {path.name}"
        )
    module: ModuleType = module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception as exc:
        raise WorkspaceCapabilityError(
            "AKASICDB_IMPORT_FAILED", f"audited module failed: {path.name}"
        ) from exc
    selected = getattr(module, class_name, None)
    if not isinstance(selected, type):
        raise WorkspaceCapabilityError(
            "AKASICDB_API_MISMATCH", f"missing audited class: {class_name}"
        )
    return selected


def _stable_sha256_embedding(text: str) -> list[float]:
    """Return a deterministic non-negative lexical sketch.

    A signed hashing trick can cancel two legitimate query terms that land in
    the same bucket.  This bounded local retriever already requires an exact
    lexical overlap before publishing a result, so non-negative counts are the
    safer fail-closed representation here.
    """

    if not isinstance(text, str):
        raise TypeError("embedding input must be text")
    vector = [0.0] * RAG_VECTOR_DIMENSIONS
    tokens = _TOKEN_RE.findall(text.casefold())
    for token in tokens:
        digest = sha256(token.encode("utf-8")).digest()
        index = int.from_bytes(digest[:4], "big") % RAG_VECTOR_DIMENSIONS
        vector[index] += 1.0
    magnitude = sqrt(sum(value * value for value in vector))
    if magnitude:
        return [value / magnitude for value in vector]
    return vector


def _chunk_text(text: str) -> tuple[str, ...]:
    normalized = "\n".join(line.rstrip() for line in text.splitlines()).strip()
    if not normalized:
        raise WorkspaceCapabilityError("EMPTY_DOCUMENT", "document has no text")
    if len(normalized) > MAX_INDEXED_TEXT_CHARS:
        raise WorkspaceCapabilityError(
            "DOCUMENT_TOO_LARGE", "document exceeds the local index character limit"
        )
    chunks: list[str] = []
    cursor = 0
    while cursor < len(normalized):
        end = min(len(normalized), cursor + RAG_CHUNK_CHARS)
        if end < len(normalized):
            boundary = normalized.rfind("\n", cursor + 1, end)
            if boundary <= cursor:
                boundary = normalized.rfind(" ", cursor + 1, end)
            if boundary > cursor + RAG_CHUNK_CHARS // 2:
                end = boundary
        chunk = normalized[cursor:end].strip()
        if chunk:
            chunks.append(chunk)
        if len(chunks) > MAX_RAG_CHUNKS_PER_DOCUMENT:
            raise WorkspaceCapabilityError(
                "TOO_MANY_CHUNKS", "document exceeds the local index chunk limit"
            )
        if end >= len(normalized):
            break
        cursor = max(cursor + 1, end - RAG_CHUNK_OVERLAP_CHARS)
    return tuple(chunks)


class AkasicDBAdapter:
    """Pinned adapter over AkasicDB's actual in-memory storage/search classes."""

    def __init__(self, clone_path: str | Path) -> None:
        root = Path(clone_path).expanduser().resolve(strict=True)
        if not root.is_dir() or root.is_symlink():
            raise WorkspaceCapabilityError(
                "AKASICDB_INVALID_CLONE", "AkasicDB path must be a local directory"
            )
        revision = _safe_clone_head(root)
        if revision != AKASICDB_AUDITED_REVISION:
            raise WorkspaceCapabilityError(
                "AKASICDB_REVISION_MISMATCH",
                "AkasicDB clone is not at the audited revision",
            )
        verified_paths: dict[str, Path] = {}
        for relative, expected in AKASICDB_AUDITED_DIGESTS.items():
            try:
                _require_sha256(expected, f"AkasicDB digest for {relative}")
            except ValueError as exc:
                raise WorkspaceCapabilityError(
                    "AKASICDB_DIGEST_INVALID", "an audited digest is malformed"
                ) from exc
            candidate = root / Path(relative)
            try:
                resolved = candidate.resolve(strict=True)
            except FileNotFoundError as exc:
                raise WorkspaceCapabilityError(
                    "AKASICDB_FILE_MISSING", f"missing audited file: {relative}"
                ) from exc
            if (
                candidate.is_symlink()
                or not resolved.is_relative_to(root)
                or not resolved.is_file()
                or _sha256_file(resolved) != expected
            ):
                raise WorkspaceCapabilityError(
                    "AKASICDB_DIGEST_MISMATCH",
                    f"AkasicDB audited file mismatch: {relative}",
                )
            verified_paths[relative] = resolved

        graph_type = _load_verified_class(
            verified_paths["akasic/storage/graph_store.py"],
            "GraphStore",
            AKASICDB_AUDITED_DIGESTS["akasic/storage/graph_store.py"],
        )
        relational_type = _load_verified_class(
            verified_paths["akasic/storage/relational_store.py"],
            "RelationalStore",
            AKASICDB_AUDITED_DIGESTS["akasic/storage/relational_store.py"],
        )
        vector_type = _load_verified_class(
            verified_paths["akasic/storage/vector_store.py"],
            "VectorStore",
            AKASICDB_AUDITED_DIGESTS["akasic/storage/vector_store.py"],
        )
        self.root = root
        self.revision = revision
        self.graph_store = graph_type()
        self.relational_store = relational_type()
        self.vector_store = vector_type()
        self._chunk_ids: set[str] = set()
        self._indexed_attachments: set[str] = set()
        self._lock = RLock()

    @property
    def chunk_count(self) -> int:
        with self._lock:
            return len(self._chunk_ids)

    @property
    def document_count(self) -> int:
        with self._lock:
            return len(self._indexed_attachments)

    def status_payload(self) -> dict[str, object]:
        license_file = any(
            (self.root / name).is_file()
            for name in ("LICENSE", "LICENSE.txt", "LICENSE.md", "COPYING")
        )
        return {
            "engine": "AkasicDB",
            "repository": AKASICDB_REPOSITORY,
            "revision": self.revision,
            "integration": "external_pinned_core_adapter",
            "storage_lifetime": "process_memory",
            "embedding": "stable_sha256_lexical_sketch_v1",
            "documents": self.document_count,
            "chunks": self.chunk_count,
            "license_claim": "MIT_README_BADGE",
            "license_file_present": license_file,
            "source_vendored": False,
        }

    def index_document(
        self,
        *,
        attachment_id: str,
        name: str,
        media_type: str,
        text: str,
    ) -> dict[str, object]:
        chunks = _chunk_text(text)
        document_entity = f"document:{attachment_id}"
        with self._lock:
            if attachment_id in self._indexed_attachments:
                return {
                    "attachment_id": attachment_id,
                    "indexed": False,
                    "already_indexed": True,
                    "chunks": 0,
                }
            for index, chunk in enumerate(chunks):
                entity = f"chunk:{attachment_id}:{index}"
                self.graph_store.add_edge(document_entity, entity, "contains")
                self.relational_store.insert(
                    entity,
                    {
                        "type": "LocalDocumentChunk",
                        "attachment_id": attachment_id,
                        "name": name,
                        "media_type": media_type,
                        "chunk_index": index,
                        "chunk": chunk,
                    },
                )
                self.vector_store.insert(entity, _stable_sha256_embedding(chunk))
                self._chunk_ids.add(entity)
            self._indexed_attachments.add(attachment_id)
        return {
            "attachment_id": attachment_id,
            "indexed": True,
            "already_indexed": False,
            "chunks": len(chunks),
        }

    def query(self, query: str, *, limit: int = 5) -> dict[str, object]:
        if not isinstance(query, str) or not query.strip():
            raise WorkspaceCapabilityError("INVALID_QUERY", "query must contain text")
        if len(query) > MAX_RAG_QUERY_CHARS:
            raise WorkspaceCapabilityError(
                "QUERY_TOO_LARGE", "query exceeds the local RAG character limit"
            )
        if (
            not isinstance(limit, int)
            or isinstance(limit, bool)
            or not 1 <= limit <= MAX_RAG_RESULTS
        ):
            raise WorkspaceCapabilityError(
                "INVALID_LIMIT", f"limit must be in [1, {MAX_RAG_RESULTS}]"
            )
        vector = _stable_sha256_embedding(query)
        query_tokens = set(_TOKEN_RE.findall(query.casefold()))
        if not any(vector):
            raise WorkspaceCapabilityError(
                "INVALID_QUERY", "query has no indexable terms"
            )
        with self._lock:
            if not self._chunk_ids:
                return {"query": query, "results": [], "count": 0}
            # This calls AkasicDB's actual public search interface.  The store is
            # private to this adapter, so every returned entity is ours.
            matches = self.vector_store.similarity_search(
                vector, top_k=len(self._chunk_ids)
            )
            results: list[dict[str, object]] = []
            for entity, score in matches:
                if entity not in self._chunk_ids:
                    continue
                numeric_score = float(score)
                if numeric_score <= 0.0:
                    continue
                record = self.relational_store.get(entity)
                chunk_text = record.get("chunk", "")
                if not isinstance(chunk_text, str) or not (
                    query_tokens & set(_TOKEN_RE.findall(chunk_text.casefold()))
                ):
                    continue
                results.append(
                    {
                        "attachment_id": record.get("attachment_id"),
                        "name": record.get("name"),
                        "media_type": record.get("media_type"),
                        "chunk_index": record.get("chunk_index"),
                        "text": chunk_text,
                        "score": round(numeric_score, 8),
                    }
                )
                if len(results) >= limit:
                    break
        return {"query": query, "results": results, "count": len(results)}


class WebAccessPolicy:
    """Authorization boundary only; this class never performs a network call."""

    def __init__(
        self,
        *,
        online_mode: bool = False,
        allowlist: tuple[str, ...] = (),
        lens_api_token_configured: bool = False,
    ) -> None:
        if not isinstance(online_mode, bool):
            raise TypeError("online_mode must be bool")
        if not isinstance(lens_api_token_configured, bool):
            raise TypeError("lens_api_token_configured must be bool")
        if len(allowlist) > MAX_WEB_ALLOWLIST_HOSTS:
            raise ValueError("web allowlist is too large")
        normalized: list[str] = []
        for host in allowlist:
            if not isinstance(host, str) or not re.fullmatch(
                r"(?=.{1,253}\Z)(?:[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.)+"
                r"[a-z]{2,63}",
                host.casefold(),
            ):
                raise ValueError("web allowlist contains an invalid hostname")
            normalized.append(host.casefold())
        self.online_mode = online_mode
        self.allowlist = tuple(sorted(set(normalized)))
        self.lens_api_token_configured = lens_api_token_configured

    def authorize_url(self, url: str) -> dict[str, object]:
        if not self.online_mode:
            raise WorkspaceCapabilityError(
                "AIR_GAP_BLOCKED", "online mode is not explicitly enabled"
            )
        if not isinstance(url, str) or len(url) > 2_048:
            raise WorkspaceCapabilityError("INVALID_URL", "URL is invalid")
        parsed = urlsplit(url)
        host = (parsed.hostname or "").casefold()
        try:
            port = parsed.port
        except ValueError as exc:
            raise WorkspaceCapabilityError(
                "INVALID_URL", "URL port is invalid"
            ) from exc
        if (
            parsed.scheme != "https"
            or not host
            or parsed.username is not None
            or parsed.password is not None
            or port not in (None, 443)
            or parsed.fragment
        ):
            raise WorkspaceCapabilityError(
                "WEB_POLICY_BLOCKED", "only credential-free HTTPS URLs are eligible"
            )
        if host not in self.allowlist:
            raise WorkspaceCapabilityError(
                "HOST_NOT_ALLOWLISTED", "URL host is not explicitly allowlisted"
            )
        return {
            "authorized": True,
            "host": host,
            "network_called": False,
            "boundary": "authorization_only",
        }

    def as_payload(self) -> dict[str, object]:
        if not self.online_mode:
            generic_state = "disabled_air_gap"
        elif not self.allowlist:
            generic_state = "disabled_empty_allowlist"
        else:
            generic_state = "authorization_ready"
        if not self.online_mode:
            lens_state = "disabled_air_gap"
        elif "api.lens.org" not in self.allowlist:
            lens_state = "allowlist_required"
        elif not self.lens_api_token_configured:
            lens_state = "credentials_required"
        else:
            lens_state = "authorization_ready"
        lens_common = {
            "provider": "Lens.org official API",
            "host": "api.lens.org",
            "method": "POST",
            "token_configured": self.lens_api_token_configured,
            "network_executor_implemented": False,
            "scraping_allowed": False,
        }
        return {
            "mode": "online_opt_in" if self.online_mode else "air_gapped",
            "state": generic_state,
            "allowlist": list(self.allowlist),
            "execution": "authorization_only",
            "executor_implemented": False,
            "lens_patent_search": {
                **lens_common,
                "state": lens_state,
                "endpoint": "/patent/search",
            },
            "lens_scholarly_search": {
                **lens_common,
                "state": lens_state,
                "endpoint": "/scholarly/search",
            },
        }


class WorkspaceCapabilityService:
    """Own local attachments, AkasicDB retrieval, and honest UI metadata."""

    def __init__(
        self,
        project_root: str | Path,
        model: VerifiedModelMetadata,
        *,
        akasicdb_path: str | Path | None = None,
        web_policy: WebAccessPolicy | None = None,
        answer_integration_enabled: bool = False,
    ) -> None:
        root = Path(project_root).resolve(strict=True)
        if not root.is_dir():
            raise ValueError("project_root must be a directory")
        if not isinstance(model, VerifiedModelMetadata):
            raise TypeError("model must be VerifiedModelMetadata")
        self.project_root = root
        self.model = model
        self.web_policy = web_policy or WebAccessPolicy()
        if not isinstance(answer_integration_enabled, bool):
            raise TypeError("answer_integration_enabled must be bool")
        self.answer_integration_enabled = answer_integration_enabled
        self.storage_root = _prepare_storage_root(root)
        (
            self._persisted_valid_blobs,
            self._persisted_blob_count,
            self._persisted_blob_bytes,
        ) = _inventory_storage_root(self.storage_root)
        self._attachments: dict[str, AttachmentRecord] = {}
        self._lock = RLock()
        self.akasicdb: AkasicDBAdapter | None = None
        self.akasicdb_error: tuple[str, str] | None = None
        selected_path = self._discover_akasicdb(akasicdb_path)
        if selected_path is None:
            self.akasicdb_error = (
                "AKASICDB_NOT_CONFIGURED",
                "set COGNI_OS_AKASICDB_DIR to an audited local clone",
            )
        else:
            try:
                self.akasicdb = AkasicDBAdapter(selected_path)
            except WorkspaceCapabilityError as exc:
                self.akasicdb_error = (exc.code, str(exc)[:256])
            except OSError:
                self.akasicdb_error = (
                    "AKASICDB_UNAVAILABLE",
                    "the configured AkasicDB clone could not be opened",
                )

    @staticmethod
    def _discover_akasicdb(explicit: str | Path | None) -> Path | None:
        raw: str | Path | None = explicit
        if raw is None:
            raw = os.environ.get("COGNI_OS_AKASICDB_DIR")
        if raw is None and os.name == "nt":
            candidate = Path(r"C:\Project\AkasicDB")
            if candidate.is_dir():
                raw = candidate
        if raw is None:
            return None
        return Path(raw)

    @classmethod
    def from_runtime_factbook(
        cls,
        project_root: str | Path,
        model_root: str | Path,
        manifest_path: str | Path,
        factbook: RuntimeFactBook,
        **kwargs: Any,
    ) -> WorkspaceCapabilityService:
        """Bind lightweight model metadata to startup-verified Fact-book digests."""

        if not isinstance(factbook, RuntimeFactBook):
            raise TypeError("factbook must be RuntimeFactBook")
        root = Path(model_root).expanduser().resolve(strict=True)
        manifest = Path(manifest_path).expanduser().resolve(strict=True)
        config = (root / "config.json").resolve(strict=True)
        if (
            not config.is_relative_to(root)
            or not config.is_file()
            or config.is_symlink()
            or _sha256_file(config) != factbook.model.config_sha256
            or _sha256_file(manifest) != factbook.model.manifest_sha256
        ):
            raise WorkspaceCapabilityError(
                "MODEL_METADATA_MISMATCH",
                "model metadata no longer matches the verified Runtime Fact-book",
            )
        if config.stat().st_size > 1024 * 1024:
            raise WorkspaceCapabilityError(
                "MODEL_CONFIG_TOO_LARGE", "model config exceeds metadata bound"
            )
        try:
            payload = json.loads(config.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise WorkspaceCapabilityError(
                "MODEL_CONFIG_INVALID", "model config is invalid"
            ) from exc
        if not isinstance(payload, dict):
            raise WorkspaceCapabilityError(
                "MODEL_CONFIG_INVALID", "model config must be an object"
            )
        modalities = ["text"]
        if isinstance(payload.get("vision_config"), dict):
            if isinstance(payload.get("image_token_id"), int):
                modalities.append("image")
            if isinstance(payload.get("video_token_id"), int):
                modalities.append("video")
        if isinstance(payload.get("audio_config"), dict) and isinstance(
            payload.get("audio_token_id"), int
        ):
            modalities.append("audio")
        model = VerifiedModelMetadata(
            model_id=(f"{factbook.model.label}:{factbook.model.manifest_sha256[:12]}"),
            label=factbook.model.label,
            architecture=factbook.model.architecture,
            manifest_sha256=factbook.model.manifest_sha256,
            config_sha256=factbook.model.config_sha256,
            checkpoint_modalities=tuple(modalities),
            # The current ModelService and AgentManager accept text prompts only.
            runtime_input_modalities=("text",),
        )
        return cls(project_root, model, **kwargs)

    def capability_payload(self) -> dict[str, object]:
        rag: dict[str, object]
        if self.akasicdb is None:
            assert self.akasicdb_error is not None
            rag = {
                "state": "unavailable",
                "answer_integration": False,
                "error": {
                    "code": self.akasicdb_error[0],
                    "message": self.akasicdb_error[1],
                },
                "repository": AKASICDB_REPOSITORY,
                "required_revision": AKASICDB_AUDITED_REVISION,
            }
        else:
            rag = {
                "state": "local_index_ready",
                "answer_integration": self.answer_integration_enabled,
                "query_api": True,
                **self.akasicdb.status_payload(),
            }
        return {
            "schema_version": 1,
            "attachments": {
                "state": "enabled",
                "count": len(self._attachments),
                "max_count": MAX_ATTACHMENT_COUNT,
                "max_bytes_each": MAX_ATTACHMENT_BYTES,
                "max_total_bytes": MAX_ATTACHMENT_TOTAL_BYTES,
                "persisted_blob_count": self._persisted_blob_count,
                "persisted_blob_bytes": self._persisted_blob_bytes,
                "accepted_media_types": sorted(set(_ALLOWED_MEDIA_TYPES.values())),
                "image_to_model_integration": False,
                "pdf_text_extraction": False,
                "catalog_lifetime": "process_memory",
                "blob_lifetime": "local_filesystem_until_manual_cleanup",
            },
            "rag": rag,
            "models": {
                "state": "single_verified_model_only",
                "switching": "idempotent_current_model_only",
                "items": [self.model.as_payload()],
            },
            "microphone": {
                "capture_state": "frontend_not_connected",
                "transcription_state": "disabled",
                "checkpoint_advertises_audio": (
                    "audio" in self.model.checkpoint_modalities
                ),
                "runtime_audio_input": False,
                "required_before_enablement": "verified local STT and audio pipeline",
            },
            "web_search": self.web_policy.as_payload(),
        }

    def list_attachments(self) -> dict[str, object]:
        with self._lock:
            items = [
                record.as_payload()
                for record in sorted(
                    self._attachments.values(), key=lambda item: item.created_at
                )
            ]
        return {"items": items, "count": len(items)}

    def add_attachment(
        self, *, name: str, media_type: str, content_base64: str
    ) -> dict[str, object]:
        if (
            not isinstance(name, str)
            or not 1 <= len(name) <= MAX_ATTACHMENT_NAME_CHARS
            or Path(name).name != name
            or name in {".", ".."}
            or any(
                ord(character) < 32 or 0x7F <= ord(character) <= 0x9F
                for character in name
            )
        ):
            raise WorkspaceCapabilityError(
                "INVALID_ATTACHMENT_NAME", "attachment name is unsafe"
            )
        suffix = Path(name).suffix.casefold()
        expected_media = _ALLOWED_MEDIA_TYPES.get(suffix)
        if expected_media is None or media_type != expected_media:
            raise WorkspaceCapabilityError(
                "UNSUPPORTED_MEDIA_TYPE", "attachment extension and media type disagree"
            )
        if not isinstance(content_base64, str):
            raise WorkspaceCapabilityError(
                "INVALID_ATTACHMENT", "attachment content must be base64 text"
            )
        if not 1 <= len(content_base64) <= MAX_ATTACHMENT_BASE64_CHARS:
            raise WorkspaceCapabilityError(
                "ATTACHMENT_TOO_LARGE",
                "attachment base64 exceeds its encoded character limit",
            )
        try:
            content = b64decode(content_base64, validate=True)
        except (Base64Error, ValueError) as exc:
            raise WorkspaceCapabilityError(
                "INVALID_BASE64", "attachment base64 is invalid"
            ) from exc
        if not 1 <= len(content) <= MAX_ATTACHMENT_BYTES:
            raise WorkspaceCapabilityError(
                "ATTACHMENT_TOO_LARGE", "attachment exceeds its byte limit"
            )
        self._validate_content(suffix, content)
        digest = sha256(content).hexdigest()
        attachment_id = digest[:24]
        stored = self.storage_root / f"{attachment_id}{suffix}"
        if not stored.resolve(strict=False).is_relative_to(self.storage_root):
            raise WorkspaceCapabilityError(
                "INVALID_ATTACHMENT_NAME", "attachment path escaped storage root"
            )
        with self._lock:
            existing = self._attachments.get(attachment_id)
            if existing is not None:
                return {**existing.as_payload(), "duplicate": True}
            stored_resolved = stored.resolve(strict=False)
            reuses_persisted_blob = stored_resolved in self._persisted_valid_blobs
            if len(self._attachments) >= MAX_ATTACHMENT_COUNT or (
                not reuses_persisted_blob
                and self._persisted_blob_count >= MAX_ATTACHMENT_COUNT
            ):
                raise WorkspaceCapabilityError(
                    "ATTACHMENT_LIMIT_REACHED", "attachment count limit reached"
                )
            if (
                not reuses_persisted_blob
                and self._persisted_blob_bytes + len(content)
                > MAX_ATTACHMENT_TOTAL_BYTES
            ):
                raise WorkspaceCapabilityError(
                    "ATTACHMENT_TOTAL_BYTES_REACHED",
                    "attachment storage total byte limit reached",
                )
            _atomic_content_write(stored, content, digest)
            created = datetime.now(timezone.utc).isoformat()
            record = AttachmentRecord(
                attachment_id,
                name,
                media_type,
                len(content),
                digest,
                stored,
                created,
                suffix in _TEXT_MEDIA_TYPES,
            )
            self._attachments[attachment_id] = record
            if not reuses_persisted_blob:
                self._persisted_valid_blobs = frozenset(
                    (*self._persisted_valid_blobs, stored.resolve(strict=True))
                )
                self._persisted_blob_count += 1
                self._persisted_blob_bytes += len(content)
        return {**record.as_payload(), "duplicate": False}

    @staticmethod
    def _validate_content(suffix: str, content: bytes) -> None:
        if suffix in _TEXT_MEDIA_TYPES:
            try:
                text = content.decode("utf-8")
            except UnicodeDecodeError as exc:
                raise WorkspaceCapabilityError(
                    "INVALID_TEXT_ENCODING", "text attachments must be UTF-8"
                ) from exc
            if "\x00" in text:
                raise WorkspaceCapabilityError(
                    "INVALID_TEXT_ENCODING", "text attachments cannot contain NUL"
                )
            if suffix == ".json":
                if len(content) > MAX_JSON_ATTACHMENT_BYTES:
                    raise WorkspaceCapabilityError(
                        "JSON_ATTACHMENT_TOO_LARGE",
                        "JSON attachment exceeds its one-megabyte parser limit",
                    )
                if not _json_nesting_is_bounded(text):
                    raise WorkspaceCapabilityError(
                        "JSON_NESTING_TOO_DEEP",
                        "JSON attachment exceeds its nesting limit",
                    )
                try:
                    json.loads(text)
                except (json.JSONDecodeError, RecursionError) as exc:
                    raise WorkspaceCapabilityError(
                        "INVALID_JSON_ATTACHMENT", "JSON attachment is invalid"
                    ) from exc
            return
        valid = {
            ".pdf": content.startswith(b"%PDF-"),
            ".png": content.startswith(b"\x89PNG\r\n\x1a\n"),
            ".jpg": content.startswith(b"\xff\xd8\xff"),
            ".jpeg": content.startswith(b"\xff\xd8\xff"),
            ".webp": (
                len(content) >= 12
                and content.startswith(b"RIFF")
                and content[8:12] == b"WEBP"
            ),
        }
        if not valid.get(suffix, False):
            raise WorkspaceCapabilityError(
                "CONTENT_TYPE_MISMATCH", "attachment signature does not match its type"
            )

    def index_attachments(self, attachment_ids: list[str]) -> dict[str, object]:
        if self.akasicdb is None:
            code, message = self.akasicdb_error or (
                "AKASICDB_UNAVAILABLE",
                "AkasicDB is unavailable",
            )
            raise WorkspaceCapabilityError(code, message)
        if (
            not isinstance(attachment_ids, list)
            or not 1 <= len(attachment_ids) <= MAX_ATTACHMENT_COUNT
            or any(not isinstance(item, str) for item in attachment_ids)
        ):
            raise WorkspaceCapabilityError(
                "INVALID_ATTACHMENT_IDS", "attachment_ids must be a bounded text list"
            )
        results: list[dict[str, object]] = []
        for attachment_id in attachment_ids:
            with self._lock:
                record = self._attachments.get(attachment_id)
            if record is None:
                raise WorkspaceCapabilityError(
                    "ATTACHMENT_NOT_FOUND", "attachment was not found"
                )
            if not record.text_indexable:
                raise WorkspaceCapabilityError(
                    "ATTACHMENT_NOT_INDEXABLE",
                    "only UTF-8 text attachments can be indexed",
                )
            if (
                record.stored_path.is_symlink()
                or not record.stored_path.is_file()
                or not record.stored_path.resolve(strict=True).is_relative_to(
                    self.storage_root
                )
                or _sha256_file(record.stored_path) != record.sha256
            ):
                raise WorkspaceCapabilityError(
                    "ATTACHMENT_INTEGRITY_FAILED",
                    "attachment changed after admission",
                )
            text = record.stored_path.read_text(encoding="utf-8")
            results.append(
                self.akasicdb.index_document(
                    attachment_id=record.attachment_id,
                    name=record.name,
                    media_type=record.media_type,
                    text=text,
                )
            )
        return {
            "engine": "AkasicDB",
            "results": results,
            "documents": self.akasicdb.document_count,
            "chunks": self.akasicdb.chunk_count,
            "answer_integration": self.answer_integration_enabled,
        }

    def query_rag(self, query: str, *, limit: int = 5) -> dict[str, object]:
        if self.akasicdb is None:
            code, message = self.akasicdb_error or (
                "AKASICDB_UNAVAILABLE",
                "AkasicDB is unavailable",
            )
            raise WorkspaceCapabilityError(code, message)
        result = self.akasicdb.query(query, limit=limit)
        return {
            "engine": "AkasicDB",
            "embedding": "stable_sha256_lexical_sketch_v1",
            "answer_integration": self.answer_integration_enabled,
            **result,
        }

    def select_model(self, model_id: str) -> dict[str, object]:
        if not isinstance(model_id, str) or model_id != self.model.model_id:
            raise WorkspaceCapabilityError(
                "MODEL_NOT_VERIFIED", "requested model is not in the verified registry"
            )
        return self.model.as_payload()


def web_policy_from_environment(environment: Mapping[str, str]) -> WebAccessPolicy:
    """Compile explicit network flags without making network access implicit."""

    online = environment.get("COGNI_OS_ONLINE_MODE") == "1"
    raw_allowlist = environment.get("COGNI_OS_WEB_ALLOWLIST", "")
    allowlist = tuple(
        item.strip().casefold() for item in raw_allowlist.split(",") if item.strip()
    )
    lens_token = bool(environment.get("COGNI_OS_LENS_API_TOKEN", "").strip())
    return WebAccessPolicy(
        online_mode=online,
        allowlist=allowlist,
        lens_api_token_configured=lens_token,
    )


__all__ = [
    "AKASICDB_AUDITED_REVISION",
    "AKASICDB_REPOSITORY",
    "AkasicDBAdapter",
    "AttachmentRecord",
    "MAX_ATTACHMENT_BYTES",
    "MAX_ATTACHMENT_TOTAL_BYTES",
    "MAX_JSON_ATTACHMENT_BYTES",
    "MAX_JSON_NESTING",
    "MAX_RAG_RESULTS",
    "VerifiedModelMetadata",
    "WebAccessPolicy",
    "WorkspaceCapabilityError",
    "WorkspaceCapabilityService",
    "web_policy_from_environment",
]
