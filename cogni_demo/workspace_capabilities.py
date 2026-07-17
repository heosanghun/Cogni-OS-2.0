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
from math import isfinite, sqrt
import os
from pathlib import Path
import re
import secrets
import stat
import subprocess
import sys
from threading import RLock
from types import ModuleType
from typing import Any, Mapping
from urllib.parse import urlsplit

from cogni_demo.lens_api import (
    LensAkasicBridge,
    LensApiClient,
    LensApiError,
    LensSearchKind,
)
from cogni_os.factbook import RuntimeFactBook


MAX_ATTACHMENT_BYTES = 8 * 1024 * 1024
MAX_ATTACHMENT_TOTAL_BYTES = 64 * 1024 * 1024
MAX_ATTACHMENT_BASE64_CHARS = 4 * ((MAX_ATTACHMENT_BYTES + 2) // 3)
MAX_ATTACHMENT_COUNT = 32
MAX_ATTACHMENT_NAME_CHARS = 128
MAX_ATTACHMENT_CATALOG_BYTES = 128 * 1024
MAX_ATTACHMENT_PREVIEW_CHARS = 12_000
MAX_RAG_QUARANTINE_MARKER_BYTES = 1_024
MAX_JSON_ATTACHMENT_BYTES = 1024 * 1024
MAX_JSON_NESTING = 64
MAX_INDEXED_TEXT_CHARS = 256_000
MAX_PDF_PAGES = 128
MAX_PDF_EXTRACTED_CHARS = MAX_INDEXED_TEXT_CHARS
MAX_PDF_WORKER_OUTPUT_BYTES = 2 * 1024 * 1024
PDF_EXTRACT_TIMEOUT_SECONDS = 8.0
PDF_EXTRACT_MEMORY_LIMIT_BYTES = 256 * 1024 * 1024
PDF_EXTRACT_CPU_LIMIT_SECONDS = 6
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
_ATTACHMENT_ID_RE = re.compile(r"[0-9a-f]{24}\Z")
_CATALOG_FILENAME = "attachment-catalog.v1.json"
_CATALOG_SCHEMA_VERSION = 1
_RAG_QUARANTINE_FILENAME = "rag-quarantine.v1.json"
_RAG_QUARANTINE_SCHEMA_VERSION = 1
_RAG_QUARANTINE_CODE = "RAG_OPERATOR_REVIEW_REQUIRED"
_RAG_QUARANTINE_MESSAGE = "local RAG state is quarantined pending operator review"
_RAG_TRANSACTION_PENDING = "pending"
_RAG_TRANSACTION_COMMITTED = "committed"
_RAG_TRANSACTION_COMMITTED_REASON = "RAG_TRANSACTION_DURABLY_COMMITTED"
_RAG_QUARANTINE_KEYS = {
    "schema_version",
    "state",
    "reason",
    "catalog_sha256",
}

try:
    from pypdf import PdfReader as _PdfReader
except Exception:  # noqa: BLE001 - optional parser must fail closed at import
    _PdfReader = None


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
class ExtractedPage:
    """One physical PDF page after bounded, deterministic normalization."""

    page_number: int
    text: str

    def __post_init__(self) -> None:
        if (
            not isinstance(self.page_number, int)
            or isinstance(self.page_number, bool)
            or self.page_number < 1
        ):
            raise ValueError("page_number must be a positive integer")
        if not isinstance(self.text, str):
            raise TypeError("page text must be str")


@dataclass(frozen=True, slots=True)
class ExtractedDocument:
    """A bounded document plus optional physical-page provenance.

    Non-paginated UTF-8 attachments use ``page_count=0`` and ``pages=()``.
    PDFs always carry every physical page, including pages with empty text.
    """

    text: str
    page_count: int
    pages: tuple[ExtractedPage, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.text, str):
            raise TypeError("document text must be str")
        if (
            not isinstance(self.page_count, int)
            or isinstance(self.page_count, bool)
            or self.page_count < 0
        ):
            raise ValueError("page_count must be a non-negative integer")
        if not isinstance(self.pages, tuple) or any(
            not isinstance(page, ExtractedPage) for page in self.pages
        ):
            raise TypeError("pages must be an ExtractedPage tuple")
        if self.pages:
            if self.page_count != len(self.pages) or tuple(
                page.page_number for page in self.pages
            ) != tuple(range(1, self.page_count + 1)):
                raise ValueError("physical PDF pages must be exact and sequential")
            if self.text != "\n\n".join(page.text for page in self.pages):
                raise ValueError("document text must match its physical pages")
        elif self.page_count != 0:
            raise ValueError("a paginated document must include every physical page")


@dataclass(frozen=True, slots=True)
class IndexedChunk:
    """One searchable excerpt with an exact normalized-source coordinate."""

    chunk_index: int
    text: str
    page_number: int | None
    char_start: int
    char_end: int
    offset_basis: str
    excerpt_sha256: str

    def as_payload(self) -> dict[str, object]:
        return {
            "chunk_index": self.chunk_index,
            "text": self.text,
            "page_number": self.page_number,
            "char_start": self.char_start,
            "char_end": self.char_end,
            "offset_basis": self.offset_basis,
            "excerpt_sha256": self.excerpt_sha256,
        }


@dataclass(frozen=True, slots=True)
class IndexedSourceSnapshot:
    """Immutable, blob-bound authority for one indexed browser excerpt."""

    attachment_id: str
    attachment_sha256: str
    name: str
    media_type: str
    chunk: IndexedChunk

    def as_payload(self) -> dict[str, object]:
        return {
            "attachment_id": self.attachment_id,
            "name": self.name,
            "media_type": self.media_type,
            **self.chunk.as_payload(),
        }

    def matches(self, record: AttachmentRecord, indexed: Mapping[str, object]) -> bool:
        return (
            record.attachment_id == self.attachment_id
            and record.sha256 == self.attachment_sha256
            and record.name == self.name
            and record.media_type == self.media_type
            and indexed == self.as_payload()
        )


def _pdf_extractor_available() -> bool:
    return _PdfReader is not None


def _normalize_extracted_text(value: str) -> str:
    """Remove control bytes that cannot cross the RAG or browser boundary."""

    normalized: list[str] = []
    for character in value.replace("\r\n", "\n").replace("\r", "\n"):
        codepoint = ord(character)
        if character in "\n\t":
            normalized.append(character)
        elif codepoint < 32 or 0x7F <= codepoint <= 0x9F:
            normalized.append(" ")
        else:
            normalized.append(character)
    return "".join(normalized).strip()


def _assign_windows_pdf_job(process: subprocess.Popen[bytes]) -> int:
    """Assign the waiting parser process to a kill-on-close bounded Job Object."""

    if os.name != "nt":
        return 0
    import ctypes
    from ctypes import wintypes

    class _BasicLimitInformation(ctypes.Structure):
        _fields_ = [
            ("PerProcessUserTimeLimit", ctypes.c_longlong),
            ("PerJobUserTimeLimit", ctypes.c_longlong),
            ("LimitFlags", wintypes.DWORD),
            ("MinimumWorkingSetSize", ctypes.c_size_t),
            ("MaximumWorkingSetSize", ctypes.c_size_t),
            ("ActiveProcessLimit", wintypes.DWORD),
            ("Affinity", ctypes.c_size_t),
            ("PriorityClass", wintypes.DWORD),
            ("SchedulingClass", wintypes.DWORD),
        ]

    class _IoCounters(ctypes.Structure):
        _fields_ = [
            ("ReadOperationCount", ctypes.c_ulonglong),
            ("WriteOperationCount", ctypes.c_ulonglong),
            ("OtherOperationCount", ctypes.c_ulonglong),
            ("ReadTransferCount", ctypes.c_ulonglong),
            ("WriteTransferCount", ctypes.c_ulonglong),
            ("OtherTransferCount", ctypes.c_ulonglong),
        ]

    class _ExtendedLimitInformation(ctypes.Structure):
        _fields_ = [
            ("BasicLimitInformation", _BasicLimitInformation),
            ("IoInfo", _IoCounters),
            ("ProcessMemoryLimit", ctypes.c_size_t),
            ("JobMemoryLimit", ctypes.c_size_t),
            ("PeakProcessMemoryUsed", ctypes.c_size_t),
            ("PeakJobMemoryUsed", ctypes.c_size_t),
        ]

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.CreateJobObjectW.argtypes = (ctypes.c_void_p, wintypes.LPCWSTR)
    kernel32.CreateJobObjectW.restype = wintypes.HANDLE
    kernel32.SetInformationJobObject.argtypes = (
        wintypes.HANDLE,
        ctypes.c_int,
        ctypes.c_void_p,
        wintypes.DWORD,
    )
    kernel32.SetInformationJobObject.restype = wintypes.BOOL
    kernel32.AssignProcessToJobObject.argtypes = (wintypes.HANDLE, wintypes.HANDLE)
    kernel32.AssignProcessToJobObject.restype = wintypes.BOOL

    job = kernel32.CreateJobObjectW(None, None)
    if not job:
        raise OSError(ctypes.get_last_error(), "CreateJobObjectW failed")
    information = _ExtendedLimitInformation()
    information.BasicLimitInformation.PerProcessUserTimeLimit = (
        PDF_EXTRACT_CPU_LIMIT_SECONDS * 10_000_000
    )
    information.BasicLimitInformation.LimitFlags = 0x00000002 | 0x00000100 | 0x00002000
    information.ProcessMemoryLimit = PDF_EXTRACT_MEMORY_LIMIT_BYTES
    if not kernel32.SetInformationJobObject(
        job,
        9,
        ctypes.byref(information),
        ctypes.sizeof(information),
    ):
        error = ctypes.get_last_error()
        kernel32.CloseHandle(job)
        raise OSError(error, "SetInformationJobObject failed")
    if not kernel32.AssignProcessToJobObject(
        job,
        wintypes.HANDLE(process._handle),  # noqa: SLF001 - native process handle
    ):
        error = ctypes.get_last_error()
        kernel32.CloseHandle(job)
        raise OSError(error, "AssignProcessToJobObject failed")
    return int(job)


def _close_windows_handle(handle: int) -> None:
    if os.name == "nt" and handle:
        import ctypes
        from ctypes import wintypes

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.CloseHandle.argtypes = (wintypes.HANDLE,)
        kernel32.CloseHandle.restype = wintypes.BOOL
        kernel32.CloseHandle(wintypes.HANDLE(handle))


def _run_pdf_extractor_document(content: bytes) -> ExtractedDocument:
    candidate = Path(__file__).with_name("pdf_extract_worker.py")
    if candidate.is_symlink():
        raise WorkspaceCapabilityError(
            "PDF_SANDBOX_UNAVAILABLE", "the PDF worker path is unsafe"
        )
    worker = candidate.resolve(strict=True)
    if worker.parent != Path(__file__).parent.resolve(strict=True):
        raise WorkspaceCapabilityError(
            "PDF_SANDBOX_UNAVAILABLE", "the PDF worker path is unsafe"
        )
    creation_flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    try:
        process = subprocess.Popen(
            [sys.executable, "-I", str(worker)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=str(worker.parent),
            creationflags=creation_flags,
        )
    except OSError as exc:
        raise WorkspaceCapabilityError(
            "PDF_SANDBOX_UNAVAILABLE", "the PDF worker could not be started"
        ) from exc
    job_handle = 0
    try:
        try:
            job_handle = _assign_windows_pdf_job(process)
        except OSError as exc:
            process.kill()
            process.communicate()
            raise WorkspaceCapabilityError(
                "PDF_SANDBOX_UNAVAILABLE", "PDF resource limits could not be applied"
            ) from exc
        try:
            stdout, _stderr = process.communicate(
                input=content,
                timeout=PDF_EXTRACT_TIMEOUT_SECONDS,
            )
        except subprocess.TimeoutExpired as exc:
            if job_handle:
                _close_windows_handle(job_handle)
                job_handle = 0
            else:
                process.kill()
            process.communicate()
            raise WorkspaceCapabilityError(
                "PDF_TEXT_EXTRACTION_TIMEOUT", "PDF extraction exceeded its time limit"
            ) from exc
    finally:
        _close_windows_handle(job_handle)
    if len(stdout) > MAX_PDF_WORKER_OUTPUT_BYTES:
        raise WorkspaceCapabilityError(
            "PDF_TEXT_LIMIT", "PDF worker output exceeds its byte limit"
        )
    try:
        payload = json.loads(stdout.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise WorkspaceCapabilityError(
            "PDF_TEXT_EXTRACTION_FAILED", "the PDF worker returned invalid output"
        ) from exc
    if not isinstance(payload, dict) or not isinstance(payload.get("ok"), bool):
        raise WorkspaceCapabilityError(
            "PDF_TEXT_EXTRACTION_FAILED", "the PDF worker response is invalid"
        )
    if process.returncode != 0 or payload["ok"] is not True:
        code = payload.get("code")
        message = payload.get("message")
        if (
            not isinstance(code, str)
            or re.fullmatch(r"[A-Z][A-Z0-9_]{0,63}", code) is None
        ):
            code = "PDF_TEXT_EXTRACTION_FAILED"
        if not isinstance(message, str) or not 1 <= len(message) <= 256:
            message = "the isolated PDF extractor rejected this document"
        raise WorkspaceCapabilityError(code, message)
    raw_pages = payload.get("pages")
    if (
        set(payload) != {"ok", "pages"}
        or not isinstance(raw_pages, list)
        or not 1 <= len(raw_pages) <= MAX_PDF_PAGES
    ):
        raise WorkspaceCapabilityError(
            "PDF_TEXT_EXTRACTION_FAILED", "the PDF worker result exceeded its contract"
        )
    pages: list[ExtractedPage] = []
    for expected_page_number, raw_page in enumerate(raw_pages, start=1):
        if not isinstance(raw_page, dict) or set(raw_page) != {
            "page_number",
            "text",
        }:
            raise WorkspaceCapabilityError(
                "PDF_TEXT_EXTRACTION_FAILED",
                "the PDF worker page result is invalid",
            )
        page_number = raw_page.get("page_number")
        text = raw_page.get("text")
        if (
            type(page_number) is not int
            or page_number != expected_page_number
            or not isinstance(text, str)
            or _normalize_extracted_text(text) != text
        ):
            raise WorkspaceCapabilityError(
                "PDF_TEXT_EXTRACTION_FAILED",
                "the PDF worker page sequence is invalid",
            )
        pages.append(ExtractedPage(page_number=page_number, text=text))
    document_text = "\n\n".join(page.text for page in pages)
    if (
        not any(page.text for page in pages)
        or len(document_text) > MAX_PDF_EXTRACTED_CHARS
    ):
        raise WorkspaceCapabilityError(
            "PDF_TEXT_LIMIT", "extracted PDF text exceeds its character limit"
        )
    return ExtractedDocument(
        text=document_text,
        page_count=len(pages),
        pages=tuple(pages),
    )


def _run_pdf_extractor_subprocess(content: bytes) -> str:
    """Compatibility wrapper for callers that consume only normalized text."""

    return _run_pdf_extractor_document(content).text


def _extract_pdf_document(content: bytes) -> ExtractedDocument:
    """Extract a PDF with every physical page preserved for provenance."""

    if _PdfReader is None:
        raise WorkspaceCapabilityError(
            "PDF_TEXT_EXTRACTION_UNAVAILABLE",
            "the optional local pypdf extractor is unavailable",
        )
    if not isinstance(content, bytes) or not 1 <= len(content) <= MAX_ATTACHMENT_BYTES:
        raise WorkspaceCapabilityError(
            "ATTACHMENT_TOO_LARGE", "PDF attachment exceeds its byte limit"
        )
    return _run_pdf_extractor_document(content)


def _extract_pdf_text(content: bytes) -> str:
    """Extract PDF text in a wall/CPU/RAM-bounded local subprocess."""

    if _PdfReader is None:
        raise WorkspaceCapabilityError(
            "PDF_TEXT_EXTRACTION_UNAVAILABLE",
            "the optional local pypdf extractor is unavailable",
        )
    if not isinstance(content, bytes) or not 1 <= len(content) <= MAX_ATTACHMENT_BYTES:
        raise WorkspaceCapabilityError(
            "ATTACHMENT_TOO_LARGE", "PDF attachment exceeds its byte limit"
        )
    text = _normalize_extracted_text(_run_pdf_extractor_subprocess(content))
    if not text or len(text) > MAX_PDF_EXTRACTED_CHARS:
        raise WorkspaceCapabilityError(
            "PDF_TEXT_LIMIT", "extracted PDF text exceeds its character limit"
        )
    return text


def _bounded_preview_text(text: str) -> tuple[str, bool]:
    normalized = _normalize_extracted_text(text)
    if len(normalized) <= MAX_ATTACHMENT_PREVIEW_CHARS:
        return normalized, False
    return normalized[:MAX_ATTACHMENT_PREVIEW_CHARS].rstrip(), True


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


def _attachment_name_is_safe(name: object) -> bool:
    return bool(
        isinstance(name, str)
        and 1 <= len(name) <= MAX_ATTACHMENT_NAME_CHARS
        and Path(name).name == name
        and name not in {".", ".."}
        and not any(
            ord(character) < 32 or 0x7F <= ord(character) <= 0x9F for character in name
        )
    )


def _atomic_json_write(path: Path, payload: Mapping[str, object]) -> None:
    """Write one bounded catalog without following a link or exposing a partial file."""

    try:
        encoded = json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise WorkspaceCapabilityError(
            "ATTACHMENT_CATALOG_INVALID", "attachment catalog could not be encoded"
        ) from exc
    if not 1 <= len(encoded) <= MAX_ATTACHMENT_CATALOG_BYTES:
        raise WorkspaceCapabilityError(
            "ATTACHMENT_CATALOG_TOO_LARGE", "attachment catalog exceeds its byte limit"
        )
    parent = path.parent.resolve(strict=True)
    if path.parent.is_symlink() or not path.resolve(strict=False).is_relative_to(
        parent
    ):
        raise WorkspaceCapabilityError(
            "ATTACHMENT_CATALOG_UNSAFE", "attachment catalog path is unsafe"
        )
    if path.exists() or path.is_symlink():
        if path.is_symlink() or not path.is_file():
            raise WorkspaceCapabilityError(
                "ATTACHMENT_CATALOG_UNSAFE", "attachment catalog is not a regular file"
            )
    temporary = parent / f".{path.name}.{os.getpid()}.{secrets.token_hex(8)}.tmp"
    descriptor = -1
    try:
        descriptor = os.open(
            temporary,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            stat.S_IRUSR | stat.S_IWUSR,
        )
        with os.fdopen(descriptor, "wb") as stream:
            descriptor = -1
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        if (
            path.is_symlink()
            or not path.resolve(strict=True).is_relative_to(parent)
            or path.stat().st_size != len(encoded)
        ):
            raise WorkspaceCapabilityError(
                "ATTACHMENT_CATALOG_UNSAFE", "attachment catalog publication failed"
            )
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _fsync_directory(path: Path, *, error_code: str, message: str) -> None:
    """Make one marker publication/removal durable on the server filesystem."""

    if os.name == "nt":
        return
    descriptor = -1
    try:
        descriptor = os.open(path, os.O_RDONLY)
        os.fsync(descriptor)
    except OSError as exc:
        raise WorkspaceCapabilityError(error_code, message) from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def _catalog_digest_for_rag_marker(catalog_path: Path) -> str:
    """Hash the bounded, regular catalog before any destructive RAG mutation."""

    parent = catalog_path.parent.resolve(strict=True)
    try:
        resolved = catalog_path.resolve(strict=True)
        if (
            catalog_path.is_symlink()
            or not resolved.is_relative_to(parent)
            or not resolved.is_file()
        ):
            raise WorkspaceCapabilityError(
                "RAG_QUARANTINE_MARKER_WRITE_FAILED",
                "RAG transaction marker could not bind the attachment catalog",
            )
        size = resolved.stat().st_size
        if not 1 <= size <= MAX_ATTACHMENT_CATALOG_BYTES:
            raise WorkspaceCapabilityError(
                "RAG_QUARANTINE_MARKER_WRITE_FAILED",
                "RAG transaction marker could not bind the attachment catalog",
            )
        content = resolved.read_bytes()
    except WorkspaceCapabilityError:
        raise
    except OSError as exc:
        raise WorkspaceCapabilityError(
            "RAG_QUARANTINE_MARKER_WRITE_FAILED",
            "RAG transaction marker could not bind the attachment catalog",
        ) from exc
    if len(content) != size:
        raise WorkspaceCapabilityError(
            "RAG_QUARANTINE_MARKER_WRITE_FAILED",
            "RAG transaction marker could not bind the attachment catalog",
        )
    return sha256(content).hexdigest()


@dataclass(frozen=True)
class _RagTransactionMarker:
    state: str
    reason: str
    catalog_sha256: str


def _rag_transaction_marker_payload(
    *, state: str, catalog_sha256: str
) -> dict[str, object]:
    if _SHA256_RE.fullmatch(catalog_sha256) is None:
        raise WorkspaceCapabilityError(
            "RAG_QUARANTINE_MARKER_INVALID",
            "RAG quarantine marker digest is invalid",
        )
    if state == _RAG_TRANSACTION_PENDING:
        reason = _RAG_QUARANTINE_CODE
    elif state == _RAG_TRANSACTION_COMMITTED:
        reason = _RAG_TRANSACTION_COMMITTED_REASON
    else:
        raise WorkspaceCapabilityError(
            "RAG_QUARANTINE_MARKER_INVALID",
            "RAG quarantine marker state is invalid",
        )
    return {
        "schema_version": _RAG_QUARANTINE_SCHEMA_VERSION,
        "state": state,
        "reason": reason,
        "catalog_sha256": catalog_sha256,
    }


def _atomic_rag_transaction_marker_write(
    path: Path,
    *,
    state: str,
    catalog_sha256: str,
    replace_existing: bool,
) -> None:
    """Atomically publish one fsync'd transaction state marker."""

    payload = _rag_transaction_marker_payload(
        state=state, catalog_sha256=catalog_sha256
    )
    encoded = json.dumps(
        payload,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")
    if not 1 <= len(encoded) <= MAX_RAG_QUARANTINE_MARKER_BYTES:
        raise WorkspaceCapabilityError(
            "RAG_QUARANTINE_MARKER_WRITE_FAILED",
            "RAG quarantine marker exceeds its byte limit",
        )
    parent = path.parent.resolve(strict=True)
    if path.parent.is_symlink() or not path.resolve(strict=False).is_relative_to(
        parent
    ):
        raise WorkspaceCapabilityError(
            "RAG_QUARANTINE_MARKER_WRITE_FAILED",
            "RAG quarantine marker path is unsafe",
        )
    if not replace_existing and (path.exists() or path.is_symlink()):
        raise WorkspaceCapabilityError(
            "RAG_QUARANTINE_MARKER_WRITE_FAILED",
            "RAG quarantine marker already requires operator review",
        )
    temporary = parent / f".{path.name}.{os.getpid()}.{secrets.token_hex(8)}.tmp"
    descriptor = -1
    try:
        descriptor = os.open(
            temporary,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            stat.S_IRUSR | stat.S_IWUSR,
        )
        with os.fdopen(descriptor, "wb") as stream:
            descriptor = -1
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        _fsync_directory(
            parent,
            error_code="RAG_QUARANTINE_MARKER_WRITE_FAILED",
            message="RAG quarantine marker publication was not durable",
        )
        if (
            path.is_symlink()
            or not path.resolve(strict=True).is_relative_to(parent)
            or not path.is_file()
            or path.read_bytes() != encoded
        ):
            raise WorkspaceCapabilityError(
                "RAG_QUARANTINE_MARKER_WRITE_FAILED",
                "RAG quarantine marker publication failed verification",
            )
    except WorkspaceCapabilityError:
        raise
    except OSError as exc:
        raise WorkspaceCapabilityError(
            "RAG_QUARANTINE_MARKER_WRITE_FAILED",
            "RAG quarantine marker could not be published",
        ) from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _load_rag_quarantine_marker(path: Path) -> _RagTransactionMarker | None:
    """Load one strict marker; any present malformed marker fails closed."""

    if not path.exists() and not path.is_symlink():
        return None
    if path.is_symlink() or not path.is_file():
        raise WorkspaceCapabilityError(
            "RAG_QUARANTINE_MARKER_INVALID",
            "RAG quarantine marker is not a regular file",
        )
    parent = path.parent.resolve(strict=True)
    try:
        resolved = path.resolve(strict=True)
        if not resolved.is_relative_to(parent):
            raise WorkspaceCapabilityError(
                "RAG_QUARANTINE_MARKER_INVALID",
                "RAG quarantine marker escaped its workspace",
            )
        size = resolved.stat().st_size
        if not 1 <= size <= MAX_RAG_QUARANTINE_MARKER_BYTES:
            raise WorkspaceCapabilityError(
                "RAG_QUARANTINE_MARKER_INVALID",
                "RAG quarantine marker exceeds its byte limit",
            )
        encoded = resolved.read_bytes()
        if len(encoded) != size:
            raise WorkspaceCapabilityError(
                "RAG_QUARANTINE_MARKER_INVALID",
                "RAG quarantine marker changed while being read",
            )

        def reject_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
            result: dict[str, object] = {}
            for key, value in pairs:
                if key in result:
                    raise ValueError("duplicate marker key")
                result[key] = value
            return result

        payload = json.loads(
            encoded.decode("utf-8"), object_pairs_hook=reject_duplicates
        )
    except WorkspaceCapabilityError:
        raise
    except (
        OSError,
        UnicodeError,
        json.JSONDecodeError,
        RecursionError,
        ValueError,
    ) as exc:
        raise WorkspaceCapabilityError(
            "RAG_QUARANTINE_MARKER_INVALID",
            "RAG quarantine marker is invalid",
        ) from exc
    if (
        not isinstance(payload, dict)
        or set(payload) != _RAG_QUARANTINE_KEYS
        or payload.get("schema_version") != _RAG_QUARANTINE_SCHEMA_VERSION
        or isinstance(payload.get("schema_version"), bool)
        or not isinstance(payload.get("state"), str)
        or payload.get("state")
        not in (_RAG_TRANSACTION_PENDING, _RAG_TRANSACTION_COMMITTED)
        or not isinstance(payload.get("reason"), str)
        or not isinstance(payload.get("catalog_sha256"), str)
        or _SHA256_RE.fullmatch(payload["catalog_sha256"]) is None
    ):
        raise WorkspaceCapabilityError(
            "RAG_QUARANTINE_MARKER_INVALID",
            "RAG quarantine marker schema is invalid",
        )
    expected_reason = (
        _RAG_QUARANTINE_CODE
        if payload["state"] == _RAG_TRANSACTION_PENDING
        else _RAG_TRANSACTION_COMMITTED_REASON
    )
    if payload["reason"] != expected_reason:
        raise WorkspaceCapabilityError(
            "RAG_QUARANTINE_MARKER_INVALID",
            "RAG quarantine marker reason is invalid",
        )
    return _RagTransactionMarker(
        state=payload["state"],
        reason=payload["reason"],
        catalog_sha256=payload["catalog_sha256"],
    )


def _atomic_rag_quarantine_write(path: Path, catalog_sha256: str) -> None:
    """Publish pending, replacing only a matching leftover committed marker."""

    observed = _load_rag_quarantine_marker(path)
    replace_existing = False
    if observed is not None:
        if (
            observed.state != _RAG_TRANSACTION_COMMITTED
            or observed.catalog_sha256 != catalog_sha256
        ):
            raise WorkspaceCapabilityError(
                "RAG_QUARANTINE_MARKER_WRITE_FAILED",
                "RAG quarantine marker already requires operator review",
            )
        replace_existing = True
    _atomic_rag_transaction_marker_write(
        path,
        state=_RAG_TRANSACTION_PENDING,
        catalog_sha256=catalog_sha256,
        replace_existing=replace_existing,
    )


def _commit_rag_transaction_marker(
    path: Path,
    *,
    expected_pending_sha256: str,
    committed_catalog_sha256: str,
) -> None:
    """Durably replace pending with a digest-bound committed state."""

    observed = _load_rag_quarantine_marker(path)
    if (
        observed is None
        or observed.state != _RAG_TRANSACTION_PENDING
        or observed.catalog_sha256 != expected_pending_sha256
    ):
        raise WorkspaceCapabilityError(
            "RAG_QUARANTINE_MARKER_COMMIT_FAILED",
            "RAG quarantine marker no longer matches this transaction",
        )
    _atomic_rag_transaction_marker_write(
        path,
        state=_RAG_TRANSACTION_COMMITTED,
        catalog_sha256=committed_catalog_sha256,
        replace_existing=True,
    )


def _cleanup_committed_rag_marker(path: Path, expected_catalog_sha256: str) -> None:
    """Best-effort cleanup after committed is already restart-safe."""

    observed = _load_rag_quarantine_marker(path)
    if (
        observed is None
        or observed.state != _RAG_TRANSACTION_COMMITTED
        or observed.catalog_sha256 != expected_catalog_sha256
    ):
        raise WorkspaceCapabilityError(
            "RAG_QUARANTINE_MARKER_CLEAR_FAILED",
            "committed RAG marker no longer matches the catalog",
        )
    parent = path.parent.resolve(strict=True)
    try:
        path.unlink()
        _fsync_directory(
            parent,
            error_code="RAG_QUARANTINE_MARKER_CLEAR_FAILED",
            message="committed RAG marker removal was not durable",
        )
    except WorkspaceCapabilityError:
        raise
    except OSError as exc:
        raise WorkspaceCapabilityError(
            "RAG_QUARANTINE_MARKER_CLEAR_FAILED",
            "RAG quarantine marker could not be removed",
        ) from exc
    if path.exists() or path.is_symlink():
        raise WorkspaceCapabilityError(
            "RAG_QUARANTINE_MARKER_CLEAR_FAILED",
            "RAG quarantine marker removal failed verification",
        )


def _rag_quarantine_required_on_startup(marker_path: Path, catalog_path: Path) -> bool:
    """Resolve crash states without promoting a pending or mismatched outcome."""

    marker = _load_rag_quarantine_marker(marker_path)
    if marker is None:
        return False
    if marker.state == _RAG_TRANSACTION_PENDING:
        return True
    catalog_sha256 = _catalog_digest_for_rag_marker(catalog_path)
    if marker.catalog_sha256 != catalog_sha256:
        return True
    try:
        _cleanup_committed_rag_marker(marker_path, catalog_sha256)
    except Exception:  # noqa: BLE001 - committed may safely remain or be absent
        remaining = _load_rag_quarantine_marker(marker_path)
        if remaining is None:
            return False
        return not (
            remaining.state == _RAG_TRANSACTION_COMMITTED
            and remaining.catalog_sha256 == catalog_sha256
        )
    return False


def _load_attachment_catalog(
    catalog_path: Path, storage_root: Path
) -> tuple[dict[str, AttachmentRecord], set[str]]:
    """Load only integrity-verified records; malformed state fails closed."""

    if not catalog_path.exists() and not catalog_path.is_symlink():
        return {}, set()
    if catalog_path.is_symlink() or not catalog_path.is_file():
        raise WorkspaceCapabilityError(
            "ATTACHMENT_CATALOG_UNSAFE", "attachment catalog is not a regular file"
        )
    try:
        resolved = catalog_path.resolve(strict=True)
        if not resolved.is_relative_to(storage_root.parent.resolve(strict=True)):
            raise WorkspaceCapabilityError(
                "ATTACHMENT_CATALOG_UNSAFE", "attachment catalog escaped its workspace"
            )
        size = resolved.stat().st_size
        if not 1 <= size <= MAX_ATTACHMENT_CATALOG_BYTES:
            raise WorkspaceCapabilityError(
                "ATTACHMENT_CATALOG_TOO_LARGE",
                "attachment catalog exceeds its byte limit",
            )
        payload = json.loads(resolved.read_text(encoding="utf-8"))
    except WorkspaceCapabilityError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError, RecursionError) as exc:
        raise WorkspaceCapabilityError(
            "ATTACHMENT_CATALOG_INVALID", "attachment catalog is invalid"
        ) from exc
    if (
        not isinstance(payload, dict)
        or set(payload) != {"schema_version", "items", "indexed_attachment_ids"}
        or payload.get("schema_version") != _CATALOG_SCHEMA_VERSION
        or not isinstance(payload.get("items"), list)
        or not isinstance(payload.get("indexed_attachment_ids"), list)
        or len(payload["items"]) > MAX_ATTACHMENT_COUNT
        or len(payload["indexed_attachment_ids"]) > MAX_ATTACHMENT_COUNT
    ):
        raise WorkspaceCapabilityError(
            "ATTACHMENT_CATALOG_INVALID", "attachment catalog schema is invalid"
        )
    records: dict[str, AttachmentRecord] = {}
    for item in payload["items"]:
        if not isinstance(item, dict) or set(item) != {
            "attachment_id",
            "name",
            "media_type",
            "size_bytes",
            "sha256",
            "created_at",
            "blob_name",
            "text_indexable",
        }:
            raise WorkspaceCapabilityError(
                "ATTACHMENT_CATALOG_INVALID", "attachment catalog item is invalid"
            )
        attachment_id = item["attachment_id"]
        name = item["name"]
        media_type = item["media_type"]
        digest = item["sha256"]
        blob_name = item["blob_name"]
        size_bytes = item["size_bytes"]
        created_at = item["created_at"]
        text_indexable = item["text_indexable"]
        suffix = Path(name).suffix.casefold() if isinstance(name, str) else ""
        if (
            not isinstance(attachment_id, str)
            or _ATTACHMENT_ID_RE.fullmatch(attachment_id) is None
            or attachment_id in records
            or not _attachment_name_is_safe(name)
            or _ALLOWED_MEDIA_TYPES.get(suffix) != media_type
            or not isinstance(digest, str)
            or _SHA256_RE.fullmatch(digest) is None
            or not digest.startswith(attachment_id)
            or blob_name != f"{attachment_id}{suffix}"
            or not isinstance(size_bytes, int)
            or isinstance(size_bytes, bool)
            or not 1 <= size_bytes <= MAX_ATTACHMENT_BYTES
            or not isinstance(created_at, str)
            or not 1 <= len(created_at) <= 64
            or not isinstance(text_indexable, bool)
            or (suffix != ".pdf" and text_indexable != (suffix in _TEXT_MEDIA_TYPES))
        ):
            raise WorkspaceCapabilityError(
                "ATTACHMENT_CATALOG_INVALID", "attachment catalog item is invalid"
            )
        try:
            timestamp = datetime.fromisoformat(created_at)
        except ValueError as exc:
            raise WorkspaceCapabilityError(
                "ATTACHMENT_CATALOG_INVALID", "attachment timestamp is invalid"
            ) from exc
        if timestamp.tzinfo is None:
            raise WorkspaceCapabilityError(
                "ATTACHMENT_CATALOG_INVALID", "attachment timestamp is invalid"
            )
        stored = storage_root / blob_name
        try:
            stored_resolved = stored.resolve(strict=True)
        except OSError as exc:
            raise WorkspaceCapabilityError(
                "ATTACHMENT_INTEGRITY_FAILED", "catalogued attachment is missing"
            ) from exc
        if (
            stored.is_symlink()
            or not stored_resolved.is_relative_to(storage_root)
            or not stored_resolved.is_file()
            or stored_resolved.stat().st_size != size_bytes
            or _sha256_file(stored_resolved) != digest
        ):
            raise WorkspaceCapabilityError(
                "ATTACHMENT_INTEGRITY_FAILED",
                "catalogued attachment failed its integrity check",
            )
        records[attachment_id] = AttachmentRecord(
            attachment_id,
            name,
            media_type,
            size_bytes,
            digest,
            stored_resolved,
            created_at,
            text_indexable or suffix == ".pdf",
        )
    indexed_raw = payload["indexed_attachment_ids"]
    if (
        any(not isinstance(item, str) for item in indexed_raw)
        or len(set(indexed_raw)) != len(indexed_raw)
        or not set(indexed_raw).issubset(records)
        or any(not records[item].text_indexable for item in indexed_raw)
    ):
        raise WorkspaceCapabilityError(
            "ATTACHMENT_CATALOG_INVALID", "attachment index state is invalid"
        )
    return records, set(indexed_raw)


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


def _normalize_index_source(text: str) -> str:
    if not isinstance(text, str):
        raise TypeError("indexed source must be text")
    return "\n".join(line.rstrip() for line in text.splitlines()).strip()


def _chunk_normalized_source(
    text: str,
    *,
    page_number: int | None,
    offset_basis: str,
    first_chunk_index: int,
) -> tuple[IndexedChunk, ...]:
    normalized = _normalize_index_source(text)
    if not normalized:
        return ()
    chunks: list[IndexedChunk] = []
    cursor = 0
    while cursor < len(normalized):
        end = min(len(normalized), cursor + RAG_CHUNK_CHARS)
        if end < len(normalized):
            boundary = normalized.rfind("\n", cursor + 1, end)
            if boundary <= cursor:
                boundary = normalized.rfind(" ", cursor + 1, end)
            if boundary > cursor + RAG_CHUNK_CHARS // 2:
                end = boundary
        segment = normalized[cursor:end]
        leading = len(segment) - len(segment.lstrip())
        trailing = len(segment.rstrip())
        char_start = cursor + leading
        char_end = cursor + trailing
        if char_start < char_end:
            chunk_text = normalized[char_start:char_end]
            chunks.append(
                IndexedChunk(
                    chunk_index=first_chunk_index + len(chunks),
                    text=chunk_text,
                    page_number=page_number,
                    char_start=char_start,
                    char_end=char_end,
                    offset_basis=offset_basis,
                    excerpt_sha256=sha256(chunk_text.encode("utf-8")).hexdigest(),
                )
            )
        if end >= len(normalized):
            break
        cursor = max(cursor + 1, end - RAG_CHUNK_OVERLAP_CHARS)
    return tuple(chunks)


def _chunk_document(document: ExtractedDocument) -> tuple[IndexedChunk, ...]:
    if not isinstance(document, ExtractedDocument):
        raise TypeError("document must be ExtractedDocument")
    bounded_text = (
        document.text if document.pages else _normalize_index_source(document.text)
    )
    if not bounded_text or len(bounded_text) > MAX_INDEXED_TEXT_CHARS:
        code = "EMPTY_DOCUMENT" if not bounded_text else "DOCUMENT_TOO_LARGE"
        message = (
            "document has no text"
            if not bounded_text
            else "document exceeds the local index character limit"
        )
        raise WorkspaceCapabilityError(code, message)
    chunks: list[IndexedChunk] = []
    if document.pages:
        for page in document.pages:
            chunks.extend(
                _chunk_normalized_source(
                    page.text,
                    page_number=page.page_number,
                    offset_basis="normalized_pdf_page_text_v1",
                    first_chunk_index=len(chunks),
                )
            )
    else:
        chunks.extend(
            _chunk_normalized_source(
                bounded_text,
                page_number=None,
                offset_basis="normalized_document_text_v1",
                first_chunk_index=0,
            )
        )
    if not chunks:
        raise WorkspaceCapabilityError("EMPTY_DOCUMENT", "document has no text")
    if len(chunks) > MAX_RAG_CHUNKS_PER_DOCUMENT:
        raise WorkspaceCapabilityError(
            "TOO_MANY_CHUNKS", "document exceeds the local index chunk limit"
        )
    return tuple(chunks)


def _chunk_text(text: str) -> tuple[str, ...]:
    """Compatibility wrapper for non-paginated callers."""

    document = ExtractedDocument(text=text, page_count=0, pages=())
    return tuple(chunk.text for chunk in _chunk_document(document))


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
        self._graph_type = graph_type
        self._relational_type = relational_type
        self._vector_type = vector_type
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
            "restart_recovery": "persistent_catalog_rebuild",
            "embedding": "stable_sha256_lexical_sketch_v1",
            "documents": self.document_count,
            "chunks": self.chunk_count,
            "license_claim": "MIT_README_BADGE",
            "license_file_present": license_file,
            "source_vendored": False,
        }

    def reset(self) -> None:
        """Replace all upstream stores so deleted documents cannot remain searchable."""

        with self._lock:
            self.graph_store = self._graph_type()
            self.relational_store = self._relational_type()
            self.vector_store = self._vector_type()
            self._chunk_ids.clear()
            self._indexed_attachments.clear()

    def index_document(
        self,
        *,
        attachment_id: str,
        name: str,
        media_type: str,
        text: str | None = None,
        document: ExtractedDocument | None = None,
    ) -> dict[str, object]:
        if (text is None) == (document is None):
            raise TypeError("provide exactly one of text or document")
        if document is None:
            if not isinstance(text, str):
                raise TypeError("text must be str")
            document = ExtractedDocument(text=text, page_count=0, pages=())
        elif not isinstance(document, ExtractedDocument):
            raise TypeError("document must be ExtractedDocument")
        chunks = _chunk_document(document)
        document_entity = f"document:{attachment_id}"
        with self._lock:
            if attachment_id in self._indexed_attachments:
                return {
                    "attachment_id": attachment_id,
                    "indexed": False,
                    "already_indexed": True,
                    "chunks": 0,
                }
            for chunk in chunks:
                entity = f"chunk:{attachment_id}:{chunk.chunk_index}"
                self.graph_store.add_edge(document_entity, entity, "contains")
                self.relational_store.insert(
                    entity,
                    {
                        "type": "LocalDocumentChunk",
                        "attachment_id": attachment_id,
                        "name": name,
                        "media_type": media_type,
                        "chunk_index": chunk.chunk_index,
                        "chunk": chunk.text,
                        "page_number": chunk.page_number,
                        "char_start": chunk.char_start,
                        "char_end": chunk.char_end,
                        "offset_basis": chunk.offset_basis,
                        "excerpt_sha256": chunk.excerpt_sha256,
                    },
                )
                self.vector_store.insert(entity, _stable_sha256_embedding(chunk.text))
                self._chunk_ids.add(entity)
            self._indexed_attachments.add(attachment_id)
        return {
            "attachment_id": attachment_id,
            "indexed": True,
            "already_indexed": False,
            "chunks": len(chunks),
            "page_count": document.page_count,
        }

    def source_preview(
        self, *, attachment_id: str, chunk_index: int
    ) -> dict[str, object]:
        """Return one exact indexed relational record without exposing internals."""

        if (
            not isinstance(attachment_id, str)
            or _ATTACHMENT_ID_RE.fullmatch(attachment_id) is None
        ):
            raise WorkspaceCapabilityError(
                "INVALID_ATTACHMENT_ID", "attachment_id is invalid"
            )
        if (
            not isinstance(chunk_index, int)
            or isinstance(chunk_index, bool)
            or not 0 <= chunk_index < MAX_RAG_CHUNKS_PER_DOCUMENT
        ):
            raise WorkspaceCapabilityError(
                "INVALID_CHUNK_INDEX", "chunk_index is invalid"
            )
        entity = f"chunk:{attachment_id}:{chunk_index}"
        with self._lock:
            if entity not in self._chunk_ids:
                raise WorkspaceCapabilityError(
                    "RAG_SOURCE_NOT_FOUND", "the indexed source was not found"
                )
            try:
                record = self.relational_store.get(entity)
            except Exception as exc:  # noqa: BLE001 - pinned adapter trust boundary
                raise WorkspaceCapabilityError(
                    "RAG_SOURCE_INTEGRITY_FAILED",
                    "the indexed source record could not be verified",
                ) from exc
        if not isinstance(record, Mapping):
            raise WorkspaceCapabilityError(
                "RAG_SOURCE_INTEGRITY_FAILED",
                "the indexed source record could not be verified",
            )
        text = record.get("chunk")
        page_number = record.get("page_number")
        char_start = record.get("char_start")
        char_end = record.get("char_end")
        offset_basis = record.get("offset_basis")
        excerpt_sha256 = record.get("excerpt_sha256")
        page_locator_valid = (
            offset_basis == "normalized_document_text_v1" and page_number is None
        ) or (
            offset_basis == "normalized_pdf_page_text_v1"
            and isinstance(page_number, int)
            and not isinstance(page_number, bool)
            and 1 <= page_number <= MAX_PDF_PAGES
        )
        if (
            record.get("type") != "LocalDocumentChunk"
            or record.get("attachment_id") != attachment_id
            or record.get("chunk_index") != chunk_index
            or not _attachment_name_is_safe(record.get("name"))
            or record.get("media_type") not in set(_ALLOWED_MEDIA_TYPES.values())
            or not isinstance(text, str)
            or not 1 <= len(text) <= RAG_CHUNK_CHARS
            or _normalize_index_source(text) != text
            or not isinstance(char_start, int)
            or isinstance(char_start, bool)
            or char_start < 0
            or not isinstance(char_end, int)
            or isinstance(char_end, bool)
            or char_end <= char_start
            or char_end - char_start != len(text)
            or not page_locator_valid
            or not isinstance(excerpt_sha256, str)
            or _SHA256_RE.fullmatch(excerpt_sha256) is None
            or excerpt_sha256 != sha256(text.encode("utf-8")).hexdigest()
        ):
            raise WorkspaceCapabilityError(
                "RAG_SOURCE_INTEGRITY_FAILED",
                "the indexed source record could not be verified",
            )
        return {
            "attachment_id": attachment_id,
            "chunk_index": chunk_index,
            "name": record["name"],
            "media_type": record["media_type"],
            "text": text,
            "page_number": page_number,
            "char_start": char_start,
            "char_end": char_end,
            "offset_basis": offset_basis,
            "excerpt_sha256": excerpt_sha256,
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
                        "page_number": record.get("page_number"),
                        "char_start": record.get("char_start"),
                        "char_end": record.get("char_end"),
                        "offset_basis": record.get("offset_basis"),
                        "excerpt_sha256": record.get("excerpt_sha256"),
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
        lens_client: LensApiClient | None = None,
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
        self.lens_client = lens_client or LensApiClient.from_environment({})
        if not isinstance(answer_integration_enabled, bool):
            raise TypeError("answer_integration_enabled must be bool")
        self.answer_integration_enabled = answer_integration_enabled
        self.storage_root = _prepare_storage_root(root)
        self.catalog_path = self.storage_root.parent / _CATALOG_FILENAME
        self.rag_quarantine_path = self.storage_root.parent / _RAG_QUARANTINE_FILENAME
        (
            self._persisted_valid_blobs,
            self._persisted_blob_count,
            self._persisted_blob_bytes,
        ) = _inventory_storage_root(self.storage_root)
        self._attachments, self._indexed_attachment_ids = _load_attachment_catalog(
            self.catalog_path, self.storage_root
        )
        self._lock = RLock()
        self._source_snapshots: dict[tuple[str, int], IndexedSourceSnapshot] = {}
        self._source_generation = 0
        self.akasicdb: AkasicDBAdapter | None = None
        self.akasicdb_error: tuple[str, str] | None = None
        try:
            rag_quarantine_present = _rag_quarantine_required_on_startup(
                self.rag_quarantine_path, self.catalog_path
            )
        except WorkspaceCapabilityError:
            # A malformed, truncated, oversized, or unsafe marker remains an
            # operator-review condition; startup must never auto-clear it.
            rag_quarantine_present = True
        selected_path = (
            None if rag_quarantine_present else self._discover_akasicdb(akasicdb_path)
        )
        if rag_quarantine_present:
            self.akasicdb_error = (
                _RAG_QUARANTINE_CODE,
                _RAG_QUARANTINE_MESSAGE,
            )
        elif selected_path is None:
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
        if self.akasicdb is not None and self._indexed_attachment_ids:
            try:
                self._rebuild_rag_index()
            except Exception:  # noqa: BLE001 - pinned adapter startup boundary
                self.akasicdb = None
                self.akasicdb_error = (
                    "AKASICDB_REINDEX_FAILED",
                    "the persistent local index could not be reconstructed",
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

    @staticmethod
    def _record_is_text_indexable(record: AttachmentRecord) -> bool:
        suffix = Path(record.name).suffix.casefold()
        if suffix == ".pdf":
            return _pdf_extractor_available()
        return record.text_indexable

    @classmethod
    def _record_payload(
        cls, record: AttachmentRecord, *, indexed: bool
    ) -> dict[str, object]:
        suffix = Path(record.name).suffix.casefold()
        text_indexable = cls._record_is_text_indexable(record)
        if suffix in _BINARY_MEDIA_TYPES and suffix != ".pdf":
            preview_kind = "image"
            preview_available = True
        elif suffix == ".pdf":
            preview_kind = "text"
            preview_available = _pdf_extractor_available()
        else:
            preview_kind = "text"
            preview_available = True
        return {
            **record.as_payload(),
            "text_indexable": text_indexable,
            "indexed": indexed,
            "preview_kind": preview_kind,
            "preview_available": preview_available,
        }

    def _verified_attachment_bytes(self, record: AttachmentRecord) -> bytes:
        try:
            resolved = record.stored_path.resolve(strict=True)
            if (
                record.stored_path.is_symlink()
                or not resolved.is_relative_to(self.storage_root)
                or not resolved.is_file()
                or resolved.stat().st_size != record.size_bytes
            ):
                raise WorkspaceCapabilityError(
                    "ATTACHMENT_INTEGRITY_FAILED",
                    "attachment changed after admission",
                )
            content = resolved.read_bytes()
        except WorkspaceCapabilityError:
            raise
        except OSError as exc:
            raise WorkspaceCapabilityError(
                "ATTACHMENT_INTEGRITY_FAILED",
                "attachment could not be read after admission",
            ) from exc
        if (
            len(content) != record.size_bytes
            or sha256(content).hexdigest() != record.sha256
        ):
            raise WorkspaceCapabilityError(
                "ATTACHMENT_INTEGRITY_FAILED",
                "attachment bytes no longer match the admitted digest",
            )
        return content

    def _read_attachment_document(self, record: AttachmentRecord) -> ExtractedDocument:
        suffix = Path(record.name).suffix.casefold()
        content = self._verified_attachment_bytes(record)
        if suffix == ".pdf":
            return _extract_pdf_document(content)
        if suffix not in _TEXT_MEDIA_TYPES:
            raise WorkspaceCapabilityError(
                "ATTACHMENT_NOT_INDEXABLE",
                "this attachment does not have a verified text extraction path",
            )
        try:
            text = content.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise WorkspaceCapabilityError(
                "INVALID_TEXT_ENCODING", "text attachments must be UTF-8"
            ) from exc
        return ExtractedDocument(text=text, page_count=0, pages=())

    def _read_attachment_text(self, record: AttachmentRecord) -> str:
        """Compatibility wrapper for previews and legacy internal callers."""

        return self._read_attachment_document(record).text

    def capability_payload(self) -> dict[str, object]:
        with self._lock:
            return self._capability_payload_locked()

    def _capability_payload_locked(self) -> dict[str, object]:
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
        web_search = self.web_policy.as_payload()
        lens_connector = {
            **self.lens_client.capability_payload(),
            "search_api": True,
            "patent_search": True,
            "scholarly_search": True,
            "citation_links": True,
            "provenance": True,
            "lens_to_akasicdb": self.akasicdb is not None,
            "lens_index_lifetime": "process_memory_until_local_index_rebuild",
        }
        web_search.update(
            {
                "execution": "official_lens_api_only",
                "executor_implemented": True,
                "official_lens_connector": lens_connector,
            }
        )
        for resource in ("lens_patent_search", "lens_scholarly_search"):
            selected = web_search.get(resource)
            if isinstance(selected, dict):
                selected.update(
                    {
                        "state": lens_connector["state"],
                        "network_executor_implemented": True,
                    }
                )
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
                "pdf_text_extraction": _pdf_extractor_available(),
                "pdf_text_extraction_backend": (
                    "pypdf" if _pdf_extractor_available() else None
                ),
                "pdf_max_pages": MAX_PDF_PAGES,
                "pdf_max_extracted_chars": MAX_PDF_EXTRACTED_CHARS,
                "pdf_process_isolation": True,
                "pdf_wall_timeout_seconds": PDF_EXTRACT_TIMEOUT_SECONDS,
                "pdf_cpu_limit_seconds": PDF_EXTRACT_CPU_LIMIT_SECONDS,
                "pdf_memory_limit_bytes": PDF_EXTRACT_MEMORY_LIMIT_BYTES,
                "preview_max_chars": MAX_ATTACHMENT_PREVIEW_CHARS,
                "catalog_lifetime": "local_filesystem_persistent",
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
            "web_search": web_search,
        }

    def list_attachments(self) -> dict[str, object]:
        with self._lock:
            items = [
                self._record_payload(
                    record,
                    indexed=record.attachment_id in self._indexed_attachment_ids,
                )
                for record in sorted(
                    self._attachments.values(), key=lambda item: item.created_at
                )
            ]
        return {"items": items, "count": len(items)}

    def add_attachment(
        self, *, name: str, media_type: str, content_base64: str
    ) -> dict[str, object]:
        if not _attachment_name_is_safe(name):
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
            self._raise_if_rag_quarantined_locked()
            existing = self._attachments.get(attachment_id)
            if existing is not None:
                return {
                    **self._record_payload(
                        existing,
                        indexed=attachment_id in self._indexed_attachment_ids,
                    ),
                    "duplicate": True,
                }
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
            try:
                _atomic_content_write(stored, content, digest)
            except WorkspaceCapabilityError:
                raise
            except (OSError, UnicodeError) as exc:
                raise WorkspaceCapabilityError(
                    "ATTACHMENT_STORE_FAILED",
                    "attachment blob could not be committed",
                ) from exc
            created = datetime.now(timezone.utc).isoformat()
            record = AttachmentRecord(
                attachment_id,
                name,
                media_type,
                len(content),
                digest,
                stored,
                created,
                suffix in _TEXT_MEDIA_TYPES or suffix == ".pdf",
            )
            self._attachments[attachment_id] = record
            try:
                self._persist_catalog()
            except WorkspaceCapabilityError as exc:
                self._attachments.pop(attachment_id, None)
                cleanup_error: OSError | None = None
                if not reuses_persisted_blob:
                    try:
                        stored.unlink()
                    except OSError as unlink_exc:
                        cleanup_error = unlink_exc
                try:
                    # `_atomic_json_write` can publish the replacement and then
                    # fail its postcondition check.  Re-publish the previous
                    # in-memory state so restart recovery observes the rollback.
                    self._persist_catalog()
                except WorkspaceCapabilityError as rollback_exc:
                    raise WorkspaceCapabilityError(
                        "ATTACHMENT_ROLLBACK_FAILED",
                        "attachment admission rollback requires operator review",
                    ) from rollback_exc
                (
                    self._persisted_valid_blobs,
                    self._persisted_blob_count,
                    self._persisted_blob_bytes,
                ) = _inventory_storage_root(self.storage_root)
                if cleanup_error is not None:
                    raise WorkspaceCapabilityError(
                        "ATTACHMENT_ROLLBACK_FAILED",
                        "attachment blob cleanup requires operator review",
                    ) from cleanup_error
                raise exc
            (
                self._persisted_valid_blobs,
                self._persisted_blob_count,
                self._persisted_blob_bytes,
            ) = _inventory_storage_root(self.storage_root)
        return {
            **self._record_payload(record, indexed=False),
            "duplicate": False,
        }

    def _catalog_payload(self) -> dict[str, object]:
        items: list[dict[str, object]] = []
        for record in sorted(
            self._attachments.values(),
            key=lambda item: (item.created_at, item.attachment_id),
        ):
            items.append(
                {
                    "attachment_id": record.attachment_id,
                    "name": record.name,
                    "media_type": record.media_type,
                    "size_bytes": record.size_bytes,
                    "sha256": record.sha256,
                    "created_at": record.created_at,
                    "blob_name": record.stored_path.name,
                    "text_indexable": record.text_indexable,
                }
            )
        return {
            "schema_version": _CATALOG_SCHEMA_VERSION,
            "items": items,
            "indexed_attachment_ids": sorted(self._indexed_attachment_ids),
        }

    def _persist_catalog(self) -> None:
        try:
            _atomic_json_write(self.catalog_path, self._catalog_payload())
        except WorkspaceCapabilityError:
            raise
        except (OSError, UnicodeError) as exc:
            raise WorkspaceCapabilityError(
                "ATTACHMENT_CATALOG_WRITE_FAILED",
                "attachment catalog could not be committed",
            ) from exc

    @staticmethod
    def _build_source_snapshots(
        records: Mapping[str, AttachmentRecord],
        documents: Mapping[str, ExtractedDocument],
    ) -> dict[tuple[str, int], IndexedSourceSnapshot]:
        """Bind every excerpt to the exact admitted blob used for indexing."""

        snapshots: dict[tuple[str, int], IndexedSourceSnapshot] = {}
        for attachment_id, record in records.items():
            document = documents.get(attachment_id)
            if document is None:
                raise WorkspaceCapabilityError(
                    "RAG_SOURCE_INTEGRITY_FAILED",
                    "the indexed source snapshot is incomplete",
                )
            for chunk in _chunk_document(document):
                key = (attachment_id, chunk.chunk_index)
                if key in snapshots:
                    raise WorkspaceCapabilityError(
                        "RAG_SOURCE_INTEGRITY_FAILED",
                        "the indexed source snapshot is ambiguous",
                    )
                snapshots[key] = IndexedSourceSnapshot(
                    attachment_id=attachment_id,
                    attachment_sha256=record.sha256,
                    name=record.name,
                    media_type=record.media_type,
                    chunk=chunk,
                )
        return snapshots

    def _quarantine_rag_locked(
        self,
        *,
        previous_attachments: Mapping[str, AttachmentRecord],
        previous_indexed_attachment_ids: set[str],
        previous_snapshots: Mapping[tuple[str, int], IndexedSourceSnapshot],
        previous_generation: int,
    ) -> None:
        """Fail closed after both a mutation and its rollback have failed."""

        self._activate_rag_quarantine_locked()
        self._attachments = dict(previous_attachments)
        self._indexed_attachment_ids = set(previous_indexed_attachment_ids)
        self._source_snapshots = dict(previous_snapshots)
        self._source_generation = previous_generation

    def _raise_if_rag_quarantined_locked(self) -> None:
        if (
            self.akasicdb is None
            and self.akasicdb_error is not None
            and self.akasicdb_error[0] == "RAG_OPERATOR_REVIEW_REQUIRED"
        ):
            code, message = self.akasicdb_error
            raise WorkspaceCapabilityError(code, message)

    def _activate_rag_quarantine_locked(self) -> None:
        self.akasicdb_error = (
            _RAG_QUARANTINE_CODE,
            _RAG_QUARANTINE_MESSAGE,
        )
        self.akasicdb = None

    def _begin_rag_transaction_locked(self) -> str:
        """Durably mark an ambiguous transaction before reset/unlink begins."""

        self._raise_if_rag_quarantined_locked()
        try:
            catalog_sha256 = _catalog_digest_for_rag_marker(self.catalog_path)
            _atomic_rag_quarantine_write(self.rag_quarantine_path, catalog_sha256)
        except Exception as exc:  # noqa: BLE001 - durable fail-closed boundary
            self._activate_rag_quarantine_locked()
            raise WorkspaceCapabilityError(
                _RAG_QUARANTINE_CODE,
                _RAG_QUARANTINE_MESSAGE,
            ) from exc
        return catalog_sha256

    def _complete_rag_transaction_locked(self, catalog_sha256: str) -> None:
        """Commit the post-outcome catalog before best-effort marker cleanup."""

        try:
            committed_catalog_sha256 = _catalog_digest_for_rag_marker(self.catalog_path)
            _commit_rag_transaction_marker(
                self.rag_quarantine_path,
                expected_pending_sha256=catalog_sha256,
                committed_catalog_sha256=committed_catalog_sha256,
            )
        except Exception as exc:  # noqa: BLE001 - durable fail-closed boundary
            self._activate_rag_quarantine_locked()
            raise WorkspaceCapabilityError(
                _RAG_QUARANTINE_CODE,
                _RAG_QUARANTINE_MESSAGE,
            ) from exc
        try:
            _cleanup_committed_rag_marker(
                self.rag_quarantine_path, committed_catalog_sha256
            )
        except Exception as exc:  # noqa: BLE001 - committed may remain or be absent
            try:
                remaining = _load_rag_quarantine_marker(self.rag_quarantine_path)
            except Exception as inspection_exc:  # noqa: BLE001
                self._activate_rag_quarantine_locked()
                raise WorkspaceCapabilityError(
                    _RAG_QUARANTINE_CODE,
                    _RAG_QUARANTINE_MESSAGE,
                ) from inspection_exc
            if remaining is None or (
                remaining.state == _RAG_TRANSACTION_COMMITTED
                and remaining.catalog_sha256 == committed_catalog_sha256
            ):
                return
            self._activate_rag_quarantine_locked()
            raise WorkspaceCapabilityError(
                _RAG_QUARANTINE_CODE,
                _RAG_QUARANTINE_MESSAGE,
            ) from exc

    def _rebuild_rag_index(self) -> None:
        with self._lock:
            if self.akasicdb is None:
                return
            records = {
                attachment_id: self._attachments[attachment_id]
                for attachment_id in sorted(self._indexed_attachment_ids)
            }
            documents = {
                attachment_id: self._read_attachment_document(record)
                for attachment_id, record in records.items()
            }
            snapshots = self._build_source_snapshots(records, documents)
            self.akasicdb.reset()
            for attachment_id, record in records.items():
                self.akasicdb.index_document(
                    attachment_id=record.attachment_id,
                    name=record.name,
                    media_type=record.media_type,
                    document=documents[attachment_id],
                )
            self._source_snapshots = snapshots
            self._source_generation += 1

    def delete_attachment(self, attachment_id: str) -> dict[str, object]:
        if (
            not isinstance(attachment_id, str)
            or _ATTACHMENT_ID_RE.fullmatch(attachment_id) is None
        ):
            raise WorkspaceCapabilityError(
                "INVALID_ATTACHMENT_ID", "attachment_id is invalid"
            )
        with self._lock:
            self._raise_if_rag_quarantined_locked()
            record = self._attachments.get(attachment_id)
            if record is None:
                raise WorkspaceCapabilityError(
                    "ATTACHMENT_NOT_FOUND", "attachment was not found"
                )
            # A successful response means both the catalog/index transaction and
            # the physical content-addressed blob deletion completed.  Read the
            # bounded blob before unlink so a later catalog/index failure can be
            # rolled back without leaving an inaccessible or misleading record.
            blob_content = self._verified_attachment_bytes(record)
            previous_attachments = dict(self._attachments)
            previous_indexed = set(self._indexed_attachment_ids)
            previous_snapshots = dict(self._source_snapshots)
            previous_generation = self._source_generation
            transaction_catalog_sha256 = self._begin_rag_transaction_locked()
            try:
                record.stored_path.unlink()
            except OSError as exc:
                self._complete_rag_transaction_locked(transaction_catalog_sha256)
                raise WorkspaceCapabilityError(
                    "ATTACHMENT_DELETE_FAILED",
                    "attachment blob could not be deleted",
                ) from exc

            self._attachments.pop(attachment_id)
            self._indexed_attachment_ids.discard(attachment_id)
            try:
                self._persist_catalog()
                self._rebuild_rag_index()
            except Exception as exc:  # noqa: BLE001 - transactional adapter boundary
                self._attachments[attachment_id] = record
                self._indexed_attachment_ids = previous_indexed
                try:
                    _atomic_content_write(
                        record.stored_path, blob_content, record.sha256
                    )
                    self._persist_catalog()
                    self._rebuild_rag_index()
                except Exception as rollback_exc:  # noqa: BLE001
                    self._quarantine_rag_locked(
                        previous_attachments=previous_attachments,
                        previous_indexed_attachment_ids=previous_indexed,
                        previous_snapshots=previous_snapshots,
                        previous_generation=previous_generation,
                    )
                    try:
                        (
                            self._persisted_valid_blobs,
                            self._persisted_blob_count,
                            self._persisted_blob_bytes,
                        ) = _inventory_storage_root(self.storage_root)
                    except WorkspaceCapabilityError:
                        # Preserve the stable rollback failure at this boundary.
                        # The durable marker already prevents restart promotion.
                        pass
                    raise WorkspaceCapabilityError(
                        "ATTACHMENT_DELETE_ROLLBACK_FAILED",
                        "attachment deletion rollback requires operator review",
                    ) from rollback_exc
                self._source_snapshots = previous_snapshots
                self._source_generation = previous_generation
                self._complete_rag_transaction_locked(transaction_catalog_sha256)
                raise WorkspaceCapabilityError(
                    "ATTACHMENT_DELETE_FAILED",
                    "attachment deletion could not be committed",
                ) from exc
            try:
                (
                    self._persisted_valid_blobs,
                    self._persisted_blob_count,
                    self._persisted_blob_bytes,
                ) = _inventory_storage_root(self.storage_root)
            except WorkspaceCapabilityError as exc:
                self._activate_rag_quarantine_locked()
                raise WorkspaceCapabilityError(
                    _RAG_QUARANTINE_CODE,
                    _RAG_QUARANTINE_MESSAGE,
                ) from exc
            self._complete_rag_transaction_locked(transaction_catalog_sha256)
        return {
            "attachment_id": attachment_id,
            "deleted": True,
            "index_removed": attachment_id in previous_indexed,
            "blob_deleted": True,
            "remaining": len(self._attachments),
            "indexed_documents": (
                self.akasicdb.document_count if self.akasicdb is not None else 0
            ),
        }

    def preview_attachment(self, attachment_id: str) -> dict[str, object]:
        """Return a bounded preview contract without exposing a host path."""

        if (
            not isinstance(attachment_id, str)
            or _ATTACHMENT_ID_RE.fullmatch(attachment_id) is None
        ):
            raise WorkspaceCapabilityError(
                "INVALID_ATTACHMENT_ID", "attachment_id is invalid"
            )
        with self._lock:
            record = self._attachments.get(attachment_id)
            if record is None:
                raise WorkspaceCapabilityError(
                    "ATTACHMENT_NOT_FOUND", "attachment was not found"
                )
            suffix = Path(record.name).suffix.casefold()
            base = {
                "attachment_id": record.attachment_id,
                "name": record.name,
                "media_type": record.media_type,
                "size_bytes": record.size_bytes,
            }
            if suffix == ".pdf":
                document = self._read_attachment_document(record)
                preview, truncated = _bounded_preview_text(document.text)
                return {
                    **base,
                    "kind": "text",
                    "text": preview,
                    "truncated": truncated,
                    "max_chars": MAX_ATTACHMENT_PREVIEW_CHARS,
                    "extraction": "pypdf",
                    "page_count": document.page_count,
                    "pages": [
                        {
                            "page_number": page.page_number,
                            "extracted_chars": len(page.text),
                        }
                        for page in document.pages
                    ],
                }
            if suffix in _TEXT_MEDIA_TYPES:
                preview, truncated = _bounded_preview_text(
                    self._read_attachment_text(record)
                )
                return {
                    **base,
                    "kind": "text",
                    "text": preview,
                    "truncated": truncated,
                    "max_chars": MAX_ATTACHMENT_PREVIEW_CHARS,
                    "extraction": "utf8",
                }
            if suffix in _BINARY_MEDIA_TYPES:
                # Verify the bytes now; the authenticated content route repeats
                # this check to close the preview/content time-of-check gap.
                self._verified_attachment_bytes(record)
                return {
                    **base,
                    "kind": "image",
                    "content_url": (
                        "/api/workspace/attachments/content?attachment_id="
                        + record.attachment_id
                    ),
                }
        raise WorkspaceCapabilityError(
            "ATTACHMENT_PREVIEW_UNAVAILABLE",
            "this attachment has no verified preview path",
        )

    def image_attachment_content(self, attachment_id: str) -> tuple[bytes, str]:
        """Read only admitted raster formats for the authenticated image route."""

        if (
            not isinstance(attachment_id, str)
            or _ATTACHMENT_ID_RE.fullmatch(attachment_id) is None
        ):
            raise WorkspaceCapabilityError(
                "INVALID_ATTACHMENT_ID", "attachment_id is invalid"
            )
        with self._lock:
            record = self._attachments.get(attachment_id)
            if record is None:
                raise WorkspaceCapabilityError(
                    "ATTACHMENT_NOT_FOUND", "attachment was not found"
                )
            if Path(record.name).suffix.casefold() not in {
                ".png",
                ".jpg",
                ".jpeg",
                ".webp",
            }:
                raise WorkspaceCapabilityError(
                    "ATTACHMENT_CONTENT_UNAVAILABLE",
                    "only admitted raster images have a content route",
                )
            return self._verified_attachment_bytes(record), record.media_type

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
        if len(set(attachment_ids)) != len(attachment_ids):
            raise WorkspaceCapabilityError(
                "INVALID_ATTACHMENT_IDS",
                "attachment_ids must be a unique bounded text list",
            )
        with self._lock:
            if self.akasicdb is None:
                code, message = self.akasicdb_error or (
                    "AKASICDB_UNAVAILABLE",
                    "AkasicDB is unavailable",
                )
                raise WorkspaceCapabilityError(code, message)
            records: dict[str, AttachmentRecord] = {}
            documents: dict[str, ExtractedDocument] = {}
            for attachment_id in attachment_ids:
                record = self._attachments.get(attachment_id)
                if record is None:
                    raise WorkspaceCapabilityError(
                        "ATTACHMENT_NOT_FOUND", "attachment was not found"
                    )
                if not self._record_is_text_indexable(record):
                    raise WorkspaceCapabilityError(
                        "ATTACHMENT_NOT_INDEXABLE",
                        "this attachment has no verified text extraction path",
                    )
                records[attachment_id] = record
                documents[attachment_id] = self._read_attachment_document(record)

            previous_attachments = dict(self._attachments)
            previous = set(self._indexed_attachment_ids)
            previous_snapshots = dict(self._source_snapshots)
            previous_generation = self._source_generation
            target = previous | set(attachment_ids)
            target_records = {item: self._attachments[item] for item in sorted(target)}
            target_documents = {
                item: (
                    documents[item]
                    if item in documents
                    else self._read_attachment_document(target_records[item])
                )
                for item in target_records
            }
            target_snapshots = self._build_source_snapshots(
                target_records, target_documents
            )
            transaction_catalog_sha256 = self._begin_rag_transaction_locked()
            try:
                self.akasicdb.reset()
                indexed_results: dict[str, dict[str, object]] = {}
                for item, record in target_records.items():
                    result = self.akasicdb.index_document(
                        attachment_id=record.attachment_id,
                        name=record.name,
                        media_type=record.media_type,
                        document=target_documents[item],
                    )
                    if item in records:
                        indexed_results[item] = result
                self._indexed_attachment_ids = target
                self._persist_catalog()
                self._source_snapshots = target_snapshots
                self._source_generation += 1
            except Exception as exc:  # noqa: BLE001 - transactional adapter boundary
                self._indexed_attachment_ids = previous
                try:
                    self._persist_catalog()
                    self._rebuild_rag_index()
                except Exception as rollback_exc:  # noqa: BLE001
                    self._quarantine_rag_locked(
                        previous_attachments=previous_attachments,
                        previous_indexed_attachment_ids=previous,
                        previous_snapshots=previous_snapshots,
                        previous_generation=previous_generation,
                    )
                    raise WorkspaceCapabilityError(
                        "RAG_INDEX_ROLLBACK_FAILED",
                        "the local RAG rollback requires operator review",
                    ) from rollback_exc
                self._source_snapshots = previous_snapshots
                self._source_generation = previous_generation
                self._complete_rag_transaction_locked(transaction_catalog_sha256)
                if isinstance(exc, WorkspaceCapabilityError):
                    raise
                raise WorkspaceCapabilityError(
                    "RAG_INDEX_FAILED", "the local RAG index could not be committed"
                ) from exc
            self._complete_rag_transaction_locked(transaction_catalog_sha256)
            results = [indexed_results[item] for item in attachment_ids]
        return {
            "engine": "AkasicDB",
            "results": results,
            "documents": self.akasicdb.document_count,
            "chunks": self.akasicdb.chunk_count,
            "answer_integration": self.answer_integration_enabled,
        }

    def reindex_attachments(self, attachment_ids: list[str]) -> dict[str, object]:
        """Force a bounded rebuild while retaining every previously indexed file."""

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
            or len(set(attachment_ids)) != len(attachment_ids)
        ):
            raise WorkspaceCapabilityError(
                "INVALID_ATTACHMENT_IDS",
                "attachment_ids must be a unique bounded text list",
            )
        with self._lock:
            if self.akasicdb is None:
                code, message = self.akasicdb_error or (
                    "AKASICDB_UNAVAILABLE",
                    "AkasicDB is unavailable",
                )
                raise WorkspaceCapabilityError(code, message)
            requested: list[AttachmentRecord] = []
            for attachment_id in attachment_ids:
                record = self._attachments.get(attachment_id)
                if record is None:
                    raise WorkspaceCapabilityError(
                        "ATTACHMENT_NOT_FOUND", "attachment was not found"
                    )
                if not self._record_is_text_indexable(record):
                    raise WorkspaceCapabilityError(
                        "ATTACHMENT_NOT_INDEXABLE",
                        "this attachment has no verified text extraction path",
                    )
                requested.append(record)

            previous_attachments = dict(self._attachments)
            previous = set(self._indexed_attachment_ids)
            previous_snapshots = dict(self._source_snapshots)
            previous_generation = self._source_generation
            target = previous | set(attachment_ids)
            records = {item: self._attachments[item] for item in sorted(target)}
            # Read, integrity-check, extract, and chunk every source before the
            # active in-memory index is reset.
            documents = {
                item: self._read_attachment_document(record)
                for item, record in records.items()
            }
            chunk_counts = {
                item: len(_chunk_document(document))
                for item, document in documents.items()
            }
            target_snapshots = self._build_source_snapshots(records, documents)
            transaction_catalog_sha256 = self._begin_rag_transaction_locked()
            try:
                self.akasicdb.reset()
                for item, record in records.items():
                    self.akasicdb.index_document(
                        attachment_id=record.attachment_id,
                        name=record.name,
                        media_type=record.media_type,
                        document=documents[item],
                    )
                self._indexed_attachment_ids = target
                self._persist_catalog()
                self._source_snapshots = target_snapshots
                self._source_generation += 1
            except Exception as exc:  # noqa: BLE001 - transactional adapter boundary
                self._indexed_attachment_ids = previous
                try:
                    self._persist_catalog()
                    self._rebuild_rag_index()
                except Exception as rollback_exc:  # noqa: BLE001
                    self._quarantine_rag_locked(
                        previous_attachments=previous_attachments,
                        previous_indexed_attachment_ids=previous,
                        previous_snapshots=previous_snapshots,
                        previous_generation=previous_generation,
                    )
                    raise WorkspaceCapabilityError(
                        "RAG_REINDEX_ROLLBACK_FAILED",
                        "the local RAG rollback requires operator review",
                    ) from rollback_exc
                self._source_snapshots = previous_snapshots
                self._source_generation = previous_generation
                self._complete_rag_transaction_locked(transaction_catalog_sha256)
                raise WorkspaceCapabilityError(
                    "RAG_REINDEX_FAILED", "the local RAG rebuild could not be committed"
                ) from exc
            self._complete_rag_transaction_locked(transaction_catalog_sha256)
        return {
            "engine": "AkasicDB",
            "reindexed_attachment_ids": list(attachment_ids),
            "results": [
                {
                    "attachment_id": record.attachment_id,
                    "chunks": chunk_counts[record.attachment_id],
                    "reindexed": record.attachment_id in previous,
                }
                for record in requested
            ],
            "documents": self.akasicdb.document_count,
            "chunks": self.akasicdb.chunk_count,
            "answer_integration": self.answer_integration_enabled,
        }

    def query_rag(self, query: str, *, limit: int = 5) -> dict[str, object]:
        with self._lock:
            if self.akasicdb is None:
                code, message = self.akasicdb_error or (
                    "AKASICDB_UNAVAILABLE",
                    "AkasicDB is unavailable",
                )
                raise WorkspaceCapabilityError(code, message)
            try:
                result = self.akasicdb.query(query, limit=limit)
            except WorkspaceCapabilityError:
                raise
            except Exception as exc:  # noqa: BLE001 - pinned adapter query boundary
                raise WorkspaceCapabilityError(
                    "RAG_QUERY_FAILED", "the local RAG query could not be completed"
                ) from exc
            if (
                not isinstance(result, Mapping)
                or set(result) != {"query", "results", "count"}
                or result.get("query") != query
                or not isinstance(result.get("results"), list)
                or not isinstance(result.get("count"), int)
                or isinstance(result.get("count"), bool)
                or result.get("count") != len(result["results"])
            ):
                raise WorkspaceCapabilityError(
                    "RAG_QUERY_INTEGRITY_FAILED",
                    "the local RAG query returned an invalid result envelope",
                )

            verified_results: list[dict[str, object]] = []
            for candidate in result["results"]:
                if not isinstance(candidate, Mapping):
                    raise WorkspaceCapabilityError(
                        "RAG_QUERY_INTEGRITY_FAILED",
                        "the local RAG query returned a malformed candidate",
                    )
                attachment_id = candidate.get("attachment_id")
                if not isinstance(attachment_id, str):
                    raise WorkspaceCapabilityError(
                        "RAG_QUERY_INTEGRITY_FAILED",
                        "the local RAG query returned a malformed candidate",
                    )
                if attachment_id not in self._indexed_attachment_ids:
                    # Lens entries are transient search material. Without an
                    # immutable local snapshot they cannot ground an answer.
                    continue
                chunk_index = candidate.get("chunk_index")
                score = candidate.get("score")
                if (
                    not isinstance(chunk_index, int)
                    or isinstance(chunk_index, bool)
                    or not isinstance(score, (int, float))
                    or isinstance(score, bool)
                    or not isfinite(float(score))
                    or not 0.0 < float(score) <= 1.00000001
                ):
                    raise WorkspaceCapabilityError(
                        "RAG_QUERY_INTEGRITY_FAILED",
                        "the local RAG query result did not match its source snapshot",
                    )
                snapshot = self._source_snapshots.get((attachment_id, chunk_index))
                record = self._attachments.get(attachment_id)
                if snapshot is None or record is None:
                    raise WorkspaceCapabilityError(
                        "RAG_QUERY_INTEGRITY_FAILED",
                        "the local RAG query result did not match its source snapshot",
                    )
                authority = snapshot.as_payload()
                if set(candidate) != set(authority) | {"score"}:
                    raise WorkspaceCapabilityError(
                        "RAG_QUERY_INTEGRITY_FAILED",
                        "the local RAG query result did not match its source snapshot",
                    )
                returned_authority = {key: candidate[key] for key in authority}
                if not snapshot.matches(record, returned_authority):
                    raise WorkspaceCapabilityError(
                        "RAG_QUERY_INTEGRITY_FAILED",
                        "the local RAG query result did not match its source snapshot",
                    )
                verified_results.append({**authority, "score": float(score)})

            return {
                "engine": "AkasicDB",
                "embedding": "stable_sha256_lexical_sketch_v1",
                "answer_integration": self.answer_integration_enabled,
                "query": query,
                "results": verified_results,
                "count": len(verified_results),
            }

    def preview_rag_source(
        self, attachment_id: str, chunk_index: int
    ) -> dict[str, object]:
        """Return one O(1), immutable source snapshot under a short read lock."""

        if self.akasicdb is None:
            code, message = self.akasicdb_error or (
                "AKASICDB_UNAVAILABLE",
                "AkasicDB is unavailable",
            )
            raise WorkspaceCapabilityError(code, message)
        if (
            not isinstance(attachment_id, str)
            or _ATTACHMENT_ID_RE.fullmatch(attachment_id) is None
        ):
            raise WorkspaceCapabilityError(
                "INVALID_ATTACHMENT_ID", "attachment_id is invalid"
            )
        if (
            not isinstance(chunk_index, int)
            or isinstance(chunk_index, bool)
            or not 0 <= chunk_index < MAX_RAG_CHUNKS_PER_DOCUMENT
        ):
            raise WorkspaceCapabilityError(
                "INVALID_CHUNK_INDEX", "chunk_index is invalid"
            )
        with self._lock:
            if self.akasicdb is None:
                code, message = self.akasicdb_error or (
                    "AKASICDB_UNAVAILABLE",
                    "AkasicDB is unavailable",
                )
                raise WorkspaceCapabilityError(code, message)
            record = self._attachments.get(attachment_id)
            if record is None or attachment_id not in self._indexed_attachment_ids:
                raise WorkspaceCapabilityError(
                    "RAG_SOURCE_NOT_FOUND", "the indexed source was not found"
                )
            snapshot = self._source_snapshots.get((attachment_id, chunk_index))
            if snapshot is None:
                raise WorkspaceCapabilityError(
                    "RAG_SOURCE_NOT_FOUND", "the indexed source was not found"
                )
            indexed = self.akasicdb.source_preview(
                attachment_id=attachment_id, chunk_index=chunk_index
            )
            # The snapshot was created from the same digest-verified bytes as
            # the committed index.  The adapter remains checked here so an
            # in-memory record mutation cannot become browser-visible authority.
            if not snapshot.matches(record, indexed):
                raise WorkspaceCapabilityError(
                    "RAG_SOURCE_INTEGRITY_FAILED",
                    "the indexed source no longer matches the admitted attachment",
                )
            return {"schema_version": 1, **snapshot.as_payload()}

    def search_lens(
        self,
        kind: LensSearchKind | str,
        query: str,
        *,
        limit: int = 5,
        index_in_akasicdb: bool = False,
    ) -> dict[str, object]:
        """Search official Lens resources and optionally index normalized evidence."""

        if not isinstance(index_in_akasicdb, bool):
            raise WorkspaceCapabilityError(
                "INVALID_LENS_INDEX_MODE", "Lens index mode must be boolean"
            )
        try:
            if not index_in_akasicdb:
                return self.lens_client.search(kind, query, limit=limit).as_payload()
            if self.akasicdb is None:
                code, message = self.akasicdb_error or (
                    "AKASICDB_UNAVAILABLE",
                    "AkasicDB is unavailable",
                )
                raise WorkspaceCapabilityError(code, message)
            result = LensAkasicBridge(self.lens_client, self.akasicdb).search_and_index(
                kind, query, limit=limit
            )
            return {
                **result,
                "index_engine": "AkasicDB",
                "index_lifetime": "process_memory_until_local_index_rebuild",
                "provenance_embedded": True,
            }
        except LensApiError as exc:
            raise WorkspaceCapabilityError(exc.code, str(exc)) from exc

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
    "ExtractedDocument",
    "ExtractedPage",
    "IndexedChunk",
    "MAX_ATTACHMENT_BYTES",
    "MAX_ATTACHMENT_PREVIEW_CHARS",
    "MAX_ATTACHMENT_TOTAL_BYTES",
    "MAX_JSON_ATTACHMENT_BYTES",
    "MAX_JSON_NESTING",
    "MAX_PDF_EXTRACTED_CHARS",
    "MAX_PDF_PAGES",
    "MAX_RAG_RESULTS",
    "VerifiedModelMetadata",
    "WebAccessPolicy",
    "WorkspaceCapabilityError",
    "WorkspaceCapabilityService",
    "web_policy_from_environment",
]
