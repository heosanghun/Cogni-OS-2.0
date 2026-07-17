"""Isolated, bounded PDF-to-text worker for the local CogniBoard service.

The HTTP process never imports an untrusted PDF into its own parser.  This
standalone worker receives one bounded PDF on stdin and returns one bounded
JSON result on stdout.  The parent adds a Windows Job Object memory/CPU cap;
POSIX workers install equivalent rlimits before importing pypdf.
"""

from __future__ import annotations

import json
import os
import sys


MAX_INPUT_BYTES = 8 * 1024 * 1024
MAX_PAGES = 128
MAX_TEXT_CHARS = 256_000
MEMORY_LIMIT_BYTES = 256 * 1024 * 1024
CPU_LIMIT_SECONDS = 6


def _emit(payload: dict[str, object], exit_code: int) -> int:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    sys.stdout.buffer.write(encoded)
    sys.stdout.buffer.flush()
    return exit_code


def _fail(code: str, message: str) -> int:
    return _emit({"ok": False, "code": code, "message": message}, 2)


def _apply_posix_limits() -> bool:
    if os.name != "posix":
        return True
    try:
        import resource

        resource.setrlimit(
            resource.RLIMIT_AS,
            (MEMORY_LIMIT_BYTES, MEMORY_LIMIT_BYTES),
        )
        resource.setrlimit(
            resource.RLIMIT_CPU,
            (CPU_LIMIT_SECONDS, CPU_LIMIT_SECONDS),
        )
    except (ImportError, OSError, ValueError):
        return False
    return True


def _normalize(value: str) -> str:
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


def main() -> int:
    if not _apply_posix_limits():
        return _fail("PDF_SANDBOX_SETUP_FAILED", "resource limits are unavailable")
    content = sys.stdin.buffer.read(MAX_INPUT_BYTES + 1)
    if not 1 <= len(content) <= MAX_INPUT_BYTES:
        return _fail("ATTACHMENT_TOO_LARGE", "PDF input exceeds its byte limit")
    try:
        from io import BytesIO

        from pypdf import PdfReader

        reader = PdfReader(BytesIO(content), strict=True)
        if reader.is_encrypted:
            return _fail("PDF_ENCRYPTED", "encrypted PDFs are not accepted")
        page_count = len(reader.pages)
        if not 1 <= page_count <= MAX_PAGES:
            return _fail("PDF_PAGE_LIMIT", "PDF page count exceeds its limit")
        pages: list[dict[str, object]] = []
        total_chars = 0
        for page_number, page in enumerate(reader.pages, start=1):
            raw = page.extract_text()
            if raw is None:
                raw = ""
            if not isinstance(raw, str):
                return _fail(
                    "PDF_TEXT_EXTRACTION_FAILED",
                    "the PDF extractor returned invalid text",
                )
            text = _normalize(raw)
            projected_total = total_chars + (2 if pages else 0) + len(text)
            if projected_total > MAX_TEXT_CHARS:
                return _fail(
                    "PDF_TEXT_LIMIT",
                    "extracted PDF text exceeds its limit",
                )
            total_chars = projected_total
            pages.append({"page_number": page_number, "text": text})
    except Exception:  # noqa: BLE001 - parser errors never cross the boundary
        return _fail("PDF_TEXT_EXTRACTION_FAILED", "the PDF parser rejected the file")
    if not any(page["text"] for page in pages):
        return _fail("PDF_NO_EXTRACTABLE_TEXT", "the PDF has no extractable text")
    return _emit(
        {
            "ok": True,
            "pages": pages,
        },
        0,
    )


if __name__ == "__main__":
    raise SystemExit(main())
