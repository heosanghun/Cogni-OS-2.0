from __future__ import annotations

from io import BytesIO
import json
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

from cogni_demo import pdf_extract_worker as worker


class _BinaryInput:
    def __init__(self, content: bytes) -> None:
        self.buffer = BytesIO(content)


class _BinaryOutput:
    def __init__(self) -> None:
        self.buffer = BytesIO()


class _Page:
    def __init__(self, text: str, *, must_not_run: bool = False) -> None:
        self.text = text
        self.must_not_run = must_not_run
        self.calls = 0

    def extract_text(self) -> str:
        self.calls += 1
        if self.must_not_run:
            raise AssertionError("worker continued after the text limit")
        return self.text


class PdfExtractWorkerTests(unittest.TestCase):
    def test_text_limit_stops_before_extracting_later_pages(self) -> None:
        first = _Page("aaaa")
        overflowing = _Page("bbbb")
        later = _Page("never", must_not_run=True)
        reader = SimpleNamespace(
            is_encrypted=False,
            pages=[first, overflowing, later],
        )
        fake_pypdf = ModuleType("pypdf")
        fake_pypdf.PdfReader = (  # type: ignore[attr-defined]
            lambda _stream, strict=True: reader
        )
        stdin = _BinaryInput(b"%PDF-1.4\nlocal")
        stdout = _BinaryOutput()

        with (
            patch.object(worker, "_apply_posix_limits", return_value=True),
            patch.object(worker, "MAX_TEXT_CHARS", 9),
            patch.dict(sys.modules, {"pypdf": fake_pypdf}),
            patch.object(sys, "stdin", stdin),
            patch.object(sys, "stdout", stdout),
        ):
            exit_code = worker.main()

        payload = json.loads(stdout.buffer.getvalue().decode("utf-8"))
        self.assertEqual(exit_code, 2)
        self.assertEqual(payload["code"], "PDF_TEXT_LIMIT")
        self.assertEqual(first.calls, 1)
        self.assertEqual(overflowing.calls, 1)
        self.assertEqual(later.calls, 0)


if __name__ == "__main__":
    unittest.main()
