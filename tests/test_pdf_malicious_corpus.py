from __future__ import annotations

from base64 import b64encode
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from cogni_demo import pdf_extract_worker, workspace_capabilities as capabilities
from cogni_demo.workspace_capabilities import (
    AKASICDB_AUDITED_DIGESTS,
    MAX_ATTACHMENT_BYTES,
    MAX_PDF_EXTRACTED_CHARS,
    MAX_PDF_PAGES,
    PDF_EXTRACT_CPU_LIMIT_SECONDS,
    PDF_EXTRACT_MEMORY_LIMIT_BYTES,
    PDF_EXTRACT_TIMEOUT_SECONDS,
    WorkspaceCapabilityError,
    WorkspaceCapabilityService,
)
from tests.test_workspace_capabilities import (
    _model,
    _pdf_bytes,
    _pdf_pages_bytes,
    _write_clone,
)


def _encrypted_pdf_bytes() -> bytes:
    from pypdf import PdfWriter

    writer = PdfWriter()
    writer.add_blank_page(width=612, height=792)
    writer.encrypt("bounded-local-secret")
    output = BytesIO()
    writer.write(output)
    return output.getvalue()


def _truncated_pdf_bytes() -> bytes:
    valid = _pdf_bytes("this valid source is deliberately truncated")
    # Remove the xref/trailer while retaining a valid PDF signature. This is
    # an actual malformed file, not a mocked parser exception.
    return valid[: max(len(b"%PDF-1.4\n"), len(valid) - 96)]


class TestMaliciousPdfCorpus(unittest.TestCase):
    def setUp(self) -> None:
        if capabilities._PdfReader is None:
            self.skipTest("optional pypdf runtime is unavailable")

    def _fixture_roots(self, root: Path) -> tuple[Path, dict[str, str]]:
        project = root / "project"
        clone = root / "AkasicDB"
        project.mkdir()
        clone.mkdir()
        digests = _write_clone(clone)
        return clone, digests

    @staticmethod
    def _admit_pdf(
        service: WorkspaceCapabilityService, name: str, content: bytes
    ) -> dict[str, object]:
        return service.add_attachment(
            name=name,
            media_type="application/pdf",
            content_base64=b64encode(content).decode("ascii"),
        )

    @staticmethod
    def _stable_public_snapshot(
        service: WorkspaceCapabilityService, baseline_id: str
    ) -> dict[str, object]:
        query = service.query_rag("baseline equilibrium evidence")
        source = service.preview_rag_source(baseline_id, 0)
        rag = service.capability_payload()["rag"]
        return {
            "attachments": service.list_attachments(),
            "query": query,
            "source": source,
            "rag_documents": rag["documents"],
            "rag_chunks": rag["chunks"],
        }

    def _assert_bounded_error(
        self,
        action,
        *,
        expected_code: str,
        forbidden_root: Path,
    ) -> None:
        with self.assertRaises(WorkspaceCapabilityError) as captured:
            action()
        self.assertEqual(captured.exception.code, expected_code)
        rendered = str(captured.exception)
        self.assertLessEqual(len(rendered), 256)
        self.assertNotIn(str(forbidden_root), rendered)
        self.assertNotIn("Traceback", rendered)
        self.assertNotIn("pypdf", rendered.casefold())
        self.assertNotIn("PdfReader", rendered)

    def test_actual_malicious_pdfs_fail_closed_without_index_or_source_mutation(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            clone, digests = self._fixture_roots(root)
            with patch.dict(AKASICDB_AUDITED_DIGESTS, digests, clear=True):
                # Recreate under the pinned digest patch so every construction,
                # including restart, uses the real adapter verification path.
                service = WorkspaceCapabilityService(
                    root / "project", _model(), akasicdb_path=clone
                )
                baseline = self._admit_pdf(
                    service,
                    "baseline.pdf",
                    _pdf_bytes("baseline equilibrium evidence"),
                )
                baseline_id = str(baseline["attachment_id"])
                service.index_attachments([baseline_id])

                corpus = (
                    (
                        "encrypted.pdf",
                        _encrypted_pdf_bytes(),
                        "PDF_ENCRYPTED",
                    ),
                    (
                        "truncated.pdf",
                        _truncated_pdf_bytes(),
                        "PDF_TEXT_EXTRACTION_FAILED",
                    ),
                    (
                        "textless.pdf",
                        _pdf_pages_bytes((None,)),
                        "PDF_NO_EXTRACTABLE_TEXT",
                    ),
                    (
                        "page-limit.pdf",
                        _pdf_pages_bytes((None,) * (MAX_PDF_PAGES + 1)),
                        "PDF_PAGE_LIMIT",
                    ),
                )

                for name, content, expected_code in corpus:
                    with self.subTest(corpus=name):
                        admitted = self._admit_pdf(service, name, content)
                        malicious_id = str(admitted["attachment_id"])
                        self.assertFalse(admitted["indexed"])
                        # Admission is intentionally artifact-only. The exact
                        # admitted catalog is the stable pre-index state.
                        admitted_snapshot = self._stable_public_snapshot(
                            service, baseline_id
                        )
                        for operation in (
                            service.preview_attachment,
                            lambda selected: service.index_attachments([selected]),
                            lambda selected: service.reindex_attachments([selected]),
                        ):
                            self._assert_bounded_error(
                                lambda op=operation: op(malicious_id),
                                expected_code=expected_code,
                                forbidden_root=root,
                            )
                            self.assertEqual(
                                self._stable_public_snapshot(service, baseline_id),
                                admitted_snapshot,
                            )

                stable_before_restart = self._stable_public_snapshot(
                    service, baseline_id
                )
                restarted = WorkspaceCapabilityService(
                    root / "project", _model(), akasicdb_path=clone
                )
                self.assertEqual(
                    self._stable_public_snapshot(restarted, baseline_id),
                    stable_before_restart,
                )

    def test_oversized_pdf_is_rejected_before_parser_or_catalog_mutation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            service = WorkspaceCapabilityService(root, _model())
            oversized = b"%PDF-1.7\n" + b"x" * (
                MAX_ATTACHMENT_BYTES + 1 - len(b"%PDF-1.7\n")
            )
            before = service.list_attachments()
            with patch.object(
                capabilities,
                "_run_pdf_extractor_document",
                side_effect=AssertionError("parser must not run"),
            ):
                self._assert_bounded_error(
                    lambda: self._admit_pdf(service, "oversized.pdf", oversized),
                    expected_code="ATTACHMENT_TOO_LARGE",
                    forbidden_root=root,
                )
            self.assertEqual(service.list_attachments(), before)
            self.assertFalse(any(service.storage_root.iterdir()))

    def test_blank_pages_preserve_physical_page_three_source_across_rebuilds(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            clone, digests = self._fixture_roots(root)
            content = _pdf_pages_bytes(
                (
                    "unrelated decoy",
                    None,
                    "unique malicious corpus evidence on physical page three",
                )
            )
            with patch.dict(AKASICDB_AUDITED_DIGESTS, digests, clear=True):
                service = WorkspaceCapabilityService(
                    root / "project", _model(), akasicdb_path=clone
                )
                admitted = self._admit_pdf(service, "physical-pages.pdf", content)
                attachment_id = str(admitted["attachment_id"])
                service.index_attachments([attachment_id])

                def source_snapshot(
                    selected: WorkspaceCapabilityService,
                ) -> dict[str, object]:
                    result = selected.query_rag("unique physical page three")
                    self.assertEqual(result["count"], 1)
                    source = result["results"][0]
                    self.assertEqual(source["page_number"], 3)
                    self.assertEqual(source["chunk_index"], 1)
                    self.assertEqual(
                        source["offset_basis"], "normalized_pdf_page_text_v1"
                    )
                    self.assertEqual(
                        source["excerpt_sha256"],
                        sha256(source["text"].encode()).hexdigest(),
                    )
                    exact = selected.preview_rag_source(attachment_id, 1)
                    for key, value in exact.items():
                        if key != "schema_version":
                            self.assertEqual(source[key], value)
                    self.assertNotIn(str(root), json.dumps(exact))
                    return exact

                expected = source_snapshot(service)
                restarted = WorkspaceCapabilityService(
                    root / "project", _model(), akasicdb_path=clone
                )
                self.assertEqual(source_snapshot(restarted), expected)
                restarted.reindex_attachments([attachment_id])
                self.assertEqual(source_snapshot(restarted), expected)

    def test_pdf_resource_limits_are_published_and_match_worker_contract(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            payload = WorkspaceCapabilityService(
                temporary, _model()
            ).capability_payload()["attachments"]
        self.assertTrue(payload["pdf_process_isolation"])
        self.assertEqual(payload["max_bytes_each"], MAX_ATTACHMENT_BYTES)
        self.assertEqual(payload["pdf_max_pages"], MAX_PDF_PAGES)
        self.assertEqual(payload["pdf_max_extracted_chars"], MAX_PDF_EXTRACTED_CHARS)
        self.assertEqual(
            payload["pdf_wall_timeout_seconds"], PDF_EXTRACT_TIMEOUT_SECONDS
        )
        self.assertEqual(
            payload["pdf_cpu_limit_seconds"], PDF_EXTRACT_CPU_LIMIT_SECONDS
        )
        self.assertEqual(
            payload["pdf_memory_limit_bytes"], PDF_EXTRACT_MEMORY_LIMIT_BYTES
        )
        self.assertEqual(pdf_extract_worker.MAX_INPUT_BYTES, MAX_ATTACHMENT_BYTES)
        self.assertEqual(pdf_extract_worker.MAX_PAGES, MAX_PDF_PAGES)
        self.assertEqual(pdf_extract_worker.MAX_TEXT_CHARS, MAX_PDF_EXTRACTED_CHARS)
        self.assertEqual(
            pdf_extract_worker.CPU_LIMIT_SECONDS, PDF_EXTRACT_CPU_LIMIT_SECONDS
        )
        self.assertEqual(
            pdf_extract_worker.MEMORY_LIMIT_BYTES,
            PDF_EXTRACT_MEMORY_LIMIT_BYTES,
        )


if __name__ == "__main__":
    unittest.main()
