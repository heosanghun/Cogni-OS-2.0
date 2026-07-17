from __future__ import annotations

from base64 import b64encode
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from cogni_demo import workspace_capabilities as capabilities
from cogni_demo.workspace_capabilities import (
    AKASICDB_AUDITED_DIGESTS,
    AKASICDB_AUDITED_REVISION,
    AkasicDBAdapter,
    MAX_ATTACHMENT_BASE64_CHARS,
    MAX_ATTACHMENT_COUNT,
    MAX_ATTACHMENT_TOTAL_BYTES,
    MAX_JSON_ATTACHMENT_BYTES,
    MAX_JSON_NESTING,
    VerifiedModelMetadata,
    WebAccessPolicy,
    WorkspaceCapabilityError,
    WorkspaceCapabilityService,
)


GRAPH_MODULE = """class GraphStore:
    def __init__(self):
        self.edges = []
    def add_edge(self, source, target, relation):
        self.edges.append((source, target, relation))
"""
RELATIONAL_MODULE = """class RelationalStore:
    def __init__(self):
        self.records = {}
    def insert(self, entity_id, properties):
        self.records[entity_id] = properties
    def get(self, entity_id):
        return self.records.get(entity_id, {})
"""
VECTOR_MODULE = """import math
class VectorStore:
    def __init__(self):
        self.vectors = {}
    def insert(self, entity_id, vector):
        self.vectors[entity_id] = vector
    def similarity_search(self, query_vector, top_k=5):
        rows = []
        for entity_id, vector in self.vectors.items():
            dot = sum(a * b for a, b in zip(query_vector, vector))
            left = math.sqrt(sum(a * a for a in query_vector))
            right = math.sqrt(sum(b * b for b in vector))
            rows.append((entity_id, 0.0 if not left or not right else dot / (left * right)))
        return sorted(rows, key=lambda row: row[1], reverse=True)[:top_k]
"""


def _model() -> VerifiedModelMetadata:
    return VerifiedModelMetadata(
        model_id="gemma4-e4b:test",
        label="dense gemma4-e4b-it",
        architecture="Gemma4ForConditionalGeneration",
        manifest_sha256="a" * 64,
        config_sha256="b" * 64,
        checkpoint_modalities=("text", "image", "audio", "video"),
        runtime_input_modalities=("text",),
    )


def _write_clone(root: Path) -> dict[str, str]:
    modules = {
        "akasic/storage/graph_store.py": GRAPH_MODULE,
        "akasic/storage/relational_store.py": RELATIONAL_MODULE,
        "akasic/storage/vector_store.py": VECTOR_MODULE,
    }
    digests: dict[str, str] = {}
    for relative, source in modules.items():
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(source, encoding="utf-8", newline="")
        digests[relative] = sha256(target.read_bytes()).hexdigest()
    git = root / ".git"
    git.mkdir()
    (git / "HEAD").write_text(AKASICDB_AUDITED_REVISION, encoding="ascii")
    return digests


def _pdf_pages_bytes(page_texts: tuple[str | None, ...]) -> bytes:
    from pypdf import PdfWriter
    from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject

    writer = PdfWriter()
    font = DictionaryObject(
        {
            NameObject("/Type"): NameObject("/Font"),
            NameObject("/Subtype"): NameObject("/Type1"),
            NameObject("/BaseFont"): NameObject("/Helvetica"),
        }
    )
    font_reference = writer._add_object(font)
    for text in page_texts:
        page = writer.add_blank_page(width=612, height=792)
        if text is None:
            continue
        page[NameObject("/Resources")] = DictionaryObject(
            {NameObject("/Font"): DictionaryObject({NameObject("/F1"): font_reference})}
        )
        stream = DecodedStreamObject()
        escaped = text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
        stream.set_data(f"BT /F1 12 Tf 72 720 Td ({escaped}) Tj ET".encode("ascii"))
        page[NameObject("/Contents")] = writer._add_object(stream)
    output = BytesIO()
    writer.write(output)
    return output.getvalue()


def _pdf_bytes(text: str) -> bytes:
    return _pdf_pages_bytes((text,))


class TestWorkspaceCapabilities(unittest.TestCase):
    def test_lexical_sketch_does_not_cancel_colliding_terms(self) -> None:
        # These two terms land in the same signed-hash bucket with opposite
        # signs.  A legitimate query must not collapse to the zero vector.
        vector = capabilities._stable_sha256_embedding("바나나 우주망원경")
        self.assertTrue(any(vector))
        self.assertTrue(all(value >= 0.0 for value in vector))

    def test_pinned_relational_digest_and_all_digests_are_sha256(self) -> None:
        self.assertEqual(
            AKASICDB_AUDITED_DIGESTS["akasic/storage/relational_store.py"],
            "1a66bb519244cbfc759848fbcac7d4584dce60c746ae0971372f4929f832daf3",
        )
        for digest in AKASICDB_AUDITED_DIGESTS.values():
            self.assertRegex(digest, r"^[0-9a-f]{64}$")

    def test_model_metadata_rejects_non_sha256_digests(self) -> None:
        with self.assertRaisesRegex(ValueError, "manifest_sha256"):
            VerifiedModelMetadata(
                model_id="x",
                label="x",
                architecture="x",
                manifest_sha256="not-a-digest",
                config_sha256="b" * 64,
                checkpoint_modalities=("text",),
                runtime_input_modalities=("text",),
            )

    def test_attachment_admission_is_atomic_bounded_and_path_free(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            service = WorkspaceCapabilityService(temporary, _model())
            content = "검증 가능한 로컬 문서".encode()
            payload = service.add_attachment(
                name="evidence.md",
                media_type="text/markdown",
                content_base64=b64encode(content).decode("ascii"),
            )

            self.assertFalse(payload["duplicate"])
            self.assertNotIn("stored_path", payload)
            self.assertNotIn(str(Path(temporary)), json.dumps(payload))
            listed = service.list_attachments()
            self.assertEqual(listed["count"], 1)
            self.assertEqual(
                service.capability_payload()["attachments"]["catalog_lifetime"],
                "local_filesystem_persistent",
            )
            blobs = list(service.storage_root.iterdir())
            self.assertEqual(len(blobs), 1)
            self.assertEqual(blobs[0].read_bytes(), content)
            self.assertFalse(any(path.name.endswith(".tmp") for path in blobs))

    def test_bounded_text_and_image_previews_do_not_expose_host_paths(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            service = WorkspaceCapabilityService(temporary, _model())
            text = service.add_attachment(
                name="long.txt",
                media_type="text/plain",
                content_base64=b64encode(b"x" * 20_000).decode(),
            )
            preview = service.preview_attachment(text["attachment_id"])
            self.assertEqual(preview["kind"], "text")
            self.assertTrue(preview["truncated"])
            self.assertEqual(
                len(preview["text"]), capabilities.MAX_ATTACHMENT_PREVIEW_CHARS
            )
            self.assertNotIn(str(Path(temporary)), json.dumps(preview))

            image_bytes = b"\x89PNG\r\n\x1a\n" + b"bounded-image"
            image = service.add_attachment(
                name="local.png",
                media_type="image/png",
                content_base64=b64encode(image_bytes).decode(),
            )
            image_preview = service.preview_attachment(image["attachment_id"])
            self.assertEqual(image_preview["kind"], "image")
            self.assertEqual(
                image_preview["content_url"],
                "/api/workspace/attachments/content?attachment_id="
                + image["attachment_id"],
            )
            content, media_type = service.image_attachment_content(
                image["attachment_id"]
            )
            self.assertEqual(content, image_bytes)
            self.assertEqual(media_type, "image/png")
            self.assertNotIn(str(Path(temporary)), json.dumps(image_preview))

    def test_pdf_extraction_preview_index_and_explicit_reindex(self) -> None:
        if capabilities._PdfReader is None:
            self.skipTest("optional pypdf runtime is unavailable")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            project = root / "project"
            clone = root / "AkasicDB"
            project.mkdir()
            clone.mkdir()
            digests = _write_clone(clone)
            with patch.dict(AKASICDB_AUDITED_DIGESTS, digests, clear=True):
                service = WorkspaceCapabilityService(
                    project, _model(), akasicdb_path=clone
                )
                pdf_content = _pdf_bytes("equilibrium evidence")
                admitted = service.add_attachment(
                    name="paper.pdf",
                    media_type="application/pdf",
                    content_base64=b64encode(pdf_content).decode(),
                )
                self.assertTrue(admitted["text_indexable"])
                preview = service.preview_attachment(admitted["attachment_id"])
                self.assertEqual(preview["extraction"], "pypdf")
                self.assertIn("equilibrium evidence", preview["text"])
                indexed = service.index_attachments([admitted["attachment_id"]])
                self.assertEqual(indexed["documents"], 1)
                result = service.query_rag("equilibrium")
                self.assertEqual(result["count"], 1)
                self.assertEqual(
                    result["results"][0]["attachment_id"], admitted["attachment_id"]
                )
                source = result["results"][0]
                self.assertEqual(source["chunk_index"], 0)
                self.assertEqual(source["page_number"], 1)
                self.assertEqual(source["offset_basis"], "normalized_pdf_page_text_v1")
                extracted = capabilities._extract_pdf_document(pdf_content)
                normalized_page = capabilities._normalize_index_source(
                    extracted.pages[0].text
                )
                self.assertEqual(
                    normalized_page[source["char_start"] : source["char_end"]],
                    source["text"],
                )
                self.assertEqual(
                    source["excerpt_sha256"],
                    sha256(source["text"].encode()).hexdigest(),
                )
                self.assertGreater(source["score"], 0)
                exact = service.preview_rag_source(
                    admitted["attachment_id"], source["chunk_index"]
                )
                self.assertEqual(
                    set(exact),
                    {
                        "schema_version",
                        "attachment_id",
                        "chunk_index",
                        "name",
                        "media_type",
                        "text",
                        "page_number",
                        "char_start",
                        "char_end",
                        "offset_basis",
                        "excerpt_sha256",
                    },
                )
                for key in (
                    "attachment_id",
                    "chunk_index",
                    "name",
                    "media_type",
                    "text",
                    "page_number",
                    "char_start",
                    "char_end",
                    "offset_basis",
                    "excerpt_sha256",
                ):
                    self.assertEqual(exact[key], source[key])
                rebuilt = service.reindex_attachments([admitted["attachment_id"]])
                self.assertEqual(
                    rebuilt["reindexed_attachment_ids"], [admitted["attachment_id"]]
                )
                self.assertTrue(rebuilt["results"][0]["reindexed"])

    def test_pdf_blank_page_and_provenance_survive_restart_and_reindex(self) -> None:
        if capabilities._PdfReader is None:
            self.skipTest("optional pypdf runtime is unavailable")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            project = root / "project"
            clone = root / "AkasicDB"
            project.mkdir()
            clone.mkdir()
            digests = _write_clone(clone)
            pdf_content = _pdf_pages_bytes(
                ("alpha evidence", None, "omega evidence on physical page three")
            )
            extracted = capabilities._extract_pdf_document(pdf_content)
            self.assertEqual(extracted.page_count, 3)
            self.assertEqual(
                tuple(page.page_number for page in extracted.pages), (1, 2, 3)
            )
            self.assertEqual(extracted.pages[1].text, "")

            with patch.dict(AKASICDB_AUDITED_DIGESTS, digests, clear=True):
                service = WorkspaceCapabilityService(
                    project, _model(), akasicdb_path=clone
                )
                admitted = service.add_attachment(
                    name="three-pages.pdf",
                    media_type="application/pdf",
                    content_base64=b64encode(pdf_content).decode(),
                )
                preview = service.preview_attachment(admitted["attachment_id"])
                self.assertEqual(preview["page_count"], 3)
                self.assertEqual(
                    preview["pages"],
                    [
                        {
                            "page_number": 1,
                            "extracted_chars": len(extracted.pages[0].text),
                        },
                        {"page_number": 2, "extracted_chars": 0},
                        {
                            "page_number": 3,
                            "extracted_chars": len(extracted.pages[2].text),
                        },
                    ],
                )
                service.index_attachments([admitted["attachment_id"]])

                def assert_page_three_source(
                    selected: WorkspaceCapabilityService,
                ) -> None:
                    result = selected.query_rag("omega physical")
                    self.assertEqual(result["count"], 1)
                    source = result["results"][0]
                    self.assertEqual(source["page_number"], 3)
                    self.assertEqual(source["chunk_index"], 1)
                    normalized_page = capabilities._normalize_index_source(
                        extracted.pages[2].text
                    )
                    self.assertEqual(
                        normalized_page[source["char_start"] : source["char_end"]],
                        source["text"],
                    )
                    self.assertEqual(
                        source["excerpt_sha256"],
                        sha256(source["text"].encode()).hexdigest(),
                    )
                    exact = selected.preview_rag_source(
                        admitted["attachment_id"], source["chunk_index"]
                    )
                    for key in (
                        "attachment_id",
                        "chunk_index",
                        "name",
                        "media_type",
                        "text",
                        "page_number",
                        "char_start",
                        "char_end",
                        "offset_basis",
                        "excerpt_sha256",
                    ):
                        self.assertEqual(exact[key], source[key])

                assert_page_three_source(service)
                restarted = WorkspaceCapabilityService(
                    project, _model(), akasicdb_path=clone
                )
                assert_page_three_source(restarted)
                restarted.reindex_attachments([admitted["attachment_id"]])
                assert_page_three_source(restarted)

    def test_pdf_extraction_is_fail_closed_when_optional_backend_is_missing(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            with patch.object(capabilities, "_PdfReader", None):
                service = WorkspaceCapabilityService(temporary, _model())
                admitted = service.add_attachment(
                    name="paper.pdf",
                    media_type="application/pdf",
                    content_base64=b64encode(b"%PDF-1.4\nlocal-only").decode(),
                )
                self.assertFalse(admitted["text_indexable"])
                self.assertFalse(
                    service.capability_payload()["attachments"]["pdf_text_extraction"]
                )
                with self.assertRaisesRegex(
                    WorkspaceCapabilityError, "pypdf extractor is unavailable"
                ):
                    service.preview_attachment(admitted["attachment_id"])

    def test_pdf_extracted_text_limit_is_enforced_before_indexing(self) -> None:
        with patch.object(
            capabilities,
            "_run_pdf_extractor_subprocess",
            return_value="x" * (capabilities.MAX_PDF_EXTRACTED_CHARS + 1),
        ):
            with self.assertRaisesRegex(WorkspaceCapabilityError, "character limit"):
                capabilities._extract_pdf_text(b"%PDF-1.4\nlocal")

    def test_pdf_subprocess_timeout_is_fail_closed_and_kills_worker(self) -> None:
        class _TimedOutProcess:
            def __init__(self) -> None:
                self.calls = 0
                self.killed = False

            def communicate(self, input=None, timeout=None):
                self.calls += 1
                if self.calls == 1:
                    raise capabilities.subprocess.TimeoutExpired("pdf", timeout)
                return b"", b""

            def kill(self) -> None:
                self.killed = True

        process = _TimedOutProcess()
        with (
            patch.object(capabilities.subprocess, "Popen", return_value=process),
            patch.object(capabilities, "_assign_windows_pdf_job", return_value=0),
        ):
            with self.assertRaisesRegex(
                WorkspaceCapabilityError, "time limit"
            ) as raised:
                capabilities._run_pdf_extractor_subprocess(b"%PDF-1.4\nlocal")
        self.assertEqual(raised.exception.code, "PDF_TEXT_EXTRACTION_TIMEOUT")
        self.assertTrue(process.killed)

    def test_pdf_parent_rejects_nonsequential_worker_pages(self) -> None:
        class _InvalidSequenceProcess:
            returncode = 0

            def communicate(self, input=None, timeout=None):
                del input, timeout
                payload = {
                    "ok": True,
                    "pages": [
                        {"page_number": 1, "text": "first"},
                        {"page_number": 3, "text": "third"},
                    ],
                }
                return json.dumps(payload).encode(), b""

        with (
            patch.object(
                capabilities.subprocess,
                "Popen",
                return_value=_InvalidSequenceProcess(),
            ),
            patch.object(capabilities, "_assign_windows_pdf_job", return_value=0),
        ):
            with self.assertRaisesRegex(
                WorkspaceCapabilityError, "page sequence"
            ) as raised:
                capabilities._run_pdf_extractor_document(b"%PDF-1.4\nlocal")
        self.assertEqual(raised.exception.code, "PDF_TEXT_EXTRACTION_FAILED")

    def test_encoded_length_is_rejected_before_base64_decode(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            service = WorkspaceCapabilityService(temporary, _model())
            with patch.object(capabilities, "b64decode") as decode:
                with self.assertRaisesRegex(
                    WorkspaceCapabilityError, "encoded character limit"
                ):
                    service.add_attachment(
                        name="large.txt",
                        media_type="text/plain",
                        content_base64="A" * (MAX_ATTACHMENT_BASE64_CHARS + 1),
                    )
            decode.assert_not_called()

    def test_storage_parent_symlink_is_fail_closed(self) -> None:
        with (
            tempfile.TemporaryDirectory() as temporary,
            tempfile.TemporaryDirectory() as outside,
        ):
            root = Path(temporary)
            try:
                (root / "outputs").symlink_to(Path(outside), target_is_directory=True)
            except OSError:
                self.skipTest("directory symlinks are unavailable")
            with self.assertRaisesRegex(
                WorkspaceCapabilityError, "unsafe path component"
            ):
                WorkspaceCapabilityService(root, _model())

    def test_process_restart_restores_the_integrity_checked_catalog(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            first = WorkspaceCapabilityService(temporary, _model())
            encoded = b64encode(b"local evidence").decode("ascii")
            first.add_attachment(
                name="one.txt", media_type="text/plain", content_base64=encoded
            )
            second = WorkspaceCapabilityService(temporary, _model())
            self.assertEqual(second.list_attachments()["count"], 1)
            admitted = second.add_attachment(
                name="one.txt", media_type="text/plain", content_base64=encoded
            )
            self.assertTrue(admitted["duplicate"])
            self.assertEqual(
                second.capability_payload()["attachments"]["persisted_blob_count"],
                1,
            )

    def test_add_rolls_back_blob_and_memory_when_catalog_write_fails(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            service = WorkspaceCapabilityService(temporary, _model())
            with patch.object(
                capabilities,
                "_atomic_json_write",
                side_effect=[OSError("catalog full"), None],
            ):
                with self.assertRaisesRegex(
                    WorkspaceCapabilityError, "catalog could not be committed"
                ):
                    service.add_attachment(
                        name="rollback.txt",
                        media_type="text/plain",
                        content_base64=b64encode(b"transactional evidence").decode(),
                    )
            self.assertEqual(service.list_attachments()["count"], 0)
            self.assertEqual(list(service.storage_root.iterdir()), [])
            self.assertEqual(
                service.capability_payload()["attachments"]["persisted_blob_count"],
                0,
            )

    def test_index_batch_rolls_back_catalog_and_rag_on_commit_failure(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            project = root / "project"
            clone = root / "AkasicDB"
            project.mkdir()
            clone.mkdir()
            digests = _write_clone(clone)
            with patch.dict(AKASICDB_AUDITED_DIGESTS, digests, clear=True):
                service = WorkspaceCapabilityService(
                    project, _model(), akasicdb_path=clone
                )
                admitted = service.add_attachment(
                    name="rollback.md",
                    media_type="text/markdown",
                    content_base64=b64encode("원자적 인덱스".encode()).decode(),
                )
                with patch.object(
                    capabilities,
                    "_atomic_json_write",
                    side_effect=[OSError("catalog full"), None],
                ):
                    with self.assertRaisesRegex(
                        WorkspaceCapabilityError, "catalog could not be committed"
                    ):
                        service.index_attachments([admitted["attachment_id"]])
                self.assertFalse(service.list_attachments()["items"][0]["indexed"])
                self.assertEqual(service.query_rag("원자적 인덱스")["count"], 0)

    def test_delete_unlink_failure_preserves_catalog_blob_and_index(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            project = root / "project"
            clone = root / "AkasicDB"
            project.mkdir()
            clone.mkdir()
            digests = _write_clone(clone)
            with patch.dict(AKASICDB_AUDITED_DIGESTS, digests, clear=True):
                service = WorkspaceCapabilityService(
                    project, _model(), akasicdb_path=clone
                )
                admitted = service.add_attachment(
                    name="retained.md",
                    media_type="text/markdown",
                    content_base64=b64encode("삭제 실패 보존".encode()).decode(),
                )
                attachment_id = admitted["attachment_id"]
                service.index_attachments([attachment_id])
                record = service._attachments[attachment_id]
                with patch.object(
                    Path, "unlink", side_effect=PermissionError("locked")
                ):
                    with self.assertRaisesRegex(
                        WorkspaceCapabilityError, "blob could not be deleted"
                    ):
                        service.delete_attachment(attachment_id)
                self.assertTrue(record.stored_path.is_file())
                self.assertTrue(service.list_attachments()["items"][0]["indexed"])
                self.assertEqual(service.query_rag("삭제 실패 보존")["count"], 1)

    def test_delete_catalog_failure_restores_the_unlinked_blob(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            service = WorkspaceCapabilityService(temporary, _model())
            admitted = service.add_attachment(
                name="restore.txt",
                media_type="text/plain",
                content_base64=b64encode(b"restore after catalog failure").decode(),
            )
            attachment_id = admitted["attachment_id"]
            stored_path = service._attachments[attachment_id].stored_path
            with patch.object(
                capabilities,
                "_atomic_json_write",
                side_effect=[OSError("catalog full"), None],
            ):
                with self.assertRaisesRegex(
                    WorkspaceCapabilityError, "deletion could not be committed"
                ):
                    service.delete_attachment(attachment_id)
            self.assertTrue(stored_path.is_file())
            self.assertEqual(service.list_attachments()["count"], 1)

    def test_restart_inventory_blocks_cumulative_blob_count_bypass(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            first = WorkspaceCapabilityService(temporary, _model())
            for index in range(MAX_ATTACHMENT_COUNT):
                first.add_attachment(
                    name=f"evidence-{index}.txt",
                    media_type="text/plain",
                    content_base64=b64encode(f"evidence {index}".encode()).decode(),
                )

            restarted = WorkspaceCapabilityService(temporary, _model())
            attachments = restarted.capability_payload()["attachments"]
            self.assertEqual(
                attachments["catalog_lifetime"], "local_filesystem_persistent"
            )
            self.assertEqual(attachments["persisted_blob_count"], MAX_ATTACHMENT_COUNT)
            self.assertLess(
                attachments["persisted_blob_bytes"], MAX_ATTACHMENT_TOTAL_BYTES
            )
            with self.assertRaisesRegex(
                WorkspaceCapabilityError, "count limit reached"
            ):
                restarted.add_attachment(
                    name="overflow.txt",
                    media_type="text/plain",
                    content_base64=b64encode(b"new persisted blob").decode(),
                )

    def test_restart_inventory_blocks_cumulative_blob_byte_bypass(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            with patch.object(capabilities, "MAX_ATTACHMENT_TOTAL_BYTES", 12):
                first = WorkspaceCapabilityService(temporary, _model())
                first.add_attachment(
                    name="first.txt",
                    media_type="text/plain",
                    content_base64=b64encode(b"12345678").decode(),
                )
                restarted = WorkspaceCapabilityService(temporary, _model())
                self.assertEqual(
                    restarted.capability_payload()["attachments"][
                        "persisted_blob_bytes"
                    ],
                    8,
                )
                with self.assertRaisesRegex(
                    WorkspaceCapabilityError, "total byte limit reached"
                ):
                    restarted.add_attachment(
                        name="second.txt",
                        media_type="text/plain",
                        content_base64=b64encode(b"abcdefgh").decode(),
                    )

    def test_restart_can_reuse_a_verified_content_addressed_blob(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            first = WorkspaceCapabilityService(temporary, _model())
            encoded = b64encode(b"same local content").decode()
            first.add_attachment(
                name="original.txt", media_type="text/plain", content_base64=encoded
            )
            restarted = WorkspaceCapabilityService(temporary, _model())
            admitted = restarted.add_attachment(
                name="renamed.txt", media_type="text/plain", content_base64=encoded
            )
            self.assertTrue(admitted["duplicate"])
            self.assertEqual(len(list(restarted.storage_root.iterdir())), 1)
            self.assertEqual(
                restarted.capability_payload()["attachments"]["persisted_blob_count"],
                1,
            )

    def test_index_state_is_rebuilt_after_restart_and_delete_removes_it(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            project = root / "project"
            clone = root / "AkasicDB"
            project.mkdir()
            clone.mkdir()
            digests = _write_clone(clone)
            with patch.dict(AKASICDB_AUDITED_DIGESTS, digests, clear=True):
                first = WorkspaceCapabilityService(
                    project, _model(), akasicdb_path=clone
                )
                admitted = first.add_attachment(
                    name="persistent.md",
                    media_type="text/markdown",
                    content_base64=b64encode("평형 탐색 영구 근거".encode()).decode(),
                )
                attachment_id = admitted["attachment_id"]
                first.index_attachments([attachment_id])

                restarted = WorkspaceCapabilityService(
                    project, _model(), akasicdb_path=clone
                )
                listed = restarted.list_attachments()
                self.assertEqual(listed["count"], 1)
                self.assertTrue(listed["items"][0]["indexed"])
                self.assertEqual(restarted.query_rag("평형 탐색")["count"], 1)

                deleted = restarted.delete_attachment(attachment_id)
                self.assertTrue(deleted["deleted"])
                self.assertTrue(deleted["index_removed"])
                self.assertTrue(deleted["blob_deleted"])
                self.assertEqual(restarted.query_rag("평형 탐색")["count"], 0)

                final = WorkspaceCapabilityService(
                    project, _model(), akasicdb_path=clone
                )
                self.assertEqual(final.list_attachments()["count"], 0)
                self.assertEqual(final.query_rag("평형 탐색")["count"], 0)

    def test_catalog_symlink_and_tampered_blob_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            service = WorkspaceCapabilityService(root, _model())
            admitted = service.add_attachment(
                name="evidence.txt",
                media_type="text/plain",
                content_base64=b64encode(b"verified evidence").decode(),
            )
            record = service._attachments[admitted["attachment_id"]]
            record.stored_path.write_bytes(b"tampered")
            with self.assertRaisesRegex(WorkspaceCapabilityError, "integrity check"):
                WorkspaceCapabilityService(root, _model())

    def test_attachment_name_rejects_c0_and_c1_control_characters(self) -> None:
        encoded = b64encode(b"safe content").decode()
        with tempfile.TemporaryDirectory() as temporary:
            service = WorkspaceCapabilityService(temporary, _model())
            for name in ("bad\x1f.txt", "bad\x7f.txt", "bad\x85.txt", "bad\x9f.txt"):
                with (
                    self.subTest(name=repr(name)),
                    self.assertRaisesRegex(WorkspaceCapabilityError, "name is unsafe"),
                ):
                    service.add_attachment(
                        name=name,
                        media_type="text/plain",
                        content_base64=encoded,
                    )

    def test_json_size_and_nesting_fail_with_bounded_errors(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            service = WorkspaceCapabilityService(temporary, _model())
            oversized = b'{"value":"' + b"x" * MAX_JSON_ATTACHMENT_BYTES + b'"}'
            with self.assertRaisesRegex(
                WorkspaceCapabilityError, "one-megabyte parser limit"
            ):
                service.add_attachment(
                    name="large.json",
                    media_type="application/json",
                    content_base64=b64encode(oversized).decode(),
                )

            deeply_nested = (
                "[" * (MAX_JSON_NESTING + 1) + "0" + "]" * (MAX_JSON_NESTING + 1)
            ).encode()
            with self.assertRaisesRegex(WorkspaceCapabilityError, "nesting limit"):
                service.add_attachment(
                    name="nested.json",
                    media_type="application/json",
                    content_base64=b64encode(deeply_nested).decode(),
                )

    def test_json_recursion_error_is_converted_to_workspace_error(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            service = WorkspaceCapabilityService(temporary, _model())
            with patch.object(capabilities.json, "loads", side_effect=RecursionError):
                with self.assertRaisesRegex(
                    WorkspaceCapabilityError, "JSON attachment is invalid"
                ):
                    service.add_attachment(
                        name="valid-shape.json",
                        media_type="application/json",
                        content_base64=b64encode(b'{"key": 1}').decode(),
                    )

    def test_akasic_adapter_indexes_real_interfaces_and_filters_nonpositive(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            clone = Path(temporary) / "AkasicDB"
            clone.mkdir()
            digests = _write_clone(clone)
            with patch.dict(AKASICDB_AUDITED_DIGESTS, digests, clear=True):
                adapter = AkasicDBAdapter(clone)
                indexed = adapter.index_document(
                    attachment_id="abc",
                    name="paper.md",
                    media_type="text/markdown",
                    text="평형 탐색과 텐서 검색을 결합합니다.",
                )
                self.assertEqual(indexed["chunks"], 1)
                result = adapter.query("평형 탐색", limit=5)
                self.assertEqual(result["count"], 1)
                source = result["results"][0]
                self.assertGreater(source["score"], 0)
                self.assertIsNone(source["page_number"])
                self.assertEqual(source["offset_basis"], "normalized_document_text_v1")
                normalized = capabilities._normalize_index_source(
                    "평형 탐색과 텐서 검색을 결합합니다."
                )
                self.assertEqual(
                    normalized[source["char_start"] : source["char_end"]],
                    source["text"],
                )
                entity = next(iter(adapter._chunk_ids))
                adapter.vector_store.similarity_search = lambda *_args, **_kwargs: [
                    (entity, 0.9)
                ]
                self.assertEqual(adapter.query("양자 회로", limit=5)["results"], [])
                adapter.vector_store.similarity_search = lambda *_args, **_kwargs: [
                    (entity, 0.0),
                    (entity, -0.25),
                ]
                self.assertEqual(adapter.query("평형", limit=5)["results"], [])

    def test_exact_source_fails_closed_when_indexed_record_is_mutated(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            project = root / "project"
            clone = root / "AkasicDB"
            project.mkdir()
            clone.mkdir()
            digests = _write_clone(clone)
            with patch.dict(AKASICDB_AUDITED_DIGESTS, digests, clear=True):
                service = WorkspaceCapabilityService(
                    project, _model(), akasicdb_path=clone
                )
                admitted = service.add_attachment(
                    name="evidence.md",
                    media_type="text/markdown",
                    content_base64=b64encode("평형 근거 문서".encode()).decode(),
                )
                service.index_attachments([admitted["attachment_id"]])
                assert service.akasicdb is not None
                entity = f"chunk:{admitted['attachment_id']}:0"
                service.akasicdb.relational_store.records[entity]["excerpt_sha256"] = (
                    "0" * 64
                )
                with self.assertRaises(WorkspaceCapabilityError) as captured:
                    service.preview_rag_source(admitted["attachment_id"], 0)
                self.assertEqual(captured.exception.code, "RAG_SOURCE_INTEGRITY_FAILED")

    def test_invalid_url_port_is_a_bounded_policy_error(self) -> None:
        policy = WebAccessPolicy(online_mode=True, allowlist=("api.lens.org",))
        with self.assertRaisesRegex(WorkspaceCapabilityError, "port is invalid"):
            policy.authorize_url("https://api.lens.org:not-a-port/patent/search")

    def test_web_policy_publishes_fail_closed_executor_flag(self) -> None:
        payload = WebAccessPolicy(
            online_mode=True,
            allowlist=("api.lens.org",),
            lens_api_token_configured=True,
        ).as_payload()
        self.assertFalse(payload["executor_implemented"])
        self.assertEqual(payload["execution"], "authorization_only")

    def test_answer_integration_state_is_explicit_and_consistent(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            project = root / "project"
            clone = root / "AkasicDB"
            project.mkdir()
            clone.mkdir()
            digests = _write_clone(clone)
            with patch.dict(AKASICDB_AUDITED_DIGESTS, digests, clear=True):
                service = WorkspaceCapabilityService(
                    project,
                    _model(),
                    akasicdb_path=clone,
                    answer_integration_enabled=True,
                )
                admitted = service.add_attachment(
                    name="paper.md",
                    media_type="text/markdown",
                    content_base64=b64encode("평형 탐색 근거".encode()).decode(),
                )
                indexed = service.index_attachments([admitted["attachment_id"]])
                queried = service.query_rag("평형 탐색")
            self.assertTrue(service.capability_payload()["rag"]["answer_integration"])
            self.assertTrue(indexed["answer_integration"])
            self.assertTrue(queried["answer_integration"])
            self.assertEqual(queried["embedding"], "stable_sha256_lexical_sketch_v1")


if __name__ == "__main__":
    unittest.main()
