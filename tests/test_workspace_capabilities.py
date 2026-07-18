from __future__ import annotations

from base64 import b64encode
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
import tempfile
from threading import Event, Thread
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


def _write_model_registry_candidate(
    registry: Path, name: str = "trusted-e4b"
) -> tuple[Path, Path]:
    model = registry / name
    model.mkdir()
    config = {
        "architectures": ["Gemma4ForConditionalGeneration"],
        "vision_config": {},
        "image_token_id": 7,
        "video_token_id": 8,
        "audio_config": {},
        "audio_token_id": 9,
    }
    (model / "config.json").write_text(
        json.dumps(config, sort_keys=True), encoding="utf-8"
    )
    (model / "model.safetensors").write_bytes(b"bounded-test-weights")
    digests = {
        relative: sha256((model / relative).read_bytes()).hexdigest()
        for relative in ("config.json", "model.safetensors")
    }
    manifest = registry / f"{name}.manifest.toml"
    manifest.write_text(
        "\n".join(
            (
                "[model]",
                'family = "gemma4"',
                'variant = "e4b"',
                'role = "instruction_tuned"',
                'source = "google/gemma-4-E4B-it"',
                'revision = "a4c2d58be94dda072b918d9db64ee85c8ed34e3f"',
                "",
                "[files]",
                f'"config.json" = "{digests["config.json"]}"',
                (f'"model.safetensors" = "{digests["model.safetensors"]}"'),
                "",
            )
        ),
        encoding="utf-8",
    )
    return model, manifest


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
    def _assert_rag_quarantined(
        self,
        service: WorkspaceCapabilityService,
        *,
        attachment_id: str,
        query: str,
    ) -> None:
        self.assertIsNone(service.akasicdb)
        self.assertEqual(
            service.akasicdb_error,
            (
                "RAG_OPERATOR_REVIEW_REQUIRED",
                "local RAG state is quarantined pending operator review",
            ),
        )
        capability = service.capability_payload()["rag"]
        self.assertEqual(capability["state"], "unavailable")
        self.assertFalse(capability["answer_integration"])
        self.assertEqual(capability["error"]["code"], "RAG_OPERATOR_REVIEW_REQUIRED")

        actions = (
            ("query", lambda: service.query_rag(query)),
            ("source", lambda: service.preview_rag_source(attachment_id, 0)),
            ("index", lambda: service.index_attachments([attachment_id])),
            ("reindex", lambda: service.reindex_attachments([attachment_id])),
            ("delete", lambda: service.delete_attachment(attachment_id)),
            (
                "add",
                lambda: service.add_attachment(
                    name="blocked-after-quarantine.md",
                    media_type="text/markdown",
                    content_base64=b64encode(b"blocked after quarantine").decode(),
                ),
            ),
        )
        for label, action in actions:
            with self.subTest(quarantined_action=label):
                with self.assertRaises(WorkspaceCapabilityError) as captured:
                    action()
                self.assertEqual(
                    captured.exception.code, "RAG_OPERATOR_REVIEW_REQUIRED"
                )

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

    def test_local_model_discovery_is_bounded_verified_and_non_selectable(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            project = root / "project"
            registry = root / "registry"
            project.mkdir()
            registry.mkdir()
            _write_model_registry_candidate(registry)

            with patch(
                "cogni_agent.model_service._verify_instruction_tuned_e4b_snapshot"
            ) as trusted_fingerprint:
                service = WorkspaceCapabilityService(
                    project,
                    _model(),
                    model_registry_root=registry,
                )

            trusted_fingerprint.assert_called_once()
            payload = service.capability_payload()["models"]
            self.assertEqual(payload["state"], "verified_local_registry")
            self.assertEqual(payload["switching"], "idempotent_current_model_only")
            self.assertEqual(payload["discovery"]["state"], "verified")
            self.assertEqual(payload["discovery"]["verified_count"], 1)
            self.assertEqual(payload["discovery"]["rejected_count"], 0)
            self.assertEqual(len(payload["items"]), 2)
            selected, discovered = payload["items"]
            self.assertTrue(selected["selected"])
            self.assertTrue(selected["selectable"])
            self.assertFalse(discovered["selected"])
            self.assertFalse(discovered["selectable"])
            self.assertEqual(
                discovered["verification"],
                "closed_world_manifest_and_trusted_fingerprint",
            )
            self.assertEqual(
                discovered["runtime_input_modalities"],
                ["text"],
            )
            self.assertEqual(
                discovered["checkpoint_modalities"],
                ["text", "image", "video", "audio"],
            )
            self.assertNotIn(str(registry), json.dumps(payload))
            with self.assertRaises(WorkspaceCapabilityError) as captured:
                service.select_model(discovered["model_id"])
            self.assertEqual(captured.exception.code, "MODEL_SWITCH_UNAVAILABLE")

    def test_local_model_discovery_rejects_tamper_without_loading(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            project = root / "project"
            registry = root / "registry"
            project.mkdir()
            registry.mkdir()
            model, _manifest = _write_model_registry_candidate(registry)
            (model / "config.json").write_text("{}", encoding="utf-8")

            service = WorkspaceCapabilityService(
                project,
                _model(),
                model_registry_root=registry,
            )

            payload = service.capability_payload()["models"]
            self.assertEqual(payload["discovery"]["state"], "verified_with_rejections")
            self.assertEqual(payload["discovery"]["verified_count"], 0)
            self.assertEqual(payload["discovery"]["rejected_count"], 1)
            self.assertEqual(
                payload["discovery"]["rejections"][0]["code"],
                "MODEL_VERIFICATION_FAILED",
            )
            self.assertEqual(len(payload["items"]), 1)

    def test_local_model_discovery_requires_pinned_trusted_fingerprint(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            project = root / "project"
            registry = root / "registry"
            project.mkdir()
            registry.mkdir()
            _write_model_registry_candidate(registry)

            service = WorkspaceCapabilityService(
                project,
                _model(),
                model_registry_root=registry,
            )

            discovery = service.capability_payload()["models"]["discovery"]
            self.assertEqual(discovery["verified_count"], 0)
            self.assertEqual(discovery["rejected_count"], 1)
            self.assertEqual(
                discovery["rejections"][0]["code"], "MODEL_VERIFICATION_FAILED"
            )

    def test_local_model_discovery_rejects_unsafe_and_oversized_registries(
        self,
    ) -> None:
        self.assertFalse(capabilities._safe_local_model_name("google/gemma-4-E4B-it"))
        self.assertFalse(capabilities._safe_local_model_name("https://example.test"))
        self.assertFalse(capabilities._safe_local_model_name("../model"))
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            project = root / "project"
            registry = root / "registry"
            project.mkdir()
            registry.mkdir()
            for index in range(3):
                (registry / f"entry-{index}.txt").write_text("x", encoding="ascii")

            with patch.object(capabilities, "MAX_LOCAL_MODEL_REGISTRY_ENTRIES", 2):
                service = WorkspaceCapabilityService(
                    project,
                    _model(),
                    model_registry_root=registry,
                )
            discovery = service.capability_payload()["models"]["discovery"]
            self.assertEqual(discovery["state"], "rejected")
            self.assertEqual(
                discovery["rejections"][0]["code"], "MODEL_REGISTRY_TOO_LARGE"
            )

            relative = capabilities.discover_verified_local_models(
                "relative-registry", selected_model=_model()
            )
            self.assertEqual(relative.state, "rejected")
            self.assertEqual(relative.rejections[0].code, "MODEL_REGISTRY_PATH_UNSAFE")

    def test_local_model_discovery_rejects_link_or_reparse_candidate(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            project = root / "project"
            registry = root / "registry"
            project.mkdir()
            registry.mkdir()
            candidate, _manifest = _write_model_registry_candidate(registry)
            original = capabilities._path_is_link_or_reparse

            def synthetic_reparse(path: Path) -> bool:
                if path == candidate:
                    return True
                return original(path)

            with patch.object(
                capabilities,
                "_path_is_link_or_reparse",
                side_effect=synthetic_reparse,
            ):
                service = WorkspaceCapabilityService(
                    project,
                    _model(),
                    model_registry_root=registry,
                )
            rejection = service.capability_payload()["models"]["discovery"][
                "rejections"
            ][0]
            self.assertEqual(rejection["candidate"], "trusted-e4b")
            self.assertEqual(rejection["code"], "MODEL_CANDIDATE_UNSAFE")

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
        self.assertEqual(process.calls, 2)

    def test_pdf_subprocess_timeout_closes_job_and_reaps_worker(self) -> None:
        class _TimedOutJobProcess:
            def __init__(self) -> None:
                self.calls = 0

            def communicate(self, input=None, timeout=None):
                self.calls += 1
                if self.calls == 1:
                    raise capabilities.subprocess.TimeoutExpired("pdf", timeout)
                self.reap_input = input
                self.reap_timeout = timeout
                return b"", b""

        process = _TimedOutJobProcess()
        with (
            patch.object(capabilities.subprocess, "Popen", return_value=process),
            patch.object(capabilities, "_assign_windows_pdf_job", return_value=73),
            patch.object(capabilities, "_close_windows_handle") as close_handle,
        ):
            with self.assertRaises(WorkspaceCapabilityError) as raised:
                capabilities._run_pdf_extractor_subprocess(b"%PDF-1.4\nlocal")
        self.assertEqual(raised.exception.code, "PDF_TEXT_EXTRACTION_TIMEOUT")
        self.assertEqual(process.calls, 2)
        self.assertIsNone(process.reap_input)
        self.assertIsNone(process.reap_timeout)
        close_handle.assert_called_once_with(73)

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

    def test_pdf_parent_rejects_non_integer_worker_page_number(self) -> None:
        class _InvalidPageTypeProcess:
            returncode = 0

            def communicate(self, input=None, timeout=None):
                del input, timeout
                payload = {
                    "ok": True,
                    "pages": [{"page_number": 1.0, "text": "first"}],
                }
                return json.dumps(payload).encode(), b""

        with (
            patch.object(
                capabilities.subprocess,
                "Popen",
                return_value=_InvalidPageTypeProcess(),
            ),
            patch.object(capabilities, "_assign_windows_pdf_job", return_value=0),
        ):
            with self.assertRaisesRegex(
                WorkspaceCapabilityError, "page sequence"
            ) as raised:
                capabilities._run_pdf_extractor_document(b"%PDF-1.4\nlocal")
        self.assertEqual(raised.exception.code, "PDF_TEXT_EXTRACTION_FAILED")

    def test_nonpaginated_length_limit_applies_after_normalization(self) -> None:
        document = capabilities.ExtractedDocument(
            text="alpha   ",
            page_count=0,
            pages=(),
        )
        with patch.object(capabilities, "MAX_INDEXED_TEXT_CHARS", 5):
            chunks = capabilities._chunk_document(document)
        self.assertEqual(tuple(chunk.text for chunk in chunks), ("alpha",))
        chunk = chunks[0]
        self.assertIsNone(chunk.page_number)
        self.assertEqual(chunk.char_start, 0)
        self.assertEqual(chunk.char_end, 5)
        self.assertEqual(chunk.offset_basis, "normalized_document_text_v1")
        self.assertEqual(
            chunk.excerpt_sha256,
            sha256(b"alpha").hexdigest(),
        )

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
                original_unlink = Path.unlink

                def fail_only_attachment_unlink(
                    selected: Path, *args: object, **kwargs: object
                ) -> None:
                    if selected == record.stored_path:
                        raise PermissionError("locked")
                    original_unlink(selected, *args, **kwargs)

                with patch.object(
                    Path,
                    "unlink",
                    autospec=True,
                    side_effect=fail_only_attachment_unlink,
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

    def test_attachment_reader_hashes_the_exact_returned_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            service = WorkspaceCapabilityService(temporary, _model())
            admitted = service.add_attachment(
                name="same-size.txt",
                media_type="text/plain",
                content_base64=b64encode(b"trusted-bytes").decode(),
            )
            record = service._attachments[admitted["attachment_id"]]
            forged = b"x" * record.size_bytes
            self.assertNotEqual(sha256(forged).hexdigest(), record.sha256)
            with patch.object(Path, "read_bytes", return_value=forged):
                with self.assertRaises(WorkspaceCapabilityError) as captured:
                    service._verified_attachment_bytes(record)
            self.assertEqual(captured.exception.code, "ATTACHMENT_INTEGRITY_FAILED")

    def test_pdf_source_snapshot_is_o1_for_repeated_concurrent_clicks(self) -> None:
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
                admitted = service.add_attachment(
                    name="snapshot.pdf",
                    media_type="application/pdf",
                    content_base64=b64encode(_pdf_bytes("immutable evidence")).decode(),
                )
                attachment_id = admitted["attachment_id"]
                service.index_attachments([attachment_id])
                expected = service.preview_rag_source(attachment_id, 0)
                results: list[dict[str, object] | None] = [None] * 8
                errors: list[BaseException] = []

                def preview(index: int) -> None:
                    try:
                        results[index] = service.preview_rag_source(attachment_id, 0)
                    except BaseException as exc:  # pragma: no cover - surfaced below
                        errors.append(exc)

                with patch.object(
                    capabilities,
                    "_extract_pdf_document",
                    side_effect=AssertionError("source GET must not re-extract PDF"),
                ):
                    threads = [
                        Thread(target=preview, args=(index,)) for index in range(8)
                    ]
                    for thread in threads:
                        thread.start()
                    for thread in threads:
                        thread.join(timeout=2)
                    self.assertTrue(all(not thread.is_alive() for thread in threads))
                self.assertEqual(errors, [])
                self.assertEqual(results, [expected] * 8)

    def test_source_get_serializes_with_reindex_and_delete_lifecycle(self) -> None:
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
                    name="lifecycle.md",
                    media_type="text/markdown",
                    content_base64=b64encode(b"lifecycle evidence").decode(),
                )
                attachment_id = admitted["attachment_id"]
                service.index_attachments([attachment_id])
                initial_generation = service._source_generation

                source_results: list[dict[str, object]] = []
                mutation_results: list[dict[str, object]] = []
                errors: list[BaseException] = []
                entered = Event()
                release = Event()
                assert service.akasicdb is not None
                original_preview = service.akasicdb.source_preview

                def blocked_preview(**kwargs):
                    entered.set()
                    if not release.wait(timeout=2):
                        raise TimeoutError("test source release timed out")
                    return original_preview(**kwargs)

                def read_source() -> None:
                    try:
                        source_results.append(
                            service.preview_rag_source(attachment_id, 0)
                        )
                    except BaseException as exc:  # pragma: no cover - surfaced below
                        errors.append(exc)

                def reindex() -> None:
                    try:
                        mutation_results.append(
                            service.reindex_attachments([attachment_id])
                        )
                    except BaseException as exc:  # pragma: no cover - surfaced below
                        errors.append(exc)

                with patch.object(
                    service.akasicdb, "source_preview", side_effect=blocked_preview
                ):
                    reader = Thread(target=read_source)
                    mutation = Thread(target=reindex)
                    reader.start()
                    self.assertTrue(entered.wait(timeout=2))
                    mutation.start()
                    self.assertTrue(mutation.is_alive())
                    release.set()
                    reader.join(timeout=2)
                    mutation.join(timeout=2)
                self.assertEqual(errors, [])
                self.assertEqual(len(source_results), 1)
                self.assertEqual(len(mutation_results), 1)
                self.assertGreater(service._source_generation, initial_generation)
                self.assertEqual(
                    service.preview_rag_source(attachment_id, 0), source_results[0]
                )

                deleted = service.delete_attachment(attachment_id)
                self.assertTrue(deleted["deleted"])
                with self.assertRaises(WorkspaceCapabilityError) as captured:
                    service.preview_rag_source(attachment_id, 0)
                self.assertEqual(captured.exception.code, "RAG_SOURCE_NOT_FOUND")

    def test_query_rag_returns_only_exact_service_owned_snapshots(self) -> None:
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
                    name="authority.md",
                    media_type="text/markdown",
                    content_base64=b64encode(b"exact authority evidence").decode(),
                )
                attachment_id = admitted["attachment_id"]
                service.index_attachments([attachment_id])

                normal = service.query_rag("authority evidence")
                self.assertEqual(normal["count"], 1)
                exact = service.preview_rag_source(attachment_id, 0)
                for key, value in exact.items():
                    if key != "schema_version":
                        self.assertEqual(normal["results"][0][key], value)

                mismatched = dict(normal["results"][0])
                mismatched["text"] = "tampered authority evidence"
                transient_lens = dict(normal["results"][0])
                transient_lens["attachment_id"] = "lens-temporary-result"
                with patch.object(
                    service.akasicdb,
                    "query",
                    return_value={
                        "query": "authority evidence",
                        "results": [mismatched],
                        "count": 1,
                    },
                ):
                    with self.assertRaises(WorkspaceCapabilityError) as captured:
                        service.query_rag("authority evidence")
                self.assertEqual(captured.exception.code, "RAG_QUERY_INTEGRITY_FAILED")

                with patch.object(
                    service.akasicdb,
                    "query",
                    return_value={
                        "query": "authority evidence",
                        "results": [transient_lens],
                        "count": 1,
                    },
                ):
                    filtered = service.query_rag("authority evidence")
                self.assertEqual(filtered["results"], [])
                self.assertEqual(filtered["count"], 0)

    def test_query_rag_serializes_with_reindex_transaction(self) -> None:
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
                    name="serialized.md",
                    media_type="text/markdown",
                    content_base64=b64encode(b"serialized committed evidence").decode(),
                )
                attachment_id = admitted["attachment_id"]
                service.index_attachments([attachment_id])
                assert service.akasicdb is not None
                original_index = service.akasicdb.index_document
                entered = Event()
                release = Event()
                query_started = Event()
                query_finished = Event()
                blocked_once = False
                errors: list[BaseException] = []
                queried: list[dict[str, object]] = []

                def blocking_index(**kwargs):
                    nonlocal blocked_once
                    if not blocked_once:
                        blocked_once = True
                        entered.set()
                        if not release.wait(timeout=2):
                            raise TimeoutError("test reindex release timed out")
                    return original_index(**kwargs)

                def reindex() -> None:
                    try:
                        service.reindex_attachments([attachment_id])
                    except BaseException as exc:  # pragma: no cover - surfaced below
                        errors.append(exc)

                def query() -> None:
                    query_started.set()
                    try:
                        queried.append(service.query_rag("committed evidence"))
                    except BaseException as exc:  # pragma: no cover - surfaced below
                        errors.append(exc)
                    finally:
                        query_finished.set()

                with patch.object(
                    service.akasicdb, "index_document", side_effect=blocking_index
                ):
                    mutation = Thread(target=reindex)
                    reader = Thread(target=query)
                    mutation.start()
                    self.assertTrue(entered.wait(timeout=2))
                    reader.start()
                    self.assertTrue(query_started.wait(timeout=2))
                    self.assertFalse(query_finished.wait(timeout=0.1))
                    release.set()
                    mutation.join(timeout=2)
                    reader.join(timeout=2)
                self.assertFalse(mutation.is_alive())
                self.assertFalse(reader.is_alive())
                self.assertEqual(errors, [])
                self.assertEqual(queried[0]["count"], 1)
                self.assertEqual(
                    queried[0]["results"][0]["attachment_id"], attachment_id
                )

    def test_reindex_runtime_error_restores_all_authority_state(self) -> None:
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
                    content_base64=b64encode(b"runtime rollback evidence").decode(),
                )
                attachment_id = admitted["attachment_id"]
                service.index_attachments([attachment_id])
                assert service.akasicdb is not None
                original_index = service.akasicdb.index_document
                previous_catalog = service.catalog_path.read_bytes()
                previous_ids = set(service._indexed_attachment_ids)
                previous_snapshots = dict(service._source_snapshots)
                previous_generation = service._source_generation
                failed_once = False

                def fail_once(**kwargs):
                    nonlocal failed_once
                    if not failed_once:
                        failed_once = True
                        raise RuntimeError("synthetic adapter failure")
                    return original_index(**kwargs)

                with patch.object(
                    service.akasicdb, "index_document", side_effect=fail_once
                ):
                    with self.assertRaises(WorkspaceCapabilityError) as captured:
                        service.reindex_attachments([attachment_id])
                self.assertEqual(captured.exception.code, "RAG_REINDEX_FAILED")
                self.assertEqual(service.catalog_path.read_bytes(), previous_catalog)
                self.assertEqual(service._indexed_attachment_ids, previous_ids)
                self.assertEqual(service._source_snapshots, previous_snapshots)
                self.assertEqual(service._source_generation, previous_generation)
                self.assertEqual(service.query_rag("rollback evidence")["count"], 1)

    def test_delete_runtime_error_restores_all_authority_state(self) -> None:
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
                first = service.add_attachment(
                    name="delete-target.md",
                    media_type="text/markdown",
                    content_base64=b64encode(b"delete rollback authority").decode(),
                )
                second = service.add_attachment(
                    name="retained.md",
                    media_type="text/markdown",
                    content_base64=b64encode(b"retained rebuild authority").decode(),
                )
                first_id = first["attachment_id"]
                second_id = second["attachment_id"]
                service.index_attachments([first_id, second_id])
                assert service.akasicdb is not None
                original_index = service.akasicdb.index_document
                previous_catalog = service.catalog_path.read_bytes()
                previous_ids = set(service._indexed_attachment_ids)
                previous_snapshots = dict(service._source_snapshots)
                previous_generation = service._source_generation
                stored_path = service._attachments[first_id].stored_path
                failed_once = False

                def fail_once(**kwargs):
                    nonlocal failed_once
                    if not failed_once:
                        failed_once = True
                        raise RuntimeError("synthetic delete rebuild failure")
                    return original_index(**kwargs)

                with patch.object(
                    service.akasicdb, "index_document", side_effect=fail_once
                ):
                    with self.assertRaises(WorkspaceCapabilityError) as captured:
                        service.delete_attachment(first_id)
                self.assertEqual(captured.exception.code, "ATTACHMENT_DELETE_FAILED")
                self.assertTrue(stored_path.is_file())
                self.assertEqual(service.catalog_path.read_bytes(), previous_catalog)
                self.assertEqual(service._indexed_attachment_ids, previous_ids)
                self.assertEqual(service._source_snapshots, previous_snapshots)
                self.assertEqual(service._source_generation, previous_generation)
                self.assertEqual(service.query_rag("delete rollback")["count"], 1)

    def test_index_double_failure_quarantines_rag(self) -> None:
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
                first = service.add_attachment(
                    name="committed.md",
                    media_type="text/markdown",
                    content_base64=b64encode(b"committed authority evidence").decode(),
                )
                second = service.add_attachment(
                    name="candidate.md",
                    media_type="text/markdown",
                    content_base64=b64encode(b"candidate authority evidence").decode(),
                )
                first_id = first["attachment_id"]
                second_id = second["attachment_id"]
                service.index_attachments([first_id])
                adapter = service.akasicdb
                assert adapter is not None
                previous_catalog = service.catalog_path.read_bytes()
                previous_records = dict(service._attachments)
                previous_ids = set(service._indexed_attachment_ids)
                previous_snapshots = dict(service._source_snapshots)
                previous_generation = service._source_generation

                with patch.object(
                    adapter,
                    "index_document",
                    side_effect=RuntimeError("persistent adapter failure"),
                ):
                    with self.assertRaises(WorkspaceCapabilityError) as captured:
                        service.index_attachments([second_id])

                self.assertEqual(captured.exception.code, "RAG_INDEX_ROLLBACK_FAILED")
                self.assertEqual(service.catalog_path.read_bytes(), previous_catalog)
                self.assertEqual(service._attachments, previous_records)
                self.assertEqual(service._indexed_attachment_ids, previous_ids)
                self.assertEqual(service._source_snapshots, previous_snapshots)
                self.assertEqual(service._source_generation, previous_generation)
                self._assert_rag_quarantined(
                    service,
                    attachment_id=first_id,
                    query="committed authority",
                )
                self.assertTrue(service.rag_quarantine_path.is_file())
                restarted = WorkspaceCapabilityService(
                    project,
                    _model(),
                    akasicdb_path=clone,
                    answer_integration_enabled=True,
                )
                self._assert_rag_quarantined(
                    restarted,
                    attachment_id=first_id,
                    query="committed authority",
                )

    def test_reindex_double_failure_quarantines_rag(self) -> None:
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
                    name="reindex.md",
                    media_type="text/markdown",
                    content_base64=b64encode(b"reindex authority evidence").decode(),
                )
                attachment_id = admitted["attachment_id"]
                service.index_attachments([attachment_id])
                adapter = service.akasicdb
                assert adapter is not None
                previous_catalog = service.catalog_path.read_bytes()
                previous_records = dict(service._attachments)
                previous_ids = set(service._indexed_attachment_ids)
                previous_snapshots = dict(service._source_snapshots)
                previous_generation = service._source_generation

                with patch.object(
                    adapter,
                    "index_document",
                    side_effect=RuntimeError("persistent adapter failure"),
                ):
                    with self.assertRaises(WorkspaceCapabilityError) as captured:
                        service.reindex_attachments([attachment_id])

                self.assertEqual(captured.exception.code, "RAG_REINDEX_ROLLBACK_FAILED")
                self.assertEqual(service.catalog_path.read_bytes(), previous_catalog)
                self.assertEqual(service._attachments, previous_records)
                self.assertEqual(service._indexed_attachment_ids, previous_ids)
                self.assertEqual(service._source_snapshots, previous_snapshots)
                self.assertEqual(service._source_generation, previous_generation)
                self._assert_rag_quarantined(
                    service,
                    attachment_id=attachment_id,
                    query="reindex authority",
                )
                self.assertTrue(service.rag_quarantine_path.is_file())
                restarted = WorkspaceCapabilityService(
                    project,
                    _model(),
                    akasicdb_path=clone,
                    answer_integration_enabled=True,
                )
                self._assert_rag_quarantined(
                    restarted,
                    attachment_id=attachment_id,
                    query="reindex authority",
                )

    def test_delete_double_failure_quarantines_rag(self) -> None:
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
                first = service.add_attachment(
                    name="delete.md",
                    media_type="text/markdown",
                    content_base64=b64encode(b"delete authority evidence").decode(),
                )
                second = service.add_attachment(
                    name="retain.md",
                    media_type="text/markdown",
                    content_base64=b64encode(b"retain authority evidence").decode(),
                )
                first_id = first["attachment_id"]
                second_id = second["attachment_id"]
                service.index_attachments([first_id, second_id])
                adapter = service.akasicdb
                assert adapter is not None
                previous_catalog = service.catalog_path.read_bytes()
                previous_records = dict(service._attachments)
                previous_ids = set(service._indexed_attachment_ids)
                previous_snapshots = dict(service._source_snapshots)
                previous_generation = service._source_generation
                stored_path = service._attachments[first_id].stored_path

                with patch.object(
                    adapter,
                    "index_document",
                    side_effect=RuntimeError("persistent adapter failure"),
                ):
                    with self.assertRaises(WorkspaceCapabilityError) as captured:
                        service.delete_attachment(first_id)

                self.assertEqual(
                    captured.exception.code, "ATTACHMENT_DELETE_ROLLBACK_FAILED"
                )
                self.assertTrue(stored_path.is_file())
                self.assertEqual(service.catalog_path.read_bytes(), previous_catalog)
                self.assertEqual(service._attachments, previous_records)
                self.assertEqual(service._indexed_attachment_ids, previous_ids)
                self.assertEqual(service._source_snapshots, previous_snapshots)
                self.assertEqual(service._source_generation, previous_generation)
                self._assert_rag_quarantined(
                    service,
                    attachment_id=first_id,
                    query="delete authority",
                )
                self.assertTrue(service.rag_quarantine_path.is_file())
                restarted = WorkspaceCapabilityService(
                    project,
                    _model(),
                    akasicdb_path=clone,
                    answer_integration_enabled=True,
                )
                self._assert_rag_quarantined(
                    restarted,
                    attachment_id=first_id,
                    query="delete authority",
                )

    def test_delete_catalog_rollback_failure_cannot_promote_orphan_on_restart(
        self,
    ) -> None:
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
                victim = service.add_attachment(
                    name="victim.md",
                    media_type="text/markdown",
                    content_base64=b64encode(b"victim authority evidence").decode(),
                )
                retained = service.add_attachment(
                    name="retained.md",
                    media_type="text/markdown",
                    content_base64=b64encode(b"retained authority evidence").decode(),
                )
                victim_id = victim["attachment_id"]
                retained_id = retained["attachment_id"]
                service.index_attachments([victim_id, retained_id])
                adapter = service.akasicdb
                assert adapter is not None
                original_persist = service._persist_catalog
                persist_calls = 0

                def persist_post_delete_then_fail_rollback() -> None:
                    nonlocal persist_calls
                    persist_calls += 1
                    if persist_calls == 1:
                        original_persist()
                        return
                    raise WorkspaceCapabilityError(
                        "ATTACHMENT_CATALOG_WRITE_FAILED",
                        "injected rollback persistence failure",
                    )

                with (
                    patch.object(
                        service,
                        "_persist_catalog",
                        side_effect=persist_post_delete_then_fail_rollback,
                    ),
                    patch.object(
                        adapter,
                        "index_document",
                        side_effect=RuntimeError("injected rebuild failure"),
                    ),
                ):
                    with self.assertRaises(WorkspaceCapabilityError) as captured:
                        service.delete_attachment(victim_id)

                self.assertEqual(
                    captured.exception.code, "ATTACHMENT_DELETE_ROLLBACK_FAILED"
                )
                self.assertNotIn(victim_id, service.catalog_path.read_text())
                orphan = service.storage_root / f"{victim_id}.md"
                self.assertTrue(orphan.is_file())
                restarted = WorkspaceCapabilityService(
                    project,
                    _model(),
                    akasicdb_path=clone,
                    answer_integration_enabled=True,
                )
                self.assertEqual(restarted.list_attachments()["count"], 1)
                self.assertNotIn(victim_id, restarted._attachments)
                self.assertTrue(orphan.is_file())
                self._assert_rag_quarantined(
                    restarted,
                    attachment_id=retained_id,
                    query="retained authority",
                )

    def test_capability_waits_for_rag_transaction_and_never_reports_partial_ready(
        self,
    ) -> None:
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
                first = service.add_attachment(
                    name="first.md",
                    media_type="text/markdown",
                    content_base64=b64encode(b"first authority evidence").decode(),
                )
                second = service.add_attachment(
                    name="second.md",
                    media_type="text/markdown",
                    content_base64=b64encode(b"second authority evidence").decode(),
                )
                first_id = first["attachment_id"]
                second_id = second["attachment_id"]
                service.index_attachments([first_id])
                adapter = service.akasicdb
                assert adapter is not None
                index_entered = Event()
                release_index = Event()
                capability_started = Event()
                capability_done = Event()
                mutation_errors: list[str] = []
                capability_results: list[dict[str, object]] = []
                index_calls = 0

                def fail_both_mutation_and_rollback(**_kwargs: object) -> None:
                    nonlocal index_calls
                    index_calls += 1
                    if index_calls == 1:
                        index_entered.set()
                        release_index.wait(2)
                    raise RuntimeError("injected persistent adapter failure")

                def mutate() -> None:
                    try:
                        service.index_attachments([second_id])
                    except WorkspaceCapabilityError as exc:
                        mutation_errors.append(exc.code)

                def read_capability() -> None:
                    capability_started.set()
                    capability_results.append(service.capability_payload()["rag"])
                    capability_done.set()

                with patch.object(
                    adapter,
                    "index_document",
                    side_effect=fail_both_mutation_and_rollback,
                ):
                    mutation_thread = Thread(target=mutate)
                    mutation_thread.start()
                    self.assertTrue(index_entered.wait(2))
                    capability_thread = Thread(target=read_capability)
                    capability_thread.start()
                    self.assertTrue(capability_started.wait(2))
                    self.assertFalse(capability_done.wait(0.05))
                    release_index.set()
                    mutation_thread.join(2)
                    capability_thread.join(2)

                self.assertFalse(mutation_thread.is_alive())
                self.assertFalse(capability_thread.is_alive())
                self.assertEqual(mutation_errors, ["RAG_INDEX_ROLLBACK_FAILED"])
                self.assertEqual(capability_results[0]["state"], "unavailable")
                self.assertFalse(capability_results[0]["answer_integration"])

    def test_quarantine_marker_tamper_variants_fail_closed(self) -> None:
        variants = {
            "truncated": b'{"schema_version":1',
            "duplicate": (
                b'{"schema_version":1,"state":"pending",'
                b'"reason":"RAG_OPERATOR_REVIEW_REQUIRED",'
                b'"reason":"RAG_OPERATOR_REVIEW_REQUIRED","catalog_sha256":"'
                + b"a" * 64
                + b'"}'
            ),
            "unknown": json.dumps(
                {
                    "schema_version": 1,
                    "state": "pending",
                    "reason": "RAG_OPERATOR_REVIEW_REQUIRED",
                    "catalog_sha256": "a" * 64,
                    "unexpected": True,
                }
            ).encode(),
            "bad_digest": json.dumps(
                {
                    "schema_version": 1,
                    "state": "pending",
                    "reason": "RAG_OPERATOR_REVIEW_REQUIRED",
                    "catalog_sha256": "not-a-digest",
                }
            ).encode(),
            "invalid_state": json.dumps(
                {
                    "schema_version": 1,
                    "state": "unknown",
                    "reason": "RAG_OPERATOR_REVIEW_REQUIRED",
                    "catalog_sha256": "a" * 64,
                }
            ).encode(),
            "committed_with_pending_reason": json.dumps(
                {
                    "schema_version": 1,
                    "state": "committed",
                    "reason": "RAG_OPERATOR_REVIEW_REQUIRED",
                    "catalog_sha256": "a" * 64,
                }
            ).encode(),
            "oversized": b"x" * (capabilities.MAX_RAG_QUARANTINE_MARKER_BYTES + 1),
        }
        for label, marker_bytes in variants.items():
            with (
                self.subTest(marker_variant=label),
                tempfile.TemporaryDirectory() as temporary,
            ):
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
                        name="marker.md",
                        media_type="text/markdown",
                        content_base64=b64encode(b"marker authority evidence").decode(),
                    )
                    service.rag_quarantine_path.write_bytes(marker_bytes)
                    restarted = WorkspaceCapabilityService(
                        project,
                        _model(),
                        akasicdb_path=clone,
                        answer_integration_enabled=True,
                    )
                    self._assert_rag_quarantined(
                        restarted,
                        attachment_id=admitted["attachment_id"],
                        query="marker authority",
                    )

    def test_quarantine_marker_symlink_fails_closed(self) -> None:
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
                    name="symlink.md",
                    media_type="text/markdown",
                    content_base64=b64encode(b"symlink authority evidence").decode(),
                )
                target = root / "marker-target.json"
                target.write_text("{}", encoding="utf-8")
                try:
                    service.rag_quarantine_path.symlink_to(target)
                except OSError as exc:
                    self.skipTest(f"symlink creation unavailable: {exc}")
                restarted = WorkspaceCapabilityService(
                    project,
                    _model(),
                    akasicdb_path=clone,
                    answer_integration_enabled=True,
                )
                self._assert_rag_quarantined(
                    restarted,
                    attachment_id=admitted["attachment_id"],
                    query="symlink authority",
                )

    def test_marker_publication_failure_aborts_before_reset_and_quarantines_process(
        self,
    ) -> None:
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
                    name="publication.md",
                    media_type="text/markdown",
                    content_base64=b64encode(
                        b"publication authority evidence"
                    ).decode(),
                )
                attachment_id = admitted["attachment_id"]
                adapter = service.akasicdb
                assert adapter is not None
                catalog_before = service.catalog_path.read_bytes()
                with (
                    patch.object(
                        capabilities,
                        "_atomic_rag_quarantine_write",
                        side_effect=OSError("injected marker publication failure"),
                    ),
                    patch.object(adapter, "reset", wraps=adapter.reset) as reset,
                ):
                    with self.assertRaises(WorkspaceCapabilityError) as captured:
                        service.index_attachments([attachment_id])
                self.assertEqual(
                    captured.exception.code, "RAG_OPERATOR_REVIEW_REQUIRED"
                )
                reset.assert_not_called()
                self.assertEqual(service.catalog_path.read_bytes(), catalog_before)
                self.assertFalse(service.rag_quarantine_path.exists())
                self._assert_rag_quarantined(
                    service,
                    attachment_id=attachment_id,
                    query="publication authority",
                )

    def test_commit_failure_before_atomic_replace_leaves_pending_on_restart(
        self,
    ) -> None:
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
                    name="before-commit.md",
                    media_type="text/markdown",
                    content_base64=b64encode(b"before commit authority").decode(),
                )
                attachment_id = admitted["attachment_id"]
                original_write = capabilities._atomic_rag_transaction_marker_write

                def fail_before_committed_replace(
                    path: Path,
                    *,
                    state: str,
                    catalog_sha256: str,
                    replace_existing: bool,
                ) -> None:
                    if state == capabilities._RAG_TRANSACTION_COMMITTED:
                        raise OSError("injected crash before committed replace")
                    original_write(
                        path,
                        state=state,
                        catalog_sha256=catalog_sha256,
                        replace_existing=replace_existing,
                    )

                with patch.object(
                    capabilities,
                    "_atomic_rag_transaction_marker_write",
                    side_effect=fail_before_committed_replace,
                ):
                    with self.assertRaises(WorkspaceCapabilityError) as captured:
                        service.index_attachments([attachment_id])
                self.assertEqual(
                    captured.exception.code, "RAG_OPERATOR_REVIEW_REQUIRED"
                )
                marker = capabilities._load_rag_quarantine_marker(
                    service.rag_quarantine_path
                )
                assert marker is not None
                self.assertEqual(marker.state, "pending")
                restarted = WorkspaceCapabilityService(
                    project,
                    _model(),
                    akasicdb_path=clone,
                    answer_integration_enabled=True,
                )
                self._assert_rag_quarantined(
                    restarted,
                    attachment_id=attachment_id,
                    query="before commit authority",
                )

    def test_commit_fsync_failure_after_atomic_replace_is_restart_safe(self) -> None:
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
                    name="after-commit.md",
                    media_type="text/markdown",
                    content_base64=b64encode(b"after commit authority").decode(),
                )
                attachment_id = admitted["attachment_id"]
                original_fsync = capabilities._fsync_directory

                def fail_after_committed_replace(
                    path: Path, *, error_code: str, message: str
                ) -> None:
                    marker = capabilities._load_rag_quarantine_marker(
                        service.rag_quarantine_path
                    )
                    if (
                        error_code == "RAG_QUARANTINE_MARKER_WRITE_FAILED"
                        and marker is not None
                        and marker.state == capabilities._RAG_TRANSACTION_COMMITTED
                    ):
                        raise WorkspaceCapabilityError(error_code, message)
                    original_fsync(path, error_code=error_code, message=message)

                with patch.object(
                    capabilities,
                    "_fsync_directory",
                    side_effect=fail_after_committed_replace,
                ):
                    with self.assertRaises(WorkspaceCapabilityError) as captured:
                        service.index_attachments([attachment_id])
                self.assertEqual(
                    captured.exception.code, "RAG_OPERATOR_REVIEW_REQUIRED"
                )
                marker = capabilities._load_rag_quarantine_marker(
                    service.rag_quarantine_path
                )
                assert marker is not None
                self.assertEqual(marker.state, "committed")
                restarted = WorkspaceCapabilityService(
                    project,
                    _model(),
                    akasicdb_path=clone,
                    answer_integration_enabled=True,
                )
                self.assertIsNotNone(restarted.akasicdb)
                self.assertEqual(
                    restarted.query_rag("after commit authority")["count"], 1
                )
                self.assertFalse(restarted.rag_quarantine_path.exists())

    def test_committed_digest_mismatch_requires_operator_review(self) -> None:
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
                    name="mismatch.md",
                    media_type="text/markdown",
                    content_base64=b64encode(b"mismatch authority").decode(),
                )
                attachment_id = admitted["attachment_id"]
                capabilities._atomic_rag_transaction_marker_write(
                    service.rag_quarantine_path,
                    state=capabilities._RAG_TRANSACTION_COMMITTED,
                    catalog_sha256="a" * 64,
                    replace_existing=False,
                )
                restarted = WorkspaceCapabilityService(
                    project,
                    _model(),
                    akasicdb_path=clone,
                    answer_integration_enabled=True,
                )
                self._assert_rag_quarantined(
                    restarted,
                    attachment_id=attachment_id,
                    query="mismatch authority",
                )

    def test_leftover_committed_marker_is_replaced_by_next_transaction(self) -> None:
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
                    name="leftover.md",
                    media_type="text/markdown",
                    content_base64=b64encode(b"leftover authority").decode(),
                )
                attachment_id = admitted["attachment_id"]
                with patch.object(
                    capabilities,
                    "_cleanup_committed_rag_marker",
                    side_effect=OSError("injected cleanup skip"),
                ):
                    service.index_attachments([attachment_id])
                marker = capabilities._load_rag_quarantine_marker(
                    service.rag_quarantine_path
                )
                assert marker is not None
                self.assertEqual(marker.state, "committed")

                rebuilt = service.reindex_attachments([attachment_id])
                self.assertEqual(rebuilt["documents"], 1)
                self.assertFalse(service.rag_quarantine_path.exists())
                self.assertEqual(service.query_rag("leftover authority")["count"], 1)

    def test_committed_cleanup_failure_remains_restart_safe(self) -> None:
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
                    name="clear.md",
                    media_type="text/markdown",
                    content_base64=b64encode(b"clear authority evidence").decode(),
                )
                attachment_id = admitted["attachment_id"]
                with patch.object(
                    capabilities,
                    "_cleanup_committed_rag_marker",
                    side_effect=OSError("injected marker clear failure"),
                ):
                    indexed = service.index_attachments([attachment_id])
                self.assertEqual(indexed["documents"], 1)
                marker = capabilities._load_rag_quarantine_marker(
                    service.rag_quarantine_path
                )
                assert marker is not None
                self.assertEqual(marker.state, "committed")
                self.assertTrue(service.rag_quarantine_path.is_file())
                restarted = WorkspaceCapabilityService(
                    project,
                    _model(),
                    akasicdb_path=clone,
                    answer_integration_enabled=True,
                )
                self.assertIsNotNone(restarted.akasicdb)
                self.assertEqual(restarted.query_rag("clear authority")["count"], 1)
                self.assertFalse(restarted.rag_quarantine_path.exists())

    def test_committed_unlink_then_directory_fsync_failure_remains_restart_safe(
        self,
    ) -> None:
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
                    name="fsync-clear.md",
                    media_type="text/markdown",
                    content_base64=b64encode(
                        b"fsync clear authority evidence"
                    ).decode(),
                )
                attachment_id = admitted["attachment_id"]
                original_fsync = capabilities._fsync_directory

                def fail_only_clear_fsync(
                    path: Path, *, error_code: str, message: str
                ) -> None:
                    if error_code == "RAG_QUARANTINE_MARKER_CLEAR_FAILED":
                        raise WorkspaceCapabilityError(error_code, message)
                    original_fsync(path, error_code=error_code, message=message)

                with patch.object(
                    capabilities,
                    "_fsync_directory",
                    side_effect=fail_only_clear_fsync,
                ):
                    indexed = service.index_attachments([attachment_id])
                self.assertEqual(indexed["documents"], 1)
                self.assertFalse(service.rag_quarantine_path.exists())
                restarted = WorkspaceCapabilityService(
                    project,
                    _model(),
                    akasicdb_path=clone,
                    answer_integration_enabled=True,
                )
                self.assertIsNotNone(restarted.akasicdb)
                self.assertEqual(
                    restarted.query_rag("fsync clear authority")["count"], 1
                )

    def test_query_runtime_error_is_normalized(self) -> None:
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
                assert service.akasicdb is not None
                with patch.object(
                    service.akasicdb,
                    "query",
                    side_effect=RuntimeError("synthetic query failure"),
                ):
                    with self.assertRaises(WorkspaceCapabilityError) as captured:
                        service.query_rag("bounded query")
                self.assertEqual(captured.exception.code, "RAG_QUERY_FAILED")

                invalid_results = (
                    {"query": "bounded query", "results": [], "count": 1},
                    {"query": "bounded query", "results": [None], "count": 1},
                )
                for invalid in invalid_results:
                    with (
                        self.subTest(invalid=invalid),
                        patch.object(service.akasicdb, "query", return_value=invalid),
                        self.assertRaises(WorkspaceCapabilityError) as captured,
                    ):
                        service.query_rag("bounded query")
                    self.assertEqual(
                        captured.exception.code, "RAG_QUERY_INTEGRITY_FAILED"
                    )

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
