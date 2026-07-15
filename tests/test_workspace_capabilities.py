from __future__ import annotations

from base64 import b64encode
from hashlib import sha256
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
                "process_memory",
            )
            blobs = list(service.storage_root.iterdir())
            self.assertEqual(len(blobs), 1)
            self.assertEqual(blobs[0].read_bytes(), content)
            self.assertFalse(any(path.name.endswith(".tmp") for path in blobs))

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

    def test_process_restart_does_not_claim_a_persistent_catalog(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            first = WorkspaceCapabilityService(temporary, _model())
            encoded = b64encode(b"local evidence").decode("ascii")
            first.add_attachment(
                name="one.txt", media_type="text/plain", content_base64=encoded
            )
            second = WorkspaceCapabilityService(temporary, _model())
            self.assertEqual(second.list_attachments()["count"], 0)
            admitted = second.add_attachment(
                name="one.txt", media_type="text/plain", content_base64=encoded
            )
            self.assertFalse(admitted["duplicate"])
            self.assertEqual(
                second.capability_payload()["attachments"]["persisted_blob_count"],
                1,
            )

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
            self.assertEqual(attachments["catalog_lifetime"], "process_memory")
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
            self.assertFalse(admitted["duplicate"])
            self.assertEqual(len(list(restarted.storage_root.iterdir())), 1)
            self.assertEqual(
                restarted.capability_payload()["attachments"]["persisted_blob_count"],
                1,
            )

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
                self.assertGreater(result["results"][0]["score"], 0)
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
