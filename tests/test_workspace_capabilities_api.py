from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from http.client import HTTPConnection
import json
from pathlib import Path
import tempfile
from threading import Event, Thread
from time import monotonic, sleep
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from cogni_agent.manager import (
    AgentManager,
    AgentTurnStartError,
    ImagePublicationEvidence,
)
from cogni_agent.model_service import ModelRuntimeIdentity
from cogni_agent.tools import WorkspaceToolExecutor
from cogni_demo.server import (
    DemoHTTPServer,
    DemoRequestHandler,
    ImageModelUnavailableError,
    MAX_AGENT_CHAT_REQUEST_BODY_BYTES,
    MAX_INDEXED_TEXT_CHARS,
    MAX_REQUEST_BODY_BYTES,
)
from cogni_demo.workspace_capabilities import WorkspaceCapabilityError
from tests.test_demo_server import manager_for


def _attested_image_status() -> dict[str, object]:
    return {
        "state": "ready",
        "selected_model_only": True,
        "configured": True,
        "processor_probed": True,
        "model_inference_attested": True,
        "runtime_ready": True,
        "disabled_reason": None,
    }


def _configured_unverified_image_status() -> dict[str, object]:
    return {
        "state": "configured_unverified",
        "selected_model_only": True,
        "configured": True,
        "processor_probed": False,
        "model_inference_attested": False,
        "runtime_ready": False,
        "first_use_attestation_allowed": True,
        "disabled_reason": "IMAGE_MODEL_INFERENCE_NOT_ATTESTED",
    }


@dataclass(frozen=True)
class _Evidence:
    source_id: str
    title: str
    text: str
    score: float | None = None
    provenance: object | None = None


@dataclass(frozen=True)
class _ImageProbe:
    turn_id: str
    image_sha256: str
    started_at: str


@dataclass(frozen=True)
class _RuntimeIdentity:
    service_nonce: str = "1" * 32
    worker_incarnation: int = 1
    worker_pid: int = 4321
    lease_epoch: int = 7
    lease_deadline_ns: int = 99
    artifact_digest: str = "a" * 64
    model_root: str = "model"
    manifest_path: str = "manifest"
    processor_root: str = "model"
    processor_manifest_path: str = "manifest"


class _RuntimeService:
    def __init__(self, identity: object | None) -> None:
        self.identity = identity

    def runtime_identity(self) -> object | None:
        return self.identity


class _ImageTokenizer:
    eos_token_id = 3

    def decode(self, tokens, **_kwargs):
        return "".join(chr(value) for value in tokens)

    def apply_chat_template(self, messages, **_kwargs):
        return "|".join(f"{item['role']}:{item['content']}" for item in messages)


class _ProductionPathImageService:
    def __init__(
        self,
        *,
        entered: Event | None = None,
        release: Event | None = None,
        error: BaseException | None = None,
    ) -> None:
        self.tokenizer = _ImageTokenizer()
        self.active_request_id = None
        self.entered = entered
        self.release = release
        self.error = error
        self.identity = ModelRuntimeIdentity(
            service_nonce="1" * 32,
            worker_incarnation=1,
            worker_pid=4321,
            lease_epoch=7,
            lease_deadline_ns=99,
            artifact_digest="a" * 64,
            model_root="model",
            manifest_path="manifest",
            processor_root="model",
            processor_manifest_path="manifest",
        )

    def start(self):
        return self

    def iter_generate_tokens(
        self,
        _prompt,
        *,
        image_content=None,
        max_new_tokens,
        **_kwargs,
    ):
        del max_new_tokens
        self.active_request_id = 1
        if self.entered is not None:
            self.entered.set()
        if self.release is not None and not self.release.wait(2):
            raise TimeoutError("production image test release timed out")
        if self.error is not None:
            raise self.error
        text = "이미지 중앙에는 파란 원이 선명하게 배치되어 있습니다."
        tokens = torch.tensor([ord(character) for character in text], dtype=torch.int64)
        yield SimpleNamespace(
            request_id=1,
            token_ids=tokens,
            generated_total=int(tokens.numel()),
            final=True,
            cancelled=False,
            finish_reason="stop",
            generation_mode="cogni_core",
            media_sha256=sha256(image_content).hexdigest(),
            runtime_identity=self.identity,
        )
        self.active_request_id = None

    def cancel(self, _request_id=None):
        return True

    def stop(self, timeout=10.0):
        del timeout


def _wait_for_agent(manager: AgentManager, timeout: float = 5.0) -> dict[str, object]:
    deadline = monotonic() + timeout
    while monotonic() < deadline:
        snapshot = manager.snapshot()
        if not manager.is_active:
            return snapshot
        sleep(0.01)
    raise AssertionError("production AgentManager path did not finish")


def _wait_for_probe_clear(server: DemoHTTPServer, timeout: float = 5.0) -> None:
    deadline = monotonic() + timeout
    while monotonic() < deadline:
        if server._image_attestation_probe is None:
            return
        sleep(0.01)
    raise AssertionError("first-use image probe did not clear")


def _terminal_image_evidence(
    identity: _RuntimeIdentity,
    image_sha256: str,
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "image_sha256": image_sha256,
        "runtime_identity": {
            "service_nonce": identity.service_nonce,
            "worker_incarnation": identity.worker_incarnation,
            "worker_pid": identity.worker_pid,
            "lease_epoch": identity.lease_epoch,
            "lease_deadline_ns": identity.lease_deadline_ns,
            "artifact_digest": identity.artifact_digest,
            "model_root": identity.model_root,
            "manifest_path": identity.manifest_path,
            "processor_root": identity.processor_root,
            "processor_manifest_path": identity.processor_manifest_path,
        },
    }


class _Agent:
    def __init__(self) -> None:
        self.availability_check = None
        self.evidence: tuple[object, ...] = ()
        self.image_content: bytes | None = None
        self.expected_image_runtime_identity: object | None = None
        self.first_use_gate: object | None = None
        self.image_publication_authorizer = None
        self.messages: list[str] = []
        self.retrieval_requested = False
        self.shutdown_called = False
        self.reset_calls = 0
        self.cancel_calls = 0
        self.last_finished_turn = None
        self.snapshot_event = Event()

    @property
    def is_active(self) -> bool:
        return False

    def start_turn(
        self,
        _message,
        _mode="chat",
        *,
        evidence=(),
        image_content=None,
        expected_image_runtime_identity=None,
        first_use_gate=None,
        image_publication_authorizer=None,
        retrieval_requested=False,
    ):
        self.messages.append(_message)
        self.evidence = tuple(evidence)
        self.image_content = image_content
        self.expected_image_runtime_identity = expected_image_runtime_identity
        self.first_use_gate = first_use_gate
        self.image_publication_authorizer = image_publication_authorizer
        self.retrieval_requested = retrieval_requested
        return "turn-rag"

    def cancel(self):
        self.cancel_calls += 1

    def snapshot(self):
        return {
            "status": "ready",
            "seq": 1,
            "conversation": [],
            "last_finished_turn": self.last_finished_turn,
        }

    def wait_snapshot(self, _after, timeout=1.0):
        self.snapshot_event.wait(timeout)
        return self.snapshot()

    def stop_model(self):
        return None

    def reset(self):
        self.reset_calls += 1

    def shutdown(self):
        self.shutdown_called = True


class _Workspace:
    def __init__(self) -> None:
        self.added: dict[str, object] | None = None
        self.raise_add: WorkspaceCapabilityError | None = None
        self.rag_results: object = [
            {
                "attachment_id": "a" * 24,
                "chunk_index": 0,
                "name": "paper.md",
                "text": "검증된 로컬 검색 근거",
                "score": 0.75,
            },
            {
                "attachment_id": "a" * 24,
                "chunk_index": 1,
                "name": "paper.md",
                "text": "무관한 근거",
                "score": 0.0,
            },
        ]
        self.queries: list[str] = []
        self.lens_queries: list[dict[str, object]] = []
        self.web_queries: list[dict[str, object]] = []
        self.web_cancellations: list[str] = []
        self.source_requests: list[tuple[str, int]] = []
        self.raise_source: WorkspaceCapabilityError | None = None
        self.source_payload: dict[str, object] | None = None

    def capability_payload(self):
        return {
            "schema_version": 1,
            "attachments": {"state": "enabled"},
            "rag": {"state": "local_index_ready"},
            "models": {
                "items": [
                    {
                        "selected": True,
                        "checkpoint_modalities": ["text", "image"],
                        "runtime_input_modalities": ["text"],
                        "unwired_checkpoint_modalities": ["image"],
                    }
                ]
            },
        }

    def list_attachments(self):
        return {"items": [], "count": 0}

    def add_attachment(self, **body):
        if self.raise_add is not None:
            raise self.raise_add
        self.added = body
        return {"attachment_id": "abc", "storage": "local_content_addressed"}

    def delete_attachment(self, attachment_id):
        return {
            "attachment_id": attachment_id,
            "deleted": True,
            "index_removed": True,
            "blob_deleted": True,
            "remaining": 0,
            "indexed_documents": 0,
        }

    def preview_attachment(self, attachment_id):
        return {
            "attachment_id": attachment_id,
            "name": "paper.txt",
            "media_type": "text/plain",
            "size_bytes": 8,
            "kind": "text",
            "text": "evidence",
            "truncated": False,
            "max_chars": 12000,
            "extraction": "utf8",
        }

    def image_attachment_content(self, attachment_id):
        if attachment_id != "a" * 24:
            raise WorkspaceCapabilityError(
                "ATTACHMENT_NOT_FOUND", "private path must not leak"
            )
        return b"\x89PNG\r\n\x1a\n", "image/png"

    def index_attachments(self, attachment_ids):
        return {"results": list(attachment_ids), "answer_integration": False}

    def reindex_attachments(self, attachment_ids):
        return {
            "reindexed_attachment_ids": list(attachment_ids),
            "documents": len(attachment_ids),
            "chunks": len(attachment_ids),
        }

    def query_rag(self, query, *, limit=5):
        self.queries.append(query)
        results = self.rag_results
        if isinstance(results, list):
            enriched = []
            for item in results[:limit]:
                selected = dict(item)
                text = selected.get("text")
                if isinstance(text, str):
                    selected.setdefault(
                        "excerpt_sha256", sha256(text.encode()).hexdigest()
                    )
                    selected.setdefault("char_start", 0)
                    selected.setdefault("char_end", len(text))
                attachment_id = selected.get("attachment_id")
                selected.setdefault(
                    "source_sha256",
                    (
                        attachment_id + "b" * 40
                        if isinstance(attachment_id, str) and len(attachment_id) == 24
                        else "b" * 64
                    ),
                )
                selected.setdefault("media_type", "text/markdown")
                selected.setdefault("representation", "normalized_extracted_excerpt_v1")
                selected.setdefault("page_number", None)
                selected.setdefault("offset_basis", "normalized_document_text_v1")
                enriched.append(selected)
            results = enriched
        return {
            "schema_version": 2,
            "engine": "AkasicDB",
            "repository": "https://github.com/heosanghun/AkasicDB.git",
            "revision": "a6c8e8ebd487e7cb86079f9804a66aaf0914d1dc",
            "retrieval_mode": "lexical_only",
            "embedding": "stable_sha256_lexical_sketch_v1",
            "semantic_embedding": False,
            "answer_integration": True,
            "answer_integration_schema": "cogni.agent.retrieval-evidence.v1",
            "query": query,
            "count": len(results) if isinstance(results, list) else 0,
            "results": results,
        }

    def preview_rag_source(self, attachment_id, chunk_index):
        self.source_requests.append((attachment_id, chunk_index))
        if self.raise_source is not None:
            raise self.raise_source
        if self.source_payload is not None:
            return self.source_payload
        selected = None
        if isinstance(self.rag_results, list):
            selected = next(
                (
                    item
                    for item in self.rag_results
                    if item.get("attachment_id") == attachment_id
                    and item.get("chunk_index") == chunk_index
                ),
                None,
            )
        selected = selected or {
            "name": "paper.md",
            "media_type": "text/markdown",
            "text": "검증된 로컬 검색 근거",
            "representation": "normalized_extracted_excerpt_v1",
            "page_number": None,
            "char_start": 0,
            "offset_basis": "normalized_document_text_v1",
        }
        text = selected["text"]
        char_start = selected.get("char_start", 0)
        return {
            "schema_version": 2,
            "attachment_id": attachment_id,
            "chunk_index": chunk_index,
            "name": selected.get("name", "paper.md"),
            "media_type": selected.get("media_type", "text/markdown"),
            "text": text,
            "representation": selected.get(
                "representation", "normalized_extracted_excerpt_v1"
            ),
            "page_number": selected.get("page_number"),
            "char_start": char_start,
            "char_end": char_start + len(text),
            "offset_basis": selected.get("offset_basis", "normalized_document_text_v1"),
            "excerpt_sha256": sha256(text.encode()).hexdigest(),
        }

    def current_rag_source_authority(self, attachment_id, chunk_index):
        return {
            "source": self.preview_rag_source(attachment_id, chunk_index),
            "source_sha256": attachment_id + "b" * 40,
        }

    def select_model(self, model_id):
        if model_id != "verified-model":
            raise WorkspaceCapabilityError(
                "MODEL_NOT_VERIFIED", "C:\\private\\model must not leak"
            )
        return {"model_id": model_id, "selected": True}

    def search_lens(self, kind, query, *, limit=5, index_in_akasicdb=False):
        request = {
            "kind": kind,
            "query": query,
            "limit": limit,
            "index_in_akasicdb": index_in_akasicdb,
        }
        self.lens_queries.append(request)
        search = {
            "provider": "Lens.org official API",
            "kind": kind,
            "total": 0,
            "count": 0,
            "results": [],
            "external_calls": 1,
        }
        return {"search": search, "indexed": []} if index_in_akasicdb else search

    def search_web(
        self, query, *, limit=5, session_online_opt_in, request_id=None
    ):
        request = {
            "query": query,
            "limit": limit,
            "session_online_opt_in": session_online_opt_in,
        }
        if request_id is not None:
            request["request_id"] = request_id
        self.web_queries.append(request)
        if session_online_opt_in is not True:
            raise WorkspaceCapabilityError(
                "WEB_SEARCH_SESSION_OPT_IN_REQUIRED", "private token must not leak"
            )
        return {
            "provider": "Brave Search official API",
            "query": query,
            "count": 0,
            "results": [],
            "external_calls": 1,
            "request_id": request_id,
        }

    def cancel_web_search(self, request_id):
        self.web_cancellations.append(request_id)
        return {"request_id": request_id, "cancelled": True, "state": "cancelled"}


class TestWorkspaceHTTPAPI(unittest.TestCase):
    def setUp(self) -> None:
        self.assets_context = tempfile.TemporaryDirectory()
        assets = Path(self.assets_context.name)
        (assets / "index.html").write_text("<main>Cogni</main>", encoding="utf-8")
        (assets / "app.css").write_text("body{}", encoding="utf-8")
        (assets / "app.js").write_text("void 0", encoding="utf-8")
        (assets / "favicon.svg").write_text("<svg/>", encoding="utf-8")
        self.validator = manager_for("success")
        self.agent = _Agent()
        self.workspace = _Workspace()
        self.server = DemoHTTPServer(
            self.validator,
            assets,
            agent_manager=self.agent,
            workspace_service=self.workspace,
            port=0,
            token="w" * 32,
            watchdog_timeout=None,
        )
        self.thread = Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.cookie = self._bootstrap()

    def tearDown(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)
        self.assets_context.cleanup()

    def _install_production_agent(
        self, service: _ProductionPathImageService
    ) -> AgentManager:
        manager = AgentManager(
            service,
            WorkspaceToolExecutor(Path(self.assets_context.name), timeout_seconds=5),
        )
        manager.availability_check = self.server._agent_compute_available
        self.server.agent_manager = manager
        return manager

    def _connection(self) -> HTTPConnection:
        return HTTPConnection("127.0.0.1", self.server.server_port, timeout=5)

    def _bootstrap(self) -> str:
        connection = self._connection()
        connection.request("GET", "/?token=" + self.server.token)
        response = connection.getresponse()
        response.read()
        self.assertEqual(response.status, 303)
        cookie = response.getheader("Set-Cookie").split(";", 1)[0]
        connection.close()
        return cookie

    def _get(self, path: str, *, authenticated: bool = True):
        connection = self._connection()
        headers = {"Cookie": self.cookie} if authenticated else {}
        connection.request("GET", path, headers=headers)
        response = connection.getresponse()
        payload = json.loads(response.read())
        status = response.status
        connection.close()
        return status, payload

    def _get_raw(self, path: str, *, authenticated: bool = True):
        connection = self._connection()
        headers = {"Cookie": self.cookie} if authenticated else {}
        connection.request("GET", path, headers=headers)
        response = connection.getresponse()
        payload = response.read()
        status = response.status
        content_type = response.getheader("Content-Type")
        connection.close()
        return status, payload, content_type

    def _post(self, path: str, body: dict[str, object]):
        connection = self._connection()
        connection.request(
            "POST",
            path,
            body=json.dumps(body).encode("utf-8"),
            headers={
                "Cookie": self.cookie,
                "Origin": self.server.origin,
                "Content-Type": "application/json",
            },
        )
        response = connection.getresponse()
        payload = json.loads(response.read())
        status = response.status
        connection.close()
        return status, payload

    def test_authenticated_exact_get_routes(self) -> None:
        status, payload = self._get("/api/workspace/capabilities")
        self.assertEqual(status, 200)
        self.assertEqual(payload["schema_version"], 1)
        status, payload = self._get("/api/workspace/attachments")
        self.assertEqual(status, 200)
        self.assertEqual(payload["count"], 0)
        attachment_id = "a" * 24
        status, payload = self._get(
            f"/api/workspace/attachments/preview?attachment_id={attachment_id}"
        )
        self.assertEqual(status, 200)
        self.assertEqual(payload["attachment_id"], attachment_id)
        status, content, content_type = self._get_raw(
            f"/api/workspace/attachments/content?attachment_id={attachment_id}"
        )
        self.assertEqual(status, 200)
        self.assertEqual(content, b"\x89PNG\r\n\x1a\n")
        self.assertEqual(content_type, "image/png")
        self.assertEqual(
            self._get("/api/workspace/capabilities", authenticated=False)[0], 403
        )
        self.assertEqual(self._get("/api/workspace/capabilities?debug=1")[0], 404)

    def test_exact_rag_source_get_is_authenticated_and_query_strict(self) -> None:
        attachment_id = "a" * 24
        route = (
            "/api/workspace/rag/source?attachment_id="
            + attachment_id
            + "&chunk_index=0"
        )
        status, payload = self._get(route)
        self.assertEqual(status, 200)
        self.assertEqual(
            set(payload),
            {
                "schema_version",
                "attachment_id",
                "chunk_index",
                "name",
                "media_type",
                "text",
                "representation",
                "page_number",
                "char_start",
                "char_end",
                "offset_basis",
                "excerpt_sha256",
            },
        )
        self.assertEqual(payload["schema_version"], 2)
        self.assertEqual(payload["attachment_id"], attachment_id)
        self.assertEqual(payload["chunk_index"], 0)
        self.assertEqual(payload["representation"], "normalized_extracted_excerpt_v1")
        self.assertEqual(
            payload["excerpt_sha256"], sha256(payload["text"].encode()).hexdigest()
        )
        self.assertEqual(self.workspace.source_requests, [(attachment_id, 0)])
        self.assertEqual(self._get(route, authenticated=False)[0], 403)

        invalid_routes = (
            "/api/workspace/rag/source",
            "/api/workspace/rag/source?attachment_id=" + attachment_id,
            "/api/workspace/rag/source?chunk_index=0",
            route + "&debug=1",
            route + "&attachment_id=" + attachment_id,
            route + "&chunk_index=1",
            "/api/workspace/rag/source?attachment_id=" + "A" * 24 + "&chunk_index=0",
            "/api/workspace/rag/source?attachment_id=" + "a" * 23 + "&chunk_index=0",
            "/api/workspace/rag/source?attachment_id=../../private&chunk_index=0",
            "/api/workspace/rag/source?attachment_id="
            + attachment_id
            + "&chunk_index=-1",
            "/api/workspace/rag/source?attachment_id="
            + attachment_id
            + "&chunk_index=128",
            "/api/workspace/rag/source?attachment_id="
            + attachment_id
            + "&chunk_index=00",
            "/api/workspace/rag/source?attachment_id="
            + attachment_id
            + "&chunk_index=%2B1",
            "/api/workspace/rag/source?chunk_index=0&attachment_id=" + attachment_id,
            route + "&",
            route.replace("&", "&&"),
            route.replace("attachment_id", "attachment%5fid"),
            route.replace(attachment_id, "%61" + attachment_id[1:]),
            route.replace("chunk_index", "chunk%5findex"),
            route.replace("chunk_index=0", "chunk_index=%30"),
        )
        for invalid in invalid_routes:
            status, error = self._get(invalid)
            self.assertEqual(status, 400, invalid)
            self.assertEqual(error["error"]["code"], "INVALID_QUERY", invalid)
        self.assertEqual(self.workspace.source_requests, [(attachment_id, 0)])

    def test_exact_rag_source_errors_and_malformed_payload_are_path_free(self) -> None:
        attachment_id = "a" * 24
        route = (
            "/api/workspace/rag/source?attachment_id="
            + attachment_id
            + "&chunk_index=0"
        )
        self.workspace.raise_source = WorkspaceCapabilityError(
            "RAG_SOURCE_NOT_FOUND", "C:\\private\\index must not leak"
        )
        status, payload = self._get(route)
        self.assertEqual(status, 400)
        self.assertEqual(payload["error"]["code"], "RAG_SOURCE_NOT_FOUND")
        self.assertNotIn("private", json.dumps(payload))

        self.workspace.raise_source = None
        valid = self.workspace.preview_rag_source(attachment_id, 0)
        self.workspace.source_payload = {**valid, "host_path": "/home/private/index"}
        status, payload = self._get(route)
        self.assertEqual(status, 503)
        self.assertEqual(payload["error"]["code"], "WORKSPACE_RESPONSE_INVALID")
        self.assertNotIn("/home/private", json.dumps(payload))

        malformed_payloads = (
            {**valid, "schema_version": True},
            {**valid, "schema_version": 1},
            {**valid, "representation": "raw_pdf_bytes"},
            {**valid, "chunk_index": False},
            {**valid, "char_start": False},
            {**valid, "char_end": True},
            {**valid, "name": "bad\x7fname.md"},
            {**valid, "name": "bad\x80name.md"},
            {**valid, "media_type": "text/html"},
            {
                **valid,
                "char_start": MAX_INDEXED_TEXT_CHARS,
                "char_end": MAX_INDEXED_TEXT_CHARS + len(valid["text"]),
            },
            {
                **valid,
                "char_start": MAX_INDEXED_TEXT_CHARS - len(valid["text"]) + 1,
                "char_end": MAX_INDEXED_TEXT_CHARS + 1,
            },
        )
        for malformed in malformed_payloads:
            with self.subTest(malformed=malformed):
                self.workspace.source_payload = malformed
                status, payload = self._get(route)
                self.assertEqual(status, 503)
                self.assertEqual(payload["error"]["code"], "WORKSPACE_RESPONSE_INVALID")

    def test_workspace_post_routes_and_bounded_errors(self) -> None:
        status, payload = self._post(
            "/api/workspace/attachments/add",
            {
                "name": "paper.txt",
                "media_type": "text/plain",
                "content_base64": "QQ==",
            },
        )
        self.assertEqual(status, 201)
        self.assertEqual(payload["attachment_id"], "abc")
        self.assertEqual(
            self._post("/api/workspace/rag/index", {"attachment_ids": ["abc"]})[0],
            200,
        )
        status, payload = self._post(
            "/api/workspace/rag/reindex", {"attachment_ids": ["abc"]}
        )
        self.assertEqual(status, 200)
        self.assertEqual(payload["reindexed_attachment_ids"], ["abc"])
        status, payload = self._post(
            "/api/workspace/attachments/delete", {"attachment_id": "abc"}
        )
        self.assertEqual(status, 200)
        self.assertTrue(payload["deleted"])
        self.assertEqual(
            self._post("/api/workspace/rag/query", {"query": "평형", "limit": 2})[1][
                "count"
            ],
            2,
        )
        self.assertEqual(
            self._post("/api/workspace/models/select", {"model_id": "verified-model"})[
                0
            ],
            200,
        )
        status, payload = self._post(
            "/api/workspace/lens/search",
            {"kind": "patent", "query": "equilibrium", "limit": 5},
        )
        self.assertEqual(status, 200)
        self.assertEqual(payload["provider"], "Lens.org official API")
        status, payload = self._post(
            "/api/workspace/lens/search-and-index",
            {"kind": "scholarly", "query": "fixed point"},
        )
        self.assertEqual(status, 200)
        self.assertIn("indexed", payload)
        self.assertEqual(
            self.workspace.lens_queries,
            [
                {
                    "kind": "patent",
                    "query": "equilibrium",
                    "limit": 5,
                    "index_in_akasicdb": False,
                },
                {
                    "kind": "scholarly",
                    "query": "fixed point",
                    "limit": 5,
                    "index_in_akasicdb": True,
                },
            ],
        )
        status, payload = self._post(
            "/api/workspace/web/search",
            {
                "query": "bounded general search",
                "limit": 3,
                "online_opt_in": True,
                "request_id": "a" * 32,
            },
        )
        self.assertEqual(status, 200)
        self.assertEqual(payload["provider"], "Brave Search official API")
        self.assertEqual(
            self.workspace.web_queries,
            [
                {
                    "query": "bounded general search",
                    "limit": 3,
                    "session_online_opt_in": True,
                    "request_id": "a" * 32,
                }
            ],
        )
        status, payload = self._post(
            "/api/workspace/web/cancel", {"request_id": "a" * 32}
        )
        self.assertEqual(status, 200)
        self.assertTrue(payload["cancelled"])
        self.assertEqual(self.workspace.web_cancellations, ["a" * 32])
        status, payload = self._post(
            "/api/workspace/web/search",
            {"query": "must remain offline", "online_opt_in": False},
        )
        self.assertEqual(status, 400)
        self.assertEqual(payload["error"]["code"], "WEB_SEARCH_SESSION_OPT_IN_REQUIRED")
        self.assertNotIn("private", json.dumps(payload))
        status, payload = self._post(
            "/api/workspace/web/search",
            {"query": "missing explicit consent"},
        )
        self.assertEqual(status, 400)
        self.assertEqual(payload["error"]["code"], "INVALID_BODY")
        status, payload = self._post(
            "/api/workspace/models/select", {"model_id": "unknown"}
        )
        self.assertEqual(status, 400)
        encoded = json.dumps(payload)
        self.assertEqual(payload["error"]["code"], "MODEL_NOT_VERIFIED")
        self.assertNotIn("private", encoded)
        self.assertNotIn("model must not leak", encoded)

    def test_attachment_route_has_its_own_limit_without_raising_global(self) -> None:
        self.assertEqual(MAX_REQUEST_BODY_BYTES, 8 * 1024)
        large = "A" * (MAX_REQUEST_BODY_BYTES + 512)
        status, _payload = self._post(
            "/api/workspace/attachments/add",
            {
                "name": "large.txt",
                "media_type": "text/plain",
                "content_base64": large,
            },
        )
        self.assertEqual(status, 201)
        status, payload = self._post("/api/workspace/rag/query", {"query": large})
        self.assertEqual(status, 413)
        self.assertEqual(payload["error"]["code"], "BODY_TOO_LARGE")

    def test_chat_body_and_message_boundaries_are_independent(self) -> None:
        self.assertGreater(MAX_AGENT_CHAT_REQUEST_BODY_BYTES, MAX_REQUEST_BODY_BYTES)
        accepted = "가" * 4_096
        status, _payload = self._post(
            "/api/agent/chat",
            {"message": accepted, "mode": "chat", "rag": False},
        )
        self.assertEqual(status, 202)
        self.assertEqual(self.agent.messages[-1], accepted)

        status, payload = self._post(
            "/api/agent/chat",
            {"message": accepted + "가", "mode": "chat", "rag": False},
        )
        self.assertEqual(status, 400)
        self.assertEqual(payload["error"]["code"], "INVALID_BODY")

    def test_image_chat_is_explicit_single_turn_bounded_and_path_free(self) -> None:
        attachment_id = "a" * 24
        with (
            patch.object(
                self.server,
                "image_to_model_integration_status",
                return_value=_attested_image_status(),
            ),
            patch.object(
                self.server, "image_to_model_integration_ready", return_value=True
            ),
            patch.object(
                self.server,
                "_attested_image_runtime_identity",
                return_value=_RuntimeIdentity(),
            ),
        ):
            status, capability = self._get("/api/workspace/capabilities")
            self.assertEqual(status, 200)
            self.assertTrue(capability["attachments"]["image_to_model_integration"])
            self.assertTrue(
                capability["attachments"]["image_capability"]["runtime_ready"]
            )
            self.assertEqual(
                capability["attachments"]["image_selection"],
                "explicit_single_next_turn",
            )
            self.assertEqual(
                capability["models"]["items"][0]["runtime_input_modalities"],
                ["text", "image"],
            )
            self.assertEqual(
                capability["models"]["items"][0]["unwired_checkpoint_modalities"],
                [],
            )
            status, payload = self._post(
                "/api/agent/chat",
                {
                    "message": "이 이미지를 설명해 주세요.",
                    "mode": "chat",
                    "rag": False,
                    "image_attachment_id": attachment_id,
                },
            )
        self.assertEqual(status, 202)
        self.assertTrue(payload["image_requested"])
        self.assertTrue(payload["image_input_admitted"])
        self.assertEqual(payload["image_media_type"], "image/png")
        self.assertEqual(self.agent.image_content, b"\x89PNG\r\n\x1a\n")
        self.assertEqual(
            self.agent.expected_image_runtime_identity,
            _RuntimeIdentity(),
        )
        encoded = json.dumps(payload)
        self.assertNotIn("89504e47", encoded.casefold())
        self.assertNotIn("C:\\", encoded)

    def test_configured_image_chat_admits_one_first_use_attestation_probe(self) -> None:
        attachment_id = "a" * 24
        with (
            patch.object(
                self.server,
                "image_to_model_integration_status",
                return_value=_configured_unverified_image_status(),
            ),
            patch.object(
                self.server,
                "image_to_model_integration_ready",
                return_value=False,
            ),
            patch.object(
                self.server,
                "start_image_agent_turn",
                return_value=("turn-image-probe", True),
            ) as start_probe,
        ):
            capability_status, capability = self._get("/api/workspace/capabilities")
            self.assertEqual(capability_status, 200)
            self.assertEqual(
                capability["attachments"]["image_selection"],
                "explicit_single_next_turn",
            )
            status, payload = self._post(
                "/api/agent/chat",
                {
                    "message": "이 이미지로 첫 사용 경로를 검증해 주세요.",
                    "mode": "chat",
                    "rag": False,
                    "image_attachment_id": attachment_id,
                },
            )

        self.assertEqual(status, 202)
        self.assertTrue(payload["image_attestation_probe"])
        self.assertTrue(payload["image_input_admitted"])
        start_probe.assert_called_once_with(
            "이 이미지로 첫 사용 경로를 검증해 주세요.",
            b"\x89PNG\r\n\x1a\n",
        )

    def test_image_attestation_is_current_worker_bound_and_fail_closed(self) -> None:
        identity = _RuntimeIdentity()
        service = _RuntimeService(identity)
        probe = _ImageProbe("turn-image", "b" * 64, "2026-07-19T00:00:00+00:00")
        completion = {
            "state": "complete",
            "finish_reason": "stop",
            "generation_mode": "cogni_core_image",
            "truncated": False,
            "generated_tokens": 3,
            "image_attestation": _terminal_image_evidence(identity, probe.image_sha256),
        }
        with patch.object(
            self.server,
            "_configured_image_model_service",
            return_value=service,
        ):
            self.server._image_attestation_probe = probe
            self.server._finish_first_image_attestation(
                probe,
                {
                    "turn_id": probe.turn_id,
                    "status": "succeeded",
                    "completion": completion,
                },
            )
            status = self.server.image_to_model_integration_status()
            self.assertEqual(status["state"], "ready")
            self.assertTrue(status["runtime_ready"])
            self.assertTrue(status["model_inference_attested"])
            self.assertRegex(str(status["attestation_id"]), r"^img1-[0-9a-f]{64}$")

            service.identity = _RuntimeIdentity(worker_incarnation=2)
            stale = self.server.image_to_model_integration_status()
            self.assertEqual(stale["state"], "configured_unverified")
            self.assertFalse(stale["runtime_ready"])
            self.assertTrue(stale["first_use_attestation_allowed"])

            failed_probe = _ImageProbe(
                "turn-fallback",
                "c" * 64,
                "2026-07-19T00:01:00+00:00",
            )
            self.server._image_attestation_probe = failed_probe
            self.server._finish_first_image_attestation(
                failed_probe,
                {
                    "turn_id": failed_probe.turn_id,
                    "status": "succeeded",
                    "completion": {
                        **completion,
                        "generation_mode": "quality_fallback",
                    },
                },
            )
            failed = self.server.image_to_model_integration_status()
            self.assertEqual(failed["state"], "configured_unverified")
            self.assertFalse(failed["runtime_ready"])
            self.assertIsNone(failed["attestation_id"])

    def test_image_attestation_rejects_terminal_and_worker_mismatches(self) -> None:
        identity = _RuntimeIdentity()
        service = _RuntimeService(identity)
        probe = _ImageProbe("turn-image", "b" * 64, "2026-07-19T00:00:00+00:00")
        completion = {
            "state": "complete",
            "finish_reason": "stop",
            "generation_mode": "cogni_core_image",
            "truncated": False,
            "generated_tokens": 3,
            "image_attestation": _terminal_image_evidence(identity, probe.image_sha256),
        }
        cases = (
            (
                "length",
                {**completion, "finish_reason": "length"},
                "succeeded",
                probe.turn_id,
            ),
            (
                "truncated",
                {**completion, "truncated": True},
                "succeeded",
                probe.turn_id,
            ),
            (
                "zero_tokens",
                {**completion, "generated_tokens": 0},
                "succeeded",
                probe.turn_id,
            ),
            ("cancelled", completion, "cancelled", probe.turn_id),
            ("failed", completion, "failed", probe.turn_id),
            ("other_turn", completion, "succeeded", "turn-other"),
            (
                "digest_mismatch",
                {
                    **completion,
                    "image_attestation": _terminal_image_evidence(identity, "c" * 64),
                },
                "succeeded",
                probe.turn_id,
            ),
        )
        with patch.object(
            self.server,
            "_configured_image_model_service",
            return_value=service,
        ):
            for name, candidate, status, turn_id in cases:
                with self.subTest(case=name):
                    self.server._image_attestation = None
                    self.server._image_attestation_probe = probe
                    self.server._finish_first_image_attestation(
                        probe,
                        {
                            "turn_id": turn_id,
                            "status": status,
                            "completion": candidate,
                        },
                    )
                    self.assertIsNone(self.server._image_attestation)
                    self.assertIsNone(self.server._image_attestation_probe)

            for successor in (
                _RuntimeIdentity(worker_incarnation=2),
                _RuntimeIdentity(lease_epoch=8),
            ):
                with self.subTest(successor=successor):
                    self.server._image_attestation = None
                    self.server._image_attestation_probe = probe
                    service.identity = successor
                    self.server._finish_first_image_attestation(
                        probe,
                        {
                            "turn_id": probe.turn_id,
                            "status": "succeeded",
                            "completion": completion,
                        },
                    )
                    self.assertIsNone(self.server._image_attestation)
                    self.assertIsNone(self.server._image_attestation_probe)

    def test_image_probe_reset_fence_and_synchronous_publication_gate(self) -> None:
        self.assertEqual(
            self.server._image_attestation_watch_seconds(
                SimpleNamespace(
                    max_decode_seconds=120.0,
                    model_service=SimpleNamespace(startup_timeout=180.0),
                )
            ),
            330.0,
        )
        probe = _ImageProbe("turn-image", "b" * 64, "2026-07-19T00:00:00+00:00")
        self.server._image_attestation_probe = probe
        status, payload = self._post("/api/agent/reset", {})
        self.assertEqual(status, 409)
        self.assertEqual(payload["error"]["code"], "COMPUTE_BUSY")
        self.assertEqual(self.agent.reset_calls, 0)
        self.server._image_attestation_probe = None

        with patch.object(
            self.server,
            "image_to_model_integration_status",
            return_value=_configured_unverified_image_status(),
        ):
            turn_id, is_probe = self.server.start_image_agent_turn("describe", b"image")

        self.assertEqual(turn_id, "turn-rag")
        self.assertTrue(is_probe)
        self.assertIsNotNone(self.agent.first_use_gate)
        self.assertTrue(self.agent.first_use_gate.wait_authorized(0.0))
        self.assertTrue(callable(self.agent.image_publication_authorizer))
        self.assertIsNotNone(self.server._image_attestation_probe)
        self.assertEqual(self.server._image_attestation_probe.turn_id, turn_id)
        self.assertIsNotNone(self.server._image_attestation_thread)
        self.assertTrue(self.server._image_attestation_thread.is_alive())
        cleanup_thread = self.server._image_attestation_thread
        self.assertEqual(self.agent.cancel_calls, 0)
        self.agent.last_finished_turn = {"turn_id": turn_id, "status": "cancelled"}
        self.agent.snapshot_event.set()
        _wait_for_probe_clear(self.server)
        cleanup_thread.join(timeout=2)
        self.assertFalse(cleanup_thread.is_alive())

    def test_first_use_publication_authorizer_is_current_worker_bound(self) -> None:
        image = b"immutable-image"
        identity = _RuntimeIdentity()
        with patch.object(
            self.server,
            "image_to_model_integration_status",
            return_value=_configured_unverified_image_status(),
        ):
            turn_id, is_probe = self.server.start_image_agent_turn("describe", image)
        self.assertTrue(is_probe)
        cleanup_thread = self.server._image_attestation_thread
        self.assertIsNotNone(cleanup_thread)
        evidence = ImagePublicationEvidence(
            turn_id=turn_id,
            finish_reason="stop",
            generation_mode="cogni_core_image",
            truncated=False,
            generated_tokens=3,
            terminal_evidence=_terminal_image_evidence(
                identity,
                sha256(image).hexdigest(),
            ),
        )

        with patch.object(
            self.server,
            "_configured_image_model_service",
            return_value=_RuntimeService(identity),
        ):
            self.agent.image_publication_authorizer(evidence)
        self.agent.snapshot_event.set()
        cleanup_thread.join(timeout=2)

        self.assertIsNone(self.server._image_attestation_probe)
        self.assertFalse(cleanup_thread.is_alive())
        self.assertIsNotNone(self.server._image_attestation)
        self.assertEqual(self.server._image_attestation.turn_id, turn_id)
        self.assertEqual(
            self.server._image_attestation.image_sha256,
            sha256(image).hexdigest(),
        )

    def test_first_use_publication_authorizer_rejects_worker_swap(self) -> None:
        image = b"immutable-image"
        admitted = _RuntimeIdentity()
        successor = _RuntimeIdentity(worker_incarnation=2)
        with patch.object(
            self.server,
            "image_to_model_integration_status",
            return_value=_configured_unverified_image_status(),
        ):
            turn_id, is_probe = self.server.start_image_agent_turn("describe", image)
        self.assertTrue(is_probe)
        cleanup_thread = self.server._image_attestation_thread
        self.assertIsNotNone(cleanup_thread)
        evidence = ImagePublicationEvidence(
            turn_id=turn_id,
            finish_reason="stop",
            generation_mode="cogni_core_image",
            truncated=False,
            generated_tokens=3,
            terminal_evidence=_terminal_image_evidence(
                admitted,
                sha256(image).hexdigest(),
            ),
        )

        with patch.object(
            self.server,
            "_configured_image_model_service",
            return_value=_RuntimeService(successor),
        ):
            with self.assertRaisesRegex(
                ImageModelUnavailableError,
                "not attested",
            ):
                self.agent.image_publication_authorizer(evidence)
        self.agent.snapshot_event.set()
        cleanup_thread.join(timeout=2)

        self.assertIsNone(self.server._image_attestation_probe)
        self.assertFalse(cleanup_thread.is_alive())
        self.assertIsNone(self.server._image_attestation)

    def test_probe_cleanup_malformed_snapshot_blocks_late_authority(self) -> None:
        image = b"immutable-image"
        identity = _RuntimeIdentity()
        malformed = {"seq": "invalid", "last_finished_turn": None}
        with (
            patch.object(
                self.server,
                "image_to_model_integration_status",
                return_value=_configured_unverified_image_status(),
            ),
            patch.object(self.agent, "snapshot", return_value=malformed),
        ):
            turn_id, is_probe = self.server.start_image_agent_turn("describe", image)
            _wait_for_probe_clear(self.server)
        self.assertTrue(is_probe)
        evidence = ImagePublicationEvidence(
            turn_id=turn_id,
            finish_reason="stop",
            generation_mode="cogni_core_image",
            truncated=False,
            generated_tokens=3,
            terminal_evidence=_terminal_image_evidence(
                identity,
                sha256(image).hexdigest(),
            ),
        )

        with patch.object(
            self.server,
            "_configured_image_model_service",
            return_value=_RuntimeService(identity),
        ):
            with self.assertRaisesRegex(
                ImageModelUnavailableError,
                "not attested",
            ):
                self.agent.image_publication_authorizer(evidence)

        self.assertIsNone(self.server._image_attestation_probe)
        self.assertIsNone(self.server._image_attestation_thread)
        self.assertIsNone(self.server._image_attestation)

    def test_probe_cleanup_deadline_blocks_late_authority(self) -> None:
        image = b"immutable-image"
        identity = _RuntimeIdentity()
        with (
            patch.object(
                self.server,
                "image_to_model_integration_status",
                return_value=_configured_unverified_image_status(),
            ),
            patch.object(
                self.server,
                "_image_attestation_watch_seconds",
                return_value=0.0,
            ),
        ):
            turn_id, is_probe = self.server.start_image_agent_turn("describe", image)
            _wait_for_probe_clear(self.server)
        self.assertTrue(is_probe)
        evidence = ImagePublicationEvidence(
            turn_id=turn_id,
            finish_reason="stop",
            generation_mode="cogni_core_image",
            truncated=False,
            generated_tokens=3,
            terminal_evidence=_terminal_image_evidence(
                identity,
                sha256(image).hexdigest(),
            ),
        )

        with patch.object(
            self.server,
            "_configured_image_model_service",
            return_value=_RuntimeService(identity),
        ):
            with self.assertRaisesRegex(
                ImageModelUnavailableError,
                "not attested",
            ):
                self.agent.image_publication_authorizer(evidence)

        self.assertIsNone(self.server._image_attestation_probe)
        self.assertIsNone(self.server._image_attestation_thread)
        self.assertIsNone(self.server._image_attestation)

    def test_production_agent_first_use_probe_does_not_self_block(self) -> None:
        image = b"immutable-image"
        service = _ProductionPathImageService()
        manager = self._install_production_agent(service)

        with (
            patch.object(
                self.server,
                "image_to_model_integration_status",
                return_value=_configured_unverified_image_status(),
            ),
            patch.object(
                self.server,
                "_configured_image_model_service",
                return_value=_RuntimeService(service.identity),
            ),
        ):
            turn_id, is_probe = self.server.start_image_agent_turn(
                "이미지를 설명해 주세요.", image
            )
            state = _wait_for_agent(manager)

        self.assertTrue(is_probe)
        self.assertEqual(state["status"], "succeeded")
        self.assertEqual(state["last_finished_turn"]["turn_id"], turn_id)
        self.assertIsNone(self.server._image_probe_admission_owner)
        self.assertIsNone(self.server._image_attestation_probe)
        self.assertIsNotNone(self.server._image_attestation)

    def test_production_agent_gate_timeout_cleans_exact_probe(self) -> None:
        service = _ProductionPathImageService()
        manager = self._install_production_agent(service)
        original_start = self.server._start_first_use_image_turn_locked

        def start_after_gate_timeout(*args, **kwargs):
            turn_id = original_start(*args, **kwargs)
            state = _wait_for_agent(manager)
            self.assertEqual(state["status"], "cancelled")
            return turn_id

        with (
            patch.object(
                self.server,
                "image_to_model_integration_status",
                return_value=_configured_unverified_image_status(),
            ),
            patch.object(
                self.server,
                "_start_first_use_image_turn_locked",
                side_effect=start_after_gate_timeout,
            ),
            patch("cogni_agent.manager.FIRST_USE_TURN_GATE_SECONDS", 0.0),
        ):
            with self.assertRaisesRegex(
                ImageModelUnavailableError,
                "execution gate could not be released",
            ):
                self.server.start_image_agent_turn("describe", b"image")

        _wait_for_probe_clear(self.server)
        self.assertIsNone(self.server._image_probe_admission_owner)
        self.assertIsNone(self.server._image_attestation_thread)
        self.assertIsNone(self.server._image_attestation)

    def test_production_agent_cancel_cleans_exact_probe_without_authority(self) -> None:
        entered = Event()
        release = Event()
        service = _ProductionPathImageService(entered=entered, release=release)
        manager = self._install_production_agent(service)

        with patch.object(
            self.server,
            "image_to_model_integration_status",
            return_value=_configured_unverified_image_status(),
        ):
            self.server.start_image_agent_turn("describe", b"image")
        self.assertTrue(entered.wait(2))
        manager.cancel()
        release.set()
        state = _wait_for_agent(manager)
        _wait_for_probe_clear(self.server)

        self.assertEqual(state["status"], "cancelled")
        self.assertIsNone(self.server._image_attestation)
        self.assertIsNone(self.server._image_attestation_thread)

    def test_production_agent_backend_failure_cleans_exact_probe(self) -> None:
        service = _ProductionPathImageService(error=RuntimeError("backend failed"))
        manager = self._install_production_agent(service)

        with patch.object(
            self.server,
            "image_to_model_integration_status",
            return_value=_configured_unverified_image_status(),
        ):
            self.server.start_image_agent_turn("describe", b"image")
        state = _wait_for_agent(manager)
        _wait_for_probe_clear(self.server)

        self.assertEqual(state["status"], "failed")
        self.assertIsNone(self.server._image_attestation)
        self.assertIsNone(self.server._image_attestation_thread)

    def test_first_use_probe_cleanup_watcher_start_failure_is_fail_closed(
        self,
    ) -> None:
        with (
            patch.object(
                self.server,
                "image_to_model_integration_status",
                return_value=_configured_unverified_image_status(),
            ),
            patch("cogni_demo.server.Thread.start", side_effect=RuntimeError("boom")),
        ):
            with self.assertRaisesRegex(
                ImageModelUnavailableError,
                "cleanup watcher could not start",
            ):
                self.server.start_image_agent_turn("describe", b"image")

        self.assertEqual(self.agent.cancel_calls, 1)
        self.assertFalse(self.agent.first_use_gate.wait_authorized(0.0))
        self.assertIsNone(self.server._image_probe_admission_owner)
        self.assertIsNone(self.server._image_attestation_probe)
        self.assertIsNone(self.server._image_attestation_thread)

    def test_fence_change_cancels_without_holding_image_lock(self) -> None:
        foreign_probe = _ImageProbe(
            "foreign-turn",
            "f" * 64,
            "2026-07-19T00:00:00+00:00",
        )
        observations: list[bool] = []
        lock_threads: list[Thread] = []

        def replace_fence(*_args, **_kwargs):
            with self.server._image_attestation_lock:
                self.server._image_attestation_probe = foreign_probe
            return "turn-rag"

        def observe_cancel_lock_order() -> None:
            acquired = Event()

            def acquire_image_lock() -> None:
                with self.server._image_attestation_lock:
                    acquired.set()

            thread = Thread(target=acquire_image_lock)
            lock_threads.append(thread)
            thread.start()
            observations.append(acquired.wait(0.5))

        with (
            patch.object(
                self.server,
                "image_to_model_integration_status",
                return_value=_configured_unverified_image_status(),
            ),
            patch.object(
                self.server,
                "_start_first_use_image_turn_locked",
                side_effect=replace_fence,
            ),
            patch.object(self.agent, "cancel", side_effect=observe_cancel_lock_order),
        ):
            with self.assertRaisesRegex(
                ImageModelUnavailableError,
                "fence changed",
            ):
                self.server.start_image_agent_turn("describe", b"image")

        for thread in lock_threads:
            thread.join(timeout=2)
        self.assertEqual(observations, [True])
        self.assertIsNone(self.server._image_probe_admission_owner)
        self.server._image_attestation_probe = None

    def test_public_image_start_requires_ready_exact_attestation(self) -> None:
        image = b"immutable-image"

        with patch.object(
            self.server,
            "_attested_image_runtime_identity",
            return_value=None,
        ):
            with self.assertRaisesRegex(
                ImageModelUnavailableError,
                "attestation is not ready",
            ):
                self.server.start_agent_turn(
                    "이미지를 설명해 주세요.",
                    "chat",
                    image_content=image,
                )
        self.assertIsNone(self.agent.image_content)

        admitted = _RuntimeIdentity()
        with patch.object(
            self.server,
            "_attested_image_runtime_identity",
            return_value=admitted,
        ):
            turn_id = self.server.start_agent_turn(
                "이미지를 설명해 주세요.",
                "chat",
                image_content=image,
            )

        self.assertEqual(turn_id, "turn-rag")
        self.assertIs(self.agent.expected_image_runtime_identity, admitted)
        self.assertIs(self.agent.image_content, image)

    def test_agent_thread_start_failure_has_controlled_http_error(self) -> None:
        with patch.object(
            self.server,
            "start_agent_turn",
            side_effect=AgentTurnStartError("thread unavailable"),
        ):
            status, payload = self._post(
                "/api/agent/chat",
                {"message": "안전하게 시작해 주세요.", "mode": "chat"},
            )

        self.assertEqual(status, 503)
        self.assertEqual(payload["error"]["code"], "AGENT_UNAVAILABLE")

    def test_image_chat_rejects_invalid_id_media_rag_task_and_unverified_runtime(
        self,
    ) -> None:
        status, capability = self._get("/api/workspace/capabilities")
        self.assertEqual(status, 200)
        self.assertFalse(capability["attachments"]["image_to_model_integration"])
        self.assertFalse(
            capability["attachments"]["image_capability"]["model_inference_attested"]
        )
        self.assertFalse(capability["attachments"]["image_capability"]["runtime_ready"])

        text_only_capability = {
            "schema_version": 1,
            "attachments": {"state": "enabled"},
            "rag": {"state": "local_index_ready"},
            "models": {
                "items": [
                    {
                        "selected": True,
                        "checkpoint_modalities": ["text"],
                        "runtime_input_modalities": ["text"],
                    }
                ]
            },
        }
        with (
            patch.object(
                self.server,
                "image_to_model_integration_status",
                return_value=_attested_image_status(),
            ),
            patch.object(
                self.workspace,
                "capability_payload",
                return_value=text_only_capability,
            ),
            patch.object(self.server, "start_image_agent_turn") as start_image,
        ):
            status, capability = self._get("/api/workspace/capabilities")
            post_status, post_payload = self._post(
                "/api/agent/chat",
                {"message": "x", "image_attachment_id": "a" * 24},
            )
        self.assertEqual(status, 200)
        self.assertFalse(capability["attachments"]["image_to_model_integration"])
        self.assertEqual(
            capability["attachments"]["image_capability"]["state"],
            "selected_checkpoint_not_supported",
        )
        self.assertEqual(post_status, 503)
        self.assertEqual(post_payload["error"]["code"], "IMAGE_MODEL_UNAVAILABLE")
        start_image.assert_not_called()

        misleading_discovered_capability = {
            "schema_version": 1,
            "attachments": {"state": "enabled"},
            "rag": {"state": "local_index_ready"},
            "models": {
                "items": [
                    {
                        "selected": True,
                        "checkpoint_modalities": ["text"],
                        "runtime_input_modalities": ["text"],
                    },
                    {
                        "selected": False,
                        "checkpoint_modalities": ["text", "image"],
                        "runtime_input_modalities": ["text"],
                    },
                ]
            },
        }
        with (
            patch.object(
                self.server,
                "image_to_model_integration_status",
                return_value=_attested_image_status(),
            ),
            patch.object(
                self.workspace,
                "capability_payload",
                return_value=misleading_discovered_capability,
            ),
        ):
            status, capability = self._get("/api/workspace/capabilities")
        self.assertEqual(status, 200)
        self.assertFalse(capability["attachments"]["image_to_model_integration"])
        self.assertEqual(
            capability["models"]["items"][1]["runtime_input_modalities"],
            ["text"],
        )

        invalid_bodies = (
            {"message": "x", "image_attachment_id": None},
            {"message": "x", "image_attachment_id": "A" * 24},
            {
                "message": "x",
                "mode": "chat",
                "rag": True,
                "image_attachment_id": "a" * 24,
            },
            {
                "message": "/status",
                "mode": "task",
                "image_attachment_id": "a" * 24,
            },
        )
        for body in invalid_bodies:
            with self.subTest(body=body):
                status, payload = self._post("/api/agent/chat", body)
                self.assertEqual(status, 400)
                self.assertEqual(payload["error"]["code"], "INVALID_BODY")

        status, payload = self._post(
            "/api/agent/chat",
            {"message": "x", "image_attachment_id": "a" * 24},
        )
        self.assertEqual(status, 503)
        self.assertEqual(payload["error"]["code"], "IMAGE_MODEL_UNAVAILABLE")

        with patch.object(
            self.server, "image_to_model_integration_ready", return_value=True
        ):
            status, payload = self._post(
                "/api/agent/chat",
                {"message": "x", "image_attachment_id": "b" * 24},
            )
        self.assertEqual(status, 400)
        self.assertEqual(payload["error"]["code"], "ATTACHMENT_NOT_FOUND")

        with (
            patch.object(
                self.server, "image_to_model_integration_ready", return_value=True
            ),
            patch.object(
                self.workspace,
                "image_attachment_content",
                return_value=(b"plain", "text/plain"),
            ),
        ):
            status, payload = self._post(
                "/api/agent/chat",
                {"message": "x", "image_attachment_id": "a" * 24},
            )
        self.assertEqual(status, 400)
        self.assertEqual(payload["error"]["code"], "WORKSPACE_RESPONSE_INVALID")

    def test_rag_chat_truncates_only_the_search_query(self) -> None:
        import cogni_agent.manager as manager_module

        for message in ("평" * 1_024, "평" * 1_025, "평형" * 2_048):
            with patch.object(
                manager_module, "RetrievalEvidence", _Evidence, create=True
            ):
                status, _payload = self._post(
                    "/api/agent/chat",
                    {"message": message, "mode": "chat", "rag": True},
                )
            self.assertEqual(status, 202)
            self.assertEqual(self.workspace.queries[-1], message[:1_024])
            self.assertEqual(self.agent.messages[-1], message)

    def test_rag_chat_maps_only_positive_local_evidence(self) -> None:
        import cogni_agent.manager as manager_module

        with patch.object(manager_module, "RetrievalEvidence", _Evidence, create=True):
            status, payload = self._post(
                "/api/agent/chat",
                {"message": "평형 검색", "mode": "chat", "rag": True},
            )
        self.assertEqual(status, 202)
        self.assertTrue(payload["rag_requested"])
        self.assertEqual(payload["rag_evidence_count"], 1)
        self.assertEqual(len(self.agent.evidence), 1)
        self.assertTrue(self.agent.retrieval_requested)
        self.assertEqual(self.agent.evidence[0].source_id, f"{'a' * 24}.0")
        provenance = self.agent.evidence[0].provenance
        self.assertEqual(provenance.retrieval_mode, "lexical_only")
        self.assertFalse(provenance.semantic_embedding)
        self.assertEqual(provenance.source_sha256, "a" * 24 + "b" * 40)
        self.assertEqual(
            provenance.answer_integration_schema,
            "cogni.agent.retrieval-evidence.v1",
        )
        self.assertEqual(provenance.indexed_excerpt_chars, len("검증된 로컬 검색 근거"))
        self.assertEqual(
            provenance.indexed_excerpt_sha256,
            sha256("검증된 로컬 검색 근거".encode()).hexdigest(),
        )

        self.workspace.rag_results = []
        with patch.object(manager_module, "RetrievalEvidence", _Evidence, create=True):
            status, payload = self._post(
                "/api/agent/chat",
                {"message": "없는 근거", "mode": "chat", "rag": True},
            )
        self.assertEqual(status, 202)
        self.assertEqual(payload["rag_evidence_count"], 0)
        self.assertTrue(self.agent.retrieval_requested)

    def test_rag_chat_bounds_five_full_chunks_to_six_thousand_characters(self) -> None:
        self.workspace.rag_results = [
            {
                "attachment_id": f"{index + 1:024x}",
                "chunk_index": index,
                "name": f"paper-{index}.md",
                "text": chr(65 + index) * 1_600,
                "score": 0.9,
            }
            for index in range(5)
        ]
        status, payload = self._post(
            "/api/agent/chat",
            {"message": "평형 검색", "mode": "chat", "rag": True},
        )

        self.assertEqual(status, 202)
        self.assertEqual(payload["rag_evidence_count"], 5)
        self.assertEqual(len(self.agent.evidence), 5)
        self.assertEqual(
            sum(len(item.text) for item in self.agent.evidence),
            6_000,
        )

    def test_malformed_rag_adapter_payload_is_bounded(self) -> None:
        import cogni_agent.manager as manager_module

        self.workspace.rag_results = [{"attachment_id": "C:\\secret"}]
        with patch.object(manager_module, "RetrievalEvidence", _Evidence, create=True):
            status, payload = self._post(
                "/api/agent/chat",
                {"message": "malformed", "mode": "chat", "rag": True},
            )
        self.assertEqual(status, 400)
        self.assertEqual(payload["error"]["code"], "WORKSPACE_RESPONSE_INVALID")
        self.assertNotIn("secret", json.dumps(payload))

    def test_rag_chat_rejects_non_answer_bearing_query_envelope(self) -> None:
        original = self.workspace.query_rag

        def query_without_answer_authority(query, *, limit=5):
            payload = original(query, limit=limit)
            payload["answer_integration"] = False
            return payload

        self.workspace.query_rag = query_without_answer_authority
        status, payload = self._post(
            "/api/agent/chat",
            {"message": "권한 없는 근거", "mode": "chat", "rag": True},
        )
        self.assertEqual(status, 400)
        self.assertEqual(payload["error"]["code"], "WORKSPACE_RESPONSE_INVALID")

    def test_rag_chat_rejects_mismatched_or_ambiguous_query_envelopes(self) -> None:
        original = self.workspace.query_rag

        def mismatched_query(query, *, limit=5):
            payload = original(query, limit=limit)
            payload["query"] = query + " altered"
            return payload

        def mismatched_count(query, *, limit=5):
            payload = original(query, limit=limit)
            payload["count"] += 1
            return payload

        def extra_envelope_key(query, *, limit=5):
            payload = original(query, limit=limit)
            payload["semantic_score"] = 1.0
            return payload

        def extra_result_key(query, *, limit=5):
            payload = original(query, limit=limit)
            payload["results"][0]["semantic_score"] = 1.0
            return payload

        def boolean_schema_version(query, *, limit=5):
            payload = original(query, limit=limit)
            payload["schema_version"] = True
            return payload

        def floating_schema_version(query, *, limit=5):
            payload = original(query, limit=limit)
            payload["schema_version"] = 2.0
            return payload

        for malformed in (
            mismatched_query,
            mismatched_count,
            extra_envelope_key,
            extra_result_key,
            boolean_schema_version,
            floating_schema_version,
        ):
            with (
                self.subTest(malformed=malformed.__name__),
                patch.object(self.workspace, "query_rag", side_effect=malformed),
            ):
                status, payload = self._post(
                    "/api/agent/chat",
                    {"message": "엄격한 근거", "mode": "chat", "rag": True},
                )
            self.assertEqual(status, 400)
            self.assertEqual(payload["error"]["code"], "WORKSPACE_RESPONSE_INVALID")

    def test_product_rag_boundary_rejects_unprovenanced_internal_evidence(self) -> None:
        unprovenanced = _Evidence(
            source_id="a" * 24 + ".0",
            title="untrusted",
            text="generic manager evidence",
            score=1.0,
        )
        with self.assertRaisesRegex(ValueError, "admitted provenance"):
            self.server.start_agent_turn(
                "internal caller",
                "chat",
                evidence=(unprovenanced,),
                retrieval_requested=True,
            )
        with self.assertRaisesRegex(ValueError, "admitted provenance"):
            self.server.start_agent_turn(
                "default flag bypass",
                "chat",
                evidence=(unprovenanced,),
            )
        self.assertEqual(self.agent.evidence, ())

        turn_id = self.server.start_agent_turn(
            "no local hit",
            "chat",
            evidence=(),
            retrieval_requested=True,
        )
        self.assertEqual(turn_id, "turn-rag")
        self.assertTrue(self.agent.retrieval_requested)

    def test_product_rag_boundary_rejects_forged_exact_shape_provenance(self) -> None:
        from cogni_agent.manager import RetrievalEvidence, RetrievalProvenance
        from cogni_demo.workspace_capabilities import (
            AKASICDB_AUDITED_REVISION,
            AKASICDB_REPOSITORY,
            RAG_ANSWER_INTEGRATION_SCHEMA,
            RAG_EMBEDDING_PROFILE,
        )

        attachment_id = "a" * 24
        text = "검증된 로컬 검색 근거"
        common = {
            "retrieval_mode": "lexical_only",
            "semantic_embedding": False,
            "answer_integration_schema": RAG_ANSWER_INTEGRATION_SCHEMA,
            "source_sha256": attachment_id + "b" * 40,
            "indexed_excerpt_chars": len(text),
        }
        forged = (
            RetrievalProvenance(
                repository="https://github.com/attacker/AkasicDB.git",
                revision=AKASICDB_AUDITED_REVISION,
                embedding=RAG_EMBEDDING_PROFILE,
                indexed_excerpt_sha256=sha256(text.encode()).hexdigest(),
                **common,
            ),
            RetrievalProvenance(
                repository=AKASICDB_REPOSITORY,
                revision="f" * 40,
                embedding=RAG_EMBEDDING_PROFILE,
                indexed_excerpt_sha256=sha256(text.encode()).hexdigest(),
                **common,
            ),
            RetrievalProvenance(
                repository=AKASICDB_REPOSITORY,
                revision=AKASICDB_AUDITED_REVISION,
                embedding="forged_lexical_profile",
                indexed_excerpt_sha256=sha256(text.encode()).hexdigest(),
                **common,
            ),
            RetrievalProvenance(
                repository=AKASICDB_REPOSITORY,
                revision=AKASICDB_AUDITED_REVISION,
                embedding=RAG_EMBEDDING_PROFILE,
                indexed_excerpt_sha256=sha256(text.encode()).hexdigest(),
                **{
                    **common,
                    "source_sha256": attachment_id + "c" * 40,
                },
            ),
        )
        for provenance in forged:
            evidence = RetrievalEvidence(
                source_id=f"{attachment_id}.0",
                title="paper.md",
                text=text,
                score=1.0,
                provenance=provenance,
            )
            with (
                self.subTest(
                    repository=provenance.repository,
                    revision=provenance.revision,
                    embedding=provenance.embedding,
                    digest=provenance.indexed_excerpt_sha256,
                ),
                self.assertRaisesRegex(ValueError, "admitted provenance"),
            ):
                self.server.start_agent_turn(
                    "forged internal caller",
                    "chat",
                    evidence=(evidence,),
                )

        forged_digest = RetrievalProvenance(
            repository=AKASICDB_REPOSITORY,
            revision=AKASICDB_AUDITED_REVISION,
            embedding=RAG_EMBEDDING_PROFILE,
            indexed_excerpt_sha256="c" * 64,
            **common,
        )
        truncated = RetrievalEvidence(
            source_id=f"{attachment_id}.0",
            title="paper.md",
            text=text[:-1],
            score=1.0,
            provenance=forged_digest,
        )
        with self.assertRaisesRegex(ValueError, "admitted provenance"):
            self.server.start_agent_turn(
                "forged digest",
                "chat",
                evidence=(truncated,),
                retrieval_requested=True,
            )
        self.assertEqual(self.agent.evidence, ())

    def test_product_rag_boundary_rechecks_current_source_snapshot(self) -> None:
        query = self.workspace.query_rag("검증된 로컬", limit=5)
        evidence = DemoRequestHandler._retrieval_evidence(
            query,
            expected_query="검증된 로컬",
        )
        self.assertEqual(len(evidence), 1)
        attachment_id = "a" * 24
        original = self.workspace.preview_rag_source(attachment_id, 0)
        changed_text = "재색인 후 변경된 근거"
        self.workspace.source_payload = {
            **original,
            "text": changed_text,
            "char_end": original["char_start"] + len(changed_text),
            "excerpt_sha256": sha256(changed_text.encode()).hexdigest(),
        }
        with self.assertRaisesRegex(ValueError, "admitted provenance"):
            self.server.start_agent_turn(
                "stale citation",
                "chat",
                evidence=evidence,
            )

        self.workspace.source_payload = None
        self.server.workspace_service = None
        with self.assertRaisesRegex(ValueError, "admitted provenance"):
            self.server.start_agent_turn(
                "missing authority service",
                "chat",
                evidence=evidence,
                retrieval_requested=True,
            )
        self.assertEqual(self.agent.evidence, ())

    def test_rag_evidence_validates_all_results_before_selecting_first_five(
        self,
    ) -> None:
        self.workspace.rag_results = [
            {
                "attachment_id": f"{index + 1:024x}",
                "chunk_index": index,
                "name": f"paper-{index}.md",
                "text": f"evidence {index}",
                "score": 0.9,
            }
            for index in range(6)
        ]
        payload = self.workspace.query_rag("all results", limit=12)
        evidence = DemoRequestHandler._retrieval_evidence(
            payload, expected_query="all results"
        )
        self.assertEqual(len(evidence), 5)

        payload["results"][5]["unexpected"] = True
        with self.assertRaises(WorkspaceCapabilityError) as captured:
            DemoRequestHandler._retrieval_evidence(
                payload, expected_query="all results"
            )
        self.assertEqual(captured.exception.code, "WORKSPACE_RESPONSE_INVALID")

    def test_workspace_routes_are_503_when_service_is_absent(self) -> None:
        self.server.workspace_service = None
        status, payload = self._get("/api/workspace/capabilities")
        self.assertEqual(status, 503)
        self.assertEqual(payload["error"]["code"], "WORKSPACE_UNAVAILABLE")
        status, payload = self._post("/api/workspace/rag/query", {"query": "local"})
        self.assertEqual(status, 503)
        self.assertEqual(payload["error"]["code"], "WORKSPACE_UNAVAILABLE")
        status, payload = self._get(
            "/api/workspace/rag/source?attachment_id=" + "a" * 24 + "&chunk_index=0"
        )
        self.assertEqual(status, 503)
        self.assertEqual(payload["error"]["code"], "WORKSPACE_UNAVAILABLE")


if __name__ == "__main__":
    unittest.main()
