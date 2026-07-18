"""CPU-only smoke for the pinned AkasicDB lexical RAG authority.

The validator deliberately makes no semantic-embedding claim.  It proves that
the audited local adapter can index, survive a service restart, publish exact
origin provenance, and remove the answer-bearing document durably.
"""

from __future__ import annotations

import argparse
from base64 import b64encode
from hashlib import sha256
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from time import monotonic, sleep
from types import SimpleNamespace

import torch

from cogni_agent.manager import ACTIVE_AGENT_STATUSES, AgentManager
from cogni_agent.tools import WorkspaceToolExecutor
from cogni_demo.server import DemoRequestHandler
from cogni_demo.workspace_capabilities import (
    AKASICDB_AUDITED_REVISION,
    AKASICDB_REPOSITORY,
    RAG_EMBEDDING_PROFILE,
    RAG_ANSWER_INTEGRATION_SCHEMA,
    RAG_QUERY_SCHEMA_VERSION,
    RAG_RETRIEVAL_MODE,
    VerifiedModelMetadata,
    WorkspaceCapabilityService,
)


_SMOKE_TEXT = "equilibrium provenance restart deletion evidence"
_SMOKE_QUERY = "equilibrium provenance"


class _SmokeTokenizer:
    eos_token_id = 3

    @staticmethod
    def decode(tokens, **_kwargs):
        return "".join(chr(value) for value in tokens)

    @staticmethod
    def apply_chat_template(messages, **_kwargs):
        return "|".join(f"{item['role']}:{item['content']}" for item in messages)


class _SmokeModelService:
    """Deterministic CPU generation backend for the answer-bridge smoke only."""

    def __init__(self) -> None:
        self.tokenizer = _SmokeTokenizer()
        self.active_request_id = None
        self.is_running = False

    def start(self):
        self.is_running = True
        return self

    def iter_generate_tokens(self, _prompt, **_kwargs):
        self.active_request_id = 1
        text = "감사된 로컬 근거입니다 [근거 1]."
        tokens = torch.tensor([ord(character) for character in text], dtype=torch.int64)
        yield SimpleNamespace(
            request_id=1,
            token_ids=tokens,
            generated_total=int(tokens.numel()),
            final=True,
            cancelled=False,
            finish_reason="stop",
            generation_mode="cogni_core",
        )
        self.active_request_id = None

    @staticmethod
    def cancel(_request_id=None):
        return True

    def stop(self, timeout=10.0):
        del timeout
        self.is_running = False


def _verify_answer_bridge(project: Path, query: dict[str, object]) -> dict[str, object]:
    evidence = DemoRequestHandler._retrieval_evidence(
        query, expected_query=_SMOKE_QUERY
    )
    manager = AgentManager(
        _SmokeModelService(),
        WorkspaceToolExecutor(project, timeout_seconds=5),
    )
    try:
        manager.start_turn(_SMOKE_QUERY, evidence=evidence, retrieval_requested=True)
        deadline = monotonic() + 5.0
        state = manager.snapshot()
        while state["status"] in ACTIVE_AGENT_STATUSES and monotonic() < deadline:
            sleep(0.01)
            state = manager.snapshot()
        if state["status"] != "succeeded":
            raise RuntimeError("RAG answer bridge did not complete")
        answer = state["conversation"][-1]
        sources = answer.get("sources")
        if (
            answer.get("generation_mode") != "cogni_core_rag"
            or "[근거 1]" not in answer.get("content", "")
            or not isinstance(sources, list)
            or len(sources) != 1
        ):
            raise RuntimeError("RAG citation/source bridge did not verify")
        provenance = sources[0].get("provenance")
        if (
            not isinstance(provenance, dict)
            or provenance.get("answer_integration_schema")
            != RAG_ANSWER_INTEGRATION_SCHEMA
            or provenance.get("selected_excerpt_sha256")
            != sha256(evidence[0].text.encode("utf-8")).hexdigest()
            or provenance.get("selected_excerpt_chars") != len(evidence[0].text)
            or provenance.get("prompt_excerpt_sha256")
            != sha256(evidence[0].text.encode("utf-8")).hexdigest()
            or provenance.get("prompt_excerpt_chars") != len(evidence[0].text)
            or provenance.get("prompt_excerpt_representation")
            != "xml_entity_escaped_v1"
        ):
            raise RuntimeError("RAG selected/prompt provenance did not verify")
        return provenance
    finally:
        manager.shutdown()


def _smoke_model() -> VerifiedModelMetadata:
    return VerifiedModelMetadata(
        model_id="akasicdb-rag-smoke",
        label="RAG boundary validator",
        architecture="metadata-only",
        manifest_sha256="a" * 64,
        config_sha256="b" * 64,
        checkpoint_modalities=("text",),
        runtime_input_modalities=("text",),
    )


def validate_akasicdb_rag(clone_path: str | Path) -> dict[str, object]:
    source_bytes = _SMOKE_TEXT.encode("utf-8")
    source_sha256 = sha256(source_bytes).hexdigest()
    with TemporaryDirectory(prefix="cogni-akasicdb-rag-") as temporary:
        project = Path(temporary)
        service = WorkspaceCapabilityService(
            project,
            _smoke_model(),
            akasicdb_path=clone_path,
            answer_integration_schema=RAG_ANSWER_INTEGRATION_SCHEMA,
        )
        admitted = service.add_attachment(
            name="rag-smoke.txt",
            media_type="text/plain",
            content_base64=b64encode(source_bytes).decode("ascii"),
        )
        attachment_id = admitted["attachment_id"]
        service.index_attachments([attachment_id])

        restarted = WorkspaceCapabilityService(
            project,
            _smoke_model(),
            akasicdb_path=clone_path,
            answer_integration_schema=RAG_ANSWER_INTEGRATION_SCHEMA,
        )
        query = restarted.query_rag(_SMOKE_QUERY)
        if (
            query.get("schema_version") != RAG_QUERY_SCHEMA_VERSION
            or query.get("repository") != AKASICDB_REPOSITORY
            or query.get("revision") != AKASICDB_AUDITED_REVISION
            or query.get("retrieval_mode") != RAG_RETRIEVAL_MODE
            or query.get("embedding") != RAG_EMBEDDING_PROFILE
            or query.get("semantic_embedding") is not False
            or query.get("answer_integration") is not True
            or query.get("count") != 1
        ):
            raise RuntimeError("AkasicDB restart query contract did not verify")
        result = query["results"][0]
        if (
            result.get("attachment_id") != attachment_id
            or result.get("source_sha256") != source_sha256
            or result.get("excerpt_sha256")
            != sha256(result["text"].encode("utf-8")).hexdigest()
        ):
            raise RuntimeError("AkasicDB source provenance did not verify")

        answer_provenance = _verify_answer_bridge(project, query)

        capability = restarted.capability_payload()["rag"]
        if capability.get("redistribution_authorized") is not False:
            raise RuntimeError("AkasicDB redistribution must remain fail closed")
        deleted = restarted.delete_attachment(attachment_id)
        final = WorkspaceCapabilityService(
            project,
            _smoke_model(),
            akasicdb_path=clone_path,
            answer_integration_schema=RAG_ANSWER_INTEGRATION_SCHEMA,
        )
        if final.query_rag(_SMOKE_QUERY).get("count") != 0:
            raise RuntimeError("deleted AkasicDB evidence survived restart")

    return {
        "schema_version": 1,
        "status": "PASS",
        "repository": AKASICDB_REPOSITORY,
        "revision": AKASICDB_AUDITED_REVISION,
        "retrieval_mode": RAG_RETRIEVAL_MODE,
        "embedding": RAG_EMBEDDING_PROFILE,
        "semantic_embedding": False,
        "answer_integration_configured": True,
        "answer_bridge_contract_verified": True,
        "answer_integration_schema": RAG_ANSWER_INTEGRATION_SCHEMA,
        "integration_schema_authority": (
            "explicit_wiring_contract_not_cryptographic_attestation"
        ),
        "generation_backend": "deterministic_cpu_fixture",
        "actual_model_inference": False,
        "production_attestation": False,
        "restart_recovery": "PASS",
        "delete_recovery": "PASS",
        "source_provenance": "PASS",
        "selected_prompt_provenance": (
            "PASS" if answer_provenance["prompt_excerpt_chars"] > 0 else "FAIL"
        ),
        "license_status": capability["license_status"],
        "redistribution_authorized": False,
        "deleted": deleted["deleted"],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clone", required=True, type=Path)
    args = parser.parse_args()
    print(
        json.dumps(
            validate_akasicdb_rag(args.clone),
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
