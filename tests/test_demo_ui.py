from __future__ import annotations

from html.parser import HTMLParser
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import unittest


ROOT = Path(__file__).resolve().parents[1]
STATIC = ROOT / "cogni_demo" / "static"


class DOMAuditParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.ids: list[str] = []
        self.references: list[tuple[str, str]] = []
        self.buttons: list[dict[str, str]] = []
        self.panels: list[dict[str, str]] = []

    def handle_starttag(self, tag: str, attrs) -> None:
        values = dict(attrs)
        identifier = values.get("id")
        if identifier:
            self.ids.append(identifier)
        for attribute in ("aria-controls", "aria-describedby", "aria-labelledby"):
            for reference in values.get(attribute, "").split():
                self.references.append((attribute, reference))
        if tag == "button":
            self.buttons.append(values)
        if "data-view-panel" in values:
            self.panels.append(values)


class TestCogniBoardUI(unittest.TestCase):
    def test_static_assets_are_local_bounded_and_csp_compatible(self) -> None:
        assets = tuple(STATIC / name for name in ("index.html", "app.css", "app.js"))
        for asset in assets:
            with self.subTest(asset=asset.name):
                self.assertTrue(asset.is_file())
                self.assertGreater(asset.stat().st_size, 0)
                self.assertLessEqual(asset.stat().st_size, 2 * 1024 * 1024)
                text = asset.read_text(encoding="utf-8")
                self.assertNotRegex(text, r"https?://|//cdn\.")

        html = assets[0].read_text(encoding="utf-8")
        self.assertNotRegex(html, r"(?i)<style\b|\sstyle\s*=")
        self.assertNotRegex(html, r"(?i)\son[a-z]+\s*=")
        self.assertIsNone(re.search(r"(?is)<script(?![^>]*\bsrc=)[^>]*>", html))
        favicon = STATIC / "favicon.svg"
        self.assertTrue(favicon.is_file())
        self.assertLessEqual(favicon.stat().st_size, 2 * 1024 * 1024)
        self.assertIn('rel="icon" type="image/svg+xml"', html)

    def test_navigation_and_evidence_taxonomy_are_complete(self) -> None:
        html = (STATIC / "index.html").read_text(encoding="utf-8")
        navigation = set(re.findall(r'data-view="([a-z]+)"', html))
        panels = set(re.findall(r'data-view-panel="([a-z]+)"', html))
        self.assertEqual(
            navigation,
            {
                "assistant",
                "mission",
                "inference",
                "architecture",
                "business",
                "evidence",
            },
        )
        self.assertEqual(navigation, panels)
        for label in ("내부 실측", "구성 검증", "설계 목표", "사업계획"):
            self.assertIn(label, html)
        self.assertIn('data-action="fullscreen"', html)
        self.assertIn("<strong>Moat</strong>", html)
        self.assertNotIn("<strong>Defense</strong>", html)
        for action in (
            "agent-send",
            "agent-cancel",
            "agent-reset",
            "evolution-run",
            "evolution-review",
        ):
            self.assertIn(f'data-action="{action}"', html)

    def test_ai_workspace_accessibility_contract_is_complete(self) -> None:
        html = (STATIC / "index.html").read_text(encoding="utf-8")
        parser = DOMAuditParser()
        parser.feed(html)
        self.assertEqual(len(parser.ids), len(set(parser.ids)), "duplicate DOM id")
        identifiers = set(parser.ids)
        for attribute, reference in parser.references:
            with self.subTest(attribute=attribute, reference=reference):
                self.assertIn(reference, identifiers)
        for button in parser.buttons:
            with self.subTest(
                button=button.get("data-action") or button.get("data-view")
            ):
                self.assertEqual(button.get("type"), "button")

        self.assertEqual(
            [
                panel["data-view-panel"]
                for panel in parser.panels
                if "hidden" not in panel
            ],
            ["assistant"],
        )
        self.assertIn('class="skip-link" href="#main-content"', html)
        self.assertRegex(
            html,
            r'id="chat-transcript"[^>]*role="log"[^>]*aria-live="polite"'
            r'[^>]*aria-relevant="additions text"[^>]*tabindex="0"',
        )
        self.assertRegex(
            html,
            r'id="agent-heading-state"[^>]*role="status"[^>]*aria-live="polite"',
        )
        self.assertIn('aria-keyshortcuts="Control+Enter Meta+Enter"', html)
        self.assertIn('id="agent-char-count" for="agent-input"', html)
        self.assertRegex(
            html,
            r'data-action="agent-focus"[^>]*aria-controls="agent-input"[^>]*disabled',
        )
        self.assertRegex(
            html, r'data-agent-mode="chat"[^>]*aria-pressed="true"[^>]*disabled'
        )
        self.assertRegex(html, r'id="agent-input"[^>]*maxlength="4096"[^>]*disabled')

    def test_ai_workspace_capabilities_start_fail_closed_and_are_accessible(
        self,
    ) -> None:
        html = (STATIC / "index.html").read_text(encoding="utf-8")
        stylesheet = (STATIC / "app.css").read_text(encoding="utf-8")

        for action in (
            "workspace-attach",
            "workspace-rag-toggle",
            "workspace-web-search",
            "workspace-microphone",
        ):
            with self.subTest(action=action):
                self.assertRegex(
                    html,
                    rf'<button class="composer-[^"]+" type="button" '
                    rf'data-action="{action}"[^>]*disabled',
                )

        self.assertRegex(
            html,
            r'id="agent-attachment-input"[^>]*type="file"[^>]*multiple[^>]*disabled',
        )
        self.assertIn('id="agent-attachment-tray"', html)
        self.assertIn('data-action="workspace-rag-reindex"', html)
        self.assertIn('id="attachment-preview-layer"', html)
        self.assertIn('id="attachment-preview-text"', html)
        self.assertIn('id="attachment-preview-image"', html)
        self.assertRegex(
            html,
            r'id="agent-attachment-live-status"[^>]*role="status"'
            r'[^>]*aria-live="polite"',
        )
        for status_id in (
            "agent-attachment-status",
            "agent-rag-status",
            "agent-web-status",
            "agent-microphone-status",
        ):
            self.assertIn(f'id="{status_id}"', html)

        self.assertIn('id="agent-model-selector"', html)
        self.assertRegex(
            html,
            r'id="agent-model-selector"[^>]*aria-describedby="agent-model-status"'
            r"[^>]*disabled",
        )
        self.assertIn("검증 모델 확인 중", html)
        self.assertIn('id="agent-model-status">확인 중', html)
        self.assertIn('id="network-mode-label">NETWORK UNVERIFIED', html)
        self.assertIn('id="external-call-count">—', html)
        self.assertRegex(
            stylesheet,
            r"(?s)\.chat-composer\s*\{[^}]*position:\s*sticky;"
            r"[^}]*bottom:\s*0;",
        )
        self.assertIn(".composer-commandbar", stylesheet)
        self.assertIn(".attachment-chip", stylesheet)
        self.assertIn(".attachment-preview-dialog", stylesheet)
        self.assertIn('.composer-tool-button[aria-pressed="true"]', stylesheet)

    def test_workspace_ui_consumes_bounded_capability_contract(self) -> None:
        script = (STATIC / "app.js").read_text(encoding="utf-8")

        for endpoint in (
            "/api/workspace/capabilities",
            "/api/workspace/attachments",
            "/api/workspace/attachments/add",
            "/api/workspace/attachments/delete",
            "/api/workspace/attachments/preview",
            "/api/workspace/attachments/content",
            "/api/workspace/rag/index",
            "/api/workspace/rag/reindex",
            "/api/workspace/lens/search",
            "/api/workspace/lens/search-and-index",
            "/api/workspace/web/search",
            "/api/workspace/web/cancel",
            "/api/workspace/models/select",
        ):
            with self.subTest(endpoint=endpoint):
                self.assertIn(endpoint, script)
        self.assertIn("MAX_ATTACHMENT_UPLOAD_BYTES = 8 * 1024 * 1024", script)
        self.assertIn("WEB_SEARCH_UI_IMPLEMENTED = true", script)
        self.assertIn("MICROPHONE_CAPTURE_UI_IMPLEMENTED = true", script)
        self.assertIn("reader.readAsDataURL(file)", script)
        self.assertIn("content_base64: contentBase64", script)
        self.assertIn('attachments.state === "enabled"', script)
        self.assertIn('rag.state === "local_index_ready"', script)
        self.assertIn('rag: ui.agentMode === "chat" && ui.ragEnabled', script)
        self.assertIn('if (ui.agentMode !== "chat") {', script)
        self.assertIn("ui.ragEnabled = false", script)
        self.assertIn("MAX_AGENT_CHAT_INPUT_CHARS = 4096", script)
        self.assertIn("MAX_AGENT_PROJECT_INPUT_CHARS = 1 * 1024 * 1024", script)
        self.assertIn('ui.agentMode === "task"', script)
        self.assertIn("input.maxLength =", script)
        self.assertIn('lens.state !== "ready"', script)
        self.assertIn("official_lens_connector", script)
        self.assertIn("general_web_connector", script)
        self.assertIn("lensCalls + generalWebCalls", script)
        self.assertIn("createGeneralWebRequestId", script)
        self.assertIn("searchGeneralWeb", script)
        self.assertIn("cancelGeneralWebSearch", script)
        self.assertIn("renderLensSearchResults", script)
        self.assertIn("verifiedLensUrl", script)
        self.assertIn("LENS_TERMS_REQUIRED", script)
        markup = (STATIC / "index.html").read_text(encoding="utf-8")
        self.assertIn("Data Sourced from The Lens", markup)
        self.assertIn("www.lens.org", markup)
        self.assertIn("about.lens.org/lens-api-terms-of-use/", markup)
        self.assertIn('id="agent-general-web-opt-in"', markup)
        self.assertIn('data-action="workspace-general-web-cancel"', markup)
        self.assertIn("요청 후 즉시 해제", markup)
        self.assertIn("앱 밖의", markup)
        self.assertIn("외부 사이트", script)
        self.assertIn("microphone.runtime_audio_input === true", script)
        self.assertIn("microphone.model_inference_attested === true", script)
        self.assertIn("processor.probe_passed === true", script)
        self.assertIn("tts.host_probe_passed === true", script)
        self.assertIn(
            "function updateExternalCallDisclosure(mode, externalCalls)", script
        )
        self.assertIn('mode === "online_opt_in"', script)
        self.assertIn(
            'setText("#external-call-count", verified ? String(externalCalls) : "—")',
            script,
        )
        self.assertIn('updateExternalCallDisclosure("", null)', script)
        self.assertIn("chip.append(name, state, imageSelect, remove)", script)
        self.assertIn('data-action="workspace-attachment-delete"', script)
        self.assertIn('data-action="workspace-attachment-preview"', script)
        self.assertIn('data-action="workspace-image-select"', script)
        self.assertIn("openWorkspaceAttachmentPreview", script)
        self.assertIn("reindexWorkspaceAttachments", script)
        self.assertIn("item.selectable !== true && item.selected !== true", script)
        self.assertIn("selector.dataset.selectableCount", script)
        self.assertIn("MODEL_SWITCH_UNAVAILABLE", script)
        server = (ROOT / "cogni_demo" / "server.py").read_text(encoding="utf-8")
        self.assertIn("COGNI_OS_MODEL_REGISTRY_DIR", server)
        self.assertIn("attachment_id ${source.attachmentId}", script)
        self.assertIn("chunk_index ${source.chunkIndex}", script)
        self.assertIn("score ${source.score.toFixed(4)}", script)
        self.assertIn('"로컬 저장·모델 미전달"', script)
        self.assertIn("selector.replaceChildren(fragment)", script)

    def test_workspace_controls_fail_closed_on_executable_capability_not_inventory(
        self,
    ) -> None:
        script = (STATIC / "app.js").read_text(encoding="utf-8")

        self.assertIn("ragAnswerIntegrationReady: false", script)
        self.assertIn("rag.answer_integration === true", script)
        self.assertIn(
            'rag.answer_integration_schema === "cogni.agent.retrieval-evidence.v1"',
            script,
        )
        self.assertIn("|| !ui.ragAnswerIntegrationReady", script)
        self.assertIn("&& ui.ragAnswerIntegrationReady", script)
        self.assertIn('return "인덱스 전용"', script)

        controls_start = script.index("function updateWorkspaceControlStates")
        controls_end = script.index("async function loadWorkspaceCapabilities")
        controls = script[controls_start:controls_end]
        self.assertIn("|| !ui.voiceTranscriptionAttemptReady", controls)
        self.assertIn(
            'const selectableCount = Number(modelSelector.dataset.selectableCount || "0")',
            controls,
        )
        self.assertIn("selectableCount < 2", controls)
        self.assertNotIn("verifiedCount < 2", controls)

        voice_start = script.index("async function startVoiceCapture")
        voice_end = script.index("function stopVoiceCapture", voice_start)
        self.assertIn(
            "|| !ui.voiceTranscriptionAttemptReady",
            script[voice_start:voice_end],
        )

        revoke_start = script.index("function revokeWorkspaceCapabilities")
        revoke_end = script.index("function applyWorkspaceCapabilities", revoke_start)
        revoke = script[revoke_start:revoke_end]
        for contract in (
            "ui.workspaceCapabilitiesLoaded = false",
            "ui.ragBackendReady = false",
            "ui.ragAnswerIntegrationReady = false",
            "ui.ragEnabled = false",
            "ui.imageModelIntegrationReady = false",
            "ui.lensConnectorReady = false",
            "ui.voiceTranscriptionConfigured = false",
            "ui.voiceTranscriptionReady = false",
            "ui.voiceTranscriptionAttemptReady = false",
            "ui.voiceSynthesisReady = false",
        ):
            with self.subTest(contract=contract):
                self.assertIn(contract, revoke)

    def test_runtime_authority_is_factbook_driven_and_fails_closed(self) -> None:
        html = (STATIC / "index.html").read_text(encoding="utf-8")
        script = (STATIC / "app.js").read_text(encoding="utf-8")
        stylesheet = (STATIC / "app.css").read_text(encoding="utf-8")

        for capability in (
            "gemma4_e4b",
            "cts_deq",
            "bio_hama",
            "system_1_5",
            "system_2_5",
            "system_3",
            "system_4",
            "aflow",
            "self_harness",
        ):
            with self.subTest(capability=capability):
                self.assertIn(f'data-capability-name="{capability}"', html)

        for disclosure in (
            "AUTHORITATIVE · ANSWER-BEARING",
            "CANARY · ANSWER-BEARING",
            "ADVISORY · TELEMETRY ONLY",
            "GATED · OFF",
            "NIGHT ONLY · INFERENCE OFF",
            "RESEARCH ARCHIVE ONLY",
            "PROPOSAL ONLY · MUTATION BLOCKED",
        ):
            with self.subTest(disclosure=disclosure):
                self.assertIn(disclosure, script)

        self.assertIn("normalizedRuntimeCapabilities(core.capabilities)", script)
        self.assertIn('return "UNVERIFIED · OFF"', script)
        self.assertIn('return "UNVERIFIED"', script)
        self.assertNotIn("AGENT_MODULE_DEFAULTS", script)
        self.assertIn('id="agent-core-badge" role="status">AUTHORITY UNVERIFIED', html)
        self.assertIn(
            'id="architecture-authority-badge">AUTHORITY UNVERIFIED',
            html,
        )
        self.assertNotRegex(
            html,
            r'evidence-badge verified" id="(?:agent-core-badge|architecture-authority-badge)"',
        )
        self.assertNotIn("문장은 경계에서 끝납니다", html)
        self.assertNotIn("내부에서는 텐서만 이동합니다", html)
        self.assertIn("MODEL IPC TENSOR BOUNDARY", html)
        self.assertIn("제품 전체의 tensor-only 주장이 아닙니다", html)
        self.assertIn("not a decoder-wide certificate", html)
        self.assertIn("[data-capability-authority]", stylesheet)
        self.assertIn("[data-execution-module].is-unavailable", stylesheet)

    def test_image_chat_selection_is_explicit_single_turn_and_accessible(self) -> None:
        script = (STATIC / "app.js").read_text(encoding="utf-8")
        stylesheet = (STATIC / "app.css").read_text(encoding="utf-8")

        self.assertIn('selectedImageAttachmentId: ""', script)
        self.assertIn("attachments.image_to_model_integration === true", script)
        self.assertIn("imageCapability.runtime_ready === true", script)
        self.assertIn("imageCapability.model_inference_attested === true", script)
        self.assertIn('imageCapability.state === "configured_unverified"', script)
        self.assertIn(
            'imageCapability.state === "first_image_attestation_in_progress"',
            script,
        )
        self.assertIn("imageCapability.first_use_attestation_allowed === true", script)
        self.assertIn("imageAttestationSettling: false", script)
        self.assertIn("구성됨 · 첫 이미지 검증 필요", script)
        self.assertIn("function settleFirstImageAttestation", script)
        self.assertIn("state?.image_attestation_probe === true", script)
        self.assertIn("function selectWorkspaceImageAttachment", script)
        self.assertIn('imageSelect.setAttribute("aria-pressed"', script)
        self.assertIn(
            "requestBody.image_attachment_id = ui.selectedImageAttachmentId", script
        )
        self.assertIn("state?.image_input_admitted === true", script)
        self.assertIn('ui.selectedImageAttachmentId = ""', script)
        self.assertIn("if (selecting) ui.ragEnabled = false", script)
        self.assertIn("if (ui.ragEnabled) ui.selectedImageAttachmentId", script)
        self.assertIn(".attachment-image-select", stylesheet)
        self.assertIn('[data-image-selected="true"]', stylesheet)

    def test_ai_workspace_uses_xss_safe_bounded_dom_rendering(self) -> None:
        script = (STATIC / "app.js").read_text(encoding="utf-8")
        for forbidden in (
            r"\.innerHTML\b",
            r"\.outerHTML\b",
            r"insertAdjacentHTML",
            r"document\.write",
            r"\beval\s*\(",
            r"new\s+Function\b",
        ):
            with self.subTest(forbidden=forbidden):
                self.assertNotRegex(script, forbidden)
        self.assertIn("renderMessageContentWithCitations", script)
        self.assertIn("message.content.slice(0, MAX_AGENT_RESPONSE_CHARS)", script)
        self.assertIn('includes(message.role) ? message.role : "assistant"', script)
        self.assertIn("messages.slice(-MAX_AGENT_DOM_MESSAGES)", script)
        self.assertIn('"cogni_core_rag"', script)
        self.assertIn("normalizedRetrievalSources(message.sources)", script)
        self.assertIn("rawTitle.replace", script)
        self.assertIn(
            "title.textContent = `[근거 ${source.number}] ${source.title}`", script
        )
        self.assertIn("sourcesList.replaceChildren(fragment)", script)
        self.assertIn("message.sources.length === 0", script)
        self.assertIn("VIEW_IDS.has(initial)", script)
        self.assertNotIn('$(`[data-view-panel="${initial}"]`)', script)

    def test_self_harness_proposal_review_is_accessible_and_read_only(self) -> None:
        html = (STATIC / "index.html").read_text(encoding="utf-8")
        script = (STATIC / "app.js").read_text(encoding="utf-8")
        stylesheet = (STATIC / "app.css").read_text(encoding="utf-8")

        self.assertRegex(
            html,
            r'data-action="evolution-review"[^>]*aria-controls="proposal-review-layer"'
            r'[^>]*aria-haspopup="dialog"[^>]*disabled',
        )
        self.assertRegex(
            html,
            r'class="proposal-review-dialog"[^>]*role="dialog"'
            r'[^>]*aria-modal="true"[^>]*aria-labelledby="proposal-review-title"',
        )
        self.assertIn('id="proposal-review-list"', html)
        self.assertIn("이 화면에는 실행·승인·적용 기능이 없습니다.", html)
        self.assertIn("/api/evolution/proposals", script)
        self.assertIn("renderProposalReview", script)
        self.assertIn('document.createElement("pre")', script)
        self.assertIn("list.replaceChildren(fragment)", script)
        self.assertIn(
            "event.target === event.currentTarget) closeProposalReview()", script
        )
        self.assertIn('event.key === "Escape"', script)
        self.assertIn(".proposal-review-dialog", stylesheet)
        self.assertNotRegex(
            script,
            r"/api/evolution/proposals/(?:apply|approve|execute|promote)",
        )
        self.assertNotRegex(script, r"\.innerHTML\b|insertAdjacentHTML")

    def test_rag_provenance_uses_text_only_dom_for_malicious_titles(self) -> None:
        script = (STATIC / "app.js").read_text(encoding="utf-8")
        malicious_title = '<img src=x onerror="alert(1)">'

        self.assertNotIn(
            malicious_title, (STATIC / "index.html").read_text(encoding="utf-8")
        )
        self.assertIn("normalizedRetrievalSources(items)", script)
        self.assertIn("rawTitle.replace", script)
        self.assertIn("function parseCanonicalRagSourceId", script)
        self.assertIn(
            r"/^[0-9a-f]{24}\.(?:0|[1-9][0-9]{0,2})$/.test(rawSourceId)",
            script,
        )
        self.assertIn(
            "const sourceIdentity = parseCanonicalRagSourceId(item.source_id)",
            script,
        )
        self.assertIn(
            "const provenance = normalizedRetrievalProvenance(item.provenance, sourceIdentity)",
            script,
        )
        self.assertIn("if (!sourceIdentity || !provenance) return", script)
        self.assertIn("function normalizedRetrievalProvenance", script)
        self.assertIn(
            'value.retrieval_mode !== "lexical_only"',
            script,
        )
        self.assertIn("value.semantic_embedding !== false", script)
        self.assertIn(
            'value.answer_integration_schema !== "cogni.agent.retrieval-evidence.v1"',
            script,
        )
        self.assertIn(
            "value.selected_excerpt_chars < value.indexed_excerpt_chars", script
        )
        self.assertIn(
            "value.selected_excerpt_sha256 !== value.indexed_excerpt_sha256",
            script,
        )
        provenance_keys_start = script.index("const RAG_PROVENANCE_EXACT_KEYS")
        provenance_keys_end = script.index("].sort());", provenance_keys_start)
        provenance_keys = set(
            re.findall(
                r'"([a-z0-9_]+)"',
                script[provenance_keys_start:provenance_keys_end],
            )
        )
        self.assertEqual(
            provenance_keys,
            {
                "answer_integration_schema",
                "embedding",
                "indexed_excerpt_chars",
                "indexed_excerpt_sha256",
                "prompt_excerpt_chars",
                "prompt_excerpt_representation",
                "prompt_excerpt_sha256",
                "repository",
                "retrieval_mode",
                "revision",
                "selected_excerpt_chars",
                "selected_excerpt_sha256",
                "selected_excerpt_truncated",
                "semantic_embedding",
                "source_sha256",
            },
        )
        self.assertIn("chunkIndex > 127", script)
        normalization_start = script.index("function normalizedRetrievalSources")
        normalization_end = script.index(
            "function configureRagEvidenceButton", normalization_start
        )
        normalization = script[normalization_start:normalization_end]
        self.assertNotIn(
            'item.source_id.replace(/[^0-9a-zA-Z._-]/g, "")',
            normalization,
        )
        self.assertIn(
            "title.textContent = `[근거 ${source.number}] ${source.title}`", script
        )
        self.assertIn("sourcesList.replaceChildren(fragment)", script)
        self.assertIn("source.provenance.retrievalMode", script)
        self.assertIn("source.provenance.selectedExcerptChars", script)
        self.assertIn("source.provenance.indexedExcerptChars", script)
        self.assertIn("selected excerpt truncated", script)
        self.assertNotRegex(script, r"\.innerHTML\b|insertAdjacentHTML")

    def test_rag_evidence_drawer_is_exact_integrity_checked_and_accessible(
        self,
    ) -> None:
        html = (STATIC / "index.html").read_text(encoding="utf-8")
        script = (STATIC / "app.js").read_text(encoding="utf-8")
        stylesheet = (STATIC / "app.css").read_text(encoding="utf-8")

        self.assertRegex(
            html,
            r'id="evidence-drawer"[^>]*role="dialog"[^>]*aria-modal="true"'
            r'[^>]*aria-labelledby="evidence-drawer-title"'
            r'[^>]*aria-describedby="evidence-drawer-status"[^>]*tabindex="-1"',
        )
        for identifier in (
            "evidence-drawer-layer",
            "evidence-drawer-title",
            "evidence-drawer-status",
            "evidence-drawer-location",
            "evidence-drawer-representation",
            "evidence-drawer-score",
            "evidence-drawer-digest",
            "evidence-drawer-excerpt",
        ):
            self.assertIn(f'id="{identifier}"', html)
        self.assertRegex(
            html,
            r'id="evidence-drawer-status"[^>]*role="status"'
            r'[^>]*aria-live="polite"[^>]*aria-atomic="true"',
        )
        self.assertGreaterEqual(html.count('data-action="rag-evidence-close"'), 2)

        self.assertIn('const RAG_SOURCE_ENDPOINT = "/api/workspace/rag/source"', script)
        self.assertIn(
            "`${RAG_SOURCE_ENDPOINT}?attachment_id=${encodeURIComponent("
            "source.attachmentId)}&chunk_index=${encodeURIComponent("
            "source.chunkIndex)}`",
            script,
        )
        self.assertIn('button.dataset.action = "rag-evidence-open"', script)
        self.assertIn(
            "button.dataset.expectedExcerptSha256 = "
            "source.provenance.indexedExcerptSha256",
            script,
        )
        self.assertIn(
            'const expectedExcerptSha256 = trigger.dataset.expectedExcerptSha256 || ""',
            script,
        )
        self.assertIn(
            "candidate.expectedExcerptSha256 === identity.expectedExcerptSha256",
            script,
        )
        self.assertIn(
            "document.createTextNode(text.slice(cursor, match.index))", script
        )
        self.assertIn("renderMessageContentWithCitations", script)
        self.assertIn("container.replaceChildren(fragment)", script)
        self.assertIn('aria-controls", "evidence-drawer-layer"', script)

        for contract_check in (
            "payload.schema_version !== 2",
            "payload.attachment_id !== requestedSource.attachmentId",
            "payload.chunk_index !== requestedSource.chunkIndex",
            'typeof payload.name !== "string"',
            'typeof payload.media_type !== "string"',
            'typeof payload.text !== "string"',
            "payload.representation !== RAG_SOURCE_REPRESENTATION",
            "!Number.isInteger(payload.char_start)",
            "!Number.isInteger(payload.char_end)",
            "!RAG_SOURCE_OFFSET_BASES.has(payload.offset_basis)",
            "!/^[0-9a-f]{64}$/.test(payload.excerpt_sha256)",
        ):
            with self.subTest(contract_check=contract_check):
                self.assertIn(contract_check, script)
        exact_keys_start = script.index("const RAG_SOURCE_EXACT_KEYS")
        exact_keys_end = script.index("].sort());", exact_keys_start)
        exact_keys = set(
            re.findall(
                r'"([a-z0-9_]+)"',
                script[exact_keys_start:exact_keys_end],
            )
        )
        self.assertEqual(
            exact_keys,
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
        self.assertIn("const payloadKeys = Object.keys(payload).sort()", script)
        self.assertIn("payloadKeys.length !== RAG_SOURCE_EXACT_KEYS.length", script)
        self.assertIn(
            "payloadKeys.some((key, index) => key !== RAG_SOURCE_EXACT_KEYS[index])",
            script,
        )
        self.assertIn(
            "payload.char_end - payload.char_start !== Array.from(payload.text).length",
            script,
        )
        self.assertIn("const documentLocationValid = (", script)
        self.assertIn("const pdfLocationValid = (", script)
        self.assertIn("if (!documentLocationValid && !pdfLocationValid)", script)
        self.assertIn(
            "exactSource.excerptSha256 !== source.expectedExcerptSha256", script
        )
        self.assertIn('ragSourceError("RAG_SOURCE_INTEGRITY_FAILED")', script)
        self.assertIn(
            "!/^[a-z0-9.+-]+\\/[a-z0-9.+-]+$/.test(payload.media_type)", script
        )
        for forbidden_alias_pattern in (
            r"payload\.title\b",
            r"payload\.excerpt\b",
            r"payload\.score\b",
        ):
            self.assertNotRegex(script, forbidden_alias_pattern)
        self.assertIn("source.chunkIndex > 127", script)
        self.assertIn('source.score === null ? "검색 점수 미제공"', script)
        self.assertIn(
            'const RAG_SOURCE_REPRESENTATION = "normalized_extracted_excerpt_v1"',
            script,
        )
        self.assertIn("NORMALIZED EXTRACTED EVIDENCE", html)
        self.assertIn("정규화 추출 발췌 · 원본 첨부 바이트 아님", html)
        self.assertIn(
            "PDF 물리 ${exactSource.pageNumber}쪽의 정규화 추출 텍스트", script
        )
        self.assertNotIn("EXACT LOCAL SOURCE", html)
        self.assertNotIn(">근거 원문<", html)
        self.assertNotIn(">원문 SHA-256<", html)
        self.assertNotIn(">검증된 원문 발췌<", html)

        digest_start = script.index("const actualDigest = await sha256HexUtf8")
        digest_compare = script.index(
            "actualDigest !== exactSource.excerptSha256", digest_start
        )
        excerpt_display = script.index(
            "excerpt.textContent = exactSource.text", digest_compare
        )
        self.assertLess(digest_start, digest_compare)
        self.assertLess(digest_compare, excerpt_display)
        self.assertIn('globalThis.crypto.subtle.digest("SHA-256", bytes)', script)
        self.assertIn('throw ragSourceError("RAG_SOURCE_INTEGRITY_FAILED")', script)
        self.assertIn("excerpt.hidden = true", script)

        self.assertIn("ui.evidenceDrawerRequestId += 1", script)
        self.assertIn("ui.evidenceDrawerAbortController.abort()", script)
        self.assertIn("requestId !== ui.evidenceDrawerRequestId", script)
        self.assertIn("evidenceDrawerOpenerIdentity: null", script)
        self.assertIn("returnFocus instanceof HTMLElement", script)
        self.assertIn("function evidenceTriggerMatchesIdentity", script)
        self.assertIn("function evidenceOpenerIdentity", script)
        self.assertIn("function latestEvidenceOpener", script)
        self.assertIn("candidate.attachmentId === identity.attachmentId", script)
        self.assertIn("candidate.chunkIndex === identity.chunkIndex", script)
        self.assertIn(
            "candidateMessage?.dataset.messageId === identity.messageId",
            script,
        )
        self.assertIn("return matching[matching.length - 1]", script)
        self.assertIn("function evidenceDrawerFallbackFocus", script)
        self.assertIn('const transcript = $("#chat-transcript")', script)
        self.assertIn('const input = $("#agent-input")', script)
        self.assertIn(
            "evidenceTriggerMatchesIdentity(returnFocus, openerIdentity)", script
        )
        self.assertIn("latestEvidenceOpener(openerIdentity)", script)
        self.assertIn("evidenceDrawerFallbackFocus()", script)
        self.assertIn(
            "event.target === event.currentTarget) closeRagEvidenceDrawer()", script
        )
        self.assertIn('event.key === "Escape"', script)
        self.assertIn('trapFocusWithin(event, $("#evidence-drawer"))', script)

        self.assertIn(".evidence-drawer-layer", stylesheet)
        self.assertIn('.evidence-drawer[data-state="error"]', stylesheet)
        self.assertRegex(
            stylesheet,
            r"(?s)@media \(max-width: 620px\).*?\.evidence-drawer\s*\{"
            r"[^}]*width:\s*100vw;[^}]*height:\s*100dvh;",
        )
        self.assertIn('"rag_no_evidence"', script)
        self.assertIn('completionCopy = "로컬 RAG · 근거 없음"', script)

    def test_streaming_cancelling_and_compute_button_states_fail_closed(self) -> None:
        html = (STATIC / "index.html").read_text(encoding="utf-8")
        script = (STATIC / "app.js").read_text(encoding="utf-8")
        stylesheet = (STATIC / "app.css").read_text(encoding="utf-8")
        self.assertIn("item.dataset.messageId = message.key", script)
        self.assertIn(
            'item.setAttribute("aria-busy", String(message.streaming))', script
        )
        self.assertIn(
            "transcript.scrollTop = nearBottom ? transcript.scrollHeight : priorScrollTop",
            script,
        )
        self.assertNotIn('nearBottom || ui.agentStatus === "generating"', script)
        self.assertIn(
            'ui.agentStatus === "cancelling" || ui.agentCancelPending', script
        )
        self.assertIn("agentRequestPending", script)
        self.assertIn("validationRequestPending", script)
        self.assertIn("evolutionRequestPending", script)
        self.assertIn("updateControlStates", script)
        self.assertIn('data-action="agent-send"', html)
        self.assertIn('data-action="agent-cancel"', html)
        self.assertIn('.agent-heading-state[data-state="cancelling"]', stylesheet)

    def test_evolution_count_prefers_source_bearing_review_queue(self) -> None:
        script = (STATIC / "app.js").read_text(encoding="utf-8")
        pending_branch = script.index(
            "const pending = Number.isInteger(evolution.pending_proposals)"
        )
        rich_fallback = script.index(
            ": evolution.rich_pending_proposals;", pending_branch
        )
        self.assertLess(pending_branch, rich_fallback)
        self.assertIn("evolution.unreviewable_proposals", script)
        self.assertIn(
            'badge.dataset.integrity = degraded ? "degraded" : "healthy"', script
        )
        self.assertIn("무결성 제외", script)

    def test_answer_completion_and_truncation_are_visible_and_actionable(self) -> None:
        script = (STATIC / "app.js").read_text(encoding="utf-8")
        stylesheet = (STATIC / "app.css").read_text(encoding="utf-8")
        for field in (
            "finish_reason",
            "continuations",
            "truncated",
            "generated_tokens",
        ):
            self.assertIn(field, script)
        self.assertIn("자동 이어쓰기 ${message.continuations}회 · 완료", script)
        self.assertIn("길이 한계 · 이어서 가능", script)
        self.assertIn("계속 이어서 답해주세요.", script)
        self.assertIn(".chat-completion-status", stylesheet)
        self.assertIn(".chat-message.is-truncated", stylesheet)
        self.assertIn(".chat-continue-button", stylesheet)

    def test_factbook_and_model_generation_are_visibly_distinct(self) -> None:
        html = (STATIC / "index.html").read_text(encoding="utf-8")
        script = (STATIC / "app.js").read_text(encoding="utf-8")
        stylesheet = (STATIC / "app.css").read_text(encoding="utf-8")

        self.assertIn("상태 확인 중", html)
        self.assertIn("Runtime Fact-book", script)
        self.assertIn("FACT-BOOK · 검증된 사실", script)
        self.assertIn(
            '["cogni_core", "cogni_core_rag", "rag_no_evidence", "conversation_fastpath", "factbook", "quality_fallback"]',
            script,
        )
        self.assertIn("대화 FAST PATH · 완료", script)
        self.assertIn("CONVERSATION READY", script)
        self.assertIn("MODEL STANDBY", script)
        self.assertIn("NOT LOADED", script)
        self.assertIn('generationMode === "factbook"', script)
        self.assertIn(
            '.chat-message[data-generation-mode="factbook"] .chat-avatar',
            stylesheet,
        )
        self.assertIn(
            '.chat-message[data-generation-mode="factbook"] .chat-completion-status',
            stylesheet,
        )
        self.assertIn(
            '.chat-message[data-generation-mode="conversation_fastpath"] .chat-avatar',
            stylesheet,
        )
        self.assertIn(
            '.chat-message[data-generation-mode="conversation_fastpath"] .chat-completion-status',
            stylesheet,
        )

    def test_quality_fallback_is_presented_as_failure_not_completion(self) -> None:
        script = (STATIC / "app.js").read_text(encoding="utf-8")
        stylesheet = (STATIC / "app.css").read_text(encoding="utf-8")

        self.assertIn("품질 검증 실패 · 복구 필요", script)
        self.assertIn("RESPONSE FAILED", script)
        self.assertNotIn("품질 안전 응답 · 완료", script)
        self.assertIn(
            '.chat-message[data-generation-mode="quality_fallback"]',
            stylesheet,
        )

    def test_api_error_copy_and_connection_recovery_are_actionable(self) -> None:
        script = (STATIC / "app.js").read_text(encoding="utf-8")
        for code in (
            "AGENT_UNAVAILABLE",
            "AUTH_REQUIRED",
            "COMPUTE_BUSY",
            "EVOLUTION_UNAVAILABLE",
            "NO_ACTIVE_AGENT_TURN",
        ):
            self.assertIn(f"{code}:", script)
        self.assertIn('error.code = "CONNECTION_LOST"', script)
        self.assertIn("describeApiError", script)
        self.assertIn("agentConnectionLost", script)
        self.assertIn("validationConnectionLost", script)
        self.assertIn("로컬 AI 연결이 복구되었습니다.", script)
        self.assertIn("검증 제어 연결이 복구되었습니다.", script)
        self.assertIn(
            "result?.deleted !== true || result?.blob_deleted !== true", script
        )

    def test_responsive_workspace_has_no_global_horizontal_scroll_contract(
        self,
    ) -> None:
        stylesheet = (STATIC / "app.css").read_text(encoding="utf-8")
        script = (STATIC / "app.js").read_text(encoding="utf-8")
        self.assertEqual(stylesheet.count("{"), stylesheet.count("}"))
        for query in (
            "@media (min-width: 1181px) and (max-height: 820px)",
            "@media (max-width: 900px)",
            "@media (max-width: 620px)",
        ):
            self.assertIn(query, stylesheet)
        self.assertRegex(
            stylesheet,
            r"(?s)\.main-stage\s*\{[^}]*overflow-x:\s*hidden;"
            r"[^}]*overflow-y:\s*auto;",
        )
        self.assertIn("grid-template-columns: repeat(6, minmax(0, 1fr));", stylesheet)
        self.assertRegex(
            stylesheet,
            r"(?s)\.chat-workspace\s*\{[^}]*min-width:\s*0;[^}]*overflow:\s*hidden;",
        )
        self.assertRegex(
            stylesheet,
            r"(?s)\.assistant-layout\s*\{[^}]*align-items:\s*start;",
        )
        self.assertRegex(
            stylesheet,
            r"(?s)@media \(min-width: 901px\)\s*\{\s*\.chat-workspace\s*\{"
            r"[^}]*height:\s*clamp\(520px, calc\(100dvh - var\(--topbar\) - 210px\), 900px\);"
            r"[^}]*min-height:\s*0;",
        )
        self.assertRegex(
            stylesheet,
            r"(?s)\.chat-transcript\s*\{[^}]*min-height:\s*0;"
            r"[^}]*overflow-y:\s*auto;[^}]*scrollbar-gutter:\s*stable;",
        )
        self.assertIn('[data-action="agent-focus"]', script)
        self.assertIn("input.focus({ preventScroll: true })", script)
        self.assertRegex(
            stylesheet,
            r"(?s)\.chat-bubble p\s*\{[^}]*overflow-wrap:\s*anywhere;"
            r"[^}]*white-space:\s*pre-wrap;",
        )

    def test_live_audit_trail_accumulates_and_presentation_flow_is_complete(
        self,
    ) -> None:
        html = (STATIC / "index.html").read_text(encoding="utf-8")
        script = (STATIC / "app.js").read_text(encoding="utf-8")
        stylesheet = (STATIC / "app.css").read_text(encoding="utf-8")
        self.assertIn("eventHistory", script)
        self.assertIn("new Map(ui.eventHistory", script)
        self.assertIn("time.dateTime = event.timestamp", script)
        self.assertIn("document.fullscreenElement", script)
        self.assertIn("runValidation();", script)
        self.assertIn("01 / 06", html)
        self.assertIn("tour-progress progress[value]", stylesheet)
        self.assertIn("실행 이벤트 보기", html)
        self.assertNotIn("원본 실행 로그 보기", html)

    def test_live_measurements_start_unverified_and_require_current_success(
        self,
    ) -> None:
        html = (STATIC / "index.html").read_text(encoding="utf-8")
        script = (STATIC / "app.js").read_text(encoding="utf-8")
        for placeholder in (
            'id="metric-vram">—</b>',
            'id="metric-depth">—</b>',
            'id="metric-residual">—</b>',
            'id="reactor-depth">—</strong>',
            'id="telemetry-vram">—</em>',
            'id="rail-device">실행 전 미측정</strong>',
            'id="rail-verdict">NOT VERIFIED</strong>',
        ):
            with self.subTest(placeholder=placeholder):
                self.assertIn(placeholder, html)
        for stale_claim in (
            "14.8469",
            "0.000230",
            "3.906e-3",
            "RTX 5090 Laptop GPU",
            "6.091ms",
        ):
            with self.subTest(stale_claim=stale_claim):
                self.assertNotIn(stale_claim, html)
                self.assertNotIn(stale_claim, script)
        self.assertIn('raw.evidence_kind !== "live_runtime_validation"', script)
        self.assertIn("if (metrics === null) return;", script)
        self.assertIn("data-live-evidence-badge", html)
        self.assertIn('class="rail-seal" data-state="ready"', html)
        self.assertIn('id="telemetry-external-calls">UNVERIFIED', html)
        self.assertIn('id="rail-external-calls">UNVERIFIED', html)
        self.assertNotIn('<dt>Network</dt><dd class="positive">BLOCKED', html)
        self.assertNotIn("100% 폐쇄망", html)

    def test_runtime_evidence_and_modal_disclosures_fail_closed(self) -> None:
        html = (STATIC / "index.html").read_text(encoding="utf-8")
        script = (STATIC / "app.js").read_text(encoding="utf-8")

        for contract in (
            "function normalizedLiveRuntimeMetrics(raw)",
            'raw.evidence_kind !== "live_runtime_validation"',
            'source !== "scripts/validate_gemma4_runtime.py --event-stream"',
            'raw.target !== "RTX 4090 24GB"',
            "raw.transition_residual > 0.005",
            "raw.vram_limit_gib > MAX_LIVE_VRAM_LIMIT_GIB",
            "raw.peak_reserved_vram_gib > raw.vram_limit_gib",
            "raw.peak_reserved_vram_gib < raw.peak_allocated_vram_gib",
            "raw.peak_vram_gib !== Math.max(",
            "resetLiveRuntimePresentation();",
            "setRuntimeStatus(status, state.stage, evidenceVerified)",
            'updateMetrics(liveMetrics, evidenceVerified ? "current" : "prior")',
            'railSeal.dataset.state = status === "succeeded" && !evidenceVerified ? "failed" : status',
            'setText("#rail-verdict", "INVALID EVIDENCE")',
            'badge.textContent = currentEvidence ? "현재 실측" : "직전 검증 실측"',
            'result.textContent = currentEvidence ? "PASS" : "PRIOR PASS"',
        ):
            with self.subTest(contract=contract):
                self.assertIn(contract, script)

        self.assertRegex(
            html,
            r'<span><svg[^>]+aria-hidden="true"><path[^>]+/></svg>'
            r'<span id="mission-external-calls">UNVERIFIED · 앱 외부 호출 상태 미검증</span></span>',
        )
        self.assertIn('setText(\n    "#mission-external-calls",', script)
        lens_start = script.index("async function searchLensOfficialApi")
        lens_end = script.index("function createGeneralWebRequestId", lens_start)
        lens_flow = script[lens_start:lens_end]
        self.assertEqual(
            lens_flow.count("await refreshWorkspaceCapabilityDisclosure()"), 1
        )
        request_start = script.index(
            "async function requestLatestWorkspaceCapabilities"
        )
        request_end = script.index(
            "async function refreshWorkspaceCapabilityDisclosure", request_start
        )
        request_flow = script[request_start:request_end]
        self.assertEqual(request_flow.count('api("/api/workspace/capabilities"'), 1)
        self.assertIn("ui.workspaceCapabilityAbortController?.abort()", request_flow)
        self.assertIn("requestId !== ui.workspaceCapabilityRequestId", request_flow)
        self.assertIn("externalSignal?.aborted", request_flow)
        self.assertIn("await Promise.race([", request_flow)
        self.assertIn("WORKSPACE_CAPABILITY_TIMEOUT", request_flow)

        refresh_start = script.index(
            "async function refreshWorkspaceCapabilityDisclosure"
        )
        refresh_end = script.index(
            "function updateWorkspaceControlStates", refresh_start
        )
        refresh_flow = script[refresh_start:refresh_end]
        self.assertEqual(
            refresh_flow.count("requestLatestWorkspaceCapabilities(options)"), 1
        )
        self.assertNotIn('api("/api/workspace/capabilities"', refresh_flow)
        self.assertIn('updateExternalCallDisclosure("", null)', refresh_flow)
        self.assertIn("revokeWorkspaceCapabilities();", refresh_flow)

        self.assertIn("function syncModalBackgroundBlock()", script)
        self.assertIn("shell.inert = modalOpen", script)
        self.assertIn('shell.setAttribute("aria-hidden", "true")', script)
        self.assertGreaterEqual(script.count("syncModalBackgroundBlock();"), 6)
        self.assertIn(
            'trapFocusWithin(event, $(".attachment-preview-dialog", attachmentLayer))',
            script,
        )
        self.assertIn(
            'trapFocusWithin(event, $(".proposal-review-dialog", proposalLayer))',
            script,
        )

    def test_workspace_capability_refresh_is_latest_request_wins(self) -> None:
        bundled_node = (
            Path.home()
            / ".cache"
            / "codex-runtimes"
            / "codex-primary-runtime"
            / "dependencies"
            / "node"
            / "bin"
            / "node.exe"
        )
        node = shutil.which("node") or (
            str(bundled_node) if bundled_node.is_file() else None
        )
        if node is None:
            self.skipTest("Node.js is not installed in this Python test environment")
        script = (STATIC / "app.js").read_text(encoding="utf-8")
        start = script.index("async function requestLatestWorkspaceCapabilities")
        end = script.index("\n}\n\nfunction updateWorkspaceControlStates", start) + 2
        request_latest = script[start:end]
        load_start = script.index("async function loadWorkspaceCapabilities")
        load_end = (
            script.index(
                "\n}\n\nasync function settleFirstImageAttestation", load_start
            )
            + 2
        )
        load_workspace = script[load_start:load_end]
        probe = (
            r"""
const assert = require("node:assert/strict");
global.window = { setTimeout, clearTimeout };
const WORKSPACE_CAPABILITY_TIMEOUT_MS = 1000;
const ui = {
  workspaceCapabilityRequestId: 0,
  workspaceCapabilityAbortController: null,
};
const pending = [];
const applied = [];
let revoked = 0;
let externalDisclosureClears = 0;
let controlUpdates = 0;
function api(path, options) {
  assert.equal(path, "/api/workspace/capabilities");
  return new Promise((resolve) => pending.push({ resolve, signal: options.signal }));
}
function applyWorkspaceCapabilities(payload) {
  applied.push(payload.version);
  return true;
}
function revokeWorkspaceCapabilities() { revoked += 1; }
function updateExternalCallDisclosure(origin, record) {
  assert.equal(origin, "");
  assert.equal(record, null);
  externalDisclosureClears += 1;
}
function setText() {}
function updateWorkspaceControlStates() { controlUpdates += 1; }
"""
            + request_latest
            + "\n"
            + load_workspace
            + r"""

(async () => {
  const older = requestLatestWorkspaceCapabilities();
  const newer = requestLatestWorkspaceCapabilities();
  assert.equal(pending.length, 2);
  assert.equal(pending[0].signal.aborted, true);
  assert.equal(pending[1].signal.aborted, false);

  pending[1].resolve({ version: "new" });
  const newerResult = await newer;
  pending[0].resolve({ version: "old" });
  const olderResult = await older;

  assert.deepEqual(applied, ["new"]);
  assert.deepEqual(
    { latest: newerResult.latest, applied: newerResult.applied },
    { latest: true, applied: true },
  );
  assert.deepEqual(
    { latest: olderResult.latest, applied: olderResult.applied },
    { latest: false, applied: false },
  );

  const external = new AbortController();
  const cancelled = requestLatestWorkspaceCapabilities({ signal: external.signal, timeoutMs: 1000 });
  assert.equal(pending.length, 3);
  external.abort();
  const cancelledResult = await Promise.race([
    cancelled,
    new Promise((_, reject) => setTimeout(() => reject(new Error("cancel remained pending")), 100)),
  ]);
  assert.deepEqual(
    { latest: cancelledResult.latest, applied: cancelledResult.applied },
    { latest: false, applied: false },
  );
  pending[2].resolve({ version: "cancelled-late" });
  await Promise.resolve();
  assert.deepEqual(applied, ["new"]);

  const timedOut = requestLatestWorkspaceCapabilities({ timeoutMs: 10 });
  assert.equal(pending.length, 4);
  const timeoutResult = await Promise.race([
    timedOut,
    new Promise((_, reject) => setTimeout(() => reject(new Error("timeout remained pending")), 200)),
  ]);
  assert.deepEqual(
    { latest: timeoutResult.latest, applied: timeoutResult.applied },
    { latest: true, applied: false },
  );
  pending[3].resolve({ version: "timeout-late" });
  await Promise.resolve();
  assert.deepEqual(applied, ["new"]);

  const refreshTimedOut = refreshWorkspaceCapabilityDisclosure({ timeoutMs: 10 });
  assert.equal(pending.length, 5);
  const refreshResult = await Promise.race([
    refreshTimedOut,
    new Promise((_, reject) => setTimeout(() => reject(new Error("refresh timeout remained pending")), 200)),
  ]);
  assert.equal(refreshResult, false);
  assert.equal(revoked, 1);
  assert.equal(externalDisclosureClears, 1);
  assert.equal(controlUpdates, 1);
  pending[4].resolve({ version: "refresh-timeout-late" });
  await Promise.resolve();
  assert.deepEqual(applied, ["new"]);

  const loadTimedOut = loadWorkspaceCapabilities();
  assert.equal(pending.length, 6);
  await Promise.race([
    loadTimedOut,
    new Promise((_, reject) => setTimeout(() => reject(new Error("load timeout remained pending")), 1200)),
  ]);
  assert.equal(revoked, 2);
  assert.equal(externalDisclosureClears, 2);
  assert.equal(controlUpdates, 2);
  pending[5].resolve({ version: "load-timeout-late" });
  await Promise.resolve();
  assert.deepEqual(applied, ["new"]);
})().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
"""
        )
        completed = subprocess.run(
            [node, "-e", probe],
            capture_output=True,
            check=False,
            text=True,
            timeout=10,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_validation_ready_and_evidence_transitions_fail_closed(self) -> None:
        node = shutil.which("node")
        if node is None:
            self.skipTest("Node.js is not installed in this Python test environment")
        script = (STATIC / "app.js").read_text(encoding="utf-8")
        start = script.index("function updateState(state)")
        end = script.index("\n}\n\nfunction formatAgentTime", start) + 2
        update_state = script[start:end]
        probe = (
            r"""
const assert = require("node:assert/strict");
const VALIDATION_STATUSES = new Set([
  "ready", "starting", "running", "cancelling", "cancelled", "succeeded", "failed", "offline",
]);
const ACTIVE_STATUSES = new Set(["starting", "running", "cancelling"]);
const ui = { lastStatus: "ready", lastEvidenceVerified: false, lastSeq: -1 };
const labels = Object.create(null);
const runtimeCalls = [];
const metricScopes = [];
const phaseCalls = [];
const toasts = [];
const railSeal = { dataset: Object.create(null) };
const document = { body: { classList: { toggle() {} } } };
function normalizedLiveRuntimeMetrics(raw) {
  return raw?.valid === true ? Object.freeze({ ...raw }) : null;
}
function setRuntimeStatus(status, stage, evidenceVerified) {
  runtimeCalls.push({ status, stage, evidenceVerified });
  labels["#runtime-pill"] = evidenceVerified ? "VERIFIED" : status.toUpperCase();
}
function updateMetrics(metrics, scope) { metricScopes.push({ metrics, scope }); }
function updatePhases(state, evidenceVerified) { phaseCalls.push({ state, evidenceVerified }); }
function renderEvents() {}
function updateControlStates() {}
function setText(selector, value) { labels[selector] = value; }
function showToast(_message, tone) { toasts.push(tone); }
function $(selector) { return selector === ".rail-seal" ? railSeal : null; }
"""
            + update_state
            + r"""

updateState({ status: "ready", stage: "ready", metrics: { valid: true } });
assert.equal(labels["#rail-verdict"], "PRIOR EVIDENCE");
assert.equal(labels["#rail-time"], "직전 검증 실측 · 현재 실행 미검증");
assert.equal(runtimeCalls.at(-1).evidenceVerified, false);
assert.equal(phaseCalls.at(-1).evidenceVerified, false);
assert.equal(metricScopes.at(-1).scope, "prior");
assert.equal(ui.lastEvidenceVerified, false);
assert.equal(railSeal.dataset.state, "ready");
assert.equal(Object.values(labels).includes("VERIFIED"), false);

updateState({ status: "hostile", stage: "postcheck", metrics: { valid: true } });
assert.equal(ui.lastStatus, "failed");
assert.equal(ui.lastEvidenceVerified, false);
assert.equal(labels["#rail-verdict"], "NEW RUN FAILED");
assert.equal(runtimeCalls.at(-1).evidenceVerified, false);
assert.equal(phaseCalls.at(-1).evidenceVerified, false);
assert.equal(metricScopes.at(-1).scope, "prior");
assert.notEqual(railSeal.dataset.state, "succeeded");

toasts.length = 0;
updateState({ status: "succeeded", stage: "postcheck", metrics: null });
assert.deepEqual(toasts, ["error"]);
assert.equal(labels["#rail-verdict"], "INVALID EVIDENCE");
assert.equal(ui.lastEvidenceVerified, false);

updateState({ status: "succeeded", stage: "postcheck", metrics: { valid: true } });
assert.deepEqual(toasts, ["error", "success"]);
assert.equal(labels["#rail-verdict"], "VERIFIED");
assert.equal(ui.lastEvidenceVerified, true);
assert.equal(metricScopes.at(-1).scope, "current");

updateState({ status: "succeeded", stage: "postcheck", metrics: null });
assert.deepEqual(toasts, ["error", "success", "error"]);
assert.equal(labels["#rail-verdict"], "INVALID EVIDENCE");
assert.equal(ui.lastEvidenceVerified, false);
updateState({ status: "succeeded", stage: "postcheck", metrics: null });
assert.deepEqual(toasts, ["error", "success", "error"]);
"""
        )
        completed = subprocess.run(
            [node, "-e", probe],
            capture_output=True,
            check=False,
            text=True,
            timeout=10,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_live_runtime_metric_validator_rejects_partial_and_out_of_range_data(
        self,
    ) -> None:
        node = shutil.which("node")
        if node is None:
            self.skipTest("Node.js is not installed in this Python test environment")
        script = (STATIC / "app.js").read_text(encoding="utf-8")
        start = script.index("function normalizedLiveRuntimeMetrics")
        end = script.index("\n}\n\nfunction showToast", start) + 2
        validator = script[start:end]
        valid = {
            "evidence_kind": "live_runtime_validation",
            "measured_at": "2026-07-17T00:00:00+00:00",
            "source": "scripts/validate_gemma4_runtime.py --event-stream",
            "target": "RTX 4090 24GB",
            "verified_files": 6,
            "model_class": "FakeGemma4",
            "hidden_size": 64,
            "load_seconds": 0.01,
            "inference_seconds": 0.02,
            "requested_depth": 100,
            "reached_depth": 100,
            "nodes_used": 301,
            "node_capacity": 301,
            "search_allocated_bytes": 14994009,
            "transition_converged": True,
            "transition_residual": 0.00390625,
            "transition_used_fallback": False,
            "cts_protocol_version": "SearchRequestV2",
            "safe_for_decode": True,
            "unsafe_silent_fallbacks": 0,
            "linear_solve_fallbacks": 0,
            "solver_rank": 16,
            "solver_history_peak": 16,
            "solver_failures": 0,
            "failed_edges": 0,
            "q_zero_backups": 0,
            "mac_budget": 1000,
            "mac_reserved": 900,
            "act_applied": 301,
            "trace_digest": "a" * 64,
            "causal_bridge_answer_bearing": True,
            "causal_bridge_bias_nonzero": True,
            "causal_bridge_bias_max": 0.04980469,
            "conditioned_generated_tokens": 1,
            "peak_allocated_vram_gib": 14.5,
            "peak_reserved_vram_gib": 14.856,
            "peak_vram_gib": 14.856,
            "vram_limit_gib": 16.7,
            "finite": True,
            "device": "CPU fixture",
        }
        probe = f"""
const assert = require("node:assert/strict");
const MAX_LIVE_VRAM_LIMIT_GIB = 16.7;
function finiteNumber(value) {{ return typeof value === "number" && Number.isFinite(value); }}
{validator}
const valid = {json.dumps(valid, ensure_ascii=True)};
assert.notEqual(normalizedLiveRuntimeMetrics(valid), null);
assert.notEqual(normalizedLiveRuntimeMetrics({{ ...valid, transition_residual: 0.005 }}), null);
assert.equal(normalizedLiveRuntimeMetrics({{ evidence_kind: "live_runtime_validation" }}), null);
const partial = {{ ...valid }}; delete partial.hidden_size;
assert.equal(normalizedLiveRuntimeMetrics(partial), null);
assert.equal(normalizedLiveRuntimeMetrics({{ ...valid, transition_residual: 0.005001 }}), null);
assert.equal(normalizedLiveRuntimeMetrics({{ ...valid, transition_residual: Number.NaN }}), null);
assert.equal(normalizedLiveRuntimeMetrics({{ ...valid, peak_vram_gib: Number.POSITIVE_INFINITY }}), null);
assert.equal(normalizedLiveRuntimeMetrics({{ ...valid, peak_vram_gib: 16.8 }}), null);
assert.equal(normalizedLiveRuntimeMetrics({{ ...valid, peak_allocated_vram_gib: Number.NaN }}), null);
assert.equal(normalizedLiveRuntimeMetrics({{ ...valid, peak_allocated_vram_gib: 15.0, peak_reserved_vram_gib: 14.9, peak_vram_gib: 15.0 }}), null);
assert.equal(normalizedLiveRuntimeMetrics({{ ...valid, peak_reserved_vram_gib: 16.8, peak_vram_gib: 16.8 }}), null);
assert.equal(normalizedLiveRuntimeMetrics({{ ...valid, peak_vram_gib: valid.peak_allocated_vram_gib }}), null);
assert.equal(normalizedLiveRuntimeMetrics({{ ...valid, reached_depth: 99 }}), null);
assert.equal(normalizedLiveRuntimeMetrics({{ ...valid, unsafe_silent_fallbacks: 1 }}), null);
"""
        completed = subprocess.run(
            [node, "-e", probe],
            capture_output=True,
            check=False,
            text=True,
            timeout=10,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_business_claims_preserve_integrity_boundaries(self) -> None:
        html = (STATIC / "index.html").read_text(encoding="utf-8")
        self.assertIn("실행 전 미측정", html)
        self.assertIn("목표 장치", html)
        self.assertIn("RTX 4090", html)
        self.assertIn("라이브 검증 전에는 과거 장치 수치를 표시하지 않습니다", html)
        self.assertIn("전체 시스템 O(1) 주장이 아닙니다", html)
        self.assertIn("컨텍스트 길이별 메모리 비교", html)
        self.assertIn("6.3ms 설계 목표", html)
        self.assertIn("SOM 확정 전", html)

    def test_double_click_and_operator_launchers_have_separate_roles(self) -> None:
        graphical = (ROOT / "Run-CogniOS-Demo.cmd").read_text(encoding="utf-8")
        diagnostic = (ROOT / "Run-CogniOS-CLI.cmd").read_text(encoding="utf-8")
        server_operator = (ROOT / "Run-CogniOS-Server-GPU5.sh").read_text(
            encoding="utf-8"
        )
        native = (ROOT / "launcher" / "CogniBoardLauncher.cs").read_text(
            encoding="utf-8"
        )
        self.assertIn("cogni_demo.server", graphical)
        self.assertIn("scripts\\run_cogniboard_server.py", graphical)
        self.assertNotIn("validate_gemma4_runtime.py", graphical)
        self.assertNotIn("validate_gemma4_runtime.py", diagnostic)
        self.assertIn("validate_master_acceptance_checklist.py", diagnostic)
        self.assertIn("COGNIBOARD_MASTER_ACCEPTANCE_CHECKLIST_KO.md", diagnostic)
        self.assertIn("CPU and Static Integrity Diagnostics", diagnostic)
        self.assertIn("ast.parse", diagnostic)
        self.assertIn("%PYTHON_ARGS% -I -B -X utf8", diagnostic)
        for forbidden in (
            "CUDA_VISIBLE_DEVICES",
            "NVIDIA_VISIBLE_DEVICES",
            "nvidia-smi",
            "server-gpu5-native",
        ):
            self.assertNotIn(forbidden, diagnostic)
        self.assertIn("--validation-profile desktop-ui-only", graphical)
        for forbidden in (
            "server-gpu5-native",
            "COGNI_OS_PHYSICAL_GPU_INDEX",
            "COGNI_OS_GPU_QUERY_CONTEXT",
            "COGNI_OS_GPU_UUID",
            "CUDA_VISIBLE_DEVICES",
            "NVIDIA_VISIBLE_DEVICES",
            "VALIDATION_PYTHON_ARGS",
            "server_gpu5_python_preflight",
            "--expected-source-commit",
        ):
            self.assertNotIn(forbidden, graphical)
        self.assertIn("torch.cuda.is_available()", graphical)
        self.assertNotIn("torch.cuda.is_available()", diagnostic)
        self.assertNotIn("import torch, transformers", diagnostic)
        self.assertIn("HF_HUB_OFFLINE", graphical)
        self.assertIn("HF_HUB_OFFLINE", diagnostic)
        for launcher in (graphical, native):
            with self.subTest(launcher=launcher[:32]):
                self.assertIn("gemma4-e4b-it", launcher)
                self.assertIn("gemma4-e4b-it.manifest.toml", launcher)

        self.assertTrue(server_operator.startswith("#!/usr/bin/bash -p\n"))
        self.assertIn("builtin exec /usr/bin/env -i", server_operator)
        self.assertIn("/usr/bin/bash --noprofile --norc -p --", server_operator)
        self.assertNotIn("/usr/bin/bash -p --noprofile --norc --", server_operator)
        self.assertIn("operator-trusted ELF executable", server_operator)
        self.assertIn('os.path.realpath("/proc/self/exe")', server_operator)
        self.assertIn(
            'exec /usr/bin/env -i "${PYTHON_ENVIRONMENT[@]}"', server_operator
        )
        self.assertIn('"${PYTHON_INVOCATION}" -I -B', server_operator)
        self.assertIn(
            'python_input="${COGNI_OS_PYTHON:-/usr/bin/python3}"', server_operator
        )
        self.assertIn("readlink -f", server_operator)
        self.assertIn("PYTHON_RESOLVED_TARGET", server_operator)
        self.assertIn("PYTHON_SENTINEL_EXPECTED", server_operator)
        self.assertIn("sanitized_stage_environment_is_exact", server_operator)
        self.assertIn("GIT_CONFIG_GLOBAL=/dev/null", server_operator)
        self.assertIn("--validation-profile server-gpu5-native", server_operator)
        self.assertIn("--validation-physical-gpu-index 5", server_operator)
        self.assertIn("--validation-gpu-query-context native-host", server_operator)
        self.assertIn("--expected-source-commit", server_operator)
        self.assertIn("GPU-84d7eeb0-65e0-a5b1-d7db-d09ef59fe03a", server_operator)
        self.assertIn('"CUDA_DEVICE_ORDER=PCI_BUS_ID"', server_operator)
        self.assertIn("status --porcelain=v1 --untracked-files=all", server_operator)
        self.assertIn("BASH_ENV", server_operator)
        self.assertNotIn("nvidia-smi", server_operator)
        self.assertNotIn("/usr/bin/docker", server_operator)
        self.assertIn("gemma4-e4b-it.manifest.toml", server_operator)
        self.assertIn("cogni_demo.server", native)
        self.assertIn("CreateNoWindow = true", native)
        self.assertIn("HF_HUB_OFFLINE", native)
        self.assertIn("load_default_bounded_cts_controller", native)
        self.assertIn("DriveType.Fixed", native)
        self.assertIn("FileAttributes.ReparsePoint", native)
        self.assertIn("QuoteArgument", native)
        self.assertIn("BeginErrorReadLine", native)
        self.assertNotIn("validate_gemma4_runtime.py", native)

        server = (ROOT / "cogni_demo" / "server.py").read_text(encoding="utf-8")
        backend_preflight = server.index(
            'load_default_bounded_cts_controller(device="cpu")',
            server.index("def main("),
        )
        http_bind = server.index("server = DemoHTTPServer(", backend_preflight)
        self.assertLess(backend_preflight, http_bind)

    def test_compiled_launcher_matches_the_instruction_tuned_source(self) -> None:
        binary = ROOT / "CogniBoard.exe"
        self.assertTrue(binary.is_file())
        self.assertGreater(binary.stat().st_size, 0)
        decoded = binary.read_bytes().decode("utf-16-le", errors="ignore")
        old_model = r"C:\Project\cognios\gemma4-e4b"
        new_model = old_model + "-it"
        self.assertIn(new_model, decoded)
        self.assertEqual(decoded.count(old_model), decoded.count(new_model))
        self.assertIn("gemma4-e4b-it.manifest.toml", decoded)
        self.assertNotIn("gemma4-e4b.manifest.toml", decoded)

        if os.name != "nt":
            self.skipTest("native assembly metadata inspection requires Windows")
        environment = dict(os.environ)
        environment["COGNI_EXE"] = str(binary)
        version = subprocess.run(
            [
                "powershell.exe",
                "-NoProfile",
                "-Command",
                "[Reflection.AssemblyName]::GetAssemblyName($env:COGNI_EXE).Version.ToString()",
            ],
            check=True,
            capture_output=True,
            encoding="utf-8",
            env=environment,
            timeout=30,
        ).stdout.strip()
        self.assertEqual(version, "0.4.1.0")


if __name__ == "__main__":
    unittest.main()
