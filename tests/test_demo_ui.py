from __future__ import annotations

from html.parser import HTMLParser
from pathlib import Path
import re
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
        for action in ("agent-send", "agent-cancel", "agent-reset", "evolution-run"):
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
        self.assertIn("content.textContent = message.content", script)
        self.assertIn("message.content.slice(0, MAX_AGENT_RESPONSE_CHARS)", script)
        self.assertIn('includes(message.role) ? message.role : "assistant"', script)
        self.assertIn("messages.slice(-MAX_AGENT_DOM_MESSAGES)", script)
        self.assertIn("VIEW_IDS.has(initial)", script)
        self.assertNotIn('$(`[data-view-panel="${initial}"]`)', script)

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
            r"[^}]*height:\s*clamp\(500px, calc\(100dvh - var\(--topbar\) - 230px\), 620px\);"
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
        self.assertIn('metrics.evidence_kind === "live_runtime_validation"', script)
        self.assertIn("if (!live) {", script)
        self.assertIn("data-live-evidence-badge", html)
        self.assertIn('class="rail-seal" data-state="ready"', html)
        self.assertIn('App external calls</dt><dd class="positive">DISABLED', html)
        self.assertNotIn('<dt>Network</dt><dd class="positive">BLOCKED', html)
        self.assertNotIn("100% 폐쇄망", html)

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
        native = (ROOT / "launcher" / "CogniBoardLauncher.cs").read_text(
            encoding="utf-8"
        )
        self.assertIn("cogni_demo.server", graphical)
        self.assertNotIn("validate_gemma4_runtime.py", graphical)
        self.assertIn("validate_gemma4_runtime.py", diagnostic)
        self.assertIn("HF_HUB_OFFLINE", graphical)
        self.assertIn("HF_HUB_OFFLINE", diagnostic)
        self.assertIn("cogni_demo.server", native)
        self.assertIn("CreateNoWindow = true", native)
        self.assertIn("HF_HUB_OFFLINE", native)
        self.assertNotIn("validate_gemma4_runtime.py", native)


if __name__ == "__main__":
    unittest.main()
