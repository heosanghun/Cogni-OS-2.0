from __future__ import annotations

from pathlib import Path
import re
import unittest


ROOT = Path(__file__).resolve().parents[1]
STATIC = ROOT / "cogni_demo" / "static"


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
            {"mission", "inference", "architecture", "business", "evidence"},
        )
        self.assertEqual(navigation, panels)
        for label in ("내부 실측", "구성 검증", "설계 목표", "사업계획"):
            self.assertIn(label, html)
        self.assertIn('data-action="fullscreen"', html)
        self.assertIn("<strong>Moat</strong>", html)
        self.assertNotIn("<strong>Defense</strong>", html)

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

    def test_business_claims_preserve_integrity_boundaries(self) -> None:
        html = (STATIC / "index.html").read_text(encoding="utf-8")
        self.assertIn("RTX 5090 Laptop GPU", html)
        self.assertIn("목표 장치", html)
        self.assertIn("RTX 4090", html)
        self.assertIn("현재 실측은 목표 RTX 4090 결과가 아닙니다", html)
        self.assertIn("전체 시스템 O(1) 주장이 아닙니다", html)
        self.assertIn("컨텍스트 길이별 메모리 비교", html)
        self.assertIn("6.3ms 설계 목표", html)
        self.assertIn("SOM 확정 전", html)

    def test_double_click_and_operator_launchers_have_separate_roles(self) -> None:
        graphical = (ROOT / "Run-CogniOS-Demo.cmd").read_text(encoding="utf-8")
        diagnostic = (ROOT / "Run-CogniOS-CLI.cmd").read_text(encoding="utf-8")
        self.assertIn("cogni_demo.server", graphical)
        self.assertNotIn("validate_gemma4_runtime.py", graphical)
        self.assertIn("validate_gemma4_runtime.py", diagnostic)
        self.assertIn("HF_HUB_OFFLINE", graphical)
        self.assertIn("HF_HUB_OFFLINE", diagnostic)


if __name__ == "__main__":
    unittest.main()
