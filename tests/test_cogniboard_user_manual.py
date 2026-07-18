from __future__ import annotations

from pathlib import Path
import re
import unittest


ROOT = Path(__file__).resolve().parents[1]
MANUAL = ROOT / "docs" / "COGNIBOARD_USER_MANUAL_PLAYBOOK_KO.md"
PDF_BUILDER = ROOT / "scripts" / "build_cogniboard_manual_pdf.py"


class TestCogniBoardUserManual(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.text = MANUAL.read_text(encoding="utf-8")

    def test_manual_targets_v040_and_documents_all_six_pages(self) -> None:
        self.assertIn("문서 기준 버전: 0.4.1", self.text)
        for title in (
            "AI 워크스페이스",
            "미션 컨트롤",
            "라이브 검증",
            "시스템 설계",
            "사업 임팩트",
            "증빙 · 로드맵",
        ):
            with self.subTest(title=title):
                self.assertRegex(
                    self.text,
                    re.compile(rf"^### 4\.\d {re.escape(title)}$", re.MULTILINE),
                )

    def test_manual_covers_every_new_operator_workflow(self) -> None:
        required = (
            "첨부·미리보기·삭제·재색인",
            "cogni_core_image",
            "AkasicDB",
            "Lens.org",
            "반드시 필요한 4개 gate",
            "LOCAL_STT_ARTIFACT_REQUIRED",
            "LOCAL_TTS_ARTIFACT_REQUIRED",
            "`/project` PoC·MVP 번들",
            "읽기 전용 제안 검토",
        )
        for marker in required:
            with self.subTest(marker=marker):
                self.assertIn(marker, self.text)

    def test_manual_keeps_external_blockers_and_claim_limits_explicit(self) -> None:
        for marker in (
            "실제 GPU에서 이미지 질문 품질·VRAM·지연",
            "Lens 계정 승인",
            "동일 manifest-bound Gemma 4",
            "다화자 WER",
            "목표 RTX 4090",
            "자동 승격은 차단",
            "독립 패킷 감사",
        ):
            with self.subTest(marker=marker):
                self.assertIn(marker, self.text)

    def test_print_layout_keeps_code_and_major_sections_together(self) -> None:
        self.assertEqual(self.text.count("<!-- PDF_PAGE_BREAK -->"), 3)
        builder = PDF_BUILDER.read_text(encoding="utf-8")
        self.assertIn('line == "<!-- PDF_PAGE_BREAK -->"', builder)
        self.assertIn("KeepTogether", builder)
        self.assertIn('Preformatted("\\n".join(block)', builder)
        self.assertIn("consume_list_continuations", builder)
        self.assertIn("not continuation[:1].isspace()", builder)

    def test_wrapped_markdown_list_items_exist_as_layout_regressions(self) -> None:
        self.assertIn(
            "- `외부 호출 0`: 현재 앱이 기록한 외부 호출 수다. 독립 패킷 감사를 대신하지\n"
            "  않는다.",
            self.text,
        )
        self.assertIn(
            "2. 브라우저에서 `LOCAL ONLY`, 리듬 `INFERENCE`, 오른쪽 Evidence Rail을\n"
            "   확인한다.",
            self.text,
        )


if __name__ == "__main__":
    unittest.main()
