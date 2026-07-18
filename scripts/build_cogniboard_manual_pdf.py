"""Render the Korean CogniBoard operator manual as a print-ready PDF."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from html import escape
import os
from pathlib import Path
import re
from runpy import run_path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    CondPageBreak,
    Flowable,
    KeepTogether,
    LongTable,
    PageBreak,
    Paragraph,
    Preformatted,
    SimpleDocTemplate,
    Spacer,
    TableStyle,
)


NAVY = colors.HexColor("#07111F")
NAVY_2 = colors.HexColor("#0D1D2F")
CYAN = colors.HexColor("#19AFCF")
CYAN_LIGHT = colors.HexColor("#DDF7FC")
VIOLET = colors.HexColor("#7567D8")
MINT = colors.HexColor("#168A62")
TEXT = colors.HexColor("#172536")
MUTED = colors.HexColor("#52677A")
LINE = colors.HexColor("#CEDBE5")
PAPER = colors.HexColor("#F7FAFC")
AMBER = colors.HexColor("#A56A00")
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
PRODUCT_VERSION = str(
    run_path(str(_PROJECT_ROOT / "cogni_os" / "version.py"))["__version__"]
)
if re.fullmatch(r"\d+\.\d+\.\d+", PRODUCT_VERSION) is None:
    raise RuntimeError("cogni_os.version.__version__ must be semantic version text")


def document_date() -> str:
    """Return a reproducible UTC date when SOURCE_DATE_EPOCH is available."""

    source_epoch = os.environ.get("SOURCE_DATE_EPOCH", "")
    if re.fullmatch(r"\d{1,12}", source_epoch):
        return (
            datetime.fromtimestamp(int(source_epoch), timezone.utc).date().isoformat()
        )
    return datetime.now().astimezone().date().isoformat()


def register_fonts() -> None:
    regular = Path(r"C:\Windows\Fonts\malgun.ttf")
    bold = Path(r"C:\Windows\Fonts\malgunbd.ttf")
    if not regular.is_file() or not bold.is_file():
        raise FileNotFoundError(
            "Malgun Gothic fonts are required for Korean PDF output"
        )
    pdfmetrics.registerFont(TTFont("Malgun", str(regular)))
    pdfmetrics.registerFont(TTFont("Malgun-Bold", str(bold)))


class ArchitectureDiagram(Flowable):
    def __init__(self, width: float = 168 * mm, height: float = 76 * mm) -> None:
        super().__init__()
        self.width = width
        self.height = height

    def _box(
        self,
        canvas,
        x: float,
        y: float,
        width: float,
        height: float,
        title: str,
        subtitle: str,
        *,
        fill,
        stroke=LINE,
        title_color=TEXT,
    ) -> None:
        canvas.setFillColor(fill)
        canvas.setStrokeColor(stroke)
        canvas.roundRect(x, y, width, height, 7, fill=1, stroke=1)
        canvas.setFillColor(title_color)
        canvas.setFont("Malgun-Bold", 9)
        canvas.drawCentredString(x + width / 2, y + height - 14, title)
        canvas.setFillColor(
            MUTED if title_color == TEXT else colors.HexColor("#CFEAF2")
        )
        canvas.setFont("Malgun", 6.8)
        canvas.drawCentredString(x + width / 2, y + 10, subtitle)

    @staticmethod
    def _arrow(canvas, x1: float, y1: float, x2: float, y2: float) -> None:
        canvas.setStrokeColor(CYAN)
        canvas.setFillColor(CYAN)
        canvas.setLineWidth(1.4)
        canvas.line(x1, y1, x2, y2)
        canvas.line(x2, y2, x2 - 5, y2 + 3)
        canvas.line(x2, y2, x2 - 5, y2 - 3)

    def draw(self) -> None:
        canvas = self.canv
        canvas.saveState()
        canvas.setFillColor(PAPER)
        canvas.setStrokeColor(LINE)
        canvas.roundRect(0, 0, self.width, self.height, 10, fill=1, stroke=1)

        top_y = self.height - 65
        box_w = 116
        gap = 25
        start_x = (self.width - (box_w * 3 + gap * 2)) / 2
        self._box(
            canvas,
            start_x,
            top_y,
            box_w,
            43,
            "왼쪽 메뉴",
            "6개 업무 화면",
            fill=colors.white,
        )
        self._box(
            canvas,
            start_x + box_w + gap,
            top_y,
            box_w,
            43,
            "중앙 작업 영역",
            "대화 · 검증 · 설계",
            fill=CYAN_LIGHT,
            stroke=CYAN,
        )
        self._box(
            canvas,
            start_x + (box_w + gap) * 2,
            top_y,
            box_w,
            43,
            "Evidence Rail",
            "현재 프로세스 증거",
            fill=colors.white,
        )
        self._arrow(
            canvas, start_x + box_w, top_y + 22, start_x + box_w + gap - 4, top_y + 22
        )
        self._arrow(
            canvas,
            start_x + box_w * 2 + gap,
            top_y + 22,
            start_x + (box_w + gap) * 2 - 4,
            top_y + 22,
        )

        canvas.setFillColor(NAVY)
        canvas.roundRect(18, 16, self.width - 36, 85, 9, fill=1, stroke=0)
        core_y = 37
        core_w = 126
        core_gap = 31
        core_x = (self.width - (core_w * 3 + core_gap * 2)) / 2
        self._box(
            canvas,
            core_x,
            core_y,
            core_w,
            45,
            "Cogni-Flow",
            "CPU 제어면 · 낮/밤 리듬",
            fill=NAVY_2,
            stroke=CYAN,
            title_color=colors.white,
        )
        self._box(
            canvas,
            core_x + core_w + core_gap,
            core_y,
            core_w,
            45,
            "Bounded Tensor IPC",
            "shape · dtype · timeout 고정",
            fill=NAVY_2,
            stroke=VIOLET,
            title_color=colors.white,
        )
        self._box(
            canvas,
            core_x + (core_w + core_gap) * 2,
            core_y,
            core_w,
            45,
            "Cogni-Core",
            "Gemma · DEQ · CTS · Sys 1.5–4",
            fill=NAVY_2,
            stroke=CYAN,
            title_color=colors.white,
        )
        self._arrow(
            canvas,
            core_x + core_w,
            core_y + 23,
            core_x + core_w + core_gap - 5,
            core_y + 23,
        )
        self._arrow(
            canvas,
            core_x + core_w * 2 + core_gap,
            core_y + 23,
            core_x + (core_w + core_gap) * 2 - 5,
            core_y + 23,
        )
        canvas.restoreState()


def inline_markup(text: str) -> str:
    value = escape(text.strip())
    value = re.sub(r"\[([^]]+)]\([^)]+\)", r"\1", value)
    value = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", value)
    value = re.sub(
        r"`([^`]+)`",
        r'<font name="Malgun-Bold" color="#33536B">\1</font>',
        value,
    )
    return value


def build_styles() -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    return {
        "cover_kicker": ParagraphStyle(
            "CoverKicker",
            parent=base["Normal"],
            fontName="Malgun-Bold",
            fontSize=10,
            textColor=colors.HexColor("#75E8FF"),
            alignment=TA_CENTER,
            leading=14,
            spaceAfter=8,
        ),
        "cover_title": ParagraphStyle(
            "CoverTitle",
            parent=base["Title"],
            fontName="Malgun-Bold",
            fontSize=31,
            textColor=colors.white,
            alignment=TA_CENTER,
            leading=39,
            spaceAfter=16,
        ),
        "cover_subtitle": ParagraphStyle(
            "CoverSubtitle",
            parent=base["Normal"],
            fontName="Malgun",
            fontSize=12,
            textColor=colors.HexColor("#C7D8E5"),
            alignment=TA_CENTER,
            leading=20,
        ),
        "h1": ParagraphStyle(
            "H1",
            parent=base["Heading1"],
            fontName="Malgun-Bold",
            fontSize=20,
            textColor=NAVY,
            leading=27,
            spaceBefore=10,
            spaceAfter=10,
            keepWithNext=True,
        ),
        "h2": ParagraphStyle(
            "H2",
            parent=base["Heading2"],
            fontName="Malgun-Bold",
            fontSize=13.2,
            textColor=CYAN,
            leading=19,
            spaceBefore=9,
            spaceAfter=6,
            keepWithNext=True,
        ),
        "body": ParagraphStyle(
            "Body",
            parent=base["BodyText"],
            fontName="Malgun",
            fontSize=9.2,
            textColor=TEXT,
            leading=15.2,
            alignment=TA_LEFT,
            spaceAfter=6,
            wordWrap="CJK",
        ),
        "bullet": ParagraphStyle(
            "Bullet",
            parent=base["BodyText"],
            fontName="Malgun",
            fontSize=8.9,
            textColor=TEXT,
            leading=14.5,
            leftIndent=12,
            firstLineIndent=-8,
            spaceAfter=3,
            wordWrap="CJK",
        ),
        "number": ParagraphStyle(
            "Number",
            parent=base["BodyText"],
            fontName="Malgun",
            fontSize=8.9,
            textColor=TEXT,
            leading=14.5,
            leftIndent=18,
            firstLineIndent=-14,
            spaceAfter=3,
            wordWrap="CJK",
        ),
        "code": ParagraphStyle(
            "Code",
            parent=base["Code"],
            fontName="Malgun",
            fontSize=7.6,
            textColor=NAVY,
            backColor=CYAN_LIGHT,
            borderColor=CYAN_LIGHT,
            borderWidth=6,
            borderPadding=8,
            leading=12,
            leftIndent=4,
            rightIndent=4,
            spaceBefore=4,
            spaceAfter=8,
        ),
        "table": ParagraphStyle(
            "Table",
            parent=base["BodyText"],
            fontName="Malgun",
            fontSize=7.2,
            textColor=TEXT,
            leading=10.5,
            wordWrap="CJK",
        ),
        "table_head": ParagraphStyle(
            "TableHead",
            parent=base["BodyText"],
            fontName="Malgun-Bold",
            fontSize=7.2,
            textColor=colors.white,
            leading=10.5,
            wordWrap="CJK",
        ),
    }


def table_flowable(rows: list[list[str]], styles: dict[str, ParagraphStyle]):
    width = 168 * mm
    columns = max(len(row) for row in rows)
    normalized = [row + [""] * (columns - len(row)) for row in rows]
    data = []
    for row_index, row in enumerate(normalized):
        style = styles["table_head"] if row_index == 0 else styles["table"]
        data.append([Paragraph(inline_markup(cell), style) for cell in row])
    table = LongTable(
        data,
        colWidths=[width / columns] * columns,
        repeatRows=1,
        hAlign="LEFT",
        splitByRow=1,
    )
    commands = [
        ("BACKGROUND", (0, 0), (-1, 0), NAVY_2),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Malgun-Bold"),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("GRID", (0, 0), (-1, -1), 0.35, LINE),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]
    for index in range(1, len(data)):
        commands.append(
            (
                "BACKGROUND",
                (0, index),
                (-1, index),
                colors.white if index % 2 else PAPER,
            )
        )
    table.setStyle(TableStyle(commands))
    return table


def markdown_story(path: Path, styles: dict[str, ParagraphStyle]):
    lines = path.read_text(encoding="utf-8").splitlines()
    start = next(index for index, line in enumerate(lines) if line.startswith("## 1."))
    lines = lines[start:]
    story = []
    paragraph: list[str] = []

    def flush_paragraph() -> None:
        if paragraph:
            story.append(Paragraph(inline_markup(" ".join(paragraph)), styles["body"]))
            paragraph.clear()

    def consume_list_continuations(value: str, next_index: int) -> tuple[str, int]:
        """Join Markdown list continuation lines into one indivisible item."""

        parts = [value]
        while next_index < len(lines):
            continuation = lines[next_index].rstrip()
            if not continuation or not continuation[:1].isspace():
                break
            parts.append(continuation.strip())
            next_index += 1
        return " ".join(parts), next_index

    index = 0
    while index < len(lines):
        line = lines[index].rstrip()
        if not line:
            flush_paragraph()
            index += 1
            continue
        if line == "<!-- PDF_PAGE_BREAK -->":
            flush_paragraph()
            story.append(PageBreak())
            index += 1
            continue
        if line.startswith("```"):
            flush_paragraph()
            language = line[3:].strip()
            block: list[str] = []
            index += 1
            while index < len(lines) and not lines[index].startswith("```"):
                block.append(lines[index])
                index += 1
            if language == "mermaid":
                story.extend(
                    [Spacer(1, 3 * mm), ArchitectureDiagram(), Spacer(1, 4 * mm)]
                )
            else:
                # Command examples are intentionally short. Keeping each one intact
                # avoids stranding a continuation fragment at the top of a page.
                story.append(
                    KeepTogether([Preformatted("\n".join(block), styles["code"])])
                )
            index += 1
            continue
        if line.startswith("|") and index + 1 < len(lines):
            separator = lines[index + 1].strip()
            if separator.startswith("|") and set(separator) <= set("|-: "):
                flush_paragraph()
                rows: list[list[str]] = []
                while index < len(lines) and lines[index].strip().startswith("|"):
                    row = [
                        cell.strip()
                        for cell in lines[index].strip().strip("|").split("|")
                    ]
                    if not (set("".join(row)) <= set("-: ")):
                        rows.append(row)
                    index += 1
                story.extend([table_flowable(rows, styles), Spacer(1, 4 * mm)])
                continue
        if line.startswith("## "):
            flush_paragraph()
            title = line[3:].strip()
            story.append(CondPageBreak(40 * mm))
            story.append(Paragraph(inline_markup(title), styles["h1"]))
            index += 1
            continue
        if line.startswith("### "):
            flush_paragraph()
            story.append(CondPageBreak(25 * mm))
            story.append(Paragraph(inline_markup(line[4:]), styles["h2"]))
            index += 1
            continue
        if line.startswith("- "):
            flush_paragraph()
            value = line[2:].strip()
            value, index = consume_list_continuations(value, index + 1)
            marker = "□" if value.startswith("[ ]") else "•"
            value = value[3:].strip() if value.startswith("[ ]") else value
            story.append(
                KeepTogether(
                    [
                        Paragraph(
                            f"{marker}&nbsp;&nbsp;{inline_markup(value)}",
                            styles["bullet"],
                        )
                    ]
                )
            )
            continue
        numbered = re.match(r"^(\d+)\.\s+(.+)$", line)
        if numbered:
            flush_paragraph()
            value, index = consume_list_continuations(numbered.group(2), index + 1)
            story.append(
                KeepTogether(
                    [
                        Paragraph(
                            f"<b>{numbered.group(1)}.</b>&nbsp;&nbsp;{inline_markup(value)}",
                            styles["number"],
                        )
                    ]
                )
            )
            continue
        paragraph.append(line.strip())
        index += 1
    flush_paragraph()
    return story


def page_background(canvas, doc) -> None:
    canvas.saveState()
    width, height = A4
    if doc.page == 1:
        canvas.setFillColor(NAVY)
        canvas.rect(0, 0, width, height, fill=1, stroke=0)
        canvas.setFillColor(CYAN)
        canvas.rect(0, height - 5 * mm, width, 5 * mm, fill=1, stroke=0)
        canvas.setStrokeColor(colors.HexColor("#18364E"))
        canvas.setLineWidth(0.5)
        for offset in range(0, 110, 14):
            canvas.circle(
                width / 2, height * 0.48, (42 + offset) * mm, fill=0, stroke=1
            )
    else:
        canvas.setFillColor(colors.white)
        canvas.rect(0, 0, width, height, fill=1, stroke=0)
        canvas.setStrokeColor(LINE)
        canvas.line(20 * mm, height - 14 * mm, width - 20 * mm, height - 14 * mm)
        canvas.setFont("Malgun-Bold", 7.5)
        canvas.setFillColor(CYAN)
        canvas.drawString(20 * mm, height - 10 * mm, "COGNIBOARD · OPERATOR PLAYBOOK")
        canvas.setFont("Malgun", 7.2)
        canvas.setFillColor(MUTED)
        canvas.drawRightString(width - 20 * mm, 10 * mm, f"{doc.page - 1:02d}")
        canvas.drawString(
            20 * mm,
            10 * mm,
            f"Cogni-OS 2.0 Genesis v{PRODUCT_VERSION} · Local only",
        )
    canvas.restoreState()


def build_pdf(input_path: Path, output_path: Path) -> None:
    register_fonts()
    styles = build_styles()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=A4,
        rightMargin=21 * mm,
        leftMargin=21 * mm,
        topMargin=20 * mm,
        bottomMargin=18 * mm,
        title="CogniBoard 사용자 매뉴얼 및 운영 플레이북",
        author="Cogni-OS 2.0 Genesis",
        subject=f"CogniBoard v{PRODUCT_VERSION} operator manual",
    )
    story = [
        Spacer(1, 57 * mm),
        Paragraph("SOVEREIGN AI MISSION CONTROL", styles["cover_kicker"]),
        Paragraph("CogniBoard", styles["cover_title"]),
        Paragraph("사용자 매뉴얼 및<br/>운영 플레이북", styles["cover_title"]),
        Spacer(1, 7 * mm),
        Paragraph(
            "로컬 Gemma 4 E4B · Cogni-Core · 라이브 검증 · Self-Harness<br/>"
            f"버전 {PRODUCT_VERSION} · {document_date()}",
            styles["cover_subtitle"],
        ),
        Spacer(1, 55 * mm),
        Paragraph(
            "LOCAL ONLY&nbsp;&nbsp;·&nbsp;&nbsp;EVIDENCE FIRST&nbsp;&nbsp;·&nbsp;&nbsp;FAIL CLOSED",
            styles["cover_kicker"],
        ),
        PageBreak(),
    ]
    story.extend(markdown_story(input_path, styles))
    doc.build(story, onFirstPage=page_background, onLaterPages=page_background)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    build_pdf(args.input.resolve(), args.output.resolve())
    print(f"manual_pdf={args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
