from __future__ import annotations

from pathlib import Path
import tempfile

import pytest

from scripts.render_outstanding_checklist import parse_master, render
from scripts.validate_master_acceptance_checklist import ChecklistValidationError


ROOT = Path(__file__).resolve().parents[1]
MASTER = ROOT / "docs" / "COGNIBOARD_MASTER_ACCEPTANCE_CHECKLIST_KO.md"


def test_outstanding_renderer_consumes_only_a_fully_validated_master() -> None:
    records = parse_master(MASTER)
    result = render(records)

    assert len(records) == 170
    assert [record.identifier for record in records] == list(range(1, 171))
    assert "전체 미완료: **170개**" in result
    assert "구현됐으나 승인 증거 미결합: **97개**" in result
    assert "| 1 | [ ] | Gemma 4 E4B-it 로컬 백본 |" in result


@pytest.mark.parametrize(
    "source",
    (
        "",
        "| 1 | [ ] | only row | `PARTIAL` | evidence | gate |\n",
    ),
)
def test_outstanding_renderer_rejects_empty_or_one_row_master(source: str) -> None:
    with tempfile.TemporaryDirectory() as temporary:
        path = Path(temporary) / "master.md"
        path.write_text(source, encoding="utf-8")
        with pytest.raises(
            ChecklistValidationError, match="ID coverage|size is outside policy"
        ):
            parse_master(path)


def test_outstanding_renderer_rejects_duplicate_summary() -> None:
    text = MASTER.read_text(encoding="utf-8")
    lines = text.splitlines()
    marker_index = next(
        index for index, line in enumerate(lines) if "현재 스냅샷 집계는" in line
    )
    marker = "\n".join(lines[marker_index : marker_index + 2])
    with tempfile.TemporaryDirectory() as temporary:
        path = Path(temporary) / "master.md"
        path.write_text(text + "\n" + marker + "\n", encoding="utf-8")
        with pytest.raises(ChecklistValidationError, match="exactly one"):
            parse_master(path)
