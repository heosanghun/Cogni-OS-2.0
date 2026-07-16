from __future__ import annotations

from scripts.render_outstanding_checklist import parse_master, render


def test_outstanding_renderer_keeps_only_unchecked_rows() -> None:
    source = "\n".join(
        (
            "| 1 | [x] | done | `COMPLETED` | evidence | keep |",
            "| 2 | [ ] | partial | `PARTIAL` | evidence | gate |",
            "| 3 | [ ] | absent | `NOT_IMPLEMENTED` | evidence | gate |",
            "| 4 | [ ] | external | `EXTERNAL_BLOCKER` | evidence | gate |",
        )
    )
    records = parse_master(source)
    result = render(records)

    assert [record.identifier for record in records] == [2, 3, 4]
    assert "| 1 |" not in result
    assert "| 2 | [ ] | partial |" in result
    assert "전체 미완료: **3개**" in result


def test_outstanding_renderer_rejects_unchecked_completed_row() -> None:
    source = "| 1 | [ ] | done | `COMPLETED` | evidence | keep |"

    try:
        parse_master(source)
    except ValueError as exc:
        assert "completed row is unchecked" in str(exc)
    else:  # pragma: no cover - defensive assertion
        raise AssertionError("unchecked completed row was accepted")


def test_outstanding_renderer_accepts_a_fully_completed_ledger() -> None:
    records = parse_master("| 1 | [x] | done | `COMPLETED` | evidence | keep |")

    assert records == ()
    assert "전체 미완료: **0개**" in render(records)
