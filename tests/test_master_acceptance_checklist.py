from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from scripts.validate_master_acceptance_checklist import (
    ChecklistValidationError,
    validate_checklist,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CHECKLIST = PROJECT_ROOT / "docs" / "COGNIBOARD_MASTER_ACCEPTANCE_CHECKLIST_KO.md"


class TestMasterAcceptanceChecklist(unittest.TestCase):
    def test_repository_ledger_covers_every_requirement_exactly_once(self) -> None:
        report = validate_checklist(CHECKLIST)
        self.assertEqual(len(report.records), 170)
        self.assertEqual(
            [record.requirement_id for record in report.records], list(range(1, 171))
        )
        self.assertEqual(sum(report.counts.values()), 170)
        self.assertEqual(
            report.as_payload()["incomplete_count"],
            sum(
                count for state, count in report.counts.items() if state != "COMPLETED"
            ),
        )

    def test_checked_box_cannot_overclaim_a_partial_requirement(self) -> None:
        text = CHECKLIST.read_text(encoding="utf-8")
        corrupted = text.replace(
            "| 4 | [ ] | RTX 4090 24GB 목표 장치 검증 | `EXTERNAL_BLOCKER` |",
            "| 4 | [x] | RTX 4090 24GB 목표 장치 검증 | `EXTERNAL_BLOCKER` |",
            1,
        )
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "checklist.md"
            path.write_text(corrupted, encoding="utf-8")
            with self.assertRaisesRegex(
                ChecklistValidationError, "checked exactly for COMPLETED"
            ):
                validate_checklist(path)

    def test_declared_summary_must_match_table(self) -> None:
        text = CHECKLIST.read_text(encoding="utf-8")
        completed = validate_checklist(CHECKLIST).counts["COMPLETED"]
        corrupted = text.replace(
            f"COMPLETED {completed}", f"COMPLETED {completed - 1}", 1
        )
        self.assertNotEqual(corrupted, text)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "checklist.md"
            path.write_text(corrupted, encoding="utf-8")
            with self.assertRaisesRegex(ChecklistValidationError, "do not match"):
                validate_checklist(path)


if __name__ == "__main__":
    unittest.main()
