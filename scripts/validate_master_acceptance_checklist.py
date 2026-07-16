"""Validate the 1--170 CogniBoard acceptance ledger.

The ledger is a release gate, not prose.  This validator rejects missing or
duplicated requirement IDs, unknown states, empty evidence/exit conditions,
and a checked box that does not exactly match ``COMPLETED``.  It also verifies
the summary totals printed near the top of the Markdown document.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
import argparse
import json
import re


EXPECTED_IDS = frozenset(range(1, 171))
VALID_STATES = frozenset(
    {"COMPLETED", "PARTIAL", "NOT_IMPLEMENTED", "EXTERNAL_BLOCKER"}
)
_ROW = re.compile(
    r"^\|\s*(?P<id>\d{1,3})\s*\|\s*\[(?P<check>[ xX])\]\s*\|"
    r"(?P<requirement>.*?)\|\s*`(?P<state>[A-Z_]+)`\s*\|"
    r"(?P<evidence>.*?)\|(?P<condition>.*?)\|\s*$"
)
_SUMMARY = re.compile(
    r"현재 스냅샷 집계는\s*`COMPLETED\s+(?P<COMPLETED>\d+)\s*/\s*"
    r"PARTIAL\s+(?P<PARTIAL>\d+)\s*/\s*NOT_IMPLEMENTED\s+"
    r"(?P<NOT_IMPLEMENTED>\d+)\s*/\s*EXTERNAL_BLOCKER\s+"
    r"(?P<EXTERNAL_BLOCKER>\d+)`",
    re.MULTILINE,
)


class ChecklistValidationError(ValueError):
    """The acceptance ledger violates its machine-checkable contract."""


@dataclass(frozen=True, slots=True)
class ChecklistRecord:
    requirement_id: int
    checked: bool
    requirement: str
    state: str
    evidence: str
    completion_condition: str


@dataclass(frozen=True, slots=True)
class ChecklistReport:
    records: tuple[ChecklistRecord, ...]
    counts: dict[str, int]

    def as_payload(self) -> dict[str, object]:
        incomplete = [
            record.requirement_id
            for record in self.records
            if record.state != "COMPLETED"
        ]
        return {
            "schema_version": 1,
            "requirements": len(self.records),
            "counts": dict(self.counts),
            "incomplete_count": len(incomplete),
            "incomplete_ids": incomplete,
            "valid": True,
        }


def validate_checklist(path: str | Path) -> ChecklistReport:
    source = Path(path)
    text = source.read_text(encoding="utf-8", errors="strict")
    records: list[ChecklistRecord] = []
    for line_number, line in enumerate(text.splitlines(), 1):
        match = _ROW.fullmatch(line)
        if match is None:
            continue
        requirement_id = int(match.group("id"))
        state = match.group("state")
        if state not in VALID_STATES:
            raise ChecklistValidationError(
                f"line {line_number}: unknown state {state!r}"
            )
        record = ChecklistRecord(
            requirement_id=requirement_id,
            checked=match.group("check").casefold() == "x",
            requirement=match.group("requirement").strip(),
            state=state,
            evidence=match.group("evidence").strip(),
            completion_condition=match.group("condition").strip(),
        )
        if not all((record.requirement, record.evidence, record.completion_condition)):
            raise ChecklistValidationError(
                f"line {line_number}: requirement, evidence, and completion "
                "condition must be non-empty"
            )
        if record.checked != (record.state == "COMPLETED"):
            raise ChecklistValidationError(
                f"line {line_number}: checkbox must be checked exactly for COMPLETED"
            )
        records.append(record)

    identifiers = [record.requirement_id for record in records]
    duplicates = sorted(
        requirement_id
        for requirement_id, count in Counter(identifiers).items()
        if count > 1
    )
    missing = sorted(EXPECTED_IDS.difference(identifiers))
    unexpected = sorted(set(identifiers).difference(EXPECTED_IDS))
    if duplicates or missing or unexpected:
        raise ChecklistValidationError(
            "requirement ID coverage failed: "
            f"duplicates={duplicates}, missing={missing}, unexpected={unexpected}"
        )
    ordered = tuple(sorted(records, key=lambda record: record.requirement_id))
    counts = {state: 0 for state in sorted(VALID_STATES)}
    counts.update(Counter(record.state for record in ordered))
    summary = _SUMMARY.search(text)
    if summary is None:
        raise ChecklistValidationError("machine-readable status summary is missing")
    declared = {state: int(summary.group(state)) for state in VALID_STATES}
    if declared != counts:
        raise ChecklistValidationError(
            f"declared status counts {declared} do not match ledger {counts}"
        )
    if sum(counts.values()) != len(EXPECTED_IDS):
        raise ChecklistValidationError("status counts do not total 170")
    return ChecklistReport(ordered, counts)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "checklist",
        nargs="?",
        default="docs/COGNIBOARD_MASTER_ACCEPTANCE_CHECKLIST_KO.md",
    )
    parser.add_argument("--json", action="store_true", dest="as_json")
    args = parser.parse_args()
    report = validate_checklist(args.checklist)
    payload = report.as_payload()
    if args.as_json:
        print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    else:
        counts = payload["counts"]
        print(
            "PASS: 170 requirements; "
            + ", ".join(f"{state}={counts[state]}" for state in sorted(counts))
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
