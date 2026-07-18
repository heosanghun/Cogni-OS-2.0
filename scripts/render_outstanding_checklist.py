"""Render the non-completed CogniBoard requirements as a separate checklist.

The master ledger remains the single source of truth.  This renderer prevents a
manually copied "remaining work" list from silently drifting away from it.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.validate_master_acceptance_checklist import validate_checklist  # noqa: E402


DEFAULT_SOURCE = ROOT / "docs" / "COGNIBOARD_MASTER_ACCEPTANCE_CHECKLIST_KO.md"
DEFAULT_OUTPUT = ROOT / "docs" / "COGNIBOARD_OUTSTANDING_IMPLEMENTATION_CHECKLIST_KO.md"
VALID_OUTSTANDING_STATES = (
    "IMPLEMENTED_UNVERIFIED",
    "NOT_IMPLEMENTED",
    "PARTIAL",
    "EXTERNAL_BLOCKER",
)


@dataclass(frozen=True, slots=True)
class Requirement:
    identifier: int
    requirement: str
    status: str
    evidence: str
    gate: str


def parse_master(
    source: str | Path,
    *,
    release_attestation: str | Path | None = None,
    release_attestation_signature: str | Path | None = None,
    verifier_public_key: str | Path | None = None,
) -> tuple[Requirement, ...]:
    """Return outstanding rows only after the full 1..170 ledger gate passes."""

    path = Path(source)
    report = validate_checklist(
        path,
        release_attestation=release_attestation,
        release_attestation_signature=release_attestation_signature,
        verifier_public_key=verifier_public_key,
    )
    return tuple(
        Requirement(
            identifier=record.requirement_id,
            requirement=record.requirement,
            status=record.state,
            evidence=record.evidence,
            gate=record.completion_condition,
        )
        for record in report.records
        if record.state != "COMPLETED"
    )


def render(records: tuple[Requirement, ...]) -> str:
    counts = {
        state: sum(record.status == state for record in records)
        for state in VALID_OUTSTANDING_STATES
    }
    lines = [
        "# CogniBoard 미완료 항목 전용 체크리스트",
        "",
        "> 이 문서는 `COGNIBOARD_MASTER_ACCEPTANCE_CHECKLIST_KO.md`에서 자동 생성됩니다.",
        "> 직접 수정하지 말고 마스터 원장의 상태·근거·승격 조건을 먼저 갱신하십시오.",
        "",
        "## 현재 집계",
        "",
        f"- 전체 미완료: **{len(records)}개**",
        f"- 구현됐으나 승인 증거 미결합: **{counts['IMPLEMENTED_UNVERIFIED']}개**",
        f"- 코드/제품 경로 미구현: **{counts['NOT_IMPLEMENTED']}개**",
        f"- 부분 구현 또는 검증 잔여: **{counts['PARTIAL']}개**",
        f"- 외부 장치·토큰·아티팩트 차단: **{counts['EXTERNAL_BLOCKER']}개**",
        "",
    ]
    headings = {
        "IMPLEMENTED_UNVERIFIED": "구현됐으나 승인 증거 미결합",
        "NOT_IMPLEMENTED": "제품 실행 경로 미구현",
        "PARTIAL": "부분 구현·검증 잔여",
        "EXTERNAL_BLOCKER": "외부 입력 없이는 완료 불가능",
    }
    for state in VALID_OUTSTANDING_STATES:
        lines.extend(
            [
                f"## {headings[state]}",
                "",
                "| ID | 체크 | 요구사항 | 현재 근거 | 완료 승격 조건 |",
                "|---:|:---:|---|---|---|",
            ]
        )
        for record in records:
            if record.status != state:
                continue
            lines.append(
                f"| {record.identifier} | [ ] | {record.requirement} | "
                f"{record.evidence} | {record.gate} |"
            )
        lines.append("")
    lines.extend(
        [
            "## 완료 판정 규칙",
            "",
            "- 코드 파일이나 버튼의 존재만으로 완료 처리하지 않습니다.",
            "- 제품 경로 연결, bounded/fail-closed 안전성, 자동 회귀 또는 실측 증거가 모두 있어야 합니다.",
            "- 구현 경로가 있어도 승인된 verifier의 exact-scope 서명 증거가 없으면 `IMPLEMENTED_UNVERIFIED`로 유지합니다.",
            "- `config/acceptance-evidence-policy.json`의 ID별 basis·kind·component·raw schema와 source-pinned SHA를 모두 만족해야 합니다.",
            "- 완료 scope는 source-pinned verifier key의 detached signed release attestation에서만 파생하며, 사용자 입력 digest는 신뢰하지 않습니다.",
            "- RTX 4090과 승인된 Lens 토큰·약관처럼 외부 입력이 필요한 항목은 제공 전까지 차단 상태를 유지합니다.",
            "- 완료된 ID는 이 문서에서 자동으로 사라지고 마스터 원장에만 `[x]`로 남습니다.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--release-attestation")
    parser.add_argument("--release-attestation-signature")
    parser.add_argument("--verifier-public-key")
    args = parser.parse_args()
    rendered = render(
        parse_master(
            args.source,
            release_attestation=args.release_attestation,
            release_attestation_signature=args.release_attestation_signature,
            verifier_public_key=args.verifier_public_key,
        )
    )
    if args.check:
        if (
            not args.output.exists()
            or args.output.read_text(encoding="utf-8") != rendered
        ):
            print(f"OUTDATED: {args.output}", file=sys.stderr)
            return 1
        print(f"PASS: {args.output}")
        return 0
    args.output.write_text(rendered, encoding="utf-8")
    print(f"WROTE: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
