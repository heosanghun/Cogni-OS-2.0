"""Validate one persisted operator Self-Harness E2E evidence chain."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

from cogni_flow.approval import Ed25519ApprovalVerifier
from cogni_flow.self_harness_e2e import (
    SelfHarnessE2ELedger,
    validate_self_harness_e2e,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Read-only validation of an operator-generated Self-Harness E2E chain. "
            "This command performs no signing, source mutation, or runner execution."
        )
    )
    parser.add_argument("--evidence-dir", required=True, type=Path)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--approval-public-key", required=True, type=Path)
    parser.add_argument("--approval-public-key-sha256", required=True)
    parser.add_argument("--approver-id", required=True, action="append")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    verifier = Ed25519ApprovalVerifier(
        args.approval_public_key,
        expected_sha256=args.approval_public_key_sha256,
        approver_ids=tuple(args.approver_id),
    )
    result = validate_self_harness_e2e(
        SelfHarnessE2ELedger(args.evidence_dir),
        args.run_id,
        verifier,
    )
    print(json.dumps(asdict(result), ensure_ascii=True, sort_keys=True))
    return 0 if result.full_e2e_complete else 2


if __name__ == "__main__":
    raise SystemExit(main())
