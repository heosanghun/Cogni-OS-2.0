"""Read-only, bounded diff views for proposal-only Self-Harness candidates."""

from __future__ import annotations

from difflib import unified_diff
from hashlib import sha256
import os
from pathlib import Path
import re
from typing import Iterable, Mapping

from .harness import PatchProposal
from .proposals import PatchProposalV1


MAX_REVIEW_ITEMS = 8
MAX_DIFF_CHARS = 40_000
MAX_SOURCE_BYTES = 256_000
_DIGEST_RE = re.compile(r"[0-9a-f]{64}\Z")


class ProposalReviewError(ValueError):
    """A review candidate no longer matches its immutable source evidence."""


def _read_current_source(root: Path, relative_name: str) -> tuple[Path, str, str]:
    relative = Path(relative_name)
    if (
        relative.is_absolute()
        or not relative.parts
        or ".." in relative.parts
        or relative.suffix != ".py"
        or len(relative.as_posix()) > 512
    ):
        raise ProposalReviewError("proposal path is unsafe")
    unresolved = root / relative
    try:
        item_stat = os.lstat(unresolved)
        resolved = unresolved.resolve(strict=True)
    except OSError as exc:
        raise ProposalReviewError("proposal source is unavailable") from exc
    attributes = getattr(item_stat, "st_file_attributes", 0)
    if (
        unresolved.is_symlink()
        or attributes & 0x400
        or root not in resolved.parents
        or resolved != unresolved.absolute()
        or not resolved.is_file()
        or item_stat.st_size > MAX_SOURCE_BYTES
    ):
        raise ProposalReviewError("proposal source crossed its safety boundary")
    try:
        payload = resolved.read_bytes()
        source = payload.decode("utf-8")
    except (OSError, UnicodeError) as exc:
        raise ProposalReviewError("proposal source is not bounded UTF-8") from exc
    return resolved, source, sha256(payload).hexdigest()


def build_proposal_review(
    project_root: str | Path,
    proposals: Iterable[PatchProposalV1],
    reviewable_patches: Mapping[str, PatchProposal],
) -> dict[str, object]:
    """Return diffs only; this API has no mutation, execution, or approval path."""

    root = Path(project_root).expanduser().resolve(strict=True)
    if not root.is_dir():
        raise ProposalReviewError("project root is not a directory")
    if not isinstance(reviewable_patches, Mapping):
        raise TypeError("reviewable_patches must be a mapping")
    records = tuple(proposals)
    if len(records) > MAX_REVIEW_ITEMS:
        records = records[:MAX_REVIEW_ITEMS]
    items: list[dict[str, object]] = []
    for record in records:
        if not isinstance(record, PatchProposalV1):
            raise ProposalReviewError("proposal record is invalid")
        patch = reviewable_patches.get(record.proposal_id)
        if not isinstance(patch, PatchProposal):
            continue
        if (
            _DIGEST_RE.fullmatch(record.proposal_id) is None
            or patch.relative_path != record.relative_path
            or patch.base_sha256.lower() != record.base_sha256
            or sha256(patch.replacement.encode("utf-8")).hexdigest()
            != record.replacement_sha256
            or len(patch.replacement.encode("utf-8")) > MAX_SOURCE_BYTES
        ):
            raise ProposalReviewError("proposal evidence and replacement disagree")
        _path, current, current_digest = _read_current_source(
            root, record.relative_path
        )
        stale = current_digest != record.base_sha256
        if stale:
            diff_text = ""
            truncated = False
        else:
            diff_text = "".join(
                unified_diff(
                    current.splitlines(keepends=True),
                    patch.replacement.splitlines(keepends=True),
                    fromfile=record.relative_path,
                    tofile=f"{record.relative_path} · proposal {record.proposal_id[:12]}",
                    n=3,
                )
            )
            truncated = len(diff_text) > MAX_DIFF_CHARS
            diff_text = diff_text[:MAX_DIFF_CHARS]
        items.append(
            {
                "proposal_id": record.proposal_id,
                "relative_path": record.relative_path,
                "base_sha256": record.base_sha256,
                "replacement_sha256": record.replacement_sha256,
                "rationale": record.rationale[:4_096],
                "expected_behavior": record.expected_behavior[:4_096],
                "risk": record.risk[:4_096],
                "reproduction_test": record.reproduction_test[:4_096],
                "rollback_trigger": record.rollback_trigger[:4_096],
                "status": "stale_base" if stale else record.status,
                "unified_diff": diff_text,
                "diff_truncated": truncated,
                "execution_allowed": False,
                "source_mutation_allowed": False,
            }
        )
    return {
        "schema_version": 1,
        "mode": "proposal_only_read_only",
        "items": items,
        "count": len(items),
        "mutation_endpoint": False,
        "execution_endpoint": False,
    }


__all__ = [
    "MAX_DIFF_CHARS",
    "MAX_REVIEW_ITEMS",
    "ProposalReviewError",
    "build_proposal_review",
]
