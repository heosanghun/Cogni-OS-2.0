from __future__ import annotations

from hashlib import sha256
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from cogni_demo.server import EvolutionController
from cogni_flow.harness import PatchProposal
from cogni_flow.proposal_review import ProposalReviewError, build_proposal_review
from cogni_flow.proposals import PROPOSAL_SCHEMA, PatchProposalV1


def _records(root: Path):
    source = "def value():\n    return 1\n"
    replacement = "def value():\n    return 2\n"
    target = root / "cogni_core" / "sample.py"
    target.parent.mkdir()
    target.write_text(source, encoding="utf-8", newline="\n")
    base = sha256(source.encode()).hexdigest()
    replacement_digest = sha256(replacement.encode()).hexdigest()
    proposal_id = "a" * 64
    patch = PatchProposal("cogni_core/sample.py", base, replacement, "repair")
    rich = PatchProposalV1(
        PROPOSAL_SCHEMA,
        proposal_id,
        ("RuntimeError", "quality", "agent"),
        ("b" * 64,),
        "cogni_core/sample.py",
        base,
        replacement_digest,
        "repair failing answer completion",
        "one complete answer",
        "unexecuted proposal",
        "pytest -q",
        "reject on regression",
        ("c" * 64,),
    )
    return target, rich, patch


class TestProposalReview(unittest.TestCase):
    def test_evolution_controller_snapshots_ledger_into_read_only_review(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            target, rich, patch = _records(root)
            before = target.read_bytes()
            harness = SimpleNamespace(
                tick=lambda: None,
                evidence_proposals=(rich,),
                proposal_ledger=SimpleNamespace(
                    project_root=root,
                    reviewable_patches=((rich.proposal_id, patch),),
                ),
            )
            controller = EvolutionController(harness)

            payload = controller.proposal_review()

            self.assertEqual(payload["count"], 1)
            self.assertFalse(payload["mutation_endpoint"])
            self.assertFalse(payload["execution_endpoint"])
            self.assertEqual(target.read_bytes(), before)
            controller.shutdown()

    def test_review_is_bounded_read_only_unified_diff(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            target, rich, patch = _records(root)
            before = target.read_bytes()
            payload = build_proposal_review(root, (rich,), {rich.proposal_id: patch})
            self.assertEqual(payload["mode"], "proposal_only_read_only")
            self.assertFalse(payload["mutation_endpoint"])
            item = payload["items"][0]
            self.assertIn("-    return 1", item["unified_diff"])
            self.assertIn("+    return 2", item["unified_diff"])
            self.assertFalse(item["execution_allowed"])
            self.assertEqual(target.read_bytes(), before)

    def test_stale_base_never_receives_a_misleading_diff(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            target, rich, patch = _records(root)
            target.write_text("def value():\n    return 3\n", encoding="utf-8")
            item = build_proposal_review(root, (rich,), {rich.proposal_id: patch})[
                "items"
            ][0]
            self.assertEqual(item["status"], "stale_base")
            self.assertEqual(item["unified_diff"], "")

    def test_evidence_mismatch_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _target, rich, patch = _records(root)
            bad = PatchProposal(
                patch.relative_path,
                patch.base_sha256,
                patch.replacement + "# changed\n",
                patch.rationale,
            )
            with self.assertRaisesRegex(ProposalReviewError, "disagree"):
                build_proposal_review(root, (rich,), {rich.proposal_id: bad})


if __name__ == "__main__":
    unittest.main()
