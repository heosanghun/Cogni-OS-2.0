from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from cogni_flow.harness import PatchPolicy, PatchProposal
from cogni_flow.proposals import (
    CandidateDraft,
    ProposalOnlyError,
    ProposalOnlySelfHarness,
)


class TestProposalOnlySelfHarness(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        (self.root / "cogni_core").mkdir()
        (self.root / "cogni_flow").mkdir()
        self.target = self.root / "cogni_core" / "value.py"
        self.target.write_text("VALUE = 1\n", encoding="utf-8")
        (self.root / "cogni_flow" / "harness.py").write_text(
            "SECURITY = True\n", encoding="utf-8"
        )
        self.harness = ProposalOnlySelfHarness(
            self.root,
            ".state/proposals",
            minimum_candidates=3,
        )

    def tearDown(self) -> None:
        self.temporary.cleanup()

    @property
    def source_sha(self) -> str:
        return sha256(self.target.read_bytes()).hexdigest()

    def _failure(self, index: int = 0, *, signature: int = 0):
        return self.harness.record_failure(
            terminal_verifier_cause=f"Verifier-{signature}",
            causal_status=f"causal-{signature}",
            agent_mechanism=f"mechanism-{signature}",
            primary_evidence_sha256=f"{index % 16:x}" * 64,
            source_sha256=self.source_sha,
            reproduction=f"python -m pytest case_{index}",
            observed_ns=index + 1,
        )

    def _drafts(self, base_sha: str | None = None) -> tuple[CandidateDraft, ...]:
        digest = self.source_sha if base_sha is None else base_sha
        return tuple(
            CandidateDraft(
                PatchProposal(
                    "cogni_core/value.py",
                    digest,
                    f"VALUE = {number}\n",
                    f"minimal candidate {number}",
                ),
                expected_behavior=f"case returns {number}",
                risk="bounded constant change",
                reproduction_test=f"pytest case_{number}",
                rollback_trigger="held-out regression or digest mismatch",
            )
            for number in (2, 3, 4)
        )

    def test_capture_coverage_and_manual_signature_precision_exceed_gates(self) -> None:
        expected = {}
        for index in range(1_000):
            signature = index % 5
            event = self._failure(index, signature=signature)
            expected[event.event_id] = event.signature
        self.assertGreaterEqual(self.harness.capture_coverage.ratio, 0.99)

        predicted = {
            trace.test_id: cluster.signature
            for cluster in self.harness.clusters()
            for trace in cluster.traces
        }
        precision = sum(
            predicted[event_id] == signature for event_id, signature in expected.items()
        ) / len(expected)
        self.assertGreaterEqual(precision, 0.95)
        self.assertEqual(len(self.harness.clusters()), 5)
        self.assertEqual(
            len(tuple(self.harness.state_directory.glob("failure-*.json"))),
            1_000,
        )

    def test_success_invariant_is_persisted_outside_the_source_tree(self) -> None:
        success = self.harness.record_success(
            verifier_code="held_out_pass",
            causal_status="verified_success",
            agent_mechanism="agent_manager",
            primary_evidence_sha256="a" * 64,
            observed_ns=10,
        )
        stored = self.harness.state_directory / f"success-{success.invariant_id}.json"
        self.assertTrue(stored.is_file())
        self.assertEqual(len(self.harness.successes), 1)

    def test_three_evidence_linked_candidates_never_mutate_source(self) -> None:
        event = self._failure()
        source_before = self.target.read_bytes()
        cluster = self.harness.clusters()[0]

        proposals = self.harness.submit_cluster_candidates(cluster, self._drafts())

        self.assertEqual(len(proposals), 3)
        self.assertEqual(self.target.read_bytes(), source_before)
        reviewable = dict(self.harness.reviewable_patches)
        for proposal, draft in zip(proposals, self._drafts()):
            self.assertFalse(proposal.source_mutation_allowed)
            self.assertEqual(proposal.status, "pending_review")
            self.assertIn(event.event_id, proposal.event_ids)
            self.assertIn(
                event.primary_evidence_sha256, proposal.primary_evidence_sha256
            )
            stored = (
                self.harness.state_directory / f"proposal-{proposal.proposal_id}.json"
            )
            self.assertTrue(stored.is_file())
            blob = (
                self.harness.replacement_blob_directory
                / f"replacement-{proposal.replacement_sha256}.utf8"
            )
            self.assertEqual(blob.read_bytes(), draft.patch.replacement.encode("utf-8"))
            self.assertEqual(reviewable[proposal.proposal_id], draft.patch)

    def test_multibyte_replacement_round_trips_exactly_across_restart(self) -> None:
        self._failure()
        drafts = list(self._drafts())
        exact = "설명 = '안전한 후보입니다'\n"
        drafts[0] = CandidateDraft(
            PatchProposal(
                "cogni_core/value.py",
                self.source_sha,
                exact,
                "UTF-8 candidate",
            ),
            "Unicode source remains exact",
            "bounded text change",
            "pytest unicode",
            "digest mismatch",
        )
        proposals = self.harness.submit_cluster_candidates(
            self.harness.clusters()[0], tuple(drafts)
        )

        restarted = ProposalOnlySelfHarness(self.root, ".state/proposals")
        restored = dict(restarted.reviewable_patches)[proposals[0].proposal_id]
        self.assertEqual(restored.replacement, exact)
        self.assertEqual(
            restored.replacement.encode("utf-8"),
            (
                restarted.replacement_blob_directory
                / f"replacement-{proposals[0].replacement_sha256}.utf8"
            ).read_bytes(),
        )

    def test_blob_byte_bound_accepts_limit_and_rejects_limit_plus_one(self) -> None:
        maximum = 32
        bounded = ProposalOnlySelfHarness(
            self.root,
            ".state/bounded-proposals",
            policy=PatchPolicy(max_bytes=maximum),
            minimum_candidates=3,
        )
        bounded.record_failure(
            terminal_verifier_cause="Bounded",
            causal_status="limit",
            agent_mechanism="test",
            primary_evidence_sha256="a" * 64,
            source_sha256=self.source_sha,
            reproduction="pytest bound",
            observed_ns=1,
        )

        def drafts(size: int) -> tuple[CandidateDraft, ...]:
            return tuple(
                CandidateDraft(
                    PatchProposal(
                        "cogni_core/value.py",
                        self.source_sha,
                        "#" + character * (size - 2) + "\n",
                        f"{size}-byte candidate",
                    ),
                    "bounded",
                    "low",
                    "pytest bound",
                    "size mismatch",
                )
                for character in "xyz"
            )

        accepted = bounded.submit_cluster_candidates(
            bounded.clusters()[0], drafts(maximum)
        )
        self.assertEqual(len(accepted), 3)
        before = set(bounded.replacement_blob_directory.iterdir())
        with self.assertRaisesRegex(ValueError, "size limit"):
            bounded.submit_cluster_candidates(
                bounded.clusters()[0], drafts(maximum + 1)
            )
        self.assertEqual(set(bounded.replacement_blob_directory.iterdir()), before)

    def test_stale_unlinked_duplicate_and_too_few_candidates_fail_closed(self) -> None:
        self._failure()
        cluster = self.harness.clusters()[0]
        with self.assertRaisesRegex(ProposalOnlyError, "minimum"):
            self.harness.submit_cluster_candidates(cluster, self._drafts()[:2])
        with self.assertRaisesRegex(ProposalOnlyError, "stale"):
            self.harness.submit_cluster_candidates(cluster, self._drafts("f" * 64))
        duplicate = self._drafts()
        with self.assertRaisesRegex(ProposalOnlyError, "distinct"):
            self.harness.submit_cluster_candidates(
                cluster,
                (duplicate[0], duplicate[0], duplicate[1]),
            )

    def test_batch_preflight_leaves_no_partial_proposal_or_blob(self) -> None:
        self._failure()
        drafts = list(self._drafts())
        drafts[1] = CandidateDraft(
            PatchProposal(
                "../escape.py",
                self.source_sha,
                "VALUE = 8\n",
                "escape",
            ),
            "never",
            "critical",
            "pytest escape",
            "always",
        )
        before_records = set(self.harness.state_directory.glob("proposal-*.json"))
        before_blobs = set(self.harness.replacement_blob_directory.iterdir())

        with self.assertRaisesRegex(ValueError, "safe relative"):
            self.harness.submit_cluster_candidates(
                self.harness.clusters()[0], tuple(drafts)
            )

        self.assertEqual(
            set(self.harness.state_directory.glob("proposal-*.json")), before_records
        )
        self.assertEqual(
            set(self.harness.replacement_blob_directory.iterdir()), before_blobs
        )
        self.assertFalse(self.harness.proposals)

    def test_commit_error_rolls_back_batch_files(self) -> None:
        self._failure()
        original = self.harness._persist_json
        proposal_writes = 0

        def fail_second_proposal(path: Path, payload: dict[str, object]) -> None:
            nonlocal proposal_writes
            if path.name.startswith("proposal-"):
                proposal_writes += 1
                if proposal_writes == 2:
                    raise OSError("injected commit failure")
            original(path, payload)

        with patch.object(self.harness, "_persist_json", new=fail_second_proposal):
            with self.assertRaisesRegex(OSError, "injected commit failure"):
                self.harness.submit_cluster_candidates(
                    self.harness.clusters()[0], self._drafts()
                )

        self.assertFalse(tuple(self.harness.state_directory.glob("proposal-*.json")))
        self.assertFalse(tuple(self.harness.replacement_blob_directory.iterdir()))
        self.assertFalse(self.harness.proposals)

    def test_incomplete_persisted_candidate_group_is_not_reviewable(self) -> None:
        self._failure()
        proposals = self.harness.submit_cluster_candidates(
            self.harness.clusters()[0], self._drafts()
        )
        for proposal in proposals[1:]:
            (
                self.harness.state_directory / f"proposal-{proposal.proposal_id}.json"
            ).unlink()

        restarted = ProposalOnlySelfHarness(self.root, ".state/proposals")
        self.assertEqual(len(restarted.proposals), 1)
        self.assertFalse(restarted.reviewable_patches)
        self.assertIn(
            "candidate minimum",
            dict(restarted.unreviewable_proposals)[proposals[0].proposal_id],
        )
        self.assertEqual(restarted.failures, self.harness.failures)

    def test_forbidden_ast_and_immutable_security_surface_are_rejected(self) -> None:
        self._failure()
        cluster = self.harness.clusters()[0]
        forbidden = list(self._drafts())
        forbidden[0] = CandidateDraft(
            PatchProposal(
                "cogni_core/value.py",
                self.source_sha,
                "import socket\nVALUE = 2\n",
                "unsafe",
            ),
            "never",
            "critical",
            "pytest unsafe",
            "always",
        )
        with self.assertRaisesRegex(ValueError, "network"):
            self.harness.submit_cluster_candidates(cluster, forbidden)

        security = self.root / "cogni_flow" / "harness.py"
        security_sha = sha256(security.read_bytes()).hexdigest()
        immutable = CandidateDraft(
            PatchProposal(
                "cogni_flow/harness.py",
                security_sha,
                "SECURITY = False\n",
                "disable",
            ),
            "disable policy",
            "critical",
            "pytest security",
            "always",
        )
        with self.assertRaisesRegex(ProposalOnlyError, "immutable"):
            self.harness.submit_cluster_candidates(
                cluster, (immutable, immutable, immutable)
            )

    def test_link_or_reparse_target_is_never_followed(self) -> None:
        self._failure()
        real = self.root / "outside.py"
        real.write_text("VALUE = 1\n", encoding="utf-8")
        link = self.root / "cogni_core" / "linked.py"
        try:
            link.symlink_to(real)
        except OSError:
            self.skipTest("symlink creation is unavailable")
        link_sha = sha256(real.read_bytes()).hexdigest()
        draft = CandidateDraft(
            PatchProposal("cogni_core/linked.py", link_sha, "VALUE = 2\n", "link"),
            "never follow",
            "critical",
            "pytest link",
            "always",
        )
        # Link digest is not part of the event evidence, so create one explicit
        # linked-source event to reach the path hardening gate.
        self.harness.record_failure(
            terminal_verifier_cause="Verifier-1",
            causal_status="causal-1",
            agent_mechanism="mechanism-1",
            primary_evidence_sha256="e" * 64,
            source_sha256=link_sha,
            reproduction="pytest link",
            observed_ns=9_999,
        )
        linked_cluster = next(
            item
            for item in self.harness.clusters()
            if item.signature[0] == "Verifier-1"
        )
        with self.assertRaisesRegex(ProposalOnlyError, "link/reparse"):
            self.harness.submit_cluster_candidates(
                linked_cluster, (draft, draft, draft)
            )

    def test_failed_candidate_is_kept_as_negative_evidence(self) -> None:
        self._failure()
        proposal = self.harness.submit_cluster_candidates(
            self.harness.clusters()[0], self._drafts()
        )[0]

        negative = self.harness.archive_negative(
            proposal.proposal_id,
            reason_code="held_out_regression",
            evidence_sha256="b" * 64,
        )

        self.assertEqual(negative.proposal_id, proposal.proposal_id)
        self.assertEqual(len(self.harness.negative_archive), 1)
        self.assertTrue(
            (
                self.harness.state_directory / f"negative-{proposal.proposal_id}.json"
            ).is_file()
        )

    def test_restart_hydrates_capture_proposals_and_negative_archive(self) -> None:
        self._failure()
        self.harness.record_success(
            verifier_code="held_out_pass",
            causal_status="verified_success",
            agent_mechanism="agent_manager",
            primary_evidence_sha256="a" * 64,
            observed_ns=10,
        )
        proposals = self.harness.submit_cluster_candidates(
            self.harness.clusters()[0], self._drafts()
        )
        self.harness.archive_negative(
            proposals[0].proposal_id,
            reason_code="held_out_regression",
            evidence_sha256="b" * 64,
        )

        restarted = ProposalOnlySelfHarness(
            self.root,
            ".state/proposals",
            minimum_candidates=3,
        )

        self.assertEqual(restarted.failures, self.harness.failures)
        self.assertEqual(restarted.successes, self.harness.successes)
        self.assertEqual(
            {item.proposal_id for item in restarted.proposals},
            {item.proposal_id for item in self.harness.proposals},
        )
        self.assertEqual(restarted.negative_archive, self.harness.negative_archive)
        self.assertEqual(restarted.capture_coverage.attempted, 1)
        self.assertEqual(restarted.capture_coverage.persisted, 1)
        self.assertEqual(restarted.capture_coverage.ratio, 1.0)
        reviewable = dict(restarted.reviewable_patches)
        self.assertEqual(len(reviewable), 2)
        self.assertNotIn(proposals[0].proposal_id, reviewable)
        self.assertEqual(
            {patch.replacement for patch in reviewable.values()},
            {"VALUE = 3\n", "VALUE = 4\n"},
        )

    def test_blob_failures_exclude_reviewable_patch_but_preserve_evidence(self) -> None:
        event = self._failure()
        proposal = self.harness.submit_cluster_candidates(
            self.harness.clusters()[0], self._drafts()
        )[0]
        blob = (
            self.harness.replacement_blob_directory
            / f"replacement-{proposal.replacement_sha256}.utf8"
        )
        original = blob.read_bytes()

        def assert_excluded(reason: str) -> None:
            restarted = ProposalOnlySelfHarness(
                self.root,
                ".state/proposals",
                minimum_candidates=3,
            )
            self.assertIn(
                event.event_id, {item.event_id for item in restarted.failures}
            )
            self.assertIn(
                proposal.proposal_id,
                {item.proposal_id for item in restarted.proposals},
            )
            self.assertNotIn(proposal.proposal_id, dict(restarted.reviewable_patches))
            failure = dict(restarted.unreviewable_proposals)[proposal.proposal_id]
            self.assertIn(reason, failure)

        blob.unlink()
        assert_excluded("missing")
        blob.write_bytes(original)

        blob.write_bytes(b"VALUE = 999\n")
        assert_excluded("digest")
        blob.write_bytes(original)

        blob.write_bytes(b"x" * (self.harness.policy.max_bytes + 1))
        assert_excluded("byte bound")
        blob.write_bytes(original)

        blob.write_bytes(b"\xff")
        assert_excluded("UTF-8")
        blob.write_bytes(original)

        blob.unlink()
        blob.mkdir()
        assert_excluded("non-regular")
        blob.rmdir()
        blob.write_bytes(original)

        outside = self.root / "outside-replacement.txt"
        outside.write_bytes(original)
        blob.unlink()
        try:
            blob.symlink_to(outside)
        except OSError:
            blob.write_bytes(original)
        else:
            assert_excluded("reparse")
            blob.unlink()
            blob.write_bytes(original)

        original_is_symlink = Path.is_symlink
        with patch.object(
            Path,
            "is_symlink",
            new=lambda candidate: candidate == blob or original_is_symlink(candidate),
        ):
            assert_excluded("reparse")

    def test_hydration_revalidates_base_path_and_python_ast(self) -> None:
        self._failure()
        proposals = self.harness.submit_cluster_candidates(
            self.harness.clusters()[0], self._drafts()
        )
        proposal = proposals[0]

        self.target.write_text("VALUE = 99\n", encoding="utf-8")
        stale = ProposalOnlySelfHarness(self.root, ".state/proposals")
        self.assertFalse(dict(stale.reviewable_patches))
        self.assertTrue(stale.failures)
        self.target.write_text("VALUE = 1\n", encoding="utf-8")

        proposal_path = (
            self.harness.state_directory / f"proposal-{proposal.proposal_id}.json"
        )
        original_payload = json.loads(proposal_path.read_text(encoding="ascii"))

        def write_variant(payload: dict[str, object]) -> tuple[str, Path]:
            identity = dict(payload)
            identity.pop("proposal_id", None)
            proposal_id = sha256(
                json.dumps(
                    identity,
                    ensure_ascii=True,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("ascii")
            ).hexdigest()
            payload = {**identity, "proposal_id": proposal_id}
            path = self.harness.state_directory / f"proposal-{proposal_id}.json"
            path.write_text(
                json.dumps(
                    payload,
                    ensure_ascii=True,
                    sort_keys=True,
                    separators=(",", ":"),
                ),
                encoding="ascii",
            )
            proposal_path.unlink()
            return proposal_id, path

        escaped_payload = json.loads(json.dumps(original_payload))
        escaped_payload["relative_path"] = "../escape.py"
        escaped_id, escaped_path = write_variant(escaped_payload)
        escaped = ProposalOnlySelfHarness(self.root, ".state/proposals")
        self.assertIn(escaped_id, dict(escaped.unreviewable_proposals))
        escaped_path.unlink()
        proposal_path.write_text(
            json.dumps(
                original_payload,
                ensure_ascii=True,
                sort_keys=True,
                separators=(",", ":"),
            ),
            encoding="ascii",
        )

        unsafe = "import socket\nVALUE = 2\n"
        unsafe_raw = unsafe.encode("utf-8")
        unsafe_digest = sha256(unsafe_raw).hexdigest()
        unsafe_blob = (
            self.harness.replacement_blob_directory
            / f"replacement-{unsafe_digest}.utf8"
        )
        unsafe_blob.write_bytes(unsafe_raw)
        unsafe_payload = json.loads(json.dumps(original_payload))
        unsafe_payload["replacement_sha256"] = unsafe_digest
        unsafe_id, unsafe_path = write_variant(unsafe_payload)
        unsafe_state = ProposalOnlySelfHarness(self.root, ".state/proposals")
        self.assertIn(unsafe_id, dict(unsafe_state.unreviewable_proposals))
        self.assertIn("network", dict(unsafe_state.unreviewable_proposals)[unsafe_id])
        unsafe_path.unlink()

    def test_hydration_rejects_tampered_identity_and_oversized_json(self) -> None:
        event = self._failure()
        event_path = self.harness.state_directory / f"failure-{event.event_id}.json"
        payload = json.loads(event_path.read_text(encoding="ascii"))
        payload["causal_status"] = "tampered"
        event_path.write_text(
            json.dumps(payload, sort_keys=True, separators=(",", ":")),
            encoding="ascii",
        )
        with self.assertRaisesRegex(ProposalOnlyError, "identity hash"):
            ProposalOnlySelfHarness(self.root, ".state/proposals")

        event_path.unlink()
        oversized = self.harness.state_directory / f"failure-{'f' * 64}.json"
        oversized.write_bytes(b"{" + b" " * (64 * 1024) + b"}")
        with self.assertRaisesRegex(ProposalOnlyError, "JSON size bound"):
            ProposalOnlySelfHarness(self.root, ".state/proposals")

    def test_hydration_rejects_canonical_unknown_event_cross_reference(self) -> None:
        self._failure()
        proposal = self.harness.submit_cluster_candidates(
            self.harness.clusters()[0], self._drafts()
        )[0]
        old_path = (
            self.harness.state_directory / f"proposal-{proposal.proposal_id}.json"
        )
        payload = json.loads(old_path.read_text(encoding="ascii"))
        payload["event_ids"] = ["f" * 64]
        identity_payload = dict(payload)
        identity_payload.pop("proposal_id")
        new_id = sha256(
            json.dumps(
                identity_payload,
                ensure_ascii=True,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("ascii")
        ).hexdigest()
        payload["proposal_id"] = new_id
        new_path = self.harness.state_directory / f"proposal-{new_id}.json"
        new_path.write_text(
            json.dumps(
                payload,
                ensure_ascii=True,
                sort_keys=True,
                separators=(",", ":"),
            ),
            encoding="ascii",
        )
        old_path.unlink()

        with self.assertRaisesRegex(ProposalOnlyError, "unknown failure event"):
            ProposalOnlySelfHarness(self.root, ".state/proposals")

    def test_hydration_rejects_negative_digest_and_reparse_record(self) -> None:
        self._failure()
        proposal = self.harness.submit_cluster_candidates(
            self.harness.clusters()[0], self._drafts()
        )[0]
        negative = self.harness.archive_negative(
            proposal.proposal_id,
            reason_code="held_out_regression",
            evidence_sha256="b" * 64,
        )
        negative_path = (
            self.harness.state_directory / f"negative-{negative.proposal_id}.json"
        )
        payload = json.loads(negative_path.read_text(encoding="ascii"))
        payload["proposal_sha256"] = "f" * 64
        negative_path.write_text(
            json.dumps(payload, sort_keys=True, separators=(",", ":")),
            encoding="ascii",
        )
        with self.assertRaisesRegex(ProposalOnlyError, "digest does not verify"):
            ProposalOnlySelfHarness(self.root, ".state/proposals")

        negative_path.unlink()
        link = self.harness.state_directory / f"negative-{'e' * 64}.json"
        target = self.root / "outside-ledger.json"
        target.write_text("{}", encoding="ascii")
        try:
            link.symlink_to(target)
        except OSError:
            self.skipTest("symlink creation is unavailable")
        with self.assertRaisesRegex(ProposalOnlyError, "non-regular/reparse"):
            ProposalOnlySelfHarness(self.root, ".state/proposals")


if __name__ == "__main__":
    unittest.main()
