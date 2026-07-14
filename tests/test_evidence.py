from __future__ import annotations

from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path
import tempfile
import unittest

from cogni_os.capabilities import EvidenceClass, baseline_capability_registry
from cogni_os.evidence import (
    AppendOnlyEvidenceJournal,
    ClaimAssessment,
    ClaimRecord,
    EvidenceError,
    EvidenceRecordV1,
    EvidenceScopeV1,
    FactBookSnapshotStore,
    MODEL_ARTIFACT_CLAIM_ID,
    assess_claims,
    build_factbook_claims,
    capability_claim_id,
    sha256_json,
    sha256_text,
)
from cogni_os.factbook import ModelArtifactFacts, RuntimeFactBook, TensorInventory


def _scope(seed: str = "a") -> EvidenceScopeV1:
    values = [sha256(f"{seed}-{index}".encode()).hexdigest() for index in range(4)]
    return EvidenceScopeV1(*values)


def _evidence(scope: EvidenceScopeV1, claim_id: str = "runtime.factbook"):
    return EvidenceRecordV1.create(
        recorded_at="2026-07-12T00:00:00+00:00",
        kind="runtime_validation",
        producer="cogni.validator",
        run_id="run.20260712.0001",
        evidence_class=EvidenceClass.VERIFIED,
        claim_ids=(claim_id,),
        artifact_sha256=sha256_text("validator-artifact"),
        payload_sha256=sha256_text("raw-validation-result"),
        scope=scope,
        metadata={"runner": "offline-gpu-01", "status": "pass"},
    )


def _claim(evidence_id: str, claim_id: str = "runtime.factbook") -> ClaimRecord:
    return ClaimRecord(
        claim_id=claim_id,
        statement="The runtime FactBook was validated under the exact local scope.",
        evidence_class=EvidenceClass.VERIFIED,
        evidence_ids=(evidence_id,),
    )


def _factbook() -> RuntimeFactBook:
    return RuntimeFactBook(
        schema_version=1,
        generated_at="2026-07-12T00:00:00+00:00",
        build_version="0.3.0-test",
        device="test GPU",
        target_device="RTX 4090 24GB",
        model=ModelArtifactFacts(
            label="gemma4-e4b-test",
            architecture="Gemma4ForConditionalGeneration",
            hidden_size=2_560,
            layers=42,
            dense=True,
            inventory=TensorInventory(
                tensor_count=2,
                stored_parameters=100,
                effective_parameters=80,
                embedding_parameters=20,
                dtype_parameters=(("BF16", 100),),
            ),
            manifest_sha256=sha256_text("model-manifest"),
            config_sha256=sha256_text("model-config"),
        ),
        capabilities=baseline_capability_registry(),
    )


def _factbook_scope(factbook: RuntimeFactBook, seed: str = "a") -> EvidenceScopeV1:
    return replace(_scope(seed), model_sha256=factbook.model.manifest_sha256)


def _capability_bundle(
    factbook: RuntimeFactBook, scope: EvidenceScopeV1
) -> tuple[tuple[ClaimRecord, ...], tuple[EvidenceRecordV1, ...]]:
    verified = tuple(
        record
        for record in factbook.capabilities.records
        if record.evidence is EvidenceClass.VERIFIED
    )
    claim_ids = (MODEL_ARTIFACT_CLAIM_ID,) + tuple(
        capability_claim_id(record.name) for record in verified
    )
    evidence = EvidenceRecordV1.create(
        recorded_at="2026-07-12T00:00:00+00:00",
        kind="release_verification",
        producer="cogni.release_validator",
        run_id="run.20260712.release",
        evidence_class=EvidenceClass.VERIFIED,
        claim_ids=claim_ids,
        artifact_sha256=sha256_text("release-validator-artifact"),
        payload_sha256=sha256_text("release-validation-json"),
        scope=scope,
    )
    mapping = {record.name: (evidence.evidence_id,) for record in verified}
    return (
        build_factbook_claims(
            factbook,
            model_evidence_ids=(evidence.evidence_id,),
            capability_evidence_ids=mapping,
        ),
        (evidence,),
    )


class TestEvidenceRecords(unittest.TestCase):
    def test_verified_claim_requires_a_content_addressed_evidence_id(self):
        with self.assertRaisesRegex(ValueError, "requires at least one evidence_id"):
            ClaimRecord(
                claim_id="runtime.factbook",
                statement="Cannot be verified without evidence.",
                evidence_class=EvidenceClass.VERIFIED,
            )
        with self.assertRaisesRegex(ValueError, "evidence_id"):
            ClaimRecord(
                claim_id="runtime.factbook",
                statement="Cannot cite a display label.",
                evidence_class=EvidenceClass.VERIFIED,
                evidence_ids=("latest-pass",),
            )

    def test_any_scope_digest_change_automatically_marks_claim_stale(self):
        current = _scope()
        record = _evidence(current)
        claim = _claim(record.evidence_id)
        self.assertEqual(
            claim.assess({record.evidence_id: record}, current),
            ClaimAssessment.VERIFIED,
        )
        replacements = {
            "model_sha256": sha256_text("changed-model"),
            "code_sha256": sha256_text("changed-code"),
            "config_sha256": sha256_text("changed-config"),
            "device_sha256": sha256_text("changed-device"),
        }
        for field, digest in replacements.items():
            with self.subTest(field=field):
                changed = replace(current, **{field: digest})
                self.assertTrue(record.is_stale(changed))
                self.assertEqual(
                    claim.assess({record.evidence_id: record}, changed),
                    ClaimAssessment.STALE,
                )

    def test_missing_and_cross_claim_evidence_fail_closed(self):
        scope = _scope()
        record = _evidence(scope, "other.claim")
        claim = _claim(record.evidence_id)
        self.assertEqual(claim.assess({}, scope), ClaimAssessment.MISSING_EVIDENCE)
        self.assertEqual(
            claim.assess({record.evidence_id: record}, scope),
            ClaimAssessment.INVALID_EVIDENCE,
        )
        # Content addressing prevents relabelling an existing record in place.
        with self.assertRaisesRegex(ValueError, "evidence_id"):
            replace(record, evidence_class=EvidenceClass.MEASURED)

    def test_factbook_scope_factory_binds_all_four_dimensions(self):
        factbook = _factbook()
        scope = EvidenceScopeV1.for_factbook(
            factbook,
            code_sha256=sha256_text("code"),
            runtime_config_sha256=sha256_text("runtime-config"),
            device_descriptor="pci=0000:01:00.0|uuid=test|driver=test",
        )
        self.assertEqual(scope.model_sha256, factbook.model.manifest_sha256)
        self.assertEqual(scope.code_sha256, sha256_text("code"))
        self.assertEqual(scope.config_sha256, sha256_text("runtime-config"))
        self.assertEqual(
            scope.device_sha256,
            sha256_text("pci=0000:01:00.0|uuid=test|driver=test"),
        )


class TestAppendOnlyEvidenceJournal(unittest.TestCase):
    def test_journal_must_be_outside_source_tree_and_is_hash_chained(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            source.mkdir()
            with self.assertRaisesRegex(EvidenceError, "outside the source tree"):
                AppendOnlyEvidenceJournal(
                    source / "outputs" / "evidence",
                    source_root=source,
                )
            self.assertFalse((source / "outputs").exists())

            journal = AppendOnlyEvidenceJournal(
                root / "runtime-data" / "evidence",
                source_root=source,
            )
            first = _evidence(_scope("first"), "claim.first")
            second = _evidence(_scope("second"), "claim.second")
            self.assertEqual(journal.append(first), 1)
            self.assertEqual(journal.append(second), 2)
            self.assertEqual(journal.read_records(), (first, second))
            lines = journal.path.read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(lines), 2)
            self.assertEqual(json.loads(lines[1])["sequence"], 2)
            self.assertEqual(
                json.loads(lines[1])["previous_sha256"],
                json.loads(lines[0])["entry_sha256"],
            )

    def test_rewrite_duplicate_and_torn_append_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            source.mkdir()
            journal = AppendOnlyEvidenceJournal(root / "data", source_root=source)
            record = _evidence(_scope())
            journal.append(record)
            with self.assertRaisesRegex(EvidenceError, "already present"):
                journal.append(record)
            journal.path.write_bytes(journal.path.read_bytes()[:-1])
            with self.assertRaisesRegex(EvidenceError, "torn"):
                journal.read_records()


class TestFactBookLastKnownGood(unittest.TestCase):
    def test_atomic_pointer_load_and_corruption_recovery(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            source.mkdir()
            store = FactBookSnapshotStore(root / "state", source_root=source)
            first_factbook = _factbook()
            scope = _factbook_scope(first_factbook)
            first_claims, first_evidence = _capability_bundle(first_factbook, scope)
            for record in first_evidence:
                store.record_evidence(record)
            first = store.publish(
                first_factbook,
                scope=scope,
                claims=first_claims,
                evidence=first_evidence,
            )
            second_factbook = replace(_factbook(), build_version="0.3.1-test")
            second_claims, second_evidence = _capability_bundle(second_factbook, scope)
            # The exact release record is already journaled and may be cited by
            # more than one immutable FactBook generation.
            second = store.publish(
                second_factbook,
                scope=scope,
                claims=second_claims,
                evidence=second_evidence,
            )
            self.assertEqual(store.load_current(scope).snapshot_id, second.snapshot_id)

            pointer = json.loads(store.pointer.read_text(encoding="utf-8"))
            latest = store.snapshots / pointer["filename"]
            latest.write_text("{corrupt", encoding="utf-8")
            recovered = store.recover_last_known_good(scope)
            self.assertEqual(recovered.snapshot_id, first.snapshot_id)
            self.assertEqual(store.load_current(scope).snapshot_id, first.snapshot_id)

    def test_stale_snapshot_and_unverified_verified_claim_never_publish(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            source.mkdir()
            store = FactBookSnapshotStore(root / "state", source_root=source)
            factbook = _factbook()
            scope = _factbook_scope(factbook)
            claims, records = _capability_bundle(factbook, scope)
            for record in records:
                store.record_evidence(record)
            store.publish(factbook, scope=scope, claims=claims, evidence=records)
            changed = replace(scope, code_sha256=sha256_text("new-code"))
            with self.assertRaisesRegex(EvidenceError, "no valid same-scope"):
                store.recover_last_known_good(changed)
            with self.assertRaisesRegex(EvidenceError, "unpublishable"):
                store.publish(factbook, scope=changed, claims=claims, evidence=records)

    def test_source_less_verified_capability_blocks_snapshot(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            source.mkdir()
            store = FactBookSnapshotStore(root / "state", source_root=source)
            factbook = _factbook()
            scope = _factbook_scope(factbook)
            unrelated = _evidence(scope)
            store.record_evidence(unrelated)
            with self.assertRaisesRegex(EvidenceError, "lack evidence-bearing claims"):
                store.publish(
                    factbook,
                    scope=scope,
                    claims=(_claim(unrelated.evidence_id),),
                    evidence=(unrelated,),
                )

    def test_snapshot_cannot_cite_unjournaled_in_memory_evidence(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            source.mkdir()
            store = FactBookSnapshotStore(root / "state", source_root=source)
            factbook = _factbook()
            scope = _factbook_scope(factbook)
            claims, records = _capability_bundle(factbook, scope)
            with self.assertRaisesRegex(EvidenceError, "raw journal"):
                store.publish(factbook, scope=scope, claims=claims, evidence=records)

    def test_publish_rejects_factbook_model_scope_mismatch(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            source.mkdir()
            store = FactBookSnapshotStore(root / "state", source_root=source)
            factbook = _factbook()
            mismatched_scope = _scope("different-model")
            claims, records = _capability_bundle(factbook, mismatched_scope)
            for record in records:
                store.record_evidence(record)

            with self.assertRaisesRegex(EvidenceError, "model manifest digest"):
                store.publish(
                    factbook,
                    scope=mismatched_scope,
                    claims=claims,
                    evidence=records,
                )

    def test_load_rejects_legacy_or_mismatched_model_scope(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            source.mkdir()
            store = FactBookSnapshotStore(root / "state", source_root=source)
            factbook = _factbook()
            scope = _factbook_scope(factbook)
            claims, records = _capability_bundle(factbook, scope)
            for record in records:
                store.record_evidence(record)
            published = store.publish(
                factbook,
                scope=scope,
                claims=claims,
                evidence=records,
            )
            original_path = next(
                path
                for path in store.snapshots.glob("*.json")
                if published.snapshot_id in path.name
            )
            original = json.loads(original_path.read_text(encoding="utf-8"))

            def point_to_variant(payload: dict[str, object]) -> None:
                identity = dict(payload)
                identity.pop("snapshot_id", None)
                snapshot_id = "fb1-" + sha256_json(identity)
                payload = {**identity, "snapshot_id": snapshot_id}
                encoded = (
                    json.dumps(
                        payload,
                        sort_keys=True,
                        separators=(",", ":"),
                        ensure_ascii=False,
                    ).encode("utf-8")
                    + b"\n"
                )
                filename = f"{published.sequence:020d}-{snapshot_id}.json"
                target = store.snapshots / filename
                target.write_bytes(encoded)
                store._atomic_pointer(
                    {
                        "schema_version": 1,
                        "sequence": published.sequence,
                        "snapshot_id": snapshot_id,
                        "filename": filename,
                        "file_sha256": sha256(encoded).hexdigest(),
                    }
                )

            mismatched = json.loads(json.dumps(original))
            mismatched["factbook"]["model"]["manifest_sha256"] = sha256_text(
                "other-model"
            )
            point_to_variant(mismatched)
            with self.assertRaisesRegex(EvidenceError, "model manifest digest"):
                store.load_current(scope)

            legacy = json.loads(json.dumps(original))
            del legacy["scope"]["model_sha256"]
            point_to_variant(legacy)
            with self.assertRaisesRegex(EvidenceError, "evidence scope schema"):
                store.load_current(scope)

    def test_claim_set_rejects_duplicate_ids(self):
        scope = _scope()
        evidence = _evidence(scope)
        claim = _claim(evidence.evidence_id)
        with self.assertRaisesRegex(EvidenceError, "duplicate claim_id"):
            assess_claims((claim, claim), (evidence,), scope)


class TestEvidenceJsonSchema(unittest.TestCase):
    def test_schema_is_strict_and_verified_requires_evidence(self):
        schema_path = Path(__file__).parents[1] / "config" / "evidence.schema.json"
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
        self.assertEqual(
            schema["$schema"], "https://json-schema.org/draft/2020-12/schema"
        )
        evidence_schema = schema["$defs"]["evidenceRecordV1"]
        claim_schema = schema["$defs"]["claimRecordV1"]
        self.assertFalse(evidence_schema["additionalProperties"])
        self.assertFalse(claim_schema["additionalProperties"])
        conditional = claim_schema["allOf"][0]
        self.assertEqual(
            conditional["if"]["properties"]["evidence_class"]["const"],
            "verified",
        )
        self.assertEqual(
            conditional["then"]["properties"]["evidence_ids"]["minItems"], 1
        )


if __name__ == "__main__":
    unittest.main()
