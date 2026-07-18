from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
import re
import tempfile
import unittest
from unittest.mock import patch

import scripts.validate_master_acceptance_checklist as checklist_validator
from scripts.validate_master_acceptance_checklist import (
    ChecklistValidationError,
    validate_checklist,
)
from scripts.validate_release_evidence import RSA_SHA256_DIGEST_INFO
from tests.test_release_evidence_validation import TEST_RSA_D, TEST_RSA_N


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CHECKLIST = PROJECT_ROOT / "docs" / "COGNIBOARD_MASTER_ACCEPTANCE_CHECKLIST_KO.md"


def _encoded(value: object) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode()


def _identity_digest(value: object) -> str:
    canonical = json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return sha256(canonical).hexdigest()


def _refresh_citation(checklist_path: Path, record_data: bytes) -> None:
    text = checklist_path.read_text(encoding="utf-8")
    refreshed, replacements = re.subn(
        r"(release/evidence/acceptance-record\.json#sha256=)[0-9a-f]{64}",
        lambda match: match.group(1) + sha256(record_data).hexdigest(),
        text,
    )
    if replacements == 0:
        raise AssertionError("acceptance citation fixture is missing")
    checklist_path.write_text(refreshed, encoding="utf-8")


def _sign(payload: bytes) -> bytes:
    width = (TEST_RSA_N.bit_length() + 7) // 8
    digest_info = RSA_SHA256_DIGEST_INFO + sha256(payload).digest()
    encoded = (
        b"\x00\x01" + b"\xff" * (width - len(digest_info) - 3) + b"\x00" + digest_info
    )
    signature = pow(int.from_bytes(encoded, "big"), TEST_RSA_D, TEST_RSA_N)
    return signature.to_bytes(width, "big").hex().encode()


class TestMasterAcceptanceChecklist(unittest.TestCase):
    @staticmethod
    def _promote(
        text: str,
        requirement_ids: tuple[int, ...],
        evidence: str,
    ) -> str:
        selected = set(requirement_ids)
        lines = text.splitlines()
        found: set[int] = set()
        for index, line in enumerate(lines):
            if not line.startswith("|"):
                continue
            fields = line.split("|")
            if len(fields) != 8:
                continue
            try:
                requirement_id = int(fields[1].strip())
            except ValueError:
                continue
            if requirement_id not in selected:
                continue
            self_state = fields[4].strip()
            if self_state != "`IMPLEMENTED_UNVERIFIED`":
                raise AssertionError(
                    f"ID {requirement_id} is not promotable in this fixture"
                )
            fields[2] = " [x] "
            fields[4] = " `COMPLETED` "
            fields[5] = f" {evidence} "
            lines[index] = "|".join(fields)
            found.add(requirement_id)
        if found != selected:
            raise AssertionError(f"promotable IDs missing: {sorted(selected - found)}")
        count = len(selected)
        joined = "\n".join(lines) + "\n"
        old = "COMPLETED 0 / IMPLEMENTED_UNVERIFIED 103"
        new = f"COMPLETED {count} / IMPLEMENTED_UNVERIFIED {103 - count}"
        if old not in joined:
            raise AssertionError("status summary fixture is stale")
        return joined.replace(old, new, 1)

    def _approved_fixture(
        self,
        root: Path,
        *,
        claim_ids: tuple[int, ...] = (1, 2),
        promoted_ids: tuple[int, ...] | None = None,
        policy_approved: bool = True,
        basis: str = "MODEL_MEASURED",
        components: tuple[str, ...] = ("evidence", "model"),
        artifact_schema: str = "cogni.acceptance.artifact.v1",
        omit_component_result: bool = False,
    ) -> dict[str, Path | str]:
        subject_root = root / "subject"
        bundle_root = root / "detached-acceptance"
        docs = bundle_root / "docs"
        evidence_root = bundle_root / "release" / "evidence"
        config = subject_root / "config"
        docs.mkdir(parents=True)
        evidence_root.mkdir(parents=True)
        config.mkdir(parents=True)

        verifier_id = "independent.lab1"
        key = {
            "schema": "cogni.rsa.public_key.v1",
            "key_id": verifier_id,
            "algorithm": "rsa-pkcs1v15-sha256",
            "modulus_hex": format(TEST_RSA_N, "x"),
            "exponent": 65537,
        }
        key_path = evidence_root / "acceptance-key.json"
        key_data = _encoded(key)
        key_path.write_bytes(key_data)
        policy = {
            "schema": "cogni.release.verifier-policy.v1",
            "status": "approved" if policy_approved else "unconfigured",
            "verifier_id": verifier_id if policy_approved else None,
            "public_key_sha256": sha256(key_data).hexdigest()
            if policy_approved
            else None,
        }
        verifier_policy_data = _encoded(policy)
        (config / "release-verifier-policy.json").write_bytes(verifier_policy_data)

        claims = sorted(f"acceptance.id{item}" for item in claim_ids)
        component_list = sorted(components)
        claim_policy_data = (
            PROJECT_ROOT / "config" / "acceptance-evidence-policy.json"
        ).read_bytes()
        (config / "acceptance-evidence-policy.json").write_bytes(claim_policy_data)
        claim_policy_sha = sha256(claim_policy_data).hexdigest()
        scope = {
            "model_sha256": "1" * 64,
            "code_sha256": "2" * 64,
            "config_sha256": "3" * 64,
            "device_sha256": "4" * 64,
            "policy_sha256": claim_policy_sha,
        }
        artifact = {
            "schema": artifact_schema,
            "status": "passed",
            "basis": basis,
            "components": component_list,
            "component_results": [
                {
                    "component": component,
                    "status": "passed",
                    "evidence_sha256": sha256(component.encode()).hexdigest(),
                }
                for component in (
                    component_list[:-1] if omit_component_result else component_list
                )
            ],
            "claim_ids": claims,
            "claim_results": [
                {
                    "claim_id": claim,
                    "status": "passed",
                    "result_sha256": sha256(claim.encode()).hexdigest(),
                }
                for claim in claims
            ],
            "policy_sha256": claim_policy_sha,
            "scope": scope,
        }
        artifact_path = evidence_root / "acceptance-artifact.json"
        artifact_data = _encoded(artifact)
        artifact_path.write_bytes(artifact_data)
        artifact_sha = sha256(artifact_data).hexdigest()
        raw_payload = {
            "schema": "cogni.acceptance.attestation.v1",
            "status": "passed",
            "basis": basis,
            "verifier_id": verifier_id,
            "source_commit": "a" * 40,
            "source_tree_digest": scope["code_sha256"],
            "model_sha256": scope["model_sha256"],
            "config_sha256": scope["config_sha256"],
            "device_sha256": scope["device_sha256"],
            "claim_ids": claims,
            "artifact_sha256": artifact_sha,
            "components": component_list,
            "policy_sha256": claim_policy_sha,
        }
        payload_path = evidence_root / "acceptance-payload.json"
        payload_data = _encoded(raw_payload)
        payload_path.write_bytes(payload_data)

        metadata = {
            "acceptance_basis": basis,
            "artifact_path": "release/evidence/acceptance-artifact.json",
            "payload_path": "release/evidence/acceptance-payload.json",
            "public_key_path": "release/evidence/acceptance-key.json",
            "signature_path": "release/evidence/acceptance-record.sig",
            "verifier_id": verifier_id,
            "components": ",".join(component_list),
            "verifier_policy_sha256": sha256(verifier_policy_data).hexdigest(),
        }
        kind = {
            "STATIC_ARTIFACT": "acceptance.static",
            "CPU_VERIFIED": "acceptance.cpu",
            "MODEL_MEASURED": "acceptance.model",
            "GPU_MEASURED": "acceptance.gpu",
            "EXTERNAL_VERIFIED": "acceptance.external",
        }[basis]
        identity = {
            "schema_version": 1,
            "recorded_at": "2026-07-18T00:00:00Z",
            "kind": kind,
            "producer": verifier_id,
            "run_id": "release.v041",
            "evidence_class": "verified",
            "claim_ids": claims,
            "artifact_sha256": artifact_sha,
            "payload_sha256": sha256(payload_data).hexdigest(),
            "scope": scope,
            "metadata": metadata,
        }
        record = {
            "record_type": "evidence",
            "evidence_id": "ev1-" + _identity_digest(identity),
            **identity,
        }
        record_path = evidence_root / "acceptance-record.json"
        record_data = _encoded(record)
        record_path.write_bytes(record_data)
        signature_path = evidence_root / "acceptance-record.sig"
        signature_path.write_bytes(_sign(record_data))
        release_attestation = {
            "schema": "cogni.release.attestation.v2",
            "status": "passed",
            "verifier_id": verifier_id,
            "source_commit": raw_payload["source_commit"],
            "summary_sha256": "5" * 64,
            "cpu_evidence_sha256": "6" * 64,
            "gpu5_evidence_sha256": "7" * 64,
            "source_tree_digest": scope["code_sha256"],
            "model_manifest_sha256": "8" * 64,
            "model_tree_digest": scope["model_sha256"],
            "config_digest": scope["config_sha256"],
            "device_digest": scope["device_sha256"],
            "runtime_evidence_sha256": "9" * 64,
            "completion_evidence_sha256": "a" * 64,
            "identity_pre_sha256": "b" * 64,
            "identity_post_sha256": "c" * 64,
            "config_evidence_sha256": "d" * 64,
            "device_evidence_sha256": "e" * 64,
            "model_inventory_sha256": "f" * 64,
            "issued_at_utc": "2026-07-18T00:00:00Z",
        }
        release_attestation_path = evidence_root / "release-attestation.json"
        release_attestation_data = _encoded(release_attestation)
        release_attestation_path.write_bytes(release_attestation_data)
        release_attestation_signature_path = evidence_root / "release-attestation.sig"
        release_attestation_signature_path.write_bytes(_sign(release_attestation_data))
        citation = (
            f"`basis={basis}` "
            "release/evidence/acceptance-record.json#sha256="
            + sha256(record_data).hexdigest()
        )
        checklist = docs / "checklist.md"
        checklist.write_text(
            self._promote(
                CHECKLIST.read_text(encoding="utf-8"),
                promoted_ids or claim_ids,
                citation,
            ),
            encoding="utf-8",
        )
        return {
            "checklist": checklist,
            "record": record_path,
            "signature": signature_path,
            "artifact": artifact_path,
            "payload": payload_path,
            "policy": config / "release-verifier-policy.json",
            "root": subject_root,
            "bundle": bundle_root,
            "verifier_policy_sha256": sha256(verifier_policy_data).hexdigest(),
            "release_attestation": release_attestation_path,
            "release_attestation_signature": release_attestation_signature_path,
            "key": key_path,
        }

    def _validate_fixture(
        self,
        fixture: dict[str, Path | str],
        checklist: Path | None = None,
    ):
        with patch.object(
            checklist_validator,
            "_VERIFIER_POLICY_SHA256",
            fixture["verifier_policy_sha256"],
        ):
            return validate_checklist(
                checklist or fixture["checklist"],
                release_attestation=fixture["release_attestation"],
                release_attestation_signature=fixture["release_attestation_signature"],
                verifier_public_key=fixture["key"],
                _source_root=fixture["root"],
            )

    def test_repository_ledger_is_conservatively_unverified(self) -> None:
        report = validate_checklist(CHECKLIST)
        self.assertEqual(len(report.records), 170)
        self.assertEqual(
            [record.requirement_id for record in report.records], list(range(1, 171))
        )
        self.assertEqual(
            report.counts,
            {
                "COMPLETED": 0,
                "EXTERNAL_BLOCKER": 5,
                "IMPLEMENTED_UNVERIFIED": 103,
                "NOT_IMPLEMENTED": 4,
                "PARTIAL": 58,
            },
        )
        self.assertEqual(report.as_payload()["incomplete_count"], 170)

    def test_tracked_template_cannot_satisfy_publish_complete_gate(self) -> None:
        with self.assertRaisesRegex(
            ChecklistValidationError, "all 170 detached effective rows COMPLETED"
        ):
            validate_checklist(CHECKLIST, require_complete=True)

    def test_invalid_context_is_rejected_even_without_completed_rows(self) -> None:
        with self.assertRaisesRegex(
            ChecklistValidationError, "not independently approved"
        ):
            validate_checklist(
                CHECKLIST,
                release_attestation="attestation.json",
                release_attestation_signature="attestation.sig",
                verifier_public_key="verifier-key.json",
            )

    def test_checked_box_cannot_overclaim_implemented_unverified(self) -> None:
        text = CHECKLIST.read_text(encoding="utf-8")
        corrupted = text.replace("| 1 | [ ] |", "| 1 | [x] |", 1)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "checklist.md"
            path.write_text(corrupted, encoding="utf-8")
            with self.assertRaisesRegex(
                ChecklistValidationError, "checked exactly for COMPLETED"
            ):
                validate_checklist(path)

    def test_declared_summary_must_match_table(self) -> None:
        text = CHECKLIST.read_text(encoding="utf-8")
        corrupted = text.replace(
            "IMPLEMENTED_UNVERIFIED 103", "IMPLEMENTED_UNVERIFIED 102", 1
        )
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "checklist.md"
            path.write_text(corrupted, encoding="utf-8")
            with self.assertRaisesRegex(ChecklistValidationError, "do not match"):
                validate_checklist(path)

    def test_arbitrary_prose_cannot_promote_every_implemented_row(self) -> None:
        lines = CHECKLIST.read_text(encoding="utf-8").splitlines()
        promoted = []
        for line in lines:
            fields = line.split("|")
            if (
                len(fields) == 8
                and fields[1].strip().isdigit()
                and "`IMPLEMENTED_UNVERIFIED`" in line
            ):
                line = line.replace("| [ ] |", "| [x] |", 1).replace(
                    "`IMPLEMENTED_UNVERIFIED`", "`COMPLETED`", 1
                )
                fields = line.split("|")
                fields[5] = " arbitrary prose "
                line = "|".join(fields)
            promoted.append(line)
        text = "\n".join(promoted) + "\n"
        text = text.replace(
            "COMPLETED 0 / IMPLEMENTED_UNVERIFIED 103",
            "COMPLETED 103 / IMPLEMENTED_UNVERIFIED 0",
            1,
        )
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "checklist.md"
            path.write_text(text, encoding="utf-8")
            with self.assertRaisesRegex(ChecklistValidationError, "approved basis"):
                validate_checklist(path)

    def test_one_approved_record_can_cover_multiple_acceptance_ids(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = self._approved_fixture(Path(temporary))
            self.assertFalse(fixture["checklist"].is_relative_to(fixture["root"]))
            self.assertTrue(fixture["checklist"].is_relative_to(fixture["bundle"]))
            report = self._validate_fixture(fixture)
            self.assertEqual(report.counts["COMPLETED"], 2)
            self.assertEqual(report.counts["IMPLEMENTED_UNVERIFIED"], 101)

    def test_public_api_rejects_caller_constructed_context_object(self) -> None:
        with self.assertRaisesRegex(TypeError, "validation_context"):
            validate_checklist(CHECKLIST, validation_context=object())

    def test_six_scalar_trusted_context_cli_is_removed(self) -> None:
        with patch(
            "sys.argv",
            [
                "validate_master_acceptance_checklist.py",
                str(CHECKLIST),
                "--expected-source-commit",
                "a" * 40,
            ],
        ):
            with self.assertRaises(SystemExit) as raised:
                checklist_validator.main()
        self.assertEqual(raised.exception.code, 2)

    def test_checklist_local_verifier_policy_cannot_replace_source_pin(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = self._approved_fixture(Path(temporary))
            with self.assertRaisesRegex(
                ChecklistValidationError, "source-pinned digest"
            ):
                validate_checklist(
                    fixture["checklist"],
                    release_attestation=fixture["release_attestation"],
                    release_attestation_signature=fixture[
                        "release_attestation_signature"
                    ],
                    verifier_public_key=fixture["key"],
                    _source_root=fixture["root"],
                )

    def test_forged_detached_context_signature_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = self._approved_fixture(Path(temporary))
            signature = fixture["release_attestation_signature"]
            signature.write_bytes(b"0" * len(signature.read_bytes()))
            with self.assertRaisesRegex(
                ChecklistValidationError, "signature is invalid"
            ):
                self._validate_fixture(fixture)

    def test_completed_row_requires_signed_detached_release_context(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = self._approved_fixture(Path(temporary))
            with self.assertRaisesRegex(
                ChecklistValidationError, "signed detached release attestation"
            ):
                validate_checklist(fixture["checklist"])

    def test_self_consistent_evidence_cannot_override_actual_context(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = self._approved_fixture(Path(temporary))
            path = fixture["release_attestation"]
            attestation = json.loads(path.read_text(encoding="utf-8"))
            attestation["source_tree_digest"] = "9" * 64
            data = _encoded(attestation)
            path.write_bytes(data)
            fixture["release_attestation_signature"].write_bytes(_sign(data))
            with self.assertRaisesRegex(
                ChecklistValidationError, "trusted current validation context"
            ):
                self._validate_fixture(fixture)

    def test_external_verified_basis_can_promote_lens_claim(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = self._approved_fixture(
                Path(temporary),
                claim_ids=(126,),
                basis="EXTERNAL_VERIFIED",
                components=("evidence", "lens"),
            )
            report = self._validate_fixture(fixture)
            self.assertEqual(report.counts["COMPLETED"], 1)

    def test_static_artifact_cannot_promote_gpu_measured_claim(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = self._approved_fixture(
                Path(temporary),
                claim_ids=(12,),
                basis="STATIC_ARTIFACT",
                components=("cts", "deq", "evidence", "gpu5"),
            )
            with self.assertRaisesRegex(
                ChecklistValidationError,
                "acceptance.id12 cannot be completed with basis STATIC_ARTIFACT",
            ):
                self._validate_fixture(fixture)

    def test_static_artifact_cannot_promote_external_lens_claim(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = self._approved_fixture(
                Path(temporary),
                claim_ids=(126,),
                basis="STATIC_ARTIFACT",
                components=("evidence", "lens"),
            )
            with self.assertRaisesRegex(
                ChecklistValidationError,
                "acceptance.id126 cannot be completed with basis STATIC_ARTIFACT",
            ):
                self._validate_fixture(fixture)

    def test_claim_policy_requires_all_evidence_components(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = self._approved_fixture(
                Path(temporary),
                claim_ids=(1,),
                components=("evidence",),
            )
            with self.assertRaisesRegex(
                ChecklistValidationError,
                "acceptance.id1 is missing required evidence components: model",
            ):
                self._validate_fixture(fixture)

    def test_raw_artifact_schema_is_closed_by_claim_policy(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = self._approved_fixture(
                Path(temporary),
                claim_ids=(1,),
                artifact_schema="unapproved.artifact.v1",
            )
            with self.assertRaisesRegex(
                ChecklistValidationError, "raw artifact is not exact-scope"
            ):
                self._validate_fixture(fixture)

    def test_component_name_without_component_result_cannot_promote(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = self._approved_fixture(
                Path(temporary), claim_ids=(1,), omit_component_result=True
            )
            with self.assertRaisesRegex(
                ChecklistValidationError, "component results are invalid"
            ):
                self._validate_fixture(fixture)

    def test_claim_policy_bytes_are_source_pinned(self) -> None:
        with patch.object(checklist_validator, "_CLAIM_POLICY_SHA256", "0" * 64):
            with self.assertRaisesRegex(
                ChecklistValidationError, "source-pinned digest"
            ):
                validate_checklist(CHECKLIST)

    def test_unconfigured_policy_blocks_otherwise_valid_promotion(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = self._approved_fixture(Path(temporary), policy_approved=False)
            with self.assertRaisesRegex(
                ChecklistValidationError, "not independently approved"
            ):
                self._validate_fixture(fixture)

    def test_missing_claim_id_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = self._approved_fixture(
                Path(temporary), claim_ids=(1, 2), promoted_ids=(1, 3)
            )
            with self.assertRaisesRegex(ChecklistValidationError, "acceptance.id3"):
                self._validate_fixture(fixture)

    def test_canonical_evidence_id_is_recomputed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = self._approved_fixture(Path(temporary))
            record_path = fixture["record"]
            record = json.loads(record_path.read_text(encoding="utf-8"))
            record["evidence_id"] = "ev1-" + "0" * 64
            data = _encoded(record)
            record_path.write_bytes(data)
            checklist_path = fixture["checklist"]
            _refresh_citation(checklist_path, data)
            with self.assertRaisesRegex(
                ChecklistValidationError, "canonical EvidenceRecordV1"
            ):
                self._validate_fixture(fixture, checklist_path)

    def test_invalid_signature_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = self._approved_fixture(Path(temporary))
            valid_signature = fixture["signature"].read_bytes()
            fixture["signature"].write_bytes(b"0" * len(valid_signature))
            with self.assertRaisesRegex(
                ChecklistValidationError, "signature is invalid"
            ):
                self._validate_fixture(fixture)

    def test_unapproved_public_key_digest_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = self._approved_fixture(Path(temporary))
            policy_path = fixture["policy"]
            policy = json.loads(policy_path.read_text(encoding="utf-8"))
            policy["public_key_sha256"] = "0" * 64
            policy_path.write_bytes(_encoded(policy))
            with self.assertRaisesRegex(
                ChecklistValidationError, "source-pinned digest"
            ):
                self._validate_fixture(fixture)

    def test_raw_payload_mutation_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = self._approved_fixture(Path(temporary))
            payload_path = fixture["payload"]
            payload = json.loads(payload_path.read_text(encoding="utf-8"))
            payload["source_commit"] = "b" * 40
            payload_path.write_bytes(_encoded(payload))
            with self.assertRaisesRegex(
                ChecklistValidationError, "raw payload digest mismatch"
            ):
                self._validate_fixture(fixture)

    def test_duplicate_evidence_json_key_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = self._approved_fixture(Path(temporary))
            record_path = fixture["record"]
            duplicated = record_path.read_bytes().replace(
                b"{", b'{"record_type":"evidence",', 1
            )
            record_path.write_bytes(duplicated)
            _refresh_citation(fixture["checklist"], duplicated)
            with self.assertRaisesRegex(
                ChecklistValidationError, "duplicate evidence JSON key"
            ):
                self._validate_fixture(fixture)

    def test_scope_type_confusion_is_rejected_cleanly(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = self._approved_fixture(Path(temporary))
            record_path = fixture["record"]
            record = json.loads(record_path.read_text(encoding="utf-8"))
            record["scope"]["model_sha256"] = 7
            identity = {
                key: value
                for key, value in record.items()
                if key not in {"record_type", "evidence_id"}
            }
            record["evidence_id"] = "ev1-" + _identity_digest(identity)
            data = _encoded(record)
            record_path.write_bytes(data)
            fixture["signature"].write_bytes(_sign(data))
            _refresh_citation(fixture["checklist"], data)
            with self.assertRaisesRegex(ChecklistValidationError, "scope is invalid"):
                self._validate_fixture(fixture)

    def test_non_completed_row_cannot_declare_approved_basis(self) -> None:
        text = CHECKLIST.read_text(encoding="utf-8").replace(
            "| 1 | [ ] | Gemma 4 E4B-it 로컬 백본 | `IMPLEMENTED_UNVERIFIED` |",
            "| 1 | [ ] | Gemma 4 E4B-it 로컬 백본 | `IMPLEMENTED_UNVERIFIED` | `basis=STATIC_ARTIFACT`",
            1,
        )
        with tempfile.TemporaryDirectory() as temporary:
            checklist = Path(temporary) / "checklist.md"
            checklist.write_text(text, encoding="utf-8")
            with self.assertRaisesRegex(ChecklistValidationError, "only COMPLETED"):
                validate_checklist(checklist)

    def test_missing_raw_artifact_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = self._approved_fixture(Path(temporary))
            fixture["artifact"].unlink()
            with self.assertRaisesRegex(
                ChecklistValidationError, "cannot resolve acceptance artifact_path"
            ):
                self._validate_fixture(fixture)

    def test_traversal_citation_is_rejected_before_read(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = self._approved_fixture(Path(temporary))
            checklist_path = fixture["checklist"]
            text = checklist_path.read_text(encoding="utf-8").replace(
                "release/evidence/acceptance-record.json#sha256=",
                "release/evidence/../acceptance-record.json#sha256=",
            )
            checklist_path.write_text(text, encoding="utf-8")
            with self.assertRaisesRegex(ChecklistValidationError, "unsafe segment"):
                self._validate_fixture(fixture, checklist_path)

    def test_linked_raw_artifact_is_rejected_when_supported(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = self._approved_fixture(Path(temporary))
            artifact = fixture["artifact"]
            target = artifact.with_name("real-artifact")
            artifact.replace(target)
            try:
                artifact.symlink_to(target.name)
            except OSError:
                self.skipTest("file symlinks are unavailable on this host")
            with self.assertRaisesRegex(ChecklistValidationError, "link/reparse"):
                self._validate_fixture(fixture)

    def test_same_length_path_swap_is_blocked_or_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            evidence_root = root / "release" / "evidence"
            evidence_root.mkdir(parents=True)
            admitted = evidence_root / "record.json"
            replacement = evidence_root / "replacement.json"
            original_data = b'{"value":"aaaaaaaa"}'
            replacement_data = b'{"value":"bbbbbbbb"}'
            self.assertEqual(len(original_data), len(replacement_data))
            admitted.write_bytes(original_data)
            replacement.write_bytes(replacement_data)
            original_read = checklist_validator.os.read
            attempted = False
            blocked = False

            def swapping_read(descriptor: int, size: int) -> bytes:
                nonlocal attempted, blocked
                if not attempted:
                    attempted = True
                    try:
                        replacement.replace(admitted)
                    except OSError:
                        blocked = True
                return original_read(descriptor, size)

            with patch.object(
                checklist_validator.os, "read", side_effect=swapping_read
            ):
                if checklist_validator.os.name == "nt":
                    data = checklist_validator._read_repository_file(
                        root=root,
                        relative="release/evidence/record.json",
                        maximum=1024,
                        label="swap fixture",
                        allowed_prefixes=("release/evidence/",),
                    )
                    self.assertTrue(blocked)
                    self.assertEqual(data, original_data)
                else:
                    with self.assertRaisesRegex(
                        ChecklistValidationError, "identity changed"
                    ):
                        checklist_validator._read_repository_file(
                            root=root,
                            relative="release/evidence/record.json",
                            maximum=1024,
                            label="swap fixture",
                            allowed_prefixes=("release/evidence/",),
                        )


if __name__ == "__main__":
    unittest.main()
