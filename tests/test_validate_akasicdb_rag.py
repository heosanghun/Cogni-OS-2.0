from __future__ import annotations

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from cogni_demo.workspace_capabilities import AKASICDB_AUDITED_DIGESTS
from scripts.validate_akasicdb_rag import validate_akasicdb_rag
from tests.test_workspace_capabilities import _write_clone


class ValidateAkasicDBRagTests(unittest.TestCase):
    def test_smoke_proves_restart_delete_and_honest_lexical_profile(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            clone = Path(temporary) / "AkasicDB"
            clone.mkdir()
            digests = _write_clone(clone)
            with patch.dict(AKASICDB_AUDITED_DIGESTS, digests, clear=True):
                result = validate_akasicdb_rag(clone)

        self.assertEqual(result["status"], "PASS")
        self.assertEqual(result["retrieval_mode"], "lexical_only")
        self.assertFalse(result["semantic_embedding"])
        self.assertTrue(result["answer_integration_configured"])
        self.assertTrue(result["answer_bridge_contract_verified"])
        self.assertEqual(
            result["answer_integration_schema"],
            "cogni.agent.retrieval-evidence.v1",
        )
        self.assertEqual(
            result["integration_schema_authority"],
            "explicit_wiring_contract_not_cryptographic_attestation",
        )
        self.assertEqual(result["generation_backend"], "deterministic_cpu_fixture")
        self.assertFalse(result["actual_model_inference"])
        self.assertFalse(result["production_attestation"])
        self.assertEqual(result["restart_recovery"], "PASS")
        self.assertEqual(result["delete_recovery"], "PASS")
        self.assertEqual(result["source_provenance"], "PASS")
        self.assertEqual(result["selected_prompt_provenance"], "PASS")
        self.assertEqual(result["license_status"], "unverified_no_license_file")
        self.assertFalse(result["redistribution_authorized"])


if __name__ == "__main__":
    unittest.main()
