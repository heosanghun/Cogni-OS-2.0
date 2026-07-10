import tempfile
import unittest
from hashlib import sha256
from pathlib import Path

from cogni_os.artifacts import ArtifactVerificationError, verify_artifact_manifest


class TestArtifactManifest(unittest.TestCase):
    def test_verified_local_artifact_set(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "model"
            root.mkdir()
            artifact = root / "weights.bin"
            artifact.write_bytes(b"offline")
            digest = sha256(b"offline").hexdigest()
            manifest = Path(tmp) / "manifest.toml"
            manifest.write_text(
                f'[files]\n"weights.bin" = "{digest}"\n', encoding="utf-8"
            )
            result = verify_artifact_manifest(root, manifest)
            self.assertEqual(result.files, (artifact.resolve(),))

    def test_digest_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "model"
            root.mkdir()
            (root / "weights.bin").write_bytes(b"offline")
            manifest = Path(tmp) / "manifest.toml"
            manifest.write_text(
                f'[files]\n"weights.bin" = "{"0" * 64}"\n', encoding="utf-8"
            )
            with self.assertRaises(ArtifactVerificationError):
                verify_artifact_manifest(root, manifest)


if __name__ == "__main__":
    unittest.main()
