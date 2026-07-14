import tempfile
import unittest
from hashlib import sha256
from pathlib import Path

from cogni_os.artifacts import (
    ArtifactVerificationError,
    verify_artifact_manifest,
    verify_closed_world_artifact_layout,
)


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

    def test_declared_model_identity_is_parsed_and_validated(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "model"
            root.mkdir()
            artifact = root / "weights.bin"
            artifact.write_bytes(b"instruction tuned")
            digest = sha256(artifact.read_bytes()).hexdigest()
            manifest = Path(tmp) / "manifest.toml"
            manifest.write_text(
                "\n".join(
                    (
                        "[model]",
                        'family = "gemma4"',
                        'variant = "E4B"',
                        'role = "instruction_tuned"',
                        'source = "google/gemma-4-E4B-it"',
                        'revision = "pinned"',
                        "",
                        "[files]",
                        f'"weights.bin" = "{digest}"',
                    )
                ),
                encoding="utf-8",
            )
            result = verify_artifact_manifest(root, manifest)
            self.assertIsNotNone(result.identity)
            self.assertEqual(
                result.digests,
                (("weights.bin", digest),),
            )
            self.assertEqual(result.identity.role, "instruction_tuned")
            self.assertEqual(result.identity.source, "google/gemma-4-E4B-it")

            manifest.write_text(
                manifest.read_text(encoding="utf-8").replace(
                    'role = "instruction_tuned"', 'role = "chat-ish"'
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                ArtifactVerificationError, "model role is unsupported"
            ):
                verify_artifact_manifest(root, manifest)

    def test_closed_world_layout_allows_only_declared_benign_snapshot_entries(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "model"
            root.mkdir()
            artifact = root / "weights.bin"
            artifact.write_bytes(b"offline")
            (root / "README.md").write_text("model card", encoding="utf-8")
            (root / ".cache").mkdir()
            (root / ".cache" / "download.metadata").write_text(
                "cache metadata", encoding="utf-8"
            )
            digest = sha256(artifact.read_bytes()).hexdigest()
            manifest = Path(tmp) / "manifest.toml"
            manifest.write_text(
                f'[files]\n"weights.bin" = "{digest}"\n', encoding="utf-8"
            )

            verified = verify_artifact_manifest(root, manifest)
            result = verify_closed_world_artifact_layout(
                verified,
                allowed_unmanifested_files=("README.md",),
                allowed_unmanifested_directories=(".cache",),
            )
            self.assertIs(result, verified)

    def test_closed_world_layout_rejects_loader_overlays_and_remote_code(self):
        loader_relevant_files = (
            "adapter_config.json",
            "adapter_model.safetensors",
            "adapter_model.bin",
            "added_tokens.json",
            "special_tokens_map.json",
            "modeling_gemma4.py",
            "configuration_gemma4.py",
            "processing_gemma4.py",
        )
        for unexpected_name in loader_relevant_files:
            with self.subTest(unexpected_name=unexpected_name):
                with tempfile.TemporaryDirectory() as tmp:
                    root = Path(tmp) / "model"
                    root.mkdir()
                    artifact = root / "weights.bin"
                    artifact.write_bytes(b"offline")
                    (root / unexpected_name).write_bytes(b"untrusted overlay")
                    digest = sha256(artifact.read_bytes()).hexdigest()
                    manifest = Path(tmp) / "manifest.toml"
                    manifest.write_text(
                        f'[files]\n"weights.bin" = "{digest}"\n',
                        encoding="utf-8",
                    )
                    verified = verify_artifact_manifest(root, manifest)

                    with self.assertRaisesRegex(
                        ArtifactVerificationError, unexpected_name.replace(".", r"\.")
                    ):
                        verify_closed_world_artifact_layout(
                            verified,
                            allowed_unmanifested_files=("README.md",),
                            allowed_unmanifested_directories=(".cache",),
                        )

    def test_closed_world_layout_rejects_unexpected_directories_and_wrong_kinds(self):
        cases = (
            ("unexpected_package", "directory"),
            ("README.md", "directory"),
            (".cache", "file"),
        )
        for unexpected_name, kind in cases:
            with self.subTest(unexpected_name=unexpected_name, kind=kind):
                with tempfile.TemporaryDirectory() as tmp:
                    root = Path(tmp) / "model"
                    root.mkdir()
                    artifact = root / "weights.bin"
                    artifact.write_bytes(b"offline")
                    unexpected = root / unexpected_name
                    if kind == "directory":
                        unexpected.mkdir()
                    else:
                        unexpected.write_bytes(b"not a directory")
                    digest = sha256(artifact.read_bytes()).hexdigest()
                    manifest = Path(tmp) / "manifest.toml"
                    manifest.write_text(
                        f'[files]\n"weights.bin" = "{digest}"\n',
                        encoding="utf-8",
                    )
                    verified = verify_artifact_manifest(root, manifest)

                    with self.assertRaisesRegex(
                        ArtifactVerificationError, unexpected_name.replace(".", r"\.")
                    ):
                        verify_closed_world_artifact_layout(
                            verified,
                            allowed_unmanifested_files=("README.md",),
                            allowed_unmanifested_directories=(".cache",),
                        )


if __name__ == "__main__":
    unittest.main()
