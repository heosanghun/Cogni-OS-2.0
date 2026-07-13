from __future__ import annotations

from hashlib import sha256
from pathlib import Path
import subprocess
import tempfile
import tomllib
import unittest
from zipfile import ZipFile

from cogni_core.cts_policy import DEFAULT_CHECKPOINT_SHA256


ROOT = Path(__file__).resolve().parents[1]


class TestReleaseBundleIntegrity(unittest.TestCase):
    def test_product_smoke_is_bound_to_the_pinned_instruction_checkpoint(self) -> None:
        script = (ROOT / "scripts" / "build_release_bundle.ps1").read_text(
            encoding="utf-8"
        )
        self.assertIn(
            "[string]$ModelPath = 'C:\\Project\\cognios\\gemma4-e4b-it'",
            script,
        )
        self.assertIn("--manifest config\\gemma4-e4b-it.manifest.toml", script)
        self.assertIn("--max-new-tokens 96", script)
        self.assertNotIn("--prompt", script)

        manifest_path = ROOT / "config" / "gemma4-e4b-it.manifest.toml"
        with manifest_path.open("rb") as stream:
            manifest = tomllib.load(stream)
        self.assertEqual(
            manifest["model"],
            {
                "family": "gemma4",
                "variant": "E4B",
                "role": "instruction_tuned",
                "source": "google/gemma-4-E4B-it",
                "revision": "a4c2d58be94dda072b918d9db64ee85c8ed34e3f",
            },
        )
        self.assertEqual(
            set(manifest["files"]),
            {
                "chat_template.jinja",
                "config.json",
                "generation_config.json",
                "model.safetensors",
                "processor_config.json",
                "tokenizer.json",
                "tokenizer_config.json",
            },
        )

    def test_checkpoint_is_exported_as_an_opaque_git_artifact(self) -> None:
        attributes = (ROOT / ".gitattributes").read_text(encoding="utf-8")
        self.assertIn("/cogni_core/cts_policy_checkpoint.json -text", attributes)

    def test_archive_with_autocrlf_disabled_preserves_checkpoint_bytes(self) -> None:
        if not (ROOT / ".git").is_dir():
            self.skipTest("exact git archive reproduction requires a Git checkout")
        with tempfile.TemporaryDirectory() as temporary:
            archive = Path(temporary) / "source.zip"
            subprocess.run(
                [
                    "git",
                    "-C",
                    str(ROOT),
                    "-c",
                    "core.autocrlf=false",
                    "archive",
                    "--format=zip",
                    "--prefix=source/",
                    f"--output={archive}",
                    "HEAD",
                ],
                check=True,
                capture_output=True,
            )
            with ZipFile(archive) as bundle:
                payload = bundle.read("source/cogni_core/cts_policy_checkpoint.json")

        self.assertEqual(sha256(payload).hexdigest(), DEFAULT_CHECKPOINT_SHA256)

    def test_release_script_pins_commit_and_publishes_atomically(self) -> None:
        script = (ROOT / "scripts" / "build_release_bundle.ps1").read_text(
            encoding="utf-8"
        )
        for contract in (
            "rev-parse --verify --end-of-options",
            "core.autocrlf=false",
            "Source archive changed the CTS policy checkpoint bytes",
            "Release output already exists; refusing to merge",
            ".cogni-release-staging-",
            "Move-Item -LiteralPath $publishStage -Destination $publishedOutput",
            "SOURCE_DATE_EPOCH",
            "commit_oid=$commitOid",
        ):
            with self.subTest(contract=contract):
                self.assertIn(contract, script)


if __name__ == "__main__":
    unittest.main()
