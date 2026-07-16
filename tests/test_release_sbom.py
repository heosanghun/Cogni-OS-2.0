from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import json
import os
import unittest
from unittest.mock import patch

from scripts.generate_release_sbom import generate, main


class TestReleaseSbom(unittest.TestCase):
    def test_generate_is_bounded_and_records_unsigned_status(self) -> None:
        root = Path(__file__).resolve().parents[1]
        with TemporaryDirectory() as temporary:
            artifact = Path(temporary) / "CogniBoard.exe"
            artifact.write_bytes(b"bounded-launcher")
            with patch.dict(os.environ, {"SOURCE_DATE_EPOCH": "1700000000"}):
                sbom, notices = generate(
                    pyproject=root / "pyproject.toml",
                    project_version="0.4.0",
                    artifacts=[artifact],
                )
            self.assertEqual(sbom["bomFormat"], "CycloneDX")
            self.assertEqual(sbom["specVersion"], "1.5")
            self.assertEqual(sbom["metadata"]["timestamp"], "2023-11-14T22:13:20Z")
            properties = {
                row["name"]: row["value"] for row in sbom["metadata"]["properties"]
            }
            self.assertEqual(
                properties["cogni:signature_status"],
                "unsigned-no-code-signing-certificate-provided",
            )
            files = [row for row in sbom["components"] if row["type"] == "file"]
            self.assertEqual(files[0]["name"], "CogniBoard.exe")
            self.assertIn("torch", notices.casefold())

    def test_cli_writes_json_and_notices(self) -> None:
        root = Path(__file__).resolve().parents[1]
        with TemporaryDirectory() as temporary:
            directory = Path(temporary)
            artifact = directory / "source.zip"
            artifact.write_bytes(b"source")
            output = directory / "SBOM.cdx.json"
            notices = directory / "THIRD_PARTY_NOTICES.md"
            result = main(
                [
                    "--pyproject",
                    str(root / "pyproject.toml"),
                    "--project-version",
                    "0.4.0",
                    "--output",
                    str(output),
                    "--notices",
                    str(notices),
                    "--artifact",
                    str(artifact),
                ]
            )
            self.assertEqual(result, 0)
            self.assertEqual(
                json.loads(output.read_text(encoding="utf-8"))["version"], 1
            )
            self.assertIn("third-party", notices.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
