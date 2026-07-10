from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


class TestValidationScriptBootstrap(unittest.TestCase):
    def test_readme_direct_commands_import_checkout_from_any_working_directory(self):
        root = Path(__file__).resolve().parents[1]
        scripts = (
            "validate_gemma4.py",
            "validate_gemma4_deq.py",
            "validate_gemma4_runtime.py",
        )
        environment = os.environ.copy()
        environment.pop("PYTHONPATH", None)
        with tempfile.TemporaryDirectory() as other_directory:
            for name in scripts:
                with self.subTest(script=name):
                    completed = subprocess.run(
                        [
                            sys.executable,
                            "-I",
                            str(root / "scripts" / name),
                            "--help",
                        ],
                        cwd=other_directory,
                        env=environment,
                        capture_output=True,
                        text=True,
                        timeout=30,
                        check=False,
                    )
                    self.assertEqual(
                        completed.returncode,
                        0,
                        msg=completed.stdout + completed.stderr,
                    )
                    self.assertIn("usage:", completed.stdout.lower())


if __name__ == "__main__":
    unittest.main()
