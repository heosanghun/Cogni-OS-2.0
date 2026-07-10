import io
import os
from pathlib import Path
import tempfile
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

from cogni_os.cli import main, validate


class TestCLI(unittest.TestCase):
    def test_doctor_reports_offline_policy(self):
        stream = io.StringIO()
        with redirect_stdout(stream):
            code = main(["doctor"])
        self.assertEqual(code, 0)
        self.assertIn("offline=True", stream.getvalue())

    def test_doctor_default_is_independent_of_current_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            previous = Path.cwd()
            try:
                os.chdir(directory)
                stream = io.StringIO()
                with redirect_stdout(stream):
                    code = main(["doctor"])
            finally:
                os.chdir(previous)
        self.assertEqual(code, 0)
        self.assertIn("project=Cogni-OS 2.0", stream.getvalue())

    @patch("cogni_os.cli.subprocess.run")
    def test_source_validation_uses_absolute_package_root(self, run):
        run.return_value.returncode = 0
        with tempfile.TemporaryDirectory() as directory:
            previous = Path.cwd()
            try:
                os.chdir(directory)
                code = validate()
            finally:
                os.chdir(previous)
        self.assertEqual(code, 0)
        command = run.call_args.args[0]
        self.assertTrue(Path(command[command.index("-s") + 1]).is_absolute())
        self.assertTrue(Path(run.call_args.kwargs["cwd"]).is_absolute())


if __name__ == "__main__":
    unittest.main()
