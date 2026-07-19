import tempfile
from pathlib import Path
import unittest

from cogni_flow.snapshot import SafeProjectSnapshotBuilder, SnapshotBoundaryError


class TestSafeProjectSnapshotBuilder(unittest.TestCase):
    def test_snapshot_copies_only_admitted_regular_project_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            source = base / "source"
            (source / "cogni_flow").mkdir(parents=True)
            (source / "tests").mkdir()
            (source / ".cogni_state").mkdir()
            (source / "models").mkdir()
            (source / "cogni_flow" / "target.py").write_text(
                "VALUE = 1\n", encoding="utf-8"
            )
            (source / "tests" / "test_target.py").write_text(
                "def test_value(): pass\n", encoding="utf-8"
            )
            (source / ".env").write_text("TOKEN=host-secret\n", encoding="utf-8")
            (source / "operator.key").write_bytes(b"private")
            (source / ".cogni_state" / "approval.json").write_text(
                "secret", encoding="utf-8"
            )
            (source / "models" / "backbone.gguf").write_bytes(b"weights")

            destination = base / "stage"
            evidence = SafeProjectSnapshotBuilder(source).copy_to(destination)

            self.assertEqual(evidence.files, 2)
            self.assertEqual(len(evidence.tree_sha256), 64)
            self.assertTrue((destination / "cogni_flow" / "target.py").is_file())
            self.assertTrue((destination / "tests" / "test_target.py").is_file())
            self.assertFalse((destination / ".env").exists())
            self.assertFalse((destination / "operator.key").exists())
            self.assertFalse((destination / ".cogni_state").exists())
            self.assertFalse((destination / "models").exists())

    def test_snapshot_rejects_link_before_external_secret_can_leak(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            source = base / "source"
            package = source / "cogni_flow"
            package.mkdir(parents=True)
            (package / "target.py").write_text("VALUE = 1\n", encoding="utf-8")
            outside = base / "outside-secret.txt"
            outside.write_text("host-secret", encoding="utf-8")
            leak = package / "leak.py"
            try:
                leak.symlink_to(outside)
            except OSError as exc:
                self.skipTest(f"link creation is unavailable: {exc}")

            destination = base / "stage"
            with self.assertRaisesRegex(SnapshotBoundaryError, "link/reparse"):
                SafeProjectSnapshotBuilder(source).copy_to(destination)
            self.assertFalse((destination / "cogni_flow" / "leak.py").exists())
            copied = list(destination.rglob("*")) if destination.exists() else []
            self.assertFalse(
                any(
                    path.is_file() and path.read_bytes() == b"host-secret"
                    for path in copied
                )
            )

    def test_snapshot_fails_closed_at_byte_bound(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            source = base / "source"
            source.mkdir()
            (source / "large.py").write_bytes(b"x" * 33)

            with self.assertRaisesRegex(SnapshotBoundaryError, "per-file bound"):
                SafeProjectSnapshotBuilder(
                    source,
                    max_files=2,
                    max_total_bytes=64,
                    max_file_bytes=32,
                ).copy_to(base / "stage")


if __name__ == "__main__":
    unittest.main()
