from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from cogni_agent.tools import (
    ToolPolicyError,
    WorkspaceToolExecutor,
    parse_tool_request,
)


class TestToolRequestParsing(unittest.TestCase):
    def test_explicit_commands_and_aliases_are_typed(self) -> None:
        self.assertEqual(parse_tool_request("프로젝트 상태").operation, "status")
        search = parse_tool_request("/search tensor --in src")
        self.assertEqual(
            (search.operation, search.argument, search.scope),
            ("search", "tensor", "src"),
        )
        save = parse_tool_request("/save result.md\nhello")
        self.assertEqual((save.argument, save.content), ("result.md", "hello"))
        self.assertIsNone(parse_tool_request("일반 대화입니다"))

    def test_unknown_or_malformed_commands_fail_closed(self) -> None:
        with self.assertRaises(ToolPolicyError):
            parse_tool_request("/shell whoami")
        with self.assertRaises(ToolPolicyError):
            parse_tool_request("/read")
        with self.assertRaises(ToolPolicyError):
            parse_tool_request("/save ../x.md\nunsafe")


class TestWorkspaceToolExecutor(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        (self.root / "src").mkdir()
        (self.root / "src" / "demo.py").write_text(
            "VALUE = 'tensor'\n", encoding="utf-8"
        )
        self.executor = WorkspaceToolExecutor(self.root, timeout_seconds=5)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_list_read_search_and_save_stay_bounded(self) -> None:
        listed = self.executor.execute(parse_tool_request("/list src"))
        self.assertTrue(listed.ok)
        self.assertIn("demo.py", listed.output)

        read = self.executor.execute(parse_tool_request("/read src/demo.py"))
        self.assertEqual(read.output.strip(), "VALUE = 'tensor'")

        search = self.executor.execute(parse_tool_request("/search tensor --in src"))
        self.assertIn("src/demo.py:1", search.output)

        saved = self.executor.execute(parse_tool_request("/save evidence.md\nverified"))
        self.assertTrue(saved.ok)
        self.assertEqual(saved.artifact, "outputs/agent-workspace/evidence.md")
        self.assertEqual(
            (self.root / saved.artifact).read_text(encoding="utf-8"), "verified"
        )

    def test_traversal_binary_and_source_write_are_rejected(self) -> None:
        escaped = self.executor.execute(parse_tool_request("/read ../secret.txt"))
        self.assertFalse(escaped.ok)
        (self.root / "binary.bin").write_bytes(b"x\x00y")
        binary = self.executor.execute(parse_tool_request("/read binary.bin"))
        self.assertFalse(binary.ok)
        with self.assertRaises(ToolPolicyError):
            parse_tool_request("/save source.py\nprint('no')")

    def test_test_command_rejects_flags_and_non_test_paths(self) -> None:
        for request in (
            "/test -k name",
            "/test src/demo.py",
            "/test ../tests/test_x.py",
        ):
            with self.subTest(request=request):
                result = self.executor.execute(parse_tool_request(request))
                self.assertFalse(result.ok)

    def test_symlink_inputs_and_output_roots_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as outside_raw:
            outside = Path(outside_raw)
            secret = outside / "secret.txt"
            secret.write_text("must stay private", encoding="utf-8")
            link = self.root / "linked.txt"
            try:
                link.symlink_to(secret)
            except OSError as exc:
                self.skipTest(f"host cannot create test symlinks: {exc}")

            read = self.executor.execute(parse_tool_request("/read linked.txt"))
            self.assertFalse(read.ok)

            outputs = self.root / "outputs"
            outputs.symlink_to(outside, target_is_directory=True)
            saved = self.executor.execute(
                parse_tool_request("/save result.md\nblocked")
            )
            self.assertFalse(saved.ok)
            self.assertFalse((outside / "agent-workspace" / "result.md").exists())


if __name__ == "__main__":
    unittest.main()
