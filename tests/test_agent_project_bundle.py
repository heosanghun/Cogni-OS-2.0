from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from cogni_agent.tools import (
    HELP_TEXT,
    ToolPolicyError,
    ToolRequest,
    WorkspaceToolExecutor,
    parse_tool_request,
)
from cogni_flow.task_plan import (
    ArtifactBundle,
    ArtifactBundleFile,
    TaskAction,
    TaskActionKind,
    TaskPolicyError,
)


def project_command(
    project: str = "hello-poc",
    files: list[dict[str, str]] | None = None,
) -> str:
    payload = {
        "schema_version": 1,
        "project": project,
        "files": files
        or [
            {"path": "main.py", "content": "print('hello')\n"},
            {"path": "web/index.html", "content": "<h1>Hello</h1>\n"},
            {"path": "config.json", "content": '{"offline":true}\n'},
        ],
    }
    return "/project\n" + json.dumps(payload, ensure_ascii=False)


class TestProjectBundleSchema(unittest.TestCase):
    def test_project_command_is_explicit_typed_and_documented(self) -> None:
        request = parse_tool_request(project_command())
        self.assertEqual(request.operation, "project")
        self.assertIsInstance(request.artifact_bundle, ArtifactBundle)
        self.assertEqual(len(request.artifact_bundle.files), 3)
        self.assertIn("/project", HELP_TEXT)
        self.assertIn("실행되지 않는 산출물", HELP_TEXT)
        sample = next(line for line in HELP_TEXT.splitlines() if line.startswith("{"))
        self.assertIsInstance(
            parse_tool_request("/project\n" + sample).artifact_bundle,
            ArtifactBundle,
        )
        self.assertIsNone(parse_tool_request("PoC 프로젝트를 알아서 만들어줘"))

    def test_schema_rejects_code_and_path_ambiguity_before_authorization(self) -> None:
        invalid_files = (
            [{"path": "../escape.py", "content": "pass\n"}],
            [{"path": "/absolute.py", "content": "pass\n"}],
            [{"path": "C:/drive.py", "content": "pass\n"}],
            [{"path": "CON.py", "content": "pass\n"}],
            [{"path": "nested\\backslash.py", "content": "pass\n"}],
            [{"path": "manifest.json", "content": "{}"}],
            [{"path": "payload.exe", "content": "text"}],
            [{"path": "broken.py", "content": "def invalid(:\n"}],
            [{"path": "broken.json", "content": '{"x":NaN}'}],
            [
                {"path": "Readme.md", "content": "one"},
                {"path": "README.MD", "content": "two"},
            ],
        )
        for files in invalid_files:
            with self.subTest(files=files), self.assertRaises(ToolPolicyError):
                parse_tool_request(project_command(files=list(files)))

        for project in ("../escape", "CON", "white space", "a/b"):
            with self.subTest(project=project), self.assertRaises(ToolPolicyError):
                parse_tool_request(project_command(project=project))

    def test_schema_rejects_extensions_duplicates_and_fixed_bounds(self) -> None:
        extra = {
            "schema_version": 1,
            "project": "demo",
            "files": [{"path": "main.py", "content": "pass\n"}],
            "execute": True,
        }
        with self.assertRaises(ToolPolicyError):
            parse_tool_request("/project\n" + json.dumps(extra))
        with self.assertRaises(ToolPolicyError):
            parse_tool_request(
                '/project\n{"schema_version":1,"project":"demo",'
                '"project":"other","files":[{"path":"a.py","content":"pass"}]}'
            )
        with self.assertRaises(ToolPolicyError):
            parse_tool_request(
                project_command(
                    files=[
                        {"path": f"file-{index}.txt", "content": "x"}
                        for index in range(13)
                    ]
                )
            )
        with self.assertRaises(ToolPolicyError):
            parse_tool_request(
                project_command(
                    files=[{"path": "big.txt", "content": "x" * (256 * 1024 + 1)}]
                )
            )

    def test_untyped_bundle_cannot_cross_the_plan_policy(self) -> None:
        bundle = ArtifactBundle("typed", (ArtifactBundleFile("main.py", "pass\n"),))
        action = TaskAction(
            "bundle",
            TaskActionKind.WRITE_ARTIFACT_BUNDLE,
            path="outputs/agent-workspace/typed",
            artifact_bundle=bundle,
        )
        self.assertIs(action.artifact_bundle, bundle)
        forged = TaskAction(
            "bundle",
            TaskActionKind.WRITE_ARTIFACT_BUNDLE,
            path="outputs/agent-workspace/typed",
            artifact_bundle={"project": "typed"},  # type: ignore[arg-type]
        )
        with tempfile.TemporaryDirectory() as root:
            executor = WorkspaceToolExecutor(root)
            with self.assertRaises(TaskPolicyError):
                executor.task_executor.policy._validate_action(
                    forged,
                    ("outputs/agent-workspace/typed",),
                )


class TestProjectBundleCommit(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.executor = WorkspaceToolExecutor(self.root, timeout_seconds=5)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_bundle_commits_atomically_with_verified_manifest_and_no_execution(
        self,
    ) -> None:
        request = parse_tool_request(project_command())
        with patch("cogni_flow.task_plan.subprocess.Popen") as spawn:
            result = self.executor.execute(request)
        spawn.assert_not_called()
        self.assertTrue(result.ok, result.output)
        self.assertEqual(
            result.artifact,
            "outputs/agent-workspace/hello-poc/manifest.json",
        )
        project = self.root / "outputs" / "agent-workspace" / "hello-poc"
        self.assertEqual(
            (project / "main.py").read_text(encoding="utf-8"), "print('hello')\n"
        )
        self.assertEqual(
            (project / "web" / "index.html").read_text(encoding="utf-8"),
            "<h1>Hello</h1>\n",
        )

        manifest_bytes = (project / "manifest.json").read_bytes()
        manifest = json.loads(manifest_bytes)
        self.assertFalse(manifest["execution_allowed"])
        self.assertFalse(manifest["source_mutation_allowed"])
        self.assertEqual(result.output.count("execution=false"), 1)
        listed = {item["path"]: item for item in manifest["files"]}
        for relative in ("main.py", "web/index.html", "config.json"):
            data = (project / relative).read_bytes()
            self.assertEqual(listed[relative]["sha256"], sha256(data).hexdigest())
            self.assertEqual(listed[relative]["size_bytes"], len(data))
        self.assertFalse(any(self.root.glob("outputs/agent-workspace/.bundle-*.tmp")))

    def test_existing_project_is_never_overwritten(self) -> None:
        first = self.executor.execute(parse_tool_request(project_command()))
        self.assertTrue(first.ok, first.output)
        main = self.root / "outputs" / "agent-workspace" / "hello-poc" / "main.py"
        before = main.read_bytes()
        second = self.executor.execute(
            parse_tool_request(
                project_command(
                    files=[{"path": "main.py", "content": "print('changed')\n"}]
                )
            )
        )
        self.assertFalse(second.ok)
        self.assertIn("overwrite refused", second.output)
        self.assertEqual(main.read_bytes(), before)
        self.assertFalse(any(self.root.glob("outputs/agent-workspace/.bundle-*.tmp")))

    def test_failed_commit_removes_private_staging_without_partial_project(
        self,
    ) -> None:
        with patch.object(
            self.executor.task_executor,
            "_rename_directory_no_replace",
            side_effect=TaskPolicyError("simulated commit refusal"),
        ):
            result = self.executor.execute(parse_tool_request(project_command()))
        self.assertFalse(result.ok)
        workspace = self.root / "outputs" / "agent-workspace"
        self.assertFalse((workspace / "hello-poc").exists())
        self.assertFalse(any(workspace.glob(".bundle-*.tmp")))

    def test_concurrent_same_project_has_exactly_one_winner(self) -> None:
        request = parse_tool_request(project_command())
        executors = [
            WorkspaceToolExecutor(self.root, timeout_seconds=5) for _ in range(2)
        ]
        with ThreadPoolExecutor(max_workers=2) as pool:
            results = list(
                pool.map(lambda executor: executor.execute(request), executors)
            )
        self.assertEqual(sum(result.ok for result in results), 1)
        self.assertEqual(sum(not result.ok for result in results), 1)
        project = self.root / "outputs" / "agent-workspace" / "hello-poc"
        self.assertTrue((project / "manifest.json").is_file())
        self.assertFalse(any(project.parent.glob(".bundle-*.tmp")))

    def test_linklike_project_target_is_not_followed(self) -> None:
        workspace = self.root / "outputs" / "agent-workspace"
        workspace.mkdir(parents=True)
        with tempfile.TemporaryDirectory() as outside_raw:
            outside = Path(outside_raw)
            target = workspace / "hello-poc"
            try:
                target.symlink_to(outside, target_is_directory=True)
            except OSError as exc:
                self.skipTest(f"host cannot create test symlinks: {exc}")
            result = self.executor.execute(parse_tool_request(project_command()))
            self.assertFalse(result.ok)
            self.assertIn("overwrite refused", result.output)
            self.assertEqual(list(outside.iterdir()), [])

    def test_forged_request_without_typed_payload_fails_closed(self) -> None:
        result = self.executor.execute(ToolRequest("project"))
        self.assertFalse(result.ok)
        self.assertIn("typed artifact bundle", result.output)


if __name__ == "__main__":
    unittest.main()
