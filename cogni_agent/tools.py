"""Bounded local tools exposed to the Cogni-OS product agent.

The model never receives a shell.  A user must select task mode and the
request must resolve to one of the typed operations in this module.  Source
files are read-only here; autonomous source changes belong exclusively to the
Self-Harness staging and promotion pipeline.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from hashlib import sha256
import os
from pathlib import Path
import re
from secrets import token_hex
import shutil
import subprocess
import sys
from time import monotonic

from cogni_flow.task_plan import (
    ArtifactBundle,
    CapabilityToken,
    ExpectedArtifact,
    RiskTier,
    TaskAction,
    TaskActionKind,
    TaskBudget,
    TaskExecutionError,
    TaskPlanError,
    TaskPlanExecutor,
    TaskPlanResult,
    TaskVerifier,
    TypedTaskPlan,
    VerifierKind,
    artifact_bundle_manifest_bytes,
    artifact_bundle_manifest_path,
    artifact_bundle_root,
    parse_artifact_bundle_payload,
)


MAX_REQUEST_CHARS = 4_096
MAX_PROJECT_COMMAND_BYTES = 1 * 1024 * 1024
MAX_RESULT_CHARS = 40_000
MAX_READ_BYTES = 128 * 1024
MAX_SAVE_BYTES = 64 * 1024
MAX_LIST_ENTRIES = 200
MAX_SEARCH_MATCHES = 100
MAX_SEARCH_FILES = 2_000
ALLOWED_SAVE_SUFFIXES = {".txt", ".md", ".json", ".csv"}
IGNORED_DIRECTORIES = {
    ".git",
    ".pytest_cache",
    ".ruff_cache",
    "__pycache__",
    "build",
    "dist",
    "work",
}


class ToolPolicyError(ValueError):
    """Raised before a request can cross a local tool boundary."""


@dataclass(frozen=True)
class ToolRequest:
    operation: str
    argument: str = ""
    scope: str = "."
    content: str = ""
    artifact_bundle: ArtifactBundle | None = None


@dataclass(frozen=True)
class ToolResult:
    operation: str
    ok: bool
    output: str
    duration_seconds: float
    artifact: str | None = None


HELP_TEXT = """안전한 로컬 작업 명령
/list [상대경로]                 파일 목록
/read <상대파일>                 bounded 파일 읽기
/search <검색어> [--in <경로>]   프로젝트 텍스트 검색
/status                          Git 변경 상태 읽기
/test [tests/파일.py]            신뢰 개발자 opt-in에서만 고정 pytest 실행
/save <파일명> 다음 줄부터 내용   outputs/agent-workspace에 결과 저장
/project 다음 줄에 JSON           최대 12파일/256KiB PoC·MVP 묶음 저장

/project JSON 스키마:
{"schema_version":1,"project":"demo","files":[{"path":"main.py","content":"print('hello')\\n"}]}

동일한 작업을 “프로젝트 상태를 보여주세요”, “docs 폴더의 파일 목록을
보여주세요”, “README.md 파일을 읽어주세요”, “전체 테스트를 실행해주세요”처럼
명확한 자연어로 요청할 수도 있습니다. 모호한 자연어는 실행하지 않고 도움말을
표시합니다. 임의 셸 명령과 소스 직접 쓰기는 허용되지 않습니다. 코드 변경은
Self-Harness의 제안 전용 T2 staging을 거쳐야 합니다. /project 결과는
outputs/agent-workspace/<project> 아래의 실행되지 않는 산출물이며, 기존 프로젝트를
덮어쓰지 않습니다."""


_SAFE_NATURAL_PATH = r"[A-Za-z0-9_.\-/]+"


def _parse_natural_task(text: str) -> ToolRequest | None:
    """Compile a deliberately small, deterministic natural-language grammar.

    This is not a model planner.  A full-string match must identify exactly one
    allowlisted operation and every path remains subject to the typed-plan path
    verifier.  Ambiguous prose returns ``None`` without issuing capability.
    """

    exact = {
        "프로젝트 상태를 보여줘": ToolRequest("status"),
        "프로젝트 상태를 보여주세요": ToolRequest("status"),
        "변경 상태를 보여줘": ToolRequest("status"),
        "변경 상태를 보여주세요": ToolRequest("status"),
        "전체 테스트를 실행해줘": ToolRequest("test"),
        "전체 테스트를 실행해주세요": ToolRequest("test"),
        "프로젝트 파일 목록을 보여줘": ToolRequest("list", "."),
        "프로젝트 파일 목록을 보여주세요": ToolRequest("list", "."),
        "show project status": ToolRequest("status"),
        "run all tests": ToolRequest("test"),
        "list project files": ToolRequest("list", "."),
    }
    normalized = " ".join(text.split())
    direct = exact.get(normalized.casefold())
    if direct is not None:
        return direct

    patterns: tuple[tuple[str, str], ...] = (
        (
            "list",
            rf"(?P<path>{_SAFE_NATURAL_PATH})\s*(?:폴더|디렉터리)(?:의)?\s*"
            r"파일\s*목록(?:을)?\s*(?:보여\s*줘|보여주세요|조회해줘|조회해주세요)",
        ),
        (
            "read",
            rf"(?P<path>{_SAFE_NATURAL_PATH})\s*파일(?:의)?\s*(?:내용(?:을)?\s*)?"
            r"(?:읽어\s*줘|읽어주세요|보여\s*줘|보여주세요)",
        ),
        (
            "test",
            r"(?P<path>tests/[A-Za-z0-9_.\-/]+\.py)\s*(?:파일(?:의)?\s*)?"
            r"테스트(?:를)?\s*(?:실행해줘|실행해주세요|돌려줘|돌려주세요)",
        ),
        (
            "list_en",
            rf"(?:list|show)(?: the)? files in (?P<path>{_SAFE_NATURAL_PATH})",
        ),
        (
            "read_en",
            rf"(?:read|show)(?: the)? file (?P<path>{_SAFE_NATURAL_PATH})",
        ),
        (
            "test_en",
            r"run(?: the)? test (?P<path>tests/[A-Za-z0-9_.\-/]+\.py)",
        ),
    )
    for operation, pattern in patterns:
        match = re.fullmatch(pattern, normalized, flags=re.IGNORECASE)
        if match is None:
            continue
        mapped = operation.removesuffix("_en")
        return ToolRequest(mapped, match.group("path"))

    korean_search = re.fullmatch(
        rf"(?:(?P<scope>{_SAFE_NATURAL_PATH})\s*(?:에서|안에서)\s*)?"
        r"[\"“](?P<query>[^\"”\r\n]{1,256})[\"”]\s*"
        r"(?:검색해줘|검색해주세요|찾아줘|찾아주세요)",
        normalized,
    )
    if korean_search is not None:
        return ToolRequest(
            "search",
            korean_search.group("query").strip(),
            korean_search.group("scope") or ".",
        )
    english_search = re.fullmatch(
        rf"(?:search|find) [\"](?P<query>[^\"\r\n]{{1,256}})[\"]"
        rf"(?: in (?P<scope>{_SAFE_NATURAL_PATH}))?",
        normalized,
        flags=re.IGNORECASE,
    )
    if english_search is not None:
        return ToolRequest(
            "search",
            english_search.group("query").strip(),
            english_search.group("scope") or ".",
        )
    return None


def parse_tool_request(message: str) -> ToolRequest | None:
    """Parse an explicit task-mode command into a typed request."""

    if not isinstance(message, str):
        raise TypeError("tool request must be text")
    text = message.strip()
    if not text:
        raise ToolPolicyError("empty task request")
    first_line = text.partition("\n")[0]
    is_project_command = first_line.split(maxsplit=1)[0].lower() == "/project"
    if is_project_command:
        if len(text.encode("utf-8")) > MAX_PROJECT_COMMAND_BYTES:
            raise ToolPolicyError("/project request exceeds its bounded input limit")
    elif len(text) > MAX_REQUEST_CHARS:
        raise ToolPolicyError("task request exceeds the bounded input limit")

    aliases = {
        "도움말": "/help",
        "작업 도움말": "/help",
        "프로젝트 상태": "/status",
        "파일 목록": "/list",
        "전체 테스트 실행": "/test",
    }
    text = aliases.get(text, text)
    if not text.startswith("/"):
        return _parse_natural_task(text)

    first_line, separator, remainder = text.partition("\n")
    command, _, raw_argument = first_line.partition(" ")
    operation = command[1:].strip().lower()
    argument = raw_argument.strip()

    if operation in {"help", "status"}:
        if argument or separator:
            raise ToolPolicyError(f"/{operation} does not accept arguments")
        return ToolRequest(operation)
    if operation == "list":
        return ToolRequest(operation, argument or ".")
    if operation == "read":
        if not argument or separator:
            raise ToolPolicyError("/read requires exactly one file path")
        return ToolRequest(operation, argument)
    if operation == "search":
        if separator:
            raise ToolPolicyError("/search must be a single line")
        query, marker, scope = argument.partition(" --in ")
        if not query.strip():
            raise ToolPolicyError("/search requires a search term")
        return ToolRequest(operation, query.strip(), (scope.strip() if marker else "."))
    if operation == "test":
        if separator:
            raise ToolPolicyError("/test must be a single line")
        return ToolRequest(operation, argument)
    if operation == "save":
        if not argument or not separator or not remainder:
            raise ToolPolicyError(
                "/save requires a filename and content on following lines"
            )
        name = Path(argument)
        if name.name != argument or name.suffix.lower() not in ALLOWED_SAVE_SUFFIXES:
            raise ToolPolicyError("/save filename must use .txt, .md, .json, or .csv")
        return ToolRequest(operation, argument, content=remainder)
    if operation == "project":
        if argument or not separator or not remainder.strip():
            raise ToolPolicyError(
                "/project requires one JSON payload on following lines"
            )
        try:
            bundle = parse_artifact_bundle_payload(remainder)
        except TaskPlanError as exc:
            raise ToolPolicyError(str(exc)) from exc
        return ToolRequest(operation, artifact_bundle=bundle)
    raise ToolPolicyError(f"unsupported task command: /{operation}")


class WorkspaceToolExecutor:
    """Execute a fixed local-tool allowlist inside one resolved project root."""

    def __init__(
        self,
        project_root: str | Path,
        *,
        timeout_seconds: int = 180,
        allow_trusted_tests: bool = False,
    ) -> None:
        root = Path(project_root).resolve(strict=True)
        if not root.is_dir():
            raise ValueError("project_root must be a directory")
        if not 1 <= timeout_seconds <= 900:
            raise ValueError("timeout_seconds must be in [1, 900]")
        self.project_root = root
        self.output_root = root / "outputs" / "agent-workspace"
        self.timeout_seconds = int(timeout_seconds)
        self.allow_trusted_tests = bool(allow_trusted_tests)
        self.task_executor = TaskPlanExecutor(
            root,
            help_text=HELP_TEXT,
        )

    def _resolve(self, raw: str, *, require_file: bool = False) -> Path:
        candidate = Path(raw or ".")
        if candidate.is_absolute() or ".." in candidate.parts:
            raise ToolPolicyError("absolute paths and parent traversal are forbidden")
        unresolved = self.project_root / candidate
        resolved = unresolved.resolve(strict=True)
        if resolved != self.project_root and not resolved.is_relative_to(
            self.project_root
        ):
            raise ToolPolicyError("path escaped the project root")
        if unresolved.is_symlink() or resolved != unresolved.absolute():
            raise ToolPolicyError("symbolic-link or junction targets are forbidden")
        if require_file and not resolved.is_file():
            raise ToolPolicyError("requested path is not a regular file")
        return resolved

    @staticmethod
    def _bounded(text: str) -> str:
        if len(text) <= MAX_RESULT_CHARS:
            return text
        return text[: MAX_RESULT_CHARS - 32] + "\n…[bounded output truncated]"

    def execute(self, request: ToolRequest) -> ToolResult:
        if not isinstance(request, ToolRequest):
            raise TypeError("request must be ToolRequest")
        started = monotonic()
        try:
            plan = self.plan(request)
            capability = self.task_executor.authorize(plan)
            result = self.task_executor.execute(plan, capability)
            output = "\n".join(item.output for item in result.action_results)
            artifact = result.artifacts[0].path if result.artifacts else None
            return ToolResult(
                request.operation,
                True,
                self._bounded(output),
                monotonic() - started,
                artifact,
            )
        except (
            OSError,
            UnicodeError,
            subprocess.SubprocessError,
            TaskExecutionError,
            TaskPlanError,
            ToolPolicyError,
        ) as exc:
            return ToolResult(
                request.operation,
                False,
                f"{type(exc).__name__}: {str(exc)[:512]}",
                monotonic() - started,
            )

    def plan(self, request: ToolRequest) -> TypedTaskPlan:
        """Compile one explicit slash command into an immutable typed plan.

        This is deliberately deterministic.  Free-form model output is never
        converted into task authority at this boundary.
        """

        if not isinstance(request, ToolRequest):
            raise TypeError("request must be ToolRequest")
        operation = request.operation
        if operation != "project" and request.artifact_bundle is not None:
            raise ToolPolicyError("only /project may carry an artifact bundle")
        action_id = f"{operation}-1"
        expected_artifacts: tuple[ExpectedArtifact, ...] = ()
        risk = RiskTier.T0
        timeout = 30.0
        cpu_seconds = 30.0
        ram_bytes = 256 * 1024**2
        verifier = TaskVerifier(VerifierKind.ALL_ACTIONS)

        if operation == "help":
            action = TaskAction(action_id, TaskActionKind.HELP)
            allowed_paths = (".",)
        elif operation == "status":
            action = TaskAction(action_id, TaskActionKind.STATUS)
            allowed_paths = (".",)
        elif operation == "list":
            action = TaskAction(action_id, TaskActionKind.LIST, path=request.argument)
            allowed_paths = (request.argument,)
        elif operation == "read":
            action = TaskAction(action_id, TaskActionKind.READ, path=request.argument)
            allowed_paths = (request.argument,)
        elif operation == "search":
            action = TaskAction(
                action_id,
                TaskActionKind.SEARCH,
                path=request.scope,
                query=request.argument,
            )
            allowed_paths = (request.scope,)
        elif operation == "test":
            if not self.allow_trusted_tests:
                raise ToolPolicyError(
                    "project test execution is gated; this runtime has no OS-level "
                    "process-tree and network sandbox attestation"
                )
            target = request.argument or "tests"
            action = TaskAction(
                action_id,
                TaskActionKind.RUN_TEST,
                path=target,
                argv=(sys.executable, "-m", "pytest", target, "-q"),
            )
            allowed_paths = (target,)
            risk = RiskTier.T1
            timeout = float(self.timeout_seconds)
            cpu_seconds = float(self.timeout_seconds)
            ram_bytes = 2 * 1024**3
            verifier = TaskVerifier(VerifierKind.PYTEST_PASS)
        elif operation == "save":
            path = f"outputs/agent-workspace/{request.argument}"
            digest = sha256(request.content.encode("utf-8")).hexdigest()
            action = TaskAction(
                action_id,
                TaskActionKind.WRITE_ARTIFACT,
                path=path,
                content=request.content,
            )
            allowed_paths = (path,)
            expected_artifacts = (ExpectedArtifact(path, digest),)
            risk = RiskTier.T1
            timeout = 10.0
            cpu_seconds = 5.0
            verifier = TaskVerifier(VerifierKind.ARTIFACT_SHA256)
        elif operation == "project":
            bundle = request.artifact_bundle
            if (
                bundle is None
                or any((request.argument, request.content))
                or request.scope != "."
            ):
                raise ToolPolicyError("/project requires only a typed artifact bundle")
            path = artifact_bundle_root(bundle)
            manifest_path = artifact_bundle_manifest_path(bundle)
            manifest = artifact_bundle_manifest_bytes(bundle)
            action = TaskAction(
                action_id,
                TaskActionKind.WRITE_ARTIFACT_BUNDLE,
                path=path,
                artifact_bundle=bundle,
            )
            allowed_paths = (path,)
            expected_artifacts = (
                ExpectedArtifact(
                    manifest_path,
                    sha256(manifest).hexdigest(),
                    max_bytes=MAX_SAVE_BYTES,
                ),
            )
            risk = RiskTier.T1
            timeout = 15.0
            cpu_seconds = 10.0
            ram_bytes = 512 * 1024**2
            verifier = TaskVerifier(VerifierKind.ARTIFACT_SHA256)
        else:
            raise ToolPolicyError("operation is not on the fixed allowlist")

        bundle_seed = (
            sha256(artifact_bundle_manifest_bytes(request.artifact_bundle)).hexdigest()
            if request.artifact_bundle is not None
            else ""
        )
        seed = "\x1f".join(
            (operation, request.argument, request.scope, request.content, bundle_seed)
        ).encode("utf-8")
        plan_id = "tool-" + sha256(seed).hexdigest()[:20]
        return TypedTaskPlan(
            plan_id=plan_id,
            objective=f"Execute explicit local /{operation} task",
            actions=(action,),
            allowed_paths=allowed_paths,
            required_inputs=(),
            expected_artifacts=expected_artifacts,
            verifier=verifier,
            budget=TaskBudget(
                time_seconds=timeout,
                cpu_seconds=cpu_seconds,
                ram_bytes=ram_bytes,
                vram_bytes=0,
                max_output_bytes=MAX_RESULT_CHARS,
            ),
            risk_tier=risk,
        )

    def authorize_plan(
        self, plan: TypedTaskPlan, *, ttl_seconds: float = 30.0
    ) -> CapabilityToken:
        """Issue one exact, short-lived, single-use local capability."""

        return self.task_executor.authorize(plan, ttl_seconds=ttl_seconds)

    def execute_plan(
        self, plan: TypedTaskPlan, capability: CapabilityToken
    ) -> TaskPlanResult:
        """Public typed-plan API; no natural-language/model parsing occurs."""

        return self.task_executor.execute(plan, capability)

    def _list(self, raw: str) -> str:
        directory = self._resolve(raw)
        if not directory.is_dir():
            raise ToolPolicyError("/list target is not a directory")
        entries: list[str] = []
        with os.scandir(directory) as iterator:
            for entry in iterator:
                if entry.name in IGNORED_DIRECTORIES:
                    continue
                if entry.is_symlink():
                    entries.append(f"[blocked-link] {entry.name}")
                    continue
                kind = "dir" if entry.is_dir(follow_symlinks=False) else "file"
                entries.append(f"[{kind}] {entry.name}")
                if len(entries) >= MAX_LIST_ENTRIES:
                    entries.append("…[entry limit reached]")
                    break
        entries.sort(key=str.casefold)
        return "\n".join(entries) or "(empty directory)"

    def _read(self, raw: str) -> str:
        path = self._resolve(raw, require_file=True)
        size = path.stat().st_size
        if size > MAX_READ_BYTES:
            raise ToolPolicyError(f"file exceeds {MAX_READ_BYTES} byte read limit")
        data = path.read_bytes()
        if b"\x00" in data:
            raise ToolPolicyError("binary files are not exposed to the text agent")
        return data.decode("utf-8", errors="strict")

    def _search(self, query: str, raw_scope: str) -> str:
        if len(query) > 256 or any(ord(char) < 32 for char in query):
            raise ToolPolicyError("search term is invalid or too long")
        scope = self._resolve(raw_scope)
        roots = [scope] if scope.is_file() else self._search_files(scope)
        matches: list[str] = []
        visited = 0
        needle = query.casefold()
        for path in roots:
            if visited >= MAX_SEARCH_FILES or len(matches) >= MAX_SEARCH_MATCHES:
                break
            if (
                path.is_symlink()
                or not path.is_file()
                or path.resolve() != path.absolute()
                or not path.resolve().is_relative_to(self.project_root)
                or any(part in IGNORED_DIRECTORIES for part in path.parts)
            ):
                continue
            visited += 1
            try:
                if path.stat().st_size > MAX_READ_BYTES:
                    continue
                data = path.read_bytes()
                if b"\x00" in data:
                    continue
                lines = data.decode("utf-8", errors="strict").splitlines()
            except (OSError, UnicodeError):
                continue
            for line_number, line in enumerate(lines, 1):
                if needle in line.casefold():
                    relative = path.relative_to(self.project_root).as_posix()
                    matches.append(f"{relative}:{line_number}: {line[:320]}")
                    if len(matches) >= MAX_SEARCH_MATCHES:
                        break
        if visited >= MAX_SEARCH_FILES or len(matches) >= MAX_SEARCH_MATCHES:
            matches.append("…[search bound reached]")
        return "\n".join(matches) or "검색 결과가 없습니다."

    def _search_files(self, scope: Path) -> Iterator[Path]:
        """Yield regular paths without following symlink or junction directories."""

        for raw_root, directories, files in os.walk(scope, followlinks=False):
            root = Path(raw_root)
            retained: list[str] = []
            for name in directories:
                candidate = root / name
                if name in IGNORED_DIRECTORIES or candidate.is_symlink():
                    continue
                try:
                    resolved = candidate.resolve(strict=True)
                except OSError:
                    continue
                if resolved == candidate.absolute() and resolved.is_relative_to(
                    self.project_root
                ):
                    retained.append(name)
            directories[:] = retained
            for name in files:
                yield root / name

    def _status(self) -> str:
        git = shutil.which("git")
        if git is None:
            raise ToolPolicyError("git executable is unavailable")
        completed = subprocess.run(
            [git, "status", "--short", "--branch"],
            cwd=self.project_root,
            env={"PATH": os.environ.get("PATH", ""), "GIT_OPTIONAL_LOCKS": "0"},
            capture_output=True,
            text=True,
            timeout=min(self.timeout_seconds, 30),
            check=False,
        )
        if completed.returncode != 0:
            raise ToolPolicyError("git status failed")
        return completed.stdout.strip() or "working tree clean"

    def _test(self, raw: str) -> str:
        target = raw.strip()
        command = [sys.executable, "-m", "pytest"]
        if target:
            path = Path(target)
            if (
                path.is_absolute()
                or ".." in path.parts
                or not path.as_posix().startswith("tests/")
                or path.suffix != ".py"
                or any(char.isspace() for char in target)
            ):
                raise ToolPolicyError("/test target must be one tests/*.py path")
            self._resolve(target, require_file=True)
            command.append(target)
        command.append("-q")
        env = {
            "PATH": os.environ.get("PATH", ""),
            "PYTHONPATH": str(self.project_root),
            "PYTHONNOUSERSITE": "1",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "HF_DATASETS_OFFLINE": "1",
            "WANDB_MODE": "offline",
            "NO_PROXY": "*",
        }
        completed = subprocess.run(
            command,
            cwd=self.project_root,
            env=env,
            capture_output=True,
            text=True,
            timeout=self.timeout_seconds,
            check=False,
        )
        output = (completed.stdout + "\n" + completed.stderr).strip()
        if completed.returncode != 0:
            raise ToolPolicyError(
                f"pytest failed ({completed.returncode})\n{output[-20_000:]}"
            )
        return output

    def _save(self, raw_name: str, content: str) -> tuple[str, str]:
        name = Path(raw_name)
        if name.name != raw_name or name.suffix.lower() not in ALLOWED_SAVE_SUFFIXES:
            raise ToolPolicyError(
                "saved artifact must be one safe filename (.txt/.md/.json/.csv)"
            )
        encoded = content.encode("utf-8")
        if not encoded or len(encoded) > MAX_SAVE_BYTES or b"\x00" in encoded:
            raise ToolPolicyError(
                "saved artifact content is empty, binary, or too large"
            )
        output_root = self._safe_output_root()
        target = (output_root / name.name).absolute()
        if not target.is_relative_to(output_root):
            raise ToolPolicyError("artifact path escaped the output directory")
        if target.exists() and (target.is_symlink() or not target.is_file()):
            raise ToolPolicyError("artifact target must be a regular file")
        temporary = output_root / f".{name.name}.{token_hex(8)}.tmp"
        descriptor = os.open(
            temporary,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o600,
        )
        try:
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(encoded)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, target)
        finally:
            temporary.unlink(missing_ok=True)
        relative = target.relative_to(self.project_root).as_posix()
        return f"저장 완료: {relative} ({len(encoded)} bytes)", relative

    def _safe_output_root(self) -> Path:
        outputs = self.project_root / "outputs"
        for directory in (outputs, self.output_root):
            if directory.exists() and (
                directory.is_symlink()
                or directory.resolve() != directory.absolute()
                or not directory.is_dir()
            ):
                raise ToolPolicyError("artifact output directory is not trusted")
            directory.mkdir(exist_ok=True)
        resolved = self.output_root.resolve(strict=True)
        if resolved != self.output_root.absolute() or not resolved.is_relative_to(
            self.project_root
        ):
            raise ToolPolicyError("artifact output directory escaped the project")
        return resolved


__all__ = [
    "HELP_TEXT",
    "ToolPolicyError",
    "ToolRequest",
    "ToolResult",
    "WorkspaceToolExecutor",
    "parse_tool_request",
]
