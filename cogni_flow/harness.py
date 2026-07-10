from __future__ import annotations

import ast
from dataclasses import dataclass
from hashlib import sha256
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from typing import Iterable, Protocol

from .rhythm import RhythmController, SystemMode


@dataclass(frozen=True)
class FailureTrace:
    test_id: str
    exception_type: str
    verifier_code: str
    mechanism: str
    excerpt: str = ""


@dataclass(frozen=True)
class WeaknessCluster:
    signature: tuple[str, str, str]
    traces: tuple[FailureTrace, ...]


def mine_weaknesses(traces: Iterable[FailureTrace]) -> list[WeaknessCluster]:
    groups: dict[tuple[str, str, str], list[FailureTrace]] = {}
    for trace in traces:
        signature = (trace.exception_type, trace.verifier_code, trace.mechanism)
        groups.setdefault(signature, []).append(trace)
    return [
        WeaknessCluster(signature, tuple(items))
        for signature, items in sorted(
            groups.items(), key=lambda item: (-len(item[1]), item[0])
        )
    ]


@dataclass(frozen=True)
class PatchProposal:
    relative_path: str
    base_sha256: str
    replacement: str
    rationale: str


@dataclass(frozen=True)
class PatchPolicy:
    allowed_roots: tuple[str, ...] = ("cogni_core", "cogni_flow")
    max_bytes: int = 256_000
    forbidden_imports: tuple[str, ...] = (
        "requests",
        "httpx",
        "aiohttp",
        "urllib",
        "socket",
        "ftplib",
        "telnetlib",
        "builtins",
        "importlib",
    )
    forbidden_calls: tuple[str, ...] = (
        "eval",
        "exec",
        "compile",
        "__import__",
        "getattr",
        "setattr",
        "delattr",
        "globals",
        "locals",
        "vars",
    )
    forbidden_attributes: tuple[str, ...] = (
        "eval",
        "exec",
        "compile",
        "__import__",
        "__builtins__",
        "__class__",
        "__code__",
        "__dict__",
        "__globals__",
        "__mro__",
        "__subclasses__",
        "write_text",
        "write_bytes",
        "unlink",
        "rmdir",
        "chmod",
        "lchmod",
        "touch",
        "symlink_to",
        "hardlink_to",
    )

    _blocked_module_use: tuple[str, ...] = ("subprocess", "smtplib")
    _blocked_os_calls: tuple[str, ...] = (
        "system",
        "popen",
        "open",
        "fdopen",
        "write",
        "remove",
        "unlink",
        "rename",
        "renames",
        "replace",
        "rmdir",
        "removedirs",
        "mkdir",
        "makedirs",
        "chmod",
        "chown",
        "lchown",
        "link",
        "symlink",
        "truncate",
        "kill",
        "killpg",
        "startfile",
    )
    _blocked_shutil_calls: tuple[str, ...] = (
        "rmtree",
        "move",
        "copy",
        "copy2",
        "copyfile",
        "copyfileobj",
        "copymode",
        "copystat",
        "copytree",
        "chown",
        "make_archive",
        "unpack_archive",
    )
    _blocked_path_calls: tuple[str, ...] = (
        "write_text",
        "write_bytes",
        "unlink",
        "rename",
        "replace",
        "mkdir",
        "rmdir",
        "chmod",
        "lchmod",
        "touch",
        "symlink_to",
        "hardlink_to",
    )

    def validate(self, proposal: PatchProposal) -> Path:
        relative = Path(proposal.relative_path)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("patch target must be a safe relative path")
        if not relative.parts or relative.parts[0] not in self.allowed_roots:
            raise ValueError("patch target is outside the declared mutable surface")
        if len(proposal.replacement.encode("utf-8")) > self.max_bytes:
            raise ValueError("replacement exceeds patch size limit")
        if relative.suffix == ".py":
            tree = ast.parse(proposal.replacement, filename=str(relative))
            self._validate_python_tree(tree)
        return relative

    def _validate_python_tree(self, tree: ast.AST) -> None:
        aliases: dict[str, str] = {}
        nodes = tuple(ast.walk(tree))

        for node in nodes:
            if isinstance(node, ast.Import):
                for alias in node.names:
                    root = alias.name.split(".")[0]
                    if root in self.forbidden_imports:
                        raise ValueError(
                            "network or dynamic-loader import rejected by policy"
                        )
                    aliases[alias.asname or root] = alias.name
            elif isinstance(node, ast.ImportFrom):
                root = (node.module or "").split(".")[0]
                if root in self.forbidden_imports:
                    raise ValueError(
                        "network or dynamic-loader import rejected by policy"
                    )
                for alias in node.names:
                    if alias.name == "*":
                        raise ValueError("wildcard import rejected by patch policy")
                    qualified = ".".join(
                        part for part in (node.module, alias.name) if part
                    )
                    aliases[alias.asname or alias.name] = qualified

        # Propagate simple aliases and Path instances.  This catches bypasses
        # such as ``runner = subprocess`` and ``p = Path(...); p.unlink()``.
        for _ in range(len(nodes) + 1):
            changed = False
            for node in nodes:
                if not isinstance(node, (ast.Assign, ast.AnnAssign)):
                    continue
                value = node.value
                if value is None:
                    continue
                qualified = self._qualified_name(value, aliases)
                if qualified is None:
                    continue
                targets = (
                    node.targets if isinstance(node, ast.Assign) else [node.target]
                )
                for target in targets:
                    if (
                        isinstance(target, ast.Name)
                        and aliases.get(target.id) != qualified
                    ):
                        aliases[target.id] = qualified
                        changed = True
            if not changed:
                break

        for node in nodes:
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
                if node.id == "__builtins__":
                    raise ValueError("builtins namespace access rejected")
                qualified = aliases.get(node.id)
                if qualified and qualified.split(".")[0] in self._blocked_module_use:
                    raise ValueError(
                        f"dangerous module use rejected: {qualified.split('.')[0]}"
                    )

            if isinstance(node, ast.Attribute):
                if node.attr in self.forbidden_attributes:
                    raise ValueError(f"forbidden attribute access: {node.attr}")
                self._reject_dangerous_target(
                    self._qualified_name(node, aliases), node.attr
                )

            if not isinstance(node, ast.Call):
                continue
            qualified = self._qualified_name(node.func, aliases)
            leaf = qualified.rsplit(".", 1)[-1] if qualified else None
            if leaf in self.forbidden_calls:
                raise ValueError(f"forbidden dynamic-code call: {leaf}")
            self._reject_dangerous_target(qualified, leaf)
            if leaf == "open" and self._open_may_write(node, qualified):
                raise ValueError("file open with write-capable mode rejected")

    def _reject_dangerous_target(self, qualified: str | None, leaf: str | None) -> None:
        if not qualified:
            return
        root = qualified.split(".", 1)[0]
        if root in self._blocked_module_use:
            raise ValueError(f"dangerous module use rejected: {root}")
        if root == "os" and leaf is not None:
            if (
                leaf in self._blocked_os_calls
                or leaf.startswith("exec")
                or leaf.startswith("spawn")
            ):
                raise ValueError(f"dangerous os operation rejected: {leaf}")
        if root == "shutil" and leaf in self._blocked_shutil_calls:
            raise ValueError(f"dangerous shutil operation rejected: {leaf}")
        if root == "pathlib" and leaf in self._blocked_path_calls:
            raise ValueError(f"dangerous pathlib operation rejected: {leaf}")

    @staticmethod
    def _qualified_name(node: ast.AST, aliases: dict[str, str]) -> str | None:
        if isinstance(node, ast.Name):
            return aliases.get(node.id, node.id)
        if isinstance(node, ast.Attribute):
            owner = PatchPolicy._qualified_name(node.value, aliases)
            return f"{owner}.{node.attr}" if owner else None
        if isinstance(node, ast.Call):
            constructor = PatchPolicy._qualified_name(node.func, aliases)
            if constructor in {"pathlib.Path", "pathlib.PurePath"}:
                return f"{constructor}()"
        return None

    @staticmethod
    def _open_may_write(node: ast.Call, qualified: str | None) -> bool:
        is_path_method = bool(qualified and qualified.startswith("pathlib."))
        mode_index = 0 if is_path_method else 1
        mode_node: ast.AST | None = (
            node.args[mode_index] if len(node.args) > mode_index else None
        )
        for keyword in node.keywords:
            if keyword.arg == "mode":
                mode_node = keyword.value
                break
        if mode_node is None:
            return False
        if not isinstance(mode_node, ast.Constant) or not isinstance(
            mode_node.value, str
        ):
            return True
        return any(flag in mode_node.value for flag in "wax+")


@dataclass(frozen=True)
class SandboxResult:
    passed: bool
    returncode: int
    output: str


class SandboxRunner(Protocol):
    kernel_isolated: bool

    def run(
        self, project: Path, command: tuple[str, ...], timeout_seconds: int
    ) -> SandboxResult: ...


class SubprocessSandbox:
    """Run trusted developer diagnostics with process-level controls only.

    This is not a candidate-code sandbox. ``SafeHarnessPatcher`` always rejects
    it. Production integrations must replace it with an OS/container runner
    that enforces network and filesystem isolation at the kernel boundary.
    """

    kernel_isolated = False

    def run(
        self, project: Path, command: tuple[str, ...], timeout_seconds: int
    ) -> SandboxResult:
        env = {
            "PATH": os.environ.get("PATH", ""),
            "PYTHONPATH": str(project),
            "PYTHONNOUSERSITE": "1",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "NO_PROXY": "*",
        }
        completed = subprocess.run(
            command,
            cwd=project,
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,
        )
        output = (completed.stdout + "\n" + completed.stderr)[-40_000:]
        return SandboxResult(completed.returncode == 0, completed.returncode, output)


@dataclass(frozen=True)
class PromotionResult:
    promoted: bool
    sandbox: SandboxResult
    target: Path


class SafeHarnessPatcher:
    """Validate and promote candidates only through kernel-enforced isolation.

    ``SubprocessSandbox`` remains a diagnostic utility, but it is deliberately
    unusable for autonomous candidate execution.  A production integration
    must inject a runner whose concrete isolation boundary attests
    ``kernel_isolated = True``.
    """

    def __init__(
        self,
        project_root: Path,
        rhythm: RhythmController,
        *,
        policy: PatchPolicy | None = None,
        sandbox: SandboxRunner | None = None,
        test_command: tuple[str, ...] | None = None,
        timeout_seconds: int = 180,
        require_kernel_isolation: bool = True,
    ) -> None:
        self.project_root = project_root.resolve()
        self.rhythm = rhythm
        self.policy = policy or PatchPolicy()
        self.sandbox = sandbox or SubprocessSandbox()
        self.test_command = test_command or (
            sys.executable,
            "-m",
            "unittest",
            "discover",
            "-s",
            "tests",
            "-v",
        )
        self.timeout_seconds = timeout_seconds
        self.require_kernel_isolation = require_kernel_isolation

    def validate_and_promote(self, proposal: PatchProposal) -> PromotionResult:
        if self.rhythm.mode != SystemMode.EVOLUTION:
            raise RuntimeError("patching is allowed only during evolution mode")
        if not self.require_kernel_isolation:
            raise RuntimeError(
                "kernel isolation cannot be disabled for autonomous patching"
            )
        if getattr(self.sandbox, "kernel_isolated", None) is not True:
            raise RuntimeError(
                "candidate execution requires a kernel-isolated SandboxRunner"
            )
        with self.rhythm.evolution_slot():
            relative = self.policy.validate(proposal)
            target = (self.project_root / relative).resolve()
            if self.project_root not in target.parents:
                raise ValueError("resolved patch target escaped project root")
            current = target.read_bytes() if target.exists() else b""
            if sha256(current).hexdigest() != proposal.base_sha256:
                raise RuntimeError("base file changed since proposal generation")

            self.rhythm.transition(
                SystemMode.VALIDATING, f"validating {relative.as_posix()}"
            )
            with tempfile.TemporaryDirectory(prefix="cogni-harness-") as tmp:
                stage = Path(tmp) / "project"
                shutil.copytree(
                    self.project_root,
                    stage,
                    ignore=shutil.ignore_patterns(
                        "work", ".git", "__pycache__", "*.pyc"
                    ),
                )
                staged_target = stage / relative
                staged_target.parent.mkdir(parents=True, exist_ok=True)
                staged_target.write_text(proposal.replacement, encoding="utf-8")
                result = self.sandbox.run(
                    stage, self.test_command, self.timeout_seconds
                )
            if not result.passed:
                self.rhythm.transition(
                    SystemMode.EVOLUTION, "candidate failed regression tests"
                )
                return PromotionResult(False, result, target)

            self.rhythm.transition(
                SystemMode.PROMOTING, f"promoting {relative.as_posix()}"
            )
            target.parent.mkdir(parents=True, exist_ok=True)
            temp_target = target.with_suffix(target.suffix + ".cogni-new")
            temp_target.write_text(proposal.replacement, encoding="utf-8")
            os.replace(temp_target, target)
            return PromotionResult(True, result, target)


def file_digest(path: Path) -> str:
    return sha256(path.read_bytes() if path.exists() else b"").hexdigest()
