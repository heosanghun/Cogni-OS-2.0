from __future__ import annotations

from pathlib import Path
import re
import unittest


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "release" / "COGNI_OS_2_VALIDATION_REPORT_KO.md"
PLAN = ROOT / "docs" / "COGNIBOARD_V041_SERVER_IMPLEMENTATION_PLAN_KO.md"
CHECKLIST = ROOT / "docs" / "COGNIBOARD_MASTER_ACCEPTANCE_CHECKLIST_KO.md"
DOCUMENTS = tuple(
    sorted(
        {
            ROOT / "README.md",
            *ROOT.joinpath("docs").rglob("*.md"),
            *ROOT.joinpath("release").rglob("*.md"),
        },
        key=lambda item: item.as_posix(),
    )
)


def _logical_command_lines(text: str) -> tuple[str, ...]:
    logical: list[str] = []
    pending = ""
    for raw in text.splitlines():
        stripped = raw.strip()
        if stripped.startswith("```"):
            continue
        fragment = re.sub(r"^\s*(?:>\s*)?(?:[-*+]\s+)?", "", raw).strip()
        continued = bool(re.search(r"(?:\\|\^|`)\s*$", fragment))
        fragment = re.sub(r"(?:\\|\^|`)\s*$", "", fragment).strip()
        pending = f"{pending} {fragment}".strip()
        if not continued:
            if pending:
                logical.append(pending)
            pending = ""
    if pending:
        logical.append(pending)
    return tuple(logical)


def _unwrap_command(value: str) -> str:
    command = value.strip().strip("\"'`")
    while True:
        previous = command
        command = re.sub(r"^\s*&\s*", "", command)
        command = re.sub(r"^\s*sudo(?:\.exe)?\s+", "", command, flags=re.I)
        command = re.sub(
            r"^\s*env(?:\.exe)?(?:\s+[A-Za-z_][A-Za-z0-9_]*=[^\s]+)*\s+",
            "",
            command,
            flags=re.I,
        )
        command = re.sub(r"^\s*cmd(?:\.exe)?\s+/(?:c|k)\s+", "", command, flags=re.I)
        command = re.sub(
            r"^\s*(?:powershell|pwsh)(?:\.exe)?\s+(?:-command|-c)\s+",
            "",
            command,
            flags=re.I,
        )
        command = re.sub(
            r"^\s*(?:ba|z|k|da)?sh(?:\.exe)?\s+(?:-c|-lc)\s+",
            "",
            command,
            flags=re.I,
        )
        command = command.strip().strip("\"'`")
        if command == previous:
            return command


def _direct_gpu_commands(text: str) -> tuple[str, ...]:
    violations: list[str] = []
    for logical in _logical_command_lines(text):
        # A command can be hidden after cmd/PowerShell's `&` call operator or a
        # shell command separator.  Examine each segment independently.
        segments = re.split(r"\s*(?:&&|;|(?<!&)&(?!&))\s*", logical)
        for segment in segments:
            environment_head = segment.strip().strip("\"'`")
            if re.match(
                r"^(?:sudo(?:\.exe)?\s+)?(?:env(?:\.exe)?\s+)?"
                r"(?:(?!(?:CUDA_VISIBLE_DEVICES|NVIDIA_VISIBLE_DEVICES)\s*=)"
                r"[A-Za-z_][A-Za-z0-9_]*=[^\s]+\s+)*"
                r"(?:CUDA_VISIBLE_DEVICES|NVIDIA_VISIBLE_DEVICES)\s*=",
                environment_head,
                flags=re.I,
            ):
                violations.append(logical)
                continue
            command = _unwrap_command(segment)
            lowered = command.casefold()
            if not command:
                continue
            validator = re.search(
                r"(?:scripts[/\\]validate_gemma4_(?:runtime|deq)\.py"
                r"|scripts\.validate_gemma4_(?:runtime|deq))\b",
                lowered,
            )
            if validator is not None:
                guard = lowered.find("scripts/gpu5_boundary_guard.py run")
                guard_backslash = lowered.find("scripts\\gpu5_boundary_guard.py run")
                guard_positions = [
                    item for item in (guard, guard_backslash) if item >= 0
                ]
                if not guard_positions or min(guard_positions) > validator.start():
                    if re.match(
                        r"^(?:(?:[a-z]:)?[/\\][^\s]+[/\\])?"
                        r"(?:py(?:\.exe)?\s+-\d+(?:\.\d+)?|python(?:\d+(?:\.\d+)*)?(?:\.exe)?)\b",
                        command,
                        flags=re.I,
                    ):
                        violations.append(logical)
                        continue
            python_command = re.match(
                r"^(?:(?:[a-z]:)?[/\\][^\s]+[/\\])?"
                r"(?:py(?:\.exe)?\s+-\d+(?:\.\d+)?|python(?:\d+(?:\.\d+)*)?(?:\.exe)?)\b",
                command,
                flags=re.I,
            )
            if python_command is not None:
                completion_validator = re.search(
                    r"(?:scripts[/\\]validate_agent_completion\.py"
                    r"|scripts\.validate_agent_completion)\b",
                    lowered,
                )
                explicit_gpu = re.search(
                    r"(?:--device(?:=|\s+)|--accelerator(?:=|\s+))"
                    r"(?:cuda(?::\d+)?|gpu)\b",
                    lowered,
                )
                if completion_validator is not None and explicit_gpu is not None:
                    violations.append(logical)
                    continue
            if re.match(
                r"^(?:(?:[a-z]:)?[/\\][^\s]+[/\\])?nvidia-smi(?:\.exe)?\b",
                command,
                flags=re.I,
            ):
                violations.append(logical)
                continue
            if re.match(
                r"^(?:(?:[a-z]:)?[/\\][^\s]+[/\\])?"
                r"(?:docker|podman|nerdctl)(?:\.exe)?\s+(?:run|create)\b",
                command,
                flags=re.I,
            ):
                violations.append(logical)
                continue
            if re.match(
                r"^(?:(?:[a-z]:)?[/\\][^\s]+[/\\])?"
                r"(?:(?:docker|podman|nerdctl)(?:\.exe)?\s+compose"
                r"|(?:docker|podman)-compose(?:\.exe)?)\b",
                command,
                flags=re.I,
            ):
                violations.append(logical)
                continue
            if re.match(
                r"^(?:CUDA_VISIBLE_DEVICES|NVIDIA_VISIBLE_DEVICES)\s*=",
                command,
                flags=re.I,
            ):
                violations.append(logical)
    return tuple(violations)


class TestHistoricalValidationReport(unittest.TestCase):
    def test_historical_gpu_observation_cannot_be_republished_as_current(self) -> None:
        text = REPORT.read_text(encoding="utf-8")

        for forbidden in (
            "python scripts\\validate_gemma4_runtime.py",
            "CURRENT-SCOPE GATES PASS",
            "PASS (measured)",
            "보존된 현재-scope 원시 실행 결과",
            "이 결과는 정확한 현재 모델·코드·config·장치 scope",
            "현재 CUDA 10,000-iteration 결과",
        ):
            with self.subTest(forbidden=forbidden):
                self.assertNotIn(forbidden, text)

        self.assertIn(
            "HISTORICAL DEV-HOST OBSERVATION; CURRENT SCOPE UNVERIFIED",
            text,
        )

    def test_all_release_documents_reject_direct_gpu_command_variants(self) -> None:
        for document in DOCUMENTS:
            text = document.read_text(encoding="utf-8")
            with self.subTest(document=document.relative_to(ROOT)):
                self.assertEqual(_direct_gpu_commands(text), ())

    def test_direct_gpu_scanner_catches_wrappers_suffixes_and_continuations(
        self,
    ) -> None:
        attacks = (
            "py -3 scripts\\validate_gemma4_runtime.py --model x",
            "python.exe scripts/validate_gemma4_deq.py --model x",
            "sudo env FOO=1 python3.11 scripts/validate_gemma4_runtime.py",
            'cmd.exe /c "python.exe scripts\\validate_gemma4_runtime.py"',
            'powershell.exe -Command "& py -3 scripts\\validate_gemma4_deq.py"',
            "& python scripts/validate_gemma4_runtime.py",
            "python `\n scripts/validate_gemma4_runtime.py",
            "docker.exe run --gpus all image",
            "sudo /usr/bin/nvidia-smi -i 5",
            "env CUDA_VISIBLE_DEVICES=5 python task.py",
            "NVIDIA_VISIBLE_DEVICES=all python task.py",
            "python -m scripts.validate_gemma4_runtime --model x",
            "python.exe -m scripts.validate_gemma4_deq --model x",
            "python scripts/validate_agent_completion.py --device cuda",
            "python -m scripts.validate_agent_completion --device=cuda:0",
            'sh -c "nvidia-smi -i 5"',
            'bash -lc "python -m scripts.validate_gemma4_runtime --model x"',
            "podman run --device nvidia.com/gpu=all image",
            "nerdctl.exe create --gpus=all image",
            "docker compose -f gpu.yml up",
            "docker-compose.exe run worker",
            "podman-compose up",
            "nerdctl compose up",
        )
        for attack in attacks:
            with self.subTest(attack=attack):
                self.assertTrue(_direct_gpu_commands(attack))

    def test_guard_wrapped_validator_recipe_is_not_reported_as_direct(self) -> None:
        guarded = (
            "/usr/bin/python3 -I -B \\\n"
            " scripts/gpu5_boundary_guard.py run \\\n"
            " -- -I -B /workspace/scripts/validate_gemma4_runtime.py"
        )
        self.assertEqual(_direct_gpu_commands(guarded), ())

    def test_guard_policy_is_the_only_documented_gpu_entrypoint(self) -> None:

        plan = PLAN.read_text(encoding="utf-8")
        for obsolete_direct_recipe in (
            "호스트 네이티브 작업은 Python 시작 전",
            "Docker 작업은 `--gpus",
            "실행 직전과 직후 호스트에서 `nvidia-smi -i 5`만 사용",
        ):
            with self.subTest(obsolete_direct_recipe=obsolete_direct_recipe):
                self.assertNotIn(obsolete_direct_recipe, plan)
        self.assertIn(
            "운영자는 네이티브 Python, Docker, validator 또는 GPU 조회 도구를 직접 실행하지",
            plan,
        )
        self.assertIn("scripts/gpu5_boundary_guard.py run", plan)
        report = REPORT.read_text(encoding="utf-8")
        self.assertIn("docs/GEMMA4_VALIDATION.md", report)
        self.assertIn("scripts/gpu5_boundary_guard.py", report)
        self.assertIn(
            "현재 v0.4.1 scope의 PASS 또는 VERIFIED 증거가 아니다",
            report,
        )


if __name__ == "__main__":
    unittest.main()
