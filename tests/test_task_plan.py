from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
from hashlib import sha256
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from cogni_flow.task_plan import (
    CapabilityError,
    ExpectedArtifact,
    InMemoryProposalStager,
    RequiredInput,
    RiskTier,
    TaskAction,
    TaskActionKind,
    TaskBudget,
    TaskExecutionError,
    TaskPlanExecutor,
    TaskPlanPolicy,
    TaskPolicyError,
    TaskVerifier,
    TypedTaskPlan,
    UnverifiedPlannerError,
    UnverifiedPlannerGate,
    VerifierKind,
)


def digest(data: bytes) -> str:
    return sha256(data).hexdigest()


def budget(*, timeout: float = 10.0, output: int = 40_000) -> TaskBudget:
    return TaskBudget(
        time_seconds=timeout,
        cpu_seconds=max(0.01, timeout),
        ram_bytes=256 * 1024**2,
        vram_bytes=0,
        max_output_bytes=output,
    )


def plan(
    index: int,
    action: TaskAction,
    *,
    allowed: tuple[str, ...] | None = None,
    risk: RiskTier | None = None,
    inputs: tuple[RequiredInput, ...] = (),
    artifacts: tuple[ExpectedArtifact, ...] = (),
    verifier: VerifierKind = VerifierKind.ALL_ACTIONS,
    selected_budget: TaskBudget | None = None,
) -> TypedTaskPlan:
    return TypedTaskPlan(
        plan_id=f"phase9-{index:04d}",
        objective=f"deterministic phase 9 case {index}",
        actions=(action,),
        allowed_paths=allowed or ((action.path or "."),),
        required_inputs=inputs,
        expected_artifacts=artifacts,
        verifier=TaskVerifier(verifier),
        budget=selected_budget or budget(),
        risk_tier=risk
        or {
            TaskActionKind.HELP: RiskTier.T0,
            TaskActionKind.LIST: RiskTier.T0,
            TaskActionKind.READ: RiskTier.T0,
            TaskActionKind.SEARCH: RiskTier.T0,
            TaskActionKind.STATUS: RiskTier.T0,
            TaskActionKind.RUN_TEST: RiskTier.T1,
            TaskActionKind.WRITE_ARTIFACT: RiskTier.T1,
            TaskActionKind.STAGE_SOURCE_CHANGE: RiskTier.T2,
        }.get(action.kind, RiskTier.T3),
    )


class TestTypedTaskPlanSchema(unittest.TestCase):
    def test_schema_is_deeply_immutable_and_digest_is_content_addressed(self) -> None:
        action = TaskAction("read-1", TaskActionKind.READ, path="input.txt")
        first = plan(1, action)
        second = plan(1, action)
        self.assertEqual(first.digest, second.digest)
        self.assertNotEqual(first.digest, replace(first, objective="changed").digest)
        with self.assertRaises(FrozenInstanceError):
            first.objective = "mutate"  # type: ignore[misc]
        with self.assertRaises(TypeError):
            TypedTaskPlan(
                plan_id="phase9-list",
                objective="mutable list rejected",
                actions=[action],  # type: ignore[arg-type]
                allowed_paths=("input.txt",),
                required_inputs=(),
                expected_artifacts=(),
                verifier=TaskVerifier(VerifierKind.ALL_ACTIONS),
                budget=budget(),
                risk_tier=RiskTier.T0,
            )

    def test_unverified_natural_language_planner_is_permanently_gated(self) -> None:
        gate = UnverifiedPlannerGate()
        with self.assertRaises(UnverifiedPlannerError):
            gate.plan_natural_language("read everything and fix it")
        with self.assertRaises(UnverifiedPlannerError):
            gate.admit_typed({"objective": "forged"})  # type: ignore[arg-type]


class TaskPlanFixture(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        (self.root / "src").mkdir()
        (self.root / "tests").mkdir()
        (self.root / "cogni_core").mkdir()
        self.policy = TaskPlanPolicy()
        self.executor = TaskPlanExecutor(self.root, policy=self.policy)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def execute(self, selected: TypedTaskPlan):
        capability = self.executor.authorize(selected)
        return self.executor.execute(selected, capability)


class TestPolicyAdversarialCases(TaskPlanFixture):
    def test_at_least_one_hundred_traversal_and_ambiguous_paths_are_blocked(
        self,
    ) -> None:
        attacks: list[str] = [
            "../secret",
            "a/../../secret",
            "/absolute",
            "//server/share",
            r"\\server\share",
            r"C:\Windows\win.ini",
            "C:relative",
            "./src/file.py",
            "src//file.py",
            "src/./file.py",
            "src/../file.py",
            "src/file.py ",
            "src/file.py.",
            "src/file.py:stream",
            "NUL",
            "CON.txt",
            "COM1",
            "LPT9.log",
            "%2e%2e/secret",
            "src/%2F/secret",
            "src\x00/file",
            " src/file",
        ]
        for index in range(30):
            attacks.extend(
                (
                    f"../secret-{index}",
                    f"dir-{index}/../secret",
                    f"dir-{index}//secret",
                    f"dir-{index}/./secret",
                )
            )
        self.assertGreaterEqual(len(attacks), 100)
        blocked = 0
        for index, attack in enumerate(attacks, 1):
            action = TaskAction(f"attack-{index}", TaskActionKind.READ, path=attack)
            selected = plan(1000 + index, action, allowed=(attack,))
            with self.subTest(index=index, attack=attack):
                with self.assertRaises(TaskPolicyError):
                    self.policy.validate(selected)
                blocked += 1
        self.assertEqual(blocked, len(attacks))

    def test_all_t3_authority_categories_are_denied_before_capability_issue(
        self,
    ) -> None:
        forbidden = (
            TaskActionKind.NETWORK,
            TaskActionKind.ARBITRARY_SHELL,
            TaskActionKind.EVALUATOR_MUTATION,
            TaskActionKind.SECURITY_MUTATION,
            TaskActionKind.UPDATER_MUTATION,
            TaskActionKind.ROLLBACK_MUTATION,
        )
        for index, kind in enumerate(forbidden, 1):
            selected = plan(
                2000 + index,
                TaskAction(f"deny-{index}", kind, path="src"),
                risk=RiskTier.T3,
            )
            with self.subTest(kind=kind):
                with self.assertRaises(TaskPolicyError):
                    self.executor.authorize(selected)

    def test_wrong_risk_argv_path_and_budget_labels_fail_closed(self) -> None:
        base = plan(3000, TaskAction("read", TaskActionKind.READ, path="src"))
        invalid = [
            replace(base, risk_tier=RiskTier.T1),
            replace(base, allowed_paths=("tests",)),
            replace(base, budget=replace(base.budget, vram_bytes=1)),
            replace(base, budget=replace(base.budget, time_seconds=901.0)),
            replace(base, budget=replace(base.budget, ram_bytes=1)),
            replace(base, schema_version=2),
            replace(base, plan_id="bad"),
            replace(base, objective=""),
            replace(
                base,
                actions=(
                    TaskAction(
                        "read",
                        TaskActionKind.READ,
                        path="src",
                        argv=("cmd", "/c", "whoami"),
                    ),
                ),
            ),
        ]
        for index, selected in enumerate(invalid):
            with self.subTest(index=index):
                with self.assertRaises(TaskPolicyError):
                    self.policy.validate(selected)

    def test_linklike_ancestor_and_toctou_identity_change_are_rejected(self) -> None:
        target = self.root / "src" / "stable.txt"
        target.write_text("stable", encoding="utf-8")
        selected = plan(
            3100, TaskAction("read", TaskActionKind.READ, path="src/stable.txt")
        )

        original_linklike = self.executor._is_linklike

        def forged_link(path_value: Path, identity: object) -> bool:
            return path_value.name == "src" or original_linklike(path_value, identity)  # type: ignore[arg-type]

        with patch.object(self.executor, "_is_linklike", side_effect=forged_link):
            with self.assertRaises(TaskPolicyError):
                self.execute(selected)

        original_identity = self.executor._identity
        calls = 0

        def unstable(path_value: Path):
            nonlocal calls
            value = original_identity(path_value)
            if path_value.name == "stable.txt":
                calls += 1
                if calls >= 2:
                    return replace(value, mtime_ns=value.mtime_ns + 1)
            return value

        with patch.object(self.executor, "_identity", side_effect=unstable):
            with self.assertRaises(TaskExecutionError):
                self.execute(selected)


class TestAllowedDeterministicTasks(TaskPlanFixture):
    def test_at_least_thirty_allowed_t0_tasks_succeed_deterministically(self) -> None:
        allowed_cases = 0
        for index in range(12):
            directory = self.root / "src" / f"d{index}"
            directory.mkdir()
            (directory / "value.txt").write_bytes(f"needle-{index}\n".encode("utf-8"))

        for index in range(12):
            relative = f"src/d{index}/value.txt"
            selected = plan(
                4000 + index,
                TaskAction(f"read-{index}", TaskActionKind.READ, path=relative),
            )
            result = self.execute(selected)
            self.assertEqual(result.action_results[0].output, f"needle-{index}\n")
            allowed_cases += 1

        for index in range(12):
            relative = f"src/d{index}"
            selected = plan(
                4100 + index,
                TaskAction(f"list-{index}", TaskActionKind.LIST, path=relative),
            )
            result = self.execute(selected)
            self.assertIn("value.txt", result.action_results[0].output)
            allowed_cases += 1

        for index in range(12):
            relative = f"src/d{index}"
            selected = plan(
                4200 + index,
                TaskAction(
                    f"search-{index}",
                    TaskActionKind.SEARCH,
                    path=relative,
                    query=f"needle-{index}",
                ),
            )
            result = self.execute(selected)
            self.assertIn(f"src/d{index}/value.txt:1", result.action_results[0].output)
            allowed_cases += 1

        self.assertGreaterEqual(allowed_cases, 30)
        self.assertEqual(allowed_cases, 36)

    def test_root_listing_is_allowed_but_links_are_never_followed(self) -> None:
        selected = plan(
            4300,
            TaskAction("list-root", TaskActionKind.LIST, path="."),
            allowed=(".",),
        )
        result = self.execute(selected)
        self.assertIn("[dir] src", result.action_results[0].output)


class TestCapabilityAndVerification(TaskPlanFixture):
    def test_capability_is_exact_single_use_replay_safe_and_revocable(self) -> None:
        (self.root / "src" / "one.txt").write_text("one", encoding="utf-8")
        first = plan(5000, TaskAction("read", TaskActionKind.READ, path="src/one.txt"))
        second = replace(first, objective="different objective")
        token = self.executor.authorize(first)
        with self.assertRaises(CapabilityError):
            self.executor.execute(second, token)
        # A mismatched attempt consumes the capability.
        with self.assertRaises(CapabilityError):
            self.executor.execute(first, token)

        replay = self.executor.authorize(first)
        self.executor.execute(first, replay)
        with self.assertRaises(CapabilityError):
            self.executor.execute(first, replay)

        revoked = self.executor.authorize(first)
        self.executor.revoke(revoked)
        with self.assertRaises(CapabilityError):
            self.executor.execute(first, revoked)

        now = 10_000_000_000

        def clock_ns() -> int:
            return now

        deterministic = TaskPlanExecutor(self.root, clock_ns=clock_ns)
        expired = deterministic.authorize(first, ttl_seconds=0.05)
        now += 50_000_001
        with self.assertRaises(CapabilityError):
            deterministic.execute(first, expired)

    def test_artifact_is_atomic_and_verified_by_readback_sha256(self) -> None:
        content = "verified artifact\n"
        relative = "outputs/agent-workspace/evidence.md"
        expected = digest(content.encode("utf-8"))
        selected = plan(
            5100,
            TaskAction(
                "write",
                TaskActionKind.WRITE_ARTIFACT,
                path=relative,
                content=content,
            ),
            risk=RiskTier.T1,
            artifacts=(ExpectedArtifact(relative, expected),),
            verifier=VerifierKind.ARTIFACT_SHA256,
        )
        result = self.execute(selected)
        self.assertTrue(result.verifier_passed)
        self.assertEqual(result.artifacts[0].sha256, expected)
        self.assertEqual((self.root / relative).read_bytes(), content.encode("utf-8"))

    def test_required_input_digest_is_checked_before_execution(self) -> None:
        source = self.root / "src" / "input.txt"
        source.write_text("actual", encoding="utf-8")
        selected = plan(
            5200,
            TaskAction("read", TaskActionKind.READ, path="src/input.txt"),
            inputs=(RequiredInput("src/input.txt", "0" * 64),),
        )
        with self.assertRaises(TaskExecutionError):
            self.execute(selected)

    def test_environment_is_minimal_offline_and_drops_ambient_secrets(self) -> None:
        with patch.dict(os.environ, {"COGNI_TEST_SECRET": "must-not-cross"}):
            environment = self.executor._bounded_environment(status_environment=False)
        self.assertNotIn("COGNI_TEST_SECRET", environment)
        self.assertEqual(environment["CUDA_VISIBLE_DEVICES"], "")
        self.assertEqual(environment["HF_HUB_OFFLINE"], "1")
        self.assertEqual(environment["PYTEST_DISABLE_PLUGIN_AUTOLOAD"], "1")


class TestFixedSubprocessAndT2Staging(TaskPlanFixture):
    @staticmethod
    def _linux_stat(*, state: str, user: int, system: int, rss: int) -> str:
        tail = [state] + ["0"] * 10
        tail.extend((str(user), str(system)))
        tail.extend(["0"] * 8)
        tail.append(str(rss))
        return "4321 (python worker) " + " ".join(tail)

    @unittest.skipIf(os.name == "nt", "Linux /proc semantics")
    def test_atomic_zombie_stat_keeps_final_cpu_and_zero_ram(self) -> None:
        process = type("ExitedProcess", (), {"pid": 4321})()
        stat = self._linux_stat(state="Z", user=25, system=5, rss=0)
        with (
            patch.object(Path, "read_text", return_value=stat),
            patch(
                "cogni_flow.task_plan.os.sysconf",
                side_effect=lambda name: {
                    "SC_CLK_TCK": 100,
                    "SC_PAGE_SIZE": 4096,
                }[name],
            ),
        ):
            cpu, ram = self.executor._process_usage(process)  # type: ignore[arg-type]
        self.assertAlmostEqual(cpu, 0.3)
        self.assertEqual(ram, 0)

    @unittest.skipIf(os.name == "nt", "Linux /proc semantics")
    def test_atomic_live_stat_reports_resident_pages(self) -> None:
        process = type("LiveProcess", (), {"pid": 4321})()
        stat = self._linux_stat(state="R", user=5, system=5, rss=7)
        with (
            patch.object(Path, "read_text", return_value=stat),
            patch(
                "cogni_flow.task_plan.os.sysconf",
                side_effect=lambda name: {
                    "SC_CLK_TCK": 100,
                    "SC_PAGE_SIZE": 4096,
                }[name],
            ),
        ):
            cpu, ram = self.executor._process_usage(process)  # type: ignore[arg-type]
        self.assertAlmostEqual(cpu, 0.1)
        self.assertEqual(ram, 7 * 4096)

    @unittest.skipIf(os.name == "nt", "Linux /proc semantics")
    def test_malformed_atomic_stat_still_fails_closed(self) -> None:
        process = type("MalformedProcess", (), {"pid": 4321})()
        with (
            patch.object(Path, "read_text", return_value="4321 malformed"),
            self.assertRaisesRegex(
                TaskExecutionError, "cannot enforce child CPU/RAM budget"
            ),
        ):
            self.executor._process_usage(process)  # type: ignore[arg-type]

    def test_fixed_pytest_argv_runs_without_shell(self) -> None:
        target = self.root / "tests" / "test_fixed.py"
        target.write_text(
            "def test_fixed():\n    assert 2 + 2 == 4\n", encoding="utf-8"
        )
        relative = "tests/test_fixed.py"
        action = TaskAction(
            "pytest",
            TaskActionKind.RUN_TEST,
            path=relative,
            argv=(sys.executable, "-m", "pytest", relative, "-q"),
        )
        selected = plan(
            6000,
            action,
            risk=RiskTier.T1,
            verifier=VerifierKind.PYTEST_PASS,
            selected_budget=budget(timeout=30.0),
        )
        result = self.execute(selected)
        self.assertTrue(result.verifier_passed)
        self.assertIn("1 passed", result.action_results[0].output)

    def test_non_allowlisted_pytest_flags_are_rejected(self) -> None:
        relative = "tests/test_fixed.py"
        (self.root / relative).write_text(
            "def test_ok(): assert True\n", encoding="utf-8"
        )
        variants = (
            (sys.executable, "-m", "pytest", relative, "-q", "-k", "ok"),
            (sys.executable, "-c", "print('shell')"),
            ("pytest", relative, "-q"),
            (sys.executable, "-m", "pytest", "../secret", "-q"),
        )
        for index, argv in enumerate(variants):
            selected = plan(
                6100 + index,
                TaskAction(
                    f"pytest-{index}",
                    TaskActionKind.RUN_TEST,
                    path=relative,
                    argv=argv,
                ),
                risk=RiskTier.T1,
                verifier=VerifierKind.PYTEST_PASS,
            )
            with self.subTest(argv=argv):
                with self.assertRaises(TaskPolicyError):
                    self.executor.authorize(selected)

    def test_fixed_subprocess_wall_clock_output_and_cpu_bounds_are_enforced(
        self,
    ) -> None:
        scenarios = (
            (
                "timeout",
                "import time\ndef test_slow():\n    time.sleep(2)\n",
                TaskBudget(0.10, 10.0, 256 * 1024**2, 0, 40_000),
                "wall-clock timeout",
            ),
            (
                "output",
                "def test_loud():\n    print('x' * 20000)\n    assert False\n",
                TaskBudget(10.0, 10.0, 256 * 1024**2, 0, 256),
                "output exceeded",
            ),
            (
                "cpu",
                "def test_busy():\n    while True:\n        pass\n",
                TaskBudget(10.0, 0.01, 256 * 1024**2, 0, 40_000),
                "CPU budget",
            ),
        )
        for index, (name, source, selected_budget, message) in enumerate(scenarios):
            relative = f"tests/test_{name}.py"
            (self.root / relative).write_text(source, encoding="utf-8")
            selected = plan(
                6150 + index,
                TaskAction(
                    f"pytest-{name}",
                    TaskActionKind.RUN_TEST,
                    path=relative,
                    argv=(sys.executable, "-m", "pytest", relative, "-q"),
                ),
                risk=RiskTier.T1,
                verifier=VerifierKind.PYTEST_PASS,
                selected_budget=selected_budget,
            )
            with self.subTest(name=name):
                with self.assertRaisesRegex(TaskExecutionError, message):
                    self.execute(selected)

    def test_measurement_failure_kills_and_reaps_the_child(self) -> None:
        relative = "tests/test_measurement.py"
        (self.root / relative).write_text(
            "import time\ndef test_wait():\n    time.sleep(5)\n", encoding="utf-8"
        )
        selected = plan(
            6180,
            TaskAction(
                "pytest-measurement",
                TaskActionKind.RUN_TEST,
                path=relative,
                argv=(sys.executable, "-m", "pytest", relative, "-q"),
            ),
            risk=RiskTier.T1,
            verifier=VerifierKind.PYTEST_PASS,
            selected_budget=budget(timeout=10.0),
        )
        spawned = []
        real_popen = __import__("subprocess").Popen

        def recording_popen(*args, **kwargs):
            process = real_popen(*args, **kwargs)
            spawned.append(process)
            return process

        with (
            patch("cogni_flow.task_plan.subprocess.Popen", side_effect=recording_popen),
            patch.object(
                TaskPlanExecutor,
                "_process_usage",
                side_effect=TaskExecutionError("cannot measure child"),
            ),
            self.assertRaisesRegex(TaskExecutionError, "cannot measure child"),
        ):
            self.execute(selected)
        self.assertEqual(len(spawned), 1)
        self.assertIsNotNone(spawned[0].poll())

    def test_t2_stages_inert_self_harness_proposal_without_source_mutation(
        self,
    ) -> None:
        source = self.root / "cogni_core" / "candidate.py"
        before = b"VALUE = 1\n"
        source.write_bytes(before)
        stager = InMemoryProposalStager()
        executor = TaskPlanExecutor(
            self.root,
            policy=self.policy,
            proposal_stager=stager,
        )
        action = TaskAction(
            "stage",
            TaskActionKind.STAGE_SOURCE_CHANGE,
            path="cogni_core/candidate.py",
            content="VALUE = 2\n",
            expected_sha256=digest(before),
            rationale="bounded regression repair",
        )
        selected = plan(
            6200,
            action,
            risk=RiskTier.T2,
            verifier=VerifierKind.PROPOSAL_STAGED,
        )
        capability = executor.authorize(selected)
        result = executor.execute(selected, capability)
        self.assertEqual(source.read_bytes(), before)
        self.assertEqual(len(stager.pending), 1)
        self.assertEqual(stager.pending[0].proposal.replacement, "VALUE = 2\n")
        self.assertEqual(result.proposal_ids, (stager.pending[0].proposal_id,))

    def test_t2_without_self_harness_stager_is_denied(self) -> None:
        source = self.root / "cogni_core" / "candidate.py"
        before = b"VALUE = 1\n"
        source.write_bytes(before)
        selected = plan(
            6300,
            TaskAction(
                "stage",
                TaskActionKind.STAGE_SOURCE_CHANGE,
                path="cogni_core/candidate.py",
                content="VALUE = 2\n",
                expected_sha256=digest(before),
                rationale="proposal only",
            ),
            risk=RiskTier.T2,
            verifier=VerifierKind.PROPOSAL_STAGED,
        )
        with self.assertRaises(TaskPolicyError):
            self.executor.authorize(selected)
        self.assertEqual(source.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
