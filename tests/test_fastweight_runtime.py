import tempfile
import unittest
from hashlib import sha256
from pathlib import Path
from unittest.mock import patch

import torch
from torch import nn

from cogni_core.adaptation import FastWeightSessionCache, OverlayAcceptanceGate
from cogni_core.deq import SolverInfo
from cogni_core.fast_weights import (
    FastWeightBackboneWrapper,
    FastWeightProgrammer,
    ResidualBottleneckAdapter,
)
from cogni_core.resources import VRAMGuard
from cogni_core.routing import ContrastiveSessionRouter
from cogni_core.search import BoundedPUCTSearch, PUCTConfig
from cogni_core.search import deq_tensor_transition
from cogni_core.fast_weight_safety import (
    fast_weight_checkpoint_architecture,
    fast_weight_checkpoint_state,
    load_verified_fast_weight_checkpoint,
)
from cogni_os.runtime import FastWeightPathResult, GenesisRuntime


class LatentBackbone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.proj = nn.Linear(4, 4, bias=False)

    def forward(self, values: torch.Tensor):
        return (self.proj(values),)


def _write_test_checkpoint(
    directory: str,
    programmer: FastWeightProgrammer,
    adapter: ResidualBottleneckAdapter,
) -> tuple[Path, str]:
    path = Path(directory) / "fast-weight-test.pt"
    torch.save(
        {
            "schema": "cogni-fast-weight-programmer-v1",
            "architecture": fast_weight_checkpoint_architecture(programmer, adapter),
            "state": fast_weight_checkpoint_state(programmer, adapter),
            "provenance": {
                "kind": "test_fixture",
                "training_run_sha256": "a" * 64,
                "training_steps": 100,
                "training_samples": 1000,
            },
            "verifier": {
                "verifier_checkpoint_sha256": "b" * 64,
                "heldout_dataset_sha256": "c" * 64,
                "verified_quality": 0.95,
                "sample_count": 128,
                "passed": True,
            },
        },
        path,
    )
    return path, sha256(path.read_bytes()).hexdigest()


def make_runtime(directory: str) -> GenesisRuntime:
    adapter = ResidualBottleneckAdapter(
        4, 2, core_operator_norm_budget=0.4, spectral_margin=0.95
    )
    # This research fixture explicitly opts into a non-identity adapter.  The
    # product default remains exact identity until a verified checkpoint is
    # loaded.
    with torch.no_grad():
        adapter.up.weight.fill_(0.125)
    programmer = FastWeightProgrammer(
        source_dim=4,
        target_dim=2,
        internal_dim=3,
        rank=1,
        max_operator_norm=0.1,
    )
    # Prove admission uses the held-out quality supplied to the runtime rather
    # than allowing the programmer to self-certify.
    with torch.no_grad():
        programmer.quality_gate[0].weight.zero_()
        programmer.quality_gate[0].bias.fill_(-20.0)
    path, digest = _write_test_checkpoint(directory, programmer, adapter)
    verified = load_verified_fast_weight_checkpoint(
        programmer,
        adapter,
        path,
        expected_sha256=digest,
        allow_test_fixture=True,
    )
    wrapped = FastWeightBackboneWrapper(LatentBackbone(), adapter, programmer)
    router = ContrastiveSessionRouter()
    sessions = FastWeightSessionCache(
        wrapped,
        gate=OverlayAcceptanceGate(
            min_quality=0.8,
            operator_norm_budget=0.1,
            composed_operator_norm_budget=0.95,
        ),
        on_sessions_removed=router.discard_many,
        trusted_programmer_sha256=verified.programmer_evidence.checkpoint_sha256,
    )
    return GenesisRuntime(
        wrapped,
        BoundedPUCTSearch(PUCTConfig(width=2, max_depth=1, max_nodes=3, simulations=1)),
        sessions=sessions,
        session_router=router,
        fast_weight_programmer=programmer,
        verified_fast_weight=verified,
        fast_weight_target=FastWeightBackboneWrapper.TARGET_MODULE,
        vram_guard=VRAMGuard(device="cpu"),
    )


class TestFastWeightRuntime(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(17)
        self.tempdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tempdir.cleanup)
        self.runtime = make_runtime(self.tempdir.name)
        self.solver_info = SolverInfo(True, 4, 1.0e-6, 0.4)
        self.stable_trace = torch.tensor([0.1 * (0.5**index) for index in range(10)])

    def test_unverified_adapter_initializes_as_exact_identity(self):
        adapter = ResidualBottleneckAdapter(
            4, 2, core_operator_norm_budget=0.4, spectral_margin=0.95
        )
        latent = torch.randn(2, 3, 4)
        self.assertTrue(torch.equal(adapter(latent), latent))

    def test_compiles_admits_and_calibrates_without_mutating_base(self):
        core = self.runtime.backbone.adapter.core
        base = core.weight.detach().clone()
        latent = torch.randn(1, 3, 4)
        result = self.runtime.compile_fast_weight_session(
            "session-a",
            latent,
            solver_info=self.solver_info,
            residual_trace=self.stable_trace,
            calibration_features=torch.tensor([[1.0, 0.0], [0.99, 0.01], [0.98, 0.02]]),
        )
        self.assertTrue(result.accepted)
        self.assertTrue(result.calibrated)
        self.assertLess(result.programmer_quality, 0.01)
        self.assertTrue(torch.equal(base, core.weight))
        self.assertEqual(self.runtime.sessions.session_ids, ("session-a",))

        values = torch.randn(1, 3, 4)
        plain = self.runtime.backbone(values)
        with self.runtime.sessions.activate("session-a"):
            adapted = self.runtime.backbone(values)
        self.assertFalse(torch.allclose(plain, adapted))
        self.assertTrue(torch.equal(base, core.weight))

    def test_rejects_unstable_nonconverged_or_batched_compilation(self):
        latent = torch.randn(1, 3, 4)
        rejected = self.runtime.compile_fast_weight_session(
            "low-quality",
            latent,
            solver_info=self.solver_info,
            residual_trace=torch.ones(10),
        )
        self.assertFalse(rejected.accepted)
        self.assertNotIn("low-quality", self.runtime.sessions.session_ids)
        with self.assertRaises(ValueError):
            self.runtime.compile_fast_weight_session(
                "not-converged",
                latent,
                solver_info=SolverInfo(False, 4, 0.1, 0.4),
                residual_trace=self.stable_trace,
            )

    def _admit_session(self) -> None:
        result = self.runtime.compile_fast_weight_session(
            "session-a",
            torch.randn(1, 3, 4),
            solver_info=self.solver_info,
            residual_trace=self.stable_trace,
            calibration_features=torch.tensor([[1.0, 0.0], [0.99, 0.01], [0.98, 0.02]]),
        )
        self.assertTrue(result.accepted)

    def test_admitted_aq_and_ood_path_bypasses_cts_once(self):
        self._admit_session()
        calls = []

        @deq_tensor_transition
        def transition(state, actions):
            calls.append("cts")
            return torch.stack([state, state])

        def policy(state):
            return torch.zeros(2), torch.zeros(())

        core = self.runtime.backbone.adapter.core
        before = core.weight.detach().clone()
        result = self.runtime.infer(
            torch.randn(1, 3, 4),
            transition,
            policy,
            session_id="session-a",
            routing_features=torch.tensor([1.0, 0.0]),
        )
        self.assertIsInstance(result.search, FastWeightPathResult)
        self.assertEqual(calls, [])
        self.assertEqual(result.session_id, "session-a")
        self.assertTrue(result.fast_weight.activated)
        self.assertFalse(result.fast_weight.fallback_to_cts)
        self.assertEqual(result.fast_weight.reason, "admitted_aq_ood_fast_path")
        self.assertTrue(torch.equal(before, core.weight))

    def test_ood_rejection_falls_back_to_full_cts_in_same_request(self):
        self._admit_session()
        calls = []

        @deq_tensor_transition
        def transition(state, actions):
            calls.append("cts")
            return torch.stack([state, state])

        def policy(state):
            return torch.zeros(2), torch.zeros(())

        result = self.runtime.infer(
            torch.randn(1, 3, 4),
            transition,
            policy,
            session_id="session-a",
            routing_features=torch.tensor([0.0, 1.0]),
        )
        self.assertNotIsInstance(result.search, FastWeightPathResult)
        self.assertEqual(calls, ["cts"])
        self.assertIsNone(result.session_id)
        self.assertFalse(result.fast_weight.activated)
        self.assertTrue(result.fast_weight.fallback_to_cts)
        self.assertEqual(result.fast_weight.reason, "ood_rejected")

    def test_missing_session_falls_back_and_router_tracks_cache_eviction(self):
        self._admit_session()
        self.assertEqual(self.runtime.session_router.session_ids, ("session-a",))
        self.assertTrue(self.runtime.sessions.discard("session-a"))
        self.assertEqual(self.runtime.session_router.session_ids, ())

        @deq_tensor_transition
        def transition(state, actions):
            return torch.stack([state, state])

        result = self.runtime.infer(
            torch.randn(1, 3, 4),
            transition,
            lambda state: (torch.zeros(2), torch.zeros(())),
            session_id="session-a",
            routing_features=torch.tensor([1.0, 0.0]),
        )
        self.assertTrue(result.fast_weight.fallback_to_cts)
        self.assertEqual(result.fast_weight.reason, "session_missing")

    def test_calibration_failure_rolls_back_cache_and_router_publication(self):
        with patch.object(
            self.runtime.session_router,
            "calibrate",
            side_effect=RuntimeError("forced calibration failure"),
        ):
            with self.assertRaisesRegex(RuntimeError, "forced calibration"):
                self.runtime.compile_fast_weight_session(
                    "broken",
                    torch.randn(1, 3, 4),
                    solver_info=self.solver_info,
                    residual_trace=self.stable_trace,
                    calibration_features=torch.tensor([[1.0, 0.0], [0.99, 0.01]]),
                )
        self.assertNotIn("broken", self.runtime.sessions.session_ids)
        self.assertNotIn("broken", self.runtime.session_router.session_ids)

    def test_loader_is_hash_pinned_test_fixture_is_explicit_and_failure_is_atomic(self):
        programmer = FastWeightProgrammer(
            source_dim=4, target_dim=2, internal_dim=3, rank=1
        )
        adapter = ResidualBottleneckAdapter(
            4, 2, core_operator_norm_budget=0.4, spectral_margin=0.95
        )
        path, digest = _write_test_checkpoint(self.tempdir.name, programmer, adapter)
        before = programmer.trunk[0].weight.detach().clone()
        with self.assertRaisesRegex(RuntimeError, "test-only"):
            load_verified_fast_weight_checkpoint(
                programmer, adapter, path, expected_sha256=digest
            )
        self.assertTrue(torch.equal(before, programmer.trunk[0].weight))
        with self.assertRaisesRegex(RuntimeError, "SHA-256"):
            load_verified_fast_weight_checkpoint(
                programmer,
                adapter,
                path,
                expected_sha256="d" * 64,
                allow_test_fixture=True,
            )
        self.assertTrue(torch.equal(before, programmer.trunk[0].weight))
        with self.assertRaises(ValueError):
            self.runtime.compile_fast_weight_session(
                "batch-two",
                torch.randn(2, 3, 4),
                solver_info=self.solver_info,
                residual_trace=self.stable_trace,
            )

    def test_checkpoint_roundtrip_includes_adapter_and_programmer(self):
        adapter_before = self.runtime.backbone.adapter.core.weight.detach().clone()
        programmer_before = (
            self.runtime.fast_weight_programmer.trunk[0].weight.detach().clone()
        )
        with tempfile.TemporaryDirectory() as tmp:
            saved = {}
            self.runtime.rhythm.enter_evolution(
                lambda: saved.setdefault("checkpoint", self.runtime.checkpoint(tmp))
            )
            path, digest = saved["checkpoint"]
            with torch.no_grad():
                self.runtime.backbone.adapter.core.weight.zero_()
                self.runtime.fast_weight_programmer.trunk[0].weight.zero_()
            self.runtime.restore_checkpoint(path, digest)
            self.assertTrue(
                torch.equal(adapter_before, self.runtime.backbone.adapter.core.weight)
            )
            self.assertTrue(
                torch.equal(
                    programmer_before,
                    self.runtime.fast_weight_programmer.trunk[0].weight,
                )
            )
            self.assertIsNone(self.runtime.verified_fast_weight)
            self.assertFalse(self.runtime.sessions.feature_enabled)


if __name__ == "__main__":
    unittest.main()
