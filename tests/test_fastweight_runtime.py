import tempfile
import unittest

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
from cogni_os.runtime import GenesisRuntime


class LatentBackbone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.proj = nn.Linear(4, 4, bias=False)

    def forward(self, values: torch.Tensor):
        return (self.proj(values),)


def make_runtime() -> GenesisRuntime:
    adapter = ResidualBottleneckAdapter(
        4, 2, core_operator_norm_budget=0.4, spectral_margin=0.95
    )
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
    wrapped = FastWeightBackboneWrapper(LatentBackbone(), adapter, programmer)
    sessions = FastWeightSessionCache(
        wrapped,
        gate=OverlayAcceptanceGate(
            min_quality=0.8,
            operator_norm_budget=0.1,
            composed_operator_norm_budget=0.95,
        ),
    )
    return GenesisRuntime(
        wrapped,
        BoundedPUCTSearch(PUCTConfig(width=2, max_depth=1, max_nodes=3, simulations=1)),
        sessions=sessions,
        session_router=ContrastiveSessionRouter(),
        fast_weight_programmer=programmer,
        fast_weight_target=FastWeightBackboneWrapper.TARGET_MODULE,
        vram_guard=VRAMGuard(device="cpu"),
    )


class TestFastWeightRuntime(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(17)
        self.runtime = make_runtime()
        self.solver_info = SolverInfo(True, 4, 1.0e-6, 0.4)

    def test_compiles_admits_and_calibrates_without_mutating_base(self):
        core = self.runtime.backbone.adapter.core
        base = core.weight.detach().clone()
        latent = torch.randn(1, 3, 4)
        result = self.runtime.compile_fast_weight_session(
            "session-a",
            latent,
            solver_info=self.solver_info,
            verified_quality=0.95,
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

    def test_rejects_unverified_nonconverged_or_batched_compilation(self):
        latent = torch.randn(1, 3, 4)
        rejected = self.runtime.compile_fast_weight_session(
            "low-quality",
            latent,
            solver_info=self.solver_info,
            verified_quality=0.2,
        )
        self.assertFalse(rejected.accepted)
        self.assertNotIn("low-quality", self.runtime.sessions.session_ids)
        with self.assertRaises(ValueError):
            self.runtime.compile_fast_weight_session(
                "not-converged",
                latent,
                solver_info=SolverInfo(False, 4, 0.1, 0.4),
                verified_quality=0.95,
            )
        with self.assertRaises(ValueError):
            self.runtime.compile_fast_weight_session(
                "batch-two",
                torch.randn(2, 3, 4),
                solver_info=self.solver_info,
                verified_quality=0.95,
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


if __name__ == "__main__":
    unittest.main()
