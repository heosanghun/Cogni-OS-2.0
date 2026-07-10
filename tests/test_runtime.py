import tempfile
import unittest
from pathlib import Path

import torch
from torch import nn

from cogni_core.adaptation import (
    FastWeightSessionCache,
    LowRankOverlay,
    OverlayAcceptanceGate,
)
from cogni_core.resources import VRAMGuard
from cogni_core.routing import ContrastiveSessionRouter
from cogni_core.search import BoundedPUCTSearch, PUCTConfig, deq_tensor_transition
from cogni_flow.rhythm import SystemMode
from cogni_os.runtime import GenesisRuntime


class ToyBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(4, 4, bias=False)

    def forward(self, x):
        return (self.proj(x),)


class TestGenesisRuntime(unittest.TestCase):
    def test_inference_combines_session_backbone_and_bounded_search(self):
        torch.manual_seed(2)
        model = ToyBackbone()
        sessions = FastWeightSessionCache(
            model,
            gate=OverlayAcceptanceGate(min_quality=0.1, operator_norm_budget=0.5),
        )
        overlay = LowRankOverlay(torch.ones(4, 1) * 0.1, torch.ones(4, 1) * 0.1)
        self.assertTrue(sessions.admit("s", {"proj": overlay}, quality=0.9).accepted)
        search = BoundedPUCTSearch(
            PUCTConfig(width=3, max_depth=3, max_nodes=10, simulations=6, ancestor_k=0)
        )
        runtime = GenesisRuntime(
            model, search, sessions=sessions, vram_guard=VRAMGuard(device="cpu")
        )

        @deq_tensor_transition
        def transition(state, actions):
            return torch.stack([state + 0.01 * (int(a) + 1) for a in actions])

        def policy(state):
            return torch.tensor([0.0, 0.5, 1.0]), -state.square().mean()

        before = model.proj.weight.detach().clone()
        result = runtime.infer(
            torch.ones(1, 4), transition, policy, session_id="s", seed=1
        )
        self.assertEqual(result.search.telemetry.node_capacity, 10)
        self.assertTrue(torch.equal(before, model.proj.weight))
        self.assertEqual(runtime.rhythm.mode, SystemMode.INFERENCE)

    def test_ood_router_forces_solver_path_without_overlay(self):
        model = ToyBackbone()
        sessions = FastWeightSessionCache(
            model,
            gate=OverlayAcceptanceGate(min_quality=0.1, operator_norm_budget=0.5),
        )
        overlay = LowRankOverlay(torch.ones(4, 1) * 0.1, torch.ones(4, 1) * 0.1)
        sessions.admit("s", {"proj": overlay}, quality=0.9)
        router = ContrastiveSessionRouter()
        router.calibrate("s", torch.tensor([[1.0, 0.0], [0.99, 0.01]]))
        search = BoundedPUCTSearch(
            PUCTConfig(width=2, max_depth=1, max_nodes=3, simulations=1, ancestor_k=0)
        )
        runtime = GenesisRuntime(
            model,
            search,
            sessions=sessions,
            session_router=router,
            vram_guard=VRAMGuard(device="cpu"),
        )

        @deq_tensor_transition
        def transition(state, actions):
            return torch.stack([state, state + 0.01])

        def policy(state):
            return torch.zeros(2), torch.zeros(())

        result = runtime.infer(
            torch.ones(1, 4),
            transition,
            policy,
            session_id="s",
            routing_features=torch.tensor([0.0, 1.0]),
        )
        self.assertIsNone(result.session_id)
        self.assertFalse(result.ood.allow_fast_path)

    def test_checkpoint_is_atomic_and_hashed(self):
        model = ToyBackbone()
        runtime = GenesisRuntime(
            model,
            BoundedPUCTSearch(
                PUCTConfig(
                    width=2, max_depth=1, max_nodes=3, simulations=1, ancestor_k=0
                )
            ),
            vram_guard=VRAMGuard(device="cpu"),
        )
        with tempfile.TemporaryDirectory() as tmp:
            holder = {}

            def save():
                holder["value"] = runtime.checkpoint(tmp)

            runtime.rhythm.enter_evolution(save)
            path, digest = holder["value"]
            self.assertTrue(path.exists())
            self.assertEqual(len(digest), 64)
            self.assertEqual(
                (Path(tmp) / "genesis-state.sha256").read_text().strip(), digest
            )
            before = model.proj.weight.detach().clone()
            with torch.no_grad():
                model.proj.weight.zero_()
            runtime.restore_checkpoint(path, digest)
            self.assertTrue(torch.equal(model.proj.weight, before))
            with self.assertRaises(RuntimeError):
                runtime.restore_checkpoint(path, "0" * 64)
            runtime.rhythm.resume_inference("checkpoint test complete")


if __name__ == "__main__":
    unittest.main()
