from dataclasses import replace
from hashlib import sha256
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
from cogni_core.fp_ewc import FixedPointFisherConfig
from cogni_core.routing import ContrastiveSessionRouter
from cogni_core.swarm import SwarmConfig, TensorSwarm
from cogni_core.swarm_sessions import SwarmSessionStateCache
from cogni_core.search import BoundedPUCTSearch, PUCTConfig, deq_tensor_transition
from cogni_core.search import (
    BoundedPUCTSearchV2,
    CertifiedBroydenTransitionV2,
    CertifiedPUCTConfigV2,
    PUCTResultV2,
    SearchControlsV2,
    SearchRequestV2,
)
from cogni_flow.rhythm import SystemMode
from cogni_os.runtime import GenesisRuntime, SearchCollaboratorsV2


class ToyBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(4, 4, bias=False)

    def forward(self, x):
        return (self.proj(x),)


class TestGenesisRuntime(unittest.TestCase):
    @staticmethod
    def _legacy_search() -> BoundedPUCTSearch:
        return BoundedPUCTSearch(
            PUCTConfig(
                width=2,
                max_depth=1,
                max_nodes=3,
                simulations=1,
                ancestor_k=0,
            )
        )

    def test_v2_builds_exact_request_and_preserves_certified_call_order(self):
        events = []
        model = ToyBackbone()
        model.register_forward_hook(lambda *_args: events.append("backbone"))

        class RecordingSearch(BoundedPUCTSearchV2):
            request = None

            def search(
                self, request, transition, action_policy, critic, meta_controller
            ):
                self.request = request
                events.append("search")
                return super().search(
                    request, transition, action_policy, critic, meta_controller
                )

        class RecordingTransition(CertifiedBroydenTransitionV2):
            def __call__(self, *args, **kwargs):
                events.append("transition")
                return super().__call__(*args, **kwargs)

        def action_policy(state):
            events.append("policy")
            return torch.zeros(3, device=state.device, dtype=torch.float32)

        def critic(state):
            events.append("critic")
            return state.new_zeros((), dtype=torch.float32)

        def meta_controller(_root):
            events.append("meta")
            return SearchControlsV2(0.5, 5.0e-3, 1.0, 1)

        search = RecordingSearch(
            CertifiedPUCTConfigV2(
                max_depth=1,
                simulations=1,
                meta_policy_macs=1,
                action_policy_macs=1,
                critic_macs=1,
                retrieval_macs=1,
                transition_macs=1,
            )
        )
        runtime = GenesisRuntime(
            model,
            search,
            search_mac_budget=4,
            vram_guard=VRAMGuard(device="cpu"),
        )
        result = runtime.infer(
            torch.ones(1, 4),
            RecordingTransition(max_iter=17),
            SearchCollaboratorsV2(action_policy, critic, meta_controller),
            seed=7,
        )
        self.assertIsInstance(search.request, SearchRequestV2)
        self.assertEqual(search.request.mac_budget, 4)
        self.assertEqual(search.request.seed, 7)
        self.assertIsInstance(result.search, PUCTResultV2)
        self.assertEqual(
            events,
            ["backbone", "search", "meta", "policy", "critic", "transition"],
        )

    def test_typed_empirical_fisher_is_night_only_and_preserves_sample_count(self):
        runtime = GenesisRuntime(
            ToyBackbone(),
            self._legacy_search(),
            vram_guard=VRAMGuard(device="cpu"),
        )
        parameter = runtime.backbone.proj.weight
        options = dict(
            domain_id="three-sample-domain",
            f_at_z=lambda z: 0.2 * z,
            z_star=torch.zeros(3, 4),
            log_likelihood_per_sample=lambda z: z.sum(dim=1) + parameter.sum(),
            named_parameters=[("proj.weight", parameter)],
            config=FixedPointFisherConfig(
                contraction_bound=0.2,
                fixed_point_tolerance=1.0e-7,
                adjoint_tolerance=1.0e-7,
            ),
            solver_converged=True,
        )
        with self.assertRaisesRegex(RuntimeError, "evolution"):
            runtime.consolidate_empirical_domain(**options)

        runtime.rhythm.enter_evolution(lambda: None)
        estimate, snapshot = runtime.consolidate_empirical_domain(**options)

        self.assertEqual(estimate.n_samples, 3)
        self.assertEqual(snapshot.n_samples, 3)
        self.assertTrue(torch.isfinite(snapshot.fisher["proj.weight"]).all())

    def test_v2_unsafe_telemetry_fails_closed(self):
        class UnsafeSearch(BoundedPUCTSearchV2):
            def __init__(self, field, value):
                super().__init__(
                    CertifiedPUCTConfigV2(
                        max_depth=1,
                        simulations=1,
                        meta_policy_macs=1,
                        action_policy_macs=1,
                        critic_macs=1,
                        retrieval_macs=1,
                        transition_macs=1,
                    )
                )
                self.field = field
                self.value = value

            def search(self, *args, **kwargs):
                result = super().search(*args, **kwargs)
                telemetry = replace(result.telemetry, **{self.field: self.value})
                return replace(result, telemetry=telemetry)

        collaborators = SearchCollaboratorsV2(
            lambda state: torch.zeros(3, device=state.device),
            lambda state: state.new_zeros(()),
            lambda _root: SearchControlsV2(0.5, 5.0e-3, 1.0, 1),
        )
        cases = (
            ("safe_for_decode", False),
            ("linear_solve_fallbacks", 1),
            ("unsafe_silent_fallbacks", 1),
        )
        for field, value in cases:
            with self.subTest(field=field):
                runtime = GenesisRuntime(
                    ToyBackbone(),
                    UnsafeSearch(field, value),
                    search_mac_budget=4,
                    vram_guard=VRAMGuard(device="cpu"),
                )
                with self.assertRaisesRegex(RuntimeError, "unsafe search telemetry"):
                    runtime.infer(
                        torch.ones(1, 4),
                        CertifiedBroydenTransitionV2(max_iter=17),
                        collaborators,
                    )

    def test_unverified_research_session_cannot_bypass_bounded_search(self):
        torch.manual_seed(2)
        model = ToyBackbone()
        sessions = FastWeightSessionCache(
            model,
            gate=OverlayAcceptanceGate(min_quality=0.1, operator_norm_budget=0.5),
            allow_unverified_research=True,
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
        self.assertTrue(result.fast_weight.fallback_to_cts)
        self.assertEqual(result.fast_weight.reason, "verified_checkpoint_missing")
        self.assertTrue(torch.equal(before, model.proj.weight))
        self.assertEqual(runtime.rhythm.mode, SystemMode.INFERENCE)

    def test_unverified_session_falls_back_before_ood_router(self):
        model = ToyBackbone()
        sessions = FastWeightSessionCache(
            model,
            gate=OverlayAcceptanceGate(min_quality=0.1, operator_norm_budget=0.5),
            allow_unverified_research=True,
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
        self.assertIsNone(result.ood)
        self.assertTrue(result.fast_weight.fallback_to_cts)
        self.assertEqual(result.fast_weight.reason, "verified_checkpoint_missing")

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

    def test_system4_runtime_uses_bounded_independent_session_state(self):
        swarm = TensorSwarm(SwarmConfig(input_dim=4, state_dim=8)).eval()
        cache = SwarmSessionStateCache(max_sessions=2, ttl_seconds=60.0)
        runtime = GenesisRuntime(
            ToyBackbone(),
            self._legacy_search(),
            swarm=swarm,
            swarm_sessions=cache,
            vram_guard=VRAMGuard(device="cpu"),
        )
        first = runtime.adapt_stream(torch.full((1, 4), 0.01), session_id="alpha")
        second = runtime.adapt_stream(torch.full((1, 4), -0.01), session_id="beta")
        self.assertTrue(bool(first.safe_for_advice))
        self.assertTrue(bool(second.safe_for_advice))
        self.assertEqual(cache.session_count, 2)
        alpha = cache.get("alpha")
        beta = cache.get("beta")
        assert alpha is not None and beta is not None
        self.assertFalse(torch.equal(alpha.joint_state, beta.joint_state))
        with self.assertRaises(ValueError):
            runtime.adapt_stream(torch.zeros(1, 4), session_id="unsafe/session")

    def test_checkpoint_carries_topology_certificate_not_ephemeral_sessions(self):
        swarm = TensorSwarm(SwarmConfig(input_dim=4, state_dim=8)).eval()
        cache = SwarmSessionStateCache(max_sessions=2, ttl_seconds=60.0)
        runtime = GenesisRuntime(
            ToyBackbone(),
            self._legacy_search(),
            swarm=swarm,
            swarm_sessions=cache,
            vram_guard=VRAMGuard(device="cpu"),
        )
        runtime.adapt_stream(torch.zeros(1, 4), session_id="primary")
        self.assertEqual(cache.session_count, 1)
        with tempfile.TemporaryDirectory() as tmp:
            saved = {}
            runtime.rhythm.enter_evolution(
                lambda: saved.setdefault("checkpoint", runtime.checkpoint(tmp))
            )
            path, digest = saved["checkpoint"]
            payload = torch.load(path, map_location="cpu", weights_only=True)
            self.assertNotIn("swarm_state", payload)
            self.assertEqual(
                set(payload["swarm_topology_certificates"]), {"normal", "crisis"}
            )
            runtime.restore_checkpoint(path, digest)
            self.assertEqual(cache.session_count, 0)

            payload["swarm_topology_certificates"]["normal"]["sha256"] = "0" * 64
            tampered = Path(tmp) / "tampered.pt"
            torch.save(payload, tampered)
            tampered_digest = sha256(tampered.read_bytes()).hexdigest()
            with self.assertRaisesRegex(RuntimeError, "topology certificate"):
                runtime.restore_checkpoint(tampered, tampered_digest)


if __name__ == "__main__":
    unittest.main()
