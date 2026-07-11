from __future__ import annotations

from contextlib import contextmanager
from types import SimpleNamespace
import unittest

import torch
from torch import Tensor

from cogni_agent.core_pipeline import (
    CorePipelineLimits,
    CoreTurnPipeline,
    CoreTurnRequest,
    FastWeightActivation,
    FastWeightCompilationPlan,
)
from cogni_core.deq import SolverInfo
from cogni_core.experts import BoundedSparseImplicitExperts, ExpertConfig
from cogni_core.resources import VRAMGuard
from cogni_core.search import BoundedPUCTSearch, PUCTConfig
from cogni_os.runtime import GenesisRuntime


class _Transition:
    def __init__(self, *, converged: bool = True) -> None:
        self.last_info = SolverInfo(converged, 4, 1.0e-6, 0.4)

    def __call__(self, state: Tensor, actions: Tensor) -> Tensor:
        return torch.stack([state for _ in actions])


class _Routing:
    def __init__(self, batch: int) -> None:
        self._mask = torch.ones(batch, 8)

    @property
    def routing_mask(self) -> Tensor:
        return self._mask


class _FakeRhythm:
    def __init__(self) -> None:
        self.active_requests = 0
        self.max_active_requests = 0

    @contextmanager
    def inference_slot(self):
        self.active_requests += 1
        self.max_active_requests = max(self.max_active_requests, self.active_requests)
        try:
            yield
        finally:
            self.active_requests -= 1


class _FakeRuntime:
    """Deterministic runtime port that records the product call graph."""

    def __init__(self) -> None:
        self.calls: list[str] = []
        self.infer_kwargs: dict[str, object] = {}
        self.compile_kwargs: dict[str, object] = {}
        self.answer_state = torch.tensor([[[1.0, 2.0, 3.0, 4.0], [3.0, 4.0, 5.0, 6.0]]])
        self.rhythm = _FakeRhythm()
        self.sessions = SimpleNamespace(session_ids=("verified_session",))
        self.session_router = object()

    def route_cognitive_state(
        self, state: Tensor, available_mask: Tensor | None = None
    ) -> _Routing:
        self.calls.append("bio_hama")
        self.routed_state = state.clone()
        self.available_mask = available_mask
        return _Routing(state.shape[0])

    def infer(self, inputs: Tensor, transition, policy_value, **kwargs):
        self.calls.append("cts_deq")
        self.infer_kwargs = dict(kwargs)
        session_id = kwargs.get("session_id")
        return SimpleNamespace(
            backbone_state=self.answer_state.clone(),
            search=SimpleNamespace(best_state=self.answer_state.clone()),
            session_id=session_id,
            ood=None,
        )

    def adapt_stream(self, observations: Tensor):
        self.calls.append("system4")
        self.swarm_observation = observations.clone()
        batch = observations.shape[0]
        latent = observations.new_full((batch, 3), 7.0)
        return SimpleNamespace(
            latent=latent,
            joint_state=observations.new_zeros(batch, 2, 3),
            regime=observations.new_zeros((), dtype=torch.long),
            residual=observations.new_zeros(batch),
            iterations=observations.new_tensor(2, dtype=torch.long),
        )

    def expert_step(
        self, z: Tensor, x: Tensor, *, track_usage: bool = True
    ) -> SimpleNamespace:
        self.calls.append("system3")
        self.expert_z = z.clone()
        self.expert_x = x.clone()
        self.track_usage = track_usage
        return SimpleNamespace(
            state=x.new_full((x.shape[0], 3), -999.0),
            routing=SimpleNamespace(novelty=x.new_zeros(x.shape[0], dtype=torch.bool)),
        )

    def compile_fast_weight_session(
        self,
        session_id: str,
        z_star: Tensor,
        **kwargs,
    ) -> SimpleNamespace:
        self.calls.append("fast_weight_compile")
        self.compile_session_id = session_id
        self.compile_state = z_star.clone()
        self.compile_kwargs = dict(kwargs)
        return SimpleNamespace(accepted=True, calibrated=True)

    def consolidate_domain(self, *args, **kwargs):  # pragma: no cover - tripwire
        self.calls.append("fp_ewc")
        raise AssertionError("FP-EWC must never run in a conversational turn")


class _MixedPrecisionRuntime(_FakeRuntime):
    def __init__(self) -> None:
        super().__init__()
        self.answer_state = self.answer_state.to(torch.bfloat16)
        self.swarm = torch.nn.Linear(4, 4).double()
        self.experts = torch.nn.Linear(4, 4).double()

    def adapt_stream(self, observations: Tensor):
        if observations.dtype != torch.float64:
            raise AssertionError("System4 input was not cast to its parameter dtype")
        return super().adapt_stream(observations)

    def expert_step(
        self, z: Tensor, x: Tensor, *, track_usage: bool = True
    ) -> SimpleNamespace:
        if z.dtype != torch.float64 or x.dtype != torch.float64:
            raise AssertionError("System3 inputs were not cast to its parameter dtype")
        return super().expert_step(z, x, track_usage=track_usage)


def _policy(state: Tensor) -> tuple[Tensor, Tensor]:
    return state.new_zeros(2), state.mean()


def _request(**overrides: object) -> CoreTurnRequest:
    values: dict[str, object] = {
        "inputs": torch.tensor([[1, 2]], dtype=torch.long),
        "cognitive_state": torch.tensor([[0.2, 0.0, 0.5, 0.1, 0.3]]),
        "backbone_kwargs": {"attention_mask": torch.ones(1, 2, dtype=torch.long)},
        "estimated_workspace_bytes": 1024,
        "seed": 5,
    }
    values.update(overrides)
    return CoreTurnRequest(**values)


class TestCoreTurnPipeline(unittest.TestCase):
    def test_default_turn_keeps_untrained_auxiliaries_advisory_only(self) -> None:
        runtime = _FakeRuntime()
        pipeline = CoreTurnPipeline(runtime, _Transition(), _policy)

        result = pipeline.run(_request())

        self.assertEqual(runtime.calls, ["bio_hama", "cts_deq", "system4", "system3"])
        self.assertNotIn("fp_ewc", runtime.calls)
        self.assertTrue(
            torch.equal(result.inference.backbone_state, runtime.answer_state)
        )
        self.assertTrue(
            torch.equal(
                result.pooled_observation,
                torch.tensor([[2.0, 3.0, 4.0, 5.0]]),
            )
        )
        self.assertTrue(result.telemetry.advisory_only)
        self.assertTrue(
            torch.equal(result.telemetry.advisory_state, torch.full((1, 3), -999.0))
        )
        self.assertFalse(runtime.track_usage)
        self.assertEqual(runtime.rhythm.active_requests, 0)
        self.assertEqual(runtime.rhythm.max_active_requests, 1)
        self.assertIsNone(runtime.infer_kwargs["session_id"])
        self.assertIsNone(runtime.infer_kwargs["routing_features"])
        self.assertIsNone(result.telemetry.fast_weight_compilation)

    def test_same_fake_runtime_inputs_have_deterministic_tensor_results(self) -> None:
        first = CoreTurnPipeline(_FakeRuntime(), _Transition(), _policy).run(_request())
        second = CoreTurnPipeline(_FakeRuntime(), _Transition(), _policy).run(
            _request()
        )

        self.assertTrue(
            torch.equal(first.inference.backbone_state, second.inference.backbone_state)
        )
        self.assertTrue(
            torch.equal(first.telemetry.swarm.latent, second.telemetry.swarm.latent)
        )
        self.assertTrue(
            torch.equal(first.telemetry.advisory_state, second.telemetry.advisory_state)
        )

    def test_bf16_backbone_is_cast_only_at_fp32_advisory_boundaries(self) -> None:
        runtime = _MixedPrecisionRuntime()

        result = CoreTurnPipeline(runtime, _Transition(), _policy).run(_request())

        self.assertEqual(result.inference.backbone_state.dtype, torch.bfloat16)
        self.assertEqual(result.pooled_observation.dtype, torch.bfloat16)
        self.assertEqual(result.telemetry.swarm.latent.dtype, torch.float64)
        self.assertEqual(result.telemetry.experts.state.dtype, torch.float64)

    def test_fast_weight_requires_explicit_ood_features_and_compiles_for_next_turn(
        self,
    ) -> None:
        runtime = _FakeRuntime()
        transition = _Transition()
        features = torch.tensor([[0.8, 0.2]])
        calibration = torch.tensor([[1.0, 0.0], [0.99, 0.01], [0.98, 0.02]])
        request = _request(
            fast_weight=FastWeightActivation("verified_session", features),
            compile_fast_weight=FastWeightCompilationPlan(
                "next_session", 0.93, calibration
            ),
        )

        result = CoreTurnPipeline(runtime, transition, _policy).run(request)

        self.assertEqual(
            runtime.calls,
            [
                "bio_hama",
                "cts_deq",
                "system4",
                "system3",
                "fast_weight_compile",
            ],
        )
        self.assertEqual(runtime.infer_kwargs["session_id"], "verified_session")
        self.assertIs(runtime.infer_kwargs["routing_features"], features)
        self.assertEqual(runtime.compile_session_id, "next_session")
        self.assertTrue(torch.equal(runtime.compile_state, runtime.answer_state))
        self.assertIs(runtime.compile_kwargs["solver_info"], transition.last_info)
        self.assertEqual(runtime.compile_kwargs["verified_quality"], 0.93)
        self.assertIs(runtime.compile_kwargs["calibration_features"], calibration)
        self.assertTrue(result.telemetry.fast_weight_compilation.accepted)

    def test_nonconverged_transition_cannot_compile_fast_weight(self) -> None:
        runtime = _FakeRuntime()
        plan = FastWeightCompilationPlan(
            "future", 0.9, torch.tensor([[1.0, 0.0], [0.9, 0.1]])
        )
        pipeline = CoreTurnPipeline(runtime, _Transition(converged=False), _policy)

        with self.assertRaisesRegex(RuntimeError, "converged DEQ evidence"):
            pipeline.run(_request(compile_fast_weight=plan))
        self.assertNotIn("fast_weight_compile", runtime.calls)
        self.assertNotIn("fp_ewc", runtime.calls)

    def test_request_validation_fails_before_any_runtime_work(self) -> None:
        limits = CorePipelineLimits(max_sequence_length=2)
        bad_requests = (
            _request(inputs=torch.ones(1, 3, dtype=torch.long)),
            _request(
                cognitive_state=torch.tensor([[0.0, 0.0, float("nan"), 0.0, 0.0]])
            ),
            _request(backbone_kwargs={"attention_mask": "not-a-tensor"}),
            _request(fast_weight=FastWeightActivation("unsafe/session", torch.ones(2))),
        )
        for request in bad_requests:
            with self.subTest(request=request):
                runtime = _FakeRuntime()
                pipeline = CoreTurnPipeline(
                    runtime, _Transition(), _policy, limits=limits
                )
                with self.assertRaises((TypeError, ValueError)):
                    pipeline.run(request)
                self.assertEqual(runtime.calls, [])

    def test_unadmitted_fast_weight_session_fails_before_day_slot(self) -> None:
        runtime = _FakeRuntime()
        request = _request(
            fast_weight=FastWeightActivation("unknown_session", torch.ones(1, 2))
        )

        with self.assertRaisesRegex(RuntimeError, "not admitted"):
            CoreTurnPipeline(runtime, _Transition(), _policy).run(request)

        self.assertEqual(runtime.calls, [])
        self.assertEqual(runtime.rhythm.max_active_requests, 0)


class TestGenesisExpertAdvisoryMode(unittest.TestCase):
    def test_runtime_can_observe_system3_without_mutating_usage(self) -> None:
        experts = BoundedSparseImplicitExperts(
            ExpertConfig(
                input_dim=4,
                state_dim=3,
                router_dim=2,
                max_experts=2,
                initial_experts=1,
                top_k=1,
                max_parameter_bytes=1_000_000,
                max_vram_bytes=1_000_000,
            )
        )
        runtime = GenesisRuntime(
            torch.nn.Identity(),
            BoundedPUCTSearch(
                PUCTConfig(
                    width=2,
                    max_depth=1,
                    max_nodes=3,
                    simulations=1,
                    ancestor_k=0,
                )
            ),
            experts=experts,
            vram_guard=VRAMGuard(device="cpu"),
        )
        before = experts.usage_ema.clone()

        output = runtime.expert_step(
            torch.zeros(1, 3), torch.ones(1, 4), track_usage=False
        )

        self.assertEqual(tuple(output.state.shape), (1, 3))
        self.assertTrue(torch.equal(experts.usage_ema, before))


if __name__ == "__main__":
    unittest.main()
