import unittest

import torch

from cogni_core.meta_router import (
    BioHAMAMetaRouter,
    CognitiveState,
    MetaRouterConfig,
    cognitive_state_tensor,
    meta_objective,
)


class CognitiveStateTests(unittest.TestCase):
    def test_component_order_broadcasting_and_bounds(self) -> None:
        memory = torch.tensor([0.2, 1.4])
        affect = torch.tensor(-1.5)
        attention = torch.tensor([0.4, 0.6])
        uncertainty = torch.tensor([0.7, -0.2])
        load = torch.tensor(0.8)
        state = cognitive_state_tensor(memory, affect, attention, uncertainty, load)
        expected = torch.tensor(
            [[0.2, -1.0, 0.4, 0.7, 0.8], [1.0, -1.0, 0.6, 0.0, 0.8]]
        )
        self.assertTrue(torch.equal(state, expected))

    def test_wrapper_is_deterministic_and_preserves_gradient(self) -> None:
        inputs = [torch.tensor(0.1 * i, requires_grad=True) for i in range(1, 6)]
        state = CognitiveState(*inputs)
        first = state.tensor
        second = state.as_tensor()
        self.assertTrue(torch.equal(first, second))
        first.sum().backward()
        self.assertTrue(all(value.grad is not None for value in inputs))

    def test_from_tensor_round_trip(self) -> None:
        raw = torch.tensor([[0.1, -0.2, 0.3, 0.4, 0.5]])
        self.assertTrue(torch.equal(CognitiveState.from_tensor(raw).tensor, raw))


class MetaRouterConfigurationTests(unittest.TestCase):
    def test_invalid_routing_budget_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            MetaRouterConfig(num_modules=4, strategic_top_k=2, tactical_top_k=3)

    def test_invalid_neuromodulation_bounds_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            MetaRouterConfig(alpha_min=0.1, alpha_base=0.05, alpha_max=0.2)
        with self.assertRaises(ValueError):
            MetaRouterConfig(gamma_min=0.5, gamma_base=1.0, gamma_max=1.0)


class HierarchicalRoutingTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(20260711)
        self.config = MetaRouterConfig(
            num_modules=6,
            hidden_dim=24,
            strategic_top_k=4,
            tactical_top_k=2,
            reactive_top_k=3,
        )
        self.router = BioHAMAMetaRouter(self.config)

    def test_masks_are_deterministic_budgeted_and_hierarchical(self) -> None:
        state = torch.tensor([[0.5, -0.2, 0.8, 0.3, 0.4], [0.3, 0.4, 0.6, 0.2, 0.2]])
        rng_before = torch.random.get_rng_state()
        first = self.router(state)
        rng_after = torch.random.get_rng_state()
        second = self.router(state)

        self.assertTrue(torch.equal(rng_before, rng_after))
        self.assertTrue(torch.equal(first.strategic_mask, second.strategic_mask))
        self.assertTrue(torch.equal(first.tactical_mask, second.tactical_mask))
        self.assertTrue(torch.equal(first.reactive_mask, second.reactive_mask))
        self.assertTrue(
            torch.equal(first.strategic_mask.sum(-1), torch.tensor([4.0, 4.0]))
        )
        self.assertTrue(
            torch.equal(first.tactical_mask.sum(-1), torch.tensor([2.0, 2.0]))
        )
        self.assertTrue(torch.all(first.tactical_mask <= first.strategic_mask))
        self.assertTrue(torch.equal(first.reactive_mask, first.tactical_mask))
        self.assertTrue(torch.equal(first.replan_mask, torch.zeros(2)))

    def test_reactive_override_is_triggered_by_uncertainty_or_load(self) -> None:
        states = torch.tensor([[0.5, 0.0, 0.5, 0.9, 0.2], [0.5, 0.0, 0.5, 0.2, 0.95]])
        decision = self.router(states)
        self.assertTrue(torch.equal(decision.replan_mask, torch.ones(2)))
        self.assertTrue(
            torch.equal(decision.reactive_mask.sum(-1), torch.tensor([3.0, 3.0]))
        )
        self.assertTrue(torch.equal(decision.routing_mask, decision.reactive_mask))

    def test_availability_mask_is_respected_at_every_level(self) -> None:
        state = torch.tensor([0.5, 0.0, 0.5, 0.95, 0.5])
        available = torch.tensor([1, 1, 1, 1, 0, 0], dtype=torch.bool)
        decision = self.router(state, available)
        unavailable = ~available
        self.assertTrue(
            torch.equal(decision.strategic_mask[unavailable], torch.zeros(2))
        )
        self.assertTrue(
            torch.equal(decision.tactical_mask[unavailable], torch.zeros(2))
        )
        self.assertTrue(
            torch.equal(decision.reactive_mask[unavailable], torch.zeros(2))
        )

    def test_hot_path_returns_only_tensor_fields(self) -> None:
        decision = self.router.route_tensor(torch.full((2, 5), 0.5))
        for value in vars(decision).values():
            self.assertIsInstance(value, torch.Tensor)


class NeuromodulationAndObjectiveTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(41)
        self.config = MetaRouterConfig(num_modules=5, strategic_top_k=3)
        self.router = BioHAMAMetaRouter(self.config)

    def test_alpha_gamma_bounds_and_monotonic_sensitivity(self) -> None:
        states = torch.tensor(
            [
                [0.5, 0.0, 0.5, 0.0, 0.0],
                [0.5, 0.0, 0.5, 0.5, 0.5],
                [0.5, 0.0, 0.5, 1.0, 1.0],
            ]
        )
        alpha_t, gamma_t = self.router.compute_neuromodulators(states)
        self.assertTrue(torch.all(alpha_t > self.config.alpha_min))
        self.assertTrue(torch.all(alpha_t < self.config.alpha_max))
        self.assertTrue(torch.all(gamma_t > self.config.gamma_min))
        self.assertTrue(torch.all(gamma_t < self.config.gamma_max))
        self.assertTrue(torch.all(alpha_t[1:] > alpha_t[:-1]))
        self.assertTrue(torch.all(gamma_t[1:] < gamma_t[:-1]))
        self.assertAlmostEqual(
            float(alpha_t[1].detach()), self.config.alpha_base, places=7
        )
        self.assertAlmostEqual(
            float(gamma_t[1].detach()), self.config.gamma_base, places=7
        )

    def test_meta_objective_matches_mean_minus_population_variance(self) -> None:
        rewards = torch.tensor([1.0, 2.0, 5.0], requires_grad=True)
        objective = meta_objective(rewards, variance_lambda=0.25)
        expected = rewards.mean() - 0.25 * rewards.var(unbiased=False)
        self.assertTrue(torch.allclose(objective, expected))
        objective.backward()
        self.assertIsNotNone(rewards.grad)
        self.assertTrue(torch.isfinite(rewards.grad).all())

    def test_gradient_flows_through_state_all_levels_and_schedules(self) -> None:
        state = torch.tensor([[0.6, -0.1, 0.7, 0.9, 0.9]], requires_grad=True)
        decision = self.router(state)
        module_weights = torch.arange(1, self.config.num_modules + 1, dtype=state.dtype)
        loss = (
            (decision.strategic_mask * module_weights).sum()
            + (decision.tactical_mask * module_weights.square()).sum()
            + (decision.reactive_mask * module_weights.pow(3)).sum()
            + 100.0 * decision.alpha_t.sum()
            + decision.gamma_t.sum()
        )
        loss.backward()

        self.assertIsNotNone(state.grad)
        self.assertTrue(torch.isfinite(state.grad).all())
        self.assertGreater(float(state.grad.abs().sum()), 0.0)
        parameter_groups = (
            self.router.state_encoder,
            self.router.strategic_head,
            self.router.tactical_head,
            self.router.reactive_head,
            self.router.neuromodulator,
        )
        for module in parameter_groups:
            gradients = [
                parameter.grad
                for parameter in module.parameters()
                if parameter.requires_grad
            ]
            self.assertTrue(any(gradient is not None for gradient in gradients))
            self.assertTrue(
                all(
                    torch.isfinite(gradient).all()
                    for gradient in gradients
                    if gradient is not None
                )
            )


if __name__ == "__main__":
    unittest.main()
