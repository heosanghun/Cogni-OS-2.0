import unittest

import torch

from cogni_core.experts import (
    BoundedSparseImplicitExperts,
    ExpertBudgetExceeded,
    ExpertContractivityError,
    ExpertConfig,
    MAINTENANCE_MERGED,
    RECRUITMENT_AFTER_MERGE,
)


def _config(**overrides) -> ExpertConfig:
    values = dict(
        input_dim=4,
        state_dim=5,
        router_dim=4,
        max_experts=4,
        initial_experts=2,
        min_experts=1,
        top_k=2,
        novelty_threshold=0.8,
        recruit_fraction=0.5,
        routing_temperature=0.2,
        spectral_margin=0.75,
        minimum_age=1,
        prune_usage_threshold=0.0,
        max_parameter_bytes=2**20,
        max_vram_bytes=4 * 2**20,
    )
    values.update(overrides)
    return ExpertConfig(**values)


class TestBoundedSparseExperts(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)

    def test_router_is_z_independent_top_k_sparse_and_normalized(self):
        model = BoundedSparseImplicitExperts(_config()).eval()
        x = torch.randn(6, 4)
        z_a = torch.randn(6, 5)
        z_b = torch.randn(6, 5) * 20
        route_a = model.route(x)
        output_a = model.mixture(z_a, x, route_a)
        route_b = model.route(x)
        output_b = model.mixture(z_b, x, route_b)

        self.assertTrue(torch.equal(route_a.gates, route_b.gates))
        self.assertFalse(torch.allclose(output_a, output_b))
        self.assertTrue(torch.allclose(route_a.gates.sum(-1), torch.ones(6)))
        self.assertTrue((route_a.gates.count_nonzero(-1) <= model.config.top_k).all())
        self.assertEqual(int(route_a.gates[:, ~model.active_mask].count_nonzero()), 0)

    def test_novelty_recruitment_is_deterministic(self):
        config = _config(initial_experts=1, min_experts=1, top_k=1)
        first = BoundedSparseImplicitExperts(config)
        # Construction and recruitment are independent of global RNG state.
        torch.manual_seed(9999)
        second = BoundedSparseImplicitExperts(config)
        with torch.no_grad():
            identity = torch.eye(4)
            first.router_weight.copy_(identity)
            second.router_weight.copy_(identity)
            first.prototypes[0].copy_(torch.tensor([1.0, 0.0, 0.0, 0.0]))
            second.prototypes[0].copy_(torch.tensor([1.0, 0.0, 0.0, 0.0]))
        novel = torch.tensor([[0.0, 1.0, 0.0, 0.0]]).repeat(3, 1)
        result_a = first.recruit_(novel)
        result_b = second.recruit_(novel)

        self.assertEqual(int(result_a.slot), 1)
        self.assertEqual(int(result_b.slot), 1)
        self.assertTrue(torch.equal(first.active_mask, second.active_mask))
        self.assertTrue(torch.equal(first.prototypes, second.prototypes))
        self.assertTrue(torch.equal(first.recurrent, second.recurrent))
        self.assertLess(float(first.route(novel).novelty.to(torch.float32).mean()), 0.5)

    def test_capacity_recycles_without_parameter_or_expert_growth(self):
        config = _config(
            max_experts=3,
            initial_experts=1,
            top_k=1,
            novelty_threshold=0.99,
            recruit_fraction=1.0,
            merge_on_capacity=True,
        )
        model = BoundedSparseImplicitExperts(config)
        with torch.no_grad():
            model.router_weight.copy_(torch.eye(4))
            model.prototypes[0].copy_(torch.tensor([1.0, 0.0, 0.0, 0.0]))
        original_parameter_count = sum(p.numel() for p in model.parameters())
        domains = [
            torch.tensor([[0.0, 1.0, 0.0, 0.0]]),
            torch.tensor([[0.0, 0.0, 1.0, 0.0]]),
            torch.tensor([[0.0, 0.0, 0.0, 1.0]]),
            torch.tensor([[-1.0, 0.0, 0.0, 0.0]]),
            torch.tensor([[0.0, -1.0, 0.0, 0.0]]),
        ]
        statuses = [int(model.recruit_(domain).status) for domain in domains]

        self.assertLessEqual(model.active_experts, config.max_experts)
        self.assertEqual(model.active_experts, config.max_experts)
        self.assertEqual(
            sum(p.numel() for p in model.parameters()), original_parameter_count
        )
        self.assertGreater(int(model.recruitment_count), config.max_experts)
        self.assertGreater(int(model.merge_count), 0)
        self.assertIn(RECRUITMENT_AFTER_MERGE, statuses)

    def test_load_balance_loss_detects_router_collapse(self):
        config = _config(
            max_experts=2,
            initial_experts=2,
            min_experts=1,
            top_k=1,
            novelty_threshold=-1.0,
        )
        model = BoundedSparseImplicitExperts(config)
        with torch.no_grad():
            model.router_weight.copy_(torch.eye(4))
            model.prototypes.copy_(
                torch.tensor([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])
            )
        balanced = torch.tensor([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]).repeat(
            8, 1
        )
        collapsed = torch.tensor([[1.0, 0.0, 0.0, 0.0]]).repeat(16, 1)
        balanced_loss = model.route(balanced).balance_loss
        collapsed_loss = model.route(collapsed).balance_loss

        balanced_value = float(balanced_loss.detach())
        collapsed_value = float(collapsed_loss.detach())
        self.assertAlmostEqual(balanced_value, 1.0, places=6)
        self.assertAlmostEqual(collapsed_value, 2.0, places=6)
        self.assertGreater(collapsed_value, balanced_value)

    def test_usage_prune_and_merge_are_bounded_and_deterministic(self):
        config = _config(
            max_experts=3,
            initial_experts=3,
            min_experts=1,
            top_k=1,
            minimum_age=1,
            prune_usage_threshold=0.02,
        )
        model = BoundedSparseImplicitExperts(config)
        with torch.no_grad():
            model.expert_age.fill_(5)
            model.usage_ema.copy_(torch.tensor([0.8, 0.0, 0.2]))
        pruned = model.maintain_()
        self.assertEqual(int(pruned.released_index), 1)
        self.assertEqual(model.active_experts, 2)

        # Force a merge after reactivating the free slot through novelty.
        with torch.no_grad():
            model.router_weight.copy_(torch.eye(4))
            model.prototypes[0].copy_(torch.tensor([1.0, 0.0, 0.0, 0.0]))
            model.prototypes[2].copy_(torch.tensor([0.9, 0.1, 0.0, 0.0]))
        model.recruit_(torch.tensor([[0.0, 0.0, 0.0, 1.0]]))
        with torch.no_grad():
            model.expert_age.fill_(5)
            model.usage_ema[model.active_mask] = 0.3
        merged = model.maintain_(force_merge=True)
        self.assertEqual(int(merged.action), MAINTENANCE_MERGED)
        self.assertEqual(model.active_experts, 2)

    def test_contractivity_projection_and_mixture_bound(self):
        config = _config(max_experts=3, initial_experts=3, top_k=2)
        model = BoundedSparseImplicitExperts(config).eval()
        with torch.no_grad():
            model.recurrent.mul_(50.0)
        model.project_contractivity_()
        norms = model.expert_spectral_norms()
        self.assertLessEqual(float(norms.max().detach()), config.spectral_margin + 1e-5)

        x = torch.randn(8, 4)
        routing = model.route(x)
        bound = model.routing_contractivity_bound(routing)
        self.assertTrue((bound <= config.spectral_margin + 1e-5).all())
        z_a = torch.randn(8, 5)
        z_b = torch.randn(8, 5)
        y_a = model.mixture(z_a, x, routing)
        y_b = model.mixture(z_b, x, routing)
        ratios = (y_a - y_b).norm(dim=-1) / (z_a - z_b).norm(dim=-1).clamp_min(1e-8)
        self.assertTrue((ratios <= config.spectral_margin + 1e-5).all())

    def test_mixture_enforces_cfire_immediately_before_recurrence(self):
        config = _config(max_experts=2, initial_experts=2, top_k=1)
        model = BoundedSparseImplicitExperts(config).eval()
        x = torch.randn(3, config.input_dim)
        z = torch.randn(3, config.state_dim)
        routing = model.route(x)
        with torch.no_grad():
            model.recurrent[model.active_mask].mul_(30.0)
        output = model.mixture(z, x, routing)
        self.assertTrue(torch.isfinite(output).all())
        self.assertTrue((model.expert_spectral_norms() < config.spectral_margin).all())

        with torch.no_grad():
            model.recurrent[0, 0, 0] = float("nan")
        with self.assertRaises(ExpertContractivityError):
            model.mixture(z, x, routing)

    def test_parameter_and_forward_vram_budgets_are_hard_limits(self):
        base = _config()
        required = BoundedSparseImplicitExperts.estimated_parameter_bytes(base)
        with self.assertRaises(ExpertBudgetExceeded):
            BoundedSparseImplicitExperts(_config(max_parameter_bytes=required - 1))

        probe = BoundedSparseImplicitExperts(base)
        tight_vram = probe.persistent_bytes + probe.estimated_working_set_bytes(
            1, include_backward=True
        )
        constrained = BoundedSparseImplicitExperts(_config(max_vram_bytes=tight_vram))
        z = torch.zeros(2, 5)
        x = torch.zeros(2, 4)
        with self.assertRaises(ExpertBudgetExceeded):
            constrained(z, x)


if __name__ == "__main__":
    unittest.main()
