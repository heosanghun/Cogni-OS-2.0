import unittest

import torch
from torch import nn

from cogni_core.adaptation import (
    FastWeightSessionCache,
    FixedPointDomainLifecycle,
    LowRankOverlay,
    OverlayAcceptanceGate,
    low_rank_operator_norm,
)


class _ToyModel(nn.Module):
    def __init__(self, dim: int = 4):
        super().__init__()
        self.proj = nn.Linear(dim, dim, bias=False)

    def forward(self, x):
        return self.proj(x)


def _small_overlay(dim: int = 4, scale: float = 0.05) -> LowRankOverlay:
    a = torch.zeros(dim, 1)
    b = torch.zeros(dim, 1)
    a[0, 0] = scale
    b[0, 0] = scale
    return LowRankOverlay(a, b)


class TestFastWeightSessions(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(12)

    def test_rank_space_operator_norm_matches_dense_update(self):
        overlay = LowRankOverlay(torch.randn(9, 3), torch.randn(7, 3))
        actual = low_rank_operator_norm(overlay)
        expected = torch.linalg.matrix_norm(overlay.b @ overlay.a.T, ord=2)
        self.assertTrue(torch.allclose(actual, expected, atol=1e-5, rtol=1e-5))

    def test_acceptance_gate_checks_quality_finiteness_and_operator_budget(self):
        safe = _small_overlay(scale=0.1)
        gate = OverlayAcceptanceGate(min_quality=0.7, operator_norm_budget=0.02)
        self.assertFalse(gate.evaluate({"proj": safe}, quality=0.69).accepted)
        self.assertTrue(gate.evaluate({"proj": safe}, quality=0.7).accepted)

        unsafe = LowRankOverlay(torch.eye(4), torch.eye(4))
        decision = gate.evaluate({"proj": unsafe}, quality=0.99)
        self.assertFalse(decision.accepted)
        self.assertIn("operator budget", decision.reason)

        nonfinite = LowRankOverlay(torch.full((4, 1), float("nan")), torch.ones(4, 1))
        self.assertFalse(gate.evaluate({"proj": nonfinite}, quality=0.99).accepted)

    def test_optional_composed_weight_bound_is_checked_without_dense_delta(self):
        model = _ToyModel()
        with torch.no_grad():
            model.proj.weight.copy_(torch.eye(4) * 0.94)
        magnitude = 0.02**0.5
        overlay = LowRankOverlay(
            torch.eye(4) * magnitude,
            torch.eye(4) * magnitude,
        )
        gate = OverlayAcceptanceGate(
            operator_norm_budget=0.1,
            composed_operator_norm_budget=0.95,
        )
        cache = FastWeightSessionCache(model, gate=gate)
        rejected = cache.admit("unsafe-composition", {"proj": overlay}, quality=0.9)
        self.assertFalse(rejected.accepted)
        self.assertIn("composed operator budget", rejected.decision.reason)
        self.assertGreater(rejected.decision.max_composed_operator_norm, 0.95)

        with torch.no_grad():
            model.proj.weight.copy_(torch.eye(4) * 0.5)
        accepted = cache.admit("safe-composition", {"proj": overlay}, quality=0.9)
        self.assertTrue(accepted.accepted)
        self.assertLessEqual(
            accepted.decision.max_composed_operator_norm,
            gate.composed_operator_norm_budget,
        )

    def test_activation_changes_output_but_never_mutates_base_weight(self):
        model = _ToyModel()
        cache = FastWeightSessionCache(
            model,
            gate=OverlayAcceptanceGate(min_quality=0.5, operator_norm_budget=0.1),
        )
        overlay = _small_overlay(scale=0.2)
        self.assertTrue(
            cache.admit("session-a", {"proj": overlay}, quality=0.9).accepted
        )

        x = torch.randn(3, 4)
        base_weight = model.proj.weight.detach().clone()
        base_output = model(x)
        with cache.activate("session-a"):
            adapted = model(x)
            self.assertTrue(torch.equal(model.proj.weight, base_weight))
            expected_update = torch.nn.functional.linear(
                torch.nn.functional.linear(x, overlay.a.T), overlay.b
            )
            self.assertTrue(torch.allclose(adapted, base_output + expected_update))
        self.assertTrue(torch.equal(model.proj.weight, base_weight))
        self.assertTrue(torch.allclose(model(x), base_output))

    def test_activation_cleanup_is_exception_safe(self):
        model = _ToyModel()
        cache = FastWeightSessionCache(model)
        cache.admit("session-a", {"proj": _small_overlay()}, quality=0.9)
        x = torch.randn(1, 4)
        expected = model(x)
        with self.assertRaisesRegex(RuntimeError, "deliberate"):
            with cache.activate("session-a"):
                raise RuntimeError("deliberate failure")
        self.assertTrue(torch.allclose(model(x), expected))
        # A removed hook and cleared active marker allow immediate reuse.
        with cache.activate("session-a"):
            self.assertFalse(torch.allclose(model(x), expected))

    def test_lru_eviction_obeys_count_and_byte_budgets(self):
        model = _ToyModel()
        overlay = _small_overlay()
        cache = FastWeightSessionCache(
            model, max_sessions=2, max_bytes=overlay.nbytes * 2
        )
        cache.admit("one", {"proj": overlay}, quality=0.9)
        cache.admit("two", {"proj": overlay}, quality=0.9)
        cache.get("one")  # two becomes least recently used.
        result = cache.admit("three", {"proj": overlay}, quality=0.9)
        self.assertEqual(result.evicted, ("two",))
        self.assertEqual(cache.session_ids, ("one", "three"))
        self.assertLessEqual(cache.total_bytes, cache.max_bytes)

        too_large = LowRankOverlay(torch.ones(4, 4), torch.ones(4, 4) * 0.001)
        # It passes the operator gate but is larger than the entire byte budget.
        rejected = cache.admit("huge", {"proj": too_large}, quality=0.9)
        self.assertFalse(rejected.accepted)
        self.assertIn("byte budget", rejected.decision.reason)


class TestFixedPointDomainLifecycle(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(21)

    def test_consolidate_preserves_diagonal_fisher_and_penalizes_drift(self):
        layer = nn.Linear(2, 2, bias=False)
        lifecycle = FixedPointDomainLifecycle(strength=2.0)
        fisher = {"weight": torch.full_like(layer.weight, 0.25)}
        lifecycle.consolidate("domain-1", layer.named_parameters(), fisher)
        self.assertTrue(
            torch.equal(
                lifecycle.regularizer.snapshots[0].fisher["weight"], fisher["weight"]
            )
        )
        self.assertEqual(
            float(lifecycle.penalty(layer.named_parameters()).detach()), 0.0
        )
        with torch.no_grad():
            layer.weight.add_(0.1)
        # 0.5 * strength(2) * four entries * fisher(.25) * drift^2(.01)
        self.assertAlmostEqual(
            float(lifecycle.penalty(layer.named_parameters()).detach()), 0.01, places=6
        )

    def test_matrix_free_fixed_point_fisher_is_consolidated(self):
        recurrent = nn.Parameter(torch.eye(3) * 0.35)
        drive = torch.randn(1, 3)
        z = torch.zeros_like(drive)
        for _ in range(100):
            z = torch.tanh(z @ recurrent.T + drive)

        lifecycle = FixedPointDomainLifecycle(strength=1.0)
        snapshot = lifecycle.estimate_and_consolidate(
            "implicit-domain",
            f_at_z=lambda state: torch.tanh(state @ recurrent.T + drive),
            z_star=z,
            log_likelihood_at_z=lambda state: -0.5 * state.square().sum(),
            named_parameters=[("recurrent", recurrent)],
        )
        self.assertEqual(lifecycle.n_consolidated, 1)
        self.assertTrue(torch.isfinite(snapshot.fisher["recurrent"]).all())
        self.assertGreater(float(snapshot.fisher["recurrent"].sum()), 0.0)
        with torch.no_grad():
            recurrent.add_(0.01)
        self.assertGreater(
            float(lifecycle.penalty([("recurrent", recurrent)]).detach()), 0.0
        )

    def test_domain_budget_merges_old_quadratics_at_fixed_memory(self):
        layer = nn.Linear(2, 2, bias=False)
        lifecycle = FixedPointDomainLifecycle(max_domains=2)
        for index in range(5):
            with torch.no_grad():
                layer.weight.fill_(index * 0.1)
            lifecycle.consolidate(
                f"domain-{index}",
                layer.named_parameters(),
                {"weight": torch.ones_like(layer.weight) * (index + 1)},
            )
        self.assertEqual(lifecycle.n_consolidated, 2)
        self.assertEqual(len(lifecycle.domains), 2)
        penalty = lifecycle.penalty(layer.named_parameters())
        self.assertTrue(torch.isfinite(penalty))

    def test_spectral_projection_occurs_before_optimizer_step(self):
        parameter = nn.Parameter(torch.eye(3) * 5.0)
        optimizer = torch.optim.SGD([parameter], lr=1.0)
        parameter.grad = torch.full_like(parameter, -10.0)
        seen_at_step = []
        original_step = optimizer.step

        def observed_step(closure=None):
            seen_at_step.append(
                float(torch.linalg.matrix_norm(parameter.detach(), ord=2))
            )
            return original_step(closure=closure)

        optimizer.step = observed_step  # type: ignore[method-assign]
        lifecycle = FixedPointDomainLifecycle(spectral_margin=0.8)
        _, report = lifecycle.optimizer_step(optimizer, [("recurrent", parameter)])
        self.assertLessEqual(seen_at_step[0], 0.8001)
        self.assertLessEqual(report.before_step["recurrent"], 0.8001)
        self.assertLessEqual(report.after_step["recurrent"], 0.8001)
        self.assertLessEqual(
            float(torch.linalg.matrix_norm(parameter.detach(), ord=2)), 0.8001
        )

    def test_batched_matrix_parameters_are_projected_independently(self):
        parameter = nn.Parameter(torch.randn(2, 3, 4, 4) * 20.0)
        lifecycle = FixedPointDomainLifecycle(spectral_margin=0.73)
        report = lifecycle.project_spectral_([("batched", parameter)])
        norms = torch.linalg.matrix_norm(parameter.detach(), ord=2)
        self.assertEqual(tuple(norms.shape), (2, 3))
        self.assertTrue((norms < lifecycle.spectral_margin).all())
        self.assertAlmostEqual(report["batched"], float(norms.max()), places=6)

    def test_optimizer_post_projection_cannot_be_disabled(self):
        parameter = nn.Parameter(torch.eye(2) * 0.2)
        optimizer = torch.optim.SGD([parameter], lr=1.0)
        lifecycle = FixedPointDomainLifecycle(spectral_margin=0.8)
        with self.assertRaises(TypeError):
            lifecycle.optimizer_step(
                optimizer,
                [("recurrent", parameter)],
                project_after=False,  # type: ignore[call-arg]
            )


if __name__ == "__main__":
    unittest.main()
