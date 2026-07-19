from dataclasses import replace
import math
import unittest
from unittest.mock import patch

import torch

from cogni_core.deq import (
    BroydenTelemetry,
    BroydenWarmStart,
    DEQConfig,
    EquilibriumLayer,
    LimitedBroydenResult,
    _broyden_inverse,
    limited_broyden_solve,
    normalized_residual,
)


class TestBoundedBroydenSolver(unittest.TestCase):
    def test_cpu_bfloat16_uses_fp32_small_solve_without_silent_fallback(
        self,
    ) -> None:
        z0 = torch.zeros(1, 4, dtype=torch.bfloat16)
        drive = torch.tensor([[0.2, -0.1, 0.3, -0.2]], dtype=torch.bfloat16)

        def root(z: torch.Tensor) -> torch.Tensor:
            return torch.tanh(0.35 * z + drive) - z

        native_solve = torch.linalg.solve
        with patch(
            "cogni_core.deq.torch.linalg.solve", wraps=native_solve
        ) as small_solve:
            result = limited_broyden_solve(
                root,
                z0,
                tolerance=2.0e-3,
                max_iter=40,
                rank=16,
                operator_id="bf16-contract-v1",
            )

        self.assertIsInstance(result, LimitedBroydenResult)
        self.assertTrue(result.converged)
        self.assertEqual(result.state.dtype, torch.bfloat16)
        self.assertTrue(torch.isfinite(result.state).all())
        self.assertGreater(result.linear_solve_attempts, 0)
        self.assertEqual(result.linear_solve_attempts, small_solve.call_count)
        self.assertEqual(result.linear_solve_fallbacks, 0)
        for call in small_solve.call_args_list:
            gram, rhs = call.args[:2]
            self.assertEqual(gram.dtype, torch.float32)
            self.assertEqual(rhs.dtype, torch.float32)

    def test_parent_warm_start_matches_fixed_point_and_reduces_iterations(self) -> None:
        z0 = torch.zeros(1, 4, dtype=torch.double)
        drive = torch.tensor([[0.2, -0.1, 0.3, -0.2]], dtype=torch.double)

        def root(z: torch.Tensor) -> torch.Tensor:
            return torch.tanh(0.35 * z + drive) - z

        cold = limited_broyden_solve(
            root,
            z0,
            tolerance=1.0e-10,
            max_iter=40,
            rank=16,
            operator_id="contractive-tanh-v1",
        )
        warm = limited_broyden_solve(
            root,
            z0,
            tolerance=1.0e-10,
            max_iter=40,
            rank=16,
            operator_id="contractive-tanh-v1",
            warm_start=cold.warm_start,
        )

        self.assertTrue(cold.converged and warm.converged)
        torch.testing.assert_close(warm.state, cold.state, atol=2.0e-10, rtol=0.0)
        self.assertLess(warm.iterations, cold.iterations)
        self.assertTrue(warm.warm_used)
        self.assertEqual(warm.warm_rejected, 0)
        self.assertIsNone(warm.warm_rejection_reason)

    def test_invalid_warm_capsules_are_rejected_then_retried_cold(self) -> None:
        z0 = torch.zeros(1, 3)

        def root(z: torch.Tensor) -> torch.Tensor:
            return torch.tanh(0.3 * z + 0.2) - z

        cold = limited_broyden_solve(
            root,
            z0,
            tolerance=1.0e-6,
            max_iter=30,
            rank=16,
            operator_id="operator-a",
        )
        cases: tuple[tuple[str, BroydenWarmStart, str], ...] = (
            ("operator_mismatch", cold.warm_start, "operator-b"),
            (
                "nonfinite",
                replace(
                    cold.warm_start,
                    state=torch.full_like(cold.warm_start.state, float("nan")),
                ),
                "operator-a",
            ),
            (
                "shape_mismatch",
                replace(cold.warm_start, state=torch.zeros(1, 4)),
                "operator-a",
            ),
            (
                "history_shape_mismatch",
                replace(
                    cold.warm_start,
                    x_history=torch.zeros(2, 1, 4),
                    f_history=torch.zeros(2, 1, 4),
                ),
                "operator-a",
            ),
        )

        for reason, capsule, operator_id in cases:
            with self.subTest(reason=reason):
                result = limited_broyden_solve(
                    root,
                    z0,
                    tolerance=1.0e-6,
                    max_iter=30,
                    rank=16,
                    operator_id=operator_id,
                    warm_start=capsule,
                )
                self.assertFalse(result.warm_used)
                self.assertEqual(result.warm_rejected, 1)
                self.assertEqual(result.warm_rejection_reason, reason)
                self.assertEqual(result.iterations, cold.iterations)
                torch.testing.assert_close(result.state, cold.state)

    def test_rank_sixteen_history_and_warm_capsule_are_iteration_bounded(self) -> None:
        z0 = torch.zeros(1, 8)

        def no_fixed_point(z: torch.Tensor) -> torch.Tensor:
            return torch.ones_like(z)

        short = limited_broyden_solve(
            no_fixed_point,
            z0,
            tolerance=1.0e-12,
            max_iter=40,
            rank=16,
            operator_id="bounded-history-v1",
        )
        long = limited_broyden_solve(
            no_fixed_point,
            z0,
            tolerance=1.0e-12,
            max_iter=200,
            rank=16,
            operator_id="bounded-history-v1",
        )

        self.assertEqual(short.rank, 16)
        self.assertEqual(long.rank, 16)
        self.assertEqual(short.history_peak, 16)
        self.assertEqual(long.history_peak, 16)
        self.assertEqual(short.warm_start.history_size, 17)
        self.assertEqual(long.warm_start.history_size, 17)
        self.assertEqual(
            short.warm_start.x_history.numel(), long.warm_start.x_history.numel()
        )
        self.assertEqual(
            short.warm_start.f_history.numel(), long.warm_start.f_history.numel()
        )

    def test_reported_residual_is_recomputed_at_returned_z_star(self) -> None:
        # At z=1 the pre-update residual is 0.5, while the returned fixed-point
        # candidate z*=1.5 has residual 0.25. This catches stale reporting.
        def root(z: torch.Tensor) -> torch.Tensor:
            return 1.0 - 0.5 * z

        z_star, iterations, residual, converged = _broyden_inverse(
            root,
            torch.zeros(1, 1, dtype=torch.double),
            tolerance=0.6,
            max_iter=4,
            history=2,
        )

        independently_measured = normalized_residual(root(z_star))
        self.assertEqual(iterations, 2)
        self.assertEqual(float(z_star.item()), 1.5)
        self.assertEqual(residual, independently_measured)
        self.assertEqual(residual, 0.25)
        self.assertTrue(converged)

    def test_solve_is_no_grad_finite_shape_stable_and_history_bounded(self) -> None:
        grad_modes: list[bool] = []
        drive = torch.tensor([[0.2, -0.1]], dtype=torch.double)

        def root(z: torch.Tensor) -> torch.Tensor:
            grad_modes.append(torch.is_grad_enabled())
            return torch.tanh(0.35 * z + drive) - z

        telemetry = BroydenTelemetry()
        z_star, _, residual, converged = _broyden_inverse(
            root,
            torch.zeros(1, 2, dtype=torch.double, requires_grad=True),
            tolerance=1.0e-12,
            max_iter=80,
            history=3,
            telemetry=telemetry,
        )

        self.assertEqual(z_star.shape, (1, 2))
        self.assertTrue(torch.isfinite(z_star).all())
        self.assertFalse(z_star.requires_grad)
        self.assertTrue(converged)
        self.assertTrue(grad_modes)
        self.assertTrue(all(not enabled for enabled in grad_modes))
        self.assertEqual(residual, normalized_residual(root(z_star)))
        self.assertLessEqual(telemetry.history_peak, 4)

    def test_small_linear_solve_fallback_is_counted_without_growing_history(
        self,
    ) -> None:
        telemetry = BroydenTelemetry()

        def root(z: torch.Tensor) -> torch.Tensor:
            return 1.0 - 0.5 * z

        with patch(
            "cogni_core.deq.torch.linalg.solve",
            side_effect=RuntimeError("singular small solve"),
        ):
            z_star, _, residual, converged = _broyden_inverse(
                root,
                torch.zeros(2, 3),
                tolerance=1.0e-12,
                max_iter=12,
                history=2,
                telemetry=telemetry,
            )

        self.assertTrue(torch.isfinite(z_star).all())
        self.assertEqual(residual, normalized_residual(root(z_star)))
        self.assertFalse(converged)
        self.assertGreater(telemetry.linear_solve_attempts, 0)
        self.assertEqual(
            telemetry.linear_solve_fallbacks,
            telemetry.linear_solve_attempts,
        )
        self.assertTrue(telemetry.used_linear_solve_fallback)
        self.assertLessEqual(telemetry.history_peak, 3)

    def test_finite_oversized_correction_is_projected_without_fallback(self) -> None:
        """Finite Anderson steps use the primary trust region, not fallback."""

        telemetry = BroydenTelemetry()

        def root(z: torch.Tensor) -> torch.Tensor:
            return torch.tanh(0.35 * z + 0.2) - z

        native_solve = torch.linalg.solve

        def oversized_solve(gram: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
            solved = native_solve(gram, rhs)
            return solved * 1.0e6

        with patch("cogni_core.deq.torch.linalg.solve", side_effect=oversized_solve):
            result = limited_broyden_solve(
                root,
                torch.zeros(1, 8, dtype=torch.bfloat16),
                tolerance=2.0e-3,
                max_iter=40,
                rank=16,
                operator_id="trust-region-projection-v1",
                telemetry=telemetry,
            )

        self.assertTrue(torch.isfinite(result.state).all())
        self.assertTrue(result.converged)
        self.assertLessEqual(result.residual, 2.0e-3)
        self.assertGreater(result.linear_solve_attempts, 0)
        self.assertEqual(result.linear_solve_fallbacks, 0)

    def test_wide_bfloat16_near_collinear_history_stays_on_primary_path(
        self,
    ) -> None:
        z0 = torch.zeros(1, 4096, dtype=torch.bfloat16)
        drive = torch.linspace(-0.2, 0.2, 4096, dtype=torch.float32).to(torch.bfloat16)[
            None, :
        ]
        observed_grams: list[torch.Tensor] = []

        def root(z: torch.Tensor) -> torch.Tensor:
            return torch.tanh(0.4 * z + drive) - z

        native_solve = torch.linalg.solve

        def recording_solve(gram: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
            observed_grams.append(gram.detach().cpu())
            return native_solve(gram, rhs)

        with patch("cogni_core.deq.torch.linalg.solve", side_effect=recording_solve):
            result = limited_broyden_solve(
                root,
                z0,
                tolerance=2.0e-3,
                max_iter=40,
                rank=16,
                operator_id="wide-bf16-near-collinear-v1",
            )

        self.assertTrue(result.converged)
        self.assertLessEqual(result.residual, 2.0e-3)
        self.assertEqual(result.linear_solve_fallbacks, 0)
        self.assertLessEqual(result.history_peak, 16)
        self.assertTrue(observed_grams)
        for gram in observed_grams:
            self.assertEqual(gram.dtype, torch.float32)
            self.assertTrue(torch.isfinite(gram).all())
            self.assertTrue(bool((torch.linalg.eigvalsh(gram) > 0).all()))

    def test_nonfinite_and_wrong_shape_roots_fail_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "changed shape"):
            _broyden_inverse(
                lambda z: z[:, :1],
                torch.zeros(1, 2),
                tolerance=1.0e-5,
                max_iter=2,
                history=1,
            )
        with self.assertRaisesRegex(ValueError, "finite floating point"):
            _broyden_inverse(
                lambda z: z,
                torch.full((1, 2), float("nan")),
                tolerance=1.0e-5,
                max_iter=2,
                history=1,
            )

        z_star, _, residual, converged = _broyden_inverse(
            lambda z: torch.full_like(z, float("nan")),
            torch.zeros(1, 2),
            tolerance=1.0e-5,
            max_iter=2,
            history=1,
        )
        self.assertTrue(torch.isfinite(z_star).all())
        self.assertTrue(math.isinf(residual))
        self.assertFalse(converged)

    def test_equilibrium_layer_exposes_small_solve_fallback_telemetry(self) -> None:
        layer = EquilibriumLayer(
            2,
            2,
            DEQConfig(
                tolerance=1.0e-12,
                max_iter=3,
                history=2,
                fallback_steps=32,
                fail_on_noncontractive=False,
            ),
        )
        with patch(
            "cogni_core.deq.torch.linalg.solve",
            side_effect=RuntimeError("forced small solve fallback"),
        ):
            output = layer(torch.ones(1, 2))

        self.assertTrue(torch.isfinite(output).all())
        self.assertIsNotNone(layer.last_info)
        self.assertTrue(layer.last_info.used_linear_solve_fallback)
        self.assertGreater(layer.last_info.linear_solve_fallbacks, 0)
        expected = normalized_residual(
            (
                torch.tanh(
                    output @ layer.recurrent.T
                    + torch.ones(1, 2) @ layer.input_weight.T
                    + layer.bias
                )
                - output
            ).detach()
        )
        self.assertEqual(layer.last_info.residual, expected)


if __name__ == "__main__":
    unittest.main()
