import math
import unittest
import warnings

import torch
from torch import nn

from cogni_core.fp_ewc import (
    AdjointConvergenceError,
    FPEWCRegularizer,
    FixedPointFisherConfig,
    FixedPointFisherError,
    estimate_empirical_fixed_point_fisher,
    spectral_guard_,
    verified_spectral_cap_,
)


def _config(contraction: float, **overrides) -> FixedPointFisherConfig:
    values = {
        "contraction_bound": contraction,
        "fixed_point_tolerance": 1.0e-7,
        "adjoint_tolerance": 1.0e-7,
        "max_adjoint_iterations": 512,
    }
    values.update(overrides)
    return FixedPointFisherConfig(**values)


class TestEmpiricalFixedPointFisher(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(71)

    def test_implicit_gradient_matches_small_explicit_inverse(self):
        recurrent = torch.tensor([[0.20, 0.10], [-0.05, 0.30]])
        drive = nn.Parameter(torch.tensor([0.4, -0.2]))
        direct = nn.Parameter(torch.tensor([0.15, -0.25]))
        score = torch.tensor([0.7, -0.4])
        fixed = torch.linalg.solve(torch.eye(2) - recurrent, drive.detach())
        z_star = fixed.unsqueeze(0)

        result = estimate_empirical_fixed_point_fisher(
            f_at_z=lambda z: z @ recurrent.T + drive,
            z_star=z_star,
            log_likelihood_per_sample=lambda z: (
                (z * score).sum(dim=1) + (direct * torch.tensor([0.3, -0.2])).sum()
            ).reshape(1),
            named_parameters=[("drive", drive), ("direct", direct)],
            config=_config(0.4),
            solver_converged=True,
        )

        expected_v = torch.linalg.solve(torch.eye(2) - recurrent.T, score)
        expected_direct = torch.tensor([0.3, -0.2])
        self.assertTrue(
            torch.allclose(result.fisher["drive"], expected_v.square(), atol=1.0e-5)
        )
        self.assertTrue(
            torch.allclose(
                result.fisher["direct"], expected_direct.square(), atol=1.0e-6
            )
        )
        self.assertEqual(result.n_samples, 1)
        self.assertTrue(result.adjoint[0].converged)
        self.assertLessEqual(result.adjoint[0].residual, 1.0e-7)

    def test_direct_term_matches_finite_difference_total_derivative(self):
        contraction = 0.3
        theta = nn.Parameter(torch.tensor([0.2]))
        z_star = (theta.detach() / (1.0 - contraction)).reshape(1, 1)
        direct_coefficient = 2.0

        result = estimate_empirical_fixed_point_fisher(
            f_at_z=lambda z: contraction * z + theta,
            z_star=z_star,
            log_likelihood_per_sample=lambda z: z[:, 0] + direct_coefficient * theta[0],
            named_parameters=[("theta", theta)],
            config=_config(contraction),
            solver_converged=True,
        )

        def equilibrium_loss(value: float) -> float:
            return value / (1.0 - contraction) + direct_coefficient * value

        epsilon = 1.0e-4
        theta_value = float(theta.detach())
        finite_difference = (
            equilibrium_loss(theta_value + epsilon)
            - equilibrium_loss(theta_value - epsilon)
        ) / (2.0 * epsilon)
        self.assertAlmostEqual(
            float(result.fisher["theta"]), finite_difference**2, places=3
        )

    def test_per_sample_plus_one_minus_one_scores_do_not_cancel(self):
        theta = nn.Parameter(torch.tensor([0.25]))
        z_star = torch.zeros((2, 1))
        result = estimate_empirical_fixed_point_fisher(
            f_at_z=lambda z: 0.2 * z,
            z_star=z_star,
            log_likelihood_per_sample=lambda _z: torch.stack((theta[0], -theta[0])),
            named_parameters=[("theta", theta)],
            config=_config(0.2),
            solver_converged=True,
        )

        self.assertEqual(result.n_samples, 2)
        self.assertAlmostEqual(float(result.fisher["theta"]), 1.0, places=6)
        self.assertEqual(len(result.adjoint), 2)

    def test_near_point_nine_five_contraction_converges_with_explicit_residual(self):
        contraction = math.nextafter(0.95, 0.0)
        drive = nn.Parameter(torch.tensor([0.01], dtype=torch.float64))
        z_star = (drive.detach() / (1.0 - contraction)).reshape(1, 1)
        result = estimate_empirical_fixed_point_fisher(
            f_at_z=lambda z: contraction * z + drive,
            z_star=z_star,
            log_likelihood_per_sample=lambda z: z[:, 0],
            named_parameters=[("drive", drive)],
            config=_config(
                contraction,
                adjoint_tolerance=1.0e-6,
                max_adjoint_iterations=700,
            ),
            solver_converged=True,
        )

        telemetry = result.adjoint[0]
        self.assertTrue(telemetry.converged)
        self.assertGreater(telemetry.iterations, 100)
        self.assertLessEqual(telemetry.residual, 1.0e-6)
        expected = (1.0 / (1.0 - contraction)) ** 2
        self.assertAlmostEqual(float(result.fisher["drive"]), expected, places=2)

    def test_noncontractive_nonconverged_and_bad_fixed_point_fail_closed(self):
        with self.assertRaisesRegex(ValueError, "contraction_bound"):
            _config(0.95)

        parameter = nn.Parameter(torch.tensor([0.1]))
        with self.assertRaisesRegex(FixedPointFisherError, "did not converge"):
            estimate_empirical_fixed_point_fisher(
                f_at_z=lambda z: 0.5 * z + parameter,
                z_star=(parameter.detach() / 0.5).reshape(1, 1),
                log_likelihood_per_sample=lambda z: z[:, 0],
                named_parameters=[("parameter", parameter)],
                config=_config(0.5),
                solver_converged=False,
            )
        with self.assertRaisesRegex(FixedPointFisherError, "fixed-point residual"):
            estimate_empirical_fixed_point_fisher(
                f_at_z=lambda z: 0.5 * z + parameter,
                z_star=torch.zeros((1, 1)),
                log_likelihood_per_sample=lambda z: z[:, 0],
                named_parameters=[("parameter", parameter)],
                config=_config(0.5),
                solver_converged=True,
            )

        contraction = 0.9
        z_star = (parameter.detach() / (1.0 - contraction)).reshape(1, 1)
        with self.assertRaises(AdjointConvergenceError) as raised:
            estimate_empirical_fixed_point_fisher(
                f_at_z=lambda z: contraction * z + parameter,
                z_star=z_star,
                log_likelihood_per_sample=lambda z: z[:, 0],
                named_parameters=[("parameter", parameter)],
                config=_config(
                    contraction,
                    adjoint_tolerance=1.0e-12,
                    max_adjoint_iterations=1,
                ),
                solver_converged=True,
            )
        self.assertFalse(raised.exception.telemetry.converged)
        self.assertEqual(raised.exception.telemetry.iterations, 1)

    def test_nonfinite_and_fisher_byte_budget_fail_closed(self):
        parameter = nn.Parameter(torch.tensor([0.1, 0.2]))
        with self.assertRaisesRegex(ValueError, "z_star"):
            estimate_empirical_fixed_point_fisher(
                f_at_z=lambda z: z,
                z_star=torch.tensor([[math.nan, 0.0]]),
                log_likelihood_per_sample=lambda z: z.sum(dim=1),
                named_parameters=[("parameter", parameter)],
                config=_config(0.5),
                solver_converged=True,
            )

        z_star = torch.zeros((1, 2))
        with self.assertRaisesRegex(FixedPointFisherError, "log likelihood"):
            estimate_empirical_fixed_point_fisher(
                f_at_z=lambda z: 0.2 * z,
                z_star=z_star,
                log_likelihood_per_sample=lambda _z: (
                    parameter.sum().reshape(1) * math.nan
                ),
                named_parameters=[("parameter", parameter)],
                config=_config(0.2),
                solver_converged=True,
            )

        with self.assertRaisesRegex(FixedPointFisherError, "byte budget"):
            estimate_empirical_fixed_point_fisher(
                f_at_z=lambda z: 0.2 * z,
                z_star=z_star,
                log_likelihood_per_sample=lambda _z: parameter.sum().reshape(1),
                named_parameters=[("parameter", parameter)],
                config=_config(0.2, max_fisher_bytes=4),
                solver_converged=True,
            )


class TestBoundedFPEWCRegularizer(unittest.TestCase):
    def test_snapshot_is_sparse_cpu_fp32_and_preserves_sample_count(self):
        used = nn.Parameter(torch.tensor([1.0, -2.0], dtype=torch.float64))
        unused = nn.Parameter(torch.tensor([3.0], dtype=torch.float64))
        regularizer = FPEWCRegularizer()
        snapshot = regularizer.consolidate_fisher(
            [("used", used), ("unused", unused)],
            {"used": torch.tensor([0.25, 0.5], dtype=torch.float64)},
            n_samples=7,
        )

        self.assertEqual(set(snapshot.fisher), {"used"})
        self.assertEqual(set(snapshot.anchor), {"used"})
        self.assertEqual(snapshot.n_samples, 7)
        self.assertEqual(snapshot.fisher["used"].device.type, "cpu")
        self.assertEqual(snapshot.anchor["used"].device.type, "cpu")
        self.assertEqual(snapshot.fisher["used"].dtype, torch.float32)
        self.assertEqual(snapshot.anchor["used"].dtype, torch.float32)

    def test_sample_weighted_merge_preserves_penalty_and_gradient(self):
        parameter = nn.Parameter(torch.tensor([0.0, 0.0]))
        regularizer = FPEWCRegularizer(strength=1.7, max_domains=4)
        with torch.no_grad():
            parameter.copy_(torch.tensor([1.0, -0.5]))
        regularizer.consolidate_fisher(
            [("weight", parameter)],
            {"weight": torch.tensor([0.4, 0.2])},
            n_samples=2,
        )
        with torch.no_grad():
            parameter.copy_(torch.tensor([-0.2, 0.8]))
        regularizer.consolidate_fisher(
            [("weight", parameter)],
            {"weight": torch.tensor([0.1, 0.6])},
            n_samples=3,
        )
        with torch.no_grad():
            parameter.copy_(torch.tensor([0.3, -0.1]))

        before = regularizer.penalty([("weight", parameter)])
        (before_grad,) = torch.autograd.grad(before, parameter)
        merged = regularizer.merge_oldest()
        after = regularizer.penalty([("weight", parameter)])
        (after_grad,) = torch.autograd.grad(after, parameter)

        self.assertEqual(merged.n_samples, 5)
        self.assertTrue(torch.allclose(before, after, atol=1.0e-6, rtol=1.0e-6))
        self.assertTrue(
            torch.allclose(before_grad, after_grad, atol=1.0e-6, rtol=1.0e-6)
        )

    def test_domain_and_byte_caps_are_hard(self):
        parameter = nn.Parameter(torch.tensor([0.0, 0.0]))
        regularizer = FPEWCRegularizer(max_domains=1, max_total_bytes=1_024)
        for index in range(3):
            with torch.no_grad():
                parameter.fill_(float(index))
            regularizer.consolidate_fisher(
                [("weight", parameter)],
                {"weight": torch.ones_like(parameter)},
                n_samples=index + 1,
            )
        self.assertEqual(len(regularizer.snapshots), 1)
        self.assertEqual(regularizer.snapshots[0].n_samples, 6)
        self.assertLessEqual(regularizer.total_bytes, regularizer.max_total_bytes)

        too_small = FPEWCRegularizer(max_total_bytes=8)
        with self.assertRaisesRegex(FixedPointFisherError, "byte budget"):
            too_small.consolidate_fisher(
                [("weight", parameter)],
                {"weight": torch.ones_like(parameter)},
                n_samples=1,
            )
        self.assertEqual(too_small.snapshots, [])


class TestVerifiedSpectralCap(unittest.TestCase):
    def test_verified_cap_handles_batched_matrices_and_legacy_alias_is_deprecated(self):
        weights = torch.stack((torch.eye(3) * 2.0, torch.eye(3) * 0.4))
        measured = verified_spectral_cap_(weights, 0.9)
        norms = torch.linalg.matrix_norm(weights, ord=2)
        self.assertLess(measured, 0.9)
        self.assertTrue((norms < 0.9).all())

        legacy = torch.eye(2) * 2.0
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            spectral_guard_(legacy, 0.8)
        self.assertTrue(any(item.category is DeprecationWarning for item in caught))
        self.assertLess(float(torch.linalg.matrix_norm(legacy, ord=2)), 0.8)


if __name__ == "__main__":
    unittest.main()
