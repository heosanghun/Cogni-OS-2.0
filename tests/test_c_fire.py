from __future__ import annotations

import unittest

import torch

from cogni_core.c_fire import CFireError, c_fire_scaled_polar_


class TestCFireScaledPolar(unittest.TestCase):
    def test_restores_isometry_instead_of_only_capping_sigma_max(self) -> None:
        weight = torch.diag(torch.tensor([100.0, 1.0e-6], dtype=torch.float64))

        certificate = c_fire_scaled_polar_(weight, max_iter=128, tolerance=1.0e-8)
        singular = torch.linalg.svdvals(weight)

        torch.testing.assert_close(
            singular, torch.full_like(singular, 0.9), atol=1e-7, rtol=1e-7
        )
        self.assertGreater(certificate.before_condition_number, 1.0e7)
        self.assertLess(certificate.after_condition_number, 1.00001)
        self.assertEqual(certificate.after_effective_rank, 2)
        self.assertLess(certificate.after_sigma_max, 0.95)

    def test_bfloat16_target_uses_high_precision_work_and_certifies_cast(self) -> None:
        generator = torch.Generator().manual_seed(712)
        weight = torch.randn(8, 4, generator=generator, dtype=torch.float32).to(
            torch.bfloat16
        )

        certificate = c_fire_scaled_polar_(weight)
        singular = torch.linalg.svdvals(weight.float())

        self.assertTrue(bool(torch.isfinite(weight).all()))
        self.assertLess(float(singular.max()), 0.95)
        self.assertLess(certificate.after_condition_number, 1.05)
        self.assertEqual(certificate.after_effective_rank, 4)

    def test_rank_deficiency_and_nonconvergence_do_not_mutate_weight(self) -> None:
        cases = (
            (torch.diag(torch.tensor([1.0, 0.0])), {}),
            (
                torch.diag(torch.tensor([1.0, 0.2])),
                {"max_iter": 1, "tolerance": 1.0e-8},
            ),
        )
        for weight, options in cases:
            with self.subTest(options=options):
                original = weight.clone()
                with self.assertRaises(CFireError):
                    c_fire_scaled_polar_(weight, **options)
                self.assertTrue(torch.equal(weight, original))

    def test_rectangular_wide_matrix_has_full_row_rank_certificate(self) -> None:
        weight = torch.tensor([[2.0, 0.5, 0.1], [0.2, 1.5, -0.3]], dtype=torch.float32)
        certificate = c_fire_scaled_polar_(weight)

        self.assertEqual(certificate.after_effective_rank, 2)
        self.assertLess(certificate.after_condition_number, 1.01)


if __name__ == "__main__":
    unittest.main()
