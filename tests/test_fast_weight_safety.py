import io
import math
import unittest

import torch

from cogni_core.fast_weight_safety import (
    ExternalVerifierEvidence,
    FactorWiseEMA,
    ResidualDecayAQGate,
    TrainedProgrammerEvidence,
)


PROGRAMMER_HASH = "1" * 64
TRAINING_HASH = "2" * 64
VERIFIER_HASH = "3" * 64
DATASET_HASH = "4" * 64


def _programmer() -> TrainedProgrammerEvidence:
    return TrainedProgrammerEvidence(
        checkpoint_sha256=PROGRAMMER_HASH,
        training_run_sha256=TRAINING_HASH,
        training_steps=200,
        training_samples=1_024,
    )


def _verifier(
    *,
    programmer_hash: str = PROGRAMMER_HASH,
    quality: float = 0.93,
    passed: bool = True,
) -> ExternalVerifierEvidence:
    return ExternalVerifierEvidence(
        verifier_checkpoint_sha256=VERIFIER_HASH,
        evaluated_programmer_sha256=programmer_hash,
        heldout_dataset_sha256=DATASET_HASH,
        verified_quality=quality,
        sample_count=128,
        passed=passed,
    )


def _stable_trace() -> torch.Tensor:
    return torch.logspace(-1, -3, steps=10, dtype=torch.float64)


class TestResidualDecayAQGate(unittest.TestCase):
    def setUp(self) -> None:
        self.gate = ResidualDecayAQGate()
        self.programmer = _programmer()
        self.verifier = _verifier()

    def evaluate(self, residuals, **overrides):
        arguments = {
            "solver_converged": True,
            "solver_used_fallback": False,
            "programmer": self.programmer,
            "verifier": self.verifier,
        }
        arguments.update(overrides)
        return self.gate.evaluate(residuals, **arguments)

    def test_stable_exact_ten_trace_returns_deterministic_certificate(self):
        first = self.evaluate(_stable_trace())
        second = self.evaluate(_stable_trace().tolist())

        self.assertTrue(first.accepted)
        self.assertEqual(first.reason, "accepted")
        self.assertEqual(len(first.residuals), 10)
        self.assertEqual(len(first.log_decays), 9)
        self.assertEqual(first.observed_count, 10)
        self.assertTrue(first.trace_was_one_dimensional)
        self.assertLess(first.terminal_residual, 5.0e-3)
        self.assertAlmostEqual(first.monotonic_fraction, 1.0)
        self.assertLess(first.log_decay_variance, 5.0e-2)
        self.assertEqual(first.digest, second.digest)
        self.assertEqual(len(first.digest), 64)

    def test_untrained_short_nonfinite_flat_and_increasing_fail_closed(self):
        cases = (
            (
                _stable_trace(),
                {"programmer": None},
                "untrained programmer",
            ),
            (_stable_trace()[:9], {}, "exactly 10"),
            (
                torch.tensor(
                    [0.1, 0.08, 0.06, 0.04, math.nan, 0.02, 0.01, 0.008, 0.004, 0.001]
                ),
                {},
                "non-finite",
            ),
            (torch.full((10,), 0.01), {}, "flat"),
            (torch.logspace(-3, -1, steps=10), {}, "increasing"),
        )
        for residuals, overrides, reason in cases:
            with self.subTest(reason=reason):
                certificate = self.evaluate(residuals, **overrides)
                self.assertFalse(certificate.accepted)
                self.assertIn(reason, certificate.reason)
                self.assertLessEqual(len(certificate.residuals), 10)
                self.assertEqual(len(certificate.digest), 64)

    def test_trace_shape_solver_and_fallback_contracts_are_fail_closed(self):
        cases = (
            (_stable_trace().repeat(2, 1), {}, "exactly 10"),
            (_stable_trace().repeat(2), {}, "exactly 10"),
            (_stable_trace(), {"solver_converged": False}, "did not converge"),
            (_stable_trace(), {"solver_used_fallback": True}, "fallback"),
        )
        for residuals, overrides, reason in cases:
            with self.subTest(reason=reason):
                certificate = self.evaluate(residuals, **overrides)
                self.assertFalse(certificate.accepted)
                self.assertIn(reason, certificate.reason)

    def test_external_verifier_is_typed_separate_and_bound_to_checkpoint(self):
        with self.assertRaisesRegex(ValueError, "cannot verify its own quality"):
            ExternalVerifierEvidence(
                verifier_checkpoint_sha256=PROGRAMMER_HASH,
                evaluated_programmer_sha256=PROGRAMMER_HASH,
                heldout_dataset_sha256=DATASET_HASH,
                verified_quality=0.99,
                sample_count=10,
                passed=True,
            )
        self.assertFalse(hasattr(self.verifier, "programmer_quality"))
        self.assertIn("verified_quality", self.verifier.__slots__)

        missing = self.evaluate(_stable_trace(), verifier=None)
        self.assertFalse(missing.accepted)
        self.assertIn("missing", missing.reason)
        mismatched = self.evaluate(
            _stable_trace(), verifier=_verifier(programmer_hash="5" * 64)
        )
        self.assertFalse(mismatched.accepted)
        self.assertIn("different programmer", mismatched.reason)
        low_quality = self.evaluate(_stable_trace(), verifier=_verifier(quality=0.79))
        self.assertFalse(low_quality.accepted)
        self.assertIn("quality gate", low_quality.reason)

    def test_terminal_variance_and_monotonic_thresholds_are_certified(self):
        terminal_high = torch.logspace(-0.5, -2.0, steps=10, dtype=torch.float64)
        terminal = self.evaluate(terminal_high)
        self.assertFalse(terminal.accepted)
        self.assertIn("terminal residual", terminal.reason)

        irregular = torch.tensor(
            [0.10, 0.09, 0.04, 0.039, 0.02, 0.019, 0.008, 0.0079, 0.002, 0.001],
            dtype=torch.float64,
        )
        variance = self.evaluate(irregular)
        self.assertFalse(variance.accepted)
        self.assertIn("variance", variance.reason)


class TestFactorWiseEMA(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(41)
        self.base = torch.zeros((3, 4), dtype=torch.float32)
        self.base[:, :3] = torch.eye(3) * 0.5

    def make_ema(self, **kwargs) -> FactorWiseEMA:
        options = {
            "decay": 0.5,
            "max_operator_norm": 0.1,
            "composed_norm_limit": 0.95,
            "strict_margin": 1.0e-5,
        }
        options.update(kwargs)
        return FactorWiseEMA(4, 3, 2, **options)

    def test_factors_have_independent_ema_and_count_without_base_mutation(self):
        ema = self.make_ema()
        a1 = torch.full((4, 2), 0.01)
        b1 = torch.full((3, 2), 0.02)
        a2 = torch.full((4, 2), 0.03)
        b2 = torch.full((3, 2), 0.04)
        base_before = self.base.clone()

        first = ema.update(a1, b1, self.base)
        self.assertTrue(first.accepted)
        self.assertEqual(first.update_count, 1)
        stored_a, stored_b = ema.factors()
        self.assertTrue(torch.equal(stored_a, a1))
        self.assertTrue(torch.equal(stored_b, b1))

        second = ema.update(a2, b2, self.base)
        self.assertTrue(second.accepted)
        self.assertEqual(ema.update_count, 2)
        stored_a, stored_b = ema.factors()
        self.assertTrue(torch.allclose(stored_a, (a1 + a2) / 2.0))
        self.assertTrue(torch.allclose(stored_b, (b1 + b2) / 2.0))
        self.assertTrue(torch.equal(self.base, base_before))
        self.assertTrue(second.base_unchanged)

    def test_every_update_projects_operator_and_strict_composed_norm(self):
        ema = self.make_ema()
        base_before = self.base.clone()
        certificate = ema.update(
            torch.full((4, 2), 8.0),
            torch.full((3, 2), 9.0),
            self.base,
        )

        self.assertTrue(certificate.accepted)
        self.assertTrue(certificate.projected)
        self.assertLessEqual(certificate.overlay_operator_norm, 0.1)
        self.assertLess(certificate.composed_upper_bound, 0.95)
        self.assertTrue(torch.equal(self.base, base_before))
        a, b = ema.factors()
        dense_norm = torch.linalg.matrix_norm(b @ a.T, ord=2)
        self.assertLessEqual(float(dense_norm), 0.1 + 1.0e-6)

    def test_unsafe_base_rejects_without_committing_state(self):
        ema = self.make_ema()
        unsafe = torch.zeros_like(self.base)
        unsafe[:, :3] = torch.eye(3) * 0.95
        certificate = ema.update(
            torch.full((4, 2), 0.01),
            torch.full((3, 2), 0.01),
            unsafe,
        )
        self.assertFalse(certificate.accepted)
        self.assertIn("no strict", certificate.reason)
        self.assertEqual(ema.update_count, 0)
        self.assertFalse(ema.initialized)
        with self.assertRaises(RuntimeError):
            ema.factors()

    def test_shape_dtype_and_device_identity_are_fixed(self):
        ema = self.make_ema()
        with self.assertRaisesRegex(ValueError, "factor shapes"):
            ema.update(torch.ones(5, 2), torch.ones(3, 2), self.base)
        with self.assertRaisesRegex(TypeError, "factor dtype"):
            ema.update(
                torch.ones(4, 2, dtype=torch.float64),
                torch.ones(3, 2, dtype=torch.float64),
                self.base,
            )
        with self.assertRaisesRegex(TypeError, "base_weight dtype"):
            ema.update(
                torch.ones(4, 2),
                torch.ones(3, 2),
                self.base.double(),
            )
        with self.assertRaisesRegex(ValueError, "finite"):
            bad = torch.ones(4, 2)
            bad[0, 0] = math.nan
            ema.update(bad, torch.ones(3, 2), self.base)

    def test_state_dict_and_hash_protected_checkpoint_reproduce_next_update(self):
        ema = self.make_ema()
        ema.update(torch.full((4, 2), 0.01), torch.full((3, 2), 0.02), self.base)

        state_clone = self.make_ema()
        state_clone.load_state_dict(ema.state_dict(), strict=True)
        self.assertEqual(state_clone.update_count, ema.update_count)
        for actual, expected in zip(state_clone.factors(), ema.factors(), strict=True):
            self.assertTrue(torch.equal(actual, expected))

        buffer = io.BytesIO()
        torch.save(ema.checkpoint(), buffer)
        buffer.seek(0)
        payload = torch.load(buffer, weights_only=True)
        restored = FactorWiseEMA.from_checkpoint(payload)
        self.assertEqual(restored.update_count, ema.update_count)
        for actual, expected in zip(restored.factors(), ema.factors(), strict=True):
            self.assertTrue(torch.equal(actual, expected))

        next_a = torch.full((4, 2), 0.015)
        next_b = torch.full((3, 2), 0.025)
        expected_certificate = ema.update(next_a, next_b, self.base)
        state_certificate = state_clone.update(next_a, next_b, self.base)
        restored_certificate = restored.update(next_a, next_b, self.base)
        self.assertEqual(expected_certificate.digest, state_certificate.digest)
        self.assertEqual(expected_certificate.digest, restored_certificate.digest)
        for actual, expected in zip(restored.factors(), ema.factors(), strict=True):
            self.assertTrue(torch.equal(actual, expected))

        tampered = restored.checkpoint()
        tampered["sha256"] = "0" * 64
        with self.assertRaisesRegex(RuntimeError, "digest verification"):
            FactorWiseEMA.from_checkpoint(tampered)


if __name__ == "__main__":
    unittest.main()
