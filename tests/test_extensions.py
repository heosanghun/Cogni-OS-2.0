import unittest

import torch

from cogni_core.cts import CognitiveTreeSearch
from cogni_core.deq import DEQConfig, EquilibriumLayer
from cogni_core.fast_weights import FastWeightProgrammer
from cogni_core.fp_ewc import (
    FPEWCRegularizer,
    estimate_fixed_point_fisher,
    spectral_guard_,
)


class TestExtensions(unittest.TestCase):
    def test_cts_runs_to_requested_depth_with_fixed_width(self):
        torch.manual_seed(1)
        transition = EquilibriumLayer(4, 4, DEQConfig(max_iter=40))
        result = CognitiveTreeSearch(transition, width=3).search(
            torch.zeros(1, 4),
            lambda state, action: state + 0.01 * (action + 1),
            lambda state: -state.square().mean(),
            depth=12,
        )
        self.assertEqual(result.depth, 12)
        self.assertEqual(result.expanded, 36)
        self.assertTrue(torch.isfinite(result.state).all())

    def test_fast_weight_rank_and_operator_bound(self):
        fwp = FastWeightProgrammer(8, rank=2, max_operator_norm=0.08)
        overlay = fwp(torch.randn(3, 5, 8))
        self.assertEqual(tuple(overlay.delta.shape), (3, 8, 8))
        self.assertTrue(
            (torch.linalg.matrix_norm(overlay.delta, ord=2) <= 0.0801).all()
        )
        self.assertTrue((overlay.quality >= 0).all() and (overlay.quality <= 1).all())

    def test_fast_weight_programmer_separates_source_and_bounded_target(self):
        fwp = FastWeightProgrammer(
            source_dim=32,
            target_dim=6,
            internal_dim=7,
            rank=2,
            max_operator_norm=0.05,
        )
        overlay = fwp(torch.randn(1, 4, 32))
        self.assertEqual(tuple(overlay.a.shape), (1, 6, 2))
        self.assertEqual(tuple(overlay.b.shape), (1, 6, 2))
        self.assertEqual(fwp.trunk[0].out_features, 7)
        self.assertTrue(
            (torch.linalg.matrix_norm(overlay.delta, ord=2) <= 0.0501).all()
        )
        legacy = FastWeightProgrammer(640, rank=1, internal_dim=4)
        self.assertEqual(tuple(legacy(torch.randn(1, 640)).a.shape), (1, 640, 1))

    def test_fp_ewc_anchor_and_spectral_guard(self):
        layer = torch.nn.Linear(4, 4, bias=False)
        reg = FPEWCRegularizer(strength=2.0)
        grads = {"weight": torch.ones_like(layer.weight)}
        reg.consolidate(layer.named_parameters(), grads)
        self.assertEqual(float(reg.penalty(layer.named_parameters()).detach()), 0.0)
        with torch.no_grad():
            layer.weight.add_(0.1)
        self.assertGreater(float(reg.penalty(layer.named_parameters()).detach()), 0.0)
        with torch.no_grad():
            layer.weight.mul_(20)
        self.assertLessEqual(spectral_guard_(layer.weight, 0.9), 0.9001)

    def test_spectral_guard_preserves_bfloat16_parameters(self):
        weight = (torch.eye(4) * 2.0).to(torch.bfloat16)
        norm = spectral_guard_(weight, 0.9)
        self.assertEqual(weight.dtype, torch.bfloat16)
        self.assertLess(norm, 0.9)

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
    def test_fp_ewc_restored_cpu_snapshot_follows_parameter_device(self):
        layer = torch.nn.Linear(3, 3, bias=False).cuda()
        reg = FPEWCRegularizer(strength=1.0)
        reg.consolidate(
            layer.named_parameters(), {"weight": torch.ones_like(layer.weight)}
        )
        snapshot = reg.snapshots[0]
        snapshot.fisher["weight"] = snapshot.fisher["weight"].cpu()
        snapshot.anchor["weight"] = snapshot.anchor["weight"].cpu()
        with torch.no_grad():
            layer.weight.add_(0.01)
        penalty = reg.penalty(layer.named_parameters())
        self.assertTrue(penalty.is_cuda)
        self.assertTrue(torch.isfinite(penalty))

    def test_matrix_free_fixed_point_fisher(self):
        torch.manual_seed(4)
        recurrent = torch.nn.Parameter(torch.eye(3) * 0.4)
        drive = torch.randn(1, 3)
        z = torch.zeros_like(drive)
        for _ in range(100):
            z = torch.tanh(z @ recurrent.T + drive)
        fisher = estimate_fixed_point_fisher(
            f_at_z=lambda state: torch.tanh(state @ recurrent.T + drive),
            z_star=z,
            log_likelihood_at_z=lambda state: -0.5 * state.square().sum(),
            named_parameters=[("recurrent", recurrent)],
        )
        self.assertEqual(fisher["recurrent"].shape, recurrent.shape)
        self.assertTrue(torch.isfinite(fisher["recurrent"]).all())
        self.assertGreater(float(fisher["recurrent"].sum()), 0.0)


if __name__ == "__main__":
    unittest.main()
