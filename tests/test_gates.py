import unittest

import torch

from cogni_core.cts import CognitiveTreeSearch
from cogni_core.deq import (
    ContractivityError,
    DEQConfig,
    EquilibriumLayer,
    normalized_residual,
)


class TestDEQGates(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)

    def test_forward_and_ift_backward_match_long_unroll(self):
        cfg = DEQConfig(tolerance=1e-7, max_iter=100, history=8)
        implicit = EquilibriumLayer(4, 4, cfg).double()
        explicit = EquilibriumLayer(4, 4, cfg).double()
        explicit.load_state_dict(implicit.state_dict())
        x1 = torch.randn(2, 4, dtype=torch.double, requires_grad=True)
        x2 = x1.detach().clone().requires_grad_(True)

        y_implicit = implicit(x1)
        loss_implicit = y_implicit.square().sum()
        loss_implicit.backward()

        z = torch.zeros_like(x2)
        for _ in range(250):
            z = torch.tanh(
                z @ explicit.recurrent.T + x2 @ explicit.input_weight.T + explicit.bias
            )
        z.square().sum().backward()

        self.assertTrue(torch.allclose(y_implicit, z, atol=2e-5, rtol=2e-5))
        self.assertTrue(torch.allclose(x1.grad, x2.grad, atol=2e-4, rtol=2e-3))
        self.assertTrue(
            torch.allclose(
                implicit.recurrent.grad, explicit.recurrent.grad, atol=3e-4, rtol=3e-3
            )
        )

    def test_noncontractive_hard_stop(self):
        layer = EquilibriumLayer(3, 3, DEQConfig(fail_on_noncontractive=True))
        with torch.no_grad():
            layer.recurrent.copy_(torch.eye(3) * 1.2)
        with self.assertRaises(ContractivityError):
            layer(torch.randn(1, 3))

    def test_fail_closed_is_the_default(self):
        self.assertTrue(DEQConfig().fail_on_noncontractive)

    def test_noncontractive_fallback(self):
        layer = EquilibriumLayer(3, 3, DEQConfig(fail_on_noncontractive=False))
        with torch.no_grad():
            layer.recurrent.copy_(torch.eye(3) * 1.1)
        out = layer(torch.randn(1, 3))
        self.assertTrue(torch.isfinite(out).all())
        self.assertTrue(layer.last_info.used_fallback)

    def test_residual_metric_is_dimension_normalized(self):
        self.assertAlmostEqual(normalized_residual(torch.ones(2, 4)), 1.0)
        self.assertAlmostEqual(normalized_residual(torch.ones(2, 4000)), 1.0)

    @unittest.skipUnless(
        torch.cuda.is_available(), "requires CUDA for active-VRAM gate"
    )
    def test_cuda_active_memory_is_flat_across_search_depth(self):
        device = torch.device("cuda")
        layer = (
            EquilibriumLayer(32, 32, DEQConfig(max_iter=30, history=8))
            .to(device)
            .eval()
        )
        cts = CognitiveTreeSearch(layer, width=3).to(device)
        root = torch.zeros(1, 32, device=device)

        def action(state, index):
            return state + (index + 1) * 0.001

        def critic(state):
            return -state.square().mean()

        def peak(depth):
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            cts.search(root, action, critic, depth)
            torch.cuda.synchronize()
            return torch.cuda.max_memory_allocated()

        peak(2)  # allocator/kernel warm-up
        shallow, deep = peak(8), peak(64)
        self.assertLessEqual(deep - shallow, 8 * 1024 * 1024)


if __name__ == "__main__":
    unittest.main()
