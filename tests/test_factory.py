import unittest
from unittest.mock import patch

import torch
from torch import nn

from cogni_core.fast_weights import FastWeightBackboneWrapper
from cogni_os.config import load_config
from cogni_os.factory import build_genesis_runtime


class TestGenesisFactory(unittest.TestCase):
    @staticmethod
    def _backbone() -> nn.Linear:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return nn.Linear(4, 4).to(device)

    def assert_auxiliary_device(self, runtime, expected: torch.device) -> None:
        modules = (
            runtime.backbone.adapter,
            runtime.fast_weight_programmer,
            runtime.swarm,
            runtime.experts,
            runtime.meta_router,
        )
        for module in modules:
            with self.subTest(module=type(module).__name__):
                devices = {
                    tensor.device
                    for tensor in (*module.parameters(), *module.buffers())
                }
                self.assertEqual(devices, {expected})

    def test_default_factory_builds_all_bounded_components(self):
        backbone = self._backbone()
        runtime = build_genesis_runtime(
            backbone, load_config(), input_dim=4, state_dim=4
        )
        self.assertIsNotNone(runtime.sessions)
        self.assertIsNotNone(runtime.swarm)
        self.assertIsNotNone(runtime.experts)
        self.assertIsNotNone(runtime.expert_lifecycle)
        self.assertIs(runtime.expert_lifecycle.pool, runtime.experts)
        self.assertIsNotNone(runtime.meta_router)
        self.assertIsInstance(runtime.backbone, FastWeightBackboneWrapper)
        self.assertIs(runtime.sessions.model, runtime.backbone)
        self.assertIs(runtime.fast_weight_programmer, runtime.backbone.programmer)
        self.assertIsNone(runtime.verified_fast_weight)
        self.assertFalse(runtime.sessions.feature_enabled)
        self.assertEqual(runtime.sessions.session_ids, ())
        self.assertEqual(runtime.fast_weight_target, "adapter.core")
        self.assertEqual(runtime.backbone.adapter.bottleneck_dim, 4)
        self.assertEqual(runtime.fast_weight_programmer.internal_dim, 128)
        self.assertLess(
            float(
                torch.linalg.matrix_norm(
                    runtime.backbone.adapter.core.weight.detach().float(), ord=2
                )
            ),
            0.95,
        )
        self.assertEqual(runtime.experts.config.max_experts, 8)
        self.assertEqual(runtime.search_engine.config.width, 3)
        expected = next(backbone.parameters()).device
        self.assert_auxiliary_device(runtime, expected)
        self.assertEqual(runtime.vram_guard.device, expected)

    def test_cuda_request_falls_back_to_one_cpu_device_when_unavailable(self):
        backbone = nn.Linear(4, 4)
        with patch("cogni_os.factory.torch.cuda.is_available", return_value=False):
            runtime = build_genesis_runtime(
                backbone,
                load_config(),
                input_dim=4,
                state_dim=4,
            )

        self.assert_auxiliary_device(runtime, torch.device("cpu"))
        self.assertEqual(runtime.vram_guard.device, torch.device("cpu"))

    def test_low_precision_backbone_keeps_certificate_plane_fp32(self):
        backbone = nn.Linear(4, 4).to(dtype=torch.bfloat16)
        with patch("cogni_os.factory.torch.cuda.is_available", return_value=False):
            runtime = build_genesis_runtime(
                backbone,
                load_config(),
                input_dim=4,
                state_dim=4,
            )

        self.assertEqual(runtime.backbone.adapter.core.weight.dtype, torch.bfloat16)
        core_norm = float(
            torch.linalg.matrix_norm(
                runtime.backbone.adapter.core.weight.detach().float(), ord=2
            )
        )
        self.assertLessEqual(
            core_norm,
            runtime.backbone.adapter.core_operator_norm_budget + 1.0e-6,
        )
        latent = torch.zeros(1, 1, 4, dtype=torch.bfloat16)
        self.assertTrue(torch.isfinite(runtime.backbone.adapter(latent)).all())
        self.assertEqual(
            next(runtime.fast_weight_programmer.parameters()).dtype,
            torch.bfloat16,
        )
        for module in (runtime.swarm, runtime.experts, runtime.meta_router):
            with self.subTest(module=type(module).__name__):
                floating = [
                    tensor
                    for tensor in (*module.parameters(), *module.buffers())
                    if tensor.is_floating_point()
                ]
                self.assertTrue(floating)
                self.assertEqual({tensor.dtype for tensor in floating}, {torch.float32})

    def test_available_cuda_policy_rejects_a_cpu_backbone_fail_closed(self):
        with patch("cogni_os.factory.torch.cuda.is_available", return_value=True):
            with self.assertRaisesRegex(RuntimeError, "backbone device"):
                build_genesis_runtime(
                    nn.Linear(4, 4),
                    load_config(),
                    input_dim=4,
                    state_dim=4,
                )

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
    def test_cuda_backbone_and_every_auxiliary_share_the_same_device(self):
        backbone = nn.Linear(4, 4).cuda()
        runtime = build_genesis_runtime(
            backbone,
            load_config(),
            input_dim=4,
            state_dim=4,
        )
        expected = next(backbone.parameters()).device

        self.assertEqual(expected.type, "cuda")
        self.assert_auxiliary_device(runtime, expected)
        self.assertEqual(runtime.vram_guard.device, expected)


if __name__ == "__main__":
    unittest.main()
