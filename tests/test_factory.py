import unittest

import torch
from torch import nn

from cogni_core.fast_weights import FastWeightBackboneWrapper
from cogni_os.config import load_config
from cogni_os.factory import build_genesis_runtime


class TestGenesisFactory(unittest.TestCase):
    def test_default_factory_builds_all_bounded_components(self):
        runtime = build_genesis_runtime(
            nn.Linear(4, 4), load_config(), input_dim=4, state_dim=4
        )
        self.assertIsNotNone(runtime.sessions)
        self.assertIsNotNone(runtime.swarm)
        self.assertIsNotNone(runtime.experts)
        self.assertIsNotNone(runtime.meta_router)
        self.assertIsInstance(runtime.backbone, FastWeightBackboneWrapper)
        self.assertIs(runtime.sessions.model, runtime.backbone)
        self.assertIs(runtime.fast_weight_programmer, runtime.backbone.programmer)
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


if __name__ == "__main__":
    unittest.main()
