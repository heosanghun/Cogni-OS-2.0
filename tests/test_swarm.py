import unittest

import torch

from cogni_core.swarm import SwarmConfig, SwarmContractivityError, TensorSwarm


class TestTensorSwarm(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(3)
        self.config = SwarmConfig(
            input_dim=4, state_dim=8, agents=8, sensory_agents=3, constraint_agents=2
        )

    def test_topologies_are_acyclic_and_contractivity_is_bounded(self):
        swarm = TensorSwarm(self.config)
        self.assertEqual(int(torch.triu(swarm.normal_topology).count_nonzero()), 0)
        self.assertEqual(int(torch.triu(swarm.crisis_topology).count_nonzero()), 0)
        self.assertLessEqual(
            float(swarm.max_local_spectral_norm().detach()),
            self.config.local_margin + 1e-5,
        )

    def test_pcas_switches_without_changing_weights(self):
        swarm = TensorSwarm(self.config).eval()
        calibration = torch.randn(256, 4) * 0.1
        swarm.monitor.fit(calibration)
        before = [p.detach().clone() for p in swarm.parameters()]
        normal = swarm(torch.zeros(2, 4))
        crisis = swarm(torch.full((2, 4), 20.0), normal.joint_state)
        self.assertEqual(int(normal.regime), 0)
        self.assertEqual(int(crisis.regime), 1)
        self.assertEqual(tuple(crisis.latent.shape), (2, 8))
        self.assertTrue(
            all(torch.equal(a, b) for a, b in zip(before, swarm.parameters()))
        )

    def test_warm_start_uses_fixed_short_path(self):
        swarm = TensorSwarm(self.config).eval()
        cold = swarm(torch.randn(2, 4))
        warm = swarm(torch.randn(2, 4), cold.joint_state)
        self.assertEqual(int(cold.iterations), self.config.cold_steps)
        self.assertEqual(int(warm.iterations), self.config.warm_steps)
        self.assertTrue(torch.isfinite(warm.residual).all())

    def test_forward_enforces_recurrent_and_coupling_cfire(self):
        swarm = TensorSwarm(self.config).eval()
        with torch.no_grad():
            swarm.recurrent.mul_(25.0)
            swarm.coupling.mul_(25.0)
        result = swarm(torch.randn(2, 4))
        self.assertTrue(torch.isfinite(result.latent).all())
        self.assertLess(
            float(swarm.max_local_spectral_norm().detach()), self.config.local_margin
        )
        self.assertLess(
            float(swarm.max_coupling_spectral_norm().detach()),
            self.config.coupling_scale,
        )

        with torch.no_grad():
            swarm.recurrent[0, 0, 0] = float("nan")
        with self.assertRaises(SwarmContractivityError):
            swarm(torch.randn(1, 4))


if __name__ == "__main__":
    unittest.main()
