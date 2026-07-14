import unittest

import torch

from cogni_core.swarm import (
    SwarmConfig,
    SwarmContractivityError,
    SwarmTopologyError,
    TensorSwarm,
)


class TestTensorSwarm(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(3)
        self.config = SwarmConfig(input_dim=4, state_dim=8)

    def test_only_exact_certified_agent_partition_is_accepted(self):
        with self.assertRaisesRegex(ValueError, "exactly 28"):
            SwarmConfig(input_dim=4, agents=8)
        with self.assertRaisesRegex(ValueError, "exactly 11"):
            SwarmConfig(input_dim=4, sensory_agents=10)
        with self.assertRaisesRegex(ValueError, "exactly 7"):
            SwarmConfig(input_dim=4, constraint_agents=6)
        self.assertEqual(self.config.reasoning_agents, 10)

    def test_topologies_have_exact_edges_reachability_and_digest(self):
        swarm = TensorSwarm(self.config)
        normal, crisis = swarm.topology_certificates
        self.assertEqual(
            (normal.sensory_agents, normal.reasoning_agents, normal.constraint_agents),
            (11, 10, 7),
        )
        self.assertEqual(normal.edge_count, 192)
        self.assertEqual(crisis.edge_count, 28)
        self.assertLessEqual(normal.maximum_reachability_steps, self.config.warm_steps)
        self.assertLessEqual(crisis.maximum_reachability_steps, self.config.warm_steps)
        self.assertEqual(len(normal.sha256), 64)
        self.assertEqual(len(crisis.sha256), 64)
        self.assertEqual(int(torch.triu(swarm.normal_topology).count_nonzero()), 0)
        self.assertEqual(int(torch.triu(swarm.crisis_topology).count_nonzero()), 0)

    def test_public_topologies_are_copies_and_private_mutation_fails_closed(self):
        swarm = TensorSwarm(self.config)
        public = swarm.normal_topology
        public[1, 0] = ~public[1, 0]
        self.assertNotEqual(public[1, 0], swarm.normal_topology[1, 0])
        with torch.no_grad():
            swarm._normal_topology[1, 0] = ~swarm._normal_topology[1, 0]
        with self.assertRaises(SwarmTopologyError):
            swarm(torch.zeros(1, 4))

    def test_operator_certificate_is_fp32_global_and_strict(self):
        swarm = TensorSwarm(self.config)
        for name in ("normal", "crisis"):
            certificate = swarm.operator_certificate(name)
            self.assertEqual(
                certificate.global_operator_norm_bound.dtype, torch.float32
            )
            self.assertEqual(
                certificate.global_operator_norm_estimate.dtype, torch.float32
            )
            self.assertTrue(certificate.certified)
            self.assertLess(
                float(certificate.global_operator_norm_bound),
                self.config.global_margin,
            )
            self.assertLessEqual(
                float(certificate.global_operator_norm_estimate),
                float(certificate.global_operator_norm_bound) + 1e-5,
            )
            radius = swarm.global_spectral_radius(name)
            self.assertTrue(torch.isfinite(radius))
            self.assertLessEqual(
                float(radius), float(certificate.local_norm_max) + 1e-5
            )

    def test_pcas_requires_persistence_and_does_not_change_weights(self):
        swarm = TensorSwarm(self.config).eval()
        calibration = torch.randn(512, 4) * 0.1
        swarm.monitor.fit(calibration)
        before = [parameter.detach().clone() for parameter in swarm.parameters()]
        normal = swarm(torch.zeros(2, 4))
        self.assertEqual(int(normal.regime), 0)
        state = normal.pcas_state
        output = normal
        for _ in range(swarm.monitor.required_enter_streak):
            output = swarm(torch.full((2, 4), 20.0), pcas_state=state)
            state = output.pcas_state
        self.assertEqual(int(output.regime), 1)
        self.assertEqual(tuple(output.latent.shape), (2, 8))
        self.assertTrue(
            all(
                torch.equal(first, second)
                for first, second in zip(before, swarm.parameters())
            )
        )

    def test_solver_reports_real_convergence_and_warm_iteration_count(self):
        swarm = TensorSwarm(self.config).eval()
        observation = torch.randn(2, 4) * 0.1
        cold = swarm(observation)
        self.assertTrue(bool(cold.converged))
        self.assertTrue(bool(cold.safe_for_advice))
        self.assertLessEqual(int(cold.iterations), self.config.cold_steps)
        self.assertLessEqual(float(cold.residual.max()), self.config.residual_tolerance)
        warm = swarm(
            observation + 1e-4,
            cold.joint_state,
            pcas_state=cold.pcas_state,
        )
        self.assertTrue(bool(warm.converged))
        self.assertLessEqual(int(warm.iterations), self.config.warm_steps)
        self.assertTrue(torch.isfinite(warm.residual).all())

    def test_unconverged_solve_returns_zero_advisory_and_safe_prior(self):
        config = SwarmConfig(
            input_dim=4,
            state_dim=8,
            cold_steps=3,
            warm_steps=3,
            residual_tolerance=1e-8,
        )
        swarm = TensorSwarm(config).eval()
        output = swarm(torch.ones(2, 4))
        self.assertFalse(bool(output.converged))
        self.assertFalse(bool(output.safe_for_advice))
        self.assertTrue(torch.equal(output.latent, torch.zeros_like(output.latent)))
        self.assertTrue(
            torch.equal(output.joint_state, torch.zeros_like(output.joint_state))
        )

    def test_forward_projects_unsafe_finite_weights_and_rejects_nan(self):
        swarm = TensorSwarm(self.config).eval()
        recurrent_version = swarm.recurrent._version
        coupling_version = swarm.coupling._version
        # Deliberately bypass autograd's version counter: the content sentinel
        # must still invalidate the cached global certificate.
        swarm.recurrent.data.mul_(25.0)
        swarm.coupling.data.mul_(25.0)
        self.assertEqual(swarm.recurrent._version, recurrent_version)
        self.assertEqual(swarm.coupling._version, coupling_version)
        result = swarm(torch.randn(2, 4) * 0.1)
        self.assertTrue(torch.isfinite(result.latent).all())
        self.assertLess(
            float(swarm.max_local_spectral_norm().detach()), self.config.local_margin
        )
        self.assertLess(
            float(swarm.max_coupling_spectral_norm().detach()),
            self.config.coupling_scale,
        )
        self.assertTrue(swarm.operator_certificate("crisis").certified)

        with torch.no_grad():
            swarm.recurrent[0, 0, 0] = float("nan")
        with self.assertRaises(SwarmContractivityError):
            swarm(torch.randn(1, 4))

    def test_adversarial_finite_inputs_never_expose_nonfinite_latent(self):
        swarm = TensorSwarm(self.config).eval()
        values = torch.tensor(
            [0.0, 1e-20, -1e-20, 1.0, -1.0, 1e6, -1e6], dtype=torch.float32
        )
        observations = values.repeat(20).reshape(-1, 4)
        output = swarm(observations)
        self.assertTrue(torch.isfinite(output.latent).all())
        self.assertTrue(torch.isfinite(output.joint_state).all())
        self.assertTrue(torch.isfinite(output.residual).all())
        if not bool(output.converged):
            self.assertTrue(torch.equal(output.latent, torch.zeros_like(output.latent)))


if __name__ == "__main__":
    unittest.main()
