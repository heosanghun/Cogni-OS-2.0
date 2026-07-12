from concurrent.futures import ThreadPoolExecutor
import unittest

import torch

from cogni_core.swarm import SwarmConfig, TensorSwarm
from cogni_core.swarm_sessions import (
    PCASMonitor,
    SwarmSessionState,
    SwarmSessionStateCache,
)


class _Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


class TestPCASMonitor(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(19)
        self.monitor = PCASMonitor(4)
        self.monitor.fit(torch.randn(1024, 4))

    def test_hysteresis_rejects_a_single_spike(self):
        state = self.monitor.initial_state()
        output = self.monitor(torch.full((1, 4), 100.0), state)
        self.assertEqual(int(output.regime), 0)
        self.assertEqual(int(output.state.enter_streak), 1)

    def test_persistent_drift_enters_and_exits_only_after_streaks(self):
        state = self.monitor.initial_state()
        for _ in range(self.monitor.required_enter_streak):
            output = self.monitor(torch.full((1, 4), 10.0), state)
            state = output.state
        self.assertEqual(int(state.regime), 1)
        # Minimum dwell and the lower exit threshold prevent immediate chatter.
        for _ in range(
            self.monitor.minimum_dwell + self.monitor.required_exit_streak + 2
        ):
            output = self.monitor(torch.zeros(1, 4), state)
            state = output.state
        self.assertEqual(int(state.regime), 0)
        self.assertEqual(int(state.switch_count), 2)

        # Historical drift may decay slowly, but low-band evidence cannot
        # trigger a second crisis after recovery.
        for _ in range(256):
            output = self.monitor(torch.zeros(1, 4), state)
            state = output.state
        self.assertEqual(int(state.regime), 0)
        self.assertEqual(int(state.switch_count), 2)

    def test_ten_thousand_alternating_adversarial_samples_have_bounded_switches(self):
        state = self.monitor.initial_state()
        high = torch.full((1, 4), 10.0)
        low = torch.zeros(1, 4)
        for index in range(10_000):
            output = self.monitor(high if index % 2 == 0 else low, state)
            state = output.state
            self.assertTrue(torch.isfinite(output.distances).all())
            self.assertTrue(torch.isfinite(state.drift_ema))
        # Alternating input can cause at most one persistent crisis entry; it
        # cannot assemble the consecutive low evidence required for exit.
        self.assertLessEqual(int(state.switch_count), 1)

    def test_calibration_rejects_nonfinite_and_remains_fp32(self):
        with self.assertRaises(ValueError):
            self.monitor.fit(torch.full((32, 4), float("nan")))
        output = self.monitor(torch.ones(2, 4, dtype=torch.float64))
        self.assertEqual(output.distances.dtype, torch.float32)

        degenerate = PCASMonitor(4)
        degenerate.fit(torch.zeros(32, 4))
        self.assertLess(
            float(degenerate.exit_threshold), float(degenerate.enter_threshold)
        )

    def test_every_finite_fp32_observation_has_finite_saturated_telemetry(self):
        maximum = torch.finfo(torch.float32).max * 0.9
        observation = torch.tensor([[maximum, -maximum, maximum, -maximum]])
        output = self.monitor(observation)
        self.assertTrue(torch.isfinite(output.distances).all())
        self.assertTrue(torch.isfinite(output.state.drift_ema))

    def test_ten_thousand_adversarial_finite_vectors_remain_finite(self):
        values = torch.linspace(-1e20, 1e20, 10_000, dtype=torch.float32)
        observations = torch.stack((values, -values, values, -values), dim=-1)
        output = self.monitor(observations)
        self.assertEqual(output.distances.shape, (10_000,))
        self.assertTrue(torch.isfinite(output.distances).all())
        self.assertTrue(torch.isfinite(output.state.drift_ema))

    def test_calibrated_distance_separates_heldout_shift_without_claim_inflation(self):
        generator = torch.Generator().manual_seed(1901)
        heldout_normal = torch.randn(10_000, 4, generator=generator)
        heldout_shift = torch.randn(10_000, 4, generator=generator) + 6.0
        normal = self.monitor(heldout_normal).distances
        shifted = self.monitor(heldout_shift).distances
        false_positive_rate = float(
            (normal >= self.monitor.enter_threshold).float().mean()
        )
        detection_rate = float((shifted >= self.monitor.enter_threshold).float().mean())
        self.assertLess(false_positive_rate, 0.03)
        self.assertGreater(detection_rate, 0.99)


class TestSwarmSessionStateCache(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(23)
        self.swarm = TensorSwarm(SwarmConfig(input_dim=4, state_dim=8)).eval()
        self.clock = _Clock()
        self.cache = SwarmSessionStateCache(
            max_sessions=2,
            ttl_seconds=10.0,
            max_state_bytes=1_000_000,
            clock=self.clock,
        )

    def _state(self, value: float) -> SwarmSessionState:
        output = self.swarm(torch.full((1, 4), value))
        self.assertTrue(bool(output.safe_for_advice))
        return SwarmSessionState(output.joint_state, output.pcas_state)

    def test_cache_is_lru_ttl_and_copy_isolated(self):
        self.cache.put("a", self._state(0.01))
        self.clock.now = 1.0
        self.cache.put("b", self._state(0.02))
        self.assertIsNotNone(self.cache.get("a"))
        self.clock.now = 2.0
        self.cache.put("c", self._state(0.03))
        self.assertIsNone(self.cache.get("b"))
        owned = self.cache.get("a")
        assert owned is not None
        owned.joint_state.add_(999)
        self.assertFalse(
            torch.equal(owned.joint_state, self.cache.get("a").joint_state)
        )
        self.clock.now = 20.0
        self.assertEqual(self.cache.session_count, 0)
        self.assertEqual(self.cache.storage_bytes, 0)

    def test_process_keeps_sessions_independent_and_commits_safe_state(self):
        first = self.cache.process("alpha", self.swarm, torch.full((1, 4), 0.01))
        second = self.cache.process("beta", self.swarm, torch.full((1, 4), -0.01))
        self.assertTrue(bool(first.safe_for_advice))
        self.assertTrue(bool(second.safe_for_advice))
        alpha = self.cache.get("alpha")
        beta = self.cache.get("beta")
        assert alpha is not None and beta is not None
        self.assertFalse(torch.equal(alpha.joint_state, beta.joint_state))
        warm = self.cache.process("alpha", self.swarm, torch.full((1, 4), 0.01001))
        self.assertLessEqual(int(warm.iterations), self.swarm.config.warm_steps)

    def test_threaded_updates_preserve_capacity_and_byte_invariants(self):
        states = [self._state(index / 100.0) for index in range(8)]

        def update(index: int) -> None:
            self.cache.put(f"session-{index}", states[index])
            self.cache.get(f"session-{index}")

        with ThreadPoolExecutor(max_workers=4) as pool:
            list(pool.map(update, range(8)))
        self.assertLessEqual(self.cache.session_count, self.cache.max_sessions)
        self.assertLessEqual(self.cache.storage_bytes, self.cache.max_state_bytes)

    def test_failed_advisory_is_not_committed(self):
        unsafe = TensorSwarm(
            SwarmConfig(
                input_dim=4,
                state_dim=8,
                cold_steps=3,
                warm_steps=3,
                residual_tolerance=1e-8,
            )
        ).eval()
        output = self.cache.process("unsafe", unsafe, torch.ones(1, 4))
        self.assertFalse(bool(output.safe_for_advice))
        self.assertIsNone(self.cache.get("unsafe"))


if __name__ == "__main__":
    unittest.main()
