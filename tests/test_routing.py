import unittest

import torch

from cogni_core.routing import ContrastiveSessionRouter


class TestContrastiveSessionRouter(unittest.TestCase):
    def test_near_queries_use_fast_path_and_ood_falls_back(self):
        torch.manual_seed(5)
        router = ContrastiveSessionRouter(minimum_threshold=0.03)
        anchors = torch.tensor([[1.0, 0.01], [1.0, -0.01], [0.99, 0.0]])
        router.calibrate("s", anchors)
        self.assertTrue(router.route("s", torch.tensor([1.0, 0.0])).allow_fast_path)
        self.assertFalse(router.route("s", torch.tensor([0.0, 1.0])).allow_fast_path)

    def test_session_capacity_is_bounded(self):
        router = ContrastiveSessionRouter(max_sessions=2)
        samples = torch.tensor([[1.0, 0.0], [0.99, 0.01]])
        router.calibrate("a", samples)
        router.calibrate("b", samples)
        self.assertEqual(router.calibrate("c", samples), ("a",))
        self.assertFalse(router.route("a", samples[0]).allow_fast_path)

    def test_hot_path_returns_only_tensor_fields(self):
        router = ContrastiveSessionRouter()
        samples = torch.tensor([[1.0, 0.0], [0.99, 0.01]])
        router.calibrate("s", samples)
        decision = router.route_tensor("s", samples[0])
        self.assertIsInstance(decision.allow_fast_path, torch.Tensor)
        self.assertIsInstance(decision.distance, torch.Tensor)
        self.assertIsInstance(decision.threshold, torch.Tensor)
        self.assertEqual(decision.allow_fast_path.dtype, torch.bool)


if __name__ == "__main__":
    unittest.main()
