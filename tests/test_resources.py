import unittest
from unittest.mock import patch

import torch

from cogni_core.resources import (
    MAX_VRAM_GIB,
    MemorySnapshot,
    ResourceBudgetExceeded,
    VRAMGuard,
)


class TestVRAMGuard(unittest.TestCase):
    def test_limit_above_absolute_ceiling_is_rejected(self):
        with self.assertRaisesRegex(ValueError, str(MAX_VRAM_GIB)):
            VRAMGuard(MAX_VRAM_GIB + 0.1, device="cpu")

    def test_negative_estimate_is_rejected_even_on_cpu(self):
        with self.assertRaises(ValueError):
            VRAMGuard(device="cpu").admit(-1)

    def test_cpu_mode_is_noop_but_auditable(self):
        guard = VRAMGuard(0.001, "cpu")
        self.assertFalse(guard.enabled)
        with guard.enforce(10**12):
            pass

    def test_cuda_admission_releases_unused_cache_then_rechecks_free_memory(self):
        guard = VRAMGuard(1.0, "cuda")
        snapshot = MemorySnapshot(128, 900, 128, 2 * 1024**3)
        with (
            patch("cogni_core.resources.torch.cuda.is_available", return_value=True),
            patch.object(guard, "snapshot", return_value=snapshot),
            patch(
                "cogni_core.resources.torch.cuda.mem_get_info",
                side_effect=((0, 2 * 1024**3), (1024, 2 * 1024**3)),
            ),
            patch("cogni_core.resources.torch.cuda.empty_cache") as empty_cache,
        ):
            guard.admit(512)
        empty_cache.assert_called_once_with()

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
    def test_cuda_admission_rejects_impossible_request(self):
        guard = VRAMGuard(0.000001, "cuda")
        with self.assertRaises(ResourceBudgetExceeded):
            guard.admit(1024**2)


if __name__ == "__main__":
    unittest.main()
