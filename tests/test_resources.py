import unittest

import torch

from cogni_core.resources import MAX_VRAM_GIB, ResourceBudgetExceeded, VRAMGuard


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

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
    def test_cuda_admission_rejects_impossible_request(self):
        guard = VRAMGuard(0.000001, "cuda")
        with self.assertRaises(ResourceBudgetExceeded):
            guard.admit(1024**2)


if __name__ == "__main__":
    unittest.main()
