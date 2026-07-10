from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from threading import RLock
from typing import Iterator

import torch


class ResourceBudgetExceeded(RuntimeError):
    pass


MAX_VRAM_GIB = 16.7


@dataclass(frozen=True)
class MemorySnapshot:
    allocated_bytes: int
    reserved_bytes: int
    peak_allocated_bytes: int
    device_total_bytes: int


class VRAMGuard:
    """Admission and postcondition checks for the inference VRAM envelope."""

    def __init__(
        self, limit_gib: float = 16.7, device: torch.device | str | None = None
    ):
        if not 0.0 < float(limit_gib) <= MAX_VRAM_GIB:
            raise ValueError(f"limit_gib must be in (0, {MAX_VRAM_GIB}]")
        self.limit_bytes = int(limit_gib * 1024**3)
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        # CUDA peak counters are process-global per device. Serialising guarded
        # regions prevents concurrent requests from resetting each other's
        # telemetry and makes the hard postcondition deterministic.
        self._lock = RLock()

    @property
    def enabled(self) -> bool:
        return self.device.type == "cuda" and torch.cuda.is_available()

    def snapshot(self) -> MemorySnapshot:
        if not self.enabled:
            return MemorySnapshot(0, 0, 0, 0)
        props = torch.cuda.get_device_properties(self.device)
        return MemorySnapshot(
            torch.cuda.memory_allocated(self.device),
            torch.cuda.memory_reserved(self.device),
            torch.cuda.max_memory_allocated(self.device),
            props.total_memory,
        )

    def admit(self, estimated_additional_bytes: int = 0) -> None:
        if estimated_additional_bytes < 0:
            raise ValueError("estimated_additional_bytes cannot be negative")
        if not self.enabled:
            return
        snap = self.snapshot()
        if snap.allocated_bytes + estimated_additional_bytes > self.limit_bytes:
            raise ResourceBudgetExceeded(
                f"VRAM admission rejected: allocated={snap.allocated_bytes}, "
                f"requested={estimated_additional_bytes}, limit={self.limit_bytes}"
            )
        free_bytes, _ = torch.cuda.mem_get_info(self.device)
        if estimated_additional_bytes > free_bytes:
            raise ResourceBudgetExceeded(
                f"VRAM admission rejected by physical free memory: "
                f"free={free_bytes}, requested={estimated_additional_bytes}"
            )

    def assert_within_limit(self) -> None:
        """Fail if current or peak active allocation crossed the hard ceiling."""

        if not self.enabled:
            return
        snap = self.snapshot()
        observed = max(snap.allocated_bytes, snap.peak_allocated_bytes)
        if observed > self.limit_bytes:
            raise ResourceBudgetExceeded(
                f"VRAM postcondition failed: observed={observed}, "
                f"limit={self.limit_bytes}"
            )

    @contextmanager
    def enforce(self, estimated_additional_bytes: int = 0) -> Iterator[None]:
        with self._lock:
            self.admit(estimated_additional_bytes)
            if self.enabled:
                torch.cuda.reset_peak_memory_stats(self.device)
            try:
                yield
            except torch.OutOfMemoryError as exc:
                if self.enabled:
                    torch.cuda.empty_cache()
                raise ResourceBudgetExceeded(
                    "CUDA allocator rejected the guarded operation"
                ) from exc
            self.assert_within_limit()


def module_storage_bytes(module: torch.nn.Module) -> int:
    """Exact parameter and persistent-buffer bytes owned by ``module``."""

    tensors = list(module.parameters()) + list(module.buffers())
    return sum(tensor.numel() * tensor.element_size() for tensor in tensors)
