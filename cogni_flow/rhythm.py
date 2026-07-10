from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from threading import RLock
from time import time
from typing import Iterator


class SystemMode(str, Enum):
    INFERENCE = "inference"
    DRAINING = "draining"
    CHECKPOINTING = "checkpointing"
    EVOLUTION = "evolution"
    VALIDATING = "validating"
    PROMOTING = "promoting"
    ROLLING_BACK = "rolling_back"
    SAFE_MODE = "safe_mode"


_ALLOWED: dict[SystemMode, set[SystemMode]] = {
    SystemMode.INFERENCE: {SystemMode.DRAINING, SystemMode.SAFE_MODE},
    SystemMode.DRAINING: {
        SystemMode.CHECKPOINTING,
        SystemMode.INFERENCE,
        SystemMode.SAFE_MODE,
    },
    SystemMode.CHECKPOINTING: {
        SystemMode.EVOLUTION,
        SystemMode.ROLLING_BACK,
        SystemMode.SAFE_MODE,
    },
    SystemMode.EVOLUTION: {
        SystemMode.VALIDATING,
        SystemMode.INFERENCE,
        SystemMode.SAFE_MODE,
    },
    SystemMode.VALIDATING: {
        SystemMode.PROMOTING,
        SystemMode.EVOLUTION,
        SystemMode.ROLLING_BACK,
    },
    SystemMode.PROMOTING: {SystemMode.INFERENCE, SystemMode.ROLLING_BACK},
    SystemMode.ROLLING_BACK: {SystemMode.INFERENCE, SystemMode.SAFE_MODE},
    SystemMode.SAFE_MODE: {SystemMode.INFERENCE},
}


@dataclass(frozen=True)
class Transition:
    source: SystemMode
    target: SystemMode
    timestamp: float
    reason: str


@dataclass
class RhythmController:
    mode: SystemMode = SystemMode.INFERENCE
    active_requests: int = 0
    history: list[Transition] = field(default_factory=list)
    active_evolution_tasks: int = field(default=0, init=False)
    _lock: RLock = field(default_factory=RLock, repr=False)

    def transition(self, target: SystemMode, reason: str) -> None:
        with self._lock:
            self._transition_locked(target, reason)

    def _transition_locked(self, target: SystemMode, reason: str) -> None:
        if target not in _ALLOWED[self.mode]:
            raise RuntimeError(
                f"illegal mode transition: {self.mode.value} -> {target.value}"
            )
        if target == SystemMode.CHECKPOINTING and self.active_requests:
            raise RuntimeError("cannot checkpoint while inference requests are active")
        if target == SystemMode.INFERENCE and self.active_evolution_tasks:
            raise RuntimeError(
                "cannot resume inference while evolution tasks are active"
            )
        source = self.mode
        self.mode = target
        self.history.append(Transition(source, target, time(), reason))

    @contextmanager
    def inference_slot(self) -> Iterator[None]:
        with self._lock:
            if self.mode != SystemMode.INFERENCE:
                raise RuntimeError(f"inference unavailable in {self.mode.value} mode")
            self.active_requests += 1
        try:
            yield
        finally:
            with self._lock:
                self.active_requests -= 1

    @contextmanager
    def evolution_slot(self) -> Iterator[None]:
        """Track one bounded evolution operation for its complete lifetime.

        Validation and promotion are evolution work too.  Keeping the counter
        live across those modes closes the race in which another thread could
        resume inference while a candidate is still executing or being
        atomically installed.
        """

        with self._lock:
            if self.mode not in {
                SystemMode.EVOLUTION,
                SystemMode.VALIDATING,
                SystemMode.PROMOTING,
            }:
                raise RuntimeError(
                    f"evolution work unavailable in {self.mode.value} mode"
                )
            self.active_evolution_tasks += 1
        try:
            yield
        finally:
            with self._lock:
                self.active_evolution_tasks -= 1

    # A descriptive alias for callers that model work rather than slots.  Keep
    # ``evolution_slot`` as the stable API used by the harness.
    evolution_task = evolution_slot

    def enter_evolution(self, checkpoint_callback) -> None:
        with self._lock:
            self._transition_locked(SystemMode.DRAINING, "night cycle requested")
            if self.active_requests:
                self._transition_locked(
                    SystemMode.INFERENCE, "drain rejected: requests remain"
                )
                raise RuntimeError(
                    "active inference requests must drain before evolution"
                )
            self._transition_locked(SystemMode.CHECKPOINTING, "inference drained")
        try:
            checkpoint_callback()
        except Exception:
            with self._lock:
                self._transition_locked(SystemMode.ROLLING_BACK, "checkpoint failed")
                self._transition_locked(
                    SystemMode.INFERENCE, "checkpoint rollback complete"
                )
            raise
        with self._lock:
            self._transition_locked(SystemMode.EVOLUTION, "checkpoint complete")

    def resume_inference(self, reason: str = "night cycle complete") -> None:
        with self._lock:
            if self.mode not in {SystemMode.EVOLUTION, SystemMode.PROMOTING}:
                raise RuntimeError(f"cannot resume inference from {self.mode.value}")
            self._transition_locked(SystemMode.INFERENCE, reason)
