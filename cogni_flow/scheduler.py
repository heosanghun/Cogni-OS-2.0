from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from threading import Lock, RLock
from time import monotonic
from typing import TYPE_CHECKING, Any, Callable

from .aflow import WorkflowSpec
from .cycle import SelfHarness
from .orchestrator import WorkflowEvolutionCoordinator
from .rhythm import RhythmController, SystemMode

if TYPE_CHECKING:
    from .aflow_research import ResearchWorkflowCoordinator


class ScheduleDecision(str, Enum):
    NOT_IDLE = "not_idle"
    INFERENCE_ACTIVE = "inference_active"
    MODE_BLOCKED = "mode_blocked"
    CYCLE_ACTIVE = "cycle_active"
    RAN = "ran"


@dataclass(frozen=True)
class ScheduleTick:
    decision: ScheduleDecision
    idle_for: float
    result: Any | None = None

    @property
    def ran(self) -> bool:
        return self.decision == ScheduleDecision.RAN


class IdleNightScheduler:
    """Cooperative monotonic-idle gate for one bounded night cycle.

    The scheduler intentionally owns no sleeping background loop. A service
    supervisor calls ``tick`` (or the identical ``run_once``) at its chosen
    cadence, which keeps time and shutdown behavior externally auditable.
    """

    def __init__(
        self,
        rhythm: RhythmController,
        cycle_runner: Callable[[], Any],
        *,
        idle_seconds: float,
        clock: Callable[[], float] = monotonic,
    ) -> None:
        if idle_seconds < 0:
            raise ValueError("idle_seconds cannot be negative")
        self.rhythm = rhythm
        self._cycle_runner = cycle_runner
        self.idle_seconds = idle_seconds
        self._clock = clock
        self._state_lock = RLock()
        self._cycle_lock = Lock()
        self._last_activity = float(clock())

    @classmethod
    def for_self_harness(
        cls,
        harness: SelfHarness,
        *,
        idle_seconds: float,
        since: float = 0.0,
        clock: Callable[[], float] = monotonic,
    ) -> IdleNightScheduler:
        return cls(
            harness.rhythm,
            lambda: harness.run_night_cycle(since=since),
            idle_seconds=idle_seconds,
            clock=clock,
        )

    @classmethod
    def for_workflow(
        cls,
        coordinator: WorkflowEvolutionCoordinator,
        initial: WorkflowSpec,
        *,
        idle_seconds: float,
        clock: Callable[[], float] = monotonic,
    ) -> IdleNightScheduler:
        return cls(
            coordinator.rhythm,
            lambda: coordinator.run(initial),
            idle_seconds=idle_seconds,
            clock=clock,
        )

    @classmethod
    def for_research_workflow(
        cls,
        coordinator: ResearchWorkflowCoordinator,
        initial: WorkflowSpec,
        *,
        idle_seconds: float,
        clock: Callable[[], float] = monotonic,
    ) -> IdleNightScheduler:
        """Schedule one archive-only Phase-10 workflow research cycle."""

        return cls(
            coordinator.rhythm,
            lambda: coordinator.run(initial),
            idle_seconds=idle_seconds,
            clock=clock,
        )

    @property
    def last_activity(self) -> float:
        with self._state_lock:
            return self._last_activity

    def note_activity(self) -> None:
        """Reset the idle window from an inference/request boundary."""

        now = float(self._clock())
        with self._state_lock:
            self._last_activity = now

    def run_once(self) -> ScheduleTick:
        now, idle_for = self._idle_snapshot()
        if idle_for < self.idle_seconds:
            return ScheduleTick(ScheduleDecision.NOT_IDLE, idle_for)
        if not self._cycle_lock.acquire(blocking=False):
            return ScheduleTick(ScheduleDecision.CYCLE_ACTIVE, idle_for)

        attempted = False
        try:
            now, idle_for = self._idle_snapshot()
            if idle_for < self.idle_seconds:
                return ScheduleTick(ScheduleDecision.NOT_IDLE, idle_for)
            if self.rhythm.mode != SystemMode.INFERENCE:
                return ScheduleTick(ScheduleDecision.MODE_BLOCKED, idle_for)
            if self.rhythm.active_requests:
                self.note_activity()
                return ScheduleTick(ScheduleDecision.INFERENCE_ACTIVE, idle_for)

            # SelfHarness/WorkflowEvolutionCoordinator own the legal sequence
            # INFERENCE -> DRAINING -> CHECKPOINTING -> EVOLUTION. The scheduler
            # never mutates the mode around them or bypasses their re-check of
            # active inference requests.
            attempted = True
            result = self._cycle_runner()
            return ScheduleTick(ScheduleDecision.RAN, idle_for, result)
        finally:
            if attempted:
                with self._state_lock:
                    completed_at = float(self._clock())
                    self._last_activity = max(now, completed_at)
            self._cycle_lock.release()

    def tick(self) -> ScheduleTick:
        return self.run_once()

    def _idle_snapshot(self) -> tuple[float, float]:
        now = float(self._clock())
        with self._state_lock:
            if now < self._last_activity:
                # A malformed injected clock must not manufacture idle time.
                self._last_activity = now
                return now, 0.0
            return now, now - self._last_activity
