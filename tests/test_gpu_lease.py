from __future__ import annotations

from threading import Barrier, Event, Lock, Thread
import unittest

from cogni_os.gpu_lease import (
    ExpiredGPULeaseError,
    GPULeaseBudgetError,
    GPULeaseBusyError,
    GPULeaseManager,
    StaleGPULeaseError,
)


class _Clock:
    def __init__(self, value: float = 100.0) -> None:
        self.value = value

    def __call__(self) -> float:
        return self.value

    def advance(self, seconds: float) -> None:
        self.value += seconds


class TestGPULeaseManager(unittest.TestCase):
    def manager(self, clock: _Clock | None = None, **kwargs) -> GPULeaseManager:
        active_clock = clock or _Clock()
        return GPULeaseManager(clock=active_clock, **kwargs)

    def test_deadline_helper_uses_the_authority_clock_domain(self) -> None:
        clock = _Clock(250.0)
        manager = self.manager(clock)

        self.assertEqual(manager.deadline_after(12.5), 262.5)
        for invalid in (0, -1, float("inf"), float("nan"), True):
            with self.subTest(invalid=invalid):
                with self.assertRaises(ValueError):
                    manager.deadline_after(invalid)

    def test_single_owner_has_epoch_purpose_budget_and_deadline(self) -> None:
        clock = _Clock()
        manager = self.manager(clock, max_vram_bytes=1_000)
        lease = manager.acquire(
            "resident-model",
            "inference",
            800,
            deadline=clock() + 30,
        )

        self.assertEqual(lease.owner, "resident-model")
        self.assertEqual(lease.purpose, "inference")
        self.assertEqual(lease.epoch, 1)
        self.assertEqual(lease.vram_budget_bytes, 800)
        self.assertEqual(lease.deadline, 130.0)
        self.assertEqual(lease.ttl_seconds, 30.0)
        self.assertEqual(manager.active, lease)
        with self.assertRaises(GPULeaseBusyError):
            manager.acquire(
                "validator",
                "validation",
                500,
                deadline=clock() + 10,
            )

    def test_stale_epoch_cannot_release_or_revoke_new_owner(self) -> None:
        clock = _Clock()
        manager = self.manager(clock)
        first = manager.acquire("agent", "inference", 100, deadline=clock() + 10)
        manager.release(first)
        second = manager.acquire("night", "evolution", 200, deadline=clock() + 20)

        self.assertEqual(second.epoch, first.epoch + 1)
        with self.assertRaises(StaleGPULeaseError):
            manager.release(first)
        with self.assertRaises(StaleGPULeaseError):
            manager.revoke(first, "late_worker_cleanup")
        self.assertEqual(manager.active, second)

    def test_expired_alive_owner_remains_fenced_until_death_is_confirmed(
        self,
    ) -> None:
        clock = _Clock()
        alive = [True]
        manager = self.manager(clock)
        expired = manager.acquire(
            "validator",
            "validation",
            100,
            deadline=clock() + 5,
            owner_alive=lambda: alive[0],
        )
        clock.advance(5)

        self.assertEqual(manager.active, expired)
        self.assertIsNone(manager.reap())
        self.assertEqual(manager.history, ())
        with self.assertRaises(ExpiredGPULeaseError):
            manager.validate(expired)
        with self.assertRaises(GPULeaseBusyError):
            manager.acquire("agent", "inference", 100, deadline=clock() + 5)

        alive[0] = False
        event = manager.reap()
        self.assertIsNotNone(event)
        assert event is not None
        self.assertEqual(event.lease, expired)
        self.assertEqual(event.reason, "owner_confirmed_dead")
        self.assertIsNone(manager.active)
        with self.assertRaises(StaleGPULeaseError):
            manager.release(expired)

        replacement = manager.acquire("agent", "inference", 100, deadline=clock() + 5)
        self.assertGreater(replacement.epoch, expired.epoch)

    def test_expired_owner_without_health_probe_stays_fenced(self) -> None:
        clock = _Clock()
        manager = self.manager(clock)
        expired = manager.acquire("worker", "validation", 100, deadline=clock() + 1)
        clock.advance(1)

        self.assertEqual(manager.active, expired)
        self.assertIsNone(manager.reap())
        with self.assertRaises(GPULeaseBusyError):
            manager.acquire("replacement", "inference", 100, deadline=clock() + 5)

        manager.release(expired)
        replacement = manager.acquire(
            "replacement", "inference", 100, deadline=clock() + 5
        )
        self.assertGreater(replacement.epoch, expired.epoch)

    def test_context_manager_releases_on_success_and_exception(self) -> None:
        clock = _Clock()
        manager = self.manager(clock)
        with manager.hold("agent", "inference", 100, deadline=clock() + 10) as lease:
            self.assertEqual(manager.validate(lease), lease)
        self.assertIsNone(manager.active)
        self.assertEqual(manager.history[-1].reason, "released")

        with self.assertRaisesRegex(ValueError, "work failed"):
            with manager.hold("evolution", "evolution", 100, deadline=clock() + 10):
                raise ValueError("work failed")
        self.assertIsNone(manager.active)

    def test_context_cleanup_does_not_mask_supervisor_revoke_or_work_error(
        self,
    ) -> None:
        clock = _Clock()
        manager = self.manager(clock)
        with manager.hold("worker", "validation", 100, deadline=clock() + 10):
            event = manager.force_revoke("worker_crash")
            self.assertIsNotNone(event)
        self.assertIsNone(manager.active)

        with self.assertRaisesRegex(RuntimeError, "primary failure"):
            with manager.hold("worker", "validation", 100, deadline=clock() + 10):
                manager.force_revoke("worker_crash")
                raise RuntimeError("primary failure")

    def test_owner_health_probe_supports_crash_safe_revoke(self) -> None:
        clock = _Clock()
        alive = [True]
        manager = self.manager(clock)
        crashed = manager.acquire(
            "child-314",
            "validation",
            100,
            deadline=clock() + 60,
            owner_alive=lambda: alive[0],
        )
        self.assertIsNone(manager.reap())
        alive[0] = False

        event = manager.reap()
        self.assertIsNotNone(event)
        assert event is not None
        self.assertEqual(event.lease, crashed)
        self.assertEqual(event.reason, "owner_confirmed_dead")
        self.assertIsNone(manager.active)
        with self.assertRaises(StaleGPULeaseError):
            manager.release(crashed)

    def test_failed_health_probe_is_fail_closed(self) -> None:
        clock = _Clock()
        manager = self.manager(clock)

        def failed_probe() -> bool:
            raise OSError("process table unavailable")

        lease = manager.acquire(
            "child",
            "validation",
            100,
            deadline=clock() + 60,
            owner_alive=failed_probe,
        )
        event = manager.reap()
        self.assertIsNone(event)
        self.assertEqual(manager.active, lease)
        self.assertEqual(manager.history, ())
        with self.assertRaises(GPULeaseBusyError):
            manager.acquire("replacement", "inference", 100, deadline=clock() + 10)
        manager.release(lease)

    def test_budget_and_purpose_are_enforced_at_acquire_and_use(self) -> None:
        clock = _Clock()
        manager = self.manager(clock, max_vram_bytes=1_000)
        with self.assertRaises(GPULeaseBudgetError):
            manager.acquire("agent", "inference", 1_001, deadline=clock() + 10)
        with self.assertRaises(ValueError):
            manager.acquire("agent", "inference", 0, deadline=clock() + 10)
        with self.assertRaises(ValueError):
            manager.acquire("agent", "inference", 1, deadline=clock())

        lease = manager.acquire("agent", "inference", 800, deadline=clock() + 10)
        self.assertEqual(
            manager.validate(lease, purpose="inference", required_vram_bytes=800),
            lease,
        )
        with self.assertRaises(StaleGPULeaseError):
            manager.validate(lease, purpose="evolution")
        with self.assertRaises(GPULeaseBudgetError):
            manager.validate(lease, required_vram_bytes=801)

    def test_force_revoke_is_idempotent_when_no_owner_exists(self) -> None:
        clock = _Clock()
        manager = self.manager(clock)
        lease = manager.acquire("server", "shutdown", 1, deadline=clock() + 10)
        event = manager.force_revoke()
        self.assertIsNotNone(event)
        assert event is not None
        self.assertEqual(event.lease, lease)
        self.assertIsNone(manager.force_revoke())

    def test_concurrent_acquire_grants_exactly_one_owner(self) -> None:
        clock = _Clock()
        manager = self.manager(clock)
        start = Barrier(3)
        attempted = Barrier(3)
        release = Event()
        output_lock = Lock()
        outcomes: list[tuple[str, str]] = []

        def contender(owner: str) -> None:
            start.wait()
            lease = None
            try:
                lease = manager.acquire(owner, "inference", 100, deadline=clock() + 30)
                outcome = "granted"
            except GPULeaseBusyError:
                outcome = "busy"
            with output_lock:
                outcomes.append((owner, outcome))
            attempted.wait()
            if lease is not None:
                release.wait(2.0)
                manager.release(lease)

        threads = [
            Thread(target=contender, args=("agent-a",)),
            Thread(target=contender, args=("agent-b",)),
        ]
        for thread in threads:
            thread.start()
        start.wait()
        attempted.wait()
        self.assertEqual(
            sorted(outcome for _owner, outcome in outcomes), ["busy", "granted"]
        )
        release.set()
        for thread in threads:
            thread.join(2.0)
            self.assertFalse(thread.is_alive())
        self.assertIsNone(manager.active)


if __name__ == "__main__":
    unittest.main()
