"""Bounded, fail-closed ownership leases for one local GPU.

The rhythm controller separates inference and evolution *inside* one runtime,
but it cannot arbitrate CUDA ownership between the resident model process,
validation subprocesses, and the Self-Harness control plane.  This module is a
small control-plane primitive for that outer boundary.  It owns no CUDA state
and deliberately performs no background work; a supervisor calls :meth:`reap`
from its existing watchdog or worker-exit path.

Leases are immutable capabilities.  Every grant receives a monotonically
increasing epoch and an opaque identifier.  All release/revoke/validation
operations compare the complete capability, so a delayed cleanup from a dead
worker can never revoke a newer owner's lease.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
import math
import secrets
from threading import RLock
from time import monotonic


DEFAULT_MAX_VRAM_BYTES = int(16.7 * 1024**3)
MAX_IDENTITY_LENGTH = 64
MAX_REVOCATION_HISTORY = 64


class GPULeaseError(RuntimeError):
    """Base class for GPU ownership contract failures."""


class GPULeaseBusyError(GPULeaseError):
    """Raised when a live lease already owns the single GPU slot."""


class StaleGPULeaseError(GPULeaseError):
    """Raised when a capability no longer names the active lease."""


class ExpiredGPULeaseError(GPULeaseError):
    """Raised when an expired capability is still fencing a live owner."""


class GPULeaseBudgetError(GPULeaseError):
    """Raised when a request exceeds the admitted VRAM envelope."""


@dataclass(frozen=True, slots=True)
class GPULease:
    """Immutable capability for one bounded GPU workload."""

    lease_id: str
    owner: str
    epoch: int
    purpose: str
    vram_budget_bytes: int
    acquired_at: float
    deadline: float

    @property
    def ttl_seconds(self) -> float:
        return max(0.0, self.deadline - self.acquired_at)


@dataclass(frozen=True, slots=True)
class LeaseRevocation:
    """Auditable record emitted whenever an active capability is removed."""

    lease: GPULease
    reason: str
    revoked_at: float


OwnerHealthCheck = Callable[[], bool]
Clock = Callable[[], float]


def _bounded_label(value: object, field: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field} must be a string")
    normalized = value.strip()
    if not 1 <= len(normalized) <= MAX_IDENTITY_LENGTH:
        raise ValueError(
            f"{field} must contain between 1 and {MAX_IDENTITY_LENGTH} characters"
        )
    if any(ord(character) < 32 or ord(character) == 127 for character in normalized):
        raise ValueError(f"{field} contains a control character")
    return normalized


class GPULeaseManager:
    """Thread-safe, single-owner lease authority for one GPU.

    ``deadline`` uses the manager's monotonic clock and is mandatory.  A lease
    cannot be renewed in place: long-running owners must release and acquire a
    new epoch.  This keeps every stale capability unambiguously invalid.

    An optional ``owner_alive`` callback lets a supervisor revoke a lease even
    when the owner cannot execute ``finally`` (for example, after a child
    process crash).  The callback is invoked only by :meth:`reap`, never by a
    background thread and never while the manager lock is held.
    """

    def __init__(
        self,
        *,
        max_vram_bytes: int = DEFAULT_MAX_VRAM_BYTES,
        clock: Clock = monotonic,
        initial_epoch: int = 0,
    ) -> None:
        if (
            not isinstance(max_vram_bytes, int)
            or isinstance(max_vram_bytes, bool)
            or max_vram_bytes <= 0
        ):
            raise ValueError("max_vram_bytes must be a positive integer")
        if not callable(clock):
            raise TypeError("clock must be callable")
        if (
            not isinstance(initial_epoch, int)
            or isinstance(initial_epoch, bool)
            or initial_epoch < 0
        ):
            raise ValueError("initial_epoch must be a non-negative integer")
        self.max_vram_bytes = max_vram_bytes
        self._clock = clock
        self._epoch = initial_epoch
        self._active: GPULease | None = None
        self._owner_alive: OwnerHealthCheck | None = None
        self._history: deque[LeaseRevocation] = deque(maxlen=MAX_REVOCATION_HISTORY)
        self._lock = RLock()

    @property
    def latest_epoch(self) -> int:
        with self._lock:
            return self._epoch

    @property
    def history(self) -> tuple[LeaseRevocation, ...]:
        with self._lock:
            return tuple(self._history)

    @property
    def active(self) -> GPULease | None:
        """Return the current fence, including an expired-but-unreaped lease.

        A deadline is an authorization boundary, not proof that the CUDA owner
        exited.  Status reads therefore never remove a fence merely because its
        deadline elapsed.
        """

        self.reap(check_owner=False)
        with self._lock:
            return self._active

    def deadline_after(self, seconds: float) -> float:
        """Return a finite deadline in this authority's monotonic time domain."""

        if (
            not isinstance(seconds, (int, float))
            or isinstance(seconds, bool)
            or not math.isfinite(float(seconds))
            or float(seconds) <= 0.0
        ):
            raise ValueError("deadline duration must be finite and positive")
        now = float(self._clock())
        deadline = now + float(seconds)
        if not math.isfinite(now) or not math.isfinite(deadline):
            raise RuntimeError("GPU lease clock or deadline is non-finite")
        return deadline

    def acquire(
        self,
        owner: str,
        purpose: str,
        vram_budget_bytes: int,
        *,
        deadline: float,
        owner_alive: OwnerHealthCheck | None = None,
    ) -> GPULease:
        """Grant the sole GPU slot or fail without changing current ownership."""

        owner_value = _bounded_label(owner, "owner")
        purpose_value = _bounded_label(purpose, "purpose")
        if (
            not isinstance(vram_budget_bytes, int)
            or isinstance(vram_budget_bytes, bool)
            or vram_budget_bytes <= 0
        ):
            raise ValueError("vram_budget_bytes must be a positive integer")
        if vram_budget_bytes > self.max_vram_bytes:
            raise GPULeaseBudgetError(
                "GPU lease VRAM budget exceeded: "
                f"requested={vram_budget_bytes}, limit={self.max_vram_bytes}"
            )
        if not isinstance(deadline, (int, float)) or isinstance(deadline, bool):
            raise TypeError("deadline must be a finite monotonic timestamp")
        deadline_value = float(deadline)
        if not math.isfinite(deadline_value):
            raise ValueError("deadline must be finite")
        if owner_alive is not None and not callable(owner_alive):
            raise TypeError("owner_alive must be callable when provided")

        # Reap only a confirmed-dead owner before deciding whether the slot is
        # busy.  Expiry alone is never proof that a CUDA process released VRAM.
        self.reap()
        now = float(self._clock())
        if not math.isfinite(now):
            raise RuntimeError("GPU lease clock returned a non-finite value")
        if deadline_value <= now:
            raise ValueError("deadline must be in the future")
        with self._lock:
            if self._active is not None:
                current = self._active
                raise GPULeaseBusyError(
                    "GPU is already leased: "
                    f"owner={current.owner}, purpose={current.purpose}, "
                    f"epoch={current.epoch}"
                )
            self._epoch += 1
            lease = GPULease(
                lease_id=secrets.token_hex(16),
                owner=owner_value,
                epoch=self._epoch,
                purpose=purpose_value,
                vram_budget_bytes=vram_budget_bytes,
                acquired_at=now,
                deadline=deadline_value,
            )
            self._active = lease
            self._owner_alive = owner_alive
            return lease

    @contextmanager
    def hold(
        self,
        owner: str,
        purpose: str,
        vram_budget_bytes: int,
        *,
        deadline: float,
        owner_alive: OwnerHealthCheck | None = None,
    ) -> Iterator[GPULease]:
        """Acquire a lease and release it on every normal or exceptional exit.

        If a watchdog already revoked the capability, cleanup ignores only the
        expected stale-capability error.  An exception raised by the workload is
        therefore never hidden by delayed context-manager cleanup.
        """

        lease = self.acquire(
            owner,
            purpose,
            vram_budget_bytes,
            deadline=deadline,
            owner_alive=owner_alive,
        )
        try:
            yield lease
        finally:
            try:
                self.release(lease)
            except StaleGPULeaseError:
                pass

    def validate(
        self,
        lease: GPULease,
        *,
        purpose: str | None = None,
        required_vram_bytes: int | None = None,
    ) -> GPULease:
        """Reject stale, mismatched, expired, or under-budget capabilities."""

        self.reap()
        with self._lock:
            active = self._require_active_locked(lease)
            now = float(self._clock())
            if not math.isfinite(now):
                raise RuntimeError("GPU lease clock returned a non-finite value")
            if now >= active.deadline:
                raise ExpiredGPULeaseError(
                    "GPU lease expired while its owner remains fenced: "
                    f"owner={active.owner}, purpose={active.purpose}, "
                    f"epoch={active.epoch}"
                )
            if purpose is not None and active.purpose != _bounded_label(
                purpose, "purpose"
            ):
                raise StaleGPULeaseError("GPU lease purpose does not match")
            if required_vram_bytes is not None:
                if (
                    not isinstance(required_vram_bytes, int)
                    or isinstance(required_vram_bytes, bool)
                    or required_vram_bytes < 0
                ):
                    raise ValueError(
                        "required_vram_bytes must be a non-negative integer"
                    )
                if required_vram_bytes > active.vram_budget_bytes:
                    raise GPULeaseBudgetError(
                        "workload exceeds its GPU lease budget: "
                        f"required={required_vram_bytes}, "
                        f"lease={active.vram_budget_bytes}"
                    )
            return active

    def release(self, lease: GPULease) -> LeaseRevocation:
        """Release exactly ``lease``; delayed stale cleanup is rejected."""

        with self._lock:
            active = self._require_active_locked(lease)
            return self._revoke_locked(active, "released")

    def revoke(
        self, lease: GPULease, reason: str = "supervisor_revoke"
    ) -> LeaseRevocation:
        """Revoke exactly ``lease`` without risking a newer epoch."""

        reason_value = _bounded_label(reason, "reason")
        with self._lock:
            active = self._require_active_locked(lease)
            return self._revoke_locked(active, reason_value)

    def force_revoke(
        self, reason: str = "supervisor_shutdown"
    ) -> LeaseRevocation | None:
        """Trusted escape hatch after the supervisor has proved process death.

        This method cannot inspect a platform process handle itself.  Callers
        must use it only after stop/join/kill has established that no CUDA owner
        remains.  Normal worker cleanup should prefer exact ``release`` or
        ``revoke`` with its immutable capability.
        """

        reason_value = _bounded_label(reason, "reason")
        with self._lock:
            if self._active is None:
                return None
            return self._revoke_locked(self._active, reason_value)

    def reap(self, *, check_owner: bool = True) -> LeaseRevocation | None:
        """Revoke only an owner whose health probe confirms process death.

        Deadline expiry alone never removes the ownership fence: a stalled CUDA
        process may still retain all of its VRAM.  Likewise, a failed health
        probe is an unknown state and remains fenced. ``check_owner=False`` is
        useful for status reads that must not invoke a platform process probe.
        Supervisors should call the default form from worker-exit paths.
        """

        now = float(self._clock())
        if not math.isfinite(now):
            raise RuntimeError("GPU lease clock returned a non-finite value")
        with self._lock:
            lease = self._active
            if lease is None:
                return None
            health = self._owner_alive if check_owner else None

        if health is None:
            return None
        try:
            alive = bool(health())
        except BaseException:
            # Process-table failure is not evidence of process death.  Retain
            # the fence so no successor can be admitted on an unknown state.
            return None
        if alive:
            return None
        with self._lock:
            # Ownership may have changed while the callback ran.  Never revoke
            # a replacement lease using an old owner's health result.
            if self._active != lease:
                return None
            return self._revoke_locked(lease, "owner_confirmed_dead", now=now)

    def _require_active_locked(self, lease: GPULease) -> GPULease:
        if not isinstance(lease, GPULease):
            raise TypeError("lease must be a GPULease capability")
        if self._active != lease:
            current = self._active
            current_epoch = None if current is None else current.epoch
            raise StaleGPULeaseError(
                "GPU lease is stale: "
                f"capability_epoch={lease.epoch}, active_epoch={current_epoch}"
            )
        return self._active

    def _revoke_locked(
        self,
        lease: GPULease,
        reason: str,
        *,
        now: float | None = None,
    ) -> LeaseRevocation:
        if self._active != lease:
            raise StaleGPULeaseError("GPU lease changed before revoke")
        event = LeaseRevocation(
            lease,
            reason,
            float(self._clock()) if now is None else now,
        )
        self._active = None
        self._owner_alive = None
        self._history.append(event)
        return event


__all__ = [
    "DEFAULT_MAX_VRAM_BYTES",
    "ExpiredGPULeaseError",
    "GPULease",
    "GPULeaseBudgetError",
    "GPULeaseBusyError",
    "GPULeaseError",
    "GPULeaseManager",
    "LeaseRevocation",
    "StaleGPULeaseError",
]
