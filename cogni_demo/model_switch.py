"""Opt-in, fail-closed model runtime switching primitives.

The controller deliberately has no CUDA query or loader-specific dependency.
Production wiring supplies a manifest-bound runtime factory and an independent
memory-release probe; tests can therefore exercise every safety transition
without touching a physical device.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from hashlib import sha256
import math
import re
import secrets
from threading import Event, Lock, RLock
from time import monotonic
from typing import Protocol

from cogni_os.gpu_lease import DEFAULT_MAX_VRAM_BYTES, GPULease


RESIDENT_LEASE_OWNER = "cogni-resident-model"
RESIDENT_LEASE_PURPOSE = "resident-model"
MAX_SWITCH_TIMEOUT_SECONDS = 300.0
MAX_SWITCH_TRANSITIONS = 24

_SHA256 = re.compile(r"[0-9a-f]{64}")
_MODEL_ID = re.compile(r"[A-Za-z0-9_.:-]{1,128}")
_BINDING_ID = re.compile(r"[0-9a-f]{32}")
_ERROR_CODE = re.compile(r"[A-Z][A-Z0-9_]{0,63}")


class ModelSwitchError(RuntimeError):
    """Base class for stable, non-diagnostic control-plane failures."""

    code = "MODEL_SWITCH_FAILED"


class ModelSwitchDisabledError(ModelSwitchError):
    code = "MODEL_SWITCH_DISABLED"


class ModelSwitchBusyError(ModelSwitchError):
    code = "MODEL_SWITCH_BUSY"


class RuntimeBundleIncompleteError(ModelSwitchError):
    code = "RUNTIME_BUNDLE_INCOMPLETE"


class AtomicRuntimeCommitError(ModelSwitchError):
    code = "ATOMIC_RUNTIME_COMMIT_FAILED"


class _TransactionFailure(ModelSwitchError):
    def __init__(self, code: str) -> None:
        if _ERROR_CODE.fullmatch(code) is None:
            raise ValueError("transaction error code is invalid")
        self.code = code
        super().__init__(code)


@dataclass(frozen=True, slots=True)
class VerifiedModelSwitchDescriptor:
    """Content-addressed authority for one pre-verified local model."""

    model_id: str
    manifest_sha256: str
    config_sha256: str
    content_digest: str

    def __post_init__(self) -> None:
        if _MODEL_ID.fullmatch(self.model_id) is None:
            raise ValueError("model_id is invalid")
        for field_name in ("manifest_sha256", "config_sha256", "content_digest"):
            if _SHA256.fullmatch(getattr(self, field_name)) is None:
                raise ValueError(f"{field_name} must be a lowercase SHA-256 digest")

    @property
    def authority_digest(self) -> str:
        digest = sha256(b"CogniBoard/model-switch-descriptor/v1\0")
        for value in (
            self.model_id,
            self.manifest_sha256,
            self.config_sha256,
            self.content_digest,
        ):
            digest.update(value.encode("ascii"))
            digest.update(b"\0")
        return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class StableResidentLeaseAuthority:
    """Mode-independent lease identity for the sole resident model worker."""

    owner: str = RESIDENT_LEASE_OWNER
    purpose: str = RESIDENT_LEASE_PURPOSE
    vram_budget_bytes: int = DEFAULT_MAX_VRAM_BYTES

    def __post_init__(self) -> None:
        if self.owner != RESIDENT_LEASE_OWNER:
            raise ValueError("resident lease owner is not the admitted stable owner")
        if self.purpose != RESIDENT_LEASE_PURPOSE:
            raise ValueError("resident lease purpose must not depend on system mode")
        if (
            not isinstance(self.vram_budget_bytes, int)
            or isinstance(self.vram_budget_bytes, bool)
            or self.vram_budget_bytes <= 0
            or self.vram_budget_bytes > DEFAULT_MAX_VRAM_BYTES
        ):
            raise ValueError("resident lease budget exceeds the 16.7 GiB boundary")


@dataclass(frozen=True, slots=True)
class RuntimeBindingEvidence:
    component: str
    binding_id: str
    model_authority_digest: str

    def __post_init__(self) -> None:
        if self.component not in {
            "resident",
            "factbook",
            "validator",
            "voice",
            "harness",
        }:
            raise ValueError("runtime binding component is invalid")
        if _BINDING_ID.fullmatch(self.binding_id) is None:
            raise ValueError("runtime binding id is invalid")
        if _SHA256.fullmatch(self.model_authority_digest) is None:
            raise ValueError("runtime model authority digest is invalid")


@dataclass(frozen=True, slots=True)
class RuntimeHealthEvidence:
    ready: bool
    binding_id: str
    model_authority_digest: str
    lease_epoch: int

    def __post_init__(self) -> None:
        if not isinstance(self.ready, bool):
            raise TypeError("ready must be bool")
        if _BINDING_ID.fullmatch(self.binding_id) is None:
            raise ValueError("health binding id is invalid")
        if _SHA256.fullmatch(self.model_authority_digest) is None:
            raise ValueError("health model authority digest is invalid")
        if (
            not isinstance(self.lease_epoch, int)
            or isinstance(self.lease_epoch, bool)
            or self.lease_epoch <= 0
        ):
            raise ValueError("health lease epoch must be positive")


@dataclass(frozen=True, slots=True)
class LeaseReleaseEvidence:
    lease: GPULease
    reason: str
    worker_death_confirmed: bool

    def __post_init__(self) -> None:
        if not isinstance(self.lease, GPULease):
            raise TypeError("released lease evidence is invalid")
        if self.reason not in {"released", "owner_confirmed_dead"}:
            raise ValueError("lease release reason is not admitted")
        if self.worker_death_confirmed is not True:
            raise ValueError("lease release lacks worker-death proof")


@dataclass(frozen=True, slots=True)
class RuntimeUnloadEvidence:
    """Worker-authored acknowledgement for one exact resident unload."""

    binding_id: str
    model_authority_digest: str
    lease_epoch: int
    acknowledged: bool

    def __post_init__(self) -> None:
        if _BINDING_ID.fullmatch(self.binding_id) is None:
            raise ValueError("unload binding id is invalid")
        if _SHA256.fullmatch(self.model_authority_digest) is None:
            raise ValueError("unload model authority digest is invalid")
        if (
            not isinstance(self.lease_epoch, int)
            or isinstance(self.lease_epoch, bool)
            or self.lease_epoch <= 0
        ):
            raise ValueError("unload lease epoch must be positive")
        if not isinstance(self.acknowledged, bool):
            raise TypeError("unload acknowledgement must be bool")


@dataclass(frozen=True, slots=True)
class MemoryReleaseEvidence:
    """Injected-probe evidence that the retired model no longer owns memory."""

    binding_id: str
    model_authority_digest: str
    lease_epoch: int
    released: bool

    def __post_init__(self) -> None:
        if _BINDING_ID.fullmatch(self.binding_id) is None:
            raise ValueError("memory-release binding id is invalid")
        if _SHA256.fullmatch(self.model_authority_digest) is None:
            raise ValueError("memory-release model authority digest is invalid")
        if (
            not isinstance(self.lease_epoch, int)
            or isinstance(self.lease_epoch, bool)
            or self.lease_epoch <= 0
        ):
            raise ValueError("memory-release lease epoch must be positive")
        if not isinstance(self.released, bool):
            raise TypeError("memory-release result must be bool")


class ModelSwitchCancellation:
    """Thread-safe, monotonic cancellation token for one switch request."""

    def __init__(self) -> None:
        self._event = Event()
        self._lock = Lock()
        self._commit_claimed = False

    def cancel(self) -> bool:
        """Request cancellation unless the atomic commit boundary was claimed."""

        with self._lock:
            if self._commit_claimed:
                return False
            self._event.set()
            return True

    def claim_commit(self) -> bool:
        """Atomically arbitrate cancellation against final publication."""

        with self._lock:
            if self._event.is_set():
                return False
            self._commit_claimed = True
            return True

    @property
    def requested(self) -> bool:
        return self._event.is_set()


class ModelBindingPort(Protocol):
    def binding_evidence(self) -> RuntimeBindingEvidence: ...


class ResidentRuntimePort(ModelBindingPort, Protocol):
    @property
    def worker_alive(self) -> bool: ...

    @property
    def gpu_lease(self) -> GPULease | None: ...

    def start(self) -> None: ...

    def stop(self, timeout_seconds: float) -> RuntimeUnloadEvidence: ...

    def healthcheck(self, timeout_seconds: float) -> RuntimeHealthEvidence: ...


class LeaseAuthorityPort(Protocol):
    @property
    def active(self) -> GPULease | None: ...

    @property
    def latest_epoch(self) -> int: ...

    @property
    def max_vram_bytes(self) -> int: ...

    def release_evidence(self, lease: GPULease) -> LeaseReleaseEvidence | None: ...


class MemoryReleaseProbePort(Protocol):
    def verify_release(
        self,
        bundle: ActiveRuntimeBundle,
        unload: RuntimeUnloadEvidence,
        timeout_seconds: float,
    ) -> MemoryReleaseEvidence: ...


class RuntimeBundleFactoryPort(Protocol):
    def build(
        self,
        descriptor: VerifiedModelSwitchDescriptor,
        lease_authority: LeaseAuthorityPort,
        lease_profile: StableResidentLeaseAuthority,
    ) -> ActiveRuntimeBundle: ...


class ModelSwitchMaintenancePort(Protocol):
    def close_admission(self, transaction_id: str) -> None: ...

    def wait_for_drain(self, transaction_id: str, timeout_seconds: float) -> bool: ...

    def checkpoint(
        self,
        transaction_id: str,
        source: VerifiedModelSwitchDescriptor,
        candidate: VerifiedModelSwitchDescriptor,
    ) -> None: ...

    def open_admission(self, transaction_id: str) -> None: ...

    def enter_safe_mode(self, transaction_id: str, error_code: str) -> None: ...


@dataclass(frozen=True, slots=True)
class ActiveRuntimeBundle:
    """One immutable, atomically swappable set of model-bound authorities."""

    descriptor: VerifiedModelSwitchDescriptor
    binding_id: str
    lease_profile: StableResidentLeaseAuthority
    lease_authority: LeaseAuthorityPort
    runtime: ResidentRuntimePort
    factbook: ModelBindingPort
    validator: ModelBindingPort
    voice: ModelBindingPort
    harness: ModelBindingPort

    def validate_complete(self, *, require_stopped: bool = False) -> None:
        try:
            if not isinstance(self.descriptor, VerifiedModelSwitchDescriptor):
                raise TypeError("descriptor")
            if _BINDING_ID.fullmatch(self.binding_id) is None:
                raise ValueError("binding_id")
            if not isinstance(self.lease_profile, StableResidentLeaseAuthority):
                raise TypeError("lease_profile")
            maximum = self.lease_authority.max_vram_bytes
            latest_epoch = self.lease_authority.latest_epoch
            if maximum != self.lease_profile.vram_budget_bytes:
                raise ValueError("lease budget")
            if (
                not isinstance(latest_epoch, int)
                or isinstance(latest_epoch, bool)
                or latest_epoch < 0
            ):
                raise ValueError("lease epoch")
            expected_digest = self.descriptor.authority_digest
            components = (
                ("resident", self.runtime),
                ("factbook", self.factbook),
                ("validator", self.validator),
                ("voice", self.voice),
                ("harness", self.harness),
            )
            for expected_component, port in components:
                evidence = port.binding_evidence()
                if (
                    not isinstance(evidence, RuntimeBindingEvidence)
                    or evidence.component != expected_component
                    or evidence.binding_id != self.binding_id
                    or evidence.model_authority_digest != expected_digest
                ):
                    raise ValueError(f"{expected_component} binding")
            alive = self.runtime.worker_alive
            lease = self.runtime.gpu_lease
            if not isinstance(alive, bool):
                raise TypeError("worker_alive")
            if lease is not None and not isinstance(lease, GPULease):
                raise TypeError("gpu_lease")
            if require_stopped and (alive or lease is not None):
                raise ValueError("candidate runtime already started")
            for method_name in ("start", "stop", "healthcheck"):
                if not callable(getattr(self.runtime, method_name, None)):
                    raise TypeError(method_name)
        except (AttributeError, TypeError, ValueError) as error:
            raise RuntimeBundleIncompleteError(
                "runtime bundle failed its complete binding contract"
            ) from error


@dataclass(frozen=True, slots=True)
class RuntimeSlotSnapshot:
    generation: int
    bundle: ActiveRuntimeBundle


class AtomicRuntimeSlot:
    """Compare-and-swap publication point for a complete runtime bundle."""

    def __init__(self, bundle: ActiveRuntimeBundle) -> None:
        bundle.validate_complete()
        self._bundle = bundle
        self._generation = 0
        self._lock = RLock()

    def snapshot(self) -> RuntimeSlotSnapshot:
        with self._lock:
            return RuntimeSlotSnapshot(self._generation, self._bundle)

    def compare_and_swap(
        self,
        expected: RuntimeSlotSnapshot,
        replacement: ActiveRuntimeBundle,
    ) -> RuntimeSlotSnapshot:
        replacement.validate_complete()
        with self._lock:
            if (
                self._generation != expected.generation
                or self._bundle is not expected.bundle
            ):
                raise AtomicRuntimeCommitError(
                    "active runtime changed before atomic commit"
                )
            self._bundle = replacement
            self._generation += 1
            return RuntimeSlotSnapshot(self._generation, replacement)


class ModelSwitchState(str, Enum):
    REQUESTED = "requested"
    DRAINING = "draining"
    CHECKPOINTING = "checkpointing"
    UNLOADING_OLD = "unloading_old"
    LOADING_CANDIDATE = "loading_candidate"
    HEALTHCHECKING = "healthchecking"
    COMMITTING = "committing"
    ROLLING_BACK = "rolling_back"
    RESTORING_OLD = "restoring_old"
    SUCCEEDED = "succeeded"
    ROLLED_BACK = "rolled_back"
    SAFE_MODE = "safe_mode"


_ALLOWED_TRANSITIONS: dict[ModelSwitchState, set[ModelSwitchState]] = {
    ModelSwitchState.REQUESTED: {
        ModelSwitchState.DRAINING,
        ModelSwitchState.ROLLING_BACK,
        ModelSwitchState.SAFE_MODE,
    },
    ModelSwitchState.DRAINING: {
        ModelSwitchState.CHECKPOINTING,
        ModelSwitchState.ROLLING_BACK,
        ModelSwitchState.SAFE_MODE,
    },
    ModelSwitchState.CHECKPOINTING: {
        ModelSwitchState.UNLOADING_OLD,
        ModelSwitchState.ROLLING_BACK,
        ModelSwitchState.SAFE_MODE,
    },
    ModelSwitchState.UNLOADING_OLD: {
        ModelSwitchState.LOADING_CANDIDATE,
        ModelSwitchState.ROLLING_BACK,
        ModelSwitchState.SAFE_MODE,
    },
    ModelSwitchState.LOADING_CANDIDATE: {
        ModelSwitchState.HEALTHCHECKING,
        ModelSwitchState.ROLLING_BACK,
        ModelSwitchState.SAFE_MODE,
    },
    ModelSwitchState.HEALTHCHECKING: {
        ModelSwitchState.COMMITTING,
        ModelSwitchState.ROLLING_BACK,
        ModelSwitchState.SAFE_MODE,
    },
    ModelSwitchState.COMMITTING: {
        ModelSwitchState.SUCCEEDED,
        ModelSwitchState.ROLLING_BACK,
        ModelSwitchState.SAFE_MODE,
    },
    ModelSwitchState.ROLLING_BACK: {
        ModelSwitchState.RESTORING_OLD,
        ModelSwitchState.ROLLED_BACK,
        ModelSwitchState.SAFE_MODE,
    },
    ModelSwitchState.RESTORING_OLD: {
        ModelSwitchState.ROLLED_BACK,
        ModelSwitchState.SAFE_MODE,
    },
    ModelSwitchState.SUCCEEDED: set(),
    ModelSwitchState.ROLLED_BACK: set(),
    ModelSwitchState.SAFE_MODE: set(),
}


@dataclass(frozen=True, slots=True)
class ModelSwitchTransition:
    sequence: int
    source: ModelSwitchState | None
    target: ModelSwitchState
    timestamp: float


@dataclass(frozen=True, slots=True)
class ModelSwitchSnapshot:
    transaction_id: str
    source_model_id: str
    candidate_model_id: str
    state: ModelSwitchState
    error_code: str | None
    rollback_restored: bool
    transitions: tuple[ModelSwitchTransition, ...]


@dataclass(slots=True)
class _MutableTransaction:
    transaction_id: str
    source_model_id: str
    candidate_model_id: str
    state: ModelSwitchState
    transitions: list[ModelSwitchTransition]
    error_code: str | None = None
    rollback_restored: bool = False

    def snapshot(self) -> ModelSwitchSnapshot:
        return ModelSwitchSnapshot(
            transaction_id=self.transaction_id,
            source_model_id=self.source_model_id,
            candidate_model_id=self.candidate_model_id,
            state=self.state,
            error_code=self.error_code,
            rollback_restored=self.rollback_restored,
            transitions=tuple(self.transitions),
        )


class ModelSwitchController:
    """Synchronous, single-flight switch transaction; disabled by default."""

    def __init__(
        self,
        slot: AtomicRuntimeSlot,
        factory: RuntimeBundleFactoryPort,
        maintenance: ModelSwitchMaintenancePort,
        *,
        enabled: bool = False,
        memory_release_probe: MemoryReleaseProbePort | None = None,
        clock=monotonic,
    ) -> None:
        if not isinstance(slot, AtomicRuntimeSlot):
            raise TypeError("slot must be AtomicRuntimeSlot")
        if not isinstance(enabled, bool):
            raise TypeError("enabled must be bool")
        if not callable(clock):
            raise TypeError("clock must be callable")
        if memory_release_probe is not None and not callable(
            getattr(memory_release_probe, "verify_release", None)
        ):
            raise TypeError("memory_release_probe must implement verify_release")
        if enabled and memory_release_probe is None:
            raise ValueError(
                "enabled model switching requires an injected memory-release probe"
            )
        self.slot = slot
        self.factory = factory
        self.maintenance = maintenance
        self.enabled = enabled
        self.memory_release_probe = memory_release_probe
        self._clock = clock
        self._run_lock = Lock()
        self._state_lock = RLock()
        self._transaction: _MutableTransaction | None = None

    def snapshot(self) -> ModelSwitchSnapshot | None:
        with self._state_lock:
            return None if self._transaction is None else self._transaction.snapshot()

    def switch(
        self,
        candidate: VerifiedModelSwitchDescriptor,
        *,
        drain_timeout_seconds: float = 30.0,
        stop_timeout_seconds: float = 10.0,
        memory_timeout_seconds: float = 10.0,
        health_timeout_seconds: float = 30.0,
        cancellation: ModelSwitchCancellation | None = None,
    ) -> ModelSwitchSnapshot:
        if not self.enabled:
            raise ModelSwitchDisabledError(
                "model switching is not explicitly enabled for this runtime"
            )
        if not self._run_lock.acquire(blocking=False):
            raise ModelSwitchBusyError("another model switch is active")
        try:
            self._require_timeout(drain_timeout_seconds, "drain timeout")
            self._require_timeout(stop_timeout_seconds, "stop timeout")
            self._require_timeout(memory_timeout_seconds, "memory timeout")
            self._require_timeout(health_timeout_seconds, "health timeout")
            if not isinstance(candidate, VerifiedModelSwitchDescriptor):
                raise TypeError("candidate must be VerifiedModelSwitchDescriptor")
            if cancellation is not None and not isinstance(
                cancellation, ModelSwitchCancellation
            ):
                raise TypeError("cancellation must be ModelSwitchCancellation or None")

            source_slot = self.slot.snapshot()
            source = source_slot.bundle
            source.validate_complete()
            self._prove_source_consistent(source)
            if candidate.model_id == source.descriptor.model_id:
                raise ValueError("candidate must differ from the active model")

            # Fact-book, validator, voice, harness and resident bindings are all
            # checked before admission is closed or either worker is touched.
            try:
                prepared = self.factory.build(
                    candidate, source.lease_authority, source.lease_profile
                )
            except Exception:
                # Factory diagnostics may contain host paths or loader secrets.
                # The public preflight contract exposes only a stable code.
                raise RuntimeBundleIncompleteError(
                    "candidate runtime factory failed"
                ) from None
            self._validate_prepared_bundle(prepared, candidate, source)

            transaction = self._begin(source.descriptor, candidate)
            source_retired = False
            candidate_started = False
            admission_closed = False
            committed_slot: RuntimeSlotSnapshot | None = None
            source_epoch_floor = source.lease_authority.latest_epoch
            try:
                self._check_cancelled(cancellation)
                self._call(
                    "ADMISSION_CLOSE_FAILED",
                    self.maintenance.close_admission,
                    transaction.transaction_id,
                )
                admission_closed = True
                self._transition(ModelSwitchState.DRAINING)
                drain_deadline = self._deadline_after(float(drain_timeout_seconds))
                drained = self._call(
                    "DRAIN_FAILED",
                    self.maintenance.wait_for_drain,
                    transaction.transaction_id,
                    self._remaining(drain_deadline, "DRAIN_TIMEOUT"),
                )
                if drained is not True or self._expired(drain_deadline):
                    raise _TransactionFailure("DRAIN_TIMEOUT")
                self._check_cancelled(cancellation)

                self._transition(ModelSwitchState.CHECKPOINTING)
                self._call(
                    "CHECKPOINT_FAILED",
                    self.maintenance.checkpoint,
                    transaction.transaction_id,
                    source.descriptor,
                    candidate,
                )
                self._check_cancelled(cancellation)

                self._transition(ModelSwitchState.UNLOADING_OLD)
                observed_slot = self.slot.snapshot()
                if (
                    observed_slot.generation != source_slot.generation
                    or observed_slot.bundle is not source
                ):
                    raise _TransactionFailure("ATOMIC_RUNTIME_CHANGED")
                self._prove_source_consistent(source)
                old_lease = source.runtime.gpu_lease
                if not isinstance(old_lease, GPULease):
                    raise _TransactionFailure("SOURCE_WORKER_NOT_RUNNING")
                unload_error: Exception | None = None
                unload: RuntimeUnloadEvidence | None = None
                stop_deadline = self._deadline_after(float(stop_timeout_seconds))
                try:
                    unload = source.runtime.stop(
                        self._remaining(stop_deadline, "OLD_UNLOAD_TIMEOUT")
                    )
                except Exception as error:
                    unload_error = error
                # A stop exception is not liveness evidence.  Prove retirement
                # independently so rollback never reopens admission on a dead
                # bundle and never starts a successor beside a survivor.
                self._prove_retired(source, old_lease)
                source_retired = True
                if unload_error is not None:
                    raise _TransactionFailure("OLD_UNLOAD_FAILED") from unload_error
                if self._expired(stop_deadline):
                    raise _TransactionFailure("OLD_UNLOAD_TIMEOUT")
                self._prove_unload_ack(source, old_lease, unload)
                self._prove_memory_released(
                    source,
                    unload,
                    float(memory_timeout_seconds),
                    "SOURCE_MEMORY_RELEASE_UNPROVEN",
                )
                self._check_cancelled(cancellation)

                self._transition(ModelSwitchState.LOADING_CANDIDATE)
                self._require_unleased(source.lease_authority)
                self._call("CANDIDATE_START_FAILED", prepared.runtime.start)
                candidate_started = True
                candidate_lease = self._prove_running(prepared, source_epoch_floor)
                self._check_cancelled(cancellation)

                self._transition(ModelSwitchState.HEALTHCHECKING)
                health_deadline = self._deadline_after(float(health_timeout_seconds))
                health = self._call(
                    "CANDIDATE_HEALTHCHECK_FAILED",
                    prepared.runtime.healthcheck,
                    self._remaining(health_deadline, "CANDIDATE_HEALTHCHECK_TIMEOUT"),
                )
                if self._expired(health_deadline):
                    raise _TransactionFailure("CANDIDATE_HEALTHCHECK_TIMEOUT")
                self._prove_healthy(prepared, candidate_lease, health)
                self._check_cancelled(cancellation)

                self._transition(ModelSwitchState.COMMITTING)
                prepared.validate_complete()
                self._claim_commit(cancellation)
                committed_slot = self.slot.compare_and_swap(source_slot, prepared)
                # The cancellation token's commit claim and this CAS form one
                # publication boundary: later cancellation is rejected, while
                # a pre-existing request cannot reach the CAS.
                self._call(
                    "ADMISSION_OPEN_FAILED",
                    self.maintenance.open_admission,
                    transaction.transaction_id,
                )
                self._transition(ModelSwitchState.SUCCEEDED)
                return self._snapshot_required()
            except Exception as error:
                failure = (
                    error
                    if isinstance(error, _TransactionFailure)
                    else _TransactionFailure(
                        error.code
                        if isinstance(error, ModelSwitchError)
                        and _ERROR_CODE.fullmatch(error.code)
                        else "MODEL_SWITCH_FAILED"
                    )
                )
                return self._rollback(
                    failure,
                    source_slot=source_slot,
                    committed_slot=committed_slot,
                    source_retired=source_retired,
                    admission_closed=admission_closed,
                    candidate=prepared,
                    candidate_started=candidate_started,
                    stop_timeout_seconds=float(stop_timeout_seconds),
                    memory_timeout_seconds=float(memory_timeout_seconds),
                    health_timeout_seconds=float(health_timeout_seconds),
                )
        finally:
            self._run_lock.release()

    @staticmethod
    def _require_timeout(value: float, field: str) -> None:
        if (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(float(value))
            or not 0.0 < float(value) <= MAX_SWITCH_TIMEOUT_SECONDS
        ):
            raise ValueError(f"{field} must be finite and bounded")

    def _deadline_after(self, timeout_seconds: float) -> float:
        now = float(self._clock())
        if not math.isfinite(now):
            raise _TransactionFailure("MODEL_SWITCH_CLOCK_INVALID")
        deadline = now + timeout_seconds
        if not math.isfinite(deadline):
            raise _TransactionFailure("MODEL_SWITCH_CLOCK_INVALID")
        return deadline

    def _remaining(self, deadline: float, code: str) -> float:
        now = float(self._clock())
        if not math.isfinite(now):
            raise _TransactionFailure("MODEL_SWITCH_CLOCK_INVALID")
        remaining = deadline - now
        if remaining <= 0.0:
            raise _TransactionFailure(code)
        return remaining

    def _expired(self, deadline: float) -> bool:
        now = float(self._clock())
        if not math.isfinite(now):
            raise _TransactionFailure("MODEL_SWITCH_CLOCK_INVALID")
        return now >= deadline

    @staticmethod
    def _check_cancelled(cancellation: ModelSwitchCancellation | None) -> None:
        if cancellation is not None and cancellation.requested:
            raise _TransactionFailure("MODEL_SWITCH_CANCELLED")

    @staticmethod
    def _claim_commit(cancellation: ModelSwitchCancellation | None) -> None:
        if cancellation is not None and not cancellation.claim_commit():
            raise _TransactionFailure("MODEL_SWITCH_CANCELLED")

    def _begin(
        self,
        source: VerifiedModelSwitchDescriptor,
        candidate: VerifiedModelSwitchDescriptor,
    ) -> _MutableTransaction:
        now = float(self._clock())
        if not math.isfinite(now):
            raise RuntimeError("model switch clock is non-finite")
        transaction = _MutableTransaction(
            transaction_id=secrets.token_hex(12),
            source_model_id=source.model_id,
            candidate_model_id=candidate.model_id,
            state=ModelSwitchState.REQUESTED,
            transitions=[
                ModelSwitchTransition(0, None, ModelSwitchState.REQUESTED, now)
            ],
        )
        with self._state_lock:
            self._transaction = transaction
        return transaction

    def _transition(self, target: ModelSwitchState) -> None:
        with self._state_lock:
            transaction = self._transaction
            if transaction is None:
                raise RuntimeError("model switch transaction is unavailable")
            if target not in _ALLOWED_TRANSITIONS[transaction.state]:
                raise RuntimeError(
                    f"illegal model switch transition: {transaction.state.value} "
                    f"-> {target.value}"
                )
            if len(transaction.transitions) >= MAX_SWITCH_TRANSITIONS:
                raise RuntimeError("model switch transition bound exceeded")
            now = float(self._clock())
            if not math.isfinite(now):
                raise RuntimeError("model switch clock is non-finite")
            source = transaction.state
            transaction.state = target
            transaction.transitions.append(
                ModelSwitchTransition(len(transaction.transitions), source, target, now)
            )

    @staticmethod
    def _call(code: str, target, *args):
        try:
            return target(*args)
        except Exception as error:
            raise _TransactionFailure(code) from error

    @staticmethod
    def _validate_prepared_bundle(
        prepared: ActiveRuntimeBundle,
        candidate: VerifiedModelSwitchDescriptor,
        source: ActiveRuntimeBundle,
    ) -> None:
        if not isinstance(prepared, ActiveRuntimeBundle):
            raise RuntimeBundleIncompleteError("factory returned no runtime bundle")
        prepared.validate_complete(require_stopped=True)
        if (
            prepared.descriptor != candidate
            or prepared.binding_id == source.binding_id
            or prepared.lease_profile != source.lease_profile
            or prepared.lease_authority is not source.lease_authority
        ):
            raise RuntimeBundleIncompleteError(
                "candidate bundle escaped the active lease authority"
            )

    @classmethod
    def _prove_source_consistent(cls, bundle: ActiveRuntimeBundle) -> None:
        alive = bundle.runtime.worker_alive
        lease = bundle.runtime.gpu_lease
        active = bundle.lease_authority.active
        if alive is not True:
            raise RuntimeBundleIncompleteError(
                "published resident worker is unavailable"
            )
        if lease is None or active != lease:
            raise RuntimeBundleIncompleteError(
                "active worker lacks its exact resident lease"
            )
        cls._prove_lease_profile(lease, bundle.lease_profile)

    @staticmethod
    def _prove_lease_profile(
        lease: GPULease, profile: StableResidentLeaseAuthority
    ) -> None:
        if (
            lease.owner != profile.owner
            or lease.purpose != profile.purpose
            or lease.vram_budget_bytes != profile.vram_budget_bytes
            or lease.epoch <= 0
        ):
            raise _TransactionFailure("RESIDENT_LEASE_AUTHORITY_MISMATCH")

    @classmethod
    def _prove_retired(
        cls, bundle: ActiveRuntimeBundle, prior_lease: GPULease | None
    ) -> None:
        if (
            bundle.runtime.worker_alive
            or bundle.runtime.gpu_lease is not None
            or bundle.lease_authority.active is not None
        ):
            raise _TransactionFailure("WORKER_DEATH_UNPROVEN")
        if prior_lease is None:
            return
        evidence = bundle.lease_authority.release_evidence(prior_lease)
        if (
            not isinstance(evidence, LeaseReleaseEvidence)
            or evidence.lease != prior_lease
            or evidence.worker_death_confirmed is not True
        ):
            raise _TransactionFailure("LEASE_RELEASE_UNPROVEN")

    @staticmethod
    def _prove_unload_ack(
        bundle: ActiveRuntimeBundle,
        prior_lease: GPULease,
        unload: RuntimeUnloadEvidence | None,
    ) -> None:
        if (
            not isinstance(unload, RuntimeUnloadEvidence)
            or unload.acknowledged is not True
            or unload.binding_id != bundle.binding_id
            or unload.model_authority_digest != bundle.descriptor.authority_digest
            or unload.lease_epoch != prior_lease.epoch
        ):
            raise _TransactionFailure("UNLOAD_ACK_UNPROVEN")

    def _prove_memory_released(
        self,
        bundle: ActiveRuntimeBundle,
        unload: RuntimeUnloadEvidence | None,
        timeout_seconds: float,
        failure_code: str,
    ) -> None:
        if not isinstance(unload, RuntimeUnloadEvidence):
            raise _TransactionFailure("UNLOAD_ACK_UNPROVEN")
        probe = self.memory_release_probe
        if probe is None:
            raise _TransactionFailure("MEMORY_RELEASE_PROBE_MISSING")
        deadline = self._deadline_after(timeout_seconds)
        evidence = self._call(
            "MEMORY_RELEASE_PROBE_FAILED",
            probe.verify_release,
            bundle,
            unload,
            self._remaining(deadline, "MEMORY_RELEASE_PROBE_TIMEOUT"),
        )
        if self._expired(deadline):
            raise _TransactionFailure("MEMORY_RELEASE_PROBE_TIMEOUT")
        if (
            not isinstance(evidence, MemoryReleaseEvidence)
            or evidence.released is not True
            or evidence.binding_id != bundle.binding_id
            or evidence.model_authority_digest != bundle.descriptor.authority_digest
            or evidence.lease_epoch != unload.lease_epoch
        ):
            raise _TransactionFailure(failure_code)

    @staticmethod
    def _require_unleased(authority: LeaseAuthorityPort) -> None:
        if authority.active is not None:
            raise _TransactionFailure("GPU_LEASE_NOT_RELEASED")

    @classmethod
    def _prove_running(cls, bundle: ActiveRuntimeBundle, epoch_floor: int) -> GPULease:
        lease = bundle.runtime.gpu_lease
        if (
            bundle.runtime.worker_alive is not True
            or not isinstance(lease, GPULease)
            or bundle.lease_authority.active != lease
            or lease.epoch <= epoch_floor
        ):
            raise _TransactionFailure("WORKER_START_UNPROVEN")
        cls._prove_lease_profile(lease, bundle.lease_profile)
        return lease

    @staticmethod
    def _prove_healthy(
        bundle: ActiveRuntimeBundle,
        lease: GPULease,
        health: RuntimeHealthEvidence,
    ) -> None:
        if (
            not isinstance(health, RuntimeHealthEvidence)
            or health.ready is not True
            or health.binding_id != bundle.binding_id
            or health.model_authority_digest != bundle.descriptor.authority_digest
            or health.lease_epoch != lease.epoch
            or bundle.runtime.worker_alive is not True
            or bundle.runtime.gpu_lease != lease
            or bundle.lease_authority.active != lease
        ):
            raise _TransactionFailure("CANDIDATE_HEALTH_UNPROVEN")
        bundle.validate_complete()

    def _rollback(
        self,
        failure: _TransactionFailure,
        *,
        source_slot: RuntimeSlotSnapshot,
        committed_slot: RuntimeSlotSnapshot | None,
        source_retired: bool,
        admission_closed: bool,
        candidate: ActiveRuntimeBundle,
        candidate_started: bool,
        stop_timeout_seconds: float,
        memory_timeout_seconds: float,
        health_timeout_seconds: float,
    ) -> ModelSwitchSnapshot:
        with self._state_lock:
            assert self._transaction is not None
            self._transaction.error_code = failure.code
        self._transition(ModelSwitchState.ROLLING_BACK)
        if failure.code in {
            "WORKER_DEATH_UNPROVEN",
            "LEASE_RELEASE_UNPROVEN",
            "GPU_LEASE_NOT_RELEASED",
            "UNLOAD_ACK_UNPROVEN",
            "SOURCE_MEMORY_RELEASE_UNPROVEN",
            "MEMORY_RELEASE_PROBE_FAILED",
            "MEMORY_RELEASE_PROBE_TIMEOUT",
            "MEMORY_RELEASE_PROBE_MISSING",
            "ADMISSION_CLOSE_FAILED",
            "SOURCE_WORKER_NOT_RUNNING",
        }:
            return self._safe_mode(failure.code)
        restored: ActiveRuntimeBundle | None = None
        restored_committed = False
        try:
            candidate_lease = candidate.runtime.gpu_lease
            if candidate_started or candidate.runtime.worker_alive or candidate_lease:
                if not isinstance(candidate_lease, GPULease):
                    raise _TransactionFailure("CANDIDATE_LEASE_UNPROVEN")
                stop_deadline = self._deadline_after(stop_timeout_seconds)
                candidate_unload = self._call(
                    "CANDIDATE_RETIRE_FAILED",
                    candidate.runtime.stop,
                    self._remaining(stop_deadline, "CANDIDATE_RETIRE_TIMEOUT"),
                )
                if self._expired(stop_deadline):
                    raise _TransactionFailure("CANDIDATE_RETIRE_TIMEOUT")
                self._prove_retired(candidate, candidate_lease)
                self._prove_unload_ack(candidate, candidate_lease, candidate_unload)
                self._prove_memory_released(
                    candidate,
                    candidate_unload,
                    memory_timeout_seconds,
                    "CANDIDATE_MEMORY_RELEASE_UNPROVEN",
                )

            if failure.code in {
                "ATOMIC_RUNTIME_CHANGED",
                "ATOMIC_RUNTIME_COMMIT_FAILED",
            }:
                return self._safe_mode(failure.code)

            if not source_retired:
                self._prove_source_consistent(source_slot.bundle)
                if admission_closed:
                    self._call(
                        "ADMISSION_RESTORE_FAILED",
                        self.maintenance.open_admission,
                        self._transaction_id(),
                    )
                self._transition(ModelSwitchState.ROLLED_BACK)
                return self._snapshot_required()

            self._transition(ModelSwitchState.RESTORING_OLD)
            restored = self.factory.build(
                source_slot.bundle.descriptor,
                source_slot.bundle.lease_authority,
                source_slot.bundle.lease_profile,
            )
            self._validate_prepared_bundle(
                restored, source_slot.bundle.descriptor, source_slot.bundle
            )
            epoch_floor = restored.lease_authority.latest_epoch
            self._call("ROLLBACK_START_FAILED", restored.runtime.start)
            restored_lease = self._prove_running(restored, epoch_floor)
            restored_health = self._call(
                "ROLLBACK_HEALTHCHECK_FAILED",
                restored.runtime.healthcheck,
                health_timeout_seconds,
            )
            self._prove_healthy(restored, restored_lease, restored_health)
            expected = committed_slot or source_slot
            self.slot.compare_and_swap(expected, restored)
            restored_committed = True
            self._call(
                "ADMISSION_RESTORE_FAILED",
                self.maintenance.open_admission,
                self._transaction_id(),
            )
            with self._state_lock:
                assert self._transaction is not None
                self._transaction.rollback_restored = True
            self._transition(ModelSwitchState.ROLLED_BACK)
            return self._snapshot_required()
        except Exception:
            if restored is not None and not restored_committed:
                restored_lease = restored.runtime.gpu_lease
                if restored.runtime.worker_alive or restored_lease is not None:
                    try:
                        restored.runtime.stop(stop_timeout_seconds)
                        self._prove_retired(restored, restored_lease)
                    except Exception:
                        pass
            return self._safe_mode(failure.code)

    def _safe_mode(self, error_code: str) -> ModelSwitchSnapshot:
        try:
            self.maintenance.enter_safe_mode(self._transaction_id(), error_code)
        except Exception:
            pass
        with self._state_lock:
            assert self._transaction is not None
            state = self._transaction.state
        if ModelSwitchState.SAFE_MODE in _ALLOWED_TRANSITIONS[state]:
            self._transition(ModelSwitchState.SAFE_MODE)
        return self._snapshot_required()

    def _transaction_id(self) -> str:
        with self._state_lock:
            if self._transaction is None:
                raise RuntimeError("model switch transaction is unavailable")
            return self._transaction.transaction_id

    def _snapshot_required(self) -> ModelSwitchSnapshot:
        snapshot = self.snapshot()
        if snapshot is None:
            raise RuntimeError("model switch transaction is unavailable")
        return snapshot


__all__ = [
    "ActiveRuntimeBundle",
    "AtomicRuntimeCommitError",
    "AtomicRuntimeSlot",
    "LeaseReleaseEvidence",
    "MemoryReleaseEvidence",
    "ModelSwitchCancellation",
    "ModelSwitchBusyError",
    "ModelSwitchController",
    "ModelSwitchDisabledError",
    "ModelSwitchError",
    "ModelSwitchSnapshot",
    "ModelSwitchState",
    "ModelSwitchTransition",
    "RuntimeBindingEvidence",
    "RuntimeBundleIncompleteError",
    "RuntimeHealthEvidence",
    "RuntimeUnloadEvidence",
    "StableResidentLeaseAuthority",
    "VerifiedModelSwitchDescriptor",
]
