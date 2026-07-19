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
from threading import Condition, Event, Lock, RLock
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


class ModelSwitchProductionUnavailableError(ModelSwitchError):
    code = "MODEL_SWITCH_PRODUCTION_UNAVAILABLE"


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
    worker_generation: int

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
        if (
            not isinstance(self.worker_generation, int)
            or isinstance(self.worker_generation, bool)
            or self.worker_generation <= 0
        ):
            raise ValueError("health worker generation must be positive")


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
    worker_generation: int
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
        if (
            not isinstance(self.worker_generation, int)
            or isinstance(self.worker_generation, bool)
            or self.worker_generation <= 0
        ):
            raise ValueError("unload worker generation must be positive")
        if not isinstance(self.acknowledged, bool):
            raise TypeError("unload acknowledgement must be bool")


@dataclass(frozen=True, slots=True)
class MemoryReleaseEvidence:
    """Injected-probe evidence that the retired model no longer owns memory."""

    binding_id: str
    model_authority_digest: str
    lease_epoch: int
    worker_generation: int
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
        if (
            not isinstance(self.worker_generation, int)
            or isinstance(self.worker_generation, bool)
            or self.worker_generation <= 0
        ):
            raise ValueError("memory-release worker generation must be positive")
        if not isinstance(self.released, bool):
            raise TypeError("memory-release result must be bool")


@dataclass(frozen=True, slots=True)
class RuntimePublicationFence:
    """Exact worker/lease/slot identity admitted at one publication boundary."""

    expected_slot_generation: int
    binding_id: str
    model_authority_digest: str
    worker_generation: int
    lease_id: str
    lease_epoch: int

    def __post_init__(self) -> None:
        for field_name in (
            "expected_slot_generation",
            "worker_generation",
            "lease_epoch",
        ):
            value = getattr(self, field_name)
            minimum = 0 if field_name == "expected_slot_generation" else 1
            if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
                raise ValueError(f"{field_name} is invalid")
        if _BINDING_ID.fullmatch(self.binding_id) is None:
            raise ValueError("publication binding id is invalid")
        if _SHA256.fullmatch(self.model_authority_digest) is None:
            raise ValueError("publication model authority digest is invalid")
        if not isinstance(self.lease_id, str) or not self.lease_id:
            raise ValueError("publication lease id is invalid")


@dataclass(frozen=True, slots=True)
class AdmissionFenceToken:
    """Capability presented when reopening request admission."""

    transaction_id: str
    slot_generation: int
    publication: RuntimePublicationFence

    def __post_init__(self) -> None:
        if not isinstance(self.transaction_id, str) or not self.transaction_id:
            raise ValueError("admission transaction id is invalid")
        if (
            not isinstance(self.slot_generation, int)
            or isinstance(self.slot_generation, bool)
            or self.slot_generation < 0
        ):
            raise ValueError("admission slot generation is invalid")
        if not isinstance(self.publication, RuntimePublicationFence):
            raise TypeError("admission publication fence is invalid")


@dataclass(frozen=True, slots=True)
class AdmissionOpenEvidence:
    token: AdmissionFenceToken
    gate_generation: int
    entry_validation_required: bool

    def __post_init__(self) -> None:
        if not isinstance(self.token, AdmissionFenceToken):
            raise TypeError("admission token acknowledgement is invalid")
        if (
            not isinstance(self.gate_generation, int)
            or isinstance(self.gate_generation, bool)
            or self.gate_generation <= 0
        ):
            raise ValueError("admission gate generation is invalid")
        if self.entry_validation_required is not True:
            raise ValueError("admission must require per-entry validation")


@dataclass(frozen=True, slots=True)
class AdmissionGateReadback:
    """Independent state readback from the atomic request gate."""

    gate_generation: int
    mode: str
    transaction_id: str | None
    token: AdmissionFenceToken | None
    in_flight_by_generation: tuple[tuple[int, int], ...]

    def __post_init__(self) -> None:
        if (
            not isinstance(self.gate_generation, int)
            or isinstance(self.gate_generation, bool)
            or self.gate_generation < 0
        ):
            raise ValueError("admission gate generation is invalid")
        if self.mode not in {"open", "closed", "safe_mode"}:
            raise ValueError("admission gate mode is invalid")
        if self.transaction_id is not None and (
            not isinstance(self.transaction_id, str) or not self.transaction_id
        ):
            raise ValueError("admission gate transaction id is invalid")
        if self.token is not None and not isinstance(self.token, AdmissionFenceToken):
            raise TypeError("admission gate token is invalid")
        if self.mode == "open" and self.token is None:
            raise ValueError("open admission gate lacks an entry token")
        if self.mode != "open" and self.token is not None:
            raise ValueError("closed admission gate retained an entry token")
        previous = -1
        for slot_generation, count in self.in_flight_by_generation:
            if (
                not isinstance(slot_generation, int)
                or isinstance(slot_generation, bool)
                or slot_generation < 0
                or slot_generation <= previous
                or not isinstance(count, int)
                or isinstance(count, bool)
                or count <= 0
            ):
                raise ValueError("admission in-flight readback is invalid")
            previous = slot_generation


@dataclass(frozen=True, slots=True)
class RuntimeRequestLease:
    """One request pin for the exact published runtime generation."""

    request_id: str
    gate_generation: int
    slot_generation: int
    binding_id: str
    model_authority_digest: str
    worker_generation: int
    lease_id: str
    lease_epoch: int
    runtime_snapshot: RuntimeSlotSnapshot

    def __post_init__(self) -> None:
        if _BINDING_ID.fullmatch(self.request_id) is None:
            raise ValueError("request lease id is invalid")
        for field_name in (
            "gate_generation",
            "slot_generation",
            "worker_generation",
            "lease_epoch",
        ):
            value = getattr(self, field_name)
            minimum = 0 if field_name == "slot_generation" else 1
            if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
                raise ValueError(f"request lease {field_name} is invalid")
        if _BINDING_ID.fullmatch(self.binding_id) is None:
            raise ValueError("request lease binding id is invalid")
        if _SHA256.fullmatch(self.model_authority_digest) is None:
            raise ValueError("request lease model authority is invalid")
        if not isinstance(self.lease_id, str) or not self.lease_id:
            raise ValueError("request lease GPU lease id is invalid")
        if (
            not isinstance(self.runtime_snapshot, RuntimeSlotSnapshot)
            or self.runtime_snapshot.generation != self.slot_generation
            or self.runtime_snapshot.bundle.binding_id != self.binding_id
            or self.runtime_snapshot.bundle.descriptor.authority_digest
            != self.model_authority_digest
        ):
            raise ValueError("request lease runtime snapshot is invalid")


@dataclass(frozen=True, slots=True)
class RuntimePreparationCleanupEvidence:
    preparation_id: str
    model_authority_digest: str
    aborted: bool
    factory_state_released: bool
    worker_absent: bool
    lease_absent: bool

    def __post_init__(self) -> None:
        if _BINDING_ID.fullmatch(self.preparation_id) is None:
            raise ValueError("runtime preparation id is invalid")
        if _SHA256.fullmatch(self.model_authority_digest) is None:
            raise ValueError("runtime preparation model authority is invalid")
        if not all(
            isinstance(value, bool)
            for value in (
                self.aborted,
                self.factory_state_released,
                self.worker_absent,
                self.lease_absent,
            )
        ):
            raise TypeError("runtime preparation cleanup evidence is invalid")


@dataclass(frozen=True, slots=True)
class RuntimePreparationDisposeEvidence:
    preparation_id: str
    model_authority_digest: str
    disposed: bool
    runtime_transferred: bool

    def __post_init__(self) -> None:
        if _BINDING_ID.fullmatch(self.preparation_id) is None:
            raise ValueError("runtime preparation id is invalid")
        if _SHA256.fullmatch(self.model_authority_digest) is None:
            raise ValueError("runtime preparation model authority is invalid")
        if not isinstance(self.disposed, bool) or not isinstance(
            self.runtime_transferred, bool
        ):
            raise TypeError("runtime preparation dispose evidence is invalid")


@dataclass(frozen=True, slots=True)
class ModelSwitchControlCapabilities:
    """Auditable control-plane capability declaration.

    Blocking calls are cooperative in this in-process primitive.  Crash journal
    recovery and product wiring intentionally remain partial, so production
    enablement must be refused even when test-mode switching is enabled.
    """

    cooperative_only: bool = True
    atomic_entry_validation: bool = True
    exact_generation_drain: bool = True
    two_phase_factory: bool = True
    independent_gate_readback: bool = True
    crash_journal: bool = False
    product_wiring: bool = False

    @property
    def production_ready(self) -> bool:
        return (
            not self.cooperative_only
            and self.atomic_entry_validation
            and self.exact_generation_drain
            and self.two_phase_factory
            and self.independent_gate_readback
            and self.crash_journal
            and self.product_wiring
        )

    @property
    def partial_capabilities(self) -> tuple[str, ...]:
        missing: list[str] = []
        if self.cooperative_only:
            missing.append("ENFORCED_WALL_CLOCK_ISOLATION")
        if not self.crash_journal:
            missing.append("CRASH_JOURNAL_RECOVERY")
        if not self.product_wiring:
            missing.append("PRODUCT_WIRING")
        return tuple(missing)


@dataclass(frozen=True, slots=True)
class SafeModeEvidence:
    transaction_id: str
    error_code: str
    admission_closed: bool

    def __post_init__(self) -> None:
        if not isinstance(self.transaction_id, str) or not self.transaction_id:
            raise ValueError("safe-mode transaction id is invalid")
        if _ERROR_CODE.fullmatch(self.error_code) is None:
            raise ValueError("safe-mode error code is invalid")
        if not isinstance(self.admission_closed, bool):
            raise TypeError("safe-mode admission proof must be bool")


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
    def worker_generation(self) -> int: ...

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

    def validate(
        self,
        lease: GPULease,
        *,
        purpose: str | None = None,
        required_vram_bytes: int | None = None,
    ) -> GPULease: ...

    def release_evidence(self, lease: GPULease) -> LeaseReleaseEvidence | None: ...


class MemoryReleaseProbePort(Protocol):
    def verify_release(
        self,
        bundle: ActiveRuntimeBundle,
        unload: RuntimeUnloadEvidence,
        timeout_seconds: float,
    ) -> MemoryReleaseEvidence: ...


class RuntimeBundlePreparationPort(Protocol):
    """Retained cleanup handle for every potentially stateful materialization."""

    @property
    def preparation_id(self) -> str: ...

    @property
    def descriptor(self) -> VerifiedModelSwitchDescriptor: ...

    def materialize(self) -> ActiveRuntimeBundle: ...

    def abort(self, timeout_seconds: float) -> RuntimePreparationCleanupEvidence: ...

    def dispose(self) -> RuntimePreparationDisposeEvidence: ...


class RuntimeBundleFactoryPort(Protocol):
    @property
    def side_effect_free_prepare(self) -> bool: ...

    def prepare(
        self,
        descriptor: VerifiedModelSwitchDescriptor,
        lease_authority: LeaseAuthorityPort,
        lease_profile: StableResidentLeaseAuthority,
    ) -> RuntimeBundlePreparationPort: ...


class ModelSwitchMaintenancePort(Protocol):
    def close_admission(self, transaction_id: str) -> None: ...

    def wait_for_drain(
        self, transaction_id: str, slot_generation: int, timeout_seconds: float
    ) -> bool: ...

    def checkpoint(
        self,
        transaction_id: str,
        source: VerifiedModelSwitchDescriptor,
        candidate: VerifiedModelSwitchDescriptor,
    ) -> None: ...

    def enter_safe_mode(
        self, transaction_id: str, error_code: str
    ) -> SafeModeEvidence: ...


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
            worker_generation = self.runtime.worker_generation
            lease = self.runtime.gpu_lease
            if not isinstance(alive, bool):
                raise TypeError("worker_alive")
            if (
                not isinstance(worker_generation, int)
                or isinstance(worker_generation, bool)
                or worker_generation < 0
            ):
                raise TypeError("worker_generation")
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
        initial_fence = self.publication_fence_for(bundle, expected_slot_generation=0)
        self._bundle = bundle
        self._generation = 0
        self._publication_fence = initial_fence
        self._lock = Lock()

    def snapshot(self) -> RuntimeSlotSnapshot:
        with self._lock:
            return RuntimeSlotSnapshot(self._generation, self._bundle)

    def publication_snapshot(
        self,
    ) -> tuple[RuntimeSlotSnapshot, RuntimePublicationFence]:
        """Return immutable slot identity without invoking injected callbacks."""

        with self._lock:
            return (
                RuntimeSlotSnapshot(self._generation, self._bundle),
                self._publication_fence,
            )

    @staticmethod
    def publication_fence_for(
        bundle: ActiveRuntimeBundle, *, expected_slot_generation: int
    ) -> RuntimePublicationFence:
        """Capture and independently validate a live bundle outside slot locks."""

        try:
            bundle.validate_complete()
            lease = bundle.runtime.gpu_lease
            worker_generation = bundle.runtime.worker_generation
            if (
                bundle.runtime.worker_alive is not True
                or not isinstance(lease, GPULease)
                or not isinstance(worker_generation, int)
                or isinstance(worker_generation, bool)
                or worker_generation <= 0
                or bundle.lease_authority.active != lease
            ):
                raise ValueError("publication runtime is unavailable")
            validated = bundle.lease_authority.validate(
                lease,
                purpose=bundle.lease_profile.purpose,
                required_vram_bytes=bundle.lease_profile.vram_budget_bytes,
            )
            if validated != lease:
                raise ValueError("publication lease validation changed identity")
            return RuntimePublicationFence(
                expected_slot_generation=expected_slot_generation,
                binding_id=bundle.binding_id,
                model_authority_digest=bundle.descriptor.authority_digest,
                worker_generation=worker_generation,
                lease_id=lease.lease_id,
                lease_epoch=lease.epoch,
            )
        except Exception as error:
            raise AtomicRuntimeCommitError(
                "runtime publication fence was not proved"
            ) from error

    def validates_entry_token(self, token: AdmissionFenceToken) -> bool:
        """Validate a request token without holding the slot lock over callbacks."""

        if not isinstance(token, AdmissionFenceToken):
            return False
        first, cached = self.publication_snapshot()
        if (
            first.generation != token.slot_generation
            or cached != token.publication
            or first.bundle.binding_id != token.publication.binding_id
        ):
            return False
        try:
            observed = self.publication_fence_for(
                first.bundle,
                expected_slot_generation=token.publication.expected_slot_generation,
            )
        except AtomicRuntimeCommitError:
            return False
        second, cached_after = self.publication_snapshot()
        return (
            observed == token.publication and second == first and cached_after == cached
        )

    def matches_cached_token(self, token: AdmissionFenceToken) -> bool:
        """Pure lock-bounded comparison used at the final admission boundary."""

        if not isinstance(token, AdmissionFenceToken):
            return False
        with self._lock:
            return (
                self._generation == token.slot_generation
                and self._publication_fence == token.publication
                and self._bundle.binding_id == token.publication.binding_id
                and self._bundle.descriptor.authority_digest
                == token.publication.model_authority_digest
            )

    def compare_and_swap(
        self,
        expected: RuntimeSlotSnapshot,
        replacement: ActiveRuntimeBundle,
        publication_fence: RuntimePublicationFence,
    ) -> RuntimeSlotSnapshot:
        replacement.validate_complete()
        # All injected runtime/lease callbacks execute before taking the slot
        # lock.  A re-entrant callback may publish first; the expected-generation
        # check below then rejects this outer CAS without clobbering it.
        self._validate_publication_fence(expected, replacement, publication_fence)
        with self._lock:
            if (
                self._generation != expected.generation
                or self._bundle is not expected.bundle
            ):
                raise AtomicRuntimeCommitError(
                    "active runtime changed before atomic commit"
                )
            previous_bundle = self._bundle
            previous_generation = self._generation
            previous_fence = self._publication_fence
            self._bundle = replacement
            self._generation += 1
            self._publication_fence = publication_fence
            committed = RuntimeSlotSnapshot(self._generation, replacement)
        try:
            # Re-prove after publication while admission is still closed.  No
            # callback runs under the slot lock, and rollback is conditional so
            # a re-entrant successor publication is never overwritten.
            self._validate_publication_fence(expected, replacement, publication_fence)
        except Exception:
            with self._lock:
                if (
                    self._generation == committed.generation
                    and self._bundle is replacement
                    and self._publication_fence == publication_fence
                ):
                    self._bundle = previous_bundle
                    self._generation = previous_generation
                    self._publication_fence = previous_fence
            raise
        with self._lock:
            if (
                self._generation != committed.generation
                or self._bundle is not replacement
                or self._publication_fence != publication_fence
            ):
                raise AtomicRuntimeCommitError(
                    "active runtime changed during atomic commit postcheck"
                )
            return committed

    @staticmethod
    def _validate_publication_fence(
        expected: RuntimeSlotSnapshot,
        replacement: ActiveRuntimeBundle,
        fence: RuntimePublicationFence,
    ) -> None:
        try:
            lease = replacement.runtime.gpu_lease
            generation = replacement.runtime.worker_generation
            if (
                not isinstance(fence, RuntimePublicationFence)
                or fence.expected_slot_generation != expected.generation
                or fence.binding_id != replacement.binding_id
                or fence.model_authority_digest
                != replacement.descriptor.authority_digest
                or replacement.runtime.worker_alive is not True
                or not isinstance(generation, int)
                or isinstance(generation, bool)
                or generation <= 0
                or fence.worker_generation != generation
                or not isinstance(lease, GPULease)
                or fence.lease_id != lease.lease_id
                or fence.lease_epoch != lease.epoch
                or replacement.lease_authority.active != lease
            ):
                raise ValueError("publication identity changed")
            validated = replacement.lease_authority.validate(
                lease,
                purpose=replacement.lease_profile.purpose,
                required_vram_bytes=replacement.lease_profile.vram_budget_bytes,
            )
            if validated != lease:
                raise ValueError("publication lease validation changed identity")
        except Exception as error:
            raise AtomicRuntimeCommitError(
                "runtime publication fence was not proved"
            ) from error


class AtomicAdmissionGate:
    """Token-gated request admission with exact runtime-generation pins.

    Installing a token merely arms the entry validator.  Every request must
    revalidate the live publication and atomically acquire a generation pin;
    there is no interval in which a bare boolean `open` admits work.
    """

    def __init__(self, slot: AtomicRuntimeSlot) -> None:
        if not isinstance(slot, AtomicRuntimeSlot):
            raise TypeError("slot must be AtomicRuntimeSlot")
        snapshot, publication = slot.publication_snapshot()
        token = AdmissionFenceToken(
            transaction_id="bootstrap",
            slot_generation=snapshot.generation,
            publication=publication,
        )
        if not slot.validates_entry_token(token):
            raise AtomicRuntimeCommitError("initial admission fence was not proved")
        self._condition = Condition(Lock())
        self._mode = "open"
        self._gate_generation = 1
        self._transaction_id: str | None = token.transaction_id
        self._token: AdmissionFenceToken | None = token
        self._active_requests: dict[str, RuntimeRequestLease] = {}
        self._in_flight: dict[int, int] = {}

    def readback(self) -> AdmissionGateReadback:
        with self._condition:
            return self._readback_locked()

    def close(self, transaction_id: str) -> AdmissionGateReadback:
        if not isinstance(transaction_id, str) or not transaction_id:
            raise ValueError("admission transaction id is invalid")
        with self._condition:
            self._mode = "closed"
            self._gate_generation += 1
            self._transaction_id = transaction_id
            self._token = None
            self._condition.notify_all()
            return self._readback_locked()

    def install(
        self,
        transaction_id: str,
        token: AdmissionFenceToken,
        slot: AtomicRuntimeSlot,
    ) -> AdmissionOpenEvidence:
        if token.transaction_id != transaction_id:
            raise AtomicRuntimeCommitError("admission token transaction changed")
        if not slot.validates_entry_token(token):
            raise AtomicRuntimeCommitError("admission entry fence was not proved")
        with self._condition:
            if self._mode != "closed" or self._transaction_id != transaction_id:
                raise AtomicRuntimeCommitError(
                    "admission gate is not closed for switch"
                )
            if any(self._in_flight.values()):
                raise AtomicRuntimeCommitError(
                    "admission gate still has in-flight work"
                )
            if not slot.matches_cached_token(token):
                raise AtomicRuntimeCommitError("admission slot changed before install")
            self._gate_generation += 1
            self._mode = "open"
            self._token = token
            evidence = AdmissionOpenEvidence(
                token=token,
                gate_generation=self._gate_generation,
                entry_validation_required=True,
            )
            self._condition.notify_all()
            return evidence

    def acquire(self, slot: AtomicRuntimeSlot) -> RuntimeRequestLease | None:
        """Acquire a request pin or reject if any publication evidence changed."""

        with self._condition:
            if self._mode != "open" or self._token is None:
                return None
            token = self._token
            gate_generation = self._gate_generation
        if not slot.validates_entry_token(token):
            return None
        with self._condition:
            if (
                self._mode != "open"
                or self._token != token
                or self._gate_generation != gate_generation
                or not slot.matches_cached_token(token)
            ):
                return None
            publication = token.publication
            runtime_snapshot = slot.snapshot()
            if runtime_snapshot.generation != token.slot_generation:
                return None
            request = RuntimeRequestLease(
                request_id=secrets.token_hex(16),
                gate_generation=gate_generation,
                slot_generation=token.slot_generation,
                binding_id=publication.binding_id,
                model_authority_digest=publication.model_authority_digest,
                worker_generation=publication.worker_generation,
                lease_id=publication.lease_id,
                lease_epoch=publication.lease_epoch,
                runtime_snapshot=runtime_snapshot,
            )
            self._active_requests[request.request_id] = request
            self._in_flight[request.slot_generation] = (
                self._in_flight.get(request.slot_generation, 0) + 1
            )
            return request

    def release(self, request: RuntimeRequestLease) -> None:
        if not isinstance(request, RuntimeRequestLease):
            raise TypeError("request must be RuntimeRequestLease")
        with self._condition:
            active = self._active_requests.pop(request.request_id, None)
            if active != request:
                raise ValueError("request lease is unknown or already released")
            count = self._in_flight.get(request.slot_generation, 0)
            if count <= 1:
                self._in_flight.pop(request.slot_generation, None)
            else:
                self._in_flight[request.slot_generation] = count - 1
            self._condition.notify_all()

    def wait_for_generation_drain(
        self, transaction_id: str, slot_generation: int, timeout_seconds: float
    ) -> bool:
        if not isinstance(transaction_id, str) or not transaction_id:
            raise ValueError("admission transaction id is invalid")
        if (
            not isinstance(slot_generation, int)
            or isinstance(slot_generation, bool)
            or slot_generation < 0
        ):
            raise ValueError("slot generation is invalid")
        if (
            not isinstance(timeout_seconds, (int, float))
            or isinstance(timeout_seconds, bool)
            or not math.isfinite(float(timeout_seconds))
            or timeout_seconds <= 0
        ):
            raise ValueError("drain timeout is invalid")
        deadline = monotonic() + float(timeout_seconds)
        with self._condition:
            if self._mode != "closed" or self._transaction_id != transaction_id:
                return False
            while self._in_flight.get(slot_generation, 0) > 0:
                remaining = deadline - monotonic()
                if remaining <= 0:
                    return False
                self._condition.wait(timeout=remaining)
                if self._mode != "closed" or self._transaction_id != transaction_id:
                    return False
            return True

    def enter_safe_mode(
        self, transaction_id: str, error_code: str
    ) -> AdmissionGateReadback:
        if not isinstance(transaction_id, str) or not transaction_id:
            raise ValueError("admission transaction id is invalid")
        if _ERROR_CODE.fullmatch(error_code) is None:
            raise ValueError("safe-mode error code is invalid")
        with self._condition:
            self._mode = "safe_mode"
            self._gate_generation += 1
            self._transaction_id = transaction_id
            self._token = None
            self._condition.notify_all()
            return self._readback_locked()

    def _readback_locked(self) -> AdmissionGateReadback:
        return AdmissionGateReadback(
            gate_generation=self._gate_generation,
            mode=self._mode,
            transaction_id=self._transaction_id,
            token=self._token,
            in_flight_by_generation=tuple(sorted(self._in_flight.items())),
        )


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
    SAFE_MODE_UNPROVEN = "safe_mode_unproven"


_ALLOWED_TRANSITIONS: dict[ModelSwitchState, set[ModelSwitchState]] = {
    ModelSwitchState.REQUESTED: {
        ModelSwitchState.DRAINING,
        ModelSwitchState.ROLLING_BACK,
        ModelSwitchState.SAFE_MODE,
        ModelSwitchState.SAFE_MODE_UNPROVEN,
    },
    ModelSwitchState.DRAINING: {
        ModelSwitchState.CHECKPOINTING,
        ModelSwitchState.ROLLING_BACK,
        ModelSwitchState.SAFE_MODE,
        ModelSwitchState.SAFE_MODE_UNPROVEN,
    },
    ModelSwitchState.CHECKPOINTING: {
        ModelSwitchState.UNLOADING_OLD,
        ModelSwitchState.ROLLING_BACK,
        ModelSwitchState.SAFE_MODE,
        ModelSwitchState.SAFE_MODE_UNPROVEN,
    },
    ModelSwitchState.UNLOADING_OLD: {
        ModelSwitchState.LOADING_CANDIDATE,
        ModelSwitchState.ROLLING_BACK,
        ModelSwitchState.SAFE_MODE,
        ModelSwitchState.SAFE_MODE_UNPROVEN,
    },
    ModelSwitchState.LOADING_CANDIDATE: {
        ModelSwitchState.HEALTHCHECKING,
        ModelSwitchState.ROLLING_BACK,
        ModelSwitchState.SAFE_MODE,
        ModelSwitchState.SAFE_MODE_UNPROVEN,
    },
    ModelSwitchState.HEALTHCHECKING: {
        ModelSwitchState.COMMITTING,
        ModelSwitchState.ROLLING_BACK,
        ModelSwitchState.SAFE_MODE,
        ModelSwitchState.SAFE_MODE_UNPROVEN,
    },
    ModelSwitchState.COMMITTING: {
        ModelSwitchState.SUCCEEDED,
        ModelSwitchState.ROLLING_BACK,
        ModelSwitchState.SAFE_MODE,
        ModelSwitchState.SAFE_MODE_UNPROVEN,
    },
    ModelSwitchState.ROLLING_BACK: {
        ModelSwitchState.RESTORING_OLD,
        ModelSwitchState.ROLLED_BACK,
        ModelSwitchState.SAFE_MODE,
        ModelSwitchState.SAFE_MODE_UNPROVEN,
    },
    ModelSwitchState.RESTORING_OLD: {
        ModelSwitchState.ROLLED_BACK,
        ModelSwitchState.SAFE_MODE,
        ModelSwitchState.SAFE_MODE_UNPROVEN,
    },
    ModelSwitchState.SUCCEEDED: set(),
    ModelSwitchState.ROLLED_BACK: set(),
    ModelSwitchState.SAFE_MODE: set(),
    ModelSwitchState.SAFE_MODE_UNPROVEN: set(),
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
    safety_error_code: str | None
    rollback_restored: bool
    cooperative_only: bool
    production_ready: bool
    partial_capabilities: tuple[str, ...]
    transitions: tuple[ModelSwitchTransition, ...]


@dataclass(slots=True)
class _MutableTransaction:
    transaction_id: str
    source_model_id: str
    candidate_model_id: str
    state: ModelSwitchState
    transitions: list[ModelSwitchTransition]
    error_code: str | None = None
    safety_error_code: str | None = None
    rollback_restored: bool = False
    capabilities: ModelSwitchControlCapabilities = ModelSwitchControlCapabilities()

    def snapshot(self) -> ModelSwitchSnapshot:
        return ModelSwitchSnapshot(
            transaction_id=self.transaction_id,
            source_model_id=self.source_model_id,
            candidate_model_id=self.candidate_model_id,
            state=self.state,
            error_code=self.error_code,
            safety_error_code=self.safety_error_code,
            rollback_restored=self.rollback_restored,
            cooperative_only=self.capabilities.cooperative_only,
            production_ready=self.capabilities.production_ready,
            partial_capabilities=self.capabilities.partial_capabilities,
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
        admission_gate: AtomicAdmissionGate | None = None,
        enabled: bool = False,
        production_enable: bool = False,
        memory_release_probe: MemoryReleaseProbePort | None = None,
        capabilities: ModelSwitchControlCapabilities | None = None,
        clock=monotonic,
    ) -> None:
        if not isinstance(slot, AtomicRuntimeSlot):
            raise TypeError("slot must be AtomicRuntimeSlot")
        if not isinstance(enabled, bool):
            raise TypeError("enabled must be bool")
        if not isinstance(production_enable, bool):
            raise TypeError("production_enable must be bool")
        if not callable(clock):
            raise TypeError("clock must be callable")
        if admission_gate is not None and not isinstance(
            admission_gate, AtomicAdmissionGate
        ):
            raise TypeError("admission_gate must be AtomicAdmissionGate")
        declared = capabilities or ModelSwitchControlCapabilities()
        if not isinstance(declared, ModelSwitchControlCapabilities):
            raise TypeError("capabilities must be ModelSwitchControlCapabilities")
        if declared != ModelSwitchControlCapabilities():
            raise ValueError(
                "in-process model switch capability evidence cannot be overridden"
            )
        if production_enable:
            raise ModelSwitchProductionUnavailableError(
                "model switching lacks enforced wall-clock isolation, crash recovery, "
                "or product wiring"
            )
        if memory_release_probe is not None and not callable(
            getattr(memory_release_probe, "verify_release", None)
        ):
            raise TypeError("memory_release_probe must implement verify_release")
        if enabled and memory_release_probe is None:
            raise ValueError(
                "enabled model switching requires an injected memory-release probe"
            )
        if enabled and admission_gate is None:
            raise ValueError(
                "enabled model switching requires an atomic admission gate"
            )
        if enabled and admission_gate is not None:
            initial_gate = admission_gate.readback()
            if (
                initial_gate.mode != "open"
                or initial_gate.token is None
                or not slot.validates_entry_token(initial_gate.token)
            ):
                raise ValueError(
                    "enabled model switching requires a gate bound to the active slot"
                )
        if enabled and getattr(factory, "side_effect_free_prepare", None) is not True:
            raise ValueError(
                "enabled model switching requires a side-effect-free factory prepare"
            )
        if enabled and not callable(getattr(factory, "prepare", None)):
            raise ValueError("enabled model switching requires a two-phase factory")
        self.slot = slot
        self.factory = factory
        self.maintenance = maintenance
        self.admission_gate = admission_gate
        self.enabled = enabled
        self.production_enable = production_enable
        self.capabilities = declared
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

            # Cancellation is checked before invoking the factory.  A factory
            # is allowed to allocate host-side loader state, so a request that
            # is already cancelled must never call it.
            if cancellation is not None and cancellation.requested:
                self._begin(source.descriptor, candidate)
                with self._state_lock:
                    assert self._transaction is not None
                    self._transaction.error_code = "MODEL_SWITCH_CANCELLED"
                self._transition(ModelSwitchState.ROLLING_BACK)
                self._transition(ModelSwitchState.ROLLED_BACK)
                return self._snapshot_required()

            # Factory.prepare is an admitted side-effect-free planning call.  The
            # returned handle is retained before materialization so every later
            # partial allocation has a mandatory abort/dispose path.
            try:
                preparation = self.factory.prepare(
                    candidate, source.lease_authority, source.lease_profile
                )
            except Exception:
                raise RuntimeBundleIncompleteError(
                    "candidate runtime factory prepare failed"
                ) from None
            try:
                self._validate_preparation_handle(preparation, candidate)
            except RuntimeBundleIncompleteError:
                self._begin(source.descriptor, candidate)
                with self._state_lock:
                    assert self._transaction is not None
                    self._transaction.error_code = "RUNTIME_BUNDLE_INCOMPLETE"
                    self._transaction.safety_error_code = "PREFLIGHT_CLEANUP_UNPROVEN"
                self._transition(ModelSwitchState.ROLLING_BACK)
                return self._safe_mode(
                    "RUNTIME_BUNDLE_INCOMPLETE",
                    safety_error_code="PREFLIGHT_CLEANUP_UNPROVEN",
                )

            prepared: ActiveRuntimeBundle | None = None
            try:
                prepared = preparation.materialize()
                self._validate_prepared_bundle(prepared, candidate, source)
            except Exception as preflight_error:
                cleanup_proven = self._abort_preparation(
                    preparation,
                    prepared,
                    source,
                    float(stop_timeout_seconds),
                    float(memory_timeout_seconds),
                )
                if cleanup_proven:
                    if isinstance(preflight_error, RuntimeBundleIncompleteError):
                        raise preflight_error
                    raise RuntimeBundleIncompleteError(
                        "candidate runtime materialization failed"
                    ) from None
                self._begin(source.descriptor, candidate)
                with self._state_lock:
                    assert self._transaction is not None
                    self._transaction.error_code = "RUNTIME_BUNDLE_INCOMPLETE"
                    self._transaction.safety_error_code = "PREFLIGHT_CLEANUP_UNPROVEN"
                self._transition(ModelSwitchState.ROLLING_BACK)
                return self._safe_mode(
                    "RUNTIME_BUNDLE_INCOMPLETE",
                    safety_error_code="PREFLIGHT_CLEANUP_UNPROVEN",
                )

            assert prepared is not None

            transaction = self._begin(source.descriptor, candidate)
            source_retired = False
            candidate_started = False
            admission_closed = False
            committed_slot: RuntimeSlotSnapshot | None = None
            source_epoch_floor = source.lease_authority.latest_epoch
            try:
                self._check_cancelled(cancellation)
                gate = self._admission_gate_required()
                gate_closed = gate.close(transaction.transaction_id)
                self._prove_gate_closed(
                    gate_closed, transaction.transaction_id, mode="closed"
                )
                self._call(
                    "ADMISSION_CLOSE_FAILED",
                    self.maintenance.close_admission,
                    transaction.transaction_id,
                )
                self._prove_gate_closed(
                    gate.readback(), transaction.transaction_id, mode="closed"
                )
                admission_closed = True
                self._transition(ModelSwitchState.DRAINING)
                drain_deadline = self._deadline_after(float(drain_timeout_seconds))
                exact_drained = gate.wait_for_generation_drain(
                    transaction.transaction_id,
                    source_slot.generation,
                    self._remaining(drain_deadline, "DRAIN_TIMEOUT"),
                )
                if exact_drained is not True:
                    raise _TransactionFailure("DRAIN_TIMEOUT")
                drained = self._call(
                    "DRAIN_FAILED",
                    self.maintenance.wait_for_drain,
                    transaction.transaction_id,
                    source_slot.generation,
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
                old_worker_generation = source.runtime.worker_generation
                if not isinstance(old_lease, GPULease):
                    raise _TransactionFailure("SOURCE_WORKER_NOT_RUNNING")
                if (
                    not isinstance(old_worker_generation, int)
                    or isinstance(old_worker_generation, bool)
                    or old_worker_generation <= 0
                ):
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
                self._prove_unload_ack(source, old_lease, old_worker_generation, unload)
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
                publication = self._prove_publication_fence(
                    prepared,
                    expected_slot_generation=source_slot.generation,
                    expected_lease=candidate_lease,
                    expected_worker_generation=health.worker_generation,
                )
                self._claim_commit(cancellation)
                committed_slot = self.slot.compare_and_swap(
                    source_slot, prepared, publication
                )
                # The cancellation token's commit claim and this CAS form one
                # publication boundary: later cancellation is rejected, while
                # a pre-existing request cannot reach the CAS.
                self._dispose_preparation(preparation, prepared)
                self._open_admission_with_fence(
                    prepared,
                    committed_slot,
                    candidate_lease,
                    health.worker_generation,
                    "ADMISSION_OPEN",
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
                    candidate_preparation=preparation,
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
            capabilities=self.capabilities,
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

    @staticmethod
    def _validate_preparation_handle(
        preparation: RuntimeBundlePreparationPort,
        candidate: VerifiedModelSwitchDescriptor,
    ) -> None:
        try:
            preparation_id = preparation.preparation_id
            descriptor = preparation.descriptor
            methods = (
                preparation.materialize,
                preparation.abort,
                preparation.dispose,
            )
        except Exception as error:
            raise RuntimeBundleIncompleteError(
                "factory returned no retained preparation handle"
            ) from error
        if (
            _BINDING_ID.fullmatch(preparation_id) is None
            or descriptor != candidate
            or not all(callable(method) for method in methods)
        ):
            raise RuntimeBundleIncompleteError(
                "runtime preparation handle escaped its authority"
            )

    def _abort_preparation(
        self,
        preparation: RuntimeBundlePreparationPort,
        prepared: object | None,
        source: ActiveRuntimeBundle,
        stop_timeout_seconds: float,
        memory_timeout_seconds: float,
        *,
        was_started: bool = False,
    ) -> bool:
        """Abort factory state and independently prove any visible worker retired."""

        runtime_cleanup_proven = True
        try:
            if isinstance(prepared, ActiveRuntimeBundle):
                alive = prepared.runtime.worker_alive
                prior_lease = prepared.runtime.gpu_lease
                prior_generation = prepared.runtime.worker_generation
                if was_started and alive is not True and prior_lease is None:
                    # A formerly resident worker vanished without an unload ack;
                    # factory cleanup evidence cannot substitute for the
                    # independent memory-release probe.
                    runtime_cleanup_proven = False
                if alive is True or prior_lease is not None:
                    if not isinstance(prior_lease, GPULease):
                        # The retained abort handle is the only authority that
                        # can clean an invalid partially-started no-lease worker.
                        if was_started:
                            runtime_cleanup_proven = False
                    else:
                        if (
                            not isinstance(prior_generation, int)
                            or isinstance(prior_generation, bool)
                            or prior_generation <= 0
                        ):
                            runtime_cleanup_proven = False
                        else:
                            try:
                                deadline = self._deadline_after(stop_timeout_seconds)
                                unload = prepared.runtime.stop(
                                    self._remaining(
                                        deadline, "PREFLIGHT_CLEANUP_TIMEOUT"
                                    )
                                )
                                if self._expired(deadline):
                                    raise _TransactionFailure(
                                        "PREFLIGHT_CLEANUP_TIMEOUT"
                                    )
                                self._prove_retired(prepared, prior_lease)
                                self._prove_unload_ack(
                                    prepared, prior_lease, prior_generation, unload
                                )
                                self._prove_memory_released(
                                    prepared,
                                    unload,
                                    memory_timeout_seconds,
                                    "PREFLIGHT_MEMORY_RELEASE_UNPROVEN",
                                )
                            except Exception:
                                runtime_cleanup_proven = False
            cleanup_deadline = self._deadline_after(stop_timeout_seconds)
            evidence = preparation.abort(
                self._remaining(cleanup_deadline, "PREFLIGHT_CLEANUP_TIMEOUT")
            )
            if self._expired(cleanup_deadline):
                return False
            if (
                not isinstance(evidence, RuntimePreparationCleanupEvidence)
                or evidence.preparation_id != preparation.preparation_id
                or evidence.model_authority_digest
                != preparation.descriptor.authority_digest
                or evidence.aborted is not True
                or evidence.factory_state_released is not True
                or evidence.worker_absent is not True
                or evidence.lease_absent is not True
            ):
                return False
            if isinstance(prepared, ActiveRuntimeBundle) and (
                prepared.runtime.worker_alive is not False
                or prepared.runtime.gpu_lease is not None
            ):
                return False
            if source.runtime.worker_alive:
                self._prove_source_consistent(source)
            return runtime_cleanup_proven
        except Exception:
            return False

    @staticmethod
    def _dispose_preparation(
        preparation: RuntimeBundlePreparationPort,
        bundle: ActiveRuntimeBundle,
    ) -> None:
        try:
            evidence = preparation.dispose()
        except Exception as error:
            raise _TransactionFailure("PREPARATION_DISPOSE_FAILED") from error
        if (
            not isinstance(evidence, RuntimePreparationDisposeEvidence)
            or evidence.preparation_id != preparation.preparation_id
            or evidence.model_authority_digest != bundle.descriptor.authority_digest
            or evidence.disposed is not True
            or evidence.runtime_transferred is not True
            or bundle.runtime.worker_alive is not True
            or bundle.runtime.gpu_lease is None
        ):
            raise _TransactionFailure("PREPARATION_DISPOSE_UNPROVEN")

    @classmethod
    def _prove_source_consistent(cls, bundle: ActiveRuntimeBundle) -> None:
        alive = bundle.runtime.worker_alive
        generation = bundle.runtime.worker_generation
        lease = bundle.runtime.gpu_lease
        active = bundle.lease_authority.active
        if (
            alive is not True
            or not isinstance(generation, int)
            or isinstance(generation, bool)
            or generation <= 0
        ):
            raise RuntimeBundleIncompleteError(
                "published resident worker is unavailable"
            )
        if lease is None or active != lease:
            raise RuntimeBundleIncompleteError(
                "active worker lacks its exact resident lease"
            )
        cls._prove_lease_profile(lease, bundle.lease_profile)
        try:
            cls._validate_authority_lease(
                bundle.lease_authority,
                bundle.lease_profile,
                lease,
                "SOURCE_LEASE_UNPROVEN",
            )
        except _TransactionFailure as error:
            raise RuntimeBundleIncompleteError(
                "published resident lease is stale or expired"
            ) from error

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

    @staticmethod
    def _validate_authority_lease(
        authority: LeaseAuthorityPort,
        profile: StableResidentLeaseAuthority,
        lease: GPULease,
        failure_code: str,
    ) -> GPULease:
        try:
            validated = authority.validate(
                lease,
                purpose=profile.purpose,
                required_vram_bytes=profile.vram_budget_bytes,
            )
        except Exception as error:
            raise _TransactionFailure(failure_code) from error
        if validated != lease:
            raise _TransactionFailure(failure_code)
        return validated

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
        prior_worker_generation: int,
        unload: RuntimeUnloadEvidence | None,
    ) -> None:
        if (
            not isinstance(unload, RuntimeUnloadEvidence)
            or unload.acknowledged is not True
            or unload.binding_id != bundle.binding_id
            or unload.model_authority_digest != bundle.descriptor.authority_digest
            or unload.lease_epoch != prior_lease.epoch
            or unload.worker_generation != prior_worker_generation
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
            or evidence.worker_generation != unload.worker_generation
        ):
            raise _TransactionFailure(failure_code)

    @staticmethod
    def _require_unleased(authority: LeaseAuthorityPort) -> None:
        if authority.active is not None:
            raise _TransactionFailure("GPU_LEASE_NOT_RELEASED")

    @classmethod
    def _prove_running(cls, bundle: ActiveRuntimeBundle, epoch_floor: int) -> GPULease:
        lease = bundle.runtime.gpu_lease
        generation = bundle.runtime.worker_generation
        if (
            bundle.runtime.worker_alive is not True
            or not isinstance(generation, int)
            or isinstance(generation, bool)
            or generation <= 0
            or not isinstance(lease, GPULease)
            or bundle.lease_authority.active != lease
            or lease.epoch <= epoch_floor
        ):
            raise _TransactionFailure("WORKER_START_UNPROVEN")
        cls._prove_lease_profile(lease, bundle.lease_profile)
        cls._validate_authority_lease(
            bundle.lease_authority,
            bundle.lease_profile,
            lease,
            "WORKER_START_UNPROVEN",
        )
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
            or health.worker_generation != bundle.runtime.worker_generation
            or bundle.runtime.worker_alive is not True
            or bundle.runtime.gpu_lease != lease
            or bundle.lease_authority.active != lease
        ):
            raise _TransactionFailure("CANDIDATE_HEALTH_UNPROVEN")
        ModelSwitchController._validate_authority_lease(
            bundle.lease_authority,
            bundle.lease_profile,
            lease,
            "CANDIDATE_HEALTH_UNPROVEN",
        )
        bundle.validate_complete()

    @classmethod
    def _prove_publication_fence(
        cls,
        bundle: ActiveRuntimeBundle,
        *,
        expected_slot_generation: int,
        expected_lease: GPULease,
        expected_worker_generation: int,
    ) -> RuntimePublicationFence:
        if (
            bundle.runtime.worker_alive is not True
            or bundle.runtime.gpu_lease != expected_lease
            or bundle.runtime.worker_generation != expected_worker_generation
            or bundle.lease_authority.active != expected_lease
        ):
            raise _TransactionFailure("PUBLICATION_FENCE_UNPROVEN")
        cls._prove_lease_profile(expected_lease, bundle.lease_profile)
        cls._validate_authority_lease(
            bundle.lease_authority,
            bundle.lease_profile,
            expected_lease,
            "PUBLICATION_FENCE_UNPROVEN",
        )
        bundle.validate_complete()
        return RuntimePublicationFence(
            expected_slot_generation=expected_slot_generation,
            binding_id=bundle.binding_id,
            model_authority_digest=bundle.descriptor.authority_digest,
            worker_generation=expected_worker_generation,
            lease_id=expected_lease.lease_id,
            lease_epoch=expected_lease.epoch,
        )

    def _open_admission_with_fence(
        self,
        bundle: ActiveRuntimeBundle,
        slot_snapshot: RuntimeSlotSnapshot,
        lease: GPULease,
        worker_generation: int,
        code_prefix: str,
    ) -> None:
        observed, cached_publication = self.slot.publication_snapshot()
        if (
            observed.generation != slot_snapshot.generation
            or observed.bundle is not bundle
        ):
            raise _TransactionFailure(f"{code_prefix}_FENCE_UNPROVEN")
        publication = self._prove_publication_fence(
            bundle,
            expected_slot_generation=cached_publication.expected_slot_generation,
            expected_lease=lease,
            expected_worker_generation=worker_generation,
        )
        if publication != cached_publication:
            raise _TransactionFailure(f"{code_prefix}_FENCE_UNPROVEN")
        token = AdmissionFenceToken(
            transaction_id=self._transaction_id(),
            slot_generation=slot_snapshot.generation,
            publication=publication,
        )
        gate = self._admission_gate_required()
        try:
            evidence = gate.install(self._transaction_id(), token, self.slot)
        except Exception as error:
            raise _TransactionFailure(f"{code_prefix}_FENCE_UNPROVEN") from error
        if (
            not isinstance(evidence, AdmissionOpenEvidence)
            or evidence.token != token
            or evidence.entry_validation_required is not True
        ):
            raise _TransactionFailure(f"{code_prefix}_FENCE_UNPROVEN")
        readback = gate.readback()
        if (
            readback.mode != "open"
            or readback.transaction_id != self._transaction_id()
            or readback.token != token
            or readback.gate_generation != evidence.gate_generation
        ):
            raise _TransactionFailure(f"{code_prefix}_FENCE_UNPROVEN")
        observed_after, cached_after = self.slot.publication_snapshot()
        if (
            observed_after.generation != slot_snapshot.generation
            or observed_after.bundle is not bundle
            or cached_after != publication
        ):
            raise _TransactionFailure(f"{code_prefix}_FENCE_UNPROVEN")
        after = self._prove_publication_fence(
            bundle,
            expected_slot_generation=publication.expected_slot_generation,
            expected_lease=lease,
            expected_worker_generation=worker_generation,
        )
        if after != publication:
            raise _TransactionFailure(f"{code_prefix}_FENCE_UNPROVEN")

    def _rollback(
        self,
        failure: _TransactionFailure,
        *,
        source_slot: RuntimeSlotSnapshot,
        committed_slot: RuntimeSlotSnapshot | None,
        source_retired: bool,
        admission_closed: bool,
        candidate: ActiveRuntimeBundle,
        candidate_preparation: RuntimeBundlePreparationPort,
        candidate_started: bool,
        stop_timeout_seconds: float,
        memory_timeout_seconds: float,
        health_timeout_seconds: float,
    ) -> ModelSwitchSnapshot:
        with self._state_lock:
            assert self._transaction is not None
            self._transaction.error_code = failure.code
        self._transition(ModelSwitchState.ROLLING_BACK)

        candidate_cleanup = self._abort_preparation(
            candidate_preparation,
            candidate,
            source_slot.bundle,
            stop_timeout_seconds,
            memory_timeout_seconds,
            was_started=candidate_started,
        )
        if not candidate_cleanup:
            return self._safe_mode(
                failure.code, safety_error_code="CANDIDATE_PREPARATION_CLEANUP_UNPROVEN"
            )

        # These failures occur before an exact unload acknowledgement and memory
        # probe have both completed.  No old or candidate worker may be created,
        # even when process death and lease release happened to be observed.
        if failure.code in {
            "OLD_UNLOAD_FAILED",
            "OLD_UNLOAD_TIMEOUT",
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
        if failure.code in {
            "ATOMIC_RUNTIME_CHANGED",
            "ATOMIC_RUNTIME_COMMIT_FAILED",
        }:
            return self._safe_mode(failure.code)

        restored: ActiveRuntimeBundle | None = None
        restored_preparation: RuntimeBundlePreparationPort | None = None
        restored_committed = False
        restored_started = False
        try:
            if not source_retired:
                self._prove_source_consistent(source_slot.bundle)
                if admission_closed:
                    source_lease = source_slot.bundle.runtime.gpu_lease
                    if not isinstance(source_lease, GPULease):
                        raise _TransactionFailure("SOURCE_WORKER_NOT_RUNNING")
                    self._open_admission_with_fence(
                        source_slot.bundle,
                        source_slot,
                        source_lease,
                        source_slot.bundle.runtime.worker_generation,
                        "ADMISSION_RESTORE",
                    )
                self._transition(ModelSwitchState.ROLLED_BACK)
                return self._snapshot_required()

            self._transition(ModelSwitchState.RESTORING_OLD)
            restored_preparation = self.factory.prepare(
                source_slot.bundle.descriptor,
                source_slot.bundle.lease_authority,
                source_slot.bundle.lease_profile,
            )
            self._validate_preparation_handle(
                restored_preparation, source_slot.bundle.descriptor
            )
            restored = restored_preparation.materialize()
            self._validate_prepared_bundle(
                restored, source_slot.bundle.descriptor, source_slot.bundle
            )
            epoch_floor = restored.lease_authority.latest_epoch
            self._call("ROLLBACK_START_FAILED", restored.runtime.start)
            restored_started = True
            restored_lease = self._prove_running(restored, epoch_floor)
            restored_worker_generation = restored.runtime.worker_generation
            health_deadline = self._deadline_after(health_timeout_seconds)
            restored_health = self._call(
                "ROLLBACK_HEALTHCHECK_FAILED",
                restored.runtime.healthcheck,
                self._remaining(health_deadline, "ROLLBACK_HEALTHCHECK_TIMEOUT"),
            )
            if self._expired(health_deadline):
                raise _TransactionFailure("ROLLBACK_HEALTHCHECK_TIMEOUT")
            self._prove_healthy(restored, restored_lease, restored_health)
            expected = committed_slot or source_slot
            publication = self._prove_publication_fence(
                restored,
                expected_slot_generation=expected.generation,
                expected_lease=restored_lease,
                expected_worker_generation=restored_worker_generation,
            )
            restored_slot = self.slot.compare_and_swap(expected, restored, publication)
            restored_committed = True
            self._dispose_preparation(restored_preparation, restored)
            self._open_admission_with_fence(
                restored,
                restored_slot,
                restored_lease,
                restored_worker_generation,
                "ADMISSION_RESTORE",
            )
            with self._state_lock:
                assert self._transaction is not None
                self._transaction.rollback_restored = True
            self._transition(ModelSwitchState.ROLLED_BACK)
            return self._snapshot_required()
        except Exception as rollback_error:
            cleanup_code = self._stable_failure_code(rollback_error, "ROLLBACK_FAILED")
            if restored_preparation is not None:
                cleanup_proven = self._abort_preparation(
                    restored_preparation,
                    restored,
                    source_slot.bundle,
                    stop_timeout_seconds,
                    memory_timeout_seconds,
                    was_started=restored_started or restored_committed,
                )
                if not cleanup_proven:
                    cleanup_code = "ROLLBACK_CLEANUP_UNPROVEN"
            return self._safe_mode(failure.code, safety_error_code=cleanup_code)

    @staticmethod
    def _stable_failure_code(error: Exception, fallback: str) -> str:
        code = getattr(error, "code", None)
        if isinstance(code, str) and _ERROR_CODE.fullmatch(code):
            return code
        return fallback

    def _admission_gate_required(self) -> AtomicAdmissionGate:
        gate = self.admission_gate
        if not isinstance(gate, AtomicAdmissionGate):
            raise _TransactionFailure("ADMISSION_GATE_MISSING")
        return gate

    @staticmethod
    def _prove_gate_closed(
        readback: AdmissionGateReadback,
        transaction_id: str,
        *,
        mode: str,
    ) -> None:
        if (
            not isinstance(readback, AdmissionGateReadback)
            or readback.mode != mode
            or readback.transaction_id != transaction_id
            or readback.token is not None
        ):
            raise _TransactionFailure("ADMISSION_GATE_READBACK_UNPROVEN")

    def _safe_mode(
        self, error_code: str, *, safety_error_code: str | None = None
    ) -> ModelSwitchSnapshot:
        transaction_id = self._transaction_id()
        proof_valid = False
        try:
            gate = self._admission_gate_required()
            gate_evidence = gate.enter_safe_mode(transaction_id, error_code)
            self._prove_gate_closed(gate_evidence, transaction_id, mode="safe_mode")
            evidence = self.maintenance.enter_safe_mode(transaction_id, error_code)
            gate_readback = gate.readback()
            self._prove_gate_closed(gate_readback, transaction_id, mode="safe_mode")
            proof_valid = (
                isinstance(evidence, SafeModeEvidence)
                and evidence.transaction_id == transaction_id
                and evidence.error_code == error_code
                and evidence.admission_closed is True
            )
        except Exception:
            proof_valid = False
        with self._state_lock:
            assert self._transaction is not None
            if safety_error_code is not None:
                self._transaction.safety_error_code = safety_error_code
            if not proof_valid:
                self._transaction.safety_error_code = "SAFE_MODE_UNPROVEN"
            state = self._transaction.state
        target = (
            ModelSwitchState.SAFE_MODE
            if proof_valid
            else ModelSwitchState.SAFE_MODE_UNPROVEN
        )
        if target in _ALLOWED_TRANSITIONS[state]:
            self._transition(target)
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
    "AdmissionFenceToken",
    "AdmissionGateReadback",
    "AdmissionOpenEvidence",
    "AtomicAdmissionGate",
    "AtomicRuntimeCommitError",
    "AtomicRuntimeSlot",
    "LeaseReleaseEvidence",
    "MemoryReleaseEvidence",
    "ModelSwitchCancellation",
    "ModelSwitchBusyError",
    "ModelSwitchControlCapabilities",
    "ModelSwitchController",
    "ModelSwitchDisabledError",
    "ModelSwitchError",
    "ModelSwitchProductionUnavailableError",
    "ModelSwitchSnapshot",
    "ModelSwitchState",
    "ModelSwitchTransition",
    "RuntimeBindingEvidence",
    "RuntimeBundleIncompleteError",
    "RuntimeHealthEvidence",
    "RuntimePreparationCleanupEvidence",
    "RuntimePreparationDisposeEvidence",
    "RuntimePublicationFence",
    "RuntimeRequestLease",
    "RuntimeUnloadEvidence",
    "SafeModeEvidence",
    "StableResidentLeaseAuthority",
    "VerifiedModelSwitchDescriptor",
]
