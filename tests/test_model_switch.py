from __future__ import annotations

from dataclasses import dataclass
from time import monotonic

import pytest

from cogni_demo.model_switch import (
    ActiveRuntimeBundle,
    AtomicRuntimeSlot,
    LeaseReleaseEvidence,
    ModelSwitchController,
    ModelSwitchDisabledError,
    ModelSwitchState,
    RuntimeBindingEvidence,
    RuntimeBundleIncompleteError,
    RuntimeHealthEvidence,
    StableResidentLeaseAuthority,
    VerifiedModelSwitchDescriptor,
)
from cogni_os.gpu_lease import GPULease, GPULeaseManager


def _descriptor(name: str, digit: str) -> VerifiedModelSwitchDescriptor:
    return VerifiedModelSwitchDescriptor(
        model_id=name,
        manifest_sha256=digit * 64,
        config_sha256=chr(ord(digit) + 1) * 64,
        content_digest=chr(ord(digit) + 2) * 64,
    )


SOURCE = _descriptor("gemma4-e4b-source", "1")
CANDIDATE = _descriptor("gemma4-e4b-candidate", "4")


@dataclass
class _Binding:
    evidence: RuntimeBindingEvidence

    def binding_evidence(self) -> RuntimeBindingEvidence:
        return self.evidence


class _LeaseAuthority:
    def __init__(self, profile: StableResidentLeaseAuthority) -> None:
        self.manager = GPULeaseManager(max_vram_bytes=profile.vram_budget_bytes)
        self._release_evidence: dict[str, LeaseReleaseEvidence] = {}

    @property
    def active(self) -> GPULease | None:
        return self.manager.active

    @property
    def latest_epoch(self) -> int:
        return self.manager.latest_epoch

    @property
    def max_vram_bytes(self) -> int:
        return self.manager.max_vram_bytes

    def acquire(self, profile: StableResidentLeaseAuthority) -> GPULease:
        return self.manager.acquire(
            profile.owner,
            profile.purpose,
            profile.vram_budget_bytes,
            deadline=self.manager.deadline_after(60.0),
            owner_alive=lambda: True,
        )

    def release(self, lease: GPULease, *, prove: bool) -> None:
        self.manager.release(lease)
        if prove:
            self._release_evidence[lease.lease_id] = LeaseReleaseEvidence(
                lease=lease,
                reason="released",
                worker_death_confirmed=True,
            )

    def release_evidence(self, lease: GPULease) -> LeaseReleaseEvidence | None:
        return self._release_evidence.get(lease.lease_id)


class _Runtime:
    def __init__(
        self,
        descriptor: VerifiedModelSwitchDescriptor,
        binding_id: str,
        authority: _LeaseAuthority,
        profile: StableResidentLeaseAuthority,
        *,
        fail_health: bool = False,
        prove_release: bool = True,
        raise_after_stop: bool = False,
    ) -> None:
        self.descriptor = descriptor
        self.binding_id = binding_id
        self.authority = authority
        self.profile = profile
        self.fail_health = fail_health
        self.prove_release = prove_release
        self.raise_after_stop = raise_after_stop
        self._alive = False
        self._lease: GPULease | None = None
        self.start_count = 0
        self.stop_count = 0

    def binding_evidence(self) -> RuntimeBindingEvidence:
        return RuntimeBindingEvidence(
            "resident", self.binding_id, self.descriptor.authority_digest
        )

    @property
    def worker_alive(self) -> bool:
        return self._alive

    @property
    def gpu_lease(self) -> GPULease | None:
        return self._lease

    def start(self) -> None:
        self.start_count += 1
        self._lease = self.authority.acquire(self.profile)
        self._alive = True

    def stop(self, timeout_seconds: float) -> None:
        assert timeout_seconds > 0
        self.stop_count += 1
        lease = self._lease
        self._alive = False
        self._lease = None
        if lease is not None:
            self.authority.release(lease, prove=self.prove_release)
        if self.raise_after_stop:
            raise RuntimeError("injected post-retirement stop failure")

    def healthcheck(self, timeout_seconds: float) -> RuntimeHealthEvidence:
        assert timeout_seconds > 0
        if self.fail_health:
            raise RuntimeError("injected health failure")
        assert self._lease is not None
        return RuntimeHealthEvidence(
            ready=True,
            binding_id=self.binding_id,
            model_authority_digest=self.descriptor.authority_digest,
            lease_epoch=self._lease.epoch,
        )


def _bundle(
    descriptor: VerifiedModelSwitchDescriptor,
    authority: _LeaseAuthority,
    profile: StableResidentLeaseAuthority,
    index: int,
    *,
    fail_health: bool = False,
    prove_release: bool = True,
    incomplete_component: str | None = None,
) -> ActiveRuntimeBundle:
    binding_id = f"{index:032x}"
    runtime = _Runtime(
        descriptor,
        binding_id,
        authority,
        profile,
        fail_health=fail_health,
        prove_release=prove_release,
    )

    def binding(component: str) -> _Binding:
        actual = component
        if component == incomplete_component:
            actual = "voice" if component != "voice" else "factbook"
        return _Binding(
            RuntimeBindingEvidence(actual, binding_id, descriptor.authority_digest)
        )

    return ActiveRuntimeBundle(
        descriptor=descriptor,
        binding_id=binding_id,
        lease_profile=profile,
        lease_authority=authority,
        runtime=runtime,
        factbook=binding("factbook"),
        validator=binding("validator"),
        voice=binding("voice"),
        harness=binding("harness"),
    )


class _Factory:
    def __init__(self) -> None:
        self.count = 1
        self.fail_health_for: set[str] = set()
        self.unproven_release_for: set[str] = set()
        self.incomplete_for: dict[str, str] = {}
        self.built: list[ActiveRuntimeBundle] = []

    def build(self, descriptor, lease_authority, lease_profile):
        self.count += 1
        bundle = _bundle(
            descriptor,
            lease_authority,
            lease_profile,
            self.count,
            fail_health=descriptor.model_id in self.fail_health_for,
            prove_release=descriptor.model_id not in self.unproven_release_for,
            incomplete_component=self.incomplete_for.get(descriptor.model_id),
        )
        self.built.append(bundle)
        return bundle


class _Maintenance:
    def __init__(self) -> None:
        self.calls: list[str] = []
        self.drained = True
        self.safe_error: str | None = None

    def close_admission(self, transaction_id: str) -> None:
        self.calls.append("close")

    def wait_for_drain(self, transaction_id: str, timeout_seconds: float) -> bool:
        self.calls.append("drain")
        return self.drained

    def checkpoint(self, transaction_id, source, candidate) -> None:
        self.calls.append("checkpoint")

    def open_admission(self, transaction_id: str) -> None:
        self.calls.append("open")

    def enter_safe_mode(self, transaction_id: str, error_code: str) -> None:
        self.calls.append("safe")
        self.safe_error = error_code


def _system(*, enabled: bool = True):
    profile = StableResidentLeaseAuthority()
    authority = _LeaseAuthority(profile)
    source = _bundle(SOURCE, authority, profile, 1)
    source.runtime.start()
    slot = AtomicRuntimeSlot(source)
    factory = _Factory()
    maintenance = _Maintenance()
    controller = ModelSwitchController(
        slot, factory, maintenance, enabled=enabled, clock=monotonic
    )
    return controller, slot, source, factory, maintenance, authority


def test_switching_is_disabled_by_default_without_side_effects() -> None:
    controller, slot, source, factory, maintenance, authority = _system(enabled=False)

    with pytest.raises(ModelSwitchDisabledError):
        controller.switch(CANDIDATE)

    assert slot.snapshot().bundle is source
    assert source.runtime.worker_alive
    assert authority.active == source.runtime.gpu_lease
    assert not factory.built
    assert not maintenance.calls


@pytest.mark.parametrize("component", ["factbook", "validator", "voice", "harness"])
def test_incomplete_candidate_bundle_is_rejected_before_drain_or_unload(
    component: str,
) -> None:
    controller, slot, source, factory, maintenance, authority = _system()
    factory.incomplete_for[CANDIDATE.model_id] = component

    with pytest.raises(RuntimeBundleIncompleteError):
        controller.switch(CANDIDATE)

    assert controller.snapshot() is None
    assert slot.snapshot().bundle is source
    assert source.runtime.worker_alive
    assert authority.active == source.runtime.gpu_lease
    assert not maintenance.calls


def test_successful_switch_proves_sequential_leases_and_atomic_bundle_commit() -> None:
    controller, slot, source, factory, maintenance, authority = _system()
    old_lease = source.runtime.gpu_lease
    assert old_lease is not None

    result = controller.switch(CANDIDATE)

    assert result.state is ModelSwitchState.SUCCEEDED
    assert result.error_code is None
    assert [item.target for item in result.transitions] == [
        ModelSwitchState.REQUESTED,
        ModelSwitchState.DRAINING,
        ModelSwitchState.CHECKPOINTING,
        ModelSwitchState.UNLOADING_OLD,
        ModelSwitchState.LOADING_CANDIDATE,
        ModelSwitchState.HEALTHCHECKING,
        ModelSwitchState.COMMITTING,
        ModelSwitchState.SUCCEEDED,
    ]
    active = slot.snapshot()
    assert active.generation == 1
    assert active.bundle.descriptor == CANDIDATE
    assert active.bundle.runtime.worker_alive
    assert active.bundle.runtime.gpu_lease == authority.active
    assert active.bundle.runtime.gpu_lease.epoch > old_lease.epoch
    assert active.bundle.runtime.gpu_lease.purpose == "resident-model"
    assert authority.release_evidence(old_lease).worker_death_confirmed
    assert maintenance.calls == ["close", "drain", "checkpoint", "open"]


def test_candidate_health_failure_retires_candidate_and_restores_old_bundle() -> None:
    controller, slot, source, factory, maintenance, authority = _system()
    factory.fail_health_for.add(CANDIDATE.model_id)

    result = controller.switch(CANDIDATE)

    assert result.state is ModelSwitchState.ROLLED_BACK
    assert result.error_code == "CANDIDATE_HEALTHCHECK_FAILED"
    assert result.rollback_restored
    active = slot.snapshot().bundle
    assert active.descriptor == SOURCE
    assert active is not source
    assert active.runtime.worker_alive
    assert active.runtime.gpu_lease == authority.active
    candidate_bundle = next(
        item for item in factory.built if item.descriptor == CANDIDATE
    )
    assert not candidate_bundle.runtime.worker_alive
    assert candidate_bundle.runtime.gpu_lease is None
    assert ModelSwitchState.ROLLING_BACK in [item.target for item in result.transitions]
    assert ModelSwitchState.RESTORING_OLD in [
        item.target for item in result.transitions
    ]


def test_unproven_candidate_lease_release_enters_safe_mode_without_old_restart() -> (
    None
):
    controller, slot, source, factory, maintenance, authority = _system()
    factory.fail_health_for.add(CANDIDATE.model_id)
    factory.unproven_release_for.add(CANDIDATE.model_id)

    result = controller.switch(CANDIDATE)

    assert result.state is ModelSwitchState.SAFE_MODE
    assert result.error_code == "CANDIDATE_HEALTHCHECK_FAILED"
    assert not result.rollback_restored
    assert maintenance.safe_error == "CANDIDATE_HEALTHCHECK_FAILED"
    assert slot.snapshot().bundle is source
    assert not source.runtime.worker_alive
    assert authority.active is None
    assert [item.descriptor for item in factory.built].count(SOURCE) == 0


def test_unproven_old_lease_release_enters_safe_mode_before_candidate_start() -> None:
    controller, slot, source, factory, maintenance, authority = _system()
    source.runtime.prove_release = False

    result = controller.switch(CANDIDATE)

    assert result.state is ModelSwitchState.SAFE_MODE
    assert result.error_code == "LEASE_RELEASE_UNPROVEN"
    assert ModelSwitchState.ROLLING_BACK in [item.target for item in result.transitions]
    assert slot.snapshot().bundle is source
    assert not source.runtime.worker_alive
    assert authority.active is None
    candidate_bundle = next(
        item for item in factory.built if item.descriptor == CANDIDATE
    )
    assert candidate_bundle.runtime.start_count == 0
    assert maintenance.safe_error == "LEASE_RELEASE_UNPROVEN"


def test_old_stop_exception_after_proven_death_restores_old_instead_of_reopening_dead_bundle() -> (
    None
):
    controller, slot, source, _factory, _maintenance, authority = _system()
    source.runtime.raise_after_stop = True

    result = controller.switch(CANDIDATE)

    assert result.state is ModelSwitchState.ROLLED_BACK
    assert result.error_code == "OLD_UNLOAD_FAILED"
    assert result.rollback_restored
    restored = slot.snapshot().bundle
    assert restored is not source
    assert restored.descriptor == SOURCE
    assert restored.runtime.worker_alive
    assert restored.runtime.gpu_lease == authority.active


def test_failed_old_restore_is_retired_before_safe_mode() -> None:
    controller, slot, source, factory, maintenance, authority = _system()
    factory.fail_health_for.update({CANDIDATE.model_id, SOURCE.model_id})

    result = controller.switch(CANDIDATE)

    assert result.state is ModelSwitchState.SAFE_MODE
    assert result.error_code == "CANDIDATE_HEALTHCHECK_FAILED"
    assert slot.snapshot().bundle is source
    assert authority.active is None
    restored = [item for item in factory.built if item.descriptor == SOURCE]
    assert len(restored) == 1
    assert not restored[0].runtime.worker_alive
    assert restored[0].runtime.gpu_lease is None
    assert maintenance.safe_error == "CANDIDATE_HEALTHCHECK_FAILED"


def test_drain_timeout_rolls_back_without_touching_the_old_worker() -> None:
    controller, slot, source, _factory, maintenance, authority = _system()
    maintenance.drained = False
    lease = source.runtime.gpu_lease

    result = controller.switch(CANDIDATE)

    assert result.state is ModelSwitchState.ROLLED_BACK
    assert result.error_code == "DRAIN_TIMEOUT"
    assert not result.rollback_restored
    assert slot.snapshot().bundle is source
    assert source.runtime.worker_alive
    assert source.runtime.gpu_lease == lease == authority.active
    assert maintenance.calls == ["close", "drain", "open"]


def test_stable_resident_lease_rejects_mode_dependent_purpose() -> None:
    with pytest.raises(ValueError, match="must not depend"):
        StableResidentLeaseAuthority(purpose="inference")
