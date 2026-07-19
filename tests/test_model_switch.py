from __future__ import annotations

from dataclasses import dataclass
from threading import Event, Thread
from time import monotonic

import pytest

from cogni_demo.model_switch import (
    ActiveRuntimeBundle,
    AtomicRuntimeSlot,
    LeaseReleaseEvidence,
    MemoryReleaseEvidence,
    ModelSwitchBusyError,
    ModelSwitchCancellation,
    ModelSwitchController,
    ModelSwitchDisabledError,
    ModelSwitchState,
    RuntimeBindingEvidence,
    RuntimeBundleIncompleteError,
    RuntimeHealthEvidence,
    RuntimeUnloadEvidence,
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
        acknowledge_unload: bool = True,
    ) -> None:
        self.descriptor = descriptor
        self.binding_id = binding_id
        self.authority = authority
        self.profile = profile
        self.fail_health = fail_health
        self.prove_release = prove_release
        self.raise_after_stop = raise_after_stop
        self.acknowledge_unload = acknowledge_unload
        self.on_health = None
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

    def stop(self, timeout_seconds: float) -> RuntimeUnloadEvidence:
        assert timeout_seconds > 0
        self.stop_count += 1
        lease = self._lease
        self._alive = False
        self._lease = None
        if lease is not None:
            self.authority.release(lease, prove=self.prove_release)
        if self.raise_after_stop:
            raise RuntimeError("injected post-retirement stop failure")
        assert lease is not None
        return RuntimeUnloadEvidence(
            binding_id=self.binding_id,
            model_authority_digest=self.descriptor.authority_digest,
            lease_epoch=lease.epoch,
            acknowledged=self.acknowledge_unload,
        )

    def healthcheck(self, timeout_seconds: float) -> RuntimeHealthEvidence:
        assert timeout_seconds > 0
        if self.fail_health:
            raise RuntimeError("injected health failure")
        assert self._lease is not None
        evidence = RuntimeHealthEvidence(
            ready=True,
            binding_id=self.binding_id,
            model_authority_digest=self.descriptor.authority_digest,
            lease_epoch=self._lease.epoch,
        )
        if self.on_health is not None:
            self.on_health()
        return evidence


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
        self.on_build = None

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
        if self.on_build is not None:
            self.on_build(bundle)
        return bundle


class _Maintenance:
    def __init__(self) -> None:
        self.calls: list[str] = []
        self.drained = True
        self.safe_error: str | None = None
        self.admission_open = True
        self.drain_started: Event | None = None
        self.drain_release: Event | None = None
        self.on_drain = None

    def close_admission(self, transaction_id: str) -> None:
        self.calls.append("close")
        self.admission_open = False

    def wait_for_drain(self, transaction_id: str, timeout_seconds: float) -> bool:
        self.calls.append("drain")
        if self.drain_started is not None:
            self.drain_started.set()
        if self.on_drain is not None:
            self.on_drain()
        if self.drain_release is not None:
            assert self.drain_release.wait(timeout=2.0)
        return self.drained

    def checkpoint(self, transaction_id, source, candidate) -> None:
        self.calls.append("checkpoint")

    def open_admission(self, transaction_id: str) -> None:
        self.calls.append("open")
        self.admission_open = True

    def enter_safe_mode(self, transaction_id: str, error_code: str) -> None:
        self.calls.append("safe")
        self.safe_error = error_code
        self.admission_open = False

    def try_admit(self) -> bool:
        return self.admission_open


class _MemoryProbe:
    def __init__(self) -> None:
        self.unreleased_models: set[str] = set()
        self.calls: list[str] = []
        self.on_verify = None

    def verify_release(
        self,
        bundle: ActiveRuntimeBundle,
        unload: RuntimeUnloadEvidence,
        timeout_seconds: float,
    ) -> MemoryReleaseEvidence:
        assert timeout_seconds > 0
        self.calls.append(bundle.descriptor.model_id)
        if self.on_verify is not None:
            self.on_verify(bundle)
        return MemoryReleaseEvidence(
            binding_id=bundle.binding_id,
            model_authority_digest=bundle.descriptor.authority_digest,
            lease_epoch=unload.lease_epoch,
            released=bundle.descriptor.model_id not in self.unreleased_models,
        )


class _Clock:
    def __init__(self) -> None:
        self.value = 100.0

    def __call__(self) -> float:
        return self.value

    def advance(self, seconds: float) -> None:
        self.value += seconds


def _system(*, enabled: bool = True, clock=monotonic):
    profile = StableResidentLeaseAuthority()
    authority = _LeaseAuthority(profile)
    source = _bundle(SOURCE, authority, profile, 1)
    source.runtime.start()
    slot = AtomicRuntimeSlot(source)
    factory = _Factory()
    maintenance = _Maintenance()
    memory_probe = _MemoryProbe()
    controller = ModelSwitchController(
        slot,
        factory,
        maintenance,
        enabled=enabled,
        memory_release_probe=memory_probe,
        clock=clock,
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
    assert maintenance.admission_open


def test_new_work_is_rejected_before_lease_drain_begins() -> None:
    controller, _slot, _source, _factory, maintenance, _authority = _system()
    observed: list[bool] = []
    maintenance.on_drain = lambda: observed.append(maintenance.try_admit())

    result = controller.switch(CANDIDATE)

    assert result.state is ModelSwitchState.SUCCEEDED
    assert observed == [False]
    assert maintenance.admission_open


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


def test_health_ack_cannot_publish_a_worker_that_died_before_commit() -> None:
    controller, slot, _source, factory, _maintenance, authority = _system()

    def configure(bundle: ActiveRuntimeBundle) -> None:
        if bundle.descriptor != CANDIDATE:
            return

        def retire_after_ack() -> None:
            lease = bundle.runtime.gpu_lease
            assert lease is not None
            bundle.runtime._alive = False
            bundle.runtime._lease = None
            authority.release(lease, prove=True)

        bundle.runtime.on_health = retire_after_ack

    factory.on_build = configure
    result = controller.switch(CANDIDATE)

    assert result.state is ModelSwitchState.SAFE_MODE
    assert result.error_code == "CANDIDATE_HEALTH_UNPROVEN"
    assert not result.rollback_restored
    assert slot.snapshot().bundle.descriptor == SOURCE
    assert slot.snapshot().bundle.descriptor != CANDIDATE


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


def test_enabled_controller_requires_an_injected_memory_release_probe() -> None:
    profile = StableResidentLeaseAuthority()
    authority = _LeaseAuthority(profile)
    source = _bundle(SOURCE, authority, profile, 1)
    source.runtime.start()

    with pytest.raises(ValueError, match="memory-release probe"):
        ModelSwitchController(
            AtomicRuntimeSlot(source), _Factory(), _Maintenance(), enabled=True
        )


def test_unload_acknowledgement_is_bound_to_exact_worker_and_lease() -> None:
    controller, slot, source, factory, maintenance, authority = _system()
    source.runtime.acknowledge_unload = False

    result = controller.switch(CANDIDATE)

    assert result.state is ModelSwitchState.SAFE_MODE
    assert result.error_code == "UNLOAD_ACK_UNPROVEN"
    assert not result.rollback_restored
    assert slot.snapshot().bundle is source
    assert not source.runtime.worker_alive
    assert authority.active is None
    assert factory.built[0].runtime.start_count == 0
    assert maintenance.safe_error == "UNLOAD_ACK_UNPROVEN"


def test_memory_release_postcondition_blocks_candidate_load_and_fails_closed() -> None:
    controller, slot, source, factory, maintenance, authority = _system()
    assert isinstance(controller.memory_release_probe, _MemoryProbe)
    controller.memory_release_probe.unreleased_models.add(SOURCE.model_id)

    result = controller.switch(CANDIDATE)

    assert result.state is ModelSwitchState.SAFE_MODE
    assert result.error_code == "SOURCE_MEMORY_RELEASE_UNPROVEN"
    assert slot.snapshot().bundle is source
    assert not source.runtime.worker_alive
    assert authority.active is None
    candidate = next(item for item in factory.built if item.descriptor == CANDIDATE)
    assert candidate.runtime.start_count == 0
    assert maintenance.safe_error == "SOURCE_MEMORY_RELEASE_UNPROVEN"


def test_drain_deadline_is_enforced_even_if_port_returns_true_late() -> None:
    clock = _Clock()
    controller, slot, source, _factory, maintenance, authority = _system(clock=clock)
    maintenance.on_drain = lambda: clock.advance(2.0)
    old_lease = source.runtime.gpu_lease

    result = controller.switch(CANDIDATE, drain_timeout_seconds=1.0)

    assert result.state is ModelSwitchState.ROLLED_BACK
    assert result.error_code == "DRAIN_TIMEOUT"
    assert slot.snapshot().bundle is source
    assert source.runtime.worker_alive
    assert source.runtime.gpu_lease == old_lease == authority.active
    assert maintenance.calls == ["close", "drain", "open"]


def test_cancellation_after_memory_release_restores_previous_model() -> None:
    controller, slot, source, _factory, maintenance, authority = _system()
    cancellation = ModelSwitchCancellation()
    assert isinstance(controller.memory_release_probe, _MemoryProbe)
    controller.memory_release_probe.on_verify = lambda bundle: (
        cancellation.cancel() if bundle.descriptor == SOURCE else None
    )

    result = controller.switch(CANDIDATE, cancellation=cancellation)

    assert result.state is ModelSwitchState.ROLLED_BACK
    assert result.error_code == "MODEL_SWITCH_CANCELLED"
    assert result.rollback_restored
    restored = slot.snapshot().bundle
    assert restored is not source
    assert restored.descriptor == SOURCE
    assert restored.runtime.worker_alive
    assert restored.runtime.gpu_lease == authority.active
    assert maintenance.calls[-1] == "open"


def test_single_flight_rejects_racing_switch_and_cancel_rolls_back() -> None:
    controller, slot, source, _factory, maintenance, authority = _system()
    cancellation = ModelSwitchCancellation()
    maintenance.drain_started = Event()
    maintenance.drain_release = Event()
    results: list[object] = []

    thread = Thread(
        target=lambda: results.append(
            controller.switch(CANDIDATE, cancellation=cancellation)
        ),
        daemon=True,
    )
    thread.start()
    assert maintenance.drain_started.wait(timeout=2.0)

    with pytest.raises(ModelSwitchBusyError):
        controller.switch(_descriptor("gemma4-e4b-racer", "7"))

    cancellation.cancel()
    maintenance.drain_release.set()
    thread.join(timeout=2.0)
    assert not thread.is_alive()
    assert len(results) == 1
    result = results[0]
    assert result.state is ModelSwitchState.ROLLED_BACK
    assert result.error_code == "MODEL_SWITCH_CANCELLED"
    assert slot.snapshot().bundle is source
    assert source.runtime.worker_alive
    assert source.runtime.gpu_lease == authority.active


def test_cancellation_and_commit_claim_have_one_atomic_winner() -> None:
    committed = ModelSwitchCancellation()
    assert committed.claim_commit()
    assert committed.cancel() is False
    assert not committed.requested

    cancelled = ModelSwitchCancellation()
    assert cancelled.cancel()
    assert cancelled.claim_commit() is False
    assert cancelled.requested


def test_candidate_cleanup_memory_failure_never_restarts_old_model() -> None:
    controller, slot, source, factory, maintenance, authority = _system()
    factory.fail_health_for.add(CANDIDATE.model_id)
    assert isinstance(controller.memory_release_probe, _MemoryProbe)
    controller.memory_release_probe.unreleased_models.add(CANDIDATE.model_id)

    result = controller.switch(CANDIDATE)

    assert result.state is ModelSwitchState.SAFE_MODE
    assert result.error_code == "CANDIDATE_HEALTHCHECK_FAILED"
    assert not result.rollback_restored
    assert slot.snapshot().bundle is source
    assert not source.runtime.worker_alive
    assert authority.active is None
    assert not any(item.descriptor == SOURCE for item in factory.built)
    assert maintenance.calls[-1] == "safe"


def test_structured_failure_snapshot_does_not_leak_host_path_or_secret() -> None:
    controller, _slot, _source, factory, _maintenance, _authority = _system()

    original_build = factory.build

    def build_with_secret(descriptor, lease_authority, lease_profile):
        bundle = original_build(descriptor, lease_authority, lease_profile)

        def fail_start() -> None:
            raise RuntimeError(r"C:\\private\\model TOKEN=do-not-leak")

        if descriptor == CANDIDATE:
            bundle.runtime.start = fail_start
        return bundle

    factory.build = build_with_secret

    result = controller.switch(CANDIDATE)
    public = repr(result)

    assert result.state is ModelSwitchState.ROLLED_BACK
    assert result.error_code == "CANDIDATE_START_FAILED"
    assert "private" not in public
    assert "do-not-leak" not in public


def test_pre_cancelled_switch_has_no_admission_or_worker_side_effects() -> None:
    controller, slot, source, _factory, maintenance, authority = _system()
    cancellation = ModelSwitchCancellation()
    cancellation.cancel()
    old_lease = source.runtime.gpu_lease

    result = controller.switch(CANDIDATE, cancellation=cancellation)

    assert result.state is ModelSwitchState.ROLLED_BACK
    assert result.error_code == "MODEL_SWITCH_CANCELLED"
    assert maintenance.calls == []
    assert slot.snapshot().bundle is source
    assert source.runtime.worker_alive
    assert source.runtime.gpu_lease == old_lease == authority.active


def test_factory_preflight_error_is_stable_and_redacted() -> None:
    controller, slot, source, factory, maintenance, authority = _system()

    def fail_build(*_args):
        raise RuntimeError(r"C:\\secret\\checkpoint API_KEY=never-publish")

    factory.build = fail_build

    with pytest.raises(RuntimeBundleIncompleteError) as captured:
        controller.switch(CANDIDATE)

    public = f"{captured.value.code}:{captured.value}"
    assert public == "RUNTIME_BUNDLE_INCOMPLETE:candidate runtime factory failed"
    assert "secret" not in public
    assert "never-publish" not in public
    assert controller.snapshot() is None
    assert maintenance.calls == []
    assert slot.snapshot().bundle is source
    assert source.runtime.worker_alive
    assert source.runtime.gpu_lease == authority.active
