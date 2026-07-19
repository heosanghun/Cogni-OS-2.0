from __future__ import annotations

from dataclasses import dataclass
from threading import Event, Thread
from time import monotonic

import pytest

from cogni_demo.model_switch import (
    ActiveRuntimeBundle,
    AdmissionFenceToken,
    AdmissionOpenEvidence,
    AtomicRuntimeCommitError,
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
    RuntimePublicationFence,
    RuntimeUnloadEvidence,
    SafeModeEvidence,
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
    def __init__(
        self, profile: StableResidentLeaseAuthority, *, clock=monotonic
    ) -> None:
        self.manager = GPULeaseManager(
            max_vram_bytes=profile.vram_budget_bytes, clock=clock
        )
        self._release_evidence: dict[str, LeaseReleaseEvidence] = {}
        self.on_validate = None

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

    def validate(
        self,
        lease: GPULease,
        *,
        purpose: str | None = None,
        required_vram_bytes: int | None = None,
    ) -> GPULease:
        if self.on_validate is not None:
            self.on_validate(lease)
        return self.manager.validate(
            lease,
            purpose=purpose,
            required_vram_bytes=required_vram_bytes,
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
        self._generation = 0
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
    def worker_generation(self) -> int:
        return self._generation

    @property
    def gpu_lease(self) -> GPULease | None:
        return self._lease

    def start(self) -> None:
        self.start_count += 1
        self._lease = self.authority.acquire(self.profile)
        self._generation += 1
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
            worker_generation=self._generation,
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
            worker_generation=self._generation,
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
        self.on_open = None
        self.prove_safe = True
        self.admission_token: AdmissionFenceToken | None = None

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

    def open_admission(
        self, transaction_id: str, token: AdmissionFenceToken
    ) -> AdmissionOpenEvidence:
        self.calls.append("open")
        if self.on_open is not None:
            self.on_open(token)
        self.admission_token = token
        self.admission_open = True
        return AdmissionOpenEvidence(token=token, opened=True)

    def enter_safe_mode(self, transaction_id: str, error_code: str) -> SafeModeEvidence:
        self.calls.append("safe")
        self.safe_error = error_code
        self.admission_open = not self.prove_safe
        return SafeModeEvidence(
            transaction_id=transaction_id,
            error_code=error_code,
            admission_closed=self.prove_safe,
        )

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
            worker_generation=unload.worker_generation,
            released=bundle.descriptor.model_id not in self.unreleased_models,
        )


class _Clock:
    def __init__(self) -> None:
        self.value = 100.0

    def __call__(self) -> float:
        return self.value

    def advance(self, seconds: float) -> None:
        self.value += seconds


def _system(*, enabled: bool = True, clock=monotonic, lease_clock=monotonic):
    profile = StableResidentLeaseAuthority()
    authority = _LeaseAuthority(profile, clock=lease_clock)
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
    controller, slot, source, factory, maintenance, authority = _system()
    cancellation = ModelSwitchCancellation()
    cancellation.cancel()
    old_lease = source.runtime.gpu_lease

    result = controller.switch(CANDIDATE, cancellation=cancellation)

    assert result.state is ModelSwitchState.ROLLED_BACK
    assert result.error_code == "MODEL_SWITCH_CANCELLED"
    assert maintenance.calls == []
    assert factory.built == []
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


def test_expired_candidate_lease_is_rejected_before_publication() -> None:
    lease_clock = _Clock()
    controller, slot, source, factory, maintenance, authority = _system(
        lease_clock=lease_clock
    )

    def configure(bundle: ActiveRuntimeBundle) -> None:
        if bundle.descriptor == CANDIDATE:
            bundle.runtime.on_health = lambda: lease_clock.advance(61.0)

    factory.on_build = configure
    result = controller.switch(CANDIDATE)

    assert result.state is ModelSwitchState.ROLLED_BACK
    assert result.error_code == "CANDIDATE_HEALTH_UNPROVEN"
    assert result.rollback_restored
    assert slot.snapshot().bundle.descriptor == SOURCE
    assert slot.snapshot().bundle.runtime.gpu_lease == authority.active
    assert maintenance.admission_open


def test_atomic_slot_rejects_stopped_replacement_even_with_forged_fence() -> None:
    _controller, slot, source, _factory, _maintenance, authority = _system()
    replacement = _bundle(CANDIDATE, authority, source.lease_profile, 20)
    source_lease = source.runtime.gpu_lease
    assert source_lease is not None
    forged = RuntimePublicationFence(
        expected_slot_generation=slot.snapshot().generation,
        binding_id=replacement.binding_id,
        model_authority_digest=replacement.descriptor.authority_digest,
        worker_generation=source.runtime.worker_generation,
        lease_id=source_lease.lease_id,
        lease_epoch=source_lease.epoch,
    )

    with pytest.raises(AtomicRuntimeCommitError):
        slot.compare_and_swap(slot.snapshot(), replacement, forged)

    assert slot.snapshot().bundle is source


def test_cas_postcheck_restores_slot_if_worker_dies_inside_publication() -> None:
    controller, slot, source, factory, _maintenance, authority = _system()

    def configure(bundle: ActiveRuntimeBundle) -> None:
        if bundle.descriptor != CANDIDATE:
            return

        def arm_publication_failure() -> None:
            validations = 0

            def fail_on_cas_postcheck(_lease: GPULease) -> None:
                nonlocal validations
                validations += 1
                if validations != 4:
                    return
                current = bundle.runtime.gpu_lease
                assert current is not None
                bundle.runtime._alive = False
                bundle.runtime._lease = None
                authority.release(current, prove=True)

            authority.on_validate = fail_on_cas_postcheck

        bundle.runtime.on_health = arm_publication_failure

    factory.on_build = configure
    result = controller.switch(CANDIDATE)

    assert result.state is ModelSwitchState.SAFE_MODE
    assert result.error_code == "ATOMIC_RUNTIME_COMMIT_FAILED"
    assert slot.snapshot().bundle is source
    assert slot.snapshot().bundle.descriptor != CANDIDATE


def test_admission_open_revalidates_candidate_after_ack() -> None:
    controller, slot, _source, factory, maintenance, authority = _system()

    def kill_acknowledged_worker(_token: AdmissionFenceToken) -> None:
        active = slot.snapshot().bundle
        if active.descriptor != CANDIDATE:
            return
        lease = active.runtime.gpu_lease
        assert lease is not None
        active.runtime._alive = False
        active.runtime._lease = None
        authority.release(lease, prove=True)

    maintenance.on_open = kill_acknowledged_worker
    result = controller.switch(CANDIDATE)

    assert result.state is ModelSwitchState.SAFE_MODE
    assert result.error_code == "PUBLICATION_FENCE_UNPROVEN"
    assert not maintenance.admission_open
    assert slot.snapshot().bundle is factory.built[0]
    assert not slot.snapshot().bundle.runtime.worker_alive


def test_admission_ack_must_echo_the_exact_publication_token() -> None:
    controller, slot, _source, _factory, maintenance, _authority = _system()
    real_open = maintenance.open_admission
    forged_once = False

    def open_with_one_forged_ack(
        transaction_id: str, token: AdmissionFenceToken
    ) -> AdmissionOpenEvidence:
        nonlocal forged_once
        if forged_once:
            return real_open(transaction_id, token)
        forged_once = True
        maintenance.calls.append("open")
        maintenance.admission_open = True
        return AdmissionOpenEvidence(
            token=AdmissionFenceToken(transaction_id + "-forged", token.publication),
            opened=True,
        )

    maintenance.open_admission = open_with_one_forged_ack
    result = controller.switch(CANDIDATE)

    assert result.state is ModelSwitchState.ROLLED_BACK
    assert result.error_code == "ADMISSION_OPEN_FENCE_UNPROVEN"
    assert result.rollback_restored
    assert maintenance.admission_open
    assert slot.snapshot().bundle.descriptor == SOURCE


def test_source_rollback_open_uses_same_post_ack_publication_fence() -> None:
    controller, slot, source, _factory, maintenance, authority = _system()
    maintenance.drained = False

    def kill_source(_token: AdmissionFenceToken) -> None:
        lease = source.runtime.gpu_lease
        assert lease is not None
        source.runtime._alive = False
        source.runtime._lease = None
        authority.release(lease, prove=True)

    maintenance.on_open = kill_source
    result = controller.switch(CANDIDATE)

    assert result.state is ModelSwitchState.SAFE_MODE
    assert result.error_code == "DRAIN_TIMEOUT"
    assert result.safety_error_code == "PUBLICATION_FENCE_UNPROVEN"
    assert not maintenance.admission_open
    assert slot.snapshot().bundle is source
    assert not source.runtime.worker_alive


def test_unproved_safe_mode_ack_is_exposed_as_distinct_terminal_state() -> None:
    controller, _slot, source, _factory, maintenance, _authority = _system()
    source.runtime.prove_release = False
    maintenance.prove_safe = False

    result = controller.switch(CANDIDATE)

    assert result.state is ModelSwitchState.SAFE_MODE_UNPROVEN
    assert result.error_code == "LEASE_RELEASE_UNPROVEN"
    assert result.safety_error_code == "SAFE_MODE_UNPROVEN"
    assert maintenance.admission_open


def test_live_invalid_factory_result_is_stopped_before_rejection() -> None:
    controller, slot, source, factory, maintenance, authority = _system()

    def violate_factory_contract(bundle: ActiveRuntimeBundle) -> None:
        if bundle.descriptor == CANDIDATE:
            bundle.runtime._alive = True

    factory.on_build = violate_factory_contract

    with pytest.raises(RuntimeBundleIncompleteError):
        controller.switch(CANDIDATE)

    invalid = factory.built[0]
    assert invalid.runtime.stop_count == 1
    assert not invalid.runtime.worker_alive
    assert invalid.runtime.gpu_lease is None
    assert maintenance.calls == []
    assert slot.snapshot().bundle is source
    assert source.runtime.gpu_lease == authority.active


def test_rollback_health_deadline_and_cleanup_code_are_preserved() -> None:
    clock = _Clock()
    controller, _slot, _source, factory, _maintenance, _authority = _system(clock=clock)
    factory.fail_health_for.add(CANDIDATE.model_id)

    def configure(bundle: ActiveRuntimeBundle) -> None:
        if bundle.descriptor == SOURCE:
            bundle.runtime.on_health = lambda: clock.advance(2.0)

    factory.on_build = configure
    result = controller.switch(CANDIDATE, health_timeout_seconds=1.0)

    assert result.state is ModelSwitchState.SAFE_MODE
    assert result.error_code == "CANDIDATE_HEALTHCHECK_FAILED"
    assert result.safety_error_code == "ROLLBACK_HEALTHCHECK_TIMEOUT"
