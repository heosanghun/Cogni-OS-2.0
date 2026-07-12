from dataclasses import dataclass
import os
from pathlib import Path
from queue import Queue
from threading import Event, Thread
from time import monotonic, monotonic_ns, sleep
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from torch import nn

from cogni_agent.model_service import (
    BaseModelMutationError,
    GenerationCancelled,
    LocalGemmaCorePipelineFactory,
    LocalGemmaModelFactory,
    ModelService,
    PRODUCTION_CTS_ACT_HARD_FLOOR,
    RequestLimitError,
    WorkerExecutionError,
    WorkerAuthorityError,
    WorkerStartupError,
    _TokenRepetitionGuard,
    _VerifiedMetaControllerV2,
    _parameter_signature,
    _prepare_base_model,
    _required_product_logits_processor,
    _worker_authority_tensor,
    _worker_main,
    load_local_tokenizer,
    truncate_repeated_tokens,
)
from cogni_agent.core_pipeline import CoreTurnRequest, CoreTurnResult
from cogni_agent.protocol import (
    DIGEST_BYTES,
    FINISH_CANCELLED,
    FINISH_ERROR,
    FINISH_LENGTH,
    STATUS_CANCELLED,
    STATUS_AUTHORITY_REJECTED,
    STATUS_MODEL_ERROR,
    STATUS_OK,
    make_generate_request,
    make_response,
    parse_response,
)
from cogni_core.cts_policy import (
    ACTION_LOGIT_BOUND,
    CTSCheckpointError,
    LearnedCTSController,
)
from cogni_core.search import (
    BoundedPUCTSearchV2,
    CertifiedBroydenTransitionV2,
)
from cogni_os.gpu_lease import (
    ExpiredGPULeaseError,
    GPULeaseBudgetError,
    GPULeaseManager,
    StaleGPULeaseError,
)
from cogni_os.runtime import SearchCollaboratorsV2


class FakeTokenizer:
    def __call__(self, text, **_kwargs):
        token_ids = [(ord(character) % 50) + 1 for character in text]
        return {
            "input_ids": torch.tensor([token_ids], dtype=torch.int64),
            "attention_mask": torch.ones((1, len(token_ids)), dtype=torch.int64),
        }

    def decode(self, token_ids, **_kwargs):
        return " ".join(str(token) for token in token_ids)


class FakeConfig:
    use_cache = True


class SafeFakeModel(nn.Module):
    def __init__(
        self,
        *,
        delay: float = 0.0,
        mutate: bool = False,
        require_core: bool = False,
        require_conditioning: bool = False,
        output_head: bool = True,
        data_mutate: bool = False,
    ):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1))
        self.config = FakeConfig()
        self.delay = delay
        self.mutate = mutate
        self.require_core = require_core
        self.require_conditioning = require_conditioning
        self.data_mutate = data_mutate
        self.core_ran = False
        self.register_buffer("bounded_state", torch.tensor([3.0]))
        self.lm_head = nn.Linear(4, 128, bias=False) if output_head else None
        if self.lm_head is not None:
            with torch.no_grad():
                rows = torch.linspace(-1.0, 1.0, 128).unsqueeze(1)
                direction = torch.tensor([[1.0, -0.5, 0.25, 0.75]])
                self.lm_head.weight.copy_(rows * direction)

    def get_output_embeddings(self):
        return self.lm_head

    def generate(self, **kwargs):
        expected_decode = {
            "do_sample": False,
            "num_beams": 1,
            "num_return_sequences": 1,
            "use_cache": False,
        }
        if any(kwargs.get(name) != value for name, value in expected_decode.items()):
            raise RuntimeError("unsafe generation options")
        if any(
            name in kwargs
            for name in (
                "temperature",
                "top_p",
                "top_k",
                "repetition_penalty",
                "no_repeat_ngram_size",
            )
        ):
            raise RuntimeError("prompt-wide logits policies break exact-copy requests")
        if "inputs_embeds" in kwargs:
            raise RuntimeError("production generation must use bounded input_ids")
        if self.training or self.weight.requires_grad or self.config.use_cache:
            raise RuntimeError("base model was not frozen for inference")
        if self.require_core and not self.core_ran:
            raise RuntimeError("base generate ran before the Cogni-Core pipeline")
        input_ids = kwargs["input_ids"]
        processors = kwargs.get("logits_processor")
        conditioned_token = None
        if self.require_conditioning:
            if not isinstance(processors, list) or len(processors) != 1:
                raise RuntimeError("production logits processor was not supplied")
            if self.lm_head is None:
                raise RuntimeError("conditioning test model has no output head")
            scores = torch.zeros(
                (1, self.lm_head.out_features),
                device=input_ids.device,
                dtype=self.lm_head.weight.dtype,
            )
            conditioned = processors[0](input_ids, scores)
            if torch.equal(conditioned, scores):
                raise RuntimeError("production logits processor had no causal effect")
            if float((conditioned - scores).abs().max()) > 0.100001:
                raise RuntimeError("production logits processor exceeded its bound")
            conditioned_token = int(conditioned.argmax(dim=-1).item())
        elif "logits_processor" in kwargs:
            raise RuntimeError("an arbitrary advisory pipeline altered base logits")
        streamer = kwargs["streamer"]
        criteria = kwargs["stopping_criteria"]
        streamer.put(input_ids)
        output = input_ids.clone()
        if self.mutate:
            self.weight.add_(1.0)
        if self.data_mutate:
            self.weight.data.add_(1.0)
        generated_tokens = (
            (conditioned_token, 102, 103, 104)
            if conditioned_token is not None
            else (int(getattr(self, "session_token", 101)), 102, 103, 104)
        )
        for token in generated_tokens[: kwargs["max_new_tokens"]]:
            if self.delay:
                sleep(self.delay)
            if any(bool(stop(output, None).all()) for stop in criteria):
                break
            value = torch.tensor([[token]], dtype=torch.int64, device=output.device)
            output = torch.cat((output, value), dim=1)
            streamer.put(value)
        streamer.end()
        return output


@dataclass(frozen=True)
class FakeFactory:
    parent_pid: int
    delay: float = 0.0
    mutate: bool = False
    require_core: bool = False
    require_conditioning: bool = False
    output_head: bool = True
    data_mutate: bool = False

    def __call__(self):
        if os.getpid() == self.parent_pid:
            raise RuntimeError("model must never load in the controller process")
        return SafeFakeModel(
            delay=self.delay,
            mutate=self.mutate,
            require_core=self.require_core,
            require_conditioning=self.require_conditioning,
            output_head=self.output_head,
            data_mutate=self.data_mutate,
        )


class AdvisoryCorePipeline:
    def __init__(self, model: SafeFakeModel, *, fail: bool = False):
        self.model = model
        self.fail = fail

    def run(self, request):
        if self.fail:
            raise RuntimeError("core failed")
        if not isinstance(request, CoreTurnRequest):
            raise TypeError("worker did not create a CoreTurnRequest")
        if request.fast_weight is not None or request.compile_fast_weight is not None:
            raise RuntimeError("unverified Fast Weight crossed the gate")
        if request.cognitive_state.shape != (1, 5):
            raise RuntimeError("BIO-HAMA state was not assembled")
        if request.inputs.device != request.cognitive_state.device:
            raise RuntimeError("core tensors crossed devices")
        if request.backbone_kwargs["attention_mask"].shape != request.inputs.shape:
            raise RuntimeError("attention tensor was not forwarded")
        self.model.core_ran = True
        # Deliberately return an absurd advisory state. The answer test below
        # proves it never replaces base Gemma generation.
        return SimpleNamespace(
            telemetry=SimpleNamespace(advisory_only=True),
            advisory_state=torch.tensor([[-999.0]], device=request.inputs.device),
        )


@dataclass(frozen=True)
class AdvisoryCoreFactory:
    fail: bool = False

    def __call__(self, model):
        return AdvisoryCorePipeline(model, fail=self.fail)


@dataclass(frozen=True)
class CheckpointFailingCoreFactory:
    """Picklable worker fixture for a content-addressed policy failure."""

    def __call__(self, _model):
        raise CTSCheckpointError("checkpoint SHA-256 verification failed")


class SessionAwareCorePipeline(AdvisoryCorePipeline):
    def run(self, request):
        result = super().run(request)
        # The IPC session is a 32-byte digest rendered as a 64-character key.
        # A -> B -> A must therefore produce A -> B -> A worker-visible state.
        self.model.session_token = 200 + int(request.swarm_session_id, 16) % 10_000
        return result


@dataclass(frozen=True)
class SessionAwareCoreFactory:
    def __call__(self, model):
        return SessionAwareCorePipeline(model)


class ProductionCorePipeline(AdvisoryCorePipeline):
    def __init__(self, model, *, result_mode: str = "valid"):
        super().__init__(model)
        self.result_mode = result_mode

    def run(self, request):
        advisory = super().run(request)
        if self.result_mode == "mutate_missing":
            with torch.no_grad():
                self.model.weight.add_(1.0)
            return advisory
        if self.result_mode == "unsampled_mutate_missing":
            # The runtime fingerprint samples 16 evenly spaced values. Index 1
            # in this 512-value tensor is deliberately outside that sample and
            # ``.data`` also bypasses Tensor._version. Core failure must still
            # publish no base-model answer.
            self.model.lm_head.weight.data.view(-1)[1].add_(12_345.0)
            return advisory
        if self.result_mode == "missing":
            return advisory
        best_state = torch.tensor(
            [[0.5, -0.25, 0.75, 1.0]],
            device=request.inputs.device,
            dtype=self.model.weight.dtype,
        )
        if self.result_mode == "bad_shape":
            best_state = best_state.squeeze(0)
        elif self.result_mode == "negated":
            best_state = -best_state
        return CoreTurnResult(
            inference=SimpleNamespace(
                search=SimpleNamespace(best_state=best_state),
            ),
            pooled_observation=best_state,
            telemetry=advisory.telemetry,
        )


@dataclass(frozen=True)
class ProductionCoreFactory(LocalGemmaCorePipelineFactory):
    result_mode: str = "valid"

    def __call__(self, model):
        return ProductionCorePipeline(model, result_mode=self.result_mode)


class RecordingTokenizerClass:
    path = None
    options = None

    @classmethod
    def from_pretrained(cls, path, **options):
        cls.path = path
        cls.options = options
        return FakeTokenizer()


class _AliveProcess:
    @staticmethod
    def is_alive():
        return True


class _ShutdownProcess:
    def __init__(self, events, *, exits_after):
        self.events = events
        self.exits_after = exits_after
        self.stage = "graceful"
        self.alive = True

    def is_alive(self):
        return self.alive

    def join(self, timeout):
        self.events.append(("join", timeout))
        if self.exits_after == self.stage:
            self.alive = False

    def terminate(self):
        self.events.append("terminate")
        self.stage = "terminate"

    def kill(self):
        self.events.append("kill")
        self.stage = "kill"


class _ShutdownQueue:
    def __init__(self, name, events):
        self.name = name
        self.events = events

    def put(self, _message, timeout):
        self.events.append((f"{self.name}.put", timeout))

    def close(self):
        self.events.append(f"{self.name}.close")

    def join_thread(self):
        self.events.append(f"{self.name}.join_thread")


class _LeaseClock:
    def __init__(self, value=100.0):
        self.value = value

    def __call__(self):
        return self.value

    def advance(self, seconds):
        self.value += seconds


class _RecordingLeaseManager(GPULeaseManager):
    def __init__(self, events, *, clock=monotonic):
        super().__init__(clock=clock)
        self.events = events

    def acquire(self, *args, **kwargs):
        self.events.append("lease.acquire")
        lease = super().acquire(*args, **kwargs)
        health = kwargs.get("owner_alive")
        self.events.append(("lease.pre_spawn_alive", bool(health())))
        return lease

    def validate(self, *args, **kwargs):
        self.events.append("lease.validate")
        return super().validate(*args, **kwargs)

    def release(self, *args, **kwargs):
        self.events.append("lease.release")
        return super().release(*args, **kwargs)


class _LeaseProcess:
    def __init__(
        self,
        events,
        ready_event,
        index,
        *,
        fail_start=False,
        auto_ready=True,
        dies_after_ready=False,
    ):
        self.events = events
        self.ready_event = ready_event
        self.index = index
        self.fail_start = fail_start
        self.auto_ready = auto_ready
        self.dies_after_ready = dies_after_ready
        self.alive = False
        self.pid = 10_000 + index

    def is_alive(self):
        return self.alive

    def start(self):
        self.events.append(("process.start", self.index))
        if self.fail_start:
            raise OSError("synthetic spawn failure")
        self.alive = True
        if self.auto_ready:
            self.ready_event.set()
        if self.dies_after_ready:
            self.alive = False

    def join(self, timeout):
        self.events.append(("process.join", self.index, timeout))
        self.alive = False

    def terminate(self):
        self.events.append(("process.terminate", self.index))
        self.alive = False

    def kill(self):
        self.events.append(("process.kill", self.index))
        self.alive = False


class _LeaseContext:
    def __init__(
        self,
        events,
        *,
        failed_starts=(),
        auto_ready=True,
        dies_after_ready=False,
    ):
        self.events = events
        self.failed_starts = set(failed_starts)
        self.auto_ready = auto_ready
        self.dies_after_ready = dies_after_ready
        self.queue_count = 0
        self.processes = []

    def Queue(self, *, maxsize):
        del maxsize
        self.queue_count += 1
        return _ShutdownQueue(f"queue-{self.queue_count}", self.events)

    @staticmethod
    def Event():
        return Event()

    def Process(self, *, target, args, name, daemon):
        del target, name, daemon
        index = len(self.processes) + 1
        process = _LeaseProcess(
            self.events,
            args[5],
            index,
            fail_start=index in self.failed_starts,
            auto_ready=self.auto_ready,
            dies_after_ready=self.dies_after_ready,
        )
        self.processes.append(process)
        self.events.append(("process.construct", index))
        return process


def service_for(
    *,
    delay=0.0,
    mutate=False,
    require_core=False,
    require_conditioning=False,
    output_head=True,
    data_mutate=False,
    core_pipeline_factory=None,
    **kwargs,
):
    return ModelService(
        FakeTokenizer(),
        FakeFactory(
            os.getpid(),
            delay=delay,
            mutate=mutate,
            require_core=require_core,
            require_conditioning=require_conditioning,
            output_head=output_head,
            data_mutate=data_mutate,
        ),
        core_pipeline_factory=core_pipeline_factory,
        max_input_tokens=32,
        max_new_tokens=8,
        startup_timeout=15,
        request_timeout=10,
        cancellation_timeout=2,
        **kwargs,
    )


class TestResidentModelService(unittest.TestCase):
    def test_checkpoint_integrity_failure_is_reported_from_spawned_worker(self):
        service = service_for(core_pipeline_factory=CheckpointFailingCoreFactory())

        with self.assertRaisesRegex(
            WorkerStartupError,
            "CTS policy checkpoint integrity verification failed",
        ):
            service.start()

        self.assertFalse(service.is_running)
        self.assertIsNone(service._process)

    def test_two_start_callers_wait_for_one_shared_ready_attempt(self):
        events = []
        context = _LeaseContext(events, auto_ready=False)
        service = service_for()
        service._context = context
        gate = Event()
        results = []
        errors = []

        def start_service():
            gate.wait()
            try:
                results.append(service.start())
            except BaseException as error:
                errors.append(error)

        threads = [Thread(target=start_service) for _ in range(2)]
        for thread in threads:
            thread.start()
        gate.set()
        deadline = monotonic() + 2
        while not context.processes and monotonic() < deadline:
            sleep(0.01)
        self.assertEqual(len(context.processes), 1)
        sleep(0.05)
        self.assertTrue(all(thread.is_alive() for thread in threads))
        self.assertEqual(results, [])

        context.processes[0].ready_event.set()
        for thread in threads:
            thread.join(2)
        try:
            self.assertEqual(errors, [])
            self.assertEqual(results, [service, service])
            self.assertEqual(len(context.processes), 1)
        finally:
            service.stop()

    def test_ready_signal_followed_by_immediate_death_is_not_success(self):
        events = []
        service = service_for()
        service._context = _LeaseContext(events, dies_after_ready=True)

        with self.assertRaisesRegex(WorkerStartupError, "failed to initialize"):
            service.start()

        self.assertIsNone(service._process)
        self.assertFalse(service.is_running)

    def test_partial_ipc_setup_failure_closes_the_first_queue(self):
        events = []

        class PartialContext:
            def __init__(self):
                self.calls = 0

            def Queue(self, *, maxsize):
                del maxsize
                self.calls += 1
                if self.calls == 1:
                    return _ShutdownQueue("partial", events)
                raise OSError("synthetic queue construction failure")

        service = service_for()
        service._context = PartialContext()

        with self.assertRaisesRegex(WorkerStartupError, "IPC setup failed"):
            service.start()

        self.assertEqual(events, ["partial.close", "partial.join_thread"])
        self.assertIsNone(service._request_queue)
        self.assertIsNone(service._process)

    def test_meta_controller_uses_dtype_aware_tolerance_floor(self):
        learned = SimpleNamespace(
            exploration=torch.tensor(1.0),
            tolerance=torch.tensor(1.0e-6),
            temperature=torch.tensor(1.0),
            act=torch.tensor(301.0),
        )
        controller = SimpleNamespace(meta_controls=lambda _root: learned)
        bounded = _VerifiedMetaControllerV2(
            controller=controller,
            tolerance_ceiling=5.0e-3,
            act_hard_floor=301,
            max_simulations=301,
        )

        bf16 = bounded(torch.ones(1, 4, dtype=torch.bfloat16))
        fp32 = bounded(torch.ones(1, 4, dtype=torch.float32))
        self.assertEqual(bf16.tolerance, 5.0e-3)
        self.assertAlmostEqual(fp32.tolerance, 1.0e-6, places=9)

    @staticmethod
    def shutdown_service(*, exits_after):
        events = []
        service = service_for()
        process = _ShutdownProcess(events, exits_after=exits_after)
        request_queue = _ShutdownQueue("request", events)
        response_queue = _ShutdownQueue("response", events)
        cancel_event = Event()
        service._process = process
        service._request_queue = request_queue
        service._response_queue = response_queue
        service._cancel_event = cancel_event
        service._ready_event = Event()
        service._failed_event = Event()
        service._active_request_id = 17
        return service, process, request_queue, response_queue, cancel_event, events

    @staticmethod
    def leased_service(
        *, purpose="inference", failed_starts=(), clock=monotonic, lifetime=60.0
    ):
        events = []
        manager = _RecordingLeaseManager(events, clock=clock)
        service = service_for(
            gpu_lease_manager=manager,
            gpu_lease_owner="test-resident",
            gpu_lease_purpose=purpose,
            gpu_lease_vram_bytes=1_024,
            worker_lifetime_seconds=lifetime,
        )
        context = _LeaseContext(events, failed_starts=failed_starts)
        service._context = context
        return service, manager, context, events

    def test_gpu_lease_is_acquired_after_process_construction_before_spawn(self):
        service, manager, _context, events = self.leased_service()

        try:
            service.start()
            lease = service.gpu_lease
            self.assertIsNotNone(lease)
            self.assertEqual(manager.active, lease)
            self.assertLess(
                events.index(("process.construct", 1)),
                events.index("lease.acquire"),
            )
            self.assertLess(
                events.index("lease.acquire"),
                events.index(("lease.pre_spawn_alive", True)),
            )
            self.assertLess(
                events.index(("lease.pre_spawn_alive", True)),
                events.index(("process.start", 1)),
            )
        finally:
            service.stop()

    def test_gpu_lease_startup_failure_confirms_death_then_releases(self):
        service, manager, _context, events = self.leased_service(failed_starts=(1,))

        with self.assertRaisesRegex(WorkerStartupError, "spawn failed"):
            service.start()

        self.assertIsNone(manager.active)
        self.assertIsNone(service.gpu_lease)
        self.assertIsNone(service._process)
        join_index = events.index(("process.join", 1, 10.0))
        ipc_index = events.index("queue-2.join_thread")
        release_index = events.index("lease.release")
        self.assertLess(join_index, ipc_index)
        self.assertLess(ipc_index, release_index)

    def test_gpu_lease_covers_the_entire_idle_resident_lifetime(self):
        service, manager, _context, events = self.leased_service()

        try:
            service.start()
            lease = service.gpu_lease
            self.assertIsNotNone(lease)
            assert lease is not None
            self.assertGreater(lease.ttl_seconds, 0.0)
            self.assertLessEqual(lease.ttl_seconds, 60.0)
            self.assertEqual(manager.active, lease)

            # No generation request occurs. Re-entering start validates and
            # retains the same lifetime lease instead of acquiring a new one.
            service.start()
            self.assertIs(service.gpu_lease, lease)
            self.assertEqual(events.count("lease.acquire"), 1)
            self.assertGreaterEqual(events.count("lease.validate"), 2)
        finally:
            service.stop()
        self.assertIsNone(manager.active)

    def test_running_worker_rejects_a_changed_purpose_provider(self):
        purpose = ["inference"]
        service, manager, context, _events = self.leased_service(
            purpose=lambda: purpose[0]
        )

        try:
            service.start()
            lease = service.gpu_lease
            process = context.processes[0]
            purpose[0] = "validation"
            with self.assertRaisesRegex(StaleGPULeaseError, "purpose"):
                service.start()
            self.assertIs(service.gpu_lease, lease)
            self.assertEqual(manager.active, lease)
            self.assertTrue(process.is_alive())
        finally:
            service.stop()

    def test_running_worker_rejects_budget_mismatch_and_expired_lifetime(self):
        clock = _LeaseClock()
        service, manager, context, _events = self.leased_service(
            clock=clock,
            lifetime=5.0,
        )

        try:
            service.start()
            lease = service.gpu_lease
            assert lease is not None
            service.gpu_lease_vram_bytes = 512
            with self.assertRaisesRegex(GPULeaseBudgetError, "exactly"):
                service.start()
            self.assertEqual(manager.active, lease)
            self.assertTrue(context.processes[0].is_alive())

            service.gpu_lease_vram_bytes = 1_024
            clock.advance(5.0)
            with self.assertRaises(ExpiredGPULeaseError):
                service.start()
            self.assertEqual(manager.active, lease)
            self.assertTrue(context.processes[0].is_alive())
        finally:
            service.stop()

    def test_crashed_worker_is_cleaned_before_restart_gets_a_new_epoch(self):
        service, manager, context, _events = self.leased_service()

        try:
            service.start()
            first = service.gpu_lease
            assert first is not None
            context.processes[0].alive = False

            service.start()
            second = service.gpu_lease
            assert second is not None
            self.assertGreater(second.epoch, first.epoch)
            self.assertEqual(manager.active, second)
            self.assertEqual(len(context.processes), 2)
            self.assertTrue(context.processes[1].is_alive())
        finally:
            service.stop()

    def test_exact_watchdog_reaped_stale_lease_is_allowed_during_cleanup(self):
        service, manager, context, _events = self.leased_service()
        service.start()
        lease = service.gpu_lease
        assert lease is not None
        context.processes[0].alive = False

        revocation = manager.reap()
        self.assertIsNotNone(revocation)
        assert revocation is not None
        self.assertEqual(revocation.lease, lease)
        self.assertEqual(revocation.reason, "owner_confirmed_dead")

        service.stop()
        self.assertIsNone(service.gpu_lease)
        self.assertIsNone(service._process)
        self.assertIsNone(manager.active)

    def test_non_reaped_stale_lease_is_not_silently_accepted(self):
        service, manager, _context, _events = self.leased_service()
        service.start()
        lease = service.gpu_lease
        assert lease is not None
        manager.release(lease)

        with self.assertRaises(StaleGPULeaseError):
            service.stop()

        self.assertIs(service.gpu_lease, lease)
        self.assertIsNotNone(service._process)
        self.assertIsNone(service._request_queue)
        self.assertIsNone(service._response_queue)
        self.assertIsNone(manager.active)

    def test_stop_gracefully_joins_before_cleaning_ipc(self):
        service, _process, _request, _response, cancel, events = self.shutdown_service(
            exits_after="graceful"
        )

        service.stop(timeout=0.25)

        self.assertTrue(cancel.is_set())
        self.assertEqual(
            events,
            [
                ("request.put", 0.2),
                ("join", 0.25),
                "request.close",
                "request.join_thread",
                "response.close",
                "response.join_thread",
            ],
        )
        self.assertIsNone(service._process)
        self.assertIsNone(service._request_queue)
        self.assertIsNone(service._response_queue)
        self.assertIsNone(service._active_request_id)

    def test_stop_escalates_from_graceful_join_to_terminate_and_join(self):
        service, _process, _request, _response, _cancel, events = self.shutdown_service(
            exits_after="terminate"
        )

        service.stop(timeout=0.25)

        self.assertEqual(
            events[:4],
            [
                ("request.put", 0.2),
                ("join", 0.25),
                "terminate",
                ("join", 2.0),
            ],
        )
        self.assertNotIn("kill", events)
        self.assertEqual(events[4], "request.close")
        self.assertIsNone(service._process)

    def test_stop_escalates_to_kill_and_confirms_death_before_ipc_cleanup(self):
        service, _process, _request, _response, _cancel, events = self.shutdown_service(
            exits_after="kill"
        )

        service.stop(timeout=0.25)

        self.assertEqual(
            events[:6],
            [
                ("request.put", 0.2),
                ("join", 0.25),
                "terminate",
                ("join", 2.0),
                "kill",
                ("join", 2.0),
            ],
        )
        self.assertEqual(events[6], "request.close")
        self.assertIsNone(service._process)

    def test_stop_retains_live_process_and_ipc_when_kill_does_not_work(self):
        service, process, request, response, cancel, events = self.shutdown_service(
            exits_after=None
        )
        manager = GPULeaseManager()
        service.gpu_lease_manager = manager
        lease = manager.acquire(
            service.gpu_lease_owner,
            service._current_gpu_purpose(),
            service.gpu_lease_vram_bytes,
            deadline=monotonic() + 60.0,
            owner_alive=process.is_alive,
        )
        service._gpu_lease = lease
        ready = service._ready_event
        failed = service._failed_event

        with self.assertRaisesRegex(WorkerExecutionError, "survived"):
            service.stop(timeout=0.25)

        self.assertEqual(
            events,
            [
                ("request.put", 0.2),
                ("join", 0.25),
                "terminate",
                ("join", 2.0),
                "kill",
                ("join", 2.0),
            ],
        )
        self.assertTrue(cancel.is_set())
        self.assertIs(service._process, process)
        self.assertIs(service._request_queue, request)
        self.assertIs(service._response_queue, response)
        self.assertIs(service._cancel_event, cancel)
        self.assertIs(service._ready_event, ready)
        self.assertIs(service._failed_event, failed)
        self.assertEqual(service._active_request_id, 17)
        self.assertIs(service.gpu_lease, lease)
        self.assertEqual(manager.active, lease)

    def test_bounded_token_guard_stops_exact_cycles_without_false_emphasis(self):
        long_pattern = torch.arange(24, dtype=torch.int64)
        guard = _TokenRepetitionGuard()
        self.assertFalse(guard.observe(long_pattern))
        self.assertTrue(guard.observe(long_pattern))
        self.assertTrue(guard.triggered)
        trimmed, repeated = truncate_repeated_tokens(
            torch.cat((torch.tensor([999]), long_pattern, long_pattern))
        )
        self.assertTrue(repeated)
        self.assertEqual(trimmed.tolist(), [999, *long_pattern.tolist()])

        short_pattern = torch.arange(8, dtype=torch.int64)
        short_guard = _TokenRepetitionGuard()
        for _ in range(5):
            self.assertFalse(short_guard.observe(short_pattern))
        self.assertTrue(short_guard.observe(short_pattern))

        emphasis = _TokenRepetitionGuard()
        self.assertFalse(emphasis.observe(torch.tensor([7] * 96)))

        prompt = torch.arange(100, 160, dtype=torch.int64)
        prompt_echo = _TokenRepetitionGuard(prompt)
        self.assertFalse(prompt_echo.observe(torch.arange(10, dtype=torch.int64)))
        self.assertFalse(prompt_echo.observe(prompt[:24]))
        self.assertFalse(
            prompt_echo.observe(torch.arange(1_000, 1_060, dtype=torch.int64))
        )
        self.assertTrue(prompt_echo.observe(prompt[:24]))
        self.assertEqual(prompt_echo.trigger_reason, "prompt_echo")
        self.assertEqual(prompt_echo.repeat_cut_index, 94)

    def test_single_worker_streams_token_tensors_and_keeps_base_frozen(self):
        with service_for() as service:
            parent_pid = os.getpid()
            chunks = list(service.iter_generate_tokens("hello", max_new_tokens=4))
            self.assertIsNotNone(service.worker_pid)
            self.assertNotEqual(service.worker_pid, parent_pid)
        self.assertTrue(chunks[-1].final)
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[-1].finish_reason, "length")
        self.assertEqual(
            torch.cat([chunk.token_ids for chunk in chunks]).tolist(),
            [101, 102, 103, 104],
        )
        observed = 0
        for chunk in chunks:
            observed += int(chunk.token_ids.numel())
            self.assertEqual(chunk.generated_total, observed)

    def test_status_ok_tokens_are_atomic_until_successful_terminal(self):
        service = service_for()
        service._process = _AliveProcess()
        service._request_queue = Queue()
        service._response_queue = Queue()
        service._cancel_event = Event()
        service._response_queue.put(
            make_response(
                1,
                STATUS_OK,
                torch.tensor([101], dtype=torch.int64),
                generated_total=1,
                final=False,
            )
        )
        service._response_queue.put(
            make_response(
                1,
                STATUS_MODEL_ERROR,
                generated_total=1,
                final=True,
                finish_reason=FINISH_ERROR,
            )
        )
        published = []
        with patch.object(service, "start", return_value=service):
            with self.assertRaises(WorkerExecutionError):
                for chunk in service.iter_generate_tokens(
                    "atomic failure", max_new_tokens=2
                ):
                    published.append(chunk)
        self.assertEqual(published, [])

    def test_controller_rejects_response_artifact_authority_tampering(self):
        service = service_for()
        service._process = _AliveProcess()
        service._request_queue = Queue()
        service._response_queue = Queue()
        service._cancel_event = Event()
        service.cancellation_timeout = 0.01
        service._response_queue.put(
            make_response(
                1,
                STATUS_OK,
                torch.tensor([101], dtype=torch.int64),
                generated_total=1,
                final=True,
                finish_reason=FINISH_LENGTH,
                artifact_digest=torch.ones(DIGEST_BYTES, dtype=torch.int64),
            )
        )
        with patch.object(service, "start", return_value=service):
            with self.assertRaisesRegex(WorkerAuthorityError, "digest authority"):
                list(
                    service.iter_generate_tokens("tampered response", max_new_tokens=1)
                )

    def test_text_boundary_decodes_only_after_tensor_worker_response(self):
        with service_for() as service:
            result = service.generate("hi", max_new_tokens=3)
        self.assertEqual(result.token_ids.tolist(), [101, 102, 103])
        self.assertEqual(result.text, "101 102 103")
        self.assertEqual(result.finish_reason, "length")

    def test_core_pipeline_runs_before_authoritative_base_generation(self):
        with service_for(
            require_core=True,
            core_pipeline_factory=AdvisoryCoreFactory(),
        ) as service:
            result = service.generate("core first", max_new_tokens=2)
        self.assertEqual(result.token_ids.tolist(), [101, 102])
        self.assertEqual(result.text, "101 102")
        self.assertEqual(result.finish_reason, "length")

    def test_worker_propagates_stable_conversation_identity_without_cross_talk(self):
        with service_for(
            require_core=True,
            core_pipeline_factory=SessionAwareCoreFactory(),
        ) as service:
            alpha_first = service.generate(
                "alpha one", max_new_tokens=1, conversation_id="conversation-A"
            )
            beta = service.generate(
                "beta", max_new_tokens=1, conversation_id="conversation-B"
            )
            alpha_second = service.generate(
                "alpha two", max_new_tokens=1, conversation_id="conversation-A"
            )
        self.assertEqual(
            alpha_first.token_ids.tolist(), alpha_second.token_ids.tolist()
        )
        self.assertNotEqual(alpha_first.token_ids.tolist(), beta.token_ids.tolist())

    def test_worker_fault_injection_rejects_stale_epoch_deadline_and_artifact(self):
        artifact = torch.arange(1, DIGEST_BYTES + 1, dtype=torch.int64)

        def execute_fault(kind: str):
            request_queue = Queue()
            response_queue = Queue()
            cancel = Event()
            ready = Event()
            failed = Event()
            lease_deadline = monotonic_ns() + 10_000_000_000
            launch = _worker_authority_tensor(artifact)
            launch[0] = 7
            launch[1] = lease_deadline
            worker = Thread(
                target=_worker_main,
                args=(
                    SafeFakeModel,
                    None,
                    request_queue,
                    response_queue,
                    cancel,
                    ready,
                    failed,
                    torch.empty((0, 1), dtype=torch.int64),
                    32,
                    1,
                    launch,
                ),
            )
            worker.start()
            self.assertTrue(ready.wait(2.0))
            self.assertFalse(failed.is_set())
            epoch = 7
            request_deadline = monotonic_ns() + 5_000_000_000
            request_artifact = artifact.clone()
            if kind == "epoch":
                epoch = 6
            elif kind == "deadline":
                request_deadline = monotonic_ns() - 1
            elif kind == "artifact":
                request_artifact[0] ^= 1
            request_queue.put(
                make_generate_request(
                    1,
                    torch.tensor([[1, 2]], dtype=torch.int64),
                    None,
                    max_new_tokens=1,
                    job_id=101,
                    lease_epoch=epoch,
                    request_deadline_ns=request_deadline,
                    lease_deadline_ns=lease_deadline,
                    artifact_digest=request_artifact,
                )
            )
            frame = parse_response(response_queue.get(timeout=2.0))
            worker.join(2.0)
            self.assertFalse(worker.is_alive())
            return frame

        for kind in ("epoch", "deadline", "artifact"):
            with self.subTest(kind=kind):
                frame = execute_fault(kind)
                self.assertTrue(frame.final)
                self.assertEqual(frame.status, STATUS_AUTHORITY_REJECTED)
                self.assertEqual(frame.token_ids.numel(), 0)

    def test_production_core_result_conditions_logits_before_generation(self):
        with service_for(
            require_core=True,
            require_conditioning=True,
            core_pipeline_factory=ProductionCoreFactory(
                max_abs_logit_bias=0.1,
            ),
        ) as service:
            result = service.generate("conditioned", max_new_tokens=2)
        with service_for(
            require_core=True,
            require_conditioning=True,
            core_pipeline_factory=ProductionCoreFactory(
                result_mode="negated",
                max_abs_logit_bias=0.1,
            ),
        ) as service:
            counterfactual = service.generate("conditioned", max_new_tokens=2)

        self.assertEqual(result.token_ids.tolist(), [127, 102])
        self.assertEqual(counterfactual.token_ids.tolist(), [0, 102])
        self.assertNotEqual(result.token_ids[0], counterfactual.token_ids[0])
        self.assertEqual(result.finish_reason, "length")

    def test_required_production_conditioning_fails_closed(self):
        cases = (
            (
                ProductionCoreFactory(result_mode="missing", gemma_lkg_fallback=False),
                True,
            ),
            (
                ProductionCoreFactory(
                    result_mode="bad_shape", gemma_lkg_fallback=False
                ),
                True,
            ),
            (ProductionCoreFactory(gemma_lkg_fallback=False), False),
        )
        for factory, output_head in cases:
            with self.subTest(
                result_mode=factory.result_mode,
                output_head=output_head,
            ):
                with service_for(
                    require_core=True,
                    require_conditioning=True,
                    output_head=output_head,
                    core_pipeline_factory=factory,
                ) as service:
                    with self.assertRaises(WorkerExecutionError):
                        service.generate("fail closed", max_new_tokens=2)

    def test_product_core_failure_never_publishes_gemma_lkg(self):
        with service_for(
            require_core=True,
            require_conditioning=False,
            core_pipeline_factory=ProductionCoreFactory(result_mode="missing"),
        ) as service:
            published = []
            with self.assertRaises(WorkerExecutionError):
                for chunk in service.iter_generate_tokens(
                    "fail closed instead of LKG", max_new_tokens=2
                ):
                    published.append(chunk)
        self.assertEqual(published, [])

    def test_unsampled_data_mutation_plus_core_failure_never_publishes_lkg(self):
        with service_for(
            require_core=True,
            require_conditioning=False,
            core_pipeline_factory=ProductionCoreFactory(
                result_mode="unsampled_mutate_missing"
            ),
        ) as service:
            published = []
            with self.assertRaises(WorkerExecutionError):
                for chunk in service.iter_generate_tokens(
                    "unsampled mutation", max_new_tokens=2
                ):
                    published.append(chunk)
        self.assertEqual(published, [])

    def test_detectable_product_core_mutation_fails_as_base_mutated(self):
        with service_for(
            require_core=True,
            require_conditioning=False,
            core_pipeline_factory=ProductionCoreFactory(result_mode="mutate_missing"),
        ) as service:
            with self.assertRaises(BaseModelMutationError):
                service.generate("mutated base", max_new_tokens=2)

    def test_production_factory_requires_contextual_bounded_conditioning(self):
        factory = LocalGemmaCorePipelineFactory()
        self.assertTrue(factory.contextual_tokens)
        self.assertLessEqual(factory.max_abs_logit_bias, 0.1)
        self.assertEqual(factory.history, 16)
        self.assertGreaterEqual(factory.max_iter, 17)
        self.assertFalse(factory.gemma_lkg_fallback)
        with self.assertRaisesRegex(ValueError, "LKG fallback is disabled"):
            LocalGemmaCorePipelineFactory(gemma_lkg_fallback=True)
        with self.assertRaisesRegex(ValueError, "contextual_tokens=True"):
            LocalGemmaCorePipelineFactory(contextual_tokens=False)
        with self.assertRaisesRegex(ValueError, "max_iter >= 17"):
            LocalGemmaCorePipelineFactory(max_iter=16)
        with self.assertRaisesRegex(ValueError, "rank 16"):
            LocalGemmaCorePipelineFactory(history=15)
        for value in (0.0, 0.100001, float("inf"), float("nan")):
            with self.subTest(max_abs_logit_bias=value):
                with self.assertRaises(ValueError):
                    LocalGemmaCorePipelineFactory(max_abs_logit_bias=value)

        model = nn.Linear(4, 4)
        model.config = SimpleNamespace(hidden_size=4)
        runtime = SimpleNamespace(search_engine=SimpleNamespace())

        def install_search(search, *, request_mac_budget):
            runtime.search_engine = search
            runtime.search_mac_budget = request_mac_budget

        runtime.install_certified_search_v2 = install_search
        backbone = nn.Identity()
        base_before = model.weight.detach().clone()
        version_before = model.weight._version
        with (
            patch(
                "cogni_agent.model_service.LocalGemmaFeatureBackbone",
                return_value=backbone,
            ) as backbone_factory,
            patch(
                "cogni_agent.model_service.build_genesis_runtime",
                return_value=runtime,
            ),
            patch("cogni_agent.model_service.load_config", return_value={}),
        ):
            pipeline = factory(model)
        backbone_factory.assert_called_once_with(model, contextual_tokens=True)
        self.assertIs(pipeline.runtime, runtime)
        self.assertIsInstance(runtime.search_engine, BoundedPUCTSearchV2)
        self.assertGreater(runtime.search_mac_budget, 0)
        costs = runtime.search_engine.config
        self.assertEqual(costs.max_depth, 100)
        self.assertEqual(costs.simulations, PRODUCTION_CTS_ACT_HARD_FLOOR)
        self.assertGreater(costs.meta_policy_macs, 0)
        self.assertGreater(costs.action_policy_macs, 0)
        self.assertGreater(costs.critic_macs, 0)
        self.assertGreater(costs.retrieval_macs, 0)
        self.assertGreater(costs.transition_macs, 0)
        self.assertEqual(costs.retrieval_macs, 301 * 4)
        self.assertEqual(
            costs.transition_macs,
            3 * factory.max_iter * 4 * (2 * factory.history + 8),
        )
        self.assertIsInstance(pipeline.transition, CertifiedBroydenTransitionV2)
        self.assertEqual(pipeline.transition.rank, 16)
        self.assertGreaterEqual(pipeline.transition.max_iter, 17)
        self.assertIsInstance(pipeline.policy_value, SearchCollaboratorsV2)
        controller = pipeline.policy_value.action_policy.__self__
        self.assertIsInstance(controller, LearnedCTSController)
        self.assertIsNot(
            pipeline.policy_value.action_policy,
            pipeline.policy_value.critic,
        )
        first = pipeline.policy_value.action_policy(
            torch.tensor([[1.0, -1.0, 0.5, -0.5]])
        )
        second = pipeline.policy_value.action_policy(
            torch.tensor([[-1.0, 1.0, -0.5, 0.5]])
        )
        self.assertFalse(torch.equal(first, second))
        self.assertLessEqual(float(first.abs().max()), ACTION_LOGIT_BOUND)
        controls = pipeline.policy_value.meta_controller(torch.ones(1, 4))
        self.assertEqual(
            controls.act_simulations,
            PRODUCTION_CTS_ACT_HARD_FLOOR,
        )
        self.assertTrue(torch.equal(model.weight, base_before))
        self.assertEqual(model.weight._version, version_before)

    def test_production_factory_rejects_tampered_or_random_controller(self):
        model = nn.Linear(4, 4)
        model.config = SimpleNamespace(hidden_size=4)
        factory = LocalGemmaCorePipelineFactory()
        cases = (
            patch(
                "cogni_agent.model_service.load_default_bounded_cts_controller",
                side_effect=CTSCheckpointError("tampered"),
            ),
            patch(
                "cogni_agent.model_service.load_default_bounded_cts_controller",
                return_value=nn.Linear(12, 3),
            ),
        )
        for controller_patch in cases:
            with self.subTest(controller_patch=controller_patch), controller_patch:
                with self.assertRaises(CTSCheckpointError):
                    factory(model)

    def test_controller_integrity_failure_marks_worker_startup_failed(self):
        def model_factory():
            model = SafeFakeModel()
            model.config.hidden_size = 4
            return model

        cases = (
            dict(side_effect=CTSCheckpointError("tampered")),
            dict(return_value=nn.Linear(12, 3)),
        )
        for loader_outcome in cases:
            with (
                self.subTest(loader_outcome=tuple(loader_outcome)),
                patch(
                    "cogni_agent.model_service.load_default_bounded_cts_controller",
                    **loader_outcome,
                ),
            ):
                ready = Event()
                failed = Event()
                _worker_main(
                    model_factory,
                    LocalGemmaCorePipelineFactory(),
                    Queue(),
                    Queue(),
                    Event(),
                    ready,
                    failed,
                    torch.empty((0, 1), dtype=torch.int64),
                    32,
                    1,
                )
                self.assertTrue(failed.is_set())
                self.assertFalse(ready.is_set())

    def test_v2_unsafe_telemetry_is_rejected_before_conditioning(self):
        model = SafeFakeModel(require_conditioning=True)
        best_state = torch.tensor([[0.5, -0.25, 0.75, 1.0]])
        cases = (
            dict(
                safe_for_decode=False,
                linear_solve_fallbacks=0,
                unsafe_silent_fallbacks=0,
            ),
            dict(
                safe_for_decode=True,
                linear_solve_fallbacks=1,
                unsafe_silent_fallbacks=0,
            ),
            dict(
                safe_for_decode=True,
                linear_solve_fallbacks=0,
                unsafe_silent_fallbacks=1,
            ),
        )
        for telemetry in cases:
            with self.subTest(telemetry=telemetry):
                result = CoreTurnResult(
                    inference=SimpleNamespace(
                        search=SimpleNamespace(
                            best_state=best_state,
                            telemetry=SimpleNamespace(**telemetry),
                        )
                    ),
                    pooled_observation=best_state,
                    telemetry=SimpleNamespace(advisory_only=True),
                )
                with self.assertRaisesRegex(RuntimeError, "unsafe CTS V2 telemetry"):
                    _required_product_logits_processor(
                        model,
                        result,
                        max_abs_logit_bias=0.05,
                    )

    def test_core_failure_blocks_base_answer_instead_of_silent_fallback(self):
        with service_for(
            require_core=True,
            core_pipeline_factory=AdvisoryCoreFactory(fail=True),
        ) as service:
            with self.assertRaises(WorkerExecutionError):
                service.generate("fail closed", max_new_tokens=2)

    def test_stop_token_ids_are_applied_by_worker_stopping_criteria(self):
        with service_for() as service:
            result = service.generate(
                "stop",
                max_new_tokens=4,
                stop_token_ids=torch.tensor([102], dtype=torch.int64),
            )
        self.assertEqual(result.token_ids.tolist(), [101, 102])
        self.assertEqual(result.finish_reason, "stop")

    def test_early_model_stop_and_token_budget_have_distinct_reasons(self):
        with service_for() as service:
            stopped = service.generate("early stop", max_new_tokens=8)
            limited = service.generate("length", max_new_tokens=3)
        self.assertEqual(stopped.token_ids.tolist(), [101, 102, 103, 104])
        self.assertEqual(stopped.finish_reason, "stop")
        self.assertEqual(limited.token_ids.tolist(), [101, 102, 103])
        self.assertEqual(limited.finish_reason, "length")

    def test_cancelled_terminal_chunk_preserves_its_finish_reason(self):
        service = service_for(delay=0.15)
        chunks = []
        errors = []

        def run():
            try:
                chunks.extend(
                    service.iter_generate_tokens("cancel chunks", max_new_tokens=8)
                )
            except BaseException as exc:  # asserted below
                errors.append(exc)

        try:
            service.start()
            worker = Thread(target=run)
            worker.start()
            for _ in range(100):
                if service.active_request_id is not None:
                    break
                sleep(0.01)
            self.assertTrue(service.cancel(service.active_request_id))
            worker.join(5)
            self.assertFalse(worker.is_alive())
            self.assertEqual(errors, [])
            self.assertTrue(chunks[-1].final)
            self.assertTrue(chunks[-1].cancelled)
            self.assertEqual(chunks[-1].finish_reason, "cancelled")
        finally:
            service.stop()

    def test_idle_timeout_resets_after_each_healthy_worker_frame(self):
        with service_for(delay=0.12) as service:
            started = monotonic()
            result = service.generate("healthy stream", max_new_tokens=4, timeout=0.25)
            elapsed = monotonic() - started
        self.assertGreater(elapsed, 0.25)
        self.assertEqual(result.finish_reason, "length")

    def test_consumer_render_time_does_not_count_as_worker_idle_time(self):
        with service_for(delay=0.01) as service:
            stream = service.iter_generate_tokens(
                "slow renderer", max_new_tokens=4, timeout=0.1
            )
            first = next(stream)
            sleep(0.2)
            chunks = [first, *stream]
        self.assertEqual(chunks[-1].finish_reason, "length")
        self.assertEqual(chunks[-1].generated_total, 4)

    def test_stalled_worker_still_expires_the_idle_timeout(self):
        with service_for(delay=0.4) as service:
            with self.assertRaises(TimeoutError):
                list(
                    service.iter_generate_tokens(
                        "stalled stream", max_new_tokens=4, timeout=0.15
                    )
                )

    def test_controller_rejects_non_monotonic_generated_total(self):
        service = service_for()
        service._process = _AliveProcess()
        service._request_queue = Queue()
        service._response_queue = Queue()
        service._cancel_event = Event()
        service._response_queue.put(
            make_response(
                1,
                STATUS_OK,
                torch.tensor([101], dtype=torch.int64),
                generated_total=1,
                final=False,
            )
        )
        service._response_queue.put(
            make_response(
                1,
                STATUS_OK,
                torch.tensor([102], dtype=torch.int64),
                generated_total=1,
                final=True,
                finish_reason=FINISH_LENGTH,
            )
        )
        # The controller drains to a valid terminal frame after rejecting the
        # corrupt counter, just as it would for an untrusted worker failure.
        service._response_queue.put(
            make_response(
                1,
                STATUS_CANCELLED,
                generated_total=1,
                final=True,
                finish_reason=FINISH_CANCELLED,
            )
        )
        with patch.object(service, "start", return_value=service):
            with self.assertRaisesRegex(WorkerExecutionError, "not monotonic"):
                list(service.iter_generate_tokens("counter", max_new_tokens=2))

    def test_cooperative_cancellation_interrupts_the_active_request(self):
        service = service_for(delay=0.15)
        errors = []

        def run():
            try:
                service.generate("cancel me", max_new_tokens=8)
            except BaseException as exc:  # asserted below
                errors.append(exc)

        try:
            service.start()
            worker = Thread(target=run)
            worker.start()
            for _ in range(100):
                if service.active_request_id is not None:
                    break
                sleep(0.01)
            self.assertTrue(service.cancel(service.active_request_id))
            worker.join(5)
            self.assertFalse(worker.is_alive())
            self.assertEqual(len(errors), 1)
            self.assertIsInstance(errors[0], GenerationCancelled)
        finally:
            service.stop()

    def test_worker_detects_and_retires_a_mutated_base_model(self):
        service = service_for(mutate=True)
        published = []
        try:
            with self.assertRaises(BaseModelMutationError):
                for chunk in service.iter_generate_tokens("mutation", max_new_tokens=1):
                    published.append(chunk)
        finally:
            service.stop()
        self.assertEqual(published, [])

    def test_sampled_signature_detects_data_grad_training_and_buffer_mutation(self):
        def prepared() -> SafeFakeModel:
            model = SafeFakeModel()
            _prepare_base_model(model)
            return model

        mutations = (
            lambda model: model.weight.data.add_(1.0),
            lambda model: model.weight.requires_grad_(True),
            lambda model: model.train(),
            lambda model: model.bounded_state.data.add_(1.0),
        )
        for mutate in mutations:
            model = prepared()
            before = _parameter_signature(model)
            mutate(model)
            with self.subTest(mutation=mutate):
                self.assertNotEqual(_parameter_signature(model), before)

    def test_sampled_signature_supports_scalar_bfloat16_tensors(self):
        model = nn.Module()
        model.register_buffer("scalar_state", torch.tensor(1.0, dtype=torch.bfloat16))

        before = _parameter_signature(model)
        self.assertEqual(_parameter_signature(model), before)

        model.scalar_state.add_(1.0)
        self.assertNotEqual(_parameter_signature(model), before)

    def test_data_mutation_fails_without_publishing_partial_tokens(self):
        service = service_for(data_mutate=True)
        published = []
        try:
            with self.assertRaises(BaseModelMutationError):
                for chunk in service.iter_generate_tokens(
                    "data mutation", max_new_tokens=2
                ):
                    published.append(chunk)
        finally:
            service.stop()
        self.assertEqual(published, [])

    def test_prompt_and_generation_bounds_are_enforced_before_worker_use(self):
        with service_for(max_prompt_chars=4) as service:
            with self.assertRaises(RequestLimitError):
                service.generate("12345")
            with self.assertRaises(RequestLimitError):
                service.generate("ok", max_new_tokens=9)

    def test_local_tokenizer_loader_forces_offline_options(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "config.json").write_text(
                '{"model_type":"gemma4","architectures":["GemmaForCausalLM"]}',
                encoding="utf-8",
            )
            (root / "model.safetensors").write_bytes(b"weights")
            (root / "tokenizer.json").write_text("{}", encoding="utf-8")
            tokenizer = load_local_tokenizer(
                root,
                tokenizer_class=RecordingTokenizerClass,
                tokenizer_kwargs={"use_fast": True},
            )
            self.assertIsInstance(tokenizer, FakeTokenizer)
            self.assertEqual(RecordingTokenizerClass.path, str(root.resolve()))
            self.assertTrue(RecordingTokenizerClass.options["local_files_only"])
            self.assertFalse(RecordingTokenizerClass.options["trust_remote_code"])
            self.assertFalse(RecordingTokenizerClass.options["force_download"])
            with self.assertRaises(ValueError):
                load_local_tokenizer(
                    root,
                    tokenizer_class=RecordingTokenizerClass,
                    tokenizer_kwargs={"local_files_only": False},
                )

            service = ModelService(FakeTokenizer(), LocalGemmaModelFactory(str(root)))
            self.assertIsInstance(
                service.core_pipeline_factory, LocalGemmaCorePipelineFactory
            )
            with self.assertRaisesRegex(ValueError, "4096"):
                ModelService(
                    FakeTokenizer(),
                    LocalGemmaModelFactory(str(root)),
                    max_input_tokens=4_097,
                )


if __name__ == "__main__":
    unittest.main()
