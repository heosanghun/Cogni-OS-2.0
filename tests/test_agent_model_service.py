from dataclasses import dataclass
import os
from pathlib import Path
from threading import Thread
from time import sleep
import tempfile
from types import SimpleNamespace
import unittest

import torch
from torch import nn

from cogni_agent.model_service import (
    BaseModelMutationError,
    GenerationCancelled,
    LocalGemmaCorePipelineFactory,
    LocalGemmaModelFactory,
    ModelService,
    RequestLimitError,
    WorkerExecutionError,
    load_local_tokenizer,
)
from cogni_agent.core_pipeline import CoreTurnRequest


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
        self, *, delay: float = 0.0, mutate: bool = False, require_core: bool = False
    ):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1))
        self.config = FakeConfig()
        self.delay = delay
        self.mutate = mutate
        self.require_core = require_core
        self.core_ran = False

    def generate(self, **kwargs):
        if kwargs.get("use_cache") is not False or kwargs.get("do_sample") is not False:
            raise RuntimeError("unsafe generation options")
        if self.training or self.weight.requires_grad or self.config.use_cache:
            raise RuntimeError("base model was not frozen for inference")
        if self.require_core and not self.core_ran:
            raise RuntimeError("base generate ran before the Cogni-Core pipeline")
        input_ids = kwargs["input_ids"]
        streamer = kwargs["streamer"]
        criteria = kwargs["stopping_criteria"]
        streamer.put(input_ids)
        output = input_ids.clone()
        if self.mutate:
            self.weight.add_(1.0)
        for token in (101, 102, 103, 104)[: kwargs["max_new_tokens"]]:
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

    def __call__(self):
        if os.getpid() == self.parent_pid:
            raise RuntimeError("model must never load in the controller process")
        return SafeFakeModel(
            delay=self.delay,
            mutate=self.mutate,
            require_core=self.require_core,
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


class RecordingTokenizerClass:
    path = None
    options = None

    @classmethod
    def from_pretrained(cls, path, **options):
        cls.path = path
        cls.options = options
        return FakeTokenizer()


def service_for(
    *,
    delay=0.0,
    mutate=False,
    require_core=False,
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
    def test_single_worker_streams_token_tensors_and_keeps_base_frozen(self):
        with service_for() as service:
            parent_pid = os.getpid()
            chunks = list(service.iter_generate_tokens("hello", max_new_tokens=4))
            self.assertIsNotNone(service.worker_pid)
            self.assertNotEqual(service.worker_pid, parent_pid)
        self.assertTrue(chunks[-1].final)
        self.assertEqual(
            torch.cat([chunk.token_ids for chunk in chunks]).tolist(),
            [101, 102, 103, 104],
        )

    def test_text_boundary_decodes_only_after_tensor_worker_response(self):
        with service_for() as service:
            result = service.generate("hi", max_new_tokens=3)
        self.assertEqual(result.token_ids.tolist(), [101, 102, 103])
        self.assertEqual(result.text, "101 102 103")

    def test_core_pipeline_runs_before_authoritative_base_generation(self):
        with service_for(
            require_core=True,
            core_pipeline_factory=AdvisoryCoreFactory(),
        ) as service:
            result = service.generate("core first", max_new_tokens=2)
        self.assertEqual(result.token_ids.tolist(), [101, 102])
        self.assertEqual(result.text, "101 102")

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
        try:
            with self.assertRaises(BaseModelMutationError):
                service.generate("mutation", max_new_tokens=1)
        finally:
            service.stop()

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
