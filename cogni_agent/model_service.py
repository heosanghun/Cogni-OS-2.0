"""Resident, single-owner local model process with tensor-only IPC."""

from __future__ import annotations

from dataclasses import dataclass
import multiprocessing
import os
from pathlib import Path
from queue import Empty, Full
import sys
from threading import Lock, RLock
from time import monotonic, sleep
from typing import Any, Callable, Iterator, Mapping

import torch
from torch import Tensor, nn

from cogni_agent.core_pipeline import CoreTurnPipeline, CoreTurnRequest
from cogni_core.backbone import (
    LocalGemmaFeatureBackbone,
    load_local_gemma,
    verify_local_gemma_path,
)
from cogni_core.meta_router import cognitive_state_tensor
from cogni_core.search import ContractiveBroydenTransition
from cogni_os.config import load_config
from cogni_os.factory import build_genesis_runtime

from .protocol import (
    HARD_MAX_INPUT_TOKENS,
    HARD_MAX_NEW_TOKENS,
    STATUS_BASE_MUTATED,
    STATUS_CANCELLED,
    STATUS_INVALID_REQUEST,
    STATUS_MODEL_ERROR,
    STATUS_OK,
    TensorMessage,
    TensorProtocolError,
    make_generate_request,
    make_response,
    make_stop_request,
    parse_request,
    parse_response,
)


HARD_MAX_PROMPT_CHARS = 64_000
HARD_MAX_RESPONSE_CHARS = 64_000
DEFAULT_RESPONSE_QUEUE_SIZE = 64


class ModelServiceError(RuntimeError):
    """Base error for the local generation service."""


class RequestLimitError(ModelServiceError):
    pass


class ServiceBusyError(ModelServiceError):
    pass


class WorkerStartupError(ModelServiceError):
    pass


class WorkerExecutionError(ModelServiceError):
    pass


class GenerationCancelled(ModelServiceError):
    pass


class BaseModelMutationError(ModelServiceError):
    pass


@dataclass(frozen=True)
class GenerationChunk:
    request_id: int
    token_ids: Tensor
    generated_total: int
    final: bool
    cancelled: bool = False


@dataclass(frozen=True)
class GenerationResult:
    request_id: int
    token_ids: Tensor
    text: str


@dataclass(frozen=True)
class LocalGemmaModelFactory:
    """Picklable worker-side loader for one verified local Gemma directory."""

    model_path: str
    vram_limit_gib: float = 16.7

    def __post_init__(self) -> None:
        root = verify_local_gemma_path(self.model_path)
        object.__setattr__(self, "model_path", str(root))
        if not 0.0 < float(self.vram_limit_gib) <= 16.7:
            raise ValueError("vram_limit_gib must lie in (0, 16.7]")

    def __call__(self) -> nn.Module:
        _force_offline_environment()
        model, _tokenizer = load_local_gemma(
            self.model_path,
            vram_limit_gib=self.vram_limit_gib,
        )
        return model


@dataclass(frozen=True)
class LocalGemmaCorePipelineFactory:
    """Build the production advisory Cogni-Core path inside the model worker."""

    transition_contraction: float = 0.4
    spectral_margin: float = 0.95
    tolerance: float = 5.0e-3
    max_iter: int = 12
    history: int = 6
    fallback_steps: int = 32

    def __post_init__(self) -> None:
        if not 0.0 < self.transition_contraction < self.spectral_margin < 1.0:
            raise ValueError("DEQ contraction must remain below the spectral margin")
        for name in ("max_iter", "history", "fallback_steps"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if not 0.0 < self.tolerance < 1.0:
            raise ValueError("DEQ tolerance must lie in (0, 1)")

    def __call__(self, model: nn.Module) -> CoreTurnPipeline:
        hidden_size = _hidden_size(model)
        runtime = build_genesis_runtime(
            LocalGemmaFeatureBackbone(model),
            load_config(),
            input_dim=hidden_size,
            state_dim=64,
        )
        width = runtime.search_engine.config.width
        transition = ContractiveBroydenTransition(
            width=width,
            contraction=self.transition_contraction,
            spectral_margin=self.spectral_margin,
            tolerance=self.tolerance,
            max_iter=self.max_iter,
            history=self.history,
            fallback_steps=self.fallback_steps,
        )

        def policy_value(state: Tensor) -> tuple[Tensor, Tensor]:
            # A deterministic dominant branch reaches the configured bounded
            # depth while the authoritative language answer remains Gemma's.
            logits = torch.full((width,), -40.0, device=state.device)
            logits[-1] = 40.0
            return logits, state.float().mean()

        return CoreTurnPipeline(runtime, transition, policy_value)


def _strict_local_options(options: Mapping[str, Any] | None) -> dict[str, Any]:
    result = dict(options or {})
    if result.get("local_files_only") is False:
        raise ValueError("local_files_only=False violates the offline policy")
    if result.get("trust_remote_code") is True:
        raise ValueError("trust_remote_code=True violates the offline policy")
    if result.get("force_download") is True:
        raise ValueError("force_download=True violates the offline policy")
    result.update(
        local_files_only=True,
        trust_remote_code=False,
        force_download=False,
    )
    return result


def load_local_tokenizer(
    model_path: str | Path,
    *,
    tokenizer_class: Any | None = None,
    tokenizer_kwargs: Mapping[str, Any] | None = None,
) -> Any:
    """Load only a verified local tokenizer; ``transformers`` stays lazy."""

    root = verify_local_gemma_path(model_path)
    _force_offline_environment()
    if tokenizer_class is None:
        try:
            from transformers import AutoTokenizer
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise RuntimeError("transformers is required for a real tokenizer") from exc
        tokenizer_class = AutoTokenizer
    tokenizer = tokenizer_class.from_pretrained(
        str(root), **_strict_local_options(tokenizer_kwargs)
    )
    if not callable(tokenizer) or not callable(getattr(tokenizer, "decode", None)):
        raise TypeError("local tokenizer must be callable and provide decode()")
    return tokenizer


def _force_offline_environment() -> None:
    os.environ.update(
        {
            "HF_HUB_OFFLINE": "1",
            "HF_HUB_DISABLE_TELEMETRY": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "HF_DATASETS_OFFLINE": "1",
            "WANDB_MODE": "offline",
            "TOKENIZERS_PARALLELISM": "false",
        }
    )


def _hidden_size(model: nn.Module) -> int:
    config = getattr(model, "config", None)
    text_config = getattr(config, "text_config", None)
    value = getattr(text_config, "hidden_size", None)
    if value is None:
        value = getattr(config, "hidden_size", None)
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise RuntimeError("could not determine the local Gemma hidden size")
    return value


def _parameter_signature(model: nn.Module) -> tuple[tuple[Any, ...], ...]:
    """Cheap mutation detector without cloning multi-gigabyte base weights."""

    return tuple(
        (
            name,
            id(parameter),
            parameter.data_ptr(),
            int(parameter._version),
            tuple(parameter.shape),
            parameter.dtype,
            parameter.device.type,
            parameter.device.index,
        )
        for name, parameter in model.named_parameters()
    )


def _prepare_base_model(model: nn.Module) -> tuple[tuple[Any, ...], ...]:
    if not isinstance(model, nn.Module):
        raise TypeError("model factory must return torch.nn.Module")
    if not callable(getattr(model, "generate", None)):
        raise TypeError("model must expose generate()")
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    config = getattr(model, "config", None)
    if config is not None and hasattr(config, "use_cache"):
        config.use_cache = False
    return _parameter_signature(model)


def _model_device(model: nn.Module) -> torch.device:
    device = getattr(model, "device", None)
    if device is not None:
        try:
            return torch.device(device)
        except (TypeError, RuntimeError):
            pass
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


class _GenerationStoppingCriteria:
    def __init__(self, cancel_event: Any, stop_token_ids: Tensor) -> None:
        self.cancel_event = cancel_event
        self.stop_token_ids = tuple(map(int, stop_token_ids.tolist()))

    def __call__(self, input_ids: Tensor, _scores: Any, **_kwargs: Any) -> Tensor:
        batch = int(input_ids.shape[0]) if input_ids.ndim else 1
        cancelled = torch.full(
            (batch,),
            bool(self.cancel_event.is_set()),
            dtype=torch.bool,
            device=input_ids.device,
        )
        if not self.stop_token_ids or input_ids.ndim != 2 or input_ids.shape[1] == 0:
            return cancelled
        stop_ids = torch.tensor(
            self.stop_token_ids, dtype=input_ids.dtype, device=input_ids.device
        )
        reached_stop = (input_ids[:, -1, None] == stop_ids[None, :]).any(dim=1)
        return cancelled | reached_stop


def _stopping_criteria(cancel_event: Any, stop_token_ids: Tensor) -> Any:
    # Transformers accepts a list-like custom criteria collection and merges
    # it into its own StoppingCriteriaList. Keeping this object dependency-free
    # avoids importing the optional runtime for injected/fake model workers.
    return [_GenerationStoppingCriteria(cancel_event, stop_token_ids)]


class _TensorResponseStreamer:
    """Convert model streamer callbacks directly into bounded token tensors."""

    def __init__(
        self,
        response_queue: Any,
        request_id: int,
        prompt_ids: Tensor,
        max_new_tokens: int,
        cancel_event: Any,
    ) -> None:
        self.response_queue = response_queue
        self.request_id = request_id
        self.prompt_ids = prompt_ids.detach().to("cpu", dtype=torch.int64).flatten()
        self.max_new_tokens = max_new_tokens
        self.cancel_event = cancel_event
        self.emitted: list[int] = []
        self._prompt_checked = False

    def put(self, value: Any) -> None:
        if self.cancel_event.is_set():
            return
        tensor = torch.as_tensor(value).detach()
        if tensor.ndim > 2 or (tensor.ndim == 2 and tensor.shape[0] != 1):
            raise TensorProtocolError("streamer only accepts batch-one token ids")
        tokens = tensor.to("cpu", dtype=torch.int64).flatten().contiguous()
        if not self._prompt_checked:
            self._prompt_checked = True
            if tokens.shape == self.prompt_ids.shape and torch.equal(
                tokens, self.prompt_ids
            ):
                return
        if not tokens.numel():
            return
        if (
            bool((tokens < 0).any())
            or len(self.emitted) + tokens.numel() > self.max_new_tokens
        ):
            raise TensorProtocolError(
                "model streamer exceeded the response token bound"
            )
        self.emitted.extend(map(int, tokens.tolist()))
        _queue_put(
            self.response_queue,
            make_response(
                self.request_id,
                STATUS_OK,
                tokens,
                generated_total=len(self.emitted),
                final=False,
            ),
        )

    def end(self) -> None:
        return


def _generated_suffix(output: Any, prompt_ids: Tensor) -> Tensor:
    sequences = getattr(output, "sequences", output)
    if (
        not isinstance(sequences, Tensor)
        or sequences.ndim != 2
        or sequences.shape[0] != 1
    ):
        raise TensorProtocolError("model returned an invalid batch-one token sequence")
    sequence = sequences.detach().to("cpu", dtype=torch.int64).flatten().contiguous()
    prompt = prompt_ids.detach().to("cpu", dtype=torch.int64).flatten()
    if sequence.numel() >= prompt.numel() and torch.equal(
        sequence[: prompt.numel()], prompt
    ):
        return sequence[prompt.numel() :].contiguous()
    return sequence


def _queue_put(queue: Any, message: TensorMessage) -> None:
    try:
        queue.put(message, timeout=2.0)
    except Full as exc:
        raise RuntimeError("bounded response queue is saturated") from exc


def _debug_worker_error(stage: str, error: BaseException) -> None:
    if os.environ.get("COGNI_AGENT_DEBUG") != "1":
        return
    detail = " ".join(f"{type(error).__name__}: {error}".replace("\x00", "").split())
    print(f"[cogni-worker:{stage}] {detail[:256]}", file=sys.stderr, flush=True)


def _cognitive_state(
    input_ids: Tensor, attention_mask: Tensor, sequence_budget: int
) -> Tensor:
    """Derive bounded BIO-HAMA telemetry from token tensors only."""

    mask = attention_mask.to(dtype=torch.float32)
    active = mask.sum(dim=1)
    load = (active / float(sequence_budget)).clamp(0.0, 1.0)
    attention = mask.mean(dim=1).clamp(0.0, 1.0)
    if input_ids.shape[1] > 1:
        pair_mask = mask[:, 1:] * mask[:, :-1]
        changes = (input_ids[:, 1:] != input_ids[:, :-1]).to(mask.dtype) * pair_mask
        uncertainty = (changes.sum(dim=1) / pair_mask.sum(dim=1).clamp_min(1.0)).clamp(
            0.0, 1.0
        )
    else:
        uncertainty = torch.zeros_like(load)
    memory = load
    affect = torch.zeros_like(load)
    return cognitive_state_tensor(memory, affect, attention, uncertainty, load)


def _run_core_turn(
    pipeline: CoreTurnPipeline,
    input_ids: Tensor,
    attention_mask: Tensor,
    *,
    sequence_budget: int,
    estimated_workspace_bytes: int,
) -> None:
    result = pipeline.run(
        CoreTurnRequest(
            inputs=input_ids,
            cognitive_state=_cognitive_state(
                input_ids, attention_mask, sequence_budget
            ),
            backbone_kwargs={"attention_mask": attention_mask},
            # Untrained Fast Weight remains inactive. Admission, external
            # quality, and OOD calibration are mandatory before a later turn
            # can supply either FastWeight field.
            fast_weight=None,
            compile_fast_weight=None,
            estimated_workspace_bytes=estimated_workspace_bytes,
        )
    )
    telemetry = getattr(result, "telemetry", None)
    if getattr(telemetry, "advisory_only", None) is not True:
        raise RuntimeError("Cogni-Core auxiliaries crossed the advisory boundary")


def _worker_main(
    model_factory: Callable[[], nn.Module],
    core_pipeline_factory: Callable[[nn.Module], CoreTurnPipeline] | None,
    request_queue: Any,
    response_queue: Any,
    cancel_event: Any,
    ready_event: Any,
    failed_event: Any,
    max_input_tokens: int,
    core_workspace_bytes: int,
) -> None:
    _force_offline_environment()
    try:
        model = model_factory()
        _prepare_base_model(model)
        pipeline = (
            None if core_pipeline_factory is None else core_pipeline_factory(model)
        )
        if pipeline is not None and not callable(getattr(pipeline, "run", None)):
            raise TypeError("core pipeline factory must return an object with run()")
        base_signature = _parameter_signature(model)
        device = _model_device(model)
    except BaseException:
        failed_event.set()
        return
    ready_event.set()

    while True:
        try:
            raw = request_queue.get()
            request = parse_request(raw)
        except BaseException:
            try:
                _queue_put(
                    response_queue,
                    make_response(0, STATUS_INVALID_REQUEST, final=True),
                )
            except BaseException:
                return
            continue
        if request is None:
            return

        streamer = _TensorResponseStreamer(
            response_queue,
            request.request_id,
            request.input_ids,
            request.max_new_tokens,
            cancel_event,
        )
        status = STATUS_OK
        remaining = torch.empty(0, dtype=torch.int64)
        stage = "request"
        try:
            input_ids = request.input_ids.to(device)
            attention_mask = request.attention_mask.to(device)
            if pipeline is not None:
                stage = "core_pipeline"
                if cancel_event.is_set():
                    status = STATUS_CANCELLED
                else:
                    # CTS guards input mutation through Tensor._version; unlike
                    # inference_mode, no_grad preserves that safety counter.
                    with torch.no_grad():
                        _run_core_turn(
                            pipeline,
                            input_ids,
                            attention_mask,
                            sequence_budget=max_input_tokens,
                            estimated_workspace_bytes=core_workspace_bytes,
                        )
                    if cancel_event.is_set():
                        status = STATUS_CANCELLED
            if status == STATUS_CANCELLED:
                output = input_ids
            else:
                stage = "base_generate"
                # The advisory pipeline never supplies logits or a latent to
                # decoding. Base Gemma remains the sole answer authority.
                with torch.inference_mode():
                    output = model.generate(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        max_new_tokens=request.max_new_tokens,
                        do_sample=False,
                        use_cache=False,
                        streamer=streamer,
                        stopping_criteria=_stopping_criteria(
                            cancel_event, request.stop_token_ids
                        ),
                    )
            stage = "token_postcheck"
            generated = _generated_suffix(output, request.input_ids)
            if generated.numel() > request.max_new_tokens:
                raise TensorProtocolError("model exceeded max_new_tokens")
            emitted = torch.tensor(streamer.emitted, dtype=torch.int64)
            if emitted.numel():
                if generated.numel() < emitted.numel() or not torch.equal(
                    generated[: emitted.numel()], emitted
                ):
                    raise TensorProtocolError("streamed and returned tokens disagree")
                remaining = generated[emitted.numel() :].contiguous()
            else:
                remaining = generated
            status = STATUS_CANCELLED if cancel_event.is_set() else STATUS_OK
        except BaseException as error:
            _debug_worker_error(stage, error)
            status = STATUS_MODEL_ERROR
            remaining = torch.empty(0, dtype=torch.int64)

        if _parameter_signature(model) != base_signature:
            status = STATUS_BASE_MUTATED
            remaining = torch.empty(0, dtype=torch.int64)
        total = len(streamer.emitted) + int(remaining.numel())
        try:
            _queue_put(
                response_queue,
                make_response(
                    request.request_id,
                    status,
                    remaining,
                    generated_total=total,
                    final=True,
                ),
            )
        except BaseException:
            return
        if status == STATUS_BASE_MUTATED:
            # Never continue serving from a model whose invariant was broken.
            return


class ModelService:
    """Own one resident model process and at most one outstanding generation."""

    def __init__(
        self,
        tokenizer: Any,
        model_factory: Callable[[], nn.Module],
        *,
        core_pipeline_factory: Callable[[nn.Module], CoreTurnPipeline] | None = None,
        core_workspace_mib: int = 512,
        max_input_tokens: int = 4_096,
        max_new_tokens: int = 512,
        max_prompt_chars: int = 32_000,
        max_response_chars: int = 32_000,
        startup_timeout: float = 180.0,
        request_timeout: float = 180.0,
        cancellation_timeout: float = 10.0,
        start_method: str = "spawn",
    ) -> None:
        if not callable(tokenizer) or not callable(getattr(tokenizer, "decode", None)):
            raise TypeError("tokenizer must be callable and provide decode()")
        if isinstance(model_factory, (str, bytes, os.PathLike)) or not callable(
            model_factory
        ):
            raise TypeError(
                "model_factory must be a callable, never a model id or path"
            )
        if core_pipeline_factory is not None and not callable(core_pipeline_factory):
            raise TypeError("core_pipeline_factory must be callable")
        if (
            not isinstance(core_workspace_mib, int)
            or isinstance(core_workspace_mib, bool)
            or not 1 <= core_workspace_mib <= 4_096
        ):
            raise ValueError("core_workspace_mib must be in [1, 4096]")
        if not 1 <= max_input_tokens <= HARD_MAX_INPUT_TOKENS:
            raise ValueError("max_input_tokens exceeds the protocol bound")
        if (
            isinstance(model_factory, LocalGemmaModelFactory)
            and max_input_tokens > 4_096
        ):
            raise ValueError(
                "the production CoreTurnPipeline caps local Gemma chat at 4096 tokens"
            )
        if not 1 <= max_new_tokens <= HARD_MAX_NEW_TOKENS:
            raise ValueError("max_new_tokens exceeds the protocol bound")
        if not 1 <= max_prompt_chars <= HARD_MAX_PROMPT_CHARS:
            raise ValueError("max_prompt_chars exceeds the hard bound")
        if not 1 <= max_response_chars <= HARD_MAX_RESPONSE_CHARS:
            raise ValueError("max_response_chars exceeds the hard bound")
        if startup_timeout <= 0 or request_timeout <= 0 or cancellation_timeout <= 0:
            raise ValueError("service timeouts must be positive")
        self.tokenizer = tokenizer
        self.model_factory = model_factory
        self.core_pipeline_factory = (
            LocalGemmaCorePipelineFactory()
            if core_pipeline_factory is None
            and isinstance(model_factory, LocalGemmaModelFactory)
            else core_pipeline_factory
        )
        self.core_workspace_bytes = int(core_workspace_mib) * 1024**2
        self.max_input_tokens = int(max_input_tokens)
        self.max_new_tokens = int(max_new_tokens)
        self.max_prompt_chars = int(max_prompt_chars)
        self.max_response_chars = int(max_response_chars)
        self.startup_timeout = float(startup_timeout)
        self.request_timeout = float(request_timeout)
        self.cancellation_timeout = float(cancellation_timeout)
        self._context = multiprocessing.get_context(start_method)
        self._request_lock = Lock()
        self._state_lock = RLock()
        self._next_request_id = 1
        self._active_request_id: int | None = None
        self._process: Any = None
        self._request_queue: Any = None
        self._response_queue: Any = None
        self._cancel_event: Any = None
        self._ready_event: Any = None
        self._failed_event: Any = None

    @classmethod
    def for_local_gemma(
        cls,
        model_path: str | Path,
        *,
        vram_limit_gib: float = 16.7,
        tokenizer_kwargs: Mapping[str, Any] | None = None,
        **service_kwargs: Any,
    ) -> ModelService:
        root = verify_local_gemma_path(model_path)
        tokenizer = load_local_tokenizer(root, tokenizer_kwargs=tokenizer_kwargs)
        return cls(
            tokenizer,
            LocalGemmaModelFactory(str(root), vram_limit_gib),
            **service_kwargs,
        )

    @property
    def is_running(self) -> bool:
        process = self._process
        return bool(process is not None and process.is_alive())

    @property
    def worker_pid(self) -> int | None:
        process = self._process
        return None if process is None else process.pid

    @property
    def active_request_id(self) -> int | None:
        with self._state_lock:
            return self._active_request_id

    def start(self) -> ModelService:
        with self._state_lock:
            if self.is_running:
                return self
            self._request_queue = self._context.Queue(maxsize=2)
            self._response_queue = self._context.Queue(
                maxsize=DEFAULT_RESPONSE_QUEUE_SIZE
            )
            self._cancel_event = self._context.Event()
            self._ready_event = self._context.Event()
            self._failed_event = self._context.Event()
            self._process = self._context.Process(
                target=_worker_main,
                args=(
                    self.model_factory,
                    self.core_pipeline_factory,
                    self._request_queue,
                    self._response_queue,
                    self._cancel_event,
                    self._ready_event,
                    self._failed_event,
                    self.max_input_tokens,
                    self.core_workspace_bytes,
                ),
                name="cogni-local-model",
                daemon=True,
            )
            self._process.start()
        deadline = monotonic() + self.startup_timeout
        while monotonic() < deadline:
            if self._ready_event.is_set():
                return self
            if self._failed_event.is_set() or not self.is_running:
                self.stop()
                raise WorkerStartupError("local model worker failed to initialize")
            sleep(0.01)
        self.stop()
        raise WorkerStartupError("local model worker startup timed out")

    def iter_generate_tokens(
        self,
        prompt: str,
        *,
        max_new_tokens: int | None = None,
        stop_token_ids: Tensor | None = None,
        timeout: float | None = None,
    ) -> Iterator[GenerationChunk]:
        self.start()
        if not self._request_lock.acquire(blocking=False):
            raise ServiceBusyError("one generation request is already active")
        completed = False
        request_id = 0
        try:
            input_ids, attention_mask = self._tokenize(prompt)
            requested = (
                self.max_new_tokens if max_new_tokens is None else max_new_tokens
            )
            if (
                not isinstance(requested, int)
                or isinstance(requested, bool)
                or not 1 <= requested <= self.max_new_tokens
            ):
                raise RequestLimitError("max_new_tokens exceeds the service budget")
            with self._state_lock:
                request_id = self._next_request_id
                self._next_request_id += 1
                self._active_request_id = request_id
                self._cancel_event.clear()
            message = make_generate_request(
                request_id,
                input_ids,
                attention_mask,
                max_new_tokens=requested,
                stop_token_ids=stop_token_ids,
            )
            try:
                self._request_queue.put(message, timeout=1.0)
            except Full as exc:
                raise ServiceBusyError("bounded request queue is full") from exc
            wait_seconds = self.request_timeout if timeout is None else float(timeout)
            if wait_seconds <= 0:
                raise ValueError("request timeout must be positive")
            deadline = monotonic() + wait_seconds
            while True:
                remaining = deadline - monotonic()
                if remaining <= 0:
                    self._cancel_event.set()
                    raise TimeoutError("local generation exceeded its deadline")
                if not self.is_running:
                    raise WorkerExecutionError(
                        "local model worker stopped unexpectedly"
                    )
                try:
                    raw = self._response_queue.get(timeout=min(0.1, remaining))
                except Empty:
                    continue
                try:
                    frame = parse_response(raw)
                except TensorProtocolError as exc:
                    raise WorkerExecutionError(
                        "worker response protocol failed"
                    ) from exc
                if frame.request_id != request_id:
                    raise WorkerExecutionError("worker response ownership changed")
                if frame.status == STATUS_OK:
                    chunk = GenerationChunk(
                        request_id,
                        frame.token_ids,
                        frame.generated_total,
                        frame.final,
                        False,
                    )
                    yield chunk
                    if frame.final:
                        completed = True
                        return
                    continue
                if frame.status == STATUS_CANCELLED:
                    completed = True
                    yield GenerationChunk(
                        request_id,
                        frame.token_ids,
                        frame.generated_total,
                        True,
                        True,
                    )
                    return
                if frame.status == STATUS_BASE_MUTATED:
                    completed = True
                    raise BaseModelMutationError("base-model immutability check failed")
                if frame.status in {STATUS_INVALID_REQUEST, STATUS_MODEL_ERROR}:
                    completed = True
                    raise WorkerExecutionError("local model worker rejected generation")
                completed = True
                raise WorkerExecutionError("local model worker returned unknown status")
        finally:
            if request_id and not completed:
                self._cancel_event.set()
                self._drain_cancelled_request(request_id)
            with self._state_lock:
                if self._active_request_id == request_id:
                    self._active_request_id = None
            self._request_lock.release()

    def generate(
        self,
        prompt: str,
        *,
        max_new_tokens: int | None = None,
        stop_token_ids: Tensor | None = None,
        timeout: float | None = None,
    ) -> GenerationResult:
        chunks: list[Tensor] = []
        request_id = 0
        cancelled = False
        for chunk in self.iter_generate_tokens(
            prompt,
            max_new_tokens=max_new_tokens,
            stop_token_ids=stop_token_ids,
            timeout=timeout,
        ):
            request_id = chunk.request_id
            if chunk.token_ids.numel():
                chunks.append(chunk.token_ids)
            cancelled = cancelled or chunk.cancelled
        tokens = torch.cat(chunks) if chunks else torch.empty(0, dtype=torch.int64)
        if tokens.numel() > self.max_new_tokens:
            raise WorkerExecutionError("worker exceeded the aggregate token budget")
        if cancelled:
            raise GenerationCancelled("local generation was cancelled")
        text = self.tokenizer.decode(
            tokens.tolist(),
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        if not isinstance(text, str) or len(text) > self.max_response_chars:
            raise RequestLimitError("decoded response exceeds its character bound")
        return GenerationResult(request_id, tokens, text)

    def cancel(self, request_id: int | None = None) -> bool:
        with self._state_lock:
            active = self._active_request_id
            if active is None or (request_id is not None and request_id != active):
                return False
            self._cancel_event.set()
            return True

    def stop(self, timeout: float = 10.0) -> None:
        if timeout <= 0:
            raise ValueError("stop timeout must be positive")
        process = self._process
        if process is None:
            return
        if self._cancel_event is not None:
            self._cancel_event.set()
        if process.is_alive() and self._request_queue is not None:
            try:
                self._request_queue.put(make_stop_request(), timeout=0.2)
            except Full:
                pass
        process.join(timeout)
        if process.is_alive():
            process.terminate()
            process.join(2.0)
        for queue in (self._request_queue, self._response_queue):
            if queue is not None:
                queue.close()
                queue.join_thread()
        with self._state_lock:
            self._process = None
            self._request_queue = None
            self._response_queue = None
            self._cancel_event = None
            self._ready_event = None
            self._failed_event = None
            self._active_request_id = None

    def _tokenize(self, prompt: str) -> tuple[Tensor, Tensor]:
        if not isinstance(prompt, str):
            raise TypeError("prompt must be text at the control boundary")
        if not prompt or len(prompt) > self.max_prompt_chars:
            raise RequestLimitError("prompt exceeds its character bound")
        if any(
            ord(character) < 32 and character not in "\t\r\n" for character in prompt
        ):
            raise RequestLimitError("prompt contains unsupported control characters")
        encoded = self.tokenizer(prompt, return_tensors="pt", truncation=False)
        if not isinstance(encoded, Mapping) or "input_ids" not in encoded:
            raise RequestLimitError("tokenizer did not return input_ids")
        input_ids = torch.as_tensor(encoded["input_ids"]).detach()
        if input_ids.ndim != 2 or input_ids.shape[0] != 1:
            raise RequestLimitError("tokenizer output must be batch one")
        if not 1 <= input_ids.shape[1] <= self.max_input_tokens:
            raise RequestLimitError("tokenized prompt exceeds its token bound")
        input_ids = input_ids.to("cpu", dtype=torch.int64).contiguous()
        raw_mask = encoded.get("attention_mask")
        attention_mask = (
            torch.ones_like(input_ids)
            if raw_mask is None
            else torch.as_tensor(raw_mask)
            .detach()
            .to("cpu", dtype=torch.int64)
            .contiguous()
        )
        return input_ids, attention_mask

    def _drain_cancelled_request(self, request_id: int) -> None:
        deadline = monotonic() + self.cancellation_timeout
        while self.is_running and monotonic() < deadline:
            try:
                frame = parse_response(self._response_queue.get(timeout=0.05))
            except Empty:
                continue
            except TensorProtocolError:
                return
            if frame.request_id == request_id and frame.final:
                return

    def __enter__(self) -> ModelService:
        return self.start()

    def __exit__(self, _exc_type: Any, _exc: Any, _traceback: Any) -> None:
        self.stop()


__all__ = [
    "BaseModelMutationError",
    "GenerationCancelled",
    "GenerationChunk",
    "GenerationResult",
    "HARD_MAX_PROMPT_CHARS",
    "HARD_MAX_RESPONSE_CHARS",
    "LocalGemmaCorePipelineFactory",
    "LocalGemmaModelFactory",
    "ModelService",
    "ModelServiceError",
    "RequestLimitError",
    "ServiceBusyError",
    "WorkerExecutionError",
    "WorkerStartupError",
    "load_local_tokenizer",
]
