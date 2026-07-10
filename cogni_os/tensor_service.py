"""Bounded tensor-only local service boundary for the Cogni-Core owner.

The data plane is deliberately narrower than Python RPC: every request and
response is a four-item tuple of CPU tensors.  Strings, JSON, arbitrary Python
objects, and network sockets never cross the hot-path boundary.

Deployment invariant
--------------------
Only the spawned worker may construct or move the real backbone to CUDA.  Do
not instantiate a GPU model in the parent and pass it as the factory closure:
``spawn`` would create another CUDA owner and can duplicate VRAM.  Production
factories must load the local model inside the worker process.  One service
worker therefore owns one GPU-backed Cogni-Core; Cogni-Flow remains the CPU
control plane.
"""

from __future__ import annotations

from collections.abc import Callable
from enum import IntEnum
import inspect
from queue import Empty, Full
from threading import Lock, RLock
from time import monotonic
from typing import Any, TypeAlias

import torch
from torch import Tensor, nn
import torch.multiprocessing as mp


class OpCode(IntEnum):
    """Numeric operations accepted by the local worker."""

    READY = 0
    INFER = 1
    PAUSE = 2
    RESUME = 3
    STOP = 4


class StatusCode(IntEnum):
    """Numeric-only worker outcomes translated by the parent control plane."""

    OK = 0
    PAUSED = 1
    INVALID_PAYLOAD = 2
    INVALID_OPCODE = 3
    WORKER_EXCEPTION = 4
    PROTOCOL_ERROR = 5


TensorMessage: TypeAlias = tuple[Tensor, Tensor, Tensor, Tensor]
ModuleFactory: TypeAlias = Callable[[], nn.Module]


class TensorServiceError(RuntimeError):
    """Base class for parent-side service errors."""


class ServiceNotRunningError(TensorServiceError):
    """Raised when an operation requires a live worker."""


class ServiceCapacityError(TensorServiceError):
    """Raised before the configured outstanding-request bound is exceeded."""


class ServiceTimeoutError(TensorServiceError):
    """Raised when the bounded wait for a response expires."""


class ServicePausedError(TensorServiceError):
    """Raised when inference is requested while the core is paused."""


class InvalidPayloadError(TensorServiceError):
    """Raised when the worker rejects an inference payload."""


class ProtocolError(TensorServiceError):
    """Raised for an invalid opcode or malformed worker response."""


class WorkerExecutionError(TensorServiceError):
    """Raised from a numeric worker failure without crossing exception text."""


_INT_DTYPES = {
    torch.int8,
    torch.int16,
    torch.int32,
    torch.int64,
    torch.uint8,
}


def _scalar(value: int) -> Tensor:
    return torch.tensor(value, dtype=torch.int64)


def _empty_payload() -> Tensor:
    return torch.empty(0, dtype=torch.float32)


def _request_message(op_code: int, request_id: int, payload: Tensor) -> TensorMessage:
    """Build the only request representation allowed on the data plane."""
    return (
        _scalar(op_code),
        _scalar(request_id),
        payload.detach().contiguous(),
        _scalar(StatusCode.OK),
    )


def _response_message(
    op_code: int, request_id: int, result: Tensor, status: StatusCode
) -> TensorMessage:
    """Build the only response representation allowed on the data plane."""
    return (
        _scalar(op_code),
        _scalar(request_id),
        result.detach().to(device="cpu").contiguous(),
        _scalar(status),
    )


def _read_integer_scalar(value: object) -> int | None:
    if not isinstance(value, Tensor):
        return None
    if value.device.type != "cpu" or value.dtype not in _INT_DTYPES:
        return None
    if value.numel() != 1:
        return None
    return int(value.item())


def _decode_message(message: object) -> tuple[int, int, Tensor, int] | None:
    if not isinstance(message, tuple) or len(message) != 4:
        return None
    if not all(isinstance(item, Tensor) for item in message):
        return None
    op_code = _read_integer_scalar(message[0])
    request_id = _read_integer_scalar(message[1])
    marker = _read_integer_scalar(message[3])
    payload = message[2]
    if op_code is None or request_id is None or marker is None:
        return None
    if payload.device.type != "cpu":
        return None
    return op_code, request_id, payload, marker


def _accepts_use_cache(module: nn.Module) -> bool:
    try:
        parameters = inspect.signature(module.forward).parameters.values()
    except (TypeError, ValueError):
        return False
    return any(
        parameter.name == "use_cache" or parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in parameters
    )


def _extract_result(output: object) -> Tensor | None:
    if isinstance(output, Tensor):
        return output
    if isinstance(output, (tuple, list)) and output:
        return output[0] if isinstance(output[0], Tensor) else None
    for attribute in ("logits", "last_hidden_state"):
        candidate = getattr(output, attribute, None)
        if isinstance(candidate, Tensor):
            return candidate
    return None


def _put_worker_response(
    response_queue: Any,
    shutdown_event: Any,
    message: TensorMessage,
) -> bool:
    """Bound response buffering while allowing forced parent shutdown."""
    while not shutdown_event.is_set():
        try:
            response_queue.put(message, timeout=0.1)
            return True
        except Full:
            continue
    return False


def _tensor_service_worker(
    module_factory: ModuleFactory,
    request_queue: Any,
    response_queue: Any,
    shutdown_event: Any,
    device: str,
) -> None:
    """Top-level spawn target; never move this function inside a class."""
    try:
        module = module_factory()
        if not isinstance(module, nn.Module):
            raise TypeError
        module = module.to(torch.device(device))
        module.eval()
        config = getattr(module, "config", None)
        if config is not None and hasattr(config, "use_cache"):
            config.use_cache = False
        pass_use_cache = _accepts_use_cache(module)
    except BaseException:
        _put_worker_response(
            response_queue,
            shutdown_event,
            _response_message(
                OpCode.READY,
                0,
                _empty_payload(),
                StatusCode.WORKER_EXCEPTION,
            ),
        )
        shutdown_event.wait(timeout=5.0)
        return

    if not _put_worker_response(
        response_queue,
        shutdown_event,
        _response_message(OpCode.READY, 0, _empty_payload(), StatusCode.OK),
    ):
        return

    paused = False
    while not shutdown_event.is_set():
        try:
            message = request_queue.get(timeout=0.1)
        except Empty:
            continue

        decoded = _decode_message(message)
        if decoded is None:
            if not _put_worker_response(
                response_queue,
                shutdown_event,
                _response_message(-1, -1, _empty_payload(), StatusCode.PROTOCOL_ERROR),
            ):
                return
            continue

        op_code, request_id, payload, marker = decoded
        if request_id <= 0 or marker != int(StatusCode.OK):
            if not _put_worker_response(
                response_queue,
                shutdown_event,
                _response_message(
                    op_code,
                    request_id,
                    _empty_payload(),
                    StatusCode.PROTOCOL_ERROR,
                ),
            ):
                return
            continue

        if (
            op_code
            in {
                int(OpCode.PAUSE),
                int(OpCode.RESUME),
                int(OpCode.STOP),
            }
            and payload.numel() != 0
        ):
            response = _response_message(
                op_code,
                request_id,
                _empty_payload(),
                StatusCode.INVALID_PAYLOAD,
            )
        elif op_code == int(OpCode.PAUSE):
            paused = True
            response = _response_message(
                op_code, request_id, _empty_payload(), StatusCode.OK
            )
        elif op_code == int(OpCode.RESUME):
            paused = False
            response = _response_message(
                op_code, request_id, _empty_payload(), StatusCode.OK
            )
        elif op_code == int(OpCode.STOP):
            response = _response_message(
                op_code, request_id, _empty_payload(), StatusCode.OK
            )
            if _put_worker_response(response_queue, shutdown_event, response):
                # Keep the tensor producer alive until the parent has decoded
                # the STOP response; this avoids shared-memory handle races.
                shutdown_event.wait(timeout=5.0)
            return
        elif op_code == int(OpCode.INFER):
            if paused:
                response = _response_message(
                    op_code,
                    request_id,
                    _empty_payload(),
                    StatusCode.PAUSED,
                )
            elif payload.numel() == 0 or (
                payload.is_floating_point() and not torch.isfinite(payload).all()
            ):
                response = _response_message(
                    op_code,
                    request_id,
                    _empty_payload(),
                    StatusCode.INVALID_PAYLOAD,
                )
            else:
                try:
                    owned_payload = payload.to(device=torch.device(device))
                    kwargs = {"use_cache": False} if pass_use_cache else {}
                    with torch.inference_mode():
                        output = module(owned_payload, **kwargs)
                    result = _extract_result(output)
                    if result is None or not torch.isfinite(result).all():
                        response = _response_message(
                            op_code,
                            request_id,
                            _empty_payload(),
                            StatusCode.INVALID_PAYLOAD,
                        )
                    else:
                        response = _response_message(
                            op_code, request_id, result, StatusCode.OK
                        )
                except BaseException:
                    # Exception types and messages deliberately remain inside
                    # the core process.  Only this numeric status crosses.
                    response = _response_message(
                        op_code,
                        request_id,
                        _empty_payload(),
                        StatusCode.WORKER_EXCEPTION,
                    )
        else:
            response = _response_message(
                op_code,
                request_id,
                _empty_payload(),
                StatusCode.INVALID_OPCODE,
            )

        if not _put_worker_response(response_queue, shutdown_event, response):
            return


class TensorService:
    """Spawned, bounded, tensor-only local inference service.

    The class is the Cogni-Flow control-plane facade.  It translates numeric
    statuses into local Python exceptions, but those exception objects and
    messages are never sent through the queues.
    """

    def __init__(
        self,
        module_factory: ModuleFactory,
        *,
        device: str | torch.device = "cpu",
        queue_capacity: int = 8,
        max_outstanding: int = 4,
        request_timeout: float = 5.0,
        startup_timeout: float = 15.0,
    ) -> None:
        if queue_capacity < 2:
            raise ValueError("queue_capacity must be at least two")
        if max_outstanding < 1 or max_outstanding >= queue_capacity:
            raise ValueError(
                "max_outstanding must be positive and below queue_capacity"
            )
        if request_timeout <= 0 or startup_timeout <= 0:
            raise ValueError("service timeouts must be positive")
        if not callable(module_factory):
            raise TypeError("module_factory must be callable")

        self.module_factory = module_factory
        self.device = str(torch.device(device))
        self.queue_capacity = int(queue_capacity)
        self.max_outstanding = int(max_outstanding)
        self.request_timeout = float(request_timeout)
        self.startup_timeout = float(startup_timeout)

        self._context = mp.get_context("spawn")
        self._request_queue: Any | None = None
        self._response_queue: Any | None = None
        self._shutdown_event: Any | None = None
        self._process: Any | None = None
        self._next_request_id = 1
        self._pending: dict[int, bool] = {}
        self._completed: dict[int, TensorMessage] = {}
        self._state_lock = RLock()
        self._receive_lock = Lock()

    @property
    def is_running(self) -> bool:
        return self._process is not None and self._process.is_alive()

    @property
    def worker_pid(self) -> int | None:
        return None if self._process is None else self._process.pid

    @property
    def outstanding(self) -> int:
        return len(self._pending)

    def start(self) -> TensorService:
        with self._state_lock:
            if self.is_running:
                return self
            self._dispose_transport(terminate=True)
            self._request_queue = self._context.Queue(self.queue_capacity)
            self._response_queue = self._context.Queue(self.queue_capacity)
            self._shutdown_event = self._context.Event()
            self._process = self._context.Process(
                target=_tensor_service_worker,
                args=(
                    self.module_factory,
                    self._request_queue,
                    self._response_queue,
                    self._shutdown_event,
                    self.device,
                ),
                daemon=True,
                name="cogni-core-tensor-owner",
            )
            self._process.start()
            self._pending.clear()
            self._completed.clear()
            self._next_request_id = 1

        try:
            response = self._read_wire(self.startup_timeout)
            decoded = _decode_message(response)
            if decoded is None:
                raise ProtocolError("worker returned a malformed READY response")
            op_code, request_id, _result, status = decoded
            if op_code != int(OpCode.READY) or request_id != 0:
                raise ProtocolError("worker returned an unexpected READY response")
            if status == int(StatusCode.WORKER_EXCEPTION):
                raise WorkerExecutionError("worker failed during module initialization")
            if status != int(StatusCode.OK):
                raise ProtocolError("worker returned an invalid READY status")
        except BaseException:
            self._dispose_transport(terminate=True)
            raise
        return self

    def submit(self, payload: Tensor) -> int:
        """Submit inference without exceeding the outstanding-request bound."""
        return self._submit_code(OpCode.INFER, payload)

    def infer(self, payload: Tensor, *, timeout: float | None = None) -> Tensor:
        request_id = self.submit(payload)
        return self.receive(request_id, timeout=timeout)

    def receive(self, request_id: int, *, timeout: float | None = None) -> Tensor:
        self._ensure_running()
        if request_id not in self._pending and request_id not in self._completed:
            raise KeyError(request_id)
        wait = self.request_timeout if timeout is None else float(timeout)
        if wait <= 0:
            raise ValueError("timeout must be positive")
        deadline = monotonic() + wait

        with self._receive_lock:
            while True:
                completed = self._completed.pop(request_id, None)
                if completed is not None:
                    self._pending.pop(request_id, None)
                    return self._translate_response(completed, request_id)
                remaining = deadline - monotonic()
                if remaining <= 0:
                    if request_id in self._pending:
                        self._pending[request_id] = False
                    raise ServiceTimeoutError(
                        f"request {request_id} exceeded its response deadline"
                    )
                try:
                    response = self._read_wire(remaining)
                except ServiceTimeoutError:
                    if request_id in self._pending:
                        self._pending[request_id] = False
                    raise ServiceTimeoutError(
                        f"request {request_id} exceeded its response deadline"
                    ) from None
                decoded = _decode_message(response)
                if decoded is None:
                    raise ProtocolError("worker returned a malformed tensor response")
                _op_code, response_id, _result, _status = decoded
                active = self._pending.get(response_id)
                if active is False:
                    self._pending.pop(response_id, None)
                    continue
                if active is None:
                    raise ProtocolError("worker returned an unknown request id")
                if response_id == request_id:
                    self._pending.pop(request_id, None)
                    return self._translate_response(response, request_id)
                self._completed[response_id] = response

    def pause(self, *, timeout: float | None = None) -> None:
        self._control_roundtrip(OpCode.PAUSE, timeout)

    def resume(self, *, timeout: float | None = None) -> None:
        self._control_roundtrip(OpCode.RESUME, timeout)

    def stop(self, *, timeout: float | None = None) -> None:
        with self._state_lock:
            if self._process is None:
                return
            if not self.is_running:
                self._dispose_transport(terminate=False)
                return
        wait = self.request_timeout if timeout is None else float(timeout)
        try:
            request_id = self._submit_code(OpCode.STOP, _empty_payload())
            self.receive(request_id, timeout=wait)
            if self._shutdown_event is not None:
                self._shutdown_event.set()
            if self._process is not None:
                self._process.join(timeout=wait)
        except (TensorServiceError, Full):
            pass
        finally:
            self._dispose_transport(terminate=True)

    def restart(self) -> TensorService:
        self.stop()
        return self.start()

    def __enter__(self) -> TensorService:
        return self.start()

    def __exit__(self, *_exc_info: object) -> None:
        self.stop()

    def _control_roundtrip(self, op_code: OpCode, timeout: float | None) -> None:
        request_id = self._submit_code(op_code, _empty_payload())
        self.receive(request_id, timeout=timeout)

    def _submit_code(self, op_code: int | OpCode, payload: Tensor) -> int:
        self._ensure_running()
        if not isinstance(payload, Tensor):
            raise TypeError("service payloads must be tensors")
        if payload.device.type != "cpu":
            raise InvalidPayloadError(
                "parent payloads must remain on CPU; the worker owns the device"
            )
        with self._state_lock:
            self._drain_available()
            if len(self._pending) >= self.max_outstanding:
                raise ServiceCapacityError("outstanding request capacity is exhausted")
            request_id = self._next_request_id
            self._next_request_id += 1
            message = _request_message(int(op_code), request_id, payload)
            if self._request_queue is None:
                raise ServiceNotRunningError("tensor service is not running")
            try:
                self._request_queue.put_nowait(message)
            except Full as error:
                raise ServiceCapacityError(
                    "request queue capacity is exhausted"
                ) from error
            self._pending[request_id] = True
            return request_id

    def _drain_available(self) -> None:
        if self._response_queue is None:
            return
        while True:
            try:
                response = self._response_queue.get_nowait()
            except Empty:
                return
            decoded = _decode_message(response)
            if decoded is None:
                continue
            _op_code, response_id, _result, _status = decoded
            active = self._pending.get(response_id)
            if active is False:
                self._pending.pop(response_id, None)
            elif active is True:
                self._completed[response_id] = response

    def _read_wire(self, timeout: float) -> TensorMessage:
        if self._response_queue is None:
            raise ServiceNotRunningError("tensor service transport is unavailable")
        try:
            return self._response_queue.get(timeout=timeout)
        except Empty as error:
            if self._process is not None and not self._process.is_alive():
                raise ServiceNotRunningError("tensor service worker exited") from error
            raise ServiceTimeoutError("tensor service response timed out") from error

    @staticmethod
    def _translate_response(response: TensorMessage, request_id: int) -> Tensor:
        decoded = _decode_message(response)
        if decoded is None:
            raise ProtocolError("worker returned a malformed tensor response")
        _op_code, response_id, result, status = decoded
        if response_id != request_id:
            raise ProtocolError("worker response request id does not match")
        if status == int(StatusCode.OK):
            return result
        if status == int(StatusCode.PAUSED):
            raise ServicePausedError("inference is disabled while the core is paused")
        if status == int(StatusCode.INVALID_PAYLOAD):
            raise InvalidPayloadError("worker rejected the tensor payload")
        if status == int(StatusCode.INVALID_OPCODE):
            raise ProtocolError("worker rejected the numeric opcode")
        if status == int(StatusCode.WORKER_EXCEPTION):
            raise WorkerExecutionError(
                f"worker execution failed for request {request_id}"
            )
        raise ProtocolError("worker returned a protocol failure status")

    def _ensure_running(self) -> None:
        if not self.is_running:
            raise ServiceNotRunningError("tensor service is not running")

    def _dispose_transport(self, *, terminate: bool) -> None:
        process = self._process
        event = self._shutdown_event
        if event is not None:
            event.set()
        if process is not None:
            if terminate and process.is_alive():
                process.terminate()
            process.join(timeout=2.0)
            if process.is_alive():
                process.kill()
                process.join(timeout=2.0)
        for transport in (self._request_queue, self._response_queue):
            if transport is not None:
                transport.close()
                transport.join_thread()
        self._process = None
        self._request_queue = None
        self._response_queue = None
        self._shutdown_event = None
        self._pending.clear()
        self._completed.clear()


__all__ = [
    "InvalidPayloadError",
    "OpCode",
    "ProtocolError",
    "ServiceCapacityError",
    "ServiceNotRunningError",
    "ServicePausedError",
    "ServiceTimeoutError",
    "StatusCode",
    "TensorMessage",
    "TensorService",
    "TensorServiceError",
    "WorkerExecutionError",
]
