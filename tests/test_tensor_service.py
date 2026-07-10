from __future__ import annotations

from time import sleep
import unittest

import torch
from torch import nn

from cogni_os.tensor_service import (
    InvalidPayloadError,
    ProtocolError,
    ServiceCapacityError,
    ServiceNotRunningError,
    ServicePausedError,
    ServiceTimeoutError,
    TensorService,
    WorkerExecutionError,
    _request_message,
)


class ToyTensorModule(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("scale", torch.tensor(3.0))
        self.received_use_cache = True

    def forward(self, value: torch.Tensor, *, use_cache: bool = True) -> torch.Tensor:
        self.received_use_cache = use_cache
        if self.training or use_cache:
            raise RuntimeError("worker did not enforce inference-only module settings")
        if torch.any(value == -99):
            raise RuntimeError("private worker detail")
        return value * self.scale


class SlowTensorModule(nn.Module):
    def forward(self, value: torch.Tensor) -> torch.Tensor:
        sleep(0.2)
        return value


class BadFactory:
    def __call__(self) -> nn.Module:
        raise RuntimeError("must not cross the service boundary")


class TestTensorProtocol(unittest.TestCase):
    def test_message_is_exactly_four_cpu_tensors(self) -> None:
        message = _request_message(1, 7, torch.ones(2))
        self.assertIsInstance(message, tuple)
        self.assertEqual(len(message), 4)
        self.assertTrue(all(isinstance(item, torch.Tensor) for item in message))
        self.assertTrue(all(item.device.type == "cpu" for item in message))


class TestTensorService(unittest.TestCase):
    def make_service(self, factory=ToyTensorModule, **kwargs) -> TensorService:
        options = {
            "queue_capacity": 3,
            "max_outstanding": 2,
            "request_timeout": 2.0,
            "startup_timeout": 10.0,
        }
        options.update(kwargs)
        return TensorService(factory, **options)

    def test_start_infer_stop_and_restart(self) -> None:
        service = self.make_service()
        service.start()
        first_pid = service.worker_pid
        self.assertTrue(service.is_running)
        self.assertTrue(
            torch.equal(
                service.infer(torch.tensor([1.0, 2.0])), torch.tensor([3.0, 6.0])
            )
        )

        service.restart()
        self.assertTrue(service.is_running)
        self.assertNotEqual(service.worker_pid, first_pid)
        self.assertTrue(
            torch.equal(service.infer(torch.tensor([4.0])), torch.tensor([12.0]))
        )
        service.stop()
        self.assertFalse(service.is_running)
        with self.assertRaises(ServiceNotRunningError):
            service.infer(torch.ones(1))

    def test_pause_rejects_inference_then_resume_restores_it(self) -> None:
        with self.make_service() as service:
            service.pause()
            with self.assertRaises(ServicePausedError):
                service.infer(torch.ones(1))
            service.resume()
            self.assertTrue(
                torch.equal(service.infer(torch.ones(1)), torch.tensor([3.0]))
            )

    def test_worker_rejects_invalid_payload_and_hides_exceptions(self) -> None:
        with self.make_service() as service:
            with self.assertRaises(InvalidPayloadError):
                service.infer(torch.empty(0))
            with self.assertRaises(InvalidPayloadError):
                service.infer(torch.tensor([float("nan")]))
            with self.assertRaises(WorkerExecutionError) as caught:
                service.infer(torch.tensor([-99.0]))
            self.assertNotIn("private worker detail", str(caught.exception))

    def test_bounded_outstanding_requests(self) -> None:
        with self.make_service(max_outstanding=1) as service:
            first = service.submit(torch.ones(1))
            self.assertEqual(service.outstanding, 1)
            with self.assertRaises(ServiceCapacityError):
                service.submit(torch.ones(1))
            self.assertTrue(torch.equal(service.receive(first), torch.tensor([3.0])))
            self.assertEqual(service.outstanding, 0)

    def test_timeout_is_bounded_and_service_can_restart(self) -> None:
        service = self.make_service(SlowTensorModule)
        service.start()
        with self.assertRaises(ServiceTimeoutError):
            service.infer(torch.ones(1), timeout=0.01)
        service.restart()
        self.assertTrue(torch.equal(service.infer(torch.ones(1)), torch.ones(1)))
        service.stop()

    def test_invalid_numeric_opcode_is_parent_protocol_error(self) -> None:
        with self.make_service() as service:
            request_id = service._submit_code(999, torch.empty(0))
            with self.assertRaises(ProtocolError):
                service.receive(request_id)

    def test_factory_failure_crosses_as_numeric_status(self) -> None:
        service = self.make_service(BadFactory())
        with self.assertRaises(WorkerExecutionError) as caught:
            service.start()
        self.assertNotIn("must not cross", str(caught.exception))
        self.assertFalse(service.is_running)


if __name__ == "__main__":
    unittest.main()
