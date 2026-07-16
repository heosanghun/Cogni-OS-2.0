from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
import os
from pathlib import Path
from types import SimpleNamespace
import unittest

from PIL import Image
import torch
from torch import nn

from cogni_agent.model_service import ModelService, RequestLimitError
from cogni_agent.multimodal import VerifiedGemma4MultimodalProcessor
from cogni_agent.protocol import (
    HARD_MAX_IMAGE_TENSOR_BYTES,
    IMAGE_FIELD_MM_TOKEN_TYPE_IDS,
    IMAGE_FIELD_PIXEL_VALUES,
    IMAGE_FIELD_POSITION_IDS,
    MODALITY_IMAGE,
    TensorProtocolError,
    make_generate_request,
    parse_request,
)


def _image_fields(
    input_length: int = 4,
) -> tuple[tuple[int, torch.Tensor], ...]:
    return (
        (
            IMAGE_FIELD_MM_TOKEN_TYPE_IDS,
            torch.zeros((1, input_length), dtype=torch.int64),
        ),
        (
            IMAGE_FIELD_PIXEL_VALUES,
            torch.linspace(-1.0, 1.0, 32, dtype=torch.float32).reshape(1, 8, 4),
        ),
        (
            IMAGE_FIELD_POSITION_IDS,
            torch.arange(16, dtype=torch.int64).reshape(1, 8, 2),
        ),
    )


def _png() -> bytes:
    output = BytesIO()
    Image.new("RGB", (16, 16), "navy").save(output, format="PNG")
    return output.getvalue()


class _Tokenizer:
    def __call__(self, text, **_kwargs):
        ids = torch.ones((1, max(1, len(text))), dtype=torch.int64)
        return {"input_ids": ids, "attention_mask": torch.ones_like(ids)}

    def decode(self, token_ids, **_kwargs):
        return " ".join(map(str, token_ids))


class _Processor:
    image_token = "<|image|>"

    def apply_chat_template(self, conversation, **_kwargs):
        if conversation[0]["content"][0]["type"] != "image":
            raise AssertionError("image chat template was not used")
        input_ids = torch.tensor([[7, 8, 9, 10]], dtype=torch.int64)
        fields = dict(_image_fields(input_length=4))
        return {
            "input_ids": input_ids,
            "attention_mask": torch.ones_like(input_ids),
            "mm_token_type_ids": fields[IMAGE_FIELD_MM_TOKEN_TYPE_IDS],
            "pixel_values": fields[IMAGE_FIELD_PIXEL_VALUES],
            "image_position_ids": fields[IMAGE_FIELD_POSITION_IDS],
        }


class _ImageAwareModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1))
        self.config = SimpleNamespace(use_cache=True)

    def generate(self, **kwargs):
        expected = {
            "pixel_values": (torch.float32, (1, 8, 4)),
            "mm_token_type_ids": (torch.int64, (1, 4)),
            "image_position_ids": (torch.int64, (1, 8, 2)),
        }
        for name, (dtype, shape) in expected.items():
            value = kwargs.get(name)
            if (
                not isinstance(value, torch.Tensor)
                or value.dtype != dtype
                or tuple(value.shape) != shape
                or not value.is_contiguous()
            ):
                raise RuntimeError("worker did not reconstruct bounded image tensors")
        if kwargs.get("do_sample") is not False or kwargs.get("use_cache") is not False:
            raise RuntimeError("decode policy is not bounded")
        input_ids = kwargs["input_ids"]
        streamer = kwargs["streamer"]
        streamer.put(input_ids)
        token = torch.tensor([[42]], dtype=torch.int64, device=input_ids.device)
        streamer.put(token)
        streamer.end()
        return torch.cat((input_ids, token), dim=1)


@dataclass(frozen=True)
class _ImageFactory:
    parent_pid: int

    def __call__(self):
        if os.getpid() == self.parent_pid:
            raise RuntimeError("model must not load in the controller")
        return _ImageAwareModel()


class TestMultimodalTensorProtocol(unittest.TestCase):
    def test_image_fields_round_trip_as_exactly_four_int64_cpu_tensors(self) -> None:
        message = make_generate_request(
            1,
            torch.tensor([[7, 8, 9, 10]], dtype=torch.int64),
            torch.ones((1, 4), dtype=torch.int64),
            max_new_tokens=2,
            image_tensors=_image_fields(),
        )

        self.assertEqual(len(message), 4)
        self.assertTrue(all(value.device.type == "cpu" for value in message))
        self.assertTrue(all(value.dtype == torch.int64 for value in message))
        self.assertTrue(all(value.is_contiguous() for value in message))
        request = parse_request(message)
        self.assertIsNotNone(request)
        self.assertEqual(request.modality, MODALITY_IMAGE)
        actual = dict(request.image_tensors)
        for code, expected in _image_fields():
            self.assertTrue(torch.equal(actual[code], expected))

    def test_nonfinite_noncontiguous_missing_and_oversized_images_fail_closed(
        self,
    ) -> None:
        input_ids = torch.ones((1, 4), dtype=torch.int64)
        mask = torch.ones_like(input_ids)
        cases = []
        missing = _image_fields()[:-1]
        cases.append(missing)
        nonfinite = list(_image_fields())
        nonfinite[1] = (
            IMAGE_FIELD_PIXEL_VALUES,
            torch.full((1, 8, 4), float("nan"), dtype=torch.float32),
        )
        cases.append(tuple(nonfinite))
        noncontiguous = list(_image_fields())
        noncontiguous[1] = (
            IMAGE_FIELD_PIXEL_VALUES,
            torch.ones((1, 4, 8), dtype=torch.float32).transpose(1, 2),
        )
        cases.append(tuple(noncontiguous))
        oversized = list(_image_fields())
        oversized[1] = (
            IMAGE_FIELD_PIXEL_VALUES,
            torch.ones(
                (1, HARD_MAX_IMAGE_TENSOR_BYTES // 4 + 1, 1),
                dtype=torch.float32,
            ),
        )
        cases.append(tuple(oversized))

        for image_tensors in cases:
            with self.subTest(shape=tuple(image_tensors[1][1].shape)):
                with self.assertRaises(TensorProtocolError):
                    make_generate_request(
                        2,
                        input_ids,
                        mask,
                        max_new_tokens=1,
                        image_tensors=image_tensors,
                    )

    def test_descriptor_tampering_is_rejected(self) -> None:
        message = list(
            make_generate_request(
                3,
                torch.ones((1, 4), dtype=torch.int64),
                None,
                max_new_tokens=1,
                image_tensors=_image_fields(),
            )
        )
        control = message[3].clone()
        # 3 protocol scalars + two SHA-256 digests. The second fixed-size
        # descriptor is pixel_values; its numel field is index 3.
        control[3 + 64 + 8 + 3] += 1
        message[3] = control
        with self.assertRaises(TensorProtocolError):
            parse_request(tuple(message))


class TestResidentImageGeneration(unittest.TestCase):
    def test_verified_image_tensors_cross_worker_and_reach_single_model(self) -> None:
        service = ModelService(
            _Tokenizer(),
            _ImageFactory(os.getpid()),
            max_input_tokens=32,
            max_new_tokens=4,
            startup_timeout=20.0,
            request_timeout=20.0,
            multimodal_processor_config=(Path(__file__), Path(__file__)),
        )
        processor = object.__new__(VerifiedGemma4MultimodalProcessor)
        processor.processor = _Processor()
        service._multimodal_processor = processor
        try:
            result = service.generate(
                "이미지를 설명하세요.",
                image_content=_png(),
                max_new_tokens=1,
                decode_mode="strict",
            )
        finally:
            service.stop()

        self.assertEqual(result.token_ids.tolist(), [42])
        self.assertEqual(result.text, "42")

    def test_image_request_without_verified_processor_fails_before_model_start(
        self,
    ) -> None:
        service = ModelService(
            _Tokenizer(),
            _ImageFactory(os.getpid()),
            max_input_tokens=32,
            max_new_tokens=4,
        )
        with self.assertRaisesRegex(RequestLimitError, "verified"):
            service.generate("설명", image_content=_png())
        self.assertIsNone(service.worker_pid)


if __name__ == "__main__":
    unittest.main()
