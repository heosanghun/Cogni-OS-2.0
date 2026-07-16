from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
import os
from pathlib import Path
from types import SimpleNamespace
import unittest
import wave

import torch
from torch import nn

from cogni_agent.model_service import (
    ModelService,
    RequestLimitError,
    _bind_media_session_digest,
    _session_digest,
)
from cogni_agent.multimodal import VerifiedGemma4MultimodalProcessor
from cogni_agent.protocol import (
    AUDIO_FIELD_INPUT_FEATURES,
    AUDIO_FIELD_INPUT_FEATURES_MASK,
    AUDIO_FIELD_MM_TOKEN_TYPE_IDS,
    HARD_MAX_AUDIO_TENSOR_BYTES,
    IMAGE_FIELD_MM_TOKEN_TYPE_IDS,
    IMAGE_FIELD_PIXEL_VALUES,
    IMAGE_FIELD_POSITION_IDS,
    MODALITY_AUDIO,
    TensorProtocolError,
    make_generate_request,
    parse_request,
)


def _wav(frame_count: int = 1_600) -> bytes:
    output = BytesIO()
    with wave.open(output, "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(16_000)
        stream.writeframes(b"\0\0" * frame_count)
    return output.getvalue()


def _audio_fields(
    input_length: int = 10,
    frame_count: int = 10,
) -> tuple[tuple[int, torch.Tensor], ...]:
    return (
        (
            AUDIO_FIELD_MM_TOKEN_TYPE_IDS,
            torch.zeros((1, input_length), dtype=torch.int64),
        ),
        (
            AUDIO_FIELD_INPUT_FEATURES,
            torch.linspace(
                -1.0,
                1.0,
                frame_count * 128,
                dtype=torch.float32,
            ).reshape(1, frame_count, 128),
        ),
        (
            AUDIO_FIELD_INPUT_FEATURES_MASK,
            torch.ones((1, frame_count), dtype=torch.bool),
        ),
    )


def _image_fields(input_length: int = 10):
    return (
        (
            IMAGE_FIELD_MM_TOKEN_TYPE_IDS,
            torch.zeros((1, input_length), dtype=torch.int64),
        ),
        (
            IMAGE_FIELD_PIXEL_VALUES,
            torch.zeros((1, 8, 4), dtype=torch.float32),
        ),
        (
            IMAGE_FIELD_POSITION_IDS,
            torch.zeros((1, 8, 2), dtype=torch.int64),
        ),
    )


class _Tokenizer:
    def __call__(self, text, **_kwargs):
        ids = torch.ones((1, max(1, len(text))), dtype=torch.int64)
        return {"input_ids": ids, "attention_mask": torch.ones_like(ids)}

    def decode(self, token_ids, **_kwargs):
        return " ".join(map(str, token_ids))


class _AudioProcessor:
    audio_token = "<|audio|>"

    def apply_chat_template(self, conversation, **_kwargs):
        if conversation[0]["content"][0]["type"] != "audio":
            raise AssertionError("audio chat template was not used")
        input_ids = torch.arange(1, 11, dtype=torch.int64).reshape(1, 10)
        fields = dict(_audio_fields())
        return {
            "input_ids": input_ids,
            "attention_mask": torch.ones_like(input_ids),
            "mm_token_type_ids": fields[AUDIO_FIELD_MM_TOKEN_TYPE_IDS],
            "input_features": fields[AUDIO_FIELD_INPUT_FEATURES],
            "input_features_mask": fields[AUDIO_FIELD_INPUT_FEATURES_MASK],
        }


class _AudioAwareModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1))
        self.config = SimpleNamespace(use_cache=True)

    def generate(self, **kwargs):
        input_ids = kwargs["input_ids"]
        token_types = kwargs.get("mm_token_type_ids")
        features = kwargs.get("input_features")
        feature_mask = kwargs.get("input_features_mask")
        valid = (
            isinstance(token_types, torch.Tensor)
            and token_types.dtype == torch.int64
            and token_types.shape == input_ids.shape
            and token_types.is_contiguous()
            and isinstance(features, torch.Tensor)
            and features.dtype == torch.float32
            and features.ndim == 3
            and features.shape[0] == 1
            and features.shape[2] == 128
            and features.is_contiguous()
            and isinstance(feature_mask, torch.Tensor)
            and feature_mask.dtype == torch.bool
            and tuple(feature_mask.shape) == tuple(features.shape[:2])
            and feature_mask.is_contiguous()
        )
        if not valid:
            raise RuntimeError("worker did not reconstruct bounded audio tensors")
        if kwargs.get("do_sample") is not False or kwargs.get("use_cache") is not False:
            raise RuntimeError("decode policy is not bounded")
        streamer = kwargs["streamer"]
        streamer.put(input_ids)
        token = torch.tensor([[43]], dtype=torch.int64, device=input_ids.device)
        streamer.put(token)
        streamer.end()
        return torch.cat((input_ids, token), dim=1)


@dataclass(frozen=True)
class _AudioFactory:
    parent_pid: int

    def __call__(self):
        if os.getpid() == self.parent_pid:
            raise RuntimeError("model must not load in the controller")
        return _AudioAwareModel()


class TestAudioTensorProtocol(unittest.TestCase):
    def test_audio_fields_round_trip_as_exactly_four_int64_cpu_tensors(self) -> None:
        input_ids = torch.arange(1, 11, dtype=torch.int64).reshape(1, 10)
        message = make_generate_request(
            1,
            input_ids,
            torch.ones_like(input_ids),
            max_new_tokens=2,
            audio_tensors=_audio_fields(),
        )

        self.assertEqual(len(message), 4)
        self.assertTrue(all(value.device.type == "cpu" for value in message))
        self.assertTrue(all(value.dtype == torch.int64 for value in message))
        self.assertTrue(all(value.is_contiguous() for value in message))
        request = parse_request(message)
        self.assertIsNotNone(request)
        self.assertEqual(request.modality, MODALITY_AUDIO)
        self.assertFalse(request.image_tensors)
        actual = dict(request.audio_tensors)
        for code, expected in _audio_fields():
            self.assertTrue(torch.equal(actual[code], expected))

    def test_invalid_audio_schemas_and_combined_media_fail_closed(self) -> None:
        input_ids = torch.ones((1, 10), dtype=torch.int64)
        mask = torch.ones_like(input_ids)
        missing = _audio_fields()[:-1]
        nonfinite = list(_audio_fields())
        nonfinite[1] = (
            AUDIO_FIELD_INPUT_FEATURES,
            torch.full((1, 10, 128), float("nan"), dtype=torch.float32),
        )
        noncontiguous = list(_audio_fields())
        noncontiguous[1] = (
            AUDIO_FIELD_INPUT_FEATURES,
            torch.ones((1, 128, 10), dtype=torch.float32).transpose(1, 2),
        )
        mismatched_mask = list(_audio_fields())
        mismatched_mask[2] = (
            AUDIO_FIELD_INPUT_FEATURES_MASK,
            torch.ones((1, 9), dtype=torch.bool),
        )
        wrong_mask_dtype = list(_audio_fields())
        wrong_mask_dtype[2] = (
            AUDIO_FIELD_INPUT_FEATURES_MASK,
            torch.ones((1, 10), dtype=torch.int64),
        )
        frame_count = HARD_MAX_AUDIO_TENSOR_BYTES // (128 * 4) + 1
        oversized = _audio_fields(frame_count=frame_count)
        cases = (
            missing,
            tuple(nonfinite),
            tuple(noncontiguous),
            tuple(mismatched_mask),
            tuple(wrong_mask_dtype),
            oversized,
        )
        for audio_tensors in cases:
            with self.subTest(shape=tuple(audio_tensors[1][1].shape)):
                with self.assertRaises(TensorProtocolError):
                    make_generate_request(
                        2,
                        input_ids,
                        mask,
                        max_new_tokens=1,
                        audio_tensors=audio_tensors,
                    )
        with self.assertRaisesRegex(TensorProtocolError, "image and audio"):
            make_generate_request(
                3,
                input_ids,
                mask,
                max_new_tokens=1,
                image_tensors=_image_fields(),
                audio_tensors=_audio_fields(),
            )

    def test_boolean_mask_payload_tampering_is_rejected(self) -> None:
        input_ids = torch.ones((1, 10), dtype=torch.int64)
        message = list(
            make_generate_request(
                4,
                input_ids,
                None,
                max_new_tokens=1,
                audio_tensors=_audio_fields(),
            )
        )
        control = message[3].clone()
        control[-1] = 2
        message[3] = control
        with self.assertRaisesRegex(TensorProtocolError, "boolean"):
            parse_request(tuple(message))


class TestResidentAudioGeneration(unittest.TestCase):
    def test_audio_content_digest_is_bound_into_response_authority(self) -> None:
        session = _session_digest("audio-test")
        first = _bind_media_session_digest(session, "01" * 32)
        second = _bind_media_session_digest(session, "02" * 32)
        self.assertFalse(torch.equal(first, second))
        self.assertTrue(
            torch.equal(first, _bind_media_session_digest(session, "01" * 32))
        )

    def _service(self) -> ModelService:
        service = ModelService(
            _Tokenizer(),
            _AudioFactory(os.getpid()),
            max_input_tokens=32,
            max_new_tokens=4,
            startup_timeout=20.0,
            request_timeout=20.0,
            multimodal_processor_config=(Path(__file__), Path(__file__)),
        )
        processor = object.__new__(VerifiedGemma4MultimodalProcessor)
        processor.processor = _AudioProcessor()
        service._multimodal_processor = processor
        return service

    def test_real_16khz_mono_wav_bundle_reaches_single_spawn_worker(self) -> None:
        service = self._service()
        try:
            result = service.generate(
                "음성을 전사하세요.",
                audio_wav_content=_wav(),
                max_new_tokens=1,
                decode_mode="strict",
            )
        finally:
            service.stop()

        self.assertEqual(result.token_ids.tolist(), [43])
        self.assertEqual(result.text, "43")

    def test_audio_without_verified_processor_fails_before_model_start(self) -> None:
        service = ModelService(
            _Tokenizer(),
            _AudioFactory(os.getpid()),
            max_input_tokens=32,
            max_new_tokens=4,
        )
        with self.assertRaisesRegex(RequestLimitError, "verified"):
            service.generate("전사", audio_wav_content=_wav())
        self.assertIsNone(service.worker_pid)

    def test_optional_real_gemma4_processor_bundle_reaches_fake_worker(self) -> None:
        model_root = os.environ.get("COGNI_TEST_GEMMA4_ROOT")
        manifest = os.environ.get("COGNI_TEST_GEMMA4_MANIFEST")
        if not model_root or not manifest:
            self.skipTest("real local Gemma 4 processor paths were not supplied")
        processor = VerifiedGemma4MultimodalProcessor(model_root, manifest)
        service = ModelService(
            _Tokenizer(),
            _AudioFactory(os.getpid()),
            max_input_tokens=32,
            max_new_tokens=4,
            startup_timeout=20.0,
            request_timeout=20.0,
            multimodal_processor_config=(model_root, manifest),
        )
        service._multimodal_processor = processor
        try:
            result = service.generate(
                "음성을 전사하세요.",
                audio_wav_content=_wav(),
                max_new_tokens=1,
                decode_mode="strict",
            )
        finally:
            service.stop()
        self.assertEqual(result.token_ids.tolist(), [43])


if __name__ == "__main__":
    unittest.main()
