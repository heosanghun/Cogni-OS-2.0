from __future__ import annotations

from io import BytesIO
import wave
import unittest
from unittest.mock import patch

from PIL import Image
import torch

from cogni_agent.multimodal import (
    MultimodalPreprocessError,
    VerifiedGemma4MultimodalProcessor,
    _validated_bundle,
)


def _wav(seconds: float = 0.05, sample_rate: int = 16_000) -> bytes:
    output = BytesIO()
    with wave.open(output, "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(sample_rate)
        stream.writeframes(b"\0\0" * max(1, int(seconds * sample_rate)))
    return output.getvalue()


def _png(size: tuple[int, int] = (16, 16)) -> bytes:
    output = BytesIO()
    Image.new("RGB", size, "navy").save(output, format="PNG")
    return output.getvalue()


class _FakeProcessor:
    image_token = "<|image|>"
    audio_token = "<|audio|>"

    def apply_chat_template(self, conversation, **kwargs):
        if kwargs.get("add_generation_prompt") is not True:
            raise AssertionError("generation prompt was omitted")
        if kwargs.get("processor_kwargs") != {"return_mm_token_type_ids": True}:
            raise AssertionError("multimodal token types were omitted")
        common = {
            "input_ids": torch.ones((1, 4), dtype=torch.int64),
            "attention_mask": torch.ones((1, 4), dtype=torch.int64),
            "mm_token_type_ids": torch.zeros((1, 4), dtype=torch.int64),
        }
        content = conversation[0]["content"]
        if content[0]["type"] == "image":
            return {
                **common,
                "pixel_values": torch.ones((1, 8, 4), dtype=torch.float32),
                "image_position_ids": torch.zeros((1, 8, 2), dtype=torch.int64),
            }
        if content[0]["type"] == "audio":
            return {
                **common,
                "input_features": torch.ones((1, 8, 4), dtype=torch.float32),
                "input_features_mask": torch.ones((1, 8), dtype=torch.bool),
            }
        raise AssertionError(content)


class TestVerifiedGemma4MultimodalProcessor(unittest.TestCase):
    def _service(self) -> VerifiedGemma4MultimodalProcessor:
        service = object.__new__(VerifiedGemma4MultimodalProcessor)
        service.processor = _FakeProcessor()
        return service

    def test_image_and_audio_produce_bounded_cpu_tensor_bundles(self) -> None:
        service = self._service()
        image = service.process_image(_png(), "그림을 설명하세요.")
        audio = service.process_audio_wav(_wav(), "음성을 전사하세요.")

        self.assertEqual(image.modality, "image")
        self.assertEqual(audio.modality, "audio")
        self.assertIn("pixel_values", image.as_mapping())
        self.assertIn("input_features", audio.as_mapping())
        self.assertTrue(all(t.device.type == "cpu" for _, t in image.tensors))
        self.assertGreater(image.tensor_bytes, 0)

    def test_audio_format_and_image_pixel_limits_fail_closed(self) -> None:
        service = self._service()
        with self.assertRaisesRegex(MultimodalPreprocessError, "16 kHz"):
            service.process_audio_wav(_wav(sample_rate=8_000), "전사")
        with patch("PIL.Image.open") as opened:
            image = opened.return_value.__enter__.return_value
            image.size = (20_000, 20_000)
            image.format = "PNG"
            with self.assertRaisesRegex(MultimodalPreprocessError, "dimensions"):
                service.process_image(_png(), "설명")

    def test_unknown_or_nonfinite_processor_outputs_are_rejected(self) -> None:
        output = {
            "input_ids": torch.ones((1, 1), dtype=torch.int64),
            "attention_mask": torch.ones((1, 1), dtype=torch.int64),
            "pixel_values": torch.tensor([[float("nan")]]),
        }
        with self.assertRaisesRegex(MultimodalPreprocessError, "non-finite"):
            _validated_bundle("image", b"x", output, frozenset(output))
        output["pixel_values"] = torch.ones((1, 1))
        output["unexpected"] = torch.ones((1, 1))
        with self.assertRaisesRegex(MultimodalPreprocessError, "unsupported"):
            _validated_bundle("image", b"x", output, frozenset(output) - {"unexpected"})


if __name__ == "__main__":
    unittest.main()
