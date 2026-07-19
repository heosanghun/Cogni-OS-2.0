from __future__ import annotations

from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory
import wave
import unittest
from unittest.mock import patch

from PIL import Image
import torch

from cogni_agent.multimodal import (
    MAX_MULTIMODAL_MANIFEST_BYTES,
    MultimodalPreprocessError,
    MultimodalTensorBundle,
    VerifiedGemma4MultimodalProcessor,
    _validated_bundle,
    is_verified_multimodal_bundle,
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

    def test_bundle_authority_is_private_and_public_views_do_not_alias(self) -> None:
        output = {
            "input_ids": torch.ones((1, 2), dtype=torch.int64),
            "attention_mask": torch.ones((1, 2), dtype=torch.int64),
            "pixel_values": torch.ones((1, 2), dtype=torch.float32),
        }
        bundle = _validated_bundle("image", b"content", output, frozenset(output))
        self.assertTrue(is_verified_multimodal_bundle(bundle, modality="image"))
        with self.assertRaises(TypeError):
            MultimodalTensorBundle()

        forged = object.__new__(MultimodalTensorBundle)
        object.__setattr__(forged, "_modality", "image")
        object.__setattr__(forged, "_content_sha256", "0" * 64)
        object.__setattr__(forged, "_tensors", tuple(sorted(output.items())))
        self.assertFalse(is_verified_multimodal_bundle(forged, modality="image"))

        first = bundle.as_mapping()
        second = bundle.as_mapping()
        self.assertIsNot(first["input_ids"], second["input_ids"])
        self.assertNotEqual(
            first["input_ids"].untyped_storage().data_ptr(),
            second["input_ids"].untyped_storage().data_ptr(),
        )
        first["input_ids"].zero_()
        self.assertTrue(bool(torch.all(second["input_ids"] == 1)))
        self.assertTrue(bool(torch.all(bundle.as_mapping()["input_ids"] == 1)))
        with self.assertRaises(TypeError):
            first["extra"] = torch.ones((1, 1))

    def test_manifest_is_bounded_and_regular_before_artifact_verifier(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory) / "model"
            root.mkdir()
            manifest = Path(directory) / "manifest.toml"
            manifest.write_bytes(b"x" * (MAX_MULTIMODAL_MANIFEST_BYTES + 1))
            with (
                patch("cogni_agent.multimodal.verify_artifact_manifest") as verifier,
                self.assertRaisesRegex(
                    MultimodalPreprocessError, "model artifacts failed verification"
                ),
            ):
                VerifiedGemma4MultimodalProcessor(root, manifest)
            verifier.assert_not_called()

            manifest.unlink()
            manifest.mkdir()
            with (
                patch("cogni_agent.multimodal.verify_artifact_manifest") as verifier,
                self.assertRaisesRegex(
                    MultimodalPreprocessError, "model artifacts failed verification"
                ),
            ):
                VerifiedGemma4MultimodalProcessor(root, manifest)
            verifier.assert_not_called()

    def test_manifest_link_or_reparse_is_rejected_before_verifier(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory) / "model"
            root.mkdir()
            manifest = Path(directory) / "manifest.toml"
            manifest.write_text("[files]\n", encoding="utf-8")
            with (
                patch(
                    "cogni_agent.multimodal._path_is_link_or_reparse",
                    return_value=True,
                ),
                patch("cogni_agent.multimodal.verify_artifact_manifest") as verifier,
                self.assertRaisesRegex(
                    MultimodalPreprocessError, "model artifacts failed verification"
                ),
            ):
                VerifiedGemma4MultimodalProcessor(root, manifest)
            verifier.assert_not_called()


if __name__ == "__main__":
    unittest.main()
