from __future__ import annotations

from hashlib import sha256
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image

from scripts.validate_gemma4_local_image import (
    BACKGROUND_RGB,
    IMAGE_SIZE,
    QUESTION,
    SQUARE_RGB,
    evaluate_response,
    run_validation,
    validation_png,
)


class _FakeService:
    def __init__(self, response: str = "중앙에는 파란색 정사각형이 있습니다.") -> None:
        self.response = response
        self.calls: list[dict[str, object]] = []
        self.stop_calls = 0

    def generate(self, prompt: str, **kwargs):
        self.calls.append({"prompt": prompt, **kwargs})
        return SimpleNamespace(
            text=self.response,
            finish_reason="stop",
            generation_mode="cogni_core",
        )

    def stop(self) -> None:
        self.stop_calls += 1


class TestGemma4LocalImageValidator(unittest.TestCase):
    def test_validation_png_is_deterministic_and_visually_bounded(self) -> None:
        first = validation_png()
        second = validation_png()
        self.assertEqual(sha256(first).digest(), sha256(second).digest())
        self.assertLessEqual(len(first), 256 * 1024)
        with Image.open(BytesIO(first)) as image:
            self.assertEqual(image.format, "PNG")
            self.assertEqual(image.size, IMAGE_SIZE)
            self.assertEqual(image.mode, "RGB")
            self.assertEqual(image.getpixel((0, 0)), BACKGROUND_RGB)
            self.assertEqual(image.getpixel((128, 128)), SQUARE_RGB)

    def test_conservative_bilingual_criterion_requires_both_concepts(self) -> None:
        for response in ("파란색 정사각형입니다.", "A blue square."):
            with self.subTest(response=response):
                self.assertTrue(evaluate_response(response)["pass"])
        for response in ("파란색입니다.", "A square.", "A red circle."):
            with self.subTest(response=response):
                self.assertFalse(evaluate_response(response)["pass"])
        with self.assertRaisesRegex(RuntimeError, "empty"):
            evaluate_response("  ")

    def test_actual_service_contract_is_manifest_bound_strict_and_stopped(self) -> None:
        fake = _FakeService()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model = root / "model"
            model.mkdir()
            manifest = root / "manifest.toml"
            manifest.write_text("schema = 1\n", encoding="utf-8")
            with patch(
                "scripts.validate_gemma4_local_image.ModelService.for_local_gemma",
                return_value=fake,
            ) as factory:
                evidence = run_validation(model, manifest)

        factory.assert_called_once_with(
            model,
            manifest_path=manifest,
            vram_limit_gib=16.7,
            max_input_tokens=4_096,
            max_new_tokens=64,
            max_prompt_chars=4_096,
            max_response_chars=512,
            startup_timeout=240.0,
            request_timeout=240.0,
        )
        self.assertEqual(fake.stop_calls, 1)
        self.assertEqual(len(fake.calls), 1)
        call = fake.calls[0]
        self.assertEqual(call["prompt"], QUESTION)
        self.assertIsInstance(call["image_content"], bytes)
        self.assertEqual(call["decode_mode"], "strict")
        self.assertEqual(call["sampling_seed"], 0)
        self.assertEqual(call["max_new_tokens"], 64)
        self.assertEqual(evidence["external_calls"], 0)
        self.assertTrue(evidence["actual_model_forward_executed"])
        self.assertTrue(evidence["image_understanding_pass"])

    def test_service_is_stopped_when_generation_raises(self) -> None:
        fake = _FakeService()

        def fail(*_args, **_kwargs):
            raise RuntimeError("synthetic generation failure")

        fake.generate = fail  # type: ignore[method-assign]
        with patch(
            "scripts.validate_gemma4_local_image.ModelService.for_local_gemma",
            return_value=fake,
        ):
            with self.assertRaisesRegex(RuntimeError, "synthetic"):
                run_validation(Path("model"), Path("manifest"))
        self.assertEqual(fake.stop_calls, 1)


if __name__ == "__main__":
    unittest.main()
