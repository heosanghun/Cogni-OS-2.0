from __future__ import annotations

from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import ModuleType
import unittest
from unittest.mock import patch

import torch

from cogni_agent.multimodal import (
    MAX_VIDEO_FRAMES,
    MultimodalPreprocessError,
    VerifiedGemma4MultimodalProcessor,
)
from cogni_os.artifacts import ArtifactIdentity, VerifiedArtifactSet


_IDENTITY = ArtifactIdentity(
    family="gemma4",
    variant="E4B",
    role="instruction_tuned",
    source="google/gemma-4-E4B-it",
    revision="a4c2d58be94dda072b918d9db64ee85c8ed34e3f",
)


class _Gemma4ProcessorBase:
    video_token = "<|video|>"

    def __init__(self) -> None:
        self.calls: list[tuple[object, dict[str, object]]] = []

    def video_processor(self, *_args, **_kwargs):
        return None

    def apply_chat_template(self, conversation, **kwargs):
        self.calls.append((conversation, kwargs))
        return {
            "input_ids": torch.ones((1, 4), dtype=torch.int64),
            "attention_mask": torch.ones((1, 4), dtype=torch.int64),
            "mm_token_type_ids": torch.ones((1, 4), dtype=torch.int64),
            "pixel_values_videos": torch.ones((1, 3, 2, 2, 3), dtype=torch.float32),
            "video_position_ids": torch.zeros((1, 3, 2), dtype=torch.int64),
        }


Gemma4Processor = type("Gemma4Processor", (_Gemma4ProcessorBase,), {})


def _frame(value: int = 0, size: tuple[int, int] = (2, 2)) -> torch.Tensor:
    height, width = size
    return torch.full((height, width, 3), value, dtype=torch.uint8)


class TestBoundedVideoProcessor(unittest.TestCase):
    def _service(
        self,
        *,
        processor: object | None = None,
        identity: ArtifactIdentity = _IDENTITY,
    ) -> VerifiedGemma4MultimodalProcessor:
        service = object.__new__(VerifiedGemma4MultimodalProcessor)
        service.processor = processor if processor is not None else Gemma4Processor()
        service.artifact_identity = identity
        service.manifest_sha256 = "a" * 64
        return service

    def _process(self, service: VerifiedGemma4MultimodalProcessor, **kwargs):
        parameters = {
            "frames": (_frame(1), _frame(2), _frame(3)),
            "prompt": "영상의 변화를 설명하세요.",
            "duration_seconds": 1.0,
            "sampling_fps": 2.0,
            "timestamps_seconds": (0.0, 0.5, 1.0),
        }
        parameters.update(kwargs)
        return service.process_video_frames(**parameters)

    def test_preprocesses_only_bounded_decoded_cpu_frames(self) -> None:
        service = self._service()

        result = self._process(service)

        self.assertEqual(result.bundle.modality, "video")
        self.assertEqual(result.sampling.frame_count, 3)
        self.assertEqual(result.sampling.total_pixels, 12)
        self.assertEqual(result.sampling.decoded_bytes, 36)
        self.assertEqual(result.sampling.input_tensor_elements, 36)
        self.assertIn("pixel_values_videos", result.as_mapping())
        self.assertGreater(result.tensor_bytes, 0)
        self.assertFalse(result.processor_identity.actual_model_inference)
        self.assertFalse(result.processor_identity.vram_measured)
        self.assertTrue(result.processor_identity.local_files_only)
        self.assertFalse(result.processor_identity.trust_remote_code)
        conversation, options = service.processor.calls[0]
        media = conversation[0]["content"][0]
        self.assertEqual(media["type"], "video")
        self.assertEqual(len(media["video"]), 3)
        self.assertEqual(media["video"][0].shape, (2, 2, 3))
        self.assertTrue(media["video"][0].flags.owndata)
        self.assertTrue(options["add_generation_prompt"])
        self.assertTrue(options["tokenize"])
        self.assertEqual(options["return_tensors"], "pt")
        self.assertEqual(
            options["processor_kwargs"], {"return_mm_token_type_ids": True}
        )

    def test_content_digest_is_deterministic_and_binds_frame_values(self) -> None:
        service = self._service()
        first = self._process(service).bundle.content_sha256
        second = self._process(service).bundle.content_sha256
        changed = self._process(
            service,
            frames=(_frame(1), _frame(2), _frame(4)),
        ).bundle.content_sha256

        self.assertEqual(first, second)
        self.assertNotEqual(first, changed)

    def test_frame_count_dimensions_and_aggregate_limits_fail_closed(self) -> None:
        service = self._service()
        with self.assertRaisesRegex(MultimodalPreprocessError, "frame count"):
            self._process(
                service,
                frames=tuple(_frame() for _ in range(MAX_VIDEO_FRAMES + 1)),
                duration_seconds=16.0,
                sampling_fps=1.0,
                timestamps_seconds=tuple(float(index) for index in range(17)),
            )
        with self.assertRaisesRegex(MultimodalPreprocessError, "dimensions"):
            self._process(
                service,
                frames=(_frame(size=(1_081, 1)),),
                timestamps_seconds=(0.0,),
            )
        with self.assertRaisesRegex(MultimodalPreprocessError, "uniform"):
            self._process(
                service,
                frames=(_frame(size=(2, 2)), _frame(size=(2, 3))),
                timestamps_seconds=(0.0, 0.5),
            )
        with patch("cogni_agent.multimodal.MAX_VIDEO_TOTAL_PIXELS", 3):
            with self.assertRaisesRegex(MultimodalPreprocessError, "total pixels"):
                self._process(
                    service,
                    frames=(_frame(),),
                    timestamps_seconds=(0.0,),
                )
        with patch("cogni_agent.multimodal.MAX_VIDEO_DECODED_BYTES", 10):
            with self.assertRaisesRegex(MultimodalPreprocessError, "decoded bytes"):
                self._process(
                    service,
                    frames=(_frame(),),
                    timestamps_seconds=(0.0,),
                )
        with patch("cogni_agent.multimodal.MAX_VIDEO_INPUT_TENSOR_ELEMENTS", 10):
            with self.assertRaisesRegex(MultimodalPreprocessError, "input tensor"):
                self._process(
                    service,
                    frames=(_frame(),),
                    timestamps_seconds=(0.0,),
                )

    def test_duration_sampling_and_timestamps_fail_closed(self) -> None:
        service = self._service()
        invalid_cases = (
            ({"duration_seconds": float("nan")}, "duration"),
            ({"duration_seconds": 30.1}, "duration"),
            ({"sampling_fps": float("inf")}, "sampling"),
            ({"sampling_fps": 4.1}, "sampling"),
            ({"timestamps_seconds": (0.0, 0.5)}, "match"),
            ({"timestamps_seconds": (0.0, 0.0, 1.0)}, "increasing"),
            ({"timestamps_seconds": (0.0, float("nan"), 1.0)}, "timestamp"),
            (
                {
                    "duration_seconds": 1.0,
                    "sampling_fps": 1.0,
                    "timestamps_seconds": (0.0, 0.5, 1.0),
                },
                "sampling",
            ),
        )
        for arguments, message in invalid_cases:
            with self.subTest(arguments=arguments):
                with self.assertRaisesRegex(MultimodalPreprocessError, message):
                    self._process(service, **arguments)

    def test_nonfinite_and_oversized_processor_outputs_fail_closed(self) -> None:
        class _NonFinite(_Gemma4ProcessorBase):
            def apply_chat_template(self, _conversation, **_kwargs):
                return {
                    "input_ids": torch.ones((1, 1), dtype=torch.int64),
                    "attention_mask": torch.ones((1, 1), dtype=torch.int64),
                    "pixel_values": torch.tensor([[float("nan")]]),
                }

        processor = type("Gemma4Processor", (_NonFinite,), {})()
        with self.assertRaisesRegex(MultimodalPreprocessError, "non-finite"):
            self._process(self._service(processor=processor))

        with patch("cogni_agent.multimodal.MAX_VIDEO_OUTPUT_TENSOR_ELEMENTS", 4):
            with self.assertRaisesRegex(MultimodalPreprocessError, "output tensor"):
                self._process(self._service())

    def test_wrong_manifest_and_unsupported_processor_fail_closed(self) -> None:
        wrong = ArtifactIdentity(
            family="gemma4",
            variant="7B",
            role="instruction_tuned",
            source="google/gemma-4-7b-it",
            revision="a" * 40,
        )
        with self.assertRaisesRegex(MultimodalPreprocessError, "manifest identity"):
            self._process(self._service(identity=wrong))

        unsupported = type(
            "Gemma4Processor",
            (),
            {
                "video_token": "<|video|>",
                "apply_chat_template": lambda *_args, **_kwargs: {},
            },
        )()
        with self.assertRaisesRegex(MultimodalPreprocessError, "video capability"):
            self._process(self._service(processor=unsupported))

    def test_snapshot_loader_is_strictly_local_and_manifest_bound(self) -> None:
        calls: list[tuple[Path, dict[str, object]]] = []

        class _AutoProcessor:
            @classmethod
            def from_pretrained(cls, root, **kwargs):
                calls.append((Path(root), kwargs))
                return Gemma4Processor()

        module = ModuleType("transformers")
        module.AutoProcessor = _AutoProcessor
        with TemporaryDirectory() as directory:
            root = Path(directory) / "model"
            root.mkdir()
            processor_path = root / "processor_config.json"
            tokenizer_path = root / "tokenizer.json"
            processor_path.write_text("{}", encoding="utf-8")
            tokenizer_path.write_text("{}", encoding="utf-8")
            manifest = Path(directory) / "manifest.toml"
            manifest.write_text("[files]\n", encoding="utf-8")
            verified = VerifiedArtifactSet(
                root=root,
                files=(processor_path, tokenizer_path),
                identity=_IDENTITY,
                digests=(("processor_config.json", "1" * 64),),
            )
            with (
                patch.dict(sys.modules, {"transformers": module}),
                patch(
                    "cogni_agent.multimodal.verify_artifact_manifest",
                    side_effect=(verified, verified),
                ),
            ):
                service = VerifiedGemma4MultimodalProcessor(root, manifest)

        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0][0], root)
        self.assertEqual(
            calls[0][1],
            {"local_files_only": True, "trust_remote_code": False},
        )
        self.assertEqual(service.artifact_identity, _IDENTITY)
        self.assertEqual(len(service.manifest_sha256), 64)


if __name__ == "__main__":
    unittest.main()
