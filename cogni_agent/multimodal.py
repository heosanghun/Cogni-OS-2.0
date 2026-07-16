"""Verified local Gemma 4 multimodal preprocessing boundary.

The processor builds the exact instruction-tuned chat-template tensors used by
the resident worker.  Raw media, paths and Python objects never cross IPC.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from io import BytesIO
from pathlib import Path
from typing import Mapping
import wave

import torch
from torch import Tensor

from cogni_os.artifacts import ArtifactVerificationError, verify_artifact_manifest


MAX_IMAGE_BYTES = 8 * 1024 * 1024
MAX_IMAGE_PIXELS = 16_000_000
MAX_AUDIO_WAV_BYTES = 2 * 1024 * 1024
MAX_AUDIO_SECONDS = 30
REQUIRED_AUDIO_RATE = 16_000
MAX_BUNDLE_TENSOR_BYTES = 64 * 1024 * 1024

_IMAGE_KEYS = frozenset(
    {
        "input_ids",
        "attention_mask",
        "mm_token_type_ids",
        "pixel_values",
        "image_position_ids",
    }
)
_AUDIO_KEYS = frozenset(
    {
        "input_ids",
        "attention_mask",
        "mm_token_type_ids",
        "input_features",
        "input_features_mask",
    }
)


class MultimodalPreprocessError(ValueError):
    """A local artifact or input failed before reaching the model worker."""


@dataclass(frozen=True, slots=True)
class MultimodalTensorBundle:
    modality: str
    content_sha256: str
    tensors: tuple[tuple[str, Tensor], ...]
    processor_verified: bool = True

    def as_mapping(self) -> dict[str, Tensor]:
        return {name: tensor for name, tensor in self.tensors}

    @property
    def tensor_bytes(self) -> int:
        return sum(tensor.numel() * tensor.element_size() for _, tensor in self.tensors)


def _bounded_prompt(prompt: object) -> str:
    if (
        not isinstance(prompt, str)
        or not prompt.strip()
        or len(prompt) > 8_192
        or any(
            ord(character) < 32 and character not in "\t\r\n" for character in prompt
        )
    ):
        raise MultimodalPreprocessError("prompt must be bounded text")
    return prompt.strip()


def _validated_bundle(
    modality: str,
    content: bytes,
    output: object,
    allowed_keys: frozenset[str],
) -> MultimodalTensorBundle:
    if not isinstance(output, Mapping) or not set(output).issubset(allowed_keys):
        raise MultimodalPreprocessError(
            "processor returned an unsupported tensor schema"
        )
    required = {
        "image": {"input_ids", "attention_mask", "pixel_values"},
        "audio": {"input_ids", "attention_mask", "input_features"},
    }[modality]
    if not required.issubset(output):
        raise MultimodalPreprocessError("processor omitted required modality tensors")
    rows: list[tuple[str, Tensor]] = []
    total_bytes = 0
    for name in sorted(output):
        value = output[name]
        if not isinstance(value, Tensor) or value.device.type != "cpu":
            raise MultimodalPreprocessError("processor outputs must be CPU tensors")
        tensor = value.detach().contiguous()
        if tensor.ndim < 1 or tensor.shape[0] != 1 or tensor.numel() < 1:
            raise MultimodalPreprocessError(
                "processor returned an invalid tensor shape"
            )
        if tensor.is_floating_point() and not bool(torch.isfinite(tensor).all()):
            raise MultimodalPreprocessError("processor returned non-finite features")
        total_bytes += tensor.numel() * tensor.element_size()
        if total_bytes > MAX_BUNDLE_TENSOR_BYTES:
            raise MultimodalPreprocessError(
                "multimodal tensor bundle exceeds its byte limit"
            )
        rows.append((name, tensor))
    return MultimodalTensorBundle(
        modality,
        sha256(content).hexdigest(),
        tuple(rows),
    )


class VerifiedGemma4MultimodalProcessor:
    """Load Gemma4Processor only from a digest-verified local model snapshot."""

    def __init__(self, model_root: str | Path, manifest_path: str | Path) -> None:
        root = Path(model_root).expanduser().resolve(strict=True)
        manifest = Path(manifest_path).expanduser().resolve(strict=True)
        try:
            before = verify_artifact_manifest(root, manifest)
        except (OSError, ArtifactVerificationError) as exc:
            raise MultimodalPreprocessError(
                "model artifacts failed verification"
            ) from exc
        names = {path.name for path in before.files}
        if "processor_config.json" not in names or "tokenizer.json" not in names:
            raise MultimodalPreprocessError(
                "verified snapshot lacks processor artifacts"
            )
        try:
            from transformers import AutoProcessor

            processor = AutoProcessor.from_pretrained(
                root,
                local_files_only=True,
                trust_remote_code=False,
            )
        except Exception as exc:
            raise MultimodalPreprocessError(
                "local Gemma 4 processor could not be loaded"
            ) from exc
        try:
            after = verify_artifact_manifest(root, manifest)
        except (OSError, ArtifactVerificationError) as exc:
            raise MultimodalPreprocessError(
                "model artifacts changed during processor load"
            ) from exc
        if before.digests != after.digests:
            raise MultimodalPreprocessError(
                "model artifacts changed during processor load"
            )
        if processor.__class__.__name__ != "Gemma4Processor":
            raise MultimodalPreprocessError(
                "verified snapshot did not load Gemma4Processor"
            )
        self.model_root = root
        self.manifest_path = manifest
        self.processor = processor

    def process_image(self, content: bytes, prompt: str) -> MultimodalTensorBundle:
        if not isinstance(content, bytes) or not 1 <= len(content) <= MAX_IMAGE_BYTES:
            raise MultimodalPreprocessError("image exceeds its byte limit")
        prompt_text = _bounded_prompt(prompt)
        try:
            from PIL import Image, UnidentifiedImageError

            with Image.open(BytesIO(content)) as image:
                width, height = image.size
                if (
                    not 1 <= width <= 8_192
                    or not 1 <= height <= 8_192
                    or width * height > MAX_IMAGE_PIXELS
                    or image.format not in {"PNG", "JPEG", "WEBP"}
                ):
                    raise MultimodalPreprocessError(
                        "image dimensions or format are unsupported"
                    )
                image.load()
                rgb = image.convert("RGB")
        except MultimodalPreprocessError:
            raise
        except (OSError, UnidentifiedImageError) as exc:
            raise MultimodalPreprocessError(
                "image could not be decoded locally"
            ) from exc
        try:
            output = self.processor.apply_chat_template(
                [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": rgb},
                            {"type": "text", "text": prompt_text},
                        ],
                    }
                ],
                add_generation_prompt=True,
                tokenize=True,
                return_tensors="pt",
                return_dict=True,
                processor_kwargs={"return_mm_token_type_ids": True},
            )
        except Exception as exc:
            raise MultimodalPreprocessError(
                "Gemma 4 image preprocessing failed"
            ) from exc
        return _validated_bundle("image", content, output, _IMAGE_KEYS)

    def process_audio_wav(self, content: bytes, prompt: str) -> MultimodalTensorBundle:
        if (
            not isinstance(content, bytes)
            or not 1 <= len(content) <= MAX_AUDIO_WAV_BYTES
        ):
            raise MultimodalPreprocessError("audio exceeds its byte limit")
        prompt_text = _bounded_prompt(prompt)
        try:
            with wave.open(BytesIO(content), "rb") as stream:
                channels = stream.getnchannels()
                sample_width = stream.getsampwidth()
                sample_rate = stream.getframerate()
                frame_count = stream.getnframes()
                if (
                    channels != 1
                    or sample_width != 2
                    or sample_rate != REQUIRED_AUDIO_RATE
                    or not 1 <= frame_count <= REQUIRED_AUDIO_RATE * MAX_AUDIO_SECONDS
                ):
                    raise MultimodalPreprocessError(
                        "audio must be mono 16-bit PCM WAV at 16 kHz and at most 30 seconds"
                    )
                frames = stream.readframes(frame_count)
        except MultimodalPreprocessError:
            raise
        except (EOFError, OSError, wave.Error) as exc:
            raise MultimodalPreprocessError(
                "audio WAV could not be decoded locally"
            ) from exc
        if len(frames) != frame_count * 2:
            raise MultimodalPreprocessError("audio WAV is truncated")
        try:
            import numpy as np

            waveform = np.frombuffer(frames, dtype="<i2").astype("float32") / 32768.0
            if waveform.size != frame_count or not bool(np.isfinite(waveform).all()):
                raise MultimodalPreprocessError("audio waveform is invalid")
            output = self.processor.apply_chat_template(
                [
                    {
                        "role": "user",
                        "content": [
                            {"type": "audio", "audio": waveform},
                            {"type": "text", "text": prompt_text},
                        ],
                    }
                ],
                add_generation_prompt=True,
                tokenize=True,
                return_tensors="pt",
                return_dict=True,
                processor_kwargs={"return_mm_token_type_ids": True},
            )
        except MultimodalPreprocessError:
            raise
        except Exception as exc:
            raise MultimodalPreprocessError(
                "Gemma 4 audio preprocessing failed"
            ) from exc
        return _validated_bundle("audio", content, output, _AUDIO_KEYS)


__all__ = [
    "MAX_AUDIO_SECONDS",
    "MAX_BUNDLE_TENSOR_BYTES",
    "MAX_IMAGE_PIXELS",
    "MultimodalPreprocessError",
    "MultimodalTensorBundle",
    "VerifiedGemma4MultimodalProcessor",
]
