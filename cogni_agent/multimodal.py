"""Verified local Gemma 4 multimodal preprocessing boundary.

The processor builds the exact instruction-tuned chat-template tensors used by
the resident worker.  Raw media, paths and Python objects never cross IPC.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from io import BytesIO
import math
from pathlib import Path
from typing import Mapping
import wave

import torch
from torch import Tensor

from cogni_os.artifacts import (
    ArtifactIdentity,
    ArtifactVerificationError,
    verify_artifact_manifest,
)

from .model_trust import verify_instruction_tuned_e4b_snapshot


MAX_IMAGE_BYTES = 8 * 1024 * 1024
MAX_IMAGE_PIXELS = 16_000_000
MAX_AUDIO_WAV_BYTES = 2 * 1024 * 1024
MAX_AUDIO_SECONDS = 30
REQUIRED_AUDIO_RATE = 16_000
MAX_BUNDLE_TENSOR_BYTES = 64 * 1024 * 1024
MAX_VIDEO_FRAMES = 16
MAX_VIDEO_WIDTH = 1_920
MAX_VIDEO_HEIGHT = 1_080
MAX_VIDEO_DURATION_SECONDS = 30.0
MAX_VIDEO_SAMPLING_FPS = 4.0
MAX_VIDEO_TOTAL_PIXELS = 16_777_216
MAX_VIDEO_DECODED_BYTES = 64 * 1024 * 1024
MAX_VIDEO_INPUT_TENSOR_ELEMENTS = MAX_VIDEO_TOTAL_PIXELS * 3
MAX_VIDEO_OUTPUT_TENSOR_ELEMENTS = 16 * 1024 * 1024

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
_VIDEO_KEYS = frozenset(
    {
        "input_ids",
        "attention_mask",
        "mm_token_type_ids",
        "pixel_values",
        "pixel_values_videos",
        "video_grid_thw",
        "video_position_ids",
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


@dataclass(frozen=True, slots=True)
class VideoSamplingMetadata:
    """Measured properties of an admitted, already-decoded frame sequence."""

    frame_count: int
    width: int
    height: int
    duration_seconds: float
    sampling_fps: float
    timestamps_seconds: tuple[float, ...]
    total_pixels: int
    decoded_bytes: int
    input_tensor_elements: int


@dataclass(frozen=True, slots=True)
class VideoProcessorIdentity:
    """Manifest and processor identity for preprocessing evidence only."""

    family: str
    variant: str
    role: str
    source: str
    revision: str
    manifest_sha256: str
    processor_class: str
    local_files_only: bool = True
    trust_remote_code: bool = False
    actual_model_inference: bool = False
    vram_measured: bool = False


@dataclass(frozen=True, slots=True)
class VideoPreprocessResult:
    """A bounded CPU tensor bundle without an inference or VRAM claim."""

    bundle: MultimodalTensorBundle
    sampling: VideoSamplingMetadata
    processor_identity: VideoProcessorIdentity

    def as_mapping(self) -> dict[str, Tensor]:
        return self.bundle.as_mapping()

    @property
    def tensor_bytes(self) -> int:
        return self.bundle.tensor_bytes

    @property
    def output_tensor_elements(self) -> int:
        return sum(tensor.numel() for _, tensor in self.bundle.tensors)


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


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validated_bundle(
    modality: str,
    content: bytes,
    output: object,
    allowed_keys: frozenset[str],
    *,
    precomputed_content_sha256: str | None = None,
) -> MultimodalTensorBundle:
    if not isinstance(output, Mapping) or not set(output).issubset(allowed_keys):
        raise MultimodalPreprocessError(
            "processor returned an unsupported tensor schema"
        )
    required = {
        "image": {"input_ids", "attention_mask", "pixel_values"},
        "audio": {"input_ids", "attention_mask", "input_features"},
        "video": {"input_ids", "attention_mask"},
    }[modality]
    if not required.issubset(output):
        raise MultimodalPreprocessError("processor omitted required modality tensors")
    if modality == "video" and not {
        "pixel_values",
        "pixel_values_videos",
    }.intersection(output):
        raise MultimodalPreprocessError("processor omitted required video features")
    if precomputed_content_sha256 is not None and (
        len(precomputed_content_sha256) != 64
        or any(
            character not in "0123456789abcdef"
            for character in precomputed_content_sha256
        )
    ):
        raise MultimodalPreprocessError("precomputed content digest is invalid")
    rows: list[tuple[str, Tensor]] = []
    total_bytes = 0
    for name in sorted(output):
        value = output[name]
        if (
            not isinstance(value, Tensor)
            or value.device.type != "cpu"
            or value.layout != torch.strided
        ):
            raise MultimodalPreprocessError("processor outputs must be CPU tensors")
        if value.ndim < 1 or value.shape[0] != 1 or value.numel() < 1:
            raise MultimodalPreprocessError(
                "processor returned an invalid tensor shape"
            )
        total_bytes += value.numel() * value.element_size()
        if total_bytes > MAX_BUNDLE_TENSOR_BYTES:
            raise MultimodalPreprocessError(
                "multimodal tensor bundle exceeds its byte limit"
            )
        try:
            tensor = value.detach().contiguous()
            finite = bool(torch.isfinite(tensor).all())
        except RuntimeError as exc:
            raise MultimodalPreprocessError(
                "processor returned an unsupported tensor"
            ) from exc
        if not finite:
            raise MultimodalPreprocessError("processor returned non-finite features")
        rows.append((name, tensor))
    return MultimodalTensorBundle(
        modality,
        precomputed_content_sha256 or sha256(content).hexdigest(),
        tuple(rows),
    )


def _bounded_video_frames(
    frames: object,
    *,
    duration_seconds: object,
    sampling_fps: object,
    timestamps_seconds: object,
) -> tuple[tuple[Tensor, ...], VideoSamplingMetadata, str]:
    if type(frames) is not tuple or not 1 <= len(frames) <= MAX_VIDEO_FRAMES:
        raise MultimodalPreprocessError("video frame count exceeds its fixed limit")
    if (
        isinstance(duration_seconds, bool)
        or not isinstance(duration_seconds, (int, float))
        or not math.isfinite(float(duration_seconds))
        or not 0.0 < float(duration_seconds) <= MAX_VIDEO_DURATION_SECONDS
    ):
        raise MultimodalPreprocessError("video duration metadata is invalid")
    if (
        isinstance(sampling_fps, bool)
        or not isinstance(sampling_fps, (int, float))
        or not math.isfinite(float(sampling_fps))
        or not 0.0 < float(sampling_fps) <= MAX_VIDEO_SAMPLING_FPS
    ):
        raise MultimodalPreprocessError("video sampling metadata is invalid")
    if type(timestamps_seconds) is not tuple or len(timestamps_seconds) != len(frames):
        raise MultimodalPreprocessError("video timestamps must match the frame count")

    duration = float(duration_seconds)
    sample_rate = float(sampling_fps)
    timestamps: list[float] = []
    previous = -1.0
    for value in timestamps_seconds:
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
        ):
            raise MultimodalPreprocessError("video timestamp metadata is invalid")
        timestamp = float(value)
        if timestamp < 0.0 or timestamp > duration or timestamp <= previous:
            raise MultimodalPreprocessError(
                "video timestamps must be strictly increasing within duration"
            )
        timestamps.append(timestamp)
        previous = timestamp
    if len(timestamps) > math.floor(duration * sample_rate + 1e-9) + 1:
        raise MultimodalPreprocessError("video frame count exceeds sampling metadata")
    if len(timestamps) > 1:
        observed_fps = (len(timestamps) - 1) / (timestamps[-1] - timestamps[0])
        if observed_fps > sample_rate * (1.0 + 1e-6):
            raise MultimodalPreprocessError("video timestamps exceed sampling metadata")

    bounded: list[Tensor] = []
    width = 0
    height = 0
    total_pixels = 0
    decoded_bytes = 0
    total_elements = 0
    digest = sha256()
    digest.update(b"cogni-video-frames-v1\0")
    digest.update(f"{duration:.9f}\0{sample_rate:.9f}\0".encode("ascii"))
    for index, value in enumerate(frames):
        if (
            not isinstance(value, Tensor)
            or value.device.type != "cpu"
            or value.layout != torch.strided
        ):
            raise MultimodalPreprocessError("video frames must be CPU tensors")
        if value.dtype != torch.uint8 or value.ndim != 3 or value.shape[2] != 3:
            raise MultimodalPreprocessError(
                "video frames must be HWC RGB uint8 tensors"
            )
        frame_height, frame_width, _channels = map(int, value.shape)
        if (
            not 1 <= frame_width <= MAX_VIDEO_WIDTH
            or not 1 <= frame_height <= MAX_VIDEO_HEIGHT
        ):
            raise MultimodalPreprocessError("video frame dimensions are unsupported")
        if index == 0:
            width, height = frame_width, frame_height
        elif (frame_width, frame_height) != (width, height):
            raise MultimodalPreprocessError("video frame dimensions must be uniform")
        total_pixels += frame_width * frame_height
        total_elements += value.numel()
        decoded_bytes += value.numel() * value.element_size()
        if total_pixels > MAX_VIDEO_TOTAL_PIXELS:
            raise MultimodalPreprocessError("video total pixels exceed their limit")
        if total_elements > MAX_VIDEO_INPUT_TENSOR_ELEMENTS:
            raise MultimodalPreprocessError(
                "video input tensor elements exceed their limit"
            )
        if decoded_bytes > MAX_VIDEO_DECODED_BYTES:
            raise MultimodalPreprocessError("video decoded bytes exceed their limit")
        # Snapshot caller-owned storage only after all allocation budgets pass.
        try:
            frame = value.detach().contiguous().clone()
            finite = bool(torch.isfinite(frame).all())
        except RuntimeError as exc:
            raise MultimodalPreprocessError(
                "video frame tensor is unsupported"
            ) from exc
        if frame.requires_grad or not finite:
            raise MultimodalPreprocessError("video frame tensor is non-finite")
        digest.update(index.to_bytes(4, "big"))
        digest.update(frame_height.to_bytes(4, "big"))
        digest.update(frame_width.to_bytes(4, "big"))
        digest.update(f"{timestamps[index]:.9f}\0".encode("ascii"))
        digest.update(frame.numpy().tobytes(order="C"))
        bounded.append(frame)
    metadata = VideoSamplingMetadata(
        frame_count=len(bounded),
        width=width,
        height=height,
        duration_seconds=duration,
        sampling_fps=sample_rate,
        timestamps_seconds=tuple(timestamps),
        total_pixels=total_pixels,
        decoded_bytes=decoded_bytes,
        input_tensor_elements=total_elements,
    )
    return tuple(bounded), metadata, digest.hexdigest()


class VerifiedGemma4MultimodalProcessor:
    """Load Gemma4Processor only from a digest-verified local model snapshot."""

    def __init__(self, model_root: str | Path, manifest_path: str | Path) -> None:
        root = Path(model_root).expanduser().resolve(strict=True)
        manifest = Path(manifest_path).expanduser().resolve(strict=True)
        try:
            manifest_digest_before = _sha256_file(manifest)
            before = verify_artifact_manifest(root, manifest)
            verify_instruction_tuned_e4b_snapshot(before)
        except (OSError, ArtifactVerificationError, ValueError) as exc:
            raise MultimodalPreprocessError(
                "model artifacts failed verification"
            ) from exc
        relative_names = {name for name, _digest in before.digests}
        if not {"processor_config.json", "tokenizer.json"}.issubset(relative_names):
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
            verify_instruction_tuned_e4b_snapshot(after)
            manifest_digest_after = _sha256_file(manifest)
        except (OSError, ArtifactVerificationError, ValueError) as exc:
            raise MultimodalPreprocessError(
                "model artifacts changed during processor load"
            ) from exc
        if (
            before.digests != after.digests
            or before.identity != after.identity
            or manifest_digest_before != manifest_digest_after
        ):
            raise MultimodalPreprocessError(
                "model artifacts changed during processor load"
            )
        if processor.__class__.__name__ != "Gemma4Processor":
            raise MultimodalPreprocessError(
                "verified snapshot did not load Gemma4Processor"
            )
        self.model_root = root
        self.manifest_path = manifest
        self.artifact_identity = before.identity
        self.manifest_sha256 = manifest_digest_before
        self.processor = processor

    def _verified_video_identity(self) -> VideoProcessorIdentity:
        identity = getattr(self, "artifact_identity", None)
        if not isinstance(identity, ArtifactIdentity) or (
            identity.family.casefold() != "gemma4"
            or identity.variant.casefold() != "e4b"
            or identity.role != "instruction_tuned"
            or identity.source.casefold() != "google/gemma-4-e4b-it"
            or len(identity.revision) not in {40, 64}
            or any(
                character not in "0123456789abcdef" for character in identity.revision
            )
        ):
            raise MultimodalPreprocessError(
                "video preprocessing requires the exact Gemma 4 E4B IT manifest identity"
            )
        manifest_sha256 = getattr(self, "manifest_sha256", None)
        if (
            not isinstance(manifest_sha256, str)
            or len(manifest_sha256) != 64
            or any(character not in "0123456789abcdef" for character in manifest_sha256)
        ):
            raise MultimodalPreprocessError(
                "video processor manifest digest is unavailable"
            )
        processor = getattr(self, "processor", None)
        processor_class = processor.__class__.__name__ if processor is not None else ""
        video_processor = getattr(processor, "video_processor", None)
        apply_chat_template = getattr(processor, "apply_chat_template", None)
        video_token = getattr(processor, "video_token", None)
        tokenizer = getattr(processor, "tokenizer", None)
        video_token_id = getattr(processor, "video_token_id", None)
        if video_token_id is None:
            video_token_id = getattr(tokenizer, "video_token_id", None)
        if (
            processor_class != "Gemma4Processor"
            or not callable(video_processor)
            or not callable(apply_chat_template)
            or not (
                isinstance(video_token, str)
                and 1 <= len(video_token) <= 128
                and not any(ord(character) < 32 for character in video_token)
                or isinstance(video_token_id, int)
                and not isinstance(video_token_id, bool)
                and video_token_id >= 0
            )
        ):
            raise MultimodalPreprocessError(
                "verified processor does not expose an executable video capability"
            )
        return VideoProcessorIdentity(
            family=identity.family,
            variant=identity.variant,
            role=identity.role,
            source=identity.source,
            revision=identity.revision,
            manifest_sha256=manifest_sha256,
            processor_class=processor_class,
        )

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

    def process_video_frames(
        self,
        frames: tuple[Tensor, ...],
        prompt: str,
        *,
        duration_seconds: float,
        sampling_fps: float,
        timestamps_seconds: tuple[float, ...],
    ) -> VideoPreprocessResult:
        """Preprocess an already-decoded, bounded CPU RGB frame sequence.

        This boundary deliberately has no file/path/URL/decoder contract. It
        neither launches a decoder nor claims that the resulting tensors were
        executed by the model. A later worker/IPC integration must add its own
        independently attested inference and memory evidence.
        """

        prompt_text = _bounded_prompt(prompt)
        identity = self._verified_video_identity()
        bounded_frames, metadata, content_sha256 = _bounded_video_frames(
            frames,
            duration_seconds=duration_seconds,
            sampling_fps=sampling_fps,
            timestamps_seconds=timestamps_seconds,
        )
        # NumPy arrays are local, already-decoded frame values. No path, URL,
        # encoded video, shell command or decoder process enters this contract.
        processor_frames = [frame.numpy().copy() for frame in bounded_frames]
        try:
            output = self.processor.apply_chat_template(
                [
                    {
                        "role": "user",
                        "content": [
                            {"type": "video", "video": processor_frames},
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
                "Gemma 4 video preprocessing failed"
            ) from exc
        bundle = _validated_bundle(
            "video",
            b"",
            output,
            _VIDEO_KEYS,
            precomputed_content_sha256=content_sha256,
        )
        output_elements = sum(tensor.numel() for _, tensor in bundle.tensors)
        if output_elements > MAX_VIDEO_OUTPUT_TENSOR_ELEMENTS:
            raise MultimodalPreprocessError(
                "video output tensor elements exceed their limit"
            )
        return VideoPreprocessResult(
            bundle=bundle,
            sampling=metadata,
            processor_identity=identity,
        )


__all__ = [
    "MAX_AUDIO_SECONDS",
    "MAX_BUNDLE_TENSOR_BYTES",
    "MAX_IMAGE_PIXELS",
    "MAX_VIDEO_DECODED_BYTES",
    "MAX_VIDEO_DURATION_SECONDS",
    "MAX_VIDEO_FRAMES",
    "MAX_VIDEO_HEIGHT",
    "MAX_VIDEO_INPUT_TENSOR_ELEMENTS",
    "MAX_VIDEO_OUTPUT_TENSOR_ELEMENTS",
    "MAX_VIDEO_SAMPLING_FPS",
    "MAX_VIDEO_TOTAL_PIXELS",
    "MAX_VIDEO_WIDTH",
    "MultimodalPreprocessError",
    "MultimodalTensorBundle",
    "VideoPreprocessResult",
    "VideoProcessorIdentity",
    "VideoSamplingMetadata",
    "VerifiedGemma4MultimodalProcessor",
]
