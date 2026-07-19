from __future__ import annotations

from base64 import b64encode
from hashlib import sha256
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
import math
import struct
import tempfile
import unittest
import wave

import torch

from cogni_agent.local_voice import (
    Gemma4ModelSpeechTranscriber,
    LocalVoiceError,
    LocalVoiceService,
    VOICE_SAMPLE_RATE,
    validate_voice_wav,
)
from cogni_agent.multimodal import _validated_bundle


def _wav(
    *, sample_rate: int = VOICE_SAMPLE_RATE, channels: int = 1, frames: int = 800
) -> bytes:
    buffer = BytesIO()
    with wave.open(buffer, "wb") as stream:
        stream.setnchannels(channels)
        stream.setsampwidth(2)
        stream.setframerate(sample_rate)
        stream.writeframes(b"\x00\x00" * frames * channels)
    return buffer.getvalue()


def _speech_wav(frames: int = 3_200) -> bytes:
    payload = b"".join(
        struct.pack("<h", int(2_000 * math.sin(2 * math.pi * 220 * i / 16_000)))
        for i in range(frames)
    )
    buffer = BytesIO()
    with wave.open(buffer, "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(16_000)
        stream.writeframes(payload)
    return buffer.getvalue()


class _Processor:
    def __init__(self) -> None:
        self.calls = 0

    def process_audio_wav(self, content: bytes, prompt: str):
        self.calls += 1
        if not content or "전사" not in prompt:
            raise AssertionError("bounded audio prompt was not supplied")
        output = {
            "input_ids": torch.ones((1, 4), dtype=torch.int64),
            "attention_mask": torch.ones((1, 4), dtype=torch.int64),
            "mm_token_type_ids": torch.zeros((1, 4), dtype=torch.int64),
            "input_features": torch.ones((1, 4, 8), dtype=torch.float32),
            "input_features_mask": torch.ones((1, 4), dtype=torch.bool),
        }
        return _validated_bundle("audio", content, output, frozenset(output))


class _Transcriber:
    local_only = True
    artifact_verified = True

    def __init__(self) -> None:
        self.calls = 0

    def transcribe(self, *, wav, tensor_bundle, language):
        self.calls += 1
        if wav.sample_rate != VOICE_SAMPLE_RATE or tensor_bundle.modality != "audio":
            raise AssertionError("validated audio boundary was bypassed")
        return "안녕하세요. 로컬 음성 전사입니다."


class _BoundModelService:
    def __init__(self, root: Path, manifest: Path) -> None:
        self.model_factory = SimpleNamespace(model_path=root, manifest_path=manifest)
        self._multimodal_processor_config = (root, manifest)
        self.artifact_digest = sha256(manifest.read_bytes()).hexdigest()
        self.calls: list[dict[str, object]] = []
        self.result_text = "전사: 안녕하세요. 로컬 전사입니다."

    def generate(
        self,
        prompt,
        *,
        audio_wav_content=None,
        max_new_tokens=None,
        total_timeout=None,
        conversation_id=None,
        decode_mode=None,
        sampling_seed=None,
    ):
        self.calls.append(
            {
                "prompt": prompt,
                "audio_wav_content": audio_wav_content,
                "max_new_tokens": max_new_tokens,
                "total_timeout": total_timeout,
                "conversation_id": conversation_id,
                "decode_mode": decode_mode,
                "sampling_seed": sampling_seed,
            }
        )
        return SimpleNamespace(text=self.result_text)


class TestLocalVoice(unittest.TestCase):
    def test_pcm_contract_is_bounded(self) -> None:
        validated = validate_voice_wav(_wav(frames=1600))
        self.assertEqual(validated.sample_rate, 16_000)
        self.assertEqual(validated.channels, 1)
        self.assertAlmostEqual(validated.duration_seconds, 0.1)
        for candidate in (_wav(sample_rate=8_000), _wav(channels=2)):
            with self.assertRaisesRegex(LocalVoiceError, "mono 16-bit PCM") as caught:
                validate_voice_wav(candidate)
            self.assertEqual(caught.exception.code, "VOICE_WAV_FORMAT_INVALID")

    def test_missing_stt_artifact_fails_after_wav_validation(self) -> None:
        service = LocalVoiceService(preprocessor_factory=lambda: _Processor())
        payload = service.capability_payload()
        self.assertEqual(payload["capture_state"], "browser_get_user_media")
        self.assertEqual(payload["transport_state"], "authenticated_loopback_ready")
        self.assertEqual(payload["transcription_state"], "local_artifact_required")
        self.assertFalse(payload["runtime_audio_input"])
        self.assertTrue(payload["processor"]["configured"])
        self.assertFalse(payload["processor"]["probe_passed"])
        self.assertFalse(payload["transcriber"]["configured"])
        self.assertFalse(payload["model_inference_attested"])
        self.assertEqual(payload["external_calls"], 0)
        self.assertEqual(
            payload["tts"]["disabled_reason"], "LOCAL_TTS_ARTIFACT_REQUIRED"
        )
        encoded = b64encode(_wav()).decode("ascii")
        with self.assertRaises(LocalVoiceError) as caught:
            service.transcribe_base64(encoded, language="ko")
        self.assertEqual(caught.exception.code, "LOCAL_STT_ARTIFACT_REQUIRED")
        with self.assertRaises(LocalVoiceError) as invalid:
            service.transcribe_base64(
                b64encode(_wav(sample_rate=8_000)).decode("ascii"), language="ko"
            )
        self.assertEqual(invalid.exception.code, "VOICE_WAV_FORMAT_INVALID")

    def test_verified_local_worker_is_the_only_text_source(self) -> None:
        processor = _Processor()
        transcriber = _Transcriber()
        service = LocalVoiceService(
            preprocessor_factory=lambda: processor,
            transcriber=transcriber,
        )
        result = service.transcribe_base64(
            b64encode(_wav(frames=3200)).decode("ascii"), language="ko"
        )
        self.assertEqual(result["transcript"], "안녕하세요. 로컬 음성 전사입니다.")
        self.assertEqual(result["source"], "verified_local_stt_artifact")
        self.assertEqual(result["external_calls"], 0)
        self.assertEqual(processor.calls, 1)
        self.assertEqual(transcriber.calls, 1)
        capability = service.capability_payload()
        self.assertEqual(capability["transcription_state"], "ready")
        self.assertTrue(capability["processor"]["probe_passed"])
        self.assertTrue(capability["model_inference_attested"])
        self.assertTrue(capability["runtime_audio_input"])

    def test_configured_processor_failure_never_attests_runtime(self) -> None:
        def fail_processor():
            raise RuntimeError("processor failed")

        service = LocalVoiceService(
            preprocessor_factory=fail_processor,
            transcriber=_Transcriber(),
        )
        before = service.capability_payload()
        self.assertEqual(before["transcription_state"], "configured_unverified")
        self.assertFalse(before["runtime_audio_input"])
        self.assertFalse(before["processor"]["probe_passed"])
        self.assertFalse(before["model_inference_attested"])

        with self.assertRaises(LocalVoiceError) as caught:
            service.transcribe_base64(b64encode(_wav()).decode("ascii"), language="ko")
        self.assertEqual(caught.exception.code, "LOCAL_AUDIO_PROCESSOR_REQUIRED")
        after = service.capability_payload()
        self.assertEqual(after["transcription_state"], "configured_unverified")
        self.assertFalse(after["processor"]["probe_passed"])
        self.assertFalse(after["model_inference_attested"])

    def test_public_processor_verified_boolean_cannot_attest_voice_bundle(self) -> None:
        transcriber = _Transcriber()
        processor = SimpleNamespace(
            process_audio_wav=lambda _content, _prompt: SimpleNamespace(
                modality="audio", processor_verified=True
            )
        )
        service = LocalVoiceService(
            preprocessor_factory=lambda: processor,
            transcriber=transcriber,
        )
        with self.assertRaises(LocalVoiceError) as caught:
            service.transcribe_base64(
                b64encode(_speech_wav()).decode("ascii"), language="ko"
            )
        self.assertEqual(caught.exception.code, "LOCAL_AUDIO_PREPROCESS_FAILED")
        self.assertEqual(transcriber.calls, 0)
        self.assertFalse(service.capability_payload()["processor"]["probe_passed"])

    def test_unverified_or_nonlocal_transcriber_is_rejected(self) -> None:
        for local_only, artifact_verified in ((False, True), (True, False)):
            transcriber = SimpleNamespace(
                local_only=local_only,
                artifact_verified=artifact_verified,
                transcribe=lambda **_kwargs: "unsafe",
            )
            with self.assertRaises(ValueError):
                LocalVoiceService(transcriber=transcriber)

    def test_manifest_bound_resident_gemma_transcriber_is_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = root / "manifest.toml"
            manifest.write_text("schema = 1\n", encoding="utf-8")
            backend = _BoundModelService(root, manifest)
            service = LocalVoiceService(
                preprocessor_factory=lambda: _Processor(),
                transcriber=Gemma4ModelSpeechTranscriber(backend),
            )
            content = _speech_wav()
            result = service.transcribe_base64(
                b64encode(content).decode("ascii"), language="ko"
            )

        self.assertEqual(result["transcript"], "안녕하세요. 로컬 전사입니다.")
        self.assertEqual(len(backend.calls), 1)
        call = backend.calls[0]
        self.assertEqual(call["audio_wav_content"], content)
        self.assertEqual(call["decode_mode"], "strict")
        self.assertEqual(call["sampling_seed"], 0)
        self.assertEqual(call["max_new_tokens"], 256)
        self.assertIn("전사문 한 개만", call["prompt"])

    def test_manifest_bound_transcriber_rejects_silence_without_generation(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = root / "manifest.toml"
            manifest.write_text("schema = 1\n", encoding="utf-8")
            backend = _BoundModelService(root, manifest)
            service = LocalVoiceService(
                preprocessor_factory=lambda: _Processor(),
                transcriber=Gemma4ModelSpeechTranscriber(backend),
            )
            with self.assertRaises(LocalVoiceError) as caught:
                service.transcribe_base64(
                    b64encode(_wav()).decode("ascii"), language="auto"
                )

        self.assertEqual(caught.exception.code, "VOICE_SILENCE_DETECTED")
        self.assertEqual(backend.calls, [])

    def test_gemma_transcriber_rejects_unbound_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = root / "manifest.toml"
            manifest.write_text("schema = 1\n", encoding="utf-8")
            backend = _BoundModelService(root, manifest)
            backend.artifact_digest = "0" * 64
            with self.assertRaises(ValueError):
                Gemma4ModelSpeechTranscriber(backend)


if __name__ == "__main__":
    unittest.main()
