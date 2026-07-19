"""Bounded, local-only speech input boundary for CogniBoard.

Browser capture and WAV transport are useful even when no speech checkpoint is
installed.  This module therefore validates the complete capture contract
before it checks the optional local STT artifact.  It never invents a
transcript: an absent or unverified transcriber fails with the stable
``LOCAL_STT_ARTIFACT_REQUIRED`` code.
"""

from __future__ import annotations

from base64 import b64decode, b64encode
from binascii import Error as Base64Error
from array import array
from dataclasses import dataclass
from hashlib import sha256
from inspect import Parameter, signature
from io import BytesIO
from pathlib import Path
from subprocess import PIPE, TimeoutExpired, run
from threading import RLock
from typing import Any, Callable, Protocol
import json
import os
import re
import sys
import wave

from .multimodal import is_verified_multimodal_bundle


MAX_VOICE_WAV_BYTES = 2 * 1024 * 1024
MAX_VOICE_BASE64_CHARS = ((MAX_VOICE_WAV_BYTES + 2) // 3) * 4
MAX_VOICE_SECONDS = 30
VOICE_SAMPLE_RATE = 16_000
MAX_TRANSCRIPT_CHARS = 4_096
MAX_TTS_TEXT_CHARS = 2_000
MAX_TTS_WAV_BYTES = 8 * 1024 * 1024
_LANGUAGES = frozenset({"auto", "ko", "en"})


class LocalVoiceError(ValueError):
    """A bounded voice request failed at a stable, non-secret boundary."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True, slots=True)
class ValidatedVoiceWav:
    content: bytes
    frame_count: int
    duration_seconds: float
    sample_rate: int = VOICE_SAMPLE_RATE
    channels: int = 1
    sample_width_bytes: int = 2


class LocalSpeechTranscriber(Protocol):
    """Future worker boundary; implementations must be verified and offline."""

    local_only: bool
    artifact_verified: bool

    def transcribe(
        self,
        *,
        wav: ValidatedVoiceWav,
        tensor_bundle: Any,
        language: str,
    ) -> str: ...


class LocalSpeechSynthesizer(Protocol):
    """Offline, verified speech-output boundary."""

    local_only: bool
    artifact_verified: bool

    def synthesize(self, *, text: str, language: str) -> dict[str, object]: ...


class Gemma4ModelSpeechTranscriber:
    """Deterministic STT adapter over the already resident verified model.

    This adapter does not load a second checkpoint.  It accepts only a
    manifest-bound production ``ModelService`` whose generate method exposes
    the explicit bounded ``audio_wav_content`` parameter.
    """

    local_only = True
    artifact_verified = True

    def __init__(
        self,
        model_service: object,
        *,
        output_observer: Callable[[str], None] | None = None,
    ) -> None:
        try:
            generate = getattr(model_service, "generate")
            parameter = signature(generate).parameters.get("audio_wav_content")
            factory = getattr(model_service, "model_factory")
            manifest = Path(getattr(factory, "manifest_path")).resolve(strict=True)
            model_path = Path(getattr(factory, "model_path")).resolve(strict=True)
            configured = getattr(model_service, "_multimodal_processor_config")
            digest = getattr(model_service, "artifact_digest")
        except (AttributeError, OSError, TypeError, ValueError) as exc:
            raise ValueError("verified Gemma audio model service is required") from exc
        if parameter is None or parameter.kind not in {
            Parameter.POSITIONAL_OR_KEYWORD,
            Parameter.KEYWORD_ONLY,
        }:
            raise ValueError("model service lacks explicit audio input")
        if (
            not isinstance(configured, tuple)
            or len(configured) != 2
            or Path(configured[0]).resolve() != model_path
            or Path(configured[1]).resolve() != manifest
            or not isinstance(digest, str)
            or not re.fullmatch(r"[0-9a-f]{64}", digest)
            or sha256(manifest.read_bytes()).hexdigest() != digest
        ):
            raise ValueError("Gemma audio processor is not manifest-bound")
        if output_observer is not None and not callable(output_observer):
            raise TypeError("output observer must be callable")
        self._service = model_service
        self._output_observer = output_observer

    def transcribe(
        self,
        *,
        wav: ValidatedVoiceWav,
        tensor_bundle: Any,
        language: str,
    ) -> str:
        if not isinstance(wav, ValidatedVoiceWav) or not is_verified_multimodal_bundle(
            tensor_bundle, modality="audio"
        ):
            raise LocalVoiceError(
                "LOCAL_AUDIO_PREPROCESS_FAILED", "verified audio tensors are required"
            )
        samples = array("h")
        with wave.open(BytesIO(wav.content), "rb") as stream:
            samples.frombytes(stream.readframes(stream.getnframes()))
        if sys.byteorder != "little":
            samples.byteswap()
        if not samples:
            raise LocalVoiceError("VOICE_SILENCE_DETECTED", "voice audio is empty")
        rms = int((sum(int(value) ** 2 for value in samples) / len(samples)) ** 0.5)
        if rms < 12:
            raise LocalVoiceError(
                "VOICE_SILENCE_DETECTED", "voice audio contains no detectable speech"
            )
        language_instruction = {
            "ko": "한국어 음성을 들리는 그대로 한국어로",
            "en": "영어 음성을 들리는 그대로 영어로",
            "auto": "음성의 언어를 유지하여 들리는 그대로",
        }[language]
        prompt = (
            f"{language_instruction} 전사하세요. 설명, 요약, 번역, 화자 표기 없이 "
            "전사문 한 개만 출력하세요. 들리지 않으면 빈 답을 만들지 마세요."
        )
        conversation_id = f"local-stt-{sha256(wav.content).hexdigest()[:24]}"
        try:
            result = self._service.generate(
                prompt,
                audio_wav_content=wav.content,
                max_new_tokens=256,
                total_timeout=180.0,
                conversation_id=conversation_id,
                decode_mode="strict",
                sampling_seed=0,
            )
            transcript = getattr(result, "text")
        except LocalVoiceError:
            raise
        except Exception as exc:
            raise LocalVoiceError(
                "LOCAL_STT_FAILED", "the verified local Gemma STT request failed"
            ) from exc
        if not isinstance(transcript, str):
            raise LocalVoiceError(
                "LOCAL_STT_OUTPUT_INVALID", "local STT returned invalid text"
            )
        if self._output_observer is not None:
            self._output_observer(transcript)
        transcript = transcript.strip().strip("\"'“”‘’")
        transcript = re.sub(
            r"^(?:전사(?:문)?|transcript)\s*[:：]\s*",
            "",
            transcript,
            flags=re.IGNORECASE,
        ).strip()
        if (
            not transcript
            or len(transcript) > MAX_TRANSCRIPT_CHARS
            or re.search(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", transcript)
            or re.search(r"(?im)^\s*(?:system|user|assistant)\s*:", transcript)
            or transcript.count("\n") > 4
        ):
            raise LocalVoiceError(
                "LOCAL_STT_OUTPUT_INVALID", "local STT returned invalid text"
            )
        return transcript


_WINDOWS_TTS_SCRIPT = r"""
$ErrorActionPreference = 'Stop'
[Console]::InputEncoding = New-Object System.Text.UTF8Encoding($false)
[Console]::OutputEncoding = New-Object System.Text.UTF8Encoding($false)
Add-Type -AssemblyName System.Speech
$request = [Console]::In.ReadToEnd() | ConvertFrom-Json
$speaker = New-Object System.Speech.Synthesis.SpeechSynthesizer
try {
    $voices = @($speaker.GetInstalledVoices() | Where-Object { $_.Enabled })
    if ($voices.Count -lt 1) { throw 'no enabled Windows speech voice' }
    $culture = if ($request.language -eq 'ko') { 'ko-KR' } elseif ($request.language -eq 'en') { 'en-US' } else { '' }
    $selected = if ($culture) { $voices | Where-Object { $_.VoiceInfo.Culture.Name -eq $culture } | Select-Object -First 1 } else { $voices | Select-Object -First 1 }
    if ($null -eq $selected) { throw 'requested Windows speech culture is unavailable' }
    $speaker.SelectVoice($selected.VoiceInfo.Name)
    $stream = New-Object System.IO.MemoryStream
    try {
        $speaker.SetOutputToWaveStream($stream)
        $speaker.Speak([string]$request.text)
        $speaker.SetOutputToNull()
        $payload = [ordered]@{
            voice = $selected.VoiceInfo.Name
            culture = $selected.VoiceInfo.Culture.Name
            audio_wav_base64 = [Convert]::ToBase64String($stream.ToArray())
        }
        [Console]::Out.Write(($payload | ConvertTo-Json -Compress))
    } finally {
        $stream.Dispose()
    }
} finally {
    $speaker.Dispose()
}
""".strip()


class WindowsSpeechSynthesizer:
    """Use an installed Windows System.Speech voice without network access.

    The child command and script are fixed.  User text is transported only as
    JSON on standard input and is never interpolated into a command line.
    """

    local_only = True
    artifact_verified = True

    def __init__(self, powershell_path: str | Path | None = None) -> None:
        if sys.platform != "win32":
            raise OSError("Windows speech synthesis is unavailable")
        candidate = Path(
            powershell_path
            or Path(os.environ.get("SystemRoot", r"C:\Windows"))
            / "System32"
            / "WindowsPowerShell"
            / "v1.0"
            / "powershell.exe"
        ).resolve(strict=True)
        expected_root = Path(os.environ.get("SystemRoot", r"C:\Windows")).resolve(
            strict=True
        )
        try:
            candidate.relative_to(expected_root)
        except ValueError as exc:
            raise OSError("PowerShell must be the Windows system executable") from exc
        if candidate.name.casefold() != "powershell.exe":
            raise OSError("PowerShell executable is invalid")
        self._powershell = candidate
        self._encoded_script = b64encode(
            _WINDOWS_TTS_SCRIPT.encode("utf-16-le")
        ).decode("ascii")

    def synthesize(self, *, text: str, language: str) -> dict[str, object]:
        if (
            not isinstance(text, str)
            or not text.strip()
            or len(text) > MAX_TTS_TEXT_CHARS
            or re.search(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", text)
        ):
            raise LocalVoiceError("TTS_TEXT_INVALID", "speech text is invalid")
        if language not in _LANGUAGES:
            raise LocalVoiceError("VOICE_LANGUAGE_INVALID", "voice language is invalid")
        request = json.dumps(
            {"text": text.strip(), "language": language},
            ensure_ascii=False,
            separators=(",", ":"),
        )
        environment = {
            key: value
            for key in ("SystemRoot", "WINDIR", "TEMP", "TMP")
            if (value := os.environ.get(key))
        }
        try:
            completed = run(
                [
                    str(self._powershell),
                    "-NoLogo",
                    "-NoProfile",
                    "-NonInteractive",
                    "-ExecutionPolicy",
                    "Bypass",
                    "-EncodedCommand",
                    self._encoded_script,
                ],
                input=request.encode("utf-8"),
                stdout=PIPE,
                stderr=PIPE,
                timeout=30.0,
                check=False,
                shell=False,
                env=environment,
                creationflags=0x08000000,
            )
        except (OSError, TimeoutExpired, UnicodeError) as exc:
            raise LocalVoiceError(
                "LOCAL_TTS_FAILED", "local Windows speech synthesis failed"
            ) from exc
        if completed.returncode != 0 or len(completed.stdout) > 12 * 1024 * 1024:
            raise LocalVoiceError(
                "LOCAL_TTS_FAILED", "local Windows speech synthesis failed"
            )
        try:
            payload = json.loads(completed.stdout.decode("utf-8", errors="strict"))
            encoded = payload["audio_wav_base64"]
            voice = payload["voice"]
            culture = payload["culture"]
            if not isinstance(encoded, str) or not isinstance(voice, str):
                raise TypeError
            if not isinstance(culture, str) or language == "ko" and culture != "ko-KR":
                raise TypeError
            if language == "en" and culture != "en-US":
                raise TypeError
            content = b64decode(encoded, validate=True)
        except (
            Base64Error,
            KeyError,
            TypeError,
            ValueError,
            json.JSONDecodeError,
        ) as exc:
            raise LocalVoiceError(
                "LOCAL_TTS_OUTPUT_INVALID", "local TTS returned invalid output"
            ) from exc
        if not 44 <= len(content) <= MAX_TTS_WAV_BYTES:
            raise LocalVoiceError(
                "LOCAL_TTS_OUTPUT_INVALID", "local TTS WAV exceeds its limit"
            )
        try:
            with wave.open(BytesIO(content), "rb") as stream:
                duration = stream.getnframes() / float(stream.getframerate())
                channels = stream.getnchannels()
                sample_rate = stream.getframerate()
                if (
                    stream.getcomptype() != "NONE"
                    or not 0 < duration <= 120.0
                    or channels not in {1, 2}
                    or not 8_000 <= sample_rate <= 48_000
                ):
                    raise ValueError
        except (EOFError, OSError, ValueError, wave.Error) as exc:
            raise LocalVoiceError(
                "LOCAL_TTS_OUTPUT_INVALID", "local TTS returned an invalid WAV"
            ) from exc
        return {
            "audio_wav_base64": encoded,
            "voice": voice[:128],
            "culture": culture,
            "duration_seconds": round(duration, 4),
            "sample_rate_hz": sample_rate,
            "channels": channels,
            "source": "verified_windows_system_speech",
            "external_calls": 0,
        }


def validate_voice_wav(content: object) -> ValidatedVoiceWav:
    """Validate mono 16-bit PCM WAV at 16 kHz, capped at 30 seconds."""

    if not isinstance(content, bytes) or not 1 <= len(content) <= MAX_VOICE_WAV_BYTES:
        raise LocalVoiceError(
            "VOICE_AUDIO_TOO_LARGE", "voice WAV exceeds its byte limit"
        )
    try:
        with wave.open(BytesIO(content), "rb") as stream:
            channels = stream.getnchannels()
            sample_width = stream.getsampwidth()
            sample_rate = stream.getframerate()
            frame_count = stream.getnframes()
            compression = stream.getcomptype()
            if (
                channels != 1
                or sample_width != 2
                or sample_rate != VOICE_SAMPLE_RATE
                or compression != "NONE"
                or not 1 <= frame_count <= VOICE_SAMPLE_RATE * MAX_VOICE_SECONDS
            ):
                raise LocalVoiceError(
                    "VOICE_WAV_FORMAT_INVALID",
                    "voice audio must be mono 16-bit PCM WAV at 16 kHz and at most 30 seconds",
                )
            frames = stream.readframes(frame_count)
    except LocalVoiceError:
        raise
    except (EOFError, OSError, wave.Error) as exc:
        raise LocalVoiceError(
            "VOICE_WAV_INVALID", "voice WAV could not be decoded"
        ) from exc
    if len(frames) != frame_count * 2:
        raise LocalVoiceError("VOICE_WAV_TRUNCATED", "voice WAV is truncated")
    return ValidatedVoiceWav(
        content=content,
        frame_count=frame_count,
        duration_seconds=frame_count / VOICE_SAMPLE_RATE,
    )


class LocalVoiceService:
    """Validate browser audio, preprocess with Gemma4Processor, then call STT.

    The Gemma processor and STT worker are deliberately separate.  The former
    proves that the verified local checkpoint can produce a bounded audio
    tensor bundle; the latter is the only component allowed to produce text.
    """

    def __init__(
        self,
        *,
        preprocessor_factory: Callable[[], Any] | None = None,
        transcriber: LocalSpeechTranscriber | None = None,
        synthesizer: LocalSpeechSynthesizer | None = None,
        tts_host_probe_passed: bool = False,
    ) -> None:
        if not isinstance(tts_host_probe_passed, bool):
            raise TypeError("tts_host_probe_passed must be bool")
        if transcriber is not None and (
            getattr(transcriber, "local_only", False) is not True
            or getattr(transcriber, "artifact_verified", False) is not True
        ):
            raise ValueError(
                "local STT transcriber must be local-only and artifact-verified"
            )
        self._preprocessor_factory = preprocessor_factory
        self._processor: Any | None = None
        self._processor_probe_passed = False
        self._model_inference_attested = False
        self._transcriber = transcriber
        if synthesizer is not None and (
            getattr(synthesizer, "local_only", False) is not True
            or getattr(synthesizer, "artifact_verified", False) is not True
        ):
            raise ValueError(
                "local TTS synthesizer must be local-only and artifact-verified"
            )
        self._synthesizer = synthesizer
        self._tts_host_probe_passed = synthesizer is not None and tts_host_probe_passed
        self._lock = RLock()

    @classmethod
    def for_verified_gemma4(
        cls,
        model_root: str | Path,
        manifest_path: str | Path,
        *,
        transcriber: LocalSpeechTranscriber | None = None,
        synthesizer: LocalSpeechSynthesizer | None = None,
        tts_host_probe_passed: bool = False,
    ) -> "LocalVoiceService":
        root = Path(model_root)
        manifest = Path(manifest_path)

        def load_processor() -> Any:
            from cogni_agent.multimodal import VerifiedGemma4MultimodalProcessor

            return VerifiedGemma4MultimodalProcessor(root, manifest)

        return cls(
            preprocessor_factory=load_processor,
            transcriber=transcriber,
            synthesizer=synthesizer,
            tts_host_probe_passed=tts_host_probe_passed,
        )

    def capability_payload(self) -> dict[str, object]:
        transcriber_configured = self._transcriber is not None
        processor_configured = self._preprocessor_factory is not None
        with self._lock:
            processor_probe_passed = self._processor_probe_passed
            model_inference_attested = self._model_inference_attested
        transcription_ready = (
            transcriber_configured
            and processor_configured
            and processor_probe_passed
            and model_inference_attested
        )
        tts_configured = self._synthesizer is not None
        tts_ready = tts_configured and self._tts_host_probe_passed
        return {
            "state": (
                "ready"
                if transcription_ready
                else "configured_unverified"
                if transcriber_configured and processor_configured
                else "capture_transport_configured"
            ),
            "capture_state": "browser_get_user_media",
            "permission_state": "requested_only_on_user_action",
            "transport_state": "authenticated_loopback_ready",
            "capture_transport": {
                "state": "configured",
                "browser_feature_probe": "client_required",
                "permission_request": "user_action_only",
                "transport": "authenticated_loopback",
            },
            "external_calls": 0,
            "input_format": {
                "container": "wav",
                "encoding": "pcm_s16le",
                "sample_rate_hz": VOICE_SAMPLE_RATE,
                "channels": 1,
                "max_seconds": MAX_VOICE_SECONDS,
                "max_bytes": MAX_VOICE_WAV_BYTES,
            },
            "processor_state": (
                "probed"
                if processor_probe_passed
                else "configured_unverified"
                if processor_configured
                else "not_configured"
            ),
            "processor": {
                "configured": processor_configured,
                "probe_passed": processor_probe_passed,
                "state": (
                    "probed"
                    if processor_probe_passed
                    else "configured_unverified"
                    if processor_configured
                    else "not_configured"
                ),
            },
            "transcriber": {
                "configured": transcriber_configured,
                "artifact_verified": transcriber_configured,
                "state": (
                    "configured_unverified"
                    if transcriber_configured and not model_inference_attested
                    else "attested"
                    if model_inference_attested
                    else "not_configured"
                ),
            },
            "model_inference_attested": model_inference_attested,
            "attestation_path": (
                "successful_local_transcription_or_guarded_validation"
            ),
            "transcription_state": (
                "ready"
                if transcription_ready
                else "configured_unverified"
                if transcriber_configured and processor_configured
                else "local_artifact_required"
            ),
            "runtime_audio_input": transcription_ready,
            "stt": {
                "mode": "local_only",
                "artifact_verified": transcriber_configured,
                "runtime_ready": transcription_ready,
                "disabled_reason": (
                    None
                    if transcription_ready
                    else "LOCAL_STT_INFERENCE_UNVERIFIED"
                    if transcriber_configured and processor_configured
                    else "LOCAL_STT_ARTIFACT_REQUIRED"
                ),
            },
            "tts": {
                "state": (
                    "ready"
                    if tts_ready
                    else "configured_unverified"
                    if tts_configured
                    else "disabled"
                ),
                "mode": "local_only",
                "source": (
                    "verified_windows_system_speech" if tts_configured else None
                ),
                "host_probe_passed": self._tts_host_probe_passed,
                "browser_playback_verified": False,
                "browser_playback_state": "unverified",
                "disabled_reason": (
                    None
                    if tts_ready
                    else "LOCAL_TTS_HOST_PROBE_REQUIRED"
                    if tts_configured
                    else "LOCAL_TTS_ARTIFACT_REQUIRED"
                ),
            },
        }

    def synthesize(
        self, text: object, *, language: object = "auto"
    ) -> dict[str, object]:
        if self._synthesizer is None:
            raise LocalVoiceError(
                "LOCAL_TTS_ARTIFACT_REQUIRED",
                "a verified local TTS artifact is required before synthesis",
            )
        if not isinstance(text, str) or not isinstance(language, str):
            raise LocalVoiceError("TTS_TEXT_INVALID", "speech text is invalid")
        return self._synthesizer.synthesize(text=text, language=language)

    def transcribe_base64(
        self, audio_wav_base64: object, *, language: object = "auto"
    ) -> dict[str, object]:
        if (
            not isinstance(audio_wav_base64, str)
            or not 1 <= len(audio_wav_base64) <= MAX_VOICE_BASE64_CHARS
        ):
            raise LocalVoiceError(
                "VOICE_AUDIO_TOO_LARGE", "voice base64 exceeds its character limit"
            )
        if not isinstance(language, str) or language not in _LANGUAGES:
            raise LocalVoiceError("VOICE_LANGUAGE_INVALID", "voice language is invalid")
        try:
            content = b64decode(audio_wav_base64, validate=True)
        except (Base64Error, ValueError) as exc:
            raise LocalVoiceError(
                "VOICE_BASE64_INVALID", "voice base64 is invalid"
            ) from exc
        wav = validate_voice_wav(content)
        if self._transcriber is None:
            raise LocalVoiceError(
                "LOCAL_STT_ARTIFACT_REQUIRED",
                "a verified local STT artifact is required before transcription",
            )
        processor = self._processor_instance()
        try:
            tensor_bundle = processor.process_audio_wav(
                content, "다음 로컬 음성을 정확히 전사하세요."
            )
        except Exception as exc:
            raise LocalVoiceError(
                "LOCAL_AUDIO_PREPROCESS_FAILED",
                "verified local audio preprocessing failed",
            ) from exc
        if not is_verified_multimodal_bundle(tensor_bundle, modality="audio"):
            raise LocalVoiceError(
                "LOCAL_AUDIO_PREPROCESS_FAILED",
                "audio processor returned an unverified bundle",
            )
        with self._lock:
            self._processor_probe_passed = True
        try:
            transcript = self._transcriber.transcribe(
                wav=wav,
                tensor_bundle=tensor_bundle,
                language=language,
            )
        except LocalVoiceError:
            raise
        except Exception as exc:
            raise LocalVoiceError(
                "LOCAL_STT_FAILED", "the verified local STT worker failed"
            ) from exc
        if (
            not isinstance(transcript, str)
            or not transcript.strip()
            or len(transcript) > MAX_TRANSCRIPT_CHARS
            or re.search(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", transcript)
        ):
            raise LocalVoiceError(
                "LOCAL_STT_OUTPUT_INVALID", "local STT returned invalid text"
            )
        with self._lock:
            self._model_inference_attested = True
        return {
            "transcript": transcript.strip(),
            "language": language,
            "source": "verified_local_stt_artifact",
            "external_calls": 0,
            "audio": {
                "duration_seconds": round(wav.duration_seconds, 4),
                "sample_rate_hz": wav.sample_rate,
                "channels": wav.channels,
                "sample_width_bytes": wav.sample_width_bytes,
            },
        }

    def _processor_instance(self) -> Any:
        with self._lock:
            if self._processor is not None:
                return self._processor
            if self._preprocessor_factory is None:
                raise LocalVoiceError(
                    "LOCAL_AUDIO_PROCESSOR_REQUIRED",
                    "a verified local Gemma audio processor is required",
                )
            try:
                processor = self._preprocessor_factory()
            except Exception as exc:
                raise LocalVoiceError(
                    "LOCAL_AUDIO_PROCESSOR_REQUIRED",
                    "the verified local Gemma audio processor could not be loaded",
                ) from exc
            if not callable(getattr(processor, "process_audio_wav", None)):
                raise LocalVoiceError(
                    "LOCAL_AUDIO_PROCESSOR_REQUIRED",
                    "the configured local audio processor is invalid",
                )
            self._processor = processor
            return processor


__all__ = [
    "LocalSpeechTranscriber",
    "LocalSpeechSynthesizer",
    "LocalVoiceError",
    "LocalVoiceService",
    "MAX_VOICE_BASE64_CHARS",
    "MAX_VOICE_SECONDS",
    "MAX_VOICE_WAV_BYTES",
    "ValidatedVoiceWav",
    "VOICE_SAMPLE_RATE",
    "WindowsSpeechSynthesizer",
    "validate_voice_wav",
]
