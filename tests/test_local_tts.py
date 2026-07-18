from __future__ import annotations

from base64 import b64encode
from io import BytesIO
from types import SimpleNamespace
from unittest.mock import patch
import json
import unittest
import wave

from cogni_agent.local_voice import (
    LocalVoiceError,
    LocalVoiceService,
    WindowsSpeechSynthesizer,
)


def _wav() -> bytes:
    buffer = BytesIO()
    with wave.open(buffer, "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(22_050)
        stream.writeframes(b"\x00\x00" * 2_205)
    return buffer.getvalue()


class _Synthesizer:
    local_only = True
    artifact_verified = True

    def synthesize(self, *, text: str, language: str) -> dict[str, object]:
        return {
            "audio_wav_base64": b64encode(_wav()).decode("ascii"),
            "voice": "test",
            "culture": "ko-KR",
            "duration_seconds": 0.1,
            "sample_rate_hz": 22_050,
            "channels": 1,
            "source": "verified_windows_system_speech",
            "external_calls": 0,
        }


class TestLocalTTS(unittest.TestCase):
    def test_service_is_capability_gated(self) -> None:
        disabled = LocalVoiceService()
        self.assertEqual(disabled.capability_payload()["tts"]["state"], "disabled")
        with self.assertRaises(LocalVoiceError) as caught:
            disabled.synthesize("안녕하세요", language="ko")
        self.assertEqual(caught.exception.code, "LOCAL_TTS_ARTIFACT_REQUIRED")

        configured = LocalVoiceService(synthesizer=_Synthesizer())
        configured_tts = configured.capability_payload()["tts"]
        self.assertEqual(configured_tts["state"], "configured_unverified")
        self.assertFalse(configured_tts["host_probe_passed"])
        self.assertFalse(configured_tts["browser_playback_verified"])

        enabled = LocalVoiceService(
            synthesizer=_Synthesizer(), tts_host_probe_passed=True
        )
        self.assertEqual(enabled.capability_payload()["tts"]["state"], "ready")
        self.assertTrue(enabled.capability_payload()["tts"]["host_probe_passed"])
        self.assertFalse(
            enabled.capability_payload()["tts"]["browser_playback_verified"]
        )
        result = enabled.synthesize("안녕하세요", language="ko")
        self.assertEqual(result["external_calls"], 0)
        self.assertEqual(result["source"], "verified_windows_system_speech")

    def test_unverified_synthesizer_is_rejected(self) -> None:
        for local_only, artifact_verified in ((False, True), (True, False)):
            synthesizer = SimpleNamespace(
                local_only=local_only,
                artifact_verified=artifact_verified,
                synthesize=lambda **_kwargs: {},
            )
            with self.assertRaises(ValueError):
                LocalVoiceService(synthesizer=synthesizer)

    @unittest.skipUnless(__import__("sys").platform == "win32", "Windows only")
    def test_windows_tts_uses_fixed_command_and_stdin_json(self) -> None:
        payload = {
            "audio_wav_base64": b64encode(_wav()).decode("ascii"),
            "voice": "Microsoft Heami Desktop",
            "culture": "ko-KR",
        }
        captured: dict[str, object] = {}

        def fake_run(command, **kwargs):
            captured["command"] = command
            captured.update(kwargs)
            return SimpleNamespace(
                returncode=0,
                stdout=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                stderr=b"",
            )

        synthesizer = WindowsSpeechSynthesizer()
        with patch("cogni_agent.local_voice.run", side_effect=fake_run):
            result = synthesizer.synthesize(
                text='안녕; Write-Output "injection"', language="ko"
            )
        command = captured["command"]
        self.assertIsInstance(command, list)
        self.assertNotIn("안녕", " ".join(command))
        self.assertFalse(captured["shell"])
        request = json.loads(captured["input"].decode("utf-8"))
        self.assertEqual(request["text"], '안녕; Write-Output "injection"')
        self.assertEqual(result["voice"], "Microsoft Heami Desktop")
        self.assertEqual(result["source"], "verified_windows_system_speech")
        self.assertEqual(result["external_calls"], 0)

    @unittest.skipUnless(__import__("sys").platform == "win32", "Windows only")
    def test_windows_tts_rejects_untrusted_output(self) -> None:
        synthesizer = WindowsSpeechSynthesizer()
        with patch(
            "cogni_agent.local_voice.run",
            return_value=SimpleNamespace(returncode=0, stdout=b"{}", stderr=b""),
        ):
            with self.assertRaises(LocalVoiceError) as caught:
                synthesizer.synthesize(text="안녕하세요", language="ko")
        self.assertEqual(caught.exception.code, "LOCAL_TTS_OUTPUT_INVALID")


if __name__ == "__main__":
    unittest.main()
