from __future__ import annotations

from base64 import b64encode
from http.client import HTTPConnection
from io import BytesIO
import json
from pathlib import Path
import tempfile
from threading import Thread
from types import SimpleNamespace
from unittest.mock import patch
import unittest
import wave

from cogni_agent.local_voice import LocalVoiceError, LocalVoiceService
from cogni_demo.server import DemoHTTPServer, _build_local_voice_service
from tests.test_demo_server import manager_for


def _wav() -> bytes:
    output = BytesIO()
    with wave.open(output, "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(16_000)
        stream.writeframes(b"\x00\x00" * 1_600)
    return output.getvalue()


class _Workspace:
    def capability_payload(self):
        return {"schema_version": 1, "microphone": {"state": "stale"}}

    def list_attachments(self):
        return {"items": [], "count": 0}


class _Processor:
    def process_audio_wav(self, _content, _prompt):
        return SimpleNamespace(modality="audio", processor_verified=True)


class _Transcriber:
    local_only = True
    artifact_verified = True

    def transcribe(self, **_kwargs):
        return "인증된 로컬 전사"


class _Synthesizer:
    local_only = True
    artifact_verified = True

    def synthesize(self, *, text, language):
        return {
            "audio_wav_base64": b64encode(_wav()).decode("ascii"),
            "voice": "검증 테스트 음성",
            "culture": "ko-KR" if language == "ko" else "en-US",
            "duration_seconds": 0.1,
            "sample_rate_hz": 16_000,
            "channels": 1,
            "source": "verified_windows_system_speech",
            "external_calls": 0,
            "text_length": len(text),
        }


class TestVoiceHTTPAPI(unittest.TestCase):
    def setUp(self) -> None:
        self.assets_context = tempfile.TemporaryDirectory()
        assets = Path(self.assets_context.name)
        (assets / "index.html").write_text("<main>Cogni</main>", encoding="utf-8")
        (assets / "app.css").write_text("body{}", encoding="utf-8")
        (assets / "app.js").write_text("void 0", encoding="utf-8")
        (assets / "favicon.svg").write_text("<svg/>", encoding="utf-8")
        self.server = DemoHTTPServer(
            manager_for("success"),
            assets,
            workspace_service=_Workspace(),
            voice_service=LocalVoiceService(
                preprocessor_factory=lambda: _Processor(),
                transcriber=_Transcriber(),
                synthesizer=_Synthesizer(),
            ),
            port=0,
            token="v" * 32,
            watchdog_timeout=None,
        )
        self.thread = Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        connection = self._connection()
        connection.request("GET", "/?token=" + self.server.token)
        response = connection.getresponse()
        response.read()
        self.assertEqual(response.status, 303)
        self.cookie = response.getheader("Set-Cookie").split(";", 1)[0]
        connection.close()

    def tearDown(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)
        self.assets_context.cleanup()

    def _connection(self) -> HTTPConnection:
        return HTTPConnection("127.0.0.1", self.server.server_port, timeout=5)

    def _post(
        self,
        body: dict[str, object],
        *,
        path: str = "/api/workspace/voice/transcribe",
        origin: bool = True,
    ):
        connection = self._connection()
        headers = {
            "Cookie": self.cookie,
            "Content-Type": "application/json",
        }
        if origin:
            headers["Origin"] = self.server.origin
        connection.request(
            "POST",
            path,
            body=json.dumps(body).encode("utf-8"),
            headers=headers,
        )
        response = connection.getresponse()
        payload = json.loads(response.read())
        status = response.status
        connection.close()
        return status, payload

    def test_capability_is_live_voice_service_state(self) -> None:
        connection = self._connection()
        connection.request(
            "GET", "/api/workspace/capabilities", headers={"Cookie": self.cookie}
        )
        response = connection.getresponse()
        payload = json.loads(response.read())
        self.assertEqual(response.status, 200)
        connection.close()
        self.assertEqual(
            payload["microphone"]["capture_state"], "browser_get_user_media"
        )
        self.assertEqual(
            payload["microphone"]["transport_state"], "authenticated_loopback_ready"
        )
        self.assertEqual(payload["microphone"]["external_calls"], 0)
        self.assertEqual(payload["microphone"]["tts"]["state"], "ready")
        self.assertEqual(
            payload["microphone"]["tts"]["source"],
            "verified_windows_system_speech",
        )

    def test_authenticated_loopback_transcription_and_fail_closed_gate(self) -> None:
        body = {
            "audio_wav_base64": b64encode(_wav()).decode("ascii"),
            "language": "ko",
        }
        status, payload = self._post(body)
        self.assertEqual(status, 200)
        self.assertEqual(payload["transcript"], "인증된 로컬 전사")
        self.assertEqual(payload["external_calls"], 0)
        self.assertEqual(self._post(body, origin=False)[0], 403)

        self.server.voice_service = LocalVoiceService(
            preprocessor_factory=lambda: _Processor()
        )
        status, payload = self._post(body)
        self.assertEqual(status, 400)
        self.assertEqual(payload["error"]["code"], "LOCAL_STT_ARTIFACT_REQUIRED")

    def test_local_synthesis_and_exact_body_contract(self) -> None:
        status, payload = self._post(
            {"text": "마지막 답변", "language": "ko"},
            path="/api/workspace/voice/synthesize",
        )
        self.assertEqual(status, 200)
        self.assertEqual(payload["external_calls"], 0)
        self.assertEqual(payload["source"], "verified_windows_system_speech")
        self.assertTrue(payload["audio_wav_base64"].startswith("UklGR"))

        status, payload = self._post(
            {"text": "답변", "language": "ko", "unexpected": True},
            path="/api/workspace/voice/synthesize",
        )
        self.assertEqual(status, 400)
        self.assertEqual(payload["error"]["code"], "INVALID_BODY")

    def test_voice_routes_do_not_depend_on_workspace_service(self) -> None:
        self.server.workspace_service = None
        status, payload = self._post(
            {"text": "로컬 음성"},
            path="/api/workspace/voice/synthesize",
        )
        self.assertEqual(status, 200)
        self.assertEqual(payload["external_calls"], 0)

    def test_synthesis_fails_closed_without_verified_voice(self) -> None:
        self.server.voice_service = LocalVoiceService(
            preprocessor_factory=lambda: _Processor(),
            transcriber=_Transcriber(),
        )
        status, payload = self._post(
            {"text": "읽어주세요"},
            path="/api/workspace/voice/synthesize",
        )
        self.assertEqual(status, 400)
        self.assertEqual(payload["error"]["code"], "LOCAL_TTS_ARTIFACT_REQUIRED")

    def test_voice_compute_is_mutually_exclusive(self) -> None:
        body = {"audio_wav_base64": b64encode(_wav()).decode("ascii")}
        for active_check in ("_agent_active", "_evolution_active"):
            with self.subTest(active_check=active_check):
                with patch.object(self.server, active_check, return_value=True):
                    status, payload = self._post(body)
                self.assertEqual(status, 409)
                self.assertEqual(payload["error"]["code"], "COMPUTE_BUSY")

        with self.server.manager._condition:
            self.server.manager._status = "running"
        try:
            status, payload = self._post(body)
        finally:
            with self.server.manager._condition:
                self.server.manager._status = "ready"
        self.assertEqual(status, 409)
        self.assertEqual(payload["error"]["code"], "COMPUTE_BUSY")


class TestVoiceProductComposition(unittest.TestCase):
    def test_reuses_exact_agent_model_service_and_probes_tts(self) -> None:
        model_service = object()
        agent = SimpleNamespace(model_service=model_service)
        service = object()
        with (
            patch("cogni_demo.server.Gemma4ModelSpeechTranscriber") as transcriber,
            patch("cogni_demo.server.WindowsSpeechSynthesizer") as synthesizer,
            patch.object(
                LocalVoiceService,
                "for_verified_gemma4",
                return_value=service,
            ) as factory,
        ):
            synthesizer.return_value.synthesize.return_value = {
                "audio_wav_base64": "UklGRg==",
                "source": "verified_windows_system_speech",
                "external_calls": 0,
            }
            result = _build_local_voice_service("model", "manifest", agent)

        self.assertIs(result, service)
        transcriber.assert_called_once_with(model_service)
        synthesizer.return_value.synthesize.assert_called_once_with(
            text="Cogni", language="auto"
        )
        factory.assert_called_once_with(
            "model",
            "manifest",
            transcriber=transcriber.return_value,
            synthesizer=synthesizer.return_value,
        )

    def test_failed_windows_probe_keeps_tts_disabled(self) -> None:
        agent = SimpleNamespace(model_service=object())
        with (
            patch("cogni_demo.server.Gemma4ModelSpeechTranscriber") as transcriber,
            patch("cogni_demo.server.WindowsSpeechSynthesizer") as synthesizer,
            patch.object(
                LocalVoiceService,
                "for_verified_gemma4",
                return_value=object(),
            ) as factory,
        ):
            synthesizer.return_value.synthesize.side_effect = LocalVoiceError(
                "LOCAL_TTS_FAILED", "probe failed"
            )
            _build_local_voice_service("model", "manifest", agent)

        factory.assert_called_once_with(
            "model",
            "manifest",
            transcriber=transcriber.return_value,
            synthesizer=None,
        )


if __name__ == "__main__":
    unittest.main()
