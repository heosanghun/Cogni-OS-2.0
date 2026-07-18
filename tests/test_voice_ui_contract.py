from __future__ import annotations

from pathlib import Path
import unittest


STATIC = Path(__file__).resolve().parents[1] / "cogni_demo" / "static"
SERVER = Path(__file__).resolve().parents[1] / "cogni_demo" / "server.py"


class TestVoiceUIContract(unittest.TestCase):
    def test_capture_transport_and_controls_are_wired(self) -> None:
        html = (STATIC / "index.html").read_text(encoding="utf-8")
        script = (STATIC / "app.js").read_text(encoding="utf-8")
        stylesheet = (STATIC / "app.css").read_text(encoding="utf-8")

        self.assertIn("MICROPHONE_CAPTURE_UI_IMPLEMENTED = true", script)
        self.assertIn("navigator.mediaDevices.getUserMedia", script)
        self.assertIn("function browserMicrophoneSupport()", script)
        self.assertIn("window.isSecureContext !== true", script)
        self.assertIn("ui.voiceBrowserCaptureReady", script)
        self.assertIn("processor.probe_passed === true", script)
        self.assertIn("microphone.model_inference_attested === true", script)
        self.assertIn("createScriptProcessor(4096, 1, 1)", script)
        self.assertIn("VOICE_SAMPLE_RATE = 16000", script)
        self.assertIn('writeAscii(0, "RIFF")', script)
        self.assertIn('writeAscii(8, "WAVE")', script)
        self.assertIn("/api/workspace/voice/transcribe", script)
        self.assertIn("LOCAL_STT_ARTIFACT_REQUIRED", script)
        self.assertIn('data-action="workspace-voice-stop"', html)
        self.assertIn('data-action="workspace-voice-cancel"', html)
        self.assertIn('id="agent-voice-capture"', html)
        self.assertIn(".voice-capture-panel", stylesheet)
        self.assertIn('.voice-capture-panel[data-state="recording"]', stylesheet)

    def test_last_answer_tts_uses_revocable_blob_playback(self) -> None:
        html = (STATIC / "index.html").read_text(encoding="utf-8")
        script = (STATIC / "app.js").read_text(encoding="utf-8")
        stylesheet = (STATIC / "app.css").read_text(encoding="utf-8")
        server = SERVER.read_text(encoding="utf-8")

        self.assertIn("/api/workspace/voice/synthesize", script)
        self.assertIn("MAX_TTS_TEXT_CHARS = 2000", script)
        self.assertIn('data-role="assistant"]:not(.is-streaming)', script)
        self.assertIn('new Blob([bytes], { type: "audio/wav" })', script)
        self.assertIn("URL.createObjectURL(blob)", script)
        self.assertIn("URL.revokeObjectURL(ui.voicePlaybackObjectUrl)", script)
        self.assertIn('window.addEventListener("pagehide"', script)
        self.assertNotIn("data:audio", script)

        self.assertIn('data-action="workspace-tts-play"', html)
        self.assertIn('data-action="workspace-tts-stop"', html)
        self.assertIn('id="agent-voice-playback"', html)
        self.assertIn('role="status" aria-live="polite"', html)
        self.assertIn(".voice-playback-panel", stylesheet)
        self.assertIn('.voice-playback-panel[data-state="playing"]', stylesheet)
        self.assertIn("media-src 'self' blob:", server)
        self.assertIn('"Permissions-Policy", "microphone=(self)"', server)
        self.assertIn("tts.host_probe_passed === true", script)
        self.assertIn("tts.browser_playback_verified === true", script)


if __name__ == "__main__":
    unittest.main()
