from __future__ import annotations

from pathlib import Path
import shutil
import subprocess
import unittest


STATIC = Path(__file__).resolve().parents[1] / "cogni_demo" / "static"
SERVER = Path(__file__).resolve().parents[1] / "cogni_demo" / "server.py"
_BUNDLED_NODE = (
    Path.home()
    / ".cache"
    / "codex-runtimes"
    / "codex-primary-runtime"
    / "dependencies"
    / "node"
    / "bin"
    / "node.exe"
)
NODE = shutil.which("node") or (str(_BUNDLED_NODE) if _BUNDLED_NODE.is_file() else None)


def _source_between(source: str, start_marker: str, end_marker: str) -> str:
    start = source.index(start_marker)
    return source[start : source.index(end_marker, start)]


class TestVoiceUIContract(unittest.TestCase):
    def test_capture_transport_and_controls_are_wired(self) -> None:
        html = (STATIC / "index.html").read_text(encoding="utf-8")
        script = (STATIC / "app.js").read_text(encoding="utf-8")
        stylesheet = (STATIC / "app.css").read_text(encoding="utf-8")

        self.assertIn("MICROPHONE_CAPTURE_UI_IMPLEMENTED = true", script)
        self.assertIn("navigator.mediaDevices.getUserMedia", script)
        self.assertIn('typeof window.MediaRecorder !== "function"', script)
        self.assertIn("function browserMicrophoneSupport()", script)
        self.assertIn("window.isSecureContext !== true", script)
        self.assertIn("ui.voiceBrowserCaptureReady", script)
        self.assertIn("processor.probe_passed === true", script)
        self.assertIn("microphone.model_inference_attested === true", script)
        self.assertIn("voiceTranscriptionConfigured: false", script)
        self.assertIn(
            'microphone.transcription_state === "configured_unverified"', script
        )
        self.assertIn('stt.mode === "local_only"', script)
        self.assertIn("transcriber.artifact_verified === true", script)
        self.assertIn(
            "ui.voiceTranscriptionAttemptReady = ui.voiceTranscriptionReady", script
        )
        self.assertIn("|| !ui.voiceTranscriptionAttemptReady", script)
        controls_start = script.index("function updateWorkspaceControlStates")
        controls_end = script.index("async function loadWorkspaceCapabilities")
        controls = script[controls_start:controls_end]
        capture_start = script.index("async function startVoiceCapture")
        capture_end = script.index("async function stopVoiceCapture", capture_start)
        capture = script[capture_start:capture_end]
        self.assertNotIn("|| !ui.voiceTranscriptionReady", controls)
        self.assertNotIn("|| !ui.voiceTranscriptionReady", capture)
        self.assertNotIn("createScriptProcessor", script)
        self.assertIn("new window.MediaRecorder(session.stream", script)
        self.assertIn("session.recorder.start(100)", script)
        self.assertIn("VOICE_SAMPLE_RATE = 16000", script)
        self.assertIn("VOICE_MAX_RECORDED_BYTES = 4 * 1024 * 1024", script)
        self.assertIn("VOICE_RECORDER_STOP_TIMEOUT_MS = 5000", script)
        self.assertIn("VOICE_DECODE_TIMEOUT_MS = 5000", script)
        self.assertIn("function stopVoiceRecorder(session)", script)
        self.assertIn("async function decodeVoiceRecording(blob, signal)", script)
        self.assertIn("context.decodeAudioData(encoded.slice(0))", script)
        self.assertIn("session.pendingStopSettle?.(voiceAbortError())", script)
        self.assertIn("session.pendingStopSettle = settle", script)
        self.assertIn("requireActiveVoiceSession(session)", script)
        self.assertIn("if (channel.length > maxFrames)", script)
        self.assertIn("await context.close().catch(() => {})", script)
        self.assertNotIn("context.resume()", script)
        self.assertIn('writeAscii(0, "RIFF")', script)
        self.assertIn('writeAscii(8, "WAVE")', script)
        self.assertIn("/api/workspace/voice/transcribe", script)
        self.assertIn("LOCAL_STT_ARTIFACT_REQUIRED", script)
        media_index = capture.index("navigator.mediaDevices.getUserMedia")
        recorder_index = capture.index("new window.MediaRecorder", media_index)
        start_index = capture.index("session.recorder.start(100)", recorder_index)
        self.assertLess(media_index, recorder_index)
        self.assertLess(recorder_index, start_index)
        stop_start = script.index("async function stopVoiceCapture")
        stop_end = script.index("function cancelVoiceCapture", stop_start)
        stop_flow = script[stop_start:stop_end]
        recorder_stop_index = stop_flow.index("await stopVoiceRecorder(session)")
        blob_index = stop_flow.index(
            "new Blob(session.encodedChunks", recorder_stop_index
        )
        decode_index = stop_flow.index(
            "await decodeVoiceRecording(recorded, session.abortController.signal)",
            blob_index,
        )
        response_index = stop_flow.index('await api("/api/workspace/voice/transcribe"')
        refresh_index = stop_flow.index("await refreshWorkspaceCapabilityDisclosure()")
        ready_index = stop_flow.index("!ui.voiceTranscriptionReady", refresh_index)
        insert_index = stop_flow.index("insertVoiceTranscript", ready_index)
        self.assertLess(recorder_stop_index, blob_index)
        self.assertLess(blob_index, decode_index)
        self.assertLess(decode_index, response_index)
        self.assertLess(response_index, refresh_index)
        self.assertLess(refresh_index, ready_index)
        self.assertLess(ready_index, insert_index)
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

    @unittest.skipUnless(NODE, "Node.js is required for deterministic voice race tests")
    def test_cancel_settles_pending_recorder_stop_once(self) -> None:
        script = (STATIC / "app.js").read_text(encoding="utf-8")
        release_source = _source_between(
            script,
            "function voiceAbortError()",
            "function resampleVoiceSamples",
        )
        stop_source = _source_between(
            script,
            "function stopVoiceRecorder(session)",
            "async function decodeVoiceRecording",
        )
        node_source = "\n".join(
            [
                "globalThis.window = { setTimeout, clearTimeout };",
                "const VOICE_RECORDER_STOP_TIMEOUT_MS = 5000;",
                "const ui = { voiceSession: null };",
                release_source,
                stop_source,
                r"""
function assert(condition, message) {
  if (!condition) throw new Error(message);
}
class FakeRecorder {
  constructor() {
    this.state = "recording";
    this.stopCalls = 0;
    this.listeners = new Map();
  }
  addEventListener(type, callback) { this.listeners.set(type, callback); }
  removeEventListener(type, callback) {
    if (this.listeners.get(type) === callback) this.listeners.delete(type);
  }
  stop() {
    this.stopCalls += 1;
    this.state = "inactive";
  }
}
(async () => {
  const recorder = new FakeRecorder();
  let trackStops = 0;
  const session = {
    recorder,
    stream: { getTracks: () => [{ stop: () => { trackStops += 1; } }] },
    stopRequested: false,
  };
  const pending = stopVoiceRecorder(session);
  assert(typeof session.pendingStopSettle === "function", "missing pending stop settler");
  releaseVoiceSession(session);
  const outcome = await Promise.race([
    pending.then(
      () => ({ resolved: true }),
      error => ({ rejected: true, name: error?.name }),
    ),
    new Promise(resolve => setTimeout(() => resolve({ timeout: true }), 100)),
  ]);
  assert(outcome.rejected && outcome.name === "AbortError", "cancel did not reject with AbortError");
  assert(!outcome.timeout, "recorder stop promise remained pending");
  assert(recorder.stopCalls === 1, "recorder.stop was not exactly once");
  assert(recorder.listeners.size === 0, "recorder listeners leaked");
  assert(session.pendingStopSettle == null, "pending settler leaked");
  assert(trackStops === 1, "media track was not released exactly once");
})().catch(error => { console.error(error.stack || error); process.exit(1); });
""",
            ]
        )
        completed = subprocess.run(
            [str(NODE), "--input-type=module", "--eval", node_source],
            capture_output=True,
            check=False,
            encoding="utf-8",
            timeout=10,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)

    @unittest.skipUnless(NODE, "Node.js is required for deterministic voice race tests")
    def test_cancel_during_decode_blocks_post_and_duplicate_stop(self) -> None:
        script = (STATIC / "app.js").read_text(encoding="utf-8")
        release_source = _source_between(
            script,
            "function voiceAbortError()",
            "function resampleVoiceSamples",
        )
        stop_flow_source = _source_between(
            script,
            "async function stopVoiceCapture()",
            "function toggleVoiceCapture()",
        )
        node_source = "\n".join(
            [
                "globalThis.window = { setTimeout, clearTimeout };",
                "const ui = { voiceSession: null, voiceCaptureState: 'idle', voiceTranscriptionReady: true };",
                "const API_ERROR_COPY = {};",
                release_source,
                r"""
function deferred() {
  let resolve;
  let reject;
  const promise = new Promise((yes, no) => { resolve = yes; reject = no; });
  return { promise, resolve, reject };
}
function assert(condition, message) {
  if (!condition) throw new Error(message);
}
let apiCalls = 0;
let refreshCalls = 0;
let inserted = 0;
let decodeGate = deferred();
let postGate = null;
function setVoiceCaptureState(state) { ui.voiceCaptureState = state; }
function updateWorkspaceControlStates() {}
function showToast() {}
async function stopVoiceRecorder(session) {
  session.stopRequested = true;
  session.recorder.state = "inactive";
}
let decodeVoiceRecording = () => decodeGate.promise;
function resampleVoiceSamples(samples) { return samples; }
function encodeVoiceWav() { return new Uint8Array([1, 2, 3]); }
function voiceBytesToBase64() { return "AQID"; }
let api = () => {
  apiCalls += 1;
  return postGate ? postGate.promise : Promise.resolve({ transcript: "unexpected" });
};
async function refreshWorkspaceCapabilityDisclosure() {
  refreshCalls += 1;
  return true;
}
function insertVoiceTranscript() { inserted += 1; return true; }
function makeSession() {
  let trackStopped = false;
  return {
    cancelled: false,
    abortController: new AbortController(),
    encodedChunks: [new Blob(["encoded"], { type: "audio/ogg" })],
    recorder: { state: "recording", mimeType: "audio/ogg", ondataavailable: null, onerror: null },
    stream: { getTracks: () => [{ stop: () => { trackStopped = true; } }] },
    stopRequested: false,
    transcriptionStarted: false,
    trackWasStopped: () => trackStopped,
  };
}
""",
                stop_flow_source,
                r"""
(async () => {
  const duringDecode = makeSession();
  ui.voiceSession = duringDecode;
  ui.voiceCaptureState = "recording";
  const decodingRun = stopVoiceCapture();
  await Promise.resolve();
  await Promise.resolve();
  assert(ui.voiceCaptureState === "encoding", "decode phase was not reached");
  cancelVoiceCapture();
  decodeGate.resolve({ samples: new Float32Array([0]), sourceRate: 16000 });
  await decodingRun;
  assert(apiCalls === 0, "cancelled decode issued a transcription POST");
  assert(refreshCalls === 0 && inserted === 0, "cancelled decode applied downstream work");
  assert(ui.voiceSession === null && ui.voiceCaptureState === "idle", "cancel did not restore idle UI");
  assert(duringDecode.trackWasStopped(), "cancelled decode leaked its media track");

  decodeVoiceRecording = async () => ({ samples: new Float32Array([0]), sourceRate: 16000 });
  postGate = deferred();
  const duringPost = makeSession();
  ui.voiceSession = duringPost;
  ui.voiceCaptureState = "recording";
  const first = stopVoiceCapture();
  const duplicate = stopVoiceCapture();
  await duplicate;
  for (let index = 0; index < 8 && apiCalls < 1; index += 1) await Promise.resolve();
  assert(apiCalls === 1, "duplicate stop issued an unexpected POST count");
  cancelVoiceCapture();
  postGate.resolve({ transcript: "must not be inserted" });
  await first;
  assert(apiCalls === 1, "cancelled POST was duplicated");
  assert(refreshCalls === 0 && inserted === 0, "cancelled POST applied stale output");
  assert(ui.voiceSession === null && ui.voiceCaptureState === "idle", "POST cancel did not restore idle UI");
})().catch(error => { console.error(error.stack || error); process.exit(1); });
""",
            ]
        )
        completed = subprocess.run(
            [str(NODE), "--input-type=module", "--eval", node_source],
            capture_output=True,
            check=False,
            encoding="utf-8",
            timeout=10,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)


if __name__ == "__main__":
    unittest.main()
