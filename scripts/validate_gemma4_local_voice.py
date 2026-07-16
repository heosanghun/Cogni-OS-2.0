"""Validate local Windows speech -> Gemma 4 audio transcription end to end."""

from __future__ import annotations

from argparse import ArgumentParser
from base64 import b64decode, b64encode
from datetime import datetime, timezone
from difflib import SequenceMatcher
from io import BytesIO
from pathlib import Path
from typing import Sequence
import json
import re
import wave

import torch
from torch.nn import functional as F

from cogni_agent.local_voice import (
    Gemma4ModelSpeechTranscriber,
    LocalVoiceService,
    WindowsSpeechSynthesizer,
)
from cogni_agent.model_service import ModelService


PHRASE = "안녕하세요. 코그니보드 로컬 음성 검증입니다."


def _resample_to_voice_contract(content: bytes) -> bytes:
    with wave.open(BytesIO(content), "rb") as source:
        channels = source.getnchannels()
        sample_width = source.getsampwidth()
        source_rate = source.getframerate()
        frames = source.readframes(source.getnframes())
    if channels != 1 or sample_width != 2:
        raise RuntimeError("Windows TTS did not return mono 16-bit PCM")
    samples = torch.frombuffer(bytearray(frames), dtype=torch.int16).to(torch.float32)
    target_count = max(1, round(samples.numel() * 16_000 / source_rate))
    converted = (
        F.interpolate(
            samples.reshape(1, 1, -1),
            size=target_count,
            mode="linear",
            align_corners=False,
        )
        .round()
        .clamp(-32_768, 32_767)
        .to(torch.int16)
    )
    output = BytesIO()
    with wave.open(output, "wb") as target:
        target.setnchannels(1)
        target.setsampwidth(2)
        target.setframerate(16_000)
        target.writeframes(converted.numpy().tobytes())
    return output.getvalue()


def _normalized(value: str) -> str:
    return re.sub(r"[^0-9A-Za-z가-힣]+", "", value).casefold()


def run_validation(model: Path, manifest: Path) -> dict[str, object]:
    synthesis = WindowsSpeechSynthesizer().synthesize(text=PHRASE, language="ko")
    voice_wav = _resample_to_voice_contract(
        b64decode(str(synthesis["audio_wav_base64"]), validate=True)
    )
    service = ModelService.for_local_gemma(
        model,
        manifest_path=manifest,
        vram_limit_gib=16.7,
        max_input_tokens=4_096,
        max_new_tokens=256,
        max_prompt_chars=32_000,
        max_response_chars=4_096,
        startup_timeout=240.0,
        request_timeout=240.0,
    )
    observed: list[str] = []
    voice = LocalVoiceService.for_verified_gemma4(
        model,
        manifest,
        transcriber=Gemma4ModelSpeechTranscriber(
            service,
            output_observer=observed.append,
        ),
        synthesizer=WindowsSpeechSynthesizer(),
    )
    try:
        try:
            result = voice.transcribe_base64(
                b64encode(voice_wav).decode("ascii"), language="ko"
            )
        except Exception:
            if observed:
                print(
                    "raw_model_output=" + json.dumps(observed[-1], ensure_ascii=False),
                    flush=True,
                )
            raise
    finally:
        service.stop()
    transcript = str(result["transcript"])
    similarity = SequenceMatcher(
        None,
        _normalized(PHRASE),
        _normalized(transcript),
        autojunk=False,
    ).ratio()
    return {
        "schema_version": 1,
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "model_root": str(model),
        "manifest": str(manifest),
        "input_phrase": PHRASE,
        "transcript": transcript,
        "normalized_similarity": round(similarity, 6),
        "minimum_similarity": 0.70,
        "transcript_pass": similarity >= 0.70,
        "tts": {
            "voice": synthesis["voice"],
            "culture": synthesis["culture"],
            "external_calls": synthesis["external_calls"],
        },
        "stt": {
            "source": result["source"],
            "external_calls": result["external_calls"],
            "sample_rate_hz": result["audio"]["sample_rate_hz"],
            "duration_seconds": result["audio"]["duration_seconds"],
        },
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = ArgumentParser()
    parser.add_argument("--model", default=r"C:\Project\cognios\gemma4-e4b-it")
    parser.add_argument("--manifest", default="config/gemma4-e4b-it.manifest.toml")
    parser.add_argument("--output")
    args = parser.parse_args(argv)
    evidence = run_validation(
        Path(args.model).resolve(strict=True),
        Path(args.manifest).resolve(strict=True),
    )
    rendered = json.dumps(evidence, ensure_ascii=False, indent=2)
    print(rendered)
    if args.output:
        output = Path(args.output).resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_name(f".{output.name}.tmp")
        temporary.write_text(rendered + "\n", encoding="utf-8")
        temporary.replace(output)
    return 0 if evidence["transcript_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
