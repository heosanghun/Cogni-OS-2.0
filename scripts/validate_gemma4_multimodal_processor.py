"""Verify local Gemma 4 image/audio preprocessing without loading model weights."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from io import BytesIO
import json
from pathlib import Path
import wave

from PIL import Image

from cogni_agent.multimodal import VerifiedGemma4MultimodalProcessor


def _image() -> bytes:
    output = BytesIO()
    Image.new("RGB", (16, 16), "navy").save(output, format="PNG")
    return output.getvalue()


def _audio() -> bytes:
    output = BytesIO()
    with wave.open(output, "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(16_000)
        stream.writeframes(b"\0\0" * 800)
    return output.getvalue()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output")
    args = parser.parse_args()

    processor = VerifiedGemma4MultimodalProcessor(args.model, args.manifest)
    image = processor.process_image(_image(), "이미지를 한 문장으로 설명하세요.")
    audio = processor.process_audio_wav(_audio(), "오디오를 확인하세요.")
    payload = {
        "schema_version": 1,
        "measured_at": datetime.now(timezone.utc).isoformat(),
        "scope": "verified_cpu_multimodal_preprocessing_only",
        "model_forward_executed": False,
        "processor": type(processor.processor).__name__,
        "image": {
            "keys": sorted(image.as_mapping()),
            "tensor_bytes": image.tensor_bytes,
            "finite": True,
        },
        "audio": {
            "keys": sorted(audio.as_mapping()),
            "tensor_bytes": audio.tensor_bytes,
            "finite": True,
        },
    }
    encoded = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        destination = Path(args.output).expanduser().resolve(strict=False)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(encoded, encoding="utf-8", newline="\n")
    print(encoded, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
