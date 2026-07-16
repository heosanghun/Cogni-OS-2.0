"""Run one bounded, deterministic Gemma 4 image-understanding check.

The validation image is generated locally and contains one large blue square.
Passing requires the model response to identify both the colour and the exact
shape.  This is deliberately a narrow smoke test, not a broad multimodal
quality or benchmark claim.
"""

from __future__ import annotations

from argparse import ArgumentParser
from datetime import datetime, timezone
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
import re
from typing import Sequence

from PIL import Image, ImageDraw

from cogni_agent.model_service import ModelService


IMAGE_SIZE = (256, 256)
BACKGROUND_RGB = (250, 250, 247)
SQUARE_RGB = (24, 110, 210)
SQUARE_BOUNDS = (48, 48, 207, 207)
QUESTION = (
    "이 이미지를 보고 중앙의 큰 도형 하나의 색상과 모양만 한 문장으로 "
    "답하세요. 설명이나 추측을 덧붙이지 마세요. "
    "Answer only the central shape's color and shape in one sentence."
)
MAX_RESPONSE_CHARS = 512
_COLOR_TERMS = ("파란", "파랑", "청색", "blue")
_SHAPE_TERMS = ("정사각", "사각형", "네모", "square")


def validation_png() -> bytes:
    """Return the exact bounded PNG used by the runtime validation."""

    image = Image.new("RGB", IMAGE_SIZE, BACKGROUND_RGB)
    ImageDraw.Draw(image).rectangle(SQUARE_BOUNDS, fill=SQUARE_RGB)
    output = BytesIO()
    image.save(output, format="PNG", optimize=False, compress_level=9)
    content = output.getvalue()
    if not 1 <= len(content) <= 256 * 1024:
        raise RuntimeError("deterministic validation PNG exceeded its byte bound")
    return content


def _term_present(response: str, term: str) -> bool:
    folded = response.casefold()
    if term.isascii():
        return re.search(rf"(?<![a-z]){re.escape(term)}(?![a-z])", folded) is not None
    return term in folded


def evaluate_response(response: str) -> dict[str, object]:
    """Apply a conservative bilingual colour-and-shape pass criterion."""

    if not isinstance(response, str):
        raise TypeError("model response must be text")
    bounded = response.strip()
    if not 1 <= len(bounded) <= MAX_RESPONSE_CHARS:
        raise RuntimeError("model response is empty or exceeds the character bound")
    matched_colors = [term for term in _COLOR_TERMS if _term_present(bounded, term)]
    matched_shapes = [term for term in _SHAPE_TERMS if _term_present(bounded, term)]
    return {
        "required_concepts": ["blue", "square"],
        "matched_color_terms": matched_colors,
        "matched_shape_terms": matched_shapes,
        "color_pass": bool(matched_colors),
        "shape_pass": bool(matched_shapes),
        "pass": bool(matched_colors and matched_shapes),
    }


def run_validation(model: Path, manifest: Path) -> dict[str, object]:
    """Load the manifest-bound service and execute one actual image forward."""

    content = validation_png()
    service = ModelService.for_local_gemma(
        model,
        manifest_path=manifest,
        vram_limit_gib=16.7,
        max_input_tokens=4_096,
        max_new_tokens=64,
        max_prompt_chars=4_096,
        max_response_chars=MAX_RESPONSE_CHARS,
        startup_timeout=240.0,
        request_timeout=240.0,
    )
    try:
        result = service.generate(
            QUESTION,
            image_content=content,
            max_new_tokens=64,
            total_timeout=240.0,
            conversation_id="validation-local-image-v1",
            decode_mode="strict",
            sampling_seed=0,
        )
    finally:
        service.stop()

    response = result.text.strip()
    evaluation = evaluate_response(response)
    return {
        "schema_version": 1,
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "scope": "actual_local_gemma4_image_forward_smoke_test",
        "model_root": str(model),
        "manifest": str(manifest),
        "artifact_manifest_required": True,
        "actual_model_forward_executed": True,
        "external_calls": 0,
        "input": {
            "source": "deterministic_locally_generated_png",
            "format": "PNG",
            "width": IMAGE_SIZE[0],
            "height": IMAGE_SIZE[1],
            "byte_count": len(content),
            "sha256": sha256(content).hexdigest(),
            "question": QUESTION,
        },
        "inference": {
            "decode_mode": "strict",
            "sampling_seed": 0,
            "max_new_tokens": 64,
            "finish_reason": result.finish_reason,
            "generation_mode": result.generation_mode,
        },
        "response": response,
        "evaluation": evaluation,
        "image_understanding_pass": evaluation["pass"],
        "claim_boundary": (
            "This result covers one fixed blue-square smoke test only; it does "
            "not establish general visual reasoning quality or VRAM conformance."
        ),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = ArgumentParser()
    parser.add_argument("--model", default=r"C:\Project\cognios\gemma4-e4b-it")
    parser.add_argument("--manifest", default="config/gemma4-e4b-it.manifest.toml")
    parser.add_argument("--output")
    args = parser.parse_args(argv)

    evidence = run_validation(
        Path(args.model).expanduser().resolve(strict=True),
        Path(args.manifest).expanduser().resolve(strict=True),
    )
    rendered = json.dumps(evidence, ensure_ascii=False, indent=2)
    print(rendered)
    if args.output:
        output = Path(args.output).expanduser().resolve(strict=False)
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_name(f".{output.name}.tmp")
        temporary.write_text(rendered + "\n", encoding="utf-8", newline="\n")
        temporary.replace(output)
    return 0 if evidence["image_understanding_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
