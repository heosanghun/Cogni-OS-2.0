from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
from time import perf_counter

import torch

# Direct ``python scripts/<name>.py`` execution puts only ``scripts`` on the
# import path.  Pin the repository root first so validation always exercises
# this checkout, independent of the caller's working directory or installation.
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from cogni_core.backbone import find_decoder_layers, load_local_gemma  # noqa: E402
from cogni_os.artifacts import verify_artifact_manifest  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--prompt", default="Cogni-OS offline validation")
    parser.add_argument("--vram-limit-gib", type=float, default=16.7)
    args = parser.parse_args()

    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    verified = verify_artifact_manifest(args.model, args.manifest)
    if not torch.cuda.is_available():
        raise RuntimeError("Gemma 4 end-to-end validation requires a CUDA device")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    started = perf_counter()
    model, tokenizer = load_local_gemma(args.model)
    load_seconds = perf_counter() - started
    inputs = {
        key: value.to("cuda")
        for key, value in tokenizer(args.prompt, return_tensors="pt").items()
    }
    started = perf_counter()
    with torch.inference_mode():
        output = model(**inputs, use_cache=False)
    torch.cuda.synchronize()
    forward_seconds = perf_counter() - started
    peak_gib = torch.cuda.max_memory_allocated() / 1024**3
    if peak_gib > args.vram_limit_gib:
        raise RuntimeError(
            f"VRAM postcondition failed: peak={peak_gib:.4f} GiB, limit={args.vram_limit_gib:.4f} GiB"
        )
    finite = bool(torch.isfinite(output.logits).all())
    if not finite:
        raise RuntimeError("Gemma 4 output contains a non-finite value")
    print(f"verified_files={len(verified.files)}")
    print(f"model_class={type(model).__name__}")
    print(f"decoder_layers={len(find_decoder_layers(model))}")
    print(f"load_seconds={load_seconds:.3f}")
    print(f"forward_seconds={forward_seconds:.3f}")
    print(f"logits_shape={tuple(output.logits.shape)}")
    print(f"peak_vram_gib={peak_gib:.4f}")
    print(f"finite={finite}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
