"""Run the bounded DEQ gate against a verified local Gemma 4 checkpoint.

This script is intentionally separate from the plain-backbone smoke test: a
successful ordinary forward must never be mistaken for a converged equilibrium
solve.  It exits non-zero when the artifact digest, solver convergence, output
finiteness, or VRAM postcondition fails.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
from time import perf_counter

import torch

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from cogni_core.backbone import inject_gemma_deq_layer, load_local_gemma  # noqa: E402
from cogni_core.deq import DEQConfig  # noqa: E402
from cogni_os.artifacts import verify_artifact_manifest  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--prompt", default="Cogni-OS DEQ offline validation")
    parser.add_argument("--layer-index", type=int, default=-1)
    parser.add_argument("--tolerance", type=float, default=5.0e-3)
    parser.add_argument("--max-iter", type=int, default=12)
    parser.add_argument("--history", type=int, default=6)
    parser.add_argument("--fallback-steps", type=int, default=16)
    parser.add_argument("--fallback-damping", type=float, default=0.35)
    parser.add_argument("--contractive-delta-scale", type=float, default=0.05)
    parser.add_argument(
        "--certified-delta-lipschitz-bound",
        type=float,
        default=None,
        help="offline-certified global upper bound for the decoder delta branch",
    )
    parser.add_argument(
        "--allow-uncertified-experimental",
        action="store_true",
        help="run a convergence smoke test without claiming contractivity certification",
    )
    parser.add_argument("--vram-limit-gib", type=float, default=16.7)
    args = parser.parse_args()
    if args.certified_delta_lipschitz_bound is None:
        if not args.allow_uncertified_experimental:
            parser.error(
                "production validation requires --certified-delta-lipschitz-bound; "
                "use --allow-uncertified-experimental only for a labelled smoke test"
            )
    elif args.allow_uncertified_experimental:
        parser.error(
            "do not combine a contractivity certificate with the experimental override"
        )

    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    verified = verify_artifact_manifest(args.model, args.manifest)
    if not torch.cuda.is_available():
        raise RuntimeError("Gemma 4 DEQ validation requires a CUDA device")

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    started = perf_counter()
    model, tokenizer = load_local_gemma(args.model)
    load_seconds = perf_counter() - started
    adapter = inject_gemma_deq_layer(
        model,
        layer_index=args.layer_index,
        config=DEQConfig(
            tolerance=args.tolerance,
            max_iter=args.max_iter,
            history=args.history,
            fallback_steps=args.fallback_steps,
            fallback_damping=args.fallback_damping,
            fail_on_noncontractive=not args.allow_uncertified_experimental,
        ),
        contractive_delta_scale=args.contractive_delta_scale,
        certified_delta_lipschitz_bound=args.certified_delta_lipschitz_bound,
    )
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
    info = adapter.last_info
    if info is None:
        raise RuntimeError("DEQ adapter did not record solver diagnostics")
    if not info.converged:
        raise RuntimeError(
            f"DEQ convergence postcondition failed: residual={info.residual:.6e}"
        )
    if peak_gib > args.vram_limit_gib:
        raise RuntimeError(
            f"VRAM postcondition failed: peak={peak_gib:.4f} GiB, "
            f"limit={args.vram_limit_gib:.4f} GiB"
        )
    finite = bool(torch.isfinite(output.logits).all())
    if not finite:
        raise RuntimeError("Gemma 4 DEQ output contains a non-finite value")

    print(f"verified_files={len(verified.files)}")
    print(f"model_class={type(model).__name__}")
    print(f"deq_layer_index={args.layer_index}")
    print(f"contractive_delta_scale={args.contractive_delta_scale:.6f}")
    certified = args.certified_delta_lipschitz_bound is not None
    print(f"contractivity_certified={certified}")
    print(f"effective_lipschitz_upper_bound={info.spectral_norm}")
    print(f"load_seconds={load_seconds:.3f}")
    print(f"forward_seconds={forward_seconds:.3f}")
    print(f"solver_converged={info.converged}")
    print(f"solver_iterations={info.iterations}")
    print(f"solver_residual={info.residual:.10f}")
    print(f"solver_used_fallback={info.used_fallback}")
    print(f"logits_shape={tuple(output.logits.shape)}")
    print(f"peak_vram_gib={peak_gib:.4f}")
    print(f"finite={finite}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
