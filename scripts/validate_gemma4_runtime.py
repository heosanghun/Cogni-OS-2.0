"""End-to-end offline Gemma 4 + bounded CTS runtime validation."""

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

from cogni_core.backbone import (  # noqa: E402
    LocalGemmaFeatureBackbone,
    load_local_gemma,
)
from cogni_core.search import ContractiveBroydenTransition  # noqa: E402
from cogni_os.artifacts import verify_artifact_manifest  # noqa: E402
from cogni_os.config import load_config  # noqa: E402
from cogni_os.factory import build_genesis_runtime  # noqa: E402
from cogni_demo.protocol import EventEmitter  # noqa: E402


def _hidden_size(model: object) -> int:
    config = getattr(model, "config", None)
    text_config = getattr(config, "text_config", None)
    value = getattr(text_config, "hidden_size", None)
    if value is None:
        value = getattr(config, "hidden_size", None)
    if not isinstance(value, int) or value <= 0:
        raise RuntimeError("could not determine local Gemma hidden size")
    return value


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--prompt", default="Cogni-OS integrated depth validation")
    parser.add_argument("--workspace-mib", type=int, default=512)
    parser.add_argument("--vram-limit-gib", type=float, default=16.7)
    parser.add_argument(
        "--event-stream",
        action="store_true",
        help="add versioned sentinel JSONL events while retaining legacy output",
    )
    args = parser.parse_args()
    if args.workspace_mib <= 0:
        parser.error("--workspace-mib must be positive")

    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    events = EventEmitter(args.event_stream)
    events.phase("verifying", 5)
    verified = verify_artifact_manifest(args.model, args.manifest)
    if not torch.cuda.is_available():
        raise RuntimeError("integrated Gemma 4 validation requires a CUDA device")

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    events.phase("loading_model", 15)
    started = perf_counter()
    model, tokenizer = load_local_gemma(args.model, vram_limit_gib=args.vram_limit_gib)
    load_seconds = perf_counter() - started
    hidden_size = _hidden_size(model)
    events.phase("building_runtime", 65)
    runtime = build_genesis_runtime(
        LocalGemmaFeatureBackbone(model),
        load_config(),
        input_dim=hidden_size,
        state_dim=64,
    )
    transition = ContractiveBroydenTransition(
        width=runtime.search_engine.config.width,
        contraction=0.4,
        spectral_margin=0.95,
        tolerance=5.0e-3,
        max_iter=12,
        history=6,
        fallback_steps=32,
    )
    tokens = tokenizer(args.prompt, return_tensors="pt")
    input_ids = tokens.pop("input_ids").to("cuda")
    backbone_kwargs = {key: value.to("cuda") for key, value in tokens.items()}
    width = runtime.search_engine.config.width

    def policy_value(state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # A deterministic dominant prior exercises one complete depth-100
        # branch while preserving fixed-width expansion at every node.
        logits = torch.full((width,), -40.0, device=state.device)
        logits[-1] = 40.0
        return logits, state.float().mean()

    events.phase("running_inference", 75)
    started = perf_counter()
    result = runtime.infer(
        input_ids,
        transition,
        policy_value,
        estimated_workspace_bytes=args.workspace_mib * 1024**2,
        backbone_kwargs=backbone_kwargs,
    )
    torch.cuda.synchronize()
    inference_seconds = perf_counter() - started
    peak_gib = torch.cuda.max_memory_allocated() / 1024**3
    finite = bool(
        torch.isfinite(result.backbone_state).all()
        and torch.isfinite(result.search.best_state).all()
    )
    telemetry = result.search.telemetry
    info = transition.last_info
    events.phase("postcheck", 95)
    if not finite:
        raise RuntimeError("integrated runtime produced a non-finite latent")
    if peak_gib > args.vram_limit_gib:
        raise RuntimeError(
            f"VRAM postcondition failed: peak={peak_gib:.4f} GiB, "
            f"limit={args.vram_limit_gib:.4f} GiB"
        )
    if telemetry.max_depth_reached != runtime.search_engine.config.max_depth:
        raise RuntimeError(
            "depth postcondition failed: "
            f"reached={telemetry.max_depth_reached}, "
            f"requested={runtime.search_engine.config.max_depth}"
        )
    if info is None or not info.converged:
        raise RuntimeError("final CTS transition did not converge")

    metrics = {
        "verified_files": len(verified.files),
        "model_class": type(model).__name__,
        "hidden_size": hidden_size,
        "load_seconds": load_seconds,
        "inference_seconds": inference_seconds,
        "requested_depth": runtime.search_engine.config.max_depth,
        "reached_depth": telemetry.max_depth_reached,
        "nodes_used": telemetry.nodes_used,
        "node_capacity": telemetry.node_capacity,
        "search_allocated_bytes": telemetry.allocated_bytes,
        "transition_converged": info.converged,
        "transition_residual": info.residual,
        "transition_used_fallback": info.used_fallback,
        "peak_vram_gib": peak_gib,
        "vram_limit_gib": args.vram_limit_gib,
        "finite": finite,
        "device": torch.cuda.get_device_properties(0).name,
    }
    print(f"verified_files={metrics['verified_files']}")
    print(f"model_class={metrics['model_class']}")
    print(f"hidden_size={metrics['hidden_size']}")
    print(f"load_seconds={metrics['load_seconds']:.3f}")
    print(f"inference_seconds={metrics['inference_seconds']:.3f}")
    print(f"requested_depth={metrics['requested_depth']}")
    print(f"reached_depth={metrics['reached_depth']}")
    print(f"nodes_used={metrics['nodes_used']}")
    print(f"node_capacity={metrics['node_capacity']}")
    print(f"search_allocated_bytes={metrics['search_allocated_bytes']}")
    print(f"transition_converged={metrics['transition_converged']}")
    print(f"transition_residual={metrics['transition_residual']:.10f}")
    print(f"transition_used_fallback={metrics['transition_used_fallback']}")
    print(f"peak_vram_gib={metrics['peak_vram_gib']:.4f}")
    print(f"finite={metrics['finite']}")
    events.result(metrics)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
