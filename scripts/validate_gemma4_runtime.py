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
    load_local_gemma,
)
from cogni_agent.core_pipeline import CoreTurnRequest  # noqa: E402
from cogni_agent.model_service import LocalGemmaCorePipelineFactory  # noqa: E402
from cogni_agent.conditioning import build_latent_logits_processor  # noqa: E402
from cogni_core.search import BoundedPUCTSearchV2  # noqa: E402
from cogni_os.artifacts import verify_artifact_manifest  # noqa: E402
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
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    load_seconds = perf_counter() - started
    hidden_size = _hidden_size(model)
    events.phase("building_runtime", 65)
    pipeline = LocalGemmaCorePipelineFactory()(model)
    runtime = pipeline.runtime
    if not isinstance(runtime.search_engine, BoundedPUCTSearchV2):
        raise RuntimeError("product runtime did not install certified CTS V2")
    tokens = tokenizer(args.prompt, return_tensors="pt")
    input_ids = tokens.pop("input_ids").to("cuda")
    backbone_kwargs = {key: value.to("cuda") for key, value in tokens.items()}

    events.phase("running_inference", 75)
    started = perf_counter()
    turn = pipeline.run(
        CoreTurnRequest(
            inputs=input_ids,
            swarm_session_id="runtime-validation",
            cognitive_state=torch.tensor(
                [[0.5, 0.0, 0.5, 0.5, 0.5]],
                device=input_ids.device,
                dtype=torch.float32,
            ),
            estimated_workspace_bytes=args.workspace_mib * 1024**2,
            backbone_kwargs=backbone_kwargs,
        )
    )
    result = turn.inference
    processor = build_latent_logits_processor(
        model,
        result.search.best_state.detach(),
        max_abs_bias=0.05,
    )
    bridge_bias = processor.bias
    bridge_bias_max = float(bridge_bias.abs().max().detach().cpu())
    bridge_bias_nonzero = bool((bridge_bias != 0).any().detach().cpu())
    if not bridge_bias_nonzero or not 0.0 < bridge_bias_max <= 0.05:
        raise RuntimeError("CTS-DEQ bridge did not produce a bounded causal bias")

    events.phase("validating_decode_bridge", 88)
    with torch.inference_mode():
        conditioned = model.generate(
            input_ids=input_ids,
            attention_mask=backbone_kwargs.get("attention_mask"),
            max_new_tokens=1,
            do_sample=False,
            use_cache=False,
            num_beams=1,
            num_return_sequences=1,
            logits_processor=[processor],
        )
    generated = conditioned[:, input_ids.shape[1] :]
    if tuple(generated.shape) != (1, 1):
        raise RuntimeError("conditioned Gemma decode did not produce exactly one token")
    torch.cuda.synchronize()
    inference_seconds = perf_counter() - started
    peak_gib = torch.cuda.max_memory_allocated() / 1024**3
    finite = bool(
        torch.isfinite(result.backbone_state).all()
        and torch.isfinite(result.search.best_state).all()
    )
    telemetry = result.search.telemetry
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
    if (
        telemetry.safe_for_decode is not True
        or telemetry.unsafe_silent_fallbacks != 0
        or telemetry.linear_solve_fallbacks != 0
        or telemetry.solver_calls <= 0
        or telemetry.solver_successes <= 0
        or telemetry.failed_edges != telemetry.solver_failures
        or telemetry.q_zero_backups != telemetry.failed_edges
    ):
        raise RuntimeError(
            "certified CTS V2 solver telemetry is not decode-safe: "
            f"safe={telemetry.safe_for_decode}, "
            f"silent={telemetry.unsafe_silent_fallbacks}, "
            f"linear={telemetry.linear_solve_fallbacks}, "
            f"calls={telemetry.solver_calls}, "
            f"successes={telemetry.solver_successes}, "
            f"failures={telemetry.solver_failures}, "
            f"depth={telemetry.max_depth_reached}"
        )

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
        "transition_converged": telemetry.safe_for_decode,
        "transition_residual": telemetry.solver_residual_max,
        "transition_used_fallback": telemetry.linear_solve_fallbacks != 0,
        "cts_protocol_version": "SearchRequestV2",
        "safe_for_decode": telemetry.safe_for_decode,
        "unsafe_silent_fallbacks": telemetry.unsafe_silent_fallbacks,
        "linear_solve_fallbacks": telemetry.linear_solve_fallbacks,
        "solver_rank": telemetry.solver_rank,
        "solver_history_peak": telemetry.solver_history_peak,
        "solver_failures": telemetry.solver_failures,
        "failed_edges": telemetry.failed_edges,
        "q_zero_backups": telemetry.q_zero_backups,
        "mac_budget": telemetry.mac_budget,
        "mac_reserved": telemetry.mac_reserved,
        "act_applied": telemetry.act_applied,
        "trace_digest": telemetry.trace_digest,
        "causal_bridge_answer_bearing": True,
        "causal_bridge_bias_nonzero": bridge_bias_nonzero,
        "causal_bridge_bias_max": bridge_bias_max,
        "conditioned_generated_tokens": int(generated.numel()),
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
    print(f"cts_protocol_version={metrics['cts_protocol_version']}")
    print(f"safe_for_decode={metrics['safe_for_decode']}")
    print(f"unsafe_silent_fallbacks={metrics['unsafe_silent_fallbacks']}")
    print(f"linear_solve_fallbacks={metrics['linear_solve_fallbacks']}")
    print(f"solver_rank={metrics['solver_rank']}")
    print(f"solver_history_peak={metrics['solver_history_peak']}")
    print(f"solver_failures={metrics['solver_failures']}")
    print(f"failed_edges={metrics['failed_edges']}")
    print(f"q_zero_backups={metrics['q_zero_backups']}")
    print(f"mac_budget={metrics['mac_budget']}")
    print(f"mac_reserved={metrics['mac_reserved']}")
    print(f"act_applied={metrics['act_applied']}")
    print(f"trace_digest={metrics['trace_digest']}")
    print(f"causal_bridge_answer_bearing={metrics['causal_bridge_answer_bearing']}")
    print(f"causal_bridge_bias_nonzero={metrics['causal_bridge_bias_nonzero']}")
    print(f"causal_bridge_bias_max={metrics['causal_bridge_bias_max']:.8f}")
    print(f"conditioned_generated_tokens={metrics['conditioned_generated_tokens']}")
    print(f"peak_vram_gib={metrics['peak_vram_gib']:.4f}")
    print(f"finite={metrics['finite']}")
    events.result(metrics)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
