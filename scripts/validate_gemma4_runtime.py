"""End-to-end offline Gemma 4 + bounded CTS runtime validation."""

from __future__ import annotations

import argparse
from builtins import BaseExceptionGroup
from contextlib import contextmanager
import math
from numbers import Real
import os
from pathlib import Path
import sys
from time import perf_counter
from typing import Any, Iterator

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
from scripts.gpu5_boundary_guard import (  # noqa: E402
    GPU5BoundaryError,
    require_project_gpu_index,
    validate_guarded_gpu5_identity,
)


def _hidden_size(model: object) -> int:
    config = getattr(model, "config", None)
    text_config = getattr(config, "text_config", None)
    value = getattr(text_config, "hidden_size", None)
    if value is None:
        value = getattr(config, "hidden_size", None)
    if not isinstance(value, int) or value <= 0:
        raise RuntimeError("could not determine local Gemma hidden size")
    return value


def _bounded_integer(name: str, minimum: int, maximum: int):
    def parse(value: str) -> int:
        try:
            parsed = int(value)
        except ValueError as error:
            raise argparse.ArgumentTypeError(f"{name} must be an integer") from error
        if not minimum <= parsed <= maximum:
            raise argparse.ArgumentTypeError(
                f"{name} must be in [{minimum}, {maximum}]"
            )
        return parsed

    return parse


def _bounded_float(name: str, minimum: float, maximum: float):
    def parse(value: str) -> float:
        try:
            parsed = float(value)
        except ValueError as error:
            raise argparse.ArgumentTypeError(f"{name} must be numeric") from error
        if not math.isfinite(parsed) or not minimum <= parsed <= maximum:
            raise argparse.ArgumentTypeError(
                f"{name} must be finite and in [{minimum}, {maximum}]"
            )
        return parsed

    return parse


def _bounded_prompt(value: str) -> str:
    if (
        not value.strip()
        or len(value) > 512
        or any(ord(character) < 32 or ord(character) == 127 for character in value)
    ):
        raise argparse.ArgumentTypeError(
            "prompt must be nonempty, at most 512 characters, and contain no controls"
        )
    return value


def _physical_gpu_index(value: str) -> int:
    try:
        parsed = int(value)
        return require_project_gpu_index(parsed)
    except (ValueError, GPU5BoundaryError) as error:
        raise argparse.ArgumentTypeError(
            "physical GPU index must be exactly 5"
        ) from error


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument(
        "--prompt",
        type=_bounded_prompt,
        default="Cogni-OS integrated depth validation",
    )
    parser.add_argument(
        "--workspace-mib",
        type=_bounded_integer("workspace-mib", 1, 4_096),
        default=512,
    )
    parser.add_argument(
        "--vram-limit-gib",
        type=_bounded_float("vram-limit-gib", 1.0, 16.7),
        default=16.7,
    )
    parser.add_argument("--physical-gpu-index", type=_physical_gpu_index, required=True)
    parser.add_argument(
        "--gpu-query-context",
        choices=("native-host", "gpu5-container"),
        required=True,
    )
    parser.add_argument(
        "--event-stream",
        action="store_true",
        help="add versioned sentinel JSONL events while retaining legacy output",
    )
    return parser


@contextmanager
def _guarded_validation_scope(
    args: argparse.Namespace,
    *,
    torch_module: Any = torch,
) -> Iterator[dict[str, Any]]:
    """Always revalidate the model scope and exact GPU5 identity after the body."""

    identity_before = validate_guarded_gpu5_identity(
        physical_gpu_index=args.physical_gpu_index,
        gpu_query_context=args.gpu_query_context,
        torch_module=torch_module,
    )
    verified_before = verify_artifact_manifest(args.model, args.manifest)
    state: dict[str, Any] = {
        "gpu_identity_before": identity_before,
        "verified_before": verified_before,
    }
    primary_error: BaseException | None = None
    try:
        yield state
    except BaseException as error:
        primary_error = error

    postcheck_errors: list[BaseException] = []
    try:
        verified_after = verify_artifact_manifest(args.model, args.manifest)
        if verified_after != verified_before:
            raise RuntimeError("model manifest scope changed during runtime validation")
        state["verified_after"] = verified_after
    except BaseException as error:
        postcheck_errors.append(error)
    try:
        identity_after = validate_guarded_gpu5_identity(
            physical_gpu_index=args.physical_gpu_index,
            gpu_query_context=args.gpu_query_context,
            torch_module=torch_module,
        )
        if identity_after != identity_before:
            raise GPU5BoundaryError(
                "guarded GPU5 identity changed during runtime validation"
            )
        state["gpu_identity_after"] = identity_after
    except BaseException as error:
        postcheck_errors.append(error)

    failures = ([] if primary_error is None else [primary_error]) + postcheck_errors
    if len(failures) == 1:
        raise failures[0]
    if failures:
        raise BaseExceptionGroup(
            "runtime validation and postcheck failures",
            failures,
        )


def _validate_runtime_postconditions(
    *,
    telemetry: Any,
    finite: bool,
    peak_allocated_gib: float,
    peak_reserved_gib: float,
    vram_limit_gib: float,
    requested_depth: int,
) -> tuple[float, bool]:
    def bounded_integer(name: str, *, minimum: int = 0) -> int:
        value = getattr(telemetry, name)
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise RuntimeError(f"CTS telemetry {name} must be an integer >= {minimum}")
        return value

    residual_value = telemetry.solver_residual_max
    if isinstance(residual_value, bool) or not isinstance(residual_value, Real):
        raise RuntimeError("CTS residual telemetry must be a real number")
    residual = float(residual_value)
    if not math.isfinite(residual) or not 0.0 <= residual <= 5.0e-3:
        raise RuntimeError(
            f"CTS residual postcondition failed: residual={residual!r}, limit=0.005"
        )
    if finite is not True:
        raise RuntimeError("integrated runtime produced a non-finite latent")
    if isinstance(peak_allocated_gib, bool) or not isinstance(peak_allocated_gib, Real):
        raise RuntimeError("allocated VRAM peak must be real numeric data")
    if isinstance(peak_reserved_gib, bool) or not isinstance(peak_reserved_gib, Real):
        raise RuntimeError("reserved VRAM peak must be real numeric data")
    allocated_peak = float(peak_allocated_gib)
    reserved_peak = float(peak_reserved_gib)
    if not (
        math.isfinite(allocated_peak)
        and math.isfinite(reserved_peak)
        and 0.0 <= allocated_peak <= reserved_peak <= vram_limit_gib
    ):
        raise RuntimeError(
            "VRAM postcondition failed: "
            f"allocated_peak={peak_allocated_gib!r} GiB, "
            f"reserved_peak={peak_reserved_gib!r} GiB, "
            f"limit={vram_limit_gib:.4f} GiB"
        )
    max_depth_reached = bounded_integer("max_depth_reached", minimum=1)
    unsafe_silent_fallbacks = bounded_integer("unsafe_silent_fallbacks")
    linear_solve_fallbacks = bounded_integer("linear_solve_fallbacks")
    solver_calls = bounded_integer("solver_calls", minimum=1)
    solver_successes = bounded_integer("solver_successes", minimum=1)
    solver_failures = bounded_integer("solver_failures")
    failed_edges = bounded_integer("failed_edges")
    q_zero_backups = bounded_integer("q_zero_backups")
    nodes_used = bounded_integer("nodes_used", minimum=1)
    node_capacity = bounded_integer("node_capacity", minimum=1)
    allocated_bytes = bounded_integer("allocated_bytes")
    if max_depth_reached != requested_depth:
        raise RuntimeError(
            "depth postcondition failed: "
            f"reached={max_depth_reached}, requested={requested_depth}"
        )
    if (
        telemetry.safe_for_decode is not True
        or unsafe_silent_fallbacks != 0
        or linear_solve_fallbacks != 0
        or solver_failures != 0
        or failed_edges != 0
        or q_zero_backups != 0
        or solver_calls != solver_successes + solver_failures
        or solver_successes != solver_calls
        or failed_edges != solver_failures
        or q_zero_backups != failed_edges
        or nodes_used > node_capacity
    ):
        raise RuntimeError(
            "certified CTS V2 solver telemetry is not decode-safe: "
            f"safe={telemetry.safe_for_decode}, "
            f"silent={unsafe_silent_fallbacks}, "
            f"linear={linear_solve_fallbacks}, "
            f"calls={solver_calls}, "
            f"successes={solver_successes}, "
            f"failures={solver_failures}, "
            f"nodes={nodes_used}/{node_capacity}, "
            f"allocated_bytes={allocated_bytes}, "
            f"depth={max_depth_reached}"
        )
    transition_converged = bool(
        residual <= 5.0e-3
        and solver_failures == 0
        and failed_edges == 0
        and q_zero_backups == 0
        and solver_successes == solver_calls
    )
    return residual, transition_converged


def _execute_runtime(
    args: argparse.Namespace,
    events: EventEmitter,
    validation: dict[str, Any],
) -> dict[str, Any]:
    events.phase("verifying", 5)
    verified = validation["verified_before"]
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
    peak_allocated_gib = torch.cuda.max_memory_allocated() / 1024**3
    peak_reserved_gib = torch.cuda.max_memory_reserved() / 1024**3
    peak_gib = max(peak_allocated_gib, peak_reserved_gib)
    finite = bool(
        torch.isfinite(result.backbone_state).all()
        and torch.isfinite(result.search.best_state).all()
    )
    telemetry = result.search.telemetry
    events.phase("postcheck", 95)
    transition_residual, transition_converged = _validate_runtime_postconditions(
        telemetry=telemetry,
        finite=finite,
        peak_allocated_gib=peak_allocated_gib,
        peak_reserved_gib=peak_reserved_gib,
        vram_limit_gib=args.vram_limit_gib,
        requested_depth=runtime.search_engine.config.max_depth,
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
        "transition_converged": transition_converged,
        "transition_residual": transition_residual,
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
        "peak_allocated_vram_gib": peak_allocated_gib,
        "peak_reserved_vram_gib": peak_reserved_gib,
        "peak_vram_gib": peak_gib,
        "vram_limit_gib": args.vram_limit_gib,
        "finite": finite,
        "device": torch.cuda.get_device_properties(0).name,
    }
    return metrics


def _print_metrics(metrics: dict[str, Any]) -> None:
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
    print(f"peak_allocated_vram_gib={metrics['peak_allocated_vram_gib']:.4f}")
    print(f"peak_reserved_vram_gib={metrics['peak_reserved_vram_gib']:.4f}")
    print(f"peak_vram_gib={metrics['peak_vram_gib']:.4f}")
    print(f"finite={metrics['finite']}")


def main() -> int:
    args = build_parser().parse_args()
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    events = EventEmitter(args.event_stream)
    with _guarded_validation_scope(args) as validation:
        metrics = _execute_runtime(args, events, validation)
    verified_after = validation["verified_after"]
    identity_after = validation["gpu_identity_after"]
    terminal_metrics = dict(metrics)
    metrics["verified_files_after"] = len(verified_after.files)
    metrics["guarded_gpu5_identity"] = identity_after.as_payload()
    _print_metrics(metrics)
    print(f"verified_files_after={metrics['verified_files_after']}")
    print(
        "guarded_gpu5_identity="
        f"{identity_after.uuid}:cuda:{identity_after.logical_device_index}"
    )
    events.result(terminal_metrics)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
