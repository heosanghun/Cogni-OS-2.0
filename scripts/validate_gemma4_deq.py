"""Run the bounded DEQ gate against a verified local Gemma 4 checkpoint.

This script is intentionally separate from the plain-backbone smoke test: a
successful ordinary forward must never be mistaken for a converged equilibrium
solve.  It exits non-zero when the artifact digest, solver convergence, output
finiteness, or VRAM postcondition fails.
"""

from __future__ import annotations

import argparse
from builtins import BaseExceptionGroup
from contextlib import contextmanager
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import sys
from time import perf_counter
from typing import Any, Iterator

import torch

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from cogni_core.backbone import inject_gemma_deq_layer, load_local_gemma  # noqa: E402
from cogni_core.deq import DEQConfig  # noqa: E402
from cogni_os.artifacts import verify_artifact_manifest  # noqa: E402
from scripts.gpu5_boundary_guard import (  # noqa: E402
    GPU5BoundaryError,
    require_project_gpu_index,
    validate_guarded_gpu5_identity,
)


MAX_PROVENANCE_BYTES = 64 * 1024
MAX_RAW_DELTA_LIPSCHITZ_BOUND = 1.0e6
MAX_EFFECTIVE_LIPSCHITZ_BOUND = 0.95
RECOGNIZED_CONTRACTIVITY_METHODS = frozenset(
    {
        "formal-global-lipschitz-bound",
        "interval-bound-propagation",
        "third-party-certified-global-bound",
    }
)


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


def _positive_bounded_float(name: str, maximum: float):
    def parse(value: str) -> float:
        try:
            parsed = float(value)
        except ValueError as error:
            raise argparse.ArgumentTypeError(f"{name} must be numeric") from error
        if not math.isfinite(parsed) or not 0.0 < parsed <= maximum:
            raise argparse.ArgumentTypeError(
                f"{name} must be finite and in (0, {maximum}]"
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
        "--prompt", type=_bounded_prompt, default="Cogni-OS DEQ offline validation"
    )
    parser.add_argument(
        "--layer-index", type=_bounded_integer("layer-index", -1, 128), default=-1
    )
    parser.add_argument(
        "--tolerance",
        type=_bounded_float("tolerance", 1.0e-9, 5.0e-3),
        default=5.0e-3,
    )
    parser.add_argument(
        "--max-iter", type=_bounded_integer("max-iter", 1, 128), default=12
    )
    parser.add_argument("--history", type=_bounded_integer("history", 1, 64), default=6)
    parser.add_argument(
        "--fallback-steps",
        type=_bounded_integer("fallback-steps", 1, 256),
        default=16,
    )
    parser.add_argument(
        "--fallback-damping",
        type=_positive_bounded_float("fallback-damping", 1.0),
        default=0.35,
    )
    parser.add_argument(
        "--contractive-delta-scale",
        type=_positive_bounded_float("contractive-delta-scale", 1.0),
        default=0.05,
    )
    parser.add_argument(
        "--certified-delta-lipschitz-bound",
        type=_bounded_float(
            "certified-delta-lipschitz-bound",
            0.0,
            MAX_RAW_DELTA_LIPSCHITZ_BOUND,
        ),
        default=None,
        help=(
            "optional raw decoder-delta bound to cross-check; this CLI number "
            "never constitutes release certification by itself"
        ),
    )
    parser.add_argument(
        "--contractivity-provenance",
        default=None,
        help=(
            "manifest-listed JSON retained for experimental traceability only; "
            "without a pinned trust anchor it never qualifies as release evidence"
        ),
    )
    parser.add_argument(
        "--allow-uncertified-experimental",
        action="store_true",
        help=(
            "non-release convergence smoke test only; its output is never eligible "
            "for production evidence"
        ),
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
    return parser


def _finite_provenance_number(data: dict[str, Any], key: str) -> float:
    value = data.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RuntimeError(f"contractivity provenance requires numeric {key}")
    parsed = float(value)
    if not math.isfinite(parsed) or parsed < 0.0:
        raise RuntimeError(
            f"contractivity provenance requires finite non-negative {key}"
        )
    return parsed


def _bounded_provenance_text(data: dict[str, Any], key: str) -> str:
    value = data.get(key)
    if (
        not isinstance(value, str)
        or not 1 <= len(value) <= 256
        or any(ord(character) < 32 or ord(character) == 127 for character in value)
    ):
        raise RuntimeError(f"contractivity provenance requires bounded text {key}")
    return value


def _effective_lipschitz_bound(scale: float, raw_bound: float | None) -> float:
    if not math.isfinite(scale) or not 0.0 < scale <= 1.0:
        raise RuntimeError("contractive delta scale must be finite and in (0, 1]")
    if raw_bound is None:
        return float("inf")
    if (
        not math.isfinite(raw_bound)
        or raw_bound < 0.0
        or raw_bound > MAX_RAW_DELTA_LIPSCHITZ_BOUND
    ):
        raise RuntimeError("raw delta Lipschitz bound is invalid")
    effective = scale * raw_bound
    if not math.isfinite(effective) or effective > MAX_EFFECTIVE_LIPSCHITZ_BOUND:
        raise RuntimeError(
            "effective Lipschitz upper bound exceeds the safety margin: "
            f"scale*raw_bound={effective!r} > {MAX_EFFECTIVE_LIPSCHITZ_BOUND}"
        )
    return effective


def _read_manifest_bound_provenance(
    provenance_name: str,
    verified: Any,
) -> tuple[dict[str, Any], str]:
    relative = Path(provenance_name)
    if relative.is_absolute() or not relative.parts or ".." in relative.parts:
        raise RuntimeError("contractivity provenance must be a safe relative path")
    relative_name = relative.as_posix()
    declared_digests = dict(verified.digests)
    expected_digest = declared_digests.get(relative_name)
    if expected_digest is None:
        raise RuntimeError(
            "contractivity provenance must be covered by the verified model manifest"
        )
    root = Path(verified.root).expanduser().resolve(strict=True)
    candidate = root / relative
    if candidate.is_symlink():
        raise RuntimeError("contractivity provenance cannot be a symlink")
    resolved = candidate.resolve(strict=True)
    if not resolved.is_relative_to(root) or not resolved.is_file():
        raise RuntimeError("contractivity provenance escaped the verified model root")
    encoded = resolved.read_bytes()
    if not encoded or len(encoded) > MAX_PROVENANCE_BYTES:
        raise RuntimeError(
            "contractivity provenance must be nonempty and at most 64 KiB"
        )
    actual_digest = sha256(encoded).hexdigest()
    if actual_digest != expected_digest:
        raise RuntimeError("contractivity provenance digest changed after verification")
    try:
        data = json.loads(encoded.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeError(
            "contractivity provenance must be strict UTF-8 JSON"
        ) from error
    if not isinstance(data, dict):
        raise RuntimeError("contractivity provenance root must be an object")
    return data, actual_digest


def _resolve_contractivity_evidence(
    args: argparse.Namespace,
    verified: Any,
) -> dict[str, Any]:
    """Classify DEQ evidence without trusting user-supplied numeric claims."""

    experimental = bool(getattr(args, "allow_uncertified_experimental", False))
    provenance_name = getattr(args, "contractivity_provenance", None)
    cli_raw_bound = getattr(args, "certified_delta_lipschitz_bound", None)
    scale = float(args.contractive_delta_scale)

    if not experimental:
        raise RuntimeError(
            "release DEQ certification is unsupported: this repository has no "
            "pinned trust key, signature verifier, or reproducible independent "
            "global-bound verifier. All current evidence must remain non-release"
        )

    if provenance_name is None:
        effective = _effective_lipschitz_bound(scale, cli_raw_bound)
        return {
            "certified": False,
            "raw_bound": cli_raw_bound,
            "effective_bound": effective,
            "evidence_class": "experimental-non-release",
            "release_evidence_eligible": False,
            "provenance_sha256": None,
            "provenance_method": None,
        }

    data, provenance_digest = _read_manifest_bound_provenance(
        str(provenance_name), verified
    )
    if data.get("schema") != "cogni.deq.contractivity.provenance.v1":
        raise RuntimeError("unsupported contractivity provenance schema")
    if data.get("independent_from_runtime_cli") is not True:
        raise RuntimeError(
            "contractivity provenance must be independent from runtime CLI values"
        )
    method = _bounded_provenance_text(data, "method")
    if method not in RECOGNIZED_CONTRACTIVITY_METHODS:
        raise RuntimeError("contractivity provenance method is not recognized")
    _bounded_provenance_text(data, "verifier")
    _bounded_provenance_text(data, "tool")
    analysis_digest = _bounded_provenance_text(data, "analysis_artifact_sha256")
    if len(analysis_digest) != 64 or any(
        character not in "0123456789abcdef" for character in analysis_digest.lower()
    ):
        raise RuntimeError(
            "contractivity provenance analysis_artifact_sha256 is invalid"
        )
    identity = getattr(verified, "identity", None)
    if identity is None or data.get("model_revision") != identity.revision:
        raise RuntimeError(
            "contractivity provenance is not bound to the verified model revision"
        )
    if data.get("layer_index") != args.layer_index:
        raise RuntimeError("contractivity provenance layer index does not match")
    provenance_scale = _finite_provenance_number(data, "contractive_delta_scale")
    if not math.isclose(provenance_scale, scale, rel_tol=0.0, abs_tol=1.0e-12):
        raise RuntimeError("contractivity provenance scale does not match")
    raw_bound = _finite_provenance_number(data, "delta_lipschitz_upper_bound")
    if cli_raw_bound is not None and not math.isclose(
        float(cli_raw_bound), raw_bound, rel_tol=0.0, abs_tol=1.0e-12
    ):
        raise RuntimeError(
            "CLI delta Lipschitz bound does not match independent provenance"
        )
    effective = _effective_lipschitz_bound(scale, raw_bound)
    declared_effective = _finite_provenance_number(
        data, "effective_lipschitz_upper_bound"
    )
    if not math.isclose(
        declared_effective, effective, rel_tol=1.0e-12, abs_tol=1.0e-12
    ):
        raise RuntimeError(
            "contractivity provenance effective bound is not scale*raw_bound"
        )
    return {
        "certified": False,
        "raw_bound": raw_bound,
        "effective_bound": effective,
        "evidence_class": "experimental-provenance-non-release",
        "release_evidence_eligible": False,
        "provenance_sha256": provenance_digest,
        "provenance_method": method,
    }


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
            raise RuntimeError("model manifest scope changed during DEQ validation")
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
                "guarded GPU5 identity changed during DEQ validation"
            )
        state["gpu_identity_after"] = identity_after
    except BaseException as error:
        postcheck_errors.append(error)

    failures = ([] if primary_error is None else [primary_error]) + postcheck_errors
    if len(failures) == 1:
        raise failures[0]
    if failures:
        raise BaseExceptionGroup(
            "DEQ validation and postcheck failures",
            failures,
        )


def _validate_deq_postconditions(
    *,
    info: Any,
    finite: bool,
    peak_gib: float,
    vram_limit_gib: float,
    tolerance: float,
    certified: bool,
) -> tuple[float, float | None]:
    residual = float(info.residual)
    residual_limit = min(float(tolerance), 5.0e-3)
    if info.converged is not True:
        raise RuntimeError(
            f"DEQ convergence postcondition failed: residual={residual!r}"
        )
    if not math.isfinite(residual) or not 0.0 <= residual <= residual_limit:
        raise RuntimeError(
            f"DEQ residual postcondition failed: residual={residual!r}, "
            f"limit={residual_limit:.10f}"
        )
    if info.used_fallback is not False:
        raise RuntimeError("DEQ fallback postcondition failed: fallback was used")
    if not math.isfinite(peak_gib) or peak_gib < 0.0 or peak_gib > vram_limit_gib:
        raise RuntimeError(
            f"VRAM postcondition failed: peak={peak_gib!r} GiB, "
            f"limit={vram_limit_gib:.4f} GiB"
        )
    if finite is not True:
        raise RuntimeError("Gemma 4 DEQ output contains a non-finite value")
    spectral_norm = None if info.spectral_norm is None else float(info.spectral_norm)
    if certified and (
        spectral_norm is None
        or not math.isfinite(spectral_norm)
        or not 0.0 <= spectral_norm <= 0.95
    ):
        raise RuntimeError(
            "certified DEQ spectral-norm postcondition failed: "
            f"spectral_norm={spectral_norm!r}"
        )
    return residual, spectral_norm


def _execute_deq(
    args: argparse.Namespace,
    validation: dict[str, Any],
) -> dict[str, Any]:
    verified = validation["verified_before"]
    evidence = _resolve_contractivity_evidence(args, verified)
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
            fail_on_noncontractive=evidence["certified"],
        ),
        contractive_delta_scale=args.contractive_delta_scale,
        certified_delta_lipschitz_bound=evidence["raw_bound"],
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
    finite = bool(torch.isfinite(output.logits).all())
    certified = bool(evidence["certified"])
    residual, spectral_norm = _validate_deq_postconditions(
        info=info,
        finite=finite,
        peak_gib=peak_gib,
        vram_limit_gib=args.vram_limit_gib,
        tolerance=args.tolerance,
        certified=certified,
    )
    if certified and not math.isclose(
        float(spectral_norm),
        float(evidence["effective_bound"]),
        rel_tol=1.0e-12,
        abs_tol=1.0e-12,
    ):
        raise RuntimeError(
            "DEQ adapter effective bound does not match trusted provenance"
        )
    return {
        "verified_files": len(verified.files),
        "model_class": type(model).__name__,
        "deq_layer_index": args.layer_index,
        "contractive_delta_scale": args.contractive_delta_scale,
        "contractivity_certified": certified,
        "release_evidence_eligible": evidence["release_evidence_eligible"],
        "evidence_class": evidence["evidence_class"],
        "contractivity_provenance_sha256": evidence["provenance_sha256"],
        "contractivity_provenance_method": evidence["provenance_method"],
        "raw_delta_lipschitz_upper_bound": evidence["raw_bound"],
        "effective_lipschitz_upper_bound": spectral_norm,
        "load_seconds": load_seconds,
        "forward_seconds": forward_seconds,
        "solver_converged": info.converged,
        "solver_iterations": info.iterations,
        "solver_residual": residual,
        "solver_used_fallback": info.used_fallback,
        "logits_shape": tuple(output.logits.shape),
        "peak_vram_gib": peak_gib,
        "finite": finite,
    }


def _print_metrics(metrics: dict[str, Any]) -> None:
    print(f"verified_files={metrics['verified_files']}")
    print(f"model_class={metrics['model_class']}")
    print(f"deq_layer_index={metrics['deq_layer_index']}")
    print(f"contractive_delta_scale={metrics['contractive_delta_scale']:.6f}")
    print(f"contractivity_certified={metrics['contractivity_certified']}")
    print(f"release_evidence_eligible={metrics['release_evidence_eligible']}")
    print(f"evidence_class={metrics['evidence_class']}")
    print(
        f"contractivity_provenance_sha256={metrics['contractivity_provenance_sha256']}"
    )
    print(
        f"contractivity_provenance_method={metrics['contractivity_provenance_method']}"
    )
    print(
        f"raw_delta_lipschitz_upper_bound={metrics['raw_delta_lipschitz_upper_bound']}"
    )
    print(
        f"effective_lipschitz_upper_bound={metrics['effective_lipschitz_upper_bound']}"
    )
    print(f"load_seconds={metrics['load_seconds']:.3f}")
    print(f"forward_seconds={metrics['forward_seconds']:.3f}")
    print(f"solver_converged={metrics['solver_converged']}")
    print(f"solver_iterations={metrics['solver_iterations']}")
    print(f"solver_residual={metrics['solver_residual']:.10f}")
    print(f"solver_used_fallback={metrics['solver_used_fallback']}")
    print(f"logits_shape={metrics['logits_shape']}")
    print(f"peak_vram_gib={metrics['peak_vram_gib']:.4f}")
    print(f"finite={metrics['finite']}")


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if not args.allow_uncertified_experimental:
        parser.error(
            "release DEQ certification is unsupported until a pinned trust key, "
            "signature verifier, or reproducible independent global-bound verifier "
            "is implemented. Use --allow-uncertified-experimental only for labelled "
            "non-release evidence"
        )
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    with _guarded_validation_scope(args) as validation:
        metrics = _execute_deq(args, validation)
    verified_after = validation["verified_after"]
    identity_after = validation["gpu_identity_after"]
    metrics["verified_files_after"] = len(verified_after.files)
    metrics["guarded_gpu5_identity"] = identity_after.as_payload()
    _print_metrics(metrics)
    print(f"verified_files_after={metrics['verified_files_after']}")
    print(
        "guarded_gpu5_identity="
        f"{identity_after.uuid}:cuda:{identity_after.logical_device_index}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
