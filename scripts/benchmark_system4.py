from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from time import perf_counter_ns

import torch

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from cogni_core.swarm import SwarmConfig, TensorSwarm  # noqa: E402


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure certified System-4 warm-path latency without claims"
    )
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--input-dim", type=int, default=64)
    parser.add_argument("--state-dim", type=int, default=64)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument(
        "--stress-switch",
        action="store_true",
        help="alternate normal/crisis observations in bounded blocks",
    )
    parser.add_argument("--switch-block", type=int, default=32)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA benchmark requested but CUDA is unavailable")
    if min(args.input_dim, args.state_dim, args.batch, args.iterations) < 1:
        raise ValueError("dimensions, batch, and iterations must be positive")
    if args.warmup < 0:
        raise ValueError("warmup cannot be negative")
    if args.switch_block < 16:
        raise ValueError("switch-block must be at least 16 steps")
    device = torch.device(args.device)
    torch.manual_seed(20260712)
    swarm = (
        TensorSwarm(SwarmConfig(input_dim=args.input_dim, state_dim=args.state_dim))
        .to(device=device, dtype=torch.float32)
        .eval()
    )
    calibration = torch.randn(1024, args.input_dim, device=device) * 0.1
    swarm.monitor.fit(calibration)
    # Use calibration-relative stimuli instead of one arbitrary random sample.
    # A fixed random "normal" point can itself land above the 0.99 threshold,
    # while an enormous constant crisis point leaves the drift EMA saturated
    # long after the phase changes.  The construction below is deterministic:
    # the normal point is the fitted mean and the crisis point has an exact
    # Mahalanobis distance of 2x the calibrated enter threshold.
    normal_observation = swarm.monitor.mean.expand(args.batch, -1).clone()
    crisis_delta = torch.zeros(args.input_dim, device=device, dtype=torch.float32)
    axis_precision = swarm.monitor.cov_inv[0, 0].clamp_min(1e-8)
    crisis_delta[0] = 2.0 * swarm.monitor.enter_threshold / torch.sqrt(axis_precision)
    crisis_observation = normal_observation + crisis_delta
    observation = normal_observation
    state = None
    detector_state = None
    for _ in range(args.warmup):
        output = swarm(observation, state, pcas_state=detector_state)
        if bool(output.safe_for_advice.detach().cpu()):
            state = output.joint_state
            detector_state = output.pcas_state

    samples_ms: list[float] = []
    residuals: list[float] = []
    regimes: list[int] = []
    expected_regimes: list[int] = []
    converged = 0
    _synchronize(device)
    for index in range(args.iterations):
        expected_regime = (index // args.switch_block) % 2 if args.stress_switch else 0
        observation = crisis_observation if expected_regime else normal_observation
        started = perf_counter_ns()
        output = swarm(observation, state, pcas_state=detector_state)
        _synchronize(device)
        samples_ms.append((perf_counter_ns() - started) / 1_000_000.0)
        residuals.append(float(output.residual.max().detach().cpu()))
        regimes.append(int(output.regime.detach().cpu()))
        expected_regimes.append(expected_regime)
        if bool(output.safe_for_advice.detach().cpu()):
            converged += 1
            state = output.joint_state
            detector_state = output.pcas_state

    samples = torch.tensor(samples_ms, dtype=torch.float64)
    normal, crisis = swarm.topology_certificates
    normal_count = expected_regimes.count(0)
    crisis_count = expected_regimes.count(1)
    false_positives = sum(
        actual == 1 and expected == 0
        for actual, expected in zip(regimes, expected_regimes, strict=True)
    )
    false_negatives = sum(
        actual == 0 and expected == 1
        for actual, expected in zip(regimes, expected_regimes, strict=True)
    )
    settled_pairs = []
    transition_delays: list[int] = []
    pending_transition: tuple[int, int] | None = None
    for index, (actual, expected) in enumerate(
        zip(regimes, expected_regimes, strict=True)
    ):
        phase_offset = index % args.switch_block
        grace = swarm.monitor.minimum_dwell + (
            swarm.monitor.required_enter_streak
            if expected
            else swarm.monitor.required_exit_streak
        )
        if phase_offset >= grace:
            settled_pairs.append((actual, expected))
        if args.stress_switch and phase_offset == 0 and index > 0:
            pending_transition = (index, expected)
        if pending_transition is not None and actual == pending_transition[1]:
            transition_delays.append(index - pending_transition[0] + 1)
            pending_transition = None
    settled_normal = sum(expected == 0 for _, expected in settled_pairs)
    settled_crisis = sum(expected == 1 for _, expected in settled_pairs)
    settled_false_positives = sum(
        actual == 1 and expected == 0 for actual, expected in settled_pairs
    )
    settled_false_negatives = sum(
        actual == 0 and expected == 1 for actual, expected in settled_pairs
    )
    expected_switches = sum(
        left != right
        for left, right in zip(expected_regimes, expected_regimes[1:], strict=False)
    )
    observed_switches = (
        0 if detector_state is None else int(detector_state.switch_count.detach().cpu())
    )
    payload = {
        "status": "measurement_only",
        "device": str(device),
        "dtype": "float32",
        "batch": args.batch,
        "input_dim": args.input_dim,
        "state_dim": args.state_dim,
        "warmup": args.warmup,
        "iterations": args.iterations,
        "stress_switch": args.stress_switch,
        "switch_block": args.switch_block,
        "latency_ms": {
            "p50": float(torch.quantile(samples, 0.50)),
            "p95": float(torch.quantile(samples, 0.95)),
            "p99": float(torch.quantile(samples, 0.99)),
            "minimum": float(samples.min()),
            "maximum": float(samples.max()),
        },
        "safe_convergence_rate": converged / args.iterations,
        "maximum_residual": max(residuals),
        "pcas": {
            "stimulus_mahalanobis_ratio": {
                "normal": 0.0,
                "crisis": 2.0,
            },
            "expected_switches": expected_switches,
            "observed_switches": observed_switches,
            "excess_switches": max(0, observed_switches - expected_switches),
            "instantaneous_phase_false_positive_rate": (
                false_positives / normal_count if normal_count else 0.0
            ),
            "instantaneous_phase_false_negative_rate": (
                false_negatives / crisis_count if crisis_count else 0.0
            ),
            "settled_false_positive_rate": (
                settled_false_positives / settled_normal if settled_normal else 0.0
            ),
            "settled_false_negative_rate": (
                settled_false_negatives / settled_crisis if settled_crisis else 0.0
            ),
            "transition_settling_steps": {
                "samples": len(transition_delays),
                "maximum": max(transition_delays, default=0),
                "mean": (
                    sum(transition_delays) / len(transition_delays)
                    if transition_delays
                    else 0.0
                ),
            },
        },
        "topology": {
            "normal_edges": normal.edge_count,
            "crisis_edges": crisis.edge_count,
            "maximum_reachability_steps": max(
                normal.maximum_reachability_steps,
                crisis.maximum_reachability_steps,
            ),
        },
        "operator_norm": {
            "normal_upper_bound": float(
                swarm.operator_certificate("normal")
                .global_operator_norm_bound.detach()
                .cpu()
            ),
            "crisis_upper_bound": float(
                swarm.operator_certificate("crisis")
                .global_operator_norm_bound.detach()
                .cpu()
            ),
            "required_margin": swarm.config.global_margin,
        },
        "spectral_radius": {
            "normal": float(swarm.global_spectral_radius("normal").cpu()),
            "crisis": float(swarm.global_spectral_radius("crisis").cpu()),
        },
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if converged == args.iterations else 2


if __name__ == "__main__":
    raise SystemExit(main())
