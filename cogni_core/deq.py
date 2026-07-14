from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable

import torch
from torch import Tensor, nn


class ContractivityError(RuntimeError):
    """Raised when an implicit transition is unsafe and fallback is disabled."""


@dataclass(frozen=True)
class DEQConfig:
    tolerance: float = 1e-5
    max_iter: int = 80
    history: int = 8
    spectral_margin: float = 0.95
    fallback_damping: float = 0.35
    fallback_steps: int = 128
    # Production must never silently return an unsafe or unconverged state.
    # Research callers may opt into the bounded fallback explicitly.
    fail_on_noncontractive: bool = True


@dataclass
class SolverInfo:
    converged: bool
    iterations: int
    residual: float
    spectral_norm: float
    used_fallback: bool = False
    linear_solve_attempts: int = 0
    linear_solve_fallbacks: int = 0
    history_peak: int = 0
    rank: int = 0
    warm_used: bool = False
    warm_rejected: int = 0

    @property
    def used_linear_solve_fallback(self) -> bool:
        return self.linear_solve_fallbacks > 0


@dataclass(slots=True)
class BroydenTelemetry:
    """Fixed-size counters for one bounded limited-memory solve."""

    linear_solve_attempts: int = 0
    linear_solve_fallbacks: int = 0
    history_peak: int = 0
    rank: int = 0
    warm_used: bool = False
    warm_rejected: int = 0
    warm_rejection_reason: str | None = None

    @property
    def used_linear_solve_fallback(self) -> bool:
        return self.linear_solve_fallbacks > 0

    def reset(self, *, rank: int = 0) -> None:
        self.linear_solve_attempts = 0
        self.linear_solve_fallbacks = 0
        self.history_peak = 0
        self.rank = int(rank)
        self.warm_used = False
        self.warm_rejected = 0
        self.warm_rejection_reason = None


@dataclass(frozen=True, slots=True)
class BroydenWarmStart:
    """Detached, rank-bounded parent state and multisecant history capsule."""

    state: Tensor
    x_history: Tensor
    f_history: Tensor
    rank: int
    operator_id: str

    @property
    def history_size(self) -> int:
        return int(self.x_history.shape[0]) if self.x_history.ndim else 0


@dataclass(frozen=True, slots=True)
class LimitedBroydenResult:
    """Typed result for a bounded solve that never hides warm/fallback state."""

    state: Tensor
    iterations: int
    residual: float
    converged: bool
    warm_start: BroydenWarmStart
    rank: int
    history_peak: int
    linear_solve_attempts: int
    linear_solve_fallbacks: int
    warm_used: bool
    warm_rejected: int
    warm_rejection_reason: str | None

    @property
    def used_linear_solve_fallback(self) -> bool:
        return self.linear_solve_fallbacks > 0


def normalized_residual(value: Tensor) -> float:
    """Maximum per-sample RMS residual, invariant to latent width and sequence length."""
    if value.ndim <= 1:
        return float(value.square().mean().sqrt())
    flattened = value.reshape(value.shape[0], -1)
    return float(flattened.square().mean(dim=-1).sqrt().max())


def spectral_norm(weight: Tensor, steps: int = 12) -> Tensor:
    """Return the exact matrix spectral norm for a fail-closed safety gate.

    ``steps`` remains in the signature for API compatibility.  Power
    iteration is only a lower estimate until convergence and therefore cannot
    certify Rule-4 contractivity.
    """

    del steps
    with torch.no_grad():
        return torch.linalg.matrix_norm(weight, ord=2)


def _checked_root_value(g: Callable[[Tensor], Tensor], z: Tensor) -> Tensor:
    value = g(z)
    if not isinstance(value, Tensor):
        raise TypeError("Broyden root function must return a Tensor")
    if value.shape != z.shape:
        raise ValueError(
            "Broyden root function changed shape: "
            f"expected {tuple(z.shape)}, received {tuple(value.shape)}"
        )
    return value


def _final_broyden_result(
    g: Callable[[Tensor], Tensor],
    candidate: Tensor,
    z0: Tensor,
    iteration: int,
    tolerance: float,
) -> tuple[Tensor, int, float, bool]:
    """Re-evaluate the residual at exactly the finite state being returned."""

    z_star = candidate.detach() if torch.isfinite(candidate).all() else z0.detach()
    residual_value = _checked_root_value(g, z_star)
    residual = (
        normalized_residual(residual_value)
        if torch.isfinite(residual_value).all()
        else float("inf")
    )
    return z_star, iteration, residual, residual <= tolerance


def _validate_broyden_inputs(
    z0: Tensor,
    *,
    tolerance: float,
    max_iter: int,
    rank: int,
    operator_id: str,
    telemetry: BroydenTelemetry | None,
) -> BroydenTelemetry:
    if not isinstance(z0, Tensor) or z0.ndim < 1 or z0.shape[0] < 1:
        raise ValueError("Broyden initial state must have a non-empty batch axis")
    if not torch.is_floating_point(z0) or not torch.isfinite(z0).all():
        raise ValueError("Broyden initial state must be finite floating point")
    if not isinstance(max_iter, int) or isinstance(max_iter, bool) or max_iter < 1:
        raise ValueError("Broyden max_iter must be a positive integer")
    if not isinstance(rank, int) or isinstance(rank, bool) or rank < 1:
        raise ValueError("Broyden rank must be a positive integer")
    if (
        not isinstance(tolerance, (int, float))
        or isinstance(tolerance, bool)
        or not math.isfinite(float(tolerance))
        or not 0.0 < float(tolerance)
    ):
        raise ValueError("Broyden tolerance must be finite and positive")
    if (
        not isinstance(operator_id, str)
        or not 1 <= len(operator_id) <= 128
        or not operator_id.isascii()
    ):
        raise ValueError("operator_id must be 1-128 ASCII characters")
    if telemetry is not None and not isinstance(telemetry, BroydenTelemetry):
        raise TypeError("telemetry must be BroydenTelemetry or None")
    stats = telemetry if telemetry is not None else BroydenTelemetry()
    stats.reset(rank=rank)
    return stats


def _warm_rejection_reason(
    warm_start: BroydenWarmStart,
    z0: Tensor,
    *,
    rank: int,
    operator_id: str,
) -> str | None:
    if warm_start.operator_id != operator_id:
        return "operator_mismatch"
    if warm_start.rank != rank:
        return "rank_mismatch"
    tensors = (warm_start.state, warm_start.x_history, warm_start.f_history)
    if any(not isinstance(value, Tensor) for value in tensors):
        return "non_tensor"
    if tuple(warm_start.state.shape) != tuple(z0.shape):
        return "shape_mismatch"
    expected_history_ndim = z0.ndim + 1
    if (
        warm_start.x_history.ndim != expected_history_ndim
        or warm_start.f_history.ndim != expected_history_ndim
        or tuple(warm_start.x_history.shape) != tuple(warm_start.f_history.shape)
        or tuple(warm_start.x_history.shape[1:]) != tuple(z0.shape)
        or warm_start.history_size > rank + 1
    ):
        return "history_shape_mismatch"
    if any(value.dtype != z0.dtype for value in tensors):
        return "dtype_mismatch"
    if any(value.device != z0.device for value in tensors):
        return "device_mismatch"
    if any(value.layout != torch.strided for value in tensors):
        return "layout_mismatch"
    if any(not bool(torch.isfinite(value).all()) for value in tensors):
        return "nonfinite"
    return None


@torch.no_grad()
def _run_limited_broyden(
    g: Callable[[Tensor], Tensor],
    z0: Tensor,
    *,
    tolerance: float,
    max_iter: int,
    rank: int,
    operator_id: str,
    warm_start: BroydenWarmStart | None,
    telemetry: BroydenTelemetry | None,
) -> tuple[Tensor, int, float, bool, list[Tensor], list[Tensor]]:
    stats = _validate_broyden_inputs(
        z0,
        tolerance=tolerance,
        max_iter=max_iter,
        rank=rank,
        operator_id=operator_id,
        telemetry=telemetry,
    )
    if warm_start is not None and not isinstance(warm_start, BroydenWarmStart):
        raise TypeError("warm_start must be BroydenWarmStart or None")

    rejection = (
        None
        if warm_start is None
        else _warm_rejection_reason(
            warm_start,
            z0,
            rank=rank,
            operator_id=operator_id,
        )
    )
    if rejection is not None:
        stats.warm_rejected = 1
        stats.warm_rejection_reason = rejection
        warm_start = None
    if warm_start is None:
        z = z0
        xs: list[Tensor] = []
        fs: list[Tensor] = []
    else:
        stats.warm_used = True
        z = warm_start.state.detach().clone()
        xs = list(warm_start.x_history.detach().unbind(0))
        fs = list(warm_start.f_history.detach().unbind(0))
        stats.history_peak = max(0, len(xs) - 1)

    # Multisecant Anderson is the scalable L-Broyden path used by the
    # reference CTS repository for non-trivial state sizes.  Histories are
    # detached, bounded, and solved only in the small history dimension.
    for iteration in range(1, max_iter + 1):
        residual_vec = _checked_root_value(g, z)
        fixed = z + residual_vec
        residual = normalized_residual(residual_vec)
        if not torch.isfinite(fixed).all():
            final = _final_broyden_result(g, z0, z0, iteration, tolerance)
            return (*final, xs, fs)
        if residual <= tolerance:
            result = _final_broyden_result(g, fixed, z0, iteration, tolerance)
            if result[3]:
                return (*result, xs, fs)
        xs.append(z.detach())
        fs.append(fixed.detach())
        if len(xs) > rank + 1:
            xs.pop(0)
            fs.pop(0)
        # ``rank`` counts multisecant correction pairs, while their endpoint
        # representation necessarily contains at most ``rank + 1`` states.
        stats.history_peak = max(stats.history_peak, max(0, len(xs) - 1))
        if len(xs) < 2:
            z = fixed
            continue
        d_f = torch.stack([fs[i] - fs[i - 1] for i in range(1, len(fs))])
        d_x = torch.stack([xs[i] - xs[i - 1] for i in range(1, len(xs))])
        d_g = d_f - d_x
        # One independent small solve per batch element.
        next_states = []
        for batch in range(z.shape[0]):
            df = d_f[:, batch].reshape(len(d_f), -1)
            dg = d_g[:, batch].reshape(len(d_g), -1)
            # CUDA and CPU linalg.solve do not support BF16/FP16. Keep the
            # large latent/history banks in their original dtype and promote
            # only this rank-sized numerical solve to FP32. Float64 research
            # tests retain their higher-precision solve.
            solve_dtype = torch.float64 if z.dtype == torch.float64 else torch.float32
            df_solve = df.to(dtype=solve_dtype)
            dg_solve = dg.to(dtype=solve_dtype)
            residual_solve = residual_vec[batch].reshape(-1).to(dtype=solve_dtype)
            gram_base = dg_solve @ dg_solve.T
            # Use scale-aware Tikhonov damping.  A fixed ridge alone becomes
            # negligible when BF16 history rows have a large dynamic range,
            # while a purely relative ridge disappears for nearly duplicate
            # rows.  The combined floor keeps the rank-sized system positive
            # definite in both regimes without allocating depth-dependent
            # state.
            diagonal_scale = gram_base.diagonal().mean().abs()
            ridge = 1.0e-6 + torch.finfo(solve_dtype).eps * 64.0 * diagonal_scale
            gram = gram_base + ridge * torch.eye(
                len(dg_solve), device=z.device, dtype=solve_dtype
            )
            rhs = dg_solve @ residual_solve
            stats.linear_solve_attempts += 1
            try:
                coeff = torch.linalg.solve(gram, rhs)
                # Type-II Anderson / multisecant inverse-Broyden update:
                # solve against delta-g, then apply the corresponding
                # delta-fixed-map correction.
                correction = coeff @ df_solve
                if (
                    not torch.isfinite(coeff).all()
                    or not torch.isfinite(correction).all()
                ):
                    raise RuntimeError("small Broyden solve returned non-finite values")
                # A regularized history solve can remain algebraically finite
                # while proposing an over-large step. Apply a deterministic
                # trust-region projection instead of discarding an otherwise
                # valid solve. This is part of the primary bounded Broyden
                # update; actual solve failures and non-finite values still use
                # the explicitly counted fixed-point fallback below.
                residual_flat = residual_solve
                correction_rms = correction.square().mean().sqrt()
                residual_rms = residual_flat.square().mean().sqrt()
                trust_floor = torch.finfo(solve_dtype).eps * 32
                trust_radius = 4.0 * residual_rms + trust_floor
                correction_scale = torch.clamp(
                    trust_radius / (correction_rms + trust_floor), max=1.0
                )
                correction = correction * correction_scale
                candidate = fixed[batch].reshape(-1).to(solve_dtype) - correction
                if not torch.isfinite(candidate).all():
                    raise RuntimeError(
                        "small Broyden correction produced a non-finite state"
                    )
                candidate = candidate.to(dtype=z.dtype).reshape_as(fixed[batch])
                if not torch.isfinite(candidate).all():
                    raise RuntimeError(
                        "small Broyden correction overflowed the state dtype"
                    )
                next_states.append(candidate)
            except RuntimeError:
                stats.linear_solve_fallbacks += 1
                next_states.append(fixed[batch])
        z = torch.stack(next_states)
    final = _final_broyden_result(g, z, z0, max_iter, tolerance)
    return (*final, xs, fs)


def _make_warm_start(
    state: Tensor,
    xs: list[Tensor],
    fs: list[Tensor],
    *,
    rank: int,
    operator_id: str,
) -> BroydenWarmStart:
    capacity = rank + 1
    selected_xs = xs[-capacity:]
    selected_fs = fs[-capacity:]
    if selected_xs:
        x_history = torch.stack(selected_xs).detach().clone()
        f_history = torch.stack(selected_fs).detach().clone()
    else:
        shape = (0, *state.shape)
        x_history = torch.empty(shape, device=state.device, dtype=state.dtype)
        f_history = torch.empty(shape, device=state.device, dtype=state.dtype)
    return BroydenWarmStart(
        state=state.detach().clone(),
        x_history=x_history,
        f_history=f_history,
        rank=rank,
        operator_id=operator_id,
    )


@torch.no_grad()
def limited_broyden_solve(
    g: Callable[[Tensor], Tensor],
    z0: Tensor,
    *,
    tolerance: float,
    max_iter: int,
    rank: int = 16,
    operator_id: str,
    warm_start: BroydenWarmStart | None = None,
    telemetry: BroydenTelemetry | None = None,
) -> LimitedBroydenResult:
    """Solve ``g(z)=0`` and return explicit bounded history/warm telemetry."""

    stats = telemetry if telemetry is not None else BroydenTelemetry()
    state, iterations, residual, converged, xs, fs = _run_limited_broyden(
        g,
        z0,
        tolerance=tolerance,
        max_iter=max_iter,
        rank=rank,
        operator_id=operator_id,
        warm_start=warm_start,
        telemetry=stats,
    )
    capsule = _make_warm_start(
        state,
        xs,
        fs,
        rank=rank,
        operator_id=operator_id,
    )
    return LimitedBroydenResult(
        state=state,
        iterations=iterations,
        residual=residual,
        converged=converged,
        warm_start=capsule,
        rank=stats.rank,
        history_peak=stats.history_peak,
        linear_solve_attempts=stats.linear_solve_attempts,
        linear_solve_fallbacks=stats.linear_solve_fallbacks,
        warm_used=stats.warm_used,
        warm_rejected=stats.warm_rejected,
        warm_rejection_reason=stats.warm_rejection_reason,
    )


@torch.no_grad()
def _broyden_inverse(
    g: Callable[[Tensor], Tensor],
    z0: Tensor,
    *,
    tolerance: float,
    max_iter: int,
    history: int,
    telemetry: BroydenTelemetry | None = None,
) -> tuple[Tensor, int, float, bool]:
    """Compatibility wrapper retaining the original four-value API."""

    state, iterations, residual, converged, _xs, _fs = _run_limited_broyden(
        g,
        z0,
        tolerance=tolerance,
        max_iter=max_iter,
        rank=history,
        operator_id="legacy",
        warm_start=None,
        telemetry=telemetry,
    )
    return state, iterations, residual, converged


def _damped_fixed_point(
    f: Callable[[Tensor], Tensor], z: Tensor, damping: float, steps: int
) -> Tensor:
    for _ in range(steps):
        z = (1.0 - damping) * z + damping * f(z)
    return z


class _ImplicitTanh(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x: Tensor,
        recurrent: Tensor,
        input_weight: Tensor,
        bias: Tensor,
        cfg: DEQConfig,
        telemetry: BroydenTelemetry,
    ):
        if not isinstance(telemetry, BroydenTelemetry):
            raise TypeError("DEQ telemetry must be BroydenTelemetry")
        telemetry.reset()
        norm_tensor = spectral_norm(recurrent)
        if not torch.isfinite(norm_tensor):
            raise ContractivityError(
                "recurrent operator has a non-finite spectral norm"
            )
        norm = float(norm_tensor)
        unsafe = norm >= cfg.spectral_margin
        if unsafe and cfg.fail_on_noncontractive:
            raise ContractivityError(
                f"spectral norm {norm:.4f} exceeds margin {cfg.spectral_margin:.4f}"
            )
        with torch.no_grad():
            drive = x @ input_weight.T + bias

            def f(z: Tensor) -> Tensor:
                return torch.tanh(z @ recurrent.T + drive)

            z0 = torch.zeros_like(drive)
            if unsafe:
                z = _damped_fixed_point(f, z0, cfg.fallback_damping, cfg.fallback_steps)
                residual = normalized_residual(f(z) - z)
                info = SolverInfo(
                    residual <= cfg.tolerance,
                    cfg.fallback_steps,
                    residual,
                    norm,
                    True,
                )
            else:
                z, iterations, residual, converged = _broyden_inverse(
                    lambda state: f(state) - state,
                    z0,
                    tolerance=cfg.tolerance,
                    max_iter=cfg.max_iter,
                    history=cfg.history,
                    telemetry=telemetry,
                )
                used_fallback = not converged
                if not converged:
                    if not torch.isfinite(z).all():
                        z = z0
                    z = _damped_fixed_point(
                        f, z, cfg.fallback_damping, cfg.fallback_steps
                    )
                    residual = normalized_residual(f(z) - z)
                    converged = residual <= cfg.tolerance
                info = SolverInfo(
                    converged,
                    iterations,
                    residual,
                    norm,
                    used_fallback,
                    telemetry.linear_solve_attempts,
                    telemetry.linear_solve_fallbacks,
                    telemetry.history_peak,
                    telemetry.rank,
                    telemetry.warm_used,
                    telemetry.warm_rejected,
                )
            if not info.converged and cfg.fail_on_noncontractive:
                raise ContractivityError(
                    "DEQ transition did not converge within the configured bounded "
                    f"solve (residual={info.residual:.4e}, "
                    f"iterations={info.iterations})"
                )
        ctx.save_for_backward(x, recurrent, input_weight, bias, z)
        ctx.cfg = cfg
        ctx.info = info
        return z

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        x, recurrent, input_weight, bias, z_star = ctx.saved_tensors
        with torch.enable_grad():
            x_ = x.detach().requires_grad_(x.requires_grad)
            w_ = recurrent.detach().requires_grad_(recurrent.requires_grad)
            u_ = input_weight.detach().requires_grad_(input_weight.requires_grad)
            b_ = bias.detach().requires_grad_(bias.requires_grad)
            z_ = z_star.detach().requires_grad_(True)
            fz = torch.tanh(z_ @ w_.T + x_ @ u_.T + b_)

            # Solve v = grad + J_f^T v. Contractivity makes this fixed-point solve stable.
            v = grad_output
            for _ in range(ctx.cfg.max_iter):
                jtv = torch.autograd.grad(
                    fz, z_, v, retain_graph=True, create_graph=False
                )[0]
                next_v = grad_output + jtv
                if normalized_residual(next_v - v) <= ctx.cfg.tolerance:
                    v = next_v
                    break
                v = next_v
            grads = torch.autograd.grad(
                fz,
                (x_, w_, u_, b_),
                v,
                allow_unused=True,
                retain_graph=False,
            )
        return (*grads, None, None)


class EquilibriumLayer(nn.Module):
    def __init__(
        self, input_dim: int, hidden_dim: int, config: DEQConfig | None = None
    ):
        super().__init__()
        self.config = config or DEQConfig()
        self.recurrent = nn.Parameter(torch.empty(hidden_dim, hidden_dim))
        self.input_weight = nn.Parameter(torch.empty(hidden_dim, input_dim))
        self.bias = nn.Parameter(torch.zeros(hidden_dim))
        self.last_info: SolverInfo | None = None
        nn.init.orthogonal_(self.recurrent)
        nn.init.xavier_uniform_(self.input_weight)
        with torch.no_grad():
            self.recurrent.mul_(0.65)

    def forward(self, x: Tensor) -> Tensor:
        telemetry = BroydenTelemetry()
        z = _ImplicitTanh.apply(
            x,
            self.recurrent,
            self.input_weight,
            self.bias,
            self.config,
            telemetry,
        )
        # autograd.Function metadata is unavailable on the returned tensor; recompute only the cheap guard fields.
        norm = float(spectral_norm(self.recurrent))
        residual = normalized_residual(
            (
                torch.tanh(z @ self.recurrent.T + x @ self.input_weight.T + self.bias)
                - z
            ).detach()
        )
        self.last_info = SolverInfo(
            residual <= self.config.tolerance,
            -1,
            residual,
            norm,
            norm >= self.config.spectral_margin,
            telemetry.linear_solve_attempts,
            telemetry.linear_solve_fallbacks,
            telemetry.history_peak,
            telemetry.rank,
            telemetry.warm_used,
            telemetry.warm_rejected,
        )
        return z
