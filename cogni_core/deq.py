from __future__ import annotations

from dataclasses import dataclass
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


def _broyden_inverse(
    g: Callable[[Tensor], Tensor],
    z0: Tensor,
    *,
    tolerance: float,
    max_iter: int,
    history: int,
) -> tuple[Tensor, int, float, bool]:
    """Limited-memory inverse-Broyden root solve for g(z)=0.

    Only ``history`` rank-one correction pairs are retained, so solver activation
    storage is independent of the requested reasoning depth.
    """
    # Multisecant Anderson is the scalable L-Broyden path used by the
    # reference CTS repository for non-trivial state sizes.  Histories are
    # detached, bounded, and solved only in the small history dimension.
    z = z0
    xs: list[Tensor] = []
    fs: list[Tensor] = []
    for iteration in range(1, max_iter + 1):
        fixed = z + g(z)
        residual_vec = fixed - z
        residual = normalized_residual(residual_vec)
        if not torch.isfinite(fixed).all():
            return z0, iteration, float("inf"), False
        if residual <= tolerance:
            return fixed, iteration, residual, True
        xs.append(z.detach())
        fs.append(fixed.detach())
        if len(xs) > history + 1:
            xs.pop(0)
            fs.pop(0)
        if len(xs) < 2:
            z = fixed
            continue
        d_f = torch.stack([fs[i] - fs[i - 1] for i in range(1, len(fs))])
        d_x = torch.stack([xs[i] - xs[i - 1] for i in range(1, len(xs))])
        # One independent small solve per batch element.
        next_states = []
        for batch in range(z.shape[0]):
            df = d_f[:, batch].reshape(len(d_f), -1)
            dx = d_x[:, batch].reshape(len(d_x), -1)
            gram = df @ df.T + 1e-6 * torch.eye(len(df), device=z.device, dtype=z.dtype)
            rhs = df @ residual_vec[batch].reshape(-1)
            try:
                coeff = torch.linalg.solve(gram, rhs)
                correction = coeff @ (df - dx)
                next_states.append(
                    (fixed[batch].reshape(-1) - correction).reshape_as(fixed[batch])
                )
            except RuntimeError:
                next_states.append(fixed[batch])
        z = torch.stack(next_states)
    residual = normalized_residual(g(z))
    return z, max_iter, residual, residual <= tolerance


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
    ):
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
                    residual <= cfg.tolerance, cfg.fallback_steps, residual, norm, True
                )
            else:
                z, iterations, residual, converged = _broyden_inverse(
                    lambda state: f(state) - state,
                    z0,
                    tolerance=cfg.tolerance,
                    max_iter=cfg.max_iter,
                    history=cfg.history,
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
                info = SolverInfo(converged, iterations, residual, norm, used_fallback)
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
        return (*grads, None)


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
        z = _ImplicitTanh.apply(
            x, self.recurrent, self.input_weight, self.bias, self.config
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
        )
        return z
