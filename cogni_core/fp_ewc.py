from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import torch
from torch import Tensor, nn


def spectral_guard_(weight: Tensor, margin: float = 0.95) -> float:
    """Project a matrix below the requested Banach safety margin in-place."""
    if weight.ndim < 2 or not weight.is_floating_point():
        raise TypeError("spectral_guard_ requires a floating-point matrix")
    if not 0.0 < margin < 1.0:
        raise ValueError("margin must be in (0, 1)")
    with torch.no_grad():
        # torch.linalg.matrix_norm does not support BF16 on all CPU/CUDA
        # backends. The exact SVD is therefore evaluated in FP32 while the
        # in-place projection preserves the original parameter dtype.
        sigma = torch.linalg.matrix_norm(weight.float(), ord=2)
        if not torch.isfinite(sigma):
            raise RuntimeError("spectral norm is non-finite")
        if sigma < margin:
            return float(sigma)
        dtype_epsilon = float(torch.finfo(weight.dtype).eps)
        interior = max(1.0e-5, 4.0 * dtype_epsilon)
        target = sigma.new_tensor(margin * (1.0 - interior))
        for _ in range(3):
            weight.mul_(
                torch.clamp(
                    target / sigma.clamp_min(1e-12),
                    max=1.0,
                ).to(weight)
            )
            projected = torch.linalg.matrix_norm(weight.float(), ord=2)
            if not torch.isfinite(projected):
                raise RuntimeError("spectral projection produced a non-finite norm")
            if projected < margin:
                return float(projected)
            # Low-precision element rounding can rarely land on the boundary;
            # move farther inside and retry a bounded number of times.
            sigma = projected
            target.mul_(0.9)
        raise RuntimeError("spectral projection failed its strict postcondition")


def estimate_fixed_point_fisher(
    *,
    f_at_z,
    z_star: Tensor,
    log_likelihood_at_z,
    named_parameters: list[tuple[str, nn.Parameter]],
    tolerance: float = 1e-5,
    max_iter: int = 80,
) -> dict[str, Tensor]:
    """Matrix-free diagonal Fisher using the DEQ implicit adjoint.

    Solves ``v = score_z + J_f(z*)^T v`` without materializing a Jacobian or
    retaining solver iterations. This is stable under the same contraction gate
    required by the forward DEQ.
    """
    z = z_star.detach().requires_grad_(True)
    with torch.enable_grad():
        fz = f_at_z(z)
        log_p = log_likelihood_at_z(z)
    (score,) = torch.autograd.grad(log_p, z, retain_graph=True)
    v = score
    for _ in range(max_iter):
        (jtv,) = torch.autograd.grad(fz, z, v, retain_graph=True)
        next_v = score + jtv
        delta = next_v - v
        if float(delta.square().mean().sqrt().detach()) <= tolerance:
            v = next_v
            break
        v = next_v
    params = [p for _, p in named_parameters]
    grads = torch.autograd.grad(fz, params, v, allow_unused=True)
    return {
        name: torch.zeros_like(param) if grad is None else grad.detach().square()
        for (name, param), grad in zip(named_parameters, grads)
    }


@dataclass
class FisherSnapshot:
    fisher: dict[str, Tensor]
    anchor: dict[str, Tensor]


@dataclass
class FPEWCRegularizer:
    strength: float = 1.0
    snapshots: list[FisherSnapshot] = field(default_factory=list)

    def consolidate(
        self,
        named_parameters: Iterable[tuple[str, nn.Parameter]],
        grads: dict[str, Tensor],
    ) -> None:
        params = list(named_parameters)
        self.snapshots.append(
            FisherSnapshot(
                {
                    name: grads[name].detach().square().clone()
                    for name, _ in params
                    if name in grads
                },
                {name: p.detach().clone() for name, p in params},
            )
        )

    def penalty(self, named_parameters: Iterable[tuple[str, nn.Parameter]]) -> Tensor:
        params = list(named_parameters)
        if not params:
            return torch.zeros(())
        total = params[0][1].new_zeros(())
        lookup = dict(params)
        for snapshot in self.snapshots:
            for name, fisher in snapshot.fisher.items():
                if name in lookup:
                    parameter = lookup[name]
                    if (
                        fisher.device != parameter.device
                        or fisher.dtype != parameter.dtype
                    ):
                        fisher = fisher.to(parameter)
                        snapshot.fisher[name] = fisher
                    anchor = snapshot.anchor[name]
                    if (
                        anchor.device != parameter.device
                        or anchor.dtype != parameter.dtype
                    ):
                        anchor = anchor.to(parameter)
                        snapshot.anchor[name] = anchor
                    total = (
                        total
                        + 0.5
                        * self.strength
                        * (fisher * (parameter - anchor).square()).sum()
                    )
        return total
