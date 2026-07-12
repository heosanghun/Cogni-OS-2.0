"""C-FIRE scaled-polar projection, distinct from a spectral cap.

The existing spectral-cap guard only shrinks the largest singular value.  It
does not improve effective rank or conditioning.  This module implements the
Newton--Schulz polar iteration used by C-FIRE-style small-operator updates:
all non-zero singular values are driven toward one and then scaled by
``gamma``.  Candidate work is computed in FP32/FP64 and committed atomically
only after the casted target tensor passes the post-certificate.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import Tensor


MAX_CFIRE_ELEMENTS = 1_048_576


class CFireError(RuntimeError):
    """Raised when a scaled-polar candidate cannot be certified."""


@dataclass(frozen=True, slots=True)
class CFireCertificate:
    gamma: float
    spectral_margin: float
    iterations: int
    orthogonality_residual: float
    before_sigma_max: float
    before_sigma_min: float
    before_condition_number: float
    before_effective_rank: int
    after_sigma_max: float
    after_sigma_min: float
    after_condition_number: float
    after_effective_rank: int


def _spectrum(value: Tensor) -> Tensor:
    work_dtype = torch.float64 if value.dtype == torch.float64 else torch.float32
    singular = torch.linalg.svdvals(value.detach().to(dtype=work_dtype))
    if singular.numel() == 0 or not bool(torch.isfinite(singular).all()):
        raise CFireError("C-FIRE spectrum is empty or non-finite")
    return singular


def _spectral_facts(singular: Tensor) -> tuple[float, float, float, int]:
    maximum = float(singular.max())
    minimum = float(singular.min())
    condition = maximum / minimum if minimum > 0.0 else float("inf")
    threshold = maximum * 1.0e-6
    effective_rank = int((singular > threshold).sum())
    return maximum, minimum, condition, effective_rank


@torch.no_grad()
def c_fire_scaled_polar_(
    weight: Tensor,
    *,
    gamma: float = 0.90,
    spectral_margin: float = 0.95,
    tolerance: float = 1.0e-5,
    max_iter: int = 96,
    min_relative_singular: float = 1.0e-12,
) -> CFireCertificate:
    """Project one bounded matrix to a certified ``gamma`` polar factor.

    Rank-deficient matrices fail without mutation because Newton--Schulz
    cannot create a missing singular direction.  This is intentionally a
    small-operator primitive; large backbone matrices must not be duplicated
    inside the 16.7 GiB inference boundary.
    """

    if not isinstance(weight, Tensor):
        raise TypeError("weight must be a tensor")
    if weight.ndim != 2 or weight.numel() == 0:
        raise ValueError("weight must be a non-empty matrix")
    if not weight.is_floating_point() or weight.is_complex():
        raise TypeError("weight must be a real floating-point matrix")
    if weight.device.type == "meta" or weight.layout != torch.strided:
        raise ValueError("weight must be a materialized strided tensor")
    if weight.numel() > MAX_CFIRE_ELEMENTS:
        raise ValueError("weight exceeds the bounded C-FIRE operator size")
    numeric = (gamma, spectral_margin, tolerance, min_relative_singular)
    if not all(
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
        for value in numeric
    ):
        raise TypeError("C-FIRE numeric controls must be finite real values")
    if not 0.0 < float(gamma) < float(spectral_margin) < 1.0:
        raise ValueError("C-FIRE requires 0 < gamma < spectral_margin < 1")
    if not 0.0 < float(tolerance) < 0.1:
        raise ValueError("tolerance must lie in (0, 0.1)")
    if not 0.0 < float(min_relative_singular) < 1.0:
        raise ValueError("min_relative_singular must lie in (0, 1)")
    if not isinstance(max_iter, int) or isinstance(max_iter, bool) or max_iter < 1:
        raise ValueError("max_iter must be a positive integer")
    if not bool(torch.isfinite(weight).all()):
        raise CFireError("C-FIRE input contains non-finite values")

    before = _spectrum(weight)
    before_facts = _spectral_facts(before)
    if before_facts[0] <= 0.0:
        raise CFireError("C-FIRE input has zero spectral norm")
    if before_facts[1] / before_facts[0] <= float(min_relative_singular):
        raise CFireError("C-FIRE input is rank-deficient at the certified precision")

    work_dtype = torch.float64 if weight.dtype == torch.float64 else torch.float32
    candidate = weight.detach().to(dtype=work_dtype) / before_facts[0]
    rows, columns = candidate.shape
    dimension = columns if rows >= columns else rows
    identity = torch.eye(dimension, device=weight.device, dtype=work_dtype)
    residual = float("inf")
    iterations = 0
    for iterations in range(1, max_iter + 1):
        gram = candidate.T @ candidate if rows >= columns else candidate @ candidate.T
        residual_tensor = torch.linalg.matrix_norm(gram - identity, ord=2)
        residual = float(residual_tensor)
        if not math.isfinite(residual):
            raise CFireError("Newton--Schulz residual became non-finite")
        if residual <= float(tolerance):
            break
        correction = 0.5 * (3.0 * identity - gram)
        candidate = (
            candidate @ correction if rows >= columns else correction @ candidate
        )
        if not bool(torch.isfinite(candidate).all()):
            raise CFireError("Newton--Schulz candidate became non-finite")
    else:
        raise CFireError(
            "Newton--Schulz polar iteration did not reach its certified tolerance"
        )

    candidate = candidate * float(gamma)
    casted = candidate.to(dtype=weight.dtype)
    after = _spectrum(casted)
    after_facts = _spectral_facts(after)
    allowed_deviation = max(
        float(tolerance) * 8.0,
        0.01 if weight.dtype in {torch.float16, torch.bfloat16} else 1.0e-4,
    )
    if (
        after_facts[0] >= float(spectral_margin)
        or abs(after_facts[0] - float(gamma)) > allowed_deviation
        or abs(after_facts[1] - float(gamma)) > allowed_deviation
        or after_facts[3] != min(rows, columns)
        or not math.isfinite(after_facts[2])
        or after_facts[2] > 1.05
    ):
        raise CFireError("casted C-FIRE candidate failed the spectral certificate")

    weight.copy_(casted)
    return CFireCertificate(
        gamma=float(gamma),
        spectral_margin=float(spectral_margin),
        iterations=iterations,
        orthogonality_residual=residual,
        before_sigma_max=before_facts[0],
        before_sigma_min=before_facts[1],
        before_condition_number=before_facts[2],
        before_effective_rank=before_facts[3],
        after_sigma_max=after_facts[0],
        after_sigma_min=after_facts[1],
        after_condition_number=after_facts[2],
        after_effective_rank=after_facts[3],
    )


__all__ = [
    "CFireCertificate",
    "CFireError",
    "MAX_CFIRE_ELEMENTS",
    "c_fire_scaled_polar_",
]
