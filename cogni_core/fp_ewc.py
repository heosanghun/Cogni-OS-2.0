"""Verified fixed-point empirical Fisher and bounded EWC snapshots.

The typed API in this module computes a true per-sample empirical Fisher.  For
each sample it solves the implicit adjoint with a contractive Picard iteration,
checks the explicit linear residual, combines the direct and implicit parameter
gradients, squares that sample's total gradient, and only then averages.

``estimate_fixed_point_fisher`` and ``spectral_guard_`` remain compatibility
wrappers.  New code should use :func:`estimate_empirical_fixed_point_fisher`
and :func:`verified_spectral_cap_` so neither a legacy estimator nor a plain
spectral projection is mislabeled as a stronger algorithm.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Callable, Iterable, Mapping
import warnings

import torch
from torch import Tensor, nn


_FP32_ELEMENT_BYTES = 4


class FixedPointFisherError(RuntimeError):
    """Raised when a fixed-point Fisher safety postcondition is not met."""


@dataclass(frozen=True, slots=True)
class AdjointTelemetry:
    """Bounded evidence for one per-sample implicit adjoint solve."""

    sample_index: int
    converged: bool
    iterations: int
    residual: float
    initial_residual: float
    contraction_bound: float
    solver: str = "contractive_picard"

    def __post_init__(self) -> None:
        if self.sample_index < 0:
            raise ValueError("sample_index must be non-negative")
        if self.iterations < 0:
            raise ValueError("iterations must be non-negative")
        if not math.isfinite(self.residual) or self.residual < 0.0:
            raise ValueError("adjoint residual must be finite and non-negative")
        if not math.isfinite(self.initial_residual) or self.initial_residual < 0.0:
            raise ValueError("initial residual must be finite and non-negative")
        if not 0.0 <= self.contraction_bound < 0.95:
            raise ValueError("contraction_bound must lie in [0, 0.95)")


class AdjointConvergenceError(FixedPointFisherError):
    """Fail-closed adjoint error retaining the final bounded telemetry."""

    def __init__(self, telemetry: AdjointTelemetry) -> None:
        self.telemetry = telemetry
        super().__init__(
            "implicit adjoint did not converge: "
            f"sample={telemetry.sample_index}, residual={telemetry.residual:.6e}, "
            f"iterations={telemetry.iterations}"
        )


@dataclass(frozen=True, slots=True)
class FixedPointFisherConfig:
    """Hard bounds and numerical tolerances for empirical FP-Fisher."""

    contraction_bound: float
    fixed_point_tolerance: float = 1.0e-5
    adjoint_tolerance: float = 1.0e-6
    max_adjoint_iterations: int = 512
    max_samples: int = 64
    max_parameters: int = 1_024
    max_state_elements: int = 1_048_576
    max_fisher_bytes: int = 64 * 1024 * 1024

    HARD_MAX_SAMPLES = 1_024
    HARD_MAX_PARAMETERS = 65_536
    HARD_MAX_STATE_ELEMENTS = 16_777_216
    HARD_MAX_FISHER_BYTES = 1024 * 1024 * 1024

    def __post_init__(self) -> None:
        numeric = (
            self.contraction_bound,
            self.fixed_point_tolerance,
            self.adjoint_tolerance,
        )
        if any(not math.isfinite(float(value)) for value in numeric):
            raise ValueError("Fisher contraction and tolerances must be finite")
        if not 0.0 <= self.contraction_bound < 0.95:
            raise ValueError("contraction_bound must lie in [0, 0.95)")
        if self.fixed_point_tolerance <= 0.0 or self.adjoint_tolerance <= 0.0:
            raise ValueError("Fisher tolerances must be positive")
        limits = {
            "max_adjoint_iterations": (self.max_adjoint_iterations, 1, 16_384),
            "max_samples": (self.max_samples, 1, self.HARD_MAX_SAMPLES),
            "max_parameters": (self.max_parameters, 1, self.HARD_MAX_PARAMETERS),
            "max_state_elements": (
                self.max_state_elements,
                1,
                self.HARD_MAX_STATE_ELEMENTS,
            ),
            "max_fisher_bytes": (
                self.max_fisher_bytes,
                1,
                self.HARD_MAX_FISHER_BYTES,
            ),
        }
        for name, (value, minimum, maximum) in limits.items():
            if (
                not isinstance(value, int)
                or isinstance(value, bool)
                or not minimum <= value <= maximum
            ):
                raise ValueError(f"{name} must lie in [{minimum}, {maximum}]")


@dataclass(frozen=True, slots=True)
class FixedPointFisherEstimate:
    """CPU-FP32 diagonal empirical Fisher with solve evidence."""

    fisher: Mapping[str, Tensor]
    n_samples: int
    fixed_point_residuals: tuple[float, ...]
    contraction_bound: float
    adjoint: tuple[AdjointTelemetry, ...]

    def __post_init__(self) -> None:
        if self.n_samples <= 0 or len(self.fixed_point_residuals) != self.n_samples:
            raise ValueError("Fisher sample metadata is inconsistent")
        if len(self.adjoint) != self.n_samples:
            raise ValueError("one adjoint telemetry record is required per sample")
        if any(
            not math.isfinite(value) or value < 0.0
            for value in self.fixed_point_residuals
        ):
            raise ValueError("fixed-point residual evidence must be finite")
        if not 0.0 <= self.contraction_bound < 0.95:
            raise ValueError("estimate contraction_bound must lie in [0, 0.95)")
        if any(
            telemetry.sample_index != index or not telemetry.converged
            for index, telemetry in enumerate(self.adjoint)
        ):
            raise ValueError("adjoint telemetry is incomplete or unconverged")
        for name, value in self.fisher.items():
            if not name:
                raise ValueError("Fisher parameter names must be non-empty")
            if value.device.type != "cpu" or value.dtype != torch.float32:
                raise ValueError("empirical Fisher tensors must be CPU FP32")
            if not torch.isfinite(value).all() or (value < 0.0).any():
                raise ValueError("empirical Fisher must be finite and non-negative")


def verified_spectral_cap_(weight: Tensor, margin: float = 0.95) -> float:
    """Project matrix slices strictly below ``margin`` and verify the result.

    The projection preserves the parameter dtype and handles arbitrary leading
    batch dimensions independently over the final two matrix dimensions.
    """

    if weight.ndim < 2 or not weight.is_floating_point():
        raise TypeError("verified_spectral_cap_ requires a floating-point matrix")
    if not math.isfinite(float(margin)) or not 0.0 < margin < 1.0:
        raise ValueError("margin must be finite and in (0, 1)")
    with torch.no_grad():
        sigma = torch.linalg.matrix_norm(weight.float(), ord=2)
        if not torch.isfinite(sigma).all():
            raise RuntimeError("spectral norm is non-finite")
        if bool((sigma < margin).all().detach().cpu()):
            return float(sigma.max().detach().cpu())
        dtype_epsilon = float(torch.finfo(weight.dtype).eps)
        interior = max(1.0e-5, 4.0 * dtype_epsilon)
        target = sigma.new_tensor(margin * (1.0 - interior))
        for _ in range(3):
            scales = torch.clamp(target / sigma.clamp_min(1.0e-12), max=1.0)
            weight.mul_(scales.to(weight)[..., None, None])
            projected = torch.linalg.matrix_norm(weight.float(), ord=2)
            if not torch.isfinite(projected).all():
                raise RuntimeError("spectral projection produced a non-finite norm")
            if bool((projected < margin).all().detach().cpu()):
                return float(projected.max().detach().cpu())
            sigma = projected
            target.mul_(0.9)
        raise RuntimeError("spectral projection failed its strict postcondition")


def spectral_guard_(weight: Tensor, margin: float = 0.95) -> float:
    """Deprecated alias for :func:`verified_spectral_cap_`."""

    warnings.warn(
        "spectral_guard_ is deprecated; use verified_spectral_cap_",
        DeprecationWarning,
        stacklevel=2,
    )
    return verified_spectral_cap_(weight, margin)


def _rms(value: Tensor) -> float:
    return float(value.detach().float().square().mean().sqrt().cpu())


def _validate_parameters(
    named_parameters: Iterable[tuple[str, nn.Parameter]], config: FixedPointFisherConfig
) -> list[tuple[str, nn.Parameter]]:
    parameters = list(named_parameters)
    if not parameters:
        raise ValueError("named_parameters must not be empty")
    if len(parameters) > config.max_parameters:
        raise FixedPointFisherError("parameter-count budget exceeded")
    names: set[str] = set()
    for name, parameter in parameters:
        if not isinstance(name, str) or not 1 <= len(name) <= 256 or name in names:
            raise ValueError("parameter names must be unique bounded text")
        if not isinstance(parameter, nn.Parameter):
            raise TypeError("named_parameters values must be nn.Parameter objects")
        if not parameter.requires_grad or not parameter.is_floating_point():
            raise ValueError("Fisher parameters must be trainable floating tensors")
        names.add(name)
    return parameters


def _vjp(
    outputs: Tensor,
    inputs: Tensor | list[nn.Parameter],
    vector: Tensor | None = None,
) -> tuple[Tensor | None, ...]:
    input_list = [inputs] if isinstance(inputs, Tensor) else inputs
    if not outputs.requires_grad:
        return tuple(None for _ in input_list)
    return torch.autograd.grad(
        outputs,
        input_list,
        grad_outputs=vector,
        retain_graph=True,
        allow_unused=True,
    )


def _solve_contracting_adjoint(
    *,
    fz: Tensor,
    z: Tensor,
    score: Tensor,
    sample_index: int,
    config: FixedPointFisherConfig,
) -> tuple[Tensor, AdjointTelemetry]:
    v = score.detach().clone()
    initial_residual: float | None = None
    final_residual = float("inf")
    for iteration in range(config.max_adjoint_iterations + 1):
        (jtv_raw,) = _vjp(fz, z, v)
        jtv = torch.zeros_like(v) if jtv_raw is None else jtv_raw.detach()
        if not torch.isfinite(jtv).all():
            raise FixedPointFisherError("adjoint VJP produced non-finite values")
        residual_tensor = v - score - jtv
        final_residual = _rms(residual_tensor)
        if not math.isfinite(final_residual):
            raise FixedPointFisherError("adjoint residual is non-finite")
        if initial_residual is None:
            initial_residual = final_residual
        if final_residual <= config.adjoint_tolerance:
            telemetry = AdjointTelemetry(
                sample_index=sample_index,
                converged=True,
                iterations=iteration,
                residual=final_residual,
                initial_residual=initial_residual,
                contraction_bound=config.contraction_bound,
            )
            return v, telemetry
        if iteration == config.max_adjoint_iterations:
            telemetry = AdjointTelemetry(
                sample_index=sample_index,
                converged=False,
                iterations=iteration,
                residual=final_residual,
                initial_residual=initial_residual,
                contraction_bound=config.contraction_bound,
            )
            raise AdjointConvergenceError(telemetry)
        v = (score + jtv).detach()
    raise AssertionError("bounded adjoint loop ended without a terminal result")


def estimate_empirical_fixed_point_fisher(
    *,
    f_at_z: Callable[[Tensor], Tensor],
    z_star: Tensor,
    log_likelihood_per_sample: Callable[[Tensor], Tensor],
    named_parameters: Iterable[tuple[str, nn.Parameter]],
    config: FixedPointFisherConfig,
    solver_converged: bool,
) -> FixedPointFisherEstimate:
    """Compute a per-sample empirical Fisher through the implicit fixed point."""

    if not isinstance(config, FixedPointFisherConfig):
        raise TypeError("config must be FixedPointFisherConfig")
    if not isinstance(solver_converged, bool):
        raise TypeError("solver_converged must be bool")
    if not solver_converged:
        raise FixedPointFisherError("forward fixed-point solver did not converge")
    if not isinstance(z_star, Tensor) or z_star.ndim < 2:
        raise ValueError("z_star must have shape [samples, ...]")
    if not z_star.is_floating_point() or not torch.isfinite(z_star).all():
        raise ValueError("z_star must be a finite floating tensor")
    batch = int(z_star.shape[0])
    if not 1 <= batch <= config.max_samples:
        raise FixedPointFisherError("sample-count budget exceeded")
    if z_star.numel() > config.max_state_elements:
        raise FixedPointFisherError("fixed-point state budget exceeded")
    named = _validate_parameters(named_parameters, config)
    parameters = [parameter for _, parameter in named]
    if any(parameter.device != z_star.device for parameter in parameters):
        raise ValueError("z_star and Fisher parameters must share one device")

    z = z_star.detach().requires_grad_(True)
    with torch.enable_grad():
        fz = f_at_z(z)
        if not isinstance(fz, Tensor) or tuple(fz.shape) != tuple(z.shape):
            raise ValueError("f_at_z must return a tensor matching z_star")
        if fz.device != z.device or fz.dtype != z.dtype:
            raise ValueError("f_at_z must preserve the fixed-point device and dtype")
        if not torch.isfinite(fz).all():
            raise FixedPointFisherError("fixed-point map produced non-finite values")
        log_values = log_likelihood_per_sample(z)
        if not isinstance(log_values, Tensor) or tuple(log_values.shape) != (batch,):
            raise ValueError("log_likelihood_per_sample must return shape [samples]")
        if log_values.device != z.device or not log_values.is_floating_point():
            raise ValueError("log likelihood must be floating and share the z device")
        if not torch.isfinite(log_values).all():
            raise FixedPointFisherError("log likelihood contains non-finite values")

        residual_flat = (fz - z).reshape(batch, -1)
        residual_values = residual_flat.float().square().mean(dim=1).sqrt()
        if not torch.isfinite(residual_values).all():
            raise FixedPointFisherError("fixed-point residual is non-finite")
        residuals = tuple(float(value.detach().cpu()) for value in residual_values)
        if any(value > config.fixed_point_tolerance for value in residuals):
            raise FixedPointFisherError(
                "z_star fails the fixed-point residual postcondition"
            )

        accumulators: dict[str, Tensor] = {}
        active_bytes = 0
        telemetry: list[AdjointTelemetry] = []
        for sample_index in range(batch):
            sample_log_likelihood = log_values[sample_index]
            (score_raw,) = _vjp(sample_log_likelihood, z)
            score = torch.zeros_like(z) if score_raw is None else score_raw.detach()
            if not torch.isfinite(score).all():
                raise FixedPointFisherError("state score contains non-finite values")
            direct = _vjp(sample_log_likelihood, parameters)
            v, sample_telemetry = _solve_contracting_adjoint(
                fz=fz,
                z=z,
                score=score,
                sample_index=sample_index,
                config=config,
            )
            telemetry.append(sample_telemetry)
            implicit = _vjp(fz, parameters, v)
            for (name, parameter), direct_grad, implicit_grad in zip(
                named, direct, implicit, strict=True
            ):
                if direct_grad is None and implicit_grad is None:
                    continue
                total = torch.zeros_like(parameter)
                if direct_grad is not None:
                    total = total + direct_grad.detach()
                if implicit_grad is not None:
                    total = total + implicit_grad.detach()
                if not torch.isfinite(total).all():
                    raise FixedPointFisherError(
                        f"total gradient is non-finite for parameter {name!r}"
                    )
                if name not in accumulators:
                    required = parameter.numel() * _FP32_ELEMENT_BYTES
                    if active_bytes + required > config.max_fisher_bytes:
                        raise FixedPointFisherError("Fisher byte budget exceeded")
                    accumulators[name] = torch.zeros(
                        parameter.shape, device="cpu", dtype=torch.float32
                    )
                    active_bytes += required
                squared = total.float().cpu().square()
                accumulators[name].add_(squared)

    if not accumulators:
        raise FixedPointFisherError("no supplied parameter has a Fisher gradient")
    fisher = {
        name: value.div(float(batch)).contiguous()
        for name, value in accumulators.items()
    }
    return FixedPointFisherEstimate(
        fisher=fisher,
        n_samples=batch,
        fixed_point_residuals=residuals,
        contraction_bound=config.contraction_bound,
        adjoint=tuple(telemetry),
    )


def estimate_fixed_point_fisher(
    *,
    f_at_z,
    z_star: Tensor,
    log_likelihood_at_z,
    named_parameters: list[tuple[str, nn.Parameter]],
    tolerance: float = 1e-5,
    max_iter: int = 80,
    contraction_bound: float = 0.94,
) -> dict[str, Tensor]:
    """Deprecated batch-one wrapper around the verified empirical estimator.

    The legacy signature carries no contraction certificate.  Its default bound
    exists only for source compatibility and must not be used as production
    evidence; production callers must construct :class:`FixedPointFisherConfig`.
    """

    warnings.warn(
        "estimate_fixed_point_fisher is deprecated; use "
        "estimate_empirical_fixed_point_fisher with typed evidence",
        DeprecationWarning,
        stacklevel=2,
    )
    if z_star.ndim < 2 or z_star.shape[0] != 1:
        raise ValueError("legacy fixed-point Fisher wrapper supports batch size 1")

    def per_sample(state: Tensor) -> Tensor:
        value = log_likelihood_at_z(state)
        if not isinstance(value, Tensor):
            raise TypeError("log_likelihood_at_z must return a tensor")
        if value.ndim == 0:
            return value.reshape(1)
        return value

    result = estimate_empirical_fixed_point_fisher(
        f_at_z=f_at_z,
        z_star=z_star,
        log_likelihood_per_sample=per_sample,
        named_parameters=named_parameters,
        config=FixedPointFisherConfig(
            contraction_bound=contraction_bound,
            fixed_point_tolerance=tolerance,
            adjoint_tolerance=tolerance,
            max_adjoint_iterations=max_iter,
        ),
        solver_converged=True,
    )
    return dict(result.fisher)


@dataclass(slots=True)
class FisherSnapshot:
    """Sparse-by-name CPU-FP32 Fisher anchors and merge metadata."""

    fisher: dict[str, Tensor]
    anchor: dict[str, Tensor]
    n_samples: int = 1
    quadratic_offset: float = 0.0

    def __post_init__(self) -> None:
        if (
            not isinstance(self.n_samples, int)
            or isinstance(self.n_samples, bool)
            or self.n_samples <= 0
        ):
            raise ValueError("FisherSnapshot n_samples must be positive")
        if not math.isfinite(self.quadratic_offset) or self.quadratic_offset < 0.0:
            raise ValueError("FisherSnapshot quadratic_offset must be finite")

    @property
    def nbytes(self) -> int:
        tensor_bytes = sum(
            value.numel() * value.element_size()
            for mapping in (self.fisher, self.anchor)
            for value in mapping.values()
        )
        return tensor_bytes + 8


@dataclass
class FPEWCRegularizer:
    """Bounded, sample-weighted diagonal EWC snapshot lifecycle."""

    strength: float = 1.0
    snapshots: list[FisherSnapshot] = field(default_factory=list)
    max_domains: int = 16
    max_total_bytes: int = 64 * 1024 * 1024

    HARD_MAX_DOMAINS = 256
    HARD_MAX_TOTAL_BYTES = 1024 * 1024 * 1024

    def __post_init__(self) -> None:
        if not math.isfinite(float(self.strength)) or self.strength < 0.0:
            raise ValueError("strength must be finite and non-negative")
        if (
            not isinstance(self.max_domains, int)
            or isinstance(self.max_domains, bool)
            or not 1 <= self.max_domains <= self.HARD_MAX_DOMAINS
        ):
            raise ValueError("max_domains exceeds its hard bound")
        if (
            not isinstance(self.max_total_bytes, int)
            or isinstance(self.max_total_bytes, bool)
            or not 1 <= self.max_total_bytes <= self.HARD_MAX_TOTAL_BYTES
        ):
            raise ValueError("max_total_bytes exceeds its hard bound")

    @property
    def total_bytes(self) -> int:
        return sum(snapshot.nbytes for snapshot in self.snapshots)

    @staticmethod
    def _cpu_fp32(value: Tensor, *, non_negative: bool, label: str) -> Tensor:
        tensor = value.detach().to(device="cpu", dtype=torch.float32).clone()
        if not torch.isfinite(tensor).all():
            raise ValueError(f"{label} must be finite")
        if non_negative and (tensor < 0.0).any():
            raise ValueError(f"{label} must be non-negative")
        return tensor

    def consolidate(
        self,
        named_parameters: Iterable[tuple[str, nn.Parameter]],
        grads: Mapping[str, Tensor],
        *,
        n_samples: int = 1,
    ) -> FisherSnapshot:
        """Compatibility API: square score tensors then consolidate Fisher."""

        fisher = {
            name: gradient.detach().float().square() for name, gradient in grads.items()
        }
        return self.consolidate_fisher(
            named_parameters,
            fisher,
            n_samples=n_samples,
        )

    def consolidate_fisher(
        self,
        named_parameters: Iterable[tuple[str, nn.Parameter]],
        fisher: Mapping[str, Tensor],
        *,
        n_samples: int,
    ) -> FisherSnapshot:
        """Store only Fisher-bearing parameters as CPU-FP32 sparse mappings."""

        if (
            not isinstance(n_samples, int)
            or isinstance(n_samples, bool)
            or n_samples <= 0
        ):
            raise ValueError("n_samples must be a positive integer")
        parameters = dict(named_parameters)
        stored_fisher: dict[str, Tensor] = {}
        stored_anchor: dict[str, Tensor] = {}
        candidate_bytes = 8
        matching = [
            (name, value, parameters[name])
            for name, value in fisher.items()
            if name in parameters
        ]
        for name, value, parameter in matching:
            if tuple(value.shape) != tuple(parameter.shape):
                raise ValueError(f"Fisher shape mismatch for parameter {name!r}")
            candidate_bytes += parameter.numel() * 2 * _FP32_ELEMENT_BYTES
        if candidate_bytes > self.max_total_bytes:
            raise FixedPointFisherError("single Fisher snapshot exceeds byte budget")
        for name, value, parameter in matching:
            stored_fisher[name] = self._cpu_fp32(
                value, non_negative=True, label=f"Fisher {name!r}"
            )
            stored_anchor[name] = self._cpu_fp32(
                parameter, non_negative=False, label=f"anchor {name!r}"
            )
        if not stored_fisher:
            raise ValueError("no Fisher entries matched the supplied parameters")
        snapshot = FisherSnapshot(stored_fisher, stored_anchor, n_samples=n_samples)
        if snapshot.nbytes > self.max_total_bytes:
            raise FixedPointFisherError("single Fisher snapshot exceeds byte budget")

        previous = list(self.snapshots)
        self.snapshots.append(snapshot)
        try:
            self._enforce_budgets()
        except BaseException:
            self.snapshots = previous
            raise
        return snapshot

    @staticmethod
    def _merge(first: FisherSnapshot, second: FisherSnapshot) -> FisherSnapshot:
        total_samples = first.n_samples + second.n_samples
        fisher: dict[str, Tensor] = {}
        anchor: dict[str, Tensor] = {}
        completion_offset = 0.0
        for name in first.fisher.keys() | second.fisher.keys():
            f1 = first.fisher.get(name)
            f2 = second.fisher.get(name)
            a1 = first.anchor.get(name)
            a2 = second.anchor.get(name)
            template = f1 if f1 is not None else f2
            if template is None:
                raise RuntimeError("merged Fisher entry has no tensor")
            zeros = torch.zeros_like(template)
            precision1 = zeros if f1 is None else f1 * float(first.n_samples)
            precision2 = zeros if f2 is None else f2 * float(second.n_samples)
            if a1 is None:
                a1 = torch.zeros_like(template)
            if a2 is None:
                a2 = torch.zeros_like(template)
            precision = precision1 + precision2
            safe = precision.clamp_min(torch.finfo(precision.dtype).eps)
            merged_anchor = (precision1 * a1 + precision2 * a2) / safe
            merged_anchor = torch.where(
                precision > 0.0, merged_anchor, torch.zeros_like(merged_anchor)
            )
            fisher[name] = precision / float(total_samples)
            anchor[name] = merged_anchor
            constant = (
                precision1 * a1.square()
                + precision2 * a2.square()
                - precision * merged_anchor.square()
            )
            completion_offset += 0.5 * float(constant.sum().clamp_min(0.0))
        return FisherSnapshot(
            fisher=fisher,
            anchor=anchor,
            n_samples=total_samples,
            quadratic_offset=(
                first.quadratic_offset + second.quadratic_offset + completion_offset
            ),
        )

    def merge_oldest(self) -> FisherSnapshot:
        if len(self.snapshots) < 2:
            raise RuntimeError("at least two Fisher snapshots are required to merge")
        merged = self._merge(self.snapshots[0], self.snapshots[1])
        self.snapshots[:2] = [merged]
        return merged

    def _enforce_budgets(self) -> None:
        while len(self.snapshots) > 1 and (
            len(self.snapshots) > self.max_domains
            or self.total_bytes > self.max_total_bytes
        ):
            self.merge_oldest()
        if len(self.snapshots) > self.max_domains:
            raise FixedPointFisherError("Fisher domain budget exceeded")
        if self.total_bytes > self.max_total_bytes:
            raise FixedPointFisherError("Fisher byte budget exceeded")

    def penalty(self, named_parameters: Iterable[tuple[str, nn.Parameter]]) -> Tensor:
        params = list(named_parameters)
        if not params:
            return torch.zeros(())
        total = params[0][1].new_zeros(())
        lookup = dict(params)
        for snapshot in self.snapshots:
            matched = False
            for name, fisher in snapshot.fisher.items():
                parameter = lookup.get(name)
                if parameter is None:
                    continue
                matched = True
                local_fisher = fisher.to(parameter)
                local_anchor = snapshot.anchor[name].to(parameter)
                total = (
                    total
                    + 0.5
                    * self.strength
                    * float(snapshot.n_samples)
                    * (local_fisher * (parameter - local_anchor).square()).sum()
                )
            if matched and snapshot.quadratic_offset:
                total = total + total.new_tensor(
                    self.strength * snapshot.quadratic_offset
                )
        return total


__all__ = [
    "AdjointConvergenceError",
    "AdjointTelemetry",
    "FPEWCRegularizer",
    "FisherSnapshot",
    "FixedPointFisherConfig",
    "FixedPointFisherError",
    "FixedPointFisherEstimate",
    "estimate_empirical_fixed_point_fisher",
    "estimate_fixed_point_fisher",
    "spectral_guard_",
    "verified_spectral_cap_",
]
