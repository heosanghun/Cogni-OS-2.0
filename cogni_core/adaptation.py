"""Memory-bounded fast adaptation and fixed-point consolidation.

This module joins the already verified System 1.5 and System 2.5 primitives
without making either one own the base model.  Fast weights are installed as
temporary ``nn.Linear`` forward hooks; a session therefore never writes to a
base parameter and cleanup is exception safe.  Long-term adaptation delegates
the matrix-free implicit Fisher calculation to :mod:`cogni_core.fp_ewc`.
"""

from __future__ import annotations

from collections import OrderedDict
from contextlib import contextmanager
from dataclasses import dataclass
from threading import RLock
from typing import Callable, Iterable, Iterator, Mapping

import torch
from torch import Tensor, nn

from .fp_ewc import (
    FPEWCRegularizer,
    FisherSnapshot,
    estimate_fixed_point_fisher,
)


@dataclass(frozen=True)
class LowRankOverlay:
    """A linear-layer update ``delta_W = B @ A.T``.

    ``a`` has shape ``[in_features, rank]`` and ``b`` has shape
    ``[out_features, rank]``.  Keeping the factors instead of materialising
    ``delta_W`` makes both session storage and application O(d * rank).
    """

    a: Tensor
    b: Tensor

    def __post_init__(self) -> None:
        if self.a.ndim != 2 or self.b.ndim != 2:
            raise ValueError("low-rank factors must both be rank-2 tensors")
        if self.a.shape[1] != self.b.shape[1]:
            raise ValueError("a and b must have the same low-rank dimension")
        if self.a.device != self.b.device:
            raise ValueError("a and b must be on the same device")
        if self.a.dtype != self.b.dtype:
            raise ValueError("a and b must have the same dtype")
        if not torch.is_floating_point(self.a) or not torch.is_floating_point(self.b):
            raise TypeError("low-rank factors must be floating-point tensors")

    @property
    def rank(self) -> int:
        return self.a.shape[1]

    @property
    def nbytes(self) -> int:
        return (self.a.numel() * self.a.element_size()) + (
            self.b.numel() * self.b.element_size()
        )

    def detached_clone(self, device: torch.device | str = "cpu") -> "LowRankOverlay":
        return LowRankOverlay(
            self.a.detach().to(device=device).clone(),
            self.b.detach().to(device=device).clone(),
        )


def low_rank_operator_norm(overlay: LowRankOverlay) -> Tensor:
    """Return ``||B A^T||_2`` without forming the dense update.

    Reduced QR factorizations yield ``B A^T = Q_b (R_b R_a^T) Q_a^T``.
    The orthonormal factors do not change non-zero singular values, so only a
    ``rank x rank`` matrix reaches the spectral-norm calculation.
    """

    with torch.no_grad():
        # QR also behaves correctly when rank exceeds a feature dimension; the
        # middle product then has min(d, r) dimensions and remains small.
        # Norm certification uses FP32 so CPU/CUDA BF16 sessions share the
        # same supported and conservative admission path.
        _, r_a = torch.linalg.qr(overlay.a.float(), mode="reduced")
        _, r_b = torch.linalg.qr(overlay.b.float(), mode="reduced")
        return torch.linalg.matrix_norm(r_b @ r_a.transpose(-1, -2), ord=2)


@dataclass(frozen=True)
class AcceptanceDecision:
    accepted: bool
    quality: float
    max_operator_norm: float
    reason: str = "accepted"
    max_composed_operator_norm: float | None = None


@dataclass(frozen=True)
class OverlayAcceptanceGate:
    """Admission gate for a generated fast-weight session."""

    min_quality: float = 0.5
    operator_norm_budget: float = 0.1
    composed_operator_norm_budget: float | None = None
    atol: float = 1e-6

    def __post_init__(self) -> None:
        if not 0.0 <= self.min_quality <= 1.0:
            raise ValueError("min_quality must be in [0, 1]")
        if self.operator_norm_budget <= 0.0:
            raise ValueError("operator_norm_budget must be positive")
        if (
            self.composed_operator_norm_budget is not None
            and self.composed_operator_norm_budget <= 0.0
        ):
            raise ValueError("composed_operator_norm_budget must be positive")

    def evaluate(
        self,
        overlays: Mapping[str, LowRankOverlay],
        quality: float | Tensor,
        *,
        base_weights: Mapping[str, Tensor] | None = None,
    ) -> AcceptanceDecision:
        if not overlays:
            return AcceptanceDecision(
                False, float(torch.as_tensor(quality)), 0.0, "empty overlay set"
            )
        quality_value = (
            float(quality.detach().mean())
            if isinstance(quality, Tensor)
            else float(quality)
        )
        if not torch.isfinite(torch.as_tensor(quality_value)):
            return AcceptanceDecision(
                False, quality_value, float("inf"), "non-finite quality"
            )
        if quality_value < self.min_quality:
            return AcceptanceDecision(
                False, quality_value, 0.0, "quality below threshold"
            )

        maximum = 0.0
        maximum_composed: float | None = None
        for name, overlay in overlays.items():
            if (
                not torch.isfinite(overlay.a).all()
                or not torch.isfinite(overlay.b).all()
            ):
                return AcceptanceDecision(
                    False, quality_value, float("inf"), f"non-finite factors: {name}"
                )
            sigma = float(low_rank_operator_norm(overlay))
            maximum = max(maximum, sigma)
            if sigma > self.operator_norm_budget + self.atol:
                return AcceptanceDecision(
                    False, quality_value, maximum, f"operator budget exceeded: {name}"
                )
            if self.composed_operator_norm_budget is not None:
                if base_weights is None or name not in base_weights:
                    return AcceptanceDecision(
                        False,
                        quality_value,
                        maximum,
                        f"base weight required for composed norm: {name}",
                    )
                base = base_weights[name].detach()
                if base.ndim != 2 or not torch.isfinite(base).all():
                    return AcceptanceDecision(
                        False,
                        quality_value,
                        maximum,
                        f"invalid base weight for composed norm: {name}",
                    )
                # Triangle inequality is a valid, allocation-bounded upper
                # bound on ||W + B A^T||_2.  It intentionally avoids forming a
                # dense delta matrix merely to admit a temporary session.
                base_sigma = float(torch.linalg.matrix_norm(base.float(), ord=2))
                composed_bound = base_sigma + sigma
                maximum_composed = max(maximum_composed or 0.0, composed_bound)
                if composed_bound > self.composed_operator_norm_budget + self.atol:
                    return AcceptanceDecision(
                        False,
                        quality_value,
                        maximum,
                        f"composed operator budget exceeded: {name}",
                        maximum_composed,
                    )
        return AcceptanceDecision(
            True,
            quality_value,
            maximum,
            max_composed_operator_norm=maximum_composed,
        )


@dataclass(frozen=True)
class SessionRecord:
    session_id: str
    overlays: Mapping[str, LowRankOverlay]
    quality: float
    nbytes: int


@dataclass(frozen=True)
class AdmissionResult:
    decision: AcceptanceDecision
    evicted: tuple[str, ...] = ()

    @property
    def accepted(self) -> bool:
        return self.decision.accepted


class FastWeightSessionCache:
    """LRU cache and exception-safe activation for low-rank sessions.

    Sessions are detached and stored on ``storage_device`` (CPU by default),
    bounding persistent VRAM use.  The cache is bounded by both session count
    and bytes.  Activating a session installs additive forward hooks and removes
    them in ``finally``; base weights are not copied, patched, or reassigned.

    Forward hooks are process-global to each module, so callers must serialize
    different active sessions.  The day/night state machine already imposes
    that execution model; this class detects accidental overlapping activation.
    """

    def __init__(
        self,
        model: nn.Module,
        *,
        gate: OverlayAcceptanceGate | None = None,
        max_sessions: int = 8,
        max_bytes: int = 64 * 1024 * 1024,
        storage_device: torch.device | str = "cpu",
    ) -> None:
        if max_sessions <= 0:
            raise ValueError("max_sessions must be positive")
        if max_bytes <= 0:
            raise ValueError("max_bytes must be positive")
        self.model = model
        self.gate = gate or OverlayAcceptanceGate()
        self.max_sessions = max_sessions
        self.max_bytes = max_bytes
        self.storage_device = torch.device(storage_device)
        self._sessions: OrderedDict[str, SessionRecord] = OrderedDict()
        self._bytes = 0
        self._active_session: str | None = None
        self._lock = RLock()

    @property
    def total_bytes(self) -> int:
        return self._bytes

    @property
    def session_ids(self) -> tuple[str, ...]:
        """Session IDs in LRU-to-MRU order."""
        return tuple(self._sessions)

    def __len__(self) -> int:
        return len(self._sessions)

    def admit(
        self,
        session_id: str,
        overlays: Mapping[str, LowRankOverlay],
        *,
        quality: float | Tensor,
    ) -> AdmissionResult:
        """Validate and cache one session, evicting least-recently-used entries."""

        if not session_id:
            raise ValueError("session_id must be non-empty")
        modules = dict(self.model.named_modules())
        base_weights: dict[str, Tensor] = {}
        stored: dict[str, LowRankOverlay] = {}
        for name, overlay in overlays.items():
            module = modules.get(name)
            if not isinstance(module, nn.Linear):
                raise ValueError(f"overlay target is not an nn.Linear: {name!r}")
            if (
                overlay.a.shape[0] != module.in_features
                or overlay.b.shape[0] != module.out_features
            ):
                raise ValueError(
                    f"overlay shape for {name!r} is incompatible with "
                    f"Linear({module.in_features}, {module.out_features})"
                )
            base_weights[name] = module.weight
            stored[name] = overlay.detached_clone(self.storage_device)

        decision = self.gate.evaluate(overlays, quality, base_weights=base_weights)
        if not decision.accepted:
            return AdmissionResult(decision)

        size = sum(overlay.nbytes for overlay in stored.values())
        if size > self.max_bytes:
            rejected = AcceptanceDecision(
                False,
                decision.quality,
                decision.max_operator_norm,
                "session exceeds byte budget",
            )
            return AdmissionResult(rejected)

        evicted: list[str] = []
        with self._lock:
            previous = self._sessions.pop(session_id, None)
            if previous is not None:
                self._bytes -= previous.nbytes
            while self._sessions and (
                len(self._sessions) >= self.max_sessions
                or self._bytes + size > self.max_bytes
            ):
                old_id, old = self._sessions.popitem(last=False)
                self._bytes -= old.nbytes
                evicted.append(old_id)
            record = SessionRecord(session_id, stored, decision.quality, size)
            self._sessions[session_id] = record
            self._bytes += size
        return AdmissionResult(decision, tuple(evicted))

    def discard(self, session_id: str) -> bool:
        with self._lock:
            if self._active_session == session_id:
                raise RuntimeError("cannot discard an active fast-weight session")
            record = self._sessions.pop(session_id, None)
            if record is None:
                return False
            self._bytes -= record.nbytes
            return True

    def get(self, session_id: str) -> SessionRecord:
        with self._lock:
            try:
                record = self._sessions.pop(session_id)
            except KeyError as exc:
                raise KeyError(f"unknown fast-weight session: {session_id!r}") from exc
            self._sessions[session_id] = record
            return record

    @staticmethod
    def _hook_for(
        overlay: LowRankOverlay,
    ) -> Callable[[nn.Module, tuple[Tensor, ...], Tensor], Tensor]:
        def add_overlay(
            module: nn.Module, inputs: tuple[Tensor, ...], output: Tensor
        ) -> Tensor:
            if not inputs:
                raise RuntimeError("linear overlay target received no positional input")
            x = inputs[0]
            # Transfers are scoped to the active call. Persistent cache storage
            # remains on CPU and no dense d-by-d delta is ever materialised.
            a = overlay.a.to(device=x.device, dtype=x.dtype, non_blocking=True)
            b = overlay.b.to(device=x.device, dtype=x.dtype, non_blocking=True)
            update = torch.nn.functional.linear(torch.nn.functional.linear(x, a.T), b)
            return output + update.to(dtype=output.dtype)

        return add_overlay

    @contextmanager
    def activate(self, session_id: str) -> Iterator[SessionRecord]:
        """Temporarily apply a cached overlay and always remove it on exit."""

        with self._lock:
            if self._active_session is not None:
                raise RuntimeError(
                    f"fast-weight session {self._active_session!r} is already active"
                )
            record = self.get(session_id)
            modules = dict(self.model.named_modules())
            handles = [
                modules[name].register_forward_hook(self._hook_for(overlay))
                for name, overlay in record.overlays.items()
            ]
            self._active_session = session_id
        try:
            yield record
        finally:
            # Remove in reverse registration order in case future hooks depend
            # on ordering. RemovableHandle.remove() is idempotent.
            for handle in reversed(handles):
                handle.remove()
            with self._lock:
                self._active_session = None


@dataclass(frozen=True)
class DomainRecord:
    domain_id: str
    n_samples: int


@dataclass(frozen=True)
class ProjectionReport:
    before_step: Mapping[str, float]
    after_step: Mapping[str, float]


class FixedPointDomainLifecycle:
    """Domain boundaries for matrix-free FP-EWC and safe optimizer steps."""

    def __init__(
        self,
        *,
        strength: float = 1.0,
        spectral_margin: float = 0.95,
        max_domains: int | None = None,
    ) -> None:
        if strength < 0.0:
            raise ValueError("strength must be non-negative")
        if not 0.0 < spectral_margin < 1.0:
            raise ValueError("spectral_margin must lie in (0, 1)")
        if max_domains is not None and max_domains <= 0:
            raise ValueError("max_domains must be positive when specified")
        self.regularizer = FPEWCRegularizer(strength=strength)
        self.spectral_margin = spectral_margin
        self.max_domains = max_domains
        self.domains: list[DomainRecord] = []

    @property
    def n_consolidated(self) -> int:
        return len(self.regularizer.snapshots)

    def consolidate(
        self,
        domain_id: str,
        named_parameters: Iterable[tuple[str, nn.Parameter]],
        diagonal_fisher: Mapping[str, Tensor],
        *,
        n_samples: int = 1,
    ) -> FisherSnapshot:
        """Anchor a domain using an already squared diagonal Fisher.

        The existing ``FPEWCRegularizer.consolidate`` accepts score-like
        tensors and squares them.  Taking the square root here preserves the
        estimator's diagonal-Fisher values rather than accidentally raising
        them to the fourth power.
        """

        if not domain_id:
            raise ValueError("domain_id must be non-empty")
        if n_samples <= 0:
            raise ValueError("n_samples must be positive")
        params = list(named_parameters)
        lookup = dict(params)
        scores: dict[str, Tensor] = {}
        for name, fisher in diagonal_fisher.items():
            if name not in lookup:
                continue
            if fisher.shape != lookup[name].shape:
                raise ValueError(f"Fisher shape mismatch for parameter {name!r}")
            if not torch.isfinite(fisher).all() or (fisher < 0).any():
                raise ValueError(
                    f"Fisher must be finite and non-negative for parameter {name!r}"
                )
            scores[name] = fisher.detach().clamp_min(0).sqrt()
        if not scores:
            raise ValueError("no Fisher entries matched the supplied parameters")

        self.regularizer.consolidate(params, scores)
        snapshot = self.regularizer.snapshots[-1]
        self.domains.append(DomainRecord(domain_id, n_samples))
        self._enforce_domain_budget()
        return snapshot

    def estimate_and_consolidate(
        self,
        domain_id: str,
        *,
        f_at_z: Callable[[Tensor], Tensor],
        z_star: Tensor,
        log_likelihood_at_z: Callable[[Tensor], Tensor],
        named_parameters: Iterable[tuple[str, nn.Parameter]],
        tolerance: float = 1e-5,
        max_iter: int = 80,
        n_samples: int = 1,
    ) -> FisherSnapshot:
        """Estimate FP-Fisher via VJPs only, then anchor the domain."""

        params = list(named_parameters)
        fisher = estimate_fixed_point_fisher(
            f_at_z=f_at_z,
            z_star=z_star,
            log_likelihood_at_z=log_likelihood_at_z,
            named_parameters=params,
            tolerance=tolerance,
            max_iter=max_iter,
        )
        return self.consolidate(domain_id, params, fisher, n_samples=n_samples)

    def penalty(self, named_parameters: Iterable[tuple[str, nn.Parameter]]) -> Tensor:
        return self.regularizer.penalty(named_parameters)

    def _enforce_domain_budget(self) -> None:
        """Merge oldest diagonal quadratics to keep snapshot memory fixed.

        For diagonal quadratics, F1(theta-a1)^2 + F2(theta-a2)^2 has the
        same gradient as (F1+F2)(theta-a_bar)^2, where a_bar is the
        Fisher-weighted mean.  The dropped term is constant in theta.
        """

        if self.max_domains is None:
            return
        while len(self.regularizer.snapshots) > self.max_domains:
            first, second = self.regularizer.snapshots[:2]
            fisher: dict[str, Tensor] = {}
            anchor: dict[str, Tensor] = {}
            for name in first.fisher.keys() | second.fisher.keys():
                f1 = first.fisher.get(name)
                f2 = second.fisher.get(name)
                a1 = first.anchor.get(name)
                a2 = second.anchor.get(name)
                if f1 is None or a1 is None:
                    fisher[name], anchor[name] = f2.clone(), a2.clone()  # type: ignore[union-attr]
                elif f2 is None or a2 is None:
                    fisher[name], anchor[name] = f1.clone(), a1.clone()
                else:
                    total = f1 + f2
                    safe = total.clamp_min(torch.finfo(total.dtype).eps)
                    fisher[name] = total
                    anchor[name] = (f1 * a1 + f2 * a2) / safe
            merged = FisherSnapshot(fisher=fisher, anchor=anchor)
            self.regularizer.snapshots[:2] = [merged]
            first_domain, second_domain = self.domains[:2]
            self.domains[:2] = [
                DomainRecord(
                    f"{first_domain.domain_id}+{second_domain.domain_id}",
                    first_domain.n_samples + second_domain.n_samples,
                )
            ]

    def project_spectral_(
        self,
        named_parameters: Iterable[tuple[str, nn.Parameter]],
    ) -> dict[str, float]:
        """Project every matrix slice below the configured DEQ margin.

        Rank-2 parameters are treated as one matrix.  For rank-3 and higher
        parameters, every matrix in the leading batch dimensions is projected
        independently over the final two dimensions.
        """

        report: dict[str, float] = {}
        for name, parameter in named_parameters:
            if parameter.ndim < 2:
                continue
            with torch.no_grad():
                norms = torch.linalg.matrix_norm(parameter, ord=2)
                if not torch.isfinite(norms).all():
                    raise RuntimeError(
                        f"non-finite spectral norm for parameter {name!r}"
                    )
                # A small interior guard avoids rounding back above the strict
                # Banach margin after multiplication/SVD recomputation.
                target = parameter.new_tensor(self.spectral_margin * (1.0 - 1e-5))
                scales = torch.clamp(target / norms.clamp_min(1e-12), max=1.0)
                parameter.mul_(scales[..., None, None])
                projected = torch.linalg.matrix_norm(parameter, ord=2)
                if not torch.isfinite(projected).all() or bool(
                    (projected >= self.spectral_margin).any().detach().cpu()
                ):
                    raise RuntimeError(
                        f"spectral projection postcondition failed for {name!r}"
                    )
                report[name] = float(projected.max().detach())
        return report

    def optimizer_step(
        self,
        optimizer: torch.optim.Optimizer,
        spectral_parameters: Iterable[tuple[str, nn.Parameter]],
        *,
        closure: Callable[[], float | Tensor] | None = None,
    ) -> tuple[float | Tensor | None, ProjectionReport]:
        """Project both before and after ``optimizer.step``.

        The pre-step projection is the required C-FIRE safety boundary.  The
        mandatory post-step projection guarantees that parameters handed to
        the next forward call remain contractive after a large optimizer
        update.  There is deliberately no unsafe switch to disable it.
        """

        params = list(spectral_parameters)
        before = self.project_spectral_(params)
        result = (
            optimizer.step(closure=closure) if closure is not None else optimizer.step()
        )
        after = self.project_spectral_(params)
        return result, ProjectionReport(before, after)


__all__ = [
    "AcceptanceDecision",
    "AdmissionResult",
    "DomainRecord",
    "FastWeightSessionCache",
    "FixedPointDomainLifecycle",
    "LowRankOverlay",
    "OverlayAcceptanceGate",
    "ProjectionReport",
    "SessionRecord",
    "low_rank_operator_norm",
]
