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
import math
from threading import RLock
from time import monotonic
from typing import Callable, Iterable, Iterator, Mapping

import torch
from torch import Tensor, nn

from .fp_ewc import (
    FPEWCRegularizer,
    FisherSnapshot,
    FixedPointFisherConfig,
    FixedPointFisherEstimate,
    estimate_empirical_fixed_point_fisher,
    estimate_fixed_point_fisher,
)
from .fast_weight_safety import VerifiedFastWeightAdmission


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
    expires_at: float
    authorization: VerifiedFastWeightAdmission | None


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
    bounding persistent VRAM use.  The cache is bounded by session count, bytes,
    overlay structures per session, and an absolute (non-sliding) TTL.
    Activating a session installs additive forward hooks transactionally and
    removes them in ``finally``; base weights are not copied, patched, or
    reassigned.

    Forward hooks are process-global to each module, so callers must serialize
    different active sessions.  The day/night state machine already imposes
    that execution model; this class detects accidental overlapping activation.

    ``on_sessions_removed`` is an optional bounded control-plane callback for
    keeping a separate OOD router in sync.  It receives one tuple of removed
    identifiers after each atomic cache mutation.  The cache is already in its
    fail-closed state if that callback raises.
    """

    DEFAULT_SESSION_TTL_SECONDS = 15.0 * 60.0
    MAX_SESSION_TTL_SECONDS = 24.0 * 60.0 * 60.0
    DEFAULT_MAX_OVERLAYS_PER_SESSION = 8
    HARD_MAX_OVERLAYS_PER_SESSION = 64

    def __init__(
        self,
        model: nn.Module,
        *,
        gate: OverlayAcceptanceGate | None = None,
        max_sessions: int = 8,
        max_bytes: int = 64 * 1024 * 1024,
        session_ttl_seconds: float = DEFAULT_SESSION_TTL_SECONDS,
        max_overlays_per_session: int = DEFAULT_MAX_OVERLAYS_PER_SESSION,
        storage_device: torch.device | str = "cpu",
        clock: Callable[[], float] = monotonic,
        on_sessions_removed: Callable[[tuple[str, ...]], None] | None = None,
        feature_enabled: bool = True,
        trusted_programmer_sha256: str | None = None,
        allow_unverified_research: bool = False,
    ) -> None:
        if (
            not isinstance(max_sessions, int)
            or isinstance(max_sessions, bool)
            or max_sessions <= 0
        ):
            raise ValueError("max_sessions must be positive")
        if (
            not isinstance(max_bytes, int)
            or isinstance(max_bytes, bool)
            or max_bytes <= 0
        ):
            raise ValueError("max_bytes must be positive")
        ttl = float(session_ttl_seconds)
        if not math.isfinite(ttl) or not 0.0 < ttl <= self.MAX_SESSION_TTL_SECONDS:
            raise ValueError(
                "session_ttl_seconds must be finite and in "
                f"(0, {self.MAX_SESSION_TTL_SECONDS}]"
            )
        if (
            not isinstance(max_overlays_per_session, int)
            or isinstance(max_overlays_per_session, bool)
            or not 1 <= max_overlays_per_session <= self.HARD_MAX_OVERLAYS_PER_SESSION
        ):
            raise ValueError(
                "max_overlays_per_session must be in "
                f"[1, {self.HARD_MAX_OVERLAYS_PER_SESSION}]"
            )
        if not callable(clock):
            raise TypeError("clock must be callable")
        if on_sessions_removed is not None and not callable(on_sessions_removed):
            raise TypeError("on_sessions_removed must be callable or None")
        if not isinstance(feature_enabled, bool):
            raise TypeError("feature_enabled must be bool")
        if not isinstance(allow_unverified_research, bool):
            raise TypeError("allow_unverified_research must be bool")
        if trusted_programmer_sha256 is not None:
            if (
                not isinstance(trusted_programmer_sha256, str)
                or len(trusted_programmer_sha256) != 64
                or any(
                    character not in "0123456789abcdef"
                    for character in trusted_programmer_sha256
                )
            ):
                raise ValueError(
                    "trusted_programmer_sha256 must be a lowercase SHA-256 digest"
                )
        self.model = model
        self.gate = gate or OverlayAcceptanceGate()
        self.max_sessions = max_sessions
        self.max_bytes = max_bytes
        self.session_ttl_seconds = ttl
        self.max_overlays_per_session = max_overlays_per_session
        self.storage_device = torch.device(storage_device)
        self._clock = clock
        self._on_sessions_removed = on_sessions_removed
        self._feature_enabled = feature_enabled
        self._trusted_programmer_sha256 = trusted_programmer_sha256
        self._allow_unverified_research = allow_unverified_research
        self._sessions: OrderedDict[str, SessionRecord] = OrderedDict()
        self._bytes = 0
        self._active_session: str | None = None
        self._lock = RLock()
        self._last_clock = self._read_clock()

    def _read_clock(self) -> float:
        value = float(self._clock())
        if not math.isfinite(value):
            raise RuntimeError("session clock must return a finite value")
        return value

    def _now_locked(self) -> float:
        value = self._read_clock()
        if value < self._last_clock:
            raise RuntimeError("session clock moved backwards")
        self._last_clock = value
        return value

    @staticmethod
    def _quality_value(quality: float | Tensor) -> float:
        if isinstance(quality, Tensor):
            if quality.numel() == 0:
                return float("nan")
            return float(quality.detach().float().mean())
        return float(quality)

    def _rejection(self, quality: float | Tensor, reason: str) -> AdmissionResult:
        return AdmissionResult(
            AcceptanceDecision(False, self._quality_value(quality), 0.0, reason)
        )

    def _assert_invariants_locked(self) -> None:
        if not 0 <= len(self._sessions) <= self.max_sessions:
            raise RuntimeError("fast-weight session-count invariant failed")
        measured = sum(record.nbytes for record in self._sessions.values())
        if measured != self._bytes or not 0 <= self._bytes <= self.max_bytes:
            raise RuntimeError("fast-weight byte-accounting invariant failed")
        if (
            self._active_session is not None
            and self._active_session not in self._sessions
        ):
            raise RuntimeError("active fast-weight session is missing from the cache")
        for record in self._sessions.values():
            if not math.isfinite(record.expires_at):
                raise RuntimeError("fast-weight session expiry must be finite")
            if len(record.overlays) > self.max_overlays_per_session:
                raise RuntimeError("fast-weight structure-count invariant failed")
            if not self._allow_unverified_research and (
                self._trusted_programmer_sha256 is None
                or record.authorization is None
                or not record.authorization.valid_for(self._trusted_programmer_sha256)
            ):
                raise RuntimeError("fast-weight session authorization invariant failed")

    def _notify_removed_locked(self, removed: tuple[str, ...]) -> None:
        if removed and self._on_sessions_removed is not None:
            self._on_sessions_removed(removed)

    def _purge_expired_locked(self, now: float | None = None) -> tuple[str, ...]:
        timestamp = self._now_locked() if now is None else now
        removed: list[str] = []
        for session_id, record in tuple(self._sessions.items()):
            if session_id == self._active_session or record.expires_at > timestamp:
                continue
            self._sessions.pop(session_id)
            self._bytes -= record.nbytes
            removed.append(session_id)
        result = tuple(removed)
        self._assert_invariants_locked()
        self._notify_removed_locked(result)
        return result

    def _clear_locked(self) -> tuple[str, ...]:
        removed = tuple(self._sessions)
        self._sessions.clear()
        self._bytes = 0
        self._assert_invariants_locked()
        self._notify_removed_locked(removed)
        return removed

    @property
    def feature_enabled(self) -> bool:
        with self._lock:
            return self._feature_enabled

    @property
    def total_bytes(self) -> int:
        with self._lock:
            self._purge_expired_locked()
            return self._bytes

    @property
    def session_ids(self) -> tuple[str, ...]:
        """Session IDs in LRU-to-MRU order."""
        with self._lock:
            self._purge_expired_locked()
            return tuple(self._sessions)

    def __len__(self) -> int:
        with self._lock:
            self._purge_expired_locked()
            return len(self._sessions)

    def purge_expired(self) -> tuple[str, ...]:
        """Remove every expired inactive session in LRU order."""

        with self._lock:
            return self._purge_expired_locked()

    def flush(self) -> tuple[str, ...]:
        """Atomically discard all sessions while keeping the feature state."""

        with self._lock:
            if self._active_session is not None:
                raise RuntimeError("cannot flush while a fast-weight session is active")
            return self._clear_locked()

    def feature_off(self) -> tuple[str, ...]:
        """Fail closed and atomically discard all temporary overlays."""

        with self._lock:
            if self._active_session is not None:
                raise RuntimeError(
                    "cannot disable Fast Weight while a session is active"
                )
            self._feature_enabled = False
            removed = self._clear_locked()
            self._assert_invariants_locked()
            return removed

    def feature_on(self) -> None:
        """Re-enable admission after an explicit control-plane decision."""

        with self._lock:
            if self._active_session is not None:
                raise RuntimeError("cannot enable Fast Weight during activation")
            if (
                self._trusted_programmer_sha256 is None
                and not self._allow_unverified_research
            ):
                raise RuntimeError(
                    "cannot enable Fast Weight without a verified programmer checkpoint"
                )
            self._feature_enabled = True
            self._assert_invariants_locked()

    def admit(
        self,
        session_id: str,
        overlays: Mapping[str, LowRankOverlay],
        *,
        quality: float | Tensor,
        authorization: VerifiedFastWeightAdmission | None = None,
    ) -> AdmissionResult:
        """Validate and cache one session, evicting least-recently-used entries."""

        if not session_id:
            raise ValueError("session_id must be non-empty")
        with self._lock:
            if not self._feature_enabled:
                return self._rejection(quality, "fast-weight feature disabled")
            if (
                not self._allow_unverified_research
                and self._trusted_programmer_sha256 is None
            ):
                return self._rejection(
                    quality, "verified programmer checkpoint missing"
                )
            if not self._allow_unverified_research and (
                not isinstance(authorization, VerifiedFastWeightAdmission)
                or not authorization.valid_for(self._trusted_programmer_sha256)
            ):
                return self._rejection(
                    quality, "verified AQ authorization missing or invalid"
                )
            if self._active_session is not None:
                raise RuntimeError("cannot admit while a fast-weight session is active")
            self._purge_expired_locked()
        if len(overlays) > self.max_overlays_per_session:
            return self._rejection(quality, "overlay structure budget exceeded")
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
        removed_during_admission: list[str] = []
        with self._lock:
            if not self._feature_enabled:
                return self._rejection(quality, "fast-weight feature disabled")
            if not self._allow_unverified_research and (
                self._trusted_programmer_sha256 is None
                or authorization is None
                or not authorization.valid_for(self._trusted_programmer_sha256)
            ):
                return self._rejection(
                    quality, "verified AQ authorization missing or invalid"
                )
            if self._active_session is not None:
                raise RuntimeError("cannot admit while a fast-weight session is active")
            expired = self._purge_expired_locked()
            evicted.extend(expired)
            now = self._now_locked()
            expires_at = now + self.session_ttl_seconds
            if not math.isfinite(expires_at):
                raise RuntimeError("fast-weight session expiry overflowed")
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
                removed_during_admission.append(old_id)
            record = SessionRecord(
                session_id,
                stored,
                decision.quality,
                size,
                expires_at,
                authorization,
            )
            self._sessions[session_id] = record
            self._bytes += size
            self._assert_invariants_locked()
            self._notify_removed_locked(tuple(removed_during_admission))
        return AdmissionResult(decision, tuple(evicted))

    def discard(self, session_id: str) -> bool:
        with self._lock:
            self._purge_expired_locked()
            if self._active_session == session_id:
                raise RuntimeError("cannot discard an active fast-weight session")
            record = self._sessions.pop(session_id, None)
            if record is None:
                return False
            self._bytes -= record.nbytes
            self._assert_invariants_locked()
            self._notify_removed_locked((session_id,))
            return True

    def get(self, session_id: str) -> SessionRecord:
        with self._lock:
            if not self._feature_enabled:
                raise RuntimeError("Fast Weight feature is disabled")
            self._purge_expired_locked()
            try:
                record = self._sessions.pop(session_id)
            except KeyError as exc:
                raise KeyError(f"unknown fast-weight session: {session_id!r}") from exc
            self._sessions[session_id] = record
            if not self._allow_unverified_research and (
                self._trusted_programmer_sha256 is None
                or record.authorization is None
                or not record.authorization.valid_for(self._trusted_programmer_sha256)
            ):
                self._sessions.pop(session_id, None)
                self._bytes -= record.nbytes
                self._assert_invariants_locked()
                self._notify_removed_locked((session_id,))
                raise RuntimeError(
                    "Fast Weight session authorization is no longer valid"
                )
            self._assert_invariants_locked()
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

        handles = []
        with self._lock:
            if not self._feature_enabled:
                raise RuntimeError("Fast Weight feature is disabled")
            self._purge_expired_locked()
            if self._active_session is not None:
                raise RuntimeError(
                    f"fast-weight session {self._active_session!r} is already active"
                )
            try:
                record = self._sessions.pop(session_id)
            except KeyError as exc:
                raise KeyError(f"unknown fast-weight session: {session_id!r}") from exc
            self._sessions[session_id] = record
            if not self._allow_unverified_research and (
                self._trusted_programmer_sha256 is None
                or record.authorization is None
                or not record.authorization.valid_for(self._trusted_programmer_sha256)
            ):
                self._sessions.pop(session_id, None)
                self._bytes -= record.nbytes
                self._assert_invariants_locked()
                self._notify_removed_locked((session_id,))
                raise RuntimeError(
                    "Fast Weight session authorization is no longer valid"
                )
            modules = dict(self.model.named_modules())
            targets: list[tuple[nn.Linear, LowRankOverlay]] = []
            for name, overlay in record.overlays.items():
                module = modules.get(name)
                if not isinstance(module, nn.Linear):
                    raise RuntimeError(
                        f"cached overlay target is no longer an nn.Linear: {name!r}"
                    )
                targets.append((module, overlay))
            try:
                for module, overlay in targets:
                    handles.append(
                        module.register_forward_hook(self._hook_for(overlay))
                    )
            except BaseException:
                for handle in reversed(handles):
                    handle.remove()
                handles.clear()
                self._assert_invariants_locked()
                raise
            self._active_session = session_id
            self._assert_invariants_locked()
        try:
            yield record
        finally:
            # Remove in reverse registration order in case future hooks depend
            # on ordering. RemovableHandle.remove() is idempotent.
            cleanup_error: BaseException | None = None
            for handle in reversed(handles):
                try:
                    handle.remove()
                except BaseException as exc:  # pragma: no cover - PyTorch handle fault
                    if cleanup_error is None:
                        cleanup_error = exc
            with self._lock:
                self._active_session = None
                self._purge_expired_locked()
                self._assert_invariants_locked()
            if cleanup_error is not None:
                raise RuntimeError(
                    "failed to remove a Fast Weight hook"
                ) from cleanup_error


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
        # The lifecycle owns domain-count merging so its DomainRecord list and
        # the sample-weighted Fisher snapshots are updated in one transaction.
        self.regularizer = FPEWCRegularizer(
            strength=strength,
            max_domains=FPEWCRegularizer.HARD_MAX_DOMAINS,
        )
        self.spectral_margin = spectral_margin
        self.max_domains = 16 if max_domains is None else max_domains
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

        self.regularizer.consolidate(params, scores, n_samples=n_samples)
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

    def estimate_empirical_and_consolidate(
        self,
        domain_id: str,
        *,
        f_at_z: Callable[[Tensor], Tensor],
        z_star: Tensor,
        log_likelihood_per_sample: Callable[[Tensor], Tensor],
        named_parameters: Iterable[tuple[str, nn.Parameter]],
        config: FixedPointFisherConfig,
        solver_converged: bool,
    ) -> tuple[FixedPointFisherEstimate, FisherSnapshot]:
        """Use the typed per-sample empirical fixed-point Fisher authority."""

        if not domain_id:
            raise ValueError("domain_id must be non-empty")
        params = list(named_parameters)
        estimate = estimate_empirical_fixed_point_fisher(
            f_at_z=f_at_z,
            z_star=z_star,
            log_likelihood_per_sample=log_likelihood_per_sample,
            named_parameters=params,
            config=config,
            solver_converged=solver_converged,
        )
        snapshot = self.regularizer.consolidate_fisher(
            params,
            estimate.fisher,
            n_samples=estimate.n_samples,
        )
        self.domains.append(DomainRecord(domain_id, estimate.n_samples))
        self._enforce_domain_budget()
        return estimate, snapshot

    def penalty(self, named_parameters: Iterable[tuple[str, nn.Parameter]]) -> Tensor:
        return self.regularizer.penalty(named_parameters)

    def _enforce_domain_budget(self) -> None:
        """Merge oldest diagonal quadratics to keep snapshot memory fixed.

        For diagonal quadratics, F1(theta-a1)^2 + F2(theta-a2)^2 has the
        same gradient as (F1+F2)(theta-a_bar)^2, where a_bar is the
        Fisher-weighted mean.  The dropped term is constant in theta.
        """

        while len(self.regularizer.snapshots) > self.max_domains:
            self.regularizer.merge_oldest()
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
