"""Fail-closed Phase-8 lifecycle for bounded System-3 experts.

The expert bank itself is an advisory tensor primitive.  This module adds the
night-only evidence sequence needed before a preallocated slot may even become
an advisory canary.  It deliberately ships with no calibration, training, or
verifier artifact: the default product state therefore has no expert answer
authority.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from hashlib import sha256
import math
import os
from pathlib import Path
from typing import Iterator, Mapping

import torch
from torch import Tensor

from .c_fire import CFireCertificate, c_fire_scaled_polar_
from .experts import (
    BoundedSparseImplicitExperts,
    EXPERT_ACTIVE,
    EXPERT_CANARY,
    EXPERT_CANDIDATE,
    EXPERT_CFIRE_CERTIFIED,
    EXPERT_FISHER,
    EXPERT_HELD_OUT,
    EXPERT_INACTIVE,
    EXPERT_QUARANTINED,
    EXPERT_TRAINED,
    RouterOutput,
)


EXPERT_PARAMETER_NAMES = ("recurrent", "input_weight", "bias")
CHECKPOINT_SCHEMA = 1


class ExpertLifecycleError(RuntimeError):
    """Raised after a candidate fails a mandatory promotion condition."""


@dataclass(frozen=True, slots=True)
class RoutedFisherSnapshot:
    """Per-slot diagonal empirical Fisher held on CPU in FP32."""

    fisher: dict[str, Tensor]
    anchor: dict[str, Tensor]
    routing_mass: Tensor
    sample_count: int
    router_digest: str
    quadratic_offset: Tensor | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.sample_count, int) or self.sample_count <= 0:
            raise ValueError("sample_count must be a positive integer")
        if len(self.router_digest) != 64:
            raise ValueError("router_digest must be a SHA-256 hex digest")
        try:
            bytes.fromhex(self.router_digest)
        except ValueError as error:
            raise ValueError("router_digest must be a SHA-256 hex digest") from error
        if (
            self.routing_mass.device.type != "cpu"
            or self.routing_mass.dtype != torch.float32
        ):
            raise ValueError("routing_mass must be a CPU FP32 tensor")
        if self.routing_mass.ndim != 1 or not bool(
            torch.isfinite(self.routing_mass).all()
        ):
            raise ValueError("routing_mass must be finite and one-dimensional")
        if bool((self.routing_mass < 0).any()):
            raise ValueError("routing_mass must be non-negative")
        if set(self.fisher) != set(self.anchor) or not self.fisher:
            raise ValueError("Fisher and anchor keys must match and be non-empty")
        slots = self.routing_mass.shape[0]
        for name in self.fisher:
            fisher = self.fisher[name]
            anchor = self.anchor[name]
            if (
                fisher.device.type != "cpu"
                or anchor.device.type != "cpu"
                or fisher.dtype != torch.float32
                or anchor.dtype != torch.float32
            ):
                raise ValueError("routed Fisher tensors must be CPU FP32")
            if fisher.shape != anchor.shape or fisher.shape[0] != slots:
                raise ValueError(f"routed Fisher shape mismatch for {name!r}")
            if not bool(torch.isfinite(fisher).all()) or bool((fisher < 0).any()):
                raise ValueError(f"Fisher {name!r} must be finite and non-negative")
            if not bool(torch.isfinite(anchor).all()):
                raise ValueError(f"anchor {name!r} must be finite")
        offset = self.quadratic_offset
        if offset is not None:
            if (
                offset.device.type != "cpu"
                or offset.dtype != torch.float32
                or offset.shape != self.routing_mass.shape
                or not bool(torch.isfinite(offset).all())
                or bool((offset < 0).any())
            ):
                raise ValueError(
                    "quadratic_offset must be finite non-negative CPU FP32"
                )

    @property
    def nbytes(self) -> int:
        tensors = [self.routing_mass, *self.fisher.values(), *self.anchor.values()]
        if self.quadratic_offset is not None:
            tensors.append(self.quadratic_offset)
        return sum(value.numel() * value.element_size() for value in tensors)


def estimate_routed_fisher(
    pool: BoundedSparseImplicitExperts,
    routing: RouterOutput,
    per_sample_gradients: Mapping[str, Tensor],
    *,
    router_digest: str,
) -> RoutedFisherSnapshot:
    """Square per-sample expert score gradients under detached route mass.

    Each supplied gradient has shape ``[batch, max_experts, ...]`` and denotes
    the score derivative before route weighting.  The detached gate is applied
    exactly once here, preventing router gradients from leaking into the
    fixed-point Fisher.
    """

    pool.assert_router_frozen(router_digest)
    batch, slots = routing.gates.shape
    if slots != pool.config.max_experts or batch <= 0:
        raise ValueError("routing shape does not match the expert pool")
    gates = routing.gates.detach().to(device="cpu", dtype=torch.float32)
    if (
        not bool(torch.isfinite(gates).all())
        or bool((gates < 0).any())
        or not torch.allclose(gates.sum(-1), torch.ones(batch), atol=1.0e-5, rtol=0.0)
        or bool((gates.count_nonzero(-1) > pool.config.top_k).any())
    ):
        raise ValueError("routing gates are not finite normalized top-k weights")
    mass = gates.sum(0)
    fisher: dict[str, Tensor] = {}
    anchor: dict[str, Tensor] = {}
    for name in EXPERT_PARAMETER_NAMES:
        if name not in per_sample_gradients:
            continue
        parameter = getattr(pool, name)
        gradient = per_sample_gradients[name]
        expected = (batch, *parameter.shape)
        if gradient.shape != expected:
            raise ValueError(f"per-sample gradient {name!r} must have shape {expected}")
        work = gradient.detach().to(device="cpu", dtype=torch.float32)
        if not bool(torch.isfinite(work).all()):
            raise ValueError(f"per-sample gradient {name!r} must be finite")
        view = (batch, slots, *((1,) * (work.ndim - 2)))
        numerator = (work.square() * gates.view(view)).sum(0)
        denominator = mass.view(slots, *((1,) * (work.ndim - 2))).clamp_min(1.0e-12)
        value = torch.where(
            denominator > 1.0e-12, numerator / denominator, torch.zeros_like(numerator)
        )
        fisher[name] = value.contiguous()
        anchor[name] = parameter.detach().to(device="cpu", dtype=torch.float32).clone()
    if not fisher:
        raise ValueError("at least one expert gradient is required")
    return RoutedFisherSnapshot(
        fisher=fisher,
        anchor=anchor,
        routing_mass=mass.contiguous(),
        sample_count=batch,
        router_digest=router_digest,
        quadratic_offset=torch.zeros(slots, dtype=torch.float32),
    )


@dataclass(slots=True)
class SparseRoutedFPEWC:
    """Bounded routed-Fisher consolidation for preallocated expert slices."""

    strength: float = 1.0
    max_domains: int = 16
    max_total_bytes: int = 64 * 1024 * 1024
    snapshots: list[RoutedFisherSnapshot] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not math.isfinite(float(self.strength)) or self.strength < 0:
            raise ValueError("strength must be finite and non-negative")
        if not 1 <= self.max_domains <= 256:
            raise ValueError("max_domains must lie in [1, 256]")
        if not 1 <= self.max_total_bytes <= 1024**3:
            raise ValueError("max_total_bytes must lie in [1, 1 GiB]")

    @property
    def total_bytes(self) -> int:
        return sum(item.nbytes for item in self.snapshots)

    def consolidate(
        self,
        pool: BoundedSparseImplicitExperts,
        snapshot: RoutedFisherSnapshot,
    ) -> None:
        pool.assert_router_frozen(snapshot.router_digest)
        for name, value in snapshot.anchor.items():
            if value.shape != getattr(pool, name).shape:
                raise ValueError(
                    f"routed Fisher does not match pool parameter {name!r}"
                )
        previous = list(self.snapshots)
        self.snapshots.append(snapshot)
        try:
            while len(self.snapshots) > 1 and (
                len(self.snapshots) > self.max_domains
                or self.total_bytes > self.max_total_bytes
            ):
                self.snapshots[:2] = [self._merge(self.snapshots[0], self.snapshots[1])]
            if (
                len(self.snapshots) > self.max_domains
                or self.total_bytes > self.max_total_bytes
            ):
                raise ExpertLifecycleError("routed FP-EWC evidence budget exceeded")
        except BaseException:
            self.snapshots = previous
            raise

    @staticmethod
    def _merge(
        first: RoutedFisherSnapshot, second: RoutedFisherSnapshot
    ) -> RoutedFisherSnapshot:
        mass = first.routing_mass + second.routing_mass
        safe_mass = mass.clamp_min(1.0e-12)
        fisher: dict[str, Tensor] = {}
        anchor: dict[str, Tensor] = {}
        offset = (
            torch.zeros_like(mass)
            if first.quadratic_offset is None
            else first.quadratic_offset.clone()
        )
        if second.quadratic_offset is not None:
            offset.add_(second.quadratic_offset)
        for name in first.fisher.keys() | second.fisher.keys():
            template = first.fisher.get(name, second.fisher.get(name))
            if template is None:
                raise RuntimeError("missing routed Fisher merge template")
            f1 = first.fisher.get(name, torch.zeros_like(template))
            f2 = second.fisher.get(name, torch.zeros_like(template))
            a1 = first.anchor.get(name, torch.zeros_like(template))
            a2 = second.anchor.get(name, torch.zeros_like(template))
            expand = (mass.shape[0], *((1,) * (template.ndim - 1)))
            m1 = first.routing_mass.view(expand)
            m2 = second.routing_mass.view(expand)
            precision1 = f1 * m1
            precision2 = f2 * m2
            precision = precision1 + precision2
            merged_anchor = torch.where(
                precision > 0,
                (precision1 * a1 + precision2 * a2) / precision.clamp_min(1.0e-12),
                torch.zeros_like(precision),
            )
            fisher[name] = precision / safe_mass.view(expand)
            anchor[name] = merged_anchor
            constant = (
                precision1 * a1.square()
                + precision2 * a2.square()
                - precision * merged_anchor.square()
            )
            offset.add_(0.5 * constant.flatten(1).sum(1).clamp_min(0))
        return RoutedFisherSnapshot(
            fisher=fisher,
            anchor=anchor,
            routing_mass=mass,
            sample_count=first.sample_count + second.sample_count,
            router_digest=sha256(
                (first.router_digest + second.router_digest).encode("ascii")
            ).hexdigest(),
            quadratic_offset=offset,
        )

    def penalty(
        self,
        pool: BoundedSparseImplicitExperts,
        *,
        slot_mask: Tensor | None = None,
    ) -> Tensor:
        total = pool.recurrent.new_zeros(())
        requested = None if slot_mask is None else slot_mask.to(pool.active_mask)
        if requested is not None and requested.shape != pool.active_mask.shape:
            raise ValueError("slot_mask shape does not match expert pool")
        for snapshot in self.snapshots:
            selected = (
                snapshot.routing_mass.gt(0).to(pool.active_mask)
                if requested is None
                else requested
            )
            for name, fisher in snapshot.fisher.items():
                parameter = getattr(pool, name)
                local_fisher = fisher.to(parameter)
                local_anchor = snapshot.anchor[name].to(parameter)
                view = (selected.shape[0], *((1,) * (parameter.ndim - 1)))
                mask = selected.view(view).to(parameter.dtype)
                total = (
                    total
                    + 0.5
                    * self.strength
                    * (mask * local_fisher * (parameter - local_anchor).square()).sum()
                )
            if snapshot.quadratic_offset is not None:
                total = (
                    total
                    + self.strength
                    * (snapshot.quadratic_offset.to(total) * selected.to(total)).sum()
                )
        return total


@dataclass(frozen=True, slots=True)
class ExternalVerifierAttestation:
    verifier_id: str
    artifact_sha256: str
    passed: bool
    independent: bool

    def __post_init__(self) -> None:
        if not self.verifier_id.strip():
            raise ValueError("verifier_id must be non-empty")
        if len(self.artifact_sha256) != 64:
            raise ValueError("artifact_sha256 must be a SHA-256 hex digest")
        try:
            bytes.fromhex(self.artifact_sha256)
        except ValueError as error:
            raise ValueError("artifact_sha256 must be a SHA-256 hex digest") from error


@dataclass(slots=True)
class _CandidateTransaction:
    slot: int
    before_state: dict[str, Tensor]
    before_requires_grad: tuple[bool, ...]
    before_fisher: list[RoutedFisherSnapshot]
    router_digest: str
    cfire: CFireCertificate | None = None
    fisher: RoutedFisherSnapshot | None = None


class ExpertCandidateLifecycle:
    """One-at-a-time, rollback-capable expert promotion transaction."""

    def __init__(
        self,
        pool: BoundedSparseImplicitExperts,
        *,
        regularizer: SparseRoutedFPEWC | None = None,
    ) -> None:
        pool.assert_phase8_profile()
        self.pool = pool
        self.regularizer = regularizer or SparseRoutedFPEWC()
        self._candidate: _CandidateTransaction | None = None
        self._last_promotion: (
            tuple[
                int,
                dict[str, Tensor],
                tuple[bool, ...],
                list[RoutedFisherSnapshot],
            ]
            | None
        ) = None
        self._attestations: dict[int, ExternalVerifierAttestation] = {}

    @property
    def candidate_slot(self) -> int | None:
        return None if self._candidate is None else self._candidate.slot

    @staticmethod
    def _clone_state(pool: BoundedSparseImplicitExperts) -> dict[str, Tensor]:
        return {
            name: value.detach().to(device="cpu").clone()
            for name, value in pool.state_dict().items()
        }

    @staticmethod
    def _requires_grad(pool: BoundedSparseImplicitExperts) -> tuple[bool, ...]:
        return tuple(parameter.requires_grad for parameter in pool.parameters())

    def _require_stage(self, expected: int) -> _CandidateTransaction:
        candidate = self._candidate
        if candidate is None or int(self.pool.slot_state[candidate.slot]) != expected:
            raise ExpertLifecycleError("expert candidate is not in the required stage")
        return candidate

    def start_candidate(self, observations: Tensor) -> int:
        if self._candidate is not None:
            raise ExpertLifecycleError("only one expert candidate may exist at a time")
        if not bool(self.pool.novelty_calibrated):
            raise ExpertLifecycleError(
                "expert recruitment requires verified novelty calibration"
            )
        routing = self.pool.route(observations)
        novel = routing.novelty
        if float(novel.float().mean()) < self.pool.config.recruit_fraction:
            raise ExpertLifecycleError(
                "candidate observations do not meet the novelty fraction"
            )
        free = (
            ~self.pool.active_mask
            & ~self.pool.quarantine_mask
            & self.pool.slot_state.eq(EXPERT_INACTIVE)
        )
        if not bool(free.any()):
            raise ExpertLifecycleError("no non-quarantined expert slot is available")
        slot = int(torch.argmax(free.to(torch.long)).detach().cpu())
        before = self._clone_state(self.pool)
        requires_grad = self._requires_grad(self.pool)
        weights = novel.to(routing.embedding.dtype)[:, None]
        prototype = (routing.embedding * weights).sum(0) / weights.sum().clamp_min(1.0)
        self.pool.prepare_candidate_slot_(slot, prototype)
        digest = self.pool.freeze_router_()
        self._candidate = _CandidateTransaction(
            slot,
            before,
            requires_grad,
            list(self.regularizer.snapshots),
            digest,
        )
        return slot

    def certify_c_fire(self) -> CFireCertificate:
        candidate = self._require_stage(EXPERT_CANDIDATE)
        try:
            certificate = c_fire_scaled_polar_(
                self.pool.recurrent[candidate.slot],
                gamma=min(0.90, self.pool.config.spectral_margin - 0.01),
                spectral_margin=self.pool.config.spectral_margin,
            )
            self.pool.slot_state[candidate.slot] = EXPERT_CFIRE_CERTIFIED
            candidate.cfire = certificate
            return certificate
        except BaseException as error:
            self.rollback_candidate(quarantine=True)
            raise ExpertLifecycleError(
                "candidate C-FIRE certification failed"
            ) from error

    @contextmanager
    def candidate_gradient_scope(self) -> Iterator[None]:
        """Mask batched parameter gradients to the candidate slice only."""

        candidate = self._require_stage(EXPERT_CFIRE_CERTIFIED)
        handles = []
        slot = candidate.slot
        for name in EXPERT_PARAMETER_NAMES:
            parameter = getattr(self.pool, name)
            mask = torch.zeros_like(parameter)
            mask[slot] = 1
            handles.append(parameter.register_hook(lambda grad, mask=mask: grad * mask))
        try:
            yield
        finally:
            for handle in handles:
                handle.remove()

    def finish_training(
        self, *, steps: int, before_loss: float, after_loss: float
    ) -> CFireCertificate:
        candidate = self._require_stage(EXPERT_CFIRE_CERTIFIED)
        if not isinstance(steps, int) or steps <= 0:
            self.rollback_candidate(quarantine=True)
            raise ExpertLifecycleError("candidate training must report positive steps")
        if (
            not all(math.isfinite(value) for value in (before_loss, after_loss))
            or after_loss > before_loss
        ):
            self.rollback_candidate(quarantine=True)
            raise ExpertLifecycleError(
                "candidate training evidence is non-finite or regressed"
            )
        try:
            self.pool.assert_router_frozen(candidate.router_digest)
            for name in EXPERT_PARAMETER_NAMES:
                parameter = getattr(self.pool, name)
                if not bool(torch.isfinite(parameter[candidate.slot]).all()):
                    raise ExpertLifecycleError("candidate parameter became non-finite")
                protected = torch.arange(
                    self.pool.config.max_experts, device=parameter.device
                ).ne(candidate.slot)
                expected = candidate.before_state[name].to(parameter)[protected]
                if not torch.equal(parameter.detach()[protected], expected):
                    raise ExpertLifecycleError(
                        "candidate optimizer changed a non-candidate expert slice"
                    )
            certificate = c_fire_scaled_polar_(
                self.pool.recurrent[candidate.slot],
                gamma=min(0.90, self.pool.config.spectral_margin - 0.01),
                spectral_margin=self.pool.config.spectral_margin,
            )
            candidate.cfire = certificate
            self.pool.slot_state[candidate.slot] = EXPERT_TRAINED
            return certificate
        except BaseException as error:
            self.rollback_candidate(quarantine=True)
            raise ExpertLifecycleError(
                "candidate post-training safety check failed"
            ) from error

    def accept_held_out(
        self,
        *,
        baseline_metric: float,
        candidate_metric: float,
        sample_count: int,
        routing_evidence: RouterOutput,
        minimum_improvement: float = 0.0,
    ) -> None:
        candidate = self._require_stage(EXPERT_TRAINED)
        values = (baseline_metric, candidate_metric, minimum_improvement)
        if (
            not all(math.isfinite(value) for value in values)
            or sample_count < 8
            or candidate_metric < baseline_metric + minimum_improvement
            or routing_evidence.gates.shape[0] != sample_count
        ):
            self.rollback_candidate(quarantine=True)
            raise ExpertLifecycleError("candidate failed held-out admission")
        try:
            self.pool.assert_router_frozen(candidate.router_digest)
            self.pool.assert_routing_not_collapsed(routing_evidence)
        except BaseException as error:
            self.rollback_candidate(quarantine=True)
            raise ExpertLifecycleError("candidate held-out router collapsed") from error
        self.pool.slot_state[candidate.slot] = EXPERT_HELD_OUT

    def consolidate_fisher(self, snapshot: RoutedFisherSnapshot) -> None:
        candidate = self._require_stage(EXPERT_HELD_OUT)
        slot = candidate.slot
        if (
            snapshot.router_digest != candidate.router_digest
            or float(snapshot.routing_mass[slot]) <= 0
        ):
            self.rollback_candidate(quarantine=True)
            raise ExpertLifecycleError("candidate lacks routed Fisher support")
        try:
            self.regularizer.consolidate(self.pool, snapshot)
            candidate.fisher = snapshot
            self.pool.slot_state[slot] = EXPERT_FISHER
        except BaseException as error:
            self.rollback_candidate(quarantine=True)
            raise ExpertLifecycleError(
                "candidate Fisher consolidation failed"
            ) from error

    def begin_canary(self) -> int:
        candidate = self._require_stage(EXPERT_FISHER)
        slot = candidate.slot
        norm = float(
            torch.linalg.matrix_norm(self.pool.recurrent[slot].detach().float(), ord=2)
        )
        if not math.isfinite(norm) or norm >= self.pool.config.spectral_margin:
            self.rollback_candidate(quarantine=True)
            raise ExpertLifecycleError(
                "candidate mixture contraction certificate failed"
            )
        self.pool.active_mask[slot] = True
        self.pool.canary_mask[slot] = True
        self.pool.answer_authority_mask[slot] = False
        # Adding an eligible route changes the whole mixture artifact.  Any
        # earlier authority attestations must be renewed against this router.
        self.pool.answer_authority_mask.zero_()
        self.pool.novelty_calibrated.fill_(False)
        self._attestations.clear()
        self.pool.slot_state[slot] = EXPERT_CANARY
        return slot

    def complete_canary(
        self,
        *,
        sample_count: int,
        failures: int,
        max_failure_rate: float = 0.01,
    ) -> int:
        candidate = self._require_stage(EXPERT_CANARY)
        if (
            sample_count < 32
            or failures < 0
            or failures > sample_count
            or not math.isfinite(max_failure_rate)
            or not 0 <= max_failure_rate < 1
            or failures / sample_count > max_failure_rate
        ):
            self.rollback_candidate(quarantine=True)
            raise ExpertLifecycleError("candidate failed canary admission")
        self.pool.assert_router_frozen(candidate.router_digest)
        slot = candidate.slot
        self.pool.canary_mask[slot] = False
        self.pool.slot_state[slot] = EXPERT_ACTIVE
        self._last_promotion = (
            slot,
            candidate.before_state,
            candidate.before_requires_grad,
            candidate.before_fisher,
        )
        self._candidate = None
        return slot

    def grant_answer_authority(
        self, slot: int, attestation: ExternalVerifierAttestation
    ) -> None:
        if int(self.pool.slot_state[slot]) != EXPERT_ACTIVE:
            raise ExpertLifecycleError("only a promoted active expert may be attested")
        if not attestation.passed or not attestation.independent:
            raise ExpertLifecycleError(
                "answer authority requires an independent passing verifier"
            )
        if attestation.artifact_sha256 != self.slot_digest(slot):
            raise ExpertLifecycleError(
                "verifier attestation does not match the expert artifact"
            )
        self.pool.answer_authority_mask[slot] = True
        self._attestations[slot] = attestation

    def slot_digest(self, slot: int) -> str:
        if not 0 <= slot < self.pool.config.max_experts:
            raise IndexError("expert slot is outside the preallocated pool")
        digest = sha256()
        digest.update(bytes.fromhex(self.pool.router_digest()))
        active = self.pool.active_mask.detach().to(device="cpu").contiguous()
        digest.update(active.view(torch.uint8).numpy().tobytes())
        for value in (
            self.pool.prototypes[slot],
            self.pool.recurrent[slot],
            self.pool.input_weight[slot],
            self.pool.bias[slot],
        ):
            tensor = value.detach().to(device="cpu").contiguous()
            digest.update(str(tuple(tensor.shape)).encode("ascii"))
            digest.update(str(tensor.dtype).encode("ascii"))
            digest.update(tensor.view(torch.uint8).numpy().tobytes())
        return digest.hexdigest()

    def rollback_candidate(self, *, quarantine: bool) -> None:
        candidate = self._candidate
        if candidate is None:
            return
        self._restore_state(candidate.before_state, candidate.before_requires_grad)
        if quarantine:
            self.pool._deactivate_slot_(candidate.slot)
            self.pool.quarantine_mask[candidate.slot] = True
            self.pool.slot_state[candidate.slot] = EXPERT_QUARANTINED
            self._attestations.pop(candidate.slot, None)
        self.regularizer.snapshots = candidate.before_fisher
        self._candidate = None

    def rollback_last_promotion(self) -> int:
        if self._candidate is not None:
            raise ExpertLifecycleError(
                "cannot roll back a promotion while another candidate is in flight"
            )
        if self._last_promotion is None:
            raise ExpertLifecycleError("there is no promoted expert to roll back")
        slot, state, requires_grad, fisher = self._last_promotion
        self._restore_state(state, requires_grad)
        self.regularizer.snapshots = fisher
        self.pool._deactivate_slot_(slot)
        self.pool.quarantine_mask[slot] = True
        self.pool.slot_state[slot] = EXPERT_QUARANTINED
        self._attestations.pop(slot, None)
        self._last_promotion = None
        return slot

    def _restore_state(
        self, state: Mapping[str, Tensor], requires_grad: tuple[bool, ...]
    ) -> None:
        current = self.pool.state_dict()
        if current.keys() != state.keys():
            raise ExpertLifecycleError("expert rollback checkpoint keys changed")
        converted: dict[str, Tensor] = {}
        for name, value in state.items():
            expected = current[name]
            if value.shape != expected.shape or value.dtype != expected.dtype:
                raise ExpertLifecycleError(
                    "expert rollback checkpoint shape/dtype changed"
                )
            if value.is_floating_point() and not bool(torch.isfinite(value).all()):
                raise ExpertLifecycleError("expert rollback checkpoint is non-finite")
            converted[name] = value.to(expected)
        self.pool.load_state_dict(converted, strict=True)
        for parameter, enabled in zip(
            self.pool.parameters(), requires_grad, strict=True
        ):
            parameter.requires_grad_(enabled)

    def write_checkpoint(self, directory: str | Path) -> tuple[Path, str]:
        if self._candidate is not None:
            raise ExpertLifecycleError(
                "stable checkpoint cannot contain an in-flight candidate"
            )
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        self._validate_pool_state()
        self._validate_authorities()
        target = directory / "system3-phase8.pt"
        temporary = target.with_suffix(".pt.tmp")
        payload = {
            "schema": CHECKPOINT_SCHEMA,
            "pool": self._clone_state(self.pool),
            "requires_grad": self._requires_grad(self.pool),
            "fisher": [
                {
                    "fisher": item.fisher,
                    "anchor": item.anchor,
                    "routing_mass": item.routing_mass,
                    "sample_count": item.sample_count,
                    "router_digest": item.router_digest,
                    "quadratic_offset": item.quadratic_offset,
                }
                for item in self.regularizer.snapshots
            ],
            "attestations": {
                slot: {
                    "verifier_id": item.verifier_id,
                    "artifact_sha256": item.artifact_sha256,
                    "passed": item.passed,
                    "independent": item.independent,
                }
                for slot, item in self._attestations.items()
            },
        }
        torch.save(payload, temporary)
        os.replace(temporary, target)
        digest = sha256(target.read_bytes()).hexdigest()
        target.with_suffix(".sha256").write_text(digest + "\n", encoding="ascii")
        return target, digest

    def restore_checkpoint(self, path: str | Path, expected_sha256: str) -> None:
        if self._candidate is not None:
            raise ExpertLifecycleError("cannot restore over an in-flight candidate")
        path = Path(path).resolve(strict=True)
        if sha256(path.read_bytes()).hexdigest() != expected_sha256.lower():
            raise ExpertLifecycleError("System 3 checkpoint digest verification failed")
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if payload.get("schema") != CHECKPOINT_SCHEMA:
            raise ExpertLifecycleError("unsupported System 3 checkpoint schema")
        before = self._clone_state(self.pool)
        requires_grad = self._requires_grad(self.pool)
        previous_fisher = list(self.regularizer.snapshots)
        previous_attestations = dict(self._attestations)
        try:
            checkpoint_grad = tuple(payload.get("requires_grad", ()))
            if len(checkpoint_grad) != len(tuple(self.pool.parameters())) or not all(
                isinstance(value, bool) for value in checkpoint_grad
            ):
                raise ExpertLifecycleError(
                    "System 3 checkpoint parameter-freeze metadata is invalid"
                )
            self._restore_state(payload["pool"], checkpoint_grad)
            self._validate_pool_state()
            snapshots = [
                RoutedFisherSnapshot(**item) for item in payload.get("fisher", [])
            ]
            if (
                len(snapshots) > self.regularizer.max_domains
                or sum(item.nbytes for item in snapshots)
                > self.regularizer.max_total_bytes
            ):
                raise ExpertLifecycleError(
                    "checkpoint routed Fisher exceeds its bounds"
                )
            self.regularizer.snapshots = snapshots
            self._attestations = {
                int(slot): ExternalVerifierAttestation(**item)
                for slot, item in payload.get("attestations", {}).items()
            }
            self._validate_authorities()
            self.pool._ensure_contractivity_()
            self._last_promotion = None
        except BaseException:
            self._restore_state(before, requires_grad)
            self.regularizer.snapshots = previous_fisher
            self._attestations = previous_attestations
            raise

    def _validate_pool_state(self) -> None:
        state = self.pool.slot_state
        stable = (
            state.eq(EXPERT_INACTIVE)
            | state.eq(EXPERT_ACTIVE)
            | state.eq(EXPERT_QUARANTINED)
        )
        if not bool(stable.all()):
            raise ExpertLifecycleError(
                "stable System 3 checkpoint contains a transient candidate state"
            )
        if not torch.equal(self.pool.active_mask, state.eq(EXPERT_ACTIVE)):
            raise ExpertLifecycleError("expert active mask and slot states disagree")
        if bool(self.pool.canary_mask.any()):
            raise ExpertLifecycleError("stable System 3 checkpoint contains a canary")
        if not torch.equal(self.pool.quarantine_mask, state.eq(EXPERT_QUARANTINED)):
            raise ExpertLifecycleError(
                "expert quarantine mask and slot states disagree"
            )
        if bool((self.pool.answer_authority_mask & ~self.pool.active_mask).any()):
            raise ExpertLifecycleError("inactive expert cannot have answer authority")
        if bool(self.pool.novelty_calibrated):
            self.pool.assert_router_frozen()
            if (
                int(self.pool.calibration_id_samples) < 2
                or int(self.pool.calibration_ood_samples) < 2
            ):
                raise ExpertLifecycleError("novelty calibration metadata is incomplete")

    def _validate_authorities(self) -> None:
        authoritative = self.pool.answer_authority_mask.nonzero().flatten().tolist()
        if set(authoritative) != set(self._attestations):
            raise ExpertLifecycleError(
                "answer-authority bits require matching verifier attestations"
            )
        for slot in authoritative:
            item = self._attestations[slot]
            if (
                not item.passed
                or not item.independent
                or item.artifact_sha256 != self.slot_digest(slot)
                or int(self.pool.slot_state[slot]) != EXPERT_ACTIVE
            ):
                raise ExpertLifecycleError(
                    "stored expert verifier attestation is invalid"
                )


__all__ = [
    "EXPERT_PARAMETER_NAMES",
    "ExpertCandidateLifecycle",
    "ExpertLifecycleError",
    "ExternalVerifierAttestation",
    "RoutedFisherSnapshot",
    "SparseRoutedFPEWC",
    "estimate_routed_fisher",
]
