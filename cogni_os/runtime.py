from __future__ import annotations

from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from hashlib import sha256
import math
from pathlib import Path
import os
from threading import RLock
from typing import Iterator

import torch
from torch import Tensor, nn

from cogni_core.adaptation import (
    AdmissionResult,
    FastWeightSessionCache,
    FixedPointDomainLifecycle,
    LowRankOverlay,
)
from cogni_core.backbone import GemmaDEQBackboneAdapter, extract_hidden_states
from cogni_core.deq import ContractivityError, EquilibriumLayer, SolverInfo
from cogni_core.fast_weights import FastWeightProgrammer, ResidualBottleneckAdapter
from cogni_core.fast_weight_safety import (
    AQCertificate,
    ResidualDecayAQGate,
    VerifiedFastWeightProgrammer,
    authorize_fast_weight_admission,
)
from cogni_core.fp_ewc import FisherSnapshot, FixedPointFisherConfig
from cogni_core.expert_lifecycle import (
    ExpertCandidateLifecycle,
    ExternalVerifierAttestation,
    RoutedFisherSnapshot,
)
from cogni_core.experts import (
    BoundedSparseImplicitExperts,
    ExpertOutput,
    RouterOutput,
)
from cogni_core.meta_router import BioHAMAMetaRouter, CognitiveState, RoutingDecision
from cogni_core.resources import VRAMGuard
from cogni_core.routing import ContrastiveSessionRouter, OODDecision
from cogni_core.search import (
    ActionPolicyV2,
    BoundedPUCTSearch,
    BoundedPUCTSearchV2,
    CertifiedTransitionV2,
    CriticV2,
    MetaControllerV2,
    PUCTResult,
    PUCTResultV2,
    PolicyValueFn,
    SearchRequestV2,
    TensorTransition,
)
from cogni_core.swarm import SwarmOutput, TensorSwarm
from cogni_core.swarm_sessions import SwarmSessionStateCache
from cogni_flow.rhythm import RhythmController, SystemMode


@dataclass(frozen=True)
class InferenceResult:
    backbone_state: Tensor
    search: PUCTResult | PUCTResultV2 | FastWeightPathResult
    session_id: str | None
    ood: OODDecision | None = None
    fast_weight: FastWeightExecutionTelemetry | None = None


@dataclass(frozen=True, slots=True)
class FastWeightExecutionTelemetry:
    """One bounded System-1.5 attempt and its same-request CTS disposition."""

    attempted: bool
    activated: bool
    fallback_to_cts: bool
    reason: str
    aq_certificate_digest: str | None = None


@dataclass(frozen=True, slots=True)
class FastWeightPathResult:
    """Authoritative root returned when an admitted overlay bypasses CTS."""

    best_state: Tensor
    nodes_used: int = 1
    reached_depth: int = 0


@dataclass(frozen=True, slots=True)
class SearchCollaboratorsV2:
    """Three independently callable learned surfaces for certified CTS V2."""

    action_policy: ActionPolicyV2
    critic: CriticV2
    meta_controller: MetaControllerV2

    def __post_init__(self) -> None:
        for name in ("action_policy", "critic", "meta_controller"):
            if not callable(getattr(self, name)):
                raise TypeError(f"{name} must be callable")


@dataclass(frozen=True)
class FastWeightCompilationResult:
    """Auditable result of compiling and admitting one temporary session."""

    admission: AdmissionResult
    programmer_quality: float | None
    calibrated: bool
    aq_certificate: AQCertificate

    @property
    def accepted(self) -> bool:
        return self.admission.accepted


class GenesisRuntime:
    """Integration boundary between the tensor core and day/night control plane.

    Session identifiers and audit metadata belong to this boundary. The search,
    swarm, expert, and meta-router hot paths exchange tensors only.
    """

    def __init__(
        self,
        backbone: nn.Module,
        search: BoundedPUCTSearch | BoundedPUCTSearchV2,
        *,
        rhythm: RhythmController | None = None,
        vram_guard: VRAMGuard | None = None,
        sessions: FastWeightSessionCache | None = None,
        domains: FixedPointDomainLifecycle | None = None,
        swarm: TensorSwarm | None = None,
        swarm_sessions: SwarmSessionStateCache | None = None,
        session_router: ContrastiveSessionRouter | None = None,
        experts: BoundedSparseImplicitExperts | None = None,
        expert_lifecycle: ExpertCandidateLifecycle | None = None,
        meta_router: BioHAMAMetaRouter | None = None,
        fast_weight_programmer: FastWeightProgrammer | None = None,
        verified_fast_weight: VerifiedFastWeightProgrammer | None = None,
        fast_weight_aq_gate: ResidualDecayAQGate | None = None,
        fast_weight_target: str = "adapter.core",
        search_mac_budget: int | None = None,
    ) -> None:
        self.backbone = backbone
        self.search_engine = search
        self.rhythm = rhythm or RhythmController()
        self.vram_guard = vram_guard or VRAMGuard()
        self.sessions = sessions
        self.domains = domains or FixedPointDomainLifecycle()
        self.swarm = swarm
        if swarm is None and swarm_sessions is not None:
            raise ValueError("swarm session cache requires TensorSwarm")
        self.swarm_sessions = (
            None if swarm is None else (swarm_sessions or SwarmSessionStateCache())
        )
        self.session_router = session_router
        self.experts = experts
        if expert_lifecycle is not None and experts is None:
            raise ValueError("System 3 lifecycle requires an expert pool")
        if expert_lifecycle is not None and expert_lifecycle.pool is not experts:
            raise ValueError("System 3 lifecycle and expert pool disagree")
        self.expert_lifecycle = expert_lifecycle
        self.meta_router = meta_router
        self.fast_weight_programmer = fast_weight_programmer
        self.verified_fast_weight = verified_fast_weight
        self.fast_weight_aq_gate = fast_weight_aq_gate or ResidualDecayAQGate()
        if verified_fast_weight is not None:
            if fast_weight_programmer is not verified_fast_weight.programmer:
                raise ValueError(
                    "verified Fast Weight handle and runtime programmer disagree"
                )
            if sessions is None or session_router is None:
                raise ValueError(
                    "verified Fast Weight requires a session cache and OOD router"
                )
        self.fast_weight_target = fast_weight_target
        if search_mac_budget is not None and (
            not isinstance(search_mac_budget, int)
            or isinstance(search_mac_budget, bool)
            or search_mac_budget < 1
        ):
            raise ValueError("search_mac_budget must be a positive integer")
        if isinstance(search, BoundedPUCTSearchV2) != (search_mac_budget is not None):
            raise ValueError(
                "certified CTS V2 and its fixed request MAC budget must be configured together"
            )
        self._search_mac_budget = search_mac_budget
        self._session_lock = RLock()

    @property
    def search_mac_budget(self) -> int | None:
        return self._search_mac_budget

    def install_certified_search_v2(
        self,
        search: BoundedPUCTSearchV2,
        *,
        request_mac_budget: int,
    ) -> None:
        """Install one fixed-budget V2 search before this runtime serves requests."""

        if not isinstance(search, BoundedPUCTSearchV2):
            raise TypeError("search must be BoundedPUCTSearchV2")
        if (
            not isinstance(request_mac_budget, int)
            or isinstance(request_mac_budget, bool)
            or request_mac_budget < 1
        ):
            raise ValueError("request_mac_budget must be a positive integer")
        if self.rhythm.mode != SystemMode.INFERENCE or self.rhythm.active_requests:
            raise RuntimeError(
                "certified search can be installed only before active work"
            )
        if isinstance(self.search_engine, BoundedPUCTSearchV2):
            raise RuntimeError("certified CTS V2 is already installed")
        self.search_engine = search
        self._search_mac_budget = request_mac_budget

    def compile_fast_weight_session(
        self,
        session_id: str,
        z_star: Tensor,
        *,
        solver_info: SolverInfo,
        residual_trace: Tensor,
        calibration_features: Tensor | None = None,
        calibration_quantile: float = 0.99,
    ) -> FastWeightCompilationResult:
        """Compile a converged batch-one fixed point into a bounded session.

        The externally verified quality and trained provenance come only from
        the hash-pinned checkpoint handle.  The programmer's learned quality
        head is telemetry and can never admit its own overlay.  The latest ten
        DEQ residuals must also pass AQ before any compiler work is admitted.
        """

        verified = self.verified_fast_weight
        if (
            self.fast_weight_programmer is None
            or self.sessions is None
            or verified is None
        ):
            raise RuntimeError(
                "Fast Weight compilation requires a verified trained checkpoint"
            )
        if not isinstance(solver_info, SolverInfo) or not solver_info.converged:
            raise ValueError("Fast Weight compilation requires a converged solver")
        if not math.isfinite(solver_info.residual):
            raise ValueError("solver residual must be finite")
        if not isinstance(residual_trace, Tensor):
            raise TypeError("residual_trace must be a tensor")
        certificate = self.fast_weight_aq_gate.evaluate(
            residual_trace,
            solver_converged=solver_info.converged,
            solver_used_fallback=(
                solver_info.used_fallback or solver_info.used_linear_solve_fallback
            ),
            programmer=verified.programmer_evidence,
            verifier=verified.verifier_evidence,
        )
        if not certificate.accepted:
            return FastWeightCompilationResult(
                AdmissionResult(
                    decision=self.sessions._rejection(  # noqa: SLF001 - same trust boundary
                        verified.verifier_evidence.verified_quality,
                        f"AQ rejected: {certificate.reason}",
                    ).decision
                ),
                None,
                False,
                certificate,
            )
        authorization = authorize_fast_weight_admission(verified, certificate)
        quality = float(verified.verifier_evidence.verified_quality)
        if z_star.ndim not in {1, 2, 3}:
            raise ValueError("z_star must be a rank-1, rank-2, or rank-3 tensor")
        batch = 1 if z_star.ndim == 1 else z_star.shape[0]
        if batch != 1:
            raise ValueError("Fast Weight compilation is restricted to batch size 1")
        if not torch.is_floating_point(z_star) or not torch.isfinite(z_star).all():
            raise ValueError("z_star must be a finite floating-point tensor")
        programmer = self.fast_weight_programmer
        if z_star.shape[-1] != programmer.source_dim:
            raise ValueError("z_star width does not match the programmer source_dim")
        if self.session_router is None:
            raise RuntimeError("OOD calibration requires a session router")
        if calibration_features is None:
            raise ValueError(
                "verified Fast Weight admission requires OOD calibration features"
            )
        if (
            calibration_features.ndim != 2
            or calibration_features.shape[0] < 2
            or not torch.is_floating_point(calibration_features)
            or not torch.isfinite(calibration_features).all()
        ):
            raise ValueError(
                "calibration_features must be finite [samples, features] data"
            )
        if not 0.5 <= calibration_quantile < 1.0:
            raise ValueError("calibration_quantile must be in [0.5, 1)")

        modules = dict(self.sessions.model.named_modules())
        target = modules.get(self.fast_weight_target)
        if not isinstance(target, nn.Linear):
            raise RuntimeError("configured Fast Weight target is not an nn.Linear")
        base_before = target.weight.detach().clone()
        try:
            parameter = next(programmer.parameters())
        except StopIteration as exc:  # defensive: a compiler must be learned/bounded
            raise RuntimeError("Fast Weight programmer owns no parameters") from exc
        compiler_input = z_star.detach().to(
            device=parameter.device, dtype=parameter.dtype
        )
        workspace = programmer.estimated_workspace_bytes(compiler_input)
        with (
            self._session_lock,
            self.rhythm.inference_slot(),
            self.vram_guard.enforce(workspace),
            torch.no_grad(),
        ):
            emitted = programmer(compiler_input)
            if emitted.a.shape[0] != 1 or emitted.b.shape[0] != 1:
                raise RuntimeError("programmer violated the batch-one contract")
            overlay = LowRankOverlay(emitted.a[0], emitted.b[0])
            admission = self.sessions.admit(
                session_id,
                {self.fast_weight_target: overlay},
                quality=quality,
                authorization=authorization,
            )
        if not torch.equal(base_before, target.weight.detach()):
            raise RuntimeError("Fast Weight compilation mutated the base core")

        calibrated = False
        if admission.accepted:
            try:
                router_evicted = self.session_router.calibrate(
                    session_id, calibration_features, quantile=calibration_quantile
                )
                for evicted_session in router_evicted:
                    self.sessions.discard(evicted_session)
                calibrated = True
            except BaseException:
                # Cache + router publication is one transaction.  A partially
                # calibrated overlay must never become eligible next request.
                self.sessions.discard(session_id)
                self.session_router.discard_many((session_id,))
                raise
        return FastWeightCompilationResult(
            admission,
            float(emitted.quality.detach().float().mean()),
            calibrated,
            certificate,
        )

    def infer(
        self,
        inputs: Tensor,
        transition: TensorTransition | CertifiedTransitionV2,
        policy_value: PolicyValueFn | SearchCollaboratorsV2,
        *,
        session_id: str | None = None,
        seed: int | None = None,
        estimated_workspace_bytes: int | None = None,
        backbone_args: tuple = (),
        backbone_kwargs: dict | None = None,
        routing_features: Tensor | None = None,
    ) -> InferenceResult:
        if self.vram_guard.enabled and (
            estimated_workspace_bytes is None or estimated_workspace_bytes <= 0
        ):
            raise ValueError(
                "CUDA inference requires a positive, profiled "
                "estimated_workspace_bytes admission value"
            )
        if (
            self.vram_guard.enabled
            and getattr(transition, "__cogni_broyden_solver__", None) is not True
        ):
            raise TypeError(
                "CUDA runtime requires a certified limited-Broyden tensor transition"
            )
        workspace_bytes = int(estimated_workspace_bytes or 0)
        decision = None
        effective_session: str | None = None
        attempted = session_id is not None
        fallback_reason = "not_requested"
        aq_digest: str | None = None
        if session_id is not None:
            fallback_reason = "session_missing"
            if self.verified_fast_weight is None:
                fallback_reason = "verified_checkpoint_missing"
            elif self.sessions is None or not self.sessions.feature_enabled:
                fallback_reason = "session_cache_disabled"
            else:
                try:
                    record = self.sessions.get(session_id)
                except (KeyError, RuntimeError):
                    record = None
                if record is not None:
                    aq_digest = record.authorization.aq_certificate.digest
                    if self.session_router is None:
                        fallback_reason = "ood_router_missing"
                    elif routing_features is None:
                        fallback_reason = "routing_features_missing"
                    else:
                        decision = self.session_router.route(
                            session_id, routing_features
                        )
                        if decision.allow_fast_path:
                            effective_session = session_id
                            fallback_reason = "admitted_aq_ood_fast_path"
                        else:
                            fallback_reason = "ood_rejected"
        fast_telemetry = FastWeightExecutionTelemetry(
            attempted=attempted,
            activated=effective_session is not None,
            fallback_to_cts=attempted and effective_session is None,
            reason=fallback_reason,
            aq_certificate_digest=aq_digest,
        )
        lock_context = (
            self._session_lock if effective_session is not None else nullcontext()
        )
        with lock_context:
            session_context = (
                self.sessions.activate(effective_session)
                if effective_session is not None
                else nullcontext()
            )
            with (
                self.rhythm.inference_slot(),
                self.vram_guard.enforce(workspace_bytes),
                session_context,
            ):
                with torch.no_grad():
                    target_weights: dict[str, Tensor] = {}
                    if effective_session is not None and self.sessions is not None:
                        cached = self.sessions.get(effective_session)
                        modules = dict(self.sessions.model.named_modules())
                        for name in cached.overlays:
                            target = modules.get(name)
                            if not isinstance(target, nn.Linear):
                                raise RuntimeError(
                                    "Fast Weight target changed after admission"
                                )
                            target_weights[name] = target.weight.detach().clone()
                    output = self.backbone(
                        inputs, *backbone_args, **(backbone_kwargs or {})
                    )
                    root = extract_hidden_states(output)
                    for name, before in target_weights.items():
                        target = dict(self.sessions.model.named_modules())[name]
                        if not torch.equal(before, target.weight.detach()):
                            raise RuntimeError(
                                "Fast Weight inference mutated a base parameter"
                            )
                if effective_session is not None:
                    return InferenceResult(
                        root,
                        FastWeightPathResult(best_state=root),
                        effective_session,
                        decision,
                        fast_telemetry,
                    )
                self.vram_guard.admit(
                    self.search_engine.estimated_preallocated_bytes(root)
                )
                if isinstance(self.search_engine, BoundedPUCTSearchV2):
                    if not isinstance(policy_value, SearchCollaboratorsV2):
                        raise TypeError(
                            "certified CTS V2 requires separate policy, critic, and meta callables"
                        )
                    if self._search_mac_budget is None:
                        raise RuntimeError(
                            "certified CTS V2 has no fixed request MAC budget"
                        )
                    request = SearchRequestV2(
                        root=root,
                        mac_budget=self._search_mac_budget,
                        seed=seed,
                    )
                    result = self.search_engine.search(
                        request,
                        transition,
                        policy_value.action_policy,
                        policy_value.critic,
                        policy_value.meta_controller,
                    )
                    self._require_safe_v2_result(result)
                else:
                    if isinstance(policy_value, SearchCollaboratorsV2):
                        raise TypeError(
                            "legacy CTS cannot consume certified V2 collaborators"
                        )
                    result = self.search_engine.search(
                        root, transition, policy_value, seed=seed
                    )
        return InferenceResult(
            root,
            result,
            effective_session,
            decision,
            fast_telemetry,
        )

    @staticmethod
    def _require_safe_v2_result(result: PUCTResultV2) -> None:
        if not isinstance(result, PUCTResultV2):
            raise TypeError("certified CTS V2 returned an untyped search result")
        telemetry = result.telemetry
        if (
            telemetry.safe_for_decode is not True
            or telemetry.linear_solve_fallbacks > 0
            or telemetry.unsafe_silent_fallbacks > 0
        ):
            raise RuntimeError(
                "certified CTS V2 refused decode after unsafe search telemetry: "
                f"safe={telemetry.safe_for_decode}, "
                f"linear_fallbacks={telemetry.linear_solve_fallbacks}, "
                f"silent_fallbacks={telemetry.unsafe_silent_fallbacks}, "
                f"solver_failures={telemetry.solver_failures}, "
                f"failed_edges={telemetry.failed_edges}, "
                f"all_fail_terminals={telemetry.all_fail_terminals}, "
                f"budget_exhausted={telemetry.mac_budget_exhausted}, "
                f"trace={telemetry.trace_digest}"
            )

    def adapt_stream(self, observations: Tensor, *, session_id: str) -> SwarmOutput:
        if self.swarm is None or self.swarm_sessions is None:
            raise RuntimeError("TensorSwarm is not configured")
        with self.rhythm.inference_slot(), self.vram_guard.enforce():
            return self.swarm_sessions.process(session_id, self.swarm, observations)

    def consolidate_domain(
        self, *args, estimated_workspace_bytes: int | None = None, **kwargs
    ):
        if self.rhythm.mode != SystemMode.EVOLUTION:
            raise RuntimeError(
                "FP-EWC domain consolidation is allowed only during evolution mode"
            )
        if self.vram_guard.enabled and (
            estimated_workspace_bytes is None or estimated_workspace_bytes <= 0
        ):
            raise ValueError(
                "CUDA consolidation requires a positive profiled workspace estimate"
            )
        with (
            self.rhythm.evolution_slot(),
            self.vram_guard.enforce(int(estimated_workspace_bytes or 0)),
        ):
            return self.domains.estimate_and_consolidate(*args, **kwargs)

    def consolidate_empirical_domain(
        self,
        domain_id: str,
        *,
        f_at_z,
        z_star: Tensor,
        log_likelihood_per_sample,
        named_parameters,
        config: FixedPointFisherConfig,
        solver_converged: bool,
        estimated_workspace_bytes: int | None = None,
    ):
        """Night-only typed per-sample empirical FP-Fisher authority."""

        if self.rhythm.mode != SystemMode.EVOLUTION:
            raise RuntimeError(
                "empirical FP-EWC consolidation is allowed only during evolution"
            )
        if self.vram_guard.enabled and (
            estimated_workspace_bytes is None or estimated_workspace_bytes <= 0
        ):
            raise ValueError(
                "CUDA consolidation requires a positive profiled workspace estimate"
            )
        with (
            self.rhythm.evolution_slot(),
            self.vram_guard.enforce(int(estimated_workspace_bytes or 0)),
        ):
            return self.domains.estimate_empirical_and_consolidate(
                domain_id,
                f_at_z=f_at_z,
                z_star=z_star,
                log_likelihood_per_sample=log_likelihood_per_sample,
                named_parameters=named_parameters,
                config=config,
                solver_converged=solver_converged,
            )

    def route_cognitive_state(
        self, state: Tensor | CognitiveState, available_mask: Tensor | None = None
    ) -> RoutingDecision:
        if self.meta_router is None:
            raise RuntimeError("BIO-HAMA meta router is not configured")
        with self.rhythm.inference_slot():
            return self.meta_router(state, available_mask)

    def expert_step(
        self, z: Tensor, x: Tensor, *, track_usage: bool = True
    ) -> ExpertOutput:
        if self.experts is None:
            raise RuntimeError("System3 expert pool is not configured")
        with self.rhythm.inference_slot(), self.vram_guard.enforce():
            return self.experts(z, x, track_usage=track_usage)

    def _require_expert_lifecycle(self) -> ExpertCandidateLifecycle:
        if self.experts is None or self.expert_lifecycle is None:
            raise RuntimeError("certified System 3 lifecycle is not configured")
        return self.expert_lifecycle

    @contextmanager
    def _expert_evolution_slot(
        self,
        *,
        estimated_workspace_bytes: int | None,
    ) -> Iterator[ExpertCandidateLifecycle]:
        lifecycle = self._require_expert_lifecycle()
        if self.rhythm.mode != SystemMode.EVOLUTION:
            raise RuntimeError(
                "System 3 mutation is allowed only during evolution mode"
            )
        if self.vram_guard.enabled and (
            estimated_workspace_bytes is None or estimated_workspace_bytes <= 0
        ):
            raise ValueError(
                "CUDA System 3 mutation requires a positive profiled workspace estimate"
            )
        with (
            self.rhythm.evolution_slot(),
            self.vram_guard.enforce(int(estimated_workspace_bytes or 0)),
        ):
            yield lifecycle

    def calibrate_expert_novelty(
        self,
        in_domain: Tensor,
        out_of_domain: Tensor,
        *,
        max_fpr: float = 0.05,
        max_fnr: float = 0.05,
        minimum_samples: int = 8,
        estimated_workspace_bytes: int | None = None,
    ) -> Tensor:
        """Night-only labelled calibration required before any candidate exists."""

        with self._expert_evolution_slot(
            estimated_workspace_bytes=estimated_workspace_bytes
        ):
            assert self.experts is not None
            return self.experts.calibrate_novelty_(
                in_domain,
                out_of_domain,
                max_fpr=max_fpr,
                max_fnr=max_fnr,
                minimum_samples=minimum_samples,
            )

    def recruit_expert(
        self,
        observations: Tensor,
        *,
        estimated_workspace_bytes: int | None = None,
    ) -> int:
        """Start a certified candidate; direct activation is intentionally forbidden."""

        with self._expert_evolution_slot(
            estimated_workspace_bytes=estimated_workspace_bytes
        ) as lifecycle:
            return lifecycle.start_candidate(observations)

    def certify_expert_candidate(self, *, estimated_workspace_bytes: int | None = None):
        with self._expert_evolution_slot(
            estimated_workspace_bytes=estimated_workspace_bytes
        ) as lifecycle:
            return lifecycle.certify_c_fire()

    @contextmanager
    def expert_candidate_training_scope(
        self, *, estimated_workspace_bytes: int | None = None
    ) -> Iterator[ExpertCandidateLifecycle]:
        """Hold the exclusive night/GPU slot while an optimizer updates one slice."""

        with (
            self._expert_evolution_slot(
                estimated_workspace_bytes=estimated_workspace_bytes
            ) as lifecycle,
            lifecycle.candidate_gradient_scope(),
        ):
            yield lifecycle

    def finish_expert_candidate_training(
        self,
        *,
        steps: int,
        before_loss: float,
        after_loss: float,
        estimated_workspace_bytes: int | None = None,
    ):
        with self._expert_evolution_slot(
            estimated_workspace_bytes=estimated_workspace_bytes
        ) as lifecycle:
            return lifecycle.finish_training(
                steps=steps,
                before_loss=before_loss,
                after_loss=after_loss,
            )

    def accept_expert_held_out(
        self,
        *,
        baseline_metric: float,
        candidate_metric: float,
        sample_count: int,
        routing_evidence: RouterOutput,
        minimum_improvement: float = 0.0,
        estimated_workspace_bytes: int | None = None,
    ) -> None:
        with self._expert_evolution_slot(
            estimated_workspace_bytes=estimated_workspace_bytes
        ) as lifecycle:
            lifecycle.accept_held_out(
                baseline_metric=baseline_metric,
                candidate_metric=candidate_metric,
                sample_count=sample_count,
                routing_evidence=routing_evidence,
                minimum_improvement=minimum_improvement,
            )

    def consolidate_expert_fisher(
        self,
        snapshot: RoutedFisherSnapshot,
        *,
        estimated_workspace_bytes: int | None = None,
    ) -> None:
        with self._expert_evolution_slot(
            estimated_workspace_bytes=estimated_workspace_bytes
        ) as lifecycle:
            lifecycle.consolidate_fisher(snapshot)

    def begin_expert_canary(
        self, *, estimated_workspace_bytes: int | None = None
    ) -> int:
        with self._expert_evolution_slot(
            estimated_workspace_bytes=estimated_workspace_bytes
        ) as lifecycle:
            return lifecycle.begin_canary()

    def complete_expert_canary(
        self,
        *,
        sample_count: int,
        failures: int,
        max_failure_rate: float = 0.01,
        estimated_workspace_bytes: int | None = None,
    ) -> int:
        with self._expert_evolution_slot(
            estimated_workspace_bytes=estimated_workspace_bytes
        ) as lifecycle:
            return lifecycle.complete_canary(
                sample_count=sample_count,
                failures=failures,
                max_failure_rate=max_failure_rate,
            )

    def grant_expert_answer_authority(
        self,
        slot: int,
        attestation: ExternalVerifierAttestation,
        *,
        estimated_workspace_bytes: int | None = None,
    ) -> None:
        with self._expert_evolution_slot(
            estimated_workspace_bytes=estimated_workspace_bytes
        ) as lifecycle:
            lifecycle.grant_answer_authority(slot, attestation)

    def rollback_expert_candidate(
        self,
        *,
        quarantine: bool = True,
        estimated_workspace_bytes: int | None = None,
    ) -> None:
        with self._expert_evolution_slot(
            estimated_workspace_bytes=estimated_workspace_bytes
        ) as lifecycle:
            lifecycle.rollback_candidate(quarantine=quarantine)

    def checkpoint(self, directory: str | Path) -> tuple[Path, str]:
        """Atomically write a local checkpoint and return its SHA-256 digest."""
        if self.rhythm.mode not in {
            SystemMode.CHECKPOINTING,
            SystemMode.EVOLUTION,
        }:
            raise RuntimeError("checkpointing is allowed only inside the night window")
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        target = directory / "genesis-state.pt"
        temporary = directory / "genesis-state.pt.tmp"
        system3_checkpoint: dict[str, str] | None = None
        if self.expert_lifecycle is not None:
            expert_path, expert_digest = self.expert_lifecycle.write_checkpoint(
                directory
            )
            system3_checkpoint = {
                "filename": expert_path.name,
                "sha256": expert_digest,
            }
        torch.save(
            {
                "backbone": self.backbone.state_dict(),
                "swarm": None if self.swarm is None else self.swarm.state_dict(),
                # Per-conversation System-4 states are TTL-bounded ephemeral
                # telemetry and never become trusted model checkpoint state.
                "swarm_topology_certificates": (
                    self._swarm_topology_certificate_payload()
                ),
                "experts": None if self.experts is None else self.experts.state_dict(),
                "system3_phase8": system3_checkpoint,
                "meta_router": (
                    None if self.meta_router is None else self.meta_router.state_dict()
                ),
                "fp_ewc": [
                    {
                        "fisher": snapshot.fisher,
                        "anchor": snapshot.anchor,
                        "n_samples": snapshot.n_samples,
                        "quadratic_offset": snapshot.quadratic_offset,
                    }
                    for snapshot in self.domains.regularizer.snapshots
                ],
                "deq_certificates": {
                    name: module.effective_lipschitz_upper_bound
                    for name, module in self.backbone.named_modules()
                    if isinstance(module, GemmaDEQBackboneAdapter)
                },
            },
            temporary,
        )
        os.replace(temporary, target)
        digest = sha256(target.read_bytes()).hexdigest()
        (directory / "genesis-state.sha256").write_text(digest + "\n", encoding="ascii")
        return target, digest

    def restore_checkpoint(self, checkpoint: str | Path, expected_sha256: str) -> None:
        if self.rhythm.mode not in {
            SystemMode.EVOLUTION,
            SystemMode.ROLLING_BACK,
        }:
            raise RuntimeError("checkpoint restore is allowed only during evolution")
        checkpoint = Path(checkpoint).resolve(strict=True)
        actual = sha256(checkpoint.read_bytes()).hexdigest()
        if actual != expected_sha256.lower():
            raise RuntimeError("checkpoint SHA-256 verification failed")
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        system3_metadata = payload.get("system3_phase8")
        system3_path: Path | None = None
        if self.expert_lifecycle is not None:
            if not isinstance(system3_metadata, dict):
                raise RuntimeError(
                    "checkpoint lacks certified System 3 lifecycle state"
                )
            filename = system3_metadata.get("filename")
            digest = system3_metadata.get("sha256")
            if (
                not isinstance(filename, str)
                or Path(filename).name != filename
                or not isinstance(digest, str)
                or len(digest) != 64
            ):
                raise RuntimeError("checkpoint System 3 lifecycle metadata is invalid")
            system3_path = (checkpoint.parent / filename).resolve(strict=True)
            if system3_path.parent != checkpoint.parent:
                raise RuntimeError(
                    "checkpoint System 3 lifecycle path escaped its directory"
                )
            if sha256(system3_path.read_bytes()).hexdigest() != digest.lower():
                raise RuntimeError(
                    "checkpoint System 3 lifecycle SHA-256 verification failed"
                )
        elif system3_metadata is not None:
            raise RuntimeError("checkpoint requires an unavailable System 3 lifecycle")
        context = (
            self.rhythm.evolution_slot()
            if self.rhythm.mode == SystemMode.EVOLUTION
            else nullcontext()
        )
        with context, self.vram_guard.enforce():
            self._validate_state_payload(self.backbone, payload["backbone"], "backbone")
            self._load_optional_state(self.swarm, payload.get("swarm"), "swarm")
            self._load_optional_state(self.experts, payload.get("experts"), "experts")
            self._load_optional_state(
                self.meta_router, payload.get("meta_router"), "meta_router"
            )
            self._validate_deq_certificates(payload.get("deq_certificates", {}))
            if "swarm_state" in payload:
                raise RuntimeError(
                    "legacy checkpoint contains forbidden ephemeral swarm state"
                )
            self._validate_swarm_topology_certificates(
                payload.get("swarm_topology_certificates", {})
            )
            self.backbone.load_state_dict(payload["backbone"], strict=True)
            if self.swarm is not None and payload.get("swarm") is not None:
                self.swarm.load_state_dict(payload["swarm"], strict=True)
            if self.experts is not None and payload.get("experts") is not None:
                self.experts.load_state_dict(payload["experts"], strict=True)
            if self.expert_lifecycle is not None:
                assert system3_path is not None
                self.expert_lifecycle.restore_checkpoint(
                    system3_path,
                    system3_metadata["sha256"],
                )
            if self.meta_router is not None and payload.get("meta_router") is not None:
                self.meta_router.load_state_dict(payload["meta_router"], strict=True)
            if self.swarm_sessions is not None:
                self.swarm_sessions.clear()
            self.domains.regularizer.snapshots = [
                FisherSnapshot(
                    fisher=item["fisher"],
                    anchor=item["anchor"],
                    n_samples=item["n_samples"],
                    quadratic_offset=item["quadratic_offset"],
                )
                for item in payload.get("fp_ewc", [])
            ]
            self._enforce_restored_contractivity()
            # A Genesis checkpoint is a different trust domain from the
            # separately hash-pinned System-1.5 artifact.  Even when tensor
            # shapes match, restore invalidates the prior FWP evidence and all
            # admitted overlays until a new verified runtime is constructed.
            if self.sessions is not None:
                self.sessions.feature_off()
            self.verified_fast_weight = None

    def _swarm_topology_certificate_payload(
        self,
    ) -> dict[str, dict[str, int | str]]:
        if self.swarm is None:
            return {}
        return {
            certificate.name: {
                "sha256": certificate.sha256,
                "agents": certificate.agents,
                "sensory_agents": certificate.sensory_agents,
                "reasoning_agents": certificate.reasoning_agents,
                "constraint_agents": certificate.constraint_agents,
                "edge_count": certificate.edge_count,
                "maximum_reachability_steps": (certificate.maximum_reachability_steps),
                "warm_step_budget": certificate.warm_step_budget,
            }
            for certificate in self.swarm.topology_certificates
        }

    def _validate_swarm_topology_certificates(self, saved: object) -> None:
        if not isinstance(saved, dict):
            raise RuntimeError("swarm topology certificate payload must be a mapping")
        current = self._swarm_topology_certificate_payload()
        if saved != current:
            raise RuntimeError(
                "checkpoint System-4 topology certificate does not match runtime"
            )

    @staticmethod
    def _validate_state_payload(
        module: nn.Module, state: dict[str, Tensor], label: str
    ) -> None:
        current = module.state_dict()
        if current.keys() != state.keys():
            raise RuntimeError(f"{label} checkpoint keys do not match runtime")
        for name, tensor in state.items():
            expected = current[name]
            if tensor.shape != expected.shape or tensor.dtype != expected.dtype:
                raise RuntimeError(
                    f"{label}.{name} shape or dtype does not match runtime"
                )
            if tensor.is_floating_point() and not torch.isfinite(tensor).all():
                raise RuntimeError(f"{label}.{name} contains non-finite values")

    @classmethod
    def _load_optional_state(
        cls, module: nn.Module | None, state: dict[str, Tensor] | None, label: str
    ) -> None:
        if state is None:
            return
        if module is None:
            raise RuntimeError(
                f"checkpoint contains {label} state but runtime does not"
            )
        cls._validate_state_payload(module, state, label)

    def _validate_deq_certificates(self, saved: dict[str, float]) -> None:
        current = {
            name: module
            for name, module in self.backbone.named_modules()
            if isinstance(module, GemmaDEQBackboneAdapter)
        }
        if current.keys() != saved.keys():
            raise ContractivityError(
                "checkpoint DEQ certificate set does not match the runtime"
            )
        for name, module in current.items():
            bound = float(saved[name])
            if bound != module.effective_lipschitz_upper_bound:
                raise ContractivityError(
                    f"checkpoint DEQ certificate changed for {name!r}"
                )
            if (
                module.config.fail_on_noncontractive
                and bound >= module.config.spectral_margin
            ):
                raise ContractivityError(
                    f"checkpoint DEQ certificate is unsafe for {name!r}"
                )

    def _enforce_restored_contractivity(self) -> None:
        spectral: list[tuple[str, nn.Parameter]] = []
        for name, module in self.backbone.named_modules():
            if isinstance(module, EquilibriumLayer):
                spectral.append((f"{name}.recurrent", module.recurrent))
            if isinstance(module, ResidualBottleneckAdapter):
                module.c_fire_()
        if spectral:
            self.domains.project_spectral_(spectral)
        if self.swarm is not None:
            self.swarm._ensure_contractivity_()
        if self.experts is not None:
            self.experts._ensure_contractivity_()
