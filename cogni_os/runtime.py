from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from hashlib import sha256
import math
from pathlib import Path
import os
from threading import RLock

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
from cogni_core.fp_ewc import FisherSnapshot
from cogni_core.experts import BoundedSparseImplicitExperts, ExpertOutput
from cogni_core.meta_router import BioHAMAMetaRouter, CognitiveState, RoutingDecision
from cogni_core.resources import VRAMGuard
from cogni_core.routing import ContrastiveSessionRouter, OODDecision
from cogni_core.search import (
    BoundedPUCTSearch,
    PUCTResult,
    PolicyValueFn,
    TensorTransition,
)
from cogni_core.swarm import SwarmOutput, TensorSwarm
from cogni_flow.rhythm import RhythmController, SystemMode


@dataclass(frozen=True)
class InferenceResult:
    backbone_state: Tensor
    search: PUCTResult
    session_id: str | None
    ood: OODDecision | None = None


@dataclass(frozen=True)
class FastWeightCompilationResult:
    """Auditable result of compiling and admitting one temporary session."""

    admission: AdmissionResult
    programmer_quality: float
    calibrated: bool

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
        search: BoundedPUCTSearch,
        *,
        rhythm: RhythmController | None = None,
        vram_guard: VRAMGuard | None = None,
        sessions: FastWeightSessionCache | None = None,
        domains: FixedPointDomainLifecycle | None = None,
        swarm: TensorSwarm | None = None,
        session_router: ContrastiveSessionRouter | None = None,
        experts: BoundedSparseImplicitExperts | None = None,
        meta_router: BioHAMAMetaRouter | None = None,
        fast_weight_programmer: FastWeightProgrammer | None = None,
        fast_weight_target: str = "adapter.core",
    ) -> None:
        self.backbone = backbone
        self.search_engine = search
        self.rhythm = rhythm or RhythmController()
        self.vram_guard = vram_guard or VRAMGuard()
        self.sessions = sessions
        self.domains = domains or FixedPointDomainLifecycle()
        self.swarm = swarm
        self.session_router = session_router
        self.experts = experts
        self.meta_router = meta_router
        self.fast_weight_programmer = fast_weight_programmer
        self.fast_weight_target = fast_weight_target
        self._swarm_state: Tensor | None = None
        self._session_lock = RLock()

    def compile_fast_weight_session(
        self,
        session_id: str,
        z_star: Tensor,
        *,
        solver_info: SolverInfo,
        verified_quality: float,
        calibration_features: Tensor | None = None,
        calibration_quantile: float = 0.99,
    ) -> FastWeightCompilationResult:
        """Compile a converged batch-one fixed point into a bounded session.

        ``verified_quality`` deliberately comes from an external held-out
        evaluation.  The programmer's learned quality head is returned as
        telemetry but can never admit its own overlay.  No base parameter is
        written, and no dense target-by-target update is materialised.
        """

        if self.fast_weight_programmer is None or self.sessions is None:
            raise RuntimeError("Fast Weight compilation is not configured")
        if not isinstance(solver_info, SolverInfo) or not solver_info.converged:
            raise ValueError("Fast Weight compilation requires a converged solver")
        if not math.isfinite(solver_info.residual):
            raise ValueError("solver residual must be finite")
        quality = float(verified_quality)
        if not math.isfinite(quality) or not 0.0 <= quality <= 1.0:
            raise ValueError("verified_quality must be finite and in [0, 1]")
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
        if calibration_features is not None:
            if self.session_router is None:
                raise RuntimeError("OOD calibration requires a session router")
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
            )
        if not torch.equal(base_before, target.weight.detach()):
            raise RuntimeError("Fast Weight compilation mutated the base core")

        calibrated = False
        if admission.accepted and calibration_features is not None:
            assert self.session_router is not None
            self.session_router.calibrate(
                session_id, calibration_features, quantile=calibration_quantile
            )
            calibrated = True
        return FastWeightCompilationResult(
            admission,
            float(emitted.quality.detach().float().mean()),
            calibrated,
        )

    def infer(
        self,
        inputs: Tensor,
        transition: TensorTransition,
        policy_value: PolicyValueFn,
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
        if session_id is not None and self.sessions is None:
            raise RuntimeError(
                "a session was requested but no FastWeightSessionCache is configured"
            )
        decision = None
        effective_session = session_id
        if session_id is not None and self.session_router is not None:
            if routing_features is None:
                raise ValueError("routing_features are required for OOD-gated sessions")
            decision = self.session_router.route(session_id, routing_features)
            if not decision.allow_fast_path:
                effective_session = None
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
                    output = self.backbone(
                        inputs, *backbone_args, **(backbone_kwargs or {})
                    )
                    root = extract_hidden_states(output)
                self.vram_guard.admit(
                    self.search_engine.estimated_preallocated_bytes(root)
                )
                result = self.search_engine.search(
                    root, transition, policy_value, seed=seed
                )
        return InferenceResult(root, result, effective_session, decision)

    def adapt_stream(self, observations: Tensor) -> SwarmOutput:
        if self.swarm is None:
            raise RuntimeError("TensorSwarm is not configured")
        with self.rhythm.inference_slot(), self.vram_guard.enforce():
            output = self.swarm(observations, self._swarm_state)
            self._swarm_state = output.joint_state.detach()
            return output

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

    def route_cognitive_state(
        self, state: Tensor | CognitiveState, available_mask: Tensor | None = None
    ) -> RoutingDecision:
        if self.meta_router is None:
            raise RuntimeError("BIO-HAMA meta router is not configured")
        with self.rhythm.inference_slot():
            return self.meta_router(state, available_mask)

    def expert_step(self, z: Tensor, x: Tensor) -> ExpertOutput:
        if self.experts is None:
            raise RuntimeError("System3 expert pool is not configured")
        with self.rhythm.inference_slot(), self.vram_guard.enforce():
            return self.experts(z, x, track_usage=True)

    def recruit_expert(self, observations: Tensor):
        if self.experts is None:
            raise RuntimeError("System3 expert pool is not configured")
        if self.rhythm.mode != SystemMode.EVOLUTION:
            raise RuntimeError(
                "expert recruitment is allowed only during evolution mode"
            )
        estimate = observations.numel() * observations.element_size() * 2
        with self.rhythm.evolution_slot(), self.vram_guard.enforce(estimate):
            return self.experts.recruit_(observations)

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
        torch.save(
            {
                "backbone": self.backbone.state_dict(),
                "swarm": None if self.swarm is None else self.swarm.state_dict(),
                "swarm_state": self._swarm_state,
                "experts": None if self.experts is None else self.experts.state_dict(),
                "meta_router": (
                    None if self.meta_router is None else self.meta_router.state_dict()
                ),
                "fp_ewc": [
                    {"fisher": snapshot.fisher, "anchor": snapshot.anchor}
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
            self.backbone.load_state_dict(payload["backbone"], strict=True)
            if self.swarm is not None and payload.get("swarm") is not None:
                self.swarm.load_state_dict(payload["swarm"], strict=True)
            if self.experts is not None and payload.get("experts") is not None:
                self.experts.load_state_dict(payload["experts"], strict=True)
            if self.meta_router is not None and payload.get("meta_router") is not None:
                self.meta_router.load_state_dict(payload["meta_router"], strict=True)
            restored_swarm_state = payload.get("swarm_state")
            if restored_swarm_state is not None:
                if self.swarm is None:
                    raise RuntimeError(
                        "checkpoint contains a live swarm state but runtime does not"
                    )
                swarm_parameter = next(self.swarm.parameters())
                restored_swarm_state = restored_swarm_state.to(swarm_parameter)
            self._swarm_state = restored_swarm_state
            self.domains.regularizer.snapshots = [
                FisherSnapshot(fisher=item["fisher"], anchor=item["anchor"])
                for item in payload.get("fp_ewc", [])
            ]
            self._enforce_restored_contractivity()

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
