"""Bounded tensor orchestration for one conversational Cogni-Core turn.

The product boundary deliberately separates the answer-bearing path from
untrained auxiliary modules.  Gemma plus CTS/DEQ remains the authoritative
inference result.  BIO-HAMA, System 4, and System 3 run as bounded tensor-only
observers and their states are returned as advisory telemetry; they are never
silently fused into the answer latent.

A Fast Weight overlay may affect a turn only when it was already admitted by
``GenesisRuntime`` and its calibrated near-OOD router allows the session.  A
new overlay is compiled only *after* the current turn, using an externally
verified quality value, so a programmer can never certify its own output.
FP-EWC is intentionally absent here because consolidation is evolution-only.
"""

from __future__ import annotations

from dataclasses import dataclass
from time import monotonic_ns
from typing import Mapping

import torch
from torch import Tensor

from cogni_core.deq import SolverInfo
from cogni_core.experts import ExpertOutput
from cogni_core.meta_router import COGNITIVE_STATE_DIM, RoutingDecision
from cogni_core.search import (
    CertifiedTransitionV2,
    PolicyValueFn,
    TensorTransition,
)
from cogni_core.swarm import SwarmOutput
from cogni_os.runtime import (
    FastWeightCompilationResult,
    GenesisRuntime,
    InferenceResult,
    SearchCollaboratorsV2,
)

from .protocol import DIGEST_BYTES, NO_DEADLINE_NS, ZERO_ARTIFACT_DIGEST


class CoreTurnAuthorityError(RuntimeError):
    """Raised when a turn's immutable worker capability is no longer valid."""


@dataclass(frozen=True)
class CorePipelineLimits:
    """Hard request and telemetry bounds for the conversational core path."""

    max_batch_size: int = 1
    max_sequence_length: int = 4_096
    max_tensor_arguments: int = 16
    max_backbone_elements: int = 16_777_216
    max_advisory_elements: int = 16_384
    max_routing_feature_elements: int = 16_384
    max_calibration_samples: int = 64

    def __post_init__(self) -> None:
        for name, value in vars(self).items():
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")


@dataclass(frozen=True)
class FastWeightActivation:
    """Request use of an already admitted and OOD-calibrated session overlay."""

    session_id: str
    routing_features: Tensor


@dataclass(frozen=True)
class FastWeightCompilationPlan:
    """Compile this turn's converged state for a later conversational turn.

    Held-out quality comes exclusively from the hash-pinned trained artifact,
    never from the request. ``residual_trace`` is the exact ten-value AQ window
    from the current certified solve. Calibration features remain mandatory.
    """

    session_id: str
    residual_trace: Tensor
    calibration_features: Tensor
    calibration_quantile: float = 0.99


@dataclass(frozen=True)
class CoreTurnRequest:
    """Tensor request accepted by :class:`CoreTurnPipeline`.

    Text tokenization and tool/control metadata belong to Cogni-Flow.  Once the
    request crosses this boundary, model arguments and module communication are
    tensors.  The session identifier is bounded control-plane metadata, not a
    value exchanged between Cogni-Core modules.
    """

    inputs: Tensor
    cognitive_state: Tensor
    swarm_session_id: str
    backbone_args: tuple[Tensor, ...] = ()
    backbone_kwargs: Mapping[str, Tensor] | None = None
    available_modules: Tensor | None = None
    fast_weight: FastWeightActivation | None = None
    compile_fast_weight: FastWeightCompilationPlan | None = None
    seed: int | None = None
    estimated_workspace_bytes: int | None = None
    request_id: int = 1
    job_id: int = 1
    lease_epoch: int = 0
    request_deadline_ns: int = NO_DEADLINE_NS
    lease_deadline_ns: int = 0
    artifact_digest: Tensor | None = None


@dataclass(frozen=True)
class CoreTurnTelemetry:
    """Bounded auxiliary evidence; none of these tensors is answer-bearing."""

    routing: RoutingDecision
    swarm: SwarmOutput
    experts: ExpertOutput
    fast_weight_compilation: FastWeightCompilationResult | None
    advisory_only: bool = True

    @property
    def advisory_state(self) -> Tensor:
        """Detached System-3 state for diagnostics or a future certified gate."""

        return self.experts.state.detach()


@dataclass(frozen=True)
class CoreTurnResult:
    """Authoritative inference plus explicitly separated advisory telemetry."""

    inference: InferenceResult
    pooled_observation: Tensor
    telemetry: CoreTurnTelemetry


class CoreTurnPipeline:
    """Coordinate the bounded Cogni-Core modules for one local chat turn.

    The execution order is:

    1. BIO-HAMA produces routing telemetry from a caller-supplied tensor state.
    2. Gemma and the certified CTS/DEQ transition produce authoritative state.
    3. System 4 observes the pooled Gemma state.
    4. System 3 consumes the System-4 latent and the same observation.
    5. An explicitly requested Fast Weight candidate may be compiled for the
       *next* turn after the transition's convergence evidence is checked.

    No method in this class enters evolution mode or invokes FP-EWC.
    """

    def __init__(
        self,
        runtime: GenesisRuntime,
        transition: TensorTransition | CertifiedTransitionV2,
        policy_value: PolicyValueFn | SearchCollaboratorsV2,
        *,
        limits: CorePipelineLimits | None = None,
    ) -> None:
        self.runtime = runtime
        self.transition = transition
        self.policy_value = policy_value
        self.limits = limits or CorePipelineLimits()

    def run(self, request: CoreTurnRequest) -> CoreTurnResult:
        prepared = self._validate_request(request)
        cognitive_state, backbone_args, backbone_kwargs = prepared
        self._validate_live_authority(request)

        # Hold one outer day slot across the whole turn. Runtime methods take
        # nested slots for standalone safety; this closes the gaps in which an
        # idle scheduler could otherwise begin evolution between stages.
        with self.runtime.rhythm.inference_slot():
            routing_input = self._to_module_float(
                cognitive_state, getattr(self.runtime, "meta_router", None)
            )
            routing = self.runtime.route_cognitive_state(
                routing_input, request.available_modules
            )
            self._validate_routing(routing, cognitive_state.shape[0])

            activation = request.fast_weight
            inference = self.runtime.infer(
                request.inputs,
                self.transition,
                self.policy_value,
                session_id=None if activation is None else activation.session_id,
                seed=request.seed,
                estimated_workspace_bytes=request.estimated_workspace_bytes,
                backbone_args=backbone_args,
                backbone_kwargs=backbone_kwargs,
                routing_features=(
                    None if activation is None else activation.routing_features
                ),
            )
            # A capability may expire during the expensive authoritative solve.
            # No advisory state or answer-bearing decode may follow expiry.
            self._validate_live_authority(request)
            observation = self._pool_backbone_state(inference.backbone_state)

            # Auxiliary modules observe a detached tensor. This makes it
            # impossible for random/untrained parameters to alter this turn's
            # answer graph while retaining their bounded telemetry value.
            swarm_input = self._to_module_float(
                observation.detach(), getattr(self.runtime, "swarm", None)
            )
            swarm = self.runtime.adapt_stream(
                swarm_input, session_id=request.swarm_session_id
            )
            self._validate_swarm(swarm, observation.shape[0])
            expert_module = getattr(self.runtime, "experts", None)
            expert_latent = self._to_module_float(swarm.latent.detach(), expert_module)
            expert_input = self._to_module_float(observation.detach(), expert_module)
            experts = self.runtime.expert_step(
                expert_latent, expert_input, track_usage=False
            )
            self._validate_experts(experts, observation.shape[0])
            self._validate_live_authority(request)

            compilation = None
            if request.compile_fast_weight is not None:
                compilation = self._compile_for_next_turn(
                    request.compile_fast_weight, inference
                )

        return CoreTurnResult(
            inference=inference,
            pooled_observation=observation,
            telemetry=CoreTurnTelemetry(
                routing=routing,
                swarm=swarm,
                experts=experts,
                fast_weight_compilation=compilation,
            ),
        )

    def _validate_request(
        self, request: CoreTurnRequest
    ) -> tuple[Tensor, tuple[Tensor, ...], dict[str, Tensor]]:
        if not isinstance(request, CoreTurnRequest):
            raise TypeError("request must be a CoreTurnRequest")
        inputs = request.inputs
        self._require_tensor(inputs, "inputs")
        if inputs.ndim != 2 or inputs.shape[0] == 0 or inputs.shape[1] == 0:
            raise ValueError("inputs must have non-empty shape [batch, sequence]")
        if inputs.shape[0] > self.limits.max_batch_size:
            raise ValueError("inputs exceed the fixed conversational batch bound")
        if inputs.shape[1] > self.limits.max_sequence_length:
            raise ValueError("inputs exceed the fixed conversational sequence bound")
        self._require_finite(inputs, "inputs")

        state = request.cognitive_state
        self._require_tensor(state, "cognitive_state")
        if not torch.is_floating_point(state):
            raise TypeError("cognitive_state must be floating point")
        if state.ndim == 1:
            state = state.unsqueeze(0)
        if tuple(state.shape) != (inputs.shape[0], COGNITIVE_STATE_DIM):
            raise ValueError(
                "cognitive_state must have shape [batch, 5] matching inputs"
            )
        if state.device != inputs.device:
            raise ValueError("cognitive_state must share the inputs device")
        self._require_finite(state, "cognitive_state")

        args = tuple(request.backbone_args)
        kwargs = dict(request.backbone_kwargs or {})
        if len(args) + len(kwargs) > self.limits.max_tensor_arguments:
            raise ValueError("too many model tensor arguments")
        for index, value in enumerate(args):
            self._validate_model_argument(value, f"backbone_args[{index}]", inputs)
        for name, value in kwargs.items():
            if not isinstance(name, str) or not name or len(name) > 64:
                raise ValueError("model argument names must be bounded non-empty text")
            self._validate_model_argument(value, f"backbone_kwargs[{name}]", inputs)

        available = request.available_modules
        if available is not None:
            self._require_tensor(available, "available_modules")
            if available.dtype is not torch.bool:
                raise TypeError("available_modules must be a bool tensor")
            if available.ndim not in {1, 2} or available.numel() > 256:
                raise ValueError("available_modules exceeds its fixed routing bound")

        if request.seed is not None and (
            not isinstance(request.seed, int)
            or isinstance(request.seed, bool)
            or not 0 <= request.seed < 2**63
        ):
            raise ValueError("seed must be an integer in [0, 2**63)")
        workspace = request.estimated_workspace_bytes
        if workspace is not None and (
            not isinstance(workspace, int)
            or isinstance(workspace, bool)
            or workspace <= 0
        ):
            raise ValueError("estimated_workspace_bytes must be a positive integer")

        if request.fast_weight is not None:
            self._validate_activation(request.fast_weight)
        if request.compile_fast_weight is not None:
            self._validate_compilation_plan(request.compile_fast_weight)
        self._validate_session_id(request.swarm_session_id)
        self._validate_authority_schema(request)
        return state, args, kwargs

    @staticmethod
    def _validate_authority_schema(request: CoreTurnRequest) -> None:
        for name in ("request_id", "job_id", "request_deadline_ns"):
            value = getattr(request, name)
            if (
                not isinstance(value, int)
                or isinstance(value, bool)
                or not 1 <= value <= NO_DEADLINE_NS
            ):
                raise ValueError(f"{name} must be a positive signed-63-bit integer")
        if (
            not isinstance(request.lease_epoch, int)
            or isinstance(request.lease_epoch, bool)
            or not 0 <= request.lease_epoch <= NO_DEADLINE_NS
        ):
            raise ValueError("lease_epoch must be a non-negative signed-63-bit integer")
        if (
            not isinstance(request.lease_deadline_ns, int)
            or isinstance(request.lease_deadline_ns, bool)
            or not 0 <= request.lease_deadline_ns <= NO_DEADLINE_NS
            or (request.lease_epoch == 0) != (request.lease_deadline_ns == 0)
        ):
            raise ValueError("lease epoch/deadline authority is inconsistent")
        digest = (
            ZERO_ARTIFACT_DIGEST
            if request.artifact_digest is None
            else request.artifact_digest
        )
        if (
            not isinstance(digest, Tensor)
            or digest.device.type != "cpu"
            or digest.dtype != torch.int64
            or digest.shape != (DIGEST_BYTES,)
            or not digest.is_contiguous()
            or bool(((digest < 0) | (digest > 255)).any())
        ):
            raise ValueError("artifact_digest must be a contiguous 32-byte CPU tensor")

    @staticmethod
    def _validate_live_authority(request: CoreTurnRequest) -> None:
        now = monotonic_ns()
        if now >= request.request_deadline_ns:
            raise CoreTurnAuthorityError("Core turn request deadline expired")
        if request.lease_epoch and now >= request.lease_deadline_ns:
            raise CoreTurnAuthorityError("Core turn GPU lease deadline expired")

    def _validate_model_argument(
        self, value: Tensor, label: str, inputs: Tensor
    ) -> None:
        self._require_tensor(value, label)
        if value.device != inputs.device:
            raise ValueError(f"{label} must share the inputs device")
        if value.numel() > self.limits.max_batch_size * self.limits.max_sequence_length:
            raise ValueError(f"{label} exceeds the fixed tensor argument bound")
        self._require_finite(value, label)

    def _validate_activation(self, activation: FastWeightActivation) -> None:
        if not isinstance(activation, FastWeightActivation):
            raise TypeError("fast_weight must be a FastWeightActivation")
        self._validate_session_id(activation.session_id)
        features = activation.routing_features
        self._require_tensor(features, "fast_weight.routing_features")
        if not torch.is_floating_point(features) or features.ndim == 0:
            raise TypeError("Fast Weight routing features must be floating point")
        if features.numel() > self.limits.max_routing_feature_elements:
            raise ValueError("Fast Weight routing features exceed their fixed bound")
        self._require_finite(features, "fast_weight.routing_features")
        sessions = getattr(self.runtime, "sessions", None)
        router = getattr(self.runtime, "session_router", None)
        if sessions is None or router is None:
            raise RuntimeError(
                "Fast Weight activation requires admission and calibrated OOD routing"
            )
        # Unknown/expired sessions are a normal same-request fail-closed case:
        # the runtime records the failed attempt and executes full CTS.

    def _validate_compilation_plan(self, plan: FastWeightCompilationPlan) -> None:
        if not isinstance(plan, FastWeightCompilationPlan):
            raise TypeError("compile_fast_weight must be a FastWeightCompilationPlan")
        self._validate_session_id(plan.session_id)
        residuals = plan.residual_trace
        self._require_tensor(residuals, "residual_trace")
        if (
            not torch.is_floating_point(residuals)
            or residuals.ndim != 1
            or residuals.numel() != 10
        ):
            raise ValueError("residual_trace must contain exactly ten floats")
        self._require_finite(residuals, "residual_trace")
        features = plan.calibration_features
        self._require_tensor(features, "calibration_features")
        if (
            not torch.is_floating_point(features)
            or features.ndim != 2
            or not 2 <= features.shape[0] <= self.limits.max_calibration_samples
            or features.numel() > self.limits.max_routing_feature_elements
        ):
            raise ValueError(
                "calibration_features must be bounded [samples, features] floating data"
            )
        self._require_finite(features, "calibration_features")
        if not 0.5 <= plan.calibration_quantile < 1.0:
            raise ValueError("calibration_quantile must be in [0.5, 1)")

    def _pool_backbone_state(self, state: Tensor) -> Tensor:
        self._require_tensor(state, "backbone_state")
        if not torch.is_floating_point(state) or state.ndim not in {2, 3}:
            raise ValueError(
                "backbone_state must be floating [batch, (sequence), hidden]"
            )
        if state.shape[0] == 0 or state.shape[-1] == 0:
            raise ValueError("backbone_state cannot be empty")
        if state.numel() > self.limits.max_backbone_elements:
            raise ValueError("backbone_state exceeds the fixed product bound")
        self._require_finite(state, "backbone_state")
        observation = state.mean(dim=1) if state.ndim == 3 else state
        if observation.numel() > self.limits.max_advisory_elements:
            raise ValueError("pooled observation exceeds the advisory bound")
        return observation.detach()

    @staticmethod
    def _to_module_float(value: Tensor, module: object) -> Tensor:
        """Cast detached control/advisory data to its module without touching the answer.

        Local Gemma and its bounded auxiliary modules may use BF16 while the
        caller-supplied cognitive telemetry is FP32. PyTorch linear/recurrent
        operators require an exact dtype match. The cast is confined to the
        detached BIO-HAMA/System-3/System-4 control plane and therefore cannot
        change the authoritative Gemma/CTS state.
        """

        parameters = getattr(module, "parameters", None)
        if not callable(parameters):
            return value.detach()
        try:
            parameter = next(parameters())
        except (StopIteration, TypeError):
            return value.detach()
        if not torch.is_floating_point(parameter):
            raise TypeError("control-plane module parameters must be floating point")
        return value.detach().to(
            device=parameter.device,
            dtype=parameter.dtype,
            non_blocking=True,
        )

    def _validate_routing(self, routing: RoutingDecision, batch: int) -> None:
        mask = getattr(routing, "routing_mask", None)
        self._require_tensor(mask, "routing.routing_mask")
        if mask.ndim != 2 or mask.shape[0] != batch or mask.numel() > 256:
            raise ValueError("BIO-HAMA routing output violates the fixed bound")
        self._require_finite(mask, "routing.routing_mask")

    def _validate_swarm(self, output: SwarmOutput, batch: int) -> None:
        if getattr(output, "advisory_only", None) is not True:
            raise RuntimeError("System-4 output crossed its advisory-only boundary")
        latent = getattr(output, "latent", None)
        self._require_tensor(latent, "swarm.latent")
        if (
            latent.ndim != 2
            or latent.shape[0] != batch
            or latent.numel() > self.limits.max_advisory_elements
        ):
            raise ValueError("System-4 latent violates the fixed advisory bound")
        self._require_finite(latent, "swarm.latent")
        safe = getattr(output, "safe_for_advice", None)
        converged = getattr(output, "converged", None)
        for value, label in (
            (safe, "swarm.safe_for_advice"),
            (converged, "swarm.converged"),
        ):
            self._require_tensor(value, label)
            if value.ndim != 0 or value.dtype != torch.bool:
                raise TypeError(f"{label} must be a scalar bool tensor")
        if bool(safe.detach().cpu()) != bool(converged.detach().cpu()):
            raise RuntimeError("System-4 safety and convergence telemetry disagree")
        if not bool(safe.detach().cpu()) and bool(
            latent.count_nonzero().detach().cpu()
        ):
            raise RuntimeError(
                "unsafe System-4 output exposed a non-zero advisory latent"
            )

    def _validate_experts(self, output: ExpertOutput, batch: int) -> None:
        state = getattr(output, "state", None)
        self._require_tensor(state, "experts.state")
        if (
            state.ndim != 2
            or state.shape[0] != batch
            or state.numel() > self.limits.max_advisory_elements
        ):
            raise ValueError("System-3 state violates the fixed advisory bound")
        self._require_finite(state, "experts.state")

    def _compile_for_next_turn(
        self, plan: FastWeightCompilationPlan, inference: InferenceResult
    ) -> FastWeightCompilationResult:
        info = getattr(self.transition, "last_info", None)
        if not isinstance(info, SolverInfo) or not info.converged:
            raise RuntimeError(
                "Fast Weight compilation requires the current converged DEQ evidence"
            )
        fast_telemetry = getattr(inference, "fast_weight", None)
        if fast_telemetry is not None and fast_telemetry.activated:
            raise RuntimeError(
                "a CTS-bypassed Fast Weight turn cannot compile another session"
            )
        best_state = getattr(inference.search, "best_state", None)
        self._require_tensor(best_state, "search.best_state")
        if best_state.numel() > self.limits.max_backbone_elements:
            raise ValueError("search.best_state exceeds the fixed product bound")
        self._require_finite(best_state, "search.best_state")
        return self.runtime.compile_fast_weight_session(
            plan.session_id,
            best_state.detach(),
            solver_info=info,
            residual_trace=plan.residual_trace,
            calibration_features=plan.calibration_features,
            calibration_quantile=plan.calibration_quantile,
        )

    @staticmethod
    def _validate_session_id(session_id: str) -> None:
        if (
            not isinstance(session_id, str)
            or not 1 <= len(session_id) <= 64
            or any(
                character
                not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_"
                for character in session_id
            )
        ):
            raise ValueError(
                "session_id must use 1-64 ASCII letters, digits, '-' or '_'"
            )

    @staticmethod
    def _require_tensor(value: object, label: str) -> None:
        if not isinstance(value, Tensor):
            raise TypeError(f"{label} must be a tensor")

    @staticmethod
    def _require_finite(value: Tensor, label: str) -> None:
        if torch.is_floating_point(value) or torch.is_complex(value):
            if not bool(torch.isfinite(value).all().detach().cpu()):
                raise ValueError(f"{label} must contain only finite values")


__all__ = [
    "CoreTurnAuthorityError",
    "CorePipelineLimits",
    "CoreTurnPipeline",
    "CoreTurnRequest",
    "CoreTurnResult",
    "CoreTurnTelemetry",
    "FastWeightActivation",
    "FastWeightCompilationPlan",
]
