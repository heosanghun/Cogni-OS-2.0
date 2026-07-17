"""Tensor-only BIO-HAMA cognitive state and hierarchical meta-routing.

The patent specification describes a five-part cognitive state followed by a
strategic, tactical, and reactive controller.  This module keeps that control
path deterministic and differentiable: there is no sampling, text parsing, or
Python mapping in ``route_tensor``.  Hard masks use straight-through relaxed
gates so the forward path is budgeted while policy gradients still reach every
router level.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F


COGNITIVE_STATE_DIM = 5


def cognitive_state_tensor(
    memory: Tensor,
    affect: Tensor,
    attention: Tensor,
    uncertainty: Tensor,
    load: Tensor,
) -> Tensor:
    """Return ``[..., 5]`` state ordered as the BIO-HAMA specification.

    The inputs are already-reduced telemetry metrics.  They must share a
    device and floating dtype, but may have broadcast-compatible batch shapes.
    Memory, attention, uncertainty, and load are normalized to ``[0, 1]``;
    affect retains valence and is normalized to ``[-1, 1]``.  Clipping is a
    tensor operation and therefore preserves gradients for in-range metrics.
    """

    components = (memory, affect, attention, uncertainty, load)
    if not all(isinstance(component, Tensor) for component in components):
        raise TypeError("all cognitive-state components must be tensors")
    if not all(torch.is_floating_point(component) for component in components):
        raise TypeError("all cognitive-state components must be floating point")
    if any(component.device != memory.device for component in components[1:]):
        raise ValueError("all cognitive-state components must share a device")
    if any(component.dtype != memory.dtype for component in components[1:]):
        raise ValueError("all cognitive-state components must share a dtype")

    broadcast = torch.broadcast_tensors(*components)
    state = torch.stack(broadcast, dim=-1)
    lower = state.new_tensor((0.0, -1.0, 0.0, 0.0, 0.0))
    upper = state.new_tensor((1.0, 1.0, 1.0, 1.0, 1.0))
    return torch.maximum(lower, torch.minimum(state, upper))


@dataclass(frozen=True)
class CognitiveState:
    """Five scalar tensor metrics that summarize the current cognitive state."""

    memory: Tensor
    affect: Tensor
    attention: Tensor
    uncertainty: Tensor
    load: Tensor

    @property
    def tensor(self) -> Tensor:
        return cognitive_state_tensor(
            self.memory,
            self.affect,
            self.attention,
            self.uncertainty,
            self.load,
        )

    def as_tensor(self) -> Tensor:
        """Return the normalized state without detaching its computation graph."""

        return self.tensor

    @classmethod
    def from_tensor(cls, state: Tensor) -> "CognitiveState":
        if not isinstance(state, Tensor):
            raise TypeError("state must be a tensor")
        if not torch.is_floating_point(state):
            raise TypeError("state must be floating point")
        if state.ndim == 0 or state.shape[-1] != COGNITIVE_STATE_DIM:
            raise ValueError("state must have final dimension 5")
        return cls(*state.unbind(dim=-1))


@dataclass(frozen=True)
class MetaRouterConfig:
    """Static budgets and neuromodulation bounds for the BIO-HAMA router."""

    num_modules: int = 8
    hidden_dim: int = 64
    strategic_top_k: int = 4
    tactical_top_k: int = 2
    reactive_top_k: int = 3
    routing_temperature: float = 0.5
    reactive_uncertainty_threshold: float = 0.75
    reactive_load_threshold: float = 0.85
    reactive_sharpness: float = 12.0
    alpha_min: float = 1e-5
    alpha_max: float = 1e-2
    alpha_base: float = 3e-4
    alpha_uncertainty_gain: float = 4.0
    gamma_min: float = 0.50
    gamma_max: float = 0.999
    gamma_base: float = 0.99
    gamma_load_gain: float = 4.0
    meta_variance_lambda: float = 0.1

    def __post_init__(self) -> None:
        if self.num_modules <= 0:
            raise ValueError("num_modules must be positive")
        if self.hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        if not 1 <= self.tactical_top_k <= self.strategic_top_k <= self.num_modules:
            raise ValueError(
                "routing budgets must satisfy tactical <= strategic <= num_modules"
            )
        if not 1 <= self.reactive_top_k <= self.num_modules:
            raise ValueError("reactive_top_k must be in [1, num_modules]")
        if self.routing_temperature <= 0.0:
            raise ValueError("routing_temperature must be positive")
        if not 0.0 <= self.reactive_uncertainty_threshold <= 1.0:
            raise ValueError("reactive uncertainty threshold must be in [0, 1]")
        if not 0.0 <= self.reactive_load_threshold <= 1.0:
            raise ValueError("reactive load threshold must be in [0, 1]")
        if self.reactive_sharpness <= 0.0:
            raise ValueError("reactive_sharpness must be positive")
        if not 0.0 <= self.alpha_min < self.alpha_base < self.alpha_max:
            raise ValueError("alpha bounds must contain alpha_base strictly")
        if self.alpha_uncertainty_gain <= 0.0:
            raise ValueError("alpha_uncertainty_gain must be positive")
        if not 0.0 <= self.gamma_min < self.gamma_base < self.gamma_max <= 1.0:
            raise ValueError("gamma bounds must contain gamma_base strictly")
        if self.gamma_load_gain <= 0.0:
            raise ValueError("gamma_load_gain must be positive")
        if self.meta_variance_lambda < 0.0:
            raise ValueError("meta_variance_lambda must be non-negative")


def _logit(probability: float) -> float:
    return math.log(probability) - math.log1p(-probability)


def _inverse_softplus(value: float) -> float:
    # log(expm1(x)) is accurate for the moderate positive gains accepted above.
    return math.log(math.expm1(value))


class BoundedNeuromodulator(nn.Module):
    """BIO-A-GRPO alpha/gamma schedules with structural monotonicity.

    ``alpha_t`` is strictly increasing in uncertainty and ``gamma_t`` is
    strictly decreasing in cognitive load.  Softplus-constrained gains retain
    those directions during training, while the affine sigmoid maps make both
    values remain inside their configured intervals.
    """

    def __init__(self, config: MetaRouterConfig):
        super().__init__()
        self.config = config
        alpha_position = (config.alpha_base - config.alpha_min) / (
            config.alpha_max - config.alpha_min
        )
        gamma_position = (config.gamma_base - config.gamma_min) / (
            config.gamma_max - config.gamma_min
        )
        self.register_buffer("alpha_base_logit", torch.tensor(_logit(alpha_position)))
        self.register_buffer("gamma_base_logit", torch.tensor(_logit(gamma_position)))
        self.raw_alpha_gain = nn.Parameter(
            torch.tensor(_inverse_softplus(config.alpha_uncertainty_gain))
        )
        self.raw_gamma_gain = nn.Parameter(
            torch.tensor(_inverse_softplus(config.gamma_load_gain))
        )

    def forward(self, state: Tensor) -> tuple[Tensor, Tensor]:
        if state.ndim == 0 or state.shape[-1] != COGNITIVE_STATE_DIM:
            raise ValueError("state must have final dimension 5")
        uncertainty = state[..., 3].clamp(0.0, 1.0)
        load = state[..., 4].clamp(0.0, 1.0)

        alpha_gain = F.softplus(self.raw_alpha_gain).to(dtype=state.dtype)
        gamma_gain = F.softplus(self.raw_gamma_gain).to(dtype=state.dtype)
        alpha_unit = torch.sigmoid(
            self.alpha_base_logit.to(dtype=state.dtype)
            + alpha_gain * (uncertainty - 0.5)
        )
        gamma_unit = torch.sigmoid(
            self.gamma_base_logit.to(dtype=state.dtype) - gamma_gain * (load - 0.5)
        )
        alpha_t = (
            self.config.alpha_min
            + (self.config.alpha_max - self.config.alpha_min) * alpha_unit
        )
        gamma_t = (
            self.config.gamma_min
            + (self.config.gamma_max - self.config.gamma_min) * gamma_unit
        )
        return alpha_t, gamma_t


def meta_objective(
    rewards: Tensor,
    variance_lambda: float | Tensor = 0.1,
    *,
    dim: int | tuple[int, ...] | None = None,
    keepdim: bool = False,
) -> Tensor:
    """Compute ``mean(reward) - lambda * population_variance(reward)``."""

    if not isinstance(rewards, Tensor) or not torch.is_floating_point(rewards):
        raise TypeError("rewards must be a floating-point tensor")
    if isinstance(variance_lambda, (int, float)) and variance_lambda < 0.0:
        raise ValueError("variance_lambda must be non-negative")
    coefficient = torch.as_tensor(
        variance_lambda, dtype=rewards.dtype, device=rewards.device
    )
    mean = rewards.mean(dim=dim, keepdim=keepdim)
    variance = rewards.var(dim=dim, unbiased=False, keepdim=keepdim)
    return mean - coefficient * variance


@dataclass(frozen=True)
class RoutingDecision:
    """All tensor outputs from one hierarchical routing decision."""

    strategic_mask: Tensor
    tactical_mask: Tensor
    reactive_mask: Tensor
    strategic_probabilities: Tensor
    tactical_probabilities: Tensor
    reactive_probabilities: Tensor
    strategic_logits: Tensor
    tactical_logits: Tensor
    reactive_logits: Tensor
    replan_mask: Tensor
    alpha_t: Tensor
    gamma_t: Tensor

    @property
    def routing_mask(self) -> Tensor:
        """Final module mask after any reactive override."""

        return self.reactive_mask


def _masked_probabilities(
    logits: Tensor, allowed: Tensor, temperature: float
) -> Tensor:
    floor = torch.finfo(logits.dtype).min
    masked_logits = logits.masked_fill(~allowed, floor)
    probabilities = torch.softmax(masked_logits / temperature, dim=-1)
    probabilities = probabilities * allowed.to(dtype=logits.dtype)
    return probabilities / probabilities.sum(dim=-1, keepdim=True).clamp_min(
        torch.finfo(logits.dtype).eps
    )


def _straight_through_top_k(
    logits: Tensor,
    *,
    k: int,
    allowed: Tensor,
    temperature: float,
) -> tuple[Tensor, Tensor]:
    """Return a hard k-hot forward mask with a sigmoid relaxation backward."""

    floor = torch.finfo(logits.dtype).min
    masked_logits = logits.masked_fill(~allowed, floor)
    indices = torch.topk(masked_logits, k=k, dim=-1, sorted=True).indices
    hard = torch.zeros_like(logits).scatter(-1, indices, 1.0)
    hard = hard * allowed.to(dtype=logits.dtype)

    boundary = masked_logits.gather(-1, indices[..., -1:]).detach()
    relaxed = torch.sigmoid((masked_logits - boundary) / temperature)
    relaxed = relaxed * allowed.to(dtype=logits.dtype)
    # Keep the forward value exactly k-hot on every backend. Grouping the
    # zero-valued gradient carrier avoids an associativity-dependent 1 ulp loss.
    mask = hard + (relaxed - relaxed.detach())
    probabilities = _masked_probabilities(logits, allowed, temperature)
    return mask, probabilities


class BioHAMAMetaRouter(nn.Module):
    """Strategic -> tactical -> reactive tensor router.

    Strategic routing reserves a coarse candidate budget.  Tactical routing
    chooses a smaller executable subset only from those candidates.  If
    uncertainty or load reaches its configured limit, the reactive controller
    deterministically replaces that subset with its safety-oriented route.
    """

    def __init__(self, config: MetaRouterConfig | None = None):
        super().__init__()
        self.config = config or MetaRouterConfig()
        hidden = self.config.hidden_dim
        modules = self.config.num_modules
        self.state_encoder = nn.Sequential(
            nn.Linear(COGNITIVE_STATE_DIM, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
        )
        self.strategic_head = nn.Linear(hidden, modules)
        self.tactical_head = nn.Linear(hidden + modules, modules)
        self.reactive_head = nn.Linear(hidden + modules, modules)
        self.neuromodulator = BoundedNeuromodulator(self.config)

    @staticmethod
    def _normalize_state(state: Tensor) -> Tensor:
        if not isinstance(state, Tensor):
            raise TypeError("state must be a tensor")
        if not torch.is_floating_point(state):
            raise TypeError("state must be floating point")
        if state.ndim == 0 or state.shape[-1] != COGNITIVE_STATE_DIM:
            raise ValueError("state must have final dimension 5")
        lower = state.new_tensor((0.0, -1.0, 0.0, 0.0, 0.0))
        upper = state.new_tensor((1.0, 1.0, 1.0, 1.0, 1.0))
        return torch.maximum(lower, torch.minimum(state, upper))

    def compute_neuromodulators(
        self, state: Tensor | CognitiveState
    ) -> tuple[Tensor, Tensor]:
        state_tensor = state.tensor if isinstance(state, CognitiveState) else state
        return self.neuromodulator(self._normalize_state(state_tensor))

    def meta_objective(self, rewards: Tensor) -> Tensor:
        return meta_objective(rewards, self.config.meta_variance_lambda)

    def forward(
        self,
        state: Tensor | CognitiveState,
        available_mask: Tensor | None = None,
    ) -> RoutingDecision:
        state_tensor = state.tensor if isinstance(state, CognitiveState) else state
        return self.route_tensor(state_tensor, available_mask)

    def route_tensor(
        self, state: Tensor, available_mask: Tensor | None = None
    ) -> RoutingDecision:
        """Run the tensor-only hot path without sampling or host synchronization."""

        state = self._normalize_state(state)
        hidden = self.state_encoder(state)
        strategic_logits = self.strategic_head(hidden)
        if available_mask is None:
            available = torch.ones_like(strategic_logits, dtype=torch.bool)
        else:
            if not isinstance(available_mask, Tensor):
                raise TypeError("available_mask must be a tensor")
            available = torch.broadcast_to(
                available_mask.to(device=state.device, dtype=torch.bool),
                strategic_logits.shape,
            )

        strategic_mask, strategic_probabilities = _straight_through_top_k(
            strategic_logits,
            k=self.config.strategic_top_k,
            allowed=available,
            temperature=self.config.routing_temperature,
        )

        tactical_input = torch.cat((hidden, strategic_probabilities), dim=-1)
        tactical_logits = self.tactical_head(tactical_input)
        strategic_candidates = (strategic_mask.detach() > 0.5) & available
        tactical_mask, tactical_probabilities = _straight_through_top_k(
            tactical_logits,
            k=self.config.tactical_top_k,
            allowed=strategic_candidates,
            temperature=self.config.routing_temperature,
        )

        reactive_input = torch.cat((hidden, tactical_probabilities), dim=-1)
        reactive_logits = self.reactive_head(reactive_input)
        override_mask, override_probabilities = _straight_through_top_k(
            reactive_logits,
            k=self.config.reactive_top_k,
            allowed=available,
            temperature=self.config.routing_temperature,
        )

        uncertainty = state[..., 3]
        load = state[..., 4]
        uncertainty_trigger = (
            uncertainty - self.config.reactive_uncertainty_threshold
        ) / max(1.0 - self.config.reactive_uncertainty_threshold, 1e-6)
        load_trigger = (load - self.config.reactive_load_threshold) / max(
            1.0 - self.config.reactive_load_threshold, 1e-6
        )
        trigger_score = torch.maximum(uncertainty_trigger, load_trigger)
        trigger_relaxed = torch.sigmoid(self.config.reactive_sharpness * trigger_score)
        trigger_hard = (
            (uncertainty >= self.config.reactive_uncertainty_threshold)
            | (load >= self.config.reactive_load_threshold)
        ).to(dtype=state.dtype)
        replan_mask = trigger_hard + trigger_relaxed - trigger_relaxed.detach()

        trigger = replan_mask.unsqueeze(-1)
        reactive_mask = tactical_mask + trigger * (override_mask - tactical_mask)
        reactive_probabilities = tactical_probabilities + trigger_relaxed.unsqueeze(
            -1
        ) * (override_probabilities - tactical_probabilities)
        alpha_t, gamma_t = self.neuromodulator(state)
        return RoutingDecision(
            strategic_mask=strategic_mask,
            tactical_mask=tactical_mask,
            reactive_mask=reactive_mask,
            strategic_probabilities=strategic_probabilities,
            tactical_probabilities=tactical_probabilities,
            reactive_probabilities=reactive_probabilities,
            strategic_logits=strategic_logits,
            tactical_logits=tactical_logits,
            reactive_logits=reactive_logits,
            replan_mask=replan_mask,
            alpha_t=alpha_t,
            gamma_t=gamma_t,
        )


# Compatibility names for the architecture terminology used in the source
# materials and master directive.  They all refer to the same implementation.
HierarchicalMetaRouter = BioHAMAMetaRouter
BioAGRPMetaRouter = BioHAMAMetaRouter
BioAGRPOMetaRouter = BioHAMAMetaRouter
Bio_A_GRPO_MetaRouter = BioHAMAMetaRouter
MetaRouter = BioHAMAMetaRouter


__all__ = [
    "BioAGRPMetaRouter",
    "BioAGRPOMetaRouter",
    "BioHAMAMetaRouter",
    "Bio_A_GRPO_MetaRouter",
    "BoundedNeuromodulator",
    "COGNITIVE_STATE_DIM",
    "CognitiveState",
    "HierarchicalMetaRouter",
    "MetaRouter",
    "MetaRouterConfig",
    "RoutingDecision",
    "cognitive_state_tensor",
    "meta_objective",
]
