from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class ExpertBudgetExceeded(RuntimeError):
    """Raised before an expert pool can exceed a configured memory envelope."""


class ExpertContractivityError(RuntimeError):
    """Raised when an expert recurrent operator cannot be made contractive."""


@dataclass(frozen=True)
class ExpertConfig:
    """Hard bounds and routing policy for a System-3 expert pool.

    The pool is allocated once at ``max_experts``. Recruitment only activates
    or recycles a slot, so neither parameters nor persistent VRAM can grow over
    the lifetime of the process.
    """

    input_dim: int
    state_dim: int = 64
    router_dim: int = 32
    max_experts: int = 8
    initial_experts: int = 1
    min_experts: int = 1
    top_k: int = 2
    novelty_threshold: float = 0.8
    recruit_fraction: float = 0.5
    routing_temperature: float = 0.25
    spectral_margin: float = 0.90
    usage_decay: float = 0.95
    prune_usage_threshold: float = 0.01
    minimum_age: int = 2
    merge_on_capacity: bool = True
    balance_coefficient: float = 0.01
    z_loss_coefficient: float = 0.001
    max_parameter_bytes: int = 8 * 1024**3
    max_vram_bytes: int = int(16.7 * 1024**3)
    backward_workspace_multiplier: int = 3
    seed: int = 0

    def __post_init__(self) -> None:
        positive = {
            "input_dim": self.input_dim,
            "state_dim": self.state_dim,
            "router_dim": self.router_dim,
            "max_experts": self.max_experts,
            "initial_experts": self.initial_experts,
            "min_experts": self.min_experts,
            "top_k": self.top_k,
            "minimum_age": self.minimum_age,
            "max_parameter_bytes": self.max_parameter_bytes,
            "max_vram_bytes": self.max_vram_bytes,
            "backward_workspace_multiplier": self.backward_workspace_multiplier,
        }
        for name, value in positive.items():
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        if not self.min_experts <= self.initial_experts <= self.max_experts:
            raise ValueError("expected min_experts <= initial_experts <= max_experts")
        if self.top_k > self.max_experts:
            raise ValueError("top_k cannot exceed max_experts")
        if not -1.0 <= self.novelty_threshold <= 1.0:
            raise ValueError("novelty_threshold must be a cosine similarity in [-1, 1]")
        if not 0.0 < self.recruit_fraction <= 1.0:
            raise ValueError("recruit_fraction must lie in (0, 1]")
        if self.routing_temperature <= 0.0:
            raise ValueError("routing_temperature must be positive")
        if not 0.0 < self.spectral_margin < 1.0:
            raise ValueError("spectral_margin must lie in (0, 1)")
        if not 0.0 <= self.usage_decay < 1.0:
            raise ValueError("usage_decay must lie in [0, 1)")
        if self.prune_usage_threshold < 0.0:
            raise ValueError("prune_usage_threshold must be non-negative")
        if self.balance_coefficient < 0.0 or self.z_loss_coefficient < 0.0:
            raise ValueError("router loss coefficients must be non-negative")


@dataclass(frozen=True)
class RouterOutput:
    """Tensor-only result of z-independent routing."""

    embedding: Tensor
    similarities: Tensor
    gates: Tensor
    top_indices: Tensor
    top_weights: Tensor
    novelty: Tensor
    novelty_score: Tensor
    balance_loss: Tensor
    z_loss: Tensor
    auxiliary_loss: Tensor


@dataclass(frozen=True)
class ExpertOutput:
    state: Tensor
    routing: RouterOutput


@dataclass(frozen=True)
class MaintenanceResult:
    action: Tensor
    kept_index: Tensor
    released_index: Tensor


@dataclass(frozen=True)
class RecruitmentResult:
    status: Tensor
    slot: Tensor
    novelty_fraction: Tensor
    maintenance_action: Tensor


# Integer tensor codes keep control-plane results serialisation-free.
MAINTENANCE_NONE = 0
MAINTENANCE_PRUNED = 1
MAINTENANCE_MERGED = 2

RECRUITMENT_NOT_NOVEL = 0
RECRUITMENT_ADDED = 1
RECRUITMENT_AFTER_PRUNE = 2
RECRUITMENT_AFTER_MERGE = 3
RECRUITMENT_CAPACITY_BLOCKED = 4


def _tensor_nbytes(tensor: Tensor) -> int:
    return tensor.numel() * tensor.element_size()


class BoundedSparseImplicitExperts(nn.Module):
    """Bounded z-independent Contractive Gated Mixture (CGM).

    ``route(x)`` never receives the transient fixed-point state. A DEQ solver
    can therefore compute its :class:`RouterOutput` once and repeatedly call
    ``mixture(z, x, routing)`` without introducing router derivatives into the
    state Jacobian. Only the selected top-k expert matrices are materialised.

    Expert storage is preallocated. This deliberately trades a known, admitted
    persistent allocation for a proof that recruitment cannot cause unbounded
    parameter or VRAM growth on a 24 GB device.
    """

    def __init__(self, config: ExpertConfig):
        super().__init__()
        self.config = config
        estimated = self.estimated_parameter_bytes(config)
        if estimated > config.max_parameter_bytes:
            raise ExpertBudgetExceeded(
                "expert parameter admission rejected: "
                f"required={estimated}, limit={config.max_parameter_bytes}"
            )

        m, d, d_in, d_router = (
            config.max_experts,
            config.state_dim,
            config.input_dim,
            config.router_dim,
        )
        self.router_weight = nn.Parameter(torch.empty(d_router, d_in))
        self.router_bias = nn.Parameter(torch.zeros(d_router))
        self.prototypes = nn.Parameter(torch.zeros(m, d_router))
        self.recurrent = nn.Parameter(torch.zeros(m, d, d))
        self.input_weight = nn.Parameter(torch.zeros(m, d, d_in))
        self.bias = nn.Parameter(torch.zeros(m, d))

        self.register_buffer("active_mask", torch.zeros(m, dtype=torch.bool))
        self.register_buffer("usage_ema", torch.zeros(m))
        self.register_buffer("dispatch_count", torch.zeros(m, dtype=torch.long))
        self.register_buffer("expert_age", torch.zeros(m, dtype=torch.long))
        self.register_buffer("usage_updates", torch.zeros((), dtype=torch.long))
        self.register_buffer("recruitment_count", torch.zeros((), dtype=torch.long))
        self.register_buffer("merge_count", torch.zeros((), dtype=torch.long))
        self.register_buffer("prune_count", torch.zeros((), dtype=torch.long))

        with torch.no_grad():
            self.router_weight.copy_(
                self._deterministic_matrix(
                    -1,
                    d_router,
                    d_in,
                    self.router_weight.device,
                    self.router_weight.dtype,
                )
            )
            for slot in range(config.initial_experts):
                self._activate_slot_(slot, self._deterministic_prototype(slot))
            self.recruitment_count.fill_(config.initial_experts)
        self._assert_current_budgets()

    @staticmethod
    def estimated_parameter_bytes(config: ExpertConfig) -> int:
        """Exact constructor-time parameter bytes at the default dtype."""

        m, d, d_in, r = (
            config.max_experts,
            config.state_dim,
            config.input_dim,
            config.router_dim,
        )
        elements = r * d_in + r + m * r + m * d * d + m * d * d_in + m * d
        element_size = torch.empty((), dtype=torch.get_default_dtype()).element_size()
        return elements * element_size

    @property
    def parameter_bytes(self) -> int:
        return sum(_tensor_nbytes(parameter) for parameter in self.parameters())

    @property
    def persistent_bytes(self) -> int:
        return self.parameter_bytes + sum(
            _tensor_nbytes(buffer) for buffer in self.buffers()
        )

    @property
    def active_experts(self) -> int:
        """Control-plane scalar; the routing hot path uses ``active_mask``."""

        return int(self.active_mask.sum().detach().cpu())

    @staticmethod
    def _deterministic_matrix(
        slot: int,
        rows: int,
        columns: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tensor:
        row = torch.arange(1, rows + 1, device=device, dtype=torch.float64)[:, None]
        column = torch.arange(1, columns + 1, device=device, dtype=torch.float64)[
            None, :
        ]
        phase = float(slot + 2)
        matrix = torch.sin(row * column * (0.017 * phase))
        matrix = matrix / math.sqrt(max(columns, 1))
        return matrix.to(dtype=dtype)

    def _deterministic_prototype(self, slot: int) -> Tensor:
        index = torch.arange(
            1,
            self.config.router_dim + 1,
            device=self.prototypes.device,
            dtype=torch.float64,
        )
        vector = torch.sin(index * (slot + 1) * 0.731)
        return F.normalize(vector.to(self.prototypes.dtype), dim=0)

    @torch.no_grad()
    def _reset_expert_(self, slot: int) -> None:
        recurrent = self._deterministic_matrix(
            slot,
            self.config.state_dim,
            self.config.state_dim,
            self.recurrent.device,
            self.recurrent.dtype,
        )
        input_weight = self._deterministic_matrix(
            slot + self.config.max_experts,
            self.config.state_dim,
            self.config.input_dim,
            self.input_weight.device,
            self.input_weight.dtype,
        )
        self.recurrent[slot].copy_(recurrent)
        self.input_weight[slot].copy_(input_weight)
        self.bias[slot].zero_()
        self._project_slot_(slot)

    @torch.no_grad()
    def _project_slot_(self, slot: int) -> None:
        weight = self.recurrent[slot]
        sigma = torch.linalg.matrix_norm(weight, ord=2).clamp_min(1e-12)
        margin = weight.new_tensor(self.config.spectral_margin * (1.0 - 1e-5))
        weight.mul_(torch.clamp(margin / sigma, max=1.0))

    @torch.no_grad()
    def project_contractivity_(self) -> None:
        """C-FIRE safety projection for every active recurrent operator."""

        norms = torch.linalg.matrix_norm(self.recurrent, ord=2).clamp_min(1e-12)
        target = self.recurrent.new_tensor(self.config.spectral_margin * (1.0 - 1e-5))
        scales = torch.clamp(target / norms, max=1.0)
        scales = torch.where(self.active_mask, scales, torch.ones_like(scales))
        self.recurrent.mul_(scales[:, None, None])

    @torch.no_grad()
    def _ensure_contractivity_(self) -> None:
        """Project if needed and verify the recurrent C-FIRE postcondition."""

        if not torch.isfinite(self.recurrent[self.active_mask]).all():
            raise ExpertContractivityError(
                "active expert has a non-finite recurrent operator"
            )
        norms = torch.linalg.matrix_norm(self.recurrent, ord=2)
        active_norms = norms[self.active_mask]
        if not torch.isfinite(active_norms).all():
            raise ExpertContractivityError(
                "active expert has a non-finite recurrent spectral norm"
            )
        if bool((active_norms >= self.config.spectral_margin).any().detach().cpu()):
            self.project_contractivity_()
            active_norms = torch.linalg.matrix_norm(self.recurrent, ord=2)[
                self.active_mask
            ]
        if not torch.isfinite(active_norms).all() or bool(
            (active_norms >= self.config.spectral_margin).any().detach().cpu()
        ):
            raise ExpertContractivityError(
                "expert C-FIRE projection failed its strict spectral postcondition"
            )

    def expert_spectral_norms(self) -> Tensor:
        norms = torch.linalg.matrix_norm(self.recurrent, ord=2)
        return torch.where(self.active_mask, norms, torch.zeros_like(norms))

    @torch.no_grad()
    def _activate_slot_(self, slot: int, prototype: Tensor) -> None:
        self._reset_expert_(slot)
        normalized = F.normalize(prototype.to(self.prototypes), dim=0)
        if bool((normalized.norm() <= 1e-8).detach().cpu()):
            normalized = self._deterministic_prototype(slot)
        self.prototypes[slot].copy_(normalized)
        self.active_mask[slot] = True
        self.usage_ema[slot] = 0
        self.dispatch_count[slot] = 0
        self.expert_age[slot] = 0

    @torch.no_grad()
    def _deactivate_slot_(self, slot: int) -> None:
        self.active_mask[slot] = False
        self.prototypes[slot].zero_()
        self.recurrent[slot].zero_()
        self.input_weight[slot].zero_()
        self.bias[slot].zero_()
        self.usage_ema[slot] = 0
        self.dispatch_count[slot] = 0
        self.expert_age[slot] = 0

    def _assert_current_budgets(self) -> None:
        parameter_bytes = self.parameter_bytes
        if parameter_bytes > self.config.max_parameter_bytes:
            raise ExpertBudgetExceeded(
                "expert parameter budget exceeded: "
                f"required={parameter_bytes}, limit={self.config.max_parameter_bytes}"
            )
        persistent = self.persistent_bytes
        if persistent > self.config.max_vram_bytes:
            raise ExpertBudgetExceeded(
                "expert persistent VRAM budget exceeded: "
                f"required={persistent}, limit={self.config.max_vram_bytes}"
            )

    def estimated_working_set_bytes(
        self,
        batch_size: int,
        *,
        element_size: int | None = None,
        include_backward: bool = False,
    ) -> int:
        """Conservative top-k workspace, independent of active expert count."""

        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        size = element_size or self.recurrent.element_size()
        b, k, m, d, d_in, r = (
            batch_size,
            self.config.top_k,
            self.config.max_experts,
            self.config.state_dim,
            self.config.input_dim,
            self.config.router_dim,
        )
        float_elements = (
            b * (d + d_in)  # caller-owned z and x, conservatively included
            + b * r  # router embedding
            + 3 * b * m  # similarities, masked logits, full gates
            + b * k  # top-k weights
            + b * k * d * d  # selected recurrent matrices
            + b * k * d * d_in  # selected input matrices
            + 4 * b * k * d  # bias, two drives, expert state
            + b * d  # mixture output
        )
        index_bytes = b * k * torch.empty((), dtype=torch.long).element_size()
        workspace = float_elements * size + index_bytes
        if include_backward:
            workspace *= self.config.backward_workspace_multiplier
        return workspace

    def _admit_forward(self, z: Tensor, x: Tensor) -> None:
        self._assert_current_budgets()
        include_backward = self.training and torch.is_grad_enabled()
        workspace = self.estimated_working_set_bytes(
            x.shape[0],
            element_size=x.element_size(),
            include_backward=include_backward,
        )
        if x.is_cuda:
            resident = torch.cuda.memory_allocated(x.device)
        else:
            resident = self.persistent_bytes
        required = resident + workspace
        if required > self.config.max_vram_bytes:
            raise ExpertBudgetExceeded(
                "expert forward VRAM admission rejected: "
                f"required={required}, limit={self.config.max_vram_bytes}"
            )

    def route(self, x: Tensor) -> RouterOutput:
        """Route solely from input ``x``; no fixed-point state is accepted."""

        if x.ndim != 2 or x.shape[-1] != self.config.input_dim or x.shape[0] == 0:
            raise ValueError(
                f"x must have non-empty shape [batch, {self.config.input_dim}]"
            )
        embedding = F.normalize(
            F.linear(x, self.router_weight, self.router_bias), dim=-1
        )
        prototypes = F.normalize(self.prototypes, dim=-1)
        similarities = embedding @ prototypes.transpose(0, 1)
        active = self.active_mask[None, :]
        masked = similarities.masked_fill(~active, -torch.inf)
        top_values, top_indices = torch.topk(masked, self.config.top_k, dim=-1)
        top_weights = torch.softmax(
            top_values / self.config.routing_temperature, dim=-1
        )
        gates = torch.zeros_like(similarities).scatter(1, top_indices, top_weights)

        max_similarity = masked.max(dim=-1).values
        novelty = max_similarity < self.config.novelty_threshold
        novelty_score = 1.0 - max_similarity

        # Switch-style balance objective: 1.0 is balanced, larger is collapsed.
        selected = gates.gt(0).to(gates.dtype)
        fractions = selected.sum(0) / selected.sum().clamp_min(1.0)
        probabilities = gates.mean(0)
        active_count = self.active_mask.sum().to(gates.dtype)
        balance_loss = active_count * (fractions * probabilities).sum()
        z_loss = (
            torch.logsumexp(masked / self.config.routing_temperature, dim=-1)
            .square()
            .mean()
        )
        auxiliary_loss = (
            self.config.balance_coefficient * balance_loss
            + self.config.z_loss_coefficient * z_loss
        )
        return RouterOutput(
            embedding,
            similarities,
            gates,
            top_indices,
            top_weights,
            novelty,
            novelty_score,
            balance_loss,
            z_loss,
            auxiliary_loss,
        )

    def mixture(self, z: Tensor, x: Tensor, routing: RouterOutput) -> Tensor:
        """Apply only top-k selected experts using precomputed input gates."""

        if z.ndim != 2 or z.shape != (x.shape[0], self.config.state_dim):
            raise ValueError(f"z must have shape [batch, {self.config.state_dim}]")
        # This is the last safety boundary before a recurrent tensor operation.
        # Direct ``mixture`` callers therefore cannot bypass C-FIRE by skipping
        # the higher-level ``forward`` method.
        self._ensure_contractivity_()
        indices = routing.top_indices
        weights = routing.top_weights
        recurrent = self.recurrent[indices]
        input_weight = self.input_weight[indices]
        bias = self.bias[indices]
        recurrent_drive = torch.einsum("bkoi,bi->bko", recurrent, z)
        input_drive = torch.einsum("bkoi,bi->bko", input_weight, x)
        expert_state = torch.tanh(recurrent_drive + input_drive + bias)
        return (weights[..., None] * expert_state).sum(dim=1)

    def forward(
        self, z: Tensor, x: Tensor, *, track_usage: bool | None = None
    ) -> ExpertOutput:
        self._admit_forward(z, x)
        routing = self.route(x)
        state = self.mixture(z, x, routing)
        should_track = self.training if track_usage is None else track_usage
        if should_track:
            self.update_usage_(routing)
        return ExpertOutput(state, routing)

    @torch.no_grad()
    def update_usage_(self, routing: RouterOutput) -> None:
        """Update bounded EMA/count buffers once per routed batch."""

        mass = routing.gates.detach().mean(0)
        first = self.usage_updates.eq(0)
        updated = (
            self.config.usage_decay * self.usage_ema
            + (1.0 - self.config.usage_decay) * mass
        )
        self.usage_ema.copy_(torch.where(first, mass, updated))
        self.usage_ema.mul_(self.active_mask)
        self.dispatch_count.add_(routing.gates.detach().gt(0).sum(0))
        self.expert_age.add_(self.active_mask.to(self.expert_age.dtype))
        self.usage_updates.add_(1)

    def routing_contractivity_bound(self, routing: RouterOutput) -> Tensor:
        """Per-sample convex upper bound on the CGM state Jacobian norm."""

        return routing.gates @ self.expert_spectral_norms()

    def _control_result(
        self, action: int, kept: int = -1, released: int = -1
    ) -> MaintenanceResult:
        device = self.active_mask.device
        return MaintenanceResult(
            torch.tensor(action, device=device, dtype=torch.long),
            torch.tensor(kept, device=device, dtype=torch.long),
            torch.tensor(released, device=device, dtype=torch.long),
        )

    @torch.no_grad()
    def maintain_(self, *, force_merge: bool = False) -> MaintenanceResult:
        """Deterministically prune a cold slot or merge the closest pair.

        A Fisher-aware lifecycle may call this at a domain boundary. Convexly
        merging recurrent matrices preserves the contractivity upper bound;
        an explicit projection is still applied as a numerical postcondition.
        """

        active_count = self.active_experts
        if active_count <= self.config.min_experts:
            return self._control_result(MAINTENANCE_NONE)

        eligible = self.active_mask & (self.expert_age >= self.config.minimum_age)
        scores = torch.where(
            eligible,
            self.usage_ema,
            torch.full_like(self.usage_ema, torch.inf),
        )
        candidate = int(torch.argmin(scores).detach().cpu())
        candidate_score = scores[candidate]
        if bool(
            (
                torch.isfinite(candidate_score)
                & (candidate_score <= self.config.prune_usage_threshold)
            )
            .detach()
            .cpu()
        ):
            self._deactivate_slot_(candidate)
            self.prune_count.add_(1)
            return self._control_result(MAINTENANCE_PRUNED, released=candidate)

        if not force_merge or active_count < 2:
            return self._control_result(MAINTENANCE_NONE)

        prototypes = F.normalize(self.prototypes, dim=-1)
        similarities = prototypes @ prototypes.T
        upper = torch.triu(torch.ones_like(similarities, dtype=torch.bool), diagonal=1)
        valid = upper & self.active_mask[:, None] & self.active_mask[None, :]
        pair_scores = similarities.masked_fill(~valid, -torch.inf)
        flat_index = int(torch.argmax(pair_scores).detach().cpu())
        keep = flat_index // self.config.max_experts
        release = flat_index % self.config.max_experts

        total_usage = self.usage_ema[keep] + self.usage_ema[release]
        alpha = torch.where(
            total_usage > 1e-12,
            self.usage_ema[keep] / total_usage.clamp_min(1e-12),
            total_usage.new_tensor(0.5),
        )
        self.recurrent[keep].lerp_(self.recurrent[release], 1.0 - alpha)
        self.input_weight[keep].lerp_(self.input_weight[release], 1.0 - alpha)
        self.bias[keep].lerp_(self.bias[release], 1.0 - alpha)
        merged_prototype = (
            alpha * self.prototypes[keep] + (1.0 - alpha) * self.prototypes[release]
        )
        if bool((merged_prototype.norm() <= 1e-8).detach().cpu()):
            merged_prototype = self._deterministic_prototype(keep)
        self.prototypes[keep].copy_(F.normalize(merged_prototype, dim=0))
        self.usage_ema[keep] = total_usage.clamp_max(1.0)
        self.dispatch_count[keep].add_(self.dispatch_count[release])
        self.expert_age[keep] = torch.maximum(
            self.expert_age[keep], self.expert_age[release]
        )
        self._project_slot_(keep)
        self._deactivate_slot_(release)
        self.merge_count.add_(1)
        return self._control_result(MAINTENANCE_MERGED, keep, release)

    @torch.no_grad()
    def recruit_(self, x: Tensor) -> RecruitmentResult:
        """Apply novelty-triggered R2P without ever increasing pool storage."""

        self._assert_current_budgets()
        routing = self.route(x)
        novelty_fraction = routing.novelty.to(x.dtype).mean()
        device = self.active_mask.device
        if bool((novelty_fraction < self.config.recruit_fraction).detach().cpu()):
            return RecruitmentResult(
                torch.tensor(RECRUITMENT_NOT_NOVEL, device=device),
                torch.tensor(-1, device=device),
                novelty_fraction,
                torch.tensor(MAINTENANCE_NONE, device=device),
            )

        maintenance = self._control_result(MAINTENANCE_NONE)
        if self.active_experts >= self.config.max_experts:
            maintenance = self.maintain_(force_merge=self.config.merge_on_capacity)
        free = ~self.active_mask
        if not bool(free.any().detach().cpu()):
            return RecruitmentResult(
                torch.tensor(RECRUITMENT_CAPACITY_BLOCKED, device=device),
                torch.tensor(-1, device=device),
                novelty_fraction,
                maintenance.action,
            )

        slot = int(torch.argmax(free.to(torch.long)).detach().cpu())
        novel_weights = routing.novelty.to(routing.embedding.dtype)[:, None]
        prototype = (routing.embedding * novel_weights).sum(
            0
        ) / novel_weights.sum().clamp_min(1.0)
        self._activate_slot_(slot, prototype)
        self.recruitment_count.add_(1)
        self._assert_current_budgets()

        action = int(maintenance.action.detach().cpu())
        if action == MAINTENANCE_PRUNED:
            status = RECRUITMENT_AFTER_PRUNE
        elif action == MAINTENANCE_MERGED:
            status = RECRUITMENT_AFTER_MERGE
        else:
            status = RECRUITMENT_ADDED
        return RecruitmentResult(
            torch.tensor(status, device=device),
            torch.tensor(slot, device=device),
            novelty_fraction,
            maintenance.action,
        )


__all__ = [
    "BoundedSparseImplicitExperts",
    "ExpertBudgetExceeded",
    "ExpertConfig",
    "ExpertContractivityError",
    "ExpertOutput",
    "MAINTENANCE_MERGED",
    "MAINTENANCE_NONE",
    "MAINTENANCE_PRUNED",
    "MaintenanceResult",
    "RECRUITMENT_ADDED",
    "RECRUITMENT_AFTER_MERGE",
    "RECRUITMENT_AFTER_PRUNE",
    "RECRUITMENT_CAPACITY_BLOCKED",
    "RECRUITMENT_NOT_NOVEL",
    "RecruitmentResult",
    "RouterOutput",
]
