from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn

from .backbone import extract_hidden_states
from .fp_ewc import verified_spectral_cap_


@dataclass(frozen=True)
class FastWeightOverlay:
    """Batched low-rank overlay emitted by :class:`FastWeightProgrammer`."""

    a: Tensor
    b: Tensor
    quality: Tensor

    @property
    def delta(self) -> Tensor:
        """Materialise the small target-space update for diagnostics only."""

        return self.b @ self.a.transpose(-1, -2)


class FastWeightProgrammer(nn.Module):
    """Compile a source latent into a bounded target-space rank-r overlay.

    The original API used one ``hidden_dim`` for the source, trunk, and target
    matrices.  That construction grows a head to ``hidden_dim**2 * rank``.
    ``source_dim``, ``target_dim``, and ``internal_dim`` are now independent;
    the largest learned projections are O(source * internal) and
    O(internal * target * rank).  Passing only ``hidden_dim`` remains supported.
    """

    MAX_INTERNAL_DIM = 512
    # Legacy single-dimension callers may use a full Gemma latent (for example
    # 2,560).  It remains finite while production factory wiring uses the much
    # smaller adapter bottleneck below.
    MAX_TARGET_DIM = 4096
    MAX_RANK = 64

    def __init__(
        self,
        hidden_dim: int | None = None,
        rank: int = 8,
        max_operator_norm: float = 0.1,
        *,
        source_dim: int | None = None,
        target_dim: int | None = None,
        internal_dim: int | None = None,
    ) -> None:
        super().__init__()
        if source_dim is None:
            source_dim = hidden_dim
        elif hidden_dim is not None and hidden_dim != source_dim:
            raise ValueError("hidden_dim and source_dim disagree")
        if source_dim is None or source_dim <= 0:
            raise ValueError("source_dim must be positive")
        target_dim = source_dim if target_dim is None else target_dim
        internal_dim = min(source_dim, 256) if internal_dim is None else internal_dim
        if not 0 < target_dim <= self.MAX_TARGET_DIM:
            raise ValueError(f"target_dim must be in [1, {self.MAX_TARGET_DIM}]")
        if not 0 < internal_dim <= self.MAX_INTERNAL_DIM:
            raise ValueError(f"internal_dim must be in [1, {self.MAX_INTERNAL_DIM}]")
        if not 0 < rank <= min(self.MAX_RANK, target_dim):
            raise ValueError("rank must be positive and no larger than target_dim")
        if max_operator_norm <= 0.0:
            raise ValueError("max_operator_norm must be positive")

        # ``hidden_dim`` is retained as a compatibility alias for callers that
        # inspect it.  It denotes the source latent dimension, as before.
        self.hidden_dim = source_dim
        self.source_dim = source_dim
        self.target_dim = target_dim
        self.internal_dim = internal_dim
        self.rank = rank
        self.max_operator_norm = float(max_operator_norm)
        self.trunk = nn.Sequential(
            nn.Linear(source_dim, internal_dim),
            nn.GELU(),
            nn.Linear(internal_dim, internal_dim),
            nn.GELU(),
        )
        self.to_a = nn.Linear(internal_dim, target_dim * rank)
        self.to_b = nn.Linear(internal_dim, target_dim * rank)
        self.quality_gate = nn.Sequential(nn.Linear(internal_dim, 1), nn.Sigmoid())

    def estimated_workspace_bytes(self, z_star: Tensor) -> int:
        """Conservative batch-one activation estimate for VRAM admission."""

        element_size = z_star.element_size()
        sequence = z_star.shape[-2] if z_star.ndim == 3 else 1
        values = (
            sequence * self.source_dim
            + (2 * self.internal_dim)
            + (2 * self.target_dim * self.rank)
            + (2 * self.rank * self.rank)
        )
        # Account for allocator/intermediate overlap without making the bound
        # depend on an unbounded request batch.
        return int(values * element_size * 3)

    @staticmethod
    def _low_rank_norm(a: Tensor, b: Tensor) -> Tensor:
        """Compute ``||B A^T||_2`` through rank-sized QR factors."""

        _, r_a = torch.linalg.qr(a, mode="reduced")
        _, r_b = torch.linalg.qr(b, mode="reduced")
        middle = r_b @ r_a.transpose(-1, -2)
        return torch.linalg.matrix_norm(middle, ord=2)

    def forward(self, z_star: Tensor) -> FastWeightOverlay:
        if z_star.ndim not in {1, 2, 3}:
            raise ValueError("z_star must have shape [d], [batch,d], or [batch,seq,d]")
        if z_star.shape[-1] != self.source_dim:
            raise ValueError(
                f"z_star width {z_star.shape[-1]} != source_dim {self.source_dim}"
            )
        pooled = z_star.mean(dim=-2) if z_star.ndim == 3 else z_star
        if pooled.ndim == 1:
            pooled = pooled.unsqueeze(0)
        h = self.trunk(pooled)
        a = self.to_a(h).reshape(-1, self.target_dim, self.rank)
        b = self.to_b(h).reshape(-1, self.target_dim, self.rank)
        sigma = self._low_rank_norm(a.float(), b.float()).clamp_min(1e-8)
        scale = torch.clamp(self.max_operator_norm / sigma, max=1.0)
        b = b * scale.to(dtype=b.dtype)[:, None, None]
        return FastWeightOverlay(a, b, self.quality_gate(h).squeeze(-1))


class ResidualBottleneckAdapter(nn.Module):
    """Contractive residual adapter whose square core accepts fast weights."""

    MAX_BOTTLENECK_DIM = 512

    def __init__(
        self,
        latent_dim: int,
        bottleneck_dim: int,
        *,
        core_operator_norm_budget: float = 0.84,
        spectral_margin: float = 0.95,
        residual_scale: float = 0.1,
    ) -> None:
        super().__init__()
        if latent_dim <= 0 or bottleneck_dim <= 0:
            raise ValueError("latent and bottleneck dimensions must be positive")
        if bottleneck_dim > self.MAX_BOTTLENECK_DIM:
            raise ValueError("bottleneck_dim exceeds the hard target-space bound")
        if not 0.0 < core_operator_norm_budget < spectral_margin < 1.0:
            raise ValueError(
                "core_operator_norm_budget must be below spectral_margin < 1"
            )
        if not 0.0 < residual_scale <= 1.0:
            raise ValueError("residual_scale must be in (0, 1]")
        self.latent_dim = latent_dim
        self.bottleneck_dim = bottleneck_dim
        self.core_operator_norm_budget = float(core_operator_norm_budget)
        self.spectral_margin = float(spectral_margin)
        self.residual_scale = float(residual_scale)
        self.down = nn.Linear(latent_dim, bottleneck_dim, bias=False)
        self.core = nn.Linear(bottleneck_dim, bottleneck_dim, bias=False)
        self.up = nn.Linear(bottleneck_dim, latent_dim, bias=False)
        self.activation = nn.GELU()
        # An unverified adapter must be an exact identity.  A trained,
        # digest-verified System 1.5 checkpoint may later replace this zero
        # projection, but random initialization is never allowed to alter the
        # authoritative Gemma/CTS latent merely because the feature is idle.
        nn.init.zeros_(self.up.weight)
        self.c_fire_()

    @torch.no_grad()
    def c_fire_(self) -> float:
        """Project the recurrent/square core below its strict safety budget."""

        norm = verified_spectral_cap_(self.core.weight, self.core_operator_norm_budget)
        if norm >= self.spectral_margin:
            raise RuntimeError("C-FIRE failed to certify the bottleneck core")
        return norm

    def forward(self, latent: Tensor) -> Tensor:
        if latent.shape[-1] != self.latent_dim:
            raise ValueError(
                f"backbone latent width {latent.shape[-1]} != {self.latent_dim}"
            )
        # Inference is read-only: C-FIRE projection belongs to checkpoint
        # admission/optimizer commit.  Re-projecting here would silently mutate
        # the base adapter on every request.  We instead re-certify immediately
        # before the square tensor operation and fail closed on drift.
        with torch.no_grad():
            norm = float(torch.linalg.matrix_norm(self.core.weight.float(), ord=2))
        if not torch.isfinite(torch.tensor(norm)) or norm > (
            self.core_operator_norm_budget + 1.0e-6
        ):
            raise RuntimeError("adapter core lost its C-FIRE certificate")
        adapted = self.up(
            self.activation(self.core(self.activation(self.down(latent))))
        )
        return latent + self.residual_scale * adapted


class FastWeightBackboneWrapper(nn.Module):
    """Attach a persistent adapter/programmer pair to a latent backbone.

    Both modules are registered below this wrapper, so the ordinary backbone
    state dict and Genesis checkpoint include their parameters atomically.
    """

    TARGET_MODULE = "adapter.core"

    def __init__(
        self,
        backbone: nn.Module,
        adapter: ResidualBottleneckAdapter,
        programmer: FastWeightProgrammer,
    ) -> None:
        super().__init__()
        if adapter.latent_dim != programmer.source_dim:
            raise ValueError("adapter latent_dim and programmer source_dim disagree")
        if adapter.bottleneck_dim != programmer.target_dim:
            raise ValueError(
                "adapter bottleneck and programmer target dimensions disagree"
            )
        self.backbone = backbone
        self.adapter = adapter
        self.programmer = programmer

    def forward(self, *args: Any, **kwargs: Any) -> Tensor:
        output = self.backbone(*args, **kwargs)
        return self.adapter(extract_hidden_states(output))
