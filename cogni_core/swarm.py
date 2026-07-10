from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn


class SwarmContractivityError(RuntimeError):
    """Raised when a swarm recurrent operator cannot satisfy C-FIRE."""


@dataclass(frozen=True)
class SwarmConfig:
    input_dim: int
    state_dim: int = 64
    agents: int = 28
    sensory_agents: int = 11
    constraint_agents: int = 7
    local_margin: float = 0.90
    coupling_scale: float = 0.05
    cold_steps: int = 24
    warm_steps: int = 10


@dataclass
class SwarmOutput:
    latent: Tensor
    joint_state: Tensor
    regime: Tensor
    residual: Tensor
    iterations: Tensor


def _project_matrix(weight: Tensor, margin: float) -> Tensor:
    sigma = torch.linalg.matrix_norm(weight, ord=2).clamp_min(1e-8)
    # Keep a small strict interior margin to survive round-off after SVD.
    target = margin * (1.0 - 1e-5)
    scale = torch.clamp(
        torch.as_tensor(target, device=weight.device, dtype=weight.dtype) / sigma,
        max=1.0,
    )
    return weight * scale[..., None, None]


class PCASMonitor(nn.Module):
    """Pre-calibrated Mahalanobis regime detector with a tensor-only hot path."""

    def __init__(self, input_dim: int):
        super().__init__()
        self.register_buffer("mean", torch.zeros(input_dim))
        self.register_buffer("cov_inv", torch.eye(input_dim))
        self.register_buffer("threshold", torch.tensor(float("inf")))
        self.register_buffer("calibrated", torch.tensor(False))

    @torch.no_grad()
    def fit(
        self, observations: Tensor, quantile: float = 0.99, regularization: float = 1e-4
    ) -> None:
        if observations.ndim != 2 or observations.shape[0] < 2:
            raise ValueError(
                "calibration requires [samples, features] with at least two samples"
            )
        mean = observations.mean(0)
        centered = observations - mean
        cov = centered.T @ centered / (observations.shape[0] - 1)
        cov = cov + regularization * torch.eye(
            cov.shape[0], device=cov.device, dtype=cov.dtype
        )
        cov_inv = torch.linalg.pinv(cov)
        distances = torch.sqrt(((centered @ cov_inv) * centered).sum(-1).clamp_min(0))
        self.mean.copy_(mean)
        self.cov_inv.copy_(cov_inv)
        self.threshold.copy_(torch.quantile(distances, quantile))
        self.calibrated.fill_(True)

    def forward(self, observations: Tensor) -> tuple[Tensor, Tensor]:
        centered = observations - self.mean
        distance = torch.sqrt(
            ((centered @ self.cov_inv) * centered).sum(-1).clamp_min(0)
        )
        # Uncalibrated monitors remain in the normal regime; no Python scalar extraction.
        regime = ((distance.mean() > self.threshold) & self.calibrated).to(torch.long)
        return regime, distance


class TensorSwarm(nn.Module):
    """Gradient-free System-4 style coupled DEQ swarm.

    Communication stays in ``[batch, agent, state]`` tensors. Both precompiled
    topologies are strictly lower triangular, so cross-agent coupling cannot
    introduce unstable eigenvalues into the global block Jacobian.
    """

    def __init__(self, config: SwarmConfig):
        super().__init__()
        if not 0 < config.sensory_agents <= config.agents:
            raise ValueError("invalid sensory agent count")
        if not 0 < config.constraint_agents <= config.agents:
            raise ValueError("invalid constraint agent count")
        self.config = config
        n, d = config.agents, config.state_dim
        self.input_projection = nn.Parameter(
            torch.empty(config.sensory_agents, d, config.input_dim)
        )
        self.recurrent = nn.Parameter(torch.empty(n, d, d))
        self.coupling = nn.Parameter(torch.empty(n, d, d))
        self.bias = nn.Parameter(torch.zeros(n, d))
        nn.init.xavier_uniform_(self.input_projection)
        nn.init.orthogonal_(self.recurrent)
        nn.init.orthogonal_(self.coupling)
        with torch.no_grad():
            self.recurrent.copy_(_project_matrix(self.recurrent, config.local_margin))
            self.coupling.copy_(_project_matrix(self.coupling, config.coupling_scale))
        self.register_buffer("normal_topology", self._normal_topology(n))
        self.register_buffer("crisis_topology", self._crisis_topology(n))
        self.monitor = PCASMonitor(config.input_dim)

    @staticmethod
    def _normal_topology(n: int) -> Tensor:
        topology = torch.tril(torch.ones(n, n), diagonal=-1)
        # Bounded fan-in keeps coupling magnitude independent of agent count.
        for row in range(n):
            active = torch.nonzero(topology[row], as_tuple=False).flatten()
            if len(active) > 3:
                topology[row, active[:-3]] = 0
        return topology

    @staticmethod
    def _crisis_topology(n: int) -> Tensor:
        topology = torch.zeros(n, n)
        for row in range(1, n):
            topology[row, row - 1] = 1
        return topology

    @torch.no_grad()
    def project_contractivity_(self) -> None:
        self.recurrent.copy_(_project_matrix(self.recurrent, self.config.local_margin))
        self.coupling.copy_(_project_matrix(self.coupling, self.config.coupling_scale))

    @torch.no_grad()
    def _ensure_contractivity_(self) -> None:
        """Project unsafe operators and verify strict per-agent postconditions."""

        if (
            not torch.isfinite(self.recurrent).all()
            or not torch.isfinite(self.coupling).all()
        ):
            raise SwarmContractivityError("swarm operator contains non-finite values")
        recurrent_norms = torch.linalg.matrix_norm(self.recurrent, ord=2)
        coupling_norms = torch.linalg.matrix_norm(self.coupling, ord=2)
        if (
            not torch.isfinite(recurrent_norms).all()
            or not torch.isfinite(coupling_norms).all()
        ):
            raise SwarmContractivityError(
                "swarm operator has a non-finite spectral norm"
            )
        needs_projection = (recurrent_norms >= self.config.local_margin).any() | (
            coupling_norms >= self.config.coupling_scale
        ).any()
        if bool(needs_projection.detach().cpu()):
            self.project_contractivity_()
            recurrent_norms = torch.linalg.matrix_norm(self.recurrent, ord=2)
            coupling_norms = torch.linalg.matrix_norm(self.coupling, ord=2)
        if (
            not torch.isfinite(recurrent_norms).all()
            or not torch.isfinite(coupling_norms).all()
            or bool((recurrent_norms >= self.config.local_margin).any().detach().cpu())
            or bool((coupling_norms >= self.config.coupling_scale).any().detach().cpu())
        ):
            raise SwarmContractivityError(
                "swarm C-FIRE projection failed its strict spectral postcondition"
            )

    def _map(self, state: Tensor, observations: Tensor, topology: Tensor) -> Tensor:
        # Normalize by fan-in and combine source latents before target-specific tensor transforms.
        normalized = topology / topology.sum(-1, keepdim=True).clamp_min(1.0)
        incoming = torch.einsum("ij,bjd->bid", normalized, state)
        recurrent = torch.einsum("bni,noi->bno", state, self.recurrent)
        coupled = torch.einsum("bni,noi->bno", incoming, self.coupling)
        drive = state.new_zeros(state.shape)
        drive[:, : self.config.sensory_agents] = torch.einsum(
            "bi,ndi->bnd", observations, self.input_projection
        )
        return torch.tanh(recurrent + coupled + drive + self.bias)

    @torch.no_grad()
    def forward(
        self, observations: Tensor, previous_state: Tensor | None = None
    ) -> SwarmOutput:
        # The C-FIRE boundary is immediately before the recurrent solve.
        self._ensure_contractivity_()
        regime, _ = self.monitor(observations)
        regime_f = regime.to(observations.dtype)
        topology = (
            1 - regime_f
        ) * self.normal_topology + regime_f * self.crisis_topology
        batch = observations.shape[0]
        if previous_state is None:
            state = observations.new_zeros(
                batch, self.config.agents, self.config.state_dim
            )
            steps = self.config.cold_steps
        else:
            expected = (batch, self.config.agents, self.config.state_dim)
            if tuple(previous_state.shape) != expected:
                raise ValueError(f"previous_state must have shape {expected}")
            state = previous_state
            steps = self.config.warm_steps
        for _ in range(steps):
            state = self._map(state, observations, topology)
        residual = (self._map(state, observations, topology) - state).norm(dim=(-1, -2))
        latent = state[:, -self.config.constraint_agents :].mean(1)
        return SwarmOutput(
            latent, state, regime, residual, torch.as_tensor(steps, device=state.device)
        )

    def max_local_spectral_norm(self) -> Tensor:
        return torch.linalg.matrix_norm(self.recurrent, ord=2).max()

    def max_coupling_spectral_norm(self) -> Tensor:
        return torch.linalg.matrix_norm(self.coupling, ord=2).max()
