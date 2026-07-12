from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import math

import torch
from torch import Tensor, nn

from .swarm_sessions import PCASMonitor, PCASState


class SwarmContractivityError(RuntimeError):
    """Raised when a swarm recurrent operator cannot satisfy C-FIRE."""


class SwarmTopologyError(RuntimeError):
    """Raised when a compiled System-4 topology fails its certificate."""


@dataclass(frozen=True)
class SwarmConfig:
    input_dim: int
    state_dim: int = 64
    agents: int = 28
    sensory_agents: int = 11
    constraint_agents: int = 7
    local_margin: float = 0.90
    coupling_scale: float = 0.05
    global_margin: float = 0.95
    operating_margin: float = 0.78
    cold_steps: int = 64
    warm_steps: int = 32
    residual_tolerance: float = 1e-4
    certificate_power_iterations: int = 16

    def __post_init__(self) -> None:
        # System 4 is a named, certified 11/10/7 topology rather than a generic
        # variable-size graph. Experiments that need other sizes belong in a
        # separate research module and cannot inherit this certificate.
        if self.agents != 28:
            raise ValueError("certified System 4 requires exactly 28 agents")
        if self.sensory_agents != 11:
            raise ValueError("certified System 4 requires exactly 11 sensory agents")
        if self.constraint_agents != 7:
            raise ValueError("certified System 4 requires exactly 7 constraint agents")
        if self.input_dim < 1 or self.state_dim < 1:
            raise ValueError("input_dim and state_dim must be positive")
        if not (
            0.0 < self.coupling_scale < self.local_margin < self.global_margin < 1.0
        ):
            raise ValueError(
                "require 0 < coupling_scale < local_margin < global_margin < 1"
            )
        if not 0.0 < self.operating_margin < self.global_margin:
            raise ValueError("operating_margin must be inside global_margin")
        if self.cold_steps < 1 or self.warm_steps < 1:
            raise ValueError("solver step budgets must be positive")
        if self.warm_steps > self.cold_steps:
            raise ValueError("warm_steps cannot exceed cold_steps")
        if not math.isfinite(self.residual_tolerance) or not (
            0.0 < self.residual_tolerance < 1.0
        ):
            raise ValueError("residual_tolerance must be finite and in (0, 1)")
        if self.certificate_power_iterations < 2:
            raise ValueError("certificate_power_iterations must be at least two")

    @property
    def reasoning_agents(self) -> int:
        return self.agents - self.sensory_agents - self.constraint_agents


@dataclass(frozen=True, slots=True)
class TopologyCertificate:
    name: str
    agents: int
    sensory_agents: int
    reasoning_agents: int
    constraint_agents: int
    edge_count: int
    maximum_reachability_steps: int
    warm_step_budget: int
    sha256: str


@dataclass(frozen=True, slots=True)
class OperatorCertificate:
    topology: str
    local_norm_max: Tensor
    coupling_norm_max: Tensor
    topology_norm: Tensor
    global_operator_norm_bound: Tensor
    global_operator_norm_estimate: Tensor
    global_margin: Tensor

    @property
    def certified(self) -> bool:
        values = (
            self.local_norm_max,
            self.coupling_norm_max,
            self.topology_norm,
            self.global_operator_norm_bound,
            self.global_operator_norm_estimate,
        )
        return bool(
            torch.stack([value.float() for value in values]).isfinite().all()
            and (self.global_operator_norm_bound < self.global_margin)
        )


@dataclass(frozen=True, slots=True)
class SwarmOutput:
    latent: Tensor
    joint_state: Tensor
    regime: Tensor
    residual: Tensor
    iterations: Tensor
    converged: Tensor
    safe_for_advice: Tensor
    operator_norm_bound: Tensor
    operator_norm_estimate: Tensor
    pcas_state: PCASState
    advisory_only: bool = True


def _project_matrix(weight: Tensor, margin: float) -> Tensor:
    """Project a batch of matrices using an FP32 certificate plane."""

    work = weight.detach().float()
    sigma = torch.linalg.matrix_norm(work, ord=2).clamp_min(1e-8)
    target = margin * (1.0 - 1e-5)
    scale = torch.clamp(
        torch.as_tensor(target, device=work.device, dtype=torch.float32) / sigma,
        max=1.0,
    )
    return (work * scale[..., None, None]).to(dtype=weight.dtype)


class TensorSwarm(nn.Module):
    """Certified, advisory-only System-4 equilibrium tensor swarm.

    The production topology is fixed at 11 sensory, 10 reasoning, and seven
    constraint agents. Agent communication is a strict lower-triangular tensor
    operator. A conservative global operator-norm bound (not merely a local
    eigenvalue claim) protects the bounded fixed-point iteration.
    """

    answer_bearing = False
    NORMAL_EDGE_COUNT = 192
    CRISIS_EDGE_COUNT = 28

    def __init__(self, config: SwarmConfig):
        super().__init__()
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

        normal = self._compile_normal_topology()
        crisis = self._compile_crisis_topology()
        self.register_buffer("_normal_topology", normal, persistent=False)
        self.register_buffer("_crisis_topology", crisis, persistent=False)
        self.register_buffer(
            "_normal_topology_reference", normal.clone(), persistent=False
        )
        self.register_buffer(
            "_crisis_topology_reference", crisis.clone(), persistent=False
        )
        self._normal_certificate = self._certify_topology(
            "normal", normal, self.NORMAL_EDGE_COUNT
        )
        self._crisis_certificate = self._certify_topology(
            "crisis", crisis, self.CRISIS_EDGE_COUNT
        )
        self.monitor = PCASMonitor(config.input_dim)
        self._operator_signature: Tensor | None = None
        self._operator_cache: tuple[OperatorCertificate, OperatorCertificate] | None = (
            None
        )
        with torch.no_grad():
            self.recurrent.copy_(_project_matrix(self.recurrent, config.local_margin))
            self.coupling.copy_(_project_matrix(self.coupling, config.coupling_scale))
            self._ensure_contractivity_()

    def _apply(self, fn, recurse: bool = True):
        result = super()._apply(fn, recurse=recurse)
        # Python-side telemetry tensors do not participate in Module._apply.
        # Re-certify lazily after every device/dtype transformation.
        self._operator_signature = None
        self._operator_cache = None
        return result

    @property
    def normal_topology(self) -> Tensor:
        """Return a copy so callers cannot mutate the certified graph."""

        return self._normal_topology.detach().clone()

    @property
    def crisis_topology(self) -> Tensor:
        """Return a copy so callers cannot mutate the certified graph."""

        return self._crisis_topology.detach().clone()

    @property
    def topology_certificates(
        self,
    ) -> tuple[TopologyCertificate, TopologyCertificate]:
        return self._normal_certificate, self._crisis_certificate

    @classmethod
    def _compile_normal_topology(cls) -> Tensor:
        n = 28
        topology = torch.zeros(n, n, dtype=torch.bool)
        # Eight-neighbour causal receptive field: 188 edges.
        for target in range(1, n):
            for source in range(max(0, target - 8), target):
                topology[target, source] = True
        # Four long skip paths make the required 192-edge compiled graph.
        for offset in range(4):
            topology[21 + offset, offset] = True
        return topology

    @classmethod
    def _compile_crisis_topology(cls) -> Tensor:
        topology = torch.zeros(28, 28, dtype=torch.bool)
        # All sensory signals meet at the first reasoning agent.
        topology[11, 0:11] = True
        # The remaining reasoning agents receive that shared latent.
        topology[12:21, 11] = True
        # Each constraint receives one reasoning signal; constraint 21 also
        # owns a direct aggregate safety shortcut. Total: 11+9+7+1 = 28.
        for offset in range(7):
            topology[21 + offset, 14 + offset] = True
        topology[21, 11] = True
        return topology

    def _certify_topology(
        self, name: str, topology: Tensor, expected_edges: int
    ) -> TopologyCertificate:
        if topology.shape != (self.config.agents, self.config.agents):
            raise SwarmTopologyError(f"{name} topology shape is invalid")
        if topology.dtype != torch.bool:
            raise SwarmTopologyError(f"{name} topology must be boolean")
        if bool(torch.triu(topology, diagonal=0).any()):
            raise SwarmTopologyError(
                f"{name} topology is not strictly lower triangular"
            )
        edges = int(topology.count_nonzero())
        if edges != expected_edges:
            raise SwarmTopologyError(
                f"{name} topology has {edges} edges; expected {expected_edges}"
            )

        constraints = range(
            self.config.agents - self.config.constraint_agents, self.config.agents
        )
        maximum = 0
        adjacency = {
            source: [
                target
                for target in range(source + 1, self.config.agents)
                if bool(topology[target, source])
            ]
            for source in range(self.config.agents)
        }
        for source in range(self.config.sensory_agents):
            distances = {source: 0}
            frontier = [source]
            while frontier:
                current = frontier.pop(0)
                for target in adjacency[current]:
                    if target not in distances:
                        distances[target] = distances[current] + 1
                        frontier.append(target)
            for target in constraints:
                if target not in distances:
                    raise SwarmTopologyError(
                        f"{name} topology cannot route sensory {source} to constraint {target}"
                    )
                maximum = max(maximum, distances[target])
        if maximum > self.config.warm_steps:
            raise SwarmTopologyError(
                f"{name} topology requires {maximum} steps, exceeding warm budget"
            )
        digest = sha256(
            topology.to(torch.uint8).contiguous().cpu().numpy().tobytes()
        ).hexdigest()
        return TopologyCertificate(
            name=name,
            agents=self.config.agents,
            sensory_agents=self.config.sensory_agents,
            reasoning_agents=self.config.reasoning_agents,
            constraint_agents=self.config.constraint_agents,
            edge_count=edges,
            maximum_reachability_steps=maximum,
            warm_step_budget=self.config.warm_steps,
            sha256=digest,
        )

    def _assert_topology_immutability(self) -> None:
        if not torch.equal(
            self._normal_topology, self._normal_topology_reference
        ) or not torch.equal(self._crisis_topology, self._crisis_topology_reference):
            raise SwarmTopologyError("a compiled System-4 topology was mutated")

    @staticmethod
    def _normalized_topology(topology: Tensor) -> Tensor:
        work = topology.float()
        return work / work.sum(-1, keepdim=True).clamp_min(1.0)

    @torch.no_grad()
    def project_contractivity_(self) -> None:
        """Project local and global operator bounds into a strict interior."""

        self._operator_signature = None
        self._operator_cache = None
        local_target = min(
            self.config.local_margin * (1.0 - 1e-5),
            self.config.operating_margin * 0.80,
        )
        topology_norm = max(
            float(
                torch.linalg.matrix_norm(
                    self._normalized_topology(self._normal_topology), ord=2
                )
            ),
            float(
                torch.linalg.matrix_norm(
                    self._normalized_topology(self._crisis_topology), ord=2
                )
            ),
        )
        available = self.config.operating_margin - local_target
        coupling_target = min(
            self.config.coupling_scale * (1.0 - 1e-5),
            available / max(topology_norm, 1e-8) * (1.0 - 1e-5),
        )
        if coupling_target <= 0.0:
            raise SwarmContractivityError(
                "no positive coupling budget remains under the global margin"
            )
        self.recurrent.copy_(_project_matrix(self.recurrent, local_target))
        self.coupling.copy_(_project_matrix(self.coupling, coupling_target))

    @torch.no_grad()
    def _parameter_signature(self) -> Tensor:
        """Content-aware mutation sentinel for the cached spectral certificate."""

        parts: list[Tensor] = []
        for parameter in (self.recurrent, self.coupling):
            flat = parameter.detach().float().reshape(-1)
            stride = max(1, flat.numel() // 128)
            sample = flat[::stride][:128]
            weights = torch.linspace(
                1.0,
                2.0,
                sample.numel(),
                device=sample.device,
                dtype=torch.float32,
            )
            parts.extend(
                (
                    flat.sum(),
                    flat.square().sum(),
                    flat.abs().sum(),
                    (sample * weights).sum(),
                    torch.as_tensor(
                        parameter._version,
                        device=flat.device,
                        dtype=torch.float32,
                    ),
                )
            )
        return torch.stack(parts)

    def _linear_map(self, state: Tensor, topology: Tensor) -> Tensor:
        recurrent = torch.einsum("ni,noi->no", state, self.recurrent.float())
        incoming = self._normalized_topology(topology) @ state
        coupled = torch.einsum("ni,noi->no", incoming, self.coupling.float())
        return recurrent + coupled

    def _linear_map_transpose(self, state: Tensor, topology: Tensor) -> Tensor:
        local = torch.einsum("no,noi->ni", state, self.recurrent.float())
        target_transpose = torch.einsum("no,noi->ni", state, self.coupling.float())
        return local + self._normalized_topology(topology).T @ target_transpose

    @torch.no_grad()
    def _global_operator_norm_estimate(self, topology: Tensor) -> Tensor:
        total = self.config.agents * self.config.state_dim
        seed = torch.arange(
            1,
            total + 1,
            device=self.recurrent.device,
            dtype=torch.float32,
        ).reshape(self.config.agents, self.config.state_dim)
        vector = torch.sin(seed).div(seed.sqrt()).contiguous()
        vector = vector / vector.norm().clamp_min(1e-12)
        for _ in range(self.config.certificate_power_iterations):
            forward = self._linear_map(vector, topology)
            transpose = self._linear_map_transpose(forward, topology)
            norm = transpose.norm()
            if not torch.isfinite(norm) or bool((norm <= 1e-12).detach().cpu()):
                return torch.zeros((), device=vector.device, dtype=torch.float32)
            vector = transpose / norm
        return self._linear_map(vector, topology).norm()

    @torch.no_grad()
    def _raw_operator_certificate(self, topology: str | Tensor) -> OperatorCertificate:
        self._assert_topology_immutability()
        if isinstance(topology, str):
            if topology == "normal":
                name, tensor = "normal", self._normal_topology
            elif topology == "crisis":
                name, tensor = "crisis", self._crisis_topology
            else:
                raise ValueError("topology must be 'normal' or 'crisis'")
        else:
            if topology.shape != (self.config.agents, self.config.agents):
                raise ValueError("topology tensor shape is invalid")
            name, tensor = "selected", topology
        recurrent_norms = torch.linalg.matrix_norm(
            self.recurrent.detach().float(), ord=2
        )
        coupling_norms = torch.linalg.matrix_norm(self.coupling.detach().float(), ord=2)
        normalized = self._normalized_topology(tensor)
        topology_norm = torch.linalg.matrix_norm(normalized, ord=2)
        local_max = recurrent_norms.max()
        coupling_max = coupling_norms.max()
        upper = local_max + topology_norm * coupling_max
        estimate = self._global_operator_norm_estimate(tensor)
        return OperatorCertificate(
            topology=name,
            local_norm_max=local_max,
            coupling_norm_max=coupling_max,
            topology_norm=topology_norm,
            global_operator_norm_bound=upper,
            global_operator_norm_estimate=estimate,
            global_margin=torch.as_tensor(
                self.config.global_margin,
                device=upper.device,
                dtype=torch.float32,
            ),
        )

    @torch.no_grad()
    def operator_certificate(self, topology: str | Tensor) -> OperatorCertificate:
        self._ensure_contractivity_()
        assert self._operator_cache is not None
        normal, crisis = self._operator_cache
        if isinstance(topology, str):
            if topology == "normal":
                return normal
            if topology == "crisis":
                return crisis
            raise ValueError("topology must be 'normal' or 'crisis'")
        if torch.equal(topology.bool(), self._normal_topology):
            return normal
        if torch.equal(topology.bool(), self._crisis_topology):
            return crisis
        return self._raw_operator_certificate(topology)

    @torch.no_grad()
    def _ensure_contractivity_(self) -> None:
        """Project unsafe tensors, then verify local and global postconditions."""

        self._assert_topology_immutability()
        if (
            not torch.isfinite(self.recurrent).all()
            or not torch.isfinite(self.coupling).all()
        ):
            raise SwarmContractivityError("swarm operator contains non-finite values")
        signature = self._parameter_signature()
        if (
            self._operator_signature is not None
            and self._operator_cache is not None
            and self._operator_signature.device == signature.device
            and torch.equal(signature, self._operator_signature)
        ):
            return
        normal = self._raw_operator_certificate("normal")
        crisis = self._raw_operator_certificate("crisis")
        needs_projection = (
            normal.local_norm_max >= self.config.local_margin
            or normal.coupling_norm_max >= self.config.coupling_scale
            or normal.global_operator_norm_bound >= self.config.global_margin
            or crisis.global_operator_norm_bound >= self.config.global_margin
        )
        if bool(needs_projection.detach().cpu()):
            self.project_contractivity_()
            normal = self._raw_operator_certificate("normal")
            crisis = self._raw_operator_certificate("crisis")
        if not normal.certified or not crisis.certified:
            raise SwarmContractivityError(
                "swarm C-FIRE projection failed its global spectral postcondition"
            )
        self._operator_signature = self._parameter_signature().detach().clone()
        self._operator_cache = (normal, crisis)

    def _map(self, state: Tensor, observations: Tensor, topology: Tensor) -> Tensor:
        normalized = self._normalized_topology(topology).to(dtype=state.dtype)
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
        self,
        observations: Tensor,
        previous_state: Tensor | None = None,
        *,
        pcas_state: PCASState | None = None,
    ) -> SwarmOutput:
        if observations.ndim != 2 or observations.shape[1] != self.config.input_dim:
            raise ValueError("observations must have shape [batch, input_dim]")
        if (
            not torch.is_floating_point(observations)
            or not torch.isfinite(observations).all()
        ):
            raise ValueError("observations must be finite floating point")
        if observations.device != self.recurrent.device:
            raise ValueError("observations and TensorSwarm must share a device")
        observations = observations.to(dtype=self.recurrent.dtype)

        self._ensure_contractivity_()
        detector = self.monitor(observations, pcas_state)
        regime_f = detector.regime.to(dtype=torch.float32)
        topology = (
            1.0 - regime_f
        ) * self._normal_topology.float() + regime_f * self._crisis_topology.float()
        assert self._operator_cache is not None
        normal_certificate, crisis_certificate = self._operator_cache
        operator_bound = (
            (1.0 - regime_f) * normal_certificate.global_operator_norm_bound
            + regime_f * crisis_certificate.global_operator_norm_bound
        )
        operator_estimate = (
            (1.0 - regime_f) * normal_certificate.global_operator_norm_estimate
            + regime_f * crisis_certificate.global_operator_norm_estimate
        )
        batch = observations.shape[0]
        expected = (batch, self.config.agents, self.config.state_dim)
        if previous_state is None:
            state = observations.new_zeros(expected)
            safe_previous = state
            step_budget = self.config.cold_steps
        else:
            if tuple(previous_state.shape) != expected:
                raise ValueError(f"previous_state must have shape {expected}")
            if (
                previous_state.device != observations.device
                or previous_state.dtype != observations.dtype
                or not torch.isfinite(previous_state).all()
            ):
                raise ValueError(
                    "previous_state must be finite and match observation placement"
                )
            state = previous_state.detach().clone()
            safe_previous = state
            step_budget = self.config.warm_steps

        residual = torch.full(
            (batch,), float("inf"), device=state.device, dtype=torch.float32
        )
        converged = torch.tensor(False, device=state.device)
        iterations = 0
        for index in range(step_budget):
            candidate = self._map(state, observations, topology)
            delta = (candidate.float() - state.float()).square().mean(dim=(-1, -2))
            residual = delta.sqrt()
            state = candidate
            iterations = index + 1
            finite = torch.isfinite(state).all() & torch.isfinite(residual).all()
            reached = residual.max() <= self.config.residual_tolerance
            if bool((finite & reached).detach().cpu()):
                converged = torch.tensor(True, device=state.device)
                break
            if not bool(finite.detach().cpu()):
                break

        safe = (
            converged
            & torch.isfinite(state).all()
            & torch.isfinite(residual).all()
            & (operator_bound < self.config.global_margin)
        )
        if bool(safe.detach().cpu()):
            joint_state = state
            latent = state[:, -self.config.constraint_agents :].mean(1)
        else:
            # System 4 is advisory only. An unconverged state is never exposed
            # as if it were a valid latent and is never committed by the cache.
            joint_state = safe_previous
            latent = observations.new_zeros(batch, self.config.state_dim)
        return SwarmOutput(
            latent=latent,
            joint_state=joint_state,
            regime=detector.regime,
            residual=residual,
            iterations=torch.as_tensor(iterations, device=state.device),
            converged=converged,
            safe_for_advice=safe,
            operator_norm_bound=operator_bound,
            operator_norm_estimate=operator_estimate,
            pcas_state=detector.state,
        )

    def max_local_spectral_norm(self) -> Tensor:
        return torch.linalg.matrix_norm(self.recurrent.float(), ord=2).max()

    def max_coupling_spectral_norm(self) -> Tensor:
        return torch.linalg.matrix_norm(self.coupling.float(), ord=2).max()

    @torch.no_grad()
    def global_spectral_radius(self, topology: str | Tensor) -> Tensor:
        """Measure spectral radius separately from the operator-norm bound.

        Both certified communication graphs are strict lower triangular, so
        the full block operator is block lower triangular and its eigenvalues
        are exactly the union of the recurrent diagonal-block eigenvalues.
        Coupling still affects the non-normal operator norm and convergence
        transient, which remain certified independently above.
        """

        certificate = self.operator_certificate(topology)
        if not certificate.certified:
            raise SwarmContractivityError(
                "spectral radius cannot be reported for an uncertified topology"
            )
        eigenvalues = torch.linalg.eigvals(self.recurrent.detach().float())
        radius = eigenvalues.abs().amax().float()
        if not bool(torch.isfinite(radius).detach().cpu()):
            raise SwarmContractivityError("global spectral radius is non-finite")
        return radius
