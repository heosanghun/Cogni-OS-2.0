from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, ClassVar, Protocol, runtime_checkable

import torch
from torch import Tensor

from .deq import (
    ContractivityError,
    SolverInfo,
    _broyden_inverse,
    _damped_fixed_point,
    normalized_residual,
)


PolicyValueFn = Callable[[Tensor], tuple[Tensor, Tensor]]
TransitionFn = Callable[[Tensor, Tensor], Tensor]


@runtime_checkable
class TensorTransition(Protocol):
    """Declared no-cache, stateless DEQ latent transition.

    The marker attributes are deliberately explicit.  Runtime structural
    checks alone cannot prove that an arbitrary Python callback is stateless,
    so strict search accepts only callbacks that opt into all four parts of
    this contract.  Implementations must not mutate the input, retain latent
    tensors, maintain a KV cache, or depend on an autoregressive history.
    """

    __cogni_tensor_transition__: bool
    __cogni_deq_transition__: bool
    __cogni_stateless__: bool
    __cogni_uses_kv_cache__: bool

    def __call__(self, state: Tensor, actions: Tensor) -> Tensor: ...


@dataclass(frozen=True, slots=True)
class DeclaredDEQTensorTransition:
    """Immutable marker wrapper created by :func:`deq_tensor_transition`.

    This is a declaration of the callback's behavior, not a sandbox.  Search
    still validates tensor shape, dtype, device, finiteness, and input
    immutability on every expansion.
    """

    callback: TransitionFn

    __cogni_tensor_transition__: ClassVar[bool] = True
    __cogni_deq_transition__: ClassVar[bool] = True
    __cogni_stateless__: ClassVar[bool] = True
    __cogni_uses_kv_cache__: ClassVar[bool] = False
    __cogni_broyden_solver__: ClassVar[bool] = False

    def __call__(self, state: Tensor, actions: Tensor) -> Tensor:
        return self.callback(state, actions)


def deq_tensor_transition(callback: TransitionFn) -> DeclaredDEQTensorTransition:
    """Declare a transition as a stateless, tensor-only, no-KV DEQ operator.

    Call this only at a trusted construction boundary after auditing the
    callback.  Automatically wrapping user-supplied callbacks would defeat
    strict mode's fail-closed policy.

    The function supports decorator syntax::

        @deq_tensor_transition
        def transition(state, actions): ...
    """

    if not callable(callback):
        raise TypeError("transition callback must be callable")
    return DeclaredDEQTensorTransition(callback)


class TransitionContractError(TypeError):
    """Raised before search when a transition lacks the strict CTS contract."""


def validate_deq_tensor_transition(transition: object) -> None:
    """Fail closed unless all tensor/DEQ/stateless/no-cache markers are exact."""

    required = {
        "__cogni_tensor_transition__": True,
        "__cogni_deq_transition__": True,
        "__cogni_stateless__": True,
        "__cogni_uses_kv_cache__": False,
    }
    invalid = [
        name
        for name, expected in required.items()
        if getattr(transition, name, None) is not expected
    ]
    if invalid:
        names = ", ".join(invalid)
        raise TransitionContractError(
            "strict CTS rejected an undeclared/stateful/KV transition; "
            f"invalid markers: {names}. Wrap a verified callback with "
            "deq_tensor_transition(), or explicitly disable strict mode only "
            "for tests/legacy migration."
        )


class ContractiveBroydenTransition:
    """Certified cache-free action transition backed by limited Broyden.

    The implicit map ``tanh(contraction * z + drive(state, action))`` has a
    global Jacobian norm no larger than ``contraction``. Solver history and
    iterations are hard-bounded and no latent or KV state is retained.
    """

    __cogni_tensor_transition__: ClassVar[bool] = True
    __cogni_deq_transition__: ClassVar[bool] = True
    __cogni_stateless__: ClassVar[bool] = True
    __cogni_uses_kv_cache__: ClassVar[bool] = False
    __cogni_broyden_solver__: ClassVar[bool] = True

    def __init__(
        self,
        *,
        width: int,
        contraction: float = 0.5,
        spectral_margin: float = 0.95,
        action_scale: float = 1.0e-3,
        tolerance: float = 1.0e-4,
        max_iter: int = 24,
        history: int = 6,
        fallback_steps: int = 64,
    ) -> None:
        if width < 1:
            raise ValueError("width must be positive")
        if not 0.0 <= contraction < spectral_margin < 1.0:
            raise ContractivityError(
                "transition requires 0 <= contraction < spectral_margin < 1"
            )
        if action_scale < 0.0:
            raise ValueError("action_scale cannot be negative")
        if tolerance <= 0.0 or max_iter < 1 or history < 1 or fallback_steps < 1:
            raise ValueError(
                "solver tolerance, iterations, and history must be positive"
            )
        self.width = int(width)
        self.contraction = float(contraction)
        self.spectral_margin = float(spectral_margin)
        self.action_scale = float(action_scale)
        self.tolerance = float(tolerance)
        self.max_iter = int(max_iter)
        self.history = int(history)
        self.fallback_steps = int(fallback_steps)
        self.last_info: SolverInfo | None = None

    @torch.no_grad()
    def __call__(self, state: Tensor, actions: Tensor) -> Tensor:
        if actions.shape != (self.width,) or actions.dtype != torch.int64:
            raise ValueError(f"actions must have int64 shape [{self.width}]")
        if actions.device != state.device:
            raise ValueError("actions and state must share one device")
        dimensions = (self.width,) + (1,) * state.ndim
        action_values = actions.to(dtype=state.dtype)
        centered = action_values - action_values.mean()
        drive = state.unsqueeze(0) + self.action_scale * centered.view(dimensions)
        z0 = torch.zeros_like(drive)

        def residual(z: Tensor) -> Tensor:
            return torch.tanh(self.contraction * z + drive) - z

        z_star, iterations, residual_value, converged = _broyden_inverse(
            residual,
            z0,
            tolerance=self.tolerance,
            max_iter=self.max_iter,
            history=self.history,
        )
        used_fallback = False
        if not converged or not torch.isfinite(z_star).all():
            used_fallback = True

            def fixed(z: Tensor) -> Tensor:
                return torch.tanh(self.contraction * z + drive)

            z_star = _damped_fixed_point(fixed, z0, 1.0, self.fallback_steps)
            residual_value = normalized_residual(fixed(z_star) - z_star)
            converged = residual_value <= self.tolerance
        self.last_info = SolverInfo(
            converged,
            iterations,
            residual_value,
            self.contraction,
            used_fallback,
        )
        if not converged or not torch.isfinite(z_star).all():
            raise ContractivityError(
                "bounded CTS Broyden transition failed to converge "
                f"(residual={residual_value:.4e}, iterations={iterations})"
            )
        return z_star


@dataclass(frozen=True)
class PUCTConfig:
    """Hard bounds and numerical controls for one latent-tree search.

    ``policy_value`` is expected to return ``(action_logits, scalar_value)``.
    ``transition`` receives one latent state and the fixed action tensor
    ``arange(width)`` and must return exactly ``[width, *state.shape]``.

    Search metadata is preallocated at ``max_nodes``.  Consequently, neither
    the requested number of simulations nor repeated visits can grow the tree
    or statistics allocation.  Model workspaces owned by callbacks are outside
    this accounting boundary.
    """

    width: int = 3
    max_depth: int = 64
    max_nodes: int = 193
    simulations: int = 128
    c_puct: float = 1.25
    discount: float = 1.0
    seed: int = 0
    selection_temperature: float = 0.0
    ancestor_k: int = 3
    ancestor_mix: float = 0.10
    ancestor_temperature: float = 1.0
    ancestor_capacity: int | None = None
    ancestor_storage_dtype: torch.dtype = torch.float16
    strict_deq_transition: bool = True

    def __post_init__(self) -> None:
        if self.width < 1:
            raise ValueError("width must be positive")
        if self.max_depth < 1:
            raise ValueError("max_depth must be positive")
        if self.max_nodes < 1 + self.width:
            raise ValueError("max_nodes must fit the root and one full expansion")
        if self.simulations < 1:
            raise ValueError("simulations must be positive")
        if self.c_puct < 0.0:
            raise ValueError("c_puct must be non-negative")
        if not 0.0 <= self.discount <= 1.0:
            raise ValueError("discount must be in [0, 1]")
        if self.selection_temperature < 0.0:
            raise ValueError("selection_temperature must be non-negative")
        if self.ancestor_k < 0:
            raise ValueError("ancestor_k must be non-negative")
        if not 0.0 <= self.ancestor_mix <= 1.0:
            raise ValueError("ancestor_mix must be in [0, 1]")
        if self.ancestor_temperature <= 0.0:
            raise ValueError("ancestor_temperature must be positive")
        if self.ancestor_capacity is not None and self.ancestor_capacity < 1:
            raise ValueError("ancestor_capacity must be positive")
        if (
            self.ancestor_capacity is not None
            and self.ancestor_capacity > self.max_nodes
        ):
            raise ValueError("ancestor_capacity cannot exceed max_nodes")
        if self.ancestor_k > 0:
            capacity = self.ancestor_capacity or self.max_nodes
            if self.ancestor_k > capacity:
                raise ValueError("ancestor_k cannot exceed ancestor capacity")
        if not self.ancestor_storage_dtype.is_floating_point:
            raise ValueError("ancestor_storage_dtype must be floating point")
        if not isinstance(self.strict_deq_transition, bool):
            raise TypeError("strict_deq_transition must be bool")


@dataclass(frozen=True)
class AncestorBatch:
    """Fixed-size tensor result returned by semantic ancestor retrieval."""

    states: Tensor
    scores: Tensor
    handles: Tensor
    valid: Tensor

    def blend(self, query: Tensor, mix: float, temperature: float = 1.0) -> Tensor:
        """Convexly blend a query with its similarity-weighted ancestors.

        Padding is represented only by a tensor mask.  The all-padding case is
        handled without a Python branch and returns ``query`` exactly.
        """

        if not 0.0 <= mix <= 1.0:
            raise ValueError("mix must be in [0, 1]")
        if temperature <= 0.0:
            raise ValueError("temperature must be positive")
        if self.states.shape[1:] != query.shape:
            raise ValueError("ancestor and query shapes differ")

        valid = self.valid.to(device=query.device)
        scores = self.scores.to(device=query.device, dtype=torch.float32)
        floor = torch.finfo(scores.dtype).min
        logits = torch.where(valid, scores / temperature, floor)
        weights = torch.softmax(logits, dim=0) * valid.to(scores.dtype)
        weights = weights / weights.sum().clamp_min(torch.finfo(weights.dtype).eps)
        view_shape = (weights.shape[0],) + (1,) * query.ndim
        context = (
            (
                self.states.to(device=query.device, dtype=query.dtype)
                * weights.view(view_shape)
            )
            .sum(dim=0)
            .to(dtype=query.dtype)
        )
        gate = valid.any().to(dtype=query.dtype)
        effective_mix = gate * float(mix)
        return (query * (1.0 - effective_mix) + context * effective_mix).to(
            dtype=query.dtype
        )


class SemanticAncestorRetriever:
    """Bounded, torch-native cosine store with lineage-safe retrieval.

    Every entry receives a monotonic integer handle and a parent handle.  A
    query follows only that parent chain before ranking by cosine similarity;
    semantically close siblings can therefore never leak into the result.
    Storage is a fixed-size ring.  When an ancestor has been evicted, traversal
    stops safely instead of following a recycled slot.
    """

    def __init__(
        self,
        feature_dim: int,
        capacity: int,
        retrieval_k: int = 3,
        *,
        storage_dtype: torch.dtype = torch.float16,
        epsilon: float = 1e-8,
        shared_state_bank: Tensor | None = None,
    ) -> None:
        if feature_dim < 1:
            raise ValueError("feature_dim must be positive")
        if capacity < 1:
            raise ValueError("capacity must be positive")
        if not 1 <= retrieval_k <= capacity:
            raise ValueError("retrieval_k must be in [1, capacity]")
        if not storage_dtype.is_floating_point:
            raise ValueError("storage_dtype must be floating point")
        if epsilon <= 0.0:
            raise ValueError("epsilon must be positive")
        if shared_state_bank is not None:
            if shared_state_bank.ndim < 2:
                raise ValueError("shared_state_bank must have shape [nodes, ...]")
            if shared_state_bank.shape[0] < capacity:
                raise ValueError("shared_state_bank must have at least capacity rows")
            if shared_state_bank.shape[-1] != feature_dim:
                raise ValueError(
                    "shared_state_bank last dimension must equal feature_dim"
                )
            if not shared_state_bank.is_floating_point():
                raise TypeError("shared_state_bank must be floating point")

        self.feature_dim = int(feature_dim)
        self.capacity = int(capacity)
        self.retrieval_k = int(retrieval_k)
        self.storage_dtype = storage_dtype
        self.epsilon = float(epsilon)

        # Lineage metadata stays on CPU.  PUCT traversal is CPU control flow,
        # so this avoids per-edge GPU scalar synchronizations.
        self._handles = torch.full((capacity,), -1, dtype=torch.int64)
        self._parents = torch.full((capacity,), -1, dtype=torch.int64)
        self._keys: Tensor | None = None
        self._shared_state_bank = shared_state_bank
        self._states: Tensor | None = None
        self._state_indices: Tensor | None = (
            torch.full((capacity,), -1, dtype=torch.int64)
            if shared_state_bank is not None
            else None
        )
        self._state_shape: tuple[int, ...] | None = None
        # Retrieval workspaces have capacity-sized shapes regardless of the
        # current lineage depth.  CPU traversal writes into these buffers and
        # copies them to fixed device buffers before ranking.
        self._candidate_slots_cpu = torch.zeros(capacity, dtype=torch.int64)
        self._candidate_valid_cpu = torch.zeros(capacity, dtype=torch.bool)
        self._candidate_slots_device: Tensor | None = None
        self._candidate_valid_device: Tensor | None = None
        self._next_handle = 0
        self._size = 0

    @property
    def size(self) -> int:
        return self._size

    def _allocate_for(self, latent: Tensor) -> None:
        if latent.ndim < 1 or latent.shape[-1] != self.feature_dim:
            raise ValueError("latent last dimension must equal feature_dim")
        if not latent.is_floating_point():
            raise TypeError("latent state must be floating point")
        self._state_shape = tuple(latent.shape)
        self._keys = torch.empty(
            (self.capacity, self.feature_dim),
            device=latent.device,
            dtype=torch.float32,
        )
        self._candidate_slots_device = torch.zeros(
            self.capacity, device=latent.device, dtype=torch.int64
        )
        self._candidate_valid_device = torch.zeros(
            self.capacity, device=latent.device, dtype=torch.bool
        )
        if self._shared_state_bank is not None:
            if tuple(self._shared_state_bank.shape[1:]) != self._state_shape:
                raise ValueError("shared_state_bank row shape must equal latent shape")
            if self._shared_state_bank.device != latent.device:
                raise ValueError("shared_state_bank and latent must share one device")
        else:
            self._states = torch.empty(
                (self.capacity, *self._state_shape),
                device=latent.device,
                dtype=self.storage_dtype,
            )

    def _validate_latent(self, latent: Tensor) -> None:
        if self._state_shape is None:
            self._allocate_for(latent)
            return
        if tuple(latent.shape) != self._state_shape:
            raise ValueError("all latent states must have one fixed shape")
        if self._shared_state_bank is not None:
            state_device = self._shared_state_bank.device
        else:
            assert self._states is not None
            state_device = self._states.device
        if latent.device != state_device:
            raise ValueError("all latent states must remain on one device")
        if not latent.is_floating_point():
            raise TypeError("latent state must be floating point")

    def _key(self, latent: Tensor) -> Tensor:
        flat = latent.detach().reshape(-1, self.feature_dim).to(dtype=torch.float32)
        pooled = flat.mean(dim=0)
        return pooled / pooled.norm().clamp_min(self.epsilon)

    def _slot_for_handle(self, handle: int) -> int:
        if handle < 0:
            return -1
        slot = handle % self.capacity
        return slot if int(self._handles[slot]) == handle else -1

    @torch.no_grad()
    def add(
        self,
        latent: Tensor,
        parent_handle: int = -1,
        *,
        state_index: int | None = None,
    ) -> int:
        """Insert one latent and return its generation-safe integer handle."""

        if parent_handle < -1:
            raise ValueError("parent_handle must be -1 or non-negative")
        if parent_handle >= self._next_handle:
            raise ValueError("parent_handle must refer to an earlier entry")
        self._validate_latent(latent)
        assert self._keys is not None

        if self._shared_state_bank is None:
            if state_index is not None:
                raise ValueError("state_index is valid only with shared_state_bank")
        else:
            if state_index is None:
                raise ValueError("state_index is required with shared_state_bank")
            if not 0 <= int(state_index) < self._shared_state_bank.shape[0]:
                raise IndexError("state_index is outside shared_state_bank")
            shared_row = self._shared_state_bank[int(state_index)]
            if latent.data_ptr() != shared_row.data_ptr():
                raise ValueError("latent must be the referenced shared_state_bank row")

        handle = self._next_handle
        slot = handle % self.capacity
        if self._shared_state_bank is None:
            assert self._states is not None
            self._states[slot].copy_(latent.detach().to(dtype=self.storage_dtype))
        else:
            assert self._state_indices is not None and state_index is not None
            self._state_indices[slot] = int(state_index)
        self._keys[slot].copy_(self._key(latent))
        self._handles[slot] = handle
        self._parents[slot] = parent_handle
        self._next_handle += 1
        self._size = min(self._size + 1, self.capacity)
        return handle

    def _empty(self, query: Tensor, k: int) -> AncestorBatch:
        return AncestorBatch(
            states=torch.zeros(
                (k, *query.shape), device=query.device, dtype=query.dtype
            ),
            scores=torch.full(
                (k,), float("-inf"), device=query.device, dtype=torch.float32
            ),
            handles=torch.full((k,), -1, dtype=torch.int64),
            valid=torch.zeros((k,), device=query.device, dtype=torch.bool),
        )

    @torch.no_grad()
    def retrieve(
        self,
        query: Tensor,
        descendant_handle: int,
        k: int | None = None,
    ) -> AncestorBatch:
        """Return the top-k semantic matches from one true ancestor chain.

        The descendant itself is excluded.  Returned tensors are padded to
        exactly ``k`` rows, making downstream shapes independent of depth.
        Equal similarities are resolved toward the nearest ancestor.
        """

        count = self.retrieval_k if k is None else int(k)
        if not 1 <= count <= self.capacity:
            raise ValueError("k must be in [1, capacity]")
        self._validate_latent(query)
        assert self._keys is not None
        assert self._candidate_slots_device is not None
        assert self._candidate_valid_device is not None

        descendant_slot = self._slot_for_handle(int(descendant_handle))
        if descendant_slot < 0:
            return self._empty(query, count)

        self._candidate_slots_cpu.zero_()
        self._candidate_valid_cpu.zero_()
        candidate_count = 0
        parent = int(self._parents[descendant_slot])
        for _ in range(self.capacity):
            slot = self._slot_for_handle(parent)
            if slot < 0:
                break
            self._candidate_slots_cpu[candidate_count] = slot
            self._candidate_valid_cpu[candidate_count] = True
            candidate_count += 1
            parent = int(self._parents[slot])

        if candidate_count == 0:
            return self._empty(query, count)

        self._candidate_slots_device.copy_(self._candidate_slots_cpu)
        self._candidate_valid_device.copy_(self._candidate_valid_cpu)
        query_key = self._key(query).to(device=self._keys.device)
        similarities = (
            self._keys.index_select(0, self._candidate_slots_device) @ query_key
        )
        similarities = torch.where(
            self._candidate_valid_device,
            similarities,
            torch.full_like(similarities, torch.finfo(similarities.dtype).min),
        )
        # Candidate order is nearest-to-oldest.  Stable sorting makes cosine
        # ties deterministic and keeps the nearest ancestor first.
        order = torch.argsort(similarities, descending=True, stable=True)
        selected_order = order[:count]
        selected_slots = self._candidate_slots_device.index_select(0, selected_order)
        selected_valid = self._candidate_valid_device.index_select(0, selected_order)

        if self._shared_state_bank is not None:
            assert self._state_indices is not None
            selected_slots_cpu = selected_slots.to(device="cpu")
            state_indices = self._state_indices.index_select(
                0, selected_slots_cpu
            ).clamp_min(0)
            selected_states = self._shared_state_bank.index_select(
                0, state_indices.to(device=self._shared_state_bank.device)
            )
        else:
            assert self._states is not None
            selected_states = self._states.index_select(0, selected_slots)

        result = self._empty(query, count)
        selected_states = selected_states.to(device=query.device, dtype=query.dtype)
        state_mask_shape = (count,) + (1,) * query.ndim
        result.states.copy_(
            torch.where(
                selected_valid.to(query.device).view(state_mask_shape),
                selected_states,
                torch.zeros_like(selected_states),
            )
        )
        selected_scores = similarities.index_select(0, selected_order).to(query.device)
        result.scores.copy_(
            torch.where(
                selected_valid.to(query.device),
                selected_scores,
                torch.full_like(selected_scores, float("-inf")),
            )
        )
        selected_slots_cpu = selected_slots.to(device="cpu")
        raw_handles = self._handles.index_select(0, selected_slots_cpu)
        result.handles.copy_(
            torch.where(
                selected_valid.to(device="cpu"),
                raw_handles,
                torch.full_like(raw_handles, -1),
            )
        )
        result.valid.copy_(selected_valid.to(query.device))
        return result

    def memory_bytes(self) -> int:
        """Return bytes owned by the retriever, excluding a shared state bank."""

        tensors: tuple[Tensor | None, ...] = (
            self._handles,
            self._parents,
            self._keys,
            self._states,
            self._state_indices,
            self._candidate_slots_cpu,
            self._candidate_valid_cpu,
            self._candidate_slots_device,
            self._candidate_valid_device,
        )
        return sum(t.numel() * t.element_size() for t in tensors if t is not None)

    @property
    def shares_state_storage(self) -> bool:
        """Whether latent values are referenced from an external fixed bank."""

        return self._shared_state_bank is not None

    def reset(self) -> None:
        """Forget all entries while retaining the fixed allocation for reuse."""

        self._handles.fill_(-1)
        self._parents.fill_(-1)
        if self._state_indices is not None:
            self._state_indices.fill_(-1)
        self._candidate_slots_cpu.zero_()
        self._candidate_valid_cpu.zero_()
        if self._candidate_slots_device is not None:
            self._candidate_slots_device.zero_()
        if self._candidate_valid_device is not None:
            self._candidate_valid_device.zero_()
        self._next_handle = 0
        self._size = 0


def puct_scores(
    priors: Tensor,
    visits: Tensor,
    value_sums: Tensor,
    parent_visits: int | Tensor,
    c_puct: float,
) -> Tensor:
    """Vectorized PUCT scores with zero-visit Q values defined as zero."""

    if (
        priors.ndim != 1
        or visits.shape != priors.shape
        or value_sums.shape != priors.shape
    ):
        raise ValueError("priors, visits, and value_sums must be equal-length vectors")
    if torch.any(visits < 0):
        raise ValueError("visit counts must be non-negative")
    if c_puct < 0.0:
        raise ValueError("c_puct must be non-negative")

    visits_f = visits.to(dtype=torch.float32)
    priors_f = priors.to(dtype=torch.float32)
    sums_f = value_sums.to(dtype=torch.float32)
    q_values = torch.where(
        visits > 0, sums_f / visits_f.clamp_min(1.0), torch.zeros_like(sums_f)
    )
    parent = torch.as_tensor(
        parent_visits, device=priors.device, dtype=torch.float32
    ).clamp_min(1.0)
    exploration = float(c_puct) * priors_f * torch.sqrt(parent) / (1.0 + visits_f)
    return q_values + exploration


class _TreeBuffers:
    """Fixed-capacity state and statistic tensors for a single search."""

    def __init__(self, root: Tensor, config: PUCTConfig) -> None:
        self.width = config.width
        self.capacity = config.max_nodes
        self.states = torch.empty(
            (config.max_nodes, *root.shape), device=root.device, dtype=root.dtype
        )
        self.states[0].copy_(root.detach())

        # Scalar tree control remains on CPU.  This avoids synchronizing a GPU
        # for every parent/child lookup while latent transitions stay resident.
        self.parents = torch.full((config.max_nodes,), -1, dtype=torch.int64)
        self.depths = torch.zeros((config.max_nodes,), dtype=torch.int32)
        self.children = torch.full(
            (config.max_nodes, config.width), -1, dtype=torch.int64
        )
        self.priors = torch.zeros((config.max_nodes, config.width), dtype=torch.float32)
        self.action_visits = torch.zeros(
            (config.max_nodes, config.width), dtype=torch.int64
        )
        self.action_value_sums = torch.zeros(
            (config.max_nodes, config.width), dtype=torch.float32
        )
        self.node_visits = torch.zeros((config.max_nodes,), dtype=torch.int64)
        self.expanded = torch.zeros((config.max_nodes,), dtype=torch.bool)
        # Traversal scratch is capacity-sized, never requested-depth-sized.
        # One tree cannot contain a simple path longer than its node budget.
        self.path_nodes = torch.full((config.max_nodes,), -1, dtype=torch.int64)
        self.path_actions = torch.full((config.max_nodes,), -1, dtype=torch.int64)
        self.action_indices_cpu = torch.arange(config.width, dtype=torch.int64)
        self.actions = torch.arange(config.width, device=root.device, dtype=torch.int64)
        self.node_count = 1

    def can_expand(self) -> bool:
        return self.node_count + self.width <= self.capacity

    def add_children(self, parent: int, child_states: Tensor) -> tuple[int, int]:
        if not self.can_expand():
            raise RuntimeError("tree capacity cannot fit one full-width expansion")
        start = self.node_count
        stop = start + self.width
        self.states[start:stop].copy_(child_states.detach())
        self.parents[start:stop] = parent
        self.depths[start:stop] = int(self.depths[parent]) + 1
        self.children[parent].copy_(self.action_indices_cpu + start)
        self.node_count = stop
        return start, stop

    def memory_bytes(self) -> int:
        tensors = (
            self.states,
            self.parents,
            self.depths,
            self.children,
            self.priors,
            self.action_visits,
            self.action_value_sums,
            self.node_visits,
            self.expanded,
            self.path_nodes,
            self.path_actions,
            self.action_indices_cpu,
            self.actions,
        )
        return sum(t.numel() * t.element_size() for t in tensors)


@dataclass(frozen=True)
class SearchTelemetry:
    """Host-side observability emitted after the tensor hot path completes."""

    simulations_requested: int
    simulations_completed: int
    nodes_used: int
    node_capacity: int
    expansions: int
    max_depth_reached: int
    capacity_exhausted: bool
    frontier_rollouts: int
    ancestor_queries: int
    allocated_bytes: int


@dataclass(frozen=True)
class PUCTResult:
    best_state: Tensor
    best_action: Tensor
    best_path: Tensor
    root_visit_counts: Tensor
    root_q_values: Tensor
    root_priors: Tensor
    telemetry: SearchTelemetry


class BoundedPUCTSearch:
    """Fixed-capacity PUCT over continuous latent tensor states.

    ``max_depth`` is a logical stopping rule.  Every persistent tensor and
    traversal workspace is instead sized by ``max_nodes``, ``width``, or the
    (subordinate) ancestor capacity.
    """

    def __init__(self, config: PUCTConfig | None = None) -> None:
        self.config = config or PUCTConfig()

    def estimated_preallocated_bytes(self, root: Tensor) -> int:
        """Return exact persistent tensor bytes allocated by ``search``.

        Callback-owned DEQ/model workspaces are deliberately excluded and must
        be admitted separately by the runtime.  The result depends on the
        fixed node/ancestor capacities and latent shape, never ``max_depth``.
        """

        if root.ndim < 1 or root.numel() == 0:
            raise ValueError("root must be a non-empty latent tensor")
        cfg = self.config
        nodes = cfg.max_nodes
        width = cfg.width
        state_bytes = nodes * root.numel() * root.element_size()
        # See _TreeBuffers.memory_bytes: per-node scalar metadata, four
        # [nodes,width] banks, and two fixed action vectors.
        tree_bytes = state_bytes + 37 * nodes + 24 * nodes * width + 16 * width
        if cfg.ancestor_k == 0:
            return tree_bytes
        capacity = cfg.ancestor_capacity or nodes
        # The retriever shares tree.states. It owns two int64 lineage arrays,
        # one int64 state-index array, float32 keys, and fixed CPU/device
        # candidate slot+mask workspaces.
        retriever_bytes = 42 * capacity + 4 * capacity * root.shape[-1]
        return tree_bytes + retriever_bytes

    @staticmethod
    def _policy_value(
        policy_value: PolicyValueFn, state: Tensor, width: int
    ) -> tuple[Tensor, float]:
        output = policy_value(state)
        if not isinstance(output, tuple) or len(output) != 2:
            raise TypeError("policy_value must return a pair of tensors")
        logits, value = output
        if not isinstance(logits, Tensor) or not isinstance(value, Tensor):
            raise TypeError("policy_value outputs must be tensors")
        if logits.shape != (width,):
            raise ValueError("policy logits must have shape [width]")
        if value.numel() != 1:
            raise ValueError("policy value must be scalar")
        if not bool(torch.isfinite(logits).all()) or not bool(
            torch.isfinite(value).all()
        ):
            raise ValueError("policy outputs must be finite")
        priors = torch.softmax(
            logits.detach().to(device="cpu", dtype=torch.float32), dim=0
        )
        return priors, float(value.detach().to(device="cpu", dtype=torch.float32))

    @staticmethod
    def _gumbel(width: int, generator: torch.Generator, scale: float) -> Tensor:
        if scale == 0.0:
            return torch.zeros(width, dtype=torch.float32)
        uniform = torch.rand(width, generator=generator, dtype=torch.float32)
        uniform = uniform.clamp_(1e-7, 1.0 - 1e-7)
        return -float(scale) * torch.log(-torch.log(uniform))

    @torch.no_grad()
    def search(
        self,
        root: Tensor,
        transition: TransitionFn,
        policy_value: PolicyValueFn,
        *,
        seed: int | None = None,
    ) -> PUCTResult:
        """Run a bounded search without retaining autograd graphs or KV state.

        Once ``max_nodes`` cannot fit another full-width expansion, simulations
        continue as value-only frontier rollouts.  No eviction can silently
        invalidate root statistics, and no requested-depth-sized tensor is
        allocated.
        """

        cfg = self.config
        if root.ndim < 1:
            raise ValueError("root must have at least one latent dimension")
        if not root.is_floating_point():
            raise TypeError("root latent state must be floating point")
        if root.numel() == 0:
            raise ValueError("root latent state cannot be empty")
        if cfg.strict_deq_transition:
            validate_deq_tensor_transition(transition)

        tree = _TreeBuffers(root.detach(), cfg)
        generator = torch.Generator(device="cpu")
        generator.manual_seed(cfg.seed if seed is None else int(seed))

        retriever: SemanticAncestorRetriever | None = None
        if cfg.ancestor_k > 0:
            retriever = SemanticAncestorRetriever(
                feature_dim=root.shape[-1],
                capacity=cfg.ancestor_capacity or cfg.max_nodes,
                retrieval_k=cfg.ancestor_k,
                storage_dtype=cfg.ancestor_storage_dtype,
                shared_state_bank=tree.states,
            )
            root_handle = retriever.add(tree.states[0], state_index=0)
            if root_handle != 0:
                raise RuntimeError("ancestor handle and tree node index diverged")

        expansions = 0
        ancestor_queries = 0
        capacity_exhausted = False
        frontier_rollouts = 0
        simulations_completed = 0

        for _ in range(cfg.simulations):
            path_nodes = tree.path_nodes
            path_actions = tree.path_actions
            path_length = 1
            node = 0
            path_nodes[0] = 0

            # PUCT selection walks only preallocated child/statistic tensors.
            while bool(tree.expanded[node]) and int(tree.depths[node]) < cfg.max_depth:
                visits = tree.action_visits[node]
                scores = puct_scores(
                    tree.priors[node],
                    visits,
                    tree.action_value_sums[node],
                    tree.node_visits[node],
                    cfg.c_puct,
                )
                scores = scores + self._gumbel(
                    cfg.width, generator, cfg.selection_temperature
                )
                action = int(torch.argmax(scores))
                child = int(tree.children[node, action])
                if child < 0:
                    raise RuntimeError("expanded node has an invalid child slot")
                if path_length >= tree.capacity:
                    raise RuntimeError("tree path exceeded fixed node capacity")
                path_actions[path_length - 1] = action
                path_nodes[path_length] = child
                path_length += 1
                node = child

            leaf_state = tree.states[node]
            policy_state = leaf_state
            if retriever is not None and int(tree.depths[node]) > 0:
                ancestor_queries += 1
                ancestors = retriever.retrieve(leaf_state, node, cfg.ancestor_k)
                policy_state = ancestors.blend(
                    leaf_state, cfg.ancestor_mix, cfg.ancestor_temperature
                )

            priors, leaf_value = self._policy_value(
                policy_value, policy_state, cfg.width
            )

            depth = int(tree.depths[node])
            if not bool(tree.expanded[node]) and depth < cfg.max_depth:
                if tree.can_expand():
                    input_version = policy_state._version
                    child_states = transition(policy_state, tree.actions)
                    if policy_state._version != input_version:
                        raise RuntimeError(
                            "transition violated the stateless contract by "
                            "mutating its input tensor"
                        )
                    if not isinstance(child_states, Tensor):
                        raise TypeError("transition must return a tensor")
                    expected = (cfg.width, *root.shape)
                    if tuple(child_states.shape) != expected:
                        raise ValueError(
                            f"transition output must have shape {expected}"
                        )
                    if child_states.device != root.device:
                        raise ValueError(
                            "transition output must stay on the root device"
                        )
                    if child_states.dtype != root.dtype:
                        raise ValueError("transition output must preserve root dtype")
                    if not bool(torch.isfinite(child_states).all()):
                        raise ValueError("transition output must be finite")

                    tree.priors[node].copy_(priors)
                    start, stop = tree.add_children(node, child_states)
                    tree.expanded[node] = True
                    if retriever is not None:
                        for child in range(start, stop):
                            handle = retriever.add(
                                tree.states[child], node, state_index=child
                            )
                            if handle != child:
                                raise RuntimeError(
                                    "ancestor handle and tree node index diverged"
                                )
                    expansions += 1
                else:
                    capacity_exhausted = True
                    frontier_rollouts += 1
            elif not bool(tree.expanded[node]):
                # Depth-limited value evaluation is also a bounded rollout;
                # it never materializes an additional latent node.
                frontier_rollouts += 1

            # Single-agent reasoning uses mean-value (not alternating-sign)
            # backup.  Every path and edge counter is bounded by its tensor.
            for position in range(path_length):
                tree.node_visits[int(path_nodes[position])] += 1
            backed_value = leaf_value
            for position in range(path_length - 2, -1, -1):
                parent = int(path_nodes[position])
                action = int(path_actions[position])
                tree.action_visits[parent, action] += 1
                tree.action_value_sums[parent, action] += backed_value
                backed_value *= cfg.discount

            simulations_completed += 1

        root_visits = tree.action_visits[0].clone()
        root_priors = tree.priors[0].clone()
        root_q = torch.where(
            root_visits > 0,
            tree.action_value_sums[0] / root_visits.to(torch.float32).clamp_min(1.0),
            torch.zeros(cfg.width, dtype=torch.float32),
        )
        if bool((root_visits > 0).any()):
            best_action_int = int(torch.argmax(root_visits))
        else:
            best_action_int = int(torch.argmax(root_priors))

        # Extract the most-visited bounded trajectory.  This returns only one
        # state, so the full tree allocation is released after this method.
        best_path_values = [0]
        best_node = 0
        while (
            bool(tree.expanded[best_node])
            and int(tree.depths[best_node]) < cfg.max_depth
            and len(best_path_values) < tree.capacity
        ):
            local_visits = tree.action_visits[best_node]
            # Expansion creates children before any outgoing edge has been
            # evaluated.  Stop at that evaluated leaf instead of returning an
            # arbitrary, never-visited child.
            if not bool((local_visits > 0).any()):
                break
            action = int(torch.argmax(local_visits))
            next_node = int(tree.children[best_node, action])
            if next_node < 0:
                break
            best_path_values.append(next_node)
            best_node = next_node

        allocated_bytes = tree.memory_bytes()
        if retriever is not None:
            allocated_bytes += retriever.memory_bytes()
        used_depths = tree.depths[: tree.node_count]
        telemetry = SearchTelemetry(
            simulations_requested=cfg.simulations,
            simulations_completed=simulations_completed,
            nodes_used=tree.node_count,
            node_capacity=cfg.max_nodes,
            expansions=expansions,
            max_depth_reached=int(used_depths.max()),
            capacity_exhausted=capacity_exhausted,
            frontier_rollouts=frontier_rollouts,
            ancestor_queries=ancestor_queries,
            allocated_bytes=allocated_bytes,
        )
        return PUCTResult(
            best_state=tree.states[best_node].clone(),
            best_action=torch.tensor(
                best_action_int, device=root.device, dtype=torch.int64
            ),
            best_path=torch.tensor(best_path_values, dtype=torch.int64),
            root_visit_counts=root_visits,
            root_q_values=root_q,
            root_priors=root_priors,
            telemetry=telemetry,
        )


__all__ = [
    "AncestorBatch",
    "BoundedPUCTSearch",
    "ContractiveBroydenTransition",
    "DeclaredDEQTensorTransition",
    "PUCTConfig",
    "PUCTResult",
    "PolicyValueFn",
    "SearchTelemetry",
    "SemanticAncestorRetriever",
    "TensorTransition",
    "TransitionContractError",
    "TransitionFn",
    "deq_tensor_transition",
    "puct_scores",
    "validate_deq_tensor_transition",
]
