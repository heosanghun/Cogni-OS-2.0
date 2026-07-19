from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from typing import Callable, ClassVar, Protocol, runtime_checkable

import torch
from torch import Tensor

from .deq import (
    BroydenWarmStart,
    ContractivityError,
    SolverInfo,
    _broyden_inverse,
    _damped_fixed_point,
    limited_broyden_solve,
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


@dataclass(frozen=True, slots=True)
class SearchControlsV2:
    """Four bounded learned controls consumed by certified CTS V2."""

    exploration: float
    tolerance: float
    policy_temperature: float
    act_simulations: int

    def __post_init__(self) -> None:
        numeric = (int, float)
        if (
            not isinstance(self.exploration, numeric)
            or isinstance(self.exploration, bool)
            or not math.isfinite(float(self.exploration))
            or not 0.0 <= float(self.exploration) <= 8.0
        ):
            raise ValueError("exploration must be in [0, 8]")
        if (
            not isinstance(self.tolerance, numeric)
            or isinstance(self.tolerance, bool)
            or not math.isfinite(float(self.tolerance))
            or not 0.0 < float(self.tolerance) <= 0.1
        ):
            raise ValueError("tolerance must be in (0, 0.1]")
        if (
            not isinstance(self.policy_temperature, numeric)
            or isinstance(self.policy_temperature, bool)
            or not math.isfinite(float(self.policy_temperature))
            or not 0.0 < float(self.policy_temperature) <= 10.0
        ):
            raise ValueError("policy_temperature must be in (0, 10]")
        if (
            not isinstance(self.act_simulations, int)
            or isinstance(self.act_simulations, bool)
            or self.act_simulations < 1
        ):
            raise ValueError("act_simulations must be a positive integer")


class ActionPolicyV2(Protocol):
    def __call__(self, state: Tensor) -> Tensor: ...


class CriticV2(Protocol):
    def __call__(self, state: Tensor) -> Tensor: ...


class MetaControllerV2(Protocol):
    def __call__(self, root: Tensor) -> SearchControlsV2: ...


@dataclass(frozen=True, slots=True)
class SearchRequestV2:
    root: Tensor
    mac_budget: int
    seed: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.root, Tensor):
            raise TypeError("root must be a tensor")
        if (
            not isinstance(self.mac_budget, int)
            or isinstance(self.mac_budget, bool)
            or self.mac_budget < 1
        ):
            raise ValueError("mac_budget must be a positive integer")
        if self.seed is not None and (
            not isinstance(self.seed, int)
            or isinstance(self.seed, bool)
            or not 0 <= self.seed < 2**63
        ):
            raise ValueError("seed must be an integer in [0, 2**63)")


@dataclass(frozen=True, slots=True)
class CertifiedPUCTConfigV2:
    """Hard production geometry and callback MAC certificates for CTS V2."""

    width: int = 3
    max_nodes: int = 301
    max_depth: int = 100
    simulations: int = 301
    discount: float = 1.0
    seed: int = 0
    retrieval_k: int = 3
    retrieval_min_depth: int = 10
    retrieval_mix: float = 0.10
    retrieval_temperature: float = 1.0
    meta_policy_macs: int = 1
    action_policy_macs: int = 1
    critic_macs: int = 1
    retrieval_macs: int = 1
    transition_macs: int = 1

    def __post_init__(self) -> None:
        if self.width != 3:
            raise ValueError("certified CTS V2 requires width=3")
        if self.max_nodes != 301:
            raise ValueError("certified CTS V2 requires max_nodes=301")
        if self.retrieval_k != 3:
            raise ValueError("certified CTS V2 requires semantic top-3")
        if self.retrieval_min_depth != 10:
            raise ValueError("certified CTS V2 requires retrieval_min_depth=10")
        if (
            not isinstance(self.max_depth, int)
            or isinstance(self.max_depth, bool)
            or self.max_depth < 1
            or self.max_depth > 300
            or not isinstance(self.simulations, int)
            or isinstance(self.simulations, bool)
            or self.simulations < 1
        ):
            raise ValueError(
                "max_depth must be in [1, 300] and simulations must be positive"
            )
        if (
            not isinstance(self.seed, int)
            or isinstance(self.seed, bool)
            or not 0 <= self.seed < 2**63
        ):
            raise ValueError("seed must be an integer in [0, 2**63)")
        if not 0.0 <= self.discount <= 1.0:
            raise ValueError("discount must be in [0, 1]")
        if not 0.0 <= self.retrieval_mix <= 1.0:
            raise ValueError("retrieval_mix must be in [0, 1]")
        if self.retrieval_temperature <= 0.0:
            raise ValueError("retrieval_temperature must be positive")
        for name in (
            "meta_policy_macs",
            "action_policy_macs",
            "critic_macs",
            "retrieval_macs",
            "transition_macs",
        ):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")


class MacBudgetLedger:
    """Fixed-category worst-case MAC reservation ledger."""

    __slots__ = (
        "budget",
        "reserved",
        "meta_policy",
        "action_policy",
        "critic",
        "retrieval",
        "transition",
        "exhausted",
    )

    _CATEGORIES = {
        "meta_policy",
        "action_policy",
        "critic",
        "retrieval",
        "transition",
    }

    def __init__(self, budget: int) -> None:
        if not isinstance(budget, int) or isinstance(budget, bool) or budget < 1:
            raise ValueError("MAC budget must be a positive integer")
        self.budget = budget
        self.reserved = 0
        self.meta_policy = 0
        self.action_policy = 0
        self.critic = 0
        self.retrieval = 0
        self.transition = 0
        self.exhausted = False

    def reserve(self, category: str, amount: int) -> bool:
        if category not in self._CATEGORIES:
            raise ValueError("unsupported MAC category")
        if not isinstance(amount, int) or isinstance(amount, bool) or amount < 0:
            raise ValueError("MAC reservation must be a non-negative integer")
        if self.exhausted or self.reserved + amount > self.budget:
            self.exhausted = True
            return False
        self.reserved += amount
        setattr(self, category, getattr(self, category) + amount)
        return True

    def reserve_policy_critic(self, policy_macs: int, critic_macs: int) -> bool:
        for amount in (policy_macs, critic_macs):
            if not isinstance(amount, int) or isinstance(amount, bool) or amount < 0:
                raise ValueError("policy and critic MACs must be non-negative integers")
        total = policy_macs + critic_macs
        if self.exhausted or self.reserved + total > self.budget:
            self.exhausted = True
            return False
        self.reserved += total
        self.action_policy += policy_macs
        self.critic += critic_macs
        return True


@dataclass(frozen=True, slots=True)
class CertifiedTransitionBatchV2:
    states: Tensor
    success: Tensor
    residuals: Tensor
    iterations: Tensor
    linear_solve_fallbacks: Tensor
    warm_used: Tensor
    warm_rejected: Tensor
    warm_starts: tuple[BroydenWarmStart | None, ...]


@runtime_checkable
class CertifiedTransitionV2(Protocol):
    __cogni_tensor_transition__: bool
    __cogni_deq_transition__: bool
    __cogni_stateless__: bool
    __cogni_uses_kv_cache__: bool
    __cogni_broyden_solver__: bool
    width: int
    rank: int
    operator_id: str

    def __call__(
        self,
        parent: Tensor,
        actions: Tensor,
        *,
        tolerance: float,
        warm_start: BroydenWarmStart | None,
    ) -> CertifiedTransitionBatchV2: ...


def validate_certified_transition_v2(transition: object) -> None:
    required = {
        "__cogni_tensor_transition__": True,
        "__cogni_deq_transition__": True,
        "__cogni_stateless__": True,
        "__cogni_uses_kv_cache__": False,
        "__cogni_broyden_solver__": True,
    }
    invalid = [
        name
        for name, expected in required.items()
        if getattr(transition, name, None) is not expected
    ]
    if invalid:
        raise TransitionContractError(
            "certified CTS V2 rejected transition markers: " + ", ".join(invalid)
        )
    if getattr(transition, "width", None) != 3:
        raise TransitionContractError("certified transition must use width=3")
    if getattr(transition, "rank", None) != 16:
        raise TransitionContractError("certified transition must use rank=16")
    operator_id = getattr(transition, "operator_id", None)
    if not isinstance(operator_id, str) or not operator_id:
        raise TransitionContractError("certified transition requires operator_id")


class CertifiedBroydenTransitionV2:
    """Per-action rank-16 solver with typed failure and warm diagnostics."""

    __cogni_tensor_transition__: ClassVar[bool] = True
    __cogni_deq_transition__: ClassVar[bool] = True
    __cogni_stateless__: ClassVar[bool] = True
    __cogni_uses_kv_cache__: ClassVar[bool] = False
    __cogni_broyden_solver__: ClassVar[bool] = True
    width: ClassVar[int] = 3
    rank: ClassVar[int] = 16

    def __init__(
        self,
        *,
        contraction: float = 0.4,
        spectral_margin: float = 0.95,
        action_scale: float = 1.0e-3,
        max_iter: int = 32,
        operator_id: str = "certified-broyden-v2",
    ) -> None:
        if not 0.0 <= contraction < spectral_margin < 1.0:
            raise ContractivityError(
                "certified transition requires contraction < spectral margin < 1"
            )
        if action_scale < 0.0:
            raise ValueError("action_scale cannot be negative")
        if not isinstance(max_iter, int) or isinstance(max_iter, bool) or max_iter < 17:
            raise ValueError("rank-16 transition requires max_iter >= 17")
        if not isinstance(operator_id, str) or not operator_id:
            raise ValueError("operator_id must be non-empty text")
        self.contraction = float(contraction)
        self.spectral_margin = float(spectral_margin)
        self.action_scale = float(action_scale)
        self.max_iter = max_iter
        self.operator_id = operator_id

    @torch.no_grad()
    def __call__(
        self,
        parent: Tensor,
        actions: Tensor,
        *,
        tolerance: float,
        warm_start: BroydenWarmStart | None,
    ) -> CertifiedTransitionBatchV2:
        if (
            not isinstance(parent, Tensor)
            or parent.ndim < 1
            or not parent.is_floating_point()
            or not bool(torch.isfinite(parent).all())
        ):
            raise ValueError("parent must be a finite floating tensor")
        if tuple(actions.shape) != (3,) or actions.dtype != torch.int64:
            raise ValueError("actions must have int64 shape [3]")
        if actions.device != parent.device:
            raise ValueError("actions and parent must share one device")
        if not torch.equal(
            actions, torch.arange(3, device=parent.device, dtype=torch.int64)
        ):
            raise ValueError("certified actions must be exactly arange(3)")

        action_values = actions.to(dtype=parent.dtype)
        centered = action_values - action_values.mean()
        states = torch.empty(
            (3, *parent.shape), device=parent.device, dtype=parent.dtype
        )
        success = torch.zeros(3, dtype=torch.bool)
        residuals = torch.full((3,), float("inf"), dtype=torch.float32)
        iterations = torch.zeros(3, dtype=torch.int64)
        linear_fallbacks = torch.zeros(3, dtype=torch.int64)
        warm_used = torch.zeros(3, dtype=torch.bool)
        warm_rejected = torch.zeros(3, dtype=torch.int64)
        capsules: list[BroydenWarmStart | None] = []

        for action in range(3):
            drive = parent + self.action_scale * centered[action]

            def residual(z: Tensor, fixed_drive: Tensor = drive) -> Tensor:
                return torch.tanh(self.contraction * z + fixed_drive) - z

            result = limited_broyden_solve(
                residual,
                torch.zeros_like(parent),
                tolerance=float(tolerance),
                max_iter=self.max_iter,
                rank=16,
                operator_id=self.operator_id,
                warm_start=warm_start,
            )
            iterations[action] = result.iterations
            residuals[action] = (
                result.residual if math.isfinite(result.residual) else float("inf")
            )
            linear_fallbacks[action] = result.linear_solve_fallbacks
            warm_used[action] = result.warm_used
            warm_rejected[action] = result.warm_rejected
            valid = (
                result.converged
                and tuple(result.state.shape) == tuple(parent.shape)
                and result.state.dtype == parent.dtype
                and result.state.device == parent.device
                and bool(torch.isfinite(result.state).all())
            )
            if valid:
                states[action].copy_(result.state)
                success[action] = True
                capsules.append(result.warm_start)
            else:
                # Typed numerical failure uses the finite parent state. Search
                # records Q=0 and never allocates this edge as a child node.
                states[action].copy_(parent)
                capsules.append(None)
        return CertifiedTransitionBatchV2(
            states=states,
            success=success,
            residuals=residuals,
            iterations=iterations,
            linear_solve_fallbacks=linear_fallbacks,
            warm_used=warm_used,
            warm_rejected=warm_rejected,
            warm_starts=tuple(capsules),
        )


class FixedCapacityTreeRetriever:
    """Stable whole-tree top-3 over successful rows in a shared 301-node arena."""

    def __init__(self, shared_state_arena: Tensor) -> None:
        if (
            not isinstance(shared_state_arena, Tensor)
            or shared_state_arena.ndim < 2
            or shared_state_arena.shape[0] != 301
            or not shared_state_arena.is_floating_point()
        ):
            raise ValueError("shared_state_arena must be floating [301, ...]")
        self.shared_state_arena = shared_state_arena
        self.capacity = 301
        self.retrieval_k = 3
        self.feature_dim = int(shared_state_arena.shape[-1])
        self._keys = torch.zeros(
            (301, self.feature_dim),
            device=shared_state_arena.device,
            dtype=torch.float32,
        )
        self._successful = torch.zeros(
            301, device=shared_state_arena.device, dtype=torch.bool
        )
        self._candidate_mask = torch.zeros_like(self._successful)
        self._slots = torch.arange(301, device=shared_state_arena.device)

    @staticmethod
    def _key(state: Tensor) -> Tensor:
        flat = state.detach().reshape(-1, state.shape[-1]).to(torch.float32)
        pooled = flat.mean(dim=0)
        return pooled / pooled.norm().clamp_min(1.0e-8)

    @torch.no_grad()
    def add(self, node: int, state: Tensor, *, success: bool) -> None:
        if not 0 <= int(node) < 301:
            raise IndexError("tree retrieval node is outside the arena")
        row = self.shared_state_arena[int(node)]
        if state.data_ptr() != row.data_ptr() or tuple(state.shape) != tuple(row.shape):
            raise ValueError("retriever state must be the exact shared arena row")
        if state.device != self.shared_state_arena.device:
            raise ValueError("retriever state changed device")
        if success:
            if not bool(torch.isfinite(state).all()):
                raise ValueError("successful retrieval state must be finite")
            self._keys[int(node)].copy_(self._key(state))
            self._successful[int(node)] = True
        else:
            self._keys[int(node)].zero_()
            self._successful[int(node)] = False

    @torch.no_grad()
    def retrieve(self, query: Tensor, *, current_node: int) -> AncestorBatch:
        if tuple(query.shape) != tuple(self.shared_state_arena.shape[1:]):
            raise ValueError("tree retrieval query shape differs from arena row")
        if query.device != self.shared_state_arena.device:
            raise ValueError("tree retrieval query changed device")
        if not bool(torch.isfinite(query).all()):
            raise ValueError("tree retrieval query must be finite")
        if not 0 <= int(current_node) < 301:
            raise IndexError("current_node is outside the arena")

        self._candidate_mask.copy_(self._successful)
        self._candidate_mask[int(current_node)] = False
        query_key = self._key(query)
        similarities = self._keys @ query_key
        floor = torch.finfo(similarities.dtype).min
        similarities = torch.where(
            self._candidate_mask, similarities, torch.full_like(similarities, floor)
        )
        order = torch.argsort(similarities, descending=True, stable=True)[:3]
        valid = self._candidate_mask.index_select(0, order)
        selected = self.shared_state_arena.index_select(0, order)
        result = AncestorBatch(
            states=torch.where(
                valid.view((3,) + (1,) * query.ndim),
                selected,
                torch.zeros_like(selected),
            ).to(dtype=query.dtype),
            scores=torch.where(
                valid,
                similarities.index_select(0, order),
                torch.full((3,), float("-inf"), device=query.device),
            ),
            handles=torch.where(
                valid.to(device="cpu"),
                order.to(device="cpu", dtype=torch.int64),
                torch.full((3,), -1, dtype=torch.int64),
            ),
            valid=valid,
        )
        return result

    @property
    def successful_nodes(self) -> int:
        return int(self._successful.sum())

    def memory_bytes(self) -> int:
        return sum(
            tensor.numel() * tensor.element_size()
            for tensor in (
                self._keys,
                self._successful,
                self._candidate_mask,
                self._slots,
            )
        )


class _CertifiedTreeBuffersV2:
    FAILED_CHILD = -2
    RANK = 16

    def __init__(self, root: Tensor) -> None:
        self.capacity = 301
        self.width = 3
        self.states = torch.empty(
            (301, *root.shape), device=root.device, dtype=root.dtype
        )
        self.states[0].copy_(root.detach())
        self.parents = torch.full((301,), -1, dtype=torch.int64)
        self.depths = torch.zeros(301, dtype=torch.int32)
        self.children = torch.full((301, 3), -1, dtype=torch.int64)
        self.edge_failed = torch.zeros((301, 3), dtype=torch.bool)
        self.priors = torch.zeros((301, 3), dtype=torch.float32)
        self.action_visits = torch.zeros((301, 3), dtype=torch.int64)
        self.action_value_sums = torch.zeros((301, 3), dtype=torch.float32)
        self.node_visits = torch.zeros(301, dtype=torch.int64)
        self.expanded = torch.zeros(301, dtype=torch.bool)
        self.terminal = torch.zeros(301, dtype=torch.bool)
        self.path_nodes = torch.full((301,), -1, dtype=torch.int64)
        self.path_actions = torch.full((301,), -1, dtype=torch.int64)
        self.actions = torch.arange(3, device=root.device, dtype=torch.int64)
        # Every successful node can retain one parent rank-16 capsule without
        # allocating in proportion to requested logical depth.
        self.warm_x = torch.zeros(
            (301, 17, *root.shape), device=root.device, dtype=root.dtype
        )
        self.warm_f = torch.zeros_like(self.warm_x)
        self.warm_counts = torch.zeros(301, dtype=torch.int16)
        self.warm_valid = torch.zeros(301, dtype=torch.bool)
        self.node_count = 1

    def can_expand_full_width(self) -> bool:
        return self.node_count + 3 <= 301

    def add_child(
        self,
        parent: int,
        action: int,
        state: Tensor,
        warm_start: BroydenWarmStart,
    ) -> int:
        if self.node_count >= 301:
            raise RuntimeError("certified tree arena is full")
        node = self.node_count
        self.node_count += 1
        self.states[node].copy_(state.detach())
        self.parents[node] = parent
        self.depths[node] = int(self.depths[parent]) + 1
        self.children[parent, action] = node
        self._store_warm(node, warm_start)
        return node

    def mark_failed(self, parent: int, action: int) -> None:
        self.children[parent, action] = self.FAILED_CHILD
        self.edge_failed[parent, action] = True
        # The failed edge receives exactly one explicit Q=0 observation and is
        # masked from every future selection.
        self.action_visits[parent, action] = 1
        self.action_value_sums[parent, action] = 0.0

    def _store_warm(self, node: int, capsule: BroydenWarmStart) -> None:
        if capsule.rank != 16:
            raise ValueError("child warm capsule rank must equal 16")
        count = capsule.history_size
        if count > 17:
            raise ValueError("child warm capsule exceeded rank-16 history")
        if (
            tuple(capsule.state.shape) != tuple(self.states[node].shape)
            or tuple(capsule.x_history.shape) != (count, *self.states[node].shape)
            or tuple(capsule.f_history.shape) != (count, *self.states[node].shape)
            or capsule.state.dtype != self.states.dtype
            or capsule.state.device != self.states.device
        ):
            raise ValueError("child warm capsule violates arena shape/device/dtype")
        self.warm_x[node].zero_()
        self.warm_f[node].zero_()
        if count:
            self.warm_x[node, :count].copy_(capsule.x_history)
            self.warm_f[node, :count].copy_(capsule.f_history)
        self.warm_counts[node] = count
        self.warm_valid[node] = True

    def warm_start(self, node: int, operator_id: str) -> BroydenWarmStart | None:
        if not bool(self.warm_valid[node]):
            return None
        count = int(self.warm_counts[node])
        return BroydenWarmStart(
            state=self.states[node],
            x_history=self.warm_x[node, :count],
            f_history=self.warm_f[node, :count],
            rank=16,
            operator_id=operator_id,
        )

    def memory_bytes(self) -> int:
        tensors = (
            self.states,
            self.parents,
            self.depths,
            self.children,
            self.edge_failed,
            self.priors,
            self.action_visits,
            self.action_value_sums,
            self.node_visits,
            self.expanded,
            self.terminal,
            self.path_nodes,
            self.path_actions,
            self.actions,
            self.warm_x,
            self.warm_f,
            self.warm_counts,
            self.warm_valid,
        )
        return sum(t.numel() * t.element_size() for t in tensors)


@dataclass(frozen=True, slots=True)
class SearchTelemetryV2:
    simulations_requested: int
    simulations_completed: int
    act_requested: int
    act_applied: int
    act_clamped: bool
    nodes_used: int
    node_capacity: int
    max_depth_reached: int
    capacity_exhausted: bool
    frontier_rollouts: int
    retrieval_queries: int
    retrieval_hits: int
    solver_calls: int
    solver_rank: int
    solver_history_peak: int
    solver_successes: int
    solver_failures: int
    solver_iterations_total: int
    solver_iterations_max: int
    solver_residual_max: float
    linear_solve_fallbacks: int
    warm_used: int
    warm_rejected: int
    failed_edges: int
    q_zero_backups: int
    all_fail_terminals: int
    mac_budget: int
    mac_reserved: int
    mac_budget_exhausted: bool
    mac_meta_policy: int
    mac_action_policy: int
    mac_critic: int
    mac_retrieval: int
    mac_transition: int
    allocated_bytes: int
    trace_digest: str
    unsafe_silent_fallbacks: int
    safe_for_decode: bool


@dataclass(frozen=True, slots=True)
class PUCTResultV2:
    best_state: Tensor
    best_action: Tensor
    best_path: Tensor
    root_visit_counts: Tensor
    root_q_values: Tensor
    root_priors: Tensor
    telemetry: SearchTelemetryV2


class BoundedPUCTSearchV2:
    """Certified width-3 CTS with fixed storage and explicit failure states.

    This path intentionally lives beside :class:`BoundedPUCTSearch`.  It does
    not change the legacy API: callers opt into the stricter contracts by
    constructing a :class:`SearchRequestV2` and supplying the four bounded
    collaborators.  The policy and controller surfaces are protocols only;
    the search kernel has no dependency on a concrete learned-policy module.
    """

    def __init__(self, config: CertifiedPUCTConfigV2 | None = None) -> None:
        self.config = config or CertifiedPUCTConfigV2()

    def estimated_preallocated_bytes(self, root: Tensor) -> int:
        """Return exact persistent tensor bytes, independent of max depth."""

        self._validate_root(root)
        nodes = 301
        state_bytes = root.numel() * root.element_size()
        # states + two [nodes, rank + 1, *state] warm-history banks
        tree_state_bytes = nodes * state_bytes * (1 + 2 * 17)
        # See _CertifiedTreeBuffersV2.memory_bytes().  All terms other than
        # the state/warm banks are fixed per node, plus one action vector.
        tree_metadata_bytes = 116 * nodes + 3 * 8
        # Retriever keys, two masks, and stable slot indices.
        retriever_bytes = nodes * (4 * int(root.shape[-1]) + 10)
        # One non-tree frontier state and one worst-case rank-16 warm capsule
        # let a saturated 301-node arena continue certified transitions without
        # allocating another tree node. Its continuation is committed to the
        # trace digest; no fictitious tree handles are materialized.
        frontier_bytes = state_bytes * (2 + 2 * 17)
        return tree_state_bytes + tree_metadata_bytes + retriever_bytes + frontier_bytes

    @staticmethod
    def _validate_root(root: Tensor) -> None:
        if not isinstance(root, Tensor):
            raise TypeError("root must be a tensor")
        if root.ndim < 1 or root.numel() == 0:
            raise ValueError("root must be a non-empty latent tensor")
        if not root.is_floating_point():
            raise TypeError("root latent state must be floating point")
        if not bool(torch.isfinite(root).all()):
            raise ValueError("root latent state must be finite")

    @staticmethod
    def _call_meta_controller(
        controller: MetaControllerV2, root: Tensor
    ) -> SearchControlsV2:
        version = root._version
        controls = controller(root)
        if root._version != version:
            raise RuntimeError("meta controller mutated the root tensor")
        if not isinstance(controls, SearchControlsV2):
            raise TypeError("meta controller must return SearchControlsV2")
        return controls

    @staticmethod
    def _call_policy_critic(
        action_policy: ActionPolicyV2,
        critic: CriticV2,
        state: Tensor,
        *,
        temperature: float,
    ) -> tuple[Tensor, float]:
        version = state._version
        logits = action_policy(state)
        if state._version != version:
            raise RuntimeError("action policy mutated its input tensor")
        version = state._version
        value = critic(state)
        if state._version != version:
            raise RuntimeError("critic mutated its input tensor")
        if not isinstance(logits, Tensor) or not isinstance(value, Tensor):
            raise TypeError("action policy and critic outputs must be tensors")
        if not logits.is_floating_point() or not value.is_floating_point():
            raise TypeError("action policy and critic outputs must be floating point")
        if tuple(logits.shape) != (3,):
            raise ValueError("action policy logits must have shape [3]")
        if logits.device != state.device:
            raise ValueError("action policy logits changed device")
        if value.numel() != 1:
            raise ValueError("critic output must be scalar")
        if value.device != state.device:
            raise ValueError("critic output changed device")
        if not bool(torch.isfinite(logits).all()) or not bool(
            torch.isfinite(value).all()
        ):
            raise ValueError("action policy and critic outputs must be finite")
        priors = torch.softmax(
            logits.detach().to(device="cpu", dtype=torch.float32) / float(temperature),
            dim=0,
        )
        return priors, float(value.detach().to(device="cpu", dtype=torch.float32))

    @staticmethod
    def _validate_warm_capsule(
        capsule: BroydenWarmStart,
        state: Tensor,
        *,
        operator_id: str,
    ) -> None:
        if not isinstance(capsule, BroydenWarmStart):
            raise TypeError("successful action requires BroydenWarmStart")
        count = capsule.history_size
        if capsule.rank != 16 or capsule.operator_id != operator_id:
            raise ValueError("warm capsule rank/operator certificate changed")
        if (
            tuple(capsule.state.shape) != tuple(state.shape)
            or tuple(capsule.x_history.shape) != (count, *state.shape)
            or tuple(capsule.f_history.shape) != (count, *state.shape)
            or count > 17
        ):
            raise ValueError("warm capsule exceeded fixed rank-16 geometry")
        tensors = (capsule.state, capsule.x_history, capsule.f_history)
        if any(value.dtype != state.dtype for value in tensors):
            raise ValueError("warm capsule changed dtype")
        if any(value.device != state.device for value in tensors):
            raise ValueError("warm capsule changed device")
        if any(value.layout != torch.strided for value in tensors):
            raise ValueError("warm capsule must use strided tensor storage")
        if any(not bool(torch.isfinite(value).all()) for value in tensors):
            raise ValueError("warm capsule contains non-finite values")
        if not torch.equal(capsule.state, state):
            raise ValueError("warm capsule state differs from transition state")

    @classmethod
    def _validate_transition_batch(
        cls,
        batch: CertifiedTransitionBatchV2,
        parent: Tensor,
        *,
        tolerance: float,
        operator_id: str,
    ) -> None:
        if not isinstance(batch, CertifiedTransitionBatchV2):
            raise TypeError(
                "certified transition must return CertifiedTransitionBatchV2"
            )
        if tuple(batch.states.shape) != (3, *parent.shape):
            raise ValueError("transition states must have shape [3, *parent.shape]")
        if batch.states.dtype != parent.dtype or batch.states.device != parent.device:
            raise ValueError("transition states must preserve parent dtype and device")
        if not bool(torch.isfinite(batch.states).all()):
            raise ValueError("transition states must be finite")

        tensor_contracts = (
            (batch.success, torch.bool, "success"),
            (batch.residuals, torch.float32, "residuals"),
            (batch.iterations, torch.int64, "iterations"),
            (
                batch.linear_solve_fallbacks,
                torch.int64,
                "linear_solve_fallbacks",
            ),
            (batch.warm_used, torch.bool, "warm_used"),
            (batch.warm_rejected, torch.int64, "warm_rejected"),
        )
        for value, dtype, name in tensor_contracts:
            if not isinstance(value, Tensor):
                raise TypeError(f"transition {name} must be a tensor")
            if tuple(value.shape) != (3,) or value.dtype != dtype:
                raise ValueError(f"transition {name} has an invalid shape or dtype")
            if value.device.type != "cpu":
                raise ValueError(f"transition {name} diagnostics must stay on CPU")
        if not isinstance(batch.warm_starts, tuple) or len(batch.warm_starts) != 3:
            raise ValueError("transition warm_starts must contain three entries")
        if bool((batch.iterations < 0).any()):
            raise ValueError("transition iterations cannot be negative")
        if bool(torch.isnan(batch.residuals).any()) or bool(
            (batch.residuals < 0).any()
        ):
            raise ValueError("transition residuals must be non-negative or infinity")
        if bool((batch.linear_solve_fallbacks < 0).any()):
            raise ValueError("linear solve fallback counts cannot be negative")
        if bool(((batch.warm_rejected < 0) | (batch.warm_rejected > 1)).any()):
            raise ValueError("warm rejection counts must be zero or one")
        if bool((batch.warm_used & (batch.warm_rejected > 0)).any()):
            raise ValueError("a warm solve cannot be both used and rejected")

        for action in range(3):
            succeeded = bool(batch.success[action])
            residual = float(batch.residuals[action])
            capsule = batch.warm_starts[action]
            if succeeded:
                if not torch.isfinite(batch.residuals[action]) or residual > tolerance:
                    raise ValueError(
                        "successful transition action lacks a certified residual"
                    )
                if capsule is None:
                    raise ValueError("successful transition action lacks warm state")
                cls._validate_warm_capsule(
                    capsule,
                    batch.states[action],
                    operator_id=operator_id,
                )
            else:
                if capsule is not None:
                    raise ValueError("failed transition action retained warm state")
                if not torch.equal(batch.states[action], parent):
                    raise ValueError(
                        "failed transition action must return parent state"
                    )

    @staticmethod
    def _backup(
        tree: _CertifiedTreeBuffersV2,
        path_length: int,
        value: float,
        discount: float,
    ) -> None:
        for position in range(path_length):
            tree.node_visits[int(tree.path_nodes[position])] += 1
        backed_value = float(value)
        for position in range(path_length - 2, -1, -1):
            parent = int(tree.path_nodes[position])
            action = int(tree.path_actions[position])
            tree.action_visits[parent, action] += 1
            tree.action_value_sums[parent, action] += backed_value
            backed_value *= float(discount)

    @staticmethod
    def _best_selectable_action(tree: _CertifiedTreeBuffersV2, node: int) -> int:
        selectable = tree.children[node] >= 0
        if not bool(selectable.any()):
            return -1
        visits = tree.action_visits[node].clone()
        visits[~selectable] = -1
        if int(visits.max()) > 0:
            return int(torch.argmax(visits))
        priors = tree.priors[node].clone()
        priors[~selectable] = torch.finfo(priors.dtype).min
        return int(torch.argmax(priors))

    @staticmethod
    def _select_best_path(tree: _CertifiedTreeBuffersV2) -> tuple[list[int], int]:
        path = [0]
        node = 0
        while bool(tree.expanded[node]) and len(path) < tree.capacity:
            action = BoundedPUCTSearchV2._best_selectable_action(tree, node)
            if action < 0:
                break
            child = int(tree.children[node, action])
            if child < 0:
                break
            path.append(child)
            node = child
        return path, node

    @staticmethod
    def _make_telemetry(
        *,
        cfg: CertifiedPUCTConfigV2,
        controls: SearchControlsV2 | None,
        effective_simulations: int,
        completed: int,
        tree: _CertifiedTreeBuffersV2,
        retriever: FixedCapacityTreeRetriever,
        ledger: MacBudgetLedger,
        capacity_exhausted: bool,
        frontier_rollouts: int,
        frontier_depth: int,
        frontier_workspace_bytes: int,
        retrieval_queries: int,
        retrieval_hits: int,
        solver_calls: int,
        solver_history_peak: int,
        solver_successes: int,
        solver_iterations_total: int,
        solver_iterations_max: int,
        solver_residual_max: float,
        linear_solve_fallbacks: int,
        warm_used: int,
        warm_rejected: int,
        failed_edges: int,
        q_zero_backups: int,
        all_fail_terminals: int,
        digest: str,
        safe_for_decode: bool,
    ) -> SearchTelemetryV2:
        act_requested = controls.act_simulations if controls is not None else 0
        return SearchTelemetryV2(
            simulations_requested=cfg.simulations,
            simulations_completed=completed,
            act_requested=act_requested,
            act_applied=effective_simulations,
            act_clamped=controls is not None and act_requested > cfg.simulations,
            nodes_used=tree.node_count,
            node_capacity=tree.capacity,
            max_depth_reached=max(
                int(tree.depths[: tree.node_count].max()), frontier_depth
            ),
            capacity_exhausted=capacity_exhausted,
            frontier_rollouts=frontier_rollouts,
            retrieval_queries=retrieval_queries,
            retrieval_hits=retrieval_hits,
            solver_calls=solver_calls,
            solver_rank=16,
            solver_history_peak=solver_history_peak,
            solver_successes=solver_successes,
            solver_failures=solver_calls - solver_successes,
            solver_iterations_total=solver_iterations_total,
            solver_iterations_max=solver_iterations_max,
            solver_residual_max=solver_residual_max,
            linear_solve_fallbacks=linear_solve_fallbacks,
            warm_used=warm_used,
            warm_rejected=warm_rejected,
            failed_edges=failed_edges,
            q_zero_backups=q_zero_backups,
            all_fail_terminals=all_fail_terminals,
            mac_budget=ledger.budget,
            mac_reserved=ledger.reserved,
            mac_budget_exhausted=ledger.exhausted,
            mac_meta_policy=ledger.meta_policy,
            mac_action_policy=ledger.action_policy,
            mac_critic=ledger.critic,
            mac_retrieval=ledger.retrieval,
            mac_transition=ledger.transition,
            allocated_bytes=(
                tree.memory_bytes()
                + retriever.memory_bytes()
                + frontier_workspace_bytes
            ),
            trace_digest=digest,
            unsafe_silent_fallbacks=0,
            safe_for_decode=safe_for_decode,
        )

    @torch.no_grad()
    def search(
        self,
        request: SearchRequestV2,
        transition: CertifiedTransitionV2,
        action_policy: ActionPolicyV2,
        critic: CriticV2,
        meta_controller: MetaControllerV2,
    ) -> PUCTResultV2:
        """Run certified search, reserving every callback's worst-case MACs."""

        if not isinstance(request, SearchRequestV2):
            raise TypeError("request must be SearchRequestV2")
        root = request.root
        self._validate_root(root)
        validate_certified_transition_v2(transition)
        if (
            not callable(action_policy)
            or not callable(critic)
            or not callable(meta_controller)
        ):
            raise TypeError("policy, critic, and meta controller must be callable")

        cfg = self.config
        tree = _CertifiedTreeBuffersV2(root.detach())
        retriever = FixedCapacityTreeRetriever(tree.states)
        retriever.add(0, tree.states[0], success=True)
        ledger = MacBudgetLedger(request.mac_budget)
        seed = cfg.seed if request.seed is None else request.seed
        trace = hashlib.sha256(f"cts-v2|{seed}|{tuple(root.shape)}".encode())

        controls: SearchControlsV2 | None = None
        effective_simulations = 0
        completed = 0
        capacity_exhausted = False
        frontier_rollouts = 0
        retrieval_queries = 0
        retrieval_hits = 0
        solver_calls = 0
        solver_history_peak = 0
        solver_successes = 0
        solver_iterations_total = 0
        solver_iterations_max = 0
        solver_residual_max = 0.0
        linear_solve_fallbacks = 0
        warm_used = 0
        warm_rejected = 0
        failed_edges = 0
        q_zero_backups = 0
        all_fail_terminals = 0
        frontier_state = torch.empty_like(root)
        frontier_warm: BroydenWarmStart | None = None
        frontier_active = False
        frontier_has_state = False
        frontier_depth = 0
        frontier_anchor_node = 0
        frontier_anchor_path_length = 1
        frontier_workspace_bytes = root.numel() * root.element_size() * (2 + 2 * 17)

        if ledger.reserve("meta_policy", cfg.meta_policy_macs):
            controls = self._call_meta_controller(meta_controller, root)
            effective_simulations = min(cfg.simulations, controls.act_simulations)

        budget_stopped = controls is None
        for simulation in range(effective_simulations):
            if budget_stopped:
                break
            if frontier_active:
                if frontier_depth >= cfg.max_depth:
                    break
                policy_state = frontier_state
                if frontier_depth >= cfg.retrieval_min_depth:
                    if not ledger.reserve("retrieval", cfg.retrieval_macs):
                        budget_stopped = True
                        break
                    ancestors = retriever.retrieve(
                        frontier_state,
                        current_node=frontier_anchor_node,
                    )
                    retrieval_queries += 1
                    retrieval_hits += int(ancestors.valid.sum())
                    policy_state = ancestors.blend(
                        frontier_state,
                        cfg.retrieval_mix,
                        cfg.retrieval_temperature,
                    )
                if not ledger.reserve_policy_critic(
                    cfg.action_policy_macs, cfg.critic_macs
                ):
                    budget_stopped = True
                    break
                priors, leaf_value = self._call_policy_critic(
                    action_policy,
                    critic,
                    policy_state,
                    temperature=controls.policy_temperature,
                )
                if not ledger.reserve("transition", cfg.transition_macs):
                    budget_stopped = True
                    break
                input_version = policy_state._version
                batch = transition(
                    policy_state,
                    tree.actions,
                    tolerance=controls.tolerance,
                    warm_start=frontier_warm,
                )
                if policy_state._version != input_version:
                    raise RuntimeError("certified transition mutated its input")
                self._validate_transition_batch(
                    batch,
                    policy_state,
                    tolerance=controls.tolerance,
                    operator_id=getattr(transition, "operator_id"),
                )
                solver_calls += 3
                solver_history_peak = max(
                    solver_history_peak,
                    max(
                        (
                            max(0, capsule.history_size - 1)
                            for capsule in batch.warm_starts
                            if capsule is not None
                        ),
                        default=0,
                    ),
                )
                success_count = int(batch.success.sum())
                solver_successes += success_count
                solver_iterations_total += int(batch.iterations.sum())
                solver_iterations_max = max(
                    solver_iterations_max, int(batch.iterations.max())
                )
                finite_residuals = batch.residuals[torch.isfinite(batch.residuals)]
                if finite_residuals.numel():
                    solver_residual_max = max(
                        solver_residual_max, float(finite_residuals.max())
                    )
                linear_solve_fallbacks += int(batch.linear_solve_fallbacks.sum())
                warm_used += int(batch.warm_used.sum())
                warm_rejected += int(batch.warm_rejected.sum())
                failed_count = 3 - success_count
                failed_edges += failed_count
                q_zero_backups += failed_count
                frontier_rollouts += 1
                if success_count == 0:
                    all_fail_terminals += 1
                    self._backup(
                        tree,
                        frontier_anchor_path_length,
                        0.0,
                        cfg.discount,
                    )
                    completed += 1
                    frontier_active = False
                    trace.update(
                        f"|{simulation}:frontier:{frontier_depth}:all-failed".encode()
                    )
                    break
                masked_priors = priors.clone()
                masked_priors[~batch.success] = torch.finfo(masked_priors.dtype).min
                action = int(torch.argmax(masked_priors))
                capsule = batch.warm_starts[action]
                assert capsule is not None
                frontier_state.copy_(batch.states[action])
                frontier_warm = capsule
                frontier_depth += 1
                frontier_has_state = True
                self._backup(
                    tree,
                    frontier_anchor_path_length,
                    leaf_value,
                    cfg.discount,
                )
                completed += 1
                trace.update(
                    (
                        f"|{simulation}:frontier:{frontier_depth}:{action}:"
                        f"{batch.success.to(torch.int8).tolist()}:"
                        f"{batch.iterations.tolist()}:"
                        f"{batch.residuals.tolist()}"
                    ).encode()
                )
                continue
            path_length = 1
            node = 0
            tree.path_nodes[0] = 0

            while bool(tree.expanded[node]) and not bool(tree.terminal[node]):
                if int(tree.depths[node]) >= cfg.max_depth:
                    break
                selectable = tree.children[node] >= 0
                if not bool(selectable.any()):
                    tree.terminal[node] = True
                    break
                scores = puct_scores(
                    tree.priors[node],
                    tree.action_visits[node],
                    tree.action_value_sums[node],
                    tree.node_visits[node],
                    controls.exploration,
                )
                scores[~selectable] = torch.finfo(scores.dtype).min
                action = int(torch.argmax(scores))
                child = int(tree.children[node, action])
                if child < 0:
                    raise RuntimeError("certified selection chose a masked edge")
                if path_length >= tree.capacity:
                    raise RuntimeError("certified path exceeded the fixed arena")
                tree.path_actions[path_length - 1] = action
                tree.path_nodes[path_length] = child
                path_length += 1
                node = child

            # Explicit terminal states are backed up as zero without invoking
            # another learned callback or silently inventing a child state.
            if bool(tree.terminal[node]):
                self._backup(tree, path_length, 0.0, cfg.discount)
                completed += 1
                trace.update(f"|{simulation}:{node}:terminal".encode())
                continue

            leaf_state = tree.states[node]
            policy_state = leaf_state
            depth = int(tree.depths[node])
            if depth >= cfg.retrieval_min_depth:
                if not ledger.reserve("retrieval", cfg.retrieval_macs):
                    budget_stopped = True
                    break
                ancestors = retriever.retrieve(leaf_state, current_node=node)
                retrieval_queries += 1
                retrieval_hits += int(ancestors.valid.sum())
                policy_state = ancestors.blend(
                    leaf_state,
                    cfg.retrieval_mix,
                    cfg.retrieval_temperature,
                )

            if not ledger.reserve_policy_critic(
                cfg.action_policy_macs, cfg.critic_macs
            ):
                budget_stopped = True
                break
            priors, leaf_value = self._call_policy_critic(
                action_policy,
                critic,
                policy_state,
                temperature=controls.policy_temperature,
            )

            backed_value = leaf_value
            succeeded = torch.zeros(3, dtype=torch.bool)
            iteration_snapshot = torch.zeros(3, dtype=torch.int64)
            residual_snapshot = torch.full((3,), float("inf"), dtype=torch.float32)
            if depth < cfg.max_depth and not bool(tree.expanded[node]):
                if tree.can_expand_full_width():
                    if not ledger.reserve("transition", cfg.transition_macs):
                        # The already-obtained critic value remains a valid
                        # bounded rollout. No callback occurs after denial.
                        budget_stopped = True
                        frontier_rollouts += 1
                    else:
                        input_version = policy_state._version
                        batch = transition(
                            policy_state,
                            tree.actions,
                            tolerance=controls.tolerance,
                            warm_start=tree.warm_start(
                                node, getattr(transition, "operator_id")
                            ),
                        )
                        if policy_state._version != input_version:
                            raise RuntimeError("certified transition mutated its input")
                        self._validate_transition_batch(
                            batch,
                            policy_state,
                            tolerance=controls.tolerance,
                            operator_id=getattr(transition, "operator_id"),
                        )
                        succeeded.copy_(batch.success)
                        iteration_snapshot.copy_(batch.iterations)
                        residual_snapshot.copy_(batch.residuals)
                        solver_calls += 3
                        solver_history_peak = max(
                            solver_history_peak,
                            max(
                                (
                                    max(0, capsule.history_size - 1)
                                    for capsule in batch.warm_starts
                                    if capsule is not None
                                ),
                                default=0,
                            ),
                        )
                        success_count = int(batch.success.sum())
                        solver_successes += success_count
                        solver_iterations_total += int(batch.iterations.sum())
                        solver_iterations_max = max(
                            solver_iterations_max, int(batch.iterations.max())
                        )
                        finite_residuals = batch.residuals[
                            torch.isfinite(batch.residuals)
                        ]
                        if finite_residuals.numel():
                            solver_residual_max = max(
                                solver_residual_max, float(finite_residuals.max())
                            )
                        linear_solve_fallbacks += int(
                            batch.linear_solve_fallbacks.sum()
                        )
                        warm_used += int(batch.warm_used.sum())
                        warm_rejected += int(batch.warm_rejected.sum())

                        masked_priors = priors * batch.success.to(torch.float32)
                        prior_total = float(masked_priors.sum())
                        if prior_total > 0.0:
                            masked_priors /= prior_total
                        elif success_count > 0:
                            masked_priors.copy_(
                                batch.success.to(torch.float32) / success_count
                            )
                        tree.priors[node].copy_(masked_priors)
                        for action in range(3):
                            if bool(batch.success[action]):
                                capsule = batch.warm_starts[action]
                                assert capsule is not None
                                child = tree.add_child(
                                    node, action, batch.states[action], capsule
                                )
                                retriever.add(child, tree.states[child], success=True)
                            else:
                                tree.mark_failed(node, action)
                                failed_edges += 1
                                q_zero_backups += 1
                        tree.expanded[node] = True
                        if success_count == 0:
                            tree.terminal[node] = True
                            all_fail_terminals += 1
                            backed_value = 0.0
                else:
                    capacity_exhausted = True
                    frontier_rollouts += 1
                    # Preserve the selected learned frontier in one fixed
                    # scratch row. Subsequent simulations continue one
                    # certified width-3 transition at a time and commit only
                    # the chosen learned action to the trace, never to the
                    # saturated 301-node arena.
                    frontier_state.copy_(policy_state)
                    frontier_warm = tree.warm_start(
                        node, getattr(transition, "operator_id")
                    )
                    frontier_active = True
                    frontier_has_state = True
                    frontier_depth = depth
                    frontier_anchor_node = node
                    frontier_anchor_path_length = path_length
            elif not bool(tree.expanded[node]):
                frontier_rollouts += 1

            self._backup(tree, path_length, backed_value, cfg.discount)
            completed += 1
            trace.update(
                (
                    f"|{simulation}:{node}:{depth}:"
                    f"{succeeded.to(torch.int8).tolist()}:"
                    f"{iteration_snapshot.tolist()}:"
                    f"{residual_snapshot.tolist()}"
                ).encode()
            )
            if budget_stopped:
                break

        root_visits = tree.action_visits[0].clone()
        root_priors = tree.priors[0].clone()
        root_q = torch.where(
            root_visits > 0,
            tree.action_value_sums[0] / root_visits.to(torch.float32).clamp_min(1.0),
            torch.zeros(3, dtype=torch.float32),
        )
        best_action_int = self._best_selectable_action(tree, 0)
        best_path_values, best_node = self._select_best_path(tree)
        tree_max_depth = int(tree.depths[: tree.node_count].max())
        frontier_is_best = frontier_has_state and frontier_depth > tree_max_depth
        best_state = frontier_state if frontier_is_best else tree.states[best_node]
        safe_for_decode = (
            completed > 0
            and best_action_int >= 0
            and bool(torch.isfinite(best_state).all())
        )
        telemetry = self._make_telemetry(
            cfg=cfg,
            controls=controls,
            effective_simulations=effective_simulations,
            completed=completed,
            tree=tree,
            retriever=retriever,
            ledger=ledger,
            capacity_exhausted=capacity_exhausted,
            frontier_rollouts=frontier_rollouts,
            frontier_depth=frontier_depth if frontier_has_state else 0,
            frontier_workspace_bytes=frontier_workspace_bytes,
            retrieval_queries=retrieval_queries,
            retrieval_hits=retrieval_hits,
            solver_calls=solver_calls,
            solver_history_peak=solver_history_peak,
            solver_successes=solver_successes,
            solver_iterations_total=solver_iterations_total,
            solver_iterations_max=solver_iterations_max,
            solver_residual_max=solver_residual_max,
            linear_solve_fallbacks=linear_solve_fallbacks,
            warm_used=warm_used,
            warm_rejected=warm_rejected,
            failed_edges=failed_edges,
            q_zero_backups=q_zero_backups,
            all_fail_terminals=all_fail_terminals,
            digest=trace.hexdigest(),
            safe_for_decode=safe_for_decode,
        )
        return PUCTResultV2(
            best_state=best_state.clone(),
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
    "ActionPolicyV2",
    "AncestorBatch",
    "BoundedPUCTSearch",
    "BoundedPUCTSearchV2",
    "CertifiedBroydenTransitionV2",
    "CertifiedPUCTConfigV2",
    "CertifiedTransitionBatchV2",
    "CertifiedTransitionV2",
    "ContractiveBroydenTransition",
    "CriticV2",
    "DeclaredDEQTensorTransition",
    "FixedCapacityTreeRetriever",
    "MacBudgetLedger",
    "MetaControllerV2",
    "PUCTConfig",
    "PUCTResult",
    "PUCTResultV2",
    "PolicyValueFn",
    "SearchControlsV2",
    "SearchRequestV2",
    "SearchTelemetry",
    "SearchTelemetryV2",
    "SemanticAncestorRetriever",
    "TensorTransition",
    "TransitionContractError",
    "TransitionFn",
    "deq_tensor_transition",
    "puct_scores",
    "validate_deq_tensor_transition",
    "validate_certified_transition_v2",
]
