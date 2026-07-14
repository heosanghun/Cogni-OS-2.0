from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from threading import RLock

import torch
from torch import Tensor


@dataclass(frozen=True)
class OODDecision:
    allow_fast_path: bool
    distance: float
    threshold: float


@dataclass(frozen=True)
class TensorOODDecision:
    """Tensor-only hot-path result: allow mask, distance, and threshold."""

    allow_fast_path: Tensor
    distance: Tensor
    threshold: Tensor


@dataclass(frozen=True)
class _SessionDistribution:
    centroid: Tensor
    threshold: Tensor


class ContrastiveSessionRouter:
    """Fixed-capacity near-OOD fallback router for Fast Weight sessions."""

    def __init__(self, max_sessions: int = 16, minimum_threshold: float = 0.02):
        if max_sessions <= 0:
            raise ValueError("max_sessions must be positive")
        if minimum_threshold <= 0:
            raise ValueError("minimum_threshold must be positive")
        self.max_sessions = max_sessions
        self.minimum_threshold = minimum_threshold
        self._sessions: OrderedDict[str, _SessionDistribution] = OrderedDict()
        self._lock = RLock()

    @property
    def session_ids(self) -> tuple[str, ...]:
        with self._lock:
            return tuple(self._sessions)

    def discard_many(self, session_ids: tuple[str, ...]) -> None:
        """Remove cache-evicted sessions from the OOD control plane atomically."""

        if not isinstance(session_ids, tuple) or any(
            not isinstance(session_id, str) for session_id in session_ids
        ):
            raise TypeError("session_ids must be a tuple of strings")
        with self._lock:
            for session_id in session_ids:
                self._sessions.pop(session_id, None)

    @staticmethod
    def _normalize(values: Tensor) -> Tensor:
        return values / values.norm(dim=-1, keepdim=True).clamp_min(1e-8)

    @torch.no_grad()
    def calibrate(
        self, session_id: str, embeddings: Tensor, quantile: float = 0.99
    ) -> tuple[str, ...]:
        if not session_id:
            raise ValueError("session_id must be non-empty")
        if embeddings.ndim != 2 or embeddings.shape[0] < 2:
            raise ValueError(
                "embeddings must have shape [samples, features] with at least two samples"
            )
        if not 0.5 <= quantile < 1.0:
            raise ValueError("quantile must be in [0.5, 1)")
        normalized = self._normalize(embeddings.detach().float())
        centroid = self._normalize(normalized.mean(0, keepdim=True)).squeeze(0)
        distances = 1.0 - normalized @ centroid
        threshold = torch.maximum(
            torch.quantile(distances, quantile),
            distances.new_tensor(self.minimum_threshold),
        )
        evicted = []
        with self._lock:
            self._sessions.pop(session_id, None)
            while len(self._sessions) >= self.max_sessions:
                old, _ = self._sessions.popitem(last=False)
                evicted.append(old)
            self._sessions[session_id] = _SessionDistribution(
                centroid.cpu(), threshold.detach().cpu()
            )
            return tuple(evicted)

    @torch.no_grad()
    def route_tensor(
        self, session_id: str, query_embedding: Tensor
    ) -> TensorOODDecision:
        """Route without extracting Python scalars or synchronizing a GPU."""

        with self._lock:
            try:
                distribution = self._sessions.pop(session_id)
            except KeyError:
                return TensorOODDecision(
                    query_embedding.new_tensor(False, dtype=torch.bool),
                    query_embedding.new_tensor(float("inf"), dtype=torch.float32),
                    query_embedding.new_tensor(0.0, dtype=torch.float32),
                )
            self._sessions[session_id] = distribution
        query = (
            query_embedding.detach()
            .float()
            .reshape(-1, query_embedding.shape[-1])
            .mean(0)
        )
        query = self._normalize(query.unsqueeze(0)).squeeze(0)
        centroid = distribution.centroid.to(
            device=query.device, dtype=query.dtype, non_blocking=True
        )
        threshold = distribution.threshold.to(
            device=query.device, dtype=query.dtype, non_blocking=True
        )
        distance = 1.0 - query @ centroid
        return TensorOODDecision(distance <= threshold, distance, threshold)

    @torch.no_grad()
    def route(self, session_id: str, query_embedding: Tensor) -> OODDecision:
        """Control-plane compatibility wrapper around the tensor hot path."""

        decision = self.route_tensor(session_id, query_embedding)
        return OODDecision(
            bool(decision.allow_fast_path.detach().cpu()),
            float(decision.distance.detach().cpu()),
            float(decision.threshold.detach().cpu()),
        )
