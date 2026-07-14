from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import math
from threading import RLock
from time import monotonic
from typing import Any, Callable

import torch
from torch import Tensor, nn


@dataclass(frozen=True, slots=True)
class PCASState:
    """Tensor state for one session's persistent regime detector."""

    regime: Tensor
    enter_streak: Tensor
    exit_streak: Tensor
    dwell: Tensor
    drift_ema: Tensor
    switch_count: Tensor

    def detached_clone(self) -> PCASState:
        return PCASState(
            self.regime.detach().clone(),
            self.enter_streak.detach().clone(),
            self.exit_streak.detach().clone(),
            self.dwell.detach().clone(),
            self.drift_ema.detach().clone(),
            self.switch_count.detach().clone(),
        )


@dataclass(frozen=True, slots=True)
class PCASOutput:
    regime: Tensor
    distances: Tensor
    state: PCASState
    calibrated: Tensor

    def __iter__(self):
        """Retain the historical ``regime, distance = monitor(x)`` API."""

        yield self.regime
        yield self.distances


class PCASMonitor(nn.Module):
    """FP32 Mahalanobis detector with hysteresis and persistent-drift gating.

    Calibration is global and immutable during inference. Dynamic regime,
    streak, dwell, and drift values are supplied by the caller, which prevents
    one user's anomaly stream from changing another user's state.
    """

    def __init__(
        self,
        input_dim: int,
        *,
        enter_streak: int = 3,
        exit_streak: int = 5,
        minimum_dwell: int = 4,
        drift_decay: float = 0.95,
        drift_enter_ratio: float = 1.05,
    ) -> None:
        super().__init__()
        if input_dim < 1:
            raise ValueError("input_dim must be positive")
        if enter_streak < 1 or exit_streak < 1 or minimum_dwell < 0:
            raise ValueError("PCAS streaks must be positive and dwell non-negative")
        if not 0.0 <= drift_decay < 1.0:
            raise ValueError("drift_decay must be in [0, 1)")
        if not math.isfinite(drift_enter_ratio) or drift_enter_ratio <= 1.0:
            raise ValueError("drift_enter_ratio must be finite and greater than 1")
        self.input_dim = input_dim
        self.required_enter_streak = enter_streak
        self.required_exit_streak = exit_streak
        self.minimum_dwell = minimum_dwell
        self.drift_decay = drift_decay
        self.drift_enter_ratio = drift_enter_ratio
        # These buffers are deliberately FP32. The certificate path converts
        # incoming activations rather than lowering covariance precision.
        self.register_buffer("mean", torch.zeros(input_dim, dtype=torch.float32))
        self.register_buffer("cov_inv", torch.eye(input_dim, dtype=torch.float32))
        self.register_buffer(
            "enter_threshold",
            torch.tensor(torch.finfo(torch.float32).max, dtype=torch.float32),
        )
        self.register_buffer(
            "exit_threshold",
            torch.tensor(torch.finfo(torch.float32).max, dtype=torch.float32),
        )
        self.register_buffer("calibrated", torch.tensor(False))

    @torch.no_grad()
    def fit(
        self,
        observations: Tensor,
        quantile: float = 0.99,
        regularization: float = 1e-4,
        *,
        exit_quantile: float = 0.90,
    ) -> None:
        if (
            observations.ndim != 2
            or observations.shape[0] < max(2, observations.shape[1] + 1)
            or observations.shape[1] != self.input_dim
        ):
            raise ValueError(
                "calibration requires [samples, input_dim] with samples > input_dim"
            )
        if (
            not torch.is_floating_point(observations)
            or not torch.isfinite(observations).all()
        ):
            raise ValueError("calibration observations must be finite floating point")
        if not 0.5 <= exit_quantile < quantile < 1.0:
            raise ValueError("require 0.5 <= exit_quantile < quantile < 1")
        if not math.isfinite(regularization) or regularization <= 0.0:
            raise ValueError("regularization must be finite and positive")

        work = observations.detach().to(device=self.mean.device, dtype=torch.float32)
        mean = work.mean(0)
        centered = work - mean
        cov = centered.T @ centered / float(work.shape[0] - 1)
        cov = cov + regularization * torch.eye(
            self.input_dim, device=work.device, dtype=torch.float32
        )
        cov_inv = torch.linalg.pinv(cov, hermitian=True)
        quadratic = ((centered @ cov_inv) * centered).sum(-1)
        quadratic = torch.nan_to_num(
            quadratic, nan=1e30, posinf=1e30, neginf=0.0
        ).clamp(0.0, 1e30)
        distances = torch.sqrt(quadratic)
        enter = torch.quantile(distances, quantile).clamp_min(1e-6)
        exit_ = torch.quantile(distances, exit_quantile).clamp_min(0.0)
        # Degenerate but valid calibration sets (for example, an all-zero
        # baseline) still receive a strict numerical hysteresis band.
        exit_ = torch.minimum(exit_, enter * (1.0 - 1e-3))
        if not (
            torch.isfinite(cov_inv).all()
            and torch.isfinite(enter)
            and torch.isfinite(exit_)
            and bool((exit_ < enter).detach().cpu())
        ):
            raise RuntimeError(
                "PCAS calibration did not produce a finite hysteresis band"
            )
        self.mean.copy_(mean)
        self.cov_inv.copy_(cov_inv)
        self.enter_threshold.copy_(enter)
        self.exit_threshold.copy_(exit_)
        self.calibrated.fill_(True)

    def initial_state(self, device: torch.device | str | None = None) -> PCASState:
        target = self.mean.device if device is None else torch.device(device)
        zero_long = torch.zeros((), device=target, dtype=torch.long)
        zero_float = torch.zeros((), device=target, dtype=torch.float32)
        return PCASState(
            regime=zero_long,
            enter_streak=zero_long.clone(),
            exit_streak=zero_long.clone(),
            dwell=torch.full((), self.minimum_dwell, device=target, dtype=torch.long),
            drift_ema=zero_float,
            switch_count=zero_long.clone(),
        )

    @staticmethod
    def _validate_state(state: PCASState, device: torch.device) -> None:
        if not isinstance(state, PCASState):
            raise TypeError("previous PCAS state must be PCASState")
        for name in (
            "regime",
            "enter_streak",
            "exit_streak",
            "dwell",
            "drift_ema",
            "switch_count",
        ):
            value = getattr(state, name)
            if (
                not isinstance(value, Tensor)
                or value.ndim != 0
                or value.device != device
            ):
                raise ValueError(f"PCAS state {name} must be a scalar on {device}")
            if value.is_floating_point() and not torch.isfinite(value):
                raise ValueError(f"PCAS state {name} must be finite")

    def forward(
        self, observations: Tensor, previous_state: PCASState | None = None
    ) -> PCASOutput:
        if observations.ndim != 2 or observations.shape[1] != self.input_dim:
            raise ValueError("observations must have shape [batch, input_dim]")
        if (
            not torch.is_floating_point(observations)
            or not torch.isfinite(observations).all()
        ):
            raise ValueError("observations must be finite floating point")
        device = observations.device
        if self.mean.device != device:
            raise ValueError("PCAS calibration and observations must share a device")
        state = self.initial_state(device) if previous_state is None else previous_state
        self._validate_state(state, device)

        work = observations.to(dtype=torch.float32)
        centered = work - self.mean.float()
        quadratic = ((centered @ self.cov_inv.float()) * centered).sum(-1)
        # A finite but extremely large observation can overflow the FP32
        # quadratic form. Saturation remains an unambiguous crisis signal and
        # keeps all exported telemetry finite.
        quadratic = torch.nan_to_num(
            quadratic, nan=1e30, posinf=1e30, neginf=0.0
        ).clamp(0.0, 1e30)
        distance = torch.sqrt(quadratic)
        score = distance.mean()
        safe_enter = self.enter_threshold.float().clamp_min(1e-6)
        drift = self.drift_decay * state.drift_ema.float() + (
            1.0 - self.drift_decay
        ) * (score / safe_enter).clamp(max=1e6)
        calibrated = self.calibrated.to(device=device)
        # A decaying historical drift signal must not re-enter crisis while
        # the current sample is already inside the low/exit band.  Without
        # this guard, a clean recovery can oscillate crisis->normal->crisis
        # until the EMA drains, despite every new observation being normal.
        current_not_low = score > self.exit_threshold.float()
        high = calibrated & (
            (score >= self.enter_threshold.float())
            | (current_not_low & (drift >= self.drift_enter_ratio))
        )
        low = calibrated & (score <= self.exit_threshold.float())
        one = torch.ones((), device=device, dtype=torch.long)
        zero = torch.zeros((), device=device, dtype=torch.long)
        enter_streak = torch.where(high, state.enter_streak + one, zero)
        exit_streak = torch.where(low, state.exit_streak + one, zero)
        eligible = state.dwell >= self.minimum_dwell
        enter = (
            (state.regime == 0)
            & eligible
            & (enter_streak >= self.required_enter_streak)
        )
        exit_ = (
            (state.regime == 1) & eligible & (exit_streak >= self.required_exit_streak)
        )
        next_regime = torch.where(
            enter,
            one,
            torch.where(exit_, zero, state.regime.to(dtype=torch.long)),
        )
        switched = next_regime != state.regime
        next_state = PCASState(
            regime=next_regime,
            enter_streak=torch.where(switched, zero, enter_streak),
            exit_streak=torch.where(switched, zero, exit_streak),
            dwell=torch.where(switched, zero, state.dwell + one),
            drift_ema=drift,
            switch_count=state.switch_count + switched.to(torch.long),
        )
        return PCASOutput(next_regime, distance, next_state, calibrated)


@dataclass(frozen=True, slots=True)
class SwarmSessionState:
    joint_state: Tensor
    pcas_state: PCASState

    @property
    def storage_bytes(self) -> int:
        tensors = (
            self.joint_state,
            self.pcas_state.regime,
            self.pcas_state.enter_streak,
            self.pcas_state.exit_streak,
            self.pcas_state.dwell,
            self.pcas_state.drift_ema,
            self.pcas_state.switch_count,
        )
        return sum(value.numel() * value.element_size() for value in tensors)

    def detached_clone(self) -> SwarmSessionState:
        return SwarmSessionState(
            self.joint_state.detach().clone(), self.pcas_state.detached_clone()
        )


@dataclass(slots=True)
class _CacheEntry:
    state: SwarmSessionState
    last_access: float


class SwarmSessionStateCache:
    """Thread-safe, TTL/LRU-bounded store for advisory System-4 state."""

    def __init__(
        self,
        *,
        max_sessions: int = 16,
        ttl_seconds: float = 900.0,
        max_state_bytes: int = 64 * 1024 * 1024,
        clock: Callable[[], float] = monotonic,
    ) -> None:
        if max_sessions < 1:
            raise ValueError("max_sessions must be positive")
        if not math.isfinite(ttl_seconds) or ttl_seconds <= 0.0:
            raise ValueError("ttl_seconds must be finite and positive")
        if max_state_bytes < 1:
            raise ValueError("max_state_bytes must be positive")
        self.max_sessions = max_sessions
        self.ttl_seconds = ttl_seconds
        self.max_state_bytes = max_state_bytes
        self._clock = clock
        self._entries: OrderedDict[str, _CacheEntry] = OrderedDict()
        self._storage_bytes = 0
        self._lock = RLock()

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
    def _validate_state(state: SwarmSessionState) -> None:
        if not isinstance(state, SwarmSessionState):
            raise TypeError("state must be SwarmSessionState")
        tensor = state.joint_state
        if (
            tensor.ndim != 3
            or not torch.is_floating_point(tensor)
            or not torch.isfinite(tensor).all()
        ):
            raise ValueError("joint_state must be a finite rank-3 floating tensor")
        PCASMonitor._validate_state(state.pcas_state, tensor.device)

    def _purge_expired_locked(self, now: float) -> int:
        expired = [
            key
            for key, entry in self._entries.items()
            if now - entry.last_access >= self.ttl_seconds
        ]
        for key in expired:
            self._storage_bytes -= self._entries.pop(key).state.storage_bytes
        return len(expired)

    def get(self, session_id: str) -> SwarmSessionState | None:
        self._validate_session_id(session_id)
        with self._lock:
            now = self._clock()
            self._purge_expired_locked(now)
            entry = self._entries.get(session_id)
            if entry is None:
                return None
            entry.last_access = now
            self._entries.move_to_end(session_id)
            return entry.state.detached_clone()

    def put(self, session_id: str, state: SwarmSessionState) -> None:
        self._validate_session_id(session_id)
        self._validate_state(state)
        owned = state.detached_clone()
        size = owned.storage_bytes
        if size > self.max_state_bytes:
            raise MemoryError("one swarm state exceeds the cache byte budget")
        with self._lock:
            now = self._clock()
            self._purge_expired_locked(now)
            old = self._entries.pop(session_id, None)
            if old is not None:
                self._storage_bytes -= old.state.storage_bytes
            while self._entries and (
                len(self._entries) >= self.max_sessions
                or self._storage_bytes + size > self.max_state_bytes
            ):
                _, evicted = self._entries.popitem(last=False)
                self._storage_bytes -= evicted.state.storage_bytes
            self._entries[session_id] = _CacheEntry(owned, now)
            self._storage_bytes += size

    def process(self, session_id: str, swarm: Any, observations: Tensor) -> Any:
        """Atomically execute and commit one safe advisory state transition."""

        self._validate_session_id(session_id)
        with self._lock:
            now = self._clock()
            self._purge_expired_locked(now)
            entry = self._entries.get(session_id)
            previous = None if entry is None else entry.state
            output = swarm(
                observations,
                None if previous is None else previous.joint_state,
                pcas_state=None if previous is None else previous.pcas_state,
            )
            safe = bool(output.safe_for_advice.detach().cpu())
            converged = bool(output.converged.detach().cpu())
            if safe and converged:
                self.put(
                    session_id,
                    SwarmSessionState(output.joint_state, output.pcas_state),
                )
            return output

    def remove(self, session_id: str) -> bool:
        self._validate_session_id(session_id)
        with self._lock:
            entry = self._entries.pop(session_id, None)
            if entry is None:
                return False
            self._storage_bytes -= entry.state.storage_bytes
            return True

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()
            self._storage_bytes = 0

    @property
    def session_count(self) -> int:
        with self._lock:
            self._purge_expired_locked(self._clock())
            return len(self._entries)

    @property
    def storage_bytes(self) -> int:
        with self._lock:
            self._purge_expired_locked(self._clock())
            return self._storage_bytes
