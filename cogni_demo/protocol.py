"""Bounded, versioned stdout protocol for the local validation worker."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from typing import Any, Mapping


EVENT_SENTINEL = "@@COGNI_EVENT@@"
PROTOCOL_VERSION = 2
MAX_EVENT_LINE_BYTES = 16 * 1024
MAX_TEXT_LENGTH = 256
ABSOLUTE_VRAM_LIMIT_GIB = 16.7

PHASE_STAGES = (
    "verifying",
    "loading_model",
    "building_runtime",
    "running_inference",
    "validating_decode_bridge",
    "postcheck",
)
TERMINAL_STAGE = "complete"


class ProtocolError(ValueError):
    """Raised when a worker line violates the bounded event contract."""


def _is_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_number(value: object) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _bounded_text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value or len(value) > MAX_TEXT_LENGTH:
        raise ProtocolError(f"{name} must be a bounded non-empty string")
    return value


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ProtocolError("worker event contains a duplicate JSON key")
        result[key] = value
    return result


@dataclass(frozen=True)
class WorkerEvent:
    sequence: int
    kind: str
    stage: str
    progress: int
    metrics: Mapping[str, Any] | None = None

    def as_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "v": PROTOCOL_VERSION,
            "seq": self.sequence,
            "kind": self.kind,
            "stage": self.stage,
            "progress": self.progress,
        }
        if self.metrics is not None:
            payload["metrics"] = dict(self.metrics)
        return payload


class EventEmitter:
    """Emit optional sentinel JSONL without changing legacy stdout lines."""

    def __init__(self, enabled: bool = False) -> None:
        self.enabled = bool(enabled)
        self._sequence = 0

    def phase(self, stage: str, progress: int) -> None:
        self._emit("phase", stage, progress)

    def result(self, metrics: Mapping[str, Any]) -> None:
        normalized = validate_terminal_metrics(metrics)
        self._emit("result", TERMINAL_STAGE, 100, normalized)

    def _emit(
        self,
        kind: str,
        stage: str,
        progress: int,
        metrics: Mapping[str, Any] | None = None,
    ) -> None:
        if not self.enabled:
            return
        self._sequence += 1
        event = WorkerEvent(self._sequence, kind, stage, progress, metrics)
        encoded = json.dumps(
            event.as_payload(), ensure_ascii=True, separators=(",", ":")
        )
        if len(encoded.encode("utf-8")) > MAX_EVENT_LINE_BYTES:
            raise ProtocolError("encoded worker event exceeds the line budget")
        print(EVENT_SENTINEL + encoded, flush=True)


_REQUIRED_METRICS: dict[str, str] = {
    "verified_files": "int",
    "model_class": "str",
    "hidden_size": "int",
    "load_seconds": "number",
    "inference_seconds": "number",
    "requested_depth": "int",
    "reached_depth": "int",
    "nodes_used": "int",
    "node_capacity": "int",
    "search_allocated_bytes": "int",
    "transition_converged": "bool",
    "transition_residual": "number",
    "transition_used_fallback": "bool",
    "cts_protocol_version": "str",
    "safe_for_decode": "bool",
    "unsafe_silent_fallbacks": "int",
    "linear_solve_fallbacks": "int",
    "solver_rank": "int",
    "solver_history_peak": "int",
    "solver_failures": "int",
    "failed_edges": "int",
    "q_zero_backups": "int",
    "mac_budget": "int",
    "mac_reserved": "int",
    "act_applied": "int",
    "trace_digest": "str",
    "causal_bridge_answer_bearing": "bool",
    "causal_bridge_bias_nonzero": "bool",
    "causal_bridge_bias_max": "number",
    "conditioned_generated_tokens": "int",
    "peak_vram_gib": "number",
    "vram_limit_gib": "number",
    "finite": "bool",
    "device": "str",
}


def validate_terminal_metrics(metrics: Mapping[str, Any]) -> dict[str, Any]:
    """Type-check and certify the terminal runtime postconditions."""

    if not isinstance(metrics, Mapping):
        raise ProtocolError("terminal metrics must be an object")
    if set(metrics) != set(_REQUIRED_METRICS):
        missing = sorted(set(_REQUIRED_METRICS) - set(metrics))
        extra = sorted(set(metrics) - set(_REQUIRED_METRICS))
        raise ProtocolError(
            f"terminal metric keys differ: missing={missing}, extra={extra}"
        )
    normalized: dict[str, Any] = {}
    for name, expected in _REQUIRED_METRICS.items():
        value = metrics[name]
        if expected == "int":
            if not _is_int(value):
                raise ProtocolError(f"{name} must be an integer")
            normalized[name] = int(value)
        elif expected == "number":
            if not _is_number(value):
                raise ProtocolError(f"{name} must be finite numeric data")
            normalized[name] = float(value)
        elif expected == "bool":
            if not isinstance(value, bool):
                raise ProtocolError(f"{name} must be a boolean")
            normalized[name] = value
        else:
            normalized[name] = _bounded_text(value, name)

    if normalized["verified_files"] <= 0 or normalized["hidden_size"] <= 0:
        raise ProtocolError("artifact and hidden-size metrics must be positive")
    if normalized["load_seconds"] < 0 or normalized["inference_seconds"] < 0:
        raise ProtocolError("timing metrics cannot be negative")
    requested = normalized["requested_depth"]
    if requested != 100 or normalized["reached_depth"] != requested:
        raise ProtocolError("the integrated demo must reach requested depth 100")
    nodes = normalized["nodes_used"]
    capacity = normalized["node_capacity"]
    if nodes <= 0 or capacity <= 0 or nodes > capacity:
        raise ProtocolError("tree node metrics violate the fixed arena")
    if normalized["search_allocated_bytes"] <= 0:
        raise ProtocolError("search allocation telemetry must be positive")
    if not normalized["transition_converged"] or not normalized["finite"]:
        raise ProtocolError("runtime convergence and finiteness are mandatory")
    if normalized["transition_residual"] < 0:
        raise ProtocolError("transition residual cannot be negative")
    if normalized["cts_protocol_version"] != "SearchRequestV2":
        raise ProtocolError("runtime validation must use certified SearchRequestV2")
    if (
        normalized["safe_for_decode"] is not True
        or normalized["unsafe_silent_fallbacks"] != 0
        or normalized["linear_solve_fallbacks"] != 0
        or normalized["transition_used_fallback"] is not False
    ):
        raise ProtocolError("CTS V2 decode safety telemetry failed closed")
    if normalized["solver_rank"] != 16 or not (
        1 <= normalized["solver_history_peak"] <= 16
    ):
        raise ProtocolError("CTS V2 must use bounded rank-16 Broyden history")
    if (
        normalized["solver_failures"] < 0
        or normalized["failed_edges"] != normalized["solver_failures"]
        or normalized["q_zero_backups"] != normalized["failed_edges"]
    ):
        raise ProtocolError("failed CTS edges must receive exactly one Q=0 backup")
    if not (0 < normalized["mac_reserved"] <= normalized["mac_budget"]):
        raise ProtocolError("CTS V2 MAC ledger violates its request budget")
    if normalized["act_applied"] != 301:
        raise ProtocolError("Depth-100 validation must apply the certified ACT floor")
    digest = normalized["trace_digest"]
    if len(digest) != 64 or any(
        character not in "0123456789abcdef" for character in digest
    ):
        raise ProtocolError("CTS V2 trace digest must be lowercase SHA-256 hex")
    if (
        not normalized["causal_bridge_answer_bearing"]
        or not normalized["causal_bridge_bias_nonzero"]
    ):
        raise ProtocolError("the causal decode bridge must be active and non-zero")
    bridge_max = normalized["causal_bridge_bias_max"]
    if not 0 < bridge_max <= 0.1:
        raise ProtocolError("causal bridge bias crossed its certified bound")
    if normalized["conditioned_generated_tokens"] != 1:
        raise ProtocolError("causal bridge validation must decode exactly one token")
    limit = normalized["vram_limit_gib"]
    peak = normalized["peak_vram_gib"]
    if not 0 < limit <= ABSOLUTE_VRAM_LIMIT_GIB or not 0 <= peak <= limit:
        raise ProtocolError("VRAM telemetry crossed the absolute safety limit")
    return normalized


def parse_event_line(line: str | bytes) -> WorkerEvent | None:
    """Parse one stdout line; ordinary legacy output returns ``None``."""

    if isinstance(line, bytes):
        if len(line) > MAX_EVENT_LINE_BYTES + len(EVENT_SENTINEL):
            raise ProtocolError("worker event line exceeds the byte budget")
        try:
            text = line.decode("utf-8", errors="strict")
        except UnicodeDecodeError as exc:
            raise ProtocolError("worker event is not valid UTF-8") from exc
    elif isinstance(line, str):
        text = line
        if len(text.encode("utf-8")) > MAX_EVENT_LINE_BYTES + len(EVENT_SENTINEL):
            raise ProtocolError("worker event line exceeds the byte budget")
    else:
        raise TypeError("worker line must be text or bytes")
    text = text.rstrip("\r\n")
    if not text.startswith(EVENT_SENTINEL):
        return None
    raw = text[len(EVENT_SENTINEL) :]
    try:
        payload = json.loads(raw, object_pairs_hook=_unique_object)
    except (json.JSONDecodeError, RecursionError) as exc:
        raise ProtocolError("worker event contains invalid JSON") from exc
    if not isinstance(payload, dict):
        raise ProtocolError("worker event payload must be an object")
    allowed = {"v", "seq", "kind", "stage", "progress", "metrics"}
    if not set(payload) <= allowed:
        raise ProtocolError("worker event contains unknown fields")
    required = {"v", "seq", "kind", "stage", "progress"}
    if not required <= set(payload):
        raise ProtocolError("worker event is missing required fields")
    if not _is_int(payload["v"]) or payload["v"] != PROTOCOL_VERSION:
        raise ProtocolError("worker event protocol version is unsupported")
    if not _is_int(payload["seq"]) or payload["seq"] <= 0:
        raise ProtocolError("worker sequence must be a positive integer")
    if not _is_int(payload["progress"]) or not 0 <= payload["progress"] <= 100:
        raise ProtocolError("worker progress must be an integer in [0, 100]")
    kind = payload["kind"]
    stage = payload["stage"]
    metrics = payload.get("metrics")
    if kind == "phase":
        if stage not in PHASE_STAGES or metrics is not None:
            raise ProtocolError("phase event has an invalid stage or metrics")
    elif kind == "result":
        if stage != TERMINAL_STAGE or payload["progress"] != 100:
            raise ProtocolError("result event must be complete at 100 percent")
        metrics = validate_terminal_metrics(metrics)
    else:
        raise ProtocolError("worker event kind is unsupported")
    return WorkerEvent(
        int(payload["seq"]), kind, stage, int(payload["progress"]), metrics
    )


__all__ = [
    "ABSOLUTE_VRAM_LIMIT_GIB",
    "EVENT_SENTINEL",
    "EventEmitter",
    "MAX_EVENT_LINE_BYTES",
    "PHASE_STAGES",
    "PROTOCOL_VERSION",
    "ProtocolError",
    "TERMINAL_STAGE",
    "WorkerEvent",
    "parse_event_line",
    "validate_terminal_metrics",
]
