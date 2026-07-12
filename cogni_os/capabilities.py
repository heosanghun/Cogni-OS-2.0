"""Versioned, evidence-bearing capability states for the product boundary.

Lifecycle readiness and functional authority are deliberately separate.  A
service can be ready while a research module remains advisory or disabled.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class EvidenceClass(str, Enum):
    """The only evidence classes accepted by the public Fact-book."""

    MEASURED = "measured"
    VERIFIED = "verified"
    TARGET = "target"
    PLAN = "plan"


class CapabilityState(str, Enum):
    """Ordered promotion states plus explicit conditional operating modes."""

    DISABLED = "disabled"
    RESEARCH = "research"
    ADVISORY = "advisory"
    CANARY = "canary"
    AUTHORITATIVE = "authoritative"
    GATED = "gated"
    NIGHT_ONLY = "night_only"
    PROPOSAL_ONLY = "proposal_only"


@dataclass(frozen=True, slots=True)
class CapabilityRecord:
    name: str
    state: CapabilityState
    evidence: EvidenceClass
    answer_bearing: bool
    runtime_mutation_allowed: bool
    detail: str

    def __post_init__(self) -> None:
        if (
            not isinstance(self.name, str)
            or not self.name
            or len(self.name) > 64
            or any(
                character not in "abcdefghijklmnopqrstuvwxyz0123456789_-"
                for character in self.name
            )
        ):
            raise ValueError("capability name must be bounded lowercase ASCII")
        if not isinstance(self.state, CapabilityState):
            raise TypeError("state must be a CapabilityState")
        if not isinstance(self.evidence, EvidenceClass):
            raise TypeError("evidence must be an EvidenceClass")
        if not isinstance(self.answer_bearing, bool):
            raise TypeError("answer_bearing must be bool")
        if not isinstance(self.runtime_mutation_allowed, bool):
            raise TypeError("runtime_mutation_allowed must be bool")
        if not isinstance(self.detail, str) or not 1 <= len(self.detail) <= 256:
            raise ValueError("capability detail must contain 1-256 characters")
        if (
            self.state
            in {
                CapabilityState.DISABLED,
                CapabilityState.RESEARCH,
                CapabilityState.ADVISORY,
                CapabilityState.NIGHT_ONLY,
                CapabilityState.PROPOSAL_ONLY,
            }
            and self.answer_bearing
        ):
            raise ValueError(f"{self.state.value} capability cannot be answer-bearing")
        if (
            self.state is CapabilityState.PROPOSAL_ONLY
            and self.runtime_mutation_allowed
        ):
            raise ValueError(
                "proposal-only capability cannot mutate the active runtime"
            )

    def as_payload(self) -> dict[str, object]:
        return {
            "name": self.name,
            "state": self.state.value,
            "evidence": self.evidence.value,
            "answer_bearing": self.answer_bearing,
            "runtime_mutation_allowed": self.runtime_mutation_allowed,
            "detail": self.detail,
        }


class CapabilityRegistry:
    """Immutable, deterministically ordered capability registry."""

    def __init__(self, records: tuple[CapabilityRecord, ...]) -> None:
        if not records:
            raise ValueError("capability registry cannot be empty")
        ordered = tuple(sorted(records, key=lambda record: record.name))
        names = tuple(record.name for record in ordered)
        if len(set(names)) != len(names):
            raise ValueError("capability names must be unique")
        self._records = ordered
        self._lookup = {record.name: record for record in ordered}

    @property
    def records(self) -> tuple[CapabilityRecord, ...]:
        return self._records

    def require(self, name: str) -> CapabilityRecord:
        try:
            return self._lookup[name]
        except KeyError as exc:
            raise KeyError(f"unknown capability: {name}") from exc

    def as_payload(self) -> list[dict[str, object]]:
        return [record.as_payload() for record in self._records]


def baseline_capability_registry() -> CapabilityRegistry:
    """Return the honest v0.2.x authority boundary.

    Component primitives may have unit tests while remaining non-authoritative
    in the conversational product path.  This baseline intentionally reflects
    causal authority rather than whether a Python class exists.
    """

    return CapabilityRegistry(
        (
            CapabilityRecord(
                "aflow",
                CapabilityState.RESEARCH,
                EvidenceClass.VERIFIED,
                False,
                False,
                "Bounded search primitives exist; no production workflow promotion.",
            ),
            CapabilityRecord(
                "bio_hama",
                CapabilityState.ADVISORY,
                EvidenceClass.VERIFIED,
                False,
                False,
                "Routing telemetry is not trained or answer-authoritative.",
            ),
            CapabilityRecord(
                "cts_deq",
                CapabilityState.CANARY,
                EvidenceClass.MEASURED,
                True,
                False,
                "Causal Gemma hidden state feeds contractive CTS-DEQ; its fixed point "
                "conditions decode through a bounded logits bias while base Gemma stays frozen.",
            ),
            CapabilityRecord(
                "gemma4_e4b",
                CapabilityState.AUTHORITATIVE,
                EvidenceClass.VERIFIED,
                True,
                False,
                "Verified local base Gemma is the current natural-language answer authority.",
            ),
            CapabilityRecord(
                "self_harness",
                CapabilityState.PROPOSAL_ONLY,
                EvidenceClass.VERIFIED,
                False,
                False,
                "Patch proposals cannot promote without an attested isolated runner.",
            ),
            CapabilityRecord(
                "system_1_5",
                CapabilityState.GATED,
                EvidenceClass.VERIFIED,
                False,
                False,
                "No trained and admitted session Fast Weight is active by default.",
            ),
            CapabilityRecord(
                "system_2_5",
                CapabilityState.NIGHT_ONLY,
                EvidenceClass.VERIFIED,
                False,
                True,
                "FP-EWC and spectral updates are restricted to the evolution lifecycle.",
            ),
            CapabilityRecord(
                "system_3",
                CapabilityState.ADVISORY,
                EvidenceClass.VERIFIED,
                False,
                False,
                "Bounded expert state is telemetry and cannot alter current answer tokens.",
            ),
            CapabilityRecord(
                "system_4",
                CapabilityState.ADVISORY,
                EvidenceClass.VERIFIED,
                False,
                False,
                "Tensor swarm state is telemetry and cannot alter current answer tokens.",
            ),
        )
    )


__all__ = [
    "CapabilityRecord",
    "CapabilityRegistry",
    "CapabilityState",
    "EvidenceClass",
    "baseline_capability_registry",
]
