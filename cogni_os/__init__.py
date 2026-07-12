"""Cogni-OS integration layer."""

from .capabilities import (
    CapabilityRecord,
    CapabilityRegistry,
    CapabilityState,
    EvidenceClass,
    baseline_capability_registry,
)
from .config import CogniConfig, load_config
from .evidence import (
    AppendOnlyEvidenceJournal,
    ClaimAssessment,
    ClaimRecord,
    EvidenceError,
    EvidenceRecordV1,
    EvidenceScopeV1,
    FactBookSnapshotStore,
    FactBookSnapshotV1,
    build_factbook_claims,
)
from .factbook import (
    RuntimeFactBook,
    build_runtime_factbook,
    build_runtime_factbook_from_verified,
)
from .version import __version__

__all__ = [
    "CapabilityRecord",
    "CapabilityRegistry",
    "CapabilityState",
    "CogniConfig",
    "AppendOnlyEvidenceJournal",
    "ClaimAssessment",
    "ClaimRecord",
    "EvidenceError",
    "EvidenceClass",
    "EvidenceRecordV1",
    "EvidenceScopeV1",
    "FactBookSnapshotStore",
    "FactBookSnapshotV1",
    "RuntimeFactBook",
    "baseline_capability_registry",
    "build_runtime_factbook",
    "build_runtime_factbook_from_verified",
    "build_factbook_claims",
    "load_config",
    "__version__",
]
