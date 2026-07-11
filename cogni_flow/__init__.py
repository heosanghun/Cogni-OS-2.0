"""Auditable control-plane primitives for Cogni-OS day/night evolution."""

from .rhythm import RhythmController, SystemMode
from .cycle import SelfHarness
from .aflow import AFlowOptimizer, WorkflowSpec
from .daemon import FailureCaptureDaemon
from .harness import SafeHarnessPatcher
from .local_proposer import LocalGemmaPatchProposer, ResolvedPatchTarget
from .logdb import LogDB
from .production import (
    ProductionHarnessConfig,
    ProductionSelfHarness,
    PromotionMode,
    RunnerAttestation,
    build_production_self_harness,
)
from .scheduler import IdleNightScheduler

__all__ = [
    "LogDB",
    "AFlowOptimizer",
    "FailureCaptureDaemon",
    "IdleNightScheduler",
    "LocalGemmaPatchProposer",
    "ProductionHarnessConfig",
    "ProductionSelfHarness",
    "PromotionMode",
    "ResolvedPatchTarget",
    "RhythmController",
    "SafeHarnessPatcher",
    "SelfHarness",
    "SystemMode",
    "RunnerAttestation",
    "WorkflowSpec",
    "build_production_self_harness",
]
