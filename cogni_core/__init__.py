"""Memory-bounded implicit reasoning primitives for Cogni-OS."""

from .deq import ContractivityError, DEQConfig, EquilibriumLayer
from .backbone import LocalGemmaFeatureBackbone
from .c_fire import CFireCertificate, CFireError, c_fire_scaled_polar_
from .experts import BoundedSparseImplicitExperts, ExpertConfig
from .expert_lifecycle import (
    ExpertCandidateLifecycle,
    ExpertLifecycleError,
    ExternalVerifierAttestation,
    RoutedFisherSnapshot,
    SparseRoutedFPEWC,
    estimate_routed_fisher,
)
from .meta_router import BioHAMAMetaRouter, CognitiveState, MetaRouterConfig
from .resources import ResourceBudgetExceeded, VRAMGuard
from .search import (
    BoundedPUCTSearch,
    BoundedPUCTSearchV2,
    CertifiedBroydenTransitionV2,
    CertifiedPUCTConfigV2,
    PUCTConfig,
    SearchControlsV2,
    SearchRequestV2,
    SemanticAncestorRetriever,
)
from .cts_policy import LearnedCTSController, load_default_bounded_cts_controller
from .swarm import SwarmConfig, TensorSwarm

__all__ = [
    "BoundedPUCTSearch",
    "BoundedPUCTSearchV2",
    "BoundedSparseImplicitExperts",
    "BioHAMAMetaRouter",
    "CFireCertificate",
    "CFireError",
    "CognitiveState",
    "ContractivityError",
    "CertifiedBroydenTransitionV2",
    "CertifiedPUCTConfigV2",
    "DEQConfig",
    "EquilibriumLayer",
    "ExpertConfig",
    "ExpertCandidateLifecycle",
    "ExpertLifecycleError",
    "ExternalVerifierAttestation",
    "MetaRouterConfig",
    "LocalGemmaFeatureBackbone",
    "LearnedCTSController",
    "PUCTConfig",
    "ResourceBudgetExceeded",
    "RoutedFisherSnapshot",
    "SemanticAncestorRetriever",
    "SearchControlsV2",
    "SearchRequestV2",
    "SwarmConfig",
    "SparseRoutedFPEWC",
    "TensorSwarm",
    "VRAMGuard",
    "c_fire_scaled_polar_",
    "estimate_routed_fisher",
    "load_default_bounded_cts_controller",
]
