"""Memory-bounded implicit reasoning primitives for Cogni-OS."""

from .deq import ContractivityError, DEQConfig, EquilibriumLayer
from .backbone import LocalGemmaFeatureBackbone
from .experts import BoundedSparseImplicitExperts, ExpertConfig
from .meta_router import BioHAMAMetaRouter, CognitiveState, MetaRouterConfig
from .resources import ResourceBudgetExceeded, VRAMGuard
from .search import BoundedPUCTSearch, PUCTConfig, SemanticAncestorRetriever
from .swarm import SwarmConfig, TensorSwarm

__all__ = [
    "BoundedPUCTSearch",
    "BoundedSparseImplicitExperts",
    "BioHAMAMetaRouter",
    "CognitiveState",
    "ContractivityError",
    "DEQConfig",
    "EquilibriumLayer",
    "ExpertConfig",
    "MetaRouterConfig",
    "LocalGemmaFeatureBackbone",
    "PUCTConfig",
    "ResourceBudgetExceeded",
    "SemanticAncestorRetriever",
    "SwarmConfig",
    "TensorSwarm",
    "VRAMGuard",
]
