from __future__ import annotations

import torch
from torch import nn

from cogni_core.adaptation import (
    FastWeightSessionCache,
    FixedPointDomainLifecycle,
    OverlayAcceptanceGate,
)
from cogni_core.experts import BoundedSparseImplicitExperts, ExpertConfig
from cogni_core.fast_weights import (
    FastWeightBackboneWrapper,
    FastWeightProgrammer,
    ResidualBottleneckAdapter,
)
from cogni_core.meta_router import BioHAMAMetaRouter, MetaRouterConfig
from cogni_core.resources import VRAMGuard, module_storage_bytes
from cogni_core.routing import ContrastiveSessionRouter
from cogni_core.search import BoundedPUCTSearch, PUCTConfig
from cogni_core.swarm import SwarmConfig, TensorSwarm
from cogni_flow.rhythm import RhythmController

from .config import CogniConfig
from .runtime import GenesisRuntime


def build_genesis_runtime(
    backbone: nn.Module,
    config: CogniConfig,
    *,
    input_dim: int,
    state_dim: int,
) -> GenesisRuntime:
    """Construct the bounded local runtime from validated TOML configuration."""
    hardware = config.section("hardware")
    requested_device = str(hardware["device"])
    device = (
        requested_device
        if requested_device != "cuda" or torch.cuda.is_available()
        else "cpu"
    )
    guard = VRAMGuard(float(hardware["vram_limit_gib"]), device)
    # Reject a caller that already loaded the backbone beyond the envelope
    # before constructing any auxiliary CUDA module.
    guard.assert_within_limit()
    cts = config.section("cts")
    fast = config.section("fast_weights")
    fp = config.section("fp_ewc")
    swarm_cfg = config.section("swarm")
    expert_cfg = config.section("experts")
    meta_cfg = config.section("meta_router")

    latent_dim = int(input_dim)
    requested_bottleneck = int(fast.get("bottleneck_dim", 64))
    if requested_bottleneck <= 0:
        raise ValueError("fast_weights.bottleneck_dim must be positive")
    bottleneck_dim = min(requested_bottleneck, latent_dim)
    internal_dim = int(fast.get("internal_dim", 128))
    rank = min(int(fast["rank"]), bottleneck_dim)
    overlay_budget = float(fast["max_operator_norm"])
    composed_budget = float(fp["spectral_margin"])
    # Leave a numerical gap so the triangle-bound admission test certifies
    # ||core + overlay||_2 strictly below the C-FIRE margin.
    core_budget = composed_budget - overlay_budget - 1.0e-3
    if core_budget <= 0.0:
        raise ValueError(
            "fast-weight operator budget must be below fp_ewc.spectral_margin"
        )
    adapter = ResidualBottleneckAdapter(
        latent_dim,
        bottleneck_dim,
        core_operator_norm_budget=core_budget,
        spectral_margin=composed_budget,
    )
    programmer = FastWeightProgrammer(
        source_dim=latent_dim,
        target_dim=bottleneck_dim,
        internal_dim=internal_dim,
        rank=rank,
        max_operator_norm=overlay_budget,
    )

    try:
        first_parameter = next(backbone.parameters())
    except StopIteration:
        backbone_device = torch.device(device)
        backbone_dtype = torch.float32
    else:
        backbone_device = first_parameter.device
        backbone_dtype = (
            first_parameter.dtype
            if first_parameter.is_floating_point()
            else torch.float32
        )
    wrapped_backbone = FastWeightBackboneWrapper(backbone, adapter, programmer)

    search = BoundedPUCTSearch(
        PUCTConfig(
            width=int(cts["width"]),
            max_depth=int(cts["max_depth"]),
            max_nodes=int(cts["max_nodes"]),
            simulations=min(
                int(cts["max_nodes"]), int(cts["max_depth"]) * int(cts["width"])
            ),
            ancestor_capacity=int(cts["latent_capacity"]),
            ancestor_k=min(3, int(cts["latent_capacity"])),
        )
    )
    sessions = FastWeightSessionCache(
        wrapped_backbone,
        gate=OverlayAcceptanceGate(
            operator_norm_budget=overlay_budget,
            composed_operator_norm_budget=composed_budget,
        ),
        max_sessions=int(fast["session_capacity"]),
    )
    domains = FixedPointDomainLifecycle(
        strength=float(fp["strength"]),
        spectral_margin=float(fp["spectral_margin"]),
        max_domains=16,
    )
    swarm = TensorSwarm(
        SwarmConfig(
            input_dim=input_dim,
            state_dim=int(swarm_cfg["state_dim"]),
            agents=int(swarm_cfg["agents"]),
            sensory_agents=int(swarm_cfg["sensory_agents"]),
            constraint_agents=int(swarm_cfg["constraint_agents"]),
            local_margin=float(swarm_cfg["local_margin"]),
            coupling_scale=float(swarm_cfg["coupling_scale"]),
        )
    )
    experts = BoundedSparseImplicitExperts(
        ExpertConfig(
            input_dim=input_dim,
            state_dim=state_dim,
            max_experts=int(expert_cfg["max_experts"]),
            initial_experts=int(expert_cfg["initial_experts"]),
            top_k=int(expert_cfg["top_k"]),
            novelty_threshold=float(expert_cfg["novelty_threshold"]),
            spectral_margin=float(expert_cfg["spectral_margin"]),
            max_parameter_bytes=int(float(expert_cfg["max_parameter_gib"]) * 1024**3),
            max_vram_bytes=int(float(hardware["vram_limit_gib"]) * 1024**3),
        )
    )
    meta_router = BioHAMAMetaRouter(
        MetaRouterConfig(
            num_modules=int(meta_cfg["num_modules"]),
            hidden_dim=int(meta_cfg["hidden_dim"]),
            strategic_top_k=int(meta_cfg["strategic_top_k"]),
            tactical_top_k=int(meta_cfg["tactical_top_k"]),
            reactive_top_k=int(meta_cfg["reactive_top_k"]),
        )
    )
    auxiliary_bytes = sum(
        module_storage_bytes(module)
        for module in (adapter, programmer, swarm, experts, meta_router)
    )
    with guard.enforce(auxiliary_bytes):
        adapter.to(device=backbone_device, dtype=backbone_dtype)
        programmer.to(device=backbone_device, dtype=backbone_dtype)
        swarm = swarm.to(device)
        experts = experts.to(device)
        meta_router = meta_router.to(device)
    return GenesisRuntime(
        wrapped_backbone,
        search,
        rhythm=RhythmController(),
        vram_guard=guard,
        sessions=sessions,
        domains=domains,
        swarm=swarm,
        session_router=ContrastiveSessionRouter(
            max_sessions=int(fast["session_capacity"])
        ),
        experts=experts,
        meta_router=meta_router,
        fast_weight_programmer=programmer,
        fast_weight_target=FastWeightBackboneWrapper.TARGET_MODULE,
    )
