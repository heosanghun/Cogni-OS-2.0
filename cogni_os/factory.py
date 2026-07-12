from __future__ import annotations

import torch
from torch import nn

from cogni_core.adaptation import (
    FastWeightSessionCache,
    FixedPointDomainLifecycle,
    OverlayAcceptanceGate,
)
from cogni_core.experts import BoundedSparseImplicitExperts, ExpertConfig
from cogni_core.expert_lifecycle import ExpertCandidateLifecycle
from cogni_core.fast_weights import (
    FastWeightBackboneWrapper,
    FastWeightProgrammer,
    ResidualBottleneckAdapter,
)
from cogni_core.fast_weight_safety import load_verified_fast_weight_checkpoint
from cogni_core.meta_router import BioHAMAMetaRouter, MetaRouterConfig
from cogni_core.resources import VRAMGuard
from cogni_core.routing import ContrastiveSessionRouter
from cogni_core.search import BoundedPUCTSearch, PUCTConfig
from cogni_core.swarm import SwarmConfig, TensorSwarm
from cogni_core.swarm_sessions import SwarmSessionStateCache
from cogni_flow.rhythm import RhythmController

from .config import CogniConfig
from .runtime import GenesisRuntime


def _backbone_placement(
    backbone: nn.Module,
    requested_device: str,
) -> tuple[torch.device, torch.dtype]:
    """Resolve one materialized device and floating dtype for the runtime.

    A configured CUDA runtime may fall back to CPU only when CUDA is genuinely
    unavailable and the supplied backbone is already on CPU.  The factory
    never moves a caller-owned backbone implicitly.
    """

    tensors = tuple(backbone.parameters()) + tuple(backbone.buffers())
    devices = {tensor.device for tensor in tensors}
    if any(device.type == "meta" for device in devices):
        raise RuntimeError("backbone tensors must be materialized before runtime build")
    if len(devices) > 1:
        rendered = ", ".join(sorted(str(device) for device in devices))
        raise RuntimeError(f"backbone spans multiple devices: {rendered}")

    cuda_available = torch.cuda.is_available()
    effective_type = "cuda" if requested_device == "cuda" and cuda_available else "cpu"
    if devices:
        backbone_device = next(iter(devices))
        if backbone_device.type != effective_type:
            fallback = (
                " (CUDA unavailable; CPU fallback required)"
                if not cuda_available
                else ""
            )
            raise RuntimeError(
                "backbone device does not match configured runtime device: "
                f"backbone={backbone_device}, runtime={effective_type}{fallback}"
            )
    elif effective_type == "cuda":
        backbone_device = torch.device("cuda", torch.cuda.current_device())
    else:
        backbone_device = torch.device("cpu")

    floating = next(
        (tensor for tensor in tensors if tensor.is_floating_point()),
        None,
    )
    backbone_dtype = torch.float32 if floating is None else floating.dtype
    return backbone_device, backbone_dtype


def _target_storage_bytes(module: nn.Module, dtype: torch.dtype) -> int:
    target_element_size = torch.empty((), dtype=dtype).element_size()
    return sum(
        tensor.numel()
        * (target_element_size if tensor.is_floating_point() else tensor.element_size())
        for tensor in (*module.parameters(), *module.buffers())
    )


def _assert_module_placement(
    module: nn.Module,
    *,
    device: torch.device,
    dtype: torch.dtype,
    name: str,
) -> None:
    for tensor in (*module.parameters(), *module.buffers()):
        if tensor.device != device:
            raise RuntimeError(
                f"{name} placement mismatch: expected {device}, got {tensor.device}"
            )
        if tensor.is_floating_point() and tensor.dtype != dtype:
            raise RuntimeError(
                f"{name} dtype mismatch: expected {dtype}, got {tensor.dtype}"
            )


def build_genesis_runtime(
    backbone: nn.Module,
    config: CogniConfig,
    *,
    input_dim: int,
    state_dim: int,
    fast_weight_checkpoint: str | None = None,
    fast_weight_checkpoint_sha256: str | None = None,
    _allow_test_fast_weight_fixture: bool = False,
) -> GenesisRuntime:
    """Construct the bounded local runtime from validated TOML configuration."""
    hardware = config.section("hardware")
    requested_device = str(hardware["device"])
    backbone_device, backbone_dtype = _backbone_placement(backbone, requested_device)
    guard = VRAMGuard(float(hardware["vram_limit_gib"]), backbone_device)
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
    if (fast_weight_checkpoint is None) != (fast_weight_checkpoint_sha256 is None):
        raise ValueError(
            "Fast Weight checkpoint path and SHA-256 must be supplied together"
        )
    verified_fast_weight = None
    if fast_weight_checkpoint is not None:
        assert fast_weight_checkpoint_sha256 is not None
        verified_fast_weight = load_verified_fast_weight_checkpoint(
            programmer,
            adapter,
            fast_weight_checkpoint,
            expected_sha256=fast_weight_checkpoint_sha256,
            allow_test_fixture=_allow_test_fast_weight_fixture,
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
    session_router = ContrastiveSessionRouter(
        max_sessions=int(fast["session_capacity"])
    )
    sessions = FastWeightSessionCache(
        wrapped_backbone,
        gate=OverlayAcceptanceGate(
            operator_norm_budget=overlay_budget,
            composed_operator_norm_budget=composed_budget,
        ),
        max_sessions=int(fast["session_capacity"]),
        on_sessions_removed=session_router.discard_many,
        feature_enabled=verified_fast_weight is not None,
        trusted_programmer_sha256=(
            None
            if verified_fast_weight is None
            else verified_fast_weight.programmer_evidence.checkpoint_sha256
        ),
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
            global_margin=float(swarm_cfg["global_margin"]),
            operating_margin=float(swarm_cfg["operating_margin"]),
            cold_steps=int(swarm_cfg["cold_steps"]),
            warm_steps=int(swarm_cfg["warm_steps"]),
            residual_tolerance=float(swarm_cfg["residual_tolerance"]),
            certificate_power_iterations=int(swarm_cfg["certificate_power_iterations"]),
        )
    )
    swarm_sessions = SwarmSessionStateCache(
        max_sessions=int(swarm_cfg["session_capacity"]),
        ttl_seconds=float(swarm_cfg["session_ttl_seconds"]),
        max_state_bytes=int(swarm_cfg["session_max_state_mib"]) * 1024**2,
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
    # Keep answer-bearing adapter/programmer tensors aligned with the Gemma
    # backbone, but retain the safety/control plane in FP32.  System 3/4 use
    # spectral norms and pseudo-inverses that PyTorch deliberately rejects for
    # BF16 on CUDA; silently casting those certificates down would invalidate
    # them.  CoreTurnPipeline owns the detached dtype boundary into these
    # modules.
    answer_modules = (adapter, programmer)
    control_modules = (swarm, experts, meta_router)
    auxiliary_bytes = sum(
        _target_storage_bytes(module, backbone_dtype) for module in answer_modules
    ) + sum(_target_storage_bytes(module, torch.float32) for module in control_modules)
    with guard.enforce(auxiliary_bytes):
        for module in answer_modules:
            module.to(device=backbone_device, dtype=backbone_dtype)
        for module in control_modules:
            module.to(device=backbone_device, dtype=torch.float32)
        # Casting a certified FP32 matrix to BF16 can move its exact spectral
        # norm across the strict budget.  Re-project and post-certify on the
        # actual answer-plane dtype before the first tensor operation.
        adapter.c_fire_()
    for name, module in (
        ("adapter", adapter),
        ("programmer", programmer),
    ):
        _assert_module_placement(
            module,
            device=backbone_device,
            dtype=backbone_dtype,
            name=name,
        )
    for name, module in (
        ("swarm", swarm),
        ("experts", experts),
        ("meta_router", meta_router),
    ):
        _assert_module_placement(
            module,
            device=backbone_device,
            dtype=torch.float32,
            name=name,
        )
    experts.assert_phase8_profile()
    expert_lifecycle = ExpertCandidateLifecycle(experts)
    return GenesisRuntime(
        wrapped_backbone,
        search,
        rhythm=RhythmController(),
        vram_guard=guard,
        sessions=sessions,
        domains=domains,
        swarm=swarm,
        swarm_sessions=swarm_sessions,
        session_router=session_router,
        experts=experts,
        expert_lifecycle=expert_lifecycle,
        meta_router=meta_router,
        fast_weight_programmer=programmer,
        verified_fast_weight=verified_fast_weight,
        fast_weight_target=FastWeightBackboneWrapper.TARGET_MODULE,
    )
