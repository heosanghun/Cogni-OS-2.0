from __future__ import annotations

from dataclasses import dataclass
from importlib import resources
from pathlib import Path
import tomllib


MAX_VRAM_GIB = 16.7


@dataclass(frozen=True)
class CogniConfig:
    raw: dict

    def section(self, name: str) -> dict:
        value = self.raw.get(name)
        if not isinstance(value, dict):
            raise KeyError(f"missing config section: {name}")
        return dict(value)

    @property
    def offline(self) -> bool:
        return bool(self.section("project").get("offline", True))


def load_config(path: str | Path | None = None) -> CogniConfig:
    """Load an explicit config or the immutable package default.

    The default is a package resource, so installed CLI commands never depend
    on the caller's current working directory.
    """

    source = (
        resources.files("cogni_os").joinpath("default.toml")
        if path is None
        else Path(path).expanduser().resolve(strict=True)
    )
    with source.open("rb") as stream:
        data = tomllib.load(stream)
    required = {
        "project",
        "hardware",
        "deq",
        "cts",
        "fast_weights",
        "fp_ewc",
        "swarm",
        "experts",
        "meta_router",
        "aflow",
        "flow",
    }
    missing = sorted(required - data.keys())
    if missing:
        raise ValueError(f"configuration missing sections: {', '.join(missing)}")
    if not data["project"].get("offline", False):
        raise ValueError("Cogni-OS requires project.offline=true")
    try:
        vram_limit = float(data["hardware"]["vram_limit_gib"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("hardware.vram_limit_gib must be numeric") from exc
    if not 0.0 < vram_limit <= MAX_VRAM_GIB:
        raise ValueError(f"hardware.vram_limit_gib must be in (0, {MAX_VRAM_GIB}]")
    if data["hardware"].get("device") not in {"cpu", "cuda"}:
        raise ValueError("hardware.device must be 'cpu' or 'cuda'")

    deq = data["deq"]
    if float(deq["tolerance"]) <= 0.0:
        raise ValueError("deq.tolerance must be positive")
    if int(deq["max_iter"]) < 1 or int(deq["history"]) < 1:
        raise ValueError("deq.max_iter and deq.history must be positive")
    if not 0.0 < float(deq["spectral_margin"]) < 1.0:
        raise ValueError("deq.spectral_margin must be in (0, 1)")
    if not 0.0 < float(deq["fallback_damping"]) <= 1.0:
        raise ValueError("deq.fallback_damping must be in (0, 1]")
    if int(deq["fallback_steps"]) < 0:
        raise ValueError("deq.fallback_steps cannot be negative")

    cts = data["cts"]
    width = int(cts["width"])
    max_nodes = int(cts["max_nodes"])
    latent_capacity = int(cts["latent_capacity"])
    if width < 1 or int(cts["max_depth"]) < 1:
        raise ValueError("cts.width and cts.max_depth must be positive")
    if max_nodes < 1 + width:
        raise ValueError("cts.max_nodes must fit root plus one full expansion")
    if not 1 <= latent_capacity <= max_nodes:
        raise ValueError("cts.latent_capacity must be in [1, max_nodes]")

    fast = data["fast_weights"]
    rank = int(fast["rank"])
    bottleneck = int(fast["bottleneck_dim"])
    internal = int(fast["internal_dim"])
    overlay_budget = float(fast["max_operator_norm"])
    fp_margin = float(data["fp_ewc"]["spectral_margin"])
    if not 0.0 < fp_margin < 1.0:
        raise ValueError("fp_ewc.spectral_margin must be in (0, 1)")
    if not 1 <= rank <= bottleneck <= 512:
        raise ValueError("fast_weights requires 1 <= rank <= bottleneck_dim <= 512")
    if not 1 <= internal <= 512:
        raise ValueError("fast_weights.internal_dim must be in [1, 512]")
    if not 0.0 < overlay_budget < fp_margin:
        raise ValueError(
            "fast_weights.max_operator_norm must be below fp_ewc.spectral_margin"
        )
    if int(fast["session_capacity"]) < 1:
        raise ValueError("fast_weights.session_capacity must be positive")

    if data["flow"].get("require_kernel_sandbox_for_production") is not True:
        raise ValueError("flow kernel sandbox requirement cannot be disabled")
    return CogniConfig(data)
