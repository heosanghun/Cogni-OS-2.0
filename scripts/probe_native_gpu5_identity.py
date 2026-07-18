"""Short-lived logical identity probe for the native CogniBoard server.

The long-lived parent must not initialize torch or a CUDA context merely to
prove its logical mapping. This isolated child validates its closed-world
environment, delegates the exact hardware/torch checks to the audited guard,
and exits before product residency begins.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Sequence
import os
from pathlib import Path
import sys
from typing import Any


_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_PHYSICAL_GPU_INDEX = 5
_GPU_QUERY_CONTEXT = "native-host"
_GPU_UUID = "GPU-84d7eeb0-65e0-a5b1-d7db-d09ef59fe03a"
_FIXED_ENVIRONMENT = {
    "PATH": "/usr/local/bin:/usr/bin:/bin",
    "LANG": "C.UTF-8",
    "LC_ALL": "C.UTF-8",
    "HF_HUB_OFFLINE": "1",
    "HF_HUB_DISABLE_TELEMETRY": "1",
    "TRANSFORMERS_OFFLINE": "1",
    "HF_DATASETS_OFFLINE": "1",
    "WANDB_MODE": "offline",
    "TOKENIZERS_PARALLELISM": "false",
    "PYTHONDONTWRITEBYTECODE": "1",
    "PYTHONHASHSEED": "0",
    "PYTHONNOUSERSITE": "1",
    "PYTHONSAFEPATH": "1",
    "COGNI_OS_GPU_UUID": _GPU_UUID,
    "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
    "CUDA_VISIBLE_DEVICES": _GPU_UUID,
    "NVIDIA_VISIBLE_DEVICES": _GPU_UUID,
}
_ENVIRONMENT_KEYS = frozenset(_FIXED_ENVIRONMENT) | {"HOME"}

if not _PROJECT_ROOT.is_dir():
    raise RuntimeError("identity probe repository root is missing")
sys.path.insert(0, str(_PROJECT_ROOT))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--physical-gpu-index", required=True, type=int)
    parser.add_argument("--gpu-query-context", required=True)
    parser.add_argument("--gpu-uuid", required=True)
    return parser


def _require_isolated_runtime() -> None:
    if (
        sys.flags.isolated != 1
        or sys.flags.dont_write_bytecode != 1
        or sys.flags.no_user_site != 1
        or sys.flags.safe_path is not True
    ):
        raise RuntimeError("identity probe requires Python -I -B")
    if frozenset(os.environ) != _ENVIRONMENT_KEYS:
        raise RuntimeError("identity probe requires an exact clean environment")
    if any(os.environ.get(name) != value for name, value in _FIXED_ENVIRONMENT.items()):
        raise RuntimeError("identity probe environment values are invalid")
    home = os.environ.get("HOME", "")
    if (
        not home.startswith("/")
        or len(home) > 4096
        or any(ord(character) < 32 or ord(character) == 127 for character in home)
    ):
        raise RuntimeError("identity probe HOME is invalid")


def main(
    argv: Sequence[str] | None = None,
    *,
    torch_module: Any | None = None,
    identity_validator: Callable[..., Any] | None = None,
) -> int:
    args = _parser().parse_args(argv)
    if (
        args.physical_gpu_index != _PHYSICAL_GPU_INDEX
        or args.gpu_query_context != _GPU_QUERY_CONTEXT
        or args.gpu_uuid != _GPU_UUID
    ):
        raise RuntimeError("identity probe received an invalid scope")
    _require_isolated_runtime()

    if (torch_module is None) != (identity_validator is None):
        raise TypeError("torch_module and identity_validator must be supplied together")
    if torch_module is None:
        # This is the sole torch import in the proof path and occurs only after
        # standard-library admission checks in this short-lived child.
        import torch as loaded_torch

        from scripts.gpu5_boundary_guard import validate_guarded_gpu5_identity

        torch_module = loaded_torch
        identity_validator = validate_guarded_gpu5_identity

    assert identity_validator is not None
    identity = identity_validator(
        physical_gpu_index=args.physical_gpu_index,
        gpu_query_context=args.gpu_query_context,
        torch_module=torch_module,
    )
    if (
        identity.physical_index != _PHYSICAL_GPU_INDEX
        or identity.uuid != _GPU_UUID
        or identity.query_context != _GPU_QUERY_CONTEXT
        or identity.logical_device_count != 1
        or identity.logical_device_index != 0
    ):
        raise RuntimeError("identity probe returned an invalid logical scope")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
