from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys

import torch

from cogni_core.cts import CognitiveTreeSearch
from cogni_core.backbone import verify_local_gemma_path
from cogni_core.deq import DEQConfig, EquilibriumLayer
from .artifacts import verify_artifact_manifest
from .config import load_config


def doctor(config_path: str | None) -> int:
    config = load_config(config_path)
    print(f"project={config.section('project')['name']}")
    print(f"offline={config.offline}")
    print(f"torch={torch.__version__}")
    print(f"cuda_available={torch.cuda.is_available()}")
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        print(f"cuda_device={props.name}")
        print(f"device_memory_gib={props.total_memory / 1024**3:.2f}")
    else:
        print("cuda_device=none (CUDA gates will be skipped)")
    return 0


def validate() -> int:
    project_root = Path(__file__).resolve().parents[1]
    tests_dir = project_root / "tests"
    if not tests_dir.is_dir():
        # Wheels intentionally omit the source test tree. Keep the installed
        # command useful with a small deterministic, network-free runtime gate.
        config = load_config()
        deq_cfg = config.section("deq")
        layer = EquilibriumLayer(
            4,
            4,
            DEQConfig(
                tolerance=float(deq_cfg["tolerance"]),
                max_iter=int(deq_cfg["max_iter"]),
                history=int(deq_cfg["history"]),
                spectral_margin=float(deq_cfg["spectral_margin"]),
                fail_on_noncontractive=True,
            ),
        ).eval()
        with torch.inference_mode():
            output = layer(torch.zeros(1, 4))
        if not torch.isfinite(output).all() or not layer.last_info:
            print("installed self-check failed", file=sys.stderr)
            return 1
        print("installed_self_check=passed")
        return 0
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "unittest",
            "discover",
            "-s",
            str(tests_dir),
            "-v",
        ],
        check=False,
        cwd=project_root,
    )
    return completed.returncode


def profile_depth(config_path: str | None) -> int:
    config = load_config(config_path)
    if not torch.cuda.is_available():
        print("CUDA unavailable; depth profile not executed")
        return 2
    deq_cfg = config.section("deq")
    cts_cfg = config.section("cts")
    device = torch.device("cuda")
    layer = (
        EquilibriumLayer(
            32,
            32,
            DEQConfig(
                tolerance=float(deq_cfg["tolerance"]),
                max_iter=int(deq_cfg["max_iter"]),
                history=int(deq_cfg["history"]),
                spectral_margin=float(deq_cfg["spectral_margin"]),
            ),
        )
        .to(device)
        .eval()
    )
    search = CognitiveTreeSearch(layer, width=int(cts_cfg["width"])).to(device)
    root = torch.zeros(1, 32, device=device)

    def action(state, index):
        return state + (index + 1) * 0.001

    def critic(state):
        return -state.square().mean()

    search.search(root, action, critic, 2)
    for depth in (8, 64, int(cts_cfg["max_depth"])):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        search.search(root, action, critic, depth)
        torch.cuda.synchronize()
        peak = torch.cuda.max_memory_allocated() / 1024**2
        print(f"depth={depth} peak_active_mib={peak:.4f}")
    return 0


def verify_model(model_path: str, manifest_path: str) -> int:
    root = verify_local_gemma_path(model_path)
    result = verify_artifact_manifest(root, manifest_path)
    print(f"model_root={result.root}")
    print(f"verified_files={len(result.files)}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="cogni-os")
    parser.add_argument(
        "--config",
        default=None,
        help="explicit TOML path (defaults to the packaged offline config)",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("doctor")
    sub.add_parser("validate")
    sub.add_parser("profile-depth")
    verify_parser = sub.add_parser("verify-model")
    verify_parser.add_argument("--model", required=True)
    verify_parser.add_argument("--manifest", required=True)
    args = parser.parse_args(argv)
    if args.command == "doctor":
        return doctor(args.config)
    if args.command == "validate":
        return validate()
    if args.command == "profile-depth":
        return profile_depth(args.config)
    if args.command == "verify-model":
        return verify_model(args.model, args.manifest)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
