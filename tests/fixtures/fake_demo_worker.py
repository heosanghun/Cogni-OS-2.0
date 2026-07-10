from __future__ import annotations

import json
import sys
from time import sleep


SENTINEL = "@@COGNI_EVENT@@"
PHASES = (
    ("verifying", 5),
    ("loading_model", 15),
    ("building_runtime", 65),
    ("running_inference", 75),
    ("postcheck", 95),
)


def emit(payload: dict) -> None:
    print(SENTINEL + json.dumps(payload, separators=(",", ":")), flush=True)


def metrics() -> dict:
    return {
        "verified_files": 6,
        "model_class": "FakeGemma4",
        "hidden_size": 64,
        "load_seconds": 0.01,
        "inference_seconds": 0.02,
        "requested_depth": 100,
        "reached_depth": 100,
        "nodes_used": 301,
        "node_capacity": 301,
        "search_allocated_bytes": 14994009,
        "transition_converged": True,
        "transition_residual": 0.00390625,
        "transition_used_fallback": False,
        "peak_vram_gib": 14.856,
        "vram_limit_gib": 16.7,
        "finite": True,
        "device": "Fake CUDA Device",
    }


def main() -> int:
    mode = sys.argv[1]
    if mode == "fail":
        print("bounded worker failure", file=sys.stderr, flush=True)
        return 7
    if mode == "malformed":
        print(SENTINEL + "{", flush=True)
        return 0
    if mode == "hang":
        emit({"v": 1, "seq": 1, "kind": "phase", "stage": "verifying", "progress": 5})
        sleep(60)
        return 0
    if mode == "stderr_flood":
        print("x" * 20000, file=sys.stderr, flush=True)
    if mode == "stdout_flood":
        print("x" * 20000, flush=True)

    sequence = 0
    for stage, progress in PHASES:
        sequence += 1
        emit(
            {
                "v": 1,
                "seq": sequence,
                "kind": "phase",
                "stage": stage,
                "progress": progress,
            }
        )
    sequence += 1
    result = metrics()
    if mode == "over_vram":
        result["peak_vram_gib"] = 17.0
    event = {
        "v": 1,
        "seq": sequence,
        "kind": "result",
        "stage": "complete",
        "progress": 100,
        "metrics": result,
    }
    emit(event)
    if mode == "duplicate_result":
        event["seq"] += 1
        emit(event)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
