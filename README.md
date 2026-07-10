# Cogni-OS 2.0 — Genesis

This repository starts from three executable gates before integrating a large language model:

1. implicit DEQ forward and IFT backward must match a long explicit unroll;
2. CUDA active memory must remain flat as CTS depth grows at fixed width;
3. non-contractive transitions must hard-stop or enter an explicit damped fallback.

The gates now pass, and the repository includes fixed-arena CTS, Fast Weight,
FP-EWC, bounded sparse experts, System4-style tensor swarms, BIO-HAMA routing,
day/night exclusion, AFlow, and a fail-closed Self-Harness control plane.

## Run

```powershell
python -m unittest discover -s tests -v
```

The CUDA gate is skipped only when CUDA is unavailable. It reports a real failure when a CUDA
device exists but peak active allocation grows by more than 8 MiB between depth 8 and depth 64.

## Scope

This is a bounded research runtime, not a claim of AGI or O(1) total system
memory. Solver history and CTS working tensors are fixed-capacity; model weights,
expert banks, logs, and external data remain separately budgeted resources. On
the attached RTX 5090 Laptop GPU, the verified local Gemma artifact plus the
integrated depth-100/301-node runtime reached a 14.8604 GiB inference peak. The
RTX 4090 target still requires the same hardware gate.

Reference implementations reviewed during development are kept under `work/upstream/` and are
not imported at runtime. In particular: `Cognitive-Tree-Search`, `System1.5_260515`, `System2.5`,
`System3`, `System4`, `System5`, and `BIO-HAMA_MAIN`.

See [architecture](docs/ARCHITECTURE.md), [security model](docs/SECURITY.md), and
[validation gates](docs/VALIDATION.md) before connecting a large local backbone or enabling
autonomous patch promotion.

The verified local Gemma 4 procedure and measured hardware results are documented in
[Gemma 4 integration](docs/GEMMA4.md).
