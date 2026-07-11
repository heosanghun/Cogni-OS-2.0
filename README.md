# Cogni-OS 2.0 — Genesis

This repository starts from three executable gates before integrating a large language model:

1. implicit DEQ forward and IFT backward must match a long explicit unroll;
2. CUDA active memory must remain flat as CTS depth grows at fixed width;
3. non-contractive transitions must hard-stop or enter an explicit damped fallback.

The gates now pass, and the repository includes a resident local Gemma chat
worker, fixed-arena CTS, Fast Weight, FP-EWC, bounded sparse experts,
System4-style tensor swarms, BIO-HAMA routing, day/night exclusion, bounded
workspace tools, AFlow, and a fail-closed Self-Harness control plane.

## Run

```powershell
python -m unittest discover -s tests -v
```

## Windows double-click demo

Double-click `CogniBoard.exe` from the repository root for the console-free
experience, or use `Run-CogniOS-Demo.cmd` as the transparent diagnostic
launcher. The CMD launcher performs an explicit Python/CUDA dependency
preflight; the native launcher locates the same local runtime without showing a
console. The server verifies the local model artifact manifest before opening
**CogniBoard**, a loopback-only graphical
mission-control interface. Its six views connect a real local AI workspace,
the customer problem, measured evidence, live Gemma 4 + CTS validation,
architecture, business model, and execution roadmap. Selecting
`실제 통합 검증 시작` launches the existing
Depth-100 worker as the sole CUDA owner; the UI never reimplements or simulates
the validation path.

The default `AI 워크스페이스` view supports bounded multi-turn conversation and
token streaming from the verified local Gemma artifact. Every chat turn runs
the advisory Cogni-Core route in this order:

`BIO-HAMA → Gemma feature backbone + DEQ/CTS → System 4 → System 3 → Gemma decode`

The final answer is produced only by deterministic base Gemma decoding with
`use_cache=False`; untrained Fast Weight overlays remain gated, and FP-EWC is
reserved for the evolution path. Task mode exposes only `/help`, `/list`,
`/read`, `/search`, `/status`, `/test`, and `/save`. It never exposes arbitrary
shell, network, or unrestricted source writes.

Self-Harness records bounded runtime failures and can generate local patch
proposals during an exclusive evolution cycle. Source promotion is disabled by
default. It becomes possible only when an operator supplies an explicitly
trusted, kernel-isolated runner attestation covering network isolation, host
filesystem isolation, an ephemeral workspace, and the exact regression and
health-check command digests. Promotion then uses a digest-verified backup
journal, atomic replacement, post-promotion health checking, and rollback.

`Run-CogniOS-CLI.cmd` retains the console diagnostic flow for operators. The
graphical server binds only to `127.0.0.1`, serves an exact local asset allowlist,
uses a per-session authentication token, and applies a restrictive CSP. It has
no CDN, analytics SDK, external API, or remote font dependency.

The default local model path is `C:\Project\cognios\gemma4-e4b`. Set
`COGNI_OS_MODEL_DIR` before launching to select another verified local path.
All Hugging Face network access and telemetry are disabled by the launcher.

The CUDA gate is skipped only when CUDA is unavailable. It reports a real failure when a CUDA
device exists but peak active allocation grows by more than 8 MiB between depth 8 and depth 64.

## Scope

This is a bounded research runtime, not a claim of AGI or O(1) total system
memory. Solver history and CTS working tensors are fixed-capacity; model weights,
expert banks, logs, and external data remain separately budgeted resources. On
the attached RTX 5090 Laptop GPU, the verified local Gemma artifact plus the
integrated depth-100/301-node runtime reached a 14.8560 GiB inference peak. The
RTX 4090 target still requires the same hardware gate.

Reference implementations reviewed during development are kept under `work/upstream/` and are
not imported at runtime. In particular: `Cognitive-Tree-Search`, `System1.5_260515`, `System2.5`,
`System3`, `System4`, `System5`, and `BIO-HAMA_MAIN`.

See [architecture](docs/ARCHITECTURE.md), [security model](docs/SECURITY.md), and
[validation gates](docs/VALIDATION.md) before connecting a large local backbone or enabling
autonomous patch promotion.

한국어 실행·대화·작업·Self-Harness 안내는
[CogniBoard AI 워크스페이스 사용 안내](docs/AGENT_WORKSPACE_KO.md)를 참고하십시오.

The verified local Gemma 4 procedure and measured hardware results are documented in
[Gemma 4 integration](docs/GEMMA4.md).

The customer positioning, evidence taxonomy, three-minute judging script, and
30/60/90-day commercialization plan are documented in the
[CogniBoard business demo plan](docs/COGNIBOARD_BUSINESS_DEMO_PLAN_KO.md).
