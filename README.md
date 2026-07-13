# Cogni-OS 2.0 Genesis — v0.3.2

Cogni-OS 2.0 is an offline, bounded research runtime for a verified local
dense Gemma 4 E4B-it artifact. Version 0.3.2 connects conversation integrity,
causal DEQ/CTS conditioning, typed local tasks, bounded research workflows,
and a **proposal-only** Self-Harness behind explicit capability and evidence
states.

This release is not an AGI claim, an RTX 4090 certification, or evidence that
every research module improves model quality. A Python implementation or a
passing component test never upgrades a capability by itself.

## Product authority at a glance

| Capability | v0.3.2 state | May affect the answer? | What is still required |
|---|---|---:|---|
| verified local Gemma 4 E4B-it | `authoritative` | yes | exact pinned instruction-tuned artifact and seven-file manifest |
| bounded conversation fast path | `product_ux` | yes, narrow social turns only | must never intercept general knowledge/code/format requests |
| causal CTS/DEQ bridge | `canary` | yes, bounded logits bias | trained adapter/Wproj and independent held-out evidence for promotion |
| BIO-HAMA | `advisory` | no | calibrated routing-quality evidence |
| System 1.5 Fast Weight | `gated` | no by default | trained checkpoint, AQ/OOD and held-out admission evidence |
| System 2.5 FP-EWC/C-FIRE | `night_only` | no during inference | multi-domain external effectiveness evidence |
| System 3 experts | `advisory` | no | calibrated/trained expert checkpoint and independent verifier |
| System 4 swarm/PCAS | `advisory` | no | production PCAS calibration and answer-quality evidence |
| AFlow/ADAS | `research` | no | attested real evaluator and held-out evidence |
| Self-Harness | `proposal_only` | no runtime mutation | Phase 12 attested sandbox and safe-promotion evidence |

The runtime Fact-book is the authority for the live process. Evidence records
are content-addressed and bound to the exact model, code, configuration, and
device scope; a scope change invalidates prior measured claims. The Fact-book
snapshot store uses an external last-known-good pointer and rejects stale or
malformed evidence.

## Phase 1–11 status

| Phase | Implemented software path | Evidence boundary |
|---:|---|---|
| 1 | Fact-book identity grounding, narrow social fast path, repetition/completion/relevance guards, bounded retry | 10-turn casual and 20-turn local-Gemma gates must both pass on the frozen release commit |
| 2 | lifecycle/GPU lease authority, IPC v4 job/epoch/deadline/artifact/session/decode-policy binding | fault-injection tests are repository evidence; target-hardware endurance remains external |
| 3 | frozen Gemma features feed a bounded equilibrium reasoner and causal decode bias | `canary`; no trained DEQ/Wproj quality artifact |
| 4 | fixed 301-node CTS V2, rank-16 solver history, bounded retrieval/policy/critic surfaces | depth-100 was measured on the attached RTX 5090 Laptop GPU, not RTX 4090 |
| 5 | strict Fast Weight checkpoint, AQ/OOD, norm, TTL and session gates | no admitted trained product checkpoint; remains `gated` |
| 6 | empirical Fisher, FP-EWC, C-FIRE and generation transaction/checkpoint | `night_only`; no three-seed BWT/FWT claim |
| 7 | exact 28-agent tensor swarm, certified topologies, sessions and PCAS controls | `advisory`; production calibration/quality evidence absent |
| 8 | bounded 8-slot, top-k=2 expert candidate lifecycle with quarantine and rollback | `advisory`; no independently verified trained expert artifact |
| 9 | immutable typed plans, one-use capabilities, bounded T0/T1 execution and T2 proposal staging | T3/network/arbitrary shell are denied; free-form model plans have no authority |
| 10 | sealed, replayable, held-in/out AFlow research evaluator and archive | `research_archive_only`; no production installation path |
| 11 | persistent success/failure ledger, causal signatures, K≥3 distinct proposals and negative archive | `proposal_only`; active source mutation is forbidden |

Detailed gates and commands are in
[Validation](docs/VALIDATION.md) and
[Gemma validation](docs/GEMMA4_VALIDATION.md).

## Run from source

Requirements are Python 3.11+, a compatible local PyTorch/CUDA environment,
and a complete local model artifact. Runtime downloads, Hub IDs, remote code,
telemetry, and external APIs are not permitted.

```powershell
python -m pytest -q
python -m ruff check .
python -m ruff format --check cogni_agent cogni_core cogni_demo cogni_flow cogni_os scripts tests
```

Verify the actual local model and bounded depth-100 runtime:

```powershell
python scripts\validate_gemma4_runtime.py `
  --model C:\Project\cognios\gemma4-e4b-it `
  --manifest config\gemma4-e4b-it.manifest.toml `
  --event-stream
```

Run the recommended 20-turn completion stress and the natural Korean gate:

```powershell
python scripts\validate_agent_completion.py `
  --model C:\Project\cognios\gemma4-e4b-it `
  --manifest config\gemma4-e4b-it.manifest.toml `
  --turns 20

python scripts\validate_agent_casual_korean.py `
  --model C:\Project\cognios\gemma4-e4b-it `
  --manifest config\gemma4-e4b-it.manifest.toml `
  --timeout 120
```

The pretrained base checkpoint at `C:\Project\cognios\gemma4-e4b` is retained
only for explicit research/canary reproduction. It is not accepted as the
public conversation runtime.

Run the System 4 stress benchmark. Its output separates instantaneous
hysteresis mismatch from post-settling PCAS errors:

```powershell
python scripts\benchmark_system4.py `
  --device cuda --iterations 10000 --stress-switch --switch-block 32
```

## Windows launcher

Build the v0.3.2 launcher from the exact source tree:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\build_windows_launcher.ps1 `
  -OutputPath release\CogniBoard-v0.3.2.exe
```

The launcher is a console-free bootstrapper, not a standalone model bundle. It
requires this source tree, local Python/CUDA dependencies, and the verified
model plus manifest. The binary is not code-signed in this research release.
Do not use older release assets as v0.3.2 evidence.

Korean operator documentation: [`docs/COGNIBOARD_USER_MANUAL_PLAYBOOK_KO.md`](docs/COGNIBOARD_USER_MANUAL_PLAYBOOK_KO.md).

The v0.3.2 validation addendum is
[`release/COGNI_OS_0.3.2_VALIDATION_ADDENDUM_KO.md`](release/COGNI_OS_0.3.2_VALIDATION_ADDENDUM_KO.md).

## Runtime boundaries

- One spawned worker owns the model and CUDA. Model IPC is bounded,
  tensor-only and bound to a lease epoch, job, deadline, artifact and session.
- Gemma decoding uses `use_cache=False`; CTS does not accumulate a KV cache.
- Solver history and active CTS search state are fixed-capacity. This is not an
  O(1) claim for model weights, logs, expert banks, or unbounded external data.
- Local tasks execute only typed, allowlisted operations. Natural language is
  never passed to a shell.
- AFlow can only create a bounded research archive.
- Self-Harness stores inert proposals and cannot install them in v0.3.2.
- The graphical server is loopback-only, uses per-session authentication and
  local allowlisted assets, and has no CDN or analytics dependency.

The latest retained pretrained-base depth-100 canary recorded 14.8469 GiB peak
allocated VRAM on an RTX 5090 Laptop GPU. That historical, scoped observation
is not E4B-it product-runtime evidence or a guarantee for other prompts,
software stacks, reserved memory, or the target RTX 4090. The 16.7 GiB
postcondition still fails closed and must be repeated with the pinned E4B-it
artifact on the target device.

## Documentation

- [Architecture and Phase 1–11 data flow](docs/ARCHITECTURE.md)
- [Security and authority boundaries](docs/SECURITY.md)
- [Validation gates and evidence classes](docs/VALIDATION.md)
- [Local Gemma validation record](docs/GEMMA4_VALIDATION.md)
- [Korean AI workspace guide](docs/AGENT_WORKSPACE_KO.md)
- [Phase 8 System 3](docs/PHASE8_SYSTEM3.md)
- [Phase 9 typed task plan](docs/PHASE9_TYPED_TASK_PLAN.md)
- [Phase 10 AFlow research executor](docs/PHASE10_AFLOW_RESEARCH.md)
