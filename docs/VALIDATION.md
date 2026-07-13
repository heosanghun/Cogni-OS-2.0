# Cogni-OS 2.0 v0.3.2 Validation Gates

## Evidence rule

A passing unit test verifies a software invariant. It does not by itself prove
model quality, target-hardware memory, air-gap deployment, trained-artifact
quality, independent verification, or safe automatic promotion.

Public claims use four evidence classes:

| Class | Meaning |
|---|---|
| `measured` | raw observation under one exact model/code/config/device scope |
| `verified` | deterministic schema, policy or software invariant verified in that scope |
| `target` | acceptance threshold, not a result |
| `plan` | future work or business plan, not a runtime capability |

An evidence record is valid only while all cited content-addressed records are
present and match the current model, code, config and device digests. The JSON
contract is [`config/evidence.schema.json`](../config/evidence.schema.json).
Runtime evidence and last-known-good snapshots are stored outside the source
tree; generated local `.cogni_state` data is not a release artifact.

The UI starts with runtime evidence marked `UNVERIFIED` and no copied metric
values. Only a successful validation run in the current process and current
scope may promote live evidence. Historical snapshots or embedded display
defaults cannot authorize `VERIFIED`.

## Release-candidate evidence status

Phase 1–11 implementation does not by itself make release measurements pass.
The retained current-scope evidence has the following status:

| Gate | Status |
|---|---|
| full source regression (`ruff`, format, `pytest`) | `PASS` |
| automated 20-turn local completion run | `PASS` |
| integrated Gemma/DEQ/CTS GPU runtime measurement | `PASS (RTX 5090 Laptop measured)` |
| final System 4 stress measurement | `PASS (measurement-only/advisory)` |
| natural Korean 10-turn release gate | `PASS (local Gemma + bounded product routes)` |
| final release bytes and `SHA256SUMS.txt` | `PASS` only after regeneration from the frozen v0.3.2 commit |

These passes are scoped to the retained source/model/config/device evidence.
They do not certify the target RTX 4090, code signing, or a standalone installer.

The retained current-scope JSON records are under `release/evidence/`: the
20-turn conversation transcript, integrated Gemma/CTS runtime result, System 4
stress result, and source-regression summary. Release byte identity is recorded
separately in `release/SHA256SUMS.txt`, which is excluded from source archives.

## Source validation

Run from the repository root:

```powershell
python -m ruff check .
python -m ruff format --check cogni_agent cogni_core cogni_demo cogni_flow cogni_os scripts tests
python -m pytest -q
```

The exact number of collected tests is intentionally not a release contract:
new security cases change it. A release candidate must retain the complete raw
command output, source commit/tree digest, Python/PyTorch versions and skipped
test reasons. No historical fixed count is accepted as v0.3.2 evidence.

The current retained run passed Ruff and format checks plus Node syntax check.
Pytest completed with 709 passed, 3 skipped, 3 deprecation warnings and
565 subtests passed in 112.44 seconds. These counts
describe this validation snapshot; they are not a permanent product contract.

The suite covers the following Phase 1–11 invariants:

- Fact-book artifact identity, exact parameter inventory, capability states,
  content-addressed evidence, scope invalidation and last-known-good recovery;
- repetition, role/control-token leakage, false completion, incomplete Korean,
  bounded regeneration and transactional conversation abort;
- lease epoch/job/deadline/artifact/session IPC binding, stale/late rejection,
  pause/resume/restart/capacity and A/B session isolation;
- DEQ forward/IFT checks, non-contractivity handling and causal decode bias;
- fixed 301-node CTS allocation, rank-16 history, bounded retrieval, policy/
  critic separation, failed-edge telemetry and MAC budget;
- Fast Weight trained-checkpoint gate, AQ/OOD calibration, composed norm,
  session TTL/LRU and fallback;
- empirical Fisher, FP-EWC anchoring/merge, C-FIRE and atomic generation
  checkpoint transactions;
- exact 28-agent System 4 topology, global operator/spectral checks, session
  isolation, PCAS hysteresis and bounded stress behavior;
- eight-slot top-k=2 System 3 profile, inactive isolation, novelty calibration,
  candidate-only training, routed Fisher, quarantine, rollback and verifier
  authority separation;
- immutable typed TaskPlans, exact risk tiers, one-use capabilities, budgets,
  path/reparse/TOCTOU defenses, default-denied T1 fixed pytest trusted opt-in,
  output artifact writes and SHA read-back;
- sealed AFlow evaluator/policy/suite, bounded DAG/search/archive, repeated
  held-in/out metrics, lineage/replay and `research_archive_only` output;
- persistent Self-Harness success/failure evidence, strict bounded restart
  hydration, causal signatures, K≥3 distinct candidates, immutable surfaces,
  negative archive and zero mutation in proposal-only mode.

## Local Gemma artifact and runtime

The model path and manifest must be local. Run the artifact gate, experimental
decoder-DEQ smoke, integrated CTS runtime and conversation completion
separately:

```powershell
python scripts\validate_gemma4.py `
  --model C:\Project\cognios\gemma4-e4b `
  --manifest config\gemma4-e4b.manifest.toml

python scripts\validate_gemma4_deq.py `
  --model C:\Project\cognios\gemma4-e4b `
  --manifest config\gemma4-e4b.manifest.toml `
  --allow-uncertified-experimental

python scripts\validate_gemma4_runtime.py `
  --model C:\Project\cognios\gemma4-e4b `
  --manifest config\gemma4-e4b.manifest.toml `
  --event-stream

python scripts\validate_agent_completion.py `
  --model C:\Project\cognios\gemma4-e4b `
  --manifest config\gemma4-e4b.manifest.toml
```

The second command is explicitly an **uncertified convergence smoke test**.
It is not proof that the whole Gemma decoder has global Lipschitz constant
below 0.95. Production use without the experimental flag requires an
independently derived decoder-delta certificate and fails before execution
when it is missing or unsafe.

The integrated runtime gate must:

- verify all manifest files before model load;
- use `use_cache=False` and report no CTS KV cache;
- reach requested depth 100 with at most 301 nodes;
- keep all backbone/search/logits tensors finite;
- expose solver rank, residual, failed edges, unsafe fallbacks, ACT/MAC budget,
  trace digest and causal-bias telemetry;
- report allocated and reserved CUDA memory plus the physical device;
- fail if the configured 16.7 GiB allocated-VRAM postcondition is exceeded.

The retained integrated run verified 6 manifest files and loaded
`Gemma4ForConditionalGeneration` with hidden size 2560. Model load took
22.532 seconds and integrated inference took 12.818 seconds. CTS reached depth
100/100 with 301/301 nodes and allocated 57,243,710 search bytes. The final DEQ
residual was 0.0015945435 with rank/history 16; linear/silent fallbacks,
solver failures and failed edges were all zero. ACT was 301 and the MAC ledger
reserved 2,008,252,704 of a 3,209,427,616 budget. The causal bridge was active
and non-zero with maximum absolute logits bias 0.0498046875. Peak allocated
VRAM was 14.8468875885 GiB, below 16.7 GiB, on an NVIDIA GeForce RTX 5090
Laptop GPU. The retained trace digest begins `77dcdadc`.

This is a scoped measurement on the development GPU, not RTX 4090
certification.

The product factory keeps Gemma last-known-good/base-only generation fallback
disabled. A response is published atomically only after Cogni-Core terminal
success and base-integrity checks. Core failure must fail closed; the bounded
static quality response used after response-repair exhaustion is not a Gemma
base-model fallback.

The completion gate supports 1–100 turns; the recommended release run is 20.
It requires a terminal stop for each turn, no hard-limit truncation, no role/
control-token leakage, no repetitive loop, no false model identity, clean
worker shutdown and bounded stream completion. The retained
`phase11_gemma_20turn_cognicore_release_final_v6_20260712.json` passed 20/20
turns with zero quality fallbacks, zero repeated sentences and zero substantive
cross-turn sentence reuse. Worker-memory coverage was 100%; 15 turns used
Cogni-Core generation, whose max and p95 latency were both 78.516 seconds, and
every turn stayed within 180 seconds. Worker cleanup and GPU-lease release were
both true. Per-process GPU memory was not reported by the driver, so that
conversation-specific metric remains `unverified`; it is not inferred from the
separate integrated-runtime peak measurement. This automated pass does not
replace the independent external human-labelled 20-turn acceptance study.

See [GEMMA4_VALIDATION.md](GEMMA4_VALIDATION.md) for the latest scoped canary
record and its limitations.

## System 4 stress gate

```powershell
python scripts\benchmark_system4.py `
  --device cuda `
  --iterations 10000 `
  --stress-switch `
  --switch-block 32
```

Record warm-path p50/p95/p99/max latency, convergence and finite rates,
operator norm, spectral radius, switch count, instantaneous mismatch,
post-grace settled FPR/FNR, and transition settling samples. Hysteresis makes
instantaneous phase mismatch different from an error after settling; both must
be reported. A synthetic switching trace is not a production PCAS calibration
corpus and cannot promote System 4 above `advisory`.

The retained CUDA 10,000-iteration run measured p50 2.9157 ms,
p95 3.893715 ms, p99 5.217861 ms and max 7.3089 ms. Convergence/finite rate was
1.0 and maximum residual was 4.6480327e-05. Operator norms were
0.32436395/0.40665507 (both below 0.95) and spectral radius was 0.15065038.
All 312/312 topology switches were observed; settled FPR/FNR were both zero,
with settling max 5 samples and mean 4 samples. This remains a synthetic
measurement-only/advisory result, not production PCAS calibration or
answer-quality evidence.

## Typed-task execution gate

T0 bounded reads and verified output-artifact writes are available. The fixed
pytest primitive is T1 and is **blocked by default** in the product. It may run
only when a trusted host explicitly opts in after accepting that the current
runtime does not attest an OS-level process-tree/network sandbox. T2 may only
stage an inert proposal and T3 remains denied.

## Required external acceptance evidence

These are not completed by the repository test suite and remain gated:

- the exact target RTX 4090 24GB, with full depth-100 decode under 16.7 GiB;
- packet/egress audit and offline dependency/provenance review;
- an independent external human-labelled 20-turn raw conversation study with
  false-complete, repetition and truncation labels;
- trained DEQ adapter/Wproj and causal held-out ablation;
- trained Fast Weight Programmer checkpoint with AQ/OOD and non-regression
  evidence;
- at least three-seed FP-EWC BWT/FWT domain results;
- production PCAS calibration corpus and System 4 quality ablation;
- trained/calibrated System 3 expert checkpoint and independent verifier;
- attested real AFlow evaluator and held-out non-regression;
- real production failure-corpus capture ≥99% and manually labelled cluster
  precision ≥95%;
- Phase 12 kernel-isolated sandbox, promotion fault injection and byte-identical
  rollback evidence;
- code signing, a signed installer/update trust chain, SBOM, licenses and
  provenance bundle.

Until those artifacts exist and validate in the current scope, the corresponding
capability must remain `gated`, `advisory`, `research`, `night_only`, or
`proposal_only` as declared by the Runtime Fact-book.
