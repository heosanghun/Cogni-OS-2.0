# Cogni-OS 2.0 v0.4.1 Validation Gates

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
Raw runtime guard evidence, raw logs and last-known-good snapshots are stored
outside the source tree in owner-only storage; generated local `.cogni_state`
data is not a release artifact. After independent review, an immutable,
content-addressed release summary may be committed under
`release/evidence/v0.4.1` only when it cites the exact external raw-evidence
digest and source scope. Raw guard evidence and raw logs are never committed.

The UI starts with runtime evidence marked `UNVERIFIED` and no copied metric
values. Only a successful validation run in the current process and current
scope may promote live evidence. Historical snapshots or embedded display
defaults cannot authorize `VERIFIED`.

## Release-candidate evidence status

Phase 1–11 implementation does not by itself make release measurements pass.
The retained current-scope evidence has the following status:

| Gate | Status |
|---|---|
| full source regression (`ruff`, format, `pytest`, Node syntax) | `REQUIRED AFTER v0.4.1 SOURCE FREEZE` |
| automated 20-turn local completion run | `REQUIRED AFTER v0.4.1 SOURCE FREEZE` |
| integrated Gemma/DEQ/CTS GPU runtime measurement | `REQUIRED AFTER v0.4.1 SOURCE FREEZE` |
| final System 4 stress measurement | `REQUIRED AFTER v0.4.1 SOURCE FREEZE` |
| natural Korean 10-turn release gate | `REQUIRED AFTER v0.4.1 SOURCE FREEZE` |
| local Korean TTS → Gemma STT smoke | `HISTORICAL DEV-HOST OBSERVATION; CURRENT SCOPE UNVERIFIED` |
| deterministic image-understanding smoke | `HISTORICAL DEV-HOST OBSERVATION; CURRENT SCOPE UNVERIFIED` |
| live Lens patent/scholarly query | `NOT RUN (user token and terms acceptance not bundled)` |
| final release bytes and `SHA256SUMS.txt` | `PASS` only after regeneration from the frozen v0.4.1 commit |

Only records that validate against the evidence schema and bind the exact
source/model/config/device and raw-evidence digests can be called current-scope
passes. The retained voice and image JSON files predate that contract and are
historical observations only; they do not certify the target RTX 4090, code
signing, or a standalone installer.

The older JSON records under `release/evidence/` remain historical comparison
material. They cannot authorize the changed v0.4.1 source scope. Final v0.4.1
records and release bytes must be regenerated from one frozen commit; release
byte identity is recorded separately in `release/SHA256SUMS.txt`, which is
excluded from source archives.

## Source validation

Run from the repository root:

```powershell
python -m ruff check .
python -m ruff format --check cogni_agent cogni_core cogni_demo cogni_flow cogni_os scripts tests
python -m pytest -q
```

The double-click launchers have intentionally separate roles:

- `Run-CogniOS-Demo.cmd` starts only the ordinary Windows
  `desktop-ui-only` appliance. A caller cannot switch it to the native GPU
  server profile.
- `Run-CogniOS-CLI.cmd` runs only the CPU/static integrity diagnostic: Python
  3.11 isolated mode, the 170-item acceptance-ledger validator, and an AST
  integrity check of the repository-anchored bootstrap. It does not load a
  model, probe CUDA, invoke `nvidia-smi`, or produce live hardware evidence.
- `Run-CogniOS-Server-GPU5.sh` is the Linux native product-server operator
  launcher. It accepts the model location only through `COGNI_OS_MODEL_DIR` and
  delegates all GPU access to the server after the shared lease is acquired.

The exact number of collected tests is intentionally not a release contract:
new security cases change it. A release candidate must retain the complete raw
command output, source commit/tree digest, Python/PyTorch versions and skipped
test reasons. No historical fixed count is accepted as v0.4.1 evidence.

The previous retained v0.3.2 run passed Ruff and format checks plus Node syntax
check, with 709 passed, 3 skipped, 3 deprecation warnings and 565 subtests in
112.44 seconds. Those counts describe only that historical snapshot and do not
pass the v0.4.1 release gate.

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
  negative archive and zero mutation in proposal-only mode;
- persistent attachment catalog/blob recovery, content and path integrity,
  PDF bounds plus isolated worker timeout/resource contract,
  preview/delete/reindex transactions and authenticated content;
- pinned AkasicDB module digests, bounded indexing/query, answer-bearing
  `RetrievalEvidence`, provenance and restart reconstruction;
- fixed-schema Gemma image/audio tensor preprocessing and IPC, malformed shape/
  dtype/size rejection, explicit one-turn image admission and request binding;
- bounded browser WAV admission, local STT/TTS fail-closed contracts, and UI
  capture/playback lifecycle cleanup;
- four-gate Lens authorization, fixed official endpoints, bounded responses,
  secret redaction, attribution, provenance and explicit Lens-to-RAG indexing;
- read-only Self-Harness diff review with no approve/apply endpoint in the
  shipped UI/default profile;
- internal ATTESTED candidate evaluation, immutable evaluation/approval
  binding, one-time Ed25519 promotion and separately signed committed rollback
  software contracts, plus the operator-only append-only validator for the
  promotion/health/separately signed rollback chain. These are implementation
  tests, not production isolation attestation, an independently issued runner
  statement, or current raw production E2E evidence.

## Workspace, multimodal, and voice gates

Run the deterministic component contracts before loading the GPU model:

```powershell
python -m pytest -q `
  tests/test_workspace_capabilities.py `
  tests/test_workspace_capabilities_api.py `
  tests/test_multimodal_processor.py `
  tests/test_multimodal_worker_ipc.py `
  tests/test_audio_worker_ipc.py `
  tests/test_local_voice.py `
  tests/test_local_tts.py `
  tests/test_voice_api.py `
  tests/test_voice_ui_contract.py `
  tests/test_lens_api.py `
  tests/test_proposal_review.py
```

These tests use injected processors/fake workers for image, audio, and decoded
video-frame contracts. They do not construct the production
`Gemma4Processor`: that path-loader is deliberately fail-closed because the
upstream path API cannot bind parser input to the verified bytes. Consequently
a passing component run does not grant current image/audio/video answer
authority or resident-Gemma STT authority.

The historical direct voice and image commands are intentionally not published:
both can select a default CUDA device and neither validator is in the sole GPU5
guard allowlist. They must not run on the laboratory server. Current-scope GPU
evidence remains blocked until each validator has an exact guarded argv,
identity-before-load and identity-after-shutdown checks, and a manifest-bound
instruction-profile mount.

The 2026-07-16 development-host voice smoke synthesized
`안녕하세요. 코그니보드 로컬 음성 검증입니다.` with Microsoft Heami Desktop
`ko-KR`, converted it to 16 kHz mono PCM, and transcribed the same phrase with
the manifest-bound resident Gemma service. Normalized similarity was 1.0,
duration was 5.0187 seconds, and both STT and TTS reported zero application
external calls. This one synthetic phrase proves only that exact pipeline
smoke; it is not WER, noisy-speech, speaker-diversity, latency, multilingual,
or accessibility acceptance evidence.

The 2026-07-16 deterministic image smoke sent a 595-byte locally generated
256×256 PNG with one blue square through the manifest-bound Cogni-Core image
path. The model returned `중앙의 큰 도형은 파란색 정사각형입니다.` with
`finish_reason=stop`, `generation_mode=cogni_core`, both concept gates true,
and zero external calls. This one fixed case does not establish general visual
reasoning or combined image-plus-depth-100 VRAM conformance. Video has no
validation route in v0.4.1.

Lens tests use an injected fake transport and prove only local policy/schema
invariants. A live request is permitted only when online opt-in, exact
`api.lens.org` allowlisting, a user token, and explicit terms acceptance all
exist. No release token is stored, and no live Lens result is claimed.

## Local Gemma artifact and runtime

The model path and manifest must be local. The product runtime and conversation
gates require the exact pinned Gemma 4 E4B-it artifact and its seven-file
manifest. The pretrained base checkpoint is retained only for the explicitly
uncertified research/canary decoder-DEQ smoke:

The GPU-bearing gates below are server operations.  After the clean exact-commit
CPU Exit Gate, the only Stage G GPU entry point is
`gpu5_boundary_guard.py run`; direct native validator commands cannot create
release evidence.  The Windows appliance starts as `desktop-ui-only`; it
preserves the UI and conversation path but cannot silently turn an unbounded
desktop device into server evidence.

```bash
/usr/bin/python3 -I -B \
  -X pycache_prefix=/home/shoon/.cognios-gpu5-guard/host-never-pycache \
  scripts/gpu5_boundary_guard.py run \
  --image cogni-os-dev@sha256:20aaf1d7cde8d6a504ba08f158a34a1907eac9413f3578acc4637f0a1b2ec8ba \
  --expected-source-commit "$(git rev-parse HEAD)" \
  --validation-artifact-profile base-canary \
  --workdir /workspace \
  --timeout 1800 \
  --evidence-filename gpu5-runtime-v041.json \
  -- -I -B /workspace/scripts/validate_gemma4_runtime.py \
  --model /models/gemma4-e4b \
  --manifest /workspace/config/gemma4-e4b.manifest.toml \
  --physical-gpu-index 5 \
  --gpu-query-context gpu5-container \
  --event-stream
```

The completion and DEQ validators use the same guard with distinct evidence
filenames and their allowlisted container commands.  An uncertified DEQ smoke
test is never release evidence.  Production DEQ evidence requires an
independently derived decoder-delta certificate and fails before execution
when provenance is missing or unsafe.

The command above is the preserved `base-canary` research/runtime path. Product
conversation evidence selects the immutable `product-e4b-it` profile and cannot
reuse the base model, manifest or validators:

```bash
/usr/bin/python3 -I -B \
  -X pycache_prefix=/home/shoon/.cognios-gpu5-guard/host-never-pycache \
  scripts/gpu5_boundary_guard.py run \
  --image cogni-os-dev@sha256:20aaf1d7cde8d6a504ba08f158a34a1907eac9413f3578acc4637f0a1b2ec8ba \
  --expected-source-commit "$(git rev-parse HEAD)" \
  --validation-artifact-profile product-e4b-it \
  --workdir /workspace \
  --timeout 3600 \
  --evidence-filename gpu5-product-e4b-it-20.json \
  -- -I -B /workspace/scripts/validate_agent_completion.py \
  --model /models/gemma4-e4b-it \
  --manifest /workspace/config/gemma4-e4b-it.manifest.toml \
  --physical-gpu-index 5 \
  --gpu-query-context gpu5-container \
  --turns 20 \
  --suite product-e4b-it-20 \
  --strict-json
```

The product release command is one ModelService/lease and exactly 20 turns. Its
fixed cases cover casual Korean, typo tolerance, follow-up context, continuation
completion, repetition resistance, exact Fact-book identity, false-7B, public
role/control-marker and cutoff rejection, and zero quality fallback. The guard
requires a single passed `cogni.agent.completion.stress.v2` JSON component and
independently recomputes its fixed turn order/routes, exact quality-check keys,
answer digests, A/B isolation, one resident PID, all 16 required GPU spot
samples, seven-file manifest binding, GPU5 identities and cleanup. The first
turn must expose `gemma4-e4b-it`, effective `4,506,496,490` and stored
`7,996,157,418` parameters; an empty turn list or self-authored PASS summary is
rejected.
Implementation alone does not create evidence: this route remains `NOT RUN`
until the exact command succeeds on idle physical GPU5 and independent review
accepts the retained component.

The operational resident server is started separately from Stage G evidence:

```bash
export COGNI_OS_PYTHON=/verified/cognios-venv/bin/python
export COGNI_OS_MODEL_DIR=/verified/local/gemma4-e4b-it
./Run-CogniOS-Server-GPU5.sh
```

The verified Gemma 4 E4B-it artifact and a Python 3.11+ environment are external
prerequisites. A trusted parent process or service manager must provide the
documented minimal environment before the launcher is loaded. This is the
pre-shebang trust boundary because loader variables (`LD_AUDIT`,
`LD_LIBRARY_PATH`) and Bash startup hooks (`BASH_ENV`) can act before the shell
script body runs.

The shebang and sanitized re-exec use `bash -p`, and security-critical
`exec`/`exit`/`printf` calls select their Bash builtins explicitly, so exported
functions cannot shadow them. The launcher then immediately re-executes itself
through an exact `/usr/bin/env -i` environment and verifies that sanitized
stage. Git,
`readlink`, the interpreter sentinel and the final Python bootstrap each run
with a purpose-specific exact environment; loader, Bash, Git configuration,
proxy, credential and unrelated caller variables are not inherited.
`COGNI_OS_PYTHON`, when set, is a bounded absolute invocation path. The
launcher preserves that path so a verified virtual environment retains its
`sys.prefix`, while separately resolving the executable target for ownership
and mode checks. The resolved target must be an operator-trusted ELF
executable. An exact isolated CPython 3.11+ runtime check requires `-I -B`,
safe-path/no-user-site flags, the expected resolved `sys.executable`, and live
`/proc/self/exe`; shell wrappers and accidental non-Python executables are
rejected. This is not cryptographic attestation of an arbitrary same-user ELF
binary. Such a threat requires a committed executable digest or equivalent
signed provenance at the operator/service-manager trust boundary. Without an
override the launcher checks `/usr/bin/python3` and fails closed when it is
unsuitable.

The launcher requires a clean lowercase 40-hex HEAD, installs the pinned GPU5
UUID visibility environment, and invokes the selected interpreter with
`-I -B` plus `--expected-source-commit`. It never calls `nvidia-smi` itself.
The server owns the shared cross-process lease, exact-index
idle/foreign-PID proof, identity check, resident lifetime, cleanup, and
postflight; a failed safety postcheck poisons the lease instead of silently
releasing it.

The integrated runtime gate must:

- verify all manifest files before model load;
- use `use_cache=False` and report no CTS KV cache;
- reach requested depth 100 with at most 301 nodes;
- keep all backbone/search/logits tensors finite;
- expose solver rank, residual, failed edges, unsafe fallbacks, ACT/MAC budget,
  trace digest and causal-bias telemetry;
- report allocated and reserved CUDA memory plus the physical device;
- fail if the configured 16.7 GiB allocated-VRAM postcondition is exceeded.

The retained historical base-checkpoint canary run verified 6 manifest files
and loaded
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

The product factory keeps Gemma last-known-good/backbone-only generation fallback
disabled. A response is published atomically only after Cogni-Core terminal
success and backbone-integrity checks. Core failure must fail closed; the bounded
static quality response used after response-repair exhaustion is not a Gemma
base-model fallback.

The completion gate supports 1–100 turns; the recommended release run is 20.
It requires a terminal stop for each turn, no hard-limit truncation, no role/
control-token leakage, no repetitive loop, no false model identity, clean
worker shutdown and bounded stream completion. The retained
`phase11_gemma_20turn_cognicore_release_final_v6_20260712.json` passed 20/20
turns with zero quality fallbacks, zero repeated sentences and zero substantive
cross-turn sentence reuse. Worker post-turn spot-sample coverage was 100%; 15 turns used
Cogni-Core generation, whose max and p95 latency were both 78.516 seconds, and
every turn stayed within 180 seconds. Worker cleanup and GPU-lease release were
both true. Per-process GPU memory was not reported by the driver, so that
conversation-specific metric remains `unverified`; it is not inferred from the
separate integrated-runtime peak measurement. This automated pass does not
replace the independent external human-labelled 20-turn acceptance study.

Completion evidence uses the
`cogni.agent.completion.stress.v2` schema. It requires one declared post-turn
spot sample per expected resident-worker turn, but it never labels the maximum
of those samples as a peak and does not certify sustained usage or whole-runtime
VRAM release. Full-runtime peak certification is owned by
`validate_gemma4_runtime.py` and its `torch.cuda.max_memory_allocated`
telemetry. The strict completion stress gate requires complete spot-sample
coverage and rejects any observed value above 16.7 GiB. Values at or below the
boundary remain point observations and cannot satisfy the separate peak gate.

See [GEMMA4_VALIDATION.md](GEMMA4_VALIDATION.md) for the latest scoped canary
record and its limitations.

## System 4 stress gate

```powershell
python scripts\benchmark_system4.py `
  --device cpu `
  --iterations 10000 `
  --stress-switch `
  --switch-block 32
```

This CPU run verifies software invariants only. A System 4 CUDA run is
`NOT RUN / BLOCKED` on the lab server until the benchmark has an exact GPU5
guard allowlist and before/after identity contract. Direct `--device cuda`
execution is prohibited.

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
- independent hostile-code production isolation evidence for the OCI
  daemon/runtime/kernel/userns/seccomp/AppArmor/socket boundary, plus an exact
  one-environment candidate→external approval→promotion→signed committed
  rollback E2E. The named local engine/image CPU integration smoke and fault
  tests do not satisfy this gate. Its schema is
  `cogni.kernel-sandbox-integration-smoke.v1`, assurance is explicitly
  `implementation_integration_smoke_only`, `production_attestation=false`, and
  `gpu_measurement=not_performed`;
- code signing, a signed installer/update trust chain, independent license
  review and distribution approval. The local builder emits a CycloneDX SBOM,
  declared-license inventory and artifact/checksum provenance but cannot turn
  metadata into legal approval or a cryptographic signature.

Until those artifacts exist and validate in the current scope, the corresponding
capability must remain `gated`, `advisory`, `research`, `night_only`, or
`proposal_only` as declared by the Runtime Fact-book.
