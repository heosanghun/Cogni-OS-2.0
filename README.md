# Cogni-OS 2.0 Genesis — v0.4.1

Cogni-OS 2.0 is an offline-by-default, bounded research runtime for a verified local
dense Gemma 4 E4B-it artifact. Version 0.4.1 adds a persistent local document
workspace, provenance-bearing AkasicDB RAG, bounded Gemma image/audio protocol
contracts, decoded-frame-only video preprocessing tests, local speech UI and
Windows TTS contracts, an
explicitly opt-in Lens patent/scholarly API connector, and a separate
request-opt-in Brave official JSON search boundary to the existing
conversation, Cogni-Core, typed-task, and
**proposal-only-by-default** Self-Harness boundary. An internal Linux
`PromotionMode.ATTESTED` path exists for evidence-bound evaluation, external
signed promotion and signed committed rollback, but it is not enabled by the
desktop UI or production-attested in this release candidate.

This release is not an AGI claim, an RTX 4090 certification, or evidence that
every research module improves model quality. A Python implementation or a
passing component test never upgrades a capability by itself.

## Product authority at a glance

| Capability | v0.4.1 state | May affect the answer? | What is still required |
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
| Self-Harness | default `proposal_only`; internal `ATTESTED` path unverified | no mutation in the shipped UI/default profile | independent production-runner statement and current raw production E2E evidence |
| persistent attachments + PDF extraction | `local_verified_path` | only after explicit RAG/image selection | deployment-specific parser and corpus review |
| AkasicDB RAG | `local_index_ready` when pinned clone verifies | yes, with bounded lexical source provenance | semantic production path-loader is disabled until verified bytes can be bound to parsing; then artifact/license/quality/poisoning evidence |
| Gemma image/audio tensor IPC | protocol and injected-processor tests only; production path-loader disabled | no current answer authority | byte-bound processor loader, guarded current-scope model run, broad quality and combined-VRAM evidence |
| decoded-frame video preprocessing | `partial_non_answer_bearing_boundary`, injected processor only | no | byte-bound processor loader, decoder worker, IPC, real Gemma forward, quality/latency/VRAM evidence |
| local STT/TTS | `historical_dev_host_observation`; current scope unverified | gated until current guarded evidence | GPU5-guarded current-scope run, multi-speaker WER, noise, language, latency and accessibility study |
| Lens patent/scholarly search | `opt_in_gated` | only after explicit search/index action | token, terms acceptance, allowlist, live validation and distribution attribution review |
| Brave general web search | `implemented_unverified`, default off | no automatic model/RAG authority | approved subscription/token, live response, browser cancel/revoke, terms and packet/egress audit |

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
| 11 | persistent success/failure ledger, causal signatures, K≥3 distinct proposals, negative archive and an operator-only E2E evidence-chain validator | the shipped UI/default profile forbids mutation; the internal `ATTESTED` path remains non-automatic and production-unattested |

Detailed gates and commands are in
[Validation](docs/VALIDATION.md) and
[Gemma validation](docs/GEMMA4_VALIDATION.md).

## Run from source

Requirements are Python 3.11+, a compatible local PyTorch/CUDA environment,
and a complete local model artifact. Runtime downloads, Hub IDs, remote code,
and telemetry are not permitted. Network access remains disabled by default;
the sole implemented exception is the official Lens API connector, which
requires explicit online mode, the exact `api.lens.org` allowlist entry, a
user-supplied API token, and recorded terms acceptance.

```powershell
python -m pytest -q
python -m ruff check .
python -m ruff format --check cogni_agent cogni_core cogni_demo cogni_flow cogni_os scripts tests
```

Start the ordinary Windows appliance with `Run-CogniOS-Demo.cmd`. Its only
supported profile is `desktop-ui-only`: conversation and the rest of
CogniBoard remain
available, but the live hardware-validation button fails closed before Popen.
The GPU5-only validator is a server evidence gate, not an implicit desktop
capability.

`Run-CogniOS-CLI.cmd` is a separate Windows CPU/static integrity diagnostic. It
uses Python isolated mode to validate the machine-readable 170-item master
acceptance ledger and parse the repository-anchored server bootstrap. It never
loads the model, probes CUDA, invokes `nvidia-smi`, or claims live hardware
evidence.

On the designated Linux server, every GPU evidence run goes through the sole
`gpu5_boundary_guard.py run` entry point.  Direct validator commands are not
release evidence.  After the clean, exact-commit CPU Exit Gate, Stage G uses:

```bash
/usr/bin/python3 -I -B \
  -X pycache_prefix=/home/shoon/.cognios-gpu5-guard/host-never-pycache \
  scripts/gpu5_boundary_guard.py run \
  --image cogni-os-dev@sha256:20aaf1d7cde8d6a504ba08f158a34a1907eac9413f3578acc4637f0a1b2ec8ba \
  --expected-source-commit "$(git rev-parse HEAD)" \
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

The product `server-gpu5-native` profile is an operational path only after the
CPU Exit Gate and Stage G approval.  Before any resident model is constructed,
it rechecks the exact device identity and single logical-device scope.  Its
Python process must start with `-I -B`, dangerous Python/loader environment
variables unset, and the documented native visibility boundary:

```bash
export COGNI_OS_PYTHON=/verified/cognios-venv/bin/python
export COGNI_OS_MODEL_DIR=/verified/local/gemma4-e4b-it
./Run-CogniOS-Server-GPU5.sh
```

The Linux operator launcher accepts no device selector from the caller. The
model artifact and a Python 3.11+ environment are external prerequisites. A
trusted parent process or service manager must start the launcher with the
documented minimal environment. That is the pre-shebang trust boundary:
loader variables such as `LD_AUDIT`/`LD_LIBRARY_PATH` and Bash startup hooks
such as `BASH_ENV` can act before a shell script body can sanitize itself.

As defense in depth, both the shebang and sanitized re-exec use privileged Bash
mode (`bash -p`) so exported functions cannot shadow `exec`, `exit`, or
`printf`; security-critical calls also select their Bash builtin explicitly.
The script immediately re-executes itself through an exact `/usr/bin/env -i`
environment and verifies that sanitized stage before it runs any repository or
Python command. Git metadata reads also use a separate fixed environment that
ignores system and global Git configuration. The
optional `COGNI_OS_PYTHON` is a bounded absolute invocation path. The launcher
preserves that path so a verified virtual environment keeps its `sys.prefix`,
while separately resolving and checking the executable target's ownership and
mode. The resolved target must be an operator-trusted ELF executable, and an
exact isolated CPython 3.11+ runtime check requires `-I -B`,
safe-path/no-user-site flags, the expected resolved `sys.executable`, and the
live `/proc/self/exe` target. This rejects shell wrappers and accidental
non-Python ELF executables such as `/usr/bin/true`; it is not cryptographic
attestation of an arbitrary same-user ELF binary. The operator/service-manager
trust boundary must protect that runtime. Defending against a malicious
same-user ELF shim requires a separately committed executable digest or
equivalent signed provenance.

The final bootstrap is invoked through another exact `/usr/bin/env -i`
allowlist. Loader, Bash, Git, proxy, credential and unrelated caller variables
are not inherited. The launcher requires a clean exact 40-hex HEAD, installs
the pinned UUID visibility environment, and passes that commit to the isolated
server bootstrap. It performs no GPU query itself; the server acquires the
shared host lease before its exact-index idle proof and identity check, holds
it for the resident lifetime, and completes cleanup plus postflight before
releasing it.

Before the native product imports any repository module, the bootstrap creates
private, quota-bounded source and model snapshots. Source files are copied from
the clean expected commit; model files are streamed into separate inodes and
checked against the closed-world manifest. The prepare process hands the
sealed re-exec the source/model content digests, identity digests, file counts,
root device/inode identities, and model byte count. The sealed process fully
re-inventories those capabilities before acquiring GPU authority. Product
imports and read-only model access use only the sealed roots; the original
workspace remains a separate, non-overlapping writable capability for bounded
outputs and reviewed Self-Harness proposals.

This removes the ordinary verify-then-use window between the mutable checkout
and resident execution, but it does not turn an unprivileged, same-UID process
into a cryptographic trust anchor. A malicious process running as the exact
service UID can change owner-controlled files or forge an unprivileged handoff.
Production operation with that attacker in scope therefore requires a
separately installed root-owned or otherwise immutable launcher/guard and
snapshot handoff, plus a distinct unprivileged runtime UID. The laboratory
workflow treats the dedicated service UID as trusted and fails closed on
foreign ownership, symlinks, hard links, or group/world-writable path
components. No native server or GPU evidence may be claimed when that
deployment precondition is not satisfied.

`server-gpu5-native`는 기존 CogniBoard 세션이나 포트 bind 경쟁 결과를 재사용하지
않습니다. 데스크톱 세션을 먼저 종료한 뒤 새 guarded 프로세스로 시작해야 하며,
그렇지 않으면 프로필·빌드·GPU 증거가 섞이지 않도록 실패 폐쇄합니다.

Omitting or conflicting with any value does not silently fall back to another
device.  A Linux server also rejects `desktop-ui-only`, because its resident
conversation model would otherwise bypass the validation-only boundary.

The guard now has two disjoint immutable artifact profiles. `base-canary`
mounts only the pretrained research checkpoint; `product-e4b-it` mounts only
the separately manifest-bound seven-file instruction artifact and accepts only
the exact integrated 20-turn product command. This is implementation
readiness, not measured evidence: the product Stage G gate remains `NOT RUN`
until the clean frozen commit passes the CPU Exit Gate and then succeeds through
the sole guarded GPU5 entry point. A GPU-bearing validator without a matching
profile and exact argv allowlist is never run on the lab server.

Its `cogni.agent.completion.stress.v2` memory evidence is one post-turn
point-in-time sample for each expected resident-worker turn. It does not
measure or certify within-turn peak, sustained usage, or whole-runtime VRAM
release. Full-runtime peak evidence comes only from
`validate_gemma4_runtime.py` and its `torch.cuda.max_memory_allocated`
telemetry. The completion gate requires complete spot-sample coverage and fails
on any observed sample above 16.7 GiB. Samples at or below that boundary are
still point observations, not peak-VRAM certification.

The pretrained base checkpoint at `C:\Project\cognios\gemma4-e4b` is retained
only for explicit research/canary reproduction. It is not accepted as the
public conversation runtime.

Run the System 4 stress benchmark. Its output separates instantaneous
hysteresis mismatch from post-settling PCAS errors:

```powershell
python scripts\benchmark_system4.py `
  --device cpu --iterations 10000 --stress-switch --switch-block 32
```

This CPU command is a functional benchmark only, not GPU performance evidence.

## Local workspace, RAG, and multimodal input

CogniBoard stores admitted `.txt`, `.md`, `.csv`, `.json`, `.pdf`, `.png`,
`.jpg`, `.jpeg`, and `.webp` attachments in a bounded local catalog. Catalog
and blob integrity are revalidated on restart; users can preview, delete, and
reindex files. PDF text extraction uses the local `pypdf` backend with page,
character, and preview limits.

Local RAG loads only the audited AkasicDB revision
`a6c8e8ebd487e7cb86079f9804a66aaf0914d1dc` and verifies the three storage
module digests before use. Retrieved chunks enter the answer prompt only
through bounded `RetrievalEvidence`, and the UI shows attachment id, chunk
index, score, and source metadata. The bundled vectorizer is deterministic and
bounded; it is **not** represented as a trained semantic embedder. A separate
closed-world semantic artifact verifier and CPU mean-pool test boundary exists,
but the production Transformers path-loader is disabled because it cannot prove
that parsed bytes are the bytes that were hashed. No reviewed model artifact is
bundled; the capability remains `quality_attested=false`, `answer_bearing=false`,
and `production_ready=false`.

Image/audio preprocessing and fixed-schema CPU tensor IPC are implemented and
covered with injected-processor tests. The production `Gemma4Processor`
constructor currently verifies the manifest and then fails closed: the upstream
path API reopens filenames, so a verify-load-verify sequence cannot bind parsed
bytes against an ABA replacement. Consequently this source state has no current
image/audio answer authority. One explicitly selected image can be used for one
turn only after a future verified processor admission.
Browser audio is accepted only as bounded mono 16-bit PCM WAV at 16 kHz and
at most 30 seconds. A retained v0.4.0 historical development-host smoke transcribed a Korean phrase with
normalized similarity 1.0 and synthesized it with the installed Microsoft
Heami `ko-KR` Windows voice, with zero application external calls. This single
synthetic phrase is not a WER or general speech-quality benchmark. Video has
only an already-decoded CPU RGB frame preprocessing boundary with bounded
sampling/pixel/byte/tensor metadata. It accepts no path, URL, container, decoder,
worker IPC, or model-forward authority.

A retained v0.4.0 historical actual-model smoke sent a locally generated 256×256 PNG containing
one blue square through the Cogni-Core image route. The response was
`중앙의 큰 도형은 파란색 정사각형입니다.`, with `finish_reason=stop` and zero
external calls. This one fixed case is not general visual-reasoning or
multimodal VRAM evidence.

Lens search does not scrape web pages. It uses bounded HTTPS POST requests to
the official patent or scholarly API only after all four authorization gates
pass. Results retain Lens provenance and can be explicitly indexed into the
local AkasicDB adapter. No live Lens query is claimed for this release because
the release process has no distributable user token or terms acceptance.

General web search is separate from Lens and targets only Brave's fixed official
JSON GET endpoint. Operator online/provider/host/token/terms gates and a fresh
per-request user opt-in are all required. DNS answers must be public and are
pinned to the TLS connection while hostname verification and SNI remain bound to
`api.search.brave.com`; redirects, HTML scraping, arbitrary URLs, and automatic
result-link fetching are blocked. Mocked tests do not claim a live subscription,
search quality, or egress audit.

## Windows launcher

This source checkout does **not** contain `release\CogniBoard-v0.4.1.exe`.
The path below is the expected build output only. Do not present it as an
available current executable until the frozen exact commit has produced the
binary and its checksum and cold-start smoke have passed.

For a complete adjacent-source **artifact-only** candidate, run the frozen
builder from a clean tracked HEAD. This example hides CUDA, forces Python/Hugging
Face offline mode, creates a new commit-specific directory, and does not launch
the generated executable:

```powershell
$commit = (git rev-parse --verify HEAD).Trim()
git diff --quiet HEAD --
if ($LASTEXITCODE -ne 0) { throw 'Tracked worktree differs from HEAD.' }
git diff --cached --quiet HEAD --
if ($LASTEXITCODE -ne 0) { throw 'Index differs from HEAD.' }

$env:CUDA_VISIBLE_DEVICES = '-1'
$env:NVIDIA_VISIBLE_DEVICES = 'void'
$env:PIP_NO_INDEX = '1'
$env:PIP_DISABLE_PIP_VERSION_CHECK = '1'
$env:HF_HUB_OFFLINE = '1'
$env:TRANSFORMERS_OFFLINE = '1'
$output = "output\release\Cogni-OS-2-Genesis-v0.4.1-unverified-$($commit.Substring(0, 12))"
if (Test-Path -LiteralPath $output) { throw 'Choose a new output directory.' }

powershell.exe -NoLogo -NoProfile -NonInteractive -ExecutionPolicy Bypass `
  -File scripts\build_release_bundle.ps1 `
  -Treeish $commit `
  -OutputDirectory $output
```

Do **not** add `-PublishRelease`: the protected publication toolchain is an
external blocker. This command creates an unsigned, explicitly `UNVERIFIED`
directory containing the launcher, its adjacent expanded source tree, source
ZIP, wheel, SBOM, notices, build manifest, and checksums. It neither loads model
weights nor runs the EXE and it does not certify GPU, VRAM, quality, or release
evidence. `CogniBoard-v0.4.1.exe` is still a bootstrapper, not a standalone
model executable.

To compile only the low-level launcher during development, use:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\build_windows_launcher.ps1 `
  -OutputPath release\CogniBoard-v0.4.1.exe
```

The launcher is a console-free bootstrapper, not a standalone model bundle. It
requires this source tree, local Python/CUDA dependencies, and the verified
model plus manifest. The binary is not code-signed in this research release.
The frozen release builder records that unsigned state and emits
`SBOM.cdx.json`, `THIRD_PARTY_NOTICES.md`, `BUILD_MANIFEST.txt`, and
`SHA256SUMS.txt`; these inventories do not replace independent license review.
Do not use older release assets as v0.4.1 evidence.

Korean operator documentation: [`docs/COGNIBOARD_USER_MANUAL_PLAYBOOK_KO.md`](docs/COGNIBOARD_USER_MANUAL_PLAYBOOK_KO.md).

The v0.4.1 release notes and validation addendum are
[`release/COGNI_OS_0.4.1_RELEASE_NOTES_KO.md`](release/COGNI_OS_0.4.1_RELEASE_NOTES_KO.md)
and
[`release/COGNI_OS_0.4.1_VALIDATION_ADDENDUM_KO.md`](release/COGNI_OS_0.4.1_VALIDATION_ADDENDUM_KO.md).

## Runtime boundaries

- One spawned worker owns the model and CUDA. Model IPC is bounded,
  tensor-only and bound to a lease epoch, job, deadline, artifact and session.
- Gemma decoding uses `use_cache=False`; CTS does not accumulate a KV cache.
- Solver history and active CTS search state are fixed-capacity. This is not an
  O(1) claim for model weights, logs, expert banks, or unbounded external data.
- Local tasks execute only typed, allowlisted operations. Natural language is
  never passed to a shell.
- AFlow can only create a bounded research archive.
- The shipped UI/default Self-Harness stores inert proposals and exposes a
  read-only diff review with no approve/apply endpoint. An internal Linux
  ATTESTED library path can evaluate a snapshot and, only after external
  Ed25519 authority, atomically promote or roll back one exact committed
  record. An operator-only append-only validator now checks the complete
  promotion/health/signed-rollback evidence chain, but no independent
  production-runner statement or current raw production E2E evidence has been
  accepted. The path is not automatic or production-attested.
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
