# Local Gemma 4 E4B Base Canary Validation — v0.3.2

## Scope

This record describes one historical local **pretrained-base research canary**,
not a universal performance claim or the public conversation runtime. The
artifact was loaded from `C:\Project\cognios\gemma4-e4b` only after
all six entries in `config/gemma4-e4b.manifest.toml` passed SHA-256
verification. No model weight is distributed in this repository.

The Runtime Fact-book derives model architecture and parameter inventory from
the verified config and bounded safetensors headers. The model is not allowed
to invent a marketing size or identify itself as a 7B model. Stored and
effective parameter counts are separate facts and must come from the current
artifact inventory.

## Reproduction

```bash
/usr/bin/python3 -I -B \
  -X pycache_prefix=/home/shoon/.cognios-gpu5-guard/host-never-pycache \
  scripts/gpu5_boundary_guard.py run \
  --image cogni-os-dev@sha256:20aaf1d7cde8d6a504ba08f158a34a1907eac9413f3578acc4637f0a1b2ec8ba \
  --expected-source-commit "$(git rev-parse HEAD)" \
  --workdir /workspace \
  --timeout 1800 \
  --evidence-filename gpu5-base-canary-v041.json \
  -- -I -B /workspace/scripts/validate_gemma4_runtime.py \
  --model /models/gemma4-e4b \
  --manifest /workspace/config/gemma4-e4b.manifest.toml \
  --physical-gpu-index 5 \
  --gpu-query-context gpu5-container \
  --event-stream
```

This is the sole Stage G GPU evidence entry point after the clean exact-commit
CPU Exit Gate.  It snapshots the named commit, uses the pinned image and UUID
container selector, and validates finite tensors, the CTS/DEQ safety contract,
causal conditioning telemetry and the 16.7 GiB allocated-memory postcondition.
The ordinary Windows CogniBoard launch supports only `desktop-ui-only`; callers
cannot select the native GPU server profile, and it does not claim this server
evidence capability.

Every `run` command is additionally blocked before Docker and before the first
GPU query unless the laboratory scheduler has issued
`/run/cognios-lab-scheduler/gpu5-reservation.json`. The artifact must be a
root-owned, non-symlink, read-only regular file with the closed schema
`cogni.lab.gpu5.reservation.v1`. It binds status `reserved`, physical index 5,
the pinned GPU5 UUID, the exact source commit, the launching effective UID, a
bounded reservation identifier, and an unexpired window of at most 24 hours.
The remaining window must cover the configured run timeout plus five minutes
for fail-closed cleanup. A project-local lock cannot exclude unrelated
laboratory jobs, so it is not a substitute for this scheduler reservation.
Until the external scheduler creates that artifact, Stage G is
`EXTERNAL_BLOCKER / NOT RUN`; operators must not hand-create or weaken the
reservation file to make a test pass.

## Latest scoped canary observation

Recorded on 2026-07-12 on the attached NVIDIA RTX 5090 Laptop GPU:

| Field | Observation |
|---|---:|
| model class | `Gemma4ForConditionalGeneration` |
| hidden size | 2,560 |
| manifest files | 6 verified |
| load time | 25.698 s |
| integrated inference time | 18.583 s |
| requested/reached CTS depth | 100 / 100 |
| arena nodes used/capacity | 300 / 301 |
| fixed search allocation | 57,243,710 bytes |
| final transition residual | 0.0002298355 |
| transition converged | yes |
| transition fallback | not used |
| solver history rank | 16 |
| failed edges / zero-Q backups | 2 / 2 |
| unsafe silent fallbacks | 0 |
| linear fallback use | 0 |
| ACT steps | 301 |
| MAC budget | 3,209,427,616 |
| peak allocated VRAM | 14.8468875885 GiB |
| causal bridge answer-bearing | yes |
| causal logits bias non-zero | yes |
| maximum absolute bias | 0.0498046875 |
| generated-token canary | 1 token |
| trace digest | `19f6a32f0977be3feb7f5a018771489c8284d97b210b0277be8bac354e028b68` |

The two failed search edges were observable and backed up with zero value; they
were not silently counted as successful transitions. The causal bridge changed
the decode logits through a bounded bias while the base Gemma parameters
remained frozen.

This table is a scoped developer observation. The raw JSONL/stdout, environment
versions, code/config/device digests and artifact hash must accompany a release
evidence record. This Markdown file alone does not authorize a `measured`
claim.

## Conversation completion

Product conversation validation uses only the exact pinned instruction-tuned
checkpoint and its seven-file manifest. The pretrained base artifact above is
not an accepted public conversation checkpoint.

The historical Windows command is not a release command. It can select the
desktop's default CUDA device and therefore must not be used on the laboratory
server. Stage G now has two immutable artifact profiles: `base-canary` retains
the pretrained six-file research checkpoint, while `product-e4b-it` seals the
distinct seven-file instruction checkpoint at `/models/gemma4-e4b-it`. The
profile is bound into the Docker labels, source/model execution scope, result
and evidence. Cross-profile model, manifest and validator combinations fail
before GPU preflight. This implementation has not been run on the laboratory
GPU in this change, so its current evidence status is `NOT RUN`, not `PASS`.
Stage G product-completion gate is `EXTERNAL_BLOCKER / NOT RUN` until the lab
scheduler reserves physical GPU 5, the exact command below finishes inside that
reservation, and its standalone JSON component passes the guard.

The sole product release command is the integrated 20-turn acceptance suite:

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

The release guard accepts exactly 20 turns and one standalone strict UTF-8 JSON
component. It rejects progress-text/JSON mixtures, duplicate object keys and
non-finite values. The guard independently recomputes the fixed case order,
the canonical prompt text at every index, Fact-book/model routes, A/B session
isolation, answer digests, raw Korean completion, repetition, role/control-token
leakage, topic anchors, exact quality-check schema, one resident PID, post-turn
GPU samples, model-manifest binding, GPU5 identity before/after and worker/lease
cleanup before it releases the host GPU5 lease. Each user turn must own exactly
one new assistant message. Turn 9 is a deliberate low-first-budget continuation
probe and must use exactly one bounded continuation; every other turn must use
zero. In particular, the identity turn must contain exactly one occurrence of
the manifest-bound `gemma4-e4b-it` label, build `0.4.1`, effective parameter
count `4,506,496,490` and stored parameter count `7,996,157,418`; a self-asserted
summary cannot substitute for those raw turn records.

The recommended release run executes 20 offline turns and checks clean terminal
reasons,
no truncation, no public role/control markers, no repetitive output, truthful
identity grounding and worker cleanup. It is release regression evidence, not
a substitute for independently reviewed task-quality evaluation.

The completion report schema is `cogni.agent.completion.stress.v2`. Its memory
coverage consists of exactly one point-in-time sample after each turn for which
the resident worker is expected. The maximum of those samples is still not a
runtime peak: this validator does not certify within-turn peaks, sustained
usage, or whole-runtime VRAM release. Peak VRAM evidence belongs to
`validate_gemma4_runtime.py`, which records
`torch.cuda.max_memory_allocated` telemetry. The strict completion stress gate
requires complete post-turn coverage and fails if any observed sample exceeds
16.7 GiB. A non-exceeding spot sample remains a point observation and cannot
promote the separate peak-VRAM gate.

Natural Korean conversation is part of the only executable product Stage G
path: the integrated exact-20 suite above.  The standalone 10-turn
`validate_agent_casual_korean.py` harness remains source-level diagnostic code
and is deliberately rejected by guarded `run`; `docker-argv` inspection does
not execute it and cannot create evidence.  This distinction prevents a
smaller diagnostic corpus from being presented as product release evidence.

The standalone harness still enforces explicit physical index 5 and query
context, pre-access guarded identity, and manifest plus identity postchecks for
future guarded integration.  Its scenarios reproduce the reported two-turn
collaboration conversation and cover greetings, paraphrases, typos, follow-ups,
context switching, and formal regressions.  Those behavioral categories are
included in the executable exact-20 product suite, which additionally requires
zero quality fallbacks, one assistant message per user turn, natural Korean
completion, no loops, and bounded per-turn latency.  See
[CASUAL_KOREAN_VALIDATION.md](CASUAL_KOREAN_VALIDATION.md).

## Experimental decoder-DEQ smoke

This section intentionally retains the pretrained base checkpoint to reproduce
the historical research canary. It is not a product conversation command.

```bash
/usr/bin/python3 -I -B \
  -X pycache_prefix=/home/shoon/.cognios-gpu5-guard/host-never-pycache \
  scripts/gpu5_boundary_guard.py run \
  --image cogni-os-dev@sha256:20aaf1d7cde8d6a504ba08f158a34a1907eac9413f3578acc4637f0a1b2ec8ba \
  --expected-source-commit "$(git rev-parse HEAD)" \
  --workdir /workspace \
  --timeout 1800 \
  --evidence-filename gpu5-deq-experimental-v041.jsonl \
  -- -I -B /workspace/scripts/validate_gemma4_deq.py \
  --model /models/gemma4-e4b \
  --manifest /workspace/config/gemma4-e4b.manifest.toml \
  --physical-gpu-index 5 \
  --gpu-query-context gpu5-container \
  --allow-uncertified-experimental
```

This experiment injects a bounded last-layer delta map and observes convergence
for the selected input. It does **not** prove global contraction of the full
Gemma decoder. Production use without the experimental switch requires an
independent decoder-delta Lipschitz certificate and fails closed when that
certificate is missing or reaches the configured safety margin.

The product canary uses a bounded post-backbone equilibrium reasoner and causal
logits conditioner. No trained DEQ adapter/Wproj artifact or independent
held-out quality report is bundled; the CTS/DEQ capability therefore remains
`canary`, not `authoritative`.

## Claims this run does not support

- RTX 4090 behavior or the 16.7 GiB envelope on that target;
- peak reserved memory or every prompt/context/software stack;
- a trained Gemma-DEQ quality improvement;
- a globally contractive full decoder;
- trained/admitted System 1.5, System 3 or PCAS artifacts;
- production air-gap packet audit;
- AGI, unlimited reasoning depth, or O(1) total system memory.

Repeat the exact gate on the target RTX 4090 with the final release artifact,
driver, CUDA, PyTorch, model manifest and configuration before changing any
target or capability state.
