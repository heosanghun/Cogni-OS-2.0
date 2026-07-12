# Local Gemma 4 E4B Validation — v0.3.2

## Scope

This record describes one local developer canary, not a universal performance
claim. The artifact was loaded from `C:\Project\cognios\gemma4-e4b` only after
all six entries in `config/gemma4-e4b.manifest.toml` passed SHA-256
verification. No model weight is distributed in this repository.

The Runtime Fact-book derives model architecture and parameter inventory from
the verified config and bounded safetensors headers. The model is not allowed
to invent a marketing size or identify itself as a 7B model. Stored and
effective parameter counts are separate facts and must come from the current
artifact inventory.

## Reproduction

```powershell
python scripts\validate_gemma4_runtime.py `
  --model C:\Project\cognios\gemma4-e4b `
  --manifest config\gemma4-e4b.manifest.toml `
  --event-stream
```

The command is local-only and validates finite tensors, the CTS/DEQ safety
contract, causal conditioning telemetry and the 16.7 GiB allocated-memory
postcondition.

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

```powershell
python scripts\validate_agent_completion.py `
  --model C:\Project\cognios\gemma4-e4b `
  --manifest config\gemma4-e4b.manifest.toml
```

The current script runs four offline turns and checks clean terminal reasons,
no truncation, no public role/control markers, no repetitive output, truthful
identity grounding and worker cleanup. It is a regression canary, not the
Phase 1 requirement for an independently reviewed 20-turn corpus.

Natural Korean conversation is a separate mandatory gate:

```powershell
python scripts\validate_agent_casual_korean.py `
  --model C:\Project\cognios\gemma4-e4b `
  --manifest config\gemma4-e4b.manifest.toml `
  --timeout 120 `
  --output C:\Project\cognios-evidence\casual-korean-v0.3.2.json
```

This reproduces the reported two-turn collaboration conversation verbatim and
adds independent greetings, paraphrases, typos, follow-ups, a context switch,
and formal regressions. Release requires zero quality fallbacks, exactly one
assistant message per user turn, natural Korean completion, no loops, and a
bounded per-turn latency. See [CASUAL_KOREAN_VALIDATION.md](CASUAL_KOREAN_VALIDATION.md).

## Experimental decoder-DEQ smoke

```powershell
python scripts\validate_gemma4_deq.py `
  --model C:\Project\cognios\gemma4-e4b `
  --manifest config\gemma4-e4b.manifest.toml `
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
