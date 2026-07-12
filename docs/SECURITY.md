# Cogni-OS 2.0 v0.3.1 Security and Safety Model

## Release boundary

Version 0.3.1 is a local research runtime with a **proposal-only** evolution
boundary. It may observe failures, build evidence-linked candidates, and store
an inert proposal archive. It may not execute candidate source, replace active
source, or promote a proposal. Safe promotion belongs to a later phase and
requires independently attested kernel isolation.

## Enforced controls

### Model, network, and process

- model loading accepts a verified local filesystem artifact only;
- SHA-256 manifest verification precedes model use;
- Hub IDs, URLs, remote code, force-download, telemetry and external APIs are
  rejected or disabled;
- the HTTP server binds to loopback, validates Host/Origin, uses an HttpOnly
  per-process session token, restrictive CSP, exact routes, bounded bodies and
  a static-asset allowlist;
- one spawned worker owns the model and CUDA;
- IPC v3 is tensor-only and binds each request/response to job, lease epoch,
  deadlines, artifact digest and session digest;
- stale epochs, expired work, wrong artifacts, late replies and cross-session
  frames are rejected;
- decode is deterministic and cache-free, with bounded inputs, outputs and
  queues;
- the product factory rejects enabling Gemma last-known-good/base-only
  generation fallback; Cogni-Core terminal success and base-integrity checks
  are required before the response is atomically published, and Core failure
  fails closed.

### Numerical and memory safety

- VRAM admission and postcondition checks use the configured 16.7 GiB hard
  limit;
- CTS search uses a fixed arena and bounded solver history rather than a
  depth-sized tensor allocation;
- non-finite or unsafe DEQ transitions are explicitly rejected or recorded as
  failed edges; they are not relabelled as convergence;
- C-FIRE/spectral projection is applied around evolution updates and checkpoint
  restoration;
- Fast Weight activation requires a trained artifact plus AQ, OOD, composed
  norm, quality and session gates; the default product has no admitted overlay;
- System 3 and System 4 cannot alter current answer tokens while their
  capability state is `advisory`.

### Conversation and facts

- the Runtime Fact-book is generated from manifest-verified local artifact
  headers, not from the language model's self-description;
- capability state and answer/runtime-mutation authority are explicit fields;
- public self-description is grounded in the Fact-book, preventing fabricated
  parameter counts and capability promotion;
- role/control token leakage, token and sentence cycles, low-information
  repetition, false completion and incomplete Korean endings are checked before
  commit;
- a failed or cancelled turn is aborted transactionally;
- one bounded response-repair generation is permitted; it is not an unbounded
  retry loop and is not a Gemma base-only fallback;
- if response repair still fails, the manager may return a bounded static safe
  response instead of publishing partial, repetitive or Core-bypassing model
  output.

### Evidence and last-known-good

- `EvidenceRecordV1` is strict canonical JSON with a content-derived id;
- every record is scoped to exact model, code, runtime configuration and device
  digests;
- verified claims require present, valid, claim-bound evidence in the current
  scope; stale scope fails closed;
- raw JSONL journals and Fact-book snapshots live outside the source tree;
- journal and snapshot sizes/counts are bounded;
- the current Fact-book is selected through an atomic last-known-good pointer;
- malformed, missing, stale or digest-mismatched evidence cannot authorize a
  claim;
- the UI starts with runtime evidence `UNVERIFIED` and without copied metrics;
  only a successful validation run in the current process and scope may
  promote live evidence.

### Typed local tasks

- deterministic slash/safe grammar produces immutable typed plans;
- every plan has exact path, input digest, expected artifact, verifier and
  time/CPU/RAM/VRAM/output budgets;
- a short-lived capability is valid for one exact plan digest and is consumed
  on first use;
- T0 bounded reads and verified output-only artifact writes are allowed;
- the T1 fixed-test primitive is blocked by default and requires explicit
  trusted-host opt-in; it is not represented as OS-level process-tree/network
  sandbox attestation;
- T2 only stages an inert proposal; T3 is permanently denied;
- arbitrary shell, network, evaluator/security/updater mutation, absolute/UNC/
  device paths, traversal, ADS, reparse escapes and unsafe executables are not
  allowed;
- path identities are checked around access to reduce TOCTOU and junction/
  symlink attacks;
- saved artifacts are securely read back and verified by exact size and
  SHA-256.

### AFlow and Self-Harness

- AFlow accepts a closed six-operator schema with no source/shell/path payload;
- evaluator, policy and suite are sealed; bounded replay lineage and held-in/
  held-out repetitions are checked;
- AFlow's only success target is `research_archive_only`;
- the proposal ledger stores bounded success and failure records, exact causal
  signatures and at least three distinct candidates per eligible signature;
- restart hydration is strict and bounded: persisted successes, failures,
  proposals and negative records are reparsed, identity/cross-reference/digest
  invariants are revalidated, and malformed, oversized, unknown or tampered
  records are rejected rather than trusted;
- each candidate links primary evidence, expected behavior, risk, reproduction
  test and rollback trigger;
- exact replacement UTF-8 is stored only as a bounded content-addressed blob
  outside mutable source roots; restart hydration rechecks its digest, size,
  encoding, AST policy, target path and current base hash before review;
- security, evaluator, updater, rollback, manifest and other protected surfaces
  are immutable;
- stale base digests, forbidden AST, path escape and source mutation fail
  closed;
- rejected replacements are recorded in a negative archive and suppressed for
  the same active process;
- source hashes are compared before and after a proposal-only night cycle; any
  change enters safe mode.

## Day/night exclusion

The lifecycle authority blocks new inference admission, drains active work,
returns the GPU lease and checkpoints before evolution. Inference cannot resume
while FP-EWC, expert candidate work, AFlow evaluation or proposal generation is
active. A timeout, stale lease, non-zero active count or checkpoint failure
aborts the transition.

The bounded failure daemon and monotonic idle scheduler do not create an
unbounded background loop. Queue overflow, writer failure and overlapping
night cycles are observable errors. SQLite is local but is not a durable
message broker; crash-durable deployment requires an OS-journaled local spool.

## Trusted host boundary

Injected local model/tokenizer objects, target resolvers, evaluator callables,
device descriptors and artifact hashes must come from the trusted host. A
Python class cannot prove that a malicious injected object is offline or that
an external evaluator is honest.

## Deliberately unavailable in v0.3.1

- automatic source promotion or live-code replacement;
- a trusted kernel-isolated Windows Sandbox/VM/container attestation;
- network/host-filesystem escape evidence for a candidate runner;
- signed executable, SBOM, installer/update trust chain, or hardware-backed
  signing identity;
- RTX 4090 hardware certification and packet-level egress audit;
- an independent external human-labelled 20-turn conversation study;
- trained DEQ/Wproj, Fast Weight Programmer and System 3 expert artifacts;
- at least three-seed FP-EWC BWT/FWT results;
- production PCAS calibration and System 4 quality evidence;
- independent held-out quality/verifier results for advisory modules;
- production code signing and a signed installer/update trust chain.

`SubprocessSandbox` and any previous atomic-replacement primitives are
development/research components, not product authority. A class name, process
boundary or marker is never accepted as proof of kernel isolation.

## Deployment responsibility

Mirror the model, tokenizer, licenses, configuration and provenance into the
offline environment before launch. Do not allow runtime downloads. The v0.3.1
launcher is a bootstrapper that requires the source tree and dependencies; it
is not a signed standalone appliance. Repeat all VRAM, air-gap, conversation
and failure-injection gates on the exact deployment hardware and software
stack.
