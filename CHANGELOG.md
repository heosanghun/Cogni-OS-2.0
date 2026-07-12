# Changelog

## 0.3.0 — 2026-07-12

### Runtime truth and evidence

- added manifest-derived Runtime Fact-book identity and exact safetensors
  inventory so the agent cannot invent a 7B parameter count;
- separated service readiness, capability authority and evidence class;
- added content-addressed `EvidenceRecordV1` records scoped to exact model,
  code, configuration and device digests;
- added external Fact-book snapshots with an atomic last-known-good pointer,
  stale-scope invalidation and strict evidence JSON schema.

### Conversation and causal Cogni-Core

- added response-quality checks for token/sentence cycles, low-information
  repetition, role/control leakage, false completion and incomplete Korean;
- limited recovery to one bounded regeneration and prevented failed turns from
  being committed;
- upgraded model IPC to job/lease epoch/deadline/artifact/session-bound v3
  frames with stale/late/cross-session rejection;
- replaced telemetry-only CTS output with a bounded causal logits conditioner;
- implemented CTS V2 with a fixed 301-node arena, rank-16 solver history,
  bounded retrieval, separate policy/critic surfaces, explicit failed-edge
  telemetry and ACT/MAC budget;
- preserved the frozen verified local Gemma as answer authority while the
  untrained causal reasoner remains a canary.

### Systems 1.5, 2.5, 3 and 4

- made Fast Weight activation require a trained checkpoint, AQ/OOD, quality,
  composed-norm and session admission gates; no default product overlay is
  admitted;
- added bounded empirical Fisher, FP-EWC, C-FIRE and atomic evolution
  generation checkpoints;
- implemented an exact 28-agent System 4 profile, topology certificates,
  global norm/spectral checks, session state and PCAS stress telemetry;
- implemented the bounded eight-slot, top-k=2 System 3 candidate lifecycle with
  calibration, candidate-only training, held-out/Fisher/canary stages,
  quarantine, checkpoint rollback and independent authority separation;
- retained System 3 and System 4 as advisory because trained/calibrated product
  artifacts and independent quality evidence are not bundled.

### Local tasks, AFlow and Self-Harness

- replaced direct natural-language execution with immutable typed task plans,
  exact risk tiers, one-use capabilities, resource/path budgets and artifact
  SHA-256 read-back;
- allowed bounded T0/T1 operations, restricted T2 to inert proposal staging,
  and permanently denied T3/network/arbitrary shell authority;
- added a sealed, replayable AFlow research executor with six typed operators,
  bounded DAG/search/archive, repeated held-in/out metrics and
  `research_archive_only` output;
- added an evidence-linked proposal-only Self-Harness with persistent success
  and failure records, exact causal signatures, K≥3 distinct candidates,
  immutable control surfaces, negative archive and zero active-source mutation;
- labelled previous candidate-execution/replacement primitives as research
  components, not v0.3.0 product authority.

### Release and UI

- aligned runtime/package/launcher metadata at version 0.3.0;
- changed UI wording to expose `gated`, `advisory`, `night_only`, `research` and
  `proposal_only` states instead of implying every loaded module is active;
- documented the console-free launcher as a source-tree bootstrapper rather
  than a standalone signed appliance;
- replaced historical fixed test counts and old GPU snapshots with scoped,
  reproducible validation commands and evidence requirements.

## Before 0.3.0 — historical research prototypes

Earlier local artifacts demonstrated initial DEQ/CTS primitives, the graphical
mission-control shell, bounded chat and safety experiments. They are retained
only for repository history. Their embedded metrics, capability labels and
launcher metadata are not valid v0.3.0 evidence and must not be distributed as
the current release.
