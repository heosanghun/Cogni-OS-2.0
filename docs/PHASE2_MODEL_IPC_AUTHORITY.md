# Phase 2 — Model IPC authority and conversation isolation

## Production boundary

The resident Gemma worker continues to accept exactly four contiguous CPU
`torch.int64` tensors. Protocol v3 adds a closed authority schema without
introducing JSON, Python objects, or natural-language IPC:

- monotonic `request_id` and bounded `job_id`;
- immutable GPU lease epoch and lease deadline;
- hard per-request monotonic deadline;
- 32-byte verified model-manifest digest;
- 32-byte SHA-256 conversation digest;
- bounded input, mask, stop-token and response tensors.

The worker echoes the complete authority on every response frame. The
controller validates it before buffering tokens. Tokens remain private until a
valid terminal frame, so stale or corrupted authority cannot publish a partial
answer.

## Fail-closed rules

1. Product startup verifies the artifact manifest in the controller.
2. The spawned worker verifies the same manifest again before model load.
3. The lease epoch/deadline and artifact digest are sealed into a CPU tensor
   after lease acquisition and before Windows `spawn` serialization.
4. Every request must match the launch authority, have fresh request/job IDs,
   and remain inside both request and lease deadlines.
5. `CoreTurnRequest` rechecks the deadline before and after the authoritative
   CTS/DEQ solve.
6. Any mismatch returns `STATUS_AUTHORITY_REJECTED`, publishes no token, and
   terminates the resident worker.

## System 4 session isolation

`AgentManager.session_id` is hashed before IPC. The worker converts only that
digest to a bounded 64-character runtime key and passes it to
`GenesisRuntime.adapt_stream`. This removes the former hard-coded `"primary"`
key while preserving tensor-only IPC and activates the existing TTL/LRU/byte-
bounded `SwarmSessionStateCache` independently for each conversation.

## Verification

- conversation sequence A → B → A reaches the worker as stable A/B keys, with
  no cross-talk;
- System 4 cache tests prove per-session state, warm reuse, TTL expiry, LRU
  eviction, copy isolation and byte bounds;
- stale lease epoch, expired deadline and altered artifact digest fault
  injection are rejected before Core execution;
- a response-side digest alteration is rejected by the controller;
- focused result: `104 passed, 36 subtests passed`; Ruff clean.
