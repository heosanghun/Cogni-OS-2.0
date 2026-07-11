# Cogni-OS 2.0 Architecture

## Trust boundaries

The system is split into two processes or deployment units:

- **Cogni-Core** owns tensor inference, DEQ solves, CTS, session overlays, FP-EWC statistics, and PCAS.
- **Cogni-Flow** owns state transitions, logs, candidate generation, sandbox validation, promotion, and rollback.

`TensorService` provides the concrete local process boundary. A single spawned
Cogni-Core worker owns the GPU model; Cogni-Flow remains CPU-side. Every data
plane request and response is exactly four CPU tensors carrying numeric opcode,
request id, payload/result, and status. The bounded queues carry no JSON,
strings, exception objects, or network traffic. PAUSE/RESUME opcodes align the
worker with the day/night controller.

The local generation protocol is version 2. Its fourth response tensor carries
a closed terminal reason (`stop`, `length`, `cancelled`, or `error`), so a
token-budget boundary cannot be committed as a successful model stop.

Natural language, source code, paths, JSON, and subprocess results never enter the Cogni-Core hot path. Core modules exchange tensors and fixed Python dataclasses whose fields are tensors or bounded numeric metadata. Cogni-Flow may use structured control data because it is outside the sub-millisecond tensor path.

## Day path

1. `RhythmController.inference_slot()` prevents evolution from starting.
2. `VRAMGuard` performs admission control against the configured 16.7 GiB envelope.
3. One resident spawned worker owns both the verified local Gemma model and the
   Cogni-Core runtime. The browser and HTTP control plane never own CUDA.
4. A bounded multi-turn controller renders the prompt locally and sends only
   CPU `int64` tensors across the worker boundary.
5. BIO-HAMA routes the turn. The local Gemma embedding is attention-pooled to
   one fixed-size advisory latent before PUCT CTS plus DEQ performs
   fixed-width, bounded-node search; CTS arena storage does not scale with the
   conversation sequence length.
6. System 4 and System 3 produce detached advisory telemetry. They cannot alter
   Gemma logits or decoded answer tokens.
7. Deterministic Gemma decoding runs with `use_cache=False` and streams bounded
   token tensors back to the controller. Native Gemma 4 EOT and quarantined
   reserved markers are stopped before public decode.
8. A bounded latent bottleneck compiles only converged states into low-rank
   Fast Weight overlays. External held-out quality, composed operator norm, and
   Near-OOD gates must all pass; overlays remain session-only.
9. PCAS can select a precompiled topology without changing parameters.
10. Failures are appended to the local LogDB.

## Night path

1. The caller drains active requests, then inference enters `DRAINING`; new
   requests are rejected. A non-zero active count aborts the night transition.
2. State is checkpointed only after the zero-active-request check.
3. Failure traces are clustered by verifier-grounded signatures.
4. The bounded failure daemon persists exceptions/timeouts locally, and the
   idle scheduler permits at most one night cycle.
5. A hash-verified local model may emit replacement text, but a trusted resolver
   selects the target path and base digest.
6. Static air-gap and AST policy checks run before execution.
7. Each candidate is tested only by a kernel-isolated runner over a staging copy;
   process-only runners are rejected before execution.
8. By default the production assembly is proposal-only, so candidate source is
   never installed.
9. Promotion requires an operator-allowlisted runner attestation proving a
   separate kernel, no network, no host-filesystem access, an ephemeral
   workspace, and exact command digests.
10. A passing candidate is journaled, atomically promoted, health-checked in a
    fresh isolated snapshot, and digest-verified rollback is automatic on
    failure. Unknown live-file digests are never overwritten.

## Memory claims

The implementation claims bounded **solver history and active search state**
under fixed width, hidden size, history rank, and node capacity. `max_depth` does
not size any CTS tensor; a fixed arena and frontier rollout handle capacity
exhaustion. It does not claim constant total memory for model weights, expert
banks, log databases, or unbounded external data. Those resources have separate
capacities and eviction policies.

## Reference lineage

The implementation was compared with the owner's public repositories:

- `Cognitive-Tree-Search`
- `System1.5_260515`
- `System2.5`
- `System3`, `System3.5`
- `System4`, `System5`
- `BIO-HAMA_MAIN`

Runtime code does not import from the reference checkouts under `work/upstream`.
The complete repository review and the resulting constraints are recorded in
[`UPSTREAM_AUDIT.md`](UPSTREAM_AUDIT.md).
