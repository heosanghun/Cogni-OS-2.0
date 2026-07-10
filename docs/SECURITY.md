# Security and Safety Model

## Enforced in this repository

- local-path-only model loading;
- offline environment flags;
- no network-capable imports in candidate patches;
- no `eval`, `exec`, `compile`, or dynamic imports in candidate patches;
- mutable-surface allowlist and traversal protection;
- base-file digest check to prevent stale overwrites;
- separate staging tree for validation;
- regression gate before atomic promotion;
- day/night mutual exclusion;
- explicit rollback and safe-mode states;
- VRAM admission and peak postcondition checks;
- spectral projection and non-contractive DEQ fallback or rejection;
- process-only sandbox runners are rejected before candidate execution;
- evolution task lifetime counters prevent inference resumption during proposal,
  validation, promotion, or workflow search.

## Phase 4 control-plane boundaries

### Failure capture daemon

`FailureCaptureDaemon` admits workflow exceptions and timeouts to a fixed-size
in-memory queue and writes them to the local SQLite `LogDB` on one daemon
thread. Queue admission is non-blocking. Capacity exhaustion is audited and
raises `FailureQueueOverflow`; it is never reported as a successful capture.
The context manager drains accepted records and joins the thread on exit. A
database writer error is audited and is raised by `flush`, future submissions,
and `stop`.

The queue is deliberately bounded but not a durable message broker. A process
or machine crash can lose records admitted but not yet committed to SQLite.
Deployments requiring crash durability must put an OS-journaled local spool in
front of `LogDB`; they must not replace it with a network queue. Audit insertion
is attempted on every overflow or writer failure, but a broken SQLite store can
prevent both the primary record and its audit record. The fail-closed exception
remains observable in that case.

### Idle/night scheduling

`IdleNightScheduler` uses an injected monotonic clock and exposes only explicit
`tick`/`run_once` calls. It does not create an unbounded sleep loop, infer wall
clock time zones, or query an OS idle service. The host must call
`note_activity` at completed inference/request boundaries and invoke `tick` at
a bounded cadence. A non-blocking cycle lock prevents overlapping night
cycles. The scheduler checks `RhythmController` state and active requests, then
delegates all legal mode transitions to `SelfHarness` or
`WorkflowEvolutionCoordinator`; those components re-check the day/night gate.

### Local model patch proposals

`LocalGemmaPatchProposer` accepts already-loaded model and tokenizer objects;
it has no model-path loader and no network client. The host is responsible for
injecting a trusted, hash-verified local Gemma instance. The proposer forces
`use_cache=False`, deterministic decoding, and hard input/output token caps.
An external trusted resolver—not model output—selects the relative path,
current source, and base SHA-256. The model supplies only full replacement
text. Markdown fences are removed only when they wrap the entire response;
mixed explanation/code responses are rejected.

Generation is not execution. The proposer performs an early static
`PatchPolicy` check and returns an inert `PatchProposal`. `SafeHarnessPatcher`
must repeat policy and digest checks, execute regression tests only inside a
kernel-isolated sandbox, and atomically promote a passing candidate. Arbitrary
objects injected as `model`, `tokenizer`, or target resolver are inside the
trusted host boundary; this class cannot prove that a malicious injected
object is offline.

## Deployment responsibility

`SubprocessSandbox` is retained only as a development diagnostic runner and is
**never accepted by `SafeHarnessPatcher`**. Candidate execution and promotion
fail closed unless an injected `SandboxRunner` explicitly represents a
kernel-isolated offline VM, Windows Sandbox, container runtime, or equivalent
filesystem/network boundary. The marker is a deployment trust boundary; its
implementation must be independently audited.

Model artifacts should be mirrored into an offline store and accompanied by hashes, licenses, tokenizer files, configuration, and provenance. No runtime component is permitted to download missing artifacts.

### Tensor service deployment

The local tensor service has no network socket and uses a bounded numeric/tensor
protocol. For a real CUDA deployment, the model factory must load the verified
artifact **inside the single worker**. Constructing a GPU model in the parent and
passing it into a Windows-spawn worker can duplicate VRAM and violates the
single-owner invariant. Current automated service tests use a CPU toy module;
the real Gemma model was measured through the in-process runtime gate to avoid
creating a second GPU owner.
