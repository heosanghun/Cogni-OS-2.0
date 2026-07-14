# Phase 8 — bounded System 3 experts

Phase 8 fixes the product profile at eight preallocated expert slots and
top-k=2 dispatch.  Recruitment never allocates another parameter tensor.
`ExpertCandidateLifecycle` is deliberately a night/evolution-plane component;
the day pipeline continues to consume System 3 as detached advisory telemetry.

## Safety invariants

- `route(x)` has no `z` argument and `assert_z_independent` checks an exact zero
  autograd derivative with respect to a probe fixed-point state.
- Padding when fewer than two experts are active repeats an active index with
  zero weight.  Inactive and quarantined matrices are never gathered and have
  zero forward contribution and gradient.
- The product profile requires `max_experts=8`, `top_k=2`, and a spectral margin
  strictly below 0.95.
- Novelty is marked unverified by default.  A labelled ID/OOD held-out set must
  produce a threshold satisfying both declared FPR and FNR bounds.
- The router and prototypes are frozen and content-digested before training.
  Candidate-gradient hooks isolate a single preallocated slice, and a
  post-training comparison rejects optimizers that changed another slice.
- Routed Fisher tensors are finite, non-negative CPU FP32 values.  Sparse
  FP-EWC snapshots merge under fixed domain and byte caps.
- Promotion is strictly ordered: inactive → C-FIRE → trained → held-out and
  non-collapsed → Fisher → canary → active.  Any failed gate atomically restores
  the previous bank and Fisher snapshots, then quarantines the failed slot.
- C-FIRE, merge, prune, or route-set changes revoke affected answer-authority
  eligibility.  A matching independent verifier attestation is required to set
  an authority bit, and the current application still does not consume that bit.
- Stable checkpoints are atomic and SHA-256 verified.  Authority bits without
  matching verifier attestations are rejected on write and restore.

## Evidence boundary

The repository contains deterministic safety fixtures, not a claimed trained
expert checkpoint, production novelty calibration, BWT/FWT benchmark, or
independent verifier result.  Consequently System 3 remains advisory by
default.  Real Gemma/System-4 ablations and long-domain raw results must be
attached by the external validation workflow before any product capability
record may be upgraded.

## Runtime integration

The factory should call `experts.assert_phase8_profile()` and construct one
`ExpertCandidateLifecycle(experts)` in the evolution control plane.  Day
inference calls `experts(...)` only for detached telemetry.  Evolution calls
the lifecycle stages under the existing day/night exclusion and VRAM guard.
No decode or logits path should inspect `answer_authority_mask` until the
external verifier integration is separately reviewed.
