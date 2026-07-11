# Changelog

## 0.2.1 — 2026-07-11

- replaced the missing-template transcript fallback with a pinned text-only
  Gemma 4 turn contract and exactly one BOS token;
- added EOS/EOT/tool and reserved-token quarantine stops, structured public
  response parsing, and protocol-v2 terminal finish reasons;
- increased the bounded answer envelope to 512 tokens per request and 1,536
  tokens per turn, with up to two same-turn continuations only after `length`;
- changed generation timeout handling from one absolute deadline to a resettable
  worker-idle deadline;
- batched streaming renders, flushed empty terminal frames, and exposed
  completion/truncation metadata plus an explicit continue action in CogniBoard;
- pooled the advisory Gemma embedding to a fixed-size CTS root so conversation
  length no longer multiplies the 301-node arena allocation;
- added a real-GPU three-turn completion validator. On the attached RTX 5090
  Laptop GPU all three turns ended with `stop`, no truncation, no role leakage,
  no public control tokens, and a clean worker shutdown.

## 0.2.0 — 2026-07-11

- added the CogniBoard local AI workspace and native Windows launcher;
- added bounded multi-turn conversation and streamed token-tensor responses;
- integrated the resident Gemma worker with BIO-HAMA, DEQ/CTS, System 4, and
  System 3 advisory execution;
- kept unverified Fast Weight overlays gated and FP-EWC evolution-only;
- added typed local workspace tools without arbitrary shell or network access;
- added exact agent/evolution HTTP APIs and single-compute ownership arbitration;
- added production Self-Harness composition with bounded failure storage,
  proposal-only default, runner attestation, journaled atomic promotion, health
  checking, and digest-verified rollback;
- added model manifest verification before the chat path can load Gemma;
- added the full GPU agent validation script and Korean operator guide;
- expanded security, mixed-precision, cancellation, UI accessibility, and
  responsive-layout regression coverage.

Internal verification on an RTX 5090 Laptop GPU completed the full
Cogni-Core-to-Gemma path with a 15.397 GiB observed GPU-memory increase over the
WDDM baseline, then returned to baseline after worker cleanup. This measurement
is not an RTX 4090 certification or an AGI claim.

## 0.1.0 — 2026-07-11

- initial bounded DEQ/CTS, Fast Weight, FP-EWC, System 3/4, BIO-HAMA, AFlow,
  Self-Harness research runtime, validation CLI, and mission-control demo.
