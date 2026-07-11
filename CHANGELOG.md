# Changelog

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
