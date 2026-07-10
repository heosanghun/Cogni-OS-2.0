# Owner Repository Audit

On 2026-07-10 the public GitHub account `heosanghun` exposed 55 repositories at the final refresh. All repository names, languages, sizes, and update timestamps were enumerated through the GitHub API. The repositories directly relevant to Cogni-OS were then cloned into `work/upstream` for source-level comparison.

## Directly reviewed

- `System1.5` and `System1.5_260515`
- `System2.5`
- `System3`, `System3.5`
- `System4`, `System5`
- `Cognitive-Tree-Search` and `Cognitive-Tree-Search-2-`
- `BIO-HAMA_MAIN`
- `AkasicDB`

## Design decisions taken from the audit

- Large-state CTS uses a bounded multisecant/Anderson path instead of a dense Jacobian.
- PUCT statistics and latent storage have hard capacities.
- Fast Weight overlays remain low-rank and session-scoped.
- The latest System1.5 audit reports that PoC Fast Weight accuracy restoration is not established at scale. Cogni-OS therefore requires an empirical quality gate and never enables a generated overlay merely because it is finite.
- System2.5 keeps FP-EWC matrix-free and projects recurrent operators below the Banach margin.
- System3 expert growth must be capped; dynamic spawning without a fixed pool contradicts the total-memory budget.
- System4 avoids `.item()` synchronization in its PCAS hot path and selects between precompiled topologies with tensor operations.
- AkasicDB-style persistent knowledge belongs to the control/storage plane, not the Cogni-Core tensor hot path.

Repositories unrelated to the Cogni-OS architecture—websites, trading
applications, teleoperation, and UI projects—were inventoried but not copied
into runtime code. Reference checkouts are excluded from imports and deployment
artifacts. The complete 55-repository inventory is included in the release
outputs.
