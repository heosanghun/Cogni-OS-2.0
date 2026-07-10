# Local Gemma 4 Integration

The local artifact at `C:\Project\cognios\gemma4-e4b` was verified against `config/gemma4-e4b.manifest.toml`. Runtime loading is local-only, rejects Hub IDs and URLs, disables remote code, and forces offline flags.

Run the real-model gate in an environment containing the optional `gemma` dependencies:

```powershell
python scripts/validate_gemma4.py `
  --model C:\Project\cognios\gemma4-e4b `
  --manifest config\gemma4-e4b.manifest.toml
```

The local artifact does not currently include a global Lipschitz certificate for
the Gemma decoder delta branch. Therefore the reproduced run is explicitly an
**uncertified convergence smoke test**, not a production contractivity gate:

```powershell
python scripts/validate_gemma4_deq.py `
  --model C:\Project\cognios\gemma4-e4b `
  --manifest config\gemma4-e4b.manifest.toml `
  --allow-uncertified-experimental
```

A production run omits that override and supplies an independently derived
`--certified-delta-lipschitz-bound`. The adapter rejects a missing or unsafe
certificate before executing the fixed-point map.

Measured on the attached RTX 5090 Laptop GPU:

- model: `Gemma4ForConditionalGeneration`
- decoder layers: 42
- BF16 checkpoint load: 25.981 s
- text forward: 0.823 s
- peak allocated VRAM: 14.8955 GiB
- output: finite logits, shape `(1, 8, 262144)`

Real last-layer DEQ injection was also executed experimentally with
`contractive_delta_scale=0.05` and normalized tolerance `5e-3`:

- converged: yes
- solver iterations: 4
- normalized residual: `0.0018844604`
- forward latency: 0.975 s for the smoke prompt
- peak allocated VRAM: 14.8955 GiB
- output: finite logits, shape `(1, 10, 262144)`
- contractivity certificate: **not available; no global `L <= 0.95` claim**

## Integrated depth-100 gate

The full local runtime gate loads Gemma plus the bounded System 3/4 and
BIO-HAMA auxiliary modules, then runs a certified limited-Broyden latent CTS
transition through a fixed 301-node arena:

```powershell
python scripts/validate_gemma4_runtime.py `
  --model C:\Project\cognios\gemma4-e4b `
  --manifest config\gemma4-e4b.manifest.toml
```

Measured result on the same GPU:

- Gemma hidden size: 2560
- requested/reached CTS depth: 100/100
- tree nodes: 301/301
- fixed search allocation: 19,617,369 bytes
- integrated inference: 4.320 s
- final transition residual: `0.00390625`
- transition fallback: not used
- integrated inference peak: 14.8604 GiB
- finite backbone and search latents: yes

The process-wide model-load peak remains 14.8955 GiB. This gate proves the
measured artifact and bounded CTS working set fit on the attached RTX 5090
Laptop GPU. The target RTX 4090 must still repeat the same command because
allocator and kernel workspaces are hardware/software dependent.

The integrated gate uses a globally contractive small latent Broyden map after
the backbone. It does not turn the entire Gemma decoder into a globally
certified DEQ; that separate production claim remains blocked on a valid
decoder-delta Lipschitz certificate.

## DEQ injection rule

A stock Gemma residual block must not be used directly as a fixed-point map: its identity residual violates the strict Banach assumption. `GemmaDEQBackboneAdapter` therefore offers `contractive_delta_scale`, which removes the identity residual, anchors the update at the explicit lower-stack state, and scales only the residual update branch. Scaling alone is **not** a proof of contraction. Fail-closed operation additionally requires a certified global upper bound for the delta branch and verifies that the scaled bound remains below the configured margin.

If a real-model solve misses its configured normalized residual tolerance, the system records fallback use and must route to the explicit solver path. It must never silently label a finite but unconverged state as a valid DEQ equilibrium.
