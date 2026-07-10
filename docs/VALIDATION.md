# Validation Gates

Run all gates with:

```powershell
python -m unittest discover -s tests -v
```

The suite covers:

- fixed-point forward accuracy against a 250-step explicit unroll;
- IFT gradient accuracy;
- non-contractive hard stop and damped fallback;
- CUDA active-memory invariance from CTS depth 8 through 64;
- bounded PUCT and latent retrieval;
- requested-depth-independent search allocation and certified Broyden transition;
- Fast Weight operator-norm and session lifecycle;
- source/target-bounded Fast Weight compilation, external quality admission,
  composed norm, OOD calibration, and checkpoint round trip;
- matrix-free fixed-point Fisher and EWC anchoring;
- System4 PCAS topology switching without parameter changes;
- day/night exclusion;
- patch AST policy, regression validation, atomic promotion, and audit logging;
- bounded failure daemon, monotonic idle scheduler, and local-model inert proposer;
- Windows-spawn tensor-only service pause/resume, capacity, timeout, restart,
  and numeric error isolation;
- offline configuration and VRAM admission.

Passing toy and component gates is necessary but not sufficient evidence for Gemma-scale quality or a 16.7 GiB end-to-end envelope. The final backbone artifact must be profiled separately on its target GPU.

The verified local-backbone gates are deliberately separate:

```powershell
python scripts/validate_gemma4.py --model C:\Project\cognios\gemma4-e4b --manifest config\gemma4-e4b.manifest.toml
python scripts/validate_gemma4_deq.py --model C:\Project\cognios\gemma4-e4b --manifest config\gemma4-e4b.manifest.toml --allow-uncertified-experimental
python scripts/validate_gemma4_runtime.py --model C:\Project\cognios\gemma4-e4b --manifest config\gemma4-e4b.manifest.toml
```

The second command as shown is a labelled, uncertified convergence smoke test.
Production validation removes the experimental flag and requires an independent
delta-branch Lipschitz certificate; the adapter then fails before execution when
the certificate is absent or the scaled bound reaches the safety margin. Both
commands enforce the 16.7 GiB allocated-VRAM postcondition and finite-logit
postcondition.

The third command is the integrated bounded-runtime gate. It must reach depth
100 with the fixed 301-node arena while the actual allocated VRAM remains at or
below 16.7 GiB.
