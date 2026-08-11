# V6 P1i U1 asymmetric-query feasibility

Status: `passed_expected_no_go`. Decision: **NO_GO_requires_model_path_change_or_retraining**.

## Probe

| N out | graph | native p2r/r2r exact | Nr | P2R | R2P | forward |
|---:|---|---|---:|---:|---:|---|
| 8192 | passed | True | 256 | 2544 | 19252 | failed_structural_incompatibility |
| 32768 | passed | True | 256 | 2544 | 63348 | not_executed_fail_fast_same_frozen_interface_blocker |

## Interface audit

- The lower-level `RegionInteractionGraphBuilder.build_metadata` accepts distinct `x_inp` and `x_out`; regional nodes are sampled from `x_inp`, while r2p targets `x_out`.
- `Heat3DGraphBuilder`, the V6 bridge, and the controlled runner bind one coordinate tensor to both sides.
- This probe froze native-1024 p2r/r2r/regional metadata bitwise and built only the N-node r2p side. Both requested graphs passed.
- At N=8192 the decoder core produced an N-node tensor and execution reached the local bypass. The bypass then rejected the shared 1024-node `Inputs.c`.
- A complete asymmetric native shape-scale call would additionally require N-node CV/Dirichlet fields while retaining anchor-derived context and scale.

## Structural potential

| N out | encoder node reduction | P2R edge reduction | P2R+R2P edge reduction | query R2P build |
|---:|---:|---:|---:|---:|
| 8192 | 87.50% | 95.27% | 79.75% | 0.018744 s |
| 32768 | 96.88% | 98.44% | 79.86% | 0.028019 s |

## Blockers

- frozen Inputs.c is shared by the 1024-node encoder and the N-node local decoder bypass
- frozen local decoder bypass requires one-to-one x_inp/x_out and c/output alignment
- native shape-scale projection requires N-node CV/BC fields while log_s/context must remain anchor-derived
- runner wrapper constructs identical x_inp/x_out and has no frozen asymmetric group contract

## Interpretation

The graph primitive can preserve the native 1024 encoder/processor graph and attach an N-node r2p query graph, and the decoder core reaches its N-node output. The frozen full model path still cannot complete because input conditions, output-local bypass conditions, and output-native projection fields have no split asymmetric interface. Adding that interface exceeds this minimal checker and requires a separately preregistered adapter validation, although no checkpoint weights were changed here.

The structural edge-count reductions are potential savings only; no successful U1 forward latency is claimed. Current B8192/E32768 production routes remain unchanged.
