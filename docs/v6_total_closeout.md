# V6 total closeout

Status: **closed**. Canonical status is role-scoped; there is no single dataset
that replaces every V6 geometry family.

This compact `main` summary binds the stable P1h/P1i runtime and evidence only.
Raw experiments, predictions, checkpoints, logs, failed attempts, smoke payloads
and large diagnostics remain on the research evidence branches.

## Role-scoped canonical datasets and models

| Role | Dataset | Model/checkpoint | Scope |
|---|---|---|---|
| V6 layer | `heat3d_v6_p1h_shared_support1024_v0` | `V6_03_V5best_P1h`, seed0 e111 | P1h shared source-aware layered support family |
| V6 formal random-block | `heat3d_v6_p1i_continuous_physics1024_v1` | `V6_06_V5best_P1i_seed0_reliable_B24`, point-global-best e559 | frozen P1i generator/material/BC/source ranges |

P1h remains the canonical V6-layer dataset. P1i is the canonical formal
random-block role and does not supersede P1h. P1i seeds 1/2 are replication
runs; their checkpoints are evidence artifacts and are not committed to main.

## P1i frozen inference strategy

The deployment family uses 1,024 source-aware conditioning anchors for Global
Context and scale, a frozen high-N query graph, and layer/interface-aware
reconstruction to the 240,825-node solver field.

- `E16384_reconstruction` is the frozen reference operating point selected by
  the preregistered accuracy non-inferiority + latency-Pareto decision chain.
- `U_v2_16384_reconstruction` and `U_v2_direct240825` remain parallel valid-only
  inference strategies; U-v2 uses bounded output-query extrapolation and the
  frozen R2P nearest-coverage repair.
- `E240825_direct_control` is an architecture control, not the reference route.
- 32,768 nodes gave only a marginal point-global improvement over 16,384 while
  source/peak error and runtime/memory did not improve consistently. It remains
  exploratory scalability evidence, not a production operating point.

The frozen route/checkpoint was opened once on corrected `test_iid=128` only
after development choices were fixed. It was not used for selection or tuning.
The peak-error increase is tail-driven and is recorded only as an applicability
boundary.

## Performance and solver semantics

Publication timing uses persistent services with the common boundary
`in-memory k/q/BC -> synchronized 240825-node result`. Fresh, resident-core,
Q1, true Q2, throughput and B16-to-B32 marginal workloads remain separate.
The WSL2 Attempt 4 matrix is primary; devbox is an independent hardware-state
replication and is not pooled as additional model seeds.

FVM240825 is the physical reference. Neural errors are surrogate errors against
that field. FVM retains conservation and physical-consistency advantages. GPU
RIGNO speedups are controlled-error workflow comparisons and do not imply
surrogate accuracy exceeds FVM. Cross-resolution model/FVM comparisons remain
nonmatched-DOF unless explicitly marked otherwise.

## Governance and applicability boundaries

- `test_iid` is a corrected confirmatory holdout; it never selected a route,
  checkpoint, threshold or claim.
- The preregistered sealed IID is a **post-development final confirmation** set.
  It remains ungenerated and unopened at V6/P1i closeout.
- P1h/P1i assume perfect interface contact (`R_contact=0`) and cannot represent
  variable contact resistance.
- Claims are bounded to frozen data generators, material/BC/source ranges,
  support semantics, checkpoints and reconstruction.
- No universal geometry/OOD, experimental-package validation or calibrated
  uncertainty claim is made.

## Evidence entry points

- `configs/heat3d_v6/v6_phase_index.json`
- `configs/heat3d_v6_p1i/v6_p1i_main_integration_manifest.json`
- `configs/heat3d_v6_p1i/v6_p1i_main_integration_receipt.json`
- `docs/v6_p1i_publication_evidence_summary.md`
- `docs/v6_p1i_error_tail_closeout.md`
- `docs/v6_p1i_closeout.md`
- `docs/v6_p1i_handoff.md`

The historical P1h core receipt remains in
`configs/heat3d_v6/v6_core_integration_manifest.json`; the P1i allowlist receipt
extends it without rewriting the prior integration history.
