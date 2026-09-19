# V8-P2B canonical physics/query contract

Date: 2026-09-20

Status: `CANONICAL_CONTRACT_IMPLEMENTED_TWO_SAMPLE_ENGINEERING_SMOKE`

Scope: Mac CPU only, two real non-LC MASS-HBM cases, no checkpoint selection and no formal or multi-case training. G1/G2/V7 frozen evidence is unchanged. The machine-readable execution evidence is `docs/v8_mass_hbm_p2b_smoke_receipt.json`.

## 1. Physical-scale contract

Sample-local coordinate normalization remains useful for graph conditioning, but it maps different physical packages into the same normalized coordinate cube. V8-P2B therefore broadcasts the following declared-geometry features to every physical point:

```text
log_Lx_m
log_Ly_m
log_Lz_m
log_xy_area_m2
log_domain_volume_m3
```

These are the V8 equivalents of the physical scale terms already present in V7 Full global context (`log_Lx_m`, `log_Ly_m`, `log_Lz_m`, `log_top_area_m2`). The volume term is the corresponding generic-domain scale. No architecture, material, layer or workload category is introduced.

The same scale values also appear in the generic V8 global physics context used by FiLM. The context retains V7 Full concepts—source power/concentration, q-weighted and harmonic conductivity, anisotropy and package scale—while replacing top/bottom-specific terms with total boundary area, total sink conductance, conductance-weighted sink-temperature offset and boundary control-volume fraction.

## 2. Canonical RIGNO bridge

The formal bridge is:

```text
u = zeros([B,1,N,1])
c = all V8 physics features
```

`k_x` is no longer placed in `u`. For Track A, `c` has 27 channels:

- 5 cell channels: converged `kx,ky,kz,q` and control volume;
- 13 generic node-aggregated boundary channels;
- 4 double-count-guarded interface channels;
- 5 broadcast physical-scale channels.

The first four cell fields and any explicit final-interface conductance are `ORACLE`; geometry, control volume, current non-LC external BC and scale fields are `PRE_SOLVE`. A future Track-B bridge must preserve `u=0,c=physics` while replacing every ORACLE value with an independently reconstructed pre-solve field.

## 3. V7 Full capacity migration

`configs/heat3d_v8/v8_p2b_canonical.json` preserves the main V7 Full capacity:

| Parameter | V8-P2B |
|---|---:|
| node latent | 96 |
| edge latent | 96 |
| processor steps | 6 |
| MLP hidden layers | 2 |
| conditioned normalization | false |

It also preserves global FiLM and a local-condition decoder bypass, but their physics sources are replaced:

- old P1i top/bottom/side flags: removed;
- old top/bottom-specific global context: replaced by generic boundary/sink aggregates;
- old eight-feature P1i bypass: replaced by all V8 condition channels;
- categorical architecture/material/workload inputs: forbidden.

The P2B output smoke uses normalized case-reference temperature rise. It does not freeze a formal V8 loss, optimizer, target standardizer or checkpoint-selection policy.

## 4. Query contract

The frozen engineering contract is:

```text
1024 sparse global conditioning support
+ query-local physical coefficients
```

The 1024 support points build the global encoder/regional representation. Every requested query point additionally supplies its local `c` coefficients to the output-local encoder and sparse r2p decoder. Thus “1024 support” does **not** mean that the model reads physics at only 1024 locations during full-field inference.

For `support == query`, the asymmetric route must reuse the input local latents. It is gated against the standard public RIGNO route at `atol=rtol=1e-6`. A mismatch is a hard failure. For `support != query`, the output-local encoder consumes query-local physics, and graph construction must use Sparse KD-tree, chunking and the geometry cache; dense query-by-regional pairwise construction is forbidden.

## 5. Support provenance

Every support selection carries one of two mandatory provenance values:

- `ORACLE_SUPPORT`: any stratum mask depends on converged solver fields;
- `PRE_SOLVE_SUPPORT`: every stratum mask is reconstructable before solving.

The current heat-source mask is `final_q != 0`; therefore both P2B smoke supports are `ORACLE_SUPPORT`. The selector itself does not consume temperature, gradients, hotspot scores or field amplitudes, but this does not make its candidate mask deployable or target-independent. Track B must fail closed unless `PRE_SOLVE_SUPPORT` is demonstrated.

## 6. Interface contract

- `rotated_homogenized_tensor`: effective `k` may be used only when explicit TBR is exactly zero; no second interface resistance feature is added.
- `explicit_series_z`: positive z-face TBR enters the neural condition only when the raw `PASS_interface_only_bonding_representation` guard is present.
- Unknown/review-required/mixed representations fail closed.

P2B does not freeze the future paired-edge interface representation or the half-cell-plus-interface resistance formula.

## 7. Geometry fingerprint claim

The fingerprint excludes label arrays, final `k/q/Rint`, workload labels, case identities and hotspot fields. It still includes `solver_geometry.json`. Until the authors confirm that this is a pre-solve mesh rather than a temperature-adapted final mesh, the correct claim is:

```text
LABEL_ARRAY_INDEPENDENT
STRICT_TARGET_INDEPENDENCE_UNCONFIRMED
```

No formal split is frozen in P2B.

## 8. Two-case smoke scope

The two cases deliberately separate physical scale and interface representation:

1. `cases/Exp0_SolverAcceleration/6765682094df176c/repeat_01`: 3D-V, `Lz=5.76 mm`, homogenized interface, 59,150 cells.
2. `cases/Exp0_SolverAcceleration/2f7b8b6c5aee5687/repeat_01`: 2.5D, `Lz=1.51 mm`, explicit nonzero z-face TBR with raw guard PASS, 71,825 cells.

Both have `Lx=Ly=65 mm`; their normalized coordinates alone cannot retain the 3.81× z-scale difference. The explicit scale channels do.

## 9. Advancement rule

An 8–32 case Track-A pilot is allowed only if the receipt passes all of these hard gates:

- both adapters and power conservation;
- `u=0,c=all physics` bridge;
- V7 Full capacity migration;
- standard/asymmetric support=query equivalence;
- explicit TBR neural path and double-count guard;
- finite joint forward/backward with decreasing two-sample loss;
- both sparse/chunked full-field queries;
- explicit `ORACLE_SUPPORT` labeling.

This conditional Track-A permission is not Track-B permission and cannot support accuracy, OOD or solver-replacement claims.

## 10. Verified P2B smoke result

The final Mac CPU receipt passed every hard gate:

- canonical bridge: PASS, with 27 `c` channels and identically zero `u`;
- Full-capacity RIGNO: 849,062 parameters;
- standard/asymmetric equivalence: maximum absolute difference `0.0` for both cases at `atol=rtol=1e-6`;
- explicit interface path: 6,624 positive guarded TBR faces, nonzero neural interface condition, PASS;
- 20-step joint loss: `0.0397868939 -> 0.0267701969`, ratio `0.672839582`;
- per-case loss ratios: `0.783952534` (homogenized) and `0.618024051` (explicit);
- full-field queries: 59,150 and 71,825 points, both Sparse KD-tree/chunked/cache PASS;
- peak Mac process RSS: 1,360,248,832 bytes;
- checkpoint written: no; accuracy claim: no.

Decision: `CONDITIONAL_GO` for an 8–32 case **Track-A-only** pilot under the same guards. Track B and P3A remain out of scope and are not authorized by this result.
