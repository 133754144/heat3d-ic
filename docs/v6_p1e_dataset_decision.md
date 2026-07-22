# V6-P1e dataset deconfounding qualification and decision

## Decision

Rebuild, do not patch P1d in place.  The immutable P1d-1024 artifacts remain
provenance, while `heat3d_v6_p1e_deconfounded1024_v0` is qualified as the V6
dataset for subsequent training work.  This qualification does not start or
configure any model training or inference.

P1d failed the deconfounding audit because top-h and package power had Pearson
correlation `0.885861` (Spearman `0.912070`), bottom-h and power had Pearson
correlation `0.528623`, every sample had eight sources, and there was no
fixed-power/fixed-geometry sweep that independently varied top and bottom h.
The old config, sample table, and source table hashes remain frozen in
`v6_p1e_p1d_baseline_deconfounding_audit.json`.

## Literature boundary

The P1e supplement records direct unequal top/bottom convection, explicit
package-to-PCB paths, PCB/chassis natural and forced convection, and
architecture-specific liquid cooling.  Numeric endpoints are used only where
the primary source directly reports a compatible scalar boundary.  3D-ICE and
MASS-HBM are architecture/liquid-cooling or high-power scenario evidence only;
neither is a source for a uniform Robin coefficient.

## Frozen design

- package powers: `2/6/10/14 W` for every BC family, exactly 256 samples each;
- main top h: `500/1000/1750/2500 W/(m2 K)`;
- main bottom h: `1/20/80/200 W/(m2 K)`;
- 8 complete `4 x 4 x 4` groups (512 cases), including a separately frozen and
  solved 128-case qualification block;
- 32 balanced orthogonal-array groups (512 cases), including IID and
  layout/BC/source-count/power-density OOD roles;
- 40 geometry groups total; all BC and power variants of a group remain in one
  split and reuse the same pre-label 1024 irregular coordinates;
- source count varies from 3--10 in IID roles, with 2 and 12 reserved for
  source-count OOD; source area, aspect ratio, clustering, layer power split,
  and inter-layer alignment are independently frozen per geometry group.

Split counts are train `640`, valid_iid `128`, test_iid `128`, and `32` each for
layout, BC, source-count, and power-density OOD.  These are dataset roles only;
no model or model-selection operation occurred in P1e.

## Qualification evidence

- power correlation with top h: Pearson `0`, Spearman `0`;
- power correlation with bottom h: Pearson `0`, Spearman `0`;
- factor-design algebraic/effective rank: `3 / 3.0000`;
- amplitude-normalized projected-field effective rank: `9.1708`;
- fixed geometry/power top and bottom peak monotonicity: `100% / 100%`;
- bottom heat-fraction monotonicity with bottom h: `100%`;
- median extreme top/bottom field relative-RMS response: `341.61% / 29.31%`;
- bottom BC qualification: `learnable_nonzero_independent_response`;
- minimum source resolution: `240` control volumes and `7` in-plane intervals;
- maximum source q: `1.3274074074e10 W/m3`;
- maximum single-source power: `7 W`;
- maximum source surface power density: `141.149 W/cm2`;
- maximum absolute energy-balance relative error: `2.395e-10`;
- all layers and interfaces are represented in every 1024-point projection.

Peak DeltaT was never a selection rule.  `409/1024` cases naturally fall in
30--80 K; the distribution has min/Q1/median/Q3/max
`8.95/35.98/65.37/113.09/308.77 K`.  The broad high-temperature tail is kept
because it is the preregistered consequence of applying the common high-power
levels to weak cooling.  It is valid for the frozen constant-property solver
contract, but it must not be presented as validation of temperature-dependent
real-material behavior at the highest temperatures.

## Frozen artifacts

- config SHA256: `8d1448005a2afb3267c891dfb5660cf5d6e2ea3e9ca6bce6abee755b3f1ae1e3`;
- manifest SHA256: `beba25459a9afd69b361135f36f7bfd8ac0393b9bab4496b498bdd251556208d`;
- literature supplement SHA256: `bf0b580f42a94cd260fe36ccb0d17545a8d16a0cf22728ff75e914600d231e7c`;
- no temperature filtering, Rth power inversion, post-solve seed/factor search,
  sample replacement, model training, or model inference was performed.
