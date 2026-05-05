# Heat3D v1 Physics-Label Medium Pilot Report

## Purpose

This report records the first 8-sample Heat3D v1 physics-label medium pilot
smoke. The goal is to verify that the region-first, volume-fraction source
assignment policy can generate a small benchmark-candidate pilot subset with
solver v2 labels and label diagnostics.

This is a medium pilot / physics-label smoke / benchmark-candidate pilot. It is
not a formal benchmark, not high-fidelity solver validation, not
model-performance evidence, and not OOD generalization evidence.

## Pilot Subset

- subset: `data/heat3d-thermal-simulation/subsets/v1_multilayer_bc_eq_physics_label_medium_pilot_v2`
- manifest draft: `configs/heat3d_v1_physics_label_medium_manifest_draft.json`
- sample count: 8
- split counts: `train=5`, `valid=1`, `test_ood_bc_candidate=1`, `test_ood_stack_candidate=1`
- source assignment: `volume_fraction`
- q policy: `fixed_density`
- solver: `heat3d_v1_reference_solver_v2`
- grid policy: pilot mid grid, 384 nodes per sample

The generated subset is ignored local data and is not committed to Git.

## Pilot Coverage

| sample | split | purpose | k shape |
|---|---|---|---|
| pilot_000 | train | baseline single hotspot | `(384, 1)` |
| pilot_001 | train | shifted hotspot | `(384, 1)` |
| pilot_002 | train | two hotspots | `(384, 1)` |
| pilot_003 | train | dual active layers | `(384, 1)` |
| pilot_004 | train | block-wise equivalent k | `(384, 1)` |
| pilot_005 | valid | diagonal anisotropy diagnostic | `(384, 3)` |
| pilot_006 | test_ood_bc_candidate | held-out top HTC candidate | `(384, 1)` |
| pilot_007 | test_ood_stack_candidate | held-out stack candidate | `(384, 1)` |

The held-out candidate samples are diagnostic candidates only. They do not
support an OOD generalization claim.

## Source Diagnostics Summary

All 8 samples used region-first `volume_fraction` projection.

| sample | active source volume | integrated power W | power rel. error | source missed |
|---|---:|---:|---:|---|
| pilot_000 | `1.742400e-09` | `1.742400e-01` | `1.592951e-16` | false |
| pilot_001 | `1.742400e-09` | `1.742400e-01` | `3.185902e-16` | false |
| pilot_002 | `2.332800e-09` | `2.332800e-01` | `0.000000e+00` | false |
| pilot_003 | `2.760000e-09` | `1.980000e-01` | `2.803593e-16` | false |
| pilot_004 | `1.742400e-09` | `2.613600e-01` | `0.000000e+00` | false |
| pilot_005 | `1.742400e-09` | `1.742400e-01` | `1.592951e-16` | false |
| pilot_006 | `1.742400e-09` | `1.742400e-01` | `1.592951e-16` | false |
| pilot_007 | `1.306800e-09` | `1.306800e-01` | `2.123934e-16` | false |

Summary:

- `source_missed_count = 0`
- max integrated q power relative error: `3.185902e-16`
- every sample records source region volume, discrete active volume, integrated
  q power, active source cell count, source volume relative error, integrated q
  power relative error, and source missed status.

## Solver v2 and Temperature Summary

| sample | T range K | convergence | residual norm | bottom Dirichlet error K |
|---|---:|---|---:|---:|
| pilot_000 | `[300.000000, 300.357477]` | true | `3.282865e-16` | `0.000000e+00` |
| pilot_001 | `[300.000000, 300.440555]` | true | `3.544078e-16` | `0.000000e+00` |
| pilot_002 | `[300.000000, 300.262273]` | true | `3.396985e-16` | `0.000000e+00` |
| pilot_003 | `[300.000000, 300.431962]` | true | `2.082906e-16` | `0.000000e+00` |
| pilot_004 | `[300.000000, 300.646413]` | true | `3.829834e-16` | `0.000000e+00` |
| pilot_005 | `[300.000000, 300.563994]` | true | `2.493929e-16` | `0.000000e+00` |
| pilot_006 | `[300.000000, 300.356368]` | true | `3.213617e-16` | `0.000000e+00` |
| pilot_007 | `[300.000000, 300.115751]` | true | `6.067208e-16` | `0.000000e+00` |

Summary:

- all 8 samples have `temperature.npy`
- all 8 samples have `label_meta.json`
- all 8 samples have finite temperature arrays
- all 8 samples have solver `convergence_flag = true`
- max residual norm: `6.067208e-16`
- max bottom Dirichlet error: `0.000000e+00`

## Label Diagnostics Summary

`scripts/check_heat3d_v1_label_diagnostics.py --subset ...medium_pilot_v2`
reported:

- diagnosed sample count: 8
- status counts: `pass=8`
- warning samples: none
- fail samples: none
- label metadata present count: 8
- label metadata missing count: 0

The diagnostics remain smoke diagnostics only. PDE residual, flux mismatch,
and global energy diagnostics are still not formal physics validation.

## Next Step

The next step is to decide whether to extend this pilot into a 64-sample
medium-small physics-label dataset. Before doing so, the generator should gain
a no-write resolver / dry-run mode for all 64 planned samples, and the same
source-power and label diagnostics checks should remain mandatory.
