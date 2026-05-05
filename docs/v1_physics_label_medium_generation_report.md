# Heat3D v1 Physics-Label Medium Dataset Smoke Report

## Purpose

This report records the 64-sample Heat3D v1 physics-label medium generation
smoke. The goal is to check whether the region-first, volume-fraction,
reference-solver-v2 label pipeline can generate a medium-size benchmark
candidate subset with source, solver, and label diagnostics.

This is a 64-sample medium generation smoke / research reference label run /
benchmark-candidate dataset. It is not a formal benchmark, not a high-fidelity
solver result, not model-performance evidence, and not OOD generalization
evidence.

## Generated Subset

- subset: `data/heat3d-thermal-simulation/subsets/v1_multilayer_bc_eq_physics_label_medium_v2`
- manifest: `configs/heat3d_v1_physics_label_medium_manifest.json`
- sample count: 64
- source assignment: `volume_fraction`
- q policy: `fixed_density`
- solver: `heat3d_v1_reference_solver_v2`
- generated data location: ignored `data/`
- old subsets overwritten: false

## Split Distribution

| split | count |
|---|---:|
| train | 48 |
| valid | 8 |
| test_id | 4 |
| test_ood_bc_candidate | 2 |
| test_ood_stack_candidate | 2 |

The `test_ood_bc_candidate` and `test_ood_stack_candidate` samples are
diagnostic candidates only. They do not support OOD generalization claims.

## Coverage Summary

Heat source pattern coverage:

- `centered_single_hotspot`: 10
- `shifted_single_hotspot`: 9
- `edge_or_corner_hotspot`: 9
- `two_hotspots_same_layer`: 9
- `dual_active_layer_hotspots`: 9
- `broad_block_power`: 10
- `multi_block_power`: 8

Thermal conductivity distribution coverage:

- `layerwise_isotropic_k`: 31
- `blockwise_isotropic_k`: 16
- `interposer_equivalent_k`: 9
- `diagonal_anisotropic_k`: 8

K-field mode coverage:

- `iso1`: 56
- `diag3`: 8

Stack template coverage:

- `baseline_4_layer`: 27
- `compact_3_layer`: 13
- `interposer_like_4_layer`: 13
- `dual_active_4_layer`: 9
- `held_out_interposer_like_candidate`: 2

Boundary-condition coverage:

- `nominal_top_h`: 37
- `low_top_h`: 13
- `high_top_h`: 12
- `held_out_top_h_candidate`: 2

Unsupported features remain out of scope: irregular footprint, explicit TSV /
BEOL / bump, contact resistance, transient simulation, multiphysics, and
`(N,6)` full tensor conductivity.

## Source Diagnostics

All 64 samples were generated with region-first `volume_fraction` source
assignment.

- `source_missed_count`: 0
- max integrated q power relative error: `4.319262e-16`
- source power diagnostics were recorded in `sample_meta.json`
- every generated sample records:
  - `source_region_volume_target`
  - `active_source_volume_discrete`
  - `integrated_q_power`
  - `active_source_cell_count`
  - `source_volume_relative_error`
  - `integrated_q_power_relative_error`
  - `source_missed`

The integrated power variation across samples is intentional. It comes from
planned source-region sizes and fixed-density q categories in the manifest, not
from unintended source discretization drift.

## Solver and Label Diagnostics

The medium checker reported:

- expected sample IDs: `medium_000` through `medium_063`
- observed sample IDs: `medium_000` through `medium_063`
- all samples have `coords.npy`, `k_field.npy`, `q_field.npy`,
  `sample_meta.json`, `temperature.npy`, and `label_meta.json`
- `label_meta.convergence_flag = true` for all samples
- max residual norm: `6.586586e-16`
- max bottom Dirichlet error: `0.000000e+00`
- warning samples: none
- fail samples: none

The label diagnostics smoke reported:

- diagnosed sample count: 64
- status counts: `pass=64`
- split counts: `train=48`, `valid=8`, `test_id=4`,
  `test_ood_bc_candidate=2`, `test_ood_stack_candidate=2`
- label metadata present count: 64
- label metadata missing count: 0
- warning samples: none
- fail samples: none

The current diagnostics remain smoke diagnostics only. PDE residual, flux
mismatch, and global energy diagnostics are still not formal physics
validation.

## Temperature Scope

The generated `temperature.npy` labels were produced by
`heat3d_v1_reference_solver_v2` and are paired with `label_meta.json` solver
metadata. They are research reference labels for this smoke pipeline, not
validated industrial or high-fidelity thermal simulation results.

The medium checker and label diagnostics verified finite arrays, expected
shapes, solver convergence flags, residual tolerance, bottom Dirichlet
consistency, and source-power consistency. They do not establish formal grid
convergence, external solver agreement, or industrial package fidelity.

## Next Step

The next step is a 64-sample medium end-to-end smoke:

1. run label diagnostics on the medium subset,
2. run the existing zero-delta bridge smoke,
3. run tiny train-valid smoke if needed,
4. run validation metrics smoke,
5. then decide whether the current v1 model is ready for a controlled training
   and validation experiment.

Any downstream model run should continue to use the same non-claim boundary:
research reference diagnostics only, not formal benchmark or model-performance
evidence.
