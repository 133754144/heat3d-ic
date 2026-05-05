# Heat3D v1 Physics-Label Medium Expansion Smoke Report

## Purpose

This report records the 24-sample Heat3D v1 medium expansion smoke. The goal is
to exercise a broader region-first, volume-fraction physics-label generation
path before deciding whether to build the planned 64-sample medium dataset.

This is a 24-sample medium expansion smoke / benchmark-candidate planning
diagnostic / research reference label run. It is not a formal benchmark, not a
high-fidelity solver result, not model-performance evidence, and not OOD
generalization evidence.

## Generated Subset

- subset: `data/heat3d-thermal-simulation/subsets/v1_multilayer_bc_eq_physics_label_medium_expansion_v2`
- manifest: `configs/heat3d_v1_physics_label_medium_expansion_manifest.json`
- sample count: 24
- source assignment: `volume_fraction`
- q policy: `fixed_density`
- solver: `heat3d_v1_reference_solver_v2`
- generated data location: ignored `data/`
- formal 64-sample dataset generated: false

## Split Distribution

| split | count |
|---|---:|
| train | 16 |
| valid | 4 |
| test_id | 2 |
| test_ood_bc_candidate | 1 |
| test_ood_stack_candidate | 1 |

The `test_ood_bc_candidate` and `test_ood_stack_candidate` samples are
diagnostic candidates only and do not support OOD generalization claims.

## Coverage Summary

Heat source pattern coverage:

- `centered_single_hotspot`: 8
- `shifted_single_hotspot`: 3
- `edge_or_corner_hotspot`: 3
- `two_hotspots_same_layer`: 3
- `dual_active_layer_hotspots`: 4
- `broad_block_power`: 3

Thermal conductivity distribution coverage:

- `layerwise_isotropic_k`: 11
- `blockwise_isotropic_k`: 7
- `diagonal_anisotropic_k`: 3
- `interposer_equivalent_k`: 3

Stack template coverage:

- `baseline_4_layer`: 11
- `compact_3_layer`: 5
- `dual_active_4_layer`: 4
- `interposer_like_4_layer`: 3
- `held_out_interposer_like_candidate`: 1

Boundary-condition coverage:

- `nominal_top_h`: 14
- `low_top_h`: 4
- `high_top_h`: 5
- `held_out_top_h_candidate`: 1

Unsupported features remain out of scope: irregular footprint, explicit TSV /
BEOL / bump, contact resistance, transient simulation, multiphysics, and
`(N,6)` full tensor conductivity.

## Source Diagnostics

All 24 samples were generated with region-first `volume_fraction` source
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

Observed integrated power range:

- min: `1.166400e-01 W`
- max: `4.233600e-01 W`

The integrated power variation comes from the planned source size and q-density
categories in the manifest, not from unintended source discretization drift.

## Solver and Label Diagnostics

The expansion checker reported:

- expected sample IDs: `medium_000` through `medium_023`
- observed sample IDs: `medium_000` through `medium_023`
- all samples have `coords.npy`, `k_field.npy`, `q_field.npy`,
  `sample_meta.json`, `temperature.npy`, and `label_meta.json`
- `label_meta.convergence_flag = true` for all samples
- max residual norm: `6.538052e-16`
- max bottom Dirichlet error: `0.000000e+00`
- warning samples: none
- fail samples: none

The label diagnostics smoke reported:

- diagnosed sample count: 24
- status counts: `pass=24`
- label metadata present count: 24
- label metadata missing count: 0
- warning samples: none
- fail samples: none

The current diagnostics remain smoke diagnostics only. PDE residual, flux
mismatch, and global energy diagnostics are still not formal physics validation.

## Temperature Range Summary

Across the 24 generated samples:

- minimum temperature: `300.000000 K`
- maximum observed peak temperature: `300.968642 K`
- all temperatures were finite
- all bottom Dirichlet checks passed at smoke tolerance

The largest temperature rise appeared in a dual-active-layer diagonal
anisotropic diagnostic sample. This is an expected smoke response and should not
be interpreted as a model or benchmark result.

## Next Step

The next step is to decide whether to:

1. extend the manifest and generator to the planned 64-sample medium dataset,
2. add a no-write resolver / dry-run for the 64-sample plan first, or
3. run the existing zero-delta / train-valid / metrics smoke on this 24-sample
   expansion subset before expanding further.

Any downstream training or comparison should continue to use the existing
non-claim boundary: benchmark-candidate diagnostics only, not formal benchmark
or model-performance evidence.
