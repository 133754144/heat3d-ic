# Heat3D v1 Physics-Label Medium1024 Dataset Plan

## Motivation

The medium256 stage has shown that the current Heat3D v1 supervised neural
operator runner can learn multilayer 3D temperature fields and that
`background_l1_relative` with `lr=0.01` can outperform `zero_delta` in the
current diagnostic protocol. It also exposed important limits: clear seed
sensitivity, persistent low-DeltaT `bin_0` background overprediction, high-bin
underprediction, and unstable OOD BC / OOD stack mean-field behavior.

The proposed medium1024 dataset is meant to test these issues more reliably. It
is not simply a 4x sample-count expansion. The intended role is a planned
research benchmark candidate that broadens source, conductivity, stack, and BC
variation while preserving auditable splits and label diagnostics.

This plan is not a completed benchmark and does not provide formal model
performance, OOD generalization, or high-fidelity package-level validation.

## Scope

The planned medium1024 scope remains controlled:

- steady 3D heat conduction labels;
- multilayer stack structures;
- explicit top-Robin / bottom-Dirichlet / side-adiabatic BC encoding;
- region-first material construction;
- rectilinear sampling;
- volume-fraction source assignment;
- fixed-density source power policy;
- equivalent interconnect / blockwise material regions where supported;
- diagonal anisotropic `k` coverage where supported.

Unsupported future directions such as additional bottom-temperature variants,
more complex package-level BCs, or industrial arbitrary 3D IC layouts are
recorded as planned metadata only. They must not be silently treated as already
implemented.

## Proposed Split

The first medium1024 manifest draft uses 1024 samples:

| split | count | purpose |
| --- | ---: | --- |
| `train` | 768 | Regular stack, regular BC, regular source/k coverage. |
| `valid` | 128 | Regular stack and BC for training diagnostics and model selection. |
| `test_id` | 64 | Regular stack and BC held out by sample identity. |
| `test_ood_bc_candidate` | 24 | Held-out BC with regular stack only. |
| `test_ood_stack_candidate` | 24 | Held-out stack with regular BC only. |
| `test_ood_combined_candidate` | 16 | Held-out stack plus held-out BC for extreme diagnostics. |

The candidate splits remain diagnostics only. `test_ood_bc_candidate` must not
mix in held-out stack templates, and `test_ood_stack_candidate` must not mix in
held-out BC categories. The combined split is intentionally separate and should
not be used as a primary OOD claim.

## Proposed Diversity Axes

### Source / q Pattern

The plan keeps the medium256 source modes and adds staged source diversity:

- `centered_single_hotspot`
- `shifted_single_hotspot`
- `edge_or_corner_hotspot`
- `two_hotspots_same_layer`
- `dual_active_layer_hotspots`
- `broad_block_power`
- `multi_block_power`
- `random_sparse_hotspots`
- `elongated_strip_power`
- `ring_or_annular_like_power`
- `checkerboard_or_patchy_power`
- `low_power_near_zero_background_cases`
- `high_dynamic_range_power_cases`

The first seven are implemented in the current medium generator family. The
remaining modes are planned metadata unless generator support is added and
smoke-checked later.

### k-Field / Material Distribution

The planned material modes are:

- `layerwise_isotropic_k`
- `blockwise_isotropic_k`
- `interposer_equivalent_k`
- `diagonal_anisotropic_k`
- `mixed_blockwise_diag_anisotropic_k`
- `high_contrast_interface_k`
- `low_k_barrier_or_TIM_variation`
- `equivalent_interconnect_region_variants`
- `locally_randomized_k_blocks`

The planned `k_field_mode` mix is:

- `iso1`: 704 samples;
- `diag3`: 320 samples;
- `diag3` fraction: 31.25%.

This raises diagonal anisotropic coverage beyond the medium256 25% target while
keeping isotropic samples dominant for continuity. New material-distribution
modes are implementation-staged and must not be assumed generator-ready.

### Stack Template

The planned stack templates are:

- `baseline_4_layer`
- `compact_3_layer`
- `interposer_like_4_layer`
- `dual_active_4_layer`
- `thick_substrate_4_layer`
- `thin_tim_high_sink_4_layer`
- `five_layer_package_like_stack`
- `dual_interposer_or_bridge_like_stack`
- `held_out_interposer_like_candidate`
- `held_out_deep_stack_candidate`
- `held_out_thin_die_candidate`

The plan intentionally increases held-out stack coverage to 40 samples across
the stack-only and combined candidate splits. New regular and held-out stack
templates require generator implementation before any full dataset generation.

### BC Category

The current V1 BC model is top Robin, bottom Dirichlet, and adiabatic sides.
The planned categories are:

- `nominal_top_h`
- `low_top_h`
- `high_top_h`
- `mixed_top_h_with_same_bottom_fixed`
- `held_out_top_h_candidate`
- `very_low_top_h_candidate`
- `very_high_top_h_candidate`

Optional bottom-temperature variants are a roadmap item only. They are not
included in the current medium1024 counts because they require explicit
generator and loader support.

### OOD Candidates

The candidate policies are:

- held-out BC categories only enter `test_ood_bc_candidate` and
  `test_ood_combined_candidate`;
- held-out stack templates only enter `test_ood_stack_candidate` and
  `test_ood_combined_candidate`;
- combined held-out samples are separated from BC-only and stack-only samples.

These splits are intended to diagnose robustness, not to claim solved OOD
generalization.

## Relationship To Medium256

Medium256 remains the fast debug and ablation dataset. It should continue to be
used for loss-stage experiments, seed-sensitivity checks, and quick controlled
training before medium1024 is generated.

Medium1024 should be treated as the next stability and generalization candidate.
It should not be used to hide medium256 weaknesses. Instead, the medium256
multi-seed summary should define which issues medium1024 must stress-test:
seed variance, low-DeltaT background bias, high-bin underprediction, and
candidate OOD stack/BC stability.

## Expected Experiments

After generator support, partial-smoke checks, and full label generation are
approved, medium1024 should support:

- `zero_delta` baseline;
- default MSE baseline;
- `background_l1_relative`;
- `lr=0.01`, seeds 0/1/2;
- optional seed 3 if uncertainty remains;
- multi-seed summary;
- split-wise and condition-wise comparison;
- error-bin analysis for low-background and high-hotspot regions.

All experiments should use fixed manifests, fixed splits, fixed training
budgets, and conservative wording.

## Risks

Key risks:

- full generation cost and solver-label runtime;
- label quality and convergence diagnostics at larger scale;
- imbalance across too many source/k/stack/BC axes;
- ablations becoming hard to interpret if too many unsupported modes are added
  at once;
- held-out BC and held-out stack splits leaking into each other;
- generator metadata drifting away from implemented behavior;
- overclaiming candidate OOD results as formal generalization.

If the current generator lacks a planned source, material, stack, or BC mode,
the correct action is to mark it as planned and implement it in a separate
small-smoke step, not to fake the coverage.

## Publication-Oriented Next Steps

Medium1024 can support more formal paper-style reporting after it is actually
implemented and generated:

- dataset composition table;
- model input/output schematic;
- split-wise performance table;
- multi-seed mean/std table;
- best/median/worst seed table;
- condition-wise OOD BC and OOD stack diagnostics;
- error-bin plots for background and hotspot behavior;
- limitations section that distinguishes controlled benchmark-candidate results
  from industrial 3D IC generalization.

The immediate next step should be manifest review and generator gap assessment,
then a partial 32/64-sample dry-run after generator support is added. Full 1024
generation should wait until those checks pass.
