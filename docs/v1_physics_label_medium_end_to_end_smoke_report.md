# Heat3D v1 Physics-Label Medium End-to-End Smoke Report

## Purpose

This report records the end-to-end smoke checks for the 64-sample
`v1_multilayer_bc_eq_physics_label_medium_v2` subset. The goal is to verify
that the existing v1 supervised route can consume the generated physics-label
medium subset through label diagnostics, zero-delta bridge, tiny training, and
validation metrics smoke.

This is an end-to-end smoke / research reference diagnostic only. It is not a
formal benchmark, not model-performance evidence, not OOD generalization
evidence, and not high-fidelity solver evidence.

## Subset Summary

- subset: `data/heat3d-thermal-simulation/subsets/v1_multilayer_bc_eq_physics_label_medium_v2`
- sample count: 64
- split counts: `train=48`, `valid=8`, `test_id=4`,
  `test_ood_bc_candidate=2`, `test_ood_stack_candidate=2`
- source assignment: `volume_fraction`
- q policy: `fixed_density`
- solver: `heat3d_v1_reference_solver_v2`
- label metadata: `label_meta.json` exists for all 64 samples

The test candidate splits remain diagnostic-only candidates and do not support
OOD generalization claims.

## Coverage Summary

Heat source pattern coverage:

- `centered_single_hotspot`: 10
- `shifted_single_hotspot`: 9
- `edge_or_corner_hotspot`: 9
- `two_hotspots_same_layer`: 9
- `dual_active_layer_hotspots`: 9
- `broad_block_power`: 10
- `multi_block_power`: 8

Thermal conductivity and k-field coverage:

- `layerwise_isotropic_k`: 31
- `blockwise_isotropic_k`: 16
- `interposer_equivalent_k`: 9
- `diagonal_anisotropic_k`: 8
- `iso1`: 56
- `diag3`: 8

Stack and BC coverage:

- `baseline_4_layer`: 27
- `compact_3_layer`: 13
- `interposer_like_4_layer`: 13
- `dual_active_4_layer`: 9
- `held_out_interposer_like_candidate`: 2
- `nominal_top_h`: 37
- `low_top_h`: 13
- `high_top_h`: 12
- `held_out_top_h_candidate`: 2

## Source and Label Diagnostics

Medium generation checker:

- sample count: 64
- `source_missed_count`: 0
- max integrated q power relative error: `4.319262e-16`
- max residual norm: `6.586586e-16`
- max bottom Dirichlet error: `0.000000e+00`
- warning samples: none
- fail samples: none

Label diagnostics:

- diagnosed sample count: 64
- status counts: `pass=64`
- label metadata present count: 64
- label metadata missing count: 0
- warning samples: none
- fail samples: none

These diagnostics remain smoke diagnostics. They do not establish formal
physics validation, grid convergence, or external solver agreement.

## Zero-Delta Bridge Smoke

The zero-delta bridge smoke ran on all 64 medium samples.

- relative BC feature view: pass
- `legacy_inputs.u = zero_delta_field`: pass
- raw absolute BC temperatures excluded from model-facing features: pass
- `T_ref` excluded from model-facing inputs: pass
- `T_ref + target_delta_u == target_temperature`: pass
- forward/loss shape contract: pass
- baseline-shift model-facing input invariance: pass

Summary:

- checked sample count: 64
- all selected zero-delta bridge smoke: true
- zero-delta forward/loss smoke ok: true

## Tiny Training Smoke

The existing zero-delta tiny training smoke ran on the 64-sample medium subset.

- steps: 3
- seed: 0
- lr: `1e-5`
- sample count: 64
- graph-shape groups: one group, `N=384`, `F=11`
- normalized loss sequence:
  `1.25570560 -> 1.25229871 -> 1.24930692 -> 1.24674785`
- final raw DeltaT MSE: `5.04401186e-03`
- final recovered temperature MSE: `5.04402956e-03`
- repeatability: pass
- checkpoint saved: false
- log file written: false

This checks interface stability only and is not a formal training experiment.

## Train/Valid Smoke

The train/valid smoke used train-only normalization.

- train samples: 48
- valid samples: 8
- ignored diagnostic/test samples: 8
- graph-shape groups: one train group and one valid group, `N=384`, `F=11`
- steps: 3
- seed: 0
- lr: `1e-5`
- repeat runs: 2
- train normalized loss: `1.25483477 -> 1.24614894`
- valid normalized loss: `0.98827767 -> 0.97971886`
- final train raw DeltaT MSE: `5.05786808e-03`
- final valid raw DeltaT MSE: `3.97648336e-03`
- final train recovered temperature MSE: `5.05788764e-03`
- final valid recovered temperature MSE: `3.97648709e-03`
- repeatability: pass
- checkpoint saved: false
- log file written: false

## Validation Metrics Smoke

The validation metrics smoke evaluated train + valid samples only.

- evaluated samples: 56
- label metadata files in evaluated samples: 56/56
- repeatability: pass
- train normalized loss: `1.25483477 -> 1.23027730`
- valid normalized loss: `0.98827767 -> 0.96513909`
- gradient finite check: pass
- per-sample metric rows: 112
- split summaries: 4
- checkpoint saved: false
- output file written: false

Split summary, recovered temperature RMSE:

- train zero-delta baseline mean: `6.55464164e-02`
- train tiny trained prediction mean: `6.65155458e-02`
- valid zero-delta baseline mean: `6.01867628e-02`
- valid tiny trained prediction mean: `6.16370680e-02`

These values are diagnostics for a tiny smoke run. They are not model
performance evidence.

## Default Regression

The old default smoke path still passes:

- `check_heat3d_v1_zero_delta_bridge.py`: pass
- `check_heat3d_v1_validation_metrics_smoke.py`: pass

The default path still reads the legacy supervised-small subset unless a subset
argument is provided.

## Compatibility Note

This stage required additive v1-only compatibility support for the new medium
stage name:

- supervised v1 loading now accepts `physics_label_medium_smoke`

No v0 public entrypoint or model core path was changed.

## Non-Claims

This report does not claim:

- formal benchmark status
- model performance
- OOD generalization
- high-fidelity solver labels
- industrial 3D IC thermal simulation fidelity

## Next Step

The next step is a short 5-epoch training smoke on the same medium subset, still
without formal performance claims or checkpoint submission.
