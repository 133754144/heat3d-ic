# Heat3D v1 Physics-Label Medium Baseline Comparison Protocol

## Purpose

This document defines the baseline comparison protocol for the
`v1_multilayer_bc_eq_physics_label_medium_v2` subset. The goal is to separate
mean-field, peak, hotspot, tail, split-wise, and condition-wise diagnostics
when comparing `zero_delta` against the current v1 graph/RIGNO path.

This is a baseline comparison protocol / diagnostic draft only. It is not a
formal benchmark, not model-performance evidence, not OOD generalization
evidence, and not high-fidelity solver evidence.

## Dataset

- subset: `v1_multilayer_bc_eq_physics_label_medium_v2`
- samples: 64
- split: `train=48`, `valid=8`, `test_id=4`,
  `test_ood_bc_candidate=2`, `test_ood_stack_candidate=2`
- label source: reference solver v2 research labels with `label_meta.json`
- generated sample location: ignored `data/`

Candidate test splits remain observational only. They must not be used for OOD
claims.

## Compared Predictors

Baseline:

- name: `zero_delta`
- prediction: `DeltaT_pred = 0`
- recovery: `T_pred = T_ref`

Model path:

- name: current v1 graph/RIGNO path
- input: `coords + k_field + q_field + relative BC features`
- bridge: `zero_delta_u_bridge`
- target: normalized `DeltaT = T - T_ref`
- recovery: `T_pred = T_ref + DeltaT_pred`

`label_meta.json` is diagnostics metadata and is not a model input.

## Required Comparison Groups

### 1. Mean-Field Metrics

Report mean-field errors by split and predictor:

- raw `DeltaT` MSE/RMSE/MAE
- recovered temperature RMSE
- recovered temperature MAE

These metrics describe average field-level fit. They are insufficient on their
own because hotspot accuracy can move differently.

### 2. Peak Metrics

Report peak-temperature diagnostics:

- true peak temperature
- predicted peak temperature
- peak temperature absolute error
- peak `DeltaT` error, when available

Peak metrics should be reported separately from RMSE/MAE. If peak diagnostics
improve while mean RMSE worsens, report both facts without claiming overall
improvement.

### 3. Hotspot Metrics

Report hotspot localization diagnostics:

- true hotspot index
- predicted hotspot index
- hotspot coordinate distance
- top-k hottest overlap, if available

Hotspot metrics are especially important for thermal design diagnostics, but
they are still smoke diagnostics until the label generator and benchmark
protocol are validated.

### 4. Tail Metrics

Report tail and worst-case diagnostics:

- recovered temperature max absolute error
- p95 absolute error, if implemented
- worst-sample summary by split

Tail metrics should be interpreted separately from mean-field metrics.

### 5. Split-Wise Summary

At minimum, report:

- train
- valid
- test_id, if explicitly enabled
- test_ood_bc_candidate, observational only
- test_ood_stack_candidate, observational only

The default comparison should evaluate train and valid only. Test candidate
splits require explicit opt-in and must remain diagnostic candidates.

### 6. Condition-Wise Summary

When sample metadata is available, aggregate diagnostics by:

- `source_pattern_tag`
- `k_region_mode`
- `k_field_mode`
- `stack_template`
- `bc_category`

Condition-wise summaries should include sample counts and should avoid
overinterpreting small groups, especially the held-out candidate categories.

## Decision Rules

- Decreasing train loss alone is insufficient.
- Valid split comparison against `zero_delta` is required before discussing any
  trained predictor behavior.
- Candidate splits are observational only and must not be called OOD
  generalization evidence.
- If mean RMSE is worse but peak or hotspot diagnostics improve, report the
  tradeoff explicitly and do not claim overall improvement.
- If a model checkpoint or prediction artifact is not saved, do not invent
  condition-wise trained metrics. Recompute them under a fixed script or mark
  them pending.
- Any future longer run should record the exact seed, learning rate, epoch
  count, normalization policy, split source, and whether checkpoint/log/output
  artifacts were generated.

## Current Script Coverage

Existing validation metrics smoke reports:

- raw `DeltaT` MSE
- recovered temperature RMSE/MAE/max absolute error
- peak temperature absolute error
- hotspot index and coordinate distance
- top-k hotspot overlap
- train/valid split summaries
- zero-delta and short trained predictor rows

Still needed for a complete comparison script:

- p95 absolute error
- explicit `DeltaT` RMSE/MAE columns
- condition-wise aggregation by source/k/stack/BC metadata
- optional diagnostic-only test split opt-in with strict non-claim labeling
- persistent machine-readable metrics output, if later approved

## Non-Claims

This protocol does not claim:

- formal benchmark status
- model performance
- OOD generalization
- high-fidelity thermal labels
- industrial 3D IC simulation validity

## Next Step

The next implementation step is a dedicated baseline comparison script that
recomputes trained predictions in-memory under fixed settings and aggregates
metrics by split and condition group without writing checkpoints or committed
outputs.
