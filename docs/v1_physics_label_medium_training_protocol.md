# Heat3D v1 Physics-Label Medium Controlled Training Protocol

## Purpose

This document defines a controlled training smoke protocol for the
`v1_multilayer_bc_eq_physics_label_medium_v2` subset. The goal is to compare
the existing zero-delta baseline with a short trained prediction under fixed,
auditable settings.

This is a controlled training smoke / baseline-comparison draft only. It is not
a formal benchmark, not model-performance evidence, not OOD generalization
evidence, and not high-fidelity solver evidence.

## Dataset

- subset: `data/heat3d-thermal-simulation/subsets/v1_multilayer_bc_eq_physics_label_medium_v2`
- sample count: 64
- split: `train=48`, `valid=8`, `test_id=4`,
  `test_ood_bc_candidate=2`, `test_ood_stack_candidate=2`
- label source: reference solver v2 research labels with `label_meta.json`
- generated data location: ignored `data/`

The `test_id`, `test_ood_bc_candidate`, and `test_ood_stack_candidate` splits
are diagnostic candidates only. They should not be used to claim OOD
generalization.

## Modeling Route

Use the existing v1 supervised route:

- external task: `coords + condition_features -> target_temperature`
- condition feature view: relative BC features
- bridge: `zero_delta_u_bridge`
- target: `DeltaT = T - T_ref`
- normalized loss: MSE on normalized `DeltaT`
- recovery: `T_pred = T_ref + DeltaT_pred`

`temperature.npy` is the supervised target. It must not be used as an
inference-time input. `label_meta.json` is diagnostic metadata and must not be
used as a model input.

## Baseline

The baseline predictor is `zero_delta`:

- `DeltaT_pred = 0`
- `T_pred = T_ref`

This baseline is compared with the short trained prediction in validation
metrics smoke. The comparison is diagnostic only.

## First Controlled Training Setting

Recommended first setting:

- epochs: 20 to 30
- seed: 0
- learning rate: `1e-5`
- repeat runs: 2
- normalization: train split only
- batching: graph-shape grouped batching
- checkpoint saved: false
- log file written: false
- output artifact committed: false

The train split provides condition-feature and target normalization statistics.
Valid and diagnostic candidate splits must not contribute to normalization
statistics.

## Metrics

Report split-wise and per-sample diagnostics where available:

- normalized `DeltaT` loss
- raw `DeltaT` MSE/RMSE/MAE
- recovered temperature RMSE/MAE/max absolute error
- peak temperature absolute error
- hotspot coordinate distance
- top-k hotspot overlap
- zero-delta baseline summary
- short trained prediction summary

Candidate test splits may be inspected later as diagnostic candidates, but this
protocol does not treat them as formal OOD tests.

## Required Checks

Before controlled training:

- label diagnostics pass on the medium subset
- zero-delta bridge smoke passes on the medium subset
- validation metrics smoke passes on the medium subset
- old default smoke remains compatible

After controlled training:

- train and valid loss sequences are finite
- gradient norm is finite
- repeatability passes within the configured tolerance
- no checkpoint/log/output artifact is committed

## Non-Claims

This protocol does not claim:

- formal benchmark status
- model performance
- OOD generalization
- high-fidelity thermal labels
- industrial 3D IC simulation validity

## Next Step

After this protocol runs cleanly, the next step is to decide whether to refine
the baseline/model comparison protocol or to further improve solver and label
diagnostics before longer training runs.
