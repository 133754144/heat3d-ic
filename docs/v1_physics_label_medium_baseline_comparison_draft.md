# Heat3D v1 Physics-Label Medium Baseline Comparison Draft

## Purpose

This document records the current baseline comparison state for the
`v1_multilayer_bc_eq_physics_label_medium_v2` subset after the 30-epoch
controlled training smoke. It separates what is already measured from what is
still pending.

This is a baseline comparison draft only. It is not a formal benchmark, not
model-performance evidence, not OOD generalization evidence, and not
high-fidelity solver evidence.

## Current Controlled Training Summary

Subset and route:

- subset: `v1_multilayer_bc_eq_physics_label_medium_v2`
- train samples: 48
- valid samples: 8
- ignored diagnostic/test samples: 8
- route: relative BC features + zero_delta bridge + normalized DeltaT target
- recovery: `T_pred = T_ref + DeltaT_pred`
- graph-shape groups: one train group and one valid group, `N=384`, `F=11`

Training smoke configuration:

- epochs: 30
- optimizer updates: 30
- learning rate: `1e-5`
- seed: 0
- repeat runs: 2
- checkpoint saved: false
- log file written: false
- output directory written: false

Loss trend:

| step | train normalized loss | valid normalized loss |
|---:|---:|---:|
| 0 | `1.25483477` | `0.98827767` |
| 5 | `1.24216497` | `0.97586554` |
| 10 | `1.23605490` | `0.97023481` |
| 15 | `1.23261607` | `0.96719760` |
| 20 | `1.23027730` | `0.96513909` |
| 25 | `1.22847807` | `0.96353477` |
| 30 | `1.22698319` | `0.96217823` |

Repeatability:

- max train loss delta: `0.000000e+00`
- max valid loss delta: `0.000000e+00`
- max grad norm delta: `0.000000e+00`
- repeatability smoke: pass

## Known Zero-Delta vs Trained Summary

The 30-step validation metrics smoke evaluated train + valid samples only.

Mean recovered temperature RMSE:

| split | zero_delta baseline | trained prediction |
|---|---:|---:|
| train | `6.55464164e-02` | `6.63992850e-02` |
| valid | `6.01867628e-02` | `6.15325886e-02` |

Mean raw `DeltaT` MSE:

| split | zero_delta baseline | trained prediction |
|---|---:|---:|
| train | `5.54010714e-03` | `4.98008001e-03` |
| valid | `4.12240883e-03` | `3.90528945e-03` |

Mean recovered temperature MAE:

| split | zero_delta baseline | trained prediction |
|---|---:|---:|
| train | `3.84877440e-02` | `5.05560007e-02` |
| valid | `3.50357890e-02` | `4.73858615e-02` |

Mean max absolute error:

| split | zero_delta baseline | trained prediction |
|---|---:|---:|
| train | `3.99655024e-01` | `3.64636739e-01` |
| valid | `3.60984802e-01` | `3.27098846e-01` |

Mean peak temperature error:

| split | zero_delta baseline | trained prediction |
|---|---:|---:|
| train | `3.99655024e-01` | `2.98206965e-01` |
| valid | `3.60984802e-01` | `2.57610321e-01` |

Mean hotspot coordinate distance:

| split | zero_delta baseline | trained prediction |
|---|---:|---:|
| train | `6.97388876e-03` | `5.62738534e-03` |
| valid | `6.98316154e-03` | `5.40181609e-03` |

Mean top-k hotspot overlap:

| split | zero_delta baseline | trained prediction |
|---|---:|---:|
| train | `2.08333333e-02` | `0.00000000e+00` |
| valid | `7.50000000e-02` | `0.00000000e+00` |

## Interpretation Boundaries

Current observed pattern:

- normalized train and valid losses decreased over the controlled smoke;
- raw `DeltaT` MSE improved versus `zero_delta` on train and valid;
- recovered temperature max and peak errors improved versus `zero_delta`;
- hotspot coordinate distance improved versus `zero_delta`;
- recovered temperature RMSE and MAE were slightly worse than `zero_delta`;
- top-k hotspot overlap was worse in this run.

This mixed result is useful for protocol design. It does not support an overall
model improvement claim.

## Condition-Wise Status

The medium subset contains metadata categories needed for condition-wise
summary:

- `source_pattern_tag`
- `k_region_mode`
- `k_field_mode`
- `stack_template`
- `bc_category`

Condition-wise zero-delta and trained summaries are not yet fixed in a
dedicated comparison script. Because the controlled training smoke did not save
predictions or a checkpoint, this draft does not fabricate condition-wise
trained metrics.

Required next condition-wise outputs:

- per-condition sample count;
- zero-delta mean-field / peak / hotspot / tail metrics;
- trained prediction mean-field / peak / hotspot / tail metrics;
- train/valid/test-candidate separation;
- explicit warning for small or diagnostic-only groups.

## Pending Items

- dedicated baseline comparison script;
- p95 absolute error;
- `DeltaT` RMSE and MAE columns;
- condition-wise aggregation;
- optional diagnostic-only test split reporting;
- machine-readable metrics output, if later approved;
- fixed protocol for longer 50/100 epoch controlled runs.

## Non-Claims

This draft does not claim:

- formal model performance;
- formal benchmark results;
- OOD generalization;
- high-fidelity solver labels;
- industrial 3D IC thermal simulation validity.

## Next Step

Implement a dedicated baseline comparison script that recomputes the trained
prediction in-memory under fixed settings, reports condition-wise metrics, and
does not save checkpoints or committed output artifacts. After that, run a
50/100 epoch controlled smoke only if the comparison protocol is stable.
