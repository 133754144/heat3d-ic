# Heat3D v1 Physics-Label Medium 5-Epoch Training Smoke Report

## Purpose

This report records a 5-epoch short training smoke on the 64-sample
`v1_multilayer_bc_eq_physics_label_medium_v2` subset. The goal is to check that
the current v1 train/valid smoke loop remains finite, repeatable, and stable on
the medium physics-label subset.

This is a 5-epoch training smoke / research reference diagnostic only. It is
not a formal training experiment, not model-performance evidence, not a formal
benchmark, and not OOD generalization evidence.

## Training Configuration

- subset: `data/heat3d-thermal-simulation/subsets/v1_multilayer_bc_eq_physics_label_medium_v2`
- route: relative BC features + zero_delta_u_bridge + normalized DeltaT target
- recovery: `T_pred = T_ref + DeltaT_pred`
- train samples: 48
- valid samples: 8
- ignored diagnostic/test samples: 8
- graph-shape groups: one train group and one valid group, `N=384`, `F=11`
- epochs: 5
- optimizer updates: 5
- learning rate: `1e-5`
- seed: 0
- repeat runs: 2
- checkpoint saved: false
- log file written: false
- output directory written: false

The smoke uses train-only normalization. Valid and diagnostic/test samples do
not contribute to condition-feature or target normalization statistics.

## Split Summary

Train samples:

- `medium_000` through `medium_047`

Valid samples:

- `medium_048` through `medium_055`

Ignored diagnostic/test samples:

- `medium_056` through `medium_063`

## Loss Trend

Run 0 normalized loss trend:

| step | train loss | valid loss |
|---:|---:|---:|
| 0 | `1.25483477` | `0.98827767` |
| 1 | `1.25155306` | `0.98504245` |
| 2 | `1.24865413` | `0.98218191` |
| 3 | `1.24614894` | `0.97971886` |
| 4 | `1.24400353` | `0.97762978` |
| 5 | `1.24216497` | `0.97586554` |

The smoke observed decreasing train and valid normalized losses over the 5
optimizer updates. This only indicates that the loop receives a finite training
signal in this controlled smoke setting.

## Final Metrics

Final run 0 metrics:

- train raw DeltaT MSE: `5.04169986e-03`
- valid raw DeltaT MSE: `3.96084413e-03`
- train recovered temperature MSE: `5.04171569e-03`
- valid recovered temperature MSE: `3.96081898e-03`
- finite check: pass
- shape check: pass
- gradient finite check: pass

Repeatability:

- repeat runs: 2
- max train loss delta: `0.000000e+00`
- max valid loss delta: `0.000000e+00`
- max grad norm delta: `0.000000e+00`
- repeatability smoke: pass

## Baseline Context

The existing validation metrics smoke also ran on this subset. Its split
summary reported recovered temperature RMSE diagnostics for zero-delta baseline
and tiny trained prediction. Those metrics remain smoke diagnostics only and are
not a formal baseline/model comparison.

## Non-Claims

This report does not claim:

- formal model performance
- formal benchmark results
- OOD generalization
- high-fidelity solver labels
- industrial 3D IC thermal simulation validity

## Next Step

The next safe step is to decide whether to run a controlled longer training
smoke or to improve solver/label diagnostics before any formal comparison
protocol. Any model comparison should require an explicit protocol before being
reported as more than smoke diagnostics.
