# Heat3D v1 Physics-Label Medium Controlled Training Report

## Purpose

This report records a 30-epoch controlled training smoke on the 64-sample
`v1_multilayer_bc_eq_physics_label_medium_v2` subset and a matching validation
metrics smoke comparing `zero_delta` with a short trained prediction.

This is a controlled training smoke / baseline-comparison draft only. It is not
a formal benchmark, not model-performance evidence, not OOD generalization
evidence, and not high-fidelity solver evidence.

## Commands

Pre-push checks on the medium subset:

```bash
python3 scripts/check_heat3d_v1_label_diagnostics.py --subset data/heat3d-thermal-simulation/subsets/v1_multilayer_bc_eq_physics_label_medium_v2
python3 scripts/check_heat3d_v1_zero_delta_bridge.py --subset data/heat3d-thermal-simulation/subsets/v1_multilayer_bc_eq_physics_label_medium_v2
python3 scripts/check_heat3d_v1_validation_metrics_smoke.py --subset data/heat3d-thermal-simulation/subsets/v1_multilayer_bc_eq_physics_label_medium_v2
```

Controlled training smoke:

```bash
python3 scripts/check_heat3d_v1_small_train_valid_smoke.py \
  --subset data/heat3d-thermal-simulation/subsets/v1_multilayer_bc_eq_physics_label_medium_v2 \
  --epochs 30 \
  --lr 1e-5 \
  --seed 0 \
  --runs 2 \
  --report-every 5
```

Validation metrics smoke:

```bash
python3 scripts/check_heat3d_v1_validation_metrics_smoke.py \
  --subset data/heat3d-thermal-simulation/subsets/v1_multilayer_bc_eq_physics_label_medium_v2 \
  --steps 30 \
  --lr 1e-5 \
  --seed 0 \
  --repeat-runs 2
```

## Training Configuration

- subset: `v1_multilayer_bc_eq_physics_label_medium_v2`
- route: relative BC features + zero_delta bridge + normalized DeltaT target
- train samples: 48
- valid samples: 8
- ignored diagnostic/test samples: 8
- graph-shape groups: one train group and one valid group, `N=384`, `F=11`
- epochs: 30
- optimizer updates: 30
- learning rate: `1e-5`
- seed: 0
- repeat runs: 2
- checkpoint saved: false
- log file written: false
- output directory written: false

The smoke used train-only normalization. Valid and diagnostic/test samples did
not contribute to condition-feature or target normalization statistics.

## Controlled Training Result

Run 0 normalized loss trend:

| step | train loss | valid loss |
|---:|---:|---:|
| 0 | `1.25483477` | `0.98827767` |
| 5 | `1.24216497` | `0.97586554` |
| 10 | `1.23605490` | `0.97023481` |
| 15 | `1.23261607` | `0.96719760` |
| 20 | `1.23027730` | `0.96513909` |
| 25 | `1.22847807` | `0.96353477` |
| 30 | `1.22698319` | `0.96217823` |

Final run 0 metrics:

- train raw DeltaT MSE: `4.98007983e-03`
- valid raw DeltaT MSE: `3.90528957e-03`
- train recovered temperature MSE: `4.98009054e-03`
- valid recovered temperature MSE: `3.90528864e-03`
- finite check: pass
- shape check: pass
- gradient finite check: pass

Repeatability:

- repeat runs: 2
- max train loss delta: `0.000000e+00`
- max valid loss delta: `0.000000e+00`
- max grad norm delta: `0.000000e+00`
- repeatability smoke: pass

## Zero-Delta Comparison Draft

The 30-step validation metrics smoke evaluated train + valid samples only.

- evaluated samples: 56
- label metadata files in evaluated samples: 56/56
- repeatability: pass
- train normalized loss: `1.25483477 -> 1.22698319`
- valid normalized loss: `0.98827767 -> 0.96217823`
- gradient finite check: pass
- per-sample metric rows: 112
- split summaries: 4

Recovered temperature RMSE split summary:

| split | predictor | mean recovered T RMSE |
|---|---|---:|
| train | zero_delta_baseline | `6.55464164e-02` |
| train | tiny_trained_prediction | `6.63992850e-02` |
| valid | zero_delta_baseline | `6.01867628e-02` |
| valid | tiny_trained_prediction | `6.15325886e-02` |

Recovered temperature max/peak diagnostics improved for the short trained
prediction in this smoke, while mean RMSE was slightly higher than the
zero-delta baseline on both train and valid splits. This should be treated as a
useful controlled diagnostic: the loop is stable and repeatable, but the current
short trained prediction is not a formal improvement claim.

## Non-Claims

This report does not claim:

- formal model performance
- formal benchmark results
- OOD generalization
- high-fidelity solver labels
- industrial 3D IC thermal simulation validity

## Next Step

The next step is to refine the baseline/model comparison protocol before longer
training runs. In particular, the protocol should separate mean-error,
max-error, peak-temperature, and hotspot metrics instead of reducing the result
to a single "better or worse" conclusion.
