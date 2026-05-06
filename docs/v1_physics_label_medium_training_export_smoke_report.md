# Heat3D v1 Medium Training Export Smoke Report

## Purpose

This report records a controlled training export smoke for the
`v1_multilayer_bc_eq_physics_label_medium_v2` subset. The goal is to verify
that the current v1 relative-BC, zero-delta bridge, normalized DeltaT route can
produce recovered-temperature predictions in a reproducible file format for
baseline comparison tooling.

This is a controlled training export smoke / trained comparison smoke only. It
is not a formal benchmark, not model-performance evidence, not OOD
generalization evidence, and not high-fidelity solver evidence.

## Runner

Script:

```bash
python3 scripts/run_heat3d_v1_medium_controlled_training_export.py \
  --epochs 5 \
  --lr 1e-5 \
  --seed 0 \
  --output-dir output/heat3d_v1_medium_runs/export_smoke_seed0 \
  --save-predictions
```

Supported arguments:

- `--subset`
- `--epochs`
- `--lr`
- `--seed`
- `--output-dir`
- `--save-predictions`

The output directory must be under ignored `output/`.

Generated ignored artifacts:

- `run_config.json`
- `loss_summary.json`
- `predictions.npz`

No checkpoint is saved.

## Training Contract

Dataset:

```text
data/heat3d-thermal-simulation/subsets/v1_multilayer_bc_eq_physics_label_medium_v2
```

Route:

```text
relative BC features + zero_delta_u_bridge + normalized DeltaT target
```

Training stats:

- train-only normalization
- train split: 48 samples
- valid split: 8 samples
- diagnostic candidate splits ignored for training
- predictions exported for all 64 samples after training

Configuration:

- epochs: `5`
- learning rate: `1e-5`
- seed: `0`

Loss summary:

| metric | initial | final |
|---|---:|---:|
| train normalized DeltaT loss | `1.25483477e+00` | `1.24216497e+00` |
| valid normalized DeltaT loss | `9.88277674e-01` | `9.75865543e-01` |

Final smoke metrics:

| metric | value |
|---|---:|
| train raw DeltaT MSE | `5.04169986e-03` |
| valid raw DeltaT MSE | `3.96084413e-03` |
| train recovered temperature MSE | `5.04171569e-03` |
| valid recovered temperature MSE | `3.96081898e-03` |
| gradient finite check | `true` |
| prediction sample count | `64` |

## Prediction Export Format

`predictions.npz` stores recovered-temperature predictions. Each array key is a
sample id, for example:

```text
medium_000
medium_001
...
medium_063
```

Each value is the recovered temperature prediction `T_pred`, not normalized
DeltaT. This lets the baseline comparison script consume predictions directly
without reconstructing model internals.

## Comparison Command

Command:

```bash
python3 scripts/compare_heat3d_v1_medium_baselines.py \
  --trained-predictions output/heat3d_v1_medium_runs/export_smoke_seed0/predictions.npz \
  --output-json output/heat3d_v1_medium_runs/export_smoke_seed0/baseline_comparison.json
```

The comparison script successfully read `predictions.npz` and wrote the JSON
summary under ignored `output/`.

## Overall Zero-Delta vs Trained Summary

| predictor | n | mean T RMSE | mean T MAE | mean max abs | mean peak T error | mean hotspot distance |
|---|---:|---:|---:|---:|---:|---:|
| zero_delta | `64` | `6.51538666e-02` | `3.82510808e-02` | `3.92192729e-01` | `3.92192729e-01` | `7.37971023e-03` |
| trained_prediction | `64` | `6.67595914e-02` | `5.08418870e-02` | `3.58022389e-01` | `2.90197073e-01` | `5.26974054e-03` |

In this 5-epoch smoke, the trained prediction does not improve mean recovered
temperature RMSE/MAE over zero_delta overall, while max/peak/hotspot diagnostics
move lower. These observations must remain diagnostic only.

## Split-Wise Summary

| split | predictor | n | mean T RMSE | mean T MAE | mean max abs | mean peak T error | mean hotspot distance |
|---|---|---:|---:|---:|---:|---:|---:|
| train | zero_delta | `48` | `6.55463677e-02` | `3.84876624e-02` | `3.99655061e-01` | `3.99655061e-01` | `7.33466603e-03` |
| train | trained_prediction | `48` | `6.69243205e-02` | `5.12085306e-02` | `3.64673822e-01` | `2.95902925e-01` | `5.40902035e-03` |
| valid | zero_delta | `8` | `6.01870855e-02` | `3.50358703e-02` | `3.60984203e-01` | `3.60984203e-01` | `7.61922496e-03` |
| valid | trained_prediction | `8` | `6.20069545e-02` | `4.80520950e-02` | `3.27361461e-01` | `2.54348155e-01` | `4.81071592e-03` |

Candidate splits were also printed by the comparison script, but they remain
observational diagnostics only and do not support OOD claims.

## Condition-Wise Comparison

The comparison script produced condition-wise zero_delta vs trained summaries
for:

- `source_pattern_tag`
- `k_region_mode`
- `k_field_mode`
- `stack_template`
- `bc_category`

This confirms that exported predictions can be joined with sample metadata for
condition-wise diagnostic summaries.

## Non-Claims

This report does not claim:

- formal benchmark status
- model performance
- OOD generalization
- high-fidelity thermal labels
- industrial 3D IC simulation validity

## Next Step

The next practical step is a remote/runbook pass: define the exact command
sequence and ignored artifact layout for a longer controlled run, then reuse
the same `predictions.npz` comparison interface for zero_delta vs trained
diagnostics.
