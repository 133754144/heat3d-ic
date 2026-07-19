# Gate 6Q V42/V43/V44 Training Preparation

Status: `prepared_not_started`. No e600 run, optimizer update, checkpoint, or
formal result was created during this preparation.

## Frozen common contract

All three candidates inherit V38 and retain its clean dataset/split, model
capacity, graph construction, seed0 model/batch/graph seeds, B28 training,
B32 validation/prediction, AdamW warmup-cosine optimizer, epoch-wise batch
regrouping, `r2r_only` edge masking at `p=0.05`, e600 budget, and native
shape-scale checkpoint contract. `init_checkpoint` is explicitly null.
Training uses only `train`; selection uses only `valid_iid`.
`test_iid`, all hard roles, and `sealed_iid` are forbidden.

## Single-variable candidates

- V42 changes only two objective formulas. The raw term is batch-global
  CV-weighted normalized SSE:
  `Σ CV·(ŷ-y)² / Σ CV·y²`. The log-scale sample weight is
  `clip(s_true² / mean_train(s_true²), 0.25, 4.0)` and is reduced as a
  normalized weighted mean. The four scalar weights remain
  `1.5/0.5/1.0/1.0`.
- V43 retains the V38 objective and adds ten raw-input-only XY features to the
  global scale-head input. Their standardizer is fit on train only. They do
  not enter shape, Global FiLM, or decoder paths.
- V44 retains V43 and adds only a shared regional MLP followed by mean,
  source-power-weighted, and control-volume-weighted aggregation. Its
  scale-pooling residual output is zero initialized; pooling is not
  softmax-only.

Resolved scientific differences are exactly:

- V42 vs V38:
  `loss.native_raw_loss_mode`,
  `loss.native_log_scale_weight_mode`.
- V43 vs V38:
  `model.scale_context_mode`,
  `model.scale_context_feature_names`.
- V44 vs V43:
  `model.scale_deepsets_mode`.

The expected parameter increments are 0 for V42, 640 for V43, and 28,896 for
V44 relative to V43. A no-update B2/N27 fixture reproduced these increments
exactly, found finite forward/backward values in all trainable groups, and
confirmed the V44 residual output parameters start at zero.

## No-update B28/N1024 memory smoke

Use the largest candidate as the upper-bound single-batch smoke. This command
does one forward/backward pass, applies no optimizer update, and writes no
checkpoint or result:

```bash
XLA_PYTHON_CLIENT_PREALLOCATE=false python scripts/smoke_heat3d_v5_gate6q_single_batch.py \
  --config configs/heat3d_v5/generated/V4P5_44_gate6q_xy_deepsets_e600.yaml \
  --batch-size 28 --grid 8,8,16
```

## Manual e600 launch commands

These commands are prepared for explicit manual use and were not executed:

```bash
python scripts/run_heat3d_v4_config.py --config configs/heat3d_v5/generated/V4P5_42_gate6q_objective_only_e600.yaml
python scripts/run_heat3d_v4_config.py --config configs/heat3d_v5/generated/V4P5_43_gate6q_xy_scale_features_e600.yaml
python scripts/run_heat3d_v4_config.py --config configs/heat3d_v5/generated/V4P5_44_gate6q_xy_deepsets_e600.yaml
```
