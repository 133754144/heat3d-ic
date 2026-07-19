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

- V42 changes only two objective formulas. The raw term is unweighted
  point-level SSE, reduced as
  `mean_batch_points((ŷ-y)²) / mean_train_points(y²)`. Its target-energy
  denominator is fit once from all train points and is fixed across batches,
  so it matches point-global weighting without batch-denominator drift.
  The log-scale sample weight is
  `clip(s_true² / mean_train(s_true²), 0.25, 4.0)` and is reduced as a
  normalized weighted mean. The four scalar weights remain
  `1.5/0.5/1.0/1.0`.
- V43 retains the V38 objective and adds ten raw-input-only XY features to the
  global scale-head input. Their standardizer is fit on train only. They do
  not enter shape, Global FiLM, or decoder paths.
- V44 retains V43 and adds only a source/volume-aware latent DeepSets module:
  a shared regional-latent MLP followed by mean, source-power-weighted, and
  control-volume-weighted aggregation. Every physical node's source and
  volume are divided across its valid P2R degree; partition-of-unity and total
  source/volume conservation are mandatory. The module does not claim
  explicit regional XY physics. Its residual output is zero initialized and
  pooling is not softmax-only.

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

## Real train-only B28/N1024 update smoke

Each candidate is checked with one real-train B28 forward/backward/AdamW
update at `MEM_FRACTION=0.85`. The smoke loads only train labels, writes no
checkpoint, and does not start the e600 loop. Remote results and peak memory
are recorded in the JSON closeout after execution.

```bash
MEM_FRACTION=0.85 XLA_PYTHON_CLIENT_MEM_FRACTION=0.85 \
  python scripts/smoke_heat3d_v5_gate6q_real_train_update.py --config <CONFIG>
```

## Manual e600 launch commands

These commands are prepared for explicit manual use and were not executed:

```bash
python scripts/run_heat3d_v4_config.py --config configs/heat3d_v5/generated/V4P5_42_gate6q_objective_only_e600.yaml
python scripts/run_heat3d_v4_config.py --config configs/heat3d_v5/generated/V4P5_43_gate6q_xy_scale_features_e600.yaml
python scripts/run_heat3d_v4_config.py --config configs/heat3d_v5/generated/V4P5_44_gate6q_xy_deepsets_e600.yaml
```
