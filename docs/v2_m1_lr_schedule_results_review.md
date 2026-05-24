# Heat3D v2 M1 LR Schedule Results Review

本文整理 SSH WSL 上已有 M1 mini-batch e50 runs 的只读结果。本文不是 formal benchmark，也不声称 V2 稳定优于 V1；当前结论只适用于 `medium1024_gapA_full1024_v2`、M1、seed0 的 research-stage controlled runs。

## Runs

| run_name | LR setting | schedule | notes |
|---|---:|---|---|
| `m1_batch_e50_seed0` | `1e-3` | constant | 原 M1 e50 baseline；远程目录缺 `train.log`，但 `loss_summary.json` / predictions / diagnostics 存在。 |
| `m1_batch_e50_lr3e4_seed0` | `3e-4` | constant | 当前 best-valid 综合最优 constant-LR run。 |
| `m1_batch_e50_lr1e4_seed0` | `1e-4` | constant | best epoch 后移，但 valid/hotspot 指标较差。 |
| `m1_batch_e50_lr3e4_decay_e5_to1e4_seed0` | `3e-4 -> 1e-4 @ epoch 5` | second_stage | 本轮新增读取结果；补跑了只读 field-shape diagnostics。 |

## File And Selection Checks

四个 run 均存在 `run_config.json`、`loss_summary.json`、`predictions.npz`、`best_predictions.npz`、`field_shape_diagnostics_best.json` 和 `field_shape_diagnostics_final.json`。其中 `m1_batch_e50_seed0` 远程目录缺 `train.log`，其余三个 run 有 `train.log` 且正常结束。

best selection 逻辑已从 `epoch_history` 重新计算 `min(valid_loss)`：

| run_name | summary_best_epoch | recomputed_best_epoch | selection_ok |
|---|---:|---:|---|
| `m1_batch_e50_seed0` | 2 | 2 | true |
| `m1_batch_e50_lr3e4_seed0` | 2 | 2 | true |
| `m1_batch_e50_lr1e4_seed0` | 20 | 20 | true |
| `m1_batch_e50_lr3e4_decay_e5_to1e4_seed0` | 20 | 20 | true |

没有发现 best selection bug。`report_every` 只影响日志显示，不影响每个 epoch 的 validation 和 best-valid selection。

## Best Checkpoint Comparison

| run_name | LR setting | best_epoch | valid_loss | raw_deltaT_mse | hotspot_mae | field_variance_ratio | centered_spatial_correlation | amplitude_ratio | peak_abs_error | top_k_overlap | bg_bias | pn_over_ratio |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `m1_batch_e50_seed0` | `1e-3` | 2 | 9.27328050e-01 | 1.37366808e-03 | 6.37404770e-02 | 2.85334270e+00 | 7.58506019e-01 | 8.72976332e-01 | 1.06097621e-01 | 6.83007813e-01 | 1.21050868e-02 | 9.90977764e-01 |
| `m1_batch_e50_lr3e4_seed0` | `3e-4` | 2 | 8.20047617e-01 | 1.03067153e-03 | 4.21602204e-02 | 1.06412495e+00 | 7.56950744e-01 | 7.89703499e-01 | 9.84493023e-02 | 6.33398438e-01 | 1.83125418e-02 | 1.00000000e+00 |
| `m1_batch_e50_lr1e4_seed0` | `1e-4` | 20 | 1.03246975e+00 | 1.60945079e-03 | 6.16895109e-02 | 4.06781688e+00 | 8.56934064e-01 | 1.34220789e+00 | 9.20868469e-02 | 6.34960938e-01 | 1.13554373e-02 | 9.83587384e-01 |
| `m1_batch_e50_lr3e4_decay_e5_to1e4_seed0` | `3e-4 -> 1e-4 @ e5` | 20 | 9.82042193e-01 | 1.52418378e-03 | 6.06208406e-02 | 3.84184602e+00 | 8.59927283e-01 | 1.27231109e+00 | 9.21632963e-02 | 6.39453125e-01 | 1.08889006e-02 | 9.83262241e-01 |

## Final Checkpoint Comparison

| run_name | LR setting | final_valid_loss | final_raw_deltaT_mse | final_hotspot_mae | field_variance_ratio | centered_spatial_correlation | amplitude_ratio | peak_abs_error | top_k_overlap | bg_bias | pn_over_ratio | final/best degradation |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `m1_batch_e50_seed0` | `1e-3` | 3.14366198e+00 | 5.30293537e-03 | 1.08159006e-01 | 1.01535904e+01 | 8.71330393e-01 | 1.53250091e+00 | 9.36106906e-02 | 6.44140625e-01 | 2.48377025e-02 | 9.96423483e-01 | +2.21633393e+00 / 3.39x |
| `m1_batch_e50_lr3e4_seed0` | `3e-4` | 3.16040659e+00 | 5.32081956e-03 | 1.09024994e-01 | 1.00846010e+01 | 8.77625713e-01 | 1.49852529e+00 | 8.39786101e-02 | 6.58984375e-01 | 2.51189210e-02 | 9.88336921e-01 | +2.34035897e+00 / 3.85x |
| `m1_batch_e50_lr1e4_seed0` | `1e-4` | 2.86302805e+00 | 4.82248934e-03 | 1.05052084e-01 | 9.64103656e+00 | 8.70452496e-01 | 1.52800992e+00 | 9.57820309e-02 | 6.40039063e-01 | 2.23296043e-02 | 9.12587047e-01 | +1.83055830e+00 / 2.77x |
| `m1_batch_e50_lr3e4_decay_e5_to1e4_seed0` | `3e-4 -> 1e-4 @ e5` | 2.90065265e+00 | 4.88991477e-03 | 1.05642118e-01 | 9.77611372e+00 | 8.70816536e-01 | 1.52681339e+00 | 9.50174482e-02 | 6.38867188e-01 | 2.24772282e-02 | 9.17767227e-01 | +1.91861045e+00 / 2.95x |

## Schedule Comparison

The `3e-4 -> 1e-4 @ epoch 5` schedule did not beat constant `3e-4` on best-valid metrics:

- best valid_loss: `0.9820` vs constant `3e-4` `0.8200`, worse by about `19.8%`.
- best raw_deltaT_mse: `0.001524` vs `0.001031`, worse by about `47.9%`.
- best hotspot_mae: `0.06062` vs `0.04216`, worse by about `43.8%`.

It did improve over constant `1e-4` on best valid_loss, raw_deltaT_mse, and hotspot_mae, but only modestly:

- best valid_loss: `0.9820` vs `1.0325`.
- best raw_deltaT_mse: `0.001524` vs `0.001609`.
- best hotspot_mae: `0.06062` vs `0.06169`.

So the second-stage run is not a new best-valid model; it is a stabilization-style tradeoff between constant `3e-4` and constant `1e-4`.

## Early-Best And Final Degradation

The schedule did mitigate early-best behavior:

- constant `3e-4`: best epoch `2`;
- constant `1e-4`: best epoch `20`;
- `3e-4 -> 1e-4 @ epoch 5`: best epoch `20`.

However, this looks closer to the constant `1e-4` trajectory than to a strictly better version of constant `3e-4`.

The schedule also reduced final degradation relative to constant `3e-4`:

- constant `3e-4`: final/best ratio `3.85x`, absolute degradation `+2.3404`;
- second-stage: final/best ratio `2.95x`, absolute degradation `+1.9186`.

This is still large. Final loss remains far worse than best-valid loss, so final-vs-best behavior remains unresolved.

## Field-Shape Interpretation

The schedule improved some best-checkpoint field-shape indicators versus constant `3e-4`:

- centered_spatial_correlation: `0.8599` vs `0.7570`, better;
- peak_abs_error: `0.09216` vs `0.09845`, better;
- top_k_overlap: `0.6395` vs `0.6334`, slightly better.

But it worsened amplitude/variance control versus constant `3e-4`:

- field_variance_ratio: `3.8418` vs `1.0641`, much farther from `1`;
- amplitude_ratio: `1.2723` vs `0.7897`; constant `3e-4` under-amplifies, while the schedule over-amplifies.

Compared with constant `1e-4`, the schedule is very similar: slightly better correlation, amplitude ratio, field variance ratio, hotspot_mae, and valid_loss; peak_abs_error is essentially tied. It does not produce a clear field-shape breakthrough.

## Hotspot And Peak

For hotspot_mae, constant `3e-4` remains clearly best:

- constant `3e-4`: `0.04216`;
- second-stage: `0.06062`;
- constant `1e-4`: `0.06169`;
- constant `1e-3`: `0.06374`.

For peak_abs_error, constant `1e-4` and second-stage are best and nearly tied:

- constant `1e-4`: `0.09209`;
- second-stage: `0.09216`;
- constant `3e-4`: `0.09845`.

This confirms the existing mismatch: valid_loss/hotspot_mae favors constant `3e-4`, while some peak/spatial diagnostics favor lower-LR behavior.

## Low-Temperature Overprediction

The schedule improves low-temperature overprediction indicators relative to constant `3e-4`:

- best bg_bias: `0.01089` vs `0.01831`;
- best pn_over_ratio: `0.9833` vs `1.0000`;
- final bg_bias: `0.02248` vs `0.02512`;
- final pn_over_ratio: `0.9178` vs `0.9883`.

It is roughly tied with, and slightly worse than, constant `1e-4` at final:

- final bg_bias: `0.02248` vs `0.02233`;
- final pn_over_ratio: `0.9178` vs `0.9126`.

Lower LR behavior helps low-temperature overprediction, but does not solve it and comes with worse best-valid/hotspot metrics.

## Recommendation

Do not treat `3e-4 -> 1e-4 @ epoch 5` as an improvement over constant `3e-4`. It is useful evidence that reducing LR after the early phase can move best_epoch later and reduce final degradation, but the drop to `1e-4` is too aggressive for best-valid and hotspot performance.

The next controlled LR test should be a gentler second-stage decay:

- base lr: `3e-4`;
- second_stage_epoch: `5`;
- second_stage_lr: `2e-4`.

Rationale:

- It keeps the same minimal controlled design as the current run.
- It tests whether milder decay preserves the constant `3e-4` best-valid/hotspot strength while improving late stabilization.
- It avoids adding warmup/cosine hyperparameters before the simple decay direction is understood.

Warmup/cosine remains a reasonable next family if `3e-4 -> 2e-4 @ epoch 5` still shows early-best or final degradation. A cosine schedule may eventually be better, but the immediate evidence supports one gentler second-stage decay before moving to a broader schedule family.
