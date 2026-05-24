# Heat3D v2 M1 lower-lr results review

## 背景

本轮只读取 SSH WSL 上已经完成的 M1 mini-batch e50 lower-lr 训练结果，并补齐已有 prediction 文件对应的只读 diagnostics。没有训练、没有运行 e50、没有修改模型、loss、optimizer、batch size 或 runner 训练逻辑。

对比对象：

- baseline: `output/heat3d_v2_runs/m1_batch_e50_seed0`, lr=1e-3
- lower-lr A: `output/heat3d_v2_runs/m1_batch_e50_lr3e4_seed0`, lr=3e-4
- lower-lr B: `output/heat3d_v2_runs/m1_batch_e50_lr1e4_seed0`, lr=1e-4

三个 run 都有 `run_config.json`、`loss_summary.json`、`predictions.npz`、`best_predictions.npz`。baseline 没有 `train.log`，两个 lower-lr run 有 `train.log` 且末尾包含 `[done] script complete`。

## 配置一致性

lower-lr 两个 run 均为 M1 mini-batch e50:

- optimizer: AdamW
- epochs: 50
- batch_size / validation_batch_size / prediction_batch_size: 4 / 4 / 4
- gradient_clip_norm: 1.0
- weight_decay: 1.0e-4
- seed: 0
- selection_metric: valid_loss
- train_metrics_schedule: half_and_final
- grad_norm_report_every: 10

baseline lr=1e-3 是早于 train-metrics/grad-norm 显式字段的 run，因此 `loss_summary.json` 中这两个字段为空，但核心模型、optimizer、batch size、loss 与 M1 baseline 对齐。

## Lower-lr 对比

| lr | best_epoch | best_valid_loss | best_valid_raw_deltaT_mse | best_hotspot_mae | best_bg_bias | best_pn_over_ratio | final_valid_loss | final/best ratio | status |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1e-3 | 2 | 9.27328050e-01 | 1.37366808e-03 | 6.37404770e-02 | 1.21050868e-02 | 9.90977764e-01 | 3.14366198e+00 | 3.390 | ok |
| 3e-4 | 2 | 8.20047617e-01 | 1.03067153e-03 | 4.21602204e-02 | 1.83125418e-02 | 1.00000000e+00 | 3.16040659e+00 | 3.854 | ok |
| 1e-4 | 20 | 1.03246975e+00 | 1.60945079e-03 | 6.16895109e-02 | 1.13554373e-02 | 9.83587384e-01 | 2.86302805e+00 | 2.773 | ok |

`lr=3e-4` improves best valid_loss, valid_raw_deltaT_mse, and hotspot MAE versus lr=1e-3, but it does not move best_epoch; the best is still epoch 2, and final degradation is slightly worse.

`lr=1e-4` moves best_epoch to 20 and has lower final/best degradation, but its best valid_loss and best valid_raw_deltaT_mse are worse than lr=1e-3 and lr=3e-4. This looks like a lower-variance / slower-training setting, not a better optimum under the current constant-lr setup.

## Best Selection Check

Selection was recomputed from `epoch_history` using min valid_loss:

| lr | summary best_epoch | recomputed min valid_loss epoch | min raw_deltaT_mse epoch | selection |
| --- | ---: | ---: | ---: | --- |
| 1e-3 | 2 | 2 | 2 | normal |
| 3e-4 | 2 | 2 | 1 | normal |
| 1e-4 | 20 | 20 | 20 | normal |

No selection bug was found. `report_every=5` affects log printing frequency only; validation is recorded every epoch in `epoch_history`.

## Curve Dynamics

Top valid_loss epochs:

- lr=1e-3: 2, 18, 20, 17, 12, 6, 36, 21, 7, 22
- lr=3e-4: 2, 1, 20, 18, 12, 7, 22, 9, 17, 14
- lr=1e-4: 20, 18, 22, 12, 1, 17, 14, 9, 6, 23

Top valid_raw_deltaT_mse epochs:

- lr=1e-3: 2, 18, 20, 17, 6, 12, 36, 21, 7, 22
- lr=3e-4: 1, 2, 20, 18, 12, 7, 9, 22, 17, 14
- lr=1e-4: 20, 18, 22, 1, 12, 17, 6, 9, 14, 8

All three curves are noisy. valid_loss direction changes were 29, 33, and 30 times respectively. lr=1e-4 delays the best epoch but does not improve the best metrics.

## Diagnostics

Lower-lr diagnostics were missing initially and were generated from existing `predictions.npz` / `best_predictions.npz` only. No training was run. Generated files include baseline comparison, error bins, run summary, condition diagnostics, and field-shape diagnostics for final and best predictions.

Best prediction diagnostics:

| lr | DeltaT RMSE | DeltaT MAE | field_variance_ratio | centered_corr | amplitude_ratio | peak_abs_error | top_k_overlap |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1e-3 | 2.94751215e-02 | 1.93511658e-02 | 2.853343 | 0.758506 | 0.872976 | 0.106098 | 0.683008 |
| 3e-4 | 2.84171844e-02 | 1.99657608e-02 | 1.064125 | 0.756951 | 0.789703 | 0.098449 | 0.633398 |
| 1e-4 | 2.32295633e-02 | 1.50051224e-02 | 4.067817 | 0.856934 | 1.342208 | 0.092087 | 0.634961 |

The diagnostics are mixed: lr=1e-4 has better overall best-prediction DeltaT RMSE/MAE, correlation, variance ratio, and peak error, but worse validation loss/raw_deltaT_mse than lr=3e-4 and does not improve top-k overlap versus lr=1e-3. This suggests selection by valid_loss alone is not fully aligned with field-shape diagnostics.

## Recommendation

Do not continue to even lower constant lr such as 5e-5 or 3e-5 as the next step. The data does not support it:

- lr=3e-4 is the best by best valid_loss, raw_deltaT_mse, and hotspot MAE, but best remains epoch 2.
- lr=1e-4 moves best_epoch to 20, but best valid_loss and raw_deltaT_mse degrade.
- All runs still show strong final degradation.

The better next step is not lower lr; it is schedule and selection work:

1. Try warmup + cosine or one-cycle / second-stage lr while keeping M1 batch size and loss fixed.
2. Keep `valid_loss` as the primary selection metric for now, but report `valid_raw_deltaT_mse` and field-shape metrics side by side because lr=1e-4 shows better field-shape diagnostics despite worse valid_loss.
3. Add seed sensitivity before treating lr=3e-4 as robust.

## V2 P0-P7 Status

| Phase | Status | Evidence | Implemented / run | Remains / caveat |
| --- | --- | --- | --- | --- |
| P0 config loader smoke | completed | `rigno/heat3d_v2_config.py`, `scripts/check_heat3d_v2_config_smoke.py`, `docs/v2_config_loader_smoke.md`, commit `fa57d7e` | Draft v2 YAML loader, validation, summary smoke. | Schema is still draft, but P0 target is met. |
| P1 config-to-command dry-run / v1 runner wrapper | completed | `rigno/heat3d_v2_runner_command.py`, `scripts/check_heat3d_v2_config_to_command_smoke.py`, `docs/v2_config_to_runner_dry_run.md`, commit `090b25f` | Config-to-runner command plan and diagnostics command builders; later used to run SSH configs. | Some draft fields remain intentionally unmapped. |
| P2 field-shape diagnostics read-only | completed | `scripts/analyze_heat3d_v2_field_shape_diagnostics.py`, `scripts/check_heat3d_v2_field_shape_diagnostics_smoke.py`, `docs/v2_field_shape_diagnostics.md`, commit `578d4ea` | Read-only final/best field-shape diagnostics; generated for baseline and lower-lr predictions. | Diagnostic, not a training loss. |
| P3 Optax AdamW minimal integration | completed | `scripts/run_heat3d_v1_medium_controlled_training_export.py`, `configs/heat3d_v2/frozen_v1_e050_adamw_lr1e3_wd1e4_seed0.yaml`, `scripts/check_heat3d_v2_optimizer_config_smoke.py`, `docs/v2_optimizer_ablation_summary.md`, commits `2e9f9ca`, later runs | Adam/AdamW, weight decay, grad clipping configs and medium1024 seed0 runs A1/A2/A3. | Optimizer alone did not solve field-shape collapse. |
| P4 model capacity ablation | partial | `docs/v2_model_capacity_ablation_plan.md`, `configs/...m1_latent64_steps4_mlp2...`, mini-batch M1 configs, `scripts/check_heat3d_v2_batch_config_smoke.py`, commits `f242def`, `eee35e7`, `4d12a3d` | M1-lite e1/e3, M1 e1, M1 e50 mini-batch, lower-lr M1 e50 runs completed on SSH. | Full capacity ablation is not complete: no M2, no systematic small/medium/large grid, full-batch M1 output dir has no summary. |
| P5 hotspot / peak loss | partial | runner `hotspot_retention_loss`, `hotspot_weight`, pseudo-negative background loss; configs include `hotspot_quantile`, `hotspot_weight`; field-shape diagnostics include peak metrics | Hotspot retention is implemented and used in `background_pseudo_negative`. | Peak-specific training loss is not implemented; `draft_peak_loss_weight` remains null. |
| P6 staged / curriculum loss | partial | runner supports `loss_weight_schedule` choices `constant`, `two_phase`, `linear_anneal`; `transition_epoch` and start/end loss weights are wired | Schedule plumbing exists and is logged. | Current controlled v2 runs use `constant`; no validated staged/curriculum controlled run yet. |
| P7 SSH medium1024 controlled runs + seed sensitivity | partial | SSH output dirs: A1/A2/A3, M1-lite, M1 e50, lower-lr M1 e50; configs have `seed: 0`, `multi_seed: []` | medium1024 controlled seed0 runs are complete for several configs. | Seed sensitivity is not complete; no multi-seed M1/lower-lr controlled runs. |

## Next Steps

1. Run one M1 schedule experiment, preferably lr=3e-4 with warmup + cosine or second-stage decay, not a lower constant lr.
2. Define a selection report that always includes valid_loss, valid_raw_deltaT_mse, and best field-shape diagnostics.
3. Add seed sensitivity for the leading candidate before claiming improvement.
