# Heat3D v2 B192 Loss Simplification Ablation Plan

本轮只准备 B192 + simplified loss configs 和 dry-run smoke，不训练、不跑 e50、不跑 diagnostics。本文是 research-stage controlled-training plan，不是 benchmark 结论。

## B192 Full-Composite Baseline

已完成的 B192 full-composite run：

- run: `output/heat3d_v2_runs/m1_batch_e50_lr3e4_B192_seed0`
- config: `configs/heat3d_v2/frozen_v1_e050_adamw_m1_batch_lr3e4_B192_seed0.yaml`
- loss: `background_pseudo_negative`
- batch_size / validation_batch_size / prediction_batch_size: `192 / 192 / 192`
- lr: `3e-4`
- epochs: `50`

关键结果：

| metric | value |
|---|---:|
| wall-clock | about `565s` |
| best_epoch | `1` |
| best_valid_loss | `1.062913` |
| best_valid_raw_deltaT_mse | `0.00124697` |
| final_valid_loss | `2.913905` |
| final_valid_raw_deltaT_mse | `0.00473993` |
| final_pn_over_ratio | `1.0` |

B192 full composite 很快，但没有超过 B4 `lr=3e-4`。它也没有自动解决 early-best：best_epoch 仍然是 `1`。final_valid_loss 比 B4 `lr=3e-4` 稍低，但 final degradation 仍明显，且 final low-temperature overprediction 仍未解决。

## Why B192 Is Still Useful

B192 每 epoch 只有约 4 updates，和 B4 的约 192 updates/epoch 不等价。它不是直接公平替代 B4，而是一个快速 ablation 平台：

- e50 wall-clock 很短，适合快速筛选 loss 方向；
- full-composite 的 early-best 失败说明 larger batch alone 不够；
- 如果 simplified loss 在 B192 下更稳定，说明当前 composite loss 可能是训练低效的重要原因；
- 如果 simplified loss 也 early-best/final degrade，则问题更可能来自 update count、optimizer schedule、model/dataset 或 selection metric。

## Prepared Experiments

### B192 Base MSE Only

Config:

`configs/heat3d_v2/frozen_v1_e050_adamw_m1_batch_lr3e4_B192_base_mse_seed0.yaml`

目标：

- 只保留 normalized DeltaT base MSE；
- 关闭 background、pseudo-negative 和 hotspot terms；
- 判断 M1 + B192 是否能先稳定学习全场 DeltaT。

该实验回答：

- full composite 是否使优化过早偏向 low-temperature / hotspot 诊断目标；
- base MSE 是否比 full composite 更稳定地降低 raw_deltaT_mse；
- final degradation 是否主要来自 composite loss 的多目标牵制。

### B192 Base MSE Plus Hotspot

Config:

`configs/heat3d_v2/frozen_v1_e050_adamw_m1_batch_lr3e4_B192_base_mse_hotspot_seed0.yaml`

目标：

- 保留 base MSE；
- 加回小权重 hotspot retention：`hotspot_weight=0.02`；
- background / pseudo-negative terms 保持关闭。

该实验回答：

- base-only 是否牺牲 hotspot；
- 一个轻量 hotspot term 是否能保留高温区域性能，而不引入 full composite 的低温/relative/pseudo-negative复杂性；
- hotspot loss 是否是可保留组件，还是当前 full composite 的瓶颈主要来自 background/pseudo-negative 项。

## Config Invariants

两个新 configs 均保持：

- dataset: `medium1024_gapA_full1024_v2`
- model: M1 (`latent64 / edge64 / processor_steps4 / mlp2`)
- optimizer: AdamW
- lr: `3e-4`
- lr_schedule: `constant`
- weight_decay: `1e-4`
- gradient_clip_norm: `1.0`
- seed: `0`
- epochs: `50`
- batch_size / validation_batch_size / prediction_batch_size: `192 / 192 / 192`
- save final predictions: true
- save best predictions: true
- selection_metric: `valid_loss`
- diagnostics section and baseline reference unchanged

Only loss mode/weights, description, output_dir, and run_name change.

## Required Post-Run Comparison

训练完成后必须和 B192 full composite、B4 `lr=3e-4` 一起比较：

- best_valid_loss
- best_valid_raw_deltaT_mse
- best_valid_hotspot_mae
- field_variance_ratio
- centered_spatial_correlation
- amplitude_ratio
- peak_abs_error
- top_k_overlap
- bg_bias
- pn_over_ratio
- final_valid_loss
- final_valid_raw_deltaT_mse
- final/best degradation
- wall-clock
- update count

尤其要分清：

- B192 是否只是因为 update count 少而欠拟合；
- simplified loss 是否改善 early-best；
- simplified loss 是否改善 final degradation；
- base-only 是否牺牲 hotspot / peak；
- base+hotspot 是否能在不使用 background/pseudo-negative terms 的情况下保留高温区域。

## Recommended Manual Order

1. 先运行 B192 base MSE only。
2. 如果 base-only raw_deltaT_mse 或 field-shape 更稳定，再运行 B192 base MSE + hotspot。
3. 不要同时启动多个长任务。
4. 训练结束后再补齐 diagnostics，并让 Codex 只读审查 output。

## Manual SSH Commands

Dry-run command generation:

```bash
ssh WSL
cd ~/myCodeGitOnly/heat3d-ic
git pull --ff-only
conda activate rigno
python scripts/check_heat3d_v2_B192_loss_simplification_configs.py
```

Training should be started manually only when requested. Do not submit `output/`.
