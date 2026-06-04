# Heat3D v2 Large-Batch Ablation Plan

本轮只准备 larger-batch configs 和 smoke，不运行 e50 长训。

## Motivation

当前 M1 主力配置使用 `batch_size=4`，每 epoch 约 192 updates。full composite loss 包含 background、pseudo-negative、hotspot 和 base MSE 多个目标，小 batch 可能让 mask/quantile 组件抖动更明显。larger batch 的目标是降低梯度噪声，观察 early-best 和 final degradation 是否缓解。

## SSH WSL Feasibility Smoke

脚本：

```bash
python scripts/check_heat3d_v2_batch_size_feasibility.py --batch-sizes 8,16,32,64,96,192
```

本轮通过 SSH WSL 对本地脚本内容做了一次短 one-step smoke。它不写 output，不训练 epoch，只对每个 batch size 构造一个 train mini-batch、forward/loss、grad/update。

结果：

| batch_size | status | elapsed_s | group_build_s | forward_s | grad_s | update_s | total_nodes | total_edges |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 8 | success | 31.26 | 4.81 | 1.23 | 13.72 | 3.98 | 3072 | 1641 |
| 16 | success | 30.76 | 5.49 | 7.26 | 14.34 | 3.66 | 6144 | 1641 |
| 32 | success | 33.52 | 7.09 | 6.99 | 15.75 | 3.69 | 12288 | 1641 |
| 64 | success | 36.70 | 9.89 | 7.24 | 15.91 | 3.66 | 24576 | 1641 |
| 96 | success | 40.42 | 12.72 | 7.43 | 16.56 | 3.70 | 36864 | 1641 |
| 192 | success | 49.40 | 21.49 | 7.51 | 16.69 | 3.70 | 73728 | 1641 |

Backend was `gpu` with `cuda:0`. B192 is feasible for a one-step smoke, but group build cost grows with batch size and e50 wall-clock is still unknown.

## Prepared Configs

Both configs are based on `configs/heat3d_v2/frozen_v1_e050_adamw_m1_batch_lr3e4_seed0.yaml` and keep model, optimizer, loss, epochs, seed, save final/best predictions, and selection metric unchanged.

| config | batch_size | validation_batch_size | prediction_batch_size | output_dir |
|---|---:|---:|---:|---|
| `configs/heat3d_v2/frozen_v1_e050_adamw_m1_batch_lr3e4_B96_seed0.yaml` | 96 | 96 | 96 | `output/heat3d_v2_runs/m1_batch_e50_lr3e4_B96_seed0` |
| `configs/heat3d_v2/frozen_v1_e050_adamw_m1_batch_lr3e4_B192_seed0.yaml` | 192 | 192 | 192 | `output/heat3d_v2_runs/m1_batch_e50_lr3e4_B192_seed0` |

## Update-Count Caveat

These are not update-count equivalent to B4:

| batch_size | updates/epoch | updates over e50 |
|---:|---:|---:|
| 4 | 192 | 9600 |
| 96 | 8 | 400 |
| 192 | 4 | 200 |

If B96/B192 appears worse, it may be because it has far fewer optimizer updates, not because larger effective batch is inherently worse. If it appears more stable, the next question is whether total update count or wall-clock budget should be controlled.

## Recommended Use

1. Start with B96 before B192 because it is less extreme and still reduces updates/epoch from 192 to 8.
2. Only run one e50 candidate at a time on SSH WSL.
3. Report best and final valid_loss, raw_deltaT_mse, hotspot_mae, field-shape metrics, final/best ratio, wall-clock, and update count.
4. If B96/B192 are feasible but underfit due low update count, consider gradient accumulation next: keep micro-batch small, accumulate an effective B32/B64/B96 update, and compare by update count.
