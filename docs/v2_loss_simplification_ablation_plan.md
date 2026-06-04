# Heat3D v2 Loss Simplification Ablation Plan

本轮只准备 loss simplification configs 和 dry-run smoke，不运行 e50。

## Motivation

当前 `background_pseudo_negative` composite loss 同时优化全场 MSE、低温背景、pseudo-negative overprediction 和 hotspot retention。已有 M1 e50 结果显示：

- constant `3e-4` best-valid 综合最优，但 best_epoch 仍很早；
- lower-LR / second-stage 改善部分 field-shape、peak 和 low-temperature metrics，但 valid_loss/hotspot 变差；
- valid_loss 与 field-shape metrics 不完全一致；
- final degradation 仍明显。

这说明当前 full composite loss 可能过于复杂。简化 loss 可以判断 M1 容量本身是否能先稳定学习全场 DeltaT，再决定是否逐步加回 background/hotspot 约束。

## Existing Runner Support

runner 已支持以下 loss modes：

- `mse`
- `background_hotspot`
- `background_l1_bias`
- `background_l1_relative`
- `background_pseudo_negative`

因此本轮不需要新增训练语义：

- base MSE only 可用 `loss.mode=mse` 表达；
- base MSE + hotspot 可用 `loss.mode=background_hotspot` 且 `background_weight=0`、`hotspot_weight=0.02` 表达；
- background/pseudo-negative 复杂项均可通过权重置零关闭；
- 原 `background_pseudo_negative` 行为不改变。

## Prepared Configs

两个 config 均基于 `configs/heat3d_v2/frozen_v1_e050_adamw_m1_batch_lr3e4_seed0.yaml`，保持 M1 model、AdamW、`lr=3e-4`、`batch_size=4`、epochs=50、seed0、final/best predictions 和 `selection_metric=valid_loss`。

| config | loss mode | enabled training objective | output_dir |
|---|---|---|---|
| `configs/heat3d_v2/frozen_v1_e050_adamw_m1_batch_lr3e4_base_mse_seed0.yaml` | `mse` | normalized DeltaT base MSE only | `output/heat3d_v2_runs/m1_batch_e50_lr3e4_base_mse_seed0` |
| `configs/heat3d_v2/frozen_v1_e050_adamw_m1_batch_lr3e4_base_mse_hotspot_seed0.yaml` | `background_hotspot` | base MSE + `0.02 * hotspot_retention_loss` | `output/heat3d_v2_runs/m1_batch_e50_lr3e4_base_mse_hotspot_seed0` |

Disabled or zeroed in both simplification configs:

- `background_l1_weight`
- `background_bias_weight`
- `background_over_weight`
- `background_relative_weight`
- `pseudo_negative_weight`

For `base_mse_hotspot`, `background_weight=0.0` and `hotspot_weight=0.02`.

## What To Learn

The ablations answer different questions:

- `base_mse_only`: can M1 reduce raw_deltaT_mse / field-shape errors without hand-shaped background penalties?
- `base_mse_hotspot`: can a small hotspot term preserve high-temperature behavior without the low-temperature pseudo-negative machinery?
- comparison to full composite: are background/pseudo-negative terms helping selection metrics or making optimization noisy?

## Risks

- Simplified loss may worsen low-temperature overprediction because it removes direct background/pseudo-negative penalties.
- `base_mse_only` may under-emphasize hotspots because high-temperature regions are a small fraction of nodes.
- `base_mse_hotspot` may still not fix final degradation if optimizer/update-count is the dominant issue.
- Selection by `valid_loss` should be interpreted carefully because each loss mode changes what valid_loss means.

## Recommended Order

1. Run `base_mse_only` first as a clean baseline for M1 capacity and optimizer behavior.
2. Run `base_mse_hotspot` next if base-only improves raw_deltaT/field-shape but sacrifices hotspot metrics.
3. Compare all results by:
   - best/final valid_loss;
   - valid_raw_deltaT_mse;
   - hotspot_mae;
   - field_variance_ratio;
   - centered_spatial_correlation;
   - amplitude_ratio;
   - peak_abs_error;
   - top_k_overlap;
   - bg_bias and pn_over_ratio.
4. Only after this decide whether to reintroduce background/pseudo-negative terms with lower weights or a schedule.
