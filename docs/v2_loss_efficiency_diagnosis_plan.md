# Heat3D v2 Loss-Efficiency Diagnosis Plan

本文只做代码审计、配置准备和短 smoke 记录，不包含 e50 训练结果。当前阶段仍是 research-stage controlled training，不是 formal benchmark。

## Current Problem

M1 mini-batch e50 已可稳定完成，但训练效率仍不理想：

- constant `lr=3e-4` 是当前 best-valid 综合最优，但 best_epoch 仍在 epoch 2；
- `3e-4 -> 1e-4 @ epoch 5` 将 best_epoch 后移到 epoch 20，但没有超过 constant `3e-4`；
- final/best degradation 仍明显，schedule run final/best 仍约 `2.95x`；
- valid_loss、raw_deltaT_mse、hotspot、field-shape、低温 overprediction 指标不完全一致；
- 当前 `batch_size=4` 每 epoch 约 192 updates，梯度噪声可能较大；
- 当前 full composite loss 目标较多，可能让优化方向互相牵制。

## Current Loss Components

当前主力 M1 config 使用 `loss.mode=background_pseudo_negative`。runner 中该模式的 total loss 为：

```text
base_mse
+ background_l1_weight * background_l1
+ background_bias_weight * background_signed_bias_loss
+ background_over_weight * background_overprediction_loss
+ background_relative_weight * background_relative_abs
+ pseudo_negative_weight * pseudo_negative_unweighted_loss
+ hotspot_weight * hotspot_retention_loss
```

组件含义：

- `base_mse`: normalized DeltaT 空间的全场 MSE；
- `background_l1`: background quantile 区域对 `abs(pred_raw_deltaT)` 的惩罚；
- `background_signed_bias_loss`: background 区域 mean raw error 的绝对偏置惩罚；
- `background_overprediction_loss`: background 区域 `relu(pred_raw_deltaT - true_raw_deltaT)`；
- `background_relative_abs`: background 区域 safe relative absolute raw error；
- `pseudo_negative_unweighted_loss`: near-zero / low-DeltaT 区域 overprediction hinge，可用 relative_l1；
- `hotspot_retention_loss`: hotspot quantile 区域 normalized DeltaT MSE；
- diagnostics 还记录 `bg_signed_bias`、`pseudo_negative_over_ratio`、`hotspot_raw_mae` 等。

## Possible Objective Conflicts

这些组件可能同时拉动不同方向：

- background L1 / overprediction 项推动低温区域预测靠近 0 或不过预测；
- hotspot retention 推动高温区域保持幅值和形状；
- normalized base MSE 追求全场平均误差；
- relative background loss 会放大 near-zero DeltaT 区域的误差权重；
- pseudo-negative overprediction 只惩罚上偏，可能与全场 MSE 的对称误差目标不一致；
- valid_loss 由 composite loss 定义，未必与 field-shape 指标单调一致。

当前证据显示这种不一致已经发生：

- constant `3e-4` best_valid_loss `0.8200`、raw_deltaT_mse `0.001031`、hotspot_mae `0.04216`，是 best-valid 综合最优；
- constant `1e-4` best_valid_loss 较差 `1.0325`，但 best-prediction centered correlation 更高 `0.8569`，peak_abs_error 更低 `0.09209`；
- second-stage `3e-4 -> 1e-4 @ e5` best_epoch 后移到 20，low-temperature bg_bias/pn_over_ratio 改善，但 valid/hotspot 不如 constant `3e-4`；
- constant `3e-4` 的 best field_variance_ratio 接近 1，但 amplitude_ratio 低于 1；低 LR runs 的 correlation/peak 改善但 variance/amplitude 过冲。

## Why Batch Size Matters

当前 `batch_size=4` 时，medium1024 的 train split 每 epoch 约 192 mini-batch updates。优点是 update 数多，缺点是每次梯度来自很少 samples：

- composite loss 中 quantile/mask 项在小 batch 上更易抖动；
- pseudo-negative 和 hotspot mask 每 batch 的样本组合变化较大；
- background/pseudo-negative/热点目标可能造成 batch-to-batch 梯度方向不稳定；
- valid curves 已显示明显震荡，early-best 和 final degradation 尚未解决。

larger batch 或 larger effective batch 可降低梯度噪声，并让 quantile/mask 项更接近全 split 分布。但它也会显著减少每 epoch update 数：

| batch_size | updates/epoch on 768 train samples |
|---:|---:|
| 4 | 192 |
| 96 | 8 |
| 192 | 4 |

因此 e50 larger-batch 不是 update-count 等价比较。正式对比时需要同时报告 update count、wall-clock、best/final metrics，后续可能改为按 total updates 或 wall-clock 设计公平实验。

## Why B192 Needs Smoke First

`batch_size=192` 会一次构造约 73728 nodes 的 batch graph，并显著增加 graph/group build 和 JAX step 内存压力。不能直接长训，必须先确认：

- arrays + graph 能构造；
- M1 forward/loss 能运行；
- grad/update step 能运行；
- 没有 OOM 或明显超时；
- B96/B192 的 per-step 时间可以接受。

本轮 SSH WSL one-step smoke 结果显示 GPU backend 下 B8/B16/B32/B64/B96/B192 均能完成单步 loss+grad+update。

## Recommended Experiment Order

1. 不继续盲目扫更低 constant LR。`1e-4` 已显示 best_epoch 后移但 best-valid/hotspot 变差。
2. 先评估 larger batch full composite loss：
   - B96；
   - B192；
   - 明确记录 update-count 不等价。
3. 再评估 loss simplification：
   - base normalized DeltaT MSE only；
   - base MSE + small hotspot retention；
   - 与 full composite loss 同 batch_size=4 对照。
4. 若 B96/B192 长训不稳定或效果不清晰，再设计 gradient accumulation，以保持 update count 和 effective batch 分离。
5. 若 simplified loss 能更稳定下降，再考虑逐步加回 low-temperature / hotspot 项，而不是一次使用 full composite。

## Training-Progress Monitor

当前 runner 已记录 per-epoch `valid_loss`、`valid_raw_deltaT_mse`、hotspot/background/pseudo-negative components、LR、grad norm reporting summary，以及 final/best metrics。后续若要进一步诊断，可以低风险补充：

- per-epoch mean mini-batch loss；
- component means by train mini-batches；
- reported grad norm mean/max；
- cumulative update count；
- LR by epoch；
- valid_loss / valid_raw_deltaT_mse / hotspot_mae 同表输出。

本轮不实现新的 monitor 语义，避免扩大 runner 改动面。
