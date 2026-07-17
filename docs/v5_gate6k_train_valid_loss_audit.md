# Gate 6K train/valid_iid loss audit

- 范围：仅 `train + valid_iid`；未访问 test/hard/sealed。
- checkpoint：V13 legacy base-MSE best e318；V32 point-global best e474。
- 四项 loss 均为逐样本、乘配置权重前的原始分量。

## Core frozen V5 metrics

| split | model | point-global % | sample-first % | raw CV RMSE K | shape CV-RMSE | scale log-RMSE |
|---|---|---:|---:|---:|---:|---:|
| train | V13 | 6.332228 | 6.112260 | 0.057987 | 0.055467 | 0.025979 |
| train | V32 | 3.791688 | 3.939269 | 0.033975 | 0.036873 | 0.013717 |
| valid_iid | V13 | 23.701205 | 20.316565 | 0.167985 | 0.150282 | 0.165392 |
| valid_iid | V32 | 22.408788 | 21.034694 | 0.160069 | 0.145697 | 0.197480 |

## Valid 重点分层

| subset | model | n | point-global % | sample-first % | raw CV RMSE K | shape CV-RMSE | scale log-RMSE |
|---|---|---:|---:|---:|---:|---:|---:|
| ΔT Q2 | V13 | 32 | 24.131730 | 22.145185 | 0.060895 | 0.161668 | 0.174057 |
| ΔT Q2 | V32 | 32 | 32.474572 | 25.819719 | 0.082940 | 0.156498 | 0.241869 |
| Q2 ∩ nominal_to_hard | V13 | 21 | 27.137565 | 25.371605 | 0.068322 | 0.175052 | 0.208113 |
| Q2 ∩ nominal_to_hard | V32 | 21 | 38.351609 | 31.194544 | 0.097638 | 0.172291 | 0.292322 |

完整 JSON 包含每个模型/划分的 mean、median、P90、P95、P99、最差样本贡献、
top-5 累计贡献，以及 signed scale error 的均值、偏置方向和 RMSE；同时分别保存
Q2、`nominal_to_hard` 和两者交集。逐样本值保存在同目录 CSV。

## Signed scale error

- train：V13/V32 的均值分别为 `+0.00928/+0.00652`，V32 的 scale log-RMSE
  从 `0.02598` 降至 `0.01372`。
- valid：V13/V32 的均值分别为 `-0.01748/-0.01730`，平均 signed bias 基本不变，
  但 V32 scale log-RMSE 从 `0.16539` 升至 `0.19748`，说明退化主要来自误差离散度
  和尾部，而非整体正负偏移。

## 两组 e600 单变量计划

| candidate | config | 相对 V32 唯一科学差异 | 状态 |
|---|---|---|---|
| O075 | `V4P5_33_gate6k_o075_log_scale` | log-scale loss `0.5 → 0.75` | prepared, not started |
| Dual | `V4P5_34_gate6k_dual_physics_attention` | `shape_attention_mode: physics_gate` | prepared, not started |

两者都继承 V32 的 random-init、seed0、e600、train B28、valid/predict B32、
数据/split/模型其余部分/optimizer/LR，以及 point-global、sample-first、
base-MSE best 与 final checkpoint 保存合同。O075/Dual 尚无结果，因此对比表不得
填入或推断性能，也不自动晋级或继续 seed。

## 结论

V32 相对 V13 在 train 四项分量和全部核心指标上均改善；valid 则是 shape、
point-global 和 raw CV RMSE 改善，但 scale 与 sample-first 退化。退化在 ΔT Q2
及 Q2 ∩ `nominal_to_hard` 更明显，支持把 O075 作为严格单变量 objective-alignment
检验；Dual 则隔离 shape attention 的增量作用。Gate 6K 不据此自动晋级，也不触发
后续 seed。
