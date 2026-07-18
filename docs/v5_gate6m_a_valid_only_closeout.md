# Gate 6M A valid-only 结果与诊断

`V4P5_35_gate6m_v32_scale_head_only_e100` 已在 WSL2 完成 e100。本文档只使用训练期保存的 `valid_iid` 预测与 train-only 统计，不读取 test、hard 或 sealed IID。

## 运行与完整性

- 实际主机：`wsl2`；训练 commit：`414ab43`；冻结 evaluator：`9c124709b2c2a90cb5698831336716a5a7a21357`。
- split：train=672，valid_iid=128；每样本 1024 nodes；valid IDs SHA256：`cf682f92aba37a8f708ba66f12351c866d54131d2c33a1fa0ec421103fe02d63`。
- normalization/context：均由 train-only 重算并一致；target/label-derived feature 数为 0。
- e100 history 连续、grad_finite=true；四类 checkpoint 的参数树 reload 精确相等，预测 reload 误差均低于 0.02 K。
- registry 声明的训练日志路径当前不存在，因此日志完整性无法核验；完成判断仅依据 `loss_summary.json`、连续 history、checkpoint 与 prediction archives。
- 访问边界：`roles_accessed=[train, valid_iid]`；`test_accessed=false`、`hard_accessed=false`、`sealed_iid_accessed=false`。

## 四 checkpoint 冻结 V5 指标

| checkpoint (epoch) | point-global % | sample-first % | raw CV K | amp | corr | hotspot K | top-5 K | strong-q K | low-ΔT bias K | low-ΔT RMSE K | over-ratio | shape | scale log | legacy MSE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| point_global_best (e18) | 22.390066 | 20.936297 | 0.159923 | 1.004633 | 0.981542 | 0.301933 | 0.466998 | 0.367884 | 0.004672 | 0.018134 | 0.414697 | 0.145697 | 0.192071 | 0.032628 |
| sample_first_best (e25) | 22.641949 | 20.840604 | 0.161713 | 0.988276 | 0.981543 | 0.303919 | 0.472847 | 0.369515 | 0.004353 | 0.017992 | 0.405193 | 0.145697 | 0.196255 | 0.033366 |
| legacy_best (e18) | 22.390484 | 20.936610 | 0.159926 | 1.004635 | 0.981542 | 0.301936 | 0.467009 | 0.367888 | 0.004672 | 0.018134 | 0.414662 | 0.145698 | 0.192076 | 0.032629 |
| final (e100) | 22.596596 | 20.939670 | 0.161343 | 0.993067 | 0.981543 | 0.303914 | 0.472037 | 0.370321 | 0.004240 | 0.017926 | 0.405490 | 0.145695 | 0.198356 | 0.033233 |

冻结契约的 `<20%` 门槛在 A 的 point-global 与 sample-first 两个视角都未通过（最佳分别 22.3901% 与 20.9363%）。`params_best_valid_point_global` 与 `params_best_valid_base_mse` 都为 e18；`params_best_valid_sample_first` 为 e25；本次没有重新选择或覆盖 checkpoint。

## 相对 V32 e474 的诊断（valid-only）

| 指标 | V32 point-global e474 | A point-global e18 | A−V32 |
|---|---:|---:|---:|
| point-global % | 22.408387 | 22.390066 | -0.018321 |
| sample-first % | 21.034804 | 20.936297 | -0.098507 |
| raw CV RMSE K | 0.160067 | 0.159923 | -0.000144 |
| amplitude ratio | 1.001418 | 1.004633 | +0.003215 |
| spatial correlation | 0.981542 | 0.981542 | +0.000000 |
| hotspot K | 0.302944 | 0.301933 | -0.001011 |
| top-5 K | 0.467499 | 0.466998 | -0.000501 |
| strong-q K | 0.366795 | 0.367884 | +0.001089 |
| low-ΔT bias K | 0.004501 | 0.004672 | +0.000170 |
| low-ΔT RMSE K | 0.018058 | 0.018134 | +0.000076 |
| low-ΔT over-ratio | 0.412114 | 0.414697 | +0.002583 |
| shape CV-RMSE | 0.145697 | 0.145697 | +0.000000 |
| scale log-RMSE | 0.197482 | 0.192071 | -0.005412 |
| legacy valid_base_mse | 0.032681 | 0.032628 | -0.000053 |

- A 的 point-global、sample-first、raw CV、scale log-RMSE、hotspot、top-5 均有很小改善；shape 基本不变，strong-q 和 low-ΔT 三项略退化。
- 由于 A 只训练 scale head，训练 epoch 1→100 的 backbone 与 shape-decoder gradient 始终为 0；scale-head gradient 从早期约 0.56 降至 e100 约 0.09。valid-base-MSE 在 e18 达到 0.0326278，e100 回升至 0.0332338；train error 继续下降而 valid 指标回退，表现为 scale-only 的轻微过拟合/尾部失配。
- e100 相对 e18：point-global +0.2065 percentage point、raw CV +0.001420 K、scale log-RMSE +0.006285；因此 final 不应替代 point-global/legacy best。
- native 分解（e18 → e100）：joint relative 20.9364% → 20.9399%；oracle-scale 14.5697% → 14.5696%；oracle-shape 13.1382% → 13.1628%；physics-scale proxy 约 51.324%。四项 valid loss 在 e100 为 shape=0.024678、log-scale=0.039346、relative=0.060633、raw-absolute=0.026033；scale-only 更新没有改变 shape 分支。

## Attention / native runtime

- 四 checkpoint 均为 `sparse_safe_v2`、256 regional nodes；normalized entropy mean 约 0.858，最大权重均值约 0.048，显著高于 uniform `1/256` 但未出现 collapse。
- attention 与 `log_inverse_kz_relative` 的 Pearson 约 −0.34 至 −0.35，与 `log1p_q_relative` 约 −0.10 至 −0.11，与 `source_present_fraction` 约 −0.10 至 −0.11；说明存在弱到中等的 conductivity-aware regional selection。
- runtime audit 通过：pooled latent width=96、scale head depth=1、`s_hat_positive=true`、`scale_attention_mode=physics_gate`、shape attention=none。

## B 状态

`V4P5_36_gate6m_v32_epoch_regroup_e600` 在当前 WSL2/Devbox 检查均没有 `run_config.json`、`loss_summary.json`、checkpoint 或 predictions；因此保留 `not_started`，不作任何性能结论。

## 结论

A 是“scale-head-only、e100”的小幅 valid-only 改善，未达到 `<20%` 可信门槛，也没有证据支持把它晋级为新的全模型候选。主要增益来自 scale 路径；shape/backbone 未更新。后续若继续，应先确认 B 是否实际启动并完成，再在同一 valid-only evaluator 下比较；本轮不启动任何训练。
