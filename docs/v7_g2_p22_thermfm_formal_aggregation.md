# P22 Therm-FM formal valid-only evaluation

devbox 上的 Therm-FM/Poseidon-T 三个 seed 已按同一冻结合同完成 200 epoch。训练只使用
P1i train 768 / valid 128；没有读取 `test_iid`、sealed 或 DeepOHeat official100。

## 结果

| seed | best epoch | best valid normalized loss | best sample-first rel. RMSE | best point-global rel. RMSE | best RMSE [K] | final sample-first rel. RMSE | walltime |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 184 | 0.007783 | 2.5107% | 3.1424% | 2.4955 | 2.5098% | 4812.7 s |
| 1 | 136 | 0.005909 | 2.4281% | 2.6919% | 2.1378 | 2.3362% | 4920.3 s |
| 2 | 181 | 0.010529 | 2.8812% | 3.6125% | 2.8688 | 2.8823% | 4953.9 s |

JSON 中历史字段名 `best_metrics` 仅表示“在 minimum valid normalized loss checkpoint
处记录的物理指标”；P23 的正式显示名称固定为
`metrics_at_loss_selected_checkpoint`，不得解释为每个物理指标各自重选的最优值。

按 **SD across training seeds** 汇总（checkpoint 仍按 normalized loss 选，不按物理指标重选）：

- best sample-first relative RMSE：**2.6066 ± 0.2413%**；point-global：**3.1489 ± 0.4603%**。
- best RMSE：**2.5007 ± 0.3656 K**；MAE：**1.6364 ± 0.1963 K**。
- final sample-first relative RMSE：**2.5761 ± 0.2790%**；final point-global：**3.1600 ± 0.4491%**。
- 平均训练耗时 **4895.7 ± 73.8 s**（约 81.6 ± 1.2 min）；峰值 allocated **5.095 GB**、reserved
  **6.451 GB**，三 seed 相同；参数量 **21,435,546**。

best→final 的物理指标变化很小且方向不一致（例如 sample-first 三 seed 分别为
`−0.0009、−0.0918、+0.0011` 个百分点），不能解释为统一退化或改动依据。三组 history finite，
checkpoint SHA 与 receipt 一致，reload state equal，未发现异常 seed。

完整 per-seed SHA、合同与隔离字段见
`docs/v7_g2_p22_thermfm_formal_aggregation.json`；大型 checkpoint 仍只保留在 devbox，未提交 Git。
该汇总是 valid-only transfer-track evidence，不是 test/generalization 结论。
