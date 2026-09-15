# GINO seed0/seed1 valid-only 诊断

观测时间：2026-09-15 14:58（devbox WSL2）。本诊断只读取 train / `valid_iid` 历史与 checkpoint 元数据；`test_iid`、sealed 和 DeepOHeat official100 均未访问。

| seed | 状态 | best epoch | best selection (`sample_first_relative_rmse_pct`) | best primary (`point_global_relative_rmse_pct`) | final selection | final primary | 选择指标 best→final | finite / reload |
|---:|---|---:|---:|---:|---:|---:|---:|---|
| 0 | `COMPLETE_FORMAL_TRAIN_REPAIRED` | 124 | 15.4951 | 18.1752 | 15.9491 | 18.6070 | +0.4540 pp | PASS / PASS |
| 1 | `COMPLETE_FORMAL_TRAIN` | 51 | 15.5856 | 19.1196 | 16.6574 | 19.5348 | +1.0717 pp | PASS / PASS |

seed0 原始进程在训练完成后因 `OrderedDict` 顺序与 `_metadata` 误纳入比较而退出；canonical comparator 已只读确认 model/optimizer/scheduler 数值状态与三个 checkpoint 完整。该修复不重训、不改变科学配置。seed1 正常写出 `run_receipt.json`，301 epochs / 231168 updates，peak allocated 2,246,255,104 bytes，launch-to-final mtime 约 9,635 s。

两条历史的 train objective 均从约 0.89 降至约 0.016–0.021，全部指标 finite；best epoch 和 best/final 差异保留为 seed variability，不用于调参。seed2 已在相同冻结合同下串行启动并观察到 epoch 1（768 updates），尚不能据此作最终结论。

详细机器可读收据见 [`v7_g2_p13_gino_seed01_valid_only_diagnostic.json`](v7_g2_p13_gino_seed01_valid_only_diagnostic.json)、seed1 完成收据和 seed2 启动收据。
