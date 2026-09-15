# GINO P1i 三 seed valid-only 汇总

观测时间：2026-09-15 17:57（devbox WSL2）。三组训练均为 301 epochs、batch 1、Open3D FixedRadiusSearch + torch-scatter；只使用 train / `valid_iid`，没有访问 `test_iid`、sealed 或 DeepOHeat official100。

| seed | 状态 | best epoch | best selection | best primary | final selection | final primary | best→final selection |
|---:|---|---:|---:|---:|---:|---:|---:|
| 0 | `COMPLETE_FORMAL_TRAIN_REPAIRED` | 124 | 15.4951 | 18.1752 | 15.9491 | 18.6070 | +0.4540 pp |
| 1 | `COMPLETE_FORMAL_TRAIN` | 51 | 15.5856 | 19.1196 | 16.6574 | 19.5348 | +1.0717 pp |
| 2 | `COMPLETE_FORMAL_TRAIN` | 163 | 13.9609 | 15.8997 | 14.2239 | 15.9130 | +0.2629 pp |

三 seed best selection 均值 ± sample SD 为 **15.0139 ± 0.9130%**；best primary 为 **17.7315 ± 1.6551%**。final selection 为 **15.6101 ± 1.2517%**，final primary 为 **18.0183 ± 1.8813%**。所有历史 finite，checkpoint state reload 均通过。

seed0 的原始训练已完成但旧 runner 因 `OrderedDict`/`_metadata` 比较误报；canonical comparator 修复仅做状态完整性核验，没有重训或改科学配置。seed1、seed2 正常写出正式 receipt。best epoch 差异与 best→final 变化保留为 seed variability，不用于事后修改配置。

三 seed 均约 2.67 小时（seed0/1/2 launch-to-final mtime 下界约 9600/9635/9640 秒）；seed0 旧运行缺少资源字段，seed1/2 峰值 allocated 约 2.25 GB。

完整 SHA、运行合同和 test 隔离字段见 [`v7_g2_p13_gino_3seed_valid_only_aggregation.json`](v7_g2_p13_gino_3seed_valid_only_aggregation.json)。官方测试仍保持 sealed，后续需单独的 evaluation-only unlock。
