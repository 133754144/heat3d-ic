# Heat3D-on-DeepOHeat-v1 P13 valid-only cohort

状态：`TRAINING_COMPLETE_EVALUATION_STILL_SEALED`。

该 cohort 使用远端 repo `043b0ba54b80ff790632c80cda6c21de46bb5111`、
`legacy` execution path、`XLA_FLAGS=--xla_gpu_deterministic_ops=false`、B24/
valid32、200 epochs、fresh start 和 native-1024 valid sample-first 选模。
三个 seed 串行运行；没有读取 P1i test/sealed 或 DeepOHeat official100。

| seed | attempt | best epoch | valid selection (%) | train wall (s) | peak memory (GB) | reload |
|---:|---|---:|---:|---:|---:|---|
| 0 | initial | 200 | 0.774624 | 1242.18 | 3.96 | PASS |
| 1 | retry1（首次入口失败后同合同重试） | 200 | 0.795786 | 1249.36 | 4.06 | PASS |
| 2 | initial | 200 | 0.769495 | 1254.08 | 3.98 | PASS |

valid selection mean ± sample SD 为 `0.779968 ± 0.013937%`；训练时间为
`1248.54 ± 5.99 s`。三个 seed 的 best epoch 均为最后 epoch，因此 best→final
selection degradation 为 0。重复推理差异（max-abs 约 0.0059–0.0068、relative-L2
约 1.13e-5–1.28e-5）仅作 nondeterministic diagnostic，不是准入阈值。

seed1 的初次 `python3` 入口失败已由
`v7_g2_p13_heat3d_v1_seed1_initial_failure_receipt.json` 记录；retry 使用显式
rigno 环境 Python 且未覆盖任何输出目录。冻结的 native runner 未输出
`point_global_relative_rmse_pct`，因此本 receipt 不补算 primary metric，也不将
该 valid-only 结果写成 test/跨域 accuracy claim。
