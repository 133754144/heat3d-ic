# G2 DeepOHeat-v1 pre-closure

状态：`VALID_ONLY_PRE_CLOSURE_COMPLETE`。本报告只使用冻结的 768 train / 128 valid physical cases；P1i `test_iid`、sealed、DeepOHeat official100 均未访问，G1 未修改。所有 GPU 训练按 seed0 → seed1 → seed2 串行完成。

## 冻结合同与证据

- DeepOHeat-v1 上游：`xlyu0127/DeepOHeat-v1@3ef3d9c41666a56b5940b39a61166ccaa5aaedb2`。
- matched runner commit：`fb2b06cc8c727a1374b4b62b804919ecfac73add`。
- official input `fs_train_volume.npy` SHA：`a39a4f51e853f9114d86feb88f74553914b2bfc68ab1c553a3a31df25893fff7`。
- subset manifest SHA：`e719665176a22213487ee92c1aac993dd01b02a51555c7cd68bf81a13b861558`；train/valid 为 768/128。
- reference-label receipt SHA：`a4bb99638a977b2004a93a88b469166ff7da697e89181e64e04152c7f96fe4fd`。
- full-field domain：101×101×56（571256 点），`deltaT_K = 25*(u-0.2)`。
- DeepOHeat matched 保持 3,303,680 参数、完整 PDE/Robin/adiabatic BC、50 functions/iteration、100000 iterations、5,000,000 function/collocation evaluations、Adam 1e-3 与 0.9/1000 指数衰减。

## A. method-native reference

官方 seed42 训练 receipt SHA 为 `51145d4785657567a1019dafc8102f248d7e4cce343bd9ea44cba68a85e63b56`，final checkpoint SHA 为 `e86ee37b5855b856d68753ae84f484b4d1adac322361df52d576c56d8c050173`。该 run 在完整 100000-row `fs_train_volume.npy` 池上训练，官方 release 没有 valid split。

Heat3D 冻结的 128 valid source IDs 全部位于该 100000-row native training pool，且 source hash 与 label mapping 逐行一致。因此在这些行上评价 native checkpoint 会产生 train/eval overlap。共同 temperature evaluator 虽然存在，但不能作为 held-out evidence。结论固定为：

`NOT_DIRECTLY_COMPARABLE`

native seed42 的 physics loss 不与 Heat3D temperature-space relative RMSE 直接比较；official100 仍 sealed。

## B. matched physical-case budget

三 seed 均使用相同的 768 train / 128 valid IDs、相同上游训练语义和 valid-only temperature-space selection。该层级是 `SAME_PHYSICAL_CASE_BUDGET`，明确不是 `SAME_INFORMATION_BUDGET`：DeepOHeat 使用完整 mesh 的 physics objective，Heat3D 使用 1024 sparse physical conditioning。

### DeepOHeat-v1 matched 三 seed

| seed | best iter | best valid sample-first rel-RMSE (%) | final (%) | best→final (pp) | wall (s) | peak RSS (GiB) | reload |
|---:|---:|---:|---:|---:|---:|---:|:---:|
| 0 | 70000 | 1.09569 | 1.13793 | 0.04224 | 2356.69 | 2.37 | PASS |
| 1 | 50000 | 1.60761 | 1.75589 | 0.14829 | 2375.70 | 2.34 | PASS |
| 2 | 60000 | 1.21220 | 1.41659 | 0.20439 | 2394.98 | 2.32 | PASS |
| **mean ± sample SD** | — | **1.30517 ± 0.26832** | **1.43681 ± 0.30948** | **0.13164 ± 0.08235** | **2375.79 ± 19.14** | **max 2.37** | all PASS |

完整机器可读汇总见 [`v7_g2_p14_deepoheat_v1_matched_aggregation.json`](v7_g2_p14_deepoheat_v1_matched_aggregation.json)。三 seed 均 finite、均有 best/final checkpoint；seed0 的早期 engineering-stopped attempt 保留为非正式证据，未与 retry1 混用。

## Heat3D 同域 valid-only evaluator

Heat3D 200e 三 seed 的 checkpoint 仍按预注册 native-1024 valid rule 固定；随后用同一 571256-point temperature evaluator 做 inference-only 评估，未用 full-field 结果重新选模。结果为：

| seed | sample-first rel-RMSE (%) | point-global rel-RMSE (%) | CV-weighted RMSE (K) | peak RMSE (K) | inference (s) |
|---:|---:|---:|---:|---:|---:|
| 0 | 1.44176 | 1.44790 | 0.37677 | 0.42392 | 24.01 |
| 1 | 1.41237 | 1.42077 | 0.36949 | 0.44821 | 23.85 |
| 2 | 1.44029 | 1.44717 | 0.37653 | 0.37849 | 25.16 |
| **mean ± sample SD** | **1.43147 ± 0.01656** | **1.43861 ± 0.01546** | **0.37426 ± 0.00414** | **0.41687 ± 0.03539** | **24.34 ± 0.71** |

机器可读 receipt/metric hashes 见 [`v7_g2_p14_heat3d_v1_fullfield_valid_aggregation.json`](v7_g2_p14_heat3d_v1_fullfield_valid_aggregation.json)。此处只报告 valid-only 同域推理，不解锁任何 test。

## 问题回答

**Q1（200e small-data Heat3D 相对 frozen method-native DeepOHeat 是否有可验证竞争力？）** 不能验证。native seed42 的 100000-row training pool 包含全部候选 valid rows；若评分会泄漏，故不作 native 优势/劣势结论。

**Q2（同 768/128 physical-case budget 谁更准、代价差多少？）** 在预先冻结的 matched cohort 上，Heat3D full-field valid sample-first 为 `1.43147 ± 0.01656%`，DeepOHeat-v1 为 `1.30517 ± 0.26832%`。这只是同 physical-case budget 的 valid-only 描述性比较，不是 same-information-budget 或 same-training-budget 比较：DeepOHeat 每次迭代执行完整 mesh PDE/BC，Heat3D 是 supervised sparse-conditioning；参数量、solver/function evaluations、训练 walltime 必须分列报告。Heat3D full-field 结果的 checkpoint selection 仍是 native-1024 预注册规则，未做 post-hoc full-field 选模。

计算代价的可比性也应谨慎解读：Heat3D 200e cohort 的训练参数量为 892,776、训练 walltime `1248.54 ± 5.99 s/seed`、峰值设备内存约 4.06 GB；matched DeepOHeat-v1 为 3,303,680 参数、`2375.79 ± 19.14 s/seed`、5,000,000 次 PDE/collocation function evaluations、峰值 RSS 约 2.55 GB（GPU peak 约 0.75 GB）。两者 epoch/iteration 定义不同，故这些数值用于报告训练负担而非宣称等预算。

## 分类与后续边界

- `DeepOHeat-v1 method-native`：semiconductor-native，`NOT_DIRECTLY_COMPARABLE`（无公开 held-out native split）。
- `DeepOHeat-v1 matched 768/128`：`SAME_PHYSICAL_CASE_BUDGET`，三 seed valid-only cohort 完成，可作为独立 cross-benchmark comparison 表。
- `Heat3D-on-DeepOHeat-v1`：同域 full-field valid inference 完成；不把 native-1024 与 full-field 指标混列。
- test/sealed/official100：仍锁定；本轮不做 Heat3D 600e、Therm-FM、multi-HTC 或任何正式 test evaluation。
