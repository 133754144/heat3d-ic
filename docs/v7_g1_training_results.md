# V7 G1 正式训练结果与开发记录

## 文档定位

本文件记录 V7 G1 正式训练矩阵的工程结果、可复现 provenance 和有限范围内的结果解读。它不是新的实验协议，也不修改 V6 frozen evidence、模型、损失、数据划分或 checkpoint-selection 规则。

结果来自 devbox 上的机器可读 receipt：`/tmp/v7_g1_formal_runs/matrix_status.json` 以及每个 run 目录下的 `v7_g1_formal_receipt.json`。本文中的汇总指标均为 `valid_iid` 上按预注册的 `sample_first_relative_rmse_pct` 选出的最佳 checkpoint，均值和 sample SD 由三个 seed（0/1/2）计算。

## 执行状态与 provenance

| 项目 | 冻结值/观测值 |
| --- | --- |
| 矩阵 | `7 variants × 3 seeds = 21 runs` |
| 最终状态 | `COMPLETE`；21 个 receipt 均为 `COMPLETE`；失败 0 |
| 数据集 | `heat3d_v6_p1i_continuous_physics1024_v1` |
| split | train 768；`valid_iid` 128；未读取 `test_iid` / sealed |
| dataset manifest SHA256 | `f19987c659968c2ac14eade1f1ef7e206c8f7eeb94f58fde5897d6e765978514` |
| full-field archive SHA256 | `49023ac1205b8e7cf7c5bf782b89fcdb34997704b3f9aa2fb2d46cf1a59163cb` |
| V7 formal code SHA | `191a7a06a681556f575a1c04e2b61cb13363efe1` |
| recorded Git commit | `78e7651bab5ef41a8ca4e42c45f64b1b98f04ea7` |
| matrix finished at | `2026-08-31T17:34:34.478207+00:00` |
| evidence flags | `publication_evidence=true`、`scientific_evidence_eligible=true`（21/21） |
| safety flags | `test_iid=false`、`sealed=false`、`solver=false`、`new_data=false`（21/21） |

原始 checkpoint、prediction、log 和 receipt 未复制进 GitHub；它们仍以 devbox 路径为准。Git 中只保留控制面与本开发记录。

## 共同训练契约

- 优化器：AdamW，`base_lr=5e-4`、`min_lr=5e-5`、`warmup=10`、`cosine_horizon=200`、`weight_decay=1e-4`、gradient clipping `1.0`。
- 预算：完整新 schedule，`200 epochs`，随机初始化。
- batching：B24，768 个 train sample 对应 32 batches/epoch；validation B32，128 个 `valid_iid` sample 对应 4 batches。
- seed contract：`model = optimizer = batch_build = batch_order = graph = run_seed`，seed set `{0,1,2}`。
- checkpoint selection：`valid_iid` 上的 `sample_first_relative_rmse_pct`，并以 earliest epoch 处理并列。
- 所有结果均属于同一个冻结的 P1i Full parent 及其 registry-defined variant delta；未做 post-hoc seed 删除、budget 修改或 metric 切换。

具体 launch、模型、统计和 support 契约见：[formal launch manifest](../configs/heat3d_v7/v7_g1_formal_launch_manifest.json)、[Full P1i parent](../configs/heat3d_v7/v7_g1_full_p1i.json)、[statistical preregistration](../configs/heat3d_v7/v7_g1_statistical_preregistration.json)、[support provider contract](../configs/heat3d_v7/v7_g1_support_provider_contract.json) 和 [scientific protocol freeze](v7_g1_scientific_protocol_freeze.md)。

## 21-run 汇总

下表使用最佳 checkpoint。百分比指标越低越好；物理区域误差单位为 K。`best epoch` 同时给出三个 seed 的范围和均值。

| Variant | best epoch（范围；均值） | sample-first relative RMSE (%) | point-global relative RMSE (%) | raw CV RMSE (K) | source-region RMSE (K) | peak RMSE (K) | interface RMSE (K) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **Full** | 152–188; 165.333 | **1.692 ± 0.048** | **2.058 ± 0.081** | **1.329 ± 0.038** | **2.781 ± 0.087** | **3.714 ± 0.022** | **0.719 ± 0.034** |
| `no_film` | 124–168; 148.000 | 1.986 ± 0.025 | 2.265 ± 0.036 | 1.503 ± 0.039 | 2.978 ± 0.045 | 4.299 ± 0.206 | 0.872 ± 0.010 |
| `layout_agnostic_stratified_support` | 80–113; 97.000 | 3.765 ± 0.019 | 3.339 ± 0.033 | 2.843 ± 0.025 | 6.541 ± 0.316 | 8.185 ± 0.701 | 0.534 ± 0.086 |
| `cv_only_support` | 105–137; 119.333 | 4.746 ± 0.009 | 4.039 ± 0.006 | 3.594 ± 0.001 | 6.503 ± 0.140 | 7.652 ± 0.094 | 0.989 ± 0.069 |
| `vanilla_RIGNO` | 138–200; 175.333 | 22.389 ± 1.567 | 22.342 ± 3.235 | 16.813 ± 2.218 | 17.932 ± 2.434 | 19.538 ± 3.850 | 2.990 ± 0.180 |
| `vanilla_RIGNO_capacity_matched` | 159–195; 182.667 | 25.162 ± 11.003 | 24.021 ± 5.459 | 17.993 ± 4.478 | 20.527 ± 3.842 | 23.275 ± 4.329 | 3.373 ± 0.965 |
| `physics_scale_only` | 8–35; 22.667 | 235.066 ± 0.184 | 172.398 ± 1.144 | 172.770 ± 0.151 | 39.153 ± 2.634 | 217.466 ± 9.971 | 56.141 ± 1.946 |

对应 run ID 是每个 variant 的 `_seed0`、`_seed1`、`_seed2`；完整映射以 [frozen launch manifest](../configs/heat3d_v7/v7_g1_formal_launch_manifest.json) 和 devbox receipt 为准。

## 最佳 checkpoint 与最终 epoch

checkpoint selection 是预注册行为，因此不能用最终 epoch 替代最佳 checkpoint。各 variant 的 `sample_first_relative_rmse_pct` 均值如下：

| Variant | selected best checkpoint | final epoch 200 |
| --- | ---: | ---: |
| Full | 1.692% | 1.741% |
| `no_film` | 1.986% | 2.067% |
| `layout_agnostic_stratified_support` | 3.765% | 3.843% |
| `cv_only_support` | 4.746% | 4.820% |
| `vanilla_RIGNO` | 22.389% | 23.158% |
| `vanilla_RIGNO_capacity_matched` | 25.162% | 25.727% |
| `physics_scale_only` | 235.066% | 235.651% |

这表明本矩阵中最终 epoch 通常略差于预注册的 selected checkpoint；该现象不构成重新选择预算或 metric 的理由。

## 结果解读

1. **Full 是当前 P1i valid_iid 契约下的最强配置。** 相对于 Full 的 `1.692%` 主指标均值，`no_film` 增加 `0.294` 个百分点（约 `17.4%`），说明 24-D global-context FiLM 有稳定但不是唯一的贡献。
2. **support ablation 显示 source/layout-aware 支撑的重要性。** `layout_agnostic_stratified_support` 和 `cv_only_support` 的主指标分别增加约 `122.5%` 和 `180.5%`。其中 layout-agnostic variant 的 interface RMSE 均值（`0.534 K`）低于 Full，但其全局、source-region 和 peak 误差更高，说明不能用单一局部指标替代整体评价。
3. **Vanilla baseline 与 Full 存在大差距。** canonical Vanilla 的主指标为 `22.389%`，capacity-matched Vanilla 为 `25.162%`；参数量已从 Full 的 `892,776` 分别控制为 `826,277` 和 `895,905`，但容量匹配没有消除差距。capacity-matched 组的跨 seed SD 为 `11.003` 个百分点，明显高于 Full 的 `0.048`，因此其结论应同时报告高 seed variance。
4. **learned scale correction 是当前冻结语义下的关键组件。** `physics_scale_only` 的主指标均值为 `235.066%`，且三个 seed 的最佳 epoch 仅为 8–35；这是一个明确的负向消融结果，但只支持当前 P1i/模型契约下的组件归因，不外推为所有数据域的结论。
5. **收敛时点存在 variant 差异。** Full 的最佳 epoch 为 152–188；Vanilla 的一个 seed 在 epoch 200 才达到最佳，capacity-matched 的两个 seed 在 194–195 才达到最佳。这提示 Vanilla 可能对 e200 预算更敏感，但正式矩阵的 200 epoch 是冻结协议，不能在观察结果后改预算。

## 参数量与 variant 语义

| Variant 类别 | receipt 中参数量 |
| --- | ---: |
| Full、两种 support variant | 892,776 |
| `no_film` | 878,696 |
| `physics_scale_only` | 845,158 |
| canonical Vanilla RIGNO | 826,277 |
| capacity-matched Vanilla RIGNO | 895,905 |

本矩阵中的 variant 不是复制 runner：

- `layout_agnostic_stratified_support` 对应 `generic_stratified_v2`，保留 boundary/interface/surface/control-volume coverage，只移除 q/k block-layout-aware quota。
- `cv_only_support` 对应 `cv_only_v1`，仅保留 1024 个 control-volume-weighted interior support。
- `no_film` 只有一个 delta：关闭 global-context FiLM；冻结的 24-D context tensor、scale semantics 和其它 Full 路径保留。
- `physics_scale_only` 保留 physics scale，仅关闭 learned residual scale correction，不是 direct-output architecture。

这些语义由 [support contract](../configs/heat3d_v7/v7_g1_support_provider_contract.json)、[variant qualification receipt](v7_g1_variant_qualification_receipt.json) 和 [scientific protocol freeze](v7_g1_scientific_protocol_freeze.md) 冻结。

## 证据边界与剩余技术债务

- 结论只覆盖一个冻结的 P1i 数据集和 128 个 `valid_iid` 样本；没有 test、sealed、OOD、external benchmark 或 high-resolution deployment 证据。
- 这里的误差比较不包含 latency、speedup、Pareto 或 FVM 结论；这些问题属于后续独立评价范围。
- `mean ± sample SD` 只描述三个 seed 的重复性；统计 preregistration 明确不以 n=3 seed 的 p-value 作为主要证据。
- capacity-matched Vanilla 的 seed variance 很高，后续报告应保留逐 seed 结果，不能只给均值。
- 最佳 checkpoint 的二进制和 predictions 仍在 devbox receipt 目录，不作为 GitHub artifact 提交；若 devbox 清理，需要先做受控归档并重新计算 SHA。
- 历史 V1–V6 runner、smoke/development wrapper 继续作为 read-only historical oracle；本结果不表示 legacy tree 已清理。

## 复核入口

- [G1 scientific protocol freeze](v7_g1_scientific_protocol_freeze.md)
- [G1 historical training dynamics](v7_g1_historical_training_dynamics.md)
- [G1 formal launch manifest](../configs/heat3d_v7/v7_g1_formal_launch_manifest.json)
- [G1 statistical preregistration](../configs/heat3d_v7/v7_g1_statistical_preregistration.json)
- [G1 support artifact freeze](../configs/heat3d_v7/v7_support_artifact_freeze.json)
- [G1 parameter fairness contract](../configs/heat3d_v7/v7_parameter_fairness_contract.json)
