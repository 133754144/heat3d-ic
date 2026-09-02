# V7 G1 正式训练结果与开发记录

## 文档定位

本文件记录 V7 G1 正式训练矩阵的工程结果、可复现 provenance 和有限范围内的结果解读。它不是新的实验协议，也不修改 V6 frozen evidence、模型、损失、数据划分或 checkpoint-selection 规则。

原始 formal evidence 来自 devbox：`/tmp/v7_g1_formal_runs/`；持久化副本位于 Git-ignored 的 `research_artifacts/v7_g1_formal_archive/`，逐文件 manifest 见 [G1 formal archive manifest](v7_g1_formal_archive_manifest.json)。H1/H1b/H3/H4 的已完成统计仍采用各 registered run 的 native 1024-point `valid_iid` evaluation；H2 的正式 primary 已切换到冻结的 240825 common domain，但在冻结 route envelope 冲突处 fail-closed，未产生正式 H2 metric/effect/CI。已存在的 240825-node 结果不删除，也未将不完整临时输出写入 archive。

本文中的 checkpoint 均为按预注册的 `valid_iid.sample_first_relative_rmse_pct` 选出的最佳 checkpoint；旧版 mean±SD 表原样保留，新增的 native 1024-point mean±SD 和 preregistered paired statistics 分开标注。

## 执行状态与 provenance

| 项目 | 冻结值/观测值 |
| --- | --- |
| 矩阵 | `7 variants × 3 seeds = 21 runs` |
| 最终状态 | training `COMPLETE`；21 个 receipt 均为 `COMPLETE`；H2 full-field `FAIL_CLOSED` |
| 数据集 | `heat3d_v6_p1i_continuous_physics1024_v1` |
| split | train 768；`valid_iid` 128；未读取 `test_iid` / sealed |
| dataset manifest SHA256 | `f19987c659968c2ac14eade1f1ef7e206c8f7eeb94f58fde5897d6e765978514` |
| full-field archive SHA256 | `49023ac1205b8e7cf7c5bf782b89fcdb34997704b3f9aa2fb2d46cf1a59163cb` |
| V7 formal code SHA | `191a7a06a681556f575a1c04e2b61cb13363efe1` |
| recorded Git commit | `78e7651bab5ef41a8ca4e42c45f64b1b98f04ea7` |
| matrix finished at | `2026-08-31T17:34:34.478207+00:00` |
| evidence flags | `publication_evidence=true`、`scientific_evidence_eligible=true`（21/21） |
| safety flags | `test_iid=false`、`sealed=false`、`solver=false`、`new_data=false`（21/21） |
| training state | `TRAINING_COMPLETE`；21/21 formal runs；失败 0 |
| statistical state | H1/H1b/H3/H4 native `COMPLETE`；H2 240825 primary `FAIL_CLOSED` |
| archive manifest SHA256 | `8c6ea7dca9cefddd676c8ce5d1f30855547ed273f70466549bfd8ae88f3305c7` |

原始 checkpoint、prediction、log 和 receipt 未复制进 GitHub；它们已同步到上述持久化 Git-ignored archive。Git 中只保留 manifest、receipt、统计文档和本开发记录；[G1 formal archive receipt](v7_g1_formal_archive_receipt.json) 与 [G1 formal completion receipt](v7_g1_formal_completion_receipt.json) 记录完整边界。

## 共同训练契约

- 优化器：AdamW，`base_lr=5e-4`、`min_lr=5e-5`、`warmup=10`、`cosine_horizon=200`、`weight_decay=1e-4`、gradient clipping `1.0`。
- 预算：完整新 schedule，`200 epochs`，随机初始化。
- batching：B24，768 个 train sample 对应 32 batches/epoch；validation B32，128 个 `valid_iid` sample 对应 4 batches。
- seed contract：`model = optimizer = batch_build = batch_order = graph = run_seed`，seed set `{0,1,2}`。
- checkpoint selection：`valid_iid` 上的 `sample_first_relative_rmse_pct`，并以 earliest epoch 处理并列。
- 所有结果均属于同一个冻结的 P1i Full parent 及其 registry-defined variant delta；未做 post-hoc seed 删除、budget 修改或 metric 切换。

具体 launch、模型、统计和 support 契约见：[formal launch manifest](../configs/heat3d_v7/v7_g1_formal_launch_manifest.json)、[Full P1i parent](../configs/heat3d_v7/v7_g1_full_p1i.json)、[statistical preregistration](../configs/heat3d_v7/v7_g1_statistical_preregistration.json)、[support provider contract](../configs/heat3d_v7/v7_g1_support_provider_contract.json) 和 [scientific protocol freeze](v7_g1_scientific_protocol_freeze.md)。

## Historical mixed-domain summary — not for direct cross-variant comparison

下表是历史保留的 21-run mixed-domain summary；它不是本次 H2 common-domain primary，不能用于跨 variant 的直接比较。当前可用于论文级归因的 native 1024 结果见下一节；H2 的 240825 formal primary 见后文的 fail-closed 记录。

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

对应 run ID 是每个 variant 的 `_seed0`、`_seed1`、`_seed2`；完整映射以 [frozen launch manifest](../configs/heat3d_v7/v7_g1_formal_launch_manifest.json) 和归档的 formal receipt 为准。

## Native 1024-point mean±SD（本次主口径）

下面是从归档的 21 个 `evaluation_best.json` 重新计算的 native 1024-point 结果；均值和 sample SD 仍按三个 seed 的 selected-best aggregate metric 计算。它保留了上表的 mean±SD 信息，同时避免把 support variant 的既有 240825-node 汇总混入本次主口径。

| Variant | sample-first relative RMSE (%) | point-global relative RMSE (%) | raw CV RMSE (K) | source-region RMSE (K) | peak RMSE (K) | interface RMSE (K) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| **Full** | **1.692 ± 0.048** | **2.058 ± 0.081** | **1.329 ± 0.038** | **2.781 ± 0.087** | **3.714 ± 0.022** | **0.719 ± 0.034** |
| `no_film` | 1.986 ± 0.025 | 2.265 ± 0.036 | 1.503 ± 0.039 | 2.979 ± 0.045 | 4.299 ± 0.206 | 0.872 ± 0.010 |
| `layout_agnostic_stratified_support` | 2.177 ± 0.040 | 2.386 ± 0.027 | 1.677 ± 0.016 | 3.332 ± 0.214 | 5.146 ± 0.484 | 0.843 ± 0.067 |
| `cv_only_support` | 2.207 ± 0.011 | 2.455 ± 0.019 | 1.693 ± 0.019 | 3.454 ± 0.091 | 4.696 ± 0.383 | 0.790 ± 0.055 |
| `vanilla_RIGNO` | 22.389 ± 1.568 | 22.342 ± 3.236 | 16.814 ± 2.219 | 17.932 ± 2.435 | 19.537 ± 3.851 | 2.989 ± 0.180 |
| `vanilla_RIGNO_capacity_matched` | 25.162 ± 11.003 | 24.021 ± 5.459 | 17.993 ± 4.478 | 20.527 ± 3.842 | 23.275 ± 4.329 | 3.373 ± 0.965 |
| `physics_scale_only` | 235.066 ± 0.184 | 172.398 ± 1.144 | 172.770 ± 0.151 | 39.153 ± 2.634 | 217.466 ± 9.970 | 56.141 ± 1.946 |

## 预注册 paired effect 与 bootstrap CI

effect 定义为 `ablation_error - Full_error`，正值表示 Full 更低。H1/H1b 的 primary 始终是 `point_global_relative_rmse_pct`；不能用 `sample_first_relative_rmse_pct` 替换。每次 bootstrap 都先在 seed 层和 seed 内的 128 个 `valid_iid` sample 层有放回重采样，再重新计算 frozen aggregate functional；共 10,000 次，随机种子 `20260829`，报告 percentile 95% CI。

| Hypothesis | Primary comparison | Full pooled | Ablation pooled | Effect | Paired median / p90 / p95 / worst-10 mean | Per-seed effects (0,1,2) | 95% CI | Claim status |
| --- | --- | ---: | ---: | ---: | --- | --- | --- | --- |
| H1 | Full vs `vanilla_RIGNO` | 2.05872 | 22.49781 | 20.43909 | 15.75036 / 38.99510 / 45.75685 / 80.51393 | 16.98945, 23.33246, 20.53158 | [17.12366, 23.60488] | **SUPERIORITY_SUPPORTED** |
| H1b | Full vs `vanilla_RIGNO_capacity_matched` | 2.05872 | 24.43118 | 22.37245 | 16.38835 / 40.31416 / 66.56916 / 116.63157 | 28.07646, 17.40245, 20.41153 | [17.44747, 27.71174] | **SUPERIORITY_SUPPORTED** |
| H2 / generic | Full vs `layout_agnostic_stratified_support` | 2.78170 K | 3.33289 K | 0.55119 K | 0.28551 / 2.71986 / 4.15901 / 7.87018 K | 0.31336, 0.59858, 0.74026 K | [0.13854, 0.97514] K | **FAIL_CLOSED_NOT_ESTIMABLE_NATIVE_1024** |
| H2 / volume-only | Full vs `cv_only_support` | 2.78170 K | 3.45463 K | 0.67293 K | 0.32133 / 2.89294 / 3.85150 / 7.84755 K | 0.57619, 0.63287, 0.80949 K | [0.37064, 0.99466] K | **FAIL_CLOSED_NOT_ESTIMABLE_NATIVE_1024** |
| H3 | Full vs `no_film` | 1.69191% | 1.98608% | 0.29417 pp | 0.30947 / 0.99768 / 1.14901 / 1.52882 pp | 0.29505, 0.23859, 0.34888 pp | [0.21456, 0.37041] pp | **SUPERIORITY_SUPPORTED** |
| H4 | Full vs `physics_scale_only` | 1.32955 K | 172.76972 K | 171.44018 K | 147.16832 / 260.28847 / 279.70957 / 321.02895 K | 171.55671, 171.46143, 171.30333 K | [164.25621, 178.46001] K | **SUPERIORITY_SUPPORTED** |

完整逐 sample effect、per-seed table 和 worst-case 行见 [per-sample effects](../research_artifacts/v7_g1_formal_archive/analysis_1024/per_sample_effects.json)、[per-seed effects](../research_artifacts/v7_g1_formal_archive/analysis_1024/per_seed_effects.md)、[hypothesis table](../research_artifacts/v7_g1_formal_archive/analysis_1024/hypothesis_effect_table.md) 和 [worst-case diagnostics](../research_artifacts/v7_g1_formal_archive/analysis_1024/worst_case_diagnostics.md)。

Superiority 只按预注册三项规则声明：paired 95% CI 排除 0、paired median 同方向、三个 seed 的 aggregate effect 同方向。H1/H1b/H3/H4 满足；H2 的 native 1024 aggregate CI 虽为正，但 support arm 有部分 sample 没有 source node，`source_region_RMSE_K` 的 paired sample unit 不对全部 128 sample 可估计，因此 H2 只作 native-support descriptive attribution 并 fail-closed，不填零、不删行，也不新增 240825-node 结果。

## H2 full-field formal 240825 primary（FAIL_CLOSED）

H2 primary 的 route 判定在读取本轮 accuracy 前已固定：`U_v2_16384_reconstruction`（`U16384→240825`）是历史 V6 canonical chain 的 primary，`U_v2_direct240825`（`U-direct-240825`）是 robustness/sensitivity route。判定依据、route config 和 SHA 见 [H2 route primary decision](../configs/heat3d_v7/v7_g1_h2_route_primary_decision.json) 与 [H2 fail-closed receipt](v7_g1_h2_fullfield_fail_closed_receipt.json)。

正式执行要求 9 个 selected-best checkpoints × 2 routes × 128 个 `valid_iid`，并在相同的 240825 physical coordinates 上计算 `source_region_RMSE_K` 及五项 secondary metrics。执行在 primary route 的 `Full_seed0 / v6p1if1_0993` 处被冻结 native `p2r_edge_indices=3074` envelope 拒绝：observed `3083`，超出 9 条 edge。改变 target、support/graph、sample population 或删除样本均会改变 frozen contract，因此未采取。primary route 仅在内存中完成 122/128 个样本，未写出 complete evaluation receipt；robustness route 未启动正式 run；任何 partial output 均未进入 archive。

| U route | Variant | Complete seeds | source-region RMSE (K), mean±SD | point-global relative RMSE (%), mean±SD | peak RMSE (K), mean±SD | paired effect `ablation−Full` | percentile 95% CI | Claim status |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| `U16384→240825` primary | Full | 0/3 | N/A | N/A | N/A | N/A | N/A | `NOT_EVALUABLE_FAIL_CLOSED` |
| `U16384→240825` primary | generic support | 0/3 | N/A | N/A | N/A | N/A | N/A | `NOT_EVALUABLE_FAIL_CLOSED` |
| `U16384→240825` primary | CV-only support | 0/3 | N/A | N/A | N/A | N/A | N/A | `NOT_EVALUABLE_FAIL_CLOSED` |
| `U-direct-240825` robustness | Full | 0/3 | N/A | N/A | N/A | N/A | N/A | `NOT_EVALUABLE_FAIL_CLOSED` |
| `U-direct-240825` robustness | generic support | 0/3 | N/A | N/A | N/A | N/A | N/A | `NOT_EVALUABLE_FAIL_CLOSED` |
| `U-direct-240825` robustness | CV-only support | 0/3 | N/A | N/A | N/A | N/A | N/A | `NOT_EVALUABLE_FAIL_CLOSED` |

因此 H2 没有合法的 paired per-sample effect、per-seed effect、pooled summary、median/p90/p95/worst-10 或 10,000-replicate bootstrap CI；不对 generic/CV-only 声明 superiority，也不从 route 差异推断 attribution robustness。该结论是执行性 FAIL-CLOSED，不是零效应或负效应。

历史核验显示两份 V6 U envelope 都明确为 `sample_count=32` 资格化 capacity；历史 `05b32ce` U16384 receipt 扩展到 `remaining_valid96`，但 Git history / frozen receipts 中没有覆盖当前 128 个 `valid_iid` 的 route-specific envelope。历史记录中的 `v6p1if1_0993` 为 `p2r_edges=3074`，本次冻结 runtime 观察为 `3083`，所以历史结果不能替代当前 V7 9-checkpoint H2 evidence，binary route equivalence 也未被宣称。

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

1. **Full vs Vanilla 支持完整 Heat3D conditioning architecture 的组合收益。** H1 在 `point_global_relative_rmse_pct` 上的 paired effect 为 `20.43909` 个百分点，95% CI `[17.12366, 23.60488]`；capacity-matched H1b 的 effect 为 `22.37245` 个百分点，95% CI `[17.44747, 27.71174]`。两个比较均满足预注册 superiority gate，但结论只限于冻结 P1i `valid_iid` native 1024-point 口径。
2. **support 只作 physics-layout-aware sparse support 的归因。** native 1024-point support arms 的 source-region aggregate effect 对 generic 和 volume-only 都为正，但部分 sample 没有 source node，paired primary unit 不完整，所以 native H2 仅作为 `native-support diagnostic / supplementary attribution`；240825 full-field H2 因 frozen envelope 冲突未可估计，不声明 superiority，也不外推为独立于 support/evaluation grid 的普遍结论。
3. **FiLM 是 secondary contribution。** H3 的 primary `sample_first_relative_rmse_pct` effect 为 `0.29417` 个百分点，95% CI `[0.21456, 0.37041]`，三个 seed 同方向；这支持在保留其它 Full context/scale 路径时，关闭 FiLM 会变差，但 FiLM 不是 Full 与 Vanilla 全部差异的替代解释。
4. **scale correction 在当前 P1i formulation 下是关键。** H4 的 primary `raw_K_CV_RMSE_K` effect 为 `171.44018 K`，95% CI `[164.25621, 178.46001]`，三个 seed 同方向。该结论仅是当前 frozen P1i formulation 下的组件归因，不外推为 test、OOD 或 external superiority。
5. **capacity-matched Vanilla 的高 seed variance 必须保留。** 其 sample-first mean±SD 为 `25.162 ± 11.003%`，逐 seed 为 `37.727% / 17.251% / 20.507%`；该逐 seed 表不能被只报告均值的摘要替代。旧表中的最佳 epoch 差异也只作描述，不能据此修改冻结的 200-epoch budget。

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
- 最佳/最终 checkpoint、predictions、history/log 和 run provenance 已持久化到 `research_artifacts/v7_g1_formal_archive/`；该目录 Git-ignored，不作为 GitHub 大型 artifact 提交，逐文件 SHA 见 archive manifest。
- H2 的正式 primary domain 是冻结的 common 240825 full field，但本轮在 envelope guard 处 fail-closed；未归档不完整的 240825 prediction/metric。既有 240825-node evidence 仍保留并必须沿用其既有 receipt，不得与 native 1024 supplementary 口径混用。
- 当前 G1 completion receipt 是 `G1_FORMAL_CLOSEOUT_BLOCKED_H2_FAIL_CLOSED`；training complete 不等于 H2 statistical closeout complete。future evaluation-only test unlock 不是本轮 blocker，`test_iid`/sealed 仍保持未访问。
- 历史 V1–V6 runner、smoke/development wrapper 继续作为 read-only historical oracle；本结果不表示 legacy tree 已清理。

## 复核入口

- [G1 scientific protocol freeze](v7_g1_scientific_protocol_freeze.md)
- [G1 historical training dynamics](v7_g1_historical_training_dynamics.md)
- [G1 formal launch manifest](../configs/heat3d_v7/v7_g1_formal_launch_manifest.json)
- [G1 statistical preregistration](../configs/heat3d_v7/v7_g1_statistical_preregistration.json)
- [G1 support artifact freeze](../configs/heat3d_v7/v7_support_artifact_freeze.json)
- [G1 parameter fairness contract](../configs/heat3d_v7/v7_parameter_fairness_contract.json)
