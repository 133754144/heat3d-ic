# V7 G1 正式训练结果与开发记录

## 文档定位

本文件记录 V7 G1 正式训练矩阵的工程结果、可复现 provenance 和有限范围内的结果解读。它不是新的实验协议，也不修改 V6 frozen evidence、模型、损失、数据划分或 checkpoint-selection 规则。

原始 formal evidence 来自 devbox：`/tmp/v7_g1_formal_runs/`；持久化副本位于 Git-ignored 的 `research_artifacts/v7_g1_formal_archive/`，逐文件 manifest 见 [G1 formal archive manifest](v7_g1_formal_archive_manifest.json)。H1/H1b/H3/H4 的已完成统计仍采用各 registered run 的 native 1024-point `valid_iid` evaluation；H2 的正式 primary 已按冻结 G1-native graph semantics + V6 U strategy 在 240825 common domain 完成；V6 `p2r=3074` 仅保留为 historical reproducibility diagnostic。已存在的 240825-node 结果不删除。

本文中的 checkpoint 均为按预注册的 `valid_iid.sample_first_relative_rmse_pct` 选出的最佳 checkpoint；旧版 mean±SD 表原样保留，新增的 native 1024-point mean±SD 和 preregistered paired statistics 分开标注。

## 执行状态与 provenance

| 项目 | 冻结值/观测值 |
| --- | --- |
| 矩阵 | `7 variants × 3 seeds = 21 runs` |
| 最终状态 | training `COMPLETE`；21 个 receipt 均为 `COMPLETE`；H2 full-field `COMPLETE` |
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
| statistical state | H1/H1b/H3/H4 native `COMPLETE`；H2 240825 primary + route robustness `COMPLETE` |
| archive manifest SHA256 | `396326724cb4f151e2e1f5f8b5c70ea1626c83fb5bebece3019e27d56590d80c` |

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

下表是历史保留的 21-run mixed-domain summary；它不是本次 H2 common-domain primary，不能用于跨 variant 的直接比较。当前可用于论文级归因的 native 1024 结果见下一节；H2 的 240825 formal primary 与 U-route robustness 见后文。

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

## Native 1024-point mean±SD（正式训练主表；H2 supplementary diagnostic）

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
| H2a (native-1024 supplementary) | Full vs `layout_agnostic_stratified_support` | 2.78170 K | 3.33289 K | 0.55119 K | 0.28551 / 2.71986 / 4.15901 / 7.87018 K | 0.31336, 0.59858, 0.74026 K | [0.13854, 0.97514] K | **FAIL_CLOSED_NOT_ESTIMABLE_NATIVE_1024** |
| H2b (native-1024 supplementary) | Full vs `cv_only_support` | 2.78170 K | 3.45463 K | 0.67293 K | 0.32133 / 2.89294 / 3.85150 / 7.84755 K | 0.57619, 0.63287, 0.80949 K | [0.37064, 0.99466] K | **FAIL_CLOSED_NOT_ESTIMABLE_NATIVE_1024** |
| H3 | Full vs `no_film` | 1.69191% | 1.98608% | 0.29417 pp | 0.30947 / 0.99768 / 1.14901 / 1.52882 pp | 0.29505, 0.23859, 0.34888 pp | [0.21456, 0.37041] pp | **SUPERIORITY_SUPPORTED** |
| H4 | Full vs `physics_scale_only` | 1.32955 K | 172.76972 K | 171.44018 K | 147.16832 / 260.28847 / 279.70957 / 321.02895 K | 171.55671, 171.46143, 171.30333 K | [164.25621, 178.46001] K | **SUPERIORITY_SUPPORTED** |

完整逐 sample effect、per-seed table 和 worst-case 行见 [per-sample effects](../research_artifacts/v7_g1_formal_archive/analysis_1024/per_sample_effects.json)、[per-seed effects](../research_artifacts/v7_g1_formal_archive/analysis_1024/per_seed_effects.md)、[hypothesis table](../research_artifacts/v7_g1_formal_archive/analysis_1024/hypothesis_effect_table.md) 和 [worst-case diagnostics](../research_artifacts/v7_g1_formal_archive/analysis_1024/worst_case_diagnostics.md)。

上表中的 native-1024 H2a/H2b 行是保留的 historical/superseded supplementary diagnostic；其中 `FAIL_CLOSED_NOT_ESTIMABLE_NATIVE_1024` 只描述旧 native-support 口径的可估计性，不能被解读为当前 G1 状态。当前 authoritative H2 结论只来自下一节的 240825 common-domain primary。

Superiority 只按预注册三项规则声明：paired 95% CI 排除 0、paired median 同方向、三个 seed 的 aggregate effect 同方向。H1/H1b/H3/H4 满足；native 1024 H2 保留为 supplementary diagnostic；正式 H2 已在共同 240825 domain 上对全部 128 `valid_iid` sample 完成 paired evaluation 与 bootstrap。

## H2 full-field formal 240825 primary and U-route robustness（COMPLETE）

H2 governance amendment 已在任何 H2 accuracy 产生前冻结。H2 是一个 hypothesis group，包含两个预注册 contrast：H2a（Full vs generic support）与 H2b（Full vs CV-only support）。V6 `p2r=3074` 仅保留为 historical reproducibility diagnostic，不再是 H2 scientific gate 或 envelope source。正式 H2 使用 G1-native 1024 graph semantics 与 frozen V6 U query/reconstruction strategy；primary 是 `U16384→240825`，`U-direct-240825` 仅作 route robustness。

Gate A/B 为 geometry-only。Full_seed0 / `v6p1if1_0993` 的 G1-native anchor real edge count 为 `p2r=3082`、`r2r=4074`，packed count 为 `3083/4075`；完整 geometry audit 覆盖 `9×128=1152` native records，两个 route 各 1152 records。native graph、support、radius、real edge set 均未被 U adapter 改变。

Gate B observed maximum real edge count + exactly one mandatory dummy 得到 frozen execution capacities：native `p2r/r2p/r2r=3175/3175/4325`；U16384 query `p2r/r2p/r2r=3175/45101/4325`；U240825 query `p2r/r2p/r2r=3175/564489/4325`。相对历史 envelope 的变化是 execution-shape-only amendment；real multisets、valid tensor prefix 和 dummy suffix invariance 均 PASS。

### H2 primary full-field variant summaries

下表保留每个 route、variant 的 3-seed mean±sample-SD；主 metric 是 `source_region_RMSE_K`。

| Route | Variant | source-region RMSE (K) | point-global relative RMSE (%) | peak RMSE (K) |
| --- | --- | ---: | ---: | ---: |
| `U16384→240825 primary` | Full | 3.80482 ± 0.149586 | 3.38519 ± 0.16284 | 5.14703 ± 0.105692 |
| `U16384→240825 primary` | generic support | 5.55262 ± 0.0723057 | 4.56933 ± 0.0876578 | 8.49609 ± 0.392335 |
| `U16384→240825 primary` | CV-only support | 5.64211 ± 0.297967 | 5.14206 ± 0.398149 | 8.46651 ± 0.392975 |
| `U-direct-240825 robustness` | Full | 3.8653 ± 0.151618 | 3.47862 ± 0.172782 | 5.20486 ± 0.110457 |
| `U-direct-240825 robustness` | generic support | 5.61211 ± 0.0703095 | 4.68699 ± 0.107174 | 8.5608 ± 0.389306 |
| `U-direct-240825 robustness` | CV-only support | 5.68564 ± 0.23868 | 5.24518 ± 0.365855 | 8.51397 ± 0.327525 |

### Preregistered paired effect table

Effect is `ablation_error − Full_error`; positive favors Full. CI is the 10,000-replicate two-level percentile bootstrap (`seed=20260829`). Superiority requires CI > 0, paired median > 0, and seed0/1/2 effects all > 0.

| Route | Comparison | Full pooled | Ablation pooled | Effect | Paired median / p90 / p95 / worst-10 | Per-seed effects (0,1,2) | 95% CI | Claim status |
| --- | --- | ---: | ---: | ---: | --- | --- | --- | --- |
| `U_v2_16384_reconstruction` | H2a: Full vs generic support | 3.80678 | 5.55293 | 1.74615 | 1.56164 / 4.05032 / 4.84999 / 8.48932 | 1.96483, 1.56857, 1.70998 | [1.48199, 2.04268] | `SUPERIORITY_SUPPORTED` |
| `U_v2_16384_reconstruction` | H2b: Full vs CV-only support | 3.80678 | 5.64735 | 1.84057 | 1.7461 / 4.07155 / 5.16232 / 8.66633 | 1.66052, 1.71218, 2.13916 | [1.5307, 2.18014] | `SUPERIORITY_SUPPORTED` |
| `U_v2_direct240825` | H2a: Full vs generic support | 3.86729 | 5.6124 | 1.74511 | 1.5709 / 4.08754 / 4.87274 / 8.48955 | 1.96409, 1.56616, 1.71016 | [1.48176, 2.04234] | `SUPERIORITY_SUPPORTED` |
| `U_v2_direct240825` | H2b: Full vs CV-only support | 3.86729 | 5.68898 | 1.8217 | 1.77288 / 4.10023 / 5.0865 / 8.65195 | 1.71689, 1.6674, 2.07672 | [1.53548, 2.13621] | `SUPERIORITY_SUPPORTED` |

### U-route robustness

两条 route 最终都落在同一 240825-node full-field physical coordinates，并复用相同 source/interface masks、CV weights 和 truth field。以下 route difference 定义为 `U-direct-240825 − U16384→240825`；它只评价 route sensitivity，不替换 primary。

| Variant | Metric | Direct seed values | U16384→240825 seed values | Direct − U16384 seed differences |
| --- | --- | --- | --- | --- |
| Full | `source_region_RMSE_K` | 3.72593, 4.02675, 3.84323 | 3.66787, 3.96445, 3.78215 | 0.0580629, 0.0622964, 0.0610834 |
| Full | `point_global_relative_rmse_pct` | 3.28586, 3.61958, 3.53041 | 3.2019, 3.51316, 3.44052 | 0.0839654, 0.106418, 0.0898947 |
| Full | `peak_RMSE_K` | 5.09574, 5.20224, 5.3166 | 5.04577, 5.13868, 5.25666 | 0.0499695, 0.0635642, 0.0599491 |
| layout_agnostic_stratified_support | `source_region_RMSE_K` | 5.69002, 5.59291, 5.55339 | 5.6327, 5.53302, 5.49213 | 0.0573221, 0.0598871, 0.0612639 |
| layout_agnostic_stratified_support | `point_global_relative_rmse_pct` | 4.81068, 4.62191, 4.62836 | 4.67047, 4.52218, 4.51535 | 0.140209, 0.0997307, 0.113019 |
| layout_agnostic_stratified_support | `peak_RMSE_K` | 9.01033, 8.33631, 8.33577 | 8.94912, 8.27006, 8.26909 | 0.0612142, 0.0662437, 0.0666747 |
| cv_only_support | `source_region_RMSE_K` | 5.44282, 5.69415, 5.91996 | 5.32838, 5.67663, 5.92131 | 0.114439, 0.0175202, -0.00135626 |
| cv_only_support | `point_global_relative_rmse_pct` | 5.00856, 5.66657, 5.06041 | 4.82133, 5.58768, 5.01716 | 0.187235, 0.0788922, 0.0432434 |
| cv_only_support | `peak_RMSE_K` | 8.19922, 8.48977, 8.85293 | 8.06984, 8.47401, 8.85568 | 0.129383, 0.0157554, -0.0027509 |

因此正式 H2 claim 只读取 `U16384→240825` primary effect rows；direct route 作为预注册 robustness。若两条 route 的 claim status/方向一致，attribution direction 对 reconstruction strategy 稳健；否则仅报告 route sensitivity，不升级主张。

native-1024 H2 结果保留为 `native-support diagnostic / supplementary attribution`，不再作为 preregistered H2 primary。

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
2. **support 只作 physics-layout-aware sparse support 的归因。** 在正式 H2 的 240825 common-domain primary（`U16384→240825`）上，Full 相对 generic support 的 source-region effect 为 `1.74615 K`，95% CI `[1.48199, 2.04268] K`；相对 CV-only support 为 `1.84057 K`，95% CI `[1.53070, 2.18014] K`，两者均满足预注册 superiority gate，且三个 seed 同方向。该结论只归因于 frozen P1i 中 physics-layout-aware sparse support 相对 Full 的差异，不外推为独立于 support/evaluation grid 的普遍结论；native 1024 结果保留为 supplementary diagnostic。
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
- H2 的正式 primary domain 是冻结的 common 240825 full field；G1-native anchor、U adapter、geometry capacity、padding invariance、primary/robustness evaluation 与统计结果均已归档。既有 240825-node evidence 仍保留并沿用其既有 receipt；native 1024 仅作 supplementary diagnostic。
- 当前 G1 completion receipt 为 `G1_FORMAL_CLOSEOUT_COMPLETE`（receipt SHA 见归档）；training、H2 statistical closeout 均已完成。future evaluation-only test unlock 不是 G1 blocker，`test_iid`/sealed 仍保持未访问。
- 历史 V1–V6 runner、smoke/development wrapper 继续作为 read-only historical oracle；本结果不表示 legacy tree 已清理。

## 复核入口

- [G1 scientific protocol freeze](v7_g1_scientific_protocol_freeze.md)
- [G1 historical training dynamics](v7_g1_historical_training_dynamics.md)
- [G1 formal launch manifest](../configs/heat3d_v7/v7_g1_formal_launch_manifest.json)
- [G1 statistical preregistration](../configs/heat3d_v7/v7_g1_statistical_preregistration.json)
- [G1 support artifact freeze](../configs/heat3d_v7/v7_support_artifact_freeze.json)
- [G1 parameter fairness contract](../configs/heat3d_v7/v7_parameter_fairness_contract.json)
- [G1 final science seal](v7_g1_final_science_seal_receipt.json)
- [G1 publication summary](v7_g1_publication_summary.md)
- [G1 documentation consistency receipt](v7_g1_documentation_consistency_receipt.json)
