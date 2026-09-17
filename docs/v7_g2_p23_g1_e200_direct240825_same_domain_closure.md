# V7-G2 P23：G1 e200 × Therm-FM 同域 valid-only 对比

状态：`COMPLETE_VALID_ONLY_REPRODUCTION_PASS`。

本轮只使用冻结的 P1i `valid_iid=128`、`65×65×57=240,825` 节点和同一
`full_fields.h5`。没有训练、调参、checkpoint 重选，也没有读取新的
`test_iid`、sealed IID 或 DeepOHeat official100。

## 资产与同域 gate

- Heat3D 使用 G1 `Full_seed0/1/2` 的 `params_best_sample_first.pkl` 与已经存在的
  `U_v2_direct240825/predictions_best.npz`；没有重新推理。
- 三个 checkpoint SHA 分别为
  `7ee050c9…86517`、`1821a8ea…6f019`、`2ed4c5d0…b151b`；三份 direct prediction
  SHA 分别为 `a261c934…ba801`、`4167d3bb…c33e5d`、`0f67cb03…f66cea`。
- 原始 G1 NPZ 是 stacked `sample_ids + prediction_deltaT_K[128,240825]` 容器；新增适配器只
  将每个 sample id 重封装为 evaluator 所需 NPZ key，数值、case 顺序、节点顺序均不变。
- Therm-FM 使用 P22/P23 loss-selected seed0/1/2 predictions，保持 native `z,x,y` 到共同
  evaluator 的确定性转置；Poseidon-T 是 external pretrained transfer，不是同预算 scratch baseline。
- 两侧 sample IDs 与 frozen valid manifest 完全一致；共享 truth archive、坐标、control-volume、
  layer-id 的 SHA 分别为 `49023ac1…163cb`、`c293ae6…0daa4`、`0bfe0b3e…9d5da`、
  `36743975…d890b9`。

完整 provenance 见
`docs/results/v7_g2_p23_g1_e200_u_v2_direct240825_domain_identity_receipt.json`。

## Reproduction gate

`REPRODUCTION_PASS`。

- G1 direct-route 的 sample-first、point-global、peak-RMSE 与历史 direct receipts 的三 seed
  差异均为 0（seed2 sample-first 仅 `8.9e-16`）。历史 `raw_K_CV_RMSE_K` 与本 evaluator 的
  unweighted `rmse_K` 不等价，未混用。
- Therm-FM 与既有 P23 common-evaluator aggregate/per-seed rows 完全一致；P22 legacy replay
  receipt 仍为 `REPLAY_REPRODUCTION_PASS`（最大报告差异 `7.49e-5`）。P22 的旧
  sample-first/peak 定义与 P23 的 CV-weighted sample-first、`abs(max(pred)-max(truth))` peak
  定义分开记录，不把语义不同的数值强行相等。

## Unified common evaluator

所有 `±` 均为 **SD across training seeds（训练随机种子间标准差）**；不是把
`3×128` 行当作 384 个独立重复。

| 模型/seed | sample-first rel. RMSE % | point-global rel. RMSE % | CV-weighted point-global % | RMSE K | MAE K | peak MAE K | peak RMSE K | true-hotspot RMSE K |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| G1 Heat3D seed0 | 2.9216 | 3.2859 | 3.1723 | 2.6094 | 1.9656 | 3.7371 | 5.0957 | 5.4090 |
| G1 Heat3D seed1 | 3.2313 | 3.6196 | 3.4925 | 2.8744 | 2.1890 | 3.7756 | 5.2022 | 6.0636 |
| G1 Heat3D seed2 | 3.2258 | 3.5304 | 3.4914 | 2.8036 | 2.1382 | 3.8956 | 5.3166 | 5.6823 |
| **G1 Heat3D mean ± SD** | **3.1262 ± 0.1772** | **3.4786 ± 0.1728** | **3.3854 ± 0.1846** | **2.7625 ± 0.1372** | **2.0976 ± 0.1171** | **3.8028 ± 0.0827** | **5.2049 ± 0.1105** | **5.7183 ± 0.3287** |
| Therm-FM seed0 | 2.4950 | 3.1424 | 3.1670 | 2.4955 | 1.6026 | 3.8559 | 5.4448 | 4.1010 |
| Therm-FM seed1 | 2.4253 | 2.6919 | 2.7617 | 2.1378 | 1.4590 | 4.1383 | 5.7778 | 3.7903 |
| Therm-FM seed2 | 2.8713 | 3.6125 | 3.6630 | 2.8689 | 1.8473 | 3.8566 | 5.4929 | 4.0668 |
| **Therm-FM mean ± SD** | **2.5972 ± 0.2399** | **3.1490 ± 0.4603** | **3.1972 ± 0.4514** | **2.5007 ± 0.3656** | **1.6363 ± 0.1963** | **3.9503 ± 0.1629** | **5.5718 ± 0.1800** | **3.9860 ± 0.1704** |

这张表是同域 valid-only 描述性比较；Therm-FM 的 external pretraining 与 dense 741-channel
输入意味着不能声称 same-information-budget。

## Paired case bootstrap

每个 case 先在三个 training seeds 内求均值，再以 physical case 为单位进行 10,000 次 paired
bootstrap（seed `230023`）。差值定义为 `Heat3D − Therm-FM`，区间为 percentile 95% CI，
结论条件化于冻结的三 seed cohort。

| 指标 | observed difference | 95% CI |
|---|---:|---:|
| sample-first relative RMSE % | 0.5290 | [0.1751, 0.8828] |
| RMSE K | 0.4462 | [0.0996, 0.7635] |
| MAE K | 0.4613 | [0.1134, 0.7809] |
| peak-temperature absolute error K | −0.1475 | [−0.9030, 0.5714] |
| true-hotspot-region RMSE K | 1.6900 | [1.2293, 2.1391] |

这些是预注册 paired estimates，不自动转写为普遍或统计学 superiority claim。

## Frozen condition analysis

严格复用先于误差读取冻结的 source-count、k-region-count 和五个输入侧 tercile/range bins；
没有根据结果改 bin。完整 21 个 bin × 三个指标见
`docs/results/v7_g2_p23_g1_e200_u_v2_direct240825_condition_analysis.json`。

`sample-first` 的 `Heat3D − Therm-FM` 差值如下（其余 RMSE/hotspot 差值在 JSON 中逐项保存）：

| 条件 | low | mid | high |
|---|---:|---:|---:|
| source_count | +0.2945 | +0.0514 | +1.0357 |
| k_region_count | +0.6676 | +0.8472 | +0.2513 |
| total_power_W | −0.2889 | +0.6695 | +1.2097 |
| source_power_heterogeneity | +0.3394 | +0.5169 | +0.7305 |
| top_h_W_m2K | −0.1732 | +0.6656 | +1.0979 |
| bottom_h_W_m2K | −0.6337 | +0.2479 | +1.9664 |
| material_conductivity_heterogeneity | −0.0153 | +0.2331 | +1.3624 |

该表是 robustness 描述，不支持无条件 winner；低 power/低 HTC bins 的局部负差值也不能作为
事后 subgroup 选择。当前安全解释是：冻结 valid128 上没有形成一致的 Heat3D full-field 优势，
hotspot 指标差值整体偏向 Therm-FM，但 peak-temperature absolute error 的 paired CI 跨 0。

## Artifacts

- 统一评估：`docs/results/v7_g2_p23_g1_e200_u_v2_direct240825_vs_thermfm_common_eval.json`
- 逐 case 表：同名 `.csv`
- paired bootstrap：`docs/results/v7_g2_p23_g1_e200_u_v2_direct240825_paired_bootstrap.json`
- reproduction gate：`docs/results/v7_g2_p23_g1_e200_u_v2_direct240825_reproduction_gate.json`
- 三份无损容器适配 receipt 与 prediction spec：同目录

V6 historical full-field 结果没有进入本轮 paired statistics；本轮是独立的
`V7 G1 Full e200 U-v2-direct240825 ×3 vs Therm-FM ×3` valid-only cohort。
