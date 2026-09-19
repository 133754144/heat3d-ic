# V7 G2 model performance report

## 1. Scope and governance

本报告只汇总已经冻结的 valid-only 证据。P1i 主比较使用同一
valid_iid=128、同一 truth archive 和明确分开的评价域；DeepOHeat 数据集结果只
作为 supplemental comparison，不能与 P1i 主表混排。`test_iid` 已有 documented
historical access，并按先冻结的 input-only selection receipt 做过一次
visualization-only inference；这些 prediction、slice metrics 和图像不进入模型选择、
checkpoint selection、Table A/B、bootstrap 或任何 accuracy claim，本轮没有新的 test
访问。`sealed IID` 仍未打开，是当前唯一的最终 confirmatory holdout；DeepOHeat
official100 也未访问。

所有均值和离散度均写作 **mean ± SD across training seeds**（训练随机种子间标准
差），不是把 3×128 个值当作独立重复。所有结果来自冻结 checkpoint、prediction
archive、evaluator 和 selection rule；没有训练、调参或 checkpoint reselection。

主指标合同：

- Table A / native1024：point_global_relative_rmse_pct。
- Table B / full-field240825：sample_first_relative_rmse_pct。
- Table A 与 Table B 不跨域排名。

## 2. Table A — P1i native1024

域为 1024 个 native physical points/case；模型为 Heat3D V7 P1i-e200、GINO 和
Transolver。形状指标严格限定为 V4/V5 定义：Corr_point、Amp_range、Corr_CV、
Amp_CVRMS；完整逐样本结果见
docs/results/v7_g2_final_evidence_aggregates.json。

### Per-seed metrics

| model | seed | sample-first rel RMSE % | point-global rel RMSE % | RMSE K | MAE K | Corr_point % | Amp_range % | Corr_CV % | Amp_CVRMS % |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Heat3D | 0 | 1.7161 | 2.0286 | 1.5819 | 1.0338 | 99.1460 | 97.4830 | 99.3016 | 100.1223 |
| Heat3D | 1 | 1.7228 | 2.1488 | 1.6756 | 1.0448 | 99.0042 | 98.1966 | 99.2083 | 99.8008 |
| Heat3D | 2 | 1.6368 | 1.9956 | 1.5561 | 0.9964 | 99.1620 | 98.0578 | 99.3048 | 99.7141 |
| GINO | 0 | 15.4950 | 18.1751 | 14.1726 | 10.4897 | 97.2552 | 117.7856 | 96.7856 | 102.8677 |
| GINO | 1 | 15.5856 | 19.1196 | 14.9091 | 10.6037 | 94.5680 | 122.5944 | 94.4948 | 102.7361 |
| GINO | 2 | 13.9609 | 15.8998 | 12.3984 | 9.2314 | 97.7953 | 116.0549 | 97.6090 | 103.0083 |
| Transolver | 0 | 16.2657 | 18.7898 | 14.6520 | 10.8173 | 96.9950 | 95.5564 | 97.2720 | 101.2695 |
| Transolver | 1 | 14.8658 | 17.5356 | 13.6740 | 10.0829 | 97.0823 | 94.5360 | 97.3797 | 100.6479 |
| Transolver | 2 | 16.8770 | 19.3441 | 15.0842 | 11.1778 | 96.6711 | 95.0135 | 96.9570 | 103.8861 |

### Aggregate (mean ± SD across training seeds)

| model | sample-first rel RMSE % | point-global rel RMSE % (primary) | RMSE K | MAE K | Corr_point % | Amp_range % | Corr_CV % | Amp_CVRMS % |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Heat3D V7 P1i-e200 | 1.6919 ± 0.0478 | **2.0577 ± 0.0806** | 1.6045 ± 0.0628 | 1.0250 ± 0.0254 | 99.1041 ± 0.0868 | 97.9125 ± 0.3784 | 99.2716 ± 0.0548 | 99.8791 ± 0.2151 |
| GINO | 15.0139 ± 0.9130 | **17.7315 ± 1.6551** | 13.8267 ± 1.2906 | 10.1083 ± 0.7615 | 96.5395 ± 1.7286 | 118.8116 ± 3.3883 | 96.2965 ± 1.6137 | 102.8707 ± 0.1361 |
| Transolver | 16.0028 ± 1.0310 | **18.5565 ± 0.9266** | 14.4701 ± 0.7225 | 10.6927 ± 0.5580 | 96.9161 ± 0.2167 | 95.0353 ± 0.5106 | 97.2029 ± 0.2197 | 101.9345 ± 1.7185 |

### P8 hierarchical seed+case paired bootstrap

这是 P8 预注册的 hierarchical bootstrap（seed slots with replacement，再在每个
选中的 seed 内对 128 physical cases 配对重采样；10,000 次，seed=20260907）。
差值定义为 Heat3D − comparator；负值代表 Heat3D 较低。point estimate 与主表
estimator 精确一致。结果文件为
docs/results/v7_g2_final_p8_hierarchical_bootstrap_native.json。

| contrast / metric | estimate | 95% percentile CI |
|---|---:|---:|
| Heat3D − GINO, sample-first % | -13.3220 | [-14.7857, -11.8276] |
| Heat3D − GINO, point-global % | -15.6738 | [-17.6361, -13.5833] |
| Heat3D − GINO, RMSE K | -12.2222 | [-13.8871, -10.5491] |
| Heat3D − GINO, MAE K | -9.0833 | [-10.2352, -7.9694] |
| Heat3D − Transolver, sample-first % | -14.3109 | [-16.0434, -12.6220] |
| Heat3D − Transolver, point-global % | -16.4988 | [-18.2251, -14.7445] |
| Heat3D − Transolver, RMSE K | -12.8655 | [-14.2372, -11.4526] |
| Heat3D − Transolver, MAE K | -9.6677 | [-10.7557, -8.6160] |

这些区间是冻结 valid cohort 上的条件性描述性推断，不是 universal-domain
superiority 证明。P8 预注册 bootstrap 的 metric list 不包含 Corr/Amp 形状诊断；
这些 shape fields 仅报告 seed-level mean ± SD，不用未预注册的 CI 反向扩展结论。

## 3. Table B — P1i full-field240825

域为同一 valid_iid=128 的 65×65×57=240,825 个 physical nodes。Heat3D 使用
冻结的 U-v2 direct-query dense sidecar；Therm-FM 使用 native 741-channel dense
input 与 loss-selected checkpoint。Therm-FM 是
external-pretrained dense-input capability reference，不是
same-information-budget baseline。

### Per-seed primary and temperature metrics

| model | seed | sample-first % | point-global % | RMSE K | MAE K | peak MAE K | hotspot RMSE K | top-1% overlap % |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Heat3D + U-v2 | 0 | 2.9216 | 3.2859 | 2.6094 | 1.9656 | 3.7371 | 5.4090 | 60.2325 |
| Heat3D + U-v2 | 1 | 3.2313 | 3.6196 | 2.8744 | 2.1890 | 3.7756 | 6.0636 | 57.7888 |
| Heat3D + U-v2 | 2 | 3.2258 | 3.5304 | 2.8036 | 2.1382 | 3.8956 | 5.6823 | 59.5738 |
| Therm-FM | 0 | 2.4950 | 3.1424 | 2.4955 | 1.6026 | 4.1383 | 4.1010 | 49.2551 |
| Therm-FM | 1 | 2.4253 | 2.6919 | 2.1378 | 1.4590 | 4.1383 | 3.7903 | 52.4154 |
| Therm-FM | 2 | 2.8713 | 3.6125 | 2.8689 | 1.8473 | 3.8566 | 4.0668 | 58.0784 |

### Aggregate (mean ± SD across training seeds)

| model | sample-first % (primary) | point-global % | CV-weighted point-global % | RMSE K | MAE K | Corr_point % | Amp_range % | Corr_CV % | Amp_CVRMS % |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Heat3D + U-v2 | **3.1262 ± 0.1772** | 3.4786 ± 0.1728 | 3.3854 ± 0.1846 | 2.7625 ± 0.1372 | 2.0976 ± 0.1171 | 98.1022 ± 0.2124 | 96.0984 ± 0.3790 | 98.9563 ± 0.0819 | 97.3367 ± 0.2102 |
| Therm-FM | **2.5972 ± 0.2399** | 3.1490 ± 0.4603 | 3.1972 ± 0.4514 | 2.5007 ± 0.3656 | 1.6363 ± 0.1963 | 97.4701 ± 0.1930 | 123.0210 ± 2.6190 | 98.2852 ± 0.1678 | 100.0987 ± 0.0140 |

### Mechanism diagnostics (mean ± SD across training seeds)

| model | signed bias K | scale_log_error mean | field_std_ratio | peak MAE K | peak RMSE K | hotspot RMSE K | top-1% true-hotspot overlap % |
|---|---:|---:|---:|---:|---:|---:|---:|
| Heat3D + U-v2 | -1.9404 ± 0.1244 | -0.02718 ± 0.00216 | 0.9630 ± 0.0094 | 3.8028 ± 0.0827 | 5.2049 ± 0.1105 | 5.7183 ± 0.3287 | 59.1984 ± 1.2643 |
| Therm-FM | -0.1314 ± 0.1289 | 0.00066 ± 0.00021 | 1.0191 ± 0.0070 | 3.9503 ± 0.1629 | 5.5718 ± 0.1800 | 3.9860 ± 0.1704 | 53.2496 ± 4.4704 |

For the corrected post-results case-bootstrap amendment (10,000 paired case
resamples, seed=230023; not preregistered), Heat3D − Therm-FM was:
sample-first +0.5290 [0.1751, 0.8828], point-global +0.3297
[-0.4929, 1.1634], RMSE +0.2618 [-0.4025, 0.9029], MAE +0.4613
[0.1134, 0.7809], peak-MAE -0.1475 [-0.9030, 0.5714], peak-RMSE
-0.3670 [-1.3182, 0.5459], and hotspot RMSE +1.7322 [0.9538, 2.4436].
This is conditional on the frozen three-seed cohort; it is not a claim of
statistical superiority.

## 4. Table C — DeepOHeat-dataset supplemental comparison

此表不与 P1i Table A/B 混排。所有行来自同一个 DeepOHeat-v1 benchmark domain
（571,256 nodes），并保留各自原生监督范式和 selection policy。

| regime | training cases | supervision / checkpoint | full-field sample-first rel RMSE % | role |
|---|---:|---|---:|---|
| DeepOHeat-full-minus-valid128 | 99,872 | native PDE/BC; validation-selected best | 1.146688 ± 0.038323 | large-data regime reference |
| DeepOHeat-768 | 768 | native PDE/BC; validation-selected best | 1.305167 ± 0.268321 | same physical-case count arm |
| Heat3D-768-e600 | 768 | supervised labels; fixed e600 endpoint, U-v2 direct-query | 0.709888 ± 0.007765 | same physical-case count arm |
| DeepOHeat-v2 | — | official code/data not available | not reported | paper/design-loop track |

Table C 的主配对是 `DeepOHeat-768 vs Heat3D-768-e600`（same physical-case
count）；`DeepOHeat-full-minus-valid128` 是 99,872-case large-data regime
reference。该表不是 same-information-budget 或 same-compute comparison；DeepOHeat
使用 PDE/BC physics-informed full mesh，而 Heat3D 使用 supervised temperature
labels 与 sparse conditioning。

## 5. Key findings and claim boundaries

- 在 P1i native1024 Table A 中，Heat3D 的 primary point-global
  2.0577 ± 0.0806% 低于 GINO 17.7315 ± 1.6551% 和 Transolver
  18.5565 ± 0.9266%；P8 hierarchical intervals 对两个 contrast 均不跨 0。
  这支持 frozen valid-only native-task 的 descriptive contrast，不支持 universal
  superiority。
- 在 full-field Table B 中，Therm-FM 的 sample-first、RMSE、MAE 与 hotspot RMSE
  平均值较低；Heat3D 的 Corr_point/Corr_CV 更高，且 U-v2 显示约 -1.94 K
  signed bias、field_std_ratio 0.963。Therm-FM 的 Amp_range 约 123% 而
  Amp_CVRMS 约 100%，说明 range overshoot 与 CV-scale calibration 不是同一现象。
  按冻结的 case-bootstrap，Therm-FM 的 sample-first、MAE 和 hotspot-region RMSE
  较低；point-global/RMSE 差值 CI 跨 0。Heat3D 的标量 peak error 略低、Corr 与
  true-hotspot top-1% overlap 较高，但 peak-error CI 跨 0。这里不指定 overall
  winner，也不作 same-information-budget claim。
- Table C 显示 Heat3D-768-e600 在该 supplemental DeepOHeat domain 上的 U-v2
  full-field 数值低于两种 DeepOHeat cohort；由于 modality、PDE/BC information、
  label budget 和 compute 不同，只能称 data-regime evidence。

Supported claims:

1. 在冻结的 P1i native1024 valid-only cohort 上，Heat3D 与 GINO/Transolver 存在
   明确的描述性性能差异。
2. 在冻结的 P1i full-field valid-only cohort 上，Heat3D U-v2 与 pretrained
   dense-input Therm-FM 可以被同域报告，且机制差异可复核。
3. Heat3D 的 U-v2 full-field 误差包含可见的负偏置/幅度校准问题；这是 diagnostic
   evidence，不是调参授权。

## Appendix — historical metric bridge (not a re-ranking)

该桥接仅解释 V6 historical full-field、V7 P1i-e200 U-v2 与 Therm-FM 证据如何演化，
不回写 Table A/B 的 cohort 身份，也不把不同训练制度变成统一预算。V7 e200 相对
V6 并非所有指标均改善，禁止写成 uniformly dominates V6。机器可读值见
`docs/results/v7_g2_historical_metric_bridge_appendix.json`；V6 三指标公式桥接见
`docs/v7_g2_p23_v6_metric_semantic_bridge.json`。

| cohort | point-global rel RMSE % | sample-first rel RMSE % | peak RMSE K | role |
|---|---:|---:|---:|---|
| Heat3D V6 historical full-field | 3.4426 ± 0.0584 | 3.9583 ± 0.0097 | 3.7746 ± 0.3785 | Tier 1 historical canonical |
| Heat3D V7 P1i-e200 U-v2 | 3.4786 ± 0.1728 | 3.1262 ± 0.1772 | 5.2049 ± 0.1105 | Tier 2 replayable reference |
| Therm-FM Poseidon-T | 3.1490 ± 0.4603 | 2.5972 ± 0.2399 | 5.5718 ± 0.1800 | Tier 2 pretrained transfer |

这里的 `±` 均为 SD across training seeds；该表只用于解释结论演化，不能作为
跨 cohort 重新排名，也不支持 V7 e200 uniformly dominates V6 的表述。

Unsupported claims:

- universal accuracy or efficiency superiority；
- same-information-budget superiority；
- sealed/test generalization superiority；
- 将 Therm-FM pretrained transfer 与 scratch-trained model 视为等价训练条件；
- 将 Table A、Table B、Table C 的数值跨域排序。

## 6. Provenance and closeout

关键 provenance 见：

- docs/v7_g2_final_evidence_freeze.json
- docs/results/v7_g2_final_evidence_aggregates.json
- docs/results/v7_g2_final_p8_hierarchical_bootstrap_native.json
- docs/results/v7_g2_final_bootstrap_amendment.json
- docs/results/v7_g2_final_archive_receipt.json
- docs/results/v7_g2_cross_host_backup_receipt.json
- docs/results/v7_g2_claim_reconciliation_receipt.json
- docs/results/v7_g2_historical_metric_bridge_appendix.json
- docs/results/v7_g2_curated_integration_audit.json
- docs/results/v7_g2_final_plotting_audit.json

当前状态为 G2_VALID_ONLY_EVIDENCE_CLOSED、G2_DEVELOPMENT_COMPLETE、
P24_READY_FOR_REVIEW_NOT_UNLOCKED。figure selection、exact-dense plotting audit
和 visualization-only receipt 见：

- docs/results/v7_g2_publication_figure_selection_receipt.json
- docs/results/v7_g2_publication_test_visualization_receipt.json
- docs/results/v7_g2_publication_figure_audit.json

这些 test 图不改变任何 valid-only 数值证据；sealed 仍未解锁。archive receipt
另行记录 source 与跨主机 backup 状态；Mac 临时镜像的持久化限制另有注明。

本分支与 canonical remote 的关系已冻结为
codex/v7-g2-baselines-finalize closeout branch based on
research/v7-g2-baselines@07e0f959caf1ca493f7a3fee3505e6240d107027。
`codex/g2-curated-integration@230980986...` 已按原样推送到 GitHub，但其 tip 的
direct parent 不是要求的 `research/v7@365576d...`，且 allowlist 中记录的 source
commit 相对当前 finalize ref 已过时；因此 curated integration audit
`BLOCKED_FAIL_CLOSED`，没有修改或 fast-forward `research/v7`，也没有创建 closeout tag。
