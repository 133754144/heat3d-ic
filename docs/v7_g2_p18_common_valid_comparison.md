# V7 G2-P18：common-valid comparison

仅使用 Heat3D valid128；本 P18 comparison 未访问 test_iid、sealed 或 DeepOHeat official100。治理上，V6 的 test_iid 已于历史路线中打开过一次作 legacy confirmatory holdout；当前 untouched final holdout 是 sealed IID。
指标为 full 571256-point temperature-space sample-first relative RMSE [%]。

## Full-resolution policy

正式跨分辨率 full-field 默认采用 **U-v2 direct-query dense inference**：冻结的
1024 conditioning points 直接查询 `101×101×56=571,256` 个 DeepOHeat-v1 官方
坐标，端到端计时包含 query-graph construction、direct-query forward 和必要
postprocess。native-1024 是独立输出视图；IDW 仅为 V6/P1h historical diagnostic，
不进入正式比较。冻结配置见
[`g2_full_resolution_u_v2_policy.json`](../configs/heat3d_v7/g2_full_resolution_u_v2_policy.json)。
表中 Heat3D 的 best 列是固定 e600 endpoint 的占位显示，并非 validation-selected best；DeepOHeat 的 best/final 列分别对应 validation-selected best 与固定 100000-iteration endpoint。

| regime | training cases | metric | best mean ± SD | final mean ± SD | wall h | peak GiB |
|---|---:|---|---:|---:|---:|---:|
| DeepOHeat-full-minus-valid128 | 99872 | sample_first_relative_rmse_pct (full 571256) | 1.146688 ± 0.038323 | 1.507919 ± 0.218354 | 0.580 | 0.640 |
| DeepOHeat-768 | 768 | sample_first_relative_rmse_pct (full 571256) | 1.305167 ± 0.268321 | 1.436807 ± 0.309476 | 0.660 | 0.700 |
| Heat3D-768-e600 | 768 | sample_first_relative_rmse_pct (U-v2 direct-query dense full 571256) | 0.709888 ± 0.007765 | 0.709888 ± 0.007765 | 0.742 | 3.868 |

## 公平性边界

DeepOHeat-768 与 Heat3D-768 共享 768/128 physical-case split，但不共享信息预算：前者使用 PDE/BC physics-informed full mesh，后者使用 supervised temperature labels 与 1024 sparse support。
DeepOHeat-full-minus-valid128 的 128 个验证 case 在训练池中被严格排除，可用于共同 valid 比较；历史 native-full pool 与 valid128 重叠，不能作 valid 排名。

## Historical diagnostics (excluded from formal comparison)

P14.1 e200 U-v2 direct-query dense mean = 0.814137%；IDW 与 oracle 仅作历史 lower-bound diagnostic，不进入正式表，也不做误差相减分解。

## Efficiency and latency boundary

| regime | parameters | training budget | model-only latency | end-to-end dense latency |
|---|---:|---|---:|---:|
| DeepOHeat-full-minus-valid128 | 3,303,680 | 100000 iterations; 50 functions/iteration | not measured | not measured |
| DeepOHeat-768 | 3,303,680 | 100000 iterations; 50 functions/iteration | not measured | not measured |
| Heat3D-768-e600 | 892,776 | 600 epochs; B24; 768 supervised cases | not measured | 1008.137 ± 11.025 s |

Latency must be remeasured on the same hardware before Pareto claims. Heat3D U-v2 direct-query dense end-to-end timing includes full-query graph construction and direct-query forward; DeepOHeat P14/P17 receipts do not contain a directly comparable dense latency measurement.
