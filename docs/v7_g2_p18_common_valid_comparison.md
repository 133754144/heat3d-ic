# V7 G2-P18：common-valid comparison

仅使用 Heat3D valid128；test_iid、sealed 与 DeepOHeat official100 均保持锁定。
指标为 full 571256-point temperature-space sample-first relative RMSE [%]。

| regime | training cases | metric | best mean ± SD | final mean ± SD | wall h | peak GiB |
|---|---:|---|---:|---:|---:|---:|
| DeepOHeat-full-minus-valid128 | 99872 | sample_first_relative_rmse_pct (full 571256) | 1.146688 ± 0.038323 | 1.507919 ± 0.218354 | 0.580 | 0.640 |
| DeepOHeat-768 | 768 | sample_first_relative_rmse_pct (full 571256) | 1.305167 ± 0.268321 | 1.436807 ± 0.309476 | 0.660 | 0.700 |
| Heat3D-768-e600 | 768 | sample_first_relative_rmse_pct (U-v2 full 571256) | 0.709888 ± 0.007765 | 0.709888 ± 0.007765 | 0.742 | 3.868 |

## 公平性边界

DeepOHeat-768 与 Heat3D-768 共享 768/128 physical-case split，但不共享信息预算：前者使用 PDE/BC physics-informed full mesh，后者使用 supervised temperature labels 与 1024 sparse support。
DeepOHeat-full-minus-valid128 的 128 个验证 case 在训练池中被严格排除，可用于共同 valid 比较；历史 native-full pool 与 valid128 重叠，不能作 valid 排名。

## P14.1 reconstruction diagnostic

e200 U-v2 dense mean = 0.814137%；IDW 与 U-v2 oracle 仅作 reconstruction floor/diagnostic，不做误差相减分解。

## Efficiency and latency boundary

| regime | parameters | training budget | model-only latency | end-to-end dense latency |
|---|---:|---|---:|---:|
| DeepOHeat-full-minus-valid128 | 3,303,680 | 100000 iterations; 50 functions/iteration | not measured | not measured |
| DeepOHeat-768 | 3,303,680 | 100000 iterations; 50 functions/iteration | not measured | not measured |
| Heat3D-768-e600 | 892,776 | 600 epochs; B24; 768 supervised cases | not measured | 1008.137 ± 11.025 s |

Latency must be remeasured on the same hardware before Pareto claims. Heat3D U-v2 end-to-end timing includes full-query graph construction and direct-query forward; DeepOHeat P14/P17 receipts do not contain a directly comparable dense latency measurement.

