# Gate 6M valid-only branch swap and gradient audit

本轮没有训练或 checkpoint 修改。仅访问 `train`（重建 normalization/context）与 `valid_iid`（评估）；`test/hard/sealed` 均未访问。

## Branch swapping

| field | point-global % | sample-first % | raw CV K | shape CV | scale log |
|---|---:|---:|---:|---:|---:|
| V32 | 22.4084 | 21.0348 | 0.160067 | 0.145697 | 0.197482 |
| O075 | 23.4299 | 19.8446 | 0.167186 | 0.150128 | 0.157300 |
| shape_V32+scale_O075 | 23.1424 | 19.4796 | 0.165309 | 0.145697 | 0.157300 |
| shape_O075+scale_V32 | 22.7412 | 21.5050 | 0.162101 | 0.150128 | 0.197482 |

## Shared-backbone gradient cosine

### V32

| loss | shape | scale | relative | raw |
|---|---:|---:|---:|---:|
| shape | 1.0000 | 0.0710 | 0.4160 | 0.0811 |
| scale | 0.0710 | 1.0000 | 0.3345 | 0.2586 |
| relative | 0.4160 | 0.3345 | 1.0000 | -0.1458 |
| raw | 0.0811 | 0.2586 | -0.1458 | 1.0000 |

### O075

| loss | shape | scale | relative | raw |
|---|---:|---:|---:|---:|
| shape | 1.0000 | 0.1268 | 0.3628 | -0.0833 |
| scale | 0.1268 | 1.0000 | 0.0718 | 0.2452 |
| relative | 0.3628 | 0.0718 | 1.0000 | 0.2728 |
| raw | -0.0833 | 0.2452 | 0.2728 | 1.0000 |

## Frozen interpretation

V32=22.4084%，O075=23.4299%；shape_V32+scale_O075=23.1424%，shape_O075+scale_V32=22.7412%。该交换只用于因果诊断，不重新选择 checkpoint。

Q1–Q4 win/loss、逐样本 point-SSE 差值和 inference-only 物理条件归因见 JSON/CSV。本结果不触发自动晋级或训练。
