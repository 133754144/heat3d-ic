# V7 G2 P20-D：accuracy/generalization benchmark matrix

状态：`PROPOSED_ACCURACY_FIRST_MATRIX_VALID_ONLY`。P20 没有启动训练或新评估；所有数字来自已关闭的 valid-only receipts。训练时间、显存和 latency 只作 supplementary characterization。

## P1i common-task（native 1024）

| model | valid sample-first relative RMSE [%] mean ± sample SD | point-global [%] mean ± sample SD | status |
|---|---:|---:|---|
| Heat3D V6 canonical | 1.629402 ± 0.013132 | 2.027348 ± 0.094738 | READY_VALID_ONLY |
| GINO | 15.013874 ± 0.913015 | 17.731492 ± 1.655147 | READY_VALID_ONLY_E3_PASS_PRIOR |
| Transolver | 16.002822 ± 1.031020 | 18.556496 ± 0.926554 | READY_VALID_ONLY_COMPLETE |

这张表仅在 native 1024 query domain 和温度单位一致时使用；它不是 same-information-budget 表。GINO 既有 seed0→seed1→seed2 结果和 Heat3D V6 结果均只用于 valid-only 证据，不能推导 test/general-domain 结论。:codex-annotation{index="1"}

## DeepOHeat-v1 native volumetric domain（571,256 points）

| regime | train physical cases | best sample-first [%] | final sample-first [%] | role |
|---|---:|---:|---:|---|
| DeepOHeat full-minus-valid128 | 99,872 | 1.146688 ± 0.038323 | 1.507919 ± 0.218354 | held-out native recipe |
| DeepOHeat matched-768 | 768 | 1.305167 ± 0.268321 | 1.436807 ± 0.309476 | same physical-case budget |
| Heat3D-on-v1 e600 U-v2 | 768 | 0.709888 ± 0.007765 | fixed e600 endpoint | method-native cross-benchmark |

`DeepOHeat-v1-native-full` 仍是 `NATIVE_REFERENCE_NOT_RANKABLE`：其 100,000-function training pool 包含候选 valid128，不能在该 valid 上形成 held-out ranking。DeepOHeat 与 Heat3D 的 PDE/BC physics supervision、dense/full-mesh information 和 supervised sparse-label information 不同，不能写成 same information budget。

## Transfer / related work

- Therm-FM：`NEEDS_AMENDMENT / BLOCKED_BY_UPSTREAM_ASSETS`。P1i dense-label manifest 完整，但 selective `model_T`、config/stats 和 deterministic point-to-grid contract 尚未可校验；保持 pretrained transfer track。
- HCP-enhanced DeepONet：`NEEDS_AMENDMENT / BLOCKED_BY_UPSTREAM_ASSETS / REFERENCE_ONLY`。官方仓库没有提交 benchmark data/checkpoint/raw outputs，且原实现要求 inline structured uniform mesh；未证明可原生接收 P1i variable geometry/material/source/BC。

因此不建立跨 capability envelope 的单一 efficiency/accuracy 总排名。所有 runtime 结果需在同硬件、同计时边界重新测量后才可进入 supplementary Pareto 描述。
