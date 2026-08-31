# G2 cross-benchmark protocol

状态：`PREPARED_NOT_LAUNCHED`。本协议只定义后续远程矩阵；本轮没有 formal/long training、SSH、test/sealed 或 Therm-FM 大包下载。

## Experiment blocks

| block | rows | shared cases | primary reporting |
|---|---|---|---|
| P1i common task | Vanilla RIGNO、GINO、Transolver、Heat3D | frozen 768 train / 128 valid_iid；`coords + 11 physical features` | `v7_metric_contract` Level-A；primary valid checkpoint + final-epoch sensitivity |
| DeepOHeat native | original DeepOHeat、Heat3D trained/evaluated on same case | 2d power map、single HTC、multi HTC 分开建表 | 每个 benchmark 自己的 output domain 与 absolute-T/field metrics |
| DeepOHeat-v1 native | original DeepOHeat-v1、Heat3D trained/evaluated on same case | surface、volumetric 分开建表 | official rel-L2/RMSE/max-L1/MAPE/PAPE definitions |

每个 native benchmark 独立成表。不同 mesh、温标、reference solver 或 benchmark 的 RMSE 不得放在同一列排名。runtime 只有在同一硬件、同一 batch/query count、相同精度与完整 I/O 边界下重新测量后才能直接比较；论文原报 speedup 只作背景。

## Split 与 label policy

- 优先复用 upstream fixed test inputs/reference fields；无 fixed train split 的 DeepOHeat HTC family 先确定性采样并冻结 case manifest。
- DeepOHeat-v1 复用 surface 10,000/10 与 volumetric 100,000/100 input split。Heat3D 所需 train truth 必须由同一 reference solver/fidelity 新生成，并给 solver/version/mesh/file SHA；不得用 test truth 训练或调参。
- surface-power rows 在 explicit Neumann-flux contract 完成前保持 `NOT_LAUNCHABLE`。HTC 与 v1 volumetric 先做 lossless data-loader qualification，再申请长训练。
- native original physics-informed rows保持作者 objective/optimizer/sampling；Heat3D rows保持 Heat3D supervised语义。分别报告 labels/solver calls、training walltime、inference walltime，不能声称 same training budget。

## Remote sequence after G1

1. 取得新的远程授权；校验 G1 已结束、目标 GPU 空闲、G2 commit 与 manifests 精确匹配。
2. P1i GINO/Transolver 各做一个 batch 的 GPU memory + epoch-walltime preflight；不得看 valid accuracy 改 radius/config。
3. 经资源 gate 后才运行预注册三 seeds；Vanilla RIGNO/Heat3D 使用 G1 frozen formal artifacts，不根据 interim result改 G2。
4. Native track 先补齐 reference train labels/data manifest，再各做 loader/one-batch gate；正式训练需另行授权。
5. 汇总时先按 benchmark 分表，再讨论 representation、label cost、pretraining/solver budget；不跨 benchmark 排 RMSE。
