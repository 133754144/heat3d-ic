# V7 G2 P1i adapter contracts

Authoritative dataset：`heat3d_v6_p1i_continuous_physics1024_v1`，manifest SHA-256 `f19987c659968c2ac14eade1f1ef7e206c8f7eeb94f58fde5897d6e765978514`。本合同只开放 train / `valid_iid`；`test_iid` 与 sealed fail closed。adapter 输入路径不得读取 `temperature.npy` 或 target，valid truth 只能由 `rigno.heat3d_runtime.evaluation.EvaluationCore` 独立加载。

共同 point-view 为 `coords[B,1024,3] + features[B,1024,11]`；feature 顺序固定为 `kx, ky, kz, q, is_top, is_bottom, is_side, is_interior, top_h, bottom_h, top_T_inf_minus_T_ref`。输出必须转换为 `prediction_deltaT_K[B,1024,1]` 后再进入 metric contract。

| 模型 | representation adapter | 信息合同 | 当前 gate |
| --- | --- | --- | --- |
| GINO | irregular input points → official CarCFD-faithful `32³` latent grid → same 1024 output queries | 不引入额外物理观测；latent grid 属于内部计算 representation；upstream shared geometry 时 batch=1；只调整既有 channels | 1 epoch train/valid/checkpoint/reload 已通过 |
| Transolver | 每个 P1i point 作为 token，coords 与 11 features 经官方 preprocess MLP 后进 Physics-Attention | same points、same features；无 dense field、solver、pretrained prior；只调整 `space_dim/fun_dim` | 1 epoch train/valid/checkpoint/reload 已通过 |
| Geo-FNO | released 2D learned deformation 无法只靠 channel adjustment 表示 3D P1i functions | 禁止复用 Elasticity 的 42-D `rr` shape code；禁止自行新增 learned 3D adapter | `NOT_RUN_scientific_incompatibility_without_algorithm_change` |
| Therm-FM | 需要显式 point-to-grid/layer rasterization 和 grid-to-point query | 必须列出插值、缺失 voxel、额外 full-field 数据与 pretrained Poseidon prior；默认不属于 common information budget | separate transfer track |
| DeepOHeat | released 441-point surface-power / 1-HTC / 2-HTC branches 不能无损表示 P1i 同时变化的 heterogeneous k/q/BC/geometry | 增加 branch、physics loss 或 full configuration encoder 会实质改变 release，不再是 original adapter | `P1i_direct_comparison_not_identifiable_without_algorithm_change` |
| DeepOHeat-v2 | 需要 known FVM operator、placement vector、trust gate、GMRES/AMG solver 与 online labels | 与固定 P1i prediction task 不同，不建立 same-table adapter | official code/data 前 deferred |

所有 adapter 必须生成 information ledger，逐项说明：原始字段、变换、插值、丢失字段、由输入可确定的派生量、外部 prior、solver/data access。只有 `external prior=false` 且没有超出同一 1024-point physical observation 的模型，才可标为 `common_information_budget=true`。

本合同的 upstream-faithful optimizer/loss/schedule 与本机一轮 gate 已由 `configs/heat3d_v7/g2_p1_protocol_v2.json` 取代旧的统一训练预算 proposal。不同 baseline 不再强制统一 epochs、LR 或 loss。
