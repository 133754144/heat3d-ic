# V7 G2 P1i adapter contracts

Authoritative dataset：`heat3d_v6_p1i_continuous_physics1024_v1`，manifest SHA-256 `f19987c659968c2ac14eade1f1ef7e206c8f7eeb94f58fde5897d6e765978514`。本合同只开放 train / `valid_iid`；`test_iid` 与 sealed fail closed。adapter 输入路径不得读取 `temperature.npy` 或 target，valid truth 只能由 `rigno.heat3d_runtime.evaluation.EvaluationCore` 独立加载。

共同 point-view 为 `coords[B,1024,3] + features[B,1024,11]`；feature 顺序固定为 `kx, ky, kz, q, is_top, is_bottom, is_side, is_interior, top_h, bottom_h, top_T_inf_minus_T_ref`。输出必须转换为 `prediction_deltaT_K[B,1024,1]` 后再进入 metric contract。

| 模型 | representation adapter | 信息合同 | 当前 gate |
| --- | --- | --- | --- |
| GINO | irregular input points → explicit `8³` latent grid → same 1024 output queries | 不引入额外物理观测；latent grid 属于内部计算 representation；upstream shared geometry 时 batch=1 | input-only 与 valid-only evaluator smoke 已通过 |
| Transolver | 每个 P1i point 作为 token，coords 与 11 features 拼接后进 Physics-Attention | same points、same features；无 dense field、solver、pretrained prior | input-only 与 valid-only evaluator smoke 已通过 |
| Geo-FNO | 待实现 point features + learned deformation + latent FNO | 禁止复用 Elasticity 的 42-D `rr` shape code；只能从 P1i coords/features 派生 | original reproduction 已过；P1i gate 未过 |
| Therm-FM | 需要显式 point-to-grid/layer rasterization 和 grid-to-point query | 必须列出插值、缺失 voxel、额外 full-field 数据与 pretrained Poseidon prior；默认不属于 common information budget | separate transfer track |
| DeepOHeat | released 441-point surface-power branch 不能无损表示 P1i heterogeneous k/q/BC | 增加 branch、physics loss 或 full configuration encoder 会实质改变 release，不再是 original adapter | scientific incompatibility，停止工程修补 |
| DeepOHeat-v2 | 需要 known FVM operator、placement vector、trust gate、GMRES/AMG solver 与 online labels | 与固定 P1i prediction task 不同，不建立 same-table adapter | official code/data 前 deferred |

所有 adapter 必须生成 information ledger，逐项说明：原始字段、变换、插值、丢失字段、由输入可确定的派生量、外部 prior、solver/data access。只有 `external prior=false` 且没有超出同一 1024-point physical observation 的模型，才可标为 `common_information_budget=true`。
