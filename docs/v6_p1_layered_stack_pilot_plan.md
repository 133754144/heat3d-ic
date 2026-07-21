# V6-P1 Layered-Stack Pilot 规格纠偏

## 状态与边界

- 数据集族：`v6_layer_v0`。
- 当前状态：`specification_only`；pilot、全量生成、模型训练和模型推理均未启动。
- 本阶段只允许层状堆叠，不允许任意三维材料块。每个 scene 只能有 1–2 个显式 active layers；每个 active layer 可有 1–4 个互不重叠热源，所有 interface/passive layers 必须 `q=0`。
- 主体标签定义为 `peak_deltaT = max(T) - 298.15 K`，冻结范围 30–80 K。

机器可读合同以 `configs/heat3d_v6/v6_dataset_spec_draft.yaml` 为准；本文解释执行次序及禁止事项。

## 1. 主体温升分布

96 个 proposed pilot 样本分成四个等宽、等数量档：

| bin | 范围 K | train | valid_iid | test_iid_sealed | 总数 |
| --- | --- | ---: | ---: | ---: | ---: |
| T30_42p5 | [30, 42.5) | 16 | 4 | 4 | 24 |
| T42p5_55 | [42.5, 55) | 16 | 4 | 4 | 24 |
| T55_67p5 | [55, 67.5) | 16 | 4 | 4 | 24 |
| T67p5_80 | [67.5, 80] | 16 | 4 | 4 | 24 |

每档内部目标温升由预注册 seed 做确定性均匀采样。不得观察生成后的总体温度直方图再选择 seed。最终高保真解必须落在 30–80 K 且与目标相差不超过 0.25 K；这是预注册物理 QC，不是依据模型效果筛样。

## 2. 由单位功率反算总功率

不再预先从固定宽功率区间采样。对已经冻结的 geometry/material/source-template/contact/BC scene：

1. 将 source template 按 control volume 精确归一化，使积分总功率为 `1 W`。
2. 用最终高保真、layer-aligned solver mesh 求解一次，得到
   `Rth_peak = peak_deltaT_1W / 1 W`。
3. 从预注册温升档取得 `target_peak_deltaT`，反算
   `P = target_peak_deltaT / Rth_peak`。
4. 按各 active layer 的 source allocation 恢复 q，并再次按 control volume 校准总功率。
5. 执行最终求解；验证 `P * Rth_peak` 与实际 peak DeltaT 的线性闭合误差不超过 `1e-8`，实际温升与目标绝对差不超过 `0.25 K`。

该反算只适用于冻结的线性稳态合同：常数 k、线性 Robin、与温度无关的 contact resistance。若以后引入温度相关 k、辐射或非线性边界，必须升级 schema，不能继续复用该公式。

文献功率只作 admissibility gate，而不是新的采样分布：package total power 必须 `0 < P <= 20 W`，任一 active layer 不得超过 `10 W`（矩阵 L19）。失败时记录原因，使用同 split、同温升档的下一条预注册 reserve scene；不得改变目标档或查看 test 排名。

## 3. 冻结主冷却与接触界面

主体 100% 使用单一冷却工况：

- ambient `298.15 K`；
- top Robin `h=500 W/(m² K)`；
- bottom Robin `h=500 W/(m² K)`；
- sides adiabatic。

该形式直接对应文献矩阵 L01/L02。bottom Dirichlet 300 K 不得进入主体数据，只能在主体合同冻结后的 verification/OOD 使用，并且不能参与功率反算、split、筛样或调参。

Contact resistance 必须绑定一个真实、具名的材料界面：metadata 至少包含 `interface_id/lower_material_id/upper_material_id/R_contact/evidence_ids`。只有相邻材料不同的 layer boundary face 可以获得 contact resistance；禁止在任意几何中面插入，也禁止在 same-material face 上施加。`0/5e-6/1e-5 m²K/W` 仍是 generation 前需复核来源的 sensitivity grid，不是已批准生产分布。

## 4. 高保真 mesh 到 1024 irregular points

标签首先在 layer-aligned 高保真 control-volume mesh 上求解，最低网格 `64×64×32`，每个物理层至少 4 个 cells；12.5% scene 执行预注册 mesh-convergence。不得直接在 1024 点图上求解并称为高保真标签。

每个 scene 在任何温度求解前冻结 1024 个唯一 irregular points：

| stratum | 点数 | 可使用的信息 |
| --- | ---: | --- |
| volume-stratified jittered | 512 | geometry/control-volume |
| active-source aware | 256 | source mask 与单位功率 q template |
| material-interface aware | 128 | layer/material interface |
| top boundary | 64 | top BC mask |
| bottom boundary | 64 | bottom BC mask |

点 seed 只能由 generator seed、无标签 scene fingerprint 和 point-schema version 得到。严禁使用 T、DeltaT、温度梯度、热点位置、插值误差或 solver residual 移动/补选点。

温度使用 trilinear 或 solver-native shape function，从高保真 mesh 插值到冻结点。12.5% QC scene 在更细独立 mesh 的相同冻结点上复核：

- point RMSE / peak DeltaT `<=0.5%`；
- point absolute error P95 `<=0.5 K`；
- peak DeltaT absolute error `<=1.0 K`。

失败可以按物理 QC 拒绝 scene，但禁止重新采点、改变 point seed，或依据插值误差调整 split。保存 high-fidelity mesh provenance、point coordinates、point seed、插值方法、reference mesh 和三项误差。

## 5. Split 与 QC

Scene fingerprint 和目标温升档在求解前决定 split；同一物理 scene、reserve family 或 point set 不得跨 split。允许的 label-aware 筛选仅限预注册物理标准：30–80 K、目标闭合、finite、能量守恒、mesh convergence 与冻结点插值误差。

明确禁止：

- 根据任何模型 train/valid/test 误差筛样；
- 根据 test 样本排名、test aggregate 或 sealed 标签重选 scene/seed；
- 观察生成后温度分布再改变四档边界或配额；
- 将 solver failure 隐式替换到别的温升档或 split；
- 把 q power calibration/rescale 记为 clipping。

q clipping 只有在 metadata 明确给出正的 clipped-node count、正的 clipping fraction 或 true event boolean 时才算发生。仅声明上下限、零事件计数或 `q_rescale_factor != 1` 不构成 clipping。主体合同发现任何实际 clipping 都拒绝样本，不进行截断修复。

## 6. 启动门

后续只有在显式用户指令后才能启动 pilot，并需先证明：YAML schema/checker 通过、四档及 split 配额冻结、unit-power solver fixture 通过、material-interface contact fixture 通过、1024 点无标签泄漏 fixture 通过、输出路径为空且 `generation_started=false`。本轮没有执行上述生成阶段。
