# V6-P1a 16-sample power calibration pilot

## 结论

本轮严格生成 16 个预注册层状样本，没有训练、模型推理、扩样、按温升筛选、重采或功率反算。16 个文献功率工况中，**没有样本自然落入 30--80 K**：L02 的 mW 级工况全部低于窗口（peak DeltaT 0.295--3.492 K），L19 的 1--20 W 工况全部高于窗口（155.616--15985.432 K）。这说明当前离散功率表在固定 P1a 几何/材料/边界下跨过了目标窗口，而不是求解器把窗口内样本筛掉。

低温样本的直接主因是 mW 级单源/组件功率。固定热源几何时，0.5/1/2 mW 的 peak DeltaT 分别为 0.295/0.591/1.182 K，呈严格线性；同为 1 mW 时，将单源拆成两个分离热源会把 peak DeltaT 从 0.591 K 降到 0.308 K，说明热源体积/空间分散是显著的次级因素。bottom Dirichlet 吸收 95.91%--98.74% 的总热量，是本合同的主导散热路径，但它对全部功率档固定，因此不能单独解释为何只有 L02 档为低温；它决定的是本 pilot 的整体热阻尺度。

## 冻结合同与功率溯源

- 功率 registry：`configs/heat3d_v6/v6_p1a_power_calibration_cases.yaml`。
- 文献矩阵 SHA256：`ef2bc1ee1fab5d63cb92e835a348361e10a773256c12429d609a4eeb23b6f852`。
- L02：DeepOHeat-v1，DOI `10.48550/arXiv.2504.03955`。直接采用文献列出的 0.5/1/2 mW component powers 及 10-component、13 mW schedule。
- L19：Quasi-3D Thermal Simulation of Integrated Circuit Systems in Packages，DOI `10.3390/en13123054`。直接采用文献列出的 1 W validation、10 W/active die 和 20 W/package 档位。多源工况只对这些已引用的 active-layer/package power 作显式等分；等分值不声称为文献 component power。
- 每个 source power、每层 source power 之和和 package total power 分列保存；`power_was_Rth_inferred=false`。没有从目标 DeltaT、Rth 或求解结果反算功率。
- 30--80 K 仅为生成后的评价窗口，`peak_deltaT_filtering=false`、`peak_deltaT_resampling=false`。

P1a 固定 BC 为 top Robin `h=500 W/(m2 K), T_inf=300 K`、bottom Dirichlet `300 K`、sides adiabatic、perfect contact。功率来源可追溯并不表示本轮复制了原论文的全部 BC；尤其 bottom Dirichlet 是本 pilot 的诊断边界。

## 网格、热源解析与 1024 点投影

原生有限体积网格为 64 x 64 x 32 intervals，即 65 x 65 x 33 = 139425 nodes；z 网格与 5 个层边界严格对齐。热源仅位于 `active_lower`/`active_upper`，每个 active layer 有 4 个厚度方向 intervals，每个热源至少覆盖 484 个控制体（合同下限 256）。57 个热源的体积为 1.47705e-12--1.53859e-12 m3，q 为 3.24972e8--6.77025e12 W/m3。

每样本的 1024 个 irregular points 在温度求解前由固定 seed 生成，分层为 volume/source/interface/top/bottom = 512/256/128/64/64。点选择不读取温度或其他标签；高保真网格场用线性插值投影。solver peak 与投影点 peak 的差值已逐样本保存，仅作插值覆盖诊断，不参与样本接受或替换。

## 16 样本结果

| sample | literature | sources/layers | package power (W) | peak DeltaT (K) | mean DeltaT (K) | Rth_peak (K/W) | bottom/top heat fraction | 30--80 K |
|---|---|---:|---:|---:|---:|---:|---:|---|
| p1a_00 | L02 | 1/1 | 0.0005 | 0.295 | 0.011 | 590.768 | 0.9874/0.0126 | no |
| p1a_01 | L02 | 1/1 | 0.001 | 0.591 | 0.023 | 590.768 | 0.9874/0.0126 | no |
| p1a_02 | L02 | 1/1 | 0.002 | 1.182 | 0.046 | 590.768 | 0.9874/0.0126 | no |
| p1a_03 | L02 | 2/1 | 0.001 | 0.308 | 0.023 | 307.854 | 0.9874/0.0126 | no |
| p1a_04 | L02 | 4/1 | 0.004 | 0.622 | 0.092 | 155.616 | 0.9874/0.0126 | no |
| p1a_05 | L02 | 4/1 | 0.008 | 1.245 | 0.184 | 155.616 | 0.9874/0.0126 | no |
| p1a_06 | L02 | 10/1 | 0.013 | 1.271 | 0.298 | 97.774 | 0.9874/0.0126 | no |
| p1a_07 | L02 | 10/2 | 0.013 | 3.492 | 0.856 | 268.578 | 0.9591/0.0409 | no |
| p1a_08 | L19 | 1/1 | 1 | 590.768 | 22.942 | 590.768 | 0.9874/0.0126 | no |
| p1a_09 | L19 | 4/1 | 1 | 155.616 | 22.942 | 155.616 | 0.9874/0.0126 | no |
| p1a_10 | L19 | 2/2 | 1 | 799.272 | 53.905 | 799.272 | 0.9670/0.0330 | no |
| p1a_11 | L19 | 1/1 | 10 | 5907.677 | 229.419 | 590.768 | 0.9874/0.0126 | no |
| p1a_12 | L19 | 4/1 | 10 | 1556.160 | 229.419 | 155.616 | 0.9874/0.0126 | no |
| p1a_13 | L19 | 2/2 | 10 | 7992.716 | 539.047 | 799.272 | 0.9670/0.0330 | no |
| p1a_14 | L19 | 2/2 | 20 | 15985.432 | 1078.094 | 799.272 | 0.9670/0.0330 | no |
| p1a_15 | L19 | 8/2 | 20 | 4746.685 | 1078.094 | 237.334 | 0.9670/0.0330 | no |

每源的体积、source power、q、覆盖控制体数和 active-layer assignment 见 `configs/heat3d_v6/v6_p1a_power_calibration_sources.csv`；更高精度的逐样本数值见 samples CSV/JSON audit。

## 受控归因

1. **单源功率**：p1a_00/01/02 的几何、source volume 和 BC 相同，功率加倍时 peak/mean DeltaT 加倍且 Rth_peak 保持 590.768 K/W。这是 L02 低温的直接证据。
2. **热源体积与空间分散**：固定 package power=1 mW，p1a_01 单源与 p1a_03 两源的 mean DeltaT 相同，但两源 peak 低 47.9%；固定 1 W，四源 p1a_09 比单源 p1a_08 的 peak 低 73.7%。因此分散主要降低 hotspot/peak，不改变同层线性系统的体积平均温升。
3. **active-layer 位置**：相同 13 mW 下，双 active-layer p1a_07 的 peak 是单层 p1a_06 的 2.75 倍，且 top heat fraction 从 1.26% 增到 4.09%。源层位置不能由“总功率”替代。
4. **bottom 定温热汇**：全部样本的 bottom heat fraction 为 95.91%--98.74%，top Robin 仅承担 1.26%--4.09%。因此 bottom Dirichlet 主导热流去向和绝对 Rth；但在这一固定 BC 内，低/高温分组首先由 0.013 W 与 1 W 之间的离散功率断层决定。

这 16 点只能识别上述合同内的因果对照，不能把 fixed-bottom 结果外推为 V6 主数据集的冷却结论，也不能据此自行插值或反算新的功率档。下一步若获授权，应先预注册额外文献支持的中间功率工况或单独的 BC counterfactual；本轮没有执行这两项。

## 完整性与复现

- Dataset manifest SHA256：`dfd81eca023cca5290e49567559005b53766aa9cfe6e0550c919b29fd79b4636`。
- 最大绝对能量守恒相对误差：`5.87851e-10`；所有字段 finite。
- 数据根目录：`data/heat3d_v6_p1a_power_calibration16_v0`（129 files，约 2.2 MiB）。仓库提交冻结 registry、manifest 镜像、逐样本/逐热源表、生成器、checker 和本报告；`data/` 仍按仓库策略忽略，不强制纳入 Git。
- 生成：`python3 scripts/generate_heat3d_v6_p1a_power_calibration_pilot.py`
- 校验：`python3 scripts/check_heat3d_v6_p1a_power_calibration_pilot.py`

生成器拒绝覆盖已有数据目录；复现时必须明确使用新的空输出目录，不能静默覆盖本轮样本。
