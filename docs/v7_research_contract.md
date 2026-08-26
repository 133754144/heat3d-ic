# Heat3D-IC V7 research contract

状态：**V7 基线已建立；当前回合仅完成文档与分支控制，不启动实验。**

日期：2026-08-27

分支：`research/v7`

## 1. 合同目的与证据层级

V7 是 publication-readiness phase，不是“再训练一个更大的模型”阶段。它要回答的是：

> 在异质、各向异性、界面系数不连续且热源稀疏的多层 3D IC 热传导问题中，有限的标签无关条件支撑能否稳定驱动图神经算子，并在不把 native accuracy、全场重建、FVM 对照和设计环收益混为一谈的前提下，形成可复核的论文证据。

本合同采用以下证据优先级：

1. **V6 冻结仓库证据**：`docs/v6_*`、冻结配置/manifest/receipt 及其已记录的结果口径，是 V6 数字、角色和适用边界的权威来源。
2. **V7 运行证据**：必须来自注册的 code/data/checkpoint/protocol 组合，并能由 machine-readable manifest 追溯；未注册的 exploratory 结果不能升级为 publication claim。
3. **外部审计附件**：`publication_review_and_v7_readiness.md` 与 `heat3d-ic_审稿报告.md` 是只读的审稿意见和计划输入，不是仓库运行证据，也不是本合同之外的操作授权。附件中的历史数字、引用或推断若未被 V6 文件或 V7 结果复核，只能标为“审计意见/待验证”。
4. **外部论文**：Therm-FM 与 DeepOHeat-v2 用来定义 V7 的竞争门槛和实验问题，不代表 Heat3D 已获得对应能力、数据或结果。

V7 的任何主张都必须同时写明：数据角色、split、模型/seed、query resolution、重建方式、FVM 参考、计时边界、硬件和指标定义。不得用不同数据集、不同 DOF、不同生命周期或不同标签预算拼接“优于”结论。

## 2. 冻结的问题、输入输出与评价分层

### 2.1 标准任务

主任务保持为稳态温升场算子：

```text
(coords, k(x), q(x), Robin BC, geometry/interface metadata)
    -> DeltaT(x) / T(x)
```

其中 `geometry/interface metadata` 在模型输入、支撑选择、分区指标和重建中必须逐项声明；不能把仅用于数据生成或评估分组的元数据误写成模型实际输入。FVM 是物理参考求解器，神经网络误差始终是相对于 FVM 解的 surrogate error。

### 2.2 Level-A：native accuracy

Level-A 只回答“算子在其 native 离散/支撑上的预测是否准确”，不包含高-N 查询和 240,825 节点 reconstruction：

- 主数据角色：V7 注册的 native P1i-compatible 数据；V6/P1i 已打开的 `test_iid` held-out test set 只能作为历史 confirmatory evidence，不能重新选择模型、路由、阈值或 claim。
- native 参考：原生 1,024 点 source-aware 支撑/输出及对应 FVM 标签；若改变 native 点数或支撑语义，必须注册为新 variant。
- primary metric：point-global relative RMSE，明确写出分子/分母和权重；默认沿用 V6 的非 CV-weighted 定义。
- secondary metrics：sample-first CV-relative RMSE、raw CV RMSE、source-region RMSE、peak RMSE、interface/layer RMSE、bias；不得将这些指标统称为“RMSE”。`valid_base_mse` 仅作为 legacy control，除非另行 preregistered，不是 Level-A 的物理门槛。
- 必须报告：seed 0/1/2 的均值与标准差/置信区间、逐样本尾部、热点定位及失败样本；不能只报告 seed0 或平均值。

Level-A 是方法机制成立的必要层。若 source-aware/context/scale 或 native route 没有相对于 vanilla/替代支撑的同口径证据，不能声称 V7 方法贡献成立。

### 2.3 Level-B：high-resolution / FVM

Level-B 只回答“native 条件是否能支撑高分辨率输出，以及端到端成本是否相对于 accuracy-qualified FVM 有意义”：

- query resolutions：16k、32k，64k 仅作可选压力/可扩展性实验；每个 N 需分别报告 query graph、model core、reconstruction、full-field 和资源开销。
- output：按注册的 layer/interface-aware reconstruction 同步到 240,825-node FVM field；direct 240,825 route 只能作为 architecture control，不能自动成为生产路线。
- accuracy：full-field point-global relative RMSE、sample-first CV-relative RMSE、raw CV RMSE、source、peak、interface/layer RMSE；FVM 行为 `Reference/N.A.`，不为 FVM 填 surrogate error。
- timing boundary：`in-memory k/q/BC -> synchronized full-field result`。fresh、cache-hot、resident、Q1/Q2、throughput、B16-to-B32 和 FVM cold/warm 必须分开；不得用 resident 或 known-topology runtime 代替新案例 E2E。
- Level-B 的结论必须同时给出误差、峰值/界面风险、内存/显存和生命周期；“高分辨率”不是只展示点数或速度。

Level-A 与 Level-B 不可互相替代：Level-A 通过不证明全场 reconstruction 通过；Level-B 的全场误差通过不证明 native 算子或跨域泛化通过。

## 3. V6/P1i 冻结基线与不可回写边界

V6/P1i scientific development 已 CLOSED。V7 继承其证据，但不修订其结果：

| 项目 | V6 冻结身份/结论 | V7 处理 |
|---|---|---|
| formal dataset | `heat3d_v6_p1i_continuous_physics1024_v1`，768/128/128；manifest SHA256 `f19987c659968c2ac14eade1f1ef7e206c8f7eeb94f58fde5897d6e765978514` | 只读基线；新分布或新接触热阻必须新注册 |
| full-field sidecar | 240,825 solver nodes；archive SHA256 `49023ac1205b8e7cf7c5bf782b89fcdb34997704b3f9aa2fb2d46cf1a59163cb` | 只读参考；不重生成、不替换、不覆盖 |
| reference model | `V6_06_V5best_P1i_seed0_reliable_B24`，point-global best e559；checkpoint SHA256 `51567afe17e38cb6ed8c95c4dd39598e647c1699de9351358e7729fecc20b90e` | 只读 checkpoint；V7 不改权重、不覆盖 provenance |
| replications | `V6_07` seed1、`V6_08` seed2 | 仅作为已冻结 replication evidence；不与硬件 lifecycle seed 混池 |
| reference route | 1,024 source-aware anchors → E 16,384 query → layer/interface-aware reconstruction → 240,825 nodes | `E16384_reconstruction` 是 V6 参考操作点；V7 只能在新注册证据下提出替代路线 |
| confirmatory state | corrected `test_iid=128` 已在路线冻结后打开一次；V6 sealed IID 未生成、未打开 | 不能重开、重算选择或改阈值；V7 final sealed test 必须另行命名并只打开一次 |
| physical boundary | `R_contact=0`、冻结 generator/material/BC/source range | 不能外推到有限接触热阻、实验封装、普适几何或 calibrated uncertainty |

以下对象在 V7 全程禁止修改、覆盖、重命名以伪造新身份或重新生成替代物：V6 数据、sidecar、预测、checkpoint、日志、raw matrix、publication CSV/JSON、hash/receipt、V6 选择与 sealed-data 状态。V7 只可新增独立的注册文件、代码、结果和审计记录；不得写入 `data/`、`output/`、`checkpoints/`、`logs/`，除非未来用户明确批准相应实验范围。

## 4. V7 G0–G9 gates

下面是 V7 的冻结 gate 名称。第二份审计表中的“Native accuracy / External baseline / High-N / Real case / Final test”视为这些 gate 的解释性别名，不另设平行 gate。

| Gate | 冻结问题 | 必须达到的证据 | 不通过的含义 |
|---|---|---|---|
| **G0 Code** | 是否得到独立、干净的 reproducible ML pipeline？ | production path 不依赖 `check_*`、`*_smoke.py`、`*_development.py`、私有跨脚本 API 或隐式 monkey patch；明确 `experiment_role`；clean import/dry-run 可执行 | 不进入正式实验；所有结果只能是历史/诊断证据 |
| **G1 Native** | source-aware / context / scale 是否优于 vanilla RIGNO？ | Level-A native accuracy；3 seeds；vanilla RIGNO、关键 support/context/scale/reconstruction 消融；参数量、训练预算、split 和指标同口径 | 方法贡献不成立，只能称特定回归管线 |
| **G2 Baselines** | 与 RIGNO、thermal SOTA/强基线是否公平？ | 至少两类代表性强基线；优先覆盖 vanilla RIGNO、FNO/DeepONet、GINO/Transolver、DeepOHeat 系、Therm-FM/DeepOHeat-v2 相关设定；同一 split、label budget、参数量范围、训练预算、硬件和 wall-clock 边界 | 无法声称 SOTA、优于竞品或方法必要性 |
| **G3 Resolution** | conditioning-query decoupling 是否在 16k/32k/[64k] 成立？ | Level-B 的 E/U Pareto；误差、重建 floor、fresh/resident/Q2、RAM/VRAM、失败/超时均完整；32k 不得只留最好值，64k 为 optional exploratory | 跨分辨率贡献不足；64k 不得写成生产能力 |
| **G4 Physics** | conservation / flux / Robin / hotspot residual 是否可信？ | 总功率守恒、离散能量残差、界面热流连续性、Robin residual、source/peak/hotspot residual；至少一组高对比度界面或受控 `R_contact>0` 实验 | 只能称监督回归，不能称 physics-aware/trustworthy surrogate |
| **G5 Generalization** | source / geometry / physics OOD 是否可解释？ | 预注册 geometry OOD、source OOD、material/Robin/strong-cooling physics OOD；报告均值、尾部、失败案例和适用域 | 泛化主张受限；不得写 arbitrary geometry/cross-chip transfer |
| **G6 External Case** | 是否至少有外部/公开/工业风格案例？ | 至少一个 public benchmark 或 F2F-style/package case；独立输入、标签、基线和 timing；与 P1i 主表分开 | DAC/ICCAD/DATE 的真实应用证据不足 |
| **G7 Design Loop** | repeated-query thermal optimization 是否产生真实收益？ | 至少一个布局、功率分配或热约束优化循环；代理、选择性 FVM/trust gate 和 solve-at-every-step 的 accuracy/peak/location/total wall-clock 对照 | EDA 故事停留在预测器，不能声称 design-scale benefit |
| **G8 Sealed** | 新 V7 sealed held-out test set 能否一次性提供最终确认？ | 所有选择、调参、route freeze 和 gate 记录完成后，单次打开新命名 sealed held-out test set；保存 raw receipt、失败样本和完整指标；V6 sealed IID 仍不动 | publication evidence 不冻结；不得挑选结果或反复查看标签 |
| **G9 Artifact** | clean checkout 是否能复现主结果？ | 依赖边界、标准 test suite/CI 或等价检查、最小 dry-run、reference inference pipeline、hash manifest、主表自动生成、无大文件入源码 | artifact/reproduction risk；不能称 publication-ready |

Gate 依赖顺序为 `G0 -> G1/G2 -> G3/G4/G5 -> G6/G7 -> G8 -> G9`。允许并行开发，但不得在前置 gate 未通过时将后置 gate 的 exploratory 输出写成主结论。G8 是最后的证据打开动作，不是调参工具。

## 5. E/U、分辨率与 external benchmark 的角色冻结

### 5.1 E/U 路线

| 路线 | V7 角色 | 禁止混用 |
|---|---|---|
| **E**：`E16384_reconstruction` | V6 frozen reference route；Level-B 的主 production candidate 和准确率—延迟 Pareto 锚点 | 不得把 V6 结果改写为 V7 新训练结果；不得把 reconstruction latency 隐去 |
| **U-v2**：`U_v2_16384_reconstruction` / `U_v2_direct240825` | 平行 route / architecture comparison；用于检验 output-query extrapolation 与 direct-output trade-off | V6 中是 valid-only characterization，不得直接升级为 test-confirmed production route；不得和 E 合并为一个 seed |
| **E240825 direct control** | architecture control，检查直接高分辨率输出的代价与误差 | 不得当作默认生产路线或用来替换 E reconstruction |

E/U 的比较必须固定 sample IDs、hardware、lifecycle、FVM reference、timing boundary 和 metrics。不同 route 的数值不得用“最优一格”拼出一条新曲线。

### 5.2 16k / 32k / 64k

- **16k**：V6 已支持的主 operating point；V7 的 primary high-resolution/FVM evidence。它必须同时通过 accuracy、peak/interface、资源和端到端边界审计。
- **32k**：exploratory scalability / resolution stress。V6 已显示其 point-global 改善边际小且 source/peak/resource 不一致，故不能默认设为生产点。只有完整通过 G3 后，才可作为 candidate operating point。
- **64k**：optional engineering stress test，不是当前能力、不是默认生产点、不是 publication claim。若实施，必须单独记录 source-stratum capacity、graph/build failure、RAM/VRAM 和是否达到 FVM/full-field 评价；失败也必须保留。

分辨率解耦的候选主张是“conditioning resolution 与 query resolution 可分离”，不是“任意 N 都有效”。每个 N 都需要自己的 query/support/reconstruction 语义和失败边界。

### 5.3 External benchmark

External benchmark 用于 G2/G5/G6 的独立泛化和现实相关性证据，不得替换 P1i native 主表或 V6 FVM 对照。优先覆盖：

- Therm-FM 所强调的 public HotSpot 与 industrial 3D-IC/package benchmark 类型；
- 至少一个 F2F-style 或公开 package/thermal case（若数据许可和标签定义允许）；
- 若只有论文叙述而无可复核数据/代码/标签，不得写成“已完成外部 benchmark”，只能写 related-work target。

External case 必须单独报告输入字段、网格/DOF、标签求解器、训练或 adaptation data、硬件、是否使用 fine-tuning、cold/fresh/resident 语义和不确定性。外部结果不能与 P1i 混算均值、seed、speedup 或 SOTA 排名。

## 6. 最新研究带来的 V7 问题，而非既成能力

### 6.1 Therm-FM

[Therm-FM: Foundation Model is ALL YOU NEED for 3D-ICs Thermal Simulation](https://arxiv.org/abs/2605.22663)（arXiv v2，2026-05）将竞争门槛推进到：预训练 PDE foundation model、steady/transient thermal adaptation、thermal-equivalent multi-fidelity、public HotSpot/industrial package benchmark，以及跨芯片仅用少量 target samples 的适配。

V7 对它的转译是：

- G2：若作为 external/application baseline，必须明确预训练数据、目标域样本数和 adaptation 预算；不能只引用其最高 speedup。
- G5/G6：外部 benchmark 与 cross-chip/geometry transfer 要分开，报告 target-sample curve 和失败域；P1i 内部 IID 结果不等价于 cross-chip few-shot。
- G3/G9：多保真或预训练路线的编译、缓存、适配和推理成本都要落入统一 lifecycle boundary；不能只比较 steady-state forward。

### 6.2 DeepOHeat-v2

[DeepOHeat-v2: Self-Improving Operator Learning for Fast and Trustworthy Thermal Optimization in 3D-IC Design](https://arxiv.org/abs/2608.16080)（arXiv v1，2026-08-17）把问题推进到高对比度多芯片界面、离散 physics loss、energy-form conditioning、matrix-preconditioned optimization、hotspot trust gate 与 solver-refined self-improvement。

V7 对它的转译是：

- G4：材料界面不连续时，连续强形式 residual 不能默认有效；V7 需注册离散控制体/能量形式或等价的界面守恒指标，并报告 `R_contact>0` 的受控行为。
- G5：OOD 不只指几何，还包括导热率对比度、Robin/强冷却和热源结构；必须报告高误差 tail 与 fail-closed 条件。
- G7：trust gate / selective FVM / self-improvement 可作为设计环候选机制，但任何被 solver 修正的样本都必须单独标识，不能把修正后结果冒充 surrogate-only accuracy。
- G8：自改进不允许触碰 sealed labels；所有更新必须由已打开的 train/valid evidence 和预注册 held-out criterion 控制。

这些论文的速度、误差和数据效率数字不直接与 Heat3D V6/V7 比较。只有在相同问题、DOF、输入边界、标签预算、硬件和生命周期定义下完成复现或对齐实验，才可进入 G2/G6 的 comparison table。

## 7. Publication claims 冻结

### 7.1 当前允许保留的 claims

以下是 V6 已有且边界明确的 claims，可在 V7 文档/论文中作为 frozen baseline 引用：

1. 在 frozen P1i generator、材料/BC/source ranges、`R_contact=0`、checkpoint 与 reconstruction semantics 内，`E16384_reconstruction` 的 V6 corrected `test_iid=128` full-field point-global relative RMSE 为 `2.992001%`；valid32 主表为 `2.7023%`。必须同时给出 sample-first、raw/source/peak/interface 指标。
2. 在 WSL2 Attempt 4 的 paired workload 边界内，E16384 的 fresh/Q2 speedup 相对 persistent CPU FVM 约为 `2.001x/2.005x`；devbox 是独立 hardware-state replication，不是额外 model seed。
3. V6 的 16k + reconstruction 是其 valid32 evidence 中的 accuracy-latency Pareto family；32k 是 exploratory；FVM 是物理参考而非被 surrogate 超越的对象。
4. V6 已识别 test peak-error tail；该边界应被公开，不能由平均指标掩盖。source-aware support、query resolution、layer/interface reconstruction 和 lifecycle-separated measurement 是可复核的研究对象，但还不是“已优于所有基线”的结果。

这些 claims 只描述 V6 冻结范围，不能被重新包装为 V7 新实验、universal geometry、industrial sign-off 或跨域迁移证据。

### 7.2 V7 候选 claims（全部 conditional）

下列四条是 V7 冻结的候选 contribution，只有在对应 gates 和统一对照完成后才可升级：

| ID | Candidate claim | 最低证据条件 |
|---|---|---|
| **C1** | **Physics-conditioned operator**：a geometry-aware graph neural operator explicitly conditioned on heterogeneous conductivity, distributed heat sources, and cooling boundary conditions | G0、G1、G2、G4；必须证明物理条件输入与机制，而不只是把字段拼接进网络 |
| **C2** | **Source-aware sparse conditioning**：a physics-aware sparse support preserves heat sources, interfaces, and thermal boundaries under a small conditioning budget | G1、G2；source/interface/Robin coverage、替代 support 消融和 native/全场误差均通过 |
| **C3** | **Resolution-decoupled inference**：conditioning-query decoupling enables high-resolution reconstruction without retraining at target resolution | G0、G3、Level-A/Level-B 分层、16k/32k 完整 Pareto；64k 不得默认包含 |
| **C4** | **Deployment-scale acceleration**：an end-to-end high-resolution surrogate workflow is evaluated against an accuracy-qualified FVM solver under matched production boundaries | G3、G4、G6、G7；同时报告误差、物理指标、fresh/resident/optimization 全流程，不能用 resident-only speedup |

候选 claim 的措辞必须带适用域，例如 `on the registered P1i-compatible distribution`、`under the matched WSL2 FVM workload` 或 `on the named external case`。未经 G5/G6 通过，不得把 C1–C4 扩写为 universal、cross-chip、industrial 或 arbitrary-geometry claim。

### 7.3 当前禁止的 publication claims

在本合同生效时，以下表述一律禁止：

- “SOTA”“优于 DeepOHeat、DeepOHeat-v2、Therm-FM、GINO、Transolver、vanilla RIGNO”或任何未完成同口径直接基线的排名；
- “arbitrary geometry / universal PDE / cross-chip generalization / few-shot adaptation”，除非 G2/G5/G6 有相应独立证据；
- “industrial sign-off accuracy”“真实封装已验证”，除非 external case 有可审计的结构、标签和流程；
- 将 `8x–9x` known-topology/runtime-only、`~265x` resident 或任意 cache-hot 数字写成新案例 full-field E2E speedup；
- “优于 FVM 的物理精度”“已满足守恒/界面通量/Robin physics”，除非 G4 逐项通过，且 FVM 仍必须保留为参考；
- “已解决 OOD、finite contact resistance、high-contrast interface、uncertainty/calibration、hotspot trustworthiness”；
- 将 32k 或 64k 写成当前生产默认或稳定任意高分辨率能力；
- 将 V6 `test_iid`、V6 sealed IID 或任何未来 V7 sealed label 用于事后调参、挑选 route、改阈值或补写历史 claim；
- 把 point-global relative RMSE、sample-first CV-relative RMSE、raw CV RMSE、peak/interface/source RMSE 混写成一个“RMSE”。

## 8. V7 推进路线与交付物

### V7.0：问题、证据和 claim 冻结（当前阶段）

- 交付本合同、gate matrix、claim/evidence map 和 V6 frozen-artifact denylist。
- 明确 Level-A/Level-B、E/U、resolution、external benchmark 和 sealed semantics。
- 不训练、不推理、不求解、不生成数据、不访问 sealed labels。

### V7.1：reference inference pipeline 解耦（G0）

- 将正式训练/推理所需的 model config、normalization、checkpoint load、features、grouping、metrics、reconstruction 收口到稳定库模块。
- 保留 legacy/smoke 脚本作为兼容层或历史证据，但不让 production/import graph 依赖它们；正式路径应成为 reproducible ML pipeline。
- 用 explicit dependency injection 替代跨脚本私有 API 和 monkey patch；所有运行写明 `experiment_role`。

### V7.2：native baselines 与机制消融（G1/G2）

- 在同一 registered native split、label budget、parameter/training budget 与硬件上完成 vanilla RIGNO、support/context/scale/reconstruction 消融。
- 选择至少两类代表性 external baselines/strong baselines；对 DeepOHeat 系、Therm-FM、DeepOHeat-v2 只在可复核且边界对齐时进行直接 comparison，否则保留为 literature target。
- 形成 seed 统计、统一主表和 negative-result 表。

### V7.3：高-N、物理与 OOD（G3/G4/G5）

- 以 16k 为主、32k 为 exploratory、64k 为 optional stress，分别记录 Level-B accuracy/latency/memory/failure。
- 加入 discrete conservation/interface/Robin/hotspot 指标，注册 high-contrast 或 finite-contact 受控实验。
- 形成 geometry/source/physics OOD 矩阵；每个 OOD split 都有适用域、失败案例和 fail-closed 条件。

### V7.4：External case 与 design loop（G6/G7）

- 至少完成一个 public/industrial-style/F2F-style case，和 P1i 主证据分表。
- 实现一次 repeated-query thermal optimization，报告 surrogate-only、trust-gated/reference-solver 和 solve-at-every-step 的 accuracy、peak/location 和总 wall-clock。

### V7.5：一次性 final test 与 artifact（G8/G9）

- 所有 route、baseline、threshold、figure/table 和 claim freeze 后，才打开新命名 V7 sealed test 一次。
- 归档 raw receipt、hash、完整指标、失败样本和版本绑定；V6 sealed IID 仍禁止触碰。
- 用 clean checkout、最小命令、标准 tests/import checks 和 machine-readable table 复核主结果；未通过 G9 不写 publication-ready。

## 9. 报告与停止规则

每项 V7 实验必须随结果记录：

```text
gate / experiment_id / dataset_role / split / model_id / seed
native_or_query_N / reconstruction / fvm_reference
metrics / timing_boundary / hardware / label_budget / code_sha / data_sha
status(pass|fail|exploratory) / failure_reason / allowed_claims
```

任何 gate 若失败，保留失败记录并停止该 claim 的升级；不得通过删除 tail、改用不同 FVM 网格、替换 lifecycle、合并 E/U seed 或重命名实验来“修复”证据。G8 一旦打开，除正式归档外不得进行新的选择或调参。

本合同提交后，本阶段停止。后续若要训练、推理、生成数据、运行 solver、打开任何 sealed labels、修改 V6 artifact 或实施 V7.1–V7.5，必须以新的明确授权和相应 preregistration 开始。

## 10. 主要依据

- [V6/P1i scientific closeout](v6_p1i_closeout.md)
- [V6/P1i publication evidence consolidation](v6_p1i_publication_evidence_summary.md)
- [V6/P1i handoff](v6_p1i_handoff.md)
- [V6 total closeout](v6_total_closeout.md)
- [V6 production inference final closeout](v6_production_inference_final_closeout.md)
- [V6 source-aware resolution limit and solver timing](v6_source_aware_resolution_limit_and_solver_timing.md)
- [V6 phase index](v6_phase_index.md)
- 外部只读审计：`publication_review_and_v7_readiness.md`、`heat3d-ic_审稿报告.md`（2026-08-26）
- [Therm-FM](https://arxiv.org/abs/2605.22663)
- [DeepOHeat-v2](https://arxiv.org/abs/2608.16080)
