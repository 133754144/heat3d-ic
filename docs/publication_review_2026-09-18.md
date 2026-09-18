# Heat3D-IC 论文发表导向审稿报告与补齐工作清单

审阅日期：2026-09-18

审阅对象：[133754144/heat3d-ic](https://github.com/133754144/heat3d-ic)

报告性质：基于实际代码、历史研究文档和冻结结果的模拟同行评审；不是会议录用承诺，也不是新的实验结果。

适用对象：论文选题、贡献收敛、实验设计、代码整改与投稿前检查。

## 1. 审稿结论

**项目已经具备论文主体，但尚不足以支持“通用、领先、可信的 3D IC 热分析神经算子”这一宽泛主张。最有发表潜力的核心，是在有限条件采样预算下，利用热源、材料分块和界面布局保留关键物理信息，并结合尺度—形状建模完成全场预测。**

已有三随机种子内部消融、共同全场的支撑点归因、外部算子基线，以及受约束的求解器计时证据。项目不应再被描述为“仅在 RIGNO 上跑通了一个热传导数据集”。然而，验证集长期复用、不同模型/路线证据拼接、强基线比较的信息预算差异、热点可靠性和真实设计闭环仍限制发表结论。

本审稿人的判断如下。这里的“偏拒稿”是假设以当前证据直接投稿时的判断，不是录用概率估计。

| 投稿方向 | 当前判断 | 核心障碍 | 有效的提升方向 |
| --- | --- | --- | --- |
| NeurIPS / ICML / ICLR 主会 | 偏拒稿 | 相对已有图算子的通用方法新意与可迁移性不足；单一生成器上的收益占主导 | 明确方法原理、解耦归因、多分布验证、与强基线的预算曲线 |
| DAC / ICCAD | 方向匹配，有竞争潜力，当前证据仍偏拒稿 | 热场预测收益尚未充分转化为设计收益；热点低估、总成本、独立最终确认不足 | 外部封装案例、可靠设计闭环、完整成本与应用风险评价 |
| DATE / ASP-DAC | 适合优先打磨 | 若主张过宽，仍会被比较公平性和验证集依赖卡住 | 收敛为稀疏物理条件表示，完成最终确认与一个有意义的应用流程 |
| ISSCC / VLSI Circuits | 当前题型不匹配 | 主体是仿真/分析算法，缺少相应电路实现与实测贡献 | 只有另行形成芯片实现或测量主线时再考虑 |

DAC 明确覆盖设计分析、仿真和建模中的算法贡献；ISSCC 强调具体测量数据，不能把两者当成同一种“集成电路顶会”评价体系。[R10–R11]

## 2. 审阅范围与证据版本

### 2.1 版本边界

本报告继承前一轮只读审阅，并在成稿时复核关键结果文档和 GitHub 分支身份。没有依据 README 推断最新研究，也没有逐行复核全部历史提交。

| 证据层 | 快照 | 使用方式 |
| --- | --- | --- |
| GitHub main | `9cb6b374cf2b6f9dde8e5f06da078dee169d9b43` | V6 冻结结果与历史主线；不是全部最新 V7 研究 |
| GitHub research/v7 | `365576daa566e0eb786eba93486910dea754cb1a` | G1 正式训练、消融、共同域归因 |
| GitHub research/v7-g2-baselines | `d9d961aa6970f60a45dea7995f1b82dab2e335d8` | G2 基线、公共评估与历史证据限制 |
| 本地 codex/v7-g2-baselines-finalize | `2960ab52ba90cea3ac4d321d36333835e3271808` | 最新 G1 e200 与 Therm-FM 同域全场比较；成稿时尚未出现在已核验 GitHub 分支中 |
| devbox 上一轮只读检查 | 主仓库 `2de1bf9`；G2 工作树 `d9d961a` | 核对远端仓库身份；没有把远端主仓库当成最新全部证据 |

本地最新比较属于可审阅的本地证据，不应标注成已经发布到 GitHub 的结果。本报告引用其数值但不替他人推送该研究分支，也不复制大型预测或 checkpoint。

### 2.2 证据强度和限制

- 表中研究数字主要来自已保存的结果、统计和追溯记录；本轮成稿不是独立重训练或全量复算。
- 前一轮对公共 evaluator 做了合成输入检查，并检查最新比较文件的 768 条记录唯一性；不等于完成所有代码测试。
- V6 seed1/seed2 的目标 checkpoint 二进制已被记录为永久丢失。历史汇总不因这一事实自动失效，但后续 replay 和独立复核能力确实受限。
- G1/G2 的大量主要比较仍是 `valid_iid`；不能写成独立 test 或 OOD 结果。
- 论文中每个主张应绑定模型、checkpoint、数据域、split、输入点数、查询点数、重建方式、指标和计时边界。
- 本报告中的补齐工作是建议与验收条件，不是授权立即训练、运行评估、生成数据或打开 sealed 标签。

## 3. 研究问题与实际贡献

### 3.1 当前任务定义

项目主要学习稳态热传导的条件到温升场映射：

\[
(\Omega,\mathbf K(\mathbf x),q(\mathbf x),\mathrm{BC})
\longmapsto \Delta T(\mathbf x),\qquad
-\nabla\cdot(\mathbf K\nabla T)=q.
\]

已验证主范围是受限的多层结构、异质/对角各向异性导热、参数化热源与冷却边界，以及理想界面接触。应逐项区分：哪些几何/材料信息是模型输入，哪些用于支撑选择，哪些仅用于标签生成或评价。

“图网络能够处理点云”不证明本项目已经解决任意几何、显式 TSV/微凸点建模、有限接触热阻、瞬态或实验封装预测。当前也不能仅因输入含物理量就称为满足物理约束的可信求解器。

### 3.2 历史研究的论文化整理

| 阶段 | 实际进展 | 在论文中的角色 |
| --- | --- | --- |
| V1–V2 | 数据与监督接口、训练诊断；发现扩容不能解决主要误差 | 研究基础和失效动机 |
| V3 | 图覆盖与信息传递路径稳定化 | 原始算子适配难点，不宜独立包装成通用新算法 |
| V4–V5 | 形状与幅值失配、物理条件、尺度表示与训练目标 | 方法形成过程，最终由同条件消融支撑 |
| V6 | 连续物理参数、随机分块、高分辨率推理/重建、FVM 计时 | 全场可行性与受约束系统证据 |
| V7 | 21 次正式内部训练、多类外部基线、统一指标与统计 | 当前论文主要证据 |

各阶段的数据分布、温升范围、边界和指标不同，不得直接绘制“V2→V7 误差持续下降”并将其解释为算法净提升。

### 3.3 建议收敛的三项贡献

1. **物理布局感知的稀疏条件表示。** 用标签无关的 q/k 分块、界面和边界覆盖，缓解小热源和局部材料结构在有限支撑下被遗漏的问题。
2. **面向热场尺度变化的条件化预测。** 将整体尺度与空间形状分开建模，用物理输入生成上下文并学习尺度修正；FiLM 是辅助机制。
3. **输入、查询和重建分辨率分离的全场评价。** 在明确路线语义的前提下，报告高分辨率误差、热点风险和完整计算成本。

其中第一项的独立归因最清楚；第二项需要更强的尺度先验对照；第三项需要补齐统一成本和新路线最终确认。

## 4. 可用于论文的现有数据

### 4.1 指标定义

令第 \(s\) 个样本第 \(i\) 个节点的真值温升为 \(y_{si}\)，误差为 \(e_{si}\)，控制体积权重为 \(w_{si}>0\)。

- **Point-global relative RMSE（PG）**：\(100\sqrt{\sum_{s,i}e_{si}^2/\sum_{s,i}y_{si}^2}\)，当前主要历史定义不含 CV 权重。
- **Sample-first CV-relative RMSE（SF-CV）**：\(\frac{100}{S}\sum_s\sqrt{\sum_iw_{si}e_{si}^2/\sum_iw_{si}y_{si}^2}\)。每个样本先归一化，再等权平均。
- **Raw CV RMSE** 与 **非加权 RMSE** 不同，均以 K 为单位；以各冻结 evaluator 的聚合定义为准。
- **Peak RMSE**：逐案例预测最大温升与真实最大温升之差的均方根；不等于真实热点区域的场误差，也不度量热点定位误差。
- `valid_base_mse` 是归一化训练/验证量，不应当作物理温升相对误差。
- 下列表格的 `±` 若无特别说明，均为三个训练种子之间的样本标准差，不是 384 个独立案例的标准误。

### 4.2 G1：内部机制成立，但需要限制归因

G1 完成 7 variants × 3 seeds = 21 runs，完整 200-epoch schedule，train 768 / valid128；checkpoint 按冻结的验证集 SF-CV 选择。[E1–E2]

| Native 1,024 点变体 | SF-CV（%） | PG（%） |
| --- | ---: | ---: |
| Full | **1.692 ± 0.048** | **2.058 ± 0.081** |
| no FiLM | 1.986 ± 0.025 | 2.265 ± 0.036 |
| vanilla RIGNO | 22.389 ± 1.568 | 22.342 ± 3.236 |
| 参数量匹配 vanilla RIGNO | 25.162 ± 11.003 | 24.021 ± 5.459 |

预注册比较中，Full 对 vanilla 的 PG 配对效应为 20.43909 个百分点，95% CI 为 [17.12366, 23.60488]；FiLM 的 SF-CV 效应为 0.29417 个百分点，95% CI 为 [0.21456, 0.37041]。[E1]

**审稿意见：** 这支持整套 Heat3D 方案相对注册 vanilla 对照的收益。实际代码中，vanilla 使用归一化温升 MSE，Full 使用 shape/scale 等组合目标，输出参数化和附加条件路径也不同，因此不能把全部差距归因于 FiLM、支撑点或某一个网络模块。相同参数量和 epoch 数不足以消除上述混杂。容量匹配 vanilla 的高种子方差也必须保留。[C1]

### 4.3 H2：最有说服力的贡献证据

不同支撑点方案的正式归因落在相同 240,825 点物理域，而不是仅比较各自采样点。主路线为 `U16384→240825`，直接查询全场为稳健性对照。[E2]

| 支撑方案 | 热源区域 RMSE（K） | 全场 PG（%） | 峰值 RMSE（K） |
| --- | ---: | ---: | ---: |
| Full 物理布局感知支撑 | **3.80482 ± 0.149586** | **3.38519 ± 0.16284** | **5.14703 ± 0.105692** |
| generic support | 5.55262 ± 0.072306 | 4.56933 ± 0.087658 | 8.49609 ± 0.392335 |
| CV-only support | 5.64211 ± 0.297967 | 5.14206 ± 0.398149 | 8.46651 ± 0.392975 |

热源区域误差按均值比约下降 31%–33%。正式配对效应为：相对 generic 下降 1.74615 K，95% CI [1.48199, 2.04268]；相对 CV-only 下降 1.84057 K，95% CI [1.53070, 2.18014]。直接查询路线保持同方向。[E1–E2]

**审稿意见：** 这是可成为正文核心的机制证据。准确名称是 `physics-layout-aware sparse support`，不是 source-amplitude-aware 或 learned sampling。generic 对照保留边界/界面覆盖，仅移除 q/k 分块配额，更适合归因；CV-only 同时移除多个结构覆盖，应解释为更大的组合变化。

### 4.4 P1i 外部算子基线：优势明显，范围仍有限

| Native 1,024 点模型 | SF-CV（%） | PG（%） | 参数量 |
| --- | ---: | ---: | ---: |
| Heat3D V6 历史 cohort | **1.6294 ± 0.0131** | **2.0273 ± 0.0947** | 892,776 |
| GINO | 15.0139 ± 0.9130 | 17.7315 ± 1.6551 | 13,673,988 |
| Transolver | 16.0028 ± 1.0310 | 18.5565 ± 0.9266 | 716,737 |

来源为冻结 valid-only native 表；Heat3D V6 按 PG 选择 checkpoint，GINO/Transolver 按验证 SF 选择。[E3]

**审稿意见：** 可以写“在本任务、本适配与冻结协议下有明显优势”，不能写“RIGNO 普遍优于 GINO/Transolver”或“全场 SOTA”。需要保留基线适配、调参预算和 checkpoint 选择差异；历史 V6 二进制丢失也限制进一步统一复算。

### 4.5 DeepOHeat-v1 外部域：精度积极，监督成本不能隐去

此处是另一物理域：101×101×56 = 571,256 点；不能与 P1i 240,825 点表混排。[E4]

| 方案 | 训练物理案例数 | 全场 SF-CV（%） | checkpoint 口径 |
| --- | ---: | ---: | --- |
| Heat3D e600 | 768 | **0.709888 ± 0.007765** | 固定 epoch 600 终点 |
| DeepOHeat-v1 matched | 768 | 1.305167 ± 0.268321 | 验证选择 best |
| DeepOHeat-v1 full-minus-valid128 | 99,872 | 1.146688 ± 0.038323 | 验证选择 best |

同案例数时，Heat3D 相对误差约降低 45.6%。但前者使用温度监督标签和稀疏支撑，后者使用 PDE/BC 物理约束；相同案例数不等于相同信息预算。标签求解、训练与推理成本必须分项列出。

P19 已有同 devbox 的推理分析，但 Heat3D 的 batch=1 全场查询与 DeepOHeat 的 batch=4 等执行边界并未完全对齐，不能据此形成新的统一 Pareto 排名。[E5] 外部域重新训练也不等于跨芯片零样本迁移。

### 4.6 Therm-FM：最新比较不支持全面领先

以下来自本地 `2960ab5` 的 G1 Full e200 × Therm-FM 同域比较。两侧使用 P1i valid128、相同 240,825 点真值和公共 evaluator；这是本地、valid-only 证据。[L1]

| 指标 | Heat3D U-v2 direct | Therm-FM |
| --- | ---: | ---: |
| SF-CV（%） | 3.1262 ± 0.1772 | **2.5972 ± 0.2399** |
| PG（%） | 3.4786 ± 0.1728 | **3.1490 ± 0.4603** |
| 非加权全场 RMSE（K） | 2.7625 ± 0.1372 | **2.5007 ± 0.3656** |
| MAE（K） | 2.0976 ± 0.1171 | **1.6363 ± 0.1963** |
| 峰值温度 RMSE（K） | **5.2049 ± 0.1105** | 5.5718 ± 0.1800 |
| 真实热点区域 RMSE（K） | 5.7183 ± 0.3287 | **3.9860 ± 0.1704** |

逐案例先平均三个种子，再做 10,000 次 paired case bootstrap，差值定义为 Heat3D−Therm-FM：SF-CV 为 +0.5290 个百分点，95% CI [0.1751, 0.8828]；热点区域 RMSE 的案例均值差为 +1.6900 K，95% CI [1.2293, 2.1391]；峰值绝对误差差为 −0.1475 K，95% CI [−0.9030, 0.5714]。[L1]

这些区间条件化于冻结的三个训练种子，不是对所有训练随机性的完全推断。案例 RMSE 的平均差也不等于上表“先聚合 SSE 再开根号”的全场 RMSE 之差，不能互相替换。

**审稿意见：** 整体误差和热点区域结果偏向 Therm-FM。不能依据较低的峰值 RMSE 均值，宣布 Heat3D 峰值预测显著领先。Therm-FM 使用 Poseidon-T 预训练和 741-channel 密集输入，不是同信息预算 scratch 对照。Heat3D 约 0.893M、Therm-FM 约 21.44M 参数，但参数少必须通过实际成本测量才能升级为效率贡献。[E6, L1]

### 4.7 V6 全场确认与速度：可靠但受限

冻结 V6 seed0、E16384 重建到 240,825 点的 corrected test128 结果为：[E7–E8]

| 指标 | 值 |
| --- | ---: |
| 全场 PG | 2.992001% |
| SF-CV | 2.948519% |
| Raw CV RMSE | 2.389097 K |
| 热源区域 RMSE | 3.940479 K |
| 峰值温度 RMSE | 5.726285 K |
| 界面区域 RMSE | 0.355507 K |

WSL2 Attempt 4 为主要计时证据：E16384 fresh 中位延迟 0.8832 s，FVM 参考 1.7007 s；正式配对工作负载的 fresh/Q2 加速分别为 2.001×/2.005×。配对比值按冻结生命周期协议计算，不等于直接相除上述两个总体中位数。devbox 是独立硬件状态复核，不是新增模型种子。[E7]

预测热点存在尾部风险：test128 逐案例峰值相对误差 p95 为 10.437191%，最大 15.987477%；最高十个案例贡献 52.30% 的峰值误差 SSE。[E8] 全场约 3% 的平均误差不能作为工业热点签核的替代条件。

### 4.8 E 与 U 的语义不能混用

| 路线 | context/scale 锚点 | 编码器输入点 | 查询点 | 最终输出 |
| --- | ---: | ---: | ---: | ---: |
| E16384 | 1,024 | **16,384** | 16,384 | 重建 240,825 |
| U-v2 16384 | 1,024 | **1,024** | 16,384 | 重建 240,825 |
| U-v2 direct240825 | 1,024 | **1,024** | 240,825 | 直接 240,825 |

E 使用高分辨率编码器输入，U-v2 才保持 native 编码器输入不变。因此 E 路线的 V6 测试结果和速度不能用于宣称“U-v2 仅用 1,024 个输入点的测试能力已经确认”。查询侧条件特征、元数据与其构造成本也需单列，不能把“1,024 个编码器输入点”误写成系统只访问 1,024 点信息。[E9]

## 5. 主要审稿问题

### M1. 新颖性仍需从工程组合提升为可解释的方法贡献

RIGNO、GINO 和 Transolver 已建立了图/几何算子及跨分辨率方法基础；把 FiLM、尺度头和物理字段加入模型，本身不足以证明通用算法新颖性。[R1–R3]

建议围绕“有限条件采样对局部热源与不连续材料信息的保留”形成方法主线。可以给出受限条件下的覆盖分析，但覆盖保证不是温度误差保证。论文应说明物理布局先验何时可获得、在哪些类型的输入上可能失效。

### M2. 主结果来自不同 cohort，尚未形成一个统一的论文身份

V6 seed0 E-route 的测试、G1 e200 的消融、DeepOHeat 域 e600 的结果和 Therm-FM transfer 的比较各自成立，不能拼接成同一个最终模型的性能。需要冻结主模型与主路线，并将其它结果标为独立任务或历史支持。

### M3. 物理尺度先验的必要性尚未被充分解释

`physics_scale_only` 的 SF-CV 约 235%，说明没有学习修正时当前尺度近似非常差。现有实验支持“修正对本实现重要”，但不能排除“学习修正在补偿不适合当前分布的先验”这一解释。需要增加无物理先验的学习尺度对照，并观察修正量随功率、导热率和 Robin 参数的分布。

### M4. 验证集开发证据不能替代独立最终证据

预注册和配对 bootstrap 提高可审计性，但 valid128 长期参与 checkpoint 选择、方法开发和方案解释，仍可能产生选择偏差。V6 test 已打开，P23 也保留误读混合角色 CSV 的记录；该事件不能自动被解释为测试调参，也不能被删除后宣称完全未访问。[E10]

新的 V7 sealed 需要独立命名、记录访问历史并在所有选择完成后一次性评价。DeepOHeat 官方参考案例曾用于求解器一致性核验，不能不核对具体案例身份就笼统宣称所有外部官方标签从未被读取。[E12]

### M5. 热场误差与物理可信度之间仍有缺口

参考求解器已有能量/边界诊断和有限接触热阻实验路径，外部 DeepOHeat 离散系统也有复现记录；这不等于神经预测全场已满足守恒和界面物理。需要对预测场计算实际离散算子残差、功率收支、Robin 残差和界面通量误差。

匹配同一离散系统的解主要验证实现一致性。独立物理精度还需要网格收敛、解析/制造解或独立求解器验证。只验证温度重建权重和为一，也不能推导通量守恒。

### M6. 泛化与真实应用证据仍不充分

P1i 内部 IID、IID 中的困难子集、在外部域重新训练、零样本跨域推理是四类不同证据。现有研究不能笼统写成 arbitrary geometry 或 cross-chip generalization。需要分别定义 source、material/cooling、geometry OOD，并完整报告尾部与失败。

### M7. 速度和训练成本尚未支持设计规模收益

DeepOHeat 系已经把竞争推进到可信度估计、求解器修正和设计环；Therm-FM 已涉及预训练、多保真与跨芯片适配。[R4–R7] 仅与一次 FVM 求解比较不足以证明实际设计价值。

固定几何、材料和边界系数而只改变热源时，线性系统的分解复用、降阶模型和热响应基函数是有意义的传统对照；不应只比较每次从头求解。传统工具 HotSpot 和 3D-ICE 本身也已有性能优化。[R8–R9]

应同时报告标签生成、训练、推理和回退成本。仅在 1,024 点监督，不意味着标签只花费了 1,024 点的求解成本。

### M8. 可复现性已显著改善，但证据归档仍是硬约束

V7 已有独立 runtime/training 模块、轻量 CI 和等价性证据，不宜重复“完全没有测试”“生产路径全面依赖 smoke”的旧评价。当前主要风险是 checkpoint 持久化、不同脚本的语义漂移、部分关键测试未接入 CI，以及多个文档保留互相矛盾的当前状态。[E11, C4]

## 6. 代码审阅意见与整改边界

以下 P0/P1/P2 表示投稿准备优先级，不是漏洞严重性分级。

| ID | 优先级 | 代码事实或风险 | 应补齐的行为 |
| --- | --- | --- | --- |
| C-01 | P0 | 公共 evaluator 的汇总只按出现的模型检查种子数量；未强制完整两模型比较，也未拒绝重复案例 | 校验模型集合、种子集合、每个模型/seed 的唯一 case 集，区分单模型诊断与完整比较状态 |
| C-02 | P0 | 读取 truth 时检查 split，但未在该处核对 truth_row 对应 sample ID | 在读取温度前校验 ID、节点顺序、坐标身份和预期 archive hash；错误映射立即失败 |
| C-03 | P0 | 公共 evaluator 默认 value_kind 为 deltaT_K；绝对温度参考值默认 300 K | 强制显式温度表示/单位/参考值，复用稳定 runtime 合同；跨数据域缺项不得静默猜测 |
| C-04 | P1 | 公共指标函数未完整拒绝非法 CV 权重 | 对 shape、finite、正值、总量和输出 finite 做断言；定义零温升边界的独立评价规则 |
| C-05 | P1 | bootstrap/条件分析依赖总行数或 cohort 长度，未充分验证每组 seed 身份唯一 | 校验每个 case 的种子集合与完整匹配；统计状态绑定输入内容 hash |
| C-06 | P1 | 现有轻量 CI 主要触发 main/research-v7，执行 runtime/training 测试；新 G2 evaluator 测试未全部被执行 | 将活跃 G2 路径及相关 unit/regression 纳入 CI；不要把 compileall 当行为测试 |
| C-07 | P1 | 分层重建主要使用邻域加权，权重和为一不保证通量/能量一致性 | 增加重建后残差和界面审计；若改变算法，作为新变体而非覆盖历史结果 |
| C-08 | P1 | V6 部分 checkpoint 二进制已丢失；Git 中 hash 无法替代内容 | 外部持久化存储、内容寻址、双副本和实际恢复演练，源码不提交大模型 |
| C-09 | P2 | E/U、T/ΔT、不同 raw RMSE、单案例均值与全局聚合容易在脚本间漂移 | 共享 typed schema 与指标库；表格同时携带模型/数据/route/metric 版本 |
| C-10 | P2 | 历史状态、预备状态、已完成结果分散且存在陈旧摘要 | 建立唯一 publication evidence index，显式 supersedes、日期和可用主张 |

定位：公共指标/汇总和 truth 读取见 [C2]；bootstrap 见 [C3]；损失差异见 [C1]；CI 见 [C4]；重建见 [C5]；温度严格合同见 [C6]。

前一轮合成输入检查确认：仅含 Heat3D 的三 seed cohort 可通过汇总；重复 seed0 案例后可接受 256/128/128 的案例数量；NaN CV 权重可产生 NaN 指标而不立即报错。以上是防错缺口，不证明当前研究结果已出错。

同一轮检查了 [L1] 的最新公共结果：768 行对应两模型×三种子×128 案例，组合唯一，未发现上述重复问题。报告必须同时保留这一反证，避免把潜在风险写成既成数据错误。

## 7. 论文行文方案与主张边界

建议题目：**Heat3D: Physics-Layout-Aware Sparse Conditioning for Graph-Based Thermal Field Prediction in 3D ICs**。

### 7.1 推荐正文结构

| 部分 | 需要回答的问题 | 主要证据 |
| --- | --- | --- |
| Introduction | 有限支撑为什么会遗漏热源/界面信息？为什么现有方法仍不足？ | 受控示例、文献边界和具体设计需求 |
| Problem and evaluation contract | 输入、输出、物理假设和成本边界是什么？ | 方程、输入信息清单、split 与指标定义 |
| Method | 支撑、尺度和图条件化各自如何工作？ | 标签无关选择、尺度构造、模块边界、复杂度 |
| Mechanism experiments | 收益来自哪里？ | G1 与 H2 共同域消融，补充对齐目标的对照 |
| Full-field and external comparisons | 相同任务下与强基线相比如何？ | 分任务主表、种子统计、预算和路线说明 |
| Deployment/application | 是否减少实际设计成本？ | 统一端到端测量与一个设计闭环 |
| Limitations | 在什么条件下失败？ | 热点尾部、OOD、接触假设、外部预训练差异 |

### 7.2 必要图表

1. 方法图：显示物理输入、native 支撑、context/scale、编码器、查询与重建；E/U 分开画。
2. 机制表：Full、generic、CV-only、no-FiLM、尺度和公平 vanilla 对照；标明共同评价域。
3. 分任务基线表：P1i native、P1i full field、DeepOHeat 外部域分表，禁止跨域排名。
4. 准确率—成本图：完整延迟、峰值内存、误差；区分 fresh、cached、resident 与训练/标签成本。
5. 热点风险图：绝对误差、低估、定位、尾部和失败案例；展示预先指定或完整选取规则。
6. 若主投 EDA：设计闭环表，包括最终真值质量、违规率、求解器调用次数和总时间。

### 7.3 可写与不可写

| 当前可写 | 当前不可写 |
| --- | --- |
| 冻结 P1i valid 分布内，布局感知支撑降低共同全场热源误差 | 任意几何、跨芯片或所有热问题均优于均匀采样 |
| Full 相对注册 vanilla 对照有组合收益 | 全部收益来自某一个新模块 |
| 在外部 DeepOHeat 域重新训练后有较低验证误差 | 无监督成本或零样本跨域领先 |
| 轻量模型在有限编码器输入下可实现高分辨率查询 | 仅凭参数量宣布最快、最低成本 |
| 冻结 V6 E 路线在特定工作负载约 2× 加速 | 将 resident/cache 热态数字写成新案例端到端加速 |
| 当前部分数据域和指标优于部分基线 | 全面 SOTA，尤其是全面优于 Therm-FM |
| FVM 是参考，误差为对参考解的代理误差 | 精度优于 FVM、已达工业签核或满足全部物理约束 |

## 8. 附录 A：应该补齐的工作与验收标准

下列项目区分“证据阻断项”“核心论文补强项”和“投稿方向扩展项”。不要求把所有可选物理扩展一次性完成；若不实施，应收窄对应主张。数值成功门槛应根据应用需求在看新结果前冻结，不能为了让当前模型通过而事后设置。

### A.1 P0：先完成的证据与软件阻断项

| ID / 状态 | 工作与研究目的 | 交付物 | 验收标准 | 依赖 |
| --- | --- | --- | --- | --- |
| W01 待完成 | 冻结论文主模型、主路线和各证据身份；解决不同 cohort 拼接 | 一份 publication manifest 与 claim/evidence 表 | 每张表绑定 code/data/checkpoint/route/metric/split；明确 G1、V6、外部 e600 和 Therm-FM；本地 L1 经原任务完成发布/归档后再升级为公开证据 | 无 |
| W02 待完成 | 修复评估入口和统计完整性约束 | C-01～C-05 的代码变更及合成异常测试 | 重复/缺失案例、错误 seed、错配 truth row、NaN CV、缺温度语义均拒绝；合法冻结 fixture 指标保持一致；不读取新 test | W01 |
| W03 待完成 | 建立实际可恢复的模型/预测归档 | 外部 archive、内容 manifest、恢复记录 | 新 G1 三 seed 与主基线能从独立副本恢复并校验；V6 丢失项显式列为历史不可恢复，不冒充修复 | W01 |
| W04 待完成 | 冻结 split 访问账本与最终测试流程 | 数据角色清单、访问历史、sealed protocol | 明确已打开 test、valid、求解器核验案例和新 final holdout；任何评价不从混合角色文件自动发现样本 | W01 |

### A.2 P1：核心论文必须补强的科学与成本证据

| ID / 状态 | 研究问题与工作 | 交付物 | 验收标准 | 依赖 |
| --- | --- | --- | --- | --- |
| W05 部分已有 | 布局支撑收益是否独立于输出域与部分实现选择？保留已完成 H2，优先补缺失的共同域、预算或最终确认 | 支撑方案×输入预算×固定输出域表；预注册统计 | 同 case、truth、模型预算和评价域；generic 与 Full 差异可解释；不在各自采样点比较后宣布全场优势 | W01～W03 |
| W06 待完成 | 尺度先验究竟提供收益，还是主要由学习修正补偿？增加纯学习尺度、对齐目标/条件的直接输出对照 | 尺度/形状 factorial 或最小可识别消融；先验误差与修正量分布 | 报告三 seed、预定义主指标与失败；即使无收益也保留；不能仅以失效的 physics-only 为唯一对照 | W02～W03 |
| W07 部分已有 | 强基线比较在何种预算下成立？统一记录监督、PDE、预训练、参数、训练与标签成本 | 分 scratch/physics-informed/pretrained 的基线表，合理调参预算说明 | 不把同案例数当同信息预算；不同任务不混排；统一终点/选择规则或明确两套视图；保留 Therm-FM 不利结果 | W01～W03 |
| W08 待完成 | 稀疏输入是否带来实际误差—成本收益？测量主 U 路线与强基线 | 相同硬件、完整物理输入到同步全场的 profiling 与 Pareto 图 | 同批量/工作负载、输出域、精度要求；含图构建/特征/打包/重建；分 cold/fresh/cached/resident；报告 RAM/VRAM、失败和重复测量离散度 | W02、W03、W07 |
| W09 部分已有 | 预测场是否物理一致、热点是否可用？补模型场残差与标签精度验证 | 守恒/离散残差/Robin/界面指标、峰值低估与定位表；至少一种独立精度验证 | 使用实际物理算子与单位；覆盖材料对比度和冷却条件；区分参考离散误差与代理误差；有限接触未测则不宣称支持 | W02～W03 |
| W10 待完成 | 在哪些分布变化下仍有效？预注册 source/material-cooling/geometry 的适用域 | OOD 矩阵、均值与尾部、失败案例 | 真正与 train 分布分离，几何组级去重；重训练和零样本分开；不将 IID 困难子集重命名为 OOD | W04、冻结后的方法 |

### A.3 投稿方向扩展与最终交付

| ID / 状态 | 工作与价值 | 交付物 | 验收标准 | 适用/依赖 |
| --- | --- | --- | --- | --- |
| W11 待完成 | 外部公开/封装风格案例与实际设计闭环 | 至少一个独立案例；每步求解、纯代理、代理+选择性求解器三种流程 | 最终设计用参考求解器核验；报告目标值、热点低估、违规、总耗时及回退次数；含可复用矩阵/降阶等合适传统对照 | 主投 DAC/ICCAD 优先；依赖 W08～W09 |
| W12 待完成 | 通用方法原理与多分布迁移 | 条件采样覆盖分析或受控机制研究；跨分布数据效率曲线 | 理论假设和失效条件明确；覆盖不冒充误差界；不能只在一个生成器上观察后外推 | 主投 ML 顶会优先；依赖 W05～W07、W10 |
| W13 待完成 | 方法、路线、基线、阈值和主张冻结后执行最终独立确认 | 新命名 sealed 的一次性结果、访问 receipt、完整失败记录 | 不挑 seed、不删除失败、不看结果后重选路线；如结果不支持主张，应收窄主张；后续开发需要新的未来确认集 | W04 及拟写主张所需开发工作完成后；需另行明确执行授权 |
| W14 部分已有 | 可复现 artifact 与投稿包 | G2 CI、安装锁定、最小公开样例、复现命令、主表生成与数据/模型访问说明 | 干净环境可运行公开样例并复现预期指标；正式资产身份可核验；不把绝对本机路径和 /tmp 当发布接口；声明许可与上游归属 | 基础部分可先做；最终验证在 W13 归档后完成 |

### A.4 最小可行投稿包与执行顺序

**所有方向共同需要：** W01～W04，W05～W09 中与主张相关的缺口，W13～W14。内部消融和外部比较已有大量结果，不应机械重跑全部历史实验。

**EDA 主线优先：** 在上述基础上完成 W11，并按应用实际需要限定 W10。未做广泛 geometry OOD 时，可以明确限定结构族；未做 finite-contact 时，不把这一能力写进题目和摘要。

**ML 主线优先：** 完成 W12 与有实质分布差异的 W10，强化 W06 的因果归因。增加数据集数量本身不保证方法新颖性。

推荐顺序：W01 → W02/W03/W04 → W05/W06/W07 → W08/W09/W10 → 选择 W11 或 W12 → 冻结全部选择 → W13 → W14 最终归档。这里的并列表示依赖关系，不构成本轮启动任何工作或代理的指令。

**下一项最聚焦的科学任务：** 在相同 P1i 全场任务、明确条件信息预算和冻结 U-v2 路线下，检验布局感知支撑能否在热点误差—端到端成本上形成稳定收益。先复用已有 G1 三 seed 预测与 H2 结果，补齐缺失的成本边界；需要新增实验的部分单独登记。不要为了寻找更好数字先开 sealed。

### A.5 停止与降级规则

- 若统一成本后没有优势，论文应主打稀疏表示/机制，而不是部署加速。
- 若尺度消融表明物理先验无益，应调整贡献解释，不隐藏负结果。
- 若 Therm-FM 仍在全场/热点上更好，应展示适用预算与代价，不宣称全面领先。
- 若物理残差不能识别危险低估，不得称为可信门控；需要更合适的风险估计或限定应用。
- 若最终测试否定开发结论，不以改阈值、改子群、换 route 或重复查看标签挽救原主张。
- 不能恢复的历史二进制列入限制；新的重训练/新 cohort 不能伪装成旧模型恢复。

## 9. 附录 B：论文背景与参考文献

| 编号 | 文献/来源 | 与本课题的关系 |
| --- | --- | --- |
| R1 | Mousavi et al. *RIGNO: A Graph-based Framework For Robust And Accurate Operator Learning For PDEs On Arbitrary Domains*. NeurIPS 2025. [论文页](https://proceedings.nips.cc/paper_files/paper/2025/hash/dcb91f43033bb1d367d1848806dee98d-Abstract-Conference.html) | 已有多尺度图算子及跨分辨率基础；本项目需区分继承与新增贡献 |
| R2 | Li et al. *Geometry-Informed Neural Operator for Large-Scale 3D PDEs*. NeurIPS 2023. [论文页](https://papers.nips.cc/paper/2023/hash/70518ea42831f02afc3a2828993935ad-Abstract-Conference.html) | 几何与离散变化的强基线背景 |
| R3 | Wu et al. *Transolver: A Fast Transformer Solver for PDEs on General Geometries*. ICML 2024. [PMLR](https://proceedings.mlr.press/v235/wu24r.html) | 一般几何的物理注意力方法，对比需解释适配和预算 |
| R4 | Liu et al. *DeepOHeat: Operator Learning-based Ultra-fast Thermal Simulation in 3D-IC Design*. DAC 2023. [论文](https://arxiv.org/abs/2302.12949) | 已有参数化热算子与工业求解器对照，排除“首次热场算子”的叙事 |
| R5 | Yu et al. *DeepOHeat-v1: Efficient Operator Learning for Fast and Trustworthy Thermal Simulation and Optimization in 3D-IC Design*. IEEE TCPMT, 2025. [作者公开论文](https://web.ece.ucsb.edu/~zhengzhang/journals/2025-TCPMT-DeepOHeat-v1.pdf) | 多尺度表示、可分离训练和可信混合优化；比较需包括标签与设计成本 |
| R6 | Huang et al. *Therm-FM: Foundation Model is ALL YOU NEED for 3D-ICs Thermal Simulation*. arXiv:2605.22663v2，DAC 2026 论文扩展版. [论文](https://arxiv.org/abs/2605.22663v2) | 预训练、多保真和少样本跨芯片适配，提高了应用证据门槛 |
| R7 | Yu et al. *DeepOHeat-v2: Self-Improving Operator Learning for Fast and Trustworthy Thermal Optimization in 3D-IC Design*. arXiv:2608.16080v1，2026-08-17 预印本. [论文](https://arxiv.org/abs/2608.16080v1) | 高对比界面、离散物理损失与热点门控；不能写成已确认顶会论文 |
| R8 | University of Virginia. *HotSpot Temperature Modeling Tool*. [官方工具页](https://lava.cs.virginia.edu/hotspot/) | 传统快速热建模与求解器优化参照 |
| R9 | EPFL ESL. *Thermal modelling / 3D-ICE*. [官方项目说明](https://www.epfl.ch/labs/esl/research/thermal-modelling/) | 封装、chiplet 与冷却应用参照；也提示纯平均场误差之外的工程需求 |
| R10 | DAC. *Research Manuscript Submissions*. [官方说明](https://dac.com/2026/research-manuscript-submissions) | 用于判断投稿题型，不将往年要求当未来截稿规则 |
| R11 | ISSCC. *Call for Papers Overview*. [官方说明](https://www.isscc.org/paper-submission-26) | 用于区分算法研究与电路实测主线，不构成投稿日期建议 |

文献机制和发表身份在前一轮审阅中通过论文/官方来源核验；不同论文的 speedup 和 error 由于问题、网格、硬件与预算不同，没有直接拿来与 Heat3D 数值排名。

## 10. 附录 C：仓库证据索引

以下 GitHub 链接固定到已核验 commit，避免分支后续变化使报告失去版本边界。

| 编号 | 文件与用途 |
| --- | --- |
| E1 | [G1 publication summary](https://github.com/133754144/heat3d-ic/blob/365576daa566e0eb786eba93486910dea754cb1a/docs/v7_g1_publication_summary.md)：正式效应、CI 与主张边界 |
| E2 | [G1 training results](https://github.com/133754144/heat3d-ic/blob/365576daa566e0eb786eba93486910dea754cb1a/docs/v7_g1_training_results.md)：21-run 结果、native 与 common-domain H2 |
| E3 | [P23 native sparse table](https://github.com/133754144/heat3d-ic/blob/d9d961aa6970f60a45dea7995f1b82dab2e335d8/docs/v7_g2_p23_table_a_native_sparse.md)：V6、GINO、Transolver |
| E4 | [P18 common-valid comparison](https://github.com/133754144/heat3d-ic/blob/d9d961aa6970f60a45dea7995f1b82dab2e335d8/docs/v7_g2_p18_common_valid_comparison.md)：DeepOHeat 外部域 |
| E5 | [P19 readiness matrix](https://github.com/133754144/heat3d-ic/blob/d9d961aa6970f60a45dea7995f1b82dab2e335d8/docs/v7_g2_p19_readiness_matrix.md)：推理测量比较边界；后续状态以对应新记录为准 |
| E6 | [P22 Therm-FM aggregation](https://github.com/133754144/heat3d-ic/blob/d9d961aa6970f60a45dea7995f1b82dab2e335d8/docs/v7_g2_p22_thermfm_formal_aggregation.md)：预训练基线训练与参数信息；主表使用 L1 统一 evaluator 指标 |
| E7 | [V6 publication evidence](https://github.com/133754144/heat3d-ic/blob/9cb6b374cf2b6f9dde8e5f06da078dee169d9b43/docs/v6_p1i_publication_evidence_summary.md)：V6 valid/test、配对计时与硬件边界 |
| E8 | [V6 peak tail closeout](https://github.com/133754144/heat3d-ic/blob/9cb6b374cf2b6f9dde8e5f06da078dee169d9b43/docs/v6_p1i_error_tail_closeout.md)：峰值误差尾部 |
| E9 | [E/U contract](https://github.com/133754144/heat3d-ic/blob/d9d961aa6970f60a45dea7995f1b82dab2e335d8/configs/heat3d_v6_p1i/v7_g0b2c_eu_contract_manifest.json)：锚点、输入、查询和重建的不同角色 |
| E10 | [P23 test-access incident](https://github.com/133754144/heat3d-ic/blob/d9d961aa6970f60a45dea7995f1b82dab2e335d8/docs/v7_g2_p23_test_access_incident.md)：访问事件事实及边界 |
| E11 | [P23 evidence salvage closeout](https://github.com/133754144/heat3d-ic/blob/d9d961aa6970f60a45dea7995f1b82dab2e335d8/docs/v7_g2_p23_evidence_salvage_closeout.md)：V6 二进制丢失与旧 aggregate 的可用范围；L1 是另一个 cohort |
| E12 | [DeepOHeat solver fidelity receipt](https://github.com/133754144/heat3d-ic/blob/d9d961aa6970f60a45dea7995f1b82dab2e335d8/docs/v7_g2_p4_deepoheat_v1_solver_fidelity_receipt.json)：官方离散系统核验及所用案例身份 |
| C1 | [p1i training adapter](https://github.com/133754144/heat3d-ic/blob/d9d961aa6970f60a45dea7995f1b82dab2e335d8/rigno/heat3d_training/p1i.py#L596)：vanilla/Full 调用、输出与损失差异 |
| C2 | [common evaluator](https://github.com/133754144/heat3d-ic/blob/d9d961aa6970f60a45dea7995f1b82dab2e335d8/scripts/evaluate_v7_g2_p23_common_fullfield.py)：温度、CV、case、seed 与真值读取约束 |
| C3 | [paired bootstrap](https://github.com/133754144/heat3d-ic/blob/d9d961aa6970f60a45dea7995f1b82dab2e335d8/scripts/analyze_v7_g2_p23_paired_bootstrap.py)：案例配对和统计完整性检查 |
| C4 | [V7 CI workflow](https://github.com/133754144/heat3d-ic/blob/d9d961aa6970f60a45dea7995f1b82dab2e335d8/.github/workflows/v7-lightweight.yml)：已有测试与尚未覆盖的活跃路径 |
| C5 | [full-field reconstruction](https://github.com/133754144/heat3d-ic/blob/d9d961aa6970f60a45dea7995f1b82dab2e335d8/rigno/heat3d_v6_full_field.py#L100)：分层邻域加权与 partition-of-unity |
| C6 | [temperature contract](https://github.com/133754144/heat3d-ic/blob/d9d961aa6970f60a45dea7995f1b82dab2e335d8/rigno/heat3d_runtime/temperature.py)：可复用的显式温度语义 |

**L1 本地证据定位：** commit `2960ab52ba90cea3ac4d321d36333835e3271808`，仓库内路径 `docs/v7_g2_p23_g1_e200_direct240825_same_domain_closure.md`。原审阅工作树为 `/Users/xuyihua/.codex/worktrees/9fc3/3D IC Heat`。统一结果路径为 `docs/results/v7_g2_p23_g1_e200_u_v2_direct240825_vs_thermfm_common_eval.json`，配套 bootstrap、domain identity 和 reproduction gate 文件位于同目录。该本机路径仅用于追溯，不是可移植的发布接口；在公开归档前，报告不提供假定可访问的 GitHub 链接。

## 11. 本次交付范围

本次只新增本 Markdown 审稿报告，复核关键文档数字、引用路径和证据身份；未修改研究代码、模型配置、冻结证据或数据。未启动训练、推理、求解、数据生成或新的测试集访问。附录 A 是后续工作清单，不是已经完成的实验，也不改变原有实验授权边界。
