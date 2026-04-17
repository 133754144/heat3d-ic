# 3D IC 热仿真 AI 文献证据整理

更新时间：2026-04-13

## 1. 综述主线

当前可稳定支撑的核心判断：

1. AI 在 3D IC / chiplet 热仿真中的作用，已经从早期的单指标回归器，扩展到完整温度场求解、瞬态预测、设计优化、参数提取和流程加速。
2. 2023-2026 年最强主线是 `operator learning / neural operator`，代表工作包括 DeepOHeat、ARO、SAU-FNO、DeepOHeat-v1、PI-ONet。
3. 当前研究的演进方向非常清晰：
   - 从规则结构到 chiplet / advanced packaging；
   - 从单保真监督学习到多保真、主动学习、迁移学习；
   - 从“快”到“可信、可泛化、可嵌入 EDA 流程”。
4. 对未来研究最重要的缺口是：复杂几何泛化、跨分辨率/跨结构迁移、物理可信度、与真实 EDA workflow 的耦合。

## 2. 已核实的本地核心文献

### A. 神经算子 / 代理求解器

#### DeepOHeat, DAC 2023

- 文件：`文献/DeepOHeat_Operator_Learning-based_Ultra-fast_Thermal_Simulation_in_3D-IC_Design.pdf`
- DOI：`10.1109/DAC56929.2023.10247998`
- AI 角色：`operator learner` + `physics-informed surrogate solver`
- 关键证据：
  - 摘要明确指出：对未见测试样例可达到 `1000x~300000x` 加速。
  - 论文给出的案例中，MAPE / peak error 维持在很低水平，目标是学习功率图、边界条件、换热系数到 3D 温度场的函数映射。
- 可支撑观点：
  - 3D IC 热仿真中的 AI 已不再只做“参数拟合”，而是直接学习参数化热方程族的解算子。

#### Enhanced Operator Learning for Scalable and Ultra-fast Thermal Simulation in 3D-IC Design, ASP-DAC 2025

- 文件：`文献/Enhanced Operator Learning for Scalable and Ultra-fast Thermal Simulation in 3D-IC Design.pdf`
- DOI：`10.1145/3658617.3703318`
- AI 角色：`scalable operator learning architecture`
- 关键证据：
  - 采用 SepONet/可分离 trunk；
  - 训练时间降低 `64x`；
  - GPU 显存降低 `31x`；
  - 相对 `l2` 误差约 `1.65%`，与 DeepOHeat 相当。
- 可支撑观点：
  - 该方向开始关注“训练可扩展性”和“高分辨率可训练性”，而不仅是推理速度。

#### Self-Attention to Operator Learning-based 3D-IC Thermal Simulation, DAC 2025

- 文件：`文献/Self-Attention_to_Operator_Learning-based_3D-IC_Thermal_Simulation.pdf`
- DOI：`10.1109/DAC63849.2025.11132988`
- AI 角色：`multi-scale feature extractor` + `transfer learning enhanced operator model`
- 关键证据：
  - 结合 self-attention、U-Net 与 FNO 的 SAU-FNO；
  - 相比传统 FEM，推理加速 `842x`；
  - 文中写明相较其他方法，`MSE reduced by over 50% compared to FNO`；
  - 使用 transfer learning 微调低保真/高保真数据。
- 可支撑观点：
  - 热仿真 AI 已开始系统解决局部热点与高频特征损失问题，并主动吸收迁移学习思路降低高保真数据需求。

#### ARO: Autoregressive Operator Learning for Transferable and Multi-fidelity 3D-IC Thermal Analysis with Active Learning, ICCAD 2024

- 文件：`文献/ARO Autoregressive Operator Learning for Transferable and Multi-fidelity 3D-IC Thermal Analysis with Active Learning.pdf`
- DOI：`10.1145/3676536.3676713`
- AI 角色：`autoregressive operator learner` + `multi-fidelity fusion engine` + `active sampler`
- 关键证据：
  - 摘要给出：相对 MTA 约 `1000x` 加速；
  - 主动学习至少减少 `25%` 数据量；
  - 文中还给出稳态平均误差低于 `0.4 K`，瞬态九个时刻平均误差较低。
- 可支撑观点：
  - 这一阶段的 AI 不仅预测温度场，还参与高保真样本选择和跨保真数据整合。

#### DeepOHeat-v1, IEEE TCPMT 2026

- 文件：`文献/DeepOHeat-v1_Efficient_Operator_Learning_for_Fast_and_Trustworthy_Thermal_Simulation_and_Optimization_in_3D-IC_Design.pdf`
- DOI：`10.1109/TCPMT.2025.3619906`
- AI 角色：`trustworthy surrogate` + `hybrid solver initializer`
- 关键证据：
  - KAN trunk 使两类案例误差分别降低 `1.25x` 和 `6.29x`；
  - 可分离训练使训练速度提升 `62x`、显存降低 `31x`；
  - 引入 confidence score；
  - 与 FD/GMRES 混合优化流程整体加速 `70.6x`。
- 可支撑观点：
  - 可信度评估与混合求解器协同，正成为 AI 热仿真从“研究 demo”走向“设计闭环”的关键步骤。

#### PI-ONet, IEEE TCPMT 2026

- 文件：`文献/PI-ONet_A_Physics-Informed_Operator_Network_for_Efficient_Thermal_Analysis_of_Multilayer_Chiplets.pdf`
- DOI：`10.1109/TCPMT.2025.3621061`
- AI 角色：`physics-informed operator network` + `equivalent-property extractor`
- 关键证据：
  - 针对 multilayer chiplets；
  - 通过 ANN 预测 bump/TSV 插层等效热导率；
  - 相对 Icepak 热分析加速最高 `35000x`；
  - 单芯片案例平均误差约 `0.224 °C`、最大误差 `0.501 °C`。
- 可支撑观点：
  - AI 已从规则 3D IC 走向复杂异构 chiplet 热分析，并与等效建模结合。

### B. 瞬态预测 / 轻量代理 / 参数提取

#### Efficient ML-Based Transient Thermal Prediction for 3D-ICs, ASP-DAC 2025

- 文件：`文献/Efficient ML-Based Transient Thermal Prediction for 3D-ICs.pdf`
- DOI：`10.1145/3658617.3697754`
- AI 角色：`transient thermal predictor`
- 关键证据：
  - MAE `1.12 °C`；
  - 最大误差 `7.27 °C`；
  - 相对商用工具预测阶段加速 `116x`；
  - 采用初始三步/后续时间步双模型，结合 space-windowed 与 time-decayed features。
- 可支撑观点：
  - 对很多工程场景而言，轻量特征工程 + ML 的瞬态预测仍具有很强实用性。

#### Fast Machine Learning Based Prediction for Temperature Simulation Using Compact Models, DATE 2025

- 文件：`文献/Fast_Machine_Learning_Based_Prediction_for_Temperature_Simulation_Using_Compact_Models.pdf`
- AI 角色：`compact-model-aware lightweight predictor`
- 关键证据：
  - 相对 PACT `up to 70x speedup`；
  - 仅用少量训练样本，3D 三层结构推理阶段 `MAE not higher than 1 °C`；
  - 方法核心是借助稳态热传导线性结构，以线性回归近似 CTM 中的逆导热矩阵。
- 可支撑观点：
  - “物理先验 + 轻量 ML”仍然是可部署性很强的一条路线。

#### Fast Thermal Modeling of TTSV via Bounded Neural Networks, ICEPT 2025

- 文件：`文献/Fast_Thermal_Modeling_of_TTSV_via_Bounded_Neural_Networks.pdf`
- DOI：`10.1109/ICEPT67137.2025.11157316`
- AI 角色：`parameter extractor`
- 关键证据：
  - 将解析 Series-Parallel Thermal Conductivity 模型与有界 BP 网络结合；
  - 相对 FEM 基准，水平等效热导率 MAPE `<= 0.35%`。
- 可支撑观点：
  - AI 在热仿真里并不总是直接输出温度场，也可以用于抽取等效热参数，服务上层更大模型。

### C. 优化与设计决策

#### Rapid heat source layout optimization ... ANN ROM + BO, Heat Transfer 2024

- 文件：`文献/Rapid heat source layout optimization in three-dimensional integrated circuits using artificial neural network reduced-order model in combination with Bayesian optimization.pdf`
- DOI：`10.1002/htj.23095`
- AI 角色：`optimization oracle`
- 关键证据：
  - ANN reduced-order model 预测偏差 `< 2%`；
  - `R^2 ≈ 0.93`；
  - BO 在 `250` 次迭代中 `4.07 s` 找到全局最优。
- 可支撑观点：
  - 代理模型最直接的价值之一，是把高维热设计空间搜索变成可实时迭代的问题。

#### Learning Peak Temperature in 3DICs by Deep Differentiable Forest, ISEDA 2024

- 文件：`文献/Learning_Peak_Temperature_in_3DICs_by_Deep_Differentiable_Forest.pdf`
- DOI：`10.1109/ISEDA62518.2024.10617838`
- AI 角色：`peak-temperature estimator`
- 关键证据：
  - 相对误差 `0.11%`；
  - 每例预测时间低于 1 s；
  - 相对 FEM `> 200x` 加速。
- 可支撑观点：
  - 如果设计目标是热点筛查，单指标学习也仍然很有价值。

#### Thermal-Aware Fixed-Outline 3-D IC Floorplanning, IEEE TVLSI 2023

- 文件：`文献/Thermal-Aware_Fixed-Outline_3-D_IC_Floorplanning_An_End-to-End_Learning-Based_Approach.pdf`
- DOI：`10.1109/TVLSI.2023.3321532`
- AI 角色：`design optimizer`
- 关键证据：
  - deep k-means 用于 tier assignment；
  - GCN + MADRL + attention 用于模块与 TSV 位置优化；
  - 摘要明确指出优于 SOTA 启发式方法；
  - 文中对 baseline 报告了温度、线长、TSV 数与运行时间的显著改进。
- 可支撑观点：
  - AI 正在把热仿真嵌入 floorplanning / placement 等 EDA 决策流程，而非仅做后验分析。

#### Optimizing Heat Source Arrangement for 3D ICs with irregular structures using machine learning methods, ICEPT 2023

- 文件：`文献/Optimizing_Heat_Source_Arrangement_for_3D_ICs_with_irregular_structures_using_machine_learning_methods.pdf`
- DOI：`10.1109/ICEPT59018.2023.10492080`
- AI 角色：`surrogate-assisted design optimization`
- 关键证据：
  - 使用 ANN + GA 优化 6 个热源位置；
  - `R value > 0.97 / > 0.98`；
  - 最大温度偏差仅 `0.24%`。
- 可支撑观点：
  - 在不规则结构热设计中，AI 适合作为有限元的代理模型，而不是替代全部物理建模。

#### Multiobjective Deep Reinforcement Learning Driven Collaborative Optimization of TSV-Based Microchannel and PDN for 3-D ICs, IEEE TCPMT 2026

- 文件：`文献/Multiobjective_Deep_Reinforcement_Learning_Driven_Collaborative_Optimization_of_TSV-Based_Microchannel_and_PDN_for_3-D_ICs.pdf`
- DOI：`10.1109/TCPMT.2025.3618021`
- AI 角色：`multi-objective optimizer`
- 关键证据：
  - 最高温度降低 `3.3%`；
  - 压降降低 `17.2%`；
  - 相比 SDRL 与 GA，收敛分别快 `57.1%` 和 `62.5%`。
- 可支撑观点：
  - 热仿真 AI 正向热-液-电协同优化扩展。

### D. 流程辅助与 advanced packaging

#### NeuralMesh, DAC 2025

- 文件：`文献/NeuralMesh_Neural_Network_For_FEM_Mesh_Generation_in_2.5D_3D_Chiplet_Thermal_Simulation.pdf`
- DOI：`10.1109/DAC63849.2025.11132601`
- AI 角色：`workflow accelerator`
- 关键证据：
  - Mesh 生成加速 `up to 45x`；
  - 热预测误差控制在 `0.8%` 以内；
  - 通过温度预测 + 几何分析指导网格优化。
- 可支撑观点：
  - AI 能有效压缩 FEM 前处理瓶颈，这对真实设计流程很重要。

#### Intelligent Design Method of Thermal Through Silicon Via for Thermal Management of Chiplet-Based System, IEEE TED 2023

- 文件：`文献/Intelligent_Design_Method_of_Thermal_Through_Silicon_via_for_Thermal_Management_of_Chiplet-Based_System.pdf`
- DOI：`10.1109/TED.2023.3302828`
- AI 角色：`BP-NN based inverse designer`
- 关键证据：
  - FEM + BP-NN + PSO 联合优化 TTSV 参数；
  - 三层温度目标与仿真结果吻合良好，如 `340.81/334.76/314.85 K` 对应目标 `340/335/315 K`。
- 可支撑观点：
  - AI 可作为结构参数反设计模块嵌入 chiplet 热管理。

## 3. 外部补充文献（联网核实）

### Fast thermal analysis for chiplet design based on graph convolution networks, ASP-DAC 2022

- 来源：NSF PAR 页面
- 链接：<https://par.nsf.gov/biblio/10393029-fast-thermal-analysis-chiplet-design-based-graph-convolution-networks>
- AI 角色：`graph-based chiplet thermal surrogate`
- 关键证据：
  - 平均 RMSE `0.31 K`；
  - 相对 HotSpot + SuperLU 加速 `2.6x`；
  - 对 6 个未见数据集无需重训，平均 RMSE `< 0.67 K`。

### A Thermal Machine Learning Solver For Chip Simulation, MLCAD 2022

- 来源：arXiv
- 链接：<https://arxiv.org/abs/2209.04741>
- DOI：`10.48550/arXiv.2209.04741`
- AI 角色：`general ML thermal solver`
- 关键证据：
  - 该工作把 CoAEMLSim 扩展到 constant / distributed HTC；
  - 明确以商用 FEM/CFD 过慢为出发点；
  - 摘要强调其相对商用求解器与 UNet 基线具有更好的 accuracy、scalability、generalizability。

### T-Fusion: Thermal Prediction of 3D ICs with Multi-fidelity Fusion, ASP-DAC 2025

- 来源：ASP-DAC 2025 technical program
- 链接：<https://www.aspdac.com/aspdac2025/archive/program/program_abst.html>
- AI 角色：`multi-fidelity fusion model`
- 关键证据：
  - 使用 tensor arithmetic + Bayesian autoregression；
  - 报告相对 COMSOL / MTA / HotSpot `10,000x~1,000,000x` 加速；
  - 瞬态预测仅需 `20` 组高精度和 `64` 组低精度数据即可将误差控制在 `1 K` 内。

### Adaptive Graph Learning for Efficient Thermal Analysis of Multi-Stacking Chiplet Systems under Interface Variations, ICCAD 2025

- 来源：Arizona State University publication page
- 链接：<https://asu.elsevierpure.com/en/publications/adaptive-graph-learning-for-efficient-thermal-analysis-of-multi-s>
- DOI：`10.1109/ICCAD66269.2025.11240700`
- AI 角色：`adaptive graph-based thermal framework`
- 关键证据：
  - 平均 MAPE `0.05%`；
  - 仿真时间为 `few hundred milliseconds`；
  - 相对传统稳态 FDM `>1000x` 加速；
  - 对界面变化和材料变化具有无需重训的适应性。

### 可作为趋势补充但暂不写定量结论的外部题目

- `Transient Thermal Analysis of Chiplet-Based Systems with Dual-Channel Message Passing Neural Networks`, ACES-China 2024
- `POD-ANN Thermal Modelling Framework for Rapid Thermal Analysis of 2.5D Chiplet Designs`, EPTC 2024

说明：已确认题目存在，但当前未获得足够摘要细节，适合在综述中作为“图消息传递/POD-ANN 正在出现”的趋势性补充，不写具体数值。

## 4. 方法学支撑文献

### Neural operators for accelerating scientific simulations and design, Nat. Rev. Phys. 2024

- 文件：`文献/Neural operators for accelerating scientific simulations and design.pdf`
- DOI：`10.1038/s42254-024-00712-5`
- 可支撑观点：
  - 神经算子学习的是函数到函数映射；
  - 具备在连续域上泛化、零样本超分辨/超评估的潜力；
  - 在科学仿真中可带来 `four to five orders of magnitude` 加速。

### Geometry-Informed Neural Operator for Large-Scale 3D PDEs, NeurIPS 2023

- 文件：`文献/Geometry-informed neural operator for large-scale 3D PDEs.pdf`
- 可支撑观点：
  - GINO 使用 point cloud + SDF + graph/Fourier operator 处理变化几何；
  - 对复杂 3D PDE 具有离散无关性和较强几何泛化；
  - 是“从规则网格走向任意域 / 复杂几何”的关键方法学依据。

### Physics-Informed Neural Operators with Exact Differentiation on Arbitrary Geometries, NeurIPS Workshop 2023

- 文件：`文献/12_Physics_Informed_Neural_Ope.pdf`
- 可支撑观点：
  - PINO 可在数据不足或分辨率不足时加入 PDE residual；
  - 进一步强调 arbitrary geometries 与 exact derivatives 的重要性。

## 5. 当前最稳妥的叙述框架

### 5.1 AI 在热仿真中的八类作用

1. `Surrogate solver`：直接预测稳态/瞬态温度场。
2. `Operator learner`：学习功率图/边界条件到温度场的连续映射。
3. `Physics regularizer`：将 PDE、边界条件、守恒约束纳入训练。
4. `Multi-fidelity fusion engine`：融合低保真与高保真样本。
5. `Active sampler`：主动选择最有价值的高保真数据。
6. `Design optimizer`：与 BO、GA、PSO、RL、floorplanning 联动。
7. `Parameter extractor`：提取等效热导率、紧凑模型参数、TTSV 参数。
8. `Workflow accelerator`：加速网格生成、前处理、热感知 EDA 决策。

### 5.2 当前领域的主要趋势

1. 从“局部指标预测”走向“完整温度场求解”。
2. 从“监督回归”走向“神经算子 / 多保真 / 主动学习 / 迁移学习 / 物理约束”。
3. 从“规则 3D IC”走向“chiplet / 2.5D / 3D advanced packaging / interface variations”。
4. 从“离线加速”走向“设计优化闭环与 workflow integration”。

### 5.3 当前瓶颈

1. 高保真标注数据昂贵。
2. 复杂几何与跨分辨率泛化仍不足。
3. 局部热点和尖锐高频特征难学。
4. 物理可信度、误差评估和不确定性控制仍然薄弱。
5. 很多方法尚未与真正的 EDA 工具链深度耦合。

## 6. 对 HTML 报告的落脚建议

建议在新 HTML 中把“AI 发挥的作用”做成表格，至少包含：

- 论文
- 年份/场景
- 方法
- AI 具体承担的角色
- 可量化收益
- 局限 / 对未来研究的启示

可重点突出：

- DeepOHeat → 把问题从单样本回归升级为热方程族算子学习；
- ARO / T-Fusion → 多保真与主动学习正在成为降低数据成本的关键；
- DeepOHeat-v1 / PI-ONet → 可信性与物理一致性开始被正面处理；
- GCN / Adaptive Graph Learning / NeuralMesh → 复杂 chiplet 几何与流程加速是下一阶段的重要落点；
- Neural operator / GINO / PINO → 为后续任意域图神经算子研究提供方法学基础。
