# 3D IC 热仿真 AI 文献矩阵

更新时间：2026-04-13

## 1. 任务聚焦

核心问题：AI 在 3D IC / chiplet 热仿真流程中到底替代、增强或重构了哪些环节？

当前归纳为五类作用：

1. 学习热传导算子，直接替代大量 PDE/FEM/FDM 求解。
2. 用低成本代理模型支撑布局、热点、TTSV、微通道等优化。
3. 用多保真、主动学习、迁移学习降低高保真样本成本。
4. 用物理约束与可信度评估提升可部署性。
5. 用 AI 改善仿真链条的上游/下游环节，如网格生成、峰值温度估计、热感知 floorplanning。

## 2. 核心论文与要点

### A. 神经算子/代理求解器主线

#### DeepOHeat, DAC 2023

- 作用：把输入功率图、边界条件、换热系数等 PDE 配置映射到 3D 温度场。
- 方法：physics-aware DeepONet / operator learning。
- 关键证据：论文摘要称对未见测试样例可实现 `1000x~300000x` 加速。
- 意义：首次把“解参数化热方程族”而不是“拟合固定设计集”作为 3D IC 热仿真目标。
- 局限：训练成本高，复杂多尺度热点与可信性问题尚未解决。

#### Enhanced Operator Learning / DeepOHeat+, ASPDAC 2025

- 作用：降低 DeepOHeat 的训练与显存成本。
- 方法：Separable Operator Network (SepONet) 分解 trunk。
- 关键证据：摘要给出 `64x` 训练时间降低、`31x` GPU 显存降低，精度与 DeepOHeat 相当。
- 意义：说明 AI 在该领域不仅用于“加速推理”，也开始针对“可训练性/可扩展性”做架构设计。

#### Self-Attention to Operator Learning-based 3D-IC Thermal Simulation

- 作用：改善 FNO 类模型对局部高频热点与长程耦合的建模能力。
- 方法：SAU-FNO，将 self-attention + U-Net + FNO 结合，并使用 transfer learning。
- 关键证据：摘要给出较 FEM `842x` 加速。
- 意义：AI 在此承担“多尺度特征提取器”角色，重点解决热点、界面附近高梯度误差。

#### ARO, 2025

- 作用：统一稳态/瞬态热预测，并面向 unseen circuits 提高可迁移性。
- 方法：autoregressive operator learning + multi-fidelity fusion + active learning。
- 关键证据：摘要给出相对 MTA `约1000x` 加速；主动学习相比伪随机策略至少减少 `25%` 数据。
- 意义：AI 从单一代理求解器升级为“数据采样策略 + 多保真融合器 + 可迁移热分析器”。

#### DeepOHeat-v1, IEEE TCPMT 2026

- 作用：提高算子学习热仿真的可信度与优化闭环能力。
- 方法：KAN trunk、可分离训练、置信度评分、与 FD/GMRES 的混合增量修正。
- 关键证据：
  - 两类案例误差分别降低 `1.25x` 与 `6.29x`；
  - 基线案例训练 `62x` 加速、显存 `31x` 降低；
  - 整体优化流程相对高保真 FD 加速 `70.6x`。
- 意义：AI 在这里不再只是“快”，而是承担“可信代理 + 优化前筛选 + 求解器协同”的角色。

#### PI-ONet, IEEE TCPMT 2026

- 作用：面向 multilayer chiplets 做高效热分析。
- 方法：physics-informed operator network + bump/TSV 插层等效热导 ANN 子模型。
- 关键证据：摘要给出相对 Icepak `最高 35000x` 加速。
- 意义：说明 chiplet/先进封装热分析已从规则 3D IC 扩展到更复杂异构层间结构。

### B. 瞬态/紧凑模型/物理代理模型

#### Efficient ML-Based Transient Thermal Prediction for 3D-ICs, ASPDAC 2025

- 作用：快速预测瞬态热轨迹。
- 方法：基于特征工程的 ML（初始三步与后续时间步分模型；space-windowed + time-decayed features）。
- 关键证据：MAE `1.12 °C`，最大误差 `7.27 °C`，较商业工具 `116x` 加速。
- 意义：AI 在这里充当“时间序列温度演化近似器”，适合热安全校验和迭代分析。

#### Fast Machine Learning Based Prediction for Temperature Simulation Using Compact Models, DATE 2025

- 作用：与 CTM 紧耦合，减少传统求解器调用。
- 方法：线性回归 + windowing，与 compact thermal model 直接对接。
- 关键证据：较 PACT `最高 70x` 加速；仅用少量样本即可维持 3D 架构中 `MAE 不高于 1 °C`。
- 意义：说明“轻量模型 + 物理先验”在 EDA 工具链中可能比纯深网更实用。

#### Fast Thermal Modeling of TTSV via Bounded Neural Networks

- 作用：预测 TTSV 各向异性等效热导率，服务更大热仿真模型。
- 方法：解析 Series-Parallel Thermal Conductivity 模型 + bounded BP network。
- 关键证据：相对 FEM 基准，水平热导率预测 MAPE `<= 0.35%`。
- 意义：AI 在这里扮演“物理参数提取器”，不是直接输出温度场，而是补足多尺度建模短板。

### C. 设计优化与热管理决策

#### Rapid heat source layout optimization ... ANN ROM + Bayesian optimization, Heat Transfer 2024

- 作用：加速热源布局、TSV 参数优化。
- 方法：ANN reduced-order model + Bayesian optimization。
- 关键证据：温度预测偏差 `< 2%`，`R^2≈0.93`，BO 在 250 次迭代内约 `4.07 s` 找到全局最优。
- 意义：AI 在这里作为“优化器的低成本温度 oracle”。

#### Optimizing Heat Source Arrangement for 3D ICs with irregular structures using machine learning methods, ICEPT 2023

- 作用：在不规则结构 3D IC 中优化多热源位置。
- 方法：ANN + genetic algorithm。
- 关键证据：R 值 `> 0.97` / `> 0.98`。
- 意义：把复杂 COMSOL 参数扫频压缩为可快速迭代的代理优化问题。

#### Learning Peak Temperature in 3DICs by Deep Differentiable Forest, ISEDA 2024

- 作用：快速预测峰值温度。
- 方法：deep differentiable forest。
- 关键证据：相对 FEM 单样例 `>200x` 加速；相对误差 `0.11%`，优于 XGBoost 与随机森林。
- 意义：当研究目标是热点/峰值风险筛查时，AI 不必输出全场，也能显著提高流程效率。

#### Thermal-Aware Fixed-Outline 3-D IC Floorplanning, IEEE TVLSI 2023

- 作用：把热约束嵌入 3D floorplanning。
- 方法：deep k-means + GCN + MADRL。
- 关键证据：论文摘要称在线长与温度优化上优于 SOTA 启发式方法。
- 意义：AI 在此是“决策搜索器”，热仿真成为奖励/约束而不是终点。

#### Multiobjective Deep Reinforcement Learning Driven Collaborative Optimization of TSV-Based Microchannel and PDN for 3-D ICs, IEEE TCPMT 2026

- 作用：联合优化 TSV 微通道散热与 PDN。
- 方法：multiobjective deep RL + CFD 环境。
- 关键证据：最高温度降低 `3.3%`，压降降低 `17.2%`，较 SDRL/GA 收敛更快。
- 意义：AI 正在从“温度预测”外延到“热-液-电多目标设计”。

### D. 仿真流程辅助

#### NeuralMesh: Neural Network For FEM Mesh Generation in 2.5D/3D Chiplet Thermal Simulation

- 作用：加速 chiplet 热仿真的网格生成与分层建模前处理。
- 方法：深度学习辅助关键区域识别、网格/分层构建、温度预测协同。
- 意义：AI 的作用不只在 PDE 近似，还在减少 FEM 前处理瓶颈。

## 3. 当前阶段判断

### 3.1 现有发展

- 第一阶段：AI 主要作为简单回归器或 ROM，用于热点、峰值温度或少量参数优化。
- 第二阶段：AI 进入完整温度场预测，开始替代部分稳态/瞬态求解。
- 第三阶段：引入神经算子、多保真、主动学习、迁移学习、物理约束，目标从“快”升级为“可泛化、可信、可嵌入设计闭环”。
- 第四阶段：扩展到 chiplet、先进封装、微通道、PDN、网格生成等更真实的协同设计场景。

### 3.2 AI 的典型作用列表

1. `Surrogate solver`：直接输出稳态/瞬态温度场。
2. `Operator learner`：学习输入函数到解函数的连续映射，提升跨分辨率泛化。
3. `Physics regularizer`：用 PDE/边界条件约束训练，降低对标注数据依赖。
4. `Multi-fidelity fusion engine`：融合低保真与高保真样本，降低数据成本。
5. `Active sampler`：挑选最有价值的高保真样本。
6. `Design optimizer`：与 BO、GA、RL 结合搜索热友好设计空间。
7. `Parameter extractor`：学习等效热导率、界面参数、紧凑模型参数。
8. `Workflow accelerator`：辅助网格生成、floorplanning、峰值温度预警。

### 3.3 领域瓶颈

- 数据稀缺：高保真 FEM/CFD 标签昂贵。
- 几何泛化差：很多方法隐含规则网格或固定 floorplan。
- 物理可信度不足：局部热点、尖锐梯度、外推工况容易失真。
- 工具链耦合弱：不少工作仍停留在独立 benchmark，而非可落地 EDA workflow。
- 多物理场不足：热-电-流-力联动场景仍以优化个例为主，缺少统一框架。

## 4. HTML 报告建议结构

1. 研究背景与问题定义
2. AI 在 3D IC 热仿真中的发展脉络
3. 按方法类别综述并列表说明“AI 发挥了什么作用”
4. 代表性工作对比表
5. 对未来方向的判断
6. 对选题的启示：为什么值得研究 RIGNO / 图神经算子 / 迁移学习

## 5. 待补充的网络文献

- Fast thermal analysis for chiplet design based on graph convolution networks
- A thermal machine learning solver for chip simulation
- 更偏 chiplet/2.5D/3D 的 2024-2026 新工作，重点找：
  - graph-based thermal surrogate
  - transient chiplet thermal analysis with message passing
  - POD + ANN for 2.5D chiplet thermal modeling
