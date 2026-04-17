# 外部补充文献与链接

更新时间：2026-04-14

## 已确认可用的外部来源

### 1. Fast thermal analysis for chiplet design based on graph convolution networks

- 类型：ASP-DAC 2022
- DOI：10.1109/ASP-DAC52403.2022.9712583
- 链接：
  - https://ieeexplore.ieee.org/document/9712583
  - https://par.nsf.gov/biblio/10393029-fast-thermal-analysis-chiplet-design-based-graph-convolution-networks
- 可用摘要要点：
  - 使用 GCN 预测 2.5D chiplet 系统热图。
  - 引入总功率全局特征、skip connection、edge attention 和 PNA 聚合器。
  - 平均 RMSE 为 0.31 K。
  - 相对 HotSpot + SuperLU 加速 2.6x。
  - 对六个未见数据集无需重训，平均 RMSE 小于 0.67 K。

### 2. A Thermal Machine Learning Solver For Chip Simulation

- 类型：MLCAD 2022
- DOI：10.1145/3551901.3556484
- arXiv：2209.04741
- 链接：
  - https://ieeexplore.ieee.org/document/9900086
  - https://dblp.org/rec/conf/mlcad/RanadeHPCKW22
  - https://arxiv.org/abs/2209.04741
- 可用摘要要点：
  - 基于 CoAEMLSim 扩展，处理常数和分布式 HTC。
  - 目标是替代慢速 FEM/CFD 片上热分析。
  - 重点强调精度、可扩展性和泛化性优于 UNet 基线。

### 3. Transient Thermal Analysis of Chiplet-Based Systems with Dual-Channel Message Passing Neural Networks

- 类型：ACES-China 2024
- 链接：
  - https://ieeexplore.ieee.org/document/10700022/
- 当前状态：
  - 已确认论文存在，但当前抓取工具未返回摘要。
  - 可在最终报告中作为“近年 chiplet 图模型趋势”的补充标题提及，避免对其未读取到的实验结论作过度展开。

### 4. POD-ANN Thermal Modelling Framework for Rapid Thermal Analysis of 2.5D Chiplet Designs

- 类型：EPTC 2024
- 链接：
  - https://ieeexplore.ieee.org/document/10909871
- 当前状态：
  - 已确认论文存在，但当前抓取工具未返回摘要。
  - 适合在报告中作为“POD + ANN 混合降阶建模”的近期代表标题补充。

### 5. T-Fusion: Thermal Prediction of 3D ICs with Multi-fidelity Fusion

- 类型：ASP-DAC 2025
- DOI：10.1145/3658617.3697749
- 链接：
  - https://www.aspdac.com/aspdac2025/archive/program/program_abst.html
  - https://dblp.org/rec/conf/aspdac/ZhangXZS25.html
- 可用摘要要点：
  - 采用 `tensor arithmetic + Bayesian autoregression` 的多保真融合框架。
  - 目标是在高保真样本稀缺时重建高保真热分布。
  - 对单核双层、四核三层与八核双层 3D IC 进行验证。
  - 相对 COMSOL、MTA 与 HotSpot 报告 `10,000x ~ 1,000,000x` 加速。
  - 在瞬态预测中，仅需 `20` 组高精度和 `64` 组低精度数据，即可把误差控制在 `1 K` 以内。

### 6. Adaptive Graph Learning for Efficient Thermal Analysis of Multi-Stacking Chiplet Systems under Interface Variations

- 类型：ICCAD 2025
- DOI：10.1109/ICCAD66269.2025.11240700
- 链接：
  - https://asu.elsevierpure.com/en/publications/adaptive-graph-learning-for-efficient-thermal-analysis-of-multi-s/
  - https://dblp.org/rec/conf/iccad/YangSCC25
- 可用摘要要点：
  - 面向 `2.5D/3D multi-stacking chiplet systems`，提出 `GNN + FEM hybrid` 热分析框架。
  - 通过附加可训练节点适配装配缺陷与界面变化导致的新温度分布。
  - 使用细粒度数值解与真实 post-silicon 热成像数据验证。
  - 报告平均 `MAPE = 0.05%`。
  - 单次热仿真耗时为几百毫秒，相对传统稳态 FDM 求解器加速 `>1000x`。
  - 对未见工艺和材料变化无需重训，仍具鲁棒适应性。

### 7. Thermal Management Challenges in 2.5D and 3D Chiplet Integration: A Review on Architecture-Cooling Co-Design

- 类型：综述，Eng 2025
- DOI：https://doi.org/10.3390/eng6120373
- 链接：
  - https://www.mdpi.com/2673-4117/6/12/373
- 可用摘要要点：
  - 聚焦 2.5D/3D chiplet 热管理挑战、封装级热建模、热界面材料与冷却协同设计。
  - 强调大功率 chiplet 系统中的热点、封装热扩散与冷却共设计问题。
  - 提出 `Thermal Feasibility Maps (TFMs)` 作为体系化比较框架。
- 适用方式：
  - 该文不是 AI 热仿真论文，但非常适合补强“为什么 chiplet 场景迫切需要高效热代理模型”的背景论证。

### 8. TDPNavigator-Placer: Thermal- and Wirelength-Aware Chiplet Placement in 2.5D Systems Through Multi-Agent Reinforcement Learning

- 类型：arXiv 预印本 / EPTC 2025 journal reference
- DOI：https://doi.org/10.48550/arXiv.2602.11187
- 链接：
  - https://arxiv.org/abs/2602.11187
- 可用摘要要点：
  - 面向 `2.5D` chiplet 系统放置问题，直接把热约束与线长目标建模为多智能体强化学习中的冲突目标。
  - 使用 `chiplet thermal design power (TDP)` 驱动热感知放置。
  - 强调现有方法常把多目标压成加权和，而该方法通过专门 agent 分工获得更好的 Pareto front。
- 适用方式：
  - 这篇工作更偏设计自动化而不是热求解器，但适合补入“AI 热分析结果正在更深地进入 chiplet 布局与系统设计决策”的未来方向部分。

### 9. 3D-ICE 4.0: Accurate and efficient thermal modeling for 2.5D/3D heterogeneous chiplet systems

- 类型：DATE 2026 / arXiv 2025 预印本
- 链接：
  - https://infoscience.epfl.ch/entities/publication/12080437-8e93-46c4-a15f-b6020057d1ab
  - https://chiplet-marketplace.com/article/3d-ice-4-0-accurate-and-efficient-thermal-modeling-for-2-5d-3d-heterogeneous-chiplet-systems
- 可用摘要要点：
  - 不是 AI 模型，而是面向异构 2.5D/3D chiplet 系统的高效热建模基线。
  - 关键改进包括材料异质性与各向异性保留、垂直层自适应划分、温度感知非均匀网格生成。
  - 相对现有工具报告 `3.61x ~ 6.46x` 加速，同时网格复杂度降低 `>23.3%`。
  - 与 COMSOL 对比，能够较好刻画横向和纵向热流。
- 适用方式：
  - 可作为“高效物理建模基线仍在持续进步，因此 AI 方法未来更可能与高效传统求解器协同而非完全替代”的支撑材料。

### 10. MFIT: Multi-Fidelity Thermal Modeling for 2.5D and 3D Multi-Chiplet Architectures

- 类型：arXiv 2024 / ACM TODAES 发表信息可检索
- DOI：10.48550/arXiv.2410.09188
- 链接：
  - https://arxiv.org/abs/2410.09188
  - https://dblp.org/rec/journals/corr/abs-2410-09188.html
- 可用摘要要点：
  - 面向 `16 / 36 / 64` 个 `2.5D` chiplets 以及 `16×3` 的 `3D` chiplet 系统，提出一组多保真热模型。
  - 核心目标是在不同设计阶段平衡速度与精度，而不是依赖单一热模型。
  - 报告执行时间可从 `days` 降到 `seconds` 甚至 `milliseconds`，同时保持可忽略的精度损失。
- 适用方式：
  - 该工作不是 AI 代理模型，但非常适合支撑“未来热设计流程将呈现多保真协同，而 AI 只是其中一层”的判断。

### 11. Optimizing Chiplet Placement in Thermally Aware Heterogeneous 2.5D Systems Using Reinforcement Learning

- 类型：EPTC 2024
- DOI：10.1109/EPTC62800.2024.10909800
- 链接：
  - https://www.researchgate.net/publication/387894229_Optimizing_Chiplet_Placement_in_Thermally_Aware_Heterogeneous_25D_Systems_Using_Reinforcement_Learning
- 可用摘要要点：
  - 提出 RL 框架优化 `2.5D heterogeneous systems` 中的 chiplet 放置。
  - 状态为空间中的 chiplet 位置，动作包括移动和旋转，奖励同时考虑温度与互连长度。
  - 在 multi-GPU 与 CPU-DRAM 系统上验证，报告相对 simulated annealing 更优。
- 备注：
  - 目前抓到的是作者上传摘要页而非 IEEE 正文页，但 DOI 与发表信息明确，可作为“热分析进入 chiplet 布局决策”的补充案例。

## 方法学基础文献

### 12. Neural operators for accelerating scientific simulations and design

- 类型：Nature Reviews Physics 2024
- DOI：https://doi.org/10.1038/s42254-024-00712-5
- 要点：
  - 神经算子适合学习函数到函数映射。
  - 可零样本超分辨和跨离散评估。
  - 在多类科学仿真中可带来 4 到 5 个数量级加速。

### 13. Geometry-Informed Neural Operator for Large-Scale 3D PDEs

- 类型：NeurIPS 2023
- 作用：
  - 说明几何感知神经算子对大规模 3D PDE 的价值。
  - 可作为“从规则网格走向复杂几何”的方法学支撑。

### 14. RIGNO: A Graph-based framework for robust and accurate operator learning for PDEs on arbitrary domains

- 类型：arXiv / OpenReview
- 作用：
  - 为后续选题落脚到图神经算子、任意域、迁移学习提供方法依据。

## 使用原则

- 最终 HTML 中，凡涉及定量结论，优先使用已读到摘要或正文信息的论文。
- 对仅确认题目但未读到摘要的论文，只用于补充趋势，不写具体数字结论。
