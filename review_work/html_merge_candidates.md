# HTML 回写候选清单

更新时间：2026-04-14

## 可以直接回写到 HTML 的外部文献

这些文献已经拿到可靠摘要或一手来源中的定量信息，可用于补强正文中的“代表工作”或“未来发展前景”：

1. Fast thermal analysis for chiplet design based on graph convolution networks
   - 适合回写位置：
     - 代表性图模型热分析工作
     - chiplet 热分析从规则结构走向图表示的趋势

2. T-Fusion: Thermal Prediction of 3D ICs with Multi-fidelity Fusion
   - 适合回写位置：
     - 多保真热预测
     - 小样本高保真热分析

3. Adaptive Graph Learning for Efficient Thermal Analysis of Multi-Stacking Chiplet Systems under Interface Variations
   - 适合回写位置：
     - chiplet 图学习热分析
     - post-silicon / interface variation / robustness

4. A Thermal Machine Learning Solver For Chip Simulation
   - 适合回写位置：
     - broader context：通用芯片热仿真 ML solver
     - 与 3D IC 专用方法进行方法谱系对照

5. Thermal Management Challenges in 2.5D and 3D Chiplet Integration: A Review on Architecture-Cooling Co-Design
   - 适合回写位置：
     - 背景与未来方向
     - 为什么 chiplet 热管理会推动 AI 热仿真需求

6. TDPNavigator-Placer: Thermal- and Wirelength-Aware Chiplet Placement in 2.5D Systems Through Multi-Agent Reinforcement Learning
   - 适合回写位置：
     - AI 结果进入设计闭环
     - thermal-aware chiplet placement

7. 3D-ICE 4.0: Accurate and efficient thermal modeling for 2.5D/3D heterogeneous chiplet systems
   - 适合回写位置：
     - 非 AI 高效热建模基线
     - 强调 AI 与高效物理模型协同的必要性

8. MFIT: Multi-Fidelity Thermal Modeling for 2.5D and 3D Multi-Chiplet Architectures
   - 适合回写位置：
     - 多保真设计流程
     - 强调未来热设计不会只依赖单一保真度模型

9. Optimizing Chiplet Placement in Thermally Aware Heterogeneous 2.5D Systems Using Reinforcement Learning
   - 适合回写位置：
     - thermal-aware chiplet placement
     - RL 进入封装级热约束设计决策

## 目前只适合标题级补充的外部文献

1. Transient Thermal Analysis of Chiplet-Based Systems with Dual-Channel Message Passing Neural Networks
   - 原因：已确认存在，但还没有拿到可靠摘要与量化指标。

2. POD-ANN Thermal Modelling Framework for Rapid Thermal Analysis of 2.5D Chiplet Designs
   - 原因：已确认存在，但仍缺少可直接引用的摘要内容。

## 下一步建议

如果继续修改 HTML，优先顺序建议为：

1. 先把 `Fast thermal analysis for chiplet design based on graph convolution networks`、`T-Fusion`、`Adaptive Graph Learning` 写入“代表性工作/未来方向”。
2. 再把 `Thermal Management Challenges in 2.5D and 3D Chiplet Integration`、`3D-ICE 4.0` 与 `MFIT` 写入“背景与趋势判断”。
3. 最后把 `TDPNavigator-Placer` 与 `Optimizing Chiplet Placement ... Using Reinforcement Learning` 作为“热分析结果进入设计决策”的补充案例。
