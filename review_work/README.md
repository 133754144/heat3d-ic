# 3D IC 热仿真 AI Review 过程文件

更新时间：2026-04-13

这个文件夹用于在对话中断后快速恢复进度。当前任务目标：

1. 阅读 `文献/` 中与 3D IC / Chiplet 热仿真相关论文。
2. 聚焦总结 AI 在热仿真中的作用、优势、局限与未来方向。
3. 输出一份新的 HTML review 报告，服务于选题报告撰写。

当前已完成：

- 阅读并参考现有 HTML：
  - `3DIC热仿真选题调研报告.html`
  - `3DIC文献分类与创新点分析.html`
  - `课题相关论文标题汇总.html`
- 初步精读的核心论文：
  - DeepOHeat (DAC 2023)
  - Enhanced Operator Learning / DeepOHeat+ (ASPDAC 2025)
  - Self-Attention to Operator Learning-based 3D-IC Thermal Simulation
  - ARO: Autoregressive Operator Learning for Transferable and Multi-fidelity 3D-IC Thermal Analysis with Active Learning
  - DeepOHeat-v1 (IEEE TCPMT 2026)
  - PI-ONet (IEEE TCPMT 2026)
  - Efficient ML-Based Transient Thermal Prediction for 3D-ICs (ASPDAC 2025)
  - Fast Machine Learning Based Prediction for Temperature Simulation Using Compact Models (DATE 2025)
  - Fast Thermal Modeling of TTSV via Bounded Neural Networks
  - Rapid heat source layout optimization ... using ANN ROM + BO
  - Learning Peak Temperature in 3DICs by Deep Differentiable Forest
  - Thermal-Aware Fixed-Outline 3-D IC Floorplanning: An End-to-End Learning-Based Approach
  - NeuralMesh: Neural Network for FEM Mesh Generation in 2.5D/3D Chiplet Thermal Simulation

过程文件说明：

- `literature_matrix.md`：按方向整理论文、AI作用和关键证据。
- 后续 HTML 将基于这些笔记生成。

建议续写顺序：

1. 先打开 `review_work/literature_matrix.md` 看分类和待补项。
2. 再打开最终生成的 HTML 检查是否需要补充新的论文或表格。
