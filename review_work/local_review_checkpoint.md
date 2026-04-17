# 本地版 review 检查点

更新时间：2026-04-14

## 当前目标

先基于 `文献/` 与 `tmp_lit_txt/` 中已经读取的本地论文，生成一份新的 HTML review 报告，暂不依赖联网补充。

## 本地版 review 拟采用结构

1. 研究背景与综述范围
2. 现有发展的阶段性总结
3. AI 在 3D IC 热仿真中的作用清单
4. 代表性论文分类评述
5. 未来发展前景
6. 对后续选题的启示
7. 参考文献

## 已确认可直接写入的核心判断

1. AI 在 3D IC 热仿真中的角色，已经从早期的峰值温度/少量指标回归，扩展到完整温度场求解、瞬态预测、参数提取、结构优化与流程加速。
2. 2023-2026 年最强主线是 `operator learning / neural operator`，代表工作包括 DeepOHeat、Enhanced Operator Learning、SAU-FNO、ARO、DeepOHeat-v1、PI-ONet。
3. 领域正在从“更快推理”走向“更强训练可扩展性、更多保真融合、更高可信度、更加贴近 chiplet/EDA 流程”。
4. 本地论文已经足够支撑以下未来方向判断：几何与结构泛化、多保真与迁移学习、可信代理模型、热-电-流体/可靠性协同、与 EDA 工作流闭环集成。

## 本地版重点引用论文

- DeepOHeat（DAC 2023）
- Enhanced Operator Learning for Scalable and Ultra-fast Thermal Simulation in 3D-IC Design（ASP-DAC 2025）
- Self-Attention to Operator Learning-based 3D-IC Thermal Simulation（DAC 2025）
- ARO: Autoregressive Operator Learning for Transferable and Multi-fidelity 3D-IC Thermal Analysis with Active Learning（ICCAD 2024）
- DeepOHeat-v1（IEEE TCPMT 2026）
- PI-ONet（IEEE TCPMT 2026）
- Efficient ML-Based Transient Thermal Prediction for 3D-ICs（ASP-DAC 2025）
- Fast Machine Learning Based Prediction for Temperature Simulation Using Compact Models（DATE 2025）
- Fast Thermal Modeling of TTSV via Bounded Neural Networks（ICEPT 2025）
- Rapid heat source layout optimization ... ANN-ROM + BO（Heat Transfer 2024）
- Learning Peak Temperature in 3DICs by Deep Differentiable Forest（ISEDA 2024）
- Thermal-Aware Fixed-Outline 3-D IC Floorplanning（IEEE TVLSI 2023）
- NeuralMesh（DAC 2025）
- Intelligent Design Method of Thermal Through Silicon Via for Thermal Management of Chiplet-Based System（IEEE TED 2023）
- Multiobjective Deep Reinforcement Learning Driven Collaborative Optimization of TSV-Based Microchannel and PDN for 3-D ICs（IEEE TCPMT 2026）

## 下一步

1. 创建新的本地版 HTML 文件。
2. 将该文件路径写回 `review_work/current_progress.md`。
3. 如本地版完成且需要继续扩充，再单独开启联网补强。
