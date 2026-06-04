# Heat3D v2 开发目标梳理

## 一句话定位

Heat3D v2 是在 frozen V1 diagnostic baseline 之上，升级训练系统、配置系统、优化器和热点/场形态诊断能力的 research-stage controlled training 阶段。

## 为什么不能继续停留在 v1 loss tuning

v1 已经完成了有价值的诊断闭环：数据契约、medium1024 Gap-A 路径、controlled training/export、best-valid 选择、error-bin 和 condition-wise diagnostics 都已经能跑通。继续只调 v1 loss 权重，会把精力耗在局部背景误差与热点保真之间的 tradeoff 上，而不能解决更根本的问题。

当前暴露出的瓶颈主要是训练系统和表达能力：模型容量偏小，训练 runner 偏 smoke-scale，优化器仍停留在手写更新路径，配置复现实验不够结构化，峰值、热点、high-bin 和 field-shape 行为缺少足够直接的诊断。v2 应该把这些系统问题先理顺，再讨论是否扩数据、升分辨率或做更正式的 benchmark-candidate preparation。

## frozen V1 baseline 的作用

frozen V1 baseline 是 v2 的 reference，不是正式 benchmark。它的价值是把当前最好的 v1 diagnostic 配置固定下来，让 v2 的每个训练系统、optimizer、capacity 或 loss/diagnostic 改动都能做 V1 baseline comparison。

这个 baseline 应至少用于回答四个问题：

- v2 final epoch 是否相对 v1 reference 有稳定改善。
- v2 best-valid epoch 是否与 final epoch 存在明显差异。
- 背景 bin、high-bin、peak 和 hotspot 指标是否出现新的 tradeoff。
- 不同 seed 下的改进是否稳定，而不是单次训练偶然结果。

## v2 第一阶段任务清单

- 将 V1-style training runner 升级为可配置 V2 training system。
- 建立 dataset/model/optimizer/loss/run/export 配置结构，减少一次性 CLI 状态。
- 通过配置测试更大 latent width、processor steps 和 MLP depth，不直接改 `rigno/models/*`。
- 引入 Optax Adam / AdamW、gradient clipping、weight decay 和 LR schedule。
- 保留 final-vs-best prediction export，并把 best-valid 选择纳入默认报告。
- 增加 hotspot、peak、p95/p99、top-k overlap、field variance ratio、spatial correlation、slice-level field-shape diagnostics。
- 固定使用 `medium1024_gapA_full1024_v2` 作为 starting dataset；`medium256` 和 `medium1024` Gap-A 只做 debug / ablation。
- 只做短 smoke 和可复现性检查，不做长训练、不生成 full dataset。

## v2 第二阶段可选扩展

- 在第一阶段稳定后，做 seed sensitivity 和 final-vs-best 行为汇总。
- 设计 staged / curriculum loss：早期背景校准，中期 hotspot retention，后期 high-bin fidelity 与 calibration。
- 引入更明确的 peak / hotspot supervised loss，但仍先保持 output-space supervision，不急于加入 PDE / BC / energy residual loss。
- 准备 benchmark-candidate protocol：固定数据、固定 split、固定 metrics、固定 baseline，而不是直接声称 formal benchmark。
- 在训练系统稳定后，再评估是否需要更强 held-out stack / BC split 或更高分辨率数据。

## 与长期研究路线的关系

长期路线不是简单做一个更低平均误差的网络，而是面向 3D IC / chiplet 热仿真的图神经算子框架。调研资料反复指向几个核心方向：任意域表示、小样本迁移、多保真修正、等效微结构参数、物理一致性检查和设计闭环。

v2 当前阶段服务于这条长期路线的基础设施建设：先让训练、配置、optimizer、capacity 和 diagnostics 可信，再逐步接入 RIGNO 任意域图表示、TSV/bump/TTSV 等效材料块、多保真低/高精度修正，以及残差、热流、边界和界面一致性检查。

## 当前不做什么

- 不修改 `rigno/models/*`。
- 不修改 v0 public entrypoints。
- 不做长时间训练。
- 不生成 full dataset。
- 不扩大到更高分辨率或更大数据集。
- 不把 v2 smoke 或 diagnostic 结果说成 formal benchmark。
- 不声称 OOD generalization solved、high-fidelity validation、production-ready solver 或 publication-ready result。

## 下一轮适合交给 Codex 的小任务

1. 梳理现有 v1 runner 的 CLI 参数与输出文件，提出 v2 config schema 草案。
2. 新增最小 dataset/model/optimizer/loss/run/export config 文件，不改模型实现。
3. 为现有 prediction export 增加 final-vs-best 文件命名和报告字段检查。
4. 实现一组只读 diagnostics 计算函数草案：peak、p95/p99、top-k overlap、variance ratio、spatial correlation。
5. 写一个 v2 smoke command runbook，只包含 Mac 本地 `py_compile` 和短 smoke，不包含训练扩展或数据生成。
