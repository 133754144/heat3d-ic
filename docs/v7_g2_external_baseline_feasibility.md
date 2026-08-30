# V7 G2 external baseline feasibility audit

状态：`formal_protocol_proposal_ready`，不是正式 G2 结果。G2 分支基于 `78e7651bab5ef41a8ca4e42c45f64b1b98f04ea7`；G1 的 Heat3D comparison reference 固定为 `G1_FORMAL_CODE_SHA=191a7a06a681556f575a1c04e2b61cb13363efe1`。本审计没有读取 P1i `test_iid`/sealed，没有启动 formal 3-seed runs，也没有写 G1 worktree。

机器可读来源见 [upstream manifest](../configs/heat3d_v7/g2_upstream_manifest.json)，可执行证据见 [reproduction receipts](v7_g2_external_reproduction_receipts.json)，实际执行环境见 [dependency lock](../configs/heat3d_v7/g2_reproduction_dependency_lock.json)，所有临时修复见 [compatibility patch log](v7_g2_compatibility_patch_log.md)。早期 [G2-A audit](v7_g2_a_baseline_reproduction_audit.md) 只覆盖 adapter/forward smoke；若两者状态不同，以本文件和新 receipt 为准。

## 1. 六篇工作真正解决的问题

### DeepOHeat

[DeepOHeat（DAC 2023）](https://arxiv.org/abs/2302.12949)解决的是“同一类 3D-IC 热方程在 power map、BC、HTC、材料/几何配置变化下的 data-free operator learning”。它用 multi-input DeepONet 编码配置函数，用物理 PDE/BC residual 训练，不需要 Celsius labels。论文展示的价值是设计循环中的快速、physics-aware surrogate；这不是 Heat3D P1i 的普通 supervised point-cloud regression。

官方 release 的 2D power-map showcase 固定 `21×21×11` 网格、单 cuboid、均匀导热率和大部分 BC，只让 441 点顶面 power map 变化。它与 P1i 的 `coords + kx/ky/kz + q + BC` 全部逐点输入既不等维，也不等信息预算。

### Therm-FM

[Therm-FM（arXiv v2，DAC 2026 扩展版）](https://arxiv.org/abs/2605.22663)解决的是利用 Poseidon/scOT pretrained PDE prior，在 steady/transient、HotSpot/industrial benchmark 和 cross-chip adaptation 中降低高保真数据需求。它的核心比较维度是 transfer/data efficiency，而不是 from-scratch 同预算。

输入是多层规则网格，依赖 benchmark normalization 和 pretrained prior。把 P1i full field rasterize 到 grid 可以形成 common physical case，但 pretrained prior 与 dense grid 是额外归纳偏置/信息路径，不能称为 same training budget。

### DeepOHeat-v2

[DeepOHeat-v2（arXiv v1）](https://arxiv.org/abs/2608.16080)解决高导热率对比多 die stack 中两件事：用离散 energy-form physics loss 和 Muon2 改善病态训练；用 hotspot residual trust gate、AMG-warm GMRES 和 online retraining 让 placement optimization 中的 surrogate 自我改进。其问题是 solver-in-the-loop thermal optimization，不是固定 P1i train/valid supervised prediction。

论文假设已知 `300×300×23` FVM operator、axis-aligned grid、solver 和 placement trajectory。即使以后公开代码，也应归入 physics/design-loop comparison，不能直接放进 P1i same-budget 主表。

### GINO

[GINO（NeurIPS 2023）](https://proceedings.neurips.cc/paper_files/paper/2023/hash/70518ea42831f02afc3a2828993935ad-Abstract-Conference.html)解决 varying geometry 上的大规模 3D PDE operator learning：input GNO 把 irregular points 映射到规则 latent grid，latent FNO 建模，output GNO 查询任意输出点。它原论文的主要任务是 CarCFD surface pressure，不是热传导；但输入/输出拓扑与 P1i irregular point field 很接近。

在 P1i 上，latent grid 是模型内部 representation，不增加物理观测；只要输入仍严格来自 1024 点 raw P1i features，可作为 common-task、common-information-budget 外部基线。

### Transolver

[Transolver（ICML 2024 Spotlight）](https://arxiv.org/abs/2402.02366)用 Physics-Attention 把大规模 mesh points 软分配到 learned physical-state slices，在 slice tokens 间做 attention 再 deslice。它针对一般几何 PDE，官方六 benchmark 以 relative L2 为主。

P1i 可以直接把 1024 points 当 tokens，`coords` 和 11 features 不需额外 dense field、solver 或 pretrained prior，因此是最直接的 common-task、same-information-budget 候选。

### Geo-FNO

[Geo-FNO（JMLR 2023）](https://arxiv.org/abs/2207.05209)学习 physical domain 到规则 latent domain 的 deformation，再做 FNO；输入可为 point cloud、mesh 或 design code。它与 GINO 同样可桥接 irregular geometry，但原仓库已 deprecated，官方建议转向 NeuralOperator。

原论文实现可以运行，但当前 NeuralOperator 没有一个与所有旧脚本逐项等价的 drop-in modern Geo-FNO recipe。因此 original reproduction 与未来 modern implementation 必须分开报告。

## 2. Original reproduction 结果

| 模型 | original/official 证据 | 结果 | 限制 |
| --- | --- | --- | --- |
| DeepOHeat | repo 内 pretrained `model_epoch_10000.pth` + 10 paper showcase power maps | 10/10 CPU inference PASS，单 case 输出 `[4851,1]` | 未复训；固定 showcase 配置 |
| Therm-FM | 官方 quick demo synthetic steady dataset、3-epoch tiny scOT、官方 `evaluate.py` | training PASS；独立 evaluation PASS | 真实 model_T 被混合 T/B/L 的 24.10 GB 单体 checkpoint tar 阻断 |
| DeepOHeat-v2 | arXiv、作者/机构和 GitHub official-code 搜索 | `official_code_not_found` | 未把第三方仓库称为 original；未自行实现 |
| GINO | maintained NeuralOperator 官方 `test_gino.py` | `49 passed` | 是实现/shape coverage，不是 CarCFD paper-number reproduction |
| Transolver | 官方 Elasticity point-cloud data + 原训练脚本 | 16/8、1 epoch Physics-Attention PASS | 缩小模型与数据；误差只作 smoke diagnostic |
| Geo-FNO | deprecated 原仓库 + 官方 Elasticity `XY/sigma/rr` | 16/8、1 epoch deformation/FNO PASS | 旧代码维护风险；不是 modern implementation |

## 3. P1i adapter contract 与公平性

共同物理样本固定为 `heat3d_v6_p1i_continuous_physics1024_v1`，manifest SHA-256 为 `f19987c659968c2ac14eade1f1ef7e206c8f7eeb94f58fde5897d6e765978514`。本阶段只允许 train 与 `valid_iid`；adapter 不读取 labels，validation truth 只能由 `EvaluationCore` 外部提供。

共同 point-view 输入是：

```text
coords [B,1024,3]
+ [kx, ky, kz, q, is_top, is_bottom, is_side, is_interior,
   top_h, bottom_h, top_T_inf_minus_T_ref] [B,1024,11]
-> prediction_deltaT_K [B,1024,1]
```

| 模型 | P1i adapter | common physical case | common information budget | 信息增加/丢失 |
| --- | --- | --- | --- | --- |
| GINO | PASS，input-only + valid-only evaluator interface | 是 | 是 | latent grid 只重表达输入；不含额外物理观测。当前 upstream shared-geometry 约束使 batch=1 |
| Transolver | PASS，input-only + valid-only evaluator interface | 是 | 是 | point tokens 原样接收 coords/features；无 solver、dense field 或 prior |
| Geo-FNO | 未实现 | 可行 | 原则上可相同 | 需要决定 11 features 如何进入 deformation/FNO；不能复用 Elasticity 的 42-D shape code |
| Therm-FM | 未实现 | full-field rasterization 后可行 | 否 | pretrained Poseidon prior；grid interpolation/voxelization；可能使用 1024 点之外的 dense/full-field representation |
| DeepOHeat | 科学性停止 | 只能重建新的多输入配置模型 | 否 | released 441-point top power map 丢失 P1i heterogeneous k/BC；补全则需新增 branch 与物理 loss，实质改变 release |
| DeepOHeat-v2 | 不适用 | 不是固定预测 protocol | 否 | 需要 FVM matrix、solver、trust gate、placement trajectory 和在线 labels |

GINO/Transolver 的既有 P1i smoke 数值来自未训练随机模型，只证明 shape、dtype、adapter 和 evaluator 可接线，不是 accuracy evidence。

## 4. Feasibility classification

| 模型 | class | 结论 |
| --- | --- | --- |
| GINO | **A** | official implementation tests 成功 + P1i smoke 成功；可进入 formal G2 main table |
| Transolver | **A** | official benchmark small-subset training 成功 + P1i smoke 成功；可进入 formal G2 main table |
| Geo-FNO | **B, deprecated-upstream qualifier** | original reproduction 成功，P1i adapter 尚未跑通；完成 adapter 后可作 legacy sensitivity baseline |
| DeepOHeat | **C, physics-aware/data-free qualifier** | pretrained original reproduction 成功，但 P1i same-budget adapter 会实质改变 released problem；适合独立 physics/design-loop comparison |
| Therm-FM | **C, pretrained/transfer qualifier; D resource-packaging qualifier** | official quick demo 可运行，但 real model_T selective reproduction 被单体大包阻断；适合 transfer/data-efficiency track |
| DeepOHeat-v2 | **D, official_code_not_found; E for P1i same-table** | 论文审计完整但无官方实现，且 solver-in-loop 任务与 P1i 主表不相容 |

## 5. Formal G2 proposal

主表只推荐 **GINO + Transolver**。两者使用同一冻结 768 train / 128 `valid_iid`、同一 1024-point raw information budget、同一 seed set `{0,1,2}`、200 epochs、AdamW warmup-cosine horizon 200，并通过 V7 `EvaluationCore` 报告完整 metric names。不得把 `point_global_relative_rmse_pct`、`sample_first_relative_rmse_pct` 或 `raw_K_CV_RMSE_K` 简写成含混的“RMSE”。

正式主表的训练目标建议固定为 train-split normalized `deltaT` pointwise MSE；这是外部 supervised architecture 的 common objective，不冒充 G1 RIGNO 的 native shape/scale composite loss。模型选择仍只用 `valid_iid sample_first_relative_rmse_pct`，tie 取最早 epoch。报告每 seed 和 mean ± sample standard deviation；不做 posthoc seed 删除、metric 切换或 test/sealed 调参。

- GINO：一个预注册 latent grid，建议 `8³` 作为首个 formal candidate；只允许一次不看 accuracy 的 memory/shape feasibility check。若显式资源 gate 失败才可改为更小 grid，改动要在训练前冻结。
- Transolver：以官方 standard-benchmark family 为语义基线，建议 8 layers、hidden 128、8 heads、64 slices；若显存 gate 失败，只能在看 valid accuracy 前按预注册缩放规则降档。
- Geo-FNO：先完成同信息预算 P1i adapter 和 1-epoch train/valid smoke。通过后只能标为 legacy/deprecated sensitivity row，不与 maintained NeuralOperator implementation 混名。

分表推荐：

- Therm-FM：`pretrained / transfer / data-efficiency competitor`，比较 10/20/30 target samples 或相同 P1i train fractions；单独报告 pretrained prior 与 grid conversion。
- DeepOHeat：`physics-aware / data-free / design-loop competitor`，在其原生配置空间比较 solver-label cost、physics residual 和 inference speed，而非强塞进 P1i supervised 主表。
- DeepOHeat-v2：待 official code/data 后做 `solver-in-loop trustworthy optimization`；主指标应是 peak gap、solver calls、wall time 和 final verified design，而非 P1i field table。

formal execution 尚未授权；本 proposal 只冻结可运行候选、信息边界、分类和下一阶段 gate。
