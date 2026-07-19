# Heat3D-IC V3 算法模型审计

审计日期：2026-07-20  
审计对象：公开仓库 `main` 的 V3 控制训练路径  
审计源：`11e9d2feb1b920b2fbd06b8e626a706b1eb4cd40`  
配套图：[`docs/v3_algorithm_flowchart.html`](v3_algorithm_flowchart.html)（可打印 HTML）和 [`docs/v3_algorithm_flowchart.svg`](v3_algorithm_flowchart.svg)（矢量源）
模型文字说明：[`docs/v3_algorithm_model_description.html`](v3_algorithm_model_description.html)

## 1. 范围与结论

本报告只审查 V3 的稳态三维温升算子，不把当前 `research/v5` 的
Global FiLM、shape--scale 或 Gate 结果回填到 V3。审查了 V3 closeout、P3
路径/优化器/图覆盖审计、V3 canonical YAML，以及 RIGNO 的 encoder--processor--decoder
实现；没有启动训练、推理、数据生成或修改 `data/`、`output/`、checkpoint 和 log。

公开 `main` 将问题定义为固定三维点云上的监督映射

\[
  (\mathbf{x},\,\mathbf{k},\,\mathbf{q},\,\mathrm{BC})
  \longmapsto T(\mathbf{x}),
\]

其中模型实际学习的是温升 `DeltaT = T - T_ref` 的归一化场，再恢复为温度。
这是一条合理的多尺度图神经算子路径，但它仍是合成、固定网格、稳态监督回归，
不是带 PDE 残差、界面通量或边界条件残差的 physics-informed solver。公开 README
也明确列出这一限制；因此当前证据可以支持“V3 图/路径/优化器稳定化阶段”，不能
单独支持“已达到可发表的跨域热预测模型”。

审稿式总判断：

* **方法链条完整**：数据规范化、物理点--区域多尺度图、RIGNO 消息传递、反归一化
  与 checkpoint 保存均可追溯。
* **主要瓶颈已定位但尚未完全消除**：学习率/优化器、seed/path scale 和图覆盖都曾
  直接改变结果；V3 的最佳标量结果来自 checkpoint fine-tune，而不是独立 scratch
  复现实验。
* **指标证据不足以宣称“RMSE < 20%”**：`valid_base_mse` 是归一化空间 MSE，不能
  直接等同于真实均方根相对误差；V3 closeout 表中的结果被明确标成 diagnostic，
  不是 publication-ready benchmark。

## 2. V3 的可复现算法契约

### 2.1 输入、目标和归一化

对每个样本，物理输入为坐标 `x`、三轴导热系数 `k=(kx,ky,kz)`、局部热源 `q`
以及边界条件特征；监督目标为

\[
  \Delta T = T - T_{ref},\qquad
  \widetilde{\Delta T}=
  \frac{\Delta T-\mu_{\Delta T}^{train}}
       {\sigma_{\Delta T}^{train}}.
\]

均值、标准差和坐标尺度只由 train 拟合。V3 S4 canonical 配置固定了
`k_encoding_mode=diag3`、`feature_view=relative_bc_features`、
`bridge=zero_delta_u_bridge`、`target=normalized_deltaT` 和
`recovery=T_ref_plus_deltaT`。训练只对 `\widetilde{\Delta T}` 计算 plain MSE；
推理时按 train 统计量反归一化，再恢复 `\hat T=T_ref+\hat{\Delta T}`。

### 2.2 三张图与 RIGNO 路径

Graph builder 为同一物理点云建立物理节点 `p` 与下采样区域节点 `r`：

1. `p2r`：物理节点向区域 latent 聚合；
2. `r2r`：区域节点上的处理器消息传递；
3. `r2p`：区域 latent 解码回物理输出节点。

V3 的 canonical 容量是 `node_latent_size=96`、`edge_latent_size=96`、
`processor_steps=6`、`mlp_hidden_layers=2`。Encoder 用物理输入和结构边特征生成
`latent_pnodes`/`latent_rnodes`；Processor 对 `r2r` 做 6 次更新；Decoder 同时使用
处理后的 `rnodes` 与原始 `latent_pnodes`，通过 `r2p` 产生每个物理节点的归一化温升。
节点聚合使用 `segment_mean`，因此不是简单的逐点 MLP。

V3 选择 `discrete_physical_coverage` 作为图半径策略，使每个物理节点至少被一个
区域节点覆盖；legacy KDTree 的 mean-4 半径可能产生零 `p2r/r2p` 覆盖。canonical
S4 配置把 `coverage_repair_policy` 记为 `none`，并同时序列化 `repair_p2r/r2p`
布尔开关；复现实验必须以解析后的 runner 配置和实际 builder 参数为准，不能只读
文件名或单个 YAML 字段。

### 2.3 优化、选择与输出

V3 稳定默认是 AdamW + warmup/cosine，`B88 sample_shuffle`，plain MSE，按
`valid_base_mse` 保存 best；S4 `FT3` 是从 `S4discretebestFT2` best checkpoint
继续的低学习率 constant-LR fine-tune。每次 run 应保存 `params_best.pkl`、
`params_final.pkl`、预测、loss summary、run config 和诊断。

算法伪代码如下：

```text
fit train-only stats (condition, coordinates, normalized DeltaT)
for each sample:
    build pmesh/rmesh and p2r, r2r, r2p graphs
    z_p, z_r = Encoder(p2r, physical features, coordinates)
    z_r = Processor(r2r, z_r, steps=6)
    y_hat_norm = Decoder(r2p, z_r, z_p)
    loss = MSE(y_hat_norm, DeltaT_norm)
    AdamW update
select lowest valid_base_mse
DeltaT_hat = inverse_normalize(y_hat_norm)
T_hat = T_ref + DeltaT_hat
report raw errors, relative errors, field-shape and hotspot diagnostics
```

## 3. 可核查证据

| 证据 | 观察 | 审稿解释 |
| --- | --- | --- |
| S4 discrete FT3 | best epoch 375，`valid_base_mse=0.0179973`，raw DeltaT MSE `3.35874e-05`；best/final 参数存在 | 这是 V3 scalar reference，不是真实 RMS-relative 的完整证明；配置自身标为 diagnostic/non-publication-ready |
| B88 seed smoke | 同一 family 的 seed0/seed1/seed2 在 e400 的 valid loss、RMSE 和 correlation 差异很大 | 不能用单一 seed 的最佳曲线代表稳定性；需要解耦 model/batch/graph seed 后做预注册重复实验 |
| P3-c 单样本优化器 sanity | manual GD `1e-5` 约 68.30% relative RMSE；Adam `1e-3` 300 epoch 约 17.81%，1000 epoch 约 9.03% | 早期 RIGNO 失败主要被优化器/学习率混淆；诊断必须先使用可比的 Adam/AdamW |
| P3-a pointwise 对照 | 同一 `sample_000` 的 128x3 MLP 约 0.583% relative RMSE | 说明该样本可拟合，RIGNO 的差距来自图/容量/优化路径，不足以证明算子泛化 |
| decoder ablation | Adam 充分训练后，清零或 shuffle `rnode`/`pnode` 均显著恶化（约 55.68%--67.49%） | decoder 是 mixed path，不应沿用 under-training 阶段的“pnode-dominant”结论 |
| graph coverage audit | legacy 半径会产生零物理节点支持；离散 coverage 更可控 | 图覆盖是算法正确性前置条件，不只是调参项 |
| LR schedule audit | 旧 two-stage 代码曾把 update count 当作 epoch | 旧结果不能支撑 delayed-e400 的因果结论，必须以修正后 runner 重放 |

## 4. 严格审稿意见

### P0：指标与 claim 边界

1. `valid_base_mse` 在归一化 DeltaT 空间，不能写成“RMSE”；必须同时报告
   raw DeltaT RMSE、真实均方根分母的 relative RMSE、sample-first 与 point-global
   聚合方式、amplitude ratio 和 spatial correlation。
2. V3 的 `S4discretebestFT3` 依赖上游 best checkpoint；若作为正式基线，必须同时给出
   scratch、warm-start 来源、checkpoint epoch/hash 和完全相同的 split/evaluator。
3. public main 的 smoke 与 V3 controlled long run 不是同一证据层级，不能把 one-sample
   execution smoke 当作 full-dataset benchmark。

### P1：科学有效性与可重复性

1. 公开实现是 supervised operator，不包含 PDE residual、能量守恒、界面连续性或
   Dirichlet/Neumann penalty。它可以预测合成标签，但对未见材料/热源/边界分布没有
   物理可行性保证。
2. seed/path-scale 分裂明显，失败 seed 出现输出 amplitude collapse；在正式模型选择前
   需要固定 seed contract、初始化/梯度诊断和至少独立重复运行。
3. graph coverage、坐标度量和 regional-node 随机化共同决定 receptive field。应记录
   每个 split 的 coverage histogram、edge count、空邻域数和 graph hash，否则不同图
   不是同一个模型实验。
4. old schedule semantics bug 使部分历史结论不可比较；旧表格需标为 historical，不能
   与修正 runner 的 e400 直接合并。

### P2：建模与数据外推

1. 固定 1024 点云和合成 generator 限制了对真实封装层、材料界面、封装几何变化的外推。
2. `p_edge_masking=0.0` 与 upstream RIGNO example 的随机图/edge masking 不同；这不是
   必须照搬的改动，但应作为明确的 sensitivity cell，而不是隐含差异。
3. 只以 plain MSE 选 best 会偏向高幅值样本；需要把 hotspot、strong-q、低温升背景和
   per-sample CV-relative 指标作为冻结 evaluator 的并列输出，而不是训练中临时改权重。

## 5. 建议的最小修改方向（按优先级）

1. **先冻结评估器**：统一 train-only normalization、split hash、1024-node 检查和
   真实 RMS-relative 公式；为 best/final 分别输出 point-global 与 sample-first。
2. **再做可比的 scratch baseline**：保持 V3 graph/model capacity 不变，固定 3 个
   model seeds，记录参数 hash、graph hash、峰值显存和每 epoch 时间；不要以 warm-start
   结果替代 scratch。
3. **做单变量图覆盖实验**：legacy、nearest repair、discrete coverage 只改变 graph
   policy；报告每个样本的空覆盖、平均度和强 q 区域覆盖，才能判断误差是否来自 routing。
4. **做容量/局部性二因素实验**：在已修正 AdamW 之上，分别改变 processor depth/MLP
   与 node-local decoder features；禁止同时改 loss、graph 和 target。
5. **只在模型稳定后加入物理正则**：先以 supervised baseline 为参照，再单独加入 PDE/
   interface/BC residual，报告每项残差是否真的减少，以及是否牺牲 valid raw RMSE。
6. **外推验证独立封存**：新 generator seed、几何/材料/热源 OOD split 在模型和超参数
   冻结后才开启；不能在看过 test/hard 后反向选择图或 loss。

## 6. 源码追踪矩阵

| 组件 | V3 证据文件 |
| --- | --- |
| controlled runner、归一化、训练/保存 | [`scripts/run_heat3d_v1_medium_controlled_training_export.py`](../scripts/run_heat3d_v1_medium_controlled_training_export.py) |
| p2r/r2r/r2p graph builder | [`rigno/graphBuilder_Heat3D.py`](../rigno/graphBuilder_Heat3D.py) |
| RIGNO encoder/processor/decoder | [`rigno/models/rigno.py`](../rigno/models/rigno.py) |
| canonical S4 FT3 contract | [`configs/heat3d_v2/v3_S4discretebestFT3_e400_adamw_latent96_s6_mlp2_B88_sample_shuffle_base_mse_constant_lr2p5e-6_discrete_radius_model_seed0_batchbuild0_batchorder0_graphseed0.yaml`](../configs/heat3d_v2/v3_S4discretebestFT3_e400_adamw_latent96_s6_mlp2_B88_sample_shuffle_base_mse_constant_lr2p5e-6_discrete_radius_model_seed0_batchbuild0_batchorder0_graphseed0.yaml) |
| V3 closeout and numerical evidence | [`docs/v3_closeout_summary.md`](v3_closeout_summary.md)、[`docs/v3_p3c_upstream_training_gap_audit.md`](v3_p3c_upstream_training_gap_audit.md)、[`docs/v3_seed_path_activation_gradient_audit.md`](v3_seed_path_activation_gradient_audit.md) |
| public project scope/limitations | [GitHub `main`](https://github.com/133754144/heat3d-ic/tree/main) |

## 7. 图示说明

[`v3_algorithm_flowchart.html`](v3_algorithm_flowchart.html) 是面向浏览器和横向打印的
轻量 HTML 外壳；[`v3_algorithm_flowchart.svg`](v3_algorithm_flowchart.svg) 是其中的
矢量源。新版参考 upstream 配图，使用输入场缩略图、物理/区域节点平面和三色边，
只保留 `Input embedding`、`Processor ×6`、`Output projection` 等短标签，去掉了上一版
流程框中的长段落。图中实线表示主数据流，橙/紫/红分别表示 p2r/r2r/r2p；视觉层级、
留白、字体和颜色克制遵循 Apple Design skill 的 purpose/hierarchy/craft 原则，但没有
引入会影响论文阅读的动效或 UI 装饰。

建议论文图注：

> **V3 Heat3D steady thermal operator.** Train-only normalization maps conductivity,
> source and boundary features onto a physical/regional interaction graph. RIGNO
> encodes physical nodes to regional latents, performs six regional message-passing
> steps, and decodes back to the physical nodes. The normalized DeltaT prediction is
> recovered to temperature before raw and relative diagnostics; graph coverage and
> optimizer/seed audits are reported separately.
