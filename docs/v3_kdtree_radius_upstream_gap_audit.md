# Heat3D v3 KDTree Support Radius 与 Upstream RIGNO 差距审计

本审计保留 upstream gap 分析，并补充 v3 P0 formal graph coverage audit 工具和 synthetic smoke。
未修改 graph/model/training 语义，未训练；审计 JSON 仅写入 ignored `output/`。

审计基线：

- 当前 Heat3D：`research/v3-startup-supervision`，commit `2d6cc778205591177475355d64ede8dfcbb6ff89`。
- 原作者 upstream：[camlab-ethz/rigno](https://github.com/camlab-ethz/rigno)，重点为
  [`rigno/models/rigno.py`](https://github.com/camlab-ethz/rigno/blob/main/rigno/models/rigno.py)。
- 原作者论文：[RIGNO, Appendix D.2](https://arxiv.org/html/2501.19205v1#A7.SS2)。
- 本地 Git 历史显示，commit `15ae3fd` 将原始 Delaunay radius 逻辑替换为当前
  KDTree 逻辑，并将 domain shifts 从 2D 扩为 3D。

## 1. 审计结论摘要

当前 `KDTree + 四近邻距离均值 * 0.8` 是一个计算简单、可直接接受 3D 坐标的工程适配，
但它不是 upstream support-radius 设计的 3D 等价实现。upstream 的核心设计目标是让
regional support regions 覆盖 domain；当前实现只估计 regional-node 局部密度，不检查
physical nodes 是否实际被覆盖，因此没有覆盖保证。

主要结论：

1. 当前策略更可能造成 `zero coverage`、`low coverage`、不均匀 receptive field，以及边界、
   薄层和界面信息传播不足；当前默认参数下，过大 edge count 不是首要风险。
2. `radius >= 0.5 -> 0.2` 是非单调的 radius reset，不是普通上限。在较粗的归一化 3D 网格上，
   它可以把本来较大的支持范围突然缩小到只覆盖 regional node 自身附近。
3. `jnp.clip(overlap_factor * r_i, a_max=r_max)` 不会把 radius 缩到原始 `r_i` 以下，但会限制
   overlap factor 修补 KDTree coverage hole 的能力。
4. 每轴独立归一化到 `[-1, 1]` 会丢失真实 3D IC 的 `x/y/z` 长宽比。对于
   `0.01 m x 0.01 m x 0.002 m` 的 stack，z 方向相对 x/y 被放大 5 倍，KDTree 邻域和
   Euclidean support sphere 不再反映真实物理距离。
5. 不建议直接回退 upstream 2D 实现。upstream helper 明确按三角形编写，不能直接作为可靠的
   3D tetrahedral implementation。建议保留 legacy path，并实现具有离散 physical-node
   coverage 保证的 3D-specific policy。

现有 v2 结果中 one-sample RIGNO memorization 仍约为 42%，而 pointwise baseline 在小样本上
明显更容易拟合。这与 graph coverage / graph-to-output bottleneck 假设一致，但不能仅凭本审计
认定 radius 是唯一原因。P0 必须先量化覆盖，再决定 P1 修复。

## 2. Upstream RIGNO Radius 设计意图

原作者实现位于 `RegionInteractionGraphBuilder._compute_minimum_support_radius()`。其流程是：

1. 对 regional nodes 构建 Delaunay triangulation。
2. 对每个三角形计算三条中线长度的约 `2/3`。
3. 对每个 regional node，取所有经过该节点的三角形中线值的最大值作为 minimum support radius。
4. 用 `overlap_factor_p2r` 和 `overlap_factor_r2p` 扩大基础 support region，再据此构建
   `p2r` 和 `r2p` edges。

论文 Appendix D.2 明确说明，该半径被定义为所有经过节点的三角形中最长中线的 `2/3`，
目标是使圆形 support regions 的并集覆盖整个 domain。代码注释也明确写出：
`ensures that the union of all support sub-regions covers the whole domain`。

该设计关注的是 Delaunay simplex 的几何空洞尺度，而不是仅关注 regional nodes 的平均局部间距。
边界节点如果邻接较大的 hull triangle，会得到较大的 radius，从而有机会覆盖边界邻域。

需要注意：

- upstream 当前实现是 2D-specific。periodic domain shifts 是 2D，代码假设 simplex 为三角形，
  `_compute_triangulation_medians()` 也按三角形边和中线公式实现。
- 论文说明相似方法可用于 3D point cloud，但没有意味着可以直接执行当前 2D helper。
- `radius >= 0.5 -> 0.2` 和全局 `r_max` clip 也存在于 upstream。它们是 upstream 的工程启发式，
  不是 Heat3D 新增；但 3D 归一化网格可能更频繁地触发这些启发式。

## 3. Heat3D 当前 KDTree 3D 改动说明

当前 `rigno/models/rigno.py:100-136` 的实现：

1. 对 subsampled regional nodes 构建 SciPy `KDTree`。
2. 对每个 regional node 查询 `self + 4` 个近邻。
3. 排除自身后，对四个近邻距离求均值，再乘 `0.8`：

   `r_i = 0.8 * mean(d_i,1 ... d_i,4)`

4. `build_metadata()` 使用：

   - `p2r`: `clip(1.5 * r_i, max=r_max)`
   - `r2p`: `clip(2.0 * r_i, max=r_max)`
   - 随后 `_get_supported_pnodes_by_rnodes()` 对 `radius >= 0.5` 执行 `radius = 0.2`

`rigno/graphBuilder_Heat3D.py` 固定 `periodic=False`，默认 `subsample_factor=4`、
`overlap_factor_p2r=1.5`、`overlap_factor_r2p=2.0`。当前没有 radius policy、cap policy 或
coordinate metric 的显式开关。

该改动的工程优点：

- KDTree 可直接处理 3D 坐标，不依赖 2D triangle helper。
- 相比 3D Delaunay tetrahedralization，通常更便宜、更容易处理较大点云。
- 对局部 regional-node 密度变化有一定自适应性。

主要限制：

- radius 只由 regional nodes 决定，不检查 physical nodes 到 regional nodes 的实际距离。
- 四近邻在 3D 中是较小、方差较高的局部样本；均值会弱化最大空洞方向。
- `0.8` 没有来自覆盖条件的推导，基础 radius 通常小于四近邻平均距离。
- 随机 subsampling 会改变局部四近邻集合，因此 coverage 对 rmesh seed 敏感。
- Python 循环逐点执行 KDTree query，虽然算法通常比 3D Delaunay 便宜，但仍有可避免的
  graph-build overhead。

## 4. 与 Upstream 的关键差异

| 方面 | Upstream RIGNO | 当前 Heat3D |
|---|---|---|
| radius 的依据 | Delaunay simplex 几何和最长 incident median | regional-node 四近邻平均距离 |
| 设计目标 | 覆盖 domain 的 minimum support region | 估计局部 regional-node spacing |
| physical-node coverage 检查 | 通过几何覆盖意图间接保证 | 无 |
| 对局部空洞的响应 | 最大 incident simplex 会放大 radius | 四近邻均值可能忽略远方向空洞 |
| 边界行为 | hull triangle 可扩大边界 radius | 大边界 radius 可能被 hard reset 为 `0.2` |
| 非均匀点云 | 由 triangulation topology 响应 | 依赖固定 `k=4` 的密度估计 |
| 3D 可执行性 | 当前 helper 不能直接可靠用于 3D | 可直接运行 3D |
| 构图成本 | 3D Delaunay 可能昂贵或不稳定 | KDTree 通常较便宜 |

关键区别不是 Delaunay 与 KDTree 工具本身，而是 radius 是否围绕“覆盖 physical domain / nodes”
构造。KDTree 可以用于覆盖保证策略，但当前 `mean-4 * 0.8` 没有实现这一目标。

## 5. 对 3D IC 热仿真模型性能的潜在影响

### 5.1 代表性场景

| 场景 | 当前策略的主要风险 |
|---|---|
| 规则 3D grid | physical grid 规则，但随机 subsampled rmesh 不规则；固定四近邻均值仍可能留下 hole。较粗网格更容易触发 hard reset。 |
| 多层薄结构 | 薄层可能只有少量 z planes，rmesh subsampling 后该层或界面附近 regional nodes 稀少；mean-4 不保证覆盖薄层 physical nodes。 |
| z 尺度远小于 x/y | 每轴归一化将 z 拉伸到与 x/y 同尺度，改变 nearest-neighbor 顺序和 support sphere 的物理含义，可能减少真实近距离的跨层连接。 |
| 边界和角点 | 边界外没有 regional nodes；局部邻域天然单侧。较大的边界 radius 又可能触发 `>=0.5 -> 0.2`，因此零覆盖和低覆盖风险最高。 |
| 材料界面附近 | graph 不感知材料或 interface。它可能因 coverage 不足阻断跨界面传播，也可能在 radius 较大时无差别跨越界面，造成过平滑。 |
| 非均匀采样点云 | 稠密 rmesh 区域的 radius 可能过小，稀疏区域的 radius 可能很大后被 reset；固定 `k=4` 对密度变化和空洞方向不稳健。 |

若 point cloud 在层间界面包含重复坐标，KDTree 的前几个距离可能为零，四近邻均值会进一步缩小
radius。当前 medium rectilinear grid 主要使用唯一坐标，但未来 layer-native point cloud 需要专门
检查这一风险。

### 5.2 对 RIGNO 信息路径的影响

- `p2r zero coverage`：该 physical node 的输入条件不能通过 `p2r` 进入 regional processor。
- `r2p zero coverage`：该输出 physical node 收不到 processed regional message。
- `low coverage`：`segment_mean` 聚合面对不均匀 degree 时，节点间有效信息量和方差不同，
  对随机 rmesh 更敏感。
- receptive field 过小：热传导是全局耦合问题；即使 `r2r` processor 可以传播全局信息，
  `p2r/r2p` 接口过窄仍会形成信息瓶颈。
- 边界/界面传播不足：可能损害 BC、层间热流和热点扩散的表示。

当前 decoder 仍接收 encoder 保留的 `latent_pnodes`，因此 zero `p2r/r2p` coverage 不等于输出
必然为零；未覆盖节点可能依赖本地 pnode path 输出。但它们缺少完整 regional context，这可以解释
one-sample memorization 困难和 full-dataset 泛化差的一部分，不能单独证明全部模型误差来源。

当前策略因 radius 偏小和 cap 存在，更可能是 edge 太少而不是 edge 太多。后续移除 cap 或增加
coverage 后，edge count、显存和 graph-build/forward 时间才会成为需要控制的风险。

### 5.3 无训练内存几何探针

为验证风险是否只存在于理论层面，本轮使用当前 builder 对三个规则 3D grid 做了无文件写入、
无训练的内存探针。坐标范围使用 `0.01 x 0.01 x 0.002 m`，builder 仍按当前逻辑逐轴归一化。
表中 zero 数均排除了 dummy node/edge。

| grid / rnodes | current p2r zero | current r2p zero | hard reset 触发 p2r / r2p | 去掉 hard reset 后 zero p2r / r2p | 同时去掉 global clip 后 zero p2r / r2p |
|---|---:|---:|---:|---:|---:|
| `8x8x6` / 96 | 189 / 384 | 288 / 384 | 68 / 96, 96 / 96 | 6 / 384, 3 / 384 | 0 / 384, 0 / 384 |
| `12x12x8` / 288 | 3 / 1152 | 0 / 1152 | 0 / 288, 0 / 288 | 3 / 1152, 0 / 1152 | 2 / 1152, 0 / 1152 |
| `16x16x12` / 768 | 18 / 3072 | 1 / 3072 | 0 / 768, 0 / 768 | 18 / 3072, 1 / 3072 | 15 / 3072, 0 / 3072 |

附加观察：

- `8x8x6` 中，hard reset 后所有 r2p radius 都变为 `0.2`。归一化网格间距大于 `0.2`，
  因此 r2p 基本只覆盖被选为 regional nodes 的 physical nodes。
- `12x12x8` 的 3 个 p2r zero nodes 全部位于边界。
- `16x16x12` 的 18 个 p2r zero nodes 中 17 个位于边界；唯一 r2p zero node 也位于边界。
- 即使不触发 hard reset，当前 KDTree radius 仍可能留下 p2r zero nodes，说明问题不只是 cap。

该探针不是正式 P0 dataset audit，但足以证明当前策略和 hard reset 可以实际制造 coverage defect。

## 6. 当前 Hard Cap、Clip 和 Normalization 的附加风险

### 6.1 `radius >= 0.5 -> 0.2`

这是当前最高风险项。它是一个不连续、非单调映射：更大的候选 radius 会得到更小的最终 radius。
其阈值和替代值都绑定到归一化坐标尺度，在 3D coarse grid 上可能高频触发。

upstream 中该逻辑用于处理 peculiar geometry 的异常大 radius，但在 upstream base radius 具有覆盖
设计意图时，它也可能破坏该意图。对当前 KDTree radius，它会进一步放大已有覆盖风险。

### 6.2 `clip(overlap_factor * r_i, a_max=r_max)`

当 overlap factor 大于等于 1 时，该 clip 不会把有效 radius 降到原始 `r_i` 以下，因此若原始
`r_i` 已保证覆盖，它主要限制额外 overlap，不一定破坏基础覆盖。

当前 KDTree `r_i` 不保证覆盖，overlap factor 本可修复部分 hole，但 global `r_max` clip 会阻止
局部 radius 继续扩大。内存探针中，去掉 global clip 后，部分 residual zero coverage 消失。

### 6.3 每轴独立归一化

`Heat3DGraphBuilder.build_metadata()` 使用每个样本自身的 min/max domain，core builder 再把每轴
独立映射到 `[-1, 1]`。因此：

- 真实 stack 的 x/y/z aspect ratio 被删除。
- 总高度不同的 stack 在 graph metric 中都变成相同高度。
- z 方向薄层相对整个 z domain 的比例仍保留，但 z 相对 x/y 的真实厚度不保留。
- hard-cap 阈值、KDTree nearest neighbors 和 Euclidean support sphere 都在变形后的 metric 中定义。

small/controlled runner 还会对 `Inputs.x` 使用 train-wide normalization，而 graph topology 和
structural coordinates 使用 builder 内的 per-sample normalization。这两套 metric 当前不是同一契约；
若样本 extent 变化或后续模型显式使用 `Inputs.x`，需要单独审计。

不建议简单改成 raw physical coordinates。真实物理尺度会使 z 距离远小于 x/y，可能反向造成
过多跨层近邻和不足的平面内连接。P1 应显式比较 per-axis、aspect-preserving 和 axis-weighted
metric，而不是隐式选择其中一个。

## 7. 后续 P0/P1 改进计划

### P0：先建立覆盖证据

建议新增独立 audit，不改 graph builder 语义。固定使用 1/4/16-sample cases，并覆盖规则 stack、
薄层、边界、interface 和 stress stack。对每个 rmesh seed 分别报告：

1. `p2r` physical sender degree、regional receiver degree。
2. `r2p` regional sender degree、physical receiver degree。
3. `r2r` in/out degree 和 isolated regional nodes。
4. zero count/ratio、degree `min/p1/p5/p10/median/p90/max`、degree 变异系数。
5. raw radius、overlap 后 radius、global clip 后 radius、hard reset 后 radius 的分位数和触发数。
6. physical-node coverage margin：对每个 physical node 统计最佳 `radius_j - distance(i,j)`。
7. 按 top/bottom/side/corner、layer、薄层、interface 邻域、material、source/hotspot、interior 分层。
8. 每样本 edge count、graph-build time、估计显存，以及不同 rmesh seed 的稳定性。

P0 至少做以下只读 A/B：

- A：当前完整 policy。
- B1：仅禁用 `>=0.5 -> 0.2`。
- B2：仅禁用 global `r_max` clip。
- B3：同时禁用 hard reset 和 global clip。
- B4：当前 KDTree metric 与 aspect-preserving / axis-weighted metric 对比。

P0 gate：

- 所有选定样本和 seeds 的 physical-node `p2r/r2p zero coverage == 0`。
- 边界、薄层和 interface 的低分位 degree 不出现明显塌陷。
- 改进不能只靠无界增加 edges；需同时报告 edge-count、graph-build time 和内存增量。
- shape stable、indices valid、所有 graph features finite。

### P1：候选 3D Radius Policy

建议按以下顺序评估：

1. **先处理 hard reset**：将当前非单调 reset 与 `none`、monotone cap、quantile-based guard 做 A/B。
2. **离散 physical-node coverage radius**：用 KDTree 将每个 physical node 分配给最近 regional node，
   每个 regional radius 取其负责 physical nodes 的最大距离，再应用可控 overlap。该策略直接保证
   当前离散 physical point set 至少被覆盖一次。
3. **hybrid radius + uncovered-node repair**：保留稀疏 radius graph，只对未覆盖 physical nodes 增加
   最近 regional edge。它可作为低 edge-count 的安全基线。
4. **fixed-degree pnode-to-rnode kNN control**：每个 physical node 连接固定数量 regional nodes，
   用于验证稳定 degree 是否优于 radius graph。
5. **3D Delaunay tetrahedral reference**：实现真正的 3D simplex/centroid 或更严格的 simplex-cover
   radius，作为最接近 upstream 设计意图的参考；需评估 Qhull 稳定性、共面/重复点和构图成本。
6. **改良 KDTree density policy**：比较第 k 近邻距离或近邻最大距离，而不是固定 mean-4 * 0.8；
   但若没有 coverage repair，它仍不能单独保证覆盖。
7. **axis-weighted / ellipsoidal support**：显式控制 z 与 x/y 的 metric，必要时后续再评估
   layer/interface-aware edges。不要在基础几何 coverage 未稳定时混入材料语义。

必须保留的 legacy 开关：

- 当前 Heat3D `kdtree_mean4_x0.8` policy。
- 当前 `radius>=0.5 -> 0.2` legacy reset。
- 当前 global `r_max` clip。
- 当前 per-axis unit-box normalization。
- upstream 2D Delaunay path，用于 2D/upstream alignment，不作为 3D 默认回退。

判断 P1 有效的指标：

- 首要：physical-node `p2r/r2p zero coverage == 0`。
- 次要：边界/薄层/interface degree 分布、coverage margin、seed 稳定性。
- 工程：edge count、graph-build time、graph memory、shape stability、finite。
- P1 之后进入 P2 时，再用固定 4/16-sample smoke 判断 loss 是否可下降、one-sample memorization 是否
  明显改善；在这些 gate 通过前不跑 full dataset。

最终建议是实现 3D-specific coverage policy，而不是直接回退 upstream 2D 代码。最稳妥的首个
候选是“离散 physical-node coverage radius 或 hybrid uncovered-node repair + 显式 metric/cap
开关”，同时用真正的 3D Delaunay policy 做 upstream-design reference。

## 8. 不建议立即做的事情

- 不直接把注释中的 upstream 2D Delaunay 代码取消注释用于 3D。
- 不只调 `0.8`、`k=4` 或 overlap factor，然后凭单一样本 loss 判断成功。
- 不在没有 edge-budget 指标时直接移除所有 cap 并跑 full dataset。
- 不用 decoder、pointwise skip 或 loss 改动掩盖尚未量化的 graph coverage defect。
- 不只检查总 edge count；总量正常仍可能同时存在大量 zero/low-coverage nodes 和局部高 degree。
- 不只使用一个 rmesh seed；当前随机 subsampling 会直接影响 KDTree radius 和 coverage。
- 不在 P0/P1/P2 gate 通过前做 full-dataset 性能或泛化结论。

## 9. P0 Formal Graph Coverage Audit

P0 新增两个独立诊断入口，不修改核心 graph builder：

- `scripts/audit_heat3d_v3_graph_coverage.py`：对真实 Heat3D sample、split、rmesh seed 和 policy
  输出逐记录 JSON。
- `scripts/check_heat3d_v3_graph_coverage_smoke.py`：固定运行 `8x8x6`、`12x12x8`、
  `16x16x12` synthetic probes；若本地存在 small 或 medium subset，再附加最多 4 个真实样本。

audit 始终报告当前 policy，并可选重放以下候选 policy：

- `candidate_no_hard_reset`
- `candidate_no_global_clip`
- `candidate_no_hard_reset_no_global_clip`

候选 policy 只在 audit 脚本内重建 `p2r/r2p` edge indices，用于只读 A/B；不会修改
`rigno/models/rigno.py` 或 `rigno/graphBuilder_Heat3D.py`。所有 coverage 与 edge count 均排除最后一个
dummy pnode、最后一个 dummy rnode，以及所有指向 dummy 的 edge。每条记录包含：

- p2r/r2p physical-node zero、low coverage 和 degree 分位数。
- p2r/r2p/r2r real edge count。
- raw、overlap、global clip、hard reset 后的 radius 分位数和 reset 触发数。
- metadata/graph shape signature、finite 检查、metadata/graph build time。
- top/bottom/side/corner/interior 分组；存在 `layer_id.npy`、`material_id.npy` 时再报告
  layer/interface/material 分组。元数据不足时记录 unavailable 原因，不使 audit 失败。

运行 synthetic-first smoke：

```bash
python scripts/check_heat3d_v3_graph_coverage_smoke.py
```

若 subset 不在当前 worktree，可显式传入：

```bash
python scripts/check_heat3d_v3_graph_coverage_smoke.py --subset /path/to/subset
```

运行 4-sample、4-seed 真实数据 audit：

```bash
python scripts/audit_heat3d_v3_graph_coverage.py \
  --max-samples 4 \
  --rmesh-seeds 0,1,2,3 \
  --candidate-no-hard-reset \
  --candidate-no-global-clip \
  --candidate-no-hard-reset-no-global-clip \
  --output-json output/heat3d_v3_graph_coverage/coverage_4sample.json
```

将 `--max-samples 4` 改为 `--max-samples 16`，即可运行 16-sample audit。仓库内 JSON 输出路径必须
被 Git ignore；脚本会拒绝写入未 ignore 的仓库路径。

### 9.1 Formal Synthetic Smoke 结果

以下结果使用 `rmesh seed=0`，均排除 dummy：

| grid / policy | p2r zero | r2p zero | p2r edges | r2p edges | hard reset p2r / r2p |
|---|---:|---:|---:|---:|---:|
| `8x8x6` current | 189 | 288 | 320 | 96 | 68 / 96 |
| `8x8x6` no hard reset | 6 | 3 | 1,340 | 1,800 | 0 / 0 |
| `8x8x6` no global clip | 189 | 288 | 320 | 96 | 68 / 96 |
| `8x8x6` no hard reset + no global clip | 0 | 0 | 1,533 | 3,219 | 0 / 0 |
| `12x12x8` current | 3 | 0 | 4,673 | 6,975 | 0 / 0 |
| `12x12x8` no hard reset | 3 | 0 | 4,673 | 6,975 | 0 / 0 |
| `12x12x8` no global clip | 2 | 18 | 5,015 | 6,534 | 3 / 100 |
| `12x12x8` no hard reset + no global clip | 2 | 0 | 5,089 | 11,354 | 0 / 0 |
| `16x16x12` current | 18 | 1 | 12,751 | 24,930 | 0 / 0 |
| `16x16x12` no hard reset | 18 | 1 | 12,751 | 24,930 | 0 / 0 |
| `16x16x12` no global clip | 15 | 0 | 12,830 | 28,305 | 1 / 5 |
| `16x16x12` no hard reset + no global clip | 15 | 0 | 12,870 | 28,664 | 0 / 0 |

三组 probe 合计：

| policy | p2r zero | r2p zero | p2r edges | r2p edges |
|---|---:|---:|---:|---:|
| current | 210 | 289 | 17,744 | 32,001 |
| no hard reset | 27 | 4 | 18,764 | 33,705 |
| no global clip | 206 | 306 | 18,165 | 34,935 |
| no hard reset + no global clip | 17 | 0 | 19,492 | 43,237 |

结论：

1. current policy 的 p2r/r2p zero coverage 已被正式 smoke 稳定复现。
2. hard reset 是 coarse grid coverage collapse 的主要放大器。
3. 仅禁用 global clip 并不安全：扩大后的 radius 可能跨过 `0.5`，随后被 hard reset 到 `0.2`；
   `12x12x8` 的 r2p zero 因此从 0 增到 18。
4. 同时禁用 hard reset 和 global clip 效果最好，但仍有 17 个 p2r zero，并使 r2p edge 总量增加
   约 35%。它是 P1 A/B 候选，不是可直接采用的最终修复。
5. 当前工作树自身没有 small 或 medium subset，因此默认 smoke 的自动 real-data 部分会跳过；
   最终 smoke 通过 `--subset` 只读 sibling linked worktree 中的 supervised-small subset，并完成了
   最多 4 个真实样本检查；同一 subset 还完成了显式 4/16-sample、4-seed audit。

基于 formal smoke，建议进入 P1 设计与小范围实现，但首选目标应是具有显式离散 coverage 保证的
3D-specific policy 或 uncovered-node repair，而不是只删除某个 cap。P1 后仍需用真实 4/16-sample、
多 seed audit 验证 zero coverage、分组低覆盖、edge budget、shape stable 和 finite。

### 9.2 Supervised-Small Real-Data Audit

本轮用本地 16-sample supervised-small subset 做了只读 real-data audit。每个样本使用
`rmesh seeds=0,1,2,3`；4-sample 结果包含 16 个 sample-seed records、1,920 个 physical-node
instances，16-sample 结果包含 64 个 records、7,808 个 physical-node instances。

| scope / policy | p2r zero | r2p zero | p2r edges | r2p edges |
|---|---:|---:|---:|---:|
| 4-sample current | 789 (41.09%) | 789 (41.09%) | 1,833 | 1,755 |
| 4-sample no hard reset | 98 (5.10%) | 70 (3.65%) | 5,688 | 9,505 |
| 4-sample no global clip | 789 (41.09%) | 789 (41.09%) | 1,833 | 1,755 |
| 4-sample no hard reset + no global clip | 60 (3.12%) | 9 (0.47%) | 6,766 | 14,232 |
| 16-sample current | 3,225 (41.30%) | 3,229 (41.36%) | 7,417 | 7,057 |
| 16-sample no hard reset | 374 (4.79%) | 266 (3.41%) | 23,496 | 39,423 |
| 16-sample no global clip | 3,225 (41.30%) | 3,229 (41.36%) | 7,417 | 7,057 |
| 16-sample no hard reset + no global clip | 234 (3.00%) | 33 (0.42%) | 27,730 | 58,427 |

真实数据结论：

1. current policy 在 supervised-small 上约 41% 的 physical-node instances 为 p2r/r2p zero coverage，
   明确确认当前问题需要进入 P1。
2. 仅禁用 global clip 与 current 完全相同；hard reset 仍主导最终 coverage。
3. 仅禁用 hard reset 已大幅改善 coverage，但仍留下 374/266 个 p2r/r2p zero。
4. 联合候选进一步降到 234/33，但 p2r/r2p edge 总量分别增至 current 的约 3.74x/8.28x，
   仍不是可直接采用的修复。
5. current 的 zero coverage 在 side group 尤其集中，但 interior 也存在 zero；不能只做边界补丁。
6. supervised-small P0 证据已足以支持一个保留 legacy 开关的 P1 prototype；medium/stress subset
   可用后，仍需完成同样的 4/16-sample gate，再决定默认 policy。

## 10. P1 Coverage Repair Prototype

P1 已实现并完成只读验证，详细设计和结果见
`docs/v3_graph_coverage_repair_plan.md`。

结论摘要：

- 默认仍为 legacy KDTree radius + no repair，synthetic 固定基线和 P0 real current summary 未变化。
- nearest repair 只为 zero-coverage physical nodes 补最近 real rnode edge；16-sample × 4-seed 达到
  p2r/r2p zero `0/0`，edge ratio 为 `1.435x/1.458x`。
- discrete physical coverage radius 将 physical nodes 分配给最近 rnode，并取负责节点最大距离作为
  最小 coverage radius；它绕过会破坏 guarantee 的 legacy clip/hard reset。16-sample × 4-seed
  达到 `0/0`，edge ratio 为 `2.676x/2.812x`。
- discrete radius + nearest repair 与 discrete radius 完全相同，说明 P1-b 已自行满足离散 coverage
  guarantee。
- 建议 P2 以 nearest repair 为主 control，并以 discrete coverage radius 做 coverage-oriented A/B。
