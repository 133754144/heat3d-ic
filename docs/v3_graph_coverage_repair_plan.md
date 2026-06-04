# Heat3D v3 P1 Graph Coverage Repair Prototype

本文件记录 v3 P1-a/P1-b graph coverage repair prototype 的设计、只读 audit 结果和 P2 建议。
本轮未修改 model、decoder、loss 或 training 语义，未训练，未运行 full-dataset controlled run。

## 1. Upstream Coverage-Oriented 设计思想

upstream RIGNO 二维 support radius 使用 Delaunay triangles 的 simplex geometry。对每个 regional
node，它从相邻 triangle 的 median 尺度构造 support radius，设计目的不是估计 regional-node
局部密度，而是让 support regions 的并集覆盖 physical domain。

Heat3D 不能直接照搬该二维 helper：

- upstream helper 按 triangle 和二维 periodic shifts 编写。
- 3D IC point cloud 需要处理 tetrahedral geometry、薄层、边界、重复/近共面点和构图成本。
- 当前 Heat3D `KDTree mean-4 * 0.8` 只反映局部 regional-node spacing，没有 coverage guarantee。

P1 继承 upstream 的核心目标，而不是直接复制二维公式：首先保证当前离散 physical nodes 都可通过
`p2r/r2p` 进入 regional path，再控制 edge budget、degree distribution 和后续模型效果。

## 2. 默认与显式开关

`Heat3DGraphBuilder` 和 `RegionInteractionGraphBuilder` 新增：

- `coverage_repair_policy="none" | "nearest_rnode"`
- `radius_policy="legacy_kdtree_mean4" | "discrete_physical_coverage"`
- `repair_p2r=True | False`
- `repair_r2p=True | False`
- `min_physical_coverage=1`

默认仍为：

```text
coverage_repair_policy = none
radius_policy = legacy_kdtree_mean4
```

默认路径继续使用原有 KDTree radius、global `r_max` clip、`radius >= 0.5 -> 0.2` hard reset 和
edge construction。synthetic 固定基线与 P0 16-sample current summary 均保持不变。

## 3. P1-a Uncovered-Node Nearest Repair

流程：

1. 先完整构建 legacy p2r/r2p radius graph。
2. p2r 和 r2p 分别统计真实 physical-node degree。
3. 对 degree 低于 `min_physical_coverage` 的节点，按距离补最近且尚未存在的 real rnode edge。
4. 保留全部已有 edge；不补 dummy node/edge。

默认 `min_physical_coverage=1` 时，每个 zero-coverage physical node 只增加一条 edge。因此该策略：

- 能显式消除 zero coverage。
- 对现有 graph 的扰动和 edge 增量最小。
- 保留 legacy radius、hard reset 和已有 edge 的行为，便于隔离 coverage hole 对模型的影响。
- 不能改善大量 degree=1 的 low-coverage 节点，也不解决 legacy radius 本身的几何缺陷。

## 4. P1-b Discrete Physical-Node Coverage Radius

流程：

1. 在 builder 的归一化 graph coordinate metric 中，将每个真实 physical node 分配给最近 real
   rnode。
2. 每个 rnode 的 base radius 取其负责 physical nodes 的最大距离。
3. 用 `nextafter(radius, +inf)` 避免数值边界使负责节点落在 support region 外。
4. p2r/r2p 使用该最小 `1.0x` coverage radius 构边。
5. 不应用 legacy global clip、legacy overlap 或非单调 hard reset，因为它们可能破坏 coverage
   guarantee 或无界放大 edge budget。

该策略是三维离散 point set 上的 coverage-oriented radius policy。它类似离散 Voronoi assignment：
每个 physical node 至少落入负责它的 regional support region。它比 nearest repair 更接近 upstream
“support regions 覆盖 domain”的理念，但会改变 radius 和 degree distribution。

初版曾继续应用 legacy `1.5/2.0` overlap；显式 4-sample smoke 虽达到 zero coverage，但 edge ratio
达到 `6.97x/10.40x`，超过 P1 gate。因此 prototype 使用最小 `1.0x` coverage radius。未来若需要
额外 overlap，应增加独立、单调、显式受控的 discrete-policy overlap 参数，而不是复用 legacy cap。

## 5. Audit 与 Gate

`scripts/audit_heat3d_v3_graph_coverage.py` 新增：

- `--candidate-nearest-repair`
- `--candidate-discrete-coverage-radius`
- `--candidate-discrete-coverage-radius-with-nearest-repair`
- builder policy、p2r/r2p repair 和 minimum coverage 开关
- `repaired_edge_count`、`repaired_physical_count`、`edge_ratio_vs_legacy`
- finite、dummy excluded、node/r2r stable 和 coverage gate summary

P1 gate：

- p2r/r2p zero coverage 均为 0。
- metadata/graph finite。
- dummy 正确排除。
- physical/regional node 与 r2r topology 相对 legacy 稳定。
- edge ratio 低于 P0 no-hard-reset + no-global-clip 的 `3.74x/8.28x` 代价。

## 6. Synthetic 结果

固定 `8x8x6`、`12x12x8`、`16x16x12`，`rmesh seed=0`：

| policy | p2r zero | r2p zero | p2r edges | r2p edges | edge ratio vs legacy |
|---|---:|---:|---:|---:|---:|
| current | 210 | 289 | 17,744 | 32,001 | 1.000 / 1.000 |
| nearest repair | 0 | 0 | 17,954 | 32,290 | 1.012 / 1.009 |
| discrete coverage radius | 0 | 0 | 8,896 | 8,896 | 0.501 / 0.278 |
| discrete radius + nearest repair | 0 | 0 | 8,896 | 8,896 | 0.501 / 0.278 |

nearest repair 分别补了 210/289 条 p2r/r2p edge，恰好对应 current zero 数。discrete radius 已保证
coverage，因此 combined policy 没有补任何 edge。

## 7. Supervised-Small Real Audit

### 4 samples x 4 seeds

| policy | p2r zero | r2p zero | p2r edges | r2p edges | edge ratio vs legacy |
|---|---:|---:|---:|---:|---:|
| current | 789 | 789 | 1,833 | 1,755 | 1.000 / 1.000 |
| nearest repair | 0 | 0 | 2,622 | 2,544 | 1.430 / 1.450 |
| discrete coverage radius | 0 | 0 | 4,903 | 4,903 | 2.675 / 2.794 |
| discrete radius + nearest repair | 0 | 0 | 4,903 | 4,903 | 2.675 / 2.794 |

### 16 samples x 4 seeds

| policy | p2r zero | r2p zero | p2r edges | r2p edges | edge ratio vs legacy |
|---|---:|---:|---:|---:|---:|
| current | 3,225 | 3,229 | 7,417 | 7,057 | 1.000 / 1.000 |
| nearest repair | 0 | 0 | 10,642 | 10,286 | 1.435 / 1.458 |
| discrete coverage radius | 0 | 0 | 19,847 | 19,847 | 2.676 / 2.812 |
| discrete radius + nearest repair | 0 | 0 | 19,847 | 19,847 | 2.676 / 2.812 |

全部 P1 records 均满足 finite、dummy excluded、node/r2r stable 和 zero-coverage gate。nearest repair
补边数与 current zero 数完全一致。discrete radius 的 low-coverage count 明显更低，但 edge 数高于
nearest repair；combined 没有增加价值。

## 8. P2 推荐

建议 P2 同时保留两个候选，但以 **nearest repair 作为主 control**：

- 它以最小 edge 增量达到 zero coverage，最适合隔离“zero coverage 是否导致 one-sample
  memorization 和 4/16-sample loss 问题”。
- 它保持 legacy radius graph 的已有 edge，回归风险和解释成本最低。

将 **discrete coverage radius 作为 coverage-oriented A/B**：

- 它更接近 upstream support-region 覆盖理念。
- 它减少 degree=1 low-coverage 节点，但会显著改变 graph degree 和 radius 分布。
- 它的 edge ratio 仍通过 P1 gate，但需要 P2 小训练判断额外 connectivity 是否改善模型效果。

不建议让 combined policy 进入 P2：discrete radius 已达到 coverage guarantee，nearest repair 没有补边。
在 4/16-sample 小训练 smoke 证明 loss 可下降、shape stable、finite 后，才考虑 medium/stress audit 或
更大 controlled run。
