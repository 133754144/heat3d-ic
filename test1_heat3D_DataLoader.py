from rigno import dataset_Heat3D
from rigno.graphBuilder_Heat3D import Heat3DGraphBuilder   # 您的 graph builder

datadir = "/home/xyh/myCode/rigno-main/dataset_3d_heat"

# ====================== 1. 加载数据集 ======================
dataset = dataset_Heat3D.Heat3DDataset(datadir)

# ====================== 2. 创建并构建图 ======================åå
builder = Heat3DGraphBuilder()
dataset.build_graph_metadata(builder)          # ← 这步已经成功

# ====================== 3. 取出第一个样本 ======================
u, x, c, g = dataset[0]
print("\n✅ 单个样本读取成功")
print(f"u shape: {u.shape}   | x shape: {x.shape}   | c shape: {c.shape}")

coords = x[0, 0]
print(f"coords shape: {coords.shape}   (应为 (4913, 3))")

# ====================== 4. 真正构建可用于模型的 Graph（关键！） ======================
print("\n正在构建实际图结构 (p2r, r2r, r2p) ...")
graphs = builder.build_graphs(g)          # ← 这就是 test2 的核心

print("✅ Graph built successfully!")

# 正确打印边数量（官方结构是 TypedGraph，edges 是 dict）
p2r_key = list(graphs.p2r.edges.keys())[0]
r2r_key = list(graphs.r2r.edges.keys())[0]
r2p_key = list(graphs.r2p.edges.keys())[0] if hasattr(graphs.r2p, 'edges') else None

print(f"p2r edges: {graphs.p2r.edges[p2r_key].n_edge[0]}")
print(f"r2r edges: {graphs.r2r.edges[r2r_key].n_edge[0]}")
print(f"r2p edges: {graphs.r2p.edges[r2p_key].n_edge[0] if r2p_key else 'None (可能被优化)'}")

# ====================== 5. Batch 测试 ======================
print("\nBatch 测试：")
u_b, x_b, c_b = dataset.get_batch([0, 1])
print(f"batch u: {u_b.shape}   | batch x: {x_b.shape}   | batch c: {c_b.shape}")