import argparse
from pathlib import Path
import sys

REPO_DIR = Path(__file__).resolve().parents[1]
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

from rigno import dataset_Heat3D
from rigno.graphBuilder_Heat3D import Heat3DGraphBuilder   # 您的 graph builder
from rigno.heat3d_paths import CANONICAL_DATA_SUBDIR, resolve_heat3d_data_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke-test Heat3D data loading and graph construction.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help=f"Directory containing sample_xxx folders. Defaults to {CANONICAL_DATA_SUBDIR}, with legacy fallback.",
    )
    return parser.parse_args()


args = parse_args()
datadir = resolve_heat3d_data_dir(args.data_dir, REPO_DIR)

# ====================== 1. 加载数据集 ======================
dataset = dataset_Heat3D.Heat3DDataset(str(datadir))
# 遍历 sample_xxx 文件夹，读取：
# coords.npy 作为空间坐标
# temperature.npy 作为目标场 u
# k.npy 和 source.npy 叠成输入系数场 c
# 然后整理成：

# u: [1, 1, N, 1]
# x: [1, 1, N, 3]
# c: [1, 1, N, 2]

# ====================== 2. 创建并构建图 ======================
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
graphs = builder.build_graphs(g)          # 覆盖旧 test2 的核心图构建检查

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
