import os
from pathlib import Path
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_DIR = Path(__file__).resolve().parents[1]
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

from rigno.heat3d_paths import resolve_heat3d_data_dir

# ===============================
# 1️⃣ 基本加载
# ===============================
sample_dir = resolve_heat3d_data_dir(repo_dir=REPO_DIR) / "sample_000"

required_files = [
    "coords.npy",
    "temperature.npy",
    "edge_index.npy",
    "k.npy",
    "source.npy"
]

for f in required_files:
    if not os.path.exists(os.path.join(sample_dir, f)):
        raise FileNotFoundError(f"Missing file: {f}")

coords = np.load(os.path.join(sample_dir, "coords.npy"))
temperature = np.load(os.path.join(sample_dir, "temperature.npy"))
edge_index = np.load(os.path.join(sample_dir, "edge_index.npy"))
k = np.load(os.path.join(sample_dir, "k.npy"))
q = np.load(os.path.join(sample_dir, "source.npy"))

# ===============================
# 2️⃣ 基本信息
# ===============================
num_nodes = coords.shape[0]
num_edges = edge_index.shape[1]

print("=== Basic Shape Info ===")
print(f"Nodes: {num_nodes}")
print(f"Edges: {num_edges}")
print(f"Node Feature dims: coords(3) + k(1) + q(1)")
print()

# ===============================
# 3️⃣ 一致性检查
# ===============================
assert temperature.shape[0] == num_nodes
assert k.shape[0] == num_nodes
assert q.shape[0] == num_nodes
assert edge_index.max() < num_nodes
assert edge_index.min() >= 0

print("Graph consistency check passed.")
print()

# ===============================
# 4️⃣ 数值统计
# ===============================
print("=== Temperature Stats ===")
print(f"Min  : {temperature.min():.6f}")
print(f"Max  : {temperature.max():.6f}")
print(f"Mean : {temperature.mean():.6f}")
print(f"Std  : {temperature.std():.6f}")
print()

print("=== Material k Stats ===")
print(f"Unique k values: {np.unique(k)}")
print()

print("=== Heat Source Stats ===")
print(f"Non-zero heat nodes: {(q>0).sum()}")
print()

# ===============================
# 5️⃣ 保存温度直方图
# ===============================
plt.figure(figsize=(6,4))
plt.hist(temperature, bins=60)
plt.title("Temperature Distribution")
plt.xlabel("Temperature")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig("temp_hist.png", dpi=300)
plt.close()

print("Saved: temp_hist.png")

# ===============================
# 6️⃣ 保存 x=0.5 截面图
# ===============================
x_coords = coords[:, 0]

mask = np.abs(x_coords - 0.5) < 0.02

if mask.sum() > 0:
    plt.figure(figsize=(6,5))
    sc = plt.scatter(
        coords[mask][:,1],
        coords[mask][:,2],
        c=temperature[mask],
        s=10
    )
    plt.colorbar(sc)
    plt.title("Temperature Slice at x=0.5")
    plt.xlabel("y")
    plt.ylabel("z")
    plt.tight_layout()
    plt.savefig("slice_x_05.png", dpi=300)
    plt.close()
    print("Saved: slice_x_05.png")
else:
    print("Warning: No nodes found near x=0.5 for slice visualization.")

print("\nAll checks completed successfully.")
