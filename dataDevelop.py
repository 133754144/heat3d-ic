import os
import numpy as np
from tqdm import tqdm
from dolfin import *

# ========= 全局参数 =========
NUM_SAMPLES = 200
MESH_RES = 16
SAVE_DIR = "dataset_3d_heat"

os.makedirs(SAVE_DIR, exist_ok=True)


# ========= 生成单样本 =========
def generate_sample(sample_id):

    # ---- 随机参数 ----
    k1 = np.random.uniform(1.0, 10.0)
    k2 = np.random.uniform(0.5, 5.0)
    q_amp = np.random.uniform(5.0, 20.0)
    r = np.random.uniform(0.08, 0.15)

    # ---- mesh ----
    mesh = UnitCubeMesh(MESH_RES, MESH_RES, MESH_RES)

    V = FunctionSpace(mesh, "P", 1)

    # ---- 材料标记 ----
    materials = MeshFunction("size_t", mesh, mesh.topology().dim())
    materials.set_all(0)

    class Right(SubDomain):
        def inside(self, x, on_boundary):
            return x[0] >= 0.5

    right = Right()
    right.mark(materials, 1)

    dx_sub = Measure("dx", domain=mesh, subdomain_data=materials)

    # ---- 定义 k(x) ----
    k = Function(V)
    k_values = k.vector().get_local()

    dof_coords = V.tabulate_dof_coordinates()
    for i, x in enumerate(dof_coords):
        if x[0] < 0.5:
            k_values[i] = k1
        else:
            k_values[i] = k2

    k.vector()[:] = k_values

    # ---- 热源 ----
    class HeatSource(UserExpression):
        def eval(self, values, x):
            if ((x[0]-0.5)**2 +
                (x[1]-0.5)**2 +
                (x[2]-0.5)**2) < r**2:
                values[0] = q_amp
            else:
                values[0] = 0.0

        def value_shape(self):
            return ()

    q = HeatSource(degree=1)

    # ---- 边界条件 ----
    bc = DirichletBC(V, Constant(0.0), "on_boundary")

    # ---- 弱形式 ----
    T = TrialFunction(V)
    v = TestFunction(V)

    a = k * dot(grad(T), grad(v)) * dx
    L = q * v * dx

    T_sol = Function(V)
    solve(a == L, T_sol, bc)

    # ---- 导出 graph 数据 ----
    coords = mesh.coordinates()
    temperature = T_sol.compute_vertex_values(mesh)

    # 构造材料与热源节点特征
    node_k = np.zeros(len(coords))
    node_q = np.zeros(len(coords))

    for i, x in enumerate(coords):
        node_k[i] = k1 if x[0] < 0.5 else k2
        if ((x[0]-0.5)**2 +
            (x[1]-0.5)**2 +
            (x[2]-0.5)**2) < r**2:
            node_q[i] = q_amp

    # ---- 构造边 ----
    cells = mesh.cells()
    edges = set()

    for cell in cells:
        for i in range(4):
            for j in range(i+1, 4):
                a_ = cell[i]
                b_ = cell[j]
                edges.add((a_, b_))
                edges.add((b_, a_))

    edge_index = np.array(list(edges)).T

    # ---- 保存 ----
    sample_dir = os.path.join(SAVE_DIR, f"sample_{sample_id:03d}")
    os.makedirs(sample_dir, exist_ok=True)

    np.save(os.path.join(sample_dir, "coords.npy"), coords)
    np.save(os.path.join(sample_dir, "temperature.npy"), temperature)
    np.save(os.path.join(sample_dir, "k.npy"), node_k)
    np.save(os.path.join(sample_dir, "source.npy"), node_q)
    np.save(os.path.join(sample_dir, "edge_index.npy"), edge_index)


# ========= 批量生成 =========
for i in tqdm(range(NUM_SAMPLES)):
    generate_sample(i)

print("Dataset generation complete.")
