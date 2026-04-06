from pathlib import Path

import jax.numpy as jnp

from rigno import dataset_Heat3D
from rigno.graphBuilder_Heat3D import Heat3DGraphBuilder


def edge_count(typed_graph):
    edge_key = list(typed_graph.edges.keys())[0]
    return typed_graph.edges[edge_key].n_edge


repo_dir = Path(__file__).resolve().parent
datadir = repo_dir / "dataset_3d_heat"

batch_indices = [0, 1, 2]

dataset = dataset_Heat3D.Heat3DDataset(str(datadir))
builder = Heat3DGraphBuilder()
dataset.build_graph_metadata(builder)

u_b, x_b, c_b, g_b = dataset.get_batch(batch_indices, return_graphs=True)

print("\n✅ Batch 张量读取成功")
print(f"batch u: {u_b.shape}")
print(f"batch x: {x_b.shape}")
print(f"batch c: {c_b.shape}")
print(f"batch g.x_pnodes_inp: {g_b.x_pnodes_inp.shape}")
print(f"batch g.x_rnodes: {g_b.x_rnodes.shape}")

assert u_b.shape == (len(batch_indices), 1, dataset.n_nodes, 1)
assert x_b.shape == (len(batch_indices), 1, dataset.n_nodes, 3)
assert c_b.shape == (len(batch_indices), 1, dataset.n_nodes, 2)
assert g_b.x_pnodes_inp.shape[0] == len(batch_indices)
assert g_b.x_pnodes_out.shape[0] == len(batch_indices)
assert g_b.x_rnodes.shape[0] == len(batch_indices)
assert g_b.r_rnodes.shape[0] == len(batch_indices)

if dataset.fix_x:
    assert jnp.allclose(g_b.x_pnodes_inp[0], g_b.x_pnodes_inp[1])
    assert jnp.allclose(g_b.x_rnodes[0], g_b.x_rnodes[1])

print("\n正在构建 batch 版实际图结构 (p2r, r2r, r2p) ...")
graphs = builder.build_graphs(g_b)

p2r_edges = edge_count(graphs.p2r)
r2r_edges = edge_count(graphs.r2r)
r2p_edges = edge_count(graphs.r2p)

print("✅ Batch Graph built successfully!")
print(f"p2r n_edge: {p2r_edges}")
print(f"r2r n_edge: {r2r_edges}")
print(f"r2p n_edge: {r2p_edges}")

assert p2r_edges.shape[0] == len(batch_indices)
assert r2r_edges.shape[0] == len(batch_indices)
assert r2p_edges.shape[0] == len(batch_indices)

print("\n✅ test3 通过：Heat3D 数据已经支持 batch 级 graph metadata，并能直接喂给作者的 build_graphs 接口")
