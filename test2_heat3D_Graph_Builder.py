import numpy as np

from rigno import dataset_Heat3D
from rigno import graphBuilder_Heat3D


datadir="/home/xyh/myCode/rigno-main/dataset_3d_heat"

dataset=dataset_Heat3D.Heat3DDataset(datadir)
# 创建 graph builder
builder = graphBuilder_Heat3D.Heat3DGraphBuilder()
# 构建 graph metadata
dataset.build_graph_metadata(builder)
u,x,c,g=dataset[0]

coords=x[0,0]

print("coords shape:",coords.shape)

# 构建 graph
graphs = builder.build_graphs(g)

print("Graph built successfully")
print("p2r edges:",graphs.p2r.edges.shape)
print("r2r edges:",graphs.r2r.edges.shape)
print("r2p edges:",graphs.r2p.edges.shape)