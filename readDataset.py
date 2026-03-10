# from rigno.dataset import Dataset
# dataset = Dataset(
#   datadir='/home/xyh/myData/RIGNO',
#   datapath='unstructured/Heat-L-Sines', # Heat-L-Sines
#   time_downsample_factor=2,
#   space_downsample_factor=1.5,  # per direction
#   n_train=100,
#   n_valid=100,
#   n_test=10,
#   preload=True,
# )
# print(dataset)

import h5py
import numpy as np

path = "/home/xyh/myData/RIGNO/unstructured/Heat-L-Sines.nc"
with h5py.File(path, "r") as f:
    print("keys:", list(f.keys()))       # 你会看到 ['u', 'x']
    print("u shape:", f["u"].shape)      # (1500,21,14047,1)
    print("x shape:", f["x"].shape)      # (1,1,14047,2)

    # 只取一个切片（避免一次性读太大）
    u0 = f["u"][:1, :, :, :]              # sample=0,time=0 -> (14047,)
    x0 = f["x"][:1, :1, :, :]              # -> (14047,2)
    u = f["u"][0]
    ut = f["u"][0, :, 0, 0]  


# print(u0[:5], x0[:5])
print(x0.shape)
print(u0.shape)
# print(x0)
print(x0[0][0].shape)
print(x0[0][0])


# import h5py
# import matplotlib
# matplotlib.use("Agg")
# import matplotlib.pyplot as plt

# nc_path = "/home/xyh/myData/RIGNO/unstructured/Heat-L-Sines.nc"
# s = 0   # sample index: 0~1499
# t = 0   # time index: 0~20

# with h5py.File(nc_path, "r") as f:
#     u = f["u"][s, t, :, 0]        # (14047,)
#     x = f["x"][0, 0, :, :]        # (14047,2)

# plt.figure(figsize=(6, 5))
# plt.scatter(x[:, 0], x[:, 1], c=u, s=2, rasterized=True)
# plt.colorbar()
# plt.axis("equal")
# plt.title(f"Heat-L-Sines | sample={s} time={t}")

# out = f"Heat-L-Sines_s{s}_t{t}.png"
# plt.savefig(out, dpi=300, bbox_inches="tight")
# plt.close()
# print("saved:", out)

# import os
# import h5py
# import imageio.v2 as imageio
# import matplotlib
# matplotlib.use("Agg")
# import matplotlib.pyplot as plt

# nc_path = "/home/xyh/myData/RIGNO/unstructured/Heat-L-Sines.nc"
# s = 0  # sample

# out_dir = f"frames_s{s}"
# os.makedirs(out_dir, exist_ok=True)

# with h5py.File(nc_path, "r") as f:
#     x = f["x"][0, 0, :, :]  # (14047,2)

#     # 为了让颜色随时间一致：先取全时间的 min/max（只读一个 sample 的 21 帧，量不大）
#     u_all = f["u"][s, :, :, 0]  # (21,14047)
#     vmin = float(u_all.min())
#     vmax = float(u_all.max())

#     frame_paths = []
#     for t in range(u_all.shape[0]):
#         u = u_all[t]

#         plt.figure(figsize=(6, 5))
#         plt.scatter(x[:, 0], x[:, 1], c=u, s=2, rasterized=True, vmin=vmin, vmax=vmax)
#         plt.colorbar()
#         plt.axis("equal")
#         plt.title(f"Heat-L-Sines | sample={s} time={t}")

#         fp = os.path.join(out_dir, f"t{t:02d}.png")
#         plt.savefig(fp, dpi=200, bbox_inches="tight")
#         plt.close()
#         frame_paths.append(fp)

# gif_path = f"Heat-L-Sines_s{s}.gif"
# imgs = [imageio.imread(p) for p in frame_paths]
# imageio.mimsave(gif_path, imgs, duration=0.3)  # 每帧 0.3s
# print("saved:", gif_path)
