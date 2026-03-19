import os
import numpy as np
import jax.numpy as jnp


class Heat3DDataset:

    def __init__(self, datadir):
        self.datadir = datadir
        self.samples = []
        self.graph_metadata = []          # ← 新增：提前初始化

        # 🔥 只保留真正的 sample_xxx 文件夹（过滤 .DS_Store 和其他垃圾）
        all_items = sorted(os.listdir(datadir))
        sample_dirs = [d for d in all_items if d.startswith("sample_")]

        print(f"找到 {len(sample_dirs)} 个有效样本文件夹（已过滤 .DS_Store）")

        for s in sample_dirs:
            path = os.path.join(datadir, s)

            coords = np.load(os.path.join(path, "coords.npy"))
            temperature = np.load(os.path.join(path, "temperature.npy"))
            k = np.load(os.path.join(path, "k.npy"))
            q = np.load(os.path.join(path, "source.npy"))

            N = coords.shape[0]

            u = temperature.reshape(1, 1, N, 1)
            x = coords.reshape(1, 1, N, 3)
            c = np.stack([k, q], axis=-1).reshape(1, 1, N, 2)

            sample = {
                "u": u,
                "x": x,
                "c": c,
                "g": None
            }
            self.samples.append(sample)

        self.n_samples = len(self.samples)
        self.n_nodes = self.samples[0]["x"].shape[2]

        print("✅ Dataset loaded")
        print(f"   样本数量: {self.n_samples}")
        print(f"   每个样本节点数: {self.n_nodes}")

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        sample = self.samples[idx]
        u = jnp.array(sample["u"])
        x = jnp.array(sample["x"])
        c = jnp.array(sample["c"])
        g = self.graph_metadata[idx] if self.graph_metadata else None   # ← 防止未构建时报错

        return u, x, c, g

    def get_batch(self, batch_indices):
        u_list, x_list, c_list = [], [], []
        for idx in batch_indices:
            u, x, c, _ = self[idx]
            u_list.append(u)
            x_list.append(x)
            c_list.append(c)

        u = jnp.concatenate(u_list, axis=0)
        x = jnp.concatenate(x_list, axis=0)
        c = jnp.concatenate(c_list, axis=0)
        return u, x, c

    def build_graph_metadata(self, builder):
        """必须在读取数据后手动调用一次"""
        self.graph_metadata = []
        for sample in self.samples:
            coords = sample["x"][0, 0]                     # (N, 3)
            metadata = builder.build_metadata(coords)
            self.graph_metadata.append(metadata)
        print(f"✅ Graph metadata 已为 {len(self.graph_metadata)} 个样本构建完成")