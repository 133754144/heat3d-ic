import os
import numpy as np
import jax.numpy as jnp
import jax.tree_util as tree

from rigno.heat3d_paths import CANONICAL_DATA_SUBDIR, LEGACY_DATA_SUBDIR, resolve_heat3d_data_dir


class Heat3DDataset:

    def __init__(self, datadir):
        self.datadir = str(resolve_heat3d_data_dir(datadir))
        self.samples = []
        self.graph_metadata = []
        self.fix_x = True

        # 🔥 只保留真正的 sample_xxx 文件夹（过滤 .DS_Store 和其他垃圾）
        all_items = sorted(os.listdir(self.datadir))
        sample_dirs = [d for d in all_items if d.startswith("sample_")]
        if not sample_dirs:
            raise FileNotFoundError(
                "No sample_xxx directories found in "
                f"{self.datadir}. Expected the local dataset under "
                f"{CANONICAL_DATA_SUBDIR} or legacy {LEGACY_DATA_SUBDIR}."
            )

        print(f"找到 {len(sample_dirs)} 个有效样本文件夹（已过滤 .DS_Store）")

        for s in sample_dirs:
            path = os.path.join(self.datadir, s)

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
        ref_coords = self.samples[0]["x"]
        self.fix_x = all(np.array_equal(ref_coords, sample["x"]) for sample in self.samples[1:])

        print("✅ Dataset loaded")
        print(f"   样本数量: {self.n_samples}")
        print(f"   每个样本节点数: {self.n_nodes}")
        print(f"   坐标是否固定: {self.fix_x}")

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        sample = self.samples[idx]
        u = jnp.array(sample["u"])
        x = jnp.array(sample["x"])
        c = jnp.array(sample["c"])
        if not self.graph_metadata:
            g = None
        elif self.fix_x:
            g = self.graph_metadata[0]
        else:
            g = self.graph_metadata[idx]

        return u, x, c, g

    def get_batch(self, batch_indices, return_graphs=False):
        u_list, x_list, c_list = [], [], []
        for idx in batch_indices:
            u, x, c, _ = self[idx]
            u_list.append(u)
            x_list.append(x)
            c_list.append(c)

        u = jnp.concatenate(u_list, axis=0)
        x = jnp.concatenate(x_list, axis=0)
        c = jnp.concatenate(c_list, axis=0)
        if not return_graphs:
            return u, x, c

        g = self.get_graph_batch(batch_indices)
        return u, x, c, g

    def get_graph_batch(self, batch_indices):
        if not self.graph_metadata:
            raise ValueError("请先调用 build_graph_metadata(builder)")

        batch_size = len(batch_indices)
        if self.fix_x:
            shared_metadata = self.graph_metadata[0]
            return tree.tree_map(
                lambda value: jnp.repeat(value, repeats=batch_size, axis=0),
                shared_metadata,
            )

        metadata_list = [self.graph_metadata[idx] for idx in batch_indices]
        return tree.tree_map(
            lambda *values: jnp.concatenate(values, axis=0),
            *metadata_list,
        )

    def build_graph_metadata(self, builder, key=None):
        """必须在读取数据后手动调用一次"""
        self.graph_metadata = []
        samples = [self.samples[0]] if self.fix_x else self.samples
        for sample in samples:
            coords = sample["x"][0, 0]                     # (N, 3)
            metadata = builder.build_metadata(coords, key=key)
            self.graph_metadata.append(metadata)
        if self.fix_x:
            print("✅ Graph metadata 已构建完成（固定坐标，仅共享 1 份）")
        else:
            print(f"✅ Graph metadata 已为 {len(self.graph_metadata)} 个样本构建完成")
