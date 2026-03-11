import os
import numpy as np
import jax.numpy as jnp


class Heat3DDataset:

    def __init__(self, datadir):

        self.datadir = datadir
        self.samples = []

        sample_dirs = sorted(os.listdir(datadir))

        for s in sample_dirs:

            path = os.path.join(datadir, s)

            coords = np.load(os.path.join(path, "coords.npy"))
            temperature = np.load(os.path.join(path, "temperature.npy"))
            k = np.load(os.path.join(path, "k.npy"))
            q = np.load(os.path.join(path, "source.npy"))

            N = coords.shape[0]

            # -----------------------------
            # RIGNO required dimensions
            # (batch,time,nodes,channels)
            # -----------------------------

            # temperature
            u = temperature.reshape(1,1,N,1)

            # coordinates
            x = coords.reshape(1,1,N,3)

            # coefficient field
            c = np.stack([k,q],axis=-1)
            c = c.reshape(1,1,N,2)

            sample = {
                "u": u,
                "x": x,
                "c": c
            }

            self.samples.append(sample)

        self.n_samples = len(self.samples)
        self.n_nodes = self.samples[0]["x"].shape[2]

        print("Dataset loaded")
        print("Samples:", self.n_samples)
        print("Nodes per sample:", self.n_nodes)

    def __len__(self):
        return self.n_samples


    def __getitem__(self, idx):

        sample = self.samples[idx]

        u = jnp.array(sample["u"])
        x = jnp.array(sample["x"])
        c = jnp.array(sample["c"])

        return u,x,c


    def get_batch(self,batch_indices):

        u_list=[]
        x_list=[]
        c_list=[]

        for idx in batch_indices:

            u,x,c=self[idx]

            u_list.append(u)
            x_list.append(x)
            c_list.append(c)

        u=jnp.concatenate(u_list,axis=0)
        x=jnp.concatenate(x_list,axis=0)
        c=jnp.concatenate(c_list,axis=0)

        return u,x,c