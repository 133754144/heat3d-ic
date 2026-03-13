import jax.numpy as jnp
from rigno.models.rigno import RegionInteractionGraphBuilder


class Heat3DGraphBuilder:

    def __init__(self):

        self.builder = RegionInteractionGraphBuilder(

            periodic=False,

            rmesh_levels=3,

            subsample_factor=4,

            overlap_factor_p2r=1.5,
            overlap_factor_r2p=2.0,

            node_coordinate_freqs=4,
        )


    def build_metadata(self, coords):

        coords = jnp.array(coords)

        domain = jnp.array([
            coords.min(axis=0),
            coords.max(axis=0)
        ])

        metadata = self.builder.build_metadata(
            x_inp=coords,
            x_out=coords,
            domain=domain
        )

        return metadata


    def build_graphs(self, metadata):

        graphs = self.builder.build_graphs(metadata)

        return graphs