import jax.numpy as jnp
from rigno.models.rigno import RegionInteractionGraphBuilder


class Heat3DGraphBuilder:

    def __init__(
        self,
        rmesh_levels=3,
        subsample_factor=4,
        overlap_factor_p2r=1.5,
        overlap_factor_r2p=2.0,
        node_coordinate_freqs=4,
        coverage_repair_policy="none",
        radius_policy="discrete_physical_coverage",
        repair_p2r=True,
        repair_r2p=True,
        min_physical_coverage=1,
    ):

        self.config = {
            "rmesh_levels": rmesh_levels,
            "subsample_factor": subsample_factor,
            "overlap_factor_p2r": overlap_factor_p2r,
            "overlap_factor_r2p": overlap_factor_r2p,
            "node_coordinate_freqs": node_coordinate_freqs,
            "coverage_repair_policy": coverage_repair_policy,
            "radius_policy": radius_policy,
            "repair_p2r": repair_p2r,
            "repair_r2p": repair_r2p,
            "min_physical_coverage": min_physical_coverage,
        }

        self.builder = RegionInteractionGraphBuilder(

            periodic=False,

            rmesh_levels=rmesh_levels,

            subsample_factor=subsample_factor,

            overlap_factor_p2r=overlap_factor_p2r,
            overlap_factor_r2p=overlap_factor_r2p,

            node_coordinate_freqs=node_coordinate_freqs,
            coverage_repair_policy=coverage_repair_policy,
            radius_policy=radius_policy,
            repair_p2r=repair_p2r,
            repair_r2p=repair_r2p,
            min_physical_coverage=min_physical_coverage,
        )


    def build_metadata(self, coords, key=None):

        coords = jnp.array(coords)

        domain = jnp.array([
            coords.min(axis=0),
            coords.max(axis=0)
        ])

        metadata = self.builder.build_metadata(
            x_inp=coords,
            x_out=coords,
            domain=domain,
            key=key,
        )

        return metadata


    def build_graphs(self, metadata):

        graphs = self.builder.build_graphs(metadata)

        return graphs
