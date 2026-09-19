"""Canonical V8 zero-delta-u bridge and asymmetric query method."""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np

from rigno.models.operator import Inputs
from rigno.models.rigno import RIGNO, RegionInteractionGraphBuilder

from .adapter import canonical_oracle_condition
from .schema import ProvenanceClass, V8OraclePhysicsView


@dataclass(frozen=True)
class V8CanonicalBridge:
    inputs: Inputs
    condition_feature_names: tuple[str, ...]
    condition_provenance: dict[str, ProvenanceClass]
    normalized_coords: np.ndarray
    coordinate_domain_m: np.ndarray


def normalized_coordinates(coords_m: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    coords = np.asarray(coords_m, dtype=np.float64)
    domain = np.stack((coords.min(axis=0), coords.max(axis=0)))
    span = domain[1] - domain[0]
    if coords.ndim != 2 or coords.shape[1] != 3 or np.any(span <= 0.0):
        raise ValueError("V8 coordinates must be nondegenerate [N,3]")
    normalized = 2.0 * (coords - domain[0]) / span - 1.0
    return normalized.astype(np.float32), domain.astype(np.float32)


def build_canonical_bridge(
    view: V8OraclePhysicsView,
    indices: np.ndarray | None = None,
) -> V8CanonicalBridge:
    condition, names, provenance = canonical_oracle_condition(view)
    normalized, domain = normalized_coordinates(view.case.coords_m)
    selected = np.arange(len(condition), dtype=np.int64) if indices is None else np.asarray(indices, dtype=np.int64)
    if selected.ndim != 1 or np.any(selected < 0) or np.any(selected >= len(condition)):
        raise ValueError("V8 bridge indices are invalid")
    c = jnp.asarray(condition[selected][None, None, :, :], dtype=jnp.float32)
    coords = jnp.asarray(normalized[selected][None, None, :, :], dtype=jnp.float32)
    u = jnp.zeros((1, 1, len(selected), 1), dtype=jnp.float32)
    return V8CanonicalBridge(
        inputs=Inputs(u=u, c=c, x_inp=coords, x_out=coords, t=None, tau=None),
        condition_feature_names=names,
        condition_provenance=provenance,
        normalized_coords=normalized,
        coordinate_domain_m=domain,
    )


def _pnode_features(inputs: Inputs) -> jnp.ndarray:
    if inputs.c is None or inputs.u.shape[:-1] != inputs.c.shape[:-1]:
        raise ValueError("canonical V8 query requires aligned u and c")
    if not bool(jnp.all(inputs.u == 0.0)):
        raise ValueError("canonical V8 query requires u=0")
    values = jnp.concatenate([inputs.u, inputs.c], axis=-1)
    values = jnp.moveaxis(values, source=(0, 1, 2, 3), destination=(0, 3, 1, 2)).squeeze(axis=3)
    return jnp.concatenate(
        [values, jnp.zeros((values.shape[0], 1, values.shape[-1]), dtype=values.dtype)],
        axis=1,
    )


def asymmetric_query_method(
    module: RIGNO,
    inputs_in: Inputs,
    inputs_out: Inputs,
    graphs,
    output_local_p2r,
    *,
    reuse_input_local_latents: bool,
    global_context,
):
    """Decode query-local physics from a 1024-point global support graph."""

    if module.concatenate_t or module.concatenate_tau:
        raise ValueError("canonical V8 query contract excludes time channels")
    if reuse_input_local_latents:
        if inputs_in.x_inp.shape != inputs_out.x_out.shape:
            raise ValueError("local-latent reuse requires equal support/query shape")
        if not bool(jnp.all(inputs_in.x_inp == inputs_out.x_out)) or not bool(
            jnp.all(inputs_in.c == inputs_out.c)
        ):
            raise ValueError("local-latent reuse requires identical support/query physics")
    features_in = _pnode_features(inputs_in)
    latent_r, latent_in = module.encoder(graphs.p2r, features_in, None, key=None)
    updated = module.processor(graphs.r2r, latent_r, None, key=None)
    updated = module._apply_global_film(updated, global_context)
    decoder_r = module._apply_shape_attention(
        updated, qk_region_features=None, global_context=global_context
    )
    if reuse_input_local_latents:
        latent_out = latent_in
    else:
        _, latent_out = module.encoder(
            output_local_p2r, _pnode_features(inputs_out), None, key=None
        )
    decoded = module.decoder(graphs.r2p, decoder_r, latent_out, None, key=None)
    output = module._prepare_features(decoded[:, :-1, :])
    return module._apply_decoder_bypass(output, inputs_out)


def output_local_p2r_graph(builder: RegionInteractionGraphBuilder, metadata):
    """Build only the query-local encoder node path; no dense query graph."""

    n_query = int(np.asarray(metadata.x_pnodes_out).shape[1] - 1)
    n_region = int(np.asarray(metadata.x_rnodes).shape[1] - 1)
    dtype = np.uint16 if max(n_query + 1, n_region + 1) < np.iinfo(np.uint16).max else np.uint32
    dummy = jnp.asarray(np.asarray([[[n_query, n_region]]], dtype=dtype))
    return builder._build_p2r_graph(
        metadata.x_pnodes_out, metadata.x_rnodes, dummy, metadata.r_rnodes
    )
