"""Stable V7 U-v2 asymmetric high-resolution query runtime.

The implementation keeps the frozen V6/P1i U-v2 semantics: the native
conditioning graph is built at 1,024 points, the output query graph is built
at the requested resolution, and the model receives the native encoder input
plus the high-resolution output query.  It contains no script imports,
runtime hooks, or module-state mutation.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Mapping

import jax
import jax.numpy as jnp
import numpy as np

from rigno.graphBuilder_Heat3D import Heat3DGraphBuilder
from rigno.heat3d_runtime.grouping import EDGE_FIELDS, GroupBuilder
from rigno.heat3d_runtime.high_n import (
    HighNRuntime,
    SupportArtifact,
    FullFieldGeometry,
)
from rigno.heat3d_runtime.session import RuntimeSession


U_V2_NUMERICAL_TOLERANCE = 1.0e-6
U_V2_MAXIMUM_NORMALIZED_OVERSHOOT = 0.25


def _features(module: Any, inputs: Any) -> tuple[Any, Any]:
    """Reproduce the frozen RIGNO physical-node feature packing."""

    batch_size = inputs.u.shape[0]
    n_points = inputs.x_inp.shape[2]
    if module.concatenate_t:
        t = jnp.asarray(inputs.t, dtype=jnp.float32)
        if t.ndim == 4:
            t = t[:, :, 0, 0]
        if t.size == 1:
            t = jnp.tile(t.reshape(1, 1), reps=(batch_size, 1))
    if module.concatenate_tau:
        tau = jnp.asarray(inputs.tau, dtype=jnp.float32)
        if tau.ndim == 4:
            tau = tau[:, :, 0, 0]
        if tau.size == 1:
            tau = jnp.tile(tau.reshape(1, 1), reps=(batch_size, 1))
    else:
        tau = None
    u = inputs.u if inputs.c is None else jnp.concatenate([inputs.u, inputs.c], axis=-1)
    result = jnp.moveaxis(u, (0, 1, 2, 3), (0, 3, 1, 2)).squeeze(axis=3)
    forced = []
    if module.concatenate_t:
        forced.append(jnp.tile(t[:, None, :], reps=(1, n_points, 1)))
    if module.concatenate_tau:
        forced.append(jnp.tile(tau[:, None, :], reps=(1, n_points, 1)))
    return jnp.concatenate([result, *forced], axis=-1), tau


def _native_finish(
    module: Any,
    *,
    psi: Any,
    processed: Any,
    pre_film: Any,
    control_volumes: Any,
    log_s_phys: Any,
    reference_temperature: Any,
    dirichlet_mask: Any,
    prescribed_temperature: Any,
    global_context: Any,
    qk_region_features: Any,
    scale_context: Any,
    scale_region_source_weights: Any,
    scale_region_volume_weights: Any,
) -> dict[str, Any]:
    volumes = module._prediction_field(control_volumes, psi, "control_volumes")
    volume_sum = jnp.sum(volumes, axis=2, keepdims=True)
    dirichlet = module._prediction_field(dirichlet_mask, psi, "dirichlet_mask") > 0.5
    psi_free = jnp.where(dirichlet, jnp.zeros_like(psi), psi)
    psi_rms = jnp.sqrt(
        jnp.sum(jnp.square(psi_free) * volumes, axis=2, keepdims=True)
        / jnp.maximum(volume_sum, module.shape_scale_epsilon)
    )
    phi_hat = psi_free / jnp.maximum(psi_rms, module.shape_scale_epsilon)
    context = module._global_context_array(
        global_context, batch_size=psi.shape[0], dtype=psi.dtype
    )
    if module.scale_head_mode == "physics_plus_pooled_latent":
        pooled = module._pooled_scale_features(
            processed,
            pre_film,
            qk_region_features=qk_region_features,
            global_context=global_context,
            scale_region_source_weights=scale_region_source_weights,
            scale_region_volume_weights=scale_region_volume_weights,
        )
        scale_context_array = module._scale_context_array(
            scale_context, batch_size=psi.shape[0], dtype=psi.dtype
        )
        scale_features = jnp.concatenate([context, scale_context_array, pooled], axis=-1)
    else:
        pooled = jnp.zeros((psi.shape[0], 0), dtype=psi.dtype)
        scale_context_array = module._scale_context_array(
            scale_context, batch_size=psi.shape[0], dtype=psi.dtype
        )
        scale_features = jnp.concatenate([context, scale_context_array], axis=-1)
    hidden = jax.nn.gelu(module.global_scale_hidden(scale_features))
    for layer in module.global_scale_extra_hidden:
        hidden = jax.nn.gelu(layer(hidden))
    residual = module.global_scale_output(hidden)[:, :, None, None]
    log_s_hat = module._sample_scalar(log_s_phys, psi, "log_s_phys") + residual
    s_hat = jnp.exp(log_s_hat)
    delta = s_hat * phi_hat
    reference = module._prediction_field(reference_temperature, psi, "reference_temperature")
    prescribed = module._prediction_field(prescribed_temperature, psi, "prescribed_temperature")
    raw_unprojected = reference + delta
    raw = jnp.where(dirichlet, prescribed, raw_unprojected)
    return {
        "psi": psi,
        "phi_hat": phi_hat,
        "s_hat": s_hat,
        "processed_rnodes": processed,
        "processed_rnodes_pre_film": pre_film,
        "raw_temperature": raw,
        "deltaT_hat": raw - reference,
        "pooled_rnodes": pooled,
    }


def _split_native_shape_scale(
    module: Any,
    inputs_in: Any,
    inputs_out: Any,
    graphs: Any,
    output_local_p2r: Any,
    *,
    split: bool,
    control_volumes: Any,
    log_s_phys: Any,
    reference_temperature: Any,
    dirichlet_mask: Any,
    prescribed_temperature: Any,
    global_context: Any,
    qk_region_features: Any,
    scale_context: Any,
    scale_region_source_weights: Any,
    scale_region_volume_weights: Any,
) -> dict[str, Any]:
    features_in, tau = _features(module, inputs_in)
    features_in = jnp.concatenate(
        [features_in, jnp.zeros((features_in.shape[0], 1, features_in.shape[-1]), dtype=features_in.dtype)],
        axis=1,
    )
    latent_r, latent_in = module.encoder(graphs.p2r, features_in, tau, key=None)
    updated = module.processor(graphs.r2r, latent_r, tau, key=None)
    pre_film = updated[:, :-1]
    updated = module._apply_global_film(updated, global_context)
    processed = updated[:, :-1]
    decoder_r = module._apply_shape_attention(
        updated, qk_region_features=qk_region_features, global_context=global_context
    )
    if split:
        features_out, tau_out = _features(module, inputs_out)
        features_out = jnp.concatenate(
            [features_out, jnp.zeros((features_out.shape[0], 1, features_out.shape[-1]), dtype=features_out.dtype)],
            axis=1,
        )
        _, latent_out = module.encoder(output_local_p2r, features_out, tau_out, key=None)
    else:
        latent_out = latent_in
    decoded = module.decoder(graphs.r2p, decoder_r, latent_out, tau, key=None)[:, :-1, :]
    pre_bypass = module._prepare_features(decoded)
    post_bypass = module._apply_decoder_bypass(pre_bypass, inputs_out if split else inputs_in)
    result = _native_finish(
        module,
        psi=post_bypass,
        processed=processed,
        pre_film=pre_film,
        control_volumes=control_volumes,
        log_s_phys=log_s_phys,
        reference_temperature=reference_temperature,
        dirichlet_mask=dirichlet_mask,
        prescribed_temperature=prescribed_temperature,
        global_context=global_context,
        qk_region_features=qk_region_features,
        scale_context=scale_context,
        scale_region_source_weights=scale_region_source_weights,
        scale_region_volume_weights=scale_region_volume_weights,
    )
    result.update(
        {
            "encoder_input_pnodes": latent_in[:, :-1],
            "encoder_output_local_pnodes": latent_out[:, :-1],
            "decoder_pre_bypass": pre_bypass,
            "decoder_post_bypass": post_bypass,
        }
    )
    return result


def _repair_uncovered(
    edge_indices: np.ndarray,
    centers: np.ndarray,
    points: np.ndarray,
    min_physical_coverage: int,
) -> np.ndarray:
    edges = np.asarray(edge_indices)
    degree = np.bincount(edges[:, 0], minlength=len(points))
    uncovered = np.flatnonzero(degree < int(min_physical_coverage))
    if not len(uncovered):
        return edges
    distance = np.linalg.norm(
        points[uncovered, None, :] - centers[None, :, :], axis=-1
    )
    nearest_order = np.argsort(distance, axis=1)
    existing = {(int(point), int(region)) for point, region in edges}
    additions: list[tuple[int, int]] = []
    for point_value in uncovered:
        point = int(point_value)
        needed = int(min_physical_coverage) - int(degree[point])
        for region_value in nearest_order[point]:
            edge = (point, int(region_value))
            if edge in existing:
                continue
            additions.append(edge)
            existing.add(edge)
            needed -= 1
            if needed == 0:
                break
    return edges if not additions else np.concatenate([edges, np.asarray(additions, dtype=edges.dtype)], axis=0)


def u_v2_asymmetric_metadata(
    builder: Heat3DGraphBuilder,
    native: Any,
    anchor_graph_coords: np.ndarray,
    query_graph_coords: np.ndarray,
    *,
    numerical_tolerance: float = U_V2_NUMERICAL_TOLERANCE,
    maximum_normalized_overshoot: float = U_V2_MAXIMUM_NORMALIZED_OVERSHOOT,
) -> tuple[Any, dict[str, Any]]:
    """Build the frozen U-v2 output-side R2P graph without changing native graph data."""

    anchor = np.asarray(anchor_graph_coords, dtype=np.float64)
    query = np.asarray(query_graph_coords, dtype=np.float64)
    domain = np.stack((anchor.min(axis=0), anchor.max(axis=0)))
    extent = domain[1] - domain[0]
    if np.any(extent <= 0.0):
        raise ValueError("degenerate native anchor domain")
    # Keep the historical operation order (domain stack, subtraction and
    # division) because KD-tree boundary predicates are sensitive to the last
    # float32 bit even when the mathematical expression is equivalent.
    query_normalized = 2.0 * (query - domain[0]) / (domain[1] - domain[0]) - 1.0
    overshoot = np.maximum(np.maximum(-1.0 - query_normalized, 0.0), np.maximum(query_normalized - 1.0, 0.0))
    maximum = float(np.max(overshoot))
    if not np.all(np.isfinite(query_normalized)):
        raise ValueError("nonfinite U-v2 normalized query coordinates")
    if maximum > float(maximum_normalized_overshoot):
        raise ValueError("U-v2 normalized query overshoot exceeds frozen cap")
    centers = np.asarray(native.x_rnodes)[0, :-1]
    base_radii = np.asarray(native.r_rnodes)[0, :-1]
    impl = builder.builder
    radii = impl._get_effective_support_radii(base_radii, impl.overlap_factor_r2p)
    raw = impl._get_supported_pnodes_by_rnodes(
        centers=centers,
        points=query_normalized,
        radii=radii,
        apply_legacy_hard_reset=(impl.radius_policy == "legacy_kdtree_mean4"),
    )
    repaired = _repair_uncovered(
        np.asarray(raw), centers, query_normalized, impl.min_physical_coverage
    )
    r2p = np.flip(np.asarray(repaired), axis=-1)
    r2p = np.concatenate((r2p, np.asarray([[len(centers), len(query_normalized)]])), axis=0)
    dtype = np.uint16 if max(len(centers) + 1, len(query_normalized) + 1) < np.iinfo(np.uint16).max else np.uint32
    r2p = r2p.astype(dtype)
    x_out = np.concatenate((query_normalized, np.zeros((1, 3), dtype=query_normalized.dtype)))
    metadata = type(native)(
        x_pnodes_inp=native.x_pnodes_inp,
        x_pnodes_out=jnp.asarray(x_out[None, ...]),
        x_rnodes=native.x_rnodes,
        r_rnodes=native.r_rnodes,
        p2r_edge_indices=native.p2r_edge_indices,
        r2r_edge_indices=native.r2r_edge_indices,
        r2r_edge_domains=native.r2r_edge_domains,
        r2p_edge_indices=jnp.asarray(r2p[None, ...]),
    )
    raw_array = np.asarray(raw)
    repaired_array = np.asarray(repaired)
    raw_degree = np.bincount(raw_array[:, 0], minlength=len(query_normalized))
    repaired_degree = np.bincount(repaired_array[:, 0], minlength=len(query_normalized))
    outside = np.any(overshoot > float(numerical_tolerance), axis=1)
    audit = {
        "mode": "u_v2_bounded_query_extrapolation_nearest_r2p_repair",
        "domain_from_native_anchor_only": True,
        "query_coordinates_clamped": False,
        "native_nodes_added": False,
        "native_graph_policy_or_radius_changed": False,
        "numerical_tolerance": float(numerical_tolerance),
        "maximum_normalized_overshoot_cap": float(maximum_normalized_overshoot),
        "maximum_normalized_overshoot": maximum,
        "outside_node_count": int(np.count_nonzero(outside)),
        "raw_uncovered_count": int(np.count_nonzero(raw_degree < impl.min_physical_coverage)),
        "repaired_uncovered_count": int(np.count_nonzero(repaired_degree < impl.min_physical_coverage)),
        "repair_edge_count": int(len(repaired_array) - len(raw_array)),
        "r2p_real_edges": int(len(r2p) - 1),
        "native_exact": {
            name: bool(np.array_equal(np.asarray(getattr(native, name)), np.asarray(getattr(metadata, name))))
            for name in (
                "x_pnodes_inp", "x_rnodes", "r_rnodes", "p2r_edge_indices",
                "r2r_edge_indices", "r2r_edge_domains",
            )
        },
    }
    return metadata, audit


def _dummy_local_p2r(builder: Heat3DGraphBuilder, metadata: Any) -> Any:
    n_p = int(np.asarray(metadata.x_pnodes_out).shape[1] - 1)
    n_r = int(np.asarray(metadata.x_rnodes).shape[1] - 1)
    dtype = np.uint16 if max(n_p + 1, n_r + 1) < np.iinfo(np.uint16).max else np.uint32
    dummy = jnp.asarray(np.asarray([[[n_p, n_r]]], dtype=dtype))
    local = type(metadata)(
        x_pnodes_inp=metadata.x_pnodes_out,
        x_pnodes_out=metadata.x_pnodes_out,
        x_rnodes=metadata.x_rnodes,
        r_rnodes=metadata.r_rnodes,
        p2r_edge_indices=dummy,
        r2r_edge_indices=metadata.r2r_edge_indices,
        r2r_edge_domains=metadata.r2r_edge_domains,
        r2p_edge_indices=None,
    )
    return builder.builder._build_p2r_graph(
        local.x_pnodes_inp, local.x_rnodes, local.p2r_edge_indices, local.r_rnodes
    )


def _combined_edge_targets(
    native: Mapping[str, int | None], query: Mapping[str, int | None]
) -> dict[str, int | None]:
    return {
        "p2r_edge_indices": native["p2r_edge_indices"],
        "r2r_edge_indices": native["r2r_edge_indices"],
        "r2r_edge_domains": native["r2r_edge_domains"],
        "r2p_edge_indices": query["r2p_edge_indices"],
    }


@dataclass(frozen=True)
class UHighNCase:
    resolution: int
    anchor: Any
    query: Any
    support: SupportArtifact
    native_metadata: Any
    query_metadata: Any
    native_group: dict[str, Any]
    query_group: dict[str, Any]
    local_p2r: Any
    audit: dict[str, Any]


@dataclass
class UHighNRuntime:
    """V7 U-v2 runtime with explicit conditioning/query resolution semantics."""

    session: RuntimeSession
    geometry: FullFieldGeometry
    graph_config: dict[str, Any]
    graph_builder_fingerprint: str | None = None

    @classmethod
    def from_session(
        cls,
        session: RuntimeSession,
        geometry: FullFieldGeometry,
        *,
        graph_builder_fingerprint: str | None = None,
    ) -> "UHighNRuntime":
        graph_config = dict(session.graph_config)
        graph_config.update(
            {
                "discrete_graph_backend": "sparse_kdtree_v1",
                "reuse_exact_p2r_for_r2p": True,
                "subsample_factor": 4,
            }
        )
        configured_session = replace(
            session,
            graph_config=graph_config,
            group_builder=GroupBuilder(
                feature_transform=session.feature_transform,
                graph_config=graph_config,
                graph_seed=int(session.run_config.get("graph_seed", 0)),
            ),
        )
        return cls(
            session=configured_session,
            geometry=geometry,
            graph_config=graph_config,
            graph_builder_fingerprint=graph_builder_fingerprint,
        )

    @staticmethod
    def _edge_counts(metadata: Any) -> dict[str, int | None]:
        return {
            field: None if getattr(metadata, field) is None else int(getattr(metadata, field).shape[1])
            for field in EDGE_FIELDS
        }

    def build_case(
        self,
        anchor: Any,
        resolution: int,
        *,
        support: SupportArtifact,
        native_edge_targets: Mapping[str, int | None],
        query_edge_targets: Mapping[str, int | None],
    ) -> UHighNCase:
        resolution = int(resolution)
        if resolution <= 1024 or resolution != len(support.selected_indices):
            raise ValueError("U-v2 requires a high-resolution support/query fixture")
        base = HighNRuntime.from_session(
            self.session,
            self.geometry,
            graph_builder_fingerprint=self.graph_builder_fingerprint,
        )
        anchor_support = base.anchor_support(anchor)
        anchor_hash = base.anchor_support(anchor).descriptor()["selected_indices_sha256"]
        native_record = base.graph_metadata(anchor, support_hash=anchor_hash)
        query = base.query_example(anchor, support)
        builder = Heat3DGraphBuilder(**self.graph_config)
        anchor_graph_coords = self.session.feature_transform.transform(anchor).graph_coords
        query_graph_coords = self.session.feature_transform.transform(query).graph_coords
        query_metadata, audit = u_v2_asymmetric_metadata(
            builder,
            native_record.metadata,
            anchor_graph_coords,
            query_graph_coords,
        )
        combined = _combined_edge_targets(native_edge_targets, query_edge_targets)
        native_compatible = HighNRuntime.compatible_edge_targets(
            native_record.metadata, native_edge_targets
        )
        native_group = self.session.build_group_from_metadata(
            [anchor],
            native_record.metadata,
            name="v7_u_v2_native_conditioning_valid_iid",
            edge_targets=native_compatible,
            context_examples=[anchor],
        )
        query_group = self.session.build_group_from_metadata(
            [query],
            query_metadata,
            name=f"v7_u_v2_direct_query_valid_iid_{resolution}",
            edge_targets=combined,
            context_examples=[anchor],
        )
        audit = dict(audit)
        audit.update(
            {
                "conditioning_resolution": 1024,
                "query_resolution": resolution,
                "direct_query": True,
                "native_edge_counts": self._edge_counts(native_record.metadata),
                "query_edge_counts": self._edge_counts(query_metadata),
                "anchor_support_hash": anchor_hash,
                "query_support_hash": support.descriptor()["selected_indices_sha256"],
            }
        )
        return UHighNCase(
            resolution=resolution,
            anchor=anchor,
            query=query,
            support=support,
            native_metadata=native_record.metadata,
            query_metadata=query_metadata,
            native_group=native_group,
            query_group=query_group,
            local_p2r=_dummy_local_p2r(builder, query_metadata),
            audit=audit,
        )

    def apply(self, case: UHighNCase) -> dict[str, Any]:
        anchor_physics = case.native_group["native_physics"]
        query_physics = case.query_group["native_physics"]
        output = self.session.model.apply(
            {"params": self.session.params},
            inputs_in=case.native_group["inputs"],
            inputs_out=case.query_group["inputs"],
            graphs=case.query_group["graphs"],
            output_local_p2r=case.local_p2r,
            split=True,
            method=_split_native_shape_scale,
            control_volumes=query_physics["control_volumes"],
            log_s_phys=anchor_physics["log_s_phys"],
            reference_temperature=query_physics["reference_temperature"],
            dirichlet_mask=query_physics["dirichlet_mask"],
            prescribed_temperature=query_physics["prescribed_temperature"],
            global_context=case.native_group.get("global_context"),
            qk_region_features=case.native_group.get("qk_region_features"),
            scale_context=case.native_group.get("scale_context"),
            scale_region_source_weights=case.native_group.get("scale_region_source_weights"),
            scale_region_volume_weights=case.native_group.get("scale_region_volume_weights"),
        )
        return output
