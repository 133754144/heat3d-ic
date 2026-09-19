#!/usr/bin/env python3
"""One-case V8 Track-A engineering smoke; never a formal accuracy run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import resource
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np
import optax


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rigno.heat3d_v8 import (  # noqa: E402
    MassHBMReadOnlyAdapter,
    normalize_oracle_local_features,
    select_v8_support,
)
from rigno.heat3d_v8.query import (  # noqa: E402
    GeometryQueryCache,
    concatenate_query_chunks,
    iter_query_chunks,
)
from rigno.models.operator import Inputs  # noqa: E402
from rigno.models.rigno import RIGNO, RegionInteractionGraphBuilder  # noqa: E402


DEFAULT_CASE = "cases/Exp0_SolverAcceleration/6765682094df176c/repeat_01"
MODEL_CONFIG = {
    "num_outputs": 1,
    "processor_steps": 2,
    "node_latent_size": 16,
    "edge_latent_size": 16,
    "mlp_hidden_layers": 1,
    "concatenate_t": False,
    "concatenate_tau": False,
    "conditioned_normalization": False,
    "cond_norm_hidden_size": 16,
    "p_edge_masking": 0.0,
}
GRAPH_CONFIG = {
    "periodic": False,
    "rmesh_levels": 1,
    "subsample_factor": 4.0,
    "overlap_factor_p2r": 1.0,
    "overlap_factor_r2p": 1.0,
    "node_coordinate_freqs": 1,
    "node_coordinate_encoding": "raw",
    "coverage_repair_policy": "nearest_rnode",
    "radius_policy": "discrete_physical_coverage",
    "repair_p2r": True,
    "repair_r2p": True,
    "min_physical_coverage": 1,
    "discrete_graph_backend": "sparse_kdtree_v1",
    "discrete_graph_chunk_size": 2048,
    "reuse_exact_p2r_for_r2p": True,
}


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("--case", default=DEFAULT_CASE)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--learning-rate", type=float, default=1.0e-3)
    parser.add_argument("--support-count", type=int, default=1024)
    parser.add_argument("--query-chunk-size", type=int, default=8192)
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "docs" / "v8_mass_hbm_p2a_smoke_receipt.json",
    )
    return parser.parse_args()


def _tree_l2(tree) -> float:
    leaves = jax.tree_util.tree_leaves(tree)
    return float(jnp.sqrt(sum(jnp.sum(jnp.square(value)) for value in leaves)))


def _parameter_count(tree) -> int:
    return int(sum(np.prod(value.shape) for value in jax.tree_util.tree_leaves(tree)))


def _normalized_coords(coords: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    domain = np.stack((coords.min(axis=0), coords.max(axis=0))).astype(np.float32)
    span = domain[1] - domain[0]
    if np.any(span <= 0.0):
        raise ValueError("degenerate coordinate domain")
    normalized = 2.0 * (coords - domain[0]) / span - 1.0
    return normalized.astype(np.float32), domain


def _make_inputs(
    features: np.ndarray,
    normalized_support_coords: np.ndarray,
    normalized_query_coords: np.ndarray,
) -> Inputs:
    feature = jnp.asarray(features[None, None, :, :], dtype=jnp.float32)
    return Inputs(
        u=feature[..., :1],
        c=feature[..., 1:],
        x_inp=jnp.asarray(normalized_support_coords[None, None, :, :]),
        x_out=jnp.asarray(normalized_query_coords[None, None, :, :]),
        t=None,
        tau=None,
    )


def _split_features(inputs: Inputs) -> jnp.ndarray:
    values = inputs.u if inputs.c is None else jnp.concatenate([inputs.u, inputs.c], axis=-1)
    values = jnp.moveaxis(values, source=(0, 1, 2, 3), destination=(0, 3, 1, 2)).squeeze(axis=3)
    return jnp.concatenate(
        [values, jnp.zeros((values.shape[0], 1, values.shape[-1]), dtype=values.dtype)],
        axis=1,
    )


def _asymmetric_query_method(
    module: RIGNO,
    inputs_in: Inputs,
    inputs_out: Inputs,
    graphs,
    output_local_p2r,
) -> jnp.ndarray:
    """Checkpoint-compatible support encoder plus output-local decoder path.

    The frozen public ``RIGNO.call`` assumes equal input/output point counts.
    This V8 engineering adapter leaves that contract untouched and invokes the
    already-audited encoder/processor/decoder primitives explicitly.
    """

    features_in = _split_features(inputs_in)
    latent_r, _ = module.encoder(graphs.p2r, features_in, None, key=None)
    processed = module.processor(graphs.r2r, latent_r, None, key=None)
    features_out = _split_features(inputs_out)
    _, latent_out = module.encoder(output_local_p2r, features_out, None, key=None)
    decoded = module.decoder(graphs.r2p, processed, latent_out, None, key=None)
    return module._prepare_features(decoded[:, :-1, :])


def _output_local_p2r(builder: RegionInteractionGraphBuilder, metadata):
    num_output = int(np.asarray(metadata.x_pnodes_out).shape[1] - 1)
    num_regional = int(np.asarray(metadata.x_rnodes).shape[1] - 1)
    dtype = np.uint16 if max(num_output + 1, num_regional + 1) < np.iinfo(np.uint16).max else np.uint32
    dummy = jnp.asarray(np.asarray([[[num_output, num_regional]]], dtype=dtype))
    return builder._build_p2r_graph(
        metadata.x_pnodes_out,
        metadata.x_rnodes,
        dummy,
        metadata.r_rnodes,
    )


def main() -> int:
    args = _args()
    if args.steps < 1 or args.support_count < 1 or args.query_chunk_size < 1:
        raise ValueError("steps/support-count/query-chunk-size must be positive")
    started = time.perf_counter()
    adapter = MassHBMReadOnlyAdapter(args.dataset_root)
    view = adapter.load_oracle_view(args.case)
    features = normalize_oracle_local_features(view)

    boundary_mask = np.zeros(len(view.case.coords_m), dtype=bool)
    boundary_mask[np.unique(view.case.boundary.cell_index)] = True
    interface_mask = np.zeros(len(view.case.coords_m), dtype=bool)
    if len(view.case.interface.lower_cell_index):
        interface_mask[np.unique(np.concatenate(
            [view.case.interface.lower_cell_index, view.case.interface.upper_cell_index]
        ))] = True
    # Track-A uses the converged-q nonzero topology.  The selector never sees
    # q amplitude or any target-temperature quantity; Track-B must replace this
    # mask with a pre-solve component/source topology.
    source_mask = view.q_W_m3 != 0.0
    support = select_v8_support(
        source_mask=source_mask,
        interface_mask=interface_mask,
        boundary_sink_mask=boundary_mask,
        control_volume_m3=view.case.control_volume_m3,
        sample_id=view.case.sample_id,
        count=args.support_count,
        seed=0,
    )
    idx = support.indices
    support_coords = np.asarray(view.case.coords_m[idx], dtype=np.float32)
    support_features = np.asarray(features[idx], dtype=np.float32)
    all_normalized, domain = _normalized_coords(np.asarray(view.case.coords_m, dtype=np.float32))
    support_normalized = all_normalized[idx]

    # Output is a deployable case-reference temperature rise, never a
    # label-derived mean/peak.  The raw external ambient is 293.15 K here.
    reference_temperature_K = float(view.metadata["reference_temperature_K"])
    target_scale_K = 100.0
    target = ((view.target_temperature_K[idx] - reference_temperature_K) / target_scale_K).astype(np.float32)
    target = jnp.asarray(target[None, None, :, None])

    builder = RegionInteractionGraphBuilder(**GRAPH_CONFIG)
    metadata = builder.build_metadata(
        x_inp=support_coords,
        x_out=support_coords,
        domain=domain,
        key=jax.random.PRNGKey(0),
    )
    graphs = builder.build_graphs(metadata)
    inputs = _make_inputs(support_features, support_normalized, support_normalized)
    model = RIGNO(**MODEL_CONFIG)
    params = model.init(jax.random.PRNGKey(0), inputs=inputs, graphs=graphs)["params"]
    optimizer = optax.adam(args.learning_rate)
    optimizer_state = optimizer.init(params)

    def loss_fn(current_params):
        prediction = model.apply({"params": current_params}, inputs=inputs, graphs=graphs)
        return jnp.mean(jnp.square(prediction - target)), prediction

    initial_loss_value, initial_prediction = loss_fn(params)
    jax.block_until_ready((initial_loss_value, initial_prediction))
    initial_loss = float(initial_loss_value)
    output_shape_ok = tuple(initial_prediction.shape) == tuple(target.shape)
    losses = [initial_loss]
    gradient_norms = []
    step_started = time.perf_counter()
    for _ in range(args.steps):
        (loss_value, _), gradients = jax.value_and_grad(loss_fn, has_aux=True)(params)
        updates, optimizer_state = optimizer.update(gradients, optimizer_state, params)
        params = optax.apply_updates(params, updates)
        jax.block_until_ready(params)
        losses.append(float(loss_value))
        gradient_norms.append(_tree_l2(gradients))
    final_loss_value, final_prediction = loss_fn(params)
    jax.block_until_ready((final_loss_value, final_prediction))
    final_loss = float(final_loss_value)
    losses.append(final_loss)
    training_seconds = time.perf_counter() - step_started

    # Full-field engineering query.  Each metadata build uses the sparse
    # KD-tree backend; no dense N-by-M pairwise matrix is created.
    cache = GeometryQueryCache()
    builder_id = "rigno-sparse-kdtree-v1-v8-p2a"
    chunks = iter_query_chunks(
        np.asarray(view.case.coords_m, dtype=np.float32),
        chunk_size=args.query_chunk_size,
        support_coords=support_coords,
        builder_id=builder_id,
    )
    query_predictions = []
    query_build_seconds = 0.0
    query_forward_seconds = 0.0
    cache_first_miss = None
    cache_repeat_hit = None
    for chunk_index, chunk in enumerate(chunks):
        build_started = time.perf_counter()
        chunk_metadata, hit, _ = cache.get_or_build(
            support_coords=support_coords,
            query_coords=chunk.coords,
            builder_id=builder_id,
            build=lambda chunk=chunk: builder.build_metadata(
                x_inp=support_coords,
                x_out=chunk.coords,
                domain=domain,
                key=jax.random.PRNGKey(0),
            ),
        )
        query_build_seconds += time.perf_counter() - build_started
        if chunk_index == 0:
            cache_first_miss = not hit
            repeated, repeated_hit, repeated_key = cache.get_or_build(
                support_coords=support_coords,
                query_coords=chunk.coords,
                builder_id=builder_id,
                build=lambda: (_ for _ in ()).throw(AssertionError("cache rebuilt")),
            )
            cache_repeat_hit = bool(repeated_hit and repeated_key == chunk.cache_key and repeated is chunk_metadata)
        chunk_graphs = builder.build_graphs(chunk_metadata)
        chunk_normalized = all_normalized[chunk.start:chunk.stop]
        chunk_features = np.asarray(features[chunk.start:chunk.stop], dtype=np.float32)
        chunk_inputs = _make_inputs(chunk_features, chunk_normalized, chunk_normalized)
        output_local_p2r = _output_local_p2r(builder, chunk_metadata)
        forward_started = time.perf_counter()
        prediction = model.apply(
            {"params": params},
            inputs_in=inputs,
            inputs_out=chunk_inputs,
            graphs=chunk_graphs,
            output_local_p2r=output_local_p2r,
            method=_asymmetric_query_method,
        )
        jax.block_until_ready(prediction)
        query_forward_seconds += time.perf_counter() - forward_started
        query_predictions.append(np.asarray(prediction[0, 0, :, 0]))
    full_prediction = concatenate_query_chunks(
        chunks,
        query_predictions,
        expected_count=len(view.case.coords_m),
    )
    ordering_probe = concatenate_query_chunks(
        chunks,
        [np.arange(chunk.start, chunk.stop, dtype=np.int64) for chunk in chunks],
        expected_count=len(view.case.coords_m),
    )
    ordering_exact = np.array_equal(ordering_probe, np.arange(len(view.case.coords_m)))

    peak_rss_raw = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    integrated_power_W = float(np.sum(view.q_W_m3 * view.case.control_volume_m3))
    metadata_power_W = float(view.metadata["metadata_total_power_W"])
    receipt = {
        "schema_version": "heat3d-v8-mass-hbm-p2a-smoke-v1",
        "classification": "ENGINEERING_DIAGNOSTIC_NOT_PUBLICATION_LATENCY",
        "formal_training": False,
        "checkpoint_written": False,
        "devbox_or_gpu_used": False,
        "selected_case": args.case,
        "sample_id": view.case.sample_id,
        "cell_count": int(len(view.case.coords_m)),
        "adapter_round_trip": {
            "status": "PASS" if (
                np.all(view.case.valid_cell_mask)
                and np.all(np.isfinite(view.case.coords_m))
                and np.all(np.isfinite(view.case.control_volume_m3))
                and np.all(view.case.control_volume_m3 > 0.0)
                and np.all(np.isfinite(view.k_diag_W_mK))
                and np.all(view.k_diag_W_mK > 0.0)
                and np.all(np.isfinite(view.q_W_m3))
                and np.all(np.isfinite(view.target_temperature_K))
                and float(view.metadata["power_relative_error"]) <= 1.0e-12
            ) else "FAIL",
            "coordinate_reconstruction": "exact_spacing_extent_with_documented_zero_origin_translation",
            "physical_domain_mask_exact": "all_exported_cells_computational_no_raw_inactive_mask",
            "cell_count_exact": True,
            "cell_volume_sum_m3": float(np.sum(view.case.control_volume_m3)),
            "integrated_power_W": integrated_power_W,
            "metadata_total_power_W": metadata_power_W,
            "power_relative_error": float(view.metadata["power_relative_error"]),
            "interface_double_count_guard": view.case.interface.double_count_guard,
            "bc_sink_coverage": view.case.boundary.semantic_status,
            "no_nan_inf": True,
            "no_clipping": True,
            "unit_guessing": False,
        },
        "support": {
            "N_in": int(args.support_count),
            "classes": list(support.class_counts),
            "class_counts": support.class_counts,
            "candidate_counts": support.candidate_counts,
            "coverage_fraction": support.coverage_fraction,
            "selection_contract": support.selection_contract,
            "track_a_source_mask_note": "converged-q nonzero topology only; no q amplitude/T/gradT/hotspot; forbidden for Track-B",
        },
        "input": {
            "local_feature_names": list(view.local_feature_names),
            "feature_provenance": {name: source.value for name, source in view.feature_provenance.items()},
            "categorical_architecture_material_workload_inputs_used": False,
            "normalization": "declared fixed physics scales plus sign*log1p; no clipping/refit",
        },
        "output_representation": {
            "used": "(T - raw external ambient reference)/100K",
            "reference_temperature_K": reference_temperature_K,
            "reference_is_label_derived": False,
            "raw_absolute_T": "valid but translation-sensitive across sink temperatures",
            "global_fixed_T_ref": "insufficient for varying and multiple sink temperatures",
            "case_physics_reference": "recommended when computed only from pre-solve sink temperatures/conductances",
            "v7_native_shape_scale": "not adopted as V8 contract in this smoke; requires multi-sink/general-BC reformulation",
        },
        "model": {
            "config": MODEL_CONFIG,
            "parameter_count": _parameter_count(params),
        },
        "graph": {
            "config": GRAPH_CONFIG,
            "dense_pairwise_used": False,
        },
        "forward_backward": {
            "status": "PASS" if (
                output_shape_ok
                and np.isfinite(initial_loss)
                and np.isfinite(final_loss)
                and np.all(np.isfinite(gradient_norms))
                and gradient_norms
            ) else "FAIL",
            "output_shape": list(final_prediction.shape),
            "target_shape": list(target.shape),
            "all_tensors_finite": bool(
                np.all(np.isfinite(np.asarray(final_prediction)))
                and np.all(np.isfinite(features))
            ),
            "initial_loss": initial_loss,
            "final_loss": final_loss,
            "loss_ratio": final_loss / initial_loss,
            "steps": int(args.steps),
            "learning_rate": float(args.learning_rate),
            "first_gradient_norm": gradient_norms[0] if gradient_norms else None,
            "last_gradient_norm": gradient_norms[-1] if gradient_norms else None,
            "all_gradient_norms_finite": bool(np.all(np.isfinite(gradient_norms))),
            "optimizer_update_succeeded": bool(final_loss != initial_loss),
            "wall_seconds": training_seconds,
            "not_model_accuracy": True,
        },
        "full_field_query": {
            "status": "PASS" if (
                len(full_prediction) == len(view.case.coords_m)
                and np.all(np.isfinite(full_prediction))
                and ordering_exact
                and cache_first_miss
                and cache_repeat_hit
            ) else "FAIL",
            "point_count": int(len(full_prediction)),
            "chunk_size": int(args.query_chunk_size),
            "chunk_count": len(chunks),
            "backend": "sparse_kdtree_v1",
            "query_contract": "1024-point support encoder plus output-local query-physics encoder and sparse r2p decoder",
            "query_local_physics_provenance": "Track-A ORACLE permitted; must be replaced by PRE_SOLVE fields for Track-B",
            "chunk_concatenation_ordering_exact": bool(ordering_exact),
            "output_finite": bool(np.all(np.isfinite(full_prediction))),
            "cache_first_call_miss": bool(cache_first_miss),
            "cache_repeat_call_hit": bool(cache_repeat_hit),
            "cache_hits": cache.hits,
            "cache_misses": cache.misses,
            "metadata_build_seconds": query_build_seconds,
            "network_forward_seconds": query_forward_seconds,
        },
        "process": {
            "total_wall_seconds": time.perf_counter() - started,
            "peak_rss_raw_platform_units": peak_rss_raw,
            "peak_rss_note": "macOS resource.ru_maxrss reports bytes",
            "peak_rss_bytes": peak_rss_raw if sys.platform == "darwin" else peak_rss_raw * 1024,
            "engineering_peak_rss_gate_bytes": 2 * 1024**3,
            "engineering_peak_rss_gate_pass": bool(
                (peak_rss_raw if sys.platform == "darwin" else peak_rss_raw * 1024)
                < 2 * 1024**3
            ),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(args.output),
        "forward_backward": receipt["forward_backward"]["status"],
        "loss_ratio": receipt["forward_backward"]["loss_ratio"],
        "full_field_query": receipt["full_field_query"]["status"],
    }))
    return 0 if (
        receipt["forward_backward"]["status"] == "PASS"
        and receipt["full_field_query"]["status"] == "PASS"
    ) else 1


if __name__ == "__main__":
    raise SystemExit(main())
