#!/usr/bin/env python3
"""Run one preregistered P5-R adaptive resolution cell on frozen valid32.

Accuracy and latency are produced by the same execution.  Historical artifacts
are used only for the fixed padding envelope and for exact-support qualification.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from typing import Any, Mapping

import h5py
import jax
import jax.numpy as jnp
import numpy as np


ROOT = Path(os.environ.get("HEAT3D_REPO_ROOT", Path(__file__).resolve().parents[1])).resolve()
for value in (ROOT, ROOT / "scripts"):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))

import benchmark_heat3d_v6_inference_qualification as qualification  # noqa: E402
import benchmark_heat3d_v6_p1i_p5_a4_p2r_r2p as a4  # noqa: E402
import run_heat3d_v1_medium_controlled_training_export as legacy  # noqa: E402
import run_heat3d_v6_p1i_anchor_high_n_development as highn  # noqa: E402
import run_heat3d_v6_p1i_graph_scale_candidate as candidate  # noqa: E402
from rigno.graphBuilder_Heat3D import Heat3DGraphBuilder  # noqa: E402
from rigno.heat3d_v6_full_field import (  # noqa: E402
    build_reconstruction_map,
    prepare_reconstruction_domain_partition,
)
from rigno.heat3d_v6_gpu_reconstruction import to_device_reconstruction_map  # noqa: E402
from rigno.heat3d_v6_p1i_anchor_query import (  # noqa: E402
    array_sha256,
    conservative_selected_control_volume,
    deterministic_nested_query_prefix,
    prepare_nested_query_geometry_cache,
)
from rigno.models.rigno import RIGNO as GraphNeuralOperator  # noqa: E402


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _stats(values: list[float]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(len(array)),
        "median_seconds": float(np.median(array)),
        "mean_seconds": float(np.mean(array)),
        "p95_seconds": float(np.quantile(array, 0.95)),
    }


def _tree_sha256(value: Any) -> str:
    digest = hashlib.sha256()
    leaves, treedef = jax.tree_util.tree_flatten(value)
    digest.update(str(treedef).encode())
    for leaf in leaves:
        array = np.ascontiguousarray(np.asarray(leaf))
        digest.update(str(array.dtype).encode()); digest.update(str(array.shape).encode()); digest.update(array.tobytes())
    return digest.hexdigest()


def _block_tree(tree: Any) -> None:
    jax.tree_util.tree_map(
        lambda value: value.block_until_ready() if hasattr(value, "block_until_ready") else value,
        tree,
    )


def _edge_targets(path: Path) -> dict[str, int | None]:
    payload = json.loads(path.read_text())
    if "padding" in payload and "actual_padding_envelope" in payload["padding"]:
        envelope = payload["padding"]["actual_padding_envelope"]
        if "query" in envelope:
            return {
                key: (None if value is None else int(value))
                for key, value in envelope["query"].items()
            }
    if "padding_envelopes" in payload and "query" in payload["padding_envelopes"]:
        return {
            key: (None if value is None else int(value))
            for key, value in payload["padding_envelopes"]["query"].items()
        }
    if "graph_cache" in payload and "edge_targets" in payload["graph_cache"]:
        return {
            key: (None if value is None else int(value))
            for key, value in payload["graph_cache"]["edge_targets"].items()
        }
    rows = payload.get("graph_metadata_artifacts", [])
    if not rows:
        raise RuntimeError(f"padding source has no edge envelope: {path}")
    targets: dict[str, int | None] = {}
    for field in qualification.EDGE_FIELDS:
        sizes = []
        for row in rows:
            with np.load(row["path"], allow_pickle=False) as archive:
                none_fields = {
                    str(value) for value in np.asarray(archive["__none_fields_utf8"]).tolist()
                }
                if field not in none_fields:
                    sizes.append(int(np.asarray(archive[field]).shape[1]))
        targets[field] = max(sizes) if sizes else None
    return targets


def _compatible_targets(
    targets: Mapping[str, int | None], metadata: Any,
) -> dict[str, int | None]:
    """Preserve the envelope while respecting exact reverse-edge omission."""
    return {
        field: None if getattr(metadata, field) is None else targets[field]
        for field in qualification.EDGE_FIELDS
    }


def _builder(
    route: Mapping[str, Any], runtime: Mapping[str, Any], anchor: Any, graph_key: Any,
) -> Heat3DGraphBuilder:
    config = dict(runtime["graph_config"])
    resolution = int(route["resolution"])
    if route["regional_policy"] == "native_factor4":
        factor = 4.0
    elif route["regional_policy"] == "factor8":
        factor = 8.0
    elif route["regional_policy"] == "fixed_Nr256":
        factor = resolution / 256.0
    else:
        raise RuntimeError("unknown P5-R regional policy")
    config.update(
        subsample_factor=factor,
        discrete_graph_backend="sparse_kdtree_v1",
        reuse_exact_p2r_for_r2p=True,
    )
    return Heat3DGraphBuilder(**config)


def _runtime(args: argparse.Namespace) -> dict[str, Any]:
    checkpoint_path = args.run_dir / "params_best_valid_point_global.pkl"
    if _sha256(checkpoint_path) != args.checkpoint_sha256:
        raise RuntimeError("P5-R checkpoint SHA drifted")
    checkpoint = legacy._load_params_checkpoint(checkpoint_path)
    run_config = json.loads((args.run_dir / "run_config.json").read_text())
    stats = highn.common._materialize_checkpoint_stats(checkpoint["train_only_normalization"])
    highn.install_checkpoint_feature_hooks(stats)
    model_config = legacy._resolve_decoder_bypass_model_config(dict(checkpoint["model_config"]), stats)
    legacy._validate_model_config(model_config)
    graph_config = dict(run_config["graph_config"])
    graph_config["discrete_graph_backend"] = "sparse_kdtree_v1"
    graph_config = dict(Heat3DGraphBuilder(**graph_config).config)
    return {
        "checkpoint": checkpoint,
        "run_config": run_config,
        "stats": stats,
        "model_config": model_config,
        "graph_config": graph_config,
    }


def _parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    for name in (
        "protocol", "binding", "artifact_root", "dataset_root", "manifest",
        "full_fields", "run_dir", "padding_result", "native_padding_result", "output",
    ):
        parser.add_argument(f"--{name.replace('_', '-')}", dest=name, type=Path, required=True)
    parser.add_argument("--route", required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--sample-count", type=int, choices=[1, 32, 96], default=32)
    parser.add_argument("--prediction-output", type=Path)
    parser.add_argument("--population-mode", choices=["frozen_valid32", "remaining_valid96"], default="frozen_valid32")
    parser.add_argument("--population-preflight", type=Path)
    parser.add_argument("--order-seed", type=int)
    parser.add_argument("--timing-repeats", type=int, default=20)
    return parser.parse_args()


def main() -> int:
    args = _parse()
    if jax.devices()[0].platform != "gpu":
        raise RuntimeError("P5-R formal sweep requires the frozen GPU backend")
    protocol = json.loads(args.protocol.read_text())
    if protocol["status"] != "preregistered_before_execution":
        raise RuntimeError("P5-R protocol is not preregistered")
    routes = {row["route"]: row for row in protocol["cells"]}
    if args.route not in routes:
        raise RuntimeError("P5-R route is not registered")
    route = routes[args.route]
    resolution = int(route["resolution"])
    direct = route["output_mode"] == "direct"
    if direct != (resolution == 240825):
        raise RuntimeError("P5-R direct/reconstruction contract drifted")

    binding = json.loads(args.binding.read_text())
    runtime = _runtime(args)
    dataset = highn._dataset(args)
    if args.population_mode == "frozen_valid32":
        anchors = highn._valid_examples(dataset, binding)
        expected_count = 32
        preflight_path = args.artifact_root / "actual_data_preflight.json"
    else:
        ordered_ids = sorted(dataset.split_ids["valid_iid"], key=lambda value: hashlib.sha256(value.encode()).hexdigest())
        valid32 = binding["development_subset"]["sample_ids"]
        if ordered_ids[:32] != valid32:
            raise RuntimeError("valid32 is not the frozen subset of formal valid128")
        index = dataset.sample_index_by_id()
        anchors = [dataset[index[sample_id]] for sample_id in ordered_ids[32:]]
        expected_count = 96
        if args.population_preflight is None:
            raise RuntimeError("remaining_valid96 requires --population-preflight")
        preflight_path = args.population_preflight
    if len(anchors) != expected_count:
        raise RuntimeError(f"P5-R population count drifted: {len(anchors)} != {expected_count}")
    preflight = json.loads(preflight_path.read_text())
    if preflight["sample_ids"] != [anchor.sample_id for anchor in anchors]:
        raise RuntimeError("P5-R population order drifted")
    anchors = anchors[: args.sample_count]
    if args.order_seed is not None:
        order = np.random.default_rng(args.order_seed).permutation(len(anchors))
        anchors = [anchors[int(index)] for index in order]
    frozen_supports = {
        row["sample_id"]: row for row in preflight.get("supports", {}).get(str(resolution), [])
    }
    full, archive_lookup = highn._full_shared(args)
    coords = np.asarray(full["coords"], dtype=np.float64)
    cv = np.asarray(full["cv"], dtype=np.float64)
    layer = np.asarray(full["layer"], dtype=np.int32)
    boundaries = highn._boundaries(anchors[0], float(np.min(coords[:, 2])))
    geometry = prepare_nested_query_geometry_cache(
        full_coords=coords, full_control_volume=cv, full_layer_id=layer,
        layer_boundaries_m=boundaries,
    )
    reconstruction_partition = prepare_reconstruction_domain_partition(
        coords=coords, layer_id=layer, boundaries=boundaries,
    )
    physics_rows = {row["sample_id"]: row for row in preflight["samples"]}
    graph_key = highn.runner._metadata_key(int(runtime["run_config"]["graph_seed"]))
    query_targets = _edge_targets(args.padding_result)
    anchor_targets = _edge_targets(args.native_padding_result)
    model = GraphNeuralOperator(**runtime["model_config"])
    params = highn.runner._device_params(runtime["checkpoint"]["params"])

    @jax.jit
    def anchor_forward(model_params: Any, group: Any) -> tuple[Any, Any]:
        output = highn.runner._model_apply(model, model_params, group)
        return output["raw_temperature"][0, 0, :, 0], output["s_hat"].reshape(-1)[0]

    @jax.jit
    def query_forward(
        model_params: Any, group: Any, weights: Any, anchor_scale: Any,
    ) -> Any:
        output = highn.runner._model_apply(model, model_params, group)
        delta = output["raw_temperature"][0, 0, :, 0] - highn.REFERENCE_K
        normalized = weights / jnp.sum(weights)
        query_scale = jnp.sqrt(jnp.sum(normalized * delta * delta))
        return delta / query_scale * anchor_scale

    rows: list[dict[str, Any]] = []
    support_metric_rows: list[dict[str, Any]] = []
    full_metric_rows: list[dict[str, Any]] = []
    oracle_metric_rows: list[dict[str, Any]] = []
    full_predictions: list[np.ndarray] = []
    stage_values: dict[str, list[float]] = {
        key: [] for key in (
            "support_plus_cv", "graph", "reconstruction_map", "group_and_h2d",
            "anchor1024", "forward", "reconstruction", "matched_continuous_e2e",
        )
    }
    compile_done = False
    peak_vram = 0
    resident_payload = None
    service_started = time.perf_counter()
    completion_offsets: list[float] = []

    with h5py.File(args.full_fields, "r") as temperature_archive:
        for number, anchor in enumerate(anchors, start=1):
            submit_offset = time.perf_counter() - service_started
            physics_path = Path(physics_rows[anchor.sample_id]["physics_cache_file"])
            with np.load(physics_path, allow_pickle=False) as physics:
                full_k = np.asarray(physics["k_xyz"], dtype=np.float64)
                full_q = np.asarray(physics["q_W_m3"], dtype=np.float64)
            truth = np.asarray(
                temperature_archive["samples/deltaT_K"][archive_lookup[anchor.sample_id]],
                dtype=np.float64,
            )
            anchor_indices, anchor_distance = highn._anchor_indices(
                anchor, coords,
                float(binding["numeric_tolerances"]["anchor_to_solver_coordinate_max_distance_m"]),
            )
            anchor_support = {
                "selected_indices": anchor_indices,
                "operator_control_volume": np.asarray(anchor.operator_point_weights, dtype=np.float64),
                "k_xyz": np.asarray(anchor.condition.condition_features[:, :3], dtype=np.float64),
                "q_W_m3": np.asarray(anchor.condition.condition_features[:, 3], dtype=np.float64),
                "layer_id": layer[anchor_indices],
            }

            total_started = time.perf_counter()
            phase = time.perf_counter()
            if resolution == 1024:
                selected = anchor_indices
                selected_cv = np.asarray(anchor.operator_point_weights, dtype=np.float64)
            elif direct:
                selected = np.arange(len(coords), dtype=np.int64)
                selected_cv = cv
            else:
                selected, _ = deterministic_nested_query_prefix(
                    sample_id=anchor.sample_id,
                    anchor_indices=anchor_indices,
                    full_q=full_q,
                    target_count=resolution,
                    geometry_cache=geometry,
                )
                selected_cv, _ = conservative_selected_control_volume(
                    full_coords=coords,
                    full_control_volume=cv,
                    full_layer_id=layer,
                    selected_indices=selected,
                )
            support_seconds = time.perf_counter() - phase
            query_support = {
                "selected_indices": selected,
                "operator_control_volume": selected_cv,
                "k_xyz": full_k[selected] if resolution != 1024 else anchor_support["k_xyz"],
                "q_W_m3": full_q[selected] if resolution != 1024 else anchor_support["q_W_m3"],
                "layer_id": layer[selected],
            }
            if resolution != 1024 and not direct:
                frozen = highn._load_support(Path(frozen_supports[anchor.sample_id]["support_file"]))
                exact_support = (
                    np.array_equal(selected, np.asarray(frozen["selected_indices"]))
                    and np.array_equal(selected_cv, np.asarray(frozen["operator_control_volume"]))
                    and np.array_equal(query_support["k_xyz"], np.asarray(frozen["k_xyz"]))
                    and np.array_equal(query_support["q_W_m3"], np.asarray(frozen["q_W_m3"]))
                )
            else:
                exact_support = True
            # The frozen high-N full-field representation uses fractional
            # overlap while native1024 inputs use the generator's binary-mask
            # fields.  Their overlap-node drift is a report-only diagnostic,
            # not a hard gate.  The gate instead proves that the anchor forward
            # receives the original checkpoint-IID k/q/CV without adaptation.
            anchor_exact = (
                anchor_distance == 0.0
                and np.array_equal(
                    anchor_support["k_xyz"],
                    np.asarray(anchor.condition.condition_features[:, :3], dtype=np.float64),
                )
                and np.array_equal(
                    anchor_support["q_W_m3"],
                    np.asarray(anchor.condition.condition_features[:, 3], dtype=np.float64),
                )
                and np.array_equal(
                    anchor_support["operator_control_volume"],
                    np.asarray(anchor.operator_point_weights, dtype=np.float64),
                )
            )
            if not exact_support or not anchor_exact:
                raise RuntimeError(f"P5-R support hard gate failed: {args.route}/{anchor.sample_id}")

            anchor_example = highn._query_example(anchor, anchor_support, coords)
            query_example = highn._query_example(anchor, query_support, coords)
            phase = time.perf_counter()
            anchor_builder = Heat3DGraphBuilder(**dict(
                runtime["graph_config"], subsample_factor=4.0,
                discrete_graph_backend="sparse_kdtree_v1", reuse_exact_p2r_for_r2p=True,
            ))
            query_builder = _builder(route, runtime, anchor, graph_key)
            anchor_metadata = anchor_builder.build_metadata(
                highn.runner._graph_coords_for_example(anchor_example, runtime["stats"]), key=graph_key,
            )
            query_metadata = query_builder.build_metadata(
                highn.runner._graph_coords_for_example(query_example, runtime["stats"]), key=graph_key,
            )
            _block_tree((anchor_metadata, query_metadata))
            graph_seconds = time.perf_counter() - phase
            compatible_anchor_targets = _compatible_targets(anchor_targets, anchor_metadata)
            compatible_query_targets = _compatible_targets(query_targets, query_metadata)
            phase = time.perf_counter()
            mapping = None
            if not direct:
                mapping, _ = build_reconstruction_map(
                    coords=coords, layer_id=layer, boundaries=boundaries,
                    support_indices=selected, empty_domain_fallback="same_layer",
                    prepared_partition=reconstruction_partition, query_workers=-1,
                )
            map_seconds = time.perf_counter() - phase

            phase = time.perf_counter()
            anchor_group = highn._model_group(highn._prepare_group(
                example=anchor_example, anchor=anchor, runtime=runtime, builder=anchor_builder,
                metadata=anchor_metadata, edge_targets=compatible_anchor_targets,
            ))
            query_group = highn._model_group(highn._prepare_group(
                example=query_example, anchor=anchor, runtime=runtime, builder=query_builder,
                metadata=query_metadata, edge_targets=compatible_query_targets,
            ))
            device_map = None if mapping is None else to_device_reconstruction_map(mapping)
            anchor_group, query_group, device_weights = jax.device_put((
                anchor_group, query_group, np.asarray(selected_cv, dtype=np.float32),
            ))
            jax.block_until_ready(device_weights)
            group_seconds = time.perf_counter() - phase

            if not compile_done:
                anchor_raw_warm, anchor_scale_warm = anchor_forward(params, anchor_group)
                jax.block_until_ready(anchor_raw_warm)
                query_warm = query_forward(params, query_group, device_weights, anchor_scale_warm)
                jax.block_until_ready(query_warm)
                if device_map is not None:
                    jax.block_until_ready(device_map.reconstruct(query_warm))
                compile_done = True
                # Qualification/JIT warmup is outside both fresh and streaming
                # service timing. Reset before replaying the first real case.
                service_started = time.perf_counter()
                submit_offset = 0.0
                completion_offsets.clear()
                # Repeat the complete first-case preprocessing after JIT warm-up;
                # otherwise its continuous span would silently omit CPU stages.
                total_started = time.perf_counter()
                phase = time.perf_counter()
                if resolution == 1024:
                    selected = anchor_indices
                    selected_cv = np.asarray(anchor.operator_point_weights, dtype=np.float64)
                elif direct:
                    selected = np.arange(len(coords), dtype=np.int64)
                    selected_cv = cv
                else:
                    selected, _ = deterministic_nested_query_prefix(
                        sample_id=anchor.sample_id, anchor_indices=anchor_indices,
                        full_q=full_q, target_count=resolution, geometry_cache=geometry,
                    )
                    selected_cv, _ = conservative_selected_control_volume(
                        full_coords=coords, full_control_volume=cv, full_layer_id=layer,
                        selected_indices=selected,
                    )
                support_seconds = time.perf_counter() - phase
                query_support = {
                    "selected_indices": selected,
                    "operator_control_volume": selected_cv,
                    "k_xyz": full_k[selected] if resolution != 1024 else anchor_support["k_xyz"],
                    "q_W_m3": full_q[selected] if resolution != 1024 else anchor_support["q_W_m3"],
                    "layer_id": layer[selected],
                }
                anchor_example = highn._query_example(anchor, anchor_support, coords)
                query_example = highn._query_example(anchor, query_support, coords)
                phase = time.perf_counter()
                anchor_builder = Heat3DGraphBuilder(**dict(
                    runtime["graph_config"], subsample_factor=4.0,
                    discrete_graph_backend="sparse_kdtree_v1",
                    reuse_exact_p2r_for_r2p=True,
                ))
                query_builder = _builder(route, runtime, anchor, graph_key)
                anchor_metadata = anchor_builder.build_metadata(
                    highn.runner._graph_coords_for_example(anchor_example, runtime["stats"]), key=graph_key,
                )
                query_metadata = query_builder.build_metadata(
                    highn.runner._graph_coords_for_example(query_example, runtime["stats"]), key=graph_key,
                )
                _block_tree((anchor_metadata, query_metadata))
                graph_seconds = time.perf_counter() - phase
                compatible_anchor_targets = _compatible_targets(anchor_targets, anchor_metadata)
                compatible_query_targets = _compatible_targets(query_targets, query_metadata)
                phase = time.perf_counter()
                mapping = None
                if not direct:
                    mapping, _ = build_reconstruction_map(
                        coords=coords, layer_id=layer, boundaries=boundaries,
                        support_indices=selected, empty_domain_fallback="same_layer",
                        prepared_partition=reconstruction_partition, query_workers=-1,
                    )
                map_seconds = time.perf_counter() - phase
                phase = time.perf_counter()
                anchor_group = highn._model_group(highn._prepare_group(
                    example=anchor_example, anchor=anchor, runtime=runtime,
                    builder=anchor_builder, metadata=anchor_metadata,
                    edge_targets=compatible_anchor_targets,
                ))
                query_group = highn._model_group(highn._prepare_group(
                    example=query_example, anchor=anchor, runtime=runtime,
                    builder=query_builder, metadata=query_metadata,
                    edge_targets=compatible_query_targets,
                ))
                device_map = None if mapping is None else to_device_reconstruction_map(mapping)
                anchor_group, query_group, device_weights = jax.device_put((
                    anchor_group, query_group, np.asarray(selected_cv, dtype=np.float32),
                ))
                jax.block_until_ready(device_weights)
                group_seconds = time.perf_counter() - phase

            phase = time.perf_counter()
            anchor_raw, anchor_scale = anchor_forward(params, anchor_group)
            jax.block_until_ready(anchor_raw)
            anchor_seconds = time.perf_counter() - phase
            phase = time.perf_counter()
            if resolution == 1024:
                support_delta = anchor_raw - highn.REFERENCE_K
            else:
                support_delta = query_forward(params, query_group, device_weights, anchor_scale)
            jax.block_until_ready(support_delta)
            forward_seconds = time.perf_counter() - phase
            phase = time.perf_counter()
            if device_map is None:
                full_delta = support_delta
            else:
                full_delta = device_map.reconstruct(support_delta)
            jax.block_until_ready(full_delta)
            reconstruction_seconds = time.perf_counter() - phase
            continuous_seconds = time.perf_counter() - total_started
            completion_offset = time.perf_counter() - service_started
            completion_offsets.append(completion_offset)
            if resident_payload is None:
                resident_payload = (anchor_group, query_group, device_weights, anchor_scale, device_map)

            support_np = np.asarray(support_delta, dtype=np.float64)
            full_np = np.asarray(full_delta, dtype=np.float64)
            if not np.all(np.isfinite(full_np)):
                raise RuntimeError(f"P5-R nonfinite prediction: {args.route}/{anchor.sample_id}")
            support_row = highn._metric_row(
                support_np, truth[selected], selected_cv, coords[selected], layer[selected], full_q[selected],
            )
            full_row = highn._metric_row(full_np, truth, cv, coords, layer, full_q)
            support_metric_rows.append(support_row)
            full_metric_rows.append(full_row)
            full_predictions.append(full_np.astype(np.float32, copy=False))
            if mapping is None:
                oracle_full = truth
            else:
                oracle_full = np.sum(
                    truth[selected][np.asarray(mapping.neighbor_local_indices)]
                    * np.asarray(mapping.neighbor_weights), axis=1,
                )
            oracle_metric_rows.append(highn._metric_row(
                oracle_full, truth, cv, coords, layer, full_q,
            ))
            device_memory = candidate.publication._device_memory()
            peak_vram = max(peak_vram, int(device_memory.get("peak_bytes_in_use", 0)))
            stages = {
                "support_plus_cv": support_seconds,
                "graph": graph_seconds,
                "reconstruction_map": map_seconds,
                "group_and_h2d": group_seconds,
                "anchor1024": anchor_seconds,
                "forward": forward_seconds,
                "reconstruction": reconstruction_seconds,
                "matched_continuous_e2e": continuous_seconds,
            }
            for key, value in stages.items():
                stage_values[key].append(value)
            rows.append({
                "sample_id": anchor.sample_id,
                "support_hash": array_sha256(selected),
                "support_exact": exact_support,
                "anchor_k_q_cv_exact": anchor_exact,
                "full_field_at_anchor_representation_drift_report_only": {
                    "delta_k_max_abs": float(np.max(np.abs(
                        full_k[anchor_indices] - anchor_support["k_xyz"]
                    ))),
                    "delta_q_max_abs": float(np.max(np.abs(
                        full_q[anchor_indices] - anchor_support["q_W_m3"]
                    ))),
                    "delta_cv_max_abs": float(np.max(np.abs(
                        cv[anchor_indices] - anchor_support["operator_control_volume"]
                    ))),
                },
                "selected_volume_m3": float(np.sum(selected_cv)),
                "full_volume_m3": float(np.sum(cv)),
                "selected_power_W": float(np.sum(query_support["q_W_m3"] * selected_cv)),
                "full_power_W": float(np.sum(full_q * cv)),
                "graph_hash": a4._canonical_hash(query_metadata),
                "anchor_group_sha256": _tree_sha256(anchor_group),
                "query_group_sha256": _tree_sha256(query_group),
                "input_physics_context_sha256": _tree_sha256((anchor_group, query_group, device_weights)),
                "regional_node_count": int(np.asarray(query_metadata.x_rnodes).shape[1] - 1),
                "p2r_edges": int(np.asarray(query_metadata.p2r_edge_indices).shape[1]),
                "r2r_edges": int(np.asarray(query_metadata.r2r_edge_indices).shape[1]),
                "timing": stages,
                "streaming": {
                    "submit_offset_seconds": submit_offset,
                    "completion_offset_seconds": completion_offset,
                    "submit_to_result_seconds": completion_offset - submit_offset,
                    "inter_completion_seconds": completion_offset if len(completion_offsets) == 1 else completion_offset - completion_offsets[-2],
                },
                "full_field_metrics": qualification.metric_accumulate([full_row], full=True),
            })
            print(f"[P5-R] {args.route} {number}/32", flush=True)

    if resident_payload is None:
        raise RuntimeError("no resident payload")
    resident_seconds = []
    resident_anchor, resident_query, resident_weights, resident_scale, resident_map = resident_payload
    for _ in range(args.timing_repeats):
        phase = time.perf_counter()
        anchor_raw, anchor_scale = anchor_forward(params, resident_anchor)
        jax.block_until_ready(anchor_raw)
        if resolution == 1024:
            resident_support = anchor_raw - highn.REFERENCE_K
        else:
            resident_support = query_forward(params, resident_query, resident_weights, anchor_scale)
        resident_full = resident_support if resident_map is None else resident_map.reconstruct(resident_support)
        jax.block_until_ready(resident_full)
        resident_seconds.append(time.perf_counter() - phase)
    result = {
        "schema_version": "heat3d_v6_p1i_p5r_resolution_cell_v1",
        "status": "passed" if args.sample_count in (32, 96) else "passed_smoke",
        "route": route,
        "sample_count": args.sample_count,
        "sample_ids": [anchor.sample_id for anchor in anchors],
        "checkpoint_sha256": args.checkpoint_sha256,
        "protocol_sha256": _sha256(args.protocol),
        "accuracy": {
            "support": qualification.metric_accumulate(support_metric_rows, full=False),
            "full_field": qualification.metric_accumulate(full_metric_rows, full=True),
            "oracle_reconstruction_floor": qualification.metric_accumulate(oracle_metric_rows, full=True),
        },
        "timing": {key: _stats(values) for key, values in stage_values.items()},
        "resident_core": _stats(resident_seconds),
        "streaming": {
            "submit_to_result": _stats([row["streaming"]["submit_to_result_seconds"] for row in rows]),
            "inter_completion": _stats([row["streaming"]["inter_completion_seconds"] for row in rows]),
            "wall_seconds": completion_offsets[-1],
            "samples_per_second": len(rows) / completion_offsets[-1],
            "order_seed": args.order_seed,
        },
        "graph": {
            "regional_node_count_mean": float(np.mean([row["regional_node_count"] for row in rows])),
            "p2r_edges_mean": float(np.mean([row["p2r_edges"] for row in rows])),
            "r2r_edges_mean": float(np.mean([row["r2r_edges"] for row in rows])),
        },
        "peak_vram_bytes": peak_vram,
        "padding_envelopes": {"anchor": anchor_targets, "query": query_targets},
        "samples": rows,
        "role_contract": protocol["role_contract"],
        "population_mode": args.population_mode,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    if args.prediction_output is not None:
        args.prediction_output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            args.prediction_output,
            sample_ids=np.asarray([anchor.sample_id for anchor in anchors]),
            full_deltaT_K=np.stack(full_predictions),
        )
    print(json.dumps({
        "status": result["status"], "route": args.route,
        "full_pg_pct": result["accuracy"]["full_field"]["point_global_true_rms_relative_rmse_pct"],
        "continuous_median_s": result["timing"]["matched_continuous_e2e"]["median_seconds"],
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
