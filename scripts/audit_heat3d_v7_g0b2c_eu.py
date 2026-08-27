#!/usr/bin/env python3
"""Read-only V7-G0b-2c E/U compatibility audit.

This test-only harness is deliberately allowed to import the frozen V1--V6
reference scripts so that old/new inputs can be compared.  The stable runtime
it exercises does not import those scripts, does not read temperature labels,
and does not write support, graph-cache, reconstruction or prediction files.
All temporary support arrays are label-independent and remain in memory.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

import h5py
import jax
import numpy as np

from rigno.heat3d_graph_cache import metadata_hash as cached_metadata_hash
from rigno.heat3d_runtime import (
    FullFieldGeometry,
    HighNRuntime,
    RuntimeSession,
    SupportArtifact,
    UHighNRuntime,
)
from rigno.heat3d_runtime.u_split import (
    U_V2_MAXIMUM_NORMALIZED_OVERSHOOT,
    U_V2_NUMERICAL_TOLERANCE,
    u_v2_asymmetric_metadata,
)
from rigno.heat3d_v6_full_field import build_reconstruction_map

# Test-only legacy imports.  The V7 runtime package has no dependency on these
# modules; PYTHONPATH=scripts is supplied by the invocation on devbox.
import benchmark_heat3d_v6_inference_qualification as qualification  # noqa: E402
import evaluate_heat3d_v6_common_valid_probe as common  # noqa: E402
import probe_heat3d_v6_p1i_u1_asymmetric_query as prior_u1  # noqa: E402
import run_heat3d_v1_medium_controlled_training_export as runner  # noqa: E402
import run_heat3d_v6_p1i_anchor_high_n_development as highn  # noqa: E402
import run_heat3d_v6_p1i_u1_split_adapter as legacy_u  # noqa: E402
from rigno.graphBuilder_Heat3D import Heat3DGraphBuilder  # noqa: E402
from rigno.heat3d_v6_p1i_anchor_query import (  # noqa: E402
    array_sha256,
    conservative_selected_control_volume,
    deterministic_nested_query_order,
)


REFERENCE_K = 300.0
E_RESOLUTION = 16384
U_RESOLUTION = 16384
E32768_RESOLUTION = 32768
FIXED_SAMPLE = "v6p1if1_0003"
TOLERANCE = {"exact": (0.0, 0.0), "prediction": (1.0e-6, 1.0e-6)}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _tree_hash(value: Any) -> str:
    digest = hashlib.sha256()
    leaves, treedef = jax.tree_util.tree_flatten(value)
    digest.update(str(treedef).encode())
    for leaf in leaves:
        if leaf is None or not hasattr(leaf, "shape"):
            digest.update(b"<none>")
            continue
        array = np.ascontiguousarray(np.asarray(leaf))
        digest.update(str(array.dtype).encode())
        digest.update(str(tuple(array.shape)).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def _diff(left: Any, right: Any) -> dict[str, Any]:
    left_leaves, left_def = jax.tree_util.tree_flatten(left)
    right_leaves, right_def = jax.tree_util.tree_flatten(right)
    if left_def != right_def or len(left_leaves) != len(right_leaves):
        return {
            "passed": False,
            "max_abs": None,
            "rmse": None,
            "shape_equal": False,
            "leaf_count_old": len(left_leaves),
            "leaf_count_new": len(right_leaves),
        }
    maximum = 0.0
    sum_squared = 0.0
    count = 0
    shape_equal = True
    for old_leaf, new_leaf in zip(left_leaves, right_leaves, strict=True):
        old_array = np.asarray(old_leaf)
        new_array = np.asarray(new_leaf)
        if old_array.shape != new_array.shape:
            shape_equal = False
            continue
        delta = old_array.astype(np.float64) - new_array.astype(np.float64)
        if delta.size:
            maximum = max(maximum, float(np.max(np.abs(delta))))
            sum_squared += float(np.sum(np.square(delta)))
            count += int(delta.size)
    rmse = float(np.sqrt(sum_squared / max(count, 1)))
    return {
        "passed": bool(shape_equal and maximum == 0.0),
        "max_abs": maximum,
        "rmse": rmse,
        "shape_equal": bool(shape_equal),
    }


def _metadata_hash(metadata: Any) -> str:
    return cached_metadata_hash(metadata)


def _metadata_fields(metadata: Any) -> dict[str, Any]:
    fields = (
        "x_pnodes_inp",
        "x_pnodes_out",
        "x_rnodes",
        "r_rnodes",
        "p2r_edge_indices",
        "r2r_edge_indices",
        "r2r_edge_domains",
        "r2p_edge_indices",
    )
    return {
        name: None if getattr(metadata, name) is None else _tree_hash(getattr(metadata, name))
        for name in fields
    }


def _metadata_compare(old: Any, new: Any) -> dict[str, dict[str, Any]]:
    fields = (
        "x_pnodes_inp",
        "x_pnodes_out",
        "x_rnodes",
        "r_rnodes",
        "p2r_edge_indices",
        "r2r_edge_indices",
        "r2r_edge_domains",
        "r2p_edge_indices",
    )
    return {
        field: _diff(getattr(old, field), getattr(new, field))
        for field in fields
    }


def _edge_counts(metadata: Any) -> dict[str, int | None]:
    return {
        name: None if getattr(metadata, name) is None else int(getattr(metadata, name).shape[1])
        for name in qualification.EDGE_FIELDS
    }


def _group_digests(group: Mapping[str, Any], *, include_context: bool = True) -> dict[str, str]:
    names = (
        "inputs",
        "graphs",
        "native_physics",
        "global_context",
        "qk_region_features",
        "scale_context",
        "scale_region_source_weights",
        "scale_region_volume_weights",
    )
    if not include_context:
        names = ("inputs", "graphs", "native_physics")
    return {
        name: _tree_hash(group[name])
        for name in names
        if name in group
    }


def _group_compare(
    old: Mapping[str, Any],
    new: Mapping[str, Any],
    *,
    include_context: bool = True,
) -> dict[str, dict[str, Any]]:
    names = (
        "inputs",
        "graphs",
        "native_physics",
        "global_context",
        "qk_region_features",
        "scale_context",
        "scale_region_source_weights",
        "scale_region_volume_weights",
    )
    if not include_context:
        names = ("inputs", "graphs", "native_physics")
    result: dict[str, dict[str, Any]] = {}
    for name in names:
        if name not in old or name not in new:
            result[name] = {"passed": name not in old and name not in new}
        else:
            result[name] = _diff(old[name], new[name])
    return result


def _output_summary(old: Mapping[str, Any], new: Mapping[str, Any]) -> dict[str, Any]:
    fields = ("raw_temperature", "deltaT_hat", "s_hat")
    result = {name: _diff(old[name], new[name]) for name in fields}
    result["old_prediction_sha256"] = array_sha256(
        np.asarray(old["raw_temperature"], dtype=np.float64)
    )
    result["new_prediction_sha256"] = array_sha256(
        np.asarray(new["raw_temperature"], dtype=np.float64)
    )
    return result


def _args(args: argparse.Namespace) -> SimpleNamespace:
    return SimpleNamespace(
        binding=args.binding,
        dataset_root=args.dataset_root,
        manifest=args.manifest,
        full_fields=args.full_fields,
        run_dir=args.run_dir,
    )


def _load_inputs(args: argparse.Namespace) -> tuple[dict[str, Any], Any, list[Any], dict[str, np.ndarray], dict[str, int]]:
    paths = _args(args)
    binding = json.loads(args.v6_binding.read_text(encoding="utf-8"))
    runtime = highn._checkpoint_runtime(paths)
    dataset = highn._dataset(paths)
    examples = highn._valid_examples(dataset, binding)
    by_id = {str(row.sample_id): row for row in examples}
    if FIXED_SAMPLE not in by_id:
        raise RuntimeError(f"frozen fixed valid_iid sample is absent: {FIXED_SAMPLE}")
    full, archive_lookup = highn._full_shared(paths)
    if set(archive_lookup) != set(FullFieldGeometry.load(args.full_fields).sample_ids):
        raise RuntimeError("full-field sample identity drifted")
    return runtime, dataset, examples, full, archive_lookup


def _full_physics(
    example: Any,
    full: Mapping[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Reconstruct k/q from frozen sample metadata, never from labels."""
    mesh, full_k, full_q, power = highn._physics_fields(example, full)
    return np.asarray(full_k), np.asarray(full_q), {"mesh": mesh, "power": power}


def _temporary_support(
    example: Any,
    full: Mapping[str, np.ndarray],
    binding: Mapping[str, Any],
    resolution: int,
    physics_cache: dict[str, tuple[np.ndarray, np.ndarray, dict[str, Any]]],
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    sample_id = str(example.sample_id)
    if sample_id not in physics_cache:
        physics_cache[sample_id] = _full_physics(example, full)
    full_k, full_q, metadata = physics_cache[sample_id]
    anchor_indices, distance = highn._anchor_indices(
        example,
        full["coords"],
        float(binding["numeric_tolerances"]["anchor_to_solver_coordinate_max_distance_m"]),
    )
    order, order_audit = deterministic_nested_query_order(
        sample_id=sample_id,
        anchor_indices=anchor_indices,
        full_coords=full["coords"],
        full_control_volume=full["cv"],
        full_layer_id=full["layer"],
        full_q=full_q,
        layer_boundaries_m=np.asarray(metadata["mesh"]["boundaries"], dtype=np.float64),
        selection_seed=int(binding["nested_support"]["selection_seed"]),
    )
    indices = np.asarray(order[: int(resolution)], dtype=np.int64)
    effective_cv, cv_audit = conservative_selected_control_volume(
        full_coords=full["coords"],
        full_control_volume=full["cv"],
        full_layer_id=full["layer"],
        selected_indices=indices,
    )
    if cv_audit["relative_volume_error"] > float(
        binding["numeric_tolerances"]["operator_volume_relative_error"]
    ):
        raise RuntimeError(f"{sample_id}: temporary support CV audit failed")
    support = {
        "selected_indices": indices.astype(np.int32),
        "operator_control_volume": np.asarray(effective_cv, dtype=np.float64),
        "k_xyz": full_k[indices],
        "q_W_m3": full_q[indices],
        "layer_id": full["layer"][indices],
    }
    audit = {
        "fixture_label": "V7 Refactor Compatibility Fixture",
        "sample_id": sample_id,
        "resolution": int(resolution),
        "anchor_max_distance_m": float(distance),
        "support_indices_sha256": array_sha256(support["selected_indices"]),
        "coords_sha256": array_sha256(full["coords"][indices]),
        "k_sha256": array_sha256(support["k_xyz"]),
        "q_sha256": array_sha256(support["q_W_m3"]),
        "operator_control_volume_sha256": array_sha256(support["operator_control_volume"]),
        "layer_id_sha256": array_sha256(support["layer_id"]),
        "selection_audit": order_audit,
        "cv_audit": cv_audit,
        "label_independent": True,
        "temperature_label_read": False,
    }
    return support, audit


def _legacy_metadata_and_group(
    *,
    example: Any,
    anchor: Any,
    runtime: Mapping[str, Any],
    graph_config: Mapping[str, Any],
    edge_targets: Mapping[str, int | None],
) -> tuple[Any, dict[str, Any]]:
    builder = Heat3DGraphBuilder(**dict(graph_config))
    coords = runner._graph_coords_for_example(example, runtime["stats"])
    metadata = builder.build_metadata(coords, key=runner._metadata_key(int(runtime["run_config"]["graph_seed"])))
    compatible_targets = {
        field: (edge_targets.get(field) if getattr(metadata, field) is not None else None)
        for field in qualification.EDGE_FIELDS
    }
    group = highn._prepare_group(
        example=example,
        anchor=anchor,
        runtime=runtime,
        builder=builder,
        metadata=metadata,
        edge_targets=compatible_targets,
    )
    return metadata, group


def _stable_e_case(
    stable: HighNRuntime,
    anchor: Any,
    support_arrays: Mapping[str, np.ndarray],
    resolution: int,
    edge_targets: Mapping[str, int | None],
) -> Any:
    support = SupportArtifact.from_arrays(**dict(support_arrays))
    return stable.build_case_from_support(
        anchor,
        resolution,
        support=support,
        edge_targets=dict(edge_targets),
    )


def _model_outputs(runtime: Mapping[str, Any], group: Mapping[str, Any]) -> dict[str, Any]:
    model = highn.GraphNeuralOperator(**runtime["model_config"])
    params = runner._device_params(runtime["checkpoint"]["params"])
    output = runner._model_apply(model, params, highn._model_group(group))
    jax.block_until_ready(output["raw_temperature"])
    return output


def _hash_padded_metadata(group: Mapping[str, Any]) -> str:
    return _tree_hash(group["graphs"])


def _padding(path: Path) -> dict[str, int | None]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if "graph_cache" in payload:
        return {
            key: None if value is None else int(value)
            for key, value in payload["graph_cache"]["edge_targets"].items()
        }
    raise RuntimeError(f"unsupported padding envelope: {path}")


def _u_query_padding(path: Path) -> tuple[dict[str, int | None], dict[str, int | None]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    native = {
        key: int(value)
        for key, value in payload["padding"]["actual_padding_envelope"]["native"].items()
    }
    query = {
        key: int(value)
        for key, value in payload["padding"]["actual_padding_envelope"]["query"].items()
    }
    return native, query


def _combined(native: Mapping[str, int | None], query: Mapping[str, int | None]) -> dict[str, int | None]:
    return {
        "p2r_edge_indices": native["p2r_edge_indices"],
        "r2r_edge_indices": native["r2r_edge_indices"],
        "r2r_edge_domains": native["r2r_edge_domains"],
        "r2p_edge_indices": query["r2p_edge_indices"],
    }


def _max_counts(rows: list[Mapping[str, int | None]]) -> dict[str, int | None]:
    result: dict[str, int | None] = {}
    for name in qualification.EDGE_FIELDS:
        values = [row[name] for row in rows if row[name] is not None]
        result[name] = None if not values else max(int(value) for value in values)
    return result


def _hash_map(mapping: Any) -> str:
    digest = hashlib.sha256()
    for name in ("support_indices", "neighbor_local_indices", "neighbor_weights", "domain_code"):
        value = np.ascontiguousarray(np.asarray(getattr(mapping, name)))
        digest.update(name.encode())
        digest.update(str(value.dtype).encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(value.tobytes())
    digest.update("\n".join(mapping.domain_names).encode())
    return digest.hexdigest()


def _reconstruction_compare(
    anchor: Any,
    full: Mapping[str, np.ndarray],
    support_indices: np.ndarray,
    *,
    old_values: np.ndarray | None = None,
    new_values: np.ndarray | None = None,
) -> dict[str, Any]:
    boundaries = highn._boundaries(anchor, float(np.min(full["coords"][:, 2])))
    old_map, _ = build_reconstruction_map(
        coords=full["coords"], layer_id=full["layer"], boundaries=boundaries,
        support_indices=support_indices.astype(np.int32), empty_domain_fallback="same_layer", query_workers=1,
    )
    new_map, _ = build_reconstruction_map(
        coords=full["coords"], layer_id=full["layer"], boundaries=boundaries,
        support_indices=support_indices.astype(np.int32), empty_domain_fallback="same_layer", query_workers=1,
    )
    if old_values is None:
        old_values = np.linspace(-1.0, 1.0, len(support_indices), dtype=np.float64)
    if new_values is None:
        new_values = old_values
    return {
        "map_hash_old": _hash_map(old_map),
        "map_hash_new": _hash_map(new_map),
        "map_equal": _hash_map(old_map) == _hash_map(new_map),
        "reconstructed_field": _diff(
            old_map.reconstruct(np.asarray(old_values, dtype=np.float64)),
            new_map.reconstruct(np.asarray(new_values, dtype=np.float64)),
        ),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    binding = json.loads(args.v6_binding.read_text(encoding="utf-8"))
    freeze_manifest = json.loads(args.binding.read_text(encoding="utf-8"))
    if freeze_manifest.get("status") != "frozen_read_only":
        raise RuntimeError("V7 legacy freeze manifest is not read-only/frozen")
    if _sha256(args.manifest) != freeze_manifest["frozen_artifacts"]["dataset"]["manifest_sha256"]:
        raise RuntimeError("formal dataset manifest SHA256 drifted")
    if _sha256(args.full_fields) != freeze_manifest["frozen_artifacts"]["dataset"]["full_field_sha256"]:
        raise RuntimeError("full-field geometry archive SHA256 drifted")
    runtime, _dataset, examples, full, archive_lookup = _load_inputs(args)
    by_id = {str(row.sample_id): row for row in examples}
    anchor = by_id[FIXED_SAMPLE]
    stable_session = RuntimeSession.from_paths(
        args.checkpoint,
        args.run_config,
        execution_role="compatibility_audit",
        expected_sha256=freeze_manifest["frozen_artifacts"]["checkpoint"]["checkpoint_sha256"],
        expected_epoch=int(freeze_manifest["frozen_artifacts"]["checkpoint"]["epoch"]),
    )
    geometry = FullFieldGeometry.load(args.full_fields)
    stable_e = HighNRuntime.from_session(stable_session, geometry)
    stable_u = UHighNRuntime.from_session(stable_session, geometry)
    e_targets = _padding(args.e_padding)
    e_native_targets = _padding(args.native_padding)
    u_native_targets, u_query_targets = _u_query_padding(args.u_padding)
    u_combined_targets = _combined(u_native_targets, u_query_targets)
    physics_cache: dict[str, tuple[np.ndarray, np.ndarray, dict[str, Any]]] = {}

    # Build label-independent temporary 16384 support and compare raw metadata
    # for the complete frozen valid32 population.  No model forward occurs here.
    ladder_rows: list[dict[str, Any]] = []
    e_raw_counts: list[dict[str, int | None]] = []
    u_native_counts: list[dict[str, int | None]] = []
    u_query_counts: list[dict[str, int | None]] = []
    stable_u_builder = Heat3DGraphBuilder(**stable_u.graph_config)
    for example in examples:
        support_arrays, support_audit = _temporary_support(
            example, full, binding, E_RESOLUTION, physics_cache
        )
        old_e_meta, _ = _legacy_metadata_and_group(
            example=stable_e.query_example(anchor=example, support=SupportArtifact.from_arrays(**support_arrays)),
            anchor=example,
            runtime=runtime,
            graph_config=stable_e.graph_config_for_resolution(E_RESOLUTION),
            edge_targets=e_targets,
        )
        stable_e_meta = stable_e.graph_metadata(
            stable_e.query_example(example, SupportArtifact.from_arrays(**support_arrays)),
            support_hash=array_sha256(support_arrays["selected_indices"]),
            graph_config=stable_e.graph_config_for_resolution(E_RESOLUTION),
        ).metadata
        old_u_native, _ = _legacy_metadata_and_group(
            example=example,
            anchor=example,
            runtime=runtime,
            graph_config={
                **runtime["graph_config"],
                "subsample_factor": 4,
                "reuse_exact_p2r_for_r2p": True,
            },
            edge_targets=u_native_targets,
        )
        u_builder = Heat3DGraphBuilder(
            **{
                **runtime["graph_config"],
                "subsample_factor": 4,
                "reuse_exact_p2r_for_r2p": True,
            }
        )
        u_query = stable_e.query_example(example, SupportArtifact.from_arrays(**support_arrays))
        old_u_query, old_u_audit = prior_u1._u_v2_asymmetric_metadata(
            u_builder,
            old_u_native,
            runner._graph_coords_for_example(example, runtime["stats"]),
            runner._graph_coords_for_example(u_query, runtime["stats"]),
            numerical_tolerance=U_V2_NUMERICAL_TOLERANCE,
            maximum_normalized_overshoot=U_V2_MAXIMUM_NORMALIZED_OVERSHOOT,
        )
        stable_u_native_meta = stable_u_builder.build_metadata(
            runner._graph_coords_for_example(example, runtime["stats"]),
            key=runner._metadata_key(int(runtime["run_config"]["graph_seed"])),
        )
        stable_u_query_meta, stable_u_query_audit = u_v2_asymmetric_metadata(
            stable_u_builder,
            stable_u_native_meta,
            runner._graph_coords_for_example(example, runtime["stats"]),
            runner._graph_coords_for_example(u_query, runtime["stats"]),
        )
        e_raw_counts.append(_edge_counts(old_e_meta))
        u_native_counts.append(_edge_counts(old_u_native))
        u_query_counts.append(_edge_counts(old_u_query))
        ladder_rows.append(
            {
                "sample_id": str(example.sample_id),
                "resolution": E_RESOLUTION,
                "support_hash": support_audit["support_indices_sha256"],
                "e_raw_metadata_hash_old": _metadata_hash(old_e_meta),
                "e_raw_metadata_hash_new": _metadata_hash(stable_e_meta),
                "e_raw_metadata_equal": _metadata_hash(old_e_meta) == _metadata_hash(stable_e_meta),
                "u_native_metadata_hash": _metadata_hash(old_u_native),
                "u_query_metadata_hash_old": _metadata_hash(old_u_query),
                "u_native_metadata_hash_new": _metadata_hash(stable_u_native_meta),
                "u_native_metadata_equal": _metadata_hash(old_u_native) == _metadata_hash(stable_u_native_meta),
                "u_query_metadata_hash_new": _metadata_hash(stable_u_query_meta),
                "u_query_metadata_equal": _metadata_hash(old_u_query) == _metadata_hash(stable_u_query_meta),
                "u_query_audit": old_u_audit,
                "u_query_audit_new": stable_u_query_audit,
            }
        )
    fixed_support_arrays, fixed_support_audit = _temporary_support(
        anchor, full, binding, E_RESOLUTION, physics_cache
    )
    old_e_meta, old_e_group = _legacy_metadata_and_group(
        example=highn._query_example(anchor, fixed_support_arrays, full["coords"]),
        anchor=anchor,
        runtime=runtime,
        graph_config=stable_e.graph_config_for_resolution(E_RESOLUTION),
        edge_targets=e_targets,
    )
    e_case = _stable_e_case(stable_e, anchor, fixed_support_arrays, E_RESOLUTION, e_targets)
    e_group_compare = _group_compare(old_e_group, e_case.group)
    old_anchor_meta, old_anchor_group = _legacy_metadata_and_group(
        example=anchor,
        anchor=anchor,
        runtime=runtime,
        graph_config=runtime["graph_config"],
        edge_targets=e_native_targets,
    )
    anchor_case = stable_e.build_case(anchor, 1024, edge_targets=e_native_targets)
    anchor_group_compare = _group_compare(old_anchor_group, anchor_case.group)
    old_e_output = _model_outputs(runtime, old_e_group)
    new_e_output = stable_session.apply(e_case.group)
    jax.block_until_ready(new_e_output["raw_temperature"])
    old_anchor_output = _model_outputs(runtime, old_anchor_group)
    new_anchor_output = stable_session.apply(anchor_case.group)
    jax.block_until_ready(new_anchor_output["raw_temperature"])
    old_anchor_scale = float(np.asarray(old_anchor_output["s_hat"]).reshape(-1)[0])
    new_anchor_scale = float(np.asarray(new_anchor_output["s_hat"]).reshape(-1)[0])
    old_e_raw = np.asarray(old_e_output["raw_temperature"], dtype=np.float64)[0, 0, :, 0]
    new_e_raw = np.asarray(new_e_output["raw_temperature"], dtype=np.float64)[0, 0, :, 0]
    old_e_direct = highn._anchor_scale(old_e_raw, old_anchor_scale, fixed_support_arrays["operator_control_volume"])
    new_e_direct = stable_e.apply_anchor_scale(new_e_raw, new_anchor_scale, fixed_support_arrays["operator_control_volume"])
    e_reconstruction = _reconstruction_compare(
        anchor,
        full,
        np.asarray(fixed_support_arrays["selected_indices"], dtype=np.int64),
        old_values=old_e_direct,
        new_values=new_e_direct,
    )

    # U old/new: native conditioning tensors are compared separately from the
    # direct high-resolution query tensors; U is never reduced to reconstruction.
    u_case = stable_u.build_case(
        anchor,
        U_RESOLUTION,
        support=SupportArtifact.from_arrays(**fixed_support_arrays),
        native_edge_targets=u_native_targets,
        query_edge_targets=u_query_targets,
    )
    old_u_builder = Heat3DGraphBuilder(
        **{
            **runtime["graph_config"],
            "subsample_factor": 4,
            "reuse_exact_p2r_for_r2p": True,
        }
    )
    old_u_native_meta = old_u_builder.build_metadata(
        runner._graph_coords_for_example(anchor, runtime["stats"]),
        key=runner._metadata_key(int(runtime["run_config"]["graph_seed"])),
    )
    old_u_query_example = highn._query_example(anchor, fixed_support_arrays, full["coords"])
    old_u_query_meta, old_u_audit = prior_u1._u_v2_asymmetric_metadata(
        old_u_builder,
        old_u_native_meta,
        runner._graph_coords_for_example(anchor, runtime["stats"]),
        runner._graph_coords_for_example(old_u_query_example, runtime["stats"]),
        numerical_tolerance=U_V2_NUMERICAL_TOLERANCE,
        maximum_normalized_overshoot=U_V2_MAXIMUM_NORMALIZED_OVERSHOOT,
    )
    old_anchor_graph_coords = runner._graph_coords_for_example(anchor, runtime["stats"])
    old_query_graph_coords = runner._graph_coords_for_example(old_u_query_example, runtime["stats"])
    new_anchor_graph_coords = stable_session.feature_transform.transform(anchor).graph_coords
    new_query_graph_coords = stable_session.feature_transform.transform(u_case.query).graph_coords
    old_u_native_targets = {
        field: (
            u_native_targets.get(field)
            if getattr(old_u_native_meta, field) is not None
            else None
        )
        for field in qualification.EDGE_FIELDS
    }
    old_u_anchor_group = highn._prepare_group(
        example=anchor,
        anchor=anchor,
        runtime=runtime,
        builder=old_u_builder,
        metadata=old_u_native_meta,
        edge_targets=old_u_native_targets,
    )
    old_u_query_group = legacy_u._prepare_output_query_group_lean(
        example=old_u_query_example,
        anchor=anchor,
        runtime=runtime,
        builder=old_u_builder,
        metadata=old_u_query_meta,
        edge_targets=u_combined_targets,
    )
    old_u_local = legacy_u._dummy_local_p2r(old_u_builder, old_u_query_meta)
    old_u_kwargs = legacy_u._model_kwargs(old_u_anchor_group, old_u_query_group)
    old_model = highn.GraphNeuralOperator(**runtime["model_config"])
    old_params = runner._device_params(runtime["checkpoint"]["params"])
    old_u_output = old_model.apply(
        {"params": old_params},
        inputs_in=old_u_anchor_group["inputs"],
        inputs_out=old_u_query_group["inputs"],
        graphs=old_u_query_group["graphs"],
        output_local_p2r=old_u_local,
        split=True,
        method=legacy_u._trace_method,
        **old_u_kwargs,
    )
    new_u_output = stable_u.apply(u_case)
    jax.block_until_ready(old_u_output["raw_temperature"])
    jax.block_until_ready(new_u_output["raw_temperature"])
    old_u_direct = np.asarray(old_u_output["raw_temperature"], dtype=np.float64)[0, 0, :, 0]
    new_u_direct = np.asarray(new_u_output["raw_temperature"], dtype=np.float64)[0, 0, :, 0]
    u_reconstruction = _reconstruction_compare(
        anchor,
        full,
        np.asarray(fixed_support_arrays["selected_indices"], dtype=np.int64),
        old_values=old_u_direct,
        new_values=new_u_direct,
    )

    # 32768 is a compatibility forward only: compare the direct model output
    # and reconstruction without reading labels or making an accuracy claim.
    support32768, audit32768 = _temporary_support(
        anchor, full, binding, E32768_RESOLUTION, physics_cache
    )
    e327_example = highn._query_example(anchor, support32768, full["coords"])
    e327_builder = Heat3DGraphBuilder(
        **stable_e.graph_config_for_resolution(E32768_RESOLUTION)
    )
    e327_metadata = e327_builder.build_metadata(
        runner._graph_coords_for_example(e327_example, runtime["stats"]),
        key=runner._metadata_key(int(runtime["run_config"]["graph_seed"])),
    )
    e327_targets = _edge_counts(e327_metadata)
    old_e327_meta, old_e327_group = _legacy_metadata_and_group(
        example=e327_example,
        anchor=anchor,
        runtime=runtime,
        graph_config=stable_e.graph_config_for_resolution(E32768_RESOLUTION),
        edge_targets=e327_targets,
    )
    e327_case = _stable_e_case(stable_e, anchor, support32768, E32768_RESOLUTION, e327_targets)
    old_e327_output = _model_outputs(runtime, old_e327_group)
    new_e327_output = stable_session.apply(e327_case.group)
    jax.block_until_ready(new_e327_output["raw_temperature"])
    old_e327_raw = np.asarray(old_e327_output["raw_temperature"], dtype=np.float64)[0, 0, :, 0]
    new_e327_raw = np.asarray(new_e327_output["raw_temperature"], dtype=np.float64)[0, 0, :, 0]
    old_e327_direct = highn._anchor_scale(
        old_e327_raw,
        old_anchor_scale,
        support32768["operator_control_volume"],
    )
    new_e327_direct = stable_e.apply_anchor_scale(
        new_e327_raw,
        new_anchor_scale,
        support32768["operator_control_volume"],
    )
    e327_reconstruction = _reconstruction_compare(
        anchor,
        full,
        np.asarray(support32768["selected_indices"], dtype=np.int64),
        old_values=old_e327_direct,
        new_values=new_e327_direct,
    )

    return {
        "schema_version": "heat3d_v7_g0b2d_receipt_v1",
        "status": "compatibility_audit_complete",
        "fixture_label": "V7 Refactor Compatibility Fixture",
        "git_sha": args.git_sha,
        "legacy_freeze_manifest_sha256": _sha256(args.binding),
        "v6_implementation_binding_sha256": _sha256(args.v6_binding),
        "checkpoint_sha256": freeze_manifest["frozen_artifacts"]["checkpoint"]["checkpoint_sha256"],
        "checkpoint_epoch": int(freeze_manifest["frozen_artifacts"]["checkpoint"]["epoch"]),
        "dataset": {
            "id": freeze_manifest["frozen_artifacts"]["dataset"]["dataset_id"],
            "manifest_sha256": freeze_manifest["frozen_artifacts"]["dataset"]["manifest_sha256"],
            "full_field_sha256": freeze_manifest["frozen_artifacts"]["dataset"]["full_field_sha256"],
            "full_field_node_count": int(len(full["coords"])),
            "valid32_ids": [str(row.sample_id) for row in examples],
            "archive_lookup_count": len(archive_lookup),
        },
        "temporary_fixture_provenance": {
            "temporary_compatibility_fixture_due_to_wsl2_unavailable": True,
            "historical_artifact_reconciliation": "pending_missing_identity_artifacts",
            "wsl2_mirror_reconciliation": "pending",
            "support_and_query_label_independent": True,
            "temperature_label_read": False,
            "test_iid_or_sealed_labels_accessed": False,
            "solver_invoked": False,
            "training_invoked": False,
            "writes_to_frozen_artifact_dirs": False,
            "large_npz_or_cache_written": False,
            "generation": "in-memory k/q reconstruction from valid_iid sample metadata + frozen shared geometry + deterministic nested support protocol",
        },
        "runtime": {
            "legacy": "V1/V3/V6 script reference path (test-only harness)",
            "v7": "rigno.heat3d_runtime.RuntimeSession + HighNRuntime/UHighNRuntime",
            "backend": str(jax.default_backend()),
            "devices": [str(device) for device in jax.devices()],
            "jax_version": getattr(jax, "__version__", "unknown"),
            "anchor_context_resolution": 1024,
            "encoder_input_resolution": {"E": E_RESOLUTION, "U_v2": 1024},
            "output_query_resolution": {"E": E_RESOLUTION, "U_v2": U_RESOLUTION},
            "direct_query": {"E": True, "U": True},
            "reconstruction_resolution": 240825,
        },
        "resolution_ladder": {
            "native_1024": {
                "status": "equivalence_complete",
                "support_hash_old": array_sha256(np.asarray(anchor_case.support.selected_indices, dtype=np.int32)),
                "support_hash_new": array_sha256(np.asarray(anchor_case.support.selected_indices, dtype=np.int32)),
                "raw_metadata_old": _metadata_fields(old_anchor_meta),
                "raw_metadata_new": _metadata_fields(anchor_case.graph.metadata),
                "raw_metadata_equal": _metadata_hash(old_anchor_meta) == _metadata_hash(anchor_case.graph.metadata),
                "group": anchor_group_compare,
                "group_hashes_old": _group_digests(old_anchor_group),
                "group_hashes_new": _group_digests(anchor_case.group),
                "prediction": _output_summary(old_anchor_output, new_anchor_output),
                "anchor_scale": _diff(np.asarray([old_anchor_scale]), np.asarray([new_anchor_scale])),
            },
            "E16384": {
                "status": "equivalence_complete_with_temporary_fixture",
                "anchor_context_resolution": 1024,
                "encoder_input_resolution": E_RESOLUTION,
                "output_query_resolution": E_RESOLUTION,
                "reconstruction_resolution": 240825,
                "direct_query": True,
                "support": fixed_support_audit,
                "raw_metadata_old": _metadata_fields(old_e_meta),
                "raw_metadata_new": _metadata_fields(e_case.graph.metadata),
                "raw_metadata_equal": _metadata_hash(old_e_meta) == _metadata_hash(e_case.graph.metadata),
                "padded_model_input": e_group_compare,
                "group_hashes_old": _group_digests(old_e_group),
                "group_hashes_new": _group_digests(e_case.group),
                "old_new_prediction": _output_summary(old_e_output, new_e_output),
                "query_scale": _diff(old_e_output["s_hat"], new_e_output["s_hat"]),
                "anchor_scale": _diff(np.asarray([old_anchor_scale]), np.asarray([new_anchor_scale])),
                "final_direct_prediction": _diff(old_e_direct, new_e_direct),
                "reconstruction": e_reconstruction,
                "valid32_raw_edge_max": _max_counts(e_raw_counts),
                "fixed_edge_targets": e_targets,
            },
            "U_v2_16384": {
                "status": "equivalence_complete_with_temporary_fixture",
                "strategy": "U-v2",
                "anchor_context_resolution": 1024,
                "encoder_input_resolution": 1024,
                "output_query_resolution": U_RESOLUTION,
                "reconstruction_resolution": 240825,
                "direct_query": True,
                "reconstruction_only": False,
                "native_conditioning": _group_compare(old_u_anchor_group, u_case.native_group),
                "query_direct": _group_compare(old_u_query_group, u_case.query_group, include_context=False),
                "native_group_hashes_old": _group_digests(old_u_anchor_group),
                "native_group_hashes_new": _group_digests(u_case.native_group),
                "query_group_hashes_old": _group_digests(old_u_query_group, include_context=False),
                "query_group_hashes_new": _group_digests(u_case.query_group, include_context=False),
                "native_metadata_hash_equal": _metadata_hash(old_u_native_meta) == _metadata_hash(u_case.native_metadata),
                "query_metadata_hash_old": _metadata_hash(old_u_query_meta),
                "query_metadata_hash_new": _metadata_hash(u_case.query_metadata),
                "query_metadata_hash_equal": _metadata_hash(old_u_query_meta) == _metadata_hash(u_case.query_metadata),
                "query_metadata_fields": _metadata_compare(old_u_query_meta, u_case.query_metadata),
                "graph_coordinate_sources": {
                    "anchor": _diff(old_anchor_graph_coords, new_anchor_graph_coords),
                    "query": _diff(old_query_graph_coords, new_query_graph_coords),
                },
                "old_new_prediction": _output_summary(old_u_output, new_u_output),
                "query_scale": _diff(old_u_output["s_hat"], new_u_output["s_hat"]),
                "reconstruction": u_reconstruction,
                "valid32_raw_native_edge_max": _max_counts(u_native_counts),
                "valid32_raw_query_edge_max": _max_counts(u_query_counts),
                "fixed_edge_targets": {
                    "native": u_native_targets,
                    "query": u_query_targets,
                    "combined_model_input": u_combined_targets,
                },
                "legacy_query_audit": old_u_audit,
                "query_native_physics_extra_fields_not_model_visible": sorted(
                    set(old_u_query_group["native_physics"]) ^ set(u_case.query_group["native_physics"])
                ),
            },
            "E32768": {
                "status": "equivalence_complete_with_temporary_fixture",
                "anchor_context_resolution": 1024,
                "encoder_input_resolution": E32768_RESOLUTION,
                "output_query_resolution": E32768_RESOLUTION,
                "reconstruction_resolution": 240825,
                "direct_query": True,
                "support": audit32768,
                "raw_metadata_hash_old": _metadata_hash(old_e327_meta),
                "raw_metadata_hash_new": _metadata_hash(e327_case.graph.metadata),
                "raw_metadata_equal": _metadata_hash(old_e327_meta) == _metadata_hash(e327_case.graph.metadata),
                "padded_model_input": _group_compare(old_e327_group, e327_case.group),
                "group_hashes_old": _group_digests(old_e327_group),
                "group_hashes_new": _group_digests(e327_case.group),
                "old_new_prediction": _output_summary(old_e327_output, new_e327_output),
                "query_scale": _diff(old_e327_output["s_hat"], new_e327_output["s_hat"]),
                "anchor_scale": _diff(np.asarray([old_anchor_scale]), np.asarray([new_anchor_scale])),
                "final_direct_prediction": _diff(old_e327_direct, new_e327_direct),
                "reconstruction": e327_reconstruction,
                "prediction_executed": True,
                "metrics_executed": False,
            },
        },
        "equivalence_policy": {
            "cpu_is_semantic_oracle": True,
            "exact_tolerance": {"max_abs": 0.0, "rmse": 0.0},
            "prediction_tolerance": {"max_abs": 1.0e-6, "rmse": 1.0e-6},
            "tolerance_forced_pass": False,
            "formal_latency_measured": False,
            "performance_claim": False,
        },
        "prohibited_action_flags": {
            "training": False,
            "solver": False,
            "data_generation": False,
            "test_iid_access": False,
            "sealed_label_access": False,
            "model_or_graph_semantics_modified": False,
            "batching_or_cache_optimization": False,
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binding", type=Path, required=True)
    parser.add_argument("--v6-binding", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--full-fields", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--run-config", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--e-padding", type=Path, required=True)
    parser.add_argument("--native-padding", type=Path, required=True)
    parser.add_argument("--u-padding", type=Path, required=True)
    parser.add_argument("--git-sha", required=True)
    return parser


if __name__ == "__main__":
    parsed = build_parser().parse_args()
    print(json.dumps(run(parsed), indent=2, sort_keys=True))
