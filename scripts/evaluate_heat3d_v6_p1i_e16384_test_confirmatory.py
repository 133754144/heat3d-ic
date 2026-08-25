#!/usr/bin/env python3
"""One-time frozen E16384 evaluation on the corrected test_iid holdout.

The label-free graph qualification runs before the HDF5 target dataset is
opened.  The evaluator preserves the checkpoint, route, support selector,
graph semantics, and reconstruction algorithm used by the frozen production
route.  It intentionally records no publication timing.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any

import h5py
import jax
import jax.numpy as jnp
import numpy as np


ROOT = Path(os.environ.get("HEAT3D_REPO_ROOT", Path(__file__).resolve().parents[1])).resolve()
for value in (ROOT, ROOT / "scripts"):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))

import benchmark_heat3d_v6_inference_qualification as qualification  # noqa: E402
import benchmark_heat3d_v6_p1i_p8_throughput_fairness as p8  # noqa: E402
import benchmark_heat3d_v6_p1i_final_e_service as final_e  # noqa: E402
import run_heat3d_v6_p1i_anchor_high_n_development as highn  # noqa: E402
import run_heat3d_v6_p1i_p5r_resolution_cell as p5r  # noqa: E402
from rigno.graphBuilder_Heat3D import Heat3DGraphBuilder  # noqa: E402
from rigno.heat3d_v6_dataset import (  # noqa: E402
    CONTINUOUS_PHYSICS_V6_DATASET_ID,
    Heat3DV6DualRobinDataset,
)
from rigno.heat3d_v6_full_field import (  # noqa: E402
    prepare_reconstruction_domain_partition,
)
from rigno.heat3d_v6_p1i_anchor_query import (  # noqa: E402
    array_sha256,
    conservative_selected_control_volume,
    deterministic_nested_query_prefix,
    prepare_nested_query_geometry_cache,
)
from rigno.models.rigno import RIGNO as GraphNeuralOperator  # noqa: E402


ROUTE = "E16384_reconstruction"
RESOLUTION = 16384
MODEL_SEED_LABEL = "model_seed0"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def tree_sha256(value: Any) -> str:
    digest = hashlib.sha256()
    leaves, treedef = jax.tree_util.tree_flatten(value)
    digest.update(str(treedef).encode())
    for leaf in leaves:
        array = np.ascontiguousarray(np.asarray(leaf))
        digest.update(str(array.dtype).encode())
        digest.update(str(tuple(array.shape)).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    for name in (
        "protocol", "binding", "dataset_root", "manifest", "full_fields",
        "run_dir", "native_padding_result", "query_padding_result", "work_root",
        "output",
    ):
        parser.add_argument(f"--{name.replace('_', '-')}", dest=name, type=Path, required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    return parser.parse_args()


def build_state(args: argparse.Namespace, protocol: dict[str, Any]) -> dict[str, Any]:
    if sha256(args.manifest) != protocol["frozen_confirmatory_evaluation"]["dataset_manifest_sha256"]:
        raise RuntimeError("formal manifest SHA drifted")
    if sha256(args.full_fields) != protocol["frozen_confirmatory_evaluation"]["full_field_archive_sha256"]:
        raise RuntimeError("full-field archive SHA drifted")
    runtime = p5r._runtime(args)
    if int(runtime["checkpoint"]["epoch"]) != protocol["frozen_confirmatory_evaluation"]["checkpoint_epoch"]:
        raise RuntimeError("checkpoint epoch drifted")
    dataset = Heat3DV6DualRobinDataset(
        args.dataset_root, args.manifest, include_roles={"train", "test_iid"},
    )
    if dataset.manifest["dataset_id"] != CONTINUOUS_PHYSICS_V6_DATASET_ID:
        raise RuntimeError("dataset ID drifted")
    ordered_ids = sorted(
        dataset.split_ids["test_iid"],
        key=lambda sample_id: hashlib.sha256(sample_id.encode()).hexdigest(),
    )
    order_sha = hashlib.sha256(("\n".join(ordered_ids) + "\n").encode()).hexdigest()
    frozen = protocol["frozen_confirmatory_evaluation"]
    if len(ordered_ids) != frozen["sample_count"] or order_sha != frozen["sample_order_sha256"]:
        raise RuntimeError("test_iid population/order drifted")
    index = dataset.sample_index_by_id()
    anchors = [dataset[index[sample_id]] for sample_id in ordered_ids]
    full, archive_lookup = highn._full_shared(args)
    coords = np.asarray(full["coords"], dtype=np.float64)
    cv = np.asarray(full["cv"], dtype=np.float64)
    layer = np.asarray(full["layer"], dtype=np.int32)
    boundaries = highn._boundaries(anchors[0], float(np.min(coords[:, 2])))
    geometry = prepare_nested_query_geometry_cache(
        full_coords=coords, full_control_volume=cv, full_layer_id=layer,
        layer_boundaries_m=boundaries,
    )
    partition = prepare_reconstruction_domain_partition(
        coords=coords, layer_id=layer, boundaries=boundaries,
    )
    anchor_config = dict(runtime["graph_config"])
    anchor_config.update(
        subsample_factor=4.0,
        discrete_graph_backend="sparse_kdtree_v1",
        reuse_exact_p2r_for_r2p=True,
    )
    query_config = dict(runtime["graph_config"])
    query_config.update(
        subsample_factor=64.0,
        discrete_graph_backend="sparse_kdtree_v1",
        reuse_exact_p2r_for_r2p=True,
    )
    state = {
        "args": args,
        "runtime": runtime,
        "binding": json.loads(args.binding.read_text()),
        "dataset": dataset,
        "anchors": anchors,
        "coords": coords,
        "cv": cv,
        "layer": layer,
        "boundaries": boundaries,
        "geometry": geometry,
        "partition": partition,
        "archive_lookup": archive_lookup,
        "graph_key": highn.runner._metadata_key(int(runtime["run_config"]["graph_seed"])),
        "anchor_config": anchor_config,
        "query_config": query_config,
        "anchor_targets": p5r._edge_targets(args.native_padding_result),
        "query_targets": p5r._edge_targets(args.query_padding_result),
        "physics": {},
        "frozen": {},
    }
    return state


def materialize_physics(state: dict[str, Any], work_root: Path) -> list[dict[str, Any]]:
    root = work_root / "test_iid_label_independent_physics"
    root.mkdir(parents=True, exist_ok=True)
    rows = []
    full = {"coords": state["coords"], "cv": state["cv"], "layer": state["layer"]}
    for anchor in state["anchors"]:
        path = root / f"{anchor.sample_id}.npz"
        if not path.is_file():
            _, k, q, power = highn._physics_fields(anchor, full)
            np.savez_compressed(path, k_xyz=k, q_W_m3=q)
        with np.load(path, allow_pickle=False) as archive:
            k = np.asarray(archive["k_xyz"], dtype=np.float64)
            q = np.asarray(archive["q_W_m3"], dtype=np.float64)
        if k.shape != (240825, 3) or q.shape != (240825,) or not np.all(np.isfinite(k)) or not np.all(np.isfinite(q)):
            raise RuntimeError(f"{anchor.sample_id}: invalid reconstructed physics")
        power = float(np.sum(q * state["cv"]))
        row = {
            "sample_id": anchor.sample_id,
            "physics_cache_file": str(path),
            "physics_cache_sha256": sha256(path),
            "full_power_W": power,
        }
        state["physics"][anchor.sample_id] = row
        rows.append(row)
    return rows


def graph_qualification_pass(state: dict[str, Any], *, widen: bool) -> list[dict[str, Any]]:
    rows = []
    cpu = jax.devices("cpu")[0]
    tolerance = float(state["binding"]["numeric_tolerances"]["anchor_to_solver_coordinate_max_distance_m"])
    for anchor in state["anchors"]:
        with np.load(state["physics"][anchor.sample_id]["physics_cache_file"], allow_pickle=False) as archive:
            full_k = np.asarray(archive["k_xyz"], dtype=np.float64)
            full_q = np.asarray(archive["q_W_m3"], dtype=np.float64)
        anchor_indices, distance = highn._anchor_indices(anchor, state["coords"], tolerance)
        if distance != 0.0:
            raise RuntimeError(f"{anchor.sample_id}: anchor coordinate drift")
        selected, _ = deterministic_nested_query_prefix(
            sample_id=anchor.sample_id, anchor_indices=anchor_indices, full_q=full_q,
            target_count=RESOLUTION, geometry_cache=state["geometry"],
        )
        selected_cv, _ = conservative_selected_control_volume(
            full_coords=state["coords"], full_control_volume=state["cv"],
            full_layer_id=state["layer"], selected_indices=selected, query_workers=1,
        )
        anchor_support = {
            "selected_indices": anchor_indices,
            "operator_control_volume": np.asarray(anchor.operator_point_weights, dtype=np.float64),
            "k_xyz": np.asarray(anchor.condition.condition_features[:, :3], dtype=np.float64),
            "q_W_m3": np.asarray(anchor.condition.condition_features[:, 3], dtype=np.float64),
            "layer_id": state["layer"][anchor_indices],
        }
        query_support = {
            "selected_indices": selected, "operator_control_volume": selected_cv,
            "k_xyz": full_k[selected], "q_W_m3": full_q[selected],
            "layer_id": state["layer"][selected],
        }
        anchor_example = highn._query_example(anchor, anchor_support, state["coords"])
        query_example = highn._query_example(anchor, query_support, state["coords"])
        anchor_builder = Heat3DGraphBuilder(**state["anchor_config"])
        query_builder = Heat3DGraphBuilder(**state["query_config"])
        with jax.default_device(cpu):
            anchor_metadata = anchor_builder.build_metadata(
                highn.runner._graph_coords_for_example(anchor_example, state["runtime"]["stats"]),
                key=state["graph_key"],
            )
            query_metadata = query_builder.build_metadata(
                highn.runner._graph_coords_for_example(query_example, state["runtime"]["stats"]),
                key=state["graph_key"],
            )
            final_e.block((anchor_metadata, query_metadata))
        counts: dict[str, dict[str, int | None]] = {}
        for name, targets, metadata in (
            ("anchor", state["anchor_targets"], anchor_metadata),
            ("query", state["query_targets"], query_metadata),
        ):
            counts[name] = {}
            for field in qualification.EDGE_FIELDS:
                value = getattr(metadata, field)
                count = None if value is None else int(np.asarray(value).shape[1])
                counts[name][field] = count
                if widen and count is not None:
                    targets[field] = max(int(targets.get(field) or 0), count)
        rows.append({
            "sample_id": anchor.sample_id,
            "support_indices_sha256": array_sha256(selected),
            "selected_cv_sha256": array_sha256(selected_cv),
            "anchor_graph": final_e.metadata_hashes(anchor_metadata),
            "query_graph": final_e.metadata_hashes(query_metadata),
            "edge_counts": counts,
        })
    return rows


def main() -> int:
    args = parse()
    if jax.devices()[0].platform != "gpu":
        raise RuntimeError("frozen E16384 confirmatory evaluation requires GPU")
    protocol = json.loads(args.protocol.read_text())
    if protocol.get("status") != "preregistered_before_test_iid_model_inference":
        raise RuntimeError("P6-A protocol is not frozen before test inference")
    frozen = protocol["frozen_confirmatory_evaluation"]
    if frozen["route"] != ROUTE or frozen["model_seed_label"] != MODEL_SEED_LABEL:
        raise RuntimeError("route/model seed binding drifted")
    state = build_state(args, protocol)
    physics_rows = materialize_physics(state, args.work_root)
    envelope_before = {
        "anchor": dict(state["anchor_targets"]),
        "query": dict(state["query_targets"]),
    }
    qualification_one = graph_qualification_pass(state, widen=True)
    envelope_after = {
        "anchor": dict(state["anchor_targets"]),
        "query": dict(state["query_targets"]),
    }
    qualification_two = graph_qualification_pass(state, widen=False)
    if qualification_one != qualification_two:
        raise RuntimeError("test_iid real graph/support replay drifted")

    runtime = state["runtime"]
    model = GraphNeuralOperator(**runtime["model_config"])
    params = highn.runner._device_params(runtime["checkpoint"]["params"])
    parameter_sha_before = tree_sha256(params)
    gpu = jax.devices("gpu")[0]

    @jax.jit
    def forward(model_params: Any, anchor_group: Any, query_group: Any,
                weights: Any, indices: Any, map_weights: Any) -> tuple[Any, Any, Any]:
        anchor_result = highn.runner._model_apply(model, model_params, anchor_group)
        anchor_scale = anchor_result["s_hat"].reshape((-1,))[0]
        query = highn.runner._model_apply(model, model_params, query_group)["raw_temperature"][:, 0, :, 0]
        query = query - highn.REFERENCE_K
        normalized = weights / jnp.sum(weights, axis=1, keepdims=True)
        query_scale = jnp.sqrt(jnp.sum(normalized * query * query, axis=1))
        support = query / query_scale[:, None] * anchor_scale
        gathered = support[jnp.arange(support.shape[0])[:, None, None], indices]
        full = jnp.sum(gathered * map_weights.astype(support.dtype), axis=2)
        return support, full, anchor_scale

    support_rows = []
    full_rows = []
    per_sample = []
    prediction_digest = hashlib.sha256()
    with h5py.File(args.full_fields, "r") as archive:
        for number, anchor in enumerate(state["anchors"], start=1):
            payload = p8.prepare_case(state, number - 1)
            device = jax.device_put((
                payload["anchor"], payload["query"], payload["weights"],
                payload["indices"], payload["map_weights"],
            ), gpu)
            final_e.block(device)
            support, prediction, anchor_scale = forward(params, *device)
            final_e.block((support, prediction, anchor_scale))
            support_np = np.asarray(support, dtype=np.float64)[0]
            prediction_np = np.asarray(prediction, dtype=np.float64)[0]
            if not np.all(np.isfinite(prediction_np)) or not np.isfinite(float(np.asarray(anchor_scale))):
                raise RuntimeError(f"{anchor.sample_id}: non-finite frozen prediction")
            truth = np.asarray(
                archive["samples/deltaT_K"][state["archive_lookup"][anchor.sample_id]],
                dtype=np.float64,
            )
            with np.load(state["physics"][anchor.sample_id]["physics_cache_file"], allow_pickle=False) as physics:
                full_q = np.asarray(physics["q_W_m3"], dtype=np.float64)
            selected = np.asarray(payload["selected"], dtype=np.int64)
            selected_cv = np.asarray(payload["selected_cv"], dtype=np.float64)
            support_row = highn._metric_row(
                support_np, truth[selected], selected_cv, state["coords"][selected],
                state["layer"][selected], full_q[selected],
            )
            full_row = highn._metric_row(
                prediction_np, truth, state["cv"], state["coords"], state["layer"], full_q,
            )
            support_rows.append(support_row)
            full_rows.append(full_row)
            full_metrics = qualification.metric_accumulate([full_row], full=True)
            support_metrics = qualification.metric_accumulate([support_row], full=False)
            per_sample.append({
                "sample_id": anchor.sample_id,
                "model_seed_label": MODEL_SEED_LABEL,
                "support": support_metrics,
                "full_field": full_metrics,
                "anchor_scale_K": float(np.asarray(anchor_scale)),
                "prediction_sha256": array_sha256(prediction_np.astype(np.float32)),
            })
            prediction_digest.update(anchor.sample_id.encode())
            prediction_digest.update(np.ascontiguousarray(prediction_np.astype(np.float32)).tobytes())
            print(f"[P6-A test confirmatory] {number}/{len(state['anchors'])}", flush=True)

    parameter_sha_after = tree_sha256(params)
    if parameter_sha_before != parameter_sha_after:
        raise RuntimeError("checkpoint parameter tree changed during evaluation")
    result = {
        "schema_version": "heat3d_v6_p1i_e16384_test_iid_confirmatory_v1",
        "status": "passed_frozen_test_iid_confirmatory",
        "route": ROUTE,
        "model_seed_label": MODEL_SEED_LABEL,
        "sample_count": len(state["anchors"]),
        "sample_ids": [anchor.sample_id for anchor in state["anchors"]],
        "sample_order_sha256": frozen["sample_order_sha256"],
        "checkpoint_epoch": int(runtime["checkpoint"]["epoch"]),
        "checkpoint_file_sha256": args.checkpoint_sha256,
        "checkpoint_parameter_sha256_before": parameter_sha_before,
        "checkpoint_parameter_sha256_after": parameter_sha_after,
        "dataset_manifest_sha256": sha256(args.manifest),
        "full_field_archive_sha256": sha256(args.full_fields),
        "accuracy": {
            "support": qualification.metric_accumulate(support_rows, full=False),
            "full_field": qualification.metric_accumulate(full_rows, full=True),
        },
        "per_sample_metrics": per_sample,
        "prediction_stream_sha256": prediction_digest.hexdigest(),
        "graph_replay": {
            "qualification_pass_count": 2,
            "all_real_graph_hashes_exact": True,
            "rows": qualification_one,
            "padding_envelope_before": envelope_before,
            "padding_envelope_after_monotonic_dummy_only_expansion": envelope_after,
        },
        "label_independent_physics_caches": physics_rows,
        "selection_statement": "one-time corrected confirmatory holdout; not used for selection, tuning, threshold revision, or route changes",
        "timing_collected": False,
        "role_contract": {
            "accessed_roles": ["train_inputs_for_frozen_standardizer", "test_iid"],
            "checkpoint_modified": False,
            "model_or_route_selection": False,
            "sealed_iid": False,
            "test_iid": True,
            "training": False,
        },
    }
    write_json(args.output, result)
    print(json.dumps({
        "status": result["status"],
        "sample_count": result["sample_count"],
        "full_field_point_global_rmse_pct": result["accuracy"]["full_field"]["point_global_true_rms_relative_rmse_pct"],
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
