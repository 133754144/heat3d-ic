#!/usr/bin/env python3
"""Final monotonic-padding numerical gate for the frozen V6 publication routes.

The orchestrator launches one independent GPU process per neural route.  Each
worker reuses one real graph per sample and changes only masked dummy capacity.
Results are written before any failed gate is returned to the orchestrator.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping

import jax
import jax.numpy as jnp
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for value in (ROOT, ROOT / "scripts"):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))

import benchmark_heat3d_v6_p1i_final_e_service as e_service  # noqa: E402
import benchmark_heat3d_v6_p1i_p8_throughput_fairness as p8  # noqa: E402
import run_heat3d_v6_p1i_anchor_high_n_development as highn  # noqa: E402
import run_heat3d_v6_p1i_p5r_resolution_cell as p5r  # noqa: E402
import run_heat3d_v6_p1i_u1_split_adapter as u1  # noqa: E402
from rigno.graphBuilder_Heat3D import Heat3DGraphBuilder  # noqa: E402
from rigno.heat3d_v6_full_field import build_reconstruction_map  # noqa: E402
from rigno.heat3d_v6_p1i_anchor_query import (  # noqa: E402
    conservative_selected_control_volume,
    deterministic_nested_query_prefix,
)
from rigno.models.rigno import RIGNO as GraphNeuralOperator  # noqa: E402


ROUTES = (
    "E16384_reconstruction",
    "U_v2_16384_reconstruction",
    "E240825_direct_control",
    "U_v2_direct240825",
)
SAMPLE_IDS = ("v6p1if1_0308", "v6p1if1_0029")
EDGE_FIELDS = (
    "p2r_edge_indices", "r2r_edge_indices", "r2r_edge_domains", "r2p_edge_indices",
)


def sha_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def block(tree: Any) -> None:
    jax.tree_util.tree_map(
        lambda value: value.block_until_ready() if hasattr(value, "block_until_ready") else value,
        tree,
    )


def host_tree(tree: Any) -> Any:
    return jax.tree_util.tree_map(lambda value: np.asarray(jax.device_get(value)), tree)


def edge_counts(metadata: Any) -> dict[str, int | None]:
    return {
        field: (None if getattr(metadata, field, None) is None
                else int(np.asarray(getattr(metadata, field)).shape[1]))
        for field in EDGE_FIELDS
    }


def capacity_fits(capacity: Mapping[str, int | None], metadata: Any) -> bool:
    for field, actual in edge_counts(metadata).items():
        if actual is not None and (capacity.get(field) is None or int(capacity[field]) < actual):
            return False
    return True


def component_delta(reference: tuple[np.ndarray, np.ndarray, np.ndarray],
                    candidate: tuple[np.ndarray, np.ndarray, np.ndarray]) -> dict[str, Any]:
    names = ("anchor_scale", "query", "final240825")
    pieces = []
    result: dict[str, Any] = {}
    for name, left, right in zip(names, reference, candidate, strict=True):
        delta = np.asarray(right, dtype=np.float64) - np.asarray(left, dtype=np.float64)
        flat = delta.reshape(-1)
        pieces.append(flat)
        result[name] = {
            "max_abs_K": float(np.max(np.abs(flat))),
            "RMSE_K": float(np.sqrt(np.mean(flat * flat))),
        }
    combined = np.concatenate(pieces)
    result["max_abs_K"] = float(np.max(np.abs(combined)))
    result["RMSE_K"] = float(np.sqrt(np.mean(combined * combined)))
    return result


def parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    for name in (
        "protocol", "binding", "artifact_root", "dataset_root", "manifest", "full_fields",
        "run_dir", "native_padding_result", "previous_seal", "qualification_run",
        "output", "output_root",
    ):
        parser.add_argument(f"--{name.replace('_', '-')}", dest=name, type=Path, required=True)
    parser.add_argument("--e16384-padding-result", type=Path, required=True)
    parser.add_argument("--e240825-padding-result", type=Path, required=True)
    parser.add_argument("--u16384-padding-result", type=Path, required=True)
    parser.add_argument("--u240825-padding-result", type=Path, required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--checkpoint-epoch", type=int, default=559)
    parser.add_argument("--worker-route", choices=ROUTES)
    parser.add_argument("--worker-output", type=Path)
    return parser.parse_args()


def query_padding(args: argparse.Namespace, route: str) -> Path:
    return {
        "E16384_reconstruction": args.e16384_padding_result,
        "E240825_direct_control": args.e240825_padding_result,
        "U_v2_16384_reconstruction": args.u16384_padding_result,
        "U_v2_direct240825": args.u240825_padding_result,
    }[route]


def route_properties(route: str) -> tuple[bool, int, bool]:
    is_u = route.startswith("U")
    resolution = 240825 if "240825" in route else 16384
    return is_u, resolution, resolution == 240825


def historical_targets(seal: dict[str, Any], route: str) -> tuple[dict[str, Any], dict[str, Any]]:
    padding = seal["runtime_state"][route]["padding_envelope"]
    if route.startswith("E"):
        return dict(padding["anchor"]), dict(padding["query"])
    return dict(padding["native"]), dict(padding["query"])


def _support(state: dict[str, Any], anchor: Any, full_k: np.ndarray, full_q: np.ndarray,
             resolution: int, direct: bool) -> tuple[Any, Any, np.ndarray, np.ndarray, np.ndarray]:
    anchor_indices, distance = highn._anchor_indices(
        anchor, state["coords"],
        float(state["binding"]["numeric_tolerances"]["anchor_to_solver_coordinate_max_distance_m"]),
    )
    if distance != 0.0:
        raise RuntimeError(f"{anchor.sample_id}: anchor coordinate drift")
    if direct:
        selected = np.arange(len(state["coords"]), dtype=np.int64)
        selected_cv = np.asarray(state["cv"], dtype=np.float64)
    else:
        selected, _ = deterministic_nested_query_prefix(
            sample_id=anchor.sample_id, anchor_indices=anchor_indices,
            full_q=full_q, target_count=resolution, geometry_cache=state["geometry"],
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
    return (
        highn._query_example(anchor, anchor_support, state["coords"]),
        highn._query_example(anchor, query_support, state["coords"]),
        selected, selected_cv, anchor_indices,
    )


def worker(args: argparse.Namespace) -> int:
    route = args.worker_route
    assert route is not None and args.worker_output is not None
    if jax.devices()[0].platform != "gpu":
        raise RuntimeError("final padding gate requires the production GPU backend")
    args.query_padding_result = query_padding(args, route)
    state = p8.runtime_state(args)
    seal = json.loads(args.previous_seal.read_text())
    is_u, resolution, direct = route_properties(route)
    new_native = p5r._edge_targets(args.native_padding_result)
    new_query = p5r._edge_targets(args.query_padding_result)
    old_native, old_query = historical_targets(seal, route)
    model = GraphNeuralOperator(**state["runtime"]["model_config"])
    params = highn.runner._device_params(state["runtime"]["checkpoint"]["params"])
    checkpoint_before = highn._tree_sha256(state["runtime"]["checkpoint"]["params"])
    cpu = jax.devices("cpu")[0]
    gpu = jax.devices("gpu")[0]

    @jax.jit
    def e_forward(model_params: Any, anchor_group: Any, query_group: Any, weights: Any,
                  indices: Any, map_weights: Any) -> tuple[Any, Any, Any]:
        anchor_result = highn.runner._model_apply(model, model_params, anchor_group)
        anchor_scale = anchor_result["s_hat"].reshape(-1)[0]
        query = highn.runner._model_apply(model, model_params, query_group)["raw_temperature"][0, 0, :, 0]
        query = query - highn.REFERENCE_K
        normalized = weights / jnp.sum(weights)
        query_scale = jnp.sqrt(jnp.sum(normalized * query * query))
        support = query / query_scale * anchor_scale
        full = support if direct else jnp.sum(
            support[indices] * map_weights.astype(support.dtype), axis=1)
        return anchor_scale, support, full

    @jax.jit
    def u_forward(model_params: Any, inputs_in: Any, inputs_out: Any, graphs: Any,
                  local_p2r: Any, kwargs: Any, indices: Any,
                  map_weights: Any) -> tuple[Any, Any, Any]:
        result = model.apply(
            {"params": model_params}, inputs_in=inputs_in, inputs_out=inputs_out,
            graphs=graphs, output_local_p2r=local_p2r, split=True,
            method=u1._trace_method, **kwargs,
        )
        support = result["raw_temperature"][0, 0, :, 0] - highn.REFERENCE_K
        anchor_scale = result["s_hat"].reshape(-1)[0]
        full = support if direct else jnp.sum(
            support[indices] * map_weights.astype(support.dtype), axis=1)
        return anchor_scale, support, full

    anchors_by_id = {anchor.sample_id: anchor for anchor in state["anchors"]}
    missing = [sample_id for sample_id in SAMPLE_IDS if sample_id not in anchors_by_id]
    if missing:
        raise RuntimeError(f"frozen valid32 missing padding-gate samples: {missing}")
    records: list[dict[str, Any]] = []
    result: dict[str, Any] = {
        "schema_version": "heat3d_v6_publication_final_padding_route_gate_v1",
        "status": "running", "route": route, "sample_ids": list(SAMPLE_IDS),
        "records": records, "padding_numerical_equivalence": "PENDING",
        "checkpoint_parameter_tree_sha256_before": checkpoint_before,
        "role_contract": {
            "accessed_roles": ["valid_iid_inputs"], "temperature_truth_read": False,
            "training": False, "accuracy_tuning": False, "test": False, "sealed": False,
        },
    }

    def persist() -> None:
        args.worker_output.parent.mkdir(parents=True, exist_ok=True)
        args.worker_output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")

    failed = False
    for sample_id in SAMPLE_IDS:
        anchor = anchors_by_id[sample_id]
        with np.load(state["physics"][sample_id]["physics_cache_file"], allow_pickle=False) as archive:
            full_k = np.asarray(archive["k_xyz"], dtype=np.float64)
            full_q = np.asarray(archive["q_W_m3"], dtype=np.float64)
        anchor_example, query_example, selected, selected_cv, _ = _support(
            state, anchor, full_k, full_q, resolution, direct)
        anchor_config = dict(state["runtime"]["graph_config"])
        anchor_config.update(subsample_factor=4.0, discrete_graph_backend="sparse_kdtree_v1",
                             reuse_exact_p2r_for_r2p=True)
        query_config = dict(state["runtime"]["graph_config"])
        query_config.update(subsample_factor=resolution / 256.0,
                            discrete_graph_backend="sparse_kdtree_v1",
                            reuse_exact_p2r_for_r2p=True)
        with jax.default_device(cpu):
            anchor_builder = Heat3DGraphBuilder(**anchor_config)
            anchor_coords = highn.runner._graph_coords_for_example(
                anchor_example, state["runtime"]["stats"])
            native = anchor_builder.build_metadata(anchor_coords, key=state["graph_key"])
            block(native)
            query_coords = highn.runner._graph_coords_for_example(
                query_example, state["runtime"]["stats"])
            if is_u:
                asymmetric, audit = u1.prior_u1._u_v2_asymmetric_metadata(
                    anchor_builder, native, anchor_coords, query_coords,
                    numerical_tolerance=float(json.loads(args.protocol.read_text())["u_v2"]["normalized_numerical_tolerance"]),
                    maximum_normalized_overshoot=float(json.loads(args.protocol.read_text())["u_v2"]["maximum_normalized_overshoot"]),
                )
                block(asymmetric)
                if not all(audit["native_exact"].values()) or audit["repaired_uncovered_count"] != 0:
                    raise RuntimeError(f"{sample_id}: frozen U-v2 graph qualification drift")
                query_builder = anchor_builder
                query_metadata = asymmetric
            else:
                query_builder = Heat3DGraphBuilder(**query_config)
                query_metadata = query_builder.build_metadata(query_coords, key=state["graph_key"])
                block(query_metadata)
            exact_native = edge_counts(native)
            exact_query = edge_counts(query_metadata)
            if not capacity_fits(new_native, native) or not capacity_fits(new_query, query_metadata):
                raise RuntimeError(f"{sample_id}: monotonic capacity does not contain real graph")
            if direct:
                indices = np.zeros((1,), dtype=np.int32)
                weights = np.ones((1,), dtype=np.float64)
            else:
                mapping, _ = build_reconstruction_map(
                    coords=state["coords"], layer_id=state["layer"],
                    boundaries=state["boundaries"], support_indices=selected,
                    empty_domain_fallback="same_layer", prepared_partition=state["partition"],
                    query_workers=1,
                )
                indices = np.asarray(mapping.neighbor_local_indices, dtype=np.int32)
                weights = np.asarray(mapping.neighbor_weights, dtype=np.float64)
            graph_hash_before = {
                "native1024": e_service.metadata_hashes(native),
                "query": e_service.metadata_hashes(query_metadata),
            }

            def predict(native_capacity: Mapping[str, int | None],
                        query_capacity: Mapping[str, int | None]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
                anchor_group = host_tree(highn._model_group(highn._prepare_group(
                    example=anchor_example, anchor=anchor, runtime=state["runtime"],
                    builder=anchor_builder, metadata=native,
                    edge_targets=p5r._compatible_targets(native_capacity, native),
                )))
                if is_u:
                    combined = u1._combined_targets(native_capacity, query_capacity)
                    output_group = host_tree(u1._prepare_output_query_group_lean(
                        example=query_example, anchor=anchor, runtime=state["runtime"],
                        builder=query_builder, metadata=query_metadata,
                        edge_targets=p5r._compatible_targets(combined, query_metadata),
                    ))
                    payload = (
                        anchor_group["inputs"], output_group["inputs"], output_group["graphs"],
                        host_tree(u1._dummy_local_p2r(query_builder, query_metadata)),
                        host_tree(u1._model_kwargs(anchor_group, output_group)), indices, weights,
                    )
                    device = jax.device_put(payload, gpu); block(device)
                    prediction = u_forward(params, *device)
                else:
                    query_group = host_tree(highn._model_group(highn._prepare_group(
                        example=query_example, anchor=anchor, runtime=state["runtime"],
                        builder=query_builder, metadata=query_metadata,
                        edge_targets=p5r._compatible_targets(query_capacity, query_metadata),
                    )))
                    payload = (anchor_group, query_group, np.asarray(selected_cv, dtype=np.float32),
                               indices, weights)
                    device = jax.device_put(payload, gpu); block(device)
                    prediction = e_forward(params, *device)
                block(prediction)
                return tuple(np.asarray(value, dtype=np.float64) for value in prediction)  # type: ignore[return-value]

            new_first = predict(new_native, new_query)
            new_repeat = predict(new_native, new_query)
            same_shape = component_delta(new_first, new_repeat)
            floor = float(same_shape["max_abs_K"])
            tolerance = max(1.0e-3, 20.0 * floor)
            comparisons: dict[str, Any] = {
                "same_shape_repeat": same_shape,
            }
            required = []
            historical_available = (
                capacity_fits(old_native, native) and capacity_fits(old_query, query_metadata))
            if historical_available:
                historical_prediction = predict(old_native, old_query)
                comparisons["historical_frozen_to_new_monotonic"] = component_delta(
                    historical_prediction, new_first)
                required.append("historical_frozen_to_new_monotonic")
            else:
                comparisons["historical_frozen_to_new_monotonic"] = {
                    "status": "not_applicable_historical_capacity_cannot_contain_real_graph"}
            if sample_id == "v6p1if1_0029":
                exact_prediction = predict(exact_native, exact_query)
                comparisons["exact_real_edge_to_new_monotonic"] = component_delta(
                    exact_prediction, new_first)
                required.append("exact_real_edge_to_new_monotonic")
            graph_hash_after = {
                "native1024": e_service.metadata_hashes(native),
                "query": e_service.metadata_hashes(query_metadata),
            }
            comparison_pass = all(
                float(comparisons[name]["max_abs_K"]) <= tolerance for name in required)
            record = {
                "sample_id": sample_id,
                "sample_role": "historical_witness" if sample_id.endswith("0308") else "max_edge_witness",
                "real_edge_counts": {"native1024": exact_native, "query": exact_query},
                "real_graph_hashes_before": graph_hash_before,
                "real_graph_hashes_after": graph_hash_after,
                "real_graph_hashes_exact": graph_hash_before == graph_hash_after,
                "historical_capacity": {"native": old_native, "query": old_query},
                "new_monotonic_capacity": {"native": new_native, "query": new_query},
                "historical_capacity_can_contain_real_graph": historical_available,
                "same_shape_floor_K": floor,
                "frozen_tolerance_formula": "max(1e-3,20*same_shape_floor_K)",
                "tolerance_K": tolerance,
                "comparisons": comparisons,
                "required_comparisons": required,
                "passed": bool(comparison_pass and graph_hash_before == graph_hash_after),
            }
            records.append(record)
            failed = failed or not record["passed"]
            result["status"] = "failed" if failed else "running"
            persist()

    checkpoint_after = highn._tree_sha256(state["runtime"]["checkpoint"]["params"])
    result.update({
        "status": "failed" if failed else "passed",
        "padding_numerical_equivalence": "NO_GO" if failed else "GO",
        "checkpoint_parameter_tree_sha256_after": checkpoint_after,
        "checkpoint_unchanged": checkpoint_before == checkpoint_after,
        "real_edges_changed": False,
        "dummy_capacity_only": True,
    })
    if checkpoint_before != checkpoint_after:
        result["status"] = "failed"
        result["padding_numerical_equivalence"] = "NO_GO_checkpoint_changed"
    persist()
    return 0 if result["status"] == "passed" else 1


def orchestrate(args: argparse.Namespace) -> int:
    qualification = json.loads(args.qualification_run.read_text())
    max_rows = [
        row for row in qualification["records"]
        if row["route"] == "native1024_anchor" and row["sample_id"] == "v6p1if1_0029"
    ]
    native_max = max(
        int(row["edge_counts"]["p2r_edge_indices"])
        for row in qualification["records"] if row["route"] == "native1024_anchor"
    )
    if len(max_rows) != 1 or int(max_rows[0]["edge_counts"]["p2r_edge_indices"]) != native_max:
        raise RuntimeError("0029 is not the frozen native1024 max-edge witness")
    args.output_root.mkdir(parents=True, exist_ok=True)
    process_records = []
    route_results = []
    failure = None
    for route in ROUTES:
        output = args.output_root / f"{route}.json"
        log = args.output_root / f"{route}.log"
        command = [
            sys.executable, str(Path(__file__).resolve()),
            "--protocol", str(args.protocol), "--binding", str(args.binding),
            "--artifact-root", str(args.artifact_root), "--dataset-root", str(args.dataset_root),
            "--manifest", str(args.manifest), "--full-fields", str(args.full_fields),
            "--run-dir", str(args.run_dir), "--native-padding-result", str(args.native_padding_result),
            "--previous-seal", str(args.previous_seal),
            "--qualification-run", str(args.qualification_run),
            "--e16384-padding-result", str(args.e16384_padding_result),
            "--e240825-padding-result", str(args.e240825_padding_result),
            "--u16384-padding-result", str(args.u16384_padding_result),
            "--u240825-padding-result", str(args.u240825_padding_result),
            "--checkpoint-sha256", args.checkpoint_sha256,
            "--checkpoint-epoch", str(args.checkpoint_epoch),
            "--output", str(args.output), "--output-root", str(args.output_root),
            "--worker-route", route, "--worker-output", str(output),
        ]
        completed = subprocess.run(command, text=True, capture_output=True, env=os.environ.copy())
        log.write_text(completed.stdout + completed.stderr)
        record = {
            "route": route, "returncode": completed.returncode,
            "output": str(output), "output_sha256": sha_file(output) if output.exists() else None,
            "log": str(log), "log_sha256": sha_file(log), "command": command,
        }
        process_records.append(record)
        if output.exists():
            route_results.append(json.loads(output.read_text()))
        if completed.returncode != 0 or not output.exists():
            failure = {**record, "stderr_tail": completed.stderr[-4000:]}
            break
    passed = (
        failure is None and len(route_results) == len(ROUTES)
        and all(row["padding_numerical_equivalence"] == "GO" for row in route_results)
    )
    result = {
        "schema_version": "heat3d_v6_publication_final_padding_gate_v1",
        "status": "passed" if passed else "failed_fail_closed",
        "padding_numerical_equivalence": "GO" if passed else "NO_GO",
        "envelope_qualification": "GO_frozen_reused_not_rerun",
        "ready_for_authoritative_valid32": "GO" if passed else "NO_GO",
        "sample_ids": list(SAMPLE_IDS), "routes": list(ROUTES),
        "process_records": process_records, "route_results": route_results,
        "failure": failure,
        "numerical_gate": "max_abs_K<=max(1e-3,20*same_shape_floor_K)",
        "role_contract": {
            "accessed_roles": ["valid_iid_inputs"], "temperature_truth_read": False,
            "training": False, "accuracy_tuning": False, "test": False, "sealed": False,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": result["status"], "routes": len(route_results), "failure": failure}))
    return 0 if passed else 1


def main() -> int:
    args = parse()
    return worker(args) if args.worker_route else orchestrate(args)


if __name__ == "__main__":
    raise SystemExit(main())
