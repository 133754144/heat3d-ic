#!/usr/bin/env python3
"""Publication timing for cached P1i GPU inference and reconstruction apply.

Qualification (hashes/equivalence), metrics, and serialization are deliberately
outside the three production timing regions emitted by this script.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import sys
import time
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np


ROOT = Path(os.environ.get("HEAT3D_REPO_ROOT", Path(__file__).resolve().parents[1])).resolve()
for value in (ROOT, ROOT / "scripts"):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))

import run_heat3d_v6_p1i_anchor_high_n_development as highn  # noqa: E402
from rigno.graphBuilder_Heat3D import Heat3DGraphBuilder  # noqa: E402
from rigno.heat3d_graph_cache import METADATA_FIELDS  # noqa: E402
from rigno.heat3d_v6_full_field import ReconstructionMap  # noqa: E402
from rigno.heat3d_v6_gpu_reconstruction import to_device_reconstruction_map  # noqa: E402
from rigno.models.rigno import RIGNO as GraphNeuralOperator  # noqa: E402
from rigno.models.rigno import RegionInteractionGraphMetadata  # noqa: E402


def _distribution(values: list[float]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(len(array)),
        "median_seconds": float(np.median(array)),
        "mean_seconds": float(np.mean(array)),
        "std_seconds": float(np.std(array)),
        "p95_seconds": float(np.percentile(array, 95)),
    }


def _load_metadata_no_audit(path: Path) -> RegionInteractionGraphMetadata:
    """Load cached arrays without hashing or rebuilding the graph."""
    with np.load(path, allow_pickle=False) as payload:
        none = {
            value.decode("utf-8")
            for value in np.asarray(payload["__none_fields_utf8"]).tolist()
        }
        values = {
            field: None if field in none else jnp.asarray(np.asarray(payload[field]))
            for field in METADATA_FIELDS
        }
    return RegionInteractionGraphMetadata(**values)


def _load_mapping_no_audit(path: Path) -> ReconstructionMap:
    """Load the frozen map without SHA calculation inside production timing."""
    with np.load(path, allow_pickle=False) as payload:
        return ReconstructionMap(
            support_indices=np.asarray(payload["support_indices"], dtype=np.int32),
            neighbor_local_indices=np.asarray(payload["neighbor_local_indices"], dtype=np.int32),
            neighbor_weights=np.asarray(payload["neighbor_weights"], dtype=np.float64),
            domain_code=np.asarray(payload["domain_code"], dtype=np.int16),
            domain_names=tuple(value.decode() for value in np.asarray(payload["domain_names"]).tolist()),
        )


def _difference(left: np.ndarray, right: np.ndarray) -> dict[str, float]:
    delta = np.asarray(left, dtype=np.float64) - np.asarray(right, dtype=np.float64)
    return {
        "max_abs_error_K": float(np.max(np.abs(delta))),
        "rmse_K": float(np.sqrt(np.mean(np.square(delta)))),
    }


def _synchronize(value: Any) -> None:
    jax.block_until_ready(value)


def _device_memory() -> dict[str, Any]:
    device = jax.devices()[0]
    try:
        stats = device.memory_stats() or {}
    except Exception:
        stats = {}
    return {
        "device": str(device),
        "platform": device.platform,
        "bytes_in_use": stats.get("bytes_in_use"),
        "peak_bytes_in_use": stats.get("peak_bytes_in_use"),
        "bytes_limit": stats.get("bytes_limit"),
    }


def _load_resolution(args: argparse.Namespace, resolution: int) -> dict[str, Any]:
    result_path = args.artifact_root / f"resolution_{resolution}.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if result["status"] != "passed" or result["sample_ids"] != args.sample_ids:
        raise RuntimeError(f"N={resolution}: frozen result/sample order is not qualified")
    graph_rows = {row["sample_id"]: row for row in result["graph_cache"]["samples"]}
    map_rows = {row["sample_id"]: row for row in result["reconstruction_cache"]["samples"]}
    support_rows = {
        row["sample_id"]: row
        for row in args.preflight["supports"][str(resolution)]
    }
    return {
        "result": result,
        "graphs": graph_rows,
        "maps": map_rows,
        "supports": support_rows,
        "edge_targets": result["graph_cache"]["edge_targets"],
    }


def _prepare_case(
    *,
    sample_id: str,
    resolution_payload: dict[str, Any],
    anchors_by_id: dict[str, Any],
    full_coords: np.ndarray,
    runtime: dict[str, Any],
) -> tuple[dict[str, Any], ReconstructionMap, Any, np.ndarray, float]:
    support_row = resolution_payload["supports"][sample_id]
    with np.load(support_row["support_file"], allow_pickle=False) as payload:
        support = {name: np.asarray(payload[name]) for name in payload.files}
    anchor = anchors_by_id[sample_id]
    example = highn._query_example(anchor, support, full_coords)
    graph_row = resolution_payload["graphs"][sample_id]
    metadata = _load_metadata_no_audit(Path(graph_row["cache_file"]))
    builder = Heat3DGraphBuilder(**runtime["graph_config"])
    group = highn._prepare_group(
        example=example,
        anchor=anchor,
        runtime=runtime,
        builder=builder,
        metadata=metadata,
        edge_targets=resolution_payload["edge_targets"],
    )
    mapping = _load_mapping_no_audit(Path(resolution_payload["maps"][sample_id]["cache_file"]))
    device_mapping = to_device_reconstruction_map(mapping)
    operator_weights = np.asarray(example.operator_point_weights, dtype=np.float32)
    anchor_scale = float(args_anchor_scales[sample_id])
    return highn._model_group(group), mapping, device_mapping, operator_weights, anchor_scale


# Set once in main so _prepare_case remains easy to call inside the timed loop.
args_anchor_scales: dict[str, float] = {}


def benchmark_resolution(
    args: argparse.Namespace,
    resolution: int,
    runtime: dict[str, Any],
    anchors: list[Any],
    full_coords: np.ndarray,
    model: Any,
    params: Any,
) -> dict[str, Any]:
    payload = _load_resolution(args, resolution)
    anchors_by_id = {anchor.sample_id: anchor for anchor in anchors}

    @jax.jit
    def model_core(model_params: Any, group: dict[str, Any], weights: Any, anchor_scale: Any) -> Any:
        output = highn.runner._model_apply(model, model_params, group)
        raw = output["raw_temperature"][0, 0, :, 0]
        delta = raw - highn.REFERENCE_K
        normalized = weights / jnp.sum(weights)
        query_scale = jnp.sqrt(jnp.sum(normalized * delta * delta))
        return delta / query_scale * anchor_scale

    @jax.jit
    def production_apply(
        model_params: Any,
        group: dict[str, Any],
        weights: Any,
        anchor_scale: Any,
        neighbor_indices: Any,
        neighbor_weights: Any,
    ) -> tuple[Any, Any]:
        support_delta = model_core(model_params, group, weights, anchor_scale)
        full_delta = jnp.sum(
            support_delta[neighbor_indices] * neighbor_weights.astype(support_delta.dtype),
            axis=1,
        )
        return support_delta, full_delta

    first_id = args.sample_ids[0]
    first = _prepare_case(
        sample_id=first_id,
        resolution_payload=payload,
        anchors_by_id=anchors_by_id,
        full_coords=full_coords,
        runtime=runtime,
    )
    first_group, first_cpu_map, first_device_map, first_weights, first_scale = first
    # Qualification/compile is explicitly outside all production timing states.
    compile_started = time.perf_counter()
    compiled = production_apply(
        params, first_group, jnp.asarray(first_weights), jnp.asarray(first_scale),
        first_device_map.neighbor_local_indices, first_device_map.neighbor_weights,
    )
    _synchronize(compiled[1])
    compile_seconds = time.perf_counter() - compile_started
    core_compile_started = time.perf_counter()
    core = model_core(params, first_group, jnp.asarray(first_weights), jnp.asarray(first_scale))
    _synchronize(core)
    core_compile_seconds = time.perf_counter() - core_compile_started

    neural_forward: list[float] = []
    warm_cache: list[float] = []
    for _ in range(args.repeats):
        started = time.perf_counter()
        support_delta = model_core(
            params, first_group, jnp.asarray(first_weights), jnp.asarray(first_scale)
        )
        _synchronize(support_delta)
        neural_forward.append(time.perf_counter() - started)
        started = time.perf_counter()
        _, full_delta = production_apply(
            params, first_group, jnp.asarray(first_weights), jnp.asarray(first_scale),
            first_device_map.neighbor_local_indices, first_device_map.neighbor_weights,
        )
        _synchronize(full_delta)
        warm_cache.append(time.perf_counter() - started)

    # New-case includes cache loads, group preparation, H2D map transfer, forward,
    # and reconstruction. It never builds a graph/map or hashes an artifact.
    new_case: list[float] = []
    equivalence_rows = []
    for sample_id in args.sample_ids:
        started = time.perf_counter()
        group, cpu_map, device_map, weights, anchor_scale = _prepare_case(
            sample_id=sample_id,
            resolution_payload=payload,
            anchors_by_id=anchors_by_id,
            full_coords=full_coords,
            runtime=runtime,
        )
        support_delta, gpu_full = production_apply(
            params, group, jnp.asarray(weights), jnp.asarray(anchor_scale),
            device_map.neighbor_local_indices, device_map.neighbor_weights,
        )
        _synchronize(gpu_full)
        new_case.append(time.perf_counter() - started)
        # Equivalence is intentionally evaluated after the production timer.
        cpu_full = cpu_map.reconstruct(np.asarray(support_delta, dtype=np.float64))
        equivalence_rows.append(_difference(cpu_full, np.asarray(gpu_full)))

    maximum = max(row["max_abs_error_K"] for row in equivalence_rows)
    maximum_rmse = max(row["rmse_K"] for row in equivalence_rows)
    tolerance = {"max_abs_error_K": 1.0e-4, "rmse_K": 1.0e-5}
    equivalent = maximum <= tolerance["max_abs_error_K"] and maximum_rmse <= tolerance["rmse_K"]
    if not equivalent:
        raise RuntimeError(f"N={resolution}: CPU/GPU reconstruction equivalence failed")
    return {
        "resolution": resolution,
        "status": "passed",
        "timing": {
            "new_case": _distribution(new_case),
            "warm_cache": _distribution(warm_cache),
            "neural_forward": _distribution(neural_forward),
            "qualification_compile_seconds_excluded": compile_seconds,
            "qualification_core_compile_seconds_excluded": core_compile_seconds,
        },
        "gpu_reconstruction_equivalence": {
            "status": "passed",
            "sample_count": len(equivalence_rows),
            "maximum_sample_max_abs_error_K": maximum,
            "maximum_sample_rmse_K": maximum_rmse,
            "tolerance": tolerance,
            "map_build_performed": False,
            "map_indices_weights_algorithm_changed": False,
        },
        "device_memory": _device_memory(),
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    flattened = []
    for row in rows:
        flattened.append({
            "resolution": row["resolution"],
            "new_case_median_seconds": row["timing"]["new_case"]["median_seconds"],
            "new_case_p95_seconds": row["timing"]["new_case"]["p95_seconds"],
            "warm_cache_median_seconds": row["timing"]["warm_cache"]["median_seconds"],
            "warm_cache_p95_seconds": row["timing"]["warm_cache"]["p95_seconds"],
            "neural_forward_median_seconds": row["timing"]["neural_forward"]["median_seconds"],
            "neural_forward_p95_seconds": row["timing"]["neural_forward"]["p95_seconds"],
            "reconstruction_max_abs_error_K": row["gpu_reconstruction_equivalence"]["maximum_sample_max_abs_error_K"],
            "reconstruction_max_rmse_K": row["gpu_reconstruction_equivalence"]["maximum_sample_rmse_K"],
            "peak_vram_bytes": row["device_memory"]["peak_bytes_in_use"],
        })
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(flattened[0]))
        writer.writeheader()
        writer.writerows(flattened)


def _write_md(path: Path, rows: list[dict[str, Any]]) -> None:
    lines = [
        "# P1i publication GPU timing", "",
        "Production states exclude qualification, fresh graph/map construction, hashing, metrics, and serialization. Every GPU interval ends after device synchronization.", "",
        "| N | new-case median/p95 (ms) | warm-cache median/p95 (ms) | neural-forward median/p95 (ms) | CPU-GPU recon max (K) |",
        "|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        t = row["timing"]
        lines.append(
            f"| {row['resolution']} | {1000*t['new_case']['median_seconds']:.3f}/{1000*t['new_case']['p95_seconds']:.3f} | "
            f"{1000*t['warm_cache']['median_seconds']:.3f}/{1000*t['warm_cache']['p95_seconds']:.3f} | "
            f"{1000*t['neural_forward']['median_seconds']:.3f}/{1000*t['neural_forward']['p95_seconds']:.3f} | "
            f"{row['gpu_reconstruction_equivalence']['maximum_sample_max_abs_error_K']:.3e} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binding", type=Path, required=True)
    parser.add_argument("--gpu-only-amendment", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--full-fields", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--resolutions", type=int, nargs="+", default=[4096, 8192, 16384, 32768, 65536])
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if jax.devices()[0].platform != "gpu":
        raise RuntimeError("publication timing requires the formal GPU backend")
    binding = highn._binding(args)
    highn._protocol_amendment(args)
    args.preflight = json.loads((args.artifact_root / "actual_data_preflight.json").read_text())
    args.sample_ids = list(binding["development_subset"]["sample_ids"])
    runtime = highn._checkpoint_runtime(args)
    dataset = highn._dataset(args)
    anchors = highn._valid_examples(dataset, binding)
    full, _ = highn._full_shared(args)
    with np.load(args.baseline_root / "resolution_1024_predictions.npz", allow_pickle=False) as payload:
        sample_ids = [str(value) for value in np.asarray(payload["sample_ids"]).tolist()]
        scales = np.asarray(payload["predicted_scales"], dtype=np.float64)
    if sample_ids != args.sample_ids:
        raise RuntimeError("frozen anchor-scale sample order drifted")
    global args_anchor_scales
    args_anchor_scales = dict(zip(sample_ids, map(float, scales), strict=True))
    model = GraphNeuralOperator(**runtime["model_config"])
    params = highn.runner._device_params(runtime["checkpoint"]["params"])
    rows = [
        benchmark_resolution(args, resolution, runtime, anchors, full["coords"], model, params)
        for resolution in args.resolutions
    ]
    result = {
        "schema_version": "heat3d_v6_p1i_publication_gpu_timing_v1",
        "status": "passed",
        "timing_contract": {
            "batch_size": 1,
            "gpu_synchronized": True,
            "new_case": "cached graph/map load + group/input prepare + H2D map transfer + model-core + GPU reconstruction apply",
            "warm_cache": "in-memory group/device map + model-core + GPU reconstruction apply",
            "neural_forward": "in-memory group + anchor-derived model-core; no full-field reconstruction",
            "excluded": ["fresh_graph_build", "map_build", "hash", "equivalence", "metrics", "labels", "serialization"],
        },
        "role_contract": {
            "accessed_roles": ["train_inputs_for_frozen_standardizer", "valid_iid_inputs"],
            "training": False, "test": False, "sealed": False, "three_seed_valid128": False,
        },
        "results": rows,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    _write_csv(args.output_csv, rows)
    _write_md(args.output_md, rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
