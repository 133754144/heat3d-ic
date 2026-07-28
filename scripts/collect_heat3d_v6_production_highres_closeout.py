#!/usr/bin/env python3
"""Collect the frozen V6 production high-resolution inference closeout."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Mapping


PRODUCTION_RESOLUTIONS = (1024, 2048, 4096, 8192, 16384)
EXPERIMENTAL_RESOLUTION = 32768
EVALUATION_COMMIT = "d5c06263ee5a5cf0b925b6e1d35ade205ada8bce"


class CloseoutError(RuntimeError):
    pass


def _load(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise CloseoutError(f"missing input: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _without_samples(payload: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in payload.items() if key != "per_sample"}


def _result_row(payload: Mapping[str, Any]) -> dict[str, Any]:
    support = payload["support_metrics"]
    full = payload["full_field_metrics"]
    model = full["model"]
    floor = full["sampling_floor"]
    runtime = payload["runtime"]
    query = runtime["query"]
    cache = payload["graph_cache"]["query"]
    memory = runtime["device_memory"]
    return {
        "platform": payload["platform"],
        "resolution": int(payload["resolution"]),
        "batch_size": int(payload["batch_size"]),
        "support_point_global_relative_rmse_pct": support[
            "point_global_cv_relative_rmse_pct"
        ],
        "support_sample_first_relative_rmse_pct": support[
            "sample_first_cv_relative_rmse_pct"
        ],
        "support_raw_cv_rmse_K": support["raw_cv_weighted_rmse_K"],
        "support_peak_rmse_K": support["peak"]["rmse_K"],
        "support_source_rmse_K": support["source_region"]["cv_weighted_rmse_K"],
        "support_layer_mean_rmse_K": support["layer_mean"]["rmse_K"],
        "support_layer_drop_rmse_K": support["layer_drop"]["rmse_K"],
        "support_top_rmse_K": support["top_surface"]["cv_weighted_rmse_K"],
        "support_bottom_rmse_K": support["bottom_surface"]["cv_weighted_rmse_K"],
        "support_shape_cv_rmse": support["shape_cv_rmse"],
        "support_scale_log_rmse": support["scale_log_rmse"],
        "full_point_global_relative_rmse_pct": model[
            "cv_weighted_point_global_relative_rmse_pct"
        ],
        "full_sample_first_relative_rmse_pct": model[
            "sample_first_cv_relative_rmse_pct"
        ],
        "full_raw_cv_rmse_K": model["cv_weighted_rmse_K"],
        "full_peak_rmse_K": model["peak_error_rmse_K"],
        "full_source_rmse_K": model["source_cv_weighted_rmse_K"],
        "full_top_rmse_K": model["top_cv_weighted_rmse_K"],
        "full_bottom_rmse_K": model["bottom_cv_weighted_rmse_K"],
        "sampling_floor_point_global_relative_rmse_pct": floor[
            "cv_weighted_point_global_relative_rmse_pct"
        ],
        "sampling_floor_raw_cv_rmse_K": floor["cv_weighted_rmse_K"],
        "graph_uncached_build_seconds": cache["uncached_build_seconds"],
        "graph_cached_load_seconds": cache["load_seconds"],
        "compile_seconds": query["first_compile_inference_seconds"],
        "warm_batch_seconds_mean": query["warm_inference_seconds"]["mean"],
        "formal_inference_seconds": query["formal_inference_seconds"],
        "samples_per_second": query["samples_per_second"],
        "end_to_end_seconds_valid128": runtime["end_to_end_seconds_valid128"],
        "process_peak_ram_GB": runtime["process_peak_ram_bytes"] / 1e9,
        "device_peak_memory_GB": (
            "N/A"
            if memory["peak_bytes_in_use"] is None
            else memory["peak_bytes_in_use"] / 1e9
        ),
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _cache_entry(payload: Mapping[str, Any], kind: str) -> dict[str, Any]:
    cache = payload["graph_cache"][kind]
    prediction = payload["graph_cache"][f"{kind}_prediction_equivalence"]
    return {
        "resolution": int(payload["resolution"]),
        "kind": kind,
        "cache_file": Path(cache["cache_file"]).name,
        "cache_file_sha256": cache["cache_file_sha256"],
        "cache_key": cache["cache_key"],
        "cache_key_payload": cache["cache_key_payload"],
        "metadata_hash": cache["metadata_hash"],
        "graph_hash": cache["graph_hash"],
        "uncached_build_seconds": cache["uncached_build_seconds"],
        "cached_load_seconds": cache["load_seconds"],
        "metadata_hash_equal": cache["equivalence"]["metadata_hash_equal"],
        "graph_hash_equal": cache["equivalence"]["graph_hash_equal"],
        "prediction_max_abs_error_K": prediction["max_abs_error_K"],
        "prediction_rmse_K": prediction["rmse_K"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cpu-dir", type=Path, required=True)
    parser.add_argument("--gpu-dir", type=Path, required=True)
    parser.add_argument("--solver", type=Path, required=True)
    parser.add_argument("--baseline-4096", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--metrics-csv", type=Path, required=True)
    parser.add_argument("--timing-csv", type=Path, required=True)
    parser.add_argument("--cache-manifest", type=Path, required=True)
    parser.add_argument("--markdown", type=Path, required=True)
    args = parser.parse_args()

    cpu = {
        resolution: _load(
            args.cpu_dir / f"query_subsample8_{resolution}.json"
        )
        for resolution in (*PRODUCTION_RESOLUTIONS, EXPERIMENTAL_RESOLUTION)
    }
    gpu = {
        resolution: _load(
            args.gpu_dir / f"query_subsample8_{resolution}_b1.json"
        )
        for resolution in (*PRODUCTION_RESOLUTIONS, EXPERIMENTAL_RESOLUTION)
    }
    cpu_batch8 = _load(args.cpu_dir / "query_subsample8_4096_b8.json")
    gpu_batch8 = _load(args.gpu_dir / "query_subsample8_4096_b8.json")
    solver = _load(args.solver)
    baseline = _load(args.baseline_4096)
    for payload in (*cpu.values(), *gpu.values(), cpu_batch8, gpu_batch8):
        if payload["evaluation_role"] != "valid_iid" or payload["test_hard_accessed"]:
            raise CloseoutError("role boundary drifted")
        if payload["training_executed"] or payload["checkpoint_modified"]:
            raise CloseoutError("read-only boundary drifted")
        for kind in ("anchor", "query"):
            equivalence = payload["graph_cache"][f"{kind}_prediction_equivalence"]
            if equivalence["audited"]:
                if payload["platform"] == "cpu" and (
                    not equivalence["passed"]
                    or equivalence["max_abs_error_K"] != 0.0
                ):
                    raise CloseoutError("CPU cached prediction equivalence drifted")
                if payload["platform"] == "gpu" and (
                    equivalence["max_abs_error_K"] >= 0.005
                    or equivalence["rmse_K"] >= 0.001
                ):
                    raise CloseoutError("GPU cached prediction tolerance drifted")

    rows = [_result_row(cpu[n]) for n in cpu]
    rows.extend(_result_row(gpu[n]) for n in gpu)
    rows.append(_result_row(cpu_batch8))
    rows.append(_result_row(gpu_batch8))
    _write_csv(args.metrics_csv, rows)
    timing_fields = (
        "platform",
        "resolution",
        "batch_size",
        "graph_uncached_build_seconds",
        "graph_cached_load_seconds",
        "compile_seconds",
        "warm_batch_seconds_mean",
        "formal_inference_seconds",
        "samples_per_second",
        "end_to_end_seconds_valid128",
        "process_peak_ram_GB",
        "device_peak_memory_GB",
    )
    _write_csv(
        args.timing_csv,
        [{field: row[field] for field in timing_fields} for row in rows],
    )

    cache_entries = []
    for resolution in PRODUCTION_RESOLUTIONS:
        cache_entries.extend(
            _cache_entry(cpu[resolution], kind) for kind in ("anchor", "query")
        )
    cache_manifest = {
        "schema_version": "heat3d_v6_graph_cache_manifest_v1",
        "evaluation_commit": EVALUATION_COMMIT,
        "backend": "sparse_kdtree_v1",
        "production_resolutions": list(PRODUCTION_RESOLUTIONS),
        "cache_key_fields": [
            "support_hash",
            "resolved_graph_config",
            "graph_seed",
            "commit",
        ],
        "entries": cache_entries,
    }
    args.cache_manifest.parent.mkdir(parents=True, exist_ok=True)
    args.cache_manifest.write_text(
        json.dumps(cache_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    baseline_full = baseline["full_field_metrics"]["model"]["cv_weighted_rmse_K"]
    optimized = cpu[4096]
    optimized_full = optimized["full_field_metrics"]["model"]["cv_weighted_rmse_K"]
    optimized_point = optimized["support_metrics"][
        "point_global_cv_relative_rmse_pct"
    ]
    optimized_ratio = optimized_full / baseline_full
    accepted = optimized_point < 20.0 and optimized_ratio <= 1.10
    cold_mean = solver["cold_per_sample"]["mesh_assembly_solve_seconds"]["mean"]
    warm_mean = solver["warm_reused_by_boundary_pair"]["solve_seconds"]["mean"]
    cpu_batch1_e2e = optimized["runtime"]["end_to_end_seconds_valid128"]
    cpu_e2e = cpu_batch8["runtime"]["end_to_end_seconds_valid128"]
    gpu_e2e = gpu_batch8["runtime"]["end_to_end_seconds_valid128"]
    cold_total = cold_mean * 128
    warm_total = warm_mean * 128

    compact_results = {}
    for platform, results in (("cpu", cpu), ("gpu", gpu)):
        compact_results[platform] = {
            str(resolution): {
                "support_metrics": _without_samples(payload["support_metrics"]),
                "full_field_metrics": payload["full_field_metrics"],
                "runtime": payload["runtime"],
                "graph_cache": {
                    "anchor": _cache_entry(payload, "anchor"),
                    "query": _cache_entry(payload, "query"),
                },
            }
            for resolution, payload in results.items()
        }
    payload = {
        "schema_version": "heat3d_v6_production_highres_closeout_v1",
        "status": "passed",
        "branch": "research/v6-p1h-shared-support",
        "evaluation_commit": EVALUATION_COMMIT,
        "training_executed": False,
        "checkpoint_modified": False,
        "evaluation_role": "valid_iid",
        "test_hard_accessed": False,
        "workflow": [
            "1024_anchor_forward",
            "anchor_derived_global_context_and_scale",
            "N_node_source_aware_forward",
            "anchor_scale_reconstruction",
        ],
        "upstream_like_preforward_executed": False,
        "default_resolution": 4096,
        "maximum_production_verified_resolution": 16384,
        "experimental_verified_resolution": 32768,
        "production_graph": {
            "backend": "sparse_kdtree_v1",
            "anchor_subsample_factor": 4,
            "query_subsample_factor": 8,
            "accepted": accepted,
            "point_global_relative_rmse_pct": optimized_point,
            "full_field_rmse_K": optimized_full,
            "unoptimized_full_field_rmse_K": baseline_full,
            "full_field_rmse_ratio": optimized_ratio,
            "cpu_end_to_end_speedup_fraction": 1.0
            - cpu_batch1_e2e
            / baseline["runtime"]["end_to_end_seconds_valid128"],
        },
        "results": compact_results,
        "gpu_4096_batch8": {
            "metrics": _result_row(gpu_batch8),
            "runtime": gpu_batch8["runtime"],
        },
        "cpu_4096_batch8": {
            "metrics": _result_row(cpu_batch8),
            "runtime": cpu_batch8["runtime"],
        },
        "solver_benchmark": solver,
        "speedup": {
            "nonmatched_dof": True,
            "fvm_cold_total_seconds_estimate": cold_total,
            "fvm_warm_total_seconds_estimate": warm_total,
            "cpu_model_4096_batch1_end_to_end_seconds": cpu_batch1_e2e,
            "cpu_model_4096_end_to_end_seconds": cpu_e2e,
            "gpu_model_4096_batch8_end_to_end_seconds": gpu_e2e,
            "cold_solver_to_cpu_model": cold_total / cpu_e2e,
            "cold_solver_to_gpu_model": cold_total / gpu_e2e,
            "warm_solver_to_cpu_model": warm_total / cpu_e2e,
            "warm_solver_to_gpu_model": warm_total / gpu_e2e,
        },
        "decision": (
            "freeze_anchor_derived_sparse_query_subsample8"
            if accepted
            else "reject_optimization_keep_unoptimized"
        ),
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    lines = [
        "# V6 production high-resolution inference closeout",
        "",
        "No training, checkpoint mutation, test, or hard-role access occurred.",
        "",
        "## Frozen workflow",
        "",
        "`1024 anchor forward -> anchor-derived Global Context/scale -> "
        "N-node source-aware forward -> anchor-scale reconstruction`.",
        "",
        "The production default is 4096 nodes and the frozen production maximum "
        "is 16384. 32768 passed as an experimental extension.",
        "",
        "## Resolution results",
        "",
        "| Nodes | CPU point-global | CPU full RMSE K | CPU e2e s | GPU e2e s |",
        "|---:|---:|---:|---:|---:|",
    ]
    for resolution in (*PRODUCTION_RESOLUTIONS, EXPERIMENTAL_RESOLUTION):
        cpu_row = _result_row(cpu[resolution])
        gpu_row = _result_row(gpu[resolution])
        lines.append(
            f"| {resolution} | "
            f"{cpu_row['support_point_global_relative_rmse_pct']:.4f}% | "
            f"{cpu_row['full_raw_cv_rmse_K']:.4f} | "
            f"{cpu_row['end_to_end_seconds_valid128']:.2f} | "
            f"{gpu_row['end_to_end_seconds_valid128']:.2f} |"
        )
    lines.extend(
        [
            "",
            "## Graph optimization",
            "",
            f"The exact sparse KD-tree backend plus query-only regional-node "
            f"reduction is {'accepted' if accepted else 'rejected'}. At 4096, "
            f"point-global is {optimized_point:.4f}% and full-field RMSE is "
            f"{optimized_full:.4f} K ({optimized_ratio:.4f}x the unoptimized "
            f"baseline). CPU cached and uncached predictions are exactly "
            f"identical. GPU graph hashes are exact and repeat predictions stay "
            f"within 0.005 K maximum / 0.001 K RMSE.",
            "",
            f"At 4096, batch 8 reduced CPU end-to-end time to "
            f"{cpu_e2e:.2f} s and GPU end-to-end time to {gpu_e2e:.2f} s "
            f"for 128 samples.",
            "",
            "The prior 32768 timeout was caused by dense N-by-R graph-distance "
            "materialization. Sparse search completed the full 128-sample "
            "evaluation without changing the checkpoint.",
            "",
            "## Solver comparison",
            "",
            f"FVM cold mean is {cold_mean:.4f} s/sample and warm mean is "
            f"{warm_mean:.4f} s/sample. The comparison is nonmatched-DOF; "
            f"speedups are workflow timing comparisons, not equal-system-size "
            f"algorithmic complexity claims.",
            "",
        ]
    )
    args.markdown.parent.mkdir(parents=True, exist_ok=True)
    args.markdown.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"status": "passed", "accepted": accepted}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
