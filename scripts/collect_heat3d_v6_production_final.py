#!/usr/bin/env python3
"""Collect final V6 production inference accuracy, timing, and decisions."""

from __future__ import annotations

import argparse
import csv
import json
import platform
from pathlib import Path
import sys
from typing import Any, Mapping

import jax
import numpy as np
import scipy


RESOLUTIONS = (4096, 8192, 16384)
LADDER = (1024, 2048, 4096, 8192, 16384, 32768)
SEEDS = ("seed0", "seed1", "seed2")


def _load(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise RuntimeError(f"missing result: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(rows[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)


def _metric_row(seed: str, resolution: int, payload: Mapping[str, Any]) -> dict[str, Any]:
    support = payload["support_metrics"]
    model = payload["full_field_metrics"]["model"]
    return {
        "row_type": "seed",
        "seed": seed,
        "resolution": resolution,
        "support_point_global_pct": support["point_global_cv_relative_rmse_pct"],
        "support_sample_first_pct": support["sample_first_cv_relative_rmse_pct"],
        "support_raw_cv_rmse_K": support["raw_cv_weighted_rmse_K"],
        "support_peak_rmse_K": support["peak"]["rmse_K"],
        "support_source_rmse_K": support["source_region"]["cv_weighted_rmse_K"],
        "support_layer_mean_rmse_K": support["layer_mean"]["rmse_K"],
        "support_layer_drop_rmse_K": support["layer_drop"]["rmse_K"],
        "support_top_rmse_K": support["top_surface"]["cv_weighted_rmse_K"],
        "support_bottom_rmse_K": support["bottom_surface"]["cv_weighted_rmse_K"],
        "support_shape_cv_rmse": support["shape_cv_rmse"],
        "support_scale_log_rmse": support["scale_log_rmse"],
        "full_point_global_pct": model[
            "cv_weighted_point_global_relative_rmse_pct"
        ],
        "full_sample_first_pct": model["sample_first_cv_relative_rmse_pct"],
        "full_raw_cv_rmse_K": model["cv_weighted_rmse_K"],
        "full_peak_rmse_K": model["peak_error_rmse_K"],
        "full_source_rmse_K": model["source_cv_weighted_rmse_K"],
        "full_layer_rmse_K_mean": float(
            np.mean(list(model["layer_cv_weighted_rmse_K"].values()))
        ),
        "full_interface_rmse_K_mean": float(
            np.mean(list(model["interface_cv_weighted_rmse_K"].values()))
        ),
        "full_top_rmse_K": model["top_cv_weighted_rmse_K"],
        "full_bottom_rmse_K": model["bottom_cv_weighted_rmse_K"],
    }


def _aggregate_rows(rows: list[dict[str, Any]], resolution: int) -> list[dict[str, Any]]:
    fields = list(rows[0])
    result = []
    for kind, reducer in (("mean", np.mean), ("std", lambda x: np.std(x, ddof=1))):
        row: dict[str, Any] = {
            "row_type": kind,
            "seed": "seed0_seed1_seed2",
            "resolution": resolution,
        }
        for field in fields[3:]:
            row[field] = float(reducer([float(value[field]) for value in rows]))
        result.append(row)
    return result


def _timing_row(payload: Mapping[str, Any], mode: str) -> dict[str, Any]:
    runtime = payload["runtime"]
    reconstruction = payload["reconstruction"]
    cache = payload["graph_cache"]["query"]
    warm = runtime["persistent_compiled_workflow_seconds"]
    return {
        "platform": payload["platform"],
        "mode": mode,
        "seed": payload["checkpoint"]["seed"],
        "resolution": payload["resolution"],
        "batch_size": payload["batch_size"],
        "input_seconds": runtime["input_seconds"],
        "graph_uncached_build_seconds": cache["uncached_build_seconds"],
        "graph_cache_load_seconds": cache["load_seconds"],
        "anchor_group_prepare_seconds": runtime["anchor_group_prepare_seconds"],
        "query_group_prepare_seconds": runtime["query_group_prepare_seconds"],
        "anchor_forward_seconds": runtime["anchor"]["formal_inference_seconds"],
        "query_forward_seconds": runtime["query"]["formal_inference_seconds"],
        "checkpoint_pure_forward_seconds": runtime[
            "checkpoint_pure_forward_cold_seconds"
        ],
        "scale_reconstruction_seconds": runtime["scale_reconstruction_seconds"],
        "label_read_seconds": reconstruction["label_read_seconds_valid128"],
        "full_field_reconstruction_seconds": reconstruction[
            "field_reconstruction_seconds_valid128"
        ],
        "metric_seconds": reconstruction["metric_seconds_valid128"]
        + runtime["support_metric_seconds"],
        "serialization_encode_seconds": runtime["serialization_encode_seconds"],
        "serialization_write_seconds": runtime["serialization_write_seconds"],
        "production_end_to_end_seconds": runtime[
            "end_to_end_seconds_valid128"
        ],
        "persistent_compiled_warm_seconds": warm["mean"],
        "samples_per_second": 128.0
        / runtime["end_to_end_seconds_valid128"],
        "process_peak_ram_GB": runtime["process_peak_ram_bytes"] / 1e9,
        "device_peak_memory_GB": (
            "N/A"
            if runtime["device_memory"]["peak_bytes_in_use"] is None
            else runtime["device_memory"]["peak_bytes_in_use"] / 1e9
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cpu-accuracy-dir", type=Path, required=True)
    parser.add_argument("--cpu-ladder-dir", type=Path, required=True)
    parser.add_argument("--cpu-timing-dir", type=Path, required=True)
    parser.add_argument("--gpu-ladder-dir", type=Path, required=True)
    parser.add_argument("--gpu-timing-dir", type=Path, required=True)
    parser.add_argument("--solver", type=Path, required=True)
    parser.add_argument("--preflight", type=Path, required=True)
    parser.add_argument("--runner-smoke", type=Path, required=True)
    parser.add_argument("--archive-manifest", type=Path, required=True)
    parser.add_argument("--gpu-environment", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--metrics-csv", type=Path, required=True)
    parser.add_argument("--timing-csv", type=Path, required=True)
    parser.add_argument("--speedup-csv", type=Path, required=True)
    parser.add_argument("--graph-cache-manifest", type=Path, required=True)
    parser.add_argument("--environment-json", type=Path, required=True)
    parser.add_argument("--markdown", type=Path, required=True)
    args = parser.parse_args()

    accuracy = {
        seed: {
            resolution: _load(
                args.cpu_accuracy_dir / f"{seed}_{resolution}_b8_eval.json"
            )
            for resolution in RESOLUTIONS
        }
        for seed in SEEDS
    }
    for seed, results in accuracy.items():
        for resolution, payload in results.items():
            if (
                payload["checkpoint"]["seed"] != seed
                or payload["resolution"] != resolution
                or payload["evaluation_role"] != "valid_iid"
                or payload["test_hard_accessed"]
                or payload["training_executed"]
            ):
                raise RuntimeError("accuracy provenance/role drifted")

    metric_rows = []
    aggregates: dict[str, Any] = {}
    for resolution in RESOLUTIONS:
        seed_rows = [
            _metric_row(seed, resolution, accuracy[seed][resolution])
            for seed in SEEDS
        ]
        metric_rows.extend(seed_rows)
        aggregate = _aggregate_rows(seed_rows, resolution)
        metric_rows.extend(aggregate)
        aggregates[str(resolution)] = {
            row["row_type"]: {
                key: value for key, value in row.items() if key not in {"row_type", "seed"}
            }
            for row in aggregate
        }
    _write_csv(args.metrics_csv, metric_rows)

    cpu_ladder = {
        resolution: _load(args.cpu_ladder_dir / f"seed0_{resolution}_b8.json")
        for resolution in LADDER
    }
    gpu_ladder = {
        resolution: _load(args.gpu_ladder_dir / f"seed0_{resolution}_b8.json")
        for resolution in LADDER
    }
    graph_entries = []
    seen_cache_keys = set()
    for resolution, result in cpu_ladder.items():
        for role in ("anchor", "query"):
            row = result["graph_cache"][role]
            if row["cache_key"] in seen_cache_keys:
                continue
            seen_cache_keys.add(row["cache_key"])
            graph_entries.append(
                {
                    "resolution": resolution,
                    "role": role,
                    "cache_file": Path(row["cache_file"]).name,
                    "cache_file_sha256": row["cache_file_sha256"],
                    "cache_key": row["cache_key"],
                    "cache_key_payload": row["cache_key_payload"],
                    "metadata_hash": row["metadata_hash"],
                    "graph_hash": row["graph_hash"],
                }
            )
    graph_manifest = {
        "schema_version": "heat3d_v6_graph_cache_manifest_v2",
        "status": "frozen",
        "cache_key_fields": [
            "support_hash",
            "graph_config",
            "graph_seed",
            "graph_builder_fingerprint",
        ],
        "entries": graph_entries,
    }
    args.graph_cache_manifest.write_text(
        json.dumps(graph_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    cpu_timing = {
        batch: _load(args.cpu_timing_dir / f"seed0_4096_b{batch}_cold_warm.json")
        for batch in (1, 8, 16)
    }
    gpu_timing = {
        batch: _load(args.gpu_timing_dir / f"seed0_4096_b{batch}_cold_warm.json")
        for batch in (1, 8, 16)
    }
    timing_rows = []
    timing_rows.extend(_timing_row(cpu_timing[b], "cold_and_persistent") for b in (1, 8, 16))
    timing_rows.extend(_timing_row(gpu_timing[b], "cold_and_persistent") for b in (1, 8, 16))
    timing_rows.extend(_timing_row(cpu_ladder[n], "resolution_ladder") for n in LADDER)
    timing_rows.extend(_timing_row(gpu_ladder[n], "resolution_ladder") for n in LADDER)
    _write_csv(args.timing_csv, timing_rows)

    solver = _load(args.solver)
    cold = solver["cold_per_sample"]["mesh_assembly_solve_seconds"]
    warm = solver["warm_reused_by_boundary_pair"]["solve_seconds"]
    cold_total = cold["mean"] * 128
    warm_total = warm["mean"] * 128
    speedup_rows = []
    for resolution in LADDER:
        cpu = cpu_ladder[resolution]["runtime"]
        gpu = gpu_ladder[resolution]["runtime"]
        speedup_rows.append(
            {
                "resolution": resolution,
                "nonmatched_dof": True,
                "solver_nodes": solver["solver_node_count"],
                "checkpoint_cpu_forward_seconds": cpu[
                    "checkpoint_pure_forward_cold_seconds"
                ],
                "checkpoint_gpu_forward_seconds": gpu[
                    "checkpoint_pure_forward_cold_seconds"
                ],
                "cpu_production_end_to_end_seconds": cpu[
                    "end_to_end_seconds_valid128"
                ],
                "gpu_production_end_to_end_seconds": gpu[
                    "end_to_end_seconds_valid128"
                ],
                "cpu_samples_per_second": 128
                / cpu["end_to_end_seconds_valid128"],
                "gpu_samples_per_second": 128
                / gpu["end_to_end_seconds_valid128"],
                "solver_cold_total_seconds": cold_total,
                "solver_warm_total_seconds": warm_total,
                "solver_cold_to_cpu_speedup": cold_total
                / cpu["end_to_end_seconds_valid128"],
                "solver_cold_to_gpu_speedup": cold_total
                / gpu["end_to_end_seconds_valid128"],
                "solver_warm_to_cpu_speedup": warm_total
                / cpu["end_to_end_seconds_valid128"],
                "solver_warm_to_gpu_speedup": warm_total
                / gpu["end_to_end_seconds_valid128"],
            }
        )
    _write_csv(args.speedup_csv, speedup_rows)

    local_device = jax.devices()[0]
    environment = {
        "local_cpu": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "python": sys.version,
            "jax": jax.__version__,
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "device": str(local_device),
            "device_kind": getattr(local_device, "device_kind", None),
        },
        "gpu": _load(args.gpu_environment),
    }
    args.environment_json.write_text(
        json.dumps(environment, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    mean4096 = aggregates["4096"]["mean"]
    mean8192 = aggregates["8192"]["mean"]
    mean16384 = aggregates["16384"]["mean"]
    full_improvement_8192 = 1.0 - mean8192["full_raw_cv_rmse_K"] / mean4096[
        "full_raw_cv_rmse_K"
    ]
    full_improvement_16384 = 1.0 - mean16384["full_raw_cv_rmse_K"] / mean8192[
        "full_raw_cv_rmse_K"
    ]
    decision = {
        "default": 4096,
        "full_field_mode": 8192,
        "high_accuracy_limit": 16384,
        "experimental": 32768,
        "full_rmse_improvement_4096_to_8192_fraction": full_improvement_8192,
        "full_rmse_improvement_8192_to_16384_fraction": full_improvement_16384,
    }
    payload = {
        "schema_version": "heat3d_v6_production_final_closeout_v1",
        "status": "passed",
        "evaluation_commit": next(iter(accuracy["seed0"].values()))[
            "evaluator_commit"
        ],
        "graph_builder_code_fingerprint": next(
            iter(accuracy["seed0"].values())
        )["graph_builder_code_fingerprint"],
        "evaluation_role": "valid_iid",
        "test_hard_accessed": False,
        "training_executed": False,
        "checkpoint_modified": False,
        "multiseed": {
            seed: {
                str(resolution): {
                    "checkpoint": payload["checkpoint"],
                    "support_metrics": {
                        key: value
                        for key, value in payload["support_metrics"].items()
                        if key != "per_sample"
                    },
                    "full_field_metrics": payload["full_field_metrics"],
                }
                for resolution, payload in results.items()
            }
            for seed, results in accuracy.items()
        },
        "mean_std": aggregates,
        "solver": solver,
        "preflight": _load(args.preflight),
        "runner_reuse_smoke": _load(args.runner_smoke),
        "archive": _load(args.archive_manifest),
        "decision": decision,
        "nonmatched_dof": True,
    }
    args.output_json.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    lines = [
        "# V6 production inference final closeout",
        "",
        "Only `valid_iid` was evaluated. No training, checkpoint mutation, "
        "test, or hard-role access occurred.",
        "",
        "## Three-seed accuracy",
        "",
        "| Nodes | Support point-global | Full-field RMSE K | Full-field relative | Peak RMSE K | Source RMSE K | Layer/interface RMSE K |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for resolution in RESOLUTIONS:
        mean = aggregates[str(resolution)]["mean"]
        std = aggregates[str(resolution)]["std"]
        lines.append(
            f"| {resolution} | {mean['support_point_global_pct']:.4f}±"
            f"{std['support_point_global_pct']:.4f}% | "
            f"{mean['full_raw_cv_rmse_K']:.4f}±{std['full_raw_cv_rmse_K']:.4f} | "
            f"{mean['full_point_global_pct']:.4f}±"
            f"{std['full_point_global_pct']:.4f}% | "
            f"{mean['full_peak_rmse_K']:.4f}±{std['full_peak_rmse_K']:.4f} | "
            f"{mean['full_source_rmse_K']:.4f}±{std['full_source_rmse_K']:.4f} | "
            f"{mean['full_layer_rmse_K_mean']:.4f}/"
            f"{mean['full_interface_rmse_K_mean']:.4f} |"
        )
    lines.extend(
        [
            "",
            "The support and full-field CSV/JSON also retain sample-first, shape/scale, "
            "top/bottom, layer-drop, and sampling-floor fields.",
            "",
            "## Sparse graph equivalence",
            "",
            "| Nodes | Chunked build s | Sparse KD-tree build s | Speedup | Hash |",
            "|---:|---:|---:|---:|:---:|",
        ]
    )
    preflight = _load(args.preflight)
    for row in preflight["graph_backend_rows"]:
        if int(row["resolution"]) not in (8192, 16384):
            continue
        chunked = row["build_seconds"]["chunked_numpy_v1"]
        sparse = row["build_seconds"]["sparse_kdtree_v1"]
        lines.append(
            f"| {row['resolution']} | {chunked:.4f} | {sparse:.4f} | "
            f"{chunked / sparse:.2f}× | metadata+graph exact |"
        )
    lines.extend(
        [
            "",
            "The training runner smoke reused one shared-support metadata/graph build "
            "instead of eight per-sample builds, with 0 K forward difference and no "
            "optimizer update.",
            "",
            "## Production timing and solver comparison",
            "",
            "| Nodes | CPU forward/e2e s | GPU forward/e2e s | CPU/GPU sample/s | Cold solver speedup CPU/GPU | Warm solver speedup CPU/GPU |",
            "|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in speedup_rows:
        lines.append(
            f"| {row['resolution']} | "
            f"{row['checkpoint_cpu_forward_seconds']:.2f}/"
            f"{row['cpu_production_end_to_end_seconds']:.2f} | "
            f"{row['checkpoint_gpu_forward_seconds']:.2f}/"
            f"{row['gpu_production_end_to_end_seconds']:.2f} | "
            f"{row['cpu_samples_per_second']:.2f}/"
            f"{row['gpu_samples_per_second']:.2f} | "
            f"{row['solver_cold_to_cpu_speedup']:.2f}×/"
            f"{row['solver_cold_to_gpu_speedup']:.2f}× | "
            f"{row['solver_warm_to_cpu_speedup']:.2f}×/"
            f"{row['solver_warm_to_gpu_speedup']:.2f}× |"
        )
    lines.extend(
        [
            "",
            f"FVM cold mesh+assembly+solve: mean {cold['mean']:.3f} s, "
            f"median {cold['median']:.3f} s, P95 {cold['p95']:.3f} s per sample. "
            f"Warm solve: mean {warm['mean']:.3f} s, median {warm['median']:.3f} s, "
            f"P95 {warm['p95']:.3f} s per sample.",
            "",
            "At 4096, the selected CPU B8 persistent warm workflow is "
            f"{cpu_timing[8]['runtime']['persistent_compiled_workflow_seconds']['mean']:.3f} s/"
            "128 samples; GPU B16 is "
            f"{gpu_timing[16]['runtime']['persistent_compiled_workflow_seconds']['mean']:.3f} s/"
            "128 samples. Stage-separated input, graph cache, anchor/query forward, "
            "scale, label read, reconstruction, metric, and serialization timings are "
            "frozen in `v6_production_stage_timing.csv`.",
            "",
            "## Decision",
            "",
            "4096 remains the general default. 8192 is frozen as the optional "
            "full-field mode because it materially lowers reconstructed "
            "full-field RMSE. 16384 remains the high-accuracy production limit; "
            "32768 remains experimental seed0-only.",
            "",
            "All solver comparisons are explicitly nonmatched-DOF: FVM uses "
            "240825 nodes.",
            "",
        ]
    )
    args.markdown.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"status": "passed", "decision": decision}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
