#!/usr/bin/env python3
"""Close out the fail-closed P1i Anchor-derived High-N development run."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def metric_row(payload: dict[str, Any]) -> dict[str, Any]:
    support = payload["support_metrics"]
    full = payload["full_field_model_plus_reconstruction"]
    floor = payload["oracle_sampling_reconstruction_floor"]
    replay = payload["fixed_input_gpu_replay"]["prediction"]
    cross = payload["cross_backend_graph_diagnostic"]
    return {
        "resolution": int(payload["resolution"]),
        "status": payload["status"],
        "support_point_global_pct": support["point_global_true_rms_relative_rmse_pct"],
        "support_sample_first_pct": support["sample_first_cv_relative_rmse_pct"],
        "support_raw_cv_rmse_K": support["raw_cv_weighted_rmse_K"],
        "support_peak_rmse_K": support["peak_rmse_K"],
        "support_source_rmse_K": support["source_rmse_K"],
        "support_background_rmse_K": support["background_rmse_K"],
        "support_interface_drop_rmse_K": support["interface_drop_rmse_K"],
        "full_point_global_pct": full["point_global_true_rms_relative_rmse_pct"],
        "full_sample_first_pct": full["sample_first_cv_relative_rmse_pct"],
        "full_raw_cv_rmse_K": full["raw_cv_weighted_rmse_K"],
        "full_peak_rmse_K": full["peak_rmse_K"],
        "full_source_rmse_K": full["source_rmse_K"],
        "full_background_rmse_K": full["background_rmse_K"],
        "full_interface_drop_rmse_K": full["interface_drop_rmse_K"],
        "oracle_full_point_global_pct": floor["point_global_true_rms_relative_rmse_pct"],
        "oracle_full_raw_cv_rmse_K": floor["raw_cv_weighted_rmse_K"],
        "gpu_replay_rmse_K": replay["rmse_K"],
        "gpu_replay_max_abs_K": replay["max_abs_error_K"],
        "cross_backend_real_edge_topology_exact": cross["real_edge_topology_exact"],
        "end_to_end_seconds": payload["runtime"]["end_to_end_seconds"],
        "peak_rss_bytes": payload["runtime"]["process_peak_rss_bytes"],
        "peak_vram_bytes": payload["runtime"]["device_memory"].get("peak_bytes_in_use"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.results_root
    preflight_path = root / "actual_data_preflight.json"
    r1024_path = root / "resolution_1024.json"
    r4096_path = root / "resolution_4096.json"
    state_path = root / "execution_state.json"
    preflight, r1024, r4096, state = map(read, (preflight_path, r1024_path, r4096_path, state_path))
    if preflight["status"] != "passed" or r1024["status"] != "passed" or r4096["status"] != "failed":
        raise RuntimeError("unexpected preflight/1024/4096 lifecycle")
    failed = [key for key, value in r4096["implementation_hard_gates"].items()
              if value is False and key not in {"test_accessed", "sealed_accessed", "training_executed"}]
    if failed != ["cross_backend_real_edge_topology_exact"]:
        raise RuntimeError(f"unexpected 4096 hard-gate failures: {failed}")
    if any((root / f"resolution_{resolution}.json").exists() for resolution in (8192, 16384, 32768)):
        raise RuntimeError("higher resolution was executed after the 4096 failure")
    rows = [metric_row(r1024), metric_row(r4096)]
    csv_path = root / "anchor_high_n_valid32_failure_closeout.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    graph = r4096["graph_cache"]["samples"][0]
    cpu = r4096["cached_uncached_prediction_equivalence"]["deterministic_cpu_hard_gate"]
    payload = {
        "schema_version": "heat3d_v6_p1i_anchor_high_n_valid32_failure_closeout_v1",
        "status": "incomplete_fail_closed_at_4096_graph_gate",
        "decision": "stop_before_8192_16384",
        "failed_resolution": 4096,
        "failed_hard_gates": failed,
        "root_cause": {
            "classification": "cross_backend_float_normalization_changes_regional_radius_and_real_edge_topology_at_4096",
            "not_model_accuracy_failure": True,
            "not_sampler_or_checkpoint_failure": True,
            "cpu_same_backend_cache_prediction_exact": cpu["cached_uncached_prediction"]["max_abs_error_K"] == 0.0,
            "gpu_same_backend_cache_hash_exact": graph["cached_uncached_hash_exact"],
            "cross_backend_real_edge_topology_exact": False,
            "cpu_metadata_hash": r4096["cross_backend_graph_diagnostic"]["cpu_metadata_hash"],
            "gpu_metadata_hash": r4096["cross_backend_graph_diagnostic"]["gpu_metadata_hash"],
            "first_sample_gpu_edge_counts": graph["raw_edge_counts"],
            "focused_diagnosis": {
                "normalized_coordinate_max_abs_drift": 2.384185791015625e-7,
                "regional_radius_max_abs_drift": 0.130975887,
                "p2r_cpu_edge_count": 11487,
                "p2r_gpu_edge_count": 11520,
                "p2r_intersection": 11373,
                "r2r_cpu_edge_count": 17731,
                "r2r_gpu_edge_count": 17729,
                "r2r_intersection": 17721,
            },
        },
        "rows": rows,
        "artifacts": {
            "preflight": {"path": str(preflight_path), "sha256": sha256(preflight_path)},
            "resolution_1024": {"path": str(r1024_path), "sha256": sha256(r1024_path)},
            "resolution_4096": {"path": str(r4096_path), "sha256": sha256(r4096_path)},
            "execution_state": {"path": str(state_path), "sha256": sha256(state_path)},
            "summary_csv": {"path": str(csv_path), "sha256": sha256(csv_path)},
        },
        "role_contract": {
            "accessed_roles": ["valid_iid"], "test_accessed": False,
            "sealed_accessed": False, "training_executed": False,
            "checkpoint_modified": False, "binding_modified": False,
            "three_seed_valid128_executed": False, "resolution_32768_executed": False,
            "resolution_8192_executed": False, "resolution_16384_executed": False,
        },
        "accuracy_used_to_change_sampler_model_or_gate": False,
    }
    json_path = root / "anchor_high_n_valid32_failure_closeout.json"
    write(json_path, payload)
    md_path = root / "anchor_high_n_valid32_failure_closeout.md"
    lines = [
        "# P1i Anchor-derived High-N valid32 failure closeout", "",
        "N=1024 passed. N=4096 was stopped by the frozen cross-backend real-edge topology gate; N=8192/16384 were not started.", "",
        "| N | status | support PG % | support SF % | support raw K | full PG % | full raw K | oracle floor PG % | GPU replay RMSE K |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['resolution']} | {row['status']} | {row['support_point_global_pct']:.6f} | "
            f"{row['support_sample_first_pct']:.6f} | {row['support_raw_cv_rmse_K']:.6f} | "
            f"{row['full_point_global_pct']:.6f} | {row['full_raw_cv_rmse_K']:.6f} | "
            f"{row['oracle_full_point_global_pct']:.6f} | {row['gpu_replay_rmse_K']:.9f} |"
        )
    lines += [
        "", "## Root cause", "",
        "CPU and GPU agree on the 1024 real-edge topology. At 4096, platform float normalization changes regional radii enough to alter p2r/r2p/r2r edge sets. Same-backend cache checks pass, so this is neither cache corruption nor a model-accuracy failure.",
        "", "The frozen binding, sampler, checkpoint, graph parameters and metrics were not changed. Test/sealed remained closed.", "",
    ]
    md_path.write_text("\n".join(lines), encoding="utf-8")
    payload["artifacts"]["closeout_markdown"] = {"path": str(md_path), "sha256": sha256(md_path)}
    write(json_path, payload)
    print(json.dumps({"status": payload["status"], "failed_resolution": 4096}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
