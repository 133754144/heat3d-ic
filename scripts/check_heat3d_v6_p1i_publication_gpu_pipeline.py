#!/usr/bin/env python3
"""Static and result checker for the P1i publication GPU pipeline."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
EXPECTED = [4096, 8192, 16384, 32768, 65536]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def check_static() -> None:
    protocol_path = ROOT / "configs/heat3d_v6_p1i/v6_p1i_publication_gpu_pipeline_protocol.json"
    protocol = json.loads(protocol_path.read_text())
    frozen = protocol["frozen_inputs"]
    require(sha256(ROOT / frozen["high_n_binding_path"]) == frozen["high_n_binding_sha256"], "binding SHA drift")
    require(sha256(ROOT / frozen["gpu_only_amendment_path"]) == frozen["gpu_only_amendment_sha256"], "amendment SHA drift")
    require(protocol["role_contract"] == {
        "training": False,
        "checkpoint_modified": False,
        "dataset_or_manifest_modified": False,
        "test_accessed": False,
        "sealed_accessed": False,
        "three_seed_valid128": False,
        "full_gpu_graph_builder": False,
        "batch_inference": False,
    }, "role contract drift")
    benchmark = (ROOT / "scripts/benchmark_heat3d_v6_p1i_publication_gpu_pipeline.py").read_text()
    require("_load_metadata_no_audit" in benchmark and "_load_mapping_no_audit" in benchmark, "no-audit cache loaders missing")
    require("build_reconstruction_map" not in benchmark, "benchmark may not build reconstruction maps")
    require("_graph_cache_one" not in benchmark, "benchmark may not rebuild/audit graphs")
    gpu_module = (ROOT / "rigno/heat3d_v6_gpu_reconstruction.py").read_text()
    require("build_reconstruction_map" not in gpu_module and "cKDTree" not in gpu_module, "GPU apply must not build maps")


def check_timing(path: Path) -> None:
    payload = json.loads(path.read_text())
    require(payload["status"] == "passed", "timing status")
    require([row["resolution"] for row in payload["results"]] == EXPECTED, "timing resolutions")
    require(payload["role_contract"]["test"] is False and payload["role_contract"]["sealed"] is False, "timing role leakage")
    for row in payload["results"]:
        require(row["status"] == "passed", "resolution timing failed")
        require(row["timing"]["new_case"]["count"] == 32, "new-case count")
        require(row["timing"]["warm_cache"]["count"] >= 30, "warm count")
        require(row["timing"]["neural_forward"]["count"] >= 30, "forward count")
        require(row["gpu_reconstruction_equivalence"]["status"] == "passed", "reconstruction gate")
        require(row["gpu_reconstruction_equivalence"]["maximum_sample_max_abs_error_K"] <= 1e-4, "reconstruction max")
        require(row["gpu_reconstruction_equivalence"]["maximum_sample_rmse_K"] <= 1e-5, "reconstruction rmse")


def check_graph(path: Path, csv_path: Path) -> None:
    payload = json.loads(path.read_text())
    require(payload["status"] == "passed_offline_cache_only", "graph audit status")
    require(payload["role_contract"] == {"training": False, "inference": False, "test": False, "sealed": False}, "graph roles")
    require(len(payload["summaries"]) == 11, "graph row count")
    for row in payload["summaries"]:
        require(row["coverage"]["p2r_zero_degree_nodes"]["max"] == 0, "p2r uncovered")
        require(row["coverage"]["r2p_zero_degree_nodes"]["max"] == 0, "r2p uncovered")
        require(np.isfinite(row["observed_physical_support_radius_m"]["median"]), "radius nonfinite")
    with csv_path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    require(len(rows) == 11, "graph CSV row count")


def check_closeout(path: Path, csv_path: Path) -> None:
    payload = json.loads(path.read_text())
    require(payload["status"] == "passed_p0_p1_p2", "closeout status")
    require([row["resolution"] for row in payload["curve"]] == EXPECTED, "closeout resolutions")
    require(payload["decision"]["priority"] == "graph reuse/fixed regional mesh", "decision drift")
    provenance = payload["execution_provenance"]
    require(len(provenance["graph_execution_commit"]) == 40, "graph execution commit")
    require(len(provenance["timing_execution_commit"]) == 40, "timing execution commit")
    require(provenance["actual_new_compute"]["fresh_graph_builds"] == 0, "fresh graph build recorded")
    require(provenance["actual_new_compute"]["reconstruction_map_builds"] == 0, "map build recorded")
    require(provenance["actual_new_compute"]["metric_or_label_evaluations"] == 0, "label/metric evaluation recorded")
    for row in payload["curve"]:
        require(row["reconstruction_max_abs_error_K"] <= 1e-4, "closeout recon max")
        require(row["reconstruction_max_rmse_K"] <= 1e-5, "closeout recon rmse")
        require(row["new_case_median_seconds"] > row["warm_cache_median_seconds"], "timing ordering")
    with csv_path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    require(len(rows) == len(EXPECTED), "closeout CSV row count")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--timing-json", type=Path)
    parser.add_argument("--graph-json", type=Path)
    parser.add_argument("--graph-csv", type=Path)
    parser.add_argument("--closeout-json", type=Path)
    parser.add_argument("--closeout-csv", type=Path)
    args = parser.parse_args()
    check_static()
    if args.timing_json:
        check_timing(args.timing_json)
    if args.graph_json:
        require(args.graph_csv is not None, "--graph-csv required")
        check_graph(args.graph_json, args.graph_csv)
    if args.closeout_json:
        require(args.closeout_csv is not None, "--closeout-csv required")
        check_closeout(args.closeout_json, args.closeout_csv)
    print(json.dumps({
        "status": "passed", "static": True,
        "timing": bool(args.timing_json), "graph": bool(args.graph_json),
        "closeout": bool(args.closeout_json),
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
