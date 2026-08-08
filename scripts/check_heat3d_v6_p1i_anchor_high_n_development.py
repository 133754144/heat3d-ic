#!/usr/bin/env python3
"""Check the frozen P1i Anchor-derived valid32 High-N development bundle."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
MANDATORY = (1024, 4096, 8192, 16384)
HIGH_N = (4096, 8192, 16384)


class CheckError(RuntimeError):
    pass


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise CheckError(message)


def finite_tree(value: Any) -> bool:
    if isinstance(value, dict):
        return all(finite_tree(item) for item in value.values())
    if isinstance(value, list):
        return all(finite_tree(item) for item in value)
    if isinstance(value, float):
        return math.isfinite(value)
    return True


def check_static(binding_path: Path) -> dict[str, Any]:
    binding = json.loads(binding_path.read_text(encoding="utf-8"))
    require(binding["status"] == "frozen_after_three_seed_r0_pass", "binding status drift")
    require(tuple(binding["resolutions"]["mandatory"]) == MANDATORY, "mandatory ladder drift")
    require(binding["resolutions"]["optional_valid_only"] == 32768, "optional ladder drift")
    require(binding["development_subset"]["count"] == 32, "valid32 count drift")
    require(len(binding["development_subset"]["sample_ids"]) == 32, "valid32 IDs drift")
    for row in binding["code_fingerprints"].values():
        require(sha256(ROOT / row["path"]) == row["sha256"], f"code fingerprint drift: {row['path']}")
    return binding


def check_results(root: Path, binding: dict[str, Any]) -> dict[str, Any]:
    preflight_path = root / "actual_data_preflight.json"
    preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
    require(preflight["status"] == "passed", "preflight failed")
    require(all(value is True for value in preflight["checks"].values()), "preflight check failed")
    ids = binding["development_subset"]["sample_ids"]
    require(preflight["sample_ids"] == ids, "preflight valid32 order drift")
    require(preflight["role_contract"]["test_accessed"] is False, "test accessed")
    require(preflight["role_contract"]["sealed_accessed"] is False, "sealed accessed")
    require(preflight["role_contract"]["training_executed"] is False, "training executed")

    previous: dict[str, np.ndarray] = {}
    for resolution in HIGH_N:
        rows = preflight["supports"][str(resolution)]
        require([row["sample_id"] for row in rows] == ids, f"N={resolution} support IDs drift")
        for row in rows:
            path = Path(row["support_file"])
            require(path.is_file() and sha256(path) == row["support_file_sha256"], "support file hash drift")
            with np.load(path, allow_pickle=False) as payload:
                indices = np.asarray(payload["selected_indices"], dtype=np.int32)
                cv = np.asarray(payload["operator_control_volume"], dtype=np.float64)
                q = np.asarray(payload["q_W_m3"], dtype=np.float64)
            require(indices.shape == (resolution,), "support shape drift")
            require(len(np.unique(indices)) == resolution, "support duplicates")
            require(np.all(np.isfinite(cv)) and np.all(cv > 0.0), "invalid operator CV")
            require(np.all(np.isfinite(q)) and np.all(q >= 0.0), "invalid q")
            if row["sample_id"] in previous:
                require(np.array_equal(indices[:len(previous[row["sample_id"]])], previous[row["sample_id"]]),
                        "nested support prefix drift")
            previous[row["sample_id"]] = indices

    result_rows = []
    for resolution in MANDATORY:
        path = root / f"resolution_{resolution}.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        require(payload["status"] == "passed", f"N={resolution} failed")
        require(payload["sample_ids"] == ids, f"N={resolution} sample order drift")
        gates = payload["implementation_hard_gates"]
        negative = {"test_accessed", "sealed_accessed", "training_executed"}
        require(all(value is True for key, value in gates.items() if key not in negative),
                f"N={resolution} positive hard gate failed")
        require(all(gates[key] is False for key in negative), f"N={resolution} forbidden role accessed")
        require(payload["accuracy_is_not_an_implementation_gate"] is True, "accuracy gate drift")
        require(finite_tree(payload["support_metrics"]), "non-finite support metric")
        require(finite_tree(payload["full_field_model_plus_reconstruction"]), "non-finite full metric")
        require(finite_tree(payload["oracle_sampling_reconstruction_floor"]), "non-finite floor metric")
        replay = payload["fixed_input_gpu_replay"]["prediction"]
        require(math.isfinite(replay["max_abs_error_K"]) and math.isfinite(replay["rmse_K"]),
                "non-finite GPU replay diagnostic")
        cache_equivalence = payload["cached_uncached_prediction_equivalence"]
        require(cache_equivalence["deterministic_cpu_hard_gate"]["status"] == "passed",
                "deterministic CPU cache gate failed")
        require(cache_equivalence["deterministic_cpu_hard_gate"]["cached_uncached_prediction"]["max_abs_error_K"] <= 1.0e-6,
                "deterministic CPU cache prediction drift")
        require(cache_equivalence["gpu_reduction_nondeterminism_is_not_graph_cache_failure"] is True,
                "GPU diagnostic classification drift")
        cross_backend = payload["cross_backend_graph_diagnostic"]
        require(cross_backend["real_edge_topology_exact"] is True,
                "cross-backend real-edge topology drift")
        require(cross_backend["known_float_normalization_drift_not_edge_topology_drift"] is True,
                "cross-backend float-drift classification missing")
        require(cross_backend["cache_directories_are_backend_isolated_key_payload_is_unchanged"] is True,
                "backend cache isolation contract drift")
        graph_rows = payload["graph_cache"]["samples"]
        require(len(graph_rows) == 32, "graph cache sample count drift")
        require(all(row["cached_uncached_hash_exact"] for row in graph_rows), "graph cache equivalence failed")
        for row in graph_rows:
            path_graph = Path(row["cache_file"])
            require(path_graph.is_file() and sha256(path_graph) == row["cache_file_sha256"], "graph cache hash drift")
        reconstruction = payload["reconstruction_cache"]["samples"]
        require(len(reconstruction) == 32, "reconstruction cache sample count drift")
        for row in reconstruction:
            path_map = Path(row["cache_file"])
            require(path_map.is_file() and sha256(path_map) == row["cache_file_sha256"], "map cache hash drift")
        prediction = payload["prediction_artifact"]
        prediction_path = Path(prediction["path"])
        require(prediction_path.is_file() and sha256(prediction_path) == prediction["sha256"],
                "prediction artifact hash drift")
        result_rows.append(payload)

    state = json.loads((root / "execution_state.json").read_text(encoding="utf-8"))
    require(state["status"] == "passed", "execution state failed")
    require([row["resolution"] for row in state["execution"]] == list(MANDATORY), "execution order drift")
    require(all(row["returncode"] == 0 for row in state["execution"]), "worker return code failed")
    require(all(row["cpu_cache_audit_returncode"] == 0 for row in state["execution"]),
            "CPU cache audit return code failed")
    require(state["higher_resolutions_started_after_4096_failure"] is False, "fail-fast contract drift")
    require(not (root / "resolution_32768.json").exists(), "forbidden 32768 result exists")

    closeout_path = root / "anchor_high_n_valid32_closeout.json"
    closeout = json.loads(closeout_path.read_text(encoding="utf-8"))
    require(closeout["status"] == "passed", "closeout failed")
    require([row["resolution"] for row in closeout["rows"]] == list(MANDATORY), "closeout ladder drift")
    require(closeout["accuracy_used_as_gate"] is False, "accuracy used as gate")
    require(closeout["role_contract"]["resolution_32768_executed"] is False, "32768 executed")
    csv_path = root / "anchor_high_n_valid32_summary.csv"
    with csv_path.open(newline="", encoding="utf-8") as handle:
        csv_rows = list(csv.DictReader(handle))
    require(len(csv_rows) == 4, "summary CSV row count drift")
    require((root / "anchor_high_n_valid32_closeout.md").is_file(), "closeout Markdown missing")
    return {"preflight": preflight, "results": result_rows, "closeout": closeout}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binding", type=Path, default=ROOT / "configs/heat3d_v6_p1i/v6_p1i_high_n_implementation_binding.json")
    parser.add_argument("--results-root", type=Path)
    args = parser.parse_args()
    binding = check_static(args.binding)
    if args.results_root is not None:
        check_results(args.results_root, binding)
    print(json.dumps({
        "status": "passed", "static_binding": True,
        "results_checked": args.results_root is not None,
        "mandatory_resolutions": list(MANDATORY),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (CheckError, KeyError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1)
