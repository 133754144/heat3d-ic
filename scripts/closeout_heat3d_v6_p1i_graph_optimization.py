#!/usr/bin/env python3
"""Close the exact graph-construction optimization screen."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import numpy as np


def values_and_distribution(paths: list[Path]) -> tuple[np.ndarray, dict[str, float | int]]:
    values = sorted(json.loads(path.read_text())["timing"]["process_cold_continuous_seconds"] for path in paths)
    q = float(np.percentile(np.asarray(values), 95))
    return np.asarray(values), {"count": len(values), "median_seconds": statistics.median(values), "p95_seconds": q}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-result", type=Path, required=True)
    parser.add_argument("--shared-result", type=Path, required=True)
    parser.add_argument("--tiled-result", type=Path, required=True)
    parser.add_argument("--shared-equivalence", type=Path, required=True)
    parser.add_argument("--tiled-equivalence", type=Path, required=True)
    parser.add_argument("--process-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    reference = json.loads(args.reference_result.read_text())
    shared = json.loads(args.shared_result.read_text())
    tiled = json.loads(args.tiled_result.read_text())
    shared_eq = json.loads(args.shared_equivalence.read_text())
    tiled_eq = json.loads(args.tiled_equivalence.read_text())
    cold_ref_values, cold_ref = values_and_distribution(sorted((args.process_root / "E32768_reference").glob("run*.json")))
    cold_shared_values, cold_shared = values_and_distribution(sorted((args.process_root / "E32768_shared_reverse").glob("run*.json")))
    def med(payload: dict, key: str) -> float:
        return float(payload["timing"][key]["median_seconds"])
    shared_gain = {
        "graph_construction_pct": 100.0 * (1.0 - med(shared, "graph_construction") / med(reference, "graph_construction")),
        "fresh_topology_pct": 100.0 * (1.0 - med(shared, "new_case_e2e") / med(reference, "new_case_e2e")),
        "process_cold_pct": 100.0 * (1.0 - cold_shared["median_seconds"] / cold_ref["median_seconds"]),
        "warm_resident_pct": 100.0 * (1.0 - med(shared, "warm_cache_e2e") / med(reference, "warm_cache_e2e")),
        "peak_vram_pct": 100.0 * (1.0 - float(shared["device_memory"]["peak_bytes_in_use"]) / float(reference["device_memory"]["peak_bytes_in_use"])),
    }
    tiled_gain = {
        "graph_construction_pct": 100.0 * (1.0 - med(tiled, "graph_construction") / med(reference, "graph_construction")),
        "fresh_topology_pct": 100.0 * (1.0 - med(tiled, "new_case_e2e") / med(reference, "new_case_e2e")),
    }
    rng = np.random.default_rng(20260810)
    bootstrap = np.asarray([
        np.median(rng.choice(cold_ref_values, len(cold_ref_values), replace=True))
        - np.median(rng.choice(cold_shared_values, len(cold_shared_values), replace=True))
        for _ in range(10000)
    ])
    process_gain_ci = {
        "seed": 20260810, "replicates": 10000,
        "median_difference_reference_minus_shared_seconds": float(np.median(bootstrap)),
        "ci95_seconds": [float(value) for value in np.percentile(bootstrap, [2.5, 97.5])],
        "probability_positive": float(np.mean(bootstrap > 0.0)),
    }
    clear_process_gain = process_gain_ci["ci95_seconds"][0] > 0.0
    payload = {
        "schema_version": "heat3d_v6_p1i_graph_optimization_closeout_v1",
        "status": "completed_no_promotion",
        "screen_cell": "E@32768 seed0 frozen valid32",
        "reference": {
            "graph_median_s": med(reference, "graph_construction"),
            "fresh_median_s": med(reference, "new_case_e2e"),
            "warm_median_s": med(reference, "warm_cache_e2e"),
            "process_cold": cold_ref,
            "peak_vram_bytes": reference["device_memory"]["peak_bytes_in_use"],
        },
        "shared_reverse": {
            "equivalence": shared_eq,
            "graph_median_s": med(shared, "graph_construction"),
            "fresh_median_s": med(shared, "new_case_e2e"),
            "warm_median_s": med(shared, "warm_cache_e2e"),
            "process_cold": cold_shared,
            "peak_vram_bytes": shared["device_memory"]["peak_bytes_in_use"],
            "gain_pct": shared_gain,
            "process_cold_unpaired_bootstrap": process_gain_ci,
            "decision": "no_promotion_process_cold_gain_not_clear_ci_includes_zero",
        },
        "gpu_tiled": {
            "equivalence": tiled_eq,
            "graph_median_s": med(tiled, "graph_construction"),
            "fresh_median_s": med(tiled, "new_case_e2e"),
            "gain_pct": tiled_gain,
            "decision": "no_go_slower_despite_edge_exactness",
        },
        "padding_bucketing": {"status": "not_run", "reason": "no preceding optimization met promotion gate"},
        "promoted_cells": [],
        "B_optimization_applied": False,
        "decision": (
            "NO-GO for production graph replacement: shared reverse is exact and modestly improves fresh topology, "
            "but its independent process-cold median gain is below the preregistered 3% clear-gain threshold; "
            "GPU tiled is edge-exact but slower. Original B/E remain frozen production comparisons."
        ),
        "role_contract": {"training": False, "test": False, "sealed": False, "valid32_only": True},
    }
    assert shared_eq["status"] == "passed" and tiled_eq["status"] == "passed"
    assert not clear_process_gain
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": payload["status"], "shared_fresh_gain_pct": shared_gain["fresh_topology_pct"], "shared_process_gain_pct": shared_gain["process_cold_pct"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
