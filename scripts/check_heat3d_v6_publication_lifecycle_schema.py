#!/usr/bin/env python3
"""Schema-only regression gate for all five publication routes and two modes.

No dataset, checkpoint, graph, GPU, or model code is loaded by this checker.
"""
from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import collect_heat3d_v6_publication_benchmark_v1_1 as collector  # noqa: E402
from heat3d_v6_publication_lifecycle_schema import (  # noqa: E402
    ROUTES, has_empty_statistic, provenance, q2_metrics, serial_metrics,
    timing_stats, validate_cell,
)


SCANNED = (
    "scripts/benchmark_heat3d_v6_p1i_final_e_service.py",
    "scripts/benchmark_heat3d_v6_p1i_u2_asymmetric_runtime.py",
    "scripts/smoke_heat3d_v6_publication_benchmark_conformance_v1_1.py",
    "scripts/collect_heat3d_v6_publication_benchmark_v1_1.py",
)


def fixture(route: str, mode: str) -> dict[str, Any]:
    ids = [f"fixture_{index:02d}" for index in range(32)]
    base: dict[str, Any] = {
        "schema_version": "heat3d_v6_publication_lifecycle_fixture_v1",
        "status": "passed", "route": route, "service_mode": mode,
        "sample_count": 32, "process_id": 1000 + ROUTES.index(route),
        "order_seed": 20260814, "ordered_sample_ids": ids,
        "aggregate_service_worker_peak_RAM_bytes": 1024,
        "measurement_provenance": provenance(
            attempted=True, matrix_completed=False, generated=False),
    }
    if route.startswith("E"):
        base["peak_vram_bytes"] = 2048
    elif route.startswith("U"):
        base["memory"] = {"peak_bytes_in_use": 2048}
    if mode == "serial":
        values = [1.0 + index / 1000.0 for index in range(32)]
        base["lifecycle_metrics"] = serial_metrics(
            cold_seconds=values[0], fresh_q1=timing_stats(values),
            cache_hot=timing_stats([0.5, 0.6]), resident=timing_stats([0.1, 0.2]),
        )
        if route.startswith("E"):
            base["serial_orders"] = [{"rows": [
                {"sample_id": sample_id, "elapsed_seconds": value}
                for sample_id, value in zip(ids, values, strict=True)
            ]}]
        elif route.startswith("U"):
            base["samples"] = [{
                "sample_id": sample_id,
                "stages": {"matched_continuous_e2e": value},
                "streaming": {"submit_to_result_seconds": value},
            } for sample_id, value in zip(ids, values, strict=True)]
        else:
            base["rows"] = [{
                "sample_id": sample_id, "submit_to_result_seconds": value,
            } for sample_id, value in zip(ids, values, strict=True)]
    else:
        base["lifecycle_metrics"] = q2_metrics(
            submit_to_result=timing_stats([1.0, 1.1]),
            inter_completion=timing_stats([0.5, 0.6]),
            throughput_samples_per_second=2.0,
            b16_to_b32_marginal_seconds=0.4,
        )
    return base


def literal_empty_stat_calls(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(), filename=str(path))
    findings: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
            continue
        if node.func.id not in {"stats", "dist", "summary", "timing_stats"}:
            continue
        if node.args and isinstance(node.args[0], (ast.List, ast.Tuple)) and not node.args[0].elts:
            findings.append(f"{path.relative_to(ROOT)}:{node.lineno}:{node.func.id}")
    return findings


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = []
    for route in ROUTES:
        for mode in ("serial", "Q2"):
            row = fixture(route, mode)
            validate_cell(row, formal=True)
            normalized = collector.normalize(row)
            json.dumps(row, allow_nan=False)
            if has_empty_statistic(row):
                raise RuntimeError(f"{route}/{mode}: empty statistic")
            rows.append({
                "route": route, "service_mode": mode, "status": "passed",
                "json_serializable": True, "required_and_NA_fields": True,
                "collector_parsed": bool(normalized), "no_empty_statistics": True,
                "Q2_B16_B32_required": mode == "Q2",
                "formal_status_and_provenance": True,
            })
    findings = [item for relative in SCANNED for item in literal_empty_stat_calls(ROOT / relative)]
    if findings:
        raise RuntimeError(f"literal empty statistic calls: {findings}")
    for bad in (
        lambda: timing_stats([]),
        lambda: q2_metrics(
            submit_to_result=timing_stats([1.0]), inter_completion=timing_stats([1.0]),
            throughput_samples_per_second=1.0, b16_to_b32_marginal_seconds=float("nan")),
    ):
        try:
            bad()
        except RuntimeError:
            pass
        else:
            raise RuntimeError("negative schema fixture did not fail closed")
    expanded = []
    for seed_index, seed in enumerate(collector.SEEDS):
        for route in ROUTES:
            for mode in ("serial", "Q2"):
                row = fixture(route, mode)
                row["order_seed"] = seed
                row["process_id"] = 10000 + seed_index * 10 + len(expanded) % 10
                expanded.append(row)
    matrix = {
        "status": "passed", "rows": expanded,
        "formal_measurement_attempted": True,
        "formal_matrix_completed": True,
        "publication_results_generated": False,
    }
    with tempfile.TemporaryDirectory(prefix="v6-lifecycle-schema-") as directory:
        raw_path = Path(directory) / "raw.json"
        collected_path = Path(directory) / "collected.json"
        raw_path.write_text(json.dumps(matrix, allow_nan=False) + "\n")
        subprocess.run(
            [sys.executable, str(ROOT / "scripts/collect_heat3d_v6_publication_benchmark_v1_1.py"),
             "--input", str(raw_path), "--output", str(collected_path)],
            check=True,
        )
        collected = json.loads(collected_path.read_text())
        if (collected.get("publication_timing_freeze") != "GO"
                or len(collected.get("route_seed_statistics", [])) != 30):
            raise RuntimeError("collector did not parse the expanded formal fixture matrix")
    result = {
        "schema_version": "heat3d_v6_publication_lifecycle_schema_regression_v1",
        "status": "passed", "benchmark_lifecycle_schema": "GO",
        "ready_for_authoritative_valid32": "GO", "fixture_count": len(rows),
        "fixtures": rows, "collector_parsed_all_10": True,
        "collector_full_30_cell_fixture_matrix_parsed": True,
        "literal_empty_stat_calls": findings,
        "negative_empty_and_B16_B32_gates_fail_closed": True,
        "gpu_smoke_executed": False, "model_inference_executed": False,
        "training": False, "test": False, "sealed": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps({"status": "passed", "fixtures": len(rows),
                      "benchmark_lifecycle_schema": "GO"}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
