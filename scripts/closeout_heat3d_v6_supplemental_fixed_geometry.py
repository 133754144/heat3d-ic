#!/usr/bin/env python3
"""Collect the preregistered V6 fixed-geometry supplemental runtime results."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/heat3d_v6_supplemental"
PROTOCOL_PATH = CONFIG / "v6_fixed_geometry_runtime_protocol.json"
RAW_ROOT = CONFIG / "fixed_geometry_runtime_raw"
JSON_PATH = CONFIG / "v6_fixed_geometry_runtime_closeout.json"
CSV_PATH = CONFIG / "v6_fixed_geometry_runtime_summary.csv"
STAGE_CSV_PATH = CONFIG / "v6_fixed_geometry_runtime_stage_decomposition.csv"
MD_PATH = ROOT / "docs/v6_supplemental_fixed_geometry_runtime_closeout.md"
HASH_PATH = CONFIG / "v6_fixed_geometry_runtime_sha256.txt"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def finite_tree(value: Any) -> bool:
    if isinstance(value, dict):
        return all(finite_tree(item) for item in value.values())
    if isinstance(value, list):
        return all(finite_tree(item) for item in value)
    if isinstance(value, float):
        return bool(np.isfinite(value))
    return True


def main() -> int:
    protocol = load_json(PROTOCOL_PATH)
    routes = list(protocol["routes"])
    raw_paths = {route: RAW_ROOT / f"{route}.json" for route in routes}
    missing = [str(path) for path in raw_paths.values() if not path.is_file()]
    if missing:
        raise SystemExit(f"missing raw artifacts: {missing}")

    csv_rows: list[dict[str, Any]] = []
    stage_rows: list[dict[str, Any]] = []
    route_records: list[dict[str, Any]] = []
    for route, path in raw_paths.items():
        payload = load_json(path)
        if payload.get("status") != "passed":
            raise SystemExit(f"{route}: status is not passed")
        if payload.get("route") != route or not finite_tree(payload):
            raise SystemExit(f"{route}: route or finite-value gate failed")
        if payload.get("temperature_files_opened") != 0:
            raise SystemExit(f"{route}: temperature labels were opened")
        if payload.get("accessed_roles") != [
            "train_inputs", "shared_full_mesh_without_temperature"
        ]:
            raise SystemExit(f"{route}: role boundary drift")
        if not payload.get("checkpoint_unchanged"):
            raise SystemExit(f"{route}: checkpoint mutation detected")
        if payload["correctness"].get("status") != "passed":
            raise SystemExit(f"{route}: correctness gate failed")
        if payload["correctness"].get("case_count") != 32:
            raise SystemExit(f"{route}: correctness population drift")
        for row in payload["correctness"]["rows"]:
            if not row.get("passed"):
                raise SystemExit(f"{route}/{row['case_id']}: numerical gate failed")
            for identity in row["static_identities"].values():
                if not all(identity.values()):
                    raise SystemExit(f"{route}/{row['case_id']}: static identity failed")
        if len(payload.get("timing_rows", [])) != 288:
            raise SystemExit(f"{route}: expected 288 timing rows")
        guards = payload.get("guardrails", {})
        if any(guards.get(key) for key in (
            "training", "temperature_labels_read", "test_iid_accessed",
            "sealed_iid_accessed", "FVM_run",
        )):
            raise SystemExit(f"{route}: forbidden action recorded")

        setup_median = float(np.median([
            float(row["setup_seconds"]) for row in payload["static_setup"]
        ]))
        amortization = {row["sweep"]: row for row in payload["amortization"]}
        summary = {(row["sweep"], row["mode"]): row for row in payload["summary"]}
        historical = payload.get("historical_devbox_comparison") or {}
        for sweep in ("K_only", "K_plus_Q_scale"):
            fresh = float(summary[(sweep, "fresh_new_case")]["timing"]["median_s"])
            for mode in ("fresh_new_case", "graph_only_reuse", "full_static_reuse"):
                row = summary[(sweep, mode)]
                stages = row["stage_summary"]
                bottleneck = max(
                    stages,
                    key=lambda name: float(stages[name]["median_s"] or 0.0),
                )
                timing = row["timing"]
                amortized = amortization[sweep]
                csv_rows.append({
                    "sweep": sweep,
                    "route": route,
                    "output_resolution": int(payload["resolution"]),
                    "mode": mode,
                    "measurement_count": int(row["sample_measurement_count"]),
                    "median_s": float(timing["median_s"]),
                    "p95_s": float(timing["p95_s"]),
                    "throughput_samples_per_second": float(
                        timing["throughput_samples_per_second"]
                    ),
                    "speedup_vs_fresh_median": float(fresh / timing["median_s"]),
                    "stage_bottleneck": bottleneck,
                    "stage_bottleneck_median_s": float(stages[bottleneck]["median_s"]),
                    "static_setup_median_s_per_geometry_route": setup_median,
                    "break_even_repeated_cases": amortized["break_even_repeated_cases"],
                    "amortized_latency_4_cases_s": amortized["amortized_latency_s"]["4"],
                    "amortized_latency_16_cases_s": amortized["amortized_latency_s"]["16"],
                    "amortized_latency_100_cases_s": amortized["amortized_latency_s"]["100"],
                    "historical_devbox_fresh_median_s": historical.get("fresh_median_s"),
                    "current_fresh_over_historical_ratio": (
                        fresh / float(historical["fresh_median_s"])
                        if historical.get("fresh_median_s") else None
                    ),
                    "execution_commit": payload["execution_commit"],
                    "raw_sha256": sha256(path),
                })
                elapsed_median = float(timing["median_s"])
                for stage_name, stage in stages.items():
                    stage_rows.append({
                        "sweep": sweep,
                        "route": route,
                        "mode": mode,
                        "stage": stage_name,
                        "median_s": float(stage["median_s"]),
                        "p95_s": float(stage["p95_s"]),
                        "share_of_elapsed_median": (
                            float(stage["median_s"]) / elapsed_median
                        ),
                        "measurement_count": int(stage["count"]),
                        "execution_commit": payload["execution_commit"],
                    })
        route_records.append({
            "route": route,
            "resolution": payload["resolution"],
            "execution_commit": payload["execution_commit"],
            "raw_path": str(path.relative_to(ROOT)),
            "raw_sha256": sha256(path),
            "correctness_max_abs_K": payload["correctness"][
                "maximum_cached_vs_standard_max_abs_K"
            ],
            "setup_median_s_per_geometry_route": setup_median,
            "checkpoint_unchanged": payload["checkpoint_unchanged"],
            "memory": payload["memory"],
            "historical_devbox_comparison": historical,
            "input_audit": payload["input_audit"],
        })

    CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    with CSV_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)
    with STAGE_CSV_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(stage_rows[0]))
        writer.writeheader()
        writer.writerows(stage_rows)

    closeout = {
        "schema_version": "heat3d_v6_p1i_fixed_geometry_runtime_closeout_v1",
        "status": "completed_passed",
        "protocol_path": str(PROTOCOL_PATH.relative_to(ROOT)),
        "protocol_sha256": sha256(PROTOCOL_PATH),
        "base_main_commit": protocol["base_main_commit"],
        "host": "devbox",
        "wsl2_status": "offline_not_used",
        "temperature_labels_opened": 0,
        "roles_accessed": ["train_inputs", "shared_full_mesh_without_temperature"],
        "representative_geometry_selection": protocol["representative_geometry_selection"],
        "sweeps": protocol["sweeps"],
        "routes": route_records,
        "summary_csv": str(CSV_PATH.relative_to(ROOT)),
        "stage_decomposition_csv": str(STAGE_CSV_PATH.relative_to(ROOT)),
        "summary_rows": csv_rows,
        "stage_decomposition_rows": stage_rows,
        "guardrails": {
            "training": False,
            "test_iid_accessed": False,
            "sealed_iid_accessed": False,
            "FVM_run": False,
            "frozen_scientific_results_modified": False,
        },
        "interpretation": {
            "historical_comparison": (
                "same devbox hardware-state continuity only; historical valid32 geometry "
                "population and this train-only fixed-geometry sweep are not pooled"
            ),
            "FVM_claim": "not generated in this supplemental experiment",
        },
    }
    JSON_PATH.write_text(
        json.dumps(closeout, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    table = []
    for row in csv_rows:
        table.append(
            f"| {row['sweep']} | {row['route']} | {row['mode']} | "
            f"{row['median_s']:.4f} | {row['p95_s']:.4f} | "
            f"{row['throughput_samples_per_second']:.2f} | "
            f"{row['speedup_vs_fresh_median']:.2f}x | {row['stage_bottleneck']} |"
        )
    geometry_lines = [
        f"- `{row['sample_id']}`: source count {row['source_count']}, "
        f"k regions {row['k_region_count']}"
        for row in protocol["representative_geometry_selection"]["samples"]
    ]
    setup_lines = []
    historical_lines = []
    for record in route_records:
        route_rows = [row for row in csv_rows if row["route"] == record["route"]]
        breaks = sorted({row["break_even_repeated_cases"] for row in route_rows})
        setup_lines.append(
            f"- `{record['route']}`: setup median "
            f"{record['setup_median_s_per_geometry_route']:.3f} s; "
            f"break-even repeated cases {breaks}."
        )
        current_fresh = next(
            row for row in route_rows
            if row["sweep"] == "K_only" and row["mode"] == "fresh_new_case"
        )
        historical_lines.append(
            f"- `{record['route']}`: current {current_fresh['median_s']:.3f} s "
            f"vs historical {current_fresh['historical_devbox_fresh_median_s']:.3f} s "
            f"({current_fresh['current_fresh_over_historical_ratio']:.3f}x)."
        )
    MD_PATH.write_text(
        "# V6/P1i fixed-geometry supplemental runtime closeout\n\n"
        "Status: **COMPLETED / PASS**. This is a train-input-only runtime "
        "supplement; it does not alter frozen V6/P1i scientific results and "
        "does not create an FVM speedup claim.\n\n"
        "## Frozen design\n\n"
        + "\n".join(geometry_lines)
        + "\n\n`K_only` keeps q byte-exact and changes k at preregistered formal-distribution "
        "quantiles. `K_plus_Q_scale` uses the same k sweep and positive "
        "alpha={0.8,0.95,1.05,1.2}; q mask and normalized spatial distribution "
        "are invariant. All inputs remain inside the formal P1i contract.\n\n"
        "## Correctness\n\n"
        "All 4 routes passed 32/32 cases. Cached and standard paths have exact "
        "support/CV, graph, reconstruction-map and prepared-payload hashes. "
        "Checkpoint parameters are unchanged; no temperature, valid, test or "
        "sealed labels were opened.\n\n"
        "## Runtime results (seconds per case)\n\n"
        "| Sweep | Route | Reuse mode | Median | P95 | Samples/s | Speedup vs fresh | Median bottleneck |\n"
        "|---|---|---|---:|---:|---:|---:|---|\n"
        + "\n".join(table)
        + "\n\n## Static setup and amortization\n\n"
        + "\n".join(setup_lines)
        + "\n\nThe CSV includes amortized latency at 4, 16, and 100 repeated cases. "
        "\n\n## Prior devbox continuity check\n\n"
        + "\n".join(historical_lines)
        + "\n\nFresh times are compared with the prior devbox lifecycle table; "
        "the comparison is hardware/runtime continuity only because the sample "
        "populations differ.\n",
        encoding="utf-8",
    )

    manifest_paths = [
        PROTOCOL_PATH, *raw_paths.values(), JSON_PATH, CSV_PATH, STAGE_CSV_PATH, MD_PATH,
    ]
    HASH_PATH.write_text(
        "".join(f"{sha256(path)}  {path.relative_to(ROOT)}\n" for path in manifest_paths),
        encoding="utf-8",
    )
    print(json.dumps({
        "status": "passed", "routes": len(routes), "csv_rows": len(csv_rows),
        "json": str(JSON_PATH.relative_to(ROOT)), "csv": str(CSV_PATH.relative_to(ROOT)),
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
