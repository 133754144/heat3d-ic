#!/usr/bin/env python3
"""Collect the frozen supplemental S0-S3 artifacts without rerunning inference."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULT_ROOT = ROOT / "configs/heat3d_v6_supplemental_publication/known_topology_results_8a81261"
PROTOCOL = ROOT / "configs/heat3d_v6_supplemental_publication/known_topology_new_physics_protocol.json"
JSON_OUT = ROOT / "configs/heat3d_v6_supplemental_publication/known_topology_publication_closeout.json"
CSV_OUT = ROOT / "configs/heat3d_v6_supplemental_publication/known_topology_publication_results.csv"
MD_OUT = ROOT / "docs/v6_supplemental_known_topology_publication_closeout.md"
ROUTES = (
    "E16384_reconstruction",
    "U_v2_16384_reconstruction",
    "U_v2_direct240825",
    "E240825_direct_control",
)
EXECUTION_COMMIT = "8a812619ab0112b4ecfc37ef18189f731180059d"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    artifacts: dict[str, dict[str, str]] = {}
    gates: dict[str, dict[str, object]] = {}
    table: list[dict[str, object]] = []
    for phase, expected_rows in (("S1", 4), ("S2", 8), ("S3", 32)):
        phase_rows: dict[str, object] = {}
        artifacts[phase] = {}
        for route in ROUTES:
            path = RESULT_ROOT / phase / f"{route}.json"
            payload = json.loads(path.read_text(encoding="utf-8"))
            passed = (
                payload["status"] == "passed"
                and payload["phase"] == phase
                and payload["route"] == route
                and payload["execution_commit"] == EXECUTION_COMMIT
                and len(payload["gate_rows"]) == expected_rows
                and all(row["passed"] for row in payload["gate_rows"])
                and not any(payload["guardrails"].values())
            )
            if not passed:
                raise RuntimeError(f"{phase}/{route}: frozen gate artifact failed")
            artifacts[phase][route] = sha256(path)
            phase_rows[route] = {
                "status": "PASS",
                "case_count": expected_rows,
                "max_prediction_drift_K": max(
                    float(row["prediction_max_abs_K"]) for row in payload["gate_rows"]),
            }
            if phase == "S3":
                if len(payload["timing_rows"]) != 64 or len(payload["summary"]) != 4:
                    raise RuntimeError(f"S3/{route}: timing population drift")
                known_by_sweep = {
                    row["sweep"]: row for row in payload["summary"]
                    if row["mode"] == "known_topology_new_physics"
                }
                for row in payload["summary"]:
                    timing = row["timing"]
                    table.append({
                        "route": route,
                        "sweep": row["sweep"],
                        "mode": row["mode"],
                        "count": int(timing["count"]),
                        "median_seconds": float(timing["median_s"]),
                        "p95_seconds": float(timing["p95_s"]),
                        "throughput_samples_per_second": float(
                            timing["throughput_samples_per_second"]),
                        "speedup_vs_fresh_median": (
                            float(known_by_sweep[row["sweep"]]["speedup_vs_fresh_median"])
                            if row["mode"] == "known_topology_new_physics" else None
                        ),
                        "stage_median_seconds": row["stage_median_seconds"],
                    })
        gates[phase] = phase_rows

    CSV_OUT.parent.mkdir(parents=True, exist_ok=True)
    fields = (
        "route", "sweep", "mode", "count", "median_seconds", "p95_seconds",
        "throughput_samples_per_second", "speedup_vs_fresh_median",
    )
    with CSV_OUT.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in table:
            writer.writerow({key: row[key] for key in fields})

    closeout = {
        "schema_version": "heat3d_v6_supplemental_known_topology_closeout_v1",
        "status": "passed",
        "gates": {"S0": "PASS", **gates},
        "execution_commit": EXECUTION_COMMIT,
        "frozen_publication_commit": protocol["frozen_publication_commit"],
        "frozen_execution_files_sha256": protocol["S0"]["execution_files_sha256"],
        "frozen_publication_runtime_protocol": protocol["frozen_publication_runtime_protocol"],
        "numerical_equivalence": protocol["numerical_equivalence"],
        "result_artifact_sha256": artifacts,
        "result_table": table,
        "invalidated_attempt": protocol["invalidated_attempt"],
        "guardrails": protocol["guardrails"],
    }
    JSON_OUT.write_text(json.dumps(closeout, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    lines = [
        "# V6 supplemental known-topology/new-physics closeout",
        "",
        f"Execution commit: `{EXECUTION_COMMIT}`. S0/S1/S2/S3 all PASS. The historical "
        "`171cd5e...` timing attempt remains invalidated and none of its timing results are reused.",
        "",
        "The benchmark uses four frozen train-only geometries. It does not train, read labels, "
        "or access valid/test/sealed roles. Fresh and known-topology spans retain the frozen "
        "publication boundary from in-memory k/q/BC to synchronized 240825-node output.",
        "",
        "| Sweep | Route | Fresh median/p95 (s) | Known median/p95 (s) | Known throughput (sample/s) | Median speedup |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for sweep in ("K_only", "K_plus_Q_scale"):
        for route in ROUTES:
            fresh = next(row for row in table if row["route"] == route and row["sweep"] == sweep and row["mode"] == "fresh_new_case")
            known = next(row for row in table if row["route"] == route and row["sweep"] == sweep and row["mode"] == "known_topology_new_physics")
            lines.append(
                f"| {sweep} | {route} | {fresh['median_seconds']:.6f} / {fresh['p95_seconds']:.6f} | "
                f"{known['median_seconds']:.6f} / {known['p95_seconds']:.6f} | "
                f"{known['throughput_samples_per_second']:.6f} | {known['speedup_vs_fresh_median']:.3f}x |"
            )
    lines += [
        "",
        "Fresh E16384 is dominated by support/CV, group packing, and reconstruction-map construction; "
        "known-topology E16384 is dominated by dynamic anchor lookup and frozen packing. Fresh direct "
        "routes are dominated by query-graph construction; after reuse, direct routes are dominated by "
        "dynamic query packing and neural forward.",
        "",
        "No resident optimization, FVM, training, accuracy selection, or label-bearing evaluation was performed.",
    ]
    MD_OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": "passed", "rows": len(table), "execution_commit": EXECUTION_COMMIT}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
