#!/usr/bin/env python3
"""Freeze the P7/U3 valid32 throughput and asymmetric-query closeout."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


def load(path: Path) -> dict:
    return json.loads(path.read_text())


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stat_median(payload: dict, key: str) -> float:
    return float(payload["runtime"]["fresh_sample"][key]["median_seconds"])


def stat_p95(payload: dict, key: str) -> float:
    return float(payload["runtime"]["fresh_sample"][key]["p95_seconds"])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--identity", type=Path, required=True)
    parser.add_argument("--profile-before", type=Path, required=True)
    parser.add_argument("--profile-after", type=Path, required=True)
    parser.add_argument("--u1", type=Path, required=True)
    parser.add_argument("--direct-smoke", type=Path, required=True)
    parser.add_argument("--direct", type=Path, required=True)
    parser.add_argument("--p6", type=Path, required=True)
    parser.add_argument("--p7", type=Path, required=True)
    parser.add_argument("--u2", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    args = parser.parse_args()

    protocol = load(args.protocol)
    identity = load(args.identity)
    before = load(args.profile_before)
    after = load(args.profile_after)
    u1 = load(args.u1)
    direct_smoke = load(args.direct_smoke)
    direct = load(args.direct)
    p6 = load(args.p6)
    p7 = load(args.p7)
    u2 = load(args.u2)
    role = protocol["role_contract"]

    assert identity["status"] == "passed" and identity["identity_hard_gate_passed"]
    assert identity["checkpoint_parameters_unchanged"]
    assert all(sample["passed"] for sample in identity["samples"])
    assert u1["status"] == "passed" and u1["sample_count"] == 32
    assert direct_smoke["status"] == "passed_smoke" and direct_smoke["sample_count"] == 1
    assert direct["status"] == "passed" and direct["sample_count"] == 32
    assert direct["output_mode"] == "direct" and direct["checkpoint_parameters_unchanged"]
    assert all(sample["shape"]["output_nodes"] == 240825 for sample in direct["samples"])
    for payload in (before, after, u1, direct_smoke, direct):
        assert payload["role_contract"] == role
    assert u2["status"] == "completed_no_go"

    p6_routes = {entry["route"]["route"]: entry for entry in p6["systems"]}
    rows: list[dict] = []

    def neural_row(name: str, payload: dict, semantic: str, batch: dict | None = None) -> dict:
        accuracy = payload["accuracy"]["full_field"]
        batch = batch or payload["batch"][0]
        return {
            "system": name,
            "resolution": payload["resolution"],
            "semantic": semantic,
            "batch_or_processes": batch["batch_size"],
            "point_global_pct": accuracy["point_global_true_rms_relative_rmse_pct"],
            "sample_first_pct": accuracy["sample_first_cv_relative_rmse_pct"],
            "raw_cv_rmse_K": accuracy["raw_cv_weighted_rmse_K"],
            "source_rmse_K": accuracy["source_rmse_K"],
            "peak_rmse_K": accuracy["peak_rmse_K"],
            "interface_rmse_K": accuracy["interface_drop_rmse_K"],
            "b1_e2e_median_s": stat_median(payload, "matched_continuous_e2e"),
            "b1_e2e_p95_s": stat_p95(payload, "matched_continuous_e2e"),
            "batch_wall_s": batch["batch_wall_seconds"],
            "samples_per_s": batch["samples_per_second"],
            "average_per_case_s": batch["average_per_case_seconds"],
            "marginal_per_case_s": batch["marginal_per_case_seconds"],
            "peak_vram_bytes": batch["peak_vram_bytes"],
            "provenance": "U3_new_valid32",
        }

    for route_name in ("E16384", "E32768"):
        payload = p6_routes[route_name]
        accuracy = payload["accuracy_reused_from_p5r"]
        for batch in payload["batch"]:
            rows.append({
                "system": route_name,
                "resolution": payload["route"]["resolution"],
                "semantic": "resident_prepared_group_inference",
                "batch_or_processes": batch["batch_size"],
                "point_global_pct": accuracy["point_global_pct"],
                "sample_first_pct": accuracy["sample_first_pct"],
                "raw_cv_rmse_K": accuracy["raw_cv_rmse_K"],
                "source_rmse_K": accuracy["source_rmse_K"],
                "peak_rmse_K": accuracy["peak_rmse_K"],
                "interface_rmse_K": accuracy["interface_rmse_K"],
                "b1_e2e_median_s": stat_median(payload, "matched_continuous_e2e"),
                "b1_e2e_p95_s": stat_p95(payload, "matched_continuous_e2e"),
                "batch_wall_s": batch["batch_wall_seconds"],
                "samples_per_s": batch["samples_per_second"],
                "average_per_case_s": batch["average_per_case_seconds"],
                "marginal_per_case_s": batch["marginal_per_case_seconds"],
                "peak_vram_bytes": batch["peak_vram_bytes"],
                "provenance": "P6_reused_valid32",
            })
    for batch in u1["batch"]:
        rows.append(neural_row("U1-32768", u1, "resident_prepared_group_inference", batch))
    rows.append(neural_row("U-direct240825", direct, "resident_direct_full_grid_inference"))
    for row in p7["rows"]:
        rows.append({
            "system": row["system"], "resolution": 16384 if row["system"] == "E16384" else 240825,
            "semantic": row["semantic"], "batch_or_processes": row["batch_or_processes"],
            "point_global_pct": None, "sample_first_pct": None, "raw_cv_rmse_K": None,
            "source_rmse_K": None, "peak_rmse_K": None, "interface_rmse_K": None,
            "b1_e2e_median_s": None, "b1_e2e_p95_s": None,
            "batch_wall_s": row["total_wall_s"], "samples_per_s": row["samples_per_s"],
            "average_per_case_s": row["average_per_case_s"], "marginal_per_case_s": row["marginal_per_case_s"],
            "peak_vram_bytes": row["peak_vram_bytes"], "provenance": row["provenance"],
        })

    before_dummy = stat_median(before, "dummy_local_p2r")
    after_dummy = stat_median(after, "dummy_local_p2r")
    direct_pg = direct["accuracy"]["full_field"]["point_global_true_rms_relative_rmse_pct"]
    e16384_pg = p6_routes["E16384"]["accuracy_reused_from_p5r"]["point_global_pct"]
    u1_pg = u1["accuracy"]["full_field"]["point_global_true_rms_relative_rmse_pct"]
    u1_b1 = stat_median(u1, "matched_continuous_e2e")
    e16384_b1 = stat_median(p6_routes["E16384"], "matched_continuous_e2e")
    direct_b1 = stat_median(direct, "matched_continuous_e2e")
    result = {
        "schema_version": "heat3d_v6_p1i_u3_asymmetric_production_closeout_v1",
        "status": "completed",
        "artifacts": {path.name: {"path": str(path), "sha256": sha256(path)} for path in (
            args.identity, args.profile_before, args.profile_after, args.u1, args.direct_smoke, args.direct
        )},
        "historical_u2": {"status": "completed_no_go_unchanged", "sha256": sha256(args.u2)},
        "identity_gate": {
            "passed": True, "sample_count": 32, "all_layer_and_output_arrays_bitwise_exact": True,
        },
        "u1_profile": {
            "before": {key: stat_median(before, key) for key in ("dummy_local_p2r", "graph_extraction", "host_tree", "inputs", "kwargs", "profiled_other")},
            "after": {key: stat_median(after, key) for key in ("dummy_local_p2r", "graph_extraction", "host_tree", "inputs", "kwargs", "profiled_other")},
            "dummy_local_p2r_speedup": before_dummy / after_dummy,
            "optimization": "direct exact _build_p2r_graph call; no full local build_graphs",
        },
        "u1_32768": {
            "point_global_pct": u1_pg, "b1_e2e_median_s": u1_b1,
            "accuracy_delta_vs_E16384_pp": u1_pg - e16384_pg,
            "b1_speedup_vs_E16384": e16384_b1 / u1_b1,
            "batch_sizes_attempted": [entry["batch_size"] for entry in u1["batch"]],
            "all_batches_passed_without_OOM": all(entry["status"] == "passed" for entry in u1["batch"]),
        },
        "direct_240825": {
            "smoke_passed": True, "valid32_passed": True, "point_global_pct": direct_pg,
            "b1_e2e_median_s": direct_b1, "b1_e2e_p95_s": stat_p95(direct, "matched_continuous_e2e"),
            "accuracy_delta_vs_E16384_pp": direct_pg - e16384_pg,
            "b1_speedup_vs_E16384": e16384_b1 / direct_b1,
            "output_nodes_all_samples": 240825, "adaptive_support": False, "reconstruction": False,
        },
        "p7": {
            "fresh_batch_GO": True,
            "best_neural_samples_per_s": p7["neural_best"]["samples_per_second"],
            "saturated_fvm_samples_per_s": p7["fvm_saturation"]["samples_per_second"],
            "throughput_ratio": p7["fresh_neural_speedup_vs_saturated_fvm"],
        },
        "decision": {
            "paper_mainline": "E16384",
            "P7": "GO",
            "U3_engineering_feasibility": "GO",
            "U3_production_replacement": "NO_GO",
            "reason": "U1-32768 does not materially beat E16384 B1 or resident throughput; U-direct240825 is faster but its valid32 PG penalty is 0.116568 pp and was not preregistered as production non-inferior.",
        },
        "rows": rows,
        "role_contract": role,
    }
    args.output_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    with args.output_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)

    u1_best = max(u1["batch"], key=lambda entry: entry["samples_per_second"])
    lines = [
        "# V6 P1i P7 + U3 runtime/throughput closeout", "",
        "All accuracy cells are frozen seed0 valid32. No training or test/sealed access occurred. Historical U2 remains unchanged.", "",
        "## Primary comparison", "",
        "| route | full PG (%) | raw CV (K) | B1 E2E median/p95 (s) | resident throughput | peak VRAM |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, payload in (("E16384", p6_routes["E16384"]), ("E32768", p6_routes["E32768"])):
        acc = payload["accuracy_reused_from_p5r"]
        best = max(payload["batch"], key=lambda entry: entry["samples_per_second"])
        lines.append(f"| {name} | {acc['point_global_pct']:.6f} | {acc['raw_cv_rmse_K']:.6f} | {stat_median(payload, 'matched_continuous_e2e'):.6f}/{stat_p95(payload, 'matched_continuous_e2e'):.6f} | B{best['batch_size']} {best['samples_per_second']:.3f} samples/s | {best['peak_vram_bytes']/1e9:.3f} GB |")
    for name, payload in (("U1-32768", u1), ("U-direct240825", direct)):
        acc = payload["accuracy"]["full_field"]
        best = max(payload["batch"], key=lambda entry: entry["samples_per_second"])
        lines.append(f"| {name} | {acc['point_global_true_rms_relative_rmse_pct']:.6f} | {acc['raw_cv_weighted_rmse_K']:.6f} | {stat_median(payload, 'matched_continuous_e2e'):.6f}/{stat_p95(payload, 'matched_continuous_e2e'):.6f} | B{best['batch_size']} {best['samples_per_second']:.3f} samples/s | {best['peak_vram_bytes']/1e9:.3f} GB |")
    lines += [
        "", "## P7 fresh throughput", "",
        f"E16384 fresh distinct-case throughput peaks at B{p7['neural_best']['batch_size']} = {p7['neural_best']['samples_per_second']:.3f} samples/s. Parallel FVM saturates at {p7['fvm_saturation']['process_count']} processes = {p7['fvm_saturation']['samples_per_second']:.3f} samples/s, a semantically matched {p7['fresh_neural_speedup_vs_saturated_fvm']:.2f}x throughput ratio. Fresh CPU preprocessing remains the dominant stage.",
        "", "## U3 exact adapter and direct output", "",
        f"The 1024→1024 identity gate passed bitwise for all 32 samples. Replacing the redundant full local graph build reduced `dummy_local_p2r` from {before_dummy:.6f} s to {after_dummy:.6f} s ({before_dummy/after_dummy:.2f}x) without changing checkpoint parameters or outputs.",
        f"U1-32768 completed actual B1/B4/B8/B16 attempts without OOM; best resident throughput is B{u1_best['batch_size']} = {u1_best['samples_per_second']:.3f} samples/s. Its optimized B1 E2E is {u1_b1:.6f} s, essentially tied with E16384 ({e16384_b1:.6f} s), while PG is {u1_pg:.6f}% (+{u1_pg-e16384_pg:.6f} pp).",
        f"The independent 1024→240825 direct smoke passed, followed by valid32: PG {direct_pg:.6f}%, B1 E2E {direct_b1:.6f} s ({e16384_b1/direct_b1:.2f}x versus E16384), peak VRAM {direct['memory']['peak_bytes_in_use']/1e9:.3f} GB. It avoids adaptive high-N support and reconstruction, but the PG penalty is +{direct_pg-e16384_pg:.6f} pp.",
        "", "## Decision", "",
        "P7 is GO. The paper production mainline remains E16384. U3 is an engineering-feasibility GO but a production-replacement NO-GO: U1-32768 does not dominate E16384, and U-direct240825's faster B1 result comes with a valid32 PG penalty exceeding the earlier +0.1 pp non-inferiority reference. A new independently preregistered confirmation would be required before promoting the direct route.",
        "", "`single-case latency`, `fresh distinct-case batch throughput`, `streamed prepared-host throughput`, and `resident inference throughput` are separate workload semantics and are not interchanged.",
    ]
    args.output_md.write_text("\n".join(lines) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
