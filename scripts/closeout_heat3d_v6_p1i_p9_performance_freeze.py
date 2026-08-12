#!/usr/bin/env python3
"""Freeze P9 persistent neural/FVM timing from preregistered raw artifacts."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    for name in ("protocol", "exact_dir", "neural", "fvm", "output_json", "output_csv", "output_md"):
        parser.add_argument(f"--{name.replace('_', '-')}", dest=name, type=Path, required=True)
    args = parser.parse_args()
    protocol, neural, fvm = load(args.protocol), load(args.neural), load(args.fvm)
    exact = {name: load(args.exact_dir / f"{name}.json") for name in protocol["neural"]["persistent_preprocessing_backends"]}
    if not all(row["status"] == "passed" and row["full_payload_exact_vs_serial"] for row in exact.values()):
        raise RuntimeError("P9 preprocessing exact gate did not pass")
    if neural["status"] != "passed" or fvm["status"] != "passed":
        raise RuntimeError("P9 timing inputs are incomplete")
    if neural["role_contract"] != protocol["role_contract"]:
        raise RuntimeError("neural role contract drift")
    if fvm["role_contract"]["training"] or fvm["role_contract"]["test"] or fvm["role_contract"]["sealed"]:
        raise RuntimeError("FVM role contract drift")

    neural_b32_sps = 32.0 / neural["full_valid32_B32"]["median_seconds"]
    neural_b16_sps = 32.0 / neural["full_valid32_two_B16"]["median_seconds"]
    fvm_sps = float(fvm["saturation_samples_per_second"])
    rows: list[dict[str, Any]] = []
    for backend, row in exact.items():
        rows.append({"system": "E16384", "semantic": "preprocessing_exact_valid32", "backend_or_processes": backend, "batch": "", "median_seconds": row["steady_wall_seconds"], "p95_seconds": "", "samples_per_second": row["samples_per_second"], "marginal_added_case_seconds": "", "speedup_vs_fvm": "", "peak_vram_bytes": "", "provenance": str(args.exact_dir / f"{backend}.json")})
    rows += [
        {"system": "E16384", "semantic": "fresh_single_case", "backend_or_processes": neural["winner"], "batch": 1, "median_seconds": neural["fresh_b1"]["median_seconds"], "p95_seconds": neural["fresh_b1"]["p95_seconds"], "samples_per_second": 1.0 / neural["fresh_b1"]["median_seconds"], "marginal_added_case_seconds": "", "speedup_vs_fvm": fvm_sps / (1.0 / neural["fresh_b1"]["median_seconds"]), "peak_vram_bytes": neural["peak_vram_bytes"], "provenance": str(args.neural)},
        {"system": "E16384", "semantic": "warm_resident", "backend_or_processes": "GPU", "batch": 1, "median_seconds": neural["resident_inference"]["1"]["median_seconds"], "p95_seconds": neural["resident_inference"]["1"]["p95_seconds"], "samples_per_second": 1.0 / neural["resident_inference"]["1"]["median_seconds"], "marginal_added_case_seconds": "", "speedup_vs_fvm": "", "peak_vram_bytes": neural["peak_vram_bytes"], "provenance": str(args.neural)},
        {"system": "E16384", "semantic": "full_valid32_2xB16", "backend_or_processes": neural["winner"], "batch": "2x16", "median_seconds": neural["full_valid32_two_B16"]["median_seconds"], "p95_seconds": neural["full_valid32_two_B16"]["p95_seconds"], "samples_per_second": neural_b16_sps, "marginal_added_case_seconds": neural["marginal_added_case"]["median_seconds"], "speedup_vs_fvm": neural_b16_sps / fvm_sps, "peak_vram_bytes": neural["peak_vram_bytes"], "provenance": str(args.neural)},
        {"system": "E16384", "semantic": "full_valid32_B32", "backend_or_processes": neural["winner"], "batch": 32, "median_seconds": neural["full_valid32_B32"]["median_seconds"], "p95_seconds": neural["full_valid32_B32"]["p95_seconds"], "samples_per_second": neural_b32_sps, "marginal_added_case_seconds": neural["marginal_added_case"]["median_seconds"], "speedup_vs_fvm": neural_b32_sps / fvm_sps, "peak_vram_bytes": neural["peak_vram_bytes"], "provenance": str(args.neural)},
    ]
    for row in fvm["rows"]:
        rows.append({"system": "FVM", "semantic": "persistent_valid32", "backend_or_processes": row["process_count"], "batch": 32, "median_seconds": row["steady_wall"]["median_seconds"], "p95_seconds": row["steady_wall"]["p95_seconds"], "samples_per_second": row["throughput"]["median_seconds"], "marginal_added_case_seconds": "", "speedup_vs_fvm": 1.0, "peak_vram_bytes": "", "provenance": str(args.fvm)})
    result = {
        "schema_version": "heat3d_v6_p1i_p9_performance_freeze_closeout_v1",
        "status": "completed_frozen",
        "decision": "freeze_E16384_performance_no_more_valid32_architecture_tuning",
        "protocol_sha256": sha256(args.protocol),
        "raw_artifacts": {"neural": {"path": str(args.neural), "sha256": sha256(args.neural)}, "fvm": {"path": str(args.fvm), "sha256": sha256(args.fvm)}, "exact": {key: {"path": str(args.exact_dir / f"{key}.json"), "sha256": sha256(args.exact_dir / f"{key}.json")} for key in exact}},
        "preprocessing_exact": {"all_backends_exact": True, "scope": protocol["neural"]["exact_hash_scope"]},
        "neural": neural,
        "fvm": fvm,
        "publication": {"fresh_B1_median_seconds": neural["fresh_b1"]["median_seconds"], "fresh_B1_p95_seconds": neural["fresh_b1"]["p95_seconds"], "warm_B1_median_seconds": neural["resident_inference"]["1"]["median_seconds"], "marginal_added_case_median_seconds": neural["marginal_added_case"]["median_seconds"], "B32_samples_per_second": neural_b32_sps, "two_B16_samples_per_second": neural_b16_sps, "fvm_saturated_processes": fvm["saturation_process_count"], "fvm_saturated_samples_per_second": fvm_sps, "B32_throughput_speedup_vs_saturated_fvm": neural_b32_sps / fvm_sps, "two_B16_throughput_speedup_vs_saturated_fvm": neural_b16_sps / fvm_sps},
        "rows": rows,
        "role_contract": protocol["role_contract"],
    }
    args.output_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    with args.output_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n"); writer.writeheader(); writer.writerows(rows)
    pub = result["publication"]
    lines = ["# V6 P1i P9 performance freeze", "", "P9 uses persistent workers with untimed warmup on the same WSL2 host. Hashing, equivalence checks, metrics and serialization are outside production timing.", "", "## Frozen metrics", "", "| metric | value |", "|---|---:|", f"| fresh B1 median / p95 | {pub['fresh_B1_median_seconds']:.6f} / {pub['fresh_B1_p95_seconds']:.6f} s |", f"| warm resident B1 median | {pub['warm_B1_median_seconds']:.6f} s |", f"| marginal added-case median | {pub['marginal_added_case_median_seconds']:.6f} s |", f"| 2xB16 valid32 throughput | {pub['two_B16_samples_per_second']:.3f} samples/s |", f"| B32 valid32 throughput | {pub['B32_samples_per_second']:.3f} samples/s |", f"| saturated FVM P={pub['fvm_saturated_processes']} | {pub['fvm_saturated_samples_per_second']:.3f} samples/s |", f"| B32 neural/FVM throughput | {pub['B32_throughput_speedup_vs_saturated_fvm']:.3f}x |", "", "All registered preprocessing backends reproduce the complete anchor/query groups, inputs, graph, physics/context, selected CV and reconstruction map hashes exactly. The route is frozen after this valid32 closeout; no test/sealed role was accessed and no training occurred."]
    args.output_md.write_text("\n".join(lines) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
