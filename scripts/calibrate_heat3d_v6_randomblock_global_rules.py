#!/usr/bin/env python3
"""Audit one preregistered global RandomBlock rule revision on pilot layouts."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import time
from typing import Any

import numpy as np

import heat3d_v6_randomblock_core as core


ROOT = Path(__file__).resolve().parent.parent
GLOBAL_CANDIDATES = {
    "v5": {
        "package_total_power_W": 16.5,
        "top_h_W_m2K": 2050.0,
        "bottom_h_W_m2K": 1000.0,
    },
    "v6": {
        "package_total_power_W": 12.5,
        "top_h_W_m2K": 1000.0,
        "bottom_h_W_m2K": 500.0,
    },
    "v7": {
        "package_total_power_W": 12.5,
        "top_h_W_m2K": 1000.0,
        "bottom_h_W_m2K": 500.0,
    },
}


def run(config_path: Path) -> dict[str, Any]:
    config = core.load_config(config_path)
    if str(config["stage"]) != "pilot128" or int(config["group_count"]) != 16:
        raise core.RandomBlockError("calibration requires the frozen pilot128")
    mesh = core.build_mesh(config["physics"])
    groups = {
        str(group["group_id"]): group for group in config["layout_groups"]
    }
    layouts = {
        group_id: core.validate_layout(group, mesh)
        for group_id, group in groups.items()
    }
    rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    for case in config["cases"]:
        variant = str(case["variant_id"])
        if variant not in GLOBAL_CANDIDATES:
            continue
        revised = dict(case)
        revised.update(GLOBAL_CANDIDATES[variant])
        group_id = str(case["group_id"])
        k_diag, q, _ = core.build_case_fields(
            revised, groups[group_id], mesh, layouts[group_id]
        )
        temperature, solver = core.solve_case(
            mesh,
            k_diag,
            q,
            top_h=float(revised["top_h_W_m2K"]),
            bottom_h=float(revised["bottom_h_W_m2K"]),
        )
        metrics = core.case_metrics(mesh, temperature, q, solver)
        rows.append(
            {
                "sample_id": case["sample_id"],
                "group_id": group_id,
                "variant_id": variant,
                "intended_temperature_bin": int(
                    case["intended_temperature_bin"]
                ),
                "realized_temperature_bin": metrics[
                    "realized_temperature_bin"
                ],
                "peak_deltaT_K": metrics["peak_deltaT_K"],
                "energy_balance_relative_error": metrics[
                    "energy_balance_relative_error"
                ],
                "linear_residual": metrics["linear_residual"],
            }
        )
    by_variant: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_variant[str(row["variant_id"])].append(row)
    summaries = {}
    for variant, variant_rows in sorted(by_variant.items()):
        peaks = [float(row["peak_deltaT_K"]) for row in variant_rows]
        bins = Counter(
            "outside"
            if row["realized_temperature_bin"] is None
            else str(row["realized_temperature_bin"])
            for row in variant_rows
        )
        intended = int(variant_rows[0]["intended_temperature_bin"])
        summaries[variant] = {
            "candidate": GLOBAL_CANDIDATES[variant],
            "sample_count": len(variant_rows),
            "peak_deltaT_K": {
                "minimum": min(peaks),
                "median": float(np.median(peaks)),
                "maximum": max(peaks),
            },
            "realized_temperature_bin_counts": dict(sorted(bins.items())),
            "all_match_intended_bin": bins == Counter({str(intended): 16}),
        }
    return {
        "schema_version": "heat3d_v6_randomblock_global_rule_calibration_v1",
        "source_dataset_id": config["dataset_id"],
        "source_protocol_sha256": config["provenance"]["protocol_sha256"],
        "method": "one global candidate per variant evaluated on all 16 frozen pilot layouts",
        "per_sample_inverse_calibration": False,
        "sample_filtering": False,
        "sample_replacement": False,
        "training": False,
        "model_inference": False,
        "elapsed_seconds": time.perf_counter() - started,
        "variants": summaries,
        "passed": all(
            bool(summary["all_match_intended_bin"])
            for summary in summaries.values()
        ),
        "maximum_energy_balance_relative_error": max(
            abs(float(row["energy_balance_relative_error"])) for row in rows
        ),
        "maximum_linear_residual": max(
            float(row["linear_residual"]) for row in rows
        ),
        "rows": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    config_path = (
        args.config if args.config.is_absolute() else ROOT / args.config
    ).resolve()
    result = run(config_path)
    if args.output:
        output = args.output if args.output.is_absolute() else ROOT / args.output
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
