#!/usr/bin/env python3
"""Build the valid-only P18 common-field comparison table.

This script combines already-produced, valid-only aggregate receipts.  It
never opens a dataset or checkpoint and deliberately keeps the native full-pool
DeepOHeat receipt out of the accuracy table because its training pool overlaps
the Heat3D valid cases.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from statistics import mean, stdev
from typing import Any


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def summary(values: list[float]) -> dict[str, float | int]:
    return {
        "n": len(values),
        "mean": mean(values),
        "sample_sd": stdev(values) if len(values) > 1 else 0.0,
    }


def read(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    boundaries = payload.get("hard_boundaries", {})
    forbidden = {
        "p1i_test_iid_accessed": boundaries.get(
            "p1i_test_iid_accessed",
            boundaries.get("test_iid_accessed", boundaries.get("test_iid")),
        ),
        "sealed_accessed": boundaries.get("sealed_accessed", boundaries.get("sealed")),
        "deepoheat_official100_accessed": boundaries.get(
            "deepoheat_official100_accessed",
            boundaries.get("official100_accessed", boundaries.get("official100")),
        ),
    }
    if any(value is not False for value in forbidden.values()):
        raise SystemExit(f"valid-only boundary is not closed: {path}")
    return {"path": str(path), "sha256": sha256(path), "payload": payload}


def md_row(row: dict[str, Any]) -> str:
    return (
        f"| {row['regime']} | {row['training_cases']} | {row['metric']} | "
        f"{row['best_mean']:.6f} ± {row['best_sd']:.6f} | "
        f"{row['final_mean']:.6f} ± {row['final_sd']:.6f} | "
        f"{row['wall_hours']:.3f} | {row['peak_memory_gib']:.3f} |"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--p15", type=Path, required=True)
    parser.add_argument("--p17", type=Path, required=True)
    parser.add_argument("--deepoheat768", type=Path, required=True)
    parser.add_argument("--heat3d200", type=Path, required=True)
    parser.add_argument("--p14u", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--markdown", type=Path, required=True)
    args = parser.parse_args()

    p15 = read(args.p15)["payload"]
    p17 = read(args.p17)["payload"]
    d768 = read(args.deepoheat768)["payload"]
    h200 = read(args.heat3d200)["payload"]
    u14 = read(args.p14u)["payload"]
    if p15.get("status") != "COMPLETE_VALID_ONLY_E600_THREE_SEEDS":
        raise SystemExit("P15 receipt is not complete valid-only e600")
    if p17.get("status") != "COMPLETE_VALID_ONLY_THREE_SEEDS":
        raise SystemExit("P17 receipt is not complete valid-only three seeds")

    p15agg = p15["aggregate"]
    p17agg = p17["aggregate"]
    d768agg = d768["aggregate"]
    u14rep = u14["representations"]["u_v2_dense_571256"]["across_seed"]

    # Derive the full-minus-valid training-pool cardinality from the frozen
    # P17 receipts rather than hard-coding the expected 100000-128 value.
    p17_case_counts = [
        int(p17["seeds"][str(seed)]["training_case_count"])
        for seed in (0, 1, 2)
    ]
    if len(set(p17_case_counts)) != 1:
        raise SystemExit(f"P17 training-pool counts disagree: {p17_case_counts}")
    p17_training_cases = p17_case_counts[0]

    p15_final_u = p15agg["dense_by_epoch"]["600"]["u_v2_dense_571256"]["sample_first_relative_rmse_pct"]
    p15_best_u = p15agg["best_common_scheduled_u_v2_metric"]
    p15_wall = p15agg["training_wall_seconds"]
    p15_mem = max(int(p15["runs"][str(seed)]["peak_device_memory"] or 0) for seed in (0, 1, 2))
    p15_param_count = int(p15["runs"]["0"].get("parameter_count") or 0)
    p15_u_latency = summary([
        float(p15["dense_receipts"][str(seed)]["600"]["runtime"]["u_v2_inference_seconds"])
        for seed in (0, 1, 2)
    ])
    p17_param_count = int(p17["seeds"]["0"].get("parameter_count") or 0)
    rows = [
        {
            "regime": "DeepOHeat-full-minus-valid128",
            "training_cases": p17_training_cases,
            "metric": "sample_first_relative_rmse_pct (full 571256)",
            "best_mean": p17agg["best_sample_first_relative_rmse_pct"]["mean"],
            "best_sd": p17agg["best_sample_first_relative_rmse_pct"]["sample_sd"],
            "final_mean": p17agg["final_sample_first_relative_rmse_pct"]["mean"],
            "final_sd": p17agg["final_sample_first_relative_rmse_pct"]["sample_sd"],
            "wall_hours": p17agg["training_wall_seconds"]["mean"] / 3600.0,
            "peak_memory_gib": p17agg["peak_device_bytes_max"] / (1024**3),
            "parameter_count": p17_param_count,
            "training_budget": "100000 iterations; 50 functions/iteration",
            "inference_latency_model_only_seconds": None,
            "inference_latency_end_to_end_dense_seconds": None,
            "latency_status": "NOT_MEASURED_IN_P17_RECEIPT",
            "modality": "PDE/BC physics-informed native full mesh",
        },
        {
            "regime": "DeepOHeat-768",
            "training_cases": 768,
            "metric": "sample_first_relative_rmse_pct (full 571256)",
            "best_mean": d768agg["best_primary_metric_pct_mean"],
            "best_sd": d768agg["best_primary_metric_pct_sample_sd"],
            "final_mean": d768agg["final_primary_metric_pct_mean"],
            "final_sd": d768agg["final_primary_metric_pct_sample_sd"],
            "wall_hours": d768agg["wall_seconds_mean"] / 3600.0,
            "peak_memory_gib": d768agg["peak_device_bytes_max"] / (1024**3),
            "parameter_count": int(d768agg.get("parameter_count", p17_param_count)),
            "training_budget": "100000 iterations; 50 functions/iteration",
            "inference_latency_model_only_seconds": None,
            "inference_latency_end_to_end_dense_seconds": None,
            "latency_status": "NOT_MEASURED_IN_P14_RECEIPT",
            "modality": "PDE/BC physics-informed native full mesh",
        },
        {
            "regime": "Heat3D-768-e600",
            "training_cases": 768,
            "metric": "sample_first_relative_rmse_pct (U-v2 full 571256)",
            "best_mean": p15_best_u["mean"],
            "best_sd": p15_best_u["sample_sd"],
            "final_mean": p15_final_u["mean"],
            "final_sd": p15_final_u["sample_sd"],
            "wall_hours": p15_wall["mean"] / 3600.0,
            "peak_memory_gib": p15_mem / (1024**3),
            "parameter_count": p15_param_count,
            "training_budget": "600 epochs; B24; 768 supervised cases",
            "inference_latency_model_only_seconds": None,
            "inference_latency_end_to_end_dense_seconds": p15_u_latency,
            "latency_status": "U-v2 dense includes full-query graph construction and direct-query forward",
            "modality": "supervised sparse 1024 support + U-v2 reconstruction",
        },
    ]

    out = {
        "schema_version": "heat3d_v7_g2_p18_common_valid_comparison_v1",
        "status": "COMPLETE_VALID_ONLY_COMPARISON_TABLE",
        "evaluation": {
            "cases": 128,
            "domain": "101x101x56=571256",
            "temperature_space": "deltaT_K",
            "primary_metric": "sample_first_relative_rmse_pct",
            "test_iid": "sealed",
            "sealed": "sealed",
            "deepoheat_official100": "sealed",
        },
        "inputs": {
            "p15": {"path": str(args.p15), "sha256": sha256(args.p15)},
            "p17": {"path": str(args.p17), "sha256": sha256(args.p17)},
            "deepoheat768": {"path": str(args.deepoheat768), "sha256": sha256(args.deepoheat768)},
            "heat3d200": {"path": str(args.heat3d200), "sha256": sha256(args.heat3d200)},
            "p14u": {"path": str(args.p14u), "sha256": sha256(args.p14u)},
        },
        "rows": rows,
        "p14_1_e200_views": {
            "heat3d_native_1024": u14["representations"]["native_1024"]["across_seed"],
            "heat3d_idw_dense": u14["representations"]["idw_dense_571256"]["across_seed"],
            "heat3d_u_v2_dense": u14rep,
            "idw_oracle_floor": u14["representations"]["oracle_idw_gt_support_571256"]["across_seed"],
            "u_v2_oracle": "NOT_DEFINED_FOR_DIRECT_QUERY",
        },
        "fairness": {
            "p17_valid_overlap": 0,
            "p17_excluded_valid_count": 128,
            "same_physical_case_budget": "DeepOHeat-768 vs Heat3D-768",
            "same_information_budget": False,
            "native_full_reference_rankable_on_valid128": False,
        },
        "efficiency_contract": {
            "rows": [
                {
                    "regime": row["regime"],
                    "training_cases": row["training_cases"],
                    "training_budget": row["training_budget"],
                    "parameter_count": row["parameter_count"],
                    "wall_hours_per_seed": row["wall_hours"],
                    "peak_memory_gib": row["peak_memory_gib"],
                    "inference_latency_model_only_seconds": row["inference_latency_model_only_seconds"],
                    "inference_latency_end_to_end_dense_seconds": row["inference_latency_end_to_end_dense_seconds"],
                    "latency_status": row["latency_status"],
                }
                for row in rows
            ],
            "latency_boundary": {
                "model_only": "model forward only; not available for DeepOHeat P17/P14 receipts",
                "end_to_end_dense": "includes query/preprocessing and dense formation; Heat3D U-v2 timing includes full-query graph construction + direct-query forward",
                "same_hardware_remeasurement_required": True,
            },
        },
        "interpretation": {
            "q1": "Compare DeepOHeat-768 and Heat3D-768 only as same physical-case/data-regime evidence.",
            "q2": "Compare full-minus-valid128 versus 768-case DeepOHeat for data-scale effect.",
            "q3": "Heat3D-768 versus full-minus-valid128 is physical-case efficiency, not equal information/compute budget.",
            "q4": "Use P14.1 oracle floors as lower-bound diagnostics; do not subtract them from prediction error.",
        },
        "hard_boundaries": {"test_iid": False, "sealed": False, "deepoheat_official100": False},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = [
        "# V7 G2-P18：common-valid comparison",
        "",
        "仅使用 Heat3D valid128；test_iid、sealed 与 DeepOHeat official100 均保持锁定。",
        "指标为 full 571256-point temperature-space sample-first relative RMSE [%]。",
        "",
        "| regime | training cases | metric | best mean ± SD | final mean ± SD | wall h | peak GiB |",
        "|---|---:|---|---:|---:|---:|---:|",
    ] + [md_row(row) for row in rows] + [
        "",
        "## 公平性边界",
        "",
        "DeepOHeat-768 与 Heat3D-768 共享 768/128 physical-case split，但不共享信息预算：前者使用 PDE/BC physics-informed full mesh，后者使用 supervised temperature labels 与 1024 sparse support。",
        "DeepOHeat-full-minus-valid128 的 128 个验证 case 在训练池中被严格排除，可用于共同 valid 比较；历史 native-full pool 与 valid128 重叠，不能作 valid 排名。",
        "",
        "## P14.1 reconstruction diagnostic",
        "",
        f"e200 U-v2 dense mean = {u14rep['sample_first_relative_rmse_pct']['mean']:.6f}%；IDW 与 U-v2 oracle 仅作 reconstruction floor/diagnostic，不做误差相减分解。",
        "",
        "## Efficiency and latency boundary",
        "",
        "| regime | parameters | training budget | model-only latency | end-to-end dense latency |",
        "|---|---:|---|---:|---:|",
    ]
    for row in rows:
        model_latency = "not measured" if row["inference_latency_model_only_seconds"] is None else f"{row['inference_latency_model_only_seconds']['mean']:.3f} s"
        dense_latency = "not measured" if row["inference_latency_end_to_end_dense_seconds"] is None else f"{row['inference_latency_end_to_end_dense_seconds']['mean']:.3f} ± {row['inference_latency_end_to_end_dense_seconds']['sample_sd']:.3f} s"
        lines.append(
            f"| {row['regime']} | {row['parameter_count']:,} | {row['training_budget']} | {model_latency} | {dense_latency} |"
        )
    lines.extend([
        "",
        "Latency must be remeasured on the same hardware before Pareto claims. Heat3D U-v2 end-to-end timing includes full-query graph construction and direct-query forward; DeepOHeat P14/P17 receipts do not contain a directly comparable dense latency measurement.",
        "",
    ])
    args.markdown.parent.mkdir(parents=True, exist_ok=True)
    args.markdown.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": out["status"], "output": str(args.output), "markdown": str(args.markdown)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
