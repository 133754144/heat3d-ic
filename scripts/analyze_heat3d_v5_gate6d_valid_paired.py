#!/usr/bin/env python3
"""Valid-only paired attribution for N3 best e402 versus Scratch-L2 best e353."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Callable

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_N3 = ROOT / "configs/heat3d_v5/gate6d/V4P5_07_frozen_gate5_valid_only_evaluation.json"
DEFAULT_L2 = ROOT / "configs/heat3d_v5/gate6d/V4P5_12_frozen_gate5_evaluation.json"
DEFAULT_JSON = ROOT / "configs/heat3d_v5/gate6d/n3_l2_valid_paired.json"
DEFAULT_MD = ROOT / "docs/v5_gate6d_n3_l2_valid_paired.md"


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n3-evaluation", type=Path, default=DEFAULT_N3)
    parser.add_argument("--l2-evaluation", type=Path, default=DEFAULT_L2)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--output-md", type=Path, default=DEFAULT_MD)
    return parser.parse_args()


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _rows(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    rows = payload["per_sample"]
    result = {str(row["sample_id"]): row for row in rows}
    if len(result) != 128 or set(row["split"] for row in rows) != {"valid_iid"}:
        raise ValueError("paired input must contain exactly 128 valid_iid samples")
    return result


def _native(row: dict[str, Any], name: str) -> float:
    return float(row["native_shape_scale"][name])


def _features(row: dict[str, Any]) -> dict[str, float]:
    context = row["attribution_context"]
    return {
        "true_cv_rms_deltaT_K": float(row["true_scale_cv_rms_K"]),
        "total_power_W": float(context["P_operator_W"]),
        "q_weighted_inverse_conductivity_mK_W": float(context["q_weighted_inverse_kz_mK_W"]),
        "source_concentration": float(context["source_concentration"]),
        "anisotropy_xy_over_z": float(context["anisotropy_xy_over_z"]),
        "top_h_W_m2K": math.exp(float(context["log_top_h_W_m2K"])),
    }


ERROR_ACCESSORS: dict[str, Callable[[dict[str, Any]], float]] = {
    "point_global_sse_K2": lambda row: float(row["point_error_squared_sum"]),
    "sample_relative_rmse_pct": lambda row: 100.0 * float(row["sample_cv_relative_rmse"]),
    "shape_cv_rmse": lambda row: float(row["shape_cv_rmse"]),
    "scale_log_abs_error": lambda row: abs(float(row["scale_log_error"])),
    "amplitude_abs_error": lambda row: abs(float(row["amplitude_ratio"]) - 1.0),
    "oracle_scale_relative_rmse_pct": lambda row: _native(row, "oracle_scale_relative_rmse_pct"),
    "oracle_shape_relative_rmse_pct": lambda row: _native(row, "oracle_shape_relative_rmse_pct"),
}


def _mean(rows: list[dict[str, Any]], key: str) -> float:
    return float(np.mean([float(row[key]) for row in rows]))


def _quartile_report(rows: list[dict[str, Any]], feature: str) -> dict[str, Any]:
    values = np.asarray([row["features"][feature] for row in rows], dtype=np.float64)
    edges = np.quantile(values, [0.0, 0.25, 0.5, 0.75, 1.0])
    bins = np.searchsorted(edges[1:-1], values, side="right")
    result = {"edges": edges.tolist(), "bins": []}
    for index in range(4):
        selected = [row for row, bin_index in zip(rows, bins, strict=True) if bin_index == index]
        result["bins"].append({
            "quartile": f"Q{index + 1}",
            "sample_count": len(selected),
            "feature_mean": _mean([{"v": row["features"][feature]} for row in selected], "v"),
            "n3_sample_relative_rmse_pct": _mean(selected, "n3_sample_relative_rmse_pct"),
            "l2_sample_relative_rmse_pct": _mean(selected, "l2_sample_relative_rmse_pct"),
            "l2_minus_n3_sample_relative_rmse_pct": _mean(selected, "delta_sample_relative_rmse_pct"),
            "n3_point_global_sse_K2": float(sum(row["n3_point_global_sse_K2"] for row in selected)),
            "l2_point_global_sse_K2": float(sum(row["l2_point_global_sse_K2"] for row in selected)),
            "l2_minus_n3_point_global_sse_K2": float(sum(row["delta_point_global_sse_K2"] for row in selected)),
            "improved_fraction": float(np.mean([row["delta_sample_relative_rmse_pct"] < 0.0 for row in selected])),
        })
    return result


def _compact(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "sample_id": row["sample_id"],
        "n3_sample_relative_rmse_pct": row["n3_sample_relative_rmse_pct"],
        "l2_sample_relative_rmse_pct": row["l2_sample_relative_rmse_pct"],
        "l2_minus_n3_sample_relative_rmse_pct": row["delta_sample_relative_rmse_pct"],
        "l2_minus_n3_point_global_sse_K2": row["delta_point_global_sse_K2"],
        "features": row["features"],
    }


def main() -> int:
    args = _args()
    n3_payload = _read(args.n3_evaluation)
    l2_payload = _read(args.l2_evaluation)
    n3_report = n3_payload["reports"]["best"]["valid_iid"]
    l2_report = l2_payload["reports"]["best"]["valid_iid"]
    n3 = _rows(n3_report)
    l2 = _rows(l2_report)
    if set(n3) != set(l2):
        raise ValueError("N3/L2 valid sample IDs differ")

    rows = []
    for sample_id in sorted(n3):
        left, right = n3[sample_id], l2[sample_id]
        if _features(left) != _features(right):
            raise ValueError(f"{sample_id}: input attribution variables differ")
        row: dict[str, Any] = {
            "sample_id": sample_id,
            "features": _features(left),
            "true_point_sse_K2": float(left["point_true_squared_sum"]),
            "n3_scale_log_error": float(left["scale_log_error"]),
            "l2_scale_log_error": float(right["scale_log_error"]),
            "delta_scale_log_signed_bias": float(right["scale_log_error"] - left["scale_log_error"]),
            "n3_amplitude_ratio": float(left["amplitude_ratio"]),
            "l2_amplitude_ratio": float(right["amplitude_ratio"]),
        }
        for name, accessor in ERROR_ACCESSORS.items():
            left_value, right_value = accessor(left), accessor(right)
            row[f"n3_{name}"] = left_value
            row[f"l2_{name}"] = right_value
            row[f"delta_{name}"] = right_value - left_value
        rows.append(row)

    quartiles = {feature: _quartile_report(rows, feature) for feature in rows[0]["features"]}
    sorted_delta = sorted(rows, key=lambda row: row["delta_sample_relative_rmse_pct"])
    positive_improvements = np.asarray(
        [max(-row["delta_sample_relative_rmse_pct"], 0.0) for row in rows], dtype=np.float64
    )
    total_improvement = float(positive_improvements.sum())
    top_shares = {}
    for count in (5, 10):
        top_shares[f"top_{count}_positive_improvement_share"] = (
            float(np.sort(positive_improvements)[-count:].sum() / total_improvement)
            if total_improvement > 0.0 else 0.0
        )
    true_scale_values = np.asarray([row["features"]["true_cv_rms_deltaT_K"] for row in rows])
    q4_threshold = float(np.quantile(true_scale_values, 0.75))
    q4_mask = true_scale_values >= q4_threshold
    q4_positive_share = (
        float(positive_improvements[q4_mask].sum() / total_improvement)
        if total_improvement > 0.0 else 0.0
    )
    top10_concentrated = top_shares["top_10_positive_improvement_share"] > 0.5
    q4_concentrated = q4_positive_share > 0.5

    aggregate = {
        "sample_count": 128,
        "n3_point_global_relative_rmse_pct": 100.0 * math.sqrt(
            sum(row["n3_point_global_sse_K2"] for row in rows) / sum(row["true_point_sse_K2"] for row in rows)
        ),
        "l2_point_global_relative_rmse_pct": 100.0 * math.sqrt(
            sum(row["l2_point_global_sse_K2"] for row in rows) / sum(row["true_point_sse_K2"] for row in rows)
        ),
        "n3_sample_first_relative_rmse_pct": _mean(rows, "n3_sample_relative_rmse_pct"),
        "l2_sample_first_relative_rmse_pct": _mean(rows, "l2_sample_relative_rmse_pct"),
        "mean_l2_minus_n3_sample_relative_rmse_pct": _mean(rows, "delta_sample_relative_rmse_pct"),
        "improved_sample_count": sum(row["delta_sample_relative_rmse_pct"] < 0.0 for row in rows),
        "regressed_sample_count": sum(row["delta_sample_relative_rmse_pct"] > 0.0 for row in rows),
        **top_shares,
        "true_cv_rms_q4_threshold_K": q4_threshold,
        "q4_positive_improvement_share": q4_positive_share,
        "improvement_concentrated_in_top10_samples": top10_concentrated,
        "improvement_concentrated_in_true_cv_rms_q4": q4_concentrated,
        "conclusion": (
            "L2 improvement is concentrated in a small top-improvement subset and true-CV-RMS Q4."
            if top10_concentrated and q4_concentrated else
            "L2 improvement is concentrated in a small top-improvement subset, but not predominantly in true-CV-RMS Q4."
            if top10_concentrated else
            "L2 improvement is not concentrated in only a small top-improvement subset."
        ),
    }
    payload = {
        "schema_version": "heat3d_v5_gate6d_valid_paired_v1",
        "data_roles": ["valid_iid"],
        "forbidden_roles_accessed": [],
        "n3": {"config_id": n3_payload["config_id"], "checkpoint_epoch": 402},
        "l2": {"config_id": l2_payload["config_id"], "checkpoint_epoch": 353},
        "aggregate": aggregate,
        "quartiles": quartiles,
        "top_improvement": [_compact(row) for row in sorted_delta[:10]],
        "top_regression": [_compact(row) for row in reversed(sorted_delta[-10:])],
        "per_sample": rows,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    md = [
        "# Gate 6D N3-L2 valid-only paired attribution",
        "",
        "只使用 `valid_iid`：N3 best e402 对 L2 best e353。test/hard 未用于本分析。",
        "",
        "| metric | N3 | L2 | L2-N3 |",
        "|---|---:|---:|---:|",
        f"| point-global relative RMSE | {aggregate['n3_point_global_relative_rmse_pct']:.6f}% | {aggregate['l2_point_global_relative_rmse_pct']:.6f}% | {aggregate['l2_point_global_relative_rmse_pct'] - aggregate['n3_point_global_relative_rmse_pct']:.6f} pp |",
        f"| sample-first relative RMSE | {aggregate['n3_sample_first_relative_rmse_pct']:.6f}% | {aggregate['l2_sample_first_relative_rmse_pct']:.6f}% | {aggregate['mean_l2_minus_n3_sample_relative_rmse_pct']:.6f} pp |",
        "",
        f"改善样本 {aggregate['improved_sample_count']}/128，退化样本 {aggregate['regressed_sample_count']}/128。",
        f"top-10 改善样本占全部正向改善 {100.0 * aggregate['top_10_positive_improvement_share']:.2f}%；true CV-RMS Q4 占 {100.0 * aggregate['q4_positive_improvement_share']:.2f}%。",
        "",
        f"结论：{aggregate['conclusion']}",
        "",
        "六个变量的四分位统计、逐样本 SSE/shape/scale/amplitude/oracle 指标和 top improvement/regression 均保存在 JSON。",
        "",
    ]
    args.output_md.write_text("\n".join(md), encoding="utf-8")
    print(json.dumps({"status": "passed", "aggregate": aggregate}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
