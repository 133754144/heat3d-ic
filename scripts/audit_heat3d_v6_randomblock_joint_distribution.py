#!/usr/bin/env python3
"""Audit RandomBlock k/q/BC joint distribution and solver stability."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parent.parent


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _quantiles(values: Sequence[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "minimum": float(np.min(array)),
        "p05": float(np.quantile(array, 0.05)),
        "median": float(np.median(array)),
        "p95": float(np.quantile(array, 0.95)),
        "maximum": float(np.max(array)),
    }


def _safe_correlation(matrix: np.ndarray) -> list[list[float]]:
    result = np.corrcoef(matrix, rowvar=False)
    if not np.all(np.isfinite(result)):
        raise RuntimeError("non-finite joint-distribution correlation")
    return [[float(value) for value in row] for row in result]


def audit(
    samples_path: Path, blocks_path: Path, support_path: Path
) -> dict[str, Any]:
    samples = _read_csv(samples_path)
    blocks = _read_csv(blocks_path)
    support = _read_csv(support_path)
    if not samples or not blocks or not support:
        raise RuntimeError("joint audit refuses empty inputs")
    block_by_sample: defaultdict[str, list[Mapping[str, str]]] = defaultdict(list)
    for row in blocks:
        block_by_sample[str(row["sample_id"])].append(row)
    features = []
    feature_names = [
        "package_total_power_W",
        "top_h_W_m2K",
        "bottom_h_W_m2K",
        "q_power_weighted_W_m3",
        "q_max_W_m3",
        "k_log_geomean_W_mK",
        "q_block_count",
        "k_block_count",
    ]
    joint_rows: list[dict[str, Any]] = []
    k_counts: Counter[str] = Counter()
    bc_power_counts: Counter[str] = Counter()
    for sample in samples:
        sample_id = str(sample["sample_id"])
        sample_blocks = block_by_sample[sample_id]
        q_rows = [row for row in sample_blocks if row["family"] == "q"]
        k_rows = [row for row in sample_blocks if row["family"] == "k"]
        if not q_rows or not k_rows:
            raise RuntimeError(f"{sample_id}: missing q or k blocks")
        q_power_weighted = sum(
            float(row["power_fraction"]) * float(row["q_W_m3"])
            for row in q_rows
        )
        q_max = max(float(row["q_W_m3"]) for row in q_rows)
        k_values = [float(row["k_x_W_mK"]) for row in k_rows]
        for value in k_values:
            k_counts[f"{value:.12g}"] += 1
        k_log_geomean = float(np.exp(np.mean(np.log(k_values))))
        vector = [
            float(sample["package_total_power_W"]),
            float(sample["top_h_W_m2K"]),
            float(sample["bottom_h_W_m2K"]),
            q_power_weighted,
            q_max,
            k_log_geomean,
            float(len(q_rows)),
            float(len(k_rows)),
        ]
        features.append(vector)
        combo = (
            f"P={vector[0]:.12g}|top_h={vector[1]:.12g}|"
            f"bottom_h={vector[2]:.12g}"
        )
        bc_power_counts[combo] += 1
        joint_rows.append(
            {
                "sample_id": sample_id,
                "group_id": sample["group_id"],
                "split_role": sample["split_role"],
                "variant_id": sample["variant_id"],
                "intended_temperature_bin": int(
                    sample["intended_temperature_bin"]
                ),
                **dict(zip(feature_names, vector)),
                "peak_deltaT_K": float(sample["peak_deltaT_K"]),
                "cg_iterations": int(sample["cg_iterations"]),
                "energy_balance_relative_error": float(
                    sample["energy_balance_relative_error"]
                ),
                "linear_residual": float(sample["linear_residual"]),
            }
        )
    feature_matrix = np.asarray(features, dtype=np.float64)
    standardized = (
        feature_matrix - np.mean(feature_matrix, axis=0, keepdims=True)
    ) / np.std(feature_matrix, axis=0, keepdims=True)
    singular_values = np.linalg.svd(standardized, compute_uv=False)
    effective_rank = int(
        np.sum(singular_values > singular_values[0] * 1.0e-10)
    )
    per_bin = {}
    for bin_index in range(4):
        rows = [
            row
            for row in joint_rows
            if int(row["intended_temperature_bin"]) == bin_index
        ]
        per_bin[str(bin_index)] = {
            "sample_count": len(rows),
            "package_total_power_W": _quantiles(
                [float(row["package_total_power_W"]) for row in rows]
            ),
            "q_max_W_m3": _quantiles(
                [float(row["q_max_W_m3"]) for row in rows]
            ),
            "k_log_geomean_W_mK": _quantiles(
                [float(row["k_log_geomean_W_mK"]) for row in rows]
            ),
            "peak_deltaT_K": _quantiles(
                [float(row["peak_deltaT_K"]) for row in rows]
            ),
        }
    support_counts = [int(row["support_node_count"]) for row in support]
    power_errors = [
        abs(float(row["energy_balance_relative_error"])) for row in joint_rows
    ]
    residuals = [float(row["linear_residual"]) for row in joint_rows]
    iterations = [float(row["cg_iterations"]) for row in joint_rows]
    dataset_id = samples_path.stem.removesuffix("_samples")
    if dataset_id.startswith("v6_"):
        dataset_id = f"heat3d_{dataset_id}"
    payload = {
        "schema_version": "heat3d_v6_randomblock_joint_audit_v1",
        "dataset_id": dataset_id,
        "sample_count": len(samples),
        "group_count": len({row["group_id"] for row in samples}),
        "split_role_counts": dict(
            sorted(Counter(row["split_role"] for row in samples).items())
        ),
        "feature_names": feature_names,
        "pearson_correlation": _safe_correlation(feature_matrix),
        "standardized_effective_rank": effective_rank,
        "standardized_singular_values": [
            float(value) for value in singular_values
        ],
        "bc_power_combination_counts": dict(sorted(bc_power_counts.items())),
        "k_palette_block_counts": dict(sorted(k_counts.items())),
        "q_max_W_m3": _quantiles(
            [float(row["q_max_W_m3"]) for row in joint_rows]
        ),
        "q_power_weighted_W_m3": _quantiles(
            [float(row["q_power_weighted_W_m3"]) for row in joint_rows]
        ),
        "q_block_count": dict(
            sorted(Counter(int(row["q_block_count"]) for row in joint_rows).items())
        ),
        "k_block_count": dict(
            sorted(Counter(int(row["k_block_count"]) for row in joint_rows).items())
        ),
        "per_intended_temperature_bin": per_bin,
        "solver_stability": {
            "cg_iterations": _quantiles(iterations),
            "maximum_energy_balance_relative_error": max(power_errors),
            "maximum_linear_residual": max(residuals),
            "all_finite": all(
                math.isfinite(value)
                for row in joint_rows
                for value in (
                    float(row["peak_deltaT_K"]),
                    float(row["linear_residual"]),
                    float(row["energy_balance_relative_error"]),
                )
            ),
        },
        "support_coverage": {
            "block_count": len(support_counts),
            "support_nodes_per_block": _quantiles(support_counts),
            "zero_coverage_count": sum(value == 0 for value in support_counts),
        },
        "guardrails": {
            "training": False,
            "model_inference": False,
            "sample_filtering": False,
            "sample_replacement": False,
        },
        "rows": joint_rows,
    }
    payload["passed"] = bool(
        payload["solver_stability"]["all_finite"]
        and payload["solver_stability"][
            "maximum_energy_balance_relative_error"
        ]
        <= 1.0e-6
        and payload["solver_stability"]["maximum_linear_residual"] <= 1.0e-7
        and payload["support_coverage"]["zero_coverage_count"] == 0
    )
    return payload


def _markdown(payload: Mapping[str, Any]) -> str:
    solver = payload["solver_stability"]
    support = payload["support_coverage"]
    lines = [
        f"# {payload['dataset_id']} joint-distribution audit",
        "",
        f"- status: `{'passed' if payload['passed'] else 'failed'}`",
        f"- samples/groups: {payload['sample_count']} / {payload['group_count']}",
        f"- standardized physics-feature rank: "
        f"{payload['standardized_effective_rank']}/{len(payload['feature_names'])}",
        f"- unique P/top-h/bottom-h combinations: "
        f"{len(payload['bc_power_combination_counts'])}",
        f"- q max range: {payload['q_max_W_m3']['minimum']:.6g}–"
        f"{payload['q_max_W_m3']['maximum']:.6g} W/m³",
        f"- CG iterations median/P95/max: "
        f"{solver['cg_iterations']['median']:.1f}/"
        f"{solver['cg_iterations']['p95']:.1f}/"
        f"{solver['cg_iterations']['maximum']:.0f}",
        f"- max energy/residual: "
        f"{solver['maximum_energy_balance_relative_error']:.3e} / "
        f"{solver['maximum_linear_residual']:.3e}",
        f"- support nodes/block min/P05/median: "
        f"{support['support_nodes_per_block']['minimum']:.0f}/"
        f"{support['support_nodes_per_block']['p05']:.1f}/"
        f"{support['support_nodes_per_block']['median']:.1f}",
        "",
        "Pearson correlation 的变量顺序见 JSON `feature_names`。该审计只描述",
        "冻结数据生成的 k/q/BC 联合结构；不训练、不推理、不按温度过滤或替换样本。",
    ]
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--blocks", type=Path, required=True)
    parser.add_argument("--support", type=Path, required=True)
    parser.add_argument("--json-output", type=Path, required=True)
    parser.add_argument("--md-output", type=Path, required=True)
    args = parser.parse_args()
    paths = {
        key: value if value.is_absolute() else ROOT / value
        for key, value in vars(args).items()
    }
    payload = audit(paths["samples"], paths["blocks"], paths["support"])
    paths["json_output"].parent.mkdir(parents=True, exist_ok=True)
    paths["json_output"].write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    paths["md_output"].parent.mkdir(parents=True, exist_ok=True)
    paths["md_output"].write_text(_markdown(payload), encoding="utf-8")
    print(
        json.dumps(
            {
                key: value
                for key, value in payload.items()
                if key != "rows"
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if payload["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
