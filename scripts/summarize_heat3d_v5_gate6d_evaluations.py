#!/usr/bin/env python3
"""Summarize frozen Gate-5 reruns and compare them with Gate-6C collector JSON."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import subprocess
import sys
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REGISTRY = ROOT / "configs/heat3d_v5/v5_gate6c_scratch_loss_registry.csv"
DEFAULT_L1 = ROOT / "configs/heat3d_v5/gate6d/V4P5_11_frozen_gate5_evaluation.json"
DEFAULT_L2 = ROOT / "configs/heat3d_v5/gate6d/V4P5_12_frozen_gate5_evaluation.json"
DEFAULT_EQ_JSON = ROOT / "configs/heat3d_v5/gate6d/evaluator_equivalence.json"
DEFAULT_EQ_MD = ROOT / "docs/v5_gate6d_evaluator_equivalence.md"
DEFAULT_METRICS_MD = ROOT / "docs/v5_gate6d_frozen_evaluation.md"
FROZEN_COMMIT = "639872abcb0f7afd3b6c2d319a7d395bde75c9a4"
COLLECTOR_REGISTRY_COMMIT = "2cb20af5be8f9e8f2d6d2e409baf4305ffd458bf"
ROLES = (
    "valid_iid",
    "test_iid",
    "hard_train_holdout",
    "hard_challenge_valid",
    "hard_challenge_test",
)
KINDS = ("best", "final")
CONFIGS = {
    "L1": "V4P5_11_gate6c_scratch_l1_tail_balanced",
    "L2": "V4P5_12_gate6c_scratch_l2_shape_balanced",
}
AGGREGATE_METRICS = (
    "point_global_relative_rmse_pct",
    "sample_first_cv_relative_rmse_pct",
    "raw_cv_weighted_rmse_K",
    "amplitude_ratio",
    "spatial_correlation",
    "hotspot_cv_weighted_rmse_K",
    "top5_cv_weighted_rmse_K",
    "strong_q_cv_weighted_rmse_K",
    "low_deltaT_background_bias_K",
    "low_deltaT_background_rmse_K",
    "low_deltaT_background_over_ratio",
    "shape_cv_rmse",
    "scale_log_rmse",
    "legacy_normalized_valid_base_mse",
)
NATIVE_METRICS = (
    "joint_relative_rmse_pct",
    "oracle_scale_relative_rmse_pct",
    "oracle_shape_relative_rmse_pct",
    "physics_scale_relative_rmse_pct",
)


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--collector-registry-commit", default=COLLECTOR_REGISTRY_COMMIT)
    parser.add_argument("--l1", type=Path, default=DEFAULT_L1)
    parser.add_argument("--l2", type=Path, default=DEFAULT_L2)
    parser.add_argument("--equivalence-json", type=Path, default=DEFAULT_EQ_JSON)
    parser.add_argument("--equivalence-md", type=Path, default=DEFAULT_EQ_MD)
    parser.add_argument("--metrics-md", type=Path, default=DEFAULT_METRICS_MD)
    return parser.parse_args()


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: expected JSON object")
    return payload


def _collector_payloads(path: Path, commit: str) -> dict[str, dict[str, Any]]:
    csv.field_size_limit(sys.maxsize)
    relative = path.resolve().relative_to(ROOT)
    source = subprocess.run(
        ["git", "show", f"{commit}:{relative.as_posix()}"],
        cwd=ROOT, check=True, capture_output=True, text=True,
    ).stdout
    rows = {row["config_id"]: row for row in csv.DictReader(source.splitlines())}
    payloads = {}
    for config_id in CONFIGS.values():
        raw = rows[config_id]["result_v5_metrics_json"]
        if not raw:
            raise ValueError(f"collector JSON missing for {config_id}")
        payloads[config_id] = json.loads(raw)
    return payloads


def _numeric_leaves(value: Any, prefix: str = "") -> Iterable[tuple[str, float]]:
    if isinstance(value, bool) or value is None:
        return
    if isinstance(value, (int, float)):
        number = float(value)
        if not math.isfinite(number):
            raise ValueError(f"non-finite metric at {prefix}")
        yield prefix, number
        return
    if isinstance(value, dict):
        for key in sorted(value):
            child = f"{prefix}.{key}" if prefix else str(key)
            yield from _numeric_leaves(value[key], child)
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            yield from _numeric_leaves(item, f"{prefix}[{index}]")


def _comparison(frozen: dict[str, Any], collector: dict[str, Any]) -> dict[str, Any]:
    frozen_metrics = dict(_numeric_leaves(frozen["reports"]))
    collector_metrics = dict(_numeric_leaves(collector["reports"]))
    if set(frozen_metrics) != set(collector_metrics):
        missing = sorted(set(frozen_metrics) - set(collector_metrics))
        extra = sorted(set(collector_metrics) - set(frozen_metrics))
        raise ValueError(f"numeric metric path mismatch: missing={missing[:5]} extra={extra[:5]}")
    rows = []
    for path in sorted(frozen_metrics):
        authoritative = frozen_metrics[path]
        observed = collector_metrics[path]
        absolute = abs(observed - authoritative)
        scale = max(abs(authoritative), abs(observed), 1.0e-30)
        rows.append({
            "path": path,
            "frozen": authoritative,
            "collector": observed,
            "absolute_delta": absolute,
            "relative_delta": absolute / scale,
            "exact": observed == authoritative,
            "within_tolerance": math.isclose(observed, authoritative, rel_tol=1.0e-9, abs_tol=1.0e-12),
        })
    mismatches = [row for row in rows if not row["exact"]]
    tolerance_mismatches = [row for row in rows if not row["within_tolerance"]]
    worst = sorted(rows, key=lambda row: (row["relative_delta"], row["absolute_delta"]), reverse=True)[:25]
    aggregate_paths = []
    for row in rows:
        path = row["path"]
        if ".per_sample[" in path or ".film_modulation." in path:
            continue
        if any(path.endswith(f".{name}") for name in (*AGGREGATE_METRICS, *NATIVE_METRICS)):
            aggregate_paths.append(row)
    return {
        "numeric_paths_compared": len(rows),
        "exact_matches": len(rows) - len(mismatches),
        "non_exact_matches": len(mismatches),
        "within_tolerance_matches": len(rows) - len(tolerance_mismatches),
        "outside_tolerance_matches": len(tolerance_mismatches),
        "exact_equivalent": not mismatches,
        "tolerance_equivalent": not tolerance_mismatches,
        "max_absolute_delta": max(row["absolute_delta"] for row in rows),
        "max_relative_delta": max(row["relative_delta"] for row in rows),
        "aggregate_metric_differences": aggregate_paths,
        "worst_differences": worst,
    }


def _fmt(value: Any) -> str:
    if value is None:
        return "-"
    return f"{float(value):.8g}"


def _metric_rows(payloads: dict[str, dict[str, Any]]) -> list[str]:
    lines = [
        "# Gate 6D frozen Gate-5 evaluator results",
        "",
        f"权威 evaluator engine commit：`{FROZEN_COMMIT}`。best 为最低 `valid_base_mse`，final 为 epoch 600。",
        "test_iid 为 `legacy_observed_test`；hard roles 为 `observed_report_only`，均未参与训练或 checkpoint selection。",
        "",
    ]
    headers = ["model", "checkpoint", "role", *AGGREGATE_METRICS, *NATIVE_METRICS]
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("|" + "---|" * len(headers))
    for label, config_id in CONFIGS.items():
        payload = payloads[config_id]
        for kind in KINDS:
            for role in ROLES:
                report = payload["reports"][kind][role]
                native = report.get("native_shape_scale") or {}
                values = [label, f"{kind} e{payload['checkpoint_metadata'][kind]['epoch']}", role]
                values.extend(_fmt(report.get(name)) for name in AGGREGATE_METRICS)
                values.extend(_fmt(native.get(name)) for name in NATIVE_METRICS)
                lines.append("| " + " | ".join(values) + " |")
    lines.append("")
    return lines


def main() -> int:
    args = _args()
    frozen_list = [_read_json(args.l1), _read_json(args.l2)]
    frozen = {payload["config_id"]: payload for payload in frozen_list}
    collector = _collector_payloads(args.registry, args.collector_registry_commit)
    if set(frozen) != set(CONFIGS.values()):
        raise ValueError("frozen evaluator config set mismatch")
    comparisons = {}
    for config_id, payload in frozen.items():
        if payload["evaluator_git_commit"] != FROZEN_COMMIT:
            raise ValueError(f"{config_id}: evaluator commit is not frozen")
        comparisons[config_id] = _comparison(payload, collector[config_id])
    all_exact = all(item["exact_equivalent"] for item in comparisons.values())
    report = {
        "schema_version": "heat3d_v5_gate6d_evaluator_equivalence_v1",
        "frozen_evaluator_commit": FROZEN_COMMIT,
        "collector_registry_commit": args.collector_registry_commit,
        "collector_evaluator_commits": {
            config_id: collector[config_id]["evaluator_git_commit"] for config_id in CONFIGS.values()
        },
        "comparison_scope": "all numeric leaves under reports.best/final for five Gate-5 roles",
        "exact_equivalent": all_exact,
        "authoritative_source": "frozen_gate5_evaluator" if not all_exact else "equivalent",
        "comparisons": comparisons,
    }
    args.equivalence_json.parent.mkdir(parents=True, exist_ok=True)
    args.equivalence_md.parent.mkdir(parents=True, exist_ok=True)
    args.equivalence_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    md = [
        "# Gate 6D evaluator equivalence",
        "",
        f"冻结 evaluator commit：`{FROZEN_COMMIT}`。比较范围为 best/final、五个 roles 下全部 numeric metric leaves。",
        "",
        "| config | numeric paths | exact | non-exact | outside 1e-9/1e-12 tolerance | max abs delta | max relative delta |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for label, config_id in CONFIGS.items():
        item = comparisons[config_id]
        md.append(
            f"| {label} | {item['numeric_paths_compared']} | {item['exact_matches']} | "
            f"{item['non_exact_matches']} | {item['outside_tolerance_matches']} | "
            f"{item['max_absolute_delta']:.8g} | {item['max_relative_delta']:.8g} |"
        )
    md.extend([
        "",
        "结论：" + (
            "两次计算逐数值完全一致。" if all_exact else
            "存在非零差异；按 Gate 6D 合同冻结 `639872ab` 结果为权威值，collector 旧值只保留作 provenance 对照。"
        ),
        "",
        "每个配置按 relative delta 排序的前 25 个差异保存在 machine-readable JSON。",
        "",
    ])
    args.equivalence_md.write_text("\n".join(md), encoding="utf-8")
    args.metrics_md.write_text("\n".join(_metric_rows(frozen)), encoding="utf-8")
    print(json.dumps({
        "status": "passed",
        "exact_equivalent": all_exact,
        "authoritative_source": report["authoritative_source"],
        "equivalence_json": str(args.equivalence_json),
        "equivalence_md": str(args.equivalence_md),
        "metrics_md": str(args.metrics_md),
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
