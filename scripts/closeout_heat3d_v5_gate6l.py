#!/usr/bin/env python3
"""Freeze Gate 6L valid-only results into the V5 registry and Markdown."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
RESULT = ROOT / "configs/heat3d_v5/gate6l/gate6l_valid_only_evaluation.json"
CHECKPOINT_CSV = ROOT / "configs/heat3d_v5/gate6l/gate6l_checkpoint_comparison.csv"
BOOTSTRAP_CSV = ROOT / "configs/heat3d_v5/gate6l/gate6l_paired_bootstrap.csv"
REGISTRY = ROOT / "configs/heat3d_v5/v5_gate6k_v32_single_variable_registry.csv"
REPORT = ROOT / "docs/v5_gate6l_valid_only_closeout.md"

MODEL_BY_ID = {
    "V4P5_33_gate6k_o075_log_scale": "O075",
    "V4P5_34_gate6k_dual_physics_attention": "Dual",
}
NEW_FIELDS = (
    "gate6l_status",
    "gate6l_evaluator_commit",
    "gate6l_training_commit",
    "gate6l_point_global_best_epoch",
    "gate6l_sample_first_best_epoch",
    "gate6l_base_mse_best_epoch",
    "gate6l_final_epoch",
    "gate6l_checkpoint_sha256_json",
    "gate6l_reload_audit_json",
    "gate6l_result_json",
    "gate6l_roles_accessed",
    "gate6l_no_auto_advancement",
)


def _compact(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _metric(metrics: dict[str, Any], key: str) -> float:
    return float(metrics["summary"][key])


def _write_registry(payload: dict[str, Any]) -> None:
    with REGISTRY.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        fields = list(reader.fieldnames or [])
    for field in NEW_FIELDS:
        if field not in fields:
            fields.append(field)

    for row in rows:
        model_name = MODEL_BY_ID[row["config_id"]]
        model = payload["models"][model_name]
        metadata = model["checkpoint_metadata"]
        row.update(
            {
                "plan_status": "completed",
                "execution_status": "completed_e600",
                "evaluation_status": "completed_valid_iid_four_checkpoint",
                "training_started": "true",
                "test_accessed": "false",
                "hard_accessed": "false",
                "sealed_iid_accessed": "false",
                "gate6k_notes": (
                    "Gate 6L valid_iid four-checkpoint closeout complete; "
                    "no automatic advancement; test/hard/sealed not accessed"
                ),
                "gate6l_status": "completed_valid_iid_four_checkpoint",
                "gate6l_evaluator_commit": payload["evaluator_commit"],
                "gate6l_training_commit": model["training_commit"],
                "gate6l_point_global_best_epoch": str(
                    metadata["point_global_best"]["epoch"]
                ),
                "gate6l_sample_first_best_epoch": str(
                    metadata["sample_first_best"]["epoch"]
                ),
                "gate6l_base_mse_best_epoch": str(
                    metadata["legacy_best"]["epoch"]
                ),
                "gate6l_final_epoch": str(metadata["final"]["epoch"]),
                "gate6l_checkpoint_sha256_json": _compact(
                    {
                        checkpoint: entry["sha256"]
                        for checkpoint, entry in metadata.items()
                    }
                ),
                "gate6l_reload_audit_json": _compact(model["reload_audit"]),
                "gate6l_result_json": str(RESULT.relative_to(ROOT)),
                "gate6l_roles_accessed": "train|valid_iid",
                "gate6l_no_auto_advancement": "true",
            }
        )

    with REGISTRY.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _write_report(payload: dict[str, Any]) -> None:
    checkpoint_rows = list(
        csv.DictReader(CHECKPOINT_CSV.open(encoding="utf-8", newline=""))
    )
    bootstrap_rows = list(
        csv.DictReader(BOOTSTRAP_CSV.open(encoding="utf-8", newline=""))
    )
    lines = [
        "# Gate 6L valid-only frozen closeout",
        "",
        "状态：`completed_valid_iid_four_checkpoint`。本轮只重放既有 checkpoint，"
        "没有训练、改参或重新选择 checkpoint；评估角色仅 `valid_iid`，"
        "`test/hard/sealed` 均未访问，也没有自动晋级。",
        "",
        f"- evaluator commit: `{payload['evaluator_commit']}`",
        f"- training commit: `{payload['models']['O075']['training_commit']}`",
        "- 统一公式：`heat3d_v5_clean_metrics_v2_true_rms`",
        "- 样本：128，节点/样本：1024；normalization/global context 仅由 train=672 拟合",
        "",
        "## 四 checkpoint 统一结果",
        "",
        "| 模型 | checkpoint | epoch | point-global % | sample-first % | raw CV K | "
        "shape CV | scale log | amplitude | correlation |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in checkpoint_rows:
        lines.append(
            f"| {row['model']} | {row['checkpoint']} | {row['epoch']} | "
            f"{float(row['point_global_relative_rmse_pct']):.4f} | "
            f"{float(row['sample_first_cv_relative_rmse_pct']):.4f} | "
            f"{float(row['raw_cv_weighted_rmse_K']):.6f} | "
            f"{float(row['shape_cv_rmse']):.6f} | "
            f"{float(row['scale_log_rmse']):.6f} | "
            f"{float(row['amplitude_ratio']):.6f} | "
            f"{float(row['spatial_correlation']):.6f} |"
        )

    lines += [
        "",
        "完整的 hotspot/top-5/strong-q、low-ΔT、legacy MSE、SHA256、参数量与 "
        "reload 结果见 JSON/CSV 工件。",
        "",
        "## Point-global-best 的冻结分层",
        "",
        "| 模型 | 分层 | n | point-global % | sample-first % | raw CV K | "
        "shape CV | scale log |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    strata_order = (
        "Q1",
        "Q2",
        "Q3",
        "Q4",
        "nominal_to_hard",
        "Q2_intersection_nominal_to_hard",
        "scale_abs_error_top10pct",
        "scale_signed_low_p10",
        "scale_signed_central_p10_p90",
        "scale_signed_high_p90",
    )
    for model_name in ("O075", "Dual"):
        reports = payload["models"][model_name]["strata"]["point_global_best"][
            "reports"
        ]
        for stratum in strata_order:
            report = reports[stratum]
            metrics = report["metrics"]
            lines.append(
                f"| {model_name} | {stratum} | {report['sample_count']} | "
                f"{float(metrics['point_global_relative_rmse_pct']):.4f} | "
                f"{float(metrics['sample_first_cv_relative_rmse_pct']):.4f} | "
                f"{float(metrics['raw_cv_weighted_rmse_K']):.6f} | "
                f"{float(metrics['shape_cv_rmse']):.6f} | "
                f"{float(metrics['scale_log_rmse']):.6f} |"
            )

    lines += [
        "",
        "## 逐样本配对结论",
        "",
        "差值方向统一为右模型减左模型；误差指标中负值表示右模型改善。"
        "CI 为固定 seed、20,000 次 paired bootstrap 的 95% 区间。",
        "",
        "| 对比 | 指标 | 差值 | 95% CI | right 改善概率 | win rate | 中位差 |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in bootstrap_rows:
        if row["metric"] not in {
            "point_global_relative_rmse_pct",
            "sample_first_cv_relative_rmse_pct",
            "raw_cv_weighted_rmse_K",
            "shape_cv_rmse",
            "scale_log_rmse",
        }:
            continue
        lines.append(
            f"| {row['pair']} | {row['metric']} | "
            f"{float(row['observed_difference']):.6f} | "
            f"[{float(row['ci95_low']):.6f}, {float(row['ci95_high']):.6f}] | "
            f"{float(row['probability_right_improves']):.4f} | "
            f"{float(row['win_rate']):.4f} | "
            f"{float(row['median_difference']):.6f} |"
        )

    lines += [
        "",
        "Tail contribution 的逐样本、Q4 与 scale-tail 贡献保存在主 JSON 和 "
        "`gate6l_paired_samples.csv`。",
        "",
        "## 冻结判断",
        "",
        "- O075 相对 V32 改善 sample-first 与 scale，但 point-global、raw CV 和 shape 退化。",
        "- Dual 相对 V32 的 point-global/raw 更接近，但 shape 与 sample-first 退化；"
        "相对 O075 虽恢复部分 Q4 point-global SSE，却显著损失 sample-first 与 scale。",
        "- 按冻结的 point-global 唯一晋级准则，O075 与 Dual 均不自动晋级；"
        "V32 保持当前候选地位。本结论不触发新实验。",
        "",
    ]
    REPORT.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    payload = json.loads(RESULT.read_text(encoding="utf-8"))
    if payload["status"] != "completed_valid_iid_only":
        raise ValueError("Gate 6L evaluator is not complete")
    _write_registry(payload)
    _write_report(payload)
    print(
        json.dumps(
            {
                "status": "completed",
                "registry": str(REGISTRY.relative_to(ROOT)),
                "report": str(REPORT.relative_to(ROOT)),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
