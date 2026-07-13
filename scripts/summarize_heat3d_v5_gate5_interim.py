#!/usr/bin/env python3
"""Build the concise tracked Gate-5 interim closeout from unified metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


CONFIGS = {
    "B0": "V4P5_04_local_bypass_global_film",
    "N0": "V4P5_05_native_physics_only",
    "N1": "V4P5_06_native_pooled_latent",
}
ROLES = (
    "valid_iid",
    "test_iid",
    "hard_train_holdout",
    "hard_challenge_valid",
    "hard_challenge_test",
)
METRICS = (
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


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    for label in CONFIGS:
        parser.add_argument(f"--{label.lower()}-metrics", type=Path, required=True)
    parser.add_argument("--n3-smoke-summary", type=Path, required=True)
    parser.add_argument("--n3-launch-host", required=True)
    parser.add_argument("--n3-launcher-pid", type=int, required=True)
    parser.add_argument("--n3-training-pid", type=int, required=True)
    parser.add_argument("--n3-launch-commit", required=True)
    parser.add_argument("--n3-tmux-session", required=True)
    parser.add_argument("--n3-output-dir", required=True)
    parser.add_argument("--n3-log-path", required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    return parser.parse_args()


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected object")
    return value


def _compact_report(payload: dict[str, Any], checkpoint: str, role: str) -> dict[str, Any]:
    source = payload["reports"][checkpoint][role]
    report = {field: source[field] for field in METRICS}
    if "native_shape_scale" in source:
        report["native_shape_scale"] = source["native_shape_scale"]
    return report


def _oracle(payloads: dict[str, dict[str, Any]]) -> dict[str, Any]:
    rows = []
    for label in ("N0", "N1"):
        native = payloads[label]["reports"]["best"]["valid_iid"]["native_shape_scale"]
        joint = float(native["joint_relative_rmse_pct"])
        scale_gain = joint - float(native["oracle_scale_relative_rmse_pct"])
        shape_gain = joint - float(native["oracle_shape_relative_rmse_pct"])
        rows.append({
            "config": label,
            "joint_relative_rmse_pct": joint,
            "oracle_scale_relative_rmse_pct": float(native["oracle_scale_relative_rmse_pct"]),
            "oracle_shape_relative_rmse_pct": float(native["oracle_shape_relative_rmse_pct"]),
            "physics_scale_relative_rmse_pct": float(native["physics_scale_relative_rmse_pct"]),
            "scale_replacement_gain_pct_points": scale_gain,
            "shape_replacement_gain_pct_points": shape_gain,
            "bottleneck": "scale_dominant" if scale_gain > shape_gain else "shape_dominant",
        })
    mean_scale_gain = sum(row["scale_replacement_gain_pct_points"] for row in rows) / len(rows)
    mean_shape_gain = sum(row["shape_replacement_gain_pct_points"] for row in rows) / len(rows)
    row_bottlenecks = {row["bottleneck"] for row in rows}
    bottleneck = (
        next(iter(row_bottlenecks))
        if len(row_bottlenecks) == 1
        else "mixed_model_dependent"
    )
    return {
        "basis": "MSE-best valid_iid; replacing one predicted component with its target oracle",
        "classification": bottleneck,
        "mean_scale_replacement_gain_pct_points": mean_scale_gain,
        "mean_shape_replacement_gain_pct_points": mean_shape_gain,
        "rows": rows,
        "caveat": "Oracle gaps diagnose removable component error; they are not standalone deployable performance.",
    }


def _smoke(payload: dict[str, Any]) -> dict[str, Any]:
    split_counts = payload.get("split_counts") or {}
    context = payload.get("global_context") or {}
    standardizer = context.get("standardizer") or context
    reload_audit = payload.get("checkpoint_prediction_reload_audit") or {}
    architecture = payload.get("native_runtime_architecture_audit") or {}
    checks = {
        "epoch_1_complete": int(payload.get("final_epoch", -1)) == 1,
        "train_672_valid_128": split_counts.get("train") == 672 and split_counts.get("valid_iid") == 128,
        "batch_28": int(payload.get("batch_size", -1)) == 28,
        "nodes_1024": str(payload.get("subset", "")).endswith("heat3d_v4_p5_clean_nohard_v0"),
        "random_initialization": not bool(payload.get("checkpoint_loaded")),
        "checkpoint_saved": bool(payload.get("checkpoint_saved")),
        "reload_prediction_match": bool(reload_audit.get("passed", reload_audit.get("status") == "passed")),
        "train_only_context_fit": (
            standardizer.get("fit_population") == "train_only"
            and int(standardizer.get("fit_sample_count", -1)) == 672
        ),
        "finite": all(
            value is None or (isinstance(value, (int, float)) and value == value)
            for value in (
                payload.get("final_valid_base_mse"),
                payload.get("final_valid_iid_loss"),
                payload.get("final_valid_iid_raw_deltaT_rmse_K"),
            )
        ),
    }
    return {
        "status": "passed" if all(checks.values()) else "failed",
        "checks": checks,
        "git_commit": payload.get("code_version_or_git_commit"),
        "output_dir": payload.get("output_dir"),
        "final_valid_base_mse": payload.get("final_valid_base_mse"),
        "checkpoint_reload_audit": reload_audit,
        "global_context": context,
        "native_runtime_architecture_audit": architecture,
    }


def _fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def _markdown(closeout: dict[str, Any]) -> str:
    lines = [
        "# V5 Gate 5 interim closeout",
        "",
        "统一 evaluator 使用 true-RMS 分母：`100 * sqrt(sum(error^2) / sum(true^2))`。",
        "`test_iid` 与全部 hard roles 仅报告，不参与训练、标准化、超参数或 checkpoint 选择。",
        "",
        "## MSE-best / final clean summary",
        "",
        "| Run | Checkpoint (epoch) | Role | point-global rel RMSE % | sample-first CV rel RMSE % | raw CV RMSE K | amp | corr | hotspot K | top5 K | strong-q K | bg bias K | bg RMSE K | bg over | shape CV-RMSE | scale log-RMSE | legacy MSE |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    fields = METRICS
    for label, run in closeout["runs"].items():
        for checkpoint in ("best", "final"):
            epoch = run["checkpoint_epochs"][checkpoint]
            for role in ("valid_iid", "test_iid"):
                row = run["reports"][checkpoint][role]
                lines.append(
                    "| " + " | ".join(
                        [label, f"{checkpoint} (e{epoch})", role]
                        + [_fmt(row[field]) for field in fields]
                    ) + " |"
                )
    lines += [
        "",
        "可信模型门槛为 valid/test point-global true-RMS relative RMSE 均 `<20%`；"
        + "、".join(
            f"{label}={'pass' if value else 'fail'}"
            for label, value in closeout["threshold_assessment"]["mse_best_pass"].items()
        )
        + "。",
        "",
        "## MSE-best hard report-only summary",
        "",
        "| Run | Role | point-global rel RMSE % | sample-first CV rel RMSE % | raw CV RMSE K |",
        "|---|---|---:|---:|---:|",
    ]
    for label, run in closeout["runs"].items():
        for role in ROLES[2:]:
            row = run["reports"]["best"][role]
            lines.append(
                f"| {label} | {role} | {_fmt(row['point_global_relative_rmse_pct'])} | "
                f"{_fmt(row['sample_first_cv_relative_rmse_pct'])} | {_fmt(row['raw_cv_weighted_rmse_K'])} |"
            )
    lines += [
        "",
        "## Native oracle bottleneck",
        "",
        "| Run | joint % | oracle-scale % | oracle-shape % | physics-scale % | scale replacement gain pp | shape replacement gain pp | bottleneck |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in closeout["oracle_bottleneck"]["rows"]:
        lines.append(
            "| " + " | ".join(_fmt(row[field]) for field in (
                "config", "joint_relative_rmse_pct", "oracle_scale_relative_rmse_pct",
                "oracle_shape_relative_rmse_pct", "physics_scale_relative_rmse_pct",
                "scale_replacement_gain_pct_points", "shape_replacement_gain_pct_points",
                "bottleneck",
            )) + " |"
        )
    lines += [
        "",
        f"判断：`{closeout['oracle_bottleneck']['classification']}`。",
        "",
        "## N3 execution smoke",
        "",
        f"状态：`{closeout['n3_smoke']['status']}`；commit `{closeout['n3_smoke']['git_commit']}`；"
        f"output `{closeout['n3_smoke']['output_dir']}`。",
        "",
        "## N3 e600 launch",
        "",
        f"状态：`{closeout['n3_launch']['status']}`；host `{closeout['n3_launch']['host']}`；"
        f"training PID `{closeout['n3_launch']['training_pid']}`；commit `{closeout['n3_launch']['commit']}`；"
        f"tmux `{closeout['n3_launch']['tmux_session']}`。",
        "",
        f"output `{closeout['n3_launch']['output_dir']}`；log `{closeout['n3_launch']['log_path']}`。",
        "",
        "完整公式、split hash、checkpoint SHA/epoch 和每个 role 的全指标保存在各远程 run 的 `v5_metrics.json`，并镜像到 V5 registry result payload。",
    ]
    return "\n".join(lines) + "\n"


def main() -> int:
    args = _args()
    payloads = {
        label: _read(getattr(args, f"{label.lower()}_metrics"))
        for label in CONFIGS
    }
    runs = {}
    for label, payload in payloads.items():
        assert payload["config_id"] == CONFIGS[label]
        runs[label] = {
            "config_id": CONFIGS[label],
            "evaluator_git_commit": payload["evaluator_git_commit"],
            "training_git_commit": payload.get("training_git_commit"),
            "metric_schema_version": payload["metric_schema_version"],
            "formulas": payload["formulas"],
            "data": payload["data"],
            "checkpoint_epochs": {
                name: payload["checkpoint_metadata"][name]["epoch"]
                for name in ("best", "final")
            },
            "reports": {
                checkpoint: {
                    role: _compact_report(payload, checkpoint, role)
                    for role in ROLES
                }
                for checkpoint in ("best", "final")
            },
        }
    threshold = {
        label: (
            float(run["reports"]["best"]["valid_iid"]["point_global_relative_rmse_pct"]) < 20.0
            and float(run["reports"]["best"]["test_iid"]["point_global_relative_rmse_pct"]) < 20.0
        )
        for label, run in runs.items()
    }
    closeout = {
        "schema_version": "heat3d_v5_gate5_interim_closeout_v1",
        "status": "evaluation_complete_n3_e600_running",
        "selection": "MSE-best means lowest valid_base_mse; final means epoch 600",
        "report_only_roles": list(ROLES[1:]),
        "runs": runs,
        "threshold_assessment": {
            "formula": "MSE-best valid_iid and test_iid point-global true-RMS relative RMSE both <20%",
            "mse_best_pass": threshold,
        },
        "oracle_bottleneck": _oracle(payloads),
        "n3_smoke": _smoke(_read(args.n3_smoke_summary)),
        "n3_launch": {
            "status": "running_e600",
            "host": args.n3_launch_host,
            "launcher_pid": args.n3_launcher_pid,
            "training_pid": args.n3_training_pid,
            "commit": args.n3_launch_commit,
            "tmux_session": args.n3_tmux_session,
            "output_dir": args.n3_output_dir,
            "log_path": args.n3_log_path,
            "initialization": "random",
            "init_checkpoint": None,
        },
    }
    if closeout["n3_smoke"]["status"] != "passed":
        raise SystemExit("N3 smoke summary does not satisfy the frozen execution checks")
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(closeout, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.output_md.write_text(_markdown(closeout), encoding="utf-8")
    print(json.dumps({
        "status": "passed",
        "oracle_bottleneck": closeout["oracle_bottleneck"]["classification"],
        "output_json": str(args.output_json),
        "output_md": str(args.output_md),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
