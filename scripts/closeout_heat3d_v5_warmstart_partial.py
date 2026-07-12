#!/usr/bin/env python3
"""Build the tracked partial closeout for the interrupted V5 warm-start run."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


RUN_ID = "V5WS_clean_short_e12_seed20260712_r2"
COMPLETED = ("v4_global_film_legacy_target", "native_shape_scale")
INCOMPLETE = "native_shape_scale_global_film"
SUMMARY_KEYS = (
    "point_global_relative_rmse_pct",
    "sample_first_cv_relative_rmse_pct",
    "raw_cv_weighted_rmse_K",
    "amplitude_ratio",
    "spatial_correlation",
    "hotspot_rmse_K",
    "top5_rmse_K",
    "strong_q_rmse_K",
    "low_deltaT_background_bias_K",
    "low_deltaT_background_rmse_K",
    "low_deltaT_background_over_ratio",
    "shape_cv_rmse",
    "scale_log_rmse",
    "valid_base_mse",
)


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    return parser.parse_args()


def _load(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected mapping: {path}")
    return payload


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _summary(payload: dict) -> dict:
    return {key: payload.get(key) for key in SUMMARY_KEYS if key in payload}


def _history_row(row: dict) -> dict:
    return {
        "epoch": row["epoch"],
        "train_loss": row.get("train_loss"),
        "gradient_norm_mean": row.get("gradient_norm_mean"),
        "gradient_norm_max": row.get("gradient_norm_max"),
        "train_components": row.get("train_components", {}),
        "valid": _summary(row.get("valid_summary", {})),
    }


def _log_events(path: Path) -> list[dict]:
    events = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(item, dict):
            events.append(item)
    return events


def _markdown(payload: dict) -> str:
    lines = [
        f"# {RUN_ID} partial closeout",
        "",
        "## 结论",
        "",
        "- 状态：`incomplete_closeout`。",
        "- 已完成：`v4_global_film_legacy_target`、`native_shape_scale`。",
        "- 未完成：`native_shape_scale_global_film`；中断原因为 `KeyboardInterrupt`。",
        "- 性能结论：`inconclusive`；`scratch_yaml_allowed: false`。",
        "- 不恢复第三个变体；现有观察不得解释为否定 Global FiLM 或 shape-scale。",
        "",
        "## 冻结基线与已完成结果",
        "",
        "| variant | primary epoch | valid sample-first CV-relative % | valid point-global relative % | raw CV-RMSE K |",
        "|---|---:|---:|---:|---:|",
    ]
    baseline = payload["baseline_valid"]
    lines.append(
        "| V4P5_02 epoch 405 | - | {sample:.6f} | {point:.6f} | {raw:.9f} |".format(
            sample=baseline["sample_first_cv_relative_rmse_pct"],
            point=baseline["point_global_relative_rmse_pct"],
            raw=baseline["raw_cv_weighted_rmse_K"],
        )
    )
    for name in COMPLETED:
        item = payload["variants"][name]
        valid = item["primary_valid"]
        lines.append(
            f"| {name} | {item['primary_relative_epoch']} | "
            f"{valid['sample_first_cv_relative_rmse_pct']:.6f} | "
            f"{valid['point_global_relative_rmse_pct']:.6f} | "
            f"{valid['raw_cv_weighted_rmse_K']:.9f} |"
        )
    lines += ["", "以上仅整理已产生的观测，不构成架构判定。", ""]
    for name in (*COMPLETED, INCOMPLETE):
        item = payload["variants"][name]
        lines += [f"## Epoch history: `{name}`", ""]
        lines += [
            "| epoch | train loss | grad mean | valid sample-first % | valid point-global % | raw CV-RMSE K |",
            "|---:|---:|---:|---:|---:|---:|",
        ]
        for row in item["history"]:
            valid = row["valid"]
            lines.append(
                "| {epoch} | {loss:.9g} | {grad:.9g} | {sample:.6f} | {point:.6f} | {raw:.9f} |".format(
                    epoch=row["epoch"], loss=row["train_loss"],
                    grad=row["gradient_norm_mean"],
                    sample=valid["sample_first_cv_relative_rmse_pct"],
                    point=valid["point_global_relative_rmse_pct"],
                    raw=valid["raw_cv_weighted_rmse_K"],
                )
            )
        lines.append("")
    lines += [
        "## 数据角色与解释边界",
        "",
        "训练仅使用 `train`，选择仅使用 `valid_iid`。`test_iid` 与 hard roles 仅为报告，未用于训练、标准化或选择。第三个变体没有 closeout artifacts；表中仅保留日志已经确认完成的 epoch 1–9。",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    args = _args()
    events = _log_events(args.log)
    baseline_event = next(item for item in events if item.get("event") == "baseline_evaluated")
    payload = {
        "schema_version": "heat3d_v5_warmstart_partial_closeout_v1",
        "run_id": RUN_ID,
        "status": "incomplete_closeout",
        "completed_variants": list(COMPLETED),
        "incomplete_variants": [INCOMPLETE],
        "interruption_reason": "KeyboardInterrupt",
        "performance_conclusion": "inconclusive",
        "scratch_yaml_allowed": False,
        "resume_incomplete_variant": False,
        "interpretation_guardrail": (
            "Existing observations must not be interpreted as rejecting Global FiLM or native shape-scale."
        ),
        "baseline": {
            "config_id": "V4P5_02_clean_baseline_raw_B28_e600",
            "checkpoint_kind": "best",
            "epoch": 405,
        },
        "baseline_valid": {
            "sample_first_cv_relative_rmse_pct": baseline_event["valid_sample_first_cv_relative_rmse_pct"],
            "point_global_relative_rmse_pct": baseline_event["valid_point_global_relative_rmse_pct"],
            "raw_cv_weighted_rmse_K": baseline_event["valid_raw_cv_weighted_rmse_K"],
        },
        "fit_roles": ["train"],
        "selection_roles": ["valid_iid"],
        "report_only_roles": [
            "test_iid", "hard_train_holdout", "hard_challenge_valid", "hard_challenge_test"
        ],
        "variants": {},
        "sources": {"remote_log": {"path": str(args.log), "sha256": _sha256(args.log)}},
    }
    for name in COMPLETED:
        root = args.artifact_root / name
        loss_path = root / "loss_summary.json"
        metrics_path = root / "clean_metrics.json"
        provenance_path = root / "provenance.json"
        loss = _load(loss_path)
        metrics = _load(metrics_path)
        provenance = _load(provenance_path)
        epoch = int(loss["primary_relative_epoch"])
        primary = next(row for row in loss["history"] if int(row["epoch"]) == epoch)
        payload["variants"][name] = {
            "status": "completed",
            "epochs_completed": len(loss["history"]),
            "primary_relative_epoch": epoch,
            "legacy_metric_epoch": int(loss["legacy_metric_epoch"]),
            "primary_valid": _summary(primary["valid_summary"]),
            "final_valid": _summary(loss["final_valid"]),
            "checkpoint_load": loss.get("checkpoint_load", {}),
            "history": [_history_row(row) for row in loss["history"]],
            "reports": {role: _summary(report) for role, report in metrics["reports"].items()},
            "observational_gate": provenance.get("gate", {}),
            "target_or_label_derived_model_inputs": provenance.get(
                "target_or_label_derived_model_inputs"
            ),
        }
        payload["sources"][name] = {
            path.name: {"sha256": _sha256(path)}
            for path in (loss_path, metrics_path, provenance_path)
        }
    interrupted = [
        item for item in events
        if item.get("event") == "epoch_complete" and item.get("variant") == INCOMPLETE
    ]
    payload["variants"][INCOMPLETE] = {
        "status": "incomplete",
        "epochs_completed": len(interrupted),
        "interruption_reason": "KeyboardInterrupt",
        "artifacts_complete": False,
        "history": [
            {
                "epoch": item["epoch"],
                "train_loss": item["train_loss"],
                "gradient_norm_mean": item["gradient_norm_mean"],
                "gradient_norm_max": None,
                "train_components": {},
                "valid": {
                    "sample_first_cv_relative_rmse_pct": item["valid_sample_first_cv_relative_rmse_pct"],
                    "point_global_relative_rmse_pct": item["valid_point_global_relative_rmse_pct"],
                    "raw_cv_weighted_rmse_K": item["valid_raw_cv_weighted_rmse_K"],
                },
            }
            for item in interrupted
        ],
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    args.output_md.write_text(_markdown(payload), encoding="utf-8")
    print(json.dumps({"status": "ok", "epochs": [12, 12, len(interrupted)]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
