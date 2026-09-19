#!/usr/bin/env python3
"""Aggregate the completed V7 e600 native1024 replication receipts.

This script consumes only the three valid_iid training receipts.  It does not
load predictions, test_iid, sealed, or any DeepOHeat official-test artifact.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs" / "heat3d_v7" / "v7_e600_budget_replication.json"
SUMMARY = ROOT / "docs" / "results" / "v7_model_performance_summary.json"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate V7 e600 budget replication")
    parser.add_argument("--output-root", type=Path, default=ROOT / "output" / "heat3d_v7_e600_budget")
    parser.add_argument("--contract", type=Path, default=CONTRACT)
    parser.add_argument("--summary", type=Path, default=SUMMARY)
    parser.add_argument("--output-json", type=Path, default=ROOT / "docs" / "results" / "v7_e600_budget_replication_aggregation.json")
    parser.add_argument("--output-md", type=Path, default=ROOT / "docs" / "v7_e600_budget_replication_report.md")
    return parser.parse_args()


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected JSON object")
    return value


def _mean_sd(values: list[float]) -> dict[str, float]:
    if len(values) != 3:
        raise ValueError("aggregation requires exactly three seeds")
    return {"mean": float(np.mean(values)), "sd_across_training_seeds": float(np.std(values, ddof=1))}


def _reference(summary: dict[str, Any], section: str, model_prefix: str) -> dict[str, Any]:
    rows = summary["sections"][section]
    for row in rows:
        if str(row["model"]).startswith(model_prefix):
            return row
    raise KeyError(model_prefix)


def run(args: argparse.Namespace) -> dict[str, Any]:
    contract = _load(args.contract.resolve())
    if contract.get("status") != "frozen_prelaunch":
        raise ValueError("aggregation contract status changed")
    receipts = []
    for seed in (0, 1, 2):
        receipt_path = args.output_root.resolve() / f"seed{seed}" / "v7_e600_budget_replication_receipt.json"
        receipt = _load(receipt_path)
        if receipt.get("status") != "COMPLETE" or int(receipt.get("seed", -1)) != seed:
            raise ValueError(f"seed{seed}: receipt is not COMPLETE")
        if receipt.get("contract_sha256") != _sha256(args.contract.resolve()):
            raise ValueError(f"seed{seed}: contract SHA mismatch")
        if receipt.get("test_iid_access") or receipt.get("sealed_access"):
            raise ValueError(f"seed{seed}: forbidden split access recorded")
        receipts.append(receipt)
    selected = [r["metrics_at_selected_checkpoint"] for r in receipts]
    keys = [
        "point_global_relative_rmse_pct",
        "sample_first_relative_rmse_pct",
        "rmse_K",
        "mae_K",
        "corr_cv_pct",
        "amp_cvrms_pct",
        "corr_point_pct",
        "amp_range_pct",
    ]
    aggregate = {
        key: _mean_sd([float(row[key]) for row in selected]) for key in keys
    }
    summary = _load(args.summary.resolve())
    e200 = _reference(summary, "A_p1i_native1024", "Heat3D V7 P1i-e200")
    v6 = _reference(summary, "A_p1i_native1024", "Heat3D V6")
    payload: dict[str, Any] = {
        "schema_version": "heat3d_v7_e600_budget_replication_aggregation_v1",
        "status": "COMPLETE_VALID_IID_NATIVE1024_ONLY",
        "experiment_id": contract["experiment_id"],
        "contract_path": str(args.contract.resolve().relative_to(ROOT)),
        "contract_sha256": _sha256(args.contract.resolve()),
        "dataset": contract["dataset"],
        "seeds": [0, 1, 2],
        "dispersion_label": "SD across training seeds",
        "checkpoint_selection": "sample_first_relative_rmse_pct; tie=earliest epoch",
        "per_seed": [
            {
                "seed": int(receipt["seed"]),
                "best_epoch": receipt["checkpoint_selection"]["best_epoch"],
                "point_global_best_epoch": receipt["checkpoint_selection"]["point_global_best_epoch"],
                "training_wall_seconds": receipt["training_wall_seconds"],
                "peak_host_rss_bytes": receipt["host_max_rss_bytes"],
                "metrics_at_selected_checkpoint": receipt["metrics_at_selected_checkpoint"],
                "metrics_at_point_global_checkpoint": receipt["metrics_at_point_global_checkpoint"],
                "metrics_at_final_checkpoint": receipt["metrics_at_final_checkpoint"],
                "checkpoint_sha256": {
                    "primary": receipt["checkpoint_selection"]["primary_checkpoint"]["sha256"],
                    "point_global": receipt["checkpoint_selection"]["point_global_checkpoint"]["sha256"],
                    "final": receipt["checkpoint_selection"]["final_checkpoint"]["sha256"],
                },
            }
            for receipt in receipts
        ],
        "aggregate_at_primary_selected_checkpoint": aggregate,
        "references": {
            "v7_e200": e200,
            "v6_e600_trained_point_global_best": v6,
        },
        "comparison_policy": {
            "v7_e600_is_independent_replication": True,
            "does_not_modify_g2_or_v6_evidence": True,
            "u_v2_full_field_evaluation": False,
            "test_iid_access": False,
            "sealed_access": False,
            "deepoheat_official100_access": False,
        },
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = [
        "# V7 P1i Full e600 budget replication",
        "",
        "状态：`COMPLETE_VALID_IID_NATIVE1024_ONLY`。这是独立的 V7 budget study，不修改 G2/V6 frozen evidence。",
        "",
        "- Dataset: `heat3d_v6_p1i_continuous_physics1024_v1`",
        "- Split: train768 / valid_iid128；test_iid、sealed、DeepOHeat official100 均未访问",
        "- Contract: 600 epochs, cosine horizon 600, B24, AdamW 5e-4 → 5e-5, fresh random initialization",
        "- Primary selection: `sample_first_relative_rmse_pct`, tie-break earliest epoch",
        "- U-v2 full-field evaluation: deferred",
        "",
        "## Native1024 selected-checkpoint metrics",
        "",
        "| Metric | V7 e600 mean | SD across training seeds |",
        "|---|---:|---:|",
    ]
    labels = {
        "point_global_relative_rmse_pct": "point-global relative RMSE [%]",
        "sample_first_relative_rmse_pct": "sample-first relative RMSE [%]",
        "rmse_K": "unweighted node RMSE [K]",
        "mae_K": "unweighted node MAE [K]",
        "corr_cv_pct": "Corr_CV [%]",
        "amp_cvrms_pct": "Amp_CVRMS [%]",
        "corr_point_pct": "Corr_point [%]",
        "amp_range_pct": "Amp_range [%]",
    }
    for key in keys:
        lines.append(f"| {labels[key]} | {aggregate[key]['mean']:.6f} | {aggregate[key]['sd_across_training_seeds']:.6f} |")
    lines += [
        "",
        "## Per-seed checkpoint/runtime",
        "",
        "| Seed | primary best epoch | point-global best epoch | wall time [h] | primary checkpoint SHA256 |",
        "|---:|---:|---:|---:|---|",
    ]
    for row in payload["per_seed"]:
        lines.append(f"| {row['seed']} | {row['best_epoch']} | {row['point_global_best_epoch']} | {float(row['training_wall_seconds']) / 3600.0:.3f} | `{row['checkpoint_sha256']['primary']}` |")
    lines += [
        "",
        "## Frozen descriptive references",
        "",
        f"- V7 e200: `{e200['global_rmse_pct']}` global, `{e200['sample_rmse_pct']}` sample-first.",
        f"- V6 e600-trained / point-global-best family: `{v6['global_rmse_pct']}` global, `{v6['sample_rmse_pct']}` sample-first.",
        "- These are descriptive cross-stage references; no checkpoint reselection or G2 table rewrite is performed.",
    ]
    args.output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return payload


def _sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    run(_parse_args())
