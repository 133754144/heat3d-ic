#!/usr/bin/env python3
"""Aggregate matched DeepOHeat-v1 receipts using valid-only metrics.

The script reads only completed matched-run receipts and never discovers or
opens a test/sealed artifact.  It intentionally reports descriptive mean and
sample SD; no seed or metric is removed after observing a result.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from pathlib import Path
from typing import Any


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--seed-dir", action="append", required=True, help="seed directory name; repeat three times")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if len(args.seed_dir) != 3:
        raise ValueError("matched aggregation requires exactly three seed directories")
    rows: list[dict[str, Any]] = []
    for name in args.seed_dir:
        path = args.root / name / "formal_training_receipt.json"
        if not path.is_file():
            raise FileNotFoundError(path)
        receipt = json.loads(path.read_text(encoding="utf-8"))
        if receipt.get("status") != "COMPLETE_MATCHED_PHYSICAL_CASE_BUDGET_TRAINING":
            raise ValueError(f"seed receipt is not complete: {path}")
        if any(receipt.get("hard_boundaries", {}).get(key) for key in ("p1i_test_iid_accessed", "sealed_accessed", "deepoheat_official100_accessed")):
            raise ValueError(f"forbidden split access in {path}")
        history = receipt["metrics"]["validation_history"]
        if not history:
            raise ValueError(f"missing valid history in {path}")
        final = history[-1]
        rows.append({
            "seed": int(receipt["seed"]),
            "directory": name,
            "receipt_sha256": sha256(path),
            "best_iteration": receipt["metrics"]["best_iteration"],
            "best_primary_metric_pct": receipt["metrics"]["best_metric"],
            "final_iteration": final["iteration"],
            "final_primary_metric_pct": final["sample_first_relative_rmse_pct"],
            "best_to_final_degradation_pct_points": final["sample_first_relative_rmse_pct"] - receipt["metrics"]["best_metric"],
            "final_physics_loss": receipt["metrics"]["final_physics_loss"],
            "wall_seconds": receipt["runtime"]["total_wall_seconds"],
            "peak_rss_bytes": receipt["runtime"]["peak_rss_bytes"],
            "parameter_count": receipt["model"]["parameter_count"],
            "pde_collocation_evaluations": receipt["model"]["pde_collocation_evaluations"],
            "checkpoint_reload": receipt["checkpoint"]["reload"],
        })
    rows.sort(key=lambda row: row["seed"])
    if [row["seed"] for row in rows] != [0, 1, 2]:
        raise ValueError("seed directories must cover 0,1,2 exactly")
    def mean_sd(key: str) -> dict[str, float]:
        values = [float(row[key]) for row in rows]
        return {"mean": float(statistics.mean(values)), "sample_sd": float(statistics.stdev(values))}
    aggregate = {
        "best_primary_metric_pct": mean_sd("best_primary_metric_pct"),
        "final_primary_metric_pct": mean_sd("final_primary_metric_pct"),
        "best_to_final_degradation_pct_points": mean_sd("best_to_final_degradation_pct_points"),
        "wall_seconds": mean_sd("wall_seconds"),
        "peak_rss_bytes_max": max(row["peak_rss_bytes"] for row in rows),
        "parameter_count": rows[0]["parameter_count"],
        "pde_collocation_evaluations": rows[0]["pde_collocation_evaluations"],
    }
    payload = {
        "schema_version": "heat3d_v7_g2_p14_deepoheat_v1_same_physical_case_budget_aggregation_v1",
        "status": "COMPLETE_VALID_ONLY_SAME_PHYSICAL_CASE_BUDGET",
        "classification": "SAME_PHYSICAL_CASE_BUDGET",
        "same_information_budget": False,
        "selection_metric": "valid full-field temperature-space sample_first_relative_rmse_pct",
        "selection_split": "valid",
        "seeds": rows,
        "aggregate": aggregate,
        "anomaly_assessment": "descriptive only; no seed/metric/config removal or post-hoc change",
        "test_or_sealed_access": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
