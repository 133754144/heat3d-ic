#!/usr/bin/env python3
"""Aggregate valid-only P17 DeepOHeat full-minus-valid128 receipts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from statistics import mean, stdev
from typing import Any


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def summary(values: list[float]) -> dict[str, float | int]:
    return {"n": len(values), "mean": mean(values), "sample_sd": stdev(values) if len(values) > 1 else 0.0}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--receipt", action="append", required=True, help="SEED=PATH")
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    if len(args.receipt) != 3:
        raise SystemExit("three receipts are required")
    rows: dict[int, tuple[Path, dict[str, Any]]] = {}
    for item in args.receipt:
        seed_text, path_text = item.split("=", 1)
        seed, path = int(seed_text), Path(path_text)
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("status") != "COMPLETE_MATCHED_PHYSICAL_CASE_BUDGET_TRAINING":
            if payload.get("status") != "COMPLETE_NATIVE_RECIPE_HELDOUT_VALIDATION_TRAINING":
                raise SystemExit(f"not a complete P17 receipt: {path}")
        if payload.get("hard_boundaries", {}).get("test_or_sealed_access", False):
            raise SystemExit(f"test boundary failed: {path}")
        rows[seed] = (path, payload)
    if set(rows) != {0, 1, 2}:
        raise SystemExit("seed receipts must be 0,1,2")
    out: dict[str, Any] = {
        "schema_version": "heat3d_v7_g2_p17_deepoheat_full_minus_valid128_aggregation_v1",
        "status": "COMPLETE_VALID_ONLY_THREE_SEEDS",
        "protocol": {
            "path": "configs/heat3d_v7/g2_p17_deepoheat_full_minus_valid128_frozen_protocol.json",
            "exclusion_manifest": "configs/heat3d_v7/g2_p17_deepoheat_full_minus_valid128_exclusion_manifest.json",
            "validation": "excluded Heat3D valid128; full 101x101x56 temperature field",
            "test_or_sealed_access": False,
        },
        "seeds": {}, "aggregate": {},
        "hard_boundaries": {"test_iid": False, "sealed": False, "official100": False},
    }
    best, final, wall, rss, device = [], [], [], [], []
    for seed in (0, 1, 2):
        path, p = rows[seed]
        history = p.get("metrics", {}).get("validation_history", [])
        if not history:
            raise SystemExit(f"validation history missing: {path}")
        best_row = min(history, key=lambda x: (float(x["sample_first_relative_rmse_pct"]), int(x["iteration"])))
        final_row = next((x for x in history if int(x["iteration"]) == 100000), history[-1])
        best_value = float(best_row["sample_first_relative_rmse_pct"])
        final_value = float(final_row["sample_first_relative_rmse_pct"])
        best.append(best_value); final.append(final_value)
        runtime = p.get("runtime", {})
        wall_value = float(runtime.get("total_wall_seconds", 0.0)); wall.append(wall_value)
        rss_value = float(runtime.get("peak_rss_bytes", 0.0)); rss.append(rss_value)
        device_value = float((runtime.get("device_after") or {}).get("memory_stats", {}).get("peak_bytes_in_use") or 0.0); device.append(device_value)
        out["seeds"][str(seed)] = {
            "receipt_path": str(path), "receipt_sha256": sha256(path),
            "training_case_count": p.get("data", {}).get("train_case_count"),
            "best_iteration": int(best_row["iteration"]), "best_metrics": best_row,
            "final_iteration": int(final_row["iteration"]), "final_metrics": final_row,
            "best_to_final_degradation_pct_points": final_value - best_value,
            "training_wall_seconds": wall_value, "peak_rss_bytes": int(rss_value),
            "peak_device_bytes": int(device_value),
            "parameter_count": p.get("model", {}).get("parameter_count"),
            "best_checkpoint": p.get("checkpoint", {}).get("best"),
            "final_checkpoint": {k: p.get("checkpoint", {}).get(k) for k in ("final_model_file", "final_model_sha256", "final_optimizer_file", "final_optimizer_sha256")},
            "checkpoint_reload": p.get("checkpoint", {}).get("reload"),
        }
    out["aggregate"] = {
        "best_sample_first_relative_rmse_pct": summary(best),
        "final_sample_first_relative_rmse_pct": summary(final),
        "best_to_final_degradation_pct_points": summary([final[i] - best[i] for i in range(3)]),
        "training_wall_seconds": summary(wall),
        "peak_rss_bytes_max": max(rss), "peak_device_bytes_max": max(device),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": out["status"], "output": str(args.output)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
