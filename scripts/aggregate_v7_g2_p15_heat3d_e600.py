#!/usr/bin/env python3
"""Aggregate P15 Heat3D e600 native and scheduled dense receipts.

All input receipts are produced by valid-only runs.  The script never opens a
dataset; it only combines frozen runner/evaluator metadata and metrics.
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


def stats(values: list[float]) -> dict[str, float | int]:
    return {"n": len(values), "mean": mean(values), "sample_sd": stdev(values) if len(values) > 1 else 0.0}


def dense_summary(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") != "PASS_VALID_ONLY_NATIVE_IDW_UV2":
        raise ValueError(f"dense receipt is not valid-only PASS: {path}")
    if any(payload.get("hard_boundaries", {}).get(key) is not False for key in (
        "p1i_test_iid_accessed", "sealed_accessed", "deepoheat_official100_accessed"
    )):
        raise ValueError(f"dense boundary failure: {path}")
    result = {"receipt_sha256": sha256(path), "path": str(path), "seed": payload["seed"]}
    result["runtime"] = payload.get("runtime", {})
    result["representations"] = {}
    for key in ("native_1024", "idw_dense_571256", "u_v2_dense_571256"):
        result["representations"][key] = payload["representations"][key].get("metrics", {})
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-receipt", action="append", required=True, help="SEED=PATH")
    parser.add_argument("--dense-receipt", action="append", required=True, help="SEED:EPOCH=PATH")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if len(args.run_receipt) != 3:
        raise SystemExit("three --run-receipt values are required")
    runs: dict[int, tuple[Path, dict[str, Any]]] = {}
    for item in args.run_receipt:
        seed_text, path_text = item.split("=", 1)
        seed = int(seed_text)
        path = Path(path_text)
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("status") != "COMPLETE_FORMAL_TRAIN" or int(payload.get("epochs", 0)) != 600:
            raise SystemExit(f"run receipt is not complete e600: {path}")
        if payload.get("test_or_sealed_access") is not False:
            raise SystemExit(f"run boundary failure: {path}")
        runs[seed] = (path, payload)
    if set(runs) != {0, 1, 2}:
        raise SystemExit("run seeds must be 0,1,2")
    dense: dict[int, dict[int, dict[str, Any]]] = {0: {}, 1: {}, 2: {}}
    for item in args.dense_receipt:
        key, path_text = item.split("=", 1)
        seed_text, epoch_text = key.split(":", 1)
        seed, epoch = int(seed_text), int(epoch_text)
        if seed not in dense:
            raise SystemExit("dense seed must be 0,1,2")
        dense[seed][epoch] = dense_summary(Path(path_text))
    if any(set(dense[seed]) != {200, 400, 600} for seed in dense):
        raise SystemExit("each seed needs dense receipts at frozen epochs 200,400,600")

    payload_out: dict[str, Any] = {
        "schema_version": "heat3d_v7_g2_p15_e600_aggregation_v1",
        "status": "COMPLETE_VALID_ONLY_E600_THREE_SEEDS",
        "protocol": {
            "path": "configs/heat3d_v7/g2_p15_heat3d_v1_e600_frozen_protocol.json",
            "dense_schedule_epochs": [200, 400, 600],
            "test_or_sealed_access": False,
        },
        "runs": {}, "dense_receipts": {}, "aggregate": {},
        "convergence": {},
    }
    native_best_values: list[float] = []
    final_values: list[float] = []
    common_best_values: list[float] = []
    for seed in (0, 1, 2):
        path, run = runs[seed]
        selection = run["selection"]
        history = run.get("history", [])
        native_by_epoch = {int(row["epoch"]): float(row["native_1024_valid_sample_first_relative_rmse_pct"]) for row in history}
        native_best_epoch = int(selection["best_epoch"])
        native_best_values.append(float(selection["best_value"]))
        final_native = native_by_epoch.get(600)
        if final_native is None:
            raise SystemExit(f"e600 final native metric missing in {path}")
        final_values.append(final_native)
        common_epoch, common_receipt = min(
            dense[seed].items(), key=lambda pair: (
                float(pair[1]["representations"]["u_v2_dense_571256"]["sample_first_relative_rmse_pct"]), pair[0]
            )
        )
        common_value = float(common_receipt["representations"]["u_v2_dense_571256"]["sample_first_relative_rmse_pct"])
        common_best_values.append(common_value)
        payload_out["runs"][str(seed)] = {
            "receipt_path": str(path), "receipt_sha256": sha256(path),
            "native_best_epoch": native_best_epoch,
            "native_best_metric": float(selection["best_value"]),
            "final_epoch": 600, "final_native_metric": final_native,
            "best_to_final_native_degradation_pct_points": final_native - float(selection["best_value"]),
            "training_wall_seconds": run.get("resource", {}).get("training_wall_seconds"),
            "peak_device_memory": run.get("resource", {}).get("peak_bytes_in_use"),
            "parameter_count": run.get("parameter_count"),
            "convergence": "CONVERGED_WITHIN_600" if native_best_epoch < 600 else "RIGHT_CENSORED_AT_600",
        }
        payload_out["dense_receipts"][str(seed)] = {
            str(epoch): value for epoch, value in sorted(dense[seed].items())
        }
        payload_out["runs"][str(seed)]["best_common_scheduled_epoch"] = common_epoch
        payload_out["runs"][str(seed)]["best_common_scheduled_metric"] = common_value
    payload_out["aggregate"] = {
        "native_best_metric": stats(native_best_values),
        "final_native_metric": stats(final_values),
        "best_common_scheduled_u_v2_metric": stats(common_best_values),
        "native_best_to_final_degradation_pct_points": stats([
            payload_out["runs"][str(seed)]["best_to_final_native_degradation_pct_points"] for seed in (0, 1, 2)
        ]),
        "training_wall_seconds": stats([
            float(payload_out["runs"][str(seed)]["training_wall_seconds"]) for seed in (0, 1, 2)
        ]),
    }
    payload_out["hard_boundaries"] = {
        "p1i_test_iid_accessed": False, "sealed_accessed": False,
        "deepoheat_official100_accessed": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload_out, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload_out["status"], "output": str(args.output)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
