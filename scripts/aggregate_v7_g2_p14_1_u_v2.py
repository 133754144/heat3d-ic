#!/usr/bin/env python3
"""Aggregate P14.1 valid-only native/IDW/U-v2 receipts.

The raw receipts are intentionally kept outside the repository (normally in
``/tmp`` on the devbox).  This script emits a compact, provenance-rich
summary suitable for version control and never reads a test/sealed artifact.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from statistics import mean, stdev
from typing import Any


REPRESENTATIONS = (
    "native_1024",
    "idw_dense_571256",
    "u_v2_dense_571256",
    "oracle_idw_gt_support_571256",
)
METRICS = (
    "sample_first_relative_rmse_pct",
    "point_global_relative_rmse_pct",
    "raw_cv_weighted_rmse_K",
    "MAE_K",
    "peak_RMSE_K",
    "peak_temperature_mean_abs_error_K",
    "peak_temperature_max_abs_error_K",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def summarize(values: list[float]) -> dict[str, float | int]:
    clean = [float(value) for value in values]
    return {
        "n": len(clean),
        "mean": mean(clean),
        "sample_sd": stdev(clean) if len(clean) > 1 else 0.0,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed-receipt", action="append", required=True,
                        help="SEED=PATH, repeated exactly three times")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if len(args.seed_receipt) != 3:
        raise SystemExit("exactly three --seed-receipt arguments are required")
    receipts: dict[str, tuple[Path, dict[str, Any]]] = {}
    for item in args.seed_receipt:
        try:
            seed_text, path_text = item.split("=", 1)
            seed = str(int(seed_text))
        except ValueError as exc:
            raise SystemExit(f"invalid --seed-receipt {item!r}; use SEED=PATH") from exc
        path = Path(path_text)
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("status") != "PASS_VALID_ONLY_NATIVE_IDW_UV2":
            raise SystemExit(f"receipt {path} is not a valid-only PASS")
        if int(payload.get("seed")) != int(seed):
            raise SystemExit(f"seed mismatch for {path}")
        if int(payload.get("max_valid")) != 128:
            raise SystemExit(f"receipt {path} is not the frozen valid128 evaluation")
        if payload.get("hard_boundaries", {}).get("p1i_test_iid_accessed") is not False:
            raise SystemExit(f"test_iid boundary failed in {path}")
        if payload.get("hard_boundaries", {}).get("sealed_accessed") is not False:
            raise SystemExit(f"sealed boundary failed in {path}")
        if payload.get("hard_boundaries", {}).get("deepoheat_official100_accessed") is not False:
            raise SystemExit(f"official100 boundary failed in {path}")
        receipts[seed] = (path, payload)
    if set(receipts) != {"0", "1", "2"}:
        raise SystemExit("seed receipts must be 0, 1, and 2")

    first = next(iter(receipts.values()))[1]
    provenance = {
        "protocol": first.get("runner", {}).get("protocol"),
        "protocol_sha256": first.get("runner", {}).get("protocol_sha256"),
        "source_sha256": first.get("data", {}).get("source_sha256"),
        "subset_manifest_sha256": first.get("data", {}).get("subset_manifest_sha256"),
        "label_receipt_sha256": first.get("data", {}).get("label_receipt_sha256"),
        "normalization_payload_sha256": first.get("data", {}).get("normalization_payload_sha256"),
        "evaluation_domain": first.get("domain"),
        "test_or_sealed_access": False,
    }
    aggregate: dict[str, Any] = {
        "schema_version": "heat3d_v7_g2_p14_1_u_v2_aggregate_v1",
        "status": "PASS_VALID_ONLY_THREE_SEEDS",
        "provenance": provenance,
        "raw_receipts": {
            seed: {"path": str(path), "sha256": sha256(path)}
            for seed, (path, _) in sorted(receipts.items())
        },
        "seeds": [0, 1, 2],
        "representations": {},
        "oracle_policy": first.get("audit", {}),
        "runtime": {},
        "hard_boundaries": {
            "test_iid_accessed": False,
            "sealed_accessed": False,
            "deepoheat_official100_accessed": False,
            "training_started": False,
            "large_predictions_persisted": False,
        },
    }
    for representation in REPRESENTATIONS:
        rows = []
        for seed in ("0", "1", "2"):
            row = receipts[seed][1]["representations"].get(representation)
            if row is None:
                raise SystemExit(f"{representation} missing in seed {seed}")
            if row.get("status") == "NOT_DEFINED_FOR_DIRECT_QUERY":
                rows.append({"seed": int(seed), "status": row["status"], "reason": row.get("reason")})
                continue
            metrics = row.get("metrics", {})
            selected = {metric: float(metrics[metric]) for metric in METRICS if metric in metrics}
            rows.append({"seed": int(seed), "sample_count": row.get("sample_count"), "metrics": selected})
        if all(row.get("status") == "NOT_DEFINED_FOR_DIRECT_QUERY" for row in rows):
            aggregate["representations"][representation] = {
                "status": "NOT_DEFINED_FOR_DIRECT_QUERY",
                "reason": rows[0].get("reason"),
                "per_seed": rows,
            }
            continue
        metric_summary: dict[str, Any] = {}
        for metric in METRICS:
            values = [row["metrics"][metric] for row in rows if "metrics" in row and metric in row["metrics"]]
            if len(values) != 3:
                continue
            metric_summary[metric] = summarize(values)
        aggregate["representations"][representation] = {
            "status": "PASS_VALID_ONLY",
            "per_seed": rows,
            "across_seed": metric_summary,
        }
    for seed in ("0", "1", "2"):
        runtime = receipts[seed][1].get("runtime", {})
        aggregate["runtime"][seed] = {
            "native_inference_seconds": runtime.get("native_inference_seconds"),
            "u_v2_inference_seconds": runtime.get("u_v2_inference_seconds"),
            "backend": runtime.get("backend"),
            "devices": runtime.get("devices"),
        }
    all_u = [aggregate["runtime"][str(seed)]["u_v2_inference_seconds"] for seed in (0, 1, 2)]
    all_native = [aggregate["runtime"][str(seed)]["native_inference_seconds"] for seed in (0, 1, 2)]
    aggregate["runtime_summary"] = {
        "native_inference_seconds": summarize(all_native),
        "u_v2_inference_seconds": summarize(all_u),
        "u_v2_latency_includes": ["full-query graph construction", "model direct-query forward"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(aggregate, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": aggregate["status"], "output": str(args.output)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
