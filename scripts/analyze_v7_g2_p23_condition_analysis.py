#!/usr/bin/env python3
"""Apply the already frozen P23 input-only condition bins to common metrics."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


METRICS = (
    "sample_first_relative_rmse_pct",
    "rmse_K",
    "true_hotspot_region_rmse_K",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--common-eval", type=Path, required=True)
    parser.add_argument("--valid-manifest", type=Path, required=True)
    parser.add_argument("--bins", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    common = json.loads(args.common_eval.read_text(encoding="utf-8"))
    if common.get("status") != "COMPLETE_VALID_ONLY":
        raise ValueError("condition analysis requires complete valid-only common evaluation")
    manifest = json.loads(args.valid_manifest.read_text(encoding="utf-8"))
    cases = manifest["cases"]
    bins = json.loads(args.bins.read_text(encoding="utf-8"))
    if bins.get("status") != "FROZEN_BEFORE_MODEL_ERROR":
        raise ValueError("condition-bin receipt is not frozen-before-model-error")
    if bins.get("source_manifest_sha256") != sha256_file(args.valid_manifest):
        raise ValueError("condition-bin source manifest SHA mismatch")
    if len(cases) != 128 or common.get("valid_case_count") != 128:
        raise ValueError("condition analysis requires exactly 128 valid cases")

    by_case_model_seed: dict[tuple[str, str], list[dict]] = {}
    for row in common["rows"]:
        by_case_model_seed.setdefault((str(row["model"]), str(row["sample_id"])), []).append(row)
    case_meta = {str(row["sample_id"]): row for row in cases}
    if set(case_meta) != {str(row["sample_id"]) for row in common["rows"]}:
        raise ValueError("common evaluator case IDs do not match frozen valid manifest")

    continuous = bins["continuous_bins"]
    rules: dict[str, dict[str, object]] = {}
    for name in ("source_count", "k_region_count"):
        rules[name] = {"kind": "discrete", "bins": bins["discrete_bins"][name]}
    for name, spec in continuous.items():
        if not isinstance(spec, dict) or "q33" not in spec or "q67" not in spec:
            continue
        rules[name] = {"kind": "continuous", "q33": spec["q33"], "q67": spec["q67"]}

    def bucket(name: str, value: float) -> str:
        rule = rules[name]
        if rule["kind"] == "discrete":
            for label, span in rule["bins"].items():
                if int(span["inclusive_range"][0]) <= int(value) <= int(span["inclusive_range"][1]):
                    return label
            raise ValueError(f"value {value} not covered by frozen discrete bins for {name}")
        if value <= float(rule["q33"]):
            return "low"
        if value <= float(rule["q67"]):
            return "mid"
        return "high"

    rows: list[dict[str, object]] = []
    for variable in rules:
        groups: dict[str, list[str]] = {"low": [], "mid": [], "high": []}
        for sid, meta in case_meta.items():
            groups[bucket(variable, float(meta[variable]))].append(sid)
        for label, ids in groups.items():
            if not ids:
                raise ValueError(f"empty frozen condition bin {variable}/{label}")
            record: dict[str, object] = {
                "condition": variable,
                "bin": label,
                "case_count": len(ids),
                "bin_definition": rules[variable],
            }
            for metric in METRICS:
                model_means: dict[str, float] = {}
                per_case: dict[str, dict[str, float]] = {}
                for model in ("Heat3D", "Therm-FM"):
                    values: list[float] = []
                    for sid in ids:
                        cohort = by_case_model_seed[(model, sid)]
                        if len(cohort) != 3:
                            raise ValueError(f"{model}/{sid} does not have exactly 3 seeds")
                        value = float(np.mean([float(row[metric]) for row in cohort]))
                        values.append(value)
                        per_case.setdefault(sid, {})[model] = value
                    model_means[model] = float(np.mean(values))
                record[f"{metric}_Heat3D_seed_mean"] = model_means["Heat3D"]
                record[f"{metric}_ThermFM_seed_mean"] = model_means["Therm-FM"]
                record[f"{metric}_paired_difference_Heat3D_minus_ThermFM"] = (
                    model_means["Heat3D"] - model_means["Therm-FM"]
                )
            rows.append(record)

    result = {
        "schema_version": "heat3d_v7_g2_p23_condition_analysis_v1",
        "status": "COMPLETE_VALID_ONLY",
        "bin_policy": "frozen input-only condition bins; no error-driven rebinning",
        "valid_manifest_sha256": sha256_file(args.valid_manifest),
        "bins_receipt_sha256": sha256_file(args.bins),
        "common_evaluation_sha256": sha256_file(args.common_eval),
        "case_count": 128,
        "metrics": list(METRICS),
        "dispersion_label": "SD across training seeds; condition rows use per-case seed means",
        "test_iid_read": False,
        "sealed_read": False,
        "deepoheat_official100_read": False,
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "rows": len(rows)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
