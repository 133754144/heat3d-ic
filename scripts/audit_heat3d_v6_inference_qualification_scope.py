#!/usr/bin/env python3
"""Describe the fixed valid sample scope used by the qualification timing."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import statistics
from typing import Any


def stats(values: list[float]) -> dict[str, float]:
    ordered = sorted(values)
    return {
        "min": min(values), "median": statistics.median(values), "mean": statistics.mean(values),
        "p95": ordered[min(len(ordered) - 1, int(0.95 * len(ordered)))], "max": max(values),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", choices=("p1i", "randomblock"), required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    valid_name = "valid_iid" if args.family == "p1i" else "valid"
    valid = [row for row in manifest["samples"] if row["split_role"] == valid_name]
    if args.family == "p1i":
        selected = sorted(valid, key=lambda row: hashlib.sha256(row["sample_id"].encode()).hexdigest())[:32]
    else:
        groups: dict[str, list[dict[str, Any]]] = {}
        for row in valid:
            groups.setdefault(str(row["group_id"]), []).append(row)
        selected = sum((sorted(groups[key], key=lambda row: row["sample_id"])[:2] for key in sorted(groups)), [])
    records = []
    for row in selected:
        relative = row.get("relative_path") or row.get("sample_dir") or row["sample_id"]
        meta = json.loads((args.dataset_root / relative / "sample_meta.json").read_text(encoding="utf-8"))
        if args.family == "p1i":
            source_count = int(meta["source_region_count"])
            k_count = int(meta["k_region_count"])
            iterations = int(meta["cg_iterations"])
            coordinate_hash = str(meta["coordinate_sha256"])
        else:
            source_count = sum(block["family"] == "q" for block in meta["blocks"])
            k_count = sum(block["family"] == "k" for block in meta["blocks"])
            iterations = int(meta["metrics"]["cg_iterations"])
            coordinate_hash = str(meta["support"]["coordinate_sha256"])
        records.append({
            "sample_id": row["sample_id"], "group_id": row.get("group_id", row["sample_id"]),
            "role": "valid_iid", "support_nodes": 1024, "solver_nodes": 240825,
            "coordinate_sha256": coordinate_hash, "source_region_count": source_count,
            "conductivity_region_count": k_count, "cg_iterations": iterations,
        })
    payload = {
        "schema_version": "heat3d_v6_inference_qualification_sample_scope_v1",
        "family": args.family, "dataset_id": manifest["dataset_id"], "sample_count": len(records),
        "selection": "sha256_rank_first32" if args.family == "p1i" else "16_groups_x_first2_variants",
        "roles": ["valid_iid"], "test_accessed": False, "sealed_accessed": False,
        "unique_support_hashes": len({row["coordinate_sha256"] for row in records}),
        "source_region_count": stats([row["source_region_count"] for row in records]),
        "conductivity_region_count": stats([row["conductivity_region_count"] for row in records]),
        "cg_iterations": stats([row["cg_iterations"] for row in records]),
        "group_multiplicity": dict(Counter(row["group_id"] for row in records)),
        "samples": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": "passed", "family": args.family, "samples": len(records)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
