#!/usr/bin/env python3
"""Synthetic-first smoke for the Heat3D v3 P0 graph coverage audit."""

from __future__ import annotations

import argparse
from pathlib import Path

from audit_heat3d_v3_graph_coverage import (
    DEFAULT_SPLIT_MAP,
    DEFAULT_SUBSET,
    POLICY_CURRENT,
    POLICY_NO_HARD_RESET_NO_GLOBAL_CLIP,
    REPO_ROOT,
    SYNTHETIC_GRID_SHAPES,
    all_policies,
    audit_real_dataset,
    make_payload,
    print_summary,
    run_synthetic_probes,
    summarize_records,
    write_json_ignored,
)


DEFAULT_OUTPUT_JSON = REPO_ROOT / "output" / "heat3d_v3_graph_coverage" / "smoke.json"
SMALL_SUBSET = (
    REPO_ROOT
    / "data"
    / "heat3d-thermal-simulation"
    / "subsets"
    / "v1_multilayer_bc_eq_supervised_small"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--subset",
        type=Path,
        default=None,
        help="Optional explicit small or medium subset for the up-to-4-sample real-data smoke.",
    )
    parser.add_argument("--split-map", type=Path, default=None)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    return parser.parse_args()


def _records_by_policy(records: list[dict]) -> dict[str, list[dict]]:
    result: dict[str, list[dict]] = {}
    for record in records:
        result.setdefault(record["policy"], []).append(record)
    return result


def _assert_record_contract(record: dict) -> None:
    required = {
        "sample_id",
        "split",
        "seed",
        "policy",
        "n_physical_nodes_real",
        "n_rnodes_real",
        "p2r_real_edge_count",
        "r2p_real_edge_count",
        "r2r_real_edge_count",
        "p2r_physical_node_coverage",
        "r2p_physical_node_coverage",
        "radius_stages",
        "graph_build_time_seconds",
        "metadata_shape_signature",
        "graph_leaf_shape_signature",
        "metadata_all_finite",
        "graph_all_finite",
        "dummy_excluded",
    }
    missing = sorted(required - set(record))
    if missing:
        raise AssertionError(f"{record.get('sample_id')}: missing required fields: {missing}")
    if record["dummy_excluded"] is not True:
        raise AssertionError(f"{record['sample_id']}: dummy nodes/edges were not excluded")
    if not record["metadata_all_finite"] or not record["graph_all_finite"]:
        raise AssertionError(
            f"{record['sample_id']}: graph metadata or graph contains non-finite values"
        )
    for name in ("p2r_physical_node_coverage", "r2p_physical_node_coverage"):
        coverage = record[name]
        if coverage["zero_count"] < 0 or coverage["zero_count"] > record["n_physical_nodes_real"]:
            raise AssertionError(f"{record['sample_id']}: invalid {name} zero_count")
        if coverage["low_coverage_count"] < coverage["zero_count"]:
            raise AssertionError(f"{record['sample_id']}: invalid {name} low coverage count")


def _find_real_subset(
    explicit_subset: Path | None,
    explicit_split_map: Path | None,
) -> tuple[Path | None, Path | None, str]:
    if explicit_subset is not None:
        if not explicit_subset.is_dir():
            raise FileNotFoundError(f"Explicit Heat3D subset does not exist: {explicit_subset}")
        return explicit_subset, explicit_split_map, "explicit"
    if DEFAULT_SUBSET.is_dir():
        split_map = DEFAULT_SPLIT_MAP if DEFAULT_SPLIT_MAP.is_file() else None
        return DEFAULT_SUBSET, split_map, "medium"
    if SMALL_SUBSET.is_dir():
        return SMALL_SUBSET, None, "small"
    return None, None, "no local small or medium Heat3D subset"


def main() -> int:
    args = parse_args()
    policies = all_policies()
    synthetic_records = run_synthetic_probes(seeds=[0], policies=policies)
    expected_records = len(SYNTHETIC_GRID_SHAPES) * len(policies)
    if len(synthetic_records) != expected_records:
        raise AssertionError(
            f"expected {expected_records} synthetic records, found {len(synthetic_records)}"
        )
    for record in synthetic_records:
        _assert_record_contract(record)

    grouped = _records_by_policy(synthetic_records)
    current = summarize_records(grouped[POLICY_CURRENT])[POLICY_CURRENT]
    combined = summarize_records(
        grouped[POLICY_NO_HARD_RESET_NO_GLOBAL_CLIP]
    )[POLICY_NO_HARD_RESET_NO_GLOBAL_CLIP]
    current_zero = current["p2r_zero_count_total"] + current["r2p_zero_count_total"]
    combined_zero = combined["p2r_zero_count_total"] + combined["r2p_zero_count_total"]
    if current_zero <= 0:
        raise AssertionError("synthetic current policy did not reproduce zero coverage")
    if combined_zero >= current_zero:
        raise AssertionError(
            "candidate_no_hard_reset_no_global_clip did not improve synthetic zero coverage"
        )

    subset, split_map, real_subset_kind = _find_real_subset(args.subset, args.split_map)
    real_records: list[dict] = []
    real_status = "skipped"
    real_reason = real_subset_kind
    if subset is not None:
        real_records = audit_real_dataset(
            subset=subset,
            split_map=split_map,
            splits=["all"],
            max_samples=4,
            seeds=[0],
            policies=policies,
            k_encoding_mode="diag3",
            boundary_mask_fallback=True,
        )
        for record in real_records:
            _assert_record_contract(record)
        real_status = "passed"
        real_reason = f"audited up to 4 samples from local {real_subset_kind} subset"

    payload = {
        "schema_version": "heat3d_v3_graph_coverage_smoke_v1",
        "diagnostic_scope": "synthetic and optional real-data graph coverage smoke; no training",
        "synthetic": make_payload(
            records=synthetic_records,
            scope="fixed synthetic Heat3D graph coverage probes; no training",
            config={"rmesh_seeds": [0], "policies": policies},
        ),
        "real_data": {
            "status": real_status,
            "reason": real_reason,
            "summary": summarize_records(real_records),
            "records": real_records,
        },
    }
    output_path = write_json_ignored(args.output_json, payload)

    print("Synthetic coverage summary:")
    print_summary(payload["synthetic"]["summary"])
    print(f"real_data_status={real_status} reason={real_reason}")
    if real_records:
        print("Real-data coverage summary:")
        print_summary(payload["real_data"]["summary"])
    print(f"wrote={output_path}")
    print("Heat3D v3 graph coverage smoke passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
