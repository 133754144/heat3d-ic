#!/usr/bin/env python3
"""Synthetic-first smoke for the Heat3D v3 P0/P1 graph coverage audit."""

from __future__ import annotations

import argparse
from pathlib import Path

import jax
import numpy as np

from audit_heat3d_v3_graph_coverage import (
    DEFAULT_SPLIT_MAP,
    DEFAULT_SUBSET,
    POLICY_CURRENT,
    POLICY_DISCRETE_COVERAGE_RADIUS,
    POLICY_DISCRETE_COVERAGE_RADIUS_WITH_NEAREST_REPAIR,
    POLICY_NEAREST_REPAIR,
    POLICY_NO_HARD_RESET_NO_GLOBAL_CLIP,
    REPO_ROOT,
    SYNTHETIC_GRID_SHAPES,
    _synthetic_grid,
    all_policies,
    audit_real_dataset,
    make_payload,
    print_summary,
    run_synthetic_probes,
    summarize_records,
    write_json_ignored,
)
from rigno.graphBuilder_Heat3D import Heat3DGraphBuilder


DEFAULT_OUTPUT_JSON = REPO_ROOT / "output" / "heat3d_v3_graph_coverage" / "smoke.json"
SMALL_SUBSET = (
    REPO_ROOT
    / "data"
    / "heat3d-thermal-simulation"
    / "subsets"
    / "v1_multilayer_bc_eq_supervised_small"
)
EXPECTED_SYNTHETIC_CURRENT = {
    "p2r_zero_count_total": 210,
    "r2p_zero_count_total": 289,
    "p2r_real_edge_count_total": 17744,
    "r2p_real_edge_count_total": 32001,
}


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
        "edge_ratio_vs_legacy",
        "repaired_edge_count",
        "repaired_physical_count",
        "node_and_r2r_stable_vs_legacy",
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
    if not record["node_and_r2r_stable_vs_legacy"]:
        raise AssertionError(f"{record['sample_id']}: candidate changed node or r2r topology")
    for name in ("p2r_physical_node_coverage", "r2p_physical_node_coverage"):
        coverage = record[name]
        if coverage["zero_count"] < 0 or coverage["zero_count"] > record["n_physical_nodes_real"]:
            raise AssertionError(f"{record['sample_id']}: invalid {name} zero_count")
        if coverage["low_coverage_count"] < coverage["zero_count"]:
            raise AssertionError(f"{record['sample_id']}: invalid {name} low coverage count")


def _assert_default_discrete_equivalence() -> None:
    coords, _, _ = _synthetic_grid(SYNTHETIC_GRID_SHAPES[0])
    key = jax.random.PRNGKey(0)
    default_metadata = Heat3DGraphBuilder().build_metadata(coords, key=key)
    explicit_metadata = Heat3DGraphBuilder(
        coverage_repair_policy="none",
        radius_policy="discrete_physical_coverage",
        repair_p2r=True,
        repair_r2p=True,
        min_physical_coverage=1,
    ).build_metadata(coords, key=key)
    default_leaves = jax.tree_util.tree_leaves(default_metadata)
    explicit_leaves = jax.tree_util.tree_leaves(explicit_metadata)
    if len(default_leaves) != len(explicit_leaves):
        raise AssertionError("default and explicit discrete metadata leaf counts differ")
    if not all(
        np.array_equal(np.asarray(default), np.asarray(explicit))
        for default, explicit in zip(default_leaves, explicit_leaves)
    ):
        raise AssertionError("default builder is not exactly equivalent to explicit discrete policy")


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
    _assert_default_discrete_equivalence()
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
    for key, expected in EXPECTED_SYNTHETIC_CURRENT.items():
        if current[key] != expected:
            raise AssertionError(
                f"legacy synthetic baseline changed: {key}={current[key]}, expected {expected}"
            )
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

    p1_summaries = {
        policy: summarize_records(grouped[policy])[policy]
        for policy in (
            POLICY_NEAREST_REPAIR,
            POLICY_DISCRETE_COVERAGE_RADIUS,
            POLICY_DISCRETE_COVERAGE_RADIUS_WITH_NEAREST_REPAIR,
        )
    }
    for policy, summary in p1_summaries.items():
        if not summary["coverage_gate_passed"]:
            raise AssertionError(f"{policy} did not reach synthetic zero coverage")
        if not summary["all_metadata_finite"] or not summary["all_graphs_finite"]:
            raise AssertionError(f"{policy} produced non-finite synthetic graph data")
        if not summary["all_node_and_r2r_stable_vs_legacy"]:
            raise AssertionError(f"{policy} changed synthetic node or r2r topology")

    nearest = p1_summaries[POLICY_NEAREST_REPAIR]
    if nearest["p2r_repaired_edge_count_total"] != current["p2r_zero_count_total"]:
        raise AssertionError("nearest repair p2r additions do not match legacy zero coverage")
    if nearest["r2p_repaired_edge_count_total"] != current["r2p_zero_count_total"]:
        raise AssertionError("nearest repair r2p additions do not match legacy zero coverage")
    if nearest["p2r_edge_ratio_vs_legacy"] >= combined["p2r_edge_ratio_vs_legacy"]:
        raise AssertionError("nearest repair p2r edge ratio did not beat P0 combined candidate")
    if nearest["r2p_edge_ratio_vs_legacy"] >= combined["r2p_edge_ratio_vs_legacy"]:
        raise AssertionError("nearest repair r2p edge ratio did not beat P0 combined candidate")

    discrete = p1_summaries[POLICY_DISCRETE_COVERAGE_RADIUS]
    discrete_with_repair = p1_summaries[
        POLICY_DISCRETE_COVERAGE_RADIUS_WITH_NEAREST_REPAIR
    ]
    if discrete["p2r_real_edge_count_total"] != discrete_with_repair["p2r_real_edge_count_total"]:
        raise AssertionError("nearest repair unexpectedly changed discrete p2r coverage graph")
    if discrete["r2p_real_edge_count_total"] != discrete_with_repair["r2p_real_edge_count_total"]:
        raise AssertionError("nearest repair unexpectedly changed discrete r2p coverage graph")
    if (
        discrete_with_repair["p2r_repaired_edge_count_total"] != 0
        or discrete_with_repair["r2p_repaired_edge_count_total"] != 0
    ):
        raise AssertionError("discrete coverage radius unexpectedly required nearest repair")

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
        "schema_version": "heat3d_v3_graph_coverage_smoke_v2",
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
