#!/usr/bin/env python3
"""Hardening smoke for Heat3D v3 graph policy compatibility.

This script checks graph construction and one batch forward only. It does not
train, change checkpoints, or write non-ignored artifacts.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from audit_heat3d_v3_graph_coverage import (  # noqa: E402
    POLICY_CURRENT,
    POLICY_DISCRETE_COVERAGE_RADIUS,
    POLICY_NEAREST_REPAIR,
    SYNTHETIC_GRID_SHAPES,
    _shape_signature,
    _synthetic_grid,
    audit_real_dataset,
    run_synthetic_probes,
    summarize_records,
    write_json_ignored,
)
from check_heat3d_v1_small_train_valid_smoke import (  # noqa: E402
    DEFAULT_MANIFEST,
    DEFAULT_SUBSET,
    MODEL_CONFIG,
    _load_manifest,
    _make_groups,
    _resolve_split_ids,
    _sample_root,
    _train_only_stats,
)
from rigno.graphBuilder_Heat3D import Heat3DGraphBuilder  # noqa: E402
from rigno.heat3d_v1_native_supervised import Heat3DV1NativeSupervisedDataset  # noqa: E402
from rigno.models.rigno import RIGNO as GraphNeuralOperator  # noqa: E402


POLICIES = {
    "legacy": {
        "audit_policy": POLICY_CURRENT,
        "builder_kwargs": {
            "radius_policy": "legacy_kdtree_mean4",
            "coverage_repair_policy": "none",
            "repair_p2r": True,
            "repair_r2p": True,
            "min_physical_coverage": 1,
        },
    },
    "nearest_repair": {
        "audit_policy": POLICY_NEAREST_REPAIR,
        "builder_kwargs": {
            "radius_policy": "legacy_kdtree_mean4",
            "coverage_repair_policy": "nearest_rnode",
            "repair_p2r": True,
            "repair_r2p": True,
            "min_physical_coverage": 1,
        },
    },
    "discrete_radius": {
        "audit_policy": POLICY_DISCRETE_COVERAGE_RADIUS,
        "builder_kwargs": {
            "radius_policy": "discrete_physical_coverage",
            "coverage_repair_policy": "none",
            "repair_p2r": True,
            "repair_r2p": True,
            "min_physical_coverage": 1,
        },
    },
}
EXPECTED_SYNTHETIC_CURRENT = {
    "p2r_zero_count_total": 210,
    "r2p_zero_count_total": 289,
    "p2r_real_edge_count_total": 17744,
    "r2p_real_edge_count_total": 32001,
}
DEFAULT_OUTPUT_JSON = (
    REPO_ROOT / "output" / "heat3d_v3_p2_policy_smoke" / "hardening.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--subset", type=Path, default=DEFAULT_SUBSET)
    parser.add_argument("--k-encoding-mode", default="diag3")
    parser.add_argument("--max-forward-samples", type=int, default=2)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    return parser.parse_args()


def _policy_builder(policy_name: str) -> Heat3DGraphBuilder:
    return Heat3DGraphBuilder(**POLICIES[policy_name]["builder_kwargs"])


def _tree_all_equal(left: Any, right: Any) -> bool:
    left_leaves = jax.tree_util.tree_leaves(left)
    right_leaves = jax.tree_util.tree_leaves(right)
    return len(left_leaves) == len(right_leaves) and all(
        np.array_equal(np.asarray(left_leaf), np.asarray(right_leaf))
        for left_leaf, right_leaf in zip(left_leaves, right_leaves)
    )


def _tree_all_finite(value: Any) -> bool:
    return all(
        bool(np.all(np.isfinite(np.asarray(leaf))))
        for leaf in jax.tree_util.tree_leaves(value)
        if hasattr(leaf, "shape")
    )


def _edge_batches(edges: Any) -> np.ndarray:
    values = np.asarray(edges)
    if values.ndim == 2:
        values = values[None, ...]
    if values.ndim != 3 or values.shape[-1] != 2:
        raise AssertionError(f"edge array must have shape [B, E, 2], found {values.shape}")
    return values.astype(np.int64)


def _check_edge_bounds(
    *,
    name: str,
    edges: Any,
    sender_count: int,
    receiver_count: int,
) -> dict[str, int]:
    batches = _edge_batches(edges)
    real_count = 0
    for batch_index, batch in enumerate(batches):
        dummy = (batch[:, 0] == sender_count) & (batch[:, 1] == receiver_count)
        real = (
            (batch[:, 0] >= 0)
            & (batch[:, 0] < sender_count)
            & (batch[:, 1] >= 0)
            & (batch[:, 1] < receiver_count)
        )
        invalid = ~(dummy | real)
        if int(np.sum(dummy)) != 1:
            raise AssertionError(
                f"{name}: batch {batch_index} expected one dummy edge, "
                f"found {int(np.sum(dummy))}"
            )
        if bool(np.any(invalid)):
            bad = batch[invalid][:5].tolist()
            raise AssertionError(f"{name}: batch {batch_index} has invalid edges {bad}")
        real_count += int(np.sum(real))
    return {
        "batch_count": int(batches.shape[0]),
        "real_edge_count": real_count,
        "edge_count_including_dummy": int(batches.shape[0] * batches.shape[1]),
    }


def _r2p_edges(metadata: Any) -> Any:
    if metadata.r2p_edge_indices is None:
        return jnp.flip(metadata.p2r_edge_indices, axis=-1)
    return metadata.r2p_edge_indices


def _check_metadata_edges(metadata: Any) -> dict[str, dict[str, int]]:
    n_pnodes_inp = int(np.asarray(metadata.x_pnodes_inp).shape[1] - 1)
    n_pnodes_out = int(np.asarray(metadata.x_pnodes_out).shape[1] - 1)
    n_rnodes = int(np.asarray(metadata.x_rnodes).shape[1] - 1)
    return {
        "p2r": _check_edge_bounds(
            name="p2r",
            edges=metadata.p2r_edge_indices,
            sender_count=n_pnodes_inp,
            receiver_count=n_rnodes,
        ),
        "r2r": _check_edge_bounds(
            name="r2r",
            edges=metadata.r2r_edge_indices,
            sender_count=n_rnodes,
            receiver_count=n_rnodes,
        ),
        "r2p": _check_edge_bounds(
            name="r2p",
            edges=_r2p_edges(metadata),
            sender_count=n_rnodes,
            receiver_count=n_pnodes_out,
        ),
    }


def _assert_default_discrete_equivalence(coords: np.ndarray) -> dict[str, Any]:
    key = jax.random.PRNGKey(0)
    default_builder = Heat3DGraphBuilder()
    explicit_builder = _policy_builder("discrete_radius")
    default_metadata = default_builder.build_metadata(coords, key=key)
    explicit_metadata = explicit_builder.build_metadata(coords, key=key)
    default_graphs = default_builder.build_graphs(default_metadata)
    explicit_graphs = explicit_builder.build_graphs(explicit_metadata)
    metadata_equal = _tree_all_equal(default_metadata, explicit_metadata)
    graph_equal = _tree_all_equal(default_graphs, explicit_graphs)
    if not metadata_equal or not graph_equal:
        raise AssertionError(
            "default Heat3DGraphBuilder is not exactly equivalent to explicit discrete_radius policy"
        )
    return {
        "metadata_equal": metadata_equal,
        "graph_equal": graph_equal,
        "metadata_shape_signature": _shape_signature(default_metadata),
        "graph_leaf_shape_signature": _shape_signature(default_graphs),
    }


def _assert_synthetic_baseline() -> dict[str, Any]:
    records = run_synthetic_probes(
        seeds=[0],
        policies=[policy["audit_policy"] for policy in POLICIES.values()],
    )
    summary = summarize_records(records)
    current = summary[POLICY_CURRENT]
    for key, expected in EXPECTED_SYNTHETIC_CURRENT.items():
        if current[key] != expected:
            raise AssertionError(
                f"legacy synthetic baseline changed: {key}={current[key]}, expected {expected}"
            )
    for policy in (POLICY_NEAREST_REPAIR, POLICY_DISCRETE_COVERAGE_RADIUS):
        if not summary[policy]["coverage_gate_passed"]:
            raise AssertionError(f"{policy} synthetic coverage gate failed")
        if not summary[policy]["all_node_and_r2r_stable_vs_legacy"]:
            raise AssertionError(f"{policy} changed synthetic node/r2r topology")
    return {
        "records": records,
        "summary": summary,
    }


def _load_forward_examples(args: argparse.Namespace) -> tuple[list[Any], dict[str, list[str]], str]:
    sample_root = _sample_root(args.subset)
    if not sample_root.is_dir():
        raise FileNotFoundError(f"Heat3D subset sample root does not exist: {sample_root}")
    manifest = _load_manifest(args.manifest) if args.manifest.is_file() else {"samples": []}
    split_ids, split_source = _resolve_split_ids(manifest, sample_root)
    train_ids = split_ids.get("train", [])
    if not train_ids:
        raise ValueError(f"Expected non-empty train split in {sample_root}")
    selected_ids = train_ids[: args.max_forward_samples]
    dataset = Heat3DV1NativeSupervisedDataset(sample_root, k_encoding_mode=args.k_encoding_mode)
    index_by_id = dataset.sample_index_by_id()
    examples = [dataset[index_by_id[sample_id]] for sample_id in selected_ids]
    return examples, split_ids, split_source


def _check_r2r_stable(candidate: Any, legacy: Any) -> bool:
    return all(
        np.array_equal(np.asarray(getattr(candidate, field)), np.asarray(getattr(legacy, field)))
        for field in ("x_rnodes", "r2r_edge_indices", "r2r_edge_domains")
    )


def _metadata_checks_for_examples(examples: list[Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for policy_name in POLICIES:
        builder = _policy_builder(policy_name)
        policy_rows = []
        for example in examples:
            key = jax.random.PRNGKey(0)
            metadata = builder.build_metadata(example.condition.coords, key=key)
            graphs = builder.build_graphs(metadata)
            legacy_metadata = _policy_builder("legacy").build_metadata(
                example.condition.coords,
                key=key,
            )
            edge_checks = _check_metadata_edges(metadata)
            if not _tree_all_finite(metadata) or not _tree_all_finite(graphs):
                raise AssertionError(f"{policy_name}/{example.sample_id}: non-finite graph data")
            if policy_name != "legacy" and not _check_r2r_stable(metadata, legacy_metadata):
                raise AssertionError(f"{policy_name}/{example.sample_id}: r2r topology changed")
            policy_rows.append(
                {
                    "sample_id": example.sample_id,
                    "metadata_shape_signature": _shape_signature(metadata),
                    "graph_leaf_shape_signature": _shape_signature(graphs),
                    "edge_checks": edge_checks,
                    "r2r_stable_vs_legacy": (
                        True
                        if policy_name == "legacy"
                        else _check_r2r_stable(metadata, legacy_metadata)
                    ),
                    "metadata_all_finite": True,
                    "graph_all_finite": True,
                }
            )
        result[policy_name] = policy_rows
    return result


def _forward_check(examples: list[Any]) -> dict[str, Any]:
    stats = _train_only_stats(examples)
    result: dict[str, Any] = {}
    for policy_name in POLICIES:
        builder = _policy_builder(policy_name)
        start = time.perf_counter()
        groups = _make_groups(examples, stats, builder)
        graph_build_time = time.perf_counter() - start
        model = GraphNeuralOperator(**MODEL_CONFIG)
        first_group = groups[0]
        params = model.init(
            jax.random.PRNGKey(0),
            inputs=first_group["inputs"],
            graphs=first_group["graphs"],
        )["params"]

        outputs = []
        finite = True
        shape_ok = True
        for group in groups:
            pred = model.apply({"params": params}, inputs=group["inputs"], graphs=group["graphs"])
            outputs.append(
                {
                    "group_name": group["name"],
                    "sample_ids": list(group["sample_ids"]),
                    "output_shape": list(pred.shape),
                    "target_shape": list(group["target_normalized"].shape),
                    "shared_metadata": bool(group["shared_metadata"]),
                }
            )
            finite = finite and bool(jnp.all(jnp.isfinite(pred)))
            shape_ok = shape_ok and pred.shape == group["target_normalized"].shape
            _check_metadata_edges(group["metadata"])

        if not finite or not shape_ok:
            raise AssertionError(f"{policy_name}: model forward failed finite/shape checks")
        result[policy_name] = {
            "group_count": len(groups),
            "graph_build_time_seconds": float(graph_build_time),
            "finite": finite,
            "shape_ok": shape_ok,
            "outputs": outputs,
        }
    return result


def _real_coverage_summary(args: argparse.Namespace) -> dict[str, Any]:
    records = audit_real_dataset(
        subset=args.subset,
        split_map=None,
        splits=["all"],
        max_samples=args.max_forward_samples,
        seeds=[0],
        policies=[policy["audit_policy"] for policy in POLICIES.values()],
        k_encoding_mode=args.k_encoding_mode,
        boundary_mask_fallback=True,
    )
    summary = summarize_records(records)
    for policy_name, policy in POLICIES.items():
        audit_policy = policy["audit_policy"]
        values = summary[audit_policy]
        if policy_name != "legacy" and not values["coverage_gate_passed"]:
            raise AssertionError(f"{policy_name}: real coverage gate failed")
        if not values["all_metadata_finite"] or not values["all_graphs_finite"]:
            raise AssertionError(f"{policy_name}: real graph finite gate failed")
        if not values["all_node_and_r2r_stable_vs_legacy"]:
            raise AssertionError(f"{policy_name}: real r2r stability gate failed")
    return {
        "records": records,
        "summary": summary,
    }


def main() -> int:
    args = parse_args()
    if args.max_forward_samples < 1:
        raise ValueError("--max-forward-samples must be >= 1")

    synthetic_coords, _, _ = _synthetic_grid(SYNTHETIC_GRID_SHAPES[0])
    synthetic_equivalence = _assert_default_discrete_equivalence(synthetic_coords)
    synthetic = _assert_synthetic_baseline()

    examples, split_ids, split_source = _load_forward_examples(args)
    real_equivalence = _assert_default_discrete_equivalence(examples[0].condition.coords)
    metadata_checks = _metadata_checks_for_examples(examples)
    forward = _forward_check(examples)
    real_coverage = _real_coverage_summary(args)

    payload = {
        "schema_version": "heat3d_v3_graph_policy_hardening_v1",
        "diagnostic_scope": "graph policy hardening and batch forward only; no training",
        "config": {
            "subset": str(args.subset),
            "manifest": str(args.manifest),
            "k_encoding_mode": args.k_encoding_mode,
            "max_forward_samples": args.max_forward_samples,
            "selected_sample_ids": [example.sample_id for example in examples],
            "split_source": split_source,
            "split_counts": {split: len(ids) for split, ids in split_ids.items()},
            "policies": POLICIES,
        },
        "default_discrete_equivalence": {
            "synthetic": synthetic_equivalence,
            "real_first_sample": real_equivalence,
        },
        "synthetic": synthetic,
        "real_coverage": real_coverage,
        "metadata_checks": metadata_checks,
        "forward": forward,
        "status": {
            "default_discrete_equivalent": True,
            "synthetic_legacy_baseline_matches_p0": True,
            "all_policy_forward_passed": True,
            "dummy_and_index_checks_passed": True,
            "finite_and_shape_checks_passed": True,
            "r2r_topology_stable": True,
        },
    }
    output_path = write_json_ignored(args.output_json, payload)

    print("Heat3D v3 graph policy hardening")
    print(f"  subset: {_sample_root(args.subset)}")
    print(f"  selected sample ids: {[example.sample_id for example in examples]}")
    print("  default discrete equivalence: passed")
    print("  synthetic current baseline: passed")
    for policy_name, policy in POLICIES.items():
        audit_policy = policy["audit_policy"]
        summary = real_coverage["summary"][audit_policy]
        print(
            f"  {policy_name}: forward=passed "
            f"p2r_zero={summary['p2r_zero_count_total']} "
            f"r2p_zero={summary['r2p_zero_count_total']} "
            f"edge_ratio={summary['p2r_edge_ratio_vs_legacy']:.3f}/"
            f"{summary['r2p_edge_ratio_vs_legacy']:.3f}"
        )
    print(f"wrote={output_path}")
    print("Heat3D v3 graph policy hardening passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
