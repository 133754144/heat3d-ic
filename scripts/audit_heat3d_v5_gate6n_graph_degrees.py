#!/usr/bin/env python3
"""Audit P5 train graph degrees and candidate r2r edge-mask rates."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rigno.graphBuilder_Heat3D import Heat3DGraphBuilder  # noqa: E402
from rigno.heat3d_v1_native_supervised import (  # noqa: E402
    Heat3DV1NativeSupervisedDataset,
)
from rigno.heat3d_v1_normalization import normalize_coords  # noqa: E402
from rigno.heat3d_v4_split_map import (  # noqa: E402
    load_sample_split_map,
    split_ids_from_sample_splits,
)
from scripts.check_heat3d_v4_registry import resolve_inherited_yaml  # noqa: E402


RATES = (0.02, 0.05, 0.10, 0.20, 0.50)
SEED_COUNT = 128
PLANNED_EPOCHS = 600
PLANNED_BATCHES_PER_EPOCH = 24


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-json")
    parser.add_argument("--output-md")
    return parser.parse_args()


def _resolved(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    resolved = resolve_inherited_yaml(payload, path)
    resolved["config_id"] = payload["config_id"]
    return resolved


def _digest(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).view(np.uint8)).hexdigest()


def _degree_summary(indices: np.ndarray, receiver_count: int) -> dict[str, Any]:
    receivers = np.asarray(indices, dtype=np.int64)[:, 1]
    degrees = np.bincount(receivers, minlength=receiver_count)[:receiver_count]
    return {
        "receiver_count": int(receiver_count),
        "edge_count_excluding_dummy": int(len(indices) - 1),
        "min_in_degree": int(degrees.min()),
        "median_in_degree": float(np.median(degrees)),
        "p95_in_degree": float(np.quantile(degrees, 0.95)),
        "max_in_degree": int(degrees.max()),
        "zero_in_degree_count": int(np.count_nonzero(degrees == 0)),
    }


def _masked_r2r_summary(
    indices: np.ndarray,
    receiver_count: int,
    rate: float,
) -> dict[str, Any]:
    edge_count = len(indices)
    kept = int((1.0 - rate) * edge_count)
    zero_counts: list[int] = []
    hashes: list[str] = []
    for seed in range(SEED_COUNT):
        order = np.asarray(
            jax.random.permutation(jax.random.PRNGKey(seed), edge_count)
        )
        selected = indices[order[:kept]]
        degrees = np.bincount(
            selected[:, 1].astype(np.int64), minlength=receiver_count
        )[:receiver_count]
        zero_counts.append(int(np.count_nonzero(degrees == 0)))
        hashes.append(_digest(selected))
    repeat_order = np.asarray(
        jax.random.permutation(jax.random.PRNGKey(0), edge_count)
    )
    repeat_hash = _digest(indices[repeat_order[:kept]])
    return {
        "rate": rate,
        "seed_count": SEED_COUNT,
        "kept_edge_count_including_dummy": kept,
        "zero_in_degree_max": max(zero_counts),
        "zero_in_degree_sum": sum(zero_counts),
        "same_seed_reproducible": hashes[0] == repeat_hash,
        "distinct_seed_changes_mask": len(set(hashes)) > 1,
    }


def _planned_key_schedule_summary(
    indices: np.ndarray,
    receiver_count: int,
    rate: float,
    model_seed: int,
) -> dict[str, Any]:
    receivers = jnp.asarray(indices[:, 1], dtype=jnp.int32)
    edge_count = len(indices)
    kept = int((1.0 - rate) * edge_count)
    keys = []
    base = jax.random.PRNGKey(model_seed)
    for epoch in range(1, PLANNED_EPOCHS + 1):
        epoch_key = jax.random.fold_in(base, epoch)
        for batch_index in range(1, PLANNED_BATCHES_PER_EPOCH + 1):
            keys.append(jax.random.fold_in(epoch_key, batch_index))
    packed_keys = jnp.stack(keys)

    @jax.jit
    def zero_counts(batch_keys):
        orders = jax.vmap(
            lambda key: jax.random.permutation(key, edge_count)
        )(batch_keys)
        selected_receivers = receivers[orders[:, :kept]]
        degrees = jax.vmap(
            lambda row: jnp.bincount(row, length=receiver_count)
        )(selected_receivers)
        return jnp.sum(degrees[:, :receiver_count] == 0, axis=1)

    chunk_size = 128
    rows: list[np.ndarray] = []
    for start in range(0, len(keys), chunk_size):
        chunk = packed_keys[start : start + chunk_size]
        actual = int(chunk.shape[0])
        if actual < chunk_size:
            chunk = jnp.concatenate(
                [chunk, jnp.repeat(chunk[-1:], chunk_size - actual, axis=0)],
                axis=0,
            )
        rows.append(np.asarray(zero_counts(chunk))[:actual])
    counts = np.concatenate(rows)
    return {
        "rate": rate,
        "model_seed": model_seed,
        "epochs": PLANNED_EPOCHS,
        "batches_per_epoch": PLANNED_BATCHES_PER_EPOCH,
        "mask_count": int(counts.size),
        "zero_in_degree_max": int(counts.max()),
        "zero_in_degree_sum": int(counts.sum()),
    }


def _main() -> dict[str, Any]:
    args = _args()
    config_path = (ROOT / args.config).resolve()
    config = _resolved(config_path)
    dataset_cfg = config["dataset"]
    graph_cfg = config["graph"]
    split_map = load_sample_split_map(ROOT / dataset_cfg["split_map_path"])
    train_ids = split_ids_from_sample_splits(split_map)["train"]
    dataset = Heat3DV1NativeSupervisedDataset(
        ROOT / dataset_cfg["subset_path"],
        k_encoding_mode="diag3",
        boundary_mask_fallback=True,
    )
    by_id = dataset.sample_index_by_id()
    coordinate_groups: dict[str, tuple[np.ndarray, list[str]]] = {}
    for sample_id in train_ids:
        coords = np.asarray(dataset[by_id[sample_id]].condition.coords)
        raw = coords.reshape(1, 1, coords.shape[0], coords.shape[1])
        normalized = np.asarray(
            normalize_coords(raw, {"coord_policy": "sample_local_isotropic"})
        ).reshape(coords.shape)
        digest = _digest(normalized)
        if digest not in coordinate_groups:
            coordinate_groups[digest] = (normalized, [])
        coordinate_groups[digest][1].append(sample_id)

    builder = Heat3DGraphBuilder(**graph_cfg)
    topologies: list[dict[str, Any]] = []
    for coordinate_hash, (coords, sample_ids) in sorted(coordinate_groups.items()):
        metadata = builder.build_metadata(
            coords, key=jax.random.PRNGKey(int(config["optimizer"]["graph_seed"]))
        )
        p2r = np.asarray(metadata.p2r_edge_indices[0], dtype=np.int64)
        r2r = np.asarray(metadata.r2r_edge_indices[0], dtype=np.int64)
        r2p_raw = metadata.r2p_edge_indices
        r2p = (
            np.asarray(r2p_raw[0], dtype=np.int64)
            if r2p_raw is not None
            else np.flip(p2r, axis=1)
        )
        physical_count = int(metadata.x_pnodes_inp.shape[1] - 1)
        regional_count = int(metadata.x_rnodes.shape[1] - 1)
        topologies.append(
            {
                "coordinate_hash": coordinate_hash,
                "sample_count": len(sample_ids),
                "first_sample_id": sample_ids[0],
                "physical_node_count": physical_count,
                "regional_node_count": regional_count,
                "degree": {
                    "p2r": _degree_summary(p2r, regional_count),
                    "r2r": _degree_summary(r2r, regional_count),
                    "r2p": _degree_summary(r2p, physical_count),
                },
                "r2r_mask_rate_audit": [
                    _masked_r2r_summary(r2r, regional_count, rate)
                    for rate in RATES
                ],
                "planned_e600_key_schedules": [
                    _planned_key_schedule_summary(
                        r2r,
                        regional_count,
                        rate,
                        int(config["optimizer"]["model_seed"]),
                    )
                    for rate in (0.02, 0.05, 0.10)
                ],
            }
        )

    payload = {
        "schema_version": "heat3d_v5_gate6n_graph_degree_audit_v1",
        "config_id": config["config_id"],
        "dataset": dataset_cfg["name"],
        "train_sample_count": len(train_ids),
        "node_count": 1024,
        "graph_seed": int(config["optimizer"]["graph_seed"]),
        "coordinate_topology_count": len(topologies),
        "mask_algorithm": "upstream_shuffle_then_prefix",
        "dummy_nodes_excluded_from_zero_degree_counts": True,
        "candidate_rates": list(RATES),
        "topologies": topologies,
    }
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    print(text, end="")
    if args.output_json:
        Path(args.output_json).write_text(text, encoding="utf-8")
    if args.output_md:
        lines = [
            "# Gate 6N P5 graph degree audit",
            "",
            f"- baseline: `{payload['config_id']}`",
            f"- train samples: {len(train_ids)}",
            f"- unique normalized-coordinate topologies: {len(topologies)}",
            "",
        ]
        for item in topologies:
            lines.append(
                f"- topology `{item['coordinate_hash'][:12]}`: "
                f"{item['sample_count']} samples, "
                f"r2r min degree={item['degree']['r2r']['min_in_degree']}"
            )
            for audit in item["r2r_mask_rate_audit"]:
                lines.append(
                    f"  - p={audit['rate']:.2f}: "
                    f"max zero-degree={audit['zero_in_degree_max']}, "
                    f"same-seed reproducible={audit['same_seed_reproducible']}"
                )
        Path(args.output_md).write_text("\n".join(lines) + "\n", encoding="utf-8")
    return payload


if __name__ == "__main__":
    _main()
