#!/usr/bin/env python3
"""Read-only audit of the Heat3D one-sample RIGNO graph/model input path."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import jax
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"
for path in (REPO_ROOT, SCRIPTS_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from check_heat3d_v1_small_train_valid_smoke import _make_batch_group, _train_only_stats
from rigno.graphBuilder_Heat3D import Heat3DGraphBuilder
from rigno.heat3d_v1_native_supervised import Heat3DV1NativeSupervisedDataset
from rigno.models.rigno import RIGNO as GraphNeuralOperator


DEFAULT_SUBSET = Path(
    "data/heat3d-thermal-simulation/subsets/"
    "v1_multilayer_bc_eq_physics_label_medium1024_gapA_full1024_v2"
)
DEFAULT_SPLIT = Path("configs/heat3d_v2/medium1024_gapA_memorization_train1_seed0.json")
MODEL_CONFIG = {
    "num_outputs": 1,
    "processor_steps": 6,
    "node_latent_size": 128,
    "edge_latent_size": 128,
    "mlp_hidden_layers": 2,
    "concatenate_tau": False,
    "concatenate_t": False,
    "conditioned_normalization": False,
    "cond_norm_hidden_size": 16,
    "p_edge_masking": 0.0,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subset", type=Path, default=DEFAULT_SUBSET)
    parser.add_argument("--split-map", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument(
        "--boundary-mask-fallback",
        dest="boundary_mask_fallback",
        action="store_true",
        default=True,
    )
    parser.add_argument(
        "--no-boundary-mask-fallback",
        dest="boundary_mask_fallback",
        action="store_false",
    )
    return parser.parse_args()


def _sample_root(path: Path) -> Path:
    samples = path / "samples"
    return samples if samples.is_dir() else path


def _train_ids(path: Path) -> list[str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    mapping = payload.get("sample_splits", payload)
    return sorted(sample_id for sample_id, split in mapping.items() if split == "train")


def _column_stats(values: np.ndarray, names: tuple[str, ...]) -> list[dict[str, Any]]:
    flat = np.asarray(values, dtype=np.float64).reshape(-1, len(names))
    return [
        {
            "name": name,
            "min": float(np.min(flat[:, index])),
            "max": float(np.max(flat[:, index])),
            "mean": float(np.mean(flat[:, index])),
            "std": float(np.std(flat[:, index])),
            "all_zero": bool(np.all(flat[:, index] == 0.0)),
            "constant": bool(np.std(flat[:, index]) < 1.0e-12),
        }
        for index, name in enumerate(names)
    ]


def _edge_coverage(edges: np.ndarray, sender_count: int, receiver_count: int) -> dict[str, Any]:
    values = np.asarray(edges).reshape(-1, 2).astype(np.int64)
    valid = values[
        (values[:, 0] >= 0)
        & (values[:, 0] < sender_count)
        & (values[:, 1] >= 0)
        & (values[:, 1] < receiver_count)
    ]
    sender_degree = np.bincount(valid[:, 0], minlength=sender_count) if valid.size else np.zeros(sender_count)
    receiver_degree = np.bincount(valid[:, 1], minlength=receiver_count) if valid.size else np.zeros(receiver_count)
    return {
        "edge_count_excluding_dummy": int(valid.shape[0]),
        "isolated_sender_count": int(np.sum(sender_degree == 0)),
        "isolated_receiver_count": int(np.sum(receiver_degree == 0)),
        "sender_degree_min": int(np.min(sender_degree)),
        "sender_degree_max": int(np.max(sender_degree)),
        "receiver_degree_min": int(np.min(receiver_degree)),
        "receiver_degree_max": int(np.max(receiver_degree)),
    }


def audit(args: argparse.Namespace) -> dict[str, Any]:
    train_ids = _train_ids(args.split_map)
    if len(train_ids) != 1:
        raise ValueError(f"{args.split_map}: expected exactly one train sample")
    dataset = Heat3DV1NativeSupervisedDataset(
        _sample_root(args.subset),
        k_encoding_mode="diag3",
        boundary_mask_fallback=args.boundary_mask_fallback,
    )
    example = dataset[dataset.sample_index_by_id()[train_ids[0]]]
    bridge = example.build_temperature_rise_legacy_inputs_from_relative_features(
        bridge_policy="zero_delta_u_bridge"
    )
    stats = _train_only_stats([example])
    builder = Heat3DGraphBuilder()
    group = _make_batch_group("train_one", [example], stats, builder)
    metadata = group["metadata"]

    physical_node_count = int(example.condition.coords.shape[0])
    regional_node_count = int(np.asarray(metadata.x_rnodes).shape[1] - 1)
    p2r = _edge_coverage(metadata.p2r_edge_indices, physical_node_count, regional_node_count)
    r2r = _edge_coverage(metadata.r2r_edge_indices, regional_node_count, regional_node_count)
    r2p = _edge_coverage(metadata.r2p_edge_indices, regional_node_count, physical_node_count)

    model = GraphNeuralOperator(**MODEL_CONFIG)
    variables = model.init(
        jax.random.PRNGKey(0),
        inputs=group["inputs"],
        graphs=group["graphs"],
    )
    pred = model.apply(variables, inputs=group["inputs"], graphs=group["graphs"])
    target = np.asarray(group["target_normalized"])
    pred_array = np.asarray(pred)

    input_feature_names = tuple(bridge.condition_feature_names)
    raw_condition = np.asarray(bridge.legacy_inputs.c)
    normalized_condition = np.asarray(group["inputs"].c)
    raw_coords = np.asarray(bridge.legacy_inputs.x_inp)
    normalized_coords = np.asarray(group["inputs"].x_inp)
    payload = {
        "diagnostic_scope": "one-sample graph/model path audit; no training",
        "sample_id": train_ids[0],
        "boundary_mask_fallback": bool(args.boundary_mask_fallback),
        "bridge": "relative_bc_features + zero_delta_u_bridge",
        "model_config": MODEL_CONFIG,
        "parameter_count": int(
            sum(np.asarray(leaf).size for leaf in jax.tree_util.tree_leaves(variables["params"]))
        ),
        "feature_names": list(input_feature_names),
        "raw_condition_shape": list(raw_condition.shape),
        "normalized_condition_shape": list(normalized_condition.shape),
        "raw_condition_stats": _column_stats(raw_condition, input_feature_names),
        "normalized_condition_stats": _column_stats(normalized_condition, input_feature_names),
        "legacy_u_shape": list(np.asarray(group["inputs"].u).shape),
        "legacy_u_all_zero": bool(np.all(np.asarray(group["inputs"].u) == 0.0)),
        "raw_coords_shape": list(raw_coords.shape),
        "raw_coords_min": np.min(raw_coords.reshape(-1, 3), axis=0).tolist(),
        "raw_coords_max": np.max(raw_coords.reshape(-1, 3), axis=0).tolist(),
        "normalized_coords_min": np.min(normalized_coords.reshape(-1, 3), axis=0).tolist(),
        "normalized_coords_max": np.max(normalized_coords.reshape(-1, 3), axis=0).tolist(),
        "physical_node_count": physical_node_count,
        "regional_node_count": regional_node_count,
        "graph_builder_config": builder.config,
        "p2r_coverage": p2r,
        "r2r_coverage": r2r,
        "r2p_coverage": r2p,
        "prediction_shape": list(pred_array.shape),
        "target_shape": list(target.shape),
        "prediction_target_shape_match": bool(pred_array.shape == target.shape),
        "loss_point_count": int(target.size),
        "initial_normalized_mse_all_nodes": float(np.mean(np.square(pred_array - target))),
        "prediction_finite": bool(np.all(np.isfinite(pred_array))),
        "target_finite": bool(np.all(np.isfinite(target))),
    }
    return payload


def main() -> int:
    args = parse_args()
    payload = audit(args)
    zero_or_constant = [
        row["name"]
        for row in payload["raw_condition_stats"]
        if row["all_zero"] or row["constant"]
    ]
    print(
        f"sample={payload['sample_id']} pnodes={payload['physical_node_count']} "
        f"rnodes={payload['regional_node_count']} params={payload['parameter_count']}"
    )
    print(
        f"p2r_isolated_pnodes={payload['p2r_coverage']['isolated_sender_count']} "
        f"r2p_isolated_pnodes={payload['r2p_coverage']['isolated_receiver_count']} "
        f"shape_match={payload['prediction_target_shape_match']}"
    )
    print(f"zero_or_constant_raw_features={zero_or_constant}")
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("Heat3D v2 one-sample graph path audit passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
