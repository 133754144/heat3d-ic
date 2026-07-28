#!/usr/bin/env python3
"""Smoke shared-support one-build graph reuse without an optimizer update."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import jax
import jax.numpy as jnp
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
for value in (ROOT, ROOT / "scripts"):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))

import evaluate_heat3d_v6_anchored_resolution as anchored  # noqa: E402
import evaluate_heat3d_v6_common_valid_probe as common  # noqa: E402
from rigno.graphBuilder_Heat3D import Heat3DGraphBuilder  # noqa: E402
from rigno.heat3d_graph_cache import graph_hash, metadata_hash  # noqa: E402
from rigno.models.rigno import RIGNO as GraphNeuralOperator  # noqa: E402
import run_heat3d_v1_medium_controlled_training_export as runner  # noqa: E402
import run_heat3d_v6_production_highres_inference as production  # noqa: E402
from run_heat3d_v3_final_probe_checkpoint_smoke import (  # noqa: E402
    install_checkpoint_feature_hooks,
)


class CountingBuilder:
    def __init__(self, builder: Heat3DGraphBuilder):
        self.builder = builder
        self.calls = 0

    def build_metadata(self, coords, key=None):
        self.calls += 1
        return self.builder.build_metadata(coords, key=key)

    def build_graphs(self, metadata):
        return self.builder.build_graphs(metadata)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--ladder", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    ladder = json.loads(args.ladder.read_text(encoding="utf-8"))
    examples, _, _ = anchored._load_examples(
        args.dataset, args.manifest, ladder["probes"]["1024"]
    )
    examples = examples[:8]
    checkpoint = runner._load_params_checkpoint(
        args.run_dir / "params_best_valid_point_global.pkl"
    )
    stats = common._materialize_checkpoint_stats(
        checkpoint["train_only_normalization"]
    )
    run_config = json.loads((args.run_dir / "run_config.json").read_text())
    graph_config = dict(run_config["graph_config"])
    graph_config["discrete_graph_backend"] = "sparse_kdtree_v1"
    builder = Heat3DGraphBuilder(**graph_config)
    coords = [runner._graph_coords_for_example(row, stats) for row in examples]
    if not all(np.array_equal(coords[0], row) for row in coords[1:]):
        raise AssertionError("fixture is not shared support")

    counting = CountingBuilder(builder)
    reused, shared = runner._build_batch_metadata_with_seed(
        counting, coords, graph_seed=int(run_config["graph_seed"])
    )
    if not shared or counting.calls != 1:
        raise AssertionError("shared support did not build metadata exactly once")

    legacy_rows = [
        builder.build_metadata(row, key=runner._metadata_key(run_config["graph_seed"]))
        for row in coords
    ]
    legacy = jax.tree_util.tree_map(
        lambda value: jnp.repeat(value, len(coords), axis=0),
        legacy_rows[0],
    )
    if metadata_hash(reused) != metadata_hash(legacy):
        raise AssertionError("reused metadata differs from legacy repeated metadata")
    reused_graphs = builder.build_graphs(reused)
    legacy_graphs = builder.build_graphs(legacy)
    if graph_hash(reused_graphs) != graph_hash(legacy_graphs):
        raise AssertionError("reused graph differs from legacy graph")

    runtime_checkpoint = dict(checkpoint)
    runtime_checkpoint["train_only_normalization"] = stats
    install_checkpoint_feature_hooks(stats)
    groups = production._prepare_cached_groups(
        examples=examples,
        run_config=run_config,
        checkpoint=runtime_checkpoint,
        builder=builder,
        metadata=legacy_rows[0],
        batch_size=8,
    )
    model_config = runner._resolve_decoder_bypass_model_config(
        dict(checkpoint["model_config"]), stats
    )
    model = GraphNeuralOperator(**model_config)
    params = runner._device_params(checkpoint["params"])
    reused_group = dict(groups[0])
    legacy_group = dict(groups[0])
    reused_group["graphs"] = reused_graphs
    legacy_group["graphs"] = legacy_graphs
    reused_output = runner._model_apply(model, params, reused_group)
    legacy_output = runner._model_apply(model, params, legacy_group)
    jax.block_until_ready(reused_output["raw_temperature"])
    jax.block_until_ready(legacy_output["raw_temperature"])
    error = np.asarray(reused_output["raw_temperature"]) - np.asarray(
        legacy_output["raw_temperature"]
    )
    max_abs = float(np.max(np.abs(error)))
    if max_abs != 0.0:
        raise AssertionError("runner forward changed under shared-support reuse")
    print(
        json.dumps(
            {
                "status": "passed",
                "batch_size": 8,
                "legacy_metadata_build_calls": 8,
                "reused_metadata_build_calls": counting.calls,
                "metadata_hash": metadata_hash(reused),
                "graph_hash": graph_hash(reused_graphs),
                "forward_max_abs_error_K": max_abs,
                "optimizer_update_executed": False,
                "training_executed": False,
                "test_hard_accessed": False,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
