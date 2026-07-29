#!/usr/bin/env python3
"""Smoke run-level shared-support graph reuse through one optimizer update."""

from __future__ import annotations

import argparse
import contextlib
import io
import json
from pathlib import Path
import sys
import time

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
    parser.add_argument("--startup-request-count", type=int, default=104)
    parser.add_argument("--output", type=Path)
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

    request_count = int(args.startup_request_count)
    if request_count < 3:
        raise ValueError("startup request count must cover train/valid/prediction")
    run_cached_builder = runner.RunSharedSupportGraphBuilder(
        Heat3DGraphBuilder(**graph_config)
    )
    cache_started = time.perf_counter()
    for index in range(request_count):
        batch_count = 8 if index < request_count - 8 else 32
        metadata, _ = runner._build_batch_metadata_with_seed(
            run_cached_builder,
            [coords[0]] * batch_count,
            graph_seed=int(run_config["graph_seed"]),
        )
        run_cached_builder.build_graphs(metadata)
    cached_startup_seconds = time.perf_counter() - cache_started
    legacy_startup_builder = Heat3DGraphBuilder(**graph_config)
    legacy_started = time.perf_counter()
    for index in range(request_count):
        batch_count = 8 if index < request_count - 8 else 32
        metadata, _ = runner._build_batch_metadata_with_seed(
            legacy_startup_builder,
            [coords[0]] * batch_count,
            graph_seed=int(run_config["graph_seed"]),
        )
        legacy_startup_builder.build_graphs(metadata)
    legacy_startup_seconds = time.perf_counter() - legacy_started
    fallback_builder = runner.RunSharedSupportGraphBuilder(
        Heat3DGraphBuilder(**graph_config)
    )
    fallback_builder.build_metadata(
        coords[0], key=runner._metadata_key(run_config["graph_seed"])
    )
    changed_coords = np.asarray(coords[0]).copy()
    changed_coords[0, 0] = np.nextafter(changed_coords[0, 0], np.inf)
    fallback_metadata = fallback_builder.build_metadata(
        changed_coords, key=runner._metadata_key(run_config["graph_seed"])
    )
    direct_fallback_metadata = Heat3DGraphBuilder(**graph_config).build_metadata(
        changed_coords, key=runner._metadata_key(run_config["graph_seed"])
    )
    fallback_equal = (
        metadata_hash(fallback_metadata)
        == metadata_hash(direct_fallback_metadata)
    )
    if (
        fallback_builder.audit["varying_support_fallback_calls"] != 1
        or not fallback_equal
    ):
        raise AssertionError("varying-support fallback changed legacy metadata")

    runtime_checkpoint = dict(checkpoint)
    runtime_checkpoint["train_only_normalization"] = stats
    install_checkpoint_feature_hooks(stats)
    with contextlib.redirect_stdout(io.StringIO()):
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
    loss_config = dict(run_config["loss"])
    edge_key = runner._training_edge_masking_key(
        model_config, model_seed=int(run_config["model_seed"]), epoch=1, batch_index=1
    )

    def loss_and_grad(group):
        def loss_fn(current_params):
            components = runner._loss_components(
                model, current_params, [group], stats, loss_config, key=edge_key
            )
            return components["total_loss"]

        return jax.value_and_grad(loss_fn)(params)

    legacy_loss, legacy_grad = loss_and_grad(legacy_group)
    reused_loss, reused_grad = loss_and_grad(reused_group)

    def tree_max_abs(left, right):
        values = [
            float(np.max(np.abs(np.asarray(a) - np.asarray(b))))
            for a, b in zip(
                jax.tree_util.tree_leaves(left),
                jax.tree_util.tree_leaves(right),
            )
        ]
        return max(values, default=0.0)

    loss_error = float(abs(float(legacy_loss) - float(reused_loss)))
    gradient_error = tree_max_abs(legacy_grad, reused_grad)
    optimizer_config = dict(run_config["optimizer_config"])
    lr_config = dict(run_config["lr_config"])
    lr_config["updates_per_epoch"] = int(run_config["updates_per_epoch"])

    def one_update(grads):
        state = runner._build_optax_state(
            params,
            epochs=int(run_config["epochs"]),
            lr_config=lr_config,
            optimizer_config=optimizer_config,
        )
        updates, _ = state["tx"].update(grads, state["state"], params)
        updates = runner._apply_native_update_controls(
            updates,
            native_enabled=model_config.get("native_output_mode")
            == "native_shape_scale",
            model_config=model_config,
            optimizer_config=optimizer_config,
        )
        return state["apply_updates"](params, updates)

    legacy_update_started = time.perf_counter()
    legacy_params = one_update(legacy_grad)
    legacy_update_seconds = time.perf_counter() - legacy_update_started
    reused_update_started = time.perf_counter()
    reused_params = one_update(reused_grad)
    reused_update_seconds = time.perf_counter() - reused_update_started
    update_error = tree_max_abs(legacy_params, reused_params)
    if max(loss_error, gradient_error, update_error) != 0.0:
        raise AssertionError("loss/gradient/update changed under run graph reuse")
    payload = {
        "status": "passed",
        "batch_size": 8,
        "legacy_metadata_build_calls": 8,
        "reused_metadata_build_calls": counting.calls,
        "run_level_cache_audit": dict(run_cached_builder.audit),
        "varying_support_fallback": {
            "audit": dict(fallback_builder.audit),
            "metadata_hash_equal_to_legacy": bool(fallback_equal),
        },
        "startup_request_contract": {
            "request_count": request_count,
            "train_requests_B8": request_count - 8,
            "valid_prediction_requests_B32": 8,
        },
        "startup_seconds": {
            "legacy": float(legacy_startup_seconds),
            "run_shared_support_cache": float(cached_startup_seconds),
            "speedup": float(legacy_startup_seconds / cached_startup_seconds),
            "seconds_removed_before_first_epoch": float(
                legacy_startup_seconds - cached_startup_seconds
            ),
        },
        "metadata_hash": metadata_hash(reused),
        "graph_hash": graph_hash(reused_graphs),
        "forward_max_abs_error_K": max_abs,
        "loss_abs_error": loss_error,
        "gradient_max_abs_error": gradient_error,
        "updated_parameter_max_abs_error": update_error,
        "optimizer_update_seconds": {
            "legacy": float(legacy_update_seconds),
            "run_shared_support_cache": float(reused_update_seconds),
        },
        "optimizer_update_executed": True,
        "optimizer_update_persisted": False,
        "formal_training_executed": False,
        "test_hard_accessed": False,
    }
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
