#!/usr/bin/env python3
"""Continuous process-cold/warm B1 timing for frozen P1i full-grid policies.

This worker deliberately performs no metric, oracle, hash, equivalence, or
serialization work until the production span has ended.
"""

from __future__ import annotations

import time
PROCESS_STARTED = time.perf_counter()

import argparse
import json
import os
from pathlib import Path
import sys

import jax
import jax.numpy as jnp
import numpy as np

ROOT = Path(os.environ.get("HEAT3D_REPO_ROOT", Path(__file__).resolve().parents[1])).resolve()
for value in (ROOT, ROOT / "scripts"):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))

import run_heat3d_v6_p1i_anchor_high_n_development as highn  # noqa: E402
import run_heat3d_v1_medium_controlled_training_export as legacy  # noqa: E402
import run_heat3d_v6_p1i_graph_scale_candidate as candidate  # noqa: E402
from rigno.graphBuilder_Heat3D import Heat3DGraphBuilder  # noqa: E402
from rigno.models.rigno import RIGNO as GraphNeuralOperator  # noqa: E402

IMPORT_FINISHED = time.perf_counter()


def distribution(values: list[float]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(len(array)), "median_seconds": float(np.median(array)),
        "p95_seconds": float(np.percentile(array, 95)),
        "mean_seconds": float(np.mean(array)), "std_seconds": float(np.std(array)),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", choices=["B", "E"], required=True)
    parser.add_argument("--optimization-mode", choices=["baseline", "reference", "shared_reverse", "gpu_tiled", "combined"], default="baseline")
    parser.add_argument("--binding", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--full-fields", type=Path, required=True)
    parser.add_argument("--preflight", type=Path, required=True)
    parser.add_argument("--native-predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--warm-repeats", type=int, default=20)
    args = parser.parse_args()

    phases: dict[str, float] = {"python_and_module_import_seconds": IMPORT_FINISHED - PROCESS_STARTED}
    phase = time.perf_counter()
    binding = json.loads(args.binding.read_text())
    sample_id = binding["development_subset"]["sample_ids"][0]
    dataset = highn.Heat3DV6DualRobinDataset(args.dataset_root, args.manifest, include_roles={"valid_iid"})
    anchor = dataset[dataset.sample_index_by_id()[sample_id]]
    with np.load(args.native_predictions, allow_pickle=False) as native:
        ids = [str(value) for value in np.asarray(native["sample_ids"]).tolist()]
        anchor_scale = float(np.asarray(native["predicted_scales"])[ids.index(sample_id)])
    preflight = json.loads(args.preflight.read_text())
    physics_path = Path(next(row for row in preflight["samples"] if row["sample_id"] == sample_id)["physics_cache_file"])
    full, _ = highn._full_shared(args)
    with np.load(physics_path, allow_pickle=False) as physics:
        support = {
            "selected_indices": np.arange(len(full["coords"]), dtype=np.int64),
            "operator_control_volume": np.asarray(full["cv"], dtype=np.float64),
            "k_xyz": np.asarray(physics["k_xyz"], dtype=np.float64),
            "q_W_m3": np.asarray(physics["q_W_m3"], dtype=np.float64),
            "layer_id": np.asarray(full["layer"], dtype=np.int32),
        }
    example = highn._query_example(anchor, support, full["coords"])
    phases["input_and_support_prepare_seconds"] = time.perf_counter() - phase

    phase = time.perf_counter()
    checkpoint = legacy._load_params_checkpoint(args.run_dir / "params_best_valid_point_global.pkl")
    run_config = json.loads((args.run_dir / "run_config.json").read_text())
    stats = highn.common._materialize_checkpoint_stats(checkpoint["train_only_normalization"])
    highn.install_checkpoint_feature_hooks(stats)
    model_config = legacy._resolve_decoder_bypass_model_config(dict(checkpoint["model_config"]), stats)
    graph_config = dict(run_config["graph_config"])
    graph_config["discrete_graph_backend"] = "sparse_kdtree_v1"
    graph_config = dict(Heat3DGraphBuilder(**graph_config).config)
    runtime = {"checkpoint": checkpoint, "run_config": run_config, "stats": stats, "model_config": model_config, "graph_config": graph_config}
    phases["checkpoint_and_runtime_load_seconds"] = time.perf_counter() - phase

    graph_key = highn.runner._metadata_key(int(run_config["graph_seed"]))
    builder = candidate._builder(
        args.candidate, anchor=anchor, runtime=runtime, graph_key=graph_key,
        physical_node_count=240825, optimization_mode=args.optimization_mode,
    )
    phase = time.perf_counter()
    metadata = builder.build_metadata(highn.runner._graph_coords_for_example(example, stats), key=graph_key)
    jax.block_until_ready(metadata.r_rnodes)
    phases["graph_construction_seconds"] = time.perf_counter() - phase
    phases["graph_stage"] = dict(builder.builder.last_build_timings)

    phase = time.perf_counter()
    edge_targets = {}
    for field in candidate.qualification.EDGE_FIELDS:
        value = getattr(metadata, field)
        edge_targets[field] = None if value is None else int(value.shape[1])
    group = highn._prepare_group(
        example=example, anchor=anchor, runtime=runtime, builder=builder,
        metadata=metadata, edge_targets=edge_targets,
    )
    group = highn._model_group(group)
    weights = jnp.asarray(example.operator_point_weights, dtype=jnp.float32)
    phases["packing_padding_seconds"] = time.perf_counter() - phase

    model = GraphNeuralOperator(**model_config)
    phase = time.perf_counter()
    params = highn.runner._device_params(checkpoint["params"])
    group, weights, scale = jax.device_put((group, weights, jnp.asarray(anchor_scale)))
    jax.block_until_ready(weights)
    phases["h2d_seconds"] = time.perf_counter() - phase

    @jax.jit
    def apply(model_params, model_group, model_weights, frozen_scale):
        output = highn.runner._model_apply(model, model_params, model_group)
        raw = output["raw_temperature"][0, 0, :, 0]
        delta = raw - highn.REFERENCE_K
        normalized = model_weights / jnp.sum(model_weights)
        query_scale = jnp.sqrt(jnp.sum(normalized * delta * delta))
        return delta / query_scale * frozen_scale

    phase = time.perf_counter()
    prediction = apply(params, group, weights, scale)
    jax.block_until_ready(prediction)
    phases["jit_plus_first_forward_and_sync_seconds"] = time.perf_counter() - phase
    production_finished = time.perf_counter()

    warm = []
    for _ in range(args.warm_repeats):
        phase = time.perf_counter()
        prediction = apply(params, group, weights, scale)
        jax.block_until_ready(prediction)
        warm.append(time.perf_counter() - phase)
    timing_finished = time.perf_counter()

    payload = {
        "schema_version": "heat3d_v6_p1i_full_grid_production_timing_v1",
        "status": "passed" if bool(np.all(np.isfinite(np.asarray(prediction)))) else "failed_nonfinite",
        "candidate": args.candidate,
        "optimization_mode": args.optimization_mode,
        "resolution": 240825,
        "sample_id": sample_id,
        "timing": {
            "process_cold_continuous_seconds": production_finished - PROCESS_STARTED,
            "process_cold_span": "process entry through synchronized first prediction",
            "decomposition": phases,
            "warm_resident_neural_forward": distribution(warm),
            "warm_timing_total_seconds_excluded_from_process_cold": timing_finished - production_finished,
        },
        "memory": candidate.publication._device_memory(),
        "direct_full_grid_output": True,
        "excluded_from_production_span": ["metrics", "oracle", "hash", "equivalence", "serialization"],
        "role_contract": {"training": False, "test": False, "sealed": False, "valid_iid": True},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True, default=lambda value: value.item() if isinstance(value, np.generic) else value) + "\n")
    print(json.dumps({"status": payload["status"], "process_cold_seconds": payload["timing"]["process_cold_continuous_seconds"]}))
    return 0 if payload["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
