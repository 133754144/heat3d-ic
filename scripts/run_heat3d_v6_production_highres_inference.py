#!/usr/bin/env python3
"""Production V6 anchor-conditioned high-resolution valid_iid inference."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time
from typing import Any, Mapping, Sequence

import h5py
import jax
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
for value in (ROOT, ROOT / "scripts"):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))

import evaluate_heat3d_v6_anchored_resolution as anchored  # noqa: E402
import evaluate_heat3d_v6_common_valid_probe as common  # noqa: E402
import evaluate_heat3d_v6_source_aware_resolution as source_eval  # noqa: E402
from rigno.graphBuilder_Heat3D import Heat3DGraphBuilder  # noqa: E402
from rigno.heat3d_graph_cache import (  # noqa: E402
    cache_key,
    cache_key_payload,
    file_sha256,
    graph_builder_code_fingerprint,
    graph_hash,
    load_metadata,
    metadata_hash,
    save_metadata,
)
from rigno.heat3d_v6_full_field import (  # noqa: E402
    FullFieldMetricAccumulator,
    build_reconstruction_map,
    load_reconstruction_map,
    save_reconstruction_map,
)
from rigno.models.rigno import RIGNO as GraphNeuralOperator  # noqa: E402
import run_heat3d_v1_medium_controlled_training_export as runner  # noqa: E402
from run_heat3d_v3_final_probe_checkpoint_smoke import (  # noqa: E402
    install_checkpoint_feature_hooks,
)


class ProductionInferenceError(RuntimeError):
    pass


def _git_head() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()


def _rss_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if sys.platform == "darwin" else value * 1024


def _device_memory() -> dict[str, Any]:
    device = jax.devices()[0]
    try:
        stats = device.memory_stats() or {}
    except Exception:
        stats = {}
    return {
        "device": str(device),
        "platform": device.platform,
        "bytes_in_use": stats.get("bytes_in_use"),
        "peak_bytes_in_use": stats.get("peak_bytes_in_use"),
        "bytes_limit": stats.get("bytes_limit"),
    }


def _block(output: Mapping[str, Any]) -> None:
    jax.block_until_ready(output["raw_temperature"])


def _distribution(values: Sequence[float]) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    if len(array) == 0:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "p95": None,
            "std": None,
            "min": None,
            "max": None,
            "values": [],
        }
    return {
        "count": int(len(array)),
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "p95": float(np.quantile(array, 0.95)),
        "std": float(np.std(array, ddof=1)) if len(array) > 1 else 0.0,
        "min": float(np.min(array)),
        "max": float(np.max(array)),
        "values": array.tolist(),
    }


class _CachedBuilder:
    def __init__(self, builder: Heat3DGraphBuilder, metadata: Any):
        self._builder = builder
        self._metadata = metadata

    def build_metadata(self, coords: np.ndarray, key=None):
        del coords, key
        return self._metadata

    def build_graphs(self, metadata):
        return self._builder.build_graphs(metadata)


def _build_or_load_graph(
    *,
    example: anchored.AnchoredExample,
    stats: Mapping[str, Any],
    graph_config: Mapping[str, Any],
    graph_seed: int,
    support_hash: str,
    graph_builder_fingerprint: str,
    cache_dir: Path,
    rebuild: bool,
    audit_equivalence: bool,
) -> tuple[Heat3DGraphBuilder, Any, dict[str, Any], Any | None]:
    key_payload = cache_key_payload(
        support_hash=support_hash,
        graph_config=graph_config,
        graph_seed=graph_seed,
        graph_builder_fingerprint=graph_builder_fingerprint,
    )
    key = cache_key(key_payload)
    path = cache_dir / f"graph_{len(example.condition.coords)}_{key}.npz"
    builder = Heat3DGraphBuilder(**dict(graph_config))
    normalized_coords = runner._graph_coords_for_example(example, dict(stats))
    built_metadata = None
    build_seconds = None
    save_audit = None
    if rebuild or not path.is_file():
        started = time.perf_counter()
        built_metadata = builder.build_metadata(
            normalized_coords, key=runner._metadata_key(graph_seed)
        )
        build_seconds = time.perf_counter() - started
        save_audit = save_metadata(path, built_metadata)
    loaded_metadata, load_audit = load_metadata(path)
    loaded_graphs = builder.build_graphs(loaded_metadata)
    loaded_graph_hash = graph_hash(loaded_graphs)
    equivalence = {
        "audited": False,
        "metadata_hash_equal": None,
        "graph_hash_equal": None,
    }
    if audit_equivalence:
        if built_metadata is None:
            started = time.perf_counter()
            built_metadata = builder.build_metadata(
                normalized_coords, key=runner._metadata_key(graph_seed)
            )
            build_seconds = time.perf_counter() - started
        built_graphs = builder.build_graphs(built_metadata)
        equivalence = {
            "audited": True,
            "metadata_hash_equal": metadata_hash(built_metadata)
            == metadata_hash(loaded_metadata),
            "graph_hash_equal": graph_hash(built_graphs) == loaded_graph_hash,
        }
        if not all(equivalence.values()):
            raise ProductionInferenceError("cached/uncached graph equivalence failed")
    return builder, loaded_metadata, {
        "cache_key": key,
        "cache_key_payload": key_payload,
        "cache_file": str(path),
        "cache_file_sha256": file_sha256(path),
        "cache_file_bytes": path.stat().st_size,
        "cache_hit": not rebuild and save_audit is None,
        "uncached_build_seconds": (
            float(build_seconds) if build_seconds is not None else None
        ),
        "save": save_audit,
        "load_seconds": load_audit["load_seconds"],
        "metadata_hash": load_audit["metadata_hash"],
        "graph_hash": loaded_graph_hash,
        "equivalence": equivalence,
    }, built_metadata


def _prepare_cached_groups(
    *,
    examples: Sequence[anchored.AnchoredExample],
    run_config: Mapping[str, Any],
    checkpoint: Mapping[str, Any],
    builder: Heat3DGraphBuilder,
    metadata: Any,
    batch_size: int,
) -> list[dict[str, Any]]:
    stats = checkpoint["train_only_normalization"]
    model_config = runner._resolve_decoder_bypass_model_config(
        dict(checkpoint["model_config"]), stats
    )
    cached_builder = _CachedBuilder(builder, metadata)
    groups = runner._make_v6_padded_groups_with_progress(
        examples,
        stats,
        cached_builder,
        "v6_production_anchor_conditioned",
        True,
        "basic",
        int(run_config["graph_seed"]),
        batch_size=batch_size,
        drop_last=False,
    )
    context_payload = run_config["global_context"]
    standardizer = context_payload["standardizer"]
    if (
        standardizer["fit_population"] != "train_only"
        or int(standardizer["fit_sample_count"]) != 768
    ):
        raise ProductionInferenceError("Global Context standardizer is not train-only")
    encoded = {
        example.sample_id: common.standardize_v6_contexts(
            [runner._global_context_row_for_example(example)], standardizer
        )[0]
        for example in examples
    }
    runner._attach_global_context_to_groups(
        groups,
        encoded,
        expected_feature_dim=int(model_config["global_context_feature_dim"]),
    )
    examples_by_id = {example.sample_id: example for example in examples}
    runner._attach_native_physics_to_groups(groups, examples_by_id)
    if (
        model_config.get("scale_pooling") == "qk_gated"
        or model_config.get("shape_attention_mode") != "none"
        or model_config.get("scale_attention_mode") != "none"
    ):
        runner._attach_qk_region_features_to_groups(
            groups,
            examples_by_id,
            feature_version=model_config["qk_region_feature_version"],
        )
    return groups


def _predict_groups(
    *,
    model: Any,
    params: Any,
    groups: Sequence[Mapping[str, Any]],
    warm_repeats: int,
) -> tuple[dict[str, np.ndarray], dict[str, float], dict[str, Any]]:
    first_started = time.perf_counter()
    first_output = runner._model_apply(model, params, groups[0])
    _block(first_output)
    first_seconds = time.perf_counter() - first_started
    warm = []
    for _ in range(warm_repeats):
        started = time.perf_counter()
        output = runner._model_apply(model, params, groups[0])
        _block(output)
        warm.append(time.perf_counter() - started)
    predictions: dict[str, np.ndarray] = {}
    scales: dict[str, float] = {}

    def collect(group: Mapping[str, Any], output: Mapping[str, Any]) -> None:
        raw = np.asarray(output["raw_temperature"], dtype=np.float64)
        scale = np.asarray(output["s_hat"], dtype=np.float64).reshape(
            len(group["sample_ids"])
        )
        for index, sample_id in enumerate(group["sample_ids"]):
            predictions[str(sample_id)] = raw[index, 0, :, 0]
            scales[str(sample_id)] = float(scale[index])

    collect(groups[0], first_output)
    formal_started = time.perf_counter()
    for group in groups[1:]:
        output = runner._model_apply(model, params, group)
        _block(output)
        collect(group, output)
    remaining_seconds = time.perf_counter() - formal_started
    if len(predictions) != 128:
        raise ProductionInferenceError("valid_iid prediction count drifted")
    sample_count = sum(len(group["sample_ids"]) for group in groups)
    formal = first_seconds + remaining_seconds
    return predictions, scales, {
        "group_count": len(groups),
        "sample_count": sample_count,
        "first_compile_inference_seconds": float(first_seconds),
        "warm_repeat_count": warm_repeats,
        "warm_inference_seconds": _distribution(warm),
        "formal_inference_seconds": float(formal),
        "samples_per_second": float(sample_count / formal),
    }


def _prediction_equivalence(
    *,
    model: Any,
    params: Any,
    group: Mapping[str, Any],
    builder: Heat3DGraphBuilder,
    cached_metadata: Any,
    fresh_metadata: Any | None,
) -> dict[str, Any]:
    if fresh_metadata is None:
        return {"audited": False, "reason": "fresh_metadata_not_requested"}
    batch_size = len(group["sample_ids"])

    def repeat_metadata(metadata):
        return jax.tree_util.tree_map(
            lambda value: (
                value
                if value is None or int(value.shape[0]) == batch_size
                else np.repeat(np.asarray(value), batch_size, axis=0)
            ),
            metadata,
        )

    cached_group = dict(group)
    fresh_group = dict(group)
    cached_group["graphs"] = builder.build_graphs(repeat_metadata(cached_metadata))
    fresh_group["graphs"] = builder.build_graphs(repeat_metadata(fresh_metadata))
    cached = runner._model_apply(model, params, cached_group)
    fresh = runner._model_apply(model, params, fresh_group)
    _block(cached)
    _block(fresh)
    error = np.asarray(cached["raw_temperature"]) - np.asarray(
        fresh["raw_temperature"]
    )
    max_abs = float(np.max(np.abs(error)))
    return {
        "audited": True,
        "max_abs_error_K": max_abs,
        "rmse_K": float(np.sqrt(np.mean(np.square(error)))),
        "passed": max_abs <= 1.0e-6,
    }


def _full_field_metrics(
    *,
    dataset: Path,
    manifest_path: Path,
    examples: Sequence[anchored.AnchoredExample],
    predictions: Mapping[str, np.ndarray],
    reconstruction_cache_dir: Path,
    support_hash: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    valid_rows = [row for row in manifest["samples"] if row["split_role"] == "valid"]
    if len(valid_rows) != 128:
        raise ProductionInferenceError("full-field valid_iid count drifted")
    example_by_id = {example.sample_id: example for example in examples}
    archive_path = dataset / "full_fields.h5"
    with h5py.File(archive_path, "r") as handle:
        coords = np.asarray(handle["mesh/coords"], dtype=np.float64)
        layer_id = np.asarray(handle["mesh/layer_id"], dtype=np.int32)
        boundaries = np.asarray(handle["mesh/boundaries"], dtype=np.float64)
        cv = np.asarray(handle["mesh/control_volume"], dtype=np.float64)
        ids = [
            value.decode() if isinstance(value, bytes) else str(value)
            for value in handle["samples/sample_id"][:]
        ]
        archive_index = {sample_id: index for index, sample_id in enumerate(ids)}
        support_coords = np.asarray(
            examples[0].condition.coords, dtype=np.float64
        )
        full_lookup = {
            tuple(row): index for index, row in enumerate(np.asarray(coords))
        }
        support_indices = np.asarray(
            [full_lookup[tuple(row)] for row in support_coords], dtype=np.int32
        )
        map_path = (
            reconstruction_cache_dir
            / f"reconstruction_{len(support_indices)}_{support_hash}.npz"
        )
        if map_path.is_file():
            mapping, map_io = load_reconstruction_map(map_path)
            map_build = None
        else:
            mapping, map_build = build_reconstruction_map(
                coords=coords,
                layer_id=layer_id,
                boundaries=boundaries,
                support_indices=support_indices,
            )
            map_io = save_reconstruction_map(map_path, mapping)
        if not np.array_equal(mapping.support_indices, support_indices):
            raise ProductionInferenceError("reconstruction support binding drifted")
        accumulator = FullFieldMetricAccumulator(
            control_volume=cv,
            layer_id=layer_id,
            boundaries=boundaries,
            coords=coords,
        )
        started = time.perf_counter()
        label_read_seconds = 0.0
        field_reconstruction_seconds = 0.0
        metric_seconds = 0.0
        for row in valid_rows:
            sample_id = str(row["sample_id"])
            archive_row = archive_index[sample_id]
            meta = example_by_id[sample_id].meta
            reference = float(meta["v6_adapter"]["reference_temperature_K"])
            phase_started = time.perf_counter()
            truth = (
                np.asarray(
                    handle["samples/temperature_K"][archive_row], dtype=np.float64
                )
                - reference
            )
            q = np.asarray(handle["samples/q_W_m3"][archive_row], dtype=np.float64)
            label_read_seconds += time.perf_counter() - phase_started
            phase_started = time.perf_counter()
            model_full = mapping.reconstruct(
                np.asarray(predictions[sample_id], dtype=np.float64)
            ) - reference
            floor_full = mapping.reconstruct(truth[support_indices])
            field_reconstruction_seconds += time.perf_counter() - phase_started
            phase_started = time.perf_counter()
            accumulator.add(
                kind="model",
                sample_id=sample_id,
                prediction_delta=model_full,
                truth_delta=truth,
                q=q,
            )
            metric_seconds += time.perf_counter() - phase_started
            accumulator.add(
                kind="sampling_floor",
                sample_id=sample_id,
                prediction_delta=floor_full,
                truth_delta=truth,
                q=q,
            )
        reconstruction_seconds = time.perf_counter() - started
    return {
        "model": accumulator.summarize("model"),
        "sampling_floor": accumulator.summarize("sampling_floor"),
    }, {
        "algorithm": "layer_interface_knn_inverse_distance_v1",
        "cache_file": str(map_path),
        "cache_file_sha256": file_sha256(map_path),
        "cache_load_or_save": map_io,
        "build": map_build,
        "reconstruction_and_metric_seconds_valid128": float(reconstruction_seconds),
        "label_read_seconds_valid128": float(label_read_seconds),
        "field_reconstruction_seconds_valid128": float(
            field_reconstruction_seconds
        ),
        "metric_seconds_valid128": float(metric_seconds),
        "label_independent": True,
        "test_hard_accessed": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--ladder", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--resolution", type=int, default=4096)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--reconstruction-cache-dir", type=Path, required=True)
    parser.add_argument("--graph-builder-fingerprint")
    parser.add_argument("--seed", choices=tuple(anchored.SEED_SPECS), default="seed0")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--warm-repeats", type=int, default=10)
    parser.add_argument("--persistent-workflow-repeats", type=int, default=1)
    parser.add_argument(
        "--graph-backend",
        choices=("dense_reference", "chunked_numpy_v1", "sparse_kdtree_v1"),
        default="chunked_numpy_v1",
    )
    parser.add_argument("--subsample-factor", type=int)
    parser.add_argument(
        "--query-subsample-factor",
        type=int,
        help=(
            "Override only the high-resolution query graph. The 1024 "
            "conditioning-anchor graph remains at the checkpoint/run-config value."
        ),
    )
    parser.add_argument("--coverage-multiplier", type=float, default=1.0)
    parser.add_argument("--graph-chunk-size", type=int, default=1024)
    parser.add_argument("--rebuild-cache", action="store_true")
    parser.add_argument("--audit-cache-equivalence", action="store_true")
    parser.add_argument("--platform", choices=("cpu", "gpu"), required=True)
    args = parser.parse_args()
    graph_fingerprint = graph_builder_code_fingerprint()
    if (
        args.graph_builder_fingerprint is not None
        and args.graph_builder_fingerprint != graph_fingerprint
    ):
        raise ProductionInferenceError("graph-builder code fingerprint drifted")
    actual_platform = jax.devices()[0].platform
    if args.platform == "cpu" and actual_platform != "cpu":
        raise ProductionInferenceError("requested CPU but JAX selected another platform")
    if args.platform == "gpu" and actual_platform not in {"gpu", "cuda"}:
        raise ProductionInferenceError("requested GPU but JAX did not select CUDA")
    input_started = time.perf_counter()
    ladder = json.loads(args.ladder.read_text(encoding="utf-8"))
    if args.resolution not in {1024, 2048, 4096, 8192, 16384, 32768}:
        raise ProductionInferenceError("resolution is outside registered ladder")
    probe = ladder["probes"][str(args.resolution)]
    anchor_probe = ladder["probes"]["1024"]
    examples, targets, public = anchored._load_examples(
        args.dataset.resolve(), args.manifest.resolve(), probe
    )
    anchor_examples, _, _ = anchored._load_examples(
        args.dataset.resolve(), args.manifest.resolve(), anchor_probe
    )
    spec = anchored.SEED_SPECS[args.seed]
    checkpoint_path = args.run_dir / "params_best_valid_point_global.pkl"
    if common._sha256(checkpoint_path) != spec["sha256"]:
        raise ProductionInferenceError("seed0 checkpoint SHA256 drifted")
    checkpoint = runner._load_params_checkpoint(checkpoint_path)
    run_config = json.loads((args.run_dir / "run_config.json").read_text())
    stats = common._materialize_checkpoint_stats(
        checkpoint["train_only_normalization"]
    )
    runtime_checkpoint = dict(checkpoint)
    runtime_checkpoint["train_only_normalization"] = stats
    install_checkpoint_feature_hooks(stats)
    anchor_graph_config = dict(run_config["graph_config"])
    anchor_graph_config["discrete_graph_backend"] = args.graph_backend
    anchor_graph_config["discrete_graph_chunk_size"] = args.graph_chunk_size
    anchor_graph_config["discrete_coverage_multiplier"] = args.coverage_multiplier
    if args.subsample_factor is not None:
        anchor_graph_config["subsample_factor"] = args.subsample_factor
    # Freeze every effective builder default into cache keys and result payloads.
    anchor_graph_config = dict(Heat3DGraphBuilder(**anchor_graph_config).config)
    query_graph_config = dict(anchor_graph_config)
    if args.query_subsample_factor is not None:
        query_graph_config["subsample_factor"] = args.query_subsample_factor
    query_graph_config = dict(Heat3DGraphBuilder(**query_graph_config).config)
    model_config = runner._resolve_decoder_bypass_model_config(
        dict(checkpoint["model_config"]), stats
    )
    model = GraphNeuralOperator(**model_config)
    params = runner._device_params(checkpoint["params"])
    input_seconds = time.perf_counter() - input_started
    started_total = time.perf_counter()
    anchor_builder, anchor_metadata, anchor_cache, anchor_fresh = _build_or_load_graph(
        example=anchor_examples[0],
        stats=stats,
        graph_config=anchor_graph_config,
        graph_seed=int(run_config["graph_seed"]),
        support_hash=anchor_probe["indices_sha256"],
        graph_builder_fingerprint=graph_fingerprint,
        cache_dir=args.cache_dir,
        rebuild=args.rebuild_cache,
        audit_equivalence=args.audit_cache_equivalence,
    )
    phase_started = time.perf_counter()
    anchor_groups = _prepare_cached_groups(
        examples=anchor_examples,
        run_config=run_config,
        checkpoint=runtime_checkpoint,
        builder=anchor_builder,
        metadata=anchor_metadata,
        batch_size=args.batch_size,
    )
    anchor_group_seconds = time.perf_counter() - phase_started
    anchor_predictions, anchor_scales, anchor_runtime = _predict_groups(
        model=model,
        params=params,
        groups=anchor_groups,
        warm_repeats=args.warm_repeats,
    )
    anchor_prediction_equivalence = _prediction_equivalence(
        model=model,
        params=params,
        group=anchor_groups[0],
        builder=anchor_builder,
        cached_metadata=anchor_metadata,
        fresh_metadata=anchor_fresh,
    )
    if args.resolution == 1024:
        query_predictions = anchor_predictions
        query_runtime = anchor_runtime
        query_cache = anchor_cache
        query_prediction_equivalence = anchor_prediction_equivalence
        query_groups = anchor_groups
        query_group_seconds = anchor_group_seconds
        scale_reconstruction_seconds = 0.0
    else:
        query_builder, query_metadata, query_cache, query_fresh = (
            _build_or_load_graph(
                example=examples[0],
                stats=stats,
                graph_config=query_graph_config,
                graph_seed=int(run_config["graph_seed"]),
                support_hash=probe["indices_sha256"],
                graph_builder_fingerprint=graph_fingerprint,
                cache_dir=args.cache_dir,
                rebuild=args.rebuild_cache,
                audit_equivalence=args.audit_cache_equivalence,
            )
        )
        phase_started = time.perf_counter()
        query_groups = _prepare_cached_groups(
            examples=examples,
            run_config=run_config,
            checkpoint=runtime_checkpoint,
            builder=query_builder,
            metadata=query_metadata,
            batch_size=args.batch_size,
        )
        query_group_seconds = time.perf_counter() - phase_started
        joint_predictions, _, query_runtime = _predict_groups(
            model=model,
            params=params,
            groups=query_groups,
            warm_repeats=args.warm_repeats,
        )
        query_prediction_equivalence = _prediction_equivalence(
            model=model,
            params=params,
            group=query_groups[0],
            builder=query_builder,
            cached_metadata=query_metadata,
            fresh_metadata=query_fresh,
        )
        phase_started = time.perf_counter()
        query_predictions = anchored._anchor_scale_predictions(
            joint_predictions,
            anchor_scales,
            examples,
            np.asarray(public["control_volume"], dtype=np.float64),
        )
        scale_reconstruction_seconds = time.perf_counter() - phase_started

    persistent_workflow_seconds = []
    for _ in range(args.persistent_workflow_repeats):
        phase_started = time.perf_counter()
        _, persistent_anchor_scales, _ = _predict_groups(
            model=model,
            params=params,
            groups=anchor_groups,
            warm_repeats=0,
        )
        if args.resolution > 1024:
            persistent_joint, _, _ = _predict_groups(
                model=model,
                params=params,
                groups=query_groups,
                warm_repeats=0,
            )
            anchored._anchor_scale_predictions(
                persistent_joint,
                persistent_anchor_scales,
                examples,
                np.asarray(public["control_volume"], dtype=np.float64),
            )
        persistent_workflow_seconds.append(time.perf_counter() - phase_started)

    phase_started = time.perf_counter()
    support_metrics = source_eval._metrics(
        predictions=query_predictions,
        examples=source_eval._query_context_examples(examples),
        targets=targets,
        public=public,
        resolution=args.resolution,
    )
    support_metric_seconds = time.perf_counter() - phase_started
    full_field_metrics, reconstruction = _full_field_metrics(
        dataset=args.dataset.resolve(),
        manifest_path=args.manifest.resolve(),
        examples=examples,
        predictions=query_predictions,
        reconstruction_cache_dir=args.reconstruction_cache_dir,
        support_hash=probe["indices_sha256"],
    )
    persistent_benchmark_seconds = float(sum(persistent_workflow_seconds))
    total_seconds = (
        time.perf_counter() - started_total - persistent_benchmark_seconds
    )
    device_memory = _device_memory()
    finite_values = [
        support_metrics["point_global_cv_relative_rmse_pct"],
        support_metrics["raw_cv_weighted_rmse_K"],
        full_field_metrics["model"]["cv_weighted_rmse_K"],
        full_field_metrics["model"]["cv_weighted_point_global_relative_rmse_pct"],
    ]
    if not all(np.isfinite(finite_values)):
        raise ProductionInferenceError("non-finite production metric")
    payload = {
        "schema_version": "heat3d_v6_production_highres_inference_v1",
        "status": "passed",
        "workflow": [
            "1024_anchor_forward",
            "anchor_derived_global_context_and_scale",
            f"{args.resolution}_node_source_aware_forward",
            "anchor_scale_reconstruction",
        ],
        "upstream_like_preforward_executed": False,
        "default_resolution": 4096,
        "maximum_verified_resolution": 16384,
        "resolution": args.resolution,
        "platform": args.platform,
        "batch_size": args.batch_size,
        "evaluation_role": "valid_iid",
        "test_hard_accessed": False,
        "training_executed": False,
        "checkpoint_modified": False,
        "checkpoint": {
            "seed": args.seed,
            "config_id": spec["config_id"],
            "epoch": spec["epoch"],
            "sha256": spec["sha256"],
        },
        "graph_config": {
            "anchor": anchor_graph_config,
            "query": query_graph_config,
        },
        "evaluator_commit": _git_head(),
        "graph_builder_code_fingerprint": graph_fingerprint,
        "graph_cache": {
            "anchor": anchor_cache,
            "query": query_cache,
            "anchor_prediction_equivalence": anchor_prediction_equivalence,
            "query_prediction_equivalence": query_prediction_equivalence,
        },
        "runtime": {
            "input_seconds": float(input_seconds),
            "anchor_group_prepare_seconds": float(anchor_group_seconds),
            "query_group_prepare_seconds": float(query_group_seconds),
            "anchor": anchor_runtime,
            "query": query_runtime,
            "checkpoint_pure_forward_cold_seconds": float(
                anchor_runtime["formal_inference_seconds"]
                + (
                    0.0
                    if args.resolution == 1024
                    else query_runtime["formal_inference_seconds"]
                )
            ),
            "scale_reconstruction_seconds": float(scale_reconstruction_seconds),
            "support_metric_seconds": float(support_metric_seconds),
            "persistent_compiled_workflow_seconds": _distribution(
                persistent_workflow_seconds
            ),
            "persistent_benchmark_seconds_excluded_from_end_to_end": (
                persistent_benchmark_seconds
            ),
            "end_to_end_seconds_valid128": float(total_seconds),
            "process_peak_ram_bytes": _rss_bytes(),
            "device_memory": device_memory,
        },
        "support_metrics": support_metrics,
        "full_field_metrics": full_field_metrics,
        "reconstruction": reconstruction,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    serialization_started = time.perf_counter()
    serialized = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    serialization_encode_seconds = time.perf_counter() - serialization_started
    serialization_started = time.perf_counter()
    args.output.write_text(serialized)
    serialization_write_seconds = time.perf_counter() - serialization_started
    payload["runtime"]["serialization_encode_seconds"] = float(
        serialization_encode_seconds
    )
    payload["runtime"]["serialization_write_seconds"] = float(
        serialization_write_seconds
    )
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "status": "passed",
                "resolution": args.resolution,
                "platform": args.platform,
                "point_global_pct": support_metrics[
                    "point_global_cv_relative_rmse_pct"
                ],
                "full_field_relative_pct": full_field_metrics["model"][
                    "cv_weighted_point_global_relative_rmse_pct"
                ],
                "end_to_end_seconds": total_seconds,
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
