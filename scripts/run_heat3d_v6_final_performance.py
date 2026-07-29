#!/usr/bin/env python3
"""Direct, non-additive V6 production/evaluation timing for one role/resolution."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import resource
import sys
import tempfile
import time
from typing import Any, Mapping

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
import run_heat3d_v1_medium_controlled_training_export as runner  # noqa: E402
import run_heat3d_v6_production_highres_inference as production  # noqa: E402
from rigno.graphBuilder_Heat3D import Heat3DGraphBuilder  # noqa: E402
from rigno.heat3d_graph_cache import graph_builder_code_fingerprint  # noqa: E402
from rigno.heat3d_v6_full_field import (  # noqa: E402
    FullFieldMetricAccumulator,
    build_reconstruction_map,
    load_reconstruction_map,
    save_reconstruction_map,
)
from rigno.models.rigno import RIGNO as GraphNeuralOperator  # noqa: E402
from run_heat3d_v3_final_probe_checkpoint_smoke import (  # noqa: E402
    install_checkpoint_feature_hooks,
)


class FinalPerformanceError(RuntimeError):
    pass


def _rss_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if sys.platform == "darwin" else value * 1024


def _distribution(values: list[float]) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
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


def _mapping(
    *,
    dataset: Path,
    examples: list[anchored.AnchoredExample],
    cache_dir: Path,
    support_hash: str,
    rebuild: bool,
):
    with h5py.File(dataset / "full_fields.h5", "r") as handle:
        coords = np.asarray(handle["mesh/coords"], dtype=np.float64)
        layer_id = np.asarray(handle["mesh/layer_id"], dtype=np.int32)
        boundaries = np.asarray(handle["mesh/boundaries"], dtype=np.float64)
        control_volume = np.asarray(
            handle["mesh/control_volume"], dtype=np.float64
        )
    lookup = {tuple(row): index for index, row in enumerate(coords)}
    support_indices = np.asarray(
        [lookup[tuple(row)] for row in np.asarray(examples[0].condition.coords)],
        dtype=np.int32,
    )
    path = cache_dir / f"reconstruction_{len(support_indices)}_{support_hash}.npz"
    cache_dir.mkdir(parents=True, exist_ok=True)
    build_seconds = None
    save_seconds = None
    load_seconds = None
    if rebuild or not path.is_file():
        started = time.perf_counter()
        mapping, audit = build_reconstruction_map(
            coords=coords,
            layer_id=layer_id,
            boundaries=boundaries,
            support_indices=support_indices,
        )
        build_seconds = time.perf_counter() - started
        started = time.perf_counter()
        save_reconstruction_map(path, mapping)
        save_seconds = time.perf_counter() - started
    started = time.perf_counter()
    mapping, _ = load_reconstruction_map(path)
    load_seconds = time.perf_counter() - started
    if not np.array_equal(mapping.support_indices, support_indices):
        raise FinalPerformanceError("reconstruction support binding drifted")
    return mapping, {
        "path": str(path),
        "cache_hit": build_seconds is None,
        "build_seconds": build_seconds,
        "save_seconds": save_seconds,
        "load_seconds": load_seconds,
        "label_independent": True,
    }, {
        "coords": coords,
        "layer_id": layer_id,
        "boundaries": boundaries,
        "control_volume": control_volume,
        "support_indices": support_indices,
    }


def _reconstruct(
    *,
    examples: list[anchored.AnchoredExample],
    predictions: Mapping[str, np.ndarray],
    mapping,
) -> tuple[np.ndarray, float]:
    started = time.perf_counter()
    full = np.stack(
        [
            mapping.reconstruct(
                np.asarray(predictions[example.sample_id], dtype=np.float64)
            )
            for example in examples
        ],
        axis=0,
    ).astype(np.float32)
    return full, float(time.perf_counter() - started)


def _serialize(path: Path, sample_ids: list[str], full: np.ndarray) -> float:
    path.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    with path.open("wb") as handle:
        np.savez(
            handle,
            sample_ids=np.asarray(sample_ids, dtype="S32"),
            temperature_K=full,
        )
    return float(time.perf_counter() - started)


def _evaluate(
    *,
    dataset: Path,
    manifest: Path,
    role: str,
    examples: list[anchored.AnchoredExample],
    support_predictions: Mapping[str, np.ndarray],
    full_predictions: np.ndarray,
    public: Mapping[str, Any],
    full_public: Mapping[str, np.ndarray],
    parent_split_role: str,
    role_sample_ids: tuple[str, ...] | None,
) -> tuple[dict[str, Any], dict[str, float]]:
    manifest_payload = json.loads(manifest.read_text(encoding="utf-8"))
    rows = [
        row
        for row in manifest_payload["samples"]
        if row["split_role"] == parent_split_role
    ]
    if role_sample_ids is not None:
        selected = set(role_sample_ids)
        rows = [row for row in rows if str(row["sample_id"]) in selected]
    expected_count = len(role_sample_ids) if role_sample_ids is not None else 128
    if len(rows) != expected_count:
        raise FinalPerformanceError(f"{role} population drifted")
    example_by_id = {example.sample_id: example for example in examples}
    sample_order = [example.sample_id for example in examples]
    if sample_order != [str(row["sample_id"]) for row in rows]:
        raise FinalPerformanceError("prediction/manifest order drifted")
    accumulator = FullFieldMetricAccumulator(
        control_volume=full_public["control_volume"],
        layer_id=full_public["layer_id"],
        boundaries=full_public["boundaries"],
        coords=full_public["coords"],
    )
    label_started = time.perf_counter()
    targets: dict[str, dict[str, np.ndarray]] = {}
    full_truth: dict[str, np.ndarray] = {}
    full_q: dict[str, np.ndarray] = {}
    with h5py.File(dataset / "full_fields.h5", "r") as handle:
        ids = [
            value.decode() if isinstance(value, bytes) else str(value)
            for value in handle["samples/sample_id"][:]
        ]
        lookup = {sample_id: index for index, sample_id in enumerate(ids)}
        for row in rows:
            sample_id = str(row["sample_id"])
            index = lookup[sample_id]
            ref = float(
                example_by_id[sample_id].meta["v6_adapter"][
                    "reference_temperature_K"
                ]
            )
            truth = (
                np.asarray(
                    handle["samples/temperature_K"][index], dtype=np.float64
                )
                - ref
            )
            q = np.asarray(
                handle["samples/q_W_m3"][index], dtype=np.float64
            )
            full_truth[sample_id] = truth
            full_q[sample_id] = q
            targets[sample_id] = {
                "deltaT_K": truth[full_public["support_indices"]],
                "q_W_m3": q[full_public["support_indices"]],
            }
    label_seconds = time.perf_counter() - label_started
    metric_started = time.perf_counter()
    for index, sample_id in enumerate(sample_order):
        ref = float(
            example_by_id[sample_id].meta["v6_adapter"][
                "reference_temperature_K"
            ]
        )
        accumulator.add(
            kind="model",
            sample_id=sample_id,
            prediction_delta=np.asarray(
                full_predictions[index], dtype=np.float64
            )
            - ref,
            truth_delta=full_truth[sample_id],
            q=full_q[sample_id],
        )
    full_metrics = accumulator.summarize("model")
    support_metrics = source_eval._metrics(
        predictions=support_predictions,
        examples=source_eval._query_context_examples(examples),
        targets=targets,
        public=public,
        resolution=int(len(examples[0].condition.coords)),
    )
    metric_seconds = time.perf_counter() - metric_started
    return {
        "support": {
            key: value
            for key, value in support_metrics.items()
            if key != "per_sample"
        },
        "full_field": full_metrics,
    }, {
        "label_read_seconds": float(label_seconds),
        "metrics_diagnostics_seconds": float(metric_seconds),
    }


def _cycle(
    *,
    args,
    checkpoint,
    run_config,
    stats,
    model,
    params,
    ladder,
    graph_fingerprint: str,
    cycle_index: int,
    rebuild: bool,
) -> dict[str, Any]:
    production_started = time.perf_counter()
    split_role = args.parent_split_role
    probe = ladder["probes"][str(args.resolution)]
    anchor_probe = ladder["probes"]["1024"]
    input_started = time.perf_counter()
    examples, targets, public = anchored._load_examples(
        args.dataset,
        args.manifest,
        probe,
        split_role=split_role,
        load_labels=False,
    )
    anchor_examples, _, _ = anchored._load_examples(
        args.dataset,
        args.manifest,
        anchor_probe,
        split_role=split_role,
        load_labels=False,
    )
    if args.role_sample_ids is not None:
        selected = set(args.role_sample_ids)
        examples = [
            example for example in examples if example.sample_id in selected
        ]
        anchor_examples = [
            example
            for example in anchor_examples
            if example.sample_id in selected
        ]
        expected = tuple(args.role_sample_ids)
        if tuple(example.sample_id for example in examples) != expected:
            raise FinalPerformanceError("query role-manifest order drifted")
        if tuple(example.sample_id for example in anchor_examples) != expected:
            raise FinalPerformanceError("anchor role-manifest order drifted")
        public = dict(public)
        public["evaluation_sample_ids"] = list(expected)
    if targets or public["labels_loaded"]:
        raise FinalPerformanceError("production input path loaded labels")
    input_seconds = time.perf_counter() - input_started

    anchor_config = dict(run_config["graph_config"])
    anchor_config.update(
        {
            "discrete_graph_backend": "sparse_kdtree_v1",
            "discrete_graph_chunk_size": 1024,
            "discrete_coverage_multiplier": 1.0,
        }
    )
    anchor_config = dict(Heat3DGraphBuilder(**anchor_config).config)
    query_config = dict(anchor_config)
    query_config["subsample_factor"] = 8
    query_config = dict(Heat3DGraphBuilder(**query_config).config)
    graph_started = time.perf_counter()
    anchor_builder, anchor_metadata, anchor_cache, _ = (
        production._build_or_load_graph(
            example=anchor_examples[0],
            stats=stats,
            graph_config=anchor_config,
            graph_seed=int(run_config["graph_seed"]),
            support_hash=anchor_probe["indices_sha256"],
            graph_builder_fingerprint=graph_fingerprint,
            cache_dir=args.graph_cache,
            rebuild=rebuild,
            audit_equivalence=False,
        )
    )
    if args.resolution == 1024:
        query_builder, query_metadata, query_cache = (
            anchor_builder,
            anchor_metadata,
            anchor_cache,
        )
    else:
        query_builder, query_metadata, query_cache, _ = (
            production._build_or_load_graph(
                example=examples[0],
                stats=stats,
                graph_config=query_config,
                graph_seed=int(run_config["graph_seed"]),
                support_hash=probe["indices_sha256"],
                graph_builder_fingerprint=graph_fingerprint,
                cache_dir=args.graph_cache,
                rebuild=rebuild,
                audit_equivalence=False,
            )
        )
    graph_seconds = time.perf_counter() - graph_started

    prepare_started = time.perf_counter()
    runtime_checkpoint = dict(checkpoint)
    runtime_checkpoint["train_only_normalization"] = stats
    anchor_groups = production._prepare_cached_groups(
        examples=anchor_examples,
        run_config=run_config,
        checkpoint=runtime_checkpoint,
        builder=anchor_builder,
        metadata=anchor_metadata,
        batch_size=args.batch_size,
    )
    query_groups = (
        anchor_groups
        if args.resolution == 1024
        else production._prepare_cached_groups(
            examples=examples,
            run_config=run_config,
            checkpoint=runtime_checkpoint,
            builder=query_builder,
            metadata=query_metadata,
            batch_size=args.batch_size,
        )
    )
    batch_prepare_seconds = time.perf_counter() - prepare_started

    model_started = time.perf_counter()
    anchor_predictions, anchor_scales, anchor_runtime = (
        production._predict_groups(
            model=model,
            params=params,
            groups=anchor_groups,
            warm_repeats=0,
            expected_sample_count=len(anchor_examples),
        )
    )
    if args.resolution == 1024:
        query_predictions = anchor_predictions
        query_runtime = anchor_runtime
        scale_seconds = 0.0
    else:
        joint_predictions, _, query_runtime = production._predict_groups(
            model=model,
            params=params,
            groups=query_groups,
            warm_repeats=0,
            expected_sample_count=len(examples),
        )
        scale_started = time.perf_counter()
        query_predictions = anchored._anchor_scale_predictions(
            joint_predictions,
            anchor_scales,
            examples,
            np.asarray(public["control_volume"], dtype=np.float64),
        )
        scale_seconds = time.perf_counter() - scale_started
    model_core_seconds = time.perf_counter() - model_started

    mapping, map_audit, full_public = _mapping(
        dataset=args.dataset,
        examples=examples,
        cache_dir=args.reconstruction_cache,
        support_hash=probe["indices_sha256"],
        rebuild=rebuild,
    )
    full_predictions, reconstruction_seconds = _reconstruct(
        examples=examples,
        predictions=query_predictions,
        mapping=mapping,
    )
    serialization_path = (
        args.serialization_dir
        / f"{args.role}_{args.resolution}_{args.mode}_{cycle_index:02d}.npz"
    )
    serialization_seconds = _serialize(
        serialization_path,
        [example.sample_id for example in examples],
        full_predictions,
    )
    production_seconds = time.perf_counter() - production_started

    metrics, evaluation_phases = _evaluate(
        dataset=args.dataset,
        manifest=args.manifest,
        role=args.role,
        examples=examples,
        support_predictions=query_predictions,
        full_predictions=full_predictions,
        public=public,
        full_public=full_public,
        parent_split_role=args.parent_split_role,
        role_sample_ids=args.role_sample_ids,
    )
    evaluation_seconds = time.perf_counter() - production_started
    serialization_path.unlink(missing_ok=True)
    return {
        "cycle_index": int(cycle_index),
        "mode": args.mode,
        "direct_timing_seconds": {
            "model_core": float(model_core_seconds),
            "full_field_production": float(production_seconds),
            "evaluation": float(evaluation_seconds),
        },
        "production_phase_seconds": {
            "input": float(input_seconds),
            "graph_load_or_build": float(graph_seconds),
            "batch_prepare": float(batch_prepare_seconds),
            "anchor_forward": float(
                anchor_runtime["formal_inference_seconds"]
            ),
            "query_forward": float(
                0.0
                if args.resolution == 1024
                else query_runtime["formal_inference_seconds"]
            ),
            "scale_reconstruction": float(scale_seconds),
            "full_field_reconstruction": float(reconstruction_seconds),
            "serialization": float(serialization_seconds),
        },
        "evaluation_phase_seconds": evaluation_phases,
        "graph_cache": {"anchor": anchor_cache, "query": query_cache},
        "reconstruction_cache": map_audit,
        "metrics": metrics,
        "process_peak_ram_bytes": _rss_bytes(),
        "device_memory": production._device_memory(),
        "production_labels_loaded": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--ladder", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--resolution", type=int, required=True)
    parser.add_argument(
        "--role",
        choices=("valid_iid", "test_iid", "hard_input_stress"),
        required=True,
    )
    parser.add_argument("--role-manifest", type=Path)
    parser.add_argument("--mode", choices=("cold", "cached", "persistent"), required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--persistent-repeats", type=int, default=3)
    parser.add_argument("--graph-cache", type=Path, required=True)
    parser.add_argument("--reconstruction-cache", type=Path, required=True)
    parser.add_argument("--serialization-dir", type=Path, required=True)
    parser.add_argument("--platform", choices=("cpu", "gpu"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.role == "valid_iid":
        if args.role_manifest is not None:
            raise FinalPerformanceError("valid_iid forbids a role manifest")
        args.parent_split_role = "valid"
        args.role_sample_ids = None
        role_manifest_sha256 = None
    elif args.role == "test_iid":
        if args.role_manifest is not None:
            raise FinalPerformanceError("test_iid forbids a role manifest")
        args.parent_split_role = "test"
        args.role_sample_ids = None
        role_manifest_sha256 = None
    else:
        if args.role_manifest is None:
            raise FinalPerformanceError("hard_input_stress requires a role manifest")
        role_payload = json.loads(args.role_manifest.read_text(encoding="utf-8"))
        if (
            role_payload.get("role_id") != "hard_input_stress_corner_v1"
            or role_payload.get("parent_split_role") != "test"
            or role_payload.get("selection_uses_target_labels") is not False
        ):
            raise FinalPerformanceError("hard role manifest contract drifted")
        args.parent_split_role = "test"
        args.role_sample_ids = tuple(
            str(value) for value in role_payload["sample_ids"]
        )
        if len(args.role_sample_ids) != len(set(args.role_sample_ids)):
            raise FinalPerformanceError("hard role manifest contains duplicates")
        role_manifest_sha256 = common._sha256(args.role_manifest)
    if args.resolution not in {4096, 8192, 16384, 32768}:
        raise FinalPerformanceError("resolution outside frozen final ladder")
    actual_platform = jax.devices()[0].platform
    if args.platform == "cpu" and actual_platform != "cpu":
        raise FinalPerformanceError("CPU requested but JAX selected non-CPU")
    if args.platform == "gpu" and actual_platform not in {"gpu", "cuda"}:
        raise FinalPerformanceError("GPU requested but CUDA is unavailable")
    spec = anchored.SEED_SPECS["seed0"]
    checkpoint_path = args.run_dir / "params_best_valid_point_global.pkl"
    if common._sha256(checkpoint_path) != spec["sha256"]:
        raise FinalPerformanceError("canonical checkpoint SHA256 drifted")
    checkpoint = runner._load_params_checkpoint(checkpoint_path)
    run_config = json.loads((args.run_dir / "run_config.json").read_text())
    stats = common._materialize_checkpoint_stats(
        checkpoint["train_only_normalization"]
    )
    install_checkpoint_feature_hooks(stats)
    model_config = runner._resolve_decoder_bypass_model_config(
        dict(checkpoint["model_config"]), stats
    )
    model = GraphNeuralOperator(**model_config)
    params = runner._device_params(checkpoint["params"])
    ladder = json.loads(args.ladder.read_text(encoding="utf-8"))
    graph_fingerprint = graph_builder_code_fingerprint()
    cycles = []
    if args.mode == "persistent":
        _cycle(
            args=args,
            checkpoint=checkpoint,
            run_config=run_config,
            stats=stats,
            model=model,
            params=params,
            ladder=ladder,
            graph_fingerprint=graph_fingerprint,
            cycle_index=0,
            rebuild=False,
        )
        for index in range(1, args.persistent_repeats + 1):
            cycles.append(
                _cycle(
                    args=args,
                    checkpoint=checkpoint,
                    run_config=run_config,
                    stats=stats,
                    model=model,
                    params=params,
                    ladder=ladder,
                    graph_fingerprint=graph_fingerprint,
                    cycle_index=index,
                    rebuild=False,
                )
            )
    else:
        cycles.append(
            _cycle(
                args=args,
                checkpoint=checkpoint,
                run_config=run_config,
                stats=stats,
                model=model,
                params=params,
                ladder=ladder,
                graph_fingerprint=graph_fingerprint,
                cycle_index=1,
                rebuild=args.mode == "cold",
            )
        )
    for cycle in cycles:
        if not (
            cycle["direct_timing_seconds"]["model_core"]
            <= cycle["direct_timing_seconds"]["full_field_production"]
            <= cycle["direct_timing_seconds"]["evaluation"]
        ):
            raise FinalPerformanceError("timing category nesting failed")
    timing_summary = {
        key: _distribution(
            [cycle["direct_timing_seconds"][key] for cycle in cycles]
        )
        for key in ("model_core", "full_field_production", "evaluation")
    }
    sample_count = int(cycles[0]["metrics"]["full_field"]["sample_count"])
    payload = {
        "schema_version": "heat3d_v6_final_performance_direct_v1",
        "status": "passed",
        "role": args.role,
        "resolution": args.resolution,
        "mode": args.mode,
        "platform": args.platform,
        "batch_size": args.batch_size,
        "sample_count": sample_count,
        "parent_split_role": args.parent_split_role,
        "role_manifest_sha256": role_manifest_sha256,
        "checkpoint": {
            "config_id": spec["config_id"],
            "epoch": spec["epoch"],
            "sha256": spec["sha256"],
            "modified": False,
        },
        "graph_builder_code_fingerprint": graph_fingerprint,
        "timing_contract": {
            "model_core": "anchor/query forward + scale reconstruction",
            "full_field_production": (
                "input + graph load/build + batch prepare + model-core + "
                "240825-node reconstruction + serialization"
            ),
            "evaluation": "full-field production + label read + metrics/diagnostics",
            "direct_single_cycle_measurements": True,
            "cross_run_phase_addition": False,
        },
        "timing_summary_seconds": timing_summary,
        "cycles": cycles,
        "hard_accessed": args.role == "hard_input_stress",
        "ood_accessed": False,
        "training_executed": False,
        "checkpoint_modified": False,
        "environment": {
            "device": str(jax.devices()[0]),
            "jax": jax.__version__,
            "python": sys.version,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": "passed",
                "role": args.role,
                "resolution": args.resolution,
                "mode": args.mode,
                "model_core_s": timing_summary["model_core"]["mean"],
                "production_s": timing_summary["full_field_production"]["mean"],
                "evaluation_s": timing_summary["evaluation"]["mean"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
