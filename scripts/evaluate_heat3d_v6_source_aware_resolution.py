#!/usr/bin/env python3
"""CPU-only valid_iid evaluation for one V6 source-aware resolution."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import resource
import sys
import time
from typing import Any, Mapping, Sequence

import jax
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
for value in (ROOT, ROOT / "scripts"):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))

import evaluate_heat3d_v6_anchored_resolution as anchored  # noqa: E402
import evaluate_heat3d_v6_common_valid_probe as common  # noqa: E402
import evaluate_heat3d_v6_volume_probe_ladder as volume  # noqa: E402
from rigno.models.rigno import RIGNO as GraphNeuralOperator  # noqa: E402
from rigno.heat3d_v6_global_context import standardize_v6_contexts  # noqa: E402
import run_heat3d_v1_medium_controlled_training_export as runner  # noqa: E402
from run_heat3d_v3_final_probe_checkpoint_smoke import install_checkpoint_feature_hooks  # noqa: E402


SEED_SPECS = anchored.SEED_SPECS
WARM_REPEATS = 10
UNACCEPTABLE_END_TO_END_SECONDS = 1800.0
UNACCEPTABLE_PEAK_RAM_BYTES = 12 * 2**30


class SourceAwareEvaluationError(RuntimeError):
    pass


def _rss_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if sys.platform == "darwin" else value * 1024


def _block(output: Mapping[str, Any]) -> None:
    jax.block_until_ready(output["raw_temperature"])


def _query_context_examples(
    examples: Sequence[anchored.AnchoredExample],
) -> list[volume.VolumeProbeV6Example]:
    return [
        volume.VolumeProbeV6Example(
            sample_id=row.sample_id,
            condition=row.condition,
            target=row.target,
            meta=row.meta,
            operator_point_weights=row.operator_point_weights,
        )
        for row in examples
    ]


def _predict_benchmark(
    *,
    run_dir: Path,
    spec: Mapping[str, Any],
    examples: Sequence[volume.VolumeProbeV6Example],
    warm_repeats: int,
) -> tuple[
    dict[str, np.ndarray],
    dict[str, float],
    dict[str, Any],
    dict[str, Any],
]:
    end_to_end_started = time.perf_counter()
    checkpoint_path = run_dir / "params_best_valid_point_global.pkl"
    run_config_path = run_dir / "run_config.json"
    summary_path = run_dir / "loss_summary.json"
    for path in (checkpoint_path, run_config_path, summary_path):
        if not path.is_file():
            raise SourceAwareEvaluationError(f"missing frozen artifact: {path}")
    if common._sha256(checkpoint_path) != spec["sha256"]:
        raise SourceAwareEvaluationError(f"{spec['config_id']}: checkpoint SHA drifted")
    checkpoint = runner._load_params_checkpoint(checkpoint_path)
    if int(checkpoint["epoch"]) != int(spec["epoch"]):
        raise SourceAwareEvaluationError("checkpoint epoch drifted")
    run_config = json.loads(run_config_path.read_text(encoding="utf-8"))
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if int(summary["point_global_best_epoch"]) != int(spec["epoch"]):
        raise SourceAwareEvaluationError("loss-summary checkpoint epoch drifted")
    stats = common._materialize_checkpoint_stats(checkpoint["train_only_normalization"])
    runtime_checkpoint = dict(checkpoint)
    runtime_checkpoint["train_only_normalization"] = stats
    install_checkpoint_feature_hooks(stats)
    rss_before = _rss_bytes()
    graph_started = time.perf_counter()
    groups = common._prepare_groups(
        examples=examples,
        run_config=run_config,
        checkpoint=runtime_checkpoint,
        batch_size=1,
    )
    graph_seconds = time.perf_counter() - graph_started
    if len(groups) != 128:
        raise SourceAwareEvaluationError("valid_iid batch=1 group count drifted")
    model_config = runner._resolve_decoder_bypass_model_config(
        dict(checkpoint["model_config"]), stats
    )
    model = GraphNeuralOperator(**model_config)
    params = runner._device_params(checkpoint["params"])
    rss_after_graph = _rss_bytes()

    first_started = time.perf_counter()
    first_output = runner._model_apply(model, params, groups[0])
    _block(first_output)
    first_compile_seconds = time.perf_counter() - first_started
    warm_seconds = []
    for _ in range(warm_repeats):
        started = time.perf_counter()
        warm_output = runner._model_apply(model, params, groups[0])
        _block(warm_output)
        warm_seconds.append(time.perf_counter() - started)

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
    remaining_started = time.perf_counter()
    for group in groups[1:]:
        output = runner._model_apply(model, params, group)
        _block(output)
        collect(group, output)
    remaining_seconds = time.perf_counter() - remaining_started
    formal_seconds = first_compile_seconds + remaining_seconds
    rss_after_inference = _rss_bytes()
    end_to_end_seconds = time.perf_counter() - end_to_end_started
    warm_benchmark_seconds = float(np.sum(np.asarray(warm_seconds, dtype=np.float64)))
    end_to_end_excluding_warm_seconds = end_to_end_seconds - warm_benchmark_seconds
    if len(predictions) != 128:
        raise SourceAwareEvaluationError("prediction count drifted")
    warm = np.asarray(warm_seconds, dtype=np.float64)
    warm_summary = (
        {
            "mean": float(np.mean(warm)),
            "median": float(np.median(warm)),
            "std": float(np.std(warm, ddof=1)),
            "min": float(np.min(warm)),
            "max": float(np.max(warm)),
            "values": warm.tolist(),
        }
        if warm_repeats
        else {
            "mean": None,
            "median": None,
            "std": None,
            "min": None,
            "max": None,
            "values": [],
        }
    )
    runtime = {
        "config_id": spec["config_id"],
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "checkpoint_sha256": common._sha256(checkpoint_path),
        "run_config_sha256": common._sha256(run_config_path),
        "loss_summary_sha256": common._sha256(summary_path),
        "parameter_count": int(checkpoint["param_count"]),
        "graph_build_seconds_valid128": float(graph_seconds),
        "graph_build_seconds_per_sample": float(graph_seconds / 128.0),
        "first_compile_inference_seconds_batch1": float(first_compile_seconds),
        "warm_repeat_count": int(warm_repeats),
        "warm_inference_seconds_batch1": warm_summary,
        "warm_benchmark_overhead_seconds": warm_benchmark_seconds,
        "formal_inference_seconds_valid128_including_first_compile": float(formal_seconds),
        "formal_inference_seconds_valid128_excluding_graph": float(formal_seconds),
        "end_to_end_seconds_valid128": float(end_to_end_seconds),
        "end_to_end_seconds_valid128_excluding_warm_benchmark": float(
            end_to_end_excluding_warm_seconds
        ),
        "peak_ram_bytes": int(max(rss_before, rss_after_graph, rss_after_inference)),
        "rss_before_bytes": int(rss_before),
        "rss_after_graph_bytes": int(rss_after_graph),
        "rss_after_inference_bytes": int(rss_after_inference),
        "gpu_memory": "N/A_CPU_only",
        "device": str(jax.devices()[0]),
        "batch_size": 1,
        "global_context_fit_population": "train_only",
        "global_context_fit_sample_count": 768,
        "edge_masking_inference_key": None,
    }
    prepared = {
        "model": model,
        "params": params,
        "groups": groups,
        "run_config": run_config,
        "model_config": model_config,
    }
    return predictions, scales, runtime, prepared


def _predict_reusing_graphs_with_anchor_context(
    *,
    prepared: Mapping[str, Any],
    anchor_context_examples: Sequence[anchored.AnchoredExample],
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Replay the same frozen graphs/inputs with anchor-derived 24D context."""

    groups = [dict(group) for group in prepared["groups"]]
    run_config = prepared["run_config"]
    model_config = prepared["model_config"]
    standardizer = run_config["global_context"]["standardizer"]
    encoded = {
        example.sample_id: standardize_v6_contexts(
            [runner._global_context_row_for_example(example)], standardizer
        )[0]
        for example in anchor_context_examples
    }
    runner._attach_global_context_to_groups(
        groups,
        encoded,
        expected_feature_dim=int(model_config["global_context_feature_dim"]),
    )
    predictions: dict[str, np.ndarray] = {}
    started = time.perf_counter()
    for group in groups:
        output = runner._model_apply(
            prepared["model"], prepared["params"], group
        )
        _block(output)
        raw = np.asarray(output["raw_temperature"], dtype=np.float64)
        for index, sample_id in enumerate(group["sample_ids"]):
            predictions[str(sample_id)] = raw[index, 0, :, 0]
    elapsed = time.perf_counter() - started
    if len(predictions) != 128:
        raise SourceAwareEvaluationError("anchor-context prediction count drifted")
    return predictions, {
        "graph_reused_from_primary": True,
        "graph_build_seconds": 0.0,
        "inference_seconds_valid128": float(elapsed),
    }


def _field_bias(
    *,
    predictions: Mapping[str, np.ndarray],
    examples: Sequence[volume.VolumeProbeV6Example],
    targets: Mapping[str, Mapping[str, np.ndarray]],
    weights: np.ndarray,
) -> float:
    signed = 0.0
    total_weight = 0.0
    for example in examples:
        reference = float(example.meta["v6_adapter"]["reference_temperature_K"])
        pred = np.asarray(predictions[example.sample_id], dtype=np.float64) - reference
        truth = np.asarray(targets[example.sample_id]["deltaT_K"], dtype=np.float64)
        signed += float(np.sum(weights * (pred - truth)))
        total_weight += float(np.sum(weights))
    return float(signed / total_weight)


def _metrics(
    *,
    predictions: Mapping[str, np.ndarray],
    examples: Sequence[volume.VolumeProbeV6Example],
    targets: Mapping[str, Mapping[str, np.ndarray]],
    public: Mapping[str, Any],
    resolution: int,
) -> dict[str, Any]:
    result = common._metrics(
        predictions=predictions,
        examples=examples,
        targets=targets,
        public=public,
    )
    result.update(
        anchored._derive_shape_scale(
            predictions,
            examples,
            targets,
            np.asarray(public["control_volume"], dtype=np.float64),
        )
    )
    result["field_cv_weighted_bias_K"] = _field_bias(
        predictions=predictions,
        examples=examples,
        targets=targets,
        weights=np.asarray(public["control_volume"], dtype=np.float64),
    )
    result["node_count"] = resolution
    return result


def _diagnostic_predictions(
    *,
    context_predictions: Mapping[str, np.ndarray],
    anchor_scales: Mapping[str, float],
    examples: Sequence[volume.VolumeProbeV6Example],
    weights: np.ndarray,
) -> dict[str, np.ndarray]:
    return anchored._anchor_scale_predictions(
        context_predictions, anchor_scales, examples, weights
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--ladder", type=Path, required=True)
    parser.add_argument("--resolution", type=int, required=True)
    parser.add_argument("--seed", choices=tuple(SEED_SPECS), required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--warm-repeats", type=int, default=WARM_REPEATS)
    args = parser.parse_args()
    if args.warm_repeats < 10:
        raise SourceAwareEvaluationError("warm benchmark requires at least 10 repeats")
    if os.environ.get("JAX_PLATFORMS") != "cpu":
        raise SourceAwareEvaluationError("formal evaluation requires JAX_PLATFORMS=cpu")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise SourceAwareEvaluationError('formal evaluation requires CUDA_VISIBLE_DEVICES=""')
    if any(device.platform != "cpu" for device in jax.devices()):
        raise SourceAwareEvaluationError("formal evaluation resolved a non-CPU device")
    ladder = json.loads(args.ladder.read_text(encoding="utf-8"))
    if (
        ladder["evaluation_role"] != "valid_iid"
        or ladder["test_hard_accessed"]
        or not ladder["strictly_nested"]
    ):
        raise SourceAwareEvaluationError("ladder role/nesting contract drifted")
    try:
        probe = ladder["probes"][str(args.resolution)]
    except KeyError as error:
        raise SourceAwareEvaluationError("resolution is not in frozen ladder") from error
    anchored_examples, targets, public = anchored._load_examples(
        args.dataset.resolve(), args.manifest.resolve(), probe
    )
    query_examples = _query_context_examples(anchored_examples)
    spec = SEED_SPECS[args.seed]
    main_predictions, main_scales, main_runtime, prepared = _predict_benchmark(
        run_dir=args.run_dir.resolve(),
        spec=spec,
        examples=query_examples,
        warm_repeats=args.warm_repeats,
    )
    if args.resolution == 1024:
        anchor_scales = main_scales
        anchor_context_predictions = main_predictions
        diagnostic_runtime = {
            "reused_main_1024": True,
            "additional_inference_seconds": 0.0,
        }
    else:
        anchor_probe = ladder["probes"]["1024"]
        anchor_examples, _, _ = anchored._load_examples(
            args.dataset.resolve(), args.manifest.resolve(), anchor_probe
        )
        _, anchor_scales, anchor_runtime, _ = _predict_benchmark(
            run_dir=args.run_dir.resolve(),
            spec=spec,
            examples=_query_context_examples(anchor_examples),
            warm_repeats=0,
        )
        anchor_context_predictions, diagnostic_forward_runtime = (
            _predict_reusing_graphs_with_anchor_context(
                prepared=prepared,
                anchor_context_examples=anchored_examples,
            )
        )
        diagnostic_runtime = {
            "reused_main_1024": False,
            "anchor_forward": anchor_runtime,
            "anchor_context_forward": diagnostic_forward_runtime,
            "additional_inference_seconds": float(
                anchor_runtime["end_to_end_seconds_valid128"]
                + diagnostic_forward_runtime["inference_seconds_valid128"]
            ),
        }
    diagnostic_predictions = _diagnostic_predictions(
        context_predictions=anchor_context_predictions,
        anchor_scales=anchor_scales,
        examples=query_examples,
        weights=np.asarray(public["control_volume"], dtype=np.float64),
    )
    modes = {
        "upstream_like_joint_context_scale": _metrics(
            predictions=main_predictions,
            examples=query_examples,
            targets=targets,
            public=public,
            resolution=args.resolution,
        ),
        "anchor_derived_context_scale_diagnostic": _metrics(
            predictions=diagnostic_predictions,
            examples=query_examples,
            targets=targets,
            public=public,
            resolution=args.resolution,
        ),
    }
    primary = modes["upstream_like_joint_context_scale"]
    finite = all(
        math.isfinite(float(value))
        for value in (
            primary["point_global_cv_relative_rmse_pct"],
            primary["sample_first_cv_relative_rmse_pct"],
            primary["raw_cv_weighted_rmse_K"],
            primary["shape_cv_rmse"],
            primary["scale_log_rmse"],
        )
    )
    stop_reasons = []
    if not finite:
        stop_reasons.append("non_finite_metric")
    if primary["point_global_cv_relative_rmse_pct"] >= 20.0:
        stop_reasons.append("point_global_ge_20_pct")
    if (
        main_runtime["end_to_end_seconds_valid128_excluding_warm_benchmark"]
        > UNACCEPTABLE_END_TO_END_SECONDS
    ):
        stop_reasons.append("unacceptable_end_to_end_runtime")
    if main_runtime["peak_ram_bytes"] > UNACCEPTABLE_PEAK_RAM_BYTES:
        stop_reasons.append("unacceptable_peak_ram")
    payload = {
        "schema_version": "heat3d_v6_source_aware_resolution_eval_v1",
        "status": "passed" if finite else "failed_non_finite",
        "resolution": args.resolution,
        "seed": args.seed,
        "config_id": spec["config_id"],
        "probe_id": probe["probe_id"],
        "primary_mode": "upstream_like_joint_context_scale",
        "diagnostic_mode": "anchor_derived_context_scale_diagnostic",
        "evaluation_role": "valid_iid",
        "test_hard_accessed": False,
        "training_executed": False,
        "checkpoint_modified": False,
        "formal_inference_platform": "local_CPU",
        "context_policy": {
            "primary": "recomputed_from_all_N_source_aware_nodes",
            "diagnostic": "frozen_canonical_1024_anchor_context_and_scale",
        },
        "runtime": main_runtime,
        "diagnostic_runtime": diagnostic_runtime,
        "modes": modes,
        "resolution_gate": {
            "passed": not stop_reasons,
            "stop_reasons": stop_reasons,
            "point_global_limit_pct": 20.0,
            "end_to_end_limit_seconds": UNACCEPTABLE_END_TO_END_SECONDS,
            "peak_ram_limit_bytes": UNACCEPTABLE_PEAK_RAM_BYTES,
        },
    }
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "status": payload["status"],
                "resolution": args.resolution,
                "seed": args.seed,
                "point_global_pct": primary["point_global_cv_relative_rmse_pct"],
                "gate_passed": payload["resolution_gate"]["passed"],
                "stop_reasons": stop_reasons,
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
