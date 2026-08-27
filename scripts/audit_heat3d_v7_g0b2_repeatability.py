#!/usr/bin/env python3
"""Characterize legacy/V7 repeatability on one frozen valid_iid model group.

This is an audit harness, not the V7 reference entrypoint.  It deliberately
loads the legacy runtime as a comparison control, while the V7 side imports
only ``rigno.heat3d_runtime``.  It reads only valid_iid samples and writes
JSON to stdout; no metrics, solver, training, or artifact generation occurs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any, Callable, Sequence

import jax
import numpy as np

from rigno.heat3d_runtime import RuntimeSession, compare_named_arrays, snapshot_group
from rigno.heat3d_runtime.checkpoint import file_sha256
from rigno.heat3d_v6_dataset import Heat3DV6DualRobinDataset

import benchmark_heat3d_v6_inference_qualification as qualification
import evaluate_heat3d_v6_common_valid_probe as common
import run_heat3d_v1_medium_controlled_training_export as legacy_runner
from run_heat3d_v3_final_probe_checkpoint_smoke import install_checkpoint_feature_hooks


MODEL_GROUP_KEYS = (
    "inputs",
    "graphs",
    "global_context",
    "native_physics",
    "qk_region_features",
    "scale_context",
    "scale_region_source_weights",
    "scale_region_volume_weights",
)


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _tree_sha256(value: Any) -> str:
    digest = hashlib.sha256()
    for index, leaf in enumerate(jax.tree_util.tree_leaves(value)):
        if leaf is None or not hasattr(leaf, "shape"):
            continue
        array = np.ascontiguousarray(np.asarray(leaf))
        digest.update(str(index).encode())
        digest.update(str(array.dtype).encode())
        digest.update(str(tuple(array.shape)).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def _model_group(group: dict[str, Any]) -> dict[str, Any]:
    return {key: group[key] for key in MODEL_GROUP_KEYS if key in group}


def _apply_result(apply: Callable[[], dict[str, Any]]) -> dict[str, np.ndarray]:
    output = apply()
    jax.block_until_ready(output["raw_temperature"])
    return {
        "prediction": np.asarray(output["raw_temperature"], dtype=np.float64).copy(),
        "scale": np.asarray(output["s_hat"], dtype=np.float64).reshape(-1).copy(),
    }


def _error(left: np.ndarray, right: np.ndarray) -> tuple[float, float]:
    difference = np.asarray(left, dtype=np.float64) - np.asarray(right, dtype=np.float64)
    return (
        float(np.max(np.abs(difference))) if difference.size else 0.0,
        float(np.sqrt(np.mean(np.square(difference)))) if difference.size else 0.0,
    )


def _distribution(values: Sequence[float]) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return {"count": 0}
    worst = int(np.argmax(array))
    return {
        "count": int(array.size),
        "min": float(np.min(array)),
        "p05": float(np.quantile(array, 0.05)),
        "median": float(np.median(array)),
        "mean": float(np.mean(array)),
        "p95": float(np.quantile(array, 0.95)),
        "max": float(np.max(array)),
        "worst_index": worst,
        "worst_value": float(array[worst]),
    }


def _repeatability(outputs: Sequence[dict[str, np.ndarray]]) -> dict[str, Any]:
    reference = outputs[0]
    prediction_abs = []
    prediction_rmse = []
    scale_abs = []
    for output in outputs:
        maximum, rmse = _error(output["prediction"], reference["prediction"])
        prediction_abs.append(maximum)
        prediction_rmse.append(rmse)
        scale_maximum, _ = _error(output["scale"], reference["scale"])
        scale_abs.append(scale_maximum)
    pair_prediction_abs = []
    pair_prediction_rmse = []
    pair_scale_abs = []
    for left in outputs:
        for right in outputs:
            maximum, rmse = _error(left["prediction"], right["prediction"])
            pair_prediction_abs.append(maximum)
            pair_prediction_rmse.append(rmse)
            scale_maximum, _ = _error(left["scale"], right["scale"])
            pair_scale_abs.append(scale_maximum)
    return {
        "repeat_count": len(outputs),
        "against_first": {
            "prediction_max_abs_K": _distribution(prediction_abs),
            "prediction_rmse_K": _distribution(prediction_rmse),
            "scale_max_abs": _distribution(scale_abs),
        },
        "all_ordered_pairs": {
            "pair_count": len(pair_prediction_abs),
            "prediction_max_abs_K": _distribution(pair_prediction_abs),
            "prediction_rmse_K": _distribution(pair_prediction_rmse),
            "scale_max_abs": _distribution(pair_scale_abs),
        },
    }


def _cross_runtime(old_outputs: Sequence[dict[str, np.ndarray]], new_outputs: Sequence[dict[str, np.ndarray]]) -> dict[str, Any]:
    prediction_abs = []
    prediction_rmse = []
    scale_abs = []
    for old in old_outputs:
        for new in new_outputs:
            maximum, rmse = _error(old["prediction"], new["prediction"])
            prediction_abs.append(maximum)
            prediction_rmse.append(rmse)
            scale_maximum, _ = _error(old["scale"], new["scale"])
            scale_abs.append(scale_maximum)
    return {
        "pair_count": len(prediction_abs),
        "prediction_max_abs_K": _distribution(prediction_abs),
        "prediction_rmse_K": _distribution(prediction_rmse),
        "scale_max_abs": _distribution(scale_abs),
        "tolerance_forced_pass": False,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--run-config", type=Path, required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--checkpoint-epoch", type=int, required=True)
    parser.add_argument("--repeats", type=int, default=10)
    return parser


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.repeats < 10:
        raise ValueError("G0b-2 repeatability requires at least 10 repetitions")
    dataset = Heat3DV6DualRobinDataset(
        args.dataset_root,
        args.manifest,
        include_roles={"valid_iid"},
    )
    if not dataset.samples:
        raise ValueError("valid_iid fixture is empty")
    example = dataset.samples[0]

    legacy_checkpoint = legacy_runner._load_params_checkpoint(args.checkpoint)
    if int(legacy_checkpoint["epoch"]) != int(args.checkpoint_epoch):
        raise ValueError("legacy checkpoint epoch mismatch")
    stats = legacy_checkpoint["train_only_normalization"]
    stats = common._materialize_checkpoint_stats(stats)
    install_checkpoint_feature_hooks(stats)
    legacy_model_config = legacy_runner._resolve_decoder_bypass_model_config(
        dict(legacy_checkpoint["model_config"]), stats
    )
    legacy_model = legacy_runner.GraphNeuralOperator(**legacy_model_config)
    legacy_params = legacy_runner._device_params(legacy_checkpoint["params"])
    legacy_builder = legacy_runner.Heat3DGraphBuilder(
        **json.loads(args.run_config.read_text(encoding="utf-8"))["graph_config"]
    )
    legacy_groups = legacy_runner._make_v6_padded_groups_with_progress(
        [example],
        stats,
        legacy_builder,
        "g0b2_legacy_control_valid_iid",
        False,
        "off",
        0,
        batch_size=1,
        drop_last=False,
    )
    legacy_group = legacy_groups[0]
    context_payload = json.loads(args.run_config.read_text(encoding="utf-8"))["global_context"]
    context = common.standardize_v6_contexts(
        [legacy_runner._global_context_row_for_example(example)],
        context_payload["standardizer"],
    )
    legacy_runner._attach_global_context_to_groups(
        legacy_groups,
        {example.sample_id: context[0]},
        expected_feature_dim=int(legacy_model_config["global_context_feature_dim"]),
    )
    legacy_runner._attach_native_physics_to_groups(legacy_groups, {example.sample_id: example})
    if (
        legacy_model_config.get("scale_pooling") == "qk_gated"
        or legacy_model_config.get("shape_attention_mode") != "none"
        or legacy_model_config.get("scale_attention_mode") != "none"
    ):
        legacy_runner._attach_qk_region_features_to_groups(
            legacy_groups,
            {example.sample_id: example},
            feature_version=legacy_model_config["qk_region_feature_version"],
        )
    if legacy_model_config.get("scale_deepsets_mode", "none") != "none":
        legacy_runner._attach_scale_deepsets_weights_to_groups(
            legacy_groups, {example.sample_id: example}
        )

    def apply_legacy() -> dict[str, Any]:
        return legacy_model.apply(
            {"params": legacy_params},
            inputs=legacy_group["inputs"],
            graphs=legacy_group["graphs"],
            global_context=legacy_group.get("global_context"),
            control_volumes=legacy_group["native_physics"]["control_volumes"],
            log_s_phys=legacy_group["native_physics"]["log_s_phys"],
            reference_temperature=legacy_group["native_physics"]["reference_temperature"],
            dirichlet_mask=legacy_group["native_physics"]["dirichlet_mask"],
            prescribed_temperature=legacy_group["native_physics"]["prescribed_temperature"],
            qk_region_features=legacy_group.get("qk_region_features"),
            scale_context=legacy_group.get("scale_context"),
            scale_region_source_weights=legacy_group.get("scale_region_source_weights"),
            scale_region_volume_weights=legacy_group.get("scale_region_volume_weights"),
            method=legacy_model.predict_native_shape_scale,
        )

    stable_session = RuntimeSession.from_paths(args.checkpoint, args.run_config)
    stable_group = stable_session.build_group([example], name="g0b2_v7_control_valid_iid")

    old_snapshot = snapshot_group(_model_group(legacy_group))
    new_snapshot = snapshot_group(_model_group(stable_group))
    input_comparison = compare_named_arrays(old_snapshot, new_snapshot)
    old_outputs = [_apply_result(apply_legacy) for _ in range(args.repeats)]
    new_outputs = [_apply_result(lambda: stable_session.apply(stable_group)) for _ in range(args.repeats)]
    old_descriptor = {
        "runtime": "legacy_runner_plus_v3_checkpoint_feature_hook",
        "checkpoint_sha256": file_sha256(args.checkpoint),
        "checkpoint_epoch": int(legacy_checkpoint["epoch"]),
        "model_config": _json_safe(legacy_model_config),
    }
    new_descriptor = stable_session.descriptor()
    git_sha = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True
    ).strip()
    return {
        "schema_version": "heat3d_v7_g0b2_gpu_repeatability_v1",
        "git_sha": git_sha,
        "dataset": {
            "role": "valid_iid",
            "id": dataset.manifest.get("dataset_id"),
            "manifest_sha256": file_sha256(args.manifest),
        },
        "sample_ids": [str(example.sample_id)],
        "resolution": 1024,
        "batch_size": 1,
        "checkpoint": {
            "path": str(args.checkpoint),
            "sha256": file_sha256(args.checkpoint),
            "expected_sha256": str(args.checkpoint_sha256),
            "epoch": int(legacy_checkpoint["epoch"]),
            "sha256_match": file_sha256(args.checkpoint) == str(args.checkpoint_sha256),
        },
        "jax": {
            "jax_version": getattr(jax, "__version__", "unknown"),
            "jaxlib_version": getattr(jax.lib, "__version__", "unknown"),
            "backend": str(jax.default_backend()),
            "devices": [str(device) for device in jax.devices()],
        },
        "runtime_identity": {
            "legacy": old_descriptor,
            "v7": new_descriptor,
        },
        "graph_and_model_visible_tensor_equivalence": input_comparison.as_dict(),
        "graph_hashes": {
            "legacy": _tree_sha256(legacy_group["graphs"]),
            "v7": _tree_sha256(stable_group["graphs"]),
        },
        "repeatability": {
            "legacy": _repeatability(old_outputs),
            "v7": _repeatability(new_outputs),
            "legacy_vs_v7": _cross_runtime(old_outputs, new_outputs),
        },
        "diagnostic_control": {
            "cpu_repeatability": "run separately with JAX_PLATFORMS=cpu",
            "deterministic_gpu_mode": "not_run",
            "tolerance_relaxed": False,
        },
        "test_iid_or_sealed_accessed": False,
        "training_or_solver_invoked": False,
        "metrics_invoked": False,
        "target_or_label_used_for_model_input": False,
    }


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    print(json.dumps(run(args), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
