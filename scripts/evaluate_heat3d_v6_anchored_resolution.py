#!/usr/bin/env python3
"""CPU-only valid_iid evaluation of one V6 anchored-ladder resolution."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import resource
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

import evaluate_heat3d_v6_common_valid_probe as common  # noqa: E402
import evaluate_heat3d_v6_volume_probe_ladder as volume  # noqa: E402
from rigno.heat3d_v1_native_supervised import V1SteadyConditionInput, V1SteadyTarget  # noqa: E402
from rigno.heat3d_v6_dataset import V6_DUAL_ROBIN_CONDITION_FEATURES  # noqa: E402
from rigno.models.rigno import RIGNO as GraphNeuralOperator  # noqa: E402
import run_heat3d_v1_medium_controlled_training_export as runner  # noqa: E402
from run_heat3d_v3_final_probe_checkpoint_smoke import install_checkpoint_feature_hooks  # noqa: E402


SEED_SPECS = {
    "seed0": {
        "config_id": "V6_03_V5best_P1h",
        "epoch": 111,
        "sha256": "3ad58c2b34a46481acb74722c80bdcadbf55a0d613bc25c4fe2d7646b91aa1f2",
    },
    "seed1": {
        "config_id": "V6_03_V5best_P1h_seed1",
        "epoch": 254,
        "sha256": "36937e4898fa633ba03abf1413a852c1e43130566e2c4a443f12f650536512ae",
    },
    "seed2": {
        "config_id": "V6_03_V5best_P1h_seed2",
        "epoch": 139,
        "sha256": "88b77e0c4835f4dcfa767a0f6be89d173b29b46db21530b9c38a47325b3adc4d",
    },
}


class AnchoredEvaluationError(RuntimeError):
    pass


@dataclass(frozen=True)
class AnchoredExample(volume.VolumeProbeV6Example):
    context_coords: np.ndarray
    context_condition: np.ndarray
    context_weights: np.ndarray

    def v6_global_context_inputs(self) -> dict[str, Any]:
        payload = super().v6_global_context_inputs()
        payload["coords"] = self.context_coords
        payload["raw_condition"] = self.context_condition
        payload["operator_point_weights"] = self.context_weights
        return payload


def _rss_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if sys.platform == "darwin" else value * 1024


def _load_examples(
    dataset_root: Path,
    manifest_path: Path,
    probe: Mapping[str, Any],
    *,
    split_role: str = "valid",
    load_labels: bool = True,
) -> tuple[list[AnchoredExample], dict[str, dict[str, np.ndarray]], dict[str, Any]]:
    if split_role not in {"valid", "test"}:
        raise AnchoredEvaluationError(
            f"anchored evaluator role must be valid or test, found {split_role!r}"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows = [row for row in manifest["samples"] if row["split_role"] == split_role]
    if len(rows) != 128:
        raise AnchoredEvaluationError(f"{split_role}_iid count drifted")
    indices = np.asarray(probe["indices"], dtype=np.int32)
    weights = np.asarray(probe["metric_weights"], dtype=np.float64)
    archive = dataset_root / "full_fields.h5"
    examples: list[AnchoredExample] = []
    targets: dict[str, dict[str, np.ndarray]] = {}
    with h5py.File(archive, "r") as handle:
        ids = [
            value.decode() if isinstance(value, bytes) else str(value)
            for value in handle["samples/sample_id"][:]
        ]
        row_index = {sample_id: index for index, sample_id in enumerate(ids)}
        anchor_indices = np.asarray(handle["support/indices"], dtype=np.int32)
        coords_all = np.asarray(handle["mesh/coords"], dtype=np.float64)
        k_all = np.asarray(handle["mesh/k_diag"], dtype=np.float64)
        layer_all = np.asarray(handle["mesh/layer_id"], dtype=np.int32)
        coords = coords_all[indices]
        k = k_all[indices]
        layer = layer_all[indices]
        anchor_coords = coords_all[anchor_indices]
        anchor_k = k_all[anchor_indices]
        flags = common._bc_features(coords)
        anchor_flags = common._bc_features(anchor_coords)
        for row in rows:
            sample_id = str(row["sample_id"])
            archive_row = row_index[sample_id]
            sample_dir = dataset_root / str(row.get("sample_dir") or sample_id)
            meta = json.loads((sample_dir / "sample_meta.json").read_text())
            top = meta["boundary_conditions"]["top"]
            bottom = meta["boundary_conditions"]["bottom"]
            q_all = np.asarray(handle["samples/q_W_m3"][archive_row], dtype=np.float64)
            q = q_all[indices]
            if load_labels:
                temp_all = np.asarray(
                    handle["samples/temperature_K"][archive_row],
                    dtype=np.float64,
                )
                temperature = temp_all[indices]
            else:
                temperature = np.full(
                    len(indices), float(bottom["T_inf_K"]), dtype=np.float64
                )
            broadcasts = np.column_stack(
                (
                    np.full(len(indices), float(top["h_W_m2K"])),
                    np.full(len(indices), float(bottom["h_W_m2K"])),
                    np.full(len(indices), float(top["T_inf_K"]) - float(bottom["T_inf_K"])),
                )
            )
            anchor_broadcasts = np.column_stack(
                (
                    np.full(1024, float(top["h_W_m2K"])),
                    np.full(1024, float(bottom["h_W_m2K"])),
                    np.full(1024, float(top["T_inf_K"]) - float(bottom["T_inf_K"])),
                )
            )
            condition = np.concatenate((k, q[:, None], flags, broadcasts), axis=1)
            anchor_condition = np.concatenate(
                (anchor_k, q_all[anchor_indices, None], anchor_flags, anchor_broadcasts),
                axis=1,
            )
            enriched = dict(meta)
            enriched["split"] = f"{split_role}_iid"
            enriched["v6_adapter"] = {
                "dataset_id": manifest["dataset_id"],
                "manifest_split_role": split_role,
                "group_id": str(row["group_id"]),
                "reference_temperature_K": float(bottom["T_inf_K"]),
                "top_T_inf_K": float(top["T_inf_K"]),
                "bottom_T_inf_K": float(bottom["T_inf_K"]),
                "bottom_boundary_semantics": "robin_not_dirichlet",
                "operator_point_measure": probe["metric_weight_policy"],
            }
            examples.append(
                AnchoredExample(
                    sample_id=sample_id,
                    condition=V1SteadyConditionInput(
                        coords=coords,
                        condition_features=condition,
                        condition_feature_names=V6_DUAL_ROBIN_CONDITION_FEATURES,
                        k_encoding_mode="diag3",
                    ),
                    target=V1SteadyTarget(target_u=temperature[:, None]),
                    meta=enriched,
                    operator_point_weights=weights,
                    context_coords=anchor_coords,
                    context_condition=anchor_condition,
                    context_weights=np.full(1024, 1.0 / 1024.0),
                )
            )
            if load_labels:
                targets[sample_id] = {
                    "deltaT_K": temperature - float(bottom["T_inf_K"]),
                    "q_W_m3": q,
                }
    public = {
        "coords": coords,
        "layer_id": layer,
        "control_volume": weights,
        "top_mask": flags[:, 0] > 0.5,
        "bottom_mask": flags[:, 1] > 0.5,
        "evaluation_sample_ids": [str(row["sample_id"]) for row in rows],
        "evaluation_role": f"{split_role}_iid",
        "labels_loaded": bool(load_labels),
    }
    public[f"{split_role}_sample_ids"] = public["evaluation_sample_ids"]
    if split_role == "valid":
        public["valid_sample_ids"] = public["evaluation_sample_ids"]
    return examples, targets, public


def _derive_shape_scale(
    predictions: Mapping[str, np.ndarray],
    examples: Sequence[AnchoredExample],
    targets: Mapping[str, Mapping[str, np.ndarray]],
    weights: np.ndarray,
) -> dict[str, float]:
    shape_errors, scale_errors, signed_scale = [], [], []
    for example in examples:
        ref = float(example.meta["v6_adapter"]["reference_temperature_K"])
        pred = np.asarray(predictions[example.sample_id], dtype=np.float64) - ref
        truth = np.asarray(targets[example.sample_id]["deltaT_K"], dtype=np.float64)
        w = weights / np.sum(weights)
        pred_scale = math.sqrt(float(np.sum(w * pred * pred)))
        true_scale = math.sqrt(float(np.sum(w * truth * truth)))
        pred_shape = pred / max(pred_scale, 1e-15)
        true_shape = truth / max(true_scale, 1e-15)
        shape_errors.append(math.sqrt(float(np.sum(w * (pred_shape - true_shape) ** 2))))
        scale_errors.append(math.log(max(pred_scale, 1e-15) / true_scale))
        signed_scale.append(pred_scale / true_scale - 1.0)
    return {
        "shape_cv_rmse": float(np.mean(shape_errors)),
        "scale_log_rmse": float(np.sqrt(np.mean(np.square(scale_errors)))),
        "scale_signed_bias": float(np.mean(signed_scale)),
    }


def _predict(
    run_dir: Path,
    spec: Mapping[str, Any],
    examples: Sequence[AnchoredExample],
) -> tuple[dict[str, np.ndarray], dict[str, float], dict[str, Any]]:
    checkpoint_path = run_dir / "params_best_valid_point_global.pkl"
    if common._sha256(checkpoint_path) != spec["sha256"]:
        raise AnchoredEvaluationError(f"{spec['config_id']}: checkpoint SHA drifted")
    checkpoint = runner._load_params_checkpoint(checkpoint_path)
    if int(checkpoint["epoch"]) != int(spec["epoch"]):
        raise AnchoredEvaluationError("checkpoint epoch drifted")
    run_config = json.loads((run_dir / "run_config.json").read_text())
    stats = common._materialize_checkpoint_stats(checkpoint["train_only_normalization"])
    runtime = dict(checkpoint)
    runtime["train_only_normalization"] = stats
    install_checkpoint_feature_hooks(stats)
    started_graph = time.perf_counter()
    groups = common._prepare_groups(
        examples=examples, run_config=run_config, checkpoint=runtime, batch_size=1
    )
    graph_seconds = time.perf_counter() - started_graph
    model_config = runner._resolve_decoder_bypass_model_config(
        dict(checkpoint["model_config"]), stats
    )
    model = GraphNeuralOperator(**model_config)
    params = runner._device_params(checkpoint["params"])
    predictions: dict[str, np.ndarray] = {}
    scales: dict[str, float] = {}
    started = time.perf_counter()
    for group in groups:
        output = runner._model_apply(model, params, group)
        raw = np.asarray(output["raw_temperature"], dtype=np.float64)
        scale = np.asarray(output["s_hat"], dtype=np.float64).reshape(len(group["sample_ids"]))
        for index, sample_id in enumerate(group["sample_ids"]):
            predictions[str(sample_id)] = raw[index, 0, :, 0]
            scales[str(sample_id)] = float(scale[index])
    return predictions, scales, {
        "graph_build_seconds": float(graph_seconds),
        "inference_seconds": float(time.perf_counter() - started),
        "process_peak_rss_bytes": _rss_bytes(),
        "device": str(jax.devices()[0]),
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "checkpoint_sha256": common._sha256(checkpoint_path),
        "parameter_count": int(checkpoint["param_count"]),
    }


def _anchor_scale_predictions(
    joint: Mapping[str, np.ndarray],
    anchor_scales: Mapping[str, float],
    examples: Sequence[AnchoredExample],
    weights: np.ndarray,
) -> dict[str, np.ndarray]:
    result = {}
    w = weights / np.sum(weights)
    for example in examples:
        ref = float(example.meta["v6_adapter"]["reference_temperature_K"])
        delta = np.asarray(joint[example.sample_id], dtype=np.float64) - ref
        joint_scale = math.sqrt(float(np.sum(w * delta * delta)))
        result[example.sample_id] = ref + delta / max(joint_scale, 1e-15) * anchor_scales[example.sample_id]
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--ladder", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--resolution", type=int, choices=(1024, 2048, 4096, 8192), required=True)
    parser.add_argument("--seed0-run", type=Path, required=True)
    parser.add_argument("--seed1-run", type=Path, required=True)
    parser.add_argument("--seed2-run", type=Path, required=True)
    parser.add_argument("--seed", action="append", choices=tuple(SEED_SPECS))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if os.environ.get("JAX_PLATFORMS") != "cpu" or os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise AnchoredEvaluationError("formal inference requires local CPU-only environment")
    ladder = json.loads(args.ladder.read_text())
    if ladder["evaluation_role"] != "valid_iid" or ladder["test_hard_accessed"]:
        raise AnchoredEvaluationError("role contract drifted")
    probe = ladder["probes"][str(args.resolution)]
    anchor_probe = ladder["probes"]["1024"]
    examples, targets, public = _load_examples(args.dataset, args.manifest, probe)
    anchor_examples, _, _ = _load_examples(args.dataset, args.manifest, anchor_probe)
    run_dirs = {
        "seed0": args.seed0_run,
        "seed1": args.seed1_run,
        "seed2": args.seed2_run,
    }
    results = {}
    selected_seeds = tuple(args.seed or SEED_SPECS)
    for seed in selected_seeds:
        spec = SEED_SPECS[seed]
        anchor_pred, anchor_scales, anchor_runtime = _predict(
            run_dirs[seed], spec, anchor_examples
        )
        if args.resolution == 1024:
            joint_pred, runtime = anchor_pred, anchor_runtime
        else:
            joint_pred, _, runtime = _predict(run_dirs[seed], spec, examples)
        anchor_scale_pred = _anchor_scale_predictions(
            joint_pred, anchor_scales, examples, np.asarray(public["control_volume"])
        )
        modes = {}
        for mode, predictions in (
            ("joint_pooling", joint_pred),
            ("anchor_derived_scale_pooling", anchor_scale_pred),
        ):
            metrics = common._metrics(
                predictions=predictions,
                examples=examples,
                targets=targets,
                public=public,
            )
            metrics["node_count"] = args.resolution
            metrics.update(
                _derive_shape_scale(
                    predictions, examples, targets, np.asarray(public["control_volume"])
                )
            )
            modes[mode] = metrics
        results[seed] = {
            "config_id": spec["config_id"],
            "runtime": runtime,
            "anchor_runtime": anchor_runtime,
            "modes": modes,
        }
    payload = {
        "schema_version": "heat3d_v6_anchored_resolution_evaluation_v1",
        "status": "passed",
        "resolution": args.resolution,
        "probe_id": probe["probe_id"],
        "conditioning_support": ladder["conditioning_support"],
        "query_support": ladder["query_support"],
        "global_context_source": "frozen_1024_anchors",
        "evaluation_role": "valid_iid",
        "test_hard_accessed": False,
        "training_executed": False,
        "checkpoint_modified": False,
        "formal_inference_platform": "local_CPU",
        "results": results,
    }
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": "passed", "resolution": args.resolution}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
