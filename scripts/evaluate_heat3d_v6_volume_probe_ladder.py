#!/usr/bin/env python3
"""Evaluate V6_03/V6_04 on one prepared volume-representative valid probe."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from typing import Any, Mapping, Sequence

import h5py
import jax
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
import sys

for value in (ROOT, ROOT / "scripts"):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))

import evaluate_heat3d_v6_common_valid_probe as common  # noqa: E402
from rigno.heat3d_v1_native_supervised import (  # noqa: E402
    V1SteadyConditionInput,
    V1SteadyTarget,
)
from rigno.heat3d_v6_dataset import (  # noqa: E402
    V6_DUAL_ROBIN_CONDITION_FEATURES,
    V6DualRobinExample,
)
from rigno.models.rigno import RIGNO as GraphNeuralOperator  # noqa: E402
import run_heat3d_v1_medium_controlled_training_export as runner  # noqa: E402
from run_heat3d_v3_final_probe_checkpoint_smoke import (  # noqa: E402
    install_checkpoint_feature_hooks,
)


DEFAULT_LADDER = (
    ROOT / "configs/heat3d_v6/v6_volume_representative_probe_ladder.json"
)
DEFAULT_MANIFEST = (
    ROOT / "configs/heat3d_v6/v6_p1h_shared_support1024_manifest.json"
)
ALLOWED_MODELS = (
    "V6_03_V5best_P1h",
    "V6_04_V5best_P1h_DualAttention",
)


class VolumeProbeError(RuntimeError):
    pass


def _memory_stats() -> dict[str, int | None]:
    raw = jax.devices()[0].memory_stats() or {}
    return {
        "bytes_in_use": (
            int(raw["bytes_in_use"]) if raw.get("bytes_in_use") is not None else None
        ),
        "peak_bytes_in_use": (
            int(raw["peak_bytes_in_use"])
            if raw.get("peak_bytes_in_use") is not None
            else None
        ),
    }


def _load_valid_examples(
    *,
    dataset_root: Path,
    manifest_path: Path,
    probe: Mapping[str, Any],
) -> tuple[
    list[V6DualRobinExample],
    dict[str, dict[str, np.ndarray]],
    dict[str, Any],
]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest["dataset_id"] != common.DATASET_ID:
        raise VolumeProbeError("manifest is not frozen P1h")
    valid_rows = [
        row for row in manifest["samples"] if row["split_role"] == "valid"
    ]
    if len(valid_rows) != 128:
        raise VolumeProbeError("valid_iid sample count drifted")
    valid_ids = [str(row["sample_id"]) for row in valid_rows]
    valid_hash = common.hashlib.sha256(
        "\n".join(valid_ids).encode("utf-8")
    ).hexdigest()
    if valid_hash != probe["valid_sample_ids_sha256"]:
        raise VolumeProbeError("valid sample binding drifted")
    if (
        probe["evaluation_role"] != "valid_iid"
        or not probe["label_independent"]
        or probe["test_hard_accessed"]
    ):
        raise VolumeProbeError("probe role/leakage contract failed")

    node_count = int(probe["node_count"])
    indices = np.asarray(probe["indices"], dtype=np.int32)
    expansion = np.asarray(probe["expansion_weights_m3"], dtype=np.float64)
    if (
        indices.shape != (node_count,)
        or expansion.shape != (node_count,)
        or len(np.unique(indices)) != node_count
    ):
        raise VolumeProbeError("probe arrays have invalid shape or uniqueness")

    archive_path = dataset_root / "full_fields.h5"
    if common._sha256(archive_path) != probe["full_field_archive_sha256"]:
        raise VolumeProbeError("full-field archive SHA256 mismatch")
    examples: list[V6DualRobinExample] = []
    targets: dict[str, dict[str, np.ndarray]] = {}
    with h5py.File(archive_path, "r") as handle:
        archive_ids = [
            value.decode("utf-8") if isinstance(value, bytes) else str(value)
            for value in handle["samples/sample_id"][:]
        ]
        archive_index = {
            sample_id: index for index, sample_id in enumerate(archive_ids)
        }
        coords = np.asarray(handle["mesh/coords"][:], dtype=np.float64)[indices]
        k_field = np.asarray(handle["mesh/k_diag"][:], dtype=np.float64)[indices]
        layer_id = np.asarray(
            handle["mesh/layer_id"][:], dtype=np.int32
        )[indices]
        raw_cv = np.asarray(
            handle["mesh/control_volume"][:], dtype=np.float64
        )[indices]
        if common._array_sha256(coords) != probe["coordinate_sha256"]:
            raise VolumeProbeError("coordinate SHA256 mismatch")
        if common._array_sha256(layer_id) != probe["layer_id_sha256"]:
            raise VolumeProbeError("layer SHA256 mismatch")
        if common._array_sha256(raw_cv) != probe["control_volume_sha256"]:
            raise VolumeProbeError("control-volume SHA256 mismatch")
        if common._array_sha256(expansion) != probe["expansion_weight_sha256"]:
            raise VolumeProbeError("expansion-weight SHA256 mismatch")
        flags = common._bc_features(coords)

        for row in valid_rows:
            sample_id = str(row["sample_id"])
            archive_row = archive_index[sample_id]
            sample_dir = dataset_root / str(row.get("sample_dir") or sample_id)
            meta = json.loads(
                (sample_dir / "sample_meta.json").read_text(encoding="utf-8")
            )
            if meta["split_role"] != "valid":
                raise VolumeProbeError(f"{sample_id}: non-valid row materialized")
            top = meta["boundary_conditions"]["top"]
            bottom = meta["boundary_conditions"]["bottom"]
            temperature = np.asarray(
                handle["samples/temperature_K"][archive_row, :],
                dtype=np.float64,
            )[indices]
            q_field = np.asarray(
                handle["samples/q_W_m3"][archive_row, :],
                dtype=np.float64,
            )[indices]
            broadcasts = np.column_stack(
                (
                    np.full(node_count, float(top["h_W_m2K"])),
                    np.full(node_count, float(bottom["h_W_m2K"])),
                    np.full(
                        node_count,
                        float(top["T_inf_K"]) - float(bottom["T_inf_K"]),
                    ),
                )
            )
            condition = np.concatenate(
                (k_field, q_field[:, None], flags, broadcasts), axis=1
            )
            if condition.shape != (
                node_count,
                len(V6_DUAL_ROBIN_CONDITION_FEATURES),
            ):
                raise VolumeProbeError(f"{sample_id}: condition width drifted")
            enriched_meta = dict(meta)
            enriched_meta["split"] = "valid_iid"
            enriched_meta["v6_adapter"] = {
                "dataset_id": common.DATASET_ID,
                "manifest_split_role": "valid",
                "group_id": str(row["group_id"]),
                "reference_temperature_K": float(bottom["T_inf_K"]),
                "top_T_inf_K": float(top["T_inf_K"]),
                "bottom_T_inf_K": float(bottom["T_inf_K"]),
                "bottom_boundary_semantics": "robin_not_dirichlet",
                "operator_point_measure": (
                    f"volume_representative_solver_probe{node_count}"
                ),
            }
            examples.append(
                V6DualRobinExample(
                    sample_id=sample_id,
                    condition=V1SteadyConditionInput(
                        coords=coords,
                        condition_features=condition,
                        condition_feature_names=V6_DUAL_ROBIN_CONDITION_FEATURES,
                        k_encoding_mode="diag3",
                    ),
                    target=V1SteadyTarget(target_u=temperature[:, None]),
                    meta=enriched_meta,
                )
            )
            targets[sample_id] = {
                "deltaT_K": temperature - float(bottom["T_inf_K"]),
                "q_W_m3": q_field,
            }
    public = {
        "coords": coords,
        "layer_id": layer_id,
        # The common metric implementation consumes this field as its physical
        # integration measure. Here it is the frozen Horvitz-Thompson measure.
        "control_volume": expansion,
        "raw_selected_control_volume": raw_cv,
        "top_mask": flags[:, 0] > 0.5,
        "bottom_mask": flags[:, 1] > 0.5,
        "valid_sample_ids": valid_ids,
    }
    return examples, targets, public


def _predict_timed(
    *,
    run_dir: Path,
    spec: Mapping[str, Any],
    examples: Sequence[V6DualRobinExample],
    batch_size: int,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    checkpoint_path = run_dir / "params_best_valid_point_global.pkl"
    run_config_path = run_dir / "run_config.json"
    summary_path = run_dir / "loss_summary.json"
    for path in (checkpoint_path, run_config_path, summary_path):
        if not path.is_file():
            raise VolumeProbeError(f"missing frozen run artifact: {path}")
    if common._sha256(checkpoint_path) != spec["checkpoint_sha256"]:
        raise VolumeProbeError(f"{run_dir.name}: checkpoint SHA256 mismatch")
    run_config = json.loads(run_config_path.read_text(encoding="utf-8"))
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    checkpoint = runner._load_params_checkpoint(checkpoint_path)
    if (
        int(checkpoint["epoch"]) != int(spec["checkpoint_epoch"])
        or checkpoint["checkpoint_kind"] != "point_global_best"
        or str(checkpoint["git_commit"])[:7] != str(spec["training_commit"])[:7]
        or int(summary["point_global_best_epoch"]) != int(spec["checkpoint_epoch"])
    ):
        raise VolumeProbeError(f"{run_dir.name}: checkpoint provenance mismatch")

    checkpoint_stats = common._materialize_checkpoint_stats(
        checkpoint["train_only_normalization"]
    )
    runtime_checkpoint = dict(checkpoint)
    runtime_checkpoint["train_only_normalization"] = checkpoint_stats
    install_checkpoint_feature_hooks(checkpoint_stats)
    memory_before = _memory_stats()
    graph_started = time.perf_counter()
    groups = common._prepare_groups(
        examples=examples,
        run_config=run_config,
        checkpoint=runtime_checkpoint,
        batch_size=batch_size,
    )
    graph_build_seconds = time.perf_counter() - graph_started
    memory_after_graph = _memory_stats()

    model_config = runner._resolve_decoder_bypass_model_config(
        dict(checkpoint["model_config"]), checkpoint_stats
    )
    model = GraphNeuralOperator(**model_config)
    params = runner._device_params(checkpoint["params"])
    predictions: dict[str, np.ndarray] = {}
    inference_started = time.perf_counter()
    for group in groups:
        output = runner._model_apply(model, params, group)
        recovered = np.asarray(output["raw_temperature"], dtype=np.float64)
        if not np.all(np.isfinite(recovered)):
            raise VolumeProbeError(f"{run_dir.name}: non-finite prediction")
        for row, sample_id in enumerate(group["sample_ids"]):
            predictions[str(sample_id)] = recovered[row, 0, :, 0]
    inference_seconds = time.perf_counter() - inference_started
    memory_after_inference = _memory_stats()
    if len(predictions) != len(examples):
        raise VolumeProbeError(f"{run_dir.name}: prediction count drifted")
    peak_candidates = [
        row["peak_bytes_in_use"]
        for row in (memory_before, memory_after_graph, memory_after_inference)
        if row["peak_bytes_in_use"] is not None
    ]
    return predictions, {
        "config_id": run_dir.name,
        "checkpoint_kind": "point_global_best",
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "checkpoint_sha256": common._sha256(checkpoint_path),
        "training_commit": str(checkpoint["git_commit"]),
        "parameter_count": int(checkpoint["param_count"]),
        "global_context_fit_population": "train_only",
        "global_context_fit_sample_count": 768,
        "edge_masking_inference_key": None,
        "batch_size": batch_size,
        "device": str(jax.devices()[0]),
        "graph_build_seconds": float(graph_build_seconds),
        "inference_seconds": float(inference_seconds),
        "peak_memory_bytes": max(peak_candidates) if peak_candidates else None,
        "memory_stats_before": memory_before,
        "memory_stats_after_graph": memory_after_graph,
        "memory_stats_after_inference": memory_after_inference,
    }


def _probe_from_ladder(path: Path, resolution: int) -> dict[str, Any]:
    ladder = json.loads(path.read_text(encoding="utf-8"))
    if (
        ladder["status"] != "prepared_not_evaluated"
        or ladder["evaluation_role"] != "valid_iid"
        or not ladder["label_independent"]
        or ladder["test_hard_accessed"]
    ):
        raise VolumeProbeError("ladder contract drifted")
    try:
        probe = ladder["probes"][str(resolution)]
    except KeyError as error:
        raise VolumeProbeError(f"unsupported resolution {resolution}") from error
    return probe


def _dry_run_payload(
    *,
    ladder_path: Path,
    resolution: int,
    models: Sequence[str],
    input_root: Path,
    batch_size: int,
) -> dict[str, Any]:
    probe = _probe_from_ladder(ladder_path, resolution)
    return {
        "status": "dry_run_passed",
        "probe_id": probe["probe_id"],
        "resolution": resolution,
        "models": list(models),
        "input_root": str(input_root),
        "batch_size": batch_size,
        "metrics": [
            "CV-weighted point-global relative RMSE",
            "sample-first CV relative RMSE",
            "raw CV RMSE K",
            "peak RMSE K",
            "source-region CV RMSE K",
            "layer mean/drop RMSE K",
            "top/bottom surface CV RMSE K",
            "graph build time",
            "inference time",
            "peak device memory",
        ],
        "evaluation_role": "valid_iid",
        "test_hard_accessed": False,
        "training_executed": False,
        "formal_inference_executed": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--ladder", type=Path, default=DEFAULT_LADDER)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--resolution", type=int, required=True)
    parser.add_argument("--model", action="append", choices=ALLOWED_MODELS)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    models = tuple(args.model or ALLOWED_MODELS)
    if args.batch_size < 1:
        raise VolumeProbeError("--batch-size must be positive")
    if args.dry_run:
        print(
            json.dumps(
                _dry_run_payload(
                    ladder_path=args.ladder,
                    resolution=args.resolution,
                    models=models,
                    input_root=args.input_root,
                    batch_size=args.batch_size,
                ),
                indent=2,
                sort_keys=True,
            )
        )
        return 0

    probe = _probe_from_ladder(args.ladder, args.resolution)
    examples, targets, public = _load_valid_examples(
        dataset_root=args.dataset.resolve(),
        manifest_path=args.manifest.resolve(),
        probe=probe,
    )
    output_models = {}
    for config_id in models:
        predictions, checkpoint = _predict_timed(
            run_dir=args.input_root.resolve() / config_id,
            spec=common.RUN_SPECS[config_id],
            examples=examples,
            batch_size=args.batch_size,
        )
        metrics = common._metrics(
            predictions=predictions,
            examples=examples,
            targets=targets,
            public=public,
        )
        metrics["node_count"] = args.resolution
        output_models[config_id] = {
            "checkpoint": checkpoint,
            "metrics": metrics,
        }
    payload = {
        "schema_version": "heat3d_v6_volume_probe_evaluation_v1",
        "status": "passed",
        "probe_id": probe["probe_id"],
        "resolution": args.resolution,
        "metric_weight_policy": probe["metric_weight_policy"],
        "evaluation_role": "valid_iid",
        "test_hard_accessed": False,
        "training_executed": False,
        "checkpoint_selection_modified": False,
        "models": output_models,
    }
    if args.write:
        if args.output_json is None:
            raise VolumeProbeError("--write requires --output-json")
        args.output_json.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    print(
        json.dumps(
            {
                "status": payload["status"],
                "probe_id": payload["probe_id"],
                "resolution": payload["resolution"],
                "models": list(output_models),
                "test_hard_accessed": False,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
