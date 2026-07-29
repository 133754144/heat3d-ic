#!/usr/bin/env python3
"""Evaluate frozen V6 point-global checkpoints on one 4096-node valid probe."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import h5py
import jax
import jax.numpy as jnp
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
for value in (ROOT, ROOT / "scripts"):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))

from rigno.graphBuilder_Heat3D import Heat3DGraphBuilder  # noqa: E402
from rigno.heat3d_v1_native_supervised import (  # noqa: E402
    V1SteadyConditionInput,
    V1SteadyTarget,
)
from rigno.heat3d_v6_dataset import (  # noqa: E402
    V6_DUAL_ROBIN_CONDITION_FEATURES,
    V6DualRobinExample,
)
from rigno.heat3d_v6_global_context import (  # noqa: E402
    standardize_v6_contexts,
)
from rigno.models.rigno import RIGNO as GraphNeuralOperator  # noqa: E402
import run_heat3d_v1_medium_controlled_training_export as runner  # noqa: E402
from run_heat3d_v3_final_probe_checkpoint_smoke import (  # noqa: E402
    install_checkpoint_feature_hooks,
)


DEFAULT_PROBE = ROOT / "configs/heat3d_v6/v6_valid_common_probe4096.json"
DEFAULT_MANIFEST = (
    ROOT / "configs/heat3d_v6/v6_p1h_shared_support1024_manifest.json"
)
DEFAULT_OUTPUT_JSON = (
    ROOT / "configs/heat3d_v6/v6_common_valid_probe4096_results.json"
)
DEFAULT_OUTPUT_MD = ROOT / "docs/v6_common_valid_probe4096_results.md"
DATASET_ID = "heat3d_v6_p1h_shared_support1024_v0"
RUN_SPECS = {
    "V6_02_V5best": {
        "checkpoint_epoch": 406,
        "checkpoint_sha256": (
            "276dcb6a278602612f1b6d149fe9b0aea3795809be429c457ce8d8cb0297ee6b"
        ),
        "training_commit": "3c90d8a",
        "model_role": "P1g geometry-adaptive historical V5-best baseline",
    },
    "V6_03_V5best_P1h": {
        "checkpoint_epoch": 111,
        "checkpoint_sha256": (
            "3ad58c2b34a46481acb74722c80bdcadbf55a0d613bc25c4fe2d7646b91aa1f2"
        ),
        "training_commit": "950a1ce",
        "model_role": "canonical model candidate",
    },
    "V6_04_V5best_P1h_DualAttention": {
        "checkpoint_epoch": 111,
        "checkpoint_sha256": (
            "a127b020da14f3c7bdc544c0068ea755d9f58f1be0ee6cd627add914a6aec122"
        ),
        "training_commit": "950a1ce",
        "model_role": "DualAttention ablation",
    },
}


class CommonProbeError(RuntimeError):
    pass


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _array_sha256(value: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(contiguous.dtype).encode("utf-8"))
    digest.update(str(tuple(contiguous.shape)).encode("utf-8"))
    digest.update(contiguous.tobytes())
    return digest.hexdigest()


def _distribution(values: Sequence[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0 or not np.all(np.isfinite(array)):
        raise CommonProbeError("distribution requires finite nonempty values")
    return {
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "p05": float(np.quantile(array, 0.05)),
        "p95": float(np.quantile(array, 0.95)),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
    }


def _bc_features(coords: np.ndarray) -> np.ndarray:
    top = np.isclose(coords[:, 2], np.max(coords[:, 2]), atol=1.0e-15)
    bottom = np.isclose(coords[:, 2], np.min(coords[:, 2]), atol=1.0e-15)
    side = (
        np.isclose(coords[:, 0], np.min(coords[:, 0]), atol=1.0e-15)
        | np.isclose(coords[:, 0], np.max(coords[:, 0]), atol=1.0e-15)
        | np.isclose(coords[:, 1], np.min(coords[:, 1]), atol=1.0e-15)
        | np.isclose(coords[:, 1], np.max(coords[:, 1]), atol=1.0e-15)
    ) & ~top & ~bottom
    interior = ~(top | bottom | side)
    result = np.column_stack((top, bottom, side, interior)).astype(np.float64)
    if not np.array_equal(np.sum(result, axis=1), np.ones(len(coords))):
        raise CommonProbeError("probe BC flags are not one-hot")
    return result


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
    if manifest["dataset_id"] != DATASET_ID:
        raise CommonProbeError("manifest is not frozen P1h")
    valid_rows = [
        row for row in manifest["samples"] if row["split_role"] == "valid"
    ]
    if len(valid_rows) != 128:
        raise CommonProbeError("valid_iid sample count drifted")
    valid_ids = [str(row["sample_id"]) for row in valid_rows]
    valid_hash = hashlib.sha256("\n".join(valid_ids).encode("utf-8")).hexdigest()
    if valid_hash != probe["valid_sample_ids_sha256"]:
        raise CommonProbeError("probe valid sample ID binding drifted")

    archive_path = dataset_root / "full_fields.h5"
    if _sha256(archive_path) != probe["full_field_archive_sha256"]:
        raise CommonProbeError("full-field archive SHA256 mismatch")
    indices = np.asarray(probe["indices"], dtype=np.int32)
    if indices.shape != (4096,) or len(np.unique(indices)) != 4096:
        raise CommonProbeError("probe index shape/uniqueness drifted")

    examples: list[V6DualRobinExample] = []
    targets: dict[str, dict[str, np.ndarray]] = {}
    with h5py.File(archive_path, "r") as handle:
        archive_ids = [
            value.decode("utf-8") if isinstance(value, bytes) else str(value)
            for value in handle["samples/sample_id"][:]
        ]
        archive_index = {sample_id: index for index, sample_id in enumerate(archive_ids)}
        if len(archive_index) != 1024:
            raise CommonProbeError("archive sample IDs are not unique")
        coords = np.asarray(handle["mesh/coords"][:], dtype=np.float64)[indices]
        k_field = np.asarray(handle["mesh/k_diag"][:], dtype=np.float64)[indices]
        layer_id = np.asarray(handle["mesh/layer_id"][:], dtype=np.int32)[indices]
        control_volume = np.asarray(
            handle["mesh/control_volume"][:], dtype=np.float64
        )[indices]
        if _array_sha256(coords) != probe["coordinate_sha256"]:
            raise CommonProbeError("probe coordinate SHA256 mismatch")
        if _array_sha256(layer_id) != probe["layer_id_sha256"]:
            raise CommonProbeError("probe layer SHA256 mismatch")
        if _array_sha256(control_volume) != probe["control_volume_sha256"]:
            raise CommonProbeError("probe control-volume SHA256 mismatch")
        flags = _bc_features(coords)

        for row in valid_rows:
            sample_id = str(row["sample_id"])
            if sample_id not in archive_index:
                raise CommonProbeError(f"{sample_id}: missing from full-field archive")
            archive_row = archive_index[sample_id]
            sample_dir = dataset_root / str(row.get("sample_dir") or sample_id)
            meta = json.loads(
                (sample_dir / "sample_meta.json").read_text(encoding="utf-8")
            )
            if meta["split_role"] != "valid":
                raise CommonProbeError(f"{sample_id}: non-valid sample materialized")
            bc = meta["boundary_conditions"]
            top = bc["top"]
            bottom = bc["bottom"]
            if top["type"] != "robin" or bottom["type"] != "robin":
                raise CommonProbeError(f"{sample_id}: dual-Robin contract drifted")
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
                    np.full(4096, float(top["h_W_m2K"])),
                    np.full(4096, float(bottom["h_W_m2K"])),
                    np.full(
                        4096,
                        float(top["T_inf_K"]) - float(bottom["T_inf_K"]),
                    ),
                )
            )
            condition = np.concatenate(
                (k_field, q_field[:, None], flags, broadcasts), axis=1
            )
            if condition.shape != (4096, len(V6_DUAL_ROBIN_CONDITION_FEATURES)):
                raise CommonProbeError(f"{sample_id}: condition width drifted")
            enriched_meta = dict(meta)
            enriched_meta["split"] = "valid_iid"
            enriched_meta["v6_adapter"] = {
                "dataset_id": DATASET_ID,
                "manifest_split_role": "valid",
                "group_id": str(row["group_id"]),
                "reference_temperature_K": float(bottom["T_inf_K"]),
                "top_T_inf_K": float(top["T_inf_K"]),
                "bottom_T_inf_K": float(bottom["T_inf_K"]),
                "bottom_boundary_semantics": "robin_not_dirichlet",
                "operator_point_measure": "equal_weight_fixed_solver_probe4096",
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
        "control_volume": control_volume,
        "top_mask": flags[:, 0] > 0.5,
        "bottom_mask": flags[:, 1] > 0.5,
        "valid_sample_ids": valid_ids,
    }
    return examples, targets, public


def _prepare_groups(
    *,
    examples: Sequence[V6DualRobinExample],
    run_config: Mapping[str, Any],
    checkpoint: Mapping[str, Any],
    batch_size: int,
) -> list[dict[str, Any]]:
    stats = checkpoint["train_only_normalization"]
    model_config = runner._resolve_decoder_bypass_model_config(
        dict(checkpoint["model_config"]), stats
    )
    if model_config["native_output_mode"] != "native_shape_scale":
        raise CommonProbeError("common probe requires native shape-scale models")
    if model_config.get("scale_context_mode", "none") != "none":
        raise CommonProbeError("unregistered scale context in common probe")
    if model_config.get("scale_deepsets_mode", "none") != "none":
        raise CommonProbeError("unregistered DeepSets mode in common probe")
    builder = Heat3DGraphBuilder(**dict(run_config["graph_config"]))
    groups = runner._make_v6_padded_groups_with_progress(
        examples,
        stats,
        builder,
        "v6_common_valid_probe4096",
        True,
        "basic",
        int(run_config["graph_seed"]),
        batch_size=batch_size,
        drop_last=False,
    )

    context_payload = run_config.get("global_context") or {}
    standardizer = context_payload.get("standardizer") or {}
    if (
        not context_payload.get("enabled")
        or standardizer.get("fit_population") != "train_only"
        or int(standardizer.get("fit_sample_count", -1)) != 768
    ):
        raise CommonProbeError("global context is not frozen train-only/768")
    encoded = {
        example.sample_id: standardize_v6_contexts(
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


def _materialize_checkpoint_stats(
    checkpoint_stats: Mapping[str, Any],
) -> dict[str, Any]:
    stats = dict(checkpoint_stats)
    stats["feature_names"] = tuple(stats.get("feature_names") or ())
    if stats.get("condition_feature_transforms"):
        stats["condition_feature_transforms"] = tuple(
            stats["condition_feature_transforms"]
        )
    stats["target_delta_mean"] = jnp.asarray(
        np.asarray(stats["target_delta_mean"], dtype=np.float32).reshape(
            1, 1, 1, 1
        )
    )
    stats["target_delta_std"] = jnp.asarray(
        np.asarray(stats["target_delta_std"], dtype=np.float32).reshape(
            1, 1, 1, 1
        )
    )
    stats["condition_mean"] = jnp.asarray(
        np.asarray(stats["condition_mean"], dtype=np.float32).reshape(
            1, 1, 1, -1
        )
    )
    stats["condition_std"] = jnp.asarray(
        np.asarray(stats["condition_std"], dtype=np.float32).reshape(
            1, 1, 1, -1
        )
    )
    return stats


def _predict(
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
            raise CommonProbeError(f"missing frozen run artifact: {path}")
    if _sha256(checkpoint_path) != spec["checkpoint_sha256"]:
        raise CommonProbeError(f"{run_dir.name}: checkpoint SHA256 mismatch")
    run_config = json.loads(run_config_path.read_text(encoding="utf-8"))
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    checkpoint = runner._load_params_checkpoint(checkpoint_path)
    if (
        int(checkpoint["epoch"]) != int(spec["checkpoint_epoch"])
        or checkpoint["checkpoint_kind"] != "point_global_best"
        or str(checkpoint["git_commit"])[:7] != str(spec["training_commit"])[:7]
        or int(summary["point_global_best_epoch"]) != int(spec["checkpoint_epoch"])
    ):
        raise CommonProbeError(f"{run_dir.name}: checkpoint provenance mismatch")
    checkpoint_stats = _materialize_checkpoint_stats(
        checkpoint["train_only_normalization"]
    )
    runtime_checkpoint = dict(checkpoint)
    runtime_checkpoint["train_only_normalization"] = checkpoint_stats
    install_checkpoint_feature_hooks(checkpoint_stats)
    groups = _prepare_groups(
        examples=examples,
        run_config=run_config,
        checkpoint=runtime_checkpoint,
        batch_size=batch_size,
    )
    model_config = runner._resolve_decoder_bypass_model_config(
        dict(checkpoint["model_config"]), checkpoint_stats
    )
    model = GraphNeuralOperator(**model_config)
    params = runner._device_params(checkpoint["params"])
    predictions: dict[str, np.ndarray] = {}
    for group_index, group in enumerate(groups, start=1):
        output = runner._model_apply(model, params, group)
        recovered = np.asarray(output["raw_temperature"], dtype=np.float64)
        if not np.all(np.isfinite(recovered)):
            raise CommonProbeError(f"{run_dir.name}: non-finite prediction")
        for row, sample_id in enumerate(group["sample_ids"]):
            predictions[str(sample_id)] = recovered[row, 0, :, 0]
        print(
            f"[common-probe] {run_dir.name} group {group_index}/{len(groups)} "
            f"samples={len(predictions)}/{len(examples)}",
            flush=True,
        )
    if len(predictions) != len(examples):
        raise CommonProbeError(f"{run_dir.name}: prediction count drifted")
    return predictions, {
        "config_id": run_dir.name,
        "checkpoint_kind": "point_global_best",
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "checkpoint_sha256": _sha256(checkpoint_path),
        "training_commit": str(checkpoint["git_commit"]),
        "run_config_sha256": _sha256(run_config_path),
        "loss_summary_sha256": _sha256(summary_path),
        "parameter_count": int(checkpoint["param_count"]),
        "global_context_fit_population": "train_only",
        "global_context_fit_sample_count": 768,
        "edge_masking_inference_key": None,
        "batch_size": batch_size,
        "device": str(jax.devices()[0]),
    }


def _weighted_sums(
    error: np.ndarray,
    truth: np.ndarray,
    weights: np.ndarray,
    mask: np.ndarray | None = None,
) -> tuple[float, float, float, float]:
    selected = np.ones(len(error), dtype=bool) if mask is None else mask
    if not np.any(selected):
        return 0.0, 0.0, 0.0, 0.0
    w = weights[selected]
    e = error[selected]
    t = truth[selected]
    return (
        float(np.sum(w * np.square(e))),
        float(np.sum(w * np.square(t))),
        float(np.sum(w * e)),
        float(np.sum(w)),
    )


def _metrics(
    *,
    predictions: Mapping[str, np.ndarray],
    examples: Sequence[V6DualRobinExample],
    targets: Mapping[str, Mapping[str, np.ndarray]],
    public: Mapping[str, Any],
) -> dict[str, Any]:
    weights = np.asarray(public["control_volume"], dtype=np.float64)
    layer_id = np.asarray(public["layer_id"], dtype=np.int32)
    top_mask = np.asarray(public["top_mask"], dtype=bool)
    bottom_mask = np.asarray(public["bottom_mask"], dtype=bool)
    totals = {
        key: np.zeros(4, dtype=np.float64)
        for key in ("all", "source", "top", "bottom")
    }
    sample_relative = []
    peak_errors = []
    layer_mean_errors = []
    layer_drop_errors = []
    per_sample = []
    for example in examples:
        sample_id = example.sample_id
        reference = float(example.meta["v6_adapter"]["reference_temperature_K"])
        prediction = np.asarray(predictions[sample_id], dtype=np.float64) - reference
        truth = np.asarray(targets[sample_id]["deltaT_K"], dtype=np.float64)
        q = np.asarray(targets[sample_id]["q_W_m3"], dtype=np.float64)
        error = prediction - truth
        source_mask = q > 0.0
        row_sums = {}
        for label, mask in (
            ("all", None),
            ("source", source_mask),
            ("top", top_mask),
            ("bottom", bottom_mask),
        ):
            values = np.asarray(
                _weighted_sums(error, truth, weights, mask), dtype=np.float64
            )
            totals[label] += values
            row_sums[label] = values
        sample_rel = math.sqrt(row_sums["all"][0] / row_sums["all"][1])
        sample_relative.append(sample_rel)
        peak_error = float(np.max(prediction) - np.max(truth))
        peak_errors.append(peak_error)

        true_layer_means = []
        pred_layer_means = []
        for layer in range(9):
            mask = layer_id == layer
            w = weights[mask]
            true_layer_means.append(float(np.sum(w * truth[mask]) / np.sum(w)))
            pred_layer_means.append(
                float(np.sum(w * prediction[mask]) / np.sum(w))
            )
        layer_error = np.asarray(pred_layer_means) - np.asarray(true_layer_means)
        layer_mean_errors.extend(layer_error.tolist())
        true_drops = np.diff(np.asarray(true_layer_means))
        pred_drops = np.diff(np.asarray(pred_layer_means))
        drop_error = pred_drops - true_drops
        layer_drop_errors.extend(drop_error.tolist())
        per_sample.append(
            {
                "sample_id": sample_id,
                "sample_first_cv_relative_rmse_pct": float(100.0 * sample_rel),
                "cv_weighted_rmse_K": float(
                    math.sqrt(row_sums["all"][0] / row_sums["all"][3])
                ),
                "peak_error_K": peak_error,
                "source_region_cv_rmse_K": float(
                    math.sqrt(row_sums["source"][0] / row_sums["source"][3])
                ),
                "layer_mean_rmse_K": float(
                    np.sqrt(np.mean(np.square(layer_error)))
                ),
                "layer_drop_rmse_K": float(
                    np.sqrt(np.mean(np.square(drop_error)))
                ),
            }
        )

    def region(label: str) -> dict[str, float]:
        sse, energy, signed, weight = totals[label]
        return {
            "cv_weighted_rmse_K": float(math.sqrt(sse / weight)),
            "cv_weighted_relative_rmse_pct": float(
                100.0 * math.sqrt(sse / energy)
            ),
            "cv_weighted_bias_K": float(signed / weight),
        }

    peak = np.asarray(peak_errors, dtype=np.float64)
    result = {
        "point_global_cv_relative_rmse_pct": region("all")[
            "cv_weighted_relative_rmse_pct"
        ],
        "sample_first_cv_relative_rmse_pct": float(
            100.0 * np.mean(sample_relative)
        ),
        "raw_cv_weighted_rmse_K": region("all")["cv_weighted_rmse_K"],
        "peak": {
            "rmse_K": float(np.sqrt(np.mean(np.square(peak)))),
            "mae_K": float(np.mean(np.abs(peak))),
            "bias_K": float(np.mean(peak)),
        },
        "source_region": region("source"),
        "layer_mean": {
            "rmse_K": float(
                np.sqrt(np.mean(np.square(layer_mean_errors)))
            ),
            "absolute_error_distribution_K": _distribution(
                np.abs(layer_mean_errors)
            ),
        },
        "layer_drop": {
            "rmse_K": float(
                np.sqrt(np.mean(np.square(layer_drop_errors)))
            ),
            "absolute_error_distribution_K": _distribution(
                np.abs(layer_drop_errors)
            ),
        },
        "top_surface": region("top"),
        "bottom_surface": region("bottom"),
        "sample_count": 128,
        "node_count": 4096,
        "per_sample": per_sample,
    }
    numeric = np.asarray(
        [
            result["point_global_cv_relative_rmse_pct"],
            result["sample_first_cv_relative_rmse_pct"],
            result["raw_cv_weighted_rmse_K"],
            result["peak"]["rmse_K"],
            result["source_region"]["cv_weighted_rmse_K"],
            result["layer_mean"]["rmse_K"],
            result["layer_drop"]["rmse_K"],
            result["top_surface"]["cv_weighted_rmse_K"],
            result["bottom_surface"]["cv_weighted_rmse_K"],
        ]
    )
    if not np.all(np.isfinite(numeric)):
        raise CommonProbeError("common-probe metrics contain non-finite values")
    return result


def _markdown(payload: Mapping[str, Any]) -> str:
    lines = [
        "# V6 common-domain valid probe evaluation",
        "",
        "The support is a frozen, label-independent set of 4096 original solver nodes.",
        "Only `valid_iid` rows were read for q/temperature inference and metrics; test/hard",
        "roles remained sealed.",
        "",
        "| model | epoch | point-global CV % | sample-first CV % | raw CV RMSE K | peak RMSE K | source RMSE K | layer mean/drop K | top/bottom K |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for config_id in RUN_SPECS:
        row = payload["models"][config_id]
        metric = row["metrics"]
        lines.append(
            f"| {config_id} | {row['checkpoint']['checkpoint_epoch']} | "
            f"{metric['point_global_cv_relative_rmse_pct']:.6f} | "
            f"{metric['sample_first_cv_relative_rmse_pct']:.6f} | "
            f"{metric['raw_cv_weighted_rmse_K']:.6f} | "
            f"{metric['peak']['rmse_K']:.6f} | "
            f"{metric['source_region']['cv_weighted_rmse_K']:.6f} | "
            f"{metric['layer_mean']['rmse_K']:.6f}/"
            f"{metric['layer_drop']['rmse_K']:.6f} | "
            f"{metric['top_surface']['cv_weighted_rmse_K']:.6f}/"
            f"{metric['bottom_surface']['cv_weighted_rmse_K']:.6f} |"
        )
    ranking = payload["ranking_by_point_global_cv"]
    lines.extend(
        [
            "",
            "## Conclusion",
            "",
            f"- Common-domain point-global ranking: "
            f"{' < '.join(row['config_id'] for row in ranking)}.",
            f"- Canonical model candidate remains `V6_03_V5best_P1h`; "
            "`V6_04_V5best_P1h_DualAttention` remains an ablation regardless of",
            "this diagnostic ranking.",
            "- These values are a 4096-node solver-support diagnostic and do not replace",
            "the historical 1024-node checkpoint-selection metrics.",
        ]
    )
    return "\n".join(lines) + "\n"


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    probe = json.loads(args.probe.read_text(encoding="utf-8"))
    if (
        probe["status"] != "frozen"
        or probe["dataset_id"] != DATASET_ID
        or probe["evaluation_role"] != "valid_iid"
        or not probe["label_independent"]
        or probe["test_hard_accessed"]
    ):
        raise CommonProbeError("probe contract is not valid-only/label-independent")
    examples, targets, public = _load_valid_examples(
        dataset_root=args.dataset.resolve(),
        manifest_path=args.manifest.resolve(),
        probe=probe,
    )
    models = {}
    for config_id, spec in RUN_SPECS.items():
        run_dir = args.input_root.resolve() / config_id
        predictions, checkpoint = _predict(
            run_dir=run_dir,
            spec=spec,
            examples=examples,
            batch_size=args.batch_size,
        )
        models[config_id] = {
            "model_role": spec["model_role"],
            "checkpoint": checkpoint,
            "metrics": _metrics(
                predictions=predictions,
                examples=examples,
                targets=targets,
                public=public,
            ),
        }
    ranking = sorted(
        (
            {
                "config_id": config_id,
                "point_global_cv_relative_rmse_pct": row["metrics"][
                    "point_global_cv_relative_rmse_pct"
                ],
            }
            for config_id, row in models.items()
        ),
        key=lambda row: row["point_global_cv_relative_rmse_pct"],
    )
    return {
        "schema_version": "heat3d_v6_common_valid_probe_results_v1",
        "status": "passed",
        "dataset_id": DATASET_ID,
        "evaluation_role": "valid_iid",
        "test_hard_accessed": False,
        "training_executed": False,
        "checkpoint_selection_modified": False,
        "evaluator_source": {
            "parent_git_commit": (
                "fa3a70ba809b887d226ec2ca6a6be2edf97e685f"
            ),
            "script": "scripts/evaluate_heat3d_v6_common_valid_probe.py",
            "script_sha256": _sha256(Path(__file__)),
        },
        "probe": {
            key: probe[key]
            for key in (
                "probe_id",
                "node_count",
                "sample_count",
                "selection_policy",
                "label_independent",
                "support_index_sha256",
                "coordinate_sha256",
                "graph_sha256",
                "metric_weight_policy",
            )
        },
        "metric_formulas": {
            "point_global_cv": (
                "sqrt(sum_sample,node(cv*error^2)/"
                "sum_sample,node(cv*true_deltaT^2))"
            ),
            "sample_first_cv": (
                "mean_sample sqrt(sum_node(cv*error^2)/"
                "sum_node(cv*true_deltaT^2))"
            ),
            "raw_cv_rmse_K": (
                "sqrt(sum_sample,node(cv*error^2)/sum_sample,node(cv))"
            ),
        },
        "models": models,
        "ranking_by_point_global_cv": ranking,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--probe", type=Path, default=DEFAULT_PROBE)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--output-md", type=Path, default=DEFAULT_OUTPUT_MD)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    if args.batch_size < 1:
        raise CommonProbeError("--batch-size must be positive")
    payload = evaluate(args)
    if args.write:
        args.output_json.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        args.output_md.write_text(_markdown(payload), encoding="utf-8")
    print(
        json.dumps(
            {
                "status": payload["status"],
                "ranking": payload["ranking_by_point_global_cv"],
                "test_hard_accessed": payload["test_hard_accessed"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
