#!/usr/bin/env python3
"""Audit the V4 P1 active training path without training.

The audit follows the registry-resolved V4 baseline into the current V1
controlled runner, then builds a small read-only feature manifest from local
sample subsets when they are available. It does not create configs, launch
training, evaluate checkpoints, or write large artifacts.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

try:
    import yaml
except ImportError as exc:  # pragma: no cover - environment issue.
    raise SystemExit("PyYAML is required for the V4 P1 audit.") from exc


REPO_ROOT = Path(
    os.environ.get("HEAT3D_REPO_ROOT", Path(__file__).resolve().parents[1])
).resolve()
SCRIPTS_DIR = REPO_ROOT / "scripts"
for path in (REPO_ROOT, SCRIPTS_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from check_heat3d_v4_registry import (  # noqa: E402
    DEFAULT_REGISTRY,
    build_inherited_yaml,
    load_registry,
    registry_rows,
    resolve_inherited_yaml,
)
from rigno.dataset_Heat3D_v1 import Heat3DV1MetadataDataset  # noqa: E402
from rigno.heat3d_v1_native_supervised import (  # noqa: E402
    V1SteadyConditionInput,
    V1SteadySupervisedExampleNative,
    V1SteadyTarget,
)
from rigno.heat3d_v1_normalization import (  # noqa: E402
    legacy_train_only_stats as _train_only_stats,
    normalize_condition,
    normalize_coords as _normalize_coords,
    normalize_target_delta,
)
from rigno.heat3d_v1_training_semantics import (  # noqa: E402
    build_legacy_zero_delta_bridge as _bridge_for,
)
from rigno.heat3d_v1_supervised import PHYSICS_LABEL_SUPERVISED_STAGES  # noqa: E402
from rigno.heat3d_v2_runner_command import (  # noqa: E402
    TRAINING_SCRIPT,
    build_v2_command_plan,
)


AUDIT_SCHEMA_VERSION = "heat3d_v4_p1_training_path_audit_v0"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "output" / "heat3d_v4_p1_audit"
DEFAULT_FINAL_PROBE_SUBSET = (
    REPO_ROOT
    / "data"
    / "heat3d-thermal-simulation"
    / "subsets"
    / "v3_final_target_probe_v0"
)
AUDIT_ALLOWED_STAGES = tuple(PHYSICS_LABEL_SUPERVISED_STAGES) + (
    "physics_label_v3_final_target_probe_v0",
)
BC_FLAG_NAMES = ("is_top", "is_bottom", "is_side", "is_interior")
BC_SCALAR_NAMES = (
    "top_h",
    "top_T_inf_minus_T_ref",
    "bottom_T_fixed_minus_T_ref",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-id", default="V4_baseline")
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument(
        "--subset",
        type=Path,
        default=None,
        help=(
            "Optional local subset for range audit. Defaults to the "
            "registry-resolved V4 dataset path, which may live only on the "
            "training server."
        ),
    )
    parser.add_argument("--final-probe-subset", type=Path, default=DEFAULT_FINAL_PROBE_SUBSET)
    parser.add_argument(
        "--max-samples-per-split",
        type=int,
        default=16,
        help="Maximum samples per split for range audit; use 0 for all samples.",
    )
    parser.add_argument(
        "--amplitude-predictions",
        type=Path,
        default=None,
        help="Optional final-probe prediction .npz for amplitude/shape diagnostics.",
    )
    parser.add_argument(
        "--amplitude-prediction-dir",
        type=Path,
        default=None,
        help="Optional directory containing final-probe *_pred.npy prediction files.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    row, config, generated_source = resolve_config(args.registry, args.config_id)
    subset = args.subset if args.subset is not None else _repo_path(config["dataset"]["subset_path"])
    output_dir = _repo_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    plan = build_v2_command_plan(config)
    local_examples, local_gaps = load_examples(
        subset,
        label="registry_or_proxy_subset",
        max_samples_per_split=args.max_samples_per_split,
    )
    final_examples, final_gaps = load_examples(
        args.final_probe_subset,
        label="final_probe",
        force_split="final_probe",
        max_samples_per_split=args.max_samples_per_split,
    )

    split_examples = split_examples_by_label(local_examples)
    if final_examples:
        split_examples["final_probe"] = final_examples

    train_examples = split_examples.get("train", [])
    stats = _train_only_stats(train_examples) if train_examples else None
    summaries = {
        split: summarize_examples(examples, stats=stats)
        for split, examples in sorted(split_examples.items())
    }
    range_comparisons = compare_splits_to_train(summaries)
    amplitude = amplitude_diagnostics(
        final_examples,
        prediction_npz=args.amplitude_predictions,
        prediction_dir=args.amplitude_prediction_dir,
    )

    training_path_audit = {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "config_id": args.config_id,
        "config_row": row,
        "config_source": generated_source,
        "resolved_dataset_subset": str(_repo_path(config["dataset"]["subset_path"])),
        "audit_dataset_subset": str(subset),
        "final_probe_subset": str(args.final_probe_subset),
        "active_training_path": active_training_path(config, plan),
        "active_batch_manifest": active_batch_manifest(train_examples, stats),
        "normalization_audit": normalization_audit(stats),
        "range_and_ood_audit": {
            "data_scope": data_scope_note(subset, config),
            "summaries": summaries,
            "comparisons_to_train": range_comparisons,
            "gaps": local_gaps + final_gaps,
        },
        "amplitude_diagnostics": amplitude,
        "artifact_record_gaps": artifact_record_gaps(),
        "provenance_field_plan": provenance_field_plan(),
    }
    feature_manifest = build_feature_manifest(stats, summaries, local_gaps + final_gaps)

    training_path_path = output_dir / "training_path_audit.json"
    feature_manifest_path = output_dir / "feature_manifest.json"
    feature_csv_path = output_dir / "feature_manifest.csv"
    write_json(training_path_path, training_path_audit)
    write_json(feature_manifest_path, feature_manifest)
    write_feature_csv(feature_csv_path, feature_manifest["entries"])
    if amplitude.get("available"):
        write_amplitude_csv(output_dir / "amplitude_diagnostics.csv", amplitude["rows"])

    print(f"wrote {training_path_path}")
    print(f"wrote {feature_manifest_path}")
    print(f"wrote {feature_csv_path}")
    print("training_started=false")
    return 0


def _repo_path(path: Path | str) -> Path:
    path = Path(path)
    return path if path.is_absolute() else REPO_ROOT / path


def resolve_config(
    registry_path: Path,
    config_id: str,
) -> tuple[dict[str, str], dict[str, Any], str]:
    registry = load_registry(_repo_path(registry_path))
    rows = registry_rows(registry)
    row = next((item for item in rows if item["config_id"] == config_id), None)
    if row is None:
        raise ValueError(f"config_id not found in registry: {config_id}")
    generated_path = _repo_path(row["generated_yaml"])
    if generated_path.is_file():
        with generated_path.open("r", encoding="utf-8") as file:
            inherited = yaml.safe_load(file)
        source = str(generated_path)
    else:
        inherited = build_inherited_yaml(row)
        source = "registry_build_inherited_yaml_fallback"
    return row, resolve_inherited_yaml(inherited, generated_path), source


def sample_root_from_subset(path: Path) -> Path:
    path = _repo_path(path)
    samples = path / "samples"
    return samples if samples.is_dir() else path


def load_examples(
    subset: Path,
    *,
    label: str,
    max_samples_per_split: int,
    force_split: str | None = None,
) -> tuple[list[V1SteadySupervisedExampleNative], list[dict[str, Any]]]:
    gaps: list[dict[str, Any]] = []
    sample_root = sample_root_from_subset(subset)
    if not sample_root.is_dir():
        return [], [
            {
                "scope": label,
                "path": str(sample_root),
                "gap": "sample_root_not_available_locally",
            }
        ]

    try:
        dataset = Heat3DV1MetadataDataset(
            sample_root,
            k_encoding_mode="diag3",
            allowed_stages=AUDIT_ALLOWED_STAGES,
            boundary_mask_fallback=True,
        )
    except Exception as exc:  # pragma: no cover - audit should preserve gaps.
        return [], [
            {
                "scope": label,
                "path": str(sample_root),
                "gap": "loader_failed",
                "error": f"{type(exc).__name__}: {exc}",
            }
        ]

    by_split: dict[str, list[V1SteadySupervisedExampleNative]] = defaultdict(list)
    for sample in dataset.samples:
        sample_dir = Path(sample["sample_dir"])
        temperature_path = sample_dir / "temperature.npy"
        if not temperature_path.is_file():
            gaps.append(
                {
                    "scope": label,
                    "sample_id": sample.get("sample_id"),
                    "gap": "temperature_label_missing",
                    "path": str(temperature_path),
                }
            )
            continue
        temperature = np.asarray(np.load(temperature_path), dtype=np.float64)
        if temperature.ndim != 2 or temperature.shape[1] != 1:
            gaps.append(
                {
                    "scope": label,
                    "sample_id": sample.get("sample_id"),
                    "gap": "temperature_shape_invalid",
                    "shape": list(temperature.shape),
                }
            )
            continue

        meta = dict(sample["meta"])
        if force_split is not None:
            meta["split"] = force_split
        condition = V1SteadyConditionInput(
            coords=np.asarray(sample["coords"], dtype=np.float64),
            condition_features=np.asarray(sample["physics_input"].features, dtype=np.float64),
            condition_feature_names=tuple(sample["physics_input"].feature_names),
            k_encoding_mode="diag3",
        )
        example = V1SteadySupervisedExampleNative(
            sample_id=str(sample["sample_id"]),
            condition=condition,
            target=V1SteadyTarget(target_u=temperature),
            meta=meta,
        )
        by_split[str(meta.get("split") or "unknown")].append(example)

    examples: list[V1SteadySupervisedExampleNative] = []
    for split in sorted(by_split):
        selected = sorted(by_split[split], key=lambda item: item.sample_id)
        if max_samples_per_split and max_samples_per_split > 0:
            selected = selected[:max_samples_per_split]
        examples.extend(selected)
    return examples, gaps


def split_examples_by_label(
    examples: list[V1SteadySupervisedExampleNative],
) -> dict[str, list[V1SteadySupervisedExampleNative]]:
    splits: dict[str, list[V1SteadySupervisedExampleNative]] = defaultdict(list)
    for example in examples:
        splits[str(example.meta.get("split") or "unknown")].append(example)
    return dict(splits)


def active_training_path(config: dict[str, Any], plan: dict[str, Any]) -> dict[str, Any]:
    command = plan.get("training_command", [])
    return {
        "registry": "configs/heat3d_v4/v4_run_registry.json",
        "generated_yaml": "configs/heat3d_v4/generated/V4_baseline.yaml",
        "prepare_entry": "scripts/prepare_heat3d_v4_run.py",
        "launch_entry": "scripts/run_heat3d_v4_config.py",
        "command_builder": "rigno/heat3d_v2_runner_command.py",
        "runner_script": TRAINING_SCRIPT,
        "actual_loader": (
            "Heat3DV1NativeSupervisedDataset -> Heat3DV1SupervisedDataset "
            "-> Heat3DV1MetadataDataset"
        ),
        "native_semantics": "coords + k(x) + q(x) + BC -> target temperature T(x)",
        "legacy_bridge": "relative_bc_features + zero_delta_u_bridge",
        "model_call": "RIGNO.apply(inputs=Inputs(u, c, x_inp, x_out), graphs=...)",
        "loss_target": "normalized DeltaT = (T - T_ref - train_mean_deltaT) / train_std_deltaT",
        "prediction_recovery": "T_pred = T_ref + pred_normalized * train_std_deltaT + train_mean_deltaT",
        "selection_metric": config.get("export", {}).get("selection_metric"),
        "dry_run_training_command": command,
        "unmapped_runner_fields": plan.get("unmapped_fields", []),
        "non_execution": plan.get("non_execution_note"),
    }


def active_batch_manifest(
    train_examples: list[V1SteadySupervisedExampleNative],
    stats: dict[str, Any] | None,
) -> dict[str, Any]:
    if not train_examples or stats is None:
        return {
            "available": False,
            "gap": "no local train examples for active batch construction",
        }
    example = sorted(train_examples, key=lambda item: item.sample_id)[0]
    bridge = _bridge_for(example)
    raw_u = np.asarray(bridge.legacy_inputs.u, dtype=np.float64)
    raw_c = np.asarray(bridge.legacy_inputs.c, dtype=np.float64)
    raw_x = np.asarray(bridge.legacy_inputs.x_inp, dtype=np.float64)
    target_delta = np.asarray(bridge.target_delta_u, dtype=np.float64)
    c_norm = np.asarray(normalize_condition(raw_c, stats), dtype=np.float64)
    target_norm = np.asarray(normalize_target_delta(target_delta, stats), dtype=np.float64)
    x_norm = np.asarray(_normalize_coords(raw_x, stats), dtype=np.float64)
    return {
        "available": True,
        "sample_id": example.sample_id,
        "x_inp": describe_tensor(x_norm, "normalized coords in [-1,1] from train coord min/span"),
        "x_out": describe_tensor(x_norm, "same normalized coords as x_inp"),
        "u": describe_tensor(raw_u, "zero_delta field; not z-scored"),
        "c": describe_tensor(c_norm, "per-feature train z-score of raw c"),
        "target": describe_tensor(target_norm, "normalized DeltaT supervision"),
        "raw_target_temperature": describe_tensor(
            np.asarray(example.target.target_u, dtype=np.float64).reshape(1, 1, -1, 1),
            "raw T(x) label in K",
        ),
        "raw_target_deltaT": describe_tensor(target_delta, "T(x) - T_ref in K"),
        "feature_names": list(bridge.condition_feature_names),
        "t_ref_value": float(bridge.t_ref_value),
        "t_ref_source": bridge.t_ref_source,
    }


def describe_tensor(array: np.ndarray, note: str) -> dict[str, Any]:
    return {
        "shape": list(array.shape),
        "range": scalar_stats(array),
        "note": note,
    }


def summarize_examples(
    examples: list[V1SteadySupervisedExampleNative],
    *,
    stats: dict[str, Any] | None,
) -> dict[str, Any]:
    if not examples:
        return {"available": False, "sample_count": 0}

    feature_names: tuple[str, ...] | None = None
    c_values: list[np.ndarray] = []
    u_values: list[np.ndarray] = []
    target_t_values: list[np.ndarray] = []
    delta_values: list[np.ndarray] = []
    t_refs: list[float] = []
    coord_values: list[np.ndarray] = []
    extents: list[np.ndarray] = []
    aspect_ratios: list[float] = []
    normalized_c_values: list[np.ndarray] = []
    normalized_target_values: list[np.ndarray] = []
    normalized_coord_values: list[np.ndarray] = []

    for example in examples:
        bridge = _bridge_for(example)
        if feature_names is None:
            feature_names = bridge.condition_feature_names
        elif feature_names != bridge.condition_feature_names:
            raise ValueError("feature-name mismatch during audit")

        coords = np.asarray(example.condition.coords, dtype=np.float64)
        extent = np.ptp(coords, axis=0)
        positive_extent = extent[extent > 0.0]
        aspect = (
            float(np.max(positive_extent) / np.min(positive_extent))
            if positive_extent.size
            else None
        )
        c_raw = np.asarray(bridge.legacy_inputs.c, dtype=np.float64)
        u_raw = np.asarray(bridge.legacy_inputs.u, dtype=np.float64)
        target_delta = np.asarray(bridge.target_delta_u, dtype=np.float64)
        target_t = np.asarray(example.target.target_u, dtype=np.float64)

        c_values.append(c_raw.reshape(-1, c_raw.shape[-1]))
        u_values.append(u_raw.reshape(-1, u_raw.shape[-1]))
        target_t_values.append(target_t.reshape(-1, 1))
        delta_values.append(target_delta.reshape(-1, 1))
        coord_values.append(coords)
        extents.append(extent)
        if aspect is not None:
            aspect_ratios.append(aspect)
        t_refs.append(float(bridge.t_ref_value))

        if stats is not None:
            normalized_c_values.append(
                np.asarray(normalize_condition(c_raw, stats), dtype=np.float64).reshape(
                    -1, c_raw.shape[-1]
                )
            )
            normalized_target_values.append(
                np.asarray(normalize_target_delta(target_delta, stats), dtype=np.float64).reshape(-1, 1)
            )
            normalized_coord_values.append(
                np.asarray(
                    _normalize_coords(np.asarray(bridge.legacy_inputs.x_inp, dtype=np.float64), stats),
                    dtype=np.float64,
                ).reshape(-1, coords.shape[-1])
            )

    c_all = np.concatenate(c_values, axis=0)
    target_t_all = np.concatenate(target_t_values, axis=0)
    delta_all = np.concatenate(delta_values, axis=0)
    coord_all = np.concatenate(coord_values, axis=0)
    extents_all = np.vstack(extents)
    names = tuple(feature_names or ())
    by_feature = {
        name: scalar_stats(c_all[:, index])
        for index, name in enumerate(names)
    }
    feature_index = {name: index for index, name in enumerate(names)}

    summary = {
        "available": True,
        "sample_count": len(examples),
        "point_count": int(sum(example.condition.coords.shape[0] for example in examples)),
        "sample_ids": [example.sample_id for example in examples],
        "feature_names": list(names),
        "u": {
            "range": scalar_stats(np.concatenate(u_values, axis=0)),
            "max_abs": float(np.max(np.abs(np.concatenate(u_values, axis=0)))),
            "all_zero": bool(np.allclose(np.concatenate(u_values, axis=0), 0.0)),
        },
        "c_by_feature": by_feature,
        "k_range": selected_feature_stats(c_all, names, [name for name in names if name.startswith("k_")]),
        "q_range": selected_feature_stats(c_all, names, ["q"]),
        "bc_flag_distribution": {
            name: {
                "range": by_feature[name],
                "mean_fraction": float(np.mean(c_all[:, feature_index[name]])),
            }
            for name in BC_FLAG_NAMES
            if name in feature_index
        },
        "bc_scalar_range": {
            name: by_feature[name]
            for name in BC_SCALAR_NAMES
            if name in feature_index
        },
        "geometry": {
            "coord_range_m": {
                "x": scalar_stats(coord_all[:, 0]),
                "y": scalar_stats(coord_all[:, 1]),
                "z": scalar_stats(coord_all[:, 2]),
            },
            "extent_m_by_sample": {
                "x": scalar_stats(extents_all[:, 0]),
                "y": scalar_stats(extents_all[:, 1]),
                "z": scalar_stats(extents_all[:, 2]),
            },
            "aspect_ratio_max_over_min_positive_extent": scalar_stats(np.asarray(aspect_ratios)),
        },
        "target_temperature_K": scalar_stats(target_t_all),
        "target_deltaT_K": scalar_stats(delta_all),
        "t_ref_values_K": scalar_stats(np.asarray(t_refs, dtype=np.float64)),
    }
    if normalized_c_values:
        summary["normalized_c_range"] = scalar_stats(np.concatenate(normalized_c_values, axis=0))
        summary["normalized_target_deltaT_range"] = scalar_stats(
            np.concatenate(normalized_target_values, axis=0)
        )
        summary["normalized_coord_range"] = scalar_stats(np.concatenate(normalized_coord_values, axis=0))
    return summary


def selected_feature_stats(
    c_all: np.ndarray,
    names: tuple[str, ...],
    selected_names: list[str],
) -> dict[str, Any]:
    indices = [names.index(name) for name in selected_names if name in names]
    if not indices:
        return {"available": False, "feature_names": selected_names}
    return {
        "available": True,
        "feature_names": [names[index] for index in indices],
        "range": scalar_stats(c_all[:, indices]),
    }


def scalar_stats(values: np.ndarray) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        return {"available": False, "count": int(array.size)}
    return {
        "available": True,
        "count": int(finite.size),
        "min": float(np.min(finite)),
        "p01": float(np.percentile(finite, 1)),
        "p05": float(np.percentile(finite, 5)),
        "p10": float(np.percentile(finite, 10)),
        "p25": float(np.percentile(finite, 25)),
        "max": float(np.max(finite)),
        "mean": float(np.mean(finite)),
        "median": float(np.median(finite)),
        "p75": float(np.percentile(finite, 75)),
        "p90": float(np.percentile(finite, 90)),
        "p95": float(np.percentile(finite, 95)),
        "p99": float(np.percentile(finite, 99)),
        "std": float(np.std(finite)),
    }


def amplitude_diagnostics(
    final_examples: list[V1SteadySupervisedExampleNative],
    *,
    prediction_npz: Path | None,
    prediction_dir: Path | None,
) -> dict[str, Any]:
    if not final_examples:
        return {
            "available": False,
            "gap": "no final-probe examples available for amplitude diagnostics",
        }
    predictions, prediction_gaps = load_prediction_sources(
        prediction_npz=prediction_npz,
        prediction_dir=prediction_dir,
    )
    if not predictions:
        return {
            "available": False,
            "gap": "no final-probe prediction source provided or readable",
            "source_gaps": prediction_gaps,
            "expected_inputs": [
                "--amplitude-predictions path/to/s5_probe_predictions.npz",
                "--amplitude-prediction-dir path/to/predictions",
            ],
        }

    rows = []
    gaps = list(prediction_gaps)
    for example in sorted(final_examples, key=lambda item: item.sample_id):
        probe_id = str(example.meta.get("probe_id") or probe_id_from_sample_id(example.sample_id))
        candidates = (
            example.sample_id,
            Path(str(example.meta.get("sample_dir", ""))).name,
            probe_id,
            f"{probe_id}_pred",
            f"{probe_id}_pred.npy",
        )
        prediction = first_prediction(predictions, candidates)
        if prediction is None:
            gaps.append(
                {
                    "sample_id": example.sample_id,
                    "probe_id": probe_id,
                    "gap": "prediction_not_found_for_sample",
                    "candidate_keys": [item for item in candidates if item],
                }
            )
            continue
        rows.append(amplitude_row(example, prediction, probe_id))

    if not rows:
        return {
            "available": False,
            "gap": "prediction sources were readable but did not match final-probe samples",
            "source_gaps": gaps,
        }
    return {
        "available": True,
        "rows": rows,
        "summary": summarize_amplitude_rows(rows),
        "source_gaps": gaps,
        "interpretation_rule": (
            "High centered/shape correlation with low scale_ratio or range_ratio "
            "indicates amplitude failure more than shape failure."
        ),
    }


def load_prediction_sources(
    *,
    prediction_npz: Path | None,
    prediction_dir: Path | None,
) -> tuple[dict[str, np.ndarray], list[dict[str, Any]]]:
    predictions: dict[str, np.ndarray] = {}
    gaps: list[dict[str, Any]] = []
    if prediction_npz is not None:
        path = _repo_path(prediction_npz)
        if path.is_file():
            with np.load(path) as loaded:
                for key in loaded.files:
                    predictions[str(key)] = np.asarray(loaded[key], dtype=np.float64)
        else:
            gaps.append({"path": str(path), "gap": "prediction_npz_not_found"})
    if prediction_dir is not None:
        path = _repo_path(prediction_dir)
        if path.is_dir():
            for item in sorted(path.glob("*.npy")):
                predictions[item.stem] = np.asarray(np.load(item), dtype=np.float64)
                predictions[item.name] = predictions[item.stem]
        else:
            gaps.append({"path": str(path), "gap": "prediction_dir_not_found"})
    return predictions, gaps


def first_prediction(
    predictions: dict[str, np.ndarray],
    candidates: tuple[str, ...],
) -> np.ndarray | None:
    for candidate in candidates:
        if candidate and candidate in predictions:
            return predictions[candidate]
    return None


def probe_id_from_sample_id(sample_id: str) -> str:
    for token in str(sample_id).replace("-", "_").split("_"):
        if token.startswith("P") and token[1:].isdigit():
            return token
    return str(sample_id)


def amplitude_row(
    example: V1SteadySupervisedExampleNative,
    prediction: np.ndarray,
    probe_id: str,
) -> dict[str, Any]:
    label = np.asarray(example.target.target_u, dtype=np.float64).reshape(-1)
    pred = np.asarray(prediction, dtype=np.float64).reshape(-1)
    if pred.shape != label.shape:
        raise ValueError(
            f"{example.sample_id}: prediction shape {pred.shape} != label shape {label.shape}"
        )
    bridge = _bridge_for(example)
    t_ref = float(bridge.t_ref_value)
    label_delta = label - t_ref
    pred_delta = pred - t_ref
    error = pred - label
    label_delta_peak = float(np.max(label_delta))
    pred_delta_peak = float(np.max(pred_delta))
    label_delta_range = float(np.max(label_delta) - np.min(label_delta))
    pred_delta_range = float(np.max(pred_delta) - np.min(pred_delta))
    return {
        "sample_id": example.sample_id,
        "probe_id": probe_id,
        "RMSE_K": rmse(error),
        "MAE_K": float(np.mean(np.abs(error))),
        "relRMSE_DeltaT": safe_ratio(rmse(error), rmse(label_delta)),
        "peak_error_K": float(pred_delta_peak - label_delta_peak),
        "abs_peak_error_K": float(abs(pred_delta_peak - label_delta_peak)),
        "mean_bias_K": float(np.mean(error)),
        "pred_peak_K": float(np.max(pred)),
        "label_peak_K": float(np.max(label)),
        "pred_range_K": float(np.max(pred) - np.min(pred)),
        "label_range_K": float(np.max(label) - np.min(label)),
        "pred_deltaT_peak_K": pred_delta_peak,
        "label_deltaT_peak_K": label_delta_peak,
        "pred_deltaT_range_K": pred_delta_range,
        "label_deltaT_range_K": label_delta_range,
        "scale_ratio": safe_ratio(pred_delta_peak, label_delta_peak),
        "range_ratio": safe_ratio(pred_delta_range, label_delta_range),
        "centered_corr": centered_corr(label_delta, pred_delta),
        "shape_corr": centered_corr(label_delta, pred_delta),
    }


def summarize_amplitude_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    numeric_keys = sorted(
        key
        for key in rows[0]
        if key not in {"sample_id", "probe_id"} and isinstance(rows[0].get(key), (int, float))
    )
    summary = {
        key: scalar_stats(
            np.asarray(
                [row[key] for row in rows if row.get(key) is not None],
                dtype=np.float64,
            )
        )
        for key in numeric_keys
    }
    centered = summary.get("centered_corr", {})
    scale = summary.get("scale_ratio", {})
    range_ratio = summary.get("range_ratio", {})
    summary["failure_mode_hint"] = failure_mode_hint(centered, scale, range_ratio)
    return summary


def failure_mode_hint(
    centered_corr_stats: dict[str, Any],
    scale_ratio_stats: dict[str, Any],
    range_ratio_stats: dict[str, Any],
) -> str:
    centered = centered_corr_stats.get("median")
    scale = scale_ratio_stats.get("median")
    range_ratio = range_ratio_stats.get("median")
    if centered is None or scale is None or range_ratio is None:
        return "insufficient_metrics"
    if centered >= 0.75 and (scale < 0.75 or range_ratio < 0.75):
        return "amplitude_failure_likely_shape_partly_preserved"
    if centered < 0.5:
        return "shape_failure_likely"
    return "mixed_or_mild_amplitude_shape_issue"


def rmse(values: np.ndarray) -> float:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    return float(np.sqrt(np.mean(np.square(array))))


def safe_ratio(numerator: float, denominator: float) -> float | None:
    if abs(float(denominator)) < 1.0e-12:
        return None
    return float(numerator) / float(denominator)


def centered_corr(label: np.ndarray, pred: np.ndarray) -> float | None:
    label_centered = np.asarray(label, dtype=np.float64).reshape(-1) - float(np.mean(label))
    pred_centered = np.asarray(pred, dtype=np.float64).reshape(-1) - float(np.mean(pred))
    denom = float(np.linalg.norm(label_centered) * np.linalg.norm(pred_centered))
    if denom < 1.0e-12:
        return None
    return float(np.dot(label_centered, pred_centered) / denom)


def compare_splits_to_train(summaries: dict[str, dict[str, Any]]) -> dict[str, Any]:
    train = summaries.get("train")
    if not train or not train.get("available"):
        return {
            "available": False,
            "gap": "no train summary; cannot compute train-relative OOD flags",
        }
    comparisons = {}
    for split, summary in summaries.items():
        if split == "train" or not summary.get("available"):
            continue
        comparisons[split] = {
            "k_outside_train": outside_train(train["k_range"], summary["k_range"]),
            "q_outside_train": outside_train(train["q_range"], summary["q_range"]),
            "bc_scalar_outside_train": compare_named_ranges(
                train["bc_scalar_range"], summary["bc_scalar_range"]
            ),
            "bc_flag_distribution_shift": compare_bc_flag_means(
                train["bc_flag_distribution"], summary["bc_flag_distribution"]
            ),
            "geometry_extent_outside_train": compare_named_ranges(
                train["geometry"]["extent_m_by_sample"],
                summary["geometry"]["extent_m_by_sample"],
            ),
            "aspect_ratio_outside_train": outside_train(
                train["geometry"]["aspect_ratio_max_over_min_positive_extent"],
                summary["geometry"]["aspect_ratio_max_over_min_positive_extent"],
            ),
            "target_temperature_outside_train": outside_train(
                train["target_temperature_K"], summary["target_temperature_K"]
            ),
            "target_deltaT_outside_train": outside_train(
                train["target_deltaT_K"], summary["target_deltaT_K"]
            ),
        }
    return comparisons


def outside_train(train_stats: dict[str, Any], candidate_stats: dict[str, Any]) -> dict[str, Any]:
    if train_stats.get("range"):
        train_stats = train_stats["range"]
    if candidate_stats.get("range"):
        candidate_stats = candidate_stats["range"]
    if not train_stats.get("available") or not candidate_stats.get("available"):
        return {"available": False}
    low = float(candidate_stats["min"]) < float(train_stats["min"])
    high = float(candidate_stats["max"]) > float(train_stats["max"])
    return {
        "available": True,
        "outside": bool(low or high),
        "below_train_min": bool(low),
        "above_train_max": bool(high),
        "train_min": float(train_stats["min"]),
        "train_max": float(train_stats["max"]),
        "candidate_min": float(candidate_stats["min"]),
        "candidate_max": float(candidate_stats["max"]),
    }


def compare_named_ranges(
    train_ranges: dict[str, dict[str, Any]],
    candidate_ranges: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    result = {}
    for name, candidate in candidate_ranges.items():
        train = train_ranges.get(name)
        result[name] = outside_train(train or {}, candidate)
    return result


def compare_bc_flag_means(
    train_flags: dict[str, dict[str, Any]],
    candidate_flags: dict[str, dict[str, Any]],
    *,
    tolerance: float = 0.05,
) -> dict[str, Any]:
    result = {}
    for name, candidate in candidate_flags.items():
        train = train_flags.get(name)
        train_mean = None if train is None else train.get("mean_fraction")
        candidate_mean = candidate.get("mean_fraction")
        if train_mean is None or candidate_mean is None:
            result[name] = {"available": False}
            continue
        diff = float(candidate_mean) - float(train_mean)
        result[name] = {
            "available": True,
            "outside": abs(diff) > tolerance,
            "train_mean_fraction": float(train_mean),
            "candidate_mean_fraction": float(candidate_mean),
            "difference": diff,
            "tolerance": float(tolerance),
        }
    return result


def normalization_audit(stats: dict[str, Any] | None) -> dict[str, Any]:
    if stats is None:
        return {
            "available": False,
            "code_path": "stats unavailable because no local train examples were loaded",
            "confirmed_from_code": normalization_code_facts(),
        }
    feature_names = list(stats["feature_names"])
    condition_mean = np.asarray(stats["condition_mean"], dtype=np.float64).reshape(-1)
    condition_std = np.asarray(stats["condition_std"], dtype=np.float64).reshape(-1)
    return {
        "available": True,
        "confirmed_from_code": normalization_code_facts(),
        "feature_names": feature_names,
        "condition_mean_by_feature": dict(zip(feature_names, map(float, condition_mean), strict=True)),
        "condition_std_by_feature": dict(zip(feature_names, map(float, condition_std), strict=True)),
        "target_delta_mean": float(np.asarray(stats["target_delta_mean"]).reshape(-1)[0]),
        "target_delta_std": float(np.asarray(stats["target_delta_std"]).reshape(-1)[0]),
        "coord_min": [float(value) for value in np.asarray(stats["coord_min"]).reshape(-1)],
        "coord_span": [float(value) for value in np.asarray(stats["coord_span"]).reshape(-1)],
        "risk_notes": normalization_risk_notes(),
    }


def normalization_code_facts() -> list[str]:
    return [
        "x_inp/x_out are raw coordinates in meters, normalized to [-1,1] with train coord_min/coord_span before model.apply.",
        "u is the zero_delta bridge field and is not z-scored.",
        "c contains k, q, BC flags, and relative BC scalars; every c channel is train per-feature z-scored.",
        "BC flags are included in c and therefore are also z-scored as continuous features.",
        "target for loss is normalized DeltaT, not raw T; recovery adds T_ref after de-normalizing DeltaT.",
        "layer_id, region_id, and material_id remain loader metadata and are not packed into Inputs.",
    ]


def normalization_risk_notes() -> list[str]:
    return [
        "k and q use linear z-score only; no log-scale or physical-unit-aware transform is applied.",
        "BC flags become continuous z-scored channels.",
        "coordinate min-max normalization can hide physical extent/aspect-ratio changes unless diagnostics keep raw geometry ranges.",
        "run artifacts store condition/target stats but not coord_min/coord_span in train_only_normalization.",
        "final-probe amplitude needs scale-ratio diagnostics because normalized loss can hide raw DeltaT scale shifts.",
    ]


def build_feature_manifest(
    stats: dict[str, Any] | None,
    summaries: dict[str, dict[str, Any]],
    gaps: list[dict[str, Any]],
) -> dict[str, Any]:
    feature_names = []
    if stats is not None:
        feature_names = list(stats["feature_names"])
    else:
        for summary in summaries.values():
            if summary.get("feature_names"):
                feature_names = list(summary["feature_names"])
                break

    entries = [
        {
            "name": "x_inp",
            "tensor": "Inputs.x_inp",
            "source": "coords.npy",
            "enters_model": True,
            "normalization": "train coord min/span -> [-1,1]",
            "semantics": "input physical node coordinates, meters before normalization",
        },
        {
            "name": "x_out",
            "tensor": "Inputs.x_out",
            "source": "coords.npy",
            "enters_model": True,
            "normalization": "same as x_inp",
            "semantics": "output physical node coordinates; same nodes in current steady path",
        },
        {
            "name": "u.zero_delta",
            "tensor": "Inputs.u",
            "source": "zero_delta_u_bridge",
            "enters_model": True,
            "normalization": "none",
            "semantics": "zero delta-temperature field; T_ref is retained outside u for recovery",
        },
    ]
    for name in feature_names:
        entries.append(
            {
                "name": name,
                "tensor": "Inputs.c",
                "source": feature_source(name),
                "enters_model": True,
                "normalization": "train per-feature z-score",
                "semantics": feature_semantics(name),
            }
        )
    entries.extend(
        [
            {
                "name": "target.normalized_deltaT",
                "tensor": "target",
                "source": "temperature.npy and T_ref from boundary metadata",
                "enters_model": False,
                "normalization": "train scalar DeltaT mean/std",
                "semantics": "loss target; raw T is recovered as T_ref + DeltaT",
            },
            {
                "name": "layer_id/region_id/material_id",
                "tensor": "metadata only",
                "source": "layer_id.npy, region_id.npy, material_id.npy",
                "enters_model": False,
                "normalization": "none",
                "semantics": "dataset metadata for generation/evaluation grouping, not current model input",
            },
        ]
    )
    return {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "entries": entries,
        "split_summaries_available": sorted(
            split for split, summary in summaries.items() if summary.get("available")
        ),
        "gaps": gaps,
    }


def feature_source(name: str) -> str:
    if name.startswith("k_"):
        return "k_field.npy encoded with diag3"
    if name == "q":
        return "q_field.npy"
    if name in BC_FLAG_NAMES:
        return "boundary_regions or coordinate boundary-mask fallback"
    if name in BC_SCALAR_NAMES:
        return "boundary_params converted to relative BC feature view"
    return "condition feature"


def feature_semantics(name: str) -> str:
    if name.startswith("k_"):
        return "thermal conductivity channel"
    if name == "q":
        return "volumetric heat generation"
    if name in BC_FLAG_NAMES:
        return "boundary-condition flag"
    if name == "top_h":
        return "top Robin convection coefficient"
    if name == "top_T_inf_minus_T_ref":
        return "top ambient temperature relative to T_ref"
    if name == "bottom_T_fixed_minus_T_ref":
        return "bottom fixed temperature relative to T_ref"
    return "condition feature"


def data_scope_note(subset: Path, config: dict[str, Any]) -> str:
    resolved = str(_repo_path(config["dataset"]["subset_path"]))
    actual = str(subset)
    if actual == resolved:
        return "registry-resolved V4 subset"
    return (
        "local proxy subset for audit only; active V4 training still resolves to "
        f"{resolved}"
    )


def artifact_record_gaps() -> list[dict[str, str]]:
    return [
        {
            "artifact": "run_config.json",
            "gap": "route is recorded as prose but target_mode, bridge_policy, feature_view, and normalization_profile are not structured fields.",
            "suggested_fields": "target_mode, bridge_policy, feature_view, normalization_profile",
        },
        {
            "artifact": "run_config.json / loss_summary.json / checkpoints",
            "gap": "train_only_normalization stores feature_names and c/target stats but omits coord_min and coord_span.",
            "suggested_fields": "coord_min, coord_span, coord_normalization_scope",
        },
        {
            "artifact": "loss_summary.json",
            "gap": "no input feature manifest hash or sample manifest hash ties results to channel semantics.",
            "suggested_fields": "input_manifest_hash, feature_manifest_hash, target_manifest_hash",
        },
        {
            "artifact": "configs/heat3d_v4/run_registry.csv",
            "gap": "result registry has metrics fields but not target/normalization/bridge provenance fields.",
            "suggested_fields": "result_target_mode, result_normalization_profile, result_bridge_policy",
        },
    ]


def provenance_field_plan() -> list[dict[str, str]]:
    return [
        {
            "field": "result_target_mode",
            "role": "result provenance",
            "reason": "Record whether metrics came from raw T, raw DeltaT, or normalized DeltaT.",
        },
        {
            "field": "result_bridge_policy",
            "role": "result provenance",
            "reason": "Record zero_delta_u_bridge versus any future T_ref or native input bridge.",
        },
        {
            "field": "result_normalization_profile",
            "role": "result provenance",
            "reason": "Record train-only c/target/coord normalization policy used by the run.",
        },
        {
            "field": "result_feature_manifest_hash",
            "role": "result provenance",
            "reason": "Tie result rows to the model-facing feature/channel manifest.",
        },
        {
            "field": "result_dataset_split_hash",
            "role": "result provenance",
            "reason": "Tie result rows to the exact train/valid split IDs.",
        },
        {
            "field": "result_final_probe_scale_ratio",
            "role": "amplitude diagnostic",
            "reason": "Differentiate low-amplitude predictions from shape mismatch.",
        },
        {
            "field": "result_final_probe_range_ratio",
            "role": "amplitude diagnostic",
            "reason": "Track predicted field dynamic range against label range.",
        },
        {
            "field": "result_final_probe_centered_corr",
            "role": "shape diagnostic",
            "reason": "Track field shape agreement independent of mean and amplitude.",
        },
        {
            "field": "result_final_probe_mean_bias_K",
            "role": "amplitude diagnostic",
            "reason": "Record signed average temperature bias.",
        },
        {
            "field": "result_final_probe_peak_error_K",
            "role": "amplitude diagnostic",
            "reason": "Record signed peak-temperature under/overprediction.",
        },
    ]


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(json_safe(payload), file, ensure_ascii=False, indent=2, sort_keys=True)
        file.write("\n")


def write_feature_csv(path: Path, entries: list[dict[str, Any]]) -> None:
    fieldnames = ["name", "tensor", "source", "enters_model", "normalization", "semantics"]
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for entry in entries:
            writer.writerow({field: entry.get(field, "") for field in fieldnames})


def write_amplitude_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "sample_id",
        "probe_id",
        "RMSE_K",
        "MAE_K",
        "relRMSE_DeltaT",
        "peak_error_K",
        "abs_peak_error_K",
        "mean_bias_K",
        "pred_peak_K",
        "label_peak_K",
        "pred_range_K",
        "label_range_K",
        "pred_deltaT_peak_K",
        "label_deltaT_peak_K",
        "pred_deltaT_range_K",
        "label_deltaT_range_K",
        "scale_ratio",
        "range_ratio",
        "centered_corr",
        "shape_corr",
    ]
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    return value


if __name__ == "__main__":
    raise SystemExit(main())
