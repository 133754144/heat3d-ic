#!/usr/bin/env python3
"""Read-only audit of Heat3D v2 input features and target construction."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rigno.heat3d_v1_native_supervised import Heat3DV1NativeSupervisedDataset


REQUIRED_RELATIVE_FEATURES = {
    "k_x",
    "k_y",
    "k_z",
    "q",
    "is_top",
    "is_bottom",
    "is_side",
    "is_interior",
    "top_h",
    "top_T_inf_minus_T_ref",
    "bottom_T_fixed_minus_T_ref",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subset", type=Path, required=True)
    parser.add_argument("--split-map", type=Path, required=True)
    parser.add_argument("--max-neighbor-samples", type=int, default=256)
    parser.add_argument("--output-json", type=Path, default=None)
    return parser.parse_args()


def _sample_root(path: Path) -> Path:
    samples = path / "samples"
    return samples if samples.is_dir() else path


def _split_map(path: Path) -> dict[str, str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    mapping = payload.get("sample_splits", payload)
    if not isinstance(mapping, dict):
        raise ValueError(f"{path}: expected mapping or sample_splits")
    return {str(sample_id): str(split) for sample_id, split in mapping.items()}


def _t_ref(example) -> tuple[float, str]:
    bridge = example.build_temperature_rise_legacy_inputs_from_relative_features(
        bridge_policy="zero_delta_u_bridge"
    )
    return float(bridge.t_ref_value), str(bridge.t_ref_source)


def _hash_arrays(*arrays: np.ndarray) -> str:
    digest = hashlib.sha256()
    for array in arrays:
        values = np.ascontiguousarray(array)
        digest.update(str(values.shape).encode("utf-8"))
        digest.update(str(values.dtype).encode("utf-8"))
        digest.update(values.tobytes())
    return digest.hexdigest()[:16]


def _safe_corr(x: list[float], y: list[float]) -> float | None:
    a = np.asarray(x, dtype=np.float64)
    b = np.asarray(y, dtype=np.float64)
    if a.size < 2 or np.std(a) < 1.0e-12 or np.std(b) < 1.0e-12:
        return None
    value = float(np.corrcoef(a, b)[0, 1])
    return value if np.isfinite(value) else None


def _split_stats(rows: list[dict[str, Any]], split: str) -> dict[str, Any]:
    selected = [row for row in rows if row["split"] == split]
    if not selected:
        return {"sample_count": 0}
    return {
        "sample_count": len(selected),
        "delta_mean_mean": float(np.mean([row["delta_mean"] for row in selected])),
        "delta_max_mean": float(np.mean([row["delta_max"] for row in selected])),
        "low_delta_fraction_mean": float(np.mean([row["low_delta_fraction"] for row in selected])),
        "q_mean_mean": float(np.mean([row["q_mean"] for row in selected])),
        "top_h_mean": float(np.mean([row["top_h"] for row in selected])),
    }


def _nearest_pairs(rows: list[dict[str, Any]], max_samples: int) -> list[dict[str, Any]]:
    selected = sorted(rows, key=lambda row: row["sample_id"])[: max(0, max_samples)]
    if len(selected) < 2:
        return []
    vectors = np.asarray([row["summary_vector"] for row in selected], dtype=np.float64)
    mean = np.mean(vectors, axis=0, keepdims=True)
    std = np.std(vectors, axis=0, keepdims=True)
    normalized = (vectors - mean) / np.where(std < 1.0e-12, 1.0, std)
    distances = np.linalg.norm(normalized[:, None, :] - normalized[None, :, :], axis=-1)
    np.fill_diagonal(distances, np.inf)
    pairs: dict[tuple[int, int], float] = {}
    for index in range(len(selected)):
        neighbor = int(np.argmin(distances[index]))
        pair = tuple(sorted((index, neighbor)))
        pairs[pair] = float(distances[index, neighbor])
    result = []
    for (left, right), distance in sorted(pairs.items(), key=lambda item: item[1])[:10]:
        a = selected[left]
        b = selected[right]
        result.append(
            {
                "sample_a": a["sample_id"],
                "sample_b": b["sample_id"],
                "summary_input_distance": distance,
                "delta_mean_abs_difference": abs(a["delta_mean"] - b["delta_mean"]),
                "delta_max_abs_difference": abs(a["delta_max"] - b["delta_max"]),
            }
        )
    return result


def audit(subset: Path, split_map_path: Path, max_neighbor_samples: int) -> dict[str, Any]:
    mapping = _split_map(split_map_path)
    dataset = Heat3DV1NativeSupervisedDataset(_sample_root(subset), k_encoding_mode="diag3")
    rows: list[dict[str, Any]] = []
    feature_name_sets: set[tuple[str, ...]] = set()
    input_to_targets: dict[str, set[str]] = {}
    max_target_construction_error = 0.0
    max_recovery_error = 0.0
    max_bc_encoding_error = 0.0
    max_zero_bridge_abs = 0.0
    boundary_type_counts: dict[str, int] = {}
    t_ref_sources: dict[str, int] = {}

    for example in dataset.samples:
        sample_id = example.sample_id
        if sample_id not in mapping:
            continue
        split = mapping[sample_id]
        bridge = example.build_temperature_rise_legacy_inputs_from_relative_features(
            bridge_policy="zero_delta_u_bridge"
        )
        feature_names = tuple(bridge.condition_feature_names)
        feature_name_sets.add(feature_names)
        feature_index = {name: index for index, name in enumerate(feature_names)}
        condition = np.asarray(bridge.legacy_inputs.c, dtype=np.float64).reshape(-1, len(feature_names))
        coords = np.asarray(example.condition.coords, dtype=np.float64)
        target_temperature = np.asarray(example.target.target_u, dtype=np.float64).reshape(-1)
        target_delta = np.asarray(bridge.target_delta_u, dtype=np.float64).reshape(-1)
        t_ref = float(bridge.t_ref_value)
        expected_delta = target_temperature - t_ref
        max_target_construction_error = max(
            max_target_construction_error, float(np.max(np.abs(target_delta - expected_delta)))
        )
        max_recovery_error = max(
            max_recovery_error, float(np.max(np.abs((t_ref + target_delta) - target_temperature)))
        )
        max_zero_bridge_abs = max(
            max_zero_bridge_abs, float(np.max(np.abs(np.asarray(bridge.legacy_inputs.u))))
        )

        params = example.meta.get("boundary_params", {})
        top = params.get("top", {}) if isinstance(params, dict) else {}
        bottom = params.get("bottom", {}) if isinstance(params, dict) else {}
        expected_bc = {
            "top_h": float(top.get("h_W_m2K", 0.0)),
            "top_T_inf_minus_T_ref": float(top.get("ambient_temperature_K", 0.0)) - t_ref,
            "bottom_T_fixed_minus_T_ref": float(bottom.get("fixed_temperature_K", 0.0)) - t_ref,
        }
        for name, expected in expected_bc.items():
            if name in feature_index:
                max_bc_encoding_error = max(
                    max_bc_encoding_error,
                    float(np.max(np.abs(condition[:, feature_index[name]] - expected))),
                )

        boundary_types = example.meta.get("boundary_types", {})
        key = json.dumps(boundary_types, sort_keys=True)
        boundary_type_counts[key] = boundary_type_counts.get(key, 0) + 1
        t_ref_sources[bridge.t_ref_source] = t_ref_sources.get(bridge.t_ref_source, 0) + 1

        means = np.mean(condition, axis=0)
        stds = np.std(condition, axis=0)
        q_values = condition[:, feature_index["q"]]
        k_values = condition[:, [feature_index["k_x"], feature_index["k_y"], feature_index["k_z"]]]
        summary_vector = np.concatenate(
            [
                means,
                stds,
                np.min(coords, axis=0),
                np.max(coords, axis=0),
            ]
        ).tolist()
        input_hash = _hash_arrays(coords, condition)
        target_hash = _hash_arrays(target_delta)
        input_to_targets.setdefault(input_hash, set()).add(target_hash)
        rows.append(
            {
                "sample_id": sample_id,
                "split": split,
                "input_hash": input_hash,
                "target_hash": target_hash,
                "q_mean": float(np.mean(q_values)),
                "q_max": float(np.max(q_values)),
                "q_nonzero_fraction": float(np.mean(np.abs(q_values) > 0.0)),
                "k_mean": float(np.mean(k_values)),
                "top_h": expected_bc["top_h"],
                "top_T_inf_minus_T_ref": expected_bc["top_T_inf_minus_T_ref"],
                "bottom_T_fixed_minus_T_ref": expected_bc["bottom_T_fixed_minus_T_ref"],
                "delta_mean": float(np.mean(target_delta)),
                "delta_max": float(np.max(target_delta)),
                "delta_std": float(np.std(target_delta)),
                "low_delta_fraction": float(np.mean(target_delta <= 0.01)),
                "summary_vector": summary_vector,
            }
        )

    train_rows = [row for row in rows if row["split"] == "train"]
    train_ids = {row["sample_id"] for row in train_rows}
    train_examples = [example for example in dataset.samples if example.sample_id in train_ids]
    train_condition = []
    train_delta = []
    for example in train_examples:
        bridge = example.build_temperature_rise_legacy_inputs_from_relative_features(
            bridge_policy="zero_delta_u_bridge"
        )
        train_condition.append(np.asarray(bridge.legacy_inputs.c, dtype=np.float64).reshape(-1, len(bridge.condition_feature_names)))
        train_delta.append(np.asarray(bridge.target_delta_u, dtype=np.float64).reshape(-1, 1))
    condition_all = np.concatenate(train_condition, axis=0)
    delta_all = np.concatenate(train_delta, axis=0)
    condition_mean = np.mean(condition_all, axis=0)
    condition_std = np.std(condition_all, axis=0)
    target_mean = float(np.mean(delta_all))
    target_std = float(np.std(delta_all))
    normalized = (delta_all - target_mean) / (target_std if target_std >= 1.0e-12 else 1.0)
    recovered = normalized * (target_std if target_std >= 1.0e-12 else 1.0) + target_mean
    normalization_recovery_error = float(np.max(np.abs(recovered - delta_all)))
    feature_names = list(next(iter(feature_name_sets))) if len(feature_name_sets) == 1 else []
    constant_features = [
        name for name, std in zip(feature_names, condition_std, strict=True) if float(std) < 1.0e-12
    ]
    duplicate_conflicts = [
        {"input_hash": input_hash, "target_hash_count": len(target_hashes)}
        for input_hash, target_hashes in input_to_targets.items()
        if len(target_hashes) > 1
    ]
    q_means = [row["q_mean"] for row in rows]
    k_means = [row["k_mean"] for row in rows]
    top_h_values = [row["top_h"] for row in rows]
    delta_means = [row["delta_mean"] for row in rows]
    delta_maxes = [row["delta_max"] for row in rows]
    split_source = (
        REPO_ROOT / "scripts" / "analyze_heat3d_v2_split_aware_diagnostics.py"
    ).read_text(encoding="utf-8")

    return {
        "diagnostic_scope": "Heat3D v2 input/target representation audit; not formal benchmark",
        "sample_count": len(rows),
        "feature_names": feature_names,
        "feature_shape_consistent": len(feature_name_sets) == 1,
        "required_relative_features_present": sorted(REQUIRED_RELATIVE_FEATURES & set(feature_names)),
        "missing_required_relative_features": sorted(REQUIRED_RELATIVE_FEATURES - set(feature_names)),
        "zero_delta_u_bridge_max_abs": max_zero_bridge_abs,
        "target_construction_max_abs_error": max_target_construction_error,
        "temperature_recovery_max_abs_error": max_recovery_error,
        "bc_encoding_max_abs_error": max_bc_encoding_error,
        "train_only_normalization": {
            "train_sample_count": len(train_rows),
            "target_delta_mean": target_mean,
            "target_delta_std": target_std,
            "normalization_recovery_max_abs_error": normalization_recovery_error,
            "constant_condition_features": constant_features,
        },
        "boundary_type_counts": boundary_type_counts,
        "t_ref_sources": t_ref_sources,
        "exact_input_duplicate_target_conflicts": duplicate_conflicts,
        "nearest_summary_input_pairs": _nearest_pairs(rows, max_neighbor_samples),
        "correlations": {
            "q_mean_vs_delta_mean": _safe_corr(q_means, delta_means),
            "q_mean_vs_delta_max": _safe_corr(q_means, delta_maxes),
            "k_mean_vs_delta_mean": _safe_corr(k_means, delta_means),
            "top_h_vs_delta_mean": _safe_corr(top_h_values, delta_means),
        },
        "split_stats": {
            split: _split_stats(rows, split)
            for split in ("train", "valid_iid", "valid_stress")
        },
        "representation_notes": {
            "coordinates": "passed separately as normalized x_inp/x_out",
            "k_q_bc": "k, q, boundary masks, top_h, relative top ambient, and relative bottom fixed temperature are in condition c",
            "zero_delta_u_bridge": "legacy u is identically zero; all physical forcing is routed through c and coordinates",
            "auxiliary_ids": "layer_id, region_id, and material_id are loaded as metadata but are not direct model inputs",
            "target_normalization": "single global train-only DeltaT mean/std; not per-sample normalization",
            "absolute_t_ref": "not model-visible in zero-delta/relative-BC view; recovery adds metadata-derived T_ref after prediction",
        },
        "metric_naming_audit": {
            "runner_raw_delta_mse": "actual mean squared raw DeltaT error",
            "split_aware_raw_deltaT_mse": (
                "misnamed: stores per-sample RMSE via _rmse(pred_delta - true_delta), then averages samples"
                if '"raw_deltaT_mse": _rmse(pred_delta - true_delta)' in split_source
                else "source pattern not found; inspect manually"
            ),
        },
    }


def main() -> int:
    args = parse_args()
    result = audit(args.subset, args.split_map, args.max_neighbor_samples)
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
