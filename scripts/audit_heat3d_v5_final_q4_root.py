#!/usr/bin/env python3
"""Read-only Q4 root-cause audit for the frozen V5 point-global candidates."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from rigno.heat3d_v5_global_context import (  # noqa: E402
    GLOBAL_CONTEXT_FEATURES,
    context_vector,
)
from evaluate_heat3d_v5_gate6q_closeout import (  # noqa: E402
    _load_examples_for_ids,
    _resolve_training_splits,
    _sample_root,
)
from run_heat3d_v5_clean_first import _raw_context_and_physics  # noqa: E402


MODELS = ("V38", "V42", "V44", "V45", "V46")
EPS = 1.0e-12


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v38", type=Path, required=True)
    parser.add_argument("--gate6q", type=Path, required=True)
    parser.add_argument("--v45", type=Path, required=True)
    parser.add_argument("--v46", type=Path, required=True)
    parser.add_argument("--reference-run-dir", type=Path, required=True)
    parser.add_argument("--json", type=Path, required=True)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--markdown", type=Path, required=True)
    return parser.parse_args()


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _ids_hash(ids: Sequence[str]) -> str:
    return hashlib.sha256("\n".join(ids).encode("utf-8")).hexdigest()


def _rank(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=np.float64)
    ranks[order] = np.arange(values.size, dtype=np.float64)
    unique, inverse, counts = np.unique(values, return_inverse=True, return_counts=True)
    del unique
    for index, count in enumerate(counts):
        if count > 1:
            ranks[inverse == index] = float(np.mean(ranks[inverse == index]))
    return ranks


def _corr(left: Sequence[float], right: Sequence[float]) -> dict[str, float]:
    x = np.asarray(left, dtype=np.float64)
    y = np.asarray(right, dtype=np.float64)
    if x.size != y.size or x.size < 2 or np.std(x) <= EPS or np.std(y) <= EPS:
        return {"pearson": 0.0, "spearman": 0.0}
    return {
        "pearson": float(np.corrcoef(x, y)[0, 1]),
        "spearman": float(np.corrcoef(_rank(x), _rank(y))[0, 1]),
    }


def _model_payloads(args: argparse.Namespace) -> dict[str, dict[str, Any]]:
    gate6q = _read(args.gate6q)
    models = {
        "V38": _read(args.v38),
        "V42": gate6q["models"]["V42"],
        "V44": gate6q["models"]["V44"],
        "V45": _read(args.v45),
        "V46": _read(args.v46),
    }
    split_hashes = set()
    metric_hashes = set()
    for label, payload in models.items():
        if payload.get("status") != "completed_valid_iid_only":
            raise ValueError(f"{label}: incomplete evaluator")
        scope = payload.get("scope", {})
        if scope.get("evaluation_roles") != ["valid_iid"] or scope.get("forbidden_roles_accessed"):
            raise ValueError(f"{label}: invalid role scope")
        split_hashes.add(payload["split"]["valid_iid_ids_sha256"])
        metric_hashes.add(payload["metric_source"]["sha256"])
    if len(split_hashes) != 1 or len(metric_hashes) != 1:
        raise ValueError("model metric/split binding differs")
    return models


def _sample_maps(models: Mapping[str, Mapping[str, Any]]) -> dict[str, dict[str, dict[str, Any]]]:
    result = {}
    for label, payload in models.items():
        rows = payload["metrics"]["point_global_best"]["per_sample"]
        result[label] = {str(row["sample_id"]): dict(row) for row in rows}
        if len(result[label]) != 128:
            raise ValueError(f"{label}: expected 128 samples")
    ids = {tuple(sorted(rows)) for rows in result.values()}
    if len(ids) != 1:
        raise ValueError("paired sample IDs differ")
    return result


def _physical_features(example: Any) -> tuple[np.ndarray, dict[str, float | int | str]]:
    raw = _raw_context_and_physics(example)
    context = raw["context"]
    relative = example.get_relative_bc_feature_view()
    names = tuple(relative.condition_feature_names)
    values = np.asarray(relative.condition_features, dtype=np.float64)
    coords = np.asarray(example.condition.coords, dtype=np.float64)
    volumes = np.asarray(raw["control_volumes"], dtype=np.float64)
    q = values[:, names.index("q")]
    threshold = max(EPS, float(np.max(np.abs(q))) * 1.0e-12)
    present = q > threshold
    z_values, z_inverse = np.unique(coords[:, 2], return_inverse=True)
    active_layers = np.unique(z_inverse[present]) if np.any(present) else np.asarray([], dtype=int)
    kx = values[:, names.index("k_x")]
    ky = values[:, names.index("k_y")]
    kz = values[:, names.index("k_z")]
    layer_k = np.asarray(
        [
            [np.mean(kx[z_inverse == index]), np.mean(ky[z_inverse == index]), np.mean(kz[z_inverse == index])]
            for index in range(z_values.size)
        ],
        dtype=np.float64,
    )
    rounded = np.round(np.log10(np.maximum(layer_k, EPS)), decimals=8)
    unique_stack_materials = int(np.unique(rounded, axis=0).shape[0])
    transitions = int(np.sum(np.any(np.abs(np.diff(rounded, axis=0)) > 1.0e-10, axis=1)))
    source_fraction = float(np.sum(volumes[present]) / np.sum(volumes))
    top_h = float(np.median(values[:, names.index("top_h")]))
    category = (
        f"h{int(round(math.log10(max(top_h, EPS))))}_"
        f"a{'high' if context['anisotropy_xy_over_z'] >= 1.0 else 'low'}_"
        f"z{int(active_layers.size)}"
    )
    row: dict[str, float | int | str] = {
        "sample_id": str(example.sample_id),
        "source_present_volume_fraction": source_fraction,
        "source_active_z_layers": int(active_layers.size),
        "source_active_z_layer_fraction": float(active_layers.size / max(z_values.size, 1)),
        "stack_z_layer_count": int(z_values.size),
        "stack_unique_conductivity_triplets": unique_stack_materials,
        "stack_conductivity_transition_fraction": float(transitions / max(z_values.size - 1, 1)),
        "top_h_W_m2K": top_h,
        "condition_category": category,
    }
    for name in GLOBAL_CONTEXT_FEATURES:
        row[name] = float(context[name])
    return context_vector(context), row


def _decomposition(rows: Mapping[str, Mapping[str, Any]], ids: Sequence[str]) -> dict[str, float]:
    return {
        field: float(sum(float(rows[sample_id][field]) for sample_id in ids))
        for field in (
            "point_error_squared_sum",
            "shape_point_sse_K2",
            "scale_point_sse_K2",
            "cross_point_sse_K2",
        )
    }


def main() -> int:
    args = _args()
    models = _model_payloads(args)
    samples = _sample_maps(models)
    run_config = _read(args.reference_run_dir / "run_config.json")
    sample_root = _sample_root(Path(str(run_config["subset"])))
    split_ids, split_source, _primary, _stress = _resolve_training_splits(
        sample_root, Path(str(run_config["split_map_path"]))
    )
    train_ids = list(split_ids.get("train") or ())
    valid_ids = list(split_ids.get("valid_iid") or ())
    if len(train_ids) != 672 or len(valid_ids) != 128 or set(train_ids).intersection(valid_ids):
        raise ValueError("unexpected train/valid split")
    if _ids_hash(valid_ids) != models["V38"]["split"]["valid_iid_ids_sha256"]:
        raise ValueError("valid split hash mismatch")
    boundary_fallback = bool(run_config.get("boundary_mask_fallback", True))
    train_examples = _load_examples_for_ids(
        sample_root, train_ids, role="train", boundary_mask_fallback=boundary_fallback
    )
    valid_examples = _load_examples_for_ids(
        sample_root, valid_ids, role="valid_iid", boundary_mask_fallback=boundary_fallback
    )
    train_vectors = []
    for example in train_examples:
        vector, _row = _physical_features(example)
        train_vectors.append(vector)
    valid_vectors = []
    physical: dict[str, dict[str, Any]] = {}
    for example in valid_examples:
        vector, row = _physical_features(example)
        valid_vectors.append(vector)
        physical[str(example.sample_id)] = row
    train_matrix = np.asarray(train_vectors, dtype=np.float64)
    valid_matrix = np.asarray(valid_vectors, dtype=np.float64)
    mean = np.mean(train_matrix, axis=0)
    std = np.std(train_matrix, axis=0)
    std = np.where(std > EPS, std, 1.0)
    train_z = (train_matrix - mean) / std
    valid_z = (valid_matrix - mean) / std
    squared = np.sum(np.square(valid_z[:, None, :] - train_z[None, :, :]), axis=2)
    nearest_index = np.argmin(squared, axis=1)
    nearest_distance = np.sqrt(squared[np.arange(valid_z.shape[0]), nearest_index])
    for index, sample_id in enumerate(valid_ids):
        physical[sample_id]["train_nn_distance_24d"] = float(nearest_distance[index])
        physical[sample_id]["train_nn_sample_id"] = train_ids[int(nearest_index[index])]
        physical[sample_id]["deltaT_quartile"] = samples["V38"][sample_id]["deltaT_quartile"]
        physical[sample_id]["true_scale_cv_rms_K"] = float(samples["V38"][sample_id]["true_scale_cv_rms_K"])

    top_sets: dict[str, dict[str, list[str]]] = {}
    decomposition: dict[str, Any] = {}
    for label in MODELS:
        ordered = sorted(valid_ids, key=lambda sid: float(samples[label][sid]["point_error_squared_sum"]), reverse=True)
        top_sets[label] = {"top5": ordered[:5], "top10": ordered[:10]}
        q4_ids = [sid for sid in valid_ids if samples[label][sid]["deltaT_quartile"] == "Q4"]
        decomposition[label] = {
            "all": _decomposition(samples[label], valid_ids),
            "Q4": _decomposition(samples[label], q4_ids),
        }
    overlap: dict[str, Any] = {}
    for left_index, left in enumerate(MODELS):
        for right in MODELS[left_index + 1 :]:
            key = f"{left}_{right}"
            overlap[key] = {}
            for count in ("top5", "top10"):
                a, b = set(top_sets[left][count]), set(top_sets[right][count])
                overlap[key][count] = {
                    "intersection_count": len(a & b),
                    "jaccard": float(len(a & b) / max(len(a | b), 1)),
                }

    feature_names = (
        "train_nn_distance_24d",
        "source_present_volume_fraction",
        "source_active_z_layer_fraction",
        "source_concentration",
        "q_weighted_inverse_kz_mK_W",
        "q_low_k_overlap_fraction",
        "anisotropy_xy_over_z",
        "top_h_W_m2K",
        "stack_unique_conductivity_triplets",
        "stack_conductivity_transition_fraction",
    )
    correlations: dict[str, Any] = {}
    coverage: dict[str, Any] = {
        "train_fit_count": len(train_ids),
        "valid_query_count": len(valid_ids),
        "feature_count": len(GLOBAL_CONTEXT_FEATURES),
        "distance_quantiles": {
            key: float(value)
            for key, value in zip(("min", "q25", "median", "q75", "max"), np.quantile(nearest_distance, [0, 0.25, 0.5, 0.75, 1]))
        },
    }
    for label in MODELS:
        errors = [float(samples[label][sid]["point_error_squared_sum"]) for sid in valid_ids]
        correlations[label] = {
            feature: _corr([float(physical[sid][feature]) for sid in valid_ids], errors)
            for feature in feature_names
        }
        top10 = set(top_sets[label]["top10"])
        coverage[label] = {
            "top10_mean_nn_distance": float(np.mean([physical[sid]["train_nn_distance_24d"] for sid in top10])),
            "rest_mean_nn_distance": float(np.mean([physical[sid]["train_nn_distance_24d"] for sid in valid_ids if sid not in top10])),
            "Q4_mean_nn_distance": float(np.mean([physical[sid]["train_nn_distance_24d"] for sid in valid_ids if physical[sid]["deltaT_quartile"] == "Q4"])),
        }

    csv_rows: list[dict[str, Any]] = []
    for sample_id in valid_ids:
        row = dict(physical[sample_id])
        for label in MODELS:
            for source, target in (
                ("point_error_squared_sum", "point_sse_K2"),
                ("shape_point_sse_K2", "shape_sse_K2"),
                ("scale_point_sse_K2", "scale_sse_K2"),
                ("cross_point_sse_K2", "cross_sse_K2"),
                ("sample_cv_relative_rmse", "sample_relative_rmse"),
            ):
                row[f"{label}_{target}"] = float(samples[label][sample_id][source])
        csv_rows.append(row)

    payload = {
        "schema_version": "heat3d_v5_final_q4_root_audit_v1",
        "status": "completed_train_valid_read_only",
        "scope": {
            "roles_accessed": ["train", "valid_iid"],
            "forbidden_roles_accessed": [],
            "test_accessed": False,
            "hard_accessed": False,
            "sealed_iid_accessed": False,
            "training_started": False,
            "target_used_for_coverage_features": False,
        },
        "split": {
            "source": split_source,
            "train_count": len(train_ids),
            "valid_iid_count": len(valid_ids),
            "train_ids_sha256": _ids_hash(train_ids),
            "valid_iid_ids_sha256": _ids_hash(valid_ids),
        },
        "inputs": {
            "models": {label: models[label]["config_id"] for label in MODELS},
            "metric_source_sha256": models["V38"]["metric_source"]["sha256"],
            "evaluator_artifacts": {
                str(path): {"sha256": _sha256(path), "bytes": path.stat().st_size}
                for path in (args.v38, args.gate6q, args.v45, args.v46)
            },
        },
        "difficult_sample_overlap": overlap,
        "top_sample_ids": top_sets,
        "shape_scale_cross_decomposition": decomposition,
        "coverage": coverage,
        "physical_feature_error_correlations": correlations,
        "sample_csv": str(args.csv),
    }
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.csv.parent.mkdir(parents=True, exist_ok=True)
    args.markdown.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    with args.csv.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(csv_rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(csv_rows)
    lines = [
        "# V5 final Q4 root-cause audit",
        "",
        "Read-only scope: train/valid_iid. Coverage uses only the frozen 24D input-derived context; no test/hard/sealed access.",
        "",
        "## Q4 decomposition",
        "",
        "| model | total SSE | Q4 SSE | Q4 shape | Q4 scale | Q4 cross |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for label in MODELS:
        all_row, q4 = decomposition[label]["all"], decomposition[label]["Q4"]
        lines.append(
            f"| {label} | {all_row['point_error_squared_sum']:.6f} | {q4['point_error_squared_sum']:.6f} | "
            f"{q4['shape_point_sse_K2']:.6f} | {q4['scale_point_sse_K2']:.6f} | {q4['cross_point_sse_K2']:.6f} |"
        )
    lines.extend(["", "Detailed overlap, coverage, correlations, and per-sample physical features are in the JSON/CSV."])
    args.markdown.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "json": str(args.json), "csv": str(args.csv)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
