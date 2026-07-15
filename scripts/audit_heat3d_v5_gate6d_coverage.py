#!/usr/bin/env python3
"""Train-fit/valid-query coverage audit in frozen 24D Global Context space."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from rigno.heat3d_v5_global_context import (  # noqa: E402
    GLOBAL_CONTEXT_FEATURES,
    context_vector,
    fit_train_only_standardizer,
)
from run_heat3d_v1_medium_controlled_training_export import (  # noqa: E402
    _load_params_checkpoint,
    _resolve_training_splits,
    _sample_root,
)
from run_heat3d_v3_final_probe_checkpoint_smoke import (  # noqa: E402
    install_checkpoint_feature_hooks,
    load_training_examples,
)
from run_heat3d_v5_clean_first import _load_examples, _physics_cache  # noqa: E402


DEFAULT_RUN = ROOT / "output/heat3d_v5_gate6c_runs/V4P5_12_gate6c_scratch_l2_shape_balanced"
DEFAULT_N3 = ROOT / "configs/heat3d_v5/gate6d/V4P5_07_frozen_gate5_valid_only_evaluation.json"
DEFAULT_L2 = ROOT / "configs/heat3d_v5/gate6d/V4P5_12_frozen_gate5_evaluation.json"
DEFAULT_JSON = ROOT / "configs/heat3d_v5/gate6d/global_context_coverage.json"
DEFAULT_MD = ROOT / "docs/v5_gate6d_global_context_coverage.md"


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--n3-evaluation", type=Path, default=DEFAULT_N3)
    parser.add_argument("--l2-evaluation", type=Path, default=DEFAULT_L2)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--output-md", type=Path, default=DEFAULT_MD)
    return parser.parse_args()


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _ids_hash(ids: Sequence[str]) -> str:
    return hashlib.sha256("\n".join(ids).encode("utf-8")).hexdigest()


def _rank(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=np.float64)
    start = 0
    while start < values.size:
        end = start + 1
        while end < values.size and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1) + 1.0
        start = end
    return ranks


def _correlation(left: np.ndarray, right: np.ndarray) -> dict[str, float]:
    if left.size != right.size or left.size < 2:
        raise ValueError("correlation arrays must align")
    pearson = float(np.corrcoef(left, right)[0, 1])
    spearman = float(np.corrcoef(_rank(left), _rank(right))[0, 1])
    if not math.isfinite(pearson) or not math.isfinite(spearman):
        raise ValueError("coverage correlation is non-finite")
    return {"pearson": pearson, "spearman": spearman}


def _per_sample(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    rows = payload["reports"]["best"]["valid_iid"]["per_sample"]
    result = {str(row["sample_id"]): row for row in rows}
    if len(result) != 128:
        raise ValueError("evaluation payload must contain 128 valid samples")
    return result


def main() -> int:
    args = _args()
    run_dir = args.run_dir.resolve()
    run_config = _read(run_dir / "run_config.json")
    checkpoint = _load_params_checkpoint(run_dir / "params_best.pkl")
    checkpoint_stats = dict(checkpoint.get("train_only_normalization") or {})
    if not checkpoint_stats:
        raise ValueError("checkpoint lacks train-only normalization metadata")
    install_checkpoint_feature_hooks(checkpoint_stats)
    train_examples = load_training_examples(run_config, checkpoint_stats)
    sample_root = _sample_root(Path(str(run_config["subset"])))
    split_map = Path(str(run_config["split_map_path"]))
    split_ids, split_source, _, _ = _resolve_training_splits(sample_root, split_map)
    train_ids = list(split_ids["train"])
    valid_ids = list(split_ids["valid_iid"])
    if len(train_ids) != 672 or len(valid_ids) != 128:
        raise ValueError("coverage contract requires train=672 and valid_iid=128")
    if [str(example.sample_id) for example in train_examples] != train_ids:
        raise ValueError("training examples differ from frozen train split")
    valid_examples = _load_examples(
        sample_root=sample_root,
        sample_ids=valid_ids,
        checkpoint_stats=checkpoint_stats,
        boundary_mask_fallback=bool(run_config.get("boundary_mask_fallback", True)),
    )
    cache = _physics_cache(list(train_examples) + list(valid_examples))
    train_contexts = [cache[sample_id]["context"] for sample_id in train_ids]
    valid_contexts = [cache[sample_id]["context"] for sample_id in valid_ids]
    standardizer = fit_train_only_standardizer(train_contexts, fit_sample_ids=train_ids)
    mean = np.asarray(standardizer["mean"], dtype=np.float64)
    std = np.asarray(standardizer["std"], dtype=np.float64)
    train_matrix = (np.vstack([context_vector(row) for row in train_contexts]) - mean) / std
    valid_matrix = (np.vstack([context_vector(row) for row in valid_contexts]) - mean) / std
    distances = np.sqrt(np.sum(np.square(valid_matrix[:, None, :] - train_matrix[None, :, :]), axis=2))
    nearest_index = np.argmin(distances, axis=1)
    nearest_distance = distances[np.arange(len(valid_ids)), nearest_index]

    n3 = _per_sample(_read(args.n3_evaluation))
    l2 = _per_sample(_read(args.l2_evaluation))
    if set(valid_ids) != set(n3) or set(valid_ids) != set(l2):
        raise ValueError("coverage and evaluator valid sample IDs differ")
    rows = []
    for index, sample_id in enumerate(valid_ids):
        n3_error = 100.0 * float(n3[sample_id]["sample_cv_relative_rmse"])
        l2_error = 100.0 * float(l2[sample_id]["sample_cv_relative_rmse"])
        rows.append({
            "sample_id": sample_id,
            "nearest_train_sample_id": train_ids[int(nearest_index[index])],
            "nearest_neighbor_distance": float(nearest_distance[index]),
            "n3_sample_relative_rmse_pct": n3_error,
            "l2_sample_relative_rmse_pct": l2_error,
            "l2_minus_n3_sample_relative_rmse_pct": l2_error - n3_error,
            "n3_point_sse_K2": float(n3[sample_id]["point_error_squared_sum"]),
            "l2_point_sse_K2": float(l2[sample_id]["point_error_squared_sum"]),
            "l2_minus_n3_point_sse_K2": float(l2[sample_id]["point_error_squared_sum"] - n3[sample_id]["point_error_squared_sum"]),
        })
    distance_array = np.asarray([row["nearest_neighbor_distance"] for row in rows])
    correlation_targets = {
        "n3_sample_relative_rmse_pct": np.asarray([row["n3_sample_relative_rmse_pct"] for row in rows]),
        "l2_sample_relative_rmse_pct": np.asarray([row["l2_sample_relative_rmse_pct"] for row in rows]),
        "l2_minus_n3_sample_relative_rmse_pct": np.asarray([row["l2_minus_n3_sample_relative_rmse_pct"] for row in rows]),
        "n3_point_sse_K2": np.asarray([row["n3_point_sse_K2"] for row in rows]),
        "l2_point_sse_K2": np.asarray([row["l2_point_sse_K2"] for row in rows]),
        "l2_minus_n3_point_sse_K2": np.asarray([row["l2_minus_n3_point_sse_K2"] for row in rows]),
    }
    correlations = {name: _correlation(distance_array, values) for name, values in correlation_targets.items()}
    edges = np.quantile(distance_array, [0.0, 0.25, 0.5, 0.75, 1.0])
    bins = np.searchsorted(edges[1:-1], distance_array, side="right")
    quartiles = []
    for index in range(4):
        selected = [row for row, bin_index in zip(rows, bins, strict=True) if bin_index == index]
        quartiles.append({
            "quartile": f"Q{index + 1}",
            "sample_count": len(selected),
            "distance_mean": float(np.mean([row["nearest_neighbor_distance"] for row in selected])),
            "n3_sample_relative_rmse_pct": float(np.mean([row["n3_sample_relative_rmse_pct"] for row in selected])),
            "l2_sample_relative_rmse_pct": float(np.mean([row["l2_sample_relative_rmse_pct"] for row in selected])),
            "l2_minus_n3_sample_relative_rmse_pct": float(np.mean([row["l2_minus_n3_sample_relative_rmse_pct"] for row in selected])),
        })
    distance_rank = _rank(distance_array) / len(rows)
    max_error = np.maximum(correlation_targets["n3_sample_relative_rmse_pct"], correlation_targets["l2_sample_relative_rmse_pct"])
    error_rank = _rank(max_error) / len(rows)
    joint_rank = distance_rank + error_rank
    worst = [rows[index] for index in np.argsort(joint_rank)[::-1][:10]]

    payload = {
        "schema_version": "heat3d_v5_gate6d_global_context_coverage_v1",
        "fit_roles": ["train"],
        "query_roles": ["valid_iid"],
        "forbidden_roles_accessed": [],
        "feature_schema": list(GLOBAL_CONTEXT_FEATURES),
        "feature_count": len(GLOBAL_CONTEXT_FEATURES),
        "target_or_label_features": [],
        "distance": "Euclidean distance in train-standardized 24D Global Context",
        "split_source": split_source,
        "train_sample_count": len(train_ids),
        "valid_sample_count": len(valid_ids),
        "train_ids_sha256": _ids_hash(train_ids),
        "valid_ids_sha256": _ids_hash(valid_ids),
        "standardizer": standardizer,
        "distance_summary": {
            "min": float(distance_array.min()),
            "mean": float(distance_array.mean()),
            "median": float(np.median(distance_array)),
            "max": float(distance_array.max()),
            "quartile_edges": edges.tolist(),
        },
        "correlations": correlations,
        "coverage_quartiles": quartiles,
        "worst_coverage_and_high_error": worst,
        "per_sample": rows,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    md = [
        "# Gate 6D Global Context coverage audit",
        "",
        "距离空间仅由 train=672 拟合的 24 维 Global Context 标准化特征构成；valid_iid=128 只作查询。没有 target-derived distance feature，未访问 test/hard。",
        "",
        "| error target | Pearson(distance,error) | Spearman(distance,error) |",
        "|---|---:|---:|",
    ]
    for name, values in correlations.items():
        md.append(f"| {name} | {values['pearson']:.6f} | {values['spearman']:.6f} |")
    md.extend([
        "",
        f"distance min/median/mean/max = {distance_array.min():.6f} / {np.median(distance_array):.6f} / {distance_array.mean():.6f} / {distance_array.max():.6f}。",
        "",
        "coverage distance 四分位、逐样本 nearest train ID，以及覆盖最差且误差最高的样本保存在 JSON。",
        "",
    ])
    args.output_md.write_text("\n".join(md), encoding="utf-8")
    print(json.dumps({"status": "passed", "distance_summary": payload["distance_summary"], "correlations": correlations}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
