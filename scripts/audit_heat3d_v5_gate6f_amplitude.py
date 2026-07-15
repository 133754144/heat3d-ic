#!/usr/bin/env python3
"""Valid-only N3 e402 versus L2 e353 amplitude attribution for Gate 6F.

The two model metric rows are reused from the frozen Gate 6D paired evaluator
payload.  This script only loads raw valid_iid input fields to add the missing
q--low-k overlap covariate; it performs no checkpoint inference or training.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from check_heat3d_v4_registry import resolve_inherited_yaml  # noqa: E402
from scripts import run_heat3d_v4_controlled_training as v4_wrapper  # noqa: E402


FEATURES = (
    ("true_cv_rms_deltaT_K", "true CV-RMS DeltaT (K)"),
    ("q_low_k_overlap_fraction", "q-low-k overlap fraction"),
    ("q_weighted_inverse_conductivity_mK_W", "q-weighted inverse-kz (m K/W)"),
    ("total_power_W", "total power (W)"),
    ("source_concentration", "source concentration"),
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--paired-json",
        type=Path,
        default=ROOT / "configs/heat3d_v5/gate6d/n3_l2_valid_paired.json",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/heat3d_v5/generated/V4P5_07_native_pooled_latent_global_film.yaml",
    )
    parser.add_argument("--output-json", type=Path, required=True)
    return parser.parse_args()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _resolved(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: YAML root must be a mapping")
    return resolve_inherited_yaml(payload, path)


def _root_path(path_text: str) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else ROOT / path


def _q_low_k_overlap_by_valid_id(config: dict[str, Any]) -> dict[str, float]:
    dataset_config = config["dataset"]
    v4_wrapper._install_profile_hooks(
        str(dataset_config["normalization_profile"]),
        str(dataset_config["condition_feature_transform"]),
        str(dataset_config["input_feature_schema"]),
        str(dataset_config["coord_policy"]),
        str(dataset_config["extent_feature_policy"]),
    )
    runner = v4_wrapper.legacy_runner
    sample_root = _root_path(str(dataset_config["subset_path"]))
    split_map = _root_path(str(dataset_config["split_map_path"]))
    split_ids, _, primary_split, _ = runner._resolve_training_splits(sample_root, split_map)
    if primary_split != "valid_iid":
        raise ValueError(f"expected valid_iid primary split, got {primary_split!r}")
    valid_ids = list(split_ids["valid_iid"])
    if len(valid_ids) != 128:
        raise ValueError(f"expected 128 valid samples, got {len(valid_ids)}")
    dataset = runner.Heat3DV1NativeSupervisedDataset(
        sample_root,
        k_encoding_mode="diag3",
        boundary_mask_fallback=False,
    )
    index_by_id = dataset.sample_index_by_id()
    values = {}
    for sample_id in valid_ids:
        example = dataset[index_by_id[sample_id]]
        context = runner._global_context_row_for_example(example)
        values[str(sample_id)] = float(context["q_low_k_overlap_fraction"])
    return values


def _quartiles(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    edges = np.quantile(values, [0.0, 0.25, 0.5, 0.75, 1.0])
    # Deterministic tie policy: increasing rank order keeps each bin near-equal.
    order = np.argsort(values, kind="mergesort")
    labels = np.empty(len(values), dtype=np.int64)
    for bin_index, indices in enumerate(np.array_split(order, 4), start=1):
        labels[indices] = bin_index
    return edges, labels


def _model_bin_summary(rows: list[dict[str, Any]], prefix: str) -> dict[str, float]:
    amplitude = np.asarray([float(row[f"{prefix}_amplitude_ratio"]) for row in rows])
    signed_bias = np.asarray([float(row[f"{prefix}_scale_log_error"]) for row in rows])
    sample_rmse = np.asarray([float(row[f"{prefix}_sample_relative_rmse_pct"]) for row in rows])
    point_sse = np.asarray([float(row[f"{prefix}_point_global_sse_K2"]) for row in rows])
    return {
        "amplitude_ratio_mean": float(np.mean(amplitude)),
        "amplitude_ratio_median": float(np.median(amplitude)),
        "signed_log_scale_bias_mean": float(np.mean(signed_bias)),
        "sample_first_relative_rmse_pct": float(np.mean(sample_rmse)),
        "point_sse_K2": float(np.sum(point_sse)),
    }


def _feature_bins(rows: list[dict[str, Any]], feature: str) -> dict[str, Any]:
    values = np.asarray([float(row["features"][feature]) for row in rows])
    edges, labels = _quartiles(values)
    bins = []
    total_n3_sse = sum(float(row["n3_point_global_sse_K2"]) for row in rows)
    total_l2_sse = sum(float(row["l2_point_global_sse_K2"]) for row in rows)
    for label in range(1, 5):
        members = [row for row, assigned in zip(rows, labels) if int(assigned) == label]
        n3 = _model_bin_summary(members, "n3")
        l2 = _model_bin_summary(members, "l2")
        n3["point_sse_share"] = n3["point_sse_K2"] / max(total_n3_sse, 1.0e-12)
        l2["point_sse_share"] = l2["point_sse_K2"] / max(total_l2_sse, 1.0e-12)
        bins.append(
            {
                "quartile": int(label),
                "sample_count": int(len(members)),
                "value_min": float(np.min([row["features"][feature] for row in members])),
                "value_max": float(np.max([row["features"][feature] for row in members])),
                "n3": n3,
                "l2": l2,
            }
        )
    return {"quartile_edges": [float(value) for value in edges], "bins": bins}


def _spearman(left: np.ndarray, right: np.ndarray) -> float:
    def rank(values: np.ndarray) -> np.ndarray:
        order = np.argsort(values, kind="mergesort")
        ranks = np.empty(len(values), dtype=np.float64)
        ranks[order] = np.arange(len(values), dtype=np.float64)
        return ranks
    left_rank, right_rank = rank(left), rank(right)
    return float(np.corrcoef(left_rank, right_rank)[0, 1])


def main() -> int:
    args = _parse_args()
    paired_path = args.paired_json.resolve()
    config_path = args.config.resolve()
    paired = json.loads(paired_path.read_text(encoding="utf-8"))
    if paired.get("data_roles") != ["valid_iid"] or paired.get("forbidden_roles_accessed") != []:
        raise ValueError("Gate 6D paired payload does not satisfy valid-only contract")
    if paired.get("n3", {}).get("checkpoint_epoch") != 402:
        raise ValueError("paired payload is not N3 best e402")
    if paired.get("l2", {}).get("checkpoint_epoch") != 353:
        raise ValueError("paired payload is not L2 best e353")
    rows = [dict(row) for row in paired["per_sample"]]
    if len(rows) != 128:
        raise ValueError(f"expected 128 paired valid rows, got {len(rows)}")
    q_low_k = _q_low_k_overlap_by_valid_id(_resolved(config_path))
    paired_ids = {str(row["sample_id"]) for row in rows}
    if set(q_low_k) != paired_ids:
        raise ValueError("valid q-low-k input contexts do not align with frozen paired rows")
    for row in rows:
        row["features"] = dict(row["features"])
        row["features"]["q_low_k_overlap_fraction"] = q_low_k[str(row["sample_id"])]
    by_feature = {feature: _feature_bins(rows, feature) for feature, _ in FEATURES}
    true_q4 = by_feature["true_cv_rms_deltaT_K"]["bins"][3]
    qk_q4 = by_feature["q_low_k_overlap_fraction"]["bins"][3]
    amplitude_underestimation = 1.0 - np.asarray([row["n3_amplitude_ratio"] for row in rows])
    true_delta = np.asarray([row["features"]["true_cv_rms_deltaT_K"] for row in rows])
    qk_overlap = np.asarray([row["features"]["q_low_k_overlap_fraction"] for row in rows])
    high_true = true_delta >= np.quantile(true_delta, 0.75)
    high_qk = qk_overlap >= np.quantile(qk_overlap, 0.75)
    intersection = high_true & high_qk
    conclusion = {
        "n3_high_true_cv_rms_amplitude_ratio_mean": true_q4["n3"]["amplitude_ratio_mean"],
        "n3_high_true_cv_rms_signed_log_scale_bias_mean": true_q4["n3"]["signed_log_scale_bias_mean"],
        "n3_high_q_low_k_overlap_amplitude_ratio_mean": qk_q4["n3"]["amplitude_ratio_mean"],
        "n3_high_q_low_k_overlap_signed_log_scale_bias_mean": qk_q4["n3"]["signed_log_scale_bias_mean"],
        "n3_underestimation_vs_q_low_k_overlap_spearman": _spearman(
            qk_overlap, amplitude_underestimation
        ),
        "high_true_and_high_q_low_k_sample_count": int(np.sum(intersection)),
        "high_true_and_high_q_low_k_n3_amplitude_ratio_mean": float(
            np.mean(np.asarray([row["n3_amplitude_ratio"] for row in rows])[intersection])
        ) if np.any(intersection) else None,
        "high_temperature_underestimation_concentrated_in_high_q_low_k": bool(
            true_q4["n3"]["amplitude_ratio_mean"] < 1.0
            and qk_q4["n3"]["amplitude_ratio_mean"] < 1.0
            and np.sum(intersection) >= 4
            and float(np.mean(np.asarray([row["n3_amplitude_ratio"] for row in rows])[intersection])) < 1.0
        ),
        "interpretation": (
            "true only when high-DeltaT and high q-low-k bins both have mean "
            "N3 amplitude ratio below one; this is descriptive valid-only evidence, not tuning input."
        ),
    }
    payload = {
        "schema_version": "heat3d_v5_gate6f_valid_only_amplitude_audit_v1",
        "models": {
            "n3": {"config_id": paired["n3"]["config_id"], "checkpoint_epoch": 402},
            "l2": {"config_id": paired["l2"]["config_id"], "checkpoint_epoch": 353},
        },
        "paired_source": str(paired_path),
        "paired_source_sha256": _sha256(paired_path),
        "config_path": str(config_path),
        "config_sha256": _sha256(config_path),
        "roles_accessed": ["valid_iid"],
        "forbidden_roles_accessed": [],
        "sealed_iid_accessed": False,
        "training_started": False,
        "checkpoint_inference_run": False,
        "features": [{"name": name, "label": label} for name, label in FEATURES],
        "bins": by_feature,
        "high_temperature_amplitude_conclusion": conclusion,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
