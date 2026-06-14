#!/usr/bin/env python3
"""Read-only S5 checkpoint smoke on the Heat3D v3 final-target probe subset."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import sys
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import jax.numpy as jnp
import numpy as np


REPO_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = Path(__file__).resolve().parent
for path in (REPO_DIR, SCRIPTS_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from rigno.dataset_Heat3D_v1 import Heat3DV1MetadataDataset  # noqa: E402
from rigno.heat3d_v1_native_supervised import (  # noqa: E402
    V1SteadyConditionInput,
    V1SteadySupervisedExampleNative,
    V1SteadyTarget,
)
from run_heat3d_v1_medium_controlled_training_export import (  # noqa: E402
    GraphNeuralOperator,
    Heat3DGraphBuilder,
    Heat3DV1NativeSupervisedDataset,
    _device_params,
    _load_params_checkpoint,
    _make_groups_with_progress,
    _predict_temperatures,
    _resolve_training_splits,
    _sample_root,
    _stable_json_hash,
    _train_only_stats,
    _write_json,
)


PROBE_STAGE = "physics_label_v3_final_target_probe_v0"
HF_REVISION = "26733ceb1aad308ba1cc5fc3b8d48537ed48c8c2"
HF_REPO_ID = "133754144X/heat3d-thermal-simulation"
DATA_SOURCE_LABEL = "local HF upload staging copy"
EPS = 1.0e-12


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run read-only S5 checkpoint inference on the v3 final-target probe "
            "subset and export metrics plus reviewer figures."
        )
    )
    parser.add_argument("--subset", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--run-config", type=Path, default=None)
    parser.add_argument(
        "--checkpoint-entry",
        action="append",
        default=None,
        help=(
            "Optional comparison entry in LABEL=CHECKPOINT=RUN_CONFIG format. "
            "When present, the script runs all entries and writes comparison "
            "JSON/MD under --output-dir."
        ),
    )
    parser.add_argument(
        "--comparison-output-json",
        type=Path,
        default=None,
        help="Optional comparison JSON path for --checkpoint-entry mode.",
    )
    parser.add_argument(
        "--comparison-output-md",
        type=Path,
        default=None,
        help="Optional comparison Markdown path for --checkpoint-entry mode.",
    )
    parser.add_argument("--provenance", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=0)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def as_column(array: np.ndarray) -> np.ndarray:
    arr = np.asarray(array, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr[:, None]
    if arr.ndim != 2 or arr.shape[1] != 1:
        raise ValueError(f"expected array shape (N,1), found {arr.shape}")
    return arr


def sample_root_from_subset(path: Path) -> Path:
    path = Path(path)
    return path / "samples" if (path / "samples").is_dir() else path


def optional_batch_size(value: int) -> int | None:
    if int(value) == 0:
        return None
    if int(value) < 0:
        raise ValueError("--batch-size must be >= 1 or 0 for one full prediction batch")
    return int(value)


def stats_from_checkpoint_payload(
    checkpoint_stats: dict[str, Any],
    train_examples: list[Any],
) -> dict[str, Any]:
    """Use train split coords, but checkpoint target/condition normalization.

    Current v3 checkpoints store target and condition normalization, but not
    coord_min/coord_span. The runner computes coordinate stats from the train
    split, so this function reproduces that path without fitting anything on
    the final-target probe samples.
    """

    stats = _train_only_stats(train_examples)
    checkpoint_feature_names = tuple(checkpoint_stats.get("feature_names") or ())
    if checkpoint_feature_names and checkpoint_feature_names != tuple(stats["feature_names"]):
        raise ValueError(
            "Checkpoint feature_names do not match the training subset: "
            f"checkpoint={checkpoint_feature_names} subset={tuple(stats['feature_names'])}"
        )
    stats["feature_names"] = checkpoint_feature_names or tuple(stats["feature_names"])
    stats["target_delta_mean"] = jnp.asarray(
        np.asarray(checkpoint_stats["target_delta_mean"], dtype=np.float32).reshape(1, 1, 1, 1)
    )
    stats["target_delta_std"] = jnp.asarray(
        np.asarray(checkpoint_stats["target_delta_std"], dtype=np.float32).reshape(1, 1, 1, 1)
    )
    stats["condition_mean"] = jnp.asarray(
        np.asarray(checkpoint_stats["condition_mean"], dtype=np.float32).reshape(1, 1, 1, -1)
    )
    stats["condition_std"] = jnp.asarray(
        np.asarray(checkpoint_stats["condition_std"], dtype=np.float32).reshape(1, 1, 1, -1)
    )
    return stats


def load_training_examples(run_config: dict[str, Any], checkpoint_stats: dict[str, Any]) -> list[Any]:
    subset_value = run_config.get("subset")
    if not subset_value:
        raise ValueError("run_config.json missing subset")
    training_root = _sample_root(Path(subset_value))
    split_map_value = run_config.get("split_map_path") or run_config.get("split_map")
    split_map = Path(split_map_value) if split_map_value else None
    split_ids, split_source, _primary, _stress = _resolve_training_splits(training_root, split_map)
    train_ids = split_ids.get("train") or []
    if not train_ids:
        raise ValueError(f"No train ids resolved from {training_root}; split_source={split_source}")
    feature_names = tuple(checkpoint_stats.get("feature_names") or ())
    k_encoding_mode = "diag3" if {"k_x", "k_y", "k_z"}.issubset(feature_names) else "native"
    dataset = Heat3DV1NativeSupervisedDataset(
        training_root,
        k_encoding_mode=k_encoding_mode,
        boundary_mask_fallback=bool(run_config.get("boundary_mask_fallback", True)),
    )
    index_by_id = dataset.sample_index_by_id()
    missing = [sample_id for sample_id in train_ids if sample_id not in index_by_id]
    if missing:
        raise FileNotFoundError(f"training subset missing train samples: {missing[:10]}")
    return [dataset[index_by_id[sample_id]] for sample_id in train_ids]


def load_probe_examples(subset: Path, checkpoint_stats: dict[str, Any]) -> tuple[list[Any], dict[str, Path]]:
    sample_root = sample_root_from_subset(subset)
    feature_names = tuple(checkpoint_stats.get("feature_names") or ())
    k_encoding_mode = "diag3" if {"k_x", "k_y", "k_z"}.issubset(feature_names) else "native"
    dataset = Heat3DV1MetadataDataset(
        sample_root,
        k_encoding_mode=k_encoding_mode,
        allowed_stages=(PROBE_STAGE,),
        boundary_mask_fallback=True,
    )
    examples: list[V1SteadySupervisedExampleNative] = []
    sample_dirs_by_id: dict[str, Path] = {}
    for sample in dataset.samples:
        sample_dir = Path(sample["sample_dir"])
        temperature = as_column(np.load(sample_dir / "temperature.npy"))
        condition = V1SteadyConditionInput(
            coords=np.asarray(sample["coords"], dtype=np.float64),
            condition_features=np.asarray(sample["physics_input"].features, dtype=np.float64),
            condition_feature_names=tuple(sample["physics_input"].feature_names),
            k_encoding_mode=k_encoding_mode,
        )
        target = V1SteadyTarget(target_u=temperature)
        example = V1SteadySupervisedExampleNative(
            sample_id=str(sample["sample_id"]),
            condition=condition,
            target=target,
            meta=dict(sample["meta"]),
        )
        examples.append(example)
        sample_dirs_by_id[example.sample_id] = sample_dir
    examples.sort(key=lambda item: str(item.meta.get("probe_id") or item.sample_id))
    return examples, sample_dirs_by_id


def probe_id_from_meta(meta: dict[str, Any], fallback: str) -> str:
    value = meta.get("probe_id")
    if value:
        return str(value)
    for token in fallback.split("_"):
        if token.startswith("P") and token[1:].isdigit():
            return token
    return fallback


def grid_axes(coords: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    coords = np.asarray(coords, dtype=np.float64)
    xs = np.unique(coords[:, 0])
    ys = np.unique(coords[:, 1])
    zs = np.unique(coords[:, 2])
    ix = np.searchsorted(xs, coords[:, 0])
    iy = np.searchsorted(ys, coords[:, 1])
    iz = np.searchsorted(zs, coords[:, 2])
    return xs, ys, zs, (ix, iy, iz)


def to_grid(values: np.ndarray, coords: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    xs, ys, zs, (ix, iy, iz) = grid_axes(coords)
    arr = np.full((len(xs), len(ys), len(zs)), np.nan, dtype=np.float64)
    arr[ix, iy, iz] = np.asarray(values, dtype=np.float64).reshape(-1)
    return arr, xs, ys, zs


def slice2d(values: np.ndarray, coords: np.ndarray, z_index: int) -> np.ndarray:
    arr, _xs, _ys, _zs = to_grid(values, coords)
    return arr[:, :, z_index].T


def effective_k(k_field: np.ndarray) -> np.ndarray:
    k = np.asarray(k_field, dtype=np.float64)
    if k.ndim == 1:
        return k
    if k.shape[1] == 1:
        return k[:, 0]
    return np.mean(k[:, : min(k.shape[1], 3)], axis=1)


def bbox_for_mask(coords: np.ndarray, mask: np.ndarray) -> list[list[float]] | None:
    mask = np.asarray(mask, dtype=bool).reshape(-1)
    if not np.any(mask):
        return None
    pts = np.asarray(coords, dtype=np.float64)[mask]
    lo = np.min(pts, axis=0)
    hi = np.max(pts, axis=0)
    span = np.ptp(np.asarray(coords, dtype=np.float64), axis=0)
    pad = np.where(span > 0.0, span * 0.01, 0.0)
    lo = np.maximum(np.min(coords, axis=0), lo - pad)
    hi = np.minimum(np.max(coords, axis=0), hi + pad)
    return [[float(v) for v in lo], [float(v) for v in hi]]


def rmse(values: np.ndarray) -> float:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    if arr.size == 0:
        return float("nan")
    return float(np.sqrt(np.mean(np.square(arr))))


def mae(values: np.ndarray) -> float:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    if arr.size == 0:
        return float("nan")
    return float(np.mean(np.abs(arr)))


def masked_rmse(error: np.ndarray, mask: np.ndarray) -> float | None:
    mask = np.asarray(mask, dtype=bool).reshape(-1)
    if not np.any(mask):
        return None
    return rmse(np.asarray(error).reshape(-1)[mask])


def top_fraction_rmse(error: np.ndarray, score: np.ndarray, fraction: float) -> float:
    score = np.asarray(score, dtype=np.float64).reshape(-1)
    error = np.asarray(error, dtype=np.float64).reshape(-1)
    count = max(1, int(math.ceil(score.size * fraction)))
    indices = np.argsort(score)[-count:]
    return rmse(error[indices])


def strong_q_mask(q: np.ndarray) -> tuple[np.ndarray, float | None, float | None]:
    qv = np.asarray(q, dtype=np.float64).reshape(-1)
    if qv.size == 0:
        return np.zeros(0, dtype=bool), None, None
    q_min = float(np.min(qv))
    q_max = float(np.max(qv))
    if q_max <= 0.0:
        return qv > 0.0, None, None
    positive = qv[qv > 0.0]
    background = q_min if q_min > 0.0 else 0.0
    if positive.size and q_min <= 0.0:
        background = 0.0
    threshold = background + 0.5 * (q_max - background)
    return qv > threshold, float(background), float(threshold)


def condition_value(meta: dict[str, Any], key: str) -> str:
    if key in meta and meta[key] not in (None, ""):
        return str(meta[key])
    generation = meta.get("generation_config")
    if isinstance(generation, dict) and key in generation:
        return str(generation[key])
    return "unknown"


def compute_metrics(
    sample_id: str,
    sample_dir: Path,
    prediction: np.ndarray,
) -> dict[str, Any]:
    meta = load_json(sample_dir / "sample_meta.json")
    coords = np.asarray(np.load(sample_dir / "coords.npy"), dtype=np.float64)
    k_field = np.asarray(np.load(sample_dir / "k_field.npy"), dtype=np.float64)
    q_field = as_column(np.load(sample_dir / "q_field.npy")).reshape(-1)
    label = as_column(np.load(sample_dir / "temperature.npy")).reshape(-1)
    pred = as_column(prediction).reshape(-1)
    if pred.shape != label.shape:
        raise ValueError(f"{sample_id}: prediction shape {pred.shape} != label shape {label.shape}")

    t_ref = float(
        ((meta.get("boundary_params") or {}).get("bottom") or {}).get("fixed_temperature_K", 300.0)
    )
    delta = label - t_ref
    error = pred - label
    q_mask = q_field > 0.0
    strong_mask, q_background, q_threshold = strong_q_mask(q_field)
    background_mask = ~q_mask
    if not np.any(background_mask) and q_threshold is not None:
        background_mask = q_field <= q_threshold
    label_score = delta
    k_eff = effective_k(k_field)
    k_unique = int(np.unique(np.round(k_eff, decimals=8)).size)

    row: dict[str, Any] = {
        "sample_id": sample_id,
        "sample_dir": sample_dir.name,
        "probe_id": probe_id_from_meta(meta, sample_id),
        "probe_family": meta.get("probe_family"),
        "intended_stressor": meta.get("intended_stressor"),
        "resolution": meta.get("resolution"),
        "k_mode": meta.get("k_mode"),
        "k_region_mode": meta.get("k_region_mode"),
        "source_category": meta.get("source_category"),
        "q_power_range": meta.get("q_power_range"),
        "bc_category": meta.get("bc_category"),
        "label_status": meta.get("label_status"),
        "RMSE": rmse(error),
        "MAE": mae(error),
        "max_abs_error": float(np.max(np.abs(error))),
        "relative_RMSE_on_DeltaT": rmse(error) / max(rmse(delta), EPS),
        "T_label_min": float(np.min(label)),
        "T_label_max": float(np.max(label)),
        "T_label_mean": float(np.mean(label)),
        "T_pred_min": float(np.min(pred)),
        "T_pred_max": float(np.max(pred)),
        "T_pred_mean": float(np.mean(pred)),
        "Tmax_error": float(np.max(pred) - np.max(label)),
        "top_1_percent_RMSE": top_fraction_rmse(error, label_score, 0.01),
        "top_5_percent_RMSE": top_fraction_rmse(error, label_score, 0.05),
        "q_region_RMSE": masked_rmse(error, q_mask),
        "background_region_RMSE": masked_rmse(error, background_mask),
        "q_nonzero_fraction": float(np.mean(q_mask)),
        "q_bbox": bbox_for_mask(coords, q_mask),
        "q_strong_bbox": bbox_for_mask(coords, strong_mask),
        "q_strong_fraction": float(np.mean(strong_mask)) if strong_mask.size else None,
        "q_strong_region_RMSE": masked_rmse(error, strong_mask),
        "q_background_value": q_background,
        "q_strong_threshold": q_threshold,
        "k_unique": k_unique,
        "k_min": float(np.min(k_eff)),
        "k_max": float(np.max(k_eff)),
    }

    if str(row["probe_id"]) == "P09" or np.asarray(k_field).shape[1] == 3:
        k3 = np.asarray(k_field, dtype=np.float64)
        if k3.ndim == 2 and k3.shape[1] >= 3:
            ratio = np.max(k3[:, :3], axis=1) / np.maximum(np.min(k3[:, :3], axis=1), EPS)
            patch_mask = ratio > 1.5
            row.update(
                {
                    "kx_min": float(np.min(k3[:, 0])),
                    "kx_max": float(np.max(k3[:, 0])),
                    "ky_min": float(np.min(k3[:, 1])),
                    "ky_max": float(np.max(k3[:, 1])),
                    "kz_min": float(np.min(k3[:, 2])),
                    "kz_max": float(np.max(k3[:, 2])),
                    "anisotropy_ratio": float(np.max(ratio)),
                    "anisotropic_patch_RMSE": masked_rmse(error, patch_mask),
                    "anisotropic_patch_bbox": bbox_for_mask(coords, patch_mask),
                }
            )

    if str(row["probe_id"]) == "P10":
        row.update(
            {
                "localized_top_contact_supported": False,
                "side_asymmetry_supported": False,
                "implemented_bc": "V1 global top Robin very_high_top_h",
            }
        )
    return row


def box_vertices(lo: np.ndarray, hi: np.ndarray) -> list[list[tuple[float, float, float]]]:
    x0, y0, z0 = lo
    x1, y1, z1 = hi
    return [
        [(x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0)],
        [(x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1)],
        [(x0, y0, z0), (x1, y0, z0), (x1, y0, z1), (x0, y0, z1)],
        [(x0, y1, z0), (x1, y1, z0), (x1, y1, z1), (x0, y1, z1)],
        [(x0, y0, z0), (x0, y1, z0), (x0, y1, z1), (x0, y0, z1)],
        [(x1, y0, z0), (x1, y1, z0), (x1, y1, z1), (x1, y0, z1)],
    ]


def add_box(ax, bbox: list[list[float]] | None, color: str, alpha: float, label: str) -> None:
    if bbox is None:
        return
    lo = np.asarray(bbox[0], dtype=np.float64)
    hi = np.asarray(bbox[1], dtype=np.float64)
    if np.any(hi < lo):
        return
    faces = box_vertices(lo, hi)
    poly = Poly3DCollection(faces, facecolors=color, edgecolors="k", linewidths=0.3, alpha=alpha)
    ax.add_collection3d(poly)
    if label:
        center = 0.5 * (lo + hi)
        ax.text(center[0], center[1], center[2], label, fontsize=6)


def plot_structure(ax, sample_dir: Path, metrics: dict[str, Any], title: str) -> None:
    meta = load_json(sample_dir / "sample_meta.json")
    coords = np.asarray(np.load(sample_dir / "coords.npy"), dtype=np.float64)
    k_field = np.asarray(np.load(sample_dir / "k_field.npy"), dtype=np.float64)
    q_field = as_column(np.load(sample_dir / "q_field.npy")).reshape(-1)
    k_eff = effective_k(k_field)
    k_low = np.percentile(k_eff, 20)
    k_high = np.percentile(k_eff, 80)
    high_mask = k_eff >= k_high if np.max(k_eff) > np.min(k_eff) else np.zeros_like(k_eff, dtype=bool)
    low_mask = k_eff <= k_low if np.max(k_eff) > np.min(k_eff) else np.zeros_like(k_eff, dtype=bool)
    strong_mask, _background, _threshold = strong_q_mask(q_field)
    add_box(ax, bbox_for_mask(coords, high_mask), "tab:blue", 0.20, "high-k")
    add_box(ax, bbox_for_mask(coords, low_mask), "gold", 0.25, "low-k")
    add_box(ax, bbox_for_mask(coords, strong_mask), "tab:red", 0.38, "source")
    if metrics.get("anisotropic_patch_bbox") is not None:
        add_box(ax, metrics.get("anisotropic_patch_bbox"), "tab:purple", 0.28, "anisotropic")
    if str(metrics.get("probe_id")) == "P10":
        lo = np.min(coords, axis=0)
        hi = np.max(coords, axis=0)
        z = hi[2]
        dz = max(float(np.ptp(coords[:, 2])) * 0.015, 1.0e-9)
        add_box(
            ax,
            [[float(lo[0]), float(lo[1]), float(z - dz)], [float(hi[0]), float(hi[1]), float(z)]],
            "tab:green",
            0.25,
            "top high-h",
        )
    lo = np.min(coords, axis=0)
    hi = np.max(coords, axis=0)
    ax.set_xlim(float(lo[0]), float(hi[0]))
    ax.set_ylim(float(lo[1]), float(hi[1]))
    ax.set_zlim(float(lo[2]), float(hi[2]))
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.set_title(title, fontsize=7)
    ax.view_init(elev=22, azim=-45)
    ax.set_box_aspect((1, 1, 0.35))


def add_image(ax, data: np.ndarray, title: str, cmap: str = "viridis") -> None:
    image = ax.imshow(np.asarray(data, dtype=np.float64), origin="lower", cmap=cmap, aspect="auto")
    ax.set_title(title, fontsize=7)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.colorbar(image, ax=ax, fraction=0.046, pad=0.02)


def figure_title(metrics: dict[str, Any], checkpoint_name: str) -> str:
    return (
        f"{metrics.get('probe_id')} / {metrics.get('k_region_mode')} / "
        f"{metrics.get('source_category')} / {metrics.get('bc_category')} / "
        f"{checkpoint_name} / data source = {DATA_SOURCE_LABEL}"
    )


def make_reviewer_sheet(
    sample_dir: Path,
    prediction: np.ndarray,
    metrics: dict[str, Any],
    checkpoint_name: str,
    output_path: Path,
) -> None:
    coords = np.asarray(np.load(sample_dir / "coords.npy"), dtype=np.float64)
    k_field = np.asarray(np.load(sample_dir / "k_field.npy"), dtype=np.float64)
    q_field = as_column(np.load(sample_dir / "q_field.npy")).reshape(-1)
    label = as_column(np.load(sample_dir / "temperature.npy")).reshape(-1)
    pred = as_column(prediction).reshape(-1)
    error = np.abs(pred - label)
    _arr, _xs, _ys, zs = to_grid(label, coords)
    q_grid, _qx, _qy, _qz = to_grid(q_field, coords)
    source_z = int(np.argmax(np.nansum(q_grid, axis=(0, 1))))
    mid_z = int(np.argmin(np.abs(zs - np.median(zs))))
    k_eff = effective_k(k_field)

    fig = plt.figure(figsize=(18, 12), constrained_layout=True)
    grid = fig.add_gridspec(3, 5)
    ax3d = fig.add_subplot(grid[:, 0], projection="3d")
    plot_structure(ax3d, sample_dir, metrics, "structure")

    panels = [
        (slice2d(k_eff, coords, source_z), "source z: k", "viridis"),
        (slice2d(q_field, coords, source_z), "source z: q", "inferno"),
        (slice2d(label, coords, source_z), "source z: T label", "magma"),
        (slice2d(pred, coords, source_z), "source z: T pred", "magma"),
        (slice2d(error, coords, source_z), "source z: abs error", "Reds"),
        (slice2d(label, coords, mid_z), "z-mid: T label", "magma"),
        (slice2d(pred, coords, mid_z), "z-mid: T pred", "magma"),
        (slice2d(error, coords, mid_z), "z-mid: abs error", "Reds"),
    ]
    for idx, (data, title, cmap) in enumerate(panels):
        row = idx // 4
        col = idx % 4 + 1
        add_image(fig.add_subplot(grid[row, col]), data, title, cmap)
    ax_text = fig.add_subplot(grid[2, 1:])
    ax_text.axis("off")
    text = (
        f"RMSE={metrics['RMSE']:.4g}  MAE={metrics['MAE']:.4g}  "
        f"relRMSE_DeltaT={metrics['relative_RMSE_on_DeltaT']:.4g}\n"
        f"Tmax_error={metrics['Tmax_error']:.4g}  top5%RMSE={metrics['top_5_percent_RMSE']:.4g}  "
        f"q_region_RMSE={metrics.get('q_region_RMSE')}\n"
        f"q_bbox={metrics.get('q_bbox')}\n"
        f"P10 localized_top_contact_supported={metrics.get('localized_top_contact_supported')}  "
        f"side_asymmetry_supported={metrics.get('side_asymmetry_supported')}"
    )
    ax_text.text(0.0, 0.9, text, va="top", ha="left", fontsize=9, family="monospace")
    fig.suptitle(figure_title(metrics, checkpoint_name), fontsize=10)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def make_structure_overview(
    sample_dirs: list[Path],
    metrics_rows: list[dict[str, Any]],
    checkpoint_name: str,
    output_path: Path,
) -> None:
    fig = plt.figure(figsize=(20, 8), constrained_layout=True)
    for index, (sample_dir, metrics) in enumerate(zip(sample_dirs, metrics_rows), start=1):
        ax = fig.add_subplot(2, 5, index, projection="3d")
        plot_structure(
            ax,
            sample_dir,
            metrics,
            f"{metrics.get('probe_id')} / {metrics.get('k_region_mode')}\n{metrics.get('source_category')}",
        )
    fig.suptitle(f"True structure overview / {checkpoint_name} / data source = {DATA_SOURCE_LABEL}", fontsize=12)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def make_error_overview(
    sample_dirs: list[Path],
    predictions: dict[str, np.ndarray],
    metrics_rows: list[dict[str, Any]],
    checkpoint_name: str,
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(10, 3, figsize=(10, 26), constrained_layout=True)
    for row_index, (sample_dir, metrics) in enumerate(zip(sample_dirs, metrics_rows)):
        coords = np.asarray(np.load(sample_dir / "coords.npy"), dtype=np.float64)
        q_field = as_column(np.load(sample_dir / "q_field.npy")).reshape(-1)
        label = as_column(np.load(sample_dir / "temperature.npy")).reshape(-1)
        pred = as_column(predictions[str(metrics["sample_id"])]).reshape(-1)
        error = np.abs(pred - label)
        q_grid, _xs, _ys, _zs = to_grid(q_field, coords)
        source_z = int(np.argmax(np.nansum(q_grid, axis=(0, 1))))
        for col, (values, title, cmap) in enumerate(
            (
                (label, "label", "magma"),
                (pred, "pred", "magma"),
                (error, "abs error", "Reds"),
            )
        ):
            add_image(
                axes[row_index, col],
                slice2d(values, coords, source_z),
                f"{metrics.get('probe_id')} {title}",
                cmap,
            )
    fig.suptitle(f"True vs pred error overview / {checkpoint_name} / data source = {DATA_SOURCE_LABEL}", fontsize=12)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def make_metric_bars(metrics_rows: list[dict[str, Any]], output_path: Path) -> None:
    probe_ids = [str(row["probe_id"]) for row in metrics_rows]
    fields = ["RMSE", "MAE", "Tmax_error", "top_5_percent_RMSE"]
    fig, axes = plt.subplots(2, 2, figsize=(16, 9), constrained_layout=True)
    for ax, field in zip(axes.reshape(-1), fields):
        values = [float(row[field]) for row in metrics_rows]
        ax.bar(probe_ids, values)
        ax.set_title(field)
        ax.tick_params(axis="x", rotation=45)
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle("S5 final-target probe metrics", fontsize=12)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def write_metrics(metrics_rows: list[dict[str, Any]], metrics_dir: Path) -> None:
    metrics_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "sample_count": len(metrics_rows),
        "metrics": metrics_rows,
        "worst_3_by_RMSE": sorted(metrics_rows, key=lambda row: row["RMSE"], reverse=True)[:3],
        "worst_3_by_relative_RMSE_on_DeltaT": sorted(
            metrics_rows,
            key=lambda row: row["relative_RMSE_on_DeltaT"],
            reverse=True,
        )[:3],
    }
    _write_json(metrics_dir / "s5_probe_metrics.json", payload)

    keys = sorted({key for row in metrics_rows for key in row})
    with (metrics_dir / "s5_probe_metrics.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in metrics_rows:
            writer.writerow(row)

    lines = [
        "# S5 Final-Target Probe Metrics",
        "",
        "| probe | RMSE | MAE | relRMSE_DeltaT | Tmax_error | top5% RMSE | q_region RMSE | background RMSE |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in metrics_rows:
        lines.append(
            "| {probe} | {rmse:.6g} | {mae:.6g} | {rel:.6g} | {tmax:.6g} | {top5:.6g} | {q} | {bg} |".format(
                probe=row["probe_id"],
                rmse=row["RMSE"],
                mae=row["MAE"],
                rel=row["relative_RMSE_on_DeltaT"],
                tmax=row["Tmax_error"],
                top5=row["top_5_percent_RMSE"],
                q=_fmt_optional(row.get("q_region_RMSE")),
                bg=_fmt_optional(row.get("background_region_RMSE")),
            )
        )
    lines.extend(["", "## Worst 3 By RMSE", ""])
    for row in payload["worst_3_by_RMSE"]:
        lines.append(f"- `{row['probe_id']}` `{row['sample_id']}` RMSE={row['RMSE']:.6g}")
    write_text(metrics_dir / "s5_probe_metrics.md", "\n".join(lines) + "\n")


def _fmt_optional(value: Any) -> str:
    if value is None:
        return "NA"
    if isinstance(value, float):
        if math.isnan(value):
            return "NA"
        return f"{value:.6g}"
    return str(value)


def write_report(
    output_dir: Path,
    devbox_head: str,
    provenance: dict[str, Any],
    checkpoint_path: Path,
    checkpoint_payload: dict[str, Any],
    metrics_rows: list[dict[str, Any]],
    figure_paths: list[Path],
) -> None:
    worst = sorted(metrics_rows, key=lambda row: row["RMSE"], reverse=True)[:3]
    lines = [
        "# S5 Final-Target Probe Checkpoint Smoke",
        "",
        "This is checkpoint inference smoke only. It is not training and not a benchmark conclusion.",
        "",
        f"- devbox HEAD: `{devbox_head}`",
        f"- data source: `{provenance.get('source')}`",
        f"- HF repo: `{provenance.get('hf_repo_id')}`",
        f"- HF revision: `{provenance.get('hf_revision')}`",
        f"- copied_from_local_hf_upload_staging: `{provenance.get('copied_from_local_hf_upload_staging')}`",
        f"- used_local_regeneration: `{provenance.get('used_local_regeneration')}`",
        f"- used_generator_this_round: `{provenance.get('used_generator_this_round')}`",
        f"- sha256_identity_check: `{provenance.get('sha256_identity_check')}`",
        f"- S5 checkpoint: `{checkpoint_path}`",
        f"- checkpoint_kind: `{checkpoint_payload.get('checkpoint_kind')}`",
        f"- checkpoint_epoch: `{checkpoint_payload.get('epoch')}`",
        f"- checkpoint_git_commit: `{checkpoint_payload.get('git_commit')}`",
        f"- checkpoint_format_version: `{checkpoint_payload.get('checkpoint_format_version')}`",
        "",
        "## Metrics",
        "",
        "| probe | RMSE | MAE | relRMSE_DeltaT | Tmax_error | top1% RMSE | top5% RMSE |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in metrics_rows:
        lines.append(
            f"| {row['probe_id']} | {row['RMSE']:.6g} | {row['MAE']:.6g} | "
            f"{row['relative_RMSE_on_DeltaT']:.6g} | {row['Tmax_error']:.6g} | "
            f"{row['top_1_percent_RMSE']:.6g} | {row['top_5_percent_RMSE']:.6g} |"
        )
    lines.extend(["", "## Worst 3 Probes", ""])
    for row in worst:
        lines.append(
            f"- `{row['probe_id']}` `{row['sample_id']}`: RMSE={row['RMSE']:.6g}, "
            f"relRMSE_DeltaT={row['relative_RMSE_on_DeltaT']:.6g}"
        )
    lines.extend(["", "## Structure Checks", ""])
    for row in metrics_rows:
        lines.append(
            f"- `{row['probe_id']}`: family={row.get('probe_family')}, "
            f"k_region={row.get('k_region_mode')}, source={row.get('source_category')}, "
            f"bc={row.get('bc_category')}, label_status={row.get('label_status')}"
        )
    lines.extend(["", "## Figure Paths", ""])
    for path in figure_paths:
        lines.append(f"- `{path}`")
    write_text(output_dir / "s5_probe_smoke_report.md", "\n".join(lines) + "\n")


def _devbox_head() -> str:
    devbox_head = "unknown"
    head_path = REPO_DIR / ".git" / "HEAD"
    try:
        import subprocess

        devbox_head = subprocess.check_output(
            ["git", "log", "-1", "--oneline"],
            cwd=REPO_DIR,
            text=True,
        ).strip()
    except Exception:
        if head_path.exists():
            devbox_head = head_path.read_text(encoding="utf-8").strip()
    return devbox_head


def _mean_metric(metrics_rows: list[dict[str, Any]], key: str) -> float | None:
    values = []
    for row in metrics_rows:
        value = row.get(key)
        if value is None:
            continue
        try:
            value = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(value):
            values.append(value)
    if not values:
        return None
    return float(np.mean(values))


def _probe_focus(metrics_rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_probe = {str(row.get("probe_id")): row for row in metrics_rows}
    return {
        "P02": by_probe.get("P02"),
        "P03": by_probe.get("P03"),
        "P09": by_probe.get("P09"),
        "P10": by_probe.get("P10"),
    }


def _comparison_entry_summary(label: str, checkpoint: Path, output_dir: Path, metrics_rows: list[dict[str, Any]]) -> dict[str, Any]:
    summary = {
        "label": label,
        "checkpoint": str(checkpoint),
        "output_dir": str(output_dir),
        "sample_count": len(metrics_rows),
        "RMSE": _mean_metric(metrics_rows, "RMSE"),
        "MAE": _mean_metric(metrics_rows, "MAE"),
        "relRMSE_DeltaT": _mean_metric(metrics_rows, "relative_RMSE_on_DeltaT"),
        "Tmax_error": _mean_metric(metrics_rows, "Tmax_error"),
        "top_5_percent_RMSE": _mean_metric(metrics_rows, "top_5_percent_RMSE"),
        "q_region_RMSE": _mean_metric(metrics_rows, "q_region_RMSE"),
        "strong_q_RMSE": _mean_metric(metrics_rows, "q_strong_region_RMSE"),
        "background_RMSE": _mean_metric(metrics_rows, "background_region_RMSE"),
        "P09_anisotropy": {},
        "P10_unsupported_gap_flags": {},
        "focus_probes": _probe_focus(metrics_rows),
    }
    p09 = summary["focus_probes"].get("P09")
    if p09:
        summary["P09_anisotropy"] = {
            "anisotropic_patch_RMSE": p09.get("anisotropic_patch_RMSE"),
            "anisotropic_background_RMSE": p09.get("anisotropic_background_RMSE"),
            "k_channel_corr_pred_error": p09.get("k_channel_corr_pred_error"),
        }
    p10 = summary["focus_probes"].get("P10")
    if p10:
        summary["P10_unsupported_gap_flags"] = {
            "localized_top_contact_supported": p10.get("localized_top_contact_supported"),
            "side_asymmetry_supported": p10.get("side_asymmetry_supported"),
            "gap_note": p10.get("gap_note"),
        }
    return summary


def _fmt_optional_value(value: Any) -> str:
    if value is None:
        return "-"
    try:
        value = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(value):
        return "-"
    return f"{value:.6g}"


def _render_comparison_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Heat3D v3 S5-Family Final Probe Comparison",
        "",
        "Read-only checkpoint inference on the existing final-target probe subset. No final-probe training is performed.",
        "",
        "## Overall",
        "",
        "| label | RMSE | MAE | relRMSE_DeltaT | Tmax_error | top5% RMSE | q-region RMSE | strong-q RMSE | background RMSE |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for item in payload["entries"]:
        lines.append(
            f"| {item['label']} | {_fmt_optional_value(item.get('RMSE'))} | "
            f"{_fmt_optional_value(item.get('MAE'))} | {_fmt_optional_value(item.get('relRMSE_DeltaT'))} | "
            f"{_fmt_optional_value(item.get('Tmax_error'))} | {_fmt_optional_value(item.get('top_5_percent_RMSE'))} | "
            f"{_fmt_optional_value(item.get('q_region_RMSE'))} | {_fmt_optional_value(item.get('strong_q_RMSE'))} | "
            f"{_fmt_optional_value(item.get('background_RMSE'))} |"
        )
    lines.extend(["", "## Focus Probes", ""])
    lines.append("| label | probe | RMSE | relRMSE_DeltaT | top5% RMSE | q-region RMSE | strong-q RMSE | note |")
    lines.append("| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |")
    for item in payload["entries"]:
        for probe_id in ("P02", "P03", "P09", "P10"):
            row = (item.get("focus_probes") or {}).get(probe_id)
            if not row:
                continue
            note = ""
            if probe_id == "P09":
                note = "anisotropy metrics recorded"
            elif probe_id == "P10":
                note = "localized top contact / side asymmetry unsupported"
            lines.append(
                f"| {item['label']} | {probe_id} | {_fmt_optional_value(row.get('RMSE'))} | "
                f"{_fmt_optional_value(row.get('relative_RMSE_on_DeltaT'))} | "
                f"{_fmt_optional_value(row.get('top_5_percent_RMSE'))} | "
                f"{_fmt_optional_value(row.get('q_region_RMSE'))} | "
                f"{_fmt_optional_value(row.get('q_strong_region_RMSE'))} | {note} |"
            )
    lines.extend(
        [
            "",
            "P10 gap flag: localized top contact and side asymmetry remain unsupported by the current input/metadata path.",
        ]
    )
    return "\n".join(lines) + "\n"


def _parse_checkpoint_entry(token: str) -> tuple[str, Path, Path]:
    parts = token.split("=")
    if len(parts) != 3:
        raise ValueError(
            "--checkpoint-entry must use LABEL=CHECKPOINT=RUN_CONFIG format, "
            f"found {token!r}"
        )
    label, checkpoint, run_config = (part.strip() for part in parts)
    if not label or not checkpoint or not run_config:
        raise ValueError(f"invalid empty field in --checkpoint-entry {token!r}")
    return label, Path(checkpoint), Path(run_config)


def _run_single_probe(
    *,
    subset: Path,
    checkpoint: Path,
    run_config_path: Path,
    provenance_path: Path,
    output_dir: Path,
    batch_size: int,
) -> dict[str, Any]:
    predictions_dir = output_dir / "predictions"
    metadata_dir = output_dir / "metadata"
    metrics_dir = output_dir / "metrics"
    figures_dir = output_dir / "figures"
    for path in (predictions_dir, metadata_dir, metrics_dir, figures_dir):
        path.mkdir(parents=True, exist_ok=True)

    if not checkpoint.is_file():
        raise FileNotFoundError(f"--checkpoint not found: {checkpoint}")
    if not run_config_path.is_file():
        raise FileNotFoundError(f"--run-config not found: {run_config_path}")
    if not provenance_path.is_file():
        raise FileNotFoundError(f"--provenance not found: {provenance_path}")

    provenance = load_json(provenance_path)
    if provenance.get("used_local_regeneration") is not False:
        raise ValueError("provenance must state used_local_regeneration=false")
    if provenance.get("used_generator_this_round") is not False:
        raise ValueError("provenance must state used_generator_this_round=false")
    if provenance.get("sha256_identity_check") != "pass":
        raise ValueError("provenance must state sha256_identity_check=pass")

    checkpoint_payload = _load_params_checkpoint(checkpoint)
    model_config = dict(checkpoint_payload.get("model_config") or {})
    checkpoint_stats = dict(checkpoint_payload.get("train_only_normalization") or {})
    if not model_config:
        raise ValueError("checkpoint missing model_config")
    if not checkpoint_stats:
        raise ValueError("checkpoint missing train_only_normalization")
    run_config = load_json(run_config_path)

    train_examples = load_training_examples(run_config, checkpoint_stats)
    stats = stats_from_checkpoint_payload(checkpoint_stats, train_examples)
    probe_examples, sample_dirs_by_id = load_probe_examples(subset, checkpoint_stats)
    graph_config = dict(run_config.get("graph_config") or {})
    builder = Heat3DGraphBuilder(**graph_config)
    graph_seed = int(run_config.get("graph_seed", 0) or 0)
    groups = _make_groups_with_progress(
        probe_examples,
        stats,
        builder,
        "v3_final_target_probe",
        False,
        "off",
        graph_seed,
        batch_size=optional_batch_size(batch_size),
        drop_last=False,
        profile_counts=None,
    )
    model = GraphNeuralOperator(**model_config)
    predictions = _predict_temperatures(
        model,
        _device_params(checkpoint_payload["params"]),
        groups,
        stats,
    )
    np.savez_compressed(predictions_dir / "s5_probe_predictions.npz", **predictions)
    for sample_id, pred in predictions.items():
        probe_id = probe_id_from_meta(load_json(sample_dirs_by_id[sample_id] / "sample_meta.json"), sample_id)
        np.save(predictions_dir / f"{probe_id}_pred.npy", pred)

    metrics_rows = [
        compute_metrics(example.sample_id, sample_dirs_by_id[example.sample_id], predictions[example.sample_id])
        for example in probe_examples
    ]
    write_metrics(metrics_rows, metrics_dir)

    checkpoint_name = checkpoint.parent.name
    ordered_sample_dirs = [sample_dirs_by_id[example.sample_id] for example in probe_examples]
    figure_paths: list[Path] = []
    for example, metrics in zip(probe_examples, metrics_rows):
        probe_id = str(metrics["probe_id"])
        path = figures_dir / f"{probe_id}_reviewer_sheet.png"
        make_reviewer_sheet(
            sample_dirs_by_id[example.sample_id],
            predictions[example.sample_id],
            metrics,
            checkpoint_name,
            path,
        )
        figure_paths.append(path)
    structure_overview = figures_dir / "reviewer_true_structure_overview.png"
    error_overview = figures_dir / "reviewer_true_vs_pred_error_overview.png"
    metric_bars = figures_dir / "reviewer_metric_bars.png"
    make_structure_overview(ordered_sample_dirs, metrics_rows, checkpoint_name, structure_overview)
    make_error_overview(ordered_sample_dirs, predictions, metrics_rows, checkpoint_name, error_overview)
    make_metric_bars(metrics_rows, metric_bars)
    figure_paths.extend([structure_overview, error_overview, metric_bars])

    devbox_head = _devbox_head()

    prediction_metadata = {
        "diagnostic_scope": "checkpoint inference smoke; no training",
        "checkpoint": str(checkpoint),
        "checkpoint_kind": checkpoint_payload.get("checkpoint_kind"),
        "checkpoint_epoch": checkpoint_payload.get("epoch"),
        "checkpoint_record": checkpoint_payload.get("record"),
        "checkpoint_git_commit": checkpoint_payload.get("git_commit"),
        "checkpoint_format_version": checkpoint_payload.get("checkpoint_format_version"),
        "model_config_hash": checkpoint_payload.get("model_config_hash") or _stable_json_hash(model_config),
        "train_stats_hash": checkpoint_payload.get("train_stats_hash"),
        "subset": str(subset),
        "sample_count": len(metrics_rows),
        "sample_ids": [example.sample_id for example in probe_examples],
        "graph_config": graph_config,
        "graph_seed": graph_seed,
        "group_count": len(groups),
        "group_sample_counts": [len(group["sample_ids"]) for group in groups],
        "provenance": provenance,
        "output_dir": str(output_dir),
        "devbox_head": devbox_head,
    }
    _write_json(metadata_dir / "s5_probe_prediction_metadata.json", prediction_metadata)
    write_report(output_dir, devbox_head, provenance, checkpoint, checkpoint_payload, metrics_rows, figure_paths)
    return {
        "metrics_rows": metrics_rows,
        "metadata": prediction_metadata,
        "output_dir": str(output_dir),
    }


def main() -> int:
    args = parse_args()
    if args.checkpoint_entry:
        entries = [_parse_checkpoint_entry(token) for token in args.checkpoint_entry]
        comparison_entries = []
        for label, checkpoint, run_config_path in entries:
            entry_output_dir = args.output_dir / label
            result = _run_single_probe(
                subset=args.subset,
                checkpoint=checkpoint,
                run_config_path=run_config_path,
                provenance_path=args.provenance,
                output_dir=entry_output_dir,
                batch_size=args.batch_size,
            )
            comparison_entries.append(
                _comparison_entry_summary(
                    label,
                    checkpoint,
                    entry_output_dir,
                    result["metrics_rows"],
                )
            )
            print(f"S5 probe comparison entry complete: label={label} samples={len(result['metrics_rows'])}")
        payload = {
            "diagnostic_scope": "S5-family final-target probe checkpoint comparison; no training",
            "subset": str(args.subset),
            "provenance": str(args.provenance),
            "output_dir": str(args.output_dir),
            "entries": comparison_entries,
            "P10_gap_note": "localized top contact and side asymmetry unsupported",
        }
        comparison_json = args.comparison_output_json or args.output_dir / "s5_family_final_probe_comparison.json"
        comparison_md = args.comparison_output_md or args.output_dir / "s5_family_final_probe_comparison.md"
        _write_json(comparison_json, payload)
        write_text(comparison_md, _render_comparison_markdown(payload))
        print(f"S5 probe comparison complete: entries={len(comparison_entries)} output_dir={args.output_dir}")
        print(f"comparison_json={comparison_json}")
        print(f"comparison_md={comparison_md}")
        return 0

    if args.checkpoint is None or args.run_config is None:
        raise ValueError("--checkpoint and --run-config are required unless --checkpoint-entry is used")
    result = _run_single_probe(
        subset=args.subset,
        checkpoint=args.checkpoint,
        run_config_path=args.run_config,
        provenance_path=args.provenance,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
    )
    print(f"S5 probe smoke complete: samples={len(result['metrics_rows'])} output_dir={args.output_dir}")
    print("worst_3_by_RMSE:")
    for row in sorted(result["metrics_rows"], key=lambda item: item["RMSE"], reverse=True)[:3]:
        print(
            f"  {row['probe_id']} {row['sample_id']} RMSE={row['RMSE']:.6g} "
            f"relRMSE_DeltaT={row['relative_RMSE_on_DeltaT']:.6g}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
