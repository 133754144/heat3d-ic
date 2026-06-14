#!/usr/bin/env python3
"""Generate clean schematic figures for the Heat3D v3 final-target probes.

This script is visualization-only. It reads existing probe arrays, an existing
prediction npz, metrics JSON, and provenance JSON. It does not train, solve, or
regenerate data.
"""

from __future__ import annotations

import argparse
from collections import deque
import json
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection
import numpy as np


EPS = 1.0e-12
VISUAL_Z_SCALE = 2.8
MAX_COMPONENTS_PER_CLASS = 6
MIN_COMPONENT_VOXELS = 2

COLOR_DOMAIN = "#cfcfcf"
COLOR_HIGH_K = "#1f77b4"
COLOR_LOW_K = "#ffbf00"
COLOR_STRONG_Q = "#d62728"
COLOR_WEAK_Q = "#d62728"
COLOR_ANISO = "#7e3fb2"
COLOR_TOP_H = "#2ca02c"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate schematic Heat3D v3 final-probe structure figures."
    )
    parser.add_argument("--subset", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--provenance", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--visual-z-scale", type=float, default=VISUAL_Z_SCALE)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def as_column(array: np.ndarray) -> np.ndarray:
    arr = np.asarray(array, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr[:, None]
    if arr.ndim != 2 or arr.shape[1] != 1:
        raise ValueError(f"expected shape (N,1), found {arr.shape}")
    return arr


def sample_root(subset: Path) -> Path:
    subset = Path(subset)
    return subset / "samples" if (subset / "samples").is_dir() else subset


def load_metrics(path: Path) -> dict[str, dict[str, Any]]:
    payload = load_json(path)
    rows = payload.get("metrics")
    if not isinstance(rows, list):
        raise ValueError(f"{path}: expected metrics list")
    result = {}
    for row in rows:
        sample_id = str(row.get("sample_id"))
        probe_id = str(row.get("probe_id"))
        result[sample_id] = row
        result[probe_id] = row
    return result


def probe_id_from_meta(meta: dict[str, Any], fallback: str) -> str:
    if meta.get("probe_id"):
        return str(meta["probe_id"])
    for token in str(fallback).split("_"):
        if token.startswith("P") and token[1:].isdigit():
            return token
    return str(fallback)


def discover_samples(subset: Path) -> list[dict[str, Any]]:
    root = sample_root(subset)
    samples = []
    for sample_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        meta = load_json(sample_dir / "sample_meta.json")
        sample_id = str(meta.get("sample_id") or sample_dir.name)
        probe_id = probe_id_from_meta(meta, sample_id)
        samples.append(
            {
                "sample_dir": sample_dir,
                "sample_id": sample_id,
                "probe_id": probe_id,
                "meta": meta,
            }
        )
    samples.sort(key=lambda item: item["probe_id"])
    if len(samples) != 10:
        raise ValueError(f"expected 10 probe samples, found {len(samples)} under {root}")
    return samples


def grid_indices(coords: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    coords = np.asarray(coords, dtype=np.float64)
    xs = np.unique(coords[:, 0])
    ys = np.unique(coords[:, 1])
    zs = np.unique(coords[:, 2])
    ix = np.searchsorted(xs, coords[:, 0])
    iy = np.searchsorted(ys, coords[:, 1])
    iz = np.searchsorted(zs, coords[:, 2])
    return xs, ys, zs, (ix, iy, iz)


def values_to_grid(values: np.ndarray, coords: np.ndarray) -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    xs, ys, zs, (ix, iy, iz) = grid_indices(coords)
    grid = np.full((len(xs), len(ys), len(zs)), np.nan, dtype=np.float64)
    grid[ix, iy, iz] = np.asarray(values, dtype=np.float64).reshape(-1)
    return grid, (xs, ys, zs)


def mask_to_grid(mask: np.ndarray, coords: np.ndarray) -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    grid, axes = values_to_grid(np.asarray(mask, dtype=np.float64), coords)
    return np.asarray(grid > 0.5, dtype=bool), axes


def effective_k(k_field: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]:
    k = np.asarray(k_field, dtype=np.float64)
    if k.ndim == 1:
        k = k[:, None]
    if k.shape[1] == 1:
        return k[:, 0], None
    if k.shape[1] >= 3:
        k3 = k[:, :3]
        ratio = np.max(k3, axis=1) / np.maximum(np.min(k3, axis=1), EPS)
        return np.mean(k3, axis=1), ratio
    raise ValueError(f"unsupported k_field shape {k.shape}")


def rounded_mode(values: np.ndarray, decimals: int = 6) -> float:
    rounded = np.round(np.asarray(values, dtype=np.float64).reshape(-1), decimals=decimals)
    unique, counts = np.unique(rounded, return_counts=True)
    if unique.size == 0:
        raise ValueError("cannot compute mode of empty array")
    return float(unique[int(np.argmax(counts))])


def material_masks(k_eff: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    background_k = rounded_mode(k_eff)
    high = np.asarray(k_eff > background_k * 1.25, dtype=bool)
    low = np.asarray(k_eff < background_k * 0.75, dtype=bool)
    rounded = np.round(k_eff, decimals=6)
    if int(np.sum(high)) < MIN_COMPONENT_VOXELS:
        high_values = [value for value in np.unique(rounded) if value > background_k]
        high = np.isin(rounded, high_values) if high_values else high
    if int(np.sum(low)) < MIN_COMPONENT_VOXELS:
        low_values = [value for value in np.unique(rounded) if value < background_k]
        low = np.isin(rounded, low_values) if low_values else low
    return high, low, background_k


def q_masks(q_field: np.ndarray) -> tuple[np.ndarray, np.ndarray, float | None, float | None]:
    q = np.asarray(q_field, dtype=np.float64).reshape(-1)
    positive = q[q > 0.0]
    if positive.size == 0:
        empty = np.zeros_like(q, dtype=bool)
        return empty, empty, None, None
    positive_fraction = float(np.mean(q > 0.0))
    if positive_fraction > 0.90:
        rounded_positive = np.round(positive, decimals=6)
        unique, counts = np.unique(rounded_positive, return_counts=True)
        positive_mode = float(unique[int(np.argmax(counts))])
        q_background = min(float(np.min(positive)), positive_mode)
    else:
        q_background = 0.0
    q_max = float(np.max(q))
    if q_max <= q_background:
        threshold = q_background
    else:
        threshold = q_background + 0.2 * (q_max - q_background)
    strong = q > threshold
    weak = (q > 0.0) & ~strong
    return strong, weak, q_background, float(threshold)


def connected_components(mask_grid: np.ndarray) -> list[dict[str, Any]]:
    mask = np.asarray(mask_grid, dtype=bool)
    visited = np.zeros(mask.shape, dtype=bool)
    components: list[dict[str, Any]] = []
    shape = mask.shape
    neighbors = ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1))
    starts = np.argwhere(mask)
    for start in starts:
        start_tuple = tuple(int(v) for v in start)
        if visited[start_tuple]:
            continue
        queue: deque[tuple[int, int, int]] = deque([start_tuple])
        visited[start_tuple] = True
        voxels = []
        while queue:
            voxel = queue.popleft()
            voxels.append(voxel)
            for dx, dy, dz in neighbors:
                nxt = (voxel[0] + dx, voxel[1] + dy, voxel[2] + dz)
                if any(nxt[axis] < 0 or nxt[axis] >= shape[axis] for axis in range(3)):
                    continue
                if visited[nxt] or not mask[nxt]:
                    continue
                visited[nxt] = True
                queue.append(nxt)
        if len(voxels) < MIN_COMPONENT_VOXELS:
            continue
        arr = np.asarray(voxels, dtype=int)
        lo = np.min(arr, axis=0)
        hi = np.max(arr, axis=0)
        components.append(
            {
                "count": int(len(voxels)),
                "lo_index": [int(v) for v in lo],
                "hi_index": [int(v) for v in hi],
            }
        )
    components.sort(key=lambda item: int(item["count"]), reverse=True)
    return components[:MAX_COMPONENTS_PER_CLASS]


def component_bbox(component: dict[str, Any], shape: tuple[int, int, int], z_scale: float) -> tuple[np.ndarray, np.ndarray]:
    lo_idx = np.asarray(component["lo_index"], dtype=np.float64)
    hi_idx = np.asarray(component["hi_index"], dtype=np.float64) + 1.0
    denom = np.asarray(shape, dtype=np.float64)
    lo = lo_idx / denom
    hi = hi_idx / denom
    lo[2] *= z_scale
    hi[2] *= z_scale
    return lo, hi


def box_faces(lo: np.ndarray, hi: np.ndarray) -> list[list[tuple[float, float, float]]]:
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


def add_cuboid(ax, lo: np.ndarray, hi: np.ndarray, color: str, alpha: float, edge_alpha: float = 0.42) -> None:
    faces = box_faces(lo, hi)
    poly = Poly3DCollection(
        faces,
        facecolors=color,
        edgecolors=(0, 0, 0, edge_alpha),
        linewidths=0.45,
        alpha=alpha,
    )
    ax.add_collection3d(poly)


def add_domain_outline(ax, z_scale: float) -> None:
    corners = np.asarray(
        [
            [0, 0, 0],
            [1, 0, 0],
            [1, 1, 0],
            [0, 1, 0],
            [0, 0, z_scale],
            [1, 0, z_scale],
            [1, 1, z_scale],
            [0, 1, z_scale],
        ],
        dtype=np.float64,
    )
    edges = [
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 0),
        (4, 5),
        (5, 6),
        (6, 7),
        (7, 4),
        (0, 4),
        (1, 5),
        (2, 6),
        (3, 7),
    ]
    lines = [[corners[a], corners[b]] for a, b in edges]
    ax.add_collection3d(Line3DCollection(lines, colors=COLOR_DOMAIN, linewidths=0.8, alpha=0.70))


def add_top_boundary(ax, z_scale: float) -> None:
    lo = np.asarray([0.0, 0.0, z_scale * 0.985])
    hi = np.asarray([1.0, 1.0, z_scale])
    add_cuboid(ax, lo, hi, COLOR_TOP_H, 0.36, edge_alpha=0.55)
    ax.text(0.05, 0.08, z_scale * 1.02, "top Robin\nvery high h", color="darkgreen", fontsize=7)


def grid_masks_for_sample(sample_dir: Path) -> dict[str, Any]:
    coords = np.asarray(np.load(sample_dir / "coords.npy"), dtype=np.float64)
    k_field = np.asarray(np.load(sample_dir / "k_field.npy"), dtype=np.float64)
    q_field = as_column(np.load(sample_dir / "q_field.npy")).reshape(-1)
    k_eff, anisotropy_ratio = effective_k(k_field)
    high_k_mask, low_k_mask, background_k = material_masks(k_eff)
    strong_q_mask, weak_q_mask, q_background, q_threshold = q_masks(q_field)
    high_grid, axes = mask_to_grid(high_k_mask, coords)
    low_grid, _axes = mask_to_grid(low_k_mask, coords)
    strong_q_grid, _axes = mask_to_grid(strong_q_mask, coords)
    weak_q_grid, _axes = mask_to_grid(weak_q_mask, coords)
    aniso_grid = None
    if anisotropy_ratio is not None:
        aniso_grid, _axes = mask_to_grid(anisotropy_ratio > 1.5, coords)
    shape = high_grid.shape
    return {
        "coords": coords,
        "k_field": k_field,
        "q_field": q_field,
        "k_eff": k_eff,
        "anisotropy_ratio": anisotropy_ratio,
        "axes": axes,
        "grid_shape": shape,
        "background_k": background_k,
        "q_background": q_background,
        "q_strong_threshold": q_threshold,
        "components": {
            "high_k": connected_components(high_grid),
            "low_k": connected_components(low_grid),
            "strong_q": connected_components(strong_q_grid),
            "weak_q": connected_components(weak_q_grid)[:1],
            "anisotropic_k": connected_components(aniso_grid) if aniso_grid is not None else [],
        },
    }


def setup_3d_axis(ax, z_scale: float) -> None:
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_zlim(0, z_scale)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.set_facecolor("white")
    ax.grid(False)
    ax.view_init(elev=24, azim=-55)
    ax.set_box_aspect((1.0, 1.0, z_scale * 0.38))
    ax.set_axis_off()


def plot_schematic(
    ax,
    sample_dir: Path,
    row: dict[str, Any],
    *,
    z_scale: float,
    show_text: bool,
) -> dict[str, Any]:
    meta = load_json(sample_dir / "sample_meta.json")
    masks = grid_masks_for_sample(sample_dir)
    shape = masks["grid_shape"]
    add_domain_outline(ax, z_scale)
    for comp in masks["components"]["weak_q"]:
        lo, hi = component_bbox(comp, shape, z_scale)
        add_cuboid(ax, lo, hi, COLOR_WEAK_Q, 0.045, edge_alpha=0.08)
    for comp in masks["components"]["high_k"]:
        lo, hi = component_bbox(comp, shape, z_scale)
        add_cuboid(ax, lo, hi, COLOR_HIGH_K, 0.42)
    for comp in masks["components"]["low_k"]:
        lo, hi = component_bbox(comp, shape, z_scale)
        add_cuboid(ax, lo, hi, COLOR_LOW_K, 0.48)
    for comp in masks["components"]["anisotropic_k"]:
        lo, hi = component_bbox(comp, shape, z_scale)
        add_cuboid(ax, lo, hi, COLOR_ANISO, 0.48)
    for comp in masks["components"]["strong_q"]:
        lo, hi = component_bbox(comp, shape, z_scale)
        add_cuboid(ax, lo, hi, COLOR_STRONG_Q, 0.58)
    if str(row.get("probe_id")) == "P10":
        add_top_boundary(ax, z_scale)
    setup_3d_axis(ax, z_scale)

    if show_text:
        text = [
            f"{row.get('probe_id')} / {row.get('k_region_mode')}",
            f"q={row.get('source_category')}",
            f"bc={row.get('bc_category')}",
            f"RMSE={float(row.get('RMSE', math.nan)):.3g}",
            f"Tmax_err={float(row.get('Tmax_error', math.nan)):.3g}",
            f"top5={float(row.get('top_5_percent_RMSE', math.nan)):.3g}",
        ]
        if row.get("probe_id") == "P09" and masks["anisotropy_ratio"] is not None:
            text.append(f"diag3 max aniso={float(np.max(masks['anisotropy_ratio'])):.3g}")
        if row.get("probe_id") == "P10":
            text.append("localized contact unsupported")
            text.append("side asymmetry unsupported")
        ax.text2D(0.02, 0.98, "\n".join(text), transform=ax.transAxes, va="top", fontsize=8)

    return {
        "grid_shape": [int(v) for v in shape],
        "background_k": float(masks["background_k"]),
        "q_background": masks["q_background"],
        "q_strong_threshold": masks["q_strong_threshold"],
        "component_counts": {key: len(value) for key, value in masks["components"].items()},
        "components": masks["components"],
    }


def short_title(row: dict[str, Any]) -> str:
    probe = row.get("probe_id")
    k_region = str(row.get("k_region_mode") or "")
    source = str(row.get("source_category") or "")
    if probe == "P10":
        first = "global very-high top-h"
    elif probe == "P09":
        first = "diag3 anisotropic patch"
    elif "bridge" in k_region:
        first = "sparse high-k bridge"
    elif "barrier" in k_region:
        first = "low-k barrier"
    elif "tsv" in k_region:
        first = "TSV-like high-k path"
    elif "interface" in k_region:
        first = "multi-scale interface"
    elif "background" in k_region:
        first = "random-block background"
    else:
        first = k_region.replace("_", " ")
    return f"{probe} {first}\nq={source} | RMSE={float(row.get('RMSE', math.nan)):.3g}"


def legend_handles() -> list[mpatches.Patch]:
    return [
        mpatches.Patch(color=COLOR_HIGH_K, label="high-k"),
        mpatches.Patch(color=COLOR_LOW_K, label="low-k"),
        mpatches.Patch(color=COLOR_STRONG_Q, label="strong q"),
        mpatches.Patch(color=COLOR_ANISO, label="anisotropic-k"),
        mpatches.Patch(color=COLOR_TOP_H, label="top high-h boundary"),
    ]


def make_structure_overview(
    samples: list[dict[str, Any]],
    metrics_by_id: dict[str, dict[str, Any]],
    output_path: Path,
    z_scale: float,
) -> dict[str, Any]:
    fig = plt.figure(figsize=(20, 10), constrained_layout=True)
    component_summary: dict[str, Any] = {}
    for index, sample in enumerate(samples, start=1):
        row = metrics_by_id.get(sample["sample_id"]) or metrics_by_id[sample["probe_id"]]
        ax = fig.add_subplot(2, 5, index, projection="3d")
        summary = plot_schematic(ax, sample["sample_dir"], row, z_scale=z_scale, show_text=False)
        component_summary[str(row["probe_id"])] = summary
        ax.set_title(short_title(row), fontsize=9, pad=0)
    fig.suptitle("Heat3D v3 final-target probe v0 - true structure schematics", fontsize=15)
    fig.legend(handles=legend_handles(), loc="lower center", ncol=5, frameon=True, fontsize=10)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return component_summary


def make_single_structure_figures(
    samples: list[dict[str, Any]],
    metrics_by_id: dict[str, dict[str, Any]],
    output_dir: Path,
    z_scale: float,
) -> list[Path]:
    paths = []
    for sample in samples:
        row = metrics_by_id.get(sample["sample_id"]) or metrics_by_id[sample["probe_id"]]
        fig = plt.figure(figsize=(7.2, 6.2), constrained_layout=True)
        ax = fig.add_subplot(1, 1, 1, projection="3d")
        plot_schematic(ax, sample["sample_dir"], row, z_scale=z_scale, show_text=True)
        fig.suptitle(short_title(row), fontsize=11)
        path = output_dir / f"{row['probe_id']}_structure_schematic.png"
        fig.savefig(path, dpi=190)
        plt.close(fig)
        paths.append(path)
    return paths


def slice2d(values: np.ndarray, coords: np.ndarray, z_index: int) -> np.ndarray:
    grid, _axes = values_to_grid(values, coords)
    return grid[:, :, z_index].T


def source_slice_index(coords: np.ndarray, q_field: np.ndarray) -> int:
    grid, _axes = values_to_grid(q_field, coords)
    by_z = np.nansum(grid, axis=(0, 1))
    return int(np.argmax(by_z))


def make_prediction_overview(
    samples: list[dict[str, Any]],
    metrics_by_id: dict[str, dict[str, Any]],
    predictions: dict[str, np.ndarray],
    output_path: Path,
) -> None:
    abs_error_max = 0.0
    prepared = []
    for sample in samples:
        sample_dir = sample["sample_dir"]
        sample_id = sample["sample_id"]
        coords = np.asarray(np.load(sample_dir / "coords.npy"), dtype=np.float64)
        q_field = as_column(np.load(sample_dir / "q_field.npy")).reshape(-1)
        label = as_column(np.load(sample_dir / "temperature.npy")).reshape(-1)
        pred = as_column(predictions[sample_id]).reshape(-1)
        error = np.abs(pred - label)
        z_idx = source_slice_index(coords, q_field)
        vmin = float(min(np.min(label), np.min(pred)))
        vmax = float(max(np.max(label), np.max(pred)))
        abs_error_max = max(abs_error_max, float(np.max(error)))
        prepared.append((sample, coords, z_idx, label, pred, error, vmin, vmax))

    fig, axes = plt.subplots(10, 3, figsize=(8.8, 20), constrained_layout=True)
    for row_idx, (sample, coords, z_idx, label, pred, error, vmin, vmax) in enumerate(prepared):
        metric = metrics_by_id.get(sample["sample_id"]) or metrics_by_id[sample["probe_id"]]
        titles = (
            f"{sample['probe_id']} label",
            "pred",
            "abs error",
        )
        arrays = (
            slice2d(label, coords, z_idx),
            slice2d(pred, coords, z_idx),
            slice2d(error, coords, z_idx),
        )
        cmaps = ("magma", "magma", "Reds")
        for col_idx, (title, arr, cmap) in enumerate(zip(titles, arrays, cmaps)):
            ax = axes[row_idx, col_idx]
            if col_idx < 2:
                im = ax.imshow(arr, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
            else:
                im = ax.imshow(arr, origin="lower", cmap=cmap, vmin=0.0, vmax=abs_error_max, aspect="auto")
            ax.set_xticks([])
            ax.set_yticks([])
            if col_idx == 0:
                ax.set_ylabel(f"RMSE {float(metric.get('RMSE', math.nan)):.2g}", fontsize=8)
            ax.set_title(title, fontsize=8)
            if row_idx in (0, 9):
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    fig.suptitle("S5 final-probe compact prediction overview - source slices", fontsize=12)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=170)
    plt.close(fig)


def make_metric_summary(
    samples: list[dict[str, Any]],
    metrics_by_id: dict[str, dict[str, Any]],
    output_path: Path,
) -> None:
    rows = [metrics_by_id.get(sample["sample_id"]) or metrics_by_id[sample["probe_id"]] for sample in samples]
    probes = [str(row["probe_id"]) for row in rows]
    worst = {"P03", "P02", "P09"}
    colors = ["#d62728" if probe in worst else "#8fb5d9" for probe in probes]
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 8.5), constrained_layout=True)
    fields = [
        ("RMSE", "RMSE"),
        ("top_5_percent_RMSE", "top-5% RMSE"),
        ("relative_RMSE_on_DeltaT", "relative RMSE on DeltaT"),
    ]
    for ax, (field, label) in zip(axes.reshape(-1)[:3], fields):
        values = [float(row[field]) for row in rows]
        ax.bar(probes, values, color=colors)
        ax.set_title(label)
        ax.grid(axis="y", alpha=0.25)
        ax.tick_params(axis="x", rotation=0)
    ax = axes.reshape(-1)[3]
    values = [float(row["Tmax_error"]) for row in rows]
    ax.barh(probes, values, color=colors)
    ax.axvline(0.0, color="black", linewidth=0.8)
    ax.set_title("Tmax error")
    ax.grid(axis="x", alpha=0.25)
    fig.suptitle("S5 final-probe schematic metrics summary (worst 3 highlighted)", fontsize=12)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def make_report(
    output_dir: Path,
    provenance: dict[str, Any],
    predictions_path: Path,
    metrics_path: Path,
    figure_paths: list[Path],
) -> None:
    lines = [
        "# Heat3D v3 Final-Probe Schematic Visualization Report",
        "",
        "This report covers schematic figures for human structure review. It does not replace numeric metrics.",
        "",
        "## Inputs",
        "",
        f"- data source: `{provenance.get('source')}`",
        f"- copied_from_local_hf_upload_staging: `{provenance.get('copied_from_local_hf_upload_staging')}`",
        f"- used_local_regeneration: `{provenance.get('used_local_regeneration')}`",
        f"- used_generator_this_round: `{provenance.get('used_generator_this_round')}`",
        f"- sha256_identity_check: `{provenance.get('sha256_identity_check')}`",
        f"- predictions: `{predictions_path}`",
        f"- metrics: `{metrics_path}`",
        "",
        "## Method",
        "",
        "- Reconstructs the regular grid from `coords.npy`.",
        "- Estimates background k as the rounded mode of effective k.",
        "- Uses high-k threshold `k_eff > background_k * 1.25` and low-k threshold `k_eff < background_k * 0.75`, with unique-value fallback.",
        "- Runs 6-connected components on high-k, low-k, strong-q, and anisotropic masks.",
        "- Displays up to 6 largest components per class and ignores components smaller than 2 voxels.",
        "- Uses strong-q threshold `q_background + 0.2 * (q_max - q_background)`.",
        "- P09 displays `diag3` anisotropic patch where anisotropy ratio exceeds 1.5; it is not labeled as full tensor-k.",
        "- P10 displays a global top Robin high-h boundary and preserves localized-contact / side-asymmetry as unsupported.",
        "",
        "## Figures",
        "",
    ]
    for path in figure_paths:
        lines.append(f"- `{path}`")
    write_text(output_dir.parent / "schematic_visualization_report.md", "\n".join(lines) + "\n")


def main() -> int:
    args = parse_args()
    if not args.predictions.is_file():
        raise FileNotFoundError(f"predictions not found: {args.predictions}")
    if not args.metrics.is_file():
        raise FileNotFoundError(f"metrics not found: {args.metrics}")
    if not args.provenance.is_file():
        raise FileNotFoundError(f"provenance not found: {args.provenance}")
    provenance = load_json(args.provenance)
    if provenance.get("used_local_regeneration") is not False:
        raise ValueError("provenance must state used_local_regeneration=false")
    if provenance.get("used_generator_this_round") is not False:
        raise ValueError("provenance must state used_generator_this_round=false")
    if provenance.get("sha256_identity_check") != "pass":
        raise ValueError("provenance must state sha256_identity_check=pass")

    samples = discover_samples(args.subset)
    metrics_by_id = load_metrics(args.metrics)
    with np.load(args.predictions) as archive:
        predictions = {key: np.asarray(archive[key], dtype=np.float64) for key in archive.files}
    missing_predictions = [sample["sample_id"] for sample in samples if sample["sample_id"] not in predictions]
    if missing_predictions:
        raise FileNotFoundError(f"predictions missing sample ids: {missing_predictions}")

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    overview = output_dir / "probe_structure_schematic_overview.png"
    pred_overview = output_dir / "probe_prediction_compact_overview.png"
    metric_summary = output_dir / "probe_metric_summary_clean.png"
    component_summary = make_structure_overview(samples, metrics_by_id, overview, args.visual_z_scale)
    single_paths = make_single_structure_figures(samples, metrics_by_id, output_dir, args.visual_z_scale)
    make_prediction_overview(samples, metrics_by_id, predictions, pred_overview)
    make_metric_summary(samples, metrics_by_id, metric_summary)
    figure_paths = [overview, pred_overview, metric_summary] + single_paths

    manifest = {
        "diagnostic_scope": "schematic visualization only; no training; no inference rerun; no data generation",
        "subset": str(args.subset),
        "predictions": str(args.predictions),
        "metrics": str(args.metrics),
        "provenance": str(args.provenance),
        "output_dir": str(output_dir),
        "visual_z_scale": float(args.visual_z_scale),
        "figure_count": len(figure_paths),
        "figures": [str(path) for path in figure_paths],
        "component_summary": component_summary,
        "data_source": "local_hf_upload_staging_copy",
        "used_local_regeneration": False,
        "used_generator_this_round": False,
        "reran_s5_inference": False,
    }
    manifest_path = output_dir / "schematic_figure_manifest.json"
    write_json(manifest_path, manifest)
    make_report(output_dir, provenance, args.predictions, args.metrics, figure_paths)
    print(f"schematic visualization complete: figures={len(figure_paths)} output_dir={output_dir}")
    print(f"manifest={manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
