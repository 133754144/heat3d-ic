#!/usr/bin/env python3
"""Coordinate-level structure audit for Heat3D v3 final-target probes."""

from __future__ import annotations

import argparse
import csv
from collections import deque
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np


REPO_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = Path(__file__).resolve().parent
for path in (REPO_DIR, SCRIPTS_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from visualize_heat3d_v3_final_probe_schematic import (  # noqa: E402
    EPS,
    as_column,
    discover_samples,
    effective_k,
    grid_indices,
    load_json,
    load_metrics,
    material_masks,
    q_masks,
    write_json,
    write_text,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit v3 final probe structure coordinates.")
    parser.add_argument("--subset", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--provenance", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def safe_float(value: Any) -> float | None:
    if value is None:
        return None
    value = float(value)
    if not math.isfinite(value):
        return None
    return value


def point_index_grid(coords: np.ndarray) -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    xs, ys, zs, (ix, iy, iz) = grid_indices(coords)
    grid = np.full((len(xs), len(ys), len(zs)), -1, dtype=int)
    grid[ix, iy, iz] = np.arange(coords.shape[0], dtype=int)
    return grid, (xs, ys, zs)


def mask_grid(mask: np.ndarray, coords: np.ndarray) -> tuple[np.ndarray, np.ndarray, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    index_grid, axes = point_index_grid(coords)
    result = np.zeros(index_grid.shape, dtype=bool)
    flat_mask = np.asarray(mask, dtype=bool).reshape(-1)
    for idx in np.flatnonzero(flat_mask):
        loc = np.argwhere(index_grid == idx)
        if loc.size:
            result[tuple(int(v) for v in loc[0])] = True
    return result, index_grid, axes


def components_with_points(mask: np.ndarray, coords: np.ndarray) -> list[dict[str, Any]]:
    grid, index_grid, _axes = mask_grid(mask, coords)
    visited = np.zeros(grid.shape, dtype=bool)
    shape = grid.shape
    neighbors = ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1))
    components: list[dict[str, Any]] = []
    for start in np.argwhere(grid):
        start_tuple = tuple(int(v) for v in start)
        if visited[start_tuple]:
            continue
        queue: deque[tuple[int, int, int]] = deque([start_tuple])
        visited[start_tuple] = True
        voxels = []
        point_indices = []
        while queue:
            voxel = queue.popleft()
            voxels.append(voxel)
            point_idx = int(index_grid[voxel])
            if point_idx >= 0:
                point_indices.append(point_idx)
            for dx, dy, dz in neighbors:
                nxt = (voxel[0] + dx, voxel[1] + dy, voxel[2] + dz)
                if any(nxt[axis] < 0 or nxt[axis] >= shape[axis] for axis in range(3)):
                    continue
                if visited[nxt] or not grid[nxt]:
                    continue
                visited[nxt] = True
                queue.append(nxt)
        if len(point_indices) < 2:
            continue
        voxel_arr = np.asarray(voxels, dtype=int)
        components.append(
            {
                "point_indices": sorted(int(idx) for idx in point_indices),
                "lo_index": [int(v) for v in np.min(voxel_arr, axis=0)],
                "hi_index": [int(v) for v in np.max(voxel_arr, axis=0)],
            }
        )
    components.sort(key=lambda item: len(item["point_indices"]), reverse=True)
    return components


def bbox_payload(coords: np.ndarray, indices: list[int]) -> dict[str, Any]:
    pts = coords[np.asarray(indices, dtype=int)]
    lo = np.min(pts, axis=0)
    hi = np.max(pts, axis=0)
    center = 0.5 * (lo + hi)
    size = hi - lo
    return {
        "bbox_min_m": [float(v) for v in lo],
        "bbox_max_m": [float(v) for v in hi],
        "bbox_center_m": [float(v) for v in center],
        "bbox_size_m": [float(v) for v in size],
    }


def bbox_distance(a: dict[str, Any] | None, b: dict[str, Any] | None) -> float | None:
    if not a or not b:
        return None
    a_min = np.asarray(a["bbox_min_m"], dtype=np.float64)
    a_max = np.asarray(a["bbox_max_m"], dtype=np.float64)
    b_min = np.asarray(b["bbox_min_m"], dtype=np.float64)
    b_max = np.asarray(b["bbox_max_m"], dtype=np.float64)
    gap = np.maximum(0.0, np.maximum(a_min - b_max, b_min - a_max))
    return float(np.linalg.norm(gap))


def bbox_intersects(a: dict[str, Any] | None, b: dict[str, Any] | None) -> bool:
    if not a or not b:
        return False
    a_min = np.asarray(a["bbox_min_m"], dtype=np.float64)
    a_max = np.asarray(a["bbox_max_m"], dtype=np.float64)
    b_min = np.asarray(b["bbox_min_m"], dtype=np.float64)
    b_max = np.asarray(b["bbox_max_m"], dtype=np.float64)
    return bool(np.all(a_min <= b_max + EPS) and np.all(b_min <= a_max + EPS))


def bbox_contains(outer: dict[str, Any] | None, inner: dict[str, Any] | None) -> bool:
    if not outer or not inner:
        return False
    outer_min = np.asarray(outer["bbox_min_m"], dtype=np.float64)
    outer_max = np.asarray(outer["bbox_max_m"], dtype=np.float64)
    inner_min = np.asarray(inner["bbox_min_m"], dtype=np.float64)
    inner_max = np.asarray(inner["bbox_max_m"], dtype=np.float64)
    return bool(np.all(outer_min <= inner_min + EPS) and np.all(inner_max <= outer_max + EPS))


def component_record(
    probe_id: str,
    kind: str,
    ordinal: int,
    coords: np.ndarray,
    component: dict[str, Any],
    values: np.ndarray,
    value_prefix: str,
    total_points: int,
) -> dict[str, Any]:
    indices = component["point_indices"]
    vals = np.asarray(values, dtype=np.float64).reshape(-1)[np.asarray(indices, dtype=int)]
    record = {
        "component_id": f"{probe_id}_{kind}_{ordinal:02d}",
        "probe_id": probe_id,
        "kind": kind,
        "point_count": int(len(indices)),
        "fraction": float(len(indices) / max(total_points, 1)),
        "lo_index": component["lo_index"],
        "hi_index": component["hi_index"],
    }
    record.update(bbox_payload(coords, indices))
    record[f"{value_prefix}_min"] = float(np.min(vals))
    record[f"{value_prefix}_max"] = float(np.max(vals))
    record[f"{value_prefix}_mean"] = float(np.mean(vals))
    return record


def nearest_distance_to(records: list[dict[str, Any]], targets: list[dict[str, Any]]) -> float | None:
    distances = [
        bbox_distance(record, target)
        for record in records
        for target in targets
    ]
    distances = [distance for distance in distances if distance is not None]
    return min(distances) if distances else None


def argmax_coord(coords: np.ndarray, values: np.ndarray) -> list[float]:
    idx = int(np.argmax(np.asarray(values).reshape(-1)))
    return [float(v) for v in coords[idx]]


def rmse(values: np.ndarray) -> float:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    return float(np.sqrt(np.mean(np.square(arr))))


def mae(values: np.ndarray) -> float:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    return float(np.mean(np.abs(arr)))


def masked_rmse(values: np.ndarray, mask: np.ndarray) -> float | None:
    mask = np.asarray(mask, dtype=bool).reshape(-1)
    if not np.any(mask):
        return None
    return rmse(np.asarray(values).reshape(-1)[mask])


def top_fraction_rmse(error: np.ndarray, score: np.ndarray, fraction: float) -> float:
    score = np.asarray(score, dtype=np.float64).reshape(-1)
    error = np.asarray(error, dtype=np.float64).reshape(-1)
    count = max(1, int(math.ceil(score.size * fraction)))
    indices = np.argsort(score)[-count:]
    return rmse(error[indices])


def unique_rounded(values: np.ndarray, max_items: int = 64) -> list[float]:
    vals = [float(v) for v in np.unique(np.round(values, decimals=6))]
    return vals[:max_items]


def source_relationships(
    high_components: list[dict[str, Any]],
    low_components: list[dict[str, Any]],
    strong_components: list[dict[str, Any]],
) -> dict[str, Any]:
    source = strong_components[0] if strong_components else None
    nearest_high = nearest_distance_to(high_components, strong_components)
    nearest_low = nearest_distance_to(low_components, strong_components)
    source_inside_low = any(bbox_contains(low, source) for low in low_components)
    source_intersects_low = any(bbox_intersects(low, source) for low in low_components)
    source_intersects_high = any(bbox_intersects(high, source) for high in high_components)
    return {
        "nearest_high_k_to_source_distance_m": safe_float(nearest_high),
        "nearest_low_k_to_source_distance_m": safe_float(nearest_low),
        "source_inside_low_k_bbox": bool(source_inside_low),
        "source_intersects_low_k_bbox": bool(source_intersects_low),
        "source_intersects_high_k_bbox": bool(source_intersects_high),
    }


def p07_tsv_summary(high_components: list[dict[str, Any]], strong_components: list[dict[str, Any]], domain_z_span: float) -> dict[str, Any]:
    if not high_components or domain_z_span <= 0.0:
        return {}
    best = max(high_components, key=lambda item: float(item["bbox_size_m"][2]))
    z_span_fraction = float(best["bbox_size_m"][2] / domain_z_span)
    return {
        "tsv_like_component_id": best["component_id"],
        "tsv_like_component_bbox": {
            "bbox_min_m": best["bbox_min_m"],
            "bbox_max_m": best["bbox_max_m"],
        },
        "z_span_fraction": z_span_fraction,
        "source_to_tsv_distance_m": safe_float(nearest_distance_to([best], strong_components)),
        "vertical_high_k_path_detected": bool(z_span_fraction >= 0.5),
    }


def dominant_summary(probe_id: str, sample: dict[str, Any], relationships: dict[str, Any], p07_summary: dict[str, Any], metrics: dict[str, Any]) -> str:
    source = sample["source_category"]
    k_region = sample["k_region_mode"]
    if probe_id == "P03":
        relation = "inside/intersects low-k barrier" if relationships["source_intersects_low_k_bbox"] else "near low-k barrier"
        return f"P03: contained hotspot is {relation}; S5 underestimates the confined peak."
    if probe_id == "P02":
        return "P02: sparse high-k bridge with weak full-domain background q and one compact strong source."
    if probe_id == "P06":
        return "P06: elongated strong source sits on weak full-domain background q in random-block material."
    if probe_id == "P07":
        if p07_summary.get("vertical_high_k_path_detected"):
            return "P07: vertical high-k path is detected; hotspot is adjacent to the TSV-like conduction route."
        return "P07: high-k route exists but vertical z-span is below TSV-like threshold."
    if probe_id == "P09":
        return "P09: diag3 anisotropic patch, not full tensor-k; source is patch-adjacent and peak is underestimated."
    if probe_id == "P10":
        return "P10: global top Robin very-high h only; localized top contact and side asymmetry are unsupported."
    return f"{probe_id}: {k_region} with {source}; RMSE={float(metrics.get('RMSE', math.nan)):.3g}."


def audit_sample(sample_entry: dict[str, Any], prediction: np.ndarray, metrics: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    sample_dir = sample_entry["sample_dir"]
    meta = sample_entry["meta"]
    probe_id = sample_entry["probe_id"]
    sample_id = sample_entry["sample_id"]
    coords = np.asarray(np.load(sample_dir / "coords.npy"), dtype=np.float64)
    k_field = np.asarray(np.load(sample_dir / "k_field.npy"), dtype=np.float64)
    q_field = as_column(np.load(sample_dir / "q_field.npy")).reshape(-1)
    label = as_column(np.load(sample_dir / "temperature.npy")).reshape(-1)
    pred = as_column(prediction).reshape(-1)
    error = pred - label
    abs_error = np.abs(error)
    k_eff, anisotropy_ratio = effective_k(k_field)
    high_mask, low_mask, background_k = material_masks(k_eff)
    strong_q_mask, weak_q_mask, q_background, q_threshold = q_masks(q_field)
    xs, ys, zs, _indices = grid_indices(coords)
    domain_min = np.min(coords, axis=0)
    domain_max = np.max(coords, axis=0)
    domain_z_span = float(domain_max[2] - domain_min[2])

    high_components = [
        component_record(probe_id, "high_k", idx, coords, comp, k_eff, "k_eff", coords.shape[0])
        for idx, comp in enumerate(components_with_points(high_mask, coords))
    ]
    low_components = [
        component_record(probe_id, "low_k", idx, coords, comp, k_eff, "k_eff", coords.shape[0])
        for idx, comp in enumerate(components_with_points(low_mask, coords))
    ]
    strong_components = [
        component_record(probe_id, "strong_q", idx, coords, comp, q_field, "q", coords.shape[0])
        for idx, comp in enumerate(components_with_points(strong_q_mask, coords))
    ]
    weak_components = [
        component_record(probe_id, "weak_q", idx, coords, comp, q_field, "q", coords.shape[0])
        for idx, comp in enumerate(components_with_points(weak_q_mask, coords))
    ]
    anisotropic_components: list[dict[str, Any]] = []
    if anisotropy_ratio is not None:
        anisotropic_components = [
            component_record(probe_id, "anisotropic_k", idx, coords, comp, anisotropy_ratio, "anisotropy_ratio", coords.shape[0])
            for idx, comp in enumerate(components_with_points(anisotropy_ratio > 1.5, coords))
        ]

    for record in high_components + low_components:
        record["distance_to_nearest_strong_q_bbox_m"] = safe_float(
            nearest_distance_to([record], strong_components)
        )

    relationships = source_relationships(high_components, low_components, strong_components)
    p07_summary = p07_tsv_summary(high_components, strong_components, domain_z_span) if probe_id == "P07" else {}
    t_ref = float(((meta.get("boundary_params") or {}).get("bottom") or {}).get("fixed_temperature_K", 300.0))
    delta = label - t_ref
    q_mask = q_field > 0.0
    background_mask = ~q_mask
    if not np.any(background_mask) and q_threshold is not None:
        background_mask = q_field <= q_threshold
    top = ((meta.get("boundary_params") or {}).get("top") or {})
    bottom = ((meta.get("boundary_params") or {}).get("bottom") or {})
    sample_payload: dict[str, Any] = {
        "probe_id": probe_id,
        "sample_id": sample_id,
        "sample_dir": sample_dir.name,
        "probe_family": meta.get("probe_family"),
        "intended_stressor": meta.get("intended_stressor"),
        "k_region_mode": meta.get("k_region_mode"),
        "source_category": meta.get("source_category"),
        "q_power_range": meta.get("q_power_range"),
        "bc_category": meta.get("bc_category"),
        "top_h_value": top.get("h_W_m2K"),
        "bottom_T_fixed": bottom.get("fixed_temperature_K"),
        "localized_top_contact_supported": bool(meta.get("localized_top_contact_supported", False)),
        "side_asymmetry_supported": bool(meta.get("side_asymmetry_supported", False)),
        "label_status": meta.get("label_status"),
        "implemented_bc": "V1 global top Robin very_high_top_h" if probe_id == "P10" else None,
        "sample_space": {
            "grid_shape": [int(len(xs)), int(len(ys)), int(len(zs))],
            "domain_bounds_m": {
                "x": [float(domain_min[0]), float(domain_max[0])],
                "y": [float(domain_min[1]), float(domain_max[1])],
                "z": [float(domain_min[2]), float(domain_max[2])],
            },
            "coordinate_values_m": {
                "x": [float(v) for v in xs],
                "y": [float(v) for v in ys],
                "z": [float(v) for v in zs],
            },
            "spacing_m": {
                "dx_unique": [float(v) for v in np.unique(np.round(np.diff(xs), decimals=12))],
                "dy_unique": [float(v) for v in np.unique(np.round(np.diff(ys), decimals=12))],
                "dz_unique": [float(v) for v in np.unique(np.round(np.diff(zs), decimals=12))],
            },
            "unit": "meter",
        },
        "k_space": {
            "k_field_shape": [int(v) for v in k_field.shape],
            "k_mode": meta.get("k_mode"),
            "background_k": float(background_k),
            "unique_k_eff_values": unique_rounded(k_eff),
            "k_eff_min": float(np.min(k_eff)),
            "k_eff_max": float(np.max(k_eff)),
            "k_eff_mean": float(np.mean(k_eff)),
            "high_k_threshold": float(background_k * 1.25),
            "low_k_threshold": float(background_k * 0.75),
            "high_k_component_count": len(high_components),
            "low_k_component_count": len(low_components),
            "high_k_components": high_components,
            "low_k_components": low_components,
        },
        "q_space": {
            "q_field_shape": [int(v) for v in np.asarray(np.load(sample_dir / "q_field.npy")).shape],
            "q_min": float(np.min(q_field)),
            "q_max": float(np.max(q_field)),
            "q_mean": float(np.mean(q_field)),
            "q_positive_fraction": float(np.mean(q_field > 0.0)),
            "q_nonzero_fraction": float(np.mean(q_field != 0.0)),
            "q_background_value": safe_float(q_background),
            "q_strong_threshold": safe_float(q_threshold),
            "strong_q_fraction": float(np.mean(strong_q_mask)),
            "weak_background_present": bool(np.mean(q_field > 0.0) > 0.90 and np.any(weak_q_mask)),
            "strong_q_component_count": len(strong_components),
            "weak_q_component_count": len(weak_components),
            "strong_background_ratio": safe_float(float(np.max(q_field) / max(q_background or 0.0, EPS))),
            "strong_q_components": strong_components,
            "weak_q_components": weak_components,
        },
        "temperature_prediction_error_space": {
            "T_label_min": float(np.min(label)),
            "T_label_max": float(np.max(label)),
            "T_label_mean": float(np.mean(label)),
            "T_label_std": float(np.std(label)),
            "T_pred_min": float(np.min(pred)),
            "T_pred_max": float(np.max(pred)),
            "T_pred_mean": float(np.mean(pred)),
            "T_pred_std": float(np.std(pred)),
            "abs_error_min": float(np.min(abs_error)),
            "abs_error_max": float(np.max(abs_error)),
            "abs_error_mean": float(np.mean(abs_error)),
            "T_label_argmax_coord_m": argmax_coord(coords, label),
            "T_pred_argmax_coord_m": argmax_coord(coords, pred),
            "abs_error_argmax_coord_m": argmax_coord(coords, abs_error),
            "Tmax_error": float(np.max(pred) - np.max(label)),
            "RMSE": rmse(error),
            "MAE": mae(error),
            "top_1_percent_RMSE": top_fraction_rmse(error, delta, 0.01),
            "top_5_percent_RMSE": top_fraction_rmse(error, delta, 0.05),
            "q_region_RMSE": safe_float(masked_rmse(error, q_mask)),
            "background_region_RMSE": safe_float(masked_rmse(error, background_mask)),
        },
        "structure_relationships": relationships,
    }
    if probe_id == "P07":
        sample_payload["structure_relationships"].update(p07_summary)
    if anisotropy_ratio is not None:
        sample_payload["k_space"].update(
            {
                "kx_min": float(np.min(k_field[:, 0])),
                "kx_max": float(np.max(k_field[:, 0])),
                "ky_min": float(np.min(k_field[:, 1])),
                "ky_max": float(np.max(k_field[:, 1])),
                "kz_min": float(np.min(k_field[:, 2])),
                "kz_max": float(np.max(k_field[:, 2])),
                "kx_mean": float(np.mean(k_field[:, 0])),
                "ky_mean": float(np.mean(k_field[:, 1])),
                "kz_mean": float(np.mean(k_field[:, 2])),
                "anisotropy_ratio_max": float(np.max(anisotropy_ratio)),
                "anisotropic_component_count": len(anisotropic_components),
                "anisotropic_components": anisotropic_components,
            }
        )
    sample_payload["dominant_structure_summary"] = dominant_summary(
        probe_id,
        sample_payload,
        relationships,
        p07_summary,
        metrics,
    )
    component_rows = high_components + low_components + strong_components + weak_components + anisotropic_components
    return sample_payload, component_rows


def flatten_summary(sample: dict[str, Any]) -> dict[str, Any]:
    temp = sample["temperature_prediction_error_space"]
    k_space = sample["k_space"]
    q_space = sample["q_space"]
    rel = sample["structure_relationships"]
    strong_bbox = None
    if q_space["strong_q_components"]:
        strong_bbox = {
            "min": q_space["strong_q_components"][0]["bbox_min_m"],
            "max": q_space["strong_q_components"][0]["bbox_max_m"],
        }
    return {
        "probe_id": sample["probe_id"],
        "sample_id": sample["sample_id"],
        "grid_shape": sample["sample_space"]["grid_shape"],
        "domain_bounds": sample["sample_space"]["domain_bounds_m"],
        "k_mode": k_space["k_mode"],
        "background_k": k_space["background_k"],
        "high_k_components": k_space["high_k_component_count"],
        "low_k_components": k_space["low_k_component_count"],
        "strong_q_components": q_space["strong_q_component_count"],
        "weak_background_present": q_space["weak_background_present"],
        "strong_q_bbox": strong_bbox,
        "Tmax_label_coord": temp["T_label_argmax_coord_m"],
        "Tmax_pred_coord": temp["T_pred_argmax_coord_m"],
        "max_error_coord": temp["abs_error_argmax_coord_m"],
        "RMSE": temp["RMSE"],
        "Tmax_error": temp["Tmax_error"],
        "nearest_high_k_to_source_distance_m": rel["nearest_high_k_to_source_distance_m"],
        "nearest_low_k_to_source_distance_m": rel["nearest_low_k_to_source_distance_m"],
        "dominant_structure_summary": sample["dominant_structure_summary"],
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: jsonish(row.get(key)) for key in keys})


def jsonish(value: Any) -> Any:
    if isinstance(value, (dict, list)):
        return repr(value)
    return value


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Heat3D v3 Final Probe Structure Audit",
        "",
        "## Dataset Summary",
        "",
        f"- dataset_name: `{payload['dataset_summary']['dataset_name']}`",
        f"- sample_count: `{payload['dataset_summary']['sample_count']}`",
        f"- data source: `{payload['dataset_summary']['provenance'].get('source')}`",
        f"- sha256_identity_check: `{payload['dataset_summary']['provenance'].get('sha256_identity_check')}`",
        "",
        "## Per-Probe Coordinate And Structure Table",
        "",
        "| probe | grid | k_mode | bg_k | high-k | low-k | strong-q | strong-q bbox | Tmax label coord | Tmax pred coord | max error coord | RMSE | Tmax err | summary |",
        "|---|---|---|---:|---:|---:|---:|---|---|---|---|---:|---:|---|",
    ]
    for row in payload["summary_rows"]:
        lines.append(
            "| {probe} | {grid} | {k_mode} | {bg:.4g} | {hi} | {lo} | {sq} | {bbox} | {tl} | {tp} | {me} | {rmse:.4g} | {tmax:.4g} | {summary} |".format(
                probe=row["probe_id"],
                grid=row["grid_shape"],
                k_mode=row["k_mode"],
                bg=float(row["background_k"]),
                hi=row["high_k_components"],
                lo=row["low_k_components"],
                sq=row["strong_q_components"],
                bbox=row["strong_q_bbox"],
                tl=row["Tmax_label_coord"],
                tp=row["Tmax_pred_coord"],
                me=row["max_error_coord"],
                rmse=float(row["RMSE"]),
                tmax=float(row["Tmax_error"]),
                summary=row["dominant_structure_summary"],
            )
        )
    lines.extend(
        [
            "",
            "## Per-Probe K-Space Summary",
            "",
            "| probe | k_eff min | k_eff max | high threshold | low threshold | high comps | low comps |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for sample in payload["samples"]:
        k = sample["k_space"]
        lines.append(
            f"| {sample['probe_id']} | {k['k_eff_min']:.4g} | {k['k_eff_max']:.4g} | {k['high_k_threshold']:.4g} | {k['low_k_threshold']:.4g} | {k['high_k_component_count']} | {k['low_k_component_count']} |"
        )
    lines.extend(
        [
            "",
            "## Per-Probe Q-Space Summary",
            "",
            "| probe | q max | q positive frac | bg q | strong threshold | strong comps | weak bg |",
            "|---|---:|---:|---:|---:|---:|---|",
        ]
    )
    for sample in payload["samples"]:
        q = sample["q_space"]
        lines.append(
            f"| {sample['probe_id']} | {q['q_max']:.4g} | {q['q_positive_fraction']:.4g} | {q['q_background_value']} | {q['q_strong_threshold']} | {q['strong_q_component_count']} | {q['weak_background_present']} |"
        )
    lines.extend(
        [
            "",
            "## Per-Probe T/Prediction/Error Summary",
            "",
            "| probe | label max | pred max | abs err max | RMSE | top5 RMSE | q RMSE | bg RMSE |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for sample in payload["samples"]:
        t = sample["temperature_prediction_error_space"]
        lines.append(
            f"| {sample['probe_id']} | {t['T_label_max']:.4g} | {t['T_pred_max']:.4g} | {t['abs_error_max']:.4g} | {t['RMSE']:.4g} | {t['top_5_percent_RMSE']:.4g} | {t['q_region_RMSE']} | {t['background_region_RMSE']} |"
        )
    lines.extend(["", "## Dominant Structure Summary", ""])
    for sample in payload["samples"]:
        lines.append(f"- `{sample['probe_id']}`: {sample['dominant_structure_summary']}")
    lines.extend(
        [
            "",
            "## Known Limitations",
            "",
            "- Components are derived from the 1024-point regular grid and represented by axis-aligned bounding boxes.",
            "- P09 is diag3 anisotropy only, not full tensor-k.",
            "- P10 is V1 global top Robin very-high h only; localized top contact and side asymmetry are unsupported.",
        ]
    )
    write_text(path, "\n".join(lines) + "\n")


def main() -> int:
    args = parse_args()
    if not args.predictions.is_file():
        raise FileNotFoundError(f"predictions not found: {args.predictions}")
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
    audited_samples = []
    component_rows = []
    for sample in samples:
        metrics = metrics_by_id.get(sample["sample_id"]) or metrics_by_id[sample["probe_id"]]
        sample_payload, components = audit_sample(sample, predictions[sample["sample_id"]], metrics)
        audited_samples.append(sample_payload)
        component_rows.extend(components)
    summary_rows = [flatten_summary(sample) for sample in audited_samples]
    dataset_summary = {
        "dataset_name": "v3_final_target_probe_v0",
        "sample_count": len(audited_samples),
        "resolution": 1024,
        "unit": "meter",
        "provenance": provenance,
        "sample_space_by_probe": {
            sample["probe_id"]: sample["sample_space"] for sample in audited_samples
        },
    }
    payload = {
        "diagnostic_scope": "read-only coordinate structure audit; no training; no inference rerun; no data generation",
        "subset": str(args.subset),
        "predictions": str(args.predictions),
        "metrics": str(args.metrics),
        "dataset_summary": dataset_summary,
        "samples": audited_samples,
        "samples_by_probe": {sample["probe_id"]: sample for sample in audited_samples},
        "summary_rows": summary_rows,
        "component_rows": component_rows,
        "used_local_regeneration": False,
        "used_generator_this_round": False,
        "reran_s5_inference": False,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.output_dir / "probe_structure_audit.json", payload)
    write_markdown(args.output_dir / "probe_structure_audit.md", payload)
    write_csv(args.output_dir / "probe_structure_audit.csv", summary_rows)
    write_csv(args.output_dir / "probe_component_table.csv", component_rows)
    print(f"structure audit complete: samples={len(audited_samples)} components={len(component_rows)} output_dir={args.output_dir}")
    for row in summary_rows:
        print(f"{row['probe_id']}: {row['dominant_structure_summary']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
