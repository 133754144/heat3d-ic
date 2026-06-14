#!/usr/bin/env python3
"""Component-level isolated figures for Heat3D v3 final-target probes."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection
import numpy as np


REPO_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = Path(__file__).resolve().parent
for path in (REPO_DIR, SCRIPTS_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


COLOR_DOMAIN = "#cfcfcf"
COLOR_HIGH_K = "#1f77b4"
COLOR_LOW_K = "#ffbf00"
COLOR_STRONG_Q = "#d62728"
COLOR_WEAK_Q = "#d62728"
COLOR_ANISO = "#7e3fb2"
COLOR_TOP_H = "#2ca02c"
VISUAL_Z_SCALE = 2.8


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Draw isolated component audit figures.")
    parser.add_argument("--subset", type=Path, required=True)
    parser.add_argument("--structure-audit", type=Path, required=True)
    parser.add_argument("--component-table", type=Path, required=True)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--visual-z-scale", type=float, default=VISUAL_Z_SCALE)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_component_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


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


def add_cuboid(ax, lo: np.ndarray, hi: np.ndarray, color: str, alpha: float, edge_alpha: float = 0.55, linewidth: float = 0.5) -> None:
    poly = Poly3DCollection(
        box_faces(lo, hi),
        facecolors=color,
        edgecolors=(0.0, 0.0, 0.0, edge_alpha),
        linewidths=linewidth,
        alpha=alpha,
    )
    ax.add_collection3d(poly)


def add_domain(ax, z_scale: float) -> None:
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
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7),
    ]
    ax.add_collection3d(
        Line3DCollection([[corners[a], corners[b]] for a, b in edges], colors=COLOR_DOMAIN, linewidths=0.8, alpha=0.7)
    )


def setup_ax(ax, z_scale: float) -> None:
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_zlim(0, z_scale)
    ax.view_init(elev=24, azim=-55)
    ax.set_box_aspect((1.0, 1.0, z_scale * 0.38))
    ax.set_axis_off()
    ax.set_facecolor("white")


def domain_bounds(sample: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    bounds = sample["sample_space"]["domain_bounds_m"]
    lo = np.asarray([bounds["x"][0], bounds["y"][0], bounds["z"][0]], dtype=np.float64)
    hi = np.asarray([bounds["x"][1], bounds["y"][1], bounds["z"][1]], dtype=np.float64)
    return lo, hi


def to_visual(point: list[float], sample: dict[str, Any], z_scale: float) -> np.ndarray:
    lo, hi = domain_bounds(sample)
    span = np.where((hi - lo) == 0.0, 1.0, hi - lo)
    out = (np.asarray(point, dtype=np.float64) - lo) / span
    out[2] *= z_scale
    return out


def component_visual_bbox(component: dict[str, Any], sample: dict[str, Any], z_scale: float) -> tuple[np.ndarray, np.ndarray]:
    lo = to_visual(component["bbox_min_m"], sample, z_scale)
    hi = to_visual(component["bbox_max_m"], sample, z_scale)
    # Inflate flat single-slice boxes enough to remain visible.
    min_size = np.asarray([0.025, 0.025, 0.05], dtype=np.float64)
    center = 0.5 * (lo + hi)
    half = np.maximum(0.5 * (hi - lo), min_size)
    lo = np.maximum(np.asarray([0.0, 0.0, 0.0]), center - half)
    hi = np.minimum(np.asarray([1.0, 1.0, z_scale]), center + half)
    return lo, hi


def kind_color(kind: str) -> str:
    if kind == "high_k":
        return COLOR_HIGH_K
    if kind == "low_k":
        return COLOR_LOW_K
    if kind == "strong_q":
        return COLOR_STRONG_Q
    if kind == "weak_q":
        return COLOR_WEAK_Q
    if kind == "anisotropic_k":
        return COLOR_ANISO
    return "#777777"


def kind_alpha(kind: str) -> float:
    if kind == "strong_q":
        return 0.82
    if kind == "anisotropic_k":
        return 0.48
    if kind == "weak_q":
        return 0.025
    return 0.30


def components_by_kind(sample: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    k = sample["k_space"]
    q = sample["q_space"]
    return {
        "high_k": list(k.get("high_k_components") or []),
        "low_k": list(k.get("low_k_components") or []),
        "strong_q": list(q.get("strong_q_components") or []),
        "weak_q": list(q.get("weak_q_components") or []),
        "anisotropic_k": list(k.get("anisotropic_components") or []),
    }


def distance_to_source(component: dict[str, Any]) -> float:
    value = component.get("distance_to_nearest_strong_q_bbox_m")
    if value is None:
        return float("inf")
    return float(value)


def nearest_component(components: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not components:
        return None
    return min(components, key=distance_to_source)


def render_components(
    ax,
    sample: dict[str, Any],
    components: list[dict[str, Any]],
    *,
    z_scale: float,
    title: str,
    top_boundary: bool = False,
) -> None:
    add_domain(ax, z_scale)
    for component in components:
        lo, hi = component_visual_bbox(component, sample, z_scale)
        add_cuboid(
            ax,
            lo,
            hi,
            kind_color(str(component.get("kind"))),
            kind_alpha(str(component.get("kind"))),
            edge_alpha=0.7 if component.get("kind") in {"strong_q", "anisotropic_k"} else 0.45,
            linewidth=0.8 if component.get("kind") in {"strong_q", "anisotropic_k"} else 0.45,
        )
    if top_boundary:
        add_cuboid(
            ax,
            np.asarray([0.0, 0.0, z_scale * 0.985]),
            np.asarray([1.0, 1.0, z_scale]),
            COLOR_TOP_H,
            0.40,
            edge_alpha=0.75,
            linewidth=0.9,
        )
    setup_ax(ax, z_scale)
    ax.set_title(title, fontsize=8)


def component_text(component: dict[str, Any]) -> str:
    lines = [
        f"component_id: {component.get('component_id')}",
        f"kind: {component.get('kind')}",
        f"point_count: {component.get('point_count')}",
        f"bbox_min_m: {component.get('bbox_min_m')}",
        f"bbox_max_m: {component.get('bbox_max_m')}",
        f"bbox_center_m: {component.get('bbox_center_m')}",
        f"bbox_size_m: {component.get('bbox_size_m')}",
    ]
    for key in ("k_eff_mean", "q_mean", "anisotropy_ratio_mean", "distance_to_nearest_strong_q_bbox_m"):
        if key in component:
            lines.append(f"{key}: {component.get(key)}")
    return "\n".join(lines)


def save_component_figure(sample: dict[str, Any], component: dict[str, Any], output_dir: Path, z_scale: float) -> Path:
    path = output_dir / f"{component['component_id']}.png"
    fig = plt.figure(figsize=(8.4, 5.2), constrained_layout=True)
    ax = fig.add_subplot(1, 2, 1, projection="3d")
    render_components(ax, sample, [component], z_scale=z_scale, title=str(component["component_id"]))
    text_ax = fig.add_subplot(1, 2, 2)
    text_ax.axis("off")
    text_ax.text(0.0, 1.0, component_text(component), va="top", ha="left", fontsize=8, family="monospace")
    fig.suptitle(f"{sample['probe_id']} isolated component audit", fontsize=11)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=170)
    plt.close(fig)
    return path


def save_gallery(sample: dict[str, Any], output_dir: Path, z_scale: float) -> Path:
    by_kind = components_by_kind(sample)
    high = by_kind["high_k"]
    low = by_kind["low_k"]
    strong = by_kind["strong_q"]
    aniso = by_kind["anisotropic_k"]
    all_components = low[:8] + high[:8] + aniso[:4] + strong[:8]
    nearest_low = nearest_component(low)
    nearest_high = nearest_component(high)
    panels: list[tuple[str, list[dict[str, Any]], bool]] = [
        ("all components overlay", all_components, False),
        ("high-k only", high, False),
        ("low-k only", low, False),
        ("strong-q only", strong, False),
        ("source + nearest low-k", strong + ([nearest_low] if nearest_low else []), False),
        ("source + nearest high-k", strong + ([nearest_high] if nearest_high else []), False),
    ]
    if sample["probe_id"] == "P09":
        panels.extend(
            [
                ("anisotropic-k only", aniso, False),
                ("source + anisotropic patch", strong + aniso[:1], False),
            ]
        )
    if sample["probe_id"] == "P10":
        panels.extend(
            [
                ("top high-h boundary only", [], True),
                ("source + top high-h boundary", strong, True),
            ]
        )
    rows = 2 if len(panels) <= 6 else 2
    cols = 3 if len(panels) <= 6 else 4
    fig = plt.figure(figsize=(cols * 4.0, rows * 3.6), constrained_layout=True)
    for index, (title, components, top_boundary) in enumerate(panels, start=1):
        ax = fig.add_subplot(rows, cols, index, projection="3d")
        render_components(ax, sample, components, z_scale=z_scale, title=title, top_boundary=top_boundary)
    fig.suptitle(
        f"{sample['probe_id']} component gallery | {sample['k_region_mode']} | {sample['source_category']}",
        fontsize=12,
    )
    path = output_dir / f"{sample['probe_id']}_component_gallery.png"
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=170)
    plt.close(fig)
    return path


def main() -> int:
    args = parse_args()
    if not args.structure_audit.is_file():
        raise FileNotFoundError(f"structure audit not found: {args.structure_audit}")
    if not args.component_table.is_file():
        raise FileNotFoundError(f"component table not found: {args.component_table}")
    if not args.metrics.is_file():
        raise FileNotFoundError(f"metrics not found: {args.metrics}")
    if not args.subset.exists():
        raise FileNotFoundError(f"subset not found: {args.subset}")

    audit = load_json(args.structure_audit)
    component_csv_rows = read_component_csv(args.component_table)
    samples = list(audit.get("samples") or [])
    if not samples:
        raise ValueError(f"{args.structure_audit}: no samples found")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    gallery_paths = []
    component_paths = []
    for sample in samples:
        gallery_paths.append(save_gallery(sample, args.output_dir, args.visual_z_scale))
        for components in components_by_kind(sample).values():
            for component in components:
                component_paths.append(save_component_figure(sample, component, args.output_dir, args.visual_z_scale))

    manifest = {
        "diagnostic_scope": "component-level isolated structure figures; no training; no inference rerun; no data generation",
        "subset": str(args.subset),
        "structure_audit": str(args.structure_audit),
        "component_table": str(args.component_table),
        "metrics": str(args.metrics),
        "output_dir": str(args.output_dir),
        "sample_count": len(samples),
        "component_csv_row_count": len(component_csv_rows),
        "gallery_count": len(gallery_paths),
        "component_figure_count": len(component_paths),
        "gallery_figures": [str(path) for path in gallery_paths],
        "component_figures": [str(path) for path in component_paths],
        "used_local_regeneration": False,
        "used_generator_this_round": False,
        "reran_s5_inference": False,
    }
    manifest_path = args.output_dir / "component_figure_manifest.json"
    write_json(manifest_path, manifest)
    print(
        "component audit figures complete: "
        f"galleries={len(gallery_paths)} components={len(component_paths)} output_dir={args.output_dir}"
    )
    print(f"manifest={manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
