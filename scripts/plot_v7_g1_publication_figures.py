#!/usr/bin/env python3
"""Render V7 G1 publication figures from frozen evidence only.

The script is intentionally an offline renderer.  It reads the existing
figure manifest, frozen H2 prediction/effect/support artifacts, and an
explicitly extracted valid_iid geometry/truth fixture.  It never loads a
checkpoint, calls a model, runs inference, or computes a new benchmark
metric.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import sys
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rigno.heat3d_training.support import (  # noqa: E402
    array_sha256,
    select_alternative_support,
)


PRIMARY_ROUTE = "U_v2_16384_reconstruction"
H2_METRIC = "source_region_RMSE_K"
VARIANT_DIRS = {
    "Full": "Full",
    "generic support": "layout_agnostic_stratified_support",
    "CV-only support": "cv_only_support",
}
SUPPORT_COLORS = {
    "q block": "#e66101",
    "k block": "#5e3c99",
    "interface": "#d73027",
    "top": "#1b9e77",
    "bottom": "#377eb8",
    "volume": "#969696",
}
SUPPORT_MARKERS = {
    "q block": "*",
    "k block": "D",
    "interface": "^",
    "top": "s",
    "bottom": "P",
    "volume": "o",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def source_record(path: Path, *, role: str, logical_path: str | None = None) -> dict[str, str]:
    return {
        "role": role,
        "path": logical_path or str(path),
        "sha256": sha256(path),
    }


def load_manifest(path: Path) -> dict[str, Any]:
    manifest = load_json(path)
    if manifest.get("schema_version") != "heat3d_v7_g1_figure_manifest_v1":
        raise ValueError("unexpected V7 G1 figure manifest schema")
    if manifest.get("no_new_inference") is not True:
        raise ValueError("figure manifest does not prohibit new inference")
    if manifest.get("no_manual_case_selection") is not True:
        raise ValueError("figure manifest does not prohibit manual case selection")
    selection = manifest.get("selection_basis") or {}
    if selection.get("route_id") != PRIMARY_ROUTE or selection.get("metric") != H2_METRIC:
        raise ValueError("figure manifest primary route/metric drifted")
    selected = selection.get("selected_rows") or {}
    for name in ("median", "p90", "p95"):
        if not isinstance(selected.get(name), dict):
            raise ValueError(f"figure manifest has no deterministic {name} case")
        if not selected[name].get("sample_id") or selected[name].get("seed") not in (0, 1, 2):
            raise ValueError(f"invalid deterministic {name} case")
    return manifest


class GridIndexer:
    def __init__(self, coords: np.ndarray) -> None:
        points = np.asarray(coords, dtype=np.float64)
        if points.shape != (240825, 3) or not np.all(np.isfinite(points)):
            raise ValueError("H2 common-domain coordinates must be finite [240825,3]")
        self.xs = np.unique(points[:, 0])
        self.ys = np.unique(points[:, 1])
        self.zs = np.unique(points[:, 2])
        if (len(self.xs), len(self.ys), len(self.zs)) != (65, 65, 57):
            raise ValueError("unexpected 240825 rectilinear grid shape")
        self.x_index = np.searchsorted(self.xs, points[:, 0])
        self.y_index = np.searchsorted(self.ys, points[:, 1])
        self.z_index = np.searchsorted(self.zs, points[:, 2])

    @property
    def shape(self) -> tuple[int, int, int]:
        return len(self.zs), len(self.ys), len(self.xs)

    def grid(self, values: np.ndarray) -> np.ndarray:
        flat = np.asarray(values, dtype=np.float64).reshape(-1)
        if flat.shape != (240825,) or not np.all(np.isfinite(flat)):
            raise ValueError("common-domain field must be finite [240825]")
        result = np.full(self.shape, np.nan, dtype=np.float64)
        result[self.z_index, self.y_index, self.x_index] = flat
        if np.isnan(result).any():
            raise ValueError("common-domain coordinates do not form a complete grid")
        return result


def _prediction_for(path: Path, sample_id: str) -> np.ndarray:
    with np.load(path, allow_pickle=False) as archive:
        required = {"sample_ids", "prediction_deltaT_K", "split", "domain_id", "point_count"}
        if not required <= set(archive.files):
            raise ValueError(f"prediction artifact missing required keys: {path}")
        if str(np.asarray(archive["split"]).item()) != "valid_iid":
            raise ValueError(f"prediction split is not valid_iid: {path}")
        if str(np.asarray(archive["domain_id"]).item()) != "heat3d_v6_p1i_full_field_240825":
            raise ValueError(f"prediction domain drifted: {path}")
        if int(np.asarray(archive["point_count"]).item()) != 240825:
            raise ValueError(f"prediction point count drifted: {path}")
        ids = np.asarray(archive["sample_ids"]).astype(str)
        matches = np.flatnonzero(ids == sample_id)
        if len(matches) != 1:
            raise ValueError(f"prediction sample lookup is not unique: {path} {sample_id}")
        prediction = np.asarray(archive["prediction_deltaT_K"][int(matches[0])], dtype=np.float64)
    if prediction.shape != (240825,) or not np.all(np.isfinite(prediction)):
        raise ValueError(f"invalid frozen prediction row: {path} {sample_id}")
    return prediction


def load_truth_fixture(path: Path) -> tuple[np.ndarray, dict[str, np.ndarray], GridIndexer]:
    with np.load(path, allow_pickle=False) as archive:
        required = {"sample_ids", "coords", "truth_deltaT_K"}
        if not required <= set(archive.files):
            raise ValueError("truth fixture missing required geometry/truth keys")
        sample_ids = np.asarray(archive["sample_ids"]).astype(str)
        coords = np.asarray(archive["coords"], dtype=np.float64)
        truth = np.asarray(archive["truth_deltaT_K"], dtype=np.float64)
    if len(sample_ids) != 3 or truth.shape != (3, 240825):
        raise ValueError("truth fixture must contain exactly three selected valid_iid rows")
    if len(set(sample_ids.tolist())) != 3:
        raise ValueError("truth fixture sample IDs are not unique")
    indexer = GridIndexer(coords)
    by_id = {sample_id: truth[i] for i, sample_id in enumerate(sample_ids)}
    return coords, by_id, indexer


def _save_figure(fig: plt.Figure, base: Path, title: str) -> list[Path]:
    base.parent.mkdir(parents=True, exist_ok=True)
    pdf = base.with_suffix(".pdf")
    svg = base.with_suffix(".svg")
    review_png = base.with_name(base.name + "_review.png")
    metadata = {"Creator": "V7 G1 frozen-evidence publication renderer", "Title": title}
    fig.savefig(pdf, bbox_inches="tight", metadata=metadata)
    fig.savefig(svg, bbox_inches="tight", metadata=metadata)
    fig.savefig(review_png, dpi=260, bbox_inches="tight", metadata={"Software": metadata["Creator"]})
    plt.close(fig)
    return [pdf, svg, review_png]


def _axis_style(ax: plt.Axes) -> None:
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(0.0, 10.0)
    ax.set_ylim(0.0, 10.0)
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    ax.grid(False)


def _draw_xy_blocks(ax: plt.Axes, meta: Mapping[str, Any]) -> None:
    q_blocks = (meta.get("q_blocks") or [])
    k_blocks = (meta.get("k_blocks") or [])
    for index, block in enumerate(q_blocks):
        x0, x1, y0, y1 = (float(value) * 10.0 for value in block["bbox_fraction_xy"])
        ax.add_patch(
            Rectangle(
                (x0, y0), x1 - x0, y1 - y0,
                facecolor="#fdb863", edgecolor="#e66101", linewidth=1.1,
                alpha=0.14, linestyle="-", zorder=1,
            )
        )
        if index < 2:
            ax.text(x0, y1, "q", color="#a63603", fontsize=6, ha="left", va="bottom", zorder=2)
    for index, block in enumerate(k_blocks):
        x0, x1, y0, y1 = (float(value) * 10.0 for value in block["bbox_fraction_xy"])
        ax.add_patch(
            Rectangle(
                (x0, y0), x1 - x0, y1 - y0,
                facecolor="#998ec3", edgecolor="#5e3c99", linewidth=1.0,
                alpha=0.10, linestyle="--", zorder=1,
            )
        )
        if index < 2:
            ax.text(x0, y0, "k", color="#542788", fontsize=6, ha="left", va="top", zorder=2)


def _draw_layer_inset(ax: plt.Axes, meta: Mapping[str, Any]) -> None:
    layers = (meta.get("physics") or {}).get("layers_bottom_to_top") or []
    thicknesses = np.asarray([float(layer["thickness_m"]) for layer in layers], dtype=np.float64)
    boundaries = np.concatenate(([0.0], np.cumsum(thicknesses))) * 1000.0
    inset = ax.inset_axes([0.79, 0.08, 0.16, 0.84])
    colors = plt.get_cmap("tab20")(np.linspace(0.0, 1.0, max(1, len(layers))))
    for index, layer in enumerate(layers):
        inset.add_patch(
            Rectangle((0.0, boundaries[index]), 1.0, boundaries[index + 1] - boundaries[index],
                      facecolor=colors[index], edgecolor="white", linewidth=0.4)
        )
    for boundary in boundaries[1:-1]:
        inset.axhline(boundary, color="black", linewidth=0.35, linestyle=":")
    inset.set_xlim(0.0, 1.0)
    inset.set_ylim(boundaries[0], boundaries[-1])
    inset.set_xticks([])
    inset.set_yticks([])
    inset.set_title("layers / interfaces", fontsize=6, pad=2)
    inset.text(0.5, boundaries[-1] * 0.02, "bottom", ha="center", va="bottom", fontsize=5, rotation=90)
    inset.text(0.5, boundaries[-1] * 0.98, "top", ha="center", va="top", fontsize=5, rotation=90)


def _classify_full_support(
    coords: np.ndarray,
    layer_id: np.ndarray,
    k_field: np.ndarray,
    q_field: np.ndarray,
    meta: Mapping[str, Any],
) -> np.ndarray:
    physics = meta.get("physics") or {}
    layers = physics.get("layers_bottom_to_top") or []
    boundaries = np.concatenate(
        ([float(coords[:, 2].min())], np.cumsum([float(row["thickness_m"]) for row in layers]) + float(coords[:, 2].min()))
    )
    background = np.asarray([row["background_k_xyz_W_mK"] for row in layers], dtype=np.float64)
    q = np.asarray(q_field, dtype=np.float64).reshape(-1)
    k = np.asarray(k_field, dtype=np.float64)
    layer = np.asarray(layer_id, dtype=np.int32).reshape(-1)
    z = np.asarray(coords[:, 2], dtype=np.float64)
    labels = np.full(len(coords), "volume", dtype="U16")
    labels[np.isclose(z, boundaries[0], atol=1.0e-15, rtol=0.0)] = "bottom"
    labels[np.isclose(z, boundaries[-1], atol=1.0e-15, rtol=0.0)] = "top"
    interface = np.zeros(len(coords), dtype=bool)
    for boundary in boundaries[1:-1]:
        interface |= np.isclose(z, boundary, atol=1.0e-15, rtol=0.0)
    interface &= ~(
        np.isclose(z, boundaries[0], atol=1.0e-15, rtol=0.0)
        | np.isclose(z, boundaries[-1], atol=1.0e-15, rtol=0.0)
    )
    labels[interface] = "interface"
    k_background = background[layer]
    k_block = np.max(np.abs(k - k_background), axis=1) > 1.0e-12
    labels[k_block] = "k block"
    labels[q > 0.0] = "q block"
    return labels


def _scatter_support(ax: plt.Axes, coords: np.ndarray, labels: Sequence[str]) -> None:
    values = np.asarray(labels).astype(str)
    for label in ("q block", "k block", "interface", "top", "bottom", "volume"):
        mask = values == label
        if not np.any(mask):
            continue
        ax.scatter(
            coords[mask, 0] * 1000.0,
            coords[mask, 1] * 1000.0,
            s=13 if label != "volume" else 8,
            marker=SUPPORT_MARKERS[label],
            color=SUPPORT_COLORS[label],
            alpha=0.82,
            linewidths=0.25,
            edgecolors="white",
            label=f"{label} ({int(np.sum(mask))})",
            zorder=3,
        )


def _support_caption() -> str:
    return (
        "Full uses the frozen physics-layout-aware sparse support: q/k-block, "
        "interface, surface, and control-volume strata. Generic support keeps "
        "the geometry/interface/surface/CV coverage but removes q/k layout quotas. "
        "CV-only uses only interior control-volume-weighted points."
    )


def render_support_figure(
    *,
    manifest: Mapping[str, Any],
    support_input: Path,
    support_meta_path: Path,
    truth_input: Path,
    output_dir: Path,
) -> tuple[list[Path], dict[str, Any]]:
    with np.load(support_input, allow_pickle=False) as archive:
        native_coords = np.asarray(archive["coords_1024"], dtype=np.float64)
        native_layer = np.asarray(archive["layer_id_1024"], dtype=np.int32)
        native_k = np.asarray(archive["k_field_1024"], dtype=np.float64)
        native_q = np.asarray(archive["q_field_1024"], dtype=np.float64)
        np.asarray(archive["control_volume_1024"], dtype=np.float64).reshape(-1)
    meta = load_json(support_meta_path)
    if native_coords.shape != (1024, 3) or native_k.shape != (1024, 3) or native_q.shape != (1024, 1):
        raise ValueError("support fixture native arrays have unexpected shapes")
    full_field = np.load(truth_input, allow_pickle=False)
    shared_coords = np.asarray(full_field["coords"], dtype=np.float64)
    shared_cv_full = np.asarray(full_field["control_volume"], dtype=np.float64).reshape(-1)
    if shared_coords.shape != (240825, 3) or shared_cv_full.shape != (240825,):
        raise ValueError("support figure shared geometry fixture has unexpected shape")
    layers = (meta.get("physics") or {}).get("layers_bottom_to_top") or []
    z0 = float(shared_coords[:, 2].min())
    boundaries = z0 + np.concatenate(([0.0], np.cumsum([float(row["thickness_m"]) for row in layers])))
    generic = select_alternative_support(
        "generic_stratified_v2",
        coords=shared_coords,
        control_volume=shared_cv_full,
        boundaries=boundaries,
        sample_id="v6p1if1_0533",
        seed=1,
    )
    cv_only = select_alternative_support(
        "cv_only_v1",
        coords=shared_coords,
        control_volume=shared_cv_full,
        boundaries=boundaries,
        sample_id="v6p1if1_0533",
        seed=1,
    )
    selections = {
        "Full": (native_coords, _classify_full_support(native_coords, native_layer, native_k, native_q, meta)),
        "generic support": (shared_coords[generic.indices], generic.strata),
        "CV-only support": (shared_coords[cv_only.indices], cv_only.strata),
    }
    fig = plt.figure(figsize=(14.5, 10.0))
    grid = fig.add_gridspec(2, 2, left=0.055, right=0.985, bottom=0.19, top=0.90, wspace=0.18, hspace=0.26)
    reference = fig.add_subplot(grid[0, 0])
    _axis_style(reference)
    _draw_xy_blocks(reference, meta)
    reference.set_title("Frozen physical / q-k layout reference", fontsize=11)
    reference.text(
        0.02, 0.98,
        "q blocks = source region\nk blocks = conductivity-layout reference",
        transform=reference.transAxes, ha="left", va="top", fontsize=8,
        bbox={"facecolor": "white", "alpha": 0.82, "edgecolor": "none", "pad": 2},
    )
    _draw_layer_inset(reference, meta)
    reference.legend(
        handles=[
            Rectangle((0, 0), 1, 1, facecolor="#fdb863", edgecolor="#e66101", alpha=0.35, label="q/source block"),
            Rectangle((0, 0), 1, 1, facecolor="#998ec3", edgecolor="#5e3c99", alpha=0.3, linestyle="--", label="k block"),
        ], loc="lower left", fontsize=7, framealpha=0.9,
    )
    support_axes = {
        "Full": fig.add_subplot(grid[0, 1]),
        "generic support": fig.add_subplot(grid[1, 0]),
        "CV-only support": fig.add_subplot(grid[1, 1]),
    }
    titles = {
        "Full": "Full — physics-layout-aware sparse support",
        "generic support": "Generic support — no q/k layout quota",
        "CV-only support": "CV-only support — interior CV only",
    }
    for name, ax in support_axes.items():
        _axis_style(ax)
        _draw_xy_blocks(ax, meta)
        coords, labels = selections[name]
        _scatter_support(ax, coords, labels)
        ax.set_title(titles[name], fontsize=10)
        counts = {label: int(np.sum(np.asarray(labels).astype(str) == label)) for label in sorted(set(np.asarray(labels).astype(str)))}
        ax.text(
            0.02, 0.02,
            " / ".join(f"{key}:{value}" for key, value in counts.items()),
            transform=ax.transAxes, fontsize=6.8, ha="left", va="bottom",
            bbox={"facecolor": "white", "alpha": 0.82, "edgecolor": "none", "pad": 1.5},
        )
    handles = [
        Line2D([0], [0], marker=SUPPORT_MARKERS[key], color="w", markerfacecolor=SUPPORT_COLORS[key],
               markersize=7, label=key, linestyle="None")
        for key in ("q block", "k block", "interface", "top", "bottom", "volume")
    ]
    fig.legend(handles=handles, loc="lower center", ncol=6, fontsize=8, frameon=False, bbox_to_anchor=(0.5, 0.095))
    fig.text(0.5, 0.035, _support_caption(), ha="center", va="bottom", fontsize=8.2, wrap=True)
    fig.suptitle("V7 G1 support mechanism — same frozen representative sample v6p1if1_0533 (seed 1)", fontsize=14, y=0.965)
    outputs = _save_figure(fig, output_dir / "G1-1_support_mechanism", "V7 G1 support mechanism")
    details = {
        "figure_id": "support_visualization",
        "sample_id": "v6p1if1_0533",
        "seed": 1,
        "selection_rule": manifest["selection_basis"]["rule"],
        "support_coordinate_sha256": {
            "Full": array_sha256(native_coords),
            "generic_support": array_sha256(shared_coords[generic.indices]),
            "cv_only_support": array_sha256(shared_coords[cv_only.indices]),
        },
        "support_indices_sha256": {
            "Full": None,
            "generic_support": generic.index_sha256,
            "cv_only_support": cv_only.index_sha256,
        },
        "caption": _support_caption(),
        "outputs": [str(path) for path in outputs],
    }
    return outputs, details


def render_h2_figure(
    *,
    manifest: Mapping[str, Any],
    archive_root: Path,
    truth_input: Path,
    output_dir: Path,
) -> tuple[list[Path], dict[str, Any]]:
    coords, truth_by_id, indexer = load_truth_fixture(truth_input)
    selected = manifest["selection_basis"]["selected_rows"]
    cases: list[dict[str, Any]] = []
    all_temp: list[np.ndarray] = []
    all_errors: list[np.ndarray] = []
    prediction_sources: list[dict[str, str]] = []
    for case_name in ("median", "p90", "p95"):
        selection = selected[case_name]
        sample_id = str(selection["sample_id"])
        seed = int(selection["seed"])
        truth = np.asarray(truth_by_id[sample_id], dtype=np.float64)
        predictions: dict[str, np.ndarray] = {}
        for label, variant_dir in VARIANT_DIRS.items():
            path = archive_root / "h2_fullfield_240825_native" / PRIMARY_ROUTE / f"{variant_dir}_seed{seed}" / "predictions_best.npz"
            predictions[label] = _prediction_for(path, sample_id)
            prediction_sources.append(source_record(path, role=f"H2 primary prediction {label}; {case_name}"))
        truth_grid = indexer.grid(truth)
        z_slice = int(np.unravel_index(int(np.argmax(truth_grid)), truth_grid.shape)[0])
        maps = {"Ground truth": truth_grid[z_slice]}
        for label in ("Full", "generic support", "CV-only support"):
            maps[label] = indexer.grid(predictions[label])[z_slice]
            error_label = f"{label} error"
            maps[error_label] = maps[label] - maps["Ground truth"]
            all_errors.append(maps[error_label])
        all_temp.extend(maps[label] for label in ("Ground truth", "Full", "generic support", "CV-only support"))
        cases.append({"name": case_name, "sample_id": sample_id, "seed": seed, "z_slice": z_slice, "maps": maps})
    temp_vmin = min(0.0, min(float(np.min(values)) for values in all_temp))
    temp_vmax = max(float(np.max(values)) for values in all_temp)
    error_vmax = max(float(np.max(np.abs(values))) for values in all_errors)
    if not np.isfinite(temp_vmax) or not np.isfinite(error_vmax) or error_vmax <= 0.0:
        raise ValueError("invalid common H2 figure scales")
    error_vmax = float(error_vmax)
    fig, axes = plt.subplots(3, 7, figsize=(20.0, 9.8), squeeze=False)
    temp_titles = ["Ground truth", "Full", "Generic", "CV-only"]
    error_titles = ["Full error", "Generic error", "CV-only error"]
    for col, title in enumerate(temp_titles + error_titles):
        axes[0, col].set_title(title, fontsize=9.2)
    temp_mappable = None
    error_mappable = None
    extent = [float(indexer.xs[0] * 1000.0), float(indexer.xs[-1] * 1000.0), float(indexer.ys[0] * 1000.0), float(indexer.ys[-1] * 1000.0)]
    for row, case in enumerate(cases):
        maps = case["maps"]
        panels = [
            (maps["Ground truth"], "turbo", (temp_vmin, temp_vmax)),
            (maps["Full"], "turbo", (temp_vmin, temp_vmax)),
            (maps["generic support"], "turbo", (temp_vmin, temp_vmax)),
            (maps["CV-only support"], "turbo", (temp_vmin, temp_vmax)),
            (maps["Full error"], "RdBu_r", (-error_vmax, error_vmax)),
            (maps["generic support error"], "RdBu_r", (-error_vmax, error_vmax)),
            (maps["CV-only support error"], "RdBu_r", (-error_vmax, error_vmax)),
        ]
        for col, (values, cmap, limits) in enumerate(panels):
            ax = axes[row, col]
            image = ax.imshow(values, origin="lower", extent=extent, aspect="equal", cmap=cmap, vmin=limits[0], vmax=limits[1], interpolation="nearest")
            if col < 4:
                temp_mappable = image
            else:
                error_mappable = image
            ax.set_xticks([])
            ax.set_yticks([])
            if col == 0:
                ax.set_ylabel(f"{case['name']}\n{case['sample_id']}\nseed {case['seed']}\nz={indexer.zs[case['z_slice']]*1000:.3f} mm", fontsize=8, rotation=90, labelpad=12)
            if row == 2:
                ax.set_xlabel("x/y (mm)", fontsize=7)
    fig.colorbar(temp_mappable, ax=axes[:, :4].ravel().tolist(), fraction=0.012, pad=0.012, label="ΔT (K)")
    fig.colorbar(error_mappable, ax=axes[:, 4:].ravel().tolist(), fraction=0.012, pad=0.012, label="prediction − truth (K)")
    fig.suptitle(
        "V7 G1 H2 full-field qualitative comparison — primary U16384 → frozen reconstruction → 240825",
        fontsize=13.5, y=0.995,
    )
    fig.text(
        0.5, 0.005,
        "All temperature panels share one ΔT scale; all error panels share one symmetric K scale. "
        "Cases are selected only by the frozen median/p90/p95 H2a primary effect rule.",
        ha="center", va="bottom", fontsize=8.2,
    )
    outputs = _save_figure(fig, output_dir / "G1-2_h2_fullfield_comparison", "V7 G1 H2 full-field comparison")
    details = {
        "figure_id": "h2_thermal_error_maps",
        "route_id": PRIMARY_ROUTE,
        "route_role": "H2 primary",
        "selection_rule": manifest["selection_basis"]["rule"],
        "selected_cases": [
            {"name": case["name"], "sample_id": case["sample_id"], "seed": case["seed"], "z_slice_index": case["z_slice"], "z_m": float(indexer.zs[case["z_slice"]])}
            for case in cases
        ],
        "common_scales": {
            "temperature_deltaT_K": [float(temp_vmin), float(temp_vmax)],
            "error_K_symmetric": [-error_vmax, error_vmax],
        },
        "prediction_sources": prediction_sources,
        "truth_source": str(truth_input),
        "outputs": [str(path) for path in outputs],
    }
    return outputs, details


def render_effect_figure(
    *,
    manifest: Mapping[str, Any],
    archive_root: Path,
    output_dir: Path,
) -> tuple[list[Path], dict[str, Any]]:
    per_sample_path = archive_root / "h2_fullfield_240825_native" / "h2_per_sample_effects.json"
    effect_table_path = archive_root / "h2_fullfield_240825_native" / "h2_hypothesis_effect_table.json"
    per_sample = load_json(per_sample_path)["rows"]
    effect_table = load_json(effect_table_path)["rows"]
    distributions: dict[str, np.ndarray] = {}
    markers: dict[str, dict[str, float]] = {}
    for variant, contrast, table_variant in (
        ("layout_agnostic_stratified_support", "H2a", "layout_agnostic_stratified_support"),
        ("cv_only_support", "H2b", "cv_only_support"),
    ):
        rows = [
            row for row in per_sample
            if row["route_id"] == PRIMARY_ROUTE and row["metric"] == H2_METRIC and row["variant"] == variant
        ]
        if len(rows) != 384 or not all(row.get("same_sample_id") and row.get("same_coordinate_grid") for row in rows):
            raise ValueError(f"unexpected frozen H2 effect rows for {contrast}")
        table_rows = [
            row for row in effect_table
            if row["route_id"] == PRIMARY_ROUTE and row["metric"] == H2_METRIC and row["ablation_variant"] == table_variant
        ]
        if len(table_rows) != 1:
            raise ValueError(f"missing stored H2 distribution summary for {contrast}")
        distribution = table_rows[0]["paired_sample_distribution"]
        distributions[contrast] = np.asarray([float(row["effect_ablation_minus_full"]) for row in rows], dtype=np.float64)
        markers[contrast] = {
            "median": float(distribution["median"]),
            "p90": float(distribution["p90"]),
            "p95": float(distribution["p95"]),
        }
    all_values = np.concatenate(list(distributions.values()))
    x_min = min(-0.5, float(np.min(all_values)))
    x_max = max(14.0, float(np.max(all_values)) * 1.04)
    bins = np.linspace(x_min, x_max, 38)
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.9), sharex=True, sharey=True)
    titles = {"H2a": "H2a — Full vs generic support", "H2b": "H2b — Full vs CV-only support"}
    colors = {"H2a": "#1b9e77", "H2b": "#7570b3"}
    ymax = 0.0
    for ax, contrast in zip(axes, ("H2a", "H2b"), strict=True):
        values = distributions[contrast]
        counts, _, _ = ax.hist(values, bins=bins, density=True, color=colors[contrast], alpha=0.66, edgecolor="white", linewidth=0.5)
        ymax = max(ymax, float(np.max(counts)))
        ax.axvline(0.0, color="black", linestyle="--", linewidth=1.1, label="zero")
        line_styles = {"median": "-", "p90": ":", "p95": "-."}
        for key, style in line_styles.items():
            ax.axvline(markers[contrast][key], color="#d95f02", linestyle=style, linewidth=1.15, label=key)
        ax.set_title(titles[contrast], fontsize=11)
        ax.set_xlabel("paired effect: ablation − Full (K)")
        ax.text(
            0.98, 0.96,
            f"n = {len(values)}\nmedian = {markers[contrast]['median']:.3f} K\np90 = {markers[contrast]['p90']:.3f} K\np95 = {markers[contrast]['p95']:.3f} K",
            transform=ax.transAxes, ha="right", va="top", fontsize=8,
            bbox={"facecolor": "white", "alpha": 0.86, "edgecolor": "none", "pad": 2},
        )
        ax.grid(axis="y", color="#dddddd", linewidth=0.5)
    axes[0].set_ylabel("density")
    axes[0].legend(loc="upper left", fontsize=8, frameon=False)
    axes[1].legend(loc="upper left", fontsize=8, frameon=False)
    axes[0].set_xlim(x_min, x_max)
    axes[0].set_ylim(0.0, ymax * 1.18 if ymax else 1.0)
    fig.suptitle("V7 G1 H2 paired per-sample effects — frozen primary U16384 route", fontsize=13, y=1.01)
    fig.text(0.5, -0.02, "Markers use the stored preregistered median, p90, and p95 summaries; no new metric is defined.", ha="center", fontsize=8.2)
    outputs = _save_figure(fig, output_dir / "G1-3_h2_paired_effect_distribution", "V7 G1 H2 paired-effect distribution")
    details = {
        "figure_id": "h2_paired_effect_distribution",
        "route_id": PRIMARY_ROUTE,
        "metric": H2_METRIC,
        "contrast_rows": {
            contrast: {
                "count": int(len(distributions[contrast])),
                "stored_markers_K": markers[contrast],
                "min_effect_K_for_plot_extent": float(np.min(distributions[contrast])),
                "max_effect_K_for_plot_extent": float(np.max(distributions[contrast])),
            }
            for contrast in ("H2a", "H2b")
        },
        "source_artifacts": [
            source_record(per_sample_path, role="frozen H2 per-sample effects"),
            source_record(effect_table_path, role="frozen H2 stored distribution summaries"),
        ],
        "outputs": [str(path) for path in outputs],
    }
    return outputs, details


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive-root", type=Path, required=True)
    parser.add_argument("--figure-manifest", type=Path, required=True)
    parser.add_argument("--truth-input", type=Path, required=True)
    parser.add_argument("--support-input", type=Path, required=True)
    parser.add_argument("--support-meta", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    archive_root = args.archive_root.resolve()
    manifest_path = args.figure_manifest.resolve()
    truth_input = args.truth_input.resolve()
    support_input = args.support_input.resolve()
    support_meta = args.support_meta.resolve()
    output_dir = args.output_dir.resolve()
    manifest = load_manifest(manifest_path)
    for path in (archive_root, truth_input, support_input, support_meta):
        if not path.exists():
            raise FileNotFoundError(path)
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []
    figure_details: list[dict[str, Any]] = []
    support_outputs, support_details = render_support_figure(
        manifest=manifest, support_input=support_input, support_meta_path=support_meta,
        truth_input=truth_input, output_dir=output_dir
    )
    outputs.extend(support_outputs)
    figure_details.append(support_details)
    h2_outputs, h2_details = render_h2_figure(
        manifest=manifest, archive_root=archive_root, truth_input=truth_input, output_dir=output_dir
    )
    outputs.extend(h2_outputs)
    figure_details.append(h2_details)
    effect_outputs, effect_details = render_effect_figure(
        manifest=manifest, archive_root=archive_root, output_dir=output_dir
    )
    outputs.extend(effect_outputs)
    figure_details.append(effect_details)
    tracked_sources = [
        source_record(manifest_path, role="frozen figure manifest"),
        source_record(REPO_ROOT / "docs/v7_g1_formal_archive_manifest.json", role="frozen archive manifest"),
        source_record(REPO_ROOT / "configs/heat3d_v7/v7_g1_support_provider_contract.json", role="frozen support provider contract"),
        source_record(REPO_ROOT / "configs/heat3d_v7/v7_support_artifact_freeze.json", role="frozen support artifact semantics"),
        source_record(REPO_ROOT / "rigno/heat3d_training/support.py", role="frozen alternative support implementation"),
        source_record(truth_input, role="valid_iid truth/shared-geometry fixture"),
        source_record(support_input, role="valid_iid native support fixture"),
        source_record(support_meta, role="valid_iid support layout metadata fixture"),
    ]
    archive_manifest = REPO_ROOT / "docs/v7_g1_formal_archive_manifest.json"
    archive_h2_root = archive_root / "h2_fullfield_240825_native"
    frozen_h2_sources = [
        source_record(archive_root / "h2_fullfield_240825_native/h2_per_sample_effects.json", role="frozen H2 per-sample effects"),
        source_record(archive_root / "h2_fullfield_240825_native/h2_hypothesis_effect_table.json", role="frozen H2 hypothesis effect table"),
        source_record(archive_root / "h2_fullfield_240825_native/h2_variant_route_summary.json", role="frozen H2 route summaries"),
    ]
    # The full-field H5 was read remotely only for the explicitly selected
    # valid_iid rows; this SHA is the frozen dataset artifact SHA in the G1
    # contracts and is recorded without copying the 916 MB source archive.
    frozen_h2_sources.append({
        "role": "frozen valid_iid full-field source archive",
        "path": "devbox:/home/xyh/myCodeGitOnly/heat3d-ic/data/heat3d_v6_p1i_continuous_physics1024_v1_full_fields/full_fields.h5",
        "sha256": "49023ac1205b8e7cf7c5bf782b89fcdb34997704b3f9aa2fb2d46cf1a59163cb",
    })
    prediction_sources = [
        source for detail in figure_details if detail["figure_id"] == "h2_thermal_error_maps" for source in detail["prediction_sources"]
    ]
    provenance = {
        "schema_version": "heat3d_v7_g1_publication_figure_provenance_v1",
        "status": "RENDERED_FROM_FROZEN_EVIDENCE",
        "scope": "V7 G1 publication figures only; no new inference or metric",
        "renderer": {
            "script": "scripts/plot_v7_g1_publication_figures.py",
            "script_sha256": sha256(Path(__file__).resolve()),
            "python": platform.python_version(),
            "numpy": np.__version__,
            "matplotlib": matplotlib.__version__,
        },
        "frozen_contract": {
            "formal_code_sha": "191a7a06a681556f575a1c04e2b61cb13363efe1",
            "domain": "P1i valid_iid only",
            "h2_primary_route": PRIMARY_ROUTE,
            "h2_primary_metric": H2_METRIC,
            "temperature_panels_shared_scale": True,
            "error_panels_shared_symmetric_scale": True,
            "no_manual_case_selection": True,
            "no_new_metric": True,
            "checkpoint_loaded": False,
            "model_forward": False,
            "training": False,
            "test_iid_access": False,
            "sealed_access": False,
            "g2_touched": False,
        },
        "input_sources": tracked_sources + frozen_h2_sources + prediction_sources,
        "figures": figure_details,
        "outputs": [
            {"path": str(path.relative_to(REPO_ROOT)), "sha256": sha256(path), "bytes": path.stat().st_size}
            for path in outputs
        ],
    }
    provenance_json = output_dir / "v7_g1_figure_provenance.json"
    provenance_json.write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    provenance_md = output_dir / "v7_g1_figure_provenance.md"
    md_lines = [
        "# V7 G1 publication figure provenance",
        "",
        "Rendered from frozen G1 evidence only. No checkpoint was loaded, no model forward/inference was run, and no new metric was defined.",
        "",
        f"- H2 primary route: `{PRIMARY_ROUTE}`",
        f"- H2 primary metric: `{H2_METRIC}`",
        "- Selection: frozen manifest median/p90/p95 rule; no manual case selection.",
        "- Temperature panels: common scale; error panels: common symmetric K scale.",
        "- Population: selected `valid_iid` rows only; test/sealed untouched; G2 untouched.",
        "",
        "## Outputs",
        "",
    ]
    for row in provenance["outputs"]:
        md_lines.append(f"- `{row['path']}` — {row['bytes']} bytes — SHA256 `{row['sha256']}`")
    md_lines.extend(["", "## Frozen input hashes", ""])
    for row in provenance["input_sources"]:
        md_lines.append(f"- {row['role']}: `{row['path']}` — SHA256 `{row['sha256']}`")
    provenance_md.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": provenance["status"], "output_count": len(outputs), "provenance": str(provenance_json)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
