#!/usr/bin/env python3
"""Visualize generated Heat3D v3 final-target probe v0 samples.

The script reads existing probe arrays and writes figures under ignored
output/. It does not train, build graphs, or modify generated data.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_SUBSET = (
    REPO_ROOT
    / "data"
    / "heat3d-thermal-simulation"
    / "subsets"
    / "v3_final_target_probe_v0"
)
DEFAULT_OUTPUT_DIR = REPO_ROOT / "output" / "heat3d_v3_final_target_probe" / "figures"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subset", type=Path, default=DEFAULT_SUBSET)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--sample-id", action="append", default=None)
    parser.add_argument("--dpi", type=int, default=160)
    return parser.parse_args()


def _configure_matplotlib(output_dir: Path):
    mpl_config = output_dir / ".mplconfig"
    xdg_cache = output_dir / ".cache"
    mpl_config.mkdir(parents=True, exist_ok=True)
    (xdg_cache / "fontconfig").mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_config))
    os.environ.setdefault("XDG_CACHE_HOME", str(xdg_cache))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"expected JSON object: {path}")
    return data


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _sample_dirs(subset: Path, sample_ids: list[str] | None) -> list[Path]:
    root = subset / "samples" if (subset / "samples").is_dir() else subset
    if not root.is_dir():
        return []
    selected = set(sample_ids or [])
    dirs = sorted(child for child in root.iterdir() if child.is_dir() and (child / "sample_meta.json").is_file())
    if selected:
        dirs = [path for path in dirs if path.name in selected]
    return dirs


def _load_sample(sample_dir: Path) -> dict[str, Any]:
    coords = np.load(sample_dir / "coords.npy")
    k_field = np.load(sample_dir / "k_field.npy")
    q_field = np.load(sample_dir / "q_field.npy")
    temperature = np.load(sample_dir / "temperature.npy")
    metadata = _read_json(sample_dir / "metadata.json")
    return {
        "sample_id": sample_dir.name,
        "coords": coords,
        "k_field": k_field,
        "q_field": q_field,
        "temperature": temperature,
        "metadata": metadata,
    }


def _effective_k(k_field: np.ndarray) -> np.ndarray:
    if k_field.shape[1] == 1:
        return k_field[:, 0]
    if k_field.shape[1] == 3:
        return np.mean(k_field, axis=1)
    return k_field[:, 0]


def _grid_axes(coords: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return tuple(np.unique(coords[:, axis]) for axis in range(3))  # type: ignore[return-value]


def _slice_values(
    coords: np.ndarray,
    values: np.ndarray,
    z_value: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    xs, ys, _ = _grid_axes(coords)
    mask = np.isclose(coords[:, 2], z_value)
    if not np.any(mask):
        raise ValueError(f"no points found for z={z_value}")
    z_coords = coords[mask]
    z_values = values[mask]
    image = np.full((ys.size, xs.size), np.nan, dtype=np.float64)
    x_index = {float(value): idx for idx, value in enumerate(xs)}
    y_index = {float(value): idx for idx, value in enumerate(ys)}
    for point, value in zip(z_coords, z_values):
        image[y_index[float(point[1])], x_index[float(point[0])]] = float(value)
    return xs, ys, image


def _z_mid(coords: np.ndarray) -> float:
    _, _, zs = _grid_axes(coords)
    return float(zs[len(zs) // 2])


def _source_z(coords: np.ndarray, q: np.ndarray) -> float:
    _, _, zs = _grid_axes(coords)
    q_flat = q[:, 0]
    if np.any(q_flat > 0.0):
        z_weighted = float(np.average(coords[:, 2], weights=np.maximum(q_flat, 0.0)))
        return float(zs[int(np.argmin(np.abs(zs - z_weighted)))])
    return _z_mid(coords)


def _format_title(metadata: dict[str, Any]) -> str:
    return (
        f"{metadata.get('probe_id')} {metadata.get('probe_family')}\n"
        f"{metadata.get('k_region_mode')} / {metadata.get('source_category')} / {metadata.get('bc_category')}"
    )


def _scatter_3d(plt, sample: dict[str, Any], output_path: Path, dpi: int) -> None:
    coords = sample["coords"]
    k_eff = _effective_k(sample["k_field"])
    q = sample["q_field"][:, 0]
    temp = sample["temperature"][:, 0]
    metadata = sample["metadata"]
    fields = [
        ("k_eff W/mK", k_eff, "viridis"),
        ("q W/m^3", q, "magma"),
        ("T K", temp, "inferno"),
    ]
    fig = plt.figure(figsize=(13.5, 4.2))
    for idx, (title, values, cmap) in enumerate(fields, start=1):
        ax = fig.add_subplot(1, 3, idx, projection="3d")
        scatter = ax.scatter(
            coords[:, 0],
            coords[:, 1],
            coords[:, 2],
            c=values,
            cmap=cmap,
            s=12,
            alpha=0.78,
            linewidths=0.0,
        )
        ax.set_title(title)
        ax.set_xlabel("x m")
        ax.set_ylabel("y m")
        ax.set_zlabel("z m")
        fig.colorbar(scatter, ax=ax, shrink=0.65, pad=0.08)
    fig.suptitle(_format_title(metadata), fontsize=10)
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def _slice_figure(plt, sample: dict[str, Any], z_value: float, output_path: Path, dpi: int, label: str) -> None:
    coords = sample["coords"]
    metadata = sample["metadata"]
    fields = [
        ("k_eff W/mK", _effective_k(sample["k_field"]), "viridis"),
        ("q W/m^3", sample["q_field"][:, 0], "magma"),
        ("T K", sample["temperature"][:, 0], "inferno"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(13.0, 3.8), constrained_layout=True)
    for ax, (title, values, cmap) in zip(axes, fields):
        xs, ys, image = _slice_values(coords, values, z_value)
        handle = ax.imshow(
            image,
            origin="lower",
            extent=[float(xs[0]), float(xs[-1]), float(ys[0]), float(ys[-1])],
            cmap=cmap,
            aspect="auto",
        )
        ax.set_title(title)
        ax.set_xlabel("x m")
        ax.set_ylabel("y m")
        fig.colorbar(handle, ax=ax, shrink=0.82)
    fig.suptitle(f"{_format_title(metadata)}\n{label} z={z_value:.6e} m", fontsize=10)
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def _p09_k_channels_figure(plt, sample: dict[str, Any], output_path: Path, dpi: int) -> None:
    k_field = sample["k_field"]
    if k_field.shape[1] != 3:
        return
    coords = sample["coords"]
    z_value = _source_z(coords, sample["q_field"])
    fig, axes = plt.subplots(1, 3, figsize=(13.0, 3.8), constrained_layout=True)
    for ax, channel, name in zip(axes, range(3), ("kx", "ky", "kz")):
        xs, ys, image = _slice_values(coords, k_field[:, channel], z_value)
        handle = ax.imshow(
            image,
            origin="lower",
            extent=[float(xs[0]), float(xs[-1]), float(ys[0]), float(ys[-1])],
            cmap="viridis",
            aspect="auto",
        )
        ax.set_title(f"{name} W/mK")
        ax.set_xlabel("x m")
        ax.set_ylabel("y m")
        fig.colorbar(handle, ax=ax, shrink=0.82)
    fig.suptitle(f"{sample['metadata'].get('probe_id')} diag3 k channels at source slice", fontsize=10)
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def _visualize_one(plt, sample_dir: Path, output_dir: Path, dpi: int) -> dict[str, Any]:
    sample = _load_sample(sample_dir)
    sample_id = sample["sample_id"]
    output_dir.mkdir(parents=True, exist_ok=True)
    z_mid = _z_mid(sample["coords"])
    z_source = _source_z(sample["coords"], sample["q_field"])
    paths = {
        "scatter_3d": output_dir / f"{sample_id}_3d_scatter.png",
        "z_mid_slice": output_dir / f"{sample_id}_zmid_slice.png",
        "source_slice": output_dir / f"{sample_id}_source_slice.png",
    }
    _scatter_3d(plt, sample, paths["scatter_3d"], dpi)
    _slice_figure(plt, sample, z_mid, paths["z_mid_slice"], dpi, "z-mid slice")
    _slice_figure(plt, sample, z_source, paths["source_slice"], dpi, "source-near slice")
    if sample["metadata"].get("probe_id") == "P09" and sample["k_field"].shape[1] == 3:
        paths["k_channels_source_slice"] = output_dir / f"{sample_id}_k_channels_source_slice.png"
        _p09_k_channels_figure(plt, sample, paths["k_channels_source_slice"], dpi)
    return {
        "sample_id": sample_id,
        "probe_id": sample["metadata"].get("probe_id"),
        "z_mid_m": z_mid,
        "z_source_m": z_source,
        "figures": {key: str(path) for key, path in paths.items()},
    }


def main() -> int:
    args = parse_args()
    sample_dirs = _sample_dirs(args.subset, args.sample_id)
    if not sample_dirs:
        raise FileNotFoundError(f"no probe samples found under {args.subset}")
    plt = _configure_matplotlib(args.output_dir)
    rows = [_visualize_one(plt, sample_dir, args.output_dir, args.dpi) for sample_dir in sample_dirs]
    payload = {
        "subset": str(args.subset),
        "output_dir": str(args.output_dir),
        "sample_count": len(rows),
        "rows": rows,
    }
    _write_json(args.output_dir / "figure_manifest.json", payload)
    print("Heat3D v3 final-target probe visualization")
    print(f"subset: {args.subset}")
    print(f"output_dir: {args.output_dir}")
    print(f"sample_count: {len(rows)}")
    for row in rows:
        print(f"- {row['sample_id']} probe={row['probe_id']} figures={len(row['figures'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
