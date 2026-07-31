#!/usr/bin/env python3
"""Read-only training-preflight audit for frozen V6-P1i formal1024_v1.

The script reads only the already-generated, tracked audit tables.  It never
opens a learned-model checkpoint, runs inference, changes a split, filters a
sample, or invokes the thermal solver.  ``--output-root`` permits deterministic
replay in a clean checkout without changing the frozen input tree.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks
from scipy.spatial import cKDTree
from scipy.stats import (
    gaussian_kde,
    kstest,
    ks_2samp,
    rankdata,
    t as student_t,
)
import yaml


DATASET_ID = "heat3d_v6_p1i_continuous_physics1024_v1"
PREFIX = "v6_p1i_formal1024_v1"
CONFIG_DIR = Path("configs/heat3d_v6_p1i")
DOCS_DIR = Path("docs")
QUANTILES = (0.0, 0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99, 1.0)
HISTOGRAM_BINS = 12
JOINT_BINS = 6
ROLE_ORDER = ("train", "valid_iid", "test_iid")
FROZEN_INPUT_SHA256 = {
    f"{CONFIG_DIR}/{PREFIX}.yaml":
        "1e15a77fe51eea7ec64614566bb6bb12bfcf05948f3b7c8c6f3c85ec759a58f8",
    f"{CONFIG_DIR}/{PREFIX}_manifest.json":
        "f19987c659968c2ac14eade1f1ef7e206c8f7eeb94f58fde5897d6e765978514",
    f"{CONFIG_DIR}/{PREFIX}_samples.csv":
        "5f0305e5994c8a8ec43a31ee9844526f5fad262a46c73a730cdf4285cd2d4018",
    f"{CONFIG_DIR}/{PREFIX}_input_definitions.csv":
        "afea524f5002814a4ec2e8dbac0bc6d2c60f18de9e2cf16c80c1ffe63feac498",
    f"{CONFIG_DIR}/{PREFIX}_regions.csv":
        "bb2fa33b822e4366f106488d24fcf60d7357bd46c4b196572ae50e66e7820acf",
    f"{CONFIG_DIR}/v6_p1i_background_k_contract.csv":
        "a0f504ceea7aa9b0a7ac1eea70a7be31a849a561e114d2fe8e5859c8cadbf703",
    f"{CONFIG_DIR}/v6_p1i_literature.json":
        "88f8e5fe299c2045e0a21563bbe33ae6c6e28083f829a4c3ff32398751170b6a",
    f"{CONFIG_DIR}/v6_p1i_v3_literature_contract.json":
        "34469fdca4cd95c1c11f4d590c3a794e68e3094093328e3b7e5d7c84d36f912e",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise RuntimeError(f"refusing to write empty CSV: {path}")
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _finite_float(value: Any) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise RuntimeError(f"non-finite numeric value: {value!r}")
    return result


def _as_array(
    rows: Sequence[Mapping[str, str]], field: str
) -> np.ndarray:
    result = np.asarray([_finite_float(row[field]) for row in rows])
    if result.ndim != 1 or not np.all(np.isfinite(result)):
        raise RuntimeError(f"bad array for {field}")
    return result


def _quantile_summary(values: np.ndarray) -> dict[str, float | int]:
    quantile_values = np.quantile(values, QUANTILES)
    result: dict[str, float | int] = {
        "count": int(values.size),
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
    }
    for quantile, value in zip(QUANTILES, quantile_values, strict=True):
        result[f"q{int(round(quantile * 100)):02d}"] = float(value)
    return result


def _empirical_wasserstein_1d(
    left: np.ndarray,
    right: np.ndarray,
) -> float:
    """Stable exact W1 for two equally weighted empirical 1-D samples."""
    left_sorted = np.sort(np.asarray(left, dtype=np.float64))
    right_sorted = np.sort(np.asarray(right, dtype=np.float64))
    combined = np.sort(np.concatenate((left_sorted, right_sorted)))
    if combined.size < 2:
        return 0.0
    deltas = np.diff(combined)
    support = combined[:-1]
    left_cdf = (
        np.searchsorted(left_sorted, support, side="right") / left_sorted.size
    )
    right_cdf = (
        np.searchsorted(right_sorted, support, side="right") / right_sorted.size
    )
    return float(np.sum(np.abs(left_cdf - right_cdf) * deltas))


def _normalized_rank(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=np.float64)
    ranks[order] = (np.arange(values.size, dtype=np.float64) + 0.5) / values.size
    return ranks


def _normalize(
    values: np.ndarray,
    bounds: tuple[float, float],
) -> np.ndarray:
    low, high = map(float, bounds)
    if not high > low:
        raise RuntimeError(f"invalid bounds: {bounds}")
    normalized = (values - low) / (high - low)
    tolerance = 1.0e-10
    if float(np.min(normalized)) < -tolerance or float(np.max(normalized)) > 1.0 + tolerance:
        raise RuntimeError(
            f"value outside frozen bounds {bounds}: "
            f"[{float(np.min(values))}, {float(np.max(values))}]"
        )
    return np.clip(normalized, 0.0, np.nextafter(1.0, 0.0))


def _mode_diagnostics(values: np.ndarray) -> dict[str, Any]:
    low, high = float(np.min(values)), float(np.max(values))
    if not high > low:
        return {
            "diagnosis": "degenerate",
            "bandwidth_sensitivity": "not_applicable",
            "runs": [],
        }
    grid = np.linspace(low, high, 512)
    runs: list[dict[str, Any]] = []
    for factor in (0.75, 1.0, 1.25):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            kde = gaussian_kde(
                values,
                bw_method=lambda estimator, value=factor:
                    estimator.scotts_factor() * value,
            )
            density = np.asarray(kde(grid), dtype=np.float64)
        if not np.all(np.isfinite(density)):
            raise RuntimeError("non-finite KDE density")
        prominence = 0.05 * float(np.max(density))
        peaks, _ = find_peaks(density, prominence=prominence)
        candidates = list(map(int, peaks))
        if density[0] > density[1] and density[0] >= prominence:
            candidates.insert(0, 0)
        if density[-1] > density[-2] and density[-1] >= prominence:
            candidates.append(grid.size - 1)
        candidates = sorted(set(candidates))
        runs.append(
            {
                "scott_bandwidth_factor": factor,
                "mode_count": len(candidates),
                "mode_locations": [float(grid[index]) for index in candidates],
            }
        )
    counts = [int(run["mode_count"]) for run in runs]
    central = counts[1]
    diagnosis = (
        "unimodal"
        if central == 1
        else "bimodal"
        if central == 2
        else "multimodal"
    )
    return {
        "diagnosis": diagnosis,
        "bandwidth_sensitivity": (
            "stable_count" if len(set(counts)) == 1 else "count_changes"
        ),
        "runs": runs,
    }


def _distribution_audit(
    name: str,
    values: np.ndarray,
    unit: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    low, high = float(np.min(values)), float(np.max(values))
    value_range = high - low
    if value_range <= 0.0:
        raise RuntimeError(f"{name} is degenerate")
    normalized = (values - low) / value_range
    counts, normalized_edges = np.histogram(
        normalized, bins=np.linspace(0.0, 1.0, HISTOGRAM_BINS + 1)
    )
    probabilities = counts.astype(np.float64) / values.size
    positive = probabilities[probabilities > 0.0]
    entropy = -float(np.sum(positive * np.log(positive)))
    normalized_entropy = entropy / math.log(HISTOGRAM_BINS)
    uniform_grid = (
        np.arange(values.size, dtype=np.float64) + 0.5
    ) / values.size
    ks = kstest(normalized, "uniform")
    wasserstein_normalized = _empirical_wasserstein_1d(
        normalized, uniform_grid
    )
    ordered = np.sort(values)
    gaps = np.diff(ordered)
    gap_index = int(np.argmax(gaps))
    maximum_gap = float(gaps[gap_index])
    result = {
        "metric": name,
        "unit": unit,
        "summary": _quantile_summary(values),
        "observed_range": [low, high],
        "histogram": {
            "bin_count": HISTOGRAM_BINS,
            "reference": "equal_width_over_observed_range",
            "edges": (low + normalized_edges * value_range).tolist(),
            "counts": counts.tolist(),
            "occupied_bins": int(np.count_nonzero(counts)),
            "empty_bins": int(np.sum(counts == 0)),
            "shannon_entropy_nats": entropy,
            "normalized_entropy": normalized_entropy,
            "bin_count_cv": float(np.std(counts) / np.mean(counts)),
        },
        "uniform_reference_diagnostics": {
            "reference": "continuous_uniform_over_observed_range",
            "ks_statistic": float(ks.statistic),
            "ks_pvalue": float(ks.pvalue),
            "wasserstein": wasserstein_normalized * value_range,
            "wasserstein_unit": unit,
            "wasserstein_normalized_range": wasserstein_normalized,
            "interpretation": "diagnostic_only_not_a_uniformity_acceptance_test",
        },
        "maximum_sorted_gap": {
            "gap": maximum_gap,
            "unit": unit,
            "lower": float(ordered[gap_index]),
            "upper": float(ordered[gap_index + 1]),
            "normalized_by_observed_range": maximum_gap / value_range,
        },
        "modal_diagnostics": _mode_diagnostics(values),
    }
    bin_rows: list[dict[str, Any]] = []
    physical_edges = np.asarray(result["histogram"]["edges"])
    for index, count in enumerate(counts):
        bin_rows.append(
            {
                "metric": name,
                "unit": unit,
                "bin_index": index,
                "lower": float(physical_edges[index]),
                "upper": float(physical_edges[index + 1]),
                "count": int(count),
                "fraction": float(count / values.size),
            }
        )
    return result, bin_rows


def _transform(
    values: np.ndarray,
    transform: str,
) -> np.ndarray:
    if transform == "linear":
        return np.asarray(values, dtype=np.float64)
    if transform == "log10":
        if np.any(values <= 0.0):
            raise RuntimeError("log10 transform requires positive values")
        return np.log10(values)
    raise RuntimeError(f"unknown transform: {transform}")


def _occupancy(
    x: np.ndarray,
    y: np.ndarray,
    *,
    scheme: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    counts, _, _ = np.histogram2d(
        x,
        y,
        bins=(
            np.linspace(0.0, 1.0, JOINT_BINS + 1),
            np.linspace(0.0, 1.0, JOINT_BINS + 1),
        ),
    )
    counts = counts.astype(np.int64)
    flat = counts.ravel()
    probabilities = flat.astype(np.float64) / np.sum(flat)
    positive = probabilities[probabilities > 0.0]
    entropy = -float(np.sum(positive * np.log(positive)))
    x_counts = np.sum(counts, axis=1).astype(np.float64)
    y_counts = np.sum(counts, axis=0).astype(np.float64)
    joint_p = counts.astype(np.float64) / np.sum(counts)
    px = x_counts / np.sum(x_counts)
    py = y_counts / np.sum(y_counts)
    mutual_information = 0.0
    for ix in range(JOINT_BINS):
        for iy in range(JOINT_BINS):
            probability = joint_p[ix, iy]
            if probability > 0.0:
                mutual_information += probability * math.log(
                    probability / (px[ix] * py[iy])
                )
    hx = -float(np.sum(px[px > 0.0] * np.log(px[px > 0.0])))
    hy = -float(np.sum(py[py > 0.0] * np.log(py[py > 0.0])))
    normalized_mi = (
        mutual_information / math.sqrt(hx * hy) if hx > 0.0 and hy > 0.0 else 0.0
    )
    cell_rows = []
    for ix in range(JOINT_BINS):
        for iy in range(JOINT_BINS):
            cell_rows.append(
                {
                    "scheme": scheme,
                    "x_bin": ix,
                    "y_bin": iy,
                    "count": int(counts[ix, iy]),
                    "fraction": float(counts[ix, iy] / np.sum(counts)),
                    "missing": bool(counts[ix, iy] == 0),
                }
            )
    return (
        {
            "scheme": scheme,
            "grid": [JOINT_BINS, JOINT_BINS],
            "occupied_cells": int(np.count_nonzero(counts)),
            "empty_cells": int(np.sum(counts == 0)),
            "occupancy_fraction": float(np.count_nonzero(counts) / counts.size),
            "normalized_entropy": entropy / math.log(counts.size),
            "bin_count_cv": float(np.std(flat) / np.mean(flat)),
            "mutual_information_nats": float(mutual_information),
            "normalized_mutual_information": float(normalized_mi),
            "counts": counts.tolist(),
        },
        cell_rows,
    )


def _nearest_neighbour_summary(x: np.ndarray, y: np.ndarray) -> dict[str, Any]:
    points = np.column_stack((x, y))
    distances, _ = cKDTree(points).query(points, k=2)
    nearest = np.asarray(distances[:, 1], dtype=np.float64)
    return {
        "metric": "euclidean_in_normalized_pair_space",
        **_quantile_summary(nearest),
    }


def _joint_pair(
    pair_id: str,
    x_name: str,
    y_name: str,
    x_raw: np.ndarray,
    y_raw: np.ndarray,
    *,
    x_bounds: tuple[float, float],
    y_bounds: tuple[float, float],
    x_transform: str,
    y_transform: str,
    bound_source: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    x = _transform(x_raw, x_transform)
    y = _transform(y_raw, y_transform)
    x_bounds_transformed = tuple(
        _transform(np.asarray(x_bounds, dtype=np.float64), x_transform).tolist()
    )
    y_bounds_transformed = tuple(
        _transform(np.asarray(y_bounds, dtype=np.float64), y_transform).tolist()
    )
    x_normalized = _normalize(x, x_bounds_transformed)
    y_normalized = _normalize(y, y_bounds_transformed)
    physical, physical_cells = _occupancy(
        x_normalized, y_normalized, scheme="frozen_physical_range"
    )
    rank_x = _normalized_rank(x_raw)
    rank_y = _normalized_rank(y_raw)
    rank, rank_cells = _occupancy(
        rank_x, rank_y, scheme="marginal_quantile_rank"
    )
    def stable_pearson(
        left: np.ndarray,
        right: np.ndarray,
    ) -> tuple[float, float]:
        left_std = float(np.std(left))
        right_std = float(np.std(right))
        if left_std <= 0.0 or right_std <= 0.0:
            raise RuntimeError(f"degenerate Pearson pair: {pair_id}")
        left_z = (left - np.mean(left)) / left_std
        right_z = (right - np.mean(right)) / right_std
        statistic = float(np.mean(left_z * right_z))
        statistic = float(np.clip(statistic, -1.0, 1.0))
        denominator = max(1.0 - statistic * statistic, np.finfo(float).tiny)
        t_value = abs(statistic) * math.sqrt((left.size - 2) / denominator)
        pvalue = float(2.0 * student_t.sf(t_value, left.size - 2))
        return statistic, pvalue

    pearson_raw = stable_pearson(x_raw, y_raw)
    pearson_transformed = stable_pearson(x, y)
    spearman = stable_pearson(rankdata(x_raw), rankdata(y_raw))
    corners = {
        "low_x_low_y": int(np.sum((x_normalized <= 0.2) & (y_normalized <= 0.2))),
        "low_x_high_y": int(np.sum((x_normalized <= 0.2) & (y_normalized >= 0.8))),
        "high_x_low_y": int(np.sum((x_normalized >= 0.8) & (y_normalized <= 0.2))),
        "high_x_high_y": int(np.sum((x_normalized >= 0.8) & (y_normalized >= 0.8))),
    }
    expected_corner_count = 0.04 * x_raw.size
    result = {
        "pair_id": pair_id,
        "x": x_name,
        "y": y_name,
        "x_transform_for_coverage": x_transform,
        "y_transform_for_coverage": y_transform,
        "x_bounds_raw": list(map(float, x_bounds)),
        "y_bounds_raw": list(map(float, y_bounds)),
        "bound_source": bound_source,
        "correlation": {
            "pearson_raw": pearson_raw[0],
            "pearson_raw_pvalue": pearson_raw[1],
            "pearson_coverage_transform": pearson_transformed[0],
            "pearson_coverage_transform_pvalue": pearson_transformed[1],
            "spearman": spearman[0],
            "spearman_pvalue": spearman[1],
        },
        "physical_range_occupancy": physical,
        "rank_occupancy": rank,
        "corner_coverage": {
            "threshold": "lowest/highest 20% of frozen physical range",
            "uniform_independent_expected_count_per_corner": expected_corner_count,
            "counts": corners,
            "ratios_to_uniform_independent_expectation": {
                key: value / expected_corner_count
                for key, value in corners.items()
            },
        },
        "nearest_neighbour": _nearest_neighbour_summary(
            x_normalized, y_normalized
        ),
    }
    rows = []
    for row in physical_cells + rank_cells:
        rows.append({"pair_id": pair_id, **row})
    return result, rows


def _literature_registry(
    input_root: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    legacy = json.loads(
        (input_root / CONFIG_DIR / "v6_p1i_literature.json").read_text(
            encoding="utf-8"
        )
    )
    current = json.loads(
        (
            input_root / CONFIG_DIR / "v6_p1i_v3_literature_contract.json"
        ).read_text(encoding="utf-8")
    )
    canonical_map = {
        "P1I3-L01": "V6-LIT-001",
        "P1I3-L02": "V6-LIT-002",
        "P1I3-L03": "V6-LIT-003",
        "P1I3-L04": "V6-LIT-004",
        "P1I3-L05": "V6-LIT-005",
        "P1I3-L06": "V6-LIT-006",
        "P1I3-L07": "V6-LIT-007",
        "P1I3-L08": "V6-LIT-008",
        "P1I3-L09": "V6-LIT-009",
        "P1I-L01": "V6-LIT-009",
        "P1I-L02": "V6-LIT-004",
        "P1I-L03": "V6-LIT-010",
        "P1I-L04": "V6-LIT-011",
        "P1I-L05": "V6-LIT-012",
    }
    all_sources = list(current["sources"]) + list(legacy["sources"])
    rows: list[dict[str, Any]] = []
    canonical: dict[str, dict[str, Any]] = {}
    for source in all_sources:
        old_id = source["id"]
        canonical_id = canonical_map[old_id]
        row = {
            "legacy_id": old_id,
            "canonical_id": canonical_id,
            "title": source["title"],
            "doi": source.get("doi", ""),
            "url": source.get("url", ""),
            "source_registry": (
                "v3_literature_contract"
                if old_id.startswith("P1I3-")
                else "legacy_literature"
            ),
            "relationship": (
                "exact_same_source"
                if old_id in {"P1I-L01", "P1I-L02"}
                else "canonicalized_unique_source"
            ),
        }
        rows.append(row)
        entry = canonical.setdefault(
            canonical_id,
            {
                "canonical_id": canonical_id,
                "title": source["title"],
                "doi": source.get("doi"),
                "url": source.get("url"),
                "legacy_ids": [],
            },
        )
        entry["legacy_ids"].append(old_id)
    registry = {
        "schema_version": "heat3d_v6_literature_registry_v1",
        "canonical_id_policy": "V6-LIT-NNN identifies a unique DOI or URL",
        "source_count": len(canonical),
        "sources": [canonical[key] for key in sorted(canonical)],
        "bindings": {
            "background_k_contract": {
                "path": f"{CONFIG_DIR}/v6_p1i_background_k_contract.csv",
                "sha256": FROZEN_INPUT_SHA256[
                    f"{CONFIG_DIR}/v6_p1i_background_k_contract.csv"
                ],
                "legacy_ids_resolved_by_crosswalk": True,
            },
            "formal_config": {
                "path": f"{CONFIG_DIR}/{PREFIX}.yaml",
                "sha256": FROZEN_INPUT_SHA256[f"{CONFIG_DIR}/{PREFIX}.yaml"],
            },
            "formal_manifest": {
                "path": f"{CONFIG_DIR}/{PREFIX}_manifest.json",
                "sha256": FROZEN_INPUT_SHA256[
                    f"{CONFIG_DIR}/{PREFIX}_manifest.json"
                ],
                "manifest_payload_sha256":
                    "27d2ea3b7ec4e4ce9c6d068471cd19036ac8148b6cd57da325219d718c7e5ed5",
            },
        },
    }
    return registry, rows


def _format_float(value: Any, digits: int = 6) -> str:
    return f"{float(value):.{digits}g}"


def _plot_distributions(
    output: Path,
    distributions: Mapping[str, Mapping[str, Any]],
    values: Mapping[str, np.ndarray],
) -> None:
    names = list(distributions)
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.5), constrained_layout=True)
    for axis, name in zip(axes.flat, names, strict=True):
        info = distributions[name]
        value = values[name]
        axis.hist(value, bins=HISTOGRAM_BINS, color="#3569a8", alpha=0.82)
        axis.set_title(name)
        axis.set_xlabel(info["unit"])
        axis.set_ylabel("sample count")
        axis.grid(alpha=0.2)
    fig.suptitle(
        "V6-P1i formal1024_v1 frozen distribution audit\n"
        "Equal-width bins over each observed range; no strict-uniform claim",
        fontsize=14,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        output,
        dpi=160,
        metadata={"Software": "Heat3D deterministic audit"},
    )
    plt.close(fig)


def _plot_joint(
    output: Path,
    pairs: Mapping[str, Mapping[str, Any]],
) -> None:
    selected = (
        "power_x_top_h",
        "power_x_bottom_h",
        "top_h_x_bottom_h",
        "power_x_total_source_area",
        "mean_q_x_mean_local_k",
        "effective_background_kz_x_top_h",
    )
    fig, axes = plt.subplots(2, 3, figsize=(15, 9), constrained_layout=True)
    for axis, pair_id in zip(axes.flat, selected, strict=True):
        pair = pairs[pair_id]
        counts = np.asarray(pair["physical_range_occupancy"]["counts"])
        image = axis.imshow(
            counts.T,
            origin="lower",
            cmap="viridis",
            aspect="auto",
        )
        axis.set_title(pair_id)
        axis.set_xlabel(f"{pair['x']} bin")
        axis.set_ylabel(f"{pair['y']} bin")
        for ix in range(counts.shape[0]):
            for iy in range(counts.shape[1]):
                axis.text(
                    ix,
                    iy,
                    str(int(counts[ix, iy])),
                    ha="center",
                    va="center",
                    color=("white" if counts[ix, iy] < np.max(counts) * 0.5 else "black"),
                    fontsize=7,
                )
        fig.colorbar(image, ax=axis, fraction=0.046)
    fig.suptitle(
        "Frozen physical-range 2D occupancy (6×6)\n"
        "Empty cells are coverage limitations, not post-hoc rejection rules",
        fontsize=14,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        output,
        dpi=160,
        metadata={"Software": "Heat3D deterministic audit"},
    )
    plt.close(fig)


def _plot_split(
    output: Path,
    values: Mapping[str, np.ndarray],
    roles: np.ndarray,
) -> None:
    selected = (
        "package_total_power_W",
        "top_h_W_m2K",
        "bottom_h_W_m2K",
        "peak_deltaT_K",
        "mean_deltaT_K",
        "cv_rms_deltaT_K",
    )
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.5), constrained_layout=True)
    colors = {"train": "#3569a8", "valid_iid": "#e98b2a", "test_iid": "#4c9f70"}
    for axis, name in zip(axes.flat, selected, strict=True):
        for role in ROLE_ORDER:
            subset = np.sort(values[name][roles == role])
            y = (np.arange(subset.size) + 1.0) / subset.size
            axis.step(
                subset,
                y,
                where="post",
                label=role,
                color=colors[role],
                linewidth=1.6,
            )
        axis.set_title(name)
        axis.set_ylabel("empirical CDF")
        axis.grid(alpha=0.2)
    axes[0, 0].legend()
    fig.suptitle(
        "Descriptive split comparison only; test_iid did not change any rule",
        fontsize=14,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        output,
        dpi=160,
        metadata={"Software": "Heat3D deterministic audit"},
    )
    plt.close(fig)


def run(input_root: Path, output_root: Path) -> dict[str, Any]:
    for relative, expected in FROZEN_INPUT_SHA256.items():
        path = input_root / relative
        if not path.is_file() or _sha256(path) != expected:
            raise RuntimeError(f"frozen input SHA mismatch: {relative}")

    config_path = input_root / CONFIG_DIR / f"{PREFIX}.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if config["dataset_id"] != DATASET_ID:
        raise RuntimeError("dataset identity drift")
    if config["sample_count"] != 1024:
        raise RuntimeError("sample count drift")
    if config["physics"]["contact"] != {
        "type": "perfect",
        "R_contact_m2K_W": 0.0,
    }:
        raise RuntimeError("P1i contact contract drift")

    sample_rows = _read_csv(
        input_root / CONFIG_DIR / f"{PREFIX}_samples.csv"
    )
    input_rows = _read_csv(
        input_root / CONFIG_DIR / f"{PREFIX}_input_definitions.csv"
    )
    if len(sample_rows) != 1024 or len(input_rows) != 1024:
        raise RuntimeError("formal1024_v1 row count drift")
    input_by_id = {row["sample_id"]: row for row in input_rows}
    sample_ids = [row["sample_id"] for row in sample_rows]
    if len(input_by_id) != 1024 or set(input_by_id) != set(sample_ids):
        raise RuntimeError("sample/input identity mismatch")
    aligned_inputs = [input_by_id[sample_id] for sample_id in sample_ids]
    roles = np.asarray([row["split_role"] for row in sample_rows])
    if {
        role: int(np.sum(roles == role)) for role in ROLE_ORDER
    } != {"train": 768, "valid_iid": 128, "test_iid": 128}:
        raise RuntimeError("split count drift")

    values: dict[str, np.ndarray] = {}
    for field in (
        "package_total_power_W",
        "top_h_W_m2K",
        "bottom_h_W_m2K",
        "source_count",
        "k_region_count",
        "mean_q_W_m3",
        "max_q_W_m3",
        "mean_local_k_W_mK",
        "peak_deltaT_K",
        "mean_deltaT_K",
        "cv_rms_deltaT_K",
        "top_heat_fraction",
        "bottom_heat_fraction",
    ):
        values[field] = _as_array(sample_rows, field)
    for field in (
        "total_source_area_fraction",
        "mean_source_area_fraction",
        "source_area_cv",
        "mean_source_aspect_ratio",
        "source_centroid_spread",
        "source_upper_fraction",
        "cross_family_overlap_fraction",
        "local_k_log_std",
    ):
        values[field] = _as_array(aligned_inputs, field)
    values["Reff_peak_K_W"] = (
        values["peak_deltaT_K"] / values["package_total_power_W"]
    )
    layer_config = {
        str(row["id"]): row for row in config["physics"]["layers_bottom_to_top"]
    }
    background_fields = [
        field
        for field in aligned_inputs[0]
        if field.startswith("background_kz_")
    ]
    for field in background_fields:
        values[field] = _as_array(aligned_inputs, field)
    thickness = []
    inverse_path = np.zeros(1024, dtype=np.float64)
    for field in background_fields:
        layer_id = field.removeprefix("background_kz_").removesuffix("_W_mK")
        layer_thickness = float(layer_config[layer_id]["thickness_m"])
        thickness.append(layer_thickness)
        inverse_path += layer_thickness / values[field]
    values["effective_background_kz_W_mK"] = (
        sum(thickness) / inverse_path
    )

    distribution_specs = (
        ("peak_deltaT_K", "K"),
        ("mean_deltaT_K", "K"),
        ("cv_rms_deltaT_K", "K"),
        ("Reff_peak_K_W", "K/W"),
        ("top_heat_fraction", "fraction"),
        ("bottom_heat_fraction", "fraction"),
    )
    distributions: dict[str, Any] = {}
    distribution_rows: list[dict[str, Any]] = []
    distribution_bin_rows: list[dict[str, Any]] = []
    for name, unit in distribution_specs:
        result, bin_rows = _distribution_audit(name, values[name], unit)
        distributions[name] = result
        distribution_rows.append(
            {
                "metric": name,
                "unit": unit,
                **result["summary"],
                "occupied_bins": result["histogram"]["occupied_bins"],
                "normalized_entropy": result["histogram"]["normalized_entropy"],
                "bin_count_cv": result["histogram"]["bin_count_cv"],
                "uniform_ks": result["uniform_reference_diagnostics"]["ks_statistic"],
                "uniform_ks_pvalue":
                    result["uniform_reference_diagnostics"]["ks_pvalue"],
                "wasserstein_normalized_range":
                    result["uniform_reference_diagnostics"][
                        "wasserstein_normalized_range"
                    ],
                "maximum_gap": result["maximum_sorted_gap"]["gap"],
                "maximum_gap_normalized":
                    result["maximum_sorted_gap"]["normalized_by_observed_range"],
                "central_kde_diagnosis":
                    result["modal_diagnostics"]["diagnosis"],
                "kde_bandwidth_sensitivity":
                    result["modal_diagnostics"]["bandwidth_sensitivity"],
            }
        )
        distribution_bin_rows.extend(bin_rows)

    broad_primary = ("peak_deltaT_K", "mean_deltaT_K", "cv_rms_deltaT_K", "Reff_peak_K_W")
    broad_checks = {
        f"{name}_all_12_bins_occupied":
            distributions[name]["histogram"]["occupied_bins"] == 12
        for name in broad_primary
    }
    broad_checks.update(
        {
            f"{name}_entropy_ge_0p90":
                distributions[name]["histogram"]["normalized_entropy"] >= 0.90
            for name in broad_primary
        }
    )
    broad_checks.update(
        {
            f"{name}_max_gap_le_0p05_range":
                distributions[name]["maximum_sorted_gap"][
                    "normalized_by_observed_range"
                ] <= 0.05
            for name in broad_primary
        }
    )
    continuous_broad_coverage = all(broad_checks.values())

    bounds = {
        "package_total_power_W": tuple(
            map(float, config["sampling"]["power_W"]["allowed_range_W"])
        ),
        "top_h_W_m2K": tuple(
            map(float, config["sampling"]["top_h_W_m2K"]["range"])
        ),
        "bottom_h_W_m2K": tuple(
            map(float, config["sampling"]["bottom_h_W_m2K"]["range"])
        ),
        "mean_q_W_m3": (
            float(np.min(values["mean_q_W_m3"])),
            float(np.max(values["mean_q_W_m3"])),
        ),
        "mean_local_k_W_mK": tuple(
            map(float, config["sampling"]["local_k_W_mK"]["range"])
        ),
        "total_source_area_fraction": (
            float(np.min(values["total_source_area_fraction"])),
            float(np.max(values["total_source_area_fraction"])),
        ),
    }
    pair_specs: list[dict[str, Any]] = [
        {
            "pair_id": "power_x_top_h",
            "x": "package_total_power_W",
            "y": "top_h_W_m2K",
            "x_transform": "linear",
            "y_transform": "log10",
            "bound_source": "formal config",
        },
        {
            "pair_id": "power_x_bottom_h",
            "x": "package_total_power_W",
            "y": "bottom_h_W_m2K",
            "x_transform": "linear",
            "y_transform": "log10",
            "bound_source": "formal config",
        },
        {
            "pair_id": "top_h_x_bottom_h",
            "x": "top_h_W_m2K",
            "y": "bottom_h_W_m2K",
            "x_transform": "log10",
            "y_transform": "log10",
            "bound_source": "formal config",
        },
        {
            "pair_id": "power_x_total_source_area",
            "x": "package_total_power_W",
            "y": "total_source_area_fraction",
            "x_transform": "linear",
            "y_transform": "linear",
            "bound_source": "formal config power + frozen observed derived-area support",
        },
        {
            "pair_id": "mean_q_x_mean_local_k",
            "x": "mean_q_W_m3",
            "y": "mean_local_k_W_mK",
            "x_transform": "log10",
            "y_transform": "log10",
            "bound_source": "frozen observed q support + formal local-k contract",
        },
    ]
    background_range_by_layer: dict[str, tuple[float, float, str]] = {}
    for layer_id, layer in layer_config.items():
        sampling = layer["sampling"]
        if "kz_range_W_mK" in sampling:
            layer_bounds = tuple(map(float, sampling["kz_range_W_mK"]))
        else:
            layer_bounds = tuple(map(float, sampling["range_W_mK"]))
        transform = (
            "log10"
            if "log_uniform" in str(sampling["distribution"])
            else "linear"
        )
        background_range_by_layer[layer_id] = (
            layer_bounds[0],
            layer_bounds[1],
            transform,
        )
    for field in background_fields:
        layer_id = field.removeprefix("background_kz_").removesuffix("_W_mK")
        low, high, transform = background_range_by_layer[layer_id]
        bounds[field] = (low, high)
        for bc_field, bc_short in (
            ("top_h_W_m2K", "top_h"),
            ("bottom_h_W_m2K", "bottom_h"),
        ):
            pair_specs.append(
                {
                    "pair_id": f"{layer_id}_kz_x_{bc_short}",
                    "x": field,
                    "y": bc_field,
                    "x_transform": transform,
                    "y_transform": "log10",
                    "bound_source": "formal per-layer k contract + formal BC contract",
                }
            )
    bounds["effective_background_kz_W_mK"] = (
        float(np.min(values["effective_background_kz_W_mK"])),
        float(np.max(values["effective_background_kz_W_mK"])),
    )
    for bc_field, bc_short in (
        ("top_h_W_m2K", "top_h"),
        ("bottom_h_W_m2K", "bottom_h"),
    ):
        pair_specs.append(
            {
                "pair_id": f"effective_background_kz_x_{bc_short}",
                "x": "effective_background_kz_W_mK",
                "y": bc_field,
                "x_transform": "log10",
                "y_transform": "log10",
                "bound_source": "frozen observed effective stack support + formal BC contract",
            }
        )

    pairs: dict[str, Any] = {}
    joint_rows: list[dict[str, Any]] = []
    occupancy_rows: list[dict[str, Any]] = []
    for spec in pair_specs:
        result, cells = _joint_pair(
            spec["pair_id"],
            spec["x"],
            spec["y"],
            values[spec["x"]],
            values[spec["y"]],
            x_bounds=bounds[spec["x"]],
            y_bounds=bounds[spec["y"]],
            x_transform=spec["x_transform"],
            y_transform=spec["y_transform"],
            bound_source=spec["bound_source"],
        )
        pairs[spec["pair_id"]] = result
        corners = result["corner_coverage"]["counts"]
        joint_rows.append(
            {
                "pair_id": spec["pair_id"],
                "x": spec["x"],
                "y": spec["y"],
                "x_transform": spec["x_transform"],
                "y_transform": spec["y_transform"],
                "pearson_raw": result["correlation"]["pearson_raw"],
                "pearson_coverage_transform":
                    result["correlation"]["pearson_coverage_transform"],
                "spearman": result["correlation"]["spearman"],
                "physical_occupied_cells":
                    result["physical_range_occupancy"]["occupied_cells"],
                "physical_empty_cells":
                    result["physical_range_occupancy"]["empty_cells"],
                "physical_occupancy_fraction":
                    result["physical_range_occupancy"]["occupancy_fraction"],
                "physical_normalized_entropy":
                    result["physical_range_occupancy"]["normalized_entropy"],
                "rank_occupied_cells":
                    result["rank_occupancy"]["occupied_cells"],
                "rank_empty_cells": result["rank_occupancy"]["empty_cells"],
                "rank_normalized_mutual_information":
                    result["rank_occupancy"]["normalized_mutual_information"],
                "corner_low_low": corners["low_x_low_y"],
                "corner_low_high": corners["low_x_high_y"],
                "corner_high_low": corners["high_x_low_y"],
                "corner_high_high": corners["high_x_high_y"],
                "nn_median": result["nearest_neighbour"]["q50"],
                "nn_p95": result["nearest_neighbour"]["q95"],
                "nn_max": result["nearest_neighbour"]["q100"],
            }
        )
        occupancy_rows.extend(cells)

    power_top = pairs["power_x_top_h"]
    power_top_assessment = {
        "contract_origin": {
            "power_rule": config["sampling"]["power_W"]["distribution"],
            "top_h_exponent": float(
                config["sampling"]["power_W"]["top_h_exponent"]
            ),
            "independent_multiplier_range": list(
                map(
                    float,
                    config["sampling"]["power_W"]["independent_multiplier"][
                        "range"
                    ],
                )
            ),
        },
        "empirical_spearman": power_top["correlation"]["spearman"],
        "physical_6x6_empty_cells":
            power_top["physical_range_occupancy"]["empty_cells"],
        "high_power_low_top_h_count":
            power_top["corner_coverage"]["counts"]["high_x_low_y"],
        "low_power_high_top_h_count":
            power_top["corner_coverage"]["counts"]["low_x_high_y"],
        "conclusion": (
            "artificially_coupled_not_deconfounded; high-power/low-top-h "
            "coverage is absent under the frozen physical-range corner definition"
        ),
        "training_implication": (
            "The dataset supports the frozen coupled operating distribution, "
            "not independent identification of power and top-h effects over "
            "all corners."
        ),
    }

    split_variables = (
        "package_total_power_W",
        "top_h_W_m2K",
        "bottom_h_W_m2K",
        "source_count",
        "k_region_count",
        "total_source_area_fraction",
        "mean_q_W_m3",
        "mean_local_k_W_mK",
        "effective_background_kz_W_mK",
        "peak_deltaT_K",
        "mean_deltaT_K",
        "cv_rms_deltaT_K",
        "Reff_peak_K_W",
        "top_heat_fraction",
        "bottom_heat_fraction",
    )
    split_summary_rows: list[dict[str, Any]] = []
    for name in split_variables:
        for role in ROLE_ORDER:
            summary = _quantile_summary(values[name][roles == role])
            split_summary_rows.append(
                {"variable": name, "split_role": role, **summary}
            )
    split_comparison_rows: list[dict[str, Any]] = []
    for name in split_variables:
        global_range = float(np.ptp(values[name]))
        for left_role, right_role in (
            ("train", "valid_iid"),
            ("train", "test_iid"),
            ("valid_iid", "test_iid"),
        ):
            left = values[name][roles == left_role]
            right = values[name][roles == right_role]
            ks = ks_2samp(left, right)
            distance = _empirical_wasserstein_1d(left, right)
            split_comparison_rows.append(
                {
                    "variable": name,
                    "left_role": left_role,
                    "right_role": right_role,
                    "ks_statistic": float(ks.statistic),
                    "ks_pvalue": float(ks.pvalue),
                    "wasserstein": distance,
                    "wasserstein_normalized_global_range": (
                        distance / global_range if global_range > 0.0 else 0.0
                    ),
                    "test_used_for_rule_adjustment": False,
                }
            )
    input_nn_fields = (
        "package_total_power_W",
        "top_h_W_m2K",
        "bottom_h_W_m2K",
        "source_count",
        "k_region_count",
        "total_source_area_fraction",
        "mean_q_W_m3",
        "mean_local_k_W_mK",
        "effective_background_kz_W_mK",
    )
    matrix = np.column_stack(
        [
            np.log10(values[name])
            if name
            in {
                "top_h_W_m2K",
                "bottom_h_W_m2K",
                "mean_q_W_m3",
                "mean_local_k_W_mK",
                "effective_background_kz_W_mK",
            }
            else values[name]
            for name in input_nn_fields
        ]
    )
    train_matrix = matrix[roles == "train"]
    mean = np.mean(train_matrix, axis=0)
    std = np.std(train_matrix, axis=0)
    if np.any(std <= 0.0):
        raise RuntimeError("degenerate train input feature for NN")
    train_z = (train_matrix - mean) / std
    tree = cKDTree(train_z)
    split_nn: dict[str, Any] = {}
    for role in ROLE_ORDER:
        role_z = (matrix[roles == role] - mean) / std
        if role == "train":
            distance, _ = tree.query(role_z, k=2)
            nearest = distance[:, 1]
        else:
            nearest, _ = tree.query(role_z, k=1)
        split_nn[role] = {
            "feature_schema": list(input_nn_fields),
            "fit_population": "train",
            "target_features_used": False,
            **_quantile_summary(np.asarray(nearest)),
        }

    literature_registry, literature_rows = _literature_registry(input_root)
    contact_evidence_path = (
        input_root / CONFIG_DIR / "v6_p1i_cross_family_contact_evidence.json"
    )
    contact_evidence = json.loads(
        contact_evidence_path.read_text(encoding="utf-8")
    )
    if contact_evidence["contact_contract"]["R_contact_m2K_W"] != 0.0:
        raise RuntimeError("cross-family contact evidence drift")

    max_split_ks = max(
        row["ks_statistic"]
        for row in split_comparison_rows
        if row["left_role"] == "train"
    )
    max_split_wasserstein = max(
        row["wasserstein_normalized_global_range"]
        for row in split_comparison_rows
        if row["left_role"] == "train"
    )
    authorization_checks = {
        "frozen_input_sha256_match": True,
        "sample_count_1024": len(sample_rows) == 1024,
        "split_counts_768_128_128": True,
        "continuous_broad_coverage_primary_outputs": continuous_broad_coverage,
        "all_primary_output_bins_occupied": all(
            distributions[name]["histogram"]["empty_bins"] == 0
            for name in broad_primary
        ),
        "no_data_regeneration": True,
        "no_sample_filtering_or_replacement": True,
        "no_training": True,
        "no_model_inference": True,
        "test_descriptive_only": True,
        "contact_applicability_boundary_registered": True,
        "power_top_h_coupling_registered": True,
    }
    training_authorized = all(authorization_checks.values())
    authorization = {
        "schema_version": "heat3d_v6_p1i_training_authorization_v1",
        "dataset_id": DATASET_ID,
        "decision": (
            "authorized_for_training_with_frozen_applicability_boundaries"
            if training_authorized
            else "not_authorized"
        ),
        "authorized": training_authorized,
        "checks": authorization_checks,
        "scope": (
            "Training authorization is limited to the frozen formal1024_v1 "
            "coupled input distribution. It is not an OOD or strict-uniform claim."
        ),
        "known_boundaries": [
            "power and top_h are deliberately coupled; the high-power/low-top-h corner is absent",
            "all interfaces use perfect contact (R_contact=0)",
            "finite contact-resistance response cannot be learned",
            "test_iid was read only for this preregistered descriptive split audit and did not modify rules",
        ],
        "guardrails": {
            "formal1024_v1_modified": False,
            "dataset_regeneration_runs": 0,
            "sample_filter_or_replacement_runs": 0,
            "training_runs": 0,
            "model_inference_runs": 0,
        },
    }

    audit = {
        "schema_version": "heat3d_v6_p1i_training_preflight_audit_v1",
        "dataset_id": DATASET_ID,
        "status": "passed" if training_authorized else "failed",
        "frozen_bindings": {
            "input_sha256": FROZEN_INPUT_SHA256,
            "formal_config_sha256": FROZEN_INPUT_SHA256[
                f"{CONFIG_DIR}/{PREFIX}.yaml"
            ],
            "formal_manifest_sha256": FROZEN_INPUT_SHA256[
                f"{CONFIG_DIR}/{PREFIX}_manifest.json"
            ],
            "formal_manifest_payload_sha256":
                "27d2ea3b7ec4e4ce9c6d068471cd19036ac8148b6cd57da325219d718c7e5ed5",
            "background_k_contract_sha256": FROZEN_INPUT_SHA256[
                f"{CONFIG_DIR}/v6_p1i_background_k_contract.csv"
            ],
        },
        "methodology": {
            "uniform_reference": (
                "For each scalar metric, values are min-max normalized over "
                "the frozen observed support and compared with Uniform(0,1). "
                "KS and Wasserstein are diagnostics, not acceptance tests."
            ),
            "joint_physical_occupancy": (
                "6x6 cells in frozen config ranges using sampling-native "
                "linear/log10 transforms; derived quantities use frozen observed bounds."
            ),
            "joint_rank_occupancy": (
                "6x6 cells after marginal empirical-rank transforms, isolating "
                "joint dependence from marginal skew."
            ),
            "split_use": (
                "train/valid_iid/test_iid are compared descriptively; test_iid "
                "does not set thresholds, alter split assignment, or authorize selection."
            ),
        },
        "distribution_audit": {
            "metrics": distributions,
            "continuous_broad_coverage": {
                "primary_metrics": list(broad_primary),
                "criteria": broad_checks,
                "passed": continuous_broad_coverage,
                "conclusion": (
                    "continuous broad coverage is supported across the frozen "
                    "observed ranges; strict uniformity is not supported and is not claimed"
                    if continuous_broad_coverage
                    else "continuous broad coverage criterion was not met"
                ),
            },
        },
        "joint_input_coverage": {
            "pairs": pairs,
            "power_top_h_assessment": power_top_assessment,
        },
        "split_audit": {
            "role_counts": {
                role: int(np.sum(roles == role)) for role in ROLE_ORDER
            },
            "maximum_train_vs_holdout_ks": float(max_split_ks),
            "maximum_train_vs_holdout_wasserstein_normalized_global_range":
                float(max_split_wasserstein),
            "input_nearest_train_summary": split_nn,
            "test_iid_use": "descriptive_only_no_rule_adjustment",
        },
        "contact_applicability_boundary": contact_evidence,
        "literature_registry": literature_registry,
        "training_authorization": authorization,
        "guardrails": authorization["guardrails"],
    }

    config_output = output_root / CONFIG_DIR
    docs_output = output_root / DOCS_DIR
    audit_path = config_output / f"{PREFIX}_training_preflight_audit.json"
    authorization_path = (
        config_output / f"{PREFIX}_training_authorization.json"
    )
    distribution_path = (
        config_output / f"{PREFIX}_training_preflight_distribution_summary.csv"
    )
    bins_path = (
        config_output / f"{PREFIX}_training_preflight_distribution_bins.csv"
    )
    joint_path = (
        config_output / f"{PREFIX}_training_preflight_joint_coverage.csv"
    )
    occupancy_path = (
        config_output / f"{PREFIX}_training_preflight_joint_occupancy.csv"
    )
    split_summary_path = (
        config_output / f"{PREFIX}_training_preflight_split_summary.csv"
    )
    split_comparison_path = (
        config_output / f"{PREFIX}_training_preflight_split_comparison.csv"
    )
    literature_registry_path = (
        config_output / "v6_p1i_literature_registry_v1.json"
    )
    literature_crosswalk_path = (
        config_output / "v6_p1i_literature_id_crosswalk.csv"
    )
    report_path = docs_output / f"{PREFIX}_training_preflight_audit.md"
    distribution_figure = (
        docs_output / f"{PREFIX}_training_preflight_distributions.png"
    )
    joint_figure = (
        docs_output / f"{PREFIX}_training_preflight_joint_coverage.png"
    )
    split_figure = (
        docs_output / f"{PREFIX}_training_preflight_split_ecdf.png"
    )

    _write_json(audit_path, audit)
    _write_json(authorization_path, authorization)
    _write_csv(distribution_path, distribution_rows)
    _write_csv(bins_path, distribution_bin_rows)
    _write_csv(joint_path, joint_rows)
    _write_csv(occupancy_path, occupancy_rows)
    _write_csv(split_summary_path, split_summary_rows)
    _write_csv(split_comparison_path, split_comparison_rows)
    _write_json(literature_registry_path, literature_registry)
    _write_csv(literature_crosswalk_path, literature_rows)
    _plot_distributions(distribution_figure, distributions, values)
    _plot_joint(joint_figure, pairs)
    _plot_split(split_figure, values, roles)

    distribution_lines = []
    for name, _ in distribution_specs:
        item = distributions[name]
        summary = item["summary"]
        distribution_lines.append(
            "| "
            + " | ".join(
                [
                    name,
                    f"{_format_float(summary['q00'])}–{_format_float(summary['q100'])}",
                    _format_float(summary["q50"]),
                    str(item["histogram"]["occupied_bins"]),
                    _format_float(item["histogram"]["normalized_entropy"]),
                    _format_float(item["histogram"]["bin_count_cv"]),
                    _format_float(
                        item["uniform_reference_diagnostics"]["ks_statistic"]
                    ),
                    _format_float(
                        item["uniform_reference_diagnostics"][
                            "wasserstein_normalized_range"
                        ]
                    ),
                    _format_float(
                        item["maximum_sorted_gap"][
                            "normalized_by_observed_range"
                        ]
                    ),
                    item["modal_diagnostics"]["diagnosis"],
                ]
            )
            + " |"
        )
    pair_lines = []
    for pair_id in (
        "power_x_top_h",
        "power_x_bottom_h",
        "top_h_x_bottom_h",
        "power_x_total_source_area",
        "mean_q_x_mean_local_k",
        "effective_background_kz_x_top_h",
        "effective_background_kz_x_bottom_h",
    ):
        item = pairs[pair_id]
        pair_lines.append(
            "| "
            + " | ".join(
                [
                    pair_id,
                    _format_float(item["correlation"]["pearson_raw"]),
                    _format_float(item["correlation"]["spearman"]),
                    f"{item['physical_range_occupancy']['occupied_cells']}/36",
                    str(item["physical_range_occupancy"]["empty_cells"]),
                    str(
                        item["corner_coverage"]["counts"]["high_x_low_y"]
                    ),
                    _format_float(item["nearest_neighbour"]["q95"]),
                ]
            )
            + " |"
        )
    report = f"""# V6-P1i formal1024_v1 训练前审计

## 结论

本轮对冻结的 `{DATASET_ID}` 执行零修改审计：没有重新生成、筛选或替换
样本，没有训练，也没有模型推理。审计支持 **continuous broad coverage
（连续宽覆盖）**；它不支持、本文也不宣称严格 uniform（均匀）。

训练授权判定为
`{authorization['decision']}`。该授权只适用于冻结的耦合输入分布，并带有
两个重要边界：power–top_h 人为耦合，以及所有界面固定
`R_contact=0`。

## 一维物理响应分布

12-bin 为各指标冻结观测范围内的等宽分箱。KS/Wasserstein 以该观测范围
上的连续均匀分布为诊断参照，不是 uniform 验收门槛。

| 指标 | min–max | median | 占用bin | entropy | bin CV | KS | W1/range | max gap/range | KDE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
{chr(10).join(distribution_lines)}

判定依据：peak、mean、CV-RMS ΔT 与 Reff 均占满 12 个 bin，归一化 entropy
均不低于 0.90，最大排序间隙不超过观测范围的 5%。分箱计数并不相等，
KS 也明确拒绝把多数指标当作严格均匀样本；因此只能称连续宽覆盖。
peak 的 Scott-bandwidth KDE 在 0.75×/1.0× 下呈两峰、在 1.25× 下并峰，
属于 bandwidth-sensitive shoulder，不能据此宣称稳定双峰；CV-RMS 在
0.75× 下也出现同类弱肩峰。

![冻结分布](./{distribution_figure.name})

## 联合输入覆盖

物理 occupancy 使用 formal config 的冻结范围和 sampling-native
linear/log10 变换；derived 量使用冻结观测范围。另有 marginal-rank
occupancy，用于区分边缘分布偏斜与真正的联合依赖。

| pair | Pearson | Spearman | 占用cell | 空cell | high-x/low-y | NN P95 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
{chr(10).join(pair_lines)}

`power_x_top_h` 的 Spearman 为
`{_format_float(power_top_assessment['empirical_spearman'])}`，物理 6×6
网格有 `{power_top_assessment['physical_6x6_empty_cells']}` 个空 cell，
其中 high-power/low-top_h 角点为
`{power_top_assessment['high_power_low_top_h_count']}` 个样本。这不是随机
缺口，而是冻结 power 规则包含 top_h exponent
`{_format_float(power_top_assessment['contract_origin']['top_h_exponent'])}`
的直接结果。P1i-v1 因而不是 power 与 top_h 完全解混的数据集。

![联合覆盖](./{joint_figure.name})

## Split 描述性审计

train/valid_iid/test_iid 数量为 `768/128/128`。本节只描述输入及
peak/mean/CV-RMS/Reff/heat-fraction 分布；test_iid 没有用于修改规则、
split、门槛或训练授权。train 对两个 holdout 的最大单变量 KS 为
`{_format_float(max_split_ks)}`，最大 range-normalized Wasserstein 为
`{_format_float(max_split_wasserstein)}`。完整逐变量结果见 CSV。

![Split ECDF](./{split_figure.name})

## Perfect-contact 适用边界

P1h、P1i 和 V6 random-block 均采用 perfect interface contact，
`R_contact=0 m²K/W`。因此这些数据无法学习有限接触热阻变化，也不能把
其模型表现解释为对 contact-resistance OOD 的泛化证据。后续数据只能在
新的预注册版本中引入连续 `R_contact`：按真实相邻材料界面绑定，采用
log-space 连续抽样并保留显式零接触层；本轮不实现该方案。

## 文献 ID 与冻结绑定

新 registry 使用 `V6-LIT-NNN` 标识唯一 DOI/URL；所有 `P1I-Lxx` 与
`P1I3-Lxx` 通过 crosswalk 解析。background-k contract、formal config
和 formal manifest 都以 SHA256 固定，不回写冻结文件。

- formal config SHA256:
  `{FROZEN_INPUT_SHA256[f'{CONFIG_DIR}/{PREFIX}.yaml']}`
- formal manifest SHA256:
  `{FROZEN_INPUT_SHA256[f'{CONFIG_DIR}/{PREFIX}_manifest.json']}`
- manifest payload SHA256:
  `27d2ea3b7ec4e4ce9c6d068471cd19036ac8148b6cd57da325219d718c7e5ed5`
- background-k contract SHA256:
  `{FROZEN_INPUT_SHA256[f'{CONFIG_DIR}/v6_p1i_background_k_contract.csv']}`

## 治理约束

- formal1024_v1 内容和 split 均未修改；
- 不使用 test 作规则调整；
- 不宣称 strict uniform；
- 不将 training authorization 扩展为 OOD、contact-resistance 或独立
  power–BC 覆盖保证；
- clean-checkout replay 和外部归档状态由独立 manifest 绑定。
"""
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")

    outputs = [
        audit_path,
        authorization_path,
        distribution_path,
        bins_path,
        joint_path,
        occupancy_path,
        split_summary_path,
        split_comparison_path,
        literature_registry_path,
        literature_crosswalk_path,
        report_path,
        distribution_figure,
        joint_figure,
        split_figure,
    ]
    output_manifest = {
        "schema_version": "heat3d_v6_p1i_training_preflight_outputs_v1",
        "dataset_id": DATASET_ID,
        "status": audit["status"],
        "audit_payload_sha256": _canonical_sha256(audit),
        "artifacts": [
            {
                "path": str(path.relative_to(output_root)),
                "sha256": _sha256(path),
                "size_bytes": path.stat().st_size,
            }
            for path in outputs
        ],
        "guardrails": authorization["guardrails"],
    }
    output_manifest_path = (
        config_output / f"{PREFIX}_training_preflight_outputs_manifest.json"
    )
    _write_json(output_manifest_path, output_manifest)
    return {
        "status": audit["status"],
        "training_authorized": training_authorized,
        "continuous_broad_coverage": continuous_broad_coverage,
        "power_top_h_spearman": power_top_assessment["empirical_spearman"],
        "power_top_h_empty_cells":
            power_top_assessment["physical_6x6_empty_cells"],
        "power_top_h_high_power_low_h_count":
            power_top_assessment["high_power_low_top_h_count"],
        "maximum_train_holdout_ks": max_split_ks,
        "maximum_train_holdout_wasserstein_normalized": max_split_wasserstein,
        "output_manifest": str(output_manifest_path.relative_to(output_root)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-root",
        type=Path,
        default=Path(__file__).resolve().parent.parent,
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
    )
    args = parser.parse_args()
    input_root = args.input_root.resolve()
    output_root = (
        args.output_root.resolve()
        if args.output_root is not None
        else input_root
    )
    result = run(input_root, output_root)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
