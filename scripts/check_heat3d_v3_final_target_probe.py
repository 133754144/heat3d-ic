#!/usr/bin/env python3
"""Check generated Heat3D v3 final-target probe v0 samples.

The checker is read-only over an existing generated subset. It validates file
presence, array shapes, finite values, required metadata tags, solver residuals,
duplicate q/k/T hashes, and paired-scene consistency when multiple resolutions
for the same probe are present.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rigno.heat3d_v1_label_diagnostics import diagnose_sample  # noqa: E402
from rigno.heat3d_v1_reference_solver_v2 import BOTTOM_TOL_K, RESIDUAL_TOL  # noqa: E402


DEFAULT_MANIFEST = REPO_ROOT / "configs" / "heat3d_v3_final_target_probe_manifest_v0.json"
DEFAULT_SUBSET = (
    REPO_ROOT
    / "data"
    / "heat3d-thermal-simulation"
    / "subsets"
    / "v3_final_target_probe_v0"
)
REQUIRED_FILES = (
    "coords.npy",
    "layer_id.npy",
    "region_id.npy",
    "material_id.npy",
    "k_field.npy",
    "q_field.npy",
    "temperature.npy",
    "sample_meta.json",
    "metadata.json",
    "label_meta.json",
)
FINITE_ARRAYS = (
    "coords.npy",
    "k_field.npy",
    "q_field.npy",
    "temperature.npy",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--subset", type=Path, default=DEFAULT_SUBSET)
    parser.add_argument("--expected-count", type=int, default=10)
    parser.add_argument("--output-json", type=Path, default=None)
    return parser.parse_args()


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"expected JSON object: {path}")
    return data


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _sample_dirs(subset: Path) -> list[Path]:
    root = subset / "samples" if (subset / "samples").is_dir() else subset
    if root.is_dir() and (root / "sample_meta.json").is_file():
        return [root]
    if not root.is_dir():
        return []
    return sorted(child for child in root.iterdir() if child.is_dir() and (child / "sample_meta.json").is_file())


def _array_hash(array: np.ndarray) -> str:
    digest = hashlib.sha256()
    digest.update(str(array.shape).encode("utf-8"))
    digest.update(str(array.dtype).encode("utf-8"))
    digest.update(np.ascontiguousarray(array).view(np.uint8))
    return digest.hexdigest()


def _load_array(sample_dir: Path, name: str, errors: list[str]) -> np.ndarray | None:
    path = sample_dir / name
    if not path.is_file():
        errors.append(f"{sample_dir.name} missing {name}")
        return None
    try:
        return np.load(path)
    except Exception as exc:
        errors.append(f"{sample_dir.name} failed to load {name}: {exc}")
        return None


def _manifest_probe_map(manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    probes = manifest.get("probes", [])
    if not isinstance(probes, list):
        raise ValueError("manifest.probes must be a list")
    return {str(item["probe_id"]): item for item in probes if isinstance(item, dict)}


def _expected_shape(manifest: dict[str, Any], resolution: int) -> tuple[int, int, int] | None:
    entry = manifest.get("resolutions", {}).get(str(resolution))
    if not isinstance(entry, dict):
        return None
    return tuple(int(value) for value in entry.get("grid_shape", []))


def _check_metadata_tags(
    sample_name: str,
    metadata: dict[str, Any],
    sample_meta: dict[str, Any],
    manifest: dict[str, Any],
    probe_map: dict[str, dict[str, Any]],
    errors: list[str],
) -> None:
    for field in manifest.get("required_metadata_fields", []):
        if field not in metadata:
            errors.append(f"{sample_name} metadata.json missing required field {field}")
        if field not in sample_meta:
            errors.append(f"{sample_name} sample_meta.json missing required field {field}")

    probe_id = str(metadata.get("probe_id"))
    if probe_id not in probe_map:
        errors.append(f"{sample_name} probe_id {probe_id!r} not found in manifest")
        return

    probe = probe_map[probe_id]
    for key in (
        "probe_family",
        "intended_stressor",
        "semantic_scene_seed",
        "k_mode",
        "k_region_mode",
        "source_category",
        "q_power_range",
        "bc_category",
    ):
        if metadata.get(key) != probe.get(key):
            errors.append(f"{sample_name} metadata {key}={metadata.get(key)!r} differs from manifest {probe.get(key)!r}")
        if sample_meta.get(key) != probe.get(key):
            errors.append(f"{sample_name} sample_meta {key}={sample_meta.get(key)!r} differs from manifest {probe.get(key)!r}")

    if metadata.get("label_status") != "physics_label_generated":
        errors.append(f"{sample_name} label_status is not physics_label_generated")


def _check_shapes(
    sample_name: str,
    arrays: dict[str, np.ndarray],
    resolution: int,
    k_mode: str,
    manifest: dict[str, Any],
    errors: list[str],
) -> dict[str, Any]:
    shapes = {name: list(array.shape) for name, array in arrays.items()}
    coords = arrays["coords.npy"]
    k_field = arrays["k_field.npy"]
    q_field = arrays["q_field.npy"]
    temperature = arrays["temperature.npy"]
    if coords.shape != (resolution, 3):
        errors.append(f"{sample_name} coords.npy expected {(resolution, 3)}, found {coords.shape}")
    expected_grid_shape = _expected_shape(manifest, resolution)
    if expected_grid_shape is None:
        errors.append(f"{sample_name} resolution {resolution} missing from manifest")
    elif int(np.prod(expected_grid_shape)) != resolution:
        errors.append(f"{sample_name} manifest grid_shape {expected_grid_shape} does not match resolution {resolution}")
    expected_k_width = 3 if k_mode == "diag3" else 1
    if k_field.shape != (resolution, expected_k_width):
        errors.append(f"{sample_name} k_field.npy expected {(resolution, expected_k_width)}, found {k_field.shape}")
    if q_field.shape != (resolution, 1):
        errors.append(f"{sample_name} q_field.npy expected {(resolution, 1)}, found {q_field.shape}")
    if temperature.shape != (resolution, 1):
        errors.append(f"{sample_name} temperature.npy expected {(resolution, 1)}, found {temperature.shape}")
    for name, array in arrays.items():
        if not np.all(np.isfinite(array)):
            errors.append(f"{sample_name} {name} contains non-finite values")
    return shapes


def _basic_stats(values: np.ndarray) -> dict[str, float]:
    flat = np.asarray(values, dtype=np.float64).reshape(-1)
    return {
        "min": float(np.min(flat)),
        "max": float(np.max(flat)),
        "mean": float(np.mean(flat)),
        "std": float(np.std(flat)),
    }


def _q_bbox(coords: np.ndarray, q_field: np.ndarray) -> dict[str, Any] | None:
    mask = q_field[:, 0] > 0.0
    if not np.any(mask):
        return None
    selected = coords[mask]
    return {
        "min_xyz_m": [float(value) for value in np.min(selected, axis=0)],
        "max_xyz_m": [float(value) for value in np.max(selected, axis=0)],
        "center_xyz_m": [float(value) for value in np.mean(selected, axis=0)],
        "point_count": int(np.sum(mask)),
    }


def _k_cluster_stats(k_field: np.ndarray) -> dict[str, Any]:
    rounded = np.round(np.asarray(k_field, dtype=np.float64), decimals=8)
    unique_rows, counts = np.unique(rounded, axis=0, return_counts=True)
    order = np.argsort(-counts)
    top_clusters = []
    for idx in order[:8]:
        value = unique_rows[int(idx)]
        top_clusters.append({
            "value": [float(item) for item in np.ravel(value)],
            "count": int(counts[int(idx)]),
            "fraction": float(counts[int(idx)] / k_field.shape[0]),
        })
    return {
        "k_unique_count": int(unique_rows.shape[0]),
        "k_cluster_count": int(unique_rows.shape[0]),
        "k_top_clusters": top_clusters,
    }


def _p09_anisotropy_stats(k_field: np.ndarray) -> dict[str, Any] | None:
    if k_field.ndim != 2 or k_field.shape[1] != 3:
        return None
    min_channel = np.min(k_field, axis=1)
    max_channel = np.max(k_field, axis=1)
    ratio = max_channel / np.maximum(min_channel, 1.0e-30)
    return {
        "kx": _basic_stats(k_field[:, 0]),
        "ky": _basic_stats(k_field[:, 1]),
        "kz": _basic_stats(k_field[:, 2]),
        "anisotropy_ratio": _basic_stats(ratio),
        "anisotropy_ratio_max": float(np.max(ratio)),
        "anisotropy_ratio_mean": float(np.mean(ratio)),
    }


def _p10_gap_confirmation(metadata: dict[str, Any], sample_meta: dict[str, Any]) -> dict[str, Any] | None:
    if metadata.get("probe_id") != "P10":
        return None
    boundary_types = sample_meta.get("boundary_types", {})
    boundary_params = sample_meta.get("boundary_params", {})
    note = str(metadata.get("generator_capability_notes", ""))
    return {
        "localized_top_contact_supported": False,
        "side_asymmetry_supported": False,
        "v1_boundary_scope_confirmed": boundary_types == {
            "top": "Robin",
            "bottom": "Dirichlet",
            "sides": "adiabatic",
        },
        "uses_global_top_robin_only": "top" in boundary_params and "h_W_m2K" in boundary_params.get("top", {}),
        "gap_note_present": "gap" in note.lower() or "not fabricated" in note.lower() or "not represented" in note.lower(),
        "note": note,
    }


def _paired_scene_checks(rows: list[dict[str, Any]], errors: list[str]) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row["probe_id"])].append(row)

    checked_pairs = 0
    deferred_4096 = 0
    for probe_id, items in groups.items():
        by_resolution = {int(item["resolution"]): item for item in items}
        if 1024 in by_resolution and 4096 in by_resolution:
            checked_pairs += 1
            ref = by_resolution[1024]
            other = by_resolution[4096]
            for key in (
                "probe_family",
                "intended_stressor",
                "semantic_scene_seed",
                "k_mode",
                "k_region_mode",
                "source_category",
                "q_power_range",
                "bc_category",
            ):
                if ref.get(key) != other.get(key):
                    errors.append(f"{probe_id} paired 1024/4096 mismatch for {key}: {ref.get(key)!r} vs {other.get(key)!r}")
        elif 1024 in by_resolution and 4096 not in by_resolution:
            deferred_4096 += 1
    return {
        "checked_pairs": checked_pairs,
        "deferred_4096_pairs": deferred_4096,
    }


def main() -> int:
    args = parse_args()
    manifest = _read_json(args.manifest)
    probe_map = _manifest_probe_map(manifest)
    sample_dirs = _sample_dirs(args.subset)
    errors: list[str] = []
    warnings: list[str] = []
    rows: list[dict[str, Any]] = []
    hashes = {"q": Counter(), "k": Counter(), "temperature": Counter()}

    if len(sample_dirs) != args.expected_count:
        errors.append(f"expected {args.expected_count} samples, found {len(sample_dirs)}")

    for sample_dir in sample_dirs:
        sample_errors: list[str] = []
        missing = [name for name in REQUIRED_FILES if not (sample_dir / name).is_file()]
        if missing:
            sample_errors.append(f"{sample_dir.name} missing required files: {missing}")
            errors.extend(sample_errors)
            continue

        sample_meta = _read_json(sample_dir / "sample_meta.json")
        metadata = _read_json(sample_dir / "metadata.json")
        label_meta = _read_json(sample_dir / "label_meta.json")
        _check_metadata_tags(sample_dir.name, metadata, sample_meta, manifest, probe_map, sample_errors)
        resolution = int(metadata.get("resolution", 0))
        k_mode = str(metadata.get("k_mode", ""))
        arrays = {name: _load_array(sample_dir, name, sample_errors) for name in FINITE_ARRAYS}
        if any(array is None for array in arrays.values()):
            errors.extend(sample_errors)
            continue
        arrays_loaded = {name: array for name, array in arrays.items() if array is not None}
        shapes = _check_shapes(sample_dir.name, arrays_loaded, resolution, k_mode, manifest, sample_errors)

        if label_meta.get("convergence_flag") is not True:
            sample_errors.append(f"{sample_dir.name} label_meta.convergence_flag is not true")
        residual = float(label_meta.get("residual_norm", float("inf")))
        bottom_error = float(label_meta.get("bottom_dirichlet_error", float("inf")))
        if residual > RESIDUAL_TOL:
            sample_errors.append(f"{sample_dir.name} residual_norm {residual:.6e} exceeds {RESIDUAL_TOL:.6e}")
        if bottom_error > BOTTOM_TOL_K:
            sample_errors.append(f"{sample_dir.name} bottom_dirichlet_error {bottom_error:.6e} exceeds {BOTTOM_TOL_K:.6e}")

        report = diagnose_sample(sample_dir)
        if report.get("overall_status") == "fail":
            sample_errors.append(f"{sample_dir.name} label diagnostics failed")
        elif report.get("overall_status") == "warning":
            warnings.append(f"{sample_dir.name} label diagnostics warning")

        q_hash = _array_hash(arrays_loaded["q_field.npy"])
        k_hash = _array_hash(arrays_loaded["k_field.npy"])
        t_hash = _array_hash(arrays_loaded["temperature.npy"])
        hashes["q"][q_hash] += 1
        hashes["k"][k_hash] += 1
        hashes["temperature"][t_hash] += 1
        q_field = arrays_loaded["q_field.npy"]
        k_field = arrays_loaded["k_field.npy"]
        temperature = arrays_loaded["temperature.npy"]
        q_nonzero_count = int(np.count_nonzero(q_field))
        q_nonzero_fraction = float(q_nonzero_count / max(q_field.shape[0], 1))
        k_cluster_stats = _k_cluster_stats(k_field)
        p09_stats = _p09_anisotropy_stats(k_field) if metadata.get("probe_id") == "P09" else None
        p10_gap = _p10_gap_confirmation(metadata, sample_meta)
        row = {
            "sample_id": sample_dir.name,
            "probe_id": metadata.get("probe_id"),
            "probe_family": metadata.get("probe_family"),
            "intended_stressor": metadata.get("intended_stressor"),
            "resolution": resolution,
            "semantic_scene_seed": metadata.get("semantic_scene_seed"),
            "k_mode": metadata.get("k_mode"),
            "k_region_mode": metadata.get("k_region_mode"),
            "source_category": metadata.get("source_category"),
            "q_power_range": metadata.get("q_power_range"),
            "bc_category": metadata.get("bc_category"),
            "label_status": metadata.get("label_status"),
            "shapes": shapes,
            "q_nonzero_count": q_nonzero_count,
            "q_nonzero_fraction": q_nonzero_fraction,
            "q_bbox": _q_bbox(arrays_loaded["coords.npy"], q_field),
            "k_unique_count": k_cluster_stats["k_unique_count"],
            "k_cluster_count": k_cluster_stats["k_cluster_count"],
            "k_top_clusters": k_cluster_stats["k_top_clusters"],
            "temperature_min_K": float(np.min(temperature)),
            "temperature_max_K": float(np.max(temperature)),
            "temperature_mean_K": float(np.mean(temperature)),
            "temperature_std_K": float(np.std(temperature)),
            "T_min": float(np.min(temperature)),
            "T_max": float(np.max(temperature)),
            "T_mean": float(np.mean(temperature)),
            "T_std": float(np.std(temperature)),
            "residual_norm": residual,
            "bottom_dirichlet_error": bottom_error,
            "label_diagnostics_status": report.get("overall_status"),
            "q_hash": q_hash,
            "k_hash": k_hash,
            "temperature_hash": t_hash,
        }
        if p09_stats is not None:
            row["p09_k_channel_stats"] = p09_stats
        if p10_gap is not None:
            row["p10_gap_confirmation"] = p10_gap
            if not p10_gap["v1_boundary_scope_confirmed"] or not p10_gap["gap_note_present"]:
                sample_errors.append(f"{sample_dir.name} P10 gap confirmation is incomplete")
        rows.append(row)
        errors.extend(sample_errors)

    for kind, counter in hashes.items():
        duplicates = [digest for digest, count in counter.items() if count > 1]
        if duplicates:
            errors.append(f"duplicate {kind} hashes detected: {len(duplicates)} duplicate hash values")

    paired = _paired_scene_checks(rows, errors)
    counts = {
        "probe_family": Counter(str(row.get("probe_family")) for row in rows),
        "k_mode": Counter(str(row.get("k_mode")) for row in rows),
        "k_region_mode": Counter(str(row.get("k_region_mode")) for row in rows),
        "source_category": Counter(str(row.get("source_category")) for row in rows),
        "q_power_range": Counter(str(row.get("q_power_range")) for row in rows),
        "bc_category": Counter(str(row.get("bc_category")) for row in rows),
        "label_status": Counter(str(row.get("label_status")) for row in rows),
    }
    payload = {
        "subset": str(args.subset),
        "manifest": str(args.manifest),
        "sample_count": len(rows),
        "expected_count": args.expected_count,
        "status": "pass" if not errors else "fail",
        "error_count": len(errors),
        "warning_count": len(warnings),
        "errors": errors,
        "warnings": warnings,
        "counts": {key: dict(value) for key, value in counts.items()},
        "paired_scene_checks": paired,
        "max_residual_norm": max((row["residual_norm"] for row in rows), default=None),
        "max_bottom_dirichlet_error": max((row["bottom_dirichlet_error"] for row in rows), default=None),
        "rows": rows,
    }
    if args.output_json is not None:
        _write_json(args.output_json, payload)

    print("Heat3D v3 final-target probe checker")
    print(f"subset: {args.subset}")
    print(f"manifest: {args.manifest}")
    print(f"sample_count: {len(rows)} expected={args.expected_count}")
    print(f"status: {payload['status']}")
    print(f"max_residual_norm: {payload['max_residual_norm']}")
    print(f"max_bottom_dirichlet_error: {payload['max_bottom_dirichlet_error']}")
    print(f"paired_scene_checks: {paired}")
    for key, value in payload["counts"].items():
        print(f"{key}_counts: {value}")
    print("per_sample_stats:")
    for row in rows:
        print(
            "- "
            f"{row['sample_id']} probe={row['probe_id']} "
            f"q_nonzero_fraction={row['q_nonzero_fraction']:.6f} "
            f"k_unique={row['k_unique_count']} "
            f"T_min={row['T_min']:.6f} T_max={row['T_max']:.6f} "
            f"T_mean={row['T_mean']:.6f} T_std={row['T_std']:.6f} "
            f"bc={row['bc_category']} label={row['label_status']}"
        )
        if row.get("p09_k_channel_stats"):
            stats = row["p09_k_channel_stats"]
            print(
                "  P09 "
                f"kx_mean={stats['kx']['mean']:.6f} ky_mean={stats['ky']['mean']:.6f} "
                f"kz_mean={stats['kz']['mean']:.6f} "
                f"anisotropy_ratio_max={stats['anisotropy_ratio_max']:.6f}"
            )
        if row.get("p10_gap_confirmation"):
            gap = row["p10_gap_confirmation"]
            print(
                "  P10 "
                f"localized_top_contact_supported={gap['localized_top_contact_supported']} "
                f"side_asymmetry_supported={gap['side_asymmetry_supported']} "
                f"v1_boundary_scope_confirmed={gap['v1_boundary_scope_confirmed']}"
            )
    if warnings:
        print(f"warnings: {warnings}")
    if errors:
        print("errors:")
        for error in errors:
            print(f"- {error}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
