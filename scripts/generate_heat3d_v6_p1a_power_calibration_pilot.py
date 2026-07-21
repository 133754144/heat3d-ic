#!/usr/bin/env python3
"""Generate the fixed 16-sample V6-P1a layered power-calibration pilot.

This is a dataset generator, not a training or inference entry point.  It uses
only the powers frozen in the P1a case registry, never filters or resamples by
temperature, and refuses to overwrite an existing dataset directory.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np
import yaml
from scipy.interpolate import RegularGridInterpolator
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import factorized


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from audit_heat3d_v5_physics_distribution import (  # noqa: E402
    _boundary_fluxes,
    _grid_contract,
)


SCHEMA = "heat3d_v6_p1a_power_calibration_v1"
DEFAULT_CASES = REPO_ROOT / "configs/heat3d_v6/v6_p1a_power_calibration_cases.yaml"
DEFAULT_DATASET = REPO_ROOT / "data/heat3d_v6_p1a_power_calibration16_v0"
DEFAULT_AUDIT = REPO_ROOT / "configs/heat3d_v6/v6_p1a_power_calibration_audit.json"
DEFAULT_MANIFEST = REPO_ROOT / "configs/heat3d_v6/v6_p1a_power_calibration_manifest.json"
DEFAULT_SAMPLES_CSV = REPO_ROOT / "configs/heat3d_v6/v6_p1a_power_calibration_samples.csv"
DEFAULT_SOURCES_CSV = REPO_ROOT / "configs/heat3d_v6/v6_p1a_power_calibration_sources.csv"
SAMPLE_FILES = (
    "coords.npy",
    "temperature.npy",
    "deltaT.npy",
    "k_field.npy",
    "q_field.npy",
    "layer_id.npy",
    "bc_features.npy",
    "sample_meta.json",
)


class GenerationError(RuntimeError):
    pass


def _json_dump(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _array_sha256(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(contiguous.dtype).encode("utf-8"))
    digest.update(str(tuple(contiguous.shape)).encode("utf-8"))
    digest.update(contiguous.tobytes())
    return digest.hexdigest()


def _load_cases(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise GenerationError("case registry must be a YAML object")
    if payload.get("schema_version") != "heat3d_v6_p1a_power_cases_v1":
        raise GenerationError("unexpected case-registry schema")
    cases = payload.get("cases")
    if not isinstance(cases, list) or len(cases) != 16:
        raise GenerationError("P1a requires exactly 16 frozen cases")
    if len({str(case["id"]) for case in cases}) != 16:
        raise GenerationError("case IDs must be unique")
    return payload


def _build_z_axis(layers: Sequence[Mapping[str, Any]]) -> tuple[np.ndarray, np.ndarray, list[float]]:
    z_parts: list[np.ndarray] = []
    boundaries = [0.0]
    cursor = 0.0
    for index, layer in enumerate(layers):
        thickness = float(layer["thickness_m"])
        intervals = int(layer["z_intervals"])
        if thickness <= 0.0 or intervals < 4:
            raise GenerationError("each layer needs positive thickness and at least four intervals")
        part = np.linspace(cursor, cursor + thickness, intervals + 1)
        z_parts.append(part if index == 0 else part[1:])
        cursor += thickness
        boundaries.append(cursor)
    z = np.concatenate(z_parts)
    boundary_array = np.asarray(boundaries, dtype=np.float64)
    layer_ids = np.searchsorted(boundary_array[1:], z, side="right")
    layer_ids = np.minimum(layer_ids, len(layers) - 1).astype(np.int32)
    return z, layer_ids, boundaries


def _build_mesh(physics: Mapping[str, Any]) -> dict[str, Any]:
    nx, ny, expected_nz = map(int, physics["solver_mesh_intervals_xyz"])
    lx, ly = map(float, physics["footprint_m"])
    x = np.linspace(0.0, lx, nx + 1)
    y = np.linspace(0.0, ly, ny + 1)
    z, z_layer_ids, boundaries = _build_z_axis(physics["layers_bottom_to_top"])
    if z.size != expected_nz + 1:
        raise GenerationError(f"z intervals mismatch: {z.size - 1} != {expected_nz}")
    mesh = np.meshgrid(x, y, z, indexing="ij")
    coords = np.column_stack([part.reshape(-1) for part in mesh])
    layer_ids = np.broadcast_to(z_layer_ids, (x.size, y.size, z.size)).reshape(-1)
    layer_k = np.asarray(
        [float(layer["k_W_mK"]) for layer in physics["layers_bottom_to_top"]],
        dtype=np.float64,
    )
    k_scalar = layer_k[layer_ids]
    k_diag = np.repeat(k_scalar[:, None], 3, axis=1)
    info = _grid_contract(coords)
    return {
        "x": x,
        "y": y,
        "z": z,
        "coords": coords,
        "layer_ids": layer_ids,
        "k_diag": k_diag,
        "info": info,
        "boundaries": boundaries,
        "layer_index": {
            str(layer["id"]): index
            for index, layer in enumerate(physics["layers_bottom_to_top"])
        },
    }


class LayeredFvmSolver:
    def __init__(self, mesh: Mapping[str, Any], physics: Mapping[str, Any]) -> None:
        self.mesh = mesh
        self.physics = physics
        self.top_h = float(physics["boundary_conditions"]["top"]["h_W_m2K"])
        self.top_t = float(physics["boundary_conditions"]["top"]["T_inf_K"])
        self.bottom_t = float(physics["boundary_conditions"]["bottom"]["T_K"])
        self.matrix, self.base_rhs = self._assemble()
        self.solve_linear = factorized(csc_matrix(self.matrix))

    def _assemble(self) -> tuple[csc_matrix, np.ndarray]:
        info = self.mesh["info"]
        x, y, z = info["axes"]
        dx, dy, dz = info["widths"]
        grid = info["grid"]
        k_diag = self.mesh["k_diag"]
        node_count = self.mesh["coords"].shape[0]
        rows: list[int] = []
        cols: list[int] = []
        data: list[float] = []
        rhs = np.zeros(node_count, dtype=np.float64)

        def add(i: int, j: int, value: float) -> None:
            rows.append(i)
            cols.append(j)
            data.append(float(value))

        for ix in range(x.size):
            for iy in range(y.size):
                for iz in range(z.size):
                    i = int(grid[ix, iy, iz])
                    if iz == 0:
                        add(i, i, 1.0)
                        rhs[i] = self.bottom_t
                        continue
                    diagonal = 0.0
                    if iz == z.size - 1:
                        robin = self.top_h * float(dx[ix] * dy[iy])
                        diagonal += robin
                        rhs[i] += robin * self.top_t
                    neighbors = (
                        (ix - 1, iy, iz, 0, float(dy[iy] * dz[iz]), float(x[ix] - x[ix - 1]) if ix else 0.0),
                        (ix + 1, iy, iz, 0, float(dy[iy] * dz[iz]), float(x[ix + 1] - x[ix]) if ix + 1 < x.size else 0.0),
                        (ix, iy - 1, iz, 1, float(dx[ix] * dz[iz]), float(y[iy] - y[iy - 1]) if iy else 0.0),
                        (ix, iy + 1, iz, 1, float(dx[ix] * dz[iz]), float(y[iy + 1] - y[iy]) if iy + 1 < y.size else 0.0),
                        (ix, iy, iz - 1, 2, float(dx[ix] * dy[iy]), float(z[iz] - z[iz - 1])),
                        (ix, iy, iz + 1, 2, float(dx[ix] * dy[iy]), float(z[iz + 1] - z[iz]) if iz + 1 < z.size else 0.0),
                    )
                    for jx, jy, jz, axis, area, distance in neighbors:
                        if not (0 <= jx < x.size and 0 <= jy < y.size and 0 <= jz < z.size):
                            continue
                        j = int(grid[jx, jy, jz])
                        ki = float(k_diag[i, axis])
                        kj = float(k_diag[j, axis])
                        conductance = (2.0 * ki * kj / (ki + kj)) * area / distance
                        diagonal += conductance
                        add(i, j, -conductance)
                    add(i, i, diagonal)
        matrix = csc_matrix((data, (rows, cols)), shape=(node_count, node_count))
        matrix.sum_duplicates()
        return matrix, rhs

    def solve(self, q: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
        q = np.asarray(q, dtype=np.float64).reshape(-1)
        weights = np.asarray(self.mesh["info"]["weights"], dtype=np.float64)
        rhs = self.base_rhs + q * weights
        bottom = np.asarray(self.mesh["coords"][:, 2] == self.mesh["z"][0])
        rhs[bottom] = self.bottom_t
        temperature = np.asarray(self.solve_linear(rhs), dtype=np.float64)
        residual = float(
            np.linalg.norm(self.matrix.dot(temperature) - rhs)
            / max(np.linalg.norm(rhs), 1.0)
        )
        if not np.all(np.isfinite(temperature)) or not math.isfinite(residual):
            raise GenerationError("solver produced non-finite output")
        top_flux, bottom_flux = _boundary_fluxes(
            self.mesh["coords"],
            self.mesh["k_diag"],
            temperature,
            self.top_h,
            self.top_t,
            self.mesh["info"],
        )
        return temperature, {
            "linear_residual": residual,
            "top_heat_flux_W": top_flux,
            "bottom_heat_flux_W": bottom_flux,
        }


SOURCE_CENTERS = (
    (0.125, 0.20), (0.375, 0.20), (0.625, 0.20), (0.875, 0.20),
    (0.125, 0.50), (0.375, 0.50), (0.625, 0.50), (0.875, 0.50),
    (0.125, 0.80), (0.375, 0.80), (0.625, 0.80), (0.875, 0.80),
)


def _build_sources(
    case: Mapping[str, Any],
    physics: Mapping[str, Any],
    mesh: Mapping[str, Any],
) -> tuple[np.ndarray, list[dict[str, Any]], dict[str, float]]:
    coords = mesh["coords"]
    weights = np.asarray(mesh["info"]["weights"], dtype=np.float64)
    layer_ids = mesh["layer_ids"]
    lx, ly = map(float, physics["footprint_m"])
    patch_x, patch_y = map(float, physics["source_patch_size_m"])
    min_count = int(physics["source_min_control_volume_count"])
    layer_occurrence: Counter[str] = Counter()
    occupied = np.zeros(coords.shape[0], dtype=bool)
    q = np.zeros(coords.shape[0], dtype=np.float64)
    source_rows: list[dict[str, Any]] = []
    layer_powers: defaultdict[str, float] = defaultdict(float)
    boundaries = mesh["boundaries"]

    for source_index, source in enumerate(case["sources"]):
        layer_name = str(source["layer"])
        if layer_name not in {"active_lower", "active_upper"}:
            raise GenerationError(f"{case['id']}: source outside active layer")
        local_index = layer_occurrence[layer_name]
        layer_occurrence[layer_name] += 1
        if local_index >= len(SOURCE_CENTERS):
            raise GenerationError(f"{case['id']}: too many sources in {layer_name}")
        cx, cy = SOURCE_CENTERS[local_index]
        x0, x1 = cx * lx - patch_x / 2.0, cx * lx + patch_x / 2.0
        y0, y1 = cy * ly - patch_y / 2.0, cy * ly + patch_y / 2.0
        layer_index = int(mesh["layer_index"][layer_name])
        mask = (
            (layer_ids == layer_index)
            & (coords[:, 0] >= x0)
            & (coords[:, 0] <= x1)
            & (coords[:, 1] >= y0)
            & (coords[:, 1] <= y1)
        )
        if np.any(mask & occupied):
            raise GenerationError(f"{case['id']}: source masks overlap")
        occupied |= mask
        count = int(np.sum(mask))
        if count < min_count:
            raise GenerationError(
                f"{case['id']} source {source_index}: {count} control volumes < {min_count}"
            )
        volume = float(np.sum(weights[mask]))
        power = float(source["power_W"])
        q_density = power / volume
        q[mask] = q_density
        layer_powers[layer_name] += power
        source_rows.append(
            {
                "sample_id": str(case["id"]),
                "source_id": f"src_{source_index:02d}",
                "active_layer": layer_name,
                "active_layer_index": layer_index,
                "bbox_m": {"x": [x0, x1], "y": [y0, y1], "z": [boundaries[layer_index], boundaries[layer_index + 1]]},
                "source_volume_m3": volume,
                "source_power_W": power,
                "q_W_m3": q_density,
                "covered_control_volume_count": count,
                "power_provenance": str(case["power_basis"]),
                "literature_id": str(case["literature_id"]),
            }
        )
    package_power = float(sum(float(source["power_W"]) for source in case["sources"]))
    realized_power = float(np.dot(q, weights))
    if not math.isclose(package_power, realized_power, rel_tol=1.0e-12, abs_tol=1.0e-14):
        raise GenerationError(f"{case['id']}: source power integration mismatch")
    return q, source_rows, dict(sorted(layer_powers.items()))


def _point_seed(base_seed: int, sample_id: str) -> int:
    digest = hashlib.sha256(
        f"{base_seed}:{sample_id}:v6_p1a_irregular_points_v1".encode("utf-8")
    ).hexdigest()
    return int(digest[:16], 16) % (2**32)


def _sample_points_before_labels(
    *,
    base_seed: int,
    sample_id: str,
    physics: Mapping[str, Any],
    mesh: Mapping[str, Any],
    sources: Sequence[Mapping[str, Any]],
) -> tuple[np.ndarray, list[str], int]:
    seed = _point_seed(base_seed, sample_id)
    rng = np.random.default_rng(seed)
    lx, ly = map(float, physics["footprint_m"])
    z0, z1 = float(mesh["z"][0]), float(mesh["z"][-1])
    pieces: list[np.ndarray] = []
    strata: list[str] = []

    volume = rng.uniform([0.0, 0.0, z0], [lx, ly, z1], size=(512, 3))
    pieces.append(volume)
    strata.extend(["volume"] * 512)

    source_points = np.empty((256, 3), dtype=np.float64)
    for index in range(256):
        source = sources[index % len(sources)]
        bbox = source["bbox_m"]
        source_points[index] = rng.uniform(
            [bbox["x"][0], bbox["y"][0], bbox["z"][0]],
            [bbox["x"][1], bbox["y"][1], bbox["z"][1]],
        )
    pieces.append(source_points)
    strata.extend(["source"] * 256)

    internal_boundaries = np.asarray(mesh["boundaries"][1:-1], dtype=np.float64)
    interface = np.empty((128, 3), dtype=np.float64)
    interface[:, :2] = rng.uniform([0.0, 0.0], [lx, ly], size=(128, 2))
    interface[:, 2] = internal_boundaries[np.arange(128) % internal_boundaries.size]
    pieces.append(interface)
    strata.extend(["interface"] * 128)

    top = np.column_stack(
        [rng.uniform(0.0, lx, 64), rng.uniform(0.0, ly, 64), np.full(64, z1)]
    )
    bottom = np.column_stack(
        [rng.uniform(0.0, lx, 64), rng.uniform(0.0, ly, 64), np.full(64, z0)]
    )
    pieces.extend([top, bottom])
    strata.extend(["top"] * 64)
    strata.extend(["bottom"] * 64)
    points = np.vstack(pieces)
    if points.shape != (1024, 3) or len(np.unique(points, axis=0)) != 1024:
        raise GenerationError(f"{sample_id}: irregular points are not 1024 unique points")
    return points, strata, seed


def _values_at_points(
    points: np.ndarray,
    physics: Mapping[str, Any],
    mesh: Mapping[str, Any],
    source_rows: Sequence[Mapping[str, Any]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    boundaries = np.asarray(mesh["boundaries"], dtype=np.float64)
    layer_ids = np.searchsorted(boundaries[1:], points[:, 2], side="right")
    layer_ids = np.minimum(layer_ids, len(physics["layers_bottom_to_top"]) - 1).astype(np.int32)
    layer_k = np.asarray(
        [float(layer["k_W_mK"]) for layer in physics["layers_bottom_to_top"]],
        dtype=np.float64,
    )
    k = np.repeat(layer_k[layer_ids, None], 3, axis=1)
    q = np.zeros(points.shape[0], dtype=np.float64)
    for source in source_rows:
        bbox = source["bbox_m"]
        mask = (
            (layer_ids == int(source["active_layer_index"]))
            & (points[:, 0] >= bbox["x"][0]) & (points[:, 0] <= bbox["x"][1])
            & (points[:, 1] >= bbox["y"][0]) & (points[:, 1] <= bbox["y"][1])
            & (points[:, 2] >= bbox["z"][0]) & (points[:, 2] <= bbox["z"][1])
        )
        q[mask] += float(source["q_W_m3"])
    return layer_ids, k, q


def _bc_features(points: np.ndarray, physics: Mapping[str, Any], mesh: Mapping[str, Any]) -> np.ndarray:
    lx, ly = map(float, physics["footprint_m"])
    z0, z1 = float(mesh["z"][0]), float(mesh["z"][-1])
    atol = 1.0e-15
    top = np.isclose(points[:, 2], z1, atol=atol)
    bottom = np.isclose(points[:, 2], z0, atol=atol)
    side = (
        np.isclose(points[:, 0], 0.0, atol=atol)
        | np.isclose(points[:, 0], lx, atol=atol)
        | np.isclose(points[:, 1], 0.0, atol=atol)
        | np.isclose(points[:, 1], ly, atol=atol)
    )
    interior = ~(top | bottom | side)
    return np.column_stack([top, bottom, side, interior]).astype(np.float64)


def _sample_metrics(
    *,
    temperature: np.ndarray,
    q: np.ndarray,
    mesh: Mapping[str, Any],
    solver_audit: Mapping[str, float],
    bottom_t: float,
) -> dict[str, float | bool]:
    weights = np.asarray(mesh["info"]["weights"], dtype=np.float64)
    delta_t = temperature - bottom_t
    power = float(np.dot(q, weights))
    volume = float(np.sum(weights))
    top_flux = float(solver_audit["top_heat_flux_W"])
    bottom_flux = float(solver_audit["bottom_heat_flux_W"])
    peak = float(np.max(delta_t))
    return {
        "package_total_power_W": power,
        "peak_deltaT_K": peak,
        "mean_deltaT_K": float(np.dot(delta_t, weights) / volume),
        "Rth_peak_K_W": peak / power,
        "top_heat_fraction": top_flux / power,
        "bottom_heat_fraction": bottom_flux / power,
        "energy_balance_relative_error": (power - top_flux - bottom_flux) / power,
        "linear_residual": float(solver_audit["linear_residual"]),
        "in_30_80_K_window": bool(30.0 <= peak <= 80.0),
    }


def _write_sample(
    target: Path,
    arrays: Mapping[str, np.ndarray],
    meta: Mapping[str, Any],
) -> dict[str, str]:
    target.mkdir(parents=True)
    for name, array in arrays.items():
        np.save(target / name, np.asarray(array), allow_pickle=False)
    _json_dump(target / "sample_meta.json", meta)
    return {name: _sha256(target / name) for name in SAMPLE_FILES}


def _csv_write(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def generate(args: argparse.Namespace) -> dict[str, Any]:
    registry = _load_cases(args.cases)
    literature_path = REPO_ROOT / registry["literature_matrix"]["path"]
    if _sha256(literature_path) != registry["literature_matrix"]["sha256"]:
        raise GenerationError("literature matrix SHA256 mismatch")
    physics = registry["physics"]
    mesh = _build_mesh(physics)
    case_plan = [
        {
            "sample_id": str(case["id"]),
            "literature_id": str(case["literature_id"]),
            "source_count": len(case["sources"]),
            "active_layers": sorted({str(source["layer"]) for source in case["sources"]}),
            "package_total_power_W": sum(float(source["power_W"]) for source in case["sources"]),
        }
        for case in registry["cases"]
    ]
    if args.dry_run:
        return {
            "schema_version": SCHEMA,
            "mode": "dry_run",
            "dataset_id": registry["dataset_id"],
            "sample_count": len(case_plan),
            "case_plan": case_plan,
            "solver_calls": 0,
            "dataset_writes": 0,
            "training_runs": 0,
            "model_inference_runs": 0,
            "peak_deltaT_filtering": False,
        }
    if args.dataset.exists():
        raise GenerationError(f"refusing to overwrite existing dataset: {args.dataset}")

    solver = LayeredFvmSolver(mesh, physics)
    sample_rows: list[dict[str, Any]] = []
    source_csv_rows: list[dict[str, Any]] = []
    manifest_rows: list[dict[str, Any]] = []
    temp_parent = args.dataset.parent
    temp_parent.mkdir(parents=True, exist_ok=True)
    temp_root = Path(tempfile.mkdtemp(prefix=f".{args.dataset.name}.", dir=temp_parent))
    try:
        for case in registry["cases"]:
            sample_id = str(case["id"])
            q, source_rows, layer_powers = _build_sources(case, physics, mesh)
            points, point_strata, point_seed = _sample_points_before_labels(
                base_seed=int(registry["seed"]),
                sample_id=sample_id,
                physics=physics,
                mesh=mesh,
                sources=source_rows,
            )
            points_sha = _array_sha256(points)
            temperature, solver_audit = solver.solve(q)
            metrics = _sample_metrics(
                temperature=temperature,
                q=q,
                mesh=mesh,
                solver_audit=solver_audit,
                bottom_t=solver.bottom_t,
            )
            interpolator = RegularGridInterpolator(
                (mesh["x"], mesh["y"], mesh["z"]),
                temperature.reshape(mesh["info"]["shape"]),
                method="linear",
                bounds_error=True,
            )
            point_temperature = np.asarray(interpolator(points), dtype=np.float64)
            point_layers, point_k, point_q = _values_at_points(points, physics, mesh, source_rows)
            point_bc = _bc_features(points, physics, mesh)
            projection_peak_gap = float(
                metrics["peak_deltaT_K"] - np.max(point_temperature - solver.bottom_t)
            )
            arrays = {
                "coords.npy": points.astype(np.float64),
                "temperature.npy": point_temperature[:, None].astype(np.float64),
                "deltaT.npy": (point_temperature - solver.bottom_t)[:, None].astype(np.float64),
                "k_field.npy": point_k.astype(np.float64),
                "q_field.npy": point_q[:, None].astype(np.float64),
                "layer_id.npy": point_layers[:, None].astype(np.int32),
                "bc_features.npy": point_bc.astype(np.float64),
            }
            source_payload = []
            for source in source_rows:
                row = dict(source)
                row["active_layer_power_W"] = layer_powers[row["active_layer"]]
                row["package_total_power_W"] = metrics["package_total_power_W"]
                source_payload.append(row)
                source_csv_rows.append(
                    {
                        "sample_id": row["sample_id"],
                        "source_id": row["source_id"],
                        "literature_id": row["literature_id"],
                        "power_provenance": row["power_provenance"],
                        "active_layer": row["active_layer"],
                        "source_volume_m3": f"{row['source_volume_m3']:.17g}",
                        "source_power_W": f"{row['source_power_W']:.17g}",
                        "q_W_m3": f"{row['q_W_m3']:.17g}",
                        "covered_control_volume_count": row["covered_control_volume_count"],
                        "active_layer_power_W": f"{row['active_layer_power_W']:.17g}",
                        "package_total_power_W": f"{row['package_total_power_W']:.17g}",
                    }
                )
            meta = {
                "schema_version": SCHEMA,
                "sample_id": sample_id,
                "dataset_id": registry["dataset_id"],
                "literature_id": str(case["literature_id"]),
                "power_basis": str(case["power_basis"]),
                "power_was_Rth_inferred": False,
                "active_layers": sorted(layer_powers),
                "active_layer_power_W": layer_powers,
                "sources": source_payload,
                "boundary_conditions": physics["boundary_conditions"],
                "contact": physics["contact"],
                "solver_mesh": {
                    "type": "layer_aligned_node_control_volume",
                    "shape": list(mesh["info"]["shape"]),
                    "node_count": int(mesh["coords"].shape[0]),
                    "axis_sha256": {
                        "x": _array_sha256(mesh["x"]),
                        "y": _array_sha256(mesh["y"]),
                        "z": _array_sha256(mesh["z"]),
                    },
                    "minimum_source_control_volume_count": min(
                        int(source["covered_control_volume_count"]) for source in source_rows
                    ),
                },
                "operator_projection": {
                    "point_count": 1024,
                    "point_seed": point_seed,
                    "point_schema": "v6_p1a_irregular_points_v1",
                    "point_coordinates_sha256": points_sha,
                    "point_coordinates_frozen_before_temperature_solve": True,
                    "strata_counts": dict(sorted(Counter(point_strata).items())),
                    "interpolation_method": "scipy_regular_grid_linear",
                    "solver_peak_minus_projected_peak_K": projection_peak_gap,
                    "label_inputs_used_for_point_selection": [],
                },
                "metrics": metrics,
                "guardrails": {
                    "peak_deltaT_filtering": False,
                    "peak_deltaT_resampling": False,
                    "sample_replacement": False,
                    "training_runs": 0,
                    "model_inference_runs": 0,
                },
            }
            file_hashes = _write_sample(temp_root / sample_id, arrays, meta)
            manifest_rows.append(
                {
                    "sample_id": sample_id,
                    "literature_id": str(case["literature_id"]),
                    "sample_dir": sample_id,
                    "file_sha256": file_hashes,
                    "point_coordinates_sha256": points_sha,
                }
            )
            sample_rows.append(
                {
                    "sample_id": sample_id,
                    "literature_id": str(case["literature_id"]),
                    "power_basis": str(case["power_basis"]),
                    "source_count": len(source_rows),
                    "active_layer_count": len(layer_powers),
                    "active_layer_power_W": json.dumps(layer_powers, sort_keys=True),
                    "package_total_power_W": metrics["package_total_power_W"],
                    "peak_deltaT_K": metrics["peak_deltaT_K"],
                    "mean_deltaT_K": metrics["mean_deltaT_K"],
                    "Rth_peak_K_W": metrics["Rth_peak_K_W"],
                    "top_heat_fraction": metrics["top_heat_fraction"],
                    "bottom_heat_fraction": metrics["bottom_heat_fraction"],
                    "energy_balance_relative_error": metrics["energy_balance_relative_error"],
                    "linear_residual": metrics["linear_residual"],
                    "in_30_80_K_window": metrics["in_30_80_K_window"],
                    "solver_peak_minus_projected_peak_K": projection_peak_gap,
                }
            )
        manifest = {
            "schema_version": SCHEMA,
            "dataset_id": registry["dataset_id"],
            "case_registry": str(args.cases.relative_to(REPO_ROOT)),
            "case_registry_sha256": _sha256(args.cases),
            "literature_matrix_sha256": registry["literature_matrix"]["sha256"],
            "sample_count": len(manifest_rows),
            "samples": manifest_rows,
            "guardrails": {
                "peak_deltaT_filtering": False,
                "peak_deltaT_resampling": False,
                "training_runs": 0,
                "model_inference_runs": 0,
            },
        }
        _json_dump(temp_root / "manifest.json", manifest)
        temp_root.rename(args.dataset)
    except Exception:
        shutil.rmtree(temp_root, ignore_errors=True)
        raise

    _json_dump(args.manifest_json, manifest)
    numeric = lambda key: np.asarray([float(row[key]) for row in sample_rows], dtype=np.float64)
    hits = [row["sample_id"] for row in sample_rows if row["in_30_80_K_window"]]
    audit = {
        "schema_version": SCHEMA,
        "dataset_id": registry["dataset_id"],
        "dataset_path": str(args.dataset.relative_to(REPO_ROOT)),
        "dataset_manifest_sha256": _sha256(args.dataset / "manifest.json"),
        "tracked_manifest_sha256": _sha256(args.manifest_json),
        "sample_count": len(sample_rows),
        "source_count": len(source_csv_rows),
        "literature_case_counts": dict(sorted(Counter(row["literature_id"] for row in sample_rows).items())),
        "power_definition": {
            "component_or_source_power": "each source row source_power_W",
            "active_layer_power": "sum source_power_W within a named active layer",
            "package_total_power": "sum active-layer power across the sample",
            "Rth_power_inference_used": False,
        },
        "peak_deltaT_evaluation_window_K": [30.0, 80.0],
        "window_hit_sample_ids": hits,
        "window_hit_count": len(hits),
        "summary": {
            key: {
                "min": float(np.min(numeric(key))),
                "median": float(np.median(numeric(key))),
                "max": float(np.max(numeric(key))),
            }
            for key in (
                "package_total_power_W",
                "peak_deltaT_K",
                "mean_deltaT_K",
                "Rth_peak_K_W",
                "top_heat_fraction",
                "bottom_heat_fraction",
                "energy_balance_relative_error",
                "solver_peak_minus_projected_peak_K",
            )
        },
        "integrity": {
            "all_metrics_finite": bool(
                all(
                    np.all(np.isfinite(numeric(key)))
                    for key in (
                        "package_total_power_W",
                        "peak_deltaT_K",
                        "mean_deltaT_K",
                        "Rth_peak_K_W",
                        "top_heat_fraction",
                        "bottom_heat_fraction",
                        "energy_balance_relative_error",
                        "linear_residual",
                    )
                )
            ),
            "max_abs_energy_balance_relative_error": float(
                np.max(np.abs(numeric("energy_balance_relative_error")))
            ),
            "min_source_control_volume_count": min(
                int(row["covered_control_volume_count"]) for row in source_csv_rows
            ),
            "point_count_per_sample": 1024,
            "point_coordinates_frozen_before_labels": True,
        },
        "guardrails": {
            "generated_samples": 16,
            "expanded_samples": 0,
            "peak_deltaT_filtering": False,
            "peak_deltaT_resampling": False,
            "training_runs": 0,
            "model_inference_runs": 0,
        },
        "samples": sample_rows,
    }
    _json_dump(args.audit_json, audit)
    sample_fields = list(sample_rows[0])
    _csv_write(
        args.samples_csv,
        [
            {
                key: (
                    value
                    if isinstance(value, (str, int, bool))
                    else f"{float(value):.17g}"
                )
                for key, value in row.items()
            }
            for row in sample_rows
        ],
        sample_fields,
    )
    source_fields = list(source_csv_rows[0])
    _csv_write(args.sources_csv, source_csv_rows, source_fields)
    return audit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--audit-json", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--manifest-json", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--samples-csv", type=Path, default=DEFAULT_SAMPLES_CSV)
    parser.add_argument("--sources-csv", type=Path, default=DEFAULT_SOURCES_CSV)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    for name in ("cases", "dataset", "audit_json", "manifest_json", "samples_csv", "sources_csv"):
        value = getattr(args, name)
        if not value.is_absolute():
            setattr(args, name, (REPO_ROOT / value).resolve())
    payload = generate(args)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
