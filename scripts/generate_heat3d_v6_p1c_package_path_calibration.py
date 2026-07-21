#!/usr/bin/env python3
"""Generate the fixed V6-P1c 8-sample package-path calibration dataset.

P1b path A is read and replayed in memory only for paired diagnostics.  The
script writes exactly eight new B/C0 samples and never trains or evaluates a
learned model.  Power, material properties, and boundary conditions are fixed
before any temperature solve and are never adapted from the result.
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
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np
import yaml
from scipy.interpolate import RegularGridInterpolator
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import LinearOperator, cg

import generate_heat3d_v6_p1a_power_calibration_pilot as p1a
import generate_heat3d_v6_p1b_logic_package_power_calibration as p1b


REPO_ROOT = Path(__file__).resolve().parent.parent
SCHEMA = "heat3d_v6_p1c_package_path_calibration_v1"
DEFAULT_CASES = REPO_ROOT / "configs/heat3d_v6/v6_p1c_package_path_calibration_cases.yaml"
DEFAULT_DATASET = REPO_ROOT / "data/heat3d_v6_p1c_package_path_calibration8_v0"
DEFAULT_AUDIT = REPO_ROOT / "configs/heat3d_v6/v6_p1c_package_path_calibration_audit.json"
DEFAULT_MANIFEST = REPO_ROOT / "configs/heat3d_v6/v6_p1c_package_path_calibration_manifest.json"
DEFAULT_SAMPLES = REPO_ROOT / "configs/heat3d_v6/v6_p1c_package_path_calibration_samples.csv"
DEFAULT_SOURCES = REPO_ROOT / "configs/heat3d_v6/v6_p1c_package_path_calibration_sources.csv"
DEFAULT_LAYERS = REPO_ROOT / "configs/heat3d_v6/v6_p1c_package_path_layer_drops.csv"
DEFAULT_INTERFACES = REPO_ROOT / "configs/heat3d_v6/v6_p1c_package_path_interface_drops.csv"
DEFAULT_PAIRED = REPO_ROOT / "configs/heat3d_v6/v6_p1c_package_path_paired_comparison.csv"


class GenerationError(RuntimeError):
    pass


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_dump(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _csv_write(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise GenerationError(f"refusing to write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _repo_path(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _load_registry(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema_version") != "heat3d_v6_p1c_package_path_cases_v1":
        raise GenerationError("unexpected P1c registry schema")
    if payload.get("sample_count") != 8 or len(payload.get("cases", [])) != 8:
        raise GenerationError("P1c requires exactly eight new cases")
    if len({str(case["id"]) for case in payload["cases"]}) != 8:
        raise GenerationError("P1c sample IDs must be unique")
    return payload


def _path_physics(registry: Mapping[str, Any], path_id: str) -> dict[str, Any]:
    common = registry["common_physics"]
    path = registry["paths"][path_id]
    layers = [dict(layer) for layer in path["prepended_layers_bottom_to_top"]]
    layers.extend(dict(layer) for layer in registry["base_stack_layers_bottom_to_top"])
    return {
        "footprint_m": list(common["footprint_m"]),
        "layers_bottom_to_top": layers,
        "solver_mesh_intervals_xyz": list(path["solver_mesh_intervals_xyz"]),
        "source_min_control_volume_count": int(common["source_min_control_volume_count"]),
        "source_min_in_plane_intervals": int(common["source_min_in_plane_intervals"]),
        "boundary_conditions": {
            "top": dict(common["top"]),
            "bottom": dict(path["bottom"]),
            "sides": dict(common["sides"]),
        },
        "contact": dict(common["contact"]),
        "operator_projection": dict(common["operator_projection"]),
    }


def _build_mesh(physics: Mapping[str, Any]) -> dict[str, Any]:
    mesh = p1a._build_mesh({
        **physics,
        "layers_bottom_to_top": [
            {**layer, "k_W_mK": float(layer.get("k_W_mK", layer.get("k_xyz_W_mK", [0, 0, 0])[2]))}
            for layer in physics["layers_bottom_to_top"]
        ],
    })
    conductivity = []
    for layer in physics["layers_bottom_to_top"]:
        if "k_xyz_W_mK" in layer:
            conductivity.append(list(map(float, layer["k_xyz_W_mK"])))
        else:
            value = float(layer["k_W_mK"])
            conductivity.append([value, value, value])
    mesh["k_diag"] = np.asarray(conductivity, dtype=np.float64)[mesh["layer_ids"]]
    return mesh


class PackagePathSolver:
    def __init__(self, mesh: Mapping[str, Any], physics: Mapping[str, Any]) -> None:
        self.mesh = mesh
        self.physics = physics
        self.top_h = float(physics["boundary_conditions"]["top"]["h_W_m2K"])
        self.top_t = float(physics["boundary_conditions"]["top"]["T_inf_K"])
        self.bottom = physics["boundary_conditions"]["bottom"]
        self.bottom_type = str(self.bottom["type"])
        self.bottom_t = float(self.bottom.get("T_K", self.top_t))
        self.matrix, self.base_rhs = self._assemble()
        diagonal = np.asarray(self.matrix.diagonal(), dtype=np.float64)
        if np.any(diagonal <= 0.0):
            raise GenerationError("package-path matrix has non-positive diagonal")
        self.preconditioner = LinearOperator(
            self.matrix.shape,
            matvec=lambda value: np.asarray(value, dtype=np.float64) / diagonal,
            dtype=np.float64,
        )

    def _assemble(self) -> tuple[csc_matrix, np.ndarray]:
        info = self.mesh["info"]
        x, y, z = info["axes"]
        dx, dy, dz = info["widths"]
        grid = info["grid"]
        k_diag = np.asarray(self.mesh["k_diag"], dtype=np.float64)
        n = self.mesh["coords"].shape[0]
        rows: list[int] = []
        cols: list[int] = []
        values: list[float] = []
        rhs = np.zeros(n, dtype=np.float64)

        def add(i: int, j: int, value: float) -> None:
            rows.append(i)
            cols.append(j)
            values.append(float(value))

        for ix in range(x.size):
            for iy in range(y.size):
                for iz in range(z.size):
                    i = int(grid[ix, iy, iz])
                    if iz == 0 and self.bottom_type == "dirichlet":
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
                        (ix, iy, iz - 1, 2, float(dx[ix] * dy[iy]), float(z[iz] - z[iz - 1]) if iz else 0.0),
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
                        if self.bottom_type == "dirichlet" and jz == 0:
                            # Eliminate the known bottom value from the free-node
                            # equation.  This preserves an SPD matrix for CG.
                            rhs[i] += conductance * self.bottom_t
                            continue
                        add(i, j, -conductance)
                    add(i, i, diagonal)
        matrix = csc_matrix((values, (rows, cols)), shape=(n, n))
        matrix.sum_duplicates()
        return matrix, rhs

    def solve(self, q: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
        weights = np.asarray(self.mesh["info"]["weights"], dtype=np.float64)
        rhs = self.base_rhs + np.asarray(q, dtype=np.float64).reshape(-1) * weights
        bottom_mask = np.isclose(self.mesh["coords"][:, 2], self.mesh["z"][0], atol=1e-15)
        if self.bottom_type == "dirichlet":
            rhs[bottom_mask] = self.bottom_t
        initial = np.full(rhs.shape, self.top_t, dtype=np.float64)
        temperature, info = cg(
            self.matrix,
            rhs,
            x0=initial,
            rtol=1.0e-11,
            atol=0.0,
            maxiter=20000,
            M=self.preconditioner,
        )
        if info != 0:
            raise GenerationError(f"package-path CG did not converge: info={info}")
        temperature = np.asarray(temperature, dtype=np.float64)
        residual = float(np.linalg.norm(self.matrix.dot(temperature) - rhs) / max(np.linalg.norm(rhs), 1.0))
        if not np.all(np.isfinite(temperature)) or not math.isfinite(residual):
            raise GenerationError("non-finite package-path solve")
        info = self.mesh["info"]
        top_indices = info["grid"][:, :, -1]
        top_area = np.asarray(info["widths"][0])[:, None] * np.asarray(info["widths"][1])[None, :]
        top_flux = float(np.sum(self.top_h * (temperature[top_indices] - self.top_t) * top_area))
        if self.bottom_type == "adiabatic":
            bottom_flux = 0.0
        else:
            bottom_indices = info["grid"][:, :, 0]
            above_indices = info["grid"][:, :, 1]
            distance = float(self.mesh["z"][1] - self.mesh["z"][0])
            kb = self.mesh["k_diag"][bottom_indices, 2]
            ka = self.mesh["k_diag"][above_indices, 2]
            conductance = (2.0 * kb * ka / (kb + ka)) * top_area / distance
            bottom_flux = float(np.sum(conductance * (temperature[above_indices] - temperature[bottom_indices])))
        return temperature, {
            "linear_residual": residual,
            "top_heat_flux_W": top_flux,
            "bottom_heat_flux_W": bottom_flux,
        }


def _build_sources(
    case: Mapping[str, Any], registry: Mapping[str, Any], physics: Mapping[str, Any], mesh: Mapping[str, Any]
) -> tuple[np.ndarray, list[dict[str, Any]], dict[str, float]]:
    topology = registry["topologies"][case["topology"]]
    package_power = float(case["package_total_power_W"])
    geometries = [p1b._bbox_geometry(source["bbox_fraction_xy"], physics["footprint_m"]) for source in topology["sources"]]
    total_area = float(sum(area for _, area in geometries))
    expected_area = float(registry["power_contract"]["topology_total_declared_source_area_m2"])
    if not math.isclose(total_area, expected_area, rel_tol=0.0, abs_tol=1e-15):
        raise GenerationError("source area contract changed")
    coords = np.asarray(mesh["coords"])
    weights = np.asarray(mesh["info"]["weights"])
    layer_ids = np.asarray(mesh["layer_ids"])
    q = np.zeros(coords.shape[0], dtype=np.float64)
    occupied = np.zeros(coords.shape[0], dtype=bool)
    rows: list[dict[str, Any]] = []
    layer_powers: defaultdict[str, float] = defaultdict(float)
    for index, (source, (bbox, area)) in enumerate(zip(topology["sources"], geometries, strict=True)):
        layer_name = str(source["layer"])
        layer_index = int(mesh["layer_index"][layer_name])
        lower_face = float(mesh["boundaries"][layer_index])
        mask = (
            (layer_ids == layer_index)
            & (coords[:, 0] >= bbox["x"][0]) & (coords[:, 0] <= bbox["x"][1])
            & (coords[:, 1] >= bbox["y"][0]) & (coords[:, 1] <= bbox["y"][1])
        )
        # Preserve the historical P1b z-node realization: only the lower-die
        # lower face was excluded because it coincided with path A Dirichlet.
        if layer_name == "silicon_die_lower":
            mask &= ~np.isclose(coords[:, 2], lower_face, atol=1e-15)
        if np.any(mask & occupied):
            raise GenerationError(f"{case['id']}: overlapping source volumes")
        occupied |= mask
        x_count = int(np.unique(coords[mask, 0]).size)
        y_count = int(np.unique(coords[mask, 1]).size)
        z_count = int(np.unique(coords[mask, 2]).size)
        count = int(np.sum(mask))
        if count < int(physics["source_min_control_volume_count"]):
            raise GenerationError(f"{case['id']}: underresolved source")
        source_power = package_power * area / total_area
        volume = float(np.sum(weights[mask]))
        q_density = source_power / volume
        q[mask] = q_density
        layer_powers[layer_name] += source_power
        layer = physics["layers_bottom_to_top"][layer_index]
        rows.append({
            "sample_id": str(case["id"]), "path": str(case["path"]), "topology": str(case["topology"]),
            "source_id": f"src_{index:02d}", "active_layer": layer_name, "active_layer_index": layer_index,
            "bbox_m": {**bbox, "z": [float(mesh["boundaries"][layer_index]), float(mesh["boundaries"][layer_index + 1])]},
            "declared_source_area_m2": area, "source_area_fraction": area / total_area,
            "source_power_W": source_power, "surface_power_density_W_m2": source_power / area,
            "geometric_source_volume_m3": area * float(layer["thickness_m"]),
            "realized_source_control_volume_m3": volume, "q_W_m3": q_density,
            "covered_control_volume_count": count, "covered_x_node_count": x_count,
            "covered_y_node_count": y_count, "covered_z_node_count": z_count,
            "resolved_x_interval_count": x_count - 1, "resolved_y_interval_count": y_count - 1,
            "physical_layer_z_interval_count": int(layer["z_intervals"]),
        })
    if not math.isclose(float(np.dot(q, weights)), package_power, rel_tol=1e-12, abs_tol=1e-13):
        raise GenerationError(f"{case['id']}: integrated source power mismatch")
    return q, rows, dict(sorted(layer_powers.items()))


def _point_inputs(
    points: np.ndarray, physics: Mapping[str, Any], mesh: Mapping[str, Any], sources: Sequence[Mapping[str, Any]]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    boundaries = np.asarray(mesh["boundaries"])
    layer_ids = np.minimum(
        np.searchsorted(boundaries[1:], points[:, 2], side="right"),
        len(physics["layers_bottom_to_top"]) - 1,
    ).astype(np.int32)
    k_table = []
    for layer in physics["layers_bottom_to_top"]:
        if "k_xyz_W_mK" in layer:
            k_table.append(list(map(float, layer["k_xyz_W_mK"])))
        else:
            value = float(layer["k_W_mK"])
            k_table.append([value, value, value])
    k = np.asarray(k_table)[layer_ids]
    q = np.zeros(points.shape[0], dtype=np.float64)
    for source in sources:
        bbox = source["bbox_m"]
        mask = (
            (layer_ids == int(source["active_layer_index"]))
            & (points[:, 0] >= bbox["x"][0]) & (points[:, 0] <= bbox["x"][1])
            & (points[:, 1] >= bbox["y"][0]) & (points[:, 1] <= bbox["y"][1])
            & (points[:, 2] >= bbox["z"][0]) & (points[:, 2] <= bbox["z"][1])
        )
        if source["active_layer"] == "silicon_die_lower":
            mask &= ~np.isclose(points[:, 2], bbox["z"][0], atol=1e-15)
        q[mask] += float(source["q_W_m3"])
    return layer_ids, k, q


def _field_diagnostics(
    *, sample_id: str, path_label: str, topology: str, power: float, temperature: np.ndarray,
    q: np.ndarray, mesh: Mapping[str, Any], physics: Mapping[str, Any], solver_audit: Mapping[str, float]
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    weights = np.asarray(mesh["info"]["weights"])
    ambient = float(physics["boundary_conditions"]["top"]["T_inf_K"])
    delta = temperature - ambient
    top_flux = float(solver_audit["top_heat_flux_W"])
    bottom_flux = float(solver_audit["bottom_heat_flux_W"])
    top_idx = mesh["info"]["grid"][:, :, -1]
    bottom_idx = mesh["info"]["grid"][:, :, 0]
    top_mean = float(np.mean(temperature[top_idx]))
    bottom_mean = float(np.mean(temperature[bottom_idx]))
    junction = float(np.dot(temperature * q, weights) / power)
    peak = float(np.max(delta))
    bottom_type = str(physics["boundary_conditions"]["bottom"]["type"])
    metrics: dict[str, Any] = {
        "package_total_power_W": power,
        "peak_deltaT_K": peak,
        "mean_deltaT_K": float(np.dot(delta, weights) / np.sum(weights)),
        "Rth_peak_K_W": peak / power,
        "top_heat_fraction": top_flux / power,
        "bottom_heat_fraction": bottom_flux / power,
        "energy_balance_relative_error": (power - top_flux - bottom_flux) / power,
        "linear_residual": float(solver_audit["linear_residual"]),
        "junction_temperature_K": junction,
        "top_surface_mean_temperature_K": top_mean,
        "board_reference_temperature_K": bottom_mean if bottom_type == "dirichlet" else None,
        "junction_to_top_R_K_W": (junction - top_mean) / power,
        "junction_to_board_R_K_W": (junction - bottom_mean) / power if bottom_type == "dirichlet" else None,
        "junction_to_ambient_R_K_W": (junction - ambient) / power,
        "junction_to_top_path_R_K_W": (junction - top_mean) / top_flux if top_flux > 0 else None,
        "junction_to_board_path_R_K_W": (junction - bottom_mean) / bottom_flux if bottom_flux > 0 else None,
        "junction_to_board_status": "defined_at_isothermal_board_exterior" if bottom_type == "dirichlet" else "not_applicable_no_board_or_bottom_heat_path",
        "in_30_80_K_window": bool(30.0 <= peak <= 80.0),
    }
    grid_t = temperature.reshape(mesh["info"]["shape"])
    layer_rows: list[dict[str, Any]] = []
    interface_rows: list[dict[str, Any]] = []
    z = np.asarray(mesh["z"])
    for index, layer in enumerate(physics["layers_bottom_to_top"]):
        z0, z1 = float(mesh["boundaries"][index]), float(mesh["boundaries"][index + 1])
        i0 = int(np.argmin(np.abs(z - z0)))
        i1 = int(np.argmin(np.abs(z - z1)))
        bottom_plane = float(np.mean(grid_t[:, :, i0]))
        top_plane = float(np.mean(grid_t[:, :, i1]))
        mask = np.asarray(mesh["layer_ids"]) == index
        layer_rows.append({
            "sample_id": sample_id, "path": path_label, "topology": topology,
            "package_total_power_W": power, "layer_index": index, "layer_id": str(layer["id"]),
            "bottom_plane_mean_temperature_K": bottom_plane,
            "top_plane_mean_temperature_K": top_plane,
            "top_minus_bottom_temperature_K": top_plane - bottom_plane,
            "absolute_axial_temperature_drop_K": abs(top_plane - bottom_plane),
            "cv_mean_temperature_K": float(np.dot(temperature[mask], weights[mask]) / np.sum(weights[mask])),
        })
        if index + 1 < len(physics["layers_bottom_to_top"]):
            lower_adj = max(i1 - 1, 0)
            upper_adj = min(i1 + 1, z.size - 1)
            interface_rows.append({
                "sample_id": sample_id, "path": path_label, "topology": topology,
                "package_total_power_W": power, "interface_index": index,
                "lower_layer": str(layer["id"]),
                "upper_layer": str(physics["layers_bottom_to_top"][index + 1]["id"]),
                "perfect_contact_temperature_jump_K": 0.0,
                "adjacent_plane_upper_minus_lower_K": float(np.mean(grid_t[:, :, upper_adj]) - np.mean(grid_t[:, :, lower_adj])),
            })
    return metrics, layer_rows, interface_rows


def _baseline_replay(
    registry: Mapping[str, Any]
) -> tuple[dict[tuple[str, float], dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    baseline = registry["baseline_A"]
    manifest_path = REPO_ROOT / baseline["manifest_path"]
    if _sha256(manifest_path) != baseline["manifest_sha256"]:
        raise GenerationError("P1b baseline manifest SHA256 mismatch")
    dataset = REPO_ROOT / baseline["dataset_path"]
    if not dataset.is_dir():
        raise GenerationError(f"P1b baseline dataset unavailable: {dataset}")
    p1b_registry = yaml.safe_load((REPO_ROOT / baseline["case_registry_path"]).read_text(encoding="utf-8"))
    physics = p1b_registry["physics"]
    mesh = p1a._build_mesh(physics)
    solver = p1a.LayeredFvmSolver(mesh, physics)
    cases = {str(case["id"]): case for case in p1b_registry["cases"]}
    rows: dict[tuple[str, float], dict[str, Any]] = {}
    layer_rows: list[dict[str, Any]] = []
    interface_rows: list[dict[str, Any]] = []
    for topology, powers in baseline["pairing"].items():
        for power_text, sample_id in powers.items():
            case = cases[str(sample_id)]
            q, _, _ = p1b._build_area_weighted_sources(case, p1b_registry, mesh)
            temperature, solver_audit = solver.solve(q)
            metrics, layers, interfaces = _field_diagnostics(
                sample_id=str(sample_id), path_label="A_near_dirichlet", topology=str(topology),
                power=float(power_text), temperature=temperature, q=q, mesh=mesh,
                physics=physics, solver_audit=solver_audit,
            )
            tracked = json.loads((dataset / str(sample_id) / "sample_meta.json").read_text(encoding="utf-8"))["metrics"]
            for key in ("peak_deltaT_K", "mean_deltaT_K", "Rth_peak_K_W", "top_heat_fraction", "bottom_heat_fraction"):
                if not math.isclose(float(metrics[key]), float(tracked[key]), rel_tol=1e-9, abs_tol=1e-11):
                    raise GenerationError(f"baseline A replay mismatch: {sample_id} {key}")
            projection = json.loads((dataset / str(sample_id) / "sample_meta.json").read_text(encoding="utf-8"))["operator_projection"]
            metrics["solver_peak_minus_projected_peak_K"] = float(projection["solver_peak_minus_projected_peak_K"])
            metrics["solver_mean_minus_projected_mean_K"] = None
            metrics["sample_id"] = str(sample_id)
            metrics["path"] = "A_near_dirichlet"
            metrics["topology"] = str(topology)
            metrics["baseline_replay_only"] = True
            rows[(str(topology), float(power_text))] = metrics
            layer_rows.extend(layers)
            interface_rows.extend(interfaces)
    return rows, layer_rows, interface_rows


def generate(args: argparse.Namespace) -> dict[str, Any]:
    registry = _load_registry(args.cases)
    plan = [{"sample_id": case["id"], "path": case["path"], "topology": case["topology"], "package_total_power_W": float(case["package_total_power_W"])} for case in registry["cases"]]
    if args.dry_run:
        return {
            "schema_version": SCHEMA, "mode": "dry_run", "sample_count": 8,
            "case_plan": plan, "new_solver_calls": 8, "baseline_A_replay_solver_calls": 4,
            "new_dataset_writes": 8, "baseline_A_dataset_writes": 0,
            "training_runs": 0, "model_inference_runs": 0,
            "peak_deltaT_filtering": False, "power_back_calculation": False,
        }
    if args.dataset.exists():
        raise GenerationError(f"refusing to overwrite existing dataset: {args.dataset}")

    baseline_rows, layer_rows, interface_rows = _baseline_replay(registry)
    path_cache: dict[str, tuple[dict[str, Any], dict[str, Any], PackagePathSolver]] = {}
    for path_id in registry["paths"]:
        physics = _path_physics(registry, path_id)
        mesh = _build_mesh(physics)
        path_cache[path_id] = (physics, mesh, PackagePathSolver(mesh, physics))

    sample_rows: list[dict[str, Any]] = []
    source_rows_csv: list[dict[str, Any]] = []
    manifest_rows: list[dict[str, Any]] = []
    args.dataset.parent.mkdir(parents=True, exist_ok=True)
    temp_root = Path(tempfile.mkdtemp(prefix=f".{args.dataset.name}.", dir=args.dataset.parent))
    try:
        for case in registry["cases"]:
            sample_id = str(case["id"])
            path_id = str(case["path"])
            topology = str(case["topology"])
            power = float(case["package_total_power_W"])
            physics, mesh, solver = path_cache[path_id]
            q, sources, layer_powers = _build_sources(case, registry, physics, mesh)
            points, point_strata, point_seed = p1a._sample_points_before_labels(
                base_seed=int(registry["seed"]), sample_id=sample_id, physics=physics, mesh=mesh, sources=sources,
            )
            points_sha = p1a._array_sha256(points)
            temperature, solver_audit = solver.solve(q)
            metrics, layers, interfaces = _field_diagnostics(
                sample_id=sample_id, path_label=path_id, topology=topology, power=power,
                temperature=temperature, q=q, mesh=mesh, physics=physics, solver_audit=solver_audit,
            )
            layer_rows.extend(layers)
            interface_rows.extend(interfaces)
            interpolator = RegularGridInterpolator(
                (mesh["x"], mesh["y"], mesh["z"]), temperature.reshape(mesh["info"]["shape"]),
                method="linear", bounds_error=True,
            )
            point_temperature = np.asarray(interpolator(points), dtype=np.float64)
            point_layers, point_k, point_q = _point_inputs(points, physics, mesh, sources)
            point_bc = p1a._bc_features(points, physics, mesh)
            peak_gap = float(metrics["peak_deltaT_K"] - np.max(point_temperature - solver.top_t))
            mean_gap = float(metrics["mean_deltaT_K"] - np.mean(point_temperature - solver.top_t))
            metrics["solver_peak_minus_projected_peak_K"] = peak_gap
            metrics["solver_mean_minus_projected_mean_K"] = mean_gap
            arrays = {
                "coords.npy": points.astype(np.float64),
                "temperature.npy": point_temperature[:, None].astype(np.float64),
                "deltaT.npy": (point_temperature - solver.top_t)[:, None].astype(np.float64),
                "k_field.npy": point_k.astype(np.float64),
                "q_field.npy": point_q[:, None].astype(np.float64),
                "layer_id.npy": point_layers[:, None].astype(np.int32),
                "bc_features.npy": point_bc.astype(np.float64),
            }
            source_payload = []
            for source in sources:
                enriched = {**source, "active_layer_power_W": layer_powers[source["active_layer"]], "package_total_power_W": power}
                source_payload.append(enriched)
                source_rows_csv.append({key: value for key, value in enriched.items() if key not in {"bbox_m", "active_layer_index"}})
            meta = {
                "schema_version": SCHEMA, "sample_id": sample_id, "dataset_id": registry["dataset_id"],
                "stack_template_id": "logic_package", "package_path": path_id, "topology": topology,
                "package_power_provenance": "explicit_user_instruction",
                "power_allocation_mode": "proportional_to_declared_source_planform_area",
                "power_was_Rth_inferred": False, "material_parameters_tuned_after_solve": False,
                "active_layers": sorted(layer_powers), "active_layer_power_W": layer_powers,
                "sources": source_payload, "layers_bottom_to_top": physics["layers_bottom_to_top"],
                "boundary_conditions": physics["boundary_conditions"], "contact": physics["contact"],
                "solver_mesh": {
                    "type": "layer_aligned_node_control_volume", "shape": list(mesh["info"]["shape"]),
                    "intervals_xyz": list(map(int, physics["solver_mesh_intervals_xyz"])),
                    "node_count": int(mesh["coords"].shape[0]),
                    "minimum_source_control_volume_count": min(int(s["covered_control_volume_count"]) for s in sources),
                    "minimum_source_in_plane_interval_count": min(min(int(s["resolved_x_interval_count"]), int(s["resolved_y_interval_count"])) for s in sources),
                    "source_face_policy": registry["common_physics"]["source_face_policy"],
                    "axis_sha256": {axis: p1a._array_sha256(mesh[axis]) for axis in ("x", "y", "z")},
                },
                "operator_projection": {
                    "point_count": 1024, "point_seed": point_seed,
                    "point_schema": "v6_p1c_irregular_points_v1",
                    "point_coordinates_sha256": points_sha,
                    "point_coordinates_frozen_before_temperature_solve": True,
                    "strata_counts": dict(sorted(Counter(point_strata).items())),
                    "interpolation_method": "scipy_regular_grid_linear",
                    "solver_peak_minus_projected_peak_K": peak_gap,
                    "solver_mean_minus_projected_mean_K": mean_gap,
                    "label_inputs_used_for_point_selection": [],
                },
                "metrics": metrics,
                "guardrails": {
                    "baseline_A_regenerated": False, "peak_deltaT_filtering": False,
                    "peak_deltaT_resampling": False, "power_back_calculation": False,
                    "material_parameter_tuning": False, "sample_replacement": False,
                    "training_runs": 0, "model_inference_runs": 0,
                },
            }
            hashes = p1a._write_sample(temp_root / sample_id, arrays, meta)
            manifest_rows.append({
                "sample_id": sample_id, "path": path_id, "topology": topology,
                "package_total_power_W": power, "sample_dir": sample_id,
                "file_sha256": hashes, "point_coordinates_sha256": points_sha,
            })
            sample_rows.append({"sample_id": sample_id, "path": path_id, "topology": topology, **metrics})
        manifest = {
            "schema_version": SCHEMA, "dataset_id": registry["dataset_id"],
            "case_registry": _repo_path(args.cases), "case_registry_sha256": _sha256(args.cases),
            "sample_count": 8, "samples": manifest_rows,
            "guardrails": {"baseline_A_regenerated": False, "generated_samples": 8, "training_runs": 0, "model_inference_runs": 0},
        }
        _json_dump(temp_root / "manifest.json", manifest)
        temp_root.rename(args.dataset)
    except Exception:
        shutil.rmtree(temp_root, ignore_errors=True)
        raise

    _json_dump(args.manifest_json, manifest)
    paired_rows: list[dict[str, Any]] = []
    all_rows = list(baseline_rows.values()) + sample_rows
    new_by_key = {(row["path"], row["topology"], float(row["package_total_power_W"])): row for row in sample_rows}
    for topology in registry["topologies"]:
        for power in map(float, registry["power_contract"]["package_total_power_W"]):
            a = baseline_rows[(topology, power)]
            for path_id in registry["paths"]:
                b = new_by_key[(path_id, topology, power)]
                paired_rows.append({
                    "topology": topology, "package_total_power_W": power,
                    "baseline_A_sample_id": a["sample_id"], "comparison_sample_id": b["sample_id"],
                    "comparison_path": path_id,
                    **{f"A_{key}": a[key] for key in (
                        "peak_deltaT_K", "mean_deltaT_K", "Rth_peak_K_W", "top_heat_fraction", "bottom_heat_fraction",
                        "junction_to_top_R_K_W", "junction_to_board_R_K_W", "energy_balance_relative_error",
                        "solver_peak_minus_projected_peak_K",
                    )},
                    **{f"comparison_{key}": b[key] for key in (
                        "peak_deltaT_K", "mean_deltaT_K", "Rth_peak_K_W", "top_heat_fraction", "bottom_heat_fraction",
                        "junction_to_top_R_K_W", "junction_to_board_R_K_W", "energy_balance_relative_error",
                        "solver_peak_minus_projected_peak_K",
                    )},
                    "peak_deltaT_ratio_vs_A": float(b["peak_deltaT_K"]) / float(a["peak_deltaT_K"]),
                    "top_heat_fraction_change_vs_A": float(b["top_heat_fraction"]) - float(a["top_heat_fraction"]),
                })

    numeric_keys = (
        "peak_deltaT_K", "mean_deltaT_K", "Rth_peak_K_W", "top_heat_fraction", "bottom_heat_fraction",
        "energy_balance_relative_error", "solver_peak_minus_projected_peak_K", "solver_mean_minus_projected_mean_K",
    )
    audit = {
        "schema_version": SCHEMA, "dataset_id": registry["dataset_id"],
        "dataset_path": _repo_path(args.dataset), "dataset_manifest_sha256": _sha256(args.dataset / "manifest.json"),
        "sample_count_new": 8, "baseline_A_generated_samples": 0, "baseline_A_solver_replay_count": 4,
        "path_case_counts": dict(sorted(Counter(row["path"] for row in sample_rows).items())),
        "literature_evidence": registry["literature_evidence"],
        "summary_by_path": {
            path: {
                key: {
                    "min": float(np.min([float(row[key]) for row in all_rows if row["path"] == path and row[key] is not None])),
                    "median": float(np.median([float(row[key]) for row in all_rows if row["path"] == path and row[key] is not None])),
                    "max": float(np.max([float(row[key]) for row in all_rows if row["path"] == path and row[key] is not None])),
                }
                for key in numeric_keys if any(row[key] is not None for row in all_rows if row["path"] == path)
            }
            for path in ("A_near_dirichlet", *registry["paths"].keys())
        },
        "top_path_assessment": {
            "B_remote_dirichlet_top_dominant_all_cases": all(float(row["top_heat_fraction"]) > 0.5 for row in sample_rows if row["path"] == "B_remote_dirichlet"),
            "C0_adiabatic_top_fraction_equals_one": all(math.isclose(float(row["top_heat_fraction"]), 1.0, rel_tol=0, abs_tol=1e-8) for row in sample_rows if row["path"] == "C0_bottom_adiabatic"),
            "C0_board_resistance_defined": False,
        },
        "integrity": {
            "all_new_metrics_finite_except_declared_not_applicable": all(
                math.isfinite(float(value)) for row in sample_rows for key, value in row.items()
                if isinstance(value, (float, int)) and not isinstance(value, bool)
            ),
            "max_abs_energy_balance_relative_error": max(abs(float(row["energy_balance_relative_error"])) for row in sample_rows),
            "max_abs_projection_peak_gap_K": max(abs(float(row["solver_peak_minus_projected_peak_K"])) for row in sample_rows),
            "point_count_per_sample": 1024,
        },
        "guardrails": {
            "new_samples_generated": 8, "baseline_A_regenerated": False,
            "expanded_samples": 0, "peak_deltaT_filtering": False,
            "peak_deltaT_resampling": False, "power_back_calculation": False,
            "material_parameter_tuning": False, "training_runs": 0, "model_inference_runs": 0,
        },
        "samples": sample_rows,
        "paired_comparisons": paired_rows,
    }
    _json_dump(args.audit_json, audit)
    _csv_write(args.samples_csv, sample_rows)
    _csv_write(args.sources_csv, source_rows_csv)
    _csv_write(args.layers_csv, layer_rows)
    _csv_write(args.interfaces_csv, interface_rows)
    _csv_write(args.paired_csv, paired_rows)
    return audit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--audit-json", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--manifest-json", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--samples-csv", type=Path, default=DEFAULT_SAMPLES)
    parser.add_argument("--sources-csv", type=Path, default=DEFAULT_SOURCES)
    parser.add_argument("--layers-csv", type=Path, default=DEFAULT_LAYERS)
    parser.add_argument("--interfaces-csv", type=Path, default=DEFAULT_INTERFACES)
    parser.add_argument("--paired-csv", type=Path, default=DEFAULT_PAIRED)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    for key, value in vars(args).items():
        if isinstance(value, Path) and not value.is_absolute():
            setattr(args, key, (REPO_ROOT / value).resolve())
    print(json.dumps(generate(args), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
