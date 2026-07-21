#!/usr/bin/env python3
"""Shared finite-volume core for V6-P1d asymmetric dual-Robin calibration."""

from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import yaml
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import LinearOperator, cg

import generate_heat3d_v6_p1a_power_calibration_pilot as p1a
import generate_heat3d_v6_p1c_package_path_calibration as p1c


REPO_ROOT = Path(__file__).resolve().parent.parent
P1C_CONFIG = REPO_ROOT / "configs/heat3d_v6/v6_p1c_package_path_calibration_cases.yaml"
AMBIENT_K = 300.0


class P1dError(RuntimeError):
    pass


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_p1c_stack() -> list[dict[str, Any]]:
    payload = yaml.safe_load(P1C_CONFIG.read_text(encoding="utf-8"))
    path = payload["paths"]["B_remote_dirichlet"]
    layers = [dict(layer) for layer in path["prepended_layers_bottom_to_top"]]
    layers.extend(dict(layer) for layer in payload["base_stack_layers_bottom_to_top"])
    return layers


def build_physics(
    *, top_h: float, bottom_h: float, mesh_intervals: Sequence[int] = (64, 64, 56),
    layers: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    if not (500.0 <= float(top_h) <= 2500.0):
        raise P1dError(f"top h outside frozen range: {top_h}")
    if float(bottom_h) not in {1.0, 2.0} and not (20.0 <= float(bottom_h) <= 200.0):
        raise P1dError(f"bottom h outside frozen range/control values: {bottom_h}")
    return {
        "footprint_m": [0.010, 0.010],
        "layers_bottom_to_top": [dict(layer) for layer in (layers or load_p1c_stack())],
        "solver_mesh_intervals_xyz": list(map(int, mesh_intervals)),
        "source_min_control_volume_count": 128,
        "source_min_in_plane_intervals": 7,
        "boundary_conditions": {
            "top": {"type": "robin", "h_W_m2K": float(top_h), "T_inf_K": AMBIENT_K},
            "bottom": {"type": "robin", "h_W_m2K": float(bottom_h), "T_inf_K": AMBIENT_K},
            "sides": {"type": "adiabatic"},
        },
        "contact": {"type": "perfect", "R_contact_m2K_W": 0.0},
        "operator_projection": {
            "point_count": 1024,
            "strata": {"volume": 512, "source": 256, "interface": 128, "top": 64, "bottom": 64},
        },
    }


def build_mesh(physics: Mapping[str, Any]) -> dict[str, Any]:
    return p1c._build_mesh(physics)


class DualRobinSolver:
    def __init__(self, mesh: Mapping[str, Any], physics: Mapping[str, Any]) -> None:
        self.mesh = mesh
        self.physics = physics
        self.top_h = float(physics["boundary_conditions"]["top"]["h_W_m2K"])
        self.bottom_h = float(physics["boundary_conditions"]["bottom"]["h_W_m2K"])
        self.top_t = float(physics["boundary_conditions"]["top"]["T_inf_K"])
        self.bottom_t = float(physics["boundary_conditions"]["bottom"]["T_inf_K"])
        self.matrix, self.base_rhs = self._assemble()
        diagonal = np.asarray(self.matrix.diagonal(), dtype=np.float64)
        if np.any(diagonal <= 0.0):
            raise P1dError("dual-Robin matrix has non-positive diagonal")
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
                    area_z = float(dx[ix] * dy[iy])
                    diagonal = 0.0
                    if iz == 0:
                        robin = self.bottom_h * area_z
                        diagonal += robin
                        rhs[i] += robin * self.bottom_t
                    if iz == z.size - 1:
                        robin = self.top_h * area_z
                        diagonal += robin
                        rhs[i] += robin * self.top_t
                    neighbors = (
                        (ix - 1, iy, iz, 0, float(dy[iy] * dz[iz]), float(x[ix] - x[ix - 1]) if ix else 0.0),
                        (ix + 1, iy, iz, 0, float(dy[iy] * dz[iz]), float(x[ix + 1] - x[ix]) if ix + 1 < x.size else 0.0),
                        (ix, iy - 1, iz, 1, float(dx[ix] * dz[iz]), float(y[iy] - y[iy - 1]) if iy else 0.0),
                        (ix, iy + 1, iz, 1, float(dx[ix] * dz[iz]), float(y[iy + 1] - y[iy]) if iy + 1 < y.size else 0.0),
                        (ix, iy, iz - 1, 2, area_z, float(z[iz] - z[iz - 1]) if iz else 0.0),
                        (ix, iy, iz + 1, 2, area_z, float(z[iz + 1] - z[iz]) if iz + 1 < z.size else 0.0),
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
        matrix = csc_matrix((values, (rows, cols)), shape=(n, n))
        matrix.sum_duplicates()
        return matrix, rhs

    def solve(self, q: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
        weights = np.asarray(self.mesh["info"]["weights"], dtype=np.float64)
        rhs = self.base_rhs + np.asarray(q, dtype=np.float64).reshape(-1) * weights
        initial = np.full(rhs.shape, AMBIENT_K, dtype=np.float64)
        temperature, info = cg(
            self.matrix, rhs, x0=initial, rtol=1.0e-11, atol=0.0,
            maxiter=20000, M=self.preconditioner,
        )
        if info != 0:
            raise P1dError(f"dual-Robin CG did not converge: info={info}")
        temperature = np.asarray(temperature, dtype=np.float64)
        residual = float(np.linalg.norm(self.matrix.dot(temperature) - rhs) / max(np.linalg.norm(rhs), 1.0))
        grid = self.mesh["info"]["grid"]
        dx, dy, _ = self.mesh["info"]["widths"]
        area = np.asarray(dx)[:, None] * np.asarray(dy)[None, :]
        top_flux = float(np.sum(self.top_h * (temperature[grid[:, :, -1]] - self.top_t) * area))
        bottom_flux = float(np.sum(self.bottom_h * (temperature[grid[:, :, 0]] - self.bottom_t) * area))
        if not np.all(np.isfinite(temperature)) or not all(map(math.isfinite, (residual, top_flux, bottom_flux))):
            raise P1dError("non-finite dual-Robin solve")
        return temperature, {
            "linear_residual": residual,
            "top_heat_flux_W": top_flux,
            "bottom_heat_flux_W": bottom_flux,
        }


def source_boxes(total_area_m2: float, layout_seed: int = 0) -> list[dict[str, Any]]:
    total_area_m2 = float(total_area_m2)
    if total_area_m2 not in {16e-6, 32e-6, 48e-6, 64e-6}:
        raise P1dError(f"unregistered total source area: {total_area_m2}")
    per_source_area = total_area_m2 / 8.0
    width = 0.00125
    height = per_source_area / width
    rng = np.random.default_rng(int(layout_seed) + 610400)
    x_base = np.asarray([0.00125, 0.00375, 0.00625, 0.00875])
    rows: list[dict[str, Any]] = []
    for layer, y_base in (("silicon_die_lower", 0.00335), ("silicon_die_upper", 0.00665)):
        x_jitter = rng.uniform(-0.00022, 0.00022, size=4) if layout_seed else np.zeros(4)
        y_jitter = float(rng.uniform(-0.00012, 0.00012)) if layout_seed else 0.0
        for x_center in x_base + x_jitter:
            y_center = y_base + y_jitter
            bbox = [
                (x_center - width / 2.0) / 0.010,
                (x_center + width / 2.0) / 0.010,
                (y_center - height / 2.0) / 0.010,
                (y_center + height / 2.0) / 0.010,
            ]
            if min(bbox) < 0.0 or max(bbox) > 1.0:
                raise P1dError("source layout exceeds footprint")
            rows.append({"layer": layer, "bbox_fraction_xy": bbox})
    return rows


def build_sources(
    *, sample_id: str, total_power_W: float, total_area_m2: float, layout_seed: int,
    physics: Mapping[str, Any], mesh: Mapping[str, Any],
) -> tuple[np.ndarray, list[dict[str, Any]], dict[str, float]]:
    specs = source_boxes(total_area_m2, layout_seed)
    coords = np.asarray(mesh["coords"], dtype=np.float64)
    weights = np.asarray(mesh["info"]["weights"], dtype=np.float64)
    layer_ids = np.asarray(mesh["layer_ids"], dtype=np.int32)
    q = np.zeros(coords.shape[0], dtype=np.float64)
    occupied = np.zeros(coords.shape[0], dtype=bool)
    rows: list[dict[str, Any]] = []
    layer_powers: defaultdict[str, float] = defaultdict(float)
    lx, ly = map(float, physics["footprint_m"])
    declared_total = 0.0
    for index, spec in enumerate(specs):
        fx0, fx1, fy0, fy1 = map(float, spec["bbox_fraction_xy"])
        bbox_xy = {"x": [fx0 * lx, fx1 * lx], "y": [fy0 * ly, fy1 * ly]}
        area = (bbox_xy["x"][1] - bbox_xy["x"][0]) * (bbox_xy["y"][1] - bbox_xy["y"][0])
        declared_total += area
        layer_name = str(spec["layer"])
        layer_index = int(mesh["layer_index"][layer_name])
        lower_face = float(mesh["boundaries"][layer_index])
        mask = (
            (layer_ids == layer_index)
            & (coords[:, 0] >= bbox_xy["x"][0]) & (coords[:, 0] <= bbox_xy["x"][1])
            & (coords[:, 1] >= bbox_xy["y"][0]) & (coords[:, 1] <= bbox_xy["y"][1])
        )
        if layer_name == "silicon_die_lower":
            mask &= ~np.isclose(coords[:, 2], lower_face, atol=1e-15)
        if np.any(mask & occupied):
            raise P1dError(f"{sample_id}: overlapping source masks")
        occupied |= mask
        count = int(np.sum(mask))
        x_count = int(np.unique(coords[mask, 0]).size)
        y_count = int(np.unique(coords[mask, 1]).size)
        z_count = int(np.unique(coords[mask, 2]).size)
        if count < int(physics["source_min_control_volume_count"]):
            raise P1dError(f"{sample_id}: underresolved source CV count {count}")
        if min(x_count - 1, y_count - 1) < int(physics["source_min_in_plane_intervals"]):
            raise P1dError(f"{sample_id}: underresolved source planform")
        source_power = float(total_power_W) / 8.0
        volume = float(np.sum(weights[mask]))
        q_density = source_power / volume
        q[mask] = q_density
        layer_powers[layer_name] += source_power
        rows.append({
            "sample_id": sample_id, "source_id": f"src_{index:02d}",
            "active_layer": layer_name, "active_layer_index": layer_index,
            "bbox_m": {**bbox_xy, "z": [float(mesh["boundaries"][layer_index]), float(mesh["boundaries"][layer_index + 1])]},
            "declared_source_area_m2": area, "source_power_W": source_power,
            "surface_power_density_W_m2": source_power / area,
            "realized_source_control_volume_m3": volume, "q_W_m3": q_density,
            "covered_control_volume_count": count,
            "covered_x_node_count": x_count, "covered_y_node_count": y_count, "covered_z_node_count": z_count,
            "resolved_x_interval_count": x_count - 1, "resolved_y_interval_count": y_count - 1,
        })
    if not math.isclose(declared_total, float(total_area_m2), rel_tol=0.0, abs_tol=2e-18):
        raise P1dError(f"{sample_id}: declared area mismatch")
    if not math.isclose(float(np.dot(q, weights)), float(total_power_W), rel_tol=1e-12, abs_tol=1e-12):
        raise P1dError(f"{sample_id}: source power integration mismatch")
    return q, rows, dict(sorted(layer_powers.items()))


def field_metrics(
    *, temperature: np.ndarray, q: np.ndarray, total_power_W: float,
    mesh: Mapping[str, Any], solver_audit: Mapping[str, float],
) -> dict[str, Any]:
    weights = np.asarray(mesh["info"]["weights"], dtype=np.float64)
    power = float(total_power_W)
    top_flux = float(solver_audit["top_heat_flux_W"])
    bottom_flux = float(solver_audit["bottom_heat_flux_W"])
    grid = mesh["info"]["grid"]
    top_mean = float(np.mean(temperature[grid[:, :, -1]]))
    bottom_mean = float(np.mean(temperature[grid[:, :, 0]]))
    junction = float(np.dot(temperature * q, weights) / power)
    delta = temperature - AMBIENT_K
    r_top = (junction - AMBIENT_K) / top_flux
    r_bottom = (junction - AMBIENT_K) / bottom_flux
    r_effective = (junction - AMBIENT_K) / power
    parallel_effective = 1.0 / (1.0 / r_top + 1.0 / r_bottom)
    return {
        "package_total_power_W": power,
        "peak_deltaT_K": float(np.max(delta)),
        "mean_deltaT_K": float(np.dot(delta, weights) / np.sum(weights)),
        "Rth_peak_K_W": float(np.max(delta)) / power,
        "top_heat_flux_W": top_flux, "bottom_heat_flux_W": bottom_flux,
        "top_heat_fraction": top_flux / power, "bottom_heat_fraction": bottom_flux / power,
        "energy_balance_relative_error": (power - top_flux - bottom_flux) / power,
        "linear_residual": float(solver_audit["linear_residual"]),
        "junction_temperature_K": junction,
        "top_surface_mean_temperature_K": top_mean,
        "bottom_surface_mean_temperature_K": bottom_mean,
        "top_branch_R_ambient_K_W": r_top,
        "bottom_branch_R_ambient_K_W": r_bottom,
        "top_internal_R_junction_to_surface_K_W": (junction - top_mean) / top_flux,
        "bottom_internal_R_junction_to_surface_K_W": (junction - bottom_mean) / bottom_flux,
        "top_film_R_surface_to_ambient_K_W": (top_mean - AMBIENT_K) / top_flux,
        "bottom_film_R_surface_to_ambient_K_W": (bottom_mean - AMBIENT_K) / bottom_flux,
        "effective_R_junction_to_ambient_K_W": r_effective,
        "parallel_branch_R_effective_K_W": parallel_effective,
        "parallel_branch_relative_closure_error": (parallel_effective - r_effective) / r_effective,
        "in_30_80_K_window": bool(30.0 <= float(np.max(delta)) <= 80.0),
    }


def layer_interface_diagnostics(
    *, sample_id: str, temperature: np.ndarray, total_power_W: float,
    mesh: Mapping[str, Any], physics: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    weights = np.asarray(mesh["info"]["weights"], dtype=np.float64)
    grid_t = temperature.reshape(mesh["info"]["shape"])
    z = np.asarray(mesh["z"])
    layer_rows: list[dict[str, Any]] = []
    interface_rows: list[dict[str, Any]] = []
    for index, layer in enumerate(physics["layers_bottom_to_top"]):
        z0, z1 = float(mesh["boundaries"][index]), float(mesh["boundaries"][index + 1])
        i0 = int(np.argmin(np.abs(z - z0)))
        i1 = int(np.argmin(np.abs(z - z1)))
        bottom_plane = float(np.mean(grid_t[:, :, i0]))
        top_plane = float(np.mean(grid_t[:, :, i1]))
        mask = np.asarray(mesh["layer_ids"]) == index
        layer_rows.append({
            "sample_id": sample_id, "package_total_power_W": float(total_power_W),
            "layer_index": index, "layer_id": str(layer["id"]),
            "bottom_plane_mean_temperature_K": bottom_plane,
            "top_plane_mean_temperature_K": top_plane,
            "top_minus_bottom_temperature_K": top_plane - bottom_plane,
            "absolute_axial_temperature_drop_K": abs(top_plane - bottom_plane),
            "cv_mean_temperature_K": float(np.dot(temperature[mask], weights[mask]) / np.sum(weights[mask])),
        })
        if index + 1 < len(physics["layers_bottom_to_top"]):
            interface_rows.append({
                "sample_id": sample_id, "package_total_power_W": float(total_power_W),
                "interface_index": index, "interface_z_m": z1,
                "lower_layer": str(layer["id"]),
                "upper_layer": str(physics["layers_bottom_to_top"][index + 1]["id"]),
                "perfect_contact_temperature_jump_K": 0.0,
                "adjacent_plane_upper_minus_lower_K": float(np.mean(grid_t[:, :, min(i1 + 1, z.size - 1)]) - np.mean(grid_t[:, :, max(i1 - 1, 0)])),
            })
    return layer_rows, interface_rows


def point_inputs(
    points: np.ndarray, physics: Mapping[str, Any], mesh: Mapping[str, Any],
    sources: Sequence[Mapping[str, Any]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return p1c._point_inputs(points, physics, mesh, sources)


def point_coverage(
    points: np.ndarray, point_strata: Sequence[str], layer_ids: np.ndarray,
    mesh: Mapping[str, Any], physics: Mapping[str, Any],
) -> dict[str, Any]:
    layer_counts = Counter(int(value) for value in np.asarray(layer_ids).reshape(-1))
    interface_points = points[np.asarray(point_strata) == "interface"]
    interface_counts = []
    for index, boundary in enumerate(mesh["boundaries"][1:-1]):
        interface_counts.append({
            "interface_index": index,
            "lower_layer": physics["layers_bottom_to_top"][index]["id"],
            "upper_layer": physics["layers_bottom_to_top"][index + 1]["id"],
            "point_count": int(np.sum(np.isclose(interface_points[:, 2], float(boundary), atol=1e-15))),
        })
    return {
        "layer_point_counts": {
            physics["layers_bottom_to_top"][index]["id"]: int(layer_counts.get(index, 0))
            for index in range(len(physics["layers_bottom_to_top"]))
        },
        "interface_point_counts": interface_counts,
        "all_layers_covered": all(layer_counts.get(index, 0) > 0 for index in range(len(physics["layers_bottom_to_top"]))),
        "all_interfaces_covered": all(row["point_count"] > 0 for row in interface_counts),
    }


def summarize_attempt(
    *, attempt_id: str, family_id: str, top_h: float, bottom_h: float,
    total_power_W: float, total_area_m2: float, metrics: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "attempt_id": attempt_id, "family_id": family_id,
        "top_h_W_m2K": float(top_h), "bottom_h_W_m2K": float(bottom_h),
        "package_total_power_W": float(total_power_W),
        "total_source_area_mm2": float(total_area_m2) * 1e6,
        **{key: metrics[key] for key in (
            "peak_deltaT_K", "mean_deltaT_K", "Rth_peak_K_W",
            "top_heat_fraction", "bottom_heat_fraction",
            "top_branch_R_ambient_K_W", "bottom_branch_R_ambient_K_W",
            "effective_R_junction_to_ambient_K_W",
            "parallel_branch_relative_closure_error",
            "energy_balance_relative_error", "linear_residual", "in_30_80_K_window",
        )},
    }
