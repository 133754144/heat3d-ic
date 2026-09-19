"""Generic boundary-face physics and the lossy node-aggregation candidate A."""

from __future__ import annotations

from typing import Iterable

import numpy as np

from .schema import ProvenanceClass, V8BoundaryRepresentation


BOUNDARY_NODE_FEATURES = (
    "boundary_area_over_volume_1_m",
    "sink_G_over_volume_W_m3K",
    "sink_G_deltaT_over_volume_W_m3",
    "prescribed_flux_power_over_volume_W_m3",
    "sink_G_nx_over_volume_W_m3K",
    "sink_G_ny_over_volume_W_m3K",
    "sink_G_nz_over_volume_W_m3K",
    "sink_G_nxnx_over_volume_W_m3K",
    "sink_G_nyny_over_volume_W_m3K",
    "sink_G_nznz_over_volume_W_m3K",
    "sink_G_nxny_over_volume_W_m3K",
    "sink_G_nxnz_over_volume_W_m3K",
    "sink_G_nynz_over_volume_W_m3K",
)


def aggregate_boundary_faces(
    *,
    cell_count: int,
    control_volume_m3: np.ndarray,
    cell_index: np.ndarray,
    area_m2: np.ndarray,
    outward_normal: np.ndarray,
    conductance_W_K: np.ndarray,
    sink_temperature_K: np.ndarray,
    prescribed_flux_W_m2: np.ndarray,
    reference_temperature_K: float,
) -> np.ndarray:
    """Aggregate incident face moments without categorical orientation labels.

    Candidate A is intentionally lossy: it preserves total area, conductance,
    sink forcing, flux forcing, first normal moment and symmetric second normal
    moment, but not the individual face list or face-to-face separation.
    """

    volume = np.asarray(control_volume_m3, dtype=np.float64).reshape(-1)
    index = np.asarray(cell_index, dtype=np.int64).reshape(-1)
    area = np.asarray(area_m2, dtype=np.float64).reshape(-1)
    normal = np.asarray(outward_normal, dtype=np.float64)
    conductance = np.asarray(conductance_W_K, dtype=np.float64).reshape(-1)
    sink = np.asarray(sink_temperature_K, dtype=np.float64).reshape(-1)
    flux = np.asarray(prescribed_flux_W_m2, dtype=np.float64).reshape(-1)
    face_count = len(index)
    if normal.shape != (face_count, 3):
        raise ValueError("boundary normals must have shape [F,3]")
    for value, name in (
        (area, "area"),
        (conductance, "conductance"),
        (sink, "sink temperature"),
        (flux, "prescribed flux"),
        (normal, "normal"),
    ):
        if not np.all(np.isfinite(value)):
            raise ValueError(f"boundary {name} must be finite")
    if np.any(index < 0) or np.any(index >= cell_count):
        raise ValueError("boundary cell index is out of range")
    if np.any(area <= 0.0) or np.any(conductance < 0.0):
        raise ValueError("boundary area must be positive and conductance nonnegative")
    norm = np.linalg.norm(normal, axis=1)
    if not np.allclose(norm, 1.0, atol=1.0e-12, rtol=0.0):
        raise ValueError("boundary normals must be unit vectors")
    if volume.shape != (cell_count,) or np.any(~np.isfinite(volume)) or np.any(volume <= 0.0):
        raise ValueError("control volumes must be finite, positive and aligned")

    result = np.zeros((cell_count, len(BOUNDARY_NODE_FEATURES)), dtype=np.float64)
    inv_volume = 1.0 / volume[index]
    np.add.at(result[:, 0], index, area * inv_volume)
    g_density = conductance * inv_volume
    np.add.at(result[:, 1], index, g_density)
    np.add.at(
        result[:, 2], index,
        g_density * (sink - float(reference_temperature_K)),
    )
    np.add.at(result[:, 3], index, flux * area * inv_volume)
    for axis in range(3):
        np.add.at(result[:, 4 + axis], index, g_density * normal[:, axis])
    products = (
        normal[:, 0] * normal[:, 0],
        normal[:, 1] * normal[:, 1],
        normal[:, 2] * normal[:, 2],
        normal[:, 0] * normal[:, 1],
        normal[:, 0] * normal[:, 2],
        normal[:, 1] * normal[:, 2],
    )
    for offset, product in enumerate(products, start=7):
        np.add.at(result[:, offset], index, g_density * product)
    return result


def cuboid_external_faces(
    *,
    shape_zyx: tuple[int, int, int],
    dx_m: float,
    dy_m: float,
    z_cell_thicknesses_m: np.ndarray,
    top_h_W_m2K: float,
    bottom_h_W_m2K: float,
    side_h_W_m2K: float,
    ambient_temperature_K: float,
    control_volume_m3: np.ndarray,
    reference_temperature_K: float,
) -> V8BoundaryRepresentation:
    """Build exterior faces from the raw solver's declared full-box grid."""

    nz, ny, nx = shape_zyx
    indices = np.arange(nz * ny * nx, dtype=np.int64).reshape(shape_zyx)
    face_cells: list[np.ndarray] = []
    areas: list[np.ndarray] = []
    normals: list[np.ndarray] = []
    conductances: list[np.ndarray] = []
    sinks: list[np.ndarray] = []
    fluxes: list[np.ndarray] = []

    def add(cells: np.ndarray, area: np.ndarray | float, normal: tuple[float, float, float], h: float) -> None:
        flat = np.asarray(cells, dtype=np.int64).reshape(-1)
        area_values = np.broadcast_to(np.asarray(area, dtype=np.float64), flat.shape)
        face_cells.append(flat)
        areas.append(area_values)
        normals.append(np.tile(np.asarray(normal, dtype=np.float64), (len(flat), 1)))
        conductances.append(float(h) * area_values)
        sinks.append(np.full(len(flat), float(ambient_temperature_K), dtype=np.float64))
        fluxes.append(np.zeros(len(flat), dtype=np.float64))

    add(indices[-1], dx_m * dy_m, (0.0, 0.0, 1.0), top_h_W_m2K)
    add(indices[0], dx_m * dy_m, (0.0, 0.0, -1.0), bottom_h_W_m2K)
    z_area_y = np.repeat(np.asarray(z_cell_thicknesses_m) * dx_m, ny)
    z_area_x = np.repeat(np.asarray(z_cell_thicknesses_m) * dy_m, nx)
    add(indices[:, 0, :], z_area_x, (0.0, -1.0, 0.0), side_h_W_m2K)
    add(indices[:, -1, :], z_area_x, (0.0, 1.0, 0.0), side_h_W_m2K)
    add(indices[:, :, 0], z_area_y, (-1.0, 0.0, 0.0), side_h_W_m2K)
    add(indices[:, :, -1], z_area_y, (1.0, 0.0, 0.0), side_h_W_m2K)

    cell_index = np.concatenate(face_cells)
    area_m2 = np.concatenate(areas)
    outward_normal = np.concatenate(normals)
    conductance_W_K = np.concatenate(conductances)
    sink_temperature_K = np.concatenate(sinks)
    prescribed_flux_W_m2 = np.concatenate(fluxes)
    node_features = aggregate_boundary_faces(
        cell_count=nz * ny * nx,
        control_volume_m3=control_volume_m3,
        cell_index=cell_index,
        area_m2=area_m2,
        outward_normal=outward_normal,
        conductance_W_K=conductance_W_K,
        sink_temperature_K=sink_temperature_K,
        prescribed_flux_W_m2=prescribed_flux_W_m2,
        reference_temperature_K=reference_temperature_K,
    )
    return V8BoundaryRepresentation(
        cell_index=cell_index,
        area_m2=area_m2,
        outward_normal=outward_normal,
        conductance_W_K=conductance_W_K,
        sink_temperature_K=sink_temperature_K,
        prescribed_flux_W_m2=prescribed_flux_W_m2,
        is_internal=np.zeros(len(cell_index), dtype=bool),
        node_feature_names=BOUNDARY_NODE_FEATURES,
        node_features=node_features,
        provenance={name: ProvenanceClass.PRE_SOLVE for name in BOUNDARY_NODE_FEATURES},
        semantic_status="EXTERNAL_FULL_BOX_FACES_FROM_RAW_SOLVER_GEOMETRY",
    )
