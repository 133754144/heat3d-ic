"""Generic V8 global physics context derived from Track-A fields."""

from __future__ import annotations

import math

import numpy as np

from .adapter import physical_scale_features
from .schema import ProvenanceClass, V8OraclePhysicsView


V8_GLOBAL_CONTEXT_FEATURES = (
    "log_P_operator_W",
    "q_weighted_local_kz_W_mK",
    "q_weighted_inverse_kz_mK_W",
    "log_source_concentration",
    "source_z_centroid_normalized",
    "log_harmonic_kx_W_mK",
    "log_harmonic_ky_W_mK",
    "log_harmonic_kz_W_mK",
    "log_anisotropy_xy_over_z",
    "log_Lx_m",
    "log_Ly_m",
    "log_Lz_m",
    "log_xy_area_m2",
    "log_domain_volume_m3",
    "log_total_boundary_area_m2",
    "log_total_sink_G_W_K",
    "sink_G_weighted_T_offset_K_over_100K",
    "boundary_cv_fraction",
)


def _weighted_harmonic(values: np.ndarray, weights: np.ndarray) -> float:
    return float(np.sum(weights) / np.sum(weights / values))


def oracle_global_context(
    view: V8OraclePhysicsView,
) -> tuple[np.ndarray, dict[str, float], dict[str, ProvenanceClass]]:
    """Adapt V7-Full global physics concepts to generic V8 boundaries.

    The q/k aggregates are explicitly ORACLE for Track A.  Geometry scales and
    boundary aggregates are PRE_SOLVE for the current non-LC handoff.
    """

    volume = np.asarray(view.case.control_volume_m3, dtype=np.float64)
    q = np.maximum(np.asarray(view.q_W_m3, dtype=np.float64), 0.0)
    k = np.asarray(view.k_diag_W_mK, dtype=np.float64)
    power_weights = q * volume
    power = float(np.sum(power_weights))
    if power <= 0.0 or np.any(k <= 0.0):
        raise ValueError("V8 global context requires positive power and conductivity")
    normalized_power = power_weights / power
    volume_weights = volume / np.sum(volume)
    kz = k[:, 2]
    q_kz = float(np.sum(normalized_power * kz))
    q_inv_kz = float(np.sum(normalized_power / kz))
    q_mean = power / float(np.sum(volume))
    q_rms = math.sqrt(float(np.sum(np.square(q) * volume) / np.sum(volume)))
    concentration = q_rms / q_mean
    coords = np.asarray(view.case.coords_m, dtype=np.float64)
    lz = float(view.case.geometry_metadata["z_extent_mm"]) * 1.0e-3
    source_z = float(np.sum(normalized_power * coords[:, 2]) / lz)
    harmonic = [_weighted_harmonic(k[:, axis], volume_weights) for axis in range(3)]
    anisotropy = math.sqrt(harmonic[0] * harmonic[1]) / harmonic[2]
    scales = physical_scale_features(view)
    boundary = view.case.boundary
    area = float(np.sum(boundary.area_m2))
    conductance = float(np.sum(boundary.conductance_W_K))
    if area <= 0.0 or conductance <= 0.0:
        raise ValueError("V8 global context requires positive boundary area/conductance")
    sink_offset = float(
        np.sum(
            boundary.conductance_W_K
            * (boundary.sink_temperature_K - float(view.metadata["reference_temperature_K"]))
        ) / conductance
    )
    boundary_cells = np.unique(boundary.cell_index)
    boundary_fraction = float(np.sum(volume[boundary_cells]) / np.sum(volume))
    context = {
        "log_P_operator_W": math.log(power),
        "q_weighted_local_kz_W_mK": q_kz,
        "q_weighted_inverse_kz_mK_W": q_inv_kz,
        "log_source_concentration": math.log(concentration),
        "source_z_centroid_normalized": source_z,
        "log_harmonic_kx_W_mK": math.log(harmonic[0]),
        "log_harmonic_ky_W_mK": math.log(harmonic[1]),
        "log_harmonic_kz_W_mK": math.log(harmonic[2]),
        "log_anisotropy_xy_over_z": math.log(anisotropy),
        **scales,
        "log_total_boundary_area_m2": math.log(area),
        "log_total_sink_G_W_K": math.log(conductance),
        "sink_G_weighted_T_offset_K_over_100K": sink_offset / 100.0,
        "boundary_cv_fraction": boundary_fraction,
    }
    vector = np.asarray([context[name] for name in V8_GLOBAL_CONTEXT_FEATURES], dtype=np.float32)
    if not np.all(np.isfinite(vector)):
        raise ValueError("V8 global context contains non-finite values")
    oracle_names = set(V8_GLOBAL_CONTEXT_FEATURES[:9])
    provenance = {
        name: (ProvenanceClass.ORACLE if name in oracle_names else ProvenanceClass.PRE_SOLVE)
        for name in V8_GLOBAL_CONTEXT_FEATURES
    }
    return vector, context, provenance
