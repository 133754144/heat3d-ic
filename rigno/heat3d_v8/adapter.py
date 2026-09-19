"""Read-only MASS-HBM Track-A adapter; frozen V7/P1i remains untouched."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from .boundary import cuboid_external_faces
from .interface import load_interface_representation
from .schema import MassHBMCase, ProvenanceClass, V8OraclePhysicsView


CORE_LOCAL_FEATURES = (
    "k_x_W_mK",
    "k_y_W_mK",
    "k_z_W_mK",
    "q_W_m3",
    "cell_volume_m3",
)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _coordinates_and_volume(geometry: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    """Reconstruct the declared zero-origin cell-centre grid exactly.

    The handoff stores no absolute origin.  Zero origin is a canonical
    translation convention; spacings, extent and cell volumes are taken
    exactly from ``solver_geometry.json`` and are translation invariant.
    """

    shape = tuple(int(value) for value in geometry["shape_zyx"])
    nz, ny, nx = shape
    dx = float(geometry["dx_m"])
    dy = float(geometry["dy_m"])
    dz = np.asarray(geometry["z_cell_thicknesses_m"], dtype=np.float64)
    if dz.shape != (nz,) or np.any(~np.isfinite(dz)) or np.any(dz <= 0.0):
        raise ValueError("invalid z_cell_thicknesses_m")
    if not np.isclose(dx * nx, float(geometry["x_extent_mm"]) * 1.0e-3, rtol=1.0e-12):
        raise ValueError("x spacing/extent mismatch")
    if not np.isclose(dy * ny, float(geometry["y_extent_mm"]) * 1.0e-3, rtol=1.0e-12):
        raise ValueError("y spacing/extent mismatch")
    if not np.isclose(np.sum(dz), float(geometry["z_extent_mm"]) * 1.0e-3, rtol=1.0e-12):
        raise ValueError("z thickness/extent mismatch")
    x = (np.arange(nx, dtype=np.float64) + 0.5) * dx
    y = (np.arange(ny, dtype=np.float64) + 0.5) * dy
    z = np.cumsum(dz) - 0.5 * dz
    zz, yy, xx = np.meshgrid(z, y, x, indexing="ij")
    coords = np.column_stack((xx.reshape(-1), yy.reshape(-1), zz.reshape(-1)))
    volume = np.broadcast_to((dx * dy * dz)[:, None, None], shape).reshape(-1).copy()
    return coords, volume


class MassHBMReadOnlyAdapter:
    """Load one case without writing under the dataset root."""

    def __init__(self, dataset_root: str | Path):
        self.dataset_root = Path(dataset_root).resolve()
        if not self.dataset_root.is_dir():
            raise FileNotFoundError(self.dataset_root)

    def _case_path(self, case_directory: str) -> Path:
        case = (self.dataset_root / case_directory).resolve()
        try:
            case.relative_to(self.dataset_root)
        except ValueError as exc:
            raise ValueError("case_directory escapes dataset root") from exc
        if not case.is_dir():
            raise FileNotFoundError(case)
        return case

    def load_oracle_view(
        self,
        case_directory: str,
        *,
        fail_on_unresolved_internal_sink: bool = True,
    ) -> V8OraclePhysicsView:
        case_dir = self._case_path(case_directory)
        geometry = _read_json(case_dir / "structure" / "solver_geometry.json")
        labels = _read_json(case_dir / "labels.json")
        boundary_json = _read_json(case_dir / "boundary_conditions" / "boundary_conditions.json")
        shape = tuple(int(value) for value in geometry["shape_zyx"])
        coords, volume = _coordinates_and_volume(geometry)
        cell_count = int(np.prod(shape))

        q_grid = np.load(case_dir / "power_input" / "power_density_map.npy", mmap_mode="r", allow_pickle=False)
        target_grid = np.load(case_dir / "temperature_output" / "temperature_map.npy", mmap_mode="r", allow_pickle=False)
        with np.load(case_dir / "thermal_parameters" / "final_material_k_maps.npz", allow_pickle=False) as archive:
            expected_keys = {"kx_w_mk", "ky_w_mk", "kz_w_mk"}
            if set(archive.files) != expected_keys:
                raise ValueError(f"unexpected conductivity archive keys: {archive.files}")
            k_grid = np.stack(
                [archive["kx_w_mk"], archive["ky_w_mk"], archive["kz_w_mk"]],
                axis=-1,
            )
        if q_grid.shape != shape or target_grid.shape != shape or k_grid.shape != shape + (3,):
            raise ValueError("field shape does not match solver geometry")
        q = np.asarray(q_grid, dtype=np.float64).reshape(-1)
        target = np.asarray(target_grid, dtype=np.float64).reshape(-1)
        k = np.asarray(k_grid, dtype=np.float64).reshape(cell_count, 3)
        if not np.all(np.isfinite(q)) or not np.all(np.isfinite(target)):
            raise ValueError("q and target temperature must be finite")
        if not np.all(np.isfinite(k)) or np.any(k <= 0.0):
            raise ValueError("conductivity must be finite and positive")

        # No explicit valid-cell mask is included in the handoff.  Every stored
        # cell has a positive finite conductivity, hence every array cell is a
        # computational control volume.  This does not claim every cell is an
        # active device: filler/mold/coolant are also computational regions.
        valid_mask = np.ones(cell_count, dtype=bool)
        sink_rows = _read_csv(case_dir / "boundary_conditions" / "lc_sink_audit.csv")
        if sink_rows and fail_on_unresolved_internal_sink:
            raise ValueError(
                "internal sink is present but the handoff contains only aggregate "
                "audit counts, not the per-cell/per-face sink topology"
            )
        external = boundary_json["external_defaults"]
        reference_temperature = float(external["ambient_temperature_K"])
        boundary = cuboid_external_faces(
            shape_zyx=shape,
            dx_m=float(geometry["dx_m"]),
            dy_m=float(geometry["dy_m"]),
            z_cell_thicknesses_m=np.asarray(geometry["z_cell_thicknesses_m"], dtype=np.float64),
            top_h_W_m2K=float(external["top_htc_W_m2K"]),
            bottom_h_W_m2K=float(external["bottom_htc_W_m2K"]),
            side_h_W_m2K=float(external["side_htc_W_m2K"]),
            ambient_temperature_K=reference_temperature,
            control_volume_m3=volume,
            reference_temperature_K=reference_temperature,
        )
        interface = load_interface_representation(
            case_dir=case_dir,
            shape_zyx=shape,
            dx_m=float(geometry["dx_m"]),
            dy_m=float(geometry["dy_m"]),
            control_volume_m3=volume,
        )
        case = MassHBMCase(
            dataset_root=self.dataset_root,
            case_directory=case_directory,
            sample_id=str(labels["sample_id"]),
            architecture_metadata=str(labels.get("architecture", "UNKNOWN")),
            shape_zyx=shape,
            coords_m=coords,
            control_volume_m3=volume,
            valid_cell_mask=valid_mask,
            geometry_metadata={
                **geometry,
                "coordinate_origin_convention": "zero_origin_translation_only",
                "valid_mask_semantics": "all stored cells are computational; active-device mask unavailable",
                "internal_sink_rows": len(sink_rows),
            },
            boundary=boundary,
            interface=interface,
        )
        local_names = (
            CORE_LOCAL_FEATURES
            + boundary.node_feature_names
            + interface.node_feature_names
        )
        local = np.column_stack((k, q, volume, boundary.node_features, interface.node_features))
        provenance = {
            "k_x_W_mK": ProvenanceClass.ORACLE,
            "k_y_W_mK": ProvenanceClass.ORACLE,
            "k_z_W_mK": ProvenanceClass.ORACLE,
            "q_W_m3": ProvenanceClass.ORACLE,
            "cell_volume_m3": ProvenanceClass.PRE_SOLVE,
            **dict(boundary.provenance),
            **dict(interface.provenance),
        }
        if local.shape != (cell_count, len(local_names)) or not np.all(np.isfinite(local)):
            raise ValueError("V8 local feature construction failed")
        expected_power = float(labels["scalar_targets"]["total_power_W"])
        integrated_power = float(np.sum(q * volume))
        relative_power_error = abs(integrated_power - expected_power) / max(abs(expected_power), 1.0e-30)
        return V8OraclePhysicsView(
            case=case,
            k_diag_W_mK=k,
            q_W_m3=q,
            target_temperature_K=target,
            local_feature_names=local_names,
            local_features=local,
            feature_provenance=provenance,
            metadata={
                "reference_temperature_K": reference_temperature,
                "integrated_power_W": integrated_power,
                "metadata_total_power_W": expected_power,
                "power_relative_error": relative_power_error,
                "no_clipping": True,
                "unit_guessing": False,
                "internal_sink_mapping_status": (
                    "NOT_PRESENT" if not sink_rows else "UNRESOLVED_AGGREGATE_ONLY"
                ),
            },
        )


def normalize_oracle_local_features(view: V8OraclePhysicsView) -> np.ndarray:
    """Deterministic dimensionless transform for engineering smoke only."""

    values = np.asarray(view.local_features, dtype=np.float64)
    names = view.local_feature_names
    volume_ref = float(np.median(view.case.control_volume_m3))
    length_ref = volume_ref ** (1.0 / 3.0)
    h_ref = 1.0e4
    q_ref = 1.0e10
    t_ref = 100.0
    scales = []
    for name in names:
        if name.startswith("k_"):
            scales.append(100.0)
        elif name == "q_W_m3":
            scales.append(q_ref)
        elif name == "cell_volume_m3":
            scales.append(volume_ref)
        elif name == "boundary_area_over_volume_1_m":
            scales.append(1.0 / length_ref)
        elif "deltaT" in name or "flux_power" in name:
            scales.append(h_ref * t_ref / length_ref)
        elif "_G_" in name:
            scales.append(h_ref / length_ref)
        else:
            raise ValueError(f"no physics scale declared for feature {name}")
    scaled = values / np.asarray(scales, dtype=np.float64)
    transformed = np.sign(scaled) * np.log1p(np.abs(scaled))
    if not np.all(np.isfinite(transformed)):
        raise ValueError("normalized V8 local features are not finite")
    return transformed.astype(np.float32)
