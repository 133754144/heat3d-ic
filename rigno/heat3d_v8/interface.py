"""MASS-HBM explicit-versus-homogenized interface semantics."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from .schema import ProvenanceClass, V8InterfaceRepresentation


INTERFACE_NODE_FEATURES = (
    "interface_G_over_volume_W_m3K",
    "interface_G_nx_over_volume_W_m3K",
    "interface_G_ny_over_volume_W_m3K",
    "interface_G_nz_over_volume_W_m3K",
)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _empty(mode: str, cell_count: int, guard: str) -> V8InterfaceRepresentation:
    return V8InterfaceRepresentation(
        mode=mode,
        lower_cell_index=np.empty(0, dtype=np.int64),
        upper_cell_index=np.empty(0, dtype=np.int64),
        area_m2=np.empty(0, dtype=np.float64),
        normal=np.empty((0, 3), dtype=np.float64),
        tbr_m2K_W=np.empty(0, dtype=np.float64),
        conductance_W_K=np.empty(0, dtype=np.float64),
        node_feature_names=INTERFACE_NODE_FEATURES,
        node_features=np.zeros((cell_count, len(INTERFACE_NODE_FEATURES)), dtype=np.float64),
        provenance={name: ProvenanceClass.ORACLE for name in INTERFACE_NODE_FEATURES},
        double_count_guard=guard,
        explicit_feature_safe=False,
    )


def load_interface_representation(
    *,
    case_dir: Path,
    shape_zyx: tuple[int, int, int],
    dx_m: float,
    dy_m: float,
    control_volume_m3: np.ndarray,
) -> V8InterfaceRepresentation:
    """Load converged TBR only when raw evidence proves separate resistance.

    ``rotated_homogenized_tensor`` cases must not receive a second Rint path:
    the HBM interface effect is already represented in the effective diagonal
    tensor.  ``explicit_series_z`` is accepted only when every detailed row
    carries the dataset's explicit no-double-count guard.
    """

    nz, ny, nx = shape_zyx
    cell_count = nz * ny * nx
    manifest = _read_csv(case_dir / "structure" / "physical_interface_manifest.csv")
    summary_mode = "UNKNOWN"
    if manifest:
        summary_mode = manifest[0].get("interface_representation", "") or manifest[0].get("representation", "")

    archive_path = case_dir / "thermal_parameters" / "final_interface_tbr_map.npz"
    with np.load(archive_path, allow_pickle=False) as archive:
        if tuple(archive.files) != ("interface_tbr_m2k_w",):
            raise ValueError(f"unexpected interface archive keys: {archive.files}")
        tbr_grid = np.asarray(archive["interface_tbr_m2k_w"], dtype=np.float64)
    expected_shape = (nz - 1, ny, nx)
    if tbr_grid.shape != expected_shape:
        raise ValueError(
            f"interface TBR shape {tbr_grid.shape} does not match z-face shape {expected_shape}"
        )
    if not np.all(np.isfinite(tbr_grid)) or np.any(tbr_grid < 0.0):
        raise ValueError("interface TBR must be finite and nonnegative")

    if summary_mode == "rotated_homogenized_tensor":
        if np.any(tbr_grid != 0.0):
            raise ValueError(
                "double-count guard: homogenized-tensor case has nonzero explicit TBR"
            )
        return _empty(
            mode=summary_mode,
            cell_count=cell_count,
            guard="PASS_HOMOGENIZED_IN_EFFECTIVE_K_EXPLICIT_TBR_FORBIDDEN",
        )
    if summary_mode != "explicit_series_z":
        raise ValueError(f"unsupported interface representation: {summary_mode!r}")

    detail = _read_csv(case_dir / "thermal_parameters" / "interfaces.csv")
    if not detail:
        raise ValueError("explicit_series_z requires nonempty interfaces.csv")
    guards = {row.get("double_count_check", "") for row in detail}
    if guards != {"PASS_interface_only_bonding_representation"}:
        raise ValueError(f"interface double-count guard failed: {sorted(guards)}")
    valid = tbr_grid > 0.0
    zface, yface, xface = np.nonzero(valid)
    if zface.size == 0:
        raise ValueError("explicit_series_z has no positive TBR faces")
    grid = np.arange(cell_count, dtype=np.int64).reshape(shape_zyx)
    lower = grid[zface, yface, xface]
    upper = grid[zface + 1, yface, xface]
    tbr = tbr_grid[valid]
    area = np.full(len(tbr), float(dx_m * dy_m), dtype=np.float64)
    conductance = area / tbr
    normal = np.tile(np.asarray((0.0, 0.0, 1.0)), (len(tbr), 1))
    volume = np.asarray(control_volume_m3, dtype=np.float64).reshape(-1)
    node_features = np.zeros((cell_count, len(INTERFACE_NODE_FEATURES)), dtype=np.float64)
    for cells, sign in ((lower, 1.0), (upper, -1.0)):
        density = conductance / volume[cells]
        np.add.at(node_features[:, 0], cells, density)
        np.add.at(node_features[:, 1], cells, density * normal[:, 0] * sign)
        np.add.at(node_features[:, 2], cells, density * normal[:, 1] * sign)
        np.add.at(node_features[:, 3], cells, density * normal[:, 2] * sign)
    return V8InterfaceRepresentation(
        mode=summary_mode,
        lower_cell_index=lower,
        upper_cell_index=upper,
        area_m2=area,
        normal=normal,
        tbr_m2K_W=tbr,
        conductance_W_K=conductance,
        node_feature_names=INTERFACE_NODE_FEATURES,
        node_features=node_features,
        provenance={name: ProvenanceClass.ORACLE for name in INTERFACE_NODE_FEATURES},
        double_count_guard="PASS_EXPLICIT_INTERFACE_ONLY_BONDING_REPRESENTATION",
        explicit_feature_safe=True,
    )
