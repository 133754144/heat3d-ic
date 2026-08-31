#!/usr/bin/env python3
"""Deterministic schema conversion for frozen semiconductor-native cases.

This utility does not train a model or solve a PDE. It preserves the released
benchmark's native nondimensional coefficients and writes pointwise Heat3D
inputs where the existing ``coords + k + q + BC`` schema is lossless. Surface
Neumann-power cases are deliberately emitted with a separate, non-model input
``top_neumann_flux_sensors`` array; they are marked incompatible with the
current 11-channel Heat3D contract rather than relabeling flux as volumetric q.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

FEATURE_NAMES = (
    "kx", "ky", "kz", "q", "is_top", "is_bottom", "is_side", "is_interior",
    "top_h", "bottom_h", "top_T_inf_minus_T_ref",
)
TEMPERATURE_SCALE_K = 25.0


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def cartesian_coords(nx: int, ny: int, nz: int, z_max: float) -> np.ndarray:
    x = np.linspace(0.0, 1.0, nx)
    y = np.linspace(0.0, 1.0, ny)
    z = np.linspace(0.0, z_max, nz)
    return np.stack(np.meshgrid(x, y, z, indexing="ij"), axis=-1).reshape(-1, 3)


def boundary_features(
    coords: np.ndarray, *, top_h: float, bottom_h: float, top_ambient_offset: float = 0.0
) -> np.ndarray:
    x, y, z = coords.T
    z_min, z_max = float(z.min()), float(z.max())
    top = np.isclose(z, z_max)
    bottom = np.isclose(z, z_min)
    lateral = np.isclose(x, 0) | np.isclose(x, 1) | np.isclose(y, 0) | np.isclose(y, 1)
    # P1i's four region flags are mutually exclusive. Top/bottom take
    # precedence at edges and corners; the lateral flag covers the remaining
    # side-wall points.
    side = lateral & ~(top | bottom)
    interior = ~(top | bottom | side)
    return np.column_stack(
        (
            top, bottom, side, interior,
            np.full(len(coords), top_h),
            np.full(len(coords), bottom_h),
            np.full(len(coords), top_ambient_offset),
        )
    ).astype(np.float32)


def htc_case(
    top_robin_k: float,
    bottom_robin_k: float,
    kind: str = "single_htc_bc",
) -> tuple[dict[str, np.ndarray], dict]:
    if kind not in {"single_htc_bc", "multi_htc_bc"}:
        raise ValueError(f"unsupported HTC benchmark kind: {kind}")
    coords = cartesian_coords(51, 51, 51, 0.55)
    conductivity = 0.2
    k_field = np.full((len(coords), 3), conductivity, dtype=np.float32)
    q = np.zeros((len(coords), 1), dtype=np.float32)
    # Released power cell spans native z interval indices [5, 6] of 11.
    in_power = (coords[:, 2] >= 0.25 - 1e-12) & (coords[:, 2] <= 0.30 + 1e-12)
    # Canonical exact dimensionalization: one released length unit is one
    # metre and DeltaT=25*(u-u_ambient) K. Thus the source coefficient in the
    # Kelvin PDE is 25 times the coefficient in the released u equation.
    q[in_power, 0] = TEMPERATURE_SCALE_K
    # DeepOHeat enforces u-0.2+(kappa/h)*du/dn=0, with its branch value named k.
    top_h = conductivity / top_robin_k
    bottom_h = conductivity / bottom_robin_k
    bc = boundary_features(coords, top_h=top_h, bottom_h=bottom_h)
    arrays = {"coords": coords.astype(np.float32), "features": np.concatenate((k_field, q, bc), axis=1)}
    metadata = {
        "benchmark_family": "DeepOHeat",
        "benchmark": kind,
        "representation": "lossless_released_nondimensional_PDE_coefficients",
        "feature_names": list(FEATURE_NAMES),
        "query_shape": [51, 51, 51],
        "target_available_in_upstream_release": False,
        "target_contract": "T_K=293.15+25*u; with Robin ambient u=0.2 use deltaT_K=25*(u-0.2)",
        "dimensionalization": "1 released length unit = 1 m; q_K=25*q_u; k unchanged; h=k/Robin_length",
        "no_solver_run": True,
    }
    return arrays, metadata


def select_array(path: Path, expected_shape: tuple[int, ...], index: int) -> np.ndarray:
    if path.suffix == ".txt":
        lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if lines[0] != "POWER MAP":
            raise ValueError(f"{path}: unsupported power-map text header")
        intervals = np.asarray([[float(value) for value in line.split()] for line in lines[2:]], dtype=np.float32)
        grid = np.zeros((intervals.shape[0] + 1, intervals.shape[1] + 1), dtype=np.float32)
        weights = np.zeros_like(grid)
        for dx in (0, 1):
            for dy in (0, 1):
                grid[dx:dx + intervals.shape[0], dy:dy + intervals.shape[1]] += intervals
                weights[dx:dx + intervals.shape[0], dy:dy + intervals.shape[1]] += 1
        raw = grid / weights / 0.00625
        return np.asarray(raw, dtype=np.float32)
    raw = np.load(path, allow_pickle=False, mmap_mode="r")
    if raw.shape == expected_shape:
        selected = raw
    elif raw.ndim == len(expected_shape) + 1 and raw.shape[1:] == expected_shape:
        selected = raw[index]
    else:
        raise ValueError(
            f"{path}: expected {expected_shape} or "
            f"[N,{','.join(map(str, expected_shape))}], got {raw.shape}"
        )
    # Convert only the selected sample. This keeps the official 8.16 GB
    # fs_train_volume.npy usable without loading the full training array.
    return np.asarray(selected, dtype=np.float32)


def volume_v1_arrays(
    power: np.ndarray, target_u: np.ndarray | None = None
) -> dict[str, np.ndarray]:
    power = np.asarray(power, dtype=np.float32)
    if power.shape != (101, 101):
        raise ValueError(f"expected one 101x101 volumetric power function, got {power.shape}")
    coords = cartesian_coords(101, 101, 56, 0.55)
    z = coords[:, 2]
    k_scalar = np.where(z < 0.10, 2.0, 0.1).astype(np.float32)
    interface = np.isclose(z, 0.10)
    k_scalar[interface] = 2 * 0.1 * 2.0 / (0.1 + 2.0)
    k_field = np.repeat(k_scalar[:, None], 3, axis=1)
    q = np.zeros((101, 101, 56), dtype=np.float32)
    q[:, :, 10] = TEMPERATURE_SCALE_K * power
    q[:, :, 11:15] = 2.0 * TEMPERATURE_SCALE_K * power[:, :, None]
    q[:, :, 15] = TEMPERATURE_SCALE_K * power
    bc = boundary_features(coords, top_h=0.1 / 2.0, bottom_h=0.1 / 40.0)
    arrays: dict[str, np.ndarray] = {
        "coords": coords.astype(np.float32),
        "features": np.concatenate((k_field, q.reshape(-1, 1), bc), axis=1),
    }
    if target_u is not None:
        target_u = np.asarray(target_u, dtype=np.float32)
        if target_u.shape != (101, 101, 56):
            raise ValueError(f"expected one 101x101x56 target field, got {target_u.shape}")
        arrays["temperature_K"] = (293.15 + 25.0 * target_u).reshape(-1, 1)
        # Both released Robin boundaries use ambient u=0.2. Heat3D's relative
        # target therefore uses 298.15 K as T_ref while preserving the exact
        # official absolute temperature above.
        arrays["deltaT_from_robin_ambient_K"] = (25.0 * (target_u - 0.2)).reshape(-1, 1)
    return arrays


def volume_v1_case(power_path: Path, target_path: Path | None, index: int) -> tuple[dict[str, np.ndarray], dict]:
    power = select_array(power_path, (101, 101), index)
    target_u = None if target_path is None else select_array(target_path, (101, 101, 56), index)
    arrays = volume_v1_arrays(power, target_u)
    metadata = {
        "benchmark_family": "DeepOHeat-v1",
        "benchmark": "volumetric",
        "representation": "lossless_with_respect_to_released_discrete_residual_coefficients",
        "feature_names": list(FEATURE_NAMES),
        "query_shape": [101, 101, 56],
        "power_input_sha256": sha256(power_path),
        "source_sample_index": index,
        "official_temperature_scaling": "temperature_K=293.15+25*u",
        "heat3d_reference_temperature_K": 298.15,
        "heat3d_target_scaling": "deltaT_from_robin_ambient_K=25*(u-0.2)",
        "heat3d_dimensionalization": "1 released length unit = 1 m; q_K=25*q_u; k unchanged; h=k/Robin_length",
        "target_included": target_path is not None,
        "no_solver_run": True,
    }
    return arrays, metadata


def surface_case(kind: str, power_path: Path, target_path: Path | None, index: int) -> tuple[dict[str, np.ndarray], dict]:
    power = select_array(power_path, (21, 21), index)
    if kind == "deepoheat_2d_power_map":
        query_shape, z_max, kappa, bottom_robin_k = (21, 21, 11), 0.5, 0.2, 0.2
    else:
        query_shape, z_max, kappa, bottom_robin_k = (101, 101, 51), 0.5, 1.0, 0.2
    coords = cartesian_coords(*query_shape, z_max)
    k_field = np.full((len(coords), 3), kappa, dtype=np.float32)
    q = np.zeros((len(coords), 1), dtype=np.float32)
    bc = boundary_features(coords, top_h=0.0, bottom_h=kappa / bottom_robin_k)
    sensor_xy = np.stack(
        np.meshgrid(np.linspace(0, 1, 21), np.linspace(0, 1, 21), indexing="ij"), axis=-1
    ).reshape(-1, 2)
    arrays = {
        "coords": coords.astype(np.float32),
        "features_without_surface_flux": np.concatenate((k_field, q, bc), axis=1),
        "top_neumann_flux_sensor_xy": sensor_xy.astype(np.float32),
        "top_neumann_flux_sensors": power.reshape(-1, 1),
    }
    if target_path is not None:
        u = select_array(target_path, query_shape, index)
        arrays["temperature_K"] = (293.15 + 25.0 * u).reshape(-1, 1)
        arrays["deltaT_from_robin_ambient_K"] = (25.0 * (u - 0.2)).reshape(-1, 1)
    metadata = {
        "benchmark_family": "DeepOHeat-v1" if kind == "deepoheat_v1_surface" else "DeepOHeat",
        "benchmark": "surface" if kind == "deepoheat_v1_surface" else "2d_power_map",
        "representation": "deterministic_schema_only_not_current_Heat3D_model_input",
        "lossless_under_current_11_channel_contract": False,
        "reason": "released input is Neumann surface flux; relabeling it as volumetric q would change the PDE",
        "required_future_contract": "explicit nonlearned boundary-flux value/type in BC schema, followed by separately qualified model input projection",
        "query_shape": list(query_shape),
        "power_input_sha256": sha256(power_path),
        "source_sample_index": index,
        "target_included": target_path is not None,
        "no_solver_run": True,
    }
    return arrays, metadata


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="case", required=True)
    htc = sub.add_parser("deepoheat-htc")
    htc.add_argument("--kind", choices=("single_htc_bc", "multi_htc_bc"), default="single_htc_bc")
    htc.add_argument("--top-robin-k", type=float, required=True)
    htc.add_argument("--bottom-robin-k", type=float, default=0.2)
    v1 = sub.add_parser("deepoheat-v1-volume")
    v1.add_argument("--power", type=Path, required=True)
    v1.add_argument("--target", type=Path)
    v1.add_argument("--index", type=int, default=0)
    surface = sub.add_parser("surface-power")
    surface.add_argument("--kind", choices=("deepoheat_2d_power_map", "deepoheat_v1_surface"), required=True)
    surface.add_argument("--power", type=Path, required=True)
    surface.add_argument("--target", type=Path)
    surface.add_argument("--index", type=int, default=0)
    for command in (htc, v1, surface):
        command.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.case == "deepoheat-htc":
        arrays, metadata = htc_case(args.top_robin_k, args.bottom_robin_k, args.kind)
    elif args.case == "deepoheat-v1-volume":
        arrays, metadata = volume_v1_case(args.power, args.target, args.index)
    else:
        arrays, metadata = surface_case(args.kind, args.power, args.target, args.index)
    if args.output.suffix != ".npz":
        raise ValueError("--output must end in .npz")
    np.savez_compressed(args.output, **arrays)
    metadata["artifact_sha256"] = sha256(args.output)
    metadata["array_shapes"] = {name: list(value.shape) for name, value in arrays.items()}
    sidecar = args.output.with_suffix(".json")
    sidecar.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
