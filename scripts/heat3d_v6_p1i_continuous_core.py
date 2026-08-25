#!/usr/bin/env python3
"""Deterministic V6-P1i continuous-physics mesh, field, support, and FVM utilities.

This module is dataset-generation code.  It does not import model code and
does not train or run learned-model inference.
"""

from __future__ import annotations

from collections import Counter
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import LinearOperator, cg
import yaml


SCHEMA_VERSION = "heat3d_v6_p1i_continuous_dataset_v1"
AMBIENT_K = 300.0
ACTIVE_LAYERS = ("silicon_die_lower", "silicon_die_upper")


class ContinuousPhysicsError(RuntimeError):
    """Raised when the frozen continuous-physics contract is violated."""


def canonical_json_sha256(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_config(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema_version") != SCHEMA_VERSION:
        raise ContinuousPhysicsError(f"{path}: unexpected continuous-physics schema")
    return payload


def _axis_widths(axis: np.ndarray) -> np.ndarray:
    axis = np.asarray(axis, dtype=np.float64)
    if axis.ndim != 1 or axis.size < 2 or np.any(np.diff(axis) <= 0.0):
        raise ContinuousPhysicsError("mesh axis must be finite, 1-D and increasing")
    result = np.empty_like(axis)
    result[0] = 0.5 * (axis[1] - axis[0])
    result[-1] = 0.5 * (axis[-1] - axis[-2])
    result[1:-1] = 0.5 * (axis[2:] - axis[:-2])
    return result


def _build_z_axis(
    layers: Sequence[Mapping[str, Any]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    parts: list[np.ndarray] = []
    boundaries = [0.0]
    cursor = 0.0
    for index, layer in enumerate(layers):
        thickness = float(layer["thickness_m"])
        intervals = int(layer["z_intervals"])
        if thickness <= 0.0 or intervals < 1:
            raise ContinuousPhysicsError("invalid layer thickness/z intervals")
        part = np.linspace(cursor, cursor + thickness, intervals + 1)
        parts.append(part if index == 0 else part[1:])
        cursor += thickness
        boundaries.append(cursor)
    z = np.concatenate(parts)
    boundary_array = np.asarray(boundaries, dtype=np.float64)
    layer_ids = np.searchsorted(boundary_array[1:], z, side="right")
    layer_ids = np.minimum(layer_ids, len(layers) - 1).astype(np.int32)
    return z, layer_ids, boundary_array


def build_mesh(physics: Mapping[str, Any]) -> dict[str, Any]:
    nx, ny, expected_nz = map(int, physics["solver_mesh_intervals_xyz"])
    lx, ly = map(float, physics["footprint_m"])
    x = np.linspace(0.0, lx, nx + 1)
    y = np.linspace(0.0, ly, ny + 1)
    layers = physics["layers_bottom_to_top"]
    z, z_layer_ids, boundaries = _build_z_axis(layers)
    if z.size != expected_nz + 1:
        raise ContinuousPhysicsError(
            f"z interval mismatch: {z.size - 1} != {expected_nz}"
        )
    meshgrid = np.meshgrid(x, y, z, indexing="ij")
    coords = np.column_stack([item.reshape(-1) for item in meshgrid])
    shape = (x.size, y.size, z.size)
    grid = np.arange(coords.shape[0], dtype=np.int64).reshape(shape)
    layer_ids = np.broadcast_to(z_layer_ids, shape).reshape(-1)
    conductivity: list[list[float]] = []
    for layer in layers:
        if "background_k_xyz_W_mK" not in layer:
            raise ContinuousPhysicsError(
                f"{layer.get('id', '<unnamed>')}: missing explicit "
                "background_k_xyz_W_mK; silent conductivity defaults are forbidden"
            )
        value = list(map(float, layer["background_k_xyz_W_mK"]))
        if len(value) != 3 or min(value) <= 0.0:
            raise ContinuousPhysicsError(
                f"invalid background conductivity for {layer.get('id')}: {value}"
            )
        conductivity.append(value)
    base_k = np.asarray(conductivity, dtype=np.float64)[layer_ids]
    dx, dy, dz = _axis_widths(x), _axis_widths(y), _axis_widths(z)
    weights = (
        dx[:, None, None] * dy[None, :, None] * dz[None, None, :]
    ).reshape(-1)
    layer_index = {
        str(layer["id"]): index for index, layer in enumerate(layers)
    }
    if not set(ACTIVE_LAYERS) <= set(layer_index):
        raise ContinuousPhysicsError("frozen active layers are missing")
    return {
        "x": x,
        "y": y,
        "z": z,
        "shape": shape,
        "grid": grid,
        "coords": coords,
        "weights": weights,
        "widths": (dx, dy, dz),
        "layer_ids": layer_ids,
        "base_k_diag": base_k,
        "boundaries": boundaries,
        "layer_index": layer_index,
        "node_count": int(coords.shape[0]),
    }


def block_mask(
    mesh: Mapping[str, Any], block: Mapping[str, Any]
) -> np.ndarray:
    layer = str(block["layer"])
    if layer not in ACTIVE_LAYERS:
        raise ContinuousPhysicsError(f"block outside active layers: {layer}")
    bbox = list(map(float, block["bbox_fraction_xy"]))
    if len(bbox) != 4:
        raise ContinuousPhysicsError("bbox_fraction_xy must contain x0,x1,y0,y1")
    x0, x1, y0, y1 = bbox
    if not (0.0 <= x0 < x1 <= 1.0 and 0.0 <= y0 < y1 <= 1.0):
        raise ContinuousPhysicsError(f"invalid bbox: {bbox}")
    coords = np.asarray(mesh["coords"], dtype=np.float64)
    layer_id = int(mesh["layer_index"][layer])
    return (
        (np.asarray(mesh["layer_ids"]) == layer_id)
        & (coords[:, 0] >= x0 * float(mesh["x"][-1]) - 1.0e-15)
        & (coords[:, 0] <= x1 * float(mesh["x"][-1]) + 1.0e-15)
        & (coords[:, 1] >= y0 * float(mesh["y"][-1]) - 1.0e-15)
        & (coords[:, 1] <= y1 * float(mesh["y"][-1]) + 1.0e-15)
    )


def _same_layer_overlap(
    blocks: Sequence[Mapping[str, Any]],
) -> bool:
    for index, left in enumerate(blocks):
        lx0, lx1, ly0, ly1 = map(float, left["bbox_fraction_xy"])
        for right in blocks[index + 1 :]:
            if str(left["layer"]) != str(right["layer"]):
                continue
            rx0, rx1, ry0, ry1 = map(float, right["bbox_fraction_xy"])
            if min(lx1, rx1) > max(lx0, rx0) and min(ly1, ry1) > max(ly0, ry0):
                return True
    return False


def validate_layout(
    group: Mapping[str, Any], mesh: Mapping[str, Any]
) -> dict[str, Any]:
    k_blocks = list(group["k_blocks"])
    q_blocks = list(group["q_blocks"])
    if not (2 <= len(k_blocks) <= 8 and 3 <= len(q_blocks) <= 10):
        raise ContinuousPhysicsError(f"{group['group_id']}: block-count contract")
    if _same_layer_overlap(k_blocks) or _same_layer_overlap(q_blocks):
        raise ContinuousPhysicsError(f"{group['group_id']}: illegal same-family overlap")
    masks: dict[str, list[np.ndarray]] = {"k": [], "q": []}
    rows: list[dict[str, Any]] = []
    dx = float(mesh["x"][1] - mesh["x"][0])
    dy = float(mesh["y"][1] - mesh["y"][0])
    for family, blocks in (("k", k_blocks), ("q", q_blocks)):
        for block_index, block in enumerate(blocks):
            bbox = list(map(float, block["bbox_fraction_xy"]))
            if family == "q" and not (
                bbox[0] >= 0.05
                and bbox[1] <= 0.95
                and bbox[2] >= 0.05
                and bbox[3] <= 0.95
            ):
                raise ContinuousPhysicsError(
                    f"{group['group_id']}: q block enters outer 5%"
                )
            mask = block_mask(mesh, block)
            masks[family].append(mask)
            count = int(mask.sum())
            coords = np.asarray(mesh["coords"])[mask]
            x_intervals = max(int(np.unique(coords[:, 0]).size) - 1, 0)
            y_intervals = max(int(np.unique(coords[:, 1]).size) - 1, 0)
            minimum = 3 if family == "q" else 4
            if min(x_intervals, y_intervals) < minimum:
                raise ContinuousPhysicsError(
                    f"{group['group_id']}: underresolved {family} block"
                )
            if family == "q" and count < 32:
                raise ContinuousPhysicsError(
                    f"{group['group_id']}: q block has {count} CVs"
                )
            rows.append(
                {
                    "family": family,
                    "block_index": block_index,
                    "layer": str(block["layer"]),
                    "control_volume_count": count,
                    "x_interval_count": x_intervals,
                    "y_interval_count": y_intervals,
                    "width_m": (bbox[1] - bbox[0])
                    * float(mesh["x"][-1]),
                    "height_m": (bbox[3] - bbox[2])
                    * float(mesh["y"][-1]),
                    "nominal_dx_m": dx,
                    "nominal_dy_m": dy,
                }
            )
    overlap_pairs = 0
    for k_mask in masks["k"]:
        for q_mask in masks["q"]:
            overlap_pairs += int(np.any(k_mask & q_mask))
    expected = int(group["cross_family_overlap_pair_count"])
    continuous_overlap_pairs = 0
    for k_block in k_blocks:
        for q_block in q_blocks:
            if str(k_block["layer"]) != str(q_block["layer"]):
                continue
            kx0, kx1, ky0, ky1 = map(
                float, k_block["bbox_fraction_xy"]
            )
            qx0, qx1, qy0, qy1 = map(
                float, q_block["bbox_fraction_xy"]
            )
            continuous_overlap_pairs += int(
                min(kx1, qx1) > max(kx0, qx0)
                and min(ky1, qy1) > max(ky0, qy0)
            )
    if continuous_overlap_pairs != expected:
        raise ContinuousPhysicsError(
            f"{group['group_id']}: continuous overlap topology "
            f"{continuous_overlap_pairs} != {expected}"
        )
    return {
        "masks": masks,
        "block_rows": rows,
        "cross_family_overlap_pair_count": continuous_overlap_pairs,
        "solver_node_overlap_pair_count": overlap_pairs,
    }


def build_case_fields(
    case: Mapping[str, Any],
    group: Mapping[str, Any],
    mesh: Mapping[str, Any],
    layout_audit: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    k_values = list(map(float, case["k_block_values_W_mK"]))
    q_fractions = list(map(float, case["q_block_power_fractions"]))
    if len(k_values) != len(group["k_blocks"]):
        raise ContinuousPhysicsError("k block/value length mismatch")
    if len(q_fractions) != len(group["q_blocks"]):
        raise ContinuousPhysicsError("q block/fraction length mismatch")
    if any(not math.isfinite(value) or value <= 0.0 for value in k_values):
        raise ContinuousPhysicsError("local k values must be finite and positive")
    if min(q_fractions) <= 0.0 or not math.isclose(
        sum(q_fractions), 1.0, rel_tol=0.0, abs_tol=1.0e-12
    ):
        raise ContinuousPhysicsError("q fractions must be positive and sum to one")
    k_diag = np.array(mesh["base_k_diag"], copy=True)
    q = np.zeros(int(mesh["node_count"]), dtype=np.float64)
    rows: list[dict[str, Any]] = []
    for block_index, (mask, value) in enumerate(
        zip(layout_audit["masks"]["k"], k_values)
    ):
        k_diag[mask, :] = value
        rows.append(
            {
                "family": "k",
                "block_index": block_index,
                "layer": group["k_blocks"][block_index]["layer"],
                "k_x_W_mK": value,
                "k_y_W_mK": value,
                "k_z_W_mK": value,
                "control_volume_count": int(mask.sum()),
            }
        )
    if not np.all(np.isfinite(k_diag)) or float(np.min(k_diag)) <= 0.0:
        raise ContinuousPhysicsError("k field is not finite positive definite")
    total_power = float(case["package_total_power_W"])
    weights = np.asarray(mesh["weights"], dtype=np.float64)
    for block_index, (mask, fraction) in enumerate(
        zip(layout_audit["masks"]["q"], q_fractions)
    ):
        volume = float(np.sum(weights[mask]))
        power = total_power * fraction
        density = power / volume
        q[mask] = density
        bbox = list(map(float, group["q_blocks"][block_index]["bbox_fraction_xy"]))
        area = (
            (bbox[1] - bbox[0])
            * float(mesh["x"][-1])
            * (bbox[3] - bbox[2])
            * float(mesh["y"][-1])
        )
        surface_density = power / area / 1.0e4
        density_bounds = case.get(
            "surface_power_density_bounds_W_cm2", [10.0, 1000.0]
        )
        if not (
            float(density_bounds[0]) - 1.0e-9
            <= surface_density
            <= float(density_bounds[1]) + 1.0e-9
        ):
            raise ContinuousPhysicsError(
                f"{case['sample_id']}: surface density {surface_density}"
            )
        q_bounds = case.get("q_bounds_W_m3", [1.0e8, 8.0e10])
        if not (
            float(q_bounds[0]) * (1.0 - 1.0e-12)
            <= density
            <= float(q_bounds[1]) * (1.0 + 1.0e-12)
        ):
            raise ContinuousPhysicsError(
                f"{case['sample_id']}: q {density} outside frozen envelope"
            )
        rows.append(
            {
                "family": "q",
                "block_index": block_index,
                "layer": group["q_blocks"][block_index]["layer"],
                "source_power_W": power,
                "power_fraction": fraction,
                "source_volume_m3": volume,
                "surface_power_density_W_cm2": surface_density,
                "q_W_m3": density,
                "control_volume_count": int(mask.sum()),
            }
        )
    realized = float(np.dot(q, weights))
    if not math.isclose(realized, total_power, rel_tol=2.0e-12, abs_tol=1.0e-12):
        raise ContinuousPhysicsError(
            f"{case['sample_id']}: power conservation {realized} != {total_power}"
        )
    return k_diag, q, rows


def _weighted_choice(
    rng: np.random.Generator,
    candidates: np.ndarray,
    count: int,
    weights: np.ndarray | None = None,
) -> np.ndarray:
    candidates = np.asarray(candidates, dtype=np.int64)
    if candidates.size < count:
        raise ContinuousPhysicsError(
            f"support candidate shortage: {candidates.size} < {count}"
        )
    probability = None
    if weights is not None:
        probability = np.asarray(weights, dtype=np.float64)
        probability = probability / np.sum(probability)
    return np.asarray(
        rng.choice(candidates, size=count, replace=False, p=probability),
        dtype=np.int64,
    )


def select_group_support(
    group: Mapping[str, Any],
    mesh: Mapping[str, Any],
    layout_audit: Mapping[str, Any],
    support_seed: int,
) -> dict[str, Any]:
    digest = hashlib.sha256(
        f"{support_seed}:{group['group_id']}:support_v1".encode("utf-8")
    ).hexdigest()
    rng = np.random.default_rng(int(digest[:16], 16))
    selected: list[int] = []
    strata: list[str] = []
    used: set[int] = set()

    def add(values: Sequence[int], label: str) -> None:
        for value in values:
            node = int(value)
            if node not in used:
                selected.append(node)
                strata.append(label)
                used.add(node)

    block_masks = [
        *layout_audit["masks"]["q"],
        *layout_audit["masks"]["k"],
    ]
    for mask in block_masks:
        candidates = np.flatnonzero(mask)
        add([int(candidates[rng.integers(candidates.size)])], "block")
    block_target = 256
    block_pool = np.flatnonzero(np.logical_or.reduce(block_masks))
    available = np.asarray([x for x in block_pool if int(x) not in used])
    add(
        _weighted_choice(
            rng,
            available,
            block_target - strata.count("block"),
            np.asarray(mesh["weights"])[available],
        ),
        "block",
    )

    boundaries = np.asarray(mesh["boundaries"], dtype=np.float64)[1:-1]
    coords = np.asarray(mesh["coords"], dtype=np.float64)
    for boundary in boundaries:
        candidates = np.flatnonzero(
            np.isclose(coords[:, 2], boundary, atol=1.0e-15)
        )
        available = np.asarray([x for x in candidates if int(x) not in used])
        add([int(available[rng.integers(available.size)])], "interface")
    interface_pool = np.flatnonzero(
        np.any(
            np.isclose(
                coords[:, 2, None], boundaries[None, :], atol=1.0e-15
            ),
            axis=1,
        )
    )
    available = np.asarray([x for x in interface_pool if int(x) not in used])
    add(
        _weighted_choice(
            rng,
            available,
            128 - strata.count("interface"),
            np.asarray(mesh["weights"])[available],
        ),
        "interface",
    )

    for label, z_value in (
        ("top", float(mesh["z"][-1])),
        ("bottom", float(mesh["z"][0])),
    ):
        candidates = np.flatnonzero(
            np.isclose(coords[:, 2], z_value, atol=1.0e-15)
        )
        available = np.asarray([x for x in candidates if int(x) not in used])
        add(
            _weighted_choice(
                rng,
                available,
                64,
                np.asarray(mesh["weights"])[available],
            ),
            label,
        )

    available = np.asarray(
        [x for x in range(int(mesh["node_count"])) if x not in used],
        dtype=np.int64,
    )
    add(
        _weighted_choice(
            rng,
            available,
            512,
            np.asarray(mesh["weights"])[available],
        ),
        "volume",
    )
    if len(selected) != 1024 or Counter(strata) != {
        "volume": 512,
        "block": 256,
        "interface": 128,
        "top": 64,
        "bottom": 64,
    }:
        raise ContinuousPhysicsError("support quota invariant failed")
    indices = np.asarray(selected, dtype=np.int64)
    support_coords = coords[indices]
    block_coverage = [
        int(np.sum(mask[indices])) for mask in block_masks
    ]
    if min(block_coverage) <= 0:
        raise ContinuousPhysicsError("support misses a preregistered block")
    layer_coverage = Counter(
        np.asarray(mesh["layer_ids"], dtype=np.int32)[indices].tolist()
    )
    if len(layer_coverage) != len(mesh["layer_index"]):
        raise ContinuousPhysicsError("support does not cover all layers")
    return {
        "indices": indices,
        "strata": np.asarray(strata),
        "coords": support_coords,
        "control_volume": np.asarray(mesh["weights"])[indices],
        "coordinate_sha256": canonical_json_sha256(support_coords.tolist()),
        "index_sha256": canonical_json_sha256(indices.tolist()),
        "block_coverage": block_coverage,
        "layer_coverage": dict(sorted(layer_coverage.items())),
    }


def boundary_flags(coords: np.ndarray, mesh: Mapping[str, Any]) -> np.ndarray:
    coords = np.asarray(coords, dtype=np.float64)
    top = np.isclose(coords[:, 2], float(mesh["z"][-1]), atol=1.0e-15)
    bottom = np.isclose(coords[:, 2], float(mesh["z"][0]), atol=1.0e-15)
    side = (
        np.isclose(coords[:, 0], float(mesh["x"][0]), atol=1.0e-15)
        | np.isclose(coords[:, 0], float(mesh["x"][-1]), atol=1.0e-15)
        | np.isclose(coords[:, 1], float(mesh["y"][0]), atol=1.0e-15)
        | np.isclose(coords[:, 1], float(mesh["y"][-1]), atol=1.0e-15)
    ) & ~top & ~bottom
    interior = ~(top | bottom | side)
    result = np.column_stack((top, bottom, side, interior)).astype(np.float64)
    if not np.all(np.sum(result, axis=1) == 1.0):
        raise ContinuousPhysicsError("boundary flags are not one-hot")
    return result


def _neighbor_faces(
    mesh: Mapping[str, Any], k_diag: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    grid = np.asarray(mesh["grid"], dtype=np.int64)
    dx, dy, dz = mesh["widths"]
    faces_i: list[np.ndarray] = []
    faces_j: list[np.ndarray] = []
    conductances: list[np.ndarray] = []
    for axis, (left, right, area, distance) in enumerate(
        (
            (
                grid[:-1, :, :],
                grid[1:, :, :],
                np.broadcast_to(
                    dy[None, :, None] * dz[None, None, :],
                    (grid.shape[0] - 1, grid.shape[1], grid.shape[2]),
                ),
                np.diff(np.asarray(mesh["x"]))[:, None, None],
            ),
            (
                grid[:, :-1, :],
                grid[:, 1:, :],
                np.broadcast_to(
                    dx[:, None, None] * dz[None, None, :],
                    (grid.shape[0], grid.shape[1] - 1, grid.shape[2]),
                ),
                np.diff(np.asarray(mesh["y"]))[None, :, None],
            ),
            (
                grid[:, :, :-1],
                grid[:, :, 1:],
                np.broadcast_to(
                    dx[:, None, None] * dy[None, :, None],
                    (grid.shape[0], grid.shape[1], grid.shape[2] - 1),
                ),
                np.diff(np.asarray(mesh["z"]))[None, None, :],
            ),
        )
    ):
        i = left.reshape(-1)
        j = right.reshape(-1)
        ki = k_diag[i, axis]
        kj = k_diag[j, axis]
        harmonic = 2.0 * ki * kj / (ki + kj)
        g = harmonic * np.asarray(area).reshape(-1) / np.broadcast_to(
            distance, left.shape
        ).reshape(-1)
        faces_i.append(i)
        faces_j.append(j)
        conductances.append(g)
    return (
        np.concatenate(faces_i),
        np.concatenate(faces_j),
        np.concatenate(conductances),
    )


def solve_case(
    mesh: Mapping[str, Any],
    k_diag: np.ndarray,
    q: np.ndarray,
    *,
    top_h: float,
    bottom_h: float,
    ambient_K: float = AMBIENT_K,
) -> tuple[np.ndarray, dict[str, float]]:
    if min(top_h, bottom_h) <= 0.0:
        raise ContinuousPhysicsError("dual-Robin h must be positive")
    i, j, g = _neighbor_faces(mesh, k_diag)
    n = int(mesh["node_count"])
    diagonal = np.bincount(
        np.concatenate((i, j)),
        weights=np.concatenate((g, g)),
        minlength=n,
    )
    rhs = np.asarray(q, dtype=np.float64) * np.asarray(
        mesh["weights"], dtype=np.float64
    )
    grid = np.asarray(mesh["grid"], dtype=np.int64)
    dx, dy, _ = mesh["widths"]
    boundary_area = (dx[:, None] * dy[None, :]).reshape(-1)
    top_nodes = grid[:, :, -1].reshape(-1)
    bottom_nodes = grid[:, :, 0].reshape(-1)
    top_robin = float(top_h) * boundary_area
    bottom_robin = float(bottom_h) * boundary_area
    diagonal[top_nodes] += top_robin
    diagonal[bottom_nodes] += bottom_robin
    rhs[top_nodes] += top_robin * float(ambient_K)
    rhs[bottom_nodes] += bottom_robin * float(ambient_K)
    rows = np.concatenate((i, j, np.arange(n, dtype=np.int64)))
    cols = np.concatenate((j, i, np.arange(n, dtype=np.int64)))
    values = np.concatenate((-g, -g, diagonal))
    matrix = csr_matrix((values, (rows, cols)), shape=(n, n))
    preconditioner = LinearOperator(
        (n, n),
        matvec=lambda value: np.asarray(value, dtype=np.float64) / diagonal,
        dtype=np.float64,
    )
    iterations = 0

    def callback(_: np.ndarray) -> None:
        nonlocal iterations
        iterations += 1

    temperature, info = cg(
        matrix,
        rhs,
        x0=np.full(n, float(ambient_K), dtype=np.float64),
        rtol=1.0e-10,
        atol=0.0,
        maxiter=20000,
        M=preconditioner,
        callback=callback,
    )
    if info != 0:
        raise ContinuousPhysicsError(f"CG failed: info={info}")
    temperature = np.asarray(temperature, dtype=np.float64)
    residual = float(
        np.linalg.norm(matrix.dot(temperature) - rhs)
        / max(np.linalg.norm(rhs), 1.0)
    )
    top_flux = float(
        np.dot(
            top_robin,
            temperature[top_nodes] - float(ambient_K),
        )
    )
    bottom_flux = float(
        np.dot(
            bottom_robin,
            temperature[bottom_nodes] - float(ambient_K),
        )
    )
    if not np.all(np.isfinite(temperature)):
        raise ContinuousPhysicsError("non-finite temperature")
    return temperature, {
        "linear_residual": residual,
        "cg_iterations": int(iterations),
        "top_heat_flux_W": top_flux,
        "bottom_heat_flux_W": bottom_flux,
    }


def case_metrics(
    mesh: Mapping[str, Any],
    temperature: np.ndarray,
    q: np.ndarray,
    solver: Mapping[str, float],
    *,
    ambient_K: float = AMBIENT_K,
) -> dict[str, Any]:
    weights = np.asarray(mesh["weights"], dtype=np.float64)
    delta = np.asarray(temperature, dtype=np.float64) - float(ambient_K)
    volume = float(np.sum(weights))
    power = float(np.dot(q, weights))
    top = float(solver["top_heat_flux_W"])
    bottom = float(solver["bottom_heat_flux_W"])
    peak = float(np.max(delta))
    return {
        "package_total_power_W": power,
        "peak_deltaT_K": peak,
        "mean_deltaT_K": float(np.dot(weights, delta) / volume),
        "cv_rms_deltaT_K": float(
            np.sqrt(np.dot(weights, delta * delta) / volume)
        ),
        "top_heat_fraction": top / power,
        "bottom_heat_fraction": bottom / power,
        "energy_balance_relative_error": (power - top - bottom) / power,
        "linear_residual": float(solver["linear_residual"]),
        "cg_iterations": int(solver["cg_iterations"]),
    }
