#!/usr/bin/env python3
"""Freeze a label-independent 4096-node solver support for V6 valid-only replay."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import fields, is_dataclass
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any

import h5py
import jax
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rigno.graphBuilder_Heat3D import Heat3DGraphBuilder  # noqa: E402


DEFAULT_ACCEPTANCE = (
    ROOT / "configs/heat3d_v6/v6_p1h_shared_support1024_acceptance.json"
)
DEFAULT_MANIFEST = (
    ROOT / "configs/heat3d_v6/v6_p1h_shared_support1024_manifest.json"
)
DEFAULT_OUTPUT = ROOT / "configs/heat3d_v6/v6_valid_common_probe4096.json"
DEFAULT_GRAPH_CONFIG = (
    ROOT / "configs/heat3d_v6/resolved/V6_02_V5best.resolved.yaml"
)
DATASET_ID = "heat3d_v6_p1h_shared_support1024_v0"
PROBE_ID = "v6_valid_common_solver_node4096_v0"
SEED = 2026072401
STRATUM_CODES = {
    "volume": 0,
    "source_allowed": 1,
    "interface": 2,
    "top": 3,
    "bottom": 4,
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _array_sha256(value: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(contiguous.dtype).encode("utf-8"))
    digest.update(str(tuple(contiguous.shape)).encode("utf-8"))
    digest.update(contiguous.tobytes())
    return digest.hexdigest()


def _tree_sha256(value: Any) -> str:
    digest = hashlib.sha256()

    def visit(name: str, item: Any) -> None:
        digest.update(name.encode("utf-8"))
        if is_dataclass(item):
            for field in fields(item):
                visit(f"{name}.{field.name}", getattr(item, field.name))
        elif isinstance(item, dict):
            for key in sorted(item):
                visit(f"{name}.{key}", item[key])
        elif isinstance(item, (tuple, list)):
            for index, child in enumerate(item):
                visit(f"{name}[{index}]", child)
        else:
            array = np.asarray(item)
            digest.update(str(array.dtype).encode("utf-8"))
            digest.update(str(tuple(array.shape)).encode("utf-8"))
            digest.update(np.ascontiguousarray(array).tobytes())

    visit("graph", value)
    return digest.hexdigest()


def _graph_sha256(coords: np.ndarray) -> str:
    config = yaml.safe_load(DEFAULT_GRAPH_CONFIG.read_text(encoding="utf-8"))
    builder = Heat3DGraphBuilder(**dict(config["graph"]))
    metadata = builder.build_metadata(coords, key=jax.random.PRNGKey(0))
    return _tree_sha256(metadata)


def _axis_indices(size: int, count: int) -> np.ndarray:
    result = np.rint(np.linspace(1, size - 2, count)).astype(np.int32)
    if len(np.unique(result)) != count:
        raise RuntimeError("probe axis selection contains duplicate indices")
    return result


def _center_z_index(layer_id_grid: np.ndarray, layer_index: int) -> int:
    candidates = np.flatnonzero(
        np.any(np.any(layer_id_grid == layer_index, axis=0), axis=0)
    )
    if candidates.size == 0:
        raise RuntimeError(f"layer {layer_index} has no solver nodes")
    return int(candidates[candidates.size // 2])


def _build_support(
    *,
    coords: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    boundaries: np.ndarray,
    layer_id: np.ndarray,
    control_volume: np.ndarray,
) -> dict[str, Any]:
    shape = (len(x), len(y), len(z))
    if shape != (65, 65, 57) or int(np.prod(shape)) != len(coords):
        raise RuntimeError(f"unexpected P1h solver mesh shape {shape}")
    expected_coords = np.stack(
        np.meshgrid(x, y, z, indexing="ij"), axis=-1
    ).reshape(-1, 3)
    if not np.array_equal(expected_coords, coords):
        raise RuntimeError("solver coordinate order differs from frozen ij-grid order")
    grid = np.arange(len(coords), dtype=np.int32).reshape(shape)
    layer_grid = layer_id.reshape(shape)
    selected: dict[int, str] = {}

    def add(index: int, stratum: str) -> None:
        previous = selected.get(index)
        if previous is not None:
            raise RuntimeError(
                f"probe support overlap at node {index}: {previous}/{stratum}"
            )
        selected[index] = stratum

    surface_axis = _axis_indices(shape[0], 16)
    for iz, stratum in ((0, "bottom"), (shape[2] - 1, "top")):
        for ix in surface_axis:
            for iy in surface_axis:
                add(int(grid[ix, iy, iz]), stratum)

    interface_axis = _axis_indices(shape[0], 8)
    for boundary in boundaries[1:-1]:
        iz = int(np.argmin(np.abs(z - float(boundary))))
        if not math.isclose(float(z[iz]), float(boundary), abs_tol=1.0e-15):
            raise RuntimeError("material interface is absent from solver nodes")
        for ix in interface_axis:
            for iy in interface_axis:
                add(int(grid[ix, iy, iz]), "interface")

    source_axis = _axis_indices(shape[0], 32)
    active_layer_indices = (4, 6)
    for layer_index in active_layer_indices:
        iz = _center_z_index(layer_grid, layer_index)
        for ix in source_axis:
            for iy in source_axis:
                add(int(grid[ix, iy, iz]), "source_allowed")

    remaining = 4096 - len(selected)
    if remaining != 1024:
        raise RuntimeError(f"unexpected residual volume quota {remaining}")
    layer_count = int(np.max(layer_id)) + 1
    quotas = np.full(layer_count, remaining // layer_count, dtype=np.int32)
    quotas[: remaining % layer_count] += 1
    rng = np.random.default_rng(SEED)
    all_selected = np.fromiter(selected, dtype=np.int64)
    side_or_surface = (
        np.isclose(coords[:, 0], x[0])
        | np.isclose(coords[:, 0], x[-1])
        | np.isclose(coords[:, 1], y[0])
        | np.isclose(coords[:, 1], y[-1])
        | np.isclose(coords[:, 2], z[0])
        | np.isclose(coords[:, 2], z[-1])
    )
    for layer_index, quota in enumerate(quotas):
        candidates = np.flatnonzero(
            (layer_id == layer_index)
            & ~side_or_surface
            & ~np.isin(np.arange(len(coords)), all_selected)
        )
        choice = np.sort(
            rng.choice(candidates, size=int(quota), replace=False)
        )
        for index in choice:
            add(int(index), "volume")

    if len(selected) != 4096:
        raise RuntimeError(f"probe contains {len(selected)} nodes, expected 4096")
    order = sorted(
        selected,
        key=lambda index: (
            STRATUM_CODES[selected[index]],
            int(layer_id[index]),
            float(coords[index, 0]),
            float(coords[index, 1]),
            float(coords[index, 2]),
        ),
    )
    indices = np.asarray(order, dtype=np.int32)
    strata = [selected[int(index)] for index in indices]
    probe_coords = coords[indices]
    probe_layers = layer_id[indices]
    probe_cv = control_volume[indices]
    if len(np.unique(indices)) != 4096:
        raise RuntimeError("probe indices are not unique")
    if not np.all(np.isfinite(probe_coords)) or np.any(probe_cv <= 0.0):
        raise RuntimeError("probe coordinates/control volumes are invalid")

    top = np.isclose(probe_coords[:, 2], z[-1])
    bottom = np.isclose(probe_coords[:, 2], z[0])
    interface_counts = []
    for interface_index, boundary in enumerate(boundaries[1:-1]):
        interface_counts.append(
            {
                "interface_index": interface_index,
                "z_m": float(boundary),
                "point_count": int(
                    np.sum(np.isclose(probe_coords[:, 2], boundary))
                ),
            }
        )
    return {
        "indices": indices,
        "coords": probe_coords,
        "layer_id": probe_layers,
        "control_volume": probe_cv,
        "strata": strata,
        "coverage": {
            "strata_counts": dict(sorted(Counter(strata).items())),
            "layer_point_counts": {
                str(index): int(np.sum(probe_layers == index))
                for index in range(layer_count)
            },
            "all_layers_covered": bool(
                all(np.any(probe_layers == index) for index in range(layer_count))
            ),
            "interfaces": interface_counts,
            "all_interfaces_covered": bool(
                all(row["point_count"] > 0 for row in interface_counts)
            ),
            "top_point_count": int(np.sum(top)),
            "bottom_point_count": int(np.sum(bottom)),
            "source_allowed_point_count": int(
                sum(stratum == "source_allowed" for stratum in strata)
            ),
        },
    }


def build(acceptance_path: Path, manifest_path: Path) -> dict[str, Any]:
    acceptance = json.loads(acceptance_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if acceptance["dataset_id"] != DATASET_ID or manifest["dataset_id"] != DATASET_ID:
        raise RuntimeError("P1h dataset binding drifted")
    dataset_root = Path(acceptance["data_paths"]["durable_copy"])
    archive = dataset_root / "full_fields.h5"
    if _sha256(archive) != acceptance["full_field_archive_sha256"]:
        raise RuntimeError("P1h full-field archive SHA256 mismatch")
    if _sha256(manifest_path) != (
        "324ca50a85698223d36c12a05d3e26b5cbc9aa00b559d067619baeb37f11e9d5"
    ):
        raise RuntimeError("P1h manifest SHA256 mismatch")
    with h5py.File(archive, "r") as handle:
        support = _build_support(
            coords=np.asarray(handle["mesh/coords"], dtype=np.float64),
            x=np.asarray(handle["mesh/x"], dtype=np.float64),
            y=np.asarray(handle["mesh/y"], dtype=np.float64),
            z=np.asarray(handle["mesh/z"], dtype=np.float64),
            boundaries=np.asarray(handle["mesh/boundaries"], dtype=np.float64),
            layer_id=np.asarray(handle["mesh/layer_id"], dtype=np.int32),
            control_volume=np.asarray(
                handle["mesh/control_volume"], dtype=np.float64
            ),
        )
    valid_ids = [
        str(row["sample_id"])
        for row in manifest["samples"]
        if row["split_role"] == "valid"
    ]
    if len(valid_ids) != 128:
        raise RuntimeError("valid_iid count drifted")
    indices = support.pop("indices")
    coords = support.pop("coords")
    layer_id = support.pop("layer_id")
    control_volume = support.pop("control_volume")
    strata = support.pop("strata")
    return {
        "schema_version": "heat3d_v6_common_valid_probe_v1",
        "status": "frozen",
        "probe_id": PROBE_ID,
        "dataset_id": DATASET_ID,
        "evaluation_role": "valid_iid",
        "sample_count": 128,
        "node_count": 4096,
        "solver_node_count": int(acceptance["solver_node_count"]),
        "seed": SEED,
        "selection_policy": "geometry_stratified_solver_nodes_v1",
        "selection_inputs": [
            "mesh.coords",
            "mesh.layer_id",
            "mesh.boundaries",
            "mesh.control_volume",
            "stack active-layer identities",
        ],
        "forbidden_selection_inputs": [
            "temperature",
            "q",
            "sample split labels",
            "model predictions",
            "model errors",
        ],
        "label_independent": True,
        "test_hard_accessed": False,
        "manifest_sha256": _sha256(manifest_path),
        "full_field_archive_sha256": _sha256(archive),
        "valid_sample_ids_sha256": hashlib.sha256(
            "\n".join(valid_ids).encode("utf-8")
        ).hexdigest(),
        "support_index_sha256": _array_sha256(indices),
        "coordinate_sha256": _array_sha256(coords),
        "graph_sha256": _graph_sha256(coords),
        "layer_id_sha256": _array_sha256(layer_id),
        "control_volume_sha256": _array_sha256(control_volume),
        "stratum_sha256": hashlib.sha256(
            "\n".join(strata).encode("utf-8")
        ).hexdigest(),
        "indices": indices.tolist(),
        "coverage": support["coverage"],
        "metric_weight_policy": (
            "selected solver-node control volumes normalized within each "
            "sample or requested region"
        ),
        "inference_executed": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--acceptance", type=Path, default=DEFAULT_ACCEPTANCE)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    payload = build(args.acceptance.resolve(), args.manifest.resolve())
    if args.write:
        args.output.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    print(
        json.dumps(
            {
                key: payload[key]
                for key in (
                    "status",
                    "probe_id",
                    "node_count",
                    "support_index_sha256",
                    "coordinate_sha256",
                    "coverage",
                    "test_hard_accessed",
                )
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
