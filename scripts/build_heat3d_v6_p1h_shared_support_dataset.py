#!/usr/bin/env python3
"""Build P1h from deterministic P1g full-field replay and shared solver nodes."""

from __future__ import annotations

import argparse
from collections import Counter
import csv
from dataclasses import fields, is_dataclass
import hashlib
import json
import math
from pathlib import Path
import shutil
import sys
import tempfile
import time
from typing import Any, Mapping, Sequence
from unittest.mock import patch

import h5py
import jax
import numpy as np
from scipy.spatial import cKDTree
import yaml


ROOT = Path(__file__).resolve().parents[1]
for value in (ROOT, ROOT / "scripts"):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))

from check_heat3d_v4_registry import resolve_inherited_yaml  # noqa: E402
import generate_heat3d_v6_p1a_power_calibration_pilot as p1a  # noqa: E402
import generate_heat3d_v6_p1e_deconfounded_dataset as generator  # noqa: E402
import heat3d_v6_p1d_core as core  # noqa: E402
from rigno.graphBuilder_Heat3D import Heat3DGraphBuilder  # noqa: E402
from rigno.heat3d_v2_runner_command import build_training_command  # noqa: E402
import run_heat3d_v1_medium_controlled_training_export as runner  # noqa: E402


PARENT_CONFIG = ROOT / "configs/heat3d_v6/v6_p1g_geometry_deconfounded1024.yaml"
PARENT_MANIFEST = ROOT / "configs/heat3d_v6/v6_p1g_geometry_deconfounded1024_manifest.json"
P1H_CONFIG = ROOT / "configs/heat3d_v6/v6_p1h_shared_support1024.yaml"
REPLAY = ROOT / "configs/heat3d_v6/v6_p1h_replay_audit.json"
BASELINE_CONFIG = ROOT / "configs/heat3d_v6/V6_02_V5best.yaml"
DATASET_ID = "heat3d_v6_p1h_shared_support1024_v0"
STEM = "v6_p1h_shared_support1024"
STRATUM_CODES = {"volume": 0, "source": 1, "interface": 2, "top": 3, "bottom": 4}
SOURCE_PROPOSALS = {
    "source_grid_12x12_v1": 12,
    "source_grid_14x14_v1": 14,
    "source_dense_16x16_v1": 16,
}
SELECTED_PROPOSAL = "source_dense_16x16_v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _array_sha256(value: np.ndarray) -> str:
    return p1a._array_sha256(np.asarray(value))


def _json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise RuntimeError(f"refusing empty CSV: {path}")
    fields_out: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields_out:
                fields_out.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields_out, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _grid_axis_indices(size: int, count: int) -> np.ndarray:
    values = np.rint(np.linspace(1, size - 2, count)).astype(np.int32)
    if len(np.unique(values)) != count:
        raise RuntimeError("support grid axis contains duplicates")
    return values


def _layer_center_iz(mesh: Mapping[str, Any], layer_index: int) -> int:
    candidates = np.flatnonzero(np.any(np.any(
        np.asarray(mesh["layer_ids"]).reshape(mesh["info"]["shape"]) == layer_index,
        axis=0,
    ), axis=0))
    if candidates.size == 0:
        raise RuntimeError(f"layer {layer_index} has no z nodes")
    return int(candidates[candidates.size // 2])


def _support_proposal(
    mesh: Mapping[str, Any], physics: Mapping[str, Any], source_side: int, seed: int
) -> dict[str, Any]:
    grid = np.asarray(mesh["info"]["grid"])
    coords = np.asarray(mesh["coords"], dtype=np.float64)
    layer_ids = np.asarray(mesh["layer_ids"], dtype=np.int32)
    shape = mesh["info"]["shape"]
    selected: dict[int, str] = {}

    def add(index: int, stratum: str) -> None:
        if index in selected:
            raise RuntimeError(f"duplicate support node {index}: {selected[index]}/{stratum}")
        selected[index] = stratum

    surface_axis = _grid_axis_indices(shape[0], 8)
    for iz, name in ((0, "bottom"), (shape[2] - 1, "top")):
        for ix in surface_axis:
            for iy in surface_axis:
                add(int(grid[ix, iy, iz]), name)

    interface_axis = _grid_axis_indices(shape[0], 4)
    for boundary in np.asarray(mesh["boundaries"])[1:-1]:
        iz = int(np.argmin(np.abs(np.asarray(mesh["z"]) - float(boundary))))
        if not math.isclose(float(mesh["z"][iz]), float(boundary), abs_tol=1e-15):
            raise RuntimeError("interface is not represented by a solver z node")
        for ix in interface_axis:
            for iy in interface_axis:
                add(int(grid[ix, iy, iz]), "interface")

    source_axis = _grid_axis_indices(shape[0], source_side)
    active_names = ("silicon_die_lower", "silicon_die_upper")
    active_layers = [int(mesh["layer_index"][name]) for name in active_names]
    for layer_index in active_layers:
        iz = _layer_center_iz(mesh, layer_index)
        for ix in source_axis:
            for iy in source_axis:
                add(int(grid[ix, iy, iz]), "source")

    volume_count = 1024 - len(selected)
    layer_count = len(physics["layers_bottom_to_top"])
    quotas = np.full(layer_count, volume_count // layer_count, dtype=np.int32)
    quotas[: volume_count % layer_count] += 1
    rng = np.random.default_rng(int(seed) + source_side * 1009)
    for layer_index, quota in enumerate(quotas):
        candidates = np.flatnonzero(layer_ids == layer_index)
        candidates = candidates[~np.isin(candidates, np.fromiter(selected, dtype=np.int64))]
        choice = np.sort(rng.choice(candidates, size=int(quota), replace=False))
        for index in choice:
            add(int(index), "volume")
    if len(selected) != 1024:
        raise RuntimeError(f"support proposal has {len(selected)} nodes")

    order = sorted(
        selected,
        key=lambda index: (
            STRATUM_CODES[selected[index]],
            int(layer_ids[index]),
            float(coords[index, 0]), float(coords[index, 1]), float(coords[index, 2]),
        ),
    )
    indices = np.asarray(order, dtype=np.int32)
    strata = [selected[int(index)] for index in indices]
    support_coords = coords[indices]
    layer_counts = Counter(int(value) for value in layer_ids[indices])
    source_covering: dict[str, Any] = {}
    domain_xy = np.asarray([(x, y) for x in mesh["x"] for y in mesh["y"]], dtype=np.float64)
    scale = np.asarray(physics["footprint_m"], dtype=np.float64)
    for name, layer_index in zip(active_names, active_layers, strict=True):
        mask = (layer_ids[indices] == layer_index) & (np.asarray(strata) == "source")
        tree = cKDTree(support_coords[mask, :2] / scale)
        distances, _ = tree.query(domain_xy / scale, k=1)
        source_covering[name] = {
            "source_support_count": int(np.sum(mask)),
            "normalized_xy_covering_radius": float(np.max(distances)),
            "normalized_xy_mean_nearest_distance": float(np.mean(distances)),
        }
    return {
        "indices": indices,
        "coords": support_coords,
        "strata": strata,
        "proposal": {
            "source_side": source_side,
            "strata_counts": dict(sorted(Counter(strata).items())),
            "layer_point_counts": {
                str(physics["layers_bottom_to_top"][index]["id"]): int(layer_counts[index])
                for index in range(layer_count)
            },
            "all_layers_covered": all(layer_counts[index] > 0 for index in range(layer_count)),
            "source_allowed_domain_covering": source_covering,
            "coordinate_sha256": _array_sha256(support_coords),
            "support_index_sha256": _array_sha256(indices),
        },
    }


def _tree_hash(value: Any) -> str:
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


def _graph_metadata(coords: np.ndarray) -> tuple[str, dict[str, Any]]:
    payload = yaml.safe_load(BASELINE_CONFIG.read_text(encoding="utf-8"))
    resolved = resolve_inherited_yaml(payload, BASELINE_CONFIG)
    resolved["config_id"] = payload["config_id"]
    command = build_training_command(resolved)
    values = list(command[2:])
    wrapper_flags = {
        "--normalization-profile", "--condition-feature-transform",
        "--input-feature-schema", "--coord-policy", "--extent-feature-policy",
    }
    cleaned: list[str] = []
    index = 0
    while index < len(values):
        if values[index] in wrapper_flags:
            index += 2
        else:
            cleaned.append(values[index])
            index += 1
    with patch.object(sys, "argv", ["build_heat3d_v6_p1h_shared_support_dataset.py", *cleaned]):
        args = runner.parse_args()
    graph_config = runner._graph_config_from_args(args)
    builder = Heat3DGraphBuilder(**graph_config)
    metadata = builder.build_metadata(coords, key=runner._metadata_key(args.graph_seed))
    return _tree_hash(metadata), {**graph_config, "graph_seed": int(args.graph_seed)}


def _summary(values: Sequence[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "min": float(np.min(array)),
        "p05": float(np.quantile(array, 0.05)),
        "median": float(np.median(array)),
        "p95": float(np.quantile(array, 0.95)),
        "max": float(np.max(array)),
        "mean": float(np.mean(array)),
    }


def _source_support_counts(
    support_coords: np.ndarray, support_layers: np.ndarray, sources: Sequence[Mapping[str, Any]]
) -> list[int]:
    counts: list[int] = []
    for source in sources:
        bbox = source["bbox_m"]
        mask = (
            (support_layers == int(source["active_layer_index"]))
            & (support_coords[:, 0] >= float(bbox["x"][0]))
            & (support_coords[:, 0] <= float(bbox["x"][1]))
            & (support_coords[:, 1] >= float(bbox["y"][0]))
            & (support_coords[:, 1] <= float(bbox["y"][1]))
            & (support_coords[:, 2] >= float(bbox["z"][0]))
            & (support_coords[:, 2] <= float(bbox["z"][1]))
        )
        counts.append(int(np.sum(mask)))
    return counts


def _max_numeric_difference(left: Any, right: Any) -> float:
    if isinstance(left, dict) and isinstance(right, dict):
        if set(left) != set(right):
            return math.inf
        return max((_max_numeric_difference(left[key], right[key]) for key in left), default=0.0)
    if isinstance(left, list) and isinstance(right, list):
        if len(left) != len(right):
            return math.inf
        return max((_max_numeric_difference(a, b) for a, b in zip(left, right, strict=True)), default=0.0)
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return abs(float(left) - float(right))
    return 0.0 if left == right else math.inf


def build(parent_dataset: Path, dataset: Path) -> dict[str, Any]:
    started = time.perf_counter()
    replay = json.loads(REPLAY.read_text(encoding="utf-8"))
    if replay.get("status") != "passed":
        raise RuntimeError("full generation requires a passed replay audit")
    config = yaml.safe_load(PARENT_CONFIG.read_text(encoding="utf-8"))
    parent_manifest = json.loads(PARENT_MANIFEST.read_text(encoding="utf-8"))
    p1h_contract = yaml.safe_load(P1H_CONFIG.read_text(encoding="utf-8"))
    if dataset.exists():
        raise RuntimeError(f"refusing to overwrite {dataset}")
    parent_rows = {str(row["sample_id"]): row for row in parent_manifest["samples"]}
    groups = {str(row["group_id"]): row for row in config["geometry_groups"]}
    cases = config["cases"]
    first = cases[0]
    base_physics = generator._physics(
        float(first["top_h_W_m2K"]), float(first["bottom_h_W_m2K"]),
        config["physics"]["solver_mesh_intervals_xyz"],
    )
    mesh = core.build_mesh(base_physics)
    proposals: dict[str, dict[str, Any]] = {}
    proposal_payloads: dict[str, dict[str, Any]] = {}
    for proposal_id, source_side in SOURCE_PROPOSALS.items():
        value = _support_proposal(mesh, base_physics, source_side, int(config["seed"]))
        proposals[proposal_id] = value
        proposal_payloads[proposal_id] = value["proposal"]
    support = proposals[SELECTED_PROPOSAL]
    support_indices = support["indices"]
    support_coords = support["coords"]
    support_strata = support["strata"]
    support_layers = np.asarray(mesh["layer_ids"], dtype=np.int32)[support_indices]
    coordinate_sha = _array_sha256(support_coords)
    support_index_sha = _array_sha256(support_indices)
    graph_sha, graph_contract = _graph_metadata(support_coords)
    reconstruction_map = generator._idw8_map(support_coords, np.asarray(mesh["coords"], dtype=np.float64))
    coverage = core.point_coverage(
        support_coords, support_strata, support_layers, mesh, base_physics
    )
    if not coverage["all_layers_covered"] or not coverage["all_interfaces_covered"]:
        raise RuntimeError("selected shared support misses layer/interface coverage")

    dataset.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{dataset.name}.", dir=dataset.parent))
    archive_path = temporary / "full_fields.h5"
    sample_rows: list[dict[str, Any]] = []
    source_rows: list[dict[str, Any]] = []
    manifest_rows: list[dict[str, Any]] = []
    source_support_counts: list[int] = []
    reconstruction_rows: list[dict[str, Any]] = []
    physical_metric_max_diff = 0.0
    source_metadata_max_diff = 0.0
    solver_seconds = 0.0
    solver_cache: dict[tuple[float, float], tuple[dict[str, Any], core.DualRobinSolver]] = {}
    group_order = [str(row["group_id"]) for row in config["geometry_groups"]]
    group_index = {group_id: index for index, group_id in enumerate(group_order)}
    string_dtype = h5py.string_dtype(encoding="utf-8")
    try:
        with h5py.File(archive_path, "w") as archive:
            archive.attrs.update({
                "schema_version": "heat3d_v6_p1h_full_field_archive_v1",
                "dataset_id": DATASET_ID,
                "parent_dataset_id": config["dataset_id"],
                "parent_config_sha256": _sha256(PARENT_CONFIG),
                "parent_manifest_sha256": _sha256(PARENT_MANIFEST),
                "solver_node_count": int(mesh["coords"].shape[0]),
                "sample_count": len(cases),
                "support_coordinate_sha256": coordinate_sha,
                "support_graph_sha256": graph_sha,
            })
            mesh_group = archive.create_group("mesh")
            for name, value in {
                "coords": mesh["coords"], "k_diag": mesh["k_diag"],
                "control_volume": mesh["info"]["weights"], "layer_id": mesh["layer_ids"],
                "x": mesh["x"], "y": mesh["y"], "z": mesh["z"],
                "boundaries": mesh["boundaries"],
            }.items():
                mesh_group.create_dataset(name, data=np.asarray(value), compression="gzip", shuffle=True)
            support_group = archive.create_group("support")
            support_group.create_dataset("indices", data=support_indices)
            support_group.create_dataset("coords", data=support_coords)
            support_group.create_dataset(
                "stratum_code",
                data=np.asarray([STRATUM_CODES[name] for name in support_strata], dtype=np.int8),
            )
            groups_group = archive.create_group("groups")
            groups_group.create_dataset("group_id", data=np.asarray(group_order, dtype=object), dtype=string_dtype)
            q_unit_dataset = groups_group.create_dataset(
                "q_unit_power_W_m3", shape=(len(group_order), mesh["coords"].shape[0]),
                dtype=np.float64, chunks=(1, mesh["coords"].shape[0]), compression="gzip", shuffle=True,
            )
            for group_id in group_order:
                group = groups[group_id]
                q_unit, _, _ = generator._build_sources(
                    f"{group_id}_unit_power", 1.0, group, base_physics, mesh
                )
                q_unit_dataset[group_index[group_id], :] = q_unit

            samples_group = archive.create_group("samples")
            samples_group.create_dataset(
                "sample_id", data=np.asarray([str(case["id"]) for case in cases], dtype=object),
                dtype=string_dtype,
            )
            samples_group.create_dataset(
                "group_index", data=np.asarray([group_index[str(case["group_id"])] for case in cases], dtype=np.int16)
            )
            samples_group.create_dataset(
                "package_total_power_W", data=np.asarray([float(case["package_total_power_W"]) for case in cases])
            )
            temperature_dataset = samples_group.create_dataset(
                "temperature_K", shape=(len(cases), mesh["coords"].shape[0]), dtype=np.float64,
                chunks=(1, mesh["coords"].shape[0]), compression="gzip", compression_opts=4, shuffle=True,
            )
            q_dataset = samples_group.create_dataset(
                "q_W_m3", shape=(len(cases), mesh["coords"].shape[0]), dtype=np.float64,
                chunks=(1, mesh["coords"].shape[0]), compression="gzip", compression_opts=4, shuffle=True,
            )
            top_flux_dataset = samples_group.create_dataset("top_heat_flux_W", shape=(len(cases),), dtype=np.float64)
            bottom_flux_dataset = samples_group.create_dataset("bottom_heat_flux_W", shape=(len(cases),), dtype=np.float64)
            residual_dataset = samples_group.create_dataset("linear_residual", shape=(len(cases),), dtype=np.float64)

            for case_index, case in enumerate(cases):
                sample_id = str(case["id"])
                group_id = str(case["group_id"])
                group = groups[group_id]
                top_h = float(case["top_h_W_m2K"])
                bottom_h = float(case["bottom_h_W_m2K"])
                key = (top_h, bottom_h)
                if key not in solver_cache:
                    physics = generator._physics(
                        top_h, bottom_h, config["physics"]["solver_mesh_intervals_xyz"]
                    )
                    solver_cache[key] = (physics, core.DualRobinSolver(mesh, physics))
                physics, solver = solver_cache[key]
                power = float(case["package_total_power_W"])
                q, sources, layer_power = generator._build_sources(
                    sample_id, power, group, physics, mesh
                )
                solve_started = time.perf_counter()
                temperature, solve_audit = solver.solve(q)
                solver_seconds += time.perf_counter() - solve_started
                temperature_dataset[case_index, :] = temperature
                q_dataset[case_index, :] = q
                top_flux_dataset[case_index] = float(solve_audit["top_heat_flux_W"])
                bottom_flux_dataset[case_index] = float(solve_audit["bottom_heat_flux_W"])
                residual_dataset[case_index] = float(solve_audit["linear_residual"])
                metrics = core.field_metrics(
                    temperature=temperature, q=q, total_power_W=power,
                    mesh=mesh, solver_audit=solve_audit,
                )
                point_temperature = temperature[support_indices]
                point_q = q[support_indices]
                point_k = np.asarray(mesh["k_diag"], dtype=np.float64)[support_indices]
                point_bc = p1a._bc_features(support_coords, physics, mesh)
                reconstruction = generator._projection_reconstruction_metrics(
                    temperature, point_temperature, reconstruction_map[0], reconstruction_map[1], mesh, physics
                )
                metrics["solver_peak_minus_projected_peak_K"] = float(
                    metrics["peak_deltaT_K"] - np.max(point_temperature - 300.0)
                )
                metrics["projected_field_cv_rms_deltaT_K"] = float(
                    np.sqrt(np.mean((point_temperature - 300.0) ** 2))
                )
                counts = _source_support_counts(support_coords, support_layers, sources)
                source_support_counts.extend(counts)
                if min(counts) <= 0:
                    raise RuntimeError(f"{sample_id}: shared support misses a source")

                parent_dir = parent_dataset / str(parent_rows[sample_id]["sample_dir"])
                parent_meta_path = parent_dir / "sample_meta.json"
                parent_meta = json.loads(parent_meta_path.read_text(encoding="utf-8"))
                projection_metric_names = {
                    "solver_peak_minus_projected_peak_K",
                    "projected_field_cv_rms_deltaT_K",
                }
                invariant_metrics = {
                    name: value for name, value in metrics.items()
                    if name not in projection_metric_names
                }
                parent_invariant_metrics = {
                    name: value for name, value in parent_meta["metrics"].items()
                    if name not in projection_metric_names
                }
                physical_metric_max_diff = max(
                    physical_metric_max_diff,
                    _max_numeric_difference(invariant_metrics, parent_invariant_metrics),
                )
                source_metadata_max_diff = max(
                    source_metadata_max_diff,
                    _max_numeric_difference(sources, parent_meta["sources"]),
                )
                meta = dict(parent_meta)
                meta.update({
                    "schema_version": "heat3d_v6_p1h_shared_support_sample_v1",
                    "dataset_id": DATASET_ID,
                    "sources": sources,
                    "active_layer_power_W": layer_power,
                    "metrics": metrics,
                    "operator_projection": {
                        "point_count": 1024,
                        "support_proposal_id": SELECTED_PROPOSAL,
                        "support_index_sha256": support_index_sha,
                        "point_coordinates_sha256": coordinate_sha,
                        "graph_sha256": graph_sha,
                        "coordinates_shared_across_all_samples": True,
                        "coordinates_are_solver_nodes": True,
                        "point_coordinates_frozen_before_temperature_solve": True,
                        "label_inputs_used_for_point_selection": [],
                        "strata_counts": dict(sorted(Counter(support_strata).items())),
                        "coverage": coverage,
                        "source_point_counts": counts,
                        "full_field_reconstruction": reconstruction,
                    },
                    "full_field_archive": {
                        "file": "full_fields.h5",
                        "temperature_dataset": "/samples/temperature_K",
                        "temperature_row": case_index,
                        "temperature_raw_sha256": _array_sha256(temperature),
                        "q_dataset": "/samples/q_W_m3",
                        "q_row": case_index,
                        "q_reconstruction": "exact_per_sample_full_q_archive_row",
                        "q_raw_sha256": _array_sha256(q),
                    },
                    "p1h_provenance": {
                        "parent_dataset_id": config["dataset_id"],
                        "parent_sample_meta_sha256": _sha256(parent_meta_path),
                        "full_field_source": "deterministic_frozen_P1g_solver_replay",
                        "interpolation_from_parent_projected_points": False,
                    },
                })
                arrays = {
                    "coords.npy": support_coords.astype(np.float64),
                    "temperature.npy": point_temperature[:, None].astype(np.float64),
                    "deltaT.npy": (point_temperature - 300.0)[:, None].astype(np.float64),
                    "k_field.npy": point_k.astype(np.float64),
                    "q_field.npy": point_q[:, None].astype(np.float64),
                    "layer_id.npy": support_layers[:, None].astype(np.int32),
                    "bc_features.npy": point_bc.astype(np.float64),
                    "bc_parameters.npy": np.tile(
                        np.asarray([top_h, bottom_h, 300.0, 300.0]), (1024, 1)
                    ),
                    "sampling_stratum.npy": np.asarray(
                        [STRATUM_CODES[name] for name in support_strata], dtype=np.int8
                    )[:, None],
                }
                hashes = generator._write_sample(temporary / sample_id, arrays, meta)
                manifest_rows.append({
                    "sample_id": sample_id,
                    "group_id": group_id,
                    "split_role": str(case["split_role"]),
                    "sample_dir": sample_id,
                    "point_coordinates_sha256": coordinate_sha,
                    "graph_sha256": graph_sha,
                    "full_temperature_raw_sha256": _array_sha256(temperature),
                    "full_q_raw_sha256": _array_sha256(q),
                    "full_field_archive_row": case_index,
                    "file_sha256": hashes,
                })
                source_rows.extend({
                    **{name: value for name, value in row.items() if name not in {"bbox_m", "active_layer_index"}},
                    "group_id": group_id,
                    "split_role": str(case["split_role"]),
                    "shared_support_point_count": counts[source_index],
                } for source_index, row in enumerate(sources))
                reconstruction_rows.append({
                    "sample_id": sample_id,
                    "split_role": str(case["split_role"]),
                    "full_field_cv_rmse_K": reconstruction["full_field_cv_rmse_K"],
                    "full_field_cv_relative_rmse": reconstruction["full_field_cv_relative_rmse"],
                    "full_field_max_abs_error_K": reconstruction["full_field_max_abs_error_K"],
                    "max_abs_layer_average_error_K": reconstruction["max_abs_layer_average_error_K"],
                    "max_abs_layer_drop_error_K": reconstruction["max_abs_layer_drop_error_K"],
                    "solver_peak_minus_projected_peak_K": metrics["solver_peak_minus_projected_peak_K"],
                    "energy_balance_relative_error": metrics["energy_balance_relative_error"],
                })
                sample_rows.append({
                    "sample_id": sample_id, "group_id": group_id,
                    "split_role": str(case["split_role"]),
                    "top_h_W_m2K": top_h, "bottom_h_W_m2K": bottom_h,
                    "package_total_power_W": power,
                    "source_count": int(group["source_count"]),
                    "layout_kind": str(group["layout_kind"]),
                    "alignment_relation": str(group["alignment_relation"]),
                    "peak_deltaT_K": metrics["peak_deltaT_K"],
                    "mean_deltaT_K": metrics["mean_deltaT_K"],
                    "top_heat_fraction": metrics["top_heat_fraction"],
                    "bottom_heat_fraction": metrics["bottom_heat_fraction"],
                    **reconstruction_rows[-1],
                })
                if (case_index + 1) % 16 == 0:
                    print(
                        f"generated {case_index + 1}/{len(cases)} "
                        f"solver_seconds={solver_seconds:.1f}", flush=True,
                    )
        archive_sha = _sha256(archive_path)
        manifest = {
            "schema_version": "heat3d_v6_p1h_shared_support_manifest_v1",
            "dataset_id": DATASET_ID,
            "config": str(P1H_CONFIG.relative_to(ROOT)),
            "config_sha256": _sha256(P1H_CONFIG),
            "parent_dataset_id": config["dataset_id"],
            "parent_config_sha256": _sha256(PARENT_CONFIG),
            "parent_manifest_sha256": _sha256(PARENT_MANIFEST),
            "sample_count": len(cases),
            "group_count": len(groups),
            "shared_coordinate_sha256": coordinate_sha,
            "shared_support_index_sha256": support_index_sha,
            "shared_graph_sha256": graph_sha,
            "full_field_archive": {
                "path": "full_fields.h5",
                "sha256": archive_sha,
                "size_bytes": archive_path.stat().st_size,
            },
            "support_selection": {
                "selected_proposal_id": SELECTED_PROPOSAL,
                "selection_basis": "preregistered_source_allowed_domain_max_then_mean_nearest_distance",
                "temperature_or_test_label_inputs": [],
                "proposals": proposal_payloads,
                "graph_contract": graph_contract,
            },
            "samples": manifest_rows,
            "guardrails": {
                "parent_projected_point_interpolation": False,
                "parent_dataset_overwritten": False,
                "canonical_dataset_changed": False,
                "post_solve_sample_filtering": False,
                "training_runs": 0,
                "model_inference_runs": 0,
            },
        }
        _json(temporary / "manifest.json", manifest)
        temporary.rename(dataset)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise

    split_counts = Counter(str(row["split_role"]) for row in manifest_rows)
    source_summary = _summary(source_support_counts)
    reconstruction_summary = {
        key: _summary([float(row[key]) for row in reconstruction_rows])
        for key in (
            "full_field_cv_rmse_K", "full_field_cv_relative_rmse",
            "full_field_max_abs_error_K", "max_abs_layer_average_error_K",
            "max_abs_layer_drop_error_K", "solver_peak_minus_projected_peak_K",
            "energy_balance_relative_error",
        )
    }
    audit = {
        "schema_version": "heat3d_v6_p1h_shared_support_audit_v1",
        "status": "passed",
        "dataset_id": DATASET_ID,
        "dataset_path": str(dataset.resolve()),
        "dataset_manifest_sha256": _sha256(dataset / "manifest.json"),
        "sample_count": len(cases),
        "group_count": len(groups),
        "split_role_counts": dict(sorted(split_counts.items())),
        "solver_shape": list(mesh["info"]["shape"]),
        "solver_node_count": int(mesh["coords"].shape[0]),
        "shared_coordinate_sha256": coordinate_sha,
        "shared_support_index_sha256": support_index_sha,
        "shared_graph_sha256": graph_sha,
        "unique_coordinate_hash_count": len({row["point_coordinates_sha256"] for row in manifest_rows}),
        "unique_graph_hash_count": len({row["graph_sha256"] for row in manifest_rows}),
        "support_proposals": proposal_payloads,
        "selected_support_proposal": SELECTED_PROPOSAL,
        "support_selection_label_inputs": [],
        "coverage": coverage,
        "source_support_point_count": source_summary,
        "sources_with_zero_support": int(sum(value == 0 for value in source_support_counts)),
        "reconstruction": reconstruction_summary,
        "physical_metric_max_abs_difference_vs_P1g": physical_metric_max_diff,
        "source_metadata_max_abs_difference_vs_P1g": source_metadata_max_diff,
        "archive": manifest["full_field_archive"],
        "archive_public_mesh_coordinate_sha256": _array_sha256(np.asarray(mesh["coords"])),
        "archive_public_k_sha256": _array_sha256(np.asarray(mesh["k_diag"])),
        "solver_seconds": solver_seconds,
        "elapsed_seconds": time.perf_counter() - started,
        "leakage": {
            "support_uses_temperature": False,
            "support_uses_test_labels": False,
            "same_support_all_splits": True,
            "groups_crossing_split": 0,
        },
        "guardrails": manifest["guardrails"],
    }
    if physical_metric_max_diff > 1e-10 or source_metadata_max_diff > 1e-10:
        raise RuntimeError("P1h physical metadata drifted from P1g")
    if source_summary["min"] <= 0 or audit["unique_coordinate_hash_count"] != 1:
        raise RuntimeError("P1h support acceptance failed")

    config_dir = ROOT / "configs/heat3d_v6"
    _json(config_dir / f"{STEM}_manifest.json", manifest)
    _json(config_dir / f"{STEM}_audit.json", audit)
    _csv(config_dir / f"{STEM}_samples.csv", sample_rows)
    _csv(config_dir / f"{STEM}_sources.csv", source_rows)
    _csv(config_dir / f"{STEM}_projection_diagnostics.csv", reconstruction_rows)
    _csv(config_dir / f"{STEM}_split_map.csv", [{
        "group_id": row["group_id"],
        "split_role": row["split_role"],
        "source_count": row["source_count"],
        "layout_kind": row["layout_kind"],
        "case_count": sum(case["group_id"] == row["group_id"] for case in cases),
    } for row in config["geometry_groups"]])
    return audit


def _write_markdown(path: Path, payload: Mapping[str, Any]) -> None:
    source = payload["source_support_point_count"]
    recon = payload["reconstruction"]
    path.write_text(
        "# V6-P1h shared solver-node support closeout\n\n"
        f"Status: `{payload['status']}`. The dataset contains {payload['sample_count']} cases in "
        f"{payload['group_count']} frozen P1g groups. All samples share one ordered 1024-node "
        f"support and one graph (`{payload['shared_graph_sha256']}`).\n\n"
        f"Source support points: min={source['min']:.0f}, p05={source['p05']:.1f}, "
        f"median={source['median']:.1f}; zero-covered sources=0. Full-field CV-RMSE "
        f"median={recon['full_field_cv_rmse_K']['median']:.6f} K and p95="
        f"{recon['full_field_cv_rmse_K']['p95']:.6f} K.\n\n"
        "The support was selected from stack/layer/interface/Robin/source-allowed geometry only. "
        "No temperature or test label entered proposal selection. P1g and the canonical dataset "
        "designation remain unchanged.\n",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-dataset", type=Path, required=True)
    parser.add_argument(
        "--dataset", type=Path,
        default=ROOT / "data/heat3d_v6_p1h_shared_support1024_v0",
    )
    args = parser.parse_args()
    payload = build(args.parent_dataset.resolve(), args.dataset.resolve())
    _write_markdown(ROOT / "docs/v6_p1h_shared_support_closeout.md", payload)
    print(json.dumps({
        "status": payload["status"],
        "dataset_path": payload["dataset_path"],
        "archive": payload["archive"],
        "shared_coordinate_sha256": payload["shared_coordinate_sha256"],
        "shared_graph_sha256": payload["shared_graph_sha256"],
        "source_support_point_count": payload["source_support_point_count"],
        "elapsed_seconds": payload["elapsed_seconds"],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
