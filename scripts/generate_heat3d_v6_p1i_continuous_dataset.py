#!/usr/bin/env python3
"""Generate the preregistered V6-P1i continuous-physics pilot.

This is dataset-generation code only. It never imports model code, starts
training, performs learned inference, or filters cases by solved temperature.
"""

from __future__ import annotations

import argparse
from collections import Counter
import copy
import csv
import hashlib
import json
import math
from pathlib import Path
import shutil
import time
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.stats import qmc
import yaml

import heat3d_v6_p1i_continuous_core as core


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = (
    ROOT / "configs/heat3d_v6_p1i/v6_p1i_pilot128_v2.yaml"
)
RESULT_DIR = ROOT / "configs/heat3d_v6_p1i"
ARRAY_FILES = (
    "coords.npy",
    "temperature.npy",
    "deltaT.npy",
    "k_field.npy",
    "q_field.npy",
    "layer_id.npy",
    "bc_features.npy",
    "control_volume.npy",
)


class Cursor:
    def __init__(self, values: np.ndarray) -> None:
        self.values = np.asarray(values, dtype=np.float64)
        self.offset = 0

    def take(self) -> float:
        if self.offset >= self.values.size:
            raise core.ContinuousPhysicsError("Sobol dimension budget exhausted")
        value = float(self.values[self.offset])
        self.offset += 1
        return value


def _json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise core.ContinuousPhysicsError(f"refusing empty CSV: {path}")
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _lerp(bounds: Sequence[float], value: float) -> float:
    low, high = map(float, bounds)
    return low + (high - low) * float(value)


def _log_lerp(bounds: Sequence[float], value: float) -> float:
    low, high = map(float, bounds)
    if low <= 0.0 or high <= low:
        raise core.ContinuousPhysicsError(f"invalid log range: {bounds}")
    return math.exp(math.log(low) + float(value) * math.log(high / low))


def _count(bounds: Sequence[int], value: float) -> int:
    low, high = map(int, bounds)
    return min(low + int(math.floor(float(value) * (high - low + 1))), high)


def _overlap(
    left: Sequence[float], right: Sequence[float], gap: float
) -> bool:
    lx0, lx1, ly0, ly1 = map(float, left)
    rx0, rx1, ry0, ry1 = map(float, right)
    return (
        min(lx1, rx1) + gap > max(lx0, rx0)
        and min(ly1, ry1) + gap > max(ly0, ry0)
    )


def _rectangles(
    cursor: Cursor,
    *,
    count: int,
    width_range: Sequence[float],
    height_range: Sequence[float],
    margin: float,
    gap: float,
    layer_probability: float,
    size_bias: float | None = None,
    size_bias_weight: float = 0.0,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    by_layer: dict[str, list[list[float]]] = {
        "silicon_die_lower": [],
        "silicon_die_upper": [],
    }
    for index in range(count):
        placed = False
        for _ in range(40):
            layer = (
                "silicon_die_upper"
                if cursor.take() < layer_probability
                else "silicon_die_lower"
            )
            width_u = cursor.take()
            height_u = cursor.take()
            if size_bias is not None:
                width_u = (
                    (1.0 - size_bias_weight) * width_u
                    + size_bias_weight * float(size_bias)
                )
                height_u = (
                    (1.0 - size_bias_weight) * height_u
                    + size_bias_weight * float(size_bias)
                )
            width = _lerp(width_range, width_u)
            height = _lerp(height_range, height_u)
            cx = margin + (1.0 - 2.0 * margin) * cursor.take()
            cy = margin + (1.0 - 2.0 * margin) * cursor.take()
            x0 = max(margin, min(cx - width / 2.0, 1.0 - margin - width))
            y0 = max(margin, min(cy - height / 2.0, 1.0 - margin - height))
            bbox = [x0, x0 + width, y0, y0 + height]
            if not any(_overlap(bbox, old, gap) for old in by_layer[layer]):
                by_layer[layer].append(bbox)
                rows.append(
                    {
                        "block_id": f"block_{index:02d}",
                        "layer": layer,
                        "bbox_fraction_xy": bbox,
                    }
                )
                placed = True
                break
        if not placed:
            raise core.ContinuousPhysicsError(
                f"unable to place {count} non-overlapping rectangles"
            )
    if not {row["layer"] for row in rows} == {
        "silicon_die_lower",
        "silicon_die_upper",
    }:
        # The final block is moved to the missing active layer without changing
        # planform. Same-family overlap remains impossible on the empty layer.
        missing = (
            "silicon_die_upper"
            if all(row["layer"] == "silicon_die_lower" for row in rows)
            else "silicon_die_lower"
        )
        rows[-1]["layer"] = missing
    return rows


def _sample_layer_backgrounds(
    layers: Sequence[Mapping[str, Any]], cursor: Cursor
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    sampled: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    for layer in layers:
        result = copy.deepcopy(dict(layer))
        spec = dict(result["sampling"])
        u = cursor.take()
        mode = str(spec["distribution"])
        if mode == "isotropic_linear":
            value = _lerp(spec["range_W_mK"], u)
            k = [value, value, value]
        elif mode == "isotropic_log_uniform":
            value = _log_lerp(spec["range_W_mK"], u)
            k = [value, value, value]
        elif mode == "correlated_linear_axes":
            kxy = _lerp(spec["kxy_range_W_mK"], u)
            kz = _lerp(spec["kz_range_W_mK"], u)
            k = [kxy, kxy, kz]
        elif mode == "correlated_log_uniform_axes":
            kxy = _log_lerp(spec["kxy_range_W_mK"], u)
            kz = _log_lerp(spec["kz_range_W_mK"], u)
            k = [kxy, kxy, kz]
        else:
            raise core.ContinuousPhysicsError(
                f"{result['id']}: unsupported k distribution {mode}"
            )
        result["background_k_xyz_W_mK"] = k
        sampled.append(result)
        rows.append(
            {
                "layer_id": result["id"],
                "background_kx_W_mK": k[0],
                "background_ky_W_mK": k[1],
                "background_kz_W_mK": k[2],
                "sampling_latent": u,
                "distribution": mode,
            }
        )
    return sampled, rows


def _cross_overlap_count(
    left: Sequence[Mapping[str, Any]], right: Sequence[Mapping[str, Any]]
) -> int:
    total = 0
    for a in left:
        for b in right:
            if a["layer"] != b["layer"]:
                continue
            ax0, ax1, ay0, ay1 = map(float, a["bbox_fraction_xy"])
            bx0, bx1, by0, by1 = map(float, b["bbox_fraction_xy"])
            total += int(
                min(ax1, bx1) > max(ax0, bx0)
                and min(ay1, by1) > max(ay0, by0)
            )
    return total


def _split_map(sample_ids: Sequence[str], counts: Mapping[str, int]) -> dict[str, str]:
    ordered = sorted(
        sample_ids,
        key=lambda value: hashlib.sha256(
            f"v6-p1i-split:{value}".encode("utf-8")
        ).hexdigest(),
    )
    expected = sum(map(int, counts.values()))
    if len(ordered) != expected:
        raise core.ContinuousPhysicsError("split count mismatch")
    result: dict[str, str] = {}
    offset = 0
    for role in ("train", "valid_iid", "test_iid"):
        width = int(counts[role])
        for sample_id in ordered[offset : offset + width]:
            result[sample_id] = role
        offset += width
    return result


def _case_from_sobol(
    index: int,
    values: np.ndarray,
    config: Mapping[str, Any],
    split_role: str,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    cursor = Cursor(values)
    physics = copy.deepcopy(dict(config["physics"]))
    layers, background_rows = _sample_layer_backgrounds(
        physics["layers_bottom_to_top"], cursor
    )
    physics["layers_bottom_to_top"] = layers
    sampling = config["sampling"]
    top_h = _log_lerp(sampling["top_h_W_m2K"]["range"], cursor.take())
    bottom_h = _log_lerp(
        sampling["bottom_h_W_m2K"]["range"], cursor.take()
    )
    severity = cursor.take()
    power_spec = sampling["power_W"]
    if power_spec["distribution"] == "continuous_uniform":
        power = _lerp(power_spec["range"], severity)
    elif (
        power_spec["distribution"]
        == "continuous_severity_top_h_power_law"
    ):
        base_power = _lerp(power_spec["base_range_W"], severity)
        power = base_power * (
            top_h / float(power_spec["top_h_reference_W_m2K"])
        ) ** float(power_spec["top_h_exponent"])
        low, high = map(float, power_spec["allowed_range_W"])
        if not low <= power <= high:
            raise core.ContinuousPhysicsError(
                f"global power law produced {power} W outside {low, high}"
            )
    else:
        raise core.ContinuousPhysicsError(
            f"unsupported power distribution: {power_spec['distribution']}"
        )
    source_count = _count(
        sampling["source_count"]["range_inclusive"], cursor.take()
    )
    source_layer_probability = _lerp(
        sampling["source_layer_upper_probability"]["range"], cursor.take()
    )
    source_spec = sampling["source_planform_fraction"]
    q_blocks = _rectangles(
        cursor,
        count=source_count,
        width_range=source_spec["width_range"],
        height_range=source_spec["height_range"],
        margin=float(source_spec["center_margin"]),
        gap=float(source_spec["same_family_gap"]),
        layer_probability=source_layer_probability,
        size_bias=severity,
        size_bias_weight=float(
            source_spec.get("size_severity_weight", 0.0)
        ),
    )
    intensity_bounds = sampling["source_power_heterogeneity"]["range"]
    raw_power = []
    for block in q_blocks:
        x0, x1, y0, y1 = map(float, block["bbox_fraction_xy"])
        area = (x1 - x0) * (y1 - y0)
        raw_power.append(area * _log_lerp(intensity_bounds, cursor.take()))
    q_fractions = np.asarray(raw_power, dtype=np.float64)
    q_fractions /= np.sum(q_fractions)

    k_count = _count(
        sampling["k_region_count"]["range_inclusive"], cursor.take()
    )
    k_spec = sampling["k_region_planform_fraction"]
    k_blocks = _rectangles(
        cursor,
        count=k_count,
        width_range=k_spec["width_range"],
        height_range=k_spec["height_range"],
        margin=float(k_spec["center_margin"]),
        gap=float(k_spec["same_family_gap"]),
        layer_probability=cursor.take(),
    )
    k_values = [
        _log_lerp(sampling["local_k_W_mK"]["range"], cursor.take())
        for _ in k_blocks
    ]
    sample_id = f"v6p1i_{index:04d}"
    group = {
        "group_id": sample_id,
        "split_role": split_role,
        "q_blocks": q_blocks,
        "k_blocks": k_blocks,
        "cross_family_overlap_pair_count": _cross_overlap_count(
            k_blocks, q_blocks
        ),
    }
    case = {
        "sample_id": sample_id,
        "group_id": sample_id,
        "split_role": split_role,
        "package_total_power_W": power,
        "continuous_severity": severity,
        "top_h_W_m2K": top_h,
        "bottom_h_W_m2K": bottom_h,
        "k_block_values_W_mK": k_values,
        "q_block_power_fractions": q_fractions.tolist(),
        "q_bounds_W_m3": [1.0e8, 8.0e10],
        "surface_power_density_bounds_W_cm2": [1.0, 1000.0],
        "sobol_index": index,
        "sobol_dimensions_consumed": cursor.offset,
    }
    latent = {
        "sample_id": sample_id,
        "sobol_index": index,
        "sobol_dimensions_consumed": cursor.offset,
        "source_count": source_count,
        "k_region_count": k_count,
        "source_layer_upper_probability": source_layer_probability,
        "top_h_W_m2K": top_h,
        "bottom_h_W_m2K": bottom_h,
        "package_total_power_W": power,
        "continuous_severity": severity,
    }
    for row in background_rows:
        row["sample_id"] = sample_id
    return case, group, physics, background_rows


def _save_sample(
    root: Path,
    *,
    arrays: Mapping[str, np.ndarray],
    metadata: Mapping[str, Any],
) -> dict[str, str]:
    root.mkdir(parents=True, exist_ok=False)
    for name, value in arrays.items():
        np.save(root / name, np.asarray(value), allow_pickle=False)
    _json(root / "sample_meta.json", metadata)
    names = [*ARRAY_FILES, "sample_meta.json"]
    return {name: core.file_sha256(root / name) for name in names}


def _protocol_checks(config: Mapping[str, Any]) -> None:
    if int(config["sample_count"]) != 128 or config["stage"] != "pilot128":
        raise core.ContinuousPhysicsError("this entrypoint is pilot128-only")
    if config["sampling"]["method"] != "scrambled_sobol":
        raise core.ContinuousPhysicsError("Sobol is the only accepted design")
    if "temperature_bin" in json.dumps(config).lower():
        raise core.ContinuousPhysicsError("temperature-bin design is forbidden")
    forbidden = (
        "training",
        "model_inference",
        "post_solve_temperature_filtering",
        "post_solve_sample_replacement",
        "per_sample_power_backsolve",
        "missing_background_k_fallback",
        "frozen_v6_modified",
        "model_error_used_for_selection",
    )
    for key in forbidden:
        if config["guardrails"].get(key) is not False:
            raise core.ContinuousPhysicsError(f"forbidden guardrail {key}")
    for layer in config["physics"]["layers_bottom_to_top"]:
        if "background_k_xyz_W_mK" not in layer:
            raise core.ContinuousPhysicsError(
                f"{layer.get('id')}: missing background k"
            )


def generate(config_path: Path, output_root: Path, *, replace: bool) -> dict[str, Any]:
    config = core.load_config(config_path)
    _protocol_checks(config)
    artifact_prefix = str(config["artifact_prefix"])
    dataset_dir = output_root / str(config["dataset_id"])
    if dataset_dir.exists():
        if not replace:
            raise core.ContinuousPhysicsError(
                f"dataset exists; pass --replace for deterministic rebuild: {dataset_dir}"
            )
        shutil.rmtree(dataset_dir)
    dataset_dir.mkdir(parents=True)
    sample_ids = [
        f"v6p1i_{index:04d}" for index in range(int(config["sample_count"]))
    ]
    split_map = _split_map(sample_ids, config["split_counts"])
    dimensions = int(config["sampling"]["dimensions"])
    sampler = qmc.Sobol(
        d=dimensions,
        scramble=bool(config["sampling"]["scramble"]),
        seed=int(config["sampling"]["seed"]),
    )
    values = sampler.random_base2(
        int(math.log2(int(config["sample_count"])))
    )
    started = time.perf_counter()
    sample_rows: list[dict[str, Any]] = []
    region_rows: list[dict[str, Any]] = []
    background_rows: list[dict[str, Any]] = []
    sample_manifest: list[dict[str, Any]] = []
    for index, sobol in enumerate(values):
        sample_id = sample_ids[index]
        case, group, physics, sampled_backgrounds = _case_from_sobol(
            index, sobol, config, split_map[sample_id]
        )
        mesh = core.build_mesh(physics)
        layout = core.validate_layout(group, mesh)
        k_diag, q_field, rows = core.build_case_fields(
            case, group, mesh, layout
        )
        support = core.select_group_support(
            group,
            mesh,
            layout,
            int(config["sampling"]["seed"]) + 1901,
        )
        temperature, solver = core.solve_case(
            mesh,
            k_diag,
            q_field,
            top_h=float(case["top_h_W_m2K"]),
            bottom_h=float(case["bottom_h_W_m2K"]),
            ambient_K=float(physics["ambient_K"]),
        )
        metrics = core.case_metrics(
            mesh,
            temperature,
            q_field,
            solver,
            ambient_K=float(physics["ambient_K"]),
        )
        indices = np.asarray(support["indices"], dtype=np.int64)
        coords = np.asarray(mesh["coords"])[indices]
        bc = np.column_stack(
            (
                core.boundary_flags(coords, mesh),
                np.full((indices.size, 1), float(case["top_h_W_m2K"])),
                np.full((indices.size, 1), float(case["bottom_h_W_m2K"])),
                np.zeros((indices.size, 1)),
            )
        )
        arrays = {
            "coords.npy": coords,
            "temperature.npy": temperature[indices],
            "deltaT.npy": temperature[indices] - float(physics["ambient_K"]),
            "k_field.npy": k_diag[indices],
            "q_field.npy": q_field[indices, None],
            "layer_id.npy": np.asarray(mesh["layer_ids"])[indices],
            "bc_features.npy": bc,
            "control_volume.npy": np.asarray(mesh["weights"])[indices],
        }
        meta = {
            **case,
            **metrics,
            "dataset_id": config["dataset_id"],
            "solver_node_count": int(mesh["node_count"]),
            "projection_node_count": int(indices.size),
            "coordinate_sha256": support["coordinate_sha256"],
            "support_index_sha256": support["index_sha256"],
            "source_region_count": len(group["q_blocks"]),
            "k_region_count": len(group["k_blocks"]),
            "minimum_support_nodes_per_region": int(
                min(support["block_coverage"])
            ),
            "physics": physics,
            "q_blocks": group["q_blocks"],
            "k_blocks": group["k_blocks"],
        }
        hashes = _save_sample(
            dataset_dir / "samples" / sample_id,
            arrays=arrays,
            metadata=meta,
        )
        source_rows = [row for row in rows if row["family"] == "q"]
        k_rows = [row for row in rows if row["family"] == "k"]
        for row in rows:
            region_rows.append(
                {
                    "sample_id": sample_id,
                    "split_role": case["split_role"],
                    **row,
                }
            )
        background_rows.extend(sampled_backgrounds)
        sample_rows.append(
            {
                "sample_id": sample_id,
                "split_role": case["split_role"],
                "sobol_index": index,
                "package_total_power_W": metrics["package_total_power_W"],
                "continuous_severity": case["continuous_severity"],
                "top_h_W_m2K": case["top_h_W_m2K"],
                "bottom_h_W_m2K": case["bottom_h_W_m2K"],
                "source_count": len(source_rows),
                "k_region_count": len(k_rows),
                "total_source_volume_m3": sum(
                    float(row["source_volume_m3"]) for row in source_rows
                ),
                "mean_q_W_m3": float(
                    np.mean([row["q_W_m3"] for row in source_rows])
                ),
                "max_q_W_m3": float(
                    np.max([row["q_W_m3"] for row in source_rows])
                ),
                "mean_local_k_W_mK": float(
                    np.mean([row["k_x_W_mK"] for row in k_rows])
                ),
                **metrics,
                "minimum_support_nodes_per_region": int(
                    min(support["block_coverage"])
                ),
                "coordinate_sha256": support["coordinate_sha256"],
                "support_index_sha256": support["index_sha256"],
            }
        )
        sample_manifest.append(
            {
                "sample_id": sample_id,
                "split_role": case["split_role"],
                "relative_path": f"samples/{sample_id}",
                "file_sha256": hashes,
            }
        )
        print(
            f"[{index + 1:03d}/128] {sample_id} "
            f"peak={metrics['peak_deltaT_K']:.3f}K "
            f"residual={metrics['linear_residual']:.2e}"
        )
    _csv(RESULT_DIR / f"{artifact_prefix}_samples.csv", sample_rows)
    _csv(RESULT_DIR / f"{artifact_prefix}_regions.csv", region_rows)
    _csv(
        RESULT_DIR / f"{artifact_prefix}_background_k_samples.csv",
        background_rows,
    )
    manifest = {
        "schema_version": "heat3d_v6_p1i_manifest_v1",
        "dataset_id": config["dataset_id"],
        "stage": config["stage"],
        "status": "generated_pending_distribution_audit",
        "sample_count": len(sample_manifest),
        "split_role_counts": dict(
            sorted(Counter(row["split_role"] for row in sample_rows).items())
        ),
        "config_path": str(config_path.relative_to(ROOT)),
        "config_sha256": core.file_sha256(config_path),
        "config_payload_sha256": core.canonical_json_sha256(config),
        "dataset_root": str(dataset_dir.relative_to(ROOT)),
        "sobol": {
            "seed": int(config["sampling"]["seed"]),
            "dimensions": dimensions,
            "scramble": bool(config["sampling"]["scramble"]),
            "design_sha256": core.canonical_json_sha256(values.tolist()),
        },
        "solver_mesh_node_count": 240825,
        "elapsed_seconds": time.perf_counter() - started,
        "guardrails": {
            "training_runs": 0,
            "model_inference_runs": 0,
            "post_solve_filtering": False,
            "frozen_v6_modified": False,
        },
        "samples": sample_manifest,
    }
    manifest["manifest_payload_sha256"] = core.canonical_json_sha256(manifest)
    manifest_path = RESULT_DIR / f"{artifact_prefix}_manifest.json"
    _json(manifest_path, manifest)
    return manifest


def preflight(config_path: Path) -> dict[str, Any]:
    config = core.load_config(config_path)
    _protocol_checks(config)
    sample_ids = [
        f"v6p1i_{index:04d}" for index in range(int(config["sample_count"]))
    ]
    split_map = _split_map(sample_ids, config["split_counts"])
    sampler = qmc.Sobol(
        d=int(config["sampling"]["dimensions"]),
        scramble=bool(config["sampling"]["scramble"]),
        seed=int(config["sampling"]["seed"]),
    )
    values = sampler.random_base2(
        int(math.log2(int(config["sample_count"])))
    )
    powers: list[float] = []
    densities: list[float] = []
    source_nodes: list[int] = []
    support_nodes: list[int] = []
    for index, sobol in enumerate(values):
        case, group, physics, _ = _case_from_sobol(
            index, sobol, config, split_map[sample_ids[index]]
        )
        mesh = core.build_mesh(physics)
        layout = core.validate_layout(group, mesh)
        _, _, rows = core.build_case_fields(case, group, mesh, layout)
        support = core.select_group_support(
            group,
            mesh,
            layout,
            int(config["sampling"]["seed"]) + 1901,
        )
        q_rows = [row for row in rows if row["family"] == "q"]
        powers.append(float(case["package_total_power_W"]))
        densities.extend(float(row["q_W_m3"]) for row in q_rows)
        source_nodes.extend(
            int(row["control_volume_count"]) for row in q_rows
        )
        support_nodes.extend(map(int, support["block_coverage"]))
    return {
        "status": "passed",
        "config": str(config_path.relative_to(ROOT)),
        "sample_count": len(values),
        "package_power_W": [min(powers), max(powers)],
        "q_W_m3": [min(densities), max(densities)],
        "minimum_source_control_volume_count": min(source_nodes),
        "minimum_support_nodes_per_region": min(support_nodes),
        "solver_runs": 0,
        "training_runs": 0,
        "model_inference_runs": 0,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-root", type=Path, default=ROOT / "data")
    parser.add_argument("--replace", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    if args.preflight_only:
        print(json.dumps(preflight(args.config.resolve()), indent=2))
        return 0
    manifest = generate(
        args.config.resolve(), args.output_root.resolve(), replace=args.replace
    )
    print(json.dumps({k: manifest[k] for k in ("dataset_id", "sample_count", "dataset_root", "elapsed_seconds")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
