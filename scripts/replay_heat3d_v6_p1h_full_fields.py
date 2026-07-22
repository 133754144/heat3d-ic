#!/usr/bin/env python3
"""Replay representative frozen P1g cases before any P1h full generation."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
from scipy.interpolate import RegularGridInterpolator
import yaml


ROOT = Path(__file__).resolve().parents[1]
for value in (ROOT, ROOT / "scripts"):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))

import generate_heat3d_v6_p1a_power_calibration_pilot as p1a  # noqa: E402
import generate_heat3d_v6_p1e_deconfounded_dataset as generator  # noqa: E402
import heat3d_v6_p1d_core as core  # noqa: E402


CONFIG = ROOT / "configs/heat3d_v6/v6_p1g_geometry_deconfounded1024.yaml"
MANIFEST = ROOT / "configs/heat3d_v6/v6_p1g_geometry_deconfounded1024_manifest.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _tokens(case: dict[str, Any], group: dict[str, Any]) -> set[str]:
    return {
        f"split:{case['split_role']}",
        f"source_count:{group['source_count']}",
        f"layout:{group['layout_kind']}",
        f"alignment:{group['alignment_relation']}",
        f"top_h:{float(case['top_h_W_m2K']):g}",
        f"bottom_h:{float(case['bottom_h_W_m2K']):g}",
        f"power:{float(case['package_total_power_W']):g}",
    }


def _representatives(config: dict[str, Any]) -> tuple[list[dict[str, Any]], list[str]]:
    groups = {str(row["group_id"]): row for row in config["geometry_groups"]}
    candidates = sorted(config["cases"], key=lambda row: str(row["id"]))
    universe: set[str] = set()
    for case in candidates:
        universe |= _tokens(case, groups[str(case["group_id"])])
    selected: list[dict[str, Any]] = []
    remaining = set(universe)
    while remaining:
        best = max(
            candidates,
            key=lambda row: (
                len(_tokens(row, groups[str(row["group_id"])]) & remaining),
                -candidates.index(row),
            ),
        )
        gain = _tokens(best, groups[str(best["group_id"])]) & remaining
        if not gain:
            raise RuntimeError(f"representative set-cover stalled: {sorted(remaining)}")
        selected.append(best)
        remaining -= gain
        candidates.remove(best)
    return selected, sorted(universe)


def replay(parent_dataset: Path) -> dict[str, Any]:
    started = time.perf_counter()
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    if _sha256(CONFIG) != "ab162724af61c745f82571e9c8f07102d5262c70a4817ace0900e894bfc4af83":
        raise RuntimeError("frozen P1g config SHA drift")
    if _sha256(MANIFEST) != "e5329d5cd6253510d87a4432d5f2ddae67259637810c29fdfb6ddf42621875a4":
        raise RuntimeError("frozen P1g manifest SHA drift")

    rows = {str(row["sample_id"]): row for row in manifest["samples"]}
    hash_failures: list[str] = []
    for sample_id, row in rows.items():
        sample_dir = parent_dataset / str(row["sample_dir"])
        for name, expected in row["file_sha256"].items():
            path = sample_dir / name
            if not path.is_file() or _sha256(path) != expected:
                hash_failures.append(f"{sample_id}/{name}")
    if hash_failures:
        raise RuntimeError(f"P1g file SHA failures: {hash_failures[:10]}")

    groups = {str(row["group_id"]): row for row in config["geometry_groups"]}
    selected, factor_universe = _representatives(config)
    first = config["cases"][0]
    base_physics = generator._physics(
        float(first["top_h_W_m2K"]),
        float(first["bottom_h_W_m2K"]),
        config["physics"]["solver_mesh_intervals_xyz"],
    )
    mesh = core.build_mesh(base_physics)
    solvers: dict[tuple[float, float], tuple[dict[str, Any], core.DualRobinSolver]] = {}
    details: list[dict[str, Any]] = []
    maxima = {
        "coordinates_m": 0.0,
        "k_W_mK": 0.0,
        "q_W_m3": 0.0,
        "projected_temperature_K": 0.0,
        "field_metric": 0.0,
    }
    metric_keys = (
        "peak_deltaT_K", "mean_deltaT_K", "top_heat_flux_W",
        "bottom_heat_flux_W", "energy_balance_relative_error", "linear_residual",
    )
    for case in selected:
        sample_id = str(case["id"])
        group = groups[str(case["group_id"])]
        top_h = float(case["top_h_W_m2K"])
        bottom_h = float(case["bottom_h_W_m2K"])
        key = (top_h, bottom_h)
        if key not in solvers:
            physics = generator._physics(
                top_h, bottom_h, config["physics"]["solver_mesh_intervals_xyz"]
            )
            solvers[key] = (physics, core.DualRobinSolver(mesh, physics))
        physics, solver = solvers[key]
        power = float(case["package_total_power_W"])
        q, sources, _ = generator._build_sources(sample_id, power, group, physics, mesh)
        seed_key = str(group.get("projection_seed_key", group["group_id"]))
        points, _, point_seed = p1a._sample_points_before_labels(
            base_seed=int(config["seed"]), sample_id=seed_key,
            physics=physics, mesh=mesh, sources=sources,
        )
        sample_dir = parent_dataset / str(rows[sample_id]["sample_dir"])
        saved_points = np.load(sample_dir / "coords.npy")
        coord_error = float(np.max(np.abs(points - saved_points)))
        _, replay_k, replay_q = core.point_inputs(saved_points, physics, mesh, sources)
        k_error = float(np.max(np.abs(replay_k - np.load(sample_dir / "k_field.npy"))))
        q_error = float(np.max(np.abs(replay_q - np.load(sample_dir / "q_field.npy").reshape(-1))))
        solve_started = time.perf_counter()
        temperature, solve_audit = solver.solve(q)
        solve_seconds = time.perf_counter() - solve_started
        interpolator = RegularGridInterpolator(
            (mesh["x"], mesh["y"], mesh["z"]),
            temperature.reshape(mesh["info"]["shape"]),
            method="linear", bounds_error=True,
        )
        replay_point_temperature = np.asarray(interpolator(saved_points), dtype=np.float64)
        saved_temperature = np.load(sample_dir / "temperature.npy").reshape(-1)
        temperature_error = float(np.max(np.abs(replay_point_temperature - saved_temperature)))
        metrics = core.field_metrics(
            temperature=temperature, q=q, total_power_W=power,
            mesh=mesh, solver_audit=solve_audit,
        )
        saved_meta = json.loads((sample_dir / "sample_meta.json").read_text(encoding="utf-8"))
        metric_errors = {
            name: abs(float(metrics[name]) - float(saved_meta["metrics"][name]))
            for name in metric_keys
        }
        metric_error = max(metric_errors.values())
        values = (coord_error, k_error, q_error, temperature_error, metric_error)
        for name, value in zip(maxima, values, strict=True):
            maxima[name] = max(maxima[name], value)
        details.append({
            "sample_id": sample_id,
            "group_id": str(case["group_id"]),
            "split_role": str(case["split_role"]),
            "source_count": int(group["source_count"]),
            "layout_kind": str(group["layout_kind"]),
            "alignment_relation": str(group["alignment_relation"]),
            "top_h_W_m2K": top_h,
            "bottom_h_W_m2K": bottom_h,
            "package_total_power_W": power,
            "point_seed": int(point_seed),
            "coordinate_max_abs_error_m": coord_error,
            "k_max_abs_error_W_mK": k_error,
            "q_max_abs_error_W_m3": q_error,
            "temperature_max_abs_error_K": temperature_error,
            "metric_abs_errors": metric_errors,
            "solve_seconds": solve_seconds,
        })

    tolerances = {
        "coordinates_m": 0.0,
        "k_W_mK": 0.0,
        "q_W_m3": 0.0,
        "projected_temperature_K": 1.0e-10,
        "field_metric": 1.0e-10,
    }
    checks = {name: maxima[name] <= tolerance for name, tolerance in tolerances.items()}
    return {
        "schema_version": "heat3d_v6_p1h_replay_audit_v1",
        "status": "passed" if all(checks.values()) else "failed",
        "parent_dataset": str(parent_dataset.resolve()),
        "parent_config_sha256": _sha256(CONFIG),
        "parent_manifest_sha256": _sha256(MANIFEST),
        "parent_file_hashes_checked": sum(len(row["file_sha256"]) for row in rows.values()),
        "parent_file_hash_failures": hash_failures,
        "solver_node_count": int(mesh["coords"].shape[0]),
        "solver_shape": list(mesh["info"]["shape"]),
        "representative_selection": "deterministic_factor_set_cover_without_labels",
        "representative_count": len(selected),
        "factor_universe": factor_universe,
        "max_abs_error": maxima,
        "tolerances": tolerances,
        "checks": checks,
        "cases": details,
        "elapsed_seconds": time.perf_counter() - started,
        "training_runs": 0,
        "model_inference_runs": 0,
    }


def _write_markdown(path: Path, payload: dict[str, Any]) -> None:
    rows = "\n".join(
        f"| {row['sample_id']} | {row['split_role']} | {row['source_count']} | "
        f"{row['temperature_max_abs_error_K']:.3e} | {max(row['metric_abs_errors'].values()):.3e} |"
        for row in payload["cases"]
    )
    path.write_text(
        "# V6-P1h deterministic replay audit\n\n"
        f"Status: `{payload['status']}`. Replayed {payload['representative_count']} label-independent "
        f"factor-cover cases on the frozen {payload['solver_shape']} solver mesh.\n\n"
        "| sample | split | sources | max projected T error K | max metric error |\n"
        "|---|---|---:|---:|---:|\n" + rows + "\n\n"
        f"All {payload['parent_file_hashes_checked']} parent manifest file hashes were checked. "
        "No temperature value was used to select replay cases.\n",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-dataset", type=Path, required=True)
    parser.add_argument("--write-json", type=Path, required=True)
    parser.add_argument("--write-md", type=Path, required=True)
    args = parser.parse_args()
    payload = replay(args.parent_dataset.resolve())
    args.write_json.parent.mkdir(parents=True, exist_ok=True)
    args.write_md.parent.mkdir(parents=True, exist_ok=True)
    args.write_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_markdown(args.write_md, payload)
    print(json.dumps({
        "status": payload["status"],
        "representative_count": payload["representative_count"],
        "max_abs_error": payload["max_abs_error"],
        "elapsed_seconds": payload["elapsed_seconds"],
    }, indent=2, sort_keys=True))
    return 0 if payload["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
