#!/usr/bin/env python3
"""Run and record every preregistered V6-P1d discrete calibration attempt."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml
import numpy as np
from scipy.optimize import linear_sum_assignment

import heat3d_v6_p1d_core as core


REPO_ROOT = Path(__file__).resolve().parent.parent
SCHEMA = "heat3d_v6_p1d_dual_robin_exploration_v1"
DEFAULT_SEARCH = REPO_ROOT / "configs/heat3d_v6/v6_p1d_asymmetric_dual_robin_search.yaml"
DEFAULT_JSON = REPO_ROOT / "configs/heat3d_v6/v6_p1d_asymmetric_dual_robin_exploration.json"
DEFAULT_CSV = REPO_ROOT / "configs/heat3d_v6/v6_p1d_asymmetric_dual_robin_exploration_attempts.csv"


def _dump_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _load(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != "heat3d_v6_p1d_dual_robin_search_v1":
        raise core.P1dError("unexpected P1d exploration schema")
    families = payload.get("bc_power_families", [])
    areas = payload["common"]["total_source_area_mm2"]
    if len(families) != 8 or len(areas) != 4:
        raise core.P1dError("P1d exploration must be 8 families x 2 powers x 4 areas")
    if len({family["family_id"] for family in families}) != 8:
        raise core.P1dError("duplicate exploration family")
    for family in families:
        if len(family["powers_W"]) != 2:
            raise core.P1dError("each exploration family requires two powers")
        core.build_physics(
            top_h=float(family["top_h_W_m2K"]),
            bottom_h=float(family["bottom_h_W_m2K"]),
        )
    return payload


def explore(args: argparse.Namespace) -> dict[str, Any]:
    search = _load(args.search)
    expected_attempts = len(search["bc_power_families"]) * 2 * len(search["common"]["total_source_area_mm2"])
    if args.dry_run:
        return {
            "schema_version": SCHEMA, "mode": "dry_run", "attempt_count": expected_attempts,
            "final_sample_generation": False, "training_runs": 0, "model_inference_runs": 0,
        }
    attempts: list[dict[str, Any]] = []
    slot_pools: list[dict[str, Any]] = []
    mesh_intervals = search["common"]["mesh_intervals_xyz"]
    for family_index, family in enumerate(search["bc_power_families"]):
        top_h = float(family["top_h_W_m2K"])
        bottom_h = float(family["bottom_h_W_m2K"])
        physics = core.build_physics(top_h=top_h, bottom_h=bottom_h, mesh_intervals=mesh_intervals)
        mesh = core.build_mesh(physics)
        solver = core.DualRobinSolver(mesh, physics)
        family_attempts: list[dict[str, Any]] = []
        for power_index, power in enumerate(map(float, family["powers_W"])):
            for area_index, area_mm2 in enumerate(map(float, search["common"]["total_source_area_mm2"])):
                attempt_id = f"x_f{family_index:02d}_p{power_index}_a{area_index}"
                q, sources, _ = core.build_sources(
                    sample_id=attempt_id, total_power_W=power,
                    total_area_m2=area_mm2 * 1e-6, layout_seed=0,
                    physics=physics, mesh=mesh,
                )
                temperature, solver_audit = solver.solve(q)
                metrics = core.field_metrics(
                    temperature=temperature, q=q, total_power_W=power,
                    mesh=mesh, solver_audit=solver_audit,
                )
                row = core.summarize_attempt(
                    attempt_id=attempt_id, family_id=str(family["family_id"]),
                    top_h=top_h, bottom_h=bottom_h, total_power_W=power,
                    total_area_m2=area_mm2 * 1e-6, metrics=metrics,
                )
                row["family_role"] = str(family["role"])
                row["attempt_retained"] = True
                family_attempts.append(row)
                attempts.append(row)
        powers = sorted(map(float, family["powers_W"]))
        targets = search["selection_contract"]["target_centers_K"]
        for bin_name, power in (("low", powers[0]), ("high", powers[1])):
            pool = [row for row in family_attempts if float(row["package_total_power_W"]) == power]
            target = float(targets[bin_name])
            slot_pools.append({
                "selection_bin": bin_name, "target_peak_deltaT_K": target,
                "family_id": str(family["family_id"]), "power_W": power,
                "pool": pool,
            })
        print(f"completed family {family_index + 1}/8: {family['family_id']}", flush=True)

    if len(attempts) != expected_attempts or len({row["attempt_id"] for row in attempts}) != expected_attempts:
        raise core.P1dError("exploration attempt count mismatch")
    quotas = search["selection_contract"]["total_source_area_quota_in_final16"]
    area_slots = [float(area) for area, count in sorted(quotas.items(), key=lambda item: float(item[0])) for _ in range(int(count))]
    if len(area_slots) != len(slot_pools):
        raise core.P1dError("area quota does not cover all final slots")
    costs = np.empty((len(slot_pools), len(area_slots)), dtype=np.float64)
    for row_index, slot in enumerate(slot_pools):
        by_area = {float(row["total_source_area_mm2"]): row for row in slot["pool"]}
        for column_index, area in enumerate(area_slots):
            candidate = by_area[area]
            costs[row_index, column_index] = abs(
                float(candidate["peak_deltaT_K"]) - float(slot["target_peak_deltaT_K"])
            )
    row_indices, column_indices = linear_sum_assignment(costs)
    selected: list[dict[str, Any]] = []
    for row_index, column_index in zip(row_indices, column_indices, strict=True):
        slot = slot_pools[int(row_index)]
        area = area_slots[int(column_index)]
        winner = next(row for row in slot["pool"] if float(row["total_source_area_mm2"]) == area)
        selected.append({
            "selection_bin": slot["selection_bin"],
            "target_peak_deltaT_K": slot["target_peak_deltaT_K"],
            "attempt_id": winner["attempt_id"], "family_id": winner["family_id"],
            "top_h_W_m2K": winner["top_h_W_m2K"],
            "bottom_h_W_m2K": winner["bottom_h_W_m2K"],
            "package_total_power_W": winner["package_total_power_W"],
            "total_source_area_mm2": winner["total_source_area_mm2"],
            "observed_exploration_peak_deltaT_K": winner["peak_deltaT_K"],
        })
    selected.sort(key=lambda row: (row["family_id"], row["selection_bin"]))
    if len(selected) != int(search["selection_contract"]["final_count"]):
        raise core.P1dError("selection count mismatch")
    payload = {
        "schema_version": SCHEMA,
        "search_config": str(args.search.relative_to(REPO_ROOT)),
        "search_config_sha256": core.sha256(args.search),
        "literature_matrix": search["literature_matrix"]["path"],
        "literature_matrix_sha256": core.sha256(REPO_ROOT / search["literature_matrix"]["path"]),
        "attempt_count": len(attempts),
        "all_attempts_retained": True,
        "attempt_deletion_count": 0,
        "selection_contract": search["selection_contract"],
        "selected_final_candidates": selected,
        "window_hit_count": sum(bool(row["in_30_80_K_window"]) for row in attempts),
        "guardrails": {
            "per_sample_Rth_power_back_calculation": False,
            "result_dependent_attempt_deletion": False,
            "final_sample_generation": False,
            "training_runs": 0,
            "model_inference_runs": 0,
        },
        "attempts": attempts,
    }
    _dump_json(args.output_json, payload)
    _write_csv(args.output_csv, attempts)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--search", type=Path, default=DEFAULT_SEARCH)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    for name in ("search", "output_json", "output_csv"):
        value = getattr(args, name)
        if not value.is_absolute():
            setattr(args, name, (REPO_ROOT / value).resolve())
    result = explore(args)
    print(json.dumps({key: result[key] for key in result if key not in {"attempts"}}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
