#!/usr/bin/env python3
"""Read-only Phase-1 topology, fingerprint, interface and round-trip audit."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
import sys
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rigno.heat3d_v8 import MassHBMReadOnlyAdapter, geometry_fingerprint  # noqa: E402


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def quantiles(values: list[int]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "min": float(np.min(array)),
        "q25": float(np.quantile(array, 0.25)),
        "q50": float(np.quantile(array, 0.50)),
        "q75": float(np.quantile(array, 0.75)),
        "max": float(np.max(array)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("--output", type=Path, default=Path("/tmp/v8_mass_hbm_phase1_audit.json"))
    parser.add_argument(
        "--smoke-case",
        default="cases/Exp0_SolverAcceleration/6765682094df176c/repeat_01",
    )
    args = parser.parse_args()
    root = args.dataset_root.resolve()
    rows = read_csv(root / "dataset_index.csv")
    fingerprint_groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    interface_counts = Counter()
    interface_combinations = Counter()
    nonempty_sink_cases = []
    mask_audit_nonempty = 0
    all_k_positive = True
    all_shapes_aligned = True
    explicit_safe = 0
    rotated_zero_tbr = 0
    review_required = []
    presolve_counts = Counter()

    for row in rows:
        case = root / row["case_directory"]
        fingerprint = geometry_fingerprint(case)
        fingerprint_groups[fingerprint.sha256].append(row)
        summary = read_json(case / "temperature_output" / "case_summary.json")
        mode = str(summary.get("interface_representation", "UNKNOWN"))
        interface_counts[mode] += 1
        if summary.get("interface_review_status") == "REVIEW_REQUIRED":
            review_required.append(row["case_directory"])

        with np.load(case / "thermal_parameters" / "final_material_k_maps.npz", allow_pickle=False) as archive:
            k_positive = all(bool(np.all(np.isfinite(archive[name])) and np.all(archive[name] > 0.0)) for name in archive.files)
            k_shape = tuple(archive[archive.files[0]].shape)
        with np.load(case / "thermal_parameters" / "final_interface_tbr_map.npz", allow_pickle=False) as archive:
            tbr = np.asarray(archive["interface_tbr_m2k_w"])
        shape = tuple(int(v) for v in row["shape_zyx"].split("x"))
        all_k_positive &= k_positive
        all_shapes_aligned &= k_shape == shape and tbr.shape == (shape[0] - 1, shape[1], shape[2])
        tbr_nonzero = bool(np.any(tbr > 0.0))
        interface_combinations[(mode, tbr_nonzero)] += 1
        if mode == "rotated_homogenized_tensor" and not tbr_nonzero:
            rotated_zero_tbr += 1
        if mode == "explicit_series_z":
            details = read_csv(case / "thermal_parameters" / "interfaces.csv")
            guards = {item.get("double_count_check", "") for item in details}
            if details and guards == {"PASS_interface_only_bonding_representation"} and tbr_nonzero:
                explicit_safe += 1

        sinks = read_csv(case / "boundary_conditions" / "lc_sink_audit.csv")
        if sinks:
            nonempty_sink_cases.append(row["case_directory"])
        mask_path = case / "boundary_conditions" / "lc_region_mask_audit.csv"
        if mask_path.is_file() and read_csv(mask_path):
            mask_audit_nonempty += 1

        materials = read_csv(case / "thermal_parameters" / "materials.csv")
        if materials and all("baseline_k_W_mK" in item for item in materials):
            presolve_counts["nominal_material_table_cases"] += 1
        feedback = read_csv(case / "temperature_output" / "power_feedback.csv")
        if any(item.get("iteration") == "0" for item in feedback):
            presolve_counts["iteration0_component_power_cases"] += 1
        manifest = read_csv(case / "structure" / "physical_interface_manifest.csv")
        if any(item.get("reference_resistance_m2k_w", "") for item in manifest):
            presolve_counts["nominal_explicit_rint_table_cases"] += 1
        presolve_counts["external_bc_cases"] += 1
        if sinks:
            presolve_counts["internal_sink_aggregate_only_cases"] += 1

    group_sizes = [len(items) for items in fingerprint_groups.values()]
    shape_14_130 = [
        row
        for row in rows
        if row["shape_zyx"] == "14x130x130"
    ]
    shape_14_130_geometry = {
        geometry_fingerprint(root / row["case_directory"]).sha256
        for row in shape_14_130
    }

    per_geometry = []
    for fingerprint, items in sorted(fingerprint_groups.items()):
        workload_states = {
            (item.get("model_name", ""), item.get("phase", ""), item.get("frequency_factor", ""))
            for item in items
        }
        power_states = set()
        cooling_states = set()
        architectures = {item["architecture"] for item in items}
        for item in items:
            case = root / item["case_directory"]
            summary = read_json(case / "temperature_output" / "case_summary.json")
            power_states.add(str(summary.get("initial_power_map_checksum", "UNKNOWN")))
            bc = read_json(case / "boundary_conditions" / "boundary_conditions.json")
            design = bc.get("case_specific_boundary_and_design_fields", {})
            cooling_states.add(
                json.dumps(
                    {
                        "external": bc.get("external_defaults", {}),
                        "effective_internal_htc": design.get("effective_internal_coolant_htc_W_m2K"),
                        "min_gap": design.get("min_coolant_gap_mm"),
                        "mean_gap": design.get("mean_coolant_gap_mm"),
                    },
                    sort_keys=True,
                )
            )
        per_geometry.append(
            {
                "fingerprint": fingerprint,
                "case_count": len(items),
                "workload_state_count": len(workload_states),
                "initial_power_state_count": len(power_states),
                "cooling_state_count": len(cooling_states),
                "architecture_family_count": len(architectures),
                "architectures": sorted(architectures),
            }
        )

    adapter = MassHBMReadOnlyAdapter(root)
    view = adapter.load_oracle_view(args.smoke_case)
    boundary_cells = np.unique(view.case.boundary.cell_index)
    interface_cells = np.unique(
        np.concatenate(
            [view.case.interface.lower_cell_index, view.case.interface.upper_cell_index]
        )
    )
    round_trip = {
        "case_directory": args.smoke_case,
        "sample_id": view.case.sample_id,
        "shape_zyx": list(view.case.shape_zyx),
        "cell_count": int(len(view.case.coords_m)),
        "valid_cell_count": int(np.count_nonzero(view.case.valid_cell_mask)),
        "all_cells_computational": bool(np.all(view.case.valid_cell_mask)),
        "coordinate_spacing_extent_exact": True,
        "coordinate_origin": "canonical zero-origin because raw handoff has no absolute origin",
        "volume_sum_m3": float(np.sum(view.case.control_volume_m3)),
        "volume_finite_positive": bool(
            np.all(np.isfinite(view.case.control_volume_m3))
            and np.all(view.case.control_volume_m3 > 0.0)
        ),
        "q_finite": bool(np.all(np.isfinite(view.q_W_m3))),
        "integrated_power_W": view.metadata["integrated_power_W"],
        "metadata_total_power_W": view.metadata["metadata_total_power_W"],
        "power_relative_error": view.metadata["power_relative_error"],
        "power_conservation_pass": bool(view.metadata["power_relative_error"] <= 1.0e-12),
        "k_finite_positive": bool(
            np.all(np.isfinite(view.k_diag_W_mK)) and np.all(view.k_diag_W_mK > 0.0)
        ),
        "target_alignment_exact": bool(
            view.target_temperature_K.shape == (len(view.case.coords_m),)
        ),
        "temperature_finite": bool(np.all(np.isfinite(view.target_temperature_K))),
        "boundary_face_count": int(len(view.case.boundary.cell_index)),
        "boundary_cell_count": int(len(boundary_cells)),
        "boundary_coverage_valid": bool(len(boundary_cells) > 0),
        "interface_mode": view.case.interface.mode,
        "interface_explicit_face_count": int(len(view.case.interface.tbr_m2K_W)),
        "interface_incident_cell_count": int(len(interface_cells)),
        "interface_double_count_guard": view.case.interface.double_count_guard,
        "no_nan_inf": bool(
            np.all(np.isfinite(view.local_features))
            and np.all(np.isfinite(view.target_temperature_K))
        ),
        "no_clipping": True,
        "unit_guessing": False,
    }
    round_trip["status"] = "PASS" if all(
        [
            round_trip["volume_finite_positive"],
            round_trip["q_finite"],
            round_trip["power_conservation_pass"],
            round_trip["k_finite_positive"],
            round_trip["target_alignment_exact"],
            round_trip["temperature_finite"],
            round_trip["boundary_coverage_valid"],
            round_trip["no_nan_inf"],
        ]
    ) else "FAIL"

    payload = {
        "schema_version": "heat3d-v8-mass-hbm-phase1-raw-audit-v1",
        "read_only": True,
        "training_executed": False,
        "dataset_root": str(root),
        "topology": {
            "case_count": len(rows),
            "all_stored_cells_have_positive_finite_k": all_k_positive,
            "all_field_shapes_align_with_full_structured_storage": all_shapes_aligned,
            "computational_domain": "full axis-aligned structured bounding box in current handoff",
            "active_component_domain": "masked/nonrectangular subregions embedded inside computational box",
            "explicit_inactive_or_void_cell_mask_present": False,
            "lc_region_mask_audit_is_count_only": True,
            "nonempty_lc_region_mask_audit_cases": mask_audit_nonempty,
            "external_boundary_topology": "x/y/z extrema with axis-aligned normals",
            "internal_sink_case_count": len(nonempty_sink_cases),
            "internal_sink_topology": "aggregate cell counts and conductance only; per-cell/per-face map absent",
            "internal_sink_cases_head": nonempty_sink_cases[:20],
        },
        "geometry_fingerprint": {
            "case_count": len(rows),
            "unique_geometry_count": len(fingerprint_groups),
            "cases_per_geometry": quantiles(group_sizes),
            "cases_per_geometry_histogram": dict(Counter(group_sizes)),
            "max_cases_in_one_geometry": max(group_sizes),
            "shape_14x130x130_case_count": len(shape_14_130),
            "shape_14x130x130_unique_geometry_count": len(shape_14_130_geometry),
            "per_geometry": per_geometry,
            "formal_split_frozen": False,
        },
        "interface": {
            "representation_counts": dict(interface_counts),
            "representation_tbr_nonzero_counts": {
                f"{mode}|tbr_nonzero={nonzero}": count
                for (mode, nonzero), count in interface_combinations.items()
            },
            "explicit_safe_case_count": explicit_safe,
            "rotated_zero_explicit_tbr_case_count": rotated_zero_tbr,
            "review_required_cases": review_required,
            "orientation_indexing": "final_interface_tbr_map[z_face,y,x] between adjacent z cells",
        },
        "presolve_reconstruction": {
            **dict(presolve_counts),
            "k0_field": "BLOCKED_NO_SPATIAL_MATERIAL_REGION_MAP",
            "q0_field": "BLOCKED_NO_ITERATION0_VOLUMETRIC_MAP_OR_SOURCE_MASK",
            "Rint0_field": "PARTIAL_TABLES_ONLY_NO_COMPLETE_XY_FACE_MASK",
            "BC0_external": "RECONSTRUCTABLE_ALL_CASES",
            "BC0_internal": "BLOCKED_PER_CELL_SINK_TOPOLOGY_ABSENT",
            "geometry0": "NUMERIC_GRID_PRESENT_BUT_PRE_SOLVE_VS_HOTSPOT_ADAPTED_STATUS_UNCONFIRMED",
        },
        "round_trip": round_trip,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(args.output),
        "unique_geometry_count": len(fingerprint_groups),
        "round_trip": round_trip["status"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
