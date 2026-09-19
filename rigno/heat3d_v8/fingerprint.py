"""Label-array-independent MASS-HBM geometry fingerprint.

The payload includes ``solver_geometry.json``.  Until its initial-versus-final
mesh provenance is confirmed, this utility makes no strict target-independent
claim.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class GeometryFingerprint:
    sha256: str
    payload: dict[str, Any]
    included_sources: tuple[str, ...]
    excluded_sources: tuple[str, ...]
    independence_claim: str


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _canonical_rows(rows: list[dict[str, str]], fields: tuple[str, ...]) -> list[dict[str, str]]:
    result = [{field: row.get(field, "") for field in fields} for row in rows]
    return sorted(result, key=lambda row: json.dumps(row, sort_keys=True))


def geometry_fingerprint(case_dir: str | Path) -> GeometryFingerprint:
    """Hash topology/design inputs only, never final state or labels."""

    case = Path(case_dir).resolve()
    geometry = _read_json(case / "structure" / "solver_geometry.json")
    placement = _canonical_rows(
        _read_csv(case / "structure" / "component_spatial_placement.csv"),
        ("component", "injection_target", "base_fraction", "dram_fraction", "distribution_method"),
    )
    interface_rows = _read_csv(case / "structure" / "physical_interface_manifest.csv")
    interface_topology = _canonical_rows(
        interface_rows,
        (
            "stack_id",
            "interface_type",
            "lower_die_index",
            "upper_die_index",
            "z_interface_index",
            "interface_normal_axis",
            "representation",
            "interface_representation",
            "assigned",
            "assigned_cell_count",
            "footprint_area_m2",
            "explicit_global_z_tbr",
            "reason",
        ),
    )
    boundary = _read_json(case / "boundary_conditions" / "boundary_conditions.json")
    design = boundary.get("case_specific_boundary_and_design_fields", {})
    cooling_geometry_fields = {
        name: design[name]
        for name in (
            "geometry_policy",
            "group_count",
            "pitch_um",
            "slab_count",
            "slab_thickness_um",
            "subgrid_geometry",
            "total_slabs",
        )
        if name in design
    }
    mask_audit_path = case / "boundary_conditions" / "lc_region_mask_audit.csv"
    mask_topology = []
    if mask_audit_path.is_file():
        mask_topology = _canonical_rows(
            _read_csv(mask_audit_path),
            (
                "active_memory_cell_count",
                "coolant_cell_count",
                "filler_or_mold_cell_count",
                "gpu_cell_count",
                "hbm_base_logic_cell_count",
                "active_memory_coolant_overlap_cell_count",
                "active_memory_filler_overlap_cell_count",
                "coolant_filler_overlap_cell_count",
            ),
        )
    payload = {
        "solver_geometry": geometry,
        "component_placement_topology": placement,
        "interface_topology": interface_topology,
        "cooling_geometry_fields": cooling_geometry_fields,
        "cooling_region_count_topology": mask_topology,
    }
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return GeometryFingerprint(
        sha256=hashlib.sha256(blob.encode("utf-8")).hexdigest(),
        payload=payload,
        included_sources=(
            "structure/solver_geometry.json",
            "structure/component_spatial_placement.csv topology columns",
            "structure/physical_interface_manifest.csv topology columns",
            "boundary_conditions.json geometry-only design fields",
            "lc_region_mask_audit.csv count topology",
        ),
        excluded_sources=(
            "temperature",
            "final k",
            "final q",
            "final Rint/current resistance",
            "hotspot/refinement state",
            "workload/model/phase",
            "case identity",
            "architecture label",
        ),
        independence_claim=(
            "LABEL_ARRAY_INDEPENDENT; STRICT_TARGET_INDEPENDENCE_UNCONFIRMED_"
            "BECAUSE_SOLVER_GEOMETRY_MESH_PROVENANCE_IS_UNKNOWN"
        ),
    )
