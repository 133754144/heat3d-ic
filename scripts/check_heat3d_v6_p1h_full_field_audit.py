#!/usr/bin/env python3
"""Validate the P1h stop decision without reading labels or generating data."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / "configs/heat3d_v6/v6_p1h_shared_support1024_full_field_audit.json"
PARENT = ROOT / "configs/heat3d_v6/v6_p1g_geometry_deconfounded1024_manifest.json"


def main() -> int:
    payload = json.loads(AUDIT.read_text(encoding="utf-8"))
    assert payload["status"] == "blocked_missing_original_solver_full_fields"
    assert payload["decision"] == "stop_without_generating_P1h"
    assert payload["parent_manifest_sha256"] == hashlib.sha256(PARENT.read_bytes()).hexdigest()
    assert payload["manifest_sample_count"] == 1024
    assert payload["missing_sample_dir_count"] == 0
    assert payload["projected_1024_complete_sample_count"] == 1024
    assert payload["projected_node_counts"] == [1024]
    assert payload["declared_solver_node_counts"] == [240825]
    assert payload["samples_with_any_solver_sized_array_count"] == 0
    assert payload["samples_with_complete_solver_coords_k_q_T_count"] == 0
    supplemental = payload["supplemental_local_search"]
    assert len(supplemental["roots"]) == 2
    assert all(row["exists"] for row in supplemental["roots"])
    assert supplemental["array_file_count_scanned"] > 0
    assert supplemental["unreadable_array_file_count"] == 0
    assert supplemental["solver_sized_array_count"] == 0
    assert supplemental["archive_candidate_count"] == 0
    assert supplemental["solver_or_full_field_name_candidate_count"] == 0
    contract = payload["contract"]
    assert contract["requires_original_solver_coordinates_k_q_T"] is True
    assert contract["allows_interpolation_from_existing_1024_points"] is False
    assert contract["p1h_generation_started"] is False
    assert contract["p1h_manifest_created"] is False
    assert contract["p1h_output_exists"] is False
    assert contract["p1g_overwritten"] is False
    assert contract["canonical_dataset_changed"] is False
    print(json.dumps({"status": "passed", "decision": payload["decision"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
