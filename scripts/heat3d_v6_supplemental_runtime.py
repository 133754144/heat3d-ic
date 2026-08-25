#!/usr/bin/env python3
"""Input-plan loader shared by frozen E/U supplemental orchestration."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from heat3d_v6_supplemental_input_adapter import (
    load_input_only_example,
    materialize_dynamic_example,
)


def load_plan(path: Path, dataset_root: Path, manifest_path: Path) -> dict[str, Any]:
    plan = json.loads(path.read_text(encoding="utf-8"))
    if plan.get("status") != "frozen_before_runner_execution":
        raise RuntimeError("supplemental input plan is not frozen")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    by_id = {str(row["sample_id"]): row for row in manifest["samples"]}
    anchors, physics = [], {}
    for row in plan["cases"]:
        base_id = str(row["base_sample_id"])
        base = load_input_only_example(dataset_root, by_id[base_id])
        with np.load(row["dynamic_arrays"], allow_pickle=False) as arrays:
            anchor_k = np.asarray(arrays["anchor_k"], dtype=np.float64)
            anchor_q = np.asarray(arrays["anchor_q"], dtype=np.float64)
        anchor = materialize_dynamic_example(
            base, anchor_k=anchor_k, anchor_q=anchor_q,
            total_power_W=float(row["total_power_W"]),
        )
        anchors.append(anchor)
        physics[base_id] = {
            "physics_cache_file": str(row["full_physics"]),
            "physics_cache_sha256": str(row["full_physics_sha256"]),
        }
    if len(anchors) != 4 or len({row.sample_id for row in anchors}) != 4:
        raise RuntimeError("supplemental runner requires four distinct train geometries")
    return {"plan": plan, "anchors": anchors, "physics": physics}
