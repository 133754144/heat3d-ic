#!/usr/bin/env python3
"""Deterministic integrity check for the frozen P1i full-field sidecar."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--sidecar-manifest", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    args = parser.parse_args()
    sidecar = json.loads(args.sidecar_manifest.read_text())
    source = json.loads(args.source_manifest.read_text())
    checks = {
        "source_dataset_id": sidecar["dataset_id"] == source["dataset_id"] == "heat3d_v6_p1i_continuous_physics1024_v1",
        "sample_count_1024": sidecar["sample_count"] == len(sidecar["samples"]) == len(source["samples"]) == 1024,
        "archive_sha256": sha256(args.archive) == sidecar["archive_sha256"],
        "source_1024_unchanged": sidecar["guardrails"]["source_1024_point_files_modified"] is False,
        "no_training_or_inference": sidecar["guardrails"]["training_runs"] == sidecar["guardrails"]["model_inference_runs"] == 0,
        "replay_machine_precision": sidecar["replay_max_abs_error"]["projected_temperature_K"] < 1e-9,
    }
    with h5py.File(args.archive, "r") as handle:
        ids = [value.decode() if isinstance(value, bytes) else str(value) for value in handle["samples/sample_id"][:]]
        source_ids = [str(row["sample_id"]) for row in source["samples"]]
        checks.update({
            "shared_mesh_240825": handle["shared/coords_m"].shape == (240825, 3),
            "shared_cv_240825": handle["shared/control_volume_m3"].shape == (240825,),
            "shared_layer_240825": handle["shared/layer_id"].shape == (240825,),
            "temperature_1024x240825": handle["samples/temperature_K"].shape == (1024, 240825),
            "deltaT_1024x240825": handle["samples/deltaT_K"].shape == (1024, 240825),
            "sample_ids_exact": ids == source_ids,
            "finite_shared_fields": all(np.all(np.isfinite(handle[path][:])) for path in ("shared/coords_m", "shared/control_volume_m3")),
        })
    payload = {"status": "passed" if all(checks.values()) else "failed", "checks": checks,
        "archive_sha256": sidecar["archive_sha256"], "sample_count": sidecar["sample_count"],
        "solver_node_count": 240825, "training_started": False}
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
