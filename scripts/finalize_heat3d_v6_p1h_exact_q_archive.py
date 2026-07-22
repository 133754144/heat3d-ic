#!/usr/bin/env python3
"""Add exact per-sample full q rows to an already solved P1h archive."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import h5py
import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[1]
for value in (ROOT, ROOT / "scripts"):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))

import generate_heat3d_v6_p1e_deconfounded_dataset as generator  # noqa: E402
import heat3d_v6_p1d_core as core  # noqa: E402


CONFIG = ROOT / "configs/heat3d_v6/v6_p1g_geometry_deconfounded1024.yaml"
TRACKED_MANIFEST = ROOT / "configs/heat3d_v6/v6_p1h_shared_support1024_manifest.json"
AUDIT = ROOT / "configs/heat3d_v6/v6_p1h_shared_support1024_audit.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    args = parser.parse_args()
    dataset = args.dataset.resolve()
    archive_path = dataset / "full_fields.h5"
    dataset_manifest_path = dataset / "manifest.json"
    manifest = json.loads(dataset_manifest_path.read_text(encoding="utf-8"))
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    groups = {str(row["group_id"]): row for row in config["geometry_groups"]}
    first = config["cases"][0]
    physics = generator._physics(
        float(first["top_h_W_m2K"]), float(first["bottom_h_W_m2K"]),
        config["physics"]["solver_mesh_intervals_xyz"],
    )
    mesh = core.build_mesh(physics)
    with h5py.File(archive_path, "r+") as archive:
        samples = archive["samples"]
        if "q_W_m3" in samples:
            q_dataset = samples["q_W_m3"]
        else:
            q_dataset = samples.create_dataset(
                "q_W_m3", shape=(len(config["cases"]), mesh["coords"].shape[0]),
                dtype=np.float64, chunks=(1, mesh["coords"].shape[0]),
                compression="gzip", compression_opts=4, shuffle=True,
            )
        for index, case in enumerate(config["cases"]):
            q, _, _ = generator._build_sources(
                str(case["id"]), float(case["package_total_power_W"]),
                groups[str(case["group_id"])], physics, mesh,
            )
            expected = manifest["samples"][index]["full_q_raw_sha256"]
            if generator.p1a._array_sha256(q) != expected:
                raise RuntimeError(f"{case['id']}: exact q hash drift")
            q_dataset[index, :] = q
        archive.attrs["exact_q_storage"] = "/samples/q_W_m3"
        archive.attrs["q_unit_power_role"] = "convenience_only_exact_q_uses_per_sample_rows"

    archive_sha = _sha256(archive_path)
    archive_record = {
        "path": "full_fields.h5",
        "sha256": archive_sha,
        "size_bytes": archive_path.stat().st_size,
        "exact_q_dataset": "/samples/q_W_m3",
    }
    for index, row in enumerate(manifest["samples"]):
        sample_meta_path = dataset / str(row["sample_dir"]) / "sample_meta.json"
        meta = json.loads(sample_meta_path.read_text(encoding="utf-8"))
        meta["full_field_archive"].update({
            "q_dataset": "/samples/q_W_m3",
            "q_row": index,
            "q_reconstruction": "exact_per_sample_full_q_archive_row",
        })
        _json(sample_meta_path, meta)
        row["file_sha256"]["sample_meta.json"] = _sha256(sample_meta_path)
    manifest["full_field_archive"] = archive_record
    _json(dataset_manifest_path, manifest)
    _json(TRACKED_MANIFEST, manifest)
    audit = json.loads(AUDIT.read_text(encoding="utf-8"))
    audit["archive"] = archive_record
    audit["dataset_manifest_sha256"] = _sha256(dataset_manifest_path)
    audit["exact_full_q_archive"] = True
    _json(AUDIT, audit)
    print(json.dumps({
        "status": "passed",
        "archive_sha256": archive_sha,
        "archive_size_bytes": archive_path.stat().st_size,
        "exact_q_rows": len(config["cases"]),
        "dataset_manifest_sha256": audit["dataset_manifest_sha256"],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
